"""Command line front end — one verb per pipeline stage.

    python -m kitchenbot download       # Hub dataset -> data/corpus.txt
    python -m kitchenbot tokenize       # corpus -> tokenizer/
    python -m kitchenbot pretrain       # tokenizer -> models/kitchenbot-base
    python -m kitchenbot qa             # recipes -> data/cooking_qa.jsonl
    python -m kitchenbot template       # attach chat template to the base
    python -m kitchenbot sft            # Q&A -> models/kitchenbot-chat (LoRA)
    python -m kitchenbot chat           # talk to local or Hub weights
    python -m kitchenbot pipeline       # every stage above, in order
    python -m kitchenbot upload-cards   # hub/*/README.md -> Hugging Face
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import HUB, PATHS


def _download(args) -> None:
    from .data import download_corpus

    download_corpus(PATHS.corpus, limit=args.limit)


def _tokenize(args) -> None:
    from .tokenization import train_bpe

    train_bpe(PATHS.corpus, PATHS.tokenizer)


def _pretrain(args) -> None:
    from .training import run_pretrain

    run_pretrain(
        out_dir=PATHS.base,
        resume=args.resume,
        push=args.push,
        wandb=args.wandb,
    )


def _qa(args) -> None:
    from .data import build_qa_dataset

    build_qa_dataset(PATHS.qa, limit=args.limit)


def _template(args) -> None:
    from .tokenization import set_chat_template

    set_chat_template(PATHS.base)


def _sft(args) -> None:
    from .training import run_sft

    run_sft(
        base_dir=PATHS.base,
        out_dir=PATHS.adapter,
        qa_path=PATHS.qa,
        resume=args.resume,
        push=args.push,
        wandb=args.wandb,
    )


def _chat(args) -> None:
    from .modeling import resolve_chat_paths

    base_id, adapter_id = resolve_chat_paths()
    _run_chat(base_id, adapter_id, getattr(args, "question", None),
              getattr(args, "max_new_tokens", 150))


def _run_chat(base_id: str, adapter_id: str, question: str | None, max_new_tokens: int) -> None:
    from .modeling import load_for_chat, stream_reply

    model, tokenizer = load_for_chat(base_id, adapter_id)

    if question:
        print("KitchenBot: ", end="", flush=True)
        stream_reply(model, tokenizer, question, max_new_tokens=max_new_tokens)
        print()
        return

    print("KitchenBot ready. Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt:
            break
        print("KitchenBot: ", end="", flush=True)
        stream_reply(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        print()


def _pipeline(args) -> None:
    """Run stages 1–7 in order, skipping the ones whose outputs already exist."""
    from .data import build_qa_dataset, download_corpus
    from .tokenization import set_chat_template, train_bpe

    if not PATHS.corpus.is_file() or args.force:
        download_corpus(PATHS.corpus, limit=args.limit)
    if not (PATHS.tokenizer / "vocab.json").is_file() or args.force:
        train_bpe(PATHS.corpus, PATHS.tokenizer)

    _pretrain(args)
    if not PATHS.qa.is_file() or args.force:
        build_qa_dataset(PATHS.qa, limit=args.limit)
    set_chat_template(PATHS.base)

    from .training import run_sft

    run_sft(
        base_dir=PATHS.base,
        out_dir=PATHS.adapter,
        qa_path=PATHS.qa,
        resume=args.resume,
        push=args.push,
        wandb=args.wandb,
    )
    if not args.no_chat:
        _chat(args)


def _upload_cards(args) -> None:
    from .hub import upload_model_cards

    upload_model_cards(dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kitchenbot",
        description="Train a tiny from-scratch cooking SLM: tokenizer → pretrain → LoRA chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"kitchenbot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, func) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    p = add("download", "Stream mise-recipes into data/corpus.txt.", _download)
    p.add_argument("--limit", type=int, default=None, help="stop after N recipes")

    add("tokenize", "Train byte-level BPE on data/corpus.txt.", _tokenize)

    for name, func, help_text in (
        ("pretrain", _pretrain, "Pretrain the base GPT-2 from scratch (GPU)."),
        ("sft", _sft, "LoRA fine-tune the chat adapter (GPU)."),
    ):
        p = add(name, help_text, func)
        p.add_argument("--resume", action="store_true", help="resume from last checkpoint")
        p.add_argument("--push", action="store_true", help=f"push result to {HUB.user}/…")
        p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")

    p = add("qa", "Build synthetic cooking Q&A for SFT.", _qa)
    p.add_argument("--limit", type=int, default=None, help="stop after N pairs")

    add("template", "Attach the Jinja chat template to the base tokenizer.", _template)

    p = add("chat", "Talk to KitchenBot (local weights, else the Hub).", _chat)
    p.add_argument("-q", "--question", default=None, help="ask once and exit")
    p.add_argument("--max-new-tokens", type=int, default=150)

    p = add("pipeline", "Run every stage in order.", _pipeline)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--force", action="store_true", help="regenerate corpus, tokenizer and Q&A")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-chat", action="store_true", help="stop after SFT")

    p = add("upload-cards", "Push hub/*/README.md to Hugging Face.", _upload_cards)
    p.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        print(
            f"error: missing dependency '{exc.name}'. "
            "Run `uv sync` (or `pip install -e .`) and retry.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
