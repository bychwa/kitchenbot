"""Tokenizer training and the chat template.

Order matters here, and it is the step easiest to get wrong:

1. train BPE on the corpus  ->  tokenizer/vocab.json + merges.txt
2. wrap it as a fast tokenizer with the special tokens
3. pretrain the LM with that exact wrapper
4. attach the chat template to the *base* tokenizer before SFT
"""

from __future__ import annotations

from pathlib import Path

from .config import PATHS, SPECIAL_TOKENS, TokenizerConfig

#: The template the published adapter was trained with. ``<|eos|>`` after every
#: assistant turn is what teaches the model to stop talking.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}{{ eos_token }}\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def train_bpe(
    corpus: Path = PATHS.corpus,
    out_dir: Path = PATHS.tokenizer,
    cfg: TokenizerConfig = TokenizerConfig(),
) -> Path:
    """Train byte-level BPE on the recipe corpus and save vocab + merges."""
    from tokenizers import ByteLevelBPETokenizer

    if not corpus.is_file():
        raise FileNotFoundError(
            f"Missing {corpus}. Run `python -m kitchenbot download` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus)],
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=list(cfg.special_tokens),
    )
    tokenizer.save_model(str(out_dir))
    print(f"Saved {out_dir}/vocab.json + merges.txt (vocab={cfg.vocab_size})")
    return out_dir


def load_tokenizer(source: str | Path = PATHS.tokenizer):
    """Wrap the trained files in a ``PreTrainedTokenizerFast`` with special tokens.

    ``source`` is either the tokenizer directory or an already-trained model
    directory (both contain ``vocab.json``/``merges.txt`` or ``tokenizer.json``).
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    source = Path(source)
    vocab, merges = source / "vocab.json", source / "merges.txt"
    if not (vocab.is_file() and merges.is_file()):
        raise FileNotFoundError(
            f"Missing {vocab} / {merges}. Run `python -m kitchenbot tokenize` first."
        )

    raw = ByteLevelBPETokenizer(str(vocab), str(merges))
    return PreTrainedTokenizerFast(
        tokenizer_object=raw._tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
    )


def set_chat_template(model_dir: Path = PATHS.base) -> str:
    """Persist ``CHAT_TEMPLATE`` into a saved model directory; returns a demo prompt."""
    from transformers import AutoTokenizer

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {model_dir}. Run `python -m kitchenbot pretrain` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(model_dir)

    demo = tokenizer.apply_chat_template(
        [{"role": "user", "content": "How do I cook rice?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"chat_template saved into {model_dir}")
    print("--- sample prompt ---")
    print(demo)
    return demo


__all__ = ["CHAT_TEMPLATE", "SPECIAL_TOKENS", "load_tokenizer", "set_chat_template", "train_bpe"]
