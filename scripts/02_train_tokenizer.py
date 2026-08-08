"""Train a small ByteLevel BPE tokenizer on the recipe corpus."""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

CORPUS = Path("data/corpus.txt")
OUT_DIR = Path("tokenizer")
VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
]


def main() -> None:
    if not CORPUS.is_file():
        raise SystemExit(f"Missing {CORPUS}. Run scripts/01_download_corpus.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(CORPUS)],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(OUT_DIR))
    print(f"Saved {OUT_DIR}/vocab.json and merges.txt (vocab={VOCAB_SIZE})")


if __name__ == "__main__":
    main()
