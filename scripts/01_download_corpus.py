"""Stream mise-recipes from the Hub into a plain-text corpus."""

from pathlib import Path

from datasets import load_dataset

DATASET = "idoyaaran/mise-recipes"
OUT_PATH = Path("data/corpus.txt")


def as_text(row: dict) -> str:
    title = row.get("title") or ""
    ings = ", ".join((i.get("name") or "") for i in (row.get("ingredients") or []))
    steps = " ".join(row.get("steps") or [])
    return f"{title}\nIngredients: {ings}\nSteps: {steps}\n"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, split="train", streaming=True)

    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for row in ds:
            out.write(as_text(row) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  … {n} recipes")

    print(f"Wrote {n} recipes → {OUT_PATH}")


if __name__ == "__main__":
    main()
