"""Turn recipes into simple cooking Q&A pairs for chat SFT."""

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path("data/cooking_qa.jsonl")
MAX_EXAMPLES = 10_000
SEED = 42

TEMPLATES = [
    ("What are the ingredients for {title}?", "{ingredients}"),
    ("How do I make {title}?", "{steps}"),
    ("Give me a short tip for making {title}.", "Key steps: {steps_short}"),
    ("What is the first step for {title}?", "{first_step}"),
]


def main() -> None:
    random.seed(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("idoyaaran/mise-recipes", split="train", streaming=True)
    count = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in ds:
            title = (row.get("title") or "this dish").strip()[:80]
            ings = ", ".join(
                (i.get("name") or "") for i in (row.get("ingredients") or [])
            )
            steps_list = row.get("steps") or []
            steps = " ".join(steps_list).strip()
            if len(steps) < 20 or not ings:
                continue

            first = steps_list[0] if steps_list else steps[:80]
            steps_short = steps[:200]
            t_q, t_a = random.choice(TEMPLATES)
            q = t_q.format(title=title)
            a = t_a.format(
                title=title,
                ingredients=ings,
                steps=steps,
                steps_short=steps_short,
                first_step=first,
            )
            record = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= MAX_EXAMPLES:
                break

    print(f"Wrote {count} examples → {OUT_PATH}")


if __name__ == "__main__":
    main()
