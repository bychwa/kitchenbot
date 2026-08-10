"""Corpus + dataset construction.

Two inputs, two outputs:

    mise-recipes (Hub, streaming)  ->  data/corpus.txt       (plain text, tokenizer food)
    mise-recipes (Hub, streaming)  ->  data/cooking_qa.jsonl (chat SFT pairs)

Streaming keeps peak RAM flat regardless of dataset size, which matters on a
small pod. Neither output is committed: both are cheap to regenerate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import DATASET_ID, DATASET_SPLIT, PATHS, ChatConfig

#: Question templates for synthetic SFT data. The list is deliberately short —
#: it is why the model answers in recipe-shaped lists (see README limitations).
QA_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("What are the ingredients for {title}?", "{ingredients}"),
    ("How do I make {title}?", "{steps}"),
    ("Give me a short tip for making {title}.", "Key steps: {steps_short}"),
    ("What is the first step for {title}?", "{first_step}"),
)


def format_recipe(row: dict) -> str:
    """Flatten one recipe row into the exact text shape the model was trained on.

    Used by both the corpus and pretraining so the tokenizer and the model never
    disagree about formatting.
    """
    title = row.get("title") or ""
    ingredients = ", ".join(
        (item.get("name") or "") for item in (row.get("ingredients") or [])
    )
    steps = " ".join(row.get("steps") or [])
    return f"{title}\nIngredients: {ingredients}\nSteps: {steps}\n"


def stream_recipes():
    """Yield recipe rows from the Hub without materialising the whole dataset."""
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)


def download_corpus(out_path: Path = PATHS.corpus, limit: int | None = None) -> int:
    """Stream the Hub dataset into ``data/corpus.txt``. Returns rows written."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for row in stream_recipes():
            out.write(format_recipe(row) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  … {written} recipes")
            if limit is not None and written >= limit:
                break

    print(f"Wrote {written} recipes → {out_path}")
    return written


def build_qa_dataset(
    out_path: Path = PATHS.qa,
    cfg: ChatConfig = ChatConfig(),
    limit: int | None = None,
) -> int:
    """Turn recipes into ``{"messages": [...]}`` rows for TRL SFT."""
    rng = random.Random(cfg.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = limit if limit is not None else cfg.max_examples
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for row in stream_recipes():
            title = (row.get("title") or "this dish").strip()[:80]
            ingredients = ", ".join(
                (item.get("name") or "") for item in (row.get("ingredients") or [])
            )
            steps_list = row.get("steps") or []
            steps = " ".join(steps_list).strip()

            # Skip rows that would teach the model to answer nothing.
            if len(steps) < 20 or not ingredients:
                continue

            question_template, answer_template = rng.choice(QA_TEMPLATES)
            answer = answer_template.format(
                title=title,
                ingredients=ingredients,
                steps=steps,
                steps_short=steps[:200],
                first_step=steps_list[0] if steps_list else steps[:80],
            )
            record = {
                "messages": [
                    {"role": "user", "content": question_template.format(title=title)},
                    {"role": "assistant", "content": answer},
                ]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written >= target:
                break

    print(f"Wrote {written} examples → {out_path}")
    return written
