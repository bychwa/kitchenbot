"""Pretrain a tiny GPT-2 base model on recipe text (causal LM)."""

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

# --- knobs (matched the weekend RunPod RTX 3090 run) ---
OUT_DIR = Path("models/kitchenbot-base")
MAX_LEN = 256
MAX_SAMPLES = 10_000
N_EMBD = 256
N_LAYER = 6
N_HEAD = 8

HF_USER = os.environ.get("HF_USER", "bychwa")
HUB_ID = f"{HF_USER}/kitchenbot-base"


def load_tokenizer() -> PreTrainedTokenizerFast:
    raw = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    return PreTrainedTokenizerFast(
        tokenizer_object=raw._tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
    )


def format_recipe(row: dict) -> str:
    title = row.get("title") or ""
    ings = ", ".join((i.get("name") or "") for i in (row.get("ingredients") or []))
    steps = " ".join(row.get("steps") or [])
    return f"<|bos|>{title}\nIngredients: {ings}\nSteps: {steps}<|eos|>"


def build_dataset(tokenizer: PreTrainedTokenizerFast) -> Dataset:
    print(f"Streaming mise-recipes (capping at {MAX_SAMPLES} samples)…")
    ds = load_dataset("idoyaaran/mise-recipes", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        rows.append({"text": format_recipe(row)})
        if i + 1 >= MAX_SAMPLES:
            break

    dataset = Dataset.from_list(rows)
    return dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain kitchenbot-base")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in models/kitchenbot-base",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    if args.resume and OUT_DIR.is_dir():
        print(f"Resuming weights from {OUT_DIR}")
        model = GPT2LMHeadModel.from_pretrained(OUT_DIR)
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=MAX_LEN,
            n_embd=N_EMBD,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(config)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Fresh model: {n_params:.1f}M params → will push to {HUB_ID}")

    dataset = build_dataset(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=3e-4,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        report_to="wandb",
        run_name="kitchenbot-pretrain",
        hub_model_id=HUB_ID,
        push_to_hub=True,
        hub_strategy="every_save",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    trainer.push_to_hub(commit_message="final base model")
    print(f"Saved locally: {OUT_DIR}")
    print(f"On Hub: https://huggingface.co/{HUB_ID}")


if __name__ == "__main__":
    main()
