"""LoRA SFT: teach the base model to answer cooking questions."""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = Path("models/kitchenbot-base")
OUT = Path("models/kitchenbot-chat")
QA_PATH = Path("data/cooking_qa.jsonl")

HF_USER = os.environ.get("HF_USER", "bychwa")
HUB_ID = f"{HF_USER}/kitchenbot-chat"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune kitchenbot-chat (LoRA)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in models/kitchenbot-chat",
    )
    args = parser.parse_args()

    if not BASE.is_dir():
        raise SystemExit(f"Missing {BASE}. Run pretrain + chat template first.")
    if not QA_PATH.is_file():
        raise SystemExit(f"Missing {QA_PATH}. Run scripts/04_build_qa_dataset.py first.")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base from {BASE}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="auto"
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
    )

    ds = load_dataset("json", data_files=str(QA_PATH), split="train")
    print(f"SFT on {len(ds)} chat examples → Hub {HUB_ID}")

    def formatting(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    train_args = SFTConfig(
        output_dir=str(OUT),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=25,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        max_length=256,
        report_to="wandb",
        run_name="kitchenbot-chat-sft",
        hub_model_id=HUB_ID,
        push_to_hub=True,
        hub_strategy="every_save",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
        formatting_func=formatting,
        peft_config=lora,
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.model.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    trainer.push_to_hub(commit_message="final chat adapter")
    print(f"Saved locally: {OUT}")
    print(f"On Hub: https://huggingface.co/{HUB_ID}")


if __name__ == "__main__":
    main()
