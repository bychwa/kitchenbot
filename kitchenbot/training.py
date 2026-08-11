"""Training stages: pretrain the base, then LoRA SFT the chat adapter.

Pushing to the Hub and logging to W&B are **opt-in** (``push=True`` /
``wandb=True``), so a fresh clone trains locally with no accounts configured.
That is the difference between "my weekend run" and "anyone can repeat this".
"""

from __future__ import annotations

from pathlib import Path

from .config import HUB, PATHS, ChatConfig, PretrainConfig
from .data import format_recipe, stream_recipes
from .modeling import build_base_model
from .tokenization import load_tokenizer



def _report_to(use_wandb: bool) -> list[str]:
    return ["wandb"] if use_wandb else ["none"]


def _precision() -> tuple[bool, str]:
    """fp16 on GPU, fp32 on CPU so the same command runs anywhere (slowly)."""
    import torch

    use_cuda = torch.cuda.is_available()
    return use_cuda, "float16" if use_cuda else "float32"


def build_pretrain_dataset(tokenizer, cfg: PretrainConfig = PretrainConfig()):
    """Stream recipes, format them, and tokenize into fixed-length blocks."""
    from datasets import Dataset

    print(f"Streaming {cfg.max_samples} recipes…")
    rows = []
    for row in stream_recipes():
        rows.append({"text": f"<|bos|>{format_recipe(row).strip()}<|eos|>"})
        if len(rows) >= cfg.max_samples:
            break

    dataset = Dataset.from_list(rows)
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=cfg.max_len),
        batched=True,
        remove_columns=["text"],
    )


def run_pretrain(
    out_dir: Path = PATHS.base,
    cfg: PretrainConfig = PretrainConfig(),
    resume: bool = False,
    push: bool = False,
    wandb: bool = False,
) -> Path:
    """Causal-LM pretraining on recipe text. ~157 steps on one RTX 3090."""
    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2LMHeadModel,
        Trainer,
        TrainingArguments,
    )

    tokenizer = load_tokenizer(PATHS.tokenizer)
    use_cuda, _ = _precision()

    if resume and out_dir.is_dir():
        print(f"Resuming weights from {out_dir}")
        model = GPT2LMHeadModel.from_pretrained(out_dir)
    else:
        model, stats = build_base_model(tokenizer, cfg)
        print(f"Fresh model: {stats['params_m']}M params")
        if push:
            print(f"Will push to {HUB.base_id}")

    dataset = build_pretrain_dataset(tokenizer, cfg)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=use_cuda,
        report_to=_report_to(wandb),
        run_name="kitchenbot-pretrain",
        hub_model_id=HUB.base_id if push else None,
        push_to_hub=push,
        hub_strategy="every_save" if push else "end",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume or None)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    if push:
        trainer.push_to_hub(commit_message="final base model")
        print(f"On Hub: https://huggingface.co/{HUB.base_id}")
    print(f"Saved locally: {out_dir}")

    # The Hub copy needs the chat template too; keep local + remote in sync.
    from .tokenization import set_chat_template

    set_chat_template(out_dir)
    return out_dir


def run_sft(
    base_dir: Path = PATHS.base,
    out_dir: Path = PATHS.adapter,
    qa_path: Path = PATHS.qa,
    cfg: ChatConfig = ChatConfig(),
    resume: bool = False,
    push: bool = False,
    wandb: bool = False,
) -> Path:
    """LoRA SFT on synthetic cooking Q&A. ~626 steps for 2 epochs."""
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {base_dir}. Run `python -m kitchenbot pretrain` first."
        )
    if not qa_path.is_file():
        raise FileNotFoundError(f"Missing {qa_path}. Run `python -m kitchenbot qa` first.")

    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base from {base_dir}")
    use_cuda, dtype = _precision()
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=dtype, device_map="auto" if use_cuda else None
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
    )

    dataset = load_dataset("json", data_files=str(qa_path), split="train")
    print(f"SFT on {len(dataset)} chat examples")

    def formatting(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    train_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=use_cuda,
        max_length=cfg.max_len,
        report_to=_report_to(wandb),
        run_name="kitchenbot-chat-sft",
        hub_model_id=HUB.chat_id if push else None,
        push_to_hub=push,
        hub_strategy="every_save" if push else "end",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=resume or None)

    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    if push:
        trainer.push_to_hub(commit_message="final chat adapter")
        print(f"On Hub: https://huggingface.co/{HUB.chat_id}")
    print(f"Saved locally: {out_dir}")
    return out_dir


__all__ = ["build_pretrain_dataset", "run_pretrain", "run_sft"]
