---
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: bychwa/kitchenbot-base
tags:
  - peft
  - lora
  - trl
  - sft
  - cooking
  - recipes
  - kitchenbot
language:
  - en
datasets:
  - idoyaaran/mise-recipes
---

# kitchenbot-chat

LoRA **chat adapter** for [`bychwa/kitchenbot-base`](https://huggingface.co/bychwa/kitchenbot-base) — the second half of a weekend experiment to learn pretrain → SFT on a niche cooking model.

| | |
|---|---|
| **Base model** | [`bychwa/kitchenbot-base`](https://huggingface.co/bychwa/kitchenbot-base) (~6.85M GPT-2, trained from scratch) |
| **Training code** | [`github.com/bychwa/kitchenbot`](https://github.com/bychwa/kitchenbot) |
| **SFT run** | [wandb · syrnpr69](https://wandb.ai/bychwa-bouer-tech/kitchenbot/runs/syrnpr69) |

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/bychwa-bouer-tech/kitchenbot/runs/syrnpr69)

## Motivation

After pretraining a tiny recipe LM, I wanted it to answer short cooking questions in a chat format — without full fine-tuning. LoRA on a single **RTX 3090** was the right tool: small adapter (~1 MB), fast iteration, same pod as pretrain.

## What this repo contains

This Hub repo is a **PEFT/LoRA adapter**, not a full model. Always load it **on top of** `bychwa/kitchenbot-base`.

| LoRA setting | Value |
|--------------|--------|
| Rank `r` | 16 |
| `lora_alpha` | 32 |
| Dropout | 0.05 |
| Target modules | `c_attn`, `c_proj` |
| Task | Causal LM (SFT via TRL) |

## Training data

10,000 synthetic Q&A pairs (`data/cooking_qa.jsonl`) built from [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes) with simple templates, e.g.:

- “What are the ingredients for {title}?”
- “How do I make {title}?”
- “What is the first step for {title}?”

Messages use a small Jinja chat template with `<|user|>` / `<|assistant|>` tokens (set on the base tokenizer before SFT).

## Hardware (RunPod)

Same pod as the base run:

| Spec | Value |
|------|--------|
| GPU | 1× NVIDIA GeForce RTX 3090 (24 GB) |
| CUDA | 13.0 |
| Python | 3.12 |
| Stack | PyTorch 2.5.1+cu121, Transformers 5.14, TRL, PEFT, W&B |

## Training procedure

Supervised fine-tuning with `trl.SFTTrainer` + LoRA.

| Hyperparameter | Value |
|----------------|--------|
| Learning rate | 2e-4 |
| Batch size | 8 |
| Grad accumulation | 4 |
| Epochs | 2 |
| Max length | 256 |
| Precision | fp16 |

### Results (train)

| Metric | Value |
|--------|--------|
| Steps | 626 |
| Train runtime | ~152 s |
| `train_loss` | ≈ 4.32 |
| Last logged step loss | ≈ 4.10 |
| Mean token accuracy | ≈ 0.44 |

Train curves only — no formal held-out quiz scoreboard shipped with this release.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "bychwa/kitchenbot-base"
adapter_id = "bychwa/kitchenbot-chat"

tok = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

messages = [{"role": "user", "content": "How do I make garlic butter pasta?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

out = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tok.eos_token_id,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

Or use the CLI from the training repo:

```bash
export HF_USER=bychwa
uv run kitchenbot chat
```

## Intended use & limitations

**Use:** casual home-kitchen Q&A demos, learning how LoRA SFT sits on a custom base, portfolio / teaching.

**Limits:**

- Still a **~7M** model — expect wrong steps, mixed recipes, and confident nonsense  
- Answers mirror the synthetic templates; not a chef or nutritionist  
- 256-token context  
- Not suitable for safety-critical or dietary medical advice  

## Reproduce

```bash
# after the base model is in models/kitchenbot-base
uv run kitchenbot qa
uv run kitchenbot template
uv run kitchenbot sft --push
```

Full walkthrough: **https://github.com/bychwa/kitchenbot**

## License

Apache-2.0 for this adapter. Base model and dataset terms also apply (`bychwa/kitchenbot-base`, `idoyaaran/mise-recipes`).

## Citations

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```
