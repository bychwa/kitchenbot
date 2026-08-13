---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt2
  - text-generation
  - cooking
  - recipes
  - from-scratch
  - kitchenbot
language:
  - en
datasets:
  - idoyaaran/mise-recipes
base_model: []
widget:
  - text: "<|bos|>Garlic Butter Pasta\nIngredients: pasta, garlic, butter\nSteps:"
    example_title: Recipe continuation
---

# kitchenbot-base

A **~6.85M** GPT-2-style language model trained **from scratch** on cooking recipes over a weekend — part of a hands-on experiment to learn the full pretrain → chat-SFT loop on a single rented GPU.

| | |
|---|---|
| **Chat fine-tune** | [`bychwa/kitchenbot-chat`](https://huggingface.co/bychwa/kitchenbot-chat) (LoRA adapter) |
| **Training code** | [`github.com/bychwa/kitchenbot`](https://github.com/bychwa/kitchenbot) |
| **Logs** | [wandb · kitchenbot](https://wandb.ai/bychwa-bouer-tech/kitchenbot) |

## Motivation

I wanted a real end-to-end run I could finish in a weekend: niche data, custom tokenizer, pretrain a small causal LM, then LoRA-tune it for Q&A. Keeping the model tiny was intentional — fit the whole loop on one **RTX 3090** pod, focus on process, and ship working artifacts.

## Model details

| | |
|---|---|
| Architecture | `GPT2LMHeadModel` |
| Parameters | ~6.85M |
| Layers / emb / heads | 6 / 256 / 8 |
| Context | 256 tokens |
| Vocab | 8000 (ByteLevel BPE, trained on the recipe corpus) |
| Special tokens | `<\|pad\|>`, `<\|unk\|>`, `<\|bos\|>`, `<\|eos\|>`, `<\|user\|>`, `<\|assistant\|>` |

Weights are ~26 MB (`safetensors`).

## Training data

Source: [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes) (streamed from the Hub).

Each example was formatted roughly as:

```text
<|bos|>{title}
Ingredients: {ingredient names}
Steps: {joined steps}<|eos|>
```

The weekend run used **~10k** recipes (`MAX_SAMPLES=10000` in the training script).

## Hardware (RunPod)

| Spec | Value |
|------|--------|
| GPU | 1× NVIDIA GeForce RTX 3090 (24 GB) |
| CUDA | 13.0 |
| Host | Linux (RunPod container) |
| Python | 3.12 |
| Stack | PyTorch 2.5.1+cu121, Transformers 5.14, Datasets, Tokenizers, Accelerate, W&B |

Approximate cost: a few dollars at ~$0.25–0.40/hr for a short pretrain + SFT session.

## Training procedure

Causal language modeling (next-token prediction) with Hugging Face `Trainer`.

| Hyperparameter | Value |
|----------------|--------|
| Learning rate | 3e-4 |
| Warmup steps | 100 |
| Batch size | 16 |
| Grad accumulation | 4 (effective batch **64**) |
| Epochs | 1 |
| Max length | 256 |
| Precision | fp16 |
| Optimizer | AdamW |
| LR schedule | linear |
| Seed | 42 |

### Results (train)

| Metric | Value |
|--------|--------|
| Steps | 157 |
| Train runtime | ~24 s (this config on 3090) |
| `train_loss` | ≈ 5.88 |
| Last logged step loss | ≈ 4.19 |

These are **training** metrics, not a held-out perplexity suite. Loss trended down; the model learns recipe-ish continuations, not general knowledge.

## Intended use

- Recipe-style **text continuation** in the cooking domain  
- Base weights for the LoRA chat adapter [`kitchenbot-chat`](https://huggingface.co/bychwa/kitchenbot-chat)  
- Teaching / portfolio example of a from-scratch SLM pipeline  

**Not** intended as a general assistant, medical/nutrition advice source, or production kitchen system.

## Limitations

- Tiny capacity → invents ingredients/steps and mixes dishes  
- 256-token context truncates long recipes  
- English cooking text bias from the source dataset  
- No safety / factuality filtering  

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "bychwa/kitchenbot-base"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = "<|bos|>Simple Tomato Sauce\nIngredients: tomatoes, garlic, olive oil\nSteps:"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=False))
```

For Q&A chat, load this base **plus** the LoRA adapter — see [`bychwa/kitchenbot-chat`](https://huggingface.co/bychwa/kitchenbot-chat).

## Reproduce

Full scripts (uv setup, corpus → tokenizer → pretrain → SFT → CLI):

**https://github.com/bychwa/kitchenbot**

```bash
export HF_USER=bychwa
export WANDB_PROJECT=kitchenbot
uv run kitchenbot pretrain --push --wandb
```

## License

Apache-2.0 for the model code/weights packaging in this card’s training setup. Respect the license/terms of [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes) for the underlying text.
