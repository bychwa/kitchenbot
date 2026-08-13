# KitchenBot

> A **6.85M-parameter cooking model** trained from scratch in a weekend — tokenizer → pretrain → LoRA chat tune → terminal demo.
> This repo is the working pipeline behind it: a small, installable Python package you can run on your own machine.

[![Base model](https://img.shields.io/badge/🤗%20model-kitchenbot--base-yellow)](https://huggingface.co/bychwa/kitchenbot-base)
[![Chat adapter](https://img.shields.io/badge/🤗%20adapter-kitchenbot--chat-yellow)](https://huggingface.co/bychwa/kitchenbot-chat)
[![W&B](https://img.shields.io/badge/📈%20logs-kitchenbot-orange)](https://wandb.ai/bychwa-bouer-tech/kitchenbot)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

**Status:** finished learning project, kept reproducible. Not a product, not a chef.

---

## The model is on Hugging Face

The finished artifacts are public and tiny — you can use them before reading a line of code.

| Artifact | What it is | Size |
|---|---|---|
| [`bychwa/kitchenbot-base`](https://huggingface.co/bychwa/kitchenbot-base) | GPT-2 style causal LM, pretrained from scratch on ~10k recipes | ~26 MB |
| [`bychwa/kitchenbot-chat`](https://huggingface.co/bychwa/kitchenbot-chat) | LoRA adapter that turns the base into a Q&A model | ~1 MB |
| [wandb · kitchenbot](https://wandb.ai/bychwa-bouer-tech/kitchenbot) | Both training runs, loss curves included | — |

The base model **continues** recipe text. The adapter makes it **answer questions**. You need both for chat: load the adapter on top of the base.

### Use it in 20 lines

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "bychwa/kitchenbot-base"
adapter_id = "bychwa/kitchenbot-chat"

tok = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model = PeftModel.from_pretrained(model, adapter_id).eval()

messages = [{"role": "user", "content": "What are the ingredients for zucchini fritters?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                     temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Needs `transformers>=4.56`, `peft`, `torch`. On CPU the whole thing loads in a few seconds. Expect short, recipe-shaped answers — [see the honest sample](#what-it-actually-sounds-like).

### Or skip the code and talk to it

```bash
uv run kitchenbot chat -q "How do I make garlic butter pasta?"
```

With no local weights, the CLI pulls straight from the Hub. That is the fastest way to see what you are reproducing.

---

## Why this repo exists

Most "train an LLM" repos are either a notebook that stops at loss or a framework with a thousand flags. I wanted the middle: **one small model, every step of the pipeline owned, each step in a file you can read in five minutes.**

The weekend was the point. So is the stuff that went wrong — the ordering bug, the adapter that came out less fluent than the base, the missing eval. Those are in [What I got wrong](#what-i-got-wrong), not hidden.

What I was actually learning:

| Stage | The thing I had to understand | Where it lives now |
|---|---|---|
| Corpus | Stream a Hub dataset instead of downloading it; keep one text format everywhere | `kitchenbot/data.py` |
| Tokenizer | BPE on a niche corpus, and why `<|user|>` must exist **before** pretraining | `kitchenbot/tokenization.py` |
| Pretrain | GPT-2 config, fp16, effective batch = batch × grad-accum, checkpoint/resume when a pod dies | `kitchenbot/training.py` |
| Data craft | Templates and filters change the model's voice more than another layer would | `kitchenbot/data.py` |
| Chat template | Jinja, and the fact it has to be set on the **base** before SFT | `kitchenbot/tokenization.py` |
| SFT | LoRA rank/targets, why the adapter is 1 MB and not 26 MB | `kitchenbot/training.py` |
| Shipping | Model cards that match the repo, pinned specs, versioned in `hub/` | `kitchenbot/hub.py` |

## What I got wrong

Kept because they are the useful part.

1. **Chat template ordering.** I attached the template after SFT at first. TRL trained on raw `<|user|>` text and the adapter learned the format but not the turn structure. The template now lives in step 5, before step 6, and `pretrain` writes it into the checkpoint.
2. **fp16 everywhere.** Every training script hardcoded `fp16=True` and `torch.float16`, so the same command exploded on a Mac. Precision is now chosen from the device ([`training._precision`](kitchenbot/training.py)).
3. **Hub push baked in.** `push_to_hub=True` meant a fork could not even train without an HF account. Pushing and W&B logging are now **opt-in flags** — the pipeline runs fully offline.
4. **256-token `max_length`.** Instruction answers get truncated mid-sentence. This is a real ceiling on SFT quality, not a cosmetic setting.
5. **Only train loss.** I have no held-out scoreboard, so "better" is a feeling. If I did this again, eval would be the first module written, not the last.
6. **The adapter is less fluent than the base.** Two epochs on 10k templated pairs taught it the shape of an answer and cost it some of the base's flow. That trade-off is the most interesting thing I learned, and it is visible in the output below.

---

## Prerequisites

| Need | Detail |
|---|---|
| **Python** | 3.10+ (the published run used 3.12) |
| **`uv`** | [install](https://docs.astral.sh/uv/) — or use `pip` if you prefer |
| **Disk** | ~3 GB for dependencies, +~1 GB if you keep local weights |
| **GPU** | Optional but needed for pretrain/SFT. The published run used **1× RTX 3090 (24 GB)** on RunPod, CUDA 13. An 8 GB card is plenty at this size. |
| **CPU-only** | Fine for `download`, `tokenize`, `qa`, `chat`. Training works (fp32) but is slow. |
| **Accounts** | **None required.** Hugging Face login only to *publish* (`--push`), W&B only for `--wandb`. |
| **Network** | Needed to stream the dataset and to fetch Hub weights. |

No API keys, no paid services, no config files.

## Quickstart

```bash
git clone https://github.com/bychwa/kitchenbot.git
cd kitchenbot
uv sync                       # installs the package + deps
uv run --extra dev pytest -q  # 7 offline tests, no GPU or network

# talk to the published model (downloads ~27 MB from the Hub)
uv run kitchenbot chat
```

### Train your own from scratch

One command runs the whole pipeline (skips stages whose outputs already exist):

```bash
uv run kitchenbot pipeline --no-chat
```

To publish under your own name and log the runs:

```bash
export HF_USER=your-huggingface-username
export WANDB_PROJECT=kitchenbot
uv sync --extra logging

uv run kitchenbot pipeline --push --wandb
```

On a GPU pod, put the job in `tmux` first — an SSH drop should not kill a run. If it dies anyway, `--resume` picks up the last checkpoint.

---

## The pipeline is a package, not a pile of scripts

Every stage is a module with one job. The CLI is a thin front end over them, so you can call the same code from Python, from a notebook, or from a shell.

```
                 kitchenbot/
                        config.py         paths + hyperparameters (single source of truth)
                        data.py           Hub streaming, recipe text, synthetic Q&A
                        tokenization.py   BPE training, fast tokenizer, chat template
                        training.py       pretrain + LoRA SFT
                        modeling.py       model build/load/generate
                        hub.py            model-card upload
                        cli.py            argparse verbs
                             │
  ┌──────────────────────────┴───────────────────────────┐
  │                    python -m kitchenbot              │
  └──────────────────────────┬───────────────────────────┘
                             │
   download → tokenize → pretrain → qa → template → sft → chat
      │          │          │        │       │        │      │
 corpus.txt  tokenizer/  base/   qa.jsonl  jinja   adapter/  you
```

| Module | Verb | Does |
|---|---|---|
| `data.py` | `download` | mise-recipes → `data/corpus.txt` |
| `tokenization.py` | `tokenize` | corpus → `tokenizer/` (ByteLevel BPE, vocab 8k) |
| `training.py` | `pretrain` | tokenizer → `models/kitchenbot-base` (GPU) |
| `data.py` | `qa` | recipes → `data/cooking_qa.jsonl` |
| `tokenization.py` | `template` | attach the Jinja chat template to the base tokenizer |
| `training.py` | `sft` | Q&A → `models/kitchenbot-chat` LoRA adapter (GPU) |
| `modeling.py` | `chat` | load local weights, else the Hub, and talk |
| `hub.py` | `upload-cards` | push `hub/*/README.md` to Hugging Face |

### Command reference

```bash
uv run kitchenbot download [--limit N]          # step 1
uv run kitchenbot tokenize                      # step 2
uv run kitchenbot pretrain [--resume] [--push] [--wandb]   # step 3
uv run kitchenbot qa [--limit N]                # step 4
uv run kitchenbot template                      # step 5
uv run kitchenbot sft [--resume] [--push] [--wandb]        # step 6
uv run kitchenbot chat [-q QUESTION]            # step 7
uv run kitchenbot pipeline [--no-chat] [--force] [--push]  # all of it
uv run kitchenbot upload-cards [--dry-run]      # publish the model cards
```

The old `scripts/01_*.py … 08_*.py` files still exist as three-line aliases, because breaking links is worse than a little duplication. New code should use the package.

`uv run kitchenbot <verb> --help` for the flags of any stage.

---

## What the weekend run actually did

| Piece | Value |
|---|---|
| Architecture | `GPT2LMHeadModel` · 6 layers · 256 emb · 8 heads · ctx 256 · vocab 8000 |
| Params | **6.85M** (~26 MB base + ~1 MB adapter) |
| Pretrain data | [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes), 10k recipes streamed |
| Chat data | 10k synthetic Q&A pairs from the same recipes, 4 templates |
| Hardware | RunPod · 1× RTX 3090 (24 GB) · CUDA 13 · Python 3.12 |
| Cost | **under $1** of GPU time; $2–8 with pod setup and idle time |

**Pretrain:** lr `3e-4`, warmup `100`, batch `16 × accum 4` (effective 64), 1 epoch → **157 steps**, train loss ≈ **5.88**.
**SFT:** LoRA `r=16`, `alpha=32`, `c_proj` + `c_attn`, lr `2e-4`, batch `8 × accum 4`, 2 epochs → **626 steps**, train loss ≈ **4.32**, token accuracy ≈ **0.44**.

Both are **train** curves. There is no held-out eval suite in this release, which is limitation number one.

## What it actually sounds like

Raw output, unedited, from the published weights at `temperature=0.7`:

```text
# base model, prompted with the start of a recipe
Chocolate Cake
Ingredients: garlic, coconut oil, basil leaves, coconut oil, cilantro, coconut oil, garlic cloves, garlic, salt
Steps: Preheat oven to 400°C). Place the onion and a large large bowl, salt and let with until smooth...
```

```bash
$ uv run kitchenbot chat -q "What are the ingredients for zucchini fritters?"
KitchenBot: , garlic, and salt, coconut oil
```

That second line is the honest state of the chat adapter. The shape is right, the content is mostly noise. A 7M model that saw 10k templated pairs for two epochs does not become a chef — and the base checkpoint, which only had to continue text, is the more convincing of the two.

## Repo layout

```
kitchenbot/               # the pipeline (installable package)
  config.py data.py tokenization.py training.py modeling.py hub.py cli.py
scripts/                  # thin aliases for the old numbered scripts
tests/                    # 7 offline tests: formats, config, CLI, shipped tokenizer
tokenizer/                # trained BPE (vocab.json + merges.txt)
hub/                      # model-card sources for the two Hub repos
data/samples/             # tiny heads for inspection
pyproject.toml            # deps, console script, packaging
```

Not committed: `data/corpus.txt`, `data/cooking_qa.jsonl`, `models/`. All three are regenerated by the pipeline or downloaded from the Hub, which keeps the repo small and the reproduction honest.

## Limitations (honest)

- **Toy model.** 6.85M params. It will hallucinate, mix recipes, and fail outside cooking.
- **Template-shaped answers.** The Q&A data is four templates deep, so replies look like ingredient lists, not conversation.
- **Adapter fluency.** SFT cost some of the base model's flow (see above).
- **256-token context, 256-token SFT length.** Long recipes and answers get truncated.
- **No held-out eval.** Metrics above are train curves.
- **English, recipes only.** Not an assistant, not nutrition advice.

## If I kept going

1. **Build the eval first** — a held-out quiz with a rubric, before touching another knob.
2. **De-template the Q&A data** — paraphrases would widen the model's voice more than more parameters.
3. **Raise context and SFT length to 512** so answers stop getting cut.
4. **Merge the LoRA** into the base for a single-artifact release.
5. **Scale to ~50M with the same modules** — the pipeline is the deliverable; parameter count is a dial.

## License

Apache-2.0. Recipe text comes from [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes) — respect that dataset's terms when redistributing derived data.
