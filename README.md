# KitchenBot

A weekend experiment: train a **tiny cooking language model from scratch**, then LoRA fine-tune it for home-kitchen Q&A — end to end on a single rented GPU.

| Artifact | Link |
|----------|------|
| Base LM (from scratch) | [bychwa/kitchenbot-base](https://huggingface.co/bychwa/kitchenbot-base) |
| Chat adapter (LoRA) | [bychwa/kitchenbot-chat](https://huggingface.co/bychwa/kitchenbot-chat) |
| Training logs | [wandb · kitchenbot](https://wandb.ai/bychwa-bouer-tech/kitchenbot) |

This repo is the **process proof**. Weights live on Hugging Face; the scripts here are how they were made.

## Motivation

I wanted one clean loop I could finish in a weekend:

1. pick a niche (recipes / home cooking)
2. train a tokenizer
3. pretrain a small causal LM
4. build a chat dataset
5. LoRA SFT
6. actually talk to it in a terminal

Keeping the model small (~**6.85M** params) was deliberate — the point was learning the pipeline on a real GPU pod, not chasing benchmark scores.

## What you get

```
mise-recipes (HF)
       │
       ▼
  corpus.txt ──► ByteLevel BPE (vocab 8k)
       │
       ▼
  kitchenbot-base   GPT-2 style, 6×256, ctx 256
       │
       ▼
  cooking_qa.jsonl ──► LoRA SFT
       │
       ▼
  kitchenbot-chat   + scripts/07_chat.py
```

| Piece | Detail |
|-------|--------|
| Architecture | `GPT2LMHeadModel` · 6 layers · 256 emb · 8 heads · ctx 256 · vocab 8000 |
| Params | ~6.85M (~26 MB base + ~1 MB LoRA) |
| Pretrain data | [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes) → ~10k recipes |
| Chat data | 10k synthetic Q&A pairs derived from the same recipes |
| Hardware | RunPod · **1× NVIDIA GeForce RTX 3090** (24 GB) · CUDA 13.0 · Python 3.12 |

## Repo layout

```
scripts/
  01_download_corpus.py
  02_train_tokenizer.py
  03_pretrain_base.py
  04_build_qa_dataset.py
  05_set_chat_template.py
  06_finetune_chat.py
  07_chat.py
tokenizer/          # trained BPE (vocab.json + merges.txt)
data/samples/       # tiny heads for inspection (full data is regenerated)
```

Full `data/corpus.txt`, `data/cooking_qa.jsonl`, and `models/` are **not** committed — regenerate or download from the Hub.

## Setup (uv)

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/bychwa/kitchenbot.git
cd kitchenbot

uv venv
source .venv/bin/activate

# GPU pod (CUDA wheel) — used on RunPod RTX 3090
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install accelerate datasets huggingface_hub peft tokenizers transformers "trl>=0.14" wandb
```


Mac / CPU-only is fine for tokenizer + corpus + a slow chat demo. **Pretrain and SFT want the GPU.**

### Auth (for training + Hub push)

```bash
hf auth login
wandb login

export HF_USER=bychwa
export WANDB_PROJECT=kitchenbot
```

## Run the pipeline

Order matters. From the repo root, with the venv active:

```bash
# 1) recipes → plain text
python scripts/01_download_corpus.py

# 2) train BPE (or use the committed tokenizer/)
python scripts/02_train_tokenizer.py

# 3) pretrain base LM  (GPU · pushes to HF)
python scripts/03_pretrain_base.py
# resume after a pod blip:
# python scripts/03_pretrain_base.py --resume

# 4) synthetic cooking Q&A
python scripts/04_build_qa_dataset.py

# 5) Jinja chat template on the base tokenizer
python scripts/05_set_chat_template.py

# 6) LoRA SFT  (GPU · pushes adapter)
python scripts/06_finetune_chat.py

# 7) talk to it
python scripts/07_chat.py
```

Chat without local weights (downloads from the Hub):

```bash
export HF_USER=bychwa
rm -rf models/kitchenbot-base models/kitchenbot-chat   # force Hub path
python scripts/07_chat.py
```

### Training knobs (what the weekend run used)

**Pretrain** (`03`): lr `3e-4`, warmup `100`, batch `16 × accum 4` (effective 64), 1 epoch, fp16, `MAX_SAMPLES=10000` → **157 steps**, train loss ≈ **5.88** (last logged step loss ≈ 4.19).

**SFT** (`06`): LoRA `r=16`, `alpha=32`, targets `c_attn` + `c_proj`, lr `2e-4`, batch `8 × accum 4`, 2 epochs → **626 steps**, train loss ≈ **4.32**, mean token accuracy ≈ **0.44**.

Use `tmux` on the pod so SSH drops do not kill the job.

## Limitations (honest)

- This is a **toy SLM**. It will hallucinate, mix recipes, and fail outside cooking.
- Chat answers are only as good as the template Q&A — not a general assistant.
- Context is **256 tokens**. Long recipes get truncated.
- Metrics above are **train** curves, not a held-out eval suite.

That was the goal: finish a real pretrain → SFT loop, ship the artifacts, and keep the story reproducible.

## License

Apache-2.0. Recipe text comes from [`idoyaaran/mise-recipes`](https://huggingface.co/datasets/idoyaaran/mise-recipes) — respect that dataset’s terms when redistributing derived data.
