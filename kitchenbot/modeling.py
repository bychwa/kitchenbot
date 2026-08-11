"""Model construction, loading and generation.

Kept separate from training so inference never has to import TRL, and so the
Hub-backed chat path and the local-weights path share one implementation.
"""

from __future__ import annotations

from pathlib import Path

from .config import HUB, PATHS, PretrainConfig


def build_base_model(tokenizer, cfg: PretrainConfig = PretrainConfig()):
    """A fresh GPT-2-style causal LM sized to the weekend run (~6.85M params)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=cfg.max_len,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    return model, summarize(model)


def summarize(model) -> dict[str, float]:
    """Parameter count in millions — printed once so the README stays honest."""
    total = sum(p.numel() for p in model.parameters())
    return {"params": total, "params_m": round(total / 1e6, 2)}


def resolve_chat_paths(
    base: str | Path = PATHS.base, adapter: str | Path = PATHS.adapter
) -> tuple[str, str]:
    """Prefer local weights, fall back to the published Hub repos."""
    if Path(base).is_dir() and Path(adapter).is_dir():
        return str(base), str(adapter)
    print(f"Local weights not found — using Hub: {HUB.base_id} + {HUB.chat_id}")
    return HUB.base_id, HUB.chat_id


def load_for_chat(base_id: str, adapter_id: str):
    """Load base + LoRA adapter, half precision on GPU, float32 on CPU."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(adapter_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model = model.to("cpu")
    model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    return model, tokenizer


def generate_reply(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """One-shot answer. Uses the streaming decoder when a streamer is requested."""
    import torch
    from transformers import TextStreamer

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def stream_reply(model, tokenizer, question: str, **kwargs) -> None:
    """Same call as ``generate_reply``, printed token-by-token to stdout."""
    import torch
    from transformers import TextStreamer

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 150),
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
