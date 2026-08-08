"""Interactive KitchenBot chat — loads local weights or Hub (base + LoRA)."""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

HF_USER = os.environ.get("HF_USER", "bychwa")
LOCAL_BASE = "models/kitchenbot-base"
LOCAL_ADAPTER = "models/kitchenbot-chat"


def resolve_paths() -> tuple[str, str]:
    if os.path.isdir(LOCAL_BASE) and os.path.isdir(LOCAL_ADAPTER):
        print("Loading from local models/")
        return LOCAL_BASE, LOCAL_ADAPTER
    base = f"{HF_USER}/kitchenbot-base"
    adapter = f"{HF_USER}/kitchenbot-chat"
    print(f"Loading from Hub: {base} + {adapter}")
    return base, adapter


def main() -> None:
    base_id, adapter_id = resolve_paths()
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    print("Loading KitchenBot…")
    tok = AutoTokenizer.from_pretrained(adapter_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map=device_map
    )
    if not use_cuda:
        model = model.to("cpu")
    model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()

    def ask(question: str, max_new_tokens: int = 150) -> None:
        messages = [{"role": "user", "content": question}]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt")
        if use_cuda:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                streamer=streamer,
            )

    print("KitchenBot ready. Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            break
        print("KitchenBot: ", end="", flush=True)
        ask(q)
        print()


if __name__ == "__main__":
    main()
