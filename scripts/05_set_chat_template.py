"""Attach a tiny Jinja chat template to the base tokenizer."""

from pathlib import Path

from transformers import AutoTokenizer

BASE = Path("models/kitchenbot-base")

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}{{ eos_token }}\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def main() -> None:
    if not BASE.is_dir():
        raise SystemExit(f"Missing {BASE}. Pretrain (script 03) first.")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(BASE)

    demo = tok.apply_chat_template(
        [{"role": "user", "content": "How do I cook rice?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"chat_template saved into {BASE}")
    print("--- sample prompt ---")
    print(demo)


if __name__ == "__main__":
    main()
