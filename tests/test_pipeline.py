"""Offline tests for the pipeline — no GPU, no network, no Hub account.

Run with:  uv run pytest -q
"""

from __future__ import annotations

import pytest

from kitchenbot.cli import build_parser
from kitchenbot.config import PATHS, HubConfig
from kitchenbot.data import QA_TEMPLATES, format_recipe

ROW = {
    "title": "Test Stew",
    "ingredients": [{"name": "salt"}, {"name": "water"}],
    "steps": ["Boil the water.", "Add salt."],
}


def test_format_recipe_is_stable():
    """Corpus and pretraining must agree on this exact shape."""
    assert format_recipe(ROW) == (
        "Test Stew\nIngredients: salt, water\nSteps: Boil the water. Add salt.\n"
    )


def test_format_recipe_survives_missing_fields():
    assert format_recipe({}) == "\nIngredients: \nSteps: \n"


def test_qa_templates_render():
    _, answer = QA_TEMPLATES[0]
    assert "salt" in answer.format(
        title="Test Stew", ingredients="salt, water", steps="", steps_short="", first_step=""
    )


def test_hub_ids_follow_env(monkeypatch):
    monkeypatch.setenv("HF_USER", "someone-else")
    hub = HubConfig()
    assert hub.base_id == "someone-else/kitchenbot-base"
    assert hub.chat_id == "someone-else/kitchenbot-chat"


def test_cli_exposes_every_stage():
    parser = build_parser()
    actions = next(a for a in parser._actions if a.dest == "command")
    assert set(actions.choices) == {
        "download",
        "tokenize",
        "pretrain",
        "qa",
        "template",
        "sft",
        "chat",
        "pipeline",
        "upload-cards",
    }


def test_committed_tokenizer_is_loadable():
    """Guards the shipped tokenizer/: if this breaks, every checkpoint is orphaned."""
    pytest.importorskip("transformers")
    from kitchenbot.tokenization import load_tokenizer

    tokenizer = load_tokenizer(PATHS.tokenizer)
    assert len(tokenizer) > 1000
    assert tokenizer.bos_token == "<|bos|>"
    assert tokenizer.eos_token == "<|eos|>"
    assert tokenizer.convert_tokens_to_ids("<|user|>") != tokenizer.unk_token_id


def test_chat_template_shape():
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    from kitchenbot.tokenization import CHAT_TEMPLATE, load_tokenizer

    tokenizer = load_tokenizer(PATHS.tokenizer)
    tokenizer.chat_template = CHAT_TEMPLATE

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "How do I cook rice?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt.startswith("<|user|>")
    assert prompt.rstrip().endswith("<|assistant|>")
