"""One place for paths, dataset ids and every hyperparameter.

Nothing else in the package hardcodes a magic number: modules import these
values, so reproducing the weekend run means reproducing this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "idoyaaran/mise-recipes"
DATASET_SPLIT = "train"

#: Byte-level BPE special tokens. ``<|user|>`` / ``<|assistant|>`` are what the
#: chat template emits, so they must exist *before* pretraining.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
)


@dataclass(frozen=True)
class Paths:
    """Where every stage reads and writes, relative to the repo root."""

    root: Path = ROOT
    corpus: Path = ROOT / "data" / "corpus.txt"
    qa: Path = ROOT / "data" / "cooking_qa.jsonl"
    samples: Path = ROOT / "data" / "samples"
    tokenizer: Path = ROOT / "tokenizer"
    models: Path = ROOT / "models"
    base: Path = ROOT / "models" / "kitchenbot-base"
    adapter: Path = ROOT / "models" / "kitchenbot-chat"
    hub_cards: Path = ROOT / "hub"


PATHS = Paths()


@dataclass(frozen=True)
class TokenizerConfig:
    """Trained in ~30 s on the full corpus; deterministic given the corpus."""

    vocab_size: int = 8_000
    min_frequency: int = 2
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS


@dataclass(frozen=True)
class PretrainConfig:
    """The exact knobs behind the published base model (157 steps, ~1.5 min)."""

    max_len: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    max_samples: int = 10_000
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    per_device_batch_size: int = 16
    grad_accum: int = 4
    epochs: int = 1
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 3


@dataclass(frozen=True)
class ChatConfig:
    """LoRA SFT settings behind the published chat adapter (626 steps)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("c_attn", "c_proj")
    learning_rate: float = 2e-4
    per_device_batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 2
    max_len: int = 256
    logging_steps: int = 25
    save_steps: int = 500
    save_total_limit: int = 3
    max_examples: int = 10_000
    seed: int = 42


@dataclass(frozen=True)
class HubConfig:
    """Hub ids are derived from ``HF_USER`` so forks publish under their own name."""

    user: str = field(default_factory=lambda: os.environ.get("HF_USER", "bychwa"))
    project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "kitchenbot")
    )

    @property
    def base_id(self) -> str:
        return f"{self.user}/kitchenbot-base"

    @property
    def chat_id(self) -> str:
        return f"{self.user}/kitchenbot-chat"

    @property
    def card_paths(self) -> dict[str, Path]:
        return {
            self.base_id: PATHS.hub_cards / "kitchenbot-base" / "README.md",
            self.chat_id: PATHS.hub_cards / "kitchenbot-chat" / "README.md",
        }


HUB = HubConfig()

#: Published weights — the artifacts this pipeline produced.
PUBLISHED = {
    "base": "https://huggingface.co/bychwa/kitchenbot-base",
    "chat": "https://huggingface.co/bychwa/kitchenbot-chat",
    "wandb": "https://wandb.ai/bychwa-bouer-tech/kitchenbot",
}
