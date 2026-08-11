"""Publish the model cards in ``hub/`` to the Hub.

Cards are versioned in this repo, then pushed as-is — so the story on
huggingface.co and the story in the README cannot drift apart.
"""

from __future__ import annotations

import os

from .config import HUB


def upload_model_cards(dry_run: bool = False) -> dict[str, str]:
    """Upload ``hub/<model>/README.md`` to each model repo. Returns repo → url."""
    cards = HUB.card_paths
    for repo_id, path in cards.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing model card: {path}")
        print(f"  {repo_id}  ←  {path}")

    if dry_run:
        print("Dry run — nothing uploaded.")
        return {repo_id: str(path) for repo_id, path in cards.items()}

    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token. Run `hf auth login`, "
            "or `export HF_TOKEN=hf_...`, then retry."
        )

    api = HfApi(token=token)
    print("Authenticated as:", api.whoami()["name"])

    uploaded = {}
    for repo_id, path in cards.items():
        uploaded[repo_id] = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Rewrite model card: training story + specs",
        )
        print(f"Uploaded {repo_id} → {uploaded[repo_id]}")
    return uploaded


__all__ = ["upload_model_cards"]
