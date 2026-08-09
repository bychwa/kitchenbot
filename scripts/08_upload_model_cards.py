"""Upload hub/*/README.md model cards to Hugging Face (needs a write token)."""

import os
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
CARDS = {
    "bychwa/kitchenbot-base": ROOT / "hub/kitchenbot-base/README.md",
    "bychwa/kitchenbot-chat": ROOT / "hub/kitchenbot-chat/README.md",
}


def main() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "No HF token found. Run:  hf auth login\n"
            "or:  export HF_TOKEN=hf_...\n"
            "then re-run this script."
        )

    api = HfApi(token=token)
    print("Authenticated as:", api.whoami()["name"])

    for repo_id, path in CARDS.items():
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        url = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Rewrite model card: weekend training story + specs",
        )
        print(f"Uploaded {repo_id} → {url}")


if __name__ == "__main__":
    main()
