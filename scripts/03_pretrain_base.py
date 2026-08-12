"""Deprecated alias for ``python -m kitchenbot pretrain``.

Pretrain the base model. The real code lives in ``kitchenbot/training.py``.
Kept so older guides and muscle memory keep working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kitchenbot.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pretrain", *sys.argv[1:]]))
