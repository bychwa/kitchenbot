"""Deprecated alias for ``python -m kitchenbot download``.

Kept so older guides and muscle memory keep working. The real code lives in
``kitchenbot/data.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kitchenbot.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
