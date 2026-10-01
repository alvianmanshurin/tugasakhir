#!/usr/bin/env python3
"""
Entry point terpadu proyek Vehicle Counting System.

    python main.py --help
    python main.py train --quick
    python main.py workflow --list

Setara dengan ``python -m src <command>``. Cara lama
(``python src/train.py``) tetap didukung.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from cli import main  # noqa: E402  (src/cli.py, setelah bootstrap path)

if __name__ == "__main__":
    raise SystemExit(main())
