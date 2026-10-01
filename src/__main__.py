"""Memungkinkan ``python -m src <command>`` dari root project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
