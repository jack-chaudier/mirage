#!/usr/bin/env python3
"""Backward-compatible wrapper for scripts/eval_mirage_aware.py."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_mirage_aware import main  # noqa: E402


if __name__ == "__main__":
    main()
