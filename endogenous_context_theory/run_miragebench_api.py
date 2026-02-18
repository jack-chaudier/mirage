#!/usr/bin/env python3
"""Backward-compatible wrapper for scripts/run_miragebench_api.py."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_miragebench_api import main  # noqa: E402


if __name__ == "__main__":
    main()
