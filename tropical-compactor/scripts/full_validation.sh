#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Running unit and property tests"
uv run pytest -q

echo "[2/5] Building distributable artifacts"
uv run python -m build

echo "[3/5] Running comprehensive functional validation"
uv run python scripts/run_full_validation.py

echo "[4/5] Running quick replay benchmark sweep"
uv run tropical-compactor-replay \
  --fractions 1.0,0.8,0.65,0.5,0.4 \
  --policies recency,l2_guarded \
  --k 3 \
  --line-count 200 \
  --output-dir artifacts/cyberops_mcp_replay_quickcheck

echo "[5/5] Validation complete"
echo "Artifacts:"
echo "  - artifacts/cyberops_mcp_replay_quickcheck/replay_rows.csv"
echo "  - artifacts/cyberops_mcp_replay_quickcheck/replay_summary.csv"
echo "  - artifacts/cyberops_mcp_replay_quickcheck/replay_summary.json"
