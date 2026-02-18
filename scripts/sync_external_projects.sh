#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RHUN_SRC="${RHUN_SRC:-/Users/jackg/rhun}"
LORIEN_SRC="${LORIEN_SRC:-/Users/jackg/lorien}"

sync_one() {
  local src="$1"
  local dst="$2"
  shift 2
  rsync -aL --prune-empty-dirs --delete "$@" "$src/" "$dst/"
}

mkdir -p "$ROOT_DIR/projects/rhun" "$ROOT_DIR/projects/lorien" "$ROOT_DIR/papers/sources/rhun" "$ROOT_DIR/papers/sources/lorien"

sync_one "$RHUN_SRC" "$ROOT_DIR/projects/rhun" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.claude/' \
  --exclude '.DS_Store' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.blg' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.toc' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.fls' \
  --exclude '*.synctex.gz' \
  --exclude 'experiments/output/' \
  --exclude 'rhun.egg-info/'

sync_one "$LORIEN_SRC" "$ROOT_DIR/projects/lorien" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.claude/' \
  --exclude '.DS_Store' \
  --exclude '.env' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.blg' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.toc' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.fls' \
  --exclude '*.synctex.gz' \
  --exclude 'node_modules/' \
  --exclude 'dist/' \
  --exclude '.npm-cache/' \
  --exclude 'src/engine/.venv/' \
  --exclude 'src/engine/scripts/output/'

# Keep examples template visible even with global .env* ignore
if [[ -f "$LORIEN_SRC/.env.example" ]]; then
  cp "$LORIEN_SRC/.env.example" "$ROOT_DIR/projects/lorien/.env.example"
fi

rsync -a --delete \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.blg' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.toc' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.fls' \
  --exclude '*.synctex.gz' \
  "$ROOT_DIR/projects/rhun/paper/" "$ROOT_DIR/papers/sources/rhun/"

rsync -a --delete \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.blg' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.toc' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.fls' \
  --exclude '*.synctex.gz' \
  "$ROOT_DIR/projects/lorien/paper/" "$ROOT_DIR/papers/sources/lorien/"

cp "$ROOT_DIR/projects/rhun/paper/main.pdf" "$ROOT_DIR/papers/paper_01_absorbing_states_in_greedy_search.pdf"
cp "$ROOT_DIR/projects/rhun/paper/streaming_draft.pdf" "$ROOT_DIR/papers/paper_02_streaming_oscillation_traps.pdf"
cp "$ROOT_DIR/projects/rhun/paper/context_algebra_draft.pdf" "$ROOT_DIR/papers/paper_03_validity_mirage_compression.pdf"

echo "Sync complete."
