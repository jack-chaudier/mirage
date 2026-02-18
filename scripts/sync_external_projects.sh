#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RHUN_SRC="${RHUN_SRC:-$REPO_ROOT/../rhun}"
LORIEN_SRC="${LORIEN_SRC:-$REPO_ROOT/../lorien}"

sync_one() {
  local src="$1"
  local dst="$2"
  shift 2
  rsync -aL --prune-empty-dirs --delete "$@" "$src/" "$dst/"
}

mkdir -p "$REPO_ROOT/projects/rhun" "$REPO_ROOT/projects/lorien" "$REPO_ROOT/papers/sources/rhun" "$REPO_ROOT/papers/sources/lorien"

sync_one "$RHUN_SRC" "$REPO_ROOT/projects/rhun" \
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
  --exclude '*.pdf' \
  --exclude 'experiments/output/' \
  --exclude 'rhun.egg-info/'

sync_one "$LORIEN_SRC" "$REPO_ROOT/projects/lorien" \
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
  --exclude '*.pdf' \
  --exclude 'node_modules/' \
  --exclude 'dist/' \
  --exclude '.npm-cache/' \
  --exclude 'src/engine/.venv/' \
  --exclude 'src/engine/scripts/output/'

# Keep examples template visible even with global .env* ignore
if [[ -f "$LORIEN_SRC/.env.example" ]]; then
  cp "$LORIEN_SRC/.env.example" "$REPO_ROOT/projects/lorien/.env.example"
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
  --exclude '*.pdf' \
  "$REPO_ROOT/projects/rhun/paper/" "$REPO_ROOT/papers/sources/rhun/"

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
  --exclude '*.pdf' \
  "$REPO_ROOT/projects/lorien/paper/" "$REPO_ROOT/papers/sources/lorien/"

# Refresh canonical top-level PDFs directly from external source repos when present.
copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  else
    echo "Warning: missing source PDF, skipped: $src" >&2
  fi
}

copy_if_exists "$RHUN_SRC/paper/main.pdf" "$REPO_ROOT/papers/paper_01_absorbing_states_in_greedy_search.pdf"
copy_if_exists "$RHUN_SRC/paper/streaming_draft.pdf" "$REPO_ROOT/papers/paper_02_streaming_oscillation_traps.pdf"
copy_if_exists "$RHUN_SRC/paper/context_algebra_draft.pdf" "$REPO_ROOT/papers/paper_03_validity_mirage_compression.pdf"
copy_if_exists "$LORIEN_SRC/paper/continuous_control_structural_regularization.pdf" "$REPO_ROOT/papers/paper_00_continuous_control_structural_regularization.pdf"

echo "Sync complete."
