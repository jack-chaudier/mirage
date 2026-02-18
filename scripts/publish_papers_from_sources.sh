#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/papers/sources/rhun"
MIRROR_DIR="$ROOT_DIR/projects/rhun/paper"
OUT_DIR="$ROOT_DIR/papers"

build_tex() {
  local tex_file="$1"
  local stem="${tex_file%.tex}"

  if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex_file"
    return
  fi

  pdflatex -interaction=nonstopmode -halt-on-error "$tex_file"
  if [[ -f "${stem}.aux" ]]; then
    bibtex "$stem" || true
  fi
  pdflatex -interaction=nonstopmode -halt-on-error "$tex_file"
  pdflatex -interaction=nonstopmode -halt-on-error "$tex_file"
}

echo "[1/4] Building RHUN papers from canonical sources..."
pushd "$SRC_DIR" >/dev/null
build_tex finite_absorbing_states.tex
build_tex streaming_oscillation_traps_draft.tex
build_tex context_algebra_draft.tex
popd >/dev/null

echo "[2/4] Updating compatibility aliases in canonical source tree..."
cp "$SRC_DIR/finite_absorbing_states.tex" "$SRC_DIR/main.tex"
cp "$SRC_DIR/finite_absorbing_states.pdf" "$SRC_DIR/main.pdf"
cp "$SRC_DIR/streaming_oscillation_traps_draft.tex" "$SRC_DIR/streaming_draft.tex"
cp "$SRC_DIR/streaming_oscillation_traps_draft.pdf" "$SRC_DIR/streaming_draft.pdf"
cp "$SRC_DIR/context_algebra_draft.tex" "$SRC_DIR/context_algebra_endogenous_semantics_draft.tex"
cp "$SRC_DIR/context_algebra_draft.pdf" "$SRC_DIR/context_algebra_endogenous_semantics_draft.pdf"

echo "[3/4] Syncing canonical sources into mirror path..."
mkdir -p "$MIRROR_DIR"
rsync -a --delete \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.blg' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.fls' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.synctex.gz' \
  --exclude '*.toc' \
  "$SRC_DIR/" "$MIRROR_DIR/"

echo "[4/4] Refreshing top-level paper PDFs..."
cp "$SRC_DIR/finite_absorbing_states.pdf" "$OUT_DIR/paper_01_absorbing_states_in_greedy_search.pdf"
cp "$SRC_DIR/streaming_oscillation_traps_draft.pdf" "$OUT_DIR/paper_02_streaming_oscillation_traps.pdf"
cp "$SRC_DIR/context_algebra_draft.pdf" "$OUT_DIR/paper_03_validity_mirage_compression.pdf"

echo "Done. Canonical papers are now published to papers/ and mirrored to projects/rhun/paper/."
