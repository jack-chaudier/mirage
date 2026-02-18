#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/papers/sources/rhun"
LORIEN_SRC_DIR="$ROOT_DIR/papers/sources/lorien"
MIRROR_DIR="$ROOT_DIR/projects/rhun/paper"
LORIEN_MIRROR_DIR="$ROOT_DIR/projects/lorien/paper"
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

echo "[1/5] Building RHUN papers from canonical sources..."
pushd "$SRC_DIR" >/dev/null
build_tex finite_absorbing_states.tex
build_tex streaming_oscillation_traps_draft.tex
build_tex context_algebra_draft.tex
popd >/dev/null

echo "[2/5] Building LORIEN intro paper from canonical sources..."
pushd "$LORIEN_SRC_DIR" >/dev/null
build_tex continuous_control_structural_regularization.tex
popd >/dev/null

echo "[3/5] Updating compatibility aliases in canonical source tree..."
cp "$SRC_DIR/finite_absorbing_states.tex" "$SRC_DIR/main.tex"
cp "$SRC_DIR/finite_absorbing_states.pdf" "$SRC_DIR/main.pdf"
cp "$SRC_DIR/streaming_oscillation_traps_draft.tex" "$SRC_DIR/streaming_draft.tex"
cp "$SRC_DIR/streaming_oscillation_traps_draft.pdf" "$SRC_DIR/streaming_draft.pdf"
cp "$SRC_DIR/context_algebra_draft.tex" "$SRC_DIR/context_algebra_endogenous_semantics_draft.tex"
cp "$SRC_DIR/context_algebra_draft.pdf" "$SRC_DIR/context_algebra_endogenous_semantics_draft.pdf"
cp "$LORIEN_SRC_DIR/continuous_control_structural_regularization.tex" "$LORIEN_SRC_DIR/paper2.tex"
cp "$LORIEN_SRC_DIR/continuous_control_structural_regularization.pdf" "$LORIEN_SRC_DIR/paper2.pdf"

echo "[4/5] Syncing canonical sources into mirror paths..."
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

mkdir -p "$LORIEN_MIRROR_DIR"
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
  "$LORIEN_SRC_DIR/" "$LORIEN_MIRROR_DIR/"

echo "[5/5] Refreshing top-level paper PDFs..."
cp "$LORIEN_SRC_DIR/continuous_control_structural_regularization.pdf" "$OUT_DIR/paper_00_continuous_control_structural_regularization.pdf"
cp "$SRC_DIR/finite_absorbing_states.pdf" "$OUT_DIR/paper_01_absorbing_states_in_greedy_search.pdf"
cp "$SRC_DIR/streaming_oscillation_traps_draft.pdf" "$OUT_DIR/paper_02_streaming_oscillation_traps.pdf"
cp "$SRC_DIR/context_algebra_draft.pdf" "$OUT_DIR/paper_03_validity_mirage_compression.pdf"

echo "Done. Canonical papers are now published to papers/ and mirrored to projects/rhun/paper/ and projects/lorien/paper/."
