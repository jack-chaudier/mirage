# RHUN Paper Sources (Canonical)

This directory is the source of truth for RHUN papers in this repository.
Edit `.tex` and `.bib` files here first.
Compiled publication PDFs are tracked in `papers/` at repo root.

After edits, publish with:

```bash
./scripts/publish_papers_from_sources.sh
```

That command rebuilds PDFs, syncs the mirror at `projects/rhun/paper/`, and refreshes
the top-level publication PDFs in `papers/`.

## Canonical files
- `finite_absorbing_states.tex` -> `main.tex`
- `streaming_oscillation_traps_draft.tex` -> `streaming_draft.tex`
- `context_algebra_endogenous_semantics_draft.tex` -> `context_algebra_draft.tex`

## Figure scripts
- `generate_figures.py` (finite-paper figures)
- `generate_streaming_figures.py` (streaming-paper figures)

## Figure directories
- `figures/` (finite)
- `figures/streaming/` (streaming)

## Interactive explainer
- `interactive_paper_explainer.html`
  - Self-contained interactive guide for `main.pdf` and `streaming_draft.pdf`
  - Includes concept map, finite `jdev` vs `k` simulator, failure-class explorer, algorithm hierarchy comparison, and streaming patience/trap playground
