# Unified Repo Map

This repository is the canonical workspace for Mirage + Rhun + Lorien research artifacts.

## Canonical Areas
- `endogenous_context_theory/`: Mirage core theory/bench/training code and result artifacts.
- `papers/`: camera-ready trilogy PDFs and source mirrors.
- `projects/rhun/`: imported Rhun project context (code, experiment runners, papers, context docs).
- `projects/lorien/`: imported Lorien project context (docs/specs/code, examples, papers).

## Paper Source Locations
- P1/P2/P3 source lineage: `papers/sources/rhun/`
- Lorien paper set: `papers/sources/lorien/`
- Native imported workspaces:
  - `projects/rhun/paper/`
  - `projects/lorien/paper/`

## Import Provenance
Imported from local source repos on 2026-02-17/18:
- `/Users/jackg/rhun`
- `/Users/jackg/lorien`

The imported trees are mirrors with hygiene filters.
File-level provenance hashes: `docs/imported-projects-sha256.txt`.

## Excluded on Import
To keep this repo clean and reproducible, these classes are excluded:
- VCS metadata: `.git/`
- Local env/secrets: `.env`, `.venv/`, `.claude/`
- Caches/build noise: `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.npm-cache/`
- Heavy runtime outputs:
  - `projects/rhun/experiments/output/`
  - `projects/lorien/src/engine/scripts/output/`
  - `node_modules/`, `dist/`
- LaTeX build intermediates: `*.aux`, `*.bbl`, `*.blg`, `*.out`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`

## Practical Rule
New work should be committed here first. Upstream `rhun`/`lorien` repos are preserved, but this repo is the unified research record.
