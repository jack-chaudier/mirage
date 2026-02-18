# Mirage: Endogenous Context, MirageBench, and Mirage-Aware Training

This repository contains the core code and curated research artifacts for the Mirage program:
- Endogenous context theory and tropical semiring validation
- MirageBench experiments across compression and difficulty frontiers
- Mirage-aware fine-tuning/evaluation workflows for explicit evidence degradation reporting

## Paper 03 Public Release Bundle

For artifact-first review (same layout intended for public release), start here:

- `endogenous_context_theory/release/README.md`

It includes:
- section-to-artifact mapping for Paper 03
- curated notebooks, tasks, results CSVs, and adapter weights
- release figures and rebuild instructions

## Paper Set
- `papers/paper_01_absorbing_states_in_greedy_search.pdf`
- `papers/paper_02_streaming_oscillation_traps.pdf`
- `papers/paper_03_validity_mirage_compression.pdf`
- Full LaTeX sources (all paper drafts + bib + figures):
  - `papers/sources/rhun/`
  - `papers/sources/lorien/`

## What Is Included
- `endogenous_context_theory/`: core code, generators, tests, and runners
- `papers/`: primary manuscript set and supporting roadmap document
- `projects/rhun/`: imported Rhun code, experiments, context docs, and paper workspace
- `projects/lorien/`: imported Lorien code, docs, specs, examples, and paper workspace
- `docs/`: consolidated research context, independent review, artifact index, reproducibility checklist
  - unified mirror map: `docs/unified-repo-map.md`
- `derived/`: independently verified summary tables (balanced vs imbalanced Qwen comparisons)
- Colab notebooks for training/evaluation:
  - `qwen_mirage_aware_training_eval_colab.ipynb`
  - `qwen_mirage_aware_balanced_ablation_colab.ipynb`
  - `qwen_mirage_aware_base_eval_colab.ipynb`
  - `gemma2b_mirage_aware_training_eval_colab.ipynb`

## Latest Verified Qwen Results (400-example eval slice)
Balanced run:
- FT pivot accuracy (all): 99.2%
- FT degradation flag rate (degraded): 95.4%
- FT silent mirage rate: 0.3%
- FT false alarm (strong): 0.0%

Imbalanced run:
- FT pivot accuracy (all): 100.0%
- FT degradation flag rate (degraded): 100.0%
- FT silent mirage rate: 0.0%
- FT false alarm (strong): 0.0%

See `derived/qwen_balanced_vs_imbalanced_comparison_2026_02_17.csv` for side-by-side metrics and `docs/massive-context-document.md` for full context.

## Real-World NTSB Benchmark (Cleaned Dataset)
Cleanup/validation gate:
- `k` violations before manifest: `6/12`
- `k` violations after manifest: `0/12`
- structural validation errors: `0` before and after

xAI non-reasoning run:
- model: `grok-4-1-fast-non-reasoning`
- results: `endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/`

Key findings:
- Primary comparison uses an exact retention match at budget `0.7`:
  - mean retention (naive vs contract): `0.686859` vs `0.686859`
  - naive silent mirage (degraded): `12/51` (`23.5%`)
  - all-trial Wilson CI at matched retention: naive `12/60` `[11.83%, 31.78%]`, contract `0/60` `[0.00%, 6.02%]`
- Secondary near-match (cross-budget): naive `0.5` vs contract `0.3`
  - mean retention: `0.478276` vs `0.450570`
- Naive compression causes large attribution shift on real incident graphs:
  - `info_shift_rate`: `40.0%` (`0.7` budget), `55.0%` (`0.5`), `76.7%` (`0.3`)
- Contract compression removes attribution shift:
  - `info_shift_rate`: `0.0%` at all budgets
- Silent mirage appears under naive LLM behavior:
  - degraded silent-mirage: `12/51` (`23.5%`) at `0.7`, `11/56` (`19.6%`) at `0.5`, `13/57` (`22.8%`) at `0.3`
  - overall naive degraded silent-mirage: `36/164` (`21.95%`, Wilson 95% CI `[16.3%, 28.9%]`)

Detailed write-up:
- `endogenous_context_theory/results/ntsb/README.md`
- `endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.md`

## Reproducibility
1. Start with `endogenous_context_theory/README.md`.
2. Use canonical scripts in `endogenous_context_theory/scripts/` and notebooks in `endogenous_context_theory/notebooks/`.
3. Follow `docs/reproducibility-checklist.md`.
4. For public/arXiv publication handoff, follow `docs/public_release_checklist.md`.

## Unified Workspace Notes
- This repo is the canonical home for ongoing work across Mirage + Rhun + Lorien.
- External repos are mirrored under `projects/` with caches, secrets, and heavy runtime outputs excluded.
- Paper 3 PDF here is synced to the latest `projects/rhun/paper/context_algebra_draft.pdf`.
- Re-sync helper: `scripts/sync_external_projects.sh`.

## Notes on Excluded Artifacts
Large tar/zip model artifacts and local environment files are intentionally excluded from git history via `.gitignore`.
