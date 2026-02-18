# Mirage: Endogenous Context, MirageBench, and Mirage-Aware Training

This repository contains the core code and curated research artifacts for the Mirage program:
- Endogenous context theory and tropical semiring validation
- MirageBench experiments across compression and difficulty frontiers
- Mirage-aware fine-tuning/evaluation workflows for explicit evidence degradation reporting

## Paper Set
- `papers/paper_01_absorbing_states_in_greedy_search.pdf`
- `papers/paper_02_streaming_oscillation_traps.pdf`
- `papers/paper_03_validity_mirage_compression.pdf`

## What Is Included
- `endogenous_context_theory/`: core code, generators, tests, and runners
- `papers/`: primary manuscript set and supporting roadmap document
- `docs/`: consolidated research context, independent review, artifact index, reproducibility checklist
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
- Naive compression causes large attribution shift on real incident graphs:
  - `info_shift_rate`: `40.0%` (`0.7` budget), `55.0%` (`0.5`), `76.7%` (`0.3`)
- Contract compression removes attribution shift:
  - `info_shift_rate`: `0.0%` at all budgets
- Silent mirage appears under naive LLM behavior:
  - degraded silent-mirage: `12/51` (`23.5%`) at `0.7`, `11/56` (`19.6%`) at `0.5`, `13/57` (`22.8%`) at `0.3`
  - overall naive degraded silent-mirage: `36/164` (`21.95%`)

Detailed write-up:
- `endogenous_context_theory/results/ntsb/README.md`

## Reproducibility
1. Start with `endogenous_context_theory/README.md`.
2. Use canonical scripts in `endogenous_context_theory/scripts/` and notebooks in `endogenous_context_theory/notebooks/`.
3. Follow `docs/reproducibility-checklist.md`.

## Notes on Excluded Artifacts
Large tar/zip model artifacts and local environment files are intentionally excluded from git history via `.gitignore`.
