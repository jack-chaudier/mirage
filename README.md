# Mirage: Endogenous Context, MirageBench, and Mirage-Aware Training

This repository contains the core code and curated research artifacts for the Mirage program:
- Endogenous context theory and tropical semiring validation
- MirageBench experiments across compression and difficulty frontiers
- Mirage-aware fine-tuning/evaluation workflows for explicit evidence degradation reporting

## What Is Included
- `endogenous_context_theory/`: core code, generators, tests, and runners
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

## Reproducibility
1. Start with `endogenous_context_theory/README.md`.
2. Use the notebook(s) and scripts in `endogenous_context_theory/` for data generation, training, and eval.
3. Follow `docs/reproducibility-checklist.md`.

## Notes on Excluded Artifacts
Large tar/zip model artifacts and local environment files are intentionally excluded from git history via `.gitignore`.
