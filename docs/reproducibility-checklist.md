# Reproducibility Checklist

## A. Verify Archive Integrity

```bash
cd /Users/jackg/mirage_research_dossier_2026-02-17_v2
shasum -a 256 -c provenance/SHA256SUMS.txt
```

## B. Recompute Both Qwen Runs from CSV

Inputs:

- Balanced: `external/downloads/mirage_aware_eval_results.csv`
- Imbalanced: `external/downloads/mirage_aware_imbalanced_eval_results_2026-02-17.csv`

Compare against:

- `derived/qwen_balanced_metrics_verified.csv`
- `derived/qwen_imbalanced_metrics_verified_2026_02_17.csv`
- `derived/qwen_balanced_vs_imbalanced_comparison_2026_02_17.csv`

## C. Validate Balanced vs Imbalanced Delta Claims

Use:

- `derived/qwen_balanced_vs_imbalanced_example_deltas_2026_02_17.csv`

Expected:

- 3 FT pivot-correctness improvements
- 0 FT regressions on this slice
- additional degraded-flagging improvements in degraded rows

## D. Validate Mirage Data Split Properties

Inspect:

- `projects/mirage/endogenous_context_theory/training_data/data_stats.json`

Check:

- train/valid task disjointness
- per-task coverage and compression-level coverage
- prereq ratio distribution includes graded bins

## E. Rebuild Balanced Train (If Needed)

```bash
cd projects/mirage/endogenous_context_theory
python3 make_balanced_train.py \
  --train-jsonl training_data/train.jsonl \
  --out-jsonl training_data/train_balanced.jsonl \
  --seed 42
```

## F. Re-run Core Theory Suite

```bash
cd projects/mirage/endogenous_context_theory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Compare to:

- `results/raw/run_all_summary.csv`
- `results/artifact_correction_report.md`

## G. Re-run Qwen Colab Ablations

Balanced notebook:

- `projects/mirage/qwen_mirage_aware_balanced_ablation_colab.ipynb`

Imbalanced notebook artifact (captured from latest run):

- `external/downloads/mirage_aware_colab_fixed_imbalanced_2026-02-17.ipynb`

Eval subset lock:

- seed = 42
- shuffle valid
- first 400 rows

## H. Evaluator Guardrail

`projects/mirage/endogenous_context_theory/eval_mirage_aware.py` should parse pivot IDs with 1–4 actor digits (`\d{1,4}`), matching notebook logic.
