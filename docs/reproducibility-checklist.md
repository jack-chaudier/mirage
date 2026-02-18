# Reproducibility Checklist

## A. Environment
```bash
cd endogenous_context_theory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## B. Run Core Theory Suite
```bash
cd endogenous_context_theory
python scripts/run_all.py
```

Check outputs:
- `endogenous_context_theory/results/summary_report.md`
- `endogenous_context_theory/results/raw/run_all_summary.csv`

## C. Generate Training Data
```bash
cd endogenous_context_theory
python scripts/generate_training_data.py \
  --output-dir data/processed \
  --categories investment,incident,narrative \
  --compression-levels 0.4,0.5,0.6 \
  --seeds 101,202,303 \
  --target-prereq-ratios 0.0,0.25,0.5,0.75 \
  --max-context-words 1200 \
  --split-seed 42 \
  --no-balance-labels
```

Validate:
- `endogenous_context_theory/data/processed/data_stats.json`

## D. Optional Balanced Train File
```bash
cd endogenous_context_theory
python scripts/make_balanced_train.py \
  --train-jsonl data/processed/train.jsonl \
  --out-jsonl data/processed/train_balanced.jsonl \
  --seed 42
```

## E. Run Mirage-Aware Training + Eval
```bash
cd endogenous_context_theory
bash scripts/train_mirage_aware.sh
```

Outputs:
- `endogenous_context_theory/results/mirage_aware_eval_results.csv`
- `endogenous_context_theory/results/mirage_aware_eval_summary.csv`

## F. Evaluate Existing Adapter Explicitly
```bash
cd endogenous_context_theory
python scripts/eval_mirage_aware.py \
  --valid-jsonl data/processed/valid.jsonl
```

## G. Compare Against Verified Tables
- `derived/qwen_balanced_metrics_verified.csv`
- `derived/qwen_imbalanced_metrics_verified_2026_02_17.csv`
- `derived/qwen_balanced_vs_imbalanced_comparison_2026_02_17.csv`

## H. Legacy Command Compatibility
These still work and delegate to `scripts/`:
- `python endogenous_context_theory/run_all.py`
- `python endogenous_context_theory/generate_training_data.py`
- `python endogenous_context_theory/eval_mirage_aware.py`
- `bash endogenous_context_theory/train_mirage_aware.sh`
