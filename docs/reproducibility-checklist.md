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

## H. NTSB Real-World Benchmark
Run validator and apply fixed manifest:

```bash
cd /Users/jackg/mirage
/Users/jackg/mirage/.venv/bin/python endogenous_context_theory/scripts/validate_ntsb_graphs.py \
  --input-json /Users/jackg/Downloads/ntsb_event_graphs.json \
  --report-json endogenous_context_theory/results/ntsb/ntsb_validation_report.before.json \
  --audit-csv endogenous_context_theory/results/ntsb/ntsb_incident_audit.before.csv

/Users/jackg/mirage/.venv/bin/python endogenous_context_theory/scripts/apply_ntsb_manifest.py \
  --input-json /Users/jackg/Downloads/ntsb_event_graphs.json \
  --manifest-csv endogenous_context_theory/data/ntsb/ntsb_k_phase_cleanup_manifest.csv \
  --output-json endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json \
  --changes-json endogenous_context_theory/results/ntsb/ntsb_manifest_apply_changes.json

/Users/jackg/mirage/.venv/bin/python endogenous_context_theory/scripts/validate_ntsb_graphs.py \
  --input-json endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json \
  --report-json endogenous_context_theory/results/ntsb/ntsb_validation_report.after.json \
  --audit-csv endogenous_context_theory/results/ntsb/ntsb_incident_audit.after.csv \
  --fail-on-errors
```

Run xAI non-reasoning benchmark:

```bash
cd /Users/jackg/mirage
set -a
source .env
set +a
export OPENAI_API_KEY="$XAI_API_KEY"
export OPENAI_BASE_URL="https://api.x.ai/v1"

/Users/jackg/mirage/.venv/bin/python endogenous_context_theory/scripts/run_ntsb_mirage_benchmark.py \
  --input-json endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json \
  --output-dir endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning \
  --backend openai \
  --model grok-4-1-fast-non-reasoning \
  --budgets 0.7,0.5,0.3 \
  --seeds 11,22,33,44,55 \
  --methods naive,contract \
  --max-tokens 512 \
  --temperature 0.0 \
  --timeout-s 120
```

Check:
- `endogenous_context_theory/results/ntsb/README.md`
- `endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/mirage_results_summary.csv`

## I. Legacy Command Compatibility
These still work and delegate to `scripts/`:
- `python endogenous_context_theory/run_all.py`
- `python endogenous_context_theory/generate_training_data.py`
- `python endogenous_context_theory/eval_mirage_aware.py`
- `bash endogenous_context_theory/train_mirage_aware.sh`
