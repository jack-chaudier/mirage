# NTSB Real-World Mirage Benchmark

This folder contains the real-incident compression benchmark artifacts built from:
`/Users/jackg/Downloads/ntsb_event_graphs.json`

## 1) Cleanup Gate

Validation before and after applying the fixed six-row manifest.

- Before cleanup: `k_violations = 6`, structural errors `= 0`
- After cleanup: `k_violations = 0`, structural errors `= 0`

Key files:
- `ntsb_validation_report.before.json`
- `ntsb_incident_audit.before.csv`
- `ntsb_manifest_apply_changes.json`
- `ntsb_validation_report.after.json`
- `ntsb_incident_audit.after.csv`

## 2) Rule Backend Baseline (Structural Check)

Deterministic backend (`--backend rule`) confirms the structural phenomenon.

- Naive compression info-shift (root-cause attribution changes):
  - 0.7 budget: `40.0%`
  - 0.5 budget: `55.0%`
  - 0.3 budget: `76.7%`
- Contract compression info-shift: `0.0%` at all budgets
- Silent mirage: `0.0%` (expected under deterministic rule backend)

Files:
- `mirage_results_per_example.csv`
- `mirage_results_summary.csv`
- `mirage_full_predictions.json`

## 3) xAI Non-Reasoning LLM Run

Model: `grok-4-1-fast-non-reasoning`

Path:
- `xai_grok_4_1_fast_non_reasoning/mirage_results_per_example.csv`
- `xai_grok_4_1_fast_non_reasoning/mirage_results_summary.csv`
- `xai_grok_4_1_fast_non_reasoning/mirage_full_predictions.json`

### Budget-level results (xAI)

| Method | Budget | Info Shift | Pivot Preservation | Pivot Accuracy | Silent Mirage (degraded) |
|---|---:|---:|---:|---:|---:|
| naive | 0.7 | 40.0% | 60.0% | 80.0% | 12/51 (23.5%) |
| naive | 0.5 | 55.0% | 45.0% | 81.7% | 11/56 (19.6%) |
| naive | 0.3 | 76.7% | 23.3% | 78.3% | 13/57 (22.8%) |
| contract | 0.7 | 0.0% | 100.0% | 100.0% | 0/4 (0.0%) |
| contract | 0.5 | 0.0% | 100.0% | 100.0% | 0/8 (0.0%) |
| contract | 0.3 | 0.0% | 100.0% | 100.0% | 0/10 (0.0%) |

### Primary retention-matched comparison (exact)

At budget `0.7`, both methods have identical mean achieved retention:
- naive: `0.686859`
- contract: `0.686859`

Silent mirage on degraded rows:
- naive: `12/51` (`23.53%`)
- contract: `0/4` (`0.00%`)

Wilson 95% CI with all-trial denominator (retention-matched panel):
- naive: `12/60` -> `[11.83%, 31.78%]`
- contract: `0/60` -> `[0.00%, 6.02%]`

### Secondary retention-matched comparison (near match)

Cross-budget comparison with close retention:
- naive at budget `0.5`: retention `0.478276`, silent `11/56` (`19.64%`), all-trial Wilson `11/60 -> [10.56%, 29.92%]`
- contract at budget `0.3`: retention `0.450570`, silent `0/10` (`0.00%`), all-trial Wilson `0/60 -> [0.00%, 6.02%]`

### Overall naive xAI behavior

- `info_shift_rate`: `57.22%` (103/180)
- degraded wrong-pivot cases: `36/164`
- degraded silent mirages: `36/164` (`21.95%`, Wilson 95% CI `[16.30%, 28.89%]`)
- `flag_degraded_rate` on degraded rows: `0.0%`
- `flag_given_wrong`: `0/36` (`0.0%`)

Interpretation:
- The structural mirage appears on real NTSB causal chains under naive compression.
- Contract enforcement removes attribution shift in this dataset.
- The non-reasoning LLM exhibits the behavioral failure mode (silent mirage) on naive compressed contexts.
- Contract success here does not depend on model honesty signaling (`flag_degraded_rate` is `0.0%` even when contract prevents mirage). The guarantee is structural: the contract prevents degraded attribution states from being presented to the model.

## 4.1) Paper Figure Table Artifacts

Generated directly from CSV outputs:
- `xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.csv`
- `xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.md`
- `xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.json`

## 5) Reproduction Commands

Validate + clean:

```bash
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
set -a
source /Users/jackg/mirage/.env
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

/Users/jackg/mirage/.venv/bin/python endogenous_context_theory/scripts/build_ntsb_paper_table.py
```
