# Blackbox Sweep Results (bf16, 5 models)

## Dataset

- Tasks: 12 MirageBench tasks
- Compression levels: `0.4`, `0.5`, `0.6`
- Models: Llama 3.1 8B, Mistral 7B v0.3, Gemma 2 9B, Phi-3 Medium 14B, Qwen 2.5 14B
- Total rows in merged CSV: 180

## Files

- `miragebench_bf16_5model_merged.csv` (canonical merged output)
- `*_results.csv` (per-model outputs)
- `miragebench_bf16_5model_summary_by_model_release.csv`
- `miragebench_bf16_5model_summary_by_category_release.csv`

## Primary Metrics

- `raw_validity`
- `pivot_preserved`
- `fixed_pivot_feasible`
- `semantic_regret`
- `valid_but_switched` (release summary derived metric)
