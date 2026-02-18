# Artifact Index

## Start Here
- `README.md`
- `papers/README.md`
- `docs/massive-context-document.md`
- `docs/independent-review-and-value-assessment.md`
- `docs/reproducibility-checklist.md`

## Papers
- `papers/paper_01_absorbing_states_in_greedy_search.pdf`
- `papers/paper_02_streaming_oscillation_traps.pdf`
- `papers/paper_03_validity_mirage_compression.pdf`
- `papers/lorien_audit_and_roadmap.docx`

## Core Theory and Experiments
- `endogenous_context_theory/src/`
- `endogenous_context_theory/tests/`
- `endogenous_context_theory/scripts/run_all.py`
- `endogenous_context_theory/results/summary_report.md`
- `endogenous_context_theory/results/artifact_correction_report.md`

## MirageBench and Training Pipeline
- `endogenous_context_theory/notebooks/miragebench_experiments_colab.ipynb`
- `endogenous_context_theory/scripts/run_miragebench_ollama.py`
- `endogenous_context_theory/scripts/run_miragebench_api.py`
- `endogenous_context_theory/scripts/generate_training_data.py`
- `endogenous_context_theory/scripts/make_balanced_train.py`
- `endogenous_context_theory/scripts/eval_mirage_aware.py`
- `endogenous_context_theory/scripts/train_mirage_aware.sh`

## Data
- `endogenous_context_theory/data/processed/data_stats.json`
- `endogenous_context_theory/data/processed/train_balanced.jsonl`
- `endogenous_context_theory/data/processed/valid.jsonl`
- `endogenous_context_theory/data/smoke/`

## Notebooks
- `qwen_mirage_aware_training_eval_colab.ipynb`
- `qwen_mirage_aware_balanced_ablation_colab.ipynb`
- `qwen_mirage_aware_base_eval_colab.ipynb`
- `gemma2b_mirage_aware_training_eval_colab.ipynb`

## Verified Result Tables
- `derived/qwen_balanced_metrics_verified.csv`
- `derived/qwen_balanced_stratified_verified.csv`
- `derived/qwen_imbalanced_metrics_verified_2026_02_17.csv`
- `derived/qwen_imbalanced_stratified_verified_2026_02_17.csv`
- `derived/qwen_imbalanced_summary_from_package_2026_02_17.json`
- `derived/qwen_balanced_vs_imbalanced_comparison_2026_02_17.csv`
- `derived/qwen_balanced_vs_imbalanced_example_deltas_2026_02_17.csv`

## NTSB Real-World Benchmark
- `endogenous_context_theory/data/ntsb/ntsb_k_phase_cleanup_manifest.csv`
- `endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json`
- `endogenous_context_theory/scripts/validate_ntsb_graphs.py`
- `endogenous_context_theory/scripts/apply_ntsb_manifest.py`
- `endogenous_context_theory/scripts/run_ntsb_mirage_benchmark.py`
- `endogenous_context_theory/scripts/build_ntsb_paper_table.py`
- `endogenous_context_theory/results/ntsb/README.md`
- `endogenous_context_theory/results/ntsb/ntsb_validation_report.before.json`
- `endogenous_context_theory/results/ntsb/ntsb_validation_report.after.json`
- `endogenous_context_theory/results/ntsb/mirage_results_summary.csv`
- `endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/mirage_results_summary.csv`
- `endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.csv`
- `endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.md`
