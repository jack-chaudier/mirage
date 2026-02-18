# Artifact Index

## Start Here
- `README.md`
- `endogenous_context_theory/release/README.md`
- `papers/README.md`
- `docs/massive-context-document.md`
- `docs/independent-review-and-value-assessment.md`
- `docs/reproducibility-checklist.md`
- `docs/unified-repo-map.md`

## Paper 03 Public Bundle
- `endogenous_context_theory/release/README.md`
- `endogenous_context_theory/release/miragebench_tasks/`
- `endogenous_context_theory/release/notebooks/`
- `endogenous_context_theory/release/results/blackbox_bf16_5model/`
- `endogenous_context_theory/release/results/kv_cache_eviction_llama31_8b/`
- `endogenous_context_theory/release/adapters/mirage_aware_v1/`
- `endogenous_context_theory/release/figures/`
- `endogenous_context_theory/release/SHA256SUMS.txt`

## Papers
- `papers/paper_00_continuous_control_structural_regularization.pdf`
- `papers/paper_01_absorbing_states_in_greedy_search.pdf`
- `papers/paper_02_streaming_oscillation_traps.pdf`
- `papers/paper_03_validity_mirage_compression.pdf`
- `papers/sources/rhun/` (all Rhun paper sources, including latest P3 TeX)
- `papers/sources/lorien/` (all Lorien paper sources)
- `papers/lorien_audit_and_roadmap.docx`

## Imported Project Context (Unified Repo)
- `projects/rhun/README.md`
- `projects/rhun/RHUN_CONTEXT.md`
- `projects/rhun/rhun/` (core Rhun library code)
- `projects/rhun/experiments/` (experiment runners; heavy outputs excluded)
- `projects/rhun/paper/` (native Rhun paper workspace)
- `projects/lorien/README.md`
- `projects/lorien/docs/`
- `projects/lorien/specs/`
- `projects/lorien/src/engine/`
- `projects/lorien/src/visualization/src/`
- `projects/lorien/paper/` (native Lorien paper workspace)
- `scripts/sync_external_projects.sh` (one-command mirror refresh)
- `docs/imported-projects-sha256.txt` (provenance hash manifest for mirrored files)

## Core Theory and Experiments
- `endogenous_context_theory/src/`
- `endogenous_context_theory/tests/`
- `endogenous_context_theory/scripts/run_all.py`
- `endogenous_context_theory/results/current_status.md`
- `endogenous_context_theory/results/summary_report.md` (historical, pre-correction)
- `endogenous_context_theory/results/artifact_correction_report.md`

## MirageBench and Training Pipeline
- `endogenous_context_theory/notebooks/legacy/miragebench_experiments_colab.ipynb`
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
- `endogenous_context_theory/notebooks/legacy/qwen_mirage_aware_training_eval_colab.ipynb`
- `endogenous_context_theory/notebooks/legacy/qwen_mirage_aware_balanced_ablation_colab.ipynb`
- `endogenous_context_theory/notebooks/legacy/qwen_mirage_aware_base_eval_colab.ipynb`
- `endogenous_context_theory/notebooks/legacy/gemma2b_mirage_aware_training_eval_colab.ipynb`

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
