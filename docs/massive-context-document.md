# Massive Context Document

**Project family:** Mirage + Rhun + Lorien  
**Updated:** February 17, 2026 (after both balanced + imbalanced Qwen runs)  
**Archive root:** `repository root`

Note: this document was imported from a broader research dossier, so some
references use archival path labels (`projects/...`, `external/...`). In this
repository, runnable code lives under `endogenous_context_theory/`.

## 0. Paper Set (Primary Reading Order)

1. `papers/paper_01_absorbing_states_in_greedy_search.pdf`
2. `papers/paper_02_streaming_oscillation_traps.pdf`
3. `papers/paper_03_validity_mirage_compression.pdf`

## 1. One-line Thesis

"Infinite context" is not just a larger token window. It is the problem of preserving endogenous semantics when interpretation depends on a pivot selected from within the sequence (argmax), making naive compression/truncation/early commitment semantically unsafe.

## 2. Program Overview (What You Have)

You have a coherent 3-stage research program:

1. **Lorien (`projects/lorien/`)**: deterministic simulation-first narrative system; origin of structural extraction failure observations.
2. **Rhun (`projects/rhun/`)**: domain-agnostic theorem + algorithm hierarchy for constrained extraction in temporal DAGs.
3. **Mirage (`projects/mirage/`)**: context algebra + MirageBench + model comparisons + mirage-aware fine-tuning.

This is a rare full arc: discovery -> formalization -> benchmark -> mitigation.

## 3. How You Got Here

## 3.1 Lorien (Discovery Layer)

From `projects/lorien/README.md`, `projects/lorien/docs/METHODOLOGY.md`, and checkpoint docs:

- Deterministic event-sourced simulation with controlled extraction and metrics.
- Reproducible evidence that structural constraints drive failure modes.
- Strong engineering hygiene (tests, audits, checkpoint artifacts).

## 3.2 Rhun (Formal Layer)

From `projects/rhun/RHUN_CONTEXT_v4.md` and paper artifacts:

- Prefix-Constraint Impossibility theorem framing.
- Failure-class decomposition.
- Constructive algorithm ladder (greedy -> viability-aware -> TP-conditioned solver -> exact oracle).
- Boundary diagnostics (pool construction differences, constraint antagonism).

## 3.3 Mirage (LLM Layer)

From `projects/mirage/endogenous_context_theory/`:

- Tropical/endogenous context algebra experiments.
- MirageBench taxonomy (`true_mirage`, `rescue`, `instability`, `stable_*`).
- Cross-model and difficulty sweeps.
- Mirage-aware fine-tuning pipeline with explicit evidence status behavior.

## 4. Core Mirage Methodology and Artifacts

## 4.1 Theory/Test Harness

Primary paths:
- `projects/mirage/endogenous_context_theory/src/`
- `projects/mirage/endogenous_context_theory/tests/`
- `projects/mirage/endogenous_context_theory/run_all.py`

Key reports:
- `projects/mirage/endogenous_context_theory/results/summary_report.md`
- `projects/mirage/endogenous_context_theory/results/artifact_correction_report.md`
- `projects/mirage/endogenous_context_theory/results/raw/run_all_summary.csv`
- `projects/mirage/endogenous_context_theory/results/raw/artifact_correction_summary.csv`

Important interpretation note:
- `run_all_summary.csv` includes pre-artifact-correction failures in some mirage/margin tests.
- `artifact_correction_report.md` captures corrected-generator reruns.

## 4.2 Training Data Pipeline

Generation scripts and outputs:
- `projects/mirage/endogenous_context_theory/generate_training_data.py`
- `projects/mirage/endogenous_context_theory/make_balanced_train.py`
- `projects/mirage/endogenous_context_theory/training_data/data_stats.json`
- `projects/mirage/endogenous_context_theory/training_data/train.jsonl`
- `projects/mirage/endogenous_context_theory/training_data/train_balanced.jsonl`
- `projects/mirage/endogenous_context_theory/training_data/valid.jsonl`

Observed structure:
- Original train split: 10,000 examples (90% degraded).
- Balanced train split: 2,000 examples (1,000 strong + 1,000 degraded).
- Valid split: 2,000 examples (90% degraded).

## 4.3 Colab/Run Notebooks

- `projects/mirage/qwen_mirage_aware_training_eval_colab.ipynb`
- `projects/mirage/qwen_mirage_aware_balanced_ablation_colab.ipynb`
- `projects/mirage/qwen_mirage_aware_base_eval_colab.ipynb`
- `projects/mirage/gemma2b_mirage_aware_training_eval_colab.ipynb`

## 5. Qwen Ablation Results (Balanced vs Imbalanced)

Both runs are on the same 400-example eval slice composition (371 degraded, 29 strong).

## 5.1 Balanced Qwen Run (Verified)

Artifacts:
- `external/downloads/mirage_aware_balanced_package.tar.gz`
- `external/downloads/mirage_aware_eval_results.csv`

Independent recompute:
- `derived/qwen_balanced_metrics_verified.csv`
- `derived/qwen_balanced_stratified_verified.csv`

Headline metrics:
- Base acc (all): 40.5%
- FT acc (all): 99.25%
- FT acc (degraded): 99.19%
- FT degradation flag (degraded): 95.42%
- FT silent mirage (degraded): 0.27%
- FT false alarm (strong): 0/29
- FT format adherence: 100%
- FT flag_given_wrong: 2/3 (66.7%)

## 5.2 Imbalanced Qwen Run (Latest, Verified)

Artifacts:
- `external/downloads/mirage_aware_imbalanced_package_2026-02-17.tar.gz`
- `external/downloads/mirage_aware_imbalanced_eval_results_2026-02-17.csv`
- `derived/qwen_imbalanced_summary_from_package_2026_02_17.json`

Independent recompute:
- `derived/qwen_imbalanced_metrics_verified_2026_02_17.csv`
- `derived/qwen_imbalanced_stratified_verified_2026_02_17.csv`

Headline metrics:
- Base acc (all): 40.5%
- FT acc (all): 100.0%
- FT acc (degraded): 100.0%
- FT degradation flag (degraded): 100.0%
- FT silent mirage (degraded): 0.0%
- FT false alarm (strong): 0/29
- FT format adherence: 100%
- FT flag_given_wrong: N/A (0 wrong FT degraded cases)

## 5.3 Direct Balanced vs Imbalanced Delta

Comparison files:
- `derived/qwen_balanced_vs_imbalanced_comparison_2026_02_17.csv`
- `derived/qwen_balanced_vs_imbalanced_example_deltas_2026_02_17.csv`

Observed deltas:
- FT pivot errors: 3 -> 0
- FT silent mirage count: 1 -> 0
- FT degraded-flag rate: 95.4% -> 100%
- 19 per-example FT output deltas detected:
  - 3 were pivot-correctness improvements
  - 17 were stricter degraded flagging improvements
  - 0 FT regressions observed on this slice

## 6. Cross-model MirageBench Evidence Already in Repo

From `projects/mirage/endogenous_context_theory/results/`:
- `haiku_summary.csv`
- `sonnet_summary.csv`
- `grok_summary.csv`
- `haiku_investment_difficulty_summary.csv`
- `grok_fast_non_reasoning_investment_difficulty_summary.csv`

Pattern reflected in these artifacts:
- Mirage and rescue are both real.
- Category sensitivity is non-trivial (investment often hardest).
- Difficulty-frontier behavior varies by model.

## 7. Independent Assessment: What Is Established vs Open

## 7.1 Established (from current artifact set)

1. The phenomenon is real and measurable.
2. You have a validated metric framework for semantic instability under compression.
3. Mirage-aware structured behavior is strongly learnable in your current synthetic regime.
4. The practical mitigation target (reducing silent mirage) is empirically tractable.

## 7.2 Open / Under-determined

1. Strong-class certainty remains limited in this 400-sample slice (`n_strong=29`).
2. Synthetic prompt/schema regularity may inflate format-learning ease; needs OOD stress tests.
3. Confidence on `flag_given_wrong` is sensitive to very small denominator when FT errors are near-zero.
4. A single canonical evaluator/report path should be enforced to prevent future drift.

Note: the prior `eval_mirage_aware.py` pivot regex footgun was fixed to support 1–4 digit actor IDs.

## 8. Reproducibility and Validation Path

## 8.1 Integrity

- `provenance/manifest.csv`
- `provenance/SHA256SUMS.txt`
- `provenance/archive_summary.json`

## 8.2 Re-run priorities for a new researcher

1. Recompute metrics from both Qwen run CSVs.
2. Re-run balanced vs imbalanced with same eval slice and seeds.
3. Add confidence intervals to all headline metrics.
4. Add balanced eval subset (200 strong + 200 degraded) and natural subset side-by-side.
5. Add oracle decomposition (information-theoretic shift vs model error).

## 9. Research Value (Current State)

This is valuable and externally credible because it combines:

- formal structure,
- reproducible synthetic benchmarks,
- cross-model black-box evaluations,
- and a deployable mitigation behavior objective.

The program is strong enough for paper-ready framing and for research engineering hiring narratives.

## 10. Most Important Files to Read First

Core synthesis:
- `README_START_HERE.md`
- `docs/independent-review-and-value-assessment.md`
- `docs/artifact-index.md`

Mirage core:
- `projects/mirage/endogenous_context_theory/README.md`
- `projects/mirage/endogenous_context_theory/results/summary_report.md`
- `projects/mirage/endogenous_context_theory/results/artifact_correction_report.md`

Qwen run artifacts:
- `external/downloads/mirage_aware_balanced_package.tar.gz`
- `external/downloads/mirage_aware_imbalanced_package_2026-02-17.tar.gz`
- `derived/qwen_balanced_vs_imbalanced_comparison_2026_02_17.csv`

Rhun:
- `projects/rhun/README.md`
- `projects/rhun/RHUN_CONTEXT_v4.md`
- `projects/rhun/paper/main.pdf`

Lorien:
- `projects/lorien/README.md`
- `projects/lorien/docs/METHODOLOGY.md`
- `projects/lorien/paper/goal_evolution_paper.pdf`

## Appendix A: Snapshot Provenance

- `rhun` commit at collection: `4e1ddc21da55649ab363b0fd15fae4af8e5bd6ae`
- `lorien` commit at collection: `6219c291793cfd45a66b4f78bf50d6dd811f3b52`
- `mirage` directory in this snapshot has no `.git` metadata at root.

## Appendix B: Archive Scale

Current authoritative totals are in `provenance/archive_summary.json`.
