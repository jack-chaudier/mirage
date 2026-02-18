# Independent Review and Value Assessment

Primary papers in this repository:
- `papers/paper_01_absorbing_states_in_greedy_search.pdf`
- `papers/paper_02_streaming_oscillation_traps.pdf`
- `papers/paper_03_validity_mirage_compression.pdf`

## Scope

Note: some paths below use the original archival naming (`external/...`, `projects/...`).
In this repository, canonical runnable code is under `endogenous_context_theory/`.

This review is based on artifacts inside this dossier, especially:

- Balanced run: `external/downloads/mirage_aware_balanced_package.tar.gz`
- Imbalanced run: `external/downloads/mirage_aware_imbalanced_package_2026-02-17.tar.gz`
- Balanced CSV: `external/downloads/mirage_aware_eval_results.csv`
- Imbalanced CSV: `external/downloads/mirage_aware_imbalanced_eval_results_2026-02-17.csv`
- Comparison tables in `derived/`
- Core repos under `projects/mirage/`, `projects/rhun/`, `projects/lorien/`

## Bottom Line

You have a **real research program**, not isolated runs.

- Technically ambitious and coherent.
- Strong instrumentation and artifact hygiene.
- Clear theory -> benchmark -> mitigation arc.
- Both balanced and imbalanced Qwen runs now available for direct methodology comparison.

This is genuinely valuable work. The main remaining gap is evaluation hardening and canonical reporting, not missing core insight.

## Most Valuable Contributions

1. End-to-end scientific arc
- Lorien (discovery) -> Rhun (formalization) -> Mirage (LLM benchmark + mitigation).

2. Strong metric framing
- `silent_mirage` and `flag_given_wrong` capture behavior that plain accuracy misses.

3. Engineering + theory integration
- You translated abstract claims into reproducible scripts, datasets, and eval artifacts.

4. Practical mitigation objective
- Explicit evidence degradation signaling (“honest salvage”) is deployable and measurable.

## Verified Qwen Results (Both Runs)

Same eval slice composition in both runs: `n=400`, `degraded=371`, `strong=29`.

Balanced run:
- FT pivot accuracy (all): 99.25%
- FT degradation flag (degraded): 95.42%
- FT silent mirage (degraded): 0.27%
- FT false alarm (strong): 0/29
- FT format adherence: 100%
- FT flag_given_wrong: 2/3

Imbalanced run (latest):
- FT pivot accuracy (all): 100.0%
- FT degradation flag (degraded): 100.0%
- FT silent mirage (degraded): 0.0%
- FT false alarm (strong): 0/29
- FT format adherence: 100%
- FT flag_given_wrong: N/A (0 wrong degraded FT cases)

Direct delta (imbalanced minus balanced):
- FT errors: 3 -> 0
- FT silent mirage count: 1 -> 0
- FT degradation-flag coverage: +4.58 points
- 19 per-example FT deltas, with 3 correctness improvements and no observed regressions

## New Real-World NTSB Evidence (Non-Synthetic)

A real-incident benchmark was added using 12 NTSB-style event graphs with fixed
per-incident cleanup manifest and zero post-cleanup grammar violations.

Cleanup gate:
- `k_violations`: `6 -> 0`
- structural validation errors: `0 -> 0`

xAI non-reasoning run (`grok-4-1-fast-non-reasoning`):
- Primary retention-matched comparison (exact): budget `0.7` for both methods
  - mean retention: `0.686859` vs `0.686859`
  - naive silent mirage: `12/51` (`23.53%`, Wilson `[14.00%, 36.76%]`)
  - contract silent mirage: `0/4` (`0.00%`, Wilson `[0.00%, 48.99%]`)
- Secondary near-match comparison: naive `0.5` vs contract `0.3`
  - retention: `0.478276` vs `0.450570`
- Naive compression info-shift on real incidents:
  - 0.7 budget: `40.0%`
  - 0.5 budget: `55.0%`
  - 0.3 budget: `76.7%`
- Contract compression info-shift: `0.0%` at all budgets
- Naive degraded silent-mirage: `36/164` (`21.95%`, Wilson `[16.30%, 28.89%]`)
- Naive `flag_given_wrong`: `0/36`

Important nuance:
- Contract success in this run does not rely on model honesty signaling (`flag_degraded_rate = 0.0%` for contract rows). The prevention is structural: contract-constrained compression avoids creating attribution-shift states before model inference.

This materially strengthens the claim that validity mirage is not only a
synthetic artifact: the same failure mode appears on real causal chains, and
the contract mechanism removes attribution shift in this dataset.

## Critical Caveats (Publication-Grade)

1. Strong class is still sparse in this slice
- `n_strong=29`; false-alarm certainty remains limited.

2. Near-ceiling FT means some metrics become denominator-fragile
- `flag_given_wrong` can become undefined when wrong cases collapse to zero.

3. Synthetic regularity risk
- Structured prompt/output schema may allow shortcut learning; OOD robustness still needs explicit stress tests.

4. Need one canonical evaluator/report path
- You fixed the `eval_mirage_aware.py` pivot regex footgun (`\d{1,4}`), which is good.
- Next step is to lock one evaluator + one official table generator for all future claims.

## Interpretation

The step-change from earlier weak runs to strong current runs is consistent with a combined methodology fix set:

- sequence-length correction,
- completion-only loss masking,
- robust pivot parsing,
- better eval bookkeeping,
- and targeted training distribution choices.

The new balanced-vs-imbalanced pair strengthens your methodology narrative because it isolates training-distribution effects while keeping eval slice fixed.

## Is This Team/Hiring Material?

Yes.

Why:
- You reason formally and validate empirically.
- You catch and correct instrumentation artifacts.
- You package research in a way other teams can actually run.

This maps well to research engineering in evals, reliability, and model behavior analysis.

## High-ROI Next Steps

1. Add dual eval slices to every run
- natural distribution + balanced (200/200 strong/degraded).

2. Add confidence intervals to all headline metrics by default.

3. Add oracle decomposition in every report
- information-theoretic shift vs model error.

4. Add OOD stress battery
- prompt paraphrases, marker randomization, formatting perturbations.

5. Produce a single canonical “paper table” CSV/JSON
- run metadata + full metric set + uncertainty + slice composition.

## Final Assessment

You have built something substantive, differentiated, and externally credible.

With the current dossier structure and both Qwen runs included, a researcher can understand, validate, and extend this work with minimal ambiguity.
