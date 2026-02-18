# Tropical Endogenous Context Semiring: Experimental Summary (Historical, Pre-Correction)

> **Status note (February 2026):** This report was generated before the interleaved-focal artifact correction pass.
> For the corrected status of Tests 11/13/14/15, use:
> - `results/current_status.md`
> - `results/artifact_correction_report.md`
> - `results/raw/artifact_correction_summary.csv`

## 1. Validation Status
- **Test 01: Tropical Exactness**: **PASS** (runtime=18.97s, key metric=violations=0; trials=8000)
- **Test 02: Associativity**: **PASS** (runtime=32.42s, key metric=violations=0; trials=16000)
- **Test 03: Monoid Subsumption**: **PASS** (runtime=8.48s, key metric=violations=0; trials=3200; multi_slot_mean=0.25)
- **Test 04: Absorbing Left Ideal**: **PASS** (runtime=8.19s, key metric=violations=0; absorbed_cases=1800; uncommitted_escape_rate_mean=0.0)
- **Test 05: Holographic Exactness**: **PASS** (runtime=40.99s, key metric=violations=0; trials=1200)
- **Test 06: Incremental Consistency**: **PASS** (runtime=8.09s, key metric=violations=0; checks=100000)
- **Test 07: Scaling Performance**: **PASS** (runtime=2.59s, key metric=max_depth=17.0)
- **Test 08: Divergence Characterization**: **PASS** (runtime=105.84s, key metric=b=1.0384723725933305; b_ci_low=1.0227306037971844; b_ci_high=1.0542141413894766)
- **Test 09: Record-Process Statistics**: **PASS** (runtime=105.84s, key metric=mae_to_harmonic_focal=0.2850620560434516)
- **Test 10: Tropical Semiring as Divergence Shield**: **PASS** (runtime=32.74s, key metric=tropical_dominates_committed=True)
- **Test 11: Compression-Induced Validity Mirage**: **FAIL** (runtime=9.02s, key metric=mirage_gap=0.0)
- **Test 12: Deterministic Mirage Witness**: **PASS** (runtime=0.05s, key metric=regret_m10=0.35)
- **Test 13: Contract-Guarded Compression**: **FAIL** (runtime=182.71s, key metric=contract_preservation_mean=0.0; naive_preservation_mean=0.0)
- **Test 14: Margin-Mirage Correlation**: **FAIL** (runtime=2.43s, key metric=correlation=0.0; p_value=1.0)
- **Test 15: Margin-Guided Adaptive Compression**: **FAIL** (runtime=9.18s, key metric=adaptive_preservation=0.0; uniform_preservation=0.0)
- **Test 16: Organic Oscillation Traps**: **PASS** (runtime=71.08s, key metric=false_negatives=0; mean_trap_rate=0.0)
- **Test 17: Tropical Streaming vs Committed**: **PASS** (runtime=71.44s, key metric=mismatches=0; committed_mean=0.0; tropical_mean=0.0)
- Exactness violations (Test 1): **0**
- Associativity violations (Test 2): **0**

## 2. Validated vs Falsified Claims
- Core algebraic soundness claims are validated in this run (exactness, associativity, tree consistency, streaming equivalence).
- Falsified tests in this run:
  - Test 11: Compression-Induced Validity Mirage
  - Test 13: Contract-Guarded Compression
  - Test 14: Margin-Mirage Correlation
  - Test 15: Margin-Guided Adaptive Compression

## 3. Surprising Findings
- Mean fraction of instances with multiple occupied tropical slots (Test 3): **0.250**.
- Mean uncommitted escape rate from absorbed prefixes (Test 4): **0.000**.

## 4. Divergence Characterization
- Estimated power-law exponent: **b = 1.038** (95% CI [1.023, 1.054]).
- Interpretation: super-linear reassignment growth supports thermodynamic divergence concerns for irrevocable commitment.

## 5. Mirage Severity and Contract Fix
- Mean mirage gap at retention <= 0.5: **0.000** (raw validity - preservation).
- Mean semantic regret at retention <= 0.5: **0.000**.
- Contract guard pivot preservation: **0.000** vs naive **0.000**.
- Contract guard achieved retention: **0.700** vs naive **0.700** (higher means less aggressive compression).

## 6. Margin Hypothesis
- Margin-preservation correlation (Pearson proxy): **undefined** (constant-input regime).
- Quartile preservation rates: Q_all=0.000.

## 7. Overall Assessment
The current empirical suite supports the Tropical Endogenous Context Semiring as a sound computational foundation for endogenous pivot tracking, with exact composition and tree compatibility under tested regimes.
