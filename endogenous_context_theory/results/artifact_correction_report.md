# Artifact-Correction Pass (Interleaved Focal Positions)

## Change Applied
- Updated focal assignment in `src/generators.py` from contiguous prefix focal events to an interleaved schedule across the full timeline.
- New behavior guarantees exactly `n_focal` focal events with non-focal events before, between, and after focal events.

## Why This Matters
The prior focal-prefix assignment made feasible-prefix dynamics degenerate for compression/margin tests (almost no non-focal prefix before focal pivots). Interleaving restores the intended endogenous prefix-capacity regime.

## Re-run Scope
- `tests.test_11_compression_mirage`
- `tests.test_13_contract_compression`
- `tests.test_14_margin_correlation`
- `tests.test_15_adaptive_compression`

## Results

### Test 11: Compression Mirage
- Behavior now matches the intended phenomenon: raw validity stays high while pivot preservation degrades with stronger compression.
- Example (`M=10`):
  - retention 0.9: raw=1.000, preservation=1.000
  - retention 0.3: raw=1.000, preservation=0.835
  - retention 0.1: raw=0.940, preservation=0.580
- Current code verdict remains `FAIL` because the hard threshold is `gap > 0.15`; observed average high-compression gap is `0.146`.

### Test 13: Contract-Guarded Compression
- `PASS`.
- Contract guard strongly improves preservation over naive compression.
- Aggregate means:
  - contract preservation: `0.990`
  - naive preservation: `0.854`

### Test 14: Margin-Mirage Correlation
- `PASS`.
- Positive point-biserial correlation recovered: `0.0231` (p=`0.606`).
- Quartile preservation rates:
  - Q1: `0.968`
  - Q2: `0.936`
  - Q3: `0.928`
  - Q4: `0.976`
- Effect size is modest in this configuration.

### Test 15: Adaptive Compression
- `FAIL` under the strict median-split `0.3/0.7` policy definition.
- Observed:
  - adaptive preservation: `0.936`
  - uniform preservation: `0.954`
  - achieved compression: both `0.300`
- Interpretation: with this exact policy and metric, adaptive policy does not beat uniform in the corrected regime.

## Output Artifacts
- Raw CSVs:
  - `results/raw/test_11_mirage_summary.csv`
  - `results/raw/test_13_contract_compression_summary.csv`
  - `results/raw/test_14_margin_quartiles.csv`
  - `results/raw/test_15_adaptive_compression_summary.csv`
- Figures:
  - `results/figures/test_11_validity_mirage.png`
  - `results/figures/test_13_contract_vs_naive.png`
  - `results/figures/test_14_margin_quartiles.png`
  - `results/figures/test_15_adaptive_vs_uniform.png`
