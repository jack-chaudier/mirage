# Endogenous Context Suite: Current Status (Post-Correction)

This file is the canonical status pointer for the corrected run configuration.

## Scope

- Artifact-correction change: interleaved focal assignment in `src/generators.py`.
- Re-run tests: 11, 13, 14, 15.
- Primary sources:
  - `results/artifact_correction_report.md`
  - `results/raw/artifact_correction_summary.csv`
  - `results/raw/test_11_mirage_summary.csv`
  - `results/raw/test_13_contract_compression_summary.csv`
  - `results/raw/test_14_margin_correlation_raw.csv`
  - `results/raw/test_15_adaptive_compression_summary.csv`

## Corrected Verdicts

- **Test 11 (Compression-Induced Validity Mirage): FAIL (threshold miss)**
  - Recovered mirage behavior (raw validity remains high while pivot preservation degrades).
  - Aggregate gap at retention `<= 0.5` across all `M`: `0.146` (threshold is `> 0.15`).
  - Example (`M=10`): retention `0.9` raw/preserve `1.000/1.000`; retention `0.3` `1.000/0.835`; retention `0.1` `0.940/0.580`.

- **Test 13 (Contract-Guarded Compression): PASS**
  - Contract mean pivot preservation: `0.990`.
  - Naive mean pivot preservation: `0.854`.
  - Mean achieved retention is matched (`0.700` vs `0.700`).

- **Test 14 (Margin-Mirage Correlation): PASS**
  - Point-biserial correlation: `0.0231` (p=`0.606`).
  - Positive direction recovered; effect size is modest.

- **Test 15 (Adaptive Compression): FAIL**
  - Adaptive pivot preservation: `0.936`.
  - Uniform-0.5 pivot preservation: `0.954`.
  - Achieved compression is matched (`0.300` vs `0.300`).

## Historical Note

- `results/summary_report.md` and `results/raw/run_all_summary.csv` include a pre-correction run where tests 11/13/14/15 appear as zero-valued failures.
- Those files are retained for provenance, but should not be used as the final status view for the corrected regime.
