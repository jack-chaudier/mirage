# RHUN Agent Workflow

This file captures the working loop used in this repository so far.

## Scope
- Work only in `~/rhun`.
- Do not modify `~/lorien` from RHUN tasks.
- Keep all runs deterministic (fixed seeds, no network calls, no LLM-in-loop experiments).

## Standard Execution Loop
1. Read context first:
   - `RHUN_CONTEXT.md`
2. Pre-check before coding:
   - Inspect relevant extraction/generator/theory modules.
   - Confirm current interfaces, constraints, and scoring semantics from code (not assumptions).
3. Implement minimal targeted changes:
   - Preserve existing experiment baselines for comparability.
   - Prefer additive changes (new modules/scripts) over invasive rewrites.
4. Validate quickly:
   - `python -m py_compile ...` for touched Python files.
   - `pytest -q` (all tests must pass).
5. Run the requested experiment/verification script.
6. Save outputs under `experiments/output/`:
   - `<name>.json`
   - `<name>_summary.md`
7. Report:
   - Paste full stdout for experiment runs.
   - Call out key aggregates and edge-case diagnostics.
8. Commit with the exact user-requested message format.

## Experiment Design Norms
- Deterministic seeds; explicit seed ranges in output metadata.
- Verification-first: check expected baselines before interpreting new conditions.
- Error decomposition over aggregate accuracy:
  - TP / FP / FN / TN where applicable.
- Keep null results: failed hypotheses are first-class findings.

## Implementation Conventions
- Python 3.11+, dataclasses, type hints, pathlib.
- Reuse existing infra:
  - `rhun/experiments/runner.py` for output/metadata saving.
  - Existing grammar/validator/scoring utilities for consistency.
- Do not silently change semantics of existing core functions unless explicitly requested.

## Current Research-State Notes
- The theorem is a sufficient-condition result with zero-FP behavior in tested regimes.
- Secondary failure modes are now taxonomy-driven; new methods should report which layer they address.
- Heuristic baselines (including `oracle_extract`) are not assumed globally optimal unless proven.
