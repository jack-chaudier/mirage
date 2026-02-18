# RHUN Experiments

This directory contains deterministic runner entrypoints.

## Core theorem/boundary
- `run_position_sweep.py`
- `run_kj_boundary*.py`
- `run_invariance_suite.py`

## Failure taxonomy and algorithm diagnostics
- `run_*diagnosis*.py`
- `run_beam_search_sweep.py`
- `run_vag_evaluation.py`
- `run_bvag_evaluation.py`
- `run_tp_solver_evaluation.py`

## Streaming suite (Exps 40-49)
- `run_pivot_stability.py`
- `run_oscillation_trap.py`
- `run_truncation_shadow.py`
- `run_organic_oscillation.py`
- `run_organic_confirmation.py`
- `run_scale_dependence.py`
- `run_commit_latency.py`
- `run_policy_regret.py`
- `run_smart_policies_fixed.py`
- `run_smart_policies_consistent.py`
- `run_pivot_diagnostics.py`

## Context algebra suite (Exps 50-57)
- `run_context_algebra_suite.py`
- `run_context_retention_sweep.py`
- `run_exp58_pivot_shift_witness.py`

## Outputs
- `output/` (JSON and summaries)
- `output/streaming/` (streaming-specific outputs)
