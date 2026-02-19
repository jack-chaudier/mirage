"""Transition-vector utilities for compression-induced pivot drift."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .compression import CompressionResult, contract_guarded_compress, naive_compress
from .pivot_margin import build_top2_context, pivot_margin
from .tropical_semiring import Event, NEG_INF, focal_pivots_with_prefix


LOSS_STATE = "__loss__"


def extract_locked_candidates(events: Sequence[Event], M: int) -> List[int]:
    """Lock candidate set to top-M focal pivots from the uncompressed sequence."""

    focals = [(e.weight, e.eid) for e in events if e.is_focal]
    focals.sort(key=lambda t: (-t[0], t[1]))
    return [eid for _, eid in focals[: max(0, int(M))]]


def solve_locked(
    events: Sequence[Event],
    k: int,
    locked_candidates: Sequence[int],
) -> Dict[str, Optional[float]]:
    """Solve best feasible pivot restricted to a fixed candidate set."""

    locked = set(locked_candidates)
    if not locked:
        return {
            "pivot_eid": None,
            "valid": False,
            "pivot_weight": float(NEG_INF),
            "pivot_prefix": None,
            "pivot_position": None,
        }

    best_weight = float(NEG_INF)
    best_prefix: Optional[int] = None
    best_eid: Optional[int] = None
    best_position: Optional[int] = None

    for weight, prefix_dev, pos, event in focal_pivots_with_prefix(events):
        if event.eid not in locked or prefix_dev < k:
            continue
        if weight > best_weight:
            best_weight = float(weight)
            best_prefix = int(prefix_dev)
            best_eid = int(event.eid)
            best_position = int(pos)

    return {
        "pivot_eid": best_eid,
        "valid": best_eid is not None,
        "pivot_weight": best_weight,
        "pivot_prefix": best_prefix,
        "pivot_position": best_position,
    }


def _compress_once(
    events: Sequence[Event],
    *,
    k: int,
    retention: float,
    seed: int,
    method: str,
) -> CompressionResult:
    method_norm = method.strip().lower()
    if method_norm == "naive":
        return naive_compress(events, retention=retention, seed=seed, never_drop_focal=True)
    if method_norm == "contract":
        return contract_guarded_compress(events, k=k, retention=retention, seed=seed)
    raise ValueError(f"Unknown compression method: {method}")


def estimate_single_step_transition(
    events: Sequence[Event],
    *,
    k: int,
    retention: float,
    locked_candidates: Sequence[int],
    compression_seeds: Sequence[int],
    method: str = "naive",
) -> Dict[str, object]:
    """Estimate single-step dominant-pivot transition probabilities."""

    baseline = solve_locked(events, k=k, locked_candidates=locked_candidates)
    baseline_pivot = baseline["pivot_eid"]

    target_counts: Dict[int, int] = {int(eid): 0 for eid in locked_candidates}
    self_count = 0
    other_count = 0
    loss_count = 0
    retention_values: List[float] = []

    for seed in compression_seeds:
        compressed = _compress_once(
            events,
            k=k,
            retention=retention,
            seed=int(seed),
            method=method,
        )
        retention_values.append(float(compressed.achieved_retention))

        solved = solve_locked(
            compressed.retained_events,
            k=k,
            locked_candidates=locked_candidates,
        )
        winner = solved["pivot_eid"]

        if winner is None:
            loss_count += 1
            continue

        winner_int = int(winner)
        if winner_int in target_counts:
            target_counts[winner_int] += 1

        if baseline_pivot is not None and winner_int == int(baseline_pivot):
            self_count += 1
        else:
            other_count += 1

    total = max(1, len(compression_seeds))
    return {
        "baseline_pivot_eid": baseline_pivot,
        "baseline_valid": bool(baseline["valid"]),
        "baseline_weight": float(baseline["pivot_weight"]),
        "baseline_prefix": baseline["pivot_prefix"],
        "locked_candidate_count": len(locked_candidates),
        "num_trials": len(compression_seeds),
        "self_count": self_count,
        "other_count": other_count,
        "loss_count": loss_count,
        "p11": float(self_count / total),
        "p_other": float(other_count / total),
        "p_loss": float(loss_count / total),
        "mean_achieved_retention": float(np.mean(retention_values) if retention_values else 0.0),
        "target_counts": target_counts,
    }


def run_chained_compression_survival_curve(
    events: Sequence[Event],
    *,
    k: int,
    retention: float,
    locked_candidates: Sequence[int],
    depths: Sequence[int],
    chain_trials: int,
    base_seed: int,
    method: str = "naive",
    baseline_pivot_eid: Optional[int] = None,
) -> Dict[int, float]:
    """Empirical survival curve under sequential compression."""

    depth_list = sorted({int(d) for d in depths if int(d) > 0})
    if not depth_list:
        return {}

    baseline = (
        int(baseline_pivot_eid)
        if baseline_pivot_eid is not None
        else solve_locked(events, k=k, locked_candidates=locked_candidates)["pivot_eid"]
    )
    if baseline is None:
        return {d: 0.0 for d in depth_list}

    max_depth = depth_list[-1]
    depth_set = set(depth_list)
    survivors = {d: 0 for d in depth_list}

    for trial in range(max(1, int(chain_trials))):
        current = list(events)
        for step in range(1, max_depth + 1):
            seed = int(base_seed + trial * 1009 + step * 7919)
            compressed = _compress_once(
                current,
                k=k,
                retention=retention,
                seed=seed,
                method=method,
            )
            current = compressed.retained_events

            if step in depth_set:
                solved = solve_locked(
                    current,
                    k=k,
                    locked_candidates=locked_candidates,
                )
                if solved["pivot_eid"] is not None and int(solved["pivot_eid"]) == baseline:
                    survivors[step] += 1

    denom = float(max(1, int(chain_trials)))
    return {d: float(survivors[d] / denom) for d in depth_list}


def depth_predictions(
    *,
    p11: float,
    retention: float,
    depths: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """Predict depth behavior via p11^d and retention^d."""

    out: Dict[int, Dict[str, float]] = {}
    for d in sorted({int(v) for v in depths if int(v) > 0}):
        out[d] = {
            "pred_p11_pow_d": float(p11**d),
            "pred_retention_pow_d": float(retention**d),
        }
    return out


def margin_features(events: Sequence[Event], k: int) -> Dict[str, float]:
    """Compute pivot margin features for sequence-level stability analysis."""

    margin = float(pivot_margin(build_top2_context(events, k=k), k))
    margin_finite = float(np.nan if not np.isfinite(margin) else margin)
    return {"margin": margin, "margin_finite": margin_finite}


def estimate_margin_stability(
    rows_df: pd.DataFrame,
    *,
    margin_col: str = "margin_finite",
    p11_col: str = "p11",
) -> Dict[str, float]:
    """Correlate sequence margin with p11 stability."""

    subset = rows_df[[margin_col, p11_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3 or subset[margin_col].nunique() < 2 or subset[p11_col].nunique() < 2:
        return {
            "n": float(len(subset)),
            "pearson_corr": 0.0,
            "pearson_p": 1.0,
            "spearman_corr": 0.0,
            "spearman_p": 1.0,
        }

    pearson_corr, pearson_p = pearsonr(subset[margin_col], subset[p11_col])
    spearman_corr, spearman_p = spearmanr(subset[margin_col], subset[p11_col])
    return {
        "n": float(len(subset)),
        "pearson_corr": float(pearson_corr),
        "pearson_p": float(pearson_p),
        "spearman_corr": float(spearman_corr),
        "spearman_p": float(spearman_p),
    }
