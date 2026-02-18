"""Compression strategies and mirage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tropical_semiring import (
    Event,
    NEG_INF,
    TropicalContext,
    best_feasible_pivot,
    build_tropical_context,
    compose_tropical,
)


@dataclass
class SolveResult:
    valid: bool
    pivot_weight: float
    pivot_prefix: Optional[int]
    pivot_eid: Optional[int]


@dataclass
class CompressionResult:
    retained_events: List[Event]
    removed_count: int
    blocked_count: int
    achieved_retention: float


def solve_with_budget(events: Sequence[Event], k: int, M: Optional[int] = None) -> SolveResult:
    """Solve for best feasible pivot, optionally restricted to top-M by weight."""

    if M is None:
        ctx = build_tropical_context(events, k)
        w = float(ctx.W[k])
        return SolveResult(valid=w > NEG_INF, pivot_weight=w, pivot_prefix=None, pivot_eid=None)

    w, prefix, eid = best_feasible_pivot(events, k, top_m=M)
    return SolveResult(valid=w > NEG_INF, pivot_weight=w, pivot_prefix=prefix, pivot_eid=eid)


def naive_compress(
    events: Sequence[Event],
    retention: float,
    seed: int,
    never_drop_focal: bool = True,
) -> CompressionResult:
    """Randomly remove events to match retention; keeps all focal events by default."""

    rng = np.random.default_rng(seed)

    retained: List[Event] = []
    removable_idx = [i for i, e in enumerate(events) if (not never_drop_focal or not e.is_focal)]
    keep_mask = np.ones(len(events), dtype=bool)

    target_remove = int(round((1.0 - retention) * len(removable_idx)))
    target_remove = max(0, min(target_remove, len(removable_idx)))
    if target_remove > 0:
        remove_idx = set(rng.choice(removable_idx, size=target_remove, replace=False).tolist())
        for i in remove_idx:
            keep_mask[i] = False

    for i, e in enumerate(events):
        if keep_mask[i]:
            retained.append(e)

    achieved_retention = len(retained) / max(1, len(events))
    return CompressionResult(
        retained_events=retained,
        removed_count=len(events) - len(retained),
        blocked_count=0,
        achieved_retention=achieved_retention,
    )


def contract_guarded_compress(
    events: Sequence[Event],
    k: int,
    retention: float,
    seed: int,
) -> CompressionResult:
    """Compression with no-absorption contract on feasible slots (x >= k)."""

    rng = np.random.default_rng(seed)
    current = list(events)
    current_ctx = build_tropical_context(current, k)

    removable_eids = [e.eid for e in events if not e.is_focal]
    rng.shuffle(removable_eids)

    target_remove = int(round((1.0 - retention) * len(removable_eids)))
    target_remove = max(0, min(target_remove, len(removable_eids)))
    if target_remove == 0:
        achieved_retention = len(current) / max(1, len(events))
        return CompressionResult(
            retained_events=current,
            removed_count=0,
            blocked_count=0,
            achieved_retention=achieved_retention,
        )

    removed = 0
    blocked = 0
    ptr = 0

    while removed < target_remove and ptr < len(removable_eids):
        # Recompute prefix/suffix contexts for current sequence once per accepted removal.
        prefix: List[TropicalContext] = [TropicalContext.empty(k)]
        for e in current:
            prefix.append(compose_tropical(prefix[-1], TropicalContext.from_event(e, k)))

        suffix: List[TropicalContext] = [TropicalContext.empty(k) for _ in range(len(current) + 1)]
        for i in range(len(current) - 1, -1, -1):
            suffix[i] = compose_tropical(TropicalContext.from_event(current[i], k), suffix[i + 1])

        eid_to_idx = {e.eid: i for i, e in enumerate(current)}
        accepted = False

        while ptr < len(removable_eids):
            eid = removable_eids[ptr]
            ptr += 1
            idx = eid_to_idx.get(eid)
            if idx is None:
                continue

            trial_ctx = compose_tropical(prefix[idx], suffix[idx + 1])

            # Vector is capped at k; x >= k is exactly x == k.
            if trial_ctx.W[k] + 1e-12 < current_ctx.W[k]:
                blocked += 1
                continue

            del current[idx]
            current_ctx = trial_ctx
            removed += 1
            accepted = True
            break

        if not accepted:
            break

    achieved_retention = len(current) / max(1, len(events))
    return CompressionResult(
        retained_events=current,
        removed_count=removed,
        blocked_count=blocked,
        achieved_retention=achieved_retention,
    )


def semantic_regret(full_weight: float, compressed_weight: float) -> float:
    if full_weight <= 0 or np.isneginf(full_weight):
        return 0.0
    if np.isneginf(compressed_weight):
        return 1.0
    return float(max(0.0, 1.0 - (compressed_weight / full_weight)))


def evaluate_mirage_cell(
    events: Sequence[Event],
    k: int,
    retention: float,
    seed: int,
    M: int,
) -> Dict[str, float]:
    full = solve_with_budget(events, k, M=None)
    compressed = naive_compress(events, retention=retention, seed=seed, never_drop_focal=True)
    comp = solve_with_budget(compressed.retained_events, k, M=M)

    raw_valid = float(comp.valid)
    pivot_pres = float(
        full.valid
        and comp.valid
        and (abs(comp.pivot_weight - full.pivot_weight) <= 1e-12)
    )
    regret = semantic_regret(full.pivot_weight, comp.pivot_weight)

    return {
        "raw_validity": raw_valid,
        "pivot_preservation": pivot_pres,
        "semantic_regret": regret,
        "full_weight": full.pivot_weight,
        "compressed_weight": comp.pivot_weight,
        "achieved_retention": compressed.achieved_retention,
    }


def deterministic_mirage_witness(k: int = 3) -> Tuple[List[Event], List[Event], List[int]]:
    """Construct a deterministic witness with >30% degraded feasible substitute."""

    # Dominant pivot (weight=20) depends on three specific non-focal events.
    full = [
        Event(0, 0.00, 1.0, "dev", False),
        Event(1, 0.05, 1.2, "dev", False),
        Event(2, 0.10, 4.0, "focal", True),
        Event(3, 0.15, 1.1, "dev", False),
        Event(4, 0.20, 20.0, "focal", True),  # Dominant full pivot, exactly k-prefix.
        Event(5, 0.25, 1.0, "dev", False),
        Event(6, 0.30, 1.0, "dev", False),
        Event(7, 0.35, 1.0, "dev", False),
        Event(8, 0.40, 13.0, "focal", True),  # Weaker substitute pivot.
        Event(9, 0.45, 1.0, "dev", False),
        Event(10, 0.50, 1.0, "dev", False),
    ]

    remove_eids = [0, 1, 3]
    compressed = [e for e in full if e.eid not in set(remove_eids)]
    return full, compressed, remove_eids
