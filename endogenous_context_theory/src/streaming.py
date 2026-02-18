"""Streaming policies and trap/cost analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .tropical_semiring import Event, NEG_INF, build_tropical_context


@dataclass
class RecordProcess:
    record_positions: List[int]
    record_weights: List[float]
    gaps_before_record: List[int]
    inter_record_gaps: List[int]
    min_gap: float
    blast_radii: List[int]
    total_cost: int


@dataclass
class StreamingOutcome:
    committed_valid: bool
    tropical_valid: bool
    finite_valid: bool
    trap: bool
    min_gap: float
    total_cost: int
    n_records: int


def running_max_record_process(events: Sequence[Event]) -> RecordProcess:
    """Track running-max record process over focal events."""

    current_max = NEG_INF
    dev_since_record = 0

    record_positions: List[int] = []
    record_weights: List[float] = []
    gaps_before_record: List[int] = []

    for idx, event in enumerate(events):
        if event.is_focal and event.weight > current_max:
            current_max = event.weight
            record_positions.append(idx)
            record_weights.append(event.weight)
            gaps_before_record.append(dev_since_record)
            dev_since_record = 0
        elif not event.is_focal:
            dev_since_record += 1

    if len(gaps_before_record) <= 1:
        inter_record_gaps: List[int] = []
    else:
        inter_record_gaps = gaps_before_record[1:]

    min_gap = float(min(gaps_before_record)) if gaps_before_record else float("inf")

    blast_radii: List[int] = []
    if len(record_positions) > 1:
        for pos in record_positions[1:]:
            # Shift invalidates all labels committed before this position.
            blast_radii.append(pos)

    total_cost = int(sum(blast_radii))
    return RecordProcess(
        record_positions=record_positions,
        record_weights=record_weights,
        gaps_before_record=gaps_before_record,
        inter_record_gaps=inter_record_gaps,
        min_gap=min_gap,
        blast_radii=blast_radii,
        total_cost=total_cost,
    )


def committed_validity(events: Sequence[Event], k: int) -> bool:
    """Validity under immediate irrevocable commitment semantics.

    The final committed pivot is the final running-max record; only its local prefix
    gap survives commitment shifts.
    """

    rp = running_max_record_process(events)
    if not rp.gaps_before_record:
        return False
    final_gap = rp.gaps_before_record[-1]
    return final_gap >= k


def finite_offline_validity(events: Sequence[Event], k: int) -> bool:
    return build_tropical_context(events, k).feasible()


def tropical_streaming_validity(events: Sequence[Event], k: int) -> bool:
    # Identical to finite validity because tropical state never commits.
    return build_tropical_context(events, k).feasible()


def streaming_outcome(events: Sequence[Event], k: int) -> StreamingOutcome:
    rp = running_max_record_process(events)
    committed = committed_validity(events, k)
    tropical = tropical_streaming_validity(events, k)
    finite = finite_offline_validity(events, k)
    trap = finite and not committed
    return StreamingOutcome(
        committed_valid=committed,
        tropical_valid=tropical,
        finite_valid=finite,
        trap=trap,
        min_gap=rp.min_gap,
        total_cost=rp.total_cost,
        n_records=len(rp.record_positions),
    )


def _prefix_nonfocal(events: Sequence[Event]) -> np.ndarray:
    prefix = np.zeros(len(events) + 1, dtype=int)
    for i, e in enumerate(events):
        prefix[i + 1] = prefix[i] + (0 if e.is_focal else 1)
    return prefix


def _nonfocal_between(prefix: np.ndarray, start_inclusive: int, end_inclusive: int) -> int:
    if end_inclusive < start_inclusive:
        return 0
    return int(prefix[end_inclusive + 1] - prefix[start_inclusive])


def deferred_commitment_policy(
    events: Sequence[Event],
    k: int,
    commit_fraction: float,
) -> Dict[str, float]:
    """Deferred commitment by committing only every ~1/f record updates."""

    rp = running_max_record_process(events)
    records = rp.record_positions
    n_records = len(records)

    if n_records == 0:
        return {
            "valid": 0.0,
            "total_cost": 0.0,
            "n_commits": 0.0,
            "min_gap": float("inf"),
        }

    step = max(1, int(round(1.0 / commit_fraction)))
    selected = list(range(0, n_records, step))
    if selected[-1] != n_records - 1:
        selected.append(n_records - 1)

    prefix = _prefix_nonfocal(events)

    gaps: List[int] = []
    for j, rec_idx in enumerate(selected):
        pos = records[rec_idx]
        if j == 0:
            prev_pos = -1
        else:
            prev_pos = records[selected[j - 1]]
        gaps.append(_nonfocal_between(prefix, prev_pos + 1, pos - 1))

    min_gap = float(min(gaps)) if gaps else float("inf")
    final_gap = gaps[-1] if gaps else 0
    valid = final_gap >= k

    blast_radii = [records[idx] for idx in selected[1:]]
    total_cost = float(sum(blast_radii))

    return {
        "valid": float(valid),
        "total_cost": total_cost,
        "n_commits": float(len(selected)),
        "min_gap": min_gap,
    }


def min_gap_predicts_trap(events: Sequence[Event], k: int) -> bool:
    rp = running_max_record_process(events)
    return rp.min_gap < k
