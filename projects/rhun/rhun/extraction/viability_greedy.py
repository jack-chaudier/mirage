"""Viability-aware greedy extraction.

Greedy extraction with a forward viability filter:
- Preserve the same pool construction used by `search.greedy_extract`.
- For each anchor branch, include only events whose addition keeps
  the sequence span-viable under remaining-slot bounds.
- Optional gap-aware filtering checks local bridge existence.
- Optional budget-aware filtering enforces bridge-slot affordability:
  B_lb(S) <= remaining_slots.
- Reuse analytic viability checks (length/coverage/TP-lock) from
  `rhun.theory.viability.compute_viability`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
from typing import Callable

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_phases
from rhun.extraction.pool_construction import bfs_pool, filtered_injection_pool, injection_pool
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase
from rhun.theory.theorem import diagnose_absorption
from rhun.theory.viability import compute_viability


def _pool_builder(strategy: str):
    if strategy == "bfs":
        return bfs_pool
    if strategy == "injection":
        return injection_pool
    if strategy == "filtered_injection":
        return filtered_injection_pool
    raise ValueError(f"Unknown pool_strategy: {strategy}")


def _event_lookup(graph: CausalGraph) -> dict[str, Event]:
    return {event.id: event for event in graph.events}


def _timespan(events: list[Event] | tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    timestamps = [float(event.timestamp) for event in events]
    return max(timestamps) - min(timestamps)


def _sorted_by_weight(events: list[Event]) -> list[Event]:
    return sorted(
        events,
        key=lambda event: (event.weight, -event.timestamp, event.id),
        reverse=True,
    )


def _empty_sequence(focal_actor: str, metadata: dict | None = None) -> ExtractedSequence:
    return ExtractedSequence(
        events=(),
        phases=(),
        focal_actor=focal_actor,
        score=0.0,
        valid=False,
        violations=("empty_candidate_set",),
        metadata=metadata or {},
    )


def _candidate_to_scored_sequence(
    graph: CausalGraph,
    candidate_events: tuple[Event, ...],
    focal_actor: str,
    grammar: GrammarConfig,
    scoring_fn: Callable[[ExtractedSequence], float],
    metadata: dict,
) -> ExtractedSequence:
    phases = classify_phases(candidate_events, min_development=grammar.min_prefix_elements)
    candidate = ExtractedSequence(
        events=candidate_events,
        phases=phases,
        focal_actor=focal_actor,
    )
    valid, violations = validate(candidate, grammar, graph)
    absorption = diagnose_absorption(candidate, grammar)
    score = scoring_fn(candidate)

    merged = dict(metadata)
    merged.update(
        {
            "absorbed": absorption["absorbed"],
            "absorption_step": absorption["absorption_step"],
            "events_after_absorption": absorption["events_after_absorption"],
        }
    )
    return replace(
        candidate,
        score=score,
        valid=valid,
        violations=tuple(violations),
        metadata=merged,
    )


def _max_achievable_span(
    selected_events: list[Event],
    remaining_events: list[Event],
    remaining_slots: int,
) -> float:
    current_span = _timespan(selected_events)
    if not selected_events:
        return 0.0
    if remaining_slots <= 0 or not remaining_events:
        return current_span

    current_min = min(float(event.timestamp) for event in selected_events)
    current_max = max(float(event.timestamp) for event in selected_events)
    future_min = min(float(event.timestamp) for event in remaining_events)
    future_max = max(float(event.timestamp) for event in remaining_events)

    if remaining_slots >= 2 and len(remaining_events) >= 2:
        return max(current_max, future_max) - min(current_min, future_min)

    extend_right = max(current_max, future_max) - current_min
    extend_left = current_max - min(current_min, future_min)
    return max(extend_right, extend_left)


def _max_possible_coverage(
    selected_events: list[Event],
    remaining_events: list[Event],
    focal_actor: str,
    min_length: int,
    remaining_slots: int,
) -> float:
    n = len(selected_events)
    focal_selected = sum(1 for event in selected_events if focal_actor in event.actors)
    remaining_focal = sum(1 for event in remaining_events if focal_actor in event.actors)
    remaining_non_focal = len(remaining_events) - remaining_focal

    best = 0.0
    max_focal_add = min(remaining_focal, remaining_slots)
    for focal_add in range(max_focal_add + 1):
        non_focal_min = max(0, min_length - (n + focal_add))
        non_focal_max = min(remaining_non_focal, remaining_slots - focal_add)
        if non_focal_min > non_focal_max:
            continue

        # To maximize coverage for fixed focal_add, use minimal non-focal additions.
        non_focal_add = non_focal_min
        total_len = n + focal_add + non_focal_add
        if total_len <= 0:
            continue
        coverage = (focal_selected + focal_add) / total_len
        if coverage > best:
            best = coverage

    return best


def _tp_locked_after_addition(
    tentative_events: list[Event],
    remaining_events: list[Event],
    grammar: GrammarConfig,
) -> bool:
    phases = classify_phases(tentative_events, min_development=grammar.min_prefix_elements)
    tp_idx = next((idx for idx, phase in enumerate(phases) if phase == Phase.TURNING_POINT), None)
    if tp_idx is None:
        return False

    dev_before_tp = sum(1 for phase in phases[:tp_idx] if phase == Phase.DEVELOPMENT)
    if dev_before_tp >= grammar.min_prefix_elements:
        return False

    tp_event = tentative_events[tp_idx]
    heavier_remaining = any(event.weight > tp_event.weight for event in remaining_events)
    return not heavier_remaining


def _theory_viability_rejection_reason(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    tentative_events: list[Event],
) -> str | None:
    viability = compute_viability(
        graph=graph,
        focal_actor=focal_actor,
        grammar=grammar,
        partial_sequence=tentative_events,
        exhaustive_limit=0,
    )
    if viability["viable"]:
        return None

    reason = str(viability["reason"])
    if reason.startswith("tp_locked_insufficient_development"):
        return reason
    if reason.startswith("insufficient_remaining_length"):
        return reason
    if reason.startswith("too_long_partial"):
        return reason
    if reason.startswith("coverage_upper_bound"):
        return reason
    return None


def _gap_bridge_viable(
    grammar: GrammarConfig,
    tentative_events: list[Event],
    remaining_events: list[Event],
    remaining_slots: int,
) -> tuple[bool, str]:
    if math.isinf(grammar.max_temporal_gap):
        return True, "gap_constraint_disabled"
    if len(tentative_events) < 2:
        return True, "gap_constraint_trivial"

    max_gap = float(grammar.max_temporal_gap)
    oversized: list[tuple[float, float]] = []
    for left, right in zip(tentative_events[:-1], tentative_events[1:]):
        gap = float(right.timestamp - left.timestamp)
        if gap > max_gap + 1e-12:
            oversized.append((float(left.timestamp), float(right.timestamp)))

    if not oversized:
        return True, "gap_constraint_satisfied"

    if remaining_slots <= 0:
        return False, "gap_bridge_no_slots_remaining"
    if len(oversized) > remaining_slots:
        return False, "gap_bridge_slots_exhausted"

    for start_ts, end_ts in oversized:
        bridge_exists = any(
            (start_ts + 1e-12) < float(event.timestamp) < (end_ts - 1e-12)
            for event in remaining_events
        )
        if not bridge_exists:
            return False, "gap_bridge_unavailable"
    return True, "gap_bridge_available"


def _compute_bridge_budget(timestamps: list[float], gap_threshold: float) -> int:
    """
    Lower bound B_lb on bridges required to make adjacent gaps <= gap_threshold.

    B_lb(S) = sum_i max(0, ceil((t_{i+1}-t_i)/G) - 1)
    """
    if len(timestamps) < 2 or math.isinf(gap_threshold):
        return 0

    budget = 0
    for left, right in zip(timestamps[:-1], timestamps[1:]):
        gap = float(right - left)
        if gap <= gap_threshold + 1e-12:
            continue
        budget += max(0, int(math.ceil(gap / gap_threshold) - 1))
    return int(budget)


def _is_viable_addition(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    selected_events: list[Event],
    candidate: Event,
    remaining_events: list[Event],
    max_sequence_length: int,
    use_theory_viability: bool,
    gap_aware_viability: bool,
    budget_aware: bool,
) -> tuple[bool, str, float, dict]:
    tentative = sorted((*selected_events, candidate), key=lambda event: (event.timestamp, event.id))
    remaining_after = [event for event in remaining_events if event.id != candidate.id]
    remaining_slots = max(0, max_sequence_length - len(tentative))
    tentative_span = _timespan(tentative)
    bridge_budget_lb = _compute_bridge_budget(
        [float(event.timestamp) for event in tentative],
        float(grammar.max_temporal_gap),
    )
    viability_meta = {
        "tentative_length": len(tentative),
        "remaining_slots": int(remaining_slots),
        "bridge_budget_lb": int(bridge_budget_lb),
    }

    max_possible_len = len(tentative) + min(remaining_slots, len(remaining_after))
    if len(tentative) > max_sequence_length:
        return False, "too_long_partial", tentative_span, viability_meta
    if max_possible_len < grammar.min_length:
        return False, "insufficient_remaining_length", tentative_span, viability_meta

    max_coverage = _max_possible_coverage(
        selected_events=tentative,
        remaining_events=remaining_after,
        focal_actor=focal_actor,
        min_length=grammar.min_length,
        remaining_slots=remaining_slots,
    )
    if max_coverage + 1e-12 < grammar.focal_actor_coverage:
        return False, "coverage_upper_bound", tentative_span, viability_meta

    required_span = float(grammar.min_timespan_fraction * graph.duration)
    max_span = _max_achievable_span(
        selected_events=tentative,
        remaining_events=remaining_after,
        remaining_slots=remaining_slots,
    )
    if max_span + 1e-12 < required_span:
        return False, "span_upper_bound", max_span, viability_meta

    if _tp_locked_after_addition(tentative, remaining_after, grammar):
        return False, "tp_locked_insufficient_development", max_span, viability_meta

    if gap_aware_viability:
        gap_ok, gap_reason = _gap_bridge_viable(
            grammar=grammar,
            tentative_events=tentative,
            remaining_events=remaining_after,
            remaining_slots=remaining_slots,
        )
        if not gap_ok:
            return False, gap_reason, max_span, viability_meta

    # Budget reservation: even if a bridge exists, ensure we can still afford
    # the minimum number of bridges implied by current oversized gaps.
    if budget_aware and not math.isinf(grammar.max_temporal_gap):
        if bridge_budget_lb > remaining_slots:
            return False, "bridge_budget_exceeded", max_span, viability_meta

    if use_theory_viability:
        theory_reason = _theory_viability_rejection_reason(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            tentative_events=tentative,
        )
        if theory_reason is not None:
            return False, f"theory_{theory_reason}", max_span, viability_meta

    return True, "viable", max_span, viability_meta


def _build_viability_filtered_candidate(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    pool_events: list[Event],
    anchor: Event,
    max_sequence_length: int,
    use_theory_viability: bool,
    gap_aware_viability: bool,
    budget_aware: bool,
) -> tuple[tuple[Event, ...], dict]:
    if anchor.id not in {event.id for event in pool_events}:
        return (), {
            "anchor_id": anchor.id,
            "anchor_rejected": True,
            "anchor_reject_reason": "anchor_not_in_pool",
            "viability_rejection_reason_counts": {},
            "viability_rejections": 0,
            "selected_count": 0,
            "budget_trace": [],
            "first_budget_block": None,
        }

    by_id = {event.id: event for event in pool_events}
    selected_ids: set[str] = set()
    selected_events: list[Event] = []
    rejection_counts: Counter[str] = Counter()
    viability_rejections = 0
    budget_trace: list[dict] = []
    first_budget_block: dict | None = None

    remaining = [event for event in pool_events if event.id != anchor.id]
    anchor_viable, anchor_reason, _anchor_span, anchor_meta = _is_viable_addition(
        graph=graph,
        focal_actor=focal_actor,
        grammar=grammar,
        selected_events=[],
        candidate=anchor,
        remaining_events=remaining,
        max_sequence_length=max_sequence_length,
        use_theory_viability=use_theory_viability,
        gap_aware_viability=gap_aware_viability,
        budget_aware=budget_aware,
    )
    if not anchor_viable:
        rejection_counts[anchor_reason] += 1
        return (), {
            "anchor_id": anchor.id,
            "anchor_rejected": True,
            "anchor_reject_reason": anchor_reason,
            "viability_rejection_reason_counts": dict(rejection_counts),
            "viability_rejections": 1,
            "selected_count": 0,
            "budget_trace": [],
            "first_budget_block": (
                {
                    "step": 0,
                    "candidate_id": anchor.id,
                    "candidate_weight_rank": 1,
                    "candidate_weight": float(anchor.weight),
                    "candidate_timestamp": float(anchor.timestamp),
                    "bridge_budget_lb": int(anchor_meta.get("bridge_budget_lb", 0)),
                    "remaining_slots": int(anchor_meta.get("remaining_slots", 0)),
                }
                if anchor_reason == "bridge_budget_exceeded"
                else None
            ),
        }

    selected_ids.add(anchor.id)
    selected_events.append(anchor)
    budget_trace.append(
        {
            "step": 0,
            "selected_event_id": anchor.id,
            "selected_count": 1,
            "bridge_budget_lb": int(anchor_meta.get("bridge_budget_lb", 0)),
            "remaining_slots": int(anchor_meta.get("remaining_slots", 0)),
        }
    )

    while len(selected_ids) < max_sequence_length:
        remaining_events = [
            event for event in _sorted_by_weight(pool_events) if event.id not in selected_ids
        ]
        if not remaining_events:
            break

        chosen: Event | None = None
        chosen_meta: dict = {}
        chosen_rank = 0
        for rank, candidate in enumerate(remaining_events, start=1):
            viable, reason, _max_span, candidate_meta = _is_viable_addition(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                selected_events=selected_events,
                candidate=candidate,
                remaining_events=remaining_events,
                max_sequence_length=max_sequence_length,
                use_theory_viability=use_theory_viability,
                gap_aware_viability=gap_aware_viability,
                budget_aware=budget_aware,
            )
            if viable:
                chosen = candidate
                chosen_meta = candidate_meta
                chosen_rank = rank
                break
            rejection_counts[reason] += 1
            viability_rejections += 1
            if reason == "bridge_budget_exceeded" and first_budget_block is None:
                first_budget_block = {
                    "step": len(selected_events),
                    "candidate_id": candidate.id,
                    "candidate_weight_rank": rank,
                    "candidate_weight": float(candidate.weight),
                    "candidate_timestamp": float(candidate.timestamp),
                    "bridge_budget_lb": int(candidate_meta.get("bridge_budget_lb", 0)),
                    "remaining_slots": int(candidate_meta.get("remaining_slots", 0)),
                }

        if chosen is None:
            break

        selected_ids.add(chosen.id)
        selected_events.append(by_id[chosen.id])
        budget_trace.append(
            {
                "step": len(selected_events) - 1,
                "selected_event_id": chosen.id,
                "selected_event_weight_rank": int(chosen_rank),
                "selected_count": len(selected_events),
                "bridge_budget_lb": int(chosen_meta.get("bridge_budget_lb", 0)),
                "remaining_slots": int(chosen_meta.get("remaining_slots", 0)),
            }
        )

    selected_events = sorted(selected_events, key=lambda event: (event.timestamp, event.id))
    return tuple(selected_events), {
        "anchor_id": anchor.id,
        "anchor_rejected": False,
        "anchor_reject_reason": None,
        "viability_rejection_reason_counts": dict(rejection_counts),
        "viability_rejections": viability_rejections,
        "selected_count": len(selected_events),
        "budget_trace": budget_trace,
        "first_budget_block": first_budget_block,
    }


def viability_aware_greedy_extract(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    pool_strategy: str = "injection",
    n_anchors: int = 8,
    max_sequence_length: int = 20,
    injection_top_n: int = 40,
    min_injection_position: float = 0.0,
    scoring_fn: Callable[[ExtractedSequence], float] = tp_weighted_score,
    use_theory_viability: bool = True,
    gap_aware_viability: bool = False,
    budget_aware: bool = False,
) -> tuple[ExtractedSequence, dict]:
    """Run greedy extraction with viability-aware filtering."""

    anchors = sorted(
        graph.events,
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )[:n_anchors]
    build_pool = _pool_builder(pool_strategy)
    by_id = _event_lookup(graph)

    candidates: list[ExtractedSequence] = []
    n_valid_candidates = 0
    total_rejections = 0
    rejection_reasons: Counter[str] = Counter()
    anchor_rejected = 0
    per_anchor_diagnostics: list[dict] = []

    for anchor in anchors:
        if pool_strategy == "filtered_injection":
            pool_ids = build_pool(
                graph=graph,
                anchor_id=anchor.id,
                focal_actor=focal_actor,
                max_depth=3,
                injection_top_n=injection_top_n,
                min_position=min_injection_position,
            )
        elif pool_strategy == "injection":
            pool_ids = build_pool(
                graph=graph,
                anchor_id=anchor.id,
                focal_actor=focal_actor,
                max_depth=3,
                injection_top_n=injection_top_n,
            )
        else:
            pool_ids = build_pool(
                graph=graph,
                anchor_id=anchor.id,
                focal_actor=focal_actor,
                max_depth=3,
            )

        pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        pool_events = sorted(pool_events, key=lambda event: (event.timestamp, event.id))

        candidate_events, diag = _build_viability_filtered_candidate(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_events=pool_events,
            anchor=anchor,
            max_sequence_length=max_sequence_length,
            use_theory_viability=use_theory_viability,
            gap_aware_viability=gap_aware_viability,
            budget_aware=budget_aware,
        )

        total_rejections += int(diag["viability_rejections"])
        rejection_reasons.update(diag["viability_rejection_reason_counts"])
        if diag["anchor_rejected"]:
            anchor_rejected += 1

        per_anchor_diagnostics.append(
            {
                **diag,
                "pool_size": len(pool_events),
            }
        )

        if not candidate_events:
            continue

        candidate = _candidate_to_scored_sequence(
            graph=graph,
            candidate_events=candidate_events,
            focal_actor=focal_actor,
            grammar=grammar,
            scoring_fn=scoring_fn,
            metadata={
                "anchor_id": anchor.id,
                "pool_size": len(pool_ids),
                "pool_ids": tuple(sorted(pool_ids)),
                "viability_rejections_for_anchor": int(diag["viability_rejections"]),
                "viability_rejection_reason_counts": dict(diag["viability_rejection_reason_counts"]),
                "anchor_rejected": bool(diag["anchor_rejected"]),
            },
        )
        if candidate.valid:
            n_valid_candidates += 1
        candidates.append(candidate)

    diagnostics = {
        "n_candidates_evaluated": len(candidates),
        "n_valid_candidates": n_valid_candidates,
        "anchor_rejected_count": anchor_rejected,
        "total_viability_rejections": total_rejections,
        "viability_rejection_reason_counts": dict(rejection_reasons),
        "gap_aware_viability": bool(gap_aware_viability),
        "budget_aware": bool(budget_aware),
        "per_anchor": per_anchor_diagnostics,
    }

    if not candidates:
        empty = _empty_sequence(
            focal_actor,
            metadata={
                "n_candidates_evaluated": 0,
                "n_valid_candidates": 0,
                "anchor_rejected_count": anchor_rejected,
                "total_viability_rejections": total_rejections,
                "viability_rejection_reason_counts": dict(rejection_reasons),
                "absorbed": False,
                "absorption_step": None,
            },
        )
        return empty, diagnostics

    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    if valid_candidates:
        best = max(valid_candidates, key=lambda sequence: sequence.score)
    else:
        best = max(candidates, key=lambda sequence: sequence.score)

    merged_metadata = dict(best.metadata)
    merged_metadata.update(
        {
            "n_candidates_evaluated": len(candidates),
            "n_valid_candidates": n_valid_candidates,
            "anchor_rejected_count": anchor_rejected,
            "total_viability_rejections": total_rejections,
            "viability_rejection_reason_counts": dict(rejection_reasons),
            "absorbed": bool(best.metadata.get("absorbed", False)),
            "absorption_step": best.metadata.get("absorption_step"),
        }
    )
    return replace(best, metadata=merged_metadata), diagnostics
