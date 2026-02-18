"""Viability characterization for partial extraction states."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Iterable, Sequence

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_phases
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


def _sorted_unique(events: Sequence[Event]) -> tuple[Event, ...]:
    by_id: dict[str, Event] = {}
    for event in events:
        by_id[event.id] = event
    return tuple(sorted(by_id.values(), key=lambda event: (event.timestamp, event.id)))


def _timespan(events: Sequence[Event]) -> float:
    if not events:
        return 0.0
    return float(events[-1].timestamp - events[0].timestamp)


def _sequence_state(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    events: tuple[Event, ...],
) -> ExtractedSequence:
    phases = classify_phases(events, min_development=grammar.min_prefix_elements)
    seq = ExtractedSequence(events=events, phases=phases, focal_actor=focal_actor)
    valid, violations = validate(seq, grammar, graph)
    return replace(seq, valid=valid, violations=tuple(violations))


def _dev_count_before_tp(phases: tuple[Phase, ...]) -> tuple[int, int | None]:
    tp_idx = next((i for i, phase in enumerate(phases) if phase == Phase.TURNING_POINT), None)
    if tp_idx is None:
        return 0, None
    development = sum(1 for phase in phases[:tp_idx] if phase == Phase.DEVELOPMENT)
    return development, tp_idx


def _exhaustive_completion_exists(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    partial: tuple[Event, ...],
    remaining: tuple[Event, ...],
) -> tuple[bool, str]:
    """
    Exact viability check for small remaining pools.

    Enumerates all appended subsets of remaining events.
    """
    if not remaining:
        seq = _sequence_state(graph, focal_actor, grammar, partial)
        return bool(seq.valid), ("exact_valid" if seq.valid else "exact_no_valid_completion")

    base = list(remaining)
    n = len(base)
    for r in range(0, n + 1):
        for idxs in combinations(range(n), r):
            extension = [base[i] for i in idxs]
            candidate = _sorted_unique((*partial, *extension))
            seq = _sequence_state(graph, focal_actor, grammar, candidate)
            if seq.valid:
                return True, "exact_completion_found"
    return False, "exact_no_valid_completion"


def compute_viability(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    partial_sequence: Sequence[Event],
    exhaustive_limit: int = 14,
) -> dict:
    """
    Determine whether a valid completion exists from a partial state.

    Returns:
        {
            "viable": bool,
            "dfa_state": str,
            "dev_count": int,
            "remaining_events": int,
            "reason": str,
        }
    """
    partial = _sorted_unique(partial_sequence)
    partial_ids = {event.id for event in partial}

    if partial:
        tail_time = partial[-1].timestamp
        remaining = tuple(
            event
            for event in graph.events_for_actor(focal_actor)
            if event.id not in partial_ids and event.timestamp > tail_time
        )
    else:
        remaining = tuple(graph.events_for_actor(focal_actor))

    state_seq = _sequence_state(graph, focal_actor, grammar, partial)
    dfa_state = state_seq.phases[-1].name if state_seq.phases else "START"
    dev_count, tp_idx = _dev_count_before_tp(state_seq.phases)

    result = {
        "viable": True,
        "dfa_state": dfa_state,
        "dev_count": dev_count,
        "remaining_events": len(remaining),
        "reason": "undetermined",
    }

    # Exact check for small states.
    if len(remaining) <= exhaustive_limit:
        viable, reason = _exhaustive_completion_exists(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            partial=partial,
            remaining=remaining,
        )
        result.update({"viable": viable, "reason": reason, "method": "exact"})
        return result

    # Necessary bounds on length.
    if len(partial) > grammar.max_length:
        result.update(
            {
                "viable": False,
                "reason": f"too_long_partial: {len(partial)} > {grammar.max_length}",
                "method": "analytic",
            }
        )
        return result

    max_possible_len = len(partial) + len(remaining)
    if max_possible_len < grammar.min_length:
        result.update(
            {
                "viable": False,
                "reason": f"insufficient_remaining_length: {max_possible_len} < {grammar.min_length}",
                "method": "analytic",
            }
        )
        return result

    # Coverage upper bound (remaining events are focal by construction).
    partial_focal = sum(1 for event in partial if focal_actor in event.actors)
    max_possible_focal = partial_focal + len(remaining)
    max_possible_coverage = (
        (max_possible_focal / max_possible_len) if max_possible_len > 0 else 0.0
    )
    if max_possible_coverage + 1e-12 < grammar.focal_actor_coverage:
        result.update(
            {
                "viable": False,
                "reason": (
                    f"coverage_upper_bound: {max_possible_coverage:.3f} < "
                    f"{grammar.focal_actor_coverage:.3f}"
                ),
                "method": "analytic",
            }
        )
        return result

    # Timespan upper bound from this state.
    if graph.duration > 0:
        if partial:
            earliest = partial[0].timestamp
            latest_now = partial[-1].timestamp
        elif remaining:
            earliest = remaining[0].timestamp
            latest_now = remaining[0].timestamp
        else:
            earliest = 0.0
            latest_now = 0.0

        latest_possible = max(
            latest_now,
            max((event.timestamp for event in remaining), default=latest_now),
        )
        max_possible_span_fraction = (latest_possible - earliest) / graph.duration
        if max_possible_span_fraction + 1e-12 < grammar.min_timespan_fraction:
            result.update(
                {
                    "viable": False,
                    "reason": (
                        f"timespan_upper_bound: {max_possible_span_fraction:.3f} < "
                        f"{grammar.min_timespan_fraction:.3f}"
                    ),
                    "method": "analytic",
                }
            )
            return result

    # Absorbing-state style TP lock:
    # if TP already assigned with insufficient development and no heavier
    # future event can reassign TP, the state is non-viable.
    tp_event = state_seq.turning_point
    if tp_event is not None and tp_idx is not None:
        if dev_count < grammar.min_prefix_elements:
            heavier_remaining = any(event.weight > tp_event.weight for event in remaining)
            if not heavier_remaining:
                result.update(
                    {
                        "viable": False,
                        "reason": (
                            "tp_locked_insufficient_development: "
                            f"dev_count={dev_count} < k={grammar.min_prefix_elements}"
                        ),
                        "method": "analytic",
                    }
                )
                return result

    # If no necessary condition rejects viability, treat as viable.
    result.update({"viable": True, "reason": "analytic_bounds_not_violated", "method": "analytic"})
    return result


def partial_at_tp_assignment(sequence: ExtractedSequence) -> tuple[Event, ...]:
    """Return the prefix ending at the first TURNING_POINT assignment."""
    tp_idx = next((i for i, phase in enumerate(sequence.phases) if phase == Phase.TURNING_POINT), None)
    if tp_idx is None:
        return sequence.events
    return sequence.events[: tp_idx + 1]


def viability_along_sequence(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    sequence_events: Iterable[Event],
    exhaustive_limit: int = 10,
) -> dict:
    """Evaluate viability at each prefix of a sequence."""
    events = _sorted_unique(tuple(sequence_events))
    states: list[dict] = []
    first_non_viable_step: int | None = None

    for step in range(1, len(events) + 1):
        partial = events[:step]
        state = compute_viability(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            partial_sequence=partial,
            exhaustive_limit=exhaustive_limit,
        )
        row = {"step": step - 1, **state}
        states.append(row)
        if not row["viable"] and first_non_viable_step is None:
            first_non_viable_step = step - 1

    return {
        "states": states,
        "first_non_viable_step": first_non_viable_step,
        "all_viable": first_non_viable_step is None,
    }
