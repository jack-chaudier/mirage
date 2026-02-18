"""Search strategies for constrained extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_phases, classify_with_turning_point
from rhun.extraction.pool_construction import bfs_pool, filtered_injection_pool, injection_pool
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase
from rhun.theory.theorem import diagnose_absorption


@dataclass(frozen=True)
class _BeamState:
    selected_ids: tuple[str, ...]
    selected_set: frozenset[str]
    weight_sum: float
    focal_hits: int
    min_timestamp: float
    max_timestamp: float


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


def _timespan(events: tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    return float(events[-1].timestamp - events[0].timestamp)


def _is_timespan_violation(violations: tuple[str, ...] | list[str]) -> bool:
    return any(v.startswith("insufficient_timespan") for v in violations)


def _causal_order_valid(events: tuple[Event, ...]) -> bool:
    """Require that any in-sequence parent appears at or before its child."""
    index = {event.id: i for i, event in enumerate(events)}
    for i, event in enumerate(events):
        for parent_id in event.causal_parents:
            parent_index = index.get(parent_id)
            if parent_index is None:
                continue
            if parent_index > i:
                return False
    return True


def _candidate_to_scored_sequence(
    graph: CausalGraph,
    candidate_events: tuple[Event, ...],
    focal_actor: str,
    grammar: GrammarConfig,
    scoring_fn: Callable[[ExtractedSequence], float],
    metadata: dict,
    override_tp_id: str | None = None,
) -> ExtractedSequence:
    if override_tp_id is not None:
        tp_idx = next(
            (i for i, e in enumerate(candidate_events) if e.id == override_tp_id),
            None,
        )
        if tp_idx is not None:
            phases = classify_with_turning_point(
                candidate_events,
                tp_idx,
                min_development=grammar.min_prefix_elements,
            )
        else:
            phases = classify_phases(
                candidate_events,
                min_development=grammar.min_prefix_elements,
            )
    else:
        phases = classify_phases(
            candidate_events,
            min_development=grammar.min_prefix_elements,
        )
    candidate = ExtractedSequence(
        events=candidate_events,
        phases=phases,
        focal_actor=focal_actor,
    )
    valid, violations = validate(candidate, grammar, graph)
    absorption = diagnose_absorption(candidate, grammar)
    score = scoring_fn(candidate)
    merged_metadata = dict(metadata)
    merged_metadata.update(
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
        metadata=merged_metadata,
    )


def _downsample_pool(
    graph: CausalGraph,
    pool_ids: set[str],
    focal_actor: str,
    anchor_id: str,
    max_sequence_length: int,
    force_include_id: str | None = None,
) -> tuple[Event, ...]:
    if not pool_ids:
        return ()

    by_id = _event_lookup(graph)
    reverse = graph.build_reverse_adjacency()

    # Keep anchor plus a chain of ancestors for causal continuity.
    kept_ids: set[str] = set()
    # Force-include the override TP event if specified.
    if force_include_id is not None and force_include_id in pool_ids:
        kept_ids.add(force_include_id)
    stack = [anchor_id]
    while stack and len(kept_ids) < max_sequence_length:
        current = stack.pop()
        if current not in pool_ids or current in kept_ids:
            continue
        kept_ids.add(current)
        for parent_id in reverse.get(current, []):
            if parent_id in pool_ids and parent_id not in kept_ids:
                stack.append(parent_id)

    # Prioritize focal-actor events by weight.
    focal_candidates = sorted(
        (
            by_id[event_id]
            for event_id in pool_ids
            if event_id in by_id and focal_actor in by_id[event_id].actors and event_id not in kept_ids
        ),
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )
    for event in focal_candidates:
        if len(kept_ids) >= max_sequence_length:
            break
        kept_ids.add(event.id)

    # Fill remainder by descending weight.
    remaining = sorted(
        (
            by_id[event_id]
            for event_id in pool_ids
            if event_id in by_id and event_id not in kept_ids
        ),
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )
    for event in remaining:
        if len(kept_ids) >= max_sequence_length:
            break
        kept_ids.add(event.id)

    selected = sorted((by_id[event_id] for event_id in kept_ids), key=lambda event: event.timestamp)
    return tuple(selected)


def _beam_partial_score(
    state: _BeamState,
    graph_duration: float,
    max_sequence_length: int,
) -> float:
    count = len(state.selected_ids)
    if count == 0:
        return float("-inf")

    span = max(0.0, state.max_timestamp - state.min_timestamp)
    span_fraction = (span / graph_duration) if graph_duration > 0 else 0.0
    coverage = state.focal_hits / count
    length_signal = min(count / max_sequence_length, 1.0) if max_sequence_length > 0 else 0.0
    return state.weight_sum + 0.25 * span_fraction + 0.10 * coverage + 0.05 * length_signal


def _beam_sequences_for_anchor(
    graph: CausalGraph,
    pool_ids: set[str],
    focal_actor: str,
    anchor_id: str,
    max_sequence_length: int,
    beam_width: int,
) -> list[tuple[Event, ...]]:
    if not pool_ids:
        return []

    by_id = _event_lookup(graph)
    anchor = by_id.get(anchor_id)
    if anchor is None:
        return []

    if anchor_id not in pool_ids:
        return []

    ordered = sorted(
        (by_id[event_id] for event_id in pool_ids if event_id in by_id and event_id != anchor_id),
        key=lambda event: (event.weight, -event.timestamp, event.id),
        reverse=True,
    )

    initial = _BeamState(
        selected_ids=(anchor_id,),
        selected_set=frozenset({anchor_id}),
        weight_sum=float(anchor.weight),
        focal_hits=int(focal_actor in anchor.actors),
        min_timestamp=float(anchor.timestamp),
        max_timestamp=float(anchor.timestamp),
    )
    beam: list[_BeamState] = [initial]

    for event in ordered:
        expanded: list[_BeamState] = []
        for state in beam:
            # Skip branch
            expanded.append(state)

            # Include branch
            if (
                len(state.selected_ids) < max_sequence_length
                and event.id not in state.selected_set
            ):
                selected_set = frozenset(set(state.selected_set) | {event.id})
                selected_ids = tuple(sorted(selected_set))
                expanded.append(
                    _BeamState(
                        selected_ids=selected_ids,
                        selected_set=selected_set,
                        weight_sum=state.weight_sum + float(event.weight),
                        focal_hits=state.focal_hits + int(focal_actor in event.actors),
                        min_timestamp=min(state.min_timestamp, float(event.timestamp)),
                        max_timestamp=max(state.max_timestamp, float(event.timestamp)),
                    )
                )

        expanded.sort(
            key=lambda state: (
                _beam_partial_score(state, graph.duration, max_sequence_length),
                state.weight_sum,
                state.selected_ids,
            ),
            reverse=True,
        )
        beam = expanded[:beam_width]

    sequences: list[tuple[Event, ...]] = []
    for state in beam:
        events = sorted((by_id[event_id] for event_id in state.selected_ids), key=lambda event: event.timestamp)
        sequences.append(tuple(events))
    return sequences


def greedy_extract(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    pool_strategy: str = "injection",  # "bfs", "injection", "filtered_injection"
    n_anchors: int = 8,
    max_sequence_length: int = 20,
    injection_top_n: int = 40,
    min_injection_position: float = 0.0,  # Only for filtered_injection
    scoring_fn: Callable[[ExtractedSequence], float] = tp_weighted_score,
    override_tp_id: str | None = None,
) -> ExtractedSequence:
    anchors = sorted(
        graph.events,
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )[:n_anchors]

    build_pool = _pool_builder(pool_strategy)

    candidates: list[ExtractedSequence] = []
    n_valid_candidates = 0

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

        # Inject override TP event into pool if specified.
        if override_tp_id is not None:
            pool_ids.add(override_tp_id)

        candidate_events = _downsample_pool(
            graph=graph,
            pool_ids=pool_ids,
            focal_actor=focal_actor,
            anchor_id=anchor.id,
            max_sequence_length=max_sequence_length,
            force_include_id=override_tp_id,
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
            },
            override_tp_id=override_tp_id,
        )
        if candidate.valid:
            n_valid_candidates += 1
        candidates.append(candidate)

    if not candidates:
        return _empty_sequence(
            focal_actor,
            metadata={
                "n_candidates_evaluated": 0,
                "n_valid_candidates": 0,
                "absorbed": False,
                "absorption_step": None,
            },
        )

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
            "absorbed": bool(best.metadata.get("absorbed", False)),
            "absorption_step": best.metadata.get("absorption_step"),
        }
    )

    return replace(best, metadata=merged_metadata)


def repair_timespan(
    graph: CausalGraph,
    sequence: ExtractedSequence,
    grammar: GrammarConfig,
    pool_ids: set[str] | None = None,
    scoring_fn: Callable[[ExtractedSequence], float] = tp_weighted_score,
    max_swaps: int = 12,
) -> ExtractedSequence:
    """
    Attempt 1-opt repair for sequences that fail due to insufficient timespan.

    The repair only swaps DEVELOPMENT events with alternatives from the existing
    candidate pool; it does not rebuild pools or rerun search.
    """

    def build_sequence(events: tuple[Event, ...]) -> ExtractedSequence:
        phases = classify_phases(events, min_development=grammar.min_prefix_elements)
        candidate = ExtractedSequence(
            events=events,
            phases=phases,
            focal_actor=sequence.focal_actor,
        )
        valid, violations = validate(candidate, grammar, graph)
        return replace(
            candidate,
            score=scoring_fn(candidate),
            valid=valid,
            violations=tuple(violations),
        )

    metadata = dict(sequence.metadata)
    required_span = grammar.min_timespan_fraction * graph.duration
    initial_span = _timespan(sequence.events)

    if pool_ids is None:
        raw_pool_ids = metadata.get("pool_ids")
        if isinstance(raw_pool_ids, (tuple, list, set)):
            pool_ids = {str(event_id) for event_id in raw_pool_ids}

    if not pool_ids:
        metadata.update(
            {
                "repair_attempted": True,
                "repair_success": False,
                "repair_failure_reason": "missing_pool_ids",
                "repair_swap_count": 0,
                "repair_initial_timespan": initial_span,
                "repair_final_timespan": initial_span,
                "repair_required_timespan": required_span,
                "repair_swaps": (),
            }
        )
        return replace(sequence, metadata=metadata)

    current = build_sequence(tuple(sequence.events))
    if current.valid:
        metadata.update(
            {
                "repair_attempted": True,
                "repair_success": True,
                "repair_failure_reason": None,
                "repair_swap_count": 0,
                "repair_initial_timespan": initial_span,
                "repair_final_timespan": _timespan(current.events),
                "repair_required_timespan": required_span,
                "repair_swaps": (),
            }
        )
        return replace(current, metadata=metadata)

    if not _is_timespan_violation(current.violations):
        metadata.update(
            {
                "repair_attempted": True,
                "repair_success": False,
                "repair_failure_reason": "not_timespan_failure",
                "repair_swap_count": 0,
                "repair_initial_timespan": initial_span,
                "repair_final_timespan": _timespan(current.events),
                "repair_required_timespan": required_span,
                "repair_swaps": (),
            }
        )
        return replace(current, metadata=metadata)

    by_id = _event_lookup(graph)
    pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
    if not pool_events:
        metadata.update(
            {
                "repair_attempted": True,
                "repair_success": False,
                "repair_failure_reason": "empty_pool_events",
                "repair_swap_count": 0,
                "repair_initial_timespan": initial_span,
                "repair_final_timespan": _timespan(current.events),
                "repair_required_timespan": required_span,
                "repair_swaps": (),
            }
        )
        return replace(current, metadata=metadata)

    swaps: list[dict] = []
    failure_reason = "no_improving_swap_available"

    for step in range(max_swaps):
        current_span = _timespan(current.events)
        if current.valid and current_span >= required_span:
            failure_reason = None
            break

        if not _is_timespan_violation(current.violations):
            failure_reason = "non_timespan_violation_remaining"
            break

        tp = current.turning_point
        if tp is None:
            failure_reason = "missing_turning_point"
            break

        current_ids = {event.id for event in current.events}
        development_indices = [
            idx for idx, phase in enumerate(current.phases) if phase == Phase.DEVELOPMENT
        ]
        if not development_indices:
            failure_reason = "no_development_events_to_swap"
            break

        current_non_timespan = [
            v for v in current.violations if not v.startswith("insufficient_timespan")
        ]

        best_candidate: dict | None = None

        for idx in development_indices:
            old_event = current.events[idx]
            for alt in pool_events:
                if alt.id == old_event.id:
                    continue
                if alt.id in current_ids:
                    continue
                if sequence.focal_actor not in alt.actors:
                    continue
                if alt.timestamp >= tp.timestamp:
                    continue
                # Keep TP stable under reclassification.
                if alt.weight >= tp.weight:
                    continue

                trial_events = list(current.events)
                trial_events[idx] = alt
                trial_events.sort(key=lambda event: (event.timestamp, event.id))
                trial_tuple = tuple(trial_events)

                if not _causal_order_valid(trial_tuple):
                    continue

                trial = build_sequence(trial_tuple)
                trial_span = _timespan(trial.events)
                span_improvement = trial_span - current_span
                if span_improvement <= 1e-12:
                    continue

                alt_pos = next(
                    (i for i, event in enumerate(trial.events) if event.id == alt.id),
                    None,
                )
                if alt_pos is None or trial.phases[alt_pos] != Phase.DEVELOPMENT:
                    continue

                trial_non_timespan = [
                    v for v in trial.violations if not v.startswith("insufficient_timespan")
                ]
                if len(trial_non_timespan) > len(current_non_timespan):
                    continue

                weight_loss = float(old_event.weight - alt.weight)
                deficit_after = max(0.0, required_span - trial_span)

                candidate = {
                    "trial": trial,
                    "old_event": old_event,
                    "new_event": alt,
                    "span_improvement": span_improvement,
                    "weight_loss": weight_loss,
                    "deficit_after": deficit_after,
                    "step": step,
                }
                if best_candidate is None:
                    best_candidate = candidate
                    continue

                better_improvement = (
                    candidate["span_improvement"] > best_candidate["span_improvement"] + 1e-12
                )
                same_improvement = (
                    abs(candidate["span_improvement"] - best_candidate["span_improvement"]) <= 1e-12
                )
                smaller_weight_loss = (
                    candidate["weight_loss"] < best_candidate["weight_loss"] - 1e-12
                )
                same_weight_loss = (
                    abs(candidate["weight_loss"] - best_candidate["weight_loss"]) <= 1e-12
                )
                smaller_deficit = (
                    candidate["deficit_after"] < best_candidate["deficit_after"] - 1e-12
                )

                if (
                    better_improvement
                    or (same_improvement and smaller_weight_loss)
                    or (same_improvement and same_weight_loss and smaller_deficit)
                ):
                    best_candidate = candidate

        if best_candidate is None:
            failure_reason = "no_improving_swap_available"
            break

        current = best_candidate["trial"]
        swaps.append(
            {
                "step": int(best_candidate["step"]),
                "old_event_id": best_candidate["old_event"].id,
                "new_event_id": best_candidate["new_event"].id,
                "old_timestamp": float(best_candidate["old_event"].timestamp),
                "new_timestamp": float(best_candidate["new_event"].timestamp),
                "old_weight": float(best_candidate["old_event"].weight),
                "new_weight": float(best_candidate["new_event"].weight),
                "weight_loss": float(best_candidate["weight_loss"]),
                "span_improvement": float(best_candidate["span_improvement"]),
                "remaining_deficit": float(best_candidate["deficit_after"]),
            }
        )

    final_span = _timespan(current.events)
    final_valid = bool(current.valid)
    repair_success = final_valid and final_span >= required_span

    metadata.update(
        {
            "repair_attempted": True,
            "repair_success": repair_success,
            "repair_failure_reason": None if repair_success else failure_reason,
            "repair_swap_count": len(swaps),
            "repair_initial_timespan": initial_span,
            "repair_final_timespan": final_span,
            "repair_required_timespan": required_span,
            "repair_weight_delta": float(current.score - sequence.score),
            "repair_swaps": tuple(swaps),
        }
    )
    return replace(current, metadata=metadata)


def beam_search_extract(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    pool_strategy: str = "injection",  # "bfs", "injection", "filtered_injection"
    n_anchors: int = 8,
    max_sequence_length: int = 20,
    injection_top_n: int = 40,
    min_injection_position: float = 0.0,  # Only for filtered_injection
    scoring_fn: Callable[[ExtractedSequence], float] = tp_weighted_score,
    beam_width: int = 4,
) -> ExtractedSequence:
    if beam_width <= 0:
        raise ValueError("beam_width must be >= 1")
    if beam_width == 1:
        return greedy_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_strategy=pool_strategy,
            n_anchors=n_anchors,
            max_sequence_length=max_sequence_length,
            injection_top_n=injection_top_n,
            min_injection_position=min_injection_position,
            scoring_fn=scoring_fn,
        )

    anchors = sorted(
        graph.events,
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )[:n_anchors]
    build_pool = _pool_builder(pool_strategy)

    candidates: list[ExtractedSequence] = []
    n_valid_candidates = 0

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

        beam_sequences = _beam_sequences_for_anchor(
            graph=graph,
            pool_ids=pool_ids,
            focal_actor=focal_actor,
            anchor_id=anchor.id,
            max_sequence_length=max_sequence_length,
            beam_width=beam_width,
        )
        for rank, sequence_events in enumerate(beam_sequences):
            if not sequence_events:
                continue

            candidate = _candidate_to_scored_sequence(
                graph=graph,
                candidate_events=sequence_events,
                focal_actor=focal_actor,
                grammar=grammar,
                scoring_fn=scoring_fn,
                metadata={
                    "anchor_id": anchor.id,
                    "pool_size": len(pool_ids),
                    "pool_ids": tuple(sorted(pool_ids)),
                    "beam_width": beam_width,
                    "beam_rank": rank,
                },
            )
            if candidate.valid:
                n_valid_candidates += 1
            candidates.append(candidate)

    if not candidates:
        return _empty_sequence(
            focal_actor,
            metadata={
                "n_candidates_evaluated": 0,
                "n_valid_candidates": 0,
                "absorbed": False,
                "absorption_step": None,
                "beam_width": beam_width,
            },
        )

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
            "absorbed": bool(best.metadata.get("absorbed", False)),
            "absorption_step": best.metadata.get("absorption_step"),
            "beam_width": beam_width,
        }
    )
    return replace(best, metadata=merged_metadata)


def _build_oracle_candidate(
    graph: CausalGraph,
    focal_actor: str,
    turning_point: Event,
    grammar: GrammarConfig,
    max_sequence_length: int,
) -> tuple[Event, ...]:
    actor_events = [event for event in graph.events if focal_actor in event.actors]
    before = [event for event in actor_events if event.timestamp < turning_point.timestamp]
    after = [event for event in actor_events if event.timestamp > turning_point.timestamp]

    before_sorted = sorted(before, key=lambda event: (event.weight, -event.timestamp), reverse=True)
    after_sorted = sorted(after, key=lambda event: (event.weight, -event.timestamp), reverse=True)

    desired_before = min(
        len(before_sorted),
        max(grammar.min_prefix_elements + 1, int(max_sequence_length * 0.6)),
    )
    chosen_before = before_sorted[:desired_before]

    remaining_budget = max_sequence_length - len(chosen_before) - 1
    desired_after = min(len(after_sorted), max(0, remaining_budget))
    chosen_after = after_sorted[:desired_after]

    selected_ids = {event.id for event in chosen_before + chosen_after}
    selected_ids.add(turning_point.id)

    # If still too short for grammar, fill from remaining actor events by weight.
    if len(selected_ids) < grammar.min_length:
        extras = [event for event in actor_events if event.id not in selected_ids]
        extras.sort(key=lambda event: (event.weight, -event.timestamp), reverse=True)
        for event in extras:
            if len(selected_ids) >= min(max_sequence_length, grammar.min_length):
                break
            selected_ids.add(event.id)

    by_id = {event.id: event for event in graph.events}
    selected = [by_id[event_id] for event_id in selected_ids if event_id in by_id]
    selected.sort(key=lambda event: event.timestamp)
    return tuple(selected)


def oracle_extract(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    tp_window: tuple[float, float] = (0.0, 1.0),  # Allowed TP positions
    max_sequence_length: int = 20,
    scoring_fn: Callable[[ExtractedSequence], float] = tp_weighted_score,
) -> tuple[ExtractedSequence | None, dict]:
    """Returns (best_valid_sequence, diagnostics_dict)."""
    lower, upper = tp_window

    candidates: list[ExtractedSequence] = []
    valid_candidates: list[ExtractedSequence] = []

    for event in graph.events:
        if focal_actor not in event.actors:
            continue
        position = graph.global_position(event)
        if position < lower or position > upper:
            continue

        candidate_events = _build_oracle_candidate(
            graph=graph,
            focal_actor=focal_actor,
            turning_point=event,
            grammar=grammar,
            max_sequence_length=max_sequence_length,
        )
        if not candidate_events:
            continue

        tp_idx = next(
            (i for i, candidate_event in enumerate(candidate_events) if candidate_event.id == event.id),
            None,
        )
        if tp_idx is None:
            continue

        phases = classify_with_turning_point(candidate_events, tp_idx)
        candidate = ExtractedSequence(
            events=candidate_events,
            phases=phases,
            focal_actor=focal_actor,
        )
        valid, violations = validate(candidate, grammar, graph)
        score = scoring_fn(candidate)
        candidate = replace(
            candidate,
            score=score,
            valid=valid,
            violations=tuple(violations),
            metadata={"forced_turning_point": event.id},
        )

        candidates.append(candidate)
        if valid:
            valid_candidates.append(candidate)

    diagnostics = {
        "n_candidates_evaluated": len(candidates),
        "n_valid_candidates": len(valid_candidates),
        "valid_fraction": (len(valid_candidates) / len(candidates)) if candidates else 0.0,
    }

    if not valid_candidates:
        return None, diagnostics

    best = max(valid_candidates, key=lambda sequence: sequence.score)
    return best, diagnostics
