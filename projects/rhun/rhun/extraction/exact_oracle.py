"""Exact oracle extraction via exhaustive turning-point and boundary search."""

from __future__ import annotations

from dataclasses import replace
from math import ceil

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_with_turning_point
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence


def _empty_result(focal_actor: str, metadata: dict) -> ExtractedSequence:
    return ExtractedSequence(
        events=(),
        phases=(),
        focal_actor=focal_actor,
        score=0.0,
        valid=False,
        violations=("no_valid_sequence",),
        metadata=metadata,
    )


def _sorted_by_weight(events: list[Event]) -> list[Event]:
    return sorted(
        events,
        key=lambda event: (event.weight, -event.timestamp, event.id),
        reverse=True,
    )


def _prefix_sums(events: list[Event]) -> list[float]:
    sums = [0.0]
    for event in events:
        sums.append(sums[-1] + float(event.weight))
    return sums


def _development_count(n_pre: int, k: int) -> int:
    n_setup = ceil(n_pre * 0.2) if n_pre > 0 else 0
    if k > 0:
        n_setup = min(n_setup, max(0, n_pre - k))
    return n_pre - n_setup


def _tp_position_penalty(n_pre: int, n_post: int) -> float:
    non_tp = n_pre + n_post
    if non_tp <= 0:
        return 0.5
    pos = n_pre / non_tp
    return 1.0 - abs(pos - 0.6) * 0.5


def exact_oracle_extract(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> tuple[ExtractedSequence, dict]:
    """
    Exact optimal extraction under current grammar/scoring semantics.

    Search space:
    - Focal-actor events only (matching the current heuristic oracle domain)
    - Turning point candidate t is fixed per outer loop
    - Exact enumeration over boundary-constrained selections before/after t
    - TP is enforced by phase assignment (same semantics as current heuristic oracle)
    """

    actor_events = [event for event in graph.events if focal_actor in event.actors]
    required_span = float(grammar.min_timespan_fraction * graph.duration)

    diagnostics = {
        "n_tp_candidates": len(actor_events),
        "n_candidates_evaluated": 0,
        "n_structurally_valid_candidates": 0,
        "n_valid_candidates": 0,
        "objective": "tp_weighted_score",
        "search_space": "focal_actor_events_only",
        "tp_forced_by_phase_assignment": True,
        "exact": True,
    }

    if not actor_events:
        empty = _empty_result(focal_actor, metadata={**diagnostics, "reason": "no_focal_events"})
        return empty, diagnostics

    by_id = {event.id: event for event in graph.events}
    best_sequence: ExtractedSequence | None = None
    best_score = float("-inf")
    best_tp_id: str | None = None

    for tp in actor_events:
        pre_events = [
            event
            for event in actor_events
            if event.timestamp < tp.timestamp
        ]
        post_events = [
            event
            for event in actor_events
            if event.timestamp > tp.timestamp
        ]

        p_count = len(pre_events)
        q_count = len(post_events)

        # Boundary helpers:
        # - first pre event index i fixes the earliest pre timestamp
        # - last post event index j fixes the latest post timestamp
        pre_tail_sorted: dict[int, list[Event]] = {}
        pre_tail_sums: dict[int, list[float]] = {}
        for i in range(p_count):
            tail = _sorted_by_weight(pre_events[i + 1 :])
            pre_tail_sorted[i] = tail
            pre_tail_sums[i] = _prefix_sums(tail)

        post_head_sorted: dict[int, list[Event]] = {}
        post_head_sums: dict[int, list[float]] = {}
        for j in range(q_count):
            head = _sorted_by_weight(post_events[:j])
            post_head_sorted[j] = head
            post_head_sums[j] = _prefix_sums(head)

        max_pre = min(p_count, grammar.max_length - 1)
        for n_pre in range(0, max_pre + 1):
            if n_pre < grammar.min_prefix_elements:
                continue
            if _development_count(n_pre, grammar.min_prefix_elements) < grammar.min_prefix_elements:
                continue

            min_post = max(0, grammar.min_length - (n_pre + 1))
            max_post = min(q_count, grammar.max_length - (n_pre + 1))
            if min_post > max_post:
                continue

            position_penalty = _tp_position_penalty(n_pre, 0)  # overwritten below for each n_post

            first_indices = [-1] if n_pre == 0 else list(range(0, p_count - n_pre + 1))
            for first_idx in first_indices:
                if n_pre == 0:
                    selected_pre: tuple[Event, ...] = ()
                    pre_weight = 0.0
                    first_ts = float(tp.timestamp)
                else:
                    tail = pre_tail_sorted[first_idx]
                    need_tail = n_pre - 1
                    if need_tail > len(tail):
                        continue
                    selected_pre = (pre_events[first_idx], *tail[:need_tail])
                    pre_weight = float(pre_events[first_idx].weight + pre_tail_sums[first_idx][need_tail])
                    first_ts = float(pre_events[first_idx].timestamp)

                for n_post in range(min_post, max_post + 1):
                    position_penalty = _tp_position_penalty(n_pre, n_post)
                    tp_term = float(tp.weight + (2.0 * tp.weight * position_penalty))

                    last_indices = [-1] if n_post == 0 else list(range(n_post - 1, q_count))
                    for last_idx in last_indices:
                        diagnostics["n_candidates_evaluated"] += 1

                        if n_post == 0:
                            selected_post: tuple[Event, ...] = ()
                            post_weight = 0.0
                            last_ts = float(tp.timestamp)
                        else:
                            head = post_head_sorted[last_idx]
                            need_head = n_post - 1
                            if need_head > len(head):
                                continue
                            selected_post = (*head[:need_head], post_events[last_idx])
                            post_weight = float(post_events[last_idx].weight + post_head_sums[last_idx][need_head])
                            last_ts = float(post_events[last_idx].timestamp)

                        if (last_ts - first_ts) + 1e-12 < required_span:
                            continue

                        diagnostics["n_structurally_valid_candidates"] += 1

                        optimistic_score = pre_weight + post_weight + tp_term
                        if optimistic_score + 1e-12 < best_score:
                            continue

                        selected_ids = [event.id for event in selected_pre]
                        selected_ids.append(tp.id)
                        selected_ids.extend(event.id for event in selected_post)
                        selected_events = tuple(
                            sorted(
                                (by_id[event_id] for event_id in selected_ids),
                                key=lambda event: (event.timestamp, event.id),
                            )
                        )
                        tp_idx = next(
                            (index for index, event in enumerate(selected_events) if event.id == tp.id),
                            None,
                        )
                        if tp_idx is None:
                            continue

                        phases = classify_with_turning_point(
                            selected_events,
                            tp_idx,
                            min_development=grammar.min_prefix_elements,
                        )
                        candidate = ExtractedSequence(
                            events=selected_events,
                            phases=phases,
                            focal_actor=focal_actor,
                        )
                        valid, violations = validate(candidate, grammar, graph)
                        if not valid:
                            continue
                        diagnostics["n_valid_candidates"] += 1

                        score = tp_weighted_score(candidate)
                        candidate = replace(
                            candidate,
                            score=score,
                            valid=True,
                            violations=tuple(violations),
                            metadata={"forced_turning_point": tp.id, "exact_oracle": True},
                        )

                        if score > best_score + 1e-12:
                            best_score = score
                            best_sequence = candidate
                            best_tp_id = tp.id

    if best_sequence is None:
        empty = _empty_result(
            focal_actor,
            metadata={**diagnostics, "reason": "no_valid_sequence", "best_turning_point_id": None},
        )
        return empty, diagnostics

    seq_meta = dict(best_sequence.metadata)
    seq_meta.update(
        {
            "n_candidates_evaluated": diagnostics["n_candidates_evaluated"],
            "n_structurally_valid_candidates": diagnostics["n_structurally_valid_candidates"],
            "n_valid_candidates": diagnostics["n_valid_candidates"],
            "best_turning_point_id": best_tp_id,
            "exact_oracle": True,
        }
    )
    best_sequence = replace(best_sequence, metadata=seq_meta)
    return best_sequence, diagnostics
