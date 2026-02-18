"""TP-conditioned label-setting solver for constrained extraction.

This decomposes endogenous turning-point (TP) choice into:
1) Outer loop over focal-actor TP candidates.
2) Inner resource-constrained label-setting DP with fixed TP + fixed n_pre.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, inf, isinf
from typing import Callable

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.pool_construction import bfs_pool, filtered_injection_pool, injection_pool
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase

EPS = 1e-12


@dataclass(frozen=True)
class _Label:
    score: float
    slots_used: int
    first_t: float
    last_index: int
    pre_count: int
    tp_selected: bool
    focal_hits: int
    selected_indices: tuple[int, ...]


def _pool_builder(strategy: str):
    if strategy == "bfs":
        return bfs_pool
    if strategy == "injection":
        return injection_pool
    if strategy == "filtered_injection":
        return filtered_injection_pool
    raise ValueError(f"Unknown pool_strategy: {strategy}")


def _sorted_focal_candidates(graph: CausalGraph, focal_actor: str) -> list[Event]:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    return sorted(
        focal_events,
        key=lambda event: (float(event.weight), -float(event.timestamp), event.id),
        reverse=True,
    )


def _n_setup_count(n_pre: int, k: int) -> int:
    n_setup = ceil(0.2 * n_pre) if n_pre > 0 else 0
    n_setup_cap = max(0, n_pre - min(k, n_pre))
    return int(min(n_setup, n_setup_cap))


def _phase_state(last_phase: int, pre_count: int, tp_selected: bool, focal_hits: int) -> tuple[int, ...]:
    return (int(last_phase), int(pre_count), int(tp_selected), int(focal_hits))


def _dominates(a: _Label, b: _Label) -> bool:
    return (
        a.score >= (b.score - EPS)
        and a.slots_used <= b.slots_used
        and a.first_t <= (b.first_t + EPS)
    )


def _insert_with_dominance(
    frontier: dict[tuple[str, tuple[int, ...]], list[_Label]],
    bucket: tuple[str, tuple[int, ...]],
    label: _Label,
) -> tuple[bool, int, int]:
    current = frontier.get(bucket, [])

    for existing in current:
        if _dominates(existing, label):
            return False, 0, 1

    kept: list[_Label] = []
    removed = 0
    for existing in current:
        if _dominates(label, existing):
            removed += 1
        else:
            kept.append(existing)

    kept.append(label)
    frontier[bucket] = kept
    return True, removed, 0


def _build_min_hops_to_target(
    events: list[Event],
    target_index: int,
    max_gap: float,
) -> tuple[int, list[float]]:
    """Lower bound hops to reach target under gap-valid temporal edges."""
    n = len(events)
    hops = [inf] * n
    if target_index < 0 or target_index >= n:
        return int(inf), hops

    hops[target_index] = 0
    if isinf(max_gap):
        for i in range(target_index):
            hops[i] = 1
        return 1, hops

    target_t = float(events[target_index].timestamp)
    for i in range(target_index - 1, -1, -1):
        left_t = float(events[i].timestamp)
        best = inf

        direct_gap = target_t - left_t
        if direct_gap > EPS and direct_gap <= max_gap + EPS:
            best = 1

        for j in range(i + 1, target_index):
            mid_t = float(events[j].timestamp)
            if mid_t <= left_t + EPS:
                continue
            if (mid_t - left_t) > max_gap + EPS:
                break
            if hops[j] != inf:
                best = min(best, 1 + hops[j])

        hops[i] = best

    # From virtual start, first pick has no gap constraint.
    return 1, hops


def _reconstruct_phases(
    selected_indices: tuple[int, ...],
    events: list[Event],
    tau_index: int,
    tau_timestamp: float,
    n_setup: int,
) -> tuple[Phase, ...]:
    phases: list[Phase] = []
    pre_seen = 0

    for index in selected_indices:
        event = events[index]
        timestamp = float(event.timestamp)
        if index == tau_index:
            phases.append(Phase.TURNING_POINT)
            continue

        if timestamp < tau_timestamp - EPS:
            phase = Phase.SETUP if pre_seen < n_setup else Phase.DEVELOPMENT
            phases.append(phase)
            pre_seen += 1
            continue

        if timestamp > tau_timestamp + EPS:
            phases.append(Phase.RESOLUTION)
            continue

        # Equal-timestamp non-TP events are intentionally excluded upstream.
        return ()

    return tuple(phases)


def _finalize_candidate(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    scoring_fn: Callable[[ExtractedSequence], float],
    events: list[Event],
    selected_indices: tuple[int, ...],
    phases: tuple[Phase, ...],
    metadata: dict,
) -> ExtractedSequence:
    selected_events = tuple(events[index] for index in selected_indices)
    candidate = ExtractedSequence(
        events=selected_events,
        phases=phases,
        focal_actor=focal_actor,
    )
    valid, violations = validate(candidate, grammar, graph)
    score = scoring_fn(candidate)
    return replace(
        candidate,
        score=float(score),
        valid=bool(valid),
        violations=tuple(violations),
        metadata=metadata,
    )


def _solve_fixed_tau_and_n_pre(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    tau: Event,
    tau_rank: int,
    pool_events: list[Event],
    n_pre: int,
    sequence_length_budget: int,
    active_max_gap: float,
    scoring_fn: Callable[[ExtractedSequence], float],
) -> tuple[ExtractedSequence | None, dict]:
    ordered = sorted(pool_events, key=lambda event: (float(event.timestamp), event.id))
    n_events = len(ordered)

    by_id = {event.id: idx for idx, event in enumerate(ordered)}
    tau_index = by_id.get(tau.id)
    if tau_index is None:
        return None, {
            "tp_candidate_id": tau.id,
            "tp_candidate_rank": int(tau_rank),
            "n_pre": int(n_pre),
            "status": "skipped_tau_not_in_pool",
            "pool_size": int(n_events),
        }

    tau_timestamp = float(tau.timestamp)
    pre_available = sum(1 for event in ordered if float(event.timestamp) < tau_timestamp - EPS)
    post_available = sum(1 for event in ordered if float(event.timestamp) > tau_timestamp + EPS)

    n_setup = _n_setup_count(n_pre=n_pre, k=grammar.min_prefix_elements)
    n_development = n_pre - n_setup
    min_post_needed = max(0, grammar.min_length - (n_pre + 1))

    if n_pre > pre_available:
        return None, {
            "tp_candidate_id": tau.id,
            "tp_candidate_rank": int(tau_rank),
            "n_pre": int(n_pre),
            "status": "skipped_n_pre_exceeds_pre_available",
            "pre_available": int(pre_available),
            "pool_size": int(n_events),
        }
    if (n_pre + 1) > sequence_length_budget:
        return None, {
            "tp_candidate_id": tau.id,
            "tp_candidate_rank": int(tau_rank),
            "n_pre": int(n_pre),
            "status": "skipped_n_pre_exceeds_length_budget",
            "sequence_length_budget": int(sequence_length_budget),
            "pool_size": int(n_events),
        }
    if n_development < grammar.min_prefix_elements:
        return None, {
            "tp_candidate_id": tau.id,
            "tp_candidate_rank": int(tau_rank),
            "n_pre": int(n_pre),
            "status": "skipped_insufficient_development_allocation",
            "n_setup": int(n_setup),
            "n_development": int(n_development),
            "required_development": int(grammar.min_prefix_elements),
            "pool_size": int(n_events),
        }
    if post_available < min_post_needed:
        return None, {
            "tp_candidate_id": tau.id,
            "tp_candidate_rank": int(tau_rank),
            "n_pre": int(n_pre),
            "status": "skipped_insufficient_post_events_for_min_length",
            "post_available": int(post_available),
            "min_post_needed": int(min_post_needed),
            "pool_size": int(n_events),
        }

    hops_from_start, hops_to_tau = _build_min_hops_to_target(
        events=ordered,
        target_index=tau_index,
        max_gap=active_max_gap,
    )

    start = _Label(
        score=0.0,
        slots_used=0,
        first_t=inf,
        last_index=-1,
        pre_count=0,
        tp_selected=False,
        focal_hits=0,
        selected_indices=(),
    )

    frontier: dict[tuple[str, tuple[int, ...]], list[_Label]] = {
        ("__START__", _phase_state(last_phase=0, pre_count=0, tp_selected=False, focal_hits=0)): [start]
    }
    live_labels = 1
    frontier_peak = 1
    labels_generated = 1
    labels_kept = 1
    labels_pruned_dominance = 0
    labels_rejected_dominance = 0
    labels_pruned_need = 0
    labels_pruned_gap_hops = 0
    labels_pruned_length_bound = 0

    for step in range(sequence_length_budget):
        active = [
            label
            for labels in frontier.values()
            for label in labels
            if label.slots_used == step
        ]
        if not active:
            break

        for label in active:
            remaining_slots = sequence_length_budget - label.slots_used

            if not label.tp_selected:
                min_needed = (n_pre - label.pre_count) + 1
            else:
                min_needed = 0
            if min_needed > remaining_slots:
                labels_pruned_need += 1
                continue

            remaining_events_count = n_events - (label.last_index + 1)
            if (label.slots_used + remaining_events_count) < grammar.min_length:
                labels_pruned_length_bound += 1
                continue

            if not label.tp_selected and not isinf(active_max_gap):
                if label.last_index < 0:
                    hop_lb = hops_from_start
                else:
                    hop_lb = hops_to_tau[label.last_index]
                if hop_lb == inf or hop_lb > remaining_slots:
                    labels_pruned_gap_hops += 1
                    continue

            last_t = (
                None
                if label.last_index < 0
                else float(ordered[label.last_index].timestamp)
            )
            for idx in range(label.last_index + 1, n_events):
                event = ordered[idx]
                timestamp = float(event.timestamp)

                if last_t is not None:
                    if timestamp <= last_t + EPS:
                        continue
                    if not isinf(active_max_gap):
                        gap = timestamp - last_t
                        if gap > active_max_gap + EPS:
                            break

                if idx == tau_index:
                    if label.tp_selected or label.pre_count != n_pre:
                        continue
                    phase = Phase.TURNING_POINT
                    new_pre_count = label.pre_count
                    new_tp_selected = True
                elif timestamp < tau_timestamp - EPS:
                    if label.tp_selected:
                        continue
                    if label.pre_count >= n_pre:
                        continue
                    phase = (
                        Phase.SETUP if label.pre_count < n_setup else Phase.DEVELOPMENT
                    )
                    new_pre_count = label.pre_count + 1
                    new_tp_selected = False
                elif timestamp > tau_timestamp + EPS:
                    if not label.tp_selected:
                        continue
                    phase = Phase.RESOLUTION
                    new_pre_count = label.pre_count
                    new_tp_selected = True
                else:
                    continue

                new_slots_used = label.slots_used + 1
                if new_slots_used > sequence_length_budget:
                    break

                new_first_t = timestamp if label.slots_used == 0 else label.first_t
                new_focal_hits = label.focal_hits + int(focal_actor in event.actors)
                last_phase = int(phase.value)
                state = _phase_state(
                    last_phase=last_phase,
                    pre_count=new_pre_count,
                    tp_selected=new_tp_selected,
                    focal_hits=new_focal_hits,
                )
                bucket = (event.id, state)

                new_label = _Label(
                    score=label.score + float(event.weight),
                    slots_used=new_slots_used,
                    first_t=new_first_t,
                    last_index=idx,
                    pre_count=new_pre_count,
                    tp_selected=new_tp_selected,
                    focal_hits=new_focal_hits,
                    selected_indices=(*label.selected_indices, idx),
                )

                labels_generated += 1
                accepted, removed, rejected = _insert_with_dominance(
                    frontier=frontier,
                    bucket=bucket,
                    label=new_label,
                )
                labels_pruned_dominance += removed
                labels_rejected_dominance += rejected
                if accepted:
                    labels_kept += 1
                    live_labels += 1 - removed
                    if live_labels > frontier_peak:
                        frontier_peak = live_labels

    best_sequence: ExtractedSequence | None = None
    best_score = -inf
    best_any: ExtractedSequence | None = None
    best_any_score = -inf
    evaluated_final_labels = 0
    valid_final_labels = 0

    for labels in frontier.values():
        for label in labels:
            if not label.tp_selected:
                continue
            if label.slots_used == 0:
                continue
            phases = _reconstruct_phases(
                selected_indices=label.selected_indices,
                events=ordered,
                tau_index=tau_index,
                tau_timestamp=tau_timestamp,
                n_setup=n_setup,
            )
            if not phases:
                continue
            evaluated_final_labels += 1

            candidate = _finalize_candidate(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                scoring_fn=scoring_fn,
                events=ordered,
                selected_indices=label.selected_indices,
                phases=phases,
                metadata={
                    "solver": "tp_conditioned_rcspp",
                    "tp_candidate_id": tau.id,
                    "tp_candidate_rank": int(tau_rank),
                    "forced_n_pre": int(n_pre),
                    "forced_n_setup": int(n_setup),
                    "active_max_gap": (
                        None if isinf(active_max_gap) else float(active_max_gap)
                    ),
                    "pool_size": int(n_events),
                    "labels_generated": int(labels_generated),
                    "labels_kept": int(labels_kept),
                    "labels_live_final": int(live_labels),
                    "frontier_peak": int(frontier_peak),
                    "labels_pruned_dominance": int(labels_pruned_dominance),
                    "labels_rejected_dominance": int(labels_rejected_dominance),
                    "labels_pruned_need": int(labels_pruned_need),
                    "labels_pruned_gap_hops": int(labels_pruned_gap_hops),
                    "labels_pruned_length_bound": int(labels_pruned_length_bound),
                },
            )
            if candidate.score > best_any_score + EPS:
                best_any_score = candidate.score
                best_any = candidate
            if not candidate.valid:
                continue
            valid_final_labels += 1
            if candidate.score > best_score + EPS:
                best_score = candidate.score
                best_sequence = candidate

    diagnostics = {
        "tp_candidate_id": tau.id,
        "tp_candidate_rank": int(tau_rank),
        "tp_timestamp": float(tau_timestamp),
        "tp_weight": float(tau.weight),
        "n_pre": int(n_pre),
        "n_setup": int(n_setup),
        "n_development": int(n_development),
        "pre_available": int(pre_available),
        "post_available": int(post_available),
        "min_post_needed": int(min_post_needed),
        "pool_size": int(n_events),
        "sequence_length_budget": int(sequence_length_budget),
        "labels_generated": int(labels_generated),
        "labels_kept": int(labels_kept),
        "labels_live_final": int(live_labels),
        "frontier_peak": int(frontier_peak),
        "labels_pruned_dominance": int(labels_pruned_dominance),
        "labels_rejected_dominance": int(labels_rejected_dominance),
        "labels_pruned_need": int(labels_pruned_need),
        "labels_pruned_gap_hops": int(labels_pruned_gap_hops),
        "labels_pruned_length_bound": int(labels_pruned_length_bound),
        "evaluated_final_labels": int(evaluated_final_labels),
        "valid_final_labels": int(valid_final_labels),
        "best_any_found": bool(best_any is not None),
        "best_any_score": (None if best_any is None else float(best_any.score)),
        "best_any_valid": (None if best_any is None else bool(best_any.valid)),
        "best_any_violations": (
            None if best_any is None else list(best_any.violations)
        ),
        "best_any_event_ids": (
            None if best_any is None else [event.id for event in best_any.events]
        ),
        "best_any_phases": (
            None if best_any is None else [phase.name for phase in best_any.phases]
        ),
        "status": ("valid_found" if best_sequence is not None else "no_valid_label"),
    }
    return best_sequence, diagnostics


def tp_conditioned_solve(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    M: int = 10,
    max_gap: float | None = None,
    pool_strategy: str = "injection",
    max_sequence_length: int = 20,
    injection_top_n: int = 40,
    min_injection_position: float = 0.0,
    scoring_fn: Callable[[ExtractedSequence], float] = tp_weighted_score,
    tp_candidate_ids: tuple[str, ...] | None = None,
) -> tuple[ExtractedSequence | None, dict]:
    """Solve extraction by TP conditioning + label-setting DP."""

    if M <= 0:
        raise ValueError("M must be >= 1")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be >= 1")

    sequence_length_budget = int(min(max_sequence_length, grammar.max_length))
    active_max_gap = float(grammar.max_temporal_gap) if max_gap is None else float(max_gap)
    build_pool = _pool_builder(pool_strategy)

    focal_candidates = _sorted_focal_candidates(graph, focal_actor)
    if not focal_candidates:
        diagnostics = {
            "solver": "tp_conditioned_rcspp",
            "status": "no_focal_tp_candidates",
            "n_focal_events_total": 0,
            "n_tp_candidates_considered": 0,
            "n_dp_runs": 0,
        }
        return None, diagnostics

    focal_rank_by_id = {event.id: (idx + 1) for idx, event in enumerate(focal_candidates)}
    if tp_candidate_ids is not None:
        requested = [str(event_id) for event_id in tp_candidate_ids]
        focal_by_id = {event.id: event for event in focal_candidates}
        tp_candidates = [focal_by_id[event_id] for event_id in requested if event_id in focal_by_id]
    else:
        tp_candidates = focal_candidates[: min(M, len(focal_candidates))]
    by_id = {event.id: event for event in graph.events}

    total_labels_generated = 0
    total_labels_kept = 0
    total_pruned_dominance = 0
    total_rejected_dominance = 0
    total_pruned_need = 0
    total_pruned_gap_hops = 0
    total_pruned_length_bound = 0
    total_evaluated_final_labels = 0
    total_valid_final_labels = 0
    n_dp_runs = 0
    n_valid_candidates = 0

    best_sequence: ExtractedSequence | None = None
    best_score = -inf
    best_run_diag: dict | None = None

    tp_summaries: list[dict] = []

    for tau in tp_candidates:
        tau_rank = int(focal_rank_by_id.get(tau.id, -1))
        if pool_strategy == "filtered_injection":
            pool_ids = build_pool(
                graph=graph,
                anchor_id=tau.id,
                focal_actor=focal_actor,
                max_depth=3,
                injection_top_n=injection_top_n,
                min_position=min_injection_position,
            )
        elif pool_strategy == "injection":
            pool_ids = build_pool(
                graph=graph,
                anchor_id=tau.id,
                focal_actor=focal_actor,
                max_depth=3,
                injection_top_n=injection_top_n,
            )
        else:
            pool_ids = build_pool(
                graph=graph,
                anchor_id=tau.id,
                focal_actor=focal_actor,
                max_depth=3,
            )

        if tau.id not in pool_ids:
            pool_ids.add(tau.id)

        pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        pool_events = sorted(pool_events, key=lambda event: (float(event.timestamp), event.id))

        tau_timestamp = float(tau.timestamp)
        pre_available = sum(1 for event in pool_events if float(event.timestamp) < tau_timestamp - EPS)
        post_available = sum(1 for event in pool_events if float(event.timestamp) > tau_timestamp + EPS)

        n_pre_min = max(grammar.min_prefix_elements, 0)
        n_pre_max = min(pre_available, max(0, sequence_length_budget - 1))

        tp_total_labels_generated = 0
        tp_total_labels_kept = 0
        tp_total_pruned_dominance = 0
        tp_total_rejected_dominance = 0
        tp_total_pruned_need = 0
        tp_total_pruned_gap_hops = 0
        tp_total_pruned_length_bound = 0
        tp_total_evaluated_final_labels = 0
        tp_total_valid_final_labels = 0
        tp_dp_runs = 0
        tp_best_score = -inf
        tp_best_n_pre: int | None = None

        if n_pre_min <= n_pre_max:
            for n_pre in range(n_pre_min, n_pre_max + 1):
                candidate, diag = _solve_fixed_tau_and_n_pre(
                    graph=graph,
                    focal_actor=focal_actor,
                    grammar=grammar,
                    tau=tau,
                    tau_rank=tau_rank,
                    pool_events=pool_events,
                    n_pre=n_pre,
                    sequence_length_budget=sequence_length_budget,
                    active_max_gap=active_max_gap,
                    scoring_fn=scoring_fn,
                )
                if str(diag.get("status", "")).startswith("skipped_"):
                    continue

                n_dp_runs += 1
                tp_dp_runs += 1

                run_labels_generated = int(diag.get("labels_generated", 0))
                run_labels_kept = int(diag.get("labels_kept", 0))
                run_pruned_dominance = int(diag.get("labels_pruned_dominance", 0))
                run_rejected_dominance = int(diag.get("labels_rejected_dominance", 0))
                run_pruned_need = int(diag.get("labels_pruned_need", 0))
                run_pruned_gap_hops = int(diag.get("labels_pruned_gap_hops", 0))
                run_pruned_length_bound = int(diag.get("labels_pruned_length_bound", 0))
                run_eval_final = int(diag.get("evaluated_final_labels", 0))
                run_valid_final = int(diag.get("valid_final_labels", 0))

                total_labels_generated += run_labels_generated
                total_labels_kept += run_labels_kept
                total_pruned_dominance += run_pruned_dominance
                total_rejected_dominance += run_rejected_dominance
                total_pruned_need += run_pruned_need
                total_pruned_gap_hops += run_pruned_gap_hops
                total_pruned_length_bound += run_pruned_length_bound
                total_evaluated_final_labels += run_eval_final
                total_valid_final_labels += run_valid_final

                tp_total_labels_generated += run_labels_generated
                tp_total_labels_kept += run_labels_kept
                tp_total_pruned_dominance += run_pruned_dominance
                tp_total_rejected_dominance += run_rejected_dominance
                tp_total_pruned_need += run_pruned_need
                tp_total_pruned_gap_hops += run_pruned_gap_hops
                tp_total_pruned_length_bound += run_pruned_length_bound
                tp_total_evaluated_final_labels += run_eval_final
                tp_total_valid_final_labels += run_valid_final

                if candidate is not None:
                    n_valid_candidates += 1
                    if candidate.score > tp_best_score + EPS:
                        tp_best_score = candidate.score
                        tp_best_n_pre = n_pre
                    if candidate.score > best_score + EPS:
                        best_score = candidate.score
                        best_sequence = candidate
                        best_run_diag = diag

        tp_summaries.append(
            {
                "tp_id": tau.id,
                "tp_rank": int(tau_rank),
                "tp_weight": float(tau.weight),
                "tp_timestamp": float(tau_timestamp),
                "pool_size": int(len(pool_events)),
                "pre_available": int(pre_available),
                "post_available": int(post_available),
                "n_pre_min": int(n_pre_min),
                "n_pre_max": int(n_pre_max),
                "dp_runs": int(tp_dp_runs),
                "labels_generated": int(tp_total_labels_generated),
                "labels_kept": int(tp_total_labels_kept),
                "labels_pruned_dominance": int(tp_total_pruned_dominance),
                "labels_rejected_dominance": int(tp_total_rejected_dominance),
                "labels_pruned_need": int(tp_total_pruned_need),
                "labels_pruned_gap_hops": int(tp_total_pruned_gap_hops),
                "labels_pruned_length_bound": int(tp_total_pruned_length_bound),
                "evaluated_final_labels": int(tp_total_evaluated_final_labels),
                "valid_final_labels": int(tp_total_valid_final_labels),
                "best_valid_for_tp": bool(tp_best_n_pre is not None),
                "best_valid_n_pre": (None if tp_best_n_pre is None else int(tp_best_n_pre)),
                "best_valid_score": (None if tp_best_n_pre is None else float(tp_best_score)),
            }
        )

    diagnostics = {
        "solver": "tp_conditioned_rcspp",
        "status": ("valid_solution_found" if best_sequence is not None else "no_valid_sequence_found"),
        "M": int(M),
        "n_focal_events_total": int(len(focal_candidates)),
        "n_tp_candidates_considered": int(len(tp_candidates)),
        "tp_candidate_ids": [event.id for event in tp_candidates],
        "sequence_length_budget": int(sequence_length_budget),
        "active_max_gap": (None if isinf(active_max_gap) else float(active_max_gap)),
        "pool_strategy": pool_strategy,
        "injection_top_n": int(injection_top_n),
        "tp_candidate_ids_override": (
            None if tp_candidate_ids is None else [str(event_id) for event_id in tp_candidate_ids]
        ),
        "n_dp_runs": int(n_dp_runs),
        "n_valid_candidates": int(n_valid_candidates),
        "total_labels_generated": int(total_labels_generated),
        "total_labels_kept": int(total_labels_kept),
        "total_labels_pruned_dominance": int(total_pruned_dominance),
        "total_labels_rejected_dominance": int(total_rejected_dominance),
        "total_labels_pruned_need": int(total_pruned_need),
        "total_labels_pruned_gap_hops": int(total_pruned_gap_hops),
        "total_labels_pruned_length_bound": int(total_pruned_length_bound),
        "total_evaluated_final_labels": int(total_evaluated_final_labels),
        "total_valid_final_labels": int(total_valid_final_labels),
        "best_tp_id": (None if best_run_diag is None else best_run_diag.get("tp_candidate_id")),
        "best_tp_rank": (None if best_run_diag is None else best_run_diag.get("tp_candidate_rank")),
        "best_n_pre": (None if best_run_diag is None else best_run_diag.get("n_pre")),
        "best_frontier_peak": (None if best_run_diag is None else best_run_diag.get("frontier_peak")),
        "best_labels_generated": (
            None if best_run_diag is None else best_run_diag.get("labels_generated")
        ),
        "tp_summaries": tp_summaries,
    }

    if best_sequence is None:
        return None, diagnostics

    merged_meta = dict(best_sequence.metadata)
    merged_meta.update(
        {
            "solver": "tp_conditioned_rcspp",
            "outer_M": int(M),
            "n_tp_candidates_considered": int(len(tp_candidates)),
            "n_dp_runs_total": int(n_dp_runs),
            "total_labels_generated": int(total_labels_generated),
            "total_labels_kept": int(total_labels_kept),
            "best_tp_id": diagnostics["best_tp_id"],
            "best_tp_rank": diagnostics["best_tp_rank"],
            "best_n_pre": diagnostics["best_n_pre"],
            "active_max_gap": diagnostics["active_max_gap"],
        }
    )
    best_sequence = replace(best_sequence, metadata=merged_meta)
    return best_sequence, diagnostics
