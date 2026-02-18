"""Diagnose pool-construction bottlenecks on beam-unreachable false negatives.

Pre-check findings from code review (required context):
- `greedy_extract` (`rhun/extraction/search.py`) picks top-weight anchors, builds one pool per
  anchor via `pool_construction`, downsamples each pool into one candidate sequence, then validates.
- `beam_search_extract` uses the exact same anchors and the exact same per-anchor pool construction
  call path as greedy (`bfs` / `injection` / `filtered_injection`); it only explores more candidate
  subsets *within those pools*.
- Therefore, if beam saturates while remaining invalid, the bottleneck is likely pool membership
  (events never entering any candidate pool), not beam width.
- `oracle_extract` does not use pool construction. It forces each focal-actor event as TP and builds
  candidates from global focal-actor events before/after TP, so oracle can succeed even when greedy/
  beam pools are temporally compressed or missing key events.

This script diagnoses the 57 cases where beam width 16 is still invalid.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from math import inf
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.pool_construction import bfs_pool, filtered_injection_pool, injection_pool
from rhun.extraction.search import oracle_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event


BEAM_SWEEP_PATH = Path(__file__).resolve().parent / "output" / "beam_search_sweep.json"

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_DEPTH = 3
INJECTION_TOP_N = 40


def _pool_builder(strategy: str):
    if strategy == "bfs":
        return bfs_pool
    if strategy == "injection":
        return injection_pool
    if strategy == "filtered_injection":
        return filtered_injection_pool
    raise ValueError(f"Unknown pool strategy: {strategy}")


def _event_index(graph: CausalGraph) -> dict[str, int]:
    return {event.id: idx for idx, event in enumerate(graph.events)}


def _span(events: list[Event] | tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    return float(events[-1].timestamp - events[0].timestamp)


def _span_fraction(span: float, duration: float) -> float:
    if duration <= 0.0:
        return 0.0
    return span / duration


def _actor_weight_rank(graph: CausalGraph, focal_actor: str) -> dict[str, int]:
    actor_events = sorted(
        graph.events_for_actor(focal_actor),
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )
    return {event.id: rank for rank, event in enumerate(actor_events, start=1)}


def _undirected_distances(graph: CausalGraph, start_id: str) -> dict[str, int]:
    adjacency = graph.build_adjacency()
    reverse = graph.build_reverse_adjacency()

    distances: dict[str, int] = {start_id: 0}
    queue: deque[str] = deque([start_id])

    while queue:
        current_id = queue.popleft()
        current_distance = distances[current_id]
        neighbors = adjacency.get(current_id, []) + reverse.get(current_id, [])
        for neighbor_id in neighbors:
            if neighbor_id in distances:
                continue
            distances[neighbor_id] = current_distance + 1
            queue.append(neighbor_id)
    return distances


def _oracle_event_payload(
    graph: CausalGraph,
    event: Event,
    phase_name: str,
    focal_actor: str,
    index_map: dict[str, int],
) -> dict:
    return {
        "id": event.id,
        "weight": float(event.weight),
        "timestamp": float(event.timestamp),
        "global_index": int(index_map[event.id]),
        "normalized_position": float(graph.global_position(event)),
        "actors": sorted(event.actors),
        "focal_actor_present": bool(focal_actor in event.actors),
        "phase": phase_name,
    }


def _missing_reason(
    event: Event,
    focal_actor: str,
    focal_rank: dict[str, int],
    anchor_distances: dict[str, dict[str, int]],
) -> tuple[str, float | None, str | None]:
    min_distance = inf
    min_anchor: str | None = None
    for anchor_id, dist in anchor_distances.items():
        d = dist.get(event.id, inf)
        if d < min_distance:
            min_distance = d
            min_anchor = anchor_id

    is_focal = focal_actor in event.actors
    rank = focal_rank.get(event.id)
    injection_eligible = bool(is_focal and rank is not None and rank <= INJECTION_TOP_N)

    if injection_eligible:
        return "unexpected_missing_injection_eligible", None if min_distance == inf else float(min_distance), min_anchor

    if not is_focal:
        if min_distance == inf:
            return "non_focal_disconnected_from_anchors", None, min_anchor
        if min_distance > MAX_DEPTH:
            return "non_focal_outside_bfs_radius", float(min_distance), min_anchor
        return "unexpected_missing_non_focal_within_bfs", float(min_distance), min_anchor

    if rank is None:
        if min_distance == inf:
            return "focal_rank_missing_disconnected", None, min_anchor
        if min_distance > MAX_DEPTH:
            return "focal_rank_missing_outside_bfs", float(min_distance), min_anchor
        return "unexpected_missing_focal_rank_missing_within_bfs", float(min_distance), min_anchor

    if min_distance == inf:
        return "focal_not_in_top_n_disconnected_from_anchors", None, min_anchor
    if min_distance > MAX_DEPTH:
        return "focal_not_in_top_n_outside_bfs_radius", float(min_distance), min_anchor
    return "unexpected_missing_focal_not_in_top_n_within_bfs", float(min_distance), min_anchor


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines = [
        f"# {meta.name}",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Target cases: {data['n_cases']}",
        f"Oracle solved: {data['oracle_solved_cases']} / {data['n_cases']}",
        "",
        "## Missing Oracle Events",
        "",
        f"- Missing fraction: {data['missing_event_fraction']:.3f}",
        f"- Mean missing per case: {data['missing_events_per_case_mean']:.2f}",
        f"- Median missing per case: {data['missing_events_per_case_median']:.2f}",
        "",
        "## Missing Phase Distribution",
        "",
        "| phase | count |",
        "|---|---:|",
    ]

    for phase, count in data["missing_phase_counts"].items():
        lines.append(f"| {phase} | {count} |")

    lines.extend(
        [
            "",
            "## Exclusion Reasons",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
    )

    for reason, count in data["missing_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Timespan Feasibility",
            "",
            f"- Cases with full oracle event set inside one anchor pool: {data['cases_with_full_oracle_in_single_pool']} / {data['oracle_solved_cases']}",
            f"- Mean max single-pool oracle coverage: {data['max_oracle_coverage_in_single_pool_mean']:.3f}",
            f"- Cases with union-pool span below grammar threshold: {data['union_timespan_insufficient_cases']} / {data['oracle_solved_cases']}",
            f"- Cases where all anchor pools are below threshold: {data['all_anchor_timespan_insufficient_cases']} / {data['oracle_solved_cases']}",
            f"- Mean oracle span fraction: {data['oracle_span_fraction_mean']:.3f}",
            f"- Mean union-pool span fraction: {data['union_pool_span_fraction_mean']:.3f}",
            f"- Mean max-anchor-pool span fraction: {data['max_anchor_pool_span_fraction_mean']:.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def run_pool_bottleneck_diagnosis() -> dict:
    if not BEAM_SWEEP_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {BEAM_SWEEP_PATH}")

    beam_data = json.loads(BEAM_SWEEP_PATH.read_text(encoding="utf-8"))
    results = beam_data["results"]
    settings = results["settings"]

    grammar = GrammarConfig(**settings["grammar"])
    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    max_sequence_length = int(settings["max_sequence_length"])

    unresolved_cases = [
        case for case in results["per_case"] if not case["results_by_width"].get("16", {}).get("valid", False)
    ]
    unresolved_cases.sort(key=lambda case: (float(case["epsilon"]), int(case["seed"]), str(case["focal_actor"])))

    generator = BurstyGenerator()
    build_pool = _pool_builder(POOL_STRATEGY)

    timer = ExperimentTimer()

    per_case: list[dict] = []

    total_oracle_events = 0
    total_missing_events = 0
    missing_counts_per_case: list[int] = []

    missing_phase_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()
    missing_actor_relation: Counter[str] = Counter()
    missing_positions: list[float] = []
    missing_weights: list[float] = []

    oracle_span_fractions: list[float] = []
    union_span_fractions: list[float] = []
    max_pool_span_fractions: list[float] = []
    max_oracle_coverage_fractions: list[float] = []

    union_insufficient_cases = 0
    all_anchor_insufficient_cases = 0
    oracle_solved_cases = 0
    cases_with_full_oracle_in_single_pool = 0

    for case in unresolved_cases:
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case["focal_actor"])

        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )
        index_map = _event_index(graph)

        oracle_sequence, oracle_diag = oracle_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            max_sequence_length=max_sequence_length,
        )

        case_payload: dict = {
            "epsilon": epsilon,
            "seed": seed,
            "focal_actor": focal_actor,
            "oracle_diagnostics": oracle_diag,
            "beam_width_16_valid": bool(case["results_by_width"]["16"]["valid"]),
        }

        if oracle_sequence is None:
            case_payload.update(
                {
                    "oracle_valid": False,
                    "oracle_events": [],
                    "anchor_pools": [],
                    "missing_oracle_events": [],
                    "timespan": None,
                }
            )
            per_case.append(case_payload)
            continue

        oracle_solved_cases += 1

        anchors = sorted(
            graph.events,
            key=lambda event: (event.weight, -event.timestamp),
            reverse=True,
        )[:N_ANCHORS]

        focal_rank = _actor_weight_rank(graph, focal_actor)

        anchor_pools: list[dict] = []
        union_pool_ids: set[str] = set()
        anchor_distances: dict[str, dict[str, int]] = {}
        anchor_pool_ids: dict[str, set[str]] = {}

        per_anchor_pool_span_fraction: list[float] = []

        for anchor in anchors:
            if POOL_STRATEGY == "filtered_injection":
                pool_ids = build_pool(
                    graph=graph,
                    anchor_id=anchor.id,
                    focal_actor=focal_actor,
                    max_depth=MAX_DEPTH,
                    injection_top_n=INJECTION_TOP_N,
                    min_position=0.0,
                )
            elif POOL_STRATEGY == "injection":
                pool_ids = build_pool(
                    graph=graph,
                    anchor_id=anchor.id,
                    focal_actor=focal_actor,
                    max_depth=MAX_DEPTH,
                    injection_top_n=INJECTION_TOP_N,
                )
            else:
                pool_ids = build_pool(
                    graph=graph,
                    anchor_id=anchor.id,
                    focal_actor=focal_actor,
                    max_depth=MAX_DEPTH,
                )

            union_pool_ids.update(pool_ids)
            anchor_distances[anchor.id] = _undirected_distances(graph, anchor.id)
            anchor_pool_ids[anchor.id] = set(pool_ids)

            pool_events = [graph.event_by_id(event_id) for event_id in pool_ids]
            pool_events = [event for event in pool_events if event is not None]
            pool_events.sort(key=lambda event: event.timestamp)

            pool_span = _span(pool_events)
            pool_span_fraction = _span_fraction(pool_span, graph.duration)
            per_anchor_pool_span_fraction.append(pool_span_fraction)

            anchor_pools.append(
                {
                    "anchor_id": anchor.id,
                    "anchor_weight": float(anchor.weight),
                    "anchor_timestamp": float(anchor.timestamp),
                    "pool_size": len(pool_ids),
                    "pool_span": pool_span,
                    "pool_span_fraction": pool_span_fraction,
                }
            )

        oracle_events_payload: list[dict] = []
        missing_oracle_events: list[dict] = []
        oracle_event_ids = {event.id for event in oracle_sequence.events}

        for event, phase in zip(oracle_sequence.events, oracle_sequence.phases):
            payload = _oracle_event_payload(
                graph=graph,
                event=event,
                phase_name=phase.name,
                focal_actor=focal_actor,
                index_map=index_map,
            )
            payload["in_any_greedy_pool"] = bool(event.id in union_pool_ids)
            oracle_events_payload.append(payload)

            if event.id in union_pool_ids:
                continue

            reason, min_distance, min_anchor = _missing_reason(
                event=event,
                focal_actor=focal_actor,
                focal_rank=focal_rank,
                anchor_distances=anchor_distances,
            )

            missing_payload = dict(payload)
            missing_payload.update(
                {
                    "reason": reason,
                    "focal_weight_rank": focal_rank.get(event.id),
                    "min_anchor_distance": min_distance,
                    "closest_anchor_id": min_anchor,
                }
            )
            missing_oracle_events.append(missing_payload)

            missing_phase_counts[phase.name] += 1
            missing_reason_counts[reason] += 1
            missing_actor_relation["focal" if focal_actor in event.actors else "non_focal"] += 1
            missing_positions.append(float(graph.global_position(event)))
            missing_weights.append(float(event.weight))

        oracle_events_n = len(oracle_sequence.events)
        missing_n = len(missing_oracle_events)
        total_oracle_events += oracle_events_n
        total_missing_events += missing_n
        missing_counts_per_case.append(missing_n)

        overlap_rows: list[tuple[str, int, float]] = []
        for anchor_id, pool_ids in anchor_pool_ids.items():
            overlap_count = len(pool_ids & oracle_event_ids)
            overlap_fraction = (overlap_count / oracle_events_n) if oracle_events_n else 0.0
            overlap_rows.append((anchor_id, overlap_count, overlap_fraction))

        best_overlap_anchor, best_overlap_count, best_overlap_fraction = max(
            overlap_rows,
            key=lambda row: row[1],
            default=(None, 0, 0.0),
        )
        any_anchor_full_oracle = any(
            overlap_count == oracle_events_n for _, overlap_count, _ in overlap_rows
        )
        if any_anchor_full_oracle:
            cases_with_full_oracle_in_single_pool += 1
        max_oracle_coverage_fractions.append(best_overlap_fraction)

        oracle_span = _span(list(oracle_sequence.events))
        oracle_span_fraction = _span_fraction(oracle_span, graph.duration)

        union_pool_events = [graph.event_by_id(event_id) for event_id in union_pool_ids]
        union_pool_events = [event for event in union_pool_events if event is not None]
        union_pool_events.sort(key=lambda event: event.timestamp)

        union_span = _span(union_pool_events)
        union_span_fraction = _span_fraction(union_span, graph.duration)

        max_anchor_span_fraction = max(per_anchor_pool_span_fraction) if per_anchor_pool_span_fraction else 0.0

        required_span_fraction = float(grammar.min_timespan_fraction)
        required_span_abs = required_span_fraction * graph.duration

        union_meets = union_span_fraction >= required_span_fraction
        any_anchor_meets = any(span >= required_span_fraction for span in per_anchor_pool_span_fraction)

        if not union_meets:
            union_insufficient_cases += 1
        if not any_anchor_meets:
            all_anchor_insufficient_cases += 1

        oracle_span_fractions.append(oracle_span_fraction)
        union_span_fractions.append(union_span_fraction)
        max_pool_span_fractions.append(max_anchor_span_fraction)

        case_payload.update(
            {
                "oracle_valid": True,
                "oracle_score": float(oracle_sequence.score),
                "oracle_violations": list(oracle_sequence.violations),
                "oracle_events": oracle_events_payload,
                "missing_oracle_events": missing_oracle_events,
                "missing_count": missing_n,
                "oracle_event_count": oracle_events_n,
                "missing_fraction": (missing_n / oracle_events_n) if oracle_events_n else 0.0,
                "max_oracle_coverage_in_single_pool": {
                    "anchor_id": best_overlap_anchor,
                    "overlap_count": int(best_overlap_count),
                    "overlap_fraction": float(best_overlap_fraction),
                    "any_anchor_contains_all_oracle_events": bool(any_anchor_full_oracle),
                },
                "anchor_pools": anchor_pools,
                "timespan": {
                    "graph_duration": float(graph.duration),
                    "required_span_fraction": required_span_fraction,
                    "required_span_abs": float(required_span_abs),
                    "oracle_span": float(oracle_span),
                    "oracle_span_fraction": float(oracle_span_fraction),
                    "union_pool_span": float(union_span),
                    "union_pool_span_fraction": float(union_span_fraction),
                    "max_anchor_pool_span_fraction": float(max_anchor_span_fraction),
                    "union_pool_meets_requirement": bool(union_meets),
                    "any_anchor_pool_meets_requirement": bool(any_anchor_meets),
                },
            }
        )

        per_case.append(case_payload)

    missing_event_fraction = (total_missing_events / total_oracle_events) if total_oracle_events else 0.0

    result_data = {
        "n_cases": len(unresolved_cases),
        "oracle_solved_cases": oracle_solved_cases,
        "settings": {
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_depth": MAX_DEPTH,
            "injection_top_n": INJECTION_TOP_N,
            "grammar": {
                "min_prefix_elements": grammar.min_prefix_elements,
                "max_phase_regressions": grammar.max_phase_regressions,
                "max_turning_points": grammar.max_turning_points,
                "min_length": grammar.min_length,
                "max_length": grammar.max_length,
                "min_timespan_fraction": grammar.min_timespan_fraction,
                "focal_actor_coverage": grammar.focal_actor_coverage,
            },
        },
        "total_oracle_events": total_oracle_events,
        "total_missing_oracle_events": total_missing_events,
        "missing_event_fraction": missing_event_fraction,
        "missing_events_per_case_mean": mean(missing_counts_per_case) if missing_counts_per_case else 0.0,
        "missing_events_per_case_median": median(missing_counts_per_case) if missing_counts_per_case else 0.0,
        "missing_phase_counts": dict(missing_phase_counts.most_common()),
        "missing_reason_counts": dict(missing_reason_counts.most_common()),
        "missing_actor_relation_counts": dict(missing_actor_relation.most_common()),
        "missing_position_mean": mean(missing_positions) if missing_positions else 0.0,
        "missing_position_median": median(missing_positions) if missing_positions else 0.0,
        "missing_weight_mean": mean(missing_weights) if missing_weights else 0.0,
        "missing_weight_median": median(missing_weights) if missing_weights else 0.0,
        "cases_with_full_oracle_in_single_pool": cases_with_full_oracle_in_single_pool,
        "max_oracle_coverage_in_single_pool_mean": mean(max_oracle_coverage_fractions)
        if max_oracle_coverage_fractions
        else 0.0,
        "max_oracle_coverage_in_single_pool_median": median(max_oracle_coverage_fractions)
        if max_oracle_coverage_fractions
        else 0.0,
        "union_timespan_insufficient_cases": union_insufficient_cases,
        "all_anchor_timespan_insufficient_cases": all_anchor_insufficient_cases,
        "oracle_span_fraction_mean": mean(oracle_span_fractions) if oracle_span_fractions else 0.0,
        "oracle_span_fraction_median": median(oracle_span_fractions) if oracle_span_fractions else 0.0,
        "union_pool_span_fraction_mean": mean(union_span_fractions) if union_span_fractions else 0.0,
        "union_pool_span_fraction_median": median(union_span_fractions) if union_span_fractions else 0.0,
        "max_anchor_pool_span_fraction_mean": mean(max_pool_span_fractions) if max_pool_span_fractions else 0.0,
        "max_anchor_pool_span_fraction_median": median(max_pool_span_fractions) if max_pool_span_fractions else 0.0,
        "per_case": per_case,
    }

    meta = ExperimentMetadata(
        name="pool_bottleneck_diagnosis",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(unresolved_cases),
        n_extractions=len(unresolved_cases),
        seed_range=(
            min((int(case["seed"]) for case in unresolved_cases), default=0),
            max((int(case["seed"]) for case in unresolved_cases), default=0),
        ),
        parameters={
            "source": str(BEAM_SWEEP_PATH.name),
            "beam_unreachable_definition": "beam_width_16_invalid",
        },
    )

    save_results(
        "pool_bottleneck_diagnosis",
        result_data,
        meta,
        summary_formatter=_summary_markdown,
    )

    return {"metadata": meta, "results": result_data}


def _print_summary(results: dict) -> None:
    print("Pool bottleneck diagnosis summary")
    print(f"  cases (beam width 16 still invalid): {results['n_cases']}")
    print(f"  oracle solved: {results['oracle_solved_cases']}/{results['n_cases']}")
    print(
        "  oracle events missing from greedy pools: "
        f"{results['total_missing_oracle_events']}/{results['total_oracle_events']} "
        f"({results['missing_event_fraction']:.3f})"
    )
    print(
        "  missing events per case: "
        f"mean={results['missing_events_per_case_mean']:.2f}, "
        f"median={results['missing_events_per_case_median']:.2f}"
    )
    print("  top exclusion reasons:")
    for reason, count in list(results["missing_reason_counts"].items())[:5]:
        print(f"    - {reason}: {count}")
    print(
        "  timespan feasibility: "
        f"union insufficient={results['union_timespan_insufficient_cases']}/{results['oracle_solved_cases']}, "
        f"all-anchor insufficient={results['all_anchor_timespan_insufficient_cases']}/{results['oracle_solved_cases']}"
    )
    print(
        "  single-pool oracle coverage: "
        f"full-coverage-cases={results['cases_with_full_oracle_in_single_pool']}/{results['oracle_solved_cases']}, "
        f"mean-max-coverage={results['max_oracle_coverage_in_single_pool_mean']:.3f}"
    )
    print(
        "  span fractions: "
        f"oracle_mean={results['oracle_span_fraction_mean']:.3f}, "
        f"union_pool_mean={results['union_pool_span_fraction_mean']:.3f}, "
        f"max_anchor_pool_mean={results['max_anchor_pool_span_fraction_mean']:.3f}"
    )


if __name__ == "__main__":
    payload = run_pool_bottleneck_diagnosis()
    _print_summary(payload["results"])
