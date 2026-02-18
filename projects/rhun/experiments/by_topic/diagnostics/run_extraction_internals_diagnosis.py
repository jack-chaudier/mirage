"""
Extraction internals diagnosis for prefix-dominance-miss false negatives.

Pre-check findings from extraction pipeline review:
1) `greedy_extract` (search.py) does NOT build sequences left-to-right by phase.
   It also does NOT explicitly pick TP first and then fill prefix/suffix.
   Instead, per anchor it does:
   a) Build a candidate pool (`pool_construction.py`).
   b) Downsample that pool into an event set (time-sorted afterward).
   c) Run `classify_phases` on the full selected set.
   d) The TP is assigned post hoc as argmax-weight within that selected set.

2) `phase_classifier.py` is purely static/post-selection labeling:
   - TP index = argmax weight in selected sequence.
   - Prefix before TP: first ceil(20%) labeled SETUP, rest DEVELOPMENT.
   - Therefore, with exactly one selected pre-TP event, DEVELOPMENT count is always 0.

3) `pool_construction.py` with strategy="injection":
   - Pool = BFS neighborhood around anchor (depth-limited) UNION top focal-actor
     events by weight (`injection_top_n`).
   - Injection can include temporally early focal events regardless of connectivity.

4) `validator.py` defines `insufficient_development` as count of DEVELOPMENT
   labels before the first TP being < k. This is computed on phase labels of the
   selected sequence only.

Implication for diagnosis:
- Because TP is assigned after selection, the key drop points are:
  (a) Pool composition (missing pre-TP development-eligible events),
  (b) Selection/downsampling (eligible events in pool but not in chosen sequence),
  (c) Classification edge-case (only one selected pre-TP event => SETUP only).
- "Causal reachability" is analyzed as a subcase where causally connected
  development-eligible events exist before TP but fail to appear in any pool.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from math import ceil
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_phases
from rhun.extraction.pool_construction import injection_pool
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.search import greedy_extract
from rhun.extraction.validator import validate
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


INPUT_PATH = Path(__file__).resolve().parent / "output" / "prefix_dominance_test.json"


def _event_lookup(graph: CausalGraph) -> dict[str, Event]:
    return {event.id: event for event in graph.events}


def _downsample_pool_local(
    graph: CausalGraph,
    pool_ids: set[str],
    focal_actor: str,
    anchor_id: str,
    max_sequence_length: int,
) -> tuple[Event, ...]:
    """Replica of search._downsample_pool for diagnostics without editing core source."""
    if not pool_ids:
        return ()

    by_id = _event_lookup(graph)
    reverse = graph.build_reverse_adjacency()

    kept_ids: set[str] = set()
    stack = [anchor_id]
    while stack and len(kept_ids) < max_sequence_length:
        current = stack.pop()
        if current not in pool_ids or current in kept_ids:
            continue
        kept_ids.add(current)
        for parent_id in reverse.get(current, []):
            if parent_id in pool_ids and parent_id not in kept_ids:
                stack.append(parent_id)

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


def _candidate_from_anchor(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    anchor: Event,
    max_sequence_length: int,
    injection_top_n: int,
) -> dict:
    pool_ids = injection_pool(
        graph=graph,
        anchor_id=anchor.id,
        focal_actor=focal_actor,
        max_depth=3,
        injection_top_n=injection_top_n,
    )

    candidate_events = _downsample_pool_local(
        graph=graph,
        pool_ids=pool_ids,
        focal_actor=focal_actor,
        anchor_id=anchor.id,
        max_sequence_length=max_sequence_length,
    )

    if not candidate_events:
        return {
            "anchor_id": anchor.id,
            "pool_ids": pool_ids,
            "candidate": None,
            "score": float("-inf"),
            "valid": False,
            "violations": ["empty_candidate_set"],
        }

    phases = classify_phases(candidate_events)
    candidate = ExtractedSequence(
        events=candidate_events,
        phases=phases,
        focal_actor=focal_actor,
    )
    valid, violations = validate(candidate, grammar, graph)
    score = tp_weighted_score(candidate)

    return {
        "anchor_id": anchor.id,
        "pool_ids": pool_ids,
        "candidate": candidate,
        "score": float(score),
        "valid": bool(valid),
        "violations": list(violations),
    }


def _select_best_candidate(candidates: list[dict]) -> dict | None:
    nonempty = [candidate for candidate in candidates if candidate["candidate"] is not None]
    if not nonempty:
        return None

    valid = [candidate for candidate in nonempty if candidate["valid"]]
    if valid:
        return max(valid, key=lambda row: row["score"])
    return max(nonempty, key=lambda row: row["score"])


def _phase_by_event_id(sequence: ExtractedSequence) -> dict[str, Phase]:
    return {event.id: phase for event, phase in zip(sequence.events, sequence.phases)}


def _connected_component(graph: CausalGraph, start_id: str) -> set[str]:
    adjacency = graph.build_adjacency()
    reverse = graph.build_reverse_adjacency()

    visited = {start_id}
    queue: deque[str] = deque([start_id])
    while queue:
        current = queue.popleft()
        neighbors = adjacency.get(current, []) + reverse.get(current, [])
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _mechanism_bucket(case: dict) -> tuple[str, str]:
    """Return (mechanism_bucket, mechanism_family)."""
    dev_in_sequence = case["n_development_before_tp_all"]
    n_focal_before = case["n_focal_before_tp_graph"]
    n_dev_eligible_graph = case["n_focal_dev_eligible_graph"]
    n_dev_eligible_any_pool = case["n_focal_dev_eligible_in_any_pool"]
    n_dev_eligible_best_seq = case["n_focal_dev_eligible_in_best_sequence"]
    n_dev_eligible_causal = case["n_focal_dev_eligible_causal_connected"]
    n_focal_before_any_pool = case["n_focal_before_tp_in_any_pool"]
    n_focal_before_best_seq = case["n_focal_before_tp_in_best_sequence"]
    n_before_tp_selected_all = case["n_before_tp_selected_all"]

    if dev_in_sequence > 0:
        return "dev_present_other_violation", "other"

    if n_focal_before == 0:
        return "no_focal_prefix_events", "other"

    if n_dev_eligible_graph == 0:
        if n_focal_before_any_pool == 0:
            return "pool_missing_only_prefix_event", "pool_construction"
        if n_focal_before_best_seq == 0:
            return "selection_skipped_only_prefix_event", "event_selection"
        return "phase_classifier_setup_only_prefix", "phase_classification"

    if n_dev_eligible_any_pool == 0:
        if n_dev_eligible_causal > 0:
            return "causal_dev_events_not_reaching_pool", "causal_reachability"
        return "pool_missing_dev_eligible_events", "pool_construction"

    if n_dev_eligible_best_seq == 0:
        if n_before_tp_selected_all <= 1:
            return "single_prefix_event_setup_after_selection", "phase_classification"
        return "dev_eligible_in_pool_but_not_selected", "event_selection"

    if n_before_tp_selected_all <= 1:
        return "selected_dev_eligible_reclassified_setup", "phase_classification"

    return "unclassified_zero_dev_pattern", "other"


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    agg = data["aggregate"]
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"Cases analyzed: {agg['n_cases']}",
        f"Zero-development cases: {agg['zero_dev_case_count']}",
        f"Mean/median focal events before TP: {agg['n_focal_before_tp_mean']:.3f} / {agg['n_focal_before_tp_median']:.3f}",
        f"Mean/median focal dev-eligible (graph-level) before TP: {agg['n_focal_dev_eligible_mean']:.3f} / {agg['n_focal_dev_eligible_median']:.3f}",
        f"Mean/median focal dev-eligible in any pool: {agg['n_focal_dev_eligible_in_any_pool_mean']:.3f} / {agg['n_focal_dev_eligible_in_any_pool_median']:.3f}",
        "",
        "## Mechanism Buckets",
        "",
    ]
    for key, value in sorted(agg["mechanism_bucket_distribution"].items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Mechanism Families", ""])
    for key, value in sorted(agg["mechanism_family_distribution"].items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Zero-Development Families", ""])
    for key, value in sorted(agg["zero_dev_mechanism_family_distribution"].items()):
        lines.append(f"- {key}: {value}")

    lines.append("")
    return "\n".join(lines)


def run_extraction_internals_diagnosis() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    source = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    params = source["metadata"]["parameters"]
    settings = params["grammar"]

    # Pull generation/extraction settings from metadata parameters.
    n_events = int(params["n_events"])
    n_actors = int(params["n_actors"])
    default_actor = str(params["focal_actor"])
    max_sequence_length = int(params["max_sequence_length"])
    grammar = GrammarConfig(**settings)

    raw_cases = source["results"]["per_case"]
    target_cases = [
        case
        for case in raw_cases
        if (not bool(case.get("prefix_dominance_holds", True)))
        and (not bool(case.get("greedy_valid", True)))
    ]

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    mechanism_bucket_counter: Counter[str] = Counter()
    mechanism_family_counter: Counter[str] = Counter()

    n_focal_before_list: list[int] = []
    n_focal_dev_eligible_list: list[int] = []
    n_focal_dev_eligible_any_pool_list: list[int] = []
    n_focal_dev_classified_dev_list: list[int] = []
    n_focal_classified_setup_list: list[int] = []
    n_focal_classified_not_selected_list: list[int] = []

    per_case_records: list[dict] = []
    mismatch_count = 0

    for case in sorted(target_cases, key=lambda row: (float(row["epsilon"]), int(row["seed"]))):
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case.get("focal_actor", default_actor))

        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )

        anchors = sorted(
            graph.events,
            key=lambda event: (event.weight, -event.timestamp),
            reverse=True,
        )[:8]

        candidate_rows: list[dict] = []
        any_pool_ids: set[str] = set()
        for anchor in anchors:
            row = _candidate_from_anchor(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                anchor=anchor,
                max_sequence_length=max_sequence_length,
                injection_top_n=40,
            )
            any_pool_ids.update(row["pool_ids"])
            candidate_rows.append(row)

        best = _select_best_candidate(candidate_rows)
        if best is None:
            continue

        best_sequence: ExtractedSequence = best["candidate"]
        best_pool_ids = set(best["pool_ids"])

        # Consistency check against production greedy implementation.
        prod_result = greedy_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_strategy="injection",
            max_sequence_length=max_sequence_length,
        )
        if [event.id for event in prod_result.events] != [event.id for event in best_sequence.events]:
            mismatch_count += 1

        phase_by_id = _phase_by_event_id(best_sequence)
        sequence_ids = [event.id for event in best_sequence.events]

        tp = best_sequence.turning_point
        if tp is None:
            continue

        tp_index_global = next(i for i, event in enumerate(graph.events) if event.id == tp.id)

        focal_events = tuple(event for event in graph.events if focal_actor in event.actors)
        focal_before_tp = tuple(event for event in focal_events if event.timestamp < tp.timestamp)

        n_focal_before = len(focal_before_tp)
        n_setup_graph = ceil(n_focal_before * 0.2) if n_focal_before > 0 else 0
        focal_dev_eligible = tuple(event for idx, event in enumerate(focal_before_tp) if idx >= n_setup_graph)

        component = _connected_component(graph, tp.id)
        focal_dev_eligible_causal = tuple(event for event in focal_dev_eligible if event.id in component)

        before_tp_selected_all = [
            event for event in best_sequence.events if event.timestamp < tp.timestamp
        ]
        n_dev_before_tp_all = sum(
            1 for phase in best_sequence.phases[: len(before_tp_selected_all)] if phase == Phase.DEVELOPMENT
        )

        focal_before_details = []
        for idx, event in enumerate(focal_before_tp):
            seq_phase = phase_by_id.get(event.id)
            focal_before_details.append(
                {
                    "id": event.id,
                    "timestamp": float(event.timestamp),
                    "weight": float(event.weight),
                    "global_index": int(next(i for i, ev in enumerate(graph.events) if ev.id == event.id)),
                    "focal_prefix_index": idx,
                    "graph_prefix_label": "SETUP" if idx < n_setup_graph else "DEVELOPMENT_ELIGIBLE",
                    "in_any_pool": bool(event.id in any_pool_ids),
                    "in_best_pool": bool(event.id in best_pool_ids),
                    "in_best_sequence": bool(event.id in phase_by_id),
                    "sequence_phase": seq_phase.name if seq_phase is not None else "NOT_SELECTED",
                    "causal_connected_to_tp": bool(event.id in component),
                }
            )

        n_focal_before_in_any_pool = sum(1 for event in focal_before_tp if event.id in any_pool_ids)
        n_focal_before_in_best_pool = sum(1 for event in focal_before_tp if event.id in best_pool_ids)
        n_focal_before_in_best_sequence = sum(1 for event in focal_before_tp if event.id in phase_by_id)

        n_focal_dev_eligible = len(focal_dev_eligible)
        n_focal_dev_eligible_any_pool = sum(1 for event in focal_dev_eligible if event.id in any_pool_ids)
        n_focal_dev_eligible_best_pool = sum(1 for event in focal_dev_eligible if event.id in best_pool_ids)
        n_focal_dev_eligible_best_sequence = sum(
            1 for event in focal_dev_eligible if event.id in phase_by_id
        )
        n_focal_dev_eligible_causal = len(focal_dev_eligible_causal)

        n_focal_classified_development = sum(
            1
            for row in focal_before_details
            if row["sequence_phase"] == "DEVELOPMENT"
        )
        n_focal_classified_setup = sum(
            1 for row in focal_before_details if row["sequence_phase"] == "SETUP"
        )
        n_focal_classified_not_selected = sum(
            1 for row in focal_before_details if row["sequence_phase"] == "NOT_SELECTED"
        )

        case_record = {
            "epsilon": epsilon,
            "seed": seed,
            "focal_actor": focal_actor,
            "tp": {
                "id": tp.id,
                "weight": float(tp.weight),
                "timestamp": float(tp.timestamp),
                "global_index": int(tp_index_global),
                "normalized_position": float(graph.global_position(tp)),
            },
            "greedy_candidate": {
                "anchor_id": best["anchor_id"],
                "score": float(best["score"]),
                "valid": bool(best["valid"]),
                "violations": list(best["violations"]),
                "sequence": [
                    {
                        "id": event.id,
                        "timestamp": float(event.timestamp),
                        "weight": float(event.weight),
                        "phase": phase.name,
                        "focal_actor_involved": bool(focal_actor in event.actors),
                    }
                    for event, phase in zip(best_sequence.events, best_sequence.phases)
                ],
            },
            "n_before_tp_selected_all": len(before_tp_selected_all),
            "n_development_before_tp_all": int(n_dev_before_tp_all),
            "n_focal_before_tp_graph": n_focal_before,
            "n_focal_before_tp_in_any_pool": n_focal_before_in_any_pool,
            "n_focal_before_tp_in_best_pool": n_focal_before_in_best_pool,
            "n_focal_before_tp_in_best_sequence": n_focal_before_in_best_sequence,
            "n_focal_dev_eligible_graph": n_focal_dev_eligible,
            "n_focal_dev_eligible_in_any_pool": n_focal_dev_eligible_any_pool,
            "n_focal_dev_eligible_in_best_pool": n_focal_dev_eligible_best_pool,
            "n_focal_dev_eligible_in_best_sequence": n_focal_dev_eligible_best_sequence,
            "n_focal_dev_eligible_causal_connected": n_focal_dev_eligible_causal,
            "n_focal_before_tp_classified_development": n_focal_classified_development,
            "n_focal_before_tp_classified_setup": n_focal_classified_setup,
            "n_focal_before_tp_classified_not_selected": n_focal_classified_not_selected,
            "focal_before_tp_phase_table": focal_before_details,
            "candidate_pool_coverage": {
                "n_anchors": len(anchors),
                "any_pool_size": len(any_pool_ids),
                "best_pool_size": len(best_pool_ids),
                "all_anchor_pool_sizes": [len(row["pool_ids"]) for row in candidate_rows],
            },
            "candidate_consistency_with_production_greedy": {
                "sequence_match": [event.id for event in prod_result.events] == sequence_ids,
                "production_sequence_ids": [event.id for event in prod_result.events],
                "diagnostic_sequence_ids": sequence_ids,
            },
        }

        bucket, family = _mechanism_bucket(case_record)
        case_record["mechanism_bucket"] = bucket
        case_record["mechanism_family"] = family

        mechanism_bucket_counter[bucket] += 1
        mechanism_family_counter[family] += 1

        n_focal_before_list.append(n_focal_before)
        n_focal_dev_eligible_list.append(n_focal_dev_eligible)
        n_focal_dev_eligible_any_pool_list.append(n_focal_dev_eligible_any_pool)
        n_focal_dev_classified_dev_list.append(n_focal_classified_development)
        n_focal_classified_setup_list.append(n_focal_classified_setup)
        n_focal_classified_not_selected_list.append(n_focal_classified_not_selected)

        per_case_records.append(case_record)

    n_cases = len(per_case_records)
    zero_dev_cases = [
        case for case in per_case_records if int(case["n_development_before_tp_all"]) == 0
    ]
    zero_dev_bucket_counter = Counter(case["mechanism_bucket"] for case in zero_dev_cases)
    zero_dev_family_counter = Counter(case["mechanism_family"] for case in zero_dev_cases)

    aggregate = {
        "n_cases": n_cases,
        "zero_dev_case_count": len(zero_dev_cases),
        "consistency_mismatch_count": mismatch_count,
        "n_focal_before_tp_mean": mean(n_focal_before_list) if n_focal_before_list else 0.0,
        "n_focal_before_tp_median": median(n_focal_before_list) if n_focal_before_list else 0.0,
        "n_focal_dev_eligible_mean": mean(n_focal_dev_eligible_list)
        if n_focal_dev_eligible_list
        else 0.0,
        "n_focal_dev_eligible_median": median(n_focal_dev_eligible_list)
        if n_focal_dev_eligible_list
        else 0.0,
        "n_focal_dev_eligible_in_any_pool_mean": mean(n_focal_dev_eligible_any_pool_list)
        if n_focal_dev_eligible_any_pool_list
        else 0.0,
        "n_focal_dev_eligible_in_any_pool_median": median(n_focal_dev_eligible_any_pool_list)
        if n_focal_dev_eligible_any_pool_list
        else 0.0,
        "n_focal_classified_development_mean": mean(n_focal_dev_classified_dev_list)
        if n_focal_dev_classified_dev_list
        else 0.0,
        "n_focal_classified_development_median": median(n_focal_dev_classified_dev_list)
        if n_focal_dev_classified_dev_list
        else 0.0,
        "n_focal_classified_setup_mean": mean(n_focal_classified_setup_list)
        if n_focal_classified_setup_list
        else 0.0,
        "n_focal_classified_setup_median": median(n_focal_classified_setup_list)
        if n_focal_classified_setup_list
        else 0.0,
        "n_focal_classified_not_selected_mean": mean(n_focal_classified_not_selected_list)
        if n_focal_classified_not_selected_list
        else 0.0,
        "n_focal_classified_not_selected_median": median(n_focal_classified_not_selected_list)
        if n_focal_classified_not_selected_list
        else 0.0,
        "mechanism_bucket_distribution": dict(sorted(mechanism_bucket_counter.items())),
        "mechanism_family_distribution": dict(sorted(mechanism_family_counter.items())),
        "zero_dev_mechanism_bucket_distribution": dict(sorted(zero_dev_bucket_counter.items())),
        "zero_dev_mechanism_family_distribution": dict(sorted(zero_dev_family_counter.items())),
        "dominant_mechanism_bucket": mechanism_bucket_counter.most_common(1)[0][0]
        if mechanism_bucket_counter
        else None,
        "dominant_zero_dev_mechanism_bucket": zero_dev_bucket_counter.most_common(1)[0][0]
        if zero_dev_bucket_counter
        else None,
    }

    data = {
        "input_path": str(INPUT_PATH),
        "input_case_count": len(raw_cases),
        "filtered_case_count": len(target_cases),
        "aggregate": aggregate,
        "per_case": per_case_records,
    }

    metadata = ExperimentMetadata(
        name="extraction_internals_diagnosis",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=n_cases,
        seed_range=(int(source["metadata"]["seed_range"][0]), int(source["metadata"]["seed_range"][1])),
        parameters={
            "source": "prefix_dominance_test.json",
            "filters": {
                "prefix_dominance_holds": False,
                "greedy_valid": False,
            },
            "pool_strategy": "injection",
            "n_anchors": 8,
            "max_sequence_length": max_sequence_length,
            "injection_top_n": 40,
            "grammar": settings,
        },
    )

    save_results(
        name="extraction_internals_diagnosis",
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )

    print(f"Cases analyzed: {aggregate['n_cases']}")
    print(f"Zero-development cases: {aggregate['zero_dev_case_count']}")
    print(
        f"Focal events before TP (mean/median): "
        f"{aggregate['n_focal_before_tp_mean']:.3f}/{aggregate['n_focal_before_tp_median']:.3f}"
    )
    print(
        f"Focal dev-eligible before TP in any pool (mean/median): "
        f"{aggregate['n_focal_dev_eligible_in_any_pool_mean']:.3f}/"
        f"{aggregate['n_focal_dev_eligible_in_any_pool_median']:.3f}"
    )
    print(
        f"Focal before-TP classified as DEVELOPMENT (mean/median): "
        f"{aggregate['n_focal_classified_development_mean']:.3f}/"
        f"{aggregate['n_focal_classified_development_median']:.3f}"
    )

    print("")
    print("Mechanism bucket distribution:")
    for bucket, count in sorted(aggregate["mechanism_bucket_distribution"].items()):
        print(f"  {bucket}: {count}")

    print("")
    print("Mechanism family distribution:")
    for family, count in sorted(aggregate["mechanism_family_distribution"].items()):
        print(f"  {family}: {count}")

    print("")
    print(f"Dominant failure pattern: {aggregate['dominant_mechanism_bucket']}")
    print(
        f"Dominant zero-development pattern: "
        f"{aggregate['dominant_zero_dev_mechanism_bucket']}"
    )
    print(f"Consistency mismatches vs production greedy: {aggregate['consistency_mismatch_count']}")

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_extraction_internals_diagnosis()
