"""Test prefix-dominance sufficient condition on false-negative cases."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event
from rhun.theory.theorem import check_precondition


INPUT_PATH = Path(__file__).resolve().parent / "output" / "fn_divergence_analysis.json"


def _relevant_events(graph: CausalGraph, focal_actor: str) -> tuple[Event, ...]:
    """
    Match theorem.py actor-selection semantics.

    check_precondition uses graph.events_for_actor(focal_actor), falling back to
    all events when the focal actor has no events.
    """
    actor_events = graph.events_for_actor(focal_actor)
    return actor_events if actor_events else graph.events


def _argmax_event(events: tuple[Event, ...]) -> Event:
    return max(events, key=lambda event: (event.weight, -event.timestamp))


def _event_payload(
    graph: CausalGraph,
    local_index_by_id: dict[str, int],
    event: Event | None,
) -> dict | None:
    if event is None:
        return None

    global_index = next((idx for idx, e in enumerate(graph.events) if e.id == event.id), None)
    return {
        "id": event.id,
        "weight": float(event.weight),
        "timestamp": float(event.timestamp),
        "global_index": int(global_index) if global_index is not None else None,
        "local_index": int(local_index_by_id[event.id]) if event.id in local_index_by_id else None,
        "normalized_position": float(graph.global_position(event)),
    }


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    aggregate = data["aggregate"]
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"Cases analyzed: {aggregate['n_cases']}",
        f"k (min_prefix_elements): {aggregate['k']}",
        "",
        f"Prefix dominance holds: {aggregate['prefix_dominance_holds_count']} "
        f"({aggregate['prefix_dominance_holds_rate']:.3f})",
        f"Greedy TP in early set: {aggregate['greedy_tp_in_early_count']} "
        f"({aggregate['greedy_tp_in_early_rate']:.3f})",
        f"Prefix dominance misses (condition false but greedy failed): "
        f"{aggregate['prefix_dominance_miss_count']} ({aggregate['prefix_dominance_miss_rate']:.3f})",
        "",
        "## Per-epsilon",
        "",
        "| epsilon | cases | dominance_holds | greedy_tp_in_early | dominance_misses |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in data["per_epsilon_summary"]:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['cases']} | {row['dominance_holds']} | "
            f"{row['greedy_tp_in_early']} | {row['dominance_misses']} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_prefix_dominance_test() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    source = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    settings = source["results"]["settings"]
    all_cases = source["results"]["per_case"]

    grammar = GrammarConfig(**settings["grammar"])
    k = grammar.min_prefix_elements
    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    default_actor = str(settings["focal_actor"])
    max_sequence_length = int(settings["max_sequence_length"])

    false_negative_cases = [
        case
        for case in all_cases
        if (not bool(case.get("recomputed_predicted_failure", True)))
        and (not bool(case.get("recomputed_greedy_valid", True)))
        and bool(case.get("oracle_valid", False))
    ]

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    per_case_records: list[dict] = []

    prefix_dominance_holds_count = 0
    greedy_tp_in_early_count = 0
    prefix_dominance_miss_count = 0
    greedy_tp_not_in_relevant_count = 0

    missing_early_count = 0
    missing_viable_count = 0

    per_epsilon = defaultdict(
        lambda: {
            "cases": 0,
            "dominance_holds": 0,
            "greedy_tp_in_early": 0,
            "dominance_misses": 0,
        }
    )

    for case in sorted(false_negative_cases, key=lambda c: (float(c["epsilon"]), int(c["seed"]))):
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case.get("focal_actor", default_actor))

        graph = generator.generate(
            BurstyConfig(
                seed=seed,
                epsilon=epsilon,
                n_events=n_events,
                n_actors=n_actors,
            )
        )

        relevant = _relevant_events(graph, focal_actor)
        local_index_by_id = {event.id: idx for idx, event in enumerate(relevant)}

        early_events = tuple(event for event in relevant if local_index_by_id[event.id] < k)
        viable_events = tuple(event for event in relevant if local_index_by_id[event.id] >= k)

        early_max = _argmax_event(early_events) if early_events else None
        viable_max = _argmax_event(viable_events) if viable_events else None

        # Proposed sufficient condition: max_weight(early) > max_weight(viable)
        if early_max is not None and viable_max is not None:
            prefix_dominance_holds = bool(early_max.weight > viable_max.weight)
        elif early_max is not None and viable_max is None:
            prefix_dominance_holds = True
        else:
            prefix_dominance_holds = False

        greedy = greedy_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_strategy="injection",
            max_sequence_length=max_sequence_length,
        )
        greedy_tp = greedy.turning_point

        greedy_tp_in_early = bool(greedy_tp is not None and greedy_tp.id in {e.id for e in early_events})
        greedy_tp_in_relevant = bool(greedy_tp is not None and greedy_tp.id in local_index_by_id)

        theorem = check_precondition(graph, focal_actor, grammar)

        if prefix_dominance_holds:
            prefix_dominance_holds_count += 1
        else:
            prefix_dominance_miss_count += 1

        if greedy_tp_in_early:
            greedy_tp_in_early_count += 1

        if not greedy_tp_in_relevant:
            greedy_tp_not_in_relevant_count += 1

        if not early_events:
            missing_early_count += 1
        if not viable_events:
            missing_viable_count += 1

        epsilon_bucket = per_epsilon[epsilon]
        epsilon_bucket["cases"] += 1
        epsilon_bucket["dominance_holds"] += int(prefix_dominance_holds)
        epsilon_bucket["greedy_tp_in_early"] += int(greedy_tp_in_early)
        epsilon_bucket["dominance_misses"] += int(not prefix_dominance_holds)

        per_case_records.append(
            {
                "epsilon": epsilon,
                "seed": seed,
                "focal_actor": focal_actor,
                "k": k,
                "n_relevant_events": len(relevant),
                "n_early_events": len(early_events),
                "n_viable_events": len(viable_events),
                "prefix_dominance_holds": prefix_dominance_holds,
                "theorem_predicted_failure": bool(theorem["predicted_failure"]),
                "theorem_max_weight_index_global": int(theorem["max_weight_index"]),
                "early_max": _event_payload(graph, local_index_by_id, early_max),
                "viable_max": _event_payload(graph, local_index_by_id, viable_max),
                "greedy_tp": _event_payload(graph, local_index_by_id, greedy_tp),
                "greedy_tp_in_early": greedy_tp_in_early,
                "greedy_tp_in_relevant": greedy_tp_in_relevant,
                "greedy_valid": bool(greedy.valid),
                "greedy_violations": list(greedy.violations),
            }
        )

    n_cases = len(per_case_records)
    n_cases_safe = n_cases if n_cases else 1

    aggregate = {
        "n_cases": n_cases,
        "k": k,
        "prefix_dominance_holds_count": prefix_dominance_holds_count,
        "prefix_dominance_holds_rate": prefix_dominance_holds_count / n_cases_safe,
        "greedy_tp_in_early_count": greedy_tp_in_early_count,
        "greedy_tp_in_early_rate": greedy_tp_in_early_count / n_cases_safe,
        "prefix_dominance_miss_count": prefix_dominance_miss_count,
        "prefix_dominance_miss_rate": prefix_dominance_miss_count / n_cases_safe,
        "greedy_tp_not_in_relevant_count": greedy_tp_not_in_relevant_count,
        "missing_early_count": missing_early_count,
        "missing_viable_count": missing_viable_count,
    }

    per_epsilon_summary = [
        {
            "epsilon": epsilon,
            "cases": data["cases"],
            "dominance_holds": data["dominance_holds"],
            "greedy_tp_in_early": data["greedy_tp_in_early"],
            "dominance_misses": data["dominance_misses"],
        }
        for epsilon, data in sorted(per_epsilon.items())
    ]

    data = {
        "input_path": str(INPUT_PATH),
        "input_case_count": len(all_cases),
        "filtered_false_negative_count": len(false_negative_cases),
        "aggregate": aggregate,
        "per_epsilon_summary": per_epsilon_summary,
        "per_case": per_case_records,
    }

    metadata = ExperimentMetadata(
        name="prefix_dominance_test",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=n_cases,
        seed_range=(int(settings["seed_start"]), int(settings["seed_end"])),
        parameters={
            "source": "fn_divergence_analysis.json",
            "n_events": n_events,
            "n_actors": n_actors,
            "focal_actor": default_actor,
            "max_sequence_length": max_sequence_length,
            "grammar": settings["grammar"],
            "partitioning": "focal-actor local index (theorem-aligned actor filter)",
        },
    )

    save_results(
        name="prefix_dominance_test",
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )

    print(f"False-negative cases analyzed: {aggregate['n_cases']}")
    print(
        "Prefix dominance holds: "
        f"{aggregate['prefix_dominance_holds_count']}/{aggregate['n_cases']} "
        f"({aggregate['prefix_dominance_holds_rate']:.3f})"
    )
    print(
        "Greedy TP in early set: "
        f"{aggregate['greedy_tp_in_early_count']}/{aggregate['n_cases']} "
        f"({aggregate['greedy_tp_in_early_rate']:.3f})"
    )
    print(
        "Prefix dominance misses (condition false but greedy failed): "
        f"{aggregate['prefix_dominance_miss_count']}/{aggregate['n_cases']} "
        f"({aggregate['prefix_dominance_miss_rate']:.3f})"
    )
    print(
        f"Greedy TP not in focal-relevant event set: {aggregate['greedy_tp_not_in_relevant_count']}"
    )

    print("")
    print("| epsilon | cases | dominance_holds | greedy_tp_in_early | dominance_misses |")
    print("|---------|-------|-----------------|--------------------|------------------|")
    for row in per_epsilon_summary:
        print(
            f"| {row['epsilon']:.2f} | {row['cases']} | {row['dominance_holds']} | "
            f"{row['greedy_tp_in_early']} | {row['dominance_misses']} |"
        )

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_prefix_dominance_test()
