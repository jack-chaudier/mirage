"""Greedoid axiom testing for RHUN feasible extraction families.

This script intentionally runs only the small-graph axiom panel.
Viability characterization on larger graphs lives in
`experiments/run_viability_characterization.py`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_phases
from rhun.extraction.validator import validate
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence


@dataclass(frozen=True)
class FeasibilityEval:
    valid: bool
    violations: tuple[str, ...]


def _violation_key(violation: str) -> str:
    return violation.split(":", maxsplit=1)[0].strip()


def _subset_to_idlist(subset: frozenset[str]) -> list[str]:
    return sorted(subset)


def _evaluate_subset(
    subset: frozenset[str],
    by_id: dict[str, Event],
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> FeasibilityEval:
    events = tuple(sorted((by_id[event_id] for event_id in subset), key=lambda event: (event.timestamp, event.id)))
    phases = classify_phases(events, min_development=grammar.min_prefix_elements)
    sequence = ExtractedSequence(events=events, phases=phases, focal_actor=focal_actor)
    valid, violations = validate(sequence, grammar, graph)
    return FeasibilityEval(valid=valid, violations=tuple(violations))


def enumerate_feasible_sets(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> tuple[set[frozenset[str]], dict[frozenset[str], FeasibilityEval], tuple[str, ...]]:
    focal_events = tuple(sorted(graph.events_for_actor(focal_actor), key=lambda event: (event.timestamp, event.id)))
    event_ids = tuple(event.id for event in focal_events)
    by_id = {event.id: event for event in graph.events}

    feasible: set[frozenset[str]] = set()
    evaluations: dict[frozenset[str], FeasibilityEval] = {}

    n = len(event_ids)
    for mask in range(1 << n):
        subset_ids = frozenset(event_ids[idx] for idx in range(n) if (mask >> idx) & 1)
        eval_result = _evaluate_subset(subset_ids, by_id, graph, focal_actor, grammar)
        evaluations[subset_ids] = eval_result
        if eval_result.valid:
            feasible.add(subset_ids)

    return feasible, evaluations, event_ids


def test_hereditary(
    feasible: set[frozenset[str]],
    evaluations: dict[frozenset[str], FeasibilityEval],
) -> tuple[bool, dict | None]:
    # Equivalent hereditary test via immediate deletions:
    # if all single-element deletions of each feasible set are feasible,
    # then every subset is feasible by induction.
    for subset in feasible:
        for event_id in subset:
            reduced = frozenset(subset - {event_id})
            if reduced in feasible:
                continue
            eval_reduced = evaluations[reduced]
            return False, {
                "feasible_superset": _subset_to_idlist(subset),
                "infeasible_subset": _subset_to_idlist(reduced),
                "violations": list(eval_reduced.violations),
            }
    return True, None


def test_accessibility(
    feasible: set[frozenset[str]],
    evaluations: dict[frozenset[str], FeasibilityEval],
) -> tuple[bool, dict | None]:
    for subset in feasible:
        if len(subset) == 0:
            continue

        removable = False
        reason_counter: Counter[str] = Counter()
        for event_id in subset:
            reduced = frozenset(subset - {event_id})
            if reduced in feasible:
                removable = True
                break
            for violation in evaluations[reduced].violations:
                reason_counter[_violation_key(violation)] += 1

        if not removable:
            return False, {
                "subset": _subset_to_idlist(subset),
                "reduction_violation_keys": dict(reason_counter),
            }

    return True, None


def test_exchange(
    feasible: set[frozenset[str]],
    evaluations: dict[frozenset[str], FeasibilityEval],
) -> tuple[bool, dict | None]:
    feasible_list = list(feasible)

    for a in feasible_list:
        for b in feasible_list:
            if len(a) <= len(b):
                continue

            donor_found = False
            reason_counter: Counter[str] = Counter()
            donors = list(a - b)
            for donor in donors:
                augmented = frozenset(set(b) | {donor})
                if augmented in feasible:
                    donor_found = True
                    break
                for violation in evaluations[augmented].violations:
                    reason_counter[_violation_key(violation)] += 1

            if not donor_found:
                return False, {
                    "A": _subset_to_idlist(a),
                    "B": _subset_to_idlist(b),
                    "candidate_donors": sorted(donors),
                    "augmentation_violation_keys": dict(reason_counter),
                }

    return True, None


def _classify_instance(accessibility: bool, exchange: bool) -> str:
    if accessibility and exchange:
        return "greedoid_candidate"
    if accessibility and not exchange:
        return "accessible_non_greedoid"
    return "not_accessible"


def run_axiom_panel(
    *,
    k_values: tuple[int, ...] = (1,),
    eps_values: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
    seed_range: tuple[int, int] = (0, 9),
    n_events: int = 20,
    n_actors: int = 3,
    focal_actor: str = "actor_0",
) -> dict:
    generator = BurstyGenerator()
    seeds = range(seed_range[0], seed_range[1] + 1)

    instances: list[dict] = []
    aggregate: dict[str, dict] = {}
    focal_counts: list[int] = []

    for k in k_values:
        grammar = GrammarConfig(
            min_prefix_elements=k,
            max_phase_regressions=0,
            max_turning_points=1,
            min_length=0,
            max_length=999,
            min_timespan_fraction=0.0,
            focal_actor_coverage=0.0,
        )

        rows_for_k: list[dict] = []
        hereditary_fails = 0
        accessibility_fails = 0
        exchange_fails = 0
        classification_counts: Counter[str] = Counter()

        for seed in seeds:
            for epsilon in eps_values:
                graph = generator.generate(
                    BurstyConfig(
                        n_events=n_events,
                        n_actors=n_actors,
                        seed=seed,
                        epsilon=epsilon,
                    )
                )

                feasible, evaluations, event_ids = enumerate_feasible_sets(
                    graph=graph,
                    focal_actor=focal_actor,
                    grammar=grammar,
                )

                n_focal = len(event_ids)
                focal_counts.append(n_focal)

                hereditary_ok, hereditary_counterexample = test_hereditary(feasible, evaluations)
                accessibility_ok, accessibility_counterexample = test_accessibility(feasible, evaluations)
                exchange_ok, exchange_counterexample = test_exchange(feasible, evaluations)

                if not hereditary_ok:
                    hereditary_fails += 1
                if not accessibility_ok:
                    accessibility_fails += 1
                if not exchange_ok:
                    exchange_fails += 1

                classification = _classify_instance(accessibility_ok, exchange_ok)
                classification_counts[classification] += 1

                instance_row = {
                    "seed": seed,
                    "epsilon": epsilon,
                    "k": k,
                    "n_focal_events": n_focal,
                    "n_subsets": 1 << n_focal,
                    "n_feasible_sets": len(feasible),
                    "hereditary_holds": hereditary_ok,
                    "accessibility_holds": accessibility_ok,
                    "exchange_holds": exchange_ok,
                    "classification": classification,
                    "counterexamples": {
                        "hereditary": hereditary_counterexample,
                        "accessibility": accessibility_counterexample,
                        "exchange": exchange_counterexample,
                    },
                }
                rows_for_k.append(instance_row)
                instances.append(instance_row)

        n_instances = len(rows_for_k)
        aggregate[str(k)] = {
            "k": k,
            "instances": n_instances,
            "hereditary_fails": hereditary_fails,
            "accessibility_fails": accessibility_fails,
            "exchange_fails": exchange_fails,
            "classification_counts": dict(classification_counts),
        }

    recommendation = (
        "If accessibility fails, frame as a non-accessible, non-hereditary feasible family "
        "(outside classical greedoid assumptions)."
    )

    focal_summary = {
        "min": min(focal_counts) if focal_counts else 0,
        "max": max(focal_counts) if focal_counts else 0,
        "mean": (sum(focal_counts) / len(focal_counts)) if focal_counts else 0.0,
    }

    return {
        "settings": {
            "n_events": n_events,
            "n_actors": n_actors,
            "epsilons": list(eps_values),
            "seed_range": [seed_range[0], seed_range[1]],
            "focal_actor": focal_actor,
            "k_values": list(k_values),
            "grammar_probe": {
                "min_length": 0,
                "max_length": 999,
                "min_timespan_fraction": 0.0,
                "focal_actor_coverage": 0.0,
            },
            "focal_event_count_summary": focal_summary,
        },
        "instances": instances,
        "aggregate_by_k": aggregate,
        "paper_framing_recommendation": recommendation,
    }


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        "# Greedoid Axiom Test",
        "",
        f"Generated: {metadata.timestamp}",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        "| k | instances | hereditary_fails | accessibility_fails | exchange_fails | classification_counts |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for k in sorted(data["aggregate_by_k"].keys(), key=int):
        row = data["aggregate_by_k"][k]
        lines.append(
            f"| {row['k']} | {row['instances']} | {row['hereditary_fails']} | "
            f"{row['accessibility_fails']} | {row['exchange_fails']} | {row['classification_counts']} |"
        )

    lines.extend(
        [
            "",
            f"Focal-event summary: {data['settings']['focal_event_count_summary']}",
            "",
            "Paper framing recommendation:",
            data["paper_framing_recommendation"],
        ]
    )
    return "\n".join(lines)


def _print_stdout_summary(data: dict) -> None:
    print("Greedoid Axiom Test Results")
    print("============================")

    settings = data["settings"]
    seed_count = settings["seed_range"][1] - settings["seed_range"][0] + 1
    total_per_k = seed_count * len(settings["epsilons"])

    print(
        f"Instances per k: {total_per_k} "
        f"({seed_count} seeds × {len(settings['epsilons'])} eps)"
    )
    print(f"Focal-event summary: {settings['focal_event_count_summary']}")

    for k in sorted(data["aggregate_by_k"].keys(), key=int):
        row = data["aggregate_by_k"][k]
        print(f"Grammar: k={row['k']}")
        print(
            "  Hereditary: "
            f"{'HOLDS' if row['hereditary_fails'] == 0 else f'FAILS in {row['hereditary_fails']}/{row['instances']} instances'}"
        )
        print(
            "  Accessibility: "
            f"{'HOLDS' if row['accessibility_fails'] == 0 else f'FAILS in {row['accessibility_fails']}/{row['instances']} instances'}"
        )
        print(
            "  Exchange: "
            f"{'HOLDS' if row['exchange_fails'] == 0 else f'FAILS in {row['exchange_fails']}/{row['instances']} instances'}"
        )
        print(f"  Classification: {row['classification_counts']}")

    print("Classification implication for paper framing:")
    print(f"  {data['paper_framing_recommendation']}")


def run_greedoid_axiom_test() -> dict:
    timer = ExperimentTimer()

    panel = run_axiom_panel()

    metadata = ExperimentMetadata(
        name="greedoid_axiom_test",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=10 * 4,
        n_extractions=0,
        seed_range=(0, 9),
        parameters={
            "contains": [
                "hereditary/accessibility/exchange exhaustive tests",
            ]
        },
    )

    save_results("greedoid_axiom_test", panel, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": panel}


if __name__ == "__main__":
    payload = run_greedoid_axiom_test()
    _print_stdout_summary(payload["results"])
