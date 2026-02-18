"""Viability characterization on greedy/oracle trajectories.

Separate from greedoid axiom testing so small-graph exhaustive checks stay fast.
"""

from __future__ import annotations

from collections import Counter

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence
from rhun.theory.theorem import check_precondition
from rhun.theory.viability import compute_viability, partial_at_tp_assignment


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _oracle_selection_order(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    oracle: ExtractedSequence,
    max_sequence_length: int = 20,
) -> tuple[Event, ...]:
    """Reconstruct oracle's selection order from search._build_oracle_candidate logic."""
    tp_id = oracle.metadata.get("forced_turning_point")
    if not isinstance(tp_id, str):
        return oracle.events

    by_id = {event.id: event for event in graph.events}
    tp = by_id.get(tp_id)
    if tp is None:
        return oracle.events

    actor_events = [event for event in graph.events if focal_actor in event.actors]
    before = [event for event in actor_events if event.timestamp < tp.timestamp]
    after = [event for event in actor_events if event.timestamp > tp.timestamp]

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
    selected_ids.add(tp.id)

    extras_added: list[Event] = []
    if len(selected_ids) < grammar.min_length:
        extras = [event for event in actor_events if event.id not in selected_ids]
        extras.sort(key=lambda event: (event.weight, -event.timestamp), reverse=True)
        for event in extras:
            if len(selected_ids) >= min(max_sequence_length, grammar.min_length):
                break
            selected_ids.add(event.id)
            extras_added.append(event)

    ordered_ids = _dedupe_preserve_order(
        [event.id for event in chosen_before]
        + [tp.id]
        + [event.id for event in chosen_after]
        + [event.id for event in extras_added]
    )
    sequence_id_set = {event.id for event in oracle.events}
    ordered_events = tuple(by_id[event_id] for event_id in ordered_ids if event_id in sequence_id_set)

    if len(ordered_events) != len(oracle.events):
        # Fallback: retain deterministic oracle event order from extraction output.
        return oracle.events
    return ordered_events


def _viability_along_selection_order(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    ordered_events: tuple[Event, ...],
) -> dict:
    states: list[dict] = []
    first_non_viable_step: int | None = None

    for step in range(1, len(ordered_events) + 1):
        partial = ordered_events[:step]
        state = compute_viability(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            partial_sequence=partial,
            exhaustive_limit=0,
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


def run_viability_characterization() -> dict:
    generator = BurstyGenerator()
    grammar = GrammarConfig(min_prefix_elements=1)
    focal_actor = "actor_0"

    theorem_failures_tested = 0
    theorem_nonviable_confirmed = 0
    theorem_viable_counterexamples = 0
    theorem_reason_counts: Counter[str] = Counter()

    oracle_paths_tested = 0
    oracle_nonviable_counterexamples = 0
    oracle_reason_counts: Counter[str] = Counter()

    case_rows: list[dict] = []

    for seed in range(50):
        graph = generator.generate(BurstyConfig(seed=seed, epsilon=0.80, n_events=200, n_actors=6))

        greedy = greedy_extract(graph, focal_actor, grammar, pool_strategy="injection")
        prediction = check_precondition(graph, focal_actor, grammar)

        if prediction["predicted_failure"] and not greedy.valid:
            theorem_failures_tested += 1
            partial = partial_at_tp_assignment(greedy)
            viability = compute_viability(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                partial_sequence=partial,
                exhaustive_limit=10,
            )

            if not viability["viable"]:
                theorem_nonviable_confirmed += 1
            else:
                theorem_viable_counterexamples += 1
            theorem_reason_counts[str(viability["reason"])] += 1

            case_rows.append(
                {
                    "seed": seed,
                    "type": "theorem_predicted_failure",
                    "predicted_failure": True,
                    "greedy_valid": bool(greedy.valid),
                    "viability": viability,
                }
            )

        oracle, _ = oracle_extract(graph, focal_actor, grammar)
        if (not greedy.valid) and (oracle is not None and oracle.valid):
            oracle_paths_tested += 1
            oracle_order = _oracle_selection_order(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                oracle=oracle,
            )
            path = _viability_along_selection_order(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                ordered_events=oracle_order,
            )
            if not path["all_viable"]:
                oracle_nonviable_counterexamples += 1
                bad_step = path["first_non_viable_step"]
                if bad_step is not None:
                    bad_reason = path["states"][bad_step]["reason"]
                    oracle_reason_counts[str(bad_reason)] += 1

            case_rows.append(
                {
                    "seed": seed,
                    "type": "oracle_path_on_greedy_failure",
                    "greedy_valid": bool(greedy.valid),
                    "oracle_valid": bool(oracle.valid),
                    "oracle_path_all_viable": bool(path["all_viable"]),
                    "first_non_viable_step": path["first_non_viable_step"],
                    "oracle_order_ids": [event.id for event in oracle_order],
                }
            )

    data = {
        "settings": {
            "epsilon": 0.80,
            "seeds": [0, 49],
            "n_events": 200,
            "n_actors": 6,
            "focal_actor": focal_actor,
            "grammar": {
                "min_prefix_elements": grammar.min_prefix_elements,
                "max_phase_regressions": grammar.max_phase_regressions,
                "max_turning_points": grammar.max_turning_points,
                "min_length": grammar.min_length,
                "max_length": grammar.max_length,
                "min_timespan_fraction": grammar.min_timespan_fraction,
                "focal_actor_coverage": grammar.focal_actor_coverage,
            },
            "notes": [
                "Targeted viability checks only; no full state-space enumeration.",
                "Oracle-path viability uses analytic mode (exhaustive_limit=0) for speed.",
            ],
        },
        "theorem_failures_tested": theorem_failures_tested,
        "theorem_nonviable_confirmed": theorem_nonviable_confirmed,
        "theorem_viable_counterexamples": theorem_viable_counterexamples,
        "theorem_reason_counts": dict(theorem_reason_counts),
        "oracle_paths_tested": oracle_paths_tested,
        "oracle_nonviable_counterexamples": oracle_nonviable_counterexamples,
        "oracle_reason_counts": dict(oracle_reason_counts),
        "cases": case_rows,
    }
    return data


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        "# Viability Characterization",
        "",
        f"Generated: {metadata.timestamp}",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"- Theorem-predicted failures tested: {data['theorem_failures_tested']}",
        (
            "- All confirmed non-viable at TP assignment: "
            f"{data['theorem_nonviable_confirmed'] == data['theorem_failures_tested']}"
        ),
        f"- Non-viable reasons: {data['theorem_reason_counts']}",
        f"- Oracle paths tested on greedy-failure cases: {data['oracle_paths_tested']}",
        f"- Oracle non-viable counterexamples: {data['oracle_nonviable_counterexamples']}",
    ]
    return "\n".join(lines)


def _print_stdout_summary(data: dict) -> None:
    print("Viability Characterization")
    print("============================")
    print(f"Theorem-predicted failures tested: {data['theorem_failures_tested']}")
    print(
        "All confirmed non-viable at TP assignment: "
        f"{'yes' if data['theorem_nonviable_confirmed'] == data['theorem_failures_tested'] else 'no'}"
    )
    print(f"Non-viable state reasons: {data['theorem_reason_counts']}")
    print(
        "Oracle never enters non-viable state: "
        f"{'confirmed' if data['oracle_nonviable_counterexamples']==0 else 'counterexample'}"
    )
    if data["oracle_nonviable_counterexamples"] > 0:
        print(f"Oracle counterexample reasons: {data['oracle_reason_counts']}")


if __name__ == "__main__":
    timer = ExperimentTimer()
    data = run_viability_characterization()
    metadata = ExperimentMetadata(
        name="viability_characterization",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=50,
        n_extractions=100,
        seed_range=(0, 49),
        parameters={"epsilon": 0.80},
    )
    save_results("viability_characterization", data, metadata, summary_formatter=_summary_markdown)
    _print_stdout_summary(data)
