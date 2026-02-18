"""Oracle-diff diagnostic for high-epsilon false negatives."""

from __future__ import annotations

from collections import Counter

from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_with_turning_point
from rhun.extraction.search import _build_oracle_candidate, greedy_extract, oracle_extract
from rhun.extraction.validator import validate
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, ExtractedSequence
from rhun.theory.theorem import check_precondition


def _violation_key(violation: str) -> str:
    return violation.split(":", maxsplit=1)[0].strip()


def _format_breakdown(breakdown: dict[str, int]) -> str:
    if not breakdown:
        return "none"
    parts = [f"{key}={value}" for key, value in sorted(breakdown.items())]
    return ", ".join(parts)


def _format_table(rows: list[dict]) -> str:
    lines = [
        "| epsilon | false_negatives | oracle_succeeded | oracle_succeeded_pct | oracle_failed | oracle_failed_pct | oracle_failed_violation_breakdown |",
        "|---------|------------------|------------------|---------------------|---------------|------------------|-----------------------------------|",
    ]

    for row in rows:
        lines.append(
            "| {epsilon:.2f} | {total_false_negatives} | {oracle_succeeded} | "
            "{oracle_succeeded_pct:.3f} | {oracle_failed} | {oracle_failed_pct:.3f} | {breakdown} |".format(
                epsilon=row["epsilon"],
                total_false_negatives=row["total_false_negatives"],
                oracle_succeeded=row["oracle_succeeded"],
                oracle_succeeded_pct=row["oracle_succeeded_pct"],
                oracle_failed=row["oracle_failed"],
                oracle_failed_pct=row["oracle_failed_pct"],
                breakdown=_format_breakdown(row["oracle_failed_violation_breakdown"]),
            )
        )

    return "\n".join(lines) + "\n"


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    return (
        f"# {metadata.name}\n\n"
        f"Generated: {metadata.timestamp}\n\n"
        f"Runtime: {metadata.runtime_seconds:.2f}s\n\n"
        + _format_table(data["rows"])
    )


def _oracle_failure_breakdown(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
    max_sequence_length: int,
) -> dict:
    """Inspect all oracle candidates and aggregate violation types when oracle finds no valid sequence."""
    counts: Counter[str] = Counter()
    candidate_count = 0

    for event in graph.events:
        if focal_actor not in event.actors:
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

        tp_idx = next((i for i, e in enumerate(candidate_events) if e.id == event.id), None)
        if tp_idx is None:
            continue

        phases = classify_with_turning_point(candidate_events, tp_idx)
        sequence = ExtractedSequence(
            events=candidate_events,
            phases=phases,
            focal_actor=focal_actor,
        )
        valid, violations = validate(sequence, grammar, graph)
        candidate_count += 1
        if valid:
            continue

        if not any(phase.name == "RESOLUTION" for phase in phases):
            counts["resolution_phase_empty"] += 1

        if not violations:
            counts["invalid_without_reported_violation"] += 1

        for violation in violations:
            counts[_violation_key(violation)] += 1

    if candidate_count == 0:
        counts["oracle_no_candidates"] += 1

    return {
        "candidate_count": candidate_count,
        "violation_counts": dict(sorted(counts.items())),
    }


def run_oracle_diff(
    epsilons: list[float] | None = None,
    seeds: range | None = None,
    n_events: int = 200,
    n_actors: int = 6,
    focal_actor: str = "actor_0",
    max_sequence_length: int = 20,
) -> dict:
    eps_values = epsilons or [0.80, 0.85, 0.90, 0.95]
    seed_values = seeds or range(0, 200)

    grammar = GrammarConfig(min_prefix_elements=1)
    generator = BurstyGenerator()
    timer = ExperimentTimer()

    rows: list[dict] = []
    per_epsilon_details: dict[str, list[dict]] = {}

    total_graphs = 0
    total_extractions = 0
    total_oracle_calls = 0

    for epsilon in eps_values:
        false_negative_count = 0
        oracle_succeeded_count = 0
        oracle_failed_count = 0
        oracle_failed_violation_counter: Counter[str] = Counter()
        seed_records: list[dict] = []

        for seed in seed_values:
            graph = generator.generate(
                BurstyConfig(
                    seed=seed,
                    epsilon=epsilon,
                    n_events=n_events,
                    n_actors=n_actors,
                )
            )
            total_graphs += 1

            theorem = check_precondition(graph, focal_actor, grammar)
            greedy = greedy_extract(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                pool_strategy="injection",
                max_sequence_length=max_sequence_length,
            )
            total_extractions += 1

            predicted_failure = bool(theorem["predicted_failure"])
            actual_failure = not greedy.valid
            false_negative = (not predicted_failure) and actual_failure

            record: dict = {
                "seed": seed,
                "epsilon": epsilon,
                "focal_actor": focal_actor,
                "max_weight_index": int(theorem["max_weight_index"]),
                "max_weight_position": float(theorem["max_weight_position"]),
                "events_before_max": int(theorem["events_before_max"]),
                "predicted_failure": predicted_failure,
                "greedy_valid": bool(greedy.valid),
                "greedy_violations": list(greedy.violations),
                "greedy_absorbed": bool(greedy.metadata.get("absorbed", False)),
                "false_negative": false_negative,
                "oracle": None,
            }

            if false_negative:
                false_negative_count += 1
                total_oracle_calls += 1

                oracle_result, oracle_diagnostics = oracle_extract(
                    graph=graph,
                    focal_actor=focal_actor,
                    grammar=grammar,
                    max_sequence_length=max_sequence_length,
                )

                oracle_succeeded = bool(oracle_result is not None and oracle_result.valid)
                oracle_payload: dict = {
                    "succeeded": oracle_succeeded,
                    "diagnostics": oracle_diagnostics,
                    "valid": bool(oracle_result.valid) if oracle_result is not None else False,
                    "violations": list(oracle_result.violations) if oracle_result is not None else [],
                    "score": float(oracle_result.score) if oracle_result is not None else None,
                    "turning_point_id": (
                        oracle_result.turning_point.id
                        if oracle_result is not None and oracle_result.turning_point is not None
                        else None
                    ),
                }

                if oracle_succeeded:
                    oracle_succeeded_count += 1
                else:
                    oracle_failed_count += 1
                    failure_breakdown = _oracle_failure_breakdown(
                        graph=graph,
                        focal_actor=focal_actor,
                        grammar=grammar,
                        max_sequence_length=max_sequence_length,
                    )
                    oracle_payload["failure_breakdown"] = failure_breakdown
                    for key, value in failure_breakdown["violation_counts"].items():
                        oracle_failed_violation_counter[key] += int(value)

                record["oracle"] = oracle_payload

            seed_records.append(record)

        false_negative_denominator = false_negative_count if false_negative_count else 1
        rows.append(
            {
                "epsilon": epsilon,
                "total_false_negatives": false_negative_count,
                "oracle_succeeded": oracle_succeeded_count,
                "oracle_succeeded_pct": oracle_succeeded_count / false_negative_denominator,
                "oracle_failed": oracle_failed_count,
                "oracle_failed_pct": oracle_failed_count / false_negative_denominator,
                "oracle_failed_violation_breakdown": dict(
                    sorted(oracle_failed_violation_counter.items())
                ),
            }
        )

        per_epsilon_details[f"{epsilon:.2f}"] = seed_records

    data = {
        "rows": rows,
        "table": _format_table(rows),
        "settings": {
            "epsilons": eps_values,
            "n_events": n_events,
            "n_actors": n_actors,
            "focal_actor": focal_actor,
            "seed_start": min(seed_values),
            "seed_end": max(seed_values),
            "max_sequence_length": max_sequence_length,
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
        "per_seed_results": per_epsilon_details,
    }

    metadata = ExperimentMetadata(
        name="oracle_diff_results",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions + total_oracle_calls,
        seed_range=(min(seed_values), max(seed_values)),
        parameters={
            "epsilons": eps_values,
            "n_events": n_events,
            "n_actors": n_actors,
            "focal_actor": focal_actor,
            "oracle_only_on_false_negatives": True,
        },
    )

    save_results(
        name="oracle_diff_results",
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )
    return {"metadata": metadata, "results": data}
