"""Diagnostic analysis of greedy-vs-oracle divergence on false negative cases."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, ExtractedSequence, Phase
from rhun.theory.theorem import check_precondition


INPUT_PATH = Path(__file__).resolve().parent / "output" / "oracle_diff_results.json"


def _violation_type(violation: str) -> str:
    return violation.split(":", maxsplit=1)[0].strip()


def _turning_point_info(
    sequence: ExtractedSequence | None,
    graph: CausalGraph,
) -> dict | None:
    if sequence is None:
        return None

    tp = sequence.turning_point
    if tp is None:
        return None

    idx = next((i for i, event in enumerate(graph.events) if event.id == tp.id), None)
    if idx is None:
        return None

    return {
        "id": tp.id,
        "index": idx,
        "normalized_position": graph.global_position(tp),
        "timestamp": tp.timestamp,
        "weight": tp.weight,
    }


def _development_positions(sequence: ExtractedSequence, graph: CausalGraph) -> dict:
    indices: list[int] = []
    normalized_positions: list[float] = []

    for event, phase in zip(sequence.events, sequence.phases):
        if phase != Phase.DEVELOPMENT:
            continue
        idx = next((i for i, g_event in enumerate(graph.events) if g_event.id == event.id), None)
        if idx is None:
            continue
        indices.append(idx)
        normalized_positions.append(graph.global_position(event))

    return {
        "count": len(indices),
        "indices": indices,
        "normalized_positions": normalized_positions,
    }


def _first_divergence_step(
    greedy_ids: list[str],
    oracle_ids: list[str],
) -> int | None:
    shared_len = min(len(greedy_ids), len(oracle_ids))
    for idx in range(shared_len):
        if greedy_ids[idx] != oracle_ids[idx]:
            return idx
    if len(greedy_ids) != len(oracle_ids):
        return shared_len
    return None


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"Total false negatives analyzed: {data['aggregate']['total_cases']}",
        f"Oracle success on false negatives: {data['aggregate']['oracle_success_count']}",
        "",
        "## Per-epsilon summary",
        "",
        "| epsilon | cases | same_tp | different_tp | mean_divergence_step |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in data["per_epsilon_summary"]:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['cases']} | {row['same_tp']} | {row['different_tp']} | {row['mean_divergence_step']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Greedy violation distribution",
            "",
        ]
    )
    for key, value in sorted(data["aggregate"]["greedy_violation_type_distribution"].items()):
        lines.append(f"- {key}: {value}")

    lines.append("")
    return "\n".join(lines)


def run_fn_divergence_analysis() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_PATH}")

    input_blob = __import__("json").loads(INPUT_PATH.read_text(encoding="utf-8"))
    settings = input_blob["results"]["settings"]
    per_seed_results = input_blob["results"]["per_seed_results"]

    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    focal_actor = str(settings["focal_actor"])
    max_sequence_length = int(settings["max_sequence_length"])
    grammar = GrammarConfig(**settings["grammar"])

    false_negative_cases: list[dict] = []
    for epsilon_key, seed_rows in per_seed_results.items():
        epsilon = float(epsilon_key)
        for row in seed_rows:
            if bool(row.get("false_negative", False)):
                false_negative_cases.append(
                    {
                        "epsilon": epsilon,
                        "seed": int(row["seed"]),
                        "focal_actor": str(row.get("focal_actor", focal_actor)),
                        "stored": row,
                    }
                )

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    case_records: list[dict] = []
    greedy_violation_type_distribution: Counter[str] = Counter()
    tp_same = 0
    tp_different = 0
    greedy_development_counts: list[int] = []
    oracle_development_counts: list[int] = []
    divergence_steps: list[int] = []

    per_epsilon = defaultdict(
        lambda: {
            "cases": 0,
            "same_tp": 0,
            "different_tp": 0,
            "divergence_steps": [],
        }
    )

    for case in sorted(false_negative_cases, key=lambda c: (c["epsilon"], c["seed"])):
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        actor = str(case["focal_actor"])

        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )

        theorem = check_precondition(graph, actor, grammar)
        greedy = greedy_extract(
            graph=graph,
            focal_actor=actor,
            grammar=grammar,
            pool_strategy="injection",
            max_sequence_length=max_sequence_length,
        )
        oracle, oracle_diagnostics = oracle_extract(
            graph=graph,
            focal_actor=actor,
            grammar=grammar,
            max_sequence_length=max_sequence_length,
        )

        greedy_tp = _turning_point_info(greedy, graph)
        oracle_tp = _turning_point_info(oracle, graph)
        same_tp = (
            greedy_tp is not None
            and oracle_tp is not None
            and str(greedy_tp["id"]) == str(oracle_tp["id"])
        )

        greedy_ids = [event.id for event in greedy.events]
        oracle_ids = [event.id for event in oracle.events] if oracle is not None else []
        divergence_step = _first_divergence_step(greedy_ids, oracle_ids)

        greedy_dev = _development_positions(greedy, graph)
        oracle_dev = _development_positions(oracle, graph) if oracle is not None else {
            "count": 0,
            "indices": [],
            "normalized_positions": [],
        }

        violation_types = [_violation_type(violation) for violation in greedy.violations]
        for violation_type in set(violation_types):
            greedy_violation_type_distribution[violation_type] += 1

        greedy_development_counts.append(int(greedy_dev["count"]))
        oracle_development_counts.append(int(oracle_dev["count"]))
        if divergence_step is not None:
            divergence_steps.append(int(divergence_step))

        epsilon_bucket = per_epsilon[epsilon]
        epsilon_bucket["cases"] += 1
        if same_tp:
            epsilon_bucket["same_tp"] += 1
        else:
            epsilon_bucket["different_tp"] += 1
        if divergence_step is not None:
            epsilon_bucket["divergence_steps"].append(int(divergence_step))

        if same_tp:
            tp_same += 1
        else:
            tp_different += 1

        record = {
            "epsilon": epsilon,
            "seed": seed,
            "focal_actor": actor,
            "stored_false_negative": bool(case["stored"].get("false_negative", False)),
            "recomputed_predicted_failure": bool(theorem["predicted_failure"]),
            "recomputed_greedy_valid": bool(greedy.valid),
            "oracle_valid": bool(oracle.valid) if oracle is not None else False,
            "oracle_diagnostics": oracle_diagnostics,
            "greedy_violation_types": violation_types,
            "greedy_primary_violation": violation_types[0] if violation_types else "none",
            "greedy_turning_point": greedy_tp,
            "oracle_turning_point": oracle_tp,
            "same_turning_point": same_tp,
            "greedy_development": greedy_dev,
            "oracle_development": oracle_dev,
            "greedy_sequence_ids": greedy_ids,
            "oracle_sequence_ids": oracle_ids,
            "divergence_step": divergence_step,
        }
        case_records.append(record)

    total_cases = len(case_records)
    oracle_success_count = sum(1 for record in case_records if record["oracle_valid"])

    per_epsilon_summary = []
    for epsilon in sorted(per_epsilon):
        bucket = per_epsilon[epsilon]
        steps = bucket["divergence_steps"]
        per_epsilon_summary.append(
            {
                "epsilon": epsilon,
                "cases": bucket["cases"],
                "same_tp": bucket["same_tp"],
                "different_tp": bucket["different_tp"],
                "mean_divergence_step": mean(steps) if steps else -1.0,
                "median_divergence_step": median(steps) if steps else -1.0,
            }
        )

    aggregate = {
        "total_cases": total_cases,
        "oracle_success_count": oracle_success_count,
        "oracle_success_rate": (oracle_success_count / total_cases) if total_cases else 0.0,
        "greedy_violation_type_distribution": dict(
            sorted(greedy_violation_type_distribution.items())
        ),
        "same_tp_count": tp_same,
        "different_tp_count": tp_different,
        "same_tp_rate": (tp_same / total_cases) if total_cases else 0.0,
        "different_tp_rate": (tp_different / total_cases) if total_cases else 0.0,
        "greedy_development_mean": mean(greedy_development_counts) if greedy_development_counts else 0.0,
        "greedy_development_median": median(greedy_development_counts)
        if greedy_development_counts
        else 0.0,
        "oracle_development_mean": mean(oracle_development_counts) if oracle_development_counts else 0.0,
        "oracle_development_median": median(oracle_development_counts)
        if oracle_development_counts
        else 0.0,
        "mean_divergence_step": mean(divergence_steps) if divergence_steps else -1.0,
        "median_divergence_step": median(divergence_steps) if divergence_steps else -1.0,
    }

    data = {
        "input_path": str(INPUT_PATH),
        "settings": settings,
        "aggregate": aggregate,
        "per_epsilon_summary": per_epsilon_summary,
        "per_case": case_records,
    }

    metadata = ExperimentMetadata(
        name="fn_divergence_analysis",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_cases,
        n_extractions=total_cases * 2,
        seed_range=(int(settings["seed_start"]), int(settings["seed_end"])),
        parameters={
            "source": "oracle_diff_results.json",
            "n_events": n_events,
            "n_actors": n_actors,
            "focal_actor": focal_actor,
            "max_sequence_length": max_sequence_length,
            "grammar": asdict(grammar),
        },
    )

    save_results(
        name="fn_divergence_analysis",
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )

    print(f"False negatives analyzed: {aggregate['total_cases']}")
    print(
        "Oracle success on false negatives: "
        f"{aggregate['oracle_success_count']}/{aggregate['total_cases']} "
        f"({aggregate['oracle_success_rate']:.3f})"
    )
    print("")
    print("Greedy violation type distribution (case-level):")
    for violation_type, count in sorted(aggregate["greedy_violation_type_distribution"].items()):
        print(f"  {violation_type}: {count}")

    print("")
    print(
        "Turning-point agreement: "
        f"same={aggregate['same_tp_count']} ({aggregate['same_tp_rate']:.3f}), "
        f"different={aggregate['different_tp_count']} ({aggregate['different_tp_rate']:.3f})"
    )

    print("")
    print(
        "Development count (greedy vs oracle): "
        f"mean {aggregate['greedy_development_mean']:.3f} vs {aggregate['oracle_development_mean']:.3f}; "
        f"median {aggregate['greedy_development_median']:.3f} vs {aggregate['oracle_development_median']:.3f}"
    )
    print(
        "Divergence step: "
        f"mean {aggregate['mean_divergence_step']:.3f}, "
        f"median {aggregate['median_divergence_step']:.3f}"
    )

    print("")
    print(
        "| epsilon | cases | same_tp | different_tp | mean_divergence_step | median_divergence_step |"
    )
    print("|---------|-------|---------|--------------|----------------------|------------------------|")
    for row in per_epsilon_summary:
        print(
            f"| {row['epsilon']:.2f} | {row['cases']} | {row['same_tp']} | {row['different_tp']} | "
            f"{row['mean_divergence_step']:.3f} | {row['median_divergence_step']:.3f} |"
        )

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_fn_divergence_analysis()
