"""Characterize greedy TP misselection on false-negative cases."""

from __future__ import annotations

import json
from collections import Counter
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
from rhun.schemas import CausalGraph, Event


INPUT_PATH = Path(__file__).resolve().parent / "output" / "fn_divergence_analysis.json"


def _violation_type(violation: str) -> str:
    return violation.split(":", maxsplit=1)[0].strip()


def _event_index(graph: CausalGraph, event_id: str) -> int:
    return next(i for i, event in enumerate(graph.events) if event.id == event_id)


def _tp_info(graph: CausalGraph, event: Event | None) -> dict | None:
    if event is None:
        return None

    index = _event_index(graph, event.id)
    return {
        "id": event.id,
        "weight": float(event.weight),
        "index": int(index),
        "normalized_position": float(graph.global_position(event)),
        "timestamp": float(event.timestamp),
    }


def _rank_in_weight_order(graph: CausalGraph, event_id: str) -> int:
    ordered = sorted(
        graph.events,
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )
    for idx, event in enumerate(ordered, start=1):
        if event.id == event_id:
            return idx
    raise ValueError(f"Event id not found for rank: {event_id}")


def _cooccurrence_matrix(violation_sets: list[set[str]]) -> dict[str, dict[str, int]]:
    unique = sorted({violation for row in violation_sets for violation in row})
    matrix: dict[str, dict[str, int]] = {}
    for a in unique:
        matrix[a] = {}
        for b in unique:
            matrix[a][b] = sum(1 for row in violation_sets if a in row and b in row)
    return matrix


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    agg = data["aggregate"]
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"False-negative cases analyzed: {agg['n_cases']}",
        "",
        "## TP Weight / Position",
        "",
        f"- Greedy TP weight mean/median: {agg['greedy_tp_weight_mean']:.4f} / {agg['greedy_tp_weight_median']:.4f}",
        f"- Oracle TP weight mean/median: {agg['oracle_tp_weight_mean']:.4f} / {agg['oracle_tp_weight_median']:.4f}",
        f"- Greedy TP index mean/median: {agg['greedy_tp_index_mean']:.3f} / {agg['greedy_tp_index_median']:.3f}",
        f"- Oracle TP index mean/median: {agg['oracle_tp_index_mean']:.3f} / {agg['oracle_tp_index_median']:.3f}",
        f"- Greedy TP normalized position mean/median: {agg['greedy_tp_pos_mean']:.4f} / {agg['greedy_tp_pos_median']:.4f}",
        f"- Oracle TP normalized position mean/median: {agg['oracle_tp_pos_mean']:.4f} / {agg['oracle_tp_pos_median']:.4f}",
        "",
        "## Oracle TP Rank",
        "",
        f"- Mean/median rank: {agg['oracle_tp_rank_mean']:.3f} / {agg['oracle_tp_rank_median']:.3f}",
        "",
        "## Same TP Cases",
        "",
        f"- Count: {agg['same_tp_case_count']}",
    ]
    return "\n".join(lines) + "\n"


def run_tp_misselection_analysis() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing required input file: {INPUT_PATH}")

    source = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    settings = source["results"]["settings"]
    candidate_cases = source["results"]["per_case"]

    grammar = GrammarConfig(**settings["grammar"])
    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    default_actor = str(settings["focal_actor"])
    max_sequence_length = int(settings["max_sequence_length"])

    false_negative_cases = [
        case
        for case in candidate_cases
        if (not bool(case.get("recomputed_predicted_failure", True)))
        and (not bool(case.get("recomputed_greedy_valid", True)))
        and bool(case.get("oracle_valid", False))
    ]

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    per_case_records: list[dict] = []
    greedy_weights: list[float] = []
    oracle_weights: list[float] = []
    greedy_indices: list[int] = []
    oracle_indices: list[int] = []
    greedy_positions: list[float] = []
    oracle_positions: list[float] = []
    oracle_ranks: list[int] = []

    greedy_higher_count = 0
    same_weight_count = 0
    oracle_higher_count = 0

    greedy_earlier_count = 0
    same_position_count = 0
    oracle_earlier_count = 0

    violation_sets: list[set[str]] = []
    violation_type_counts: Counter[str] = Counter()

    same_tp_cases: list[dict] = []

    for case in sorted(false_negative_cases, key=lambda row: (float(row["epsilon"]), int(row["seed"]))):
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

        greedy = greedy_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_strategy="injection",
            max_sequence_length=max_sequence_length,
        )
        oracle, oracle_diag = oracle_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            max_sequence_length=max_sequence_length,
        )

        greedy_tp = _tp_info(graph, greedy.turning_point)
        oracle_tp = _tp_info(graph, oracle.turning_point if oracle is not None else None)
        if greedy_tp is None or oracle_tp is None:
            continue

        greedy_w = float(greedy_tp["weight"])
        oracle_w = float(oracle_tp["weight"])
        greedy_i = int(greedy_tp["index"])
        oracle_i = int(oracle_tp["index"])
        greedy_p = float(greedy_tp["normalized_position"])
        oracle_p = float(oracle_tp["normalized_position"])

        greedy_weights.append(greedy_w)
        oracle_weights.append(oracle_w)
        greedy_indices.append(greedy_i)
        oracle_indices.append(oracle_i)
        greedy_positions.append(greedy_p)
        oracle_positions.append(oracle_p)

        if greedy_w > oracle_w:
            greedy_higher_count += 1
        elif greedy_w < oracle_w:
            oracle_higher_count += 1
        else:
            same_weight_count += 1

        if greedy_i < oracle_i:
            greedy_earlier_count += 1
        elif greedy_i > oracle_i:
            oracle_earlier_count += 1
        else:
            same_position_count += 1

        oracle_rank = _rank_in_weight_order(graph, str(oracle_tp["id"]))
        oracle_ranks.append(oracle_rank)

        violation_types = sorted({_violation_type(v) for v in greedy.violations})
        vset = set(violation_types)
        violation_sets.append(vset)
        for violation_type in violation_types:
            violation_type_counts[violation_type] += 1

        same_tp = str(greedy_tp["id"]) == str(oracle_tp["id"])

        record = {
            "seed": seed,
            "epsilon": epsilon,
            "focal_actor": focal_actor,
            "greedy_valid": bool(greedy.valid),
            "oracle_valid": bool(oracle.valid) if oracle is not None else False,
            "oracle_diagnostics": oracle_diag,
            "greedy_tp": greedy_tp,
            "oracle_tp": oracle_tp,
            "same_tp": same_tp,
            "tp_weight_delta_greedy_minus_oracle": greedy_w - oracle_w,
            "tp_index_delta_greedy_minus_oracle": greedy_i - oracle_i,
            "tp_position_delta_greedy_minus_oracle": greedy_p - oracle_p,
            "oracle_tp_weight_rank": oracle_rank,
            "greedy_violation_types": violation_types,
            "greedy_violations": list(greedy.violations),
            "greedy_development_count": greedy.n_development,
            "oracle_development_count": oracle.n_development if oracle is not None else None,
        }
        per_case_records.append(record)

        if same_tp:
            same_tp_cases.append(
                {
                    "seed": seed,
                    "epsilon": epsilon,
                    "tp_id": str(greedy_tp["id"]),
                    "tp_weight": greedy_w,
                    "tp_index": greedy_i,
                    "tp_position": greedy_p,
                    "greedy_violation_types": violation_types,
                    "greedy_violations": list(greedy.violations),
                    "greedy_development_count": greedy.n_development,
                    "oracle_development_count": oracle.n_development if oracle is not None else None,
                }
            )

    oracle_rank_distribution = dict(sorted(Counter(oracle_ranks).items()))
    cooccurrence = _cooccurrence_matrix(violation_sets)

    aggregate = {
        "n_cases": len(per_case_records),
        "greedy_tp_weight_mean": mean(greedy_weights) if greedy_weights else 0.0,
        "greedy_tp_weight_median": median(greedy_weights) if greedy_weights else 0.0,
        "oracle_tp_weight_mean": mean(oracle_weights) if oracle_weights else 0.0,
        "oracle_tp_weight_median": median(oracle_weights) if oracle_weights else 0.0,
        "greedy_tp_index_mean": mean(greedy_indices) if greedy_indices else 0.0,
        "greedy_tp_index_median": median(greedy_indices) if greedy_indices else 0.0,
        "oracle_tp_index_mean": mean(oracle_indices) if oracle_indices else 0.0,
        "oracle_tp_index_median": median(oracle_indices) if oracle_indices else 0.0,
        "greedy_tp_pos_mean": mean(greedy_positions) if greedy_positions else 0.0,
        "greedy_tp_pos_median": median(greedy_positions) if greedy_positions else 0.0,
        "oracle_tp_pos_mean": mean(oracle_positions) if oracle_positions else 0.0,
        "oracle_tp_pos_median": median(oracle_positions) if oracle_positions else 0.0,
        "oracle_tp_rank_mean": mean(oracle_ranks) if oracle_ranks else 0.0,
        "oracle_tp_rank_median": median(oracle_ranks) if oracle_ranks else 0.0,
        "oracle_tp_rank_distribution": oracle_rank_distribution,
        "greedy_higher_weight_count": greedy_higher_count,
        "same_weight_count": same_weight_count,
        "oracle_higher_weight_count": oracle_higher_count,
        "greedy_earlier_position_count": greedy_earlier_count,
        "same_position_count": same_position_count,
        "oracle_earlier_position_count": oracle_earlier_count,
        "greedy_violation_type_distribution": dict(sorted(violation_type_counts.items())),
        "violation_cooccurrence_matrix": cooccurrence,
        "same_tp_case_count": len(same_tp_cases),
    }

    data = {
        "input_path": str(INPUT_PATH),
        "input_case_count": len(candidate_cases),
        "filtered_false_negative_count": len(false_negative_cases),
        "aggregate": aggregate,
        "same_tp_cases": same_tp_cases,
        "per_case": per_case_records,
    }

    metadata = ExperimentMetadata(
        name="tp_misselection_analysis",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(per_case_records),
        n_extractions=len(per_case_records) * 2,
        seed_range=(int(settings["seed_start"]), int(settings["seed_end"])),
        parameters={
            "source": "fn_divergence_analysis.json",
            "n_events": n_events,
            "n_actors": n_actors,
            "focal_actor": default_actor,
            "max_sequence_length": max_sequence_length,
            "grammar": settings["grammar"],
        },
    )

    save_results(
        name="tp_misselection_analysis",
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )

    print(f"False-negative cases analyzed: {aggregate['n_cases']}")
    print(
        f"TP weight mean/median (greedy vs oracle): "
        f"{aggregate['greedy_tp_weight_mean']:.4f}/{aggregate['greedy_tp_weight_median']:.4f} vs "
        f"{aggregate['oracle_tp_weight_mean']:.4f}/{aggregate['oracle_tp_weight_median']:.4f}"
    )
    print(
        f"TP position mean/median (greedy idx vs oracle idx): "
        f"{aggregate['greedy_tp_index_mean']:.3f}/{aggregate['greedy_tp_index_median']:.3f} vs "
        f"{aggregate['oracle_tp_index_mean']:.3f}/{aggregate['oracle_tp_index_median']:.3f}"
    )
    print(
        f"TP normalized position mean/median (greedy vs oracle): "
        f"{aggregate['greedy_tp_pos_mean']:.4f}/{aggregate['greedy_tp_pos_median']:.4f} vs "
        f"{aggregate['oracle_tp_pos_mean']:.4f}/{aggregate['oracle_tp_pos_median']:.4f}"
    )

    n_cases = aggregate["n_cases"] if aggregate["n_cases"] else 1
    print("")
    print(
        f"Weight comparison counts: greedy_higher={aggregate['greedy_higher_weight_count']} "
        f"({aggregate['greedy_higher_weight_count']/n_cases:.3f}), same={aggregate['same_weight_count']} "
        f"({aggregate['same_weight_count']/n_cases:.3f}), oracle_higher={aggregate['oracle_higher_weight_count']} "
        f"({aggregate['oracle_higher_weight_count']/n_cases:.3f})"
    )
    print(
        f"Position comparison counts (by index): greedy_earlier={aggregate['greedy_earlier_position_count']} "
        f"({aggregate['greedy_earlier_position_count']/n_cases:.3f}), same={aggregate['same_position_count']} "
        f"({aggregate['same_position_count']/n_cases:.3f}), oracle_earlier={aggregate['oracle_earlier_position_count']} "
        f"({aggregate['oracle_earlier_position_count']/n_cases:.3f})"
    )

    print("")
    print(
        f"Oracle TP rank mean/median: "
        f"{aggregate['oracle_tp_rank_mean']:.3f}/{aggregate['oracle_tp_rank_median']:.3f}"
    )
    print("Oracle TP rank distribution (rank: count):")
    for rank, count in aggregate["oracle_tp_rank_distribution"].items():
        print(f"  {rank}: {count}")

    print("")
    print("Violation co-occurrence matrix:")
    keys = sorted(aggregate["violation_cooccurrence_matrix"].keys())
    if keys:
        print("  types:", ", ".join(keys))
        for row_key in keys:
            row = aggregate["violation_cooccurrence_matrix"][row_key]
            row_values = ", ".join(f"{col}={row[col]}" for col in keys)
            print(f"  {row_key}: {row_values}")
    else:
        print("  (no violations found)")

    print("")
    print(f"Same-TP cases: {aggregate['same_tp_case_count']}")
    for case in same_tp_cases:
        print(
            f"  seed={case['seed']} eps={case['epsilon']:.2f} tp={case['tp_id']} "
            f"idx={case['tp_index']} w={case['tp_weight']:.4f} "
            f"violations={case['greedy_violation_types']}"
        )

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_tp_misselection_analysis()
