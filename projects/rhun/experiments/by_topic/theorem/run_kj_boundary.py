"""Experiment 10: k-j boundary sweep for prefix-constraint impossibility."""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from statistics import mean, median

from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph
from rhun.theory.theorem import check_precondition, diagnose_absorption


OUTPUT_NAME = "kj_boundary"
FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
SEEDS = range(100)
K_VALUES = list(range(6))
N_EVENTS = 200
N_ACTORS = 6

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

J_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
)


def _make_grammar(k: int) -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=k,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )


def _oracle_grammar() -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=0,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )


def _compute_focal_j(graph: CausalGraph, focal_actor: str) -> dict:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    focal_events_sorted = sorted(
        focal_events,
        key=lambda event: (float(event.timestamp), event.id),
    )
    if not focal_events_sorted:
        return {
            "j": None,
            "n_focal_events": 0,
            "max_weight_event_id": None,
            "max_weight_event_timestamp": None,
            "max_weight_event_weight": None,
        }

    max_weight_event = max(
        focal_events_sorted,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )
    j = next(
        index for index, event in enumerate(focal_events_sorted) if event.id == max_weight_event.id
    )
    return {
        "j": int(j),
        "n_focal_events": int(len(focal_events_sorted)),
        "max_weight_event_id": max_weight_event.id,
        "max_weight_event_timestamp": float(max_weight_event.timestamp),
        "max_weight_event_weight": float(max_weight_event.weight),
    }


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _is_in_j_bin(j_value: int, lo: int, hi: int | None) -> bool:
    if hi is None:
        return j_value >= lo
    return lo <= j_value <= hi


def _canonical_theorem_prediction(j_value: int, k_value: int) -> bool:
    return bool(j_value < k_value)


def _aggregate_kj_cells(records: list[dict]) -> list[dict]:
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        buckets[(int(record["k"]), int(record["j"]))].append(record)

    cells: list[dict] = []
    for (k, j), bucket in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        n_instances = len(bucket)
        greedy_valid_count = sum(1 for row in bucket if row["greedy_valid"])
        theorem_predicted_failure_count = sum(1 for row in bucket if row["theorem_predicted_failure"])
        absorbed_count = sum(1 for row in bucket if row["absorbed"])
        false_positive_count = sum(
            1 for row in bucket if row["theorem_predicted_failure"] and row["greedy_valid"]
        )
        false_negative_count = sum(
            1 for row in bucket if (not row["theorem_predicted_failure"]) and (not row["greedy_valid"])
        )
        theorem_correct = sum(
            1
            for row in bucket
            if bool(row["theorem_predicted_failure"]) == (not bool(row["greedy_valid"]))
        )

        cells.append(
            {
                "k": int(k),
                "j": int(j),
                "n_instances": int(n_instances),
                "greedy_valid_count": int(greedy_valid_count),
                "greedy_valid_rate": _safe_rate(greedy_valid_count, n_instances),
                "theorem_predicted_failure_count": int(theorem_predicted_failure_count),
                "theorem_accuracy": _safe_rate(theorem_correct, n_instances),
                "false_positive_count": int(false_positive_count),
                "false_negative_count": int(false_negative_count),
                "absorbed_count": int(absorbed_count),
                "j_lt_k": bool(j < k),
            }
        )
    return cells


def _aggregate_k_marginals(records: list[dict]) -> list[dict]:
    marginals: list[dict] = []
    for k in K_VALUES:
        bucket = [record for record in records if int(record["k"]) == k]
        n_instances = len(bucket)
        greedy_valid_count = sum(1 for row in bucket if row["greedy_valid"])
        theorem_correct = sum(
            1
            for row in bucket
            if bool(row["theorem_predicted_failure"]) == (not bool(row["greedy_valid"]))
        )
        false_positive_count = sum(
            1 for row in bucket if row["theorem_predicted_failure"] and row["greedy_valid"]
        )
        false_negative_count = sum(
            1 for row in bucket if (not row["theorem_predicted_failure"]) and (not row["greedy_valid"])
        )
        j_values = [int(row["j"]) for row in bucket]
        marginals.append(
            {
                "k": int(k),
                "n_instances": int(n_instances),
                "greedy_valid_count": int(greedy_valid_count),
                "greedy_valid_rate": _safe_rate(greedy_valid_count, n_instances),
                "theorem_accuracy": _safe_rate(theorem_correct, n_instances),
                "false_positive_count": int(false_positive_count),
                "false_positive_rate": _safe_rate(false_positive_count, n_instances),
                "false_negative_count": int(false_negative_count),
                "false_negative_rate": _safe_rate(false_negative_count, n_instances),
                "mean_j": float(mean(j_values)) if j_values else None,
            }
        )
    return marginals


def _aggregate_j_distribution(graph_level_rows: list[dict]) -> dict:
    j_values = [int(row["j"]) for row in graph_level_rows]
    if not j_values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "histogram": {},
        }

    histogram: dict[int, int] = defaultdict(int)
    for value in j_values:
        histogram[value] += 1

    return {
        "mean": float(mean(j_values)),
        "median": float(median(j_values)),
        "min": int(min(j_values)),
        "max": int(max(j_values)),
        "histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
    }


def _build_binned_kj_matrix(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for k in K_VALUES:
        row_cells: list[dict] = []
        k_bucket = [record for record in records if int(record["k"]) == k]
        for label, lo, hi in J_BINS:
            cell_rows = [record for record in k_bucket if _is_in_j_bin(int(record["j"]), lo, hi)]
            n_instances = len(cell_rows)
            greedy_valid_count = sum(1 for item in cell_rows if item["greedy_valid"])
            if hi is None:
                j_lt_k_region = False
            else:
                j_lt_k_region = bool(hi < k)

            row_cells.append(
                {
                    "label": label,
                    "n_instances": int(n_instances),
                    "greedy_valid_rate": _safe_rate(greedy_valid_count, n_instances)
                    if n_instances > 0
                    else None,
                    "j_lt_k_region": j_lt_k_region,
                }
            )
        rows.append({"k": int(k), "cells": row_cells})
    return rows


def _build_diagonal_summary(kj_cells: list[dict]) -> list[dict]:
    by_kj = {(int(cell["k"]), int(cell["j"])): cell for cell in kj_cells}
    diagonal: list[dict] = []
    for k in K_VALUES:
        left_cell = by_kj.get((k, k - 1))
        at_cell = by_kj.get((k, k))
        right_cell = by_kj.get((k, k + 1))
        diagonal.append(
            {
                "k": int(k),
                "j_eq_k_minus_1": {
                    "j": int(k - 1),
                    "rate": None if left_cell is None else float(left_cell["greedy_valid_rate"]),
                    "n_instances": 0 if left_cell is None else int(left_cell["n_instances"]),
                },
                "j_eq_k": {
                    "j": int(k),
                    "rate": None if at_cell is None else float(at_cell["greedy_valid_rate"]),
                    "n_instances": 0 if at_cell is None else int(at_cell["n_instances"]),
                },
                "j_eq_k_plus_1": {
                    "j": int(k + 1),
                    "rate": None if right_cell is None else float(right_cell["greedy_valid_rate"]),
                    "n_instances": 0 if right_cell is None else int(right_cell["n_instances"]),
                },
            }
        )
    return diagonal


def _build_verification(
    records: list[dict],
    kj_cells: list[dict],
    k_marginals: list[dict],
) -> dict:
    by_kj = {(int(cell["k"]), int(cell["j"])): cell for cell in kj_cells}
    k0 = next((row for row in k_marginals if int(row["k"]) == 0), None)
    k0_full_valid = bool(k0 is not None and abs(float(k0["greedy_valid_rate"]) - 1.0) <= 1e-12)

    zero_fp_per_k = all(int(row["false_positive_count"]) == 0 for row in k_marginals)

    sharp_diagonal_by_k: list[dict] = []
    for k in K_VALUES:
        below_rates = [
            float(cell["greedy_valid_rate"])
            for cell in kj_cells
            if int(cell["k"]) == k and int(cell["j"]) < k
        ]
        at_or_above_rates = [
            float(cell["greedy_valid_rate"])
            for cell in kj_cells
            if int(cell["k"]) == k and int(cell["j"]) >= k
        ]
        sharp_diagonal_by_k.append(
            {
                "k": int(k),
                "all_j_lt_k_zero": bool(all(rate <= 1e-12 for rate in below_rates)),
                "all_j_ge_k_high_ge_0_95": bool(all(rate >= 0.95 for rate in at_or_above_rates))
                if at_or_above_rates
                else None,
            }
        )

    population_cells: list[dict] = []
    for k in K_VALUES:
        for j in range(6):
            n_instances = sum(
                1 for record in records if int(record["k"]) == k and int(record["j"]) == j
            )
            population_cells.append(
                {
                    "k": int(k),
                    "j": int(j),
                    "n_instances": int(n_instances),
                    "meets_min_10": bool(n_instances >= 10),
                }
            )
    population_ok = all(cell["meets_min_10"] for cell in population_cells)

    return {
        "k0_baseline_full_valid": k0_full_valid,
        "zero_false_positives_per_k": zero_fp_per_k,
        "sharp_diagonal_by_k": sharp_diagonal_by_k,
        "j_population_min_10_per_cell_j0_to_j5": population_ok,
        "population_cells_j0_to_j5": population_cells,
    }


def _evaluate_graph(task: tuple[float, int]) -> dict:
    epsilon, seed = task

    graph = BurstyGenerator().generate(
        BurstyConfig(
            seed=int(seed),
            epsilon=float(epsilon),
            n_events=N_EVENTS,
            n_actors=N_ACTORS,
        )
    )

    oracle_result, _oracle_diag = exact_oracle_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=_oracle_grammar(),
    )
    if not oracle_result.valid:
        return {
            "epsilon": float(epsilon),
            "seed": int(seed),
            "oracle_feasible": False,
            "graph_row": None,
            "records": [],
            "theorem_builtin_mismatch_count": 0,
        }

    focal = _compute_focal_j(graph, FOCAL_ACTOR)
    if focal["j"] is None:
        return {
            "epsilon": float(epsilon),
            "seed": int(seed),
            "oracle_feasible": False,
            "graph_row": None,
            "records": [],
            "theorem_builtin_mismatch_count": 0,
        }

    j_value = int(focal["j"])
    n_focal_events = int(focal["n_focal_events"])
    graph_row = {
        "epsilon": float(epsilon),
        "seed": int(seed),
        "j": int(j_value),
        "n_focal_events": int(n_focal_events),
        "max_weight_event_id": focal["max_weight_event_id"],
    }

    records: list[dict] = []
    theorem_builtin_mismatch_count = 0
    for k in K_VALUES:
        grammar = _make_grammar(k)
        result = greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
        )

        theorem_check = check_precondition(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
        )
        theorem_predicted_failure = bool(theorem_check["predicted_failure"])
        theorem_from_focal_j = _canonical_theorem_prediction(j_value, k)
        if theorem_predicted_failure != theorem_from_focal_j:
            theorem_builtin_mismatch_count += 1

        absorption = diagnose_absorption(result, grammar)
        records.append(
            {
                "epsilon": float(epsilon),
                "seed": int(seed),
                "k": int(k),
                "j": int(j_value),
                "n_focal_events": int(n_focal_events),
                "greedy_valid": bool(result.valid),
                "greedy_score": float(result.score) if result.valid else None,
                "greedy_n_development": int(result.n_development),
                "theorem_predicted_failure": bool(theorem_predicted_failure),
                "theorem_predicted_failure_from_focal_j": bool(theorem_from_focal_j),
                "absorbed": bool(absorption.get("absorbed", False)),
                "j_lt_k": bool(j_value < k),
            }
        )

    return {
        "epsilon": float(epsilon),
        "seed": int(seed),
        "oracle_feasible": True,
        "graph_row": graph_row,
        "records": records,
        "theorem_builtin_mismatch_count": int(theorem_builtin_mismatch_count),
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines: list[str] = [
        "# k-j boundary sweep (Exp 10)",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
    ]

    oracle = data["oracle_feasibility"]
    lines.extend(
        [
            "## Oracle Feasibility",
            "",
            (
                f"- Oracle feasible graphs: {oracle['oracle_feasible']}/{oracle['total_graphs']} "
                f"({100.0 * oracle['feasible_rate']:.1f}%)"
            ),
            f"- Oracle infeasible discarded: {oracle['oracle_infeasible']}",
            "",
        ]
    )

    lines.extend(
        [
            "## k-j Matrix (Greedy Valid Rate)",
            "",
            "| k | 0 | 1 | 2 | 3 | 4 | 5 | 6-10 | 11-20 | 21+ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["kj_matrix_binned"]:
        rendered_cells = []
        for cell in row["cells"]:
            if cell["greedy_valid_rate"] is None:
                rendered_cells.append("n/a")
            else:
                marker = "*" if cell["j_lt_k_region"] else ""
                rendered_cells.append(
                    f"{marker}{cell['greedy_valid_rate']:.3f} ({cell['n_instances']})"
                )
        lines.append(f"| {row['k']} | " + " | ".join(rendered_cells) + " |")
    lines.extend(["", "*`*` marks `j < k` cells (theorem-predicted failure region).", ""])

    lines.extend(["## Diagonal Boundary Check", ""])
    for row in data["diagonal_boundary"]:
        k = int(row["k"])
        left = row["j_eq_k_minus_1"]
        center = row["j_eq_k"]
        right = row["j_eq_k_plus_1"]
        lines.append(
            f"- k={k}: "
            f"j={left['j']} valid={_format_rate(left['rate'])} (n={left['n_instances']}) | "
            f"j={center['j']} valid={_format_rate(center['rate'])} (n={center['n_instances']}) | "
            f"j={right['j']} valid={_format_rate(right['rate'])} (n={right['n_instances']})"
        )
    lines.append("")

    theorem_overall = data["theorem_overall"]
    lines.extend(
        [
            "## Theorem Accuracy",
            "",
            (
                f"- Overall accuracy: {theorem_overall['accuracy']:.3f} "
                f"(FP={theorem_overall['false_positive_count']}, "
                f"FN={theorem_overall['false_negative_count']})"
            ),
            "",
            "| k | n | greedy_valid_rate | theorem_accuracy | FP_rate | FN_rate | mean_j |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["k_marginals"]:
        lines.append(
            "| {k} | {n} | {valid:.3f} | {acc:.3f} | {fp:.3f} | {fn:.3f} | {mean_j} |".format(
                k=row["k"],
                n=row["n_instances"],
                valid=row["greedy_valid_rate"],
                acc=row["theorem_accuracy"],
                fp=row["false_positive_rate"],
                fn=row["false_negative_rate"],
                mean_j=f"{row['mean_j']:.2f}" if row["mean_j"] is not None else "n/a",
            )
        )
    lines.append("")

    j_dist = data["j_distribution"]
    lines.extend(
        [
            "## j Distribution",
            "",
            f"- Mean: {j_dist['mean']:.3f}" if j_dist["mean"] is not None else "- Mean: n/a",
            f"- Median: {j_dist['median']:.3f}" if j_dist["median"] is not None else "- Median: n/a",
            f"- Min / Max: {j_dist['min']} / {j_dist['max']}",
            "",
            "Histogram:",
            "",
            "| j | count |",
            "|---|---:|",
        ]
    )
    for j_key, count in j_dist["histogram"].items():
        lines.append(f"| {j_key} | {count} |")
    lines.append("")

    verification = data["verification"]
    lines.extend(
        [
            "## Verification Checks",
            "",
            (
                f"- k=0 baseline full validity: "
                f"{'yes' if verification['k0_baseline_full_valid'] else 'no'}"
            ),
            (
                f"- Zero false positives per k: "
                f"{'yes' if verification['zero_false_positives_per_k'] else 'no'}"
            ),
            (
                f"- j=0..5 cell population >=10: "
                f"{'yes' if verification['j_population_min_10_per_cell_j0_to_j5'] else 'no'}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_kj_boundary() -> dict:
    timer = ExperimentTimer()

    records: list[dict] = []
    graph_level_rows: list[dict] = []

    tasks = [(epsilon, seed) for epsilon in EPSILONS for seed in SEEDS]
    total_graphs = len(tasks)
    oracle_calls = total_graphs
    greedy_calls = 0
    oracle_feasible = 0
    oracle_infeasible = 0
    theorem_builtin_mismatch_count = 0

    cpu_count = os.cpu_count() or 1
    max_workers = min(12, max(1, cpu_count - 1))
    print(
        f"Running {len(tasks)} graph evaluations with {max_workers} workers...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, payload in enumerate(executor.map(_evaluate_graph, tasks, chunksize=1), start=1):
            if bool(payload["oracle_feasible"]):
                oracle_feasible += 1
                graph_level_rows.append(payload["graph_row"])
                rows = payload["records"]
                records.extend(rows)
                greedy_calls += len(rows)
            else:
                oracle_infeasible += 1
            theorem_builtin_mismatch_count += int(payload["theorem_builtin_mismatch_count"])

            if idx % 50 == 0 or idx == len(tasks):
                print(
                    f"Progress: {idx}/{len(tasks)} graphs | oracle_feasible={oracle_feasible}",
                    flush=True,
                )

    kj_matrix = _aggregate_kj_cells(records)
    k_marginals = _aggregate_k_marginals(records)
    j_distribution = _aggregate_j_distribution(graph_level_rows)
    kj_matrix_binned = _build_binned_kj_matrix(records)
    diagonal_boundary = _build_diagonal_summary(kj_matrix)
    verification = _build_verification(records, kj_matrix, k_marginals)

    total_instances = len(records)
    theorem_fp = sum(
        1 for row in records if row["theorem_predicted_failure"] and row["greedy_valid"]
    )
    theorem_fn = sum(
        1 for row in records if (not row["theorem_predicted_failure"]) and (not row["greedy_valid"])
    )
    theorem_correct = sum(
        1
        for row in records
        if bool(row["theorem_predicted_failure"]) == (not bool(row["greedy_valid"]))
    )

    data = {
        "parameters": {
            "epsilons": EPSILONS,
            "k_range": K_VALUES,
            "seeds_per_epsilon": len(SEEDS),
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_actors": N_ACTORS,
            "n_events": N_EVENTS,
            "focal_actor": FOCAL_ACTOR,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "constraints_disabled": ["min_timespan_fraction", "max_temporal_gap"],
        },
        "oracle_feasibility": {
            "total_graphs": int(total_graphs),
            "oracle_feasible": int(oracle_feasible),
            "oracle_infeasible": int(oracle_infeasible),
            "feasible_rate": _safe_rate(oracle_feasible, total_graphs),
        },
        "records": records,
        "kj_matrix": kj_matrix,
        "k_marginals": k_marginals,
        "j_distribution": j_distribution,
        "kj_matrix_binned": kj_matrix_binned,
        "diagonal_boundary": diagonal_boundary,
        "theorem_overall": {
            "n_instances": int(total_instances),
            "predicted_failure_count": int(sum(1 for row in records if row["theorem_predicted_failure"])),
            "false_positive_count": int(theorem_fp),
            "false_negative_count": int(theorem_fn),
            "accuracy": _safe_rate(theorem_correct, total_instances),
            "check_precondition_mismatch_count": int(theorem_builtin_mismatch_count),
        },
        "verification": verification,
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=oracle_calls + greedy_calls,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "epsilons": EPSILONS,
            "k_values": K_VALUES,
            "oracle_calls": oracle_calls,
            "greedy_calls": greedy_calls,
            "oracle_feasible_graphs": oracle_feasible,
            "oracle_infeasible_graphs": oracle_infeasible,
        },
    )
    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)

    print("\nk-j Boundary Sweep")
    print("==================")
    oracle = data["oracle_feasibility"]
    print(
        f"Oracle feasible: {oracle['oracle_feasible']}/{oracle['total_graphs']} "
        f"({100.0 * oracle['feasible_rate']:.1f}%)"
    )
    print("")
    for row in data["k_marginals"]:
        print(
            f"k={row['k']}: valid={100.0 * row['greedy_valid_rate']:.1f}%  "
            f"theorem_acc={row['theorem_accuracy']:.3f}  "
            f"FP={row['false_positive_count']}  FN={row['false_negative_count']}"
        )
    print("")
    print("Diagonal boundary:")
    for row in data["diagonal_boundary"]:
        left = row["j_eq_k_minus_1"]
        center = row["j_eq_k"]
        right = row["j_eq_k_plus_1"]
        print(
            f"  k={row['k']}: "
            f"j={left['j']} valid={_format_rate(left['rate'])} | "
            f"j={center['j']} valid={_format_rate(center['rate'])} | "
            f"j={right['j']} valid={_format_rate(right['rate'])}"
        )

    print("")
    print("Verification:")
    print(f"  k=0 baseline full validity: {verification['k0_baseline_full_valid']}")
    print(f"  zero false positives per k: {verification['zero_false_positives_per_k']}")
    print(
        "  j=0..5 population >=10 per (k,j): "
        f"{verification['j_population_min_10_per_cell_j0_to_j5']}"
    )
    print(
        "  check_precondition mismatches vs focal-j theorem: "
        f"{data['theorem_overall']['check_precondition_mismatch_count']}"
    )

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_kj_boundary()
