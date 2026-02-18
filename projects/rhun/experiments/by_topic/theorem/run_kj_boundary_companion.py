"""Companion analysis for k-j boundary: j_focal vs j_theorem."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph
from rhun.theory.theorem import check_precondition


OUTPUT_NAME = "kj_boundary_companion"
SOURCE_PATH = Path(__file__).resolve().parent / "output" / "kj_boundary.json"

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


def _graph_key(epsilon: float, seed: int) -> str:
    return f"{float(epsilon):.2f}|{int(seed)}"


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _compute_focal_j(graph: CausalGraph, focal_actor: str) -> int | None:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    focal_events_sorted = sorted(
        focal_events,
        key=lambda event: (float(event.timestamp), event.id),
    )
    if not focal_events_sorted:
        return None

    max_weight_event = max(
        focal_events_sorted,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )
    return int(
        next(
            i for i, event in enumerate(focal_events_sorted) if event.id == max_weight_event.id
        )
    )


def _is_in_bin(j: int, lo: int, hi: int | None) -> bool:
    if hi is None:
        return j >= lo
    return lo <= j <= hi


def _aggregate_cells(records: list[dict], j_field: str, pred_field: str) -> list[dict]:
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in records:
        buckets[(int(row["k"]), int(row[j_field]))].append(row)

    cells: list[dict] = []
    for (k, j), bucket in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(bucket)
        valid_count = sum(1 for row in bucket if row["greedy_valid"])
        pred_count = sum(1 for row in bucket if row[pred_field])
        fp = sum(1 for row in bucket if row[pred_field] and row["greedy_valid"])
        fn = sum(1 for row in bucket if (not row[pred_field]) and (not row["greedy_valid"]))
        correct = sum(
            1 for row in bucket if bool(row[pred_field]) == (not bool(row["greedy_valid"]))
        )
        cells.append(
            {
                "k": int(k),
                "j": int(j),
                "n_instances": int(n),
                "greedy_valid_count": int(valid_count),
                "greedy_valid_rate": _safe_rate(valid_count, n),
                "predicted_failure_count": int(pred_count),
                "theorem_accuracy": _safe_rate(correct, n),
                "false_positive_count": int(fp),
                "false_negative_count": int(fn),
                "j_lt_k": bool(j < k),
            }
        )
    return cells


def _aggregate_marginals(records: list[dict], pred_field: str, k_values: list[int]) -> list[dict]:
    rows: list[dict] = []
    for k in k_values:
        bucket = [row for row in records if int(row["k"]) == int(k)]
        n = len(bucket)
        valid_count = sum(1 for row in bucket if row["greedy_valid"])
        fp = sum(1 for row in bucket if row[pred_field] and row["greedy_valid"])
        fn = sum(1 for row in bucket if (not row[pred_field]) and (not row["greedy_valid"]))
        correct = sum(
            1 for row in bucket if bool(row[pred_field]) == (not bool(row["greedy_valid"]))
        )
        rows.append(
            {
                "k": int(k),
                "n_instances": int(n),
                "greedy_valid_rate": _safe_rate(valid_count, n),
                "theorem_accuracy": _safe_rate(correct, n),
                "false_positive_count": int(fp),
                "false_positive_rate": _safe_rate(fp, n),
                "false_negative_count": int(fn),
                "false_negative_rate": _safe_rate(fn, n),
            }
        )
    return rows


def _binned_matrix(records: list[dict], j_field: str, k_values: list[int]) -> list[dict]:
    rows: list[dict] = []
    for k in k_values:
        k_rows = [row for row in records if int(row["k"]) == int(k)]
        cells: list[dict] = []
        for label, lo, hi in J_BINS:
            bucket = [row for row in k_rows if _is_in_bin(int(row[j_field]), lo, hi)]
            n = len(bucket)
            valid_count = sum(1 for row in bucket if row["greedy_valid"])
            cells.append(
                {
                    "label": label,
                    "n_instances": int(n),
                    "greedy_valid_rate": _safe_rate(valid_count, n) if n > 0 else None,
                    "j_lt_k_region": bool(False if hi is None else hi < k),
                }
            )
        rows.append({"k": int(k), "cells": cells})
    return rows


def _diagonal(cells: list[dict], k_values: list[int]) -> list[dict]:
    by_kj = {(int(row["k"]), int(row["j"])): row for row in cells}
    rows: list[dict] = []
    for k in k_values:
        left = by_kj.get((k, k - 1))
        at = by_kj.get((k, k))
        right = by_kj.get((k, k + 1))
        rows.append(
            {
                "k": int(k),
                "j_eq_k_minus_1": {
                    "j": int(k - 1),
                    "rate": None if left is None else float(left["greedy_valid_rate"]),
                    "n_instances": 0 if left is None else int(left["n_instances"]),
                },
                "j_eq_k": {
                    "j": int(k),
                    "rate": None if at is None else float(at["greedy_valid_rate"]),
                    "n_instances": 0 if at is None else int(at["n_instances"]),
                },
                "j_eq_k_plus_1": {
                    "j": int(k + 1),
                    "rate": None if right is None else float(right["greedy_valid_rate"]),
                    "n_instances": 0 if right is None else int(right["n_instances"]),
                },
            }
        )
    return rows


def _overall(records: list[dict], pred_field: str) -> dict:
    n = len(records)
    fp = sum(1 for row in records if row[pred_field] and row["greedy_valid"])
    fn = sum(1 for row in records if (not row[pred_field]) and (not row["greedy_valid"]))
    correct = sum(
        1 for row in records if bool(row[pred_field]) == (not bool(row["greedy_valid"]))
    )
    return {
        "n_instances": int(n),
        "predicted_failure_count": int(sum(1 for row in records if row[pred_field])),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "accuracy": _safe_rate(correct, n),
    }


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines: list[str] = [
        "# k-j boundary companion report",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Definitions",
        "",
        "- `j_focal`: temporal rank of max-weight focal event among focal-actor events",
        "- `j_theorem`: `check_precondition(...)[\"events_before_max\"]`",
        "",
        "## Mismatch",
        "",
        (
            f"- Graphs with `j_focal != j_theorem`: "
            f"{data['j_comparison']['graph_mismatch_count']}/{data['j_comparison']['n_graphs']} "
            f"({100.0 * data['j_comparison']['graph_mismatch_rate']:.1f}%)"
        ),
        (
            f"- Mean |j_focal - j_theorem|: "
            f"{data['j_comparison']['mean_abs_delta']:.3f}"
        ),
        (
            f"- Record-level prediction mismatches "
            f"(`j_focal < k` vs `j_theorem < k`): "
            f"{data['definition_disagreement']['count']}/{data['n_records']}"
        ),
        "",
        "## Overall Theorem Metrics",
        "",
        (
            f"- Using `j_focal`: acc={data['focal_definition']['overall']['accuracy']:.3f}, "
            f"FP={data['focal_definition']['overall']['false_positive_count']}, "
            f"FN={data['focal_definition']['overall']['false_negative_count']}"
        ),
        (
            f"- Using `j_theorem`: acc={data['theorem_definition']['overall']['accuracy']:.3f}, "
            f"FP={data['theorem_definition']['overall']['false_positive_count']}, "
            f"FN={data['theorem_definition']['overall']['false_negative_count']}"
        ),
        "",
        "## k-j Matrix (j_theorem, Greedy Valid Rate)",
        "",
        "| k | 0 | 1 | 2 | 3 | 4 | 5 | 6-10 | 11-20 | 21+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["theorem_definition"]["kj_matrix_binned"]:
        rendered = []
        for cell in row["cells"]:
            if cell["greedy_valid_rate"] is None:
                rendered.append("n/a")
            else:
                marker = "*" if cell["j_lt_k_region"] else ""
                rendered.append(f"{marker}{cell['greedy_valid_rate']:.3f} ({cell['n_instances']})")
        lines.append(f"| {row['k']} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "*`*` marks bins fully inside `j < k`.",
            "",
            "## Per-k Comparison",
            "",
            "| k | focal_acc | focal_FP | focal_FN | theorem_acc | theorem_FP | theorem_FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    focal_by_k = {int(row["k"]): row for row in data["focal_definition"]["k_marginals"]}
    theorem_by_k = {int(row["k"]): row for row in data["theorem_definition"]["k_marginals"]}
    for k in data["k_values"]:
        f = focal_by_k[int(k)]
        t = theorem_by_k[int(k)]
        lines.append(
            f"| {k} | {f['theorem_accuracy']:.3f} | {f['false_positive_count']} | "
            f"{f['false_negative_count']} | {t['theorem_accuracy']:.3f} | "
            f"{t['false_positive_count']} | {t['false_negative_count']} |"
        )

    lines.extend(["", "## Diagonal Boundary (j_theorem)", ""])
    for row in data["theorem_definition"]["diagonal"]:
        left = row["j_eq_k_minus_1"]
        center = row["j_eq_k"]
        right = row["j_eq_k_plus_1"]
        lines.append(
            f"- k={row['k']}: "
            f"j={left['j']} valid={_format_rate(left['rate'])} (n={left['n_instances']}) | "
            f"j={center['j']} valid={_format_rate(center['rate'])} (n={center['n_instances']}) | "
            f"j={right['j']} valid={_format_rate(right['rate'])} (n={right['n_instances']})"
        )
    lines.append("")

    lines.extend(
        [
            "## Consistency",
            "",
            (
                f"- `check_precondition` predicted_failure matches "
                f"`j_theorem < k` on records: "
                f"{data['consistency']['predicted_failure_match_count']}/"
                f"{data['consistency']['n_records']} "
                f"(mismatch={data['consistency']['predicted_failure_mismatch_count']})"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_kj_boundary_companion() -> dict:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing source results: {SOURCE_PATH}. Run experiments/run_kj_boundary.py first."
        )

    timer = ExperimentTimer()

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    data = source["results"]
    params = data["parameters"]
    focal_actor = str(params["focal_actor"])
    n_events = int(params["n_events"])
    n_actors = int(params["n_actors"])
    k_values = [int(k) for k in params["k_range"]]

    base_records = data["records"]
    graph_keys = sorted(
        {
            _graph_key(float(row["epsilon"]), int(row["seed"]))
            for row in base_records
        }
    )

    generator = BurstyGenerator()
    graph_comparison: dict[str, dict] = {}

    for key in graph_keys:
        epsilon_str, seed_str = key.split("|")
        epsilon = float(epsilon_str)
        seed = int(seed_str)

        graph = generator.generate(
            BurstyConfig(
                seed=seed,
                epsilon=epsilon,
                n_events=n_events,
                n_actors=n_actors,
            )
        )

        j_focal = _compute_focal_j(graph, focal_actor)
        theorem_probe = check_precondition(
            graph=graph,
            focal_actor=focal_actor,
            grammar=_make_grammar(0),
        )
        j_theorem = int(theorem_probe["events_before_max"])

        graph_comparison[key] = {
            "epsilon": epsilon,
            "seed": seed,
            "j_focal_recomputed": None if j_focal is None else int(j_focal),
            "j_theorem": int(j_theorem),
            "theorem_max_weight_index": int(theorem_probe["max_weight_index"]),
            "theorem_max_weight_position": float(theorem_probe["max_weight_position"]),
            "delta_j": None if j_focal is None else int(j_focal - j_theorem),
            "abs_delta_j": None if j_focal is None else int(abs(j_focal - j_theorem)),
            "j_equal": bool(j_focal == j_theorem),
        }

    comparison_records: list[dict] = []
    predicted_mismatch_count = 0
    disagreement_by_k: dict[int, int] = defaultdict(int)
    for row in base_records:
        key = _graph_key(float(row["epsilon"]), int(row["seed"]))
        g = graph_comparison[key]
        j_focal = int(row["j"])
        j_theorem = int(g["j_theorem"])
        k = int(row["k"])

        pred_focal = bool(j_focal < k)
        pred_theorem = bool(j_theorem < k)
        pred_check = bool(row["theorem_predicted_failure"])
        if pred_theorem != pred_check:
            predicted_mismatch_count += 1
        if pred_focal != pred_theorem:
            disagreement_by_k[k] += 1

        comparison_records.append(
            {
                "epsilon": float(row["epsilon"]),
                "seed": int(row["seed"]),
                "k": int(k),
                "greedy_valid": bool(row["greedy_valid"]),
                "greedy_score": row["greedy_score"],
                "j_focal": int(j_focal),
                "j_theorem": int(j_theorem),
                "j_delta": int(j_focal - j_theorem),
                "pred_failure_from_j_focal": pred_focal,
                "pred_failure_from_j_theorem": pred_theorem,
                "pred_failure_from_check_precondition": pred_check,
                "pred_check_matches_j_theorem": bool(pred_theorem == pred_check),
            }
        )

    focal_cells = _aggregate_cells(
        comparison_records,
        j_field="j_focal",
        pred_field="pred_failure_from_j_focal",
    )
    theorem_cells = _aggregate_cells(
        comparison_records,
        j_field="j_theorem",
        pred_field="pred_failure_from_j_theorem",
    )

    focal_marginals = _aggregate_marginals(
        comparison_records,
        pred_field="pred_failure_from_j_focal",
        k_values=k_values,
    )
    theorem_marginals = _aggregate_marginals(
        comparison_records,
        pred_field="pred_failure_from_j_theorem",
        k_values=k_values,
    )

    deltas = [
        int(info["abs_delta_j"])
        for info in graph_comparison.values()
        if info["abs_delta_j"] is not None
    ]
    mismatch_count = sum(1 for info in graph_comparison.values() if not info["j_equal"])

    result = {
        "source_file": str(SOURCE_PATH),
        "k_values": k_values,
        "n_records": int(len(comparison_records)),
        "n_graphs": int(len(graph_comparison)),
        "j_comparison": {
            "n_graphs": int(len(graph_comparison)),
            "graph_mismatch_count": int(mismatch_count),
            "graph_mismatch_rate": _safe_rate(mismatch_count, len(graph_comparison)),
            "mean_abs_delta": float(mean(deltas)) if deltas else 0.0,
            "median_abs_delta": float(median(deltas)) if deltas else 0.0,
            "max_abs_delta": int(max(deltas)) if deltas else 0,
            "graph_level": sorted(graph_comparison.values(), key=lambda row: (row["epsilon"], row["seed"])),
        },
        "records": comparison_records,
        "focal_definition": {
            "overall": _overall(comparison_records, pred_field="pred_failure_from_j_focal"),
            "kj_matrix": focal_cells,
            "k_marginals": focal_marginals,
            "kj_matrix_binned": _binned_matrix(comparison_records, j_field="j_focal", k_values=k_values),
            "diagonal": _diagonal(focal_cells, k_values=k_values),
        },
        "theorem_definition": {
            "overall": _overall(comparison_records, pred_field="pred_failure_from_j_theorem"),
            "kj_matrix": theorem_cells,
            "k_marginals": theorem_marginals,
            "kj_matrix_binned": _binned_matrix(comparison_records, j_field="j_theorem", k_values=k_values),
            "diagonal": _diagonal(theorem_cells, k_values=k_values),
        },
        "consistency": {
            "n_records": int(len(comparison_records)),
            "predicted_failure_match_count": int(len(comparison_records) - predicted_mismatch_count),
            "predicted_failure_mismatch_count": int(predicted_mismatch_count),
        },
        "definition_disagreement": {
            "count": int(sum(disagreement_by_k.values())),
            "by_k": {str(k): int(v) for k, v in sorted(disagreement_by_k.items())},
        },
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=int(len(graph_comparison)),
        n_extractions=0,
        seed_range=(0, 99),
        parameters={
            "source_file": str(SOURCE_PATH),
            "definitions": ["j_focal", "j_theorem"],
        },
    )
    save_results(OUTPUT_NAME, result, metadata, summary_formatter=_summary_markdown)

    print("k-j Boundary Companion")
    print("======================")
    print(
        f"Graphs with j mismatch: {mismatch_count}/{len(graph_comparison)} "
        f"({100.0 * _safe_rate(mismatch_count, len(graph_comparison)):.1f}%)"
    )
    focal_overall = result["focal_definition"]["overall"]
    theorem_overall = result["theorem_definition"]["overall"]
    print(
        "Focal-j theorem metrics: "
        f"acc={focal_overall['accuracy']:.3f} FP={focal_overall['false_positive_count']} "
        f"FN={focal_overall['false_negative_count']}"
    )
    print(
        "Theorem-j theorem metrics: "
        f"acc={theorem_overall['accuracy']:.3f} FP={theorem_overall['false_positive_count']} "
        f"FN={theorem_overall['false_negative_count']}"
    )
    print(
        "check_precondition consistency: "
        f"{result['consistency']['predicted_failure_match_count']}/"
        f"{result['consistency']['n_records']} "
        f"(mismatch={result['consistency']['predicted_failure_mismatch_count']})"
    )
    print(
        "definition disagreement (j_focal<k vs j_theorem<k): "
        f"{result['definition_disagreement']['count']}/{result['n_records']} "
        f"by_k={result['definition_disagreement']['by_k']}"
    )

    print("\nDiagonal (j_theorem):")
    for row in result["theorem_definition"]["diagonal"]:
        left = row["j_eq_k_minus_1"]
        center = row["j_eq_k"]
        right = row["j_eq_k_plus_1"]
        print(
            f"  k={row['k']}: "
            f"j={left['j']} valid={_format_rate(left['rate'])} | "
            f"j={center['j']} valid={_format_rate(center['rate'])} | "
            f"j={right['j']} valid={_format_rate(right['rate'])}"
        )

    return {"metadata": metadata, "results": result}


if __name__ == "__main__":
    run_kj_boundary_companion()
