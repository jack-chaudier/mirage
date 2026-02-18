"""Run k-j boundary sweep across multiple epsilon values."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.theory.theorem import check_precondition


OUTPUT_NAME = "kj_boundary_multi_epsilon"
BASELINE_PATH = Path(__file__).resolve().parent / "output" / "kj_boundary.json"


def _compute_rows(
    epsilon: float,
    seeds: range,
    k_values: range,
    n_events: int,
    n_actors: int,
) -> tuple[list[dict], int, int]:
    generator = BurstyGenerator()
    bucket: dict[tuple[int, int], list[int]] = defaultdict(list)

    total_graphs = 0
    total_extractions = 0

    for seed in seeds:
        graph = generator.generate(
            BurstyConfig(
                seed=seed,
                epsilon=epsilon,
                n_events=n_events,
                n_actors=n_actors,
            )
        )
        total_graphs += 1

        for actor in sorted(graph.actors):
            for k in k_values:
                grammar = GrammarConfig.parametric(k)
                extraction = greedy_extract(graph, actor, grammar, pool_strategy="injection")
                precondition = check_precondition(graph, actor, grammar)
                j = int(precondition["events_before_max"])

                bucket[(k, j)].append(int(extraction.valid))
                total_extractions += 1

    rows: list[dict] = []
    for (k, j), outcomes in sorted(bucket.items()):
        n = len(outcomes)
        rows.append(
            {
                "k": k,
                "j": j,
                "n": n,
                "success_rate": (sum(outcomes) / n) if n else 0.0,
            }
        )

    return rows, total_graphs, total_extractions


def _rows_to_grid(rows: list[dict]) -> dict[str, dict[str, float]]:
    grid: dict[str, dict[str, float]] = {}
    for row in rows:
        k_key = str(row["k"])
        j_key = str(row["j"])
        grid.setdefault(k_key, {})[j_key] = float(row["success_rate"])
    return grid


def _weighted_success(rows: list[dict]) -> float:
    total_n = sum(row["n"] for row in rows)
    if total_n == 0:
        return 0.0
    return sum(row["success_rate"] * row["n"] for row in rows) / total_n


def _summary_for_epsilon(rows: list[dict], k_values: range) -> list[dict]:
    summary: list[dict] = []

    for k in k_values:
        k_rows = [row for row in rows if row["k"] == k]
        below = [row for row in k_rows if row["j"] < k]
        above = [row for row in k_rows if row["j"] >= k]

        summary.append(
            {
                "k": k,
                "n_j_cells": len(k_rows),
                "mean_success_all_j": _weighted_success(k_rows),
                "success_j_lt_k": _weighted_success(below) if below else None,
                "success_j_ge_k": _weighted_success(above) if above else None,
                "n_below": int(sum(row["n"] for row in below)),
                "n_above": int(sum(row["n"] for row in above)),
            }
        )

    return summary


def _format_kj_table(rows: list[dict]) -> str:
    lines = [
        "| k | j | n | success_rate |",
        "|---|---|---|--------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['j']} | {row['n']} | {row['success_rate']:.3f} |"
        )
    return "\n".join(lines)


def _format_summary_table(summary_rows: list[dict]) -> str:
    lines = [
        "| k | n_j_cells | mean_success_all_j | success_j_lt_k | success_j_ge_k | n_below | n_above |",
        "|---|-----------|--------------------|---------------|---------------|--------:|--------:|",
    ]
    for row in summary_rows:
        below = "n/a" if row["success_j_lt_k"] is None else f"{row['success_j_lt_k']:.3f}"
        above = "n/a" if row["success_j_ge_k"] is None else f"{row['success_j_ge_k']:.3f}"
        lines.append(
            f"| {row['k']} | {row['n_j_cells']} | {row['mean_success_all_j']:.3f} | "
            f"{below} | {above} | {row['n_below']} | {row['n_above']} |"
        )
    return "\n".join(lines)


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"Epsilons: {data['epsilons']}",
        "",
    ]

    for eps in data["epsilons"]:
        eps_key = f"{eps:.2f}"
        lines.extend(
            [
                f"## Epsilon {eps_key}",
                "",
                _format_summary_table(data["summary_by_epsilon"][eps_key]),
                "",
            ]
        )

    return "\n".join(lines)


def _verify_epsilon_050(rows_050: list[dict]) -> None:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Expected baseline file missing for verification: {BASELINE_PATH}"
        )

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_rows = baseline["results"]["rows"]

    baseline_map = {(int(r["k"]), int(r["j"])): (int(r["n"]), float(r["success_rate"])) for r in baseline_rows}
    current_map = {(int(r["k"]), int(r["j"])): (int(r["n"]), float(r["success_rate"])) for r in rows_050}

    if baseline_map.keys() != current_map.keys():
        missing = sorted(set(baseline_map.keys()) - set(current_map.keys()))
        extra = sorted(set(current_map.keys()) - set(baseline_map.keys()))
        raise RuntimeError(
            "epsilon=0.50 verification failed: key mismatch. "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    mismatches: list[str] = []
    for key in sorted(baseline_map.keys()):
        b_n, b_sr = baseline_map[key]
        c_n, c_sr = current_map[key]
        if b_n != c_n or abs(b_sr - c_sr) > 1e-12:
            mismatches.append(
                f"{key}: baseline(n={b_n},sr={b_sr}) != current(n={c_n},sr={c_sr})"
            )
            if len(mismatches) >= 10:
                break

    if mismatches:
        raise RuntimeError(
            "epsilon=0.50 verification failed: value mismatch. " + "; ".join(mismatches)
        )


def run_kj_boundary_multi_epsilon() -> dict:
    epsilons = [0.30, 0.50, 0.70, 0.90]
    seeds = range(1, 51)
    k_values = range(0, 6)
    n_events = 200
    n_actors = 6

    timer = ExperimentTimer()

    rows_by_epsilon: dict[str, list[dict]] = {}
    summary_by_epsilon: dict[str, list[dict]] = {}
    grid_by_epsilon: dict[str, dict[str, dict[str, float]]] = {}

    total_graphs = 0
    total_extractions = 0

    for eps in epsilons:
        rows, n_graphs, n_extractions = _compute_rows(
            epsilon=eps,
            seeds=seeds,
            k_values=k_values,
            n_events=n_events,
            n_actors=n_actors,
        )
        eps_key = f"{eps:.2f}"
        rows_by_epsilon[eps_key] = rows
        summary_by_epsilon[eps_key] = _summary_for_epsilon(rows, k_values)
        grid_by_epsilon[eps_key] = _rows_to_grid(rows)

        total_graphs += n_graphs
        total_extractions += n_extractions

    _verify_epsilon_050(rows_by_epsilon["0.50"])

    data = {
        "epsilons": epsilons,
        "seed_range": [min(seeds), max(seeds)],
        "k_values": list(k_values),
        "n_events": n_events,
        "n_actors": n_actors,
        "rows_by_epsilon": rows_by_epsilon,
        "grid_by_epsilon": grid_by_epsilon,
        "summary_by_epsilon": summary_by_epsilon,
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions,
        seed_range=(min(seeds), max(seeds)),
        parameters={
            "epsilons": epsilons,
            "k_values": list(k_values),
            "n_events": n_events,
            "n_actors": n_actors,
            "pool_strategy": "injection",
            "verification_against_kj_boundary_epsilon_050": True,
        },
    )

    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)

    print("epsilon=0.50 verification: OK (matches experiments/output/kj_boundary.json)")
    for eps in epsilons:
        eps_key = f"{eps:.2f}"
        print(f"\n## epsilon={eps_key}")
        print(_format_kj_table(rows_by_epsilon[eps_key]))

    print("\n## summary_by_epsilon")
    for eps in epsilons:
        eps_key = f"{eps:.2f}"
        print(f"\n### epsilon={eps_key}")
        print(_format_summary_table(summary_by_epsilon[eps_key]))

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_kj_boundary_multi_epsilon()
