"""Final k-j boundary sweep with j_dev_pool and j_dev_output diagnostics."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib
import numpy as np

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, Phase


matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_NAME = "kj_boundary_final"
HEATMAP_FINAL_NAME = "kj_boundary_final_heatmap.png"
HEATMAP_JFOCAL_NAME = "kj_boundary_final_heatmap_jfocal.png"

FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
SEEDS = range(100)
K_VALUES = [0, 1, 2, 3, 4, 5]
N_EVENTS = 200
N_ACTORS = 6

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

HEATMAP_J_MAX = 10  # 0..9 + 10+
TARGET_HASH_SEED = "1"


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == TARGET_HASH_SEED:
        return

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = TARGET_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _make_grammar(k: int) -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=k,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )


def _safe_rate(numer: int | float, denom: int | float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str) -> Event:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    if not focal_events:
        raise ValueError(f"No focal events for actor={focal_actor}")
    return max(
        focal_events,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )


def _j_focal(
    graph: CausalGraph,
    focal_actor: str,
    tp_timestamp: float,
) -> int:
    return int(
        sum(
            1
            for event in graph.events
            if focal_actor in event.actors and float(event.timestamp) < float(tp_timestamp)
        )
    )


def _j_dev_pool(
    n_pre_tp_all_events: int,
    k: int,
) -> int:
    n_setup = ceil(0.2 * n_pre_tp_all_events) if n_pre_tp_all_events > 0 else 0
    if k > 0:
        max_setup_for_development = max(0, n_pre_tp_all_events - k)
        n_setup = min(n_setup, max_setup_for_development)
    n_dev = n_pre_tp_all_events - n_setup
    return int(n_dev)


def _j_dev_output(result) -> int:
    tp_idx = next(
        (idx for idx, phase in enumerate(result.phases) if phase == Phase.TURNING_POINT),
        None,
    )
    if tp_idx is None:
        return 0
    return int(sum(1 for phase in result.phases[:tp_idx] if phase == Phase.DEVELOPMENT))


def _heatmap_row(
    by_j_counts: dict[int, dict[str, int]],
) -> list[float | None]:
    row: list[float | None] = []
    for bin_idx in range(HEATMAP_J_MAX):
        cell = by_j_counts.get(bin_idx)
        if cell is None:
            row.append(None)
        else:
            row.append(_safe_rate(cell["greedy_valid"], cell["n"]))

    big_cells = [payload for j, payload in by_j_counts.items() if j >= HEATMAP_J_MAX]
    if big_cells:
        n = sum(payload["n"] for payload in big_cells)
        valid = sum(payload["greedy_valid"] for payload in big_cells)
        row.append(_safe_rate(valid, n))
    else:
        row.append(None)
    return row


def _render_heatmap(
    matrix: list[list[float | None]],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    values = np.array(
        [[np.nan if value is None else float(value) for value in row] for row in matrix],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto", origin="upper")

    x_labels = [str(i) for i in range(HEATMAP_J_MAX)] + [f"{HEATMAP_J_MAX}+"]
    y_labels = [str(k) for k in K_VALUES]

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("k (min_prefix_elements)")
    ax.set_title(title)

    ax.plot([0, 5], [0, 5], linestyle="--", linewidth=1.6, color="black")

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isnan(value):
                text = "n/a"
                color = "black"
            else:
                text = f"{value:.2f}"
                color = "black" if value >= 0.60 else "white"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Greedy validity rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    results_by_k = data["results_by_k"]
    heatmap_pool = data["heatmap_j_dev_pool"]["matrix"]
    j_bins = data["heatmap_j_dev_pool"]["j_bins"]

    lines: list[str] = [
        "# k-j boundary final (j_dev_pool + j_dev_output)",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Heatmap (j_dev_pool): `{data['heatmap_j_dev_pool']['file']}`",
        f"Heatmap (j_focal): `{data['heatmap_j_focal']['file']}`",
        "",
        "## Summary Table",
        "",
        (
            "| k | valid_rate | theorem_accuracy_j_focal | theorem_accuracy_j_dev_output | "
            "theorem_accuracy_j_dev_pool | FP_j_focal | FP_j_dev_output | FP_j_dev_pool |"
        ),
        "|---|-----------:|-------------------------:|------------------------------:|----------------------------:|----------:|---------------:|-------------:|",
    ]

    for k in K_VALUES:
        row = results_by_k[str(k)]
        lines.append(
            f"| {k} | {row['greedy_valid_rate']:.3f} | "
            f"{row['theorem_metrics']['j_focal']['accuracy']:.3f} | "
            f"{row['theorem_metrics']['j_dev_output']['accuracy']:.3f} | "
            f"{row['theorem_metrics']['j_dev_pool']['accuracy']:.3f} | "
            f"{row['theorem_metrics']['j_focal']['false_positives']} | "
            f"{row['theorem_metrics']['j_dev_output']['false_positives']} | "
            f"{row['theorem_metrics']['j_dev_pool']['false_positives']} |"
        )

    lines.extend(
        [
            "",
            "## Heatmap Matrix (j_dev_pool, bins 0-10+)",
            "",
            "| k | " + " | ".join(j_bins) + " |",
            "|---|" + "|".join(["---:"] * len(j_bins)) + "|",
        ]
    )
    for k, row in zip(K_VALUES, heatmap_pool, strict=True):
        rendered = ["n/a" if value is None else f"{value:.3f}" for value in row]
        lines.append(f"| {k} | " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "## j_dev_pool Boundary Analysis",
            "",
            "| k | below_weighted | below_max_cell | diagonal | above_weighted | above_min_cell |",
            "|---|---------------:|---------------:|---------:|---------------:|---------------:|",
        ]
    )
    for row in data["j_dev_pool_boundary"]["by_k"]:
        lines.append(
            f"| {row['k']} | {_format_rate(row['below_weighted_rate'])} | "
            f"{_format_rate(row['below_max_cell_rate'])} | {_format_rate(row['diagonal_rate'])} | "
            f"{_format_rate(row['above_weighted_rate'])} | {_format_rate(row['above_min_cell_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## FP Counts by j Definition",
            "",
            f"- FP_j_focal: {data['fp_counts_total']['j_focal']}",
            f"- FP_j_dev_output: {data['fp_counts_total']['j_dev_output']}",
            f"- FP_j_dev_pool: {data['fp_counts_total']['j_dev_pool']}",
            "",
            "## Zero-FP Statement",
            "",
            f"- {data['zero_fp_statement']}",
            "",
            "## Pass/Fail Criteria",
            "",
            f"- j_dev_pool or j_dev_output FP count == 0: {data['criteria']['zero_fp_dev_definition']}",
            f"- Below diagonal (j_dev, k>=2) all cells <= 2%: {data['criteria']['below_diagonal_ok_k_ge_2']}",
            f"- Above diagonal (j_dev >= k+2) all cells >= 90%: {data['criteria']['above_diagonal_ok_j_ge_k_plus_2']}",
            f"- k=0 validity is 100%: {data['criteria']['k0_full_validity']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_kj_boundary_final() -> dict:
    _ensure_hash_seed()
    timer = ExperimentTimer()
    generator = BurstyGenerator()

    by_k_j: dict[int, dict[str, dict[int, dict[str, int]]]] = {
        k: {
            "j_focal": defaultdict(lambda: {"n": 0, "greedy_valid": 0}),
            "j_dev_pool": defaultdict(lambda: {"n": 0, "greedy_valid": 0}),
            "j_dev_output": defaultdict(lambda: {"n": 0, "greedy_valid": 0}),
        }
        for k in K_VALUES
    }

    theorem_counts: dict[int, dict[str, dict[str, int]]] = {
        k: {
            "j_focal": {"correct": 0, "fp": 0, "fn": 0, "n": 0},
            "j_dev_pool": {"correct": 0, "fp": 0, "fn": 0, "n": 0},
            "j_dev_output": {"correct": 0, "fp": 0, "fn": 0, "n": 0},
        }
        for k in K_VALUES
    }

    totals: dict[int, dict[str, int]] = {
        k: {"total_instances": 0, "greedy_valid_count": 0} for k in K_VALUES
    }

    tp_identity_match_count = 0
    tp_identity_mismatch_count = 0

    total_graphs = 0
    total_extractions = 0

    for epsilon in EPSILONS:
        for seed in SEEDS:
            graph = generator.generate(
                BurstyConfig(
                    seed=int(seed),
                    epsilon=float(epsilon),
                    n_events=N_EVENTS,
                    n_actors=N_ACTORS,
                )
            )
            total_graphs += 1

            max_focal_event = _max_weight_focal_event(graph, FOCAL_ACTOR)
            tp_timestamp = float(max_focal_event.timestamp)
            n_pre_tp_all_events = sum(
                1 for event in graph.events if float(event.timestamp) < tp_timestamp
            )
            j_focal_value = _j_focal(
                graph=graph,
                focal_actor=FOCAL_ACTOR,
                tp_timestamp=tp_timestamp,
            )

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
                total_extractions += 1

                actual_tp = result.turning_point
                if actual_tp is not None and actual_tp.id == max_focal_event.id:
                    tp_identity_match_count += 1
                else:
                    tp_identity_mismatch_count += 1

                j_dev_pool_value = _j_dev_pool(
                    n_pre_tp_all_events=n_pre_tp_all_events,
                    k=k,
                )
                j_dev_output_value = _j_dev_output(result)

                j_values = {
                    "j_focal": int(j_focal_value),
                    "j_dev_pool": int(j_dev_pool_value),
                    "j_dev_output": int(j_dev_output_value),
                }

                totals[k]["total_instances"] += 1
                totals[k]["greedy_valid_count"] += int(result.valid)

                actual_failure = bool(not result.valid)
                for j_name, j_value in j_values.items():
                    pred_failure = bool(j_value < k)
                    counters = theorem_counts[k][j_name]
                    counters["n"] += 1
                    counters["correct"] += int(pred_failure == actual_failure)
                    counters["fp"] += int(pred_failure and result.valid)
                    counters["fn"] += int((not pred_failure) and (not result.valid))

                    by_k_j[k][j_name][j_value]["n"] += 1
                    by_k_j[k][j_name][j_value]["greedy_valid"] += int(result.valid)

    results_by_k: dict[str, dict] = {}

    for k in K_VALUES:
        by_j_focal = {}
        for j in sorted(by_k_j[k]["j_focal"].keys()):
            payload = by_k_j[k]["j_focal"][j]
            by_j_focal[str(j)] = {
                "n": int(payload["n"]),
                "greedy_valid": int(payload["greedy_valid"]),
                "greedy_valid_rate": _safe_rate(payload["greedy_valid"], payload["n"]),
            }

        by_j_dev_pool = {}
        for j in sorted(by_k_j[k]["j_dev_pool"].keys()):
            payload = by_k_j[k]["j_dev_pool"][j]
            by_j_dev_pool[str(j)] = {
                "n": int(payload["n"]),
                "greedy_valid": int(payload["greedy_valid"]),
                "greedy_valid_rate": _safe_rate(payload["greedy_valid"], payload["n"]),
            }

        by_j_dev_output = {}
        for j in sorted(by_k_j[k]["j_dev_output"].keys()):
            payload = by_k_j[k]["j_dev_output"][j]
            by_j_dev_output[str(j)] = {
                "n": int(payload["n"]),
                "greedy_valid": int(payload["greedy_valid"]),
                "greedy_valid_rate": _safe_rate(payload["greedy_valid"], payload["n"]),
            }

        metrics = {}
        for j_name in ("j_focal", "j_dev_output", "j_dev_pool"):
            counters = theorem_counts[k][j_name]
            metrics[j_name] = {
                "accuracy": _safe_rate(counters["correct"], counters["n"]),
                "false_positives": int(counters["fp"]),
                "false_negatives": int(counters["fn"]),
                "n_instances": int(counters["n"]),
            }

        total_instances = totals[k]["total_instances"]
        results_by_k[str(k)] = {
            "total_instances": int(total_instances),
            "greedy_valid_count": int(totals[k]["greedy_valid_count"]),
            "greedy_valid_rate": _safe_rate(totals[k]["greedy_valid_count"], total_instances),
            "theorem_metrics": metrics,
            "by_j_focal": by_j_focal,
            "by_j_dev_pool": by_j_dev_pool,
            "by_j_dev_output": by_j_dev_output,
        }

    heatmap_j_bins = [str(i) for i in range(HEATMAP_J_MAX)] + [f"{HEATMAP_J_MAX}+"]
    heatmap_j_dev_pool_matrix = [
        _heatmap_row(by_k_j[k]["j_dev_pool"]) for k in K_VALUES
    ]
    heatmap_j_focal_matrix = [
        _heatmap_row(by_k_j[k]["j_focal"]) for k in K_VALUES
    ]

    fp_counts_total = {
        "j_focal": int(
            sum(theorem_counts[k]["j_focal"]["fp"] for k in K_VALUES)
        ),
        "j_dev_output": int(
            sum(theorem_counts[k]["j_dev_output"]["fp"] for k in K_VALUES)
        ),
        "j_dev_pool": int(
            sum(theorem_counts[k]["j_dev_pool"]["fp"] for k in K_VALUES)
        ),
    }

    boundary_rows: list[dict] = []
    below_rates_for_criteria: list[float] = []
    above_rates_for_criteria: list[float] = []
    for k in K_VALUES:
        cells = by_k_j[k]["j_dev_pool"]
        below = []
        above = []
        diagonal = None
        for j, payload in cells.items():
            rate = _safe_rate(payload["greedy_valid"], payload["n"])
            if j < k:
                below.append((j, rate, payload["n"]))
                if k >= 2:
                    below_rates_for_criteria.append(rate)
            elif j == k:
                diagonal = rate
            else:
                above.append((j, rate, payload["n"]))
                if j >= k + 2:
                    above_rates_for_criteria.append(rate)

        below_weighted = (
            _safe_rate(
                sum(rate * n for _, rate, n in below),
                sum(n for _, _, n in below),
            )
            if below
            else None
        )
        below_max = max((rate for _, rate, _ in below), default=None)
        above_weighted = (
            _safe_rate(
                sum(rate * n for _, rate, n in above),
                sum(n for _, _, n in above),
            )
            if above
            else None
        )
        above_min = min((rate for _, rate, _ in above), default=None)

        boundary_rows.append(
            {
                "k": int(k),
                "below_weighted_rate": below_weighted,
                "below_max_cell_rate": below_max,
                "diagonal_rate": diagonal,
                "above_weighted_rate": above_weighted,
                "above_min_cell_rate": above_min,
            }
        )

    zero_fp_defs = [
        name for name, count in fp_counts_total.items() if int(count) == 0
    ]
    if zero_fp_defs:
        zero_fp_statement = (
            "Zero false positives achieved by: " + ", ".join(zero_fp_defs) + "."
        )
    else:
        zero_fp_statement = "No j definition achieved zero false positives in this run."

    data = {
        "parameters": {
            "k_values": K_VALUES,
            "epsilon_values": EPSILONS,
            "seeds_per_epsilon": len(SEEDS),
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "focal_actor": FOCAL_ACTOR,
            "generator": "BurstyGenerator",
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "grammar": {
                "min_prefix_elements": "k",
                "min_timespan_fraction": 0.0,
                "max_temporal_gap": "inf",
            },
            "pythonhashseed": TARGET_HASH_SEED,
            "j_definitions": ["j_focal", "j_dev_output", "j_dev_pool"],
        },
        "results_by_k": results_by_k,
        "fp_counts_total": fp_counts_total,
        "tp_identity": {
            "tp_match_count": int(tp_identity_match_count),
            "tp_mismatch_count": int(tp_identity_mismatch_count),
        },
        "j_dev_pool_boundary": {
            "by_k": boundary_rows,
        },
        "zero_fp_statement": zero_fp_statement,
        "criteria": {
            "zero_fp_dev_definition": bool(
                fp_counts_total["j_dev_pool"] == 0 or fp_counts_total["j_dev_output"] == 0
            ),
            "below_diagonal_ok_k_ge_2": bool(
                all(rate <= 0.02 + 1e-12 for rate in below_rates_for_criteria)
                if below_rates_for_criteria
                else True
            ),
            "above_diagonal_ok_j_ge_k_plus_2": bool(
                all(rate >= 0.90 - 1e-12 for rate in above_rates_for_criteria)
                if above_rates_for_criteria
                else False
            ),
            "k0_full_validity": bool(
                abs(results_by_k["0"]["greedy_valid_rate"] - 1.0) <= 1e-12
            ),
        },
        "heatmap_j_dev_pool": {
            "j_bins": heatmap_j_bins,
            "matrix": heatmap_j_dev_pool_matrix,
            "file": f"experiments/output/{HEATMAP_FINAL_NAME}",
        },
        "heatmap_j_focal": {
            "j_bins": heatmap_j_bins,
            "matrix": heatmap_j_focal_matrix,
            "file": f"experiments/output/{HEATMAP_JFOCAL_NAME}",
        },
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "k_values": K_VALUES,
            "epsilons": EPSILONS,
            "graphs": total_graphs,
            "extractions": total_extractions,
            "pythonhashseed": TARGET_HASH_SEED,
        },
    )
    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)

    output_root = Path(__file__).resolve().parent / "output"
    heatmap_final_path = output_root / HEATMAP_FINAL_NAME
    heatmap_jfocal_path = output_root / HEATMAP_JFOCAL_NAME

    _render_heatmap(
        matrix=heatmap_j_dev_pool_matrix,
        output_path=heatmap_final_path,
        title="Greedy Validity by Prefix Requirement (k) and Development-Eligible Events Before TP",
        x_label="j_dev_pool (development-eligible events before TP; 10+ binned)",
    )
    _render_heatmap(
        matrix=heatmap_j_focal_matrix,
        output_path=heatmap_jfocal_path,
        title="Greedy Validity by Prefix Requirement (k) and Focal-Only Events Before TP (j_focal)",
        x_label="j_focal (strict focal events before TP; 10+ binned)",
    )

    print("k-j Boundary Final")
    print("==================")
    print(f"Total graphs: {total_graphs}")
    print(f"Total extractions: {total_extractions}")
    print(f"FP counts: {fp_counts_total}")
    print(
        f"tp_match={tp_identity_match_count} tp_mismatch={tp_identity_mismatch_count}"
    )
    print(f"Heatmap j_dev_pool: {heatmap_final_path}")
    print(f"Heatmap j_focal: {heatmap_jfocal_path}")

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_kj_boundary_final()
