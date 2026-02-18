"""Clean k-j boundary re-run with strict j_focal and grammar-aware classifier."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from math import isnan
from pathlib import Path

import matplotlib
import numpy as np

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, Phase


matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_NAME = "kj_boundary_clean"
HEATMAP_NAME = "kj_boundary_clean_heatmap.png"

FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
SEEDS = range(100)
K_VALUES = list(range(6))  # 0..5
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


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str) -> Event:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    if not focal_events:
        raise ValueError(f"No focal events for actor={focal_actor}")

    return max(
        focal_events,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )


def _j_focal_strict(graph: CausalGraph, focal_actor: str, max_focal_event: Event) -> int:
    """Strict j_focal: count focal events with timestamp < TP timestamp."""
    return int(
        sum(
            1
            for event in graph.events
            if focal_actor in event.actors and float(event.timestamp) < float(max_focal_event.timestamp)
        )
    )


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _j_bin(j_value: int) -> str:
    return str(j_value) if j_value < HEATMAP_J_MAX else f"{HEATMAP_J_MAX}+"


def _oracle_feasible_fast(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> bool:
    """
    Fast feasibility probe for the pure-prefix regime.

    With timespan/gap disabled and focal-only candidates, feasibility reduces to
    whether there exists a turning-point candidate with enough strict-prefix and
    enough total length capacity.
    """
    focal_events = sorted(
        [event for event in graph.events if focal_actor in event.actors],
        key=lambda event: (float(event.timestamp), event.id),
    )
    if not focal_events:
        return False

    for tp in focal_events:
        n_pre_available = sum(1 for event in focal_events if event.timestamp < tp.timestamp)
        n_post_available = sum(1 for event in focal_events if event.timestamp > tp.timestamp)

        max_n_pre = min(n_pre_available, grammar.max_length - 1)
        for n_pre in range(grammar.min_prefix_elements, max_n_pre + 1):
            min_n_post = max(0, grammar.min_length - (n_pre + 1))
            max_n_post = min(n_post_available, grammar.max_length - (n_pre + 1))
            if min_n_post <= max_n_post:
                return True

    return False


def _render_heatmap(
    matrix: list[list[float | None]],
    output_path: Path,
) -> None:
    values = np.array(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in matrix
        ],
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

    ax.set_xlabel("j_focal (strict temporal rank; 10+ binned)")
    ax.set_ylabel("k (min_prefix_elements)")
    ax.set_title("Greedy Validity Rate by Prefix Requirement (k) and TP Position (j_focal)")

    # Theorem boundary (j_focal = k) for k in [0,5].
    ax.plot([0, 5], [0, 5], linestyle="--", linewidth=1.6, color="black")

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if isnan(value):
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
    matrix = data["heatmap"]["matrix"]
    j_labels = data["heatmap"]["j_bins"]

    lines: list[str] = [
        "# k-j boundary clean re-run",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Heatmap figure: `{data['heatmap']['file']}`",
        "",
        "## Summary Table",
        "",
        "| k | total | valid_rate | theorem_accuracy | FP | FN |",
        "|---|------:|-----------:|-----------------:|---:|---:|",
    ]
    for k in K_VALUES:
        row = results_by_k[str(k)]
        lines.append(
            f"| {k} | {row['total_instances']} | {row['greedy_valid_rate']:.3f} | "
            f"{row['theorem_accuracy']:.3f} | {row['false_positives']} | {row['false_negatives']} |"
        )

    lines.extend(
        [
            "",
            "## Heatmap Matrix (Greedy Validity Rate)",
            "",
            "| k | " + " | ".join(j_labels) + " |",
            "|---|" + "|".join(["---:"] * len(j_labels)) + "|",
        ]
    )
    for k, row in zip(K_VALUES, matrix, strict=True):
        rendered = []
        for value in row:
            rendered.append("n/a" if value is None else f"{value:.3f}")
        lines.append(f"| {k} | " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "## Boundary Regions",
            "",
            "### Below Diagonal (j_focal < k)",
            "",
            "| k | weighted_rate | max_cell_rate | cells |",
            "|---|--------------:|--------------:|------:|",
        ]
    )
    for row in data["boundary"]["below_by_k"]:
        lines.append(
            f"| {row['k']} | {_format_rate(row['weighted_rate'])} | "
            f"{_format_rate(row['max_cell_rate'])} | {row['n_cells']} |"
        )

    lines.extend(
        [
            "",
            "### Diagonal (j_focal = k)",
            "",
            "| k | n | validity_rate |",
            "|---|--:|--------------:|",
        ]
    )
    for row in data["boundary"]["diagonal"]:
        lines.append(f"| {row['k']} | {row['n']} | {_format_rate(row['validity_rate'])} |")

    lines.extend(
        [
            "",
            "### Above Diagonal (j_focal > k)",
            "",
            "| k | weighted_rate | min_cell_rate | cells |",
            "|---|--------------:|--------------:|------:|",
        ]
    )
    for row in data["boundary"]["above_by_k"]:
        lines.append(
            f"| {row['k']} | {_format_rate(row['weighted_rate'])} | "
            f"{_format_rate(row['min_cell_rate'])} | {row['n_cells']} |"
        )

    lines.extend(
        [
            "",
            "## Explicit Counts",
            "",
            f"- FP count under j_focal (`j_focal < k` but greedy valid): {data['false_positives_total']}",
            (
                "- FP timestamp-tie count "
                "(`j_focal=0` with equal-timestamp prefix before extraction TP): "
                f"{data['false_positives_timestamp_tie']}"
            ),
            (
                "- Grammar-aware classifier check failures "
                "(k>=1, TP has prefix, but no DEVELOPMENT in prefix): "
                f"{data['classifier_verification']['total_failures']}"
            ),
            "",
            "## Figure Criteria Checks",
            "",
            f"- Below diagonal (k>=2) all cells <= 5%: {data['criteria']['below_diagonal_ok_k_ge_2']}",
            (
                "- Above diagonal (j_focal>k and j_focal>=2) all cells >= 90%: "
                f"{data['criteria']['above_diagonal_ok_j_gt_k_j_ge_2']}"
            ),
            f"- FP count <= 8: {data['criteria']['fp_count_ok_le_8']}",
            f"- k=0 overall validity = 100%: {data['criteria']['k0_full_validity']}",
            (
                "- Oracle feasibility near 100% (all k weighted): "
                f"{data['criteria']['oracle_feasible_weighted_rate']:.3f}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_kj_boundary_clean() -> dict:
    _ensure_hash_seed()
    timer = ExperimentTimer()
    generator = BurstyGenerator()

    by_k_j: dict[int, dict[int, dict[str, int]]] = {
        k: defaultdict(lambda: {"n": 0, "greedy_valid": 0}) for k in K_VALUES
    }

    totals: dict[int, dict[str, int]] = {
        k: {
            "total_instances": 0,
            "greedy_valid_count": 0,
            "oracle_feasible_count": 0,
            "theorem_correct": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        for k in K_VALUES
    }

    fp_timestamp_ties = 0
    fp_total = 0

    classifier_checks: dict[int, dict[str, int]] = {
        k: {"checked": 0, "ok": 0, "failures": 0} for k in K_VALUES if k >= 1
    }

    fast_oracle_fallback_calls = 0
    fast_oracle_fallback_disagreements = 0

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
            j_focal = _j_focal_strict(graph, FOCAL_ACTOR, max_focal_event)

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

                oracle_fast = _oracle_feasible_fast(graph, FOCAL_ACTOR, grammar)
                oracle_feasible = oracle_fast
                if not oracle_fast:
                    fast_oracle_fallback_calls += 1
                    oracle_exact, _ = exact_oracle_extract(graph, FOCAL_ACTOR, grammar)
                    oracle_feasible = bool(oracle_exact.valid)
                    if oracle_feasible != oracle_fast:
                        fast_oracle_fallback_disagreements += 1

                predicted_failure = bool(j_focal < k)
                actual_failure = bool(not result.valid)
                theorem_correct = bool(predicted_failure == actual_failure)

                bucket = totals[k]
                bucket["total_instances"] += 1
                bucket["greedy_valid_count"] += int(result.valid)
                bucket["oracle_feasible_count"] += int(oracle_feasible)
                bucket["theorem_correct"] += int(theorem_correct)

                if predicted_failure and result.valid:
                    bucket["false_positives"] += 1
                    fp_total += 1

                    if j_focal == 0:
                        tp_idx = next(
                            (idx for idx, phase in enumerate(result.phases) if phase == Phase.TURNING_POINT),
                            None,
                        )
                        if tp_idx is not None:
                            tp_event = result.events[tp_idx]
                            equal_ts_prefix = any(
                                abs(float(event.timestamp) - float(tp_event.timestamp)) <= 1e-12
                                for event in result.events[:tp_idx]
                            )
                            if equal_ts_prefix:
                                fp_timestamp_ties += 1

                if (not predicted_failure) and (not result.valid):
                    bucket["false_negatives"] += 1

                by_k_j[k][j_focal]["n"] += 1
                by_k_j[k][j_focal]["greedy_valid"] += int(result.valid)

                if k >= 1:
                    tp_idx = next(
                        (idx for idx, phase in enumerate(result.phases) if phase == Phase.TURNING_POINT),
                        None,
                    )
                    if tp_idx is not None and tp_idx > 0:
                        classifier_checks[k]["checked"] += 1
                        prefix_development = sum(
                            1 for phase in result.phases[:tp_idx] if phase == Phase.DEVELOPMENT
                        )
                        if prefix_development >= 1:
                            classifier_checks[k]["ok"] += 1
                        else:
                            classifier_checks[k]["failures"] += 1

    results_by_k: dict[str, dict] = {}
    heatmap_bins = [str(i) for i in range(HEATMAP_J_MAX)] + [f"{HEATMAP_J_MAX}+"]
    heatmap_matrix: list[list[float | None]] = []

    below_by_k: list[dict] = []
    above_by_k: list[dict] = []
    diagonal: list[dict] = []

    below_cells_k_ge_2_rates: list[float] = []
    above_cells_rates_j_gt_k_j_ge_2: list[float] = []

    for k in K_VALUES:
        totals_k = totals[k]
        total_instances = totals_k["total_instances"]
        by_j_focal = {}
        sorted_js = sorted(by_k_j[k].keys())

        row_rates: list[float | None] = []
        for bin_label in heatmap_bins:
            if bin_label.endswith("+"):
                lo = HEATMAP_J_MAX
                bin_rows = [by_k_j[k][j] for j in sorted_js if j >= lo]
            else:
                target = int(bin_label)
                bin_rows = [by_k_j[k][target]] if target in by_k_j[k] else []

            n = sum(row["n"] for row in bin_rows)
            valid = sum(row["greedy_valid"] for row in bin_rows)
            row_rates.append(None if n == 0 else _safe_rate(valid, n))
        heatmap_matrix.append(row_rates)

        below_cells = []
        above_cells = []
        diag_cell = None
        for j in sorted_js:
            n = int(by_k_j[k][j]["n"])
            valid = int(by_k_j[k][j]["greedy_valid"])
            rate = _safe_rate(valid, n)
            by_j_focal[str(j)] = {
                "n": n,
                "greedy_valid": valid,
                "greedy_valid_rate": rate,
            }

            cell = {"j": int(j), "n": n, "rate": rate}
            if j < k:
                below_cells.append(cell)
                if k >= 2:
                    below_cells_k_ge_2_rates.append(rate)
            elif j == k:
                diag_cell = cell
            else:
                above_cells.append(cell)
                if j >= 2:
                    above_cells_rates_j_gt_k_j_ge_2.append(rate)

        if diag_cell is None:
            diag_cell = {"j": int(k), "n": 0, "rate": None}
        diagonal.append({"k": int(k), "n": int(diag_cell["n"]), "validity_rate": diag_cell["rate"]})

        below_weighted = (
            _safe_rate(
                sum(cell["rate"] * cell["n"] for cell in below_cells),
                sum(cell["n"] for cell in below_cells),
            )
            if below_cells
            else None
        )
        below_max = max((cell["rate"] for cell in below_cells), default=None)
        below_by_k.append(
            {
                "k": int(k),
                "weighted_rate": below_weighted,
                "max_cell_rate": below_max,
                "n_cells": len(below_cells),
                "cells": below_cells,
            }
        )

        above_weighted = (
            _safe_rate(
                sum(cell["rate"] * cell["n"] for cell in above_cells),
                sum(cell["n"] for cell in above_cells),
            )
            if above_cells
            else None
        )
        above_min = min((cell["rate"] for cell in above_cells), default=None)
        above_by_k.append(
            {
                "k": int(k),
                "weighted_rate": above_weighted,
                "min_cell_rate": above_min,
                "n_cells": len(above_cells),
                "cells": above_cells,
            }
        )

        results_by_k[str(k)] = {
            "total_instances": int(total_instances),
            "greedy_valid_count": int(totals_k["greedy_valid_count"]),
            "greedy_valid_rate": _safe_rate(totals_k["greedy_valid_count"], total_instances),
            "oracle_feasible_count": int(totals_k["oracle_feasible_count"]),
            "theorem_accuracy": _safe_rate(totals_k["theorem_correct"], total_instances),
            "false_positives": int(totals_k["false_positives"]),
            "false_negatives": int(totals_k["false_negatives"]),
            "by_j_focal": by_j_focal,
        }

    criteria = {
        "below_diagonal_ok_k_ge_2": bool(
            all(rate <= 0.05 + 1e-12 for rate in below_cells_k_ge_2_rates)
            if below_cells_k_ge_2_rates
            else True
        ),
        "above_diagonal_ok_j_gt_k_j_ge_2": bool(
            all(rate >= 0.90 - 1e-12 for rate in above_cells_rates_j_gt_k_j_ge_2)
            if above_cells_rates_j_gt_k_j_ge_2
            else False
        ),
        "fp_count_ok_le_8": bool(fp_total <= 8),
        "k0_full_validity": bool(abs(results_by_k["0"]["greedy_valid_rate"] - 1.0) <= 1e-12),
        "oracle_feasible_weighted_rate": _safe_rate(
            sum(totals[k]["oracle_feasible_count"] for k in K_VALUES),
            sum(totals[k]["total_instances"] for k in K_VALUES),
        ),
    }

    data = {
        "parameters": {
            "k_values": K_VALUES,
            "epsilon_values": EPSILONS,
            "seeds_per_epsilon": len(SEEDS),
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "generator": "BurstyGenerator",
            "focal_actor": FOCAL_ACTOR,
            "grammar_template": {
                "min_prefix_elements": "k",
                "min_timespan_fraction": 0.0,
                "max_temporal_gap": "inf",
            },
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "j_definition": "j_focal_strict_timestamp",
            "strict_precedence_rule": "timestamp < tp_timestamp",
            "continuous_constraints_disabled": True,
            "classifier_mode": "grammar_aware_min_development_preservation_active",
            "pythonhashseed": TARGET_HASH_SEED,
        },
        "results_by_k": results_by_k,
        "boundary": {
            "below_by_k": below_by_k,
            "diagonal": diagonal,
            "above_by_k": above_by_k,
        },
        "false_positives_total": int(fp_total),
        "false_positives_timestamp_tie": int(fp_timestamp_ties),
        "classifier_verification": {
            "by_k": {str(k): checks for k, checks in classifier_checks.items()},
            "total_checked": int(sum(checks["checked"] for checks in classifier_checks.values())),
            "total_failures": int(sum(checks["failures"] for checks in classifier_checks.values())),
            "all_checks_passed": bool(all(checks["failures"] == 0 for checks in classifier_checks.values())),
        },
        "oracle_feasibility_probe": {
            "method": "fast_constructive_prefix_probe_with_exact_fallback_on_false",
            "fallback_calls": int(fast_oracle_fallback_calls),
            "fallback_disagreements": int(fast_oracle_fallback_disagreements),
        },
        "criteria": criteria,
        "heatmap": {
            "j_bins": heatmap_bins,
            "matrix": heatmap_matrix,
            "file": f"experiments/output/{HEATMAP_NAME}",
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
    heatmap_path = output_root / HEATMAP_NAME
    _render_heatmap(heatmap_matrix, heatmap_path)

    print("k-j Boundary Clean Re-run")
    print("=========================")
    print(f"Total graphs: {total_graphs}")
    print(f"Total extractions: {total_extractions}")
    print(f"FP total (j_focal): {fp_total}")
    print(f"FP timestamp ties: {fp_timestamp_ties}")
    print(f"k=0 validity: {results_by_k['0']['greedy_valid_rate']:.3f}")
    print(f"Classifier verification failures: {data['classifier_verification']['total_failures']}")
    print(f"Heatmap: {heatmap_path}")

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_kj_boundary_clean()
