"""Invariance suite for k=2 theorem behavior across graph parameter variations."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.schemas import CausalGraph, Event


matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_NAME = "invariance_suite"
ACTOR_PLOT_NAME = "invariance_suite_actor_plot.png"

FOCAL_ACTOR = "actor_0"
K_VALUE = 2
EPSILONS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = range(50)

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

TARGET_HASH_SEED = "1"


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == TARGET_HASH_SEED:
        return

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = TARGET_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _make_grammar() -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=K_VALUE,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )


def _safe_rate(numer: int | float, denom: int | float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str) -> Event:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    if not focal_events:
        raise ValueError(f"No focal events for actor={focal_actor}")
    return max(
        focal_events,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )


def _j_dev_pool(
    graph: CausalGraph,
    tp_timestamp: float,
    k: int,
) -> int:
    n_pre = sum(1 for event in graph.events if float(event.timestamp) < float(tp_timestamp))
    n_setup = int(np.ceil(0.2 * n_pre)) if n_pre > 0 else 0
    if k > 0:
        max_setup_for_development = max(0, n_pre - k)
        n_setup = min(n_setup, max_setup_for_development)
    return int(n_pre - n_setup)


def _mix_seed(base_seed: int, epsilon: float, salt: int) -> int:
    eps_key = int(round(epsilon * 100))
    return int((base_seed * 1_000_003 + eps_key * 9_176 + salt * 37) % (2**32 - 1))


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _sample_weight_distribution(
    distribution: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(0.0, 1.0, size=n)
    if distribution == "power_law":
        raw = rng.pareto(2.0, size=n) + 1.0
        return _normalize_0_1(raw)
    if distribution == "bimodal":
        choose_left = rng.random(size=n) < 0.5
        samples = np.empty(n, dtype=float)
        left_n = int(np.sum(choose_left))
        right_n = n - left_n
        if left_n > 0:
            samples[choose_left] = rng.beta(2.0, 5.0, size=left_n)
        if right_n > 0:
            samples[~choose_left] = rng.beta(5.0, 2.0, size=right_n)
        return samples
    raise ValueError(f"Unknown distribution: {distribution}")


def _replace_weights(
    graph: CausalGraph,
    distribution: str,
    epsilon: float,
    seed: int,
) -> CausalGraph:
    rng = np.random.default_rng(_mix_seed(seed, epsilon, salt=17))
    samples = _sample_weight_distribution(distribution, len(graph.events), rng)

    new_events = []
    for event, weight in zip(graph.events, samples, strict=True):
        new_events.append(
            replace(
                event,
                weight=float(weight),
                metadata={
                    **event.metadata,
                    "weight_override_distribution": distribution,
                },
            )
        )

    return replace(
        graph,
        events=tuple(new_events),
        metadata={
            **graph.metadata,
            "weight_override_distribution": distribution,
        },
    )


def _evaluate_configuration(
    *,
    sweep_name: str,
    parameter_name: str,
    parameter_value: str | int,
    graph_builder,
    grammar: GrammarConfig,
) -> dict:
    n_instances = 0
    valid_count = 0
    theorem_correct = 0
    fp_count = 0
    fn_count = 0
    j_dev_sum = 0.0
    predicted_failure_count = 0

    fp_examples: list[dict] = []
    by_epsilon: dict[str, dict[str, float | int]] = {}

    for epsilon in EPSILONS:
        eps_n = 0
        eps_valid = 0
        eps_correct = 0
        eps_fp = 0
        eps_fn = 0
        eps_j_sum = 0.0
        eps_pred_fail = 0

        for seed in SEEDS:
            graph = graph_builder(epsilon=epsilon, seed=seed)

            max_focal_event = _max_weight_focal_event(graph, FOCAL_ACTOR)
            j_dev = _j_dev_pool(graph, float(max_focal_event.timestamp), K_VALUE)

            result = greedy_extract(
                graph=graph,
                focal_actor=FOCAL_ACTOR,
                grammar=grammar,
                pool_strategy=POOL_STRATEGY,
                n_anchors=N_ANCHORS,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                injection_top_n=INJECTION_TOP_N,
            )

            pred_failure = bool(j_dev < K_VALUE)
            actual_failure = bool(not result.valid)

            n_instances += 1
            valid_count += int(result.valid)
            theorem_correct += int(pred_failure == actual_failure)
            fp_count += int(pred_failure and result.valid)
            fn_count += int((not pred_failure) and actual_failure)
            j_dev_sum += float(j_dev)
            predicted_failure_count += int(pred_failure)

            eps_n += 1
            eps_valid += int(result.valid)
            eps_correct += int(pred_failure == actual_failure)
            eps_fp += int(pred_failure and result.valid)
            eps_fn += int((not pred_failure) and actual_failure)
            eps_j_sum += float(j_dev)
            eps_pred_fail += int(pred_failure)

            if pred_failure and result.valid and len(fp_examples) < 10:
                fp_examples.append(
                    {
                        "epsilon": float(epsilon),
                        "seed": int(seed),
                        "j_dev_pool": int(j_dev),
                        "max_weight_focal_event_id": max_focal_event.id,
                        "max_weight_focal_event_timestamp": float(max_focal_event.timestamp),
                        "n_development": int(result.n_development),
                        "tp_event_id": None if result.turning_point is None else result.turning_point.id,
                    }
                )

        by_epsilon[f"{epsilon:.1f}"] = {
            "n_instances": int(eps_n),
            "validity_rate": _safe_rate(eps_valid, eps_n),
            "theorem_accuracy": _safe_rate(eps_correct, eps_n),
            "fp_count": int(eps_fp),
            "fn_count": int(eps_fn),
            "mean_j_dev_pool": _safe_rate(eps_j_sum, eps_n),
            "predicted_failure_rate": _safe_rate(eps_pred_fail, eps_n),
        }

    return {
        "sweep": sweep_name,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "n_instances": int(n_instances),
        "valid_count": int(valid_count),
        "validity_rate": _safe_rate(valid_count, n_instances),
        "theorem_accuracy": _safe_rate(theorem_correct, n_instances),
        "fp_count": int(fp_count),
        "fn_count": int(fn_count),
        "mean_j_dev_pool": _safe_rate(j_dev_sum, n_instances),
        "predicted_failure_count": int(predicted_failure_count),
        "predicted_failure_rate": _safe_rate(predicted_failure_count, n_instances),
        "by_epsilon": by_epsilon,
        "fp_examples": fp_examples,
    }


def _render_actor_plot(
    actor_configs: list[dict],
    output_path: Path,
) -> None:
    x = [int(row["parameter_value"]) for row in actor_configs]
    y_j = [float(row["mean_j_dev_pool"]) for row in actor_configs]
    y_fp = [_safe_rate(int(row["fp_count"]), int(row["n_instances"])) for row in actor_configs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, y_j, marker="o", color="#1f77b4", linewidth=2)
    ax1.set_xlabel("n_actors")
    ax1.set_ylabel("mean j_dev_pool", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, axis="both", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, y_fp, marker="s", color="#d62728", linewidth=2, linestyle="--")
    ax2.set_ylabel("theorem FP rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Invariance A: Actor Count vs mean j_dev_pool and FP rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _table_lines(configs: list[dict]) -> list[str]:
    lines = [
        "| parameter_value | validity_rate | theorem_accuracy | FP | FN | mean_j_dev_pool |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in configs:
        lines.append(
            f"| {row['parameter_value']} | {row['validity_rate']:.3f} | "
            f"{row['theorem_accuracy']:.3f} | {row['fp_count']} | {row['fn_count']} | "
            f"{row['mean_j_dev_pool']:.3f} |"
        )
    return lines


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines: list[str] = [
        "# invariance suite (k=2, j_dev_pool)",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Actor plot: `experiments/output/{ACTOR_PLOT_NAME}`",
        "",
        "## A. Actor Count Sweep",
        "",
    ]
    lines.extend(_table_lines(data["sweeps"]["actor_count"]["configs"]))

    lines.extend(["", "## B. Event Density Sweep", ""])
    lines.extend(_table_lines(data["sweeps"]["event_density"]["configs"]))

    lines.extend(["", "## C. Weight Distribution Sweep", ""])
    lines.extend(_table_lines(data["sweeps"]["weight_distribution"]["configs"]))

    lines.extend(["", "## D. Multi-burst Topology", ""])
    lines.extend(_table_lines(data["sweeps"]["multiburst_topology"]["configs"]))

    lines.extend(["", "## Key Question", ""])
    lines.append(
        f"- FP=0 across all configurations at k=2: {data['overall']['all_configs_fp_zero']}"
    )

    if data["overall"]["all_configs_fp_zero"]:
        lines.append("- No violating configuration found.")
    else:
        lines.append("- Configurations with FP > 0:")
        for issue in data["overall"]["fp_violations"]:
            lines.append(
                f"  - {issue['sweep']}::{issue['parameter_value']} FP={issue['fp_count']} "
                f"(rate={issue['fp_rate']:.3f})"
            )
            if issue["fp_examples"]:
                first = issue["fp_examples"][0]
                lines.append(
                    f"    first_example: eps={first['epsilon']:.1f}, seed={first['seed']}, "
                    f"j_dev={first['j_dev_pool']}, tp={first['tp_event_id']}"
                )

    lines.append("")
    return "\n".join(lines)


def run_invariance_suite() -> dict:
    _ensure_hash_seed()
    timer = ExperimentTimer()
    grammar = _make_grammar()

    sweeps: dict[str, dict] = {}
    total_graphs = 0
    total_extractions = 0

    # A: actor count sweep
    actor_values = [2, 4, 6, 10, 20, 50]
    actor_configs = []
    for n_actors in actor_values:
        def _build_actor_graph(*, epsilon: float, seed: int, n_actors_local=n_actors):
            return BurstyGenerator().generate(
                BurstyConfig(
                    seed=int(seed),
                    epsilon=float(epsilon),
                    n_events=200,
                    n_actors=int(n_actors_local),
                )
            )

        result = _evaluate_configuration(
            sweep_name="actor_count",
            parameter_name="n_actors",
            parameter_value=int(n_actors),
            graph_builder=_build_actor_graph,
            grammar=grammar,
        )
        actor_configs.append(result)
        total_graphs += result["n_instances"]
        total_extractions += result["n_instances"]
    sweeps["actor_count"] = {"configs": actor_configs}

    # B: event density sweep
    density_values = [50, 100, 200, 500, 1000]
    density_configs = []
    for n_events in density_values:
        def _build_density_graph(*, epsilon: float, seed: int, n_events_local=n_events):
            return BurstyGenerator().generate(
                BurstyConfig(
                    seed=int(seed),
                    epsilon=float(epsilon),
                    n_events=int(n_events_local),
                    n_actors=6,
                )
            )

        result = _evaluate_configuration(
            sweep_name="event_density",
            parameter_name="n_events",
            parameter_value=int(n_events),
            graph_builder=_build_density_graph,
            grammar=grammar,
        )
        density_configs.append(result)
        total_graphs += result["n_instances"]
        total_extractions += result["n_instances"]
    sweeps["event_density"] = {"configs": density_configs}

    # C: weight distribution sweep
    distribution_values = ["uniform", "power_law", "bimodal"]
    distribution_configs = []
    for distribution in distribution_values:
        def _build_dist_graph(*, epsilon: float, seed: int, distribution_local=distribution):
            base_graph = BurstyGenerator().generate(
                BurstyConfig(
                    seed=int(seed),
                    epsilon=float(epsilon),
                    n_events=200,
                    n_actors=6,
                )
            )
            return _replace_weights(
                graph=base_graph,
                distribution=str(distribution_local),
                epsilon=float(epsilon),
                seed=int(seed),
            )

        result = _evaluate_configuration(
            sweep_name="weight_distribution",
            parameter_name="distribution",
            parameter_value=str(distribution),
            graph_builder=_build_dist_graph,
            grammar=grammar,
        )
        distribution_configs.append(result)
        total_graphs += result["n_instances"]
        total_extractions += result["n_instances"]
    sweeps["weight_distribution"] = {"configs": distribution_configs}

    # D: multi-burst topology (defaults, epsilon loop retained as condition buckets)
    def _build_multiburst_graph(*, epsilon: float, seed: int):
        combo_seed = _mix_seed(int(seed), float(epsilon), salt=99)
        return MultiBurstGenerator().generate(
            MultiBurstConfig(
                seed=int(combo_seed),
                n_events=200,
                n_actors=6,
            )
        )

    multiburst_result = _evaluate_configuration(
        sweep_name="multiburst_topology",
        parameter_name="generator",
        parameter_value="MultiBurstGenerator(defaults)",
        graph_builder=_build_multiburst_graph,
        grammar=grammar,
    )
    sweeps["multiburst_topology"] = {"configs": [multiburst_result]}
    total_graphs += multiburst_result["n_instances"]
    total_extractions += multiburst_result["n_instances"]

    all_configs = (
        sweeps["actor_count"]["configs"]
        + sweeps["event_density"]["configs"]
        + sweeps["weight_distribution"]["configs"]
        + sweeps["multiburst_topology"]["configs"]
    )
    fp_violations = [
        {
            "sweep": row["sweep"],
            "parameter_value": row["parameter_value"],
            "fp_count": int(row["fp_count"]),
            "fp_rate": _safe_rate(int(row["fp_count"]), int(row["n_instances"])),
            "fp_examples": row["fp_examples"],
        }
        for row in all_configs
        if int(row["fp_count"]) > 0
    ]

    output_root = Path(__file__).resolve().parent / "output"
    actor_plot_path = output_root / ACTOR_PLOT_NAME
    _render_actor_plot(actor_configs, actor_plot_path)

    data = {
        "parameters": {
            "k": K_VALUE,
            "epsilons": EPSILONS,
            "seeds_per_epsilon": len(SEEDS),
            "seed_range": [min(SEEDS), max(SEEDS)],
            "grammar": {
                "min_prefix_elements": K_VALUE,
                "min_timespan_fraction": 0.0,
                "max_temporal_gap": "inf",
            },
            "focal_actor": FOCAL_ACTOR,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "pythonhashseed": TARGET_HASH_SEED,
        },
        "sweeps": sweeps,
        "overall": {
            "n_configurations": int(len(all_configs)),
            "all_configs_fp_zero": bool(len(fp_violations) == 0),
            "fp_violations": fp_violations,
        },
        "artifacts": {
            "actor_plot_file": f"experiments/output/{ACTOR_PLOT_NAME}",
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
            "k": K_VALUE,
            "epsilons": EPSILONS,
            "sweeps": ["A_actor_count", "B_event_density", "C_weight_distribution", "D_multiburst"],
            "pythonhashseed": TARGET_HASH_SEED,
        },
    )
    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)

    print("Invariance suite complete")
    print("=========================")
    print(f"Configurations: {len(all_configs)}")
    print(f"FP-violating configs: {len(fp_violations)}")
    print(f"Actor plot: {actor_plot_path}")

    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_invariance_suite()
