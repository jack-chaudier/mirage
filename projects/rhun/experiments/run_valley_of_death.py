"""Test non-monotonic regularization (valley of death) on synthetic graphs."""

from __future__ import annotations

from statistics import mean

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import Event


def _sequence_span(events: tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    return float(events[-1].timestamp - events[0].timestamp)


def _compute_metrics(
    graphs: list,
    grammar_builder,
) -> dict:
    valid_count = 0
    n_extractions = 0

    valid_scores: list[float] = []
    development_counts: list[int] = []
    span_values: list[float] = []
    span_fraction_values: list[float] = []

    for graph in graphs:
        for actor in sorted(graph.actors):
            grammar = grammar_builder()
            result = greedy_extract(
                graph=graph,
                focal_actor=actor,
                grammar=grammar,
                pool_strategy="injection",
            )
            n_extractions += 1
            valid_count += int(result.valid)

            if result.valid:
                valid_scores.append(float(result.score))

            development_counts.append(int(result.n_development))
            span = _sequence_span(result.events)
            span_values.append(span)
            span_fraction_values.append(span / graph.duration if graph.duration > 0 else 0.0)

    validity_rate = (valid_count / n_extractions) if n_extractions else 0.0

    return {
        "n_extractions": n_extractions,
        "valid_count": valid_count,
        "validity_rate": validity_rate,
        "mean_score_valid": mean(valid_scores) if valid_scores else None,
        "mean_development_count": mean(development_counts) if development_counts else 0.0,
        "mean_timespan": mean(span_values) if span_values else 0.0,
        "mean_timespan_fraction": mean(span_fraction_values) if span_fraction_values else 0.0,
    }


def _detect_valley(rows: list[dict], value_key: str, descending: bool) -> dict:
    ordered = sorted(rows, key=lambda row: row[value_key], reverse=descending)
    rates = [float(row["validity_rate"]) for row in ordered]
    values = [row[value_key] for row in ordered]

    local_minima: list[dict] = []
    tol = 1e-12
    for i in range(1, len(rates) - 1):
        if rates[i] + tol < rates[i - 1] and rates[i] + tol < rates[i + 1]:
            local_minima.append(
                {
                    "value": values[i],
                    "validity_rate": rates[i],
                    "left": rates[i - 1],
                    "right": rates[i + 1],
                }
            )

    monotonic_non_decreasing = all(rates[i] + tol >= rates[i - 1] for i in range(1, len(rates)))

    max_valley_depth = 0.0
    deepest_valley_value = None
    for i in range(1, len(rates) - 1):
        prev_max = max(rates[:i])
        future_max = max(rates[i + 1 :])
        if prev_max > rates[i] and future_max > rates[i]:
            depth = min(prev_max, future_max) - rates[i]
            if depth > max_valley_depth:
                max_valley_depth = depth
                deepest_valley_value = values[i]

    return {
        "ordered_values": values,
        "ordered_rates": rates,
        "has_local_minimum": bool(local_minima),
        "local_minima": local_minima,
        "monotonic_non_decreasing_under_relaxation": monotonic_non_decreasing,
        "max_valley_depth": max_valley_depth,
        "deepest_valley_value": deepest_valley_value,
    }


def _spark(rate: float, width: int = 40) -> str:
    filled = int(round(max(0.0, min(1.0, rate)) * width))
    return "#" * filled + "." * (width - filled)


def _print_curve(title: str, rows: list[dict], value_key: str, descending: bool) -> None:
    print(title)
    ordered = sorted(rows, key=lambda row: row[value_key], reverse=descending)
    for row in ordered:
        value = row[value_key]
        rate = row["validity_rate"]
        print(f"  {value:>5} | {rate:0.3f} | {_spark(rate)}")


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines = [
        f"# {meta.name}",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
    ]

    lines.append("## Sweep A: Prefix Constraint (k)")
    lines.append("")
    for eps_key, payload in data["sweep_a_prefix"].items():
        lines.append(f"### epsilon = {eps_key}")
        lines.append("")
        lines.append("| k | validity_rate | mean_score_valid | mean_development_count | mean_timespan_fraction |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sorted(payload["rows"], key=lambda r: r["k"], reverse=True):
            score = "n/a" if row["mean_score_valid"] is None else f"{row['mean_score_valid']:.3f}"
            lines.append(
                f"| {row['k']} | {row['validity_rate']:.3f} | {score} | "
                f"{row['mean_development_count']:.3f} | {row['mean_timespan_fraction']:.3f} |"
            )
        analysis = payload["valley_analysis"]
        lines.append("")
        lines.append(
            "Valley detected: "
            f"{analysis['has_local_minimum']} (max_depth={analysis['max_valley_depth']:.3f})"
        )
        lines.append("")

    lines.append("## Sweep B: Timespan Constraint")
    lines.append("")
    for eps_key, payload in data["sweep_b_timespan"].items():
        lines.append(f"### epsilon = {eps_key}")
        lines.append("")
        lines.append("| min_timespan_fraction | validity_rate | mean_score_valid | mean_development_count | mean_timespan_fraction |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sorted(payload["rows"], key=lambda r: r["min_timespan_fraction"], reverse=True):
            score = "n/a" if row["mean_score_valid"] is None else f"{row['mean_score_valid']:.3f}"
            lines.append(
                f"| {row['min_timespan_fraction']:.2f} | {row['validity_rate']:.3f} | {score} | "
                f"{row['mean_development_count']:.3f} | {row['mean_timespan_fraction']:.3f} |"
            )
        analysis = payload["valley_analysis"]
        lines.append("")
        lines.append(
            "Valley detected: "
            f"{analysis['has_local_minimum']} (max_depth={analysis['max_valley_depth']:.3f})"
        )
        lines.append("")

    return "\n".join(lines)


def run_valley_of_death() -> dict:
    epsilons = [0.50, 0.90]
    seeds = range(0, 200)
    n_events = 200
    n_actors = 6

    # Sweep A: Prefix strictness axis
    k_values = list(range(0, 11))
    fixed_timespan_fraction = 0.15

    # Sweep B: Timespan strictness axis
    timespan_values = [round(x * 0.05, 2) for x in range(10, -1, -1)]  # 0.50 -> 0.00
    fixed_k_for_timespan = 1

    timer = ExperimentTimer()
    generator = BurstyGenerator()

    graphs_by_epsilon: dict[float, list] = {}
    for eps in epsilons:
        graphs_by_epsilon[eps] = [
            generator.generate(
                BurstyConfig(
                    seed=seed,
                    epsilon=eps,
                    n_events=n_events,
                    n_actors=n_actors,
                )
            )
            for seed in seeds
        ]

    sweep_a_prefix: dict[str, dict] = {}
    sweep_b_timespan: dict[str, dict] = {}

    total_extractions = 0

    for eps in epsilons:
        eps_key = f"{eps:.2f}"
        graphs = graphs_by_epsilon[eps]

        prefix_rows: list[dict] = []
        for k in k_values:
            metrics = _compute_metrics(
                graphs,
                grammar_builder=lambda k=k: GrammarConfig(
                    min_prefix_elements=k,
                    min_timespan_fraction=fixed_timespan_fraction,
                ),
            )
            total_extractions += metrics["n_extractions"]
            prefix_rows.append({"k": k, **metrics})

        sweep_a_prefix[eps_key] = {
            "rows": prefix_rows,
            "valley_analysis": _detect_valley(prefix_rows, value_key="k", descending=True),
        }

        timespan_rows: list[dict] = []
        for min_span in timespan_values:
            metrics = _compute_metrics(
                graphs,
                grammar_builder=lambda min_span=min_span: GrammarConfig(
                    min_prefix_elements=fixed_k_for_timespan,
                    min_timespan_fraction=min_span,
                ),
            )
            total_extractions += metrics["n_extractions"]
            timespan_rows.append({"min_timespan_fraction": min_span, **metrics})

        sweep_b_timespan[eps_key] = {
            "rows": timespan_rows,
            "valley_analysis": _detect_valley(
                timespan_rows,
                value_key="min_timespan_fraction",
                descending=True,
            ),
        }

    data = {
        "settings": {
            "epsilons": epsilons,
            "seed_range": [min(seeds), max(seeds)],
            "n_events": n_events,
            "n_actors": n_actors,
            "pool_strategy": "injection",
            "sweep_a": {
                "axis": "min_prefix_elements",
                "values": k_values,
                "fixed_min_timespan_fraction": fixed_timespan_fraction,
            },
            "sweep_b": {
                "axis": "min_timespan_fraction",
                "values": timespan_values,
                "fixed_min_prefix_elements": fixed_k_for_timespan,
            },
        },
        "sweep_a_prefix": sweep_a_prefix,
        "sweep_b_timespan": sweep_b_timespan,
    }

    metadata = ExperimentMetadata(
        name="valley_of_death",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(epsilons) * len(seeds),
        n_extractions=total_extractions,
        seed_range=(min(seeds), max(seeds)),
        parameters={
            "epsilons": epsilons,
            "k_values": k_values,
            "timespan_values": timespan_values,
            "fixed_timespan_fraction": fixed_timespan_fraction,
            "fixed_k_for_timespan": fixed_k_for_timespan,
        },
    )

    save_results("valley_of_death", data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}


def _print_stdout_summary(payload: dict) -> None:
    results = payload["results"]

    print("Sweep A: Prefix constraint (k) strict->relaxed (10 -> 0)")
    for eps_key, sweep in results["sweep_a_prefix"].items():
        print(f"epsilon={eps_key}")
        _print_curve("k | validity_rate", sweep["rows"], value_key="k", descending=True)
        va = sweep["valley_analysis"]
        print(
            f"  valley_detected={va['has_local_minimum']} "
            f"max_depth={va['max_valley_depth']:.3f}"
        )

    print("\nSweep B: Timespan constraint strict->relaxed (0.50 -> 0.00)")
    for eps_key, sweep in results["sweep_b_timespan"].items():
        print(f"epsilon={eps_key}")
        _print_curve(
            "min_timespan_fraction | validity_rate",
            sweep["rows"],
            value_key="min_timespan_fraction",
            descending=True,
        )
        va = sweep["valley_analysis"]
        print(
            f"  valley_detected={va['has_local_minimum']} "
            f"max_depth={va['max_valley_depth']:.3f}"
        )


if __name__ == "__main__":
    output = run_valley_of_death()
    _print_stdout_summary(output)
