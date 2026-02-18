"""Sweep front-loading parameter epsilon and measure greedy extraction failure."""

from __future__ import annotations

from statistics import mean

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.theory.theorem import check_precondition, verify_prediction


def _format_table(rows: list[dict]) -> str:
    lines = [
        "| epsilon | n_graphs | validity_rate | mean_score | theorem_accuracy | absorption_rate | mean_max_w_position |",
        "|---------|----------|---------------|------------|------------------|-----------------|---------------------|",
    ]
    for row in rows:
        lines.append(
            "| {epsilon:.2f} | {n_graphs} | {validity_rate:.3f} | {mean_score:.3f} | "
            "{theorem_accuracy:.3f} | {absorption_rate:.3f} | {mean_max_w_position:.3f} |".format(
                **row
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


def run_position_sweep(
    epsilons: list[float] | None = None,
    seeds: range | None = None,
) -> dict:
    eps_values = epsilons or [round(i * 0.05, 2) for i in range(21)]
    seed_values = seeds or range(1, 51)

    grammar = GrammarConfig.strict()
    generator = BurstyGenerator()
    timer = ExperimentTimer()

    rows: list[dict] = []
    total_graphs = 0
    total_extractions = 0

    for epsilon in eps_values:
        valid_count = 0
        prediction_correct_count = 0
        absorption_count = 0
        max_positions: list[float] = []
        scores: list[float] = []
        extraction_count = 0

        for seed in seed_values:
            graph = generator.generate(BurstyConfig(seed=seed, epsilon=epsilon))
            total_graphs += 1

            for actor in sorted(graph.actors):
                extraction = greedy_extract(
                    graph=graph,
                    focal_actor=actor,
                    grammar=grammar,
                    pool_strategy="injection",
                )
                precondition = check_precondition(graph, actor, grammar)
                comparison = verify_prediction(graph, actor, grammar, extraction)

                extraction_count += 1
                total_extractions += 1
                valid_count += int(extraction.valid)
                prediction_correct_count += int(comparison["prediction_correct"])
                absorption_count += int(bool(extraction.metadata.get("absorbed", False)))
                max_positions.append(float(precondition["max_weight_position"]))
                scores.append(float(extraction.score))

        rows.append(
            {
                "epsilon": epsilon,
                "n_graphs": len(seed_values),
                "validity_rate": (valid_count / extraction_count) if extraction_count else 0.0,
                "mean_score": mean(scores) if scores else 0.0,
                "theorem_accuracy": (
                    prediction_correct_count / extraction_count if extraction_count else 0.0
                ),
                "absorption_rate": (absorption_count / extraction_count) if extraction_count else 0.0,
                "mean_max_w_position": mean(max_positions) if max_positions else 0.0,
            }
        )

    data = {
        "rows": rows,
        "table": _format_table(rows),
    }
    metadata = ExperimentMetadata(
        name="position_sweep",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions,
        seed_range=(min(seed_values), max(seed_values)),
        parameters={
            "epsilons": eps_values,
            "grammar": {"min_prefix_elements": grammar.min_prefix_elements},
            "pool_strategy": "injection",
        },
    )
    save_results("position_sweep", data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}
