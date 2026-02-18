"""Approximation ratio analysis: greedy score quality vs oracle on valid extractions."""

from __future__ import annotations

from statistics import mean, median, pstdev

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _aggregate_group(group: dict) -> dict:
    ratios = group["ratios"]

    ratio_mean = mean(ratios) if ratios else None
    ratio_median = median(ratios) if ratios else None
    ratio_min = min(ratios) if ratios else None
    ratio_std = pstdev(ratios) if len(ratios) >= 2 else (0.0 if len(ratios) == 1 else None)

    return {
        "epsilon": group["epsilon"],
        "k": group["k"],
        "total_cases": group["total_cases"],
        "both_valid_count": group["both_valid_count"],
        "greedy_valid_oracle_invalid": group["greedy_valid_oracle_invalid"],
        "oracle_valid_greedy_invalid": group["oracle_valid_greedy_invalid"],
        "both_invalid": group["both_invalid"],
        "ratio_mean": ratio_mean,
        "ratio_median": ratio_median,
        "ratio_min": ratio_min,
        "ratio_std": ratio_std,
        "ratio_lt_0_90": group["ratio_lt_0_90"],
        "ratio_lt_0_80": group["ratio_lt_0_80"],
        "ratio_gt_1_00": group["ratio_gt_1_00"],
    }


def _print_table(summary_rows: list[dict]) -> None:
    print(
        "| eps | k | both_valid | ratio_mean | ratio_median | ratio_min | ratio_std | <0.90 | <0.80 | "
        ">1.00 | g_valid/o_invalid | o_valid/g_invalid |"
    )
    print("|-----|---|------------|------------|--------------|-----------|-----------|-------|-------|-------|------------------|------------------|")

    for row in summary_rows:
        print(
            f"| {row['epsilon']:.2f} | {row['k']} | {row['both_valid_count']:>10} | "
            f"{_fmt(row['ratio_mean']):>10} | {_fmt(row['ratio_median']):>12} | {_fmt(row['ratio_min']):>9} | "
            f"{_fmt(row['ratio_std']):>9} | {row['ratio_lt_0_90']:>5} | {row['ratio_lt_0_80']:>5} | "
            f"{row['ratio_gt_1_00']:>5} | "
            f"{row['greedy_valid_oracle_invalid']:>16} | {row['oracle_valid_greedy_invalid']:>16} |"
        )


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        "## Approximation Ratio by (epsilon, k)",
        "",
        "| epsilon | k | total_cases | both_valid | ratio_mean | ratio_median | ratio_min | ratio_std | ratio<0.90 | ratio<0.80 | ratio>1.00 | greedy_valid_oracle_invalid | oracle_valid_greedy_invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in data["summary_rows"]:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | {row['total_cases']} | {row['both_valid_count']} | "
            f"{_fmt(row['ratio_mean'])} | {_fmt(row['ratio_median'])} | {_fmt(row['ratio_min'])} | {_fmt(row['ratio_std'])} | "
            f"{row['ratio_lt_0_90']} | {row['ratio_lt_0_80']} | {row['ratio_gt_1_00']} | {row['greedy_valid_oracle_invalid']} | "
            f"{row['oracle_valid_greedy_invalid']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Approximation ratio is computed only when both greedy and oracle are valid: `greedy_score / oracle_score`.",
            "- `greedy_valid_oracle_invalid` should ideally be zero for a true oracle; non-zero values indicate oracle-construction limits.",
            "- `ratio>1.00` indicates cases where the current oracle baseline scored below greedy despite both being valid.",
            "- `oracle_valid_greedy_invalid` counts false negatives where greedy fails on solvable cases.",
            "",
        ]
    )

    return "\n".join(lines)


def run_approximation_ratio() -> dict:
    epsilons = [0.30, 0.50, 0.70, 0.90]
    k_values = [0, 1, 2, 3]
    seeds = range(1, 51)
    n_events = 200
    n_actors = 6

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    # Deterministically pre-generate graphs for each epsilon/seed.
    graphs_by_eps_seed: dict[tuple[float, int], object] = {}
    for epsilon in epsilons:
        for seed in seeds:
            graphs_by_eps_seed[(epsilon, seed)] = generator.generate(
                BurstyConfig(
                    seed=seed,
                    epsilon=epsilon,
                    n_events=n_events,
                    n_actors=n_actors,
                )
            )

    groups: dict[tuple[float, int], dict] = {}
    per_case: list[dict] = []

    for epsilon in epsilons:
        for k in k_values:
            groups[(epsilon, k)] = {
                "epsilon": epsilon,
                "k": k,
                "total_cases": 0,
                "both_valid_count": 0,
                "greedy_valid_oracle_invalid": 0,
                "oracle_valid_greedy_invalid": 0,
                "both_invalid": 0,
                "ratio_lt_0_90": 0,
                "ratio_lt_0_80": 0,
                "ratio_gt_1_00": 0,
                "ratios": [],
            }

    for epsilon in epsilons:
        for seed in seeds:
            graph = graphs_by_eps_seed[(epsilon, seed)]
            for actor in sorted(graph.actors):
                for k in k_values:
                    grammar = GrammarConfig.parametric(k)
                    group = groups[(epsilon, k)]
                    group["total_cases"] += 1

                    greedy = greedy_extract(
                        graph=graph,
                        focal_actor=actor,
                        grammar=grammar,
                        pool_strategy="injection",
                    )
                    oracle, oracle_diag = oracle_extract(
                        graph=graph,
                        focal_actor=actor,
                        grammar=grammar,
                    )

                    greedy_valid = bool(greedy.valid)
                    oracle_valid = bool(oracle is not None and oracle.valid)

                    ratio = None
                    if greedy_valid and oracle_valid:
                        group["both_valid_count"] += 1
                        oracle_score = float(oracle.score)
                        greedy_score = float(greedy.score)
                        if oracle_score > 0:
                            ratio = greedy_score / oracle_score
                            group["ratios"].append(ratio)
                            if ratio > 1.00:
                                group["ratio_gt_1_00"] += 1
                            if ratio < 0.90:
                                group["ratio_lt_0_90"] += 1
                            if ratio < 0.80:
                                group["ratio_lt_0_80"] += 1
                    elif greedy_valid and not oracle_valid:
                        group["greedy_valid_oracle_invalid"] += 1
                    elif oracle_valid and not greedy_valid:
                        group["oracle_valid_greedy_invalid"] += 1
                    else:
                        group["both_invalid"] += 1

                    per_case.append(
                        {
                            "epsilon": epsilon,
                            "k": k,
                            "seed": seed,
                            "focal_actor": actor,
                            "greedy_valid": greedy_valid,
                            "oracle_valid": oracle_valid,
                            "greedy_score": float(greedy.score),
                            "oracle_score": float(oracle.score) if oracle is not None else None,
                            "approximation_ratio": ratio,
                            "greedy_violations": list(greedy.violations),
                            "oracle_violations": list(oracle.violations) if oracle is not None else None,
                            "oracle_diagnostics": oracle_diag,
                        }
                    )

    summary_rows = [_aggregate_group(groups[(epsilon, k)]) for epsilon in epsilons for k in k_values]

    # Keep deterministic ordering in outputs.
    summary_rows.sort(key=lambda row: (row["epsilon"], row["k"]))

    data = {
        "settings": {
            "epsilons": epsilons,
            "k_values": k_values,
            "seed_range": [min(seeds), max(seeds)],
            "n_events": n_events,
            "n_actors": n_actors,
            "pool_strategy": "injection",
            "n_cases_per_group": len(seeds) * n_actors,
        },
        "summary_rows": summary_rows,
        "per_case": per_case,
    }

    total_cases = len(per_case)
    metadata = ExperimentMetadata(
        name="approximation_ratio",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(epsilons) * len(seeds),
        n_extractions=total_cases * 2,
        seed_range=(min(seeds), max(seeds)),
        parameters={
            "epsilons": epsilons,
            "k_values": k_values,
            "n_events": n_events,
            "n_actors": n_actors,
            "ratio_definition": "greedy_score / oracle_score for both-valid cases",
        },
    )

    save_results("approximation_ratio", data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    payload = run_approximation_ratio()
    _print_table(payload["results"]["summary_rows"])
