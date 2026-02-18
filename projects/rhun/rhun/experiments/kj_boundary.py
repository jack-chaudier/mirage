"""Joint sweep of k (prefix requirement) and j (max-weight temporal index)."""

from __future__ import annotations

from collections import defaultdict

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.experiments.runner import (
    ExperimentMetadata,
    ExperimentTimer,
    save_results,
    utc_timestamp,
)
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.theory.theorem import check_precondition


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        "| k | j | n | success_rate |",
        "|---|---|---|--------------|",
    ]
    for row in data["rows"]:
        lines.append(
            f"| {row['k']} | {row['j']} | {row['n']} | {row['success_rate']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_kj_boundary(
    seeds: range | None = None,
    k_values: range | None = None,
    epsilon: float = 0.5,
) -> dict:
    seed_values = seeds or range(1, 51)
    k_range = k_values or range(0, 6)

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    bucket: dict[tuple[int, int], list[int]] = defaultdict(list)
    total_extractions = 0
    total_graphs = 0

    for seed in seed_values:
        graph = generator.generate(BurstyConfig(seed=seed, epsilon=epsilon))
        total_graphs += 1

        for actor in sorted(graph.actors):
            for k in k_range:
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

    data = {
        "rows": rows,
        "epsilon": epsilon,
    }
    metadata = ExperimentMetadata(
        name="kj_boundary",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions,
        seed_range=(min(seed_values), max(seed_values)),
        parameters={"k_values": list(k_range), "epsilon": epsilon},
    )
    save_results("kj_boundary", data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}
