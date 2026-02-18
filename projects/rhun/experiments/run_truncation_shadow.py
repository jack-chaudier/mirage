"""Experiment 42: truncation-shadow diagnostics for prefix growth."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event
from rhun.theory.theorem import check_precondition, diagnose_absorption


FOCAL_ACTOR = "actor_0"
EPSILONS = [0.3, 0.5, 0.7, 0.8, 0.9]
SEEDS = range(50)
PREFIX_LENGTHS = [50, 100, 150, 200, 300, 500, 750, 1000]
N_EVENTS = 1000
N_ACTORS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "truncation_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "truncation_summary.md"


def _grammar_k1() -> GrammarConfig:
    return GrammarConfig(min_prefix_elements=1)


def _prefix_subgraph(graph: CausalGraph, prefix_length: int) -> CausalGraph:
    prefix_events = tuple(graph.events[:prefix_length])
    kept_ids = {event.id for event in prefix_events}

    rebuilt_events = []
    for event in prefix_events:
        rebuilt_events.append(
            Event(
                id=event.id,
                timestamp=float(event.timestamp),
                weight=float(event.weight),
                actors=event.actors,
                causal_parents=tuple(parent for parent in event.causal_parents if parent in kept_ids),
                metadata=dict(event.metadata),
            )
        )

    subgraph_metadata = dict(graph.metadata)
    subgraph_metadata["prefix_length"] = int(prefix_length)
    return CausalGraph(
        events=tuple(rebuilt_events),
        actors=graph.actors,
        seed=graph.seed,
        metadata=subgraph_metadata,
    )


def _run_instance(generator: BurstyGenerator, epsilon: float, seed: int) -> list[dict]:
    graph = generator.generate(
        BurstyConfig(
            n_events=N_EVENTS,
            n_actors=N_ACTORS,
            seed=seed,
            epsilon=epsilon,
        )
    )
    grammar = _grammar_k1()

    rows: list[dict] = []
    for prefix_length in PREFIX_LENGTHS:
        subgraph = _prefix_subgraph(graph, prefix_length=prefix_length)
        extraction = greedy_extract(
            graph=subgraph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
        )
        prediction = check_precondition(subgraph, FOCAL_ACTOR, grammar)
        absorption = diagnose_absorption(extraction, grammar)
        turning_point = extraction.turning_point

        rows.append(
            {
                "epsilon": float(epsilon),
                "seed": int(seed),
                "T": int(prefix_length),
                "valid": bool(extraction.valid),
                "predicted_failure": bool(prediction["predicted_failure"]),
                "absorbed": bool(absorption["absorbed"]),
                "score": float(extraction.score),
                "n_development": int(extraction.n_development),
                "pivot_timestamp": None if turning_point is None else float(turning_point.timestamp),
                "pivot_weight": None if turning_point is None else float(turning_point.weight),
            }
        )

    return rows


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["T"]))].append(row)

    summary_rows: list[dict] = []
    for epsilon in EPSILONS:
        for prefix_length in PREFIX_LENGTHS:
            bucket = grouped[(epsilon, prefix_length)]
            n = len(bucket)
            if n == 0:
                continue

            validity_rate = sum(1 for row in bucket if row["valid"]) / n
            absorption_rate = sum(1 for row in bucket if row["absorbed"]) / n
            mean_score = mean(float(row["score"]) for row in bucket)
            mean_n_dev = mean(float(row["n_development"]) for row in bucket)
            theorem_accuracy = (
                sum(
                    1
                    for row in bucket
                    if bool(row["predicted_failure"]) == (not bool(row["valid"]))
                )
                / n
            )

            summary_rows.append(
                {
                    "epsilon": float(epsilon),
                    "T": int(prefix_length),
                    "validity_rate": float(validity_rate),
                    "absorption_rate": float(absorption_rate),
                    "mean_score": float(mean_score),
                    "mean_n_dev": float(mean_n_dev),
                    "theorem_accuracy": float(theorem_accuracy),
                }
            )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | T | validity_rate | absorption_rate | mean_score | mean_n_dev | theorem_accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['T']} | {row['validity_rate']:.3f} | "
            f"{row['absorption_rate']:.3f} | {row['mean_score']:.3f} | "
            f"{row['mean_n_dev']:.3f} | {row['theorem_accuracy']:.3f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = BurstyGenerator()
    records: list[dict] = []
    total = len(EPSILONS) * len(SEEDS)
    completed = 0

    for epsilon in EPSILONS:
        for seed in SEEDS:
            records.extend(_run_instance(generator, epsilon=epsilon, seed=seed))
            completed += 1
        print(f"Completed epsilon={epsilon:.2f} ({completed}/{total} graph instances)")

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
