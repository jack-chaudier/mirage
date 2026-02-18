"""Experiment 45: scale dependence of organic streaming-trap rates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator


FOCAL_ACTOR = "actor_0"
N_EVENTS_VALUES = [100, 200, 500, 1000]
EPSILONS = [0.20, 0.40, 0.60]
K_VALUES = [2, 3]
SEEDS = range(100)
N_ACTORS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "scale_dependence_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "scale_dependence_summary.md"


def _sorted_events(graph) -> tuple:
    return tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))


def _simulate_commit_now_streaming(graph, k: int) -> dict:
    # Reused commit-now semantics from experiment 43.
    events = _sorted_events(graph)
    n_events = len(events)
    event_timestamps = {event.id: float(event.timestamp) for event in events}

    current_pivot = None
    current_pivot_weight = float("-inf")
    committed_labels: dict[str, str] = {}
    tp_committed = False

    pivot_steps: list[int] = []
    n_pivot_shifts = 0

    for step, event in enumerate(events):
        focal = FOCAL_ACTOR in event.actors

        if focal and float(event.weight) > current_pivot_weight:
            if tp_committed:
                n_pivot_shifts += 1

            current_pivot = event
            current_pivot_weight = float(event.weight)
            pivot_steps.append(step)
            committed_labels[event.id] = "TURNING_POINT"
            tp_committed = True
            continue

        if tp_committed:
            if float(event.timestamp) < float(current_pivot.timestamp):
                committed_labels[event.id] = "DEVELOPMENT"
            elif float(event.timestamp) > float(current_pivot.timestamp):
                committed_labels[event.id] = "RESOLUTION"
            elif event.id != current_pivot.id:
                committed_labels[event.id] = "RESOLUTION"
        else:
            committed_labels[event.id] = "DEVELOPMENT"

    if not tp_committed or current_pivot is None:
        return {
            "commit_now_valid": False,
            "min_gap": int(n_events),
            "n_pivot_shifts": int(n_pivot_shifts),
        }

    final_dev_count = sum(
        1
        for event_id, label in committed_labels.items()
        if label == "DEVELOPMENT"
        and event_timestamps[event_id] < float(current_pivot.timestamp)
    )
    commit_now_valid = bool(final_dev_count >= k)

    first_pivot_step = int(pivot_steps[0]) if pivot_steps else n_events
    inter_shift_gaps = [
        int(pivot_steps[idx] - pivot_steps[idx - 1]) for idx in range(1, len(pivot_steps))
    ]
    effective_gaps = [first_pivot_step] + inter_shift_gaps
    min_gap = min(effective_gaps) if effective_gaps else n_events

    return {
        "commit_now_valid": commit_now_valid,
        "min_gap": int(min_gap),
        "n_pivot_shifts": int(n_pivot_shifts),
    }


def _run_instance(
    generator: BurstyGenerator,
    n_events: int,
    epsilon: float,
    k: int,
    seed: int,
) -> dict:
    graph = generator.generate(
        BurstyConfig(
            n_events=n_events,
            n_actors=N_ACTORS,
            seed=seed,
            epsilon=epsilon,
        )
    )

    finite = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=GrammarConfig(min_prefix_elements=k),
    )
    finite_valid = bool(finite.valid)

    commit_now = _simulate_commit_now_streaming(graph, k=k)
    commit_now_valid = bool(commit_now["commit_now_valid"])

    organic_trap = bool(finite_valid and (not commit_now_valid))
    n_focal_events = sum(1 for event in graph.events if FOCAL_ACTOR in event.actors)

    return {
        "n_events": int(n_events),
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "commit_now_valid": commit_now_valid,
        "organic_trap": organic_trap,
        "min_gap": int(commit_now["min_gap"]),
        "n_pivot_shifts": int(commit_now["n_pivot_shifts"]),
        "n_focal_events": int(n_focal_events),
    }


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(int(row["n_events"]), float(row["epsilon"]), int(row["k"]))].append(row)

    summary_rows: list[dict] = []
    for n_events in N_EVENTS_VALUES:
        for epsilon in EPSILONS:
            for k in K_VALUES:
                bucket = grouped[(n_events, epsilon, k)]
                n_instances = len(bucket)
                if n_instances == 0:
                    continue

                finite_valid_pct = (
                    100.0 * sum(1 for row in bucket if bool(row["finite_valid"])) / n_instances
                )
                streaming_valid_pct = (
                    100.0 * sum(1 for row in bucket if bool(row["commit_now_valid"])) / n_instances
                )
                organic_trap_pct = (
                    100.0 * sum(1 for row in bucket if bool(row["organic_trap"])) / n_instances
                )
                min_gaps = [float(row["min_gap"]) for row in bucket]
                mean_min_gap = mean(min_gaps)
                median_min_gap = median(min_gaps)
                mean_n_shifts = mean(float(row["n_pivot_shifts"]) for row in bucket)
                mean_n_focal = mean(float(row["n_focal_events"]) for row in bucket)

                summary_rows.append(
                    {
                        "n_events": int(n_events),
                        "epsilon": float(epsilon),
                        "k": int(k),
                        "n_instances": int(n_instances),
                        "finite_valid_pct": float(finite_valid_pct),
                        "streaming_valid_pct": float(streaming_valid_pct),
                        "organic_trap_pct": float(organic_trap_pct),
                        "mean_min_gap": float(mean_min_gap),
                        "median_min_gap": float(median_min_gap),
                        "mean_n_shifts": float(mean_n_shifts),
                        "mean_n_focal": float(mean_n_focal),
                    }
                )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| n_events | epsilon | k | n_instances | finite_valid_pct | streaming_valid_pct | organic_trap_pct | mean_min_gap | median_min_gap | mean_n_shifts | mean_n_focal |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['n_events']} | {row['epsilon']:.2f} | {row['k']} | {row['n_instances']} | "
            f"{row['finite_valid_pct']:.1f} | {row['streaming_valid_pct']:.1f} | "
            f"{row['organic_trap_pct']:.1f} | {row['mean_min_gap']:.2f} | "
            f"{row['median_min_gap']:.2f} | {row['mean_n_shifts']:.2f} | {row['mean_n_focal']:.2f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_scale_summary(summary_rows: list[dict]) -> None:
    print("=== SCALE DEPENDENCE SUMMARY ===")
    print("")
    print("Organic trap rate by n_events (averaged across epsilon and k):")
    for n_events in N_EVENTS_VALUES:
        bucket = [row for row in summary_rows if int(row["n_events"]) == n_events]
        trap_rate = mean(float(row["organic_trap_pct"]) for row in bucket) if bucket else 0.0
        mean_min_gap = mean(float(row["mean_min_gap"]) for row in bucket) if bucket else 0.0
        mean_shifts = mean(float(row["mean_n_shifts"]) for row in bucket) if bucket else 0.0
        mean_focal = mean(float(row["mean_n_focal"]) for row in bucket) if bucket else 0.0
        print(
            f"  n={n_events}:  {trap_rate:.2f}%  "
            f"(mean_min_gap={mean_min_gap:.2f}, mean_shifts={mean_shifts:.2f}, mean_focal={mean_focal:.2f})"
        )

    print("")
    print("Does trap rate decrease monotonically with n for each (epsilon, k)?")
    for epsilon in EPSILONS:
        for k in K_VALUES:
            rates = []
            for n_events in N_EVENTS_VALUES:
                row = next(
                    (
                        entry
                        for entry in summary_rows
                        if int(entry["n_events"]) == n_events
                        and abs(float(entry["epsilon"]) - epsilon) <= 1e-12
                        and int(entry["k"]) == k
                    ),
                    None,
                )
                rates.append(float(row["organic_trap_pct"]) if row is not None else 0.0)
            monotone = all(rates[idx] >= rates[idx + 1] for idx in range(len(rates) - 1))
            flag = "MONOTONE DECREASING" if monotone else "NOT MONOTONE"
            rate_str = ", ".join(
                f"n={n_events} -> {rate:.2f}%" for n_events, rate in zip(N_EVENTS_VALUES, rates, strict=True)
            )
            print(f"  epsilon={epsilon:.2f}, k={k}: {rate_str}  [{flag}]")

    print("")
    print("Prediction: Under record theory, min_gap should grow as O(n_focal / (ln n_focal)^2).")
    print("At n_focal ~ n/6:")
    for n_events in N_EVENTS_VALUES:
        n_focal = n_events / 6.0
        expected = n_focal / (math.log(n_focal) ** 2) if n_focal > 1.0 else 0.0
        print(
            f"  n={n_events:<4} -> n_focal~{n_focal:.0f},  "
            f"expected min_gap ~ {expected:.1f}"
        )
    print("So traps should decrease with n but slowly, especially for k=3.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = BurstyGenerator()
    records: list[dict] = []

    total = len(N_EVENTS_VALUES) * len(EPSILONS) * len(K_VALUES) * len(SEEDS)
    completed = 0
    for n_events in N_EVENTS_VALUES:
        for epsilon in EPSILONS:
            for k in K_VALUES:
                for seed in SEEDS:
                    records.append(
                        _run_instance(
                            generator=generator,
                            n_events=n_events,
                            epsilon=epsilon,
                            k=k,
                            seed=seed,
                        )
                    )
                    completed += 1
                print(
                    f"Completed n_events={n_events}, epsilon={epsilon:.2f}, k={k} "
                    f"({completed}/{total} instances)"
                )

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)
    _print_scale_summary(summary_rows)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
