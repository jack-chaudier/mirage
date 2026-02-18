"""Experiment 41: oscillation-induced streaming failure diagnostics."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.schemas import CausalGraph, Event


FOCAL_ACTOR = "actor_0"
K_VALUES = [1, 2, 3]
OSCILLATION_PERIODS = [3, 5, 10, 20, 50]
SEEDS = range(20)
N_EVENTS = 200

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "oscillation_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "oscillation_summary.md"


def _prefix_only_grammar(k: int) -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=k,
        min_length=1,
        max_length=N_EVENTS,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
        focal_actor_coverage=0.0,
    )


def _build_weights(seed: int, oscillation_period: int) -> list[float]:
    rng = random.Random(seed)
    weights = [rng.uniform(0.1, 0.5) for _ in range(N_EVENTS)]

    first_spike = max(1, oscillation_period // 2)
    spike_index = 0
    for idx in range(first_spike, N_EVENTS, oscillation_period):
        if spike_index % 2 == 0:
            weights[idx] = 0.9 + rng.uniform(0.0, 0.05)
        else:
            weights[idx] = 0.92 + rng.uniform(0.0, 0.05)
        spike_index += 1

    return weights


def _build_graph(weights: list[float], seed: int, oscillation_period: int) -> CausalGraph:
    events = tuple(
        Event(
            id=f"e{i:04d}",
            timestamp=float(i),
            weight=float(weights[i]),
            actors=frozenset({FOCAL_ACTOR}),
            causal_parents=(() if i == 0 else (f"e{i - 1:04d}",)),
            metadata={
                "generator": "oscillation_trap",
                "oscillation_period": oscillation_period,
            },
        )
        for i in range(N_EVENTS)
    )

    return CausalGraph(
        events=events,
        actors=frozenset({FOCAL_ACTOR}),
        seed=seed,
        metadata={
            "generator": "oscillation_trap",
            "oscillation_period": oscillation_period,
        },
    )


def _simulate_commit_now(events: tuple[Event, ...], k: int) -> dict:
    # Irreversible commitment model:
    # - Before any salient pivot appears, events are DEVELOPMENT.
    # - Once a pivot is committed, subsequent events are effectively RESOLUTION.
    # - Pivot shifts cannot retroactively recover those resolution commitments.
    current_pivot: Event | None = None
    current_pivot_weight = float("-inf")
    development_committed = 0
    n_pivot_shifts = 0

    for event in events:
        if current_pivot is None:
            if event.weight >= 0.9:
                current_pivot = event
                current_pivot_weight = float(event.weight)
            else:
                development_committed += 1
            continue

        if event.weight > current_pivot_weight:
            n_pivot_shifts += 1
            current_pivot = event
            current_pivot_weight = float(event.weight)

    if current_pivot is None:
        return {
            "n_pivot_shifts": 0,
            "commit_now_j_dev": int(development_committed),
            "commit_now_valid": bool(development_committed >= k),
        }

    final_j_dev = int(development_committed)
    final_valid = final_j_dev >= k
    return {
        "n_pivot_shifts": int(n_pivot_shifts),
        "commit_now_j_dev": int(final_j_dev),
        "commit_now_valid": bool(final_valid),
    }


def _simulate_commit_delayed(events: tuple[Event, ...], k: int) -> dict:
    candidate_idx: int | None = None
    candidate_weight = float("-inf")
    lock_step: int | None = None
    locked_pivot_idx: int | None = None

    for step, event in enumerate(events):
        if candidate_idx is None or event.weight > candidate_weight:
            candidate_idx = step
            candidate_weight = float(event.weight)

        if lock_step is None and candidate_idx is not None:
            dev_before_candidate = candidate_idx
            if dev_before_candidate >= k:
                lock_step = step
                locked_pivot_idx = candidate_idx

    if locked_pivot_idx is None:
        return {
            "commit_delayed_valid": False,
            "commit_delayed_lock_step": None,
        }

    final_j_dev = locked_pivot_idx
    return {
        "commit_delayed_valid": bool(final_j_dev >= k),
        "commit_delayed_lock_step": int(lock_step) if lock_step is not None else None,
    }


def _run_instance(k: int, oscillation_period: int, seed: int) -> dict:
    weights = _build_weights(seed=seed, oscillation_period=oscillation_period)
    graph = _build_graph(weights=weights, seed=seed, oscillation_period=oscillation_period)
    events = graph.events

    commit_now = _simulate_commit_now(events, k=k)
    commit_delayed = _simulate_commit_delayed(events, k=k)

    finite_result = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=_prefix_only_grammar(k),
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=N_EVENTS,
        injection_top_n=N_EVENTS,
    )

    return {
        "k": int(k),
        "oscillation_period": int(oscillation_period),
        "seed": int(seed),
        "n_pivot_shifts": int(commit_now["n_pivot_shifts"]),
        "commit_now_j_dev": int(commit_now["commit_now_j_dev"]),
        "commit_now_valid": bool(commit_now["commit_now_valid"]),
        "commit_delayed_valid": bool(commit_delayed["commit_delayed_valid"]),
        "commit_delayed_lock_step": commit_delayed["commit_delayed_lock_step"],
        "finite_greedy_valid": bool(finite_result.valid),
    }


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(int(row["k"]), int(row["oscillation_period"]))].append(row)

    summary_rows: list[dict] = []
    for k in K_VALUES:
        for period in OSCILLATION_PERIODS:
            bucket = grouped[(k, period)]
            n = len(bucket)
            if n == 0:
                continue
            commit_now_valid_pct = 100.0 * sum(1 for row in bucket if row["commit_now_valid"]) / n
            commit_delayed_valid_pct = (
                100.0 * sum(1 for row in bucket if row["commit_delayed_valid"]) / n
            )
            finite_greedy_valid_pct = (
                100.0 * sum(1 for row in bucket if row["finite_greedy_valid"]) / n
            )
            mean_pivot_shifts = mean(float(row["n_pivot_shifts"]) for row in bucket)

            summary_rows.append(
                {
                    "k": int(k),
                    "oscillation_period": int(period),
                    "commit_now_valid_pct": float(commit_now_valid_pct),
                    "commit_delayed_valid_pct": float(commit_delayed_valid_pct),
                    "finite_greedy_valid_pct": float(finite_greedy_valid_pct),
                    "mean_pivot_shifts": float(mean_pivot_shifts),
                }
            )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| k | osc_period | commit_now_valid_pct | commit_delayed_valid_pct | finite_greedy_valid_pct | mean_pivot_shifts |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['k']} | {row['oscillation_period']} | {row['commit_now_valid_pct']:.1f} | "
            f"{row['commit_delayed_valid_pct']:.1f} | {row['finite_greedy_valid_pct']:.1f} | "
            f"{row['mean_pivot_shifts']:.2f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    total = len(K_VALUES) * len(OSCILLATION_PERIODS) * len(SEEDS)
    completed = 0

    for k in K_VALUES:
        for period in OSCILLATION_PERIODS:
            for seed in SEEDS:
                records.append(_run_instance(k=k, oscillation_period=period, seed=seed))
                completed += 1
            print(f"Completed k={k}, period={period} ({completed}/{total} instances)")

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
