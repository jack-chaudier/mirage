"""Experiment 40: streaming pivot stability profile."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from rhun.generators.bursty import BurstyConfig, BurstyGenerator


FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
SEEDS = range(100)
N_EVENTS = 200
N_ACTORS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "pivot_stability_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "pivot_stability_summary.md"


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _run_instance(generator: BurstyGenerator, epsilon: float, seed: int) -> dict:
    graph = generator.generate(
        BurstyConfig(
            n_events=N_EVENTS,
            n_actors=N_ACTORS,
            seed=seed,
            epsilon=epsilon,
        )
    )

    events = tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))
    total_events = len(events)
    n_focal_events = sum(1 for event in events if FOCAL_ACTOR in event.actors)

    pivot_id_at_step: list[str | None] = []
    pivot_weight_at_step: list[float | None] = []
    current_pivot = None

    for event in events:
        if FOCAL_ACTOR in event.actors:
            if current_pivot is None or event.weight > current_pivot.weight:
                current_pivot = event

        pivot_id_at_step.append(None if current_pivot is None else current_pivot.id)
        pivot_weight_at_step.append(None if current_pivot is None else float(current_pivot.weight))

    n_pivot_shifts = 0
    last_shift_step = 0
    for idx in range(1, total_events):
        if pivot_id_at_step[idx] != pivot_id_at_step[idx - 1]:
            n_pivot_shifts += 1
            last_shift_step = idx

    last_shift_fraction = (last_shift_step / total_events) if total_events > 0 else 0.0

    final_pivot_position: float | None
    if current_pivot is None or total_events == 0:
        final_pivot_position = None
    else:
        min_timestamp = float(events[0].timestamp)
        max_timestamp = float(events[-1].timestamp)
        if abs(max_timestamp - min_timestamp) <= 1e-12:
            final_pivot_position = 0.5
        else:
            final_pivot_position = float(
                (current_pivot.timestamp - min_timestamp) / (max_timestamp - min_timestamp)
            )

    return {
        "seed": int(seed),
        "epsilon": float(epsilon),
        "n_pivot_shifts": int(n_pivot_shifts),
        "last_shift_step": int(last_shift_step),
        "last_shift_fraction": float(last_shift_fraction),
        "final_pivot_position": final_pivot_position,
        "total_events": int(total_events),
        "n_focal_events": int(n_focal_events),
    }


def _build_summary(records: list[dict]) -> list[dict]:
    by_epsilon: dict[float, list[dict]] = defaultdict(list)
    for row in records:
        by_epsilon[float(row["epsilon"])].append(row)

    summary_rows: list[dict] = []
    for epsilon in EPSILONS:
        bucket = by_epsilon.get(epsilon, [])
        if not bucket:
            continue

        shifts = [float(row["n_pivot_shifts"]) for row in bucket]
        last_shift_fracs = [float(row["last_shift_fraction"]) for row in bucket]
        pivot_positions = [
            float(row["final_pivot_position"])
            for row in bucket
            if row["final_pivot_position"] is not None
        ]
        stable_count = sum(1 for row in bucket if float(row["last_shift_fraction"]) < 0.5)
        pct_stable_by_half = 100.0 * stable_count / len(bucket)

        summary_rows.append(
            {
                "epsilon": float(epsilon),
                "mean_n_shifts": mean(shifts),
                "std_n_shifts": _std(shifts),
                "mean_last_shift_frac": mean(last_shift_fracs),
                "std_last_shift_frac": _std(last_shift_fracs),
                "mean_final_pivot_pos": mean(pivot_positions) if pivot_positions else float("nan"),
                "pct_stable_by_half": pct_stable_by_half,
            }
        )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | mean_n_shifts | std_n_shifts | mean_last_shift_frac | std_last_shift_frac | mean_final_pivot_pos | pct_stable_by_half |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['mean_n_shifts']:.3f} | {row['std_n_shifts']:.3f} | "
            f"{row['mean_last_shift_frac']:.3f} | {row['std_last_shift_frac']:.3f} | "
            f"{row['mean_final_pivot_pos']:.3f} | {row['pct_stable_by_half']:.1f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = BurstyGenerator()
    records: list[dict] = []

    for epsilon in EPSILONS:
        for seed in SEEDS:
            records.append(_run_instance(generator, epsilon=epsilon, seed=seed))
        print(f"Completed epsilon={epsilon:.2f} ({len(SEEDS)} seeds)")

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)

    eps005_rows = [row for row in records if abs(float(row["epsilon"]) - 0.05) <= 1e-12]
    mean_shifts = mean(float(row["n_pivot_shifts"]) for row in eps005_rows)
    mean_focal_events = mean(float(row["n_focal_events"]) for row in eps005_rows)
    expected_ln = math.log(mean_focal_events) if mean_focal_events > 0 else 0.0
    print(
        "At epsilon=0.05, mean shifts = "
        f"{mean_shifts:.3f} (expected ~ln(N_focal) = {expected_ln:.3f})"
    )

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
