"""Experiment 46: buffered commitment latency profile on organic traps."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator


FOCAL_ACTOR = "actor_0"
EPSILONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
K_VALUES = [1, 2, 3]
SEEDS = range(200)
N_EVENTS = 200
N_ACTORS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "commit_latency_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "commit_latency_summary.md"


def _sorted_events(graph) -> tuple:
    return tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0.0:
        return float(min(values))
    if q >= 100.0:
        return float(max(values))

    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _simulate_commit_now_streaming(graph, k: int) -> dict:
    # Reused commit-now semantics from experiments 43-44.
    events = _sorted_events(graph)
    event_timestamps = {event.id: float(event.timestamp) for event in events}

    current_pivot = None
    current_pivot_weight = float("-inf")
    committed_labels: dict[str, str] = {}
    tp_committed = False

    for event in events:
        focal = FOCAL_ACTOR in event.actors
        if focal and float(event.weight) > current_pivot_weight:
            current_pivot = event
            current_pivot_weight = float(event.weight)
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
        return {"commit_now_valid": False}

    final_dev_count = sum(
        1
        for event_id, label in committed_labels.items()
        if label == "DEVELOPMENT"
        and event_timestamps[event_id] < float(current_pivot.timestamp)
    )
    return {"commit_now_valid": bool(final_dev_count >= k)}


def _simulate_buffered_commitment(graph, k: int) -> dict:
    # Variant B from experiment 44.
    events = _sorted_events(graph)
    event_timestamps = {event.id: float(event.timestamp) for event in events}

    events_seen: list = []
    best_focal = None
    best_weight = float("-inf")

    committed = False
    committed_pivot = None
    committed_dev_count = 0
    commit_step: int | None = None
    n_candidates_skipped = 0

    for step, event in enumerate(events):
        events_seen.append(event)
        focal = FOCAL_ACTOR in event.actors

        if committed:
            continue

        if focal and float(event.weight) > best_weight:
            best_focal = event
            best_weight = float(event.weight)

            pre_pivot_count = sum(
                1
                for seen in events_seen
                if float(seen.timestamp) < float(best_focal.timestamp)
                and seen.id != best_focal.id
            )
            if pre_pivot_count >= k:
                committed = True
                committed_pivot = best_focal
                commit_step = step
                committed_dev_count = pre_pivot_count
            else:
                n_candidates_skipped += 1

    if committed and committed_pivot is not None:
        min_ts = float(events[0].timestamp) if events else 0.0
        max_ts = float(events[-1].timestamp) if events else 0.0
        if abs(max_ts - min_ts) <= 1e-12:
            pivot_position = 0.5
        else:
            pivot_position = float((committed_pivot.timestamp - min_ts) / (max_ts - min_ts))
        commit_fraction = float(commit_step / len(events)) if events else 0.0
    else:
        pivot_position = None
        commit_fraction = None

    buffered_valid = bool(committed and committed_dev_count >= k)
    return {
        "buffered_committed": bool(committed),
        "buffered_commit_step": commit_step,
        "buffered_commit_fraction": commit_fraction,
        "buffered_pivot_position": pivot_position,
        "buffered_valid": buffered_valid,
        "n_candidates_skipped": int(n_candidates_skipped),
    }


def _run_instance(generator: BurstyGenerator, epsilon: float, k: int, seed: int) -> dict:
    graph = generator.generate(
        BurstyConfig(
            n_events=N_EVENTS,
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
    buffered = _simulate_buffered_commitment(graph, k=k)

    commit_now_valid = bool(commit_now["commit_now_valid"])
    organic_trap = bool(finite_valid and (not commit_now_valid))

    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "commit_now_valid": commit_now_valid,
        "buffered_committed": bool(buffered["buffered_committed"]),
        "buffered_commit_step": buffered["buffered_commit_step"],
        "buffered_commit_fraction": buffered["buffered_commit_fraction"],
        "buffered_pivot_position": buffered["buffered_pivot_position"],
        "buffered_valid": bool(buffered["buffered_valid"]),
        "n_candidates_skipped": int(buffered["n_candidates_skipped"]),
        "organic_trap": organic_trap,
    }


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _safe_median(values: list[float]) -> float:
    return float(median(values)) if values else float("nan")


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    summary_rows: list[dict] = []
    for epsilon in EPSILONS:
        for k in K_VALUES:
            bucket = grouped[(epsilon, k)]
            n_instances = len(bucket)
            if n_instances == 0:
                continue

            committed = [row for row in bucket if bool(row["buffered_committed"])]
            committed_fracs = [
                float(row["buffered_commit_fraction"])
                for row in committed
                if row["buffered_commit_fraction"] is not None
            ]

            trap_rows = [row for row in bucket if bool(row["organic_trap"])]
            trap_committed = [row for row in trap_rows if bool(row["buffered_committed"])]
            trap_fracs = [
                float(row["buffered_commit_fraction"])
                for row in trap_committed
                if row["buffered_commit_fraction"] is not None
            ]

            nontrap_rows = [row for row in bucket if not bool(row["organic_trap"])]
            nontrap_committed = [row for row in nontrap_rows if bool(row["buffered_committed"])]
            nontrap_fracs = [
                float(row["buffered_commit_fraction"])
                for row in nontrap_committed
                if row["buffered_commit_fraction"] is not None
            ]

            buffered_commit_pct = (100.0 * len(committed) / n_instances) if n_instances > 0 else 0.0
            mean_commit_frac = _safe_mean(committed_fracs)
            median_commit_frac = _safe_median(committed_fracs)
            p90_commit_frac = _percentile(committed_fracs, 90.0)

            mean_commit_frac_traps = _safe_mean(trap_fracs)
            median_commit_frac_traps = _safe_median(trap_fracs)
            mean_commit_frac_non_traps = _safe_mean(nontrap_fracs)
            median_commit_frac_non_traps = _safe_median(nontrap_fracs)

            mean_candidates_skipped = _safe_mean(
                [float(row["n_candidates_skipped"]) for row in bucket]
            )
            mean_candidates_skipped_traps = _safe_mean(
                [float(row["n_candidates_skipped"]) for row in trap_rows]
            )
            mean_candidates_skipped_non_traps = _safe_mean(
                [float(row["n_candidates_skipped"]) for row in nontrap_rows]
            )

            summary_rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": int(k),
                    "n_instances": int(n_instances),
                    "buffered_commit_pct": float(buffered_commit_pct),
                    "mean_commit_frac": mean_commit_frac,
                    "median_commit_frac": median_commit_frac,
                    "p90_commit_frac": p90_commit_frac,
                    "mean_commit_frac_traps": mean_commit_frac_traps,
                    "median_commit_frac_traps": median_commit_frac_traps,
                    "mean_commit_frac_non_traps": mean_commit_frac_non_traps,
                    "median_commit_frac_non_traps": median_commit_frac_non_traps,
                    "mean_candidates_skipped": mean_candidates_skipped,
                    "mean_candidates_skipped_traps": mean_candidates_skipped_traps,
                    "mean_candidates_skipped_non_traps": mean_candidates_skipped_non_traps,
                }
            )

    return summary_rows


def _format_value(value: float) -> str:
    if value != value:  # NaN check
        return "n/a"
    return f"{value:.3f}"


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | k | n_instances | buffered_commit_pct | mean_commit_frac | median_commit_frac | p90_commit_frac | mean_commit_frac_traps | median_commit_frac_traps | mean_commit_frac_non_traps | median_commit_frac_non_traps | mean_candidates_skipped | mean_candidates_skipped_traps | mean_candidates_skipped_non_traps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | {row['n_instances']} | "
            f"{row['buffered_commit_pct']:.1f} | "
            f"{_format_value(row['mean_commit_frac'])} | {_format_value(row['median_commit_frac'])} | {_format_value(row['p90_commit_frac'])} | "
            f"{_format_value(row['mean_commit_frac_traps'])} | {_format_value(row['median_commit_frac_traps'])} | "
            f"{_format_value(row['mean_commit_frac_non_traps'])} | {_format_value(row['median_commit_frac_non_traps'])} | "
            f"{_format_value(row['mean_candidates_skipped'])} | {_format_value(row['mean_candidates_skipped_traps'])} | "
            f"{_format_value(row['mean_candidates_skipped_non_traps'])} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_percentile_table(records: list[dict]) -> None:
    percentiles = [10, 25, 50, 75, 90, 95]

    print("=== BUFFERED COMMITMENT LATENCY PROFILE ===")
    print("")
    print("Overall (all instances that committed):")
    print("  Percentile |  k=1  |  k=2  |  k=3")
    print("  ---------- | ----- | ----- | -----")
    for p in percentiles:
        values = {}
        for k in K_VALUES:
            vals = [
                float(row["buffered_commit_fraction"])
                for row in records
                if int(row["k"]) == k
                and bool(row["buffered_committed"])
                and row["buffered_commit_fraction"] is not None
            ]
            values[k] = _percentile(vals, float(p))
        print(
            f"  {p:<10} | {values[1]:5.2f} | {values[2]:5.2f} | {values[3]:5.2f}"
        )

    print("")
    print("Organic trap instances only:")
    print("  Percentile |  k=1  |  k=2  |  k=3")
    print("  ---------- | ----- | ----- | -----")
    for p in percentiles:
        values = {}
        for k in K_VALUES:
            vals = [
                float(row["buffered_commit_fraction"])
                for row in records
                if int(row["k"]) == k
                and bool(row["organic_trap"])
                and bool(row["buffered_committed"])
                and row["buffered_commit_fraction"] is not None
            ]
            values[k] = _percentile(vals, float(p))
        print(
            f"  {p:<10} | {values[1]:5.2f} | {values[2]:5.2f} | {values[3]:5.2f}"
        )

    print("")
    overall_median = _percentile(
        [
            float(row["buffered_commit_fraction"])
            for row in records
            if bool(row["buffered_committed"]) and row["buffered_commit_fraction"] is not None
        ],
        50.0,
    )
    if overall_median < 0.15:
        print(
            'Interpretation:\n'
            '  "Buffering is nearly free — commitment happens in the first 15% of events"'
        )
    elif overall_median > 0.50:
        print(
            'Interpretation:\n'
            '  "Buffering is expensive — system waits for half the stream before committing"'
        )
    else:
        print(
            "Interpretation:\n"
            "  Buffering has moderate latency cost — commitment happens mid-early stream."
        )

    trap_mean = _safe_mean(
        [
            float(row["buffered_commit_fraction"])
            for row in records
            if bool(row["organic_trap"])
            and bool(row["buffered_committed"])
            and row["buffered_commit_fraction"] is not None
        ]
    )
    nontrap_mean = _safe_mean(
        [
            float(row["buffered_commit_fraction"])
            for row in records
            if (not bool(row["organic_trap"]))
            and bool(row["buffered_committed"])
            and row["buffered_commit_fraction"] is not None
        ]
    )
    print("")
    print(
        "  Key comparison: commit_frac for trap instances vs non-trap instances."
    )
    print(
        f"  Mean trap commit_frac={trap_mean:.3f}, mean non-trap commit_frac={nontrap_mean:.3f}."
    )
    if trap_mean > nontrap_mean + 1e-9:
        print(
            "  Trap instances commit later on average, so buffering is doing real delay-work on hard cases."
        )
    else:
        print(
            "  Trap and non-trap commit times are similar, so pivot selection dominates over pure delay."
        )

    print("")
    print("=== CANDIDATES SKIPPED ===")
    print("Mean focal events skipped before commitment:")
    for k in K_VALUES:
        overall_skip = _safe_mean(
            [float(row["n_candidates_skipped"]) for row in records if int(row["k"]) == k]
        )
        trap_skip = _safe_mean(
            [
                float(row["n_candidates_skipped"])
                for row in records
                if int(row["k"]) == k and bool(row["organic_trap"])
            ]
        )
        print(f"  k={k}: {overall_skip:.1f} overall, {trap_skip:.1f} on trap instances")
    print("")
    print(
        "This measures how many 'wrong' pivots the buffered policy avoids.\n"
        "If trap instances skip more candidates, the buffer is actively rejecting\n"
        "dangerous early pivots that commit-now would have locked onto."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = BurstyGenerator()
    records: list[dict] = []

    total = len(EPSILONS) * len(K_VALUES) * len(SEEDS)
    completed = 0
    for epsilon in EPSILONS:
        for k in K_VALUES:
            for seed in SEEDS:
                records.append(_run_instance(generator, epsilon=epsilon, k=k, seed=seed))
                completed += 1
            print(
                f"Completed epsilon={epsilon:.2f}, k={k} "
                f"({completed}/{total} instances)"
            )

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)
    _print_percentile_table(records)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
