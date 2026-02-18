"""Experiment 43: organic oscillation-trap diagnostics on bursty traces."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

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
RAW_PATH = OUTPUT_DIR / "organic_oscillation_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "organic_oscillation_summary.md"


def _simulate_commit_now_streaming(graph, k: int) -> dict:
    events = tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))
    n_events = len(events)
    event_timestamps = {event.id: float(event.timestamp) for event in events}

    current_pivot = None
    current_pivot_weight = float("-inf")
    committed_labels: dict[str, str] = {}
    dev_count = 0
    tp_committed = False

    pivot_steps: list[int] = []
    n_pivot_shifts = 0
    last_shift_step = 0

    for step, event in enumerate(events):
        focal = FOCAL_ACTOR in event.actors

        if focal and float(event.weight) > current_pivot_weight:
            if tp_committed:
                n_pivot_shifts += 1
                last_shift_step = step

            current_pivot = event
            current_pivot_weight = float(event.weight)
            pivot_steps.append(step)

            # Conservative recount: only already-committed development labels
            # that are temporally before the new pivot remain prefix-eligible.
            dev_count = sum(
                1
                for event_id, label in committed_labels.items()
                if label == "DEVELOPMENT"
                and event_timestamps[event_id] < float(current_pivot.timestamp)
            )

            committed_labels[event.id] = "TURNING_POINT"
            tp_committed = True
            continue

        if tp_committed:
            if float(event.timestamp) < float(current_pivot.timestamp):
                committed_labels[event.id] = "DEVELOPMENT"
                dev_count += 1
            elif float(event.timestamp) > float(current_pivot.timestamp):
                committed_labels[event.id] = "RESOLUTION"
            else:
                # Same-timestamp events after TP commitment are treated as non-prefix.
                if event.id != current_pivot.id:
                    committed_labels[event.id] = "RESOLUTION"
        else:
            committed_labels[event.id] = "DEVELOPMENT"
            dev_count += 1

    if not tp_committed or current_pivot is None:
        inter_shift_gaps: list[int] = []
        first_pivot_step = n_events
        min_inter_shift_gap = n_events
        return {
            "streaming_valid": False,
            "n_pivot_shifts": int(n_pivot_shifts),
            "last_shift_step": int(last_shift_step),
            "last_shift_fraction": 0.0,
            "first_pivot_step": int(first_pivot_step),
            "inter_shift_gaps": inter_shift_gaps,
            "min_inter_shift_gap": int(min_inter_shift_gap),
            "final_dev_count": 0,
            "tp_committed": False,
        }

    final_dev_count = sum(
        1
        for event_id, label in committed_labels.items()
        if label == "DEVELOPMENT"
        and event_timestamps[event_id] < float(current_pivot.timestamp)
    )
    streaming_valid = bool(final_dev_count >= k)

    first_pivot_step = int(pivot_steps[0]) if pivot_steps else n_events
    inter_shift_gaps = [
        int(pivot_steps[idx] - pivot_steps[idx - 1]) for idx in range(1, len(pivot_steps))
    ]
    effective_gaps = [first_pivot_step] + inter_shift_gaps
    min_inter_shift_gap = min(effective_gaps) if effective_gaps else n_events
    if n_pivot_shifts == 0:
        last_shift_fraction = 0.0
    else:
        last_shift_fraction = float(last_shift_step / n_events) if n_events > 0 else 0.0

    return {
        "streaming_valid": streaming_valid,
        "n_pivot_shifts": int(n_pivot_shifts),
        "last_shift_step": int(last_shift_step),
        "last_shift_fraction": float(last_shift_fraction),
        "first_pivot_step": int(first_pivot_step),
        "inter_shift_gaps": inter_shift_gaps,
        "min_inter_shift_gap": int(min_inter_shift_gap),
        "final_dev_count": int(final_dev_count),
        "tp_committed": True,
    }


def _classify(finite_valid: bool, streaming_valid: bool) -> str:
    if finite_valid and streaming_valid:
        return "both_valid"
    if (not finite_valid) and (not streaming_valid):
        return "both_invalid"
    if finite_valid and (not streaming_valid):
        return "organic_trap"
    return "streaming_only"


def _run_instance(generator: BurstyGenerator, epsilon: float, k: int, seed: int) -> dict:
    graph = generator.generate(
        BurstyConfig(
            n_events=N_EVENTS,
            n_actors=N_ACTORS,
            seed=seed,
            epsilon=epsilon,
        )
    )
    grammar = GrammarConfig(min_prefix_elements=k)
    finite_result = greedy_extract(graph, focal_actor=FOCAL_ACTOR, grammar=grammar)
    finite_valid = bool(finite_result.valid)

    streaming = _simulate_commit_now_streaming(graph, k=k)
    streaming_valid = bool(streaming["streaming_valid"])
    classification = _classify(finite_valid=finite_valid, streaming_valid=streaming_valid)

    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "streaming_valid": streaming_valid,
        "classification": classification,
        "n_pivot_shifts": int(streaming["n_pivot_shifts"]),
        "first_pivot_step": int(streaming["first_pivot_step"]),
        "min_inter_shift_gap": int(streaming["min_inter_shift_gap"]),
        "last_shift_fraction": float(streaming["last_shift_fraction"]),
        "final_dev_count": int(streaming["final_dev_count"]),
        "inter_shift_gaps": list(streaming["inter_shift_gaps"]),
    }


def _build_summary(records: list[dict]) -> list[dict]:
    buckets: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        buckets[(float(row["epsilon"]), int(row["k"]))].append(row)

    summary_rows: list[dict] = []
    for epsilon in EPSILONS:
        for k in K_VALUES:
            bucket = buckets[(epsilon, k)]
            n_instances = len(bucket)
            if n_instances == 0:
                continue

            finite_valid_pct = (
                100.0 * sum(1 for row in bucket if bool(row["finite_valid"])) / n_instances
            )
            streaming_valid_pct = (
                100.0 * sum(1 for row in bucket if bool(row["streaming_valid"])) / n_instances
            )
            organic_trap_count = sum(1 for row in bucket if row["classification"] == "organic_trap")
            organic_trap_pct = 100.0 * organic_trap_count / n_instances
            mean_pivot_shifts = mean(float(row["n_pivot_shifts"]) for row in bucket)
            mean_min_gap = mean(float(row["min_inter_shift_gap"]) for row in bucket)
            pct_min_gap_lt_k = (
                100.0
                * sum(1 for row in bucket if float(row["min_inter_shift_gap"]) < float(k))
                / n_instances
            )

            summary_rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": int(k),
                    "n_instances": int(n_instances),
                    "finite_valid_pct": float(finite_valid_pct),
                    "streaming_valid_pct": float(streaming_valid_pct),
                    "organic_trap_pct": float(organic_trap_pct),
                    "organic_trap_count": int(organic_trap_count),
                    "mean_pivot_shifts": float(mean_pivot_shifts),
                    "mean_min_gap": float(mean_min_gap),
                    "pct_min_gap_lt_k": float(pct_min_gap_lt_k),
                }
            )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | k | n_instances | finite_valid_pct | streaming_valid_pct | organic_trap_pct | mean_pivot_shifts | mean_min_gap | pct_min_gap_lt_k |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | {row['n_instances']} | "
            f"{row['finite_valid_pct']:.1f} | {row['streaming_valid_pct']:.1f} | "
            f"{row['organic_trap_pct']:.1f} | {row['mean_pivot_shifts']:.2f} | "
            f"{row['mean_min_gap']:.2f} | {row['pct_min_gap_lt_k']:.1f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_terminal_summary(summary_rows: list[dict], total_instances: int) -> None:
    total_organic_traps = sum(int(row["organic_trap_count"]) for row in summary_rows)
    trap_pct = (100.0 * total_organic_traps / total_instances) if total_instances > 0 else 0.0

    ranked = sorted(
        summary_rows,
        key=lambda row: (float(row["organic_trap_pct"]), float(row["epsilon"]), int(row["k"])),
        reverse=True,
    )
    top5 = ranked[:5]

    print("=== ORGANIC OSCILLATION TRAP SUMMARY ===")
    print(f"Total instances: {total_instances}")
    print(f"Total organic traps: {total_organic_traps} ({trap_pct:.2f}%)")
    print("")
    print("Top 5 cells by organic trap rate:")
    for row in top5:
        print(
            f"  epsilon={row['epsilon']:.2f}, k={row['k']}: "
            f"{row['organic_trap_pct']:.2f}% organic trap rate "
            f"({row['organic_trap_count']}/{row['n_instances']} instances)"
        )
    print("")
    if total_organic_traps > 0:
        print("Conclusion: Organic traps detected")
    else:
        print("Conclusion: No organic traps — phenomenon is adversarial-only")


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
    _print_terminal_summary(summary_rows, total_instances=len(records))

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
