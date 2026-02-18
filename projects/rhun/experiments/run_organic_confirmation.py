"""Experiment 44: organic trap confirmation via delayed commitment policies."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator


FOCAL_ACTOR = "actor_0"
EPSILONS = [0.20, 0.40, 0.60]
K_VALUES = [1, 2, 3]
SEEDS = range(200)
N_EVENTS = 200
N_ACTORS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "organic_confirmation_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "organic_confirmation_summary.md"


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
            "n_pivot_shifts": int(n_pivot_shifts),
            "last_shift_step": int(last_shift_step),
            "last_shift_fraction": 0.0,
            "min_gap": int(n_events),
            "final_dev_count": 0,
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
    if n_pivot_shifts == 0:
        last_shift_fraction = 0.0
    else:
        last_shift_fraction = float(last_shift_step / n_events) if n_events > 0 else 0.0

    return {
        "commit_now_valid": commit_now_valid,
        "n_pivot_shifts": int(n_pivot_shifts),
        "last_shift_step": int(last_shift_step),
        "last_shift_fraction": float(last_shift_fraction),
        "min_gap": int(min_gap),
        "final_dev_count": int(final_dev_count),
    }


def _simulate_kstability_delayed(graph, k: int) -> dict:
    events = _sorted_events(graph)
    events_seen: list = []

    candidate_pivot = None
    candidate_weight = float("-inf")
    stability_counter = 0

    committed = False
    committed_pivot = None
    committed_dev_count = 0
    commit_step: int | None = None

    for step, event in enumerate(events):
        events_seen.append(event)
        focal = FOCAL_ACTOR in event.actors

        if committed:
            continue

        if focal and float(event.weight) > candidate_weight:
            candidate_pivot = event
            candidate_weight = float(event.weight)
            stability_counter = 0
        else:
            stability_counter += 1

        if candidate_pivot is not None and stability_counter >= k:
            committed = True
            committed_pivot = candidate_pivot
            commit_step = step
            committed_dev_count = sum(
                1
                for seen in events_seen
                if float(seen.timestamp) < float(committed_pivot.timestamp)
                and seen.id != committed_pivot.id
            )

    kstable_valid = bool(committed and committed_dev_count >= k)
    return {
        "kstable_valid": kstable_valid,
        "kstable_committed": bool(committed),
        "kstable_commit_step": commit_step,
        "kstable_dev_count": int(committed_dev_count),
    }


def _simulate_buffered_delayed(graph, k: int) -> dict:
    events = _sorted_events(graph)
    events_seen: list = []

    best_focal = None
    best_weight = float("-inf")

    committed = False
    committed_dev_count = 0
    commit_step: int | None = None

    for step, event in enumerate(events):
        events_seen.append(event)
        focal = FOCAL_ACTOR in event.actors

        if committed:
            continue

        if focal and float(event.weight) > best_weight:
            best_focal = event
            best_weight = float(event.weight)

        if best_focal is None:
            continue

        pre_pivot_count = sum(
            1
            for seen in events_seen
            if float(seen.timestamp) < float(best_focal.timestamp)
            and seen.id != best_focal.id
        )
        if pre_pivot_count >= k:
            committed = True
            commit_step = step
            committed_dev_count = pre_pivot_count

    buffered_valid = bool(committed and committed_dev_count >= k)
    return {
        "buffered_valid": buffered_valid,
        "buffered_committed": bool(committed),
        "buffered_commit_step": commit_step,
        "buffered_dev_count": int(committed_dev_count),
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
    kstable = _simulate_kstability_delayed(graph, k=k)
    buffered = _simulate_buffered_delayed(graph, k=k)

    commit_now_valid = bool(commit_now["commit_now_valid"])
    organic_trap = bool(finite_valid and (not commit_now_valid))
    kstable_recovers = bool(organic_trap and kstable["kstable_valid"])
    buffered_recovers = bool(organic_trap and buffered["buffered_valid"])

    gap_predicts_trap = bool((commit_now["min_gap"] < k) == (not commit_now_valid))
    gap_false_positive = bool((commit_now["min_gap"] < k) and commit_now_valid)
    gap_false_negative = bool((commit_now["min_gap"] >= k) and (not commit_now_valid))

    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "commit_now_valid": commit_now_valid,
        "kstable_valid": bool(kstable["kstable_valid"]),
        "kstable_committed": bool(kstable["kstable_committed"]),
        "kstable_commit_step": kstable["kstable_commit_step"],
        "buffered_valid": bool(buffered["buffered_valid"]),
        "buffered_committed": bool(buffered["buffered_committed"]),
        "buffered_commit_step": buffered["buffered_commit_step"],
        "organic_trap": organic_trap,
        "kstable_recovers": kstable_recovers,
        "buffered_recovers": buffered_recovers,
        "min_gap": int(commit_now["min_gap"]),
        "n_pivot_shifts": int(commit_now["n_pivot_shifts"]),
        "last_shift_fraction": float(commit_now["last_shift_fraction"]),
        "final_dev_count": int(commit_now["final_dev_count"]),
        "gap_predicts_trap": gap_predicts_trap,
        "gap_false_positive": gap_false_positive,
        "gap_false_negative": gap_false_negative,
    }


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    summary_rows: list[dict] = []
    for epsilon in EPSILONS:
        for k in K_VALUES:
            bucket = grouped[(epsilon, k)]
            finite_valid_rows = [row for row in bucket if bool(row["finite_valid"])]
            n_finite_valid = len(finite_valid_rows)

            organic_traps = sum(1 for row in finite_valid_rows if bool(row["organic_trap"]))
            kstable_recovers = sum(1 for row in finite_valid_rows if bool(row["kstable_recovers"]))
            buffered_recovers = sum(1 for row in finite_valid_rows if bool(row["buffered_recovers"]))

            kstable_recovery_pct = (
                (100.0 * kstable_recovers / organic_traps) if organic_traps > 0 else 0.0
            )
            buffered_recovery_pct = (
                (100.0 * buffered_recovers / organic_traps) if organic_traps > 0 else 0.0
            )

            gap_fp = sum(1 for row in finite_valid_rows if bool(row["gap_false_positive"]))
            gap_fn = sum(1 for row in finite_valid_rows if bool(row["gap_false_negative"]))
            gap_correct = sum(1 for row in finite_valid_rows if bool(row["gap_predicts_trap"]))
            gap_accuracy = (100.0 * gap_correct / n_finite_valid) if n_finite_valid > 0 else 0.0

            summary_rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": int(k),
                    "n_finite_valid": int(n_finite_valid),
                    "organic_traps": int(organic_traps),
                    "kstable_recovers": int(kstable_recovers),
                    "kstable_recovery_pct": float(kstable_recovery_pct),
                    "buffered_recovers": int(buffered_recovers),
                    "buffered_recovery_pct": float(buffered_recovery_pct),
                    "gap_FP": int(gap_fp),
                    "gap_FN": int(gap_fn),
                    "gap_accuracy_among_finite_valid": float(gap_accuracy),
                }
            )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | k | n_finite_valid | organic_traps | kstable_recovers | kstable_recovery_pct | buffered_recovers | buffered_recovery_pct | gap_FP | gap_FN | gap_accuracy_among_finite_valid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | {row['n_finite_valid']} | "
            f"{row['organic_traps']} | {row['kstable_recovers']} | {row['kstable_recovery_pct']:.1f} | "
            f"{row['buffered_recovers']} | {row['buffered_recovery_pct']:.1f} | "
            f"{row['gap_FP']} | {row['gap_FN']} | {row['gap_accuracy_among_finite_valid']:.1f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_global_summary(records: list[dict]) -> None:
    finite_valid_rows = [row for row in records if bool(row["finite_valid"])]
    organic_traps = [row for row in finite_valid_rows if bool(row["organic_trap"])]
    n_organic = len(organic_traps)

    kstable_recovers = sum(1 for row in organic_traps if bool(row["kstable_valid"]))
    buffered_recovers = sum(1 for row in organic_traps if bool(row["buffered_valid"]))

    kstable_pct = (100.0 * kstable_recovers / n_organic) if n_organic > 0 else 0.0
    buffered_pct = (100.0 * buffered_recovers / n_organic) if n_organic > 0 else 0.0

    tp = sum(1 for row in finite_valid_rows if row["min_gap"] < row["k"] and (not row["commit_now_valid"]))
    fp = sum(1 for row in finite_valid_rows if row["min_gap"] < row["k"] and row["commit_now_valid"])
    fn = sum(1 for row in finite_valid_rows if row["min_gap"] >= row["k"] and (not row["commit_now_valid"]))
    tn = sum(1 for row in finite_valid_rows if row["min_gap"] >= row["k"] and row["commit_now_valid"])
    total = len(finite_valid_rows)

    accuracy = (100.0 * (tp + tn) / total) if total > 0 else 0.0
    precision = (100.0 * tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    print("=== COMMIT-DELAYED RECOVERY SUMMARY ===")
    print(f"Total organic traps: {n_organic}")
    print("")
    print(
        f"K-stability (Variant A): recovers {kstable_recovers} / {n_organic} "
        f"({kstable_pct:.2f}%)"
    )
    print(
        f"Buffered (Variant B):    recovers {buffered_recovers} / {n_organic} "
        f"({buffered_pct:.2f}%)"
    )
    print("")
    print("=== MECHANISM VERIFICATION (among finite-valid instances only) ===")
    print("min_gap < k as predictor of commit-now failure:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  False positives: {fp} (gap < k but commit-now valid)")
    print(f"  False negatives: {fn} (gap >= k but commit-now invalid)")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")


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
    _print_global_summary(records)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
