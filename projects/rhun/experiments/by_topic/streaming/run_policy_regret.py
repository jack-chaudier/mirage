"""Experiment 47: policy regret analysis for streaming commitment policies."""

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
RAW_PATH = OUTPUT_DIR / "policy_regret_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "policy_regret_summary.md"


def _sorted_events(graph) -> tuple:
    return tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))


def _simulate_commit_now(graph, k: int) -> dict:
    # Reused commit-now semantics from experiments 43-46.
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
        return {
            "commit_now_valid": False,
            "commit_now_pivot_id": None,
            "commit_now_pivot_weight": None,
        }

    final_dev_count = sum(
        1
        for event_id, label in committed_labels.items()
        if label == "DEVELOPMENT"
        and event_timestamps[event_id] < float(current_pivot.timestamp)
    )
    commit_now_valid = bool(final_dev_count >= k)
    return {
        "commit_now_valid": commit_now_valid,
        "commit_now_pivot_id": current_pivot.id,
        "commit_now_pivot_weight": float(current_pivot.weight),
    }


def _simulate_buffered(graph, k: int) -> dict:
    # Variant B from experiments 44/46.
    events = _sorted_events(graph)
    events_seen: list = []

    best_focal = None
    best_weight = float("-inf")

    committed = False
    committed_dev_count = 0

    for event in events:
        events_seen.append(event)
        if committed:
            continue

        focal = FOCAL_ACTOR in event.actors
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
                committed_dev_count = pre_pivot_count

    buffered_valid = bool(committed and committed_dev_count >= k)
    return {
        "buffered_valid": buffered_valid,
        "buffered_pivot_id": None if best_focal is None else best_focal.id if committed else None,
        "buffered_pivot_weight": (
            None if best_focal is None else float(best_focal.weight) if committed else None
        ),
    }


def _forced_score_for_pivot(
    graph,
    grammar: GrammarConfig,
    pivot_id: str | None,
    cache: dict[str, tuple[bool, float]],
) -> tuple[bool, float]:
    if pivot_id is None:
        return False, 0.0
    if pivot_id in cache:
        return cache[pivot_id]

    forced = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        override_tp_id=pivot_id,
    )
    valid = bool(forced.valid)
    score = float(forced.score) if valid else 0.0
    cache[pivot_id] = (valid, score)
    return valid, score


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


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

    finite = greedy_extract(graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar)
    finite_valid = bool(finite.valid)
    finite_score = float(finite.score) if finite_valid else 0.0
    finite_tp = finite.turning_point
    finite_pivot_id = None if finite_tp is None else finite_tp.id
    finite_pivot_weight = None if finite_tp is None else float(finite_tp.weight)

    commit_now = _simulate_commit_now(graph, k=k)
    buffered = _simulate_buffered(graph, k=k)

    score_cache: dict[str, tuple[bool, float]] = {}

    commit_now_valid = bool(commit_now["commit_now_valid"])
    commit_now_pivot_id = commit_now["commit_now_pivot_id"]
    if commit_now_valid and commit_now_pivot_id is not None:
        _, commit_now_score = _forced_score_for_pivot(
            graph=graph,
            grammar=grammar,
            pivot_id=commit_now_pivot_id,
            cache=score_cache,
        )
    else:
        commit_now_score = 0.0

    buffered_valid = bool(buffered["buffered_valid"])
    buffered_pivot_id = buffered["buffered_pivot_id"]
    if buffered_valid and buffered_pivot_id is not None:
        _, buffered_score = _forced_score_for_pivot(
            graph=graph,
            grammar=grammar,
            pivot_id=buffered_pivot_id,
            cache=score_cache,
        )
    else:
        buffered_score = 0.0

    commit_now_regret = float(finite_score - commit_now_score)
    buffered_regret = float(finite_score - buffered_score)

    organic_trap = bool(finite_valid and (not commit_now_valid))
    both_valid = bool(finite_valid and commit_now_valid)
    both_invalid = bool((not finite_valid) and (not commit_now_valid))

    same_pivot_cn = bool(finite_pivot_id is not None and commit_now_pivot_id == finite_pivot_id)
    same_pivot_buf = bool(finite_pivot_id is not None and buffered_pivot_id == finite_pivot_id)

    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "finite_score": float(finite_score),
        "finite_pivot_id": finite_pivot_id,
        "finite_pivot_weight": finite_pivot_weight,
        "commit_now_valid": commit_now_valid,
        "commit_now_score": float(commit_now_score),
        "commit_now_pivot_id": commit_now_pivot_id,
        "commit_now_pivot_weight": commit_now["commit_now_pivot_weight"],
        "buffered_valid": buffered_valid,
        "buffered_score": float(buffered_score),
        "buffered_pivot_id": buffered_pivot_id,
        "buffered_pivot_weight": buffered["buffered_pivot_weight"],
        "commit_now_regret": float(commit_now_regret),
        "buffered_regret": float(buffered_regret),
        "organic_trap": organic_trap,
        "both_valid": both_valid,
        "both_invalid": both_invalid,
        "same_pivot_commit_now": same_pivot_cn,
        "same_pivot_buffered": same_pivot_buf,
    }


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

            finite_valid_pct = 100.0 * sum(1 for row in bucket if row["finite_valid"]) / n_instances
            cn_valid_pct = 100.0 * sum(1 for row in bucket if row["commit_now_valid"]) / n_instances
            buf_valid_pct = 100.0 * sum(1 for row in bucket if row["buffered_valid"]) / n_instances

            mean_finite_score = _safe_mean([float(row["finite_score"]) for row in bucket])
            mean_cn_score = _safe_mean([float(row["commit_now_score"]) for row in bucket])
            mean_buf_score = _safe_mean([float(row["buffered_score"]) for row in bucket])

            mean_cn_regret = _safe_mean([float(row["commit_now_regret"]) for row in bucket])
            mean_buf_regret = _safe_mean([float(row["buffered_regret"]) for row in bucket])

            both_valid_cn_rows = [
                row for row in bucket if bool(row["finite_valid"]) and bool(row["commit_now_valid"])
            ]
            both_valid_buf_rows = [
                row for row in bucket if bool(row["finite_valid"]) and bool(row["buffered_valid"])
            ]
            mean_cn_regret_bv = _safe_mean(
                [float(row["commit_now_regret"]) for row in both_valid_cn_rows]
            )
            mean_buf_regret_bv = _safe_mean(
                [float(row["buffered_regret"]) for row in both_valid_buf_rows]
            )

            pct_same_pivot_cn = (
                100.0 * sum(1 for row in bucket if bool(row["same_pivot_commit_now"])) / n_instances
            )
            pct_same_pivot_buf = (
                100.0 * sum(1 for row in bucket if bool(row["same_pivot_buffered"])) / n_instances
            )

            summary_rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": int(k),
                    "n_instances": int(n_instances),
                    "finite_valid_pct": float(finite_valid_pct),
                    "cn_valid_pct": float(cn_valid_pct),
                    "buf_valid_pct": float(buf_valid_pct),
                    "mean_finite_score": float(mean_finite_score),
                    "mean_cn_score": float(mean_cn_score),
                    "mean_buf_score": float(mean_buf_score),
                    "mean_cn_regret": float(mean_cn_regret),
                    "mean_buf_regret": float(mean_buf_regret),
                    "mean_cn_regret_when_both_valid": float(mean_cn_regret_bv),
                    "mean_buf_regret_when_both_valid": float(mean_buf_regret_bv),
                    "pct_same_pivot_cn": float(pct_same_pivot_cn),
                    "pct_same_pivot_buf": float(pct_same_pivot_buf),
                }
            )

    return summary_rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | k | n_instances | finite_valid_pct | cn_valid_pct | buf_valid_pct | mean_finite_score | mean_cn_score | mean_buf_score | mean_cn_regret | mean_buf_regret | mean_cn_regret_when_both_valid | mean_buf_regret_when_both_valid | pct_same_pivot_cn | pct_same_pivot_buf |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | {row['n_instances']} | "
            f"{row['finite_valid_pct']:.1f} | {row['cn_valid_pct']:.1f} | {row['buf_valid_pct']:.1f} | "
            f"{row['mean_finite_score']:.3f} | {row['mean_cn_score']:.3f} | {row['mean_buf_score']:.3f} | "
            f"{row['mean_cn_regret']:.3f} | {row['mean_buf_regret']:.3f} | "
            f"{row['mean_cn_regret_when_both_valid']:.3f} | {row['mean_buf_regret_when_both_valid']:.3f} | "
            f"{row['pct_same_pivot_cn']:.1f} | {row['pct_same_pivot_buf']:.1f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_stdout_summary(records: list[dict]) -> None:
    print("=== POLICY REGRET ANALYSIS ===")
    print("")

    print("Overall regret (all instances):")
    print("  k | Commit-now mean regret | Buffered mean regret | Regret reduction")
    print("  - | --------------------- | -------------------- | ----------------")
    for k in K_VALUES:
        rows = [row for row in records if int(row["k"]) == k]
        cn_reg = _safe_mean([float(row["commit_now_regret"]) for row in rows])
        buf_reg = _safe_mean([float(row["buffered_regret"]) for row in rows])
        reduction = cn_reg - buf_reg
        print(f"  {k} | {cn_reg:21.3f} | {buf_reg:20.3f} | {reduction:16.3f}")

    print("")
    print("Regret on VALID instances only (both finite and streaming valid):")
    print("  k | CN instances | CN mean regret | Buf instances | Buf mean regret")
    print("  - | ------------ | -------------- | ------------- | ---------------")
    for k in K_VALUES:
        cn_rows = [
            row
            for row in records
            if int(row["k"]) == k and bool(row["finite_valid"]) and bool(row["commit_now_valid"])
        ]
        buf_rows = [
            row
            for row in records
            if int(row["k"]) == k and bool(row["finite_valid"]) and bool(row["buffered_valid"])
        ]
        cn_reg = _safe_mean([float(row["commit_now_regret"]) for row in cn_rows])
        buf_reg = _safe_mean([float(row["buffered_regret"]) for row in buf_rows])
        print(
            f"  {k} | {len(cn_rows):12d} | {cn_reg:14.3f} | {len(buf_rows):13d} | {buf_reg:15.3f}"
        )

    print("")
    print("Pivot agreement rates:")
    print("  k | CN same pivot as finite | Buf same pivot as finite")
    print("  - | ----------------------- | ------------------------")
    for k in K_VALUES:
        rows = [row for row in records if int(row["k"]) == k]
        n = len(rows)
        cn_same = 100.0 * sum(1 for row in rows if bool(row["same_pivot_commit_now"])) / n
        buf_same = 100.0 * sum(1 for row in rows if bool(row["same_pivot_buffered"])) / n
        print(f"  {k} | {cn_same:23.2f} | {buf_same:24.2f}")

    print("")
    print("=== INTERPRETATION ===")
    cn_bv = _safe_mean(
        [
            float(row["commit_now_regret"])
            for row in records
            if bool(row["finite_valid"]) and bool(row["commit_now_valid"])
        ]
    )
    buf_all = _safe_mean([float(row["buffered_regret"]) for row in records])
    cn_same_all = _safe_mean([1.0 if row["same_pivot_commit_now"] else 0.0 for row in records]) * 100.0
    buf_same_all = _safe_mean([1.0 if row["same_pivot_buffered"] else 0.0 for row in records]) * 100.0

    if cn_bv <= 0.05:
        print("If CN regret_when_both_valid ≈ 0: commit-now quality is fine when it doesn't trap.")
        print("  -> The case for buffered is purely trap avoidance.")
    else:
        print("If CN regret_when_both_valid > 0: commit-now produces worse extractions even when valid.")
        print("  -> The case for buffered includes quality improvement, not just trap avoidance.")

    if abs(buf_all) <= 0.05:
        print('If buffered regret ≈ 0 across all instances: buffered matches offline optimal.')
        print('  -> "Buffered commitment is essentially a free lunch."')
    else:
        print("If buffered regret > 0 on some instances: buffered is better than commit-now but still suboptimal.")
        print("  -> Motivates investigating more sophisticated policies.")

    if cn_same_all >= 95.0:
        print("Pivot agreement tells the mechanism story:")
        print("If CN and finite usually pick the same pivot, regret comes from assembly differences.")
    else:
        print("Pivot agreement tells the mechanism story:")
        print("If they pick different pivots, regret comes from pivot selection.")
    print(f"(Observed pivot agreement: CN={cn_same_all:.2f}%, BUF={buf_same_all:.2f}%)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = BurstyGenerator()
    records: list[dict] = []

    total = len(EPSILONS) * len(K_VALUES) * len(SEEDS)
    completed = 0
    for epsilon in EPSILONS:
        for k in K_VALUES:
            for seed in SEEDS:
                records.append(_run_instance(generator=generator, epsilon=epsilon, k=k, seed=seed))
                completed += 1
            print(f"Completed epsilon={epsilon:.2f}, k={k} ({completed}/{total} instances)")

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)
    _print_stdout_summary(records)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
