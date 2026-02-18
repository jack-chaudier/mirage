"""Experiment 48: comparison of max-weight-aware streaming policies."""

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

DEFERRED_FRACTIONS = [0.10, 0.25, 0.50, 0.75]

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "smart_policies_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "smart_policies_summary.md"


def _sorted_events(graph) -> tuple:
    return tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _n_predecessors(events_seen: list, event) -> int:
    event_ts = float(event.timestamp)
    return sum(1 for seen in events_seen if float(seen.timestamp) < event_ts and seen.id != event.id)


def _simulate_commit_now(graph, k: int) -> dict:
    # Reused from experiments 43-47.
    events = _sorted_events(graph)
    n_events = len(events)
    event_timestamps = {event.id: float(event.timestamp) for event in events}

    current_pivot = None
    current_pivot_weight = float("-inf")
    committed_labels: dict[str, str] = {}
    tp_committed = False
    first_commit_step: int | None = None
    record_steps: list[int] = []

    for step, event in enumerate(events):
        focal = FOCAL_ACTOR in event.actors
        if focal and float(event.weight) > current_pivot_weight:
            if first_commit_step is None:
                first_commit_step = step
            current_pivot = event
            current_pivot_weight = float(event.weight)
            committed_labels[event.id] = "TURNING_POINT"
            tp_committed = True
            record_steps.append(step)
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
            "committed": False,
            "valid": False,
            "pivot_id": None,
            "pivot_weight": None,
            "commit_step": None,
            "commit_fraction": None,
            "record_steps": record_steps,
        }

    final_dev_count = sum(
        1
        for event_id, label in committed_labels.items()
        if label == "DEVELOPMENT" and event_timestamps[event_id] < float(current_pivot.timestamp)
    )
    commit_now_valid = bool(final_dev_count >= k)
    commit_fraction = float(first_commit_step / n_events) if n_events > 0 and first_commit_step is not None else None
    return {
        "committed": True,
        "valid": commit_now_valid,
        "pivot_id": current_pivot.id,
        "pivot_weight": float(current_pivot.weight),
        "commit_step": first_commit_step,
        "commit_fraction": commit_fraction,
        "record_steps": record_steps,
    }


def _simulate_naive_buffered(graph, k: int) -> dict:
    # First viable focal event (ignores weight records).
    events = _sorted_events(graph)
    n_events = len(events)
    events_seen: list = []

    committed = False
    commit_step: int | None = None
    pivot = None

    for step, event in enumerate(events):
        events_seen.append(event)
        if committed:
            continue
        if FOCAL_ACTOR not in event.actors:
            continue

        if _n_predecessors(events_seen, event) >= k:
            committed = True
            commit_step = step
            pivot = event

    if not committed or pivot is None:
        return {
            "committed": False,
            "valid": False,
            "pivot_id": None,
            "pivot_weight": None,
            "commit_step": None,
            "commit_fraction": None,
        }

    commit_fraction = float(commit_step / n_events) if n_events > 0 and commit_step is not None else None
    return {
        "committed": True,
        "valid": True,
        "pivot_id": pivot.id,
        "pivot_weight": float(pivot.weight),
        "commit_step": commit_step,
        "commit_fraction": commit_fraction,
    }


def _simulate_fvr_buffered(graph, k: int) -> dict:
    # Policy 3: first viable record-setting focal event.
    events = _sorted_events(graph)
    n_events = len(events)
    events_seen: list = []

    running_max = None
    running_max_weight = float("-inf")
    committed = False
    commit_step: int | None = None

    for step, event in enumerate(events):
        events_seen.append(event)
        if committed:
            continue

        focal = FOCAL_ACTOR in event.actors
        if focal and float(event.weight) > running_max_weight:
            running_max = event
            running_max_weight = float(event.weight)

            if _n_predecessors(events_seen, event) >= k:
                committed = True
                commit_step = step

    if not committed or running_max is None:
        return {
            "committed": False,
            "valid": False,
            "pivot_id": None,
            "pivot_weight": None,
            "commit_step": None,
            "commit_fraction": None,
        }

    commit_fraction = float(commit_step / n_events) if n_events > 0 and commit_step is not None else None
    return {
        "committed": True,
        "valid": True,
        "pivot_id": running_max.id,
        "pivot_weight": float(running_max.weight),
        "commit_step": commit_step,
        "commit_fraction": commit_fraction,
    }


def _simulate_deferred_running_max(graph, k: int, fraction: float) -> dict:
    # Policy 4: defer commitment until a minimum observation window is met.
    events = _sorted_events(graph)
    n_events = len(events)
    threshold = fraction * n_events
    events_seen: list = []

    running_max = None
    running_max_weight = float("-inf")
    committed = False
    commit_step: int | None = None

    for step, event in enumerate(events):
        events_seen.append(event)
        focal = FOCAL_ACTOR in event.actors
        if focal and float(event.weight) > running_max_weight:
            running_max = event
            running_max_weight = float(event.weight)

        if committed:
            continue

        if step >= threshold and running_max is not None:
            if _n_predecessors(events_seen, running_max) >= k:
                committed = True
                commit_step = step

    if not committed or running_max is None:
        return {
            "committed": False,
            "valid": False,
            "pivot_id": None,
            "pivot_weight": None,
            "commit_step": None,
            "commit_fraction": None,
        }

    commit_fraction = float(commit_step / n_events) if n_events > 0 and commit_step is not None else None
    return {
        "committed": True,
        "valid": True,
        "pivot_id": running_max.id,
        "pivot_weight": float(running_max.weight),
        "commit_step": commit_step,
        "commit_fraction": commit_fraction,
    }


def _min_record_gap(record_steps: list[int]) -> float:
    if len(record_steps) < 2:
        return float("inf")
    return float(min(record_steps[idx] - record_steps[idx - 1] for idx in range(1, len(record_steps))))


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


def _apply_scoring(
    policy: dict,
    graph,
    grammar: GrammarConfig,
    finite_pivot_id: str | None,
    score_cache: dict[str, tuple[bool, float]],
) -> dict:
    if policy["valid"] and policy["pivot_id"] is not None:
        _, score = _forced_score_for_pivot(
            graph=graph,
            grammar=grammar,
            pivot_id=policy["pivot_id"],
            cache=score_cache,
        )
    else:
        score = 0.0

    pivot_same = bool(finite_pivot_id is not None and policy["pivot_id"] == finite_pivot_id)
    return {
        "committed": bool(policy["committed"]),
        "valid": bool(policy["valid"]),
        "score": float(score),
        "pivot_id": policy["pivot_id"],
        "pivot_weight": policy["pivot_weight"],
        "pivot_same_as_finite": pivot_same,
        "commit_step": policy["commit_step"],
        "commit_fraction": policy["commit_fraction"],
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
    grammar = GrammarConfig(min_prefix_elements=k)

    finite = greedy_extract(graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar)
    finite_valid = bool(finite.valid)
    finite_score = float(finite.score) if finite_valid else 0.0
    finite_tp = finite.turning_point
    finite_pivot_id = None if finite_tp is None else finite_tp.id
    finite_pivot_weight = None if finite_tp is None else float(finite_tp.weight)

    raw_cn = _simulate_commit_now(graph, k=k)
    raw_naive = _simulate_naive_buffered(graph, k=k)
    raw_fvr = _simulate_fvr_buffered(graph, k=k)
    raw_def10 = _simulate_deferred_running_max(graph, k=k, fraction=0.10)
    raw_def25 = _simulate_deferred_running_max(graph, k=k, fraction=0.25)
    raw_def50 = _simulate_deferred_running_max(graph, k=k, fraction=0.50)
    raw_def75 = _simulate_deferred_running_max(graph, k=k, fraction=0.75)

    min_gap = _min_record_gap(raw_cn["record_steps"])
    if min_gap >= float(k):
        raw_hybrid = {
            **raw_cn,
            "mode": "commit-now",
        }
    else:
        raw_hybrid = {
            **raw_fvr,
            "mode": "buffered-fallback",
        }

    score_cache: dict[str, tuple[bool, float]] = {}

    cn = _apply_scoring(raw_cn, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    naive = _apply_scoring(raw_naive, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    fvr = _apply_scoring(raw_fvr, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    def10 = _apply_scoring(raw_def10, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    def25 = _apply_scoring(raw_def25, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    def50 = _apply_scoring(raw_def50, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    def75 = _apply_scoring(raw_def75, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    hybrid = _apply_scoring(raw_hybrid, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)

    def regret(score: float) -> float:
        return float(finite_score - score)

    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "finite_score": float(finite_score),
        "finite_pivot_id": finite_pivot_id,
        "finite_pivot_weight": finite_pivot_weight,
        "cn_committed": cn["committed"],
        "cn_valid": cn["valid"],
        "cn_score": cn["score"],
        "cn_regret": regret(cn["score"]),
        "cn_pivot_id": cn["pivot_id"],
        "cn_pivot_weight": cn["pivot_weight"],
        "cn_pivot_same_as_finite": cn["pivot_same_as_finite"],
        "cn_commit_step": cn["commit_step"],
        "cn_commit_fraction": cn["commit_fraction"],
        "naive_committed": naive["committed"],
        "naive_valid": naive["valid"],
        "naive_score": naive["score"],
        "naive_regret": regret(naive["score"]),
        "naive_pivot_id": naive["pivot_id"],
        "naive_pivot_weight": naive["pivot_weight"],
        "naive_pivot_same_as_finite": naive["pivot_same_as_finite"],
        "naive_commit_step": naive["commit_step"],
        "naive_commit_fraction": naive["commit_fraction"],
        "fvr_committed": fvr["committed"],
        "fvr_valid": fvr["valid"],
        "fvr_score": fvr["score"],
        "fvr_regret": regret(fvr["score"]),
        "fvr_pivot_id": fvr["pivot_id"],
        "fvr_pivot_weight": fvr["pivot_weight"],
        "fvr_pivot_same_as_finite": fvr["pivot_same_as_finite"],
        "fvr_commit_step": fvr["commit_step"],
        "fvr_commit_fraction": fvr["commit_fraction"],
        "def10_committed": def10["committed"],
        "def10_valid": def10["valid"],
        "def10_score": def10["score"],
        "def10_regret": regret(def10["score"]),
        "def10_pivot_id": def10["pivot_id"],
        "def10_pivot_weight": def10["pivot_weight"],
        "def10_pivot_same_as_finite": def10["pivot_same_as_finite"],
        "def10_commit_step": def10["commit_step"],
        "def10_commit_fraction": def10["commit_fraction"],
        "def25_committed": def25["committed"],
        "def25_valid": def25["valid"],
        "def25_score": def25["score"],
        "def25_regret": regret(def25["score"]),
        "def25_pivot_id": def25["pivot_id"],
        "def25_pivot_weight": def25["pivot_weight"],
        "def25_pivot_same_as_finite": def25["pivot_same_as_finite"],
        "def25_commit_step": def25["commit_step"],
        "def25_commit_fraction": def25["commit_fraction"],
        "def50_committed": def50["committed"],
        "def50_valid": def50["valid"],
        "def50_score": def50["score"],
        "def50_regret": regret(def50["score"]),
        "def50_pivot_id": def50["pivot_id"],
        "def50_pivot_weight": def50["pivot_weight"],
        "def50_pivot_same_as_finite": def50["pivot_same_as_finite"],
        "def50_commit_step": def50["commit_step"],
        "def50_commit_fraction": def50["commit_fraction"],
        "def75_committed": def75["committed"],
        "def75_valid": def75["valid"],
        "def75_score": def75["score"],
        "def75_regret": regret(def75["score"]),
        "def75_pivot_id": def75["pivot_id"],
        "def75_pivot_weight": def75["pivot_weight"],
        "def75_pivot_same_as_finite": def75["pivot_same_as_finite"],
        "def75_commit_step": def75["commit_step"],
        "def75_commit_fraction": def75["commit_fraction"],
        "hybrid_ub_committed": hybrid["committed"],
        "hybrid_ub_valid": hybrid["valid"],
        "hybrid_ub_score": hybrid["score"],
        "hybrid_ub_regret": regret(hybrid["score"]),
        "hybrid_ub_pivot_id": hybrid["pivot_id"],
        "hybrid_ub_pivot_weight": hybrid["pivot_weight"],
        "hybrid_ub_pivot_same_as_finite": hybrid["pivot_same_as_finite"],
        "hybrid_ub_commit_step": hybrid["commit_step"],
        "hybrid_ub_commit_fraction": hybrid["commit_fraction"],
        "hybrid_ub_mode": raw_hybrid["mode"],
        "hybrid_ub_min_gap": None if min_gap == float("inf") else float(min_gap),
    }


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    rows: list[dict] = []
    for epsilon in EPSILONS:
        for k in K_VALUES:
            bucket = grouped[(epsilon, k)]
            n = len(bucket)
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": int(k),
                    "finite_valid": 100.0 * sum(1 for row in bucket if row["finite_valid"]) / n,
                    "cn_valid": 100.0 * sum(1 for row in bucket if row["cn_valid"]) / n,
                    "naive_valid": 100.0 * sum(1 for row in bucket if row["naive_valid"]) / n,
                    "fvr_valid": 100.0 * sum(1 for row in bucket if row["fvr_valid"]) / n,
                    "def25_valid": 100.0 * sum(1 for row in bucket if row["def25_valid"]) / n,
                    "def50_valid": 100.0 * sum(1 for row in bucket if row["def50_valid"]) / n,
                    "def75_valid": 100.0 * sum(1 for row in bucket if row["def75_valid"]) / n,
                    "hybrid_ub_valid": 100.0 * sum(1 for row in bucket if row["hybrid_ub_valid"]) / n,
                    "finite_score": _safe_mean([float(row["finite_score"]) for row in bucket]),
                    "cn_score": _safe_mean([float(row["cn_score"]) for row in bucket]),
                    "naive_score": _safe_mean([float(row["naive_score"]) for row in bucket]),
                    "fvr_score": _safe_mean([float(row["fvr_score"]) for row in bucket]),
                    "def25_score": _safe_mean([float(row["def25_score"]) for row in bucket]),
                    "def50_score": _safe_mean([float(row["def50_score"]) for row in bucket]),
                    "def75_score": _safe_mean([float(row["def75_score"]) for row in bucket]),
                    "hybrid_ub_score": _safe_mean([float(row["hybrid_ub_score"]) for row in bucket]),
                }
            )
    return rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | k | finite_valid | cn_valid | naive_valid | fvr_valid | def25_valid | def50_valid | def75_valid | hybrid_ub_valid | finite_score | cn_score | naive_score | fvr_score | def25_score | def50_score | def75_score | hybrid_ub_score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | "
            f"{row['finite_valid']:.1f} | {row['cn_valid']:.1f} | {row['naive_valid']:.1f} | {row['fvr_valid']:.1f} | "
            f"{row['def25_valid']:.1f} | {row['def50_valid']:.1f} | {row['def75_valid']:.1f} | {row['hybrid_ub_valid']:.1f} | "
            f"{row['finite_score']:.3f} | {row['cn_score']:.3f} | {row['naive_score']:.3f} | {row['fvr_score']:.3f} | "
            f"{row['def25_score']:.3f} | {row['def50_score']:.3f} | {row['def75_score']:.3f} | {row['hybrid_ub_score']:.3f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_policy_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    policy_defs = [
        ("finite", "Finite (offline)"),
        ("cn", "Commit-now"),
        ("naive", "Naive buffered"),
        ("fvr", "First-viable-record buf"),
        ("def10", "Deferred (f=0.10)"),
        ("def25", "Deferred (f=0.25)"),
        ("def50", "Deferred (f=0.50)"),
        ("def75", "Deferred (f=0.75)"),
        ("hybrid_ub", "Hybrid (retrospective UB)"),
    ]

    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    aggregated: dict[str, dict[str, float]] = {}
    for key, label in policy_defs:
        cell_valid: list[float] = []
        cell_score: list[float] = []
        cell_regret: list[float] = []
        cell_pivot_weight: list[float] = []

        for bucket in grouped.values():
            n = len(bucket)
            valid_pct = 100.0 * sum(1 for row in bucket if bool(row[f"{key}_valid"])) / n
            mean_score = _safe_mean([float(row[f"{key}_score"]) for row in bucket])
            mean_regret = _safe_mean([float(row[f"{key}_regret"]) for row in bucket]) if key != "finite" else 0.0
            pivots = [
                float(row[f"{key}_pivot_weight"])
                for row in bucket
                if row[f"{key}_pivot_weight"] is not None
            ]
            mean_pivot_weight = _safe_mean(pivots)

            cell_valid.append(valid_pct)
            cell_score.append(mean_score)
            cell_regret.append(mean_regret)
            cell_pivot_weight.append(mean_pivot_weight)

        aggregated[label] = {
            "valid_pct": _safe_mean(cell_valid),
            "mean_score": _safe_mean(cell_score),
            "mean_regret": _safe_mean(cell_regret),
            "mean_pivot_weight": _safe_mean(cell_pivot_weight),
        }

    return aggregated


def _print_stdout_summary(records: list[dict]) -> None:
    aggregated = _aggregate_policy_metrics(records)

    print("=== SMART POLICY COMPARISON ===")
    print("")
    print("Aggregated across all epsilon (mean of cell means):")
    print("")
    print("Policy                          | Valid%  | Mean Score | Mean Regret | Mean Pivot Weight")
    print("------------------------------- | ------: | ---------: | ----------: | ----------------:")

    ordered = [
        "Finite (offline)",
        "Commit-now",
        "Naive buffered",
        "First-viable-record buf",
        "Deferred (f=0.10)",
        "Deferred (f=0.25)",
        "Deferred (f=0.50)",
        "Deferred (f=0.75)",
        "Hybrid (retrospective UB)",
    ]
    for label in ordered:
        metrics = aggregated[label]
        print(
            f"{label:<31} | "
            f"{metrics['valid_pct']:6.1f} | "
            f"{metrics['mean_score']:10.2f} | "
            f"{metrics['mean_regret']:10.2f} | "
            f"{metrics['mean_pivot_weight']:16.3f}"
        )

    print("")
    naive_regret = aggregated["Naive buffered"]["mean_regret"]
    fvr_regret = aggregated["First-viable-record buf"]["mean_regret"]
    recovery = ((naive_regret - fvr_regret) / naive_regret * 100.0) if naive_regret > 0 else 0.0
    best_deferred = min(
        [
            ("f=0.10", aggregated["Deferred (f=0.10)"]["mean_regret"]),
            ("f=0.25", aggregated["Deferred (f=0.25)"]["mean_regret"]),
            ("f=0.50", aggregated["Deferred (f=0.50)"]["mean_regret"]),
            ("f=0.75", aggregated["Deferred (f=0.75)"]["mean_regret"]),
        ],
        key=lambda item: item[1],
    )
    best_policy = min(ordered[1:], key=lambda label: aggregated[label]["mean_regret"])

    print("Key questions answered:")
    print(
        "1. Does first-viable-record improve over naive buffered in pivot weight? "
        f"(Observed: {'YES' if aggregated['First-viable-record buf']['mean_pivot_weight'] > aggregated['Naive buffered']['mean_pivot_weight'] else 'NO'})"
    )
    print(
        "2. Does first-viable-record close most of the regret gap? "
        f"(Observed recovery vs naive: {recovery:.1f}% of naive regret gap)"
    )
    print(
        "3. Do deferred variants close the gap? At what patience level? "
        f"(Best deferred: {best_deferred[0]} with mean regret {best_deferred[1]:.2f})"
    )
    print(
        "4. Does hybrid (retro UB) beat everything? "
        f"(Best non-finite policy by regret: {best_policy})"
    )

    print("")
    print("=== HYBRID (RETROSPECTIVE UB) BREAKDOWN ===")
    n_total = len(records)
    mode_cn = [row for row in records if row["hybrid_ub_mode"] == "commit-now"]
    mode_fb = [row for row in records if row["hybrid_ub_mode"] == "buffered-fallback"]
    pct_mode_cn = 100.0 * len(mode_cn) / n_total
    pct_mode_fb = 100.0 * len(mode_fb) / n_total
    valid_cn = 100.0 * sum(1 for row in mode_cn if row["hybrid_ub_valid"]) / len(mode_cn) if mode_cn else 0.0
    valid_fb = 100.0 * sum(1 for row in mode_fb if row["hybrid_ub_valid"]) / len(mode_fb) if mode_fb else 0.0
    print(f"Hybrid used commit-now mode: {pct_mode_cn:.1f}% of instances")
    print(f"Hybrid used buffered-fallback mode: {pct_mode_fb:.1f}% of instances")
    print(f"Hybrid validity when using commit-now: {valid_cn:.1f}%")
    print(f"Hybrid validity when using buffered-fallback: {valid_fb:.1f}%")

    print("")
    print("=== FIRST-VIABLE-RECORD BUFFERED DETAILS ===")
    fvr_committed = sum(1 for row in records if row["fvr_committed"])
    fvr_same = sum(1 for row in records if row["fvr_pivot_same_as_finite"])
    fvr_commit_frac = _safe_mean(
        [float(row["fvr_commit_fraction"]) for row in records if row["fvr_commit_fraction"] is not None]
    )
    fvr_pw = _safe_mean(
        [float(row["fvr_pivot_weight"]) for row in records if row["fvr_pivot_weight"] is not None]
    )
    naive_pw = _safe_mean(
        [float(row["naive_pivot_weight"]) for row in records if row["naive_pivot_weight"] is not None]
    )
    finite_pw = _safe_mean(
        [float(row["finite_pivot_weight"]) for row in records if row["finite_pivot_weight"] is not None]
    )
    fvr_no_commit = n_total - fvr_committed
    print(f"FVR committed: {100.0 * fvr_committed / n_total:.1f}% of instances")
    print(f"FVR picked same pivot as finite: {100.0 * fvr_same / n_total:.1f}%")
    print(f"Mean commit fraction: {fvr_commit_frac:.3f}")
    print(
        f"Mean pivot weight (FVR): {fvr_pw:.3f} vs mean pivot weight (naive): {naive_pw:.3f} "
        f"vs mean pivot weight (finite): {finite_pw:.3f}"
    )
    print(f"Cases where FVR couldn't commit (global max has < k predecessors): {fvr_no_commit}")


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
