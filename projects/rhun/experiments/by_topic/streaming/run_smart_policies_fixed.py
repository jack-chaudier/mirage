"""Experiment 48b: corrected smart-policy comparison with deferred bug fix."""

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

DEFERRED_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "smart_policies_fixed_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "smart_policies_fixed_summary.md"


def _sorted_events(graph) -> tuple:
    return tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _deferred_key(fraction: float) -> str:
    return f"def{int(round(fraction * 100)):02d}"


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
    # Policy 4 (fixed): defer commitment and freeze the pivot at commit time.
    events = _sorted_events(graph)
    n_events = len(events)
    threshold = fraction * n_events
    events_seen: list = []

    running_max = None
    running_max_weight = float("-inf")
    committed = False
    commit_step: int | None = None
    committed_pivot = None
    committed_pivot_weight = None

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
                committed_pivot = running_max
                committed_pivot_weight = running_max_weight

    if not committed or committed_pivot is None:
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
        "pivot_id": committed_pivot.id,
        "pivot_weight": float(committed_pivot_weight),
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
    cache: dict[str, tuple[bool, float, bool]],
) -> tuple[bool, float, bool]:
    if pivot_id is None:
        return False, 0.0, False
    if pivot_id in cache:
        return cache[pivot_id]

    forced = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        override_tp_id=pivot_id,
    )
    extraction_valid = bool(forced.valid)
    score = float(forced.score) if extraction_valid else 0.0

    has_higher_focal = False
    if not extraction_valid:
        pivot_event = graph.event_by_id(pivot_id)
        if pivot_event is not None:
            pivot_weight = float(pivot_event.weight)
            has_higher_focal = any(
                FOCAL_ACTOR in event.actors
                and event.id != pivot_id
                and float(event.weight) > pivot_weight
                for event in forced.events
            )

    cache[pivot_id] = (extraction_valid, score, has_higher_focal)
    return extraction_valid, score, has_higher_focal


def _apply_scoring(
    policy: dict,
    graph,
    grammar: GrammarConfig,
    finite_pivot_id: str | None,
    score_cache: dict[str, tuple[bool, float, bool]],
) -> dict:
    extraction_valid = False
    score = 0.0
    has_higher_focal = False

    if policy["committed"] and policy["pivot_id"] is not None:
        extraction_valid, score, has_higher_focal = _forced_score_for_pivot(
            graph=graph,
            grammar=grammar,
            pivot_id=policy["pivot_id"],
            cache=score_cache,
        )

    committed = bool(policy["committed"])
    committed_but_invalid = committed and (not extraction_valid)

    return {
        "committed": committed,
        "extraction_valid": extraction_valid,
        "score": float(score) if extraction_valid else 0.0,
        "pivot_id": policy["pivot_id"],
        "pivot_weight": policy["pivot_weight"],
        "pivot_same_as_finite": bool(finite_pivot_id and policy["pivot_id"] == finite_pivot_id),
        "commit_step": policy["commit_step"],
        "commit_fraction": policy["commit_fraction"],
        "committed_but_invalid": committed_but_invalid,
        "has_higher_focal_when_invalid": committed_but_invalid and bool(has_higher_focal),
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

    raw_deferred: dict[str, dict] = {}
    for fraction in DEFERRED_FRACTIONS:
        raw_deferred[_deferred_key(fraction)] = _simulate_deferred_running_max(graph, k=k, fraction=fraction)

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

    score_cache: dict[str, tuple[bool, float, bool]] = {}

    cn = _apply_scoring(raw_cn, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)
    naive = _apply_scoring(
        raw_naive,
        graph=graph,
        grammar=grammar,
        finite_pivot_id=finite_pivot_id,
        score_cache=score_cache,
    )
    fvr = _apply_scoring(raw_fvr, graph=graph, grammar=grammar, finite_pivot_id=finite_pivot_id, score_cache=score_cache)

    deferred_scored: dict[str, dict] = {}
    for key, raw in raw_deferred.items():
        deferred_scored[key] = _apply_scoring(
            raw,
            graph=graph,
            grammar=grammar,
            finite_pivot_id=finite_pivot_id,
            score_cache=score_cache,
        )

    hybrid = _apply_scoring(
        raw_hybrid,
        graph=graph,
        grammar=grammar,
        finite_pivot_id=finite_pivot_id,
        score_cache=score_cache,
    )

    def regret(score: float) -> float:
        return float(finite_score - score)

    policy_results = [cn, naive, fvr, hybrid, *deferred_scored.values()]
    diag_committed_but_invalid = sum(1 for policy in policy_results if policy["committed_but_invalid"])
    diag_has_higher_focal = sum(1 for policy in policy_results if policy["has_higher_focal_when_invalid"])

    row = {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "finite_score": float(finite_score),
        "finite_pivot_id": finite_pivot_id,
        "finite_pivot_weight": finite_pivot_weight,
        "cn_committed": cn["committed"],
        "cn_valid": bool(raw_cn["valid"]),
        "cn_ext_valid": cn["extraction_valid"],
        "cn_score": cn["score"],
        "cn_regret": regret(cn["score"]),
        "cn_pivot_id": cn["pivot_id"],
        "cn_pivot_weight": cn["pivot_weight"],
        "cn_pivot_same_as_finite": cn["pivot_same_as_finite"],
        "cn_commit_step": cn["commit_step"],
        "cn_commit_fraction": cn["commit_fraction"],
        "naive_committed": naive["committed"],
        "naive_ext_valid": naive["extraction_valid"],
        "naive_score": naive["score"],
        "naive_regret": regret(naive["score"]),
        "naive_pivot_id": naive["pivot_id"],
        "naive_pivot_weight": naive["pivot_weight"],
        "naive_pivot_same_as_finite": naive["pivot_same_as_finite"],
        "naive_commit_step": naive["commit_step"],
        "naive_commit_fraction": naive["commit_fraction"],
        "fvr_committed": fvr["committed"],
        "fvr_ext_valid": fvr["extraction_valid"],
        "fvr_score": fvr["score"],
        "fvr_regret": regret(fvr["score"]),
        "fvr_pivot_id": fvr["pivot_id"],
        "fvr_pivot_weight": fvr["pivot_weight"],
        "fvr_pivot_same_as_finite": fvr["pivot_same_as_finite"],
        "fvr_commit_step": fvr["commit_step"],
        "fvr_commit_fraction": fvr["commit_fraction"],
        "hybrid_ub_committed": hybrid["committed"],
        "hybrid_ub_ext_valid": hybrid["extraction_valid"],
        "hybrid_ub_score": hybrid["score"],
        "hybrid_ub_regret": regret(hybrid["score"]),
        "hybrid_ub_pivot_id": hybrid["pivot_id"],
        "hybrid_ub_pivot_weight": hybrid["pivot_weight"],
        "hybrid_ub_pivot_same_as_finite": hybrid["pivot_same_as_finite"],
        "hybrid_ub_commit_step": hybrid["commit_step"],
        "hybrid_ub_commit_fraction": hybrid["commit_fraction"],
        "hybrid_ub_mode": raw_hybrid["mode"],
        "hybrid_ub_min_gap": None if min_gap == float("inf") else float(min_gap),
        "diag_committed_but_invalid": int(diag_committed_but_invalid),
        "diag_has_higher_focal": int(diag_has_higher_focal),
    }

    for key, scored in deferred_scored.items():
        row[f"{key}_committed"] = scored["committed"]
        row[f"{key}_ext_valid"] = scored["extraction_valid"]
        row[f"{key}_score"] = scored["score"]
        row[f"{key}_regret"] = regret(scored["score"])
        row[f"{key}_pivot_id"] = scored["pivot_id"]
        row[f"{key}_pivot_weight"] = scored["pivot_weight"]
        row[f"{key}_pivot_same_as_finite"] = scored["pivot_same_as_finite"]
        row[f"{key}_commit_step"] = scored["commit_step"]
        row[f"{key}_commit_fraction"] = scored["commit_fraction"]

    return row


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    deferred_keys = [_deferred_key(fraction) for fraction in DEFERRED_FRACTIONS]

    rows: list[dict] = []
    for epsilon in EPSILONS:
        for k in K_VALUES:
            bucket = grouped[(epsilon, k)]
            n = len(bucket)

            summary_row = {
                "epsilon": float(epsilon),
                "k": int(k),
                "finite_valid": 100.0 * sum(1 for row in bucket if row["finite_valid"]) / n,
                "cn_valid": 100.0 * sum(1 for row in bucket if row["cn_valid"]) / n,
                "naive_committed": 100.0 * sum(1 for row in bucket if row["naive_committed"]) / n,
                "naive_ext_valid": 100.0 * sum(1 for row in bucket if row["naive_ext_valid"]) / n,
                "fvr_committed": 100.0 * sum(1 for row in bucket if row["fvr_committed"]) / n,
                "fvr_ext_valid": 100.0 * sum(1 for row in bucket if row["fvr_ext_valid"]) / n,
                "hybrid_ub_ext_valid": 100.0 * sum(1 for row in bucket if row["hybrid_ub_ext_valid"]) / n,
                "finite_score": _safe_mean([float(row["finite_score"]) for row in bucket]),
                "cn_score": _safe_mean([float(row["cn_score"]) for row in bucket]),
                "naive_score": _safe_mean([float(row["naive_score"]) for row in bucket]),
                "fvr_score": _safe_mean([float(row["fvr_score"]) for row in bucket]),
                "hybrid_ub_score": _safe_mean([float(row["hybrid_ub_score"]) for row in bucket]),
            }

            for key in deferred_keys:
                summary_row[f"{key}_committed"] = 100.0 * sum(1 for row in bucket if row[f"{key}_committed"]) / n
                summary_row[f"{key}_ext_valid"] = 100.0 * sum(1 for row in bucket if row[f"{key}_ext_valid"]) / n
                summary_row[f"{key}_score"] = _safe_mean([float(row[f"{key}_score"]) for row in bucket])

            rows.append(summary_row)

    return rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    deferred_keys = [_deferred_key(fraction) for fraction in DEFERRED_FRACTIONS]

    headers = [
        "epsilon",
        "k",
        "finite_valid",
        "cn_valid",
        "naive_committed",
        "naive_ext_valid",
        "fvr_committed",
        "fvr_ext_valid",
    ]
    for key in deferred_keys:
        headers.extend([f"{key}_committed", f"{key}_ext_valid"])
    headers.extend(
        [
            "hybrid_ub_ext_valid",
            "finite_score",
            "cn_score",
            "naive_score",
            "fvr_score",
        ]
    )
    for key in deferred_keys:
        headers.append(f"{key}_score")
    headers.append("hybrid_ub_score")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:" for _ in headers]) + "|",
    ]

    for row in summary_rows:
        values = [
            f"{row['epsilon']:.2f}",
            str(row["k"]),
            f"{row['finite_valid']:.1f}",
            f"{row['cn_valid']:.1f}",
            f"{row['naive_committed']:.1f}",
            f"{row['naive_ext_valid']:.1f}",
            f"{row['fvr_committed']:.1f}",
            f"{row['fvr_ext_valid']:.1f}",
        ]
        for key in deferred_keys:
            values.append(f"{row[f'{key}_committed']:.1f}")
            values.append(f"{row[f'{key}_ext_valid']:.1f}")

        values.extend(
            [
                f"{row['hybrid_ub_ext_valid']:.1f}",
                f"{row['finite_score']:.3f}",
                f"{row['cn_score']:.3f}",
                f"{row['naive_score']:.3f}",
                f"{row['fvr_score']:.3f}",
            ]
        )

        for key in deferred_keys:
            values.append(f"{row[f'{key}_score']:.3f}")

        values.append(f"{row['hybrid_ub_score']:.3f}")

        lines.append("| " + " | ".join(values) + " |")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_policy_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    deferred_keys = [_deferred_key(fraction) for fraction in DEFERRED_FRACTIONS]
    policy_defs = [
        ("finite", "Finite (offline)"),
        ("cn", "Commit-now"),
        ("naive", "Naive buffered"),
        ("fvr", "First-viable-record"),
    ]
    policy_defs.extend((key, f"Deferred (f={fraction:.2f})") for key, fraction in zip(deferred_keys, DEFERRED_FRACTIONS))
    policy_defs.append(("hybrid_ub", "Hybrid (retrospective UB)"))

    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    aggregated: dict[str, dict[str, float]] = {}
    for key, label in policy_defs:
        cell_committed: list[float] = []
        cell_ext_valid: list[float] = []
        cell_score: list[float] = []
        cell_regret: list[float] = []
        cell_pivot_weight: list[float] = []
        cell_same_pivot: list[float] = []
        cell_commit_frac: list[float] = []

        for bucket in grouped.values():
            n = len(bucket)

            if key == "finite":
                committed_pct = 100.0
                ext_valid_pct = 100.0 * sum(1 for row in bucket if row["finite_valid"]) / n
                mean_score = _safe_mean([float(row["finite_score"]) for row in bucket])
                mean_regret = 0.0
                pivot_weights = [
                    float(row["finite_pivot_weight"])
                    for row in bucket
                    if row["finite_pivot_weight"] is not None
                ]
                same_pivot_pct = 100.0
                commit_fracs = [0.0]
            else:
                committed_pct = 100.0 * sum(1 for row in bucket if row[f"{key}_committed"]) / n
                ext_valid_pct = 100.0 * sum(1 for row in bucket if row[f"{key}_ext_valid"]) / n
                mean_score = _safe_mean([float(row[f"{key}_score"]) for row in bucket])
                mean_regret = _safe_mean([float(row[f"{key}_regret"]) for row in bucket])
                pivot_weights = [
                    float(row[f"{key}_pivot_weight"])
                    for row in bucket
                    if row[f"{key}_pivot_weight"] is not None
                ]
                same_pivot_pct = 100.0 * sum(1 for row in bucket if row[f"{key}_pivot_same_as_finite"]) / n
                commit_fracs = [
                    float(row[f"{key}_commit_fraction"])
                    for row in bucket
                    if row[f"{key}_commit_fraction"] is not None
                ]

            cell_committed.append(committed_pct)
            cell_ext_valid.append(ext_valid_pct)
            cell_score.append(mean_score)
            cell_regret.append(mean_regret)
            cell_pivot_weight.append(_safe_mean(pivot_weights))
            cell_same_pivot.append(same_pivot_pct)
            cell_commit_frac.append(_safe_mean(commit_fracs))

        aggregated[label] = {
            "committed_pct": _safe_mean(cell_committed),
            "ext_valid_pct": _safe_mean(cell_ext_valid),
            "mean_score": _safe_mean(cell_score),
            "mean_regret": _safe_mean(cell_regret),
            "mean_pivot_weight": _safe_mean(cell_pivot_weight),
            "same_pivot_pct": _safe_mean(cell_same_pivot),
            "mean_commit_frac": _safe_mean(cell_commit_frac),
        }

    return aggregated


def _print_stdout_summary(records: list[dict]) -> None:
    aggregated = _aggregate_policy_metrics(records)

    print("=== CORRECTED SMART POLICY COMPARISON ===")
    print("")
    print("Aggregated across all epsilon (mean of cell means):")
    print("")
    print("Policy                   | Committed% | Extr.Valid% | Mean Score | Mean Regret | Pivot Wt | Same Pivot%")

    ordered = [
        "Finite (offline)",
        "Commit-now",
        "Naive buffered",
        "First-viable-record",
    ]
    ordered.extend([f"Deferred (f={fraction:.2f})" for fraction in DEFERRED_FRACTIONS])
    ordered.append("Hybrid (retrospective UB)")

    for label in ordered:
        metrics = aggregated[label]
        print(
            f"{label:<24} | "
            f"{metrics['committed_pct']:10.1f} | "
            f"{metrics['ext_valid_pct']:10.1f} | "
            f"{metrics['mean_score']:10.2f} | "
            f"{metrics['mean_regret']:10.2f} | "
            f"{metrics['mean_pivot_weight']:8.3f} | "
            f"{metrics['same_pivot_pct']:10.1f}"
        )

    print("")
    print("=== DEFERRED SWEEP DETAIL ===")
    print("")
    print("Shows how each metric evolves with patience:")
    print("")
    print("f     | Committed% | Extr.Valid% | Mean Score | Same Pivot% | Mean Commit Frac")
    for fraction in DEFERRED_FRACTIONS:
        label = f"Deferred (f={fraction:.2f})"
        metrics = aggregated[label]
        print(
            f"{fraction:0.2f}  | "
            f"{metrics['committed_pct']:10.1f} | "
            f"{metrics['ext_valid_pct']:10.1f} | "
            f"{metrics['mean_score']:10.2f} | "
            f"{metrics['same_pivot_pct']:10.1f} | "
            f"{metrics['mean_commit_frac']:15.3f}"
        )

    cn_score = aggregated["Commit-now"]["mean_score"]
    cn_valid = aggregated["Commit-now"]["ext_valid_pct"]

    deferred_entries = [
        (fraction, aggregated[f"Deferred (f={fraction:.2f})"]) for fraction in DEFERRED_FRACTIONS
    ]
    first_score_exceed = next((entry for entry in deferred_entries if entry[1]["mean_score"] > cn_score), None)
    first_break_even = next(
        (
            entry
            for entry in deferred_entries
            if entry[1]["mean_score"] > cn_score and entry[1]["ext_valid_pct"] > cn_valid
        ),
        None,
    )

    print("")
    print("=== KEY TRADEOFF ===")
    print("")
    if first_score_exceed is None:
        print(f"The operating point where score exceeds commit-now's mean score ({cn_score:.2f}):")
        print("  No deferred setting exceeds commit-now mean score on this grid.")
    else:
        frac, metrics = first_score_exceed
        print(f"The operating point where score exceeds commit-now's mean score ({cn_score:.2f}):")
        print(
            f"  Deferred (f={frac:.2f}) achieves mean score {metrics['mean_score']:.2f} "
            f"at {metrics['ext_valid_pct']:.1f}% extraction validity."
        )

    if first_break_even is None:
        print("\nThe \"break-even\" patience level where deferred first dominates commit-now:")
        print("  No deferred setting beats commit-now on both score and extraction validity in this sweep.")
    else:
        frac, metrics = first_break_even
        print("\nThe \"break-even\" patience level where deferred first dominates commit-now:")
        print(
            f"  f={frac:.2f} (score {metrics['mean_score']:.2f} > cn score {cn_score:.2f}, "
            f"ext_valid {metrics['ext_valid_pct']:.1f}% > cn valid {cn_valid:.1f}%)"
        )

    total_committed_invalid = sum(int(row["diag_committed_but_invalid"]) for row in records)
    total_has_higher = sum(int(row["diag_has_higher_focal"]) for row in records)
    total_no_higher = total_committed_invalid - total_has_higher
    pct_higher = 100.0 * total_has_higher / total_committed_invalid if total_committed_invalid > 0 else 0.0
    pct_no_higher = 100.0 * total_no_higher / total_committed_invalid if total_committed_invalid > 0 else 0.0

    print("")
    print("=== OVERRIDE_TP_ID DIAGNOSTIC ===")
    print(f"Instances committed but extraction invalid: {total_committed_invalid}")
    print(
        "  Of which: forced sequence contains higher-weight focal event: "
        f"{total_has_higher} ({pct_higher:.1f}%)"
    )
    print(
        "  Of which: forced sequence does NOT contain higher-weight focal: "
        f"{total_no_higher} ({pct_no_higher:.1f}%)"
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
