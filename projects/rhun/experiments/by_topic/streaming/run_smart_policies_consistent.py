"""Experiment 48c: TP-consistent forced extraction and corrected policy comparison."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph


FOCAL_ACTOR = "actor_0"
EPSILONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
K_VALUES = [1, 2, 3]
SEEDS = range(200)
N_EVENTS = 200
N_ACTORS = 6

DEFERRED_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "smart_policies_consistent_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "smart_policies_consistent_summary.md"


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
    # Policy 4 (fixed): defer commitment and freeze pivot at commit time.
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


def _tp_consistent_graph(
    graph: CausalGraph,
    focal_actor: str,
    forced_tp_id: str,
) -> CausalGraph:
    """Demote higher-weight focal events so forced TP remains focal argmax."""
    forced_tp = graph.event_by_id(forced_tp_id)
    if forced_tp is None:
        return graph

    forced_tp_weight = float(forced_tp.weight)
    demoted_actor = f"{focal_actor}__demoted"

    changed = False
    new_events = []
    for event in graph.events:
        if event.id == forced_tp_id:
            new_events.append(event)
            continue

        if focal_actor in event.actors and float(event.weight) > forced_tp_weight:
            actors = set(event.actors)
            actors.discard(focal_actor)
            actors.add(demoted_actor)
            new_events.append(replace(event, actors=frozenset(actors)))
            changed = True
        else:
            new_events.append(event)

    if not changed:
        return graph

    new_actor_set = set(graph.actors)
    new_actor_set.add(demoted_actor)

    return CausalGraph(
        events=tuple(new_events),
        actors=frozenset(new_actor_set),
        seed=graph.seed,
        metadata=dict(graph.metadata),
    )


def _forced_score_for_pivot(
    graph: CausalGraph,
    grammar: GrammarConfig,
    pivot_id: str | None,
    consistent: bool,
    cache: dict[str, tuple[bool, float]],
) -> tuple[bool, float]:
    if pivot_id is None:
        return False, 0.0
    if pivot_id in cache:
        return cache[pivot_id]

    scoring_graph = _tp_consistent_graph(graph, FOCAL_ACTOR, pivot_id) if consistent else graph
    forced = greedy_extract(
        graph=scoring_graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        override_tp_id=pivot_id,
    )
    extraction_valid = bool(forced.valid)
    score = float(forced.score) if extraction_valid else 0.0

    cache[pivot_id] = (extraction_valid, score)
    return extraction_valid, score


def _apply_scoring(
    policy: dict,
    graph: CausalGraph,
    grammar: GrammarConfig,
    finite_pivot_id: str | None,
    raw_cache: dict[str, tuple[bool, float]],
    consistent_cache: dict[str, tuple[bool, float]],
) -> dict:
    ext_valid_raw = False
    score_raw = 0.0
    ext_valid_con = False
    score_con = 0.0

    if policy["committed"] and policy["pivot_id"] is not None:
        ext_valid_raw, score_raw = _forced_score_for_pivot(
            graph=graph,
            grammar=grammar,
            pivot_id=policy["pivot_id"],
            consistent=False,
            cache=raw_cache,
        )
        ext_valid_con, score_con = _forced_score_for_pivot(
            graph=graph,
            grammar=grammar,
            pivot_id=policy["pivot_id"],
            consistent=True,
            cache=consistent_cache,
        )

    return {
        "committed": bool(policy["committed"]),
        "pivot_id": policy["pivot_id"],
        "pivot_weight": policy["pivot_weight"],
        "pivot_same_as_finite": bool(finite_pivot_id and policy["pivot_id"] == finite_pivot_id),
        "commit_step": policy["commit_step"],
        "commit_fraction": policy["commit_fraction"],
        "ext_valid_raw": bool(ext_valid_raw),
        "score_raw": float(score_raw) if ext_valid_raw else 0.0,
        "ext_valid_consistent": bool(ext_valid_con),
        "score_consistent": float(score_con) if ext_valid_con else 0.0,
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

    raw_cache: dict[str, tuple[bool, float]] = {}
    consistent_cache: dict[str, tuple[bool, float]] = {}

    cn = _apply_scoring(
        raw_cn,
        graph=graph,
        grammar=grammar,
        finite_pivot_id=finite_pivot_id,
        raw_cache=raw_cache,
        consistent_cache=consistent_cache,
    )
    naive = _apply_scoring(
        raw_naive,
        graph=graph,
        grammar=grammar,
        finite_pivot_id=finite_pivot_id,
        raw_cache=raw_cache,
        consistent_cache=consistent_cache,
    )
    fvr = _apply_scoring(
        raw_fvr,
        graph=graph,
        grammar=grammar,
        finite_pivot_id=finite_pivot_id,
        raw_cache=raw_cache,
        consistent_cache=consistent_cache,
    )

    deferred_scored: dict[str, dict] = {}
    for key, raw in raw_deferred.items():
        deferred_scored[key] = _apply_scoring(
            raw,
            graph=graph,
            grammar=grammar,
            finite_pivot_id=finite_pivot_id,
            raw_cache=raw_cache,
            consistent_cache=consistent_cache,
        )

    hybrid = _apply_scoring(
        raw_hybrid,
        graph=graph,
        grammar=grammar,
        finite_pivot_id=finite_pivot_id,
        raw_cache=raw_cache,
        consistent_cache=consistent_cache,
    )

    def regret(score: float) -> float:
        return float(finite_score - score)

    cn_streaming_valid = bool(raw_cn["valid"])
    cn_streaming_score = float(cn["score_consistent"]) if cn_streaming_valid else 0.0

    row = {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "finite_valid": finite_valid,
        "finite_score": float(finite_score),
        "finite_pivot_id": finite_pivot_id,
        "finite_pivot_weight": finite_pivot_weight,
        "cn_streaming_valid": cn_streaming_valid,
        "cn_streaming_score": cn_streaming_score,
        "cn_streaming_regret": regret(cn_streaming_score),
        "cn_committed": cn["committed"],
        "cn_ext_valid_raw": cn["ext_valid_raw"],
        "cn_score_raw": cn["score_raw"],
        "cn_regret_raw": regret(cn["score_raw"]),
        "cn_ext_valid_consistent": cn["ext_valid_consistent"],
        "cn_score_consistent": cn["score_consistent"],
        "cn_regret_consistent": regret(cn["score_consistent"]),
        "cn_pivot_id": cn["pivot_id"],
        "cn_pivot_weight": cn["pivot_weight"],
        "cn_pivot_same_as_finite": cn["pivot_same_as_finite"],
        "cn_commit_step": cn["commit_step"],
        "cn_commit_fraction": cn["commit_fraction"],
        "naive_committed": naive["committed"],
        "naive_ext_valid_raw": naive["ext_valid_raw"],
        "naive_score_raw": naive["score_raw"],
        "naive_regret_raw": regret(naive["score_raw"]),
        "naive_ext_valid_consistent": naive["ext_valid_consistent"],
        "naive_score_consistent": naive["score_consistent"],
        "naive_regret_consistent": regret(naive["score_consistent"]),
        "naive_pivot_id": naive["pivot_id"],
        "naive_pivot_weight": naive["pivot_weight"],
        "naive_pivot_same_as_finite": naive["pivot_same_as_finite"],
        "naive_commit_step": naive["commit_step"],
        "naive_commit_fraction": naive["commit_fraction"],
        "fvr_committed": fvr["committed"],
        "fvr_ext_valid_raw": fvr["ext_valid_raw"],
        "fvr_score_raw": fvr["score_raw"],
        "fvr_regret_raw": regret(fvr["score_raw"]),
        "fvr_ext_valid_consistent": fvr["ext_valid_consistent"],
        "fvr_score_consistent": fvr["score_consistent"],
        "fvr_regret_consistent": regret(fvr["score_consistent"]),
        "fvr_pivot_id": fvr["pivot_id"],
        "fvr_pivot_weight": fvr["pivot_weight"],
        "fvr_pivot_same_as_finite": fvr["pivot_same_as_finite"],
        "fvr_commit_step": fvr["commit_step"],
        "fvr_commit_fraction": fvr["commit_fraction"],
        "hybrid_ub_committed": hybrid["committed"],
        "hybrid_ub_ext_valid_raw": hybrid["ext_valid_raw"],
        "hybrid_ub_score_raw": hybrid["score_raw"],
        "hybrid_ub_regret_raw": regret(hybrid["score_raw"]),
        "hybrid_ub_ext_valid_consistent": hybrid["ext_valid_consistent"],
        "hybrid_ub_score_consistent": hybrid["score_consistent"],
        "hybrid_ub_regret_consistent": regret(hybrid["score_consistent"]),
        "hybrid_ub_pivot_id": hybrid["pivot_id"],
        "hybrid_ub_pivot_weight": hybrid["pivot_weight"],
        "hybrid_ub_pivot_same_as_finite": hybrid["pivot_same_as_finite"],
        "hybrid_ub_commit_step": hybrid["commit_step"],
        "hybrid_ub_commit_fraction": hybrid["commit_fraction"],
        "hybrid_ub_mode": raw_hybrid["mode"],
        "hybrid_ub_min_gap": None if min_gap == float("inf") else float(min_gap),
    }

    for key, scored in deferred_scored.items():
        row[f"{key}_committed"] = scored["committed"]
        row[f"{key}_ext_valid_raw"] = scored["ext_valid_raw"]
        row[f"{key}_score_raw"] = scored["score_raw"]
        row[f"{key}_regret_raw"] = regret(scored["score_raw"])
        row[f"{key}_ext_valid_consistent"] = scored["ext_valid_consistent"]
        row[f"{key}_score_consistent"] = scored["score_consistent"]
        row[f"{key}_regret_consistent"] = regret(scored["score_consistent"])
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
                "finite_score": _safe_mean([float(row["finite_score"]) for row in bucket]),
                "cn_streaming_valid": 100.0 * sum(1 for row in bucket if row["cn_streaming_valid"]) / n,
                "cn_streaming_score": _safe_mean([float(row["cn_streaming_score"]) for row in bucket]),
                "cn_committed": 100.0 * sum(1 for row in bucket if row["cn_committed"]) / n,
                "cn_ext_valid_raw": 100.0 * sum(1 for row in bucket if row["cn_ext_valid_raw"]) / n,
                "cn_score_raw": _safe_mean([float(row["cn_score_raw"]) for row in bucket]),
                "cn_ext_valid_consistent": 100.0 * sum(1 for row in bucket if row["cn_ext_valid_consistent"]) / n,
                "cn_score_consistent": _safe_mean([float(row["cn_score_consistent"]) for row in bucket]),
                "naive_committed": 100.0 * sum(1 for row in bucket if row["naive_committed"]) / n,
                "naive_ext_valid_raw": 100.0 * sum(1 for row in bucket if row["naive_ext_valid_raw"]) / n,
                "naive_score_raw": _safe_mean([float(row["naive_score_raw"]) for row in bucket]),
                "naive_ext_valid_consistent": 100.0 * sum(1 for row in bucket if row["naive_ext_valid_consistent"]) / n,
                "naive_score_consistent": _safe_mean([float(row["naive_score_consistent"]) for row in bucket]),
                "fvr_committed": 100.0 * sum(1 for row in bucket if row["fvr_committed"]) / n,
                "fvr_ext_valid_raw": 100.0 * sum(1 for row in bucket if row["fvr_ext_valid_raw"]) / n,
                "fvr_score_raw": _safe_mean([float(row["fvr_score_raw"]) for row in bucket]),
                "fvr_ext_valid_consistent": 100.0 * sum(1 for row in bucket if row["fvr_ext_valid_consistent"]) / n,
                "fvr_score_consistent": _safe_mean([float(row["fvr_score_consistent"]) for row in bucket]),
                "hybrid_ub_committed": 100.0 * sum(1 for row in bucket if row["hybrid_ub_committed"]) / n,
                "hybrid_ub_ext_valid_raw": 100.0 * sum(1 for row in bucket if row["hybrid_ub_ext_valid_raw"]) / n,
                "hybrid_ub_score_raw": _safe_mean([float(row["hybrid_ub_score_raw"]) for row in bucket]),
                "hybrid_ub_ext_valid_consistent": 100.0
                * sum(1 for row in bucket if row["hybrid_ub_ext_valid_consistent"])
                / n,
                "hybrid_ub_score_consistent": _safe_mean([float(row["hybrid_ub_score_consistent"]) for row in bucket]),
            }

            for key in deferred_keys:
                summary_row[f"{key}_committed"] = 100.0 * sum(1 for row in bucket if row[f"{key}_committed"]) / n
                summary_row[f"{key}_ext_valid_raw"] = (
                    100.0 * sum(1 for row in bucket if row[f"{key}_ext_valid_raw"]) / n
                )
                summary_row[f"{key}_score_raw"] = _safe_mean([float(row[f"{key}_score_raw"]) for row in bucket])
                summary_row[f"{key}_ext_valid_consistent"] = (
                    100.0 * sum(1 for row in bucket if row[f"{key}_ext_valid_consistent"]) / n
                )
                summary_row[f"{key}_score_consistent"] = _safe_mean(
                    [float(row[f"{key}_score_consistent"]) for row in bucket]
                )

            rows.append(summary_row)

    return rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    deferred_keys = [_deferred_key(fraction) for fraction in DEFERRED_FRACTIONS]

    headers = [
        "epsilon",
        "k",
        "finite_valid",
        "finite_score",
        "cn_streaming_valid",
        "cn_streaming_score",
        "cn_committed",
        "cn_ext_valid_raw",
        "cn_score_raw",
        "cn_ext_valid_consistent",
        "cn_score_consistent",
        "naive_committed",
        "naive_ext_valid_raw",
        "naive_score_raw",
        "naive_ext_valid_consistent",
        "naive_score_consistent",
        "fvr_committed",
        "fvr_ext_valid_raw",
        "fvr_score_raw",
        "fvr_ext_valid_consistent",
        "fvr_score_consistent",
    ]

    for key in deferred_keys:
        headers.extend(
            [
                f"{key}_committed",
                f"{key}_ext_valid_raw",
                f"{key}_score_raw",
                f"{key}_ext_valid_consistent",
                f"{key}_score_consistent",
            ]
        )

    headers.extend(
        [
            "hybrid_ub_committed",
            "hybrid_ub_ext_valid_raw",
            "hybrid_ub_score_raw",
            "hybrid_ub_ext_valid_consistent",
            "hybrid_ub_score_consistent",
        ]
    )

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:" for _ in headers]) + "|",
    ]

    for row in summary_rows:
        values = [
            f"{row['epsilon']:.2f}",
            str(row["k"]),
            f"{row['finite_valid']:.1f}",
            f"{row['finite_score']:.3f}",
            f"{row['cn_streaming_valid']:.1f}",
            f"{row['cn_streaming_score']:.3f}",
            f"{row['cn_committed']:.1f}",
            f"{row['cn_ext_valid_raw']:.1f}",
            f"{row['cn_score_raw']:.3f}",
            f"{row['cn_ext_valid_consistent']:.1f}",
            f"{row['cn_score_consistent']:.3f}",
            f"{row['naive_committed']:.1f}",
            f"{row['naive_ext_valid_raw']:.1f}",
            f"{row['naive_score_raw']:.3f}",
            f"{row['naive_ext_valid_consistent']:.1f}",
            f"{row['naive_score_consistent']:.3f}",
            f"{row['fvr_committed']:.1f}",
            f"{row['fvr_ext_valid_raw']:.1f}",
            f"{row['fvr_score_raw']:.3f}",
            f"{row['fvr_ext_valid_consistent']:.1f}",
            f"{row['fvr_score_consistent']:.3f}",
        ]

        for key in deferred_keys:
            values.extend(
                [
                    f"{row[f'{key}_committed']:.1f}",
                    f"{row[f'{key}_ext_valid_raw']:.1f}",
                    f"{row[f'{key}_score_raw']:.3f}",
                    f"{row[f'{key}_ext_valid_consistent']:.1f}",
                    f"{row[f'{key}_score_consistent']:.3f}",
                ]
            )

        values.extend(
            [
                f"{row['hybrid_ub_committed']:.1f}",
                f"{row['hybrid_ub_ext_valid_raw']:.1f}",
                f"{row['hybrid_ub_score_raw']:.3f}",
                f"{row['hybrid_ub_ext_valid_consistent']:.1f}",
                f"{row['hybrid_ub_score_consistent']:.3f}",
            ]
        )

        lines.append("| " + " | ".join(values) + " |")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_policy_metrics(records: list[dict], prefix: str) -> dict[str, float]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    cell_committed: list[float] = []
    cell_ext_raw: list[float] = []
    cell_score_raw: list[float] = []
    cell_ext_con: list[float] = []
    cell_score_con: list[float] = []
    cell_regret_raw: list[float] = []
    cell_regret_con: list[float] = []
    cell_same_pivot: list[float] = []
    cell_commit_frac: list[float] = []

    for bucket in grouped.values():
        n = len(bucket)
        cell_committed.append(100.0 * sum(1 for row in bucket if row[f"{prefix}_committed"]) / n)
        cell_ext_raw.append(100.0 * sum(1 for row in bucket if row[f"{prefix}_ext_valid_raw"]) / n)
        cell_score_raw.append(_safe_mean([float(row[f"{prefix}_score_raw"]) for row in bucket]))
        cell_ext_con.append(100.0 * sum(1 for row in bucket if row[f"{prefix}_ext_valid_consistent"]) / n)
        cell_score_con.append(_safe_mean([float(row[f"{prefix}_score_consistent"]) for row in bucket]))
        cell_regret_raw.append(_safe_mean([float(row[f"{prefix}_regret_raw"]) for row in bucket]))
        cell_regret_con.append(_safe_mean([float(row[f"{prefix}_regret_consistent"]) for row in bucket]))
        cell_same_pivot.append(100.0 * sum(1 for row in bucket if row[f"{prefix}_pivot_same_as_finite"]) / n)
        commit_fracs = [
            float(row[f"{prefix}_commit_fraction"])
            for row in bucket
            if row[f"{prefix}_commit_fraction"] is not None
        ]
        cell_commit_frac.append(_safe_mean(commit_fracs))

    return {
        "committed_pct": _safe_mean(cell_committed),
        "ext_valid_raw_pct": _safe_mean(cell_ext_raw),
        "score_raw": _safe_mean(cell_score_raw),
        "ext_valid_con_pct": _safe_mean(cell_ext_con),
        "score_con": _safe_mean(cell_score_con),
        "regret_raw": _safe_mean(cell_regret_raw),
        "regret_con": _safe_mean(cell_regret_con),
        "same_pivot_pct": _safe_mean(cell_same_pivot),
        "mean_commit_frac": _safe_mean(cell_commit_frac),
    }


def _aggregate_finite(records: list[dict]) -> dict[str, float]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    cell_valid: list[float] = []
    cell_score: list[float] = []
    cell_pw: list[float] = []

    for bucket in grouped.values():
        n = len(bucket)
        cell_valid.append(100.0 * sum(1 for row in bucket if row["finite_valid"]) / n)
        cell_score.append(_safe_mean([float(row["finite_score"]) for row in bucket]))
        pivots = [float(row["finite_pivot_weight"]) for row in bucket if row["finite_pivot_weight"] is not None]
        cell_pw.append(_safe_mean(pivots))

    return {
        "committed_pct": 100.0,
        "ext_valid_raw_pct": _safe_mean(cell_valid),
        "score_raw": _safe_mean(cell_score),
        "ext_valid_con_pct": _safe_mean(cell_valid),
        "score_con": _safe_mean(cell_score),
        "regret_raw": 0.0,
        "regret_con": 0.0,
        "same_pivot_pct": 100.0,
        "mean_pivot_weight": _safe_mean(cell_pw),
    }


def _aggregate_cn_streaming(records: list[dict]) -> dict[str, float]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    cell_valid: list[float] = []
    cell_score: list[float] = []
    cell_regret: list[float] = []

    for bucket in grouped.values():
        n = len(bucket)
        cell_valid.append(100.0 * sum(1 for row in bucket if row["cn_streaming_valid"]) / n)
        cell_score.append(_safe_mean([float(row["cn_streaming_score"]) for row in bucket]))
        cell_regret.append(_safe_mean([float(row["cn_streaming_regret"]) for row in bucket]))

    return {
        "ext_valid_pct": _safe_mean(cell_valid),
        "score": _safe_mean(cell_score),
        "regret": _safe_mean(cell_regret),
    }


def _print_stdout_summary(records: list[dict]) -> None:
    finite = _aggregate_finite(records)
    cn_forced = _aggregate_policy_metrics(records, "cn")
    cn_stream = _aggregate_cn_streaming(records)
    naive = _aggregate_policy_metrics(records, "naive")
    fvr = _aggregate_policy_metrics(records, "fvr")
    deferred = {fraction: _aggregate_policy_metrics(records, _deferred_key(fraction)) for fraction in DEFERRED_FRACTIONS}
    hybrid = _aggregate_policy_metrics(records, "hybrid_ub")

    print("=== TP-CONSISTENT POLICY COMPARISON ===")
    print("")
    print("Aggregated across all epsilon (mean of cell means):")
    print("")
    print("                              |---- Raw (Exp48b) ----|--- TP-Consistent ---|")
    print("Policy                   | Committed% | ExtV_raw | Score_raw | ExtV_con | Score_con")

    def _row(label: str, metrics: dict[str, float]) -> None:
        print(
            f"{label:<24} | "
            f"{metrics['committed_pct']:10.1f} | "
            f"{metrics['ext_valid_raw_pct']:8.1f} | "
            f"{metrics['score_raw']:9.2f} | "
            f"{metrics['ext_valid_con_pct']:8.1f} | "
            f"{metrics['score_con']:9.2f}"
        )

    _row("Finite (offline)", finite)
    _row("Commit-now (forced extr)", cn_forced)
    print(
        f"{'Commit-now (streaming)':<24} | "
        f"{'':10} | "
        f"{cn_stream['ext_valid_pct']:8.1f} | "
        f"{cn_stream['score']:9.2f} | "
        f"{'':8} | {'':9}"
    )
    _row("Naive buffered", naive)
    _row("First-viable-record", fvr)
    for fraction in DEFERRED_FRACTIONS:
        _row(f"Deferred (f={fraction:.2f})", deferred[fraction])

    _row("Hybrid (retrospective UB)", hybrid)

    print("")
    print("=== IMPACT OF TP-CONSISTENT SCORING ===")
    print("")
    print("Policies most affected (largest validity gain from consistent scoring):")
    impact_rows = [
        ("Naive buffered", naive),
        ("FVR", fvr),
        ("Deferred (f=0.10)", deferred[0.10]),
    ]
    for label, metrics in impact_rows:
        delta = metrics["ext_valid_con_pct"] - metrics["ext_valid_raw_pct"]
        print(
            f"  {label:<20}: raw {metrics['ext_valid_raw_pct']:.1f}% "
            f"→ consistent {metrics['ext_valid_con_pct']:.1f}% (Δ{delta:.1f} pp)"
        )

    print("")
    print("=== CORRECTED PARETO CURVE (TP-consistent) ===")
    print("")
    print("f     | Committed% | ExtV_con | Score_con | Same Pivot% | Commit Frac")
    for fraction in DEFERRED_FRACTIONS:
        metrics = deferred[fraction]
        print(
            f"{fraction:0.2f}  | "
            f"{metrics['committed_pct']:10.1f} | "
            f"{metrics['ext_valid_con_pct']:8.1f} | "
            f"{metrics['score_con']:9.2f} | "
            f"{metrics['same_pivot_pct']:10.1f} | "
            f"{metrics['mean_commit_frac']:11.3f}"
        )

    print("")
    print("=== STREAMING-AWARE COMPARISON ===")
    print("")
    print("For the paper, the fair comparison uses:")
    print("- Commit-now: STREAMING validity and streaming-gated score")
    print("- Deferred: TP-CONSISTENT extraction validity and score")
    print("")
    print("Policy                   | Effective Valid% | Effective Score | Effective Regret")
    print(
        f"{'Commit-now (streaming)':<24} | "
        f"{cn_stream['ext_valid_pct']:16.1f} | "
        f"{cn_stream['score']:14.2f} | "
        f"{cn_stream['regret']:15.2f}"
    )
    for fraction in [0.10, 0.25, 0.50]:
        metrics = deferred[fraction]
        print(
            f"{'Deferred (f=' + format(fraction, '.2f') + ', con)':<24} | "
            f"{metrics['ext_valid_con_pct']:16.1f} | "
            f"{metrics['score_con']:14.2f} | "
            f"{metrics['regret_con']:15.2f}"
        )

    break_even = next(
        (
            fraction
            for fraction in DEFERRED_FRACTIONS
            if deferred[fraction]["score_con"] > cn_stream["score"]
            and deferred[fraction]["ext_valid_con_pct"] > cn_stream["ext_valid_pct"]
        ),
        None,
    )
    print("")
    if break_even is None:
        print("Break-even: no deferred level dominates commit-now streaming on both metrics in this sweep.")
    else:
        print(f"Break-even: deferred first dominates commit-now at f={break_even:.2f}")


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
