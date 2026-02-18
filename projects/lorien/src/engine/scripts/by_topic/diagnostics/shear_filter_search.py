"""Shear-filtered proto-keep injection experiment.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.shear_filter_search
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from narrativefield.extraction import arc_search as arc_search_core
from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import _Candidate, search_arc
from narrativefield.extraction.arc_validator import validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.schema.events import BeatType, DeltaKind, Event, EventType
from scripts.feasible_volume_sweep import _prepare_seed_contexts
from scripts.k_sweep_experiment import DINNER_PARTY_AGENTS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "shear_filter_experiment.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "shear_filter_summary.md"
DEFAULT_SEEDS = list(range(1, 51))

FOCAL_SEEDS = [2, 3, 9, 25, 32, 33, 35, 38, 43]
BASELINE_INVALID_SEEDS_EXPECTED = [2, 3, 9, 25, 32, 33, 35, 38, 43]
BASELINE_MEAN_Q_ALL_EXPECTED = 0.684
BASELINE_ALL_VALID_EXPECTED = 0.64

EARLY_CUTOFF = 0.20
MIDRANGE_LOWER = 0.30
MIDRANGE_UPPER = 0.70
MAX_EVENTS = 20
MAX_ANCHORS = 8
PROTO_TOPK = MAX_EVENTS * 2
EPSILON = 1e-12

ConditionName = Literal[
    "baseline",
    "temporal_filter",
    "shear_filter_median",
    "shear_filter_p25",
]
CONDITIONS: list[ConditionName] = [
    "baseline",
    "temporal_filter",
    "shear_filter_median",
    "shear_filter_p25",
]


@dataclass
class SearchOutcome:
    seed: int
    condition: str
    agent_id: str
    valid: bool
    q_score: float | None
    tp_position: float | None
    tp_event_type: str | None
    dev_beat_count: int
    total_beats: int
    event_ids: list[str]
    beats: list[str]
    violations: list[str]
    total_candidates: int
    candidates_with_dev_beats: int
    proto_keep_ids: list[str]
    proto_keep_positions: list[float]
    proto_keep_threshold: float | None
    proto_keep_shear: list[float]
    error: str | None = None


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _parse_seeds(raw: str) -> list[int]:
    text = raw.strip()
    if "," in text:
        out = [int(item.strip()) for item in text.split(",") if item.strip()]
        return sorted(set(out))
    if "-" in text:
        left, right = text.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            raise ValueError("Seed range must satisfy end >= start.")
        return list(range(start, end + 1))
    return [int(text)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _event_pos(event: Event, simulation_end_tick: int) -> float:
    return float(int(event.tick_id) / max(int(simulation_end_tick), 1))


def _dev_beat_count(beats: list[BeatType]) -> int:
    return int(sum(1 for beat in beats if beat in {BeatType.COMPLICATION, BeatType.ESCALATION}))


def _tp_info(
    *,
    events: list[Event],
    beats: list[BeatType],
    simulation_end_tick: int,
) -> tuple[float | None, str | None]:
    for event, beat in zip(events, beats):
        if beat == BeatType.TURNING_POINT:
            return (_event_pos(event, simulation_end_tick), str(event.type.value))
    return (None, None)


def _scored_anchor_pool(window: list[Event]) -> list[tuple[float, Event]]:
    scored: list[tuple[float, Event]] = []
    for event in window:
        if event.type == EventType.INTERNAL:
            continue
        if event.type in {EventType.SOCIAL_MOVE, EventType.OBSERVE} and not arc_search_core._has_meaningful_deltas(event):
            continue
        scored.append((arc_search_core._event_importance(event), event))
    scored.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))
    return scored


def _default_anchor_selection(
    *,
    scored_events: list[tuple[float, Event]],
    protagonist: str,
    max_anchors: int = MAX_ANCHORS,
) -> list[Event]:
    anchors = [event for _, event in scored_events[: max(1, min(max_anchors, len(scored_events)))]]
    proto_anchors = [event for event in anchors if arc_search_core._involves(event, protagonist)]
    if proto_anchors:
        anchors = proto_anchors
    return anchors


def _all_midrange_anchor_selection(
    *,
    scored_events: list[tuple[float, Event]],
    protagonist: str,
    simulation_end_tick: int,
    max_anchors: int = MAX_ANCHORS,
) -> list[Event]:
    protagonist_scored = [
        (importance, event)
        for importance, event in scored_events
        if arc_search_core._involves(event, protagonist)
    ]
    selection_pool = protagonist_scored if protagonist_scored else scored_events

    midrange = [
        (importance, event)
        for importance, event in selection_pool
        if MIDRANGE_LOWER <= _event_pos(event, simulation_end_tick) <= MIDRANGE_UPPER
    ]
    midrange.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))
    return [event for _, event in midrange[:max_anchors]]


def _agent_delta_relevance(event: Event, agent_id: str) -> float:
    score = 0.0
    for delta in event.deltas:
        touches_primary = delta.agent == agent_id
        touches_secondary = delta.agent_b == agent_id
        touches = touches_primary or touches_secondary
        if not touches:
            continue

        if touches_primary:
            score += 1.0
        if touches_secondary:
            score += 0.75

        if delta.kind in {
            DeltaKind.BELIEF,
            DeltaKind.RELATIONSHIP,
            DeltaKind.SECRET_STATE,
            DeltaKind.COMMITMENT,
        }:
            score += 0.5
        elif delta.kind == DeltaKind.PACING:
            score += 0.35
        else:
            score += 0.2

        value = delta.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score += 0.1 * min(abs(float(value)), 1.0)

    return score


def _agent_relative_importance(event: Event, agent_id: str) -> float:
    """Approximate event salience for a specific agent.

    Decomposition choice:
    - Start from global `_event_importance(event)`.
    - Downweight events that do not involve the agent (participant factor 0.2).
    - Re-boost when deltas explicitly touch the agent (`delta.agent` / `delta.agent_b`).
    """
    global_importance = float(arc_search_core._event_importance(event))
    participant_factor = 1.0 if arc_search_core._involves(event, agent_id) else 0.2
    delta_signal = _agent_delta_relevance(event, agent_id)
    delta_multiplier = 1.0 + 0.25 * min(delta_signal, 3.0)
    return float(global_importance * participant_factor * delta_multiplier)


def _coefficient_of_variation(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr)) if arr.size else 0.0
    if abs(mean_val) <= EPSILON:
        return 0.0
    return float(np.std(arr, ddof=0) / mean_val)


def _compute_importance_and_shear(
    events: list[Event],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    importance: dict[str, dict[str, float]] = {}
    shear_cv: dict[str, float] = {}

    for event in events:
        by_agent = {
            agent: _agent_relative_importance(event, agent)
            for agent in DINNER_PARTY_AGENTS
        }
        importance[event.id] = by_agent
        shear_cv[event.id] = _coefficient_of_variation(list(by_agent.values()))

    return (importance, shear_cv)


def _baseline_proto_keep_ids(window: list[Event], protagonist: str) -> set[str]:
    proto_events_sorted = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (-arc_search_core._event_importance(event), event.sim_time, event.id),
    )
    proto_keep_ids: set[str] = {event.id for event in proto_events_sorted[:PROTO_TOPK]}

    proto_by_time = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick),
    )
    for event in proto_by_time[:3]:
        proto_keep_ids.add(event.id)
    for event in proto_by_time[-3:]:
        proto_keep_ids.add(event.id)

    return proto_keep_ids


def _build_proto_keep_ids(
    *,
    window: list[Event],
    protagonist: str,
    condition: ConditionName,
    simulation_end_tick: int,
    importance_by_event: dict[str, dict[str, float]],
) -> tuple[set[str], list[float], float | None]:
    by_id = {event.id: event for event in window}
    baseline_ids = _baseline_proto_keep_ids(window, protagonist)

    threshold: float | None = None
    selected_ids: set[str] = set(baseline_ids)

    if condition == "temporal_filter":
        selected_ids = {
            event_id
            for event_id in baseline_ids
            if event_id in by_id and _event_pos(by_id[event_id], simulation_end_tick) >= EARLY_CUTOFF
        }
    elif condition in {"shear_filter_median", "shear_filter_p25"}:
        agent_scores = [
            float(importance_by_event[event.id][protagonist])
            for event in window
            if event.id in importance_by_event
        ]
        if agent_scores:
            if condition == "shear_filter_median":
                threshold = float(np.median(np.asarray(agent_scores, dtype=float)))
            else:
                threshold = float(np.percentile(np.asarray(agent_scores, dtype=float), 25.0))

        selected_ids = {
            event_id
            for event_id in baseline_ids
            if event_id in importance_by_event
            and float(importance_by_event[event_id][protagonist]) >= float(threshold or 0.0)
        }

        # Keep at least one bridge candidate to avoid degenerate empty injections.
        if not selected_ids and baseline_ids:
            best_id = max(
                baseline_ids,
                key=lambda event_id: (
                    float(importance_by_event.get(event_id, {}).get(protagonist, 0.0)),
                    event_id,
                ),
            )
            selected_ids = {best_id}

    selected_positions = sorted(
        _event_pos(by_id[event_id], simulation_end_tick)
        for event_id in selected_ids
        if event_id in by_id
    )
    return (selected_ids, selected_positions, threshold)


def _search_with_condition(
    *,
    seed: int,
    all_events: list[Event],
    protagonist: str,
    total_sim_time: float | None,
    simulation_end_tick: int,
    condition: ConditionName,
    importance_by_event: dict[str, dict[str, float]],
    shear_by_event: dict[str, float],
) -> SearchOutcome:
    window = list(all_events)
    window.sort(key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick))

    if not window:
        return SearchOutcome(
            seed=int(seed),
            condition=condition,
            agent_id=protagonist,
            valid=False,
            q_score=None,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            event_ids=[],
            beats=[],
            violations=["No events in time window"],
            total_candidates=0,
            candidates_with_dev_beats=0,
            proto_keep_ids=[],
            proto_keep_positions=[],
            proto_keep_threshold=None,
            proto_keep_shear=[],
        )

    by_id: dict[str, Event] = {event.id: event for event in all_events}
    reverse_links = arc_search_core._build_reverse_links(all_events)
    window_ids = {event.id for event in window}

    anchor_scored = _scored_anchor_pool(window)
    if condition == "baseline":
        anchors = _default_anchor_selection(
            scored_events=anchor_scored,
            protagonist=protagonist,
            max_anchors=MAX_ANCHORS,
        )
    else:
        # Match prior temporal-filter experiment anchor strategy.
        anchors = _all_midrange_anchor_selection(
            scored_events=anchor_scored,
            protagonist=protagonist,
            simulation_end_tick=simulation_end_tick,
            max_anchors=MAX_ANCHORS,
        )

    proto_keep_ids, proto_keep_positions, proto_keep_threshold = _build_proto_keep_ids(
        window=window,
        protagonist=protagonist,
        condition=condition,
        simulation_end_tick=simulation_end_tick,
        importance_by_event=importance_by_event,
    )

    if not anchors:
        return SearchOutcome(
            seed=int(seed),
            condition=condition,
            agent_id=protagonist,
            valid=False,
            q_score=None,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            event_ids=[],
            beats=[],
            violations=["No anchors selected"],
            total_candidates=0,
            candidates_with_dev_beats=0,
            proto_keep_ids=sorted(proto_keep_ids),
            proto_keep_positions=[float(x) for x in proto_keep_positions],
            proto_keep_threshold=proto_keep_threshold,
            proto_keep_shear=sorted(
                float(shear_by_event[event_id])
                for event_id in proto_keep_ids
                if event_id in shear_by_event
            ),
        )

    candidates: list[_Candidate] = []

    for anchor in anchors:
        neighborhood = arc_search_core._causal_neighborhood(
            anchor_id=anchor.id,
            by_id=by_id,
            reverse_links=reverse_links,
            allowed_ids=window_ids,
            max_depth=3,
        )

        pool_ids = {anchor.id} | neighborhood | proto_keep_ids
        pool = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        pool.sort(key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick))

        selected = arc_search_core._downsample_preserving_continuity(
            pool=pool,
            bridge_pool=window,
            protagonist=protagonist,
            anchor_id=anchor.id,
            max_events=MAX_EVENTS,
        )

        beats = classify_beats(selected)
        beats = arc_search_core._enforce_monotonic_beats(selected, beats)
        validation = validate_arc(events=selected, beats=beats, total_sim_time=total_sim_time)
        arc_score = score_arc(selected, beats) if validation.valid else None

        candidates.append(
            _Candidate(
                events=selected,
                beats=beats,
                validation=validation,
                protagonist=protagonist,
                arc_score=arc_score,
                importance_sum=sum(arc_search_core._event_importance(event) for event in selected),
                anchor_id=anchor.id,
            )
        )

    candidates_with_dev_beats = int(sum(1 for candidate in candidates if _dev_beat_count(candidate.beats) >= 1))

    valid = [candidate for candidate in candidates if candidate.validation.valid]
    if valid:
        valid_with_aftermath = [candidate for candidate in valid if arc_search_core._has_post_peak_consequence(candidate.events, candidate.beats)]
        if valid_with_aftermath:
            best = max(
                valid_with_aftermath,
                key=lambda candidate: (
                    float(candidate.arc_score.composite) if candidate.arc_score is not None else 0.0,
                    candidate.importance_sum,
                ),
            )
        else:
            best = max(
                valid,
                key=lambda candidate: (
                    float(candidate.arc_score.composite) if candidate.arc_score is not None else 0.0,
                    candidate.importance_sum,
                ),
            )
            widened = arc_search_core._extend_candidate_with_post_peak_consequence(
                candidate=best,
                window=window,
                protagonist=protagonist,
                max_events=MAX_EVENTS,
                total_sim_time=total_sim_time,
                grammar_config=None,
            )
            if widened is not None:
                best = widened
                candidates.append(widened)
                candidates_with_dev_beats = int(sum(1 for candidate in candidates if _dev_beat_count(candidate.beats) >= 1))
    else:
        aftermath_candidates = [candidate for candidate in candidates if arc_search_core._has_post_peak_consequence(candidate.events, candidate.beats)]
        candidate_pool = aftermath_candidates if aftermath_candidates else candidates
        best = min(candidate_pool, key=lambda candidate: (len(candidate.validation.violations), -candidate.importance_sum))

    tp_position, tp_event_type = _tp_info(
        events=best.events,
        beats=best.beats,
        simulation_end_tick=simulation_end_tick,
    )
    q_score = float(score_arc(best.events, best.beats).composite) if best.events else None

    return SearchOutcome(
        seed=int(seed),
        condition=condition,
        agent_id=protagonist,
        valid=bool(best.validation.valid),
        q_score=q_score,
        tp_position=tp_position,
        tp_event_type=tp_event_type,
        dev_beat_count=_dev_beat_count(best.beats),
        total_beats=len(best.beats),
        event_ids=[str(event.id) for event in best.events],
        beats=[str(beat.value) for beat in best.beats],
        violations=[str(v) for v in best.validation.violations],
        total_candidates=int(len(candidates)),
        candidates_with_dev_beats=int(candidates_with_dev_beats),
        proto_keep_ids=sorted(proto_keep_ids),
        proto_keep_positions=[float(x) for x in proto_keep_positions],
        proto_keep_threshold=proto_keep_threshold,
        proto_keep_shear=sorted(
            float(shear_by_event[event_id])
            for event_id in proto_keep_ids
            if event_id in shear_by_event
        ),
    )


def _error_outcome(seed: int, condition: ConditionName, agent: str, error: Exception) -> SearchOutcome:
    return SearchOutcome(
        seed=int(seed),
        condition=condition,
        agent_id=agent,
        valid=False,
        q_score=None,
        tp_position=None,
        tp_event_type=None,
        dev_beat_count=0,
        total_beats=0,
        event_ids=[],
        beats=[],
        violations=["search_crash"],
        total_candidates=0,
        candidates_with_dev_beats=0,
        proto_keep_ids=[],
        proto_keep_positions=[],
        proto_keep_threshold=None,
        proto_keep_shear=[],
        error=f"{type(error).__name__}: {error}",
    )


def _run_condition(
    *,
    condition: ConditionName,
    seed_contexts: list[Any],
    importance_by_seed: dict[int, dict[str, dict[str, float]]],
    shear_by_seed: dict[int, dict[str, float]],
) -> dict[int, dict[str, SearchOutcome]]:
    per_seed: dict[int, dict[str, SearchOutcome]] = {}
    total = len(seed_contexts)

    for idx, context in enumerate(seed_contexts, start=1):
        seed = int(context.seed)
        print(
            f"[condition={condition}] seed {idx:03d}/{total:03d} ({seed})",
            flush=True,
        )

        importance = importance_by_seed[seed]
        shear = shear_by_seed[seed]
        per_seed[seed] = {}

        for agent in DINNER_PARTY_AGENTS:
            try:
                outcome = _search_with_condition(
                    seed=seed,
                    all_events=context.events,
                    protagonist=agent,
                    total_sim_time=context.total_sim_time,
                    simulation_end_tick=context.simulation_end_tick,
                    condition=condition,
                    importance_by_event=importance,
                    shear_by_event=shear,
                )
            except Exception as exc:  # defensive: keep experiment running per requirement
                outcome = _error_outcome(seed=seed, condition=condition, agent=agent, error=exc)
            per_seed[seed][agent] = outcome

    return per_seed


def _aggregate_condition(
    *,
    condition: ConditionName,
    seeds: list[int],
    per_seed: dict[int, dict[str, SearchOutcome]],
    baseline_invalid_set: set[int],
) -> dict[str, Any]:
    diana_rows = [per_seed[int(seed)]["diana"] for seed in seeds]
    diana_valid = [row for row in diana_rows if row.valid]
    diana_invalid_seeds = sorted(int(row.seed) for row in diana_rows if not row.valid)
    baseline_valid_set = set(int(seed) for seed in seeds) - set(baseline_invalid_set)

    focal_fixed = sorted(
        int(row.seed)
        for row in diana_rows
        if int(row.seed) in baseline_invalid_set and row.valid
    )
    focal_regressed = sorted(
        int(row.seed)
        for row in diana_rows
        if int(row.seed) in baseline_valid_set and not row.valid
    )

    failing_rows = [row for row in diana_rows if not row.valid]
    failing_candidates_total = int(sum(int(row.total_candidates) for row in failing_rows))
    failing_candidates_dev = int(sum(int(row.candidates_with_dev_beats) for row in failing_rows))
    failing_dev_fraction = float(failing_candidates_dev / failing_candidates_total) if failing_candidates_total > 0 else 0.0

    diana_metrics = {
        "valid_count": int(len(diana_valid)),
        "invalid_count": int(len(diana_rows) - len(diana_valid)),
        "invalid_seeds": diana_invalid_seeds,
        "focal_fixed_count": int(len(focal_fixed)),
        "focal_fixed_seeds": focal_fixed,
        "focal_regressed_count": int(len(focal_regressed)),
        "focal_regressed_seeds": focal_regressed,
        "mean_q_valid_only": float(_mean([float(row.q_score) for row in diana_valid if row.q_score is not None])),
        "mean_tp_position_all_arcs": float(_mean([float(row.tp_position) for row in diana_rows if row.tp_position is not None])),
        "mean_development_beats_all_arcs": float(_mean([float(row.dev_beat_count) for row in diana_rows])),
        "total_candidates_failing_seeds": failing_candidates_total,
        "fraction_candidates_with_dev_ge_1_failing_seeds": failing_dev_fraction,
        "crashed_seeds": sorted(int(row.seed) for row in diana_rows if row.error is not None),
    }

    per_agent_valid_count: dict[str, int] = {agent: 0 for agent in DINNER_PARTY_AGENTS}
    seed_mean_q: list[float] = []
    seed_va: list[float] = []
    all_valid_count = 0
    crashed_seed_map: dict[int, list[str]] = {}
    per_seed_summary: dict[str, Any] = {}

    for seed in seeds:
        by_agent = per_seed[int(seed)]
        valid_scores: list[float] = []
        valid_count = 0
        crashed_agents: list[str] = []

        for agent in DINNER_PARTY_AGENTS:
            row = by_agent[agent]
            if row.error is not None:
                crashed_agents.append(agent)
            if row.valid:
                valid_count += 1
                per_agent_valid_count[agent] += 1
                if row.q_score is not None:
                    valid_scores.append(float(row.q_score))

        mean_q_seed = float(_mean(valid_scores))
        va_seed = float(mean_q_seed * (valid_count / float(len(DINNER_PARTY_AGENTS)))) if valid_count > 0 else 0.0
        seed_mean_q.append(mean_q_seed)
        seed_va.append(va_seed)

        if valid_count == len(DINNER_PARTY_AGENTS):
            all_valid_count += 1

        if crashed_agents:
            crashed_seed_map[int(seed)] = sorted(crashed_agents)

        per_seed_summary[str(seed)] = {
            "valid_count": int(valid_count),
            "mean_q": mean_q_seed,
            "va_score": va_seed,
            "all_valid": bool(valid_count == len(DINNER_PARTY_AGENTS)),
            "agents": {
                agent: asdict(by_agent[agent])
                for agent in DINNER_PARTY_AGENTS
            },
        }

    total = len(seeds)
    full_metrics = {
        "mean_q": float(_mean(seed_mean_q)),
        "va_score": float(_mean(seed_va)),
        "all_valid_rate": float(all_valid_count / total) if total > 0 else 0.0,
        "per_agent_validity": {
            agent: (float(per_agent_valid_count[agent] / total) if total > 0 else 0.0)
            for agent in DINNER_PARTY_AGENTS
        },
    }

    return {
        "condition": condition,
        "diana": diana_metrics,
        "full": full_metrics,
        "crashed_seeds": {
            str(seed): agents
            for seed, agents in sorted(crashed_seed_map.items())
        },
        "per_seed": per_seed_summary,
    }


def _baseline_verification(
    *,
    baseline_summary: dict[str, Any],
    parity_mismatch_count: int,
) -> dict[str, Any]:
    observed_invalid = [int(seed) for seed in baseline_summary["diana"]["invalid_seeds"]]
    observed_mean_q = float(baseline_summary["full"]["mean_q"])
    observed_all_valid = float(baseline_summary["full"]["all_valid_rate"])

    checks = {
        "diana_invalid_count": int(baseline_summary["diana"]["invalid_count"]) == 9,
        "diana_invalid_seed_list": observed_invalid == BASELINE_INVALID_SEEDS_EXPECTED,
        "mean_q_all_agents": round(observed_mean_q, 3) == BASELINE_MEAN_Q_ALL_EXPECTED,
        "all_valid_rate": abs(observed_all_valid - BASELINE_ALL_VALID_EXPECTED) <= 1e-12,
        "search_arc_parity_diana": int(parity_mismatch_count) == 0,
    }
    passed = all(bool(v) for v in checks.values())

    return {
        "passed": bool(passed),
        "checks": checks,
        "expected": {
            "diana_invalid_count": 9,
            "diana_invalid_seeds": list(BASELINE_INVALID_SEEDS_EXPECTED),
            "mean_q_all_agents": BASELINE_MEAN_Q_ALL_EXPECTED,
            "all_valid_rate": BASELINE_ALL_VALID_EXPECTED,
            "diana_parity_mismatch_count": 0,
        },
        "observed": {
            "diana_invalid_count": int(baseline_summary["diana"]["invalid_count"]),
            "diana_invalid_seeds": observed_invalid,
            "mean_q_all_agents": observed_mean_q,
            "all_valid_rate": observed_all_valid,
            "diana_parity_mismatch_count": int(parity_mismatch_count),
        },
    }


def _temporal_verification(
    *,
    temporal_summary: dict[str, Any],
) -> dict[str, Any]:
    valid_count = int(temporal_summary["diana"]["valid_count"])
    focal_fixed = int(temporal_summary["diana"]["focal_fixed_count"])

    checks = {
        "diana_valid_47_of_50": valid_count == 47,
        "focal_fixed_7_of_9": focal_fixed == 7,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "diana_valid_count": valid_count,
            "focal_fixed_count": focal_fixed,
            "invalid_seeds": [int(seed) for seed in temporal_summary["diana"]["invalid_seeds"]],
        },
    }


def _mean_proto_keep_shear(row: SearchOutcome) -> float:
    return float(_mean([float(x) for x in row.proto_keep_shear]))


def _shear_diagnostic(
    *,
    seeds: list[int],
    per_seed_by_condition: dict[str, dict[int, dict[str, SearchOutcome]]],
) -> dict[str, Any]:
    baseline_rows = [per_seed_by_condition["baseline"][int(seed)]["diana"] for seed in seeds]

    baseline_valid = [row for row in baseline_rows if row.valid]
    baseline_invalid = [row for row in baseline_rows if not row.valid]

    valid_shear = [_mean_proto_keep_shear(row) for row in baseline_valid]
    invalid_shear = [_mean_proto_keep_shear(row) for row in baseline_invalid]

    all_seed_shear = np.asarray([_mean_proto_keep_shear(row) for row in baseline_rows], dtype=float)
    all_seed_invalid = np.asarray([1.0 if not row.valid else 0.0 for row in baseline_rows], dtype=float)

    corr = 0.0
    if all_seed_shear.size >= 2 and np.std(all_seed_shear, ddof=0) > 0.0 and np.std(all_seed_invalid, ddof=0) > 0.0:
        corr = float(np.corrcoef(all_seed_shear, all_seed_invalid)[0, 1])

    high_invalid_rate = 0.0
    low_invalid_rate = 0.0
    if all_seed_shear.size:
        q75 = float(np.percentile(all_seed_shear, 75.0))
        q25 = float(np.percentile(all_seed_shear, 25.0))

        high_mask = all_seed_shear >= q75
        low_mask = all_seed_shear <= q25
        if np.any(high_mask):
            high_invalid_rate = float(np.mean(all_seed_invalid[high_mask]))
        if np.any(low_mask):
            low_invalid_rate = float(np.mean(all_seed_invalid[low_mask]))

    filtered_means: dict[str, float] = {}
    for condition in ["temporal_filter", "shear_filter_median", "shear_filter_p25"]:
        rows = [per_seed_by_condition[condition][int(seed)]["diana"] for seed in seeds]
        filtered_means[condition] = float(_mean([_mean_proto_keep_shear(row) for row in rows]))

    interpretation = {
        "baseline_invalid_has_higher_shear": float(_mean(invalid_shear)) > float(_mean(valid_shear)),
        "high_shear_correlates_with_invalidity": bool(corr > 0.0 and high_invalid_rate > low_invalid_rate),
        "correlation_seed_proto_shear_vs_invalid": float(corr),
        "high_quartile_invalid_rate": float(high_invalid_rate),
        "low_quartile_invalid_rate": float(low_invalid_rate),
    }

    return {
        "baseline": {
            "mean_proto_keep_shear_valid_diana_seeds": float(_mean(valid_shear)),
            "mean_proto_keep_shear_invalid_diana_seeds": float(_mean(invalid_shear)),
            "per_seed_mean_proto_keep_shear": {
                str(int(row.seed)): float(_mean_proto_keep_shear(row))
                for row in baseline_rows
            },
        },
        "filtered_proto_keep_mean_shear": filtered_means,
        "interpretation": interpretation,
    }


def _build_summary_markdown(
    *,
    seeds: list[int],
    verification: dict[str, Any],
    temporal_verification: dict[str, Any] | None,
    condition_summaries: dict[str, dict[str, Any]],
    shear_diag: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Shear-Filtered Injection Experiment")
    lines.append("")
    lines.append("## Agent-Relative Importance Decomposition")
    lines.append("")
    lines.append(
        "Per-agent importance is computed as `global_event_importance * participant_factor * delta_multiplier`, "
        "where participant factor is 1.0 for direct participant events and 0.2 for non-participant events, "
        "and delta multiplier is driven by deltas that explicitly reference the agent (`delta.agent` / `delta.agent_b`)."
    )
    lines.append("")

    baseline = condition_summaries["baseline"]
    lines.append("## Baseline Verification")
    lines.append("")
    lines.append(f"- Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    lines.append(f"- Baseline passed: {'yes' if verification['passed'] else 'no'}")
    lines.append(f"- Diana invalid: {baseline['diana']['invalid_count']}/50")
    lines.append(f"- Diana invalid seeds: {baseline['diana']['invalid_seeds']}")
    lines.append(f"- Mean Q (all 6 agents): {baseline['full']['mean_q']:.3f}")
    lines.append(f"- All-valid rate: {baseline['full']['all_valid_rate'] * 100.0:.1f}%")
    lines.append("")

    lines.append("## Diana Outcomes")
    lines.append("")
    lines.append(
        "| Condition | Diana Valid | Diana Invalid | Focal Fixed | Focal Regressed | Mean Q (valid) | Mean TP Pos | Mean Dev Beats | Fail Candidates | Fail Candidate Dev Fraction |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for condition in CONDITIONS:
        row = condition_summaries[condition]["diana"]
        lines.append(
            "| "
            f"{condition} | "
            f"{int(row['valid_count'])}/50 | "
            f"{int(row['invalid_count'])}/50 | "
            f"{int(row['focal_fixed_count'])}/9 | "
            f"{int(row['focal_regressed_count'])} | "
            f"{float(row['mean_q_valid_only']):.3f} | "
            f"{float(row['mean_tp_position_all_arcs']):.3f} | "
            f"{float(row['mean_development_beats_all_arcs']):.2f} | "
            f"{int(row['total_candidates_failing_seeds'])} | "
            f"{float(row['fraction_candidates_with_dev_ge_1_failing_seeds']):.3f} |"
        )
    lines.append("")

    lines.append("## Full 6-Agent Metrics")
    lines.append("")
    lines.append("| Condition | Mean Q | VA | All-Valid | Per-Agent Validity |")
    lines.append("|---|---|---|---|---|")
    for condition in CONDITIONS:
        row = condition_summaries[condition]["full"]
        per_agent = ", ".join(
            f"{agent}:{100.0 * float(row['per_agent_validity'][agent]):.1f}%"
            for agent in DINNER_PARTY_AGENTS
        )
        lines.append(
            "| "
            f"{condition} | "
            f"{float(row['mean_q']):.3f} | "
            f"{float(row['va_score']):.3f} | "
            f"{100.0 * float(row['all_valid_rate']):.1f}% | "
            f"{per_agent} |"
        )
    lines.append("")

    if temporal_verification is not None:
        lines.append("## Temporal Reproduction Check")
        lines.append("")
        lines.append(f"- Reproduced (47/50 valid, 7/9 fixed): {'yes' if temporal_verification['passed'] else 'no'}")
        lines.append(
            f"- Observed Diana valid: {int(temporal_verification['observed']['diana_valid_count'])}/50"
        )
        lines.append(
            f"- Observed focal fixed: {int(temporal_verification['observed']['focal_fixed_count'])}/9"
        )
        lines.append(f"- Temporal invalid seeds: {temporal_verification['observed']['invalid_seeds']}")
        lines.append("")

    lines.append("## Shear Diagnostic (CV of 6-agent importance vector)")
    lines.append("")
    lines.append(
        "| Metric | Value |"
    )
    lines.append("|---|---|")
    lines.append(
        f"| Baseline mean proto shear (Diana-valid seeds) | {float(shear_diag['baseline']['mean_proto_keep_shear_valid_diana_seeds']):.3f} |"
    )
    lines.append(
        f"| Baseline mean proto shear (Diana-invalid seeds) | {float(shear_diag['baseline']['mean_proto_keep_shear_invalid_diana_seeds']):.3f} |"
    )
    for condition in ["temporal_filter", "shear_filter_median", "shear_filter_p25"]:
        lines.append(
            f"| Mean proto shear ({condition}) | {float(shear_diag['filtered_proto_keep_mean_shear'][condition]):.3f} |"
        )
    lines.append(
        f"| Corr(seed proto shear, Diana invalid) baseline | {float(shear_diag['interpretation']['correlation_seed_proto_shear_vs_invalid']):.3f} |"
    )
    lines.append(
        f"| Invalid rate in highest-shear quartile | {100.0 * float(shear_diag['interpretation']['high_quartile_invalid_rate']):.1f}% |"
    )
    lines.append(
        f"| Invalid rate in lowest-shear quartile | {100.0 * float(shear_diag['interpretation']['low_quartile_invalid_rate']):.1f}% |"
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append(
        "- High-shear events correlate with contamination: "
        f"{'yes' if bool(shear_diag['interpretation']['high_shear_correlates_with_invalidity']) else 'no'}"
    )
    lines.append(
        "- Invalid seeds contain higher-shear proto events: "
        f"{'yes' if bool(shear_diag['interpretation']['baseline_invalid_has_higher_shear']) else 'no'}"
    )
    lines.append("")

    crash_rows = []
    for condition in CONDITIONS:
        crashed = condition_summaries[condition]["crashed_seeds"]
        if crashed:
            crash_rows.append((condition, crashed))

    lines.append("## Crashes")
    lines.append("")
    if not crash_rows:
        lines.append("- No seed crashes.")
    else:
        for condition, crashed in crash_rows:
            lines.append(f"- {condition}: {crashed}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shear-filtered proto-keep injection search experiment.")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--output-summary", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_json_path = _resolve_path(args.output_json)
    output_summary_path = _resolve_path(args.output_summary)
    seeds = _parse_seeds(args.seeds) if args.seeds else list(DEFAULT_SEEDS)

    print()
    print("=== SHEAR-FILTERED INJECTION EXPERIMENT ===")
    print()
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Conditions: {CONDITIONS}")
    print(f"Focal seeds: {FOCAL_SEEDS}")
    print()

    prep_start = time.time()
    seed_contexts = _prepare_seed_contexts(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    print(f"Prepared depth-2 full-evolution contexts in {time.time() - prep_start:.1f}s.")

    importance_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    shear_by_seed: dict[int, dict[str, float]] = {}
    shear_matrix_cv: dict[str, dict[str, float]] = {}

    for idx, context in enumerate(seed_contexts, start=1):
        seed = int(context.seed)
        print(f"[importance] seed {idx:03d}/{len(seed_contexts):03d} ({seed})", flush=True)
        importance_map, shear_map = _compute_importance_and_shear(context.events)
        importance_by_seed[seed] = importance_map
        shear_by_seed[seed] = shear_map
        shear_matrix_cv[str(seed)] = {str(event_id): float(cv) for event_id, cv in sorted(shear_map.items())}

    print()
    print("Running baseline verification condition first...", flush=True)
    baseline_per_seed = _run_condition(
        condition="baseline",
        seed_contexts=seed_contexts,
        importance_by_seed=importance_by_seed,
        shear_by_seed=shear_by_seed,
    )

    baseline_invalid_set = set(int(seed) for seed in BASELINE_INVALID_SEEDS_EXPECTED)
    condition_summaries: dict[str, dict[str, Any]] = {
        "baseline": _aggregate_condition(
            condition="baseline",
            seeds=seeds,
            per_seed=baseline_per_seed,
            baseline_invalid_set=baseline_invalid_set,
        )
    }
    per_seed_by_condition: dict[str, dict[int, dict[str, SearchOutcome]]] = {
        "baseline": baseline_per_seed,
    }

    parity_mismatches: list[dict[str, Any]] = []
    for context in seed_contexts:
        seed = int(context.seed)
        instrumented = baseline_per_seed[seed]["diana"]
        if instrumented.error is not None:
            parity_mismatches.append(
                {
                    "seed": seed,
                    "reason": instrumented.error,
                }
            )
            continue

        direct = search_arc(
            all_events=context.events,
            protagonist="diana",
            max_events=MAX_EVENTS,
            total_sim_time=context.total_sim_time,
            grammar_config=None,
        )
        direct_ids = [str(event.id) for event in direct.events]
        direct_beats = [str(beat.value) for beat in direct.beats]

        if instrumented.event_ids != direct_ids or instrumented.beats != direct_beats:
            parity_mismatches.append(
                {
                    "seed": seed,
                    "instrumented_event_ids": instrumented.event_ids,
                    "direct_event_ids": direct_ids,
                    "instrumented_beats": instrumented.beats,
                    "direct_beats": direct_beats,
                }
            )

    baseline_check = _baseline_verification(
        baseline_summary=condition_summaries["baseline"],
        parity_mismatch_count=len(parity_mismatches),
    )

    if not baseline_check["passed"]:
        payload = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment": "shear_filter_search",
                "status": "baseline_failed",
                "seeds": [int(seed) for seed in seeds],
            },
            "baseline_verification": baseline_check,
            "parity_mismatches": parity_mismatches,
        }
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        raise RuntimeError(
            "Baseline verification failed; stopping before experimental conditions. "
            f"Observed={baseline_check['observed']}"
        )

    print("Baseline verification passed.", flush=True)
    print()

    for condition in ["temporal_filter", "shear_filter_median", "shear_filter_p25"]:
        per_seed = _run_condition(
            condition=condition,
            seed_contexts=seed_contexts,
            importance_by_seed=importance_by_seed,
            shear_by_seed=shear_by_seed,
        )
        per_seed_by_condition[condition] = per_seed
        condition_summaries[condition] = _aggregate_condition(
            condition=condition,
            seeds=seeds,
            per_seed=per_seed,
            baseline_invalid_set=baseline_invalid_set,
        )

    temporal_check = _temporal_verification(temporal_summary=condition_summaries["temporal_filter"])
    shear_diag = _shear_diagnostic(
        seeds=seeds,
        per_seed_by_condition=per_seed_by_condition,
    )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": "shear_filter_search",
            "seeds": [int(seed) for seed in seeds],
            "conditions": list(CONDITIONS),
            "focal_seeds": [int(seed) for seed in FOCAL_SEEDS],
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "shear_metric": "coefficient_of_variation(std/mean) over 6D agent-relative importance vector",
            "early_cutoff": float(EARLY_CUTOFF),
        },
        "baseline_verification": baseline_check,
        "temporal_filter_reproduction": temporal_check,
        "condition_summaries": condition_summaries,
        "shear_diagnostic": shear_diag,
        "shear_matrix_cv": shear_matrix_cv,
        "parity_mismatches": parity_mismatches,
    }

    summary_text = _build_summary_markdown(
        seeds=seeds,
        verification=baseline_check,
        temporal_verification=temporal_check,
        condition_summaries=condition_summaries,
        shear_diag=shear_diag,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(
        "Temporal reproduction: "
        f"passed={'yes' if temporal_check['passed'] else 'no'}, "
        f"diana_valid={temporal_check['observed']['diana_valid_count']}/50, "
        f"focal_fixed={temporal_check['observed']['focal_fixed_count']}/9"
    )


if __name__ == "__main__":
    main()
