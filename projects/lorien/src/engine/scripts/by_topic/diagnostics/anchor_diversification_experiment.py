"""Anchor diversification experiment for Diana arc extraction.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.anchor_diversification_experiment
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from narrativefield.extraction import arc_search as arc_search_core
from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import _Candidate, search_arc
from narrativefield.extraction.arc_validator import validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.schema.events import BeatType, Event, EventType
from scripts.feasible_volume_sweep import _prepare_seed_contexts
from scripts.k_sweep_experiment import DINNER_PARTY_AGENTS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "anchor_diversification.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "anchor_diversification_summary.md"
DEFAULT_SEEDS = list(range(1, 51))
FOCAL_SEEDS = [2, 3, 9, 25, 32, 33, 35, 38, 43]
MIDRANGE_LOWER = 0.30
MIDRANGE_UPPER = 0.70
MIN_MIDRANGE_ANCHORS = 2
MAX_ANCHORS = 8
EPSILON = 1e-12

StrategyName = Literal["default", "diversified", "all_midrange"]


@dataclass
class FocalSeedDetail:
    seed: int
    valid: bool
    q_score: float | None
    tp_position: float | None
    dev_beat_count: int
    total_candidates: int
    candidates_with_dev_beats: int
    anchor_positions: list[float]
    available_midrange_anchors: int


@dataclass
class StrategyResult:
    strategy: str
    diana_valid_count: int
    diana_invalid_count: int
    diana_invalid_seeds: list[int]
    diana_mean_q: float
    diana_mean_tp_position: float
    diana_mean_dev_beats: float
    focal_seeds_detail: list[FocalSeedDetail]


@dataclass
class FullExtractionResult:
    strategy: str
    mean_q: float
    va: float
    all_valid_rate: float
    per_agent_validity: dict[str, float]
    diana_invalid_count: int


@dataclass
class SearchOutcome:
    seed: int
    strategy: StrategyName
    valid: bool
    q_score: float
    tp_position: float | None
    tp_event_type: str | None
    dev_beat_count: int
    total_beats: int
    event_ids: list[str]
    beats: list[str]
    violations: list[str]
    total_candidates: int
    candidates_with_dev_beats: int
    anchor_positions: list[float]
    available_midrange_anchors: int


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


def _dev_beat_count(beats: list[BeatType]) -> int:
    return int(sum(1 for beat in beats if beat in {BeatType.COMPLICATION, BeatType.ESCALATION}))


def _event_pos(event: Event, simulation_end_tick: int) -> float:
    return float(int(event.tick_id) / max(int(simulation_end_tick), 1))


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


def _scored_anchor_pool(
    *,
    window: list[Event],
) -> list[tuple[float, Event]]:
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


def _select_diversified_anchors(
    *,
    scored_events: list[tuple[float, Event]],
    simulation_end_tick: int,
    max_anchors: int = MAX_ANCHORS,
    min_midrange: int = MIN_MIDRANGE_ANCHORS,
    midrange_lower: float = MIDRANGE_LOWER,
    midrange_upper: float = MIDRANGE_UPPER,
) -> tuple[list[Event], int]:
    midrange: list[tuple[float, Event]] = []
    other: list[tuple[float, Event]] = []
    for importance, event in scored_events:
        pos = _event_pos(event, simulation_end_tick)
        if midrange_lower <= pos <= midrange_upper:
            midrange.append((importance, event))
        else:
            other.append((importance, event))

    midrange.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))
    other.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))

    forced_mid = [event for _, event in midrange[:min_midrange]]
    forced_ids = {event.id for event in forced_mid}

    remaining_pool = [(importance, event) for importance, event in scored_events if event.id not in forced_ids]
    remaining_pool.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))

    fill = [event for _, event in remaining_pool[: max(0, max_anchors - len(forced_mid))]]
    anchors = forced_mid + fill

    # Deduplicate while preserving order.
    deduped: list[Event] = []
    seen: set[str] = set()
    for event in anchors:
        if event.id in seen:
            continue
        deduped.append(event)
        seen.add(event.id)
    return (deduped, int(len(midrange)))


def _select_all_midrange_anchors(
    *,
    scored_events: list[tuple[float, Event]],
    simulation_end_tick: int,
    max_anchors: int = MAX_ANCHORS,
    midrange_lower: float = MIDRANGE_LOWER,
    midrange_upper: float = MIDRANGE_UPPER,
) -> tuple[list[Event], int]:
    midrange: list[tuple[float, Event]] = []
    for importance, event in scored_events:
        pos = _event_pos(event, simulation_end_tick)
        if midrange_lower <= pos <= midrange_upper:
            midrange.append((importance, event))
    midrange.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))
    anchors = [event for _, event in midrange[:max_anchors]]
    return (anchors, int(len(midrange)))


def _choose_anchors(
    *,
    strategy: StrategyName,
    scored_events: list[tuple[float, Event]],
    protagonist: str,
    simulation_end_tick: int,
) -> tuple[list[Event], int]:
    if strategy == "default":
        anchors = _default_anchor_selection(
            scored_events=scored_events,
            protagonist=protagonist,
            max_anchors=MAX_ANCHORS,
        )
        available_midrange = sum(
            1 for _, event in scored_events
            if MIDRANGE_LOWER <= _event_pos(event, simulation_end_tick) <= MIDRANGE_UPPER
        )
        return (anchors, int(available_midrange))

    protagonist_scored = [(importance, event) for importance, event in scored_events if arc_search_core._involves(event, protagonist)]
    selection_pool = protagonist_scored if protagonist_scored else scored_events

    if strategy == "diversified":
        return _select_diversified_anchors(
            scored_events=selection_pool,
            simulation_end_tick=simulation_end_tick,
            max_anchors=MAX_ANCHORS,
            min_midrange=MIN_MIDRANGE_ANCHORS,
            midrange_lower=MIDRANGE_LOWER,
            midrange_upper=MIDRANGE_UPPER,
        )
    if strategy == "all_midrange":
        return _select_all_midrange_anchors(
            scored_events=selection_pool,
            simulation_end_tick=simulation_end_tick,
            max_anchors=MAX_ANCHORS,
            midrange_lower=MIDRANGE_LOWER,
            midrange_upper=MIDRANGE_UPPER,
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def _search_with_anchor_strategy(
    *,
    seed: int,
    all_events: list[Event],
    protagonist: str,
    total_sim_time: float | None,
    simulation_end_tick: int,
    strategy: StrategyName,
) -> SearchOutcome:
    window = list(all_events)
    window.sort(key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick))

    if not window:
        return SearchOutcome(
            seed=int(seed),
            strategy=strategy,
            valid=False,
            q_score=0.0,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            event_ids=[],
            beats=[],
            violations=["No events in time window"],
            total_candidates=0,
            candidates_with_dev_beats=0,
            anchor_positions=[],
            available_midrange_anchors=0,
        )

    by_id: dict[str, Event] = {event.id: event for event in all_events}
    reverse_links = arc_search_core._build_reverse_links(all_events)
    window_ids = {event.id for event in window}

    anchor_scored = _scored_anchor_pool(window=window)
    anchors, available_midrange = _choose_anchors(
        strategy=strategy,
        scored_events=anchor_scored,
        protagonist=protagonist,
        simulation_end_tick=simulation_end_tick,
    )
    anchor_positions = [_event_pos(event, simulation_end_tick) for event in anchors]

    proto_events_sorted = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (-arc_search_core._event_importance(event), event.sim_time, event.id),
    )
    proto_keep_ids = {event.id for event in proto_events_sorted[: 20 * 2]}
    proto_by_time = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick),
    )
    for event in proto_by_time[:3]:
        proto_keep_ids.add(event.id)
    for event in proto_by_time[-3:]:
        proto_keep_ids.add(event.id)

    candidates: list[_Candidate] = []
    anchor_id_to_index: dict[str, int] = {}

    for anchor_index, anchor in enumerate(anchors):
        anchor_id_to_index[anchor.id] = int(anchor_index)
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
            max_events=20,
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

    candidates_with_dev_beats = int(
        sum(1 for candidate in candidates if _dev_beat_count(candidate.beats) >= 1)
    )

    if not candidates:
        return SearchOutcome(
            seed=int(seed),
            strategy=strategy,
            valid=False,
            q_score=0.0,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            event_ids=[],
            beats=[],
            violations=["No anchors selected"],
            total_candidates=0,
            candidates_with_dev_beats=0,
            anchor_positions=anchor_positions,
            available_midrange_anchors=int(available_midrange),
        )

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
                max_events=20,
                total_sim_time=total_sim_time,
                grammar_config=None,
            )
            if widened is not None:
                best = widened
                candidates.append(widened)
                candidates_with_dev_beats = int(
                    sum(1 for candidate in candidates if _dev_beat_count(candidate.beats) >= 1)
                )
    else:
        aftermath_candidates = [candidate for candidate in candidates if arc_search_core._has_post_peak_consequence(candidate.events, candidate.beats)]
        candidate_pool = aftermath_candidates if aftermath_candidates else candidates
        best = min(candidate_pool, key=lambda candidate: (len(candidate.validation.violations), -candidate.importance_sum))

    tp_position, tp_event_type = _tp_info(
        events=best.events,
        beats=best.beats,
        simulation_end_tick=simulation_end_tick,
    )
    q_score = float(score_arc(best.events, best.beats).composite) if best.events else 0.0

    return SearchOutcome(
        seed=int(seed),
        strategy=strategy,
        valid=bool(best.validation.valid),
        q_score=float(q_score),
        tp_position=tp_position,
        tp_event_type=tp_event_type,
        dev_beat_count=_dev_beat_count(best.beats),
        total_beats=len(best.beats),
        event_ids=[str(event.id) for event in best.events],
        beats=[str(beat.value) for beat in best.beats],
        violations=[str(v) for v in best.validation.violations],
        total_candidates=int(len(candidates)),
        candidates_with_dev_beats=int(candidates_with_dev_beats),
        anchor_positions=[float(pos) for pos in anchor_positions],
        available_midrange_anchors=int(available_midrange),
    )


def _strategy_result(
    *,
    strategy: StrategyName,
    outcomes: list[SearchOutcome],
) -> StrategyResult:
    valid_outcomes = [outcome for outcome in outcomes if outcome.valid]
    invalid = sorted(int(outcome.seed) for outcome in outcomes if not outcome.valid)
    focal_set = set(FOCAL_SEEDS)
    focal_details = [
        FocalSeedDetail(
            seed=int(outcome.seed),
            valid=bool(outcome.valid),
            q_score=float(outcome.q_score) if outcome.valid else None,
            tp_position=float(outcome.tp_position) if outcome.tp_position is not None else None,
            dev_beat_count=int(outcome.dev_beat_count),
            total_candidates=int(outcome.total_candidates),
            candidates_with_dev_beats=int(outcome.candidates_with_dev_beats),
            anchor_positions=[float(pos) for pos in outcome.anchor_positions],
            available_midrange_anchors=int(outcome.available_midrange_anchors),
        )
        for outcome in outcomes
        if int(outcome.seed) in focal_set
    ]
    focal_details.sort(key=lambda row: int(row.seed))

    return StrategyResult(
        strategy=strategy,
        diana_valid_count=int(len(valid_outcomes)),
        diana_invalid_count=int(len(outcomes) - len(valid_outcomes)),
        diana_invalid_seeds=invalid,
        diana_mean_q=float(_mean([float(outcome.q_score) for outcome in valid_outcomes])),
        diana_mean_tp_position=float(_mean([float(outcome.tp_position) for outcome in valid_outcomes if outcome.tp_position is not None])),
        diana_mean_dev_beats=float(_mean([float(outcome.dev_beat_count) for outcome in valid_outcomes])),
        focal_seeds_detail=focal_details,
    )


def _full_extraction_result(
    *,
    strategy: StrategyName,
    seed_contexts: list[Any],
    diana_outcomes_by_seed: dict[int, SearchOutcome],
) -> FullExtractionResult:
    per_agent_valid_count: dict[str, int] = {agent: 0 for agent in DINNER_PARTY_AGENTS}
    seed_mean_q: list[float] = []
    seed_va: list[float] = []
    all_valid_count = 0
    diana_invalid_count = 0

    for context in seed_contexts:
        by_agent: dict[str, SearchOutcome] = {}
        seed = int(context.seed)
        by_agent["diana"] = diana_outcomes_by_seed[seed]

        for agent in DINNER_PARTY_AGENTS:
            if agent == "diana":
                continue
            result = search_arc(
                all_events=context.events,
                protagonist=agent,
                max_events=20,
                total_sim_time=context.total_sim_time,
                grammar_config=None,
            )
            q_score = float(score_arc(result.events, result.beats).composite) if result.events else 0.0
            tp_position, tp_event_type = _tp_info(
                events=result.events,
                beats=result.beats,
                simulation_end_tick=context.simulation_end_tick,
            )
            by_agent[agent] = SearchOutcome(
                seed=seed,
                strategy=strategy,
                valid=bool(result.validation.valid),
                q_score=float(q_score),
                tp_position=tp_position,
                tp_event_type=tp_event_type,
                dev_beat_count=_dev_beat_count(result.beats),
                total_beats=len(result.beats),
                event_ids=[str(event.id) for event in result.events],
                beats=[str(beat.value) for beat in result.beats],
                violations=[str(v) for v in result.validation.violations],
                total_candidates=0,
                candidates_with_dev_beats=0,
                anchor_positions=[],
                available_midrange_anchors=0,
            )

        valid_scores: list[float] = []
        valid_count = 0
        for agent in DINNER_PARTY_AGENTS:
            row = by_agent[agent]
            if row.valid:
                valid_count += 1
                per_agent_valid_count[agent] += 1
                valid_scores.append(float(row.q_score))

        mean_q = float(_mean(valid_scores))
        va = float(mean_q * (valid_count / float(len(DINNER_PARTY_AGENTS)))) if valid_count > 0 else 0.0
        seed_mean_q.append(mean_q)
        seed_va.append(va)

        if valid_count == len(DINNER_PARTY_AGENTS):
            all_valid_count += 1
        if not by_agent["diana"].valid:
            diana_invalid_count += 1

    total = len(seed_contexts)
    return FullExtractionResult(
        strategy=strategy,
        mean_q=float(_mean(seed_mean_q)),
        va=float(_mean(seed_va)),
        all_valid_rate=float(all_valid_count / total) if total > 0 else 0.0,
        per_agent_validity={
            agent: (float(per_agent_valid_count[agent] / total) if total > 0 else 0.0)
            for agent in DINNER_PARTY_AGENTS
        },
        diana_invalid_count=int(diana_invalid_count),
    )


def _interpretation(
    *,
    default_result: StrategyResult,
    diversified_result: StrategyResult,
) -> dict[str, Any]:
    default_invalid = set(int(seed) for seed in default_result.diana_invalid_seeds)
    diversified_valid = {int(detail.seed) for detail in diversified_result.focal_seeds_detail if detail.valid}
    fixed = int(len(default_invalid & diversified_valid))
    total = int(len(FOCAL_SEEDS))

    if fixed == total:
        return {
            "starvation_fully_explained": True,
            "description": (
                "Diversified anchors fix all 9 focal failures, indicating anchor-selection bias "
                "fully explains Diana starvation in these seeds."
            ),
            "focal_fixed_count": fixed,
            "focal_total": total,
        }
    if 6 <= fixed <= 8:
        return {
            "starvation_fully_explained": False,
            "description": (
                f"Diversified anchors fix {fixed}/{total} focal failures. Anchor bias explains most "
                "failures, with residual seeds likely constrained by local causal structure."
            ),
            "focal_fixed_count": fixed,
            "focal_total": total,
        }
    return {
        "starvation_fully_explained": False,
        "description": (
            f"Diversified anchors fix only {fixed}/{total} focal failures. Starvation is not fully "
            "explained by anchor bias; deeper structural limitations remain."
        ),
        "focal_fixed_count": fixed,
        "focal_total": total,
    }


def _detail_cell(detail: FocalSeedDetail) -> str:
    tp = "na" if detail.tp_position is None else f"{float(detail.tp_position):.3f}"
    if detail.valid:
        return f"VALID (tp={tp}, dev={int(detail.dev_beat_count)})"
    return f"INVALID (tp={tp}, dev={int(detail.dev_beat_count)})"


def _build_summary_markdown(
    *,
    diana_results: dict[str, StrategyResult],
    full_results: dict[str, FullExtractionResult],
    interpretation: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Anchor Diversification Experiment")
    lines.append("")
    lines.append("## Diana Validity by Strategy")
    lines.append("")
    lines.append("| Strategy | Diana Valid | Diana Invalid | Mean Q | Mean TP Pos | Mean Dev Beats |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in (
        ("default", "default"),
        ("diversified", "diversified (2 mid)"),
        ("all_midrange", "all_midrange"),
    ):
        row = diana_results[key]
        lines.append(
            "| "
            f"{label} | "
            f"{int(row.diana_valid_count)}/50 | "
            f"{int(row.diana_invalid_count)}/50 | "
            f"{float(row.diana_mean_q):.3f} | "
            f"{float(row.diana_mean_tp_position):.3f} | "
            f"{float(row.diana_mean_dev_beats):.2f} |"
        )
    lines.append("")

    lines.append("## Focal Seed Detail (Originally-Failing Seeds)")
    lines.append("")
    lines.append("| Seed | Default | Diversified | All-Midrange |")
    lines.append("|---|---|---|---|")
    by_strategy_seed: dict[str, dict[int, FocalSeedDetail]] = {
        key: {int(row.seed): row for row in value.focal_seeds_detail}
        for key, value in diana_results.items()
    }
    for seed in FOCAL_SEEDS:
        default_cell = _detail_cell(by_strategy_seed["default"][seed])
        diversified_cell = _detail_cell(by_strategy_seed["diversified"][seed])
        all_mid_cell = _detail_cell(by_strategy_seed["all_midrange"][seed])
        lines.append(f"| {seed} | {default_cell} | {diversified_cell} | {all_mid_cell} |")
    lines.append("")

    lines.append("## Full 6-Agent Extraction")
    lines.append("")
    lines.append("| Strategy | Mean Q | VA | All-Valid | Diana Valid % |")
    lines.append("|---|---|---|---|---|")
    for key, label in (
        ("default", "default"),
        ("diversified", "diversified"),
        ("all_midrange", "all_midrange"),
    ):
        row = full_results[key]
        diana_valid_rate = 1.0 - float(row.diana_invalid_count / 50.0)
        lines.append(
            "| "
            f"{label} | "
            f"{float(row.mean_q):.3f} | "
            f"{float(row.va):.3f} | "
            f"{float(row.all_valid_rate) * 100.0:.1f}% | "
            f"{diana_valid_rate * 100.0:.1f}% |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append(
        f"- Focal seeds fixed by diversification: {int(interpretation['focal_fixed_count'])}/{int(interpretation['focal_total'])}"
    )
    lines.append(
        "- Starvation fully explained by anchor bias: "
        f"{'yes' if bool(interpretation['starvation_fully_explained']) else 'no'}"
    )
    lines.append(f"- {str(interpretation['description'])}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor diversification experiment (depth-2, full evolution).")
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
    print("=== ANCHOR DIVERSIFICATION EXPERIMENT ===")
    print()
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print("Strategies: default, diversified (>=2 midrange anchors), all_midrange")
    print(f"Focal seeds: {FOCAL_SEEDS}")
    print()

    prep_start = time.time()
    seed_contexts = _prepare_seed_contexts(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    print(f"Prepared depth-2 contexts in {time.time() - prep_start:.1f}s.")
    print()

    strategies: list[StrategyName] = ["default", "diversified", "all_midrange"]
    diana_results: dict[str, StrategyResult] = {}
    full_results: dict[str, FullExtractionResult] = {}
    diana_outcomes_by_strategy: dict[str, dict[int, SearchOutcome]] = {}
    default_parity_mismatches: list[dict[str, Any]] = []

    for strategy in strategies:
        print(f"[strategy={strategy}] evaluating Diana outcomes...", flush=True)
        per_seed_outcomes: dict[int, SearchOutcome] = {}

        for context in seed_contexts:
            outcome = _search_with_anchor_strategy(
                seed=int(context.seed),
                all_events=context.events,
                protagonist="diana",
                total_sim_time=context.total_sim_time,
                simulation_end_tick=context.simulation_end_tick,
                strategy=strategy,
            )
            per_seed_outcomes[int(context.seed)] = outcome

            if strategy == "default":
                direct = search_arc(
                    all_events=context.events,
                    protagonist="diana",
                    max_events=20,
                    total_sim_time=context.total_sim_time,
                    grammar_config=None,
                )
                direct_ids = [str(event.id) for event in direct.events]
                direct_beats = [str(beat.value) for beat in direct.beats]
                if outcome.event_ids != direct_ids or outcome.beats != direct_beats:
                    default_parity_mismatches.append(
                        {
                            "seed": int(context.seed),
                            "instrumented_event_ids": list(outcome.event_ids),
                            "direct_event_ids": direct_ids,
                            "instrumented_beats": list(outcome.beats),
                            "direct_beats": direct_beats,
                        }
                    )

        diana_outcomes_by_strategy[strategy] = per_seed_outcomes
        strategy_result = _strategy_result(
            strategy=strategy,
            outcomes=[per_seed_outcomes[int(context.seed)] for context in seed_contexts],
        )
        diana_results[strategy] = strategy_result

        print(f"[strategy={strategy}] evaluating full 6-agent extraction...", flush=True)
        full_results[strategy] = _full_extraction_result(
            strategy=strategy,
            seed_contexts=seed_contexts,
            diana_outcomes_by_seed=per_seed_outcomes,
        )

    expected_invalid = sorted(int(seed) for seed in FOCAL_SEEDS)
    default_invalid = sorted(int(seed) for seed in diana_results["default"].diana_invalid_seeds)
    if default_invalid != expected_invalid:
        raise RuntimeError(
            "Default strategy baseline mismatch. "
            f"Expected invalid seeds {expected_invalid}, got {default_invalid}."
        )

    if default_parity_mismatches:
        raise RuntimeError(
            f"Default strategy did not match direct search_arc for {len(default_parity_mismatches)} seeds."
        )

    diversified_details = {int(row.seed): row for row in diana_results["diversified"].focal_seeds_detail}
    diversified_midrange_shortfalls: list[int] = []
    diversified_midrange_failures: list[int] = []
    for seed in FOCAL_SEEDS:
        detail = diversified_details[seed]
        in_window = [
            pos for pos in detail.anchor_positions
            if MIDRANGE_LOWER - EPSILON <= float(pos) <= MIDRANGE_UPPER + EPSILON
        ]
        if len(in_window) < MIN_MIDRANGE_ANCHORS:
            if int(detail.available_midrange_anchors) < MIN_MIDRANGE_ANCHORS:
                diversified_midrange_shortfalls.append(seed)
            else:
                diversified_midrange_failures.append(seed)
    if diversified_midrange_failures:
        raise RuntimeError(
            "Diversified strategy failed to include required midrange anchors for focal seeds: "
            f"{diversified_midrange_failures}"
        )

    all_mid_details = {int(row.seed): row for row in diana_results["all_midrange"].focal_seeds_detail}
    all_midrange_window_failures: list[int] = []
    all_midrange_shortfalls: list[int] = []
    for seed in FOCAL_SEEDS:
        detail = all_mid_details[seed]
        positions = [float(pos) for pos in detail.anchor_positions]
        in_window = [
            pos for pos in positions
            if MIDRANGE_LOWER - EPSILON <= float(pos) <= MIDRANGE_UPPER + EPSILON
        ]
        if len(in_window) != len(positions):
            all_midrange_window_failures.append(seed)
        if int(detail.available_midrange_anchors) < MAX_ANCHORS:
            all_midrange_shortfalls.append(seed)
    if all_midrange_window_failures:
        raise RuntimeError(
            "All-midrange strategy selected anchors outside the midrange window for seeds: "
            f"{all_midrange_window_failures}"
        )

    interpretation = _interpretation(
        default_result=diana_results["default"],
        diversified_result=diana_results["diversified"],
    )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": [int(seed) for seed in seeds],
            "experiment": "anchor_diversification",
            "strategies": list(strategies),
            "focal_seeds": [int(seed) for seed in FOCAL_SEEDS],
            "midrange_window": [float(MIDRANGE_LOWER), float(MIDRANGE_UPPER)],
            "min_midrange_anchors": int(MIN_MIDRANGE_ANCHORS),
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
        },
        "diana_only": {
            "default": asdict(diana_results["default"]),
            "diversified": asdict(diana_results["diversified"]),
            "all_midrange": asdict(diana_results["all_midrange"]),
        },
        "full_extraction": {
            "default": asdict(full_results["default"]),
            "diversified": asdict(full_results["diversified"]),
            "all_midrange": asdict(full_results["all_midrange"]),
        },
        "verification": {
            "default_invalid_seed_expected": expected_invalid,
            "default_invalid_seed_observed": default_invalid,
            "default_parity_mismatch_count": int(len(default_parity_mismatches)),
            "diversified_midrange_shortfalls_due_to_availability": diversified_midrange_shortfalls,
            "all_midrange_anchor_shortfalls_due_to_availability": all_midrange_shortfalls,
        },
        "interpretation": interpretation,
    }

    summary_text = _build_summary_markdown(
        diana_results=diana_results,
        full_results=full_results,
        interpretation=interpretation,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(
        "Default full-extraction check: "
        f"mean_q={full_results['default'].mean_q:.3f}, "
        f"va={full_results['default'].va:.3f}, "
        f"all_valid={full_results['default'].all_valid_rate * 100.0:.1f}%, "
        f"diana_valid={(1.0 - full_results['default'].diana_invalid_count / len(seed_contexts)) * 100.0:.1f}%"
    )


if __name__ == "__main__":
    main()
