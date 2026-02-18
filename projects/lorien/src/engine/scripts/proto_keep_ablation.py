"""Proto-keep ablation: pool injection vs causal reachability.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.proto_keep_ablation
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
from narrativefield.extraction.arc_search import _Candidate
from narrativefield.extraction.arc_validator import validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.schema.events import BeatType, Event, EventType
from scripts.feasible_volume_sweep import _prepare_seed_contexts

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "proto_keep_ablation.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "proto_keep_ablation_summary.md"
DEFAULT_SEEDS = list(range(1, 51))
FOCAL_SEEDS = [2, 3, 9, 25, 32, 33, 35, 38, 43]
EXPECTED_BASELINE_INVALID = [2, 3, 9, 25, 32, 33, 35, 38, 43]

MAX_ANCHORS = 8
MAX_EVENTS = 20
PROTO_TOPK = 20 * 2
MIDRANGE_LOWER = 0.30
MIDRANGE_UPPER = 0.70
EARLY_CUTOFF = 0.20

PoolMode = Literal["baseline", "no_proto_inject", "temporal_filter", "no_proto_inject_deep_bfs"]
POOL_MODES: list[PoolMode] = [
    "baseline",
    "no_proto_inject",
    "temporal_filter",
    "no_proto_inject_deep_bfs",
]
EPSILON = 1e-12


@dataclass
class FocalSeedPoolDetail:
    seed: int
    pool_mode: str
    valid: bool
    q_score: float | None
    tp_position: float | None
    dev_beat_count: int
    total_candidates: int
    candidates_with_dev_beats: int
    early_catastrophe_in_pool: bool
    earliest_event_position: float
    pool_size_per_anchor: list[int]


@dataclass
class PoolAblationResult:
    pool_mode: str
    diana_valid_count: int
    diana_invalid_count: int
    diana_invalid_seeds: list[int]
    diana_mean_q: float
    diana_mean_tp_position: float
    diana_mean_dev_beats: float
    focal_seeds_detail: list[FocalSeedPoolDetail]


@dataclass
class SearchOutcome:
    seed: int
    pool_mode: str
    valid: bool
    q_score: float
    tp_position: float | None
    tp_event_type: str | None
    dev_beat_count: int
    total_beats: int
    total_candidates: int
    candidates_with_dev_beats: int
    early_catastrophe_in_pool: bool
    earliest_event_position: float
    pool_size_per_anchor: list[int]
    anchor_positions: list[float]
    available_midrange_anchors: int
    selected_event_ids: list[str]
    selected_event_positions: list[float]
    selected_event_types: list[str]
    beats: list[str]
    violations: list[str]
    neighborhood_union_ids: list[str]
    proto_keep_ids: list[str]
    proto_keep_positions: list[float]
    bridge_injected_event_ids: list[str]
    bridge_injected_positions: list[float]


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


def _select_all_midrange_anchors(
    *,
    scored_events: list[tuple[float, Event]],
    protagonist: str,
    simulation_end_tick: int,
    max_anchors: int = MAX_ANCHORS,
) -> tuple[list[Event], int]:
    protagonist_scored = [(importance, event) for importance, event in scored_events if arc_search_core._involves(event, protagonist)]
    selection_pool = protagonist_scored if protagonist_scored else scored_events

    midrange = [
        (importance, event)
        for importance, event in selection_pool
        if MIDRANGE_LOWER <= _event_pos(event, simulation_end_tick) <= MIDRANGE_UPPER
    ]
    midrange.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))
    anchors = [event for _, event in midrange[:max_anchors]]
    return (anchors, int(len(midrange)))


def _baseline_proto_keep_ids(
    *,
    window: list[Event],
    protagonist: str,
    simulation_end_tick: int,
) -> tuple[set[str], list[float]]:
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

    by_id = {event.id: event for event in window}
    positions = sorted(
        _event_pos(by_id[event_id], simulation_end_tick)
        for event_id in proto_keep_ids
        if event_id in by_id
    )
    return (proto_keep_ids, positions)


def _temporal_filtered_proto_keep_ids(
    *,
    window: list[Event],
    protagonist: str,
    simulation_end_tick: int,
) -> tuple[set[str], list[float]]:
    proto_events_sorted = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (-arc_search_core._event_importance(event), event.sim_time, event.id),
    )
    proto_keep_ids: set[str] = set()
    for event in proto_events_sorted[:PROTO_TOPK]:
        if _event_pos(event, simulation_end_tick) >= EARLY_CUTOFF:
            proto_keep_ids.add(event.id)

    proto_by_time = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick),
    )
    for event in proto_by_time[-3:]:
        if _event_pos(event, simulation_end_tick) >= EARLY_CUTOFF:
            proto_keep_ids.add(event.id)

    by_id = {event.id: event for event in window}
    positions = sorted(
        _event_pos(by_id[event_id], simulation_end_tick)
        for event_id in proto_keep_ids
        if event_id in by_id
    )
    return (proto_keep_ids, positions)


def _search_with_pool_strategy(
    *,
    seed: int,
    all_events: list[Event],
    protagonist: str,
    total_sim_time: float | None,
    simulation_end_tick: int,
    pool_mode: PoolMode,
) -> SearchOutcome:
    window = list(all_events)
    window.sort(key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick))

    if not window:
        return SearchOutcome(
            seed=int(seed),
            pool_mode=pool_mode,
            valid=False,
            q_score=0.0,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            total_candidates=0,
            candidates_with_dev_beats=0,
            early_catastrophe_in_pool=False,
            earliest_event_position=0.0,
            pool_size_per_anchor=[],
            anchor_positions=[],
            available_midrange_anchors=0,
            selected_event_ids=[],
            selected_event_positions=[],
            selected_event_types=[],
            beats=[],
            violations=["No events in time window"],
            neighborhood_union_ids=[],
            proto_keep_ids=[],
            proto_keep_positions=[],
            bridge_injected_event_ids=[],
            bridge_injected_positions=[],
        )

    by_id: dict[str, Event] = {event.id: event for event in all_events}
    reverse_links = arc_search_core._build_reverse_links(all_events)
    window_ids = {event.id for event in window}

    anchor_scored = _scored_anchor_pool(window)
    anchors, available_midrange = _select_all_midrange_anchors(
        scored_events=anchor_scored,
        protagonist=protagonist,
        simulation_end_tick=simulation_end_tick,
        max_anchors=MAX_ANCHORS,
    )
    anchor_positions = [_event_pos(event, simulation_end_tick) for event in anchors]

    proto_keep_ids: set[str] = set()
    proto_keep_positions: list[float] = []
    if pool_mode == "baseline":
        proto_keep_ids, proto_keep_positions = _baseline_proto_keep_ids(
            window=window,
            protagonist=protagonist,
            simulation_end_tick=simulation_end_tick,
        )
    elif pool_mode == "temporal_filter":
        proto_keep_ids, proto_keep_positions = _temporal_filtered_proto_keep_ids(
            window=window,
            protagonist=protagonist,
            simulation_end_tick=simulation_end_tick,
        )

    bfs_depth = 5 if pool_mode == "no_proto_inject_deep_bfs" else 3
    pool_size_per_anchor: list[int] = []
    bridge_injected_ids: set[str] = set()
    neighborhood_union_ids: set[str] = set()
    candidates: list[_Candidate] = []

    for anchor in anchors:
        neighborhood = arc_search_core._causal_neighborhood(
            anchor_id=anchor.id,
            by_id=by_id,
            reverse_links=reverse_links,
            allowed_ids=window_ids,
            max_depth=bfs_depth,
        )
        neighborhood_union_ids.update(neighborhood)

        if pool_mode in {"baseline", "temporal_filter"}:
            pool_ids = {anchor.id} | neighborhood | proto_keep_ids
        else:
            pool_ids = {anchor.id} | neighborhood

        pool_size_per_anchor.append(int(len(pool_ids)))
        pool = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        pool.sort(key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick))

        selected = arc_search_core._downsample_preserving_continuity(
            pool=pool,
            bridge_pool=window,
            protagonist=protagonist,
            anchor_id=anchor.id,
            max_events=MAX_EVENTS,
        )
        selected_ids = {event.id for event in selected}
        bridge_injected_ids.update(selected_ids - set(pool_ids))

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

    if not candidates:
        return SearchOutcome(
            seed=int(seed),
            pool_mode=pool_mode,
            valid=False,
            q_score=0.0,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            total_candidates=0,
            candidates_with_dev_beats=0,
            early_catastrophe_in_pool=False,
            earliest_event_position=0.0,
            pool_size_per_anchor=pool_size_per_anchor,
            anchor_positions=[float(pos) for pos in anchor_positions],
            available_midrange_anchors=int(available_midrange),
            selected_event_ids=[],
            selected_event_positions=[],
            selected_event_types=[],
            beats=[],
            violations=["No anchors selected"],
            neighborhood_union_ids=sorted(neighborhood_union_ids),
            proto_keep_ids=sorted(proto_keep_ids),
            proto_keep_positions=[float(pos) for pos in proto_keep_positions],
            bridge_injected_event_ids=sorted(bridge_injected_ids),
            bridge_injected_positions=[],
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

    selected_positions = [_event_pos(event, simulation_end_tick) for event in best.events]
    earliest_position = min(selected_positions) if selected_positions else 0.0
    early_event = any(float(pos) < EARLY_CUTOFF for pos in selected_positions)

    bridge_injected_positions = sorted(
        _event_pos(by_id[event_id], simulation_end_tick)
        for event_id in bridge_injected_ids
        if event_id in by_id
    )

    return SearchOutcome(
        seed=int(seed),
        pool_mode=pool_mode,
        valid=bool(best.validation.valid),
        q_score=float(q_score),
        tp_position=tp_position,
        tp_event_type=tp_event_type,
        dev_beat_count=_dev_beat_count(best.beats),
        total_beats=len(best.beats),
        total_candidates=int(len(candidates)),
        candidates_with_dev_beats=int(candidates_with_dev_beats),
        early_catastrophe_in_pool=bool(early_event),
        earliest_event_position=float(earliest_position),
        pool_size_per_anchor=[int(x) for x in pool_size_per_anchor],
        anchor_positions=[float(pos) for pos in anchor_positions],
        available_midrange_anchors=int(available_midrange),
        selected_event_ids=[str(event.id) for event in best.events],
        selected_event_positions=[float(pos) for pos in selected_positions],
        selected_event_types=[str(event.type.value) for event in best.events],
        beats=[str(beat.value) for beat in best.beats],
        violations=[str(v) for v in best.validation.violations],
        neighborhood_union_ids=sorted(neighborhood_union_ids),
        proto_keep_ids=sorted(proto_keep_ids),
        proto_keep_positions=[float(pos) for pos in proto_keep_positions],
        bridge_injected_event_ids=sorted(bridge_injected_ids),
        bridge_injected_positions=[float(pos) for pos in bridge_injected_positions],
    )


def _build_result(
    *,
    pool_mode: PoolMode,
    outcomes: list[SearchOutcome],
) -> PoolAblationResult:
    valid = [row for row in outcomes if row.valid]
    invalid_seeds = sorted(int(row.seed) for row in outcomes if not row.valid)
    focal_set = set(int(seed) for seed in FOCAL_SEEDS)
    focal_details = [
        FocalSeedPoolDetail(
            seed=int(row.seed),
            pool_mode=pool_mode,
            valid=bool(row.valid),
            q_score=float(row.q_score) if row.valid else None,
            tp_position=float(row.tp_position) if row.tp_position is not None else None,
            dev_beat_count=int(row.dev_beat_count),
            total_candidates=int(row.total_candidates),
            candidates_with_dev_beats=int(row.candidates_with_dev_beats),
            early_catastrophe_in_pool=bool(row.early_catastrophe_in_pool),
            earliest_event_position=float(row.earliest_event_position),
            pool_size_per_anchor=[int(x) for x in row.pool_size_per_anchor],
        )
        for row in outcomes
        if int(row.seed) in focal_set
    ]
    focal_details.sort(key=lambda row: int(row.seed))

    return PoolAblationResult(
        pool_mode=pool_mode,
        diana_valid_count=int(len(valid)),
        diana_invalid_count=int(len(outcomes) - len(valid)),
        diana_invalid_seeds=invalid_seeds,
        diana_mean_q=float(_mean([float(row.q_score) for row in valid])),
        diana_mean_tp_position=float(_mean([float(row.tp_position) for row in valid if row.tp_position is not None])),
        diana_mean_dev_beats=float(_mean([float(row.dev_beat_count) for row in valid])),
        focal_seeds_detail=focal_details,
    )


def _tp_text(row: FocalSeedPoolDetail) -> str:
    if row.tp_position is None:
        return "na"
    return f"{float(row.tp_position):.3f}"


def _mechanism_interpretation(
    *,
    results: dict[str, PoolAblationResult],
) -> dict[str, Any]:
    focal_set = set(int(seed) for seed in FOCAL_SEEDS)
    baseline_invalid = set(int(seed) for seed in results["baseline"].diana_invalid_seeds)
    target_focal = baseline_invalid & focal_set

    no_inject_detail = {int(row.seed): row for row in results["no_proto_inject"].focal_seeds_detail}
    temporal_detail = {int(row.seed): row for row in results["temporal_filter"].focal_seeds_detail}
    deep_detail = {int(row.seed): row for row in results["no_proto_inject_deep_bfs"].focal_seeds_detail}

    focal_fixed_no_inject = int(sum(1 for seed in target_focal if bool(no_inject_detail[seed].valid)))
    focal_fixed_temporal = int(sum(1 for seed in target_focal if bool(temporal_detail[seed].valid)))
    focal_fixed_deep_bfs = int(sum(1 for seed in target_focal if bool(deep_detail[seed].valid)))
    deep_bfs_early_tp = int(
        sum(
            1
            for seed in target_focal
            if deep_detail[seed].tp_position is not None and float(deep_detail[seed].tp_position) < EARLY_CUTOFF
        )
    )
    deep_bfs_early_event = int(sum(1 for seed in target_focal if bool(deep_detail[seed].early_catastrophe_in_pool)))

    if focal_fixed_no_inject >= 7:
        proto_is_sole = True
        causal_contributes = False
        description = (
            f"Removing proto_keep_ids injection fixes {focal_fixed_no_inject}/9 seeds. "
            "Pool injection is the primary contamination vector."
        )
    elif focal_fixed_no_inject >= 3:
        proto_is_sole = False
        causal_contributes = True
        description = (
            f"Mixed mechanism: proto_keep_ids explains {focal_fixed_no_inject}/9 seeds, "
            "causal reachability explains the remainder."
        )
    else:
        proto_is_sole = False
        causal_contributes = True
        description = (
            "Proto_keep_ids is NOT the primary vector. Causal graph structure independently funnels "
            "early catastrophes into mid-arc BFS neighborhoods."
        )

    if focal_fixed_deep_bfs < focal_fixed_no_inject:
        description += " Deep BFS (depth-5) recovers causal paths to early events that depth-3 misses."
    if focal_fixed_temporal >= 7 and focal_fixed_no_inject < 3:
        description += (
            f" Temporal filtering still fixes {focal_fixed_temporal}/9 focal seeds, indicating that early "
            "proto_keep_ids injection is high-leverage even though complete proto removal over-prunes the search."
        )

    return {
        "proto_keep_ids_is_sole_vector": bool(proto_is_sole),
        "causal_reachability_contributes": bool(causal_contributes),
        "description": description,
        "focal_fixed_no_inject": int(focal_fixed_no_inject),
        "focal_fixed_temporal_filter": int(focal_fixed_temporal),
        "focal_fixed_no_inject_deep_bfs": int(focal_fixed_deep_bfs),
        "deep_bfs_early_tp_count": int(deep_bfs_early_tp),
        "deep_bfs_early_event_count": int(deep_bfs_early_event),
    }


def _build_summary_markdown(
    *,
    results: dict[str, PoolAblationResult],
    interpretation: dict[str, Any],
) -> str:
    baseline_focal_invalid = {int(seed) for seed in results["baseline"].diana_invalid_seeds if int(seed) in set(FOCAL_SEEDS)}

    def focal_fixed(mode: str) -> int:
        details = {int(row.seed): row for row in results[mode].focal_seeds_detail}
        return int(sum(1 for seed in baseline_focal_invalid if bool(details[seed].valid)))

    details_by_mode: dict[str, dict[int, FocalSeedPoolDetail]] = {
        mode: {int(row.seed): row for row in results[mode].focal_seeds_detail}
        for mode in POOL_MODES
    }

    lines: list[str] = []
    lines.append("# Proto-Keep-IDs Ablation")
    lines.append("")
    lines.append("## Diana Validity by Pool Mode")
    lines.append("")
    lines.append("| Pool Mode | Diana Valid | Diana Invalid | Focal Fixed | Mean Q | Mean TP Pos |")
    lines.append("|---|---|---|---|---|---|")
    for mode, label in (
        ("baseline", "baseline (all-mid + proto)"),
        ("no_proto_inject", "no_proto_inject"),
        ("temporal_filter", "temporal_filter"),
        ("no_proto_inject_deep_bfs", "no_proto_inject_deep_bfs"),
    ):
        row = results[mode]
        lines.append(
            "| "
            f"{label} | "
            f"{int(row.diana_valid_count)}/50 | "
            f"{int(row.diana_invalid_count)}/50 | "
            f"{focal_fixed(mode)}/9 | "
            f"{float(row.diana_mean_q):.3f} | "
            f"{float(row.diana_mean_tp_position):.3f} |"
        )
    lines.append("")

    lines.append("## Focal Seed Detail")
    lines.append("")
    lines.append("| Seed | Baseline TP | No-Inject TP | Temporal TP | Deep-BFS TP | Early Event In Pool? |")
    lines.append("|---|---|---|---|---|---|")
    for seed in FOCAL_SEEDS:
        b = details_by_mode["baseline"][seed]
        n = details_by_mode["no_proto_inject"][seed]
        t = details_by_mode["temporal_filter"][seed]
        d = details_by_mode["no_proto_inject_deep_bfs"][seed]
        early_flags = (
            f"B:{'Y' if b.early_catastrophe_in_pool else 'N'}, "
            f"N:{'Y' if n.early_catastrophe_in_pool else 'N'}, "
            f"T:{'Y' if t.early_catastrophe_in_pool else 'N'}, "
            f"D:{'Y' if d.early_catastrophe_in_pool else 'N'}"
        )
        lines.append(
            f"| {seed} | {_tp_text(b)} | {_tp_text(n)} | {_tp_text(t)} | {_tp_text(d)} | {early_flags} |"
        )
    lines.append("")

    lines.append("## Mechanism Determination")
    lines.append(
        "- Proto-keep-ids is sole contamination vector: "
        f"{'yes' if bool(interpretation['proto_keep_ids_is_sole_vector']) else 'no'}"
    )
    lines.append(
        "- Causal reachability also reaches early events: "
        f"{'yes' if bool(interpretation['causal_reachability_contributes']) else 'no'}"
    )
    lines.append(f"- Interpretation: {str(interpretation['description'])}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Proto-keep ablation (all-midrange anchors, depth-2 full evolution).")
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
    print("=== PROTO-KEEP ABLATION ===")
    print()
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print("Pool modes: baseline, no_proto_inject, temporal_filter, no_proto_inject_deep_bfs")
    print()

    prep_start = time.time()
    seed_contexts = _prepare_seed_contexts(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    print(f"Prepared depth-2 contexts in {time.time() - prep_start:.1f}s.")
    print()

    outcomes_by_mode: dict[str, dict[int, SearchOutcome]] = {}
    results: dict[str, PoolAblationResult] = {}

    for mode in POOL_MODES:
        print(f"[mode={mode}] evaluating Diana outcomes...", flush=True)
        per_seed: dict[int, SearchOutcome] = {}
        for context in seed_contexts:
            outcome = _search_with_pool_strategy(
                seed=int(context.seed),
                all_events=context.events,
                protagonist="diana",
                total_sim_time=context.total_sim_time,
                simulation_end_tick=context.simulation_end_tick,
                pool_mode=mode,
            )
            per_seed[int(context.seed)] = outcome
        outcomes_by_mode[mode] = per_seed
        ordered = [per_seed[int(context.seed)] for context in seed_contexts]
        results[mode] = _build_result(pool_mode=mode, outcomes=ordered)

    baseline_invalid = sorted(int(seed) for seed in results["baseline"].diana_invalid_seeds)
    if baseline_invalid != EXPECTED_BASELINE_INVALID:
        raise RuntimeError(
            "Baseline mismatch against Experiment C all-midrange seeds. "
            f"Expected {EXPECTED_BASELINE_INVALID}, got {baseline_invalid}."
        )

    baseline_tp_compare: dict[str, Any] = {
        "available": False,
        "mismatch_count": 0,
        "mismatches": [],
    }
    anchor_div_path = OUTPUT_DIR / "anchor_diversification.json"
    if anchor_div_path.exists():
        raw = json.loads(anchor_div_path.read_text(encoding="utf-8"))
        prior = list(((raw.get("diana_only") or {}).get("all_midrange") or {}).get("focal_seeds_detail") or [])
        prior_by_seed = {int(row["seed"]): row for row in prior}
        mismatches: list[dict[str, Any]] = []
        current_by_seed = {int(row.seed): row for row in results["baseline"].focal_seeds_detail}
        for seed in FOCAL_SEEDS:
            old_tp = prior_by_seed.get(seed, {}).get("tp_position")
            new_tp = current_by_seed[seed].tp_position
            old_val = None if old_tp is None else float(old_tp)
            new_val = None if new_tp is None else float(new_tp)
            same = (
                old_val is None and new_val is None
            ) or (
                old_val is not None and new_val is not None and abs(old_val - new_val) <= EPSILON
            )
            if not same:
                mismatches.append(
                    {"seed": int(seed), "prior_tp": old_val, "current_tp": new_val}
                )
        baseline_tp_compare = {
            "available": True,
            "mismatch_count": int(len(mismatches)),
            "mismatches": mismatches,
            "reference_file": str(anchor_div_path),
        }
        if mismatches:
            raise RuntimeError(
                "Baseline TP positions do not match prior all-midrange results: "
                f"{mismatches}"
            )

    # Verification (2): baseline early IDs in no-inject must be absent or reachable via BFS neighborhood.
    baseline_to_no_inject: list[dict[str, Any]] = []
    non_bfs_violations: list[int] = []
    for seed in FOCAL_SEEDS:
        base = outcomes_by_mode["baseline"][seed]
        no_inject = outcomes_by_mode["no_proto_inject"][seed]
        early_ids = [
            event_id
            for event_id, pos in zip(base.selected_event_ids, base.selected_event_positions)
            if float(pos) < EARLY_CUTOFF
        ]
        no_inject_ids = set(no_inject.selected_event_ids)
        present = [event_id for event_id in early_ids if event_id in no_inject_ids]
        bfs_ids = set(no_inject.neighborhood_union_ids)
        present_via_bfs = [event_id for event_id in present if event_id in bfs_ids]
        present_not_bfs = [event_id for event_id in present if event_id not in bfs_ids]
        baseline_to_no_inject.append(
            {
                "seed": int(seed),
                "baseline_early_event_ids": early_ids,
                "present_in_no_inject_arc": present,
                "present_via_bfs": present_via_bfs,
                "present_without_bfs_explanation": present_not_bfs,
            }
        )
        if present_not_bfs:
            non_bfs_violations.append(int(seed))
    if non_bfs_violations:
        raise RuntimeError(
            "No-inject mode contains baseline early events not explained by BFS reachability: "
            f"{non_bfs_violations}"
        )

    # Verification (3): temporal filter must not inject events earlier than cutoff.
    temporal_injection_violations: list[int] = []
    for seed in seeds:
        positions = outcomes_by_mode["temporal_filter"][int(seed)].proto_keep_positions
        if any(float(pos) < EARLY_CUTOFF - EPSILON for pos in positions):
            temporal_injection_violations.append(int(seed))
    if temporal_injection_violations:
        raise RuntimeError(
            "Temporal filter injected events below early cutoff for seeds: "
            f"{temporal_injection_violations}"
        )

    bridge_injection_counts = {
        mode: int(
            sum(
                1
                for seed in seeds
                if outcomes_by_mode[mode][int(seed)].bridge_injected_event_ids
            )
        )
        for mode in POOL_MODES
    }
    bridge_injection_focal_counts = {
        mode: int(
            sum(
                1
                for seed in FOCAL_SEEDS
                if outcomes_by_mode[mode][int(seed)].bridge_injected_event_ids
            )
        )
        for mode in POOL_MODES
    }

    interpretation = _mechanism_interpretation(results=results)
    summary_text = _build_summary_markdown(results=results, interpretation=interpretation)

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": [int(seed) for seed in seeds],
            "experiment": "proto_keep_ids_ablation",
            "pool_modes": list(POOL_MODES),
            "anchor_strategy": "all_midrange",
            "focal_seeds": [int(seed) for seed in FOCAL_SEEDS],
            "midrange_window": [float(MIDRANGE_LOWER), float(MIDRANGE_UPPER)],
            "early_cutoff": float(EARLY_CUTOFF),
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "downsample_bridge_pool_behavior": (
                "_downsample_preserving_continuity calls _repair_causal_continuity(selected, bridge_pool, ...), "
                "which can insert bridge events from bridge_pool outside the constructed pool."
            ),
        },
        "results": {
            mode: asdict(results[mode]) for mode in POOL_MODES
        },
        "verification": {
            "baseline_expected_invalid_seeds": list(EXPECTED_BASELINE_INVALID),
            "baseline_observed_invalid_seeds": baseline_invalid,
            "baseline_tp_against_anchor_diversification": baseline_tp_compare,
            "baseline_to_no_inject_early_event_trace": baseline_to_no_inject,
            "temporal_filter_injection_violations": temporal_injection_violations,
            "bridge_pool_injection_seed_counts": bridge_injection_counts,
            "bridge_pool_injection_focal_seed_counts": bridge_injection_focal_counts,
        },
        "interpretation": interpretation,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(
        "Baseline check: "
        f"Diana invalid={results['baseline'].diana_invalid_count}/50, "
        f"invalid_seeds={results['baseline'].diana_invalid_seeds}"
    )


if __name__ == "__main__":
    main()
