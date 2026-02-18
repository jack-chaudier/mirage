"""Instrument arc-search candidate pools for depth-2 full-evolution runs.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.search_candidate_logger
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from narrativefield.extraction import arc_search as arc_search_core
from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import (
    ArcSearchDiagnostics,
    ArcSearchResult,
    _Candidate,
    search_arc,
)
from narrativefield.extraction.arc_validator import GrammarConfig, validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.schema.events import BeatType, Event, EventType
from scripts.feasible_volume_sweep import _prepare_seed_contexts, _search_with_config, _tp_global_positions
from scripts.k_sweep_experiment import DINNER_PARTY_AGENTS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "candidate_pool_data.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "candidate_pool_summary.md"
DEFAULT_SEEDS = list(range(1, 51))
EXPECTED_INVALID_SEEDS = [2, 3, 9, 25, 32, 33, 35, 38, 43]
RELAXED_DEV0_CONFIG = replace(GrammarConfig(), min_development_beats=0)
EPSILON = 1e-12


@dataclass
class CandidateRecord:
    seed: int
    agent: str
    anchor_index: int
    candidate_index: int
    q_score: float
    tp_position: float | None
    tp_event_type: str | None
    dev_beat_count: int
    total_beats: int
    strict_valid: bool
    relaxed_valid: bool
    is_search_winner: bool
    event_ids: list[str]
    beats: list[str]
    strict_violations: list[str]
    candidate_source: str


@dataclass
class WinnerRecord:
    seed: int
    agent: str
    q_score: float
    tp_position: float | None
    tp_event_type: str | None
    dev_beat_count: int
    total_beats: int
    strict_valid: bool
    relaxed_valid: bool
    event_ids: list[str]
    beats: list[str]
    strict_violations: list[str]


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


def _tp_info(
    *,
    events: list[Event],
    beats: list[BeatType],
    simulation_end_tick: int,
) -> tuple[float | None, str | None]:
    for event, beat in zip(events, beats):
        if beat != BeatType.TURNING_POINT:
            continue
        denom = max(int(simulation_end_tick), 1)
        return (float(int(event.tick_id) / denom), str(event.type.value))
    return (None, None)


def _candidate_to_record(
    *,
    seed: int,
    agent: str,
    candidate: _Candidate,
    anchor_index: int,
    candidate_index: int,
    simulation_end_tick: int,
    total_sim_time: float | None,
    candidate_source: str,
) -> CandidateRecord:
    q_score = float(score_arc(candidate.events, candidate.beats).composite) if candidate.events else 0.0
    tp_position, tp_event_type = _tp_info(
        events=candidate.events,
        beats=candidate.beats,
        simulation_end_tick=simulation_end_tick,
    )
    relaxed_validation = validate_arc(
        events=candidate.events,
        beats=candidate.beats,
        total_sim_time=total_sim_time,
        grammar_config=RELAXED_DEV0_CONFIG,
    )
    return CandidateRecord(
        seed=int(seed),
        agent=str(agent),
        anchor_index=int(anchor_index),
        candidate_index=int(candidate_index),
        q_score=q_score,
        tp_position=tp_position,
        tp_event_type=tp_event_type,
        dev_beat_count=_dev_beat_count(candidate.beats),
        total_beats=len(candidate.beats),
        strict_valid=bool(candidate.validation.valid),
        relaxed_valid=bool(relaxed_validation.valid),
        is_search_winner=False,
        event_ids=[str(event.id) for event in candidate.events],
        beats=[str(beat.value) for beat in candidate.beats],
        strict_violations=[str(v) for v in candidate.validation.violations],
        candidate_source=str(candidate_source),
    )


def _mark_winner(
    *,
    records: list[CandidateRecord],
    winner: _Candidate,
    winner_score: float,
    seed: int,
    agent: str,
    simulation_end_tick: int,
    total_sim_time: float | None,
) -> WinnerRecord:
    winner_event_ids = [str(event.id) for event in winner.events]
    winner_beats = [str(beat.value) for beat in winner.beats]

    winner_row: CandidateRecord | None = None
    for row in records:
        if row.event_ids == winner_event_ids and row.beats == winner_beats and not row.is_search_winner:
            row.is_search_winner = True
            winner_row = row
            break

    if winner_row is None:
        # Defensive fallback: keep output complete if winner was created outside base anchors.
        tp_position, tp_event_type = _tp_info(
            events=winner.events,
            beats=winner.beats,
            simulation_end_tick=simulation_end_tick,
        )
        relaxed_validation = validate_arc(
            events=winner.events,
            beats=winner.beats,
            total_sim_time=total_sim_time,
            grammar_config=RELAXED_DEV0_CONFIG,
        )
        winner_row = CandidateRecord(
            seed=int(seed),
            agent=str(agent),
            anchor_index=-1,
            candidate_index=-1,
            q_score=float(winner_score),
            tp_position=tp_position,
            tp_event_type=tp_event_type,
            dev_beat_count=_dev_beat_count(winner.beats),
            total_beats=len(winner.beats),
            strict_valid=bool(winner.validation.valid),
            relaxed_valid=bool(relaxed_validation.valid),
            is_search_winner=True,
            event_ids=winner_event_ids,
            beats=winner_beats,
            strict_violations=[str(v) for v in winner.validation.violations],
            candidate_source="fallback_winner_synthesized",
        )
        records.append(winner_row)

    return WinnerRecord(
        seed=int(seed),
        agent=str(agent),
        q_score=float(winner_score),
        tp_position=winner_row.tp_position,
        tp_event_type=winner_row.tp_event_type,
        dev_beat_count=int(winner_row.dev_beat_count),
        total_beats=int(winner_row.total_beats),
        strict_valid=bool(winner_row.strict_valid),
        relaxed_valid=bool(winner_row.relaxed_valid),
        event_ids=list(winner_row.event_ids),
        beats=list(winner_row.beats),
        strict_violations=list(winner_row.strict_violations),
    )


def _instrument_search_arc(
    *,
    seed: int,
    all_events: list[Event],
    protagonist: str,
    total_sim_time: float | None,
    simulation_end_tick: int,
    grammar_config: GrammarConfig | None,
) -> tuple[WinnerRecord, list[CandidateRecord], ArcSearchResult]:
    """Replica of `search_arc` with candidate-level capture."""
    window = list(all_events)
    window.sort(key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick))

    if not window:
        empty = ArcSearchResult(
            events=[],
            beats=[],
            protagonist=protagonist,
            validation=validate_arc(events=[], beats=[], total_sim_time=total_sim_time),
            diagnostics=ArcSearchDiagnostics(
                violations=["No events in time window"],
                primary_failure="No events in time window",
            ),
        )
        winner = WinnerRecord(
            seed=int(seed),
            agent=str(protagonist),
            q_score=0.0,
            tp_position=None,
            tp_event_type=None,
            dev_beat_count=0,
            total_beats=0,
            strict_valid=False,
            relaxed_valid=False,
            event_ids=[],
            beats=[],
            strict_violations=["No events in time window"],
        )
        return winner, [], empty

    agent_scores = arc_search_core._score_agents(window)
    if not protagonist:
        protagonist = max(agent_scores, key=agent_scores.get)  # type: ignore[arg-type]

    by_id: dict[str, Event] = {event.id: event for event in all_events}
    reverse_links = arc_search_core._build_reverse_links(all_events)
    window_ids = {event.id for event in window}

    anchor_scored: list[tuple[float, Event]] = []
    for event in window:
        if event.type == EventType.INTERNAL:
            continue
        if event.type in {EventType.SOCIAL_MOVE, EventType.OBSERVE} and not arc_search_core._has_meaningful_deltas(event):
            continue
        anchor_scored.append((arc_search_core._event_importance(event), event))
    anchor_scored.sort(key=lambda row: (-row[0], row[1].sim_time, row[1].id))

    anchors = [event for _, event in anchor_scored[: max(1, min(8, len(anchor_scored)))]]
    proto_anchors = [event for event in anchors if arc_search_core._involves(event, protagonist)]
    if proto_anchors:
        anchors = proto_anchors

    proto_events_sorted = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (-arc_search_core._event_importance(event), event.sim_time, event.id),
    )
    proto_keep_ids = {event.id for event in proto_events_sorted[:20 * 2]}
    proto_by_time = sorted(
        [event for event in window if arc_search_core._involves(event, protagonist) and event.type != EventType.INTERNAL],
        key=lambda event: (event.sim_time, event.tick_id, event.order_in_tick),
    )
    for event in proto_by_time[:3]:
        proto_keep_ids.add(event.id)
    for event in proto_by_time[-3:]:
        proto_keep_ids.add(event.id)

    candidates: list[_Candidate] = []
    candidate_rows: list[CandidateRecord] = []
    anchor_id_to_index: dict[str, int] = {}
    per_anchor_counts: dict[str, int] = {}

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
        if grammar_config is None:
            beats = arc_search_core._enforce_monotonic_beats(selected, beats)
            strict_validation = validate_arc(events=selected, beats=beats, total_sim_time=total_sim_time)
        else:
            beats = arc_search_core._enforce_monotonic_beats(selected, beats, grammar_config=grammar_config)
            strict_validation = validate_arc(
                events=selected,
                beats=beats,
                total_sim_time=total_sim_time,
                grammar_config=grammar_config,
            )

        strict_arc_score = score_arc(selected, beats) if strict_validation.valid else None
        candidate = _Candidate(
            events=selected,
            beats=beats,
            validation=strict_validation,
            protagonist=protagonist,
            arc_score=strict_arc_score,
            importance_sum=sum(arc_search_core._event_importance(event) for event in selected),
            anchor_id=anchor.id,
        )
        candidates.append(candidate)

        candidate_index = per_anchor_counts.get(anchor.id, 0)
        per_anchor_counts[anchor.id] = candidate_index + 1
        candidate_rows.append(
            _candidate_to_record(
                seed=seed,
                agent=protagonist,
                candidate=candidate,
                anchor_index=anchor_index,
                candidate_index=candidate_index,
                    simulation_end_tick=simulation_end_tick,
                    total_sim_time=total_sim_time,
                    candidate_source="anchor",
                )
            )

    valid_candidates = [candidate for candidate in candidates if candidate.validation.valid]
    diagnostics: ArcSearchDiagnostics | None = None

    if valid_candidates:
        with_aftermath = [candidate for candidate in valid_candidates if arc_search_core._has_post_peak_consequence(candidate.events, candidate.beats)]
        if with_aftermath:
            best = max(
                with_aftermath,
                key=lambda candidate: (
                    float(candidate.arc_score.composite) if candidate.arc_score is not None else 0.0,
                    candidate.importance_sum,
                ),
            )
        else:
            best = max(
                valid_candidates,
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
                grammar_config=grammar_config,
            )
            if widened is not None:
                best = widened
                anchor_index = anchor_id_to_index.get(best.anchor_id, -1)
                candidate_index = per_anchor_counts.get(best.anchor_id, 0)
                per_anchor_counts[best.anchor_id] = candidate_index + 1
                candidate_rows.append(
                    _candidate_to_record(
                        seed=seed,
                        agent=protagonist,
                        candidate=best,
                        anchor_index=anchor_index,
                        candidate_index=candidate_index,
                        simulation_end_tick=simulation_end_tick,
                        total_sim_time=total_sim_time,
                        candidate_source="post_peak_widened",
                    )
                )
    else:
        aftermath_candidates = [
            candidate for candidate in candidates if arc_search_core._has_post_peak_consequence(candidate.events, candidate.beats)
        ]
        candidate_pool = aftermath_candidates if aftermath_candidates else candidates
        best = min(candidate_pool, key=lambda candidate: (len(candidate.validation.violations), -candidate.importance_sum))

        rule_counts: dict[str, int] = {}
        for candidate in candidates:
            for violation in candidate.validation.violations:
                key = arc_search_core._normalize_violation(violation)
                rule_counts[key] = int(rule_counts.get(key, 0) + 1)
        keep_ids = [event.id for event in best.events]
        drop_ids = [event.id for event in window if event.id not in set(keep_ids)]
        diagnostics = ArcSearchDiagnostics(
            violations=list(best.validation.violations),
            suggested_protagonist=protagonist,
            suggested_time_window=(best.events[0].sim_time, best.events[-1].sim_time) if best.events else None,
            suggested_keep_ids=keep_ids,
            suggested_drop_ids=drop_ids[:200],
            primary_failure=(best.validation.violations[0] if best.validation.violations else ""),
            rule_failure_counts=rule_counts,
            best_candidate_violation_count=len(best.validation.violations),
            candidates_evaluated=len(candidates),
            best_candidate_violations=list(best.validation.violations),
        )

    winner_score = float(score_arc(best.events, best.beats).composite) if best.events else 0.0
    winner = _mark_winner(
        records=candidate_rows,
        winner=best,
        winner_score=winner_score,
        seed=seed,
        agent=protagonist,
        simulation_end_tick=simulation_end_tick,
        total_sim_time=total_sim_time,
    )

    search_result = ArcSearchResult(
        events=best.events,
        beats=best.beats,
        protagonist=best.protagonist,
        validation=best.validation,
        arc_score=best.arc_score,
        diagnostics=diagnostics,
    )
    return winner, candidate_rows, search_result


def _candidate_stats(
    *,
    candidate_rows: list[CandidateRecord],
    seed_set: set[int],
) -> dict[str, Any]:
    rows = [row for row in candidate_rows if int(row.seed) in seed_set]
    with_tp = [row for row in rows if row.tp_position is not None]
    dev0 = [row for row in with_tp if int(row.dev_beat_count) == 0]
    dev1p = [row for row in with_tp if int(row.dev_beat_count) >= 1]
    exploit = [
        row for row in with_tp
        if row.tp_position is not None and float(row.tp_position) < 0.25 and int(row.dev_beat_count) == 0
    ]
    winners = [row for row in rows if bool(row.is_search_winner)]

    return {
        "total_candidates": int(len(rows)),
        "candidates_with_tp": int(len(with_tp)),
        "dev0_count": int(len(dev0)),
        "dev1p_count": int(len(dev1p)),
        "dev0_pct": float((len(dev0) / len(with_tp)) * 100.0) if with_tp else 0.0,
        "dev1p_pct": float((len(dev1p) / len(with_tp)) * 100.0) if with_tp else 0.0,
        "dev0_mean_q": float(_mean([float(row.q_score) for row in dev0])),
        "dev1p_mean_q": float(_mean([float(row.q_score) for row in dev1p])),
        "q_gap_dev0_minus_dev1p": float(
            _mean([float(row.q_score) for row in dev0]) - _mean([float(row.q_score) for row in dev1p])
        ),
        "exploit_basin_count": int(len(exploit)),
        "exploit_basin_mean_q": float(_mean([float(row.q_score) for row in exploit])),
        "winner_count": int(len(winners)),
        "winner_all_dev0": bool(winners and all(int(row.dev_beat_count) == 0 for row in winners)),
        "winner_mean_tp": float(_mean([float(row.tp_position) for row in winners if row.tp_position is not None])),
        "winner_mean_q": float(_mean([float(row.q_score) for row in winners])),
    }


def _build_summary_markdown(
    *,
    invalid_stats: dict[str, Any],
    valid_stats: dict[str, Any],
    invalid_seeds: list[int],
    valid_seed_sample: list[int],
) -> str:
    lines: list[str] = []
    lines.append("# Candidate Pool Analysis")
    lines.append("")
    lines.append(f"## Invalid Seeds (N={len(invalid_seeds)})")
    lines.append(f"- Seeds: {', '.join(str(seed) for seed in invalid_seeds)}")
    lines.append(f"- Total candidates evaluated for Diana: {int(invalid_stats['total_candidates'])}")
    lines.append(
        f"- Candidates with dev_beats=0: {int(invalid_stats['dev0_count'])} "
        f"({float(invalid_stats['dev0_pct']):.1f}%)"
    )
    lines.append(
        f"- Candidates with dev_beats>=1: {int(invalid_stats['dev1p_count'])} "
        f"({float(invalid_stats['dev1p_pct']):.1f}%)"
    )
    lines.append(f"- Mean Q of dev_beats=0 candidates: {float(invalid_stats['dev0_mean_q']):.3f}")
    lines.append(f"- Mean Q of dev_beats>=1 candidates: {float(invalid_stats['dev1p_mean_q']):.3f}")
    lines.append(f"- Q gap (dev=0 mean - dev>=1 mean): {float(invalid_stats['q_gap_dev0_minus_dev1p']):.3f}")
    lines.append("")

    lines.append("## The Exploit Basin")
    lines.append(
        f"- Candidates at TP position < 0.25 with dev_beats=0: {int(invalid_stats['exploit_basin_count'])}"
    )
    lines.append(f"- Mean Q of exploit basin candidates: {float(invalid_stats['exploit_basin_mean_q']):.3f}")
    lines.append("- These are the candidates the strict grammar blocks.")
    lines.append("")

    lines.append("## Search Winner Analysis")
    lines.append(
        f"- In all {len(invalid_seeds)} invalid seeds, the search winner has dev_beats=0: "
        f"{'yes' if bool(invalid_stats['winner_all_dev0']) else 'no'}"
    )
    lines.append(
        f"- Mean TP position of search winners in invalid seeds: "
        f"{float(invalid_stats['winner_mean_tp']):.3f}"
    )
    lines.append(
        f"- Mean Q of search winners in invalid seeds: "
        f"{float(invalid_stats['winner_mean_q']):.3f}"
    )
    lines.append("")

    lines.append(f"## Valid Seeds (N={len(valid_seed_sample)} sample)")
    lines.append(f"- Sample seeds: {', '.join(str(seed) for seed in valid_seed_sample)}")
    lines.append(f"- Total candidates evaluated for Diana: {int(valid_stats['total_candidates'])}")
    lines.append(
        f"- Candidates with dev_beats=0: {int(valid_stats['dev0_count'])} "
        f"({float(valid_stats['dev0_pct']):.1f}%)"
    )
    lines.append(
        f"- Candidates with dev_beats>=1: {int(valid_stats['dev1p_count'])} "
        f"({float(valid_stats['dev1p_pct']):.1f}%)"
    )
    lines.append(f"- Mean Q of dev_beats=0 candidates: {float(valid_stats['dev0_mean_q']):.3f}")
    lines.append(f"- Mean Q of dev_beats>=1 candidates: {float(valid_stats['dev1p_mean_q']):.3f}")
    lines.append(f"- Q gap (dev=0 mean - dev>=1 mean): {float(valid_stats['q_gap_dev0_minus_dev1p']):.3f}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Log arc-search candidate pools for Diana (depth-2 full evolution).")
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
    print("=== SEARCH CANDIDATE LOGGER ===")
    print()
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print("Agent candidate logging: Diana (all candidates), all agents (winners only)")
    print()

    prep_start = time.time()
    seed_contexts = _prepare_seed_contexts(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    print(f"Prepared depth-2 contexts in {time.time() - prep_start:.1f}s.")
    print()

    diana_candidate_rows: list[CandidateRecord] = []
    diana_winners: list[WinnerRecord] = []
    all_agent_winners: list[WinnerRecord] = []
    parity_mismatches: list[dict[str, Any]] = []
    feasible_mismatches: list[dict[str, Any]] = []

    for idx, context in enumerate(seed_contexts, start=1):
        print(f"[seed {idx:03d}/{len(seed_contexts):03d}] logging candidates", flush=True)

        diana_winner, diana_rows, _ = _instrument_search_arc(
            seed=int(context.seed),
            all_events=context.events,
            protagonist="diana",
            total_sim_time=context.total_sim_time,
            simulation_end_tick=context.simulation_end_tick,
            grammar_config=None,
        )
        diana_candidate_rows.extend(diana_rows)
        diana_winners.append(diana_winner)

        direct = search_arc(
            all_events=context.events,
            protagonist="diana",
            max_events=20,
            total_sim_time=context.total_sim_time,
            grammar_config=None,
        )
        direct_ids = [str(event.id) for event in direct.events]
        direct_beats = [str(beat.value) for beat in direct.beats]
        if diana_winner.event_ids != direct_ids or diana_winner.beats != direct_beats:
            parity_mismatches.append(
                {
                    "seed": int(context.seed),
                    "instrumented_event_ids": list(diana_winner.event_ids),
                    "direct_event_ids": direct_ids,
                    "instrumented_beats": list(diana_winner.beats),
                    "direct_beats": direct_beats,
                }
            )

        feasible_row = _search_with_config(
            events=context.events,
            protagonist="diana",
            total_sim_time=context.total_sim_time,
            grammar_config=None,
        )
        feasible_ids = [str(event_id) for event_id in feasible_row["event_ids"]]
        feasible_beats = [str(beat) for beat in feasible_row["beats"]]
        feasible_tp_positions = _tp_global_positions(
            event_ids=feasible_ids,
            beats=feasible_beats,
            by_id={str(event.id): event for event in context.events},
            simulation_end_tick=context.simulation_end_tick,
        )
        winner_tp = diana_winner.tp_position
        feasible_tp = float(feasible_tp_positions[0]) if feasible_tp_positions else None
        feasible_score = feasible_row.get("score")
        score_match = (
            feasible_score is None
            or abs(float(feasible_score) - float(diana_winner.q_score)) <= EPSILON
        )
        tp_match = (
            winner_tp is None and feasible_tp is None
        ) or (
            winner_tp is not None and feasible_tp is not None and abs(float(winner_tp) - float(feasible_tp)) <= EPSILON
        )
        if diana_winner.event_ids != feasible_ids or diana_winner.beats != feasible_beats or not tp_match or not score_match:
            feasible_mismatches.append(
                {
                    "seed": int(context.seed),
                    "instrumented_event_ids": list(diana_winner.event_ids),
                    "feasible_event_ids": feasible_ids,
                    "instrumented_beats": list(diana_winner.beats),
                    "feasible_beats": feasible_beats,
                    "instrumented_tp_position": winner_tp,
                    "feasible_tp_position": feasible_tp,
                    "instrumented_q_score": float(diana_winner.q_score),
                    "feasible_q_score": feasible_score,
                }
            )

        for agent in DINNER_PARTY_AGENTS:
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
            relaxed_validation = validate_arc(
                events=result.events,
                beats=result.beats,
                total_sim_time=context.total_sim_time,
                grammar_config=RELAXED_DEV0_CONFIG,
            )
            all_agent_winners.append(
                WinnerRecord(
                    seed=int(context.seed),
                    agent=str(agent),
                    q_score=q_score,
                    tp_position=tp_position,
                    tp_event_type=tp_event_type,
                    dev_beat_count=_dev_beat_count(result.beats),
                    total_beats=len(result.beats),
                    strict_valid=bool(result.validation.valid),
                    relaxed_valid=bool(relaxed_validation.valid),
                    event_ids=[str(event.id) for event in result.events],
                    beats=[str(beat.value) for beat in result.beats],
                    strict_violations=[str(v) for v in result.validation.violations],
                )
            )

    invalid_seed_set = sorted(int(row.seed) for row in diana_winners if not bool(row.strict_valid))
    valid_seed_pool = [int(seed) for seed in seeds if int(seed) not in set(invalid_seed_set)]

    by_seed_dev_counts: dict[int, tuple[int, int]] = {}
    for seed in valid_seed_pool:
        rows = [row for row in diana_candidate_rows if int(row.seed) == seed]
        dev0_count = sum(1 for row in rows if int(row.dev_beat_count) == 0)
        dev1p_count = sum(1 for row in rows if int(row.dev_beat_count) >= 1)
        by_seed_dev_counts[seed] = (dev0_count, dev1p_count)

    mixed_valid_seeds = [
        seed for seed in valid_seed_pool
        if by_seed_dev_counts.get(seed, (0, 0))[0] > 0 and by_seed_dev_counts.get(seed, (0, 0))[1] > 0
    ]
    remaining_valid_seeds = [seed for seed in valid_seed_pool if seed not in set(mixed_valid_seeds)]
    valid_seed_sample = (mixed_valid_seeds + remaining_valid_seeds)[:9]

    invalid_stats = _candidate_stats(candidate_rows=diana_candidate_rows, seed_set=set(invalid_seed_set))
    valid_stats = _candidate_stats(candidate_rows=diana_candidate_rows, seed_set=set(valid_seed_sample))

    expected_set = sorted(int(seed) for seed in EXPECTED_INVALID_SEEDS)
    expected_match = expected_set == invalid_seed_set
    strict_valid_valid_sample = all(
        bool(row.strict_valid) and int(row.dev_beat_count) >= 1
        for row in diana_winners
        if int(row.seed) in set(valid_seed_sample)
    )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": "search_candidate_pool_visualization",
            "condition": "depth2_full_evolution_strict_search",
            "seeds": [int(seed) for seed in seeds],
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
        },
        "diana": {
            "candidate_records": [asdict(row) for row in diana_candidate_rows],
            "winners": [asdict(row) for row in diana_winners],
            "invalid_seed_set_from_strict": invalid_seed_set,
            "mixed_valid_seeds": [int(seed) for seed in mixed_valid_seeds],
            "valid_seed_sample": [int(seed) for seed in valid_seed_sample],
            "stats_invalid_seeds": invalid_stats,
            "stats_valid_seed_sample": valid_stats,
        },
        "all_agents": {
            "winner_records": [asdict(row) for row in all_agent_winners],
            "agents": list(DINNER_PARTY_AGENTS),
        },
        "verification": {
            "expected_invalid_seed_set": expected_set,
            "matches_expected_invalid_seed_set": bool(expected_match),
            "instrumented_vs_search_arc": {
                "mismatch_count": int(len(parity_mismatches)),
                "mismatches": parity_mismatches,
            },
            "instrumented_vs_feasible_volume_search": {
                "mismatch_count": int(len(feasible_mismatches)),
                "mismatches": feasible_mismatches,
            },
            "valid_seed_sample_winners_strict_and_dev1p": bool(strict_valid_valid_sample),
        },
    }

    summary_text = _build_summary_markdown(
        invalid_stats=invalid_stats,
        valid_stats=valid_stats,
        invalid_seeds=invalid_seed_set,
        valid_seed_sample=valid_seed_sample,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(f"Observed Diana invalid seeds: {invalid_seed_set}")
    print(
        "Verification: "
        f"search_arc mismatches={len(parity_mismatches)}, "
        f"feasible mismatches={len(feasible_mismatches)}"
    )


if __name__ == "__main__":
    main()
