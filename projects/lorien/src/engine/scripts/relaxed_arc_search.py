from __future__ import annotations

from contextlib import contextmanager

import narrativefield.extraction.arc_search as arc_search_mod
from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.rashomon import RashomonArc, RashomonSet
from narrativefield.schema.events import BeatType, Event

from scripts.relaxed_arc_validator import validate_arc_relaxed


def _enforce_beats_relaxed(events: list[Event], beats: list[BeatType]) -> list[BeatType]:
    """
    Relaxed beat repair used during search-time candidate shaping:
    - allow at most one adjacent phase regression (repair from second onward)
    - allow up to two turning points (force into [1, 2] range)
    """
    if len(beats) < 2:
        return beats

    result = list(beats)

    # Phase progression: allow one local regression.
    regression_count = 0
    prev_phase = arc_search_mod.PHASE_ORDER[result[0]]
    for i in range(1, len(result)):
        cur_phase = arc_search_mod.PHASE_ORDER[result[i]]
        if cur_phase < prev_phase:
            regression_count += 1
            if regression_count > 1:
                result[i] = arc_search_mod._select_beat_for_phase(
                    phase_level=prev_phase,
                    index=i,
                    current_beats=result,
                    events=events,
                )
                cur_phase = arc_search_mod.PHASE_ORDER[result[i]]
        prev_phase = cur_phase

    # Turning points: allow one or two.
    tp_indices = [i for i, beat in enumerate(result) if beat == BeatType.TURNING_POINT]
    if len(tp_indices) == 0:
        n = len(events)
        mid_start = n // 4
        mid_end = 3 * n // 4
        if mid_start < mid_end:
            best_idx = max(range(mid_start, mid_end), key=lambda j: events[j].metrics.tension)
            result[best_idx] = BeatType.TURNING_POINT
    elif len(tp_indices) > 2:
        keep = sorted(tp_indices, key=lambda j: events[j].metrics.tension, reverse=True)[:2]
        keep_set = set(keep)
        keep_sorted = sorted(keep)
        first_keep = keep_sorted[0]
        last_keep = keep_sorted[-1]

        for idx in tp_indices:
            if idx in keep_set:
                continue
            if idx < first_keep:
                result[idx] = BeatType.ESCALATION
            elif idx > last_keep:
                result[idx] = BeatType.CONSEQUENCE
            else:
                # Between surviving turning points, keep development pressure.
                result[idx] = BeatType.ESCALATION

    return result


@contextmanager
def _relaxed_search_context():
    original_validator = arc_search_mod.validate_arc
    original_enforcer = arc_search_mod._enforce_monotonic_beats
    arc_search_mod.validate_arc = validate_arc_relaxed
    arc_search_mod._enforce_monotonic_beats = _enforce_beats_relaxed
    try:
        yield
    finally:
        arc_search_mod.validate_arc = original_validator
        arc_search_mod._enforce_monotonic_beats = original_enforcer


def search_arc_relaxed(*args, **kwargs):
    with _relaxed_search_context():
        return arc_search_mod.search_arc(*args, **kwargs)


def extract_rashomon_set_relaxed(
    events: list[Event],
    seed: int,
    agents: list[str] | None = None,
    max_events_per_arc: int = 20,
    total_sim_time: float | None = None,
) -> RashomonSet:
    if agents is None:
        discovered_agents: set[str] = set()
        for event in events:
            discovered_agents.add(event.source_agent)
            discovered_agents.update(event.target_agents)
        agents = sorted(discovered_agents)

    arcs: list[RashomonArc] = []
    with _relaxed_search_context():
        for agent in agents:
            search_result = arc_search_mod.search_arc(
                all_events=events,
                protagonist=agent,
                max_events=max_events_per_arc,
                total_sim_time=total_sim_time,
            )
            arc_score = None
            if search_result.validation.valid:
                arc_score = search_result.arc_score or score_arc(search_result.events, search_result.beats)

            arcs.append(
                RashomonArc(
                    protagonist=agent,
                    search_result=search_result,
                    arc_score=arc_score,
                    events=search_result.events,
                    beats=search_result.beats,
                    valid=search_result.validation.valid,
                    violation_count=len(search_result.validation.violations),
                    violations=list(search_result.validation.violations),
                )
            )

    arcs.sort(
        key=lambda arc: (
            0 if arc.valid else 1,
            -(arc.arc_score.composite if arc.arc_score else 0.0),
            arc.violation_count,
            arc.protagonist,
        )
    )
    return RashomonSet(seed=seed, total_events=len(events), arcs=arcs)
