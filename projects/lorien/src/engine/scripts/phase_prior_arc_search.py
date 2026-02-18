from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import narrativefield.extraction.arc_search as arc_search_mod
from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.rashomon import RashomonArc, RashomonSet
from narrativefield.extraction.types import ArcValidation
from narrativefield.schema.events import BeatType, Event


def _build_phase_prior_validator(
    *,
    original_validate: Callable[..., ArcValidation],
    simulation_end_tick: int,
    tp_lower: float,
    tp_upper: float,
) -> Callable[..., ArcValidation]:
    denom = max(int(simulation_end_tick), 1)

    def _validate_arc_with_phase_prior(
        *,
        events: list[Event],
        beats: list[BeatType],
        total_sim_time: float | None = None,
    ) -> ArcValidation:
        strict_validation = original_validate(
            events=events,
            beats=beats,
            total_sim_time=total_sim_time,
        )
        if not strict_validation.valid:
            return strict_validation

        tp_index: int | None = None
        for index, beat in enumerate(beats):
            if beat == BeatType.TURNING_POINT:
                tp_index = index
                break
        if tp_index is None or tp_index >= len(events):
            return strict_validation

        tp_tick = int(events[tp_index].tick_id)
        tp_global_pos = float(tp_tick / denom)
        if tp_global_pos < tp_lower or tp_global_pos > tp_upper:
            violations = list(strict_validation.violations)
            violations.append(
                f"Turning point outside phase-prior window: {tp_global_pos:.3f} not in [{tp_lower:.2f}, {tp_upper:.2f}]"
            )
            return ArcValidation(valid=False, violations=tuple(violations))

        return strict_validation

    return _validate_arc_with_phase_prior


@contextmanager
def _phase_prior_search_context(
    *,
    simulation_end_tick: int,
    tp_lower: float,
    tp_upper: float,
):
    original_validator = arc_search_mod.validate_arc
    patched_validator = _build_phase_prior_validator(
        original_validate=original_validator,
        simulation_end_tick=simulation_end_tick,
        tp_lower=tp_lower,
        tp_upper=tp_upper,
    )
    arc_search_mod.validate_arc = patched_validator
    try:
        yield
    finally:
        arc_search_mod.validate_arc = original_validator


def search_arc_phase_prior(
    *,
    all_events: list[Event],
    time_start: float | None = None,
    time_end: float | None = None,
    agent_ids: list[str] | None = None,
    protagonist: str | None = None,
    max_events: int = 20,
    total_sim_time: float | None = None,
    tp_lower: float = 0.25,
    tp_upper: float = 0.70,
):
    simulation_end_tick = max((int(event.tick_id) for event in all_events), default=1)
    with _phase_prior_search_context(
        simulation_end_tick=simulation_end_tick,
        tp_lower=tp_lower,
        tp_upper=tp_upper,
    ):
        return arc_search_mod.search_arc(
            all_events=all_events,
            time_start=time_start,
            time_end=time_end,
            agent_ids=agent_ids,
            protagonist=protagonist,
            max_events=max_events,
            total_sim_time=total_sim_time,
        )


def extract_rashomon_set_phase_prior(
    events: list[Event],
    seed: int,
    agents: list[str] | None = None,
    max_events_per_arc: int = 20,
    total_sim_time: float | None = None,
    tp_lower: float = 0.25,
    tp_upper: float = 0.70,
) -> RashomonSet:
    if agents is None:
        discovered_agents: set[str] = set()
        for event in events:
            discovered_agents.add(event.source_agent)
            discovered_agents.update(event.target_agents)
        agents = sorted(discovered_agents)

    simulation_end_tick = max((int(event.tick_id) for event in events), default=1)

    arcs: list[RashomonArc] = []
    with _phase_prior_search_context(
        simulation_end_tick=simulation_end_tick,
        tp_lower=tp_lower,
        tp_upper=tp_upper,
    ):
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
