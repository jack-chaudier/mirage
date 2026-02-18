from __future__ import annotations

from dataclasses import dataclass

from narrativefield.extraction.types import ArcValidation
from narrativefield.schema.events import BeatType, Event


@dataclass(frozen=True)
class GrammarConfig:
    """Parameterized arc grammar used by sweeps and constrained search."""

    min_development_beats: int = 1
    max_phase_regressions: int = 0
    max_turning_points: int = 1
    min_beat_count: int = 4
    max_beat_count: int = 20
    min_timespan_fraction: float = 0.15
    protagonist_coverage: float = 0.60

    def to_dict(self) -> dict[str, int | float]:
        return {
            "min_development_beats": int(self.min_development_beats),
            "max_phase_regressions": int(self.max_phase_regressions),
            "max_turning_points": int(self.max_turning_points),
            "min_beat_count": int(self.min_beat_count),
            "max_beat_count": int(self.max_beat_count),
            "min_timespan_fraction": float(self.min_timespan_fraction),
            "protagonist_coverage": float(self.protagonist_coverage),
        }


_PHASE_ORDER: dict[BeatType, int] = {
    BeatType.SETUP: 0,
    BeatType.COMPLICATION: 1,
    BeatType.ESCALATION: 1,
    BeatType.TURNING_POINT: 2,
    BeatType.CONSEQUENCE: 3,
}


def validate_arc(
    *,
    events: list[Event],
    beats: list[BeatType],
    total_sim_time: float | None = None,
    grammar_config: GrammarConfig | None = None,
) -> ArcValidation:
    """
    Validate an arc against the BeatType grammar and structural constraints.

    Source: specs/metrics/story-extraction.md Section 2.2.
    """
    if grammar_config is None:
        return _validate_arc_strict(events=events, beats=beats, total_sim_time=total_sim_time)
    return _validate_arc_configured(
        events=events,
        beats=beats,
        total_sim_time=total_sim_time,
        grammar_config=grammar_config,
    )


def _validate_arc_strict(
    *,
    events: list[Event],
    beats: list[BeatType],
    total_sim_time: float | None = None,
) -> ArcValidation:
    """Legacy strict validator path. Keep behavior unchanged for compatibility."""
    violations: list[str] = []

    beat_sequence = beats

    # Rule 1: >=1 SETUP
    if beat_sequence.count(BeatType.SETUP) < 1:
        violations.append("Missing SETUP beat")

    # Rule 2: >=1 COMPLICATION or ESCALATION
    development = [b for b in beat_sequence if b in {BeatType.COMPLICATION, BeatType.ESCALATION}]
    if len(development) < 1:
        violations.append("Missing COMPLICATION or ESCALATION beat")

    # Rule 3: exactly 1 TURNING_POINT
    tp_count = beat_sequence.count(BeatType.TURNING_POINT)
    if tp_count != 1:
        violations.append(f"Expected 1 TURNING_POINT, found {tp_count}")

    # Rule 4: >=1 CONSEQUENCE
    if beat_sequence.count(BeatType.CONSEQUENCE) < 1:
        violations.append("Missing CONSEQUENCE beat")

    # Rule 5: phase order
    for i in range(1, len(beat_sequence)):
        if _PHASE_ORDER.get(beat_sequence[i], 0) < _PHASE_ORDER.get(beat_sequence[i - 1], 0):
            violations.append(
                f"Order violation: {beat_sequence[i].value} after {beat_sequence[i - 1].value}"
            )
            break

    # Rule 6: beat count (min 4, max 20)
    if len(beat_sequence) < 4:
        violations.append(f"Too few beats: {len(beat_sequence)} < 4")
    if len(beat_sequence) > 20:
        violations.append(f"Too many beats: {len(beat_sequence)} > 20")

    # Rule 7: protagonist consistency (>=60% of events include one agent)
    agent_counts = _count_agent_appearances(events)
    total = len(events)
    if total > 0 and agent_counts:
        max_agent, max_count = max(agent_counts.items(), key=lambda x: x[1])
        if (max_count / total) < 0.6:
            violations.append(
                f"No protagonist: most frequent agent '{max_agent}' appears in "
                f"{max_count}/{total} events ({max_count/total:.0%})"
            )

    # Rule 8: causal connectivity (causal link to prior OR participant overlap with previous)
    arc_event_ids = {e.id for e in events}
    for i, event in enumerate(events[1:], 1):
        has_causal_link = any(link in arc_event_ids for link in event.causal_links)
        shares_participant = bool(_participants(event) & _participants(events[i - 1]))
        if not has_causal_link and not shares_participant:
            violations.append(f"Causal gap at event {event.id}: no causal link or participant overlap")
            break

    # Rule 9: time span >= 15% of total sim time, with an absolute floor of 10 sim minutes
    if events:
        span = float(events[-1].sim_time) - float(events[0].sim_time)
        if span < 10.0:
            violations.append(f"Arc too short: spans {span:.1f} sim minutes (minimum 10)")
        if total_sim_time is not None and total_sim_time > 0:
            if span < 0.15 * float(total_sim_time):
                violations.append(
                    f"Arc too short: spans {span:.1f} sim minutes (<15% of total {float(total_sim_time):.1f})"
                )

    return ArcValidation(valid=len(violations) == 0, violations=tuple(violations))


def _validate_arc_configured(
    *,
    events: list[Event],
    beats: list[BeatType],
    total_sim_time: float | None,
    grammar_config: GrammarConfig,
) -> ArcValidation:
    """Configurable validator path used for regularization sweeps."""
    violations: list[str] = []
    beat_sequence = beats

    # Rule 1: >=1 SETUP (kept fixed).
    if beat_sequence.count(BeatType.SETUP) < 1:
        violations.append("Missing SETUP beat")

    # Rule 2: minimum development beats.
    development = [b for b in beat_sequence if b in {BeatType.COMPLICATION, BeatType.ESCALATION}]
    if len(development) < int(grammar_config.min_development_beats):
        if int(grammar_config.min_development_beats) <= 1:
            violations.append("Missing COMPLICATION or ESCALATION beat")
        else:
            violations.append(
                f"Too few development beats: {len(development)} < {int(grammar_config.min_development_beats)}"
            )

    # Rule 3: turning point count.
    tp_count = beat_sequence.count(BeatType.TURNING_POINT)
    max_turning_points = int(grammar_config.max_turning_points)
    if max_turning_points <= 1:
        if tp_count != 1:
            violations.append(f"Expected 1 TURNING_POINT, found {tp_count}")
    else:
        if tp_count < 1:
            violations.append("Expected at least 1 TURNING_POINT, found 0")
        if tp_count > max_turning_points:
            violations.append(
                f"Too many TURNING_POINT beats: {tp_count} > {max_turning_points}"
            )

    # Rule 4: >=1 CONSEQUENCE (kept fixed).
    if beat_sequence.count(BeatType.CONSEQUENCE) < 1:
        violations.append("Missing CONSEQUENCE beat")

    # Rule 5: phase order with configurable regression budget.
    first_regression: tuple[BeatType, BeatType] | None = None
    regressions = 0
    for i in range(1, len(beat_sequence)):
        prev = beat_sequence[i - 1]
        cur = beat_sequence[i]
        if _PHASE_ORDER.get(cur, 0) < _PHASE_ORDER.get(prev, 0):
            regressions += 1
            if first_regression is None:
                first_regression = (cur, prev)

    max_phase_regressions = int(grammar_config.max_phase_regressions)
    if regressions > max_phase_regressions:
        if max_phase_regressions == 0 and first_regression is not None:
            violations.append(
                f"Order violation: {first_regression[0].value} after {first_regression[1].value}"
            )
        else:
            violations.append(
                f"Order violation: phase regressions={regressions} (max {max_phase_regressions})"
            )

    # Rule 6: beat count.
    min_beat_count = int(grammar_config.min_beat_count)
    max_beat_count = int(grammar_config.max_beat_count)
    if len(beat_sequence) < min_beat_count:
        violations.append(f"Too few beats: {len(beat_sequence)} < {min_beat_count}")
    if len(beat_sequence) > max_beat_count:
        violations.append(f"Too many beats: {len(beat_sequence)} > {max_beat_count}")

    # Rule 7: protagonist consistency.
    agent_counts = _count_agent_appearances(events)
    total = len(events)
    protagonist_coverage = float(grammar_config.protagonist_coverage)
    if total > 0 and agent_counts and protagonist_coverage > 0.0:
        max_agent, max_count = max(agent_counts.items(), key=lambda x: x[1])
        if (max_count / total) < protagonist_coverage:
            violations.append(
                f"No protagonist: most frequent agent '{max_agent}' appears in "
                f"{max_count}/{total} events ({max_count/total:.0%})"
            )

    # Rule 8: causal connectivity (kept fixed).
    arc_event_ids = {e.id for e in events}
    for i, event in enumerate(events[1:], 1):
        has_causal_link = any(link in arc_event_ids for link in event.causal_links)
        shares_participant = bool(_participants(event) & _participants(events[i - 1]))
        if not has_causal_link and not shares_participant:
            violations.append(f"Causal gap at event {event.id}: no causal link or participant overlap")
            break

    # Rule 9: time span.
    if events:
        span = float(events[-1].sim_time) - float(events[0].sim_time)
        if span < 10.0:
            violations.append(f"Arc too short: spans {span:.1f} sim minutes (minimum 10)")
        min_fraction = float(grammar_config.min_timespan_fraction)
        if total_sim_time is not None and total_sim_time > 0 and min_fraction > 0.0:
            minimum_span = min_fraction * float(total_sim_time)
            if span < minimum_span:
                violations.append(
                    f"Arc too short: spans {span:.1f} sim minutes "
                    f"(<{min_fraction * 100.0:.0f}% of total {float(total_sim_time):.1f})"
                )

    return ArcValidation(valid=len(violations) == 0, violations=tuple(violations))


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _count_agent_appearances(events: list[Event]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        for agent in _participants(e):
            counts[agent] = counts.get(agent, 0) + 1
    return counts
