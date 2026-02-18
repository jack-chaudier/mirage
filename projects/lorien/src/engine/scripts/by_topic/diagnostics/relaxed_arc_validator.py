from __future__ import annotations

from narrativefield.extraction.types import ArcValidation
from narrativefield.schema.events import BeatType, Event


def validate_arc_relaxed(
    *,
    events: list[Event],
    beats: list[BeatType],
    total_sim_time: float | None = None,
) -> ArcValidation:
    """
    Relaxed arc validator:
    - TURNING_POINT count: allow 1 or 2 (strict requires exactly 1)
    - Phase regressions: allow at most one adjacent phase regression
    All other rules are unchanged from extraction.arc_validator.validate_arc.
    """
    violations: list[str] = []
    beat_sequence = beats

    # Rule 1: >=1 SETUP
    if beat_sequence.count(BeatType.SETUP) < 1:
        violations.append("Missing SETUP beat")

    # Rule 2: >=1 COMPLICATION or ESCALATION
    development = [b for b in beat_sequence if b in {BeatType.COMPLICATION, BeatType.ESCALATION}]
    if len(development) < 1:
        violations.append("Missing COMPLICATION or ESCALATION beat")

    # Rule 3 (relaxed): 1 or 2 TURNING_POINT beats allowed
    tp_count = beat_sequence.count(BeatType.TURNING_POINT)
    if tp_count < 1 or tp_count > 2:
        violations.append(f"Expected 1 or 2 TURNING_POINT, found {tp_count}")

    # Rule 4: >=1 CONSEQUENCE
    if beat_sequence.count(BeatType.CONSEQUENCE) < 1:
        violations.append("Missing CONSEQUENCE beat")

    # Rule 5 (relaxed): allow at most one phase regression.
    phase_order = {
        BeatType.SETUP: 0,
        BeatType.COMPLICATION: 1,
        BeatType.ESCALATION: 1,
        BeatType.TURNING_POINT: 2,
        BeatType.CONSEQUENCE: 3,
    }
    regression_count = 0
    for i in range(1, len(beat_sequence)):
        if phase_order.get(beat_sequence[i], 0) < phase_order.get(beat_sequence[i - 1], 0):
            regression_count += 1
    if regression_count > 1:
        violations.append(f"Order violation: phase regressions={regression_count} (max 1)")

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


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _count_agent_appearances(events: list[Event]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        for agent in _participants(event):
            counts[agent] = counts.get(agent, 0) + 1
    return counts
