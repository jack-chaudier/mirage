from __future__ import annotations

from narrativefield.schema.events import BeatType, Event, EventType


def classify_beats(events: list[Event]) -> list[BeatType]:
    """
    Classify each event in an arc into a BeatType.

    Source: specs/metrics/story-extraction.md Section 3.

    MVP note: the full spec path allows scene-context-aware beat labeling.
    This implementation intentionally classifies using event type + tension/position
    only; scene context is omitted for now.
    """
    n = len(events)
    if n == 0:
        return []

    tensions = [float(e.metrics.tension) for e in events]
    peak_idx = tensions.index(max(tensions)) if tensions else 0

    beats: list[BeatType] = []
    for i, event in enumerate(events):
        position_ratio = i / max(n - 1, 1)
        beats.append(
            _classify_single_beat(
                event=event,
                position_ratio=position_ratio,
                peak_tension_idx=peak_idx,
                event_index=i,
                total_events=n,
                tensions=tensions,
            )
        )

    return repair_beats(events, beats)


def _classify_single_beat(
    *,
    event: Event,
    position_ratio: float,
    peak_tension_idx: int,
    event_index: int,
    total_events: int,
    tensions: list[float],
) -> BeatType:
    # Rule 1: CATASTROPHE or CONFLICT at peak -> turning point.
    if event_index == peak_tension_idx and event.type in {EventType.CATASTROPHE, EventType.CONFLICT}:
        return BeatType.TURNING_POINT

    # Rule 2: REVEAL with irony collapse -> turning point.
    if event.type == EventType.REVEAL and event.metrics.irony_collapse and event.metrics.irony_collapse.detected:
        return BeatType.TURNING_POINT

    # Rule 3: position-based defaults.
    if position_ratio < 0.25:
        if event.type in {EventType.CHAT, EventType.SOCIAL_MOVE, EventType.OBSERVE}:
            return BeatType.SETUP
        if event.type in {EventType.CONFIDE, EventType.LIE, EventType.REVEAL}:
            return BeatType.COMPLICATION
        return BeatType.SETUP

    # Rule 4: development window [0.25, 0.70).
    if position_ratio < 0.70:
        if event_index > 0 and tensions[event_index] > tensions[event_index - 1]:
            return BeatType.ESCALATION
        return BeatType.COMPLICATION

    # Rule 5: late window [0.70, 1.0] trends to consequence/aftermath.
    return BeatType.CONSEQUENCE


def repair_beats(events: list[Event], beats: list[BeatType]) -> list[BeatType]:
    """
    Light repair pass to enforce the grammar prerequisites for downstream validation/scoring.

    Source: specs/metrics/story-extraction.md Section 3.2.
    """
    if not beats:
        return beats

    # TURNING_POINT creation/deduplication is handled in one place:
    # extraction.arc_search._enforce_monotonic_beats.

    if beats[0] != BeatType.SETUP:
        beats[0] = BeatType.SETUP

    if BeatType.CONSEQUENCE not in beats:
        # If the arc ends on the turning point, we cannot create a CONSEQUENCE beat
        # without moving the turning point earlier. Leave it missing so validation
        # can explain what to fix (select a longer aftermath).
        if beats[-1] != BeatType.TURNING_POINT:
            beats[-1] = BeatType.CONSEQUENCE

    # Ensure at least one development beat.
    if not any(b in {BeatType.COMPLICATION, BeatType.ESCALATION} for b in beats):
        if len(beats) >= 2:
            beats[1] = BeatType.COMPLICATION

    return beats
