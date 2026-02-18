from __future__ import annotations

from narrativefield.extraction.arc_search import _enforce_monotonic_beats
from narrativefield.extraction.beat_classifier import classify_beats, repair_beats
from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType


def _evt(idx: int, *, event_type: EventType = EventType.CHAT, tension: float = 0.2) -> Event:
    return Event(
        id=f"bc_{idx}",
        sim_time=float(idx),
        tick_id=idx,
        order_in_tick=0,
        type=event_type,
        source_agent="alice",
        target_agents=["bob"],
        location_id="dining_table",
        causal_links=[f"bc_{idx - 1}"] if idx > 0 else [],
        deltas=[],
        description=f"beat classifier event {idx}",
        metrics=EventMetrics(tension=tension, irony=0.1, significance=0.1),
    )


def test_repair_beats_forces_first_to_setup() -> None:
    events = [_evt(i) for i in range(6)]
    beats = [
        BeatType.COMPLICATION,
        BeatType.ESCALATION,
        BeatType.COMPLICATION,
        BeatType.SETUP,
        BeatType.CONSEQUENCE,
        BeatType.CONSEQUENCE,
    ]

    repaired = repair_beats(events, list(beats))
    assert repaired[0] == BeatType.SETUP


def test_repair_beats_first_already_setup() -> None:
    events = [_evt(i) for i in range(5)]
    beats = [
        BeatType.SETUP,
        BeatType.COMPLICATION,
        BeatType.ESCALATION,
        BeatType.TURNING_POINT,
        BeatType.CONSEQUENCE,
    ]

    repaired = repair_beats(events, list(beats))
    assert repaired == beats


def test_repair_beats_no_setup_anywhere() -> None:
    events = [_evt(i) for i in range(4)]
    beats = [
        BeatType.COMPLICATION,
        BeatType.ESCALATION,
        BeatType.TURNING_POINT,
        BeatType.CONSEQUENCE,
    ]

    repaired = repair_beats(events, list(beats))
    assert repaired[0] == BeatType.SETUP


def test_late_setup_survives_monotonic_when_first_is_setup() -> None:
    events = [
        _evt(0, event_type=EventType.REVEAL, tension=0.95),
        _evt(1, event_type=EventType.CONFLICT, tension=0.40),
        _evt(2, event_type=EventType.CHAT, tension=0.20),
        _evt(3, event_type=EventType.CONFLICT, tension=0.70),
        _evt(4, event_type=EventType.REVEAL, tension=0.85),
        _evt(5, event_type=EventType.CHAT, tension=0.30),
        _evt(6, event_type=EventType.CHAT, tension=0.25),
        _evt(7, event_type=EventType.CHAT, tension=0.20),
    ]

    beats = classify_beats(events)
    repaired = _enforce_monotonic_beats(events, beats)
    assert repaired[0] == BeatType.SETUP
