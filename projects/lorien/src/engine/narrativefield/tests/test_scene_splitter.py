from __future__ import annotations

from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType
from narrativefield.storyteller.scene_splitter import split_into_scenes


def _ev(
    *,
    eid: str,
    sim_time: float,
    tick_id: int,
    order: int,
    etype: EventType,
    source: str,
    targets: list[str],
    location: str,
    beat_type: BeatType | None = None,
    tension: float = 0.0,
) -> Event:
    return Event(
        id=eid,
        sim_time=sim_time,
        tick_id=tick_id,
        order_in_tick=order,
        type=etype,
        source_agent=source,
        target_agents=list(targets),
        location_id=location,
        causal_links=[],
        deltas=[],
        description="",
        beat_type=beat_type,
        metrics=EventMetrics(tension=float(tension)),
    )


def test_split_into_scenes_mini_arc_respects_location_and_turning_point_context() -> None:
    events = [
        _ev(
            eid="e1",
            sim_time=0.0,
            tick_id=0,
            order=0,
            etype=EventType.CHAT,
            source="victor",
            targets=["marcus"],
            location="dining_table",
            beat_type=BeatType.SETUP,
        ),
        _ev(
            eid="e2",
            sim_time=1.0,
            tick_id=0,
            order=1,
            etype=EventType.CONFIDE,
            source="elena",
            targets=["marcus"],
            location="dining_table",
            beat_type=BeatType.COMPLICATION,
        ),
        _ev(
            eid="e3",
            sim_time=2.0,
            tick_id=1,
            order=0,
            etype=EventType.CONFLICT,
            source="victor",
            targets=["marcus"],
            location="dining_table",
            beat_type=BeatType.ESCALATION,
            tension=0.62,
        ),
        # Balcony beat with a time gap afterwards (should not create a tiny chunk split).
        _ev(
            eid="e4",
            sim_time=5.0,
            tick_id=2,
            order=0,
            etype=EventType.REVEAL,
            source="victor",
            targets=["diana"],
            location="balcony",
            beat_type=BeatType.ESCALATION,
            tension=0.71,
        ),
        _ev(
            eid="e5",
            sim_time=20.0,  # gap > 10 minutes
            tick_id=3,
            order=0,
            etype=EventType.INTERNAL,
            source="victor",
            targets=[],
            location="balcony",
            beat_type=BeatType.ESCALATION,
        ),
        # Turning point back at the dining table.
        _ev(
            eid="e6",
            sim_time=22.0,
            tick_id=4,
            order=0,
            etype=EventType.CONFLICT,
            source="victor",
            targets=["marcus"],
            location="dining_table",
            beat_type=BeatType.TURNING_POINT,
            tension=0.69,
        ),
        # Consequences.
        _ev(
            eid="e7",
            sim_time=25.0,
            tick_id=5,
            order=0,
            etype=EventType.CHAT,
            source="diana",
            targets=["lydia"],
            location="dining_table",
            beat_type=BeatType.CONSEQUENCE,
        ),
        _ev(
            eid="e8",
            sim_time=26.0,
            tick_id=5,
            order=1,
            etype=EventType.INTERNAL,
            source="victor",
            targets=[],
            location="dining_table",
            beat_type=BeatType.CONSEQUENCE,
        ),
    ]

    chunks = split_into_scenes(events)
    assert 2 <= len(chunks) <= 3

    # Location change (dining_table -> balcony) creates a split between e3 and e4.
    event_to_chunk: dict[str, int] = {}
    for idx, ch in enumerate(chunks):
        for e in ch.events:
            event_to_chunk[e.id] = idx
    assert event_to_chunk["e3"] != event_to_chunk["e4"]

    # Turning point is not isolated, and marks the chunk as pivotal.
    tp_chunks = [ch for ch in chunks if any(e.beat_type == BeatType.TURNING_POINT for e in ch.events)]
    assert len(tp_chunks) == 1
    tp_chunk = tp_chunks[0]
    assert len(tp_chunk.events) >= 2
    assert tp_chunk.is_pivotal is True


def test_split_into_scenes_empty_and_singleton() -> None:
    assert split_into_scenes([]) == []

    one = _ev(
        eid="only",
        sim_time=0.0,
        tick_id=0,
        order=0,
        etype=EventType.CHAT,
        source="victor",
        targets=["elena"],
        location="dining_table",
        beat_type=BeatType.SETUP,
    )
    chunks = split_into_scenes([one])
    assert len(chunks) == 1
    assert chunks[0].events[0].id == "only"


def test_same_location_still_splits_on_time_gap() -> None:
    events = [
        _ev(
            eid="a1",
            sim_time=0.0,
            tick_id=0,
            order=0,
            etype=EventType.CHAT,
            source="thorne",
            targets=["victor"],
            location="dining_table",
        ),
        _ev(
            eid="a2",
            sim_time=1.0,
            tick_id=0,
            order=1,
            etype=EventType.CHAT,
            source="victor",
            targets=["thorne"],
            location="dining_table",
        ),
        _ev(
            eid="a3",
            sim_time=2.0,
            tick_id=1,
            order=0,
            etype=EventType.OBSERVE,
            source="lydia",
            targets=[],
            location="dining_table",
        ),
        # Gap > 10 minutes should split here even without location change.
        _ev(
            eid="a4",
            sim_time=20.0,
            tick_id=10,
            order=0,
            etype=EventType.INTERNAL,
            source="victor",
            targets=[],
            location="dining_table",
        ),
        _ev(
            eid="a5",
            sim_time=21.0,
            tick_id=10,
            order=1,
            etype=EventType.CHAT,
            source="diana",
            targets=["marcus"],
            location="dining_table",
        ),
        _ev(
            eid="a6",
            sim_time=22.0,
            tick_id=11,
            order=0,
            etype=EventType.PHYSICAL,
            source="elena",
            targets=[],
            location="dining_table",
        ),
    ]

    chunks = split_into_scenes(events)
    assert len(chunks) >= 2
