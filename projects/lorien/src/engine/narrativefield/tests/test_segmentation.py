from __future__ import annotations

from narrativefield.metrics.segmentation import SegmentationConfig, segment_into_scenes
from narrativefield.schema.events import Event, EventMetrics, EventType
from narrativefield.simulation.scenarios import create_dinner_party_world


def _make_event(
    event_id: str,
    *,
    event_type: EventType,
    tick_id: int,
    sim_time: float,
    source: str = "thorne",
    targets: list[str] | None = None,
    location: str = "dining_table",
    destination: str | None = None,
) -> Event:
    metadata = {"destination": destination} if destination else None
    return Event(
        id=event_id,
        sim_time=sim_time,
        tick_id=tick_id,
        order_in_tick=0,
        type=event_type,
        source_agent=source,
        target_agents=targets or [],
        location_id=location,
        causal_links=[],
        deltas=[],
        description=event_id,
        content_metadata=metadata,
        metrics=EventMetrics(tension=0.25),
    )


def test_social_move_scene_boundary_uses_destination_location() -> None:
    world = create_dinner_party_world()
    locations = world.definition.locations
    events = [
        _make_event(
            "e1",
            event_type=EventType.CHAT,
            tick_id=1,
            sim_time=1.0,
            source="thorne",
            targets=["elena"],
            location="dining_table",
        ),
        _make_event(
            "m1",
            event_type=EventType.SOCIAL_MOVE,
            tick_id=2,
            sim_time=2.0,
            source="thorne",
            location="dining_table",
            destination="balcony",
        ),
        _make_event(
            "e2",
            event_type=EventType.CHAT,
            tick_id=3,
            sim_time=3.0,
            source="thorne",
            targets=["elena"],
            location="balcony",
        ),
    ]

    scenes = segment_into_scenes(events, locations=locations, config=SegmentationConfig(min_scene_size=1))
    assert len(scenes) == 2
    assert scenes[0].event_ids == ["e1"]
    assert scenes[1].event_ids == ["m1", "e2"]
    assert scenes[1].location == "balcony"


def test_consecutive_social_moves_to_different_destinations_split() -> None:
    world = create_dinner_party_world()
    locations = world.definition.locations
    events = [
        _make_event(
            "m1",
            event_type=EventType.SOCIAL_MOVE,
            tick_id=1,
            sim_time=1.0,
            source="elena",
            location="dining_table",
            destination="balcony",
        ),
        _make_event(
            "m2",
            event_type=EventType.SOCIAL_MOVE,
            tick_id=2,
            sim_time=2.0,
            source="elena",
            location="balcony",
            destination="kitchen",
        ),
    ]

    scenes = segment_into_scenes(events, locations=locations, config=SegmentationConfig(min_scene_size=1))
    assert len(scenes) == 2
    assert scenes[0].event_ids == ["m1"]
    assert scenes[1].event_ids == ["m2"]
    assert scenes[0].location == "balcony"
    assert scenes[1].location == "kitchen"
