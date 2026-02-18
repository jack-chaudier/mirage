from __future__ import annotations

from narrativefield.schema.events import DeltaKind, Event, EventMetrics, EventType
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, apply_tick_updates


def test_pacing_deltas_only_attach_to_same_source_agent_events() -> None:
    world = create_dinner_party_world()
    event = Event(
        id="evt_test",
        sim_time=0.0,
        tick_id=1,
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent="thorne",
        target_agents=["elena"],
        location_id="dining_table",
        causal_links=[],
        deltas=[],
        description="thorne chats with elena",
        metrics=EventMetrics(),
    )

    tick_events = [event]
    apply_tick_updates(world, tick_events, SimulationConfig())

    pacing_deltas = [d for d in tick_events[0].deltas if d.kind == DeltaKind.PACING]
    assert pacing_deltas, "expected pacing deltas on source-agent event"
    assert all(d.agent == "thorne" for d in pacing_deltas)
