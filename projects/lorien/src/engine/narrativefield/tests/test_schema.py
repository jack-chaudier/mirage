from __future__ import annotations

from narrativefield.schema import (
    AgentState,
    BeliefState,
    BeatType,
    CharacterFlaw,
    DeltaKind,
    DeltaOp,
    Event,
    EventMetrics,
    EventType,
    FlawType,
    GoalVector,
    IronyCollapseInfo,
    PacingState,
    RelationshipState,
    SecretDefinition,
    StateDelta,
    build_index_tables,
)


def test_event_roundtrip() -> None:
    e = Event(
        id="evt_0001",
        sim_time=1.5,
        tick_id=1,
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent="a",
        target_agents=["b"],
        location_id="loc",
        causal_links=[],
        deltas=[
            StateDelta(
                kind=DeltaKind.RELATIONSHIP,
                agent="a",
                agent_b="b",
                attribute="trust",
                op=DeltaOp.ADD,
                value=-0.2,
                reason_code="TEST",
                reason_display="test",
            )
        ],
        description="hello",
        beat_type=BeatType.SETUP,
        metrics=EventMetrics(
            tension=0.1,
            irony=0.0,
            significance=0.0,
            thematic_shift={"truth_deception": 0.2},
            tension_components={"danger": 0.1},
            irony_collapse=IronyCollapseInfo(detected=False, drop=0.0, collapsed_beliefs=[], score=0.0),
        ),
    )

    data = e.to_dict()
    e2 = Event.from_dict(data)
    assert e2.to_dict() == data


def test_agent_roundtrip() -> None:
    a = AgentState(
        id="a",
        name="Agent A",
        location="loc",
        goals=GoalVector(
            safety=0.5,
            status=0.5,
            closeness={"b": 0.2},
            secrecy=0.5,
            truth_seeking=0.5,
            autonomy=0.5,
            loyalty=0.5,
        ),
        flaws=[
            CharacterFlaw(
                flaw_type=FlawType.PRIDE,
                strength=0.8,
                trigger="status_threat",
                effect="overweight_status",
                description="test",
            )
        ],
        pacing=PacingState(),
        emotional_state={"anger": 0.1},
        relationships={"b": RelationshipState(trust=0.25, affection=0.0, obligation=0.0)},
        beliefs={"secret_1": BeliefState.UNKNOWN},
        alcohol_level=0.0,
        commitments=[],
    )

    data = a.to_dict()
    a2 = AgentState.from_dict(data)
    assert a2.to_dict() == data


def test_build_index_tables() -> None:
    e1 = Event(
        id="evt_0001",
        sim_time=0.0,
        tick_id=0,
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent="a",
        target_agents=["b"],
        location_id="loc",
        causal_links=[],
        deltas=[],
        description="hello",
    )
    e2 = Event(
        id="evt_0002",
        sim_time=1.5,
        tick_id=1,
        order_in_tick=0,
        type=EventType.OBSERVE,
        source_agent="b",
        target_agents=[],
        location_id="loc",
        causal_links=["evt_0001"],
        deltas=[
            StateDelta(
                kind=DeltaKind.BELIEF,
                agent="b",
                attribute="secret_1",
                op=DeltaOp.SET,
                value="suspects",
            )
        ],
        description="observe",
    )

    tables = build_index_tables([e1, e2])
    assert tables.event_by_id["evt_0002"].id == "evt_0002"
    assert tables.participants_by_event["evt_0001"] == ["a", "b"]
    assert tables.events_by_agent["a"] == ["evt_0001"]
    assert tables.events_by_agent["b"] == ["evt_0001", "evt_0002"]
    assert tables.events_by_location["loc"] == ["evt_0001", "evt_0002"]
    assert tables.events_by_secret["secret_1"] == ["evt_0002"]
    assert tables.events_by_pair[("a", "b")] == ["evt_0001"]
    assert tables.forward_links["evt_0001"] == ["evt_0002"]


def test_secret_holder_string_backcompat() -> None:
    raw = {
        "id": "secret_1",
        "holder": "a",
        "about": None,
        "content_type": "knowledge",
        "description": "x",
        "truth_value": True,
        "initial_knowers": ["a"],
        "initial_suspecters": [],
        "dramatic_weight": 0.5,
        "reveal_consequences": "y",
    }
    s = SecretDefinition.from_dict(raw)
    assert s.holder == ["a"]

