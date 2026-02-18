from __future__ import annotations

from narrativefield.metrics.significance import compute_significance
from narrativefield.schema.events import DeltaKind, DeltaOp, Event, EventType, StateDelta


def _event(
    eid: str,
    *,
    sim_time: float,
    tick_id: int,
    etype: EventType = EventType.CHAT,
    source: str = "agent_a",
    targets: list[str] | None = None,
    location: str = "dining_table",
    causal_links: list[str] | None = None,
    deltas: list[StateDelta] | None = None,
) -> Event:
    return Event(
        id=eid,
        sim_time=sim_time,
        tick_id=tick_id,
        order_in_tick=0,
        type=etype,
        source_agent=source,
        target_agents=targets or [],
        location_id=location,
        causal_links=causal_links or [],
        deltas=deltas or [],
        description=f"event {eid}",
    )


def test_event_with_no_deltas_and_no_causal_links_scores_near_zero() -> None:
    lead = _event("lead", sim_time=1.0, tick_id=1, location="kitchen")
    quiet = _event("quiet", sim_time=2.0, tick_id=2, location="kitchen")

    score = compute_significance(quiet, [lead, quiet])

    assert 0.0 <= score < 0.1


def test_high_impact_event_scores_near_one() -> None:
    parents = [
        _event("p1", sim_time=1.0, tick_id=1, location="dining_table"),
        _event("p2", sim_time=2.0, tick_id=2, location="dining_table"),
        _event("p3", sim_time=3.0, tick_id=3, location="kitchen"),
        _event("p4", sim_time=4.0, tick_id=4, location="foyer"),
    ]
    high = _event(
        "high",
        sim_time=5.0,
        tick_id=5,
        etype=EventType.CATASTROPHE,
        source="agent_a",
        targets=["agent_b", "agent_c", "agent_d"],
        location="balcony",
        causal_links=["p1", "p2", "p3", "p4"],
        deltas=[
            StateDelta(
                kind=DeltaKind.BELIEF,
                agent="agent_b",
                attribute="secret_1",
                op=DeltaOp.SET,
                value="believes_true",
            ),
            StateDelta(
                kind=DeltaKind.BELIEF,
                agent="agent_c",
                attribute="secret_2",
                op=DeltaOp.SET,
                value="believes_false",
            ),
            StateDelta(
                kind=DeltaKind.RELATIONSHIP,
                agent="agent_a",
                agent_b="agent_b",
                attribute="trust",
                op=DeltaOp.ADD,
                value=-0.8,
            ),
            StateDelta(
                kind=DeltaKind.COMMITMENT,
                agent="agent_a",
                attribute="commitment",
                op=DeltaOp.SET,
                value="vow",
            ),
            StateDelta(
                kind=DeltaKind.AGENT_EMOTION,
                agent="agent_a",
                attribute="anger",
                op=DeltaOp.ADD,
                value=0.25,
            ),
            StateDelta(
                kind=DeltaKind.WORLD_RESOURCE,
                agent="world",
                attribute="social_order",
                op=DeltaOp.ADD,
                value=-1.0,
            ),
        ],
    )
    followers = [
        _event(
            "c1",
            sim_time=6.0,
            tick_id=6,
            etype=EventType.OBSERVE,
            source="agent_e",
            causal_links=["high"],
        ),
        _event(
            "c2",
            sim_time=7.0,
            tick_id=7,
            etype=EventType.CHAT,
            source="agent_f",
            targets=["agent_a"],
            causal_links=["high"],
        ),
    ]

    events = [*parents, high, *followers]
    score = compute_significance(high, events)

    assert score >= 0.95


def test_compute_significance_is_deterministic() -> None:
    e1 = _event("e1", sim_time=3.0, tick_id=3, location="foyer")
    e2 = _event(
        "e2",
        sim_time=2.0,
        tick_id=2,
        etype=EventType.REVEAL,
        source="agent_b",
        targets=["agent_a"],
        causal_links=["e1"],
        deltas=[
            StateDelta(
                kind=DeltaKind.BELIEF,
                agent="agent_a",
                attribute="secret_1",
                op=DeltaOp.SET,
                value="believes_true",
            )
        ],
    )
    e3 = _event("e3", sim_time=1.0, tick_id=1, location="kitchen")

    events = [e1, e2, e3]
    score_a = compute_significance(e2, events)
    score_b = compute_significance(e2, list(reversed(events)))
    score_c = compute_significance(e2, events)

    assert score_a == score_b == score_c


def test_edge_cases_empty_targets_and_no_causal_links() -> None:
    base = _event("base", sim_time=1.0, tick_id=1, location="dining_table")
    empty_targets = _event(
        "empty_targets",
        sim_time=2.0,
        tick_id=2,
        source="agent_x",
        targets=[],
        location="kitchen",
        causal_links=["base"],
    )
    no_causal_links = _event(
        "no_causal",
        sim_time=3.0,
        tick_id=3,
        source="agent_y",
        targets=["agent_x"],
        location="kitchen",
        causal_links=[],
    )
    all_events = [base, empty_targets, no_causal_links]

    score_empty_targets = compute_significance(empty_targets, all_events)
    score_no_causal_links = compute_significance(no_causal_links, all_events)

    assert 0.0 <= score_empty_targets <= 1.0
    assert 0.0 <= score_no_causal_links <= 1.0
