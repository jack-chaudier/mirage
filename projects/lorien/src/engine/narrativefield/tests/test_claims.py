from __future__ import annotations

import json
from hashlib import sha256
from random import Random

import pytest

from narrativefield.metrics import irony, tension
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import AgentState, BeliefState, GoalVector, PacingState
from narrativefield.schema.events import DeltaKind, DeltaOp, Event, EventMetrics, EventType, StateDelta
from narrativefield.schema.world import ClaimDefinition, Location, SecretDefinition, WorldDefinition
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, apply_delta, generate_witness_events, run_simulation
from narrativefield.simulation.types import WorldState


def _loc(location_id: str = "dining_table") -> Location:
    return Location(
        id=location_id,
        name="Dining Table",
        privacy=0.1,
        capacity=6,
        adjacent=[],
        overhear_from=[],
        overhear_probability=0.0,
        description="Shared space",
    )


def _agent(
    agent_id: str,
    *,
    beliefs: dict[str, BeliefState] | None = None,
    location: str = "dining_table",
) -> AgentState:
    return AgentState(
        id=agent_id,
        name=agent_id.title(),
        location=location,
        goals=GoalVector(),
        flaws=[],
        pacing=PacingState(),
        emotional_state={},
        relationships={},
        beliefs=dict(beliefs or {}),
        alcohol_level=0.0,
        commitments=[],
    )


def _event(
    event_id: str,
    *,
    tick_id: int,
    source: str,
    targets: list[str],
    deltas: list[StateDelta],
    location_id: str = "dining_table",
    event_type: EventType = EventType.CHAT,
    sim_time: float | None = None,
) -> Event:
    return Event(
        id=event_id,
        sim_time=float(tick_id if sim_time is None else sim_time),
        tick_id=tick_id,
        order_in_tick=0,
        type=event_type,
        source_agent=source,
        target_agents=targets,
        location_id=location_id,
        causal_links=[],
        deltas=deltas,
        description=f"{event_type.value} event",
        metrics=EventMetrics(),
    )


def _secret(secret_id: str, *, truth: bool = True) -> SecretDefinition:
    return SecretDefinition(
        id=secret_id,
        holder=["a"],
        about="a",
        content_type="financial",
        description=f"Secret {secret_id}",
        truth_value=truth,
        initial_knowers=["a"],
        initial_suspecters=[],
        dramatic_weight=0.8,
        reveal_consequences="Consequence",
    )


def _claim(claim_id: str, *, truth_status: str = "true", scope: str = "private", propagation_rate: float = 0.0) -> ClaimDefinition:
    return ClaimDefinition(
        id=claim_id,
        description=f"Claim {claim_id}",
        truth_status=truth_status,
        claim_type="rumor",
        scope=scope,
        holder=["a"],
        about="a",
        content_type="financial",
        source_event_ids=[],
        initial_knowers=["a"],
        initial_suspecters=[],
        dramatic_weight=0.8,
        reveal_consequences="Consequence",
        propagation_rate=propagation_rate,
        decay_rate=0.1,
    )


def test_claim_definition_roundtrip() -> None:
    claim = _claim("claim_roundtrip", truth_status="contested", scope="location:dining_table", propagation_rate=0.7)
    assert ClaimDefinition.from_dict(claim.to_dict()) == claim


def test_secret_to_claim_conversion() -> None:
    secret = _secret("secret_conversion", truth=True)
    claim = secret.to_claim()
    assert claim.id == secret.id
    assert claim.description == secret.description
    assert claim.truth_status == "true"
    assert claim.claim_type == "secret"
    assert claim.scope == "private"
    assert claim.holder == secret.holder
    assert claim.about == secret.about
    assert claim.content_type == secret.content_type
    assert claim.initial_knowers == secret.initial_knowers
    assert claim.initial_suspecters == secret.initial_suspecters
    assert claim.dramatic_weight == secret.dramatic_weight
    assert claim.reveal_consequences == secret.reveal_consequences
    assert claim.propagation_rate == 0.0
    assert claim.decay_rate == 0.0
    assert claim.source_event_ids == []


def test_world_definition_all_claims_merges() -> None:
    s1 = _secret("secret_1", truth=True)
    s2 = _secret("secret_2", truth=False)
    c1 = _claim("claim_1", truth_status="unknown")
    world = WorldDefinition(
        id="w",
        name="world",
        description="desc",
        sim_duration_minutes=60.0,
        ticks_per_minute=2.0,
        locations={"dining_table": _loc()},
        secrets={s1.id: s1, s2.id: s2},
        claims={c1.id: c1},
    )
    merged = world.all_claims
    assert set(merged.keys()) == {"secret_1", "secret_2", "claim_1"}
    assert merged["secret_1"].claim_type == "secret"
    assert merged["secret_2"].truth_status == "false"
    assert merged["claim_1"] == c1


def test_world_definition_claims_backward_compat() -> None:
    raw = create_dinner_party_world().definition.to_dict()
    raw.pop("claims", None)
    parsed = WorldDefinition.from_dict(raw)
    assert parsed.claims == {}
    assert set(parsed.all_claims.keys()) == set(parsed.secrets.keys())


def test_claim_irony_true_status() -> None:
    secret = _secret("x", truth=True)
    claim = secret.to_claim()
    beliefs = {
        "a": {"x": BeliefState.BELIEVES_FALSE},
        "b": {"x": BeliefState.UNKNOWN},
    }
    secret_per_agent, secret_scene = irony.compute_snapshot_irony(beliefs=beliefs, secrets={"x": secret})
    claim_per_agent, claim_scene = irony.compute_snapshot_irony(
        beliefs=beliefs,
        secrets={},
        claims={"x": claim},
    )
    assert claim_scene == pytest.approx(secret_scene)
    assert claim_per_agent == pytest.approx(secret_per_agent)


def test_claim_irony_unknown_status() -> None:
    claim = _claim("c_unknown", truth_status="unknown")
    beliefs = {
        "a": {"c_unknown": BeliefState.BELIEVES_TRUE},
        "b": {"c_unknown": BeliefState.BELIEVES_FALSE},
    }
    per_agent, scene = irony.compute_snapshot_irony(beliefs=beliefs, secrets={}, claims={"c_unknown": claim})
    assert scene == 0.0
    assert all(score == 0.0 for score in per_agent.values())


def test_claim_irony_contested_status() -> None:
    claim = _claim("c_contested", truth_status="contested")
    beliefs = {
        "a": {"c_contested": BeliefState.BELIEVES_TRUE},
        "b": {"c_contested": BeliefState.BELIEVES_FALSE},
    }
    per_agent, scene = irony.compute_snapshot_irony(beliefs=beliefs, secrets={}, claims={"c_contested": claim})
    assert per_agent["a"] == pytest.approx(0.5)
    assert per_agent["b"] == pytest.approx(0.25)
    assert scene == pytest.approx(0.375)


def test_claim_information_gap_public_scope() -> None:
    claim = _claim("c_public", truth_status="true", scope="public")
    agents = {
        "a": _agent("a", beliefs={"c_public": BeliefState.BELIEVES_TRUE}),
        "b": _agent("b", beliefs={"c_public": BeliefState.BELIEVES_FALSE}),
    }
    event = _event("evt_1", tick_id=1, source="a", targets=["b"], deltas=[])
    tension.run_tension_pipeline([event], agents=agents, secrets={}, claims={"c_public": claim})
    assert event.metrics.tension_components["information_gap"] == 0.0


def test_claim_information_gap_private_scope() -> None:
    secret = _secret("shared_id", truth=True)
    claim = secret.to_claim()
    agents_secret = {
        "a": _agent("a", beliefs={"shared_id": BeliefState.BELIEVES_TRUE}),
        "b": _agent("b", beliefs={"shared_id": BeliefState.BELIEVES_FALSE}),
    }
    agents_claim = {
        "a": _agent("a", beliefs={"shared_id": BeliefState.BELIEVES_TRUE}),
        "b": _agent("b", beliefs={"shared_id": BeliefState.BELIEVES_FALSE}),
    }
    event_secret = _event("evt_secret", tick_id=1, source="a", targets=["b"], deltas=[])
    event_claim = _event("evt_claim", tick_id=1, source="a", targets=["b"], deltas=[])

    tension.run_tension_pipeline([event_secret], agents=agents_secret, secrets={"shared_id": secret})
    tension.run_tension_pipeline([event_claim], agents=agents_claim, secrets={}, claims={"shared_id": claim})

    assert event_claim.metrics.tension_components["information_gap"] == pytest.approx(
        event_secret.metrics.tension_components["information_gap"]
    )


def test_belief_delta_works_for_claims() -> None:
    claim = _claim("claim_delta", truth_status="true")
    world = WorldState(
        definition=WorldDefinition(
            id="w_claim",
            name="Claim World",
            description="desc",
            sim_duration_minutes=60.0,
            ticks_per_minute=2.0,
            locations={"dining_table": _loc()},
            secrets={},
            claims={claim.id: claim},
        ),
        agents={"a": _agent("a")},
    )
    delta = StateDelta(
        kind=DeltaKind.BELIEF,
        agent="a",
        attribute="claim_delta",
        op=DeltaOp.SET,
        value=BeliefState.BELIEVES_TRUE.value,
    )
    apply_delta(world, delta)
    assert world.agents["a"].beliefs["claim_delta"] == BeliefState.BELIEVES_TRUE


def test_rumor_propagation_from_claim_belief_delta() -> None:
    claim = _claim("claim_spread", truth_status="unknown", propagation_rate=1.0)
    world = WorldState(
        definition=WorldDefinition(
            id="w_rumor",
            name="Rumor World",
            description="desc",
            sim_duration_minutes=60.0,
            ticks_per_minute=2.0,
            locations={"dining_table": _loc()},
            secrets={},
            claims={claim.id: claim},
        ),
        agents={
            "a": _agent("a", beliefs={"claim_spread": BeliefState.BELIEVES_TRUE}),
            "b": _agent("b", beliefs={}),
            "c": _agent("c", beliefs={}),
        },
    )
    primary = _event(
        "evt_chat",
        tick_id=1,
        source="a",
        targets=["b"],
        event_type=EventType.CHAT,
        deltas=[
            StateDelta(
                kind=DeltaKind.BELIEF,
                agent="b",
                agent_b="a",
                attribute="claim_spread",
                op=DeltaOp.SET,
                value=BeliefState.SUSPECTS.value,
            )
        ],
    )
    witnesses = generate_witness_events([primary], world, tick_id=1, start_order=1, rng=Random(7))
    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.type == EventType.OBSERVE
    assert witness.causal_links == ["evt_chat"]
    belief_deltas = [d for d in witness.deltas if d.kind == DeltaKind.BELIEF]
    assert len(belief_deltas) == 1
    assert belief_deltas[0].attribute == "claim_spread"
    assert belief_deltas[0].value == BeliefState.SUSPECTS.value


def test_existing_dinner_party_unchanged() -> None:
    world = create_dinner_party_world()
    rng = Random(42)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, rng, cfg)
    sim_output = {
        "format_version": "1.0.0",
        "metadata": {
            "simulation_id": "sim_seed_42_claims_regression",
            "scenario": "dinner_party",
            "total_ticks": world.tick_id,
            "total_sim_time": world.sim_time,
            "agent_count": len(world.agents),
            "event_count": len(events),
            "snapshot_interval": world.definition.snapshot_interval,
            "timestamp": "2026-02-11T00:00:00Z",
        },
        "initial_state": snapshots[0],
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "claims": [c.to_dict() for c in world.definition.claims.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }

    parsed = parse_simulation_output(sim_output)
    out = run_metrics_pipeline(parsed)

    rows: list[list[object]] = []
    collapse: list[list[object]] = []
    for event in out.events:
        info_gap = float((event.metrics.tension_components or {}).get("information_gap", 0.0))
        rows.append([event.id, round(info_gap, 6), round(float(event.metrics.irony), 6)])
        if event.metrics.irony_collapse and event.metrics.irony_collapse.detected:
            collapse.append(
                [
                    event.id,
                    round(float(event.metrics.irony_collapse.drop), 6),
                    len(event.metrics.irony_collapse.collapsed_beliefs),
                ]
            )

    payload = {"rows": rows, "collapse": collapse, "count": len(out.events)}
    fingerprint = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    assert len(out.events) == 174
    assert len(collapse) == 21
    assert fingerprint == "eb25d5c288a923f3cdb7b2383b9a3cd336d854b5172295326e527de1bba442ee"


def test_claims_in_metrics_pipeline() -> None:
    loc = _loc()
    secret = _secret("secret_x", truth=True)
    claim = _claim("claim_x", truth_status="false", propagation_rate=0.5)

    initial_agents = {
        "a": _agent("a", beliefs={"secret_x": BeliefState.UNKNOWN, "claim_x": BeliefState.UNKNOWN}),
        "b": _agent("b", beliefs={"secret_x": BeliefState.UNKNOWN, "claim_x": BeliefState.UNKNOWN}),
        "c": _agent("c", beliefs={"secret_x": BeliefState.UNKNOWN, "claim_x": BeliefState.UNKNOWN}),
    }
    events = [
        _event(
            "evt_1",
            tick_id=1,
            source="a",
            targets=["b"],
            deltas=[
                StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent="b",
                    agent_b="a",
                    attribute="secret_x",
                    op=DeltaOp.SET,
                    value=BeliefState.SUSPECTS.value,
                )
            ],
        ),
        _event(
            "evt_2",
            tick_id=2,
            source="b",
            targets=["c"],
            deltas=[
                StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent="c",
                    agent_b="b",
                    attribute="claim_x",
                    op=DeltaOp.SET,
                    value=BeliefState.BELIEVES_TRUE.value,
                )
            ],
        ),
        _event(
            "evt_3",
            tick_id=3,
            source="c",
            targets=["a"],
            deltas=[],
        ),
    ]

    sim_output = {
        "format_version": "1.0.0",
        "metadata": {
            "simulation_id": "sim_claims_pipeline",
            "scenario": "test",
            "total_ticks": 3,
            "total_sim_time": 3.0,
            "agent_count": len(initial_agents),
            "event_count": len(events),
            "snapshot_interval": 20,
            "timestamp": "2026-02-11T00:00:00Z",
        },
        "initial_state": {"agents": {aid: a.to_dict() for aid, a in initial_agents.items()}},
        "snapshots": [],
        "events": [e.to_dict() for e in events],
        "secrets": [secret.to_dict()],
        "claims": [claim.to_dict()],
        "locations": [loc.to_dict()],
    }

    parsed = parse_simulation_output(sim_output)
    out = run_metrics_pipeline(parsed)

    assert any(float(e.metrics.irony) > 0.0 for e in out.events)
    assert any(float((e.metrics.tension_components or {}).get("information_gap", 0.0)) > 0.0 for e in out.events)
