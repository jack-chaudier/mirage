from __future__ import annotations

from random import Random

import logging

from narrativefield.integration.bundler import BundleInputs, bundle_for_renderer
from narrativefield.integration.event_bundler import BundlerConfig, bundle_events
from narrativefield.metrics.pipeline import compute_significance, parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.events import DeltaKind, DeltaOp, Event, EventMetrics, EventType, StateDelta
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation


def test_metrics_pipeline_builds_valid_renderer_payload_smoke() -> None:
    world = create_dinner_party_world()
    rng = Random(42)
    cfg = SimulationConfig(
        tick_limit=120,
        event_limit=80,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )

    events, snapshots = run_simulation(world, rng, cfg)

    sim_output = {
        "format_version": "1.0.0",
        "metadata": {
            "simulation_id": "sim_test",
            "scenario": "dinner_party",
            "total_ticks": world.tick_id,
            "total_sim_time": world.sim_time,
            "agent_count": len(world.agents),
            "event_count": len(events),
            "snapshot_interval": world.definition.snapshot_interval,
            "timestamp": "2026-02-07T00:00:00Z",
        },
        "initial_state": snapshots[0],
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
        "world_canon": world.canon.to_dict() if world.canon is not None else None,
    }

    parsed = parse_simulation_output(sim_output)
    out = run_metrics_pipeline(parsed)
    payload = bundle_for_renderer(
        BundleInputs(
            metadata=parsed.metadata,
            initial_agents=parsed.initial_agents,
            snapshots=out.belief_snapshots,
            events=out.events,
            scenes=out.scenes,
            secrets=parsed.secrets,
            locations=parsed.locations,
            world_canon=parsed.world_canon.to_dict() if parsed.world_canon is not None else None,
        )
    )

    assert payload["format_version"] == "1.0.0"
    # After bundling, payload events may be fewer than the original sim events.
    assert len(payload["events"]) <= len(events)
    assert payload["metadata"]["event_count"] == len(payload["events"])
    assert payload["metadata"].get("raw_event_count") == len(events)
    assert payload["metadata"]["agent_count"] == len(payload["agents"]) == len(world.agents)
    assert isinstance(payload.get("world_canon"), dict)

    # All events have populated tension components with canonical keys in [0,1].
    keys = {
        "danger",
        "time_pressure",
        "goal_frustration",
        "relationship_volatility",
        "information_gap",
        "resource_scarcity",
        "moral_cost",
        "irony_density",
    }
    for e in payload["events"]:
        assert isinstance(e, dict)
        metrics = e["metrics"]
        comps = metrics["tension_components"]
        assert set(comps.keys()) == keys
        for v in comps.values():
            assert 0.0 <= float(v) <= 1.0
        assert 0.0 <= float(metrics["tension"]) <= 1.0

    # Scenes cover all events exactly once.
    event_ids = {e["id"] for e in payload["events"]}
    seen: dict[str, str] = {}
    for sc in payload["scenes"]:
        for eid in sc["event_ids"]:
            assert eid in event_ids
            assert eid not in seen, f"event {eid} appears in multiple scenes ({seen[eid]}, {sc['id']})"
            seen[eid] = sc["id"]
    assert event_ids == set(seen.keys())

    # Belief snapshots are present and shaped correctly.
    assert payload["belief_snapshots"], "expected belief_snapshots"
    allowed = {"unknown", "suspects", "believes_true", "believes_false"}
    for snap in payload["belief_snapshots"]:
        beliefs = snap["beliefs"]
        for row in beliefs.values():
            for state in row.values():
                assert state in allowed


def test_significance_populated_and_observe_threshold_is_meaningful_seed42() -> None:
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
            "simulation_id": "sim_significance_test",
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
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }

    parsed = parse_simulation_output(sim_output)
    run_metrics_pipeline(parsed)

    # Significance should be populated with non-trivial variation and spread.
    significance_values = [float(e.metrics.significance) for e in parsed.events]
    sorted_values = sorted(significance_values)
    q10 = sorted_values[int(len(sorted_values) * 0.10)]
    q90 = sorted_values[max(0, int(len(sorted_values) * 0.90) - 1)]
    assert max(significance_values) > 0.3
    assert min(significance_values) < 0.2
    assert (max(significance_values) - min(significance_values)) > 0.35
    assert (q90 - q10) > 0.12

    top5 = sorted(parsed.events, key=lambda e: float(e.metrics.significance), reverse=True)[:5]
    assert any(e.type == EventType.CATASTROPHE for e in top5)
    assert any(
        any(
            d.kind == DeltaKind.BELIEF
            and isinstance(d.value, str)
            and d.value in {"suspects", "believes_true", "believes_false"}
            for d in e.deltas
        )
        for e in top5
    )

    def _has_meaningful_delta(event) -> bool:
        return any(
            d.kind in {DeltaKind.BELIEF, DeltaKind.RELATIONSHIP, DeltaKind.SECRET_STATE}
            for d in event.deltas
        )

    raw_plain_observes = [e for e in parsed.events if e.type == EventType.OBSERVE and not _has_meaningful_delta(e)]
    assert any(float(e.metrics.significance) < 0.3 for e in raw_plain_observes)

    # Synthetic sanity check: significance threshold should separate low/high OBSERVE events.
    low = Event(
        id="obs_low",
        sim_time=1.0,
        tick_id=1,
        order_in_tick=0,
        type=EventType.OBSERVE,
        source_agent="thorne",
        target_agents=[],
        location_id="dining_table",
        causal_links=[],
        deltas=[],
        description="low significance observe",
        metrics=EventMetrics(tension=0.05),
    )
    high_belief = StateDelta(
        kind=DeltaKind.BELIEF,
        agent="thorne",
        attribute="secret_affair_01",
        op=DeltaOp.SET,
        value="believes_true",
    )
    high = Event(
        id="obs_high",
        sim_time=1.2,
        tick_id=2,
        order_in_tick=0,
        type=EventType.OBSERVE,
        source_agent="thorne",
        target_agents=["elena", "marcus", "lydia"],
        location_id="dining_table",
        causal_links=["seed_anchor"],
        deltas=[high_belief],
        description="high significance observe",
        metrics=EventMetrics(tension=0.95),
    )
    host = Event(
        id="host",
        sim_time=1.4,
        tick_id=3,
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent="thorne",
        target_agents=["elena"],
        location_id="dining_table",
        causal_links=["obs_low", "obs_high"],
        deltas=[],
        description="host event",
        metrics=EventMetrics(tension=0.20),
    )
    synthetic = [low, high, host]
    compute_significance(synthetic)
    bundled = bundle_events(synthetic, BundlerConfig(observe_attach_dt=2.0, observe_significance_threshold=0.3))
    kept_ids = {e.id for e in bundled.events}

    assert low.metrics.significance < 0.3
    assert high.metrics.significance >= 0.3
    assert "obs_low" not in kept_ids
    assert "obs_high" in kept_ids


def test_pipeline_populates_significance_for_small_event_set() -> None:
    world = create_dinner_party_world()
    rng = Random(19)
    cfg = SimulationConfig(
        tick_limit=60,
        event_limit=35,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, rng, cfg)

    sim_output = {
        "format_version": "1.0.0",
        "metadata": {
            "simulation_id": "sim_small_significance_test",
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
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }

    parsed = parse_simulation_output(sim_output)
    run_metrics_pipeline(parsed)

    significance_values = [float(e.metrics.significance) for e in parsed.events]
    assert significance_values
    assert all(0.0 <= value <= 1.0 for value in significance_values)
    assert any(value > 0.0 for value in significance_values)


def test_pipeline_warns_on_duplicate_ids_and_dangling_links(caplog) -> None:
    world = create_dinner_party_world()
    rng = Random(7)
    cfg = SimulationConfig(
        tick_limit=80,
        event_limit=40,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, rng, cfg)
    assert events

    duplicate = events[0]
    duplicated_events = [*events, duplicate]
    # Add a dangling causal ref to an existing event.
    duplicated_events[1].causal_links = [*duplicated_events[1].causal_links, "evt_missing_parent"]

    sim_output = {
        "format_version": "1.0.0",
        "metadata": {
            "simulation_id": "sim_warn_test",
            "scenario": "dinner_party",
            "total_ticks": world.tick_id,
            "total_sim_time": world.sim_time,
            "agent_count": len(world.agents),
            "event_count": len(duplicated_events),
            "snapshot_interval": world.definition.snapshot_interval,
            "timestamp": "2026-02-11T00:00:00Z",
        },
        "initial_state": snapshots[0],
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in duplicated_events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }

    parsed = parse_simulation_output(sim_output)
    with caplog.at_level(logging.WARNING):
        run_metrics_pipeline(parsed)

    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "duplicate event ids" in warning_text.lower()
    assert "dangling causal links" in warning_text.lower()
