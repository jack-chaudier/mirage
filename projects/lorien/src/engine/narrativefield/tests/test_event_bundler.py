"""Tests for the Narrative Bundler (event compression layer)."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

from narrativefield.integration.event_bundler import BundlerConfig, bundle_events
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.events import DeltaKind, DeltaOp, Event, EventMetrics, EventType, StateDelta
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation


def _make_event(
    eid: str,
    *,
    etype: EventType = EventType.CHAT,
    sim_time: float = 0.0,
    tick_id: int = 0,
    order: int = 0,
    source: str = "agent_a",
    targets: list[str] | None = None,
    location: str = "dining_room",
    causal_links: list[str] | None = None,
    deltas: list[StateDelta] | None = None,
    content_metadata: dict[str, Any] | None = None,
    significance: float = 0.0,
) -> Event:
    return Event(
        id=eid,
        sim_time=sim_time,
        tick_id=tick_id,
        order_in_tick=order,
        type=etype,
        source_agent=source,
        target_agents=targets or [],
        location_id=location,
        causal_links=causal_links or [],
        deltas=deltas or [],
        description=f"event {eid}",
        content_metadata=content_metadata,
        metrics=EventMetrics(significance=significance),
    )


class TestSocialMovePreservation:
    def test_consecutive_social_moves_are_preserved(self) -> None:
        events = [
            _make_event(
                "m1",
                etype=EventType.SOCIAL_MOVE,
                sim_time=0.0,
                tick_id=1,
                source="agent_a",
                location="dining_room",
                content_metadata={"destination": "balcony"},
            ),
            _make_event(
                "m2",
                etype=EventType.SOCIAL_MOVE,
                sim_time=1.0,
                tick_id=2,
                source="agent_a",
                location="balcony",
                content_metadata={"destination": "kitchen"},
            ),
            _make_event("c1", etype=EventType.CONFLICT, sim_time=2.0, tick_id=3, source="agent_a"),
        ]
        result = bundle_events(events, config=BundlerConfig(move_squash_dt=3.0, move_attach_dt=6.0))

        assert [e.id for e in result.events] == ["m1", "m2", "c1"]
        assert result.stats.moves_squashed == 0
        assert result.stats.moves_attached == 0
        assert not result.dropped_ids

    def test_social_move_not_attached_to_host_event(self) -> None:
        events = [
            _make_event(
                "m1",
                etype=EventType.SOCIAL_MOVE,
                sim_time=1.0,
                tick_id=1,
                source="agent_a",
                location="dining_room",
                content_metadata={"destination": "kitchen"},
            ),
            _make_event(
                "host",
                etype=EventType.CHAT,
                sim_time=2.0,
                tick_id=2,
                source="agent_a",
                targets=["agent_b"],
                location="kitchen",
                causal_links=["m1"],
            ),
        ]
        result = bundle_events(events, config=BundlerConfig(move_attach_dt=6.0))

        by_id = {e.id: e for e in result.events}
        assert "m1" in by_id
        assert "host" in by_id
        assert result.stats.moves_attached == 0
        assert "moves" not in (by_id["host"].content_metadata or {})


class TestObserveAttach:
    def test_low_importance_observe_attached_to_next_event(self) -> None:
        events = [
            _make_event(
                "obs1",
                etype=EventType.OBSERVE,
                sim_time=1.0,
                tick_id=1,
                source="agent_a",
                location="dining_room",
            ),
            _make_event(
                "chat1",
                etype=EventType.CHAT,
                sim_time=1.5,
                tick_id=2,
                source="agent_b",
                location="dining_room",
            ),
        ]
        result = bundle_events(events, config=BundlerConfig(observe_attach_dt=2.0))

        assert len(result.events) == 1
        host = result.events[0]
        assert host.id == "chat1"
        assert host.content_metadata is not None
        assert "witnesses" in host.content_metadata
        assert len(host.content_metadata["witnesses"]) == 1
        assert host.content_metadata["witnesses"][0]["observer"] == "agent_a"
        assert result.stats.observes_attached == 1

    def test_observe_with_belief_delta_preserved(self) -> None:
        belief_delta = StateDelta(
            kind=DeltaKind.BELIEF,
            agent="agent_a",
            attribute="knows_secret",
            op=DeltaOp.SET,
            value="suspects",
        )
        events = [
            _make_event(
                "obs1",
                etype=EventType.OBSERVE,
                sim_time=1.0,
                tick_id=1,
                source="agent_a",
                deltas=[belief_delta],
            ),
            _make_event("chat1", etype=EventType.CHAT, sim_time=1.5, tick_id=2, source="agent_b"),
        ]
        result = bundle_events(events)

        assert len(result.events) == 2
        assert result.stats.observes_attached == 0

    def test_observe_high_significance_preserved(self) -> None:
        events = [
            _make_event("obs1", etype=EventType.OBSERVE, sim_time=1.0, tick_id=1, significance=0.8),
            _make_event("chat1", etype=EventType.CHAT, sim_time=1.5, tick_id=2),
        ]
        result = bundle_events(events, config=BundlerConfig(observe_significance_threshold=0.3))

        assert len(result.events) == 2
        assert result.stats.observes_attached == 0

    def test_observe_different_location_not_attached(self) -> None:
        events = [
            _make_event("obs1", etype=EventType.OBSERVE, sim_time=1.0, tick_id=1, location="kitchen"),
            _make_event("chat1", etype=EventType.CHAT, sim_time=1.5, tick_id=2, location="balcony"),
        ]
        result = bundle_events(events, config=BundlerConfig(observe_attach_dt=2.0))

        assert len(result.events) == 2
        assert result.stats.observes_attached == 0


class TestCausalRewiring:
    def test_rewires_links_to_observe_host(self) -> None:
        events = [
            _make_event(
                "e0",
                etype=EventType.CHAT,
                sim_time=0.0,
                tick_id=0,
                source="agent_a",
                location="dining_room",
            ),
            _make_event(
                "obs1",
                etype=EventType.OBSERVE,
                sim_time=1.0,
                tick_id=1,
                source="agent_b",
                location="dining_room",
                causal_links=["e0"],
            ),
            _make_event(
                "host",
                etype=EventType.CHAT,
                sim_time=2.0,
                tick_id=2,
                source="agent_a",
                location="dining_room",
                causal_links=["obs1"],
            ),
            _make_event(
                "later",
                etype=EventType.CHAT,
                sim_time=3.0,
                tick_id=3,
                source="agent_a",
                location="dining_room",
                causal_links=["obs1"],
            ),
        ]
        result = bundle_events(events, config=BundlerConfig(observe_attach_dt=2.0))

        assert "obs1" in result.dropped_ids
        by_id = {e.id: e for e in result.events}
        assert "host" in by_id
        assert "later" in by_id
        assert "host" in by_id["later"].causal_links
        assert "obs1" not in by_id["later"].causal_links


class TestBundlerIntegration:
    def test_bundled_output_preserves_social_moves(self) -> None:
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
                "simulation_id": "bundler_test",
                "scenario": "dinner_party",
                "total_ticks": world.tick_id,
                "total_sim_time": world.sim_time,
                "agent_count": len(world.agents),
                "event_count": len(events),
                "snapshot_interval": world.definition.snapshot_interval,
                "timestamp": "2026-02-08T00:00:00Z",
            },
            "initial_state": snapshots[0],
            "snapshots": snapshots[1:],
            "events": [e.to_dict() for e in events],
            "secrets": [s.to_dict() for s in world.definition.secrets.values()],
            "locations": [loc.to_dict() for loc in world.definition.locations.values()],
        }

        parsed = parse_simulation_output(sim_output)
        out = run_metrics_pipeline(parsed)

        assert out.bundle_result is not None
        stats = out.bundle_result.stats
        assert stats.input_count == len(events)
        assert stats.output_count <= stats.input_count
        assert stats.output_count == len(out.events)
        assert stats.moves_squashed == 0
        assert stats.moves_attached == 0

        in_social_moves = [e for e in parsed.events if e.type == EventType.SOCIAL_MOVE]
        out_social_moves = [e for e in out.events if e.type == EventType.SOCIAL_MOVE]
        assert len(out_social_moves) == len(in_social_moves)

        bundled_ids = {e.id for e in out.events}
        scene_event_ids: set[str] = set()
        for sc in out.scenes:
            for eid in sc.event_ids:
                assert eid in bundled_ids, f"Scene references unknown event {eid}"
                assert eid not in scene_event_ids, f"Event {eid} in multiple scenes"
                scene_event_ids.add(eid)
        assert bundled_ids == scene_event_ids

        dropped = out.bundle_result.dropped_ids
        for e in out.events:
            for link in e.causal_links:
                assert link not in dropped, f"Event {e.id} still references dropped event {link}"

    def test_bundler_on_real_artifact_file(self) -> None:
        artifact = Path(__file__).resolve().parents[4] / "data" / "dinner_party_001.nf-sim.json"
        if not artifact.exists():
            return

        raw = json.loads(artifact.read_text(encoding="utf-8"))
        events = [Event.from_dict(e) for e in raw["events"]]
        input_social_moves = [e for e in events if e.type == EventType.SOCIAL_MOVE]

        result = bundle_events(events)
        output_social_moves = [e for e in result.events if e.type == EventType.SOCIAL_MOVE]

        assert len(output_social_moves) == len(input_social_moves)
        assert result.stats.moves_squashed == 0
        assert result.stats.moves_attached == 0

        for e in result.events:
            for link in e.causal_links:
                assert link not in result.dropped_ids

    def test_empty_input(self) -> None:
        result = bundle_events([])
        assert result.events == []
        assert result.stats.input_count == 0
        assert result.stats.output_count == 0

    def test_no_attachable_events_passthrough(self) -> None:
        events = [
            _make_event("e1", etype=EventType.CHAT, sim_time=0.0, tick_id=1),
            _make_event("e2", etype=EventType.CONFLICT, sim_time=1.0, tick_id=2),
            _make_event("e3", etype=EventType.REVEAL, sim_time=2.0, tick_id=3),
        ]
        result = bundle_events(events)

        assert len(result.events) == 3
        assert result.stats.moves_squashed == 0
        assert result.stats.moves_attached == 0
        assert result.stats.observes_attached == 0
