from __future__ import annotations

import json
from pathlib import Path
from random import Random

from narrativefield.simulation import run as run_cli
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, apply_tick_updates, run_simulation


def _run_world(seed: int, *, event_limit: int = 200, tick_limit: int = 300):
    world = create_dinner_party_world()
    rng = Random(seed)
    cfg = SimulationConfig(
        tick_limit=tick_limit,
        event_limit=event_limit,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, rng, cfg)
    return world, events, snapshots


def test_location_memory_staining() -> None:
    world, events, _ = _run_world(42)
    catastrophe_locations = {e.location_id for e in events if e.type.value == "catastrophe"}
    assert catastrophe_locations, "expected at least one catastrophe location"
    assert world.canon is not None

    for location_id in catastrophe_locations:
        mem = world.canon.location_memory.get(location_id)
        assert mem is not None
        assert mem.tension_residue > 0.0


def test_location_memory_decay() -> None:
    world, _, _ = _run_world(42, event_limit=40, tick_limit=80)
    assert world.canon is not None

    location_id = sorted(world.canon.location_memory)[0]
    world.canon.location_memory[location_id].tension_residue = 1.0
    before = world.canon.location_memory[location_id].tension_residue

    apply_tick_updates(world, [], SimulationConfig())
    after = world.canon.location_memory[location_id].tension_residue

    assert after < before
    assert after == before * 0.97


def test_entity_refs_auto_populated() -> None:
    _, events, _ = _run_world(42)
    assert events

    assert all(e.entities.locations for e in events)
    assert all(e.location_id in e.entities.locations for e in events)
    assert any(e.entities.claims for e in events)


def test_canon_in_simulation_output(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = Path(tmp_path) / "sim.json"
    rc = run_cli.main(
        [
            "--scenario",
            "dinner_party",
            "--seed",
            "42",
            "--event-limit",
            "80",
            "--output",
            str(out),
        ]
    )
    assert rc == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "world_canon" in payload
    world_canon = payload["world_canon"]
    assert isinstance(world_canon, dict)
    assert isinstance(world_canon.get("location_memory"), dict)
    assert isinstance(world_canon.get("claim_states"), dict)


def test_canon_persistence_across_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out_a = Path(tmp_path) / "sim_a.json"
    out_b = Path(tmp_path) / "sim_b.json"

    rc_a = run_cli.main(
        [
            "--scenario",
            "dinner_party",
            "--seed",
            "42",
            "--event-limit",
            "200",
            "--output",
            str(out_a),
        ]
    )
    assert rc_a == 0

    data_a = json.loads(out_a.read_text(encoding="utf-8"))
    canon_a = data_a["world_canon"]
    residue_locations = {
        location_id
        for location_id, mem in (canon_a.get("location_memory") or {}).items()
        if float((mem or {}).get("tension_residue", 0.0)) > 0.0
    }
    assert residue_locations

    rc_b = run_cli.main(
        [
            "--scenario",
            "dinner_party",
            "--seed",
            "51",
            "--event-limit",
            "1",
            "--canon",
            str(out_a),
            "--output",
            str(out_b),
        ]
    )
    assert rc_b == 0

    data_b = json.loads(out_b.read_text(encoding="utf-8"))

    initial_agents = (data_b.get("initial_state") or {}).get("agents") or {}
    claim_states_a = canon_a.get("claim_states") or {}
    for claim_id, states_by_agent in claim_states_a.items():
        for agent_id, belief_state in (states_by_agent or {}).items():
            beliefs = ((initial_agents.get(agent_id) or {}).get("beliefs") or {})
            if claim_id in beliefs:
                assert beliefs[claim_id] == belief_state

    canon_b = data_b.get("world_canon") or {}
    for location_id in residue_locations:
        mem_b = (canon_b.get("location_memory") or {}).get(location_id) or {}
        assert float(mem_b.get("tension_residue", 0.0)) > 0.0


def test_claim_states_snapshot() -> None:
    world, _, _ = _run_world(42)
    assert world.canon is not None

    expected_claim_ids = set(world.definition.all_claims)
    assert expected_claim_ids <= set(world.canon.claim_states)

    allowed_states = {"unknown", "suspects", "believes_true", "believes_false"}
    for claim_id in expected_claim_ids:
        state_map = world.canon.claim_states[claim_id]
        assert state_map
        assert set(state_map) == set(world.agents)
        assert set(state_map.values()) <= allowed_states
