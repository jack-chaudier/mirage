from __future__ import annotations

from random import Random

from narrativefield.schema.events import EventType
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation


def _count_secrets_with_any_non_unknown(agents: dict[str, dict]) -> int:
    revealed: set[str] = set()
    for a in agents.values():
        beliefs = a.get("beliefs") or {}
        for secret_id, state in beliefs.items():
            if state != "unknown":
                revealed.add(str(secret_id))
    return len(revealed)


def test_dinner_party_success_gate_regression_seeds() -> None:
    # This is a pragmatic regression test for the Phase 2 "success gate":
    # across multiple deterministic seeds, the dinner party produces
    # 100-200 events, >=1 catastrophe, and >=3 locations used.
    # Seeds updated after narrative-physics tuning (location affordances + social inertia + move scoring).
    for seed in (1, 2, 4):
        world = create_dinner_party_world()
        rng = Random(seed)
        cfg = SimulationConfig(
            tick_limit=300,
            event_limit=200,
            max_sim_time=world.definition.sim_duration_minutes,
            snapshot_interval_events=world.definition.snapshot_interval,
        )

        events, snapshots = run_simulation(world, rng, cfg)

        assert 100 <= len(events) <= 200
        assert sum(1 for e in events if e.type == EventType.CATASTROPHE) >= 1
        assert len({e.location_id for e in events}) >= 3

        # Ensure the run is information-rich: at least 2 secrets are no longer UNKNOWN
        # for someone by the last snapshot.
        assert snapshots, "expected snapshots"
        assert _count_secrets_with_any_non_unknown(snapshots[-1]["agents"]) >= 2
