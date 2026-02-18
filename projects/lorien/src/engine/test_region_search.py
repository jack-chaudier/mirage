"""
Regression test for region-to-arc search.

Proves that search_region_for_arcs returns valid arcs for typical region
selections, where the old "validate entire region" approach returned 0%.
"""
from __future__ import annotations

from pathlib import Path
from random import Random

from narrativefield.extraction.arc_search import search_region_for_arcs
from narrativefield.schema.events import Event


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_full_pipeline(seed: int = 42) -> tuple[list[Event], float]:
    """Run sim + metrics and return (events, total_sim_time)."""
    from narrativefield.integration.bundler import BundleInputs, bundle_for_renderer
    from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
    from narrativefield.simulation.scenarios import create_dinner_party_world
    from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation

    world = create_dinner_party_world()
    rng = Random(seed)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    raw_events, snapshots = run_simulation(world, rng, cfg)

    sim_result = {
        "format_version": "1.0.0",
        "metadata": {
            "scenario": "dinner_party",
            "total_ticks": world.tick_id,
            "total_sim_time": world.sim_time,
            "agent_count": len(world.agents),
            "event_count": len(raw_events),
            "seed": seed,
        },
        "initial_state": snapshots[0] if snapshots else {},
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in raw_events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }

    parsed = parse_simulation_output(sim_result)
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
        )
    )

    events = [Event.from_dict(e) for e in payload["events"]]
    events.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
    total_sim_time = float(payload.get("metadata", {}).get("total_sim_time", 120.0))
    return events, total_sim_time


def test_region_search_finds_valid_arcs() -> None:
    """
    Core regression: pick a mid-dinner region and assert that
    search_region_for_arcs returns at least 1 valid arc.
    """
    events, total_sim_time = _run_full_pipeline(seed=42)
    assert len(events) > 0, "Pipeline produced no events"

    # Pick a region covering the middle of the dinner party.
    all_times = [e.sim_time for e in events]
    t_min, t_max = min(all_times), max(all_times)
    time_start = t_min + (t_max - t_min) * 0.1
    time_end = t_max - (t_max - t_min) * 0.1

    region_events = [e for e in events if time_start <= e.sim_time <= time_end]
    print(f"\nRegion [{time_start:.1f}, {time_end:.1f}]: {len(region_events)} events")

    candidates = search_region_for_arcs(
        events=events,
        time_start=time_start,
        time_end=time_end,
        total_sim_time=total_sim_time,
    )

    # 1. At least 1 valid arc candidate returned.
    assert len(candidates) >= 1, (
        f"Expected at least 1 valid arc candidate, got {len(candidates)}. "
        f"Region had {len(region_events)} events."
    )

    # 2. The best candidate has a positive composite score.
    best = candidates[0]
    assert best.score.composite > 0.0, (
        f"Best candidate score {best.score.composite} should be > 0.0"
    )

    # 3. All returned candidates have is_valid == True.
    for i, c in enumerate(candidates):
        assert c.validation.valid, (
            f"Candidate {i} ({c.protagonist}) should be valid but has "
            f"violations: {c.validation.violations}"
        )

    # 4. All returned event_ids are within the requested time region.
    for i, c in enumerate(candidates):
        for e in c.events:
            assert time_start <= e.sim_time <= time_end, (
                f"Candidate {i} event {e.id} at t={e.sim_time} outside "
                f"region [{time_start}, {time_end}]"
            )

    # Print results for manual inspection.
    print(f"\n  Found {len(candidates)} valid arc candidate(s):")
    for i, c in enumerate(candidates):
        beat_summary = ", ".join(b.value for b in c.beats)
        print(f"  [{i+1}] protagonist={c.protagonist}, "
              f"events={len(c.events)}, "
              f"score={c.score.composite:.4f}")
        print(f"      beats: {beat_summary}")
        print(f"      explanation: {c.explanation}")


def test_region_search_with_agent_filter() -> None:
    """Filtering by agent should return arcs for that protagonist."""
    events, total_sim_time = _run_full_pipeline(seed=42)

    all_times = [e.sim_time for e in events]
    t_min, t_max = min(all_times), max(all_times)

    # Pick the most common agent.
    from collections import Counter
    agent_counts = Counter(e.source_agent for e in events)
    top_agent = agent_counts.most_common(1)[0][0]

    candidates = search_region_for_arcs(
        events=events,
        time_start=t_min,
        time_end=t_max,
        agent_filter=top_agent,
        total_sim_time=total_sim_time,
    )

    if candidates:
        assert all(c.protagonist == top_agent for c in candidates), (
            f"All candidates should have protagonist={top_agent}"
        )
        print(f"\n  Agent filter '{top_agent}': {len(candidates)} candidate(s), "
              f"best score={candidates[0].score.composite:.4f}")


def test_region_search_empty_region() -> None:
    """A region with no events should return empty list."""
    events, total_sim_time = _run_full_pipeline(seed=42)

    candidates = search_region_for_arcs(
        events=events,
        time_start=9999.0,
        time_end=9999.1,
        total_sim_time=total_sim_time,
    )
    assert candidates == [], f"Expected empty list for empty region, got {len(candidates)}"
