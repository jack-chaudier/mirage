from __future__ import annotations

from random import Random

from narrativefield.extraction.arc_search import _enforce_monotonic_beats
from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import search_arc
from narrativefield.extraction.arc_validator import validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.extraction.beat_sheet import build_beat_sheet
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.extraction.prose_generator import build_llm_prompt
from narrativefield.schema.events import (
    BeatType,
    DeltaKind,
    DeltaOp,
    Event,
    EventMetrics,
    EventType,
    StateDelta,
)
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation


def _evt(i: int, *, t: float, et: EventType, tension: float, src: str = "a") -> Event:
    return Event(
        id=f"e{i:03d}",
        sim_time=t,
        tick_id=i,
        order_in_tick=0,
        type=et,
        source_agent=src,
        target_agents=["b"] if et in {EventType.CHAT, EventType.CONFLICT, EventType.REVEAL} else [],
        location_id="dining_table",
        causal_links=[f"e{i-1:03d}"] if i > 1 else [],
        deltas=[],
        description=f"{et.value} {i}",
        metrics=EventMetrics(
            tension=tension,
            irony=0.5,
            significance=0.0,
            thematic_shift={},
            tension_components={
                "danger": 0.0,
                "time_pressure": 0.0,
                "goal_frustration": 0.0,
                "relationship_volatility": 0.0,
                "information_gap": 0.0,
                "resource_scarcity": 0.0,
                "moral_cost": 0.0,
                "irony_density": 0.0,
            },
            irony_collapse=None,
        ),
    )


def test_story_extraction_happy_path() -> None:
    events = [
        _evt(1, t=0.0, et=EventType.CHAT, tension=0.10),
        _evt(2, t=3.0, et=EventType.OBSERVE, tension=0.15),
        _evt(3, t=6.0, et=EventType.CONFLICT, tension=0.60),
        _evt(4, t=9.0, et=EventType.REVEAL, tension=0.75),
        _evt(5, t=12.0, et=EventType.CONFLICT, tension=0.95),
        _evt(6, t=15.0, et=EventType.CHAT, tension=0.40),
        _evt(7, t=18.0, et=EventType.CHAT, tension=0.25),
    ]
    # Ensure protagonist dominance.
    for e in events:
        e.source_agent = "a"

    beats = classify_beats(events)
    assert len(beats) == len(events)
    assert BeatType.TURNING_POINT in beats

    validation = validate_arc(events=events, beats=beats, total_sim_time=120.0)
    assert validation.valid, validation.violations

    score = score_arc(events, beats)
    assert 0.0 <= score.composite <= 1.0

    beat_sheet = build_beat_sheet(
        events=events,
        beats=beats,
        protagonist="a",
        genre_preset="default",
        arc_score=score,
        agents_manifest={"a": {"id": "a", "name": "A", "goal_summary": "x", "primary_flaw": "y"}},
        secrets={},
        scenes=[],
    )
    prompt = build_llm_prompt(beat_sheet)
    assert "BEAT SEQUENCE" in prompt


def test_arc_search_basic() -> None:
    """Create ~30 synthetic events; search_arc should select <= max_events with alice as protagonist."""
    events: list[Event] = []
    # Build a 30-event timeline spanning 0-30 sim-minutes.
    # Alice dominates as source agent.
    types_tension: list[tuple[EventType, float]] = [
        # Setup phase (0-8 min).
        (EventType.CHAT, 0.10),
        (EventType.OBSERVE, 0.12),
        (EventType.CHAT, 0.15),
        (EventType.SOCIAL_MOVE, 0.14),
        (EventType.CHAT, 0.18),
        (EventType.OBSERVE, 0.16),
        (EventType.CONFIDE, 0.25),
        (EventType.CHAT, 0.20),
        # Rising action (8-18 min).
        (EventType.LIE, 0.35),
        (EventType.CONFLICT, 0.45),
        (EventType.CHAT, 0.40),
        (EventType.REVEAL, 0.55),
        (EventType.CONFLICT, 0.60),
        (EventType.CONFIDE, 0.50),
        (EventType.LIE, 0.65),
        (EventType.CONFLICT, 0.70),
        (EventType.OBSERVE, 0.55),
        (EventType.CHAT, 0.60),
        # Climax (18-22 min).
        (EventType.CONFLICT, 0.80),
        (EventType.REVEAL, 0.85),
        (EventType.CATASTROPHE, 0.95),
        (EventType.CONFLICT, 0.90),
        # Falling action (22-30 min).
        (EventType.CHAT, 0.50),
        (EventType.OBSERVE, 0.40),
        (EventType.CHAT, 0.35),
        (EventType.SOCIAL_MOVE, 0.30),
        (EventType.CHAT, 0.25),
        (EventType.INTERNAL, 0.20),
        (EventType.CHAT, 0.15),
        (EventType.OBSERVE, 0.10),
    ]

    for i, (et, tension) in enumerate(types_tension):
        t = float(i)  # 1 sim-min per event, 0-29.
        # Alice is source for 80% of events, bob for the rest.
        src = "alice" if i % 5 != 4 else "bob"
        targets: list[str] = []
        if et in {EventType.CHAT, EventType.CONFLICT, EventType.REVEAL, EventType.CONFIDE}:
            targets = ["bob"] if src == "alice" else ["alice"]
        deltas: list[StateDelta] = []
        if et in {EventType.CONFIDE, EventType.REVEAL}:
            deltas = [
                StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent=src,
                    attribute="trust",
                    op=DeltaOp.ADD,
                    value=0.1,
                ),
            ]
        events.append(
            Event(
                id=f"evt-{i:03d}",
                sim_time=t,
                tick_id=i,
                order_in_tick=0,
                type=et,
                source_agent=src,
                target_agents=targets,
                location_id="dining_table",
                causal_links=[f"evt-{i - 1:03d}"] if i > 0 else [],
                deltas=deltas,
                description=f"{et.value} at t={t}",
                metrics=EventMetrics(
                    tension=tension,
                    irony=0.3,
                    significance=tension * 0.5,
                ),
            )
        )

    result = search_arc(events, time_start=0.0, time_end=30.0, max_events=15, total_sim_time=120.0)

    assert len(result.events) <= 15
    assert len(result.events) == len(result.beats)
    assert result.protagonist == "alice"

    # The selected events should be time-ordered.
    times = [e.sim_time for e in result.events]
    assert times == sorted(times)

    # Check no causal gaps (ArcSearch should prefer continuity).
    causal_gaps = [v for v in result.validation.violations if "Causal gap" in v]
    assert len(causal_gaps) == 0, f"Unexpected causal gaps: {causal_gaps}"


def test_arc_search_diagnostics_on_failure() -> None:
    """With no clear protagonist, diagnostics should suggest one."""
    # Create 8 events equally distributed among 4 agents.
    agents = ["alice", "bob", "carol", "dave"]
    events: list[Event] = []
    for i in range(8):
        src = agents[i % 4]
        events.append(
            Event(
                id=f"eq-{i:03d}",
                sim_time=float(i),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT,
                source_agent=src,
                target_agents=[agents[(i + 1) % 4]],
                location_id="dining_table",
                causal_links=[f"eq-{i - 1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"chat {i}",
                metrics=EventMetrics(tension=0.1, irony=0.1, significance=0.1),
            )
        )

    result = search_arc(events, max_events=6, total_sim_time=120.0)

    # Should still pick a protagonist (the one with highest score).
    assert result.protagonist != ""
    # Diagnostics should be present since the arc is likely too short / no turning point.
    assert result.diagnostics is not None
    assert result.diagnostics.suggested_protagonist != ""


def test_enforce_monotonic_beats_resolves_duplicate_turning_points_deterministically() -> None:
    events = [
        _evt(1, t=0.0, et=EventType.CHAT, tension=0.10),
        _evt(2, t=2.0, et=EventType.CONFLICT, tension=0.60),
        _evt(3, t=4.0, et=EventType.REVEAL, tension=0.85),
        _evt(4, t=6.0, et=EventType.CHAT, tension=0.30),
    ]
    beats = [
        BeatType.SETUP,
        BeatType.TURNING_POINT,
        BeatType.TURNING_POINT,
        BeatType.CONSEQUENCE,
    ]

    repaired_a = _enforce_monotonic_beats(events, beats)
    repaired_b = _enforce_monotonic_beats(events, beats)

    assert repaired_a == repaired_b
    assert repaired_a.count(BeatType.TURNING_POINT) == 1
    # Keep highest-tension TP at index 2; downgrade earlier duplicate.
    assert repaired_a[2] == BeatType.TURNING_POINT
    assert repaired_a[1] == BeatType.ESCALATION


def test_arc_search_includes_post_peak_consequence_for_seed51() -> None:
    world = create_dinner_party_world()
    rng = Random(51)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    sim_events, snapshots = run_simulation(world, rng, cfg)
    sim_output = {
        "format_version": "1.0.0",
        "metadata": {
            "simulation_id": "sim_seed51",
            "scenario": "dinner_party",
            "total_ticks": world.tick_id,
            "total_sim_time": world.sim_time,
            "agent_count": len(world.agents),
            "event_count": len(sim_events),
            "snapshot_interval": world.definition.snapshot_interval,
            "timestamp": "2026-02-11T00:00:00Z",
        },
        "initial_state": snapshots[0],
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in sim_events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }
    parsed = parse_simulation_output(sim_output)
    metrics_out = run_metrics_pipeline(parsed)

    result = search_arc(
        all_events=metrics_out.events,
        time_start=0.0,
        time_end=world.definition.sim_duration_minutes,
        max_events=20,
        total_sim_time=world.definition.sim_duration_minutes,
    )
    assert result.events
    assert result.beats

    peak_idx = max(range(len(result.events)), key=lambda idx: result.events[idx].metrics.tension)
    assert any(b == BeatType.CONSEQUENCE for b in result.beats[peak_idx + 1 :])
