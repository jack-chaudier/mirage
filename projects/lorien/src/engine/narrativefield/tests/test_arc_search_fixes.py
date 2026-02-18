from __future__ import annotations

from functools import lru_cache
from random import Random

from narrativefield.extraction.arc_search import (
    _downsample_preserving_continuity,
    _enforce_monotonic_beats,
    fallback_event_sort_key,
    search_arc,
)
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation


def _evt(
    idx: int,
    *,
    event_type: EventType = EventType.CHAT,
    source: str = "alice",
    targets: list[str] | None = None,
    tension: float = 0.2,
    significance: float = 0.1,
) -> Event:
    return Event(
        id=f"e{idx}",
        sim_time=float(idx),
        tick_id=idx,
        order_in_tick=0,
        type=event_type,
        source_agent=source,
        target_agents=targets if targets is not None else ["bob"],
        location_id="dining_table",
        causal_links=[f"e{idx - 1}"] if idx > 0 else [],
        deltas=[],
        description=f"event {idx}",
        metrics=EventMetrics(tension=tension, irony=0.1, significance=significance),
    )


@lru_cache(maxsize=64)
def _seed_arc_result(seed: int):
    world = create_dinner_party_world()
    world.canon = init_canon_from_world(world.definition, None)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, Random(seed), cfg)
    payload = {
        "format_version": "1.0.0",
        "metadata": {
            "scenario": "dinner_party",
            "seed": int(seed),
            "total_ticks": int(world.tick_id),
            "total_sim_time": float(world.sim_time),
            "agent_count": int(len(world.agents)),
            "event_count": int(len(events)),
            "snapshot_interval": int(world.definition.snapshot_interval),
            "truncated": bool(world.truncated),
        },
        "initial_state": snapshots[0] if snapshots else {},
        "snapshots": snapshots[1:] if len(snapshots) > 1 else [],
        "events": [event.to_dict() for event in events],
        "secrets": [secret.to_dict() for secret in world.definition.secrets.values()],
        "claims": [claim.to_dict() for claim in world.definition.claims.values()],
        "locations": [location.to_dict() for location in world.definition.locations.values()],
        "world_canon": world.canon.to_dict() if world.canon is not None else WorldCanon().to_dict(),
    }

    parsed = parse_simulation_output(payload)
    metrics_output = run_metrics_pipeline(parsed)
    total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
    return search_arc(all_events=metrics_output.events, total_sim_time=total_sim_time)


def test_monotonic_beats_uses_tension_for_phase1() -> None:
    events = [
        _evt(0, tension=0.2),
        _evt(1, tension=0.8),
        _evt(2, tension=0.9),
    ]
    beats = [BeatType.COMPLICATION, BeatType.SETUP, BeatType.TURNING_POINT]

    repaired = _enforce_monotonic_beats(events, beats)
    assert repaired[1] == BeatType.ESCALATION


def test_monotonic_beats_falling_tension_gets_complication() -> None:
    events = [
        _evt(0, tension=0.8),
        _evt(1, tension=0.2),
        _evt(2, tension=0.9),
    ]
    beats = [BeatType.COMPLICATION, BeatType.SETUP, BeatType.TURNING_POINT]

    repaired = _enforce_monotonic_beats(events, beats)
    assert repaired[1] == BeatType.COMPLICATION


def test_downsample_includes_post_peak_event() -> None:
    pool: list[Event] = []
    for idx in range(25):
        if idx <= 10:
            source = "alice"
            significance = 1.0 - (idx * 0.01)
        elif idx in {11, 12}:
            source = "alice"
            significance = 0.02
        else:
            source = "carol"
            significance = 0.01

        if idx < 10:
            tension = 0.1 + (idx * 0.05)
        elif idx == 10:
            tension = 0.95
        else:
            tension = max(0.1, 0.8 - ((idx - 10) * 0.04))

        pool.append(
            _evt(
                idx,
                source=source,
                targets=["bob"] if source == "alice" else ["dave"],
                tension=tension,
                significance=significance,
            )
        )

    selected = _downsample_preserving_continuity(
        pool=pool,
        bridge_pool=pool,
        protagonist="alice",
        anchor_id="e10",
        max_events=6,
    )
    selected_ids = {event.id for event in selected}

    assert len(selected) <= 6
    assert "e24" in selected_ids  # endpoint keep
    assert any(event_id in selected_ids for event_id in {"e11", "e12"})


def test_downsample_no_swap_when_post_peak_naturally_selected() -> None:
    pool: list[Event] = []
    for idx in range(25):
        if idx <= 10:
            source = "alice"
            significance = 1.0 - (idx * 0.01)
        elif idx == 11:
            source = "alice"
            significance = 2.0
        elif idx == 12:
            source = "alice"
            significance = 0.01
        else:
            source = "carol"
            significance = 0.01

        tension = 0.95 if idx == 10 else (0.1 + (0.02 * idx))
        pool.append(
            _evt(
                idx,
                source=source,
                targets=["bob"] if source == "alice" else ["dave"],
                tension=tension,
                significance=significance,
            )
        )

    selected = _downsample_preserving_continuity(
        pool=pool,
        bridge_pool=pool,
        protagonist="alice",
        anchor_id="e10",
        max_events=6,
    )
    selected_ids = {event.id for event in selected}

    assert "e11" in selected_ids
    assert "e12" not in selected_ids


def test_significance_tiebreaker_prefers_protagonist() -> None:
    non_protagonist = _evt(
        1,
        event_type=EventType.CONFLICT,
        source="carol",
        targets=["dave"],
        significance=0.582,
    )
    protagonist_event = _evt(
        2,
        event_type=EventType.CHAT,
        source="alice",
        targets=["bob"],
        significance=0.582,
    )

    ranked = sorted(
        [non_protagonist, protagonist_event],
        key=lambda event: fallback_event_sort_key(event, "alice"),
    )
    assert ranked[0].id == protagonist_event.id


def test_significance_tiebreaker_prefers_dramatic_types() -> None:
    catastrophe = _evt(
        1,
        event_type=EventType.CATASTROPHE,
        source="alice",
        targets=["bob"],
        significance=0.582,
    )
    chat = _evt(
        2,
        event_type=EventType.CHAT,
        source="alice",
        targets=["bob"],
        significance=0.582,
    )

    ranked = sorted([chat, catastrophe], key=lambda event: fallback_event_sort_key(event, "alice"))
    assert ranked[0].id == catastrophe.id


def test_tp_dedup_post_tp_becomes_consequence() -> None:
    events = [_evt(i, tension=0.1 + (i * 0.01)) for i in range(20)]
    events[5].metrics.tension = 0.80
    events[10].metrics.tension = 0.95
    events[15].metrics.tension = 0.70

    beats = [BeatType.COMPLICATION for _ in range(20)]
    beats[0] = BeatType.SETUP
    beats[5] = BeatType.TURNING_POINT
    beats[10] = BeatType.TURNING_POINT
    beats[15] = BeatType.TURNING_POINT

    repaired = _enforce_monotonic_beats(events, beats)

    assert repaired[5] == BeatType.ESCALATION
    assert repaired[10] == BeatType.TURNING_POINT
    assert repaired[15] == BeatType.CONSEQUENCE


def test_tp_dedup_pre_tp_becomes_escalation() -> None:
    events = [_evt(i, tension=0.2 + (i * 0.01)) for i in range(16)]
    events[3].metrics.tension = 0.70
    events[12].metrics.tension = 0.98

    beats = [BeatType.COMPLICATION for _ in range(16)]
    beats[0] = BeatType.SETUP
    beats[3] = BeatType.TURNING_POINT
    beats[12] = BeatType.TURNING_POINT

    repaired = _enforce_monotonic_beats(events, beats)

    assert repaired[3] == BeatType.ESCALATION
    assert repaired[12] == BeatType.TURNING_POINT


def test_tp_dedup_post_sweep_enforces_consequence() -> None:
    events = [_evt(i, tension=0.15 + (i * 0.01)) for i in range(18)]
    events[5].metrics.tension = 0.82
    events[10].metrics.tension = 0.96

    beats = [BeatType.COMPLICATION for _ in range(18)]
    beats[0] = BeatType.SETUP
    beats[5] = BeatType.TURNING_POINT
    beats[10] = BeatType.TURNING_POINT
    beats[16] = BeatType.COMPLICATION

    repaired = _enforce_monotonic_beats(events, beats)

    assert repaired[10] == BeatType.TURNING_POINT
    assert repaired[16] == BeatType.CONSEQUENCE


def test_tp_dedup_single_tp_unchanged() -> None:
    events = [_evt(i, tension=0.1 + (i * 0.1)) for i in range(5)]
    beats = [
        BeatType.SETUP,
        BeatType.COMPLICATION,
        BeatType.ESCALATION,
        BeatType.TURNING_POINT,
        BeatType.CONSEQUENCE,
    ]

    repaired = _enforce_monotonic_beats(events, beats)

    assert repaired == beats
    assert repaired.count(BeatType.TURNING_POINT) == 1
    assert repaired[3] == BeatType.TURNING_POINT


def test_seed_sweep_1_20_all_valid_or_near() -> None:
    valid_runs = sum(1 for seed in range(1, 21) if _seed_arc_result(seed).validation.valid)
    assert valid_runs >= 18


def test_previously_valid_arcs_unchanged() -> None:
    baseline = {
        1: ("thorne", 0.682371),
        3: ("marcus", 0.724853),
        4: ("marcus", 0.749089),
        5: ("victor", 0.696479),
        6: ("lydia", 0.713345),
        8: ("marcus", 0.641371),
        11: ("victor", 0.643118),
        15: ("marcus", 0.725571),
        17: ("marcus", 0.707010),
        20: ("victor", 0.641865),
    }

    for seed, (expected_protagonist, expected_score) in baseline.items():
        result = _seed_arc_result(seed)
        assert result.validation.valid
        assert result.protagonist == expected_protagonist
        assert result.arc_score is not None
        assert abs(float(result.arc_score.composite) - expected_score) <= 0.01
