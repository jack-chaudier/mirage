from __future__ import annotations

from functools import lru_cache
from random import Random

import narrativefield.extraction.rashomon as rashomon_module
from narrativefield.extraction.arc_search import ArcSearchDiagnostics, ArcSearchResult
from narrativefield.extraction.rashomon import RashomonArc, RashomonSet, extract_rashomon_set
from narrativefield.extraction.types import ArcScore, ArcValidation
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation

DINNER_PARTY_AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]


def _evt(
    event_id: str,
    *,
    source: str = "thorne",
    targets: list[str] | None = None,
    sim_time: float = 1.0,
    event_type: EventType = EventType.CHAT,
    tension: float = 0.5,
) -> Event:
    return Event(
        id=event_id,
        sim_time=sim_time,
        tick_id=max(1, int(sim_time * 10)),
        order_in_tick=0,
        type=event_type,
        source_agent=source,
        target_agents=targets if targets is not None else ["elena"],
        location_id="dining_table",
        causal_links=[],
        deltas=[],
        description=f"event {event_id}",
        metrics=EventMetrics(tension=tension, irony=0.1, significance=0.4),
    )


def _score(composite: float) -> ArcScore:
    return ArcScore(
        composite=composite,
        tension_variance=0.5,
        peak_tension=0.7,
        tension_shape=0.6,
        significance=0.5,
        thematic_coherence=0.5,
        irony_arc=0.5,
        protagonist_dominance=0.5,
    )


def _result(
    *,
    protagonist: str,
    events: list[Event],
    beats: list[BeatType],
    valid: bool,
    score: float | None = None,
    violations: tuple[str, ...] = (),
) -> ArcSearchResult:
    return ArcSearchResult(
        events=events,
        beats=beats,
        protagonist=protagonist,
        validation=ArcValidation(valid=valid, violations=violations),
        arc_score=_score(score) if score is not None else None,
        diagnostics=None,
    )


def _arc(
    *,
    protagonist: str,
    events: list[Event],
    beats: list[BeatType],
    valid: bool = True,
    score: float = 0.6,
    violations: tuple[str, ...] = (),
) -> RashomonArc:
    result = _result(
        protagonist=protagonist,
        events=events,
        beats=beats,
        valid=valid,
        score=score if valid else None,
        violations=violations,
    )
    return RashomonArc(
        protagonist=protagonist,
        search_result=result,
        arc_score=_score(score) if valid else None,
        events=events,
        beats=beats,
        valid=valid,
        violation_count=len(violations),
        violations=list(violations),
    )


def _lookup_overlap(matrix: dict[str, float], protagonist_a: str, protagonist_b: str) -> float:
    left, right = sorted((protagonist_a, protagonist_b))
    return matrix[f"{left}-{right}"]


def test_rashomon_set_returns_all_agents(monkeypatch) -> None:
    seed_event = _evt("e_seed")
    score_map = {
        "thorne": 0.71,
        "elena": 0.62,
        "marcus": 0.69,
        "lydia": 0.58,
        "diana": 0.55,
        "victor": 0.76,
    }

    def fake_search_arc(*, all_events, protagonist, max_events, total_sim_time):
        del all_events, max_events, total_sim_time
        return _result(
            protagonist=protagonist,
            events=[_evt(f"e_{protagonist}", source=protagonist)],
            beats=[BeatType.SETUP],
            valid=True,
            score=score_map[protagonist],
        )

    monkeypatch.setattr(rashomon_module, "search_arc", fake_search_arc)
    rashomon_set = extract_rashomon_set(
        events=[seed_event],
        seed=1,
        agents=list(DINNER_PARTY_AGENTS),
    )

    assert len(rashomon_set.arcs) == 6
    assert {arc.protagonist for arc in rashomon_set.arcs} == set(DINNER_PARTY_AGENTS)


def test_rashomon_set_sorted_by_validity_then_score(monkeypatch) -> None:
    validations = {
        "thorne": (True, 0.67, ()),
        "elena": (False, None, ("Missing TURNING_POINT beat", "Arc too short")),
        "marcus": (True, 0.73, ()),
        "lydia": (False, None, ("Missing CONSEQUENCE beat",)),
        "diana": (True, 0.61, ()),
        "victor": (True, 0.79, ()),
    }

    def fake_search_arc(*, all_events, protagonist, max_events, total_sim_time):
        del all_events, max_events, total_sim_time
        valid, score, violations = validations[protagonist]
        return _result(
            protagonist=protagonist,
            events=[_evt(f"e_{protagonist}", source=protagonist)],
            beats=[BeatType.SETUP],
            valid=valid,
            score=score,
            violations=violations,
        )

    monkeypatch.setattr(rashomon_module, "search_arc", fake_search_arc)
    rashomon_set = extract_rashomon_set(events=[_evt("e0")], seed=2, agents=list(DINNER_PARTY_AGENTS))

    ordered = [arc.protagonist for arc in rashomon_set.arcs]
    assert ordered[:4] == ["victor", "marcus", "thorne", "diana"]
    assert ordered[4:] == ["lydia", "elena"]
    assert all(arc.valid for arc in rashomon_set.arcs[:4])
    assert all(not arc.valid for arc in rashomon_set.arcs[4:])


def test_overlap_matrix_symmetric() -> None:
    e1 = _evt("e1")
    e2 = _evt("e2")
    e3 = _evt("e3")
    e4 = _evt("e4")
    rashomon_set = RashomonSet(
        seed=1,
        total_events=4,
        arcs=[
            _arc(
                protagonist="thorne",
                events=[e1, e2],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT],
            ),
            _arc(
                protagonist="elena",
                events=[e2, e3],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT],
            ),
            _arc(
                protagonist="victor",
                events=[e1, e4],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT],
            ),
        ],
    )
    matrix = rashomon_set.overlap_matrix

    assert len(matrix) == 3
    assert _lookup_overlap(matrix, "thorne", "elena") == _lookup_overlap(matrix, "elena", "thorne")
    assert _lookup_overlap(matrix, "thorne", "victor") == _lookup_overlap(matrix, "victor", "thorne")
    assert _lookup_overlap(matrix, "elena", "victor") == _lookup_overlap(matrix, "victor", "elena")


def test_overlap_matrix_range() -> None:
    rashomon_set = RashomonSet(
        seed=3,
        total_events=5,
        arcs=[
            _arc(
                protagonist="thorne",
                events=[_evt("e1"), _evt("e2")],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT],
            ),
            _arc(
                protagonist="elena",
                events=[_evt("e2"), _evt("e3"), _evt("e4")],
                beats=[BeatType.SETUP, BeatType.ESCALATION, BeatType.TURNING_POINT],
            ),
            _arc(
                protagonist="victor",
                events=[_evt("e5")],
                beats=[BeatType.TURNING_POINT],
            ),
        ],
    )

    for value in rashomon_set.overlap_matrix.values():
        assert 0.0 <= value <= 1.0


def test_turning_point_overlap_captures_shared_events() -> None:
    shared_tp = _evt("e_tp", event_type=EventType.CONFLICT)
    rashomon_set = RashomonSet(
        seed=7,
        total_events=4,
        arcs=[
            _arc(
                protagonist="thorne",
                events=[_evt("e1"), shared_tp, _evt("e2")],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT, BeatType.CONSEQUENCE],
            ),
            _arc(
                protagonist="elena",
                events=[_evt("e3"), shared_tp, _evt("e4")],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT, BeatType.CONSEQUENCE],
            ),
            _arc(
                protagonist="victor",
                events=[_evt("e5")],
                beats=[BeatType.SETUP],
                valid=False,
                score=0.0,
                violations=("Missing TURNING_POINT beat",),
            ),
        ],
    )

    overlap = rashomon_set.turning_point_overlap()
    assert overlap["e_tp"] == ["elena", "thorne"]


def test_rashomon_set_roundtrip() -> None:
    diagnostics = ArcSearchDiagnostics(
        violations=["Missing CONSEQUENCE beat"],
        suggested_protagonist="thorne",
        suggested_time_window=(1.0, 9.0),
        suggested_keep_ids=["e1", "e2"],
        suggested_drop_ids=["e3"],
        primary_failure="Missing CONSEQUENCE beat",
        rule_failure_counts={"missing_consequence": 2},
        best_candidate_violation_count=1,
        candidates_evaluated=4,
        best_candidate_violations=["Missing CONSEQUENCE beat"],
    )
    invalid_result = ArcSearchResult(
        events=[_evt("e1")],
        beats=[BeatType.SETUP],
        protagonist="thorne",
        validation=ArcValidation(valid=False, violations=("Missing CONSEQUENCE beat",)),
        arc_score=None,
        diagnostics=diagnostics,
    )
    rashomon_set = RashomonSet(
        seed=11,
        total_events=2,
        arcs=[
            _arc(
                protagonist="victor",
                events=[_evt("e2"), _evt("e3")],
                beats=[BeatType.SETUP, BeatType.TURNING_POINT],
            ),
            RashomonArc(
                protagonist="thorne",
                search_result=invalid_result,
                arc_score=None,
                events=invalid_result.events,
                beats=invalid_result.beats,
                valid=False,
                violation_count=1,
                violations=["Missing CONSEQUENCE beat"],
            ),
        ],
    )

    payload = rashomon_set.to_dict()
    reconstructed = RashomonSet.from_dict(payload)
    assert reconstructed.to_dict() == payload


@lru_cache(maxsize=8)
def _seed_rashomon(seed: int) -> RashomonSet:
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
    return extract_rashomon_set(
        events=metrics_output.events,
        seed=seed,
        agents=list(DINNER_PARTY_AGENTS),
        total_sim_time=total_sim_time,
    )


def test_rashomon_seed_42_regression() -> None:
    rashomon_set = _seed_rashomon(42)
    valid_protagonists = [arc.protagonist for arc in rashomon_set.arcs if arc.valid]

    assert len(rashomon_set.arcs) == 6
    assert rashomon_set.valid_count == 6
    assert valid_protagonists == ["marcus", "elena", "lydia", "thorne", "victor", "diana"]
