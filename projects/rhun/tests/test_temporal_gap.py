from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


def _graph_with_large_jump() -> CausalGraph:
    events = (
        Event(id="e0", timestamp=0.0, weight=0.2, actors=frozenset({"A"})),
        Event(id="e1", timestamp=1.0, weight=0.3, actors=frozenset({"A"}), causal_parents=("e0",)),
        Event(id="e2", timestamp=2.0, weight=0.4, actors=frozenset({"A"}), causal_parents=("e1",)),
        Event(id="e3", timestamp=12.0, weight=0.9, actors=frozenset({"A"}), causal_parents=("e2",)),
    )
    return CausalGraph(events=events, actors=frozenset({"A"}), seed=11)


def _sequence(graph: CausalGraph) -> ExtractedSequence:
    return ExtractedSequence(
        events=graph.events,
        phases=(Phase.SETUP, Phase.DEVELOPMENT, Phase.DEVELOPMENT, Phase.TURNING_POINT),
        focal_actor="A",
    )


def test_max_temporal_gap_violation_detected() -> None:
    graph = _graph_with_large_jump()
    seq = _sequence(graph)
    grammar = GrammarConfig(
        min_prefix_elements=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        max_temporal_gap=3.0,
        focal_actor_coverage=0.0,
    )

    valid, violations = validate(seq, grammar, graph)

    assert valid is False
    assert any(v.startswith("max_temporal_gap:") for v in violations)


def test_max_temporal_gap_noop_when_infinite() -> None:
    graph = _graph_with_large_jump()
    seq = _sequence(graph)
    grammar = GrammarConfig(
        min_prefix_elements=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
        focal_actor_coverage=0.0,
    )

    valid, violations = validate(seq, grammar, graph)

    assert valid is True
    assert not any(v.startswith("max_temporal_gap:") for v in violations)


def test_gap_violation_reports_event_ids() -> None:
    graph = _graph_with_large_jump()
    seq = _sequence(graph)
    grammar = GrammarConfig(
        min_prefix_elements=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        max_temporal_gap=2.0,
        focal_actor_coverage=0.0,
    )

    _, violations = validate(seq, grammar, graph)

    gap_violations = [v for v in violations if v.startswith("max_temporal_gap:")]
    assert gap_violations
    assert "e2" in gap_violations[0]
    assert "e3" in gap_violations[0]
