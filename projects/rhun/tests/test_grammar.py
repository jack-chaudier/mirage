from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.validator import validate
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


def _graph() -> CausalGraph:
    events = tuple(
        Event(id=f"e{i}", timestamp=float(i), weight=0.1 * (i + 1), actors=frozenset({"A"}))
        for i in range(6)
    )
    return CausalGraph(events=events, actors=frozenset({"A"}), seed=1)


def test_validator_catches_each_violation_type() -> None:
    graph = _graph()
    grammar = GrammarConfig.strict()

    # too_short + insufficient_development + insufficient_timespan
    short_seq = ExtractedSequence(
        events=graph.events[:1],
        phases=(Phase.TURNING_POINT,),
        focal_actor="A",
    )
    _, short_violations = validate(short_seq, grammar, graph)
    assert any(v.startswith("too_short") for v in short_violations)
    assert any(v.startswith("insufficient_development") for v in short_violations)
    assert any(v.startswith("insufficient_timespan") for v in short_violations)

    # phase_regression
    regression_seq = ExtractedSequence(
        events=graph.events[:4],
        phases=(Phase.SETUP, Phase.TURNING_POINT, Phase.DEVELOPMENT, Phase.RESOLUTION),
        focal_actor="A",
    )
    _, regression_violations = validate(regression_seq, grammar, graph)
    assert any(v.startswith("phase_regression") for v in regression_violations)

    # too_many_turning_points
    too_many_tp = ExtractedSequence(
        events=graph.events[:4],
        phases=(Phase.SETUP, Phase.TURNING_POINT, Phase.TURNING_POINT, Phase.RESOLUTION),
        focal_actor="A",
    )
    _, tp_violations = validate(too_many_tp, grammar, graph)
    assert any(v.startswith("too_many_turning_points") for v in tp_violations)

    # too_long
    long_events = tuple(
        Event(id=f"x{i}", timestamp=float(i), weight=0.2, actors=frozenset({"A"})) for i in range(25)
    )
    long_graph = CausalGraph(events=long_events, actors=frozenset({"A"}), seed=2)
    long_seq = ExtractedSequence(
        events=long_graph.events,
        phases=tuple([Phase.SETUP] * 5 + [Phase.DEVELOPMENT] * 10 + [Phase.TURNING_POINT] + [Phase.RESOLUTION] * 9),
        focal_actor="A",
    )
    _, long_violations = validate(long_seq, grammar, long_graph)
    assert any(v.startswith("too_long") for v in long_violations)

    # insufficient_coverage
    mixed_events = (
        Event(id="a", timestamp=0.0, weight=0.1, actors=frozenset({"A"})),
        Event(id="b", timestamp=1.0, weight=0.2, actors=frozenset({"B"})),
        Event(id="c", timestamp=2.0, weight=0.3, actors=frozenset({"B"})),
        Event(id="d", timestamp=3.0, weight=0.4, actors=frozenset({"B"})),
    )
    mixed_graph = CausalGraph(events=mixed_events, actors=frozenset({"A", "B"}), seed=3)
    low_cov = ExtractedSequence(
        events=mixed_graph.events,
        phases=(Phase.SETUP, Phase.DEVELOPMENT, Phase.TURNING_POINT, Phase.RESOLUTION),
        focal_actor="A",
    )
    _, coverage_violations = validate(low_cov, grammar, mixed_graph)
    assert any(v.startswith("insufficient_coverage") for v in coverage_violations)


def test_strict_grammar_rejects_no_development() -> None:
    graph = _graph()
    seq = ExtractedSequence(
        events=graph.events[:4],
        phases=(Phase.SETUP, Phase.TURNING_POINT, Phase.RESOLUTION, Phase.RESOLUTION),
        focal_actor="A",
    )
    valid, violations = validate(seq, GrammarConfig.strict(), graph)

    assert valid is False
    assert any(v.startswith("insufficient_development") for v in violations)


def test_vacuous_grammar_accepts_everything() -> None:
    graph = _graph()
    seq = ExtractedSequence(
        events=graph.events,
        phases=(
            Phase.RESOLUTION,
            Phase.SETUP,
            Phase.TURNING_POINT,
            Phase.DEVELOPMENT,
            Phase.TURNING_POINT,
            Phase.SETUP,
        ),
        focal_actor="A",
    )
    valid, violations = validate(seq, GrammarConfig.vacuous(), graph)

    assert valid is True
    assert violations == []


def test_parametric_grammar_respects_k() -> None:
    graph = _graph()
    seq = ExtractedSequence(
        events=graph.events[:4],
        phases=(Phase.SETUP, Phase.DEVELOPMENT, Phase.TURNING_POINT, Phase.RESOLUTION),
        focal_actor="A",
    )

    valid_k0, _ = validate(
        seq,
        GrammarConfig.parametric(
            0,
            min_timespan_fraction=0.0,
            focal_actor_coverage=0.0,
        ),
        graph,
    )
    valid_k1, _ = validate(
        seq,
        GrammarConfig.parametric(
            1,
            min_timespan_fraction=0.0,
            focal_actor_coverage=0.0,
        ),
        graph,
    )
    valid_k2, violations_k2 = validate(
        seq,
        GrammarConfig.parametric(
            2,
            min_timespan_fraction=0.0,
            focal_actor_coverage=0.0,
        ),
        graph,
    )

    assert valid_k0 is True
    assert valid_k1 is True
    assert valid_k2 is False
    assert any(v.startswith("insufficient_development") for v in violations_k2)
