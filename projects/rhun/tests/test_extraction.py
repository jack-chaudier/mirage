from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.schemas import CausalGraph, Event


def handcrafted_graph() -> CausalGraph:
    actors = frozenset({"A", "B", "C"})
    timestamps = [i / 10 for i in range(10)]
    weights = [0.10, 1.00, 0.25, 0.35, 0.45, 0.55, 0.60, 0.50, 0.40, 0.20]
    participant_sets = [
        frozenset({"A"}),
        frozenset({"A", "B", "C"}),
        frozenset({"A"}),
        frozenset({"A", "B"}),
        frozenset({"B"}),
        frozenset({"A"}),
        frozenset({"A", "C"}),
        frozenset({"C"}),
        frozenset({"A", "B"}),
        frozenset({"A"}),
    ]

    events = []
    for idx, (timestamp, weight, event_actors) in enumerate(
        zip(timestamps, weights, participant_sets, strict=True)
    ):
        event_id = f"e{idx}"
        parents = (f"e{idx - 1}",) if idx > 0 else ()
        events.append(
            Event(
                id=event_id,
                timestamp=timestamp,
                weight=weight,
                actors=event_actors,
                causal_parents=parents,
            )
        )

    return CausalGraph(events=tuple(events), actors=actors, seed=99)


def test_greedy_preserves_development_with_k_ge_1() -> None:
    graph = handcrafted_graph()
    grammar = GrammarConfig.parametric(
        1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        focal_actor_coverage=0.0,
    )

    result = greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=10,
        injection_top_n=10,
    )

    assert result.valid is True
    assert result.n_development >= 1
    assert not any("insufficient_development" in violation for violation in result.violations)


def test_greedy_succeeds_when_k_zero() -> None:
    graph = handcrafted_graph()
    grammar = GrammarConfig.parametric(
        0,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        focal_actor_coverage=0.0,
    )

    result = greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=10,
        injection_top_n=10,
    )

    assert result.valid is True


def test_oracle_search_finds_valid_extraction() -> None:
    graph = handcrafted_graph()
    grammar = GrammarConfig.parametric(
        1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        focal_actor_coverage=0.0,
    )

    best, diagnostics = oracle_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        tp_window=(0.2, 1.0),
        max_sequence_length=10,
    )

    assert diagnostics["n_candidates_evaluated"] > 0
    assert diagnostics["n_valid_candidates"] > 0
    assert best is not None
    assert best.valid is True
    assert best.n_development >= 1
