from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.extraction.viability_greedy import _compute_bridge_budget, viability_aware_greedy_extract
from rhun.schemas import CausalGraph, Event


def _timespan_trap_graph() -> CausalGraph:
    events = (
        Event(id="e0", timestamp=0.0, weight=1.00, actors=frozenset({"A"})),
        Event(id="e1", timestamp=1.0, weight=0.95, actors=frozenset({"A"}), causal_parents=("e0",)),
        Event(id="e2", timestamp=2.0, weight=0.90, actors=frozenset({"A"}), causal_parents=("e1",)),
        Event(id="e3", timestamp=3.0, weight=0.85, actors=frozenset({"A"}), causal_parents=("e2",)),
        Event(id="e4", timestamp=9.0, weight=0.25, actors=frozenset({"A"}), causal_parents=("e3",)),
        Event(id="e5", timestamp=10.0, weight=0.20, actors=frozenset({"A"}), causal_parents=("e4",)),
    )
    return CausalGraph(events=events, actors=frozenset({"A"}), seed=7)


def _gap_trap_graph() -> CausalGraph:
    events = (
        Event(id="g0", timestamp=0.0, weight=1.00, actors=frozenset({"A"})),
        Event(id="g1", timestamp=1.0, weight=0.95, actors=frozenset({"A"}), causal_parents=("g0",)),
        Event(id="g2", timestamp=2.0, weight=0.90, actors=frozenset({"A"}), causal_parents=("g1",)),
        Event(id="g3", timestamp=20.0, weight=0.30, actors=frozenset({"A"}), causal_parents=("g2",)),
    )
    return CausalGraph(events=events, actors=frozenset({"A"}), seed=8)


def _budget_trap_graph() -> CausalGraph:
    events = (
        Event(id="b0", timestamp=0.0, weight=1.00, actors=frozenset({"A"})),
        Event(id="b2", timestamp=1.0, weight=0.25, actors=frozenset({"A"})),
        Event(id="b3", timestamp=2.0, weight=0.24, actors=frozenset({"A"})),
        Event(id="b4", timestamp=3.0, weight=0.23, actors=frozenset({"A"})),
        Event(id="b1", timestamp=20.0, weight=0.95, actors=frozenset({"A"})),
    )
    return CausalGraph(events=events, actors=frozenset({"A"}), seed=10)


def _handcrafted_graph() -> CausalGraph:
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


def test_viability_aware_recovers_span_with_endpoint_swap() -> None:
    graph = _timespan_trap_graph()
    grammar = GrammarConfig.parametric(
        0,
        min_length=4,
        max_length=4,
        min_timespan_fraction=0.70,
        focal_actor_coverage=0.0,
    )

    greedy = greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=1,
        max_sequence_length=4,
        injection_top_n=6,
    )
    vag, diagnostics = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=1,
        max_sequence_length=4,
        injection_top_n=6,
    )

    assert greedy.valid is False
    assert any(v.startswith("insufficient_timespan") for v in greedy.violations)
    assert vag.valid is True
    assert diagnostics["total_viability_rejections"] > 0
    assert "span_upper_bound" in diagnostics["viability_rejection_reason_counts"]


def test_viability_aware_preserves_simple_valid_case() -> None:
    graph = _handcrafted_graph()
    grammar = GrammarConfig.parametric(
        1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        focal_actor_coverage=0.0,
    )

    greedy = greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=10,
        injection_top_n=10,
    )
    vag, diagnostics = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=10,
        injection_top_n=10,
    )

    assert greedy.valid is True
    assert vag.valid is True
    assert diagnostics["n_candidates_evaluated"] >= 1


def test_gap_aware_viability_rejects_unbridgeable_jump() -> None:
    graph = _gap_trap_graph()
    grammar = GrammarConfig.parametric(
        0,
        min_length=4,
        max_length=4,
        min_timespan_fraction=0.60,
        max_temporal_gap=3.0,
        focal_actor_coverage=0.0,
    )

    span_only, span_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=1,
        max_sequence_length=4,
        injection_top_n=10,
        gap_aware_viability=False,
    )
    gap_aware, gap_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=1,
        max_sequence_length=4,
        injection_top_n=10,
        gap_aware_viability=True,
    )

    assert span_only.valid is False
    assert any(v.startswith("max_temporal_gap:") for v in span_only.violations)
    assert gap_aware.valid is False
    assert any(
        key.startswith("gap_bridge_")
        for key in gap_diag["viability_rejection_reason_counts"].keys()
    )
    assert not any(
        key.startswith("gap_bridge_")
        for key in span_diag["viability_rejection_reason_counts"].keys()
    )


def test_bridge_budget_lower_bound_formula() -> None:
    timestamps = [0.0, 20.0]
    g = 3.0
    assert _compute_bridge_budget(timestamps, g) == 6


def test_budget_aware_viability_rejects_unaffordable_bridge_chain() -> None:
    graph = _budget_trap_graph()
    grammar = GrammarConfig.parametric(
        0,
        min_length=4,
        max_length=5,
        min_timespan_fraction=0.60,
        max_temporal_gap=3.0,
        focal_actor_coverage=0.0,
    )

    gap_aware, gap_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=1,
        max_sequence_length=5,
        injection_top_n=10,
        gap_aware_viability=True,
        budget_aware=False,
    )
    budget_aware, budget_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=1,
        max_sequence_length=5,
        injection_top_n=10,
        gap_aware_viability=True,
        budget_aware=True,
    )

    assert gap_aware.valid is False
    assert budget_aware.valid is False
    assert "bridge_budget_exceeded" in budget_diag["viability_rejection_reason_counts"]
    assert "bridge_budget_exceeded" not in gap_diag["viability_rejection_reason_counts"]

    per_anchor = budget_diag["per_anchor"]
    assert per_anchor
    assert any(anchor["first_budget_block"] is not None for anchor in per_anchor)
    assert any(anchor["budget_trace"] for anchor in per_anchor)


def test_budget_aware_is_noop_without_gap_constraint() -> None:
    graph = _handcrafted_graph()
    grammar = GrammarConfig.parametric(
        1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
        focal_actor_coverage=0.0,
    )

    standard, _standard_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=10,
        injection_top_n=10,
        gap_aware_viability=True,
        budget_aware=False,
    )
    budgeted, _budget_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=8,
        max_sequence_length=10,
        injection_top_n=10,
        gap_aware_viability=True,
        budget_aware=True,
    )

    assert standard.valid == budgeted.valid
    assert abs(standard.score - budgeted.score) <= 1e-9
    assert tuple(event.id for event in standard.events) == tuple(event.id for event in budgeted.events)


def test_viability_aware_returns_empty_sequence_when_no_pool_candidates() -> None:
    graph = _handcrafted_graph()
    grammar = GrammarConfig.parametric(1)

    vag, diagnostics = viability_aware_greedy_extract(
        graph=graph,
        focal_actor="A",
        grammar=grammar,
        pool_strategy="injection",
        n_anchors=0,
        max_sequence_length=5,
        injection_top_n=1,
    )

    assert vag.valid is False
    assert "empty_candidate_set" in vag.violations
    assert diagnostics["n_candidates_evaluated"] == 0
