from __future__ import annotations

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_phases
from rhun.extraction.search import greedy_extract
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase
from rhun.theory.theorem import check_precondition, diagnose_absorption, verify_prediction


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
        parents = (f"e{idx - 1}",) if idx > 0 else ()
        events.append(
            Event(
                id=f"e{idx}",
                timestamp=timestamp,
                weight=weight,
                actors=event_actors,
                causal_parents=parents,
            )
        )

    return CausalGraph(events=tuple(events), actors=actors, seed=99)


def _max_at_zero_graph() -> CausalGraph:
    base = _handcrafted_graph()
    events = list(base.events)
    events[0] = Event(
        id=events[0].id,
        timestamp=events[0].timestamp,
        weight=1.2,
        actors=events[0].actors,
        causal_parents=events[0].causal_parents,
    )
    return CausalGraph(events=tuple(events), actors=base.actors, seed=100)


def _dummy_result(valid: bool) -> ExtractedSequence:
    event = Event(id="x", timestamp=0.0, weight=0.1, actors=frozenset({"A"}))
    return ExtractedSequence(
        events=(event,),
        phases=(Phase.TURNING_POINT,),
        focal_actor="A",
        valid=valid,
    )


def test_check_precondition_identifies_early_max() -> None:
    graph = _handcrafted_graph()
    grammar = GrammarConfig.parametric(2)

    pre = check_precondition(graph, "A", grammar)

    assert pre["events_before_max"] == 1
    assert pre["precondition_met"] is True
    assert pre["predicted_failure"] is True


def test_verify_prediction_tp_tn_fp_fn() -> None:
    early_graph = _handcrafted_graph()  # predicted failure for k=2
    late_graph = _handcrafted_graph()  # predicted success for k=1

    tp = verify_prediction(early_graph, "A", GrammarConfig.parametric(2), _dummy_result(False))
    tn = verify_prediction(late_graph, "A", GrammarConfig.parametric(1), _dummy_result(True))
    fp = verify_prediction(early_graph, "A", GrammarConfig.parametric(2), _dummy_result(True))
    fn = verify_prediction(late_graph, "A", GrammarConfig.parametric(1), _dummy_result(False))

    assert tp["prediction_correct"] is True
    assert tn["prediction_correct"] is True
    assert fp["prediction_correct"] is False
    assert fp["mismatch_reason"] == "predicted_failure_but_valid_sequence_found"
    assert fn["prediction_correct"] is False
    assert fn["mismatch_reason"] == "predicted_success_but_no_valid_sequence_found"


def test_max_weight_position_zero_predicts_and_observes_failure() -> None:
    graph = _max_at_zero_graph()
    grammar = GrammarConfig.parametric(
        1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        focal_actor_coverage=0.0,
    )

    pre = check_precondition(graph, "A", grammar)
    result = greedy_extract(graph, "A", grammar, pool_strategy="injection", max_sequence_length=10)

    assert pre["max_weight_index"] == 0
    assert pre["predicted_failure"] is True
    assert result.valid is False


def test_diagnose_absorption_detects_step_and_waste() -> None:
    graph = _handcrafted_graph()
    events = graph.events[:6]
    phases = classify_phases(events)
    seq = ExtractedSequence(events=events, phases=phases, focal_actor="A")

    diagnosis = diagnose_absorption(seq, GrammarConfig.parametric(1))

    assert diagnosis["absorbed"] is True
    assert diagnosis["absorption_step"] == 1
    assert diagnosis["development_at_absorption"] == 0
    assert diagnosis["events_after_absorption"] == len(events) - 2


def test_diagnose_absorption_false_for_valid_mid_sequence_tp() -> None:
    graph = _handcrafted_graph()
    seq = ExtractedSequence(
        events=(graph.events[0], graph.events[2], graph.events[6], graph.events[9]),
        phases=(Phase.SETUP, Phase.DEVELOPMENT, Phase.TURNING_POINT, Phase.RESOLUTION),
        focal_actor="A",
        valid=True,
    )

    diagnosis = diagnose_absorption(seq, GrammarConfig.parametric(1))
    assert diagnosis["absorbed"] is False
    assert diagnosis["absorption_step"] is None
