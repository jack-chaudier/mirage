from __future__ import annotations

from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


def _sample_graph() -> CausalGraph:
    events = (
        Event(id="e0", timestamp=0.0, weight=0.1, actors=frozenset({"A"})),
        Event(
            id="e1",
            timestamp=0.5,
            weight=0.8,
            actors=frozenset({"A", "B"}),
            causal_parents=("e0",),
        ),
        Event(
            id="e2",
            timestamp=1.0,
            weight=0.2,
            actors=frozenset({"B"}),
            causal_parents=("e1",),
        ),
    )
    return CausalGraph(events=events, actors=frozenset({"A", "B"}), seed=1)


def test_causal_graph_properties_and_lookup() -> None:
    graph = _sample_graph()

    assert graph.n_events == 3
    assert graph.n_actors == 2
    assert graph.duration == 1.0
    assert graph.event_by_id("e1") == graph.events[1]
    assert graph.event_by_id("missing") is None

    actor_events = graph.events_for_actor("A")
    assert tuple(event.id for event in actor_events) == ("e0", "e1")


def test_event_order_and_global_position() -> None:
    graph = _sample_graph()

    assert [event.timestamp for event in graph.events] == sorted(
        event.timestamp for event in graph.events
    )
    assert graph.global_position(graph.events[0]) == 0.0
    assert graph.global_position(graph.events[1]) == 0.5
    assert graph.global_position(graph.events[2]) == 1.0


def test_extracted_sequence_properties() -> None:
    graph = _sample_graph()
    sequence = ExtractedSequence(
        events=graph.events,
        phases=(Phase.SETUP, Phase.DEVELOPMENT, Phase.TURNING_POINT),
        focal_actor="A",
    )

    assert sequence.n_events == 3
    assert sequence.n_development == 1
    assert sequence.turning_point == graph.events[2]
