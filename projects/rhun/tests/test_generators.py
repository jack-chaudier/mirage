from __future__ import annotations

from statistics import mean

from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.uniform import UniformConfig, UniformGenerator


def _max_weight_position(graph) -> float:
    if not graph.events:
        return 0.0
    max_event = max(graph.events, key=lambda event: (event.weight, -event.timestamp))
    idx = next(i for i, event in enumerate(graph.events) if event.id == max_event.id)
    return idx / (len(graph.events) - 1) if len(graph.events) > 1 else 0.0


def _assert_valid_dag(graph) -> None:
    by_id = {event.id: event for event in graph.events}
    timestamps = [event.timestamp for event in graph.events]
    assert timestamps == sorted(timestamps)

    for event in graph.events:
        for parent_id in event.causal_parents:
            assert parent_id in by_id
            assert by_id[parent_id].timestamp <= event.timestamp


def test_uniform_generator_produces_valid_graph() -> None:
    config = UniformConfig(n_events=60, n_actors=5, seed=123)
    graph = UniformGenerator().generate(config)

    assert graph.n_events == 60
    assert graph.n_actors == 5
    _assert_valid_dag(graph)


def test_bursty_generator_epsilon_front_loads_max_weight() -> None:
    generator = BurstyGenerator()
    low_positions = []
    high_positions = []

    for seed in range(1, 26):
        low = generator.generate(BurstyConfig(n_events=120, seed=seed, epsilon=0.0))
        high = generator.generate(BurstyConfig(n_events=120, seed=seed, epsilon=1.0))
        low_positions.append(_max_weight_position(low))
        high_positions.append(_max_weight_position(high))

    assert mean(high_positions) < mean(low_positions)
    assert mean(high_positions) < 0.25


def test_generator_determinism() -> None:
    uniform_cfg = UniformConfig(seed=77)
    bursty_cfg = BurstyConfig(seed=77, epsilon=0.6)

    g1 = UniformGenerator().generate(uniform_cfg)
    g2 = UniformGenerator().generate(uniform_cfg)
    b1 = BurstyGenerator().generate(bursty_cfg)
    b2 = BurstyGenerator().generate(bursty_cfg)

    assert g1 == g2
    assert b1 == b2


def test_event_and_actor_count_match_config() -> None:
    uniform = UniformGenerator().generate(UniformConfig(n_events=50, n_actors=4, seed=1))
    bursty = BurstyGenerator().generate(BurstyConfig(n_events=75, n_actors=7, seed=1))

    assert uniform.n_events == 50
    assert uniform.n_actors == 4
    assert bursty.n_events == 75
    assert bursty.n_actors == 7
