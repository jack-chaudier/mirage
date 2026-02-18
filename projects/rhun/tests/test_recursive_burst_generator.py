from __future__ import annotations

from statistics import mean

from rhun.generators.recursive_burst import RecursiveBurstConfig, RecursiveBurstGenerator


def _max_weight_position(graph) -> float:
    if not graph.events:
        return 0.0
    max_event = max(graph.events, key=lambda event: (event.weight, -event.timestamp))
    idx = next(i for i, event in enumerate(graph.events) if event.id == max_event.id)
    return idx / (len(graph.events) - 1) if len(graph.events) > 1 else 0.0


def test_recursive_burst_generator_is_deterministic() -> None:
    generator = RecursiveBurstGenerator()
    config = RecursiveBurstConfig(seed=123, n_events=120, n_actors=5, epsilon=0.7)
    g1 = generator.generate(config)
    g2 = generator.generate(config)
    assert g1 == g2


def test_recursive_burst_frontloads_high_weights_with_epsilon() -> None:
    generator = RecursiveBurstGenerator()
    low_positions = []
    high_positions = []

    for seed in range(1, 21):
        low = generator.generate(RecursiveBurstConfig(seed=seed, n_events=160, epsilon=0.1))
        high = generator.generate(RecursiveBurstConfig(seed=seed, n_events=160, epsilon=0.9))
        low_positions.append(_max_weight_position(low))
        high_positions.append(_max_weight_position(high))

    assert mean(high_positions) < mean(low_positions)
