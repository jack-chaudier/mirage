from __future__ import annotations

from statistics import mean

from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator


def _assert_valid_dag(graph) -> None:
    by_id = {event.id: event for event in graph.events}
    timestamps = [event.timestamp for event in graph.events]
    assert timestamps == sorted(timestamps)

    for event in graph.events:
        for parent_id in event.causal_parents:
            assert parent_id in by_id
            assert by_id[parent_id].timestamp <= event.timestamp


def _region_counts(graph, config: MultiBurstConfig) -> dict[str, int]:
    c1, c2 = sorted(config.burst_centers)
    w = config.burst_width
    b1 = (max(0.0, c1 - w), min(1.0, c1 + w))
    b2 = (max(0.0, c2 - w), min(1.0, c2 + w))

    counts = {"burst1": 0, "burst2": 0, "inter": 0, "outside": 0}
    inter = (b1[1], b2[0]) if b1[1] < b2[0] else None

    for event in graph.events:
        t = float(event.timestamp)
        if b1[0] <= t <= b1[1]:
            counts["burst1"] += 1
        elif b2[0] <= t <= b2[1]:
            counts["burst2"] += 1
        elif inter is not None and inter[0] <= t <= inter[1]:
            counts["inter"] += 1
        else:
            counts["outside"] += 1

    return counts


def test_multiburst_generator_produces_valid_graph() -> None:
    config = MultiBurstConfig(n_events=150, n_actors=6, seed=123)
    graph = MultiBurstGenerator().generate(config)

    assert graph.n_events == 150
    assert graph.n_actors == 6
    _assert_valid_dag(graph)


def test_multiburst_generator_is_deterministic() -> None:
    config = MultiBurstConfig(n_events=180, n_actors=5, seed=77)
    generator = MultiBurstGenerator()

    g1 = generator.generate(config)
    g2 = generator.generate(config)

    assert g1 == g2


def test_multiburst_density_concentrates_in_bursts() -> None:
    generator = MultiBurstGenerator()
    burst_counts = []
    inter_counts = []

    for seed in range(1, 16):
        config = MultiBurstConfig(
            n_events=200,
            n_actors=6,
            seed=seed,
            burst_centers=(0.2, 0.8),
            burst_width=0.1,
            inter_burst_density=0.3,
        )
        graph = generator.generate(config)
        counts = _region_counts(graph, config)
        burst_counts.append(counts["burst1"] + counts["burst2"])
        inter_counts.append(counts["inter"])

    assert mean(burst_counts) > mean(inter_counts)


def test_multiburst_weights_are_higher_in_bursts() -> None:
    config = MultiBurstConfig(
        n_events=220,
        n_actors=6,
        seed=5,
        burst_centers=(0.2, 0.8),
        burst_width=0.1,
        burst_weight_boost=3.0,
        inter_burst_density=0.3,
    )
    graph = MultiBurstGenerator().generate(config)

    c1, c2 = sorted(config.burst_centers)
    w = config.burst_width
    b1 = (max(0.0, c1 - w), min(1.0, c1 + w))
    b2 = (max(0.0, c2 - w), min(1.0, c2 + w))

    burst_weights = [
        event.weight
        for event in graph.events
        if (b1[0] <= event.timestamp <= b1[1]) or (b2[0] <= event.timestamp <= b2[1])
    ]
    inter_weights = [
        event.weight
        for event in graph.events
        if (b1[1] < event.timestamp < b2[0])
    ]

    assert burst_weights
    assert inter_weights
    assert mean(burst_weights) > mean(inter_weights)
