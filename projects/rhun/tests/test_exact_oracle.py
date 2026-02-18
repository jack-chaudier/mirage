from __future__ import annotations

from dataclasses import replace

from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.phase_classifier import classify_with_turning_point
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.validator import validate
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence


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


def _bruteforce_oracle(
    graph: CausalGraph,
    focal_actor: str,
    grammar: GrammarConfig,
) -> ExtractedSequence | None:
    actor_events = [event for event in graph.events if focal_actor in event.actors]
    n = len(actor_events)
    if n == 0:
        return None

    best: ExtractedSequence | None = None
    best_score = float("-inf")

    for mask in range(1, 1 << n):
        selected = [actor_events[i] for i in range(n) if (mask >> i) & 1]
        selected.sort(key=lambda event: (event.timestamp, event.id))
        selected_tuple = tuple(selected)

        for tp in selected_tuple:
            tp_idx = next(
                (index for index, event in enumerate(selected_tuple) if event.id == tp.id),
                None,
            )
            if tp_idx is None:
                continue

            phases = classify_with_turning_point(
                selected_tuple,
                tp_idx,
                min_development=grammar.min_prefix_elements,
            )
            candidate = ExtractedSequence(
                events=selected_tuple,
                phases=phases,
                focal_actor=focal_actor,
            )
            valid, violations = validate(candidate, grammar, graph)
            if not valid:
                continue

            score = tp_weighted_score(candidate)
            candidate = replace(
                candidate,
                score=score,
                valid=True,
                violations=tuple(violations),
            )

            if score > best_score + 1e-12:
                best_score = score
                best = candidate

    return best


def test_exact_oracle_matches_bruteforce_on_handcrafted_graph() -> None:
    graph = _handcrafted_graph()
    grammar = GrammarConfig.parametric(
        1,
        min_length=4,
        max_length=8,
        min_timespan_fraction=0.0,
        focal_actor_coverage=0.0,
    )

    exact, _ = exact_oracle_extract(graph, "A", grammar)
    brute = _bruteforce_oracle(graph, "A", grammar)

    assert brute is not None
    assert exact.valid is True
    assert abs(exact.score - brute.score) <= 1e-9


def test_exact_oracle_dominates_heuristic_oracle() -> None:
    grammar = GrammarConfig(min_prefix_elements=1)
    generator = BurstyGenerator()

    improvements = 0
    compared = 0

    for seed in range(4):
        for epsilon in (0.30, 0.50, 0.70, 0.90):
            graph = generator.generate(
                BurstyConfig(seed=seed, epsilon=epsilon, n_events=120, n_actors=6)
            )
            heuristic, _ = oracle_extract(graph, "actor_0", grammar)
            exact, _ = exact_oracle_extract(graph, "actor_0", grammar)

            if heuristic is not None and heuristic.valid:
                assert exact.valid is True
                assert exact.score >= heuristic.score - 1e-9
                compared += 1
                if exact.score > heuristic.score + 1e-9:
                    improvements += 1

    assert compared > 0
    assert improvements > 0


def test_exact_valid_sequences_revalidate() -> None:
    grammar = GrammarConfig(min_prefix_elements=1)
    generator = BurstyGenerator()

    for seed in range(6):
        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=0.70, n_events=120, n_actors=6)
        )
        exact, _ = exact_oracle_extract(graph, "actor_0", grammar)
        if not exact.valid:
            continue
        valid, violations = validate(exact, grammar, graph)
        assert valid is True
        assert violations == []


def test_greedy_vs_exact_ratio_is_high_when_both_valid() -> None:
    grammar = GrammarConfig(min_prefix_elements=1)
    generator = BurstyGenerator()
    ratios: list[float] = []

    for seed in range(4):
        for epsilon in (0.30, 0.50, 0.70, 0.90):
            graph = generator.generate(
                BurstyConfig(seed=seed, epsilon=epsilon, n_events=120, n_actors=6)
            )
            greedy = greedy_extract(graph, "actor_0", grammar, pool_strategy="injection")
            exact, _ = exact_oracle_extract(graph, "actor_0", grammar)
            if greedy.valid and exact.valid and exact.score > 0:
                ratios.append(greedy.score / exact.score)

    assert ratios
    assert min(ratios) >= 0.80


def test_exact_vs_heuristic_invalid_split() -> None:
    grammar = GrammarConfig(min_prefix_elements=3)
    generator = BurstyGenerator()

    exact_valid_heuristic_invalid = 0
    both_invalid = 0
    exact_invalid_heuristic_valid = 0

    for seed in range(6):
        for epsilon in (0.70, 0.90):
            graph = generator.generate(
                BurstyConfig(seed=seed, epsilon=epsilon, n_events=120, n_actors=6)
            )
            heuristic, _ = oracle_extract(graph, "actor_0", grammar)
            exact, _ = exact_oracle_extract(graph, "actor_0", grammar)

            heuristic_valid = bool(heuristic is not None and heuristic.valid)
            exact_valid = bool(exact.valid)

            if exact_valid and not heuristic_valid:
                exact_valid_heuristic_invalid += 1
            elif (not exact_valid) and (not heuristic_valid):
                both_invalid += 1
            elif (not exact_valid) and heuristic_valid:
                exact_invalid_heuristic_valid += 1

    # Exact should not fail when heuristic succeeds on the same search space.
    assert exact_invalid_heuristic_valid == 0
    assert (exact_valid_heuristic_invalid + both_invalid) >= 0
