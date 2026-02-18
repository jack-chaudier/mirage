"""Validation against sequential phase grammar constraints."""

from __future__ import annotations

import math

from rhun.extraction.grammar import GrammarConfig
from rhun.schemas import CausalGraph, ExtractedSequence, Phase


def _phase_regressions(phases: tuple[Phase, ...]) -> list[tuple[int, Phase, Phase]]:
    regressions: list[tuple[int, Phase, Phase]] = []
    for i in range(1, len(phases)):
        previous = phases[i - 1]
        current = phases[i]
        if current < previous:
            regressions.append((i, previous, current))
    return regressions


def validate(
    sequence: ExtractedSequence,
    grammar: GrammarConfig,
    graph: CausalGraph,
) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_violations)."""

    violations: list[str] = []

    n = len(sequence.events)
    if n < grammar.min_length:
        violations.append(f"too_short: {n} < {grammar.min_length}")
    if n > grammar.max_length:
        violations.append(f"too_long: {n} > {grammar.max_length}")

    phases = sequence.phases
    tp_indices = [i for i, phase in enumerate(phases) if phase == Phase.TURNING_POINT]
    n_tp = len(tp_indices)
    if n_tp > grammar.max_turning_points:
        violations.append(f"too_many_turning_points: {n_tp} > {grammar.max_turning_points}")

    regressions = _phase_regressions(phases)
    if len(regressions) > grammar.max_phase_regressions:
        for i, previous, current in regressions:
            violations.append(
                f"phase_regression: {previous.name} -> {current.name} at position {i}"
            )

    if tp_indices:
        first_tp = tp_indices[0]
        n_development = sum(1 for phase in phases[:first_tp] if phase == Phase.DEVELOPMENT)
    else:
        n_development = 0

    if n_development < grammar.min_prefix_elements:
        violations.append(
            f"insufficient_development: {n_development} < {grammar.min_prefix_elements}"
        )

    if sequence.events:
        sequence_span = sequence.events[-1].timestamp - sequence.events[0].timestamp
    else:
        sequence_span = 0.0
    if graph.duration > 0:
        span_fraction = sequence_span / graph.duration
    else:
        span_fraction = 0.0
    if span_fraction < grammar.min_timespan_fraction:
        violations.append(
            f"insufficient_timespan: {span_fraction:.3f} < {grammar.min_timespan_fraction}"
        )

    if not math.isinf(grammar.max_temporal_gap) and n >= 2:
        for i in range(n - 1):
            left = sequence.events[i]
            right = sequence.events[i + 1]
            gap = float(right.timestamp - left.timestamp)
            if gap > grammar.max_temporal_gap + 1e-12:
                violations.append(
                    "max_temporal_gap: "
                    f"gap={gap:.3f} > max={grammar.max_temporal_gap:.3f} "
                    f"between events {left.id} and {right.id}"
                )

    if n > 0:
        focal_hits = sum(1 for event in sequence.events if sequence.focal_actor in event.actors)
        coverage = focal_hits / n
    else:
        coverage = 0.0
    if coverage < grammar.focal_actor_coverage:
        violations.append(
            f"insufficient_coverage: {coverage:.3f} < {grammar.focal_actor_coverage}"
        )

    return (len(violations) == 0), violations
