from __future__ import annotations

from narrativefield.extraction.types import ArcScore
from narrativefield.schema.events import BeatType, Event

TENSION_VARIANCE_NORMALIZATION = 0.013


def score_arc(events: list[Event], beats: list[BeatType]) -> ArcScore:
    """
    Soft-scoring for grammar-valid arcs.

    Source: specs/metrics/story-extraction.md Section 4.1.
    """
    tensions = [float(e.metrics.tension) for e in events]

    # 1. Tension variance
    tension_variance = _variance(tensions)
    tension_variance_score = min(tension_variance / TENSION_VARIANCE_NORMALIZATION, 1.0)

    # 2. Peak tension
    peak_tension = max(tensions) if tensions else 0.0
    peak_tension_score = peak_tension

    # 3. Tension shape
    tension_shape_score = _evaluate_tension_shape(tensions)

    # 4. Turning point significance (Phase 5; defaults to 0.0 for now)
    tp_events = [e for e, b in zip(events, beats) if b == BeatType.TURNING_POINT]
    tp_significance = float(tp_events[0].metrics.significance) if tp_events else 0.0
    significance_score = tp_significance

    # 5. Thematic coherence
    thematic_coherence_score = _thematic_coherence(events)

    # 6. Irony arc
    irony_arc_score = _evaluate_irony_arc(events)

    # 7. Protagonist dominance
    protagonist_score = _protagonist_dominance(events)

    composite = (
        0.20 * tension_variance_score
        + 0.15 * peak_tension_score
        + 0.15 * tension_shape_score
        + 0.15 * significance_score
        + 0.15 * thematic_coherence_score
        + 0.10 * irony_arc_score
        + 0.10 * protagonist_score
    )

    return ArcScore(
        composite=float(composite),
        tension_variance=float(tension_variance_score),
        peak_tension=float(peak_tension_score),
        tension_shape=float(tension_shape_score),
        significance=float(significance_score),
        thematic_coherence=float(thematic_coherence_score),
        irony_arc=float(irony_arc_score),
        protagonist_dominance=float(protagonist_score),
    )


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def _evaluate_tension_shape(tensions: list[float]) -> float:
    """
    Score classic dramatic shape: rise -> peak -> fall.
    """
    if len(tensions) < 3:
        return 0.0

    peak_idx = tensions.index(max(tensions))
    peak_position = peak_idx / (len(tensions) - 1)

    if 0.5 <= peak_position <= 0.8:
        position_score = 1.0
    elif 0.4 <= peak_position <= 0.9:
        position_score = 0.7
    else:
        position_score = 0.3

    pre_peak = tensions[: peak_idx + 1]
    post_peak = tensions[peak_idx:]

    rising_count = sum(1 for i in range(1, len(pre_peak)) if pre_peak[i] >= pre_peak[i - 1])
    rising_ratio = rising_count / max(len(pre_peak) - 1, 1)

    falling_count = sum(1 for i in range(1, len(post_peak)) if post_peak[i] <= post_peak[i - 1])
    falling_ratio = falling_count / max(len(post_peak) - 1, 1)

    shape_score = (rising_ratio + falling_ratio) / 2.0
    return position_score * 0.5 + shape_score * 0.5


def _thematic_coherence(events: list[Event]) -> float:
    axis_totals: dict[str, float] = {}
    for e in events:
        for axis, delta in (e.metrics.thematic_shift or {}).items():
            axis_totals[axis] = axis_totals.get(axis, 0.0) + abs(float(delta))

    if not axis_totals:
        return 0.5

    total = sum(axis_totals.values())
    if total <= 1e-9:
        return 0.5

    return max(axis_totals.values()) / total


def _evaluate_irony_arc(events: list[Event]) -> float:
    if len(events) < 4:
        return 0.0
    ironies = [float(e.metrics.irony) for e in events]
    mid = len(ironies) // 2
    first = sum(ironies[:mid]) / max(mid, 1)
    second = sum(ironies[mid:]) / max(len(ironies) - mid, 1)

    if first > 0 and second < first:
        drop_ratio = (first - second) / first
        return min(drop_ratio, 1.0)
    return 0.2


def _protagonist_dominance(events: list[Event]) -> float:
    counts: dict[str, int] = {}
    for e in events:
        for a in {e.source_agent, *e.target_agents}:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / max(1, len(events))
