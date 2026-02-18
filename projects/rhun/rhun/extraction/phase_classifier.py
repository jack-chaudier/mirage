"""Phase classification for extracted event sequences."""

from __future__ import annotations

from math import ceil
from typing import Sequence

from rhun.schemas import Event, Phase


def turning_point_index(events: Sequence[Event]) -> int | None:
    """Return the deterministic turning-point index (argmax weight, earliest tie)."""
    if not events:
        return None
    return max(range(len(events)), key=lambda i: (events[i].weight, -events[i].timestamp, -i))


def enforce_monotonic(phases: Sequence[Phase]) -> tuple[Phase, ...]:
    """Ensure phases never regress by clamping each phase to max(seen_so_far)."""
    if not phases:
        return ()

    monotonic: list[Phase] = []
    current = phases[0]
    monotonic.append(current)
    for phase in phases[1:]:
        if phase < current:
            monotonic.append(current)
        else:
            current = phase
            monotonic.append(phase)
    return tuple(monotonic)


def classify_with_turning_point(
    events: Sequence[Event],
    turning_point_idx: int,
    min_development: int = 0,
) -> tuple[Phase, ...]:
    """
    Classify phases using a forced turning-point index.

    `min_development` preserves capacity in the prefix for development labels.
    When positive, setup allocation is capped so at least that many prefix
    events can be labeled DEVELOPMENT when they exist.
    """
    if not events:
        return ()

    phases: list[Phase] = [Phase.SETUP] * len(events)

    n_before_tp = max(0, turning_point_idx)
    n_setup = ceil(n_before_tp * 0.2) if n_before_tp > 0 else 0
    if min_development > 0:
        max_setup_for_development = max(0, n_before_tp - min_development)
        n_setup = min(n_setup, max_setup_for_development)

    for idx in range(turning_point_idx):
        phases[idx] = Phase.SETUP if idx < n_setup else Phase.DEVELOPMENT

    phases[turning_point_idx] = Phase.TURNING_POINT

    for idx in range(turning_point_idx + 1, len(events)):
        phases[idx] = Phase.RESOLUTION

    return enforce_monotonic(phases)


def classify_phases(
    events: Sequence[Event],
    min_development: int = 0,
) -> tuple[Phase, ...]:
    """Classify events into monotonic sequential phases."""
    tp_idx = turning_point_index(events)
    if tp_idx is None:
        return ()
    return classify_with_turning_point(
        events,
        tp_idx,
        min_development=min_development,
    )
