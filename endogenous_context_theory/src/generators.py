"""Synthetic event generators for endogenous context experiments."""

from __future__ import annotations

from typing import List

import numpy as np

from .tropical_semiring import Event


def _timestamp(i: int, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(i / (n - 1))


def _interleaved_focal_mask(n: int, n_focal: int) -> List[bool]:
    """Distribute exactly n_focal focal flags approximately evenly over n events."""

    if n_focal < 0:
        raise ValueError("n_focal must be >= 0")
    if n_focal > n:
        raise ValueError("n_focal cannot exceed n")
    if n_focal == 0:
        return [False] * n
    if n_focal == n:
        return [True] * n

    mask = [False] * n
    prev_bucket = -1
    for i in range(n):
        bucket = (i * n_focal) // n
        if bucket != prev_bucket and bucket < n_focal:
            mask[i] = True
        prev_bucket = bucket
    return mask


def bursty_generator(n: int, n_focal: int, epsilon: float, seed: int) -> List[Event]:
    """Bursty generator with front-loaded high-weight zone."""

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")
    if n_focal > n:
        raise ValueError("n_focal cannot exceed n")

    rng = np.random.default_rng(seed)
    focal_mask = _interleaved_focal_mask(n=n, n_focal=n_focal)
    events: List[Event] = []
    for i in range(n):
        t = _timestamp(i, n)
        if t <= epsilon:
            weight = float(rng.uniform(5.0, 20.0))
        else:
            weight = float(rng.uniform(0.1, 5.0))

        is_focal = focal_mask[i]
        actor = "focal" if is_focal else f"dev_{i % 7}"
        events.append(
            Event(
                eid=i,
                timestamp=t,
                weight=weight,
                actor=actor,
                is_focal=is_focal,
            )
        )
    return events


def multi_burst_generator(
    n: int,
    n_focal: int,
    n_bursts: int,
    seed: int,
    burst_width: float = 0.15,
) -> List[Event]:
    """Multi-burst generator with high-weight clusters and low valleys."""

    if n_bursts <= 0:
        raise ValueError("n_bursts must be positive")
    if n_focal > n:
        raise ValueError("n_focal cannot exceed n")

    rng = np.random.default_rng(seed)
    focal_mask = _interleaved_focal_mask(n=n, n_focal=n_focal)
    centers = np.linspace(0.0, 1.0, n_bursts + 2)[1:-1]
    half_width = burst_width / 2.0

    events: List[Event] = []
    for i in range(n):
        t = _timestamp(i, n)
        in_burst = np.any(np.abs(centers - t) <= half_width)
        if in_burst:
            weight = float(rng.uniform(5.0, 20.0))
        else:
            weight = float(rng.uniform(0.1, 2.0))

        is_focal = focal_mask[i]
        actor = "focal" if is_focal else f"dev_{i % 11}"
        events.append(
            Event(
                eid=i,
                timestamp=t,
                weight=weight,
                actor=actor,
                is_focal=is_focal,
            )
        )
    return events


def adversarial_oscillation_generator(
    n: int,
    k: int,
    osc_period: int,
    seed: int,
) -> List[Event]:
    """Alternating focal spikes that force repeated running-max pivot shifts."""

    if osc_period <= 1:
        raise ValueError("osc_period must be > 1")

    rng = np.random.default_rng(seed)
    p_eff = max(1, osc_period // 2)

    events: List[Event] = []
    spike_index = 0
    for i in range(n):
        t = _timestamp(i, n)
        phase = i % osc_period
        is_spike = phase == 0 or phase == p_eff

        if is_spike:
            actor = "focal_a" if spike_index % 2 == 0 else "focal_b"
            # Strictly increasing spikes induce record shifts.
            weight = float(10.0 + spike_index + rng.uniform(0.0, 0.05))
            is_focal = True
            spike_index += 1
        else:
            actor = f"dev_{(i + k) % 13}"
            weight = float(rng.uniform(0.1, 2.0))
            is_focal = False

        events.append(
            Event(
                eid=i,
                timestamp=t,
                weight=weight,
                actor=actor,
                is_focal=is_focal,
            )
        )
    return events


def streaming_generator(n: int, n_focal: int, epsilon: float, seed: int) -> List[Event]:
    """Streaming generator using bursty weights (one event at a time)."""

    return bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
