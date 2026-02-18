"""Top-2 tropical extension for pivot-margin stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tropical_semiring import Event, NEG_INF


@dataclass
class TropicalTop2Context:
    k: int
    W1: np.ndarray
    W2: np.ndarray
    d_total: int

    @classmethod
    def empty(cls, k: int) -> "TropicalTop2Context":
        return cls(
            k=k,
            W1=np.full(k + 1, NEG_INF, dtype=float),
            W2=np.full(k + 1, NEG_INF, dtype=float),
            d_total=0,
        )

    @classmethod
    def from_event(cls, event: Event, k: int) -> "TropicalTop2Context":
        W1 = np.full(k + 1, NEG_INF, dtype=float)
        W2 = np.full(k + 1, NEG_INF, dtype=float)
        d_total = 0
        if event.is_focal:
            W1[0] = event.weight
        else:
            d_total = 1
        return cls(k=k, W1=W1, W2=W2, d_total=d_total)


def _insert_top2(W1: np.ndarray, W2: np.ndarray, idx: int, value: float, tol: float = 1e-12) -> None:
    if np.isneginf(value):
        return

    if value > W1[idx] + tol:
        W2[idx] = W1[idx]
        W1[idx] = value
    elif abs(value - W1[idx]) <= tol:
        if value > W2[idx]:
            W2[idx] = value
    elif value > W2[idx]:
        W2[idx] = value


def compose_top2(left: TropicalTop2Context, right: TropicalTop2Context) -> TropicalTop2Context:
    if left.k != right.k:
        raise ValueError("Cannot compose contexts with different k")

    k = left.k
    new_W1 = np.full(k + 1, NEG_INF, dtype=float)
    new_W2 = np.full(k + 1, NEG_INF, dtype=float)

    # Left contributes in-place.
    for x in range(k + 1):
        _insert_top2(new_W1, new_W2, x, left.W1[x])
        _insert_top2(new_W1, new_W2, x, left.W2[x])

    # Right shifts by left d_total.
    for x in range(k + 1):
        shifted = x + left.d_total
        if shifted > k:
            shifted = k
        _insert_top2(new_W1, new_W2, shifted, right.W1[x])
        _insert_top2(new_W1, new_W2, shifted, right.W2[x])

    return TropicalTop2Context(k=k, W1=new_W1, W2=new_W2, d_total=left.d_total + right.d_total)


def build_top2_context(events: Sequence[Event], k: int) -> TropicalTop2Context:
    acc = TropicalTop2Context.empty(k)
    for event in events:
        acc = compose_top2(acc, TropicalTop2Context.from_event(event, k))
    return acc


def pivot_margin(ctx: TropicalTop2Context, slot: int) -> float:
    w1 = ctx.W1[slot]
    w2 = ctx.W2[slot]
    if np.isneginf(w1):
        return 0.0
    if np.isneginf(w2):
        return float("inf")
    return float(w1 - w2)
