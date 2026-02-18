"""Core algebra for the Tropical Endogenous Context Semiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

NEG_INF = float("-inf")


@dataclass(frozen=True)
class Event:
    """Single timeline event."""

    eid: int
    timestamp: float
    weight: float
    actor: str
    is_focal: bool


@dataclass
class TropicalContext:
    """Tropical context element T = (W, d_total)."""

    k: int
    W: np.ndarray
    d_total: int

    @classmethod
    def empty(cls, k: int) -> "TropicalContext":
        return cls(k=k, W=np.full(k + 1, NEG_INF, dtype=float), d_total=0)

    @classmethod
    def from_event(cls, event: Event, k: int) -> "TropicalContext":
        W = np.full(k + 1, NEG_INF, dtype=float)
        if event.is_focal:
            W[0] = event.weight
            d_total = 0
        else:
            d_total = 1
        return cls(k=k, W=W, d_total=d_total)

    def copy(self) -> "TropicalContext":
        return TropicalContext(k=self.k, W=self.W.copy(), d_total=self.d_total)

    def compose(self, other: "TropicalContext") -> "TropicalContext":
        return compose_tropical(self, other)

    def feasible(self) -> bool:
        return self.W[self.k] > NEG_INF

    def best_weight(self) -> float:
        m = float(np.max(self.W))
        return m

    def best_slot(self) -> int:
        if np.all(np.isneginf(self.W)):
            return -1
        return int(np.argmax(self.W))


@dataclass
class OriginalContext:
    """Original Paper-3 monoid context C = (w*, d_total, d_pre)."""

    w_star: float
    d_total: int
    d_pre: int

    @classmethod
    def empty(cls) -> "OriginalContext":
        return cls(w_star=NEG_INF, d_total=0, d_pre=0)

    @classmethod
    def from_event(cls, event: Event) -> "OriginalContext":
        if event.is_focal:
            return cls(w_star=event.weight, d_total=0, d_pre=0)
        return cls(w_star=NEG_INF, d_total=1, d_pre=0)

    def compose(self, other: "OriginalContext") -> "OriginalContext":
        w_star = max(self.w_star, other.w_star)
        d_total = self.d_total + other.d_total
        if self.w_star >= other.w_star:
            d_pre = self.d_pre
        else:
            d_pre = self.d_total + other.d_pre
        return OriginalContext(w_star=w_star, d_total=d_total, d_pre=d_pre)


def compose_tropical(left: TropicalContext, right: TropicalContext) -> TropicalContext:
    """Tropical composition for adjacent blocks left then right."""

    if left.k != right.k:
        raise ValueError("Cannot compose contexts with different k")

    k = left.k
    d_total = left.d_total + right.d_total
    W_new = np.full(k + 1, NEG_INF, dtype=float)

    # Left pivots keep capacity slots unchanged.
    np.maximum(W_new, left.W, out=W_new)

    # Right pivots gain full development capacity from the left block.
    for x_b, w in enumerate(right.W):
        if np.isneginf(w):
            continue
        x_new = x_b + left.d_total
        if x_new > k:
            x_new = k
        if w > W_new[x_new]:
            W_new[x_new] = w

    return TropicalContext(k=k, W=W_new, d_total=d_total)


def build_tropical_context(events: Sequence[Event], k: int) -> TropicalContext:
    """Left-fold tropical composition over a sequence of events."""

    acc = TropicalContext.empty(k)
    for event in events:
        acc = compose_tropical(acc, TropicalContext.from_event(event, k))
    return acc


def brute_force_tropical_context(events: Sequence[Event], k: int) -> TropicalContext:
    """Ground-truth W via direct pivot enumeration."""

    W = np.full(k + 1, NEG_INF, dtype=float)
    d_total = 0
    for event in events:
        if event.is_focal:
            slot = d_total if d_total < k else k
            if event.weight > W[slot]:
                W[slot] = event.weight
        else:
            d_total += 1
    return TropicalContext(k=k, W=W, d_total=d_total)


def build_original_context(events: Sequence[Event]) -> OriginalContext:
    """Left-fold composition in the original monoid."""

    acc = OriginalContext.empty()
    for event in events:
        acc = acc.compose(OriginalContext.from_event(event))
    return acc


def count_occupied_slots(W: np.ndarray) -> int:
    return int(np.sum(~np.isneginf(W)))


def focal_pivots_with_prefix(
    events: Sequence[Event],
) -> List[Tuple[float, int, int, Event]]:
    """Return tuples of (weight, prefix_dev, position, event) for focal pivots."""

    pivots: List[Tuple[float, int, int, Event]] = []
    dev = 0
    for pos, event in enumerate(events):
        if event.is_focal:
            pivots.append((event.weight, dev, pos, event))
        else:
            dev += 1
    return pivots


def best_feasible_pivot(
    events: Sequence[Event],
    k: int,
    top_m: Optional[int] = None,
) -> Tuple[float, Optional[int], Optional[int]]:
    """Best feasible pivot among all focal events (or top_m by weight).

    Returns (weight, prefix_dev, eid) or (-inf, None, None).
    """

    pivots = focal_pivots_with_prefix(events)
    if not pivots:
        return NEG_INF, None, None

    if top_m is not None and top_m > 0:
        pivots = sorted(pivots, key=lambda t: t[0], reverse=True)[:top_m]

    best_weight = NEG_INF
    best_prefix: Optional[int] = None
    best_eid: Optional[int] = None
    for weight, prefix_dev, _, event in pivots:
        if prefix_dev < k:
            continue
        if weight > best_weight:
            best_weight = weight
            best_prefix = prefix_dev
            best_eid = event.eid
    return best_weight, best_prefix, best_eid


def max_slot_for_weight(W: np.ndarray, weight: float, tol: float = 1e-12) -> Optional[int]:
    if np.isneginf(weight):
        return None
    matches = np.where(np.abs(W - weight) <= tol)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def harmonic_number(n: int) -> float:
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1, dtype=float)))
