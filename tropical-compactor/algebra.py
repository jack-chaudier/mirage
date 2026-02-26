"""L2 tropical-algebra frontier scan with provenance tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def _validate_k(k: int) -> None:
    if k < 0:
        raise ValueError("k must be >= 0")


@dataclass(slots=True)
class ChunkState:
    """L1 state for one chunk in the stream."""

    chunk_id: str
    weight: float
    d_total: int
    text: str

    def __post_init__(self) -> None:
        if self.d_total < 0:
            raise ValueError("d_total must be >= 0")


@dataclass(slots=True)
class Provenance:
    """Witness of which pivot/predecessors realize a frontier slot."""

    pivot_id: str
    pred_ids: list[str]
    weight: float


@dataclass(slots=True)
class L2Summary:
    """L2 frontier summary for a block of chunks."""

    k: int
    d_total: int
    W: list[float]
    provenance: list[Optional[Provenance]]
    predecessor_tail: list[str]

    @classmethod
    def identity(cls, k: int) -> "L2Summary":
        _validate_k(k)
        return cls(
            k=k,
            d_total=0,
            W=[-math.inf] * (k + 1),
            provenance=[None] * (k + 1),
            predecessor_tail=[],
        )

    @classmethod
    def from_chunk(cls, chunk: ChunkState, k: int) -> "L2Summary":
        _validate_k(k)
        W = [-math.inf] * (k + 1)
        prov: list[Optional[Provenance]] = [None] * (k + 1)

        if math.isfinite(chunk.weight):
            W[0] = chunk.weight
            prov[0] = Provenance(
                pivot_id=chunk.chunk_id,
                pred_ids=[],
                weight=chunk.weight,
            )

        d_total = min(k, max(0, chunk.d_total))
        pred_tail = [chunk.chunk_id] if d_total > 0 and k > 0 else []

        return cls(
            k=k,
            d_total=d_total,
            W=W,
            provenance=prov,
            predecessor_tail=pred_tail,
        )


def compose(prev: L2Summary, incoming: L2Summary, k: int) -> L2Summary:
    """
    Compose adjacent L2 summaries.

    Implements:
        W_new[j] = max(W_prev[j], W_incoming[max(0, j - d_prev)])
    """

    _validate_k(k)
    if prev.k != incoming.k or prev.k != k:
        raise ValueError("Mismatched k in L2 summary composition")

    new_d = min(k, prev.d_total + incoming.d_total)
    new_W = [-math.inf] * (k + 1)
    new_prov: list[Optional[Provenance]] = [None] * (k + 1)

    for j in range(k + 1):
        # Option A: winner lives in previous block.
        w_a = prev.W[j]
        prov_a = prev.provenance[j]

        # Option B: winner lives in incoming block and consumes left predecessors.
        idx = max(0, j - prev.d_total)
        w_b = incoming.W[idx]
        prov_b_base = incoming.provenance[idx]
        prov_b: Optional[Provenance] = None

        if math.isfinite(w_b) and prov_b_base is not None:
            needed_from_left = j - idx
            left_preds = prev.predecessor_tail[-needed_from_left:] if needed_from_left > 0 else []
            combined = left_preds + list(prov_b_base.pred_ids)
            pred_ids = combined[-j:] if j > 0 else []
            prov_b = Provenance(
                pivot_id=prov_b_base.pivot_id,
                pred_ids=pred_ids,
                weight=prov_b_base.weight,
            )

        if w_b > w_a:
            new_W[j] = w_b
            new_prov[j] = prov_b
        else:
            # Tie-break toward left side for deterministic composition.
            new_W[j] = w_a
            new_prov[j] = prov_a

    new_tail = (prev.predecessor_tail + incoming.predecessor_tail)[-k:] if k > 0 else []

    return L2Summary(
        k=k,
        d_total=new_d,
        W=new_W,
        provenance=new_prov,
        predecessor_tail=new_tail,
    )


def l2_scan(chunks: list[ChunkState], k: int) -> L2Summary:
    """Fold chunks left-to-right through the L2 algebra."""

    state = L2Summary.identity(k)
    for chunk in chunks:
        state = compose(state, L2Summary.from_chunk(chunk, k), k)
    return state


def protected_set(chunks: list[ChunkState], k: int) -> tuple[set[str], bool]:
    """
    Return (protected_ids, contract_satisfied).

    contract_satisfied is true only when a valid k-predecessor pivot exists.
    """

    summary = l2_scan(chunks, k)
    prov = summary.provenance[k]

    if prov is None:
        return set(), False

    protected = {prov.pivot_id} | set(prov.pred_ids)
    return protected, True
