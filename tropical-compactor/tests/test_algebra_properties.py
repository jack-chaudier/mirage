from __future__ import annotations

import math
import random

from algebra import ChunkState, l2_scan, protected_set


def _random_chunks(rng: random.Random, n: int) -> list[ChunkState]:
    chunks: list[ChunkState] = []
    for i in range(n):
        chunk_id = f"c{i}"
        role_roll = rng.random()
        if role_roll < 0.45:
            chunks.append(ChunkState(chunk_id=chunk_id, weight=-math.inf, d_total=1, text="pred"))
        elif role_roll < 0.85:
            chunks.append(ChunkState(chunk_id=chunk_id, weight=rng.uniform(0.0, 25.0), d_total=0, text="pivot"))
        else:
            chunks.append(ChunkState(chunk_id=chunk_id, weight=-math.inf, d_total=0, text="noise"))
    return chunks


def _bruteforce(chunks: list[ChunkState], k: int) -> tuple[list[float], list[tuple[str, list[str], float] | None], int]:
    W = [-math.inf] * (k + 1)
    prov: list[tuple[str, list[str], float] | None] = [None] * (k + 1)

    predecessor_prefix: list[list[str]] = []
    seen_preds: list[str] = []
    for chunk in chunks:
        predecessor_prefix.append(seen_preds.copy())
        if chunk.d_total > 0:
            seen_preds.append(chunk.chunk_id)

    for j in range(k + 1):
        best_weight = -math.inf
        best_idx: int | None = None
        best_pred_ids: list[str] = []
        best_pivot_id: str | None = None

        for idx, chunk in enumerate(chunks):
            if not math.isfinite(chunk.weight):
                continue

            preds_before = predecessor_prefix[idx]
            if len(preds_before) < j:
                continue

            if chunk.weight > best_weight:
                best_weight = chunk.weight
                best_idx = idx
                best_pivot_id = chunk.chunk_id
                best_pred_ids = preds_before[-j:] if j > 0 else []

        if best_idx is not None and best_pivot_id is not None:
            W[j] = best_weight
            prov[j] = (best_pivot_id, best_pred_ids, best_weight)

    d_total = min(k, sum(1 for chunk in chunks if chunk.d_total > 0))
    return W, prov, d_total


def test_l2_scan_matches_bruteforce_frontier_and_provenance() -> None:
    rng = random.Random(12345)

    for _ in range(120):
        n = rng.randint(1, 30)
        k = rng.randint(0, 5)
        chunks = _random_chunks(rng, n)

        summary = l2_scan(chunks, k)
        exp_W, exp_prov, exp_d = _bruteforce(chunks, k)

        assert summary.d_total == exp_d
        assert summary.W == exp_W

        for j in range(k + 1):
            got = summary.provenance[j]
            expected = exp_prov[j]
            if expected is None:
                assert got is None
                continue

            assert got is not None
            exp_pivot, exp_preds, exp_weight = expected
            assert got.pivot_id == exp_pivot
            assert got.pred_ids == exp_preds
            assert got.weight == exp_weight


def test_protected_set_matches_bruteforce_k_slot() -> None:
    rng = random.Random(54321)

    for _ in range(80):
        n = rng.randint(1, 25)
        k = rng.randint(0, 5)
        chunks = _random_chunks(rng, n)

        _, exp_prov, _ = _bruteforce(chunks, k)
        expected = exp_prov[k]

        protected, feasible = protected_set(chunks, k)

        if expected is None:
            assert feasible is False
            assert protected == set()
            continue

        pivot_id, pred_ids, _ = expected
        assert feasible is True
        assert protected == {pivot_id, *pred_ids}
