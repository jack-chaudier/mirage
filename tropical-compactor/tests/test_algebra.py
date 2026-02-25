from __future__ import annotations

import math

from algebra import ChunkState, compose, l2_scan, protected_set


def _pred(chunk_id: str) -> ChunkState:
    return ChunkState(chunk_id=chunk_id, weight=-math.inf, d_total=1, text="pred")


def _pivot(chunk_id: str, weight: float) -> ChunkState:
    return ChunkState(chunk_id=chunk_id, weight=weight, d_total=0, text="pivot")


def test_l2_scan_and_protected_set_returns_expected_witness() -> None:
    chunks = [
        _pred("a"),
        _pred("b"),
        _pivot("p1", 5.0),
        _pred("c"),
        _pred("d"),
        _pivot("p2", 10.0),
    ]

    summary = l2_scan(chunks, k=3)
    assert summary.W[0] == 10.0
    assert summary.W[3] == 10.0

    prov = summary.provenance[3]
    assert prov is not None
    assert prov.pivot_id == "p2"
    assert prov.pred_ids == ["b", "c", "d"]

    protected, feasible = protected_set(chunks, k=3)
    assert feasible is True
    assert protected == {"p2", "b", "c", "d"}


def test_protected_set_infeasible_when_no_pivot_exists() -> None:
    chunks = [_pred("a"), _pred("b"), _pred("c")]
    protected, feasible = protected_set(chunks, k=2)

    assert feasible is False
    assert protected == set()


def test_compose_associativity_on_weight_frontier() -> None:
    k = 3
    left = l2_scan([_pred("a1"), _pivot("p1", 1.0)], k)
    middle = l2_scan([_pred("b1"), _pred("b2"), _pivot("p2", 3.0)], k)
    right = l2_scan([_pred("c1"), _pivot("p3", 2.0)], k)

    ab_c = compose(compose(left, middle, k), right, k)
    a_bc = compose(left, compose(middle, right, k), k)

    assert ab_c.d_total == a_bc.d_total
    assert ab_c.W == a_bc.W

    for j in range(k + 1):
        p_left = ab_c.provenance[j]
        p_right = a_bc.provenance[j]
        if p_left is None or p_right is None:
            assert p_left is p_right
            continue
        assert p_left.pivot_id == p_right.pivot_id
        assert p_left.pred_ids == p_right.pred_ids
        assert p_left.weight == p_right.weight
