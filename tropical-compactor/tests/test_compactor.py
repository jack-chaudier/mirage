from __future__ import annotations

from compactor import evict_l2_guarded, evict_recency, token_count


def _chunk(chunk_id: str, text: str) -> dict:
    return {"id": chunk_id, "text": text, "token_count": token_count(text)}


def test_evict_recency_keeps_newest_chunks() -> None:
    chunks = [
        _chunk("1", "alpha alpha alpha alpha"),
        _chunk("2", "beta beta beta beta"),
        _chunk("3", "gamma gamma gamma gamma"),
    ]

    budget = chunks[1]["token_count"] + chunks[2]["token_count"]
    kept, audit = evict_recency(chunks, budget)

    assert [c["id"] for c in kept] == ["2", "3"]
    assert audit["policy"] == "recency"
    assert audit["tokens_kept"] <= budget
    assert audit["dropped_ids"] == ["1"]


def test_l2_guarded_keeps_protected_before_filler() -> None:
    chunks = [
        _chunk("1", "protected-one"),
        _chunk("2", "filler-old"),
        _chunk("3", "filler-new"),
    ]

    budget = chunks[0]["token_count"] + chunks[2]["token_count"]
    kept, audit = evict_l2_guarded(chunks, budget, protected_ids={"1"}, k=1)

    assert [c["id"] for c in kept] == ["1", "3"]
    assert audit["policy"] == "l2_guarded"
    assert audit["breach_ids"] == []
    assert audit["contract_satisfied"] is True
    assert audit["protection_satisfied"] is True


def test_l2_guarded_reports_breach_when_budget_too_small() -> None:
    chunks = [
        _chunk("1", "protected-one"),
        _chunk("2", "protected-two-longer"),
        _chunk("3", "filler"),
    ]

    budget = chunks[0]["token_count"]
    kept, audit = evict_l2_guarded(chunks, budget, protected_ids={"1", "2"}, k=2)

    assert [c["id"] for c in kept] == ["1"]
    assert audit["contract_satisfied"] is False
    assert audit["protection_satisfied"] is False
    assert "2" in audit["breach_ids"]
    assert "2" in audit["dropped_ids"]


def test_l2_guarded_breach_uses_protected_priority_order() -> None:
    chunks = [
        _chunk("pred_old", "old predecessor"),
        _chunk("pred_new", "new predecessor"),
        _chunk("pivot", "pivot-summary chunk with important objective"),
        _chunk("tail", "latest filler"),
    ]

    pivot_tokens = chunks[2]["token_count"]
    budget = pivot_tokens + chunks[3]["token_count"]
    kept, audit = evict_l2_guarded(
        chunks,
        budget,
        protected_ids={"pred_old", "pred_new", "pivot"},
        k=2,
        protected_priority=["pivot", "pred_new", "pred_old"],
    )

    kept_ids = [c["id"] for c in kept]
    assert "pivot" in kept_ids
    assert "pred_old" in audit["breach_ids"]
    assert audit["contract_satisfied"] is False
