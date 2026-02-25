from __future__ import annotations

from compactor import token_count
from server import compact, inspect, messages_to_chunks, tag


MESSAGES = [
    {"id": "m1", "text": "Constraint: must preserve API contract."},
    {"id": "m2", "text": "Please implement the tropical compactor MCP server."},
    {"id": "m3", "text": "Traceback: assertion failure in CI."},
    {"id": "m4", "text": "Additional neutral context."},
]


def test_messages_to_chunks_normalizes_content_and_ids() -> None:
    chunks = messages_to_chunks(
        [
            {
                "content": [
                    {"type": "text", "text": "Please implement this."},
                    {"type": "text", "text": "Constraint: do not break API."},
                ]
            }
        ]
    )

    assert chunks[0]["id"] == "msg_0"
    assert "Please implement this." in chunks[0]["text"]


def test_tag_tool_returns_role_annotations() -> None:
    out = tag(MESSAGES)
    assert isinstance(out, list)
    assert len(out) == len(MESSAGES)
    roles = {entry["id"]: entry["role"] for entry in out}
    assert roles["m2"] == "pivot"


def test_inspect_tool_reports_feasible_k_arc() -> None:
    out = inspect(MESSAGES, k=1)
    assert out["feasible"] is True
    assert out["pivot_id"] == "m2"
    assert "m1" in out["protected_ids"]


def test_compact_l2_guarded_preserves_protected_ids_under_budget() -> None:
    budget = token_count(MESSAGES[0]["text"]) + token_count(MESSAGES[1]["text"])
    out = compact(MESSAGES, token_budget=budget, policy="l2_guarded", k=1)

    assert "error" not in out
    kept_ids = {msg["id"] for msg in out["messages"]}
    assert {"m1", "m2"}.issubset(kept_ids)
    assert out["audit"]["policy"] == "l2_guarded"


def test_compact_recency_policy_and_invalid_policy_handling() -> None:
    budget = token_count(MESSAGES[2]["text"]) + token_count(MESSAGES[3]["text"])
    out = compact(MESSAGES, token_budget=budget, policy="recency", k=1)

    assert "error" not in out
    assert out["audit"]["policy"] == "recency"
    assert out["audit"]["tokens_after"] <= budget

    bad = compact(MESSAGES, token_budget=budget, policy="bad-policy", k=1)
    assert "error" in bad
