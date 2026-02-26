from __future__ import annotations

from compactor import token_count
from server import compact, compact_auto, inspect, inspect_horizon, messages_to_chunks, retention_floor, tag


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
    assert all("confidence" in entry for entry in out)


def test_inspect_tool_reports_feasible_k_arc() -> None:
    out = inspect(MESSAGES, k=1)
    assert out["feasible"] is True
    assert out["pivot_id"] == "m2"
    assert "m1" in out["protected_ids"]
    assert all("confidence" in chunk for chunk in out["tagged_chunks"])


def test_compact_l2_guarded_preserves_protected_ids_under_budget() -> None:
    budget = token_count(MESSAGES[0]["text"]) + token_count(MESSAGES[1]["text"])
    out = compact(MESSAGES, token_budget=budget, policy="l2_guarded", k=1)

    assert "error" not in out
    kept_ids = {msg["id"] for msg in out["messages"]}
    assert {"m1", "m2"}.issubset(kept_ids)
    assert out["audit"]["policy"] == "l2_guarded"
    assert out["audit"]["feasible"] is True
    assert out["audit"]["contract_satisfied"] is True
    assert out["audit"]["protection_satisfied"] is True
    assert out["audit"]["guard_effective"] is True
    assert out["audit"]["guard_reason"] == "active"


def test_compact_recency_policy_and_invalid_policy_handling() -> None:
    budget = token_count(MESSAGES[2]["text"]) + token_count(MESSAGES[3]["text"])
    out = compact(MESSAGES, token_budget=budget, policy="recency", k=1)

    assert "error" not in out
    assert out["audit"]["policy"] == "recency"
    assert out["audit"]["tokens_after"] <= budget
    assert out["audit"]["guard_effective"] is None
    assert out["audit"]["guard_reason"] == "not_applicable"

    bad = compact(MESSAGES, token_budget=budget, policy="bad-policy", k=1)
    assert "error" in bad


def test_role_hint_override_is_respected_through_server_path() -> None:
    messages = [
        {
            "id": "a",
            "text": "Please implement the deployment pipeline right now.",
            "role_hint": "predecessor",
        },
        {
            "id": "b",
            "text": "Objective: create endpoint hardening plan.",
            "role_hint": "pivot",
        },
    ]

    out = inspect(messages, k=1)
    assert out["feasible"] is True
    assert out["pivot_id"] == "b"
    assert out["protected_ids"] == ["a", "b"]


def test_guard_effective_requires_feasible_k_slot() -> None:
    messages = [
        {"id": "n1", "text": "neutral status update", "role_hint": "noise"},
        {"id": "n2", "text": "another neutral note", "role_hint": "noise"},
    ]

    out = compact(messages, token_budget=100, policy="l2_guarded", k=3)
    assert "error" not in out
    assert out["audit"]["feasible"] is False
    assert out["audit"]["contract_satisfied"] is True
    assert out["audit"]["protection_satisfied"] is True
    assert out["audit"]["guard_effective"] is False
    assert out["audit"]["guard_reason"] == "infeasible_k_slot"


def test_inspect_horizon_reports_feasible_slots() -> None:
    out = inspect_horizon(MESSAGES, k_max=3)
    assert out["k_max"] == 3
    assert out["k_max_feasible"] == 1
    assert out["feasible_slots"] == [0, 1]
    assert len(out["slots"]) == 4


def test_compact_auto_adaptive_selects_highest_feasible_k() -> None:
    budget = token_count(MESSAGES[0]["text"]) + token_count(MESSAGES[1]["text"])
    out = compact_auto(MESSAGES, token_budget=budget, k_target=3, mode="adaptive")
    assert "error" not in out
    assert out["audit"]["policy_selected"] == "l2_guarded"
    assert out["audit"]["k_selected"] == 1
    assert out["audit"]["feasible_target"] is False
    kept_ids = {msg["id"] for msg in out["messages"]}
    assert {"m1", "m2"}.issubset(kept_ids)


def test_compact_auto_budget_aware_k_selection() -> None:
    messages = [
        {"id": "p1", "text": "Constraint: API contract and backward compatibility.", "role_hint": "predecessor"},
        {"id": "p2", "text": "Constraint: keep audit log table.", "role_hint": "predecessor"},
        {"id": "p3", "text": "Constraint: do not store plaintext passwords.", "role_hint": "predecessor"},
        {"id": "pv", "text": "Please build the service now with all constraints.", "role_hint": "pivot"},
        {"id": "n1", "text": "filler noise", "role_hint": "noise"},
    ]
    # This budget can usually carry k=1 protected floor but not k=3 floor.
    budget = token_count(messages[2]["text"]) + token_count(messages[3]["text"]) + token_count(messages[4]["text"])
    out = compact_auto(messages, token_budget=budget, k_target=3, mode="adaptive")
    assert "error" not in out
    assert out["audit"]["k_selected"] in (0, 1, 2, 3)
    assert out["audit"]["policy_selected"] == "l2_guarded"
    assert out["audit"]["feasible_slots"] == [0, 1, 2, 3]


def test_compact_auto_strict_keeps_context_on_infeasible_target() -> None:
    out = compact_auto(MESSAGES, token_budget=20, k_target=3, mode="strict")
    assert "error" not in out
    assert out["audit"]["policy_selected"] == "none"
    assert out["audit"]["guard_reason"] == "strict_infeasible_target"
    assert out["messages"] == MESSAGES


def test_compact_iterative_guarded_reports_iterative_audit_fields() -> None:
    messages = [
        {"id": "c1", "text": "Must keep API contract", "role_hint": "predecessor"},
        {"id": "c2", "text": "Must keep audit logs", "role_hint": "predecessor"},
        {"id": "c3", "text": "Must add regression tests", "role_hint": "predecessor"},
        {"id": "pivot", "text": "Please implement the build now", "role_hint": "pivot"},
        {"id": "noise1", "text": "filler status note", "role_hint": "noise"},
        {"id": "noise2", "text": "filler parking note", "role_hint": "noise"},
    ]
    budget = token_count(messages[0]["text"]) + token_count(messages[1]["text"]) + token_count(messages[2]["text"]) + token_count(messages[3]["text"])
    out = compact(messages, token_budget=budget, policy="l2_iterative_guarded", k=3)
    assert "error" not in out
    assert out["audit"]["policy"] == "l2_iterative_guarded"
    assert "iterative_checked" in out["audit"]
    assert "iterative_accepted" in out["audit"]
    assert "iterative_blocked" in out["audit"]
    assert out["audit"]["guard_reason"] in {"active", "protected_breach_due_to_budget"}


def test_l2_guarded_reports_feasible_before_and_after() -> None:
    messages = [
        {"id": "c1", "text": "Must keep API contract", "role_hint": "predecessor"},
        {"id": "c2", "text": "Must keep audit logs", "role_hint": "predecessor"},
        {"id": "c3", "text": "Must add regression tests", "role_hint": "predecessor"},
        {"id": "pivot", "text": "Please implement the build now", "role_hint": "pivot"},
    ]
    out = compact(messages, token_budget=15, policy="l2_guarded", k=3)
    assert "error" not in out
    assert out["audit"]["feasible_before"] is True
    assert out["audit"]["feasible_after"] is False
    assert out["audit"]["feasible"] is False
    assert out["audit"]["guard_reason"] == "protected_breach_due_to_budget"


def test_retention_floor_returns_model_outputs() -> None:
    out = retention_floor(MESSAGES, k=1, horizon=50, failure_prob=0.05)
    assert "error" not in out
    assert out["required_predecessor_count"] >= 0
    assert out["feasible_now"] in {True, False}
    assert "count_retention_floor_clipped" in out
    assert "model_note" in out
