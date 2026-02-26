#!/usr/bin/env python3
"""Comprehensive functional validation for tropical-compactor."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from server import compact, compact_auto, inspect, inspect_horizon, retention_floor


ROOT = Path(__file__).resolve().parents[1]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fixture_k3() -> list[dict[str, str]]:
    return [
        {"id": "n1", "role": "user", "text": "n01 casual note", "role_hint": "noise"},
        {"id": "n2", "role": "assistant", "text": "n02 casual response", "role_hint": "noise"},
        {"id": "c1", "role": "user", "text": "Must use SQLite only.", "role_hint": "predecessor"},
        {"id": "c2", "role": "user", "text": "Must keep /api/v1 routes.", "role_hint": "predecessor"},
        {"id": "c3", "role": "user", "text": "Must include audit_log table.", "role_hint": "predecessor"},
        {"id": "c4", "role": "user", "text": "Must not store plaintext passwords.", "role_hint": "predecessor"},
        {
            "id": "c5",
            "role": "user",
            "text": "Must include regression tests for login and refresh.",
            "role_hint": "predecessor",
        },
        {
            "id": "pivot",
            "role": "assistant",
            "text": "Understood all constraints. I will now build the app step by step.",
            "role_hint": "pivot",
        },
        {"id": "f1", "role": "user", "text": "n03 filler", "role_hint": "noise"},
        {"id": "f2", "role": "user", "text": "n04 filler", "role_hint": "noise"},
        {"id": "f3", "role": "user", "text": "n05 filler", "role_hint": "noise"},
        {"id": "qa", "role": "user", "text": "Run final QA now.", "role_hint": "noise"},
        {
            "id": "summary",
            "role": "assistant",
            "text": "Final QA complete. All constraints verified.",
            "role_hint": "noise",
        },
    ]


def _run_policy_comparison() -> dict[str, object]:
    messages = _fixture_k3()
    horizon = inspect_horizon(messages, k_max=5)
    _assert("error" not in horizon, f"inspect_horizon failed: {horizon}")
    _assert(horizon["k_max_feasible"] is not None, "k horizon should be feasible for fixture")

    inspect_k3 = inspect(messages, k=3)
    _assert(inspect_k3["feasible"] is True, "fixture should be feasible at k=3")
    _assert(inspect_k3["pivot_id"] == "pivot", "pivot id mismatch")

    budgets = [79, 60, 35]
    rows: list[dict[str, object]] = []
    for budget in budgets:
        rec = compact(messages, token_budget=budget, policy="recency", k=3)
        l2 = compact(messages, token_budget=budget, policy="l2_guarded", k=3)
        l2_iter = compact(messages, token_budget=budget, policy="l2_iterative_guarded", k=3)

        _assert("error" not in rec, f"recency failed at {budget}: {rec}")
        _assert("error" not in l2, f"l2_guarded failed at {budget}: {l2}")
        _assert("error" not in l2_iter, f"l2_iterative_guarded failed at {budget}: {l2_iter}")

        rec_ids = {m["id"] for m in rec["messages"]}
        l2_ids = {m["id"] for m in l2["messages"]}
        l2_iter_ids = {m["id"] for m in l2_iter["messages"]}

        _assert(l2["audit"]["tokens_after"] <= budget, "l2 budget overflow")
        _assert(l2_iter["audit"]["tokens_after"] <= budget, "l2_iter budget overflow")
        _assert(l2_iter["audit"]["iterative_checked"] >= 0, "iterative audit missing")
        _assert(l2_iter["audit"]["policy"] == "l2_iterative_guarded", "iterative policy mismatch")

        if budget <= 60:
            _assert("pivot" in l2_ids, "l2_guarded should keep pivot in tight budget")
            _assert("pivot" in l2_iter_ids, "l2_iterative_guarded should keep pivot in tight budget")
        if budget <= 35:
            _assert("pivot" not in rec_ids, "recency should drop pivot in tight budget")

        rows.append(
            {
                "budget": budget,
                "recency_kept": sorted(rec_ids),
                "l2_guarded_kept": sorted(l2_ids),
                "l2_iterative_kept": sorted(l2_iter_ids),
                "l2_guard_effective": l2["audit"]["guard_effective"],
                "l2_iter_guard_effective": l2_iter["audit"]["guard_effective"],
                "l2_iter_checked": l2_iter["audit"]["iterative_checked"],
                "l2_iter_blocked": l2_iter["audit"]["iterative_blocked"],
            }
        )

    return {"horizon": horizon, "inspect_k3": inspect_k3, "rows": rows}


def _run_auto_and_floor_checks() -> dict[str, object]:
    messages = _fixture_k3()
    auto = compact_auto(messages, token_budget=120, k_target=5, mode="adaptive")
    _assert("error" not in auto, f"compact_auto failed: {auto}")
    _assert(auto["audit"]["policy_selected"] == "l2_guarded", "auto should select l2_guarded")
    _assert(auto["audit"]["k_selected"] is not None, "auto should select a k")

    strict = compact_auto(messages, token_budget=60, k_target=5, mode="strict")
    _assert("error" not in strict, f"compact_auto strict failed: {strict}")
    _assert(strict["audit"]["policy_selected"] in {"none", "l2_guarded"}, "unexpected strict policy")

    floor = retention_floor(messages, k=3, horizon=100, failure_prob=0.01)
    _assert("error" not in floor, f"retention_floor failed: {floor}")
    _assert(floor["required_predecessor_count"] >= 0, "invalid predecessor floor")
    _assert("model_note" in floor, "missing floor note")

    return {"auto": auto["audit"], "strict": strict["audit"], "floor": floor}


def _stdio_smoke() -> dict[str, object]:
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)
    alive = proc.poll() is None
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    _assert(alive, "stdio server did not stay alive during smoke window")
    return {"alive_for_1s": alive, "returncode": proc.returncode}


def main() -> None:
    report = {
        "policy_comparison": _run_policy_comparison(),
        "auto_and_floor": _run_auto_and_floor_checks(),
        "stdio_smoke": _stdio_smoke(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
