"""tropical-compactor MCP server (stdio transport)."""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from algebra import ChunkState, L2Summary, l2_scan
from compactor import evict_l2_guarded, evict_recency, token_count
from tagger import tag_chunk_detailed


# IMPORTANT: stdout is reserved for MCP JSON-RPC transport.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tropical-compactor")

mcp = FastMCP("tropical-compactor")


def _message_text(msg: dict[str, Any]) -> str:
    text = msg.get("text")
    if isinstance(text, str):
        return text

    content = msg.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str):
                parts.append(part_text)
                continue
            fallback = part.get("content")
            if isinstance(fallback, str):
                parts.append(fallback)
        return "\n".join(parts)

    return ""


def messages_to_chunks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize incoming messages into compactable chunk records."""

    if not isinstance(messages, list):
        raise ValueError("messages must be a list of objects")

    total = len(messages)
    chunks: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be an object")

        chunk_id = str(msg.get("id") or f"msg_{i}")
        text = _message_text(msg)
        speaker_value = msg.get("role")
        speaker = speaker_value if isinstance(speaker_value, str) else None

        role_hint = msg.get("role_hint")
        role_hint_value = role_hint if isinstance(role_hint, str) else None
        tagged = tag_chunk_detailed(
            text=text,
            role_hint=role_hint_value,
            speaker=speaker,
            index=i,
            total=total,
        )

        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "role": tagged.role,
                "weight": tagged.weight,
                "confidence": tagged.confidence,
                "reasons": tagged.reasons,
                "speaker": speaker,
                "token_count": token_count(text),
                "original": msg,
            }
        )

    return chunks


def _semantic_reorder(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder chunks so predecessors precede the pivot for L2 scan.

    The L2 algebra requires predecessors *before* the pivot in scan order
    (W[j] lifts when an incoming pivot consumes j left-predecessors).  In real
    conversations the user often states the task first and adds constraints
    later, so predecessors appear *after* the pivot chronologically.

    Strategy: find the highest-weight pivot.  Any predecessors that appear
    *after* it in the original sequence are relocated to just before it.
    Predecessors already before the pivot stay in place.  This handles the
    conversation pattern (pivot first, constraints later) without breaking
    the specification pattern (constraints listed, then pivot) or multi-arc
    benchmarks where each arc has its own predecessor→pivot ordering.
    """

    # Find highest-weight pivot index.
    best_pivot_idx: int | None = None
    best_weight = -float("inf")
    for i, c in enumerate(chunks):
        if c["role"] == "pivot" and c["weight"] > best_weight:
            best_weight = c["weight"]
            best_pivot_idx = i

    if best_pivot_idx is None:
        return chunks  # No pivot found; nothing to reorder.

    # Find the last pivot position in the original sequence.
    last_pivot_idx = max(i for i, c in enumerate(chunks) if c["role"] == "pivot")

    # Only move predecessors that appear AFTER the best pivot AND have no
    # pivot after them in the original sequence.  A predecessor with a pivot
    # to its right is already "serving" that later pivot and should stay put.
    post_preds: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        if (
            i > best_pivot_idx
            and c["role"] == "predecessor"
            and i > last_pivot_idx
        ):
            post_preds.append(c)
        else:
            remaining.append(c)

    if not post_preds:
        return chunks  # All predecessors already before a pivot; no change needed.

    # Insert the orphaned predecessors just before the best pivot.
    pivot_pos = next(i for i, c in enumerate(remaining) if c is chunks[best_pivot_idx])
    result = remaining[:pivot_pos] + post_preds + remaining[pivot_pos:]
    return result


def chunks_to_algebra_states(chunks: list[dict[str, Any]], k: int) -> list[ChunkState]:
    """Map normalized chunks into L2 `ChunkState` values.

    Applies semantic reordering so that predecessors precede the pivot in the
    scan, regardless of their chronological position in the conversation.
    """

    ordered = _semantic_reorder(chunks)
    return [
        ChunkState(
            chunk_id=str(chunk["id"]),
            weight=float(chunk["weight"]),
            d_total=1 if chunk["role"] == "predecessor" else 0,
            text=str(chunk["text"]),
        )
        for chunk in ordered
    ]


def _protection_from_summary(summary: L2Summary, k: int) -> tuple[set[str], list[str], bool]:
    prov = summary.provenance[k]
    if prov is None:
        return set(), [], False

    protected = {prov.pivot_id, *prov.pred_ids}
    # Breach mode should preserve pivot first, then closest predecessors.
    priority = [prov.pivot_id, *reversed(prov.pred_ids)]
    return protected, priority, True


def _horizon_from_summary(
    summary: L2Summary,
    token_by_id: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    slots: list[dict[str, Any]] = []
    feasible_slots: list[int] = []

    for k in range(summary.k + 1):
        prov = summary.provenance[k]
        feasible = prov is not None
        if feasible:
            feasible_slots.append(k)
        protected_ids = sorted({prov.pivot_id, *prov.pred_ids}) if prov is not None else []
        protected_token_floor = None
        if protected_ids and token_by_id is not None:
            protected_token_floor = int(sum(token_by_id.get(cid, 0) for cid in protected_ids))

        slots.append(
            {
                "k": k,
                "feasible": feasible,
                "pivot_id": (prov.pivot_id if prov is not None else None),
                "predecessor_ids": (prov.pred_ids if prov is not None else []),
                "protected_ids": protected_ids,
                "protected_token_floor": protected_token_floor,
                "W": (round(summary.W[k], 4) if math.isfinite(summary.W[k]) else None),
            }
        )

    k_max_feasible = max(feasible_slots) if feasible_slots else None
    return slots, k_max_feasible


def _invalid(message: str) -> dict[str, str]:
    return {"error": message}


def _select_auto_k(slots: list[dict[str, Any]], k_target: int, token_budget: int) -> tuple[int | None, bool]:
    feasible_slots = [slot for slot in slots if slot["feasible"]]
    feasible_target = any(slot["k"] == k_target for slot in feasible_slots)
    if feasible_target:
        return k_target, True

    within_budget = []
    for slot in feasible_slots:
        floor = slot.get("protected_token_floor")
        if isinstance(floor, int) and floor <= token_budget:
            within_budget.append(slot)

    if within_budget:
        return int(max(slot["k"] for slot in within_budget)), False
    if feasible_slots:
        return int(max(slot["k"] for slot in feasible_slots)), False
    return None, False


def _compact_iterative_guarded(
    chunks: list[dict[str, Any]],
    token_budget: int,
    k: int,
    preserve_pivot: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokens_before = sum(int(chunk["token_count"]) for chunk in chunks)
    states = chunks_to_algebra_states(chunks, k)
    summary = l2_scan(states, k)
    protected_ids, protected_priority, feasible = _protection_from_summary(summary, k)
    baseline_prov = summary.provenance[k]
    baseline_weight = summary.W[k]
    baseline_pivot = baseline_prov.pivot_id if baseline_prov is not None else None

    if not feasible:
        kept, audit = evict_recency(chunks, token_budget)
        return kept, {
            "policy": "l2_iterative_guarded",
            "k": k,
            "protected_ids": [],
            "breach_ids": [],
            "dropped_ids": audit["dropped_ids"],
            "tokens_kept": int(audit["tokens_kept"]),
            "contract_satisfied": True,
            "protection_satisfied": True,
            "feasible_before": False,
            "feasible_after": False,
            "feasible": False,
            "guard_effective": False,
            "guard_reason": "infeasible_k_slot",
            "tokens_before": tokens_before,
            "tokens_after": int(audit["tokens_kept"]),
            "iterative_checked": 0,
            "iterative_accepted": 0,
            "iterative_blocked": 0,
            "iterative_preserve_pivot": preserve_pivot,
            "baseline_pivot_id": None,
            "baseline_W_k": None,
            "final_W_k": None,
            "iterative_fallback_breach": False,
        }

    current = list(chunks)
    current_tokens = tokens_before
    candidate_ids = [str(chunk["id"]) for chunk in current if str(chunk["id"]) not in protected_ids]

    checked = 0
    accepted = 0
    blocked = 0

    for cid in candidate_ids:
        if current_tokens <= token_budget:
            break

        idx = next((i for i, chunk in enumerate(current) if str(chunk["id"]) == cid), None)
        if idx is None:
            continue

        trial = current[:idx] + current[idx + 1 :]
        checked += 1
        trial_summary = l2_scan(chunks_to_algebra_states(trial, k), k)
        trial_prov = trial_summary.provenance[k]

        allowed = False
        if trial_prov is not None and trial_summary.W[k] + 1e-12 >= baseline_weight:
            if not preserve_pivot or trial_prov.pivot_id == baseline_pivot:
                allowed = True

        if allowed:
            removed_tokens = int(current[idx]["token_count"])
            current = trial
            current_tokens -= removed_tokens
            accepted += 1
        else:
            blocked += 1

    fallback_breach = False
    if current_tokens > token_budget:
        current_summary = l2_scan(chunks_to_algebra_states(current, k), k)
        curr_protected, curr_priority, _ = _protection_from_summary(current_summary, k)
        kept, breach_audit = evict_l2_guarded(
            current,
            token_budget,
            curr_protected,
            k,
            protected_priority=curr_priority,
        )
        fallback_breach = True
        kept_ids = {str(chunk["id"]) for chunk in kept}
    else:
        kept = current
        kept_ids = {str(chunk["id"]) for chunk in kept}
        breach_audit = {
            "dropped_ids": [str(chunk["id"]) for chunk in chunks if str(chunk["id"]) not in kept_ids],
            "tokens_kept": current_tokens,
        }

    dropped_ids = [str(chunk["id"]) for chunk in chunks if str(chunk["id"]) not in kept_ids]
    breach_ids = sorted(protected_ids - kept_ids)
    protection_satisfied = len(breach_ids) == 0

    final_summary = l2_scan(chunks_to_algebra_states(kept, k), k)
    final_prov = final_summary.provenance[k]
    final_feasible = final_prov is not None
    guard_effective = bool(final_feasible and protection_satisfied)

    if not feasible:
        guard_reason = "infeasible_k_slot"
    elif not protection_satisfied:
        guard_reason = "protected_breach_due_to_budget"
    elif not final_feasible:
        guard_reason = "post_compaction_infeasible"
    else:
        guard_reason = "active"

    audit = {
        "policy": "l2_iterative_guarded",
        "k": k,
        "protected_ids": sorted(protected_ids),
        "breach_ids": breach_ids,
        "dropped_ids": dropped_ids,
        "tokens_kept": int(breach_audit["tokens_kept"]),
        "contract_satisfied": protection_satisfied,
        "protection_satisfied": protection_satisfied,
        "feasible_before": feasible,
        "feasible_after": final_feasible,
        "feasible": final_feasible,
        "guard_effective": guard_effective,
        "guard_reason": guard_reason,
        "tokens_before": tokens_before,
        "tokens_after": int(breach_audit["tokens_kept"]),
        "iterative_checked": checked,
        "iterative_accepted": accepted,
        "iterative_blocked": blocked,
        "iterative_preserve_pivot": preserve_pivot,
        "baseline_pivot_id": baseline_pivot,
        "baseline_W_k": (round(baseline_weight, 4) if math.isfinite(baseline_weight) else None),
        "final_W_k": (
            round(final_summary.W[k], 4) if math.isfinite(final_summary.W[k]) else None
        ),
        "iterative_fallback_breach": fallback_breach,
    }
    return kept, audit


@mcp.tool()
def compact(
    messages: list[dict[str, Any]],
    token_budget: int = 4000,
    policy: str = "l2_guarded",
    k: int = 3,
    preserve_pivot: bool = True,
) -> dict[str, Any]:
    """
    Compact conversation messages to a token budget.

    policies:
    - l2_guarded: protect pivot + k predecessors from L2 scan.
    - l2_iterative_guarded: iterative removal that preserves W[k] and (optionally) pivot identity.
    - recency: keep newest chunks only.

    Audit semantics:
    - contract_satisfied/protection_satisfied: no protected chunks were dropped.
    - feasible: L2 witness exists at slot k (W[k] finite).
    - guard_effective: both feasible and protection_satisfied are true.
    """

    if token_budget < 0:
        return _invalid("token_budget must be >= 0")
    if k < 0:
        return _invalid("k must be >= 0")

    try:
        chunks = messages_to_chunks(messages)
    except ValueError as exc:
        return _invalid(str(exc))

    tokens_before = sum(chunk["token_count"] for chunk in chunks)

    if policy == "l2_guarded":
        states = chunks_to_algebra_states(chunks, k)
        summary = l2_scan(states, k)
        prot_ids, protected_priority, feasible_before = _protection_from_summary(summary, k)
        kept, audit = evict_l2_guarded(
            chunks,
            token_budget,
            prot_ids,
            k,
            protected_priority=protected_priority,
        )
        final_summary = l2_scan(chunks_to_algebra_states(kept, k), k)
        feasible_after = final_summary.provenance[k] is not None
        protection_satisfied = bool(audit.get("protection_satisfied"))
        guard_effective = bool(feasible_before and feasible_after and protection_satisfied)

        audit["feasible_before"] = feasible_before
        audit["feasible_after"] = feasible_after
        audit["feasible"] = feasible_after
        audit["guard_effective"] = guard_effective
        if not feasible_before:
            audit["guard_reason"] = "infeasible_k_slot"
        elif not protection_satisfied:
            audit["guard_reason"] = "protected_breach_due_to_budget"
        elif not feasible_after:
            audit["guard_reason"] = "post_compaction_infeasible"
        else:
            audit["guard_reason"] = "active"
        audit["tokens_before"] = tokens_before
        audit["tokens_after"] = int(audit.get("tokens_kept", 0))
    elif policy == "l2_iterative_guarded":
        kept, audit = _compact_iterative_guarded(
            chunks=chunks,
            token_budget=token_budget,
            k=k,
            preserve_pivot=preserve_pivot,
        )
    elif policy == "recency":
        kept, audit = evict_recency(chunks, token_budget)
        audit["feasible"] = None
        audit["guard_effective"] = None
        audit["guard_reason"] = "not_applicable"
        audit["tokens_before"] = tokens_before
        audit["tokens_after"] = int(audit.get("tokens_kept", 0))
        audit["protected_ids"] = []
        audit["breach_ids"] = []
        audit["contract_satisfied"] = None
        audit["protection_satisfied"] = None
    else:
        return _invalid(
            f"Unknown policy '{policy}'. Use 'l2_guarded', 'l2_iterative_guarded', or 'recency'."
        )

    surviving_originals = [chunk["original"] for chunk in kept]
    return {"messages": surviving_originals, "audit": audit}


@mcp.tool()
def inspect(messages: list[dict[str, Any]], k: int = 3) -> dict[str, Any]:
    """Inspect L2 frontier feasibility and protected chunks without compacting."""

    if k < 0:
        return _invalid("k must be >= 0")

    try:
        chunks = messages_to_chunks(messages)
    except ValueError as exc:
        return _invalid(str(exc))

    states = chunks_to_algebra_states(chunks, k)
    summary = l2_scan(states, k)
    prot_ids, _, feasible = _protection_from_summary(summary, k)
    prov = summary.provenance[k]

    W_display = [round(w, 4) if math.isfinite(w) else None for w in summary.W]

    return {
        "feasible": feasible,
        "protected_ids": sorted(prot_ids),
        "pivot_id": (prov.pivot_id if prov is not None else None),
        "predecessor_ids": (prov.pred_ids if prov is not None else []),
        "W": W_display,
        "tagged_chunks": [
            {
                "id": chunk["id"],
                "role": chunk["role"],
                "weight": (round(chunk["weight"], 4) if math.isfinite(chunk["weight"]) else None),
                "confidence": round(float(chunk["confidence"]), 4),
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ],
    }


@mcp.tool()
def inspect_horizon(messages: list[dict[str, Any]], k_max: int = 8) -> dict[str, Any]:
    """Inspect feasibility across all k-slots from 0..k_max."""

    if k_max < 0:
        return _invalid("k_max must be >= 0")

    try:
        chunks = messages_to_chunks(messages)
    except ValueError as exc:
        return _invalid(str(exc))

    states = chunks_to_algebra_states(chunks, k_max)
    summary = l2_scan(states, k_max)
    token_by_id = {str(chunk["id"]): int(chunk["token_count"]) for chunk in chunks}
    slots, k_max_feasible = _horizon_from_summary(summary, token_by_id=token_by_id)

    return {
        "k_max": k_max,
        "k_max_feasible": k_max_feasible,
        "feasible_slots": [slot["k"] for slot in slots if slot["feasible"]],
        "slots": slots,
        "W": [round(w, 4) if math.isfinite(w) else None for w in summary.W],
        "tagged_chunks": [
            {
                "id": chunk["id"],
                "role": chunk["role"],
                "weight": (round(chunk["weight"], 4) if math.isfinite(chunk["weight"]) else None),
                "confidence": round(float(chunk["confidence"]), 4),
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ],
    }


@mcp.tool()
def compact_auto(
    messages: list[dict[str, Any]],
    token_budget: int = 4000,
    k_target: int = 3,
    mode: str = "adaptive",
    fallback_policy: str = "recency",
    preserve_pivot: bool = True,
) -> dict[str, Any]:
    """
    Auto-select compaction behavior based on feasible k-slots.

    mode:
    - adaptive: choose highest feasible k <= k_target; fallback when no feasible slots.
    - strict: keep context unchanged if k_target is infeasible.
    """

    if token_budget < 0:
        return _invalid("token_budget must be >= 0")
    if k_target < 0:
        return _invalid("k_target must be >= 0")
    if mode not in {"adaptive", "strict"}:
        return _invalid("mode must be 'adaptive' or 'strict'")
    if fallback_policy not in {"recency", "l2_guarded", "l2_iterative_guarded"}:
        return _invalid("fallback_policy must be 'recency', 'l2_guarded', or 'l2_iterative_guarded'")

    horizon = inspect_horizon(messages, k_max=k_target)
    if "error" in horizon:
        return horizon

    slots = list(horizon["slots"])
    feasible_slots = list(horizon["feasible_slots"])
    k_selected, feasible_target = _select_auto_k(slots=slots, k_target=k_target, token_budget=token_budget)

    if feasible_target:
        selected_policy = "l2_guarded"
        out = compact(
            messages,
            token_budget=token_budget,
            policy="l2_guarded",
            k=k_target,
            preserve_pivot=preserve_pivot,
        )
    elif mode == "adaptive" and k_selected is not None:
        selected_policy = "l2_guarded"
        out = compact(
            messages,
            token_budget=token_budget,
            policy="l2_guarded",
            k=int(k_selected),
            preserve_pivot=preserve_pivot,
        )
    elif mode == "adaptive":
        selected_policy = fallback_policy
        fallback_k = int(k_selected) if k_selected is not None else 0
        out = compact(
            messages,
            token_budget=token_budget,
            policy=fallback_policy,
            k=fallback_k,
            preserve_pivot=preserve_pivot,
        )
    else:
        selected_policy = "none"
        chunks = messages_to_chunks(messages)
        tokens_before = sum(chunk["token_count"] for chunk in chunks)
        out = {
            "messages": messages,
            "audit": {
                "policy": "auto",
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "dropped_ids": [],
                "feasible": False,
                "contract_satisfied": None,
                "protection_satisfied": None,
                "guard_effective": False,
                "guard_reason": "strict_infeasible_target",
                "protected_ids": [],
                "breach_ids": [],
            },
        }

    if "error" in out:
        return out

    out_audit = out.get("audit", {})
    out_audit["policy_requested"] = "auto"
    out_audit["policy_selected"] = selected_policy
    out_audit["mode"] = mode
    out_audit["k_target"] = k_target
    out_audit["k_selected"] = k_selected
    out_audit["feasible_target"] = feasible_target
    out_audit["feasible_slots"] = feasible_slots
    out_audit["k_max_feasible"] = horizon["k_max_feasible"]
    out["audit"] = out_audit
    return out


@mcp.tool()
def retention_floor(
    messages: list[dict[str, Any]],
    k: int = 3,
    horizon: int = 100,
    failure_prob: float = 0.01,
) -> dict[str, Any]:
    """
    Estimate an operational retention floor using record-gap style approximations.

    The estimate uses:
      pi0_target = 1 - (1 - failure_prob)^(1 / horizon)
      required_mu_pre ~= k / pi0_target
    where mu_pre is expected retained predecessor count per epoch.
    """

    if k < 0:
        return _invalid("k must be >= 0")
    if horizon <= 0:
        return _invalid("horizon must be > 0")
    if failure_prob <= 0.0 or failure_prob >= 1.0:
        return _invalid("failure_prob must satisfy 0 < failure_prob < 1")

    try:
        chunks = messages_to_chunks(messages)
    except ValueError as exc:
        return _invalid(str(exc))

    states = chunks_to_algebra_states(chunks, k)
    summary = l2_scan(states, k)
    protected_ids, _, feasible_now = _protection_from_summary(summary, k)

    pred_chunks = [chunk for chunk in chunks if chunk["role"] == "predecessor"]
    pivot_chunks = [chunk for chunk in chunks if chunk["role"] == "pivot"]
    n_pred = len(pred_chunks)
    n_total = len(chunks)

    pi0_target = 1.0 - (1.0 - failure_prob) ** (1.0 / horizon)
    required_mu_pre = float(k / max(pi0_target, 1e-12))
    required_pred_count = int(math.ceil(required_mu_pre))
    count_retention_floor = (required_pred_count / n_pred) if n_pred > 0 else math.inf
    count_retention_floor_clipped = (
        min(1.0, count_retention_floor) if math.isfinite(count_retention_floor) else None
    )

    avg_pred_tokens = (
        sum(int(chunk["token_count"]) for chunk in pred_chunks) / n_pred if n_pred > 0 else 0.0
    )
    avg_pivot_tokens = (
        sum(int(chunk["token_count"]) for chunk in pivot_chunks) / len(pivot_chunks)
        if pivot_chunks
        else 0.0
    )

    approx_token_floor = None
    if math.isfinite(count_retention_floor):
        approx_token_floor = int(round(required_pred_count * avg_pred_tokens + avg_pivot_tokens))

    exact_contract_floor = None
    if feasible_now and protected_ids:
        token_by_id = {str(chunk["id"]): int(chunk["token_count"]) for chunk in chunks}
        exact_contract_floor = int(sum(token_by_id.get(cid, 0) for cid in protected_ids))

    feasible_by_counts = (n_pred >= required_pred_count) if n_pred > 0 else (required_pred_count <= 0)
    current_pi0_estimate = min(1.0, (k / n_pred)) if n_pred > 0 else 1.0
    current_horizon_failure_estimate = 1.0 - (1.0 - current_pi0_estimate) ** horizon

    return {
        "k": k,
        "horizon": horizon,
        "failure_prob_target": failure_prob,
        "pi0_target": pi0_target,
        "required_mu_predecessor": required_mu_pre,
        "required_predecessor_count": required_pred_count,
        "predecessor_count_observed": n_pred,
        "message_count": n_total,
        "predecessor_fraction": (n_pred / n_total if n_total > 0 else 0.0),
        "count_retention_floor": (count_retention_floor if math.isfinite(count_retention_floor) else None),
        "count_retention_floor_clipped": count_retention_floor_clipped,
        "approx_token_floor": approx_token_floor,
        "exact_contract_token_floor": exact_contract_floor,
        "feasible_now": feasible_now,
        "feasible_by_count_model": feasible_by_counts,
        "current_pi0_estimate": current_pi0_estimate,
        "current_horizon_failure_estimate": current_horizon_failure_estimate,
        "model_note": (
            "count_retention_floor is a count-based approximation from predecessor density; "
            "exact_contract_token_floor is exact for current tagged/provenanced context only."
        ),
    }


@mcp.tool()
def tag(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | dict[str, str]:
    """Tag messages by inferred role and pivot weight, without compacting."""

    try:
        chunks = messages_to_chunks(messages)
    except ValueError as exc:
        return _invalid(str(exc))

    return [
        {
            "id": chunk["id"],
            "role": chunk["role"],
            "weight": (round(chunk["weight"], 4) if math.isfinite(chunk["weight"]) else None),
            "confidence": round(float(chunk["confidence"]), 4),
            "reasons": chunk["reasons"],
            "token_count": chunk["token_count"],
            "text_preview": chunk["text"][:120],
        }
        for chunk in chunks
    ]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
