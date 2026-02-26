"""Token-budgeted eviction policies."""

from __future__ import annotations

from typing import Any

import tiktoken


_ENC = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_ENC.encode(text))


def _chunk_tokens(chunk: dict[str, Any]) -> int:
    value = chunk.get("token_count")
    if isinstance(value, int) and value >= 0:
        return value
    return token_count(str(chunk.get("text", "")))


def evict_recency(chunks: list[dict[str, Any]], token_budget: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep most-recent chunks up to the token budget."""

    if token_budget < 0:
        raise ValueError("token_budget must be >= 0")

    kept: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    total = 0

    for chunk in reversed(chunks):
        cid = str(chunk["id"])
        t = _chunk_tokens(chunk)
        if total + t <= token_budget:
            kept.append(chunk)
            kept_ids.add(cid)
            total += t

    kept.reverse()
    dropped = [str(c["id"]) for c in chunks if str(c["id"]) not in kept_ids]

    return kept, {
        "policy": "recency",
        "dropped_ids": dropped,
        "tokens_kept": total,
    }


def evict_l2_guarded(
    chunks: list[dict[str, Any]],
    token_budget: int,
    protected_ids: set[str],
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Keep protected chunks first, then fill with recency-ranked non-protected chunks.

    Any dropped protected chunk is logged as a contract breach.
    """

    if token_budget < 0:
        raise ValueError("token_budget must be >= 0")

    protected = [c for c in chunks if str(c["id"]) in protected_ids]
    non_protected = [c for c in chunks if str(c["id"]) not in protected_ids]

    protected_tokens = sum(_chunk_tokens(c) for c in protected)
    breach: list[str] = []

    if protected_tokens > token_budget:
        kept_protected: list[dict[str, Any]] = []
        total = 0
        for chunk in protected:
            cid = str(chunk["id"])
            t = _chunk_tokens(chunk)
            if total + t <= token_budget:
                kept_protected.append(chunk)
                total += t
            else:
                breach.append(cid)
        protected = kept_protected
        protected_tokens = total

    remaining_budget = token_budget - protected_tokens
    filler_tokens = 0
    kept_non_protected: list[dict[str, Any]] = []

    for chunk in reversed(non_protected):
        t = _chunk_tokens(chunk)
        if filler_tokens + t <= remaining_budget:
            kept_non_protected.append(chunk)
            filler_tokens += t

    kept_non_protected.reverse()

    kept_ids = {str(c["id"]) for c in protected} | {str(c["id"]) for c in kept_non_protected}
    kept = [c for c in chunks if str(c["id"]) in kept_ids]
    dropped = [str(c["id"]) for c in chunks if str(c["id"]) not in kept_ids]

    return kept, {
        "policy": "l2_guarded",
        "k": k,
        "protected_ids": sorted(protected_ids),
        "breach_ids": breach,
        "dropped_ids": dropped,
        "tokens_kept": protected_tokens + filler_tokens,
        # Means protected IDs were not dropped by budgeting.
        "contract_satisfied": len(breach) == 0,
        # Alias kept for clarity in downstream audit consumers.
        "protection_satisfied": len(breach) == 0,
    }
