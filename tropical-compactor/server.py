"""tropical-compactor MCP server (stdio transport)."""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from algebra import ChunkState, l2_scan, protected_set
from compactor import evict_l2_guarded, evict_recency, token_count
from tagger import tag_chunk


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

    chunks: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be an object")

        chunk_id = str(msg.get("id") or f"msg_{i}")
        text = _message_text(msg)

        role_hint = msg.get("role_hint")
        role_hint_value = role_hint if isinstance(role_hint, str) else None
        role, weight = tag_chunk(text, role_hint=role_hint_value)

        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "role": role,
                "weight": weight,
                "token_count": token_count(text),
                "original": msg,
            }
        )

    return chunks


def chunks_to_algebra_states(chunks: list[dict[str, Any]], k: int) -> list[ChunkState]:
    """Map normalized chunks into L2 `ChunkState` values."""

    return [
        ChunkState(
            chunk_id=str(chunk["id"]),
            weight=float(chunk["weight"]),
            d_total=1 if chunk["role"] == "predecessor" else 0,
            text=str(chunk["text"]),
        )
        for chunk in chunks
    ]


def _invalid(message: str) -> dict[str, str]:
    return {"error": message}


@mcp.tool()
def compact(
    messages: list[dict[str, Any]],
    token_budget: int = 4000,
    policy: str = "l2_guarded",
    k: int = 3,
) -> dict[str, Any]:
    """
    Compact conversation messages to a token budget.

    policies:
    - l2_guarded: protect pivot + k predecessors from L2 scan.
    - recency: keep newest chunks only.
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
        prot_ids, feasible = protected_set(states, k)
        kept, audit = evict_l2_guarded(chunks, token_budget, prot_ids, k)
        audit["feasible"] = feasible
        audit["tokens_before"] = tokens_before
        audit["tokens_after"] = int(audit.get("tokens_kept", 0))
    elif policy == "recency":
        kept, audit = evict_recency(chunks, token_budget)
        audit["feasible"] = None
        audit["tokens_before"] = tokens_before
        audit["tokens_after"] = int(audit.get("tokens_kept", 0))
        audit["protected_ids"] = []
        audit["breach_ids"] = []
        audit["contract_satisfied"] = None
    else:
        return _invalid(f"Unknown policy '{policy}'. Use 'l2_guarded' or 'recency'.")

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
    prot_ids, feasible = protected_set(states, k)

    prov = summary.provenance[k]
    pivot_id = prov.pivot_id if prov is not None else None
    pred_ids = prov.pred_ids if prov is not None else []

    W_display = [round(w, 4) if math.isfinite(w) else None for w in summary.W]

    return {
        "feasible": feasible,
        "protected_ids": sorted(prot_ids),
        "pivot_id": pivot_id,
        "predecessor_ids": pred_ids,
        "W": W_display,
        "tagged_chunks": [
            {
                "id": chunk["id"],
                "role": chunk["role"],
                "weight": (round(chunk["weight"], 4) if math.isfinite(chunk["weight"]) else None),
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ],
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
            "token_count": chunk["token_count"],
            "text_preview": chunk["text"][:120],
        }
        for chunk in chunks
    ]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
