"""Replay harness for CyberOps-style compaction benchmarking via MCP tool functions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from compactor import token_count
from server import compact


DECOY_LINES = {"W1": 156, "W2": 162, "W3": 168, "P": 175}


@dataclass(frozen=True)
class VariantLayout:
    variant: str
    description: str
    primary_lines: dict[str, int]


DEFAULT_LAYOUTS = [
    VariantLayout(
        variant="base_30_110",
        description="Primary arc centered early (approx lines 30-110).",
        primary_lines={"W1": 32, "W2": 55, "W3": 78, "P": 110},
    ),
    VariantLayout(
        variant="shift_15_95",
        description="Primary arc shifted earlier.",
        primary_lines={"W1": 15, "W2": 40, "W3": 68, "P": 95},
    ),
    VariantLayout(
        variant="shift_50_130",
        description="Primary arc shifted later.",
        primary_lines={"W1": 50, "W2": 75, "W3": 102, "P": 130},
    ),
]


def make_nonce(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6].upper()


def _message_id(variant: str, line_no: int) -> str:
    return f"{variant}_L{line_no:03d}"


def _noise_text(line_no: int) -> str:
    return (
        f"L{line_no:03d} INFO routine telemetry update; "
        f"no root-cause assertion in this line."
    )


def build_variant_messages(layout: VariantLayout, line_count: int = 200) -> dict[str, Any]:
    variant = layout.variant
    primary = layout.primary_lines

    primary_nonces = {
        "ROOT_CAUSE": make_nonce(f"{variant}_primary_root"),
        "WARNING_1": make_nonce(f"{variant}_primary_w1"),
        "WARNING_2": make_nonce(f"{variant}_primary_w2"),
        "WARNING_3": make_nonce(f"{variant}_primary_w3"),
    }
    decoy_nonces = {
        "ROOT_CAUSE": make_nonce(f"{variant}_decoy_root"),
        "WARNING_1": make_nonce(f"{variant}_decoy_w1"),
        "WARNING_2": make_nonce(f"{variant}_decoy_w2"),
        "WARNING_3": make_nonce(f"{variant}_decoy_w3"),
    }

    special: dict[int, dict[str, str]] = {
        primary["W1"]: {
            "role_hint": "predecessor",
            "text": (
                f"L{primary['W1']:03d} WARN unusual credential spray detected "
                f"(verification code: {primary_nonces['WARNING_1']})."
            ),
        },
        primary["W2"]: {
            "role_hint": "predecessor",
            "text": (
                f"L{primary['W2']:03d} WARN lateral movement seen across file services "
                f"(verification code: {primary_nonces['WARNING_2']})."
            ),
        },
        primary["W3"]: {
            "role_hint": "predecessor",
            "text": (
                f"L{primary['W3']:03d} WARN staged exfil channels observed "
                f"(verification code: {primary_nonces['WARNING_3']})."
            ),
        },
        primary["P"]: {
            "role_hint": "pivot",
            "text": (
                f"L{primary['P']:03d} CRITICAL data breach declared; severity score 95; "
                f"2.1TB exfiltrated (verification code: {primary_nonces['ROOT_CAUSE']})."
            ),
        },
        DECOY_LINES["W1"]: {
            "role_hint": "predecessor",
            "text": (
                f"L{DECOY_LINES['W1']:03d} WARN CPU temperature spike on cache node "
                f"(verification code: {decoy_nonces['WARNING_1']})."
            ),
        },
        DECOY_LINES["W2"]: {
            "role_hint": "predecessor",
            "text": (
                f"L{DECOY_LINES['W2']:03d} WARN fan RPM dropping, thermal alert triggered "
                f"(verification code: {decoy_nonces['WARNING_2']})."
            ),
        },
        DECOY_LINES["W3"]: {
            "role_hint": "predecessor",
            "text": (
                f"L{DECOY_LINES['W3']:03d} WARN thermal throttling engaged on cache node "
                f"(verification code: {decoy_nonces['WARNING_3']})."
            ),
        },
        DECOY_LINES["P"]: {
            "role_hint": "pivot",
            "text": (
                f"L{DECOY_LINES['P']:03d} CRITICAL cache node overheated; severity score 38; "
                f"service degraded (verification code: {decoy_nonces['ROOT_CAUSE']})."
            ),
        },
    }

    messages: list[dict[str, str]] = []
    for line_no in range(1, line_count + 1):
        sid = _message_id(variant, line_no)
        if line_no in special:
            payload = special[line_no]
            messages.append(
                {
                    "id": sid,
                    "text": payload["text"],
                    "role_hint": payload["role_hint"],
                }
            )
        else:
            messages.append(
                {
                    "id": sid,
                    "text": _noise_text(line_no),
                    "role_hint": "noise",
                }
            )

    primary_ids = {
        _message_id(variant, primary["W1"]),
        _message_id(variant, primary["W2"]),
        _message_id(variant, primary["W3"]),
        _message_id(variant, primary["P"]),
    }
    decoy_ids = {
        _message_id(variant, DECOY_LINES["W1"]),
        _message_id(variant, DECOY_LINES["W2"]),
        _message_id(variant, DECOY_LINES["W3"]),
        _message_id(variant, DECOY_LINES["P"]),
    }

    return {
        "variant": variant,
        "description": layout.description,
        "messages": messages,
        "primary_ids": primary_ids,
        "decoy_ids": decoy_ids,
    }


def _parse_fractions(spec: str) -> list[float]:
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if not items:
        raise ValueError("fractions cannot be empty")

    vals = [float(v) for v in items]
    for v in vals:
        if v <= 0.0 or v > 1.0:
            raise ValueError("each fraction must satisfy 0 < fraction <= 1")
    return vals


def _parse_policies(spec: str) -> list[str]:
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if not items:
        raise ValueError("policies cannot be empty")
    for p in items:
        if p not in {"l2_guarded", "recency"}:
            raise ValueError(f"unsupported policy '{p}'")
    return items


def run_replay(
    fractions: Iterable[float],
    policies: Iterable[str],
    k: int,
    line_count: int = 200,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    variants = [build_variant_messages(layout, line_count=line_count) for layout in DEFAULT_LAYOUTS]
    for variant in variants:
        messages = variant["messages"]
        primary_ids = set(variant["primary_ids"])
        decoy_ids = set(variant["decoy_ids"])

        total_tokens = sum(token_count(msg["text"]) for msg in messages)

        for fraction in fractions:
            budget = max(1, int(total_tokens * fraction))
            for policy in policies:
                out = compact(messages=messages, token_budget=budget, policy=policy, k=k)
                if "error" in out:
                    raise RuntimeError(f"compaction error for policy={policy}: {out['error']}")

                kept = out["messages"]
                audit = out["audit"]
                kept_ids = {str(m["id"]) for m in kept if "id" in m}

                primary_full = int(primary_ids.issubset(kept_ids))
                decoy_full = int(decoy_ids.issubset(kept_ids))

                tokens_before = int(audit.get("tokens_before", total_tokens))
                tokens_after = int(audit.get("tokens_after", 0))
                realized_ret = (tokens_after / tokens_before) if tokens_before > 0 else 0.0

                rows.append(
                    {
                        "variant": variant["variant"],
                        "policy": policy,
                        "fraction": float(fraction),
                        "token_budget": int(budget),
                        "tokens_before": tokens_before,
                        "tokens_after": tokens_after,
                        "realized_token_retention": realized_ret,
                        "primary_full": primary_full,
                        "decoy_full": decoy_full,
                        "pivot_preservation": primary_full,
                        "feasible": audit.get("feasible"),
                        "contract_satisfied": audit.get("contract_satisfied"),
                        "breach_count": len(audit.get("breach_ids", [])),
                        "protected_count": len(audit.get("protected_ids", [])),
                    }
                )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), float(row["fraction"]))
        buckets.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (policy, fraction), vals in sorted(buckets.items(), key=lambda x: (x[0][0], -x[0][1])):
        n = len(vals)
        summary.append(
            {
                "policy": policy,
                "fraction": fraction,
                "n": n,
                "primary_full_rate": sum(int(v["primary_full"]) for v in vals) / n,
                "decoy_full_rate": sum(int(v["decoy_full"]) for v in vals) / n,
                "pivot_preservation_rate": sum(int(v["pivot_preservation"]) for v in vals) / n,
                "realized_token_retention_mean": sum(float(v["realized_token_retention"]) for v in vals) / n,
                "contract_satisfied_rate": sum(
                    1 if v["contract_satisfied"] is True else 0 for v in vals
                )
                / n,
                "mean_breach_count": sum(int(v["breach_count"]) for v in vals) / n,
            }
        )

    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay CyberOps-style compaction through tropical-compactor tools.")
    parser.add_argument("--fractions", default="1.0,0.8,0.65,0.5,0.4")
    parser.add_argument("--policies", default="recency,l2_guarded")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--line-count", type=int, default=200)
    parser.add_argument("--output-dir", default="artifacts/cyberops_mcp_replay")
    args = parser.parse_args()

    fractions = _parse_fractions(args.fractions)
    policies = _parse_policies(args.policies)

    rows = run_replay(fractions=fractions, policies=policies, k=args.k, line_count=args.line_count)
    summary = summarize_rows(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "replay_rows.csv"
    summary_csv = out_dir / "replay_summary.csv"
    summary_json = out_dir / "replay_summary.json"

    _write_csv(detail_csv, rows)
    _write_csv(summary_csv, summary)
    summary_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print(f"Wrote: {detail_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
