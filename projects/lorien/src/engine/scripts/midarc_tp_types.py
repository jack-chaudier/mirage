"""Analyze valid mid-arc TP event types from existing midarc feasibility output.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.midarc_tp_types
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_INPUT_PATH = OUTPUT_DIR / "midarc_feasibility.json"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "midarc_tp_types.json"
DEFAULT_SEED_AUTOPSY_PATH = OUTPUT_DIR / "seed_autopsy.json"

# Order requested in prompt.
DISPLAY_EVENT_TYPES = [
    "catastrophe",
    "observe",
    "conflict",
    "reveal",
    "chat",
    "social_move",
    "internal",
    "confide",
    "physical",
    "lie",
]


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_seed_autopsy(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _extract_score_from_autopsy(_autopsy: dict[str, Any] | None, _seed: int, _event_id: str) -> float | None:
    # Placeholder hook for future schema support; current source already contains scores.
    return None


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_pos(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _verdict(type_counts: dict[str, int], total_valid: int) -> str:
    if total_valid <= 0:
        return "No valid mid-arc arcs found."

    kinetic = int(type_counts.get("catastrophe", 0)) + int(type_counts.get("conflict", 0))
    epistemic = (
        int(type_counts.get("observe", 0))
        + int(type_counts.get("internal", 0))
        + int(type_counts.get("social_move", 0))
    )
    kinetic_share = float(kinetic / total_valid)
    epistemic_share = float(epistemic / total_valid)

    if kinetic_share > 0.70:
        return "Kinetic: valid mid-arc arcs use later explosions, not epistemic events. Fix = better search exploration."
    if epistemic_share > 0.50:
        return "Epistemic: the classifier CAN recognize quiet events as TPs. Fix = agent-relative scoring to find them."
    return "Mixed landscape: both kinetic and epistemic TPs produce valid arcs."


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TP event types in valid mid-arc Diana arcs.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seed-autopsy", type=str, default=str(DEFAULT_SEED_AUTOPSY_PATH))
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    autopsy_path = _resolve_path(args.seed_autopsy)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    per_seed = list(payload.get("per_seed") or [])
    autopsy_payload = _load_seed_autopsy(autopsy_path)

    valid_rows: list[dict[str, Any]] = []
    per_seed_breakdown: list[dict[str, Any]] = []
    top_scoring_per_seed: list[dict[str, Any]] = []
    type_buckets: dict[str, list[dict[str, Any]]] = {}

    for seed_row in per_seed:
        seed = int(seed_row.get("seed", 0))
        candidates = list(seed_row.get("candidates") or [])
        valid_candidates = [row for row in candidates if bool(row.get("has_valid_arc", False))]

        seed_counts: dict[str, int] = {}
        for row in valid_candidates:
            event_type = str(row.get("event_type") or "unknown")
            seed_counts[event_type] = int(seed_counts.get(event_type, 0) + 1)

            best_arc = row.get("best_arc") or {}
            score = _safe_float(best_arc.get("score"))
            if score is None:
                score = _extract_score_from_autopsy(
                    autopsy_payload,
                    seed,
                    str(row.get("event_id") or ""),
                )

            record = {
                "seed": seed,
                "event_id": str(row.get("event_id") or ""),
                "event_type": event_type,
                "global_pos": float(row.get("global_pos", 0.0)),
                "score": score,
            }
            valid_rows.append(record)
            type_buckets.setdefault(event_type, []).append(record)

        per_seed_breakdown.append(
            {
                "seed": seed,
                "valid_total": int(len(valid_candidates)),
                "type_counts": dict(sorted(seed_counts.items())),
            }
        )

        top = None
        valid_with_score = [
            row for row in valid_candidates
            if _safe_float((row.get("best_arc") or {}).get("score")) is not None
        ]
        if valid_with_score:
            valid_with_score.sort(
                key=lambda row: (
                    float((row.get("best_arc") or {}).get("score")),
                    float(row.get("global_pos", 0.0)),
                    str(row.get("event_id") or ""),
                ),
                reverse=True,
            )
            best = valid_with_score[0]
            top = {
                "seed": seed,
                "best_tp_type": str(best.get("event_type") or "unknown"),
                "best_tp_event_id": str(best.get("event_id") or ""),
                "global_pos": float(best.get("global_pos", 0.0)),
                "score": float((best.get("best_arc") or {}).get("score")),
            }
        top_scoring_per_seed.append(top or {"seed": seed, "best_tp_type": None, "best_tp_event_id": None, "global_pos": None, "score": None})

    total_valid = int(len(valid_rows))
    observed_types = sorted(type_buckets.keys())
    ordered_types = [t for t in DISPLAY_EVENT_TYPES if t in observed_types] + [
        t for t in observed_types if t not in DISPLAY_EVENT_TYPES
    ]

    type_summary: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for event_type in ordered_types:
        rows = type_buckets.get(event_type, [])
        count = int(len(rows))
        type_counts[event_type] = count
        scores = [float(row["score"]) for row in rows if row["score"] is not None]
        positions = [float(row["global_pos"]) for row in rows]
        type_summary.append(
            {
                "event_type": event_type,
                "count": count,
                "share": float(count / total_valid) if total_valid else 0.0,
                "mean_tp_global_pos": _mean(positions),
                "median_tp_global_pos": float(median(positions)) if positions else None,
                "mean_score": _mean(scores),
            }
        )

    verdict = _verdict(type_counts, total_valid)

    out_payload = {
        "config": {
            "input_file": str(input_path),
            "seed_autopsy_file": str(autopsy_path),
            "total_seeds": int(len(per_seed)),
            "total_valid_midarc_arcs": total_valid,
        },
        "type_summary": type_summary,
        "per_seed_breakdown": per_seed_breakdown,
        "top_scoring_arc_per_seed": top_scoring_per_seed,
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print()
    print(f"=== VALID MID-ARC TURNING POINT EVENT TYPES ({total_valid} valid arcs, {len(per_seed)} seeds) ===")
    print()
    print("Event type        Count    Share    Mean TP pos    Mean score (if avail)")
    for row in type_summary:
        print(
            f"{row['event_type'].upper():<16}"
            f"{int(row['count']):>6}    "
            f"{float(row['share']) * 100.0:>5.1f}%    "
            f"{_fmt_pos(row['mean_tp_global_pos']):>6}         "
            f"{_fmt_score(row['mean_score']):>6}"
        )

    print()
    print("=== PER-SEED TP TYPE BREAKDOWN ===")
    for row in sorted(per_seed_breakdown, key=lambda item: int(item["seed"])):
        parts = [
            f"{event_type.upper()}={int(count)}"
            for event_type, count in sorted(
                row["type_counts"].items(),
                key=lambda item: (-int(item[1]), item[0]),
            )
        ]
        detail = "  ".join(parts) if parts else "(none)"
        print(f"seed={int(row['seed']):02d}: {detail}  (N valid total={int(row['valid_total'])})")

    print()
    print("=== TOP SCORING ARC PER SEED: TP EVENT DETAILS ===")
    for row in sorted(top_scoring_per_seed, key=lambda item: int(item["seed"])):
        if row["best_tp_type"] is None:
            print(f"seed={int(row['seed']):02d}: no valid candidate")
            continue
        print(
            f"seed={int(row['seed']):02d}: "
            f"best_tp_type={str(row['best_tp_type']).upper()}  "
            f"best_tp_event_id={row['best_tp_event_id']}  "
            f"global_pos={float(row['global_pos']):.2f}  "
            f"score={float(row['score']):.3f}"
        )

    print()
    print("=== VERDICT ===")
    print(verdict)


if __name__ == "__main__":
    main()
