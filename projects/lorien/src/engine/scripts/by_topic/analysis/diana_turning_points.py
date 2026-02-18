"""Analyze Diana turning-point positions from existing diana_timestamps output.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.diana_turning_points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_INPUT_PATH = OUTPUT_DIR / "diana_timestamps.json"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "diana_turning_points.json"


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _segment(global_pos: float) -> str:
    if global_pos < 0.25:
        return "early"
    if global_pos <= 0.70:
        return "mid"
    return "late"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [float(row["turning_point_global_pos"]) for row in records]
    before = [int(row["events_before_turning_point"]) for row in records]
    after = [int(row["events_after_turning_point"]) for row in records]

    segment_counts = {"early": 0, "mid": 0, "late": 0}
    for value in positions:
        segment_counts[_segment(value)] += 1

    return {
        "n": int(len(records)),
        "turning_point_global_pos": {
            "mean": float(_mean(positions)) if positions else None,
            "median": float(median(positions)) if positions else None,
            "min": float(min(positions)) if positions else None,
            "max": float(max(positions)) if positions else None,
        },
        "segment_counts": segment_counts,
        "events_before_turning_point_mean": float(_mean([float(v) for v in before])) if before else None,
        "events_after_turning_point_mean": float(_mean([float(v) for v in after])) if after else None,
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _verdict(valid_summary: dict[str, Any], invalid_summary: dict[str, Any]) -> str:
    n_valid = int(valid_summary["n"])
    n_invalid = int(invalid_summary["n"])
    valid_early = int(valid_summary["segment_counts"]["early"])
    invalid_early = int(invalid_summary["segment_counts"]["early"])
    valid_mean = valid_summary["turning_point_global_pos"]["mean"]
    invalid_mean = invalid_summary["turning_point_global_pos"]["mean"]

    invalid_early_share = float(invalid_early / n_invalid) if n_invalid else 0.0
    valid_early_share = float(valid_early / n_valid) if n_valid else 0.0
    mean_gap = (
        abs(float(valid_mean) - float(invalid_mean))
        if valid_mean is not None and invalid_mean is not None
        else 0.0
    )

    if invalid_early_share > 0.60 and valid_early_share < 0.40 and mean_gap >= 0.15:
        return "Premature phase collapse: search anchors climax too early, closing development phase"
    return "Phase placement is not the differentiator — beat vocabulary is the issue"


def _print_summary(
    *,
    valid_summary: dict[str, Any],
    invalid_summary: dict[str, Any],
    invalid_records: list[dict[str, Any]],
    verdict: str,
) -> None:
    print()
    print("=== DIANA TURNING POINT POSITION: VALID vs INVALID ===")
    print()

    print(f"Valid arcs (N={int(valid_summary['n'])}):")
    valid_pos = valid_summary["turning_point_global_pos"]
    print(
        "  Turning point global_pos:  "
        f"mean={_fmt(valid_pos['mean'])}  median={_fmt(valid_pos['median'])}  "
        f"min={_fmt(valid_pos['min'])}  max={_fmt(valid_pos['max'])}"
    )
    valid_seg = valid_summary["segment_counts"]
    print(
        "  Segment:  "
        f"early(<0.25): {int(valid_seg['early'])}   "
        f"mid(0.25-0.70): {int(valid_seg['mid'])}   "
        f"late(>0.70): {int(valid_seg['late'])}"
    )
    print(
        "  Events before TP (mean): "
        f"{_fmt(valid_summary['events_before_turning_point_mean'], 1)}   "
        f"Events after TP (mean): {_fmt(valid_summary['events_after_turning_point_mean'], 1)}"
    )
    print()

    print(f"Invalid arcs (N={int(invalid_summary['n'])}):")
    invalid_pos = invalid_summary["turning_point_global_pos"]
    print(
        "  Turning point global_pos:  "
        f"mean={_fmt(invalid_pos['mean'])}  median={_fmt(invalid_pos['median'])}  "
        f"min={_fmt(invalid_pos['min'])}  max={_fmt(invalid_pos['max'])}"
    )
    invalid_seg = invalid_summary["segment_counts"]
    print(
        "  Segment:  "
        f"early(<0.25): {int(invalid_seg['early'])}   "
        f"mid(0.25-0.70): {int(invalid_seg['mid'])}   "
        f"late(>0.70): {int(invalid_seg['late'])}"
    )
    print(
        "  Events before TP (mean): "
        f"{_fmt(invalid_summary['events_before_turning_point_mean'], 1)}   "
        f"Events after TP (mean): {_fmt(invalid_summary['events_after_turning_point_mean'], 1)}"
    )
    print()

    print("=== PER-SEED DETAIL (invalid arcs only) ===")
    for row in sorted(invalid_records, key=lambda item: int(item["seed"])):
        print(
            f"  seed={int(row['seed']):02d}  TP_pos={float(row['turning_point_global_pos']):.2f}  "
            f"events_before={int(row['events_before_turning_point']):02d}  "
            f"events_after={int(row['events_after_turning_point']):02d}"
        )
    print()
    print("=== VERDICT ===")
    print(verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Diana turning point positions from diana_timestamps.json.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    per_seed = list(payload.get("per_seed") or [])

    records: list[dict[str, Any]] = []
    skipped_no_turning_point: list[dict[str, Any]] = []
    skipped_missing_diana: list[int] = []

    for row in per_seed:
        seed = int(row.get("seed", 0))
        diana = row.get("diana")
        if not isinstance(diana, dict):
            skipped_missing_diana.append(seed)
            continue

        event_rows = list(diana.get("event_rows") or [])
        tp_indices = [index for index, event in enumerate(event_rows) if str(event.get("beat")) == "turning_point"]
        if not tp_indices:
            skipped_no_turning_point.append({"seed": seed, "valid": bool(diana.get("valid", False))})
            continue

        tp_index = int(tp_indices[0])
        tp_event = event_rows[tp_index]
        records.append(
            {
                "seed": seed,
                "valid": bool(diana.get("valid", False)),
                "turning_point_global_pos": float(tp_event.get("global_pos", 0.0)),
                "turning_point_tick_id": int(tp_event.get("tick_id", 0)),
                "events_before_turning_point": int(tp_index),
                "events_after_turning_point": int(len(event_rows) - tp_index - 1),
                "turning_point_count_in_arc": int(len(tp_indices)),
            }
        )

    valid_records = [row for row in records if bool(row["valid"])]
    invalid_records = [row for row in records if not bool(row["valid"])]

    valid_summary = _summarize(valid_records)
    invalid_summary = _summarize(invalid_records)
    verdict = _verdict(valid_summary, invalid_summary)

    out_payload = {
        "config": {
            "input_file": str(input_path),
            "total_seeds_seen": int(len(per_seed)),
            "records_with_turning_point": int(len(records)),
            "skipped_no_turning_point": int(len(skipped_no_turning_point)),
            "skipped_missing_diana": int(len(skipped_missing_diana)),
        },
        "records": records,
        "valid_summary": valid_summary,
        "invalid_summary": invalid_summary,
        "invalid_per_seed_detail": sorted(invalid_records, key=lambda row: int(row["seed"])),
        "skipped_no_turning_point": skipped_no_turning_point,
        "skipped_missing_diana": skipped_missing_diana,
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_summary(
        valid_summary=valid_summary,
        invalid_summary=invalid_summary,
        invalid_records=invalid_records,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
