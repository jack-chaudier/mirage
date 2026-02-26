#!/usr/bin/env python3
"""Analyze tropical-compactor telemetry from Claude Code sessions.

Reads ~/.claude/compactor-telemetry.jsonl and produces summary reports.
Can also ingest session-specific JSONL files from artifacts/session_telemetry/.

Usage:
    python scripts/analyze_telemetry.py                          # analyze global telemetry
    python scripts/analyze_telemetry.py --session <session.jsonl> # analyze one session
    python scripts/analyze_telemetry.py --all-sessions            # analyze all sessions
    python scripts/analyze_telemetry.py --export-csv              # export to CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_TELEMETRY = Path.home() / ".claude" / "compactor-telemetry.jsonl"
SESSION_DIR = ROOT / "artifacts" / "session_telemetry"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  WARN: skipping malformed line {line_no} in {path}", file=sys.stderr)
    return entries


def summarize_entries(entries: list[dict[str, Any]], label: str = "") -> dict[str, Any]:
    if not entries:
        return {"label": label, "count": 0, "note": "no entries"}

    tool_counts = Counter(e.get("tool", "unknown") for e in entries)
    k_values = [e["k_max_feasible"] for e in entries if e.get("k_max_feasible") is not None]
    pred_counts = [e["predecessor_count"] for e in entries if e.get("predecessor_count") is not None]
    turns = [e["session_turn"] for e in entries if e.get("session_turn") is not None]

    # Track feasibility over time
    horizon_calls = [e for e in entries if e.get("tool") == "inspect_horizon"]
    feasibility_trace = []
    for h in horizon_calls:
        feasibility_trace.append({
            "turn": h.get("session_turn"),
            "k_max_feasible": h.get("k_max_feasible"),
            "predecessor_count": h.get("predecessor_count"),
        })

    # Detect compaction events (k drops)
    k_drops = []
    for i in range(1, len(feasibility_trace)):
        prev_k = feasibility_trace[i - 1].get("k_max_feasible")
        curr_k = feasibility_trace[i].get("k_max_feasible")
        if prev_k is not None and curr_k is not None and curr_k < prev_k:
            k_drops.append({
                "from_turn": feasibility_trace[i - 1].get("turn"),
                "to_turn": feasibility_trace[i].get("turn"),
                "k_before": prev_k,
                "k_after": curr_k,
                "delta": curr_k - prev_k,
            })

    return {
        "label": label,
        "total_calls": len(entries),
        "tool_counts": dict(tool_counts.most_common()),
        "k_max_feasible": {
            "values": k_values,
            "min": min(k_values) if k_values else None,
            "max": max(k_values) if k_values else None,
            "mean": sum(k_values) / len(k_values) if k_values else None,
        },
        "predecessor_counts": {
            "min": min(pred_counts) if pred_counts else None,
            "max": max(pred_counts) if pred_counts else None,
            "mean": sum(pred_counts) / len(pred_counts) if pred_counts else None,
        },
        "session_turns": {
            "min": min(turns) if turns else None,
            "max": max(turns) if turns else None,
        },
        "feasibility_trace": feasibility_trace,
        "k_drops_detected": k_drops,
        "compaction_events": len(k_drops),
    }


def export_csv(entries: list[dict[str, Any]], path: Path) -> None:
    if not entries:
        print("No entries to export.", file=sys.stderr)
        return

    fieldnames = ["timestamp", "tool", "result_summary", "k_max_feasible",
                  "pivot_id", "predecessor_count", "session_turn"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    print(f"Exported {len(entries)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tropical-compactor telemetry.")
    parser.add_argument("--session", type=str, help="Path to a specific session JSONL file")
    parser.add_argument("--all-sessions", action="store_true", help="Analyze all session files")
    parser.add_argument("--export-csv", action="store_true", help="Export to CSV")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "session_telemetry"),
                        help="Output directory for reports")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []

    # Always include global telemetry
    global_entries = load_jsonl(GLOBAL_TELEMETRY)
    if global_entries:
        report = summarize_entries(global_entries, label="global")
        reports.append(report)
        print(f"\n=== Global Telemetry ({len(global_entries)} entries) ===")
        print(json.dumps(report, indent=2, default=str))

    # Session-specific
    if args.session:
        session_path = Path(args.session)
        entries = load_jsonl(session_path)
        report = summarize_entries(entries, label=session_path.stem)
        reports.append(report)
        print(f"\n=== Session: {session_path.stem} ({len(entries)} entries) ===")
        print(json.dumps(report, indent=2, default=str))

    if args.all_sessions:
        for jsonl_file in sorted(SESSION_DIR.glob("*.jsonl")):
            entries = load_jsonl(jsonl_file)
            report = summarize_entries(entries, label=jsonl_file.stem)
            reports.append(report)
            print(f"\n=== Session: {jsonl_file.stem} ({len(entries)} entries) ===")
            print(json.dumps(report, indent=2, default=str))

    if args.export_csv and global_entries:
        export_csv(global_entries, out_dir / "telemetry_export.csv")

    # Write combined report
    if reports:
        report_path = out_dir / "telemetry_report.json"
        report_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote report: {report_path}")

    if not global_entries and not args.session and not args.all_sessions:
        print("No telemetry data found yet.")
        print(f"  Global: {GLOBAL_TELEMETRY}")
        print(f"  Sessions: {SESSION_DIR}")
        print("  Run a Claude Code session with the compactor to generate data.")


if __name__ == "__main__":
    main()
