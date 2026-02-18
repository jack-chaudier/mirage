#!/usr/bin/env python3
"""Validate NTSB-style incident event graph JSON and emit audit artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

VALID_PHASES = {"SETUP", "DEVELOPMENT", "TURNING_POINT", "RESOLUTION"}
REQUIRED_TOP_LEVEL_KEYS = {"metadata", "incidents"}
REQUIRED_INCIDENT_KEYS = {
    "id",
    "title",
    "date",
    "ntsb_id",
    "summary",
    "probable_cause",
    "focal_actor",
    "k_recommended",
    "pivot_ground_truth",
    "competing_pivots",
    "events",
}
REQUIRED_EVENT_KEYS = {"id", "t", "actor", "weight", "desc", "phase"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="/Users/jackg/Downloads/ntsb_event_graphs.json",
        help="Input incident graph JSON.",
    )
    parser.add_argument(
        "--report-json",
        default="endogenous_context_theory/results/ntsb/ntsb_validation_report.json",
        help="Output summary report JSON.",
    )
    parser.add_argument(
        "--audit-csv",
        default="endogenous_context_theory/results/ntsb/ntsb_incident_audit.csv",
        help="Output per-incident audit CSV.",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero when structural validation errors are found.",
    )
    return parser.parse_args()


def _error(errs: List[Dict[str, Any]], incident_id: str, code: str, detail: str) -> None:
    errs.append({"incident_id": incident_id, "code": code, "detail": detail})


def _require_fields(container: Dict[str, Any], required: set[str]) -> List[str]:
    return sorted(k for k in required if k not in container)


def _dev_before_pivot(events: List[Dict[str, Any]], pivot_t: int) -> int:
    return sum(
        1
        for e in events
        if e.get("phase") == "DEVELOPMENT"
        and isinstance(e.get("t"), int)
        and e["t"] < pivot_t
    )


def _incident_stats(incident: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    incident_id = str(incident.get("id", "<missing>"))

    missing_fields = _require_fields(incident, REQUIRED_INCIDENT_KEYS)
    if missing_fields:
        _error(issues, incident_id, "missing_incident_fields", ",".join(missing_fields))

    events = incident.get("events")
    if not isinstance(events, list) or not events:
        _error(issues, incident_id, "events_missing_or_empty", "events must be non-empty list")
        return ({"incident_id": incident_id}, issues)

    event_ids: List[str] = []
    time_values: List[int] = []
    phase_counter: Counter[str] = Counter()

    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            _error(issues, incident_id, "event_not_object", f"index={idx}")
            continue

        missing_event_fields = _require_fields(event, REQUIRED_EVENT_KEYS)
        if missing_event_fields:
            _error(
                issues,
                incident_id,
                "missing_event_fields",
                f"event_index={idx};missing={','.join(missing_event_fields)}",
            )

        event_id = event.get("id")
        if isinstance(event_id, str):
            event_ids.append(event_id)
        else:
            _error(issues, incident_id, "event_id_invalid", f"index={idx}")

        t_value = event.get("t")
        if isinstance(t_value, int):
            time_values.append(t_value)
        else:
            _error(issues, incident_id, "event_t_invalid", f"event_id={event_id}")

        weight = event.get("weight")
        if not isinstance(weight, (int, float)):
            _error(issues, incident_id, "event_weight_invalid", f"event_id={event_id}")

        phase = event.get("phase")
        if isinstance(phase, str):
            phase_counter[phase] += 1
            if phase not in VALID_PHASES:
                _error(issues, incident_id, "event_phase_invalid", f"event_id={event_id};phase={phase}")
        else:
            _error(issues, incident_id, "event_phase_missing", f"event_id={event_id}")

    if len(event_ids) != len(set(event_ids)):
        _error(issues, incident_id, "duplicate_event_ids", "event ids must be unique")

    if time_values != sorted(time_values):
        _error(issues, incident_id, "non_monotonic_time", "event t values must be non-decreasing")

    pivot_id = incident.get("pivot_ground_truth")
    if pivot_id not in set(event_ids):
        _error(issues, incident_id, "pivot_missing", f"pivot_ground_truth={pivot_id}")

    competing = incident.get("competing_pivots")
    if not isinstance(competing, list):
        _error(issues, incident_id, "competing_pivots_invalid", "must be list")
        competing = []

    for cp in competing:
        if cp not in set(event_ids):
            _error(issues, incident_id, "competing_pivot_missing", f"competing_pivot={cp}")

    pivot_t = None
    if pivot_id in set(event_ids):
        pivot_event = next(e for e in events if isinstance(e, dict) and e.get("id") == pivot_id)
        pivot_t = pivot_event.get("t") if isinstance(pivot_event.get("t"), int) else None

    k_recommended = incident.get("k_recommended")
    if not isinstance(k_recommended, int):
        _error(issues, incident_id, "k_recommended_invalid", f"k_recommended={k_recommended}")

    dev_before_pivot = None
    k_violation = None
    if isinstance(pivot_t, int):
        dev_before_pivot = _dev_before_pivot(events, pivot_t)
    if isinstance(k_recommended, int) and isinstance(dev_before_pivot, int):
        k_violation = dev_before_pivot < k_recommended

    stats: Dict[str, Any] = {
        "incident_id": incident_id,
        "title": incident.get("title"),
        "n_events": len(events),
        "n_setup": phase_counter.get("SETUP", 0),
        "n_development": phase_counter.get("DEVELOPMENT", 0),
        "n_turning_point": phase_counter.get("TURNING_POINT", 0),
        "n_resolution": phase_counter.get("RESOLUTION", 0),
        "pivot_id": pivot_id,
        "pivot_t": pivot_t,
        "k_recommended": k_recommended,
        "dev_before_pivot": dev_before_pivot,
        "k_violation": k_violation,
        "competing_pivot_count": len(competing),
        "issue_count": len([i for i in issues if i["incident_id"] == incident_id]),
    }
    return stats, issues


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    audit_path = Path(args.audit_csv).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))

    errors: List[Dict[str, Any]] = []
    top_missing = sorted(k for k in REQUIRED_TOP_LEVEL_KEYS if k not in payload)
    if top_missing:
        _error(errors, "<top-level>", "missing_top_level_keys", ",".join(top_missing))

    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        _error(errors, "<top-level>", "incidents_not_list", "incidents must be list")
        incidents = []

    rows: List[Dict[str, Any]] = []
    for incident in incidents:
        if not isinstance(incident, dict):
            _error(errors, "<unknown>", "incident_not_object", str(type(incident).__name__))
            continue
        stats, issues = _incident_stats(incident)
        rows.append(stats)
        errors.extend(issues)

    audit_df = pd.DataFrame(rows)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["incident_id"], kind="stable")
    audit_df.to_csv(audit_path, index=False)

    k_violation_ids: List[str] = []
    if not audit_df.empty and "k_violation" in audit_df.columns:
        k_violation_ids = [
            str(v)
            for v in audit_df.loc[audit_df["k_violation"] == True, "incident_id"].tolist()  # noqa: E712
        ]

    report: Dict[str, Any] = {
        "input_json": str(input_path),
        "incidents_count": len(incidents),
        "audit_csv": str(audit_path),
        "errors_count": len(errors),
        "k_violation_count": len(k_violation_ids),
        "k_violation_incidents": k_violation_ids,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation report: {report_path}")
    print(f"Incident audit CSV: {audit_path}")
    print(f"Incidents: {len(incidents)}")
    print(f"Errors: {len(errors)}")
    print(f"k violations: {len(k_violation_ids)}")

    if k_violation_ids:
        print("k violation incidents:")
        for iid in k_violation_ids:
            print(f"  - {iid}")

    if args.fail_on_errors and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
