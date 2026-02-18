#!/usr/bin/env python3
"""Apply per-incident cleanup manifest to NTSB event graph JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="/Users/jackg/Downloads/ntsb_event_graphs.json",
        help="Source JSON file.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="endogenous_context_theory/data/ntsb/ntsb_k_phase_cleanup_manifest.csv",
        help="Manifest CSV file.",
    )
    parser.add_argument(
        "--output-json",
        default="endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json",
        help="Cleaned output JSON file.",
    )
    parser.add_argument(
        "--changes-json",
        default="endogenous_context_theory/results/ntsb/ntsb_manifest_apply_changes.json",
        help="Write applied change log to this JSON path.",
    )
    return parser.parse_args()


def _parse_k_value(raw: str) -> int:
    cleaned = (raw or "").strip().lower().replace(" ", "")
    if not cleaned.startswith("k="):
        raise ValueError(f"Expected k=... value, got: {raw!r}")
    return int(cleaned.split("=", 1)[1])


def _is_blank_target(raw: str) -> bool:
    return (raw or "").strip() in {"", "-", "—", "--", "None", "none"}


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "incident_id",
            "action_type",
            "target_event_id",
            "old_value",
            "new_value",
            "justification",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    changes_path = Path(args.changes_json).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    changes_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        raise ValueError("Input JSON must contain incidents list")

    manifest_rows = _load_manifest(manifest_path)
    incidents_by_id = {str(i.get("id")): i for i in incidents if isinstance(i, dict)}

    seen_actions = set()
    applied_changes: List[Dict[str, Any]] = []

    for idx, row in enumerate(manifest_rows, start=1):
        incident_id = row["incident_id"]
        action_type = row["action_type"]
        target_event_id = row["target_event_id"]
        old_value = row["old_value"]
        new_value = row["new_value"]
        justification = row["justification"]

        action_key = (incident_id, action_type, target_event_id)
        if action_key in seen_actions:
            raise ValueError(f"Duplicate manifest action: {action_key}")
        seen_actions.add(action_key)

        incident = incidents_by_id.get(incident_id)
        if incident is None:
            raise ValueError(f"Manifest row {idx}: incident_id not found: {incident_id}")

        if action_type == "k_adjust":
            if not _is_blank_target(target_event_id):
                raise ValueError(
                    f"Manifest row {idx}: k_adjust target_event_id must be blank/-, got {target_event_id!r}"
                )
            current_k = incident.get("k_recommended")
            expected_old_k = _parse_k_value(old_value)
            expected_new_k = _parse_k_value(new_value)
            if current_k != expected_old_k:
                raise ValueError(
                    f"Manifest row {idx}: {incident_id} k mismatch; "
                    f"expected {expected_old_k}, found {current_k}"
                )
            incident["k_recommended"] = expected_new_k
            applied_changes.append(
                {
                    "incident_id": incident_id,
                    "action_type": action_type,
                    "target_event_id": None,
                    "old_value": old_value,
                    "new_value": new_value,
                    "justification": justification,
                }
            )
            continue

        if action_type == "phase_relabel":
            if _is_blank_target(target_event_id):
                raise ValueError(
                    f"Manifest row {idx}: phase_relabel requires target_event_id"
                )
            events = incident.get("events")
            if not isinstance(events, list):
                raise ValueError(f"Manifest row {idx}: incident {incident_id} has invalid events")
            event = next((e for e in events if isinstance(e, dict) and e.get("id") == target_event_id), None)
            if event is None:
                raise ValueError(
                    f"Manifest row {idx}: event {target_event_id} not found in {incident_id}"
                )
            current_phase = event.get("phase")
            if current_phase != old_value:
                raise ValueError(
                    f"Manifest row {idx}: phase mismatch for {incident_id}/{target_event_id}; "
                    f"expected {old_value}, found {current_phase}"
                )
            event["phase"] = new_value
            applied_changes.append(
                {
                    "incident_id": incident_id,
                    "action_type": action_type,
                    "target_event_id": target_event_id,
                    "old_value": old_value,
                    "new_value": new_value,
                    "justification": justification,
                }
            )
            continue

        raise ValueError(f"Manifest row {idx}: unknown action_type {action_type!r}")

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "input_json": str(input_path),
        "manifest_csv": str(manifest_path),
        "output_json": str(output_path),
        "applied_changes_count": len(applied_changes),
        "applied_changes": applied_changes,
    }
    changes_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Applied {len(applied_changes)} manifest changes.")
    print(f"Cleaned JSON: {output_path}")
    print(f"Change log: {changes_path}")


if __name__ == "__main__":
    main()
