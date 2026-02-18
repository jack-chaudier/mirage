"""Analyze Rashomon sweep output for recurring structural wound patterns.

Run:
    cd src/engine && python -m scripts.analyze_wounds --input scripts/output/rashomon_sweep.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BEAT_ORDER = ["setup", "complication", "escalation", "turning_point", "consequence"]


def _resolve_output_path(output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return output_path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_DIR / f"wound_analysis_{timestamp}.json"


def _pct_counter(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        out[key] = {
            "count": int(count),
            "pct": (float(count) / float(total)) if total > 0 else 0.0,
        }
    return out


def _format_pair(source: str, target: str) -> str:
    return f"{source}-{target}"


def _sorted_pair(source: str, target: str) -> tuple[str, str]:
    left, right = sorted((source, target))
    return left, right


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _print_analysis_summary(payload: dict[str, Any], threshold: float) -> None:
    tp = payload["turning_point_analysis"]
    profiles = payload["protagonist_profiles"]
    wound_candidates = payload["wound_candidates"]
    total_valid_arcs = int(payload["total_valid_arcs"])

    print()
    print("=== WOUND ANALYSIS ===")
    print()

    print("Turning Point Event Types:")
    for event_type, stats in tp["by_event_type"].items():
        print(
            f"  {event_type.upper():<14} {int(stats['count']):>3}/{total_valid_arcs:<3} "
            f"({float(stats['pct']) * 100.0:>4.1f}%)"
        )
    print()

    print("Turning Point Locations:")
    for location, stats in tp["by_location"].items():
        print(
            f"  {location:<14} {int(stats['count']):>3}/{total_valid_arcs:<3} "
            f"({float(stats['pct']) * 100.0:>4.1f}%)"
        )
    print()

    print(f"Wound Candidates (agent-pair x location patterns recurring >{threshold:.0%}):")
    if not wound_candidates:
        print("  (none)")
    else:
        for candidate in wound_candidates[:10]:
            print(f"  {candidate['pattern']:<30} {float(candidate['frequency']) * 100.0:>5.1f}%")
    print()

    print("Protagonist Profiles:")
    for protagonist, profile in sorted(
        profiles.items(),
        key=lambda item: (-float(item[1]["validity_rate"]), item[0]),
    ):
        print(
            f"  {protagonist:<8} {float(profile['validity_rate']) * 100.0:>5.1f}% valid, "
            f"mean score {float(profile['mean_score']):.2f}, "
            f"mean events {float(profile['mean_event_count']):.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze recurring wound patterns from Rashomon sweep output.")
    parser.add_argument("--input", required=True, type=str, help="Path to rashomon sweep JSON.")
    parser.add_argument("--output", default=None, type=str, help="Path to output wound-analysis JSON.")
    parser.add_argument(
        "--threshold",
        default=0.25,
        type=float,
        help="Minimum seed recurrence frequency for wound candidates (0.0-1.0).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sweep = json.loads(input_path.read_text(encoding="utf-8"))
    per_seed = list(sweep.get("per_seed") or [])
    total_seeds = int(sweep.get("total_seeds", len(per_seed)))

    turning_type = Counter()
    turning_location = Counter()
    turning_pair = Counter()
    turning_total = 0

    escalation_type = Counter()
    escalation_location = Counter()
    escalation_pair = Counter()
    escalation_total = 0

    location_beat_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    high_tension_location = Counter()
    high_tension_total = 0

    turning_struct_count: Counter[tuple[str, str, tuple[str, ...], str]] = Counter()
    turning_struct_seeds: dict[tuple[str, str, tuple[str, ...], str], set[int]] = defaultdict(set)

    protagonist_total = Counter()
    protagonist_valid = Counter()
    protagonist_scores: dict[str, list[float]] = defaultdict(list)
    protagonist_event_counts: dict[str, list[int]] = defaultdict(list)
    protagonist_beat_counts: dict[str, Counter[str]] = defaultdict(Counter)
    protagonist_tp_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    protagonist_structural_diversity: dict[str, set[tuple[str, str, tuple[str, ...], str]]] = defaultdict(set)

    pattern_seed_hits: dict[tuple[tuple[str, str], str], set[int]] = defaultdict(set)
    pattern_event_types: dict[tuple[tuple[str, str], str], set[str]] = defaultdict(set)
    pattern_beats: dict[tuple[tuple[str, str], str], set[str]] = defaultdict(set)

    total_valid_arcs = 0
    for seed_row in per_seed:
        seed = int(seed_row.get("seed", 0))
        for arc in list(seed_row.get("arcs") or []):
            protagonist = str(arc.get("protagonist") or "")
            if protagonist:
                protagonist_total[protagonist] += 1
            is_valid = bool(arc.get("valid", False))
            if is_valid and protagonist:
                protagonist_valid[protagonist] += 1
            if not is_valid:
                continue

            total_valid_arcs += 1
            score = arc.get("score")
            if score is not None and protagonist:
                protagonist_scores[protagonist].append(_safe_float(score))
            if protagonist:
                protagonist_event_counts[protagonist].append(int(arc.get("event_count", 0) or 0))

            beats = [str(beat) for beat in (arc.get("beats") or [])]
            events = list(arc.get("events") or [])
            for beat in beats:
                if protagonist:
                    protagonist_beat_counts[protagonist][beat] += 1

            for event_row, beat in zip(events, beats):
                event_type = str(event_row.get("type") or "unknown")
                location = str(event_row.get("location_id") or "unknown")
                source = str(event_row.get("source_agent") or "unknown")
                targets = [str(target) for target in (event_row.get("target_agents") or [])]
                target_head = targets[0] if targets else "(none)"
                pair = _format_pair(source, target_head)

                location_beat_matrix[location][beat] += 1
                tension = _safe_float(event_row.get("tension"), default=0.0)
                if beat in {"escalation", "turning_point", "consequence"} and tension >= 0.6:
                    high_tension_location[location] += 1
                    high_tension_total += 1

                if beat == "turning_point":
                    turning_total += 1
                    turning_type[event_type] += 1
                    turning_location[location] += 1
                    turning_pair[pair] += 1

                    struct_key = (event_type, source, tuple(sorted(targets)), location)
                    turning_struct_count[struct_key] += 1
                    turning_struct_seeds[struct_key].add(seed)
                    if protagonist:
                        protagonist_tp_type_counts[protagonist][event_type] += 1
                        protagonist_structural_diversity[protagonist].add(struct_key)

                if beat == "escalation":
                    escalation_total += 1
                    escalation_type[event_type] += 1
                    escalation_location[location] += 1
                    escalation_pair[pair] += 1

                if beat in {"turning_point", "escalation"}:
                    pair_key = (_sorted_pair(source, target_head), location)
                    pattern_seed_hits[pair_key].add(seed)
                    pattern_event_types[pair_key].add(event_type)
                    pattern_beats[pair_key].add(beat)

    most_recurring_events: list[dict[str, Any]] = []
    for struct_key, count in turning_struct_count.most_common(20):
        event_type, source, targets, location = struct_key
        seeds = sorted(turning_struct_seeds[struct_key])
        targets_display = ",".join(targets) if targets else "(none)"
        most_recurring_events.append(
            {
                "description_pattern": f"{event_type}:{source}->{targets_display}@{location}",
                "structure": {
                    "event_type": event_type,
                    "source_agent": source,
                    "target_agents": list(targets),
                    "location_id": location,
                },
                "frequency": int(count),
                "seed_frequency": (len(seeds) / float(total_seeds)) if total_seeds else 0.0,
                "seeds": seeds,
            }
        )

    location_matrix_payload = {
        location: {beat: int(counter.get(beat, 0)) for beat in BEAT_ORDER}
        for location, counter in sorted(location_beat_matrix.items())
    }
    high_tension_payload = {
        location: {
            "count": int(count),
            "pct": (float(count) / float(high_tension_total)) if high_tension_total else 0.0,
        }
        for location, count in sorted(high_tension_location.items(), key=lambda item: (-item[1], item[0]))
    }

    protagonists = sorted(protagonist_total.keys())
    protagonist_profiles: dict[str, dict[str, Any]] = {}
    for protagonist in protagonists:
        total = int(protagonist_total.get(protagonist, 0))
        valid = int(protagonist_valid.get(protagonist, 0))
        scores = protagonist_scores.get(protagonist, [])
        event_counts = protagonist_event_counts.get(protagonist, [])
        tp_type_counts = protagonist_tp_type_counts.get(protagonist, Counter())
        beat_counts = protagonist_beat_counts.get(protagonist, Counter())
        most_common_tp_type = tp_type_counts.most_common(1)[0][0] if tp_type_counts else None
        typical_beat_distribution = {
            beat: (float(beat_counts.get(beat, 0)) / float(valid)) if valid else 0.0
            for beat in BEAT_ORDER
        }
        protagonist_profiles[protagonist] = {
            "validity_rate": (float(valid) / float(total)) if total else 0.0,
            "mean_score": mean(scores) if scores else 0.0,
            "mean_event_count": mean(event_counts) if event_counts else 0.0,
            "typical_beat_distribution": typical_beat_distribution,
            "most_common_tp_type": most_common_tp_type,
            "structural_diversity": len(protagonist_structural_diversity.get(protagonist, set())),
        }

    wound_candidates: list[dict[str, Any]] = []
    for pattern_key, seed_hits in pattern_seed_hits.items():
        (agent_a, agent_b), location = pattern_key
        recurrence = (len(seed_hits) / float(total_seeds)) if total_seeds else 0.0
        if recurrence < float(args.threshold):
            continue
        event_types = sorted(pattern_event_types[pattern_key])
        beats = sorted(pattern_beats[pattern_key])
        wound_candidates.append(
            {
                "pattern": f"{agent_a}-{agent_b} @ {location}",
                "frequency": recurrence,
                "description": (
                    f"{agent_a} and {agent_b} at {location} produce "
                    f"{'/'.join(event_types)} during {'/'.join(beats)} in {recurrence:.0%} of seeds"
                ),
                "seeds": sorted(seed_hits),
                "event_types": event_types,
                "beat_types": beats,
            }
        )
    wound_candidates.sort(key=lambda item: (-float(item["frequency"]), item["pattern"]))

    turning_point_analysis = {
        "by_event_type": _pct_counter(turning_type, turning_total),
        "by_location": _pct_counter(turning_location, turning_total),
        "by_agent_pair": _pct_counter(turning_pair, turning_total),
        "most_recurring_events": most_recurring_events,
    }
    escalation_analysis = {
        "by_event_type": _pct_counter(escalation_type, escalation_total),
        "by_location": _pct_counter(escalation_location, escalation_total),
        "by_agent_pair": _pct_counter(escalation_pair, escalation_total),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "threshold": float(args.threshold),
        "total_valid_arcs": int(total_valid_arcs),
        "turning_point_analysis": turning_point_analysis,
        "escalation_analysis": escalation_analysis,
        "location_clustering": {
            "location_beat_matrix": location_matrix_payload,
            "high_tension_locations": high_tension_payload,
        },
        "protagonist_profiles": protagonist_profiles,
        "wound_candidates": wound_candidates,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _print_analysis_summary(payload, threshold=float(args.threshold))
    print()
    print(f"Saved wound analysis: {output_path}")


if __name__ == "__main__":
    main()
