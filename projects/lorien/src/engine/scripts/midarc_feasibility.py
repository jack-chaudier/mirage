"""Mid-arc turning-point feasibility diagnostic for Diana's strict-invalid seeds.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.midarc_feasibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_validator import validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.schema.events import BeatType, Event
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.k_sweep_experiment import _generate_canon_after_b_for_seed, _simulate_story
from scripts.test_goal_evolution import _evolution_profiles

DEFAULT_INVALID_SEEDS = [2, 3, 9, 25, 32, 33, 35, 38, 43]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_DTP_PATH = OUTPUT_DIR / "diana_turning_points.json"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "midarc_feasibility.json"
MIDARC_LOWER = 0.25
MIDARC_UPPER = 0.70
MAX_BEFORE = 10
MAX_AFTER = 9
MAX_EVENTS = 20
DEV_BEATS = {BeatType.COMPLICATION, BeatType.ESCALATION}


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_invalid_seeds(path: Path) -> list[int]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        detail = list(payload.get("invalid_per_seed_detail") or [])
        seeds = sorted({int(row.get("seed", 0)) for row in detail if int(row.get("seed", 0)) > 0})
        if seeds:
            return seeds
    return list(DEFAULT_INVALID_SEEDS)


def _involves_diana(event: Event) -> bool:
    return event.source_agent == "diana" or "diana" in event.target_agents


def _global_pos(event: Event, simulation_end_tick: int) -> float:
    return float(int(event.tick_id) / max(int(simulation_end_tick), 1))


def _event_sort_key(event: Event) -> tuple[int, int, str]:
    return (int(event.tick_id), int(event.order_in_tick), str(event.id))


def _beats_with_forced_tp(events: list[Event], tp_index: int) -> list[BeatType]:
    beats = list(classify_beats(events))

    # Force exactly one turning point at tp_index.
    for i, beat in enumerate(beats):
        if i == tp_index:
            beats[i] = BeatType.TURNING_POINT
        elif beat == BeatType.TURNING_POINT:
            beats[i] = BeatType.ESCALATION if i < tp_index else BeatType.CONSEQUENCE

    # Ensure no consequences before TP; keep development pressure before it.
    for i in range(tp_index):
        if beats[i] == BeatType.CONSEQUENCE:
            beats[i] = BeatType.ESCALATION

    # Ensure all beats after TP are consequences to preserve phase progression.
    for i in range(tp_index + 1, len(beats)):
        if beats[i] in {BeatType.SETUP, BeatType.COMPLICATION, BeatType.ESCALATION}:
            beats[i] = BeatType.CONSEQUENCE

    # Keep the arc opening in setup.
    if beats and beats[0] != BeatType.SETUP:
        beats[0] = BeatType.SETUP

    # If no development before TP, inject one.
    has_dev_before = any(beats[i] in DEV_BEATS for i in range(tp_index))
    if not has_dev_before and tp_index >= 1:
        inject_idx = min(1, tp_index - 1)
        beats[inject_idx] = BeatType.COMPLICATION

    return beats


def _beat_counts(beats: list[BeatType]) -> dict[str, int]:
    out = {
        "setup": 0,
        "complication": 0,
        "escalation": 0,
        "turning_point": 0,
        "consequence": 0,
    }
    for beat in beats:
        key = str(beat.value)
        if key in out:
            out[key] += 1
    return out


def _candidate_attempts(
    *,
    involved_events: list[Event],
    candidate_index: int,
    total_sim_time: float | None,
) -> list[dict[str, Any]]:
    before_pool = involved_events[:candidate_index]
    tp_event = involved_events[candidate_index]
    after_pool = involved_events[candidate_index + 1 :]

    max_before = min(MAX_BEFORE, len(before_pool))
    max_after = min(MAX_AFTER, len(after_pool))

    attempts: list[dict[str, Any]] = []
    for before_count in range(1, max_before + 1):
        for after_count in range(1, max_after + 1):
            selected = before_pool[-before_count:] + [tp_event] + after_pool[:after_count]
            if len(selected) < 4 or len(selected) > MAX_EVENTS:
                continue
            selected.sort(key=_event_sort_key)
            local_tp_index = selected.index(tp_event)

            beats = _beats_with_forced_tp(selected, local_tp_index)
            validation = validate_arc(
                events=selected,
                beats=beats,
                total_sim_time=total_sim_time,
            )
            counts = _beat_counts(beats)
            has_dev_pair = counts["complication"] > 0 and counts["escalation"] > 0

            score = None
            if validation.valid:
                score = float(score_arc(selected, beats).composite)

            attempts.append(
                {
                    "before_count": int(before_count),
                    "after_count": int(after_count),
                    "arc_event_count": int(len(selected)),
                    "valid": bool(validation.valid),
                    "violations": [str(v) for v in validation.violations],
                    "beat_counts": counts,
                    "has_development_beats": bool(has_dev_pair),
                    "score": score,
                    "events_before_tp": int(local_tp_index),
                    "events_after_tp": int(len(selected) - local_tp_index - 1),
                }
            )
    return attempts


def _best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attempts:
        return None

    valid_attempts = [row for row in attempts if bool(row["valid"])]
    if valid_attempts:
        valid_attempts.sort(
            key=lambda row: (
                float(row["score"] or 0.0),
                int(row["beat_counts"]["complication"] + row["beat_counts"]["escalation"]),
                int(row["arc_event_count"]),
            ),
            reverse=True,
        )
        return valid_attempts[0]

    attempts.sort(
        key=lambda row: (
            int(len(row["violations"])),
            -int(row["beat_counts"]["complication"] + row["beat_counts"]["escalation"]),
            -int(row["arc_event_count"]),
        )
    )
    return attempts[0]


def _format_beats(counts: dict[str, int]) -> str:
    return (
        f"SETUP={int(counts['setup'])} "
        f"COMPLICATION={int(counts['complication'])} "
        f"ESCALATION={int(counts['escalation'])} "
        f"TP={int(counts['turning_point'])} "
        f"CONSEQUENCE={int(counts['consequence'])}"
    )


def _print_seed_block(seed_row: dict[str, Any]) -> None:
    seed = int(seed_row["seed"])
    type_counts = seed_row["midarc_event_type_counts"]
    type_str = ", ".join(
        f"{event_type.upper()}: {int(count)}"
        for event_type, count in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    print(f"seed={seed:02d}:")
    print(
        f"  Diana events in [0.25, 0.70]: {int(seed_row['midarc_candidate_pool_size'])} total "
        f"({type_str})"
    )
    print(f"  Candidates tested: {int(seed_row['candidates_tested'])}")
    print(f"  Valid arcs found: {int(seed_row['valid_arcs_found'])}")
    print(f"  Arcs with development beats: {int(seed_row['arcs_with_development_beats'])}")

    best = seed_row.get("best_candidate")
    if best is None:
        print("  Best candidate: none")
        return

    best_arc = best["best_arc"]
    print(
        f"  Best candidate: event_id={best['event_id']} "
        f"type={best['event_type']} global_pos={float(best['global_pos']):.2f}"
    )
    print(f"    beats: {_format_beats(best_arc['beat_counts'])}")
    status = "YES" if bool(best_arc["valid"]) else "NO"
    print(
        f"    valid: {status}  violations: {best_arc['violations']}"
    )


def _verdict(seeds_with_valid_midarc_tp: int) -> str:
    if seeds_with_valid_midarc_tp == 0:
        return (
            "True feasibility void: no valid classical arc exists with mid-arc TP for Diana under full evolution"
        )
    return (
        "Search limitation: valid mid-arc arcs exist but the search doesn't find them. "
        f"{seeds_with_valid_midarc_tp}/9 seeds have feasible alternatives"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test mid-arc TP feasibility for Diana invalid seeds.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--invalid-seeds-from", type=str, default=str(DEFAULT_DTP_PATH))
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    invalid_seed_source = _resolve_path(args.invalid_seeds_from)
    seeds = _load_invalid_seeds(invalid_seed_source)
    evolutions = _evolution_profiles()["full"]

    per_seed: list[dict[str, Any]] = []
    total_candidates_tested = 0
    total_valid_arcs_found = 0
    seeds_with_valid_midarc_tp = 0
    seeds_with_dev_beats = 0
    seeds_with_zero_midarc_events = 0

    print()
    print("=== MID-ARC TP FEASIBILITY TEST (9 invalid seeds) ===")
    print()

    for seed in seeds:
        canon_after_b = _generate_canon_after_b_for_seed(
            seed=seed,
            event_limit=int(args.event_limit),
            tick_limit=int(args.tick_limit),
        )
        story = _simulate_story(
            label="full_evolution",
            seed=seed,
            loaded_canon=canon_after_b,
            tick_limit=int(args.tick_limit),
            event_limit=int(args.event_limit),
            evolutions=evolutions,
        )
        parsed = parse_simulation_output(story.payload)
        metrics_output = run_metrics_pipeline(parsed)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
        simulation_end_tick = max((int(event.tick_id) for event in metrics_output.events), default=1)

        diana_events = [event for event in metrics_output.events if _involves_diana(event)]
        diana_events.sort(key=_event_sort_key)

        midarc_candidates: list[tuple[int, Event, float]] = []
        for idx, event in enumerate(diana_events):
            pos = _global_pos(event, simulation_end_tick)
            if MIDARC_LOWER <= pos <= MIDARC_UPPER:
                midarc_candidates.append((idx, event, pos))

        candidate_type_counts: dict[str, int] = {}
        for _idx, event, _pos in midarc_candidates:
            key = str(event.type.value)
            candidate_type_counts[key] = int(candidate_type_counts.get(key, 0) + 1)

        candidate_rows: list[dict[str, Any]] = []
        valid_arcs_found = 0
        dev_arcs_found = 0
        for candidate_index, event, pos in midarc_candidates:
            attempts = _candidate_attempts(
                involved_events=diana_events,
                candidate_index=candidate_index,
                total_sim_time=total_sim_time,
            )
            best = _best_attempt(attempts)
            has_valid = any(bool(row["valid"]) for row in attempts)
            has_dev = any(bool(row["has_development_beats"]) for row in attempts)
            if has_valid:
                valid_arcs_found += 1
            if has_dev:
                dev_arcs_found += 1

            candidate_rows.append(
                {
                    "event_id": str(event.id),
                    "event_type": str(event.type.value),
                    "tick_id": int(event.tick_id),
                    "global_pos": float(pos),
                    "attempts_tested": int(len(attempts)),
                    "has_valid_arc": bool(has_valid),
                    "has_development_beats": bool(has_dev),
                    "best_arc": best,
                }
            )

        best_candidate = None
        if candidate_rows:
            sorted_candidates = sorted(
                candidate_rows,
                key=lambda row: (
                    0 if bool(row["has_valid_arc"]) else 1,
                    -float((row["best_arc"] or {}).get("score") or 0.0),
                    int(len((row["best_arc"] or {}).get("violations") or [])),
                    str(row["event_id"]),
                ),
            )
            best_candidate = sorted_candidates[0]

        seed_row = {
            "seed": int(seed),
            "simulation_end_tick": int(simulation_end_tick),
            "diana_involved_events_total": int(len(diana_events)),
            "midarc_candidate_pool_size": int(len(midarc_candidates)),
            "midarc_event_type_counts": candidate_type_counts,
            "candidates_tested": int(len(candidate_rows)),
            "valid_arcs_found": int(valid_arcs_found),
            "arcs_with_development_beats": int(dev_arcs_found),
            "best_candidate": best_candidate,
            "candidates": candidate_rows,
        }
        per_seed.append(seed_row)
        _print_seed_block(seed_row)
        print()

        total_candidates_tested += int(len(candidate_rows))
        total_valid_arcs_found += int(valid_arcs_found)
        if int(valid_arcs_found) > 0:
            seeds_with_valid_midarc_tp += 1
        if int(dev_arcs_found) > 0:
            seeds_with_dev_beats += 1
        if int(len(midarc_candidates)) == 0:
            seeds_with_zero_midarc_events += 1

    summary = {
        "seeds_with_at_least_one_valid_midarc_tp": int(seeds_with_valid_midarc_tp),
        "seeds_with_at_least_one_arc_with_dev_beats": int(seeds_with_dev_beats),
        "seeds_with_zero_midarc_diana_events": int(seeds_with_zero_midarc_events),
        "total_midarc_candidates_tested": int(total_candidates_tested),
        "total_valid_arcs_found": int(total_valid_arcs_found),
    }
    verdict = _verdict(seeds_with_valid_midarc_tp)

    print("=== SUMMARY ===")
    print(f"Seeds with at least one valid mid-arc TP:     {summary['seeds_with_at_least_one_valid_midarc_tp']}/9")
    print(f"Seeds with at least one arc with dev beats:   {summary['seeds_with_at_least_one_arc_with_dev_beats']}/9")
    print(f"Seeds with zero mid-arc Diana events:         {summary['seeds_with_zero_midarc_diana_events']}/9")
    print(f"Total mid-arc candidates tested:              {summary['total_midarc_candidates_tested']}")
    print(f"Total valid arcs found:                       {summary['total_valid_arcs_found']}")
    print()
    print("=== VERDICT ===")
    print(verdict)

    payload = {
        "config": {
            "invalid_seeds_source": str(invalid_seed_source),
            "seeds": [int(seed) for seed in seeds],
            "condition": "full_evolution",
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "midarc_window": {"lower": MIDARC_LOWER, "upper": MIDARC_UPPER},
            "manual_arc_construction": {
                "max_events": MAX_EVENTS,
                "max_before": MAX_BEFORE,
                "max_after": MAX_AFTER,
            },
        },
        "per_seed": per_seed,
        "summary": summary,
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
