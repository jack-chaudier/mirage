"""Phase prior regularizer test for full-evolution runs.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.phase_prior_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from narrativefield.extraction.rashomon import extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.k_sweep_experiment import (
    DINNER_PARTY_AGENTS,
    _generate_canon_after_b_for_seed,
    _simulate_story,
)
from scripts.phase_prior_arc_search import extract_rashomon_set_phase_prior
from scripts.test_goal_evolution import _evolution_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "phase_prior_test.json"
BEAT_TYPES = ["setup", "complication", "escalation", "turning_point", "consequence"]


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _parse_seeds(raw: str) -> list[int]:
    text = raw.strip()
    if "," in text:
        out = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not out:
            raise ValueError("No valid seeds parsed from comma list.")
        return sorted(set(out))
    if "-" in text:
        left, right = text.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            raise ValueError("Seed range must satisfy end >= start.")
        return list(range(start, end + 1))
    return [int(text)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _arc_payload(arc: Any, simulation_end_tick: int) -> dict[str, Any]:
    beat_counts = {beat: 0 for beat in BEAT_TYPES}
    for beat in arc.beats:
        beat_value = str(beat.value)
        if beat_value in beat_counts:
            beat_counts[beat_value] += 1

    tp_index: int | None = None
    for index, beat in enumerate(arc.beats):
        if str(beat.value) == "turning_point":
            tp_index = index
            break

    turning_point_tick_id: int | None = None
    turning_point_global_pos: float | None = None
    events_before_tp: int | None = None
    events_after_tp: int | None = None
    if tp_index is not None and tp_index < len(arc.events):
        turning_point_tick_id = int(arc.events[tp_index].tick_id)
        denom = max(int(simulation_end_tick), 1)
        turning_point_global_pos = float(turning_point_tick_id / denom)
        events_before_tp = int(tp_index)
        events_after_tp = int(len(arc.events) - tp_index - 1)

    return {
        "valid": bool(arc.valid),
        "violations": [str(v) for v in arc.violations],
        "turning_point_global_pos": turning_point_global_pos,
        "turning_point_tick_id": turning_point_tick_id,
        "beat_counts": beat_counts,
        "arc_event_count": int(len(arc.events)),
        "events_before_tp": events_before_tp,
        "events_after_tp": events_after_tp,
    }


def _valid_count(rows: list[dict[str, Any]], *, mode: str, agent: str) -> int:
    return sum(1 for row in rows if bool(row[mode][agent]["valid"]))


def _verdict(healed_count: int) -> str:
    if healed_count == 9:
        return "Complete healing: premature phase collapse is the sole cause of Diana invalidity"
    if healed_count in {7, 8}:
        return "Near-complete healing: phase collapse is the dominant cause with minor secondary factors"
    if healed_count == 0:
        return "No healing: the phase prior alone does not resolve Diana invalidity (search may lack viable mid-arc candidates)"
    return "Partial healing: phase collapse is necessary but not sufficient — secondary failure modes exist"


def _fmt_tp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _fmt_beats(beat_counts: dict[str, int]) -> str:
    return (
        f"SETUP={int(beat_counts['setup'])} "
        f"COMPLICATION={int(beat_counts['complication'])} "
        f"ESCALATION={int(beat_counts['escalation'])} "
        f"TURNING_POINT={int(beat_counts['turning_point'])} "
        f"CONSEQUENCE={int(beat_counts['consequence'])}"
    )


def _print_summary(
    *,
    diana_strict_valid: int,
    diana_phase_valid: int,
    healing_rows: list[dict[str, Any]],
    healed_count: int,
    thorne_strict_valid: int,
    thorne_phase_valid: int,
    thorne_regressions: list[int],
    marcus_strict_valid: int,
    marcus_phase_valid: int,
    marcus_regressions: list[int],
    diana_strict_valid_retained: int,
    diana_phase_tp_stats: dict[str, float | None],
    verdict: str,
) -> None:
    diana_strict_invalid = 50 - diana_strict_valid
    diana_phase_invalid = 50 - diana_phase_valid

    print()
    print("=== PHASE PRIOR TEST (full evolution, 50 seeds) ===")
    print()
    print(f"Diana strict extraction:       {diana_strict_valid} valid, {diana_strict_invalid} invalid")
    print(f"Diana phase-prior extraction:  {diana_phase_valid} valid, {diana_phase_invalid} invalid")
    print()
    print("=== HEALING DETAIL (9 strict-invalid seeds) ===")
    for row in healing_rows:
        phase_status = "VALID" if bool(row["phase_prior"]["valid"]) else "INVALID"
        print(
            f"  seed={int(row['seed']):02d}  "
            f"strict: INVALID (TP={_fmt_tp(row['strict']['turning_point_global_pos'])})  "
            f"->  phase-prior: {phase_status} (TP={_fmt_tp(row['phase_prior']['turning_point_global_pos'])})"
        )
        print(f"    beats: {_fmt_beats(row['phase_prior']['beat_counts'])}")
    print()
    print(f"Healed: {healed_count}/9")
    print(f"Not healed: {9 - healed_count}/9")
    print()
    print("=== REGRESSION CHECK ===")
    print(
        f"Thorne: strict {thorne_strict_valid} valid -> phase-prior {thorne_phase_valid} valid "
        f"({len(thorne_regressions)} regressions)"
    )
    print(
        f"Marcus: strict {marcus_strict_valid} valid -> phase-prior {marcus_phase_valid} valid "
        f"({len(marcus_regressions)} regressions)"
    )
    print(
        f"Diana (strict-valid): {diana_strict_valid_retained}/{diana_strict_valid} remained valid under phase-prior"
    )
    print()
    print("=== PHASE-PRIOR TP POSITIONS (Diana, all 50 seeds) ===")
    print(
        f"  mean={_fmt_tp(diana_phase_tp_stats['mean'])}  "
        f"median={_fmt_tp(diana_phase_tp_stats['median'])}  "
        f"min={_fmt_tp(diana_phase_tp_stats['min'])}  "
        f"max={_fmt_tp(diana_phase_tp_stats['max'])}"
    )
    print()
    print("=== VERDICT ===")
    print(verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-prior turning-point window test.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument("--tp-lower", type=float, default=0.25)
    parser.add_argument("--tp-upper", type=float, default=0.70)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    seeds = _parse_seeds(args.seeds)
    evolutions = _evolution_profiles()["full"]

    per_seed: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds, start=1):
        print(f"[{i:03d}/{len(seeds):03d}] seed={seed} condition=full_evolution", flush=True)
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

        strict_set = extract_rashomon_set(
            events=metrics_output.events,
            seed=seed,
            agents=list(DINNER_PARTY_AGENTS),
            total_sim_time=total_sim_time,
        )
        phase_prior_set = extract_rashomon_set_phase_prior(
            events=metrics_output.events,
            seed=seed,
            agents=list(DINNER_PARTY_AGENTS),
            total_sim_time=total_sim_time,
            tp_lower=float(args.tp_lower),
            tp_upper=float(args.tp_upper),
        )

        strict_by_agent = {str(arc.protagonist): arc for arc in strict_set.arcs}
        phase_by_agent = {str(arc.protagonist): arc for arc in phase_prior_set.arcs}

        row: dict[str, Any] = {
            "seed": int(seed),
            "simulation_end_tick": int(simulation_end_tick),
            "strict": {},
            "phase_prior": {},
        }
        for agent in DINNER_PARTY_AGENTS:
            row["strict"][agent] = _arc_payload(strict_by_agent[agent], simulation_end_tick)
            row["phase_prior"][agent] = _arc_payload(phase_by_agent[agent], simulation_end_tick)
        per_seed.append(row)

    diana_strict_valid = _valid_count(per_seed, mode="strict", agent="diana")
    diana_phase_valid = _valid_count(per_seed, mode="phase_prior", agent="diana")
    diana_strict_invalid = 50 - diana_strict_valid
    if diana_strict_valid != 41 or diana_strict_invalid != 9:
        strict_invalid_seeds = [
            int(row["seed"])
            for row in per_seed
            if not bool(row["strict"]["diana"]["valid"])
        ]
        raise RuntimeError(
            "Strict extraction does not match expected Diana 41/9 split. "
            f"Observed: valid={diana_strict_valid}, invalid={diana_strict_invalid}, "
            f"strict_invalid_seeds={strict_invalid_seeds}"
        )

    strict_invalid_rows = [row for row in per_seed if not bool(row["strict"]["diana"]["valid"])]
    healing_rows: list[dict[str, Any]] = []
    healed_count = 0
    not_healed_rows: list[dict[str, Any]] = []
    for row in strict_invalid_rows:
        detail = {
            "seed": int(row["seed"]),
            "strict": row["strict"]["diana"],
            "phase_prior": row["phase_prior"]["diana"],
        }
        healing_rows.append(detail)
        if bool(row["phase_prior"]["diana"]["valid"]):
            healed_count += 1
        else:
            not_healed_rows.append(detail)

    thorne_strict_valid = _valid_count(per_seed, mode="strict", agent="thorne")
    thorne_phase_valid = _valid_count(per_seed, mode="phase_prior", agent="thorne")
    thorne_regressions = [
        int(row["seed"])
        for row in per_seed
        if bool(row["strict"]["thorne"]["valid"]) and not bool(row["phase_prior"]["thorne"]["valid"])
    ]

    marcus_strict_valid = _valid_count(per_seed, mode="strict", agent="marcus")
    marcus_phase_valid = _valid_count(per_seed, mode="phase_prior", agent="marcus")
    marcus_regressions = [
        int(row["seed"])
        for row in per_seed
        if bool(row["strict"]["marcus"]["valid"]) and not bool(row["phase_prior"]["marcus"]["valid"])
    ]

    diana_strict_valid_rows = [row for row in per_seed if bool(row["strict"]["diana"]["valid"])]
    diana_strict_valid_retained = sum(
        1 for row in diana_strict_valid_rows if bool(row["phase_prior"]["diana"]["valid"])
    )

    diana_phase_tp_positions = [
        float(row["phase_prior"]["diana"]["turning_point_global_pos"])
        for row in per_seed
        if row["phase_prior"]["diana"]["turning_point_global_pos"] is not None
    ]
    diana_phase_tp_stats = {
        "mean": float(_mean(diana_phase_tp_positions)) if diana_phase_tp_positions else None,
        "median": float(median(diana_phase_tp_positions)) if diana_phase_tp_positions else None,
        "min": float(min(diana_phase_tp_positions)) if diana_phase_tp_positions else None,
        "max": float(max(diana_phase_tp_positions)) if diana_phase_tp_positions else None,
    }

    verdict = _verdict(healed_count)

    payload = {
        "config": {
            "seeds": [int(seed) for seed in seeds],
            "condition": "full_evolution",
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "turning_point_window": {
                "lower": float(args.tp_lower),
                "upper": float(args.tp_upper),
            },
        },
        "strict_summary": {
            "diana": {"valid": int(diana_strict_valid), "invalid": int(50 - diana_strict_valid)},
            "thorne": {"valid": int(thorne_strict_valid), "invalid": int(50 - thorne_strict_valid)},
            "marcus": {"valid": int(marcus_strict_valid), "invalid": int(50 - marcus_strict_valid)},
        },
        "phase_prior_summary": {
            "diana": {"valid": int(diana_phase_valid), "invalid": int(50 - diana_phase_valid)},
            "thorne": {"valid": int(thorne_phase_valid), "invalid": int(50 - thorne_phase_valid)},
            "marcus": {"valid": int(marcus_phase_valid), "invalid": int(50 - marcus_phase_valid)},
        },
        "healing_detail": {
            "strict_invalid_seeds": [int(row["seed"]) for row in strict_invalid_rows],
            "rows": healing_rows,
            "healed_count": int(healed_count),
            "not_healed_count": int(len(not_healed_rows)),
            "not_healed_rows": not_healed_rows,
        },
        "regression_check": {
            "thorne": {
                "strict_valid": int(thorne_strict_valid),
                "phase_prior_valid": int(thorne_phase_valid),
                "regressions": thorne_regressions,
                "regression_count": int(len(thorne_regressions)),
            },
            "marcus": {
                "strict_valid": int(marcus_strict_valid),
                "phase_prior_valid": int(marcus_phase_valid),
                "regressions": marcus_regressions,
                "regression_count": int(len(marcus_regressions)),
            },
            "diana_strict_valid_retained": {
                "retained_valid_count": int(diana_strict_valid_retained),
                "strict_valid_total": int(diana_strict_valid),
            },
        },
        "diana_phase_prior_tp_positions": diana_phase_tp_stats,
        "per_seed": per_seed,
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_summary(
        diana_strict_valid=diana_strict_valid,
        diana_phase_valid=diana_phase_valid,
        healing_rows=healing_rows,
        healed_count=healed_count,
        thorne_strict_valid=thorne_strict_valid,
        thorne_phase_valid=thorne_phase_valid,
        thorne_regressions=thorne_regressions,
        marcus_strict_valid=marcus_strict_valid,
        marcus_phase_valid=marcus_phase_valid,
        marcus_regressions=marcus_regressions,
        diana_strict_valid_retained=diana_strict_valid_retained,
        diana_phase_tp_stats=diana_phase_tp_stats,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
