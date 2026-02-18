"""Base-rate mid-arc TP feasibility diagnostic for Diana's strict-valid seeds.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.baserate_feasibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.events import Event
from scripts.k_sweep_experiment import _generate_canon_after_b_for_seed, _simulate_story
from scripts.midarc_feasibility import (
    MIDARC_LOWER,
    MIDARC_UPPER,
    _best_attempt,
    _candidate_attempts,
    _event_sort_key,
    _global_pos,
    _involves_diana,
)
from scripts.test_goal_evolution import _evolution_profiles

INVALID_SEEDS = [2, 3, 9, 25, 32, 33, 35, 38, 43]
VALID_SEEDS = [seed for seed in range(1, 51) if seed not in set(INVALID_SEEDS)]
if len(VALID_SEEDS) != 41:
    raise RuntimeError("Expected 41 valid seeds in range [1, 50].")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "baserate_feasibility.json"
DEFAULT_INVALID_BASELINE_PATH = OUTPUT_DIR / "midarc_feasibility.json"
DEFAULT_INVALID_VALID_SCORE = 0.547
PAREIDOLIA_GAP_THRESHOLD_PP = 10.0


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))

    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def _fmt_percent(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{float(rate) * 100.0:.1f}%"


def _fmt_score(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{float(score):.3f}"


def _load_invalid_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": str(path),
            "seed_count": len(INVALID_SEEDS),
            "candidates_tested": 0,
            "valid_arcs_found": 0,
            "validity_rate": None,
            "mean_valid_score_recomputed": None,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    per_seed = list(payload.get("per_seed") or [])
    config = payload.get("config") or {}

    candidates_tested = int(summary.get("total_midarc_candidates_tested", 0))
    valid_arcs_found = int(summary.get("total_valid_arcs_found", 0))
    validity_rate = (
        float(valid_arcs_found / candidates_tested)
        if candidates_tested > 0
        else None
    )

    valid_scores: list[float] = []
    for seed_row in per_seed:
        for candidate in list(seed_row.get("candidates") or []):
            if not bool(candidate.get("has_valid_arc", False)):
                continue
            best_arc = candidate.get("best_arc") or {}
            score = best_arc.get("score")
            if score is not None:
                valid_scores.append(float(score))

    seeds = [int(seed) for seed in list(config.get("seeds") or []) if int(seed) > 0]
    return {
        "source": str(path),
        "seed_count": int(len(seeds) or len(per_seed) or len(INVALID_SEEDS)),
        "candidates_tested": int(candidates_tested),
        "valid_arcs_found": int(valid_arcs_found),
        "validity_rate": validity_rate,
        "mean_valid_score_recomputed": _mean(valid_scores),
    }


def _verdict(
    *,
    valid_seed_rate: float | None,
    invalid_seed_rate: float | None,
    gap_threshold_pp: float,
) -> dict[str, Any]:
    if valid_seed_rate is None or invalid_seed_rate is None:
        return {
            "classification": "insufficient_data",
            "gap_pp": None,
            "threshold_pp": float(gap_threshold_pp),
            "message": (
                "Insufficient data to determine pareidolia verdict because one side lacks "
                "a measurable validity rate."
            ),
        }

    gap_pp = abs(float(valid_seed_rate - invalid_seed_rate) * 100.0)
    if gap_pp <= float(gap_threshold_pp):
        return {
            "classification": "pareidolia_confirmed",
            "gap_pp": float(gap_pp),
            "threshold_pp": float(gap_threshold_pp),
            "message": (
                "Pareidolia confirmed: grammar is uniformly permissive. "
                "89% reflects grammar properties, not seed-specific narrative richness."
            ),
        }
    return {
        "classification": "pareidolia_rejected",
        "gap_pp": float(gap_pp),
        "threshold_pp": float(gap_threshold_pp),
        "message": (
            "Pareidolia rejected: mid-arc validity rate is seed-dependent. "
            "The 89% in invalid seeds reflects genuine narrative landscape properties."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-rate test for Diana mid-arc TP validity.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument("--invalid-baseline", type=str, default=str(DEFAULT_INVALID_BASELINE_PATH))
    parser.add_argument("--invalid-mean-score", type=float, default=DEFAULT_INVALID_VALID_SCORE)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    invalid_baseline_path = _resolve_path(args.invalid_baseline)
    invalid_baseline = _load_invalid_baseline(invalid_baseline_path)

    evolutions = _evolution_profiles()["full"]

    per_seed_results: list[dict[str, Any]] = []
    all_valid_scores: list[float] = []
    per_seed_validity_rates: list[float] = []

    total_candidates_tested = 0
    total_valid_arcs_found = 0
    seeds_with_zero_midarc_events = 0

    for seed in VALID_SEEDS:
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

        midarc_event_type_counts: dict[str, int] = {}
        for _candidate_index, event, _pos in midarc_candidates:
            event_type = str(event.type.value)
            midarc_event_type_counts[event_type] = int(midarc_event_type_counts.get(event_type, 0) + 1)

        candidate_rows: list[dict[str, Any]] = []
        valid_arcs_found = 0
        valid_tp_type_counts: dict[str, int] = {}
        seed_valid_scores: list[float] = []

        for candidate_index, event, pos in midarc_candidates:
            attempts = _candidate_attempts(
                involved_events=diana_events,
                candidate_index=candidate_index,
                total_sim_time=total_sim_time,
            )
            best = _best_attempt(attempts)
            has_valid = any(bool(row["valid"]) for row in attempts)
            if has_valid:
                valid_arcs_found += 1
                event_type = str(event.type.value)
                valid_tp_type_counts[event_type] = int(valid_tp_type_counts.get(event_type, 0) + 1)
                if best is not None and best.get("score") is not None:
                    seed_valid_scores.append(float(best["score"]))

            candidate_rows.append(
                {
                    "event_id": str(event.id),
                    "event_type": str(event.type.value),
                    "tick_id": int(event.tick_id),
                    "global_pos": float(pos),
                    "attempts_tested": int(len(attempts)),
                    "has_valid_arc": bool(has_valid),
                    "best_arc": best,
                }
            )

        candidates_tested = int(len(candidate_rows))
        validity_rate = (
            float(valid_arcs_found / candidates_tested)
            if candidates_tested > 0
            else 0.0
        )

        per_seed_results.append(
            {
                "seed": int(seed),
                "simulation_end_tick": int(simulation_end_tick),
                "diana_involved_events_total": int(len(diana_events)),
                "midarc_candidate_pool_size": int(len(midarc_candidates)),
                "midarc_event_type_counts": dict(sorted(midarc_event_type_counts.items())),
                "candidates_tested": int(candidates_tested),
                "valid_arcs_found": int(valid_arcs_found),
                "validity_rate": float(validity_rate),
                "valid_tp_event_type_counts": dict(sorted(valid_tp_type_counts.items())),
                "mean_valid_score": _mean(seed_valid_scores),
                "candidates": candidate_rows,
            }
        )

        total_candidates_tested += int(candidates_tested)
        total_valid_arcs_found += int(valid_arcs_found)
        all_valid_scores.extend(seed_valid_scores)
        per_seed_validity_rates.append(float(validity_rate))
        if len(midarc_candidates) == 0:
            seeds_with_zero_midarc_events += 1

    overall_validity_rate = (
        float(total_valid_arcs_found / total_candidates_tested)
        if total_candidates_tested > 0
        else None
    )
    mean_per_seed_validity_rate = _mean(per_seed_validity_rates)
    mean_valid_score = _mean(all_valid_scores)

    distribution = {
        "min": _percentile(per_seed_validity_rates, 0.0),
        "p25": _percentile(per_seed_validity_rates, 0.25),
        "median": _percentile(per_seed_validity_rates, 0.5),
        "p75": _percentile(per_seed_validity_rates, 0.75),
        "max": _percentile(per_seed_validity_rates, 1.0),
    }

    invalid_rate = invalid_baseline.get("validity_rate")
    verdict = _verdict(
        valid_seed_rate=overall_validity_rate,
        invalid_seed_rate=float(invalid_rate) if invalid_rate is not None else None,
        gap_threshold_pp=PAREIDOLIA_GAP_THRESHOLD_PP,
    )

    print()
    print("=== BASE RATE VALIDITY: DIANA MID-ARC BRUTE FORCE (41 valid seeds) ===")
    print()
    print(f"Total mid-arc candidates tested:  {int(total_candidates_tested)}")
    print(f"Total valid arcs found:           {int(total_valid_arcs_found)}")
    print(
        f"Overall validity rate:            {_fmt_percent(overall_validity_rate)}  "
        f"(compare: invalid seeds = {_fmt_percent(invalid_rate)})"
    )
    print()
    print("Per-seed validity rate distribution:")
    print(
        f"  min={_fmt_percent(distribution['min'])}  "
        f"p25={_fmt_percent(distribution['p25'])}  "
        f"median={_fmt_percent(distribution['median'])}  "
        f"p75={_fmt_percent(distribution['p75'])}  "
        f"max={_fmt_percent(distribution['max'])}"
    )
    print()
    print(
        f"Mean score of valid mid-arc arcs:  {_fmt_score(mean_valid_score)}  "
        f"(compare: invalid seeds = {_fmt_score(float(args.invalid_mean_score))})"
    )
    print()
    print("=== COMPARISON ===")
    print("                    Invalid seeds (9)    Valid seeds (41)")
    print(
        f"Candidates tested:  {int(invalid_baseline.get('candidates_tested', 0)):<20}"
        f"{int(total_candidates_tested)}"
    )
    print(
        f"Valid arcs:         {int(invalid_baseline.get('valid_arcs_found', 0)):<20}"
        f"{int(total_valid_arcs_found)}"
    )
    print(
        f"Validity rate:      {_fmt_percent(invalid_rate):<20}"
        f"{_fmt_percent(overall_validity_rate)}"
    )
    print(
        f"Mean valid score:   {_fmt_score(float(args.invalid_mean_score)):<20}"
        f"{_fmt_score(mean_valid_score)}"
    )
    print()
    print("=== VERDICT ===")
    print("If valid-seed rate is within 10pp of invalid-seed rate (i.e., 78-98%):")
    print(
        "  -> \"Pareidolia confirmed: grammar is uniformly permissive. "
        "89% reflects grammar properties, not seed-specific narrative richness.\""
    )
    print("If valid-seed rate differs by >10pp:")
    print(
        "  -> \"Pareidolia rejected: mid-arc validity rate is seed-dependent. "
        "The 89% in invalid seeds reflects genuine narrative landscape properties.\""
    )
    print()
    print(verdict["message"])

    payload = {
        "config": {
            "condition": "full_evolution",
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "invalid_seeds": [int(seed) for seed in INVALID_SEEDS],
            "valid_seeds": [int(seed) for seed in VALID_SEEDS],
            "midarc_window": {"lower": MIDARC_LOWER, "upper": MIDARC_UPPER},
            "invalid_baseline_source": str(invalid_baseline_path),
            "invalid_mean_score_reference": float(args.invalid_mean_score),
            "pareidolia_gap_threshold_pp": float(PAREIDOLIA_GAP_THRESHOLD_PP),
            "method_parity": "Imports and uses helper functions directly from scripts.midarc_feasibility",
        },
        "valid_seed_results": per_seed_results,
        "aggregate_valid_seeds": {
            "seed_count": int(len(VALID_SEEDS)),
            "seeds_with_zero_midarc_diana_events": int(seeds_with_zero_midarc_events),
            "total_midarc_candidates_tested": int(total_candidates_tested),
            "total_valid_arcs_found": int(total_valid_arcs_found),
            "overall_validity_rate": overall_validity_rate,
            "mean_per_seed_validity_rate": mean_per_seed_validity_rate,
            "per_seed_validity_rate_distribution": distribution,
            "mean_valid_score": mean_valid_score,
            "valid_score_count": int(len(all_valid_scores)),
        },
        "invalid_baseline": invalid_baseline,
        "comparison": {
            "invalid_seed_count": int(invalid_baseline.get("seed_count", len(INVALID_SEEDS))),
            "valid_seed_count": int(len(VALID_SEEDS)),
            "invalid_candidates_tested": int(invalid_baseline.get("candidates_tested", 0)),
            "valid_candidates_tested": int(total_candidates_tested),
            "invalid_valid_arcs_found": int(invalid_baseline.get("valid_arcs_found", 0)),
            "valid_valid_arcs_found": int(total_valid_arcs_found),
            "invalid_validity_rate": invalid_rate,
            "valid_validity_rate": overall_validity_rate,
            "rate_gap_pp": verdict.get("gap_pp"),
            "invalid_mean_valid_score_reference": float(args.invalid_mean_score),
            "invalid_mean_valid_score_recomputed": invalid_baseline.get("mean_valid_score_recomputed"),
            "valid_mean_valid_score": mean_valid_score,
        },
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
