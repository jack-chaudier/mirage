"""Feasible-volume sweep across a parameterized grammar family.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.feasible_volume_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import search_arc
from narrativefield.extraction.arc_validator import GrammarConfig, validate_arc
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.events import BeatType, Event
from scripts.k_sweep_experiment import DINNER_PARTY_AGENTS, _generate_canon_after_b_for_seed, _simulate_story
from scripts.midarc_feasibility import (
    MAX_AFTER,
    MAX_BEFORE,
    MAX_EVENTS,
    MIDARC_LOWER,
    MIDARC_UPPER,
    _beats_with_forced_tp,
    _event_sort_key,
    _global_pos,
    _involves_diana,
)
from scripts.test_goal_evolution import _evolution_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "feasible_volume_sweep.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "feasible_volume_summary.md"
DEFAULT_SEEDS = list(range(1, 51))
EPSILON = 1e-12


@dataclass(frozen=True)
class SweepDefinition:
    label: str
    config: GrammarConfig
    # Keep strict-level default behavior byte-for-byte aligned with current pipeline.
    use_strict_implementation: bool = False


@dataclass(frozen=True)
class CandidateAttempt:
    events: list[Event]
    beats: list[BeatType]


@dataclass(frozen=True)
class MidArcCandidateContext:
    event_id: str
    event_type: str
    global_pos: float
    attempts: list[CandidateAttempt]


@dataclass(frozen=True)
class SeedContext:
    seed: int
    events: list[Event]
    total_sim_time: float | None
    simulation_end_tick: int
    midarc_candidates: list[MidArcCandidateContext]


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _parse_seeds(raw: str) -> list[int]:
    text = raw.strip()
    if "," in text:
        out = [int(item.strip()) for item in text.split(",") if item.strip()]
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


def _validate_with_config(
    *,
    events: list[Event],
    beats: list[BeatType],
    total_sim_time: float | None,
    grammar_config: GrammarConfig | None,
) -> bool:
    if grammar_config is None:
        return bool(validate_arc(events=events, beats=beats, total_sim_time=total_sim_time).valid)
    return bool(
        validate_arc(
            events=events,
            beats=beats,
            total_sim_time=total_sim_time,
            grammar_config=grammar_config,
        ).valid
    )


def _build_candidate_attempts(
    *,
    involved_events: list[Event],
    candidate_index: int,
) -> list[CandidateAttempt]:
    before_pool = involved_events[:candidate_index]
    tp_event = involved_events[candidate_index]
    after_pool = involved_events[candidate_index + 1 :]

    max_before = min(MAX_BEFORE, len(before_pool))
    max_after = min(MAX_AFTER, len(after_pool))
    attempts: list[CandidateAttempt] = []

    for before_count in range(1, max_before + 1):
        for after_count in range(1, max_after + 1):
            selected = before_pool[-before_count:] + [tp_event] + after_pool[:after_count]
            if len(selected) < 4 or len(selected) > MAX_EVENTS:
                continue
            selected.sort(key=_event_sort_key)
            local_tp_index = selected.index(tp_event)
            beats = _beats_with_forced_tp(selected, local_tp_index)
            attempts.append(CandidateAttempt(events=selected, beats=beats))

    return attempts


def _prepare_seed_contexts(
    *,
    seeds: list[int],
    event_limit: int,
    tick_limit: int,
) -> list[SeedContext]:
    contexts: list[SeedContext] = []
    evolutions = _evolution_profiles()["full"]

    total = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        print(f"[prep {idx:03d}/{total:03d}] seed={seed}", flush=True)
        canon_after_b = _generate_canon_after_b_for_seed(
            seed=seed,
            event_limit=event_limit,
            tick_limit=tick_limit,
        )
        story = _simulate_story(
            label="full_evolution",
            seed=seed,
            loaded_canon=canon_after_b,
            tick_limit=tick_limit,
            event_limit=event_limit,
            evolutions=evolutions,
        )
        parsed = parse_simulation_output(story.payload)
        metrics_output = run_metrics_pipeline(parsed)

        events = list(metrics_output.events)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
        simulation_end_tick = max((int(event.tick_id) for event in events), default=1)

        diana_events = [event for event in events if _involves_diana(event)]
        diana_events.sort(key=_event_sort_key)

        midarc_candidates: list[MidArcCandidateContext] = []
        for event_index, event in enumerate(diana_events):
            pos = _global_pos(event, simulation_end_tick)
            if MIDARC_LOWER <= pos <= MIDARC_UPPER:
                attempts = _build_candidate_attempts(
                    involved_events=diana_events,
                    candidate_index=event_index,
                )
                midarc_candidates.append(
                    MidArcCandidateContext(
                        event_id=str(event.id),
                        event_type=str(event.type.value),
                        global_pos=float(pos),
                        attempts=attempts,
                    )
                )

        contexts.append(
            SeedContext(
                seed=int(seed),
                events=events,
                total_sim_time=total_sim_time,
                simulation_end_tick=int(simulation_end_tick),
                midarc_candidates=midarc_candidates,
            )
        )

    return contexts


def _measure_feasibility(
    *,
    seed_contexts: list[SeedContext],
    grammar_config: GrammarConfig | None,
) -> dict[str, Any]:
    total_candidates = 0
    valid_candidates = 0
    per_seed_rates: list[float] = []
    per_seed: list[dict[str, Any]] = []

    for context in seed_contexts:
        seed_total = len(context.midarc_candidates)
        seed_valid = 0

        for candidate in context.midarc_candidates:
            has_valid_arc = False
            for attempt in candidate.attempts:
                if _validate_with_config(
                    events=attempt.events,
                    beats=attempt.beats,
                    total_sim_time=context.total_sim_time,
                    grammar_config=grammar_config,
                ):
                    has_valid_arc = True
                    break
            if has_valid_arc:
                seed_valid += 1

        seed_rate = float(seed_valid / seed_total) if seed_total > 0 else 0.0
        per_seed_rates.append(seed_rate)
        per_seed.append(
            {
                "seed": int(context.seed),
                "total_candidates": int(seed_total),
                "valid_candidates": int(seed_valid),
                "rate": float(seed_rate),
            }
        )

        total_candidates += seed_total
        valid_candidates += seed_valid

    rate = float(valid_candidates / total_candidates) if total_candidates > 0 else 0.0
    return {
        "total_candidates": int(total_candidates),
        "valid_candidates": int(valid_candidates),
        "rate": float(rate),
        "per_seed_rates": per_seed_rates,
        "per_seed": per_seed,
    }


def _search_with_config(
    *,
    events: list[Event],
    protagonist: str,
    total_sim_time: float | None,
    grammar_config: GrammarConfig | None,
) -> dict[str, Any]:
    result = search_arc(
        all_events=events,
        protagonist=protagonist,
        max_events=20,
        total_sim_time=total_sim_time,
        grammar_config=grammar_config,
    )
    is_valid = bool(result.validation.valid)
    score = None
    if is_valid:
        arc_score = result.arc_score or score_arc(result.events, result.beats)
        score = float(arc_score.composite)

    return {
        "valid": bool(is_valid),
        "score": score,
        "event_ids": [str(event.id) for event in result.events],
        "beats": [str(beat.value) for beat in result.beats],
        "violations": [str(v) for v in result.validation.violations],
    }


def _tp_global_positions(
    *,
    event_ids: list[str],
    beats: list[str],
    by_id: dict[str, Event],
    simulation_end_tick: int,
) -> list[float]:
    positions: list[float] = []
    denom = max(int(simulation_end_tick), 1)
    for event_id, beat in zip(event_ids, beats):
        if beat != BeatType.TURNING_POINT.value:
            continue
        event = by_id.get(event_id)
        if event is None:
            continue
        positions.append(float(int(event.tick_id) / denom))
    return positions


def _measure_search_quality(
    *,
    seed_contexts: list[SeedContext],
    grammar_config: GrammarConfig | None,
) -> dict[str, Any]:
    seed_mean_scores: list[float] = []
    seed_va_scores: list[float] = []
    all_valid_count = 0
    per_agent_valid_count: dict[str, int] = {agent: 0 for agent in DINNER_PARTY_AGENTS}
    per_seed: list[dict[str, Any]] = []
    diana_invalid_count = 0
    diana_invalid_seeds: list[int] = []
    diana_invalid_tp_positions: list[float] = []

    total_seeds = len(seed_contexts)

    for context in seed_contexts:
        by_id = {str(event.id): event for event in context.events}
        per_agent: dict[str, dict[str, Any]] = {}
        valid_scores: list[float] = []
        valid_count = 0

        for agent in DINNER_PARTY_AGENTS:
            row = _search_with_config(
                events=context.events,
                protagonist=agent,
                total_sim_time=context.total_sim_time,
                grammar_config=grammar_config,
            )
            tp_positions = _tp_global_positions(
                event_ids=row["event_ids"],
                beats=row["beats"],
                by_id=by_id,
                simulation_end_tick=context.simulation_end_tick,
            )
            row["turning_point_global_positions"] = tp_positions
            per_agent[agent] = row

            if bool(row["valid"]):
                valid_count += 1
                per_agent_valid_count[agent] += 1
                if row["score"] is not None:
                    valid_scores.append(float(row["score"]))

        mean_q = float(_mean(valid_scores))
        va = float(mean_q * (valid_count / float(len(DINNER_PARTY_AGENTS)))) if valid_count > 0 else 0.0
        all_valid = valid_count == len(DINNER_PARTY_AGENTS)

        if all_valid:
            all_valid_count += 1

        diana_row = per_agent["diana"]
        if not bool(diana_row["valid"]):
            diana_invalid_count += 1
            diana_invalid_seeds.append(int(context.seed))
            tp_positions = list(diana_row.get("turning_point_global_positions") or [])
            if tp_positions:
                diana_invalid_tp_positions.append(float(tp_positions[0]))

        per_seed.append(
            {
                "seed": int(context.seed),
                "valid_arc_count": int(valid_count),
                "mean_q": float(mean_q),
                "va": float(va),
                "all_valid": bool(all_valid),
                "per_agent": per_agent,
            }
        )
        seed_mean_scores.append(mean_q)
        seed_va_scores.append(va)

    return {
        "mean_q": float(_mean(seed_mean_scores)),
        "va": float(_mean(seed_va_scores)),
        "all_valid_rate": float(all_valid_count / total_seeds) if total_seeds > 0 else 0.0,
        "per_agent_validity": {
            agent: float(per_agent_valid_count[agent] / total_seeds) if total_seeds > 0 else 0.0
            for agent in DINNER_PARTY_AGENTS
        },
        "diana_invalid_count": int(diana_invalid_count),
        "diana_invalid_seeds": diana_invalid_seeds,
        "diana_invalid_tp_positions": diana_invalid_tp_positions,
        "per_seed": per_seed,
    }


def _regularization_path() -> list[SweepDefinition]:
    return [
        SweepDefinition(
            label="extra_strict",
            config=GrammarConfig(
                min_development_beats=2,
                max_phase_regressions=0,
                max_turning_points=1,
                min_beat_count=6,
                max_beat_count=18,
                min_timespan_fraction=0.20,
                protagonist_coverage=0.70,
            ),
        ),
        SweepDefinition(
            label="strict",
            config=GrammarConfig(),
            use_strict_implementation=True,
        ),
        SweepDefinition(
            label="slightly_relaxed",
            config=GrammarConfig(
                min_development_beats=1,
                max_phase_regressions=0,
                max_turning_points=1,
                min_beat_count=3,
                max_beat_count=25,
                min_timespan_fraction=0.10,
                protagonist_coverage=0.50,
            ),
        ),
        SweepDefinition(
            label="moderately_relaxed",
            config=GrammarConfig(
                min_development_beats=1,
                max_phase_regressions=1,
                max_turning_points=2,
                min_beat_count=3,
                max_beat_count=25,
                min_timespan_fraction=0.10,
                protagonist_coverage=0.40,
            ),
        ),
        SweepDefinition(
            label="substantially_relaxed",
            config=GrammarConfig(
                min_development_beats=1,
                max_phase_regressions=1,
                max_turning_points=2,
                min_beat_count=3,
                max_beat_count=30,
                min_timespan_fraction=0.05,
                protagonist_coverage=0.30,
            ),
        ),
        SweepDefinition(
            label="very_relaxed",
            config=GrammarConfig(
                min_development_beats=0,
                max_phase_regressions=1,
                max_turning_points=2,
                min_beat_count=2,
                max_beat_count=30,
                min_timespan_fraction=0.05,
                protagonist_coverage=0.20,
            ),
        ),
        SweepDefinition(
            label="nearly_vacuous",
            config=GrammarConfig(
                min_development_beats=0,
                max_phase_regressions=2,
                max_turning_points=3,
                min_beat_count=2,
                max_beat_count=999,
                min_timespan_fraction=0.0,
                protagonist_coverage=0.0,
            ),
        ),
        SweepDefinition(
            label="vacuous",
            config=GrammarConfig(
                min_development_beats=0,
                max_phase_regressions=999,
                max_turning_points=999,
                min_beat_count=1,
                max_beat_count=999,
                min_timespan_fraction=0.0,
                protagonist_coverage=0.0,
            ),
        ),
    ]


def _single_dimension_sweeps() -> dict[str, list[SweepDefinition]]:
    strict = GrammarConfig()
    return {
        "min_development_beats": [
            SweepDefinition(
                label=f"min_development_beats={value}",
                config=replace(strict, min_development_beats=int(value)),
            )
            for value in (0, 1, 2, 3)
        ],
        "max_phase_regressions": [
            SweepDefinition(
                label=f"max_phase_regressions={value}",
                config=replace(strict, max_phase_regressions=int(value)),
            )
            for value in (0, 1, 2, 999)
        ],
        "protagonist_coverage": [
            SweepDefinition(
                label=f"protagonist_coverage={value:.2f}",
                config=replace(strict, protagonist_coverage=float(value)),
            )
            for value in (0.0, 0.20, 0.40, 0.60, 0.80)
        ],
        "min_timespan_fraction": [
            SweepDefinition(
                label=f"min_timespan_fraction={value:.2f}",
                config=replace(strict, min_timespan_fraction=float(value)),
            )
            for value in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
        ],
    }


def _evaluate_definition(
    *,
    definition: SweepDefinition,
    seed_contexts: list[SeedContext],
) -> dict[str, Any]:
    active_config = None if definition.use_strict_implementation else definition.config
    feasibility = _measure_feasibility(
        seed_contexts=seed_contexts,
        grammar_config=active_config,
    )
    search_quality = _measure_search_quality(
        seed_contexts=seed_contexts,
        grammar_config=active_config,
    )
    return {
        "config": definition.config.to_dict(),
        "feasibility": feasibility,
        "search_quality": search_quality,
    }


def _group_feasibility_rate(
    *,
    per_seed_rows: list[dict[str, Any]],
    seed_filter: set[int],
) -> float:
    total_candidates = 0
    valid_candidates = 0
    for row in per_seed_rows:
        seed = int(row["seed"])
        if seed not in seed_filter:
            continue
        total_candidates += int(row["total_candidates"])
        valid_candidates += int(row["valid_candidates"])
    if total_candidates <= 0:
        return 0.0
    return float(valid_candidates / total_candidates)


def _build_verification(
    *,
    regularization_path: list[dict[str, Any]],
) -> dict[str, Any]:
    level1 = next(row for row in regularization_path if int(row["level"]) == 1)
    level4 = next(row for row in regularization_path if int(row["level"]) == 4)

    level1_search = level1["search_quality"]
    level1_feasibility = level1["feasibility"]

    diana_invalid_seeds = set(int(seed) for seed in level1_search["diana_invalid_seeds"])
    all_seeds = set(int(row["seed"]) for row in level1_feasibility["per_seed"])
    diana_valid_seeds = all_seeds - diana_invalid_seeds

    invalid_seed_feasibility = _group_feasibility_rate(
        per_seed_rows=level1_feasibility["per_seed"],
        seed_filter=diana_invalid_seeds,
    )
    valid_seed_feasibility = _group_feasibility_rate(
        per_seed_rows=level1_feasibility["per_seed"],
        seed_filter=diana_valid_seeds,
    )

    feasibility_rates = [float(row["feasibility"]["rate"]) for row in regularization_path]
    all_valid_rates = [float(row["search_quality"]["all_valid_rate"]) for row in regularization_path]
    feasibility_monotonic = all(
        feasibility_rates[idx + 1] + EPSILON >= feasibility_rates[idx]
        for idx in range(len(feasibility_rates) - 1)
    )
    all_valid_not_monotonic_increasing = any(
        all_valid_rates[idx + 1] + EPSILON < all_valid_rates[idx]
        for idx in range(len(all_valid_rates) - 1)
    )

    return {
        "level1_strict": {
            "diana_invalid_count": int(level1_search["diana_invalid_count"]),
            "invalid_seed_feasibility_rate": float(invalid_seed_feasibility),
            "valid_seed_feasibility_rate": float(valid_seed_feasibility),
            "mean_q": float(level1_search["mean_q"]),
            "va": float(level1_search["va"]),
            "all_valid_rate": float(level1_search["all_valid_rate"]),
        },
        "level4_substantially_relaxed": {
            "all_valid_rate": float(level4["search_quality"]["all_valid_rate"]),
            "mean_q": float(level4["search_quality"]["mean_q"]),
            "va": float(level4["search_quality"]["va"]),
        },
        "monotonicity_checks": {
            "feasibility_non_decreasing": bool(feasibility_monotonic),
            "all_valid_not_monotonic_increasing": bool(all_valid_not_monotonic_increasing),
        },
    }


def _largest_single_dimension_effect(
    *,
    sweeps: dict[str, list[dict[str, Any]]],
    strict_all_valid_rate: float,
) -> dict[str, Any]:
    effect_rows: list[dict[str, Any]] = []
    for dimension, rows in sweeps.items():
        deltas = [
            abs(float(row["search_quality"]["all_valid_rate"]) - float(strict_all_valid_rate))
            for row in rows
        ]
        max_delta = max(deltas) if deltas else 0.0
        effect_rows.append(
            {
                "dimension": dimension,
                "max_abs_all_valid_delta": float(max_delta),
            }
        )
    effect_rows.sort(key=lambda row: (-float(row["max_abs_all_valid_delta"]), row["dimension"]))
    return effect_rows[0] if effect_rows else {"dimension": "n/a", "max_abs_all_valid_delta": 0.0}


def _phase_transition_summary(regularization_path: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for left, right in zip(regularization_path, regularization_path[1:]):
        delta_all_valid = float(right["search_quality"]["all_valid_rate"]) - float(
            left["search_quality"]["all_valid_rate"]
        )
        delta_mean_q = float(right["search_quality"]["mean_q"]) - float(left["search_quality"]["mean_q"])
        rows.append(
            {
                "from_level": int(left["level"]),
                "to_level": int(right["level"]),
                "delta_all_valid_rate": float(delta_all_valid),
                "delta_mean_q": float(delta_mean_q),
            }
        )

    if not rows:
        return {"description": "No transition data (insufficient levels).", "transitions": []}

    steepest_drop = min(rows, key=lambda row: float(row["delta_all_valid_rate"]))
    magnitude = abs(float(steepest_drop["delta_all_valid_rate"]))
    if magnitude >= 0.20:
        description = (
            f"Sharp transition around levels {steepest_drop['from_level']} -> "
            f"{steepest_drop['to_level']} (all-valid drop {magnitude * 100.0:.1f}pp)."
        )
    elif magnitude >= 0.10:
        description = (
            f"Moderate inflection around levels {steepest_drop['from_level']} -> "
            f"{steepest_drop['to_level']} (all-valid drop {magnitude * 100.0:.1f}pp)."
        )
    else:
        description = "Decline appears gradual with no sharp all-valid phase boundary."

    return {
        "description": description,
        "steepest_drop": steepest_drop,
        "transitions": rows,
    }


def _build_summary_markdown(
    *,
    regularization_path: list[dict[str, Any]],
    sweeps: dict[str, list[dict[str, Any]]],
    verification: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Feasible Volume Sweep Summary")
    lines.append("")
    lines.append("| Level | Label | Feasibility Rate | Mean Q | VA | All-Valid Rate | Diana Invalid Count |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in regularization_path:
        lines.append(
            "| "
            f"{int(row['level'])} | {str(row['label'])} | "
            f"{float(row['feasibility']['rate']):.3f} | "
            f"{float(row['search_quality']['mean_q']):.3f} | "
            f"{float(row['search_quality']['va']):.3f} | "
            f"{float(row['search_quality']['all_valid_rate']):.3f} | "
            f"{int(row['search_quality']['diana_invalid_count'])} |"
        )
    lines.append("")

    strict_all_valid = float(next(row for row in regularization_path if int(row["level"]) == 1)["search_quality"]["all_valid_rate"])
    strongest = _largest_single_dimension_effect(
        sweeps=sweeps,
        strict_all_valid_rate=strict_all_valid,
    )
    lines.append("## Largest Single-Dimension Effect")
    lines.append(
        f"- Largest effect by `all_valid_rate` delta: `{strongest['dimension']}` "
        f"(max |delta| = {float(strongest['max_abs_all_valid_delta']) * 100.0:.1f}pp vs strict)."
    )
    lines.append("")

    phase_summary = _phase_transition_summary(regularization_path)
    lines.append("## Phase-Transition Read")
    lines.append(f"- {phase_summary['description']}")
    lines.append("")

    level1 = verification["level1_strict"]
    level4 = verification["level4_substantially_relaxed"]
    checks = verification["monotonicity_checks"]
    lines.append("## Verification Checks")
    lines.append(
        "- Level 1 strict: "
        f"Diana invalid={int(level1['diana_invalid_count'])}/50, "
        f"feasibility(invalid seeds)={float(level1['invalid_seed_feasibility_rate']) * 100.0:.1f}%, "
        f"feasibility(valid seeds)={float(level1['valid_seed_feasibility_rate']) * 100.0:.1f}%, "
        f"mean Q={float(level1['mean_q']):.3f}, VA={float(level1['va']):.3f}, "
        f"all-valid={float(level1['all_valid_rate']) * 100.0:.1f}%."
    )
    lines.append(
        "- Level 4 substantially relaxed: "
        f"all-valid={float(level4['all_valid_rate']) * 100.0:.1f}%, "
        f"mean Q={float(level4['mean_q']):.3f}, VA={float(level4['va']):.3f}."
    )
    lines.append(
        "- Feasibility monotonic non-decreasing across regularization path: "
        f"{'yes' if bool(checks['feasibility_non_decreasing']) else 'no'}."
    )
    lines.append(
        "- All-valid is not monotonic increasing across regularization path: "
        f"{'yes' if bool(checks['all_valid_not_monotonic_increasing']) else 'no'}."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Feasible-volume sweep across parameterized grammar levels.")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--output-summary", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_json_path = _resolve_path(args.output_json)
    output_summary_path = _resolve_path(args.output_summary)
    seeds = _parse_seeds(args.seeds)

    regularization_defs = _regularization_path()
    single_dimension_defs = _single_dimension_sweeps()
    single_dimension_count = sum(len(rows) for rows in single_dimension_defs.values())
    total_config_count = len(regularization_defs) + single_dimension_count

    print()
    print("=== FEASIBLE VOLUME SWEEP ===")
    print()
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Regularization path levels: {len(regularization_defs)}")
    print(f"Single-dimension configs: {single_dimension_count}")
    print(f"Total grammar configs: {total_config_count}")
    print()
    print("Estimated deterministic workload:")
    print(f"- Simulation + metrics runs: {len(seeds)}")
    print(f"- Search runs (all six agents): {len(seeds) * total_config_count * len(DINNER_PARTY_AGENTS)}")
    print("- Feasibility validations: data-dependent (typically ~2M validator calls at this scale)")
    print()

    prep_start = time.time()
    seed_contexts = _prepare_seed_contexts(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    prep_elapsed = time.time() - prep_start
    print(f"Prepared seed contexts in {prep_elapsed:.1f}s.")
    print()

    run_start = time.time()
    completed = 0
    regularization_rows: list[dict[str, Any]] = []
    single_dimension_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in single_dimension_defs}

    for level, definition in enumerate(regularization_defs):
        completed += 1
        elapsed = time.time() - run_start
        eta = (elapsed / completed) * (total_config_count - completed) if completed > 0 else 0.0
        print(
            f"[{completed:02d}/{total_config_count:02d}] regularization level={level} "
            f"label={definition.label} (elapsed={elapsed:.1f}s, eta={eta:.1f}s)",
            flush=True,
        )
        row = _evaluate_definition(
            definition=definition,
            seed_contexts=seed_contexts,
        )
        row["level"] = int(level)
        row["label"] = str(definition.label)
        regularization_rows.append(row)

    for dimension, definitions in single_dimension_defs.items():
        for definition in definitions:
            completed += 1
            elapsed = time.time() - run_start
            eta = (elapsed / completed) * (total_config_count - completed) if completed > 0 else 0.0
            print(
                f"[{completed:02d}/{total_config_count:02d}] single-dim {dimension} "
                f"{definition.label} (elapsed={elapsed:.1f}s, eta={eta:.1f}s)",
                flush=True,
            )
            row = _evaluate_definition(
                definition=definition,
                seed_contexts=seed_contexts,
            )
            row["label"] = definition.label
            single_dimension_rows[dimension].append(row)

    verification = _build_verification(regularization_path=regularization_rows)
    phase_summary = _phase_transition_summary(regularization_rows)
    strongest = _largest_single_dimension_effect(
        sweeps=single_dimension_rows,
        strict_all_valid_rate=float(
            next(row for row in regularization_rows if int(row["level"]) == 1)["search_quality"]["all_valid_rate"]
        ),
    )

    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": [int(seed) for seed in seeds],
            "condition": "full_evolution_k6",
            "agent_focus": "Diana",
            "search_agents": ["all_six"],
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "midarc_window": {"lower": float(MIDARC_LOWER), "upper": float(MIDARC_UPPER)},
            "manual_arc_construction": {
                "max_events": int(MAX_EVENTS),
                "max_before": int(MAX_BEFORE),
                "max_after": int(MAX_AFTER),
            },
        },
        "regularization_path": regularization_rows,
        "single_dimension_sweeps": single_dimension_rows,
        "analysis": {
            "largest_single_dimension_effect": strongest,
            "phase_transition": phase_summary,
        },
        "verification": verification,
    }

    summary_text = _build_summary_markdown(
        regularization_path=regularization_rows,
        sweeps=single_dimension_rows,
        verification=verification,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_text, encoding="utf-8")

    total_elapsed = time.time() - run_start
    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(f"Sweep runtime (excluding prep): {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
