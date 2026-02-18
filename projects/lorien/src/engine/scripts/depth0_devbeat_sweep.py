"""Depth-0 vs depth-2 min_development_beats sweep.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.depth0_devbeat_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from narrativefield.extraction.arc_validator import GrammarConfig
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.feasible_volume_sweep import (
    MidArcCandidateContext,
    SeedContext,
    SweepDefinition,
    _build_candidate_attempts,
    _measure_feasibility,
    _measure_search_quality,
    _prepare_seed_contexts as _prepare_seed_contexts_depth2,
)
from scripts.k_sweep_experiment import _simulate_story
from scripts.midarc_feasibility import MIDARC_LOWER, MIDARC_UPPER, _event_sort_key, _global_pos, _involves_diana
from scripts.test_goal_evolution import _evolution_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "depth0_devbeat_sweep.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "depth0_devbeat_summary.md"
DEFAULT_SEEDS = list(range(1, 51))
EPSILON = 1e-12

DEVBEAT_SWEEP: list[SweepDefinition] = [
    SweepDefinition(label="dev=0", config=replace(GrammarConfig(), min_development_beats=0)),
    SweepDefinition(label="dev=1 (strict)", config=GrammarConfig(), use_strict_implementation=True),
    SweepDefinition(label="dev=2", config=replace(GrammarConfig(), min_development_beats=2)),
    SweepDefinition(label="dev=3", config=replace(GrammarConfig(), min_development_beats=3)),
]


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


def _prepare_seed_contexts_depth0(
    *,
    seeds: list[int],
    event_limit: int,
    tick_limit: int,
) -> list[SeedContext]:
    """Prepare depth-0 seed contexts (fresh canon) with full evolution profiles."""
    contexts: list[SeedContext] = []
    evolutions = _evolution_profiles()["full"]

    total = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        print(f"[prep-d0 {idx:03d}/{total:03d}] seed={seed}", flush=True)
        story = _simulate_story(
            label="full_evolution_d0",
            seed=seed,
            loaded_canon=None,
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


def _evaluate_definition(
    *,
    definition: SweepDefinition,
    seed_contexts: list[SeedContext],
) -> dict[str, Any]:
    active_config = None if definition.use_strict_implementation else definition.config
    feasibility = _measure_feasibility(seed_contexts=seed_contexts, grammar_config=active_config)
    search_quality = _measure_search_quality(seed_contexts=seed_contexts, grammar_config=active_config)
    return {
        "min_development_beats": int(definition.config.min_development_beats),
        "label": str(definition.label),
        "feasibility": {
            "rate": float(feasibility["rate"]),
            "total_candidates": int(feasibility["total_candidates"]),
            "valid_candidates": int(feasibility["valid_candidates"]),
        },
        "search_quality": {
            "mean_q": float(search_quality["mean_q"]),
            "va": float(search_quality["va"]),
            "all_valid_rate": float(search_quality["all_valid_rate"]),
            "diana_invalid_count": int(search_quality["diana_invalid_count"]),
            "per_agent_validity": dict(search_quality["per_agent_validity"]),
        },
    }


def _cliff_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    by_dev = {int(row["min_development_beats"]): row for row in results}
    if 0 not in by_dev or 1 not in by_dev:
        raise ValueError("Sweep results must include min_development_beats=0 and 1.")

    dev0 = float(by_dev[0]["search_quality"]["all_valid_rate"])
    dev1 = float(by_dev[1]["search_quality"]["all_valid_rate"])
    delta_pp = (dev0 - dev1) * 100.0
    return {
        "dev0_all_valid": float(dev0),
        "dev1_all_valid": float(dev1),
        "delta_pp": float(delta_pp),
    }


def _interpret_cliff(*, depth0_delta_pp: float, depth2_delta_pp: float) -> str:
    if abs(depth0_delta_pp - depth2_delta_pp) < 5.0:
        return "search_algorithmic"
    if depth0_delta_pp < (depth2_delta_pp - 5.0):
        return "density_amplified"
    return "canon_scaffolded"


def _safe_ratio(*, numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= EPSILON:
        return None
    return float(numerator / denominator)


def _results_by_dev(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["min_development_beats"]): row for row in results}


def _build_summary_markdown(
    *,
    depth0_results: list[dict[str, Any]],
    depth2_results: list[dict[str, Any]],
    interpretation: str,
) -> str:
    d0 = _results_by_dev(depth0_results)
    d2 = _results_by_dev(depth2_results)

    lines: list[str] = []
    lines.append("# Depth-0 vs Depth-2: min_development_beats Sweep")
    lines.append("")
    lines.append("| min_dev_beats | D0 All-Valid | D0 VA | D0 Diana Inv | D2 All-Valid | D2 VA | D2 Diana Inv |")
    lines.append("|---|---|---|---|---|---|---|")
    for dev in (0, 1, 2, 3):
        d0_row = d0[dev]
        d2_row = d2[dev]
        lines.append(
            "| "
            f"{dev} | "
            f"{float(d0_row['search_quality']['all_valid_rate']) * 100.0:.1f}% | "
            f"{float(d0_row['search_quality']['va']):.3f} | "
            f"{int(d0_row['search_quality']['diana_invalid_count'])} | "
            f"{float(d2_row['search_quality']['all_valid_rate']) * 100.0:.1f}% | "
            f"{float(d2_row['search_quality']['va']):.3f} | "
            f"{int(d2_row['search_quality']['diana_invalid_count'])} |"
        )
    lines.append("")

    d0_cliff = _cliff_stats(depth0_results)
    d2_cliff = _cliff_stats(depth2_results)
    lines.append("## 0→1 Cliff Comparison")
    lines.append(
        "- Depth-0: "
        f"{float(d0_cliff['dev0_all_valid']) * 100.0:.1f}% -> "
        f"{float(d0_cliff['dev1_all_valid']) * 100.0:.1f}% "
        f"(delta: {float(d0_cliff['delta_pp']):.1f}pp)"
    )
    lines.append(
        "- Depth-2: "
        f"{float(d2_cliff['dev0_all_valid']) * 100.0:.1f}% -> "
        f"{float(d2_cliff['dev1_all_valid']) * 100.0:.1f}% "
        f"(delta: {float(d2_cliff['delta_pp']):.1f}pp)"
    )
    lines.append(f"- Interpretation: {interpretation}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-0 vs depth-2 min_development_beats single-dimension sweep.")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--output-summary", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_json_path = _resolve_path(args.output_json)
    output_summary_path = _resolve_path(args.output_summary)
    seeds = _parse_seeds(args.seeds) if args.seeds else list(DEFAULT_SEEDS)

    print()
    print("=== DEPTH-0 VS DEPTH-2 DEV-BEAT SWEEP ===")
    print()
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"Configs: {len(DEVBEAT_SWEEP)} (min_development_beats in [0,1,2,3])")
    print()

    prep_start = time.time()
    depth0_contexts = _prepare_seed_contexts_depth0(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    depth0_prep_elapsed = time.time() - prep_start
    print(f"Prepared depth-0 contexts in {depth0_prep_elapsed:.1f}s.")

    prep_start = time.time()
    depth2_contexts = _prepare_seed_contexts_depth2(
        seeds=seeds,
        event_limit=int(args.event_limit),
        tick_limit=int(args.tick_limit),
    )
    depth2_prep_elapsed = time.time() - prep_start
    print(f"Prepared depth-2 contexts in {depth2_prep_elapsed:.1f}s.")
    print()

    run_start = time.time()
    depth0_results: list[dict[str, Any]] = []
    depth2_results: list[dict[str, Any]] = []

    for idx, definition in enumerate(DEVBEAT_SWEEP, start=1):
        print(f"[depth0 {idx:02d}/{len(DEVBEAT_SWEEP):02d}] {definition.label}", flush=True)
        depth0_results.append(_evaluate_definition(definition=definition, seed_contexts=depth0_contexts))

    for idx, definition in enumerate(DEVBEAT_SWEEP, start=1):
        print(f"[depth2 {idx:02d}/{len(DEVBEAT_SWEEP):02d}] {definition.label}", flush=True)
        depth2_results.append(_evaluate_definition(definition=definition, seed_contexts=depth2_contexts))

    depth0_cliff = _cliff_stats(depth0_results)
    depth2_cliff = _cliff_stats(depth2_results)
    interpretation = _interpret_cliff(
        depth0_delta_pp=float(depth0_cliff["delta_pp"]),
        depth2_delta_pp=float(depth2_cliff["delta_pp"]),
    )
    cliff_ratio = _safe_ratio(
        numerator=float(depth0_cliff["delta_pp"]),
        denominator=float(depth2_cliff["delta_pp"]),
    )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds": [int(seed) for seed in seeds],
            "experiment": "depth0_vs_depth2_min_development_beats_sweep",
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "runtime_seconds": float(time.time() - run_start),
        },
        "depth0": {
            "results": depth0_results,
        },
        "depth2": {
            "results": depth2_results,
        },
        "comparison": {
            "depth0_cliff": depth0_cliff,
            "depth2_cliff": depth2_cliff,
            "cliff_ratio": cliff_ratio,
            "interpretation": interpretation,
        },
    }

    summary_text = _build_summary_markdown(
        depth0_results=depth0_results,
        depth2_results=depth2_results,
        interpretation=interpretation,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_text, encoding="utf-8")

    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(f"Runtime (excluding prep): {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
