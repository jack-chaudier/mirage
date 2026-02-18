"""Decoupled alpha sweep: continuous vs categorical commitments.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.decoupled_alpha_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import search_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.events import BeatType, Event
from narrativefield.simulation.scenarios import create_dinner_party_world
from scripts.k_sweep_experiment import DINNER_PARTY_AGENTS, _generate_canon_after_b_for_seed, _simulate_story
from scripts.test_goal_evolution import AgentEvolution, _evolution_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_JSON_PATH = OUTPUT_DIR / "decoupled_alpha_sweep.json"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "decoupled_alpha_summary.md"
DEFAULT_PLOTS_PATH = OUTPUT_DIR / "decoupled_alpha_plots.html"

BASELINE_VERIFY_SEEDS = list(range(1, 51))
SWEEP_SEEDS = list(range(1, 21))
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))

BASELINE_INVALID_SEEDS_EXPECTED = [2, 3, 9, 25, 32, 33, 35, 38, 43]
BASELINE_MEAN_Q_ALL_EXPECTED = 0.684
BASELINE_ALL_VALID_EXPECTED = 0.64

PAPER2_VA_EXPECTED = {
    0.0: 0.644,
    0.1: 0.659,
    0.2: 0.597,
    0.3: 0.630,
    0.4: 0.673,
    0.5: 0.684,
    0.6: 0.659,
    0.7: 0.607,
    0.8: 0.623,
    0.9: 0.641,
    1.0: 0.651,
}

EPSILON = 1e-12

SweepName = Literal["coupled", "continuous_only", "categorical_only"]
SWEEPS: tuple[SweepName, ...] = ("coupled", "continuous_only", "categorical_only")


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.var(arr, ddof=0))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=0))


def _event_pos(event: Event, simulation_end_tick: int) -> float:
    return float(int(event.tick_id) / max(int(simulation_end_tick), 1))


def _dev_beat_count(beats: list[BeatType]) -> int:
    return int(sum(1 for beat in beats if beat in {BeatType.COMPLICATION, BeatType.ESCALATION}))


def _tp_info(
    *,
    events: list[Event],
    beats: list[BeatType],
    simulation_end_tick: int,
) -> tuple[float | None, str | None]:
    for event, beat in zip(events, beats):
        if beat == BeatType.TURNING_POINT:
            return (_event_pos(event, simulation_end_tick), str(event.type.value))
    return (None, None)


def _extract_baseline_profiles() -> dict[str, dict[str, Any]]:
    world = create_dinner_party_world()
    profiles: dict[str, dict[str, Any]] = {}

    for agent_id in DINNER_PARTY_AGENTS:
        state = world.agents[agent_id]
        profiles[agent_id] = {
            "goals_scalar": {
                "safety": float(state.goals.safety),
                "status": float(state.goals.status),
                "secrecy": float(state.goals.secrecy),
                "truth_seeking": float(state.goals.truth_seeking),
                "autonomy": float(state.goals.autonomy),
                "loyalty": float(state.goals.loyalty),
            },
            "closeness": {k: float(v) for k, v in sorted(state.goals.closeness.items())},
            "relationships": {
                target: {
                    "trust": float(rel.trust),
                    "affection": float(rel.affection),
                    "obligation": float(rel.obligation),
                }
                for target, rel in sorted(state.relationships.items())
            },
            "commitments": [str(item) for item in state.commitments],
        }

    return profiles


def _evolution_to_profile_dict(evolution: AgentEvolution) -> dict[str, Any]:
    return {
        "goals_scalar": {k: float(v) for k, v in sorted(evolution.goals_scalar.items())},
        "closeness": {k: float(v) for k, v in sorted(evolution.closeness.items())},
        "relationships": {
            target: {attr: float(value) for attr, value in sorted(values.items())}
            for target, values in sorted(evolution.relationships.items())
        },
        "commitments": None if evolution.commitments is None else [str(x) for x in evolution.commitments],
    }


def _copy_evolution(evolution: AgentEvolution) -> AgentEvolution:
    return AgentEvolution(
        goals_scalar={k: float(v) for k, v in evolution.goals_scalar.items()},
        closeness={k: float(v) for k, v in evolution.closeness.items()},
        relationships={
            target: {attr: float(value) for attr, value in values.items()}
            for target, values in evolution.relationships.items()
        },
        commitments=None if evolution.commitments is None else tuple(str(x) for x in evolution.commitments),
    )


def _interpolate_continuous(
    *,
    baseline_profile: dict[str, Any],
    evolved_profile: dict[str, Any],
    alpha: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    a = float(alpha)

    base_goals = baseline_profile.get("goals_scalar") or {}
    evol_goals = evolved_profile.get("goals_scalar") or {}
    goals: dict[str, float] = {}
    for dim in sorted(set(base_goals) | set(evol_goals)):
        base_val = float(base_goals.get(dim, 0.0))
        evol_val = float(evol_goals.get(dim, base_val))
        goals[dim] = (1.0 - a) * base_val + a * evol_val

    base_close = baseline_profile.get("closeness") or {}
    evol_close = evolved_profile.get("closeness") or {}
    closeness: dict[str, float] = {}
    for target in sorted(set(base_close) | set(evol_close)):
        base_val = float(base_close.get(target, 0.0))
        evol_val = float(evol_close.get(target, base_val))
        closeness[target] = (1.0 - a) * base_val + a * evol_val

    base_rels = baseline_profile.get("relationships") or {}
    evol_rels = evolved_profile.get("relationships") or {}
    relationships: dict[str, dict[str, float]] = {}
    for target in sorted(set(base_rels) | set(evol_rels)):
        base_row = base_rels.get(target) or {}
        evol_row = evol_rels.get(target) or {}
        row: dict[str, float] = {}
        for field in ("trust", "affection", "obligation"):
            base_val = float(base_row.get(field, 0.0))
            evol_val = float(evol_row.get(field, base_val))
            row[field] = (1.0 - a) * base_val + a * evol_val
        relationships[target] = row

    return (goals, closeness, relationships)


def _commitments_for_sweep(
    *,
    sweep: SweepName,
    alpha: float,
    baseline_commitments: list[str] | None,
    evolved_commitments: list[str] | None,
) -> tuple[str, ...] | None:
    if sweep == "continuous_only":
        choice = baseline_commitments
    elif sweep == "categorical_only":
        choice = evolved_commitments
    else:
        choice = baseline_commitments if float(alpha) < 0.5 else evolved_commitments

    if choice is None:
        return None
    return tuple(str(item) for item in choice)


def _build_evolutions_for_sweep(
    *,
    sweep: SweepName,
    alpha: float,
    baseline_profiles: dict[str, dict[str, Any]],
    evolved_profiles: dict[str, AgentEvolution],
) -> dict[str, AgentEvolution]:
    out: dict[str, AgentEvolution] = {}

    # Non-Thorne agents are fixed at alpha=1.0 (exactly Paper 2 Section 5.1 setup).
    for agent in DINNER_PARTY_AGENTS:
        if agent != "thorne":
            out[agent] = _copy_evolution(evolved_profiles[agent])
            continue

        base_profile = baseline_profiles[agent]
        evolved_profile = _evolution_to_profile_dict(evolved_profiles[agent])

        goals, closeness, relationships = _interpolate_continuous(
            baseline_profile=base_profile,
            evolved_profile=evolved_profile,
            alpha=float(alpha),
        )

        commitments = _commitments_for_sweep(
            sweep=sweep,
            alpha=float(alpha),
            baseline_commitments=(base_profile.get("commitments") or None),
            evolved_commitments=(evolved_profile.get("commitments") if "commitments" in evolved_profile else None),
        )

        out[agent] = AgentEvolution(
            goals_scalar=goals,
            closeness=closeness,
            relationships=relationships,
            commitments=commitments,
        )

    return out


def _canon_for_seed(
    *,
    seed: int,
    canon_cache: dict[int, Any],
    event_limit: int,
    tick_limit: int,
):
    if seed not in canon_cache:
        canon_cache[seed] = _generate_canon_after_b_for_seed(
            seed=seed,
            event_limit=event_limit,
            tick_limit=tick_limit,
        )
    return canon_cache[seed]


def _run_single_seed(
    *,
    seed: int,
    canon_cache: dict[int, Any],
    event_limit: int,
    tick_limit: int,
    evolutions: dict[str, AgentEvolution],
    sweep: str,
    alpha: float,
) -> dict[str, Any]:
    try:
        canon_after_b = _canon_for_seed(
            seed=seed,
            canon_cache=canon_cache,
            event_limit=event_limit,
            tick_limit=tick_limit,
        )
        story = _simulate_story(
            label=f"{sweep}_alpha_{alpha:.1f}",
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

        per_agent: dict[str, Any] = {}
        valid_scores: list[float] = []
        valid_count = 0

        for agent in DINNER_PARTY_AGENTS:
            search_result = search_arc(
                all_events=events,
                protagonist=agent,
                max_events=20,
                total_sim_time=total_sim_time,
                grammar_config=None,
            )
            q_score = float(score_arc(search_result.events, search_result.beats).composite) if search_result.events else 0.0
            tp_position, tp_event_type = _tp_info(
                events=search_result.events,
                beats=search_result.beats,
                simulation_end_tick=simulation_end_tick,
            )
            dev_count = _dev_beat_count(search_result.beats)

            per_agent[agent] = {
                "valid": bool(search_result.validation.valid),
                "q_score": float(q_score),
                "tp_position": tp_position,
                "tp_event_type": tp_event_type,
                "dev_beat_count": int(dev_count),
                "total_beats": int(len(search_result.beats)),
                "violations": [str(v) for v in search_result.validation.violations],
            }

            if bool(search_result.validation.valid):
                valid_count += 1
                valid_scores.append(float(q_score))

        mean_q = float(_mean(valid_scores))
        va_score = float(mean_q * (valid_count / float(len(DINNER_PARTY_AGENTS)))) if valid_count > 0 else 0.0

        diana_row = per_agent["diana"]

        return {
            "seed": int(seed),
            "sweep": sweep,
            "alpha": float(alpha),
            "valid_count": int(valid_count),
            "seed_validity_score": float(valid_count / float(len(DINNER_PARTY_AGENTS))),
            "mean_q": mean_q,
            "va_score": va_score,
            "all_valid": bool(valid_count == len(DINNER_PARTY_AGENTS)),
            "diana_valid": bool(diana_row["valid"]),
            "diana_tp_position": diana_row["tp_position"],
            "per_agent": per_agent,
            "error": None,
        }

    except Exception as exc:  # defensive continuation for long sweeps
        return {
            "seed": int(seed),
            "sweep": sweep,
            "alpha": float(alpha),
            "valid_count": 0,
            "seed_validity_score": 0.0,
            "mean_q": 0.0,
            "va_score": 0.0,
            "all_valid": False,
            "diana_valid": False,
            "diana_tp_position": None,
            "per_agent": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _summarize_bucket(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows[seed] for seed in sorted(rows)]

    mean_q_values = [float(row["mean_q"]) for row in ordered]
    va_values = [float(row["va_score"]) for row in ordered]
    validity_scores = [float(row["seed_validity_score"]) for row in ordered]
    all_valid_rate = _mean([1.0 if bool(row["all_valid"]) else 0.0 for row in ordered])

    per_agent_valid_count: dict[str, int] = {agent: 0 for agent in DINNER_PARTY_AGENTS}
    diana_invalid = 0
    diana_tp_positions: list[float] = []
    crashed_seeds: list[int] = []

    for row in ordered:
        if row.get("error") is not None:
            crashed_seeds.append(int(row["seed"]))

        per_agent = row.get("per_agent") or {}
        for agent in DINNER_PARTY_AGENTS:
            agent_row = per_agent.get(agent) or {}
            if bool(agent_row.get("valid")):
                per_agent_valid_count[agent] += 1

        if not bool(row.get("diana_valid")):
            diana_invalid += 1
        tp = row.get("diana_tp_position")
        if tp is not None:
            diana_tp_positions.append(float(tp))

    total = len(ordered)
    return {
        "mean_q": float(_mean(mean_q_values)),
        "va_score": float(_mean(va_values)),
        "all_valid_rate": float(all_valid_rate),
        "per_agent_validity": {
            agent: (float(per_agent_valid_count[agent] / total) if total > 0 else 0.0)
            for agent in DINNER_PARTY_AGENTS
        },
        "validity_std": float(_std(validity_scores)),
        "validity_variance": float(_variance(validity_scores)),
        "diana_invalid_count": int(diana_invalid),
        "mean_diana_tp_position": float(_mean(diana_tp_positions)),
        "crashed_seeds": sorted(crashed_seeds),
        "seed_validity_scores": validity_scores,
        "seed_va_scores": va_values,
    }


def _baseline_anchor_verification(rows_50: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows_50[seed] for seed in sorted(rows_50)]

    diana_invalid_seeds = sorted(
        int(row["seed"])
        for row in ordered
        if not bool((row.get("per_agent") or {}).get("diana", {}).get("valid"))
    )

    summaries = _summarize_bucket(rows_50)

    checks = {
        "diana_invalid_count": len(diana_invalid_seeds) == 9,
        "diana_invalid_seeds": diana_invalid_seeds == BASELINE_INVALID_SEEDS_EXPECTED,
        "mean_q_all_agents": round(float(summaries["mean_q"]), 3) == BASELINE_MEAN_Q_ALL_EXPECTED,
        "all_valid_rate": abs(float(summaries["all_valid_rate"]) - BASELINE_ALL_VALID_EXPECTED) <= 1e-12,
    }

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "expected": {
            "diana_invalid_count": 9,
            "diana_invalid_seeds": list(BASELINE_INVALID_SEEDS_EXPECTED),
            "mean_q_all_agents": BASELINE_MEAN_Q_ALL_EXPECTED,
            "all_valid_rate": BASELINE_ALL_VALID_EXPECTED,
        },
        "observed": {
            "diana_invalid_count": int(len(diana_invalid_seeds)),
            "diana_invalid_seeds": diana_invalid_seeds,
            "mean_q_all_agents": float(summaries["mean_q"]),
            "all_valid_rate": float(summaries["all_valid_rate"]),
        },
    }


def _coupled_reproduction_verification(coupled_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    alphas_sorted = sorted(float(alpha) for alpha in coupled_summary)

    def _key(alpha_val: float) -> str:
        return f"{alpha_val:.1f}"

    best_alpha = max(
        alphas_sorted,
        key=lambda alpha: (
            float(coupled_summary[_key(alpha)]["va_score"]),
            float(coupled_summary[_key(alpha)]["all_valid_rate"]),
            float(coupled_summary[_key(alpha)]["mean_q"]),
            -float(alpha),
        ),
    )

    va_05 = float(coupled_summary["0.5"]["va_score"])
    all_valid_05 = float(coupled_summary["0.5"]["all_valid_rate"])
    va_00 = float(coupled_summary["0.0"]["va_score"])
    va_10 = float(coupled_summary["1.0"]["va_score"])

    mae_against_paper = _mean(
        [
            abs(float(coupled_summary[f"{alpha:.1f}"]["va_score"]) - float(expected))
            for alpha, expected in sorted(PAPER2_VA_EXPECTED.items())
        ]
    )

    checks = {
        "peak_alpha_is_0_5": abs(best_alpha - 0.5) <= EPSILON,
        "alpha_0_5_va_near_0_684": abs(va_05 - 0.684) <= 0.03,
        "alpha_0_5_all_valid_near_95pct": abs(all_valid_05 - 0.95) <= 0.10,
        "alpha_0_endpoint_near_paper": abs(va_00 - PAPER2_VA_EXPECTED[0.0]) <= 0.05,
        "alpha_1_endpoint_near_paper": abs(va_10 - PAPER2_VA_EXPECTED[1.0]) <= 0.05,
        "curve_mae_vs_paper_reasonable": mae_against_paper <= 0.05,
    }

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "best_alpha_by_va": float(best_alpha),
            "alpha_0_5": {
                "va_score": va_05,
                "all_valid_rate": all_valid_05,
            },
            "alpha_0_0_va": va_00,
            "alpha_1_0_va": va_10,
            "curve_mae_vs_paper": float(mae_against_paper),
        },
    }


def _bootstrap_mean_ci(values: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(float(np.mean(arr[idx])))
    low = float(np.percentile(np.asarray(means, dtype=float), 2.5))
    high = float(np.percentile(np.asarray(means, dtype=float), 97.5))
    return (low, high)


def _best_alpha(summary_by_alpha: dict[str, dict[str, Any]]) -> float:
    return max(
        [float(alpha) for alpha in summary_by_alpha],
        key=lambda alpha: (
            float(summary_by_alpha[f"{alpha:.1f}"]["va_score"]),
            float(summary_by_alpha[f"{alpha:.1f}"]["all_valid_rate"]),
            float(summary_by_alpha[f"{alpha:.1f}"]["mean_q"]),
            -float(alpha),
        ),
    )


def _interaction_analysis(
    *,
    run_rows: dict[str, dict[str, dict[int, dict[str, Any]]]],
    summary: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    interaction_by_alpha: dict[str, float] = {}

    baseline_va = float(summary["coupled"]["0.0"]["va_score"])

    for alpha in ALPHAS:
        key = f"{alpha:.1f}"
        coupled_va = float(summary["coupled"][key]["va_score"])
        cont_va = float(summary["continuous_only"][key]["va_score"])
        cat_va = float(summary["categorical_only"][key]["va_score"])
        interaction = coupled_va - (cont_va + cat_va - baseline_va)
        interaction_by_alpha[key] = float(interaction)

    # Seed-level interaction at alpha=0.5 for a significance proxy.
    alpha_key = "0.5"
    seed_values: list[float] = []
    for seed in SWEEP_SEEDS:
        coupled_seed = float(run_rows["coupled"][alpha_key][seed]["va_score"])
        cont_seed = float(run_rows["continuous_only"][alpha_key][seed]["va_score"])
        cat_seed = float(run_rows["categorical_only"][alpha_key][seed]["va_score"])
        baseline_seed = float(run_rows["coupled"]["0.0"][seed]["va_score"])
        seed_values.append(coupled_seed - (cont_seed + cat_seed - baseline_seed))

    ci_low, ci_high = _bootstrap_mean_ci(seed_values, n_boot=10000, seed=0)
    mean_val = float(_mean(seed_values))
    std_val = float(_std(seed_values))
    sem = float(std_val / math.sqrt(len(seed_values))) if seed_values else 0.0
    z_like = float(mean_val / sem) if sem > 0 else 0.0

    return {
        "interaction_va_by_alpha": interaction_by_alpha,
        "alpha_0_5_seed_level": {
            "mean": mean_val,
            "std": std_val,
            "sem": sem,
            "z_like": z_like,
            "bootstrap_95ci": [ci_low, ci_high],
            "significant_nonzero": bool(ci_low > 0.0 or ci_high < 0.0),
        },
    }


def _line_chart_svg(
    *,
    title: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> str:
    width = 900
    height = 380
    margin_left = 70
    margin_right = 20
    margin_top = 40
    margin_bottom = 55
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_x = [x for rows in series.values() for x, _ in rows]
    all_y = [y for rows in series.values() for _, y in rows]

    x_min = min(all_x) if all_x else 0.0
    x_max = max(all_x) if all_x else 1.0
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    y_pad = 0.08 * (y_max - y_min)
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    def sx(x: float) -> float:
        if abs(x_max - x_min) <= 1e-12:
            return float(margin_left)
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return margin_top + ((y_hi - y) / (y_hi - y_lo)) * plot_h

    colors = {
        "coupled": "#005F73",
        "continuous_only": "#BB3E03",
        "categorical_only": "#0A9396",
    }

    parts: list[str] = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{title}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#F8F5F0"/>')
    parts.append(f'<text x="{margin_left}" y="24" font-size="18" font-family="Georgia, serif" fill="#1F2933">{title}</text>')

    # Axes
    x0 = margin_left
    y0 = margin_top + plot_h
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_w}" y2="{y0}" stroke="#334E68" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#334E68" stroke-width="1.5"/>')

    # Y ticks
    for i in range(5):
        frac = i / 4.0
        y_val = y_lo + (1.0 - frac) * (y_hi - y_lo)
        y_px = sy(y_val)
        parts.append(f'<line x1="{x0}" y1="{y_px:.2f}" x2="{margin_left + plot_w}" y2="{y_px:.2f}" stroke="#D9E2EC" stroke-width="1"/>')
        parts.append(
            f'<text x="{x0 - 8}" y="{y_px + 4:.2f}" text-anchor="end" font-size="11" '
            f'font-family="Menlo, monospace" fill="#334E68">{y_val:.3f}</text>'
        )

    # X ticks
    for alpha in ALPHAS:
        x_px = sx(float(alpha))
        parts.append(f'<line x1="{x_px:.2f}" y1="{y0}" x2="{x_px:.2f}" y2="{y0 + 5}" stroke="#334E68" stroke-width="1"/>')
        parts.append(
            f'<text x="{x_px:.2f}" y="{y0 + 18}" text-anchor="middle" font-size="11" '
            f'font-family="Menlo, monospace" fill="#334E68">{alpha:.1f}</text>'
        )

    # Lines
    legend_y = margin_top + 8
    legend_x = margin_left + 240
    for idx, (name, rows) in enumerate(series.items()):
        color = colors.get(name, "#486581")
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in rows)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6" points="{pts}"/>')
        for x, y in rows:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.7" fill="{color}"/>')

        lx = legend_x + idx * 200
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 24}" y2="{legend_y}" stroke="{color}" stroke-width="2.6"/>')
        parts.append(
            f'<text x="{lx + 30}" y="{legend_y + 4}" font-size="12" '
            f'font-family="Menlo, monospace" fill="#1F2933">{name}</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" '
        'font-size="12" font-family="Menlo, monospace" fill="#334E68">alpha</text>'
    )
    parts.append(
        f'<text x="18" y="{margin_top + plot_h / 2:.2f}" transform="rotate(-90 18 {margin_top + plot_h / 2:.2f})" '
        f'text-anchor="middle" font-size="12" font-family="Menlo, monospace" fill="#334E68">{y_label}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_plots_html(summary: dict[str, dict[str, dict[str, Any]]]) -> str:
    va_series: dict[str, list[tuple[float, float]]] = {}
    var_series: dict[str, list[tuple[float, float]]] = {}

    for sweep in SWEEPS:
        va_series[sweep] = [
            (float(alpha), float(summary[sweep][f"{alpha:.1f}"]["va_score"]))
            for alpha in ALPHAS
        ]
        var_series[sweep] = [
            (float(alpha), float(summary[sweep][f"{alpha:.1f}"]["validity_std"]))
            for alpha in ALPHAS
        ]

    va_svg = _line_chart_svg(
        title="VA vs alpha by sweep",
        y_label="validity-adjusted score",
        series=va_series,
    )
    var_svg = _line_chart_svg(
        title="Validity susceptibility (std of seed validity score)",
        y_label="std(valid_count / 6)",
        series=var_series,
    )

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "  <title>Decoupled Alpha Sweep Plots</title>",
            "  <style>",
            "    body { margin: 0; font-family: Georgia, serif; background: #F2EFEA; color: #1F2933; }",
            "    .wrap { max-width: 980px; margin: 0 auto; padding: 24px 18px 30px; }",
            "    h1 { font-size: 24px; margin: 0 0 14px; }",
            "    p { font-size: 14px; margin: 0 0 12px; line-height: 1.45; }",
            "    .card { background: #FFFFFF; border: 1px solid #D9E2EC; border-radius: 12px; padding: 12px; margin-bottom: 14px; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <div class=\"wrap\">",
            "    <h1>Decoupled alpha sweep</h1>",
            "    <p>Three sweeps over alpha (0.0..1.0): coupled, continuous_only, categorical_only.</p>",
            "    <div class=\"card\">",
            va_svg,
            "    </div>",
            "    <div class=\"card\">",
            var_svg,
            "    </div>",
            "  </div>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _build_summary_markdown(
    *,
    baseline_verification: dict[str, Any],
    coupled_verification: dict[str, Any],
    summary: dict[str, dict[str, dict[str, Any]]],
    interaction: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Decoupled Alpha Sweep")
    lines.append("")

    lines.append("## Baseline Verification")
    lines.append("")
    lines.append(f"- Baseline anchors passed: {'yes' if baseline_verification['passed'] else 'no'}")
    lines.append(
        f"- Diana invalid seeds: {baseline_verification['observed']['diana_invalid_seeds']} "
        f"(count={baseline_verification['observed']['diana_invalid_count']})"
    )
    lines.append(
        f"- Mean Q (all 6 agents, 50 seeds): {float(baseline_verification['observed']['mean_q_all_agents']):.3f}"
    )
    lines.append(
        f"- All-valid rate: {100.0 * float(baseline_verification['observed']['all_valid_rate']):.1f}%"
    )
    lines.append("")

    lines.append("## Coupled Sweep Verification")
    lines.append("")
    lines.append(f"- Coupled reproduction passed: {'yes' if coupled_verification['passed'] else 'no'}")
    lines.append(
        f"- Best alpha by VA: {float(coupled_verification['observed']['best_alpha_by_va']):.1f}"
    )
    lines.append(
        f"- alpha=0.5: VA={float(coupled_verification['observed']['alpha_0_5']['va_score']):.3f}, "
        f"all-valid={100.0 * float(coupled_verification['observed']['alpha_0_5']['all_valid_rate']):.1f}%"
    )
    lines.append("")

    lines.append("## Sweep Tables")
    lines.append("")
    lines.append(
        "| Sweep | alpha | Mean Q | VA | All-Valid | Validity Std | Diana Invalid | Diana Mean TP |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for sweep in SWEEPS:
        for alpha in ALPHAS:
            row = summary[sweep][f"{alpha:.1f}"]
            lines.append(
                "| "
                f"{sweep} | {alpha:.1f} | "
                f"{float(row['mean_q']):.3f} | "
                f"{float(row['va_score']):.3f} | "
                f"{100.0 * float(row['all_valid_rate']):.1f}% | "
                f"{float(row['validity_std']):.3f} | "
                f"{int(row['diana_invalid_count'])}/20 | "
                f"{float(row['mean_diana_tp_position']):.3f} |"
            )
    lines.append("")

    best_by_sweep = {
        sweep: _best_alpha(summary[sweep])
        for sweep in SWEEPS
    }

    lines.append("## Analysis 1: alpha=0.5 peak")
    lines.append("")
    lines.append(
        f"- Best alpha by VA: coupled={best_by_sweep['coupled']:.1f}, "
        f"continuous_only={best_by_sweep['continuous_only']:.1f}, "
        f"categorical_only={best_by_sweep['categorical_only']:.1f}"
    )

    if abs(best_by_sweep["coupled"] - 0.5) <= EPSILON and (
        abs(best_by_sweep["continuous_only"] - 0.5) > EPSILON
        or abs(best_by_sweep["categorical_only"] - 0.5) > EPSILON
    ):
        verdict = "alpha=0.5 peak is coupling-sensitive (not fully explained by either decoupled sweep alone)."
    elif abs(best_by_sweep["continuous_only"] - 0.5) <= EPSILON and abs(best_by_sweep["categorical_only"] - 0.5) > EPSILON:
        verdict = "continuous interpolation is the dominant driver of the alpha=0.5 peak."
    elif abs(best_by_sweep["continuous_only"] - 0.5) > EPSILON and abs(best_by_sweep["categorical_only"] - 0.5) <= EPSILON:
        verdict = "categorical commitment state is the dominant driver of the alpha=0.5 peak."
    else:
        verdict = "alpha=0.5 is not unique in decoupled sweeps; effect appears distributed or weakly identified."

    lines.append(f"- Verdict: {verdict}")
    lines.append("")

    lines.append("## Analysis 2: Variance/Susceptibility")
    lines.append("")
    for sweep in SWEEPS:
        best_var_alpha = max(
            ALPHAS,
            key=lambda alpha: float(summary[sweep][f"{alpha:.1f}"]["validity_std"]),
        )
        best_var_value = float(summary[sweep][f"{best_var_alpha:.1f}"]["validity_std"])
        lines.append(
            f"- {sweep}: max std(validity) at alpha={best_var_alpha:.1f} with std={best_var_value:.3f}"
        )
    lines.append("")

    lines.append("## Analysis 3: Interaction")
    lines.append("")
    lines.append("| alpha | Interaction (VA) |")
    lines.append("|---|---|")
    for alpha in ALPHAS:
        key = f"{alpha:.1f}"
        lines.append(f"| {alpha:.1f} | {float(interaction['interaction_va_by_alpha'][key]):.4f} |")

    alpha05 = interaction["alpha_0_5_seed_level"]
    lines.append("")
    lines.append(
        f"- alpha=0.5 interaction mean={float(alpha05['mean']):.4f}, "
        f"95% CI=[{float(alpha05['bootstrap_95ci'][0]):.4f}, {float(alpha05['bootstrap_95ci'][1]):.4f}], "
        f"significant_nonzero={'yes' if bool(alpha05['significant_nonzero']) else 'no'}"
    )
    lines.append("")

    crash_rows: list[str] = []
    for sweep in SWEEPS:
        for alpha in ALPHAS:
            row = summary[sweep][f"{alpha:.1f}"]
            crashed = row.get("crashed_seeds") or []
            if crashed:
                crash_rows.append(f"- {sweep} alpha={alpha:.1f}: {crashed}")

    lines.append("## Crashes")
    lines.append("")
    if crash_rows:
        lines.extend(crash_rows)
    else:
        lines.append("- No seed crashes.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoupled alpha sweep (continuous vs categorical commitments).")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--output-summary", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-plots", type=str, default=str(DEFAULT_PLOTS_PATH))
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_json_path = _resolve_path(args.output_json)
    output_summary_path = _resolve_path(args.output_summary)
    output_plots_path = _resolve_path(args.output_plots)

    print()
    print("=== DECOUPLED ALPHA SWEEP ===")
    print()
    print(f"Baseline verification seeds: {BASELINE_VERIFY_SEEDS[0]}..{BASELINE_VERIFY_SEEDS[-1]}")
    print(f"Sweep seeds: {SWEEP_SEEDS[0]}..{SWEEP_SEEDS[-1]}")
    print(f"Sweeps: {list(SWEEPS)}")
    print(f"Alphas: {[f'{alpha:.1f}' for alpha in ALPHAS]}")
    print()

    start_time = time.monotonic()

    evolved_profiles = _evolution_profiles()["full"]
    baseline_profiles = _extract_baseline_profiles()

    canon_cache: dict[int, Any] = {}

    # Critical rule: verify baseline anchors before any experimental condition.
    print("Running baseline verification condition (depth-2, full evolution, strict grammar)...")
    baseline_rows: dict[int, dict[str, Any]] = {}
    for idx, seed in enumerate(BASELINE_VERIFY_SEEDS, start=1):
        print(f"[baseline verify] seed {idx:03d}/{len(BASELINE_VERIFY_SEEDS):03d} ({seed})", flush=True)
        baseline_rows[seed] = _run_single_seed(
            seed=seed,
            canon_cache=canon_cache,
            event_limit=int(args.event_limit),
            tick_limit=int(args.tick_limit),
            evolutions={agent: _copy_evolution(evolved_profiles[agent]) for agent in DINNER_PARTY_AGENTS},
            sweep="baseline_verification",
            alpha=1.0,
        )

    baseline_verification = _baseline_anchor_verification(baseline_rows)
    if not baseline_verification["passed"]:
        payload = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment": "decoupled_alpha_sweep",
                "status": "baseline_failed",
            },
            "baseline_verification": baseline_verification,
        }
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        raise RuntimeError(
            "Baseline verification failed; stopping before alpha sweeps. "
            f"Observed={baseline_verification['observed']}"
        )

    print("Baseline verification passed.")
    print()

    run_rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        sweep: {f"{alpha:.1f}": {} for alpha in ALPHAS}
        for sweep in SWEEPS
    }

    # Step 2: coupled sweep first, verify against Paper 2 before decoupled sweeps.
    print("Running coupled sweep first (verification gate)...")
    for alpha in ALPHAS:
        evolutions = _build_evolutions_for_sweep(
            sweep="coupled",
            alpha=float(alpha),
            baseline_profiles=baseline_profiles,
            evolved_profiles=evolved_profiles,
        )
        for idx, seed in enumerate(SWEEP_SEEDS, start=1):
            print(
                f"[coupled alpha={alpha:.1f}] seed {idx:02d}/{len(SWEEP_SEEDS):02d} ({seed})",
                flush=True,
            )
            run_rows["coupled"][f"{alpha:.1f}"][seed] = _run_single_seed(
                seed=seed,
                canon_cache=canon_cache,
                event_limit=int(args.event_limit),
                tick_limit=int(args.tick_limit),
                evolutions=evolutions,
                sweep="coupled",
                alpha=float(alpha),
            )

    summary: dict[str, dict[str, dict[str, Any]]] = {
        "coupled": {
            f"{alpha:.1f}": _summarize_bucket(run_rows["coupled"][f"{alpha:.1f}"])
            for alpha in ALPHAS
        },
        "continuous_only": {},
        "categorical_only": {},
    }

    coupled_verification = _coupled_reproduction_verification(summary["coupled"])
    if not coupled_verification["passed"]:
        payload = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment": "decoupled_alpha_sweep",
                "status": "coupled_verification_failed",
            },
            "baseline_verification": baseline_verification,
            "coupled_verification": coupled_verification,
            "coupled_summary": summary["coupled"],
        }
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        raise RuntimeError(
            "Coupled alpha sweep failed reproduction gate; stopping before decoupled sweeps. "
            f"Observed={coupled_verification['observed']}"
        )

    print("Coupled sweep verification passed.")
    print()

    for sweep in ["continuous_only", "categorical_only"]:
        print(f"Running {sweep} sweep...")
        for alpha in ALPHAS:
            evolutions = _build_evolutions_for_sweep(
                sweep=sweep,
                alpha=float(alpha),
                baseline_profiles=baseline_profiles,
                evolved_profiles=evolved_profiles,
            )
            for idx, seed in enumerate(SWEEP_SEEDS, start=1):
                print(
                    f"[{sweep} alpha={alpha:.1f}] seed {idx:02d}/{len(SWEEP_SEEDS):02d} ({seed})",
                    flush=True,
                )
                run_rows[sweep][f"{alpha:.1f}"][seed] = _run_single_seed(
                    seed=seed,
                    canon_cache=canon_cache,
                    event_limit=int(args.event_limit),
                    tick_limit=int(args.tick_limit),
                    evolutions=evolutions,
                    sweep=sweep,
                    alpha=float(alpha),
                )

        summary[sweep] = {
            f"{alpha:.1f}": _summarize_bucket(run_rows[sweep][f"{alpha:.1f}"])
            for alpha in ALPHAS
        }

    interaction = _interaction_analysis(run_rows=run_rows, summary=summary)

    plots_html = _build_plots_html(summary)
    summary_md = _build_summary_markdown(
        baseline_verification=baseline_verification,
        coupled_verification=coupled_verification,
        summary=summary,
        interaction=interaction,
    )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": "decoupled_alpha_sweep",
            "runtime_seconds": float(time.monotonic() - start_time),
            "event_limit": int(args.event_limit),
            "tick_limit": int(args.tick_limit),
            "baseline_verify_seeds": [int(seed) for seed in BASELINE_VERIFY_SEEDS],
            "sweep_seeds": [int(seed) for seed in SWEEP_SEEDS],
            "alphas": [float(alpha) for alpha in ALPHAS],
            "sweeps": list(SWEEPS),
        },
        "baseline_verification": baseline_verification,
        "coupled_verification": coupled_verification,
        "sweeps": {
            sweep: {
                alpha: {
                    str(seed): run_rows[sweep][alpha][seed]
                    for seed in sorted(run_rows[sweep][alpha])
                }
                for alpha in sorted(run_rows[sweep])
            }
            for sweep in SWEEPS
        },
        "summary": summary,
        "analysis": {
            "best_alpha_by_sweep": {
                sweep: float(_best_alpha(summary[sweep]))
                for sweep in SWEEPS
            },
            "interaction": interaction,
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(summary_md, encoding="utf-8")

    output_plots_path.parent.mkdir(parents=True, exist_ok=True)
    output_plots_path.write_text(plots_html, encoding="utf-8")

    print()
    print(f"Wrote JSON: {output_json_path}")
    print(f"Wrote summary: {output_summary_path}")
    print(f"Wrote plots: {output_plots_path}")
    print(
        "Best alpha by VA: "
        f"coupled={_best_alpha(summary['coupled']):.1f}, "
        f"continuous_only={_best_alpha(summary['continuous_only']):.1f}, "
        f"categorical_only={_best_alpha(summary['categorical_only']):.1f}"
    )


if __name__ == "__main__":
    main()
