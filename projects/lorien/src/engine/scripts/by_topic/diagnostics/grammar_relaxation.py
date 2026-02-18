"""Grammar relaxation diagnostic with full re-extraction under two grammars.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.grammar_relaxation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narrativefield.extraction.rashomon import RashomonSet, extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.k_sweep_experiment import (
    DINNER_PARTY_AGENTS,
    _generate_canon_after_b_for_seed,
    _simulate_story,
)
from scripts.relaxed_arc_search import extract_rashomon_set_relaxed
from scripts.test_goal_evolution import _evolution_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "grammar_relaxation.json"
DEFAULT_SWEEP_PATH = OUTPUT_DIR / "k_sweep_experiment.json"
EPSILON = 1e-10


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    evolved_agents: tuple[str, ...]
    k: int
    condition_type: str


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


def _summarize_rashomon(rashomon: RashomonSet) -> dict[str, Any]:
    per_agent_scores: dict[str, float | None] = {agent: None for agent in DINNER_PARTY_AGENTS}
    per_agent_valid: dict[str, bool] = {agent: False for agent in DINNER_PARTY_AGENTS}
    valid_scores: list[float] = []
    invalid_agents: list[str] = []

    for arc in rashomon.arcs:
        agent = str(arc.protagonist)
        score = float(arc.arc_score.composite) if arc.arc_score is not None else None
        per_agent_scores[agent] = score
        per_agent_valid[agent] = bool(arc.valid)
        if arc.valid and score is not None:
            valid_scores.append(score)
        else:
            invalid_agents.append(agent)

    invalid_agents_sorted = sorted(invalid_agents)
    valid_agents_sorted = sorted(agent for agent, valid in per_agent_valid.items() if valid)
    mean_score = _mean(valid_scores)
    valid_arc_count = int(rashomon.valid_count)
    va_score = float(mean_score * (valid_arc_count / float(len(DINNER_PARTY_AGENTS))))

    return {
        "mean_score": float(mean_score),
        "va_score": float(va_score),
        "valid_arc_count": int(valid_arc_count),
        "invalid_agents": invalid_agents_sorted,
        "valid_agents": valid_agents_sorted,
        "per_agent_scores": per_agent_scores,
        "per_agent_valid": per_agent_valid,
    }


def _select_reference_condition(run: dict[str, Any]) -> str | None:
    k = int(run.get("k", -1))
    condition_type = str(run.get("condition_type") or "")
    evolved = {str(agent) for agent in (run.get("evolved_agents") or [])}

    if k == 0 and condition_type == "baseline":
        return "baseline"
    if k == 6 and condition_type == "full_evolution":
        return "full_evolution"
    if k == 5 and condition_type == "subset" and "thorne" not in evolved:
        return "excl_thorne"
    return None


def _load_reference(sweep_path: Path, seeds: list[int]) -> dict[tuple[int, str], dict[str, Any]]:
    raw = json.loads(sweep_path.read_text(encoding="utf-8"))
    runs = list(raw.get("runs") or [])
    seed_set = set(seeds)
    reference: dict[tuple[int, str], dict[str, Any]] = {}

    for run in runs:
        seed = int(run.get("seed", 0))
        if seed not in seed_set:
            continue
        cond = _select_reference_condition(run)
        if cond is None:
            continue
        reference[(seed, cond)] = {
            "mean_score": float(run.get("mean_score", 0.0)),
            "invalid_agents": sorted(str(agent) for agent in (run.get("invalid_agents") or [])),
        }

    required = {(seed, cond) for seed in seeds for cond in ("baseline", "full_evolution", "excl_thorne")}
    missing = sorted(required - set(reference.keys()))
    if missing:
        preview = ", ".join(f"{seed}:{cond}" for seed, cond in missing[:10])
        raise ValueError(f"Missing strict reference rows in k_sweep_experiment.json: {preview}")
    return reference


def _condition_specs() -> list[ConditionSpec]:
    all_agents = tuple(DINNER_PARTY_AGENTS)
    excl_thorne_agents = tuple(agent for agent in DINNER_PARTY_AGENTS if agent != "thorne")
    return [
        ConditionSpec(
            key="baseline",
            label="Baseline (k=0)",
            evolved_agents=(),
            k=0,
            condition_type="baseline",
        ),
        ConditionSpec(
            key="full_evolution",
            label="Full Evolution (k=6)",
            evolved_agents=all_agents,
            k=6,
            condition_type="full_evolution",
        ),
        ConditionSpec(
            key="excl_thorne",
            label="Excl-Thorne (k=5)",
            evolved_agents=excl_thorne_agents,
            k=5,
            condition_type="subset",
        ),
    ]


def _run_single(
    *,
    seed: int,
    condition: ConditionSpec,
    canon_after_b,
    full_profiles: dict[str, Any],
    tick_limit: int,
    event_limit: int,
) -> dict[str, dict[str, Any]]:
    evolutions = {agent: full_profiles[agent] for agent in condition.evolved_agents}
    story = _simulate_story(
        label=condition.key,
        seed=seed,
        loaded_canon=canon_after_b,
        tick_limit=tick_limit,
        event_limit=event_limit,
        evolutions=evolutions,
    )

    parsed = parse_simulation_output(story.payload)
    metrics_output = run_metrics_pipeline(parsed)
    total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None

    strict_rashomon = extract_rashomon_set(
        events=metrics_output.events,
        seed=seed,
        agents=list(DINNER_PARTY_AGENTS),
        total_sim_time=total_sim_time,
    )
    relaxed_rashomon = extract_rashomon_set_relaxed(
        events=metrics_output.events,
        seed=seed,
        agents=list(DINNER_PARTY_AGENTS),
        total_sim_time=total_sim_time,
    )
    return {
        "strict": _summarize_rashomon(strict_rashomon),
        "relaxed": _summarize_rashomon(relaxed_rashomon),
    }


def _aggregate_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    mean_scores = [float(row["mean_score"]) for row in rows]
    va_scores = [float(row["va_score"]) for row in rows]
    valid_counts = [int(row["valid_arc_count"]) for row in rows]
    per_agent_invalid_rate = {
        agent: float(sum(1 for row in rows if not bool(row["per_agent_valid"][agent])) / n) if n else 0.0
        for agent in DINNER_PARTY_AGENTS
    }
    return {
        "mean_score": float(_mean(mean_scores)),
        "va_score": float(_mean(va_scores)),
        "all_valid_rate": float(sum(1 for count in valid_counts if count == len(DINNER_PARTY_AGENTS)) / n)
        if n
        else 0.0,
        "mean_valid_count": float(_mean([float(count) for count in valid_counts])),
        "per_agent_invalid_rate": per_agent_invalid_rate,
    }


def _per_seed_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_score": float(summary["mean_score"]),
        "valid_arc_count": int(summary["valid_arc_count"]),
        "valid_agents": list(summary["valid_agents"]),
        "invalid_agents": list(summary["invalid_agents"]),
        "per_agent_scores": dict(summary["per_agent_scores"]),
    }


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _print_summary(
    *,
    strict: dict[str, dict[str, Any]],
    relaxed: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    print()
    print("=== GRAMMAR RELAXATION DIAGNOSTIC ===")
    print()
    print("Condition                        Strict Mean   Relax Mean    Strict VA    Relax VA   Strict All6   Relax All6")
    order = ["baseline", "full_evolution", "excl_thorne"]
    labels = {
        "baseline": "Baseline (k=0)",
        "full_evolution": "Full Evolution (k=6)",
        "excl_thorne": "Excl-Thorne (k=5)",
    }
    for key in order:
        s = strict[key]
        r = relaxed[key]
        print(
            f"{labels[key]:<30} "
            f"{float(s['mean_score']):>11.4f} "
            f"{float(r['mean_score']):>11.4f} "
            f"{float(s['va_score']):>11.4f} "
            f"{float(r['va_score']):>11.4f} "
            f"{_format_pct(float(s['all_valid_rate'])):>12} "
            f"{_format_pct(float(r['all_valid_rate'])):>12}"
        )
    print()

    strict_adv = float(comparison["excl_thorne_advantage_strict_va"])
    relaxed_adv = float(comparison["excl_thorne_advantage_relaxed_va"])
    reduction = float(comparison["advantage_reduction_pct"])
    print(f"Excl-Thorne VA advantage (strict):  {strict_adv:+.6f}")
    print(f"Excl-Thorne VA advantage (relaxed): {relaxed_adv:+.6f}")
    print(f"Advantage reduction: {reduction:+.2f}%")
    print()

    print("Per-agent recovery in full-evolution runs (strict invalid -> relaxed valid):")
    print("agent      strict_invalid  relaxed_invalid  recovered")
    for agent in DINNER_PARTY_AGENTS:
        row = comparison["per_agent_recovery"][agent]
        print(
            f"{agent:<10} {int(row['strict_invalid']):>14} "
            f"{int(row['relaxed_invalid']):>15} {int(row['recovered']):>10}"
        )
    print()

    rec_score = comparison["recovered_arc_mean_score"]
    both_score = comparison["both_valid_arc_mean_score"]
    rec_text = "null" if rec_score is None else f"{float(rec_score):.4f}"
    both_text = "null" if both_score is None else f"{float(both_score):.4f}"
    print(f"Recovered-arc mean score: {rec_text}")
    print(f"Both-valid mean score:    {both_text}")
    print()

    stability = comparison["score_changed_for_both_valid"]
    print(
        "Score stability (both-valid arcs): "
        f"mean_abs_delta={float(stability['mean_abs_delta']):.6f}, "
        f"max_delta={float(stability['max_delta']):.6f}, "
        f"n_changed={int(stability['n_changed'])}"
    )
    print()

    verdict = comparison["verdict"]
    print(f"Verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict-vs-relaxed grammar extraction diagnostic.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--k-sweep", type=str, default=str(DEFAULT_SWEEP_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    sweep_path = _resolve_path(args.k_sweep)
    if not sweep_path.exists():
        raise FileNotFoundError(f"k-sweep reference not found: {sweep_path}")

    seeds = _parse_seeds(args.seeds)
    conditions = _condition_specs()
    reference = _load_reference(sweep_path, seeds)
    full_profiles = _evolution_profiles()["full"]

    run_matrix: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in seeds}
    strict_mismatches: list[dict[str, Any]] = []

    total_runs = len(seeds) * len(conditions)
    completed = 0
    for seed in seeds:
        canon_after_b = _generate_canon_after_b_for_seed(
            seed=seed,
            event_limit=int(args.event_limit),
            tick_limit=int(args.tick_limit),
        )
        for condition in conditions:
            completed += 1
            print(
                f"[{completed:03d}/{total_runs:03d}] "
                f"seed={seed} condition={condition.key}",
                flush=True,
            )
            result = _run_single(
                seed=seed,
                condition=condition,
                canon_after_b=canon_after_b,
                full_profiles=full_profiles,
                tick_limit=int(args.tick_limit),
                event_limit=int(args.event_limit),
            )
            run_matrix[seed][condition.key] = result

            strict_summary = result["strict"]
            expected = reference[(seed, condition.key)]
            mean_delta = abs(float(strict_summary["mean_score"]) - float(expected["mean_score"]))
            invalid_match = list(strict_summary["invalid_agents"]) == list(expected["invalid_agents"])
            if mean_delta > EPSILON or not invalid_match:
                strict_mismatches.append(
                    {
                        "seed": int(seed),
                        "condition": condition.key,
                        "mean_score_expected": float(expected["mean_score"]),
                        "mean_score_actual": float(strict_summary["mean_score"]),
                        "mean_score_abs_delta": float(mean_delta),
                        "invalid_expected": list(expected["invalid_agents"]),
                        "invalid_actual": list(strict_summary["invalid_agents"]),
                    }
                )

    strict_matches = len(strict_mismatches) == 0
    if not strict_matches:
        preview = strict_mismatches[:5]
        raise RuntimeError(
            "Strict extraction did not match k-sweep reference; aborting analysis. "
            f"First mismatches: {json.dumps(preview, ensure_ascii=True)}"
        )

    strict_by_condition: dict[str, list[dict[str, Any]]] = {condition.key: [] for condition in conditions}
    relaxed_by_condition: dict[str, list[dict[str, Any]]] = {condition.key: [] for condition in conditions}
    for seed in seeds:
        for condition in conditions:
            strict_by_condition[condition.key].append(run_matrix[seed][condition.key]["strict"])
            relaxed_by_condition[condition.key].append(run_matrix[seed][condition.key]["relaxed"])

    strict_agg = {key: _aggregate_condition(rows) for key, rows in strict_by_condition.items()}
    relaxed_agg = {key: _aggregate_condition(rows) for key, rows in relaxed_by_condition.items()}

    strict_adv = float(strict_agg["excl_thorne"]["va_score"]) - float(strict_agg["full_evolution"]["va_score"])
    relaxed_adv = float(relaxed_agg["excl_thorne"]["va_score"]) - float(relaxed_agg["full_evolution"]["va_score"])
    if abs(strict_adv) <= EPSILON:
        reduction_pct = 0.0
    else:
        reduction_pct = float(((strict_adv - relaxed_adv) / strict_adv) * 100.0)

    per_agent_recovery: dict[str, dict[str, int]] = {
        agent: {"strict_invalid": 0, "relaxed_invalid": 0, "recovered": 0}
        for agent in DINNER_PARTY_AGENTS
    }
    recovered_scores: list[float] = []
    both_valid_scores: list[float] = []
    both_valid_abs_deltas: list[float] = []

    for seed in seeds:
        strict_full = run_matrix[seed]["full_evolution"]["strict"]
        relaxed_full = run_matrix[seed]["full_evolution"]["relaxed"]
        for agent in DINNER_PARTY_AGENTS:
            strict_valid = bool(strict_full["per_agent_valid"][agent])
            relaxed_valid = bool(relaxed_full["per_agent_valid"][agent])

            if not strict_valid:
                per_agent_recovery[agent]["strict_invalid"] += 1
            if not relaxed_valid:
                per_agent_recovery[agent]["relaxed_invalid"] += 1
            if (not strict_valid) and relaxed_valid:
                per_agent_recovery[agent]["recovered"] += 1
                relaxed_score = relaxed_full["per_agent_scores"][agent]
                if relaxed_score is not None:
                    recovered_scores.append(float(relaxed_score))

            if strict_valid and relaxed_valid:
                strict_score = strict_full["per_agent_scores"][agent]
                relaxed_score = relaxed_full["per_agent_scores"][agent]
                if strict_score is not None and relaxed_score is not None:
                    both_valid_scores.append(float(relaxed_score))
                    both_valid_abs_deltas.append(abs(float(relaxed_score) - float(strict_score)))

    recovered_arc_mean_score = _mean(recovered_scores) if recovered_scores else None
    both_valid_arc_mean_score = _mean(both_valid_scores) if both_valid_scores else None
    score_changed_for_both_valid = {
        "mean_abs_delta": float(_mean(both_valid_abs_deltas)) if both_valid_abs_deltas else 0.0,
        "max_delta": float(max(both_valid_abs_deltas)) if both_valid_abs_deltas else 0.0,
        "n_changed": int(sum(1 for delta in both_valid_abs_deltas if delta > EPSILON)),
    }

    verdict = (
        "Grammar is the bottleneck"
        if reduction_pct > 50.0
        else "Evolution profile is the bottleneck"
    )

    comparison = {
        "excl_thorne_advantage_strict_va": float(strict_adv),
        "excl_thorne_advantage_relaxed_va": float(relaxed_adv),
        "advantage_reduction_pct": float(reduction_pct),
        "per_agent_recovery": per_agent_recovery,
        "recovered_arc_mean_score": recovered_arc_mean_score,
        "both_valid_arc_mean_score": both_valid_arc_mean_score,
        "score_changed_for_both_valid": score_changed_for_both_valid,
        "verdict": verdict,
    }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        row = {
            "seed": int(seed),
            "baseline_strict": _per_seed_payload(run_matrix[seed]["baseline"]["strict"]),
            "baseline_relaxed": _per_seed_payload(run_matrix[seed]["baseline"]["relaxed"]),
            "full_strict": _per_seed_payload(run_matrix[seed]["full_evolution"]["strict"]),
            "full_relaxed": _per_seed_payload(run_matrix[seed]["full_evolution"]["relaxed"]),
            "excl_thorne_strict": _per_seed_payload(run_matrix[seed]["excl_thorne"]["strict"]),
            "excl_thorne_relaxed": _per_seed_payload(run_matrix[seed]["excl_thorne"]["relaxed"]),
        }
        per_seed.append(row)

    payload = {
        "config": {
            "seeds": [int(seed) for seed in seeds],
            "conditions": ["baseline", "full_evolution", "excl_thorne"],
            "strict_rules": {"max_turning_points": 1, "max_phase_regressions": 0},
            "relaxed_rules": {"max_turning_points": 2, "max_phase_regressions": 1},
            "total_runs": int(total_runs),
            "strict_matches_k_sweep": strict_matches,
        },
        "strict": strict_agg,
        "relaxed": relaxed_agg,
        "comparison": comparison,
        "per_seed": per_seed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_summary(strict=strict_agg, relaxed=relaxed_agg, comparison=comparison)


if __name__ == "__main__":
    main()
