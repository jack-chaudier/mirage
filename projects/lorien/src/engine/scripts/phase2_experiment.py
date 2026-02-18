"""Phase 2 experiment runner: alpha sweep + amplifier/repair factorial.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.phase2_experiment

Notes:
- Reuses simulation/analysis infrastructure from k_sweep_experiment.
- Supports checkpoint+resume across all phases.
- Preserves A1 Thorne alpha resolution by default (no coarse A2 overwrite).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from narrativefield.schema.canon import WorldCanon
from narrativefield.simulation.scenarios import create_dinner_party_world
from scripts.k_sweep_experiment import (
    DEFAULT_WOUND_PATH,
    DINNER_PARTY_AGENTS,
    ConditionSpec as KSConditionSpec,
    _generate_canon_after_b_for_seed,
    _load_wound_patterns_or_fail,
    _parse_seeds,
    _run_determinism_gate,
    _run_to_record,
    _save_json,
)
from scripts.test_goal_evolution import AgentEvolution

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "phase2_experiment.json"
DEFAULT_K_SWEEP_PATH = OUTPUT_DIR / "k_sweep_experiment.json"

AGENT_ORDER = tuple(DINNER_PARTY_AGENTS)
A1_ALPHAS = tuple(round(x * 0.1, 1) for x in range(11))
A2_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
A2_AGENT_ORDER = ("diana", "elena", "lydia", "marcus", "victor")


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=0)) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if float(np.std(x_arr, ddof=0)) == 0.0 or float(np.std(y_arr, ddof=0)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if math.isnan(corr):
        return 0.0
    return corr


def _validity_adjusted_score(mean_score: float, valid_arc_count: int, total_agents: int = 6) -> float:
    if int(valid_arc_count) <= 0:
        return 0.0
    return float(mean_score) * (float(valid_arc_count) / float(total_agents))


def _alpha_vector_default(value: float) -> dict[str, float]:
    return {agent: float(value) for agent in AGENT_ORDER}


def _canonical_alpha_vector(raw: dict[str, float] | None) -> dict[str, float]:
    out = _alpha_vector_default(0.0)
    if raw is None:
        return out
    for agent in AGENT_ORDER:
        out[agent] = float(raw.get(agent, 0.0))
    return out


def _alpha_vector_token(alpha_vector: dict[str, float]) -> str:
    canon = _canonical_alpha_vector(alpha_vector)
    return ",".join(f"{agent}:{canon[agent]:.3f}" for agent in AGENT_ORDER)


def _active_agents(alpha_vector: dict[str, float], eps: float = 1e-12) -> tuple[str, ...]:
    canon = _canonical_alpha_vector(alpha_vector)
    return tuple(agent for agent in AGENT_ORDER if canon[agent] > eps)


def _load_k_sweep_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"k-sweep artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "runs" not in payload or "config" not in payload:
        raise ValueError(f"Invalid k-sweep artifact format: {path}")
    return payload


def _deserialize_evolution_profiles(raw: dict[str, Any]) -> dict[str, AgentEvolution]:
    out: dict[str, AgentEvolution] = {}
    for agent in AGENT_ORDER:
        row = raw.get(agent)
        if not isinstance(row, dict):
            raise ValueError(f"Missing/invalid evolved profile for '{agent}' in k-sweep artifact.")

        goals_scalar = {str(k): float(v) for k, v in (row.get("goals_scalar") or {}).items()}
        closeness = {str(k): float(v) for k, v in (row.get("closeness") or {}).items()}
        relationships: dict[str, dict[str, float]] = {}
        for target, attrs in (row.get("relationships") or {}).items():
            relationships[str(target)] = {
                "trust": float((attrs or {}).get("trust", 0.0)),
                "affection": float((attrs or {}).get("affection", 0.0)),
                "obligation": float((attrs or {}).get("obligation", 0.0)),
            }

        commitments_raw = row.get("commitments", None)
        commitments: tuple[str, ...] | None
        if commitments_raw is None:
            commitments = None
        else:
            commitments = tuple(str(item) for item in commitments_raw)

        out[agent] = AgentEvolution(
            goals_scalar=goals_scalar,
            closeness=closeness,
            relationships=relationships,
            commitments=commitments,
        )
    return out


def _serialize_profiles(profiles: dict[str, AgentEvolution]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for agent in AGENT_ORDER:
        row = profiles[agent]
        out[agent] = {
            "goals_scalar": {k: float(v) for k, v in sorted(row.goals_scalar.items())},
            "closeness": {k: float(v) for k, v in sorted(row.closeness.items())},
            "relationships": {
                target: {attr: float(value) for attr, value in sorted(values.items())}
                for target, values in sorted(row.relationships.items())
            },
            "commitments": None if row.commitments is None else [str(x) for x in row.commitments],
        }
    return out


def _extract_baseline_profiles() -> dict[str, dict[str, Any]]:
    world = create_dinner_party_world()
    profiles: dict[str, dict[str, Any]] = {}
    for agent in AGENT_ORDER:
        state = world.agents[agent]
        profiles[agent] = {
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


def _interpolate_profile(
    baseline_profile: dict[str, Any],
    evolved_profile: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    a = float(alpha)
    base_goals = baseline_profile.get("goals_scalar") or {}
    evol_goals = evolved_profile.get("goals_scalar") or {}

    result_goals: dict[str, float] = {}
    for dim in sorted(set(base_goals) | set(evol_goals)):
        base_val = float(base_goals.get(dim, 0.0))
        evol_val = float(evol_goals.get(dim, base_val))
        result_goals[dim] = (1.0 - a) * base_val + a * evol_val

    base_close = baseline_profile.get("closeness") or {}
    evol_close = evolved_profile.get("closeness") or {}
    result_close: dict[str, float] = {}
    for target in sorted(set(base_close) | set(evol_close)):
        base_val = float(base_close.get(target, 0.0))
        evol_val = float(evol_close.get(target, base_val))
        result_close[target] = (1.0 - a) * base_val + a * evol_val

    base_rels = baseline_profile.get("relationships") or {}
    evol_rels = evolved_profile.get("relationships") or {}
    rel_targets = sorted(set(base_rels) | set(evol_rels))
    result_rels: dict[str, dict[str, float]] = {}
    for target in rel_targets:
        base_row = base_rels.get(target) or {}
        evol_row = evol_rels.get(target) or {}
        result_rels[target] = {}
        for field in ("trust", "affection", "obligation"):
            base_val = float(base_row.get(field, 0.0))
            evol_val = float(evol_row.get(field, base_val))
            result_rels[target][field] = (1.0 - a) * base_val + a * evol_val

    baseline_commitments = baseline_profile.get("commitments")
    evolved_commitments = evolved_profile.get("commitments")
    if a < 0.5:
        result_commitments = baseline_commitments
    else:
        result_commitments = evolved_commitments

    if result_commitments is None:
        commitments: list[str] | None = None
    else:
        commitments = [str(item) for item in result_commitments]

    return {
        "goals_scalar": result_goals,
        "closeness": result_close,
        "relationships": result_rels,
        "commitments": commitments,
    }


def _build_interpolated_evolutions(
    alpha_vector: dict[str, float],
    baseline_profiles: dict[str, dict[str, Any]],
    evolved_profiles: dict[str, AgentEvolution],
) -> dict[str, AgentEvolution]:
    canon_alpha = _canonical_alpha_vector(alpha_vector)
    out: dict[str, AgentEvolution] = {}

    for agent in AGENT_ORDER:
        alpha = float(canon_alpha[agent])
        if alpha <= 1e-12:
            continue

        base = baseline_profiles[agent]
        evol_obj = evolved_profiles[agent]
        evol = {
            "goals_scalar": {k: float(v) for k, v in evol_obj.goals_scalar.items()},
            "closeness": {k: float(v) for k, v in evol_obj.closeness.items()},
            "relationships": {
                t: {
                    "trust": float(vals.get("trust", 0.0)),
                    "affection": float(vals.get("affection", 0.0)),
                    "obligation": float(vals.get("obligation", 0.0)),
                }
                for t, vals in evol_obj.relationships.items()
            },
            "commitments": None if evol_obj.commitments is None else list(evol_obj.commitments),
        }
        interpolated = _interpolate_profile(base, evol, alpha)

        commitments_raw = interpolated.get("commitments")
        commitments: tuple[str, ...] | None
        if commitments_raw is None:
            commitments = None
        else:
            commitments = tuple(str(item) for item in commitments_raw)

        out[agent] = AgentEvolution(
            goals_scalar={k: float(v) for k, v in (interpolated.get("goals_scalar") or {}).items()},
            closeness={k: float(v) for k, v in (interpolated.get("closeness") or {}).items()},
            relationships={
                target: {
                    "trust": float((attrs or {}).get("trust", 0.0)),
                    "affection": float((attrs or {}).get("affection", 0.0)),
                    "obligation": float((attrs or {}).get("obligation", 0.0)),
                }
                for target, attrs in (interpolated.get("relationships") or {}).items()
            },
            commitments=commitments,
        )
    return out


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    means = [float(row.get("mean_score", 0.0)) for row in rows]
    va_scores = [float(row.get("validity_adjusted_score", 0.0)) for row in rows]
    valid_all = [1.0 if int(row.get("valid_arc_count", 0)) == 6 else 0.0 for row in rows]
    return {
        "mean": float(_mean(means)),
        "std": float(_std(means)),
        "valid_rate": float(_mean(valid_all)),
        "va_score": float(_mean(va_scores)),
        "n": int(len(rows)),
    }


def _build_pareto_alpha(curve: dict[float, dict[str, float | int]]) -> float:
    if not curve:
        return 0.0

    alphas = sorted(curve)
    pareto: list[float] = []
    for alpha in alphas:
        point = curve[alpha]
        dominated = False
        for other in alphas:
            if other == alpha:
                continue
            ref = curve[other]
            better_or_equal = (
                float(ref["mean"]) >= float(point["mean"]) and float(ref["valid_rate"]) >= float(point["valid_rate"])
            )
            strictly_better = (
                float(ref["mean"]) > float(point["mean"]) or float(ref["valid_rate"]) > float(point["valid_rate"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(alpha)

    return max(
        pareto,
        key=lambda a: (
            float(curve[a]["va_score"]),
            float(curve[a]["valid_rate"]),
            float(curve[a]["mean"]),
            -float(a),
        ),
    )


def _best_alpha_by_metric(curve: dict[float, dict[str, float | int]], metric: str) -> float:
    return max(
        curve,
        key=lambda a: (
            float(curve[a][metric]),
            float(curve[a]["valid_rate"]),
            float(curve[a]["mean"]),
            -float(a),
        ),
    )


def _k_sweep_key(seed: int, condition_type: str, k: int, evolved_agents: tuple[str, ...]) -> tuple[int, str, int, tuple[str, ...]]:
    return (int(seed), str(condition_type), int(k), tuple(sorted(str(agent) for agent in evolved_agents)))


def _build_k_sweep_lookup(k_sweep_runs: list[dict[str, Any]]) -> dict[tuple[int, str, int, tuple[str, ...]], dict[str, Any]]:
    lookup: dict[tuple[int, str, int, tuple[str, ...]], dict[str, Any]] = {}
    for row in k_sweep_runs:
        key = _k_sweep_key(
            seed=int(row.get("seed", 0)),
            condition_type=str(row.get("condition_type", "")),
            k=int(row.get("k", 0)),
            evolved_agents=tuple(row.get("evolved_agents") or []),
        )
        if key not in lookup:
            lookup[key] = row
    return lookup


def _decorate_run_record(
    base_record: dict[str, Any],
    *,
    run_id: str,
    experiment: str,
    phase: str,
    condition_label: str,
    alpha_vector: dict[str, float],
    factorial_depth: int | None,
    factorial_profiles: str | None,
    reused_from_k_sweep: bool,
) -> dict[str, Any]:
    out = copy.deepcopy(base_record)
    canon_alpha = _canonical_alpha_vector(alpha_vector)
    out["run_id"] = run_id
    out["experiment"] = experiment
    out["phase"] = phase
    out["condition_label"] = condition_label
    out["alpha_vector"] = {agent: float(canon_alpha[agent]) for agent in AGENT_ORDER}
    out["factorial_depth"] = factorial_depth
    out["factorial_profiles"] = factorial_profiles
    out["validity_adjusted_score"] = _validity_adjusted_score(
        float(out.get("mean_score", 0.0)),
        int(out.get("valid_arc_count", 0)),
        total_agents=len(AGENT_ORDER),
    )
    out["reused_from_k_sweep"] = bool(reused_from_k_sweep)
    return out


def _choose_best_coordinate_alpha(alpha_stats: dict[float, dict[str, float | int]]) -> float:
    return max(
        alpha_stats,
        key=lambda alpha: (
            float(alpha_stats[alpha]["va_score"]),
            float(alpha_stats[alpha]["valid_rate"]),
            float(alpha_stats[alpha]["mean"]),
            -float(alpha),
        ),
    )


def _condition_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    stats = _summary_stats(runs)
    return {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "valid_rate": float(stats["valid_rate"]),
        "va_score": float(stats["va_score"]),
        "n": int(stats["n"]),
    }


def _effect_delta(conditions: dict[str, dict[str, float]], left: str, right: str, metric: str) -> float:
    return float(conditions[left][metric] - conditions[right][metric])


def _fmt_pct(value: float) -> str:
    return f"{(100.0 * float(value)):.1f}%"


def _print_summary(payload: dict[str, Any]) -> None:
    alpha = payload["analysis"]["alpha_sweep"]
    factorial = payload["analysis"]["factorial"]

    print("\n=== ALPHA-SWEEP RESULTS ===\n")
    print("Thorne alpha-curve (others fixed at alpha=1.0):")
    for alpha_key in sorted(alpha["thorne_alpha_curve"], key=lambda x: float(x)):
        row = alpha["thorne_alpha_curve"][alpha_key]
        print(
            f"  alpha={float(alpha_key):.2f}: "
            f"Q={float(row['mean']):.4f}  "
            f"valid={_fmt_pct(float(row['valid_rate']))}  "
            f"VA={float(row['va_score']):.4f}"
        )

    print("\n  Optimal alpha (by VA score):", alpha["thorne_optimal_alpha"]["by_validity_adjusted"])
    print("  Optimal alpha (by raw mean):", alpha["thorne_optimal_alpha"]["by_mean_score"])

    print("\nCoordinate descent result (Thorne locked from A1):")
    for agent in AGENT_ORDER:
        print(f"  {agent:<7}: alpha*={float(alpha['coordinate_descent']['optimal_alphas'][agent]):.3f}")

    print("\nOptimal alpha-vector vs references (50 seeds):")
    print("                  Mean Q    Valid%   VA Score")
    labels = [
        ("optimal_alpha", "Optimal alpha"),
        ("full_evolution", "Full (k=6)"),
        ("excl_thorne_k5", "Excl-Thorne"),
        ("baseline", "Baseline"),
        ("fresh", "Fresh"),
    ]
    for key, title in labels:
        row = alpha["optimal_vs_references"].get(key)
        if row is None:
            continue
        print(
            f"  {title:<12}: {float(row['mean']):.4f}    "
            f"{_fmt_pct(float(row['valid_rate'])):<7}  {float(row['va_score']):.4f}"
        )

    print("\n=== FACTORIAL RESULTS ===\n")
    if factorial.get("conditions"):
        print("                  Mean Q    Valid%   VA Score")
        for key in ["d0_default", "d0_evolved", "d0_full", "d2_default", "d2_evolved", "d2_full"]:
            row = factorial["conditions"].get(key)
            if row is None:
                continue
            print(
                f"  {key:<12}: {float(row['mean']):.4f}    "
                f"{_fmt_pct(float(row['valid_rate'])):<7}  {float(row['va_score']):.4f}"
            )

        decomp = factorial.get("effect_decomposition") or {}
        if decomp:
            print("\nEffect decomposition:")
            print(
                f"  Amplifier effect:      {float(decomp['amplifier_effect']):+.4f} "
                f"(VA: {float(decomp['amplifier_effect_va']):+.4f})"
            )
            print(
                f"  Degradation effect:    {float(decomp['degradation_effect']):+.4f} "
                f"(VA: {float(decomp['degradation_effect_va']):+.4f})"
            )
            print(
                f"  Repair effect:         {float(decomp['repair_effect']):+.4f} "
                f"(VA: {float(decomp['repair_effect_va']):+.4f})"
            )
            print(
                f"  Residual degradation:  {float(decomp['residual_degradation']):+.4f} "
                f"(VA: {float(decomp['residual_degradation_va']):+.4f})"
            )
            print(
                f"  Overshoot:             {float(decomp['overshoot_effect']):+.4f} "
                f"(VA: {float(decomp['overshoot_effect_va']):+.4f})"
            )
    else:
        print("Factorial phase skipped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase-2 alpha/factorial experiments.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seed list: comma-separated (e.g., 1,2,3) or inclusive range (e.g., 1-50).",
    )
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument(
        "--wound-analysis",
        type=str,
        default=str(DEFAULT_WOUND_PATH),
    )
    parser.add_argument(
        "--k-sweep-input",
        type=str,
        default=str(DEFAULT_K_SWEEP_PATH),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N newly completed runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
    )
    parser.add_argument(
        "--skip-factorial",
        action="store_true",
        help="Skip Experiment B factorial conditions.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Experiment A3 reference validation.",
    )
    parser.add_argument(
        "--resweep-thorne",
        action="store_true",
        help="Opt-in: include Thorne in A2 coarse-grid coordinate descent.",
    )
    args = parser.parse_args()

    start_time = time.monotonic()
    output_path = _resolve_path(args.output)
    wound_path = _resolve_path(args.wound_analysis)
    k_sweep_path = _resolve_path(args.k_sweep_input)

    wounds = _load_wound_patterns_or_fail(wound_path)
    k_sweep_payload = _load_k_sweep_payload(k_sweep_path)
    k_sweep_runs = list(k_sweep_payload.get("runs") or [])
    k_sweep_lookup = _build_k_sweep_lookup(k_sweep_runs)

    evolved_raw = ((k_sweep_payload.get("config") or {}).get("evolution_profiles") or {})
    evolved_profiles = _deserialize_evolution_profiles(evolved_raw)
    baseline_profiles = _extract_baseline_profiles()

    k_sweep_seeds = [int(x) for x in ((k_sweep_payload.get("config") or {}).get("seeds") or [])]
    if args.seeds is None:
        seeds_all = sorted(dict.fromkeys(k_sweep_seeds))
    else:
        seeds_all = _parse_seeds(args.seeds)
    if not seeds_all:
        raise ValueError("No seeds available for phase2 experiment.")

    seeds_20 = list(seeds_all[:20])
    if not seeds_20:
        raise ValueError("No seeds available for alpha-sweep phase.")

    print("Running determinism gate...")
    gate_canon = _generate_canon_after_b_for_seed(7, args.event_limit, args.tick_limit)
    determinism_gate = _run_determinism_gate(
        canon_after_b=gate_canon,
        event_limit=args.event_limit,
        tick_limit=args.tick_limit,
        full_profiles=evolved_profiles,
        wounds=wounds,
    )
    if not determinism_gate.get("ok"):
        raise RuntimeError(f"Determinism gate failed: {determinism_gate}")

    runs: list[dict[str, Any]] = []
    run_index: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        loaded_runs = list(prior.get("runs") or [])
        for row in loaded_runs:
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            runs.append(row)
            run_index[run_id] = row
        print(f"Resuming from {output_path}: {len(runs)} completed runs found.")

    canon_cache: dict[int, WorldCanon] = {}

    def _canon_for_seed(seed: int) -> WorldCanon:
        if seed not in canon_cache:
            canon_cache[seed] = _generate_canon_after_b_for_seed(seed, args.event_limit, args.tick_limit)
        return canon_cache[seed]

    checkpoint_counter = 0

    def _write_payload(analysis: dict[str, Any]) -> None:
        payload = {
            "experiment": "phase2_alpha_factorial",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "seeds": [int(s) for s in seeds_all],
                "alpha_seeds": [int(s) for s in seeds_20],
                "event_limit": int(args.event_limit),
                "tick_limit": int(args.tick_limit),
                "checkpoint_every": int(args.checkpoint_every),
                "resume": bool(args.resume),
                "resweep_thorne": bool(args.resweep_thorne),
                "skip_factorial": bool(args.skip_factorial),
                "skip_validation": bool(args.skip_validation),
                "k_sweep_input": str(k_sweep_path),
                "wound_analysis": str(wound_path),
                "determinism_gate": determinism_gate,
                "evolution_profiles": _serialize_profiles(evolved_profiles),
                "baseline_profiles": baseline_profiles,
                "runtime_seconds": float(time.monotonic() - start_time),
            },
            "runs": runs,
            "analysis": analysis,
        }
        _save_json(output_path, payload)

    def _append_run(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal checkpoint_counter
        run_id = str(record["run_id"])
        runs.append(record)
        run_index[run_id] = record
        checkpoint_counter += 1
        if checkpoint_counter >= max(1, int(args.checkpoint_every)):
            checkpoint_counter = 0
            _write_payload({})
            print(f"Checkpoint: {len(runs)} runs saved to {output_path}")
        return record

    def _run_or_load(
        *,
        run_id: str,
        seed: int,
        condition_type: str,
        condition_label: str,
        alpha_vector: dict[str, float],
        use_canon: bool,
        experiment: str,
        phase: str,
        factorial_depth: int | None,
        factorial_profiles: str | None,
        reuse_from_k_sweep_key: tuple[int, str, int, tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        existing = run_index.get(run_id)
        if existing is not None:
            return existing

        if reuse_from_k_sweep_key is not None:
            reused = k_sweep_lookup.get(reuse_from_k_sweep_key)
            if reused is not None:
                return _append_run(
                    _decorate_run_record(
                        reused,
                        run_id=run_id,
                        experiment=experiment,
                        phase=phase,
                        condition_label=condition_label,
                        alpha_vector=alpha_vector,
                        factorial_depth=factorial_depth,
                        factorial_profiles=factorial_profiles,
                        reused_from_k_sweep=True,
                    )
                )

        active = _active_agents(alpha_vector)
        spec = KSConditionSpec(
            condition_type=condition_type,
            k=len(active),
            evolved_agents=active,
            use_canon=bool(use_canon),
        )
        interpolated = _build_interpolated_evolutions(alpha_vector, baseline_profiles, evolved_profiles)
        loaded_canon = _canon_for_seed(seed) if use_canon else None

        base_record = _run_to_record(
            seed=seed,
            spec=spec,
            loaded_canon=loaded_canon,
            event_limit=args.event_limit,
            tick_limit=args.tick_limit,
            full_profiles=interpolated,
            wounds=wounds,
            collect_components=True,
        )

        return _append_run(
            _decorate_run_record(
                base_record,
                run_id=run_id,
                experiment=experiment,
                phase=phase,
                condition_label=condition_label,
                alpha_vector=alpha_vector,
                factorial_depth=factorial_depth,
                factorial_profiles=factorial_profiles,
                reused_from_k_sweep=False,
            )
        )

    print("Running A1: Thorne-only alpha sweep...")
    a1_by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for alpha in A1_ALPHAS:
        vec = _alpha_vector_default(1.0)
        vec["thorne"] = float(alpha)
        for seed in seeds_20:
            run_id = f"a1|seed={seed}|alpha={alpha:.3f}"
            rec = _run_or_load(
                run_id=run_id,
                seed=seed,
                condition_type="alpha_sweep",
                condition_label=f"a1_thorne_{alpha:.3f}",
                alpha_vector=vec,
                use_canon=True,
                experiment="alpha_sweep",
                phase="A1",
                factorial_depth=None,
                factorial_profiles=None,
            )
            a1_by_alpha[float(alpha)].append(rec)

    thorne_curve: dict[float, dict[str, float | int]] = {
        alpha: _summary_stats(rows) for alpha, rows in sorted(a1_by_alpha.items())
    }

    a1_best_by_va = _best_alpha_by_metric(thorne_curve, "va_score")
    thorne_optimal_alpha = {
        "by_mean_score": float(_best_alpha_by_metric(thorne_curve, "mean")),
        "by_validity_adjusted": float(a1_best_by_va),
        "by_pareto": float(_build_pareto_alpha(thorne_curve)),
    }

    print("Running A2: coordinate descent sweep...")
    current_alphas = _alpha_vector_default(1.0)
    current_alphas["thorne"] = float(a1_best_by_va)

    a2_agent_order = list(A2_AGENT_ORDER)
    if args.resweep_thorne:
        a2_agent_order = ["thorne", *a2_agent_order]

    convergence_history: list[dict[str, Any]] = []
    for agent in a2_agent_order:
        alpha_stats: dict[float, dict[str, float | int]] = {}
        for alpha in A2_ALPHAS:
            trial = dict(current_alphas)
            trial[agent] = float(alpha)
            trial_rows: list[dict[str, Any]] = []
            for seed in seeds_20:
                run_id = (
                    f"a2|seed={seed}|agent={agent}|alpha={alpha:.3f}|"
                    f"vector={_alpha_vector_token(trial)}"
                )
                rec = _run_or_load(
                    run_id=run_id,
                    seed=seed,
                    condition_type="alpha_coord",
                    condition_label=f"a2_{agent}_{alpha:.3f}",
                    alpha_vector=trial,
                    use_canon=True,
                    experiment="alpha_sweep",
                    phase="A2",
                    factorial_depth=None,
                    factorial_profiles=None,
                )
                trial_rows.append(rec)

            stats = _summary_stats(trial_rows)
            alpha_stats[float(alpha)] = stats
            convergence_history.append(
                {
                    "agent": agent,
                    "alpha": float(alpha),
                    "alpha_vector": _canonical_alpha_vector(trial),
                    "mean_q": float(stats["mean"]),
                    "va_score": float(stats["va_score"]),
                    "valid_rate": float(stats["valid_rate"]),
                }
            )

        best_alpha = _choose_best_coordinate_alpha(alpha_stats)
        current_alphas[agent] = float(best_alpha)

    final_alphas = _canonical_alpha_vector(current_alphas)

    if not args.resweep_thorne and abs(float(final_alphas["thorne"]) - float(a1_best_by_va)) > 1e-12:
        raise RuntimeError("A2 Thorne handoff violated: coarse sweep changed locked A1 Thorne alpha.")

    if not args.skip_validation:
        print("Running A3: optimal-vs-reference validation...")
    a3_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds_all:
        # Optimal alpha (depth-2)
        run_opt = _run_or_load(
            run_id=f"a3|seed={seed}|condition=optimal_alpha",
            seed=seed,
            condition_type="alpha_optimal",
            condition_label="optimal_alpha",
            alpha_vector=final_alphas,
            use_canon=True,
            experiment="alpha_sweep",
            phase="A3",
            factorial_depth=None,
            factorial_profiles=None,
        )
        a3_by_label["optimal_alpha"].append(run_opt)

        if args.skip_validation:
            continue

        # Baseline (depth-2 default) -> k-sweep baseline
        run_base = _run_or_load(
            run_id=f"a3|seed={seed}|condition=baseline",
            seed=seed,
            condition_type="baseline",
            condition_label="baseline",
            alpha_vector=_alpha_vector_default(0.0),
            use_canon=True,
            experiment="alpha_sweep",
            phase="A3",
            factorial_depth=None,
            factorial_profiles=None,
            reuse_from_k_sweep_key=_k_sweep_key(seed, "baseline", 0, ()),
        )
        a3_by_label["baseline"].append(run_base)

        # Fresh (depth-0 default) -> k-sweep fresh
        run_fresh = _run_or_load(
            run_id=f"a3|seed={seed}|condition=fresh",
            seed=seed,
            condition_type="fresh",
            condition_label="fresh",
            alpha_vector=_alpha_vector_default(0.0),
            use_canon=False,
            experiment="alpha_sweep",
            phase="A3",
            factorial_depth=None,
            factorial_profiles=None,
            reuse_from_k_sweep_key=_k_sweep_key(seed, "fresh", 0, ()),
        )
        a3_by_label["fresh"].append(run_fresh)

        # Full evolution (depth-2, alpha=1.0) -> k-sweep full_evolution
        run_full = _run_or_load(
            run_id=f"a3|seed={seed}|condition=full_evolution",
            seed=seed,
            condition_type="full_evolution",
            condition_label="full_evolution",
            alpha_vector=_alpha_vector_default(1.0),
            use_canon=True,
            experiment="alpha_sweep",
            phase="A3",
            factorial_depth=None,
            factorial_profiles=None,
            reuse_from_k_sweep_key=_k_sweep_key(seed, "full_evolution", 6, tuple(AGENT_ORDER)),
        )
        a3_by_label["full_evolution"].append(run_full)

        # Excluding Thorne (depth-2, k=5 subset)
        excl_vec = _alpha_vector_default(1.0)
        excl_vec["thorne"] = 0.0
        run_excl = _run_or_load(
            run_id=f"a3|seed={seed}|condition=excl_thorne_k5",
            seed=seed,
            condition_type="subset",
            condition_label="excl_thorne_k5",
            alpha_vector=excl_vec,
            use_canon=True,
            experiment="alpha_sweep",
            phase="A3",
            factorial_depth=None,
            factorial_profiles=None,
            reuse_from_k_sweep_key=_k_sweep_key(
                seed,
                "subset",
                5,
                ("elena", "marcus", "lydia", "diana", "victor"),
            ),
        )
        a3_by_label["excl_thorne_k5"].append(run_excl)

    factorial_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not args.skip_factorial:
        print("Running B: amplifier/repair factorial...")
        factorial_conditions: list[tuple[str, bool, dict[str, float], int, str, tuple[int, str, int, tuple[str, ...]] | None]] = [
            (
                "d0_default",
                False,
                _alpha_vector_default(0.0),
                0,
                "default",
                None,
            ),
            (
                "d0_evolved",
                False,
                final_alphas,
                0,
                "evolved",
                None,
            ),
            (
                "d0_full",
                False,
                _alpha_vector_default(1.0),
                0,
                "full",
                None,
            ),
            (
                "d2_default",
                True,
                _alpha_vector_default(0.0),
                2,
                "default",
                _k_sweep_key(0, "baseline", 0, ()),
            ),
            (
                "d2_evolved",
                True,
                final_alphas,
                2,
                "evolved",
                None,
            ),
            (
                "d2_full",
                True,
                _alpha_vector_default(1.0),
                2,
                "full",
                _k_sweep_key(0, "full_evolution", 6, tuple(AGENT_ORDER)),
            ),
        ]

        for seed in seeds_all:
            for label, use_canon, alpha_vec, depth, profile_type, templated_reuse in factorial_conditions:
                reuse_key = None
                if templated_reuse is not None:
                    reuse_key = (seed, templated_reuse[1], templated_reuse[2], templated_reuse[3])
                if label == "d0_default":
                    reuse_key = _k_sweep_key(seed, "fresh", 0, ())
                rec = _run_or_load(
                    run_id=f"b|seed={seed}|condition={label}",
                    seed=seed,
                    condition_type=f"factorial_{label}",
                    condition_label=label,
                    alpha_vector=alpha_vec,
                    use_canon=use_canon,
                    experiment="factorial",
                    phase="B",
                    factorial_depth=depth,
                    factorial_profiles=profile_type,
                    reuse_from_k_sweep_key=reuse_key,
                )
                factorial_by_label[label].append(rec)

    thorne_curve_json = {
        f"{alpha:.1f}": {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "valid_rate": float(stats["valid_rate"]),
            "va_score": float(stats["va_score"]),
            "n": int(stats["n"]),
        }
        for alpha, stats in sorted(thorne_curve.items())
    }

    optimal_vs_refs: dict[str, dict[str, float]] = {
        "optimal_alpha": _condition_summary(a3_by_label["optimal_alpha"]),
    }
    if not args.skip_validation:
        optimal_vs_refs.update(
            {
                "full_evolution": _condition_summary(a3_by_label["full_evolution"]),
                "excl_thorne_k5": _condition_summary(a3_by_label["excl_thorne_k5"]),
                "baseline": _condition_summary(a3_by_label["baseline"]),
                "fresh": _condition_summary(a3_by_label["fresh"]),
            }
        )

    alpha_analysis = {
        "thorne_alpha_curve": thorne_curve_json,
        "thorne_optimal_alpha": thorne_optimal_alpha,
        "coordinate_descent": {
            "optimal_alphas": {agent: float(final_alphas[agent]) for agent in AGENT_ORDER},
            "convergence_history": convergence_history,
            "thorne_locked_from_a1": bool(not args.resweep_thorne),
            "a1_locked_thorne_alpha": float(a1_best_by_va),
        },
        "optimal_vs_references": optimal_vs_refs,
    }

    factorial_analysis: dict[str, Any] = {
        "conditions": {},
        "effect_decomposition": {},
        "canon_benefit_analysis": {},
    }

    if not args.skip_factorial:
        condition_summary = {label: _condition_summary(rows) for label, rows in sorted(factorial_by_label.items())}
        factorial_analysis["conditions"] = condition_summary

        effects = {
            "amplifier_effect": _effect_delta(condition_summary, "d0_evolved", "d0_default", "mean"),
            "degradation_effect": _effect_delta(condition_summary, "d2_default", "d0_default", "mean"),
            "repair_effect": _effect_delta(condition_summary, "d2_evolved", "d2_default", "mean"),
            "residual_degradation": _effect_delta(condition_summary, "d2_evolved", "d0_evolved", "mean"),
            "overshoot_effect": _effect_delta(condition_summary, "d0_full", "d0_evolved", "mean"),
            "amplifier_effect_va": _effect_delta(condition_summary, "d0_evolved", "d0_default", "va_score"),
            "degradation_effect_va": _effect_delta(condition_summary, "d2_default", "d0_default", "va_score"),
            "repair_effect_va": _effect_delta(condition_summary, "d2_evolved", "d2_default", "va_score"),
            "residual_degradation_va": _effect_delta(condition_summary, "d2_evolved", "d0_evolved", "va_score"),
            "overshoot_effect_va": _effect_delta(condition_summary, "d0_full", "d0_evolved", "va_score"),
        }
        factorial_analysis["effect_decomposition"] = effects

        d0_by_seed = {int(row["seed"]): row for row in factorial_by_label["d0_default"]}
        d2_by_seed = {int(row["seed"]): row for row in factorial_by_label["d2_default"]}

        canon_effects: list[float] = []
        entropy_predictor: list[float] = []
        boundary_predictor: list[float] = []
        wound_predictor: list[float] = []
        helps = 0
        hurts = 0

        for seed in sorted(set(d0_by_seed) & set(d2_by_seed)):
            d0_row = d0_by_seed[seed]
            d2_row = d2_by_seed[seed]
            effect = float(d2_row["mean_score"]) - float(d0_row["mean_score"])
            canon_effects.append(effect)
            if effect > 0.0:
                helps += 1
            elif effect < 0.0:
                hurts += 1

            entropy_predictor.append(float((d0_row.get("belief_entropy") or {}).get("start", 0.0)))
            boundary_predictor.append(float((d0_row.get("relationship_extremes") or {}).get("boundary_fraction", 0.0)))
            wound_predictor.append(float(d0_row.get("wound_count", 0.0)))

        factorial_analysis["canon_benefit_analysis"] = {
            "seeds_where_canon_helps": int(helps),
            "seeds_where_canon_hurts": int(hurts),
            "mean_canon_effect": float(_mean(canon_effects)),
            "std_canon_effect": float(_std(canon_effects)),
            "correlation_belief_entropy_vs_canon_effect": float(_pearson(entropy_predictor, canon_effects)),
            "correlation_boundary_fraction_vs_canon_effect": float(_pearson(boundary_predictor, canon_effects)),
            "correlation_wound_count_vs_canon_effect": float(_pearson(wound_predictor, canon_effects)),
        }

    analysis = {
        "alpha_sweep": alpha_analysis,
        "factorial": factorial_analysis,
    }

    _write_payload(analysis)
    _print_summary({"analysis": analysis})
    print(f"\nSaved experiment artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
