"""Re-score-only diagnostic: strict arcs passed through relaxed validator.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.rescore_only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.rashomon import extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.k_sweep_experiment import (
    DINNER_PARTY_AGENTS,
    _generate_canon_after_b_for_seed,
    _simulate_story,
)
from scripts.relaxed_arc_validator import validate_arc_relaxed
from scripts.test_goal_evolution import _evolution_profiles

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "rescore_only.json"
DEFAULT_GRAMMAR_RELAX_PATH = OUTPUT_DIR / "grammar_relaxation.json"
EPSILON = 1e-10


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    evolved_agents: tuple[str, ...]


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


def _condition_specs() -> list[ConditionSpec]:
    all_agents = tuple(DINNER_PARTY_AGENTS)
    excl_thorne = tuple(agent for agent in DINNER_PARTY_AGENTS if agent != "thorne")
    return [
        ConditionSpec(key="baseline", label="baseline", evolved_agents=()),
        ConditionSpec(key="full_evolution", label="full_evolution", evolved_agents=all_agents),
        ConditionSpec(key="excl_thorne", label="excl_thorne", evolved_agents=excl_thorne),
    ]


def _strict_summary_from_rashomon(rashomon) -> dict[str, Any]:
    valid_scores: list[float] = []
    invalid_agents: list[str] = []
    for arc in rashomon.arcs:
        if arc.valid and arc.arc_score is not None:
            valid_scores.append(float(arc.arc_score.composite))
        else:
            invalid_agents.append(str(arc.protagonist))
    return {
        "mean_score": float(_mean(valid_scores)),
        "invalid_agents": sorted(invalid_agents),
        "valid_arc_count": int(rashomon.valid_count),
    }


def _load_grammar_relaxation_reference(path: Path, seeds: list[int]) -> dict[tuple[int, str], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = list(raw.get("per_seed") or [])
    reference: dict[tuple[int, str], dict[str, Any]] = {}
    seed_set = set(seeds)
    for row in rows:
        seed = int(row.get("seed", 0))
        if seed not in seed_set:
            continue
        reference[(seed, "baseline")] = {
            "mean_score": float((row.get("baseline_strict") or {}).get("mean_score", 0.0)),
            "invalid_agents": sorted(str(agent) for agent in ((row.get("baseline_strict") or {}).get("invalid_agents") or [])),
        }
        reference[(seed, "full_evolution")] = {
            "mean_score": float((row.get("full_strict") or {}).get("mean_score", 0.0)),
            "invalid_agents": sorted(str(agent) for agent in ((row.get("full_strict") or {}).get("invalid_agents") or [])),
        }
        reference[(seed, "excl_thorne")] = {
            "mean_score": float((row.get("excl_thorne_strict") or {}).get("mean_score", 0.0)),
            "invalid_agents": sorted(
                str(agent) for agent in ((row.get("excl_thorne_strict") or {}).get("invalid_agents") or [])
            ),
        }

    required = {(seed, cond) for seed in seeds for cond in ("baseline", "full_evolution", "excl_thorne")}
    missing = sorted(required - set(reference.keys()))
    if missing:
        preview = ", ".join(f"{seed}:{cond}" for seed, cond in missing[:10])
        raise ValueError(f"Missing strict reference rows in grammar_relaxation.json: {preview}")
    return reference


def _violation_category(message: str) -> str:
    text = message.lower()
    if "turning_point" in text:
        return "turning_point_count"
    if "order violation" in text or "phase regressions" in text:
        return "phase_monotonicity"
    if "no protagonist" in text:
        return "protagonist_centrality"
    if "causal gap" in text:
        return "causal_continuity"
    if "arc too short" in text or "spans" in text:
        return "time_span"
    if "too few beats" in text or "too many beats" in text:
        return "beat_count"
    return "other"


def _empty_violation_counter() -> dict[str, int]:
    return {
        "turning_point_count": 0,
        "phase_monotonicity": 0,
        "protagonist_centrality": 0,
        "causal_continuity": 0,
        "time_span": 0,
        "beat_count": 0,
        "other": 0,
    }


def _print_summary(
    *,
    monotonicity: dict[str, Any],
    recovery_ceiling: dict[str, Any],
    still_invalid_violation_types: dict[str, int],
    diana_analysis: dict[str, Any],
    implied_all_valid_rate: dict[str, dict[str, float]],
) -> None:
    print()
    print("=== MONOTONICITY INVARIANT ===")
    status = "PASS" if bool(monotonicity["invariant_holds"]) else "FAIL"
    print(
        f"[{status}]: {int(monotonicity['n_strict_valid'])} strict-valid arcs tested, "
        f"{int(monotonicity['n_strict_valid_and_relaxed_valid'])} passed relaxed validation"
    )
    print()

    n_strict_invalid = int(recovery_ceiling["n_strict_invalid"])
    n_recovered = int(recovery_ceiling["n_recovered"])
    n_still = int(recovery_ceiling["n_still_invalid"])
    recovered_pct = (100.0 * n_recovered / n_strict_invalid) if n_strict_invalid else 0.0
    still_pct = (100.0 * n_still / n_strict_invalid) if n_strict_invalid else 0.0
    print("=== RECOVERY CEILING ===")
    print(f"Strict-invalid arcs: {n_strict_invalid} total")
    print(f"  Recovered by relaxed validator: {n_recovered} ({recovered_pct:.1f}%)")
    print(f"  Still invalid under relaxed: {n_still} ({still_pct:.1f}%)")
    print()

    print("Recovery by agent (full_evolution):")
    print("  agent      strict_invalid  recovered  still_invalid")
    by_agent_full = recovery_ceiling["by_agent_full_evolution"]
    for agent in DINNER_PARTY_AGENTS:
        row = by_agent_full[agent]
        print(
            f"  {agent:<10} {int(row['strict_invalid']):>13} "
            f"{int(row['recovered']):>10} {int(row['still_invalid']):>13}"
        )
    print()

    print("Recovery by condition:")
    print("  condition       strict_invalid  recovered")
    by_condition = recovery_ceiling["by_condition"]
    for condition in ("baseline", "full_evolution", "excl_thorne"):
        row = by_condition[condition]
        print(f"  {condition:<15} {int(row['strict_invalid']):>13} {int(row['recovered']):>10}")
    print()

    print("Recovered arcs failed strict grammar for:")
    for key, value in recovery_ceiling["recovered_violation_types"].items():
        print(f"  {key}: {int(value)}")
    print()

    print("Still-invalid arcs fail relaxed grammar for:")
    for key, value in still_invalid_violation_types.items():
        print(f"  {key}: {int(value)}")
    print()

    print("=== DIANA ANALYSIS (full_evolution) ===")
    print(f"Diana strict-invalid: {int(diana_analysis['strict_invalid_count'])}")
    print(f"Diana recovered by relaxed: {int(diana_analysis['relaxed_valid_count'])}")
    print(f"Diana strict violations: {diana_analysis['strict_violations']}")
    diana_mean = diana_analysis["recovered_mean_score"]
    mean_text = "null" if diana_mean is None else f"{float(diana_mean):.4f}"
    print(f"Diana recovered mean score: {mean_text}")
    print()

    print("=== IMPLIED ALL-VALID RATES (strict search + relaxed validator) ===")
    print("  condition       strict    rescore_relaxed")
    for condition in ("baseline", "full_evolution", "excl_thorne"):
        row = implied_all_valid_rate[condition]
        strict_text = f"{100.0 * float(row['strict']):.0f}%"
        relaxed_text = f"{100.0 * float(row['rescore_relaxed']):.0f}%"
        print(f"  {condition:<15} {strict_text:<8}  {relaxed_text:<8}")
    print()

    print("=== VERDICT ===")
    if bool(monotonicity["invariant_holds"]):
        print(
            "Validator is a true relaxation. The grammar relaxation collapse was caused by "
            "search dynamics, not validator mis-specification. The strict grammar's search "
            "heuristics act as a structural regularizer."
        )
    else:
        print(
            "MONOTONICITY VIOLATION: relaxed validator is not a superset of strict. "
            "Grammar relaxation results cannot be interpreted until this is fixed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score strict arcs using relaxed validator only.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--grammar-relaxation", type=str, default=str(DEFAULT_GRAMMAR_RELAX_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    grammar_relax_path = _resolve_path(args.grammar_relaxation)
    if not grammar_relax_path.exists():
        raise FileNotFoundError(f"grammar_relaxation output not found: {grammar_relax_path}")

    seeds = _parse_seeds(args.seeds)
    conditions = _condition_specs()
    profiles_full = _evolution_profiles()["full"]
    strict_reference = _load_grammar_relaxation_reference(grammar_relax_path, seeds)

    strict_match_errors: list[dict[str, Any]] = []
    monotonicity_violations: list[dict[str, Any]] = []

    n_strict_valid = 0
    n_strict_valid_and_relaxed_valid = 0
    n_strict_invalid = 0
    n_recovered = 0
    n_still_invalid = 0

    by_agent = {agent: {"strict_invalid": 0, "recovered": 0, "still_invalid": 0} for agent in DINNER_PARTY_AGENTS}
    by_agent_full = {agent: {"strict_invalid": 0, "recovered": 0, "still_invalid": 0} for agent in DINNER_PARTY_AGENTS}
    by_condition = {
        "baseline": {"strict_invalid": 0, "recovered": 0},
        "full_evolution": {"strict_invalid": 0, "recovered": 0},
        "excl_thorne": {"strict_invalid": 0, "recovered": 0},
    }

    recovered_violation_types = _empty_violation_counter()
    still_invalid_violation_types = _empty_violation_counter()
    recovered_scores: list[float] = []

    diana_full_strict_invalid_count = 0
    diana_full_relaxed_valid_count = 0
    diana_strict_violations: list[str] = []
    diana_recovered_scores: list[float] = []

    per_run_valid_counts: dict[tuple[int, str], dict[str, int]] = {}

    total_runs = len(seeds) * len(conditions)
    done = 0
    for seed in seeds:
        canon_after_b = _generate_canon_after_b_for_seed(
            seed=seed,
            event_limit=int(args.event_limit),
            tick_limit=int(args.tick_limit),
        )
        for condition in conditions:
            done += 1
            print(f"[{done:03d}/{total_runs:03d}] seed={seed} condition={condition.key}", flush=True)

            evolutions = {agent: profiles_full[agent] for agent in condition.evolved_agents}
            story = _simulate_story(
                label=condition.key,
                seed=seed,
                loaded_canon=canon_after_b,
                tick_limit=int(args.tick_limit),
                event_limit=int(args.event_limit),
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
            strict_summary = _strict_summary_from_rashomon(strict_rashomon)
            expected = strict_reference[(seed, condition.key)]
            mean_delta = abs(float(strict_summary["mean_score"]) - float(expected["mean_score"]))
            invalid_match = list(strict_summary["invalid_agents"]) == list(expected["invalid_agents"])
            if mean_delta > EPSILON or not invalid_match:
                strict_match_errors.append(
                    {
                        "seed": int(seed),
                        "condition": condition.key,
                        "mean_score_expected": float(expected["mean_score"]),
                        "mean_score_actual": float(strict_summary["mean_score"]),
                        "mean_score_abs_delta": float(mean_delta),
                        "invalid_agents_expected": list(expected["invalid_agents"]),
                        "invalid_agents_actual": list(strict_summary["invalid_agents"]),
                    }
                )

            strict_valid_count = 0
            relaxed_valid_count = 0

            for arc in strict_rashomon.arcs:
                agent = str(arc.protagonist)
                strict_valid = bool(arc.valid)
                strict_violations = list(arc.violations)
                relaxed_validation = validate_arc_relaxed(
                    events=arc.events,
                    beats=arc.beats,
                    total_sim_time=total_sim_time,
                )
                relaxed_valid = bool(relaxed_validation.valid)
                relaxed_violations = list(relaxed_validation.violations)

                if strict_valid:
                    strict_valid_count += 1
                    n_strict_valid += 1
                    if relaxed_valid:
                        relaxed_valid_count += 1
                        n_strict_valid_and_relaxed_valid += 1
                    else:
                        monotonicity_violations.append(
                            {
                                "seed": int(seed),
                                "condition": condition.key,
                                "agent": agent,
                                "strict_violations": strict_violations,
                                "relaxed_violations": relaxed_violations,
                            }
                        )
                else:
                    n_strict_invalid += 1
                    by_agent[agent]["strict_invalid"] += 1
                    by_condition[condition.key]["strict_invalid"] += 1
                    if condition.key == "full_evolution":
                        by_agent_full[agent]["strict_invalid"] += 1
                    if condition.key == "full_evolution" and agent == "diana":
                        diana_full_strict_invalid_count += 1
                        diana_strict_violations.extend(strict_violations)

                    if relaxed_valid:
                        relaxed_valid_count += 1
                        n_recovered += 1
                        by_agent[agent]["recovered"] += 1
                        by_condition[condition.key]["recovered"] += 1
                        if condition.key == "full_evolution":
                            by_agent_full[agent]["recovered"] += 1

                        recovered_scores.append(float(score_arc(arc.events, arc.beats).composite))
                        categories = {_violation_category(v) for v in strict_violations}
                        for category in categories:
                            recovered_violation_types[category] += 1

                        if condition.key == "full_evolution" and agent == "diana":
                            diana_full_relaxed_valid_count += 1
                            diana_recovered_scores.append(float(score_arc(arc.events, arc.beats).composite))
                    else:
                        n_still_invalid += 1
                        by_agent[agent]["still_invalid"] += 1
                        if condition.key == "full_evolution":
                            by_agent_full[agent]["still_invalid"] += 1
                        categories = {_violation_category(v) for v in relaxed_violations}
                        for category in categories:
                            still_invalid_violation_types[category] += 1

            per_run_valid_counts[(seed, condition.key)] = {
                "strict": int(strict_valid_count),
                "rescore_relaxed": int(relaxed_valid_count),
            }

    if strict_match_errors:
        preview = strict_match_errors[:5]
        raise RuntimeError(
            "Strict replay mismatch vs grammar_relaxation strict output; aborting. "
            f"First mismatches: {json.dumps(preview, ensure_ascii=True)}"
        )

    implied_all_valid_rate: dict[str, dict[str, float]] = {}
    for condition in ("baseline", "full_evolution", "excl_thorne"):
        strict_all = 0
        relaxed_all = 0
        for seed in seeds:
            row = per_run_valid_counts[(seed, condition)]
            if int(row["strict"]) == len(DINNER_PARTY_AGENTS):
                strict_all += 1
            if int(row["rescore_relaxed"]) == len(DINNER_PARTY_AGENTS):
                relaxed_all += 1
        implied_all_valid_rate[condition] = {
            "strict": float(strict_all / len(seeds)) if seeds else 0.0,
            "rescore_relaxed": float(relaxed_all / len(seeds)) if seeds else 0.0,
        }

    monotonicity = {
        "invariant_holds": bool(len(monotonicity_violations) == 0),
        "n_strict_valid": int(n_strict_valid),
        "n_strict_valid_and_relaxed_valid": int(n_strict_valid_and_relaxed_valid),
        "violations": monotonicity_violations,
    }
    recovery_ceiling = {
        "n_strict_invalid": int(n_strict_invalid),
        "n_recovered": int(n_recovered),
        "n_still_invalid": int(n_still_invalid),
        "n_strict_invalid_but_relaxed_valid": int(n_recovered),
        "n_strict_invalid_and_relaxed_invalid": int(n_still_invalid),
        "by_agent": by_agent,
        "by_agent_full_evolution": by_agent_full,
        "by_condition": by_condition,
        "recovered_arc_mean_score": (_mean(recovered_scores) if recovered_scores else None),
        "recovered_violation_types": recovered_violation_types,
    }
    diana_analysis = {
        "condition": "full_evolution",
        "strict_invalid_count": int(diana_full_strict_invalid_count),
        "relaxed_valid_count": int(diana_full_relaxed_valid_count),
        "strict_violations": sorted(set(diana_strict_violations)),
        "recovered_scores": [float(score) for score in diana_recovered_scores],
        "recovered_mean_score": (_mean(diana_recovered_scores) if diana_recovered_scores else None),
    }

    payload = {
        "config": {
            "total_arcs": int(len(seeds) * len(conditions) * len(DINNER_PARTY_AGENTS)),
            "total_runs": int(len(seeds) * len(conditions)),
            "seeds": [int(seed) for seed in seeds],
            "conditions": ["baseline", "full_evolution", "excl_thorne"],
        },
        "monotonicity": monotonicity,
        "recovery_ceiling": recovery_ceiling,
        "still_invalid_violation_types": still_invalid_violation_types,
        "diana_analysis": diana_analysis,
        "implied_all_valid_rate": implied_all_valid_rate,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_summary(
        monotonicity=monotonicity,
        recovery_ceiling=recovery_ceiling,
        still_invalid_violation_types=still_invalid_violation_types,
        diana_analysis=diana_analysis,
        implied_all_valid_rate=implied_all_valid_rate,
    )


if __name__ == "__main__":
    main()
