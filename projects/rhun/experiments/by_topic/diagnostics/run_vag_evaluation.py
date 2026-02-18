"""Evaluate Viability-Aware Greedy (VAG) on Layer 2/3 and broad sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BEAM_SWEEP_PATH = OUTPUT_DIR / "beam_search_sweep.json"
POOL_DIAG_PATH = OUTPUT_DIR / "pool_bottleneck_diagnosis.json"


def _case_key(epsilon: float, seed: int, focal_actor: str) -> str:
    return f"{epsilon:.2f}|{seed}|{focal_actor}"


def _eval_case(
    graph,
    focal_actor: str,
    grammar: GrammarConfig,
    pool_strategy: str,
    n_anchors: int,
    max_sequence_length: int,
    injection_top_n: int,
) -> dict:
    greedy = greedy_extract(
        graph=graph,
        focal_actor=focal_actor,
        grammar=grammar,
        pool_strategy=pool_strategy,
        n_anchors=n_anchors,
        max_sequence_length=max_sequence_length,
        injection_top_n=injection_top_n,
    )
    vag, vag_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor=focal_actor,
        grammar=grammar,
        pool_strategy=pool_strategy,
        n_anchors=n_anchors,
        max_sequence_length=max_sequence_length,
        injection_top_n=injection_top_n,
    )
    oracle, oracle_diag = oracle_extract(
        graph=graph,
        focal_actor=focal_actor,
        grammar=grammar,
        max_sequence_length=max_sequence_length,
    )

    oracle_valid = bool(oracle is not None and oracle.valid)
    greedy_valid = bool(greedy.valid)
    vag_valid = bool(vag.valid)

    greedy_ratio = None
    vag_ratio = None
    if oracle_valid and oracle is not None and oracle.score > 0:
        if greedy_valid:
            greedy_ratio = float(greedy.score / oracle.score)
        if vag_valid:
            vag_ratio = float(vag.score / oracle.score)

    return {
        "greedy_valid": greedy_valid,
        "vag_valid": vag_valid,
        "oracle_valid": oracle_valid,
        "greedy_score": float(greedy.score),
        "vag_score": float(vag.score),
        "oracle_score": (float(oracle.score) if oracle is not None else None),
        "greedy_ratio_vs_oracle": greedy_ratio,
        "vag_ratio_vs_oracle": vag_ratio,
        "greedy_violations": list(greedy.violations),
        "vag_violations": list(vag.violations),
        "oracle_violations": (list(oracle.violations) if oracle is not None else None),
        "vag_diagnostics": vag_diag,
        "oracle_diagnostics": oracle_diag,
    }


def _evaluate_case_list(
    cases: list[dict],
    generator: BurstyGenerator,
    grammar: GrammarConfig,
    n_events: int,
    n_actors: int,
    pool_strategy: str,
    n_anchors: int,
    max_sequence_length: int,
    injection_top_n: int,
) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case["focal_actor"])
        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )
        metrics = _eval_case(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_strategy=pool_strategy,
            n_anchors=n_anchors,
            max_sequence_length=max_sequence_length,
            injection_top_n=injection_top_n,
        )
        rows.append(
            {
                "epsilon": epsilon,
                "seed": seed,
                "focal_actor": focal_actor,
                "case_key": _case_key(epsilon, seed, focal_actor),
                **metrics,
            }
        )
    rows.sort(key=lambda row: (row["epsilon"], row["seed"], row["focal_actor"]))
    return rows


def _cohort_summary(rows: list[dict]) -> dict:
    n_cases = len(rows)
    greedy_valid = sum(1 for row in rows if row["greedy_valid"])
    vag_valid = sum(1 for row in rows if row["vag_valid"])
    improvements = sum(1 for row in rows if row["vag_valid"] and not row["greedy_valid"])
    regressions = sum(1 for row in rows if row["greedy_valid"] and not row["vag_valid"])
    both_valid = sum(1 for row in rows if row["greedy_valid"] and row["vag_valid"])
    both_invalid = sum(1 for row in rows if not row["greedy_valid"] and not row["vag_valid"])
    return {
        "n_cases": n_cases,
        "greedy_valid_count": greedy_valid,
        "vag_valid_count": vag_valid,
        "improvements_over_greedy": improvements,
        "regressions_vs_greedy": regressions,
        "both_valid": both_valid,
        "both_invalid": both_invalid,
    }


def _ratio_summary(rows: list[dict], key: str) -> dict:
    ratios = [row[key] for row in rows if row[key] is not None]
    if not ratios:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "below_0_90": 0,
        }
    return {
        "count": len(ratios),
        "mean": float(mean(ratios)),
        "median": float(median(ratios)),
        "min": float(min(ratios)),
        "below_0_90": int(sum(1 for value in ratios if value < 0.90)),
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    layer3 = data["cohorts"]["layer3"]
    layer2 = data["cohorts"]["layer2"]
    full = data["cohorts"]["full_sweep"]
    overall = data["overall"]
    vag_ratio = data["ratio_summary"]["vag_vs_oracle"]
    lines = [
        "# vag_evaluation",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Recovery",
        "",
        f"- Layer 3 recovery: {layer3['vag_valid_count']}/{layer3['n_cases']}",
        f"- Layer 2 recovery: {layer2['vag_valid_count']}/{layer2['n_cases']}",
        f"- Full-sweep improvements over greedy: {full['improvements_over_greedy']}",
        f"- Full-sweep regressions vs greedy: {full['regressions_vs_greedy']}",
        "",
        "## Net",
        "",
        f"- Total improvements over greedy: {overall['improvements_over_greedy']}",
        f"- Total regressions vs greedy: {overall['regressions_vs_greedy']}",
        f"- Net improvement: {overall['net_improvement']}",
        "",
        "## VAG Approximation Ratio",
        "",
        f"- Count: {vag_ratio['count']}",
        f"- Mean: {vag_ratio['mean']:.4f}" if vag_ratio["mean"] is not None else "- Mean: n/a",
        f"- Min: {vag_ratio['min']:.4f}" if vag_ratio["min"] is not None else "- Min: n/a",
        f"- Cases below 0.90: {vag_ratio['below_0_90']}",
        "",
    ]
    return "\n".join(lines)


def run_vag_evaluation() -> dict:
    if not BEAM_SWEEP_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {BEAM_SWEEP_PATH}")
    if not POOL_DIAG_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {POOL_DIAG_PATH}")

    beam = json.loads(BEAM_SWEEP_PATH.read_text(encoding="utf-8"))["results"]
    pool = json.loads(POOL_DIAG_PATH.read_text(encoding="utf-8"))["results"]

    settings = beam["settings"]
    grammar = GrammarConfig(**settings["grammar"])
    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    max_sequence_length = int(settings["max_sequence_length"])
    pool_strategy = "injection"
    n_anchors = 8
    injection_top_n = 40

    layer3_cases = [
        {
            "epsilon": float(row["epsilon"]),
            "seed": int(row["seed"]),
            "focal_actor": str(row["focal_actor"]),
        }
        for row in pool["per_case"]
        if bool(row.get("oracle_valid", False))
    ]
    layer3_cases.sort(key=lambda row: (row["epsilon"], row["seed"], row["focal_actor"]))

    layer2_cases = [
        {
            "epsilon": float(row["epsilon"]),
            "seed": int(row["seed"]),
            "focal_actor": str(row["focal_actor"]),
        }
        for row in beam["per_case"]
        if row.get("first_valid_width") == 2
    ]
    layer2_cases.sort(key=lambda row: (row["epsilon"], row["seed"], row["focal_actor"]))

    full_sweep_cases = [
        {
            "epsilon": epsilon,
            "seed": seed,
            "focal_actor": "actor_0",
        }
        for seed in range(0, 50)
        for epsilon in [0.30, 0.50, 0.70, 0.90]
    ]

    generator = BurstyGenerator()
    timer = ExperimentTimer()

    layer3_rows = _evaluate_case_list(
        cases=layer3_cases,
        generator=generator,
        grammar=grammar,
        n_events=n_events,
        n_actors=n_actors,
        pool_strategy=pool_strategy,
        n_anchors=n_anchors,
        max_sequence_length=max_sequence_length,
        injection_top_n=injection_top_n,
    )
    layer2_rows = _evaluate_case_list(
        cases=layer2_cases,
        generator=generator,
        grammar=grammar,
        n_events=n_events,
        n_actors=n_actors,
        pool_strategy=pool_strategy,
        n_anchors=n_anchors,
        max_sequence_length=max_sequence_length,
        injection_top_n=injection_top_n,
    )
    full_sweep_rows = _evaluate_case_list(
        cases=full_sweep_cases,
        generator=generator,
        grammar=grammar,
        n_events=n_events,
        n_actors=n_actors,
        pool_strategy=pool_strategy,
        n_anchors=n_anchors,
        max_sequence_length=max_sequence_length,
        injection_top_n=injection_top_n,
    )

    all_rows = layer3_rows + layer2_rows + full_sweep_rows

    cohorts = {
        "layer3": _cohort_summary(layer3_rows),
        "layer2": _cohort_summary(layer2_rows),
        "full_sweep": _cohort_summary(full_sweep_rows),
    }

    total_improvements = sum(row["vag_valid"] and not row["greedy_valid"] for row in all_rows)
    total_regressions = sum(row["greedy_valid"] and not row["vag_valid"] for row in all_rows)
    overall = {
        "n_cases": len(all_rows),
        "improvements_over_greedy": int(total_improvements),
        "regressions_vs_greedy": int(total_regressions),
        "net_improvement": int(total_improvements - total_regressions),
    }

    ratio_summary = {
        "greedy_vs_oracle": _ratio_summary(all_rows, "greedy_ratio_vs_oracle"),
        "vag_vs_oracle": _ratio_summary(all_rows, "vag_ratio_vs_oracle"),
    }

    results = {
        "settings": {
            "n_events": n_events,
            "n_actors": n_actors,
            "max_sequence_length": max_sequence_length,
            "grammar": {
                "min_prefix_elements": grammar.min_prefix_elements,
                "max_phase_regressions": grammar.max_phase_regressions,
                "max_turning_points": grammar.max_turning_points,
                "min_length": grammar.min_length,
                "max_length": grammar.max_length,
                "min_timespan_fraction": grammar.min_timespan_fraction,
                "focal_actor_coverage": grammar.focal_actor_coverage,
            },
            "pool_strategy": pool_strategy,
            "n_anchors": n_anchors,
            "injection_top_n": injection_top_n,
            "full_sweep_epsilons": [0.30, 0.50, 0.70, 0.90],
            "full_sweep_seed_range": [0, 49],
        },
        "cohorts": cohorts,
        "overall": overall,
        "ratio_summary": ratio_summary,
        "per_case": {
            "layer3": layer3_rows,
            "layer2": layer2_rows,
            "full_sweep": full_sweep_rows,
        },
    }

    metadata = ExperimentMetadata(
        name="vag_evaluation",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(layer3_cases) + len(layer2_cases) + len(full_sweep_cases),
        n_extractions=(len(layer3_cases) + len(layer2_cases) + len(full_sweep_cases)) * 3,
        seed_range=(0, 199),
        parameters={
            "layer3_source": str(POOL_DIAG_PATH.name),
            "layer2_source": str(BEAM_SWEEP_PATH.name),
        },
    )

    save_results("vag_evaluation", results, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": results}


def _print_summary(results: dict) -> None:
    layer3 = results["cohorts"]["layer3"]
    layer2 = results["cohorts"]["layer2"]
    full = results["cohorts"]["full_sweep"]
    overall = results["overall"]
    vag_ratio = results["ratio_summary"]["vag_vs_oracle"]

    print("Viability-Aware Greedy Results")
    print("================================")
    print(f"Layer 3 recovery: {layer3['vag_valid_count']}/{layer3['n_cases']}")
    print(f"Layer 2 recovery: {layer2['vag_valid_count']}/{layer2['n_cases']}")
    print(f"Full-sweep improvements over greedy: {full['improvements_over_greedy']}")
    print(f"Full-sweep regressions vs greedy: {full['regressions_vs_greedy']}")
    print(f"Total improvements over greedy: {overall['improvements_over_greedy']}")
    print(f"Cases where VAG < greedy: {overall['regressions_vs_greedy']}")
    print(f"Net improvement: {overall['net_improvement']}")
    print()
    print("VAG approximation ratio (oracle-normalized):")
    if vag_ratio["count"] == 0:
        print("  No valid VAG/oracle overlaps.")
    else:
        print(f"  Mean: {vag_ratio['mean']:.4f}")
        print(f"  Min: {vag_ratio['min']:.4f}")
        print(f"  Cases below 0.90: {vag_ratio['below_0_90']}")


if __name__ == "__main__":
    payload = run_vag_evaluation()
    _print_summary(payload["results"])
