"""Experiment 37: Exogenous TP Validation.

Script A: k-j sweep comparing endogenous (max-weight) vs exogenous (median-timestamp) TP.
Script B: 57 Layer 3 cases re-run with oracle TP override.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Phase


OUTPUT_NAME = "exogenous_tp_validation"
FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
SEEDS = range(100)
K_VALUES = [0, 1, 2, 3, 4, 5]
N_EVENTS = 200
N_ACTORS = 6

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

# Layer 3 parameters (same as original failure taxonomy)
LAYER3_EPSILON = 0.80
LAYER3_K = 1
LAYER3_SEEDS = range(150)

# Grammar for layer 3 (with gap constraint)
LAYER3_GRAMMAR = GrammarConfig(
    min_prefix_elements=1,
    min_timespan_fraction=0.3,
    max_temporal_gap=float("inf"),
)


def _make_grammar(k: int) -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=k,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )


def _safe_rate(numer: int | float, denom: int | float) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str):
    focal_events = [e for e in graph.events if focal_actor in e.actors]
    if not focal_events:
        return None
    return max(focal_events, key=lambda e: (float(e.weight), -float(e.timestamp)))


def _median_timestamp_focal_event(graph: CausalGraph, focal_actor: str):
    """Select the median-timestamp focal event as exogenous TP."""
    focal_events = sorted(
        [e for e in graph.events if focal_actor in e.actors],
        key=lambda e: float(e.timestamp),
    )
    if not focal_events:
        return None
    mid = len(focal_events) // 2
    return focal_events[mid]


def _j_dev_pool(n_pre_tp_all_events: int, k: int) -> int:
    n_setup = ceil(0.2 * n_pre_tp_all_events) if n_pre_tp_all_events > 0 else 0
    if k > 0:
        max_setup = max(0, n_pre_tp_all_events - k)
        n_setup = min(n_setup, max_setup)
    return n_pre_tp_all_events - n_setup


def _compute_j_dev_for_tp(graph: CausalGraph, tp_event, k: int) -> int:
    """Compute j_dev_pool for a given TP event."""
    tp_timestamp = float(tp_event.timestamp)
    n_pre = sum(1 for e in graph.events if float(e.timestamp) < tp_timestamp)
    return _j_dev_pool(n_pre, k)


def _evaluate_graph_script_a(task: tuple[float, int]) -> dict:
    """Script A: evaluate one graph under endogenous and exogenous TP."""
    epsilon, seed = task

    graph = BurstyGenerator().generate(
        BurstyConfig(seed=int(seed), epsilon=float(epsilon), n_events=N_EVENTS, n_actors=N_ACTORS)
    )

    # Identify TP events
    endo_tp = _max_weight_focal_event(graph, FOCAL_ACTOR)
    exo_tp = _median_timestamp_focal_event(graph, FOCAL_ACTOR)
    if endo_tp is None or exo_tp is None:
        return {"epsilon": float(epsilon), "seed": int(seed), "valid": False, "records": []}

    records: list[dict] = []
    for k in K_VALUES:
        grammar = _make_grammar(k)

        # Endogenous (default behavior)
        endo_result = greedy_extract(
            graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar,
            pool_strategy=POOL_STRATEGY, n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH, injection_top_n=INJECTION_TOP_N,
        )
        endo_j_dev = _compute_j_dev_for_tp(graph, endo_tp, k)

        # Exogenous (median-timestamp focal event as TP)
        exo_result = greedy_extract(
            graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar,
            pool_strategy=POOL_STRATEGY, n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH, injection_top_n=INJECTION_TOP_N,
            override_tp_id=exo_tp.id,
        )
        exo_j_dev = _compute_j_dev_for_tp(graph, exo_tp, k)

        records.append({
            "epsilon": float(epsilon),
            "seed": int(seed),
            "k": int(k),
            "endo_valid": bool(endo_result.valid),
            "endo_j_dev": int(endo_j_dev),
            "endo_below_diagonal": bool(endo_j_dev < k),
            "exo_valid": bool(exo_result.valid),
            "exo_j_dev": int(exo_j_dev),
            "exo_below_diagonal": bool(exo_j_dev < k),
            "endo_tp_id": endo_tp.id,
            "exo_tp_id": exo_tp.id,
            "endo_tp_timestamp": float(endo_tp.timestamp),
            "exo_tp_timestamp": float(exo_tp.timestamp),
        })

    return {"epsilon": float(epsilon), "seed": int(seed), "valid": True, "records": records}


def _run_script_a() -> dict:
    """Script A: k-j sweep under endogenous vs exogenous TP."""
    print("Script A: k-j sweep under endogenous vs exogenous TP...", flush=True)
    tasks = [(epsilon, seed) for epsilon in EPSILONS for seed in SEEDS]

    all_records: list[dict] = []
    cpu_count = os.cpu_count() or 1
    max_workers = min(12, max(1, cpu_count - 1))

    print(f"Running {len(tasks)} graph evaluations with {max_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, payload in enumerate(executor.map(_evaluate_graph_script_a, tasks, chunksize=4), 1):
            if payload["valid"]:
                all_records.extend(payload["records"])
            if idx % 100 == 0 or idx == len(tasks):
                print(f"  Progress: {idx}/{len(tasks)}", flush=True)

    # Aggregate by k
    endo_stats: dict[int, dict] = {}
    exo_stats: dict[int, dict] = {}
    for k in K_VALUES:
        k_records = [r for r in all_records if r["k"] == k]
        n = len(k_records)
        if n == 0:
            continue

        endo_valid = sum(1 for r in k_records if r["endo_valid"])
        exo_valid = sum(1 for r in k_records if r["exo_valid"])
        endo_below = sum(1 for r in k_records if r["endo_below_diagonal"])
        exo_below = sum(1 for r in k_records if r["exo_below_diagonal"])

        # Check diagonal barrier: in below-diagonal cases, is validity exactly 0?
        endo_below_records = [r for r in k_records if r["endo_below_diagonal"]]
        exo_below_records = [r for r in k_records if r["exo_below_diagonal"]]
        endo_below_valid = sum(1 for r in endo_below_records if r["endo_valid"])
        exo_below_valid = sum(1 for r in exo_below_records if r["exo_valid"])

        endo_j_devs = [r["endo_j_dev"] for r in k_records]
        exo_j_devs = [r["exo_j_dev"] for r in k_records]

        endo_stats[k] = {
            "n_instances": n,
            "valid_count": endo_valid,
            "valid_rate": _safe_rate(endo_valid, n),
            "failure_rate": _safe_rate(n - endo_valid, n),
            "p_below_diagonal": _safe_rate(endo_below, n),
            "below_diagonal_count": endo_below,
            "below_diagonal_valid_count": endo_below_valid,
            "diagonal_barrier_holds": bool(endo_below_valid == 0),
            "j_dev_mean": float(mean(endo_j_devs)) if endo_j_devs else None,
            "j_dev_median": float(median(endo_j_devs)) if endo_j_devs else None,
        }
        exo_stats[k] = {
            "n_instances": n,
            "valid_count": exo_valid,
            "valid_rate": _safe_rate(exo_valid, n),
            "failure_rate": _safe_rate(n - exo_valid, n),
            "p_below_diagonal": _safe_rate(exo_below, n),
            "below_diagonal_count": exo_below,
            "below_diagonal_valid_count": exo_below_valid,
            "diagonal_barrier_holds": bool(exo_below_valid == 0),
            "j_dev_mean": float(mean(exo_j_devs)) if exo_j_devs else None,
            "j_dev_median": float(median(exo_j_devs)) if exo_j_devs else None,
        }

    # Overall stats
    all_endo_below = sum(1 for r in all_records if r["endo_below_diagonal"] and r["k"] >= 1)
    all_exo_below = sum(1 for r in all_records if r["exo_below_diagonal"] and r["k"] >= 1)
    all_k_ge_1 = sum(1 for r in all_records if r["k"] >= 1)
    all_endo_fail = sum(1 for r in all_records if not r["endo_valid"] and r["k"] >= 1)
    all_exo_fail = sum(1 for r in all_records if not r["exo_valid"] and r["k"] >= 1)

    # Specific epsilon=0.80 stats for the paper
    eps80_records = [r for r in all_records if abs(r["epsilon"] - 0.80) < 0.001 and r["k"] >= 1]
    eps80_endo_below = sum(1 for r in eps80_records if r["endo_below_diagonal"])
    eps80_exo_below = sum(1 for r in eps80_records if r["exo_below_diagonal"])
    eps80_n = len(eps80_records)

    return {
        "endogenous": endo_stats,
        "exogenous": exo_stats,
        "overall": {
            "n_instances_k_ge_1": all_k_ge_1,
            "endo_p_below_diagonal": _safe_rate(all_endo_below, all_k_ge_1),
            "exo_p_below_diagonal": _safe_rate(all_exo_below, all_k_ge_1),
            "endo_overall_failure_rate": _safe_rate(all_endo_fail, all_k_ge_1),
            "exo_overall_failure_rate": _safe_rate(all_exo_fail, all_k_ge_1),
        },
        "epsilon_080": {
            "n_instances_k_ge_1": eps80_n,
            "endo_p_below_diagonal": _safe_rate(eps80_endo_below, eps80_n),
            "exo_p_below_diagonal": _safe_rate(eps80_exo_below, eps80_n),
        },
        "n_total_records": len(all_records),
    }


def _run_script_b() -> dict:
    """Script B: Re-run Layer 3 failure cases with oracle TP override."""
    print("\nScript B: Layer 3 cases with oracle TP override...", flush=True)

    grammar = GrammarConfig(
        min_prefix_elements=LAYER3_K,
        min_timespan_fraction=0.15,
        max_temporal_gap=float("inf"),
    )

    # Find Layer 3 cases: greedy fails but oracle succeeds
    layer3_cases: list[dict] = []
    for seed in LAYER3_SEEDS:
        graph = BurstyGenerator().generate(
            BurstyConfig(seed=int(seed), epsilon=LAYER3_EPSILON, n_events=N_EVENTS, n_actors=N_ACTORS)
        )

        greedy_result = greedy_extract(
            graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar,
            pool_strategy=POOL_STRATEGY, n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH, injection_top_n=INJECTION_TOP_N,
        )

        if greedy_result.valid:
            continue

        # Check if oracle can solve it
        oracle_result, _ = exact_oracle_extract(
            graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar,
        )
        if not oracle_result.valid:
            continue

        # This is a Layer 3 case
        oracle_tp = oracle_result.turning_point
        if oracle_tp is None:
            continue

        layer3_cases.append({
            "seed": int(seed),
            "oracle_tp_id": oracle_tp.id,
            "greedy_violations": list(greedy_result.violations),
        })

    print(f"Found {len(layer3_cases)} Layer 3 cases (greedy fails, oracle succeeds)", flush=True)

    # Re-run with oracle TP override
    recovered = 0
    still_failing = 0
    failure_reasons: dict[str, int] = defaultdict(int)
    case_results: list[dict] = []

    for i, case in enumerate(layer3_cases, 1):
        graph = BurstyGenerator().generate(
            BurstyConfig(seed=case["seed"], epsilon=LAYER3_EPSILON, n_events=N_EVENTS, n_actors=N_ACTORS)
        )

        override_result = greedy_extract(
            graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar,
            pool_strategy=POOL_STRATEGY, n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH, injection_top_n=INJECTION_TOP_N,
            override_tp_id=case["oracle_tp_id"],
        )

        if override_result.valid:
            recovered += 1
            status = "recovered"
        else:
            still_failing += 1
            status = "still_failing"
            for v in override_result.violations:
                # Normalize violation type
                vtype = v.split(":")[0].strip()
                failure_reasons[vtype] += 1

        case_results.append({
            "seed": case["seed"],
            "oracle_tp_id": case["oracle_tp_id"],
            "override_valid": bool(override_result.valid),
            "override_violations": list(override_result.violations),
            "original_violations": case["greedy_violations"],
            "status": status,
        })

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(layer3_cases)}, recovered={recovered}", flush=True)

    print(f"  Recovered: {recovered}/{len(layer3_cases)}", flush=True)
    print(f"  Still failing: {still_failing}/{len(layer3_cases)}", flush=True)

    return {
        "n_cases": len(layer3_cases),
        "recovered_with_oracle_tp": recovered,
        "still_failing": still_failing,
        "failure_reasons": dict(failure_reasons),
        "cases": case_results,
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    sa = data["script_a"]
    sb = data["script_b"]

    lines = [
        "# Experiment 37: Exogenous TP Validation",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Script A: k-j Sweep (Endogenous vs Exogenous TP)",
        "",
        "| k | Endo valid% | Exo valid% | Endo P(j<k) | Exo P(j<k) | Endo barrier | Exo barrier |",
        "|---|------------:|----------:|------------:|----------:|:-------------|:------------|",
    ]

    for k in K_VALUES:
        endo = sa["endogenous"].get(k, {})
        exo = sa["exogenous"].get(k, {})
        lines.append(
            f"| {k} | {100*endo.get('valid_rate', 0):.1f} | {100*exo.get('valid_rate', 0):.1f} | "
            f"{100*endo.get('p_below_diagonal', 0):.1f}% | {100*exo.get('p_below_diagonal', 0):.1f}% | "
            f"{endo.get('diagonal_barrier_holds', 'n/a')} | {exo.get('diagonal_barrier_holds', 'n/a')} |"
        )

    overall = sa["overall"]
    eps80 = sa["epsilon_080"]
    lines.extend([
        "",
        "### Overall (k >= 1)",
        "",
        f"- Endogenous P(j_dev < k): {100*overall['endo_p_below_diagonal']:.1f}%",
        f"- Exogenous P(j_dev < k): {100*overall['exo_p_below_diagonal']:.1f}%",
        f"- Endogenous failure rate: {100*overall['endo_overall_failure_rate']:.1f}%",
        f"- Exogenous failure rate: {100*overall['exo_overall_failure_rate']:.1f}%",
        "",
        f"### At epsilon=0.80 (k >= 1)",
        "",
        f"- Endogenous P(j_dev < k): {100*eps80['endo_p_below_diagonal']:.1f}%",
        f"- Exogenous P(j_dev < k): {100*eps80['exo_p_below_diagonal']:.1f}%",
        "",
        "## Script B: Layer 3 Cases with Oracle TP Override",
        "",
        f"- Total Layer 3 cases found: {sb['n_cases']}",
        f"- Recovered with oracle TP: {sb['recovered_with_oracle_tp']}/{sb['n_cases']}",
        f"- Still failing: {sb['still_failing']}/{sb['n_cases']}",
    ])

    if sb["failure_reasons"]:
        lines.extend([
            "",
            "### Remaining failure reasons:",
            "",
        ])
        for reason, count in sorted(sb["failure_reasons"].items()):
            lines.append(f"- {reason}: {count}")

    return "\n".join(lines)


def run_exogenous_tp() -> dict:
    timer = ExperimentTimer()

    script_a_results = _run_script_a()
    script_b_results = _run_script_b()

    data = {
        "script_a": script_a_results,
        "script_b": script_b_results,
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=script_a_results["n_total_records"] // len(K_VALUES) + script_b_results["n_cases"],
        n_extractions=script_a_results["n_total_records"] * 2 + script_b_results["n_cases"] * 3,
        seed_range=(0, max(max(SEEDS), max(LAYER3_SEEDS))),
        parameters={
            "script_a": {
                "epsilons": EPSILONS,
                "k_values": K_VALUES,
                "seeds_per_epsilon": len(SEEDS),
                "exogenous_method": "median_timestamp_focal_event",
            },
            "script_b": {
                "epsilon": LAYER3_EPSILON,
                "k": LAYER3_K,
                "seeds": list(LAYER3_SEEDS),
            },
        },
    )
    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_exogenous_tp()
