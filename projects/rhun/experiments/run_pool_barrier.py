"""Experiment 39: Pool Temporal Coverage Barrier verification on TP-solver residuals."""

from __future__ import annotations

import json
from pathlib import Path

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.pool_construction import injection_pool
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator


OUTPUT_NAME = "pool_barrier_verification"
ROOT = Path(__file__).resolve().parents[1]
FOCAL_ACTOR = "actor_0"
N_EVENTS = 200
N_ACTORS = 6

# Grammar with gap constraint (same as tp_solver evaluation)
BURSTY_GRAMMAR = GrammarConfig(
    min_prefix_elements=1,
    min_timespan_fraction=0.3,
    max_temporal_gap=0.14,
)
MULTIBURST_GRAMMAR = GrammarConfig(
    min_prefix_elements=1,
    min_timespan_fraction=0.3,
    max_temporal_gap=0.14,
)

# Residual cases from tp_solver_pool_diagnosis.md
RESIDUAL_CASES: list[dict] = [
    {"topology": "multiburst_gap", "seed": 40, "epsilon": None},
    {"topology": "bursty_gap", "seed": 29, "epsilon": 0.5},
    {"topology": "bursty_gap", "seed": 33, "epsilon": 0.5},
    {"topology": "bursty_gap", "seed": 39, "epsilon": 0.5},
    {"topology": "bursty_gap", "seed": 43, "epsilon": 0.5},
    {"topology": "bursty_gap", "seed": 52, "epsilon": 0.5},
    {"topology": "bursty_gap", "seed": 123, "epsilon": 0.5},
    {"topology": "bursty_gap", "seed": 134, "epsilon": 0.5},
]


def _generate_graph(case: dict):
    """Generate graph for a residual case."""
    if case["topology"] == "multiburst_gap":
        return MultiBurstGenerator().generate(
            MultiBurstConfig(seed=case["seed"], n_events=N_EVENTS, n_actors=N_ACTORS)
        )
    else:
        return BurstyGenerator().generate(
            BurstyConfig(
                seed=case["seed"],
                epsilon=case["epsilon"],
                n_events=N_EVENTS,
                n_actors=N_ACTORS,
            )
        )


def _get_grammar(case: dict) -> GrammarConfig:
    if case["topology"] == "multiburst_gap":
        return MULTIBURST_GRAMMAR
    return BURSTY_GRAMMAR


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines = [
        "# Experiment 39: Pool Temporal Coverage Barrier Verification",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Barrier Condition: tau(P) < s * T",
        "",
        "| # | Topology | Seed | pool_span | required_span | span_deficit | barrier_holds | violation_type |",
        "|---|----------|-----:|----------:|--------------:|-------------:|--------------:|----------------|",
    ]

    for i, case_result in enumerate(data["cases"], 1):
        lines.append(
            f"| {i} | {case_result['topology']} | {case_result['seed']} | "
            f"{case_result['pool_span']:.4f} | {case_result['required_span']:.4f} | "
            f"{case_result['span_deficit']:.4f} | {case_result['barrier_holds']} | "
            f"{case_result['solver_violation_type']} |"
        )

    summary = data["summary"]
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Total residual cases: {summary['total_cases']}",
        f"- Bursty cases: {summary['n_bursty']}",
        f"- Multi-burst cases: {summary['n_multiburst']}",
        f"- Bursty cases with barrier condition: {summary['bursty_barrier_count']}/{summary['n_bursty']}",
        f"- Multi-burst cases with barrier condition: {summary['multiburst_barrier_count']}/{summary['n_multiburst']}",
        f"- Overall barrier-explained: {summary['total_barrier_count']}/{summary['total_cases']}",
    ])
    return "\n".join(lines)


def run_pool_barrier() -> dict:
    timer = ExperimentTimer()

    case_results: list[dict] = []

    for i, case in enumerate(RESIDUAL_CASES, 1):
        print(f"Case {i}/{len(RESIDUAL_CASES)}: {case['topology']} seed={case['seed']}", flush=True)

        graph = _generate_graph(case)
        grammar = _get_grammar(case)

        # Build pool (same as solver uses: BFS + injection from anchor)
        # Use the greedy approach to get the pool
        greedy_result = greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy="injection",
            n_anchors=8,
            max_sequence_length=20,
            injection_top_n=40,
        )

        # Get pool IDs from the best greedy candidate's metadata
        pool_ids_raw = greedy_result.metadata.get("pool_ids", ())
        pool_ids = set(str(eid) for eid in pool_ids_raw)

        # Compute pool temporal coverage
        by_id = {event.id: event for event in graph.events}
        pool_events = [by_id[eid] for eid in pool_ids if eid in by_id]

        if pool_events:
            pool_min_t = min(float(e.timestamp) for e in pool_events)
            pool_max_t = max(float(e.timestamp) for e in pool_events)
            pool_span = pool_max_t - pool_min_t
        else:
            pool_span = 0.0

        required_span = grammar.min_timespan_fraction * graph.duration
        span_deficit = max(0.0, required_span - pool_span)
        barrier_holds = bool(pool_span < required_span - 1e-12)

        # Get oracle result for comparison
        oracle_result, oracle_diag = exact_oracle_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
        )
        if oracle_result.valid:
            oracle_events = oracle_result.events
            oracle_span = float(oracle_events[-1].timestamp - oracle_events[0].timestamp) if oracle_events else 0.0
        else:
            oracle_span = 0.0

        # Check solver violation type from the greedy result
        violations = list(greedy_result.violations)
        violation_type = "none" if greedy_result.valid else (
            "insufficient_timespan" if any("insufficient_timespan" in v for v in violations)
            else violations[0] if violations else "unknown"
        )

        result = {
            "topology": case["topology"],
            "seed": case["seed"],
            "epsilon": case.get("epsilon"),
            "pool_size": len(pool_ids),
            "pool_span": float(pool_span),
            "required_span": float(required_span),
            "span_deficit": float(span_deficit),
            "barrier_holds": barrier_holds,
            "oracle_span": float(oracle_span),
            "oracle_valid": bool(oracle_result.valid),
            "solver_violation_type": violation_type,
            "greedy_violations": violations,
        }
        case_results.append(result)
        print(f"  pool_span={pool_span:.4f}, required={required_span:.4f}, "
              f"barrier={barrier_holds}, violation={violation_type}", flush=True)

    # Summary
    n_bursty = sum(1 for c in case_results if "bursty" in c["topology"] and "multi" not in c["topology"])
    n_multiburst = sum(1 for c in case_results if "multiburst" in c["topology"])
    bursty_barrier = sum(1 for c in case_results
                         if "bursty" in c["topology"] and "multi" not in c["topology"] and c["barrier_holds"])
    multiburst_barrier = sum(1 for c in case_results
                             if "multiburst" in c["topology"] and c["barrier_holds"])

    data = {
        "cases": case_results,
        "summary": {
            "total_cases": len(case_results),
            "n_bursty": n_bursty,
            "n_multiburst": n_multiburst,
            "bursty_barrier_count": bursty_barrier,
            "multiburst_barrier_count": multiburst_barrier,
            "total_barrier_count": bursty_barrier + multiburst_barrier,
        },
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(RESIDUAL_CASES),
        n_extractions=len(RESIDUAL_CASES) * 2,  # greedy + oracle each
        seed_range=(min(c["seed"] for c in RESIDUAL_CASES), max(c["seed"] for c in RESIDUAL_CASES)),
        parameters={
            "residual_cases": len(RESIDUAL_CASES),
            "grammar": {
                "min_prefix_elements": 1,
                "min_timespan_fraction": 0.3,
                "max_temporal_gap": 0.14,
            },
        },
    )
    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_pool_barrier()
