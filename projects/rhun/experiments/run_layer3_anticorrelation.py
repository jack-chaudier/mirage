"""Layer 3 anti-correlation diagnostic.

Measures whether the weight-optimal subset of greedy candidate-pool events
is temporally compressed below the grammar timespan threshold.

Hypothesis: In the 57 Layer 3 beam-unreachable cases, the top-m heaviest
events (m = oracle sequence length) have span < required span.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.pool_construction import bfs_pool, filtered_injection_pool, injection_pool
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
POOL_DIAG_PATH = OUTPUT_DIR / "pool_bottleneck_diagnosis.json"
BEAM_SWEEP_PATH = OUTPUT_DIR / "beam_search_sweep.json"


def _pool_builder(strategy: str):
    if strategy == "bfs":
        return bfs_pool
    if strategy == "injection":
        return injection_pool
    if strategy == "filtered_injection":
        return filtered_injection_pool
    raise ValueError(f"Unknown pool strategy: {strategy}")


def _timespan(events: list[Event] | tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    timestamps = [float(event.timestamp) for event in events]
    return max(timestamps) - min(timestamps)


def _ratio(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        return float("inf")
    return value / baseline


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    ss_y = sum((y - mean_y) ** 2 for y in ys)
    if ss_x <= 1e-12 or ss_y <= 1e-12:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(ss_x * ss_y)


def _weight_centroid_timestamp(events: list[Event]) -> float:
    if not events:
        return 0.0
    total_weight = sum(float(event.weight) for event in events)
    if total_weight <= 0.0:
        return mean(float(event.timestamp) for event in events)
    return sum(float(event.weight) * float(event.timestamp) for event in events) / total_weight


def _sorted_by_weight(events: list[Event]) -> list[Event]:
    return sorted(
        events,
        key=lambda event: (event.weight, -event.timestamp, event.id),
        reverse=True,
    )


def _reconstruct_anchor_pools(
    graph: CausalGraph,
    focal_actor: str,
    pool_strategy: str,
    n_anchors: int,
    max_depth: int,
    injection_top_n: int,
) -> tuple[dict[str, set[str]], set[str]]:
    build_pool = _pool_builder(pool_strategy)
    anchors = sorted(
        graph.events,
        key=lambda event: (event.weight, -event.timestamp),
        reverse=True,
    )[:n_anchors]

    pools_by_anchor: dict[str, set[str]] = {}
    union_pool_ids: set[str] = set()

    for anchor in anchors:
        if pool_strategy == "filtered_injection":
            pool_ids = build_pool(
                graph=graph,
                anchor_id=anchor.id,
                focal_actor=focal_actor,
                max_depth=max_depth,
                injection_top_n=injection_top_n,
                min_position=0.0,
            )
        elif pool_strategy == "injection":
            pool_ids = build_pool(
                graph=graph,
                anchor_id=anchor.id,
                focal_actor=focal_actor,
                max_depth=max_depth,
                injection_top_n=injection_top_n,
            )
        else:
            pool_ids = build_pool(
                graph=graph,
                anchor_id=anchor.id,
                focal_actor=focal_actor,
                max_depth=max_depth,
            )
        pool_set = set(pool_ids)
        pools_by_anchor[anchor.id] = pool_set
        union_pool_ids.update(pool_set)

    return pools_by_anchor, union_pool_ids


def diagnose_case(
    graph: CausalGraph,
    epsilon: float,
    focal_actor: str,
    grammar: GrammarConfig,
    greedy_result: ExtractedSequence,
    oracle_result: ExtractedSequence,
    selected_pool_events: list[Event],
    union_pool_events: list[Event],
) -> dict:
    by_id = {event.id: event for event in graph.events}
    oracle_events = list(oracle_result.events)

    m = len(oracle_events)
    required_span = float(grammar.min_timespan_fraction * graph.duration)

    selected_pool_sorted = _sorted_by_weight(selected_pool_events)
    top_m = selected_pool_sorted[:m]

    span_top_m = _timespan(top_m)
    span_selected_pool = _timespan(selected_pool_events)
    span_union_pool = _timespan(union_pool_events)
    span_oracle = _timespan(oracle_events)
    span_greedy = _timespan(list(greedy_result.events))

    top_m_weight = float(sum(event.weight for event in top_m))
    oracle_weight = float(sum(event.weight for event in oracle_events))
    greedy_weight = float(sum(event.weight for event in greedy_result.events))

    weight_sacrifice = top_m_weight - oracle_weight
    weight_sacrifice_fraction = (weight_sacrifice / top_m_weight) if top_m_weight > 0 else 0.0

    centroid_selected = _weight_centroid_timestamp(selected_pool_events)
    weights_selected = [float(event.weight) for event in selected_pool_events]
    distances_selected = [abs(float(event.timestamp) - centroid_selected) for event in selected_pool_events]
    corr_selected = _pearson_r(weights_selected, distances_selected)

    centroid_union = _weight_centroid_timestamp(union_pool_events)
    weights_union = [float(event.weight) for event in union_pool_events]
    distances_union = [abs(float(event.timestamp) - centroid_union) for event in union_pool_events]
    corr_union = _pearson_r(weights_union, distances_union)

    top_m_positions = [float(graph.global_position(event)) for event in top_m]
    top_m_position_min = min(top_m_positions) if top_m_positions else 0.0
    top_m_position_max = max(top_m_positions) if top_m_positions else 0.0
    top_m_position_mean = mean(top_m_positions) if top_m_positions else 0.0
    top_m_frontload_fraction = (
        sum(1 for pos in top_m_positions if pos <= epsilon) / len(top_m_positions)
        if top_m_positions
        else 0.0
    )

    oracle_ids = {event.id for event in oracle_events}
    selected_pool_ids = {event.id for event in selected_pool_events}
    union_pool_ids = {event.id for event in union_pool_events}
    oracle_in_selected = len(oracle_ids & selected_pool_ids)
    oracle_in_union = len(oracle_ids & union_pool_ids)

    return {
        "m": int(m),
        "required_span": required_span,
        "span_top_m": span_top_m,
        "span_selected_pool": span_selected_pool,
        "span_union_pool": span_union_pool,
        "span_oracle": span_oracle,
        "span_greedy": span_greedy,
        "span_top_m_ratio": _ratio(span_top_m, required_span),
        "span_selected_pool_ratio": _ratio(span_selected_pool, required_span),
        "span_union_pool_ratio": _ratio(span_union_pool, required_span),
        "span_oracle_ratio": _ratio(span_oracle, required_span),
        "span_greedy_ratio": _ratio(span_greedy, required_span),
        "top_m_below_threshold": bool(span_top_m < required_span),
        "top_m_weight": top_m_weight,
        "oracle_weight": oracle_weight,
        "greedy_weight": greedy_weight,
        "weight_sacrifice": weight_sacrifice,
        "weight_sacrifice_fraction": weight_sacrifice_fraction,
        "weight_timestamp_correlation": corr_selected,
        "weight_timestamp_correlation_union_pool": corr_union,
        "top_m_event_ids": [event.id for event in top_m],
        "top_m_timestamps": [float(event.timestamp) for event in top_m],
        "top_m_positions": top_m_positions,
        "top_m_position_min": top_m_position_min,
        "top_m_position_max": top_m_position_max,
        "top_m_position_mean": top_m_position_mean,
        "top_m_frontload_fraction": float(top_m_frontload_fraction),
        "oracle_span_meets_requirement": bool(span_oracle >= required_span),
        "oracle_events_in_selected_pool": int(oracle_in_selected),
        "oracle_events_in_selected_pool_fraction": (
            float(oracle_in_selected / m) if m > 0 else 0.0
        ),
        "oracle_events_in_union_pool": int(oracle_in_union),
        "oracle_events_in_union_pool_fraction": (float(oracle_in_union / m) if m > 0 else 0.0),
        "pool_sizes": {
            "selected_pool_size": len(selected_pool_events),
            "union_pool_size": len(union_pool_events),
        },
        "greedy": {
            "valid": bool(greedy_result.valid),
            "violations": list(greedy_result.violations),
            "n_events": len(greedy_result.events),
            "score": float(greedy_result.score),
            "selected_anchor_id": greedy_result.metadata.get("anchor_id"),
        },
        "oracle": {
            "valid": bool(oracle_result.valid),
            "violations": list(oracle_result.violations),
            "n_events": len(oracle_result.events),
            "score": float(oracle_result.score),
            "forced_turning_point": oracle_result.metadata.get("forced_turning_point"),
        },
        "selected_pool_events": [
            {
                "id": event.id,
                "weight": float(event.weight),
                "timestamp": float(event.timestamp),
                "position": float(graph.global_position(by_id[event.id])),
            }
            for event in selected_pool_events
        ],
    }


def _summary_markdown(results: dict, meta: ExperimentMetadata) -> str:
    agg = results["aggregates"]
    ver = results["verification"]
    lines = [
        f"# {meta.name}",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Cases analyzed: {results['n_cases']}",
        "",
        "## Anti-correlation Hypothesis",
        "",
        f"- Top-m span below threshold: {agg['top_m_below_count']}/{results['n_cases']} ({agg['top_m_below_rate']:.3f})",
        f"- Mean span_top_m / required_span: {agg['span_top_m_ratio_mean']:.3f}",
        f"- Min span_top_m / required_span: {agg['span_top_m_ratio_min']:.3f}",
        "",
        "## Pool Span",
        "",
        f"- All union pools above threshold: {ver['all_union_pools_above_threshold']}",
        f"- All selected pools above threshold: {ver['all_selected_pools_above_threshold']}",
        f"- Mean span_union_pool / required_span: {agg['span_union_pool_ratio_mean']:.3f}",
        f"- Mean span_selected_pool / required_span: {agg['span_selected_pool_ratio_mean']:.3f}",
        "",
        "## Oracle Comparison",
        "",
        f"- All oracle spans above threshold: {ver['all_oracle_spans_above_threshold']}",
        f"- Mean oracle span / required_span: {agg['span_oracle_ratio_mean']:.3f}",
        f"- Mean weight sacrifice (oracle vs top-m): {agg['weight_sacrifice_fraction_mean']:.3f}",
        f"- Max weight sacrifice: {agg['weight_sacrifice_fraction_max']:.3f}",
        "",
        "## Weight-Timestamp Correlation",
        "",
        f"- Mean Pearson r (selected pool): {agg['weight_timestamp_correlation_mean']:.3f}",
        f"- Mean Pearson r (union pool): {agg['weight_timestamp_correlation_union_mean']:.3f}",
        f"- All selected-pool r negative: {agg['all_selected_pool_correlations_negative']}",
        f"- All union-pool r negative: {agg['all_union_pool_correlations_negative']}",
        "",
    ]
    return "\n".join(lines)


def run_layer3_anticorrelation() -> dict:
    if not POOL_DIAG_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {POOL_DIAG_PATH}")
    if not BEAM_SWEEP_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {BEAM_SWEEP_PATH}")

    pool_payload = json.loads(POOL_DIAG_PATH.read_text(encoding="utf-8"))
    beam_payload = json.loads(BEAM_SWEEP_PATH.read_text(encoding="utf-8"))

    cases = [case for case in pool_payload["results"]["per_case"] if bool(case.get("oracle_valid", False))]
    cases.sort(key=lambda case: (float(case["epsilon"]), int(case["seed"]), str(case["focal_actor"])))

    beam_settings = beam_payload["results"]["settings"]
    n_events = int(beam_settings["n_events"])
    n_actors = int(beam_settings["n_actors"])
    max_sequence_length = int(beam_settings["max_sequence_length"])
    grammar = GrammarConfig(**beam_settings["grammar"])

    pool_settings = pool_payload["results"]["settings"]
    pool_strategy = str(pool_settings["pool_strategy"])
    n_anchors = int(pool_settings["n_anchors"])
    max_depth = int(pool_settings["max_depth"])
    injection_top_n = int(pool_settings["injection_top_n"])

    generator = BurstyGenerator()
    timer = ExperimentTimer()
    per_case_results: list[dict] = []

    for case in cases:
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case["focal_actor"])

        graph = generator.generate(
            BurstyConfig(
                seed=seed,
                epsilon=epsilon,
                n_events=n_events,
                n_actors=n_actors,
            )
        )

        greedy = greedy_extract(
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
        if oracle is None:
            raise RuntimeError(
                f"Oracle failed unexpectedly for case epsilon={epsilon}, seed={seed}, focal_actor={focal_actor}"
            )

        anchor_pools, union_pool_ids = _reconstruct_anchor_pools(
            graph=graph,
            focal_actor=focal_actor,
            pool_strategy=pool_strategy,
            n_anchors=n_anchors,
            max_depth=max_depth,
            injection_top_n=injection_top_n,
        )
        by_id = {event.id: event for event in graph.events}
        union_pool_events = [by_id[event_id] for event_id in sorted(union_pool_ids) if event_id in by_id]

        raw_selected_pool = greedy.metadata.get("pool_ids")
        if isinstance(raw_selected_pool, (list, tuple, set)):
            selected_pool_ids = {str(event_id) for event_id in raw_selected_pool}
        else:
            selected_anchor_id = greedy.metadata.get("anchor_id")
            selected_pool_ids = set(anchor_pools.get(str(selected_anchor_id), set()))
            if not selected_pool_ids:
                selected_pool_ids = set(union_pool_ids)

        selected_pool_events = [by_id[event_id] for event_id in sorted(selected_pool_ids) if event_id in by_id]

        diag = diagnose_case(
            graph=graph,
            epsilon=epsilon,
            focal_actor=focal_actor,
            grammar=grammar,
            greedy_result=greedy,
            oracle_result=oracle,
            selected_pool_events=selected_pool_events,
            union_pool_events=union_pool_events,
        )

        required_fraction = float(case["timespan"]["required_span_fraction"])
        union_fraction_from_pool_diag = float(case["timespan"]["union_pool_span_fraction"])
        union_ratio_from_pool_diag = (
            union_fraction_from_pool_diag / required_fraction if required_fraction > 0 else float("inf")
        )

        diag.update(
            {
                "epsilon": epsilon,
                "seed": seed,
                "focal_actor": focal_actor,
                "oracle_diagnostics": oracle_diag,
                "greedy_has_timespan_violation": any(
                    violation.startswith("insufficient_timespan") for violation in greedy.violations
                ),
                "union_pool_ratio_from_pool_diagnosis": union_ratio_from_pool_diag,
                "union_pool_ratio_reconstruction_delta": abs(
                    diag["span_union_pool_ratio"] - union_ratio_from_pool_diag
                ),
            }
        )
        per_case_results.append(diag)

    n_cases = len(per_case_results)
    if n_cases == 0:
        raise RuntimeError("No oracle-valid cases available for anti-correlation diagnosis.")

    top_m_ratios = [row["span_top_m_ratio"] for row in per_case_results]
    selected_pool_ratios = [row["span_selected_pool_ratio"] for row in per_case_results]
    union_pool_ratios = [row["span_union_pool_ratio"] for row in per_case_results]
    oracle_ratios = [row["span_oracle_ratio"] for row in per_case_results]
    greedy_ratios = [row["span_greedy_ratio"] for row in per_case_results]
    sacrifices = [row["weight_sacrifice_fraction"] for row in per_case_results]
    top_m_frontload = [row["top_m_frontload_fraction"] for row in per_case_results]

    corr_selected = [
        row["weight_timestamp_correlation"]
        for row in per_case_results
        if row["weight_timestamp_correlation"] is not None
    ]
    corr_union = [
        row["weight_timestamp_correlation_union_pool"]
        for row in per_case_results
        if row["weight_timestamp_correlation_union_pool"] is not None
    ]

    top_m_below_count = sum(1 for row in per_case_results if row["top_m_below_threshold"])
    greedy_timespan_failures = sum(
        1 for row in per_case_results if bool(row["greedy_has_timespan_violation"])
    )

    verification = {
        "all_union_pools_above_threshold": bool(all(ratio >= 1.0 for ratio in union_pool_ratios)),
        "all_selected_pools_above_threshold": bool(all(ratio >= 1.0 for ratio in selected_pool_ratios)),
        "all_oracle_spans_above_threshold": bool(all(ratio >= 1.0 for ratio in oracle_ratios)),
    }

    # Verification checks requested by the diagnostic specification.
    assert verification["all_union_pools_above_threshold"], (
        "Check 1 failed: expected all union pool spans >= required span."
    )
    assert verification["all_oracle_spans_above_threshold"], (
        "Check 3 failed: expected all oracle spans >= required span."
    )

    aggregates = {
        "top_m_below_count": int(top_m_below_count),
        "top_m_below_rate": float(top_m_below_count / n_cases),
        "span_top_m_ratio_mean": float(mean(top_m_ratios)),
        "span_top_m_ratio_median": float(median(top_m_ratios)),
        "span_top_m_ratio_min": float(min(top_m_ratios)),
        "span_top_m_ratio_max": float(max(top_m_ratios)),
        "span_selected_pool_ratio_mean": float(mean(selected_pool_ratios)),
        "span_selected_pool_ratio_min": float(min(selected_pool_ratios)),
        "span_union_pool_ratio_mean": float(mean(union_pool_ratios)),
        "span_union_pool_ratio_min": float(min(union_pool_ratios)),
        "span_oracle_ratio_mean": float(mean(oracle_ratios)),
        "span_oracle_ratio_min": float(min(oracle_ratios)),
        "span_greedy_ratio_mean": float(mean(greedy_ratios)),
        "greedy_timespan_failure_count": int(greedy_timespan_failures),
        "weight_sacrifice_fraction_mean": float(mean(sacrifices)),
        "weight_sacrifice_fraction_median": float(median(sacrifices)),
        "weight_sacrifice_fraction_max": float(max(sacrifices)),
        "weight_timestamp_correlation_mean": float(mean(corr_selected)) if corr_selected else None,
        "weight_timestamp_correlation_union_mean": float(mean(corr_union)) if corr_union else None,
        "all_selected_pool_correlations_negative": bool(
            corr_selected and all(r < 0.0 for r in corr_selected)
        ),
        "all_union_pool_correlations_negative": bool(corr_union and all(r < 0.0 for r in corr_union)),
        "top_m_frontload_fraction_mean": float(mean(top_m_frontload)),
    }

    results = {
        "n_cases": n_cases,
        "sources": {
            "pool_bottleneck_diagnosis": str(POOL_DIAG_PATH),
            "beam_search_sweep": str(BEAM_SWEEP_PATH),
        },
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
            "max_depth": max_depth,
            "injection_top_n": injection_top_n,
            "m_definition": "oracle_sequence_length",
        },
        "verification": verification,
        "aggregates": aggregates,
        "per_case": per_case_results,
    }

    meta = ExperimentMetadata(
        name="layer3_anticorrelation",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=2 * n_cases,
        seed_range=(
            min(int(row["seed"]) for row in per_case_results),
            max(int(row["seed"]) for row in per_case_results),
        ),
        parameters={
            "source_case_set": "pool_bottleneck_diagnosis oracle-valid cases",
            "top_m_definition": "top-m heaviest events from greedy-selected pool",
        },
    )

    save_results(
        "layer3_anticorrelation",
        results,
        meta,
        summary_formatter=_summary_markdown,
    )
    return {"metadata": meta, "results": results}


def _print_summary(results: dict) -> None:
    agg = results["aggregates"]
    ver = results["verification"]
    n_cases = results["n_cases"]

    print("Layer 3 Anti-Correlation Diagnostic")
    print("=====================================")
    print(f"Cases analyzed: {n_cases}")
    print()
    print("Anti-correlation hypothesis:")
    print(
        "  Top-m span below threshold: "
        f"{agg['top_m_below_count']}/{n_cases} ({100.0 * agg['top_m_below_rate']:.1f}%)"
    )
    print(
        "  Mean span_top_m / required_span: "
        f"{agg['span_top_m_ratio_mean']:.3f}"
    )
    print(
        "  Min span_top_m / required_span: "
        f"{agg['span_top_m_ratio_min']:.3f}"
    )
    print()
    print("Pool span (confirmation):")
    print(
        "  All union pools above threshold: "
        f"{'yes' if ver['all_union_pools_above_threshold'] else 'no'}"
    )
    print(
        "  All selected pools above threshold: "
        f"{'yes' if ver['all_selected_pools_above_threshold'] else 'no'}"
    )
    print(
        "  Mean span_union_pool / required_span: "
        f"{agg['span_union_pool_ratio_mean']:.3f}"
    )
    print()
    print("Oracle comparison:")
    print(
        "  All oracle spans above threshold: "
        f"{'yes' if ver['all_oracle_spans_above_threshold'] else 'no'}"
    )
    print(
        "  Mean oracle span / required_span: "
        f"{agg['span_oracle_ratio_mean']:.3f}"
    )
    print(
        "  Mean weight sacrifice (oracle vs top-m): "
        f"{100.0 * agg['weight_sacrifice_fraction_mean']:.2f}%"
    )
    print(
        "  Max weight sacrifice: "
        f"{100.0 * agg['weight_sacrifice_fraction_max']:.2f}%"
    )
    print()
    print("Weight-timestamp correlation:")
    print(
        "  Mean Pearson r (selected pool): "
        f"{agg['weight_timestamp_correlation_mean']:.3f}"
    )
    print(
        "  Mean Pearson r (union pool): "
        f"{agg['weight_timestamp_correlation_union_mean']:.3f}"
    )
    print(
        "  All selected-pool cases negative: "
        f"{'yes' if agg['all_selected_pool_correlations_negative'] else 'no'}"
    )
    print(
        "  All union-pool cases negative: "
        f"{'yes' if agg['all_union_pool_correlations_negative'] else 'no'}"
    )
    print()

    if agg["top_m_below_rate"] >= 0.90:
        interpretation = (
            "The weight-optimal core is structurally denser than the timespan constraint allows. "
            "Layer 3 failure is objective-constraint anti-correlation, not search depth or pool quality."
        )
    elif agg["top_m_below_rate"] < 0.50:
        interpretation = (
            "The anti-correlation hypothesis is insufficient. "
            "Other mechanisms materially contribute to assembly compression."
        )
    else:
        interpretation = (
            "Anti-correlation explains a substantial subset, but not all cases. "
            "Additional mechanisms likely contribute."
        )

    print("Interpretation:")
    print(f"  {interpretation}")


if __name__ == "__main__":
    payload = run_layer3_anticorrelation()
    _print_summary(payload["results"])
