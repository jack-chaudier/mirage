"""Multi-burst distribution diagnostic sweep.

Evaluates how two temporal weight peaks change failure modes and quality
relative to single-burst findings.
"""

from __future__ import annotations

import math
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.theory.theorem import check_precondition


SEEDS = range(0, 50)
FOCAL_ACTOR = "actor_0"
N_EVENTS = 200
N_ACTORS = 6
K = 1
BURSTY_REFERENCE_R = -0.491

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

BASE_CONFIG = {
    "n_events": N_EVENTS,
    "n_actors": N_ACTORS,
    "burst_centers": (0.2, 0.8),
    "burst_width": 0.1,
    "burst_weight_boost": 3.0,
    "inter_burst_density": 0.3,
}


def _region_of(timestamp: float, config: MultiBurstConfig) -> str:
    c1, c2 = sorted(config.burst_centers)
    width = config.burst_width
    b1 = (max(0.0, c1 - width), min(1.0, c1 + width))
    b2 = (max(0.0, c2 - width), min(1.0, c2 + width))
    if b1[0] <= timestamp <= b1[1]:
        return "burst1"
    if b2[0] <= timestamp <= b2[1]:
        return "burst2"
    if b1[1] < b2[0] and (b1[1] <= timestamp <= b2[0]):
        return "inter"
    return "outside"


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _ratio(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return float(n / d)


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    g = data["aggregates"]["generator_properties"]
    e = data["aggregates"]["extraction"]
    q = data["aggregates"]["quality"]

    lines = [
        "# multiburst_diagnostic",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Generator properties",
        "",
        (
            "- Mean focal events per region: "
            f"burst1={g['mean_focal_events_by_region']['burst1']:.2f}, "
            f"inter={g['mean_focal_events_by_region']['inter']:.2f}, "
            f"burst2={g['mean_focal_events_by_region']['burst2']:.2f}"
        ),
        (
            "- Max-weight event location: "
            f"burst1={100.0 * g['max_weight_location_fraction']['burst1']:.1f}%, "
            f"burst2={100.0 * g['max_weight_location_fraction']['burst2']:.1f}%, "
            f"inter={100.0 * g['max_weight_location_fraction']['inter']:.1f}%"
        ),
        (
            "- Max-weight focal event location: "
            f"burst1={100.0 * g['max_weight_focal_location_fraction']['burst1']:.1f}%, "
            f"burst2={100.0 * g['max_weight_focal_location_fraction']['burst2']:.1f}%, "
            f"inter={100.0 * g['max_weight_focal_location_fraction']['inter']:.1f}%"
        ),
        f"- Mean Pearson r (focal weight vs timestamp): {g['mean_focal_weight_timestamp_r']:.3f}",
        "",
        "## Extraction results (k=1)",
        "",
        f"- Greedy validity: {100.0 * e['validity_rate']['greedy']:.1f}%",
        f"- Oracle validity: {100.0 * e['validity_rate']['oracle']:.1f}%",
        f"- Exact oracle validity: {100.0 * e['validity_rate']['exact_oracle']:.1f}%",
        f"- VAG (span-only) validity: {100.0 * e['validity_rate']['vag']:.1f}%",
        "",
        "## Failure modes (greedy)",
        "",
        f"- Theorem-predicted (j < k): {e['theorem_predicted_cases']} cases",
        f"- Theorem false positives: {e['theorem_false_positives']} cases",
        f"- Timespan violations: {e['timespan_violation_cases']} cases",
        f"- Dev-count violations: {e['development_violation_cases']} cases",
        f"- Gap violations: {e['gap_violation_cases']} cases",
        "",
        "## Quality when greedy succeeds",
        "",
        f"- Mean greedy/exact ratio: {q['greedy_exact_ratio_mean']:.4f}" if q["greedy_exact_ratio_mean"] is not None else "- Mean greedy/exact ratio: n/a",
        f"- Min greedy/exact ratio: {q['greedy_exact_ratio_min']:.4f}" if q["greedy_exact_ratio_min"] is not None else "- Min greedy/exact ratio: n/a",
        f"- Cases below 0.90: {q['greedy_exact_ratio_below_0_90']}",
        "",
        "## Anti-correlation comparison",
        "",
        f"- Bursty generator r (reference): {BURSTY_REFERENCE_R:.3f}",
        f"- Multi-burst generator r: {g['mean_focal_weight_timestamp_r']:.3f}",
        "",
    ]
    return "\n".join(lines)


def run_multiburst_diagnostic() -> dict:
    timer = ExperimentTimer()
    generator = MultiBurstGenerator()
    grammar = GrammarConfig.parametric(K)

    per_case: list[dict] = []

    max_weight_location_counts = {"burst1": 0, "inter": 0, "burst2": 0, "outside": 0}
    max_weight_focal_location_counts = {"burst1": 0, "inter": 0, "burst2": 0, "outside": 0}
    focal_weight_timestamp_rs: list[float] = []

    mean_focal_region_counts: dict[str, list[float]] = {
        "burst1": [],
        "inter": [],
        "burst2": [],
        "outside": [],
    }

    greedy_valid_count = 0
    oracle_valid_count = 0
    exact_valid_count = 0
    vag_valid_count = 0

    theorem_predicted_cases = 0
    theorem_false_positives = 0
    timespan_violation_cases = 0
    development_violation_cases = 0
    gap_violation_cases = 0

    greedy_exact_ratios: list[float] = []

    for seed in SEEDS:
        config = MultiBurstConfig(seed=seed, **BASE_CONFIG)
        graph = generator.generate(config)

        region_events: dict[str, list] = {"burst1": [], "inter": [], "burst2": [], "outside": []}
        for event in graph.events:
            region_events[_region_of(float(event.timestamp), config)].append(event)

        focal_events = list(graph.events_for_actor(FOCAL_ACTOR))
        focal_region_events: dict[str, list] = {"burst1": [], "inter": [], "burst2": [], "outside": []}
        for event in focal_events:
            focal_region_events[_region_of(float(event.timestamp), config)].append(event)

        for region in mean_focal_region_counts:
            mean_focal_region_counts[region].append(float(len(focal_region_events[region])))

        max_weight_event = max(graph.events, key=lambda event: (event.weight, -event.timestamp, event.id))
        max_weight_region = _region_of(float(max_weight_event.timestamp), config)
        max_weight_location_counts[max_weight_region] += 1

        max_weight_focal_region = None
        if focal_events:
            max_weight_focal = max(
                focal_events,
                key=lambda event: (event.weight, -event.timestamp, event.id),
            )
            max_weight_focal_region = _region_of(float(max_weight_focal.timestamp), config)
            max_weight_focal_location_counts[max_weight_focal_region] += 1

        r = None
        if focal_events:
            r = _pearson_r(
                [float(event.weight) for event in focal_events],
                [float(event.timestamp) for event in focal_events],
            )
            if r is not None:
                focal_weight_timestamp_rs.append(r)

        theorem = check_precondition(graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar)
        theorem_predicted = bool(theorem["predicted_failure"])
        if theorem_predicted:
            theorem_predicted_cases += 1

        greedy = greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
        )
        oracle, oracle_diag = oracle_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        exact, exact_diag = exact_oracle_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
        )
        vag, vag_diag = viability_aware_greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
        )

        greedy_valid = bool(greedy.valid)
        oracle_valid = bool(oracle is not None and oracle.valid)
        exact_valid = bool(exact.valid)
        vag_valid = bool(vag.valid)

        if greedy_valid:
            greedy_valid_count += 1
        if oracle_valid:
            oracle_valid_count += 1
        if exact_valid:
            exact_valid_count += 1
        if vag_valid:
            vag_valid_count += 1

        if theorem_predicted and greedy_valid:
            theorem_false_positives += 1

        greedy_timespan_violation = any(v.startswith("insufficient_timespan") for v in greedy.violations)
        greedy_dev_violation = any(v.startswith("insufficient_development") for v in greedy.violations)
        greedy_gap_violation = any(v.startswith("max_temporal_gap:") for v in greedy.violations)

        if greedy_timespan_violation:
            timespan_violation_cases += 1
        if greedy_dev_violation:
            development_violation_cases += 1
        if greedy_gap_violation:
            gap_violation_cases += 1

        greedy_exact_ratio = None
        if greedy_valid and exact_valid and exact.score > 0:
            greedy_exact_ratio = float(greedy.score / exact.score)
            greedy_exact_ratios.append(greedy_exact_ratio)

        case_payload = {
            "seed": seed,
            "focal_actor": FOCAL_ACTOR,
            "generator_config": {
                "n_events": config.n_events,
                "n_actors": config.n_actors,
                "burst_centers": [float(v) for v in config.burst_centers],
                "burst_width": float(config.burst_width),
                "burst_weight_boost": float(config.burst_weight_boost),
                "inter_burst_density": float(config.inter_burst_density),
            },
            "region_counts": {region: len(events) for region, events in region_events.items()},
            "region_mean_weight": {
                region: _safe_mean([float(event.weight) for event in events])
                for region, events in region_events.items()
            },
            "focal_region_counts": {region: len(events) for region, events in focal_region_events.items()},
            "focal_region_mean_weight": {
                region: _safe_mean([float(event.weight) for event in events])
                for region, events in focal_region_events.items()
            },
            "max_weight_event": {
                "id": max_weight_event.id,
                "weight": float(max_weight_event.weight),
                "timestamp": float(max_weight_event.timestamp),
                "region": max_weight_region,
            },
            "max_weight_focal_region": max_weight_focal_region,
            "focal_weight_timestamp_pearson_r": r,
            "theorem": theorem,
            "greedy": {
                "valid": greedy_valid,
                "score": float(greedy.score),
                "violations": list(greedy.violations),
            },
            "oracle": {
                "valid": oracle_valid,
                "score": float(oracle.score) if oracle is not None else None,
                "violations": list(oracle.violations) if oracle is not None else ["oracle_no_candidate"],
                "diagnostics": oracle_diag,
            },
            "exact_oracle": {
                "valid": exact_valid,
                "score": float(exact.score),
                "violations": list(exact.violations),
                "diagnostics": exact_diag,
            },
            "vag": {
                "valid": vag_valid,
                "score": float(vag.score),
                "violations": list(vag.violations),
                "diagnostics": vag_diag,
            },
            "greedy_exact_ratio": greedy_exact_ratio,
            "greedy_violation_flags": {
                "timespan": greedy_timespan_violation,
                "development": greedy_dev_violation,
                "gap": greedy_gap_violation,
            },
        }
        per_case.append(case_payload)

    n_cases = len(per_case)

    aggregates = {
        "generator_properties": {
            "mean_focal_events_by_region": {
                region: float(mean(values)) if values else 0.0
                for region, values in mean_focal_region_counts.items()
            },
            "max_weight_location_counts": max_weight_location_counts,
            "max_weight_location_fraction": {
                region: _ratio(count, n_cases)
                for region, count in max_weight_location_counts.items()
            },
            "max_weight_focal_location_counts": max_weight_focal_location_counts,
            "max_weight_focal_location_fraction": {
                region: _ratio(count, n_cases)
                for region, count in max_weight_focal_location_counts.items()
            },
            "mean_focal_weight_timestamp_r": float(mean(focal_weight_timestamp_rs))
            if focal_weight_timestamp_rs
            else 0.0,
            "median_focal_weight_timestamp_r": float(median(focal_weight_timestamp_rs))
            if focal_weight_timestamp_rs
            else 0.0,
            "focal_weight_timestamp_r_count": len(focal_weight_timestamp_rs),
        },
        "extraction": {
            "validity_count": {
                "greedy": greedy_valid_count,
                "oracle": oracle_valid_count,
                "exact_oracle": exact_valid_count,
                "vag": vag_valid_count,
            },
            "validity_rate": {
                "greedy": _ratio(greedy_valid_count, n_cases),
                "oracle": _ratio(oracle_valid_count, n_cases),
                "exact_oracle": _ratio(exact_valid_count, n_cases),
                "vag": _ratio(vag_valid_count, n_cases),
            },
            "theorem_predicted_cases": theorem_predicted_cases,
            "theorem_false_positives": theorem_false_positives,
            "timespan_violation_cases": timespan_violation_cases,
            "development_violation_cases": development_violation_cases,
            "gap_violation_cases": gap_violation_cases,
        },
        "quality": {
            "greedy_exact_ratio_count": len(greedy_exact_ratios),
            "greedy_exact_ratio_mean": _safe_mean(greedy_exact_ratios),
            "greedy_exact_ratio_median": _safe_median(greedy_exact_ratios),
            "greedy_exact_ratio_min": float(min(greedy_exact_ratios)) if greedy_exact_ratios else None,
            "greedy_exact_ratio_below_0_90": int(
                sum(1 for value in greedy_exact_ratios if value < 0.90)
            ),
        },
    }

    results = {
        "settings": {
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "focal_actor": FOCAL_ACTOR,
            "grammar": {
                "min_prefix_elements": grammar.min_prefix_elements,
                "max_phase_regressions": grammar.max_phase_regressions,
                "max_turning_points": grammar.max_turning_points,
                "min_length": grammar.min_length,
                "max_length": grammar.max_length,
                "min_timespan_fraction": grammar.min_timespan_fraction,
                "max_temporal_gap": grammar.max_temporal_gap,
                "focal_actor_coverage": grammar.focal_actor_coverage,
            },
            "generator_base_config": BASE_CONFIG,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
        },
        "aggregates": aggregates,
        "per_case": per_case,
        "references": {
            "bursty_weight_timestamp_r": BURSTY_REFERENCE_R,
        },
    }

    metadata = ExperimentMetadata(
        name="multiburst_diagnostic",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=n_cases * 4,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "generator": "MultiBurstGenerator",
            "focal_actor": FOCAL_ACTOR,
            "k": K,
        },
    )

    save_results(
        "multiburst_diagnostic",
        results,
        metadata,
        summary_formatter=_summary_markdown,
    )
    return {"metadata": metadata, "results": results}


if __name__ == "__main__":
    payload = run_multiburst_diagnostic()
    agg = payload["results"]["aggregates"]
    ext = agg["extraction"]
    gen = agg["generator_properties"]
    qual = agg["quality"]

    print("Multi-Burst Diagnostic (50 seeds)")
    print("=" * 35)
    print(
        "Generator properties:\n"
        f"  Mean focal events per region: burst1={gen['mean_focal_events_by_region']['burst1']:.2f}, "
        f"inter={gen['mean_focal_events_by_region']['inter']:.2f}, "
        f"burst2={gen['mean_focal_events_by_region']['burst2']:.2f}\n"
        f"  Mean Pearson r (weight vs timestamp): {gen['mean_focal_weight_timestamp_r']:.3f}"
    )
    print(
        "Extraction results (k=1):\n"
        f"  Greedy validity: {100.0 * ext['validity_rate']['greedy']:.1f}%\n"
        f"  Oracle validity: {100.0 * ext['validity_rate']['oracle']:.1f}%\n"
        f"  Exact oracle validity: {100.0 * ext['validity_rate']['exact_oracle']:.1f}%\n"
        f"  VAG validity: {100.0 * ext['validity_rate']['vag']:.1f}%"
    )
    if qual["greedy_exact_ratio_mean"] is not None:
        print(
            "Quality (greedy succeeds):\n"
            f"  Mean greedy/exact ratio: {qual['greedy_exact_ratio_mean']:.4f}\n"
            f"  Min greedy/exact ratio: {qual['greedy_exact_ratio_min']:.4f}\n"
            f"  Cases below 0.90: {qual['greedy_exact_ratio_below_0_90']}"
        )
