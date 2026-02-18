"""Adversarial max_temporal_gap experiment for Viability-Aware Greedy (VAG).

Goal: test whether span-only viability is blind to local gap constraints.
"""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import Event, ExtractedSequence


EPSILONS = [0.50, 0.70, 0.90]
SEEDS = range(0, 50)
N_EVENTS = 200
N_ACTORS = 6
FOCAL_ACTOR = "actor_0"

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40


def _adjacent_gaps(events: tuple[Event, ...] | list[Event]) -> list[float]:
    if len(events) < 2:
        return []
    return [float(events[i + 1].timestamp - events[i].timestamp) for i in range(len(events) - 1)]


def _max_adjacent_gap(events: tuple[Event, ...] | list[Event]) -> float:
    gaps = _adjacent_gaps(events)
    return max(gaps) if gaps else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _safe_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q50": None,
            "q60": None,
            "q70": None,
            "q75": None,
            "q80": None,
            "q85": None,
            "q90": None,
            "q95": None,
        }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "q50": float(_quantile(values, 0.50)),
        "q60": float(_quantile(values, 0.60)),
        "q70": float(_quantile(values, 0.70)),
        "q75": float(_quantile(values, 0.75)),
        "q80": float(_quantile(values, 0.80)),
        "q85": float(_quantile(values, 0.85)),
        "q90": float(_quantile(values, 0.90)),
        "q95": float(_quantile(values, 0.95)),
    }


def _sequence_payload(seq: ExtractedSequence | None) -> dict:
    if seq is None:
        return {
            "valid": False,
            "score": None,
            "n_events": 0,
            "violations": ["oracle_no_candidate"],
            "max_adjacent_gap": 0.0,
            "mean_adjacent_gap": 0.0,
            "gap_violation_count": 0,
        }
    gaps = _adjacent_gaps(seq.events)
    return {
        "valid": bool(seq.valid),
        "score": float(seq.score),
        "n_events": len(seq.events),
        "violations": list(seq.violations),
        "max_adjacent_gap": float(max(gaps) if gaps else 0.0),
        "mean_adjacent_gap": float(mean(gaps) if gaps else 0.0),
        "gap_violation_count": int(
            sum(1 for violation in seq.violations if str(violation).startswith("max_temporal_gap:"))
        ),
    }


def _calibrate_gap_threshold(
    generator: BurstyGenerator,
    grammar_no_gap: GrammarConfig,
) -> tuple[dict, float]:
    greedy_max_gaps: list[float] = []
    oracle_max_gaps: list[float] = []
    per_case: list[dict] = []

    for seed in SEEDS:
        for epsilon in EPSILONS:
            graph = generator.generate(
                BurstyConfig(seed=seed, epsilon=epsilon, n_events=N_EVENTS, n_actors=N_ACTORS)
            )
            greedy = greedy_extract(
                graph=graph,
                focal_actor=FOCAL_ACTOR,
                grammar=grammar_no_gap,
                pool_strategy=POOL_STRATEGY,
                n_anchors=N_ANCHORS,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                injection_top_n=INJECTION_TOP_N,
            )
            oracle, oracle_diag = oracle_extract(
                graph=graph,
                focal_actor=FOCAL_ACTOR,
                grammar=grammar_no_gap,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )

            row = {
                "seed": seed,
                "epsilon": epsilon,
                "greedy_valid": bool(greedy.valid),
                "oracle_valid": bool(oracle is not None and oracle.valid),
                "greedy_max_gap": None,
                "oracle_max_gap": None,
                "oracle_diagnostics": oracle_diag,
            }

            if greedy.valid:
                gm = _max_adjacent_gap(greedy.events)
                greedy_max_gaps.append(gm)
                row["greedy_max_gap"] = gm
            if oracle is not None and oracle.valid:
                om = _max_adjacent_gap(oracle.events)
                oracle_max_gaps.append(om)
                row["oracle_max_gap"] = om
            per_case.append(row)

    greedy_stats = _safe_stats(greedy_max_gaps)
    oracle_stats = _safe_stats(oracle_max_gaps)

    candidate_percentiles = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    candidate_rows: list[dict] = []
    for p in candidate_percentiles:
        threshold = _quantile(greedy_max_gaps, p)
        greedy_pass = (
            sum(1 for value in greedy_max_gaps if value <= threshold + 1e-12) / len(greedy_max_gaps)
            if greedy_max_gaps
            else 0.0
        )
        oracle_pass = (
            sum(1 for value in oracle_max_gaps if value <= threshold + 1e-12) / len(oracle_max_gaps)
            if oracle_max_gaps
            else 0.0
        )
        candidate_rows.append(
            {
                "percentile": p,
                "threshold": float(threshold),
                "greedy_pass_rate": float(greedy_pass),
                "oracle_pass_rate": float(oracle_pass),
            }
        )

    selected = candidate_rows[0] if candidate_rows else {"percentile": 0.75, "threshold": 0.0}
    for row in candidate_rows:
        if row["greedy_pass_rate"] >= 0.50 and row["greedy_pass_rate"] <= 0.95:
            selected = row
            break

    calibration = {
        "settings": {
            "epsilons": EPSILONS,
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "focal_actor": FOCAL_ACTOR,
        },
        "greedy_max_gap_stats": greedy_stats,
        "oracle_max_gap_stats": oracle_stats,
        "candidate_thresholds": candidate_rows,
        "selected_threshold": {
            "value": float(selected["threshold"]),
            "source_percentile": float(selected["percentile"]),
            "source_distribution": "greedy_max_gap",
            "greedy_pass_rate": float(selected["greedy_pass_rate"]),
            "oracle_pass_rate": float(selected["oracle_pass_rate"]),
        },
        "per_case": per_case,
    }
    return calibration, float(selected["threshold"])


def _evaluate_algorithms(graph, grammar: GrammarConfig) -> dict:
    greedy = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        pool_strategy=POOL_STRATEGY,
        n_anchors=N_ANCHORS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        injection_top_n=INJECTION_TOP_N,
    )
    vag_span, vag_span_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        pool_strategy=POOL_STRATEGY,
        n_anchors=N_ANCHORS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        injection_top_n=INJECTION_TOP_N,
        gap_aware_viability=False,
    )
    vag_gap, vag_gap_diag = viability_aware_greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        pool_strategy=POOL_STRATEGY,
        n_anchors=N_ANCHORS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        injection_top_n=INJECTION_TOP_N,
        gap_aware_viability=True,
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

    return {
        "greedy": _sequence_payload(greedy),
        "vag_span_only": {
            **_sequence_payload(vag_span),
            "diagnostics": vag_span_diag,
        },
        "vag_gap_aware": {
            **_sequence_payload(vag_gap),
            "diagnostics": vag_gap_diag,
        },
        "oracle": {
            **_sequence_payload(oracle),
            "diagnostics": oracle_diag,
        },
        "exact_oracle": {
            **_sequence_payload(exact),
            "diagnostics": exact_diag,
        },
    }


def _aggregate_validity(per_case: list[dict], condition: str, algorithm: str) -> dict:
    rows = [row["results"][condition][algorithm] for row in per_case]
    valid_count = sum(1 for row in rows if row["valid"])
    n = len(rows)
    return {
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / n) if n else 0.0,
        "gap_violation_count": int(sum(row["gap_violation_count"] for row in rows)),
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    table = data["summary_table"]
    lines = [
        "# gap_adversarial",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Calibrated G: {data['calibration']['selected_threshold']['value']:.4f}",
        (
            "Calibration source: "
            f"greedy max-gap percentile p={data['calibration']['selected_threshold']['source_percentile']:.2f}"
        ),
        "",
        "## Validity Rates",
        "",
        "| algorithm | no_gap | with_gap | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['algorithm']} | {row['no_gap_rate']:.3f} | {row['with_gap_rate']:.3f} | {row['delta']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            f"- VAG regressions vs greedy (with gap): {data['key_metrics']['vag_regressions_with_gap']}",
            f"- VAG regressions vs greedy (no gap): {data['key_metrics']['vag_regressions_no_gap']}",
            f"- VAG failures vs oracle (with gap): {data['key_metrics']['vag_failures_vs_oracle_with_gap']}",
            f"- VAG failures vs exact oracle (with gap): {data['key_metrics']['vag_failures_vs_exact_with_gap']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_gap_adversarial() -> dict:
    generator = BurstyGenerator()
    calibration_timer = ExperimentTimer()

    grammar_no_gap = GrammarConfig(
        min_prefix_elements=1,
        max_phase_regressions=0,
        max_turning_points=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.15,
        max_temporal_gap=float("inf"),
        focal_actor_coverage=0.60,
    )

    calibration, calibrated_gap = _calibrate_gap_threshold(
        generator=generator,
        grammar_no_gap=grammar_no_gap,
    )

    calibration_meta = ExperimentMetadata(
        name="gap_calibration",
        timestamp=utc_timestamp(),
        runtime_seconds=calibration_timer.elapsed(),
        n_graphs=len(SEEDS) * len(EPSILONS),
        n_extractions=len(SEEDS) * len(EPSILONS) * 2,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "calibrated_gap": calibrated_gap,
            "source_distribution": "greedy_max_gap",
        },
    )
    save_results("gap_calibration", calibration, calibration_meta)

    grammar_gap = GrammarConfig(
        min_prefix_elements=grammar_no_gap.min_prefix_elements,
        max_phase_regressions=grammar_no_gap.max_phase_regressions,
        max_turning_points=grammar_no_gap.max_turning_points,
        min_length=grammar_no_gap.min_length,
        max_length=grammar_no_gap.max_length,
        min_timespan_fraction=grammar_no_gap.min_timespan_fraction,
        max_temporal_gap=calibrated_gap,
        focal_actor_coverage=grammar_no_gap.focal_actor_coverage,
    )

    timer = ExperimentTimer()
    per_case: list[dict] = []

    for seed in SEEDS:
        for epsilon in EPSILONS:
            graph = generator.generate(
                BurstyConfig(seed=seed, epsilon=epsilon, n_events=N_EVENTS, n_actors=N_ACTORS)
            )
            no_gap_results = _evaluate_algorithms(graph=graph, grammar=grammar_no_gap)
            with_gap_results = _evaluate_algorithms(graph=graph, grammar=grammar_gap)

            per_case.append(
                {
                    "seed": seed,
                    "epsilon": epsilon,
                    "results": {
                        "no_gap": no_gap_results,
                        "with_gap": with_gap_results,
                    },
                }
            )

    algorithm_keys = ["greedy", "vag_span_only", "vag_gap_aware", "oracle", "exact_oracle"]
    aggregate = {
        condition: {
            algo: _aggregate_validity(per_case, condition, algo)
            for algo in algorithm_keys
        }
        for condition in ["no_gap", "with_gap"]
    }

    summary_table = []
    for algo in algorithm_keys:
        no_gap_rate = aggregate["no_gap"][algo]["valid_rate"]
        with_gap_rate = aggregate["with_gap"][algo]["valid_rate"]
        summary_table.append(
            {
                "algorithm": algo,
                "no_gap_rate": float(no_gap_rate),
                "with_gap_rate": float(with_gap_rate),
                "delta": float(with_gap_rate - no_gap_rate),
            }
        )

    vag_regressions_no_gap = sum(
        1
        for row in per_case
        if row["results"]["no_gap"]["greedy"]["valid"]
        and not row["results"]["no_gap"]["vag_span_only"]["valid"]
    )
    vag_regressions_with_gap = sum(
        1
        for row in per_case
        if row["results"]["with_gap"]["greedy"]["valid"]
        and not row["results"]["with_gap"]["vag_span_only"]["valid"]
    )
    vag_failures_vs_oracle_with_gap = sum(
        1
        for row in per_case
        if row["results"]["with_gap"]["oracle"]["valid"]
        and not row["results"]["with_gap"]["vag_span_only"]["valid"]
    )
    vag_failures_vs_exact_with_gap = sum(
        1
        for row in per_case
        if row["results"]["with_gap"]["exact_oracle"]["valid"]
        and not row["results"]["with_gap"]["vag_span_only"]["valid"]
    )

    data = {
        "settings": {
            "epsilons": EPSILONS,
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "focal_actor": FOCAL_ACTOR,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "grammar_no_gap": asdict(grammar_no_gap),
            "grammar_with_gap": asdict(grammar_gap),
        },
        "calibration": calibration,
        "aggregate": aggregate,
        "summary_table": summary_table,
        "key_metrics": {
            "vag_regressions_with_gap": int(vag_regressions_with_gap),
            "vag_regressions_no_gap": int(vag_regressions_no_gap),
            "vag_failures_vs_oracle_with_gap": int(vag_failures_vs_oracle_with_gap),
            "vag_failures_vs_exact_with_gap": int(vag_failures_vs_exact_with_gap),
        },
        "per_case": per_case,
    }

    meta = ExperimentMetadata(
        name="gap_adversarial",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(SEEDS) * len(EPSILONS),
        n_extractions=len(SEEDS) * len(EPSILONS) * 10,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "calibrated_gap": calibrated_gap,
            "calibration_source_percentile": calibration["selected_threshold"]["source_percentile"],
        },
    )
    save_results("gap_adversarial", data, meta, summary_formatter=_summary_markdown)
    return {"metadata": meta, "results": data}


def _print_summary(results: dict) -> None:
    selected = results["calibration"]["selected_threshold"]
    print("Gap Adversarial Results")
    print("========================")
    print(
        "Calibration: "
        f"G={selected['value']:.4f} (p={selected['source_percentile']:.2f} of greedy max-gaps)"
    )
    print()
    print("                    | No gap | With gap | Delta")
    print("--------------------|--------|----------|-------")
    for row in results["summary_table"]:
        print(
            f"{row['algorithm']:<20}| {row['no_gap_rate']:.3f}  | "
            f"{row['with_gap_rate']:.3f}    | {row['delta']:+.3f}"
        )
    print()
    print(
        "VAG regressions vs greedy (with gap): "
        f"{results['key_metrics']['vag_regressions_with_gap']}"
    )
    print(
        "VAG regressions vs greedy (no gap): "
        f"{results['key_metrics']['vag_regressions_no_gap']}"
    )
    print(
        "VAG failures vs oracle (with gap): "
        f"{results['key_metrics']['vag_failures_vs_oracle_with_gap']}"
    )
    print(
        "VAG failures vs exact oracle (with gap): "
        f"{results['key_metrics']['vag_failures_vs_exact_with_gap']}"
    )


if __name__ == "__main__":
    payload = run_gap_adversarial()
    _print_summary(payload["results"])
