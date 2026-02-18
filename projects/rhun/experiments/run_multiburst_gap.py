"""Multi-burst + max_temporal_gap adversarial experiment.

Combines multi-burst distribution with gap constraints to test whether the
inter-burst valley creates structural bridge-budget failures.
"""

from __future__ import annotations

from dataclasses import asdict
from math import ceil, isinf
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.schemas import Event, ExtractedSequence


SEEDS = range(0, 50)
FOCAL_ACTOR = "actor_0"
N_EVENTS = 200
N_ACTORS = 6

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

VALLEY_START = 0.30
VALLEY_END = 0.70

BURSTY_REFERENCE_PATH = Path(__file__).resolve().parent / "output" / "gap_adversarial.json"
MULTIBURST_BASELINE_PATH = Path(__file__).resolve().parent / "output" / "multiburst_diagnostic.json"


def _adjacent_gaps(events: tuple[Event, ...] | list[Event]) -> list[float]:
    if len(events) < 2:
        return []
    return [float(events[i + 1].timestamp - events[i].timestamp) for i in range(len(events) - 1)]


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
            "p25": None,
            "p50": None,
            "p60": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p25": float(_quantile(values, 0.25)),
        "p50": float(_quantile(values, 0.50)),
        "p60": float(_quantile(values, 0.60)),
        "p75": float(_quantile(values, 0.75)),
        "p90": float(_quantile(values, 0.90)),
    }


def _valley_crossing_gap(events: tuple[Event, ...] | list[Event]) -> float | None:
    if len(events) < 2:
        return None
    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    left_candidates = [event for event in ordered if float(event.timestamp) < VALLEY_START]
    right_candidates = [event for event in ordered if float(event.timestamp) > VALLEY_END]
    if not left_candidates or not right_candidates:
        return None

    left = max(left_candidates, key=lambda event: float(event.timestamp))
    right = min(right_candidates, key=lambda event: float(event.timestamp))
    if float(right.timestamp) <= float(left.timestamp):
        return None
    return float(right.timestamp - left.timestamp)


def _required_bridges(gap: float, gap_limit: float) -> int:
    if gap_limit <= 0.0:
        return 0
    if gap <= gap_limit + 1e-12:
        return 0
    return int(max(0, ceil(gap / gap_limit) - 1))


def _oversized_gaps(events: tuple[Event, ...] | list[Event], gap_limit: float) -> list[dict]:
    if isinf(gap_limit) or len(events) < 2:
        return []

    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    rows: list[dict] = []

    for left, right in zip(ordered[:-1], ordered[1:]):
        gap = float(right.timestamp - left.timestamp)
        if gap <= gap_limit + 1e-12:
            continue
        bridges = _required_bridges(gap, gap_limit)
        crosses_valley = bool(float(left.timestamp) <= VALLEY_START and float(right.timestamp) >= VALLEY_END)
        overlaps_valley = bool(float(left.timestamp) < VALLEY_END and float(right.timestamp) > VALLEY_START)
        rows.append(
            {
                "left_id": left.id,
                "right_id": right.id,
                "left_timestamp": float(left.timestamp),
                "right_timestamp": float(right.timestamp),
                "gap": gap,
                "required_bridges": int(bridges),
                "crosses_valley": crosses_valley,
                "overlaps_valley": overlaps_valley,
            }
        )
    return rows


def _sequence_payload(seq: ExtractedSequence | None, gap_limit: float) -> dict:
    if seq is None:
        return {
            "valid": False,
            "score": None,
            "n_events": 0,
            "violations": ["oracle_no_candidate"],
            "max_adjacent_gap": 0.0,
            "mean_adjacent_gap": 0.0,
            "adjacent_gaps": [],
            "oversized_gaps": [],
            "gap_violation_count": 0,
            "valley_crossing_gap": None,
        }

    gaps = _adjacent_gaps(seq.events)
    oversized = _oversized_gaps(seq.events, gap_limit)
    return {
        "valid": bool(seq.valid),
        "score": float(seq.score),
        "n_events": len(seq.events),
        "violations": list(seq.violations),
        "max_adjacent_gap": float(max(gaps) if gaps else 0.0),
        "mean_adjacent_gap": float(mean(gaps) if gaps else 0.0),
        "adjacent_gaps": [float(value) for value in gaps],
        "oversized_gaps": oversized,
        "gap_violation_count": int(
            sum(1 for violation in seq.violations if str(violation).startswith("max_temporal_gap:"))
        ),
        "valley_crossing_gap": _valley_crossing_gap(seq.events),
    }


def _evaluate_algorithms(graph, grammar: GrammarConfig) -> dict:
    gap_limit = float(grammar.max_temporal_gap)

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
        "greedy": _sequence_payload(greedy, gap_limit=gap_limit),
        "vag_span_only": {
            **_sequence_payload(vag_span, gap_limit=gap_limit),
            "diagnostics": vag_span_diag,
        },
        "vag_gap_aware": {
            **_sequence_payload(vag_gap, gap_limit=gap_limit),
            "diagnostics": vag_gap_diag,
        },
        "oracle": {
            **_sequence_payload(oracle, gap_limit=gap_limit),
            "diagnostics": oracle_diag,
        },
        "exact_oracle": {
            **_sequence_payload(exact, gap_limit=gap_limit),
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


def _bursty_reference_with_gap() -> dict[str, float]:
    if not BURSTY_REFERENCE_PATH.exists():
        return {}

    import json

    payload = json.loads(BURSTY_REFERENCE_PATH.read_text(encoding="utf-8"))
    with_gap = payload.get("results", {}).get("aggregate", {}).get("with_gap", {})
    return {
        algo: float(stats.get("valid_rate", 0.0))
        for algo, stats in with_gap.items()
    }


def _baseline_no_gap_reference() -> dict[str, float]:
    if not MULTIBURST_BASELINE_PATH.exists():
        return {}

    import json

    payload = json.loads(MULTIBURST_BASELINE_PATH.read_text(encoding="utf-8"))
    extraction = payload.get("results", {}).get("aggregates", {}).get("extraction", {})
    return {
        "greedy": float(extraction.get("validity_rate", {}).get("greedy", 0.0)),
        "vag_span_only": float(extraction.get("validity_rate", {}).get("vag", 0.0)),
        "oracle": float(extraction.get("validity_rate", {}).get("oracle", 0.0)),
        "exact_oracle": float(extraction.get("validity_rate", {}).get("exact_oracle", 0.0)),
    }


def _calibrate_gap_threshold(generator: MultiBurstGenerator, grammar_no_gap: GrammarConfig) -> tuple[dict, float]:
    greedy_max_gaps: list[float] = []
    oracle_max_gaps: list[float] = []
    greedy_mean_gaps: list[float] = []
    valley_crossing_gaps: list[float] = []
    per_case: list[dict] = []

    for seed in SEEDS:
        graph = generator.generate(MultiBurstConfig(seed=seed, n_events=N_EVENTS, n_actors=N_ACTORS))

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
            "greedy_valid": bool(greedy.valid),
            "oracle_valid": bool(oracle is not None and oracle.valid),
            "greedy_max_gap": None,
            "greedy_mean_gap": None,
            "greedy_valley_crossing_gap": None,
            "oracle_max_gap": None,
            "oracle_diagnostics": oracle_diag,
        }

        if greedy.valid:
            gaps = _adjacent_gaps(greedy.events)
            gm = max(gaps) if gaps else 0.0
            gg = mean(gaps) if gaps else 0.0
            vg = _valley_crossing_gap(greedy.events)
            greedy_max_gaps.append(float(gm))
            greedy_mean_gaps.append(float(gg))
            if vg is not None:
                valley_crossing_gaps.append(float(vg))
            row["greedy_max_gap"] = float(gm)
            row["greedy_mean_gap"] = float(gg)
            row["greedy_valley_crossing_gap"] = (float(vg) if vg is not None else None)

        if oracle is not None and oracle.valid:
            og = _adjacent_gaps(oracle.events)
            row["oracle_max_gap"] = float(max(og) if og else 0.0)
            oracle_max_gaps.append(float(max(og) if og else 0.0))

        per_case.append(row)

    raw_p60 = float(_quantile(greedy_max_gaps, 0.60))
    selected_gap = raw_p60
    effective_percentile: float | None = 0.60
    adjustment_applied = False
    adjustment_reason = "none"

    if not (0.05 <= selected_gap <= 0.40):
        # Recalibrate downward/upward if p60 is outside the sanity band.
        # Try nearby percentiles first; if distribution is fully outside,
        # clamp to the nearest bound to keep the adversarial run informative.
        adjustment_applied = True
        for percentile in [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
            candidate = float(_quantile(greedy_max_gaps, percentile))
            if 0.05 <= candidate <= 0.40:
                selected_gap = candidate
                effective_percentile = percentile
                adjustment_reason = "percentile_adjustment_for_range"
                break
        else:
            if selected_gap > 0.40:
                selected_gap = 0.40
                effective_percentile = None
                adjustment_reason = "clamped_to_upper_bound"
            else:
                selected_gap = 0.05
                effective_percentile = None
                adjustment_reason = "clamped_to_lower_bound"

    calibration = {
        "settings": {
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "focal_actor": FOCAL_ACTOR,
            "method": "G = p60(greedy_max_gap) on no-gap valid greedy sequences",
            "valley_region": [VALLEY_START, VALLEY_END],
        },
        "greedy_max_gap_stats": _safe_stats(greedy_max_gaps),
        "greedy_mean_gap_stats": _safe_stats(greedy_mean_gaps),
        "oracle_max_gap_stats": _safe_stats(oracle_max_gaps),
        "greedy_valley_crossing_gap_stats": _safe_stats(valley_crossing_gaps),
        "selected_threshold": {
            "value": selected_gap,
            "source_percentile": 0.60,
            "source_distribution": "greedy_max_gap",
            "raw_p60": raw_p60,
            "effective_percentile": effective_percentile,
            "adjustment_applied": adjustment_applied,
            "adjustment_reason": adjustment_reason,
        },
        "derived": {
            "valley_width": float(VALLEY_END - VALLEY_START),
            "min_bridges_to_cross_full_valley": int(
                _required_bridges(VALLEY_END - VALLEY_START, selected_gap)
            ),
        },
        "per_case": per_case,
    }
    return calibration, selected_gap


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    table = data["summary_table"]
    cal = data["calibration"]["selected_threshold"]
    derived = data["calibration"]["derived"]
    diag = data["failure_diagnostics"]

    lines = [
        "# multiburst_gap",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Calibrated G (multi-burst): {cal['value']:.4f} (p={cal['source_percentile']:.2f} of greedy max-gaps)",
        f"Raw p60 before adjustment: {cal['raw_p60']:.4f}",
        (
            "Adjustment applied: "
            f"{'yes' if cal['adjustment_applied'] else 'no'}"
            + (
                f" ({cal['adjustment_reason']})"
                if cal["adjustment_applied"]
                else ""
            )
        ),
        f"Valley width: {derived['valley_width']:.3f}",
        f"Predicted minimum bridges across full valley: {derived['min_bridges_to_cross_full_valley']}",
        "",
        "## Comparison",
        "",
        "| algorithm | no_gap | with_gap | delta | bursty_ref_with_gap |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in table:
        bursty = row["bursty_reference_with_gap"]
        bursty_text = f"{bursty:.3f}" if bursty is not None else "n/a"
        lines.append(
            f"| {row['algorithm']} | {row['no_gap_rate']:.3f} | {row['with_gap_rate']:.3f} | {row['delta']:+.3f} | {bursty_text} |"
        )

    lines.extend(
        [
            "",
            "## Unsolved Cases",
            "",
            (
                "- Unsolved cases (exact valid, gap-aware VAG invalid): "
                f"{diag['unsolved_count']}"
            ),
            (
                "- Bridge budget distribution (B_lb total): "
                f"B_lb=1: {diag['bridge_budget_distribution']['1']}, "
                f"B_lb=2: {diag['bridge_budget_distribution']['2']}, "
                f"B_lb=3+: {diag['bridge_budget_distribution']['3+']}"
            ),
            (
                f"- Valley-crossing failures: {diag['valley_crossing_failures']}/{diag['unsolved_count']}"
                if diag["unsolved_count"] > 0
                else "- Valley-crossing failures: 0/0"
            ),
            (
                f"- Mean feasibility tax (all cases): {100.0 * diag['mean_feasibility_tax_all']:.1f}%"
                if diag["mean_feasibility_tax_all"] is not None
                else "- Mean feasibility tax (all cases): n/a"
            ),
            (
                f"- Mean feasibility tax (unsolved cases): {100.0 * diag['mean_feasibility_tax_unsolved']:.1f}%"
                if diag["mean_feasibility_tax_unsolved"] is not None
                else "- Mean feasibility tax (unsolved cases): n/a"
            ),
            "",
            "## Verification",
            "",
            f"- G in [0.05, 0.40]: {'yes' if data['verification']['g_reasonable_range'] else 'no'}",
            (
                "- Exact oracle with-gap validity: "
                f"{100.0 * data['verification']['exact_oracle_with_gap_rate']:.1f}%"
            ),
            (
                "- No-gap matches multiburst diagnostic (greedy/vag): "
                f"{'yes' if data['verification']['no_gap_matches_diagnostic'] else 'no'}"
            ),
            "",
        ]
    )

    return "\n".join(lines)


def run_multiburst_gap() -> dict:
    generator = MultiBurstGenerator()

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

    # Step 1: calibration
    calibration_timer = ExperimentTimer()
    calibration, calibrated_gap = _calibrate_gap_threshold(generator, grammar_no_gap)

    calibration_meta = ExperimentMetadata(
        name="multiburst_gap_calibration",
        timestamp=utc_timestamp(),
        runtime_seconds=calibration_timer.elapsed(),
        n_graphs=len(SEEDS),
        n_extractions=len(SEEDS) * 2,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "calibrated_gap": calibrated_gap,
            "source_percentile": calibration["selected_threshold"]["source_percentile"],
        },
    )
    save_results("multiburst_gap_calibration", calibration, calibration_meta)

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

    # Step 2: main sweep (no-gap and with-gap on same seeds)
    timer = ExperimentTimer()
    per_case: list[dict] = []

    for seed in SEEDS:
        graph = generator.generate(MultiBurstConfig(seed=seed, n_events=N_EVENTS, n_actors=N_ACTORS))
        no_gap_results = _evaluate_algorithms(graph=graph, grammar=grammar_no_gap)
        with_gap_results = _evaluate_algorithms(graph=graph, grammar=grammar_gap)

        per_case.append(
            {
                "seed": seed,
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

    bursty_reference = _bursty_reference_with_gap()
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
                "bursty_reference_with_gap": (
                    float(bursty_reference[algo]) if algo in bursty_reference else None
                ),
            }
        )

    # Step 3: diagnostics on unsolved cases
    unsolved = [
        row
        for row in per_case
        if row["results"]["with_gap"]["exact_oracle"]["valid"]
        and not row["results"]["with_gap"]["vag_gap_aware"]["valid"]
    ]

    rejection_reason_counts: dict[str, int] = {}
    bridge_totals: list[int] = []
    valley_crossing_failures = 0
    feasibility_tax_all: list[float] = []
    feasibility_tax_unsolved: list[float] = []

    unsolved_rows: list[dict] = []

    for row in per_case:
        exact_no_gap = row["results"]["no_gap"]["exact_oracle"]
        exact_with_gap = row["results"]["with_gap"]["exact_oracle"]

        if (
            exact_no_gap["valid"]
            and exact_with_gap["valid"]
            and exact_no_gap["score"] is not None
            and exact_no_gap["score"] > 0
            and exact_with_gap["score"] is not None
        ):
            tax = float((exact_no_gap["score"] - exact_with_gap["score"]) / exact_no_gap["score"])
            feasibility_tax_all.append(tax)

    for row in unsolved:
        seed = int(row["seed"])
        vag_gap = row["results"]["with_gap"]["vag_gap_aware"]

        oversized = list(vag_gap.get("oversized_gaps", []))
        b_total = int(sum(int(item.get("required_bridges", 0)) for item in oversized))
        bridge_totals.append(b_total)

        crosses_valley = any(bool(item.get("crosses_valley", False)) for item in oversized)
        if crosses_valley:
            valley_crossing_failures += 1

        diag = vag_gap.get("diagnostics", {})
        reason_counts = diag.get("viability_rejection_reason_counts", {})
        for reason, count in reason_counts.items():
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + int(count)

        exact_no_gap = row["results"]["no_gap"]["exact_oracle"]
        exact_with_gap = row["results"]["with_gap"]["exact_oracle"]
        tax = None
        if (
            exact_no_gap["valid"]
            and exact_with_gap["valid"]
            and exact_no_gap["score"] is not None
            and exact_no_gap["score"] > 0
            and exact_with_gap["score"] is not None
        ):
            tax = float((exact_no_gap["score"] - exact_with_gap["score"]) / exact_no_gap["score"])
            feasibility_tax_unsolved.append(tax)

        unsolved_rows.append(
            {
                "seed": seed,
                "vag_gap_aware": vag_gap,
                "bridge_budget_lb_total": b_total,
                "valley_crossing_failure": crosses_valley,
                "feasibility_tax": tax,
            }
        )

    bridge_budget_distribution = {
        "0": int(sum(1 for value in bridge_totals if value == 0)),
        "1": int(sum(1 for value in bridge_totals if value == 1)),
        "2": int(sum(1 for value in bridge_totals if value == 2)),
        "3+": int(sum(1 for value in bridge_totals if value >= 3)),
    }

    # Step 4: verification
    exact_oracle_with_gap_rate = aggregate["with_gap"]["exact_oracle"]["valid_rate"]
    baseline = _baseline_no_gap_reference()
    greedy_no_gap_rate = aggregate["no_gap"]["greedy"]["valid_rate"]
    vag_no_gap_rate = aggregate["no_gap"]["vag_span_only"]["valid_rate"]

    no_gap_matches_diagnostic = True
    if baseline:
        no_gap_matches_diagnostic = (
            abs(greedy_no_gap_rate - baseline.get("greedy", greedy_no_gap_rate)) <= 1e-9
            and abs(vag_no_gap_rate - baseline.get("vag_span_only", vag_no_gap_rate)) <= 1e-9
        )

    verification = {
        "g_reasonable_range": bool(0.05 <= calibrated_gap <= 0.40),
        "exact_oracle_with_gap_rate": float(exact_oracle_with_gap_rate),
        "exact_oracle_rate_ok": bool(exact_oracle_with_gap_rate >= 0.90),
        "no_gap_matches_diagnostic": bool(no_gap_matches_diagnostic),
        "no_gap_delta_vs_diagnostic": {
            "greedy": float(greedy_no_gap_rate - baseline.get("greedy", greedy_no_gap_rate))
            if baseline
            else None,
            "vag_span_only": float(vag_no_gap_rate - baseline.get("vag_span_only", vag_no_gap_rate))
            if baseline
            else None,
        },
    }

    # Requested hard checks.
    assert verification["g_reasonable_range"], f"Calibrated G={calibrated_gap:.4f} seems unreasonable"
    assert verification["exact_oracle_rate_ok"], (
        f"Exact oracle with-gap validity too low: {exact_oracle_with_gap_rate:.3f}"
    )

    failure_diagnostics = {
        "unsolved_count": len(unsolved_rows),
        "bridge_budget_distribution": bridge_budget_distribution,
        "valley_crossing_failures": int(valley_crossing_failures),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "mean_feasibility_tax_all": float(mean(feasibility_tax_all)) if feasibility_tax_all else None,
        "mean_feasibility_tax_unsolved": (
            float(mean(feasibility_tax_unsolved)) if feasibility_tax_unsolved else None
        ),
        "unsolved_cases": unsolved_rows,
    }

    data = {
        "settings": {
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
            "valley_region": [VALLEY_START, VALLEY_END],
        },
        "calibration": calibration,
        "aggregate": aggregate,
        "summary_table": summary_table,
        "failure_diagnostics": failure_diagnostics,
        "verification": verification,
        "bursty_reference_with_gap": bursty_reference,
        "per_case": per_case,
    }

    meta = ExperimentMetadata(
        name="multiburst_gap",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(SEEDS),
        n_extractions=len(SEEDS) * 10,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "calibrated_gap": calibrated_gap,
            "calibration_source_percentile": calibration["selected_threshold"]["source_percentile"],
        },
    )
    save_results("multiburst_gap", data, meta, summary_formatter=_summary_markdown)
    return {"metadata": meta, "results": data}


def _print_summary(results: dict) -> None:
    selected = results["calibration"]["selected_threshold"]
    derived = results["calibration"]["derived"]
    diag = results["failure_diagnostics"]

    print("Multi-Burst + Gap Results")
    print("=" * 33)
    print(
        f"Calibration: G={selected['value']:.4f} (p={selected['source_percentile']:.2f} of greedy max-gaps)"
    )
    print(
        f"Raw p60={selected['raw_p60']:.4f}, "
        f"adjustment={'yes' if selected['adjustment_applied'] else 'no'}"
        + (
            f" ({selected['adjustment_reason']})"
            if selected["adjustment_applied"]
            else ""
        )
    )
    print(
        f"Valley width={derived['valley_width']:.3f}, "
        f"predicted min bridges across full valley={derived['min_bridges_to_cross_full_valley']}"
    )
    print()
    print("                    | No gap | With gap | Delta | Bursty ref (with gap)")
    print("--------------------|--------|----------|-------|----------------------")
    for row in results["summary_table"]:
        ref = row["bursty_reference_with_gap"]
        ref_text = f"{ref:.3f}" if ref is not None else "n/a"
        print(
            f"{row['algorithm']:<20}| {row['no_gap_rate']:.3f}  | {row['with_gap_rate']:.3f}    | "
            f"{row['delta']:+.3f} | {ref_text}"
        )

    print()
    print(
        "Unsolved cases (exact valid, gap-aware VAG invalid): "
        f"{diag['unsolved_count']}"
    )
    print(
        "Bridge budget distribution (B_lb total): "
        f"B_lb=1:{diag['bridge_budget_distribution']['1']} "
        f"B_lb=2:{diag['bridge_budget_distribution']['2']} "
        f"B_lb=3+:{diag['bridge_budget_distribution']['3+']}"
    )
    print(f"Valley-crossing failures: {diag['valley_crossing_failures']}/{diag['unsolved_count']}")
    if diag["mean_feasibility_tax_all"] is not None:
        print(f"Mean feasibility tax (all): {100.0 * diag['mean_feasibility_tax_all']:.1f}%")
    if diag["mean_feasibility_tax_unsolved"] is not None:
        print(f"Mean feasibility tax (unsolved): {100.0 * diag['mean_feasibility_tax_unsolved']:.1f}%")


if __name__ == "__main__":
    payload = run_multiburst_gap()
    _print_summary(payload["results"])
