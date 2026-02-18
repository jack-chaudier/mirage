"""Targeted comparison on multi-burst graphs.

Checks robustness of key findings:
- bimodal quality (greedy/exact ratio)
- one-swap sufficiency on Layer-3-like failures
- VAG recovery behavior
- theorem false-positive rate
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.schemas import Event
from rhun.theory.theorem import check_precondition


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DIAGNOSTIC_PATH = OUTPUT_DIR / "multiburst_diagnostic.json"


def _timespan(events: list[Event] | tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    timestamps = [float(event.timestamp) for event in events]
    return max(timestamps) - min(timestamps)


def _set_distance(left: set[str], right: set[str]) -> dict:
    left_only = left - right
    right_only = right - left
    symmetric = len(left_only) + len(right_only)
    union = len(left | right)
    jaccard = (symmetric / union) if union > 0 else 0.0
    return {
        "left_only": sorted(left_only),
        "right_only": sorted(right_only),
        "symmetric_difference": symmetric,
        "union_count": union,
        "jaccard_distance": float(jaccard),
    }


def _sorted_by_weight(events: list[Event]) -> list[Event]:
    return sorted(
        events,
        key=lambda event: (event.weight, -event.timestamp, event.id),
        reverse=True,
    )


def _temporal_extremes(events: list[Event] | tuple[Event, ...]) -> set[str]:
    if not events:
        return set()
    min_t = min(float(event.timestamp) for event in events)
    max_t = max(float(event.timestamp) for event in events)
    return {
        event.id
        for event in events
        if abs(float(event.timestamp) - min_t) <= 1e-12 or abs(float(event.timestamp) - max_t) <= 1e-12
    }


def _span_from_ids(ids: set[str], by_id: dict[str, Event]) -> float:
    events = [by_id[event_id] for event_id in ids if event_id in by_id]
    return _timespan(events)


def _min_swaps_for_span(top_m_ids: set[str], pool_events: list[Event], required_span: float) -> dict:
    by_id = {event.id: event for event in pool_events}
    if not top_m_ids:
        return {
            "min_swaps_for_span": None,
            "span_feasible_after_swaps": False,
            "initial_span": 0.0,
            "final_span": 0.0,
            "swap_sequence": [],
            "failure_reason": "empty_top_m",
        }

    current_ids = set(top_m_ids)
    pool_ids = set(by_id.keys())
    extremes = _temporal_extremes(pool_events)
    swaps: list[dict] = []

    initial_span = _span_from_ids(current_ids, by_id)
    if initial_span + 1e-12 >= required_span:
        return {
            "min_swaps_for_span": 0,
            "span_feasible_after_swaps": True,
            "initial_span": initial_span,
            "final_span": initial_span,
            "swap_sequence": swaps,
            "failure_reason": None,
        }

    for _step in range(len(top_m_ids) + 1):
        current_span = _span_from_ids(current_ids, by_id)
        if current_span + 1e-12 >= required_span:
            break

        best_swap = None
        current_events = [by_id[event_id] for event_id in current_ids if event_id in by_id]
        add_candidates = [by_id[event_id] for event_id in (pool_ids - current_ids)]

        for remove_event in current_events:
            for add_event in add_candidates:
                trial_ids = (current_ids - {remove_event.id}) | {add_event.id}
                trial_span = _span_from_ids(trial_ids, by_id)
                span_gain = trial_span - current_span
                if span_gain <= 1e-12:
                    continue

                candidate = {
                    "remove_id": remove_event.id,
                    "add_id": add_event.id,
                    "remove_weight": float(remove_event.weight),
                    "add_weight": float(add_event.weight),
                    "remove_timestamp": float(remove_event.timestamp),
                    "add_timestamp": float(add_event.timestamp),
                    "weight_loss": float(remove_event.weight - add_event.weight),
                    "span_before": current_span,
                    "span_after": trial_span,
                    "span_gain": span_gain,
                    "add_is_temporal_extreme": bool(add_event.id in extremes),
                }

                if best_swap is None:
                    best_swap = candidate
                    continue

                better_gain = candidate["span_gain"] > best_swap["span_gain"] + 1e-12
                same_gain = abs(candidate["span_gain"] - best_swap["span_gain"]) <= 1e-12
                lower_weight_loss = candidate["weight_loss"] < best_swap["weight_loss"] - 1e-12
                same_weight_loss = abs(candidate["weight_loss"] - best_swap["weight_loss"]) <= 1e-12
                prefer_endpoint = (
                    candidate["add_is_temporal_extreme"] and not best_swap["add_is_temporal_extreme"]
                )

                if (
                    better_gain
                    or (same_gain and lower_weight_loss)
                    or (same_gain and same_weight_loss and prefer_endpoint)
                    or (
                        same_gain
                        and same_weight_loss
                        and candidate["add_id"] < best_swap["add_id"]
                    )
                ):
                    best_swap = candidate

        if best_swap is None:
            final_span = _span_from_ids(current_ids, by_id)
            return {
                "min_swaps_for_span": None,
                "span_feasible_after_swaps": False,
                "initial_span": initial_span,
                "final_span": final_span,
                "swap_sequence": swaps,
                "failure_reason": "no_span_improving_swap",
            }

        current_ids.remove(best_swap["remove_id"])
        current_ids.add(best_swap["add_id"])
        swaps.append(best_swap)

    final_span = _span_from_ids(current_ids, by_id)
    feasible = final_span + 1e-12 >= required_span
    return {
        "min_swaps_for_span": len(swaps) if feasible else None,
        "span_feasible_after_swaps": feasible,
        "initial_span": initial_span,
        "final_span": final_span,
        "swap_sequence": swaps,
        "failure_reason": None if feasible else "max_swaps_exhausted",
    }


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    bq = data["aggregates"]["bimodal_quality"]
    sw = data["aggregates"]["one_swap"]
    vr = data["aggregates"]["vag_recovery"]
    th = data["aggregates"]["theorem"]

    lines = [
        "# multiburst_comparison",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Bimodal quality",
        "",
        f"- Cases with greedy+exact valid: {bq['count']}",
        f"- Mean greedy/exact ratio: {bq['mean']:.4f}" if bq["mean"] is not None else "- Mean greedy/exact ratio: n/a",
        f"- Median greedy/exact ratio: {bq['median']:.4f}" if bq["median"] is not None else "- Median greedy/exact ratio: n/a",
        f"- Min greedy/exact ratio: {bq['min']:.4f}" if bq["min"] is not None else "- Min greedy/exact ratio: n/a",
        f"- Cases below 0.90: {bq['below_0_90']}",
        "",
        "## One-swap sufficiency (Layer-3-like cases)",
        "",
        f"- Layer-3-like case count: {sw['layer3_like_count']}",
        f"- Top-m span below threshold: {sw['top_m_below_threshold_count']}/{sw['layer3_like_count']}",
        f"- 1 swap sufficient: {sw['min_swaps_eq_1']}/{sw['layer3_like_count']}",
        f"- 2 swaps sufficient: {sw['min_swaps_eq_2']}/{sw['layer3_like_count']}",
        f"- 3+ swaps needed: {sw['min_swaps_ge_3']}/{sw['layer3_like_count']}",
        f"- Unresolved by improving-swap search: {sw['min_swaps_unresolved']}/{sw['layer3_like_count']}",
        "",
        "## VAG recovery",
        "",
        f"- Overall greedy validity: {100.0 * vr['greedy_valid_rate']:.1f}%",
        f"- Overall VAG validity: {100.0 * vr['vag_valid_rate']:.1f}%",
        f"- Improvements over greedy: {vr['improvements_over_greedy']}",
        f"- Regressions vs greedy: {vr['regressions_vs_greedy']}",
        f"- Layer-3-like recovered by VAG: {vr['layer3_like_recovered_by_vag']}/{sw['layer3_like_count']}",
        "",
        "## Theorem check",
        "",
        f"- Predicted-failure cases (j < k): {th['predicted_failure_count']}",
        f"- False positives: {th['false_positive_count']}",
        "",
    ]
    return "\n".join(lines)


def run_multiburst_comparison() -> dict:
    if not DIAGNOSTIC_PATH.exists():
        raise FileNotFoundError(
            f"Missing required input: {DIAGNOSTIC_PATH}. Run run_multiburst_diagnostic.py first."
        )

    source = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    diag = source["results"]
    settings = diag["settings"]

    grammar = GrammarConfig(**settings["grammar"])
    pool_strategy = str(settings["pool_strategy"])
    n_anchors = int(settings["n_anchors"])
    max_sequence_length = int(settings["max_sequence_length"])
    injection_top_n = int(settings["injection_top_n"])
    focal_actor = str(settings["focal_actor"])

    generator = MultiBurstGenerator()
    timer = ExperimentTimer()

    rows = diag["per_case"]
    rows.sort(key=lambda row: int(row["seed"]))

    greedy_exact_ratios: list[float] = []

    predicted_failure_count = 0
    false_positive_count = 0

    layer3_like_cases: list[dict] = []

    greedy_valid_count = 0
    vag_valid_count = 0
    improvements_over_greedy = 0
    regressions_vs_greedy = 0
    layer3_like_recovered_by_vag = 0

    for row in rows:
        seed = int(row["seed"])
        config = MultiBurstConfig(
            seed=seed,
            n_events=int(row["generator_config"]["n_events"]),
            n_actors=int(row["generator_config"]["n_actors"]),
            burst_centers=(
                float(row["generator_config"]["burst_centers"][0]),
                float(row["generator_config"]["burst_centers"][1]),
            ),
            burst_width=float(row["generator_config"]["burst_width"]),
            burst_weight_boost=float(row["generator_config"]["burst_weight_boost"]),
            inter_burst_density=float(row["generator_config"]["inter_burst_density"]),
        )
        graph = generator.generate(config)

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
        exact, exact_diag = exact_oracle_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
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

        theorem = check_precondition(graph=graph, focal_actor=focal_actor, grammar=grammar)
        predicted_failure = bool(theorem["predicted_failure"])
        greedy_valid = bool(greedy.valid)
        oracle_valid = bool(oracle is not None and oracle.valid)
        exact_valid = bool(exact.valid)
        vag_valid = bool(vag.valid)

        if predicted_failure:
            predicted_failure_count += 1
            if greedy_valid:
                false_positive_count += 1

        if greedy_valid:
            greedy_valid_count += 1
        if vag_valid:
            vag_valid_count += 1
        if vag_valid and not greedy_valid:
            improvements_over_greedy += 1
        if greedy_valid and not vag_valid:
            regressions_vs_greedy += 1

        if greedy_valid and exact_valid and exact.score > 0:
            greedy_exact_ratios.append(float(greedy.score / exact.score))

        is_layer3_like = (
            (not greedy_valid)
            and exact_valid
            and any(v.startswith("insufficient_timespan") for v in greedy.violations)
        )
        if not is_layer3_like:
            continue

        if vag_valid:
            layer3_like_recovered_by_vag += 1

        by_id = {event.id: event for event in graph.events}
        raw_pool_ids = greedy.metadata.get("pool_ids")
        if not isinstance(raw_pool_ids, (tuple, list, set)):
            pool_ids = set()
        else:
            pool_ids = {str(event_id) for event_id in raw_pool_ids}

        pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        pool_events = sorted(pool_events, key=lambda event: (event.timestamp, event.id))

        target = exact if exact_valid else oracle
        if target is None:
            continue
        target_ids = {event.id for event in target.events}

        m = len(target.events)
        top_m_events = _sorted_by_weight(pool_events)[:m] if m > 0 else []
        top_m_ids = {event.id for event in top_m_events}

        required_span = float(grammar.min_timespan_fraction * graph.duration)
        span_top_m = _timespan(top_m_events)
        swaps = _min_swaps_for_span(top_m_ids, pool_events, required_span)

        set_distance = _set_distance(top_m_ids, target_ids)
        pool_extremes = _temporal_extremes(pool_events)

        topm_only_ids = set_distance["left_only"]
        target_only_ids = set_distance["right_only"]

        layer3_like_cases.append(
            {
                "seed": seed,
                "target_type": "exact" if exact_valid else "oracle",
                "required_span": required_span,
                "span_top_m": span_top_m,
                "span_top_m_ratio": (span_top_m / required_span) if required_span > 0 else float("inf"),
                "top_m_below_threshold": bool(span_top_m + 1e-12 < required_span),
                "set_distance_topm_vs_target": set_distance,
                "min_swaps": swaps,
                "swap_in_is_temporal_extreme": bool(
                    swaps["swap_sequence"] and swaps["swap_sequence"][0]["add_id"] in pool_extremes
                ),
                "oracle_valid": oracle_valid,
                "exact_valid": exact_valid,
                "oracle_diagnostics": oracle_diag,
                "exact_diagnostics": exact_diag,
                "vag_valid": vag_valid,
                "vag_diagnostics": vag_diag,
                "topm_only_ids": topm_only_ids,
                "target_only_ids": target_only_ids,
            }
        )

    n_cases = len(rows)

    one_swap = sum(
        1
        for case in layer3_like_cases
        if case["min_swaps"]["min_swaps_for_span"] == 1
    )
    two_swap = sum(
        1
        for case in layer3_like_cases
        if case["min_swaps"]["min_swaps_for_span"] == 2
    )
    three_plus = sum(
        1
        for case in layer3_like_cases
        if case["min_swaps"]["min_swaps_for_span"] is not None
        and case["min_swaps"]["min_swaps_for_span"] >= 3
    )
    unresolved = sum(
        1 for case in layer3_like_cases if case["min_swaps"]["min_swaps_for_span"] is None
    )
    top_m_below = sum(1 for case in layer3_like_cases if case["top_m_below_threshold"])

    symmetric_values = [
        case["set_distance_topm_vs_target"]["symmetric_difference"] for case in layer3_like_cases
    ]

    results = {
        "settings": {
            "source": str(DIAGNOSTIC_PATH.name),
            "grammar": settings["grammar"],
            "pool_strategy": pool_strategy,
            "n_anchors": n_anchors,
            "max_sequence_length": max_sequence_length,
            "injection_top_n": injection_top_n,
            "focal_actor": focal_actor,
        },
        "aggregates": {
            "bimodal_quality": {
                "count": len(greedy_exact_ratios),
                "mean": _safe_mean(greedy_exact_ratios),
                "median": _safe_median(greedy_exact_ratios),
                "min": float(min(greedy_exact_ratios)) if greedy_exact_ratios else None,
                "below_0_90": int(sum(1 for value in greedy_exact_ratios if value < 0.90)),
            },
            "one_swap": {
                "layer3_like_count": len(layer3_like_cases),
                "top_m_below_threshold_count": top_m_below,
                "symmetric_difference_mean": _safe_mean([float(v) for v in symmetric_values]),
                "symmetric_difference_median": _safe_median([float(v) for v in symmetric_values]),
                "min_swaps_eq_1": one_swap,
                "min_swaps_eq_2": two_swap,
                "min_swaps_ge_3": three_plus,
                "min_swaps_unresolved": unresolved,
            },
            "vag_recovery": {
                "n_cases": n_cases,
                "greedy_valid_count": greedy_valid_count,
                "vag_valid_count": vag_valid_count,
                "greedy_valid_rate": float(greedy_valid_count / n_cases) if n_cases else 0.0,
                "vag_valid_rate": float(vag_valid_count / n_cases) if n_cases else 0.0,
                "improvements_over_greedy": improvements_over_greedy,
                "regressions_vs_greedy": regressions_vs_greedy,
                "layer3_like_recovered_by_vag": layer3_like_recovered_by_vag,
            },
            "theorem": {
                "predicted_failure_count": predicted_failure_count,
                "false_positive_count": false_positive_count,
            },
        },
        "per_case": {
            "layer3_like": layer3_like_cases,
        },
    }

    metadata = ExperimentMetadata(
        name="multiburst_comparison",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=n_cases * 4,
        seed_range=(min(int(row["seed"]) for row in rows), max(int(row["seed"]) for row in rows)),
        parameters={
            "source": str(DIAGNOSTIC_PATH.name),
            "analysis": [
                "bimodal_quality",
                "one_swap_sufficiency",
                "vag_recovery",
                "theorem_false_positives",
            ],
        },
    )

    save_results(
        "multiburst_comparison",
        results,
        metadata,
        summary_formatter=_summary_markdown,
    )
    return {"metadata": metadata, "results": results}


if __name__ == "__main__":
    payload = run_multiburst_comparison()
    agg = payload["results"]["aggregates"]
    one = agg["one_swap"]
    vag = agg["vag_recovery"]
    qual = agg["bimodal_quality"]

    print("Multi-Burst Comparison")
    print("=" * 24)
    print(
        f"Bimodal quality: mean={qual['mean']:.4f} min={qual['min']:.4f} below_0.90={qual['below_0_90']}"
        if qual["mean"] is not None and qual["min"] is not None
        else "Bimodal quality: n/a"
    )
    print(
        "One-swap sufficiency:\n"
        f"  layer3_like={one['layer3_like_count']}\n"
        f"  top-m below threshold={one['top_m_below_threshold_count']}\n"
        f"  1-swap={one['min_swaps_eq_1']} 2-swap={one['min_swaps_eq_2']} "
        f"3+-swap={one['min_swaps_ge_3']} unresolved={one['min_swaps_unresolved']}"
    )
    print(
        "VAG recovery:\n"
        f"  greedy_valid={vag['greedy_valid_count']}/{vag['n_cases']}\n"
        f"  vag_valid={vag['vag_valid_count']}/{vag['n_cases']}\n"
        f"  improvements={vag['improvements_over_greedy']} regressions={vag['regressions_vs_greedy']}"
    )
