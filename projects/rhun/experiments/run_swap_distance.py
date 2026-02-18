"""Swap-distance verification on Layer 3 assembly-compressed false negatives.

Measures how far greedy and top-m (weight-optimal pool core) are from oracle
on the 57 Layer 3 cases and estimates swap distance to restore span feasibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import Event


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LAYER3_ANTICORR_PATH = OUTPUT_DIR / "layer3_anticorrelation.json"


def _timespan(events: list[Event] | tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    timestamps = [float(event.timestamp) for event in events]
    return max(timestamps) - min(timestamps)


def _set_distance(left: set[str], right: set[str]) -> dict:
    left_only = left - right
    right_only = right - left
    symmetric = len(left_only) + len(right_only)
    union_n = len(left | right)
    jaccard_distance = (symmetric / union_n) if union_n > 0 else 0.0
    return {
        "left_only_ids": sorted(left_only),
        "right_only_ids": sorted(right_only),
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "symmetric_difference": symmetric,
        "union_count": union_n,
        "jaccard_distance": jaccard_distance,
    }


def _temporal_extreme_ids(events: list[Event] | tuple[Event, ...]) -> set[str]:
    if not events:
        return set()
    min_t = min(float(event.timestamp) for event in events)
    max_t = max(float(event.timestamp) for event in events)
    return {
        event.id
        for event in events
        if float(event.timestamp) == min_t or float(event.timestamp) == max_t
    }


def _sorted_by_weight(events: list[Event]) -> list[Event]:
    return sorted(
        events,
        key=lambda event: (event.weight, -event.timestamp, event.id),
        reverse=True,
    )


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


def _span_from_ids(ids: set[str], by_id: dict[str, Event]) -> float:
    events = [by_id[event_id] for event_id in ids if event_id in by_id]
    return _timespan(events)


def _min_swaps_for_span(
    top_m_ids: set[str],
    pool_events: list[Event],
    required_span: float,
) -> dict:
    by_id = {event.id: event for event in pool_events}
    if not top_m_ids:
        return {
            "min_swaps_for_span": None,
            "span_feasible_after_swaps": False,
            "initial_span": 0.0,
            "final_span": 0.0,
            "swap_sequence": [],
            "swap_in_is_temporal_extreme": False,
            "failure_reason": "empty_top_m",
        }

    current_ids = set(top_m_ids)
    pool_ids = set(by_id)
    temporal_extremes = _temporal_extreme_ids(pool_events)
    swaps: list[dict] = []

    initial_span = _span_from_ids(current_ids, by_id)
    if initial_span + 1e-12 >= required_span:
        return {
            "min_swaps_for_span": 0,
            "span_feasible_after_swaps": True,
            "initial_span": initial_span,
            "final_span": initial_span,
            "swap_sequence": swaps,
            "swap_in_is_temporal_extreme": False,
            "failure_reason": None,
        }

    for _step in range(len(top_m_ids) + 1):
        current_span = _span_from_ids(current_ids, by_id)
        if current_span + 1e-12 >= required_span:
            break

        best_swap: dict | None = None
        current_events = [by_id[event_id] for event_id in current_ids if event_id in by_id]
        add_candidates = [by_id[event_id] for event_id in (pool_ids - current_ids)]

        for remove_event in current_events:
            for add_event in add_candidates:
                trial_ids = (current_ids - {remove_event.id}) | {add_event.id}
                trial_span = _span_from_ids(trial_ids, by_id)
                span_gain = trial_span - current_span
                if span_gain <= 1e-12:
                    continue

                weight_loss = float(remove_event.weight - add_event.weight)
                candidate = {
                    "remove_id": remove_event.id,
                    "add_id": add_event.id,
                    "remove_weight": float(remove_event.weight),
                    "add_weight": float(add_event.weight),
                    "remove_timestamp": float(remove_event.timestamp),
                    "add_timestamp": float(add_event.timestamp),
                    "weight_loss": weight_loss,
                    "span_before": current_span,
                    "span_after": trial_span,
                    "span_gain": span_gain,
                    "add_is_temporal_extreme": bool(add_event.id in temporal_extremes),
                }

                if best_swap is None:
                    best_swap = candidate
                    continue

                better_gain = candidate["span_gain"] > best_swap["span_gain"] + 1e-12
                same_gain = abs(candidate["span_gain"] - best_swap["span_gain"]) <= 1e-12
                less_weight_loss = candidate["weight_loss"] < best_swap["weight_loss"] - 1e-12
                same_weight_loss = abs(candidate["weight_loss"] - best_swap["weight_loss"]) <= 1e-12
                better_extreme = (
                    candidate["add_is_temporal_extreme"] and not best_swap["add_is_temporal_extreme"]
                )

                if (
                    better_gain
                    or (same_gain and less_weight_loss)
                    or (same_gain and same_weight_loss and better_extreme)
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
                "swap_in_is_temporal_extreme": bool(
                    swaps and swaps[0]["add_is_temporal_extreme"]
                ),
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
        "swap_in_is_temporal_extreme": bool(swaps and swaps[0]["add_is_temporal_extreme"]),
        "failure_reason": None if feasible else "max_swaps_exhausted",
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    agg = data["aggregates"]
    lines = [
        "# swap_distance",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        f"Cases analyzed: {data['n_cases']}",
        "",
        "## Top-m vs Oracle",
        "",
        f"- Median symmetric difference: {agg['topm_oracle_symmetric_median']:.1f} events",
        f"- Mean symmetric difference: {agg['topm_oracle_symmetric_mean']:.2f} events",
        f"- Cases with symmetric diff == 2 (one swap): {agg['topm_oracle_diff_eq_2']}/{data['n_cases']}",
        f"- Cases with symmetric diff <= 4 (two swaps): {agg['topm_oracle_diff_lte_4']}/{data['n_cases']}",
        "",
        "## Minimum Swaps for Span Feasibility",
        "",
        f"- 1 swap sufficient: {agg['min_swaps_eq_1']}/{data['n_cases']}",
        f"- 2 swaps sufficient: {agg['min_swaps_eq_2']}/{data['n_cases']}",
        f"- 3+ swaps needed: {agg['min_swaps_ge_3']}/{data['n_cases']}",
        f"- Unresolved by improving-swap search: {agg['min_swaps_unresolved']}/{data['n_cases']}",
        "",
        "## Swapped-in Characterization",
        "",
        f"- Temporal endpoint in pool: {agg['swap_in_pool_endpoint_count']}/{agg['swap_in_total_count']}",
        f"- Temporal endpoint in oracle: {agg['swap_in_oracle_endpoint_count']}/{agg['swap_in_total_count']}",
        f"- Mean weight rank of swap-in: {agg['swap_in_weight_rank_mean']:.2f}",
        "",
        "## Greedy vs Oracle",
        "",
        f"- Median Jaccard distance: {agg['greedy_oracle_jaccard_median']:.3f}",
        f"- Mean Jaccard distance: {agg['greedy_oracle_jaccard_mean']:.3f}",
        f"- Mean symmetric difference: {agg['greedy_oracle_symmetric_mean']:.2f} events",
        "",
    ]
    return "\n".join(lines)


def run_swap_distance() -> dict:
    if not LAYER3_ANTICORR_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {LAYER3_ANTICORR_PATH}")

    source = json.loads(LAYER3_ANTICORR_PATH.read_text(encoding="utf-8"))
    anti = source["results"]
    settings = anti["settings"]
    grammar = GrammarConfig(**settings["grammar"])

    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    max_sequence_length = int(settings["max_sequence_length"])
    pool_strategy = str(settings["pool_strategy"])
    n_anchors = int(settings["n_anchors"])
    injection_top_n = int(settings["injection_top_n"])

    case_rows = anti["per_case"]
    case_rows.sort(key=lambda row: (float(row["epsilon"]), int(row["seed"]), str(row["focal_actor"])))

    generator = BurstyGenerator()
    timer = ExperimentTimer()
    per_case: list[dict] = []

    for row in case_rows:
        epsilon = float(row["epsilon"])
        seed = int(row["seed"])
        focal_actor = str(row["focal_actor"])

        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )
        by_id = {event.id: event for event in graph.events}

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
                f"Oracle failed for layer3 case epsilon={epsilon}, seed={seed}, focal_actor={focal_actor}"
            )

        raw_pool_ids = greedy.metadata.get("pool_ids")
        if not isinstance(raw_pool_ids, (tuple, list, set)):
            raise RuntimeError(
                f"Greedy candidate missing pool_ids for epsilon={epsilon}, seed={seed}, focal_actor={focal_actor}"
            )
        pool_ids = {str(event_id) for event_id in raw_pool_ids}
        pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
        pool_events = sorted(pool_events, key=lambda event: (event.timestamp, event.id))

        oracle_events = list(oracle.events)
        greedy_events = list(greedy.events)
        m = len(oracle_events)
        top_m_events = _sorted_by_weight(pool_events)[:m]

        greedy_ids = {event.id for event in greedy_events}
        oracle_ids = {event.id for event in oracle_events}
        top_m_ids = {event.id for event in top_m_events}

        greedy_vs_oracle = _set_distance(greedy_ids, oracle_ids)
        topm_vs_oracle = _set_distance(top_m_ids, oracle_ids)

        pool_sorted_by_weight = _sorted_by_weight(pool_events)
        pool_rank = {event.id: rank for rank, event in enumerate(pool_sorted_by_weight, start=1)}
        pool_extremes = _temporal_extreme_ids(pool_events)
        oracle_extremes = _temporal_extreme_ids(oracle_events)

        pool_positions = [float(graph.global_position(event)) for event in pool_events]
        q25 = _quantile(pool_positions, 0.25)
        q75 = _quantile(pool_positions, 0.75)

        oracle_only_vs_topm_ids = sorted(oracle_ids - top_m_ids)
        topm_only_vs_oracle_ids = sorted(top_m_ids - oracle_ids)

        oracle_only_vs_topm = []
        for event_id in oracle_only_vs_topm_ids:
            event = by_id[event_id]
            oracle_only_vs_topm.append(
                {
                    "id": event.id,
                    "weight": float(event.weight),
                    "timestamp": float(event.timestamp),
                    "position": float(graph.global_position(event)),
                    "pool_weight_rank": int(pool_rank.get(event.id, -1)),
                    "is_oracle_temporal_endpoint": bool(event.id in oracle_extremes),
                    "is_pool_temporal_extreme": bool(event.id in pool_extremes),
                }
            )

        topm_only_vs_oracle = []
        for event_id in topm_only_vs_oracle_ids:
            event = by_id[event_id]
            position = float(graph.global_position(event))
            topm_only_vs_oracle.append(
                {
                    "id": event.id,
                    "weight": float(event.weight),
                    "timestamp": float(event.timestamp),
                    "position": position,
                    "pool_weight_rank": int(pool_rank.get(event.id, -1)),
                    "is_pool_temporal_core_q25_q75": bool(q25 <= position <= q75),
                    "is_pool_temporal_extreme": bool(event.id in pool_extremes),
                }
            )

        required_span = float(grammar.min_timespan_fraction * graph.duration)
        swap_plan = _min_swaps_for_span(top_m_ids=top_m_ids, pool_events=pool_events, required_span=required_span)

        per_case.append(
            {
                "epsilon": epsilon,
                "seed": seed,
                "focal_actor": focal_actor,
                "m": int(m),
                "required_span": required_span,
                "pool_size": len(pool_events),
                "greedy_valid": bool(greedy.valid),
                "oracle_valid": bool(oracle.valid),
                "greedy_violations": list(greedy.violations),
                "oracle_violations": list(oracle.violations),
                "oracle_diagnostics": oracle_diag,
                "greedy_event_ids": sorted(greedy_ids),
                "oracle_event_ids": sorted(oracle_ids),
                "top_m_event_ids": sorted(top_m_ids),
                "greedy_vs_oracle": greedy_vs_oracle,
                "top_m_vs_oracle": topm_vs_oracle,
                "oracle_only_vs_top_m": oracle_only_vs_topm,
                "top_m_only_vs_oracle": topm_only_vs_oracle,
                "min_swaps_for_span": swap_plan["min_swaps_for_span"],
                "span_feasible_after_swaps": bool(swap_plan["span_feasible_after_swaps"]),
                "initial_top_m_span": float(swap_plan["initial_span"]),
                "final_swapped_span": float(swap_plan["final_span"]),
                "swap_sequence": swap_plan["swap_sequence"],
                "swap_in_is_temporal_extreme": bool(swap_plan["swap_in_is_temporal_extreme"]),
                "swap_search_failure_reason": swap_plan["failure_reason"],
                "selected_pool_feasible": bool(row["span_selected_pool_ratio"] >= 1.0),
            }
        )

    n_cases = len(per_case)
    if n_cases == 0:
        raise RuntimeError("No cases found in layer3_anticorrelation input.")

    topm_sym = [row["top_m_vs_oracle"]["symmetric_difference"] for row in per_case]
    topm_jaccard = [row["top_m_vs_oracle"]["jaccard_distance"] for row in per_case]
    greedy_sym = [row["greedy_vs_oracle"]["symmetric_difference"] for row in per_case]
    greedy_jaccard = [row["greedy_vs_oracle"]["jaccard_distance"] for row in per_case]

    swap_values = [row["min_swaps_for_span"] for row in per_case if row["min_swaps_for_span"] is not None]
    one_swap_cases = sum(1 for row in per_case if row["min_swaps_for_span"] == 1)
    two_swap_cases = sum(1 for row in per_case if row["min_swaps_for_span"] == 2)
    three_plus_cases = sum(1 for row in per_case if row["min_swaps_for_span"] is not None and row["min_swaps_for_span"] >= 3)
    unresolved_cases = sum(1 for row in per_case if row["min_swaps_for_span"] is None)

    selected_pool_feasible = [row for row in per_case if row["selected_pool_feasible"]]
    one_swap_selected_pool_feasible = sum(
        1 for row in selected_pool_feasible if row["min_swaps_for_span"] == 1
    )

    swap_in_payloads = [payload for row in per_case for payload in row["oracle_only_vs_top_m"]]
    swap_in_total = len(swap_in_payloads)
    swap_in_pool_endpoint = sum(1 for payload in swap_in_payloads if payload["is_pool_temporal_extreme"])
    swap_in_oracle_endpoint = sum(1 for payload in swap_in_payloads if payload["is_oracle_temporal_endpoint"])
    swap_in_ranks = [payload["pool_weight_rank"] for payload in swap_in_payloads if payload["pool_weight_rank"] > 0]

    endpoint_swaps = sum(1 for row in per_case if row["swap_in_is_temporal_extreme"])

    aggregates = {
        "topm_oracle_symmetric_median": float(median(topm_sym)),
        "topm_oracle_symmetric_mean": float(mean(topm_sym)),
        "topm_oracle_jaccard_mean": float(mean(topm_jaccard)),
        "topm_oracle_diff_eq_2": int(sum(1 for value in topm_sym if value == 2)),
        "topm_oracle_diff_lte_4": int(sum(1 for value in topm_sym if value <= 4)),
        "min_swaps_eq_1": int(one_swap_cases),
        "min_swaps_eq_2": int(two_swap_cases),
        "min_swaps_ge_3": int(three_plus_cases),
        "min_swaps_unresolved": int(unresolved_cases),
        "selected_pool_feasible_count": int(len(selected_pool_feasible)),
        "one_swap_selected_pool_feasible": int(one_swap_selected_pool_feasible),
        "swap_in_total_count": int(swap_in_total),
        "swap_in_pool_endpoint_count": int(swap_in_pool_endpoint),
        "swap_in_oracle_endpoint_count": int(swap_in_oracle_endpoint),
        "swap_in_weight_rank_mean": float(mean(swap_in_ranks)) if swap_in_ranks else 0.0,
        "greedy_oracle_jaccard_median": float(median(greedy_jaccard)),
        "greedy_oracle_jaccard_mean": float(mean(greedy_jaccard)),
        "greedy_oracle_symmetric_mean": float(mean(greedy_sym)),
    }

    verification = {
        "one_swap_cases": int(one_swap_cases),
        "endpoint_swaps": int(endpoint_swaps),
        "jaccard_greedy_vs_oracle_mean": float(mean(greedy_jaccard)),
    }

    results = {
        "n_cases": n_cases,
        "source": str(LAYER3_ANTICORR_PATH),
        "settings": settings,
        "aggregates": aggregates,
        "verification": verification,
        "per_case": per_case,
    }

    metadata = ExperimentMetadata(
        name="swap_distance",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=2 * n_cases,
        seed_range=(
            min(int(row["seed"]) for row in per_case),
            max(int(row["seed"]) for row in per_case),
        ),
        parameters={
            "top_m_definition": "m heaviest events from greedy-selected pool where m=|oracle|",
            "swap_search": "greedy best span-improving swap with min weight-loss tie-break",
        },
    )

    save_results("swap_distance", results, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": results}


def _print_summary(results: dict) -> None:
    agg = results["aggregates"]
    ver = results["verification"]
    n_cases = results["n_cases"]

    print("Swap Distance Verification")
    print("============================")
    print(f"Cases analyzed: {n_cases}")
    print()
    print("Top-m vs Oracle:")
    print(f"  Median symmetric difference: {agg['topm_oracle_symmetric_median']:.1f} events")
    print(f"  Mean symmetric difference: {agg['topm_oracle_symmetric_mean']:.2f} events")
    print(
        "  Cases with symmetric diff == 2 (one swap): "
        f"{agg['topm_oracle_diff_eq_2']}/{n_cases}"
    )
    print(
        "  Cases with symmetric diff <= 4 (two swaps): "
        f"{agg['topm_oracle_diff_lte_4']}/{n_cases}"
    )
    print()
    print("Minimum swaps for span feasibility:")
    print(f"  1 swap sufficient: {agg['min_swaps_eq_1']}/{n_cases}")
    print(f"  2 swaps sufficient: {agg['min_swaps_eq_2']}/{n_cases}")
    print(f"  3+ swaps needed: {agg['min_swaps_ge_3']}/{n_cases}")
    print(f"  Unresolved by improving-swap search: {agg['min_swaps_unresolved']}/{n_cases}")
    print(
        "  One-swap in selected-pool-feasible cases: "
        f"{agg['one_swap_selected_pool_feasible']}/{agg['selected_pool_feasible_count']}"
    )
    print()
    print("Swapped-in event characterization:")
    print(
        "  Temporal endpoint (pool earliest/latest): "
        f"{agg['swap_in_pool_endpoint_count']}/{agg['swap_in_total_count']} swap-ins"
    )
    print(
        "  Temporal endpoint (oracle earliest/latest): "
        f"{agg['swap_in_oracle_endpoint_count']}/{agg['swap_in_total_count']} swap-ins"
    )
    print(f"  Mean weight rank of swap-in: {agg['swap_in_weight_rank_mean']:.2f}")
    print()
    print("Greedy vs Oracle:")
    print(f"  Median Jaccard distance: {agg['greedy_oracle_jaccard_median']:.3f}")
    print(f"  Mean Jaccard distance: {agg['greedy_oracle_jaccard_mean']:.3f}")
    print(f"  Mean symmetric difference: {agg['greedy_oracle_symmetric_mean']:.2f} events")
    print()
    print(
        "Checks:"
        f" one_swap_cases={ver['one_swap_cases']}/{n_cases},"
        f" endpoint_swaps={ver['endpoint_swaps']}/{n_cases},"
        f" mean_jaccard_greedy_vs_oracle={ver['jaccard_greedy_vs_oracle_mean']:.3f}"
    )


if __name__ == "__main__":
    payload = run_swap_distance()
    _print_summary(payload["results"])
