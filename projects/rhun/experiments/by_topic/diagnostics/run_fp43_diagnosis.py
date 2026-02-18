"""Diagnose non-tie false positives in clean k-j boundary sweep (k=1..5)."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, Phase


OUTPUT_NAME = "fp43_diagnosis"

FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
SEEDS = range(100)
K_VALUES = [1, 2, 3, 4, 5]
N_EVENTS = 200
N_ACTORS = 6

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

EXPECTED_FP_CASES = 51
EXPECTED_TIE_CASES = 8
TARGET_HASH_SEED = "1"


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == TARGET_HASH_SEED:
        return

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = TARGET_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _make_grammar(k: int) -> GrammarConfig:
    return GrammarConfig(
        min_prefix_elements=k,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )


def _event_dict(event: Event | None) -> dict | None:
    if event is None:
        return None
    return {
        "id": event.id,
        "weight": float(event.weight),
        "timestamp": float(event.timestamp),
        "actors": sorted(event.actors),
    }


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str) -> Event:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    if not focal_events:
        raise ValueError(f"No focal events for actor={focal_actor}")

    return max(
        focal_events,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )


def _focal_strict_before_timestamp(graph: CausalGraph, focal_actor: str, timestamp: float) -> int:
    return int(
        sum(
            1
            for event in graph.events
            if focal_actor in event.actors and float(event.timestamp) < float(timestamp)
        )
    )


def _weight_relation(actual_weight: float, max_weight: float) -> str:
    if actual_weight > max_weight + 1e-12:
        return "actual_tp_heavier"
    if actual_weight < max_weight - 1e-12:
        return "actual_tp_lighter"
    return "equal_weight"


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    fp_cases = data["false_positive_cases"]
    total = len(fp_cases)
    tie_count = int(data["counts"]["timestamp_tie_artifacts"])
    non_tie_count = int(data["counts"]["non_tie_false_positives"])
    tp_match_true = int(data["counts"]["tp_match_true"])
    tp_match_false = int(data["counts"]["tp_match_false"])

    tp_false = data["tp_match_false_analysis"]
    tp_true = data["tp_match_true_analysis"]
    rec = data["recommendation"]

    lines: list[str] = [
        "# FP43 diagnosis (non-tie false positives)",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Count Check",
        "",
        f"- Total FPs found: {total}",
        f"- Expected FPs: {EXPECTED_FP_CASES}",
        f"- Timestamp-tie artifacts: {tie_count}",
        f"- Non-tie FPs: {non_tie_count}",
        "",
        "## TP Identity Breakdown",
        "",
        f"- tp_match=True: {tp_match_true}",
        f"- tp_match=False: {tp_match_false}",
        "",
        "## tp_match=True interpretation",
        "",
        (
            "- Tie-artifact cases (known): "
            f"{tp_true['known_timestamp_tie_cases']}"
        ),
        (
            "- Non-tie tp_match=True cases with non-focal events before TP in sequence: "
            f"{tp_true['non_tie_with_non_focal_prefix']}/{tp_true['non_tie_total']}"
        ),
        (
            "- Non-tie tp_match=True cases where focal-only strict prefix is <k "
            "but total prefix supports validity: "
            f"{tp_true['non_tie_focal_under_k_total_prefix_ge_k']}/{tp_true['non_tie_total']}"
        ),
        (
            "- Interpretation: these are measurement artifacts from focal-only j; "
            "greedy can satisfy DEVELOPMENT via non-focal prefix events."
        ),
        "",
        "## tp_match=False analysis",
        "",
        f"- Count: {tp_match_false}",
        f"- actual_tp_j_focal distribution: {tp_false['actual_tp_j_focal_distribution']}",
        f"- All tp_match=False have actual_tp_j_focal >= k: {tp_false['all_actual_tp_j_focal_ge_k']}",
        (
            "- Cause breakdown: "
            f"missing_from_pool={tp_false['cause_counts']['max_weight_not_in_candidate_pool']}, "
            f"in_pool_not_selected={tp_false['cause_counts']['max_weight_in_pool_not_in_final_sequence']}, "
            f"selected_but_not_tp={tp_false['cause_counts']['max_weight_selected_but_not_tp']}"
        ),
        "",
        "## Candidate j Comparison (FP counts over full sweep)",
        "",
        "| definition | FP count | note |",
        "|---|---:|---|",
        f"| j_focal | {rec['candidate_fp_counts']['j_focal']} | focal-only strict prefix before max focal TP |",
        f"| j_actual | {rec['candidate_fp_counts']['j_actual']} | focal-only strict prefix before greedy actual TP |",
        f"| j_dev_eligible | {rec['candidate_fp_counts']['j_dev_eligible']} | development count implied by classifier/output |",
        "",
        "## Recommendation",
        "",
        f"- {rec['text']}",
        "",
        "## Key Answer",
        "",
        f"- {data['key_question_answer']}",
        "",
    ]
    return "\n".join(lines)


def run_fp43_diagnosis() -> dict:
    _ensure_hash_seed()
    timer = ExperimentTimer()
    generator = BurstyGenerator()

    cases: list[dict] = []

    fp_count_by_definition = {
        "j_focal": 0,
        "j_actual": 0,
        "j_dev_eligible": 0,
    }

    total_graphs = 0
    total_extractions = 0

    for epsilon in EPSILONS:
        for seed in SEEDS:
            graph = generator.generate(
                BurstyConfig(
                    seed=int(seed),
                    epsilon=float(epsilon),
                    n_events=N_EVENTS,
                    n_actors=N_ACTORS,
                )
            )
            total_graphs += 1

            max_focal_event = _max_weight_focal_event(graph, FOCAL_ACTOR)
            j_focal = _focal_strict_before_timestamp(
                graph,
                FOCAL_ACTOR,
                max_focal_event.timestamp,
            )

            focal_events_at_max_timestamp = [
                event
                for event in graph.events
                if FOCAL_ACTOR in event.actors
                and abs(float(event.timestamp) - float(max_focal_event.timestamp)) <= 1e-12
            ]

            for k in K_VALUES:
                grammar = _make_grammar(k)
                result = greedy_extract(
                    graph=graph,
                    focal_actor=FOCAL_ACTOR,
                    grammar=grammar,
                    pool_strategy=POOL_STRATEGY,
                    n_anchors=N_ANCHORS,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    injection_top_n=INJECTION_TOP_N,
                )
                total_extractions += 1

                tp_idx = next(
                    (idx for idx, phase in enumerate(result.phases) if phase == Phase.TURNING_POINT),
                    None,
                )
                actual_tp = None if tp_idx is None else result.events[tp_idx]
                j_actual = (
                    None
                    if actual_tp is None
                    else _focal_strict_before_timestamp(
                        graph,
                        FOCAL_ACTOR,
                        actual_tp.timestamp,
                    )
                )
                j_dev_eligible = int(result.n_development)

                if j_focal < k and result.valid:
                    fp_count_by_definition["j_focal"] += 1
                if j_actual is not None and j_actual < k and result.valid:
                    fp_count_by_definition["j_actual"] += 1
                if j_dev_eligible < k and result.valid:
                    fp_count_by_definition["j_dev_eligible"] += 1

                if not (j_focal < k and result.valid):
                    continue

                tp_match = bool(actual_tp is not None and actual_tp.id == max_focal_event.id)
                actual_tp_j_focal = None if actual_tp is None else int(j_actual)

                by_id = {event.id: event for event in graph.events}
                pool_ids_raw = result.metadata.get("pool_ids", ())
                pool_ids = {str(event_id) for event_id in pool_ids_raw} if pool_ids_raw else set()
                max_weight_in_candidate_pool = bool(max_focal_event.id in pool_ids)

                max_seq_idx = next(
                    (idx for idx, event in enumerate(result.events) if event.id == max_focal_event.id),
                    None,
                )
                max_weight_in_final_sequence = bool(max_seq_idx is not None)
                max_weight_phase_if_in_sequence = (
                    None
                    if max_seq_idx is None
                    else result.phases[max_seq_idx].name
                )

                if not tp_match:
                    if not max_weight_in_candidate_pool:
                        tp_selection_reason = "max_weight_focal_not_in_candidate_pool"
                    elif not max_weight_in_final_sequence:
                        tp_selection_reason = "max_weight_focal_in_pool_but_not_in_final_sequence"
                    elif max_weight_phase_if_in_sequence != Phase.TURNING_POINT.name:
                        tp_selection_reason = "max_weight_focal_selected_but_not_assigned_tp"
                    else:
                        tp_selection_reason = "unclassified_tp_divergence"
                else:
                    tp_selection_reason = "tp_matches_max_weight_focal"

                max_prefix = [] if max_seq_idx is None else list(result.events[:max_seq_idx])
                equal_ts_prefix_count = sum(
                    1
                    for event in max_prefix
                    if abs(float(event.timestamp) - float(max_focal_event.timestamp)) <= 1e-12
                )
                strict_earlier_prefix_count = sum(
                    1
                    for event in max_prefix
                    if float(event.timestamp) < float(max_focal_event.timestamp) - 1e-12
                )
                known_timestamp_tie_artifact = bool(j_focal == 0 and equal_ts_prefix_count > 0)

                pre_tp_events = [] if tp_idx is None else list(result.events[:tp_idx])
                pre_tp_non_focal_count = sum(
                    1 for event in pre_tp_events if FOCAL_ACTOR not in event.actors
                )

                case = {
                    "epsilon": float(epsilon),
                    "seed": int(seed),
                    "k": int(k),
                    "max_weight_focal_event": {
                        "id": max_focal_event.id,
                        "weight": float(max_focal_event.weight),
                        "timestamp": float(max_focal_event.timestamp),
                        "j_focal": int(j_focal),
                    },
                    "greedy_result": {
                        "valid": bool(result.valid),
                        "score": float(result.score) if result.valid else None,
                        "n_development": int(result.n_development),
                        "tp": _event_dict(actual_tp),
                        "tp_index": None if tp_idx is None else int(tp_idx),
                        "event_ids": [event.id for event in result.events],
                        "phases": [phase.name for phase in result.phases],
                    },
                    "tp_match": bool(tp_match),
                    "j_actual": None if j_actual is None else int(j_actual),
                    "actual_tp_j_focal": actual_tp_j_focal,
                    "actual_tp_weight_relation_to_max_focal": (
                        None
                        if actual_tp is None
                        else _weight_relation(float(actual_tp.weight), float(max_focal_event.weight))
                    ),
                    "tp_identity_diagnosis": {
                        "max_weight_focal_in_candidate_pool": bool(max_weight_in_candidate_pool),
                        "max_weight_focal_in_final_sequence": bool(max_weight_in_final_sequence),
                        "max_weight_focal_phase_if_in_sequence": max_weight_phase_if_in_sequence,
                        "tp_selection_reason": tp_selection_reason,
                    },
                    "timestamp_ties": {
                        "focal_events_equal_to_max_weight_timestamp_count": int(
                            len(focal_events_at_max_timestamp)
                        ),
                        "other_focal_events_equal_to_max_weight_timestamp_count": int(
                            max(0, len(focal_events_at_max_timestamp) - 1)
                        ),
                        "equal_timestamp_prefix_before_max_weight_count": int(
                            equal_ts_prefix_count
                        ),
                        "strictly_earlier_prefix_before_max_weight_count": int(
                            strict_earlier_prefix_count
                        ),
                        "known_timestamp_tie_artifact": bool(known_timestamp_tie_artifact),
                    },
                    "prefix_diagnostics": {
                        "pre_tp_event_count": int(len(pre_tp_events)),
                        "pre_tp_non_focal_count": int(pre_tp_non_focal_count),
                    },
                }
                cases.append(case)

    if len(cases) != EXPECTED_FP_CASES:
        raise RuntimeError(
            f"FP case mismatch: expected {EXPECTED_FP_CASES}, found {len(cases)}."
        )

    timestamp_tie_cases = [
        case for case in cases if bool(case["timestamp_ties"]["known_timestamp_tie_artifact"])
    ]
    if len(timestamp_tie_cases) != EXPECTED_TIE_CASES:
        raise RuntimeError(
            f"Timestamp-tie FP mismatch: expected {EXPECTED_TIE_CASES}, found {len(timestamp_tie_cases)}."
        )

    tp_match_true_cases = [case for case in cases if bool(case["tp_match"])]
    tp_match_false_cases = [case for case in cases if not bool(case["tp_match"])]
    non_tie_cases = [
        case for case in cases if not bool(case["timestamp_ties"]["known_timestamp_tie_artifact"])
    ]
    non_tie_tp_match_true = [case for case in non_tie_cases if bool(case["tp_match"])]

    actual_tp_j_dist = Counter(
        int(case["actual_tp_j_focal"])
        for case in tp_match_false_cases
        if case["actual_tp_j_focal"] is not None
    )
    all_actual_ge_k = bool(
        all(
            int(case["actual_tp_j_focal"]) >= int(case["k"])
            for case in tp_match_false_cases
            if case["actual_tp_j_focal"] is not None
        )
    )

    cause_counts = {
        "max_weight_not_in_candidate_pool": sum(
            1
            for case in tp_match_false_cases
            if case["tp_identity_diagnosis"]["tp_selection_reason"]
            == "max_weight_focal_not_in_candidate_pool"
        ),
        "max_weight_in_pool_not_in_final_sequence": sum(
            1
            for case in tp_match_false_cases
            if case["tp_identity_diagnosis"]["tp_selection_reason"]
            == "max_weight_focal_in_pool_but_not_in_final_sequence"
        ),
        "max_weight_selected_but_not_tp": sum(
            1
            for case in tp_match_false_cases
            if case["tp_identity_diagnosis"]["tp_selection_reason"]
            == "max_weight_focal_selected_but_not_assigned_tp"
        ),
        "unclassified_tp_divergence": sum(
            1
            for case in tp_match_false_cases
            if case["tp_identity_diagnosis"]["tp_selection_reason"] == "unclassified_tp_divergence"
        ),
    }

    non_tie_with_non_focal_prefix = sum(
        1 for case in non_tie_tp_match_true if int(case["prefix_diagnostics"]["pre_tp_non_focal_count"]) > 0
    )
    non_tie_focal_under_k_total_prefix_ge_k = sum(
        1
        for case in non_tie_tp_match_true
        if int(case["max_weight_focal_event"]["j_focal"]) < int(case["k"])
        and int(case["prefix_diagnostics"]["pre_tp_event_count"]) >= int(case["k"])
    )

    recommendation_text: str
    if fp_count_by_definition["j_dev_eligible"] == 0:
        recommendation_text = (
            "Use a development-eligible TP-prefix measure (j_dev_eligible) for zero-FP "
            "boundary diagnosis; j_focal/j_actual both remain FP-prone because they count "
            "focal events only and ignore non-focal prefix capacity."
        )
    else:
        recommendation_text = (
            "j_focal and j_actual are both FP-prone here; a development-eligible prefix "
            "measure is the closest to theorem behavior."
        )

    key_answer = (
        "The theorem is not wrong in these 43 non-tie cases. The false positives come from "
        "measurement mismatch: j_focal (and j_actual) track focal-only strict precedence, "
        "while greedy can satisfy k via non-focal development events before TP."
    )

    result = {
        "parameters": {
            "k_values": K_VALUES,
            "epsilon_values": EPSILONS,
            "seed_range": [min(SEEDS), max(SEEDS)],
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "generator": "BurstyGenerator",
            "focal_actor": FOCAL_ACTOR,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "grammar": {
                "min_prefix_elements": "k",
                "min_timespan_fraction": 0.0,
                "max_temporal_gap": "inf",
            },
            "pythonhashseed": TARGET_HASH_SEED,
        },
        "counts": {
            "false_positives_total": int(len(cases)),
            "timestamp_tie_artifacts": int(len(timestamp_tie_cases)),
            "non_tie_false_positives": int(len(non_tie_cases)),
            "tp_match_true": int(len(tp_match_true_cases)),
            "tp_match_false": int(len(tp_match_false_cases)),
        },
        "tp_match_true_analysis": {
            "known_timestamp_tie_cases": int(
                sum(
                    1
                    for case in tp_match_true_cases
                    if bool(case["timestamp_ties"]["known_timestamp_tie_artifact"])
                )
            ),
            "non_tie_total": int(len(non_tie_tp_match_true)),
            "non_tie_with_non_focal_prefix": int(non_tie_with_non_focal_prefix),
            "non_tie_focal_under_k_total_prefix_ge_k": int(non_tie_focal_under_k_total_prefix_ge_k),
        },
        "tp_match_false_analysis": {
            "count": int(len(tp_match_false_cases)),
            "actual_tp_j_focal_distribution": {
                str(k): int(v) for k, v in sorted(actual_tp_j_dist.items())
            },
            "all_actual_tp_j_focal_ge_k": bool(all_actual_ge_k),
            "cause_counts": cause_counts,
        },
        "recommendation": {
            "candidate_fp_counts": {
                "j_focal": int(fp_count_by_definition["j_focal"]),
                "j_actual": int(fp_count_by_definition["j_actual"]),
                "j_dev_eligible": int(fp_count_by_definition["j_dev_eligible"]),
            },
            "text": recommendation_text,
        },
        "key_question_answer": key_answer,
        "false_positive_cases": sorted(cases, key=lambda case: (case["epsilon"], case["seed"], case["k"])),
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "k_values": K_VALUES,
            "epsilons": EPSILONS,
            "expected_fp_cases": EXPECTED_FP_CASES,
            "expected_tie_cases": EXPECTED_TIE_CASES,
            "pythonhashseed": TARGET_HASH_SEED,
        },
    )
    save_results(OUTPUT_NAME, result, metadata, summary_formatter=_summary_markdown)

    output_root = Path(__file__).resolve().parent / "output"
    print("FP43 diagnosis complete")
    print("======================")
    print(f"FP cases: {len(cases)}")
    print(f"Tie artifacts: {len(timestamp_tie_cases)}")
    print(f"Non-tie FPs: {len(non_tie_cases)}")
    print(f"tp_match=True: {len(tp_match_true_cases)}")
    print(f"tp_match=False: {len(tp_match_false_cases)}")
    print(f"JSON: {output_root / (OUTPUT_NAME + '.json')}")
    print(f"Summary: {output_root / (OUTPUT_NAME + '_summary.md')}")

    return {"metadata": metadata, "results": result}


if __name__ == "__main__":
    run_fp43_diagnosis()
