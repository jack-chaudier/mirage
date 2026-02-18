"""Diagnose the 8 false-positive cases from the k-j boundary sweep (k=1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, Phase
from rhun.theory.theorem import check_precondition


OUTPUT_NAME = "fp8_diagnosis"
FOCAL_ACTOR = "actor_0"
EPSILONS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
SEEDS = range(100)
N_EVENTS = 200
N_ACTORS = 6
K_VALUE = 1

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

EXPECTED_FP_CASES = 8
TARGET_HASH_SEED = "144"


def _ensure_hash_seed() -> None:
    """
    Re-exec with a fixed hash seed so set-iteration tie-breaks are reproducible.

    Experiment 28 metrics were produced under one deterministic interpreter hash
    seed. Without pinning, this edge-case count drifts due to tie order in pool
    traversal.
    """
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


def _event_to_dict(event: Event) -> dict:
    return {
        "event_id": event.id,
        "weight": float(event.weight),
        "timestamp": float(event.timestamp),
        "actors": sorted(event.actors),
    }


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str) -> Event:
    focal_events = graph.events_for_actor(focal_actor)
    if not focal_events:
        raise ValueError(f"No focal events for actor={focal_actor}")
    return max(focal_events, key=lambda event: (float(event.weight), -float(event.timestamp)))


def _focal_index(graph: CausalGraph, focal_actor: str, event_id: str) -> int | None:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    focal_events_sorted = sorted(
        focal_events,
        key=lambda event: (float(event.timestamp), event.id),
    )
    for idx, event in enumerate(focal_events_sorted):
        if event.id == event_id:
            return int(idx)
    return None


def _phase_name(phase: Phase | None) -> str | None:
    if phase is None:
        return None
    return phase.name


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    cases = data["cases"]
    n_cases = len(cases)
    j_focal_zero = sum(1 for case in cases if int(case["max_weight_focal_event"]["j_focal"]) == 0)
    theorem_max_matches = sum(
        1
        for case in cases
        if bool(case["max_weight_focal_event"]["matches_theorem_max_weight_event"])
    )
    assigned_tp = sum(
        1
        for case in cases
        if bool(case["phase_classifier_for_max_weight_event"]["assigned_as_turning_point"])
    )
    classifier_overrode = sum(
        1
        for case in cases
        if bool(case["phase_classifier_for_max_weight_event"]["classifier_overrode_turning_point"])
    )
    heavier_non_focal_selected = sum(
        1
        for case in cases
        if bool(case["heavier_non_focal_injection"]["selected_sequence_contains_heavier_non_focal"])
    )
    heavier_non_focal_pool = sum(
        1
        for case in cases
        if bool(case["heavier_non_focal_injection"]["pool_contains_heavier_non_focal"])
    )
    tie_prefix_cases = sum(
        1
        for case in cases
        if int(case["timestamp_tie_diagnostic"]["equal_timestamp_prefix_count"]) > 0
    )
    strict_earlier_prefix_cases = sum(
        1
        for case in cases
        if int(case["timestamp_tie_diagnostic"]["strictly_earlier_prefix_count"]) > 0
    )

    if (
        theorem_max_matches == n_cases
        and j_focal_zero == n_cases
        and assigned_tp == n_cases
        and strict_earlier_prefix_cases == 0
        and tie_prefix_cases == n_cases
    ):
        conclusion = (
            "The theorem is not wrong here. `j_theorem=0` is a timestamp-tie measurement artifact: "
            "the TP event is first in strict graph order, but equal-timestamp events can still appear "
            "before it in the extracted sequence and satisfy k=1."
        )
    elif assigned_tp == 0 and theorem_max_matches == n_cases and j_focal_zero == n_cases:
        conclusion = (
            "These are measurement/assumption mismatches, not theorem failures: "
            "the max-weight focal event is truly j_focal=0 but is not the sequence TP."
        )
    else:
        conclusion = (
            "At least one case behaves like a potential theorem counterexample; "
            "see per-case details."
        )

    lines: list[str] = [
        "# FP8 diagnosis (k=1, j_theorem=0)",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Count Check",
        "",
        f"- Expected FP cases: {EXPECTED_FP_CASES}",
        f"- Found FP cases: {n_cases}",
        "",
        "## Findings",
        "",
        f"- max-weight focal event matches theorem event: {theorem_max_matches}/{n_cases}",
        f"- max-weight focal event has `j_focal=0`: {j_focal_zero}/{n_cases}",
        f"- max-weight focal event assigned TP by classifier: {assigned_tp}/{n_cases}",
        f"- classifier override (not TP or absent from sequence): {classifier_overrode}/{n_cases}",
        (
            f"- heavier non-focal event present in chosen pool: "
            f"{heavier_non_focal_pool}/{n_cases}"
        ),
        (
            f"- heavier non-focal event selected into final sequence: "
            f"{heavier_non_focal_selected}/{n_cases}"
        ),
        f"- cases with equal-timestamp prefix before TP: {tie_prefix_cases}/{n_cases}",
        f"- cases with strictly-earlier prefix before TP: {strict_earlier_prefix_cases}/{n_cases}",
        "",
        "## Conclusion",
        "",
        f"- {conclusion}",
        "",
        "## Cases",
        "",
        "| epsilon | seed | focal_event | j_theorem | j_focal | focal_as_tp | sequence_tp |",
        "|---|---:|---|---:|---:|---:|---|",
    ]

    for case in cases:
        focal = case["max_weight_focal_event"]
        assignment = case["phase_classifier_for_max_weight_event"]
        lines.append(
            f"| {case['epsilon']:.2f} | {case['seed']} | {focal['event_id']} | "
            f"{focal['j_theorem']} | {focal['j_focal']} | "
            f"{int(bool(assignment['assigned_as_turning_point']))} | "
            f"{assignment['sequence_turning_point_event_id']} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_fp8_diagnosis() -> dict:
    _ensure_hash_seed()
    timer = ExperimentTimer()
    grammar = _make_grammar(K_VALUE)
    generator = BurstyGenerator()

    total_graphs = 0
    total_extractions = 0
    cases: list[dict] = []

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

            precondition = check_precondition(
                graph=graph,
                focal_actor=FOCAL_ACTOR,
                grammar=grammar,
            )
            j_theorem = int(precondition["events_before_max"])

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

            if not (j_theorem == 0 and bool(result.valid)):
                continue

            theorem_max_idx = int(precondition["max_weight_index"])
            theorem_max_event = graph.events[theorem_max_idx]
            max_focal_event = _max_weight_focal_event(graph, FOCAL_ACTOR)
            j_focal = _focal_index(graph, FOCAL_ACTOR, theorem_max_event.id)

            seq_event_ids = [event.id for event in result.events]
            seq_phase_names = [phase.name for phase in result.phases]
            sequence_tp = result.turning_point
            sequence_tp_id = None if sequence_tp is None else sequence_tp.id

            max_event_seq_idx = next(
                (idx for idx, event in enumerate(result.events) if event.id == theorem_max_event.id),
                None,
            )
            max_event_phase = (
                None if max_event_seq_idx is None else result.phases[max_event_seq_idx]
            )
            assigned_as_tp = bool(max_event_phase == Phase.TURNING_POINT)
            classifier_override_reason: str | None
            if max_event_seq_idx is None:
                classifier_override_reason = "max_weight_focal_event_not_selected"
            elif not assigned_as_tp:
                classifier_override_reason = "max_weight_focal_event_selected_but_not_tp"
            else:
                classifier_override_reason = None

            prefix_before_max = (
                []
                if max_event_seq_idx is None
                else list(result.events[:max_event_seq_idx])
            )
            equal_timestamp_prefix_count = sum(
                1
                for event in prefix_before_max
                if abs(float(event.timestamp) - float(theorem_max_event.timestamp)) <= 1e-12
            )
            strictly_earlier_prefix_count = sum(
                1
                for event in prefix_before_max
                if float(event.timestamp) < float(theorem_max_event.timestamp) - 1e-12
            )

            by_id = {event.id: event for event in graph.events}
            pool_ids_raw = result.metadata.get("pool_ids", ())
            pool_ids = {str(event_id) for event_id in pool_ids_raw} if pool_ids_raw else set()
            pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
            heavier_non_focal_in_pool = sorted(
                (
                    event
                    for event in pool_events
                    if FOCAL_ACTOR not in event.actors and float(event.weight) > float(theorem_max_event.weight)
                ),
                key=lambda event: (float(event.weight), -float(event.timestamp), event.id),
                reverse=True,
            )
            selected_ids = set(seq_event_ids)
            heavier_non_focal_selected = [
                event for event in heavier_non_focal_in_pool if event.id in selected_ids
            ]

            top_heavier_pool = None
            if heavier_non_focal_in_pool:
                top_heavier_pool = _event_to_dict(heavier_non_focal_in_pool[0])

            top_heavier_selected = None
            if heavier_non_focal_selected:
                top_heavier_selected = _event_to_dict(heavier_non_focal_selected[0])

            case = {
                "epsilon": float(epsilon),
                "seed": int(seed),
                "k": int(K_VALUE),
                "max_weight_focal_event": {
                    "event_id": theorem_max_event.id,
                    "weight": float(theorem_max_event.weight),
                    "timestamp": float(theorem_max_event.timestamp),
                    "temporal_index_full_graph": int(theorem_max_idx),
                    "j_theorem": int(j_theorem),
                    "j_focal": None if j_focal is None else int(j_focal),
                    "matches_theorem_max_weight_event": bool(theorem_max_event.id == max_focal_event.id),
                    "recomputed_focal_max_event_id": max_focal_event.id,
                },
                "greedy": {
                    "valid": bool(result.valid),
                    "score": float(result.score) if result.valid else None,
                    "n_development": int(result.n_development),
                    "event_ids": seq_event_ids,
                    "phases": seq_phase_names,
                    "sequence": [
                        {"event_id": event.id, "phase": phase.name}
                        for event, phase in zip(result.events, result.phases, strict=True)
                    ],
                },
                "phase_classifier_for_max_weight_event": {
                    "event_id": theorem_max_event.id,
                    "in_sequence": bool(max_event_seq_idx is not None),
                    "sequence_index": None if max_event_seq_idx is None else int(max_event_seq_idx),
                    "assigned_phase": _phase_name(max_event_phase),
                    "assigned_as_turning_point": bool(assigned_as_tp),
                    "classifier_overrode_turning_point": bool(not assigned_as_tp),
                    "override_reason": classifier_override_reason,
                    "sequence_turning_point_event_id": sequence_tp_id,
                    "sequence_turning_point_weight": None
                    if sequence_tp is None
                    else float(sequence_tp.weight),
                    "sequence_turning_point_is_focal": None
                    if sequence_tp is None
                    else bool(FOCAL_ACTOR in sequence_tp.actors),
                },
                "heavier_non_focal_injection": {
                    "pool_contains_heavier_non_focal": bool(heavier_non_focal_in_pool),
                    "heaviest_non_focal_in_pool": top_heavier_pool,
                    "selected_sequence_contains_heavier_non_focal": bool(heavier_non_focal_selected),
                    "heaviest_non_focal_selected": top_heavier_selected,
                },
                "timestamp_tie_diagnostic": {
                    "max_weight_event_timestamp": float(theorem_max_event.timestamp),
                    "prefix_event_ids_before_max_in_sequence": [event.id for event in prefix_before_max],
                    "prefix_event_timestamps_before_max_in_sequence": [
                        float(event.timestamp) for event in prefix_before_max
                    ],
                    "equal_timestamp_prefix_count": int(equal_timestamp_prefix_count),
                    "strictly_earlier_prefix_count": int(strictly_earlier_prefix_count),
                },
            }
            cases.append(case)

    if len(cases) != EXPECTED_FP_CASES:
        raise RuntimeError(
            "FP count mismatch: expected "
            f"{EXPECTED_FP_CASES}, found {len(cases)}. "
            "Check hash seed + sweep parameters."
        )

    result = {
        "parameters": {
            "epsilons": EPSILONS,
            "seed_range": [min(SEEDS), max(SEEDS)],
            "k": int(K_VALUE),
            "n_events": int(N_EVENTS),
            "n_actors": int(N_ACTORS),
            "focal_actor": FOCAL_ACTOR,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": int(N_ANCHORS),
            "max_sequence_length": int(MAX_SEQUENCE_LENGTH),
            "injection_top_n": int(INJECTION_TOP_N),
            "constraints_disabled": ["min_timespan_fraction", "max_temporal_gap"],
            "pythonhashseed": TARGET_HASH_SEED,
        },
        "verification": {
            "expected_fp_case_count": EXPECTED_FP_CASES,
            "actual_fp_case_count": len(cases),
        },
        "cases": sorted(cases, key=lambda row: (row["epsilon"], row["seed"])),
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_graphs,
        n_extractions=total_extractions,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "k": K_VALUE,
            "epsilons": EPSILONS,
            "pythonhashseed": TARGET_HASH_SEED,
        },
    )
    save_results(OUTPUT_NAME, result, metadata, summary_formatter=_summary_markdown)

    output_root = Path(__file__).resolve().parent / "output"
    print("FP8 diagnosis complete")
    print("=====================")
    print(f"Cases found: {len(cases)} (expected {EXPECTED_FP_CASES})")
    print(f"Hash seed pinned: {TARGET_HASH_SEED}")
    print(f"JSON: {output_root / (OUTPUT_NAME + '.json')}")
    print(f"Summary: {output_root / (OUTPUT_NAME + '_summary.md')}")

    return {"metadata": metadata, "results": result}


if __name__ == "__main__":
    run_fp8_diagnosis()
