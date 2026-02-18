"""Experiments 50-57: Context algebra validation suite."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.tp_conditioned_solver import tp_conditioned_solve
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.generators.recursive_burst import RecursiveBurstConfig, RecursiveBurstGenerator
from rhun.schemas import CausalGraph, Event, Phase
from rhun.theory.context_algebra import (
    ContextState,
    bridge_budget_lower_bound,
    build_context_state,
    compress_events,
    compose_context_states,
    context_equivalent,
    detect_class_a,
    detect_class_b,
    detect_class_c,
    detect_class_d,
    development_eligible_count,
    induced_subgraph,
    is_absorbing,
    no_absorption_invariant,
)
from rhun.theory.theorem import check_precondition


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FOCAL_ACTOR = "actor_0"
TARGET_HASH_SEED = "1"


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    exp50_case_limit: int
    exp51_n_sequences: int
    exp52_n_sequences: int
    exp53_n_sequences: int
    exp54_n_sequences: int
    exp55_seeds_per_cell: int
    exp56_max_states: int
    exp56_l_max: int
    exp57_n_samples: int


SMOKE_SCALE = ScaleConfig(
    name="smoke",
    exp50_case_limit=138,
    exp51_n_sequences=80,
    exp52_n_sequences=40,
    exp53_n_sequences=30,
    exp54_n_sequences=36,
    exp55_seeds_per_cell=1,
    exp56_max_states=16,
    exp56_l_max=6,
    exp57_n_samples=30,
)

FULL_SCALE = ScaleConfig(
    name="full",
    exp50_case_limit=138,
    exp51_n_sequences=500,
    exp52_n_sequences=1000,
    exp53_n_sequences=300,
    exp54_n_sequences=300,
    exp55_seeds_per_cell=5,
    exp56_max_states=60,
    exp56_l_max=16,
    exp57_n_samples=200,
)


def _is_insufficient_development(violations: tuple[str, ...] | list[str]) -> bool:
    return any(str(v).startswith("insufficient_development") for v in violations)


def _is_insufficient_timespan(violations: tuple[str, ...] | list[str]) -> bool:
    return any(str(v).startswith("insufficient_timespan") for v in violations)


def _turning_point_id(sequence) -> str | None:
    tp = sequence.turning_point
    if tp is None:
        return None
    return str(tp.id)


def _tp_prefix_count(sequence) -> int:
    tp_idx = next((idx for idx, phase in enumerate(sequence.phases) if phase == Phase.TURNING_POINT), None)
    if tp_idx is None:
        return 0
    return int(tp_idx)


def _safe_rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom > 0 else 0.0


def _metric_counts(truth: list[bool], pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(truth, pred, strict=True) if t and p)
    tn = sum(1 for t, p in zip(truth, pred, strict=True) if (not t) and (not p))
    fp = sum(1 for t, p in zip(truth, pred, strict=True) if (not t) and p)
    fn = sum(1 for t, p in zip(truth, pred, strict=True) if t and (not p))
    n = len(truth)
    return {
        "n": int(n),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": _safe_rate(tp + tn, n),
        "precision": _safe_rate(tp, tp + fp),
        "recall": _safe_rate(tp, tp + fn),
    }


def _load_exp50_baseline(limit: int) -> list[dict]:
    """
    Load the established 138-case baseline if available.

    Falls back to deterministic regeneration if the file is absent.
    """
    path = OUTPUT_DIR / "fn_divergence_analysis.json"
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        rows = blob.get("results", {}).get("per_case", [])
        baseline = [
            {
                "seed": int(row["seed"]),
                "epsilon": float(row["epsilon"]),
                "source": "fn_divergence_analysis",
            }
            for row in rows
        ]
        baseline.sort(key=lambda row: (row["epsilon"], row["seed"]))
        return baseline[:limit]

    # Fallback path: regenerate from a fixed grid.
    baseline: list[dict] = []
    epsilons = [0.80, 0.85, 0.90, 0.95]
    for epsilon in epsilons:
        for seed in range(200):
            baseline.append({"seed": seed, "epsilon": epsilon, "source": "regenerated"})
            if len(baseline) >= limit:
                return baseline
    return baseline[:limit]


def run_exp50_context_state_formalization(scale: ScaleConfig) -> tuple[dict, list[dict]]:
    """
    Experiment 50:
    Validate context-state sufficiency for class diagnosis.
    """
    generator = BurstyGenerator()
    grammar = GrammarConfig(min_prefix_elements=1)
    M = 25

    baseline = _load_exp50_baseline(limit=scale.exp50_case_limit)
    case_rows: list[dict] = []
    exp56_seed_rows: list[dict] = []

    truth_a: list[bool] = []
    pred_a: list[bool] = []
    truth_b: list[bool] = []
    pred_b: list[bool] = []
    truth_c: list[bool] = []
    pred_c: list[bool] = []
    truth_d: list[bool] = []
    pred_d: list[bool] = []
    truth_a_fail: list[bool] = []
    pred_a_fail: list[bool] = []
    truth_b_fail: list[bool] = []
    pred_b_fail: list[bool] = []
    truth_c_fail: list[bool] = []
    pred_c_fail: list[bool] = []
    truth_d_fail: list[bool] = []
    pred_d_fail: list[bool] = []

    for row in baseline:
        epsilon = float(row["epsilon"])
        seed = int(row["seed"])
        graph = generator.generate(
            BurstyConfig(
                n_events=200,
                n_actors=6,
                seed=seed,
                epsilon=epsilon,
            )
        )
        greedy = greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy="injection",
            n_anchors=8,
            max_sequence_length=20,
            injection_top_n=40,
        )
        oracle, _oracle_diag = oracle_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            max_sequence_length=20,
        )
        theorem = check_precondition(graph, FOCAL_ACTOR, grammar)

        state = build_context_state(greedy.events, focal_actor=FOCAL_ACTOR, M=M)

        tp_prefix = _tp_prefix_count(greedy)
        observed_d_pre = development_eligible_count(tp_prefix)

        ordered_seq_events = tuple(sorted(greedy.events, key=lambda event: (float(event.timestamp), event.id)))
        focal_seq_events = [event for event in ordered_seq_events if FOCAL_ACTOR in event.actors]
        if focal_seq_events:
            focal_pivot = max(
                focal_seq_events,
                key=lambda event: (float(event.weight), -float(event.timestamp), event.id),
            )
            focal_prefix_count = next(
                idx for idx, event in enumerate(ordered_seq_events) if event.id == focal_pivot.id
            )
            focal_d_pre = development_eligible_count(focal_prefix_count)
        else:
            focal_pivot = None
            focal_prefix_count = 0
            focal_d_pre = 0

        class_a_truth = bool((not greedy.valid) and int(focal_d_pre) < int(grammar.min_prefix_elements))
        class_b_truth = bool(
            (not greedy.valid)
            and _is_insufficient_development(greedy.violations)
            and int(tp_prefix) >= int(grammar.min_prefix_elements)
            and int(greedy.n_development) < int(grammar.min_prefix_elements)
        )
        class_c_truth = bool(
            (not greedy.valid)
            and (oracle is not None and oracle.valid)
            and (_turning_point_id(greedy) != _turning_point_id(oracle))
        )
        class_d_truth = bool(
            (not greedy.valid)
            and _is_insufficient_timespan(greedy.violations)
            and (oracle is not None and oracle.valid)
        )

        class_a_pred = detect_class_a(state, k=grammar.min_prefix_elements)
        class_b_pred = detect_class_b(state, k=grammar.min_prefix_elements)
        class_c_pred = detect_class_c(state, k=grammar.min_prefix_elements)
        class_d_pred = detect_class_d(
            state=state,
            min_timespan_fraction=float(grammar.min_timespan_fraction),
            graph_duration=float(graph.duration),
            min_length=int(grammar.min_length),
        )

        truth_a.append(class_a_truth)
        pred_a.append(class_a_pred)
        truth_b.append(class_b_truth)
        pred_b.append(class_b_pred)
        truth_c.append(class_c_truth)
        pred_c.append(class_c_pred)
        truth_d.append(class_d_truth)
        pred_d.append(class_d_pred)
        if not greedy.valid:
            truth_a_fail.append(class_a_truth)
            pred_a_fail.append(class_a_pred)
            truth_b_fail.append(class_b_truth)
            pred_b_fail.append(class_b_pred)
            truth_c_fail.append(class_c_truth)
            pred_c_fail.append(class_c_pred)
            truth_d_fail.append(class_d_truth)
            pred_d_fail.append(class_d_pred)

        case_payload = {
            "seed": seed,
            "epsilon": epsilon,
            "greedy_valid": bool(greedy.valid),
            "greedy_violations": list(greedy.violations),
            "greedy_n_development": int(greedy.n_development),
            "observed_d_pre": int(observed_d_pre),
            "focal_pivot_id": None if focal_pivot is None else str(focal_pivot.id),
            "focal_prefix_count": int(focal_prefix_count),
            "focal_d_pre": int(focal_d_pre),
            "state_q": str(state.q),
            "state_r": asdict(state.r),
            "state_top_pivots": [asdict(candidate) for candidate in state.pivots[:5]],
            "class_truth": {
                "A_absorbing": class_a_truth,
                "B_pipeline_coupling": class_b_truth,
                "C_commitment_timing": class_c_truth,
                "D_assembly_compression": class_d_truth,
            },
            "class_pred": {
                "A_absorbing": class_a_pred,
                "B_pipeline_coupling": class_b_pred,
                "C_commitment_timing": class_c_pred,
                "D_assembly_compression": class_d_pred,
            },
        }
        case_rows.append(case_payload)

        exp56_seed_rows.append(
            {
                "seed": seed,
                "epsilon": epsilon,
                "graph": graph,
                "state": state,
            }
        )

    results = {
        "n_cases": len(case_rows),
        "n_failure_cases": int(sum(1 for row in case_rows if not row["greedy_valid"])),
        "baseline_source": baseline[0]["source"] if baseline else "none",
        "class_metrics_all_cases": {
            "A_absorbing": _metric_counts(truth_a, pred_a),
            "B_pipeline_coupling": _metric_counts(truth_b, pred_b),
            "C_commitment_timing": _metric_counts(truth_c, pred_c),
            "D_assembly_compression": _metric_counts(truth_d, pred_d),
        },
        "class_metrics_failure_only": {
            "A_absorbing": _metric_counts(truth_a_fail, pred_a_fail),
            "B_pipeline_coupling": _metric_counts(truth_b_fail, pred_b_fail),
            "C_commitment_timing": _metric_counts(truth_c_fail, pred_c_fail),
            "D_assembly_compression": _metric_counts(truth_d_fail, pred_d_fail),
        },
        "target_check": {
            "all_class_accuracies_100pct_failure_only": bool(
                all(
                    abs(float(metric["accuracy"]) - 1.0) <= 1e-12
                    for metric in [
                        _metric_counts(truth_a_fail, pred_a_fail),
                        _metric_counts(truth_b_fail, pred_b_fail),
                        _metric_counts(truth_c_fail, pred_c_fail),
                        _metric_counts(truth_d_fail, pred_d_fail),
                    ]
                )
            )
        },
        "cases": case_rows,
    }
    return results, exp56_seed_rows


def _split_four_segments(
    events: tuple[Event, ...],
    rng: random.Random,
) -> list[tuple[Event, ...]]:
    n = len(events)
    if n < 8:
        return [events, (), (), ()]
    cuts = sorted(rng.sample(range(1, n), 3))
    a, b, c = cuts
    return [
        tuple(events[:a]),
        tuple(events[a:b]),
        tuple(events[b:c]),
        tuple(events[c:]),
    ]


def _q_r_equal(left: ContextState, right: ContextState) -> bool:
    return bool(left.q == right.q and left.r == right.r)


def _pivot_only_equal(left: ContextState, right: ContextState) -> bool:
    return context_equivalent(left, right, compare_pivots=True)


def run_exp51_composition_operator(scale: ScaleConfig) -> dict:
    """
    Experiment 51:
    Validate composition operator and associativity.
    """
    generator = BurstyGenerator()
    rng = random.Random(51007)
    eps_grid = [0.2, 0.5, 0.8]
    M = 25

    pairwise_total = 0
    pairwise_qr_exact = 0
    pairwise_pivot_exact = 0

    assoc_total = 0
    assoc_violations = 0
    assoc_qr_violations = 0

    discrepancy_samples: list[dict] = []

    for idx in range(scale.exp51_n_sequences):
        epsilon = float(eps_grid[idx % len(eps_grid)])
        seed = int(idx)
        graph = generator.generate(
            BurstyConfig(
                n_events=200,
                n_actors=6,
                seed=seed,
                epsilon=epsilon,
            )
        )
        segments = _split_four_segments(graph.events, rng=rng)
        states = [build_context_state(segment, focal_actor=FOCAL_ACTOR, M=M) for segment in segments]

        # Pairwise checks for AB, BC, CD against ground truth concatenation.
        pair_specs = [
            (0, 1),
            (1, 2),
            (2, 3),
        ]
        for i, j in pair_specs:
            composed = compose_context_states(states[i], states[j], M=M)
            truth = build_context_state((*segments[i], *segments[j]), focal_actor=FOCAL_ACTOR, M=M)

            pairwise_total += 1
            if _q_r_equal(composed, truth):
                pairwise_qr_exact += 1
            if _pivot_only_equal(composed, truth):
                pairwise_pivot_exact += 1
            elif len(discrepancy_samples) < 12:
                discrepancy_samples.append(
                    {
                        "seed": seed,
                        "epsilon": epsilon,
                        "pair": [i, j],
                        "composed_q": composed.q,
                        "truth_q": truth.q,
                        "composed_r": asdict(composed.r),
                        "truth_r": asdict(truth.r),
                        "composed_pivots": [asdict(candidate) for candidate in composed.pivots[:5]],
                        "truth_pivots": [asdict(candidate) for candidate in truth.pivots[:5]],
                    }
                )

        # Associativity check on first three segments: (AB)C vs A(BC)
        left_assoc = compose_context_states(
            compose_context_states(states[0], states[1], M=M),
            states[2],
            M=M,
        )
        right_assoc = compose_context_states(
            states[0],
            compose_context_states(states[1], states[2], M=M),
            M=M,
        )
        assoc_total += 1
        if not context_equivalent(left_assoc, right_assoc, compare_pivots=True):
            assoc_violations += 1
        if not _q_r_equal(left_assoc, right_assoc):
            assoc_qr_violations += 1

    return {
        "n_sequences": int(scale.exp51_n_sequences),
        "M": int(M),
        "pairwise_exact_match": {
            "n_total": int(pairwise_total),
            "q_r_exact": int(pairwise_qr_exact),
            "q_r_exact_rate": _safe_rate(pairwise_qr_exact, pairwise_total),
            "pivot_exact": int(pairwise_pivot_exact),
            "pivot_exact_rate": _safe_rate(pairwise_pivot_exact, pairwise_total),
        },
        "associativity": {
            "n_total": int(assoc_total),
            "violations_any_component": int(assoc_violations),
            "violation_rate_any_component": _safe_rate(assoc_violations, assoc_total),
            "violations_q_r": int(assoc_qr_violations),
            "violation_rate_q_r": _safe_rate(assoc_qr_violations, assoc_total),
        },
        "discrepancy_samples": discrepancy_samples,
    }


def _make_graph_for_exp52(
    case_index: int,
    n_events: int,
    m_values: tuple[int, ...],
) -> tuple[CausalGraph, dict]:
    condition = case_index % 4
    k = [1, 2, 3][case_index % 3]
    if not m_values:
        raise ValueError("m_values must not be empty")
    M = int(m_values[case_index % len(m_values)])

    if condition in {0, 1}:
        epsilon = [0.2, 0.5, 0.8][case_index % 3]
        graph = BurstyGenerator().generate(
            BurstyConfig(
                n_events=n_events,
                n_actors=6,
                seed=case_index,
                epsilon=epsilon,
            )
        )
        condition_name = "bursty"
    else:
        graph = MultiBurstGenerator().generate(
            MultiBurstConfig(
                n_events=n_events,
                n_actors=6,
                seed=case_index,
            )
        )
        condition_name = "multiburst"

    with_gap = condition in {1, 3}
    max_gap = 0.11 if with_gap else float("inf")
    grammar = GrammarConfig(
        min_prefix_elements=int(k),
        max_phase_regressions=0,
        max_turning_points=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.15,
        max_temporal_gap=max_gap,
        focal_actor_coverage=0.60,
    )

    return graph, {
        "condition": condition_name,
        "with_gap": with_gap,
        "k": int(k),
        "M": int(M),
        "max_gap": max_gap,
    }


def _run_tp_solver_validity(graph: CausalGraph, grammar: GrammarConfig, M: int, tp_ids: tuple[str, ...] | None = None) -> dict:
    seq, diag = tp_conditioned_solve(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        M=int(M),
        max_gap=float(grammar.max_temporal_gap),
        pool_strategy="injection",
        max_sequence_length=20,
        injection_top_n=40,
        tp_candidate_ids=tp_ids,
    )
    return {
        "valid": bool(seq is not None and seq.valid),
        "score": None if seq is None else float(seq.score),
        "n_events": 0 if seq is None else len(seq.events),
        "diag": diag,
    }


def run_exp52_compression_with_guards(
    scale: ScaleConfig,
    m_values: tuple[int, ...] = (5, 10, 25),
    retention_ratio: float = 0.50,
) -> dict:
    """
    Experiment 52:
    Compression strategies under no-absorption invariants.
    """
    strategies = ["naive", "bridge_preserving", "contract_guarded"]
    per_case: list[dict] = []

    validity_by_strategy = {name: [] for name in strategies}
    retain_ratio_by_strategy = {name: [] for name in strategies}
    contract_pressure_by_strategy = {name: [] for name in strategies}
    theorem_counterexamples = 0
    theorem_checks = 0

    n_events = 120 if scale.name == "smoke" else 200

    for case_index in range(scale.exp52_n_sequences):
        graph, meta = _make_graph_for_exp52(case_index, n_events=n_events, m_values=m_values)
        grammar = GrammarConfig(
            min_prefix_elements=int(meta["k"]),
            max_phase_regressions=0,
            max_turning_points=1,
            min_length=4,
            max_length=20,
            min_timespan_fraction=0.15,
            max_temporal_gap=float(meta["max_gap"]),
            focal_actor_coverage=0.60,
        )
        M = int(meta["M"])
        full_events = tuple(graph.events)
        target_size = max(8, int(round(len(full_events) * float(retention_ratio))))

        pre_state = build_context_state(full_events, focal_actor=FOCAL_ACTOR, M=M)
        top_m_ids = tuple(candidate.event_id for candidate in pre_state.pivots)
        pre_full = _run_tp_solver_validity(graph=graph, grammar=grammar, M=M)
        pre_top_m = _run_tp_solver_validity(graph=graph, grammar=grammar, M=M, tp_ids=top_m_ids)

        case_payload: dict[str, Any] = {
            "case_index": int(case_index),
            "meta": meta,
            "pre": {
                "full_solver_valid": bool(pre_full["valid"]),
                "top_M_solver_valid": bool(pre_top_m["valid"]),
                "top_M_ids": list(top_m_ids),
                "no_absorption_invariant_pre": bool(no_absorption_invariant(pre_state, k=grammar.min_prefix_elements)),
            },
            "strategies": {},
        }

        for strategy in strategies:
            compressed_events, compression_diag = compress_events(
                events=full_events,
                focal_actor=FOCAL_ACTOR,
                k=grammar.min_prefix_elements,
                M=M,
                target_size=target_size,
                strategy=strategy,
                max_gap=float(grammar.max_temporal_gap),
                min_length=grammar.min_length,
            )
            compressed_graph = induced_subgraph(graph, [event.id for event in compressed_events])
            post = _run_tp_solver_validity(graph=compressed_graph, grammar=grammar, M=M)

            retained_ratio = _safe_rate(len(compressed_events), len(full_events))
            validity_by_strategy[strategy].append(bool(post["valid"]))
            retain_ratio_by_strategy[strategy].append(float(retained_ratio))
            contract_pressure_by_strategy[strategy].append(
                int(compression_diag["contract_violating_drop_attempts"])
            )

            if strategy == "contract_guarded" and bool(pre_top_m["valid"]):
                theorem_checks += 1
                if not bool(post["valid"]):
                    theorem_counterexamples += 1

            case_payload["strategies"][strategy] = {
                "compressed_size": int(len(compressed_events)),
                "retained_ratio": float(retained_ratio),
                "solver_valid": bool(post["valid"]),
                "solver_score": post["score"],
                "compression_diag": compression_diag,
            }
        per_case.append(case_payload)

    aggregate = {}
    for strategy in strategies:
        validity = validity_by_strategy[strategy]
        retains = retain_ratio_by_strategy[strategy]
        pressure = contract_pressure_by_strategy[strategy]
        aggregate[strategy] = {
            "n_cases": len(validity),
            "post_compression_valid_rate": _safe_rate(sum(1 for v in validity if v), len(validity)),
            "mean_retained_ratio": float(mean(retains)) if retains else 0.0,
            "median_retained_ratio": float(median(retains)) if retains else 0.0,
            "mean_contract_violating_drop_attempts": float(mean(pressure)) if pressure else 0.0,
            "median_contract_violating_drop_attempts": float(median(pressure)) if pressure else 0.0,
        }

    by_condition: dict[str, list[dict]] = {}
    for row in per_case:
        meta = row["meta"]
        condition_key = f"{meta['condition']}_{'gap' if meta['with_gap'] else 'nogap'}"
        by_condition.setdefault(condition_key, []).append(row)

    aggregate_by_condition: dict[str, dict[str, dict[str, float]]] = {}
    for condition_key, rows in sorted(by_condition.items()):
        condition_stats: dict[str, dict[str, float]] = {}
        n_rows = len(rows)
        for strategy in strategies:
            valid_rate = _safe_rate(
                sum(1 for row in rows if row["strategies"][strategy]["solver_valid"]),
                n_rows,
            )
            retained_vals = [float(row["strategies"][strategy]["retained_ratio"]) for row in rows]
            rejected_gap_guard_vals = [
                float(row["strategies"][strategy]["compression_diag"].get("rejected_gap_guard", 0))
                for row in rows
            ]
            condition_stats[strategy] = {
                "n_cases": int(n_rows),
                "valid_rate": float(valid_rate),
                "mean_retained_ratio": float(mean(retained_vals)) if retained_vals else 0.0,
                "mean_rejected_gap_guard": (
                    float(mean(rejected_gap_guard_vals)) if rejected_gap_guard_vals else 0.0
                ),
            }
        aggregate_by_condition[condition_key] = condition_stats

    return {
        "n_cases": int(scale.exp52_n_sequences),
        "m_values": [int(value) for value in m_values],
        "retention_ratio": float(retention_ratio),
        "aggregate": aggregate,
        "aggregate_by_condition": aggregate_by_condition,
        "contract_theorem_candidate": {
            "n_checks_pre_top_M_feasible": int(theorem_checks),
            "counterexamples": int(theorem_counterexamples),
            "counterexample_rate": _safe_rate(theorem_counterexamples, theorem_checks),
            "supported": bool(theorem_counterexamples == 0),
        },
        "dominance_check": {
            "contract_vs_naive_validity_delta": float(
                aggregate["contract_guarded"]["post_compression_valid_rate"]
                - aggregate["naive"]["post_compression_valid_rate"]
            ),
            "contract_vs_bridge_validity_delta": float(
                aggregate["contract_guarded"]["post_compression_valid_rate"]
                - aggregate["bridge_preserving"]["post_compression_valid_rate"]
            ),
        },
        "cases": per_case,
    }


def run_exp53_antagonism_under_composition(scale: ScaleConfig) -> dict:
    """
    Experiment 53:
    Segment-wise compression antagonism under composition.
    """
    generator = MultiBurstGenerator()
    grammar = GrammarConfig(
        min_prefix_elements=2,
        max_phase_regressions=0,
        max_turning_points=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.15,
        max_temporal_gap=0.11,
        focal_actor_coverage=0.60,
    )
    M = 10
    split_boundary = 0.50

    segment_valid = []
    whole_valid = []
    gap_class_counts: dict[str, int] = {}
    per_case: list[dict] = []

    for seed in range(scale.exp53_n_sequences):
        graph = generator.generate(
            MultiBurstConfig(
                n_events=200,
                n_actors=6,
                seed=seed,
            )
        )
        events = tuple(graph.events)
        left = tuple(event for event in events if float(event.timestamp) < split_boundary)
        right = tuple(event for event in events if float(event.timestamp) >= split_boundary)

        left_target = max(4, int(round(len(left) * 0.60)))
        right_target = max(4, int(round(len(right) * 0.60)))
        left_comp, left_diag = compress_events(
            events=left,
            focal_actor=FOCAL_ACTOR,
            k=grammar.min_prefix_elements,
            M=M,
            target_size=left_target,
            strategy="contract_guarded",
            max_gap=float(grammar.max_temporal_gap),
            min_length=grammar.min_length,
        )
        right_comp, right_diag = compress_events(
            events=right,
            focal_actor=FOCAL_ACTOR,
            k=grammar.min_prefix_elements,
            M=M,
            target_size=right_target,
            strategy="contract_guarded",
            max_gap=float(grammar.max_temporal_gap),
            min_length=grammar.min_length,
        )

        segment_events = tuple(sorted((*left_comp, *right_comp), key=lambda event: (event.timestamp, event.id)))
        whole_target = max(8, int(round(len(events) * 0.60)))
        whole_comp, whole_diag = compress_events(
            events=events,
            focal_actor=FOCAL_ACTOR,
            k=grammar.min_prefix_elements,
            M=M,
            target_size=whole_target,
            strategy="contract_guarded",
            max_gap=float(grammar.max_temporal_gap),
            min_length=grammar.min_length,
        )

        segment_graph = induced_subgraph(graph, [event.id for event in segment_events])
        whole_graph = induced_subgraph(graph, [event.id for event in whole_comp])

        segment_solver = _run_tp_solver_validity(graph=segment_graph, grammar=grammar, M=M)
        whole_solver = _run_tp_solver_validity(graph=whole_graph, grammar=grammar, M=M)
        segment_valid.append(bool(segment_solver["valid"]))
        whole_valid.append(bool(whole_solver["valid"]))

        left_state = build_context_state(left_comp, focal_actor=FOCAL_ACTOR, M=M)
        right_state = build_context_state(right_comp, focal_actor=FOCAL_ACTOR, M=M)
        composed_state = compose_context_states(left_state, right_state, M=M)

        mechanism = "none"
        if whole_solver["valid"] and (not segment_solver["valid"]):
            if detect_class_c(composed_state, k=grammar.min_prefix_elements):
                mechanism = "class_c_commitment_timing"
            elif detect_class_d(
                composed_state,
                min_timespan_fraction=float(grammar.min_timespan_fraction),
                graph_duration=float(graph.duration),
                min_length=int(grammar.min_length),
            ):
                mechanism = "class_d_assembly_compression"
            else:
                mechanism = "other_cross_segment_antagonism"
            gap_class_counts[mechanism] = gap_class_counts.get(mechanism, 0) + 1

        per_case.append(
            {
                "seed": int(seed),
                "segment_valid": bool(segment_solver["valid"]),
                "whole_valid": bool(whole_solver["valid"]),
                "mechanism": mechanism,
                "segment_sizes": {"left": len(left_comp), "right": len(right_comp), "total": len(segment_events)},
                "whole_size": len(whole_comp),
                "left_diag": left_diag,
                "right_diag": right_diag,
                "whole_diag": whole_diag,
            }
        )

    segment_rate = _safe_rate(sum(1 for v in segment_valid if v), len(segment_valid))
    whole_rate = _safe_rate(sum(1 for v in whole_valid if v), len(whole_valid))
    return {
        "n_cases": int(scale.exp53_n_sequences),
        "segmentwise_valid_rate": float(segment_rate),
        "whole_sequence_valid_rate": float(whole_rate),
        "validity_gap_whole_minus_segment": float(whole_rate - segment_rate),
        "cross_segment_failure_mechanism_counts": dict(sorted(gap_class_counts.items())),
        "cases": per_case,
    }


def _stream_with_deferred_composition(
    graph: CausalGraph,
    grammar: GrammarConfig,
    f: float,
    M: int,
    chunk_size: int = 50,
) -> dict:
    events = tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))
    n_total = len(events)
    idx = 0

    running_state = build_context_state((), focal_actor=FOCAL_ACTOR, M=M)
    running_events: list[Event] = []
    pivot_history: list[str | None] = []
    compose_batch_sizes: list[int] = []

    while idx < n_total:
        base_take = min(chunk_size, n_total - idx)
        batch = list(events[idx : idx + base_take])
        idx += base_take

        remaining = max(0, n_total - idx)
        extra = int(round(float(f) * float(remaining)))
        if extra > 0:
            extra_take = min(extra, n_total - idx)
            batch.extend(events[idx : idx + extra_take])
            idx += extra_take

        batch_target = max(4, int(round(len(batch) * 0.70)))
        compressed_batch, _ = compress_events(
            events=batch,
            focal_actor=FOCAL_ACTOR,
            k=grammar.min_prefix_elements,
            M=M,
            target_size=batch_target,
            strategy="contract_guarded",
            max_gap=float(grammar.max_temporal_gap),
            min_length=grammar.min_length,
        )
        batch_state = build_context_state(compressed_batch, focal_actor=FOCAL_ACTOR, M=M)
        running_state = compose_context_states(running_state, batch_state, M=M)

        running_events.extend(compressed_batch)
        running_events = sorted({event.id: event for event in running_events}.values(), key=lambda event: (event.timestamp, event.id))
        compose_batch_sizes.append(len(batch))
        pivot_history.append(running_state.pivots[0].event_id if running_state.pivots else None)

    pivot_shifts = 0
    for left, right in zip(pivot_history[:-1], pivot_history[1:]):
        if left != right:
            pivot_shifts += 1

    compressed_graph = induced_subgraph(graph, [event.id for event in running_events])
    final = _run_tp_solver_validity(graph=compressed_graph, grammar=grammar, M=M)
    return {
        "valid": bool(final["valid"]),
        "score": final["score"],
        "retained_ratio": _safe_rate(len(running_events), len(events)),
        "pivot_shifts": int(pivot_shifts),
        "n_compositions": len(compose_batch_sizes),
        "mean_batch_size": float(mean(compose_batch_sizes)) if compose_batch_sizes else 0.0,
        "median_batch_size": float(median(compose_batch_sizes)) if compose_batch_sizes else 0.0,
    }


def run_exp54_deferred_composition(scale: ScaleConfig) -> dict:
    """
    Experiment 54:
    Patience on composition steps in streaming mode.
    """
    generator = BurstyGenerator()
    patience_values = [0.0, 0.10, 0.25, 0.50]
    M = 10

    rows: list[dict] = []
    by_f: dict[float, list[dict]] = {f: [] for f in patience_values}

    for i in range(scale.exp54_n_sequences):
        epsilon = [0.2, 0.5, 0.8][i % 3]
        k = [1, 2, 3][i % 3]
        graph = generator.generate(
            BurstyConfig(
                n_events=200,
                n_actors=6,
                seed=i,
                epsilon=epsilon,
            )
        )
        grammar = GrammarConfig(
            min_prefix_elements=k,
            max_phase_regressions=0,
            max_turning_points=1,
            min_length=4,
            max_length=20,
            min_timespan_fraction=0.15,
            max_temporal_gap=0.11,
            focal_actor_coverage=0.60,
        )

        for f in patience_values:
            out = _stream_with_deferred_composition(graph=graph, grammar=grammar, f=f, M=M)
            row = {
                "seed": int(i),
                "epsilon": float(epsilon),
                "k": int(k),
                "patience_f": float(f),
                **out,
            }
            rows.append(row)
            by_f[f].append(row)

    summary_rows = []
    for f in patience_values:
        bucket = by_f[f]
        valid_rate = _safe_rate(sum(1 for row in bucket if row["valid"]), len(bucket))
        valid_scores = [float(row["score"]) for row in bucket if row["valid"] and row["score"] is not None]
        summary_rows.append(
            {
                "patience_f": float(f),
                "n_cases": len(bucket),
                "valid_rate": float(valid_rate),
                "mean_score_on_valid": float(mean(valid_scores)) if valid_scores else None,
                "mean_retained_ratio": float(mean(float(row["retained_ratio"]) for row in bucket)) if bucket else 0.0,
                "mean_pivot_shifts": float(mean(float(row["pivot_shifts"]) for row in bucket)) if bucket else 0.0,
                "mean_batch_size": float(mean(float(row["mean_batch_size"]) for row in bucket)) if bucket else 0.0,
            }
        )

    base = next((row for row in summary_rows if abs(row["patience_f"] - 0.0) <= 1e-12), None)
    nonzero = [row for row in summary_rows if row["patience_f"] > 0.0]
    dominates_base = False
    if base is not None and nonzero:
        dominates_base = any(
            float(row["valid_rate"]) >= float(base["valid_rate"]) and float(row["mean_batch_size"]) >= float(base["mean_batch_size"])
            for row in nonzero
        )

    return {
        "n_cases": int(scale.exp54_n_sequences),
        "summary_by_patience": summary_rows,
        "dominance_check_nonzero_vs_zero": bool(dominates_base),
        "rows": rows,
    }


def _recursive_reduce_events(
    graph: CausalGraph,
    grammar: GrammarConfig,
    M: int,
    block_size: int,
) -> dict:
    events = tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))
    chunks = [tuple(events[i : i + block_size]) for i in range(0, len(events), block_size)]

    level_rows: list[dict] = []
    depth = 0
    while len(chunks) > 1:
        compressed_chunks: list[tuple[Event, ...]] = []
        for chunk in chunks:
            target = max(4, int(round(len(chunk) * 0.60)))
            compressed, _diag = compress_events(
                events=chunk,
                focal_actor=FOCAL_ACTOR,
                k=grammar.min_prefix_elements,
                M=M,
                target_size=target,
                strategy="contract_guarded",
                max_gap=float(grammar.max_temporal_gap),
                min_length=grammar.min_length,
            )
            compressed_chunks.append(compressed)

        level_events = tuple(
            sorted(
                {event.id: event for chunk in compressed_chunks for event in chunk}.values(),
                key=lambda event: (event.timestamp, event.id),
            )
        )
        level_graph = induced_subgraph(graph, [event.id for event in level_events])
        level_solver = _run_tp_solver_validity(graph=level_graph, grammar=grammar, M=M)
        level_rows.append(
            {
                "depth": int(depth),
                "n_chunks": int(len(chunks)),
                "n_events_retained": int(len(level_events)),
                "valid": bool(level_solver["valid"]),
                "score": level_solver["score"],
            }
        )

        merged: list[tuple[Event, ...]] = []
        for i in range(0, len(compressed_chunks), 2):
            left = compressed_chunks[i]
            right = compressed_chunks[i + 1] if (i + 1) < len(compressed_chunks) else ()
            merged.append(tuple(sorted((*left, *right), key=lambda event: (event.timestamp, event.id))))
        chunks = merged
        depth += 1

    root_events = chunks[0] if chunks else ()
    root_graph = induced_subgraph(graph, [event.id for event in root_events])
    root_solver = _run_tp_solver_validity(graph=root_graph, grammar=grammar, M=M)
    return {
        "root_events_retained": int(len(root_events)),
        "root_solver": root_solver,
        "levels": level_rows,
        "depth": int(depth),
    }


def run_exp55_recursive_depth(scale: ScaleConfig) -> dict:
    """
    Experiment 55:
    Recursive composition depth scaling checks.
    """
    generator = RecursiveBurstGenerator()
    if scale.name == "smoke":
        n_values = [100, 500, 2000]
        block_values = [20, 50]
    else:
        n_values = [100, 500, 2000, 10000]
        block_values = [20, 50, 100]
    M = 10

    rows: list[dict] = []
    for n_events in n_values:
        for block_size in block_values:
            for seed in range(scale.exp55_seeds_per_cell):
                k = [1, 2, 3][seed % 3]
                grammar = GrammarConfig(
                    min_prefix_elements=k,
                    max_phase_regressions=0,
                    max_turning_points=1,
                    min_length=4,
                    max_length=20,
                    min_timespan_fraction=0.15,
                    max_temporal_gap=0.11,
                    focal_actor_coverage=0.60,
                )
                graph = generator.generate(
                    RecursiveBurstConfig(
                        n_events=n_events,
                        n_actors=6,
                        seed=(1000 * n_events) + (100 * block_size) + seed,
                        epsilon=0.60,
                        levels=3,
                        branch_factor=2,
                    )
                )

                flat_start = time.perf_counter()
                flat_solver = _run_tp_solver_validity(graph=graph, grammar=grammar, M=M)
                flat_time = float(time.perf_counter() - flat_start)

                rec_start = time.perf_counter()
                recursive = _recursive_reduce_events(
                    graph=graph,
                    grammar=grammar,
                    M=M,
                    block_size=block_size,
                )
                rec_time = float(time.perf_counter() - rec_start)

                flat_valid = bool(flat_solver["valid"])
                rec_valid = bool(recursive["root_solver"]["valid"])
                flat_score = float(flat_solver["score"]) if flat_solver["score"] is not None else None
                rec_score = float(recursive["root_solver"]["score"]) if recursive["root_solver"]["score"] is not None else None
                score_ratio = None
                if flat_score is not None and flat_score > 1e-12 and rec_score is not None:
                    score_ratio = float(rec_score / flat_score)

                first_degrade_depth = None
                for level_row in recursive["levels"]:
                    if flat_valid and (not bool(level_row["valid"])):
                        first_degrade_depth = int(level_row["depth"])
                        break

                rows.append(
                    {
                        "n_events": int(n_events),
                        "block_size": int(block_size),
                        "seed": int(seed),
                        "k": int(k),
                        "flat_valid": flat_valid,
                        "recursive_valid": rec_valid,
                        "flat_score": flat_score,
                        "recursive_score": rec_score,
                        "score_ratio": score_ratio,
                        "flat_runtime_seconds": float(flat_time),
                        "recursive_runtime_seconds": float(rec_time),
                        "runtime_ratio_recursive_over_flat": (float(rec_time / flat_time) if flat_time > 0 else None),
                        "first_degrade_depth": first_degrade_depth,
                        "depth": int(recursive["depth"]),
                        "root_events_retained": int(recursive["root_events_retained"]),
                        "levels": recursive["levels"],
                    }
                )

    by_cell: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["n_events"]), int(row["block_size"]))
        by_cell.setdefault(key, []).append(row)

    summary_cells = []
    for (n_events, block_size), bucket in sorted(by_cell.items()):
        valid_flat = [bool(row["flat_valid"]) for row in bucket]
        valid_rec = [bool(row["recursive_valid"]) for row in bucket]
        flat_valid_count = sum(1 for v in valid_flat if v)
        rec_valid_count = sum(1 for v in valid_rec if v)
        validity_ratio = (
            None
            if flat_valid_count == 0
            else _safe_rate(rec_valid_count, flat_valid_count)
        )
        score_ratios = [float(row["score_ratio"]) for row in bucket if row["score_ratio"] is not None]
        runtime_ratios = [
            float(row["runtime_ratio_recursive_over_flat"])
            for row in bucket
            if row["runtime_ratio_recursive_over_flat"] is not None
        ]
        summary_cells.append(
            {
                "n_events": int(n_events),
                "block_size": int(block_size),
                "n_cases": len(bucket),
                "recursive_valid_rate": _safe_rate(sum(1 for v in valid_rec if v), len(valid_rec)),
                "flat_valid_rate": _safe_rate(sum(1 for v in valid_flat if v), len(valid_flat)),
                "validity_retention_ratio_vs_flat": (
                    None if validity_ratio is None else float(validity_ratio)
                ),
                "mean_score_ratio_recursive_over_flat": (float(mean(score_ratios)) if score_ratios else None),
                "mean_runtime_ratio_recursive_over_flat": (float(mean(runtime_ratios)) if runtime_ratios else None),
            }
        )

    return {
        "n_cases": len(rows),
        "summary_cells": summary_cells,
        "target_check": {
            "all_defined_cells_validity_retention_ge_90pct": bool(
                all(
                    float(cell["validity_retention_ratio_vs_flat"]) >= 0.90
                    for cell in summary_cells
                    if cell["validity_retention_ratio_vs_flat"] is not None
                )
            ),
            "n_cells_with_defined_retention": int(
                sum(1 for cell in summary_cells if cell["validity_retention_ratio_vs_flat"] is not None)
            ),
        },
        "rows": rows,
    }


def run_exp56_absorbing_ideal(scale: ScaleConfig, exp50_context_rows: list[dict]) -> dict:
    """
    Experiment 56:
    Verify absorbing ideal closure under suffix composition.
    """
    grammar = GrammarConfig(min_prefix_elements=1)
    M = 25

    candidates = [row for row in exp50_context_rows if is_absorbing(row["state"], k=grammar.min_prefix_elements)]
    candidates = candidates[: scale.exp56_max_states]

    checks = 0
    violations = 0
    examples: list[dict] = []

    for row in candidates:
        graph: CausalGraph = row["graph"]
        state: ContextState = row["state"]
        last_t = state.t_bounds[1]
        if last_t is None:
            continue
        suffix_events = [event for event in graph.events if float(event.timestamp) > float(last_t)]
        if not suffix_events:
            continue

        for L in range(1, scale.exp56_l_max + 1):
            suffix = tuple(suffix_events[:L])
            if not suffix:
                continue
            suffix_state = build_context_state(suffix, focal_actor=FOCAL_ACTOR, M=M)
            composed = compose_context_states(state, suffix_state, M=M)
            checks += 1
            if not is_absorbing(composed, k=grammar.min_prefix_elements):
                violations += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "seed": int(row["seed"]),
                            "epsilon": float(row["epsilon"]),
                            "suffix_length": int(L),
                            "base_state": {
                                "q": state.q,
                                "r": asdict(state.r),
                                "pivots": [asdict(candidate) for candidate in state.pivots[:5]],
                            },
                            "composed_state": {
                                "q": composed.q,
                                "r": asdict(composed.r),
                                "pivots": [asdict(candidate) for candidate in composed.pivots[:5]],
                            },
                        }
                    )

    return {
        "n_absorbing_states_tested": len(candidates),
        "l_max": int(scale.exp56_l_max),
        "checks": int(checks),
        "violations": int(violations),
        "violation_rate": _safe_rate(violations, checks),
        "examples": examples,
    }


def _is_valid_context_state(state: ContextState) -> bool:
    if int(state.r.slots_used) < 0:
        return False
    if int(state.r.dev_count) < 0:
        return False
    if int(state.r.prefix_count) < 0:
        return False
    if int(state.r.setup_count) < 0:
        return False
    if int(state.r.post_count) < 0:
        return False
    if state.t_bounds[0] is not None and state.t_bounds[1] is not None:
        if float(state.t_bounds[1]) + 1e-12 < float(state.t_bounds[0]):
            return False
    return True


def run_exp57_closure(scale: ScaleConfig) -> dict:
    """
    Experiment 57:
    Algebra closure properties for candidate operations.
    """
    generator = BurstyGenerator()
    grammar = GrammarConfig(min_prefix_elements=1, max_temporal_gap=0.11)
    M = 10

    op_stats: dict[str, dict[str, int]] = {
        "composition": {"checks": 0, "invalid_output": 0, "absorbing_escape": 0, "pi_mismatch": 0},
        "compression": {"checks": 0, "invalid_output": 0, "absorbing_escape": 0, "pi_mismatch": 0},
        "pivot_update": {"checks": 0, "invalid_output": 0, "absorbing_escape": 0, "pi_mismatch": 0},
        "split_at_point": {"checks": 0, "invalid_output": 0, "absorbing_escape": 0, "pi_mismatch": 0},
    }
    compression_escape_samples: list[dict] = []
    compression_mismatch_samples: list[dict] = []

    for seed in range(scale.exp57_n_samples):
        epsilon = [0.2, 0.5, 0.8][seed % 3]
        graph = generator.generate(
            BurstyConfig(
                n_events=200,
                n_actors=6,
                seed=seed,
                epsilon=epsilon,
            )
        )
        events = tuple(graph.events)
        state = build_context_state(events, focal_actor=FOCAL_ACTOR, M=M)
        base_abs = is_absorbing(state, k=grammar.min_prefix_elements)

        # composition
        split = max(1, min(len(events) - 1, 50 + (seed % 100)))
        left = tuple(events[:split])
        right = tuple(events[split:])
        left_state = build_context_state(left, focal_actor=FOCAL_ACTOR, M=M)
        right_state = build_context_state(right, focal_actor=FOCAL_ACTOR, M=M)
        composed = compose_context_states(left_state, right_state, M=M)
        truth = build_context_state(events, focal_actor=FOCAL_ACTOR, M=M)
        op_stats["composition"]["checks"] += 1
        if not _is_valid_context_state(composed):
            op_stats["composition"]["invalid_output"] += 1
        if base_abs and (not is_absorbing(composed, k=grammar.min_prefix_elements)):
            op_stats["composition"]["absorbing_escape"] += 1
        if not context_equivalent(composed, truth, compare_pivots=True):
            op_stats["composition"]["pi_mismatch"] += 1

        # compression
        compressed, _diag = compress_events(
            events=events,
            focal_actor=FOCAL_ACTOR,
            k=grammar.min_prefix_elements,
            M=M,
            target_size=max(8, int(round(len(events) * 0.60))),
            strategy="contract_guarded",
            max_gap=float(grammar.max_temporal_gap),
            min_length=grammar.min_length,
        )
        comp_state = build_context_state(compressed, focal_actor=FOCAL_ACTOR, M=M)
        op_stats["compression"]["checks"] += 1
        if not _is_valid_context_state(comp_state):
            op_stats["compression"]["invalid_output"] += 1
        escape = base_abs and (not is_absorbing(comp_state, k=grammar.min_prefix_elements))
        if escape:
            op_stats["compression"]["absorbing_escape"] += 1
        invariant_break = not no_absorption_invariant(comp_state, k=grammar.min_prefix_elements)
        if invariant_break:
            op_stats["compression"]["pi_mismatch"] += 1

        comp_ids = {event.id for event in compressed}
        base_top = state.pivots[0].event_id if state.pivots else None
        comp_top = comp_state.pivots[0].event_id if comp_state.pivots else None
        if escape and len(compression_escape_samples) < 12:
            compression_escape_samples.append(
                {
                    "seed": int(seed),
                    "epsilon": float(epsilon),
                    "base_top_pivot": base_top,
                    "compressed_top_pivot": comp_top,
                    "base_top_retained": bool(base_top in comp_ids) if base_top is not None else None,
                    "dropped_base_pivots": [
                        candidate.event_id for candidate in state.pivots if candidate.event_id not in comp_ids
                    ][:10],
                    "base_q": state.q,
                    "compressed_q": comp_state.q,
                    "base_r": asdict(state.r),
                    "compressed_r": asdict(comp_state.r),
                }
            )
        if invariant_break and len(compression_mismatch_samples) < 12:
            compression_mismatch_samples.append(
                {
                    "seed": int(seed),
                    "epsilon": float(epsilon),
                    "base_top_pivot": base_top,
                    "compressed_top_pivot": comp_top,
                    "base_top_retained": bool(base_top in comp_ids) if base_top is not None else None,
                    "dropped_base_pivots": [
                        candidate.event_id for candidate in state.pivots if candidate.event_id not in comp_ids
                    ][:10],
                    "base_q": state.q,
                    "compressed_q": comp_state.q,
                    "base_r": asdict(state.r),
                    "compressed_r": asdict(comp_state.r),
                    "compression_size": int(len(compressed)),
                }
            )

        # pivot-update (append one deterministic suffix event if available)
        update_state = state
        if right:
            singleton_state = build_context_state((right[0],), focal_actor=FOCAL_ACTOR, M=M)
            update_state = compose_context_states(left_state, singleton_state, M=M)
        op_stats["pivot_update"]["checks"] += 1
        if not _is_valid_context_state(update_state):
            op_stats["pivot_update"]["invalid_output"] += 1
        if base_abs and (not is_absorbing(update_state, k=grammar.min_prefix_elements)):
            op_stats["pivot_update"]["absorbing_escape"] += 1
        if not update_state.pivots:
            op_stats["pivot_update"]["pi_mismatch"] += 1

        # split-at-point closure
        split2 = max(1, min(len(events) - 1, 20 + (3 * seed % 160)))
        left2 = tuple(events[:split2])
        right2 = tuple(events[split2:])
        recomposed = compose_context_states(
            build_context_state(left2, focal_actor=FOCAL_ACTOR, M=M),
            build_context_state(right2, focal_actor=FOCAL_ACTOR, M=M),
            M=M,
        )
        truth2 = build_context_state(events, focal_actor=FOCAL_ACTOR, M=M)
        op_stats["split_at_point"]["checks"] += 1
        if not _is_valid_context_state(recomposed):
            op_stats["split_at_point"]["invalid_output"] += 1
        if base_abs and (not is_absorbing(recomposed, k=grammar.min_prefix_elements)):
            op_stats["split_at_point"]["absorbing_escape"] += 1
        if not context_equivalent(recomposed, truth2, compare_pivots=True):
            op_stats["split_at_point"]["pi_mismatch"] += 1

    operation_rows = {}
    for op, stats in op_stats.items():
        checks = int(stats["checks"])
        operation_rows[op] = {
            **stats,
            "closure_violation_rate": _safe_rate(
                int(stats["invalid_output"]) + int(stats["pi_mismatch"]),
                checks,
            ),
            "absorbing_escape_rate": _safe_rate(int(stats["absorbing_escape"]), checks),
        }

    return {
        "n_samples": int(scale.exp57_n_samples),
        "operations": operation_rows,
        "compression_diagnostics": {
            "absorbing_escape_samples": compression_escape_samples,
            "invariant_break_samples": compression_mismatch_samples,
        },
    }


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        "## Experiments Included",
    ]
    for key in sorted(data["experiments"].keys()):
        lines.append(f"- Exp {key}")

    if "50" in data["experiments"]:
        exp = data["experiments"]["50"]
        fm = exp["class_metrics_failure_only"]
        lines.extend(
            [
                "",
                "## Exp 50",
                f"- Cases: {exp['n_cases']}",
                f"- Failure cases: {exp['n_failure_cases']}",
                f"- A accuracy (failure-only): {fm['A_absorbing']['accuracy']:.3f}",
                f"- B accuracy (failure-only): {fm['B_pipeline_coupling']['accuracy']:.3f}",
                f"- C accuracy (failure-only): {fm['C_commitment_timing']['accuracy']:.3f}",
                f"- D accuracy (failure-only): {fm['D_assembly_compression']['accuracy']:.3f}",
            ]
        )

    if "51" in data["experiments"]:
        exp = data["experiments"]["51"]
        lines.extend(
            [
                "",
                "## Exp 51",
                f"- q/r exact rate: {exp['pairwise_exact_match']['q_r_exact_rate']:.3f}",
                f"- pivot exact rate: {exp['pairwise_exact_match']['pivot_exact_rate']:.3f}",
                f"- associativity violation rate: {exp['associativity']['violation_rate_any_component']:.3f}",
            ]
        )

    if "52" in data["experiments"]:
        exp = data["experiments"]["52"]
        lines.extend(
            [
                "",
                "## Exp 52",
                f"- retention_ratio: {exp.get('retention_ratio', 0.5):.2f}",
                f"- M values: {exp.get('m_values', [])}",
                f"- naive validity: {exp['aggregate']['naive']['post_compression_valid_rate']:.3f}",
                f"- bridge validity: {exp['aggregate']['bridge_preserving']['post_compression_valid_rate']:.3f}",
                f"- contract validity: {exp['aggregate']['contract_guarded']['post_compression_valid_rate']:.3f}",
                f"- theorem counterexample rate: {exp['contract_theorem_candidate']['counterexample_rate']:.3f}",
            ]
        )
        if exp.get("aggregate_by_condition"):
            lines.extend(
                [
                    "",
                    "| condition | naive | bridge | contract | contract-naive |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for condition_key, stats in sorted(exp["aggregate_by_condition"].items()):
                naive = float(stats["naive"]["valid_rate"])
                bridge = float(stats["bridge_preserving"]["valid_rate"])
                contract = float(stats["contract_guarded"]["valid_rate"])
                lines.append(
                    f"| {condition_key} | {naive:.3f} | {bridge:.3f} | {contract:.3f} | {contract - naive:.3f} |"
                )

    if "53" in data["experiments"]:
        exp = data["experiments"]["53"]
        lines.extend(
            [
                "",
                "## Exp 53",
                f"- segment-wise validity: {exp['segmentwise_valid_rate']:.3f}",
                f"- whole-sequence validity: {exp['whole_sequence_valid_rate']:.3f}",
                f"- validity gap (whole - segment): {exp['validity_gap_whole_minus_segment']:.3f}",
            ]
        )

    if "54" in data["experiments"]:
        exp = data["experiments"]["54"]
        lines.extend(
            [
                "",
                "## Exp 54",
                f"- nonzero patience dominates f=0: {exp['dominance_check_nonzero_vs_zero']}",
            ]
        )
        lines.append("")
        lines.append("| f | valid_rate | mean_pivot_shifts | mean_batch_size |")
        lines.append("|---:|---:|---:|---:|")
        for row in exp["summary_by_patience"]:
            lines.append(
                f"| {row['patience_f']:.2f} | {row['valid_rate']:.3f} | "
                f"{row['mean_pivot_shifts']:.3f} | {row['mean_batch_size']:.2f} |"
            )

    if "55" in data["experiments"]:
        exp = data["experiments"]["55"]
        lines.extend(
            [
                "",
                "## Exp 55",
                (
                    "- target (>=90% retention on defined cells): "
                    f"{exp['target_check']['all_defined_cells_validity_retention_ge_90pct']}"
                ),
            ]
        )
        lines.append("")
        lines.append("| n | B | retention_vs_flat | runtime_ratio_rec_over_flat |")
        lines.append("|---:|---:|---:|---:|")
        for row in exp["summary_cells"]:
            retention = row["validity_retention_ratio_vs_flat"]
            retention_str = "n/a" if retention is None else f"{retention:.3f}"
            runtime_ratio = row["mean_runtime_ratio_recursive_over_flat"]
            runtime_str = "n/a" if runtime_ratio is None else f"{runtime_ratio:.3f}"
            lines.append(
                f"| {row['n_events']} | {row['block_size']} | "
                f"{retention_str} | {runtime_str} |"
            )

    if "56" in data["experiments"]:
        exp = data["experiments"]["56"]
        lines.extend(
            [
                "",
                "## Exp 56",
                f"- checks: {exp['checks']}",
                f"- violations: {exp['violations']}",
                f"- violation rate: {exp['violation_rate']:.3f}",
            ]
        )

    if "57" in data["experiments"]:
        exp = data["experiments"]["57"]
        lines.extend(
            [
                "",
                "## Exp 57",
                "| operation | closure_violation_rate | absorbing_escape_rate |",
                "|---|---:|---:|",
            ]
        )
        for name, row in exp["operations"].items():
            lines.append(
                f"| {name} | {row['closure_violation_rate']:.3f} | {row['absorbing_escape_rate']:.3f} |"
            )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        choices=["smoke", "full"],
        default="smoke",
        help="Runtime scale preset.",
    )
    parser.add_argument(
        "--experiments",
        default="50,51,52,53,54,55,56,57",
        help="Comma-separated experiment IDs to run.",
    )
    parser.add_argument(
        "--output-name",
        default="context_algebra_suite",
        help="Output basename under experiments/output.",
    )
    parser.add_argument(
        "--exp52-n",
        type=int,
        default=None,
        help="Override number of sequences for Exp 52.",
    )
    parser.add_argument(
        "--exp53-n",
        type=int,
        default=None,
        help="Override number of sequences for Exp 53.",
    )
    parser.add_argument(
        "--exp52-m-values",
        default="5,10,25",
        help="Comma-separated M values for Exp 52 pivot candidate set.",
    )
    parser.add_argument(
        "--exp52-retention",
        type=float,
        default=0.50,
        help="Retention ratio for Exp 52 compression target size.",
    )
    return parser.parse_args()


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == TARGET_HASH_SEED:
        return

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = TARGET_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main() -> None:
    _ensure_hash_seed()
    args = parse_args()
    selected = [item.strip() for item in args.experiments.split(",") if item.strip()]
    scale = SMOKE_SCALE if args.scale == "smoke" else FULL_SCALE
    if args.exp52_n is not None:
        scale = replace(scale, exp52_n_sequences=max(1, int(args.exp52_n)))
    if args.exp53_n is not None:
        scale = replace(scale, exp53_n_sequences=max(1, int(args.exp53_n)))
    exp52_m_values = tuple(
        int(item.strip())
        for item in str(args.exp52_m_values).split(",")
        if item.strip()
    )

    print(f"[context-algebra] scale={scale.name} experiments={selected}", flush=True)
    timer = ExperimentTimer()

    exp_results: dict[str, dict] = {}
    n_graphs_total = 0
    n_extractions_total = 0

    exp56_context_rows: list[dict] = []

    if "50" in selected:
        print("[exp50] running context state sufficiency checks...", flush=True)
        exp50, exp56_context_rows = run_exp50_context_state_formalization(scale=scale)
        exp_results["50"] = exp50
        n_graphs_total += int(exp50["n_cases"])
        n_extractions_total += int(exp50["n_cases"]) * 2
        print(
            "[exp50] done:",
            {
                key: round(value["accuracy"], 3)
                for key, value in exp50["class_metrics_failure_only"].items()
            },
            flush=True,
        )

    if "51" in selected:
        print("[exp51] running composition/associativity checks...", flush=True)
        exp51 = run_exp51_composition_operator(scale=scale)
        exp_results["51"] = exp51
        n_graphs_total += int(exp51["n_sequences"])
        n_extractions_total += int(exp51["pairwise_exact_match"]["n_total"])
        print(
            "[exp51] done:",
            {
                "q_r_exact_rate": round(exp51["pairwise_exact_match"]["q_r_exact_rate"], 3),
                "assoc_violation_rate": round(exp51["associativity"]["violation_rate_any_component"], 3),
            },
            flush=True,
        )

    if "52" in selected:
        print("[exp52] running compression guard evaluation...", flush=True)
        exp52 = run_exp52_compression_with_guards(
            scale=scale,
            m_values=exp52_m_values,
            retention_ratio=float(args.exp52_retention),
        )
        exp_results["52"] = exp52
        n_graphs_total += int(exp52["n_cases"])
        n_extractions_total += int(exp52["n_cases"]) * 5
        print(
            "[exp52] done:",
            {
                key: round(value["post_compression_valid_rate"], 3)
                for key, value in exp52["aggregate"].items()
            },
            flush=True,
        )

    if "53" in selected:
        print("[exp53] running antagonism under composition...", flush=True)
        exp53 = run_exp53_antagonism_under_composition(scale=scale)
        exp_results["53"] = exp53
        n_graphs_total += int(exp53["n_cases"])
        n_extractions_total += int(exp53["n_cases"]) * 2
        print(
            "[exp53] done:",
            {
                "segment_valid_rate": round(exp53["segmentwise_valid_rate"], 3),
                "whole_valid_rate": round(exp53["whole_sequence_valid_rate"], 3),
            },
            flush=True,
        )

    if "54" in selected:
        print("[exp54] running deferred composition sweep...", flush=True)
        exp54 = run_exp54_deferred_composition(scale=scale)
        exp_results["54"] = exp54
        n_graphs_total += int(exp54["n_cases"])
        n_extractions_total += int(exp54["n_cases"]) * 4
        print(
            "[exp54] done:",
            {
                row["patience_f"]: round(row["valid_rate"], 3)
                for row in exp54["summary_by_patience"]
            },
            flush=True,
        )

    if "55" in selected:
        print("[exp55] running recursive depth validation...", flush=True)
        exp55 = run_exp55_recursive_depth(scale=scale)
        exp_results["55"] = exp55
        n_graphs_total += int(exp55["n_cases"])
        n_extractions_total += int(exp55["n_cases"]) * 2
        print(
            "[exp55] done:",
            {
                "cells": len(exp55["summary_cells"]),
                "all_ge_90pct": exp55["target_check"]["all_defined_cells_validity_retention_ge_90pct"],
            },
            flush=True,
        )

    if "56" in selected:
        if not exp56_context_rows:
            print(
                "[exp56] missing exp50 context rows; regenerating exp50 subset for absorbing seeds...",
                flush=True,
            )
            exp50, exp56_context_rows = run_exp50_context_state_formalization(scale=scale)
            exp_results.setdefault("50", exp50)
        print("[exp56] running absorbing ideal verification...", flush=True)
        exp56 = run_exp56_absorbing_ideal(scale=scale, exp50_context_rows=exp56_context_rows)
        exp_results["56"] = exp56
        n_graphs_total += int(exp56["n_absorbing_states_tested"])
        print(
            "[exp56] done:",
            {
                "checks": exp56["checks"],
                "violations": exp56["violations"],
            },
            flush=True,
        )

    if "57" in selected:
        print("[exp57] running closure property checks...", flush=True)
        exp57 = run_exp57_closure(scale=scale)
        exp_results["57"] = exp57
        n_graphs_total += int(exp57["n_samples"])
        n_extractions_total += int(exp57["n_samples"]) * 3
        print(
            "[exp57] done:",
            {
                name: round(row["closure_violation_rate"], 3)
                for name, row in exp57["operations"].items()
            },
            flush=True,
        )

    data = {
        "scale": asdict(scale),
        "experiments": exp_results,
    }
    metadata = ExperimentMetadata(
        name=args.output_name,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=int(n_graphs_total),
        n_extractions=int(n_extractions_total),
        seed_range=(0, max(0, int(n_graphs_total) - 1)),
        parameters={
            "scale": scale.name,
            "selected_experiments": selected,
        },
    )

    save_results(
        name=args.output_name,
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )
    print(f"[context-algebra] completed in {metadata.runtime_seconds:.2f}s", flush=True)
    print(f"[context-algebra] output: {OUTPUT_DIR / f'{args.output_name}.json'}", flush=True)


if __name__ == "__main__":
    main()
