"""Evaluate TP-conditioned RCSPP solver on known gap-constrained failures."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.extraction.tp_conditioned_solver import tp_conditioned_solve
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


DEFAULT_OUTPUT_NAME = "tp_solver_evaluation_m25"
DEFAULT_BASELINE_NAME = "tp_solver_evaluation"

FOCAL_ACTOR = "actor_0"
N_EVENTS = 200
N_ACTORS = 6

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40

TARGET_HASH_SEED = "1"


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == TARGET_HASH_SEED:
        return

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = TARGET_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])


def _sorted_focal_events(graph: CausalGraph, focal_actor: str) -> list[Event]:
    return sorted(
        [event for event in graph.events if focal_actor in event.actors],
        key=lambda event: (float(event.weight), -float(event.timestamp), event.id),
        reverse=True,
    )


def _turning_point_index(sequence: ExtractedSequence) -> int | None:
    for idx, phase in enumerate(sequence.phases):
        if phase == Phase.TURNING_POINT:
            return idx
    return None


def _seq_payload(seq: ExtractedSequence | None) -> dict:
    if seq is None:
        return {
            "valid": False,
            "score": None,
            "n_events": 0,
            "violations": ["no_sequence"],
            "event_ids": [],
            "turning_point_id": None,
            "turning_point_index": None,
            "n_development": 0,
            "metadata": {},
        }

    tp = seq.turning_point
    tp_idx = _turning_point_index(seq)
    return {
        "valid": bool(seq.valid),
        "score": float(seq.score),
        "n_events": len(seq.events),
        "violations": list(seq.violations),
        "event_ids": [event.id for event in seq.events],
        "turning_point_id": (None if tp is None else tp.id),
        "turning_point_index": (None if tp_idx is None else int(tp_idx)),
        "n_development": int(seq.n_development),
        "metadata": dict(seq.metadata),
    }


def _run_one_case(graph: CausalGraph, grammar: GrammarConfig, tp_solver_m: int) -> dict:
    results: dict[str, dict] = {}

    def run_timed(fn):
        start = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - start
        return value, float(elapsed)

    greedy, greedy_s = run_timed(
        lambda: greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
        )
    )
    results["greedy"] = {
        **_seq_payload(greedy),
        "runtime_seconds": float(greedy_s),
        "runtime_ms": float(greedy_s * 1000.0),
    }

    vag_span, vag_span_s = run_timed(
        lambda: viability_aware_greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
            gap_aware_viability=False,
            budget_aware=False,
        )
    )
    vag_span_seq, vag_span_diag = vag_span
    results["vag_span_only"] = {
        **_seq_payload(vag_span_seq),
        "runtime_seconds": float(vag_span_s),
        "runtime_ms": float(vag_span_s * 1000.0),
        "diagnostics": vag_span_diag,
    }

    vag_gap, vag_gap_s = run_timed(
        lambda: viability_aware_greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
            gap_aware_viability=True,
            budget_aware=False,
        )
    )
    vag_gap_seq, vag_gap_diag = vag_gap
    results["vag_gap_aware"] = {
        **_seq_payload(vag_gap_seq),
        "runtime_seconds": float(vag_gap_s),
        "runtime_ms": float(vag_gap_s * 1000.0),
        "diagnostics": vag_gap_diag,
    }

    bvag, bvag_s = run_timed(
        lambda: viability_aware_greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy=POOL_STRATEGY,
            n_anchors=N_ANCHORS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
            gap_aware_viability=True,
            budget_aware=True,
        )
    )
    bvag_seq, bvag_diag = bvag
    results["vag_budget_aware"] = {
        **_seq_payload(bvag_seq),
        "runtime_seconds": float(bvag_s),
        "runtime_ms": float(bvag_s * 1000.0),
        "diagnostics": bvag_diag,
    }

    tp_result, tp_s = run_timed(
        lambda: tp_conditioned_solve(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            M=int(tp_solver_m),
            max_gap=float(grammar.max_temporal_gap),
            pool_strategy=POOL_STRATEGY,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            injection_top_n=INJECTION_TOP_N,
        )
    )
    tp_seq, tp_diag = tp_result
    results["tp_conditioned_solver"] = {
        **_seq_payload(tp_seq),
        "runtime_seconds": float(tp_s),
        "runtime_ms": float(tp_s * 1000.0),
        "diagnostics": tp_diag,
    }

    exact, exact_s = run_timed(
        lambda: exact_oracle_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
        )
    )
    exact_seq, exact_diag = exact
    results["exact_oracle"] = {
        **_seq_payload(exact_seq),
        "runtime_seconds": float(exact_s),
        "runtime_ms": float(exact_s * 1000.0),
        "diagnostics": exact_diag,
    }

    focal_sorted = _sorted_focal_events(graph, FOCAL_ACTOR)
    focal_rank = {event.id: idx + 1 for idx, event in enumerate(focal_sorted)}
    exact_tp_id = results["exact_oracle"]["turning_point_id"]
    results["context"] = {
        "exact_tp_rank_among_focal": (
            None if exact_tp_id is None else focal_rank.get(str(exact_tp_id))
        ),
        "n_focal_events": len(focal_sorted),
        "tp_solver_M": int(tp_solver_m),
    }

    return results


def _aggregate(rows: list[dict], algo: str) -> dict:
    n = len(rows)
    valid_count = sum(1 for row in rows if bool(row["results"][algo]["valid"]))
    valid_scores = [
        float(row["results"][algo]["score"])
        for row in rows
        if bool(row["results"][algo]["valid"]) and row["results"][algo]["score"] is not None
    ]
    runtimes_s = [float(row["results"][algo]["runtime_seconds"]) for row in rows]
    return {
        "n_cases": int(n),
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / n) if n > 0 else 0.0,
        "mean_score_valid": (float(mean(valid_scores)) if valid_scores else None),
        "median_runtime_ms": (
            float(median(runtimes_s) * 1000.0) if runtimes_s else None
        ),
    }


def _timing_stats(rows: list[dict], algo: str) -> dict:
    values = [float(row["results"][algo]["runtime_seconds"]) for row in rows]
    if not values:
        return {
            "median_seconds": None,
            "max_seconds": None,
            "p95_seconds": None,
        }
    return {
        "median_seconds": float(median(values)),
        "max_seconds": float(max(values)),
        "p95_seconds": float(_quantile(values, 0.95)),
    }


def _tp_label_stats(rows: list[dict]) -> dict:
    values: list[int] = []
    for row in rows:
        diag = row["results"]["tp_conditioned_solver"].get("diagnostics", {})
        total = diag.get("total_labels_generated")
        if total is None:
            meta = row["results"]["tp_conditioned_solver"].get("metadata", {})
            total = meta.get("total_labels_generated")
        if total is not None:
            values.append(int(total))

    if not values:
        return {
            "median_total_labels_generated": None,
            "max_total_labels_generated": None,
        }
    return {
        "median_total_labels_generated": int(median(values)),
        "max_total_labels_generated": int(max(values)),
    }


def _diagnose_tp_failure(row: dict, tp_solver_m: int) -> str:
    tp = row["results"]["tp_conditioned_solver"]
    exact = row["results"]["exact_oracle"]
    diag = tp.get("diagnostics", {})
    exact_tp_id = exact.get("turning_point_id")
    exact_tp_rank = row["results"]["context"].get("exact_tp_rank_among_focal")
    exact_tp_index = exact.get("turning_point_index")

    if exact_tp_rank is not None and int(exact_tp_rank) > int(tp_solver_m):
        return "M_too_small_exact_tp_outside_top_M"
    if exact_tp_id is not None and exact_tp_id not in set(diag.get("tp_candidate_ids", [])):
        return "exact_tp_not_in_enumerated_candidates"

    tp_summaries = diag.get("tp_summaries", [])
    exact_tp_summary = None
    for summary in tp_summaries:
        if summary.get("tp_id") == exact_tp_id:
            exact_tp_summary = summary
            break

    if exact_tp_summary is None:
        return "exact_tp_missing_from_solver_tp_summaries"

    if exact_tp_index is not None:
        n_pre_min = int(exact_tp_summary.get("n_pre_min", 0))
        n_pre_max = int(exact_tp_summary.get("n_pre_max", -1))
        if int(exact_tp_index) < n_pre_min or int(exact_tp_index) > n_pre_max:
            return "n_pre_enumeration_miss"

    if int(exact_tp_summary.get("dp_runs", 0)) == 0:
        return "no_dp_runs_for_exact_tp"

    total_rejections = int(diag.get("total_labels_rejected_dominance", 0))
    total_pruned = int(diag.get("total_labels_pruned_dominance", 0))
    total_kept = int(diag.get("total_labels_kept", 0))
    if (total_rejections + total_pruned) > (8 * max(1, total_kept)):
        return "possible_label_pruning_error"

    return "pool_or_transition_constraint_mismatch"


def _recovery_analysis(rows: list[dict], tp_solver_m: int) -> dict:
    bvag_fail_tp_success = [
        row
        for row in rows
        if (not bool(row["results"]["vag_budget_aware"]["valid"]))
        and bool(row["results"]["tp_conditioned_solver"]["valid"])
    ]
    bvag_fail_exact_valid = [
        row
        for row in rows
        if (not bool(row["results"]["vag_budget_aware"]["valid"]))
        and bool(row["results"]["exact_oracle"]["valid"])
    ]
    tp_fail_exact_valid = [
        row
        for row in rows
        if (not bool(row["results"]["tp_conditioned_solver"]["valid"]))
        and bool(row["results"]["exact_oracle"]["valid"])
    ]
    tp_regressions_vs_bvag = [
        row
        for row in rows
        if bool(row["results"]["vag_budget_aware"]["valid"])
        and (not bool(row["results"]["tp_conditioned_solver"]["valid"]))
    ]

    by_rank: dict[str, int] = {}
    labels_generated: list[int] = []
    frontier_peaks: list[int] = []
    recovered_samples: list[dict] = []

    for row in bvag_fail_tp_success:
        solver = row["results"]["tp_conditioned_solver"]
        meta = solver.get("metadata", {})
        diag = solver.get("diagnostics", {})
        tp_rank = meta.get("best_tp_rank")
        if tp_rank is None:
            tp_rank = diag.get("best_tp_rank")
        rank_key = str(tp_rank) if tp_rank is not None else "unknown"
        by_rank[rank_key] = by_rank.get(rank_key, 0) + 1

        lg = diag.get("total_labels_generated", meta.get("total_labels_generated"))
        if lg is not None:
            labels_generated.append(int(lg))
        fp = diag.get("best_frontier_peak", meta.get("frontier_peak"))
        if fp is not None:
            frontier_peaks.append(int(fp))

        if len(recovered_samples) < 20:
            recovered_samples.append(
                {
                    "seed": row.get("seed"),
                    "epsilon": row.get("epsilon"),
                    "tp_candidate_id": meta.get("best_tp_id", diag.get("best_tp_id")),
                    "tp_candidate_rank": tp_rank,
                    "labels_generated": (None if lg is None else int(lg)),
                    "frontier_peak": (None if fp is None else int(fp)),
                }
            )

    failure_diagnosis_counts: dict[str, int] = {}
    failure_samples: list[dict] = []
    for row in tp_fail_exact_valid:
        reason = _diagnose_tp_failure(row, tp_solver_m=tp_solver_m)
        failure_diagnosis_counts[reason] = failure_diagnosis_counts.get(reason, 0) + 1
        if len(failure_samples) < 30:
            failure_samples.append(
                {
                    "seed": row.get("seed"),
                    "epsilon": row.get("epsilon"),
                    "reason": reason,
                    "exact_tp_id": row["results"]["exact_oracle"].get("turning_point_id"),
                    "exact_tp_rank": row["results"]["context"].get("exact_tp_rank_among_focal"),
                    "exact_tp_index": row["results"]["exact_oracle"].get("turning_point_index"),
                    "tp_solver_diag": row["results"]["tp_conditioned_solver"].get("diagnostics", {}),
                }
            )

    return {
        "bvag_fail_exact_valid_count": int(len(bvag_fail_exact_valid)),
        "bvag_fail_tp_success_count": int(len(bvag_fail_tp_success)),
        "tp_recovery_rate_over_bvag_fail_exact_valid": (
            float(len(bvag_fail_tp_success) / len(bvag_fail_exact_valid))
            if bvag_fail_exact_valid
            else 0.0
        ),
        "tp_regressions_vs_bvag_count": int(len(tp_regressions_vs_bvag)),
        "tp_candidate_rank_distribution_on_recoveries": by_rank,
        "mean_labels_generated_on_recoveries": (
            float(mean(labels_generated)) if labels_generated else None
        ),
        "mean_frontier_peak_on_recoveries": (
            float(mean(frontier_peaks)) if frontier_peaks else None
        ),
        "recovered_case_samples": recovered_samples,
        "tp_fail_exact_valid_count": int(len(tp_fail_exact_valid)),
        "tp_fail_exact_valid_diagnosis_counts": failure_diagnosis_counts,
        "tp_fail_exact_valid_samples": failure_samples,
    }


def _run_multiburst_regime(grammar: GrammarConfig, tp_solver_m: int) -> list[dict]:
    generator = MultiBurstGenerator()
    rows: list[dict] = []

    for seed in range(0, 50):
        graph = generator.generate(
            MultiBurstConfig(
                seed=seed,
                n_events=N_EVENTS,
                n_actors=N_ACTORS,
            )
        )
        rows.append(
            {
                "seed": int(seed),
                "results": _run_one_case(graph, grammar, tp_solver_m=tp_solver_m),
            }
        )
        print(f"[multiburst] seed={seed} complete")

    return rows


def _run_bursty_regime(grammar: GrammarConfig, tp_solver_m: int) -> list[dict]:
    generator = BurstyGenerator()
    rows: list[dict] = []

    epsilon = 0.5
    for seed in range(0, 150):
        graph = generator.generate(
            BurstyConfig(
                seed=seed,
                epsilon=epsilon,
                n_events=N_EVENTS,
                n_actors=N_ACTORS,
            )
        )
        rows.append(
            {
                "epsilon": float(epsilon),
                "seed": int(seed),
                "results": _run_one_case(graph, grammar, tp_solver_m=tp_solver_m),
            }
        )
        if seed % 10 == 0:
            print(f"[bursty] seed={seed} complete")

    return rows


def _format_table(aggregate: dict) -> list[str]:
    algo_order = [
        "greedy",
        "vag_span_only",
        "vag_gap_aware",
        "vag_budget_aware",
        "tp_conditioned_solver",
        "exact_oracle",
    ]
    display = {
        "greedy": "Greedy",
        "vag_span_only": "Span-VAG",
        "vag_gap_aware": "Gap-aware VAG",
        "vag_budget_aware": "BVAG",
        "tp_conditioned_solver": "TP-conditioned solver",
        "exact_oracle": "Exact oracle",
    }

    lines = [
        "| Algorithm | Validity Rate | Mean Score (valid) | Median Runtime (ms) |",
        "|---|---:|---:|---:|",
    ]
    for algo in algo_order:
        row = aggregate[algo]
        mean_score = (
            f"{row['mean_score_valid']:.4f}"
            if row["mean_score_valid"] is not None
            else "n/a"
        )
        runtime = (
            f"{row['median_runtime_ms']:.2f}"
            if row["median_runtime_ms"] is not None
            else "n/a"
        )
        lines.append(
            f"| {display[algo]} | {row['valid_rate']:.3f} | {mean_score} | {runtime} |"
        )
    return lines


def _format_timing_table(timing: dict, label_stats: dict) -> list[str]:
    algo_order = [
        "greedy",
        "vag_span_only",
        "vag_gap_aware",
        "vag_budget_aware",
        "tp_conditioned_solver",
        "exact_oracle",
    ]
    lines = [
        "| Algorithm | median_seconds | p95_seconds | max_seconds |",
        "|---|---:|---:|---:|",
    ]
    for algo in algo_order:
        row = timing.get(algo, {})
        med = row.get("median_seconds")
        p95 = row.get("p95_seconds")
        mx = row.get("max_seconds")
        lines.append(
            f"| {algo} | "
            f"{('n/a' if med is None else f'{med:.4f}')} | "
            f"{('n/a' if p95 is None else f'{p95:.4f}')} | "
            f"{('n/a' if mx is None else f'{mx:.4f}')} |"
        )
    lines.append("")
    lines.append(
        "- TP-solver total_labels_generated: "
        f"median={label_stats.get('median_total_labels_generated')}, "
        f"max={label_stats.get('max_total_labels_generated')}"
    )
    return lines


def _case_key(topology: str, row: dict) -> str:
    seed = int(row.get("seed"))
    if topology == "bursty_gap":
        epsilon = float(row.get("epsilon", 0.5))
        return f"{topology}|{epsilon:.1f}|{seed}"
    return f"{topology}|{seed}"


def _build_case_lookup(topology: str, rows: list[dict]) -> dict[str, dict]:
    return {_case_key(topology, row): row for row in rows}


def _baseline_impact(new_topologies: dict, baseline_path: Path) -> dict:
    if not baseline_path.exists():
        return {
            "baseline_available": False,
            "baseline_path": str(baseline_path),
            "m_too_small_recovery": {
                "multiburst_recovered": None,
                "multiburst_total": None,
                "bursty_recovered": None,
                "bursty_total": None,
                "newly_recovered_case_details": [],
            },
            "regressions_vs_m10": {
                "count": None,
                "cases": [],
            },
        }

    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = payload["results"]["topologies"]

    new_lookup = {
        "multiburst_gap": _build_case_lookup("multiburst_gap", new_topologies["multiburst_gap"]["per_case"]),
        "bursty_gap": _build_case_lookup("bursty_gap", new_topologies["bursty_gap"]["per_case"]),
    }

    baseline_lookup = {
        "multiburst_gap": _build_case_lookup("multiburst_gap", baseline["multiburst_gap"]["per_case"]),
        "bursty_gap": _build_case_lookup("bursty_gap", baseline["bursty_gap"]["per_case"]),
    }

    m_too_small_cases: list[dict] = []
    for topo in ["multiburst_gap", "bursty_gap"]:
        samples = baseline[topo]["recovery_analysis"].get("tp_fail_exact_valid_samples", [])
        for sample in samples:
            if sample.get("reason") != "M_too_small_exact_tp_outside_top_M":
                continue
            key = _case_key(topo, sample)
            m_too_small_cases.append(
                {
                    "topology": topo,
                    "key": key,
                    "seed": sample.get("seed"),
                    "epsilon": sample.get("epsilon"),
                }
            )

    newly_recovered: list[dict] = []
    multiburst_total = 0
    multiburst_recovered = 0
    bursty_total = 0
    bursty_recovered = 0
    for case in m_too_small_cases:
        topo = case["topology"]
        row = new_lookup[topo].get(case["key"])
        if row is None:
            continue
        solved = bool(row["results"]["tp_conditioned_solver"]["valid"])
        if topo == "multiburst_gap":
            multiburst_total += 1
            multiburst_recovered += int(solved)
        else:
            bursty_total += 1
            bursty_recovered += int(solved)
        if solved:
            solver = row["results"]["tp_conditioned_solver"]
            meta = solver.get("metadata", {})
            diag = solver.get("diagnostics", {})
            newly_recovered.append(
                {
                    "topology": topo,
                    "seed": int(case["seed"]),
                    "epsilon": (None if case.get("epsilon") is None else float(case["epsilon"])),
                    "tp_id": meta.get("best_tp_id", diag.get("best_tp_id")),
                    "tp_weight_rank": meta.get("best_tp_rank", diag.get("best_tp_rank")),
                }
            )

    regressions: list[dict] = []
    for topo in ["multiburst_gap", "bursty_gap"]:
        for key, row in baseline_lookup[topo].items():
            baseline_solved = bool(row["results"]["tp_conditioned_solver"]["valid"])
            if not baseline_solved:
                continue
            now = new_lookup[topo].get(key)
            now_solved = bool(now and now["results"]["tp_conditioned_solver"]["valid"])
            if not now_solved:
                regressions.append(
                    {
                        "topology": topo,
                        "seed": int(row.get("seed")),
                        "epsilon": row.get("epsilon"),
                        "case_key": key,
                    }
                )

    return {
        "baseline_available": True,
        "baseline_path": str(baseline_path),
        "m_too_small_recovery": {
            "multiburst_recovered": int(multiburst_recovered),
            "multiburst_total": int(multiburst_total),
            "bursty_recovered": int(bursty_recovered),
            "bursty_total": int(bursty_total),
            "newly_recovered_case_details": newly_recovered,
        },
        "regressions_vs_m10": {
            "count": int(len(regressions)),
            "cases": regressions,
        },
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    lines: list[str] = [
        "# tp_solver_evaluation_m25",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
    ]

    for key in ["multiburst_gap", "bursty_gap"]:
        regime = data["topologies"][key]
        lines.append(f"## {regime['label']}")
        lines.append("")
        lines.extend(_format_table(regime["aggregate"]))
        lines.append("")
        recovery = regime["recovery_analysis"]
        lines.append(
            "- BVAG failures where exact is valid: "
            f"{recovery['bvag_fail_exact_valid_count']}"
        )
        lines.append(
            "- TP-solver recoveries among those: "
            f"{recovery['bvag_fail_tp_success_count']}/"
            f"{recovery['bvag_fail_exact_valid_count']}"
        )
        lines.append(
            "- TP-solver regressions vs BVAG: "
            f"{recovery['tp_regressions_vs_bvag_count']}"
        )
        lines.append(
            "- TP rank distribution on recovered cases: "
            f"{recovery['tp_candidate_rank_distribution_on_recoveries']}"
        )
        lines.append(
            "- Mean labels generated on recovered cases: "
            + (
                f"{recovery['mean_labels_generated_on_recoveries']:.2f}"
                if recovery["mean_labels_generated_on_recoveries"] is not None
                else "n/a"
            )
        )
        lines.append(
            "- Mean frontier peak on recovered cases: "
            + (
                f"{recovery['mean_frontier_peak_on_recoveries']:.2f}"
                if recovery["mean_frontier_peak_on_recoveries"] is not None
                else "n/a"
            )
        )
        lines.append(
            "- TP-solver failures where exact succeeds: "
            f"{recovery['tp_fail_exact_valid_count']}"
        )
        lines.append(
            "- Failure diagnosis counts: "
            f"{recovery['tp_fail_exact_valid_diagnosis_counts']}"
        )
        lines.append("")
        lines.append("### Per-instance Timing")
        lines.append("")
        lines.extend(_format_timing_table(regime["timing"], regime["tp_solver_labels"]))
        lines.append("")

    lines.append("## M=25 Impact vs M=10")
    lines.append("")
    impact = data["overall"]["m25_vs_m10"]
    if not impact["baseline_available"]:
        lines.append(f"- Baseline not found: `{impact['baseline_path']}`")
    else:
        mrec = impact["m_too_small_recovery"]
        lines.append(
            "- Multi-burst M-too-small recovered: "
            f"{mrec['multiburst_recovered']}/{mrec['multiburst_total']}"
        )
        lines.append(
            "- Bursty M-too-small recovered: "
            f"{mrec['bursty_recovered']}/{mrec['bursty_total']}"
        )
        lines.append(
            "- Newly recovered TP ranks: "
            f"{[{k: v for k, v in case.items() if k in ['topology', 'seed', 'epsilon', 'tp_weight_rank']} for case in mrec['newly_recovered_case_details']]}"
        )
        reg = impact["regressions_vs_m10"]
        lines.append(f"- Regressions vs M=10 solved set: {reg['count']}")

    lines.append("")
    lines.append("## Prediction Check")
    lines.append("")
    lines.append(
        f"- Multi-burst TP-solver recovered (expected 19-20/20): "
        f"{data['topologies']['multiburst_gap']['recovery_analysis']['bvag_fail_tp_success_count']}/"
        f"{data['topologies']['multiburst_gap']['recovery_analysis']['bvag_fail_exact_valid_count']}"
    )
    lines.append(
        f"- Bursty TP-solver recovered (expected 28-33/33): "
        f"{data['topologies']['bursty_gap']['recovery_analysis']['bvag_fail_tp_success_count']}/"
        f"{data['topologies']['bursty_gap']['recovery_analysis']['bvag_fail_exact_valid_count']}"
    )
    lines.append(
        "- Zero regressions (TP-solver not worse than BVAG): "
        f"{data['overall']['tp_not_worse_than_bvag']}"
    )
    lines.append("")
    return "\n".join(lines)


def run_tp_solver_evaluation(
    *,
    tp_solver_m: int = 25,
    output_name: str = DEFAULT_OUTPUT_NAME,
    baseline_output_name: str = DEFAULT_BASELINE_NAME,
) -> dict:
    _ensure_hash_seed()

    timer = ExperimentTimer()

    grammar_multiburst = GrammarConfig(
        min_prefix_elements=1,
        min_timespan_fraction=0.3,
        max_temporal_gap=0.4000,
    )
    grammar_bursty = GrammarConfig(
        min_prefix_elements=1,
        min_timespan_fraction=0.3,
        max_temporal_gap=0.1402,
    )

    multiburst_rows = _run_multiburst_regime(grammar_multiburst, tp_solver_m=tp_solver_m)
    bursty_rows = _run_bursty_regime(grammar_bursty, tp_solver_m=tp_solver_m)

    def summarize(rows: list[dict], label: str, parameters: dict) -> dict:
        algo_order = [
            "greedy",
            "vag_span_only",
            "vag_gap_aware",
            "vag_budget_aware",
            "tp_conditioned_solver",
            "exact_oracle",
        ]
        aggregate = {algo: _aggregate(rows, algo) for algo in algo_order}
        timing = {algo: _timing_stats(rows, algo) for algo in algo_order}
        recovery = _recovery_analysis(rows, tp_solver_m=tp_solver_m)
        tp_solver_labels = _tp_label_stats(rows)
        return {
            "label": label,
            "parameters": parameters,
            "n_cases": int(len(rows)),
            "aggregate": aggregate,
            "timing": timing,
            "tp_solver_labels": tp_solver_labels,
            "recovery_analysis": recovery,
            "per_case": rows,
        }

    topologies = {
        "multiburst_gap": summarize(
            rows=multiburst_rows,
            label="Multi-burst + gap (Exp 26 regime)",
            parameters={
                "generator": "MultiBurstGenerator(defaults)",
                "seed_range": [0, 49],
                "n_events": N_EVENTS,
                "n_actors": N_ACTORS,
                "focal_actor": FOCAL_ACTOR,
                "grammar": {
                    "min_prefix_elements": grammar_multiburst.min_prefix_elements,
                    "min_timespan_fraction": grammar_multiburst.min_timespan_fraction,
                    "max_temporal_gap": float(grammar_multiburst.max_temporal_gap),
                },
            },
        ),
        "bursty_gap": summarize(
            rows=bursty_rows,
            label="Bursty + gap (Exp 23 regime)",
            parameters={
                "generator": "BurstyGenerator",
                "epsilon": 0.5,
                "seed_range": [0, 149],
                "n_events": N_EVENTS,
                "n_actors": N_ACTORS,
                "focal_actor": FOCAL_ACTOR,
                "grammar": {
                    "min_prefix_elements": grammar_bursty.min_prefix_elements,
                    "min_timespan_fraction": grammar_bursty.min_timespan_fraction,
                    "max_temporal_gap": float(grammar_bursty.max_temporal_gap),
                },
            },
        ),
    }

    multiburst_regr = topologies["multiburst_gap"]["recovery_analysis"]["tp_regressions_vs_bvag_count"]
    bursty_regr = topologies["bursty_gap"]["recovery_analysis"]["tp_regressions_vs_bvag_count"]

    baseline_path = Path(__file__).resolve().parent / "output" / f"{baseline_output_name}.json"
    m25_vs_m10 = _baseline_impact(topologies, baseline_path=baseline_path)
    regressions_vs_m10 = int(m25_vs_m10["regressions_vs_m10"].get("count") or 0)
    if m25_vs_m10["baseline_available"] and regressions_vs_m10 > 0:
        raise RuntimeError(
            "Regression check failed: some M=10 solved cases are not solved at M=25: "
            f"{m25_vs_m10['regressions_vs_m10']['cases']}"
        )

    overall = {
        "tp_not_worse_than_bvag": bool((multiburst_regr + bursty_regr) == 0),
        "total_tp_regressions_vs_bvag": int(multiburst_regr + bursty_regr),
        "total_cases": int(len(multiburst_rows) + len(bursty_rows)),
        "m25_vs_m10": m25_vs_m10,
    }

    results = {
        "settings": {
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": int(N_ANCHORS),
            "max_sequence_length": int(MAX_SEQUENCE_LENGTH),
            "injection_top_n": int(INJECTION_TOP_N),
            "tp_solver_M": int(tp_solver_m),
            "baseline_output_name": baseline_output_name,
        },
        "topologies": topologies,
        "overall": overall,
    }

    metadata = ExperimentMetadata(
        name=output_name,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=int(len(multiburst_rows) + len(bursty_rows)),
        n_extractions=int((len(multiburst_rows) + len(bursty_rows)) * 6),
        seed_range=(0, 149),
        parameters={
            "topologies": ["multiburst_gap", "bursty_gap"],
            "tp_solver_M": int(tp_solver_m),
            "focal_actor": FOCAL_ACTOR,
            "output_name": output_name,
        },
    )
    save_results(
        name=output_name,
        data=results,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=25, help="Top-M focal TP candidates")
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output basename under experiments/output",
    )
    parser.add_argument(
        "--baseline-output-name",
        type=str,
        default=DEFAULT_BASELINE_NAME,
        help="Baseline run basename for no-regression comparison",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_tp_solver_evaluation(
        tp_solver_m=int(args.m),
        output_name=str(args.output_name),
        baseline_output_name=str(args.baseline_output_name),
    )
