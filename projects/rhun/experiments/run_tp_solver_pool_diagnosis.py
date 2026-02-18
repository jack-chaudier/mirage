"""Diagnose TP-solver pool/transition mismatch cases from M=10 run."""

from __future__ import annotations

import json
from pathlib import Path

from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.pool_construction import injection_pool
from rhun.extraction.scoring import tp_weighted_score
from rhun.extraction.tp_conditioned_solver import _solve_fixed_tau_and_n_pre
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.schemas import CausalGraph, Event, ExtractedSequence, Phase


M10_PATH = Path(__file__).resolve().parent / "output" / "tp_solver_evaluation.json"
M25_PATH = Path(__file__).resolve().parent / "output" / "tp_solver_evaluation_m25.json"
OUTPUT_MD = Path(__file__).resolve().parent / "output" / "tp_solver_pool_diagnosis.md"

FOCAL_ACTOR = "actor_0"
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40


def _turning_point_index(sequence: ExtractedSequence) -> int | None:
    for idx, phase in enumerate(sequence.phases):
        if phase == Phase.TURNING_POINT:
            return idx
    return None


def _sorted_focal(graph: CausalGraph) -> list[Event]:
    return sorted(
        [event for event in graph.events if FOCAL_ACTOR in event.actors],
        key=lambda event: (float(event.weight), -float(event.timestamp), event.id),
        reverse=True,
    )


def _case_key(topology: str, seed: int, epsilon: float | None) -> str:
    if topology == "bursty_gap":
        return f"{topology}|{float(epsilon):.1f}|{int(seed)}"
    return f"{topology}|{int(seed)}"


def _build_lookup(topology: str, rows: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in rows:
        key = _case_key(topology, int(row["seed"]), row.get("epsilon"))
        lookup[key] = row
    return lookup


def _grammar_for(topology: str) -> GrammarConfig:
    if topology == "multiburst_gap":
        return GrammarConfig(
            min_prefix_elements=1,
            min_timespan_fraction=0.3,
            max_temporal_gap=0.4000,
        )
    return GrammarConfig(
        min_prefix_elements=1,
        min_timespan_fraction=0.3,
        max_temporal_gap=0.1402,
    )


def _graph_for(topology: str, seed: int, epsilon: float | None) -> CausalGraph:
    if topology == "multiburst_gap":
        return MultiBurstGenerator().generate(
            MultiBurstConfig(seed=int(seed), n_events=200, n_actors=6)
        )
    return BurstyGenerator().generate(
        BurstyConfig(seed=int(seed), epsilon=float(epsilon), n_events=200, n_actors=6)
    )


def _exact_sequence(graph: CausalGraph, grammar: GrammarConfig) -> ExtractedSequence:
    seq, _diag = exact_oracle_extract(graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar)
    if not seq.valid:
        raise RuntimeError("Exact oracle did not produce valid sequence")
    return seq


def _best_attempt_same_tp(
    graph: CausalGraph,
    grammar: GrammarConfig,
    exact_tp: Event,
    exact_tp_rank: int,
) -> dict:
    by_id = {event.id: event for event in graph.events}
    pool_ids = injection_pool(
        graph=graph,
        anchor_id=exact_tp.id,
        focal_actor=FOCAL_ACTOR,
        max_depth=3,
        injection_top_n=INJECTION_TOP_N,
    )
    pool_ids.add(exact_tp.id)
    pool_events = [by_id[event_id] for event_id in pool_ids if event_id in by_id]
    pool_events = sorted(pool_events, key=lambda event: (float(event.timestamp), event.id))

    tau_timestamp = float(exact_tp.timestamp)
    pre_available = sum(1 for event in pool_events if float(event.timestamp) < tau_timestamp)
    n_pre_min = max(grammar.min_prefix_elements, 0)
    n_pre_max = min(pre_available, max(0, MAX_SEQUENCE_LENGTH - 1))

    best_valid: ExtractedSequence | None = None
    best_valid_score = float("-inf")
    best_valid_n_pre: int | None = None

    best_any: dict | None = None

    run_diags: list[dict] = []
    if n_pre_min <= n_pre_max:
        for n_pre in range(n_pre_min, n_pre_max + 1):
            candidate, diag = _solve_fixed_tau_and_n_pre(
                graph=graph,
                focal_actor=FOCAL_ACTOR,
                grammar=grammar,
                tau=exact_tp,
                tau_rank=exact_tp_rank,
                pool_events=pool_events,
                n_pre=n_pre,
                sequence_length_budget=MAX_SEQUENCE_LENGTH,
                active_max_gap=float(grammar.max_temporal_gap),
                scoring_fn=tp_weighted_score,
            )
            run_diags.append(diag)

            best_any_score = diag.get("best_any_score")
            if best_any_score is not None:
                if best_any is None or float(best_any_score) > float(best_any["score"]):
                    best_any = {
                        "n_pre": int(n_pre),
                        "score": float(best_any_score),
                        "valid": bool(diag.get("best_any_valid", False)),
                        "violations": list(diag.get("best_any_violations") or []),
                        "event_ids": list(diag.get("best_any_event_ids") or []),
                        "phases": list(diag.get("best_any_phases") or []),
                    }

            if candidate is not None and candidate.valid and float(candidate.score) > best_valid_score:
                best_valid = candidate
                best_valid_score = float(candidate.score)
                best_valid_n_pre = int(n_pre)

    return {
        "pool_ids": pool_ids,
        "pre_available": int(pre_available),
        "n_pre_min": int(n_pre_min),
        "n_pre_max": int(n_pre_max),
        "best_valid": best_valid,
        "best_valid_n_pre": best_valid_n_pre,
        "best_any": best_any,
        "run_diags": run_diags,
    }


def _classification(
    *,
    tp_in_candidate_set: bool,
    exact_seq_in_pool: bool,
    n_pre_gap: bool,
    tp_summary: dict | None,
) -> str:
    if not tp_in_candidate_set:
        return "exact_tp_not_in_candidate_set"
    if not exact_seq_in_pool:
        return "pool_construction_difference"
    if n_pre_gap:
        return "n_pre_enumeration_gap"

    if tp_summary is not None:
        pruned = int(tp_summary.get("labels_pruned_dominance", 0))
        kept = int(tp_summary.get("labels_kept", 0))
        if kept > 0 and pruned > (5 * kept):
            return "likely_label_pruning_too_aggressive"

    return "transition_or_constraint_mismatch_under_fixed_tp"


def run_tp_solver_pool_diagnosis() -> dict:
    if not M10_PATH.exists():
        raise FileNotFoundError(f"Missing baseline file: {M10_PATH}")
    if not M25_PATH.exists():
        raise FileNotFoundError(f"Missing M25 file: {M25_PATH}")

    m10 = json.loads(M10_PATH.read_text(encoding="utf-8"))["results"]
    m25 = json.loads(M25_PATH.read_text(encoding="utf-8"))["results"]

    target_cases: list[dict] = []
    for topology in ["multiburst_gap", "bursty_gap"]:
        samples = m10["topologies"][topology]["recovery_analysis"].get("tp_fail_exact_valid_samples", [])
        for sample in samples:
            if sample.get("reason") != "pool_or_transition_constraint_mismatch":
                continue
            target_cases.append(
                {
                    "topology": topology,
                    "seed": int(sample["seed"]),
                    "epsilon": (None if sample.get("epsilon") is None else float(sample["epsilon"])),
                }
            )

    m25_lookup = {
        "multiburst_gap": _build_lookup("multiburst_gap", m25["topologies"]["multiburst_gap"]["per_case"]),
        "bursty_gap": _build_lookup("bursty_gap", m25["topologies"]["bursty_gap"]["per_case"]),
    }

    rows: list[dict] = []
    for case in target_cases:
        topology = case["topology"]
        seed = int(case["seed"])
        epsilon = case["epsilon"]
        key = _case_key(topology, seed, epsilon)

        m25_row = m25_lookup[topology].get(key)
        if m25_row is None:
            raise RuntimeError(f"Missing case in M25 output: {key}")

        grammar = _grammar_for(topology)
        graph = _graph_for(topology, seed, epsilon)
        exact = _exact_sequence(graph, grammar)
        exact_tp = exact.turning_point
        if exact_tp is None:
            raise RuntimeError("Exact oracle sequence missing TP")

        focal_sorted = _sorted_focal(graph)
        focal_rank = {event.id: idx + 1 for idx, event in enumerate(focal_sorted)}
        exact_tp_rank = int(focal_rank[exact_tp.id])

        solver_diag = m25_row["results"]["tp_conditioned_solver"].get("diagnostics", {})
        tp_candidates = set(str(event_id) for event_id in solver_diag.get("tp_candidate_ids", []))
        tp_in_candidate_set = bool(exact_tp.id in tp_candidates)

        tp_summary = None
        for summary in solver_diag.get("tp_summaries", []):
            if summary.get("tp_id") == exact_tp.id:
                tp_summary = summary
                break

        best_attempt = _best_attempt_same_tp(
            graph=graph,
            grammar=grammar,
            exact_tp=exact_tp,
            exact_tp_rank=exact_tp_rank,
        )

        exact_event_ids = [event.id for event in exact.events]
        exact_seq_in_pool = all(event_id in best_attempt["pool_ids"] for event_id in exact_event_ids)

        exact_tp_idx = _turning_point_index(exact)
        if exact_tp_idx is None:
            raise RuntimeError("Exact oracle sequence missing TP index")

        n_pre_min = int(best_attempt["n_pre_min"])
        n_pre_max = int(best_attempt["n_pre_max"])
        n_pre_gap = bool(exact_tp_idx < n_pre_min or exact_tp_idx > n_pre_max)

        prevent_reason = _classification(
            tp_in_candidate_set=tp_in_candidate_set,
            exact_seq_in_pool=exact_seq_in_pool,
            n_pre_gap=n_pre_gap,
            tp_summary=tp_summary,
        )

        best_valid = best_attempt["best_valid"]
        best_any = best_attempt["best_any"]

        rows.append(
            {
                "topology": topology,
                "seed": seed,
                "epsilon": epsilon,
                "exact_tp": {
                    "id": exact_tp.id,
                    "weight": float(exact_tp.weight),
                    "timestamp": float(exact_tp.timestamp),
                    "weight_rank_focal": exact_tp_rank,
                },
                "tp_in_candidate_set_m25": tp_in_candidate_set,
                "exact_sequence_in_tp_pool": exact_seq_in_pool,
                "exact_tp_index": int(exact_tp_idx),
                "solver_n_pre_min": int(n_pre_min),
                "solver_n_pre_max": int(n_pre_max),
                "n_pre_enumeration_gap": n_pre_gap,
                "prevent_reason": prevent_reason,
                "exact_sequence": {
                    "event_ids": [event.id for event in exact.events],
                    "phases": [phase.name for phase in exact.phases],
                },
                "solver_best_attempt_same_tp": {
                    "best_valid_found": bool(best_valid is not None),
                    "best_valid_n_pre": best_attempt["best_valid_n_pre"],
                    "best_valid": (
                        None
                        if best_valid is None
                        else {
                            "score": float(best_valid.score),
                            "event_ids": [event.id for event in best_valid.events],
                            "phases": [phase.name for phase in best_valid.phases],
                            "violations": list(best_valid.violations),
                        }
                    ),
                    "best_any": best_any,
                },
            }
        )

    lines: list[str] = [
        "# TP Solver Pool/Transition Diagnosis (M=25)",
        "",
        f"Source M10: `{M10_PATH}`",
        f"Source M25: `{M25_PATH}`",
        "",
        f"Total diagnosed cases: {len(rows)}",
        "",
    ]

    for idx, row in enumerate(rows, start=1):
        topo = row["topology"]
        eps = row["epsilon"]
        eps_part = "" if eps is None else f", epsilon={eps:.1f}"
        lines.append(f"## Case {idx}: {topo}, seed={row['seed']}{eps_part}")
        lines.append("")
        tp = row["exact_tp"]
        lines.append(
            "1. exact oracle TP: "
            f"id={tp['id']}, weight={tp['weight']:.6f}, timestamp={tp['timestamp']:.6f}, "
            f"weight_rank_focal={tp['weight_rank_focal']}"
        )
        lines.append(
            "2. exact TP in TP-solver candidate set at M=25: "
            f"{row['tp_in_candidate_set_m25']}"
        )
        lines.append(
            "3. blocking diagnosis: "
            f"{row['prevent_reason']} "
            f"(exact_sequence_in_tp_pool={row['exact_sequence_in_tp_pool']}, "
            f"n_pre_exact={row['exact_tp_index']}, "
            f"solver_n_pre_range=[{row['solver_n_pre_min']}, {row['solver_n_pre_max']}])"
        )

        exact_seq = row["exact_sequence"]
        best_same_tp = row["solver_best_attempt_same_tp"]
        lines.append("4. exact oracle sequence (event_ids / phases):")
        lines.append(f"   - event_ids: {exact_seq['event_ids']}")
        lines.append(f"   - phases: {exact_seq['phases']}")
        lines.append("5. solver best attempt for same TP:")
        lines.append(
            f"   - best_valid_found={best_same_tp['best_valid_found']}, "
            f"best_valid_n_pre={best_same_tp['best_valid_n_pre']}"
        )
        lines.append(f"   - best_valid: {best_same_tp['best_valid']}")
        lines.append(f"   - best_any: {best_same_tp['best_any']}")
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved diagnosis to {OUTPUT_MD}")

    return {
        "n_cases": len(rows),
        "rows": rows,
    }


if __name__ == "__main__":
    run_tp_solver_pool_diagnosis()
