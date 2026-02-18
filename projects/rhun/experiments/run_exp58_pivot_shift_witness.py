"""Exp58: deterministic adversarial pivot-shift witness for compression contracts."""

from __future__ import annotations

import argparse
import os
import sys

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.tp_conditioned_solver import tp_conditioned_solve
from rhun.schemas import CausalGraph, Event
from rhun.theory.context_algebra import (
    build_context_state,
    compress_events,
    compose_context_states,
    context_equivalent,
    development_eligible_count,
)


FOCAL_ACTOR = "actor_0"
TARGET_HASH_SEED = "1"


def _parse_solver_m_list(raw: str) -> tuple[int, ...]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("solver M list cannot be empty")
    if any(value <= 0 for value in values):
        raise ValueError("solver M values must be >= 1")
    return tuple(values)


def _build_exp58_graph() -> CausalGraph:
    events: list[Event] = []
    previous_id: str | None = None

    def add_event(
        event_id: str,
        timestamp: float,
        weight: float,
        block: int,
        role: str,
    ) -> None:
        nonlocal previous_id
        parents = () if previous_id is None else (previous_id,)
        events.append(
            Event(
                id=event_id,
                timestamp=float(timestamp),
                weight=float(weight),
                actors=frozenset({FOCAL_ACTOR}),
                causal_parents=parents,
                metadata={"block": int(block), "role": str(role)},
            )
        )
        previous_id = event_id

    # Block 1: early local pivot with low-weight latent suffix capacity.
    add_event("e01", 0.020, 0.90, 1, "prefix_high")
    add_event("e02", 0.080, 0.85, 1, "prefix_high")
    add_event("e03", 0.120, 5.00, 1, "local_pivot")

    latent_times = [0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500]
    for idx, timestamp in enumerate(latent_times, start=4):
        add_event(f"e{idx:02d}", timestamp, 0.20 + 0.01 * (idx - 4), 1, "latent_low")

    timestamp = 0.510
    for idx in range(11, 41):
        add_event(
            f"e{idx:02d}",
            min(0.950, timestamp),
            0.60 + 0.20 * ((idx - 11) % 5) / 4.0,
            1,
            "filler_high",
        )
        timestamp += 0.0148

    # Block 2: higher pivot that forces right-branch composition.
    add_event("e41", 0.960, 10.00, 2, "global_pivot")
    timestamp = 0.965
    for idx in range(42, 51):
        add_event(
            f"e{idx:02d}",
            min(1.000, timestamp),
            0.40 + 0.20 * ((idx - 42) % 4) / 3.0,
            2,
            "post_pivot",
        )
        timestamp += 0.004

    ordered = tuple(sorted(events, key=lambda event: (event.timestamp, event.id)))
    return CausalGraph(
        events=ordered,
        actors=frozenset({FOCAL_ACTOR}),
        seed=58,
        metadata={"experiment": "exp58_pivot_shift_witness"},
    )


def _state_digest(state) -> dict:
    pivots = [
        {
            "event_id": str(candidate.event_id),
            "weight": float(candidate.weight),
            "timestamp": float(candidate.timestamp),
            "prefix_count": int(candidate.prefix_count),
            "j_dev": int(candidate.j_dev),
        }
        for candidate in state.pivots[:3]
    ]
    return {
        "q": str(state.q),
        "slots_used": int(state.r.slots_used),
        "dev_count": int(state.r.dev_count),
        "prefix_count": int(state.r.prefix_count),
        "setup_count": int(state.r.setup_count),
        "post_count": int(state.r.post_count),
        "t_bounds": (
            None if state.t_bounds[0] is None else float(state.t_bounds[0]),
            None if state.t_bounds[1] is None else float(state.t_bounds[1]),
        ),
        "top_pivots": pivots,
    }


def _solve(
    graph: CausalGraph,
    grammar: GrammarConfig,
    m_solver: int,
    forced_tp_id: str | None = None,
) -> dict:
    sequence, diagnostics = tp_conditioned_solve(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        M=int(m_solver),
        max_gap=float("inf"),
        pool_strategy="injection",
        max_sequence_length=20,
        injection_top_n=40,
        tp_candidate_ids=(str(forced_tp_id),) if forced_tp_id is not None else None,
    )
    return {
        "valid": bool(sequence is not None and sequence.valid),
        "score": None if sequence is None else float(sequence.score),
        "turning_point_id": (
            None if (sequence is None or sequence.turning_point is None) else str(sequence.turning_point.id)
        ),
        "violations": (None if sequence is None else list(sequence.violations)),
        "diagnostics": diagnostics,
    }


def run_exp58(
    k: int,
    target_size: int,
    compression_m: int,
    solver_m_list: tuple[int, ...],
) -> dict:
    graph = _build_exp58_graph()
    block1 = tuple(event for event in graph.events if int(event.metadata.get("block", 0)) == 1)
    block2 = tuple(event for event in graph.events if int(event.metadata.get("block", 0)) == 2)

    grammar = GrammarConfig(
        min_prefix_elements=int(k),
        max_phase_regressions=0,
        max_turning_points=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
        focal_actor_coverage=0.60,
    )

    full_solver = {str(m): _solve(graph=graph, grammar=grammar, m_solver=int(m)) for m in solver_m_list}
    full_top_pivot_by_m = {
        str(m): (
            full_solver[str(m)]["turning_point_id"]
            if full_solver[str(m)]["valid"]
            else None
        )
        for m in solver_m_list
    }
    block2_state = build_context_state(block2, focal_actor=FOCAL_ACTOR, M=int(compression_m))

    strategies: dict[str, dict] = {}
    for strategy in ("naive", "contract_guarded"):
        compressed_block1, compression_diag = compress_events(
            events=block1,
            focal_actor=FOCAL_ACTOR,
            k=int(k),
            M=int(compression_m),
            target_size=int(target_size),
            strategy=strategy,
            max_gap=float("inf"),
            min_length=grammar.min_length,
        )
        composed_events = tuple(sorted((*compressed_block1, *block2), key=lambda event: (event.timestamp, event.id)))
        composed_graph = CausalGraph(
            events=composed_events,
            actors=graph.actors,
            seed=graph.seed,
            metadata=dict(graph.metadata),
        )

        left_state = build_context_state(compressed_block1, focal_actor=FOCAL_ACTOR, M=int(compression_m))
        composed_state = compose_context_states(
            left=left_state,
            right=block2_state,
            M=int(compression_m),
        )
        composed_ground_truth = build_context_state(
            composed_events,
            focal_actor=FOCAL_ACTOR,
            M=int(compression_m),
        )

        left_top = None if not left_state.pivots else left_state.pivots[0]
        right_top = None if not block2_state.pivots else block2_state.pivots[0]
        composed_top = None if not composed_state.pivots else composed_state.pivots[0]

        pivot_shift_right_branch = bool(
            left_top is not None and right_top is not None and float(right_top.weight) > float(left_top.weight)
        )

        solver_free: dict[str, dict] = {}
        solver_fixed: dict[str, dict | None] = {}
        pivot_preservation: dict[str, bool] = {}
        semantic_regret: dict[str, float | None] = {}

        for m in solver_m_list:
            key = str(m)
            free = _solve(graph=composed_graph, grammar=grammar, m_solver=int(m))
            forced_tp_id = full_top_pivot_by_m[key]
            fixed = (
                None
                if forced_tp_id is None
                else _solve(
                    graph=composed_graph,
                    grammar=grammar,
                    m_solver=int(m),
                    forced_tp_id=str(forced_tp_id),
                )
            )
            regret = None
            if fixed is not None and free["score"] is not None and fixed["score"] is not None:
                regret = float(free["score"] - fixed["score"])

            solver_free[key] = free
            solver_fixed[key] = fixed
            pivot_preservation[key] = bool(
                free["valid"]
                and forced_tp_id is not None
                and free["turning_point_id"] is not None
                and str(free["turning_point_id"]) == str(forced_tp_id)
            )
            semantic_regret[key] = regret

        strategies[strategy] = {
            "retained_block1_size": int(len(compressed_block1)),
            "retained_block1_ids": [event.id for event in compressed_block1],
            "compression_diag": compression_diag,
            "state_left": _state_digest(left_state),
            "state_block2": _state_digest(block2_state),
            "state_composed": _state_digest(composed_state),
            "state_composed_ground_truth": _state_digest(composed_ground_truth),
            "composition_exact_match": bool(
                context_equivalent(
                    left=composed_state,
                    right=composed_ground_truth,
                    compare_pivots=True,
                )
            ),
            "pivot_shift_right_branch": pivot_shift_right_branch,
            "left_top_pivot": (
                None
                if left_top is None
                else {
                    "event_id": str(left_top.event_id),
                    "weight": float(left_top.weight),
                    "j_dev": int(left_top.j_dev),
                }
            ),
            "right_top_pivot": (
                None
                if right_top is None
                else {
                    "event_id": str(right_top.event_id),
                    "weight": float(right_top.weight),
                    "j_dev": int(right_top.j_dev),
                }
            ),
            "composed_top_pivot": (
                None
                if composed_top is None
                else {
                    "event_id": str(composed_top.event_id),
                    "weight": float(composed_top.weight),
                    "j_dev": int(composed_top.j_dev),
                }
            ),
            # Proxy for latent prefix capacity if right block contributes the TP.
            "left_d_total_proxy": int(development_eligible_count(len(compressed_block1))),
            "solver": solver_free,
            "solver_fixed_pivot": solver_fixed,
            "pivot_preservation": pivot_preservation,
            "semantic_regret": semantic_regret,
        }

    primary_solver_m = str(solver_m_list[0])
    naive_primary_valid = bool(strategies["naive"]["solver"][primary_solver_m]["valid"])
    contract_primary_valid = bool(strategies["contract_guarded"]["solver"][primary_solver_m]["valid"])
    full_primary_valid = bool(full_solver[primary_solver_m]["valid"])

    witness = {
        "primary_solver_M": int(solver_m_list[0]),
        "full_feasible": full_primary_valid,
        "naive_infeasible_after_compress": (not naive_primary_valid),
        "contract_feasible_after_compress": contract_primary_valid,
        "separation_holds": bool(
            full_primary_valid and (not naive_primary_valid) and contract_primary_valid
        ),
    }

    return {
        "settings": {
            "k": int(k),
            "target_size_block1": int(target_size),
            "compression_M": int(compression_m),
            "solver_M_list": [int(value) for value in solver_m_list],
        },
        "graph": {
            "n_events_total": int(len(graph.events)),
            "n_events_block1": int(len(block1)),
            "n_events_block2": int(len(block2)),
            "duration": float(graph.duration),
        },
        "full_solver": full_solver,
        "strategies": strategies,
        "witness": witness,
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    witness = data["witness"]
    primary_m = str(witness["primary_solver_M"])

    lines = [
        f"# {meta.name}",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Witness Check (Primary Solver M)",
        "",
        f"- Primary `M`: {witness['primary_solver_M']}",
        f"- Full graph feasible: {witness['full_feasible']}",
        f"- Naive infeasible after compression: {witness['naive_infeasible_after_compress']}",
        f"- Contract feasible after compression: {witness['contract_feasible_after_compress']}",
        f"- Separation holds: {witness['separation_holds']}",
        "",
        "## Solver Outcomes",
        "",
        (
            "| strategy | retained_block1 | pivot_shift_right_branch | composition_exact | "
            "M | free_valid | free_tp | fixed_valid | fixed_tp | pivot_preserved | semantic_regret |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]

    for strategy in ("naive", "contract_guarded"):
        strategy_row = data["strategies"][strategy]
        for m, result in strategy_row["solver"].items():
            fixed = strategy_row["solver_fixed_pivot"][m]
            fixed_valid = "n/a" if fixed is None else str(bool(fixed["valid"]))
            fixed_tp = "n/a" if fixed is None else str(fixed["turning_point_id"])
            regret = strategy_row["semantic_regret"][m]
            regret_text = "n/a" if regret is None else f"{float(regret):.6f}"
            lines.append(
                f"| {strategy} | {strategy_row['retained_block1_size']} | "
                f"{strategy_row['pivot_shift_right_branch']} | {strategy_row['composition_exact_match']} | "
                f"{m} | {result['valid']} | {result['turning_point_id']} | "
                f"{fixed_valid} | {fixed_tp} | {strategy_row['pivot_preservation'][m]} | {regret_text} |"
            )

    lines.extend(
        [
            "",
            "## Contract Pressure",
            "",
            "| strategy | drops_accepted | rejected_contract | rejected_gap_guard |",
            "|---|---:|---:|---:|",
        ]
    )

    for strategy in ("naive", "contract_guarded"):
        diag = data["strategies"][strategy]["compression_diag"]
        lines.append(
            f"| {strategy} | {diag.get('drops_accepted', 0)} | "
            f"{diag.get('rejected_contract', 0)} | {diag.get('rejected_gap_guard', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Primary-Solver Detail",
            "",
            f"- Naive valid (`M={primary_m}`): {data['strategies']['naive']['solver'][primary_m]['valid']}",
            f"- Naive fixed-pivot valid (`M={primary_m}`): "
            f"{None if data['strategies']['naive']['solver_fixed_pivot'][primary_m] is None else data['strategies']['naive']['solver_fixed_pivot'][primary_m]['valid']}",
            f"- Contract valid (`M={primary_m}`): {data['strategies']['contract_guarded']['solver'][primary_m]['valid']}",
            f"- Contract fixed-pivot valid (`M={primary_m}`): "
            f"{None if data['strategies']['contract_guarded']['solver_fixed_pivot'][primary_m] is None else data['strategies']['contract_guarded']['solver_fixed_pivot'][primary_m]['valid']}",
            f"- Naive top pivot after compose: "
            f"{data['strategies']['naive']['composed_top_pivot']}",
            f"- Contract top pivot after compose: "
            f"{data['strategies']['contract_guarded']['composed_top_pivot']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--compression-m", type=int, default=10)
    parser.add_argument("--solver-m-list", default="1,10")
    parser.add_argument("--output-name", default="exp58_pivot_shift_witness")
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
    timer = ExperimentTimer()

    solver_m_list = _parse_solver_m_list(args.solver_m_list)
    data = run_exp58(
        k=int(args.k),
        target_size=int(args.target_size),
        compression_m=int(args.compression_m),
        solver_m_list=solver_m_list,
    )

    metadata = ExperimentMetadata(
        name=str(args.output_name),
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=1,
        n_extractions=int(len(solver_m_list) * 5),
        seed_range=(58, 58),
        parameters=data["settings"],
    )
    save_results(
        name=str(args.output_name),
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )

    witness = data["witness"]
    print(f"Primary solver M={witness['primary_solver_M']}")
    print(f"Full feasible: {witness['full_feasible']}")
    print(f"Naive infeasible after compression: {witness['naive_infeasible_after_compress']}")
    print(f"Contract feasible after compression: {witness['contract_feasible_after_compress']}")
    print(f"Separation holds: {witness['separation_holds']}")
    for strategy in ("naive", "contract_guarded"):
        strategy_data = data["strategies"][strategy]
        print(
            f"{strategy}: retained_block1={strategy_data['retained_block1_size']} "
            f"pivot_shift={strategy_data['pivot_shift_right_branch']} "
            f"composition_exact={strategy_data['composition_exact_match']}"
        )
        for m in solver_m_list:
            result = strategy_data["solver"][str(m)]
            fixed = strategy_data["solver_fixed_pivot"][str(m)]
            fixed_valid = None if fixed is None else fixed["valid"]
            fixed_tp = None if fixed is None else fixed["turning_point_id"]
            print(
                f"  M={m}: valid={result['valid']} "
                f"tp={result['turning_point_id']} score={result['score']} "
                f"fixed_valid={fixed_valid} fixed_tp={fixed_tp} "
                f"pivot_preserved={strategy_data['pivot_preservation'][str(m)]} "
                f"semantic_regret={strategy_data['semantic_regret'][str(m)]}"
            )


if __name__ == "__main__":
    main()
