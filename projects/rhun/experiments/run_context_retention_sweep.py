"""Retention sweep for compression-contract crossover (multi-burst+gap)."""

from __future__ import annotations

import argparse
import os
from statistics import mean
import sys

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.extraction.tp_conditioned_solver import tp_conditioned_solve
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.theory.context_algebra import compress_events, induced_subgraph


FOCAL_ACTOR = "actor_0"
STRATEGIES = ["naive", "bridge_preserving", "contract_guarded"]
TARGET_HASH_SEED = "1"


def _retention_grid() -> list[float]:
    return [round(0.50 - 0.05 * i, 2) for i in range(9)]  # 0.50 -> 0.10


def _run_solver(
    graph,
    grammar: GrammarConfig,
    M: int,
    solver: str,
    forced_tp_id: str | None = None,
) -> dict:
    if solver == "tp":
        seq, _diag = tp_conditioned_solve(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            M=int(M),
            max_gap=float(grammar.max_temporal_gap),
            pool_strategy="injection",
            max_sequence_length=20,
            injection_top_n=40,
            tp_candidate_ids=(str(forced_tp_id),) if forced_tp_id is not None else None,
        )
    elif solver == "greedy":
        if forced_tp_id is not None:
            return {
                "valid": False,
                "score": None,
                "turning_point_id": None,
                "violations": ["forced_tp_not_supported_for_greedy"],
            }
        seq = greedy_extract(
            graph=graph,
            focal_actor=FOCAL_ACTOR,
            grammar=grammar,
            pool_strategy="injection",
            n_anchors=8,
            max_sequence_length=20,
            injection_top_n=40,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return {
        "valid": bool(seq is not None and seq.valid),
        "score": None if seq is None else float(seq.score),
        "turning_point_id": (
            None if (seq is None or seq.turning_point is None) else str(seq.turning_point.id)
        ),
        "violations": (None if seq is None else list(seq.violations)),
    }


def run_sweep(
    seeds: int,
    m_value: int,
    solver: str,
    n_events: int = 200,
    n_actors: int = 6,
    k: int = 3,
    max_gap: float = 0.11,
) -> dict:
    generator = MultiBurstGenerator()
    grammar = GrammarConfig(
        min_prefix_elements=int(k),
        max_phase_regressions=0,
        max_turning_points=1,
        min_length=4,
        max_length=20,
        min_timespan_fraction=0.15,
        max_temporal_gap=float(max_gap),
        focal_actor_coverage=0.60,
    )

    retentions = _retention_grid()
    rows: list[dict] = []

    total = len(retentions) * int(seeds)
    done = 0
    for retention in retentions:
        for seed in range(int(seeds)):
            graph = generator.generate(
                MultiBurstConfig(
                    n_events=int(n_events),
                    n_actors=int(n_actors),
                    seed=int(seed),
                )
            )
            full_events = tuple(graph.events)
            target_size = max(8, int(round(len(full_events) * float(retention))))
            full = _run_solver(
                graph,
                grammar=grammar,
                M=int(m_value),
                solver=str(solver),
            )

            row = {
                "seed": int(seed),
                "retention": float(retention),
                "target_size": int(target_size),
                "slack_ratio": float(target_size / grammar.max_length),
                "full": full,
                "strategies": {},
            }

            for strategy in STRATEGIES:
                compressed, diag = compress_events(
                    events=full_events,
                    focal_actor=FOCAL_ACTOR,
                    k=grammar.min_prefix_elements,
                    M=int(m_value),
                    target_size=target_size,
                    strategy=strategy,
                    max_gap=float(grammar.max_temporal_gap),
                    min_length=grammar.min_length,
                )
                compressed_graph = induced_subgraph(graph, [event.id for event in compressed])
                solved = _run_solver(
                    compressed_graph,
                    grammar=grammar,
                    M=int(m_value),
                    solver=str(solver),
                )
                fixed_pivot = None
                if str(solver) == "tp" and full.get("turning_point_id") is not None:
                    fixed_pivot = _run_solver(
                        compressed_graph,
                        grammar=grammar,
                        M=int(m_value),
                        solver=str(solver),
                        forced_tp_id=str(full["turning_point_id"]),
                    )
                semantic_regret = None
                if (
                    fixed_pivot is not None
                    and solved["score"] is not None
                    and fixed_pivot["score"] is not None
                ):
                    semantic_regret = float(solved["score"] - fixed_pivot["score"])
                row["strategies"][strategy] = {
                    "valid": bool(solved["valid"]),
                    "score": solved["score"],
                    "turning_point_id": solved["turning_point_id"],
                    "pivot_preserved": bool(
                        full.get("valid")
                        and full.get("turning_point_id") is not None
                        and solved.get("valid")
                        and solved.get("turning_point_id") is not None
                        and str(solved["turning_point_id"]) == str(full["turning_point_id"])
                    ),
                    "fixed_pivot": fixed_pivot,
                    "semantic_regret": semantic_regret,
                    "retained_ratio": float(len(compressed) / len(full_events)),
                    "compression_diag": diag,
                }

            rows.append(row)
            done += 1
        print(f"Completed retention={retention:.2f} ({done}/{total})", flush=True)

    by_retention: dict[float, list[dict]] = {}
    for row in rows:
        by_retention.setdefault(float(row["retention"]), []).append(row)

    curve_rows: list[dict] = []
    crossover_retention: float | None = None
    for retention in sorted(by_retention.keys(), reverse=True):
        bucket = by_retention[retention]
        n = len(bucket)
        naive = sum(1 for row in bucket if row["strategies"]["naive"]["valid"]) / n
        bridge = sum(1 for row in bucket if row["strategies"]["bridge_preserving"]["valid"]) / n
        contract = sum(1 for row in bucket if row["strategies"]["contract_guarded"]["valid"]) / n
        delta = float(contract - naive)
        if crossover_retention is None and delta > 1e-12:
            crossover_retention = float(retention)

        rejected_gap_guard_mean = mean(
            float(row["strategies"]["contract_guarded"]["compression_diag"].get("rejected_gap_guard", 0))
            for row in bucket
        )

        semantic_metrics: dict[str, float | None] = {}
        for strategy in STRATEGIES:
            pivot_preservation_flags = [
                row["strategies"][strategy]["pivot_preserved"]
                for row in bucket
                if row["strategies"][strategy]["valid"]
                and row["full"]["valid"]
                and row["full"]["turning_point_id"] is not None
            ]
            pivot_preservation_rate = (
                float(sum(1 for flag in pivot_preservation_flags if flag) / len(pivot_preservation_flags))
                if pivot_preservation_flags
                else None
            )

            fixed_rows = [
                row["strategies"][strategy]["fixed_pivot"]
                for row in bucket
                if row["strategies"][strategy]["fixed_pivot"] is not None
            ]
            fixed_valid_rate = (
                float(sum(1 for result in fixed_rows if result["valid"]) / len(fixed_rows))
                if fixed_rows
                else None
            )

            regrets = [
                float(row["strategies"][strategy]["semantic_regret"])
                for row in bucket
                if row["strategies"][strategy]["semantic_regret"] is not None
            ]
            semantic_regret_mean = float(mean(regrets)) if regrets else None

            semantic_metrics[f"{strategy}_pivot_preservation_rate"] = pivot_preservation_rate
            semantic_metrics[f"{strategy}_fixed_pivot_valid_rate"] = fixed_valid_rate
            semantic_metrics[f"{strategy}_mean_semantic_regret"] = semantic_regret_mean

        curve_rows.append(
            {
                "retention": float(retention),
                "n_cases": int(n),
                "naive_valid_rate": float(naive),
                "bridge_valid_rate": float(bridge),
                "contract_valid_rate": float(contract),
                "contract_minus_naive": delta,
                "mean_slack_ratio": float(mean(float(row["slack_ratio"]) for row in bucket)),
                "mean_rejected_gap_guard_contract": float(rejected_gap_guard_mean),
                **semantic_metrics,
            }
        )

    return {
        "settings": {
            "seeds": int(seeds),
            "n_events": int(n_events),
            "n_actors": int(n_actors),
            "k": int(k),
            "max_gap": float(max_gap),
            "M": int(m_value),
            "solver": str(solver),
            "retentions": retentions,
        },
        "curve": curve_rows,
        "crossover_retention_contract_over_naive": crossover_retention,
        "rows": rows,
    }


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    def _fmt(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"

    lines = [
        f"# {meta.name}",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        f"Solver: {data['settings']['solver']}",
        "",
        f"- Crossover retention (contract > naive): {data['crossover_retention_contract_over_naive']}",
        "",
        (
            "| retention | slack_ratio | naive | bridge | contract | contract-naive | "
            "naive_pivot_preserve | contract_pivot_preserve | "
            "naive_fixed_valid | contract_fixed_valid | "
            "naive_semantic_regret | contract_semantic_regret | "
            "mean_rejected_gap_guard |"
        ),
        (
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for row in data["curve"]:
        lines.append(
            f"| {row['retention']:.2f} | {row['mean_slack_ratio']:.2f} | "
            f"{row['naive_valid_rate']:.3f} | {row['bridge_valid_rate']:.3f} | "
            f"{row['contract_valid_rate']:.3f} | {row['contract_minus_naive']:.3f} | "
            f"{_fmt(row.get('naive_pivot_preservation_rate'))} | "
            f"{_fmt(row.get('contract_guarded_pivot_preservation_rate'))} | "
            f"{_fmt(row.get('naive_fixed_pivot_valid_rate'))} | "
            f"{_fmt(row.get('contract_guarded_fixed_pivot_valid_rate'))} | "
            f"{_fmt(row.get('naive_mean_semantic_regret'))} | "
            f"{_fmt(row.get('contract_guarded_mean_semantic_regret'))} | "
            f"{row['mean_rejected_gap_guard_contract']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--solver", choices=["tp", "greedy"], default="tp")
    parser.add_argument("--output-name", default="context_retention_sweep")
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

    data = run_sweep(
        seeds=int(args.seeds),
        m_value=int(args.m),
        solver=str(args.solver),
    )
    metadata = ExperimentMetadata(
        name=str(args.output_name),
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=int(len(data["curve"]) * int(args.seeds)),
        n_extractions=int(len(data["rows"]) * len(STRATEGIES)),
        seed_range=(0, int(args.seeds) - 1),
        parameters=data["settings"],
    )
    save_results(
        name=str(args.output_name),
        data=data,
        metadata=metadata,
        summary_formatter=_summary_markdown,
    )

    print(f"Crossover retention (contract > naive): {data['crossover_retention_contract_over_naive']}")
    for row in data["curve"]:
        print(
            f"ret={row['retention']:.2f} "
            f"naive={row['naive_valid_rate']:.3f} "
            f"bridge={row['bridge_valid_rate']:.3f} "
            f"contract={row['contract_valid_rate']:.3f} "
            f"delta={row['contract_minus_naive']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
