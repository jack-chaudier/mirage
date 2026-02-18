"""Beam-search evaluation on residual FN cases and k-j boundary at epsilon=0.90."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import beam_search_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.theory.theorem import check_precondition


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ORACLE_DIFF_PATH = OUTPUT_DIR / "oracle_diff_results.json"
TP_MISSELECTION_PATH = OUTPUT_DIR / "tp_misselection_analysis.json"


def _case_key(epsilon: float, seed: int, focal_actor: str) -> str:
    return f"{epsilon:.2f}|{seed}|{focal_actor}"


def _tp_payload(graph, sequence):
    tp = sequence.turning_point
    if tp is None:
        return None
    global_index = next((i for i, event in enumerate(graph.events) if event.id == tp.id), None)
    return {
        "id": tp.id,
        "weight": float(tp.weight),
        "timestamp": float(tp.timestamp),
        "global_index": int(global_index) if global_index is not None else None,
        "normalized_position": float(graph.global_position(tp)),
    }


def _weighted_success(rows: list[dict]) -> float | None:
    if not rows:
        return None
    denom = sum(row["n"] for row in rows)
    if denom == 0:
        return None
    return sum(row["success_rate"] * row["n"] for row in rows) / denom


def _summary_markdown_beam(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"Residual FN cases: {data['n_cases']}",
        "",
        "| beam_width | valid_count | valid_rate | mean_valid_score |",
        "|---|---:|---:|---:|",
    ]
    for row in data["per_width_summary"]:
        score = "n/a" if row["mean_valid_score"] is None else f"{row['mean_valid_score']:.3f}"
        lines.append(
            f"| {row['beam_width']} | {row['valid_count']} | {row['valid_rate']:.3f} | {score} |"
        )

    lines.extend(
        [
            "",
            "## First Valid Width",
            "",
        ]
    )
    for width, count in sorted(data["first_valid_width_distribution"].items(), key=lambda x: x[0]):
        lines.append(f"- {width}: {count}")

    lines.append("")
    return "\n".join(lines)


def _summary_markdown_kj(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        "",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        f"Epsilon: {data['epsilon']}",
        "",
    ]

    for width in data["beam_widths"]:
        width_key = str(width)
        lines.extend(
            [
                f"## Beam Width {width}",
                "",
                "| k | mean_success_all_j | success_j_lt_k | success_j_ge_k | n_below | n_above |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in data["summary_by_width"][width_key]:
            below = "n/a" if row["success_j_lt_k"] is None else f"{row['success_j_lt_k']:.3f}"
            above = "n/a" if row["success_j_ge_k"] is None else f"{row['success_j_ge_k']:.3f}"
            lines.append(
                f"| {row['k']} | {row['mean_success_all_j']:.3f} | {below} | {above} | "
                f"{row['n_below']} | {row['n_above']} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_beam_search_sweep() -> None:
    if not ORACLE_DIFF_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {ORACLE_DIFF_PATH}")

    source = json.loads(ORACLE_DIFF_PATH.read_text(encoding="utf-8"))
    settings = source["results"]["settings"]

    grammar = GrammarConfig(**settings["grammar"])
    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    max_sequence_length = int(settings["max_sequence_length"])

    residual_cases: list[dict] = []
    for eps_key, rows in source["results"]["per_seed_results"].items():
        epsilon = float(eps_key)
        for row in rows:
            if row.get("false_negative", False):
                residual_cases.append(
                    {
                        "epsilon": epsilon,
                        "seed": int(row["seed"]),
                        "focal_actor": str(row.get("focal_actor", settings["focal_actor"])),
                    }
                )

    residual_cases.sort(key=lambda item: (item["epsilon"], item["seed"], item["focal_actor"]))

    beam_widths = [1, 2, 4, 8, 16]
    generator = BurstyGenerator()

    tp_rank_lookup: dict[str, int] = {}
    if TP_MISSELECTION_PATH.exists():
        rank_source = json.loads(TP_MISSELECTION_PATH.read_text(encoding="utf-8"))
        for row in rank_source["results"].get("per_case", []):
            key = _case_key(float(row["epsilon"]), int(row["seed"]), str(row.get("focal_actor", "actor_0")))
            tp_rank_lookup[key] = int(row["oracle_tp_weight_rank"])

    beam_timer = ExperimentTimer()

    per_case_results: list[dict] = []
    per_width_valid_scores: dict[int, list[float]] = {width: [] for width in beam_widths}
    per_width_valid_counts: Counter[int] = Counter()

    first_valid_distribution: Counter[str] = Counter()
    first_valid_rank_groups: dict[str, list[int]] = defaultdict(list)

    for case in residual_cases:
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case["focal_actor"])
        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )

        width_results: dict[str, dict] = {}
        first_valid_width: int | None = None

        for width in beam_widths:
            result = beam_search_extract(
                graph=graph,
                focal_actor=focal_actor,
                grammar=grammar,
                pool_strategy="injection",
                max_sequence_length=max_sequence_length,
                beam_width=width,
            )

            is_valid = bool(result.valid)
            if is_valid:
                per_width_valid_counts[width] += 1
                per_width_valid_scores[width].append(float(result.score))
                if first_valid_width is None:
                    first_valid_width = width

            width_results[str(width)] = {
                "valid": is_valid,
                "score": float(result.score),
                "tp": _tp_payload(graph, result),
                "n_development": int(result.n_development),
            }

        first_valid_label = str(first_valid_width) if first_valid_width is not None else "none"
        first_valid_distribution[first_valid_label] += 1

        key = _case_key(epsilon, seed, focal_actor)
        rank = tp_rank_lookup.get(key)
        if rank is not None:
            first_valid_rank_groups[first_valid_label].append(rank)

        per_case_results.append(
            {
                "epsilon": epsilon,
                "seed": seed,
                "focal_actor": focal_actor,
                "oracle_tp_weight_rank": rank,
                "first_valid_width": first_valid_width,
                "results_by_width": width_results,
            }
        )

    per_width_summary: list[dict] = []
    n_cases = len(residual_cases)
    for width in beam_widths:
        valid_count = int(per_width_valid_counts[width])
        valid_scores = per_width_valid_scores[width]
        per_width_summary.append(
            {
                "beam_width": width,
                "valid_count": valid_count,
                "valid_rate": (valid_count / n_cases) if n_cases else 0.0,
                "mean_valid_score": mean(valid_scores) if valid_scores else None,
            }
        )

    first_valid_rank_summary = {
        bucket: {
            "count": len(ranks),
            "mean_oracle_tp_rank": (mean(ranks) if ranks else None),
        }
        for bucket, ranks in sorted(first_valid_rank_groups.items(), key=lambda x: x[0])
    }

    beam_data = {
        "n_cases": n_cases,
        "beam_widths": beam_widths,
        "settings": settings,
        "per_width_summary": per_width_summary,
        "first_valid_width_distribution": dict(first_valid_distribution),
        "first_valid_width_oracle_rank_summary": first_valid_rank_summary,
        "per_case": per_case_results,
    }

    beam_meta = ExperimentMetadata(
        name="beam_search_sweep",
        timestamp=utc_timestamp(),
        runtime_seconds=beam_timer.elapsed(),
        n_graphs=n_cases,
        n_extractions=n_cases * len(beam_widths),
        seed_range=(int(settings["seed_start"]), int(settings["seed_end"])),
        parameters={
            "source": "oracle_diff_results.json",
            "beam_widths": beam_widths,
            "pool_strategy": "injection",
            "max_sequence_length": max_sequence_length,
            "epsilon_values": settings["epsilons"],
        },
    )
    save_results(
        "beam_search_sweep",
        beam_data,
        beam_meta,
        summary_formatter=_summary_markdown_beam,
    )

    print("Beam sweep on residual FN cases")
    for row in per_width_summary:
        score = "n/a" if row["mean_valid_score"] is None else f"{row['mean_valid_score']:.3f}"
        print(
            f"  w={row['beam_width']}: valid={row['valid_count']}/{n_cases} "
            f"({row['valid_rate']:.3f}), mean_valid_score={score}"
        )
    print("  first_valid_width_distribution:", dict(first_valid_distribution))

    # --- k-j boundary sweep with beam search at epsilon=0.90 ---
    kj_timer = ExperimentTimer()
    kj_epsilon = 0.90
    kj_widths = [1, 4, 8]
    seeds = range(1, 51)
    k_values = range(0, 6)

    graph_cache = {
        seed: generator.generate(
            BurstyConfig(seed=seed, epsilon=kj_epsilon, n_events=n_events, n_actors=n_actors)
        )
        for seed in seeds
    }

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    total_extractions = 0

    for width in kj_widths:
        for seed in seeds:
            graph = graph_cache[seed]
            for actor in sorted(graph.actors):
                for k in k_values:
                    grammar_k = GrammarConfig.parametric(
                        k,
                        max_phase_regressions=grammar.max_phase_regressions,
                        max_turning_points=grammar.max_turning_points,
                        min_length=grammar.min_length,
                        max_length=grammar.max_length,
                        min_timespan_fraction=grammar.min_timespan_fraction,
                        focal_actor_coverage=grammar.focal_actor_coverage,
                    )
                    result = beam_search_extract(
                        graph=graph,
                        focal_actor=actor,
                        grammar=grammar_k,
                        pool_strategy="injection",
                        max_sequence_length=max_sequence_length,
                        beam_width=width,
                    )
                    precondition = check_precondition(graph, actor, grammar_k)
                    j = int(precondition["events_before_max"])

                    buckets[(width, k, j)].append(int(result.valid))
                    total_extractions += 1

    rows: list[dict] = []
    for (width, k, j), outcomes in sorted(buckets.items()):
        n = len(outcomes)
        rows.append(
            {
                "beam_width": width,
                "k": k,
                "j": j,
                "n": n,
                "success_rate": (sum(outcomes) / n) if n else 0.0,
            }
        )

    summary_by_width: dict[str, list[dict]] = {}
    for width in kj_widths:
        width_rows = [row for row in rows if row["beam_width"] == width]
        k_summary: list[dict] = []
        for k in k_values:
            k_rows = [row for row in width_rows if row["k"] == k]
            below = [row for row in k_rows if row["j"] < k]
            above = [row for row in k_rows if row["j"] >= k]
            k_summary.append(
                {
                    "k": k,
                    "mean_success_all_j": _weighted_success(k_rows) or 0.0,
                    "success_j_lt_k": _weighted_success(below),
                    "success_j_ge_k": _weighted_success(above),
                    "n_below": int(sum(row["n"] for row in below)),
                    "n_above": int(sum(row["n"] for row in above)),
                }
            )
        summary_by_width[str(width)] = k_summary

    kj_data = {
        "epsilon": kj_epsilon,
        "beam_widths": kj_widths,
        "k_values": list(k_values),
        "seed_range": [min(seeds), max(seeds)],
        "rows": rows,
        "summary_by_width": summary_by_width,
    }

    kj_meta = ExperimentMetadata(
        name="kj_boundary_beam_search",
        timestamp=utc_timestamp(),
        runtime_seconds=kj_timer.elapsed(),
        n_graphs=len(seeds) * len(kj_widths),
        n_extractions=total_extractions,
        seed_range=(min(seeds), max(seeds)),
        parameters={
            "epsilon": kj_epsilon,
            "beam_widths": kj_widths,
            "pool_strategy": "injection",
            "max_sequence_length": max_sequence_length,
        },
    )
    save_results(
        "kj_boundary_beam_search",
        kj_data,
        kj_meta,
        summary_formatter=_summary_markdown_kj,
    )

    print("\nK-J boundary at epsilon=0.90 with beam search")
    for width in kj_widths:
        print(f"  beam_width={width}")
        for row in summary_by_width[str(width)]:
            below = "n/a" if row["success_j_lt_k"] is None else f"{row['success_j_lt_k']:.3f}"
            above = "n/a" if row["success_j_ge_k"] is None else f"{row['success_j_ge_k']:.3f}"
            print(
                f"    k={row['k']}: mean={row['mean_success_all_j']:.3f}, "
                f"j<k={below}, j>=k={above}"
            )


if __name__ == "__main__":
    run_beam_search_sweep()
