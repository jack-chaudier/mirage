"""Diagnose weight-distribution false positives from invariance suite."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import CausalGraph, Event, Phase


SOURCE_PATH = Path(__file__).resolve().parent / "output" / "invariance_suite.json"
OUTPUT_SUMMARY = Path(__file__).resolve().parent / "output" / "weight_fp_diagnosis_summary.md"

FOCAL_ACTOR = "actor_0"
TARGET_HASH_SEED = "1"


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == TARGET_HASH_SEED:
        return

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = TARGET_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _mix_seed(base_seed: int, epsilon: float, salt: int) -> int:
    eps_key = int(round(epsilon * 100))
    return int((base_seed * 1_000_003 + eps_key * 9_176 + salt * 37) % (2**32 - 1))


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _sample_weight_distribution(
    distribution: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(0.0, 1.0, size=n)
    if distribution == "power_law":
        raw = rng.pareto(2.0, size=n) + 1.0
        return _normalize_0_1(raw)
    if distribution == "bimodal":
        choose_left = rng.random(size=n) < 0.5
        samples = np.empty(n, dtype=float)
        left_n = int(np.sum(choose_left))
        right_n = n - left_n
        if left_n > 0:
            samples[choose_left] = rng.beta(2.0, 5.0, size=left_n)
        if right_n > 0:
            samples[~choose_left] = rng.beta(5.0, 2.0, size=right_n)
        return samples
    raise ValueError(f"Unknown distribution: {distribution}")


def _replace_weights(
    graph: CausalGraph,
    distribution: str,
    epsilon: float,
    seed: int,
) -> CausalGraph:
    rng = np.random.default_rng(_mix_seed(seed, epsilon, salt=17))
    samples = _sample_weight_distribution(distribution, len(graph.events), rng)

    new_events = []
    for event, weight in zip(graph.events, samples, strict=True):
        new_events.append(
            replace(
                event,
                weight=float(weight),
                metadata={
                    **event.metadata,
                    "weight_override_distribution": distribution,
                },
            )
        )

    return replace(
        graph,
        events=tuple(new_events),
        metadata={
            **graph.metadata,
            "weight_override_distribution": distribution,
        },
    )


def _max_weight_focal_event(graph: CausalGraph, focal_actor: str) -> Event:
    focal_events = [event for event in graph.events if focal_actor in event.actors]
    if not focal_events:
        raise ValueError(f"No focal events for actor={focal_actor}")
    return max(
        focal_events,
        key=lambda event: (float(event.weight), -float(event.timestamp)),
    )


def _j_dev_pool(
    graph: CausalGraph,
    tp_timestamp: float,
    k: int,
) -> int:
    n_pre = sum(1 for event in graph.events if float(event.timestamp) < float(tp_timestamp))
    n_setup = ceil(0.2 * n_pre) if n_pre > 0 else 0
    if k > 0:
        max_setup_for_development = max(0, n_pre - k)
        n_setup = min(n_setup, max_setup_for_development)
    return int(n_pre - n_setup)


def _event_row(event: Event) -> dict:
    return {
        "id": event.id,
        "timestamp": float(event.timestamp),
        "weight": float(event.weight),
        "actors": sorted(event.actors),
    }


def _summarize_case(case: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"### {case['distribution']} | eps={case['epsilon']:.1f} | seed={case['seed']}"
    )
    lines.append("")
    lines.append(
        f"- max_weight_focal_event: `{case['max_weight_focal_event']['id']}` "
        f"(w={case['max_weight_focal_event']['weight']:.6f}, "
        f"t={case['max_weight_focal_event']['timestamp']:.6f})"
    )
    lines.append(f"- j_dev_pool: {case['j_dev_pool']}")
    lines.append(f"- j_dev_output: {case['j_dev_output']}")
    lines.append(
        f"- greedy TP: `{case['greedy_tp']['id']}` "
        f"(w={case['greedy_tp']['weight']:.6f}, t={case['greedy_tp']['timestamp']:.6f})"
    )
    lines.append(f"- tp_match: {case['tp_match']}")
    lines.append(
        f"- max_weight_in_candidate_pool: {case['max_weight_in_candidate_pool']} | "
        f"in_final_sequence: {case['max_weight_in_final_sequence']} | "
        f"phase_if_selected: {case['max_weight_phase_if_selected']}"
    )
    lines.append("- events at max_focal tp_timestamp:")
    if case["events_at_tp_timestamp"]:
        for row in case["events_at_tp_timestamp"]:
            lines.append(
                f"  - `{row['id']}` t={row['timestamp']:.6f} w={row['weight']:.6f} "
                f"actors={row['actors']}"
            )
    else:
        lines.append("  - none")

    lines.append("- development events in greedy output:")
    if case["development_events"]:
        for row in case["development_events"]:
            lines.append(
                f"  - idx={row['sequence_index']} `{row['id']}` t={row['timestamp']:.6f} "
                f"actors={row['actors']} phase={row['phase']}"
            )
    else:
        lines.append("  - none")

    lines.append(f"- diagnosis: {case['diagnosis']}")
    lines.append("")
    return "\n".join(lines)


def run_weight_fp_diagnosis() -> dict:
    _ensure_hash_seed()

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing invariance results: {SOURCE_PATH}")

    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    data = payload["results"]

    params = data["parameters"]
    k = int(params["k"])
    epsilons = [float(e) for e in params["epsilons"]]
    seed_start, seed_end = [int(v) for v in params["seed_range"]]
    n_events = 200
    n_actors = 6

    weight_configs = data["sweeps"]["weight_distribution"]["configs"]
    expected_total = int(sum(int(cfg["fp_count"]) for cfg in weight_configs))

    grammar = GrammarConfig(
        min_prefix_elements=k,
        min_timespan_fraction=0.0,
        max_temporal_gap=float("inf"),
    )

    all_cases: list[dict] = []
    by_distribution_counts: dict[str, int] = {}
    by_distribution_expected: dict[str, int] = {
        str(cfg["parameter_value"]): int(cfg["fp_count"]) for cfg in weight_configs
    }

    for cfg in weight_configs:
        distribution = str(cfg["parameter_value"])
        dist_cases: list[dict] = []

        for epsilon in epsilons:
            for seed in range(seed_start, seed_end + 1):
                base_graph = BurstyGenerator().generate(
                    BurstyConfig(
                        seed=int(seed),
                        epsilon=float(epsilon),
                        n_events=n_events,
                        n_actors=n_actors,
                    )
                )
                graph = _replace_weights(
                    graph=base_graph,
                    distribution=distribution,
                    epsilon=float(epsilon),
                    seed=int(seed),
                )

                max_focal = _max_weight_focal_event(graph, FOCAL_ACTOR)
                tp_timestamp = float(max_focal.timestamp)
                j_dev_pool_value = _j_dev_pool(graph, tp_timestamp, k)

                result = greedy_extract(
                    graph=graph,
                    focal_actor=FOCAL_ACTOR,
                    grammar=grammar,
                    pool_strategy="injection",
                    n_anchors=8,
                    max_sequence_length=20,
                    injection_top_n=40,
                )

                if not (j_dev_pool_value < k and bool(result.valid)):
                    continue

                j_dev_output_value = int(result.n_development)
                tp = result.turning_point
                if tp is None:
                    continue

                tp_match = bool(tp.id == max_focal.id)

                pool_ids_raw = result.metadata.get("pool_ids", ())
                pool_ids = {str(event_id) for event_id in pool_ids_raw} if pool_ids_raw else set()
                max_weight_in_pool = bool(max_focal.id in pool_ids)

                max_seq_idx = next(
                    (idx for idx, event in enumerate(result.events) if event.id == max_focal.id),
                    None,
                )
                max_in_sequence = bool(max_seq_idx is not None)
                max_phase = None if max_seq_idx is None else result.phases[max_seq_idx].name

                events_at_ts = [
                    _event_row(event)
                    for event in graph.events
                    if abs(float(event.timestamp) - tp_timestamp) <= 1e-12
                ]

                dev_events = []
                for idx, (event, phase) in enumerate(zip(result.events, result.phases, strict=True)):
                    if phase == Phase.DEVELOPMENT:
                        dev_events.append(
                            {
                                "sequence_index": int(idx),
                                "id": event.id,
                                "timestamp": float(event.timestamp),
                                "actors": sorted(event.actors),
                                "phase": phase.name,
                            }
                        )

                if tp_match:
                    diagnosis = (
                        "TP matches the max-weight focal event, but j_dev_pool is too low because it "
                        "is computed from strict global pre-TP timestamps while realized DEVELOPMENT "
                        "comes from selected sequence structure and classifier allocation."
                    )
                else:
                    diagnosis = (
                        "TP mismatch: j_dev_pool uses the max-weight focal timestamp (very early), "
                        "but greedy selects a later TP. This later TP allows enough prefix DEVELOPMENT "
                        "events, so j_dev_output >= k while j_dev_pool < k."
                    )

                dist_cases.append(
                    {
                        "distribution": distribution,
                        "epsilon": float(epsilon),
                        "seed": int(seed),
                        "k": int(k),
                        "max_weight_focal_event": _event_row(max_focal),
                        "j_dev_pool": int(j_dev_pool_value),
                        "j_dev_output": int(j_dev_output_value),
                        "greedy_tp": _event_row(tp),
                        "tp_match": bool(tp_match),
                        "events_at_tp_timestamp": events_at_ts,
                        "development_events": dev_events,
                        "max_weight_in_candidate_pool": bool(max_weight_in_pool),
                        "max_weight_in_final_sequence": bool(max_in_sequence),
                        "max_weight_phase_if_selected": max_phase,
                        "diagnosis": diagnosis,
                    }
                )

        by_distribution_counts[distribution] = len(dist_cases)
        expected_dist = by_distribution_expected[distribution]
        if len(dist_cases) != expected_dist:
            raise RuntimeError(
                f"FP count mismatch for distribution={distribution}: expected {expected_dist}, "
                f"found {len(dist_cases)}"
            )
        all_cases.extend(dist_cases)

    if len(all_cases) != expected_total:
        raise RuntimeError(
            f"Total FP mismatch: expected {expected_total}, found {len(all_cases)}"
        )

    all_cases_sorted = sorted(
        all_cases,
        key=lambda row: (row["distribution"], row["epsilon"], row["seed"]),
    )

    tp_match_true = sum(1 for case in all_cases_sorted if bool(case["tp_match"]))
    tp_match_false = len(all_cases_sorted) - tp_match_true

    lines: list[str] = [
        "# Weight-Distribution FP Diagnosis",
        "",
        f"Source: `{SOURCE_PATH}`",
        f"Total reproduced FP cases: {len(all_cases_sorted)} (expected {expected_total})",
        f"tp_match=True: {tp_match_true}",
        f"tp_match=False: {tp_match_false}",
        "",
        "## Distribution Counts",
        "",
        "| distribution | FP count |",
        "|---|---:|",
    ]
    for distribution in sorted(by_distribution_counts.keys()):
        lines.append(f"| {distribution} | {by_distribution_counts[distribution]} |")

    lines.append("")
    lines.append("## Per-Case Diagnosis")
    lines.append("")

    for case in all_cases_sorted:
        lines.append(_summarize_case(case))

    OUTPUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote diagnosis summary: {OUTPUT_SUMMARY}")
    print(f"Reproduced FP cases: {len(all_cases_sorted)}")

    return {
        "n_cases": len(all_cases_sorted),
        "cases": all_cases_sorted,
        "output_summary": str(OUTPUT_SUMMARY),
    }


if __name__ == "__main__":
    run_weight_fp_diagnosis()
