"""Run 1-opt timespan-repair sweep on residual failure cohorts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, repair_timespan
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.schemas import Event


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BEAM_SWEEP_PATH = OUTPUT_DIR / "beam_search_sweep.json"
ORACLE_DIFF_PATH = OUTPUT_DIR / "oracle_diff_results.json"


def _case_key(epsilon: float, seed: int, focal_actor: str) -> str:
    return f"{epsilon:.2f}|{seed}|{focal_actor}"


def _timespan(events: tuple[Event, ...]) -> float:
    if not events:
        return 0.0
    return float(events[-1].timestamp - events[0].timestamp)


def _to_cases_from_beam_unreachable(beam_data: dict) -> list[dict]:
    cases: list[dict] = []
    for row in beam_data["results"]["per_case"]:
        if row["results_by_width"].get("16", {}).get("valid", False):
            continue
        cases.append(
            {
                "epsilon": float(row["epsilon"]),
                "seed": int(row["seed"]),
                "focal_actor": str(row["focal_actor"]),
            }
        )
    cases.sort(key=lambda c: (c["epsilon"], c["seed"], c["focal_actor"]))
    return cases


def _to_cases_from_oracle_fn(oracle_data: dict) -> list[dict]:
    cases: list[dict] = []
    per_seed = oracle_data["results"]["per_seed_results"]
    for eps_key, rows in per_seed.items():
        epsilon = float(eps_key)
        for row in rows:
            if not bool(row.get("false_negative", False)):
                continue
            cases.append(
                {
                    "epsilon": epsilon,
                    "seed": int(row["seed"]),
                    "focal_actor": str(row.get("focal_actor", "actor_0")),
                }
            )
    cases.sort(key=lambda c: (c["epsilon"], c["seed"], c["focal_actor"]))
    return cases


def _beam_recovered_set(beam_data: dict) -> set[str]:
    recovered: set[str] = set()
    for row in beam_data["results"]["per_case"]:
        width1_valid = bool(row["results_by_width"].get("1", {}).get("valid", False))
        first_valid = row.get("first_valid_width")
        if width1_valid:
            continue
        if first_valid is not None and int(first_valid) > 1:
            recovered.add(_case_key(float(row["epsilon"]), int(row["seed"]), str(row["focal_actor"])))
    return recovered


def _run_case_set(
    name: str,
    cases: list[dict],
    grammar: GrammarConfig,
    n_events: int,
    n_actors: int,
    max_sequence_length: int,
) -> dict:
    generator = BurstyGenerator()

    per_case: list[dict] = []
    repaired_case_keys: set[str] = set()

    repaired_swap_counts: list[int] = []
    repaired_weight_loss_frac: list[float] = []
    repaired_score_loss_frac: list[float] = []

    failure_reasons: Counter[str] = Counter()

    repaired_count = 0
    greedy_invalid_count = 0

    for case in cases:
        epsilon = float(case["epsilon"])
        seed = int(case["seed"])
        focal_actor = str(case["focal_actor"])
        case_key = _case_key(epsilon, seed, focal_actor)

        graph = generator.generate(
            BurstyConfig(seed=seed, epsilon=epsilon, n_events=n_events, n_actors=n_actors)
        )

        greedy = greedy_extract(
            graph=graph,
            focal_actor=focal_actor,
            grammar=grammar,
            pool_strategy="injection",
            max_sequence_length=max_sequence_length,
        )
        greedy_valid = bool(greedy.valid)
        if not greedy_valid:
            greedy_invalid_count += 1

        raw_pool = greedy.metadata.get("pool_ids")
        pool_ids: set[str] | None = None
        if isinstance(raw_pool, (tuple, list, set)):
            pool_ids = {str(event_id) for event_id in raw_pool}

        repaired = repair_timespan(
            graph=graph,
            sequence=greedy,
            grammar=grammar,
            pool_ids=pool_ids,
        )

        repaired_valid = bool(repaired.valid)
        repair_success = (not greedy_valid) and repaired_valid

        greedy_span = _timespan(greedy.events)
        repaired_span = _timespan(repaired.events)

        greedy_weight = float(sum(event.weight for event in greedy.events))
        repaired_weight = float(sum(event.weight for event in repaired.events))
        greedy_score = float(greedy.score)
        repaired_score = float(repaired.score)

        weight_loss_abs = greedy_weight - repaired_weight
        score_loss_abs = greedy_score - repaired_score

        weight_loss_frac = (weight_loss_abs / greedy_weight) if greedy_weight > 0 else 0.0
        score_loss_frac = (score_loss_abs / greedy_score) if greedy_score > 0 else 0.0

        swap_count = int(repaired.metadata.get("repair_swap_count", 0))
        repair_reason = repaired.metadata.get("repair_failure_reason")

        if repair_success:
            repaired_count += 1
            repaired_case_keys.add(case_key)
            repaired_swap_counts.append(swap_count)
            repaired_weight_loss_frac.append(weight_loss_frac)
            repaired_score_loss_frac.append(score_loss_frac)
        elif not greedy_valid:
            failure_reasons[str(repair_reason)] += 1

        per_case.append(
            {
                "case_key": case_key,
                "epsilon": epsilon,
                "seed": seed,
                "focal_actor": focal_actor,
                "greedy_valid": greedy_valid,
                "greedy_violations": list(greedy.violations),
                "repaired_valid": repaired_valid,
                "repair_success": repair_success,
                "repair_failure_reason": repair_reason,
                "swap_count": swap_count,
                "swaps": list(repaired.metadata.get("repair_swaps", ())),
                "greedy_timespan": greedy_span,
                "repaired_timespan": repaired_span,
                "required_timespan": float(grammar.min_timespan_fraction * graph.duration),
                "greedy_timespan_fraction": (greedy_span / graph.duration) if graph.duration > 0 else 0.0,
                "repaired_timespan_fraction": (repaired_span / graph.duration) if graph.duration > 0 else 0.0,
                "greedy_weight_sum": greedy_weight,
                "repaired_weight_sum": repaired_weight,
                "weight_loss_abs": weight_loss_abs,
                "weight_loss_frac": weight_loss_frac,
                "greedy_score": greedy_score,
                "repaired_score": repaired_score,
                "score_loss_abs": score_loss_abs,
                "score_loss_frac": score_loss_frac,
            }
        )

    n_cases = len(cases)
    repaired_rate = (repaired_count / n_cases) if n_cases else 0.0

    return {
        "name": name,
        "n_cases": n_cases,
        "greedy_invalid_cases": greedy_invalid_count,
        "repaired_count": repaired_count,
        "repaired_rate": repaired_rate,
        "mean_swaps_repaired": mean(repaired_swap_counts) if repaired_swap_counts else None,
        "mean_weight_loss_frac_repaired": mean(repaired_weight_loss_frac) if repaired_weight_loss_frac else None,
        "mean_score_loss_frac_repaired": mean(repaired_score_loss_frac) if repaired_score_loss_frac else None,
        "failure_reason_counts": dict(failure_reasons),
        "repaired_case_keys": sorted(repaired_case_keys),
        "per_case": per_case,
    }


def _summary_markdown(data: dict, metadata: ExperimentMetadata) -> str:
    lines = [
        f"# {metadata.name}",
        "",
        f"Generated: {metadata.timestamp}",
        f"Runtime: {metadata.runtime_seconds:.2f}s",
        "",
        "| cohort | n_cases | repaired_count | repaired_rate | mean_swaps_repaired | mean_weight_loss_frac_repaired | mean_score_loss_frac_repaired |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cohort in data["cohorts"]:
        mean_swaps = (
            "n/a"
            if cohort["mean_swaps_repaired"] is None
            else f"{cohort['mean_swaps_repaired']:.3f}"
        )
        mean_weight_loss = (
            "n/a"
            if cohort["mean_weight_loss_frac_repaired"] is None
            else f"{cohort['mean_weight_loss_frac_repaired']:.3f}"
        )
        mean_score_loss = (
            "n/a"
            if cohort["mean_score_loss_frac_repaired"] is None
            else f"{cohort['mean_score_loss_frac_repaired']:.3f}"
        )
        lines.append(
            f"| {cohort['name']} | {cohort['n_cases']} | {cohort['repaired_count']} | {cohort['repaired_rate']:.3f} | "
            f"{mean_swaps} | {mean_weight_loss} | {mean_score_loss} |"
        )

    lines.extend([
        "",
        "## Beam Comparison",
        "",
        f"- Beam-recovered cases (w>1): {data['beam_comparison']['beam_recovered_count']}",
        f"- Repair-recovered cases on full FN cohort: {data['beam_comparison']['repair_recovered_count']}",
        f"- Intersection: {data['beam_comparison']['intersection_count']}",
        f"- Repair superset of beam recoveries: {data['beam_comparison']['repair_is_superset_of_beam']}",
        "",
    ])
    return "\n".join(lines)


def _print_summary(data: dict) -> None:
    print("Repair sweep summary")
    print(
        "| cohort | n_cases | repaired | repaired_rate | mean_swaps | mean_weight_loss_frac | mean_score_loss_frac |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for cohort in data["cohorts"]:
        ms = "n/a" if cohort["mean_swaps_repaired"] is None else f"{cohort['mean_swaps_repaired']:.3f}"
        mw = (
            "n/a"
            if cohort["mean_weight_loss_frac_repaired"] is None
            else f"{cohort['mean_weight_loss_frac_repaired']:.3f}"
        )
        msf = (
            "n/a"
            if cohort["mean_score_loss_frac_repaired"] is None
            else f"{cohort['mean_score_loss_frac_repaired']:.3f}"
        )
        print(
            f"| {cohort['name']} | {cohort['n_cases']} | {cohort['repaired_count']} | {cohort['repaired_rate']:.3f} | "
            f"{ms} | {mw} | {msf} |"
        )

        if cohort["failure_reason_counts"]:
            top_reasons = sorted(
                cohort["failure_reason_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            reason_str = ", ".join(f"{reason}={count}" for reason, count in top_reasons)
            print(f"  failure_reasons: {reason_str}")

    beam = data["beam_comparison"]
    print("Beam comparison on full FN cohort")
    print(
        f"  beam_recovered={beam['beam_recovered_count']}, "
        f"repair_recovered={beam['repair_recovered_count']}, "
        f"intersection={beam['intersection_count']}, "
        f"repair_superset={beam['repair_is_superset_of_beam']}"
    )


def run_repair_sweep() -> dict:
    if not BEAM_SWEEP_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {BEAM_SWEEP_PATH}")
    if not ORACLE_DIFF_PATH.exists():
        raise FileNotFoundError(f"Missing required input: {ORACLE_DIFF_PATH}")

    beam_data = json.loads(BEAM_SWEEP_PATH.read_text(encoding="utf-8"))
    oracle_data = json.loads(ORACLE_DIFF_PATH.read_text(encoding="utf-8"))

    settings = oracle_data["results"]["settings"]
    grammar = GrammarConfig(**settings["grammar"])
    n_events = int(settings["n_events"])
    n_actors = int(settings["n_actors"])
    max_sequence_length = int(settings["max_sequence_length"])

    beam_unreachable_cases = _to_cases_from_beam_unreachable(beam_data)
    full_fn_cases = _to_cases_from_oracle_fn(oracle_data)

    timer = ExperimentTimer()

    cohort_57 = _run_case_set(
        name="beam_unreachable_57",
        cases=beam_unreachable_cases,
        grammar=grammar,
        n_events=n_events,
        n_actors=n_actors,
        max_sequence_length=max_sequence_length,
    )

    cohort_75 = _run_case_set(
        name="full_fn_75",
        cases=full_fn_cases,
        grammar=grammar,
        n_events=n_events,
        n_actors=n_actors,
        max_sequence_length=max_sequence_length,
    )

    beam_recovered = _beam_recovered_set(beam_data)
    repair_recovered = set(cohort_75["repaired_case_keys"])

    beam_comparison = {
        "beam_recovered_count": len(beam_recovered),
        "repair_recovered_count": len(repair_recovered),
        "intersection_count": len(beam_recovered & repair_recovered),
        "repair_only_count": len(repair_recovered - beam_recovered),
        "beam_only_count": len(beam_recovered - repair_recovered),
        "repair_is_superset_of_beam": beam_recovered.issubset(repair_recovered),
    }

    data = {
        "settings": {
            "source_beam_sweep": str(BEAM_SWEEP_PATH.name),
            "source_oracle_diff": str(ORACLE_DIFF_PATH.name),
            "grammar": settings["grammar"],
            "n_events": n_events,
            "n_actors": n_actors,
            "max_sequence_length": max_sequence_length,
            "pool_strategy": "injection",
        },
        "cohorts": [cohort_57, cohort_75],
        "beam_comparison": beam_comparison,
    }

    total_cases = len(beam_unreachable_cases) + len(full_fn_cases)
    metadata = ExperimentMetadata(
        name="repair_sweep",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_cases,
        n_extractions=total_cases,
        seed_range=(int(settings["seed_start"]), int(settings["seed_end"])),
        parameters={
            "cohorts": ["beam_unreachable_57", "full_fn_75"],
            "repair_method": "1-opt_timespan_swap_from_existing_pool",
        },
    )

    save_results("repair_sweep", data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    payload = run_repair_sweep()
    _print_summary(payload["results"])
