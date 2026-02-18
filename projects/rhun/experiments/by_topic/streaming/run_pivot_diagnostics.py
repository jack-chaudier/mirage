"""Experiment 49: pivot-arrival and deferred-commit diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator


FOCAL_ACTOR = "actor_0"
EPSILONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
K_VALUES = [1, 2, 3]
SEEDS = range(200)
N_EVENTS = 200
N_ACTORS = 6
DEFERRED_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "streaming"
RAW_PATH = OUTPUT_DIR / "pivot_diagnostics_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "pivot_diagnostics_summary.md"


def _sorted_events(graph) -> tuple:
    return tuple(sorted(graph.events, key=lambda event: (event.timestamp, event.id)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0.0:
        return float(min(values))
    if q >= 100.0:
        return float(max(values))

    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    w = pos - lo
    return float(ordered[lo] * (1.0 - w) + ordered[hi] * w)


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _format_value(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.3f}"


def _forced_score_for_pivot(
    graph,
    grammar: GrammarConfig,
    pivot_id: str | None,
    cache: dict[str, tuple[bool, float]],
) -> tuple[bool, float]:
    if pivot_id is None:
        return False, 0.0
    if pivot_id in cache:
        return cache[pivot_id]

    forced = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        override_tp_id=pivot_id,
    )
    valid = bool(forced.valid)
    score = float(forced.score) if valid else 0.0
    cache[pivot_id] = (valid, score)
    return valid, score


def _argmax_focal_event(events: tuple) -> object | None:
    focal_events = [event for event in events if FOCAL_ACTOR in event.actors]
    if not focal_events:
        return None
    return max(focal_events, key=lambda event: (float(event.weight), -float(event.timestamp)))


def _compute_running_max_trajectory(
    events: tuple,
    global_max_id: str | None,
    k: int,
) -> dict:
    events_seen: list = []
    running_max_id_at_step: list[str | None] = []
    running_max_weight = float("-inf")
    running_max_event = None
    predecessor_count_by_id: dict[str, int] = {}

    for event in events:
        preds = sum(1 for seen in events_seen if float(seen.timestamp) < float(event.timestamp))
        predecessor_count_by_id[event.id] = preds
        events_seen.append(event)

    records: list[dict] = []
    for step, event in enumerate(events):
        focal = FOCAL_ACTOR in event.actors
        if focal and float(event.weight) > running_max_weight:
            running_max_event = event
            running_max_weight = float(event.weight)
            records.append(
                {
                    "record_step": int(step),
                    "record_weight": float(event.weight),
                    "record_has_k_preds": bool(predecessor_count_by_id[event.id] >= k),
                    "record_is_global_max": bool(global_max_id is not None and event.id == global_max_id),
                    "record_id": event.id,
                }
            )

        running_max_id_at_step.append(None if running_max_event is None else running_max_event.id)

    return {
        "records": records,
        "running_max_id_at_step": running_max_id_at_step,
        "predecessor_count_by_id": predecessor_count_by_id,
    }


def _deferred_commit_for_fraction(
    fraction: float,
    events: tuple,
    running_max_id_at_step: list[str | None],
    predecessor_count_by_id: dict[str, int],
    id_to_event: dict[str, object],
    global_max_id: str | None,
    k: int,
) -> dict:
    n_events = len(events)
    threshold_step = int(fraction * n_events)
    if threshold_step >= n_events:
        threshold_step = n_events - 1
    if threshold_step < 0:
        threshold_step = 0

    rm_id_at_threshold = running_max_id_at_step[threshold_step] if n_events > 0 else None
    rm_event_at_threshold = None if rm_id_at_threshold is None else id_to_event[rm_id_at_threshold]
    rm_matches_global = bool(rm_id_at_threshold is not None and rm_id_at_threshold == global_max_id)
    rm_weight_at_threshold = None if rm_event_at_threshold is None else float(rm_event_at_threshold.weight)

    committed = False
    commit_step: int | None = None
    committed_pivot_id: str | None = None

    if rm_id_at_threshold is not None and predecessor_count_by_id.get(rm_id_at_threshold, 0) >= k:
        committed = True
        commit_step = threshold_step
        committed_pivot_id = rm_id_at_threshold
    else:
        for step in range(threshold_step + 1, n_events):
            current_id = running_max_id_at_step[step]
            prev_id = running_max_id_at_step[step - 1] if step > 0 else None
            if current_id is None or current_id == prev_id:
                continue
            if predecessor_count_by_id.get(current_id, 0) >= k:
                committed = True
                commit_step = step
                committed_pivot_id = current_id
                break

    committed_event = None if committed_pivot_id is None else id_to_event[committed_pivot_id]
    commit_fraction = (
        float(commit_step / n_events) if committed and commit_step is not None and n_events > 0 else None
    )
    committed_weight = None if committed_event is None else float(committed_event.weight)

    return {
        "fraction": float(fraction),
        "threshold_step": int(threshold_step),
        "running_max_at_fraction_id": rm_id_at_threshold,
        "running_max_at_fraction_weight": rm_weight_at_threshold,
        "running_max_at_fraction_matches_global": rm_matches_global,
        "committed": bool(committed),
        "commit_step": commit_step,
        "commit_fraction": commit_fraction,
        "committed_pivot_id": committed_pivot_id,
        "committed_pivot_weight": committed_weight,
    }


def _classify_regret(exact_match: bool, regret: float) -> str:
    if exact_match:
        return "exact_match"
    if regret < 1.0:
        return "near_miss"
    return "significant_gap"


def _run_instance(generator: BurstyGenerator, epsilon: float, k: int, seed: int) -> dict:
    graph = generator.generate(
        BurstyConfig(
            n_events=N_EVENTS,
            n_actors=N_ACTORS,
            seed=seed,
            epsilon=epsilon,
        )
    )
    grammar = GrammarConfig(min_prefix_elements=k)
    events = _sorted_events(graph)
    n_events = len(events)
    id_to_event = {event.id: event for event in events}

    finite = greedy_extract(graph=graph, focal_actor=FOCAL_ACTOR, grammar=grammar)
    finite_valid = bool(finite.valid)
    finite_score = float(finite.score) if finite_valid else 0.0

    global_max_event = _argmax_focal_event(events)
    finite_tp = finite.turning_point if finite.turning_point is not None else global_max_event
    if finite_tp is None:
        # Degenerate: no focal events.
        finite_pivot_id = None
        finite_pivot_weight = None
        finite_pivot_step = None
        finite_pivot_frac = None
        finite_pivot_focal_rank = None
    else:
        finite_pivot_id = finite_tp.id
        finite_pivot_weight = float(finite_tp.weight)
        finite_pivot_step = next(
            (idx for idx, event in enumerate(events) if event.id == finite_pivot_id),
            None,
        )
        finite_pivot_frac = (
            float(finite_pivot_step / n_events) if finite_pivot_step is not None and n_events > 0 else None
        )
        focal_order = [event for event in events if FOCAL_ACTOR in event.actors]
        finite_pivot_focal_rank = next(
            (idx + 1 for idx, event in enumerate(focal_order) if event.id == finite_pivot_id),
            None,
        )

    trajectory = _compute_running_max_trajectory(
        events=events,
        global_max_id=finite_pivot_id,
        k=k,
    )

    running_max_id_at_step = trajectory["running_max_id_at_step"]
    predecessor_count_by_id = trajectory["predecessor_count_by_id"]

    deferred: dict[str, dict] = {}
    score_cache: dict[str, tuple[bool, float]] = {}

    for fraction in DEFERRED_FRACTIONS:
        key = f"{fraction:.2f}"
        decision = _deferred_commit_for_fraction(
            fraction=fraction,
            events=events,
            running_max_id_at_step=running_max_id_at_step,
            predecessor_count_by_id=predecessor_count_by_id,
            id_to_event=id_to_event,
            global_max_id=finite_pivot_id,
            k=k,
        )
        if decision["committed"] and decision["committed_pivot_id"] is not None:
            policy_valid, policy_score = _forced_score_for_pivot(
                graph=graph,
                grammar=grammar,
                pivot_id=decision["committed_pivot_id"],
                cache=score_cache,
            )
            score_value = float(policy_score) if policy_valid else 0.0
            valid_value = bool(policy_valid)
        else:
            score_value = 0.0
            valid_value = False

        if finite_pivot_weight is not None and decision["committed_pivot_weight"] is not None:
            weight_gap = float(finite_pivot_weight - decision["committed_pivot_weight"])
        else:
            weight_gap = None

        regret = float(finite_score - score_value)
        exact_match = bool(
            finite_pivot_id is not None
            and decision["committed_pivot_id"] is not None
            and decision["committed_pivot_id"] == finite_pivot_id
        )
        classification = _classify_regret(exact_match=exact_match, regret=regret)

        deferred[key] = {
            **decision,
            "valid": valid_value,
            "score": float(score_value),
            "regret": regret,
            "weight_gap": weight_gap,
            "exact_match": exact_match,
            "classification": classification,
        }

    records = trajectory["records"]
    n_records = len(records)
    if n_records > 0:
        last_record_step = int(records[-1]["record_step"])
        last_record_frac = float(last_record_step / n_events) if n_events > 0 else None
    else:
        last_record_step = None
        last_record_frac = None

    f10 = deferred["0.10"]
    f25 = deferred["0.25"]

    running_max_at_10_weight = f10["running_max_at_fraction_weight"]
    if finite_pivot_weight is not None and running_max_at_10_weight is not None and finite_pivot_weight > 0:
        running_max_at_10_ratio = float(running_max_at_10_weight / finite_pivot_weight)
    else:
        running_max_at_10_ratio = None

    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "seed": int(seed),
        "n_events": int(n_events),
        "finite_valid": finite_valid,
        "finite_score": float(finite_score),
        "finite_pivot_id": finite_pivot_id,
        "finite_pivot_weight": finite_pivot_weight,
        "finite_pivot_step": finite_pivot_step,
        "finite_pivot_frac": finite_pivot_frac,
        "finite_pivot_focal_rank": finite_pivot_focal_rank,
        "running_max_records": records,
        "n_records": int(n_records),
        "last_record_step": last_record_step,
        "last_record_frac": last_record_frac,
        "deferred": deferred,
        "f10_exact_match": bool(f10["exact_match"]),
        "f25_exact_match": bool(f25["exact_match"]),
        "f10_weight_gap": f10["weight_gap"],
        "f10_regret": float(f10["regret"]),
        "f25_regret": float(f25["regret"]),
        "f10_classification": f10["classification"],
        "running_max_at_10_matches_global": bool(f10["running_max_at_fraction_matches_global"]),
        "running_max_at_10_weight_ratio": running_max_at_10_ratio,
    }


def _build_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(float(row["epsilon"]), int(row["k"]))].append(row)

    rows: list[dict] = []
    for epsilon in EPSILONS:
        for k in K_VALUES:
            bucket = grouped[(epsilon, k)]
            pivot_fracs = [float(row["finite_pivot_frac"]) for row in bucket if row["finite_pivot_frac"] is not None]
            before_10 = sum(1 for row in bucket if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.10)
            before_25 = sum(1 for row in bucket if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.25)
            exact_f10 = sum(1 for row in bucket if bool(row["f10_exact_match"]))
            exact_f25 = sum(1 for row in bucket if bool(row["f25_exact_match"]))

            f10_weight_gaps = [float(row["f10_weight_gap"]) for row in bucket if row["f10_weight_gap"] is not None]
            f10_regrets = [float(row["f10_regret"]) for row in bucket]
            f25_regrets = [float(row["f25_regret"]) for row in bucket]

            n = len(bucket)
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": int(k),
                    "finite_pivot_frac_p50": _percentile(pivot_fracs, 50.0),
                    "finite_pivot_frac_p75": _percentile(pivot_fracs, 75.0),
                    "finite_pivot_frac_p90": _percentile(pivot_fracs, 90.0),
                    "finite_pivot_frac_p95": _percentile(pivot_fracs, 95.0),
                    "pct_global_max_before_10pct": 100.0 * before_10 / n,
                    "pct_global_max_before_25pct": 100.0 * before_25 / n,
                    "pct_exact_match_f10": 100.0 * exact_f10 / n,
                    "pct_exact_match_f25": 100.0 * exact_f25 / n,
                    "mean_weight_gap_f10": _safe_mean(f10_weight_gaps),
                    "mean_regret_f10": _safe_mean(f10_regrets),
                    "mean_regret_f25": _safe_mean(f25_regrets),
                }
            )
    return rows


def _write_summary_markdown(summary_rows: list[dict]) -> None:
    lines = [
        "| epsilon | k | finite_pivot_frac_p50 | finite_pivot_frac_p75 | finite_pivot_frac_p90 | finite_pivot_frac_p95 | pct_global_max_before_10pct | pct_global_max_before_25pct | pct_exact_match_f10 | pct_exact_match_f25 | mean_weight_gap_f10 | mean_regret_f10 | mean_regret_f25 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['k']} | "
            f"{_format_value(row['finite_pivot_frac_p50'])} | {_format_value(row['finite_pivot_frac_p75'])} | "
            f"{_format_value(row['finite_pivot_frac_p90'])} | {_format_value(row['finite_pivot_frac_p95'])} | "
            f"{row['pct_global_max_before_10pct']:.1f} | {row['pct_global_max_before_25pct']:.1f} | "
            f"{row['pct_exact_match_f10']:.1f} | {row['pct_exact_match_f25']:.1f} | "
            f"{_format_value(row['mean_weight_gap_f10'])} | {_format_value(row['mean_regret_f10'])} | {_format_value(row['mean_regret_f25'])} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _auto_key_finding(records: list[dict]) -> str:
    total = len(records)
    pct_before10 = 100.0 * sum(
        1 for row in records if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.10
    ) / total

    nonmatch_ratios = [
        float(row["running_max_at_10_weight_ratio"])
        for row in records
        if row["running_max_at_10_weight_ratio"] is not None
        and (not bool(row["running_max_at_10_matches_global"]))
    ]
    mean_nonmatch_ratio = _safe_mean(nonmatch_ratios)
    mean_regret_f10 = _safe_mean([float(row["f10_regret"]) for row in records])

    if pct_before10 >= 70.0:
        return f"Hypothesis A confirmed: global max arrives before 10% in {pct_before10:.1f}% of instances."
    if mean_nonmatch_ratio == mean_nonmatch_ratio and mean_nonmatch_ratio >= 0.95:
        pct = 100.0 * mean_nonmatch_ratio
        return (
            "Hypothesis B confirmed: running max at 10% is "
            f"{pct:.1f}% of global max weight, producing only {mean_regret_f10:.2f} regret."
        )

    by_eps = {}
    for epsilon in EPSILONS:
        bucket = [row for row in records if abs(float(row["epsilon"]) - epsilon) <= 1e-12]
        if not bucket:
            continue
        by_eps[epsilon] = 100.0 * sum(
            1 for row in bucket if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.10
        ) / len(bucket)

    high_eps = [eps for eps, pct in by_eps.items() if pct >= 70.0]
    if high_eps:
        cutoff = min(high_eps)
        return (
            f"Mixed: Hypothesis A holds for eps>={cutoff:.2f}, "
            f"Hypothesis B dominates below that range."
        )
    return "Mixed: neither hypothesis dominates globally; both effects are present."


def _print_stdout_summary(records: list[dict]) -> None:
    pivot_fracs = [float(row["finite_pivot_frac"]) for row in records if row["finite_pivot_frac"] is not None]
    print("=== PIVOT ARRIVAL DIAGNOSTICS ===")
    print("")
    print("Global max focal pivot arrival (finite_pivot_frac):")
    print(
        "  Overall: "
        f"p50={_percentile(pivot_fracs,50):.2f}  "
        f"p75={_percentile(pivot_fracs,75):.2f}  "
        f"p90={_percentile(pivot_fracs,90):.2f}  "
        f"p95={_percentile(pivot_fracs,95):.2f}  "
        f"p99={_percentile(pivot_fracs,99):.2f}"
    )
    print("")
    print("By epsilon:")
    for epsilon in EPSILONS:
        bucket = [
            float(row["finite_pivot_frac"])
            for row in records
            if abs(float(row["epsilon"]) - epsilon) <= 1e-12 and row["finite_pivot_frac"] is not None
        ]
        print(
            f"  eps={epsilon:.2f}: "
            f"p50={_percentile(bucket,50):.2f}  "
            f"p75={_percentile(bucket,75):.2f}  "
            f"p90={_percentile(bucket,90):.2f}  "
            f"p95={_percentile(bucket,95):.2f}"
        )

    total = len(records)
    before10 = 100.0 * sum(
        1 for row in records if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.10
    ) / total
    before25 = 100.0 * sum(
        1 for row in records if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.25
    ) / total
    before50 = 100.0 * sum(
        1 for row in records if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.50
    ) / total

    print("")
    print("=== HYPOTHESIS A vs B ===")
    print("")
    print(f"Fraction of instances where global max appears before 10%:  {before10:.1f}%")
    print(f"Fraction of instances where global max appears before 25%:  {before25:.1f}%")
    print(f"Fraction of instances where global max appears before 50%:  {before50:.1f}%")
    print("")
    print("By epsilon (pct global max before 10%):")
    for epsilon in EPSILONS:
        bucket = [row for row in records if abs(float(row["epsilon"]) - epsilon) <= 1e-12]
        pct = 100.0 * sum(
            1 for row in bucket if row["finite_pivot_frac"] is not None and row["finite_pivot_frac"] < 0.10
        ) / len(bucket)
        print(f"  eps={epsilon:.2f}: {pct:.1f}%")

    exact_f10 = 100.0 * sum(1 for row in records if bool(row["f10_exact_match"])) / total
    weight_gap_f10 = _safe_mean([float(row["f10_weight_gap"]) for row in records if row["f10_weight_gap"] is not None])
    regret_f10 = _safe_mean([float(row["f10_regret"]) for row in records])
    cls_counts = defaultdict(int)
    for row in records:
        cls_counts[str(row["f10_classification"])] += 1

    print("")
    print("=== DEFERRED COMMIT ANALYSIS (f=0.10) ===")
    print("")
    print(f"Pivot matches finite exactly: {exact_f10:.1f}%")
    print(f"Mean weight gap (finite - deferred): {weight_gap_f10:.3f}")
    print(f"Mean score regret: {regret_f10:.2f}")
    print("")
    print("Classification:")
    for name in ["exact_match", "near_miss", "significant_gap"]:
        pct = 100.0 * cls_counts[name] / total
        print(f"  {name}: {pct:.1f}%")

    n_records_mean = _safe_mean([float(row["n_records"]) for row in records])
    last_record_step_mean = _safe_mean(
        [float(row["last_record_step"]) for row in records if row["last_record_step"] is not None]
    )
    last_record_frac_mean = _safe_mean(
        [float(row["last_record_frac"]) for row in records if row["last_record_frac"] is not None]
    )
    print("")
    print("=== RUNNING MAX TRAJECTORY STATS ===")
    print("")
    print(f"Mean records per instance: {n_records_mean:.2f}")
    print(f"Mean step of last record: {last_record_step_mean:.2f} (frac: {last_record_frac_mean:.2f})")
    print("")
    print("Record weight convergence:")
    for idx in [1, 2, 3]:
        values = []
        ratios = []
        for row in records:
            records_list = row["running_max_records"]
            if len(records_list) < idx:
                continue
            weight = float(records_list[idx - 1]["record_weight"])
            values.append(weight)
            finite_weight = row["finite_pivot_weight"]
            if finite_weight is not None and float(finite_weight) > 0:
                ratios.append(weight / float(finite_weight))
        mean_w = _safe_mean(values)
        mean_ratio = _safe_mean(ratios) * 100.0 if ratios else float("nan")
        print(f"  After record {idx}: mean weight = {mean_w:.3f} ({mean_ratio:.1f}% of global max)")

    print("")
    print("=== KEY FINDING ===")
    print(_auto_key_finding(records))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = BurstyGenerator()
    records: list[dict] = []

    total = len(EPSILONS) * len(K_VALUES) * len(SEEDS)
    completed = 0
    for epsilon in EPSILONS:
        for k in K_VALUES:
            for seed in SEEDS:
                records.append(_run_instance(generator=generator, epsilon=epsilon, k=k, seed=seed))
                completed += 1
            print(f"Completed epsilon={epsilon:.2f}, k={k} ({completed}/{total} instances)")

    RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_rows = _build_summary(records)
    _write_summary_markdown(summary_rows)
    _print_stdout_summary(records)

    print(f"Wrote raw output to {RAW_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
