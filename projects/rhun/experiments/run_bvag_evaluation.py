"""Budget-Reservation VAG (BVAG) evaluation.

Compares:
- greedy
- VAG span-only
- VAG gap-aware
- BVAG (gap-aware + budget-aware)
- oracle
- exact_oracle

Across bursty and multi-burst distributions, with and without max_temporal_gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.generators.bursty import BurstyConfig, BurstyGenerator
from rhun.generators.multiburst import MultiBurstConfig, MultiBurstGenerator
from rhun.schemas import ExtractedSequence


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
GAP_ADVERSARIAL_PATH = OUTPUT_DIR / "gap_adversarial.json"
MULTIBURST_GAP_PATH = OUTPUT_DIR / "multiburst_gap.json"

BURSTY_EPSILONS = [0.50, 0.70, 0.90]
SEEDS = range(0, 50)
N_EVENTS = 200
N_ACTORS = 6
FOCAL_ACTOR = "actor_0"

POOL_STRATEGY = "injection"
N_ANCHORS = 8
MAX_SEQUENCE_LENGTH = 20
INJECTION_TOP_N = 40


def _load_reference_gap(path: Path, fallback: float) -> float:
    if not path.exists():
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(
        payload.get("results", {})
        .get("calibration", {})
        .get("selected_threshold", {})
        .get("value", fallback)
    )


def _seq_payload(seq: ExtractedSequence | None) -> dict:
    if seq is None:
        return {
            "valid": False,
            "score": None,
            "n_events": 0,
            "violations": ["oracle_no_candidate"],
            "event_ids": [],
            "selected_anchor_id": None,
        }
    return {
        "valid": bool(seq.valid),
        "score": float(seq.score),
        "n_events": len(seq.events),
        "violations": list(seq.violations),
        "event_ids": [event.id for event in seq.events],
        "selected_anchor_id": seq.metadata.get("anchor_id"),
    }


def _evaluate_algorithms(graph, grammar: GrammarConfig) -> dict:
    greedy = greedy_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        pool_strategy=POOL_STRATEGY,
        n_anchors=N_ANCHORS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        injection_top_n=INJECTION_TOP_N,
    )
    vag_span, vag_span_diag = viability_aware_greedy_extract(
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
    vag_gap, vag_gap_diag = viability_aware_greedy_extract(
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
    bvag, bvag_diag = viability_aware_greedy_extract(
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
    oracle, oracle_diag = oracle_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    exact, exact_diag = exact_oracle_extract(
        graph=graph,
        focal_actor=FOCAL_ACTOR,
        grammar=grammar,
    )

    return {
        "greedy": _seq_payload(greedy),
        "vag_span_only": {
            **_seq_payload(vag_span),
            "diagnostics": vag_span_diag,
        },
        "vag_gap_aware": {
            **_seq_payload(vag_gap),
            "diagnostics": vag_gap_diag,
        },
        "vag_budget_aware": {
            **_seq_payload(bvag),
            "diagnostics": bvag_diag,
        },
        "oracle": {
            **_seq_payload(oracle),
            "diagnostics": oracle_diag,
        },
        "exact_oracle": {
            **_seq_payload(exact),
            "diagnostics": exact_diag,
        },
    }


def _aggregate_validity(rows: list[dict], algo: str) -> dict:
    vals = [row["results"][algo]["valid"] for row in rows]
    valid_count = sum(1 for value in vals if value)
    n = len(vals)
    return {
        "valid_count": int(valid_count),
        "valid_rate": float(valid_count / n) if n else 0.0,
    }


def _ratio_summary(rows: list[dict], algo: str) -> dict:
    ratios: list[float] = []
    for row in rows:
        candidate = row["results"][algo]
        exact = row["results"]["exact_oracle"]
        if not candidate["valid"] or not exact["valid"]:
            continue
        exact_score = exact["score"]
        if exact_score is None or exact_score <= 0:
            continue
        score = candidate["score"]
        if score is None:
            continue
        ratios.append(float(score / exact_score))

    if not ratios:
        return {"count": 0, "mean": None, "median": None, "min": None}
    return {
        "count": len(ratios),
        "mean": float(mean(ratios)),
        "median": float(median(ratios)),
        "min": float(min(ratios)),
    }


def _chosen_anchor_diag(diagnostics: dict, selected_anchor_id: str | None) -> dict | None:
    if selected_anchor_id is None:
        return None
    for row in diagnostics.get("per_anchor", []):
        if row.get("anchor_id") == selected_anchor_id:
            return row
    return None


def _cohort_gap_metrics(rows: list[dict]) -> dict:
    gap_failures = [
        row
        for row in rows
        if row["results"]["exact_oracle"]["valid"]
        and not row["results"]["vag_gap_aware"]["valid"]
    ]
    recovered = [
        row for row in gap_failures if row["results"]["vag_budget_aware"]["valid"]
    ]
    regressions = [
        row
        for row in rows
        if row["results"]["vag_gap_aware"]["valid"]
        and not row["results"]["vag_budget_aware"]["valid"]
    ]

    first_steps: list[int] = []
    first_ranks: list[int] = []
    first_blb: list[int] = []
    sample_cases: list[dict] = []

    for row in recovered:
        bvag = row["results"]["vag_budget_aware"]
        bvag_diag = bvag.get("diagnostics", {})
        chosen_anchor = _chosen_anchor_diag(bvag_diag, bvag.get("selected_anchor_id"))
        first_block = None
        budget_trace = []
        if chosen_anchor is not None:
            first_block = chosen_anchor.get("first_budget_block")
            budget_trace = chosen_anchor.get("budget_trace", [])

        if first_block is not None:
            first_steps.append(int(first_block.get("step", 0)))
            first_ranks.append(int(first_block.get("candidate_weight_rank", 0)))
            first_blb.append(int(first_block.get("bridge_budget_lb", 0)))

        if len(sample_cases) < 8:
            sample_cases.append(
                {
                    "seed": row.get("seed"),
                    "epsilon": row.get("epsilon"),
                    "selected_anchor_id": bvag.get("selected_anchor_id"),
                    "first_budget_block": first_block,
                    "budget_trace": budget_trace,
                }
            )

    return {
        "gap_aware_failure_count": len(gap_failures),
        "bvag_recovered_count": len(recovered),
        "bvag_regressions_vs_gap_aware": len(regressions),
        "first_budget_block_mean_step": (float(mean(first_steps)) if first_steps else None),
        "first_budget_block_mean_weight_rank": (float(mean(first_ranks)) if first_ranks else None),
        "first_budget_block_mean_blb": (float(mean(first_blb)) if first_blb else None),
        "trajectory_samples": sample_cases,
    }


def _cohort_nogap_equality(rows: list[dict]) -> dict:
    all_valid_equal = True
    all_score_equal = True
    all_events_equal = True

    for row in rows:
        gap = row["results"]["vag_gap_aware"]
        budget = row["results"]["vag_budget_aware"]

        if bool(gap["valid"]) != bool(budget["valid"]):
            all_valid_equal = False

        gap_score = gap["score"]
        budget_score = budget["score"]
        if gap_score is None and budget_score is None:
            pass
        elif gap_score is None or budget_score is None:
            all_score_equal = False
        elif abs(float(gap_score) - float(budget_score)) > 1e-12:
            all_score_equal = False

        if list(gap["event_ids"]) != list(budget["event_ids"]):
            all_events_equal = False

    return {
        "all_valid_equal": all_valid_equal,
        "all_score_equal": all_score_equal,
        "all_events_equal": all_events_equal,
        "identical": bool(all_valid_equal and all_score_equal and all_events_equal),
    }


def _run_bursty_cohort(grammar: GrammarConfig) -> list[dict]:
    rows: list[dict] = []
    generator = BurstyGenerator()

    for seed in SEEDS:
        for epsilon in BURSTY_EPSILONS:
            graph = generator.generate(
                BurstyConfig(seed=seed, epsilon=epsilon, n_events=N_EVENTS, n_actors=N_ACTORS)
            )
            results = _evaluate_algorithms(graph, grammar)
            rows.append(
                {
                    "dataset": "bursty",
                    "seed": seed,
                    "epsilon": epsilon,
                    "results": results,
                }
            )
    return rows


def _run_multiburst_cohort(grammar: GrammarConfig) -> list[dict]:
    rows: list[dict] = []
    generator = MultiBurstGenerator()

    for seed in SEEDS:
        graph = generator.generate(MultiBurstConfig(seed=seed, n_events=N_EVENTS, n_actors=N_ACTORS))
        results = _evaluate_algorithms(graph, grammar)
        rows.append(
            {
                "dataset": "multiburst",
                "seed": seed,
                "results": results,
            }
        )
    return rows


def _summarize_cohort(rows: list[dict], with_gap: bool) -> dict:
    algorithms = [
        "greedy",
        "vag_span_only",
        "vag_gap_aware",
        "vag_budget_aware",
        "oracle",
        "exact_oracle",
    ]
    summary = {
        "n_cases": len(rows),
        "validity": {algo: _aggregate_validity(rows, algo) for algo in algorithms},
        "ratio_vs_exact": {
            algo: _ratio_summary(rows, algo)
            for algo in ["greedy", "vag_gap_aware", "vag_budget_aware"]
        },
    }
    if with_gap:
        summary["gap_metrics"] = _cohort_gap_metrics(rows)
    else:
        summary["no_gap_bvag_vs_gapaware"] = _cohort_nogap_equality(rows)
    return summary


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    def r(path: dict, key: str) -> float:
        return float(path["validity"][key]["valid_rate"])

    bursty_gap = data["cohorts"]["bursty_with_gap"]
    multiburst_gap = data["cohorts"]["multiburst_with_gap"]

    lines = [
        "# bvag_evaluation",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Bursty + Gap",
        "",
        f"- Gap-aware VAG validity: {100.0 * r(bursty_gap, 'vag_gap_aware'):.1f}%",
        f"- BVAG validity: {100.0 * r(bursty_gap, 'vag_budget_aware'):.1f}%",
        (
            "- BVAG recovered from gap-aware failures: "
            f"{bursty_gap['gap_metrics']['bvag_recovered_count']}/"
            f"{bursty_gap['gap_metrics']['gap_aware_failure_count']}"
        ),
        f"- BVAG regressions vs gap-aware: {bursty_gap['gap_metrics']['bvag_regressions_vs_gap_aware']}",
        (
            "- BVAG/exact ratio (feasible): "
            f"mean={bursty_gap['ratio_vs_exact']['vag_budget_aware']['mean']:.4f}, "
            f"min={bursty_gap['ratio_vs_exact']['vag_budget_aware']['min']:.4f}"
            if bursty_gap["ratio_vs_exact"]["vag_budget_aware"]["mean"] is not None
            else "- BVAG/exact ratio (feasible): n/a"
        ),
        (
            "- Mean step where budget filter first activates: "
            f"{bursty_gap['gap_metrics']['first_budget_block_mean_step']:.2f}"
            if bursty_gap["gap_metrics"]["first_budget_block_mean_step"] is not None
            else "- Mean step where budget filter first activates: n/a"
        ),
        (
            "- Mean weight rank of blocked candidate: "
            f"{bursty_gap['gap_metrics']['first_budget_block_mean_weight_rank']:.2f}"
            if bursty_gap["gap_metrics"]["first_budget_block_mean_weight_rank"] is not None
            else "- Mean weight rank of blocked candidate: n/a"
        ),
        (
            "- Mean B_lb at first block: "
            f"{bursty_gap['gap_metrics']['first_budget_block_mean_blb']:.2f}"
            if bursty_gap["gap_metrics"]["first_budget_block_mean_blb"] is not None
            else "- Mean B_lb at first block: n/a"
        ),
        "",
        "## Multi-burst + Gap",
        "",
        f"- Gap-aware VAG validity: {100.0 * r(multiburst_gap, 'vag_gap_aware'):.1f}%",
        f"- BVAG validity: {100.0 * r(multiburst_gap, 'vag_budget_aware'):.1f}%",
        (
            "- BVAG recovered from gap-aware failures: "
            f"{multiburst_gap['gap_metrics']['bvag_recovered_count']}/"
            f"{multiburst_gap['gap_metrics']['gap_aware_failure_count']}"
        ),
        f"- BVAG regressions vs gap-aware: {multiburst_gap['gap_metrics']['bvag_regressions_vs_gap_aware']}",
        (
            "- BVAG/exact ratio (feasible): "
            f"mean={multiburst_gap['ratio_vs_exact']['vag_budget_aware']['mean']:.4f}, "
            f"min={multiburst_gap['ratio_vs_exact']['vag_budget_aware']['min']:.4f}"
            if multiburst_gap["ratio_vs_exact"]["vag_budget_aware"]["mean"] is not None
            else "- BVAG/exact ratio (feasible): n/a"
        ),
        (
            "- Mean step where budget filter first activates: "
            f"{multiburst_gap['gap_metrics']['first_budget_block_mean_step']:.2f}"
            if multiburst_gap["gap_metrics"]["first_budget_block_mean_step"] is not None
            else "- Mean step where budget filter first activates: n/a"
        ),
        (
            "- Mean weight rank of blocked candidate: "
            f"{multiburst_gap['gap_metrics']['first_budget_block_mean_weight_rank']:.2f}"
            if multiburst_gap["gap_metrics"]["first_budget_block_mean_weight_rank"] is not None
            else "- Mean weight rank of blocked candidate: n/a"
        ),
        (
            "- Mean B_lb at first block: "
            f"{multiburst_gap['gap_metrics']['first_budget_block_mean_blb']:.2f}"
            if multiburst_gap["gap_metrics"]["first_budget_block_mean_blb"] is not None
            else "- Mean B_lb at first block: n/a"
        ),
        "",
        "## No-gap Regression Check",
        "",
        (
            "- Bursty: BVAG == VAG gap-aware? "
            f"{'yes' if data['cohorts']['bursty_no_gap']['no_gap_bvag_vs_gapaware']['identical'] else 'no'}"
        ),
        (
            "- Multi-burst: BVAG == VAG gap-aware? "
            f"{'yes' if data['cohorts']['multiburst_no_gap']['no_gap_bvag_vs_gapaware']['identical'] else 'no'}"
        ),
        "",
    ]
    return "\n".join(lines)


def run_bvag_evaluation() -> dict:
    bursty_gap_g = _load_reference_gap(GAP_ADVERSARIAL_PATH, fallback=0.14022757962841484)
    multiburst_gap_g = _load_reference_gap(MULTIBURST_GAP_PATH, fallback=0.4)

    grammar_bursty_gap = GrammarConfig.parametric(1, max_temporal_gap=bursty_gap_g)
    grammar_multiburst_gap = GrammarConfig.parametric(1, max_temporal_gap=multiburst_gap_g)
    grammar_no_gap = GrammarConfig.parametric(1, max_temporal_gap=float("inf"))

    timer = ExperimentTimer()

    bursty_with_gap_rows = _run_bursty_cohort(grammar_bursty_gap)
    bursty_no_gap_rows = _run_bursty_cohort(grammar_no_gap)
    multiburst_with_gap_rows = _run_multiburst_cohort(grammar_multiburst_gap)
    multiburst_no_gap_rows = _run_multiburst_cohort(grammar_no_gap)

    cohorts = {
        "bursty_with_gap": _summarize_cohort(bursty_with_gap_rows, with_gap=True),
        "bursty_no_gap": _summarize_cohort(bursty_no_gap_rows, with_gap=False),
        "multiburst_with_gap": _summarize_cohort(multiburst_with_gap_rows, with_gap=True),
        "multiburst_no_gap": _summarize_cohort(multiburst_no_gap_rows, with_gap=False),
    }

    results = {
        "settings": {
            "seed_range": [min(SEEDS), max(SEEDS)],
            "bursty_epsilons": BURSTY_EPSILONS,
            "n_events": N_EVENTS,
            "n_actors": N_ACTORS,
            "focal_actor": FOCAL_ACTOR,
            "pool_strategy": POOL_STRATEGY,
            "n_anchors": N_ANCHORS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "injection_top_n": INJECTION_TOP_N,
            "bursty_gap_threshold": bursty_gap_g,
            "multiburst_gap_threshold": multiburst_gap_g,
        },
        "cohorts": cohorts,
        "per_case": {
            "bursty_with_gap": bursty_with_gap_rows,
            "bursty_no_gap": bursty_no_gap_rows,
            "multiburst_with_gap": multiburst_with_gap_rows,
            "multiburst_no_gap": multiburst_no_gap_rows,
        },
    }

    metadata = ExperimentMetadata(
        name="bvag_evaluation",
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=len(bursty_with_gap_rows) + len(bursty_no_gap_rows) + len(multiburst_with_gap_rows) + len(multiburst_no_gap_rows),
        n_extractions=(len(bursty_with_gap_rows) + len(bursty_no_gap_rows) + len(multiburst_with_gap_rows) + len(multiburst_no_gap_rows)) * 6,
        seed_range=(min(SEEDS), max(SEEDS)),
        parameters={
            "bursty_gap_threshold": bursty_gap_g,
            "multiburst_gap_threshold": multiburst_gap_g,
        },
    )

    save_results("bvag_evaluation", results, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": results}


def _print_summary(results: dict) -> None:
    bursty_gap = results["cohorts"]["bursty_with_gap"]
    multiburst_gap = results["cohorts"]["multiburst_with_gap"]

    def rate(cohort: dict, key: str) -> float:
        return float(cohort["validity"][key]["valid_rate"])

    print("BVAG Evaluation Results")
    print("=" * 24)
    print(
        "Bursty + Gap:\n"
        f"  Gap-aware VAG validity: {100.0 * rate(bursty_gap, 'vag_gap_aware'):.1f}%\n"
        f"  BVAG validity: {100.0 * rate(bursty_gap, 'vag_budget_aware'):.1f}%\n"
        f"  BVAG recovered from gap-aware failures: {bursty_gap['gap_metrics']['bvag_recovered_count']}/"
        f"{bursty_gap['gap_metrics']['gap_aware_failure_count']}\n"
        f"  BVAG regressions vs gap-aware: {bursty_gap['gap_metrics']['bvag_regressions_vs_gap_aware']}\n"
        f"  Mean first budget-block step: {bursty_gap['gap_metrics']['first_budget_block_mean_step']}"
    )
    print(
        "Multi-burst + Gap:\n"
        f"  Gap-aware VAG validity: {100.0 * rate(multiburst_gap, 'vag_gap_aware'):.1f}%\n"
        f"  BVAG validity: {100.0 * rate(multiburst_gap, 'vag_budget_aware'):.1f}%\n"
        f"  BVAG recovered from gap-aware failures: {multiburst_gap['gap_metrics']['bvag_recovered_count']}/"
        f"{multiburst_gap['gap_metrics']['gap_aware_failure_count']}\n"
        f"  BVAG regressions vs gap-aware: {multiburst_gap['gap_metrics']['bvag_regressions_vs_gap_aware']}\n"
        f"  Mean first budget-block step: {multiburst_gap['gap_metrics']['first_budget_block_mean_step']}"
    )


if __name__ == "__main__":
    payload = run_bvag_evaluation()
    _print_summary(payload["results"])
