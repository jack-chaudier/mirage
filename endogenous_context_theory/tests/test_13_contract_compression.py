from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.compression import (
    contract_guarded_compress,
    naive_compress,
    semantic_regret,
    solve_with_budget,
)
from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 13: Contract-Guarded Compression"
CLAIM = "Claim: no-absorption contract improves pivot preservation while sacrificing some compression." 


def _metrics(full_weight: float, valid: bool, weight: float) -> Dict[str, float]:
    return {
        "raw_validity": float(valid),
        "pivot_preservation": float(valid and abs(weight - full_weight) <= 1e-12),
        "semantic_regret": semantic_regret(full_weight, weight),
    }


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    n = 200
    k = 3
    epsilon = 0.5
    n_focal = 80
    M_values = [1, 5, 10, 25]
    retentions = [0.9, 0.7, 0.5, 0.3, 0.1]
    seeds = range(200)

    rows: List[Dict[str, float]] = []

    for retention in retentions:
        for seed in seeds:
            events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
            full = solve_with_budget(events, k=k, M=None)

            guarded = contract_guarded_compress(events, k=k, retention=retention, seed=seed + 991)
            naive = naive_compress(events, retention=retention, seed=seed + 991, never_drop_focal=True)

            for M in M_values:
                naive_solve = solve_with_budget(naive.retained_events, k=k, M=M)
                guard_solve = solve_with_budget(guarded.retained_events, k=k, M=M)

                naive_m = _metrics(full.pivot_weight, naive_solve.valid, naive_solve.pivot_weight)
                guard_m = _metrics(full.pivot_weight, guard_solve.valid, guard_solve.pivot_weight)

                rows.append(
                    {
                        "method": "naive",
                        "M": M,
                        "retention_target": retention,
                        "seed": seed,
                        "achieved_retention": naive.achieved_retention,
                        **naive_m,
                    }
                )
                rows.append(
                    {
                        "method": "contract",
                        "M": M,
                        "retention_target": retention,
                        "seed": seed,
                        "achieved_retention": guarded.achieved_retention,
                        **guard_m,
                    }
                )

    raw_df = pd.DataFrame(rows)
    save_csv(raw_df, raw_dir, "test_13_contract_compression_raw.csv")

    summary = (
        raw_df.groupby(["method", "M", "retention_target"], as_index=False)
        .agg(
            raw_validity=("raw_validity", "mean"),
            pivot_preservation=("pivot_preservation", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            achieved_retention=("achieved_retention", "mean"),
        )
        .sort_values(["M", "retention_target", "method"], ascending=[True, False, True])
    )
    save_csv(summary, raw_dir, "test_13_contract_compression_summary.csv")
    print_table(summary, max_rows=40)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, ls in [("naive", "--"), ("contract", "-")]:
        sub = summary[(summary["method"] == method) & (summary["M"] == 10)].sort_values("retention_target")
        ax.plot(
            1.0 - sub["retention_target"],
            sub["pivot_preservation"],
            marker="o",
            linestyle=ls,
            label=f"{method} (M=10)",
        )
    ax.set_xlabel("compression rate")
    ax.set_ylabel("pivot preservation")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Contract Guard vs Naive Compression")
    ax.legend()
    save_figure(fig, fig_dir, "test_13_contract_vs_naive.png")

    naive_pres = summary[summary["method"] == "naive"]["pivot_preservation"].mean()
    contract_pres = summary[summary["method"] == "contract"]["pivot_preservation"].mean()
    naive_ret = summary[summary["method"] == "naive"]["achieved_retention"].mean()
    contract_ret = summary[summary["method"] == "contract"]["achieved_retention"].mean()

    improved = contract_pres > naive_pres
    tighter = contract_ret >= naive_ret  # guard should typically keep more events (higher retention).

    verdict = "PASS" if improved else "FAIL"
    print(
        f"\nVerdict: {verdict} | contract_pres={contract_pres:.3f}, naive_pres={naive_pres:.3f}, "
        f"contract_ret={contract_ret:.3f}, naive_ret={naive_ret:.3f}, guard_keeps_more={tighter}"
    )

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "contract_preservation_mean": float(contract_pres),
        "naive_preservation_mean": float(naive_pres),
    }


if __name__ == "__main__":
    run()
