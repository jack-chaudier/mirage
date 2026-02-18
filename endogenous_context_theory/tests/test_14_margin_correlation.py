from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from src.compression import naive_compress, solve_with_budget
from src.generators import bursty_generator
from src.pivot_margin import build_top2_context, pivot_margin
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 14: Margin-Mirage Correlation"
CLAIM = "Hypothesis: larger pivot margin predicts higher pivot preservation under compression."


def _finite_margin(m: float, cap: float) -> float:
    if np.isfinite(m):
        return m
    return cap


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
    retention = 0.5
    seeds = range(500)

    rows: List[Dict[str, float]] = []

    for seed in seeds:
        events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
        full = solve_with_budget(events, k=k, M=None)
        top2 = build_top2_context(events, k=k)
        margin = pivot_margin(top2, k)

        comp = naive_compress(events, retention=retention, seed=seed + 404, never_drop_focal=True)
        comp_solve = solve_with_budget(comp.retained_events, k=k, M=None)

        preserved = float(
            full.valid and comp_solve.valid and abs(full.pivot_weight - comp_solve.pivot_weight) <= 1e-12
        )

        rows.append(
            {
                "seed": seed,
                "margin": margin,
                "preserved": preserved,
                "full_valid": float(full.valid),
                "compressed_valid": float(comp_solve.valid),
            }
        )

    raw_df = pd.DataFrame(rows)
    finite = raw_df["margin"].replace([np.inf, -np.inf], np.nan).dropna()
    cap = float(finite.max() * 1.5) if len(finite) else 1.0
    raw_df["margin_finite"] = raw_df["margin"].apply(lambda m: _finite_margin(float(m), cap))

    save_csv(raw_df, raw_dir, "test_14_margin_correlation_raw.csv")

    if raw_df["margin_finite"].nunique() < 2 or raw_df["preserved"].nunique() < 2:
        corr, pval = 0.0, 1.0
    else:
        corr, pval = pointbiserialr(raw_df["preserved"], raw_df["margin_finite"])

    try:
        quartile_bins = pd.qcut(
            raw_df["margin_finite"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
        raw_df["quartile"] = quartile_bins.astype(str)
    except ValueError:
        raw_df["quartile"] = "Q_all"

    quart = (
        raw_df.groupby("quartile", as_index=False)
        .agg(pivot_preservation=("preserved", "mean"), count=("seed", "count"))
        .sort_values("quartile")
    )
    save_csv(quart, raw_dir, "test_14_margin_quartiles.csv")

    print_table(quart)
    print(f"\nPoint-biserial correlation = {corr:.4f}, p-value = {pval:.3e}")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(quart["quartile"], quart["pivot_preservation"], color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("margin quartile")
    ax.set_ylabel("pivot preservation rate")
    ax.set_title("Pivot Margin Predicts Preservation")
    save_figure(fig, fig_dir, "test_14_margin_quartiles.png")

    verdict = "PASS" if corr > 0 else "FAIL"
    print(f"\nVerdict: {verdict} | correlation={corr:.4f}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "correlation": float(corr),
        "p_value": float(pval),
    }


if __name__ == "__main__":
    run()
