from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.compression import naive_compress, semantic_regret, solve_with_budget
from src.generators import bursty_generator
from src.pivot_margin import build_top2_context, pivot_margin
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 15: Margin-Guided Adaptive Compression"
CLAIM = "Claim: margin-guided retention improves preservation at similar or better compression ratio."


def _eval(events, k: int, retention: float, seed: int) -> Dict[str, float]:
    full = solve_with_budget(events, k=k, M=None)
    comp = naive_compress(events, retention=retention, seed=seed, never_drop_focal=True)
    comp_solve = solve_with_budget(comp.retained_events, k=k, M=None)
    return {
        "raw_validity": float(comp_solve.valid),
        "pivot_preservation": float(
            full.valid and comp_solve.valid and abs(comp_solve.pivot_weight - full.pivot_weight) <= 1e-12
        ),
        "semantic_regret": semantic_regret(full.pivot_weight, comp_solve.pivot_weight),
        "achieved_retention": comp.achieved_retention,
        "achieved_compression": 1.0 - comp.achieved_retention,
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
    seeds = range(500)

    margins: List[float] = []
    events_cache: Dict[int, List] = {}

    for seed in seeds:
        events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
        events_cache[seed] = events
        m = pivot_margin(build_top2_context(events, k=k), k)
        margins.append(float(m if np.isfinite(m) else 1e6))

    median_margin = float(np.median(margins))

    rows: List[Dict[str, float]] = []
    for seed in seeds:
        events = events_cache[seed]
        m = pivot_margin(build_top2_context(events, k=k), k)
        m_fin = float(m if np.isfinite(m) else 1e6)

        adaptive_retention = 0.3 if m_fin > median_margin else 0.7

        adaptive = _eval(events, k=k, retention=adaptive_retention, seed=seed + 700)
        uniform = _eval(events, k=k, retention=0.5, seed=seed + 700)

        rows.append({"method": "adaptive", "seed": seed, "margin": m_fin, **adaptive})
        rows.append({"method": "uniform_0.5", "seed": seed, "margin": m_fin, **uniform})

    raw_df = pd.DataFrame(rows)
    save_csv(raw_df, raw_dir, "test_15_adaptive_compression_raw.csv")

    summary = (
        raw_df.groupby("method", as_index=False)
        .agg(
            raw_validity=("raw_validity", "mean"),
            pivot_preservation=("pivot_preservation", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            achieved_retention=("achieved_retention", "mean"),
            achieved_compression=("achieved_compression", "mean"),
        )
        .sort_values("method")
    )
    save_csv(summary, raw_dir, "test_15_adaptive_compression_summary.csv")
    print_table(summary)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    axes[0].bar(summary["method"], summary["pivot_preservation"], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Pivot Preservation")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(summary["method"], summary["achieved_compression"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Achieved Compression")
    axes[1].tick_params(axis="x", rotation=20)

    save_figure(fig, fig_dir, "test_15_adaptive_vs_uniform.png")

    ad = summary[summary["method"] == "adaptive"].iloc[0]
    un = summary[summary["method"] == "uniform_0.5"].iloc[0]

    better_pres = float(ad["pivot_preservation"]) > float(un["pivot_preservation"])
    nonworse_comp = float(ad["achieved_compression"]) >= float(un["achieved_compression"]) - 0.02
    verdict = "PASS" if better_pres and nonworse_comp else "FAIL"

    print(
        f"\nVerdict: {verdict} | better_preservation={better_pres}, "
        f"adaptive_compression={ad['achieved_compression']:.3f}, uniform={un['achieved_compression']:.3f}"
    )

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "adaptive_preservation": float(ad["pivot_preservation"]),
        "uniform_preservation": float(un["pivot_preservation"]),
    }


if __name__ == "__main__":
    run()
