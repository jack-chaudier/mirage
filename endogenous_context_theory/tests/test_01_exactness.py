from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator, multi_burst_generator
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv, save_figure, set_plot_style
from src.tropical_semiring import build_tropical_context, brute_force_tropical_context
from tests.helpers import _same_context


TEST_NAME = "Test 01: Tropical Exactness"
CLAIM = "Claim: Left-fold tropical composition exactly matches brute-force W across all cells."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
    bursts = [2, 3, 4]
    ks = [1, 2, 3, 4, 5]
    seeds = range(200)
    n = 200
    n_focal = 80

    rows: List[Dict[str, float]] = []

    for k in ks:
        for eps in epsilons:
            violations = 0
            trials = 0
            for seed in seeds:
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=eps, seed=seed)
                fold_ctx = build_tropical_context(events, k)
                brute_ctx = brute_force_tropical_context(events, k)
                trials += 1
                if not _same_context(fold_ctx, brute_ctx):
                    violations += 1
            rows.append(
                {
                    "generator": "bursty",
                    "param": eps,
                    "k": k,
                    "trials": trials,
                    "violations": violations,
                    "match_rate": 1.0 - (violations / trials),
                }
            )

        for nb in bursts:
            violations = 0
            trials = 0
            for seed in seeds:
                events = multi_burst_generator(n=n, n_focal=n_focal, n_bursts=nb, seed=seed)
                fold_ctx = build_tropical_context(events, k)
                brute_ctx = brute_force_tropical_context(events, k)
                trials += 1
                if not _same_context(fold_ctx, brute_ctx):
                    violations += 1
            rows.append(
                {
                    "generator": "multi_burst",
                    "param": nb,
                    "k": k,
                    "trials": trials,
                    "violations": violations,
                    "match_rate": 1.0 - (violations / trials),
                }
            )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_01_exactness.csv")

    summary = (
        df.groupby(["generator", "param"], as_index=False)
        .agg(trials=("trials", "sum"), violations=("violations", "sum"), match_rate=("match_rate", "mean"))
        .sort_values(["generator", "param"])
    )
    print_table(summary)

    set_plot_style()
    pivot = df.copy()
    pivot["row"] = pivot["generator"] + "_" + pivot["param"].astype(str)
    hm = pivot.pivot(index="row", columns="k", values="match_rate").sort_index()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(hm.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(hm.columns)))
    ax.set_xticklabels(hm.columns.tolist())
    ax.set_yticks(np.arange(len(hm.index)))
    ax.set_yticklabels(hm.index.tolist())
    ax.set_xlabel("k")
    ax.set_ylabel("Generator/Parameter")
    ax.set_title("Exactness Pass Rate Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("match rate")
    save_figure(fig, fig_dir, "test_01_exactness_heatmap.png")

    total_violations = int(df["violations"].sum())
    verdict = pass_fail(total_violations)
    print(f"\nVerdict: {verdict} | total violations = {total_violations}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "violations": total_violations,
        "trials": int(df["trials"].sum()),
    }


if __name__ == "__main__":
    run()
