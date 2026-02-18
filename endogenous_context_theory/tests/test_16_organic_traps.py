from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv, save_figure, set_plot_style
from src.streaming import min_gap_predicts_trap, streaming_outcome


TEST_NAME = "Test 16: Organic Oscillation Traps"
CLAIM = "Claim: committed streaming has high trap rates; min_gap < k has 100% trap recall (0 FN)."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ns = [100, 200, 500, 1000]
    epsilons = [0.1, 0.3, 0.5, 0.7]
    ks = [1, 2, 3]
    seeds = range(200)

    rows: List[Dict[str, float]] = []

    for n in ns:
        n_focal = max(10, n // 2)
        for eps in epsilons:
            for k in ks:
                traps = 0
                fn = 0
                trials = 0

                for seed in seeds:
                    events = bursty_generator(n=n, n_focal=n_focal, epsilon=eps, seed=seed)
                    out = streaming_outcome(events, k=k)
                    pred = min_gap_predicts_trap(events, k=k)

                    if out.trap:
                        traps += 1
                        if not pred:
                            fn += 1
                    trials += 1

                recall = 1.0 if traps == 0 else 1.0 - (fn / traps)
                rows.append(
                    {
                        "n": n,
                        "epsilon": eps,
                        "k": k,
                        "trials": trials,
                        "trap_rate": traps / trials,
                        "traps": traps,
                        "false_negatives": fn,
                        "recall": recall,
                    }
                )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_16_organic_traps.csv")
    print_table(df.sort_values(["n", "epsilon", "k"]), max_rows=40)

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for ax, n in zip(axes.ravel(), ns):
        sub = df[df["n"] == n]
        hm = sub.pivot(index="epsilon", columns="k", values="trap_rate").sort_index()
        im = ax.imshow(hm.values, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
        ax.set_title(f"n={n}")
        ax.set_xticks(np.arange(len(hm.columns)))
        ax.set_xticklabels(hm.columns.tolist())
        ax.set_yticks(np.arange(len(hm.index)))
        ax.set_yticklabels(hm.index.tolist())
        ax.set_xlabel("k")
        ax.set_ylabel("epsilon")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("trap rate")
    save_figure(fig, fig_dir, "test_16_trap_rate_heatmaps.png")

    total_fn = int(df["false_negatives"].sum())
    verdict = pass_fail(total_fn)
    print(f"\nVerdict: {verdict} | total false negatives for min_gap<k predictor = {total_fn}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "false_negatives": total_fn,
        "mean_trap_rate": float(df["trap_rate"].mean()),
    }


if __name__ == "__main__":
    run()
