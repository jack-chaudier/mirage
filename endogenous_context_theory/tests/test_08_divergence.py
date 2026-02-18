from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, power_law_fit, print_header, print_table, save_csv, save_figure, set_plot_style
from src.streaming import running_max_record_process


TEST_NAME = "Test 08: Divergence Characterization"
CLAIM = "Claim: cumulative committed reassignment cost grows super-linearly in n under record shifts."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    epsilons = [0.3, 0.5, 0.7]
    k = 3
    seeds = range(200)

    rows: List[Dict[str, float]] = []

    for eps in epsilons:
        for n in ns:
            n_focal = max(10, n // 2)
            for seed in seeds:
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=eps, seed=seed)
                rp = running_max_record_process(events)
                total_cost = rp.total_cost
                n_records = len(rp.record_positions)
                rows.append(
                    {
                        "epsilon": eps,
                        "n": n,
                        "k": k,
                        "seed": seed,
                        "n_records": n_records,
                        "total_cost": total_cost,
                        "cost_per_n": total_cost / n,
                        "cost_per_nlogn": total_cost / (n * max(1e-12, math.log(n))),
                    }
                )

    raw_df = pd.DataFrame(rows)
    save_csv(raw_df, raw_dir, "test_08_divergence_raw.csv")

    agg = (
        raw_df.groupby(["epsilon", "n"], as_index=False)
        .agg(
            mean_records=("n_records", "mean"),
            mean_total_cost=("total_cost", "mean"),
            mean_cost_per_n=("cost_per_n", "mean"),
            mean_cost_per_nlogn=("cost_per_nlogn", "mean"),
        )
        .sort_values(["epsilon", "n"])
    )
    save_csv(agg, raw_dir, "test_08_divergence_summary.csv")
    print_table(agg, max_rows=30)

    fit_rows: List[Dict[str, float]] = []
    for eps in epsilons:
        sub = agg[agg["epsilon"] == eps]
        fit = power_law_fit(sub["n"].to_numpy(), sub["mean_total_cost"].to_numpy())
        fit_rows.append({"epsilon": eps, **fit})

    pooled = (
        agg.groupby("n", as_index=False)
        .agg(mean_total_cost=("mean_total_cost", "mean"))
        .sort_values("n")
    )
    pooled_fit = power_law_fit(pooled["n"].to_numpy(), pooled["mean_total_cost"].to_numpy())
    fit_rows.append({"epsilon": "pooled", **pooled_fit})

    fit_df = pd.DataFrame(fit_rows)
    save_csv(fit_df, raw_dir, "test_08_divergence_fit.csv")
    print("\nPower-law fits (cost ~ a*n^b):")
    print_table(fit_df)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    colors = {0.3: "#1f77b4", 0.5: "#ff7f0e", 0.7: "#2ca02c"}
    for eps in epsilons:
        sub = agg[agg["epsilon"] == eps]
        fit = fit_df[fit_df["epsilon"] == eps].iloc[0]
        x = sub["n"].to_numpy(dtype=float)
        y = sub["mean_total_cost"].to_numpy(dtype=float)
        ax.loglog(x, y, "o", color=colors[eps], label=f"eps={eps} mean")
        y_hat = fit["a"] * (x ** fit["b"])
        ax.loglog(x, y_hat, "-", color=colors[eps], alpha=0.8, label=f"fit eps={eps}, b={fit['b']:.2f}")

    x_ref = np.array(ns, dtype=float)
    y1 = x_ref
    y1 = y1 / y1[0] * pooled["mean_total_cost"].to_numpy()[0]
    y15 = x_ref ** 1.5
    y15 = y15 / y15[0] * pooled["mean_total_cost"].to_numpy()[0]
    ax.loglog(x_ref, y1, "--", color="black", label="slope b=1.0")
    ax.loglog(x_ref, y15, ":", color="black", label="slope b=1.5")

    ax.set_xlabel("n")
    ax.set_ylabel("mean total reassignment cost")
    ax.set_title("Divergence: log-log Cost Scaling")
    ax.legend(ncol=2)
    save_figure(fig, fig_dir, "test_08_divergence_loglog.png")

    pooled_b = float(pooled_fit["b"])
    verdict = "PASS" if pooled_b > 1.0 else "FAIL"
    print(
        f"\nVerdict: {verdict} | pooled exponent b={pooled_b:.3f} "
        f"(95% CI [{pooled_fit['b_ci_low']:.3f}, {pooled_fit['b_ci_high']:.3f}])"
    )

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "b": pooled_b,
        "b_ci_low": float(pooled_fit["b_ci_low"]),
        "b_ci_high": float(pooled_fit["b_ci_high"]),
    }


if __name__ == "__main__":
    run()
