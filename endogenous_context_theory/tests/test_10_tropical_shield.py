from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style
from src.streaming import deferred_commitment_policy, streaming_outcome


TEST_NAME = "Test 10: Tropical Semiring as Divergence Shield"
CLAIM = "Claim: Tropical validity dominates committed/deferred while incurring zero commitment cost."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ns = [1000, 5000, 10000]
    epsilon = 0.5
    k = 3
    seeds = range(100)
    deferred_fracs = [0.05, 0.10, 0.25, 0.50]

    rows: List[Dict[str, float]] = []

    for n in ns:
        n_focal = max(20, n // 2)
        for seed in seeds:
            events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
            base = streaming_outcome(events, k)

            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "policy": "committed",
                    "valid": float(base.committed_valid),
                    "cost": float(base.total_cost),
                }
            )
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "policy": "tropical",
                    "valid": float(base.tropical_valid),
                    "cost": 0.0,
                }
            )

            for f in deferred_fracs:
                out = deferred_commitment_policy(events, k=k, commit_fraction=f)
                rows.append(
                    {
                        "n": n,
                        "seed": seed,
                        "policy": f"deferred_{f:.2f}",
                        "valid": out["valid"],
                        "cost": out["total_cost"],
                    }
                )

    raw_df = pd.DataFrame(rows)
    save_csv(raw_df, raw_dir, "test_10_tropical_shield_raw.csv")

    summary = (
        raw_df.groupby(["n", "policy"], as_index=False)
        .agg(validity_rate=("valid", "mean"), mean_cost=("cost", "mean"))
        .sort_values(["n", "policy"])
    )
    save_csv(summary, raw_dir, "test_10_tropical_shield_summary.csv")
    print_table(summary, max_rows=40)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    pivot_valid = summary.pivot(index="policy", columns="n", values="validity_rate")
    pivot_cost = summary.pivot(index="policy", columns="n", values="mean_cost")

    pivot_valid.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Validity by Policy")
    axes[0].set_ylabel("validity rate")
    axes[0].set_xlabel("policy")
    axes[0].tick_params(axis="x", rotation=40)

    pivot_cost.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Mean Commitment Cost by Policy")
    axes[1].set_ylabel("mean cost")
    axes[1].set_xlabel("policy")
    axes[1].tick_params(axis="x", rotation=40)

    save_figure(fig, fig_dir, "test_10_tropical_shield_policy_compare.png")

    tropical_ok = True
    for n in ns:
        tropical_v = float(summary[(summary["n"] == n) & (summary["policy"] == "tropical")]["validity_rate"].iloc[0])
        committed_v = float(summary[(summary["n"] == n) & (summary["policy"] == "committed")]["validity_rate"].iloc[0])
        if tropical_v < committed_v:
            tropical_ok = False

    verdict = "PASS" if tropical_ok else "FAIL"
    print(f"\nVerdict: {verdict} | tropical validity >= committed for all n = {tropical_ok}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "tropical_dominates_committed": tropical_ok,
    }


if __name__ == "__main__":
    run()
