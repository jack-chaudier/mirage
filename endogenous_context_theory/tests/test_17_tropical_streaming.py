from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv, save_figure, set_plot_style
from src.streaming import streaming_outcome


TEST_NAME = "Test 17: Tropical Streaming vs Committed"
CLAIM = "Claim: tropical streaming validity matches offline finite validity exactly and exceeds committed validity."


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
    mismatch = 0

    for n in ns:
        n_focal = max(10, n // 2)
        for eps in epsilons:
            for k in ks:
                committed_sum = 0.0
                tropical_sum = 0.0
                finite_sum = 0.0
                trials = 0

                for seed in seeds:
                    events = bursty_generator(n=n, n_focal=n_focal, epsilon=eps, seed=seed)
                    out = streaming_outcome(events, k)
                    committed_sum += float(out.committed_valid)
                    tropical_sum += float(out.tropical_valid)
                    finite_sum += float(out.finite_valid)
                    if out.tropical_valid != out.finite_valid:
                        mismatch += 1
                    trials += 1

                rows.append(
                    {
                        "n": n,
                        "epsilon": eps,
                        "k": k,
                        "trials": trials,
                        "committed_validity": committed_sum / trials,
                        "tropical_validity": tropical_sum / trials,
                        "finite_validity": finite_sum / trials,
                    }
                )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_17_tropical_streaming.csv")
    print_table(df.sort_values(["n", "epsilon", "k"]), max_rows=40)

    overall = pd.DataFrame(
        {
            "policy": ["committed", "tropical", "finite"],
            "validity": [
                df["committed_validity"].mean(),
                df["tropical_validity"].mean(),
                df["finite_validity"].mean(),
            ],
        }
    )
    save_csv(overall, raw_dir, "test_17_tropical_streaming_overall.csv")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(overall["policy"], overall["validity"], color=["#d62728", "#2ca02c", "#1f77b4"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean validity rate")
    ax.set_title("Committed vs Tropical vs Finite Validity")
    save_figure(fig, fig_dir, "test_17_policy_validity_bar.png")

    verdict = pass_fail(mismatch)
    print(f"\nVerdict: {verdict} | tropical-finite mismatches = {mismatch}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "mismatches": int(mismatch),
        "committed_mean": float(overall[overall["policy"] == "committed"]["validity"].iloc[0]),
        "tropical_mean": float(overall[overall["policy"] == "tropical"]["validity"].iloc[0]),
    }


if __name__ == "__main__":
    run()
