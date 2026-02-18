from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.compression import evaluate_mirage_cell
from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 11: Compression-Induced Validity Mirage"
CLAIM = "Claim: raw validity remains high while pivot preservation collapses under compression."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    n = 200
    k = 3
    M_values = [1, 5, 10, 25]
    retentions = [0.9, 0.7, 0.5, 0.3, 0.1]
    seeds = range(200)
    epsilon = 0.5
    n_focal = 80

    rows: List[Dict[str, float]] = []

    for M in M_values:
        for retention in retentions:
            for seed in seeds:
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
                out = evaluate_mirage_cell(events, k=k, retention=retention, seed=seed + 1234, M=M)
                rows.append(
                    {
                        "M": M,
                        "retention": retention,
                        "compression_rate": 1.0 - retention,
                        "seed": seed,
                        **out,
                    }
                )

    raw_df = pd.DataFrame(rows)
    save_csv(raw_df, raw_dir, "test_11_mirage_raw.csv")

    summary = (
        raw_df.groupby(["M", "retention", "compression_rate"], as_index=False)
        .agg(
            raw_validity=("raw_validity", "mean"),
            pivot_preservation=("pivot_preservation", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            achieved_retention=("achieved_retention", "mean"),
        )
        .sort_values(["M", "retention"], ascending=[True, False])
    )
    save_csv(summary, raw_dir, "test_11_mirage_summary.csv")
    print_table(summary, max_rows=40)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for M in M_values:
        sub = summary[summary["M"] == M].sort_values("compression_rate")
        ax.plot(
            sub["compression_rate"],
            sub["raw_validity"],
            marker="o",
            label=f"Raw valid (M={M})",
        )
        ax.plot(
            sub["compression_rate"],
            sub["pivot_preservation"],
            marker="s",
            linestyle="--",
            label=f"Pivot preserved (M={M})",
        )

    ax.set_xlabel("compression rate (1 - retention)")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Validity Mirage Under Compression")
    ax.legend(ncol=2)
    save_figure(fig, fig_dir, "test_11_validity_mirage.png")

    # Mirage strength: average raw-valid minus pivot-preservation at high compression.
    high_comp = summary[summary["retention"] <= 0.5]
    gap = float((high_comp["raw_validity"] - high_comp["pivot_preservation"]).mean())
    verdict = "PASS" if gap > 0.15 else "FAIL"
    print(f"\nVerdict: {verdict} | average mirage gap (retention<=0.5) = {gap:.3f}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "mirage_gap": gap,
    }


if __name__ == "__main__":
    run()
