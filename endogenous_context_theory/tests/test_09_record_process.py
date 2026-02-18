from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style
from src.streaming import running_max_record_process
from src.tropical_semiring import harmonic_number


TEST_NAME = "Test 09: Record-Process Statistics"
CLAIM = "Claim: record counts scale ~ harmonic/log; min-gap distribution explains trap susceptibility."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    epsilons = [0.3, 0.5, 0.7]
    seeds = range(200)

    rows: List[Dict[str, float]] = []
    gap_rows: List[Dict[str, float]] = []

    for eps in epsilons:
        for n in ns:
            n_focal = max(10, n // 2)
            for seed in seeds:
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=eps, seed=seed)
                rp = running_max_record_process(events)

                rows.append(
                    {
                        "epsilon": eps,
                        "n": n,
                        "n_focal": n_focal,
                        "seed": seed,
                        "n_records": len(rp.record_positions),
                        "min_gap": rp.min_gap if np.isfinite(rp.min_gap) else np.nan,
                    }
                )

                for g in rp.inter_record_gaps:
                    gap_rows.append(
                        {
                            "epsilon": eps,
                            "n": n,
                            "seed": seed,
                            "gap": g,
                        }
                    )

    raw_df = pd.DataFrame(rows)
    gaps_df = pd.DataFrame(gap_rows)

    save_csv(raw_df, raw_dir, "test_09_record_process_raw.csv")
    save_csv(gaps_df, raw_dir, "test_09_record_process_gaps.csv")

    agg = (
        raw_df.groupby("n", as_index=False)
        .agg(mean_records=("n_records", "mean"), mean_min_gap=("min_gap", "mean"), n_focal=("n_focal", "mean"))
        .sort_values("n")
    )
    agg["harmonic_n"] = agg["n"].apply(lambda v: harmonic_number(int(v)))
    agg["harmonic_nfocal"] = agg["n_focal"].apply(lambda v: harmonic_number(int(v)))
    agg["ln_n"] = np.log(agg["n"])

    save_csv(agg, raw_dir, "test_09_record_process_summary.csv")
    print_table(agg)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.plot(agg["n"], agg["mean_records"], "o-", label="mean records")
    ax.plot(agg["n"], agg["ln_n"], "--", label="ln(n)")
    ax.plot(agg["n"], agg["harmonic_n"], ":", label="H_n")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("mean records")
    ax.set_title("Record Counts vs Log/Harmonic References")
    ax.legend()

    ax = axes[1]
    for n in [200, 2000, 20000]:
        sub = raw_df[raw_df["n"] == n]["min_gap"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.4, density=True, label=f"n={n}")
    ax.set_xlabel("minimum gap")
    ax.set_ylabel("density")
    ax.set_title("Min-Gap Distribution by n")
    ax.legend()

    save_figure(fig, fig_dir, "test_09_records_and_min_gap.png")

    mae = float(np.mean(np.abs(agg["mean_records"] - agg["harmonic_nfocal"])))
    verdict = "PASS" if mae <= 1.0 else "FAIL"
    print(f"\nVerdict: {verdict} | MAE(mean_records, H_n_focal) = {mae:.3f}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "mae_to_harmonic_focal": mae,
    }


if __name__ == "__main__":
    run()
