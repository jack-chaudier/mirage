from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import adversarial_oscillation_generator, bursty_generator, multi_burst_generator, streaming_generator
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv, save_figure, set_plot_style
from src.tropical_semiring import build_original_context, build_tropical_context, count_occupied_slots, max_slot_for_weight


TEST_NAME = "Test 03: Monoid Subsumption"
CLAIM = "Claim: Tropical matches original monoid on best pivot while retaining richer multi-slot information."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ks = [1, 2, 3, 5]
    seeds = range(200)
    n = 220
    n_focal = 90

    gen_specs = [
        ("bursty", lambda seed: bursty_generator(n=n, n_focal=n_focal, epsilon=0.5, seed=seed)),
        ("multi_burst", lambda seed: multi_burst_generator(n=n, n_focal=n_focal, n_bursts=3, seed=seed)),
        ("streaming", lambda seed: streaming_generator(n=n, n_focal=n_focal, epsilon=0.5, seed=seed)),
        ("adversarial", lambda seed: adversarial_oscillation_generator(n=n, k=3, osc_period=8, seed=seed)),
    ]

    rows: List[Dict[str, float]] = []

    for gname, gfn in gen_specs:
        for k in ks:
            w_violations = 0
            slot_violations = 0
            multi_slot_instances = 0
            trials = 0

            for seed in seeds:
                events = gfn(seed)
                tctx = build_tropical_context(events, k)
                mctx = build_original_context(events)

                t_w = float(np.max(tctx.W))
                m_w = float(mctx.w_star)
                if not (
                    (np.isneginf(t_w) and np.isneginf(m_w))
                    or abs(t_w - m_w) <= 1e-12
                ):
                    w_violations += 1

                slot = max_slot_for_weight(tctx.W, t_w)
                expected_slot = min(mctx.d_pre, k)
                if slot != expected_slot:
                    slot_violations += 1

                if count_occupied_slots(tctx.W) > 1:
                    multi_slot_instances += 1

                trials += 1

            rows.append(
                {
                    "generator": gname,
                    "k": k,
                    "trials": trials,
                    "w_match_rate": 1.0 - (w_violations / trials),
                    "slot_match_rate": 1.0 - (slot_violations / trials),
                    "w_violations": w_violations,
                    "slot_violations": slot_violations,
                    "multi_slot_rate": multi_slot_instances / trials,
                }
            )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_03_monoid_subsumption.csv")
    print_table(df.sort_values(["generator", "k"]))

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = df.copy()
    plot_df["cell"] = plot_df["generator"] + "_k" + plot_df["k"].astype(str)
    ax.bar(plot_df["cell"], plot_df["multi_slot_rate"], color="#287D8EFF")
    ax.set_ylabel("fraction with >1 occupied slot")
    ax.set_xlabel("cell")
    ax.set_xticklabels(plot_df["cell"], rotation=60, ha="right")
    ax.set_title("Tropical Richness Beyond Monoid")
    save_figure(fig, fig_dir, "test_03_monoid_richness.png")

    total_violations = int(df["w_violations"].sum() + df["slot_violations"].sum())
    verdict = pass_fail(total_violations)
    print(f"\nVerdict: {verdict} | total violations = {total_violations}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "violations": total_violations,
        "trials": int(df["trials"].sum()),
        "multi_slot_mean": float(df["multi_slot_rate"].mean()),
    }


if __name__ == "__main__":
    run()
