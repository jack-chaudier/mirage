from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv, save_figure, set_plot_style
from src.tropical_semiring import Event, TropicalContext, build_tropical_context, compose_tropical


TEST_NAME = "Test 04: Absorbing Left Ideal"
CLAIM = "Claim: Under committed semantics, absorbed states are permanent; uncommitted suffixes can escape."


def _incremental_absorption_index(events: List[Event], k: int) -> int | None:
    ctx = TropicalContext.empty(k)
    seen_focal = False
    for i, e in enumerate(events):
        if e.is_focal:
            seen_focal = True
        ctx = compose_tropical(ctx, TropicalContext.from_event(e, k))
        if seen_focal and np.isneginf(ctx.W[k]):
            return i
    return None


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    epsilons = [0.7, 0.8, 0.9]
    ks = [2, 3, 4]
    seeds = range(200)
    n = 220
    n_focal = 90

    rows: List[Dict[str, float]] = []

    for eps in epsilons:
        for k in ks:
            committed_violations = 0
            absorbed_cases = 0
            uncommitted_escapes = 0

            for seed in seeds:
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=eps, seed=seed)
                idx = _incremental_absorption_index(events, k)
                if idx is None:
                    continue

                absorbed_cases += 1
                prefix = events[: idx + 1]
                suffix = events[idx + 1 :]

                demoted_suffix = [
                    Event(
                        eid=e.eid,
                        timestamp=e.timestamp,
                        weight=e.weight,
                        actor=e.actor,
                        is_focal=False,
                    )
                    for e in suffix
                ]

                committed_final = build_tropical_context(prefix + demoted_suffix, k)
                if not np.isneginf(committed_final.W[k]):
                    committed_violations += 1

                uncommitted_final = build_tropical_context(prefix + suffix, k)
                if uncommitted_final.W[k] > float("-inf"):
                    uncommitted_escapes += 1

            rows.append(
                {
                    "epsilon": eps,
                    "k": k,
                    "absorbed_cases": absorbed_cases,
                    "committed_violations": committed_violations,
                    "uncommitted_escape_rate": (
                        uncommitted_escapes / absorbed_cases if absorbed_cases > 0 else np.nan
                    ),
                    "committed_escape_rate": (
                        committed_violations / absorbed_cases if absorbed_cases > 0 else np.nan
                    ),
                }
            )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_04_absorbing_ideal.csv")
    print_table(df.sort_values(["epsilon", "k"]))

    set_plot_style()
    plot_df = df.copy()
    plot_df["cell"] = "eps=" + plot_df["epsilon"].astype(str) + ",k=" + plot_df["k"].astype(str)

    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, plot_df["committed_escape_rate"], width=width, label="committed")
    ax.bar(x + width / 2, plot_df["uncommitted_escape_rate"], width=width, label="uncommitted")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["cell"], rotation=45, ha="right")
    ax.set_ylabel("escape rate")
    ax.set_title("Absorption Escape Rate: Committed vs Uncommitted")
    ax.legend()
    save_figure(fig, fig_dir, "test_04_absorption_escape_rates.png")

    total_violations = int(df["committed_violations"].sum())
    verdict = pass_fail(total_violations)
    print(f"\nVerdict: {verdict} | committed violations = {total_violations}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "violations": total_violations,
        "absorbed_cases": int(df["absorbed_cases"].sum()),
        "uncommitted_escape_rate_mean": float(df["uncommitted_escape_rate"].mean()),
    }


if __name__ == "__main__":
    run()
