from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.compression import deterministic_mirage_witness, semantic_regret, solve_with_budget
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 12: Deterministic Mirage Witness"
CLAIM = "Claim: one fixed sequence demonstrates feasible full solve, infeasible M=1 compression, and degraded feasible M=10 compression."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    k = 3
    full_events, compressed_events, removed_eids = deterministic_mirage_witness(k=k)

    full = solve_with_budget(full_events, k=k, M=None)
    comp_m1 = solve_with_budget(compressed_events, k=k, M=1)
    comp_m10 = solve_with_budget(compressed_events, k=k, M=10)

    regret_m10 = semantic_regret(full.pivot_weight, comp_m10.pivot_weight)

    rows: List[Dict[str, float]] = [
        {
            "state": "full",
            "valid": full.valid,
            "pivot_weight": full.pivot_weight,
            "removed_eids": "[]",
        },
        {
            "state": "compressed_M1",
            "valid": comp_m1.valid,
            "pivot_weight": comp_m1.pivot_weight,
            "removed_eids": str(removed_eids),
        },
        {
            "state": "compressed_M10",
            "valid": comp_m10.valid,
            "pivot_weight": comp_m10.pivot_weight,
            "removed_eids": str(removed_eids),
        },
    ]

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_12_deterministic_witness.csv")
    print_table(df)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.bar(df["state"], df["pivot_weight"].replace(float("-inf"), 0.0), color=["#2ca02c", "#d62728", "#ff7f0e"])
    ax.set_ylabel("selected pivot weight")
    ax.set_title("Deterministic Mirage Witness")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, fig_dir, "test_12_deterministic_witness.png")

    cond_full = full.valid
    cond_m1 = not comp_m1.valid
    cond_m10 = comp_m10.valid
    cond_regret = regret_m10 > 0.30

    verdict = "PASS" if (cond_full and cond_m1 and cond_m10 and cond_regret) else "FAIL"
    print(
        f"\nVerdict: {verdict} | full_feasible={cond_full}, M1_infeasible={cond_m1}, "
        f"M10_feasible={cond_m10}, regret={regret_m10:.3f}"
    )

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "regret_m10": regret_m10,
    }


if __name__ == "__main__":
    run()
