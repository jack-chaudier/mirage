from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.holographic_tree import HolographicContextTree
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 07: Scaling Performance"
CLAIM = "Claim: append ~ O(n log n), root/pivot queries ~ O(log n), depth ~ log2(n)."


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ns = [100, 500, 1000, 5000, 10000, 50000, 100000]
    k = 3
    seed = 7

    rows: List[Dict[str, float]] = []

    for n in ns:
        n_focal = max(10, n // 3)
        events = bursty_generator(n=n, n_focal=n_focal, epsilon=0.5, seed=seed)

        tree = HolographicContextTree(k=k)

        t0 = time.perf_counter()
        for e in events:
            tree.append(e)
        append_total = time.perf_counter() - t0

        q_root = 200
        t1 = time.perf_counter()
        for _ in range(q_root):
            tree.get_root_summary()
        root_per_query = (time.perf_counter() - t1) / q_root

        q_pivot = 200
        t2 = time.perf_counter()
        for _ in range(q_pivot):
            tree.find_pivot_block()
        pivot_per_query = (time.perf_counter() - t2) / q_pivot

        rows.append(
            {
                "n": n,
                "append_total_s": append_total,
                "append_per_event_us": (append_total / n) * 1e6,
                "root_query_us": root_per_query * 1e6,
                "pivot_query_us": pivot_per_query * 1e6,
                "depth": tree.depth(),
                "forest_size": tree.forest_size(),
                "log2_n": math.log2(n),
            }
        )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_07_scaling.csv")
    print_table(df)

    set_plot_style()

    n_vals = df["n"].to_numpy(dtype=float)
    append_vals = df["append_total_s"].to_numpy(dtype=float)
    root_vals = df["root_query_us"].to_numpy(dtype=float)
    pivot_vals = df["pivot_query_us"].to_numpy(dtype=float)
    depth_vals = df["depth"].to_numpy(dtype=float)

    ref_nlogn = n_vals * np.log2(n_vals)
    ref_nlogn = ref_nlogn / ref_nlogn[0] * append_vals[0]

    ref_log = np.log2(n_vals)
    ref_log_root = ref_log / ref_log[0] * root_vals[0]
    ref_log_pivot = ref_log / ref_log[0] * pivot_vals[0]

    ref_depth = np.log2(n_vals)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.loglog(n_vals, append_vals, "o-", label="append total")
    ax.loglog(n_vals, ref_nlogn, "--", label="O(n log n) ref")
    ax.set_xlabel("n")
    ax.set_ylabel("seconds")
    ax.set_title("Append Scaling")
    ax.legend()

    ax = axes[1]
    ax.loglog(n_vals, root_vals, "o-", label="root query")
    ax.loglog(n_vals, pivot_vals, "s-", label="pivot query")
    ax.loglog(n_vals, ref_log_root, "--", label="O(log n) ref (root)")
    ax.loglog(n_vals, ref_log_pivot, ":", label="O(log n) ref (pivot)")
    ax.set_xlabel("n")
    ax.set_ylabel("microseconds/query")
    ax.set_title("Query Scaling")
    ax.legend()

    ax = axes[2]
    ax.loglog(n_vals, depth_vals, "o-", label="depth")
    ax.loglog(n_vals, ref_depth, "--", label="log2(n) ref")
    ax.set_xlabel("n")
    ax.set_ylabel("depth")
    ax.set_title("Depth Scaling")
    ax.legend()

    save_figure(fig, fig_dir, "test_07_scaling_loglog.png")

    depth_ok = bool(np.all(df["depth"].to_numpy() <= np.ceil(df["log2_n"].to_numpy()) + 1))
    verdict = "PASS" if depth_ok else "FAIL"
    print(f"\nVerdict: {verdict} | depth bound respected = {depth_ok}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "max_depth": float(df["depth"].max()),
    }


if __name__ == "__main__":
    run()
