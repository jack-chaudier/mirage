from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.holographic_tree import HolographicContextTree
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv
from src.tropical_semiring import build_tropical_context


TEST_NAME = "Test 05: Holographic Exactness"
CLAIM = "Claim: Tree root summary matches exact sequential tropical fold for all tested n and k."


def _same_context(a, b) -> bool:
    if a.d_total != b.d_total:
        return False
    a_inf = np.isneginf(a.W)
    b_inf = np.isneginf(b.W)
    if not np.array_equal(a_inf, b_inf):
        return False
    mask = ~a_inf
    return bool(np.allclose(a.W[mask], b.W[mask], atol=1e-12, rtol=0.0))


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, _ = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    ns = [100, 500, 1000, 5000]
    ks = [1, 3, 5]
    seeds = range(100)

    rows: List[Dict[str, float]] = []

    for n in ns:
        for k in ks:
            violations = 0
            trials = 0
            n_focal = max(10, n // 3)

            for seed in seeds:
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=0.5, seed=seed)

                tree = HolographicContextTree(k=k)
                for e in events:
                    tree.append(e)

                root_ctx = tree.get_root_summary()
                seq_ctx = build_tropical_context(events, k)
                trials += 1
                if not _same_context(root_ctx, seq_ctx):
                    violations += 1

            rows.append(
                {
                    "n": n,
                    "k": k,
                    "trials": trials,
                    "violations": violations,
                    "pass_rate": 1.0 - (violations / trials),
                }
            )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_05_holographic_exactness.csv")
    print_table(df.sort_values(["n", "k"]))

    total_violations = int(df["violations"].sum())
    verdict = pass_fail(total_violations)
    print(f"\nVerdict: {verdict} | total violations = {total_violations}")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "violations": total_violations,
        "trials": int(df["trials"].sum()),
    }


if __name__ == "__main__":
    run()
