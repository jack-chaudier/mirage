from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.holographic_tree import HolographicContextTree
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv
from src.tropical_semiring import TropicalContext, compose_tropical


TEST_NAME = "Test 06: Incremental Consistency"
CLAIM = "Claim: After every append, tree root equals sequential fold (no intermediate drift)."


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

    n = 2000
    k = 3
    seeds = range(50)
    n_focal = 700

    rows: List[Dict[str, float]] = []
    total_checks = 0
    total_violations = 0

    for seed in seeds:
        events = bursty_generator(n=n, n_focal=n_focal, epsilon=0.5, seed=seed)
        tree = HolographicContextTree(k=k)
        seq_ctx = TropicalContext.empty(k)

        violations = 0
        checks = 0

        for e in events:
            tree.append(e)
            seq_ctx = compose_tropical(seq_ctx, TropicalContext.from_event(e, k))
            root_ctx = tree.get_root_summary()
            checks += 1
            if not _same_context(root_ctx, seq_ctx):
                violations += 1

        total_checks += checks
        total_violations += violations
        rows.append(
            {
                "seed": seed,
                "checks": checks,
                "violations": violations,
                "pass_rate": 1.0 - (violations / checks),
            }
        )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_06_incremental_consistency.csv")
    print_table(df.describe().reset_index().rename(columns={"index": "stat"}), max_rows=30)

    verdict = pass_fail(total_violations)
    print(f"\nVerdict: {verdict} | total violations = {total_violations} over {total_checks} checks")

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "violations": int(total_violations),
        "checks": int(total_checks),
    }


if __name__ == "__main__":
    run()
