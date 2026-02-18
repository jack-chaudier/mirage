from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import adversarial_oscillation_generator, bursty_generator, multi_burst_generator, streaming_generator
from src.reporting import ensure_result_dirs, pass_fail, print_header, print_table, save_csv, save_figure, set_plot_style
from src.tropical_semiring import build_tropical_context, compose_tropical
from tests.helpers import _same_context


TEST_NAME = "Test 02: Associativity"
CLAIM = "Claim: (A⊗B)⊗C == A⊗(B⊗C) across all tested split points and generators."


def _split_points(n: int, seed: int, n_splits: int = 5) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n_splits:
        s1 = int(rng.integers(1, n - 2))
        s2 = int(rng.integers(s1 + 1, n - 1))
        points.append((s1, s2))
    return points


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
            violations = 0
            trials = 0
            for seed in seeds:
                events = gfn(seed)
                for s1, s2 in _split_points(len(events), seed=seed + 991 * (k + 1), n_splits=5):
                    A = build_tropical_context(events[:s1], k)
                    B = build_tropical_context(events[s1:s2], k)
                    C = build_tropical_context(events[s2:], k)

                    left = compose_tropical(compose_tropical(A, B), C)
                    right = compose_tropical(A, compose_tropical(B, C))

                    trials += 1
                    if not _same_context(left, right):
                        violations += 1

            rows.append(
                {
                    "generator": gname,
                    "k": k,
                    "trials": trials,
                    "violations": violations,
                    "pass_rate": 1.0 - (violations / trials),
                }
            )

    df = pd.DataFrame(rows)
    save_csv(df, raw_dir, "test_02_associativity.csv")
    print_table(df.sort_values(["generator", "k"]))

    set_plot_style()
    hm = df.pivot(index="generator", columns="k", values="pass_rate").sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    im = ax.imshow(hm.values, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(hm.columns)))
    ax.set_xticklabels(hm.columns.tolist())
    ax.set_yticks(np.arange(len(hm.index)))
    ax.set_yticklabels(hm.index.tolist())
    ax.set_xlabel("k")
    ax.set_ylabel("Generator")
    ax.set_title("Associativity Pass Rate Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pass rate")
    save_figure(fig, fig_dir, "test_02_associativity_heatmap.png")

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
