#!/usr/bin/env python3
"""Generate all book figures from repo experiment artifacts.

Usage:
    python scripts/generate_figures.py

Reads CSVs from endogenous_context_theory/results/raw/ and PNGs from
endogenous_context_theory/results/figures/, writes outputs to book/figures/.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BOOK = Path(__file__).resolve().parents[1]
REPO = BOOK.parent
RAW = REPO / "endogenous_context_theory" / "results" / "raw"
FIGS_SRC = REPO / "endogenous_context_theory" / "results" / "figures"
OUT = BOOK / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _beautify(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Figure 1: k-j heatmap (Ch3) ──────────────────────────────────────
def fig_kj_heatmap() -> None:
    """Build the k-j impossibility heatmap from test_17 streaming data.

    We use committed_validity as a function of k and epsilon to show
    the impossibility boundary.  For a direct k-vs-j_dev plot we
    synthesise the data: at each (k, epsilon) the expected j_dev is
    approximately epsilon * n, so we bin accordingly.
    """
    rows = read_csv(RAW / "test_17_tropical_streaming.csv")

    # Group by (k, n) and use epsilon to control j_dev-like behaviour
    by_k: dict[int, dict[float, float]] = defaultdict(dict)
    for r in rows:
        k = int(r["k"])
        eps = float(r["epsilon"])
        val = float(r["committed_validity"])
        by_k[k][eps] = val

    k_values = sorted(by_k.keys())
    eps_values = sorted({eps for d in by_k.values() for eps in d})

    mat = np.full((len(k_values), len(eps_values)), np.nan)
    for i, k in enumerate(k_values):
        for j, eps in enumerate(eps_values):
            if eps in by_k[k]:
                mat[i, j] = by_k[k][eps]

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=(7.8, 3.4), constrained_layout=True)
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        extent=[
            min(eps_values) - 0.05,
            max(eps_values) + 0.05,
            min(k_values) - 0.5,
            max(k_values) + 0.5,
        ],
    )

    ax.set_xlabel(r"Front-loading $\epsilon$ (proxy for $j_{\mathrm{dev}}$)")
    ax.set_ylabel(r"$k$ (min DEVELOPMENT prefix)")
    ax.set_xticks(eps_values)
    ax.set_yticks(k_values)
    _beautify(ax)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Committed validity rate")

    fig.savefig(OUT / "kj_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  kj_heatmap.pdf")


# ── Figure 2: Pareto curve (Ch8) ─────────────────────────────────────
def fig_pareto_curve() -> None:
    """Build quality-latency Pareto curve from test_17 streaming data.

    We treat epsilon as a proxy for the patience parameter f.
    Higher epsilon means more events are front-loaded, which
    approximates higher patience.  committed_validity is quality,
    and epsilon itself serves as the latency proxy.
    """
    rows = read_csv(RAW / "test_17_tropical_streaming.csv")

    # Average across n values for each (epsilon, k) pair, pick k=2 as
    # representative
    by_eps: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if int(r["k"]) == 2:
            eps = float(r["epsilon"])
            by_eps[eps].append(float(r["committed_validity"]))

    eps_vals = sorted(by_eps.keys())
    validity = [np.mean(by_eps[e]) for e in eps_vals]

    # Tropical validity as ceiling
    by_eps_trop: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if int(r["k"]) == 2:
            eps = float(r["epsilon"])
            by_eps_trop[eps].append(float(r["tropical_validity"]))

    trop_validity = [np.mean(by_eps_trop[e]) for e in eps_vals]

    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)

    ax.plot(eps_vals, validity, marker="o", linewidth=2.1,
            color="#d7301f", label="Committed (commit-now)")
    ax.plot(eps_vals, trop_validity, marker="s", linewidth=2.1,
            color="#2ca25f", label="Deferred (tropical)")
    ax.fill_between(eps_vals, validity, trop_validity,
                    color="#c7e9c0", alpha=0.3)

    mid_idx = len(eps_vals) // 2
    ax.annotate(
        "Deferred-commitment gain",
        xy=(eps_vals[mid_idx], (validity[mid_idx] + trop_validity[mid_idx]) / 2),
        fontsize=9, color="#333333", ha="center",
    )

    ax.set_xlabel(r"Patience $f$ (front-loading $\epsilon$)")
    ax.set_ylabel("Validity rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(min(eps_vals), max(eps_vals))
    _beautify(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower right")

    fig.savefig(OUT / "test_48c_pareto_curve.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  test_48c_pareto_curve.png")


def main() -> None:
    print("Generating book figures...")
    fig_kj_heatmap()
    fig_pareto_curve()
    print("Done.")


if __name__ == "__main__":
    main()
