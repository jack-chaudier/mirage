#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(rel: str) -> dict:
    with (ROOT / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_success_envelope_fit() -> tuple[float, float]:
    """Return linear 95% envelope fit j_dev ~= C * k + D."""
    try:
        fit = load_json("experiments/output/success_envelope.json")["results"]["logistic_fit"]["linear_fit_95"]
        return float(fit["C"]), float(fit["D"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 5.31, 12.23


def _beautify_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_kj_heatmap() -> None:
    data = load_json("experiments/output/kj_boundary_final.json")["results"]
    by_k = data["results_by_k"]
    k_values = sorted(int(k) for k in by_k.keys())
    max_j = 60

    mat = np.full((len(k_values), max_j + 1), np.nan)
    for i, k in enumerate(k_values):
        by_j = by_k[str(k)].get("by_j_dev_pool", {})
        for j_str, cell in by_j.items():
            try:
                j = int(j_str)
            except ValueError:
                continue
            if 0 <= j <= max_j:
                mat[i, j] = float(cell["greedy_valid_rate"])

    masked = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        extent=[-0.5, max_j + 0.5, min(k_values) - 0.5, max(k_values) + 0.5],
    )
    diag_x = np.linspace(0, min(max_j, max(k_values)), 200)
    ax.plot(
        diag_x,
        diag_x,
        color="black",
        linestyle="-",
        linewidth=1.4,
        label=r"$j_{dev} = k$ (impossibility)",
    )

    slope, intercept = load_success_envelope_fit()
    k_line = np.linspace(min(k_values), max(k_values), 200)
    j_95_line = slope * k_line + intercept
    visible = (j_95_line >= 0.0) & (j_95_line <= max_j)
    ax.plot(
        j_95_line[visible],
        k_line[visible],
        color="#ffd166",
        linestyle="--",
        linewidth=1.4,
        label="95% success envelope",
    )

    ax.set_xlabel(r"$j_{dev}$ (development-eligible events before TP)")
    ax.set_ylabel(r"$k$ (min DEVELOPMENT prefix)")
    ax.set_xticks(np.arange(0, max_j + 1, 10))
    ax.set_yticks(k_values)
    ax.tick_params(axis="both")

    ax.legend(loc="upper right", frameon=False, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Greedy validity rate")

    fig.savefig(
        OUT / "kj_heatmap_envelope.pdf",
        bbox_inches="tight",
        transparent=False,
    )
    fig.savefig(
        OUT / "kj_heatmap.pdf",
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(fig)


def fig_position_sweep() -> None:
    rows = load_json("experiments/output/position_sweep.json")["results"]["rows"]
    rows = sorted(rows, key=lambda r: r["epsilon"])

    x = [float(r["epsilon"]) for r in rows]
    y_valid = [float(r["validity_rate"]) for r in rows]
    y_absorb = [float(r["absorption_rate"]) for r in rows]
    y_theorem = [float(r["theorem_accuracy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    ax.plot(x, y_valid, marker="o", linewidth=1.8, label="Validity rate")
    ax.plot(x, y_absorb, marker="s", linewidth=1.8, label="Absorption rate")
    ax.plot(x, y_theorem, marker="^", linewidth=1.8, label="Theorem accuracy")

    ax.set_xlabel(r"Front-loading $\epsilon$")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(min(x), max(x))
    _beautify_axes(ax)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower left")

    fig.savefig(OUT / "position_sweep.pdf")
    plt.close(fig)


def fig_hierarchy() -> None:
    topo = load_json("experiments/output/tp_solver_evaluation_m25.json")["results"]["topologies"]

    bursty = topo["bursty_gap"]["aggregate"]
    multiburst = topo["multiburst_gap"]["aggregate"]

    labels = [
        "Greedy",
        "Span-VAG",
        "Gap-VAG",
        "BVAG",
        "TP-Solver\n(M=25)",
        "Oracle",
    ]
    bursty_vals = [
        bursty["greedy"]["valid_rate"],
        bursty["vag_span_only"]["valid_rate"],
        bursty["vag_gap_aware"]["valid_rate"],
        bursty["vag_budget_aware"]["valid_rate"],
        bursty["tp_conditioned_solver"]["valid_rate"],
        bursty["exact_oracle"]["valid_rate"],
    ]
    multiburst_vals = [
        multiburst["greedy"]["valid_rate"],
        multiburst["vag_span_only"]["valid_rate"],
        multiburst["vag_gap_aware"]["valid_rate"],
        multiburst["vag_budget_aware"]["valid_rate"],
        multiburst["tp_conditioned_solver"]["valid_rate"],
        multiburst["exact_oracle"]["valid_rate"],
    ]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    bars_b = ax.bar(x - width / 2, np.array(bursty_vals) * 100.0, width, label="Bursty+gap")
    bars_m = ax.bar(x + width / 2, np.array(multiburst_vals) * 100.0, width, label="Multi-burst+gap")

    # Highlight adversarial span-VAG collapse on multi-burst.
    bars_m[1].set_color("#b2182b")
    bars_m[1].set_edgecolor("black")
    bars_m[1].set_linewidth(1.0)

    ax.set_ylabel("Validity (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _beautify_axes(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.savefig(OUT / "hierarchy_comparison.pdf")
    plt.close(fig)


def fig_antagonism() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 2.6), constrained_layout=True)

    # Timeline with two bursts separated by a sparse valley.
    ax.axvspan(0.05, 0.32, color="#92c5de", alpha=0.4)
    ax.axvspan(0.68, 0.95, color="#92c5de", alpha=0.4)
    ax.axvspan(0.32, 0.68, color="#f4a582", alpha=0.2)

    burst_points = np.array([0.08, 0.12, 0.18, 0.25, 0.72, 0.78, 0.84, 0.91])
    valley_bridge = np.array([0.45, 0.55])

    ax.scatter(burst_points, np.ones_like(burst_points) * 0.7, s=45, c="#2166ac", label="High-weight burst events")
    ax.scatter(valley_bridge, np.ones_like(valley_bridge) * 0.35, s=45, c="#b2182b", label="Low-weight bridge events")

    # Span-VAG preference (skips valley bridges, causing gap violation).
    ax.plot([0.08, 0.25, 0.72, 0.91], [0.7, 0.7, 0.7, 0.7], color="#2166ac", linewidth=2)
    ax.plot([0.25, 0.72], [0.7, 0.7], color="#b2182b", linestyle="--", linewidth=1.8)

    ax.text(0.5, 0.83, "Span viability favors burst endpoints", ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.15, "Gap feasibility requires valley bridges", ha="center", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([])
    ax.set_xlabel("Normalized time")
    _beautify_axes(ax, grid_axis="x")
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=8)

    fig.savefig(OUT / "antagonism.pdf")
    plt.close(fig)


def fig_validity_mirage_tp() -> None:
    curve = load_json("experiments/output/context_retention_sweep_tp_semantics_m10.json")["results"]["curve"]
    curve = sorted(curve, key=lambda row: float(row["retention"]), reverse=True)

    retention = np.array([float(row["retention"]) for row in curve])
    x = 1.0 - retention
    y_naive_valid = np.array([float(row["naive_valid_rate"]) for row in curve])
    y_contract_valid = np.array([float(row["contract_valid_rate"]) for row in curve])
    y_naive_pivot = np.array([float(row["naive_pivot_preservation_rate"]) for row in curve])
    y_contract_pivot = np.array([float(row["contract_guarded_pivot_preservation_rate"]) for row in curve])

    fig, ax = plt.subplots(figsize=(7.8, 3.9), constrained_layout=True)
    ax.fill_between(
        x,
        y_naive_pivot,
        y_naive_valid,
        color="#fddbc7",
        alpha=0.35,
        label="Mirage gap (naive)",
    )
    ax.plot(x, y_naive_valid, marker="o", linewidth=2.1, color="#1f77b4", label="Naive validity")
    ax.plot(
        x,
        y_contract_valid,
        marker="s",
        linewidth=2.1,
        color="#2ca25f",
        label="Contract validity",
    )
    ax.plot(
        x,
        y_naive_pivot,
        marker="^",
        linewidth=2.0,
        linestyle="--",
        color="#d7301f",
        label="Naive pivot preservation",
    )
    ax.plot(
        x,
        y_contract_pivot,
        marker="d",
        linewidth=2.0,
        linestyle="--",
        color="#984ea3",
        label="Contract pivot preservation",
    )

    ax.set_xlabel("Compression rate (1 - retention)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(min(x), max(x))
    ax.set_xticks(x)
    _beautify_axes(ax, grid_axis="y")
    ax.annotate(
        "Pivot consistency drops\nwhile validity stays flat",
        xy=(0.85, 0.45),
        xytext=(0.66, 0.30),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=9,
        color="#333333",
    )
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")

    fig.savefig(OUT / "validity_mirage_tp.pdf")
    plt.close(fig)


def main() -> None:
    fig_kj_heatmap()
    fig_position_sweep()
    fig_hierarchy()
    fig_antagonism()
    fig_validity_mirage_tp()


if __name__ == "__main__":
    main()
