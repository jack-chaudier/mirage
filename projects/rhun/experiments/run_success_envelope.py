"""Experiment 38: Empirical Success Envelope via logistic regression on k-j boundary data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import minimize
from scipy.special import expit, logit

from rhun.experiments.runner import ExperimentMetadata, ExperimentTimer, save_results, utc_timestamp


OUTPUT_NAME = "success_envelope"
ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

KJ_DATA_PATH = ROOT / "experiments" / "output" / "kj_boundary_final.json"


def _load_kj_data() -> list[tuple[int, int, int, int]]:
    """Load (k, j_dev, n_instances, n_valid) from kj_boundary_final.json."""
    with open(KJ_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)
    rbk = results["results_by_k"]

    cells: list[tuple[int, int, int, int]] = []
    for k_str, k_data in rbk.items():
        k = int(k_str)
        bjdp = k_data.get("by_j_dev_pool", {})
        for j_str, j_data in bjdp.items():
            j = int(j_str)
            n = int(j_data["n"])
            v = int(j_data["greedy_valid"])
            cells.append((k, j, n, v))
    return cells


def _expand_binary(cells: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand aggregated cells to individual binary observations for logistic regression."""
    j_list: list[int] = []
    k_list: list[int] = []
    y_list: list[int] = []

    for k, j, n, v in cells:
        j_list.extend([j] * n)
        k_list.extend([k] * n)
        y_list.extend([1] * v + [0] * (n - v))

    return np.array(j_list, dtype=float), np.array(k_list, dtype=float), np.array(y_list, dtype=float)


def _neg_log_likelihood(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for logistic regression."""
    z = X @ params
    # Clip for numerical stability
    z = np.clip(z, -500, 500)
    ll = np.sum(y * z - np.log1p(np.exp(z)))
    return -ll


def _fit_logistic(
    j_arr: np.ndarray, k_arr: np.ndarray, y_arr: np.ndarray
) -> dict:
    """Fit P(valid) = sigmoid(alpha * j_dev + beta * k + gamma)."""
    # Design matrix: [j_dev, k, 1]
    X = np.column_stack([j_arr, k_arr, np.ones_like(j_arr)])

    # Initial guess
    x0 = np.array([0.1, -0.5, 0.0])

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(X, y_arr),
        method="L-BFGS-B",
        options={"maxiter": 10000},
    )

    alpha, beta, gamma = result.x

    # Predicted probabilities
    z = X @ result.x
    p_hat = expit(z)

    # AUC (manual calculation)
    pos_scores = p_hat[y_arr == 1]
    neg_scores = p_hat[y_arr == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos > 0 and n_neg > 0:
        auc = float(np.mean(pos_scores[:, None] > neg_scores[None, :]))
    else:
        auc = None

    # Calibration: compare predicted vs observed in decile bins
    calibration = []
    for q in range(10):
        lo = q * 0.1
        hi = (q + 1) * 0.1
        mask = (p_hat >= lo) & (p_hat < hi)
        if mask.sum() > 0:
            calibration.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "predicted_mean": float(np.mean(p_hat[mask])),
                "observed_mean": float(np.mean(y_arr[mask])),
            })

    # P(valid) = 0.95 contour: j_dev_95(k) = (logit(0.95) - beta * k - gamma) / alpha
    logit_95 = float(logit(0.95))
    logit_50 = float(logit(0.50))

    if abs(alpha) > 1e-12:
        contour_95 = lambda k_val: (logit_95 - beta * k_val - gamma) / alpha
        contour_50 = lambda k_val: (logit_50 - beta * k_val - gamma) / alpha
    else:
        contour_95 = lambda k_val: float("inf")
        contour_50 = lambda k_val: float("inf")

    # Evaluate contours at k = 1..5
    contour_95_by_k = {str(k): float(contour_95(k)) for k in range(1, 6)}
    contour_50_by_k = {str(k): float(contour_50(k)) for k in range(1, 6)}

    # Linear fit: j_dev_95 ≈ C * k + D
    k_vals_contour = np.array([1, 2, 3, 4, 5], dtype=float)
    j_vals_contour = np.array([contour_95(k) for k in k_vals_contour])
    # Simple linear fit
    A_lin = np.column_stack([k_vals_contour, np.ones_like(k_vals_contour)])
    coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, j_vals_contour, rcond=None)
    C, D = float(coeffs_lin[0]), float(coeffs_lin[1])

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "converged": bool(result.success),
        "n_iterations": int(result.nit),
        "neg_log_likelihood": float(result.fun),
        "auc": auc,
        "calibration": calibration,
        "contour_95_by_k": contour_95_by_k,
        "contour_50_by_k": contour_50_by_k,
        "linear_fit_95": {
            "C": C,
            "D": D,
            "equation": f"j_dev_95(k) ≈ {C:.2f} * k + {D:.2f}",
        },
    }


def _generate_envelope_heatmap(
    cells: list[tuple[int, int, int, int]],
    fit_result: dict,
) -> None:
    """Generate k-j heatmap with impossibility diagonal and empirical 95% envelope."""
    # Build heatmap matrix
    k_values = sorted(set(c[0] for c in cells))
    max_j = 60

    mat = np.full((len(k_values), max_j + 1), np.nan)
    for k, j, n, v in cells:
        if k in k_values and 0 <= j <= max_j:
            i = k_values.index(k)
            mat[i, j] = v / n if n > 0 else np.nan

    masked = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e0e0e0")

    fig, ax = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        extent=[-0.5, max_j + 0.5, min(k_values) - 0.5, max(k_values) + 0.5],
    )

    # Impossibility diagonal (solid)
    diag_x = np.linspace(0, min(max_j, max(k_values)), 200)
    ax.plot(diag_x, diag_x, color="black", linestyle="-", linewidth=1.4,
            label=r"$j_{dev} = k$ (impossibility)")

    # 95% success envelope (dashed)
    alpha = fit_result["alpha"]
    beta = fit_result["beta"]
    gamma = fit_result["gamma"]
    logit_95 = logit(0.95)
    if abs(alpha) > 1e-12:
        k_line = np.linspace(1, max(k_values), 100)
        j_95_line = (logit_95 - beta * k_line - gamma) / alpha
        ax.plot(j_95_line, k_line, color="black", linestyle="--", linewidth=1.4,
                label="95% success envelope")

    ax.set_xlabel(r"$j_{dev}$ (development-eligible events before TP)")
    ax.set_ylabel(r"$k$ (min DEVELOPMENT prefix)")
    ax.set_xticks(np.arange(0, max_j + 1, 10))
    ax.set_yticks(k_values)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Greedy validity rate")

    fig.savefig(FIGURE_DIR / "kj_heatmap_envelope.pdf")
    plt.close(fig)


def _summary_markdown(data: dict, meta: ExperimentMetadata) -> str:
    fit = data["logistic_fit"]
    lines = [
        "# Experiment 38: Empirical Success Envelope",
        "",
        f"Generated: {meta.timestamp}",
        f"Runtime: {meta.runtime_seconds:.2f}s",
        "",
        "## Logistic Regression",
        "",
        f"Model: P(valid) = sigmoid({fit['alpha']:.4f} * j_dev + {fit['beta']:.4f} * k + {fit['gamma']:.4f})",
        f"Converged: {fit['converged']}",
        f"AUC: {fit['auc']:.4f}" if fit['auc'] else "AUC: N/A",
        "",
        "## P(valid) = 0.95 Contour",
        "",
        f"Linear approximation: {fit['linear_fit_95']['equation']}",
        "",
        "| k | j_dev at P=0.95 | j_dev at P=0.50 |",
        "|---|----------------:|----------------:|",
    ]
    for k in range(1, 6):
        j95 = fit["contour_95_by_k"][str(k)]
        j50 = fit["contour_50_by_k"][str(k)]
        lines.append(f"| {k} | {j95:.2f} | {j50:.2f} |")

    lines.extend([
        "",
        "## Calibration",
        "",
        "| Bin | N | Predicted | Observed |",
        "|-----|----:|----------:|---------:|",
    ])
    for cal in fit["calibration"]:
        lines.append(
            f"| {cal['bin']} | {cal['n']} | {cal['predicted_mean']:.3f} | {cal['observed_mean']:.3f} |"
        )

    lines.extend([
        "",
        f"## Figure",
        "",
        "Updated heatmap with envelope: `paper/figures/kj_heatmap_envelope.pdf`",
    ])
    return "\n".join(lines)


def run_success_envelope() -> dict:
    timer = ExperimentTimer()

    print("Loading k-j boundary data...", flush=True)
    cells = _load_kj_data()
    total_instances = sum(c[2] for c in cells)
    print(f"Loaded {len(cells)} cells, {total_instances} instances", flush=True)

    # Filter to k >= 1 (k=0 is degenerate)
    cells_filtered = [(k, j, n, v) for k, j, n, v in cells if k >= 1]
    print(f"After filtering k>=1: {len(cells_filtered)} cells", flush=True)

    j_arr, k_arr, y_arr = _expand_binary(cells_filtered)
    print(f"Binary observations: {len(y_arr)} (valid={int(y_arr.sum())}, invalid={int(len(y_arr) - y_arr.sum())})", flush=True)

    print("Fitting logistic regression...", flush=True)
    fit_result = _fit_logistic(j_arr, k_arr, y_arr)
    print(f"  alpha={fit_result['alpha']:.4f}, beta={fit_result['beta']:.4f}, gamma={fit_result['gamma']:.4f}", flush=True)
    print(f"  AUC={fit_result['auc']:.4f}" if fit_result['auc'] else "  AUC=N/A", flush=True)
    print(f"  95% envelope: {fit_result['linear_fit_95']['equation']}", flush=True)

    print("Generating heatmap with envelope...", flush=True)
    _generate_envelope_heatmap(cells, fit_result)

    data = {
        "n_cells": len(cells_filtered),
        "n_instances": int(len(y_arr)),
        "n_valid": int(y_arr.sum()),
        "logistic_fit": fit_result,
    }

    metadata = ExperimentMetadata(
        name=OUTPUT_NAME,
        timestamp=utc_timestamp(),
        runtime_seconds=timer.elapsed(),
        n_graphs=total_instances,
        n_extractions=0,
        seed_range=(0, 0),
        parameters={
            "source": str(KJ_DATA_PATH),
            "filter": "k >= 1",
            "model": "logistic(j_dev, k)",
        },
    )
    save_results(OUTPUT_NAME, data, metadata, summary_formatter=_summary_markdown)
    return {"metadata": metadata, "results": data}


if __name__ == "__main__":
    run_success_envelope()
