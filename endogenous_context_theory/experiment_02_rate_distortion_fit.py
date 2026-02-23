#!/usr/bin/env python3
"""Rate-distortion validation for semantic drift under memory compression.

This experiment compares four fits:

- fit_original: two-candidate, constrained M1 > M2 (legacy baseline)
- fit_A: two-candidate, unconstrained (priority structural fix)
- fit_B: M-candidate with fixed M_eff=10
- fit_C: M-candidate with free M_eff in [2, 25]

All fits use k=3 and are optimized on naive pivot-preservation SSE.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import binom


def _tail_prob_integer_n(n: int, r: float, k: int) -> float:
    """Exact tail P(Bin(n, r) >= k) for integer n, integer k."""

    if k <= 0:
        return 1.0
    if n < 0 or k > n:
        return 0.0
    return float(binom.sf(k - 1, n, r))


def _tail_prob_continuous_n(n: float, r: float, k: int) -> float:
    """Continuous n by linear interpolation between floor/ceil integer n."""

    if n <= 0.0:
        return 0.0
    n0 = int(math.floor(n))
    n1 = int(math.ceil(n))
    if n0 == n1:
        return _tail_prob_integer_n(n0, r, k)
    w = float(n - n0)
    p0 = _tail_prob_integer_n(n0, r, k)
    p1 = _tail_prob_integer_n(n1, r, k)
    return float((1.0 - w) * p0 + w * p1)


def tail_prob(n: float, r: float, k: float) -> float:
    """P(Bin(n, r) >= k) with continuous n and continuous k support."""

    if k <= 0.0:
        return 1.0
    k0 = int(math.floor(k))
    k1 = int(math.ceil(k))
    if k0 == k1:
        return _tail_prob_continuous_n(n=n, r=r, k=k0)
    w = float(k - k0)
    p0 = _tail_prob_continuous_n(n=n, r=r, k=k0)
    p1 = _tail_prob_continuous_n(n=n, r=r, k=k1)
    return float((1.0 - w) * p0 + w * p1)


def predicted_fixed_pivot_feasibility(r: float, m1: float, k: float) -> float:
    """P(X1 >= k), X1 ~ Bin(M1, r)."""

    return tail_prob(n=m1, r=r, k=k)


def predicted_pivot_preservation_two(r: float, m1: float, m2: float, k: float) -> float:
    """Two-candidate preservation: P(X1>=k) / [P(X1>=k) + P(X1<k)P(X2>=k)]."""

    p1 = tail_prob(n=m1, r=r, k=k)
    p2 = tail_prob(n=m2, r=r, k=k)
    denom = p1 + (1.0 - p1) * p2
    if denom <= 0.0:
        return float("nan")
    return float(p1 / denom)


def predicted_raw_validity_two(r: float, m1: float, m2: float, k: float) -> float:
    """Two-candidate raw validity: 1 - P(X1<k)P(X2<k)."""

    p1 = tail_prob(n=m1, r=r, k=k)
    p2 = tail_prob(n=m2, r=r, k=k)
    return float(1.0 - (1.0 - p1) * (1.0 - p2))


def predicted_substitution_two(r: float, m1: float, m2: float, k: float) -> float:
    """Two-candidate substitution probability D(r) = P(X1<k)P(X2>=k)."""

    p1 = tail_prob(n=m1, r=r, k=k)
    p2 = tail_prob(n=m2, r=r, k=k)
    return float((1.0 - p1) * p2)


def predicted_pivot_preservation_m(
    r: float,
    m1: float,
    m2: float,
    k: float,
    m_eff: float,
) -> float:
    """M-candidate preservation with one dominant + (M-1) non-dominant pivots."""

    if m_eff < 2.0:
        return float("nan")
    p1 = tail_prob(n=m1, r=r, k=k)
    p2 = tail_prob(n=m2, r=r, k=k)
    q2 = 1.0 - p2
    survive_non_dom = 1.0 - (q2 ** (m_eff - 1.0))
    denom = p1 + (1.0 - p1) * survive_non_dom
    if denom <= 0.0:
        return float("nan")
    return float(p1 / denom)


def predicted_raw_validity_m(r: float, m1: float, m2: float, k: float, m_eff: float) -> float:
    """M-candidate raw validity: 1 - P(X1<k)P(X2<k)^(M-1)."""

    if m_eff < 2.0:
        return float("nan")
    p1 = tail_prob(n=m1, r=r, k=k)
    p2 = tail_prob(n=m2, r=r, k=k)
    q2 = 1.0 - p2
    return float(1.0 - (1.0 - p1) * (q2 ** (m_eff - 1.0)))


def predicted_substitution_m(r: float, m1: float, m2: float, k: float, m_eff: float) -> float:
    """M-candidate substitution: P(X1<k) * P(at least one non-dominant survives)."""

    if m_eff < 2.0:
        return float("nan")
    p1 = tail_prob(n=m1, r=r, k=k)
    p2 = tail_prob(n=m2, r=r, k=k)
    q2 = 1.0 - p2
    survive_non_dom = 1.0 - (q2 ** (m_eff - 1.0))
    return float((1.0 - p1) * survive_non_dom)


def _predict_pivot_by_fit(fit: dict[str, Any], r: float) -> float:
    if fit["model_type"] == "two":
        return predicted_pivot_preservation_two(r=r, m1=fit["M1"], m2=fit["M2"], k=fit["k"])
    return predicted_pivot_preservation_m(
        r=r,
        m1=fit["M1"],
        m2=fit["M2"],
        k=fit["k"],
        m_eff=fit["M_eff"],
    )


def _predict_raw_by_fit(fit: dict[str, Any], r: float) -> float:
    if fit["model_type"] == "two":
        return predicted_raw_validity_two(r=r, m1=fit["M1"], m2=fit["M2"], k=fit["k"])
    return predicted_raw_validity_m(
        r=r,
        m1=fit["M1"],
        m2=fit["M2"],
        k=fit["k"],
        m_eff=fit["M_eff"],
    )


def _predict_substitution_by_fit(fit: dict[str, Any], r: float) -> float:
    if fit["model_type"] == "two":
        return predicted_substitution_two(r=r, m1=fit["M1"], m2=fit["M2"], k=fit["k"])
    return predicted_substitution_m(
        r=r,
        m1=fit["M1"],
        m2=fit["M2"],
        k=fit["k"],
        m_eff=fit["M_eff"],
    )


def _build_empirical_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    naive = pd.DataFrame(
        {
            "strategy": "naive",
            "retention": [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10],
            "pivot_preservation": [0.790, 0.730, 0.700, 0.630, 0.590, 0.500, 0.480, 0.394, 0.354],
            "fixed_pivot_feasibility": [0.980, 0.970, 0.970, 0.960, 0.920, 0.890, 0.860, 0.850, 0.750],
            "raw_validity": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.990, 0.990],
        }
    )
    contract = pd.DataFrame(
        {
            "strategy": "contract_guarded",
            "retention": [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10],
            "pivot_preservation": [0.770, 0.700, 0.670, 0.570, 0.580, 0.450, 0.404, 0.333, 0.354],
            "fixed_pivot_feasibility": [0.990, 0.980, 0.960, 0.940, 0.930, 0.910, 0.890, 0.870, 0.780],
            "raw_validity": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.990, 0.990, 0.990],
        }
    )
    return naive, contract


def _sse_two(
    m1: float,
    m2: float,
    k: float,
    rs: np.ndarray,
    y_obs: np.ndarray,
) -> float:
    preds = np.array([predicted_pivot_preservation_two(r, m1, m2, k) for r in rs], dtype=float)
    if np.any(~np.isfinite(preds)):
        return 1e9
    return float(np.sum((preds - y_obs) ** 2))


def _sse_m(
    m1: float,
    m2: float,
    k: float,
    m_eff: float,
    rs: np.ndarray,
    y_obs: np.ndarray,
) -> float:
    preds = np.array([predicted_pivot_preservation_m(r, m1, m2, k, m_eff) for r in rs], dtype=float)
    if np.any(~np.isfinite(preds)):
        return 1e9
    return float(np.sum((preds - y_obs) ** 2))


def _fit_original_constrained(
    starts: list[tuple[float, float]],
    rs: np.ndarray,
    y_obs: np.ndarray,
    k: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local: list[dict[str, Any]] = []

    def objective(theta: np.ndarray) -> float:
        m1, m2 = float(theta[0]), float(theta[1])
        if m1 <= m2:
            return 1e6 + (m2 - m1 + 1.0) * 1e4
        return _sse_two(m1=m1, m2=m2, k=k, rs=rs, y_obs=y_obs)

    for m1_start, m2_start in starts:
        result = minimize(
            objective,
            x0=np.array([m1_start, m2_start], dtype=float),
            bounds=[(1.0, 100.0), (1.0, 100.0)],
            method="L-BFGS-B",
        )
        m1_hat, m2_hat = [float(v) for v in result.x]
        local.append(
            {
                "start_m1": m1_start,
                "start_m2": m2_start,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "M1": m1_hat,
                "M2": m2_hat,
                "SSE_naive": _sse_two(m1=m1_hat, m2=m2_hat, k=k, rs=rs, y_obs=y_obs),
            }
        )
    best = min(local, key=lambda row: row["SSE_naive"])
    fit = {
        "fit_name": "fit_original",
        "model_type": "two",
        "M1": float(best["M1"]),
        "M2": float(best["M2"]),
        "k": float(k),
        "M_eff": 2.0,
        "SSE_naive": float(best["SSE_naive"]),
        "local_optima": local,
    }
    return fit, local


def _fit_A_unconstrained(
    starts: list[tuple[float, float]],
    rs: np.ndarray,
    y_obs: np.ndarray,
    k: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local: list[dict[str, Any]] = []

    def objective(theta: np.ndarray) -> float:
        m1, m2 = float(theta[0]), float(theta[1])
        return _sse_two(m1=m1, m2=m2, k=k, rs=rs, y_obs=y_obs)

    for m1_start, m2_start in starts:
        result = minimize(
            objective,
            x0=np.array([m1_start, m2_start], dtype=float),
            bounds=[(1.0, 100.0), (1.0, 100.0)],
            method="L-BFGS-B",
        )
        m1_hat, m2_hat = [float(v) for v in result.x]
        local.append(
            {
                "start_m1": m1_start,
                "start_m2": m2_start,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "M1": m1_hat,
                "M2": m2_hat,
                "SSE_naive": _sse_two(m1=m1_hat, m2=m2_hat, k=k, rs=rs, y_obs=y_obs),
            }
        )
    best = min(local, key=lambda row: row["SSE_naive"])
    fit = {
        "fit_name": "fit_A",
        "model_type": "two",
        "M1": float(best["M1"]),
        "M2": float(best["M2"]),
        "k": float(k),
        "M_eff": 2.0,
        "SSE_naive": float(best["SSE_naive"]),
        "local_optima": local,
    }
    return fit, local


def _fit_B_mfixed(
    starts: list[tuple[float, float]],
    rs: np.ndarray,
    y_obs: np.ndarray,
    k: float,
    m_eff: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local: list[dict[str, Any]] = []

    def objective(theta: np.ndarray) -> float:
        m1, m2 = float(theta[0]), float(theta[1])
        return _sse_m(m1=m1, m2=m2, k=k, m_eff=m_eff, rs=rs, y_obs=y_obs)

    for m1_start, m2_start in starts:
        result = minimize(
            objective,
            x0=np.array([m1_start, m2_start], dtype=float),
            bounds=[(1.0, 100.0), (1.0, 100.0)],
            method="L-BFGS-B",
        )
        m1_hat, m2_hat = [float(v) for v in result.x]
        local.append(
            {
                "start_m1": m1_start,
                "start_m2": m2_start,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "M1": m1_hat,
                "M2": m2_hat,
                "SSE_naive": _sse_m(m1=m1_hat, m2=m2_hat, k=k, m_eff=m_eff, rs=rs, y_obs=y_obs),
            }
        )
    best = min(local, key=lambda row: row["SSE_naive"])
    fit = {
        "fit_name": "fit_B",
        "model_type": "m",
        "M1": float(best["M1"]),
        "M2": float(best["M2"]),
        "k": float(k),
        "M_eff": float(m_eff),
        "SSE_naive": float(best["SSE_naive"]),
        "local_optima": local,
    }
    return fit, local


def _fit_C_mfree(
    starts: list[tuple[float, float]],
    m_starts: list[float],
    rs: np.ndarray,
    y_obs: np.ndarray,
    k: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local: list[dict[str, Any]] = []

    def objective(theta: np.ndarray) -> float:
        m1, m2, m_eff = float(theta[0]), float(theta[1]), float(theta[2])
        return _sse_m(m1=m1, m2=m2, k=k, m_eff=m_eff, rs=rs, y_obs=y_obs)

    for m1_start, m2_start in starts:
        for m_start in m_starts:
            result = minimize(
                objective,
                x0=np.array([m1_start, m2_start, m_start], dtype=float),
                bounds=[(1.0, 100.0), (1.0, 100.0), (2.0, 25.0)],
                method="L-BFGS-B",
            )
            m1_hat, m2_hat, m_eff_hat = [float(v) for v in result.x]
            local.append(
                {
                    "start_m1": m1_start,
                    "start_m2": m2_start,
                    "start_m_eff": m_start,
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "M1": m1_hat,
                    "M2": m2_hat,
                    "M_eff": m_eff_hat,
                    "SSE_naive": _sse_m(
                        m1=m1_hat,
                        m2=m2_hat,
                        k=k,
                        m_eff=m_eff_hat,
                        rs=rs,
                        y_obs=y_obs,
                    ),
                }
            )
    best = min(local, key=lambda row: row["SSE_naive"])
    fit = {
        "fit_name": "fit_C",
        "model_type": "m",
        "M1": float(best["M1"]),
        "M2": float(best["M2"]),
        "k": float(k),
        "M_eff": float(best["M_eff"]),
        "SSE_naive": float(best["SSE_naive"]),
        "local_optima": local,
    }
    return fit, local


def _attach_eval_metrics(fit: dict[str, Any], naive: pd.DataFrame, contract: pd.DataFrame) -> dict[str, Any]:
    naive_r = naive["retention"].to_numpy(dtype=float)
    contract_r = contract["retention"].to_numpy(dtype=float)
    pred_naive = np.array([_predict_pivot_by_fit(fit, r) for r in naive_r], dtype=float)
    pred_contract = np.array([_predict_pivot_by_fit(fit, r) for r in contract_r], dtype=float)
    obs_naive = naive["pivot_preservation"].to_numpy(dtype=float)
    obs_contract = contract["pivot_preservation"].to_numpy(dtype=float)
    fit["SSE_naive"] = float(np.sum((pred_naive - obs_naive) ** 2))
    fit["SSE_contract"] = float(np.sum((pred_contract - obs_contract) ** 2))
    fit["MAE_naive"] = float(np.mean(np.abs(pred_naive - obs_naive)))
    fit["MAE_contract"] = float(np.mean(np.abs(pred_contract - obs_contract)))
    fit["M1_less_than_M2"] = bool(fit["M1"] < fit["M2"])
    return fit


def _critical_retention(fit: dict[str, Any]) -> tuple[float, str]:
    def f(r: float) -> float:
        return _predict_substitution_by_fit(fit, r) - 0.5

    lo, hi = 1e-6, 1.0 - 1e-6
    f_lo, f_hi = f(lo), f(hi)
    if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi <= 0.0):
        return float(brentq(f, lo, hi, maxiter=200)), "root_brentq"
    alt = minimize_scalar(lambda x: abs(f(float(x))), bounds=(lo, hi), method="bounded")
    return float(alt.x), "nearest_abs_error"


def _build_saved_table(
    naive: pd.DataFrame,
    contract: pd.DataFrame,
    fit_A: dict[str, Any],
    fit_B: dict[str, Any],
    fit_C: dict[str, Any],
    best_fit: dict[str, Any],
) -> pd.DataFrame:
    all_rows = pd.concat([naive, contract], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for row in all_rows.itertuples(index=False):
        r = float(row.retention)
        pred_A = _predict_pivot_by_fit(fit_A, r)
        pred_B = _predict_pivot_by_fit(fit_B, r)
        pred_C = _predict_pivot_by_fit(fit_C, r)
        pred_best = _predict_pivot_by_fit(best_fit, r)
        pred_best_fix = predicted_fixed_pivot_feasibility(r=r, m1=best_fit["M1"], k=best_fit["k"])
        pred_best_raw = _predict_raw_by_fit(best_fit, r)
        rows.append(
            {
                "strategy": str(row.strategy),
                "retention": r,
                "obs_piv_pres": float(row.pivot_preservation),
                "pred_piv_pres_fit_A": pred_A,
                "pred_piv_pres_fit_B": pred_B,
                "pred_piv_pres_fit_C": pred_C,
                "best_fit_name": str(best_fit["fit_name"]),
                "pred_piv_pres_best": pred_best,
                "residual_best": float(row.pivot_preservation) - pred_best,
                "obs_fix_feas": float(row.fixed_pivot_feasibility),
                "pred_fix_feas_best": pred_best_fix,
                "obs_raw_val": float(row.raw_validity),
                "pred_raw_val_best": pred_best_raw,
                "mirage_gap_obs": float(row.raw_validity) - float(row.pivot_preservation),
                "mirage_gap_pred_best": pred_best_raw - pred_best,
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "retention"], ascending=[True, False]).reset_index(drop=True)


def _plot_model_comparison(
    naive: pd.DataFrame,
    contract: pd.DataFrame,
    fit_A: dict[str, Any],
    fit_B: dict[str, Any],
    fit_C: dict[str, Any],
    output_path: Path,
) -> None:
    x_curve = np.linspace(0.08, 0.52, 450)
    y_A = np.array([_predict_pivot_by_fit(fit_A, r) for r in x_curve], dtype=float)
    y_B = np.array([_predict_pivot_by_fit(fit_B, r) for r in x_curve], dtype=float)
    y_C = np.array([_predict_pivot_by_fit(fit_C, r) for r in x_curve], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x_curve, y_A, color="#1f77b4", linewidth=2.0, label="Fit A (2-candidate, unconstrained)")
    ax.plot(x_curve, y_B, color="#2ca02c", linewidth=2.0, label="Fit B (M-candidate, M=10)")
    ax.plot(x_curve, y_C, color="#9467bd", linewidth=2.0, label="Fit C (M-candidate, M free)")
    ax.scatter(naive["retention"], naive["pivot_preservation"], color="#111111", s=45, marker="o", label="Naive (obs)")
    ax.scatter(
        contract["retention"],
        contract["pivot_preservation"],
        color="#ff7f0e",
        s=55,
        marker="s",
        label="Contract-guarded (obs)",
    )
    ax.set_xlabel("Retention")
    ax.set_ylabel("Pivot Preservation")
    ax.set_title("Model Comparison: Pivot Preservation vs Retention")
    ax.set_xlim(0.08, 0.52)
    ax.set_ylim(0.20, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_diagnostics(
    naive: pd.DataFrame,
    contract: pd.DataFrame,
    best_fit: dict[str, Any],
    output_path: Path,
) -> None:
    x_curve = np.linspace(0.08, 0.52, 450)
    piv_curve = np.array([_predict_pivot_by_fit(best_fit, r) for r in x_curve], dtype=float)
    raw_curve = np.array([_predict_raw_by_fit(best_fit, r) for r in x_curve], dtype=float)
    fix_curve = np.array(
        [predicted_fixed_pivot_feasibility(r=r, m1=best_fit["M1"], k=best_fit["k"]) for r in x_curve],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_curve, raw_curve, color="#2ca02c", linewidth=2.0, label="Pred raw validity")
    ax.plot(x_curve, piv_curve, color="#1f77b4", linewidth=2.0, label="Pred pivot preservation")
    ax.plot(x_curve, fix_curve, color="#d62728", linewidth=2.0, label="Pred fixed-pivot feasibility")
    ax.fill_between(x_curve, piv_curve, raw_curve, color="#6baed6", alpha=0.20, label="Pred mirage gap")

    ax.scatter(naive["retention"], naive["raw_validity"], color="#2ca02c", marker="o", s=34, label="Naive raw")
    ax.scatter(contract["retention"], contract["raw_validity"], color="#2ca02c", marker="s", s=34, facecolors="none", label="Contract raw")
    ax.scatter(naive["retention"], naive["pivot_preservation"], color="#1f77b4", marker="o", s=34, label="Naive pivot")
    ax.scatter(contract["retention"], contract["pivot_preservation"], color="#1f77b4", marker="s", s=34, facecolors="none", label="Contract pivot")
    ax.scatter(naive["retention"], naive["fixed_pivot_feasibility"], color="#d62728", marker="o", s=34, label="Naive fixed")
    ax.scatter(contract["retention"], contract["fixed_pivot_feasibility"], color="#d62728", marker="s", s=34, facecolors="none", label="Contract fixed")

    ax.set_xlabel("Retention")
    ax.set_ylabel("Probability")
    ax.set_title("The Validity Mirage: Three Semantic Diagnostics")
    ax.set_xlim(0.08, 0.52)
    ax.set_ylim(0.20, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_residuals(
    naive: pd.DataFrame,
    contract: pd.DataFrame,
    best_fit: dict[str, Any],
    output_path: Path,
) -> None:
    naive_sub = naive.sort_values("retention")
    contract_sub = contract.sort_values("retention")

    res_naive = naive_sub["pivot_preservation"].to_numpy(dtype=float) - np.array(
        [_predict_pivot_by_fit(best_fit, r) for r in naive_sub["retention"].to_numpy(dtype=float)],
        dtype=float,
    )
    res_contract = contract_sub["pivot_preservation"].to_numpy(dtype=float) - np.array(
        [_predict_pivot_by_fit(best_fit, r) for r in contract_sub["retention"].to_numpy(dtype=float)],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(naive_sub["retention"], res_naive, marker="o", color="#1f77b4", linewidth=1.2, label="naive residual")
    ax.plot(contract_sub["retention"], res_contract, marker="s", color="#ff7f0e", linewidth=1.2, label="contract residual")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Retention")
    ax.set_ylabel("Residual (obs - pred), pivot preservation")
    ax.set_title("Model Residuals by Compression Strategy")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and compare rate-distortion pivot-preservation models.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory for experiment artifacts (CSV/JSON/PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    naive, contract = _build_empirical_data()
    rs = naive["retention"].to_numpy(dtype=float)
    y_naive = naive["pivot_preservation"].to_numpy(dtype=float)
    k = 3.0

    print("=== SANITY CHECKS (pre-fit) ===")
    m1_sanity, m2_sanity = 5.0, 15.0
    p_r1 = predicted_pivot_preservation_two(r=1.0, m1=m1_sanity, m2=m2_sanity, k=k)
    p_r0 = predicted_pivot_preservation_two(r=1e-6, m1=m1_sanity, m2=m2_sanity, k=k)
    raw_r03 = predicted_raw_validity_two(r=0.30, m1=m1_sanity, m2=m2_sanity, k=k)
    piv_r03 = predicted_pivot_preservation_two(r=0.30, m1=m1_sanity, m2=m2_sanity, k=k)
    print(f"At r=1.0: pivot preservation = {p_r1:.6f} (should be 1.0)")
    print(f"At r->0 with M1<M2: pivot preservation = {p_r0:.6f} (should drop below 0.5)")
    print(
        f"At r=0.30: raw validity={raw_r03:.6f}, pivot preservation={piv_r03:.6f} "
        "(raw validity should exceed pivot preservation)."
    )

    base_starts = [(20, 10), (30, 15), (50, 20), (10, 5), (40, 8)]
    extra_starts = [(5, 15), (3, 20), (4, 12)]
    starts_extended = base_starts + extra_starts

    fit_original, _ = _fit_original_constrained(starts=base_starts, rs=rs, y_obs=y_naive, k=k)
    fit_A, _ = _fit_A_unconstrained(starts=starts_extended, rs=rs, y_obs=y_naive, k=k)
    fit_B, _ = _fit_B_mfixed(starts=starts_extended, rs=rs, y_obs=y_naive, k=k, m_eff=10.0)
    fit_C, _ = _fit_C_mfree(starts=starts_extended, m_starts=[10.0, 6.0, 15.0], rs=rs, y_obs=y_naive, k=k)

    fit_original = _attach_eval_metrics(fit_original, naive=naive, contract=contract)
    fit_A = _attach_eval_metrics(fit_A, naive=naive, contract=contract)
    fit_B = _attach_eval_metrics(fit_B, naive=naive, contract=contract)
    fit_C = _attach_eval_metrics(fit_C, naive=naive, contract=contract)

    if fit_A["M1_less_than_M2"]:
        print("\n[THEORY LINK] Fit A found M1 < M2 (dominant pivot has fewer predecessors).")
        print("This supports the absorbing-state asymmetry connection to compression vulnerability.")

    summary_rows = pd.DataFrame(
        [
            {
                "Fit": "fit_original",
                "M1": fit_original["M1"],
                "M2": fit_original["M2"],
                "k": fit_original["k"],
                "M_eff": fit_original["M_eff"],
                "SSE_naive": fit_original["SSE_naive"],
                "SSE_contract": fit_original["SSE_contract"],
                "MAE_naive": fit_original["MAE_naive"],
                "MAE_contract": fit_original["MAE_contract"],
                "M1 < M2?": fit_original["M1_less_than_M2"],
            },
            {
                "Fit": "fit_A",
                "M1": fit_A["M1"],
                "M2": fit_A["M2"],
                "k": fit_A["k"],
                "M_eff": fit_A["M_eff"],
                "SSE_naive": fit_A["SSE_naive"],
                "SSE_contract": fit_A["SSE_contract"],
                "MAE_naive": fit_A["MAE_naive"],
                "MAE_contract": fit_A["MAE_contract"],
                "M1 < M2?": fit_A["M1_less_than_M2"],
            },
            {
                "Fit": "fit_B",
                "M1": fit_B["M1"],
                "M2": fit_B["M2"],
                "k": fit_B["k"],
                "M_eff": fit_B["M_eff"],
                "SSE_naive": fit_B["SSE_naive"],
                "SSE_contract": fit_B["SSE_contract"],
                "MAE_naive": fit_B["MAE_naive"],
                "MAE_contract": fit_B["MAE_contract"],
                "M1 < M2?": fit_B["M1_less_than_M2"],
            },
            {
                "Fit": "fit_C",
                "M1": fit_C["M1"],
                "M2": fit_C["M2"],
                "k": fit_C["k"],
                "M_eff": fit_C["M_eff"],
                "SSE_naive": fit_C["SSE_naive"],
                "SSE_contract": fit_C["SSE_contract"],
                "MAE_naive": fit_C["MAE_naive"],
                "MAE_contract": fit_C["MAE_contract"],
                "M1 < M2?": fit_C["M1_less_than_M2"],
            },
        ]
    )

    print("\n=== FIT SUMMARY TABLE ===")
    print(
        summary_rows[
            ["Fit", "M1", "M2", "k", "M_eff", "SSE_naive", "SSE_contract", "MAE_naive", "MAE_contract", "M1 < M2?"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    best_candidates = [fit_A, fit_B, fit_C]
    best_fit = min(best_candidates, key=lambda row: row["SSE_naive"])

    retentions = sorted(naive["retention"].unique().tolist(), reverse=True)
    obs_naive_by_r = dict(zip(naive["retention"], naive["pivot_preservation"], strict=True))
    obs_contract_by_r = dict(zip(contract["retention"], contract["pivot_preservation"], strict=True))
    compare_rows = []
    for r in retentions:
        compare_rows.append(
            {
                "retention": float(r),
                "obs_naive": float(obs_naive_by_r[r]),
                "pred_A": _predict_pivot_by_fit(fit_A, r),
                "pred_B": _predict_pivot_by_fit(fit_B, r),
                "pred_C": _predict_pivot_by_fit(fit_C, r),
                "obs_contract": float(obs_contract_by_r[r]),
            }
        )
    compare_df = pd.DataFrame(compare_rows)

    print("\n=== RETENTION-BY-RETENTION COMPARISON ===")
    print(compare_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    full_table = _build_saved_table(
        naive=naive,
        contract=contract,
        fit_A=fit_A,
        fit_B=fit_B,
        fit_C=fit_C,
        best_fit=best_fit,
    )
    csv_path = output_dir / "experiment_02_rate_distortion_fit.csv"
    full_table.to_csv(csv_path, index=False)

    plot_a = output_dir / "experiment_02_pivot_preservation_fit.png"
    plot_b = output_dir / "experiment_02_validity_mirage_diagnostics.png"
    plot_c = output_dir / "experiment_02_residuals_by_strategy.png"
    _plot_model_comparison(naive=naive, contract=contract, fit_A=fit_A, fit_B=fit_B, fit_C=fit_C, output_path=plot_a)
    _plot_best_diagnostics(naive=naive, contract=contract, best_fit=best_fit, output_path=plot_b)
    _plot_best_residuals(naive=naive, contract=contract, best_fit=best_fit, output_path=plot_c)

    crit_r, crit_method = _critical_retention(best_fit)
    h = 1e-4
    d_left = _predict_substitution_by_fit(best_fit, max(1e-6, crit_r - h))
    d_right = _predict_substitution_by_fit(best_fit, min(1.0 - 1e-6, crit_r + h))
    d_dr = float((d_right - d_left) / (2.0 * h))

    mirage_gap_r050 = float(_predict_raw_by_fit(best_fit, 0.50) - _predict_pivot_by_fit(best_fit, 0.50))

    naive_sub = naive.sort_values("retention")
    pred_best_naive = np.array([_predict_pivot_by_fit(best_fit, r) for r in naive_sub["retention"]], dtype=float)
    pred_best_raw_naive = np.array([_predict_raw_by_fit(best_fit, r) for r in naive_sub["retention"]], dtype=float)
    obs_piv_naive = naive_sub["pivot_preservation"].to_numpy(dtype=float)
    obs_raw_naive = naive_sub["raw_validity"].to_numpy(dtype=float)

    obs_feas_loss = float(np.mean(1.0 - obs_raw_naive))
    obs_scoring_loss = float(np.mean(obs_raw_naive - obs_piv_naive))
    pred_feas_loss = float(np.mean(1.0 - pred_best_raw_naive))
    pred_scoring_loss = float(np.mean(pred_best_raw_naive - pred_best_naive))
    scoring_gap_explained = (
        float(pred_scoring_loss / obs_scoring_loss) if obs_scoring_loss > 0 else float("nan")
    )
    scoring_gap_unexplained = float(obs_scoring_loss - pred_scoring_loss)

    summary = {
        "fit_original": {
            "M1": fit_original["M1"],
            "M2": fit_original["M2"],
            "k": fit_original["k"],
            "M_eff": fit_original["M_eff"],
            "SSE_naive": fit_original["SSE_naive"],
            "SSE_contract": fit_original["SSE_contract"],
            "MAE_naive": fit_original["MAE_naive"],
            "MAE_contract": fit_original["MAE_contract"],
            "M1_less_than_M2": fit_original["M1_less_than_M2"],
            "local_optima": fit_original["local_optima"],
        },
        "fit_A": {
            "M1": fit_A["M1"],
            "M2": fit_A["M2"],
            "k": fit_A["k"],
            "M_eff": fit_A["M_eff"],
            "SSE_naive": fit_A["SSE_naive"],
            "SSE_contract": fit_A["SSE_contract"],
            "MAE_naive": fit_A["MAE_naive"],
            "MAE_contract": fit_A["MAE_contract"],
            "M1_less_than_M2": fit_A["M1_less_than_M2"],
            "local_optima": fit_A["local_optima"],
        },
        "fit_B": {
            "M1": fit_B["M1"],
            "M2": fit_B["M2"],
            "k": fit_B["k"],
            "M_eff": fit_B["M_eff"],
            "SSE_naive": fit_B["SSE_naive"],
            "SSE_contract": fit_B["SSE_contract"],
            "MAE_naive": fit_B["MAE_naive"],
            "MAE_contract": fit_B["MAE_contract"],
            "M1_less_than_M2": fit_B["M1_less_than_M2"],
            "local_optima": fit_B["local_optima"],
        },
        "fit_C": {
            "M1": fit_C["M1"],
            "M2": fit_C["M2"],
            "k": fit_C["k"],
            "M_eff": fit_C["M_eff"],
            "SSE_naive": fit_C["SSE_naive"],
            "SSE_contract": fit_C["SSE_contract"],
            "MAE_naive": fit_C["MAE_naive"],
            "MAE_contract": fit_C["MAE_contract"],
            "M1_less_than_M2": fit_C["M1_less_than_M2"],
            "local_optima": fit_C["local_optima"],
        },
        "best_fit": best_fit["fit_name"],
        "theory_validation": bool(best_fit["M1_less_than_M2"]),
        "critical_retention": crit_r,
        "critical_retention_method": crit_method,
        "phase_transition_dD_dr": d_dr,
        "mirage_gap_at_r050": mirage_gap_r050,
        "decomposition": {
            "obs_feasibility_loss_mean": obs_feas_loss,
            "obs_scoring_rerank_loss_mean": obs_scoring_loss,
            "pred_feasibility_loss_mean": pred_feas_loss,
            "pred_scoring_rerank_loss_mean": pred_scoring_loss,
            "scoring_rerank_fraction_explained": scoring_gap_explained,
            "scoring_rerank_unexplained_mean": scoring_gap_unexplained,
        },
    }
    summary_path = output_dir / "experiment_02_rate_distortion_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    asymmetry_gain = float(fit_original["SSE_naive"] - fit_A["SSE_naive"])
    count_gain = float(fit_A["SSE_naive"] - min(fit_B["SSE_naive"], fit_C["SSE_naive"]))
    if abs(asymmetry_gain) >= abs(count_gain):
        dominant_fix = "asymmetry correction"
    else:
        dominant_fix = "candidate-count modeling"

    interpretation = (
        f"Best fit is {best_fit['fit_name']} with M1={best_fit['M1']:.3f}, M2={best_fit['M2']:.3f}, "
        f"M_eff={best_fit['M_eff']:.3f}; M1<M2 is {best_fit['M1_less_than_M2']}, "
        "which tests the absorbing-state hypothesis that dominant pivots are predecessor-poor. "
        f"The larger improvement came from {dominant_fix} (delta SSE asymmetry={asymmetry_gain:.6f}, "
        f"delta SSE candidate-count={count_gain:.6f}). Residual error remains "
        f"(MAE naive={best_fit['MAE_naive']:.4f}, contract={best_fit['MAE_contract']:.4f}), suggesting "
        "a mechanism beyond pure feasibility geometry, consistent with scoring-based substitution effects. "
        f"On naive data, observed loss decomposes into feasibility={obs_feas_loss:.4f} and scoring-rerank="
        f"{obs_scoring_loss:.4f}; the best-fit model explains about {100.0 * scoring_gap_explained:.1f}% "
        "of scoring-rerank loss."
    )
    print("\n=== INTERPRETATION ===")
    print(interpretation)
    print(f"\nSaved table: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plots: {plot_a}, {plot_b}, {plot_c}")


if __name__ == "__main__":
    main()
