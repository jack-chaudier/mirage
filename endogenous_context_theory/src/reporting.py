"""Reporting helpers for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_result_dirs(base: Path) -> Tuple[Path, Path]:
    raw_dir = base / "raw"
    fig_dir = base / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, fig_dir


def save_csv(df: pd.DataFrame, raw_dir: Path, filename: str) -> Path:
    path = raw_dir / filename
    df.to_csv(path, index=False)
    return path


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, fig_dir: Path, filename: str) -> Path:
    path = fig_dir / filename
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def print_header(name: str, claim: str) -> None:
    print("\n" + "=" * 96)
    print(name)
    print(claim)
    print("=" * 96)


def print_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    if len(df) > max_rows:
        preview = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
        print(preview.to_string(index=False))
        print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))


def pass_fail(violations: int, expected_zero: bool = True) -> str:
    if expected_zero:
        return "PASS" if violations == 0 else "FAIL"
    return "PASS" if violations > 0 else "FAIL"


def power_law_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    lx = np.log(x)
    ly = np.log(y)

    slope, intercept = np.polyfit(lx, ly, 1)
    y_hat = slope * lx + intercept
    resid = ly - y_hat

    n = len(lx)
    if n > 2:
        s_err = np.sqrt(np.sum(resid**2) / (n - 2))
        x_var = np.sum((lx - np.mean(lx)) ** 2)
        slope_se = s_err / np.sqrt(x_var) if x_var > 0 else np.nan
    else:
        slope_se = np.nan

    ci_low = slope - 1.96 * slope_se if np.isfinite(slope_se) else np.nan
    ci_high = slope + 1.96 * slope_se if np.isfinite(slope_se) else np.nan

    return {
        "a": float(np.exp(intercept)),
        "b": float(slope),
        "b_ci_low": float(ci_low),
        "b_ci_high": float(ci_high),
    }
