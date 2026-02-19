from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.generators import bursty_generator
from src.pivot_transitions import (
    depth_predictions,
    estimate_margin_stability,
    estimate_single_step_transition,
    extract_locked_candidates,
    margin_features,
    run_chained_compression_survival_curve,
)
from src.reporting import ensure_result_dirs, print_header, print_table, save_csv, save_figure, set_plot_style


TEST_NAME = "Test 18: Transition-Vector Depth Validation"
CLAIM = "Claim: dominant-pivot survival decays with repeated compression; compare p11^d and retention^d against empirical chained survival."


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float_list(name: str, default: Sequence[float]) -> List[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(float(token))
        except ValueError:
            continue
    return vals or list(default)


def _convergence_precheck(
    *,
    k: int,
    M: int,
    epsilon: float,
    retention: float,
    n: int,
    n_focal: int,
    pilot_sequence_seeds: Sequence[int],
    candidate_compression_counts: Sequence[int],
) -> tuple[int, pd.DataFrame]:
    rows: List[Dict[str, float]] = []

    for comp_count in candidate_compression_counts:
        p11_values: List[float] = []
        for seq_seed in pilot_sequence_seeds:
            events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=int(seq_seed))
            locked = extract_locked_candidates(events, M=M)
            if not locked:
                continue

            seed_start = 10_000_000 + int(seq_seed) * 10_000 + int(comp_count) * 13
            compression_seeds = range(seed_start, seed_start + int(comp_count))
            est = estimate_single_step_transition(
                events,
                k=k,
                retention=retention,
                locked_candidates=locked,
                compression_seeds=compression_seeds,
                method="naive",
            )
            if not est["baseline_valid"]:
                continue
            p11_values.append(float(est["p11"]))

        rows.append(
            {
                "compression_seeds": int(comp_count),
                "pilot_n_sequences": float(len(p11_values)),
                "mean_p11": float(np.mean(p11_values) if p11_values else 0.0),
                "std_p11": float(np.std(p11_values) if p11_values else 0.0),
            }
        )

    pre = pd.DataFrame(rows).sort_values("compression_seeds").reset_index(drop=True)
    pre["delta_mean_from_prev"] = np.nan
    pre["delta_std_from_prev"] = np.nan

    selected = int(pre["compression_seeds"].iloc[-1]) if len(pre) else int(candidate_compression_counts[-1])
    for idx in range(1, len(pre)):
        dm = abs(float(pre.loc[idx, "mean_p11"] - pre.loc[idx - 1, "mean_p11"]))
        ds = abs(float(pre.loc[idx, "std_p11"] - pre.loc[idx - 1, "std_p11"]))
        pre.loc[idx, "delta_mean_from_prev"] = dm
        pre.loc[idx, "delta_std_from_prev"] = ds
        if dm < 0.01 and ds < 0.01:
            selected = int(pre.loc[idx, "compression_seeds"])
            break

    return selected, pre


def run(results_dir: Path | None = None) -> Dict[str, float]:
    root = Path(__file__).resolve().parents[1]
    if results_dir is None:
        results_dir = root / "results"
    raw_dir, fig_dir = ensure_result_dirs(results_dir)

    print_header(TEST_NAME, CLAIM)

    n = _env_int("ECT_TEST18_N", 200)
    n_focal = _env_int("ECT_TEST18_N_FOCAL", 80)
    k = _env_int("ECT_TEST18_K", 3)
    M = _env_int("ECT_TEST18_M", 10)
    sequence_seed_count = _env_int("ECT_TEST18_SEQUENCE_SEEDS", 200)
    chain_trials = _env_int("ECT_TEST18_CHAIN_TRIALS", 200)
    epsilons = _env_float_list("ECT_TEST18_EPSILONS", [0.3, 0.5, 0.7])
    retentions = _env_float_list("ECT_TEST18_RETENTIONS", [0.9, 0.7, 0.5, 0.3, 0.1])
    depths = [1, 2, 4, 8]

    selected_comp_seeds, precheck_df = _convergence_precheck(
        k=k,
        M=M,
        epsilon=0.5,
        retention=0.5,
        n=n,
        n_focal=n_focal,
        pilot_sequence_seeds=range(min(40, sequence_seed_count)),
        candidate_compression_counts=[100, 200, 500],
    )
    save_csv(precheck_df, raw_dir, "test_18_transition_convergence_check.csv")

    rows: List[Dict[str, float]] = []
    for eps_idx, epsilon in enumerate(epsilons):
        for ret_idx, retention in enumerate(retentions):
            for seq_seed in range(sequence_seed_count):
                events = bursty_generator(n=n, n_focal=n_focal, epsilon=float(epsilon), seed=seq_seed)
                locked = extract_locked_candidates(events, M=M)

                seed_start = (
                    1_000_000_000
                    + eps_idx * 10_000_000
                    + ret_idx * 1_000_000
                    + seq_seed * 10_000
                )
                compression_seeds = range(seed_start, seed_start + selected_comp_seeds)
                est = estimate_single_step_transition(
                    events,
                    k=k,
                    retention=float(retention),
                    locked_candidates=locked,
                    compression_seeds=compression_seeds,
                    method="naive",
                )
                margin = margin_features(events, k=k)
                chain_curve = run_chained_compression_survival_curve(
                    events,
                    k=k,
                    retention=float(retention),
                    locked_candidates=locked,
                    depths=depths,
                    chain_trials=chain_trials,
                    base_seed=seed_start + 777_777,
                    method="naive",
                    baseline_pivot_eid=(
                        int(est["baseline_pivot_eid"]) if est["baseline_pivot_eid"] is not None else None
                    ),
                )
                preds = depth_predictions(
                    p11=float(est["p11"]),
                    retention=float(retention),
                    depths=depths,
                )

                row: Dict[str, float] = {
                    "epsilon": float(epsilon),
                    "retention": float(retention),
                    "compression_rate": float(1.0 - retention),
                    "sequence_seed": float(seq_seed),
                    "k": float(k),
                    "M": float(M),
                    "selected_compression_seeds": float(selected_comp_seeds),
                    "chain_trials": float(chain_trials),
                    "locked_candidate_count": float(est["locked_candidate_count"]),
                    "baseline_valid": float(est["baseline_valid"]),
                    "baseline_pivot_eid": (
                        float(est["baseline_pivot_eid"])
                        if est["baseline_pivot_eid"] is not None
                        else np.nan
                    ),
                    "baseline_weight": float(est["baseline_weight"]),
                    "baseline_prefix": (
                        float(est["baseline_prefix"]) if est["baseline_prefix"] is not None else np.nan
                    ),
                    "p11": float(est["p11"]),
                    "p_other": float(est["p_other"]),
                    "p_loss": float(est["p_loss"]),
                    "mean_achieved_retention": float(est["mean_achieved_retention"]),
                    "margin": float(margin["margin"]),
                    "margin_finite": float(margin["margin_finite"])
                    if np.isfinite(margin["margin_finite"])
                    else np.nan,
                }

                for d in depths:
                    row[f"pred_p11_pow_d{d}"] = float(preds[d]["pred_p11_pow_d"])
                    row[f"pred_retention_pow_d{d}"] = float(preds[d]["pred_retention_pow_d"])
                    row[f"empirical_chain_survival_d{d}"] = float(chain_curve[d])
                rows.append(row)

    raw_df = pd.DataFrame(rows)

    finite_margin = raw_df["margin_finite"].replace([np.inf, -np.inf], np.nan).dropna()
    cap = float(finite_margin.max() * 1.5) if len(finite_margin) else 1.0
    raw_df["margin_finite"] = raw_df["margin_finite"].fillna(cap)

    for d in depths:
        raw_df[f"abs_err_p11_d{d}"] = np.abs(raw_df[f"empirical_chain_survival_d{d}"] - raw_df[f"pred_p11_pow_d{d}"])
        raw_df[f"abs_err_retention_d{d}"] = np.abs(
            raw_df[f"empirical_chain_survival_d{d}"] - raw_df[f"pred_retention_pow_d{d}"]
        )

    save_csv(raw_df, raw_dir, "test_18_transition_vector_raw.csv")

    agg_cols = {
        "p11": "mean",
        "p_other": "mean",
        "p_loss": "mean",
        "mean_achieved_retention": "mean",
        "margin_finite": "mean",
        "baseline_valid": "mean",
        "selected_compression_seeds": "mean",
        "chain_trials": "mean",
    }
    for d in depths:
        agg_cols[f"pred_p11_pow_d{d}"] = "mean"
        agg_cols[f"pred_retention_pow_d{d}"] = "mean"
        agg_cols[f"empirical_chain_survival_d{d}"] = "mean"
        agg_cols[f"abs_err_p11_d{d}"] = "mean"
        agg_cols[f"abs_err_retention_d{d}"] = "mean"

    summary = (
        raw_df.groupby(["epsilon", "retention", "compression_rate"], as_index=False)
        .agg(agg_cols)
        .sort_values(["epsilon", "retention"], ascending=[True, False])
    )
    save_csv(summary, raw_dir, "test_18_transition_vector_summary.csv")
    print_table(summary, max_rows=30)

    overall_corr = estimate_margin_stability(raw_df[raw_df["baseline_valid"] > 0.5])
    corr_rows: List[Dict[str, float]] = [dict(group="overall", epsilon=np.nan, **overall_corr)]
    for epsilon in sorted(raw_df["epsilon"].unique().tolist()):
        sub = raw_df[(raw_df["epsilon"] == epsilon) & (raw_df["baseline_valid"] > 0.5)]
        corr = estimate_margin_stability(sub)
        corr_rows.append(dict(group=f"epsilon_{epsilon:.2f}", epsilon=float(epsilon), **corr))
    corr_df = pd.DataFrame(corr_rows)
    save_csv(corr_df, raw_dir, "test_18_margin_p11_correlation.csv")
    print("\nMargin vs p11 correlations:")
    print(corr_df.to_string(index=False))

    # Margin-decile profile for plotting.
    try:
        raw_df["margin_decile"] = pd.qcut(
            raw_df["margin_finite"], q=10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop"
        ).astype(str)
    except ValueError:
        raw_df["margin_decile"] = "D_all"
    decile_df = (
        raw_df.groupby("margin_decile", as_index=False)
        .agg(mean_p11=("p11", "mean"), count=("p11", "count"))
        .sort_values("margin_decile")
    )
    save_csv(decile_df, raw_dir, "test_18_margin_deciles.csv")

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # Panel A: single-step p11 vs compression rate by epsilon.
    for epsilon in sorted(summary["epsilon"].unique().tolist()):
        sub = summary[summary["epsilon"] == epsilon].sort_values("compression_rate")
        axes[0].plot(
            sub["compression_rate"],
            sub["p11"],
            marker="o",
            label=f"epsilon={epsilon:.1f}",
        )
    axes[0].set_xlabel("compression rate (1 - retention)")
    axes[0].set_ylabel("mean p11")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Single-Step Self-Transition")
    axes[0].legend()

    # Panel B: depth curves (empirical vs predictions) at selected retentions.
    summary_by_ret = summary.groupby("retention", as_index=False).mean(numeric_only=True)
    selected_retentions = [r for r in [0.7, 0.5, 0.3] if r in set(summary_by_ret["retention"])]
    cmap = plt.get_cmap("viridis", max(1, len(selected_retentions)))
    for idx, retention in enumerate(selected_retentions):
        row = summary_by_ret[summary_by_ret["retention"] == retention].iloc[0]
        emp = [row[f"empirical_chain_survival_d{d}"] for d in depths]
        p11_pred = [row[f"pred_p11_pow_d{d}"] for d in depths]
        ret_pred = [row[f"pred_retention_pow_d{d}"] for d in depths]
        color = cmap(idx)
        axes[1].plot(depths, emp, marker="o", color=color, label=f"empirical r={retention:.1f}")
        axes[1].plot(depths, p11_pred, marker="s", linestyle="--", color=color, label=f"p11^d r={retention:.1f}")
        axes[1].plot(depths, ret_pred, marker="x", linestyle=":", color=color, label=f"ret^d r={retention:.1f}")
    axes[1].set_xlabel("depth d")
    axes[1].set_ylabel("survival probability")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Depth Survival: Empirical vs Predictions")
    axes[1].legend(fontsize=8, ncol=1)

    # Panel C: margin decile vs p11.
    axes[2].plot(decile_df["margin_decile"], decile_df["mean_p11"], marker="o")
    axes[2].set_xlabel("margin decile")
    axes[2].set_ylabel("mean p11")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Margin Predicts Transition Stability")
    axes[2].tick_params(axis="x", rotation=45)

    save_figure(fig, fig_dir, "test_18_transition_vector_depth.png")

    # Verdict checks
    monotonic_ok = True
    for epsilon in sorted(summary["epsilon"].unique().tolist()):
        vals = summary[summary["epsilon"] == epsilon].sort_values("retention", ascending=False)["p11"].to_numpy()
        if len(vals) >= 2 and np.any(np.diff(vals) > 0.03):
            monotonic_ok = False
            break

    low_ret = summary[summary["retention"] <= 0.5]
    if len(low_ret) > 0:
        depth_drop = float(low_ret["empirical_chain_survival_d1"].mean() - low_ret["empirical_chain_survival_d8"].mean())
    else:
        depth_drop = 0.0
    depth_ok = depth_drop > 0.05

    better_depths = 0
    for d in depths:
        err_p11 = float(summary[f"abs_err_p11_d{d}"].mean())
        err_ret = float(summary[f"abs_err_retention_d{d}"].mean())
        if err_p11 <= err_ret:
            better_depths += 1
    baseline_valid_rate = float(raw_df["baseline_valid"].mean())
    verdict = "PASS" if monotonic_ok and depth_ok and baseline_valid_rate >= 0.95 else "FAIL"

    print(
        "\nVerdict: "
        f"{verdict} | monotonic_ok={monotonic_ok}, depth_drop={depth_drop:.3f}, "
        f"prediction_beats_retention_depths={better_depths}/{len(depths)}, "
        f"margin_corr={overall_corr['pearson_corr']:.4f}, "
        f"baseline_valid_rate={baseline_valid_rate:.3f}"
    )

    return {
        "name": TEST_NAME,
        "verdict": verdict,
        "selected_compression_seeds": float(selected_comp_seeds),
        "mean_depth_drop_low_ret": float(depth_drop),
        "prediction_better_depths": float(better_depths),
        "margin_corr": float(overall_corr["pearson_corr"]),
    }


if __name__ == "__main__":
    run()
