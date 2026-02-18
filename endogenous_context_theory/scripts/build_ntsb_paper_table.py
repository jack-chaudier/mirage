#!/usr/bin/env python3
"""Build paper-ready retention-matched figure table from NTSB benchmark outputs."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-example-csv",
        default="endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/mirage_results_per_example.csv",
    )
    p.add_argument(
        "--summary-csv",
        default="endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/mirage_results_summary.csv",
    )
    p.add_argument(
        "--out-csv",
        default="endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.csv",
    )
    p.add_argument(
        "--out-md",
        default="endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.md",
    )
    p.add_argument(
        "--out-json",
        default="endogenous_context_theory/results/ntsb/xai_grok_4_1_fast_non_reasoning/paper_figure_table_retention_matched.json",
    )
    return p.parse_args()


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + (z * z / n)
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z * sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / denom
    return (center - margin, center + margin)


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{100.0 * x:.1f}%"


def fmt_pct2(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{100.0 * x:.2f}%"


def _nan_to_none(value: object) -> object:
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def build_row(
    panel: str,
    pair_id: str,
    label: str,
    method: str,
    budget: float,
    per: pd.DataFrame,
    summary: pd.DataFrame,
) -> Dict[str, object]:
    g = per[(per["method"] == method) & (per["budget"] == budget)]
    if g.empty:
        raise ValueError(f"No rows for method={method}, budget={budget}")

    s = summary[(summary["method"] == method) & (summary["budget"] == budget)]
    if s.empty:
        raise ValueError(f"No summary row for method={method}, budget={budget}")
    srow = s.iloc[0]

    deg = g[g["oracle_degraded"] == True]  # noqa: E712
    strong = g[g["oracle_degraded"] == False]  # noqa: E712
    wrong_deg = deg[deg["model_error"] == True]  # noqa: E712

    silent_k = int(deg["silent_mirage"].sum())
    silent_n = int(len(deg))
    silent_rate = float(silent_k / silent_n) if silent_n else float("nan")
    sil_lo, sil_hi = wilson_interval(silent_k, silent_n)

    overall_wrong_k = int(wrong_deg["flagged_degraded"].sum())
    overall_wrong_n = int(len(wrong_deg))
    if overall_wrong_n:
        fgw = overall_wrong_k / overall_wrong_n
        fgw_lo, fgw_hi = wilson_interval(overall_wrong_k, overall_wrong_n)
    else:
        fgw = float("nan")
        fgw_lo, fgw_hi = (float("nan"), float("nan"))

    return {
        "panel": panel,
        "pair_id": pair_id,
        "label": label,
        "method": method,
        "budget": float(budget),
        "n_total": int(len(g)),
        "n_degraded": silent_n,
        "n_strong": int(len(strong)),
        "mean_achieved_retention": float(srow["mean_achieved_retention"]),
        "pivot_accuracy": float(srow["pivot_accuracy"]),
        "pivot_preservation": float(srow["pivot_preservation"]),
        "info_shift_rate": float(srow["info_shift_rate"]),
        "silent_mirage_count": silent_k,
        "silent_mirage_n": silent_n,
        "silent_mirage_rate": silent_rate,
        "silent_mirage_wilson_low": sil_lo,
        "silent_mirage_wilson_high": sil_hi,
        "flag_degraded_rate": float(srow["flag_degraded_rate"]),
        "flag_given_wrong": fgw,
        "flag_given_wrong_wilson_low": fgw_lo,
        "flag_given_wrong_wilson_high": fgw_hi,
    }


def main() -> None:
    args = parse_args()
    per_path = Path(args.per_example_csv).resolve()
    sum_path = Path(args.summary_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    out_json = Path(args.out_json).resolve()

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    per = pd.read_csv(per_path)
    summary = pd.read_csv(sum_path)

    rows: List[Dict[str, object]] = []

    # Primary: exact retention match at budget 0.7
    rows.append(build_row("primary_exact_retention_match", "A", "naive@0.7", "naive", 0.7, per, summary))
    rows.append(build_row("primary_exact_retention_match", "A", "contract@0.7", "contract", 0.7, per, summary))

    # Secondary: near retention match across budgets
    rows.append(build_row("secondary_cross_budget_near_match", "B", "naive@0.5", "naive", 0.5, per, summary))
    rows.append(build_row("secondary_cross_budget_near_match", "B", "contract@0.3", "contract", 0.3, per, summary))

    table = pd.DataFrame(rows)

    # pair-level retention gap helper
    pair_gaps = {}
    for pid, grp in table.groupby("pair_id"):
        if len(grp) == 2:
            gap = abs(float(grp.iloc[0]["mean_achieved_retention"]) - float(grp.iloc[1]["mean_achieved_retention"]))
            pair_gaps[pid] = gap
        else:
            pair_gaps[pid] = float("nan")
    table["pair_retention_gap"] = table["pair_id"].map(pair_gaps)

    # overall naive degraded silent mirage (headline)
    naive = per[per["method"] == "naive"]
    naive_deg = naive[naive["oracle_degraded"] == True]  # noqa: E712
    k = int(naive_deg["silent_mirage"].sum())
    n = int(len(naive_deg))
    lo, hi = wilson_interval(k, n)

    headline = {
        "naive_overall_degraded_silent_k": k,
        "naive_overall_degraded_silent_n": n,
        "naive_overall_degraded_silent_rate": (k / n) if n else float("nan"),
        "naive_overall_degraded_silent_wilson_low": lo,
        "naive_overall_degraded_silent_wilson_high": hi,
    }

    table.to_csv(out_csv, index=False)
    json_headline = {k: _nan_to_none(v) for k, v in headline.items()}
    json_rows = [{k: _nan_to_none(v) for k, v in row.items()} for row in rows]
    out_json.write_text(
        json.dumps({"headline": json_headline, "rows": json_rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Build markdown table
    md_lines = []
    md_lines.append("# Paper Figure Table: Retention-Matched Mirage Comparison")
    md_lines.append("")
    md_lines.append("## Headline")
    md_lines.append(
        f"Naive degraded silent mirage overall: {k}/{n} ({fmt_pct2(k/n if n else float('nan'))}), "
        f"Wilson 95% CI [{fmt_pct2(lo)}, {fmt_pct2(hi)}]."
    )
    md_lines.append("")
    md_lines.append("## Panel A (Primary): Exact Retention Match")
    md_lines.append("Retention is exactly matched between methods at budget 0.7.")
    md_lines.append("")
    md_lines.append("| Method | Budget | Mean Retention | Info Shift | Pivot Preservation | Silent Mirage (degraded) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    a = table[table["pair_id"] == "A"]
    for _, r in a.iterrows():
        sm = f"{int(r['silent_mirage_count'])}/{int(r['silent_mirage_n'])} ({fmt_pct(r['silent_mirage_rate'])})"
        md_lines.append(
            f"| {r['method']} | {r['budget']:.1f} | {r['mean_achieved_retention']:.6f} | {fmt_pct(r['info_shift_rate'])} | {fmt_pct(r['pivot_preservation'])} | {sm} |"
        )
    md_lines.append("")
    md_lines.append("## Panel B (Secondary): Cross-Budget Near Match")
    md_lines.append("Secondary comparison with close retention values (naive 0.5 vs contract 0.3).")
    md_lines.append("")
    md_lines.append("| Method | Budget | Mean Retention | Info Shift | Pivot Preservation | Silent Mirage (degraded) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    b = table[table["pair_id"] == "B"]
    for _, r in b.iterrows():
        sm = f"{int(r['silent_mirage_count'])}/{int(r['silent_mirage_n'])} ({fmt_pct(r['silent_mirage_rate'])})"
        md_lines.append(
            f"| {r['method']} | {r['budget']:.1f} | {r['mean_achieved_retention']:.6f} | {fmt_pct(r['info_shift_rate'])} | {fmt_pct(r['pivot_preservation'])} | {sm} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
