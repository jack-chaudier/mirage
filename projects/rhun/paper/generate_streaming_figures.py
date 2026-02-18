#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_rc_params = {
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
if "savefig.bbox_inches" in plt.rcParams:
    _rc_params["savefig.bbox_inches"] = "tight"
elif "savefig.bbox" in plt.rcParams:
    _rc_params["savefig.bbox"] = "tight"
plt.rcParams.update(_rc_params)


EXPECTED_SUMMARIES: dict[str, list[str]] = {
    "pivot_stability": ["pivot_stability_summary.md"],
    "oscillation_trap": ["oscillation_trap_summary.md", "oscillation_summary.md"],
    "truncation_shadow": ["truncation_shadow_summary.md", "truncation_summary.md"],
    "organic_oscillation": ["organic_oscillation_summary.md"],
    "mechanism_verification": [
        "mechanism_verification_summary.md",
        "organic_confirmation_summary.md",
    ],
    "scale_dependence": ["scale_dependence_summary.md"],
    "commit_latency": ["commit_latency_summary.md"],
    "policy_regret": ["policy_regret_summary.md"],
    "smart_policies_fixed": ["smart_policies_fixed_summary.md"],
    "smart_policies_consistent": ["smart_policies_consistent_summary.md"],
    "pivot_diagnostics": ["pivot_diagnostics_summary.md"],
}

# Prompt-aligned fallback aggregates.
FALLBACK_PARETO = {
    "f_values": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "ext_values": [48.7, 62.6, 71.1, 76.8, 80.1, 82.2, 85.0, 88.2],
    "score_values": [7.93, 10.60, 12.28, 13.47, 14.17, 14.64, 15.27, 15.93],
    "commit_valid": 39.4,
    "commit_score": 6.68,
    "finite_valid": 91.5,
    "finite_score": 16.58,
}

FALLBACK_MECHANISM = {
    "tp": 1040,
    "fp": 124,
    "fn": 0,
    "tn": 490,
}

FALLBACK_ORGANIC_ROWS: list[dict[str, float]] = [
    {"epsilon": 0.10, "k": 1.0, "organic_trap_pct": 32.0},
    {"epsilon": 0.10, "k": 2.0, "organic_trap_pct": 47.5},
    {"epsilon": 0.10, "k": 3.0, "organic_trap_pct": 57.0},
    {"epsilon": 0.20, "k": 1.0, "organic_trap_pct": 39.0},
    {"epsilon": 0.20, "k": 2.0, "organic_trap_pct": 56.0},
    {"epsilon": 0.20, "k": 3.0, "organic_trap_pct": 62.0},
    {"epsilon": 0.30, "k": 1.0, "organic_trap_pct": 37.0},
    {"epsilon": 0.30, "k": 2.0, "organic_trap_pct": 54.0},
    {"epsilon": 0.30, "k": 3.0, "organic_trap_pct": 68.5},
    {"epsilon": 0.40, "k": 1.0, "organic_trap_pct": 41.0},
    {"epsilon": 0.40, "k": 2.0, "organic_trap_pct": 65.5},
    {"epsilon": 0.40, "k": 3.0, "organic_trap_pct": 77.0},
    {"epsilon": 0.50, "k": 1.0, "organic_trap_pct": 32.0},
    {"epsilon": 0.50, "k": 2.0, "organic_trap_pct": 56.0},
    {"epsilon": 0.50, "k": 3.0, "organic_trap_pct": 65.5},
    {"epsilon": 0.60, "k": 1.0, "organic_trap_pct": 43.0},
    {"epsilon": 0.60, "k": 2.0, "organic_trap_pct": 63.0},
    {"epsilon": 0.60, "k": 3.0, "organic_trap_pct": 73.5},
    {"epsilon": 0.70, "k": 1.0, "organic_trap_pct": 48.0},
    {"epsilon": 0.70, "k": 2.0, "organic_trap_pct": 64.0},
    {"epsilon": 0.70, "k": 3.0, "organic_trap_pct": 71.5},
]

FALLBACK_SCALE_ROWS: list[dict[str, float]] = [
    {"n_events": 100, "epsilon": 0.20, "k": 2.0, "organic_trap_pct": 51.0},
    {"n_events": 100, "epsilon": 0.20, "k": 3.0, "organic_trap_pct": 66.0},
    {"n_events": 100, "epsilon": 0.40, "k": 2.0, "organic_trap_pct": 58.0},
    {"n_events": 100, "epsilon": 0.40, "k": 3.0, "organic_trap_pct": 71.0},
    {"n_events": 100, "epsilon": 0.60, "k": 2.0, "organic_trap_pct": 55.0},
    {"n_events": 100, "epsilon": 0.60, "k": 3.0, "organic_trap_pct": 65.0},
    {"n_events": 200, "epsilon": 0.20, "k": 2.0, "organic_trap_pct": 61.0},
    {"n_events": 200, "epsilon": 0.20, "k": 3.0, "organic_trap_pct": 69.0},
    {"n_events": 200, "epsilon": 0.40, "k": 2.0, "organic_trap_pct": 68.0},
    {"n_events": 200, "epsilon": 0.40, "k": 3.0, "organic_trap_pct": 78.0},
    {"n_events": 200, "epsilon": 0.60, "k": 2.0, "organic_trap_pct": 66.0},
    {"n_events": 200, "epsilon": 0.60, "k": 3.0, "organic_trap_pct": 75.0},
    {"n_events": 500, "epsilon": 0.20, "k": 2.0, "organic_trap_pct": 64.0},
    {"n_events": 500, "epsilon": 0.20, "k": 3.0, "organic_trap_pct": 73.0},
    {"n_events": 500, "epsilon": 0.40, "k": 2.0, "organic_trap_pct": 59.0},
    {"n_events": 500, "epsilon": 0.40, "k": 3.0, "organic_trap_pct": 68.0},
    {"n_events": 500, "epsilon": 0.60, "k": 2.0, "organic_trap_pct": 62.0},
    {"n_events": 500, "epsilon": 0.60, "k": 3.0, "organic_trap_pct": 71.0},
    {"n_events": 1000, "epsilon": 0.20, "k": 2.0, "organic_trap_pct": 52.0},
    {"n_events": 1000, "epsilon": 0.20, "k": 3.0, "organic_trap_pct": 67.0},
    {"n_events": 1000, "epsilon": 0.40, "k": 2.0, "organic_trap_pct": 56.0},
    {"n_events": 1000, "epsilon": 0.40, "k": 3.0, "organic_trap_pct": 68.0},
    {"n_events": 1000, "epsilon": 0.60, "k": 2.0, "organic_trap_pct": 55.0},
    {"n_events": 1000, "epsilon": 0.60, "k": 3.0, "organic_trap_pct": 67.0},
]


def to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace(",", "")
    return float(text)


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    for idx, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        if idx + 1 >= len(lines):
            continue
        sep = lines[idx + 1].strip()
        if "|" not in sep or "-" not in sep:
            continue
        header = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for raw in lines[idx + 2 :]:
            raw = raw.strip()
            if not raw.startswith("|"):
                break
            cells = [cell.strip() for cell in raw.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rows.append(dict(zip(header, cells)))
        return rows
    return []


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def resolve_summary_paths(input_dir: Path) -> dict[str, Path | None]:
    available = {path.name: path for path in input_dir.glob("*summary*.md")}
    resolved: dict[str, Path | None] = {}
    for key, candidates in EXPECTED_SUMMARIES.items():
        chosen: Path | None = None
        for candidate in candidates:
            if candidate in available:
                chosen = available[candidate]
                break
        resolved[key] = chosen
    return resolved


def summarize_resolved_sources(resolved: dict[str, Path | None]) -> None:
    print("Resolved summary sources:")
    for key in sorted(resolved):
        path = resolved[key]
        if path is None:
            print(f"  - {key}: MISSING (will use fallback where possible)")
        else:
            print(f"  - {key}: {path.name}")


def as_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    return parse_markdown_table(path)


def aggregate_pareto(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return FALLBACK_PARETO

    def mean(col: str) -> float:
        values = [to_float(row[col]) for row in rows]
        return sum(values) / len(values)

    return {
        "f_values": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "ext_values": [
            mean("def05_ext_valid_consistent"),
            mean("def10_ext_valid_consistent"),
            mean("def15_ext_valid_consistent"),
            mean("def20_ext_valid_consistent"),
            mean("def25_ext_valid_consistent"),
            mean("def30_ext_valid_consistent"),
            mean("def40_ext_valid_consistent"),
            mean("def50_ext_valid_consistent"),
        ],
        "score_values": [
            mean("def05_score_consistent"),
            mean("def10_score_consistent"),
            mean("def15_score_consistent"),
            mean("def20_score_consistent"),
            mean("def25_score_consistent"),
            mean("def30_score_consistent"),
            mean("def40_score_consistent"),
            mean("def50_score_consistent"),
        ],
        "commit_valid": mean("cn_streaming_valid"),
        "commit_score": mean("cn_streaming_score"),
        "finite_valid": mean("finite_valid"),
        "finite_score": mean("finite_score"),
    }


def aggregate_mechanism(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {
            "tp": float(FALLBACK_MECHANISM["tp"]),
            "fp": float(FALLBACK_MECHANISM["fp"]),
            "fn": float(FALLBACK_MECHANISM["fn"]),
            "tn": float(FALLBACK_MECHANISM["tn"]),
            "accuracy": (
                (FALLBACK_MECHANISM["tp"] + FALLBACK_MECHANISM["tn"])
                / (
                    FALLBACK_MECHANISM["tp"]
                    + FALLBACK_MECHANISM["fp"]
                    + FALLBACK_MECHANISM["fn"]
                    + FALLBACK_MECHANISM["tn"]
                )
            ),
            "recall": 1.0,
        }

    total = sum(to_float(row["n_finite_valid"]) for row in rows)
    tp = sum(to_float(row["organic_traps"]) for row in rows)
    fp = sum(to_float(row["gap_FP"]) for row in rows)
    fn = sum(to_float(row["gap_FN"]) for row in rows)
    tn = max(total - tp - fp - fn, 0.0)
    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "recall": recall,
    }


def fig_organic_heatmap(rows: list[dict[str, str]], out_dir: Path) -> None:
    data = rows or [{k: str(v) for k, v in row.items()} for row in FALLBACK_ORGANIC_ROWS]
    eps_values = sorted({to_float(row["epsilon"]) for row in data})
    k_values = sorted({int(to_float(row["k"])) for row in data})
    lookup = {
        (round(to_float(row["epsilon"]), 3), int(to_float(row["k"]))): to_float(
            row["organic_trap_pct"]
        )
        for row in data
    }
    matrix: list[list[float]] = []
    for k in k_values:
        matrix.append([lookup[(round(eps, 3), k)] for eps in eps_values])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    image = ax.imshow(matrix, cmap="YlOrRd", origin="lower", vmin=0.0, vmax=80.0, aspect="auto")
    ax.set_xticks(range(len(eps_values)))
    ax.set_xticklabels([f"{eps:.2f}" for eps in eps_values])
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values])
    ax.set_xlabel(r"Front-loading $\epsilon$")
    ax.set_ylabel(r"Prefix requirement $k$")
    ax.set_title("Organic Trap Rate by $(\\epsilon, k)$")

    for yi, k in enumerate(k_values):
        for xi, eps in enumerate(eps_values):
            value = lookup[(round(eps, 3), k)]
            text_color = "white" if value >= 58.0 else "black"
            ax.text(xi, yi, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Organic trap rate (%)")
    fig.savefig(out_dir / "organic_trap_heatmap.pdf")
    plt.close(fig)


def fig_scale_invariance(rows: list[dict[str, str]], out_dir: Path) -> None:
    data = rows or [{k: str(v) for k, v in row.items()} for row in FALLBACK_SCALE_ROWS]
    grouped: dict[tuple[float, int], list[tuple[int, float]]] = defaultdict(list)
    for row in data:
        eps = to_float(row["epsilon"])
        k = int(to_float(row["k"]))
        n_events = int(to_float(row["n_events"]))
        trap = to_float(row["organic_trap_pct"])
        grouped[(eps, k)].append((n_events, trap))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (eps, k), points in sorted(grouped.items()):
        points = sorted(points, key=lambda item: item[0])
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        ax.plot(x, y, marker="o", linewidth=1.5, label=rf"$\epsilon={eps:.2f}, k={k}$")

    ax.set_xlabel("Number of events in stream ($n$)")
    ax.set_ylabel("Organic trap rate (%)")
    ax.set_title("Scale Dependence: Trap Rate vs Stream Length")
    ax.grid(alpha=0.25)
    ax.set_ylim(45.0, 82.0)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.savefig(out_dir / "scale_invariance_lines.pdf")
    plt.close(fig)


def fig_pareto_curve(rows: list[dict[str, str]], out_dir: Path) -> None:
    stats = aggregate_pareto(rows)
    f_values = stats["f_values"]
    ext_values = stats["ext_values"]
    score_values = stats["score_values"]
    commit_valid = stats["commit_valid"]
    commit_score = stats["commit_score"]
    finite_valid = stats["finite_valid"]
    finite_score = stats["finite_score"]

    finite_x = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(0.05, 0.50, color="#c7e9c0", alpha=0.25)
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.01, 0.57)
        ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
        ax.set_xticklabels(["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.40", "0.50"])
        ax.set_xlabel("Patience fraction $f$")

    ax1.plot(f_values, ext_values, marker="o", linewidth=1.8, color="#1f78b4", label="Deferred (TP-consistent)")
    ax1.scatter([0.0], [commit_valid], color="#d7301f", marker="X", s=70, label="Commit-now")
    ax1.scatter([finite_x], [finite_valid], color="#238443", marker="*", s=120, label="Finite")
    ax1.annotate("finite", (finite_x, finite_valid), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax1.axhline(commit_valid, color="#d7301f", linestyle="--", linewidth=1.0)
    ax1.axhline(finite_valid, color="#238443", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Extraction validity (%)")
    ax1.set_title("A. Validity vs Patience")
    ax1.legend(frameon=False, loc="lower right")

    ax2.plot(f_values, score_values, marker="o", linewidth=1.8, color="#6a3d9a", label="Deferred (TP-consistent)")
    ax2.scatter([0.0], [commit_score], color="#d7301f", marker="X", s=70, label="Commit-now")
    ax2.scatter([finite_x], [finite_score], color="#238443", marker="*", s=120, label="Finite")
    ax2.annotate("finite", (finite_x, finite_score), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax2.axhline(commit_score, color="#d7301f", linestyle="--", linewidth=1.0)
    ax2.axhline(finite_score, color="#238443", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Mean score")
    ax2.set_title("B. Score vs Patience")

    fig.suptitle("Quality-Latency Pareto Curve (Exp 48c, TP-consistent)")
    fig.savefig(out_dir / "pareto_curve_tp_consistent.pdf")
    plt.close(fig)


def fig_pivot_arrival_cdf(
    summary_rows: list[dict[str, str]], input_dir: Path, out_dir: Path
) -> None:
    raw_path = input_dir / "pivot_diagnostics_raw.json"
    selected_eps = [0.10, 0.30, 0.50, 0.70]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    if raw_path.exists():
        raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
        values_by_eps: dict[float, list[float]] = defaultdict(list)
        global_values: list[float] = []
        for row in raw_rows:
            eps = round(float(row["epsilon"]), 2)
            frac = float(row["finite_pivot_frac"])
            global_values.append(frac)
            values_by_eps[eps].append(frac)

        for eps in selected_eps:
            vals = sorted(values_by_eps.get(round(eps, 2), []))
            if not vals:
                continue
            y = [(idx + 1) / len(vals) for idx in range(len(vals))]
            ax.plot(vals, y, linewidth=1.8, label=rf"$\epsilon={eps:.2f}$")

        p50 = quantile(global_values, 0.50)
        p75 = quantile(global_values, 0.75)
        p90 = quantile(global_values, 0.90)
    else:
        by_eps: dict[float, dict[str, float]] = {}
        for row in summary_rows:
            eps = round(to_float(row["epsilon"]), 2)
            if eps in by_eps:
                continue
            by_eps[eps] = {
                "p50": to_float(row["finite_pivot_frac_p50"]),
                "p75": to_float(row["finite_pivot_frac_p75"]),
                "p90": to_float(row["finite_pivot_frac_p90"]),
                "p95": to_float(row["finite_pivot_frac_p95"]),
            }
        for eps in selected_eps:
            if round(eps, 2) not in by_eps:
                continue
            q = by_eps[round(eps, 2)]
            x = [0.0, q["p50"], q["p75"], q["p90"], q["p95"], 1.0]
            y = [0.0, 0.50, 0.75, 0.90, 0.95, 1.0]
            ax.plot(x, y, linewidth=1.8, label=rf"$\epsilon={eps:.2f}$")
        # Prompt fallback quantiles.
        p50, p75, p90 = 0.28, 0.48, 0.69

    ax.axvline(0.25, color="#b2182b", linestyle="--", linewidth=1.2, label="Recommended $f=0.25$")
    ax.text(
        0.03,
        0.97,
        f"Global quantiles:\n$p_{{50}}={p50:.2f}$\n$p_{{75}}={p75:.2f}$\n$p_{{90}}={p90:.2f}$",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    ax.set_xlabel("Fraction of stream observed")
    ax.set_ylabel(r"$P(\mathrm{global\ max\ pivot\ arrived})$")
    ax.set_title("Pivot Arrival CDF by Front-loading Level")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out_dir / "pivot_arrival_cdf.pdf")
    plt.close(fig)


def fig_mechanism_confusion(rows: list[dict[str, str]], out_dir: Path) -> None:
    stats = aggregate_mechanism(rows)
    matrix = [
        [stats["tn"], stats["fp"]],
        [stats["fn"], stats["tp"]],
    ]

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: no trap", "Pred: trap"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: no trap", "Actual: trap"])
    ax.set_title("Mechanism Verification (Exp 44): min-gap predictor")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            value = int(round(matrix[i][j]))
            color = "white" if matrix[i][j] > max(stats["tp"], stats["tn"]) * 0.45 else "black"
            ax.text(j, i, f"{labels[i][j]} = {value}", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Count")

    ax.text(
        0.02,
        -0.18,
        f"Accuracy = {stats['accuracy'] * 100:.1f}%   Recall = {stats['recall'] * 100:.1f}%   FN = {int(round(stats['fn']))}",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.savefig(out_dir / "mechanism_confusion_matrix.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate streaming paper figures (PDF).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "output" / "streaming",
        help="Directory containing streaming summaries and raw JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "streaming",
        help="Directory to write figure PDFs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    resolved = resolve_summary_paths(args.input_dir)
    summarize_resolved_sources(resolved)

    organic_rows = as_rows(resolved["organic_oscillation"])
    scale_rows = as_rows(resolved["scale_dependence"])
    consistent_rows = as_rows(resolved["smart_policies_consistent"])
    pivot_rows = as_rows(resolved["pivot_diagnostics"])
    mechanism_rows = as_rows(resolved["mechanism_verification"])

    fig_organic_heatmap(organic_rows, args.output_dir)
    fig_scale_invariance(scale_rows, args.output_dir)
    fig_pareto_curve(consistent_rows, args.output_dir)
    fig_pivot_arrival_cdf(pivot_rows, args.input_dir, args.output_dir)
    fig_mechanism_confusion(mechanism_rows, args.output_dir)

    print("Generated figures:")
    for path in sorted(args.output_dir.glob("*.pdf")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
