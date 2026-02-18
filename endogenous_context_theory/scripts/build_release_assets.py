#!/usr/bin/env python3
"""Build curated release assets for Paper 03.

This script generates:
1. Exported MirageBench 12-task set (JSON + index CSV)
2. KV eviction merged/summary CSVs from per-retention checkpoints
3. Publication-style release figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_miragebench_ollama import (  # type: ignore
    _load_notebook_runtime,
    _patch_runtime_with_methodology_fixes,
    _validate_investment_ground_truth,
)


def _json_safe(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def export_miragebench_tasks(root: Path, out_dir: Path) -> None:
    nb_path = root / "notebooks" / "miragebench_experiments_colab.ipynb"
    runtime = _load_notebook_runtime(nb_path)
    _patch_runtime_with_methodology_fixes(runtime)
    tasks = runtime["build_miragebench_v01"]()
    _validate_investment_ground_truth(tasks)

    json_path = out_dir / "miragebench_v01_tasks.json"
    index_path = out_dir / "miragebench_v01_task_index.csv"

    task_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    for task in tasks:
        row = {
            "task_id": task.task_id,
            "category": task.category,
            "k": int(task.k),
            "question": task.question,
            "pivot_ground_truth": task.pivot_ground_truth,
            "decoy_pivot": task.decoy_pivot,
            "full_context": task.full_context,
            "compressed_context": task.compressed_context,
            "answer_ground_truth": task.answer_ground_truth,
            "decoy_answer": task.decoy_answer,
            "metadata": task.metadata,
        }
        task_rows.append(_json_safe(row))

        index_rows.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "k": int(task.k),
                "pivot_ground_truth": task.pivot_ground_truth,
                "decoy_pivot": task.decoy_pivot,
                "full_context_chars": len(task.full_context),
                "compressed_context_chars": len(task.compressed_context),
            }
        )

    json_path.write_text(json.dumps(task_rows, indent=2), encoding="utf-8")

    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)


def build_kv_tables(kv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    retention_files = sorted(kv_dir.glob("kv_cache_eviction_retention_*.csv"))
    if not retention_files:
        raise FileNotFoundError(f"No retention checkpoint files found in {kv_dir}")

    frames = [pd.read_csv(p) for p in retention_files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["retention", "task_id"]).reset_index(drop=True)

    if "has_pivot_header" not in merged.columns:
        merged["has_pivot_header"] = (
            merged["truncated_answer"].fillna("").str.contains(r"PIVOT_ID\s*=", regex=True).astype(int)
        )
    else:
        backfill = merged["truncated_answer"].fillna("").str.contains(r"PIVOT_ID\s*=", regex=True).astype(int)
        merged["has_pivot_header"] = merged["has_pivot_header"].fillna(backfill).astype(int)

    summary_base = (
        merged.groupby("retention", as_index=False)
        .agg(
            has_pivot_header=("has_pivot_header", "mean"),
            pivot_preserved=("pivot_preserved", "mean"),
            fixed_pivot_feasible=("fixed_pivot_feasible", "mean"),
            raw_validity=("raw_validity", "mean"),
            semantic_regret=("semantic_regret", "mean"),
        )
        .sort_values("retention", ascending=False)
    )

    header_subset = (
        merged[merged["has_pivot_header"] == 1]
        .groupby("retention", as_index=False)
        .agg(pivot_preserved_given_header=("pivot_preserved", "mean"))
    )

    summary = summary_base.merge(header_subset, on="retention", how="left")
    summary["pivot_preserved_given_header"] = summary["pivot_preserved_given_header"].fillna(0.0)

    merged.to_csv(kv_dir / "kv_cache_eviction_mirage_results.csv", index=False)
    summary.to_csv(kv_dir / "kv_cache_eviction_mirage_summary_by_retention.csv", index=False)
    return merged, summary


def build_blackbox_tables(blackbox_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_path = blackbox_dir / "miragebench_bf16_5model_merged.csv"
    if not merged_path.exists():
        merged_path = blackbox_dir / "miragebench_bf16_7model_merged.csv"
    merged = pd.read_csv(merged_path)
    merged["valid_but_switched"] = (
        (merged["raw_validity"] > 0.5) & (merged["pivot_preserved"] == 0)
    ).astype(int)

    by_model = (
        merged.groupby("model_name", as_index=False)
        .agg(
            raw_validity=("raw_validity", "mean"),
            pivot_preserved=("pivot_preserved", "mean"),
            fixed_pivot_feasible=("fixed_pivot_feasible", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            valid_but_switched=("valid_but_switched", "mean"),
        )
        .sort_values("model_name")
    )

    by_category = (
        merged.groupby("category", as_index=False)
        .agg(
            raw_validity=("raw_validity", "mean"),
            pivot_preserved=("pivot_preserved", "mean"),
            fixed_pivot_feasible=("fixed_pivot_feasible", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            valid_but_switched=("valid_but_switched", "mean"),
        )
        .sort_values("category")
    )

    by_model.to_csv(blackbox_dir / "miragebench_bf16_5model_summary_by_model_release.csv", index=False)
    by_category.to_csv(blackbox_dir / "miragebench_bf16_5model_summary_by_category_release.csv", index=False)
    return by_model, by_category


def build_figures(figures_dir: Path, by_model: pd.DataFrame, kv_summary: pd.DataFrame) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Mirage gap by model
    model_names = by_model["model_name"].tolist()
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width / 2, by_model["raw_validity"], width=width, label="Raw Validity")
    ax.bar(x + width / 2, by_model["pivot_preserved"], width=width, label="Pivot Preserved")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Score")
    ax.set_title("MirageBench: Validity-Preservation Gap by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "blackbox_validity_vs_pivot_preservation.png", dpi=180)
    plt.close(fig)

    # Figure 2: KV retention effects
    kv_sorted = kv_summary.sort_values("retention")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(kv_sorted["retention"], kv_sorted["has_pivot_header"], marker="o", label="Protocol Compliance")
    ax.plot(
        kv_sorted["retention"],
        kv_sorted["pivot_preserved_given_header"],
        marker="o",
        label="Pivot Preserved | Header=1",
    )
    ax.plot(kv_sorted["retention"], kv_sorted["pivot_preserved"], marker="o", label="Pivot Preserved (Overall)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("KV Retention")
    ax.set_ylabel("Rate")
    ax.set_title("KV Eviction: Protocol Collapse vs Pivot Substitution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "kv_retention_protocol_vs_pivot.png", dpi=180)
    plt.close(fig)

    # Figure 3: Release artifact flow map
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")

    boxes = [
        (0.05, 0.58, 0.25, 0.30, "MirageBench Generator\n+ 12 Tasks\nrelease/miragebench_tasks"),
        (0.37, 0.58, 0.25, 0.30, "Blackbox Sweep\n5 Models bf16\nrelease/results/blackbox_bf16_5model"),
        (0.69, 0.58, 0.25, 0.30, "KV Eviction Sweep\nLlama 3.1 8B bf16\nrelease/results/kv_cache_eviction_llama31_8b"),
        (0.21, 0.12, 0.25, 0.30, "Mirage-Aware Adapter\nrelease/adapters/mirage_aware_v1"),
        (0.54, 0.12, 0.25, 0.30, "Paper 03 Sections\npaper_03_new_sections.md"),
    ]
    for x0, y0, w, h, label in boxes:
        rect = plt.Rectangle((x0, y0), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.30, 0.73), (0.37, 0.73)),
        ((0.62, 0.73), (0.69, 0.73)),
        ((0.50, 0.58), (0.66, 0.42)),
        ((0.18, 0.58), (0.30, 0.42)),
        ((0.34, 0.27), (0.54, 0.27)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "linewidth": 1.5},
        )

    ax.set_title("Paper 03 Release Artifact Flow", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "release_artifact_flow.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated Paper 03 release assets.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    release = root / "release"

    tasks_dir = release / "miragebench_tasks"
    blackbox_dir = release / "results" / "blackbox_bf16_5model"
    kv_dir = release / "results" / "kv_cache_eviction_llama31_8b"
    figs_dir = release / "figures"

    tasks_dir.mkdir(parents=True, exist_ok=True)

    export_miragebench_tasks(root, tasks_dir)
    kv_merged, kv_summary = build_kv_tables(kv_dir)
    by_model, _ = build_blackbox_tables(blackbox_dir)
    build_figures(figs_dir, by_model, kv_summary)

    print("Release assets built:")
    print(f"- Tasks: {tasks_dir}")
    print(f"- Blackbox: {blackbox_dir}")
    print(f"- KV: {kv_dir} (rows={len(kv_merged)})")
    print(f"- Figures: {figs_dir}")


if __name__ == "__main__":
    main()
