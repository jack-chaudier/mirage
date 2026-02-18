#!/usr/bin/env python3
"""Run MirageBench black-box evaluation against local Ollama models.

This script reuses the MirageBench task generation and metric logic from
`mirage_bench_colab.ipynb`, but swaps in Ollama's local generation API.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

NOTEBOOK_CELL_IDS = (8, 9, 10, 16, 17)
DEFAULT_MODELS = ("llama3.1:8b", "mistral:7b", "qwen2.5:7b")
MODEL_LABELS = {
    "llama3.1:8b": "Llama",
    "mistral:7b": "Mistral",
    "qwen2.5:7b": "Qwen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MirageBench 12x3xN Ollama model sweep and export comparison tables."
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Ollama model tags (default: llama3.1:8b,mistral:7b,qwen2.5:7b).",
    )
    parser.add_argument(
        "--compression-levels",
        default="0.4,0.5,0.6",
        help="Comma-separated compression levels (default: 0.4,0.5,0.6).",
    )
    parser.add_argument(
        "--max-tasks-per-model",
        type=int,
        default=12,
        help="Maximum tasks per model (default: 12).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama generation temperature (default: 0.0).",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=220,
        help="Ollama max generated tokens (default: 220).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=240.0,
        help="HTTP timeout per generation call (default: 240).",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw",
        help="Output directory relative to endogenous_context_theory (default: results/raw).",
    )
    return parser.parse_args()


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _load_notebook_runtime(notebook_path: Path) -> Dict[str, Any]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    module_name = "__miragebench_runtime__"
    runtime_module = types.ModuleType(module_name)
    sys.modules[module_name] = runtime_module
    runtime: Dict[str, Any] = runtime_module.__dict__
    runtime.update(
        {
        "__name__": module_name,
        "__file__": str(notebook_path),
        "SEED": 42,
        "np": np,
        "pd": pd,
        "json": json,
        "re": re,
        "warnings": warnings,
        "cosine": cosine,
        "TfidfVectorizer": TfidfVectorizer,
        "tqdm": tqdm,
        "asdict": asdict,
        "dataclass": dataclass,
        "field": field,
        "Path": Path,
        "Any": Any,
        "Callable": Callable,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        }
    )

    for cell_id in NOTEBOOK_CELL_IDS:
        cell = nb["cells"][cell_id]
        if cell.get("cell_type") != "code":
            raise RuntimeError(f"Notebook cell {cell_id} is not code.")
        exec("".join(cell.get("source", [])), runtime)

    def make_prompt(context: str, question: str) -> str:
        return (
            "You are a precise analyst. Follow the scoring rule in the prompt exactly.\n\n"
            + context.strip()
            + "\n\nQuestion:\n"
            + question.strip()
            + "\n\nAnswer:"
        )

    # Keep semantic regret lightweight and fully local (TF-IDF fallback path).
    runtime["_semantic_embedder"] = None
    runtime["_get_semantic_embedder"] = lambda: None
    runtime["make_prompt"] = make_prompt
    return runtime


def _fetch_ollama_tags(host: str) -> List[str]:
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=20)
    response.raise_for_status()
    data = response.json()
    return [item.get("name", "") for item in data.get("models", []) if item.get("name")]


def _build_ollama_generator(
    host: str,
    model_tag: str,
    temperature: float,
    num_predict: int,
    timeout_seconds: float,
) -> Callable[[str], str]:
    url = f"{host.rstrip('/')}/api/generate"

    def _generate(prompt: str) -> str:
        response = requests.post(
            url,
            json={
                "model": model_tag,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": float(temperature), "num_predict": int(num_predict)},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()

    return _generate


def _to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _patch_runtime_with_methodology_fixes(runtime: Dict[str, Any]) -> None:
    """Override selected notebook functions with corrected benchmark methodology."""

    MirageBenchTask = runtime["MirageBenchTask"]
    np_mod = runtime["np"]
    pd_mod = runtime["pd"]
    tqdm_fn = runtime["tqdm"]
    make_prompt_fn = runtime["make_prompt"]
    render_compressed_variant_fn = runtime["render_compressed_variant"]
    extract_pivot_id_fn = runtime["extract_pivot_id"]
    raw_validity_score_fn = runtime["raw_validity_score"]
    semantic_regret_fn = runtime["semantic_regret"]
    seed_value = runtime["SEED"]
    _long_note_fn = runtime["_long_note"]
    _render_context_fn = runtime["_render_context"]
    _compress_records_to_target_fn = runtime["_compress_records_to_target"]
    _build_question_fn = runtime["_build_question"]
    investment_difficulty_presets: Dict[str, Dict[str, float]] = {
        # Baseline (existing behavior).
        "easy": {"cumulative_boost": 6.0, "peer_margin": 0.8, "decoy_weekly": 4.8},
        # Narrower pivot-decoy margin.
        "medium": {"cumulative_boost": 2.5, "peer_margin": 0.3, "decoy_weekly": 5.0},
        # Tight margin where decoy is plausibly maximal after compression.
        "hard": {"cumulative_boost": 1.0, "peer_margin": 0.1, "decoy_weekly": 5.2},
        # Near-tie regime to probe precision limits.
        "extreme": {"cumulative_boost": 0.3, "peer_margin": 0.05, "decoy_weekly": 5.3},
    }

    def build_investment_task(
        task_num: int,
        k: int = 3,
        target_words: int = 3600,
        difficulty: str = "easy",
        difficulty_overrides: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Build investment tasks while guaranteeing pivot ground-truth consistency."""

        rng = np_mod.random.default_rng(2000 + task_num)
        task_id = f"B{task_num:02d}"
        n_events = 84

        pivot_idx = int(rng.integers(50, 62))
        decoy_idx = int(min(n_events - 6, pivot_idx + rng.integers(7, 13)))
        true_setup_idx = [pivot_idx - 9, pivot_idx - 6, pivot_idx - 3, pivot_idx - 1]
        decoy_setup_idx = [decoy_idx - 5, decoy_idx - 3, decoy_idx - 1]

        positions = [
            "NorthRiver Utilities Carry",
            "Aurelia AI Semiconductor Basket",
            "Helios Grid Infrastructure",
            "BlueHarbor Treasury Arbitrage",
            "Cinder Logistics Credit",
        ]
        pivot_position_name = "Helios Grid Infrastructure"
        decoy_position_name = "Aurelia AI Semiconductor Basket"
        cfg = dict(
            investment_difficulty_presets.get(
                difficulty,
                investment_difficulty_presets["easy"],
            )
        )
        if difficulty_overrides:
            cfg.update(difficulty_overrides)
        cumulative_boost = float(cfg["cumulative_boost"])
        pivot_peer_margin = float(cfg["peer_margin"])
        decoy_weekly = float(cfg.get("decoy_weekly", 4.8))
        pivot_weekly = float(cfg.get("pivot_weekly", 5.4))
        decoy_jump = float(cfg.get("decoy_jump", 1.2))

        records: List[Dict[str, Any]] = []
        cumulative = {p: 0.0 for p in positions}
        pivot_ceiling: Optional[float] = None
        pivot_self_margin = max(0.01, min(0.1, pivot_peer_margin * 0.5))

        for i in range(n_events):
            marker = f"{task_id}-E{i+1:03d}"
            wk = f"Week-{i+1:02d}"
            position = positions[i % len(positions)]

            role = "routine"
            weekly = float(rng.normal(0.8, 0.9))

            if i in true_setup_idx:
                role = "setup"
                weekly = float(rng.normal(0.2, 0.2))
            elif i in decoy_setup_idx:
                role = "decoy_setup"
                weekly = float(rng.normal(0.6, 0.3))
            elif i == pivot_idx:
                role = "pivot"
                position = pivot_position_name
                weekly = pivot_weekly
            elif i == decoy_idx:
                role = "candidate"
                position = decoy_position_name
                weekly = decoy_weekly

            cumulative[position] += weekly

            if i == pivot_idx:
                cumulative[position] = max(cumulative.values()) + cumulative_boost
                pivot_ceiling = cumulative[position]
            if i == decoy_idx and pivot_ceiling is not None:
                cumulative[position] = min(
                    max(v for k2, v in cumulative.items() if k2 != position) + decoy_jump,
                    pivot_ceiling - pivot_peer_margin,
                )

            # Hard clamp post-pivot entries so no later record can overtake pivot ground truth.
            if pivot_ceiling is not None and i > pivot_idx:
                cap = pivot_ceiling - (
                    pivot_self_margin if position == pivot_position_name else pivot_peer_margin
                )
                cumulative[position] = min(cumulative[position], cap)

            cum_val = cumulative[position]
            regime = int(rng.integers(1, 6))

            note_role = (
                "pivot"
                if role == "pivot"
                else "decoy"
                if role in {"candidate", "decoy_setup"}
                else role
            )
            note = _long_note_fn(
                rng,
                note_role
                if note_role in {"setup", "pivot", "decoy", "routine"}
                else "routine",
                "portfolio research",
            )

            line = (
                f"[{marker}] {wk} | Position={position} | WeeklyReturn={weekly:+.2f}% | "
                f"CumulativeReturn={cum_val:.2f}% | RegimeScore={regime} | {note}"
            )
            records.append(
                {
                    "marker": marker,
                    "role": role,
                    "line": line,
                    "position": position,
                    "cum": cum_val,
                }
            )

        pivot_marker = f"{task_id}-E{pivot_idx+1:03d}"
        decoy_marker = f"{task_id}-E{decoy_idx+1:03d}"
        true_setup_markers = [f"{task_id}-E{i+1:03d}" for i in true_setup_idx]
        decoy_setup_markers = [f"{task_id}-E{i+1:03d}" for i in decoy_setup_idx]

        # Safety check: enforce generator-ground-truth agreement.
        max_cum_marker = max(records, key=lambda r: float(r["cum"]))["marker"]
        if max_cum_marker != pivot_marker:
            raise RuntimeError(
                f"Investment task {task_id} invalid: max cumulative marker {max_cum_marker} != pivot {pivot_marker}."
            )

        pivot_position = next(r["position"] for r in records if r["marker"] == pivot_marker)
        decoy_position = next(r["position"] for r in records if r["marker"] == decoy_marker)

        preamble = (
            f"Investment Committee Timeline {task_id}\n"
            "The portfolio diary contains noisy market commentary and low-salience condition markers.\n"
            "Interpretation rule: anchor analysis must follow the highest cumulative-return position."
        )
        rule_block = (
            "Rule reminder: anchor position = entry with max(CumulativeReturn) in the full timeline."
        )

        full_context, spans, appendix_text = _render_context_fn(
            preamble=preamble,
            records=records,
            rule_block=rule_block,
            appendix_target_words=target_words,
            rng=rng,
        )

        layout = {
            "preamble": preamble,
            "records": records,
            "rule_block": rule_block,
            "appendix_text": appendix_text,
            "spans": spans,
            "protected_markers": [pivot_marker, decoy_marker, *decoy_setup_markers],
            "candidate_markers": [pivot_marker, decoy_marker],
            "candidate_requirements": {
                pivot_marker: true_setup_markers,
                decoy_marker: decoy_setup_markers,
            },
            "pivot_setup_markers": true_setup_markers,
            "decoy_setup_markers": decoy_setup_markers,
            "pivot_position": pivot_position,
            "decoy_position": decoy_position,
            "difficulty_level": difficulty,
            "difficulty_config": cfg,
        }

        compressed_context, actual_drop, removed_markers = _compress_records_to_target_fn(
            layout,
            target_drop_fraction=0.50,
            rng=rng,
        )

        question = _build_question_fn("investment")
        answer_gt = (
            f"PIVOT_ID={pivot_marker}. Anchor position is {pivot_position}. "
            f"Prerequisite market conditions are encoded in {true_setup_markers[0]}, {true_setup_markers[1]}, {true_setup_markers[2]} before {pivot_marker}."
        )
        decoy_answer = (
            f"PIVOT_ID={decoy_marker}. A coherent but wrong narrative centers {decoy_position} and cites "
            f"{decoy_setup_markers[0]}, {decoy_setup_markers[1]} as enabling conditions."
        )

        layout["compression_default_drop"] = actual_drop
        layout["removed_markers_default"] = removed_markers

        return MirageBenchTask(
            task_id=task_id,
            category="investment",
            full_context=full_context,
            compressed_context=compressed_context,
            question=question,
            pivot_ground_truth=pivot_marker,
            answer_ground_truth=answer_gt,
            decoy_pivot=decoy_marker,
            decoy_answer=decoy_answer,
            k=k,
            metadata=layout,
        )

    def _classify_pivot_outcome(task: Any, full_pivot: str, compressed_pivot: str) -> str:
        full_correct = bool(full_pivot and full_pivot == task.pivot_ground_truth)
        comp_correct = bool(compressed_pivot and compressed_pivot == task.pivot_ground_truth)

        if not full_pivot or not compressed_pivot:
            return "unresolved"
        if full_pivot == compressed_pivot:
            return "stable_correct" if full_correct else "stable_wrong"
        if full_correct and not comp_correct:
            return "true_mirage"
        if (not full_correct) and comp_correct:
            return "rescue"
        if (not full_correct) and (not comp_correct):
            return "instability"
        return "other"

    def run_blackbox_eval(
        tasks: List[Any],
        model_generators: Dict[str, Callable[[str], str]],
        compression_levels: Sequence[float] = (0.4, 0.5, 0.6),
        max_tasks_per_model: Optional[int] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for model_name, gen_fn in model_generators.items():
            sub_tasks = tasks if max_tasks_per_model is None else tasks[:max_tasks_per_model]
            for task in tqdm_fn(sub_tasks, desc=f"Evaluating {model_name}"):
                full_prompt = make_prompt_fn(task.full_context, task.question)
                try:
                    # Generate once per (task, model) and reuse across compression levels.
                    full_answer = gen_fn(full_prompt)
                    full_pivot = extract_pivot_id_fn(
                        full_answer, [task.pivot_ground_truth, task.decoy_pivot]
                    )
                    raw_validity_full = raw_validity_score_fn(full_answer, task)
                except Exception as exc:
                    for lvl in compression_levels:
                        rows.append(
                            {
                                "task_id": task.task_id,
                                "model_name": model_name,
                                "compression_level": lvl,
                                "category": task.category,
                                "error": str(exc),
                            }
                        )
                    continue

                for lvl in compression_levels:
                    try:
                        comp_context = render_compressed_variant_fn(
                            task, drop_fraction=lvl, seed=seed_value
                        )
                        comp_prompt = make_prompt_fn(comp_context, task.question)
                        compressed_answer = gen_fn(comp_prompt)

                        compressed_pivot = extract_pivot_id_fn(
                            compressed_answer,
                            [task.pivot_ground_truth, task.decoy_pivot],
                        )
                        raw_validity_compressed = raw_validity_score_fn(compressed_answer, task)

                        full_pivot_correct = int(full_pivot == task.pivot_ground_truth)
                        compressed_pivot_correct = int(compressed_pivot == task.pivot_ground_truth)
                        pivot_preserved = bool(
                            full_pivot and compressed_pivot and full_pivot == compressed_pivot
                        )
                        pivot_outcome = _classify_pivot_outcome(
                            task, full_pivot, compressed_pivot
                        )
                        high_validity = int(raw_validity_compressed >= 0.70)

                        row = {
                            "task_id": task.task_id,
                            "model_name": model_name,
                            "full_answer": full_answer,
                            "compressed_answer": compressed_answer,
                            "raw_validity": raw_validity_compressed,
                            "raw_validity_full": raw_validity_full,
                            "raw_validity_compressed": raw_validity_compressed,
                            "pivot_preserved": int(pivot_preserved),
                            "semantic_regret": semantic_regret_fn(full_answer, compressed_answer),
                            "compression_level": float(lvl),
                            "category": task.category,
                            "full_pivot": full_pivot,
                            "compressed_pivot": compressed_pivot,
                            "full_pivot_matches_ground_truth": full_pivot_correct,
                            "pivot_matches_ground_truth": compressed_pivot_correct,
                            "pivot_outcome": pivot_outcome,
                            "high_validity_flag": high_validity,
                            "true_mirage_flag": int(
                                (pivot_outcome == "true_mirage") and high_validity
                            ),
                            "rescue_flag": int((pivot_outcome == "rescue") and high_validity),
                            "instability_flag": int(
                                (pivot_outcome == "instability") and high_validity
                            ),
                        }
                        # Backward-compatible column name; now counts only true mirage.
                        row["mirage_flag"] = row["true_mirage_flag"]
                        rows.append(row)
                    except Exception as exc:
                        rows.append(
                            {
                                "task_id": task.task_id,
                                "model_name": model_name,
                                "compression_level": lvl,
                                "category": task.category,
                                "error": str(exc),
                            }
                        )

        return pd_mod.DataFrame(rows)

    runtime["build_investment_task"] = build_investment_task
    runtime["run_blackbox_eval"] = run_blackbox_eval
    runtime["classify_pivot_outcome"] = _classify_pivot_outcome
    runtime["INVESTMENT_DIFFICULTY_PRESETS"] = investment_difficulty_presets


def _validate_investment_ground_truth(tasks: Sequence[Any]) -> None:
    bad: List[str] = []
    for task in tasks:
        if getattr(task, "category", "") != "investment":
            continue
        records = task.metadata.get("records", [])
        if not records:
            bad.append(f"{task.task_id}: missing records")
            continue
        max_marker = max(records, key=lambda r: float(r.get("cum", float("-inf")))).get("marker")
        if max_marker != task.pivot_ground_truth:
            bad.append(
                f"{task.task_id}: pivot_ground_truth={task.pivot_ground_truth}, max_cum_marker={max_marker}"
            )
    if bad:
        bad_text = "; ".join(bad)
        raise RuntimeError(f"Investment ground-truth validation failed: {bad_text}")


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
    notebook_path = here / "mirage_bench_colab.ipynb"
    output_dir = (here / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_models = _parse_csv_strings(args.models)
    compression_levels = _parse_csv_floats(args.compression_levels)

    runtime = _load_notebook_runtime(notebook_path)
    _patch_runtime_with_methodology_fixes(runtime)
    tasks = runtime["build_miragebench_v01"]()
    _validate_investment_ground_truth(tasks)

    available_tags = set(_fetch_ollama_tags(args.host))
    missing = [m for m in requested_models if m not in available_tags]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Requested model(s) not pulled in Ollama: {missing_text}. Run `ollama pull <model>` first."
        )

    model_generators: Dict[str, Callable[[str], str]] = {}
    for model_tag in requested_models:
        model_name = MODEL_LABELS.get(model_tag, model_tag)
        model_generators[model_name] = _build_ollama_generator(
            host=args.host,
            model_tag=model_tag,
            temperature=args.temperature,
            num_predict=args.num_predict,
            timeout_seconds=args.timeout_seconds,
        )

    results_df = runtime["run_blackbox_eval"](
        tasks=tasks,
        model_generators=model_generators,
        compression_levels=compression_levels,
        max_tasks_per_model=args.max_tasks_per_model,
    )
    results_path = output_dir / "miragebench_ollama_blackbox_results.csv"
    results_df.to_csv(results_path, index=False)

    usable = results_df.copy()
    if "error" in usable.columns:
        if usable["error"].notna().any():
            error_count = int(usable["error"].notna().sum())
            print(f"Warning: {error_count} rows contain errors and are excluded from summaries.")
            usable = usable[usable["error"].isna()].copy()
    if usable.empty:
        raise RuntimeError("No successful evaluation rows were produced.")

    usable = _to_numeric(
        usable,
        cols=(
            "raw_validity",
            "raw_validity_full",
            "raw_validity_compressed",
            "semantic_regret",
            "pivot_preserved",
            "pivot_matches_ground_truth",
            "full_pivot_matches_ground_truth",
            "true_mirage_flag",
            "rescue_flag",
            "instability_flag",
            "high_validity_flag",
            "mirage_flag",
            "compression_level",
        ),
    )
    if "raw_validity_compressed" not in usable.columns and "raw_validity" in usable.columns:
        usable["raw_validity_compressed"] = usable["raw_validity"]
    if "raw_validity_full" not in usable.columns:
        usable["raw_validity_full"] = np.nan
    if "true_mirage_flag" not in usable.columns and "mirage_flag" in usable.columns:
        usable["true_mirage_flag"] = usable["mirage_flag"]
    if "rescue_flag" not in usable.columns:
        usable["rescue_flag"] = np.nan
    if "instability_flag" not in usable.columns:
        usable["instability_flag"] = np.nan
    if "full_pivot_matches_ground_truth" not in usable.columns:
        usable["full_pivot_matches_ground_truth"] = np.nan
    if "high_validity_flag" not in usable.columns:
        usable["high_validity_flag"] = np.nan

    summary_by_level = (
        usable.groupby(["model_name", "category", "compression_level"], as_index=False)
        .agg(
            raw_validity_compressed=("raw_validity_compressed", "mean"),
            raw_validity_full=("raw_validity_full", "mean"),
            pivot_preservation=("pivot_preserved", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            full_context_pivot_gt_rate=("full_pivot_matches_ground_truth", "mean"),
            compressed_pivot_gt_rate=("pivot_matches_ground_truth", "mean"),
            true_mirage_rate=("true_mirage_flag", "mean"),
            rescue_rate=("rescue_flag", "mean"),
            instability_rate=("instability_flag", "mean"),
            high_validity_rate=("high_validity_flag", "mean"),
            n=("task_id", "count"),
        )
        .sort_values(["model_name", "category", "compression_level"])
    )
    summary_by_level_path = output_dir / "miragebench_ollama_blackbox_summary.csv"
    summary_by_level.to_csv(summary_by_level_path, index=False)

    leaderboard = (
        usable.groupby("model_name", as_index=False)
        .agg(
            mean_raw_validity_compressed=("raw_validity_compressed", "mean"),
            mean_raw_validity_full=("raw_validity_full", "mean"),
            mean_pivot_preservation=("pivot_preserved", "mean"),
            mean_semantic_regret=("semantic_regret", "mean"),
            true_mirage_rate=("true_mirage_flag", "mean"),
            rescue_rate=("rescue_flag", "mean"),
            instability_rate=("instability_flag", "mean"),
            full_context_pivot_gt_rate=("full_pivot_matches_ground_truth", "mean"),
            compressed_pivot_gt_rate=("pivot_matches_ground_truth", "mean"),
            n=("task_id", "count"),
        )
        .sort_values(
            [
                "full_context_pivot_gt_rate",
                "mean_pivot_preservation",
                "mean_raw_validity_compressed",
            ],
            ascending=[False, False, False],
        )
    )
    leaderboard_path = output_dir / "miragebench_ollama_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    summary_table = (
        usable.groupby(["model_name", "category"], as_index=False)
        .agg(
            raw_validity_compressed=("raw_validity_compressed", "mean"),
            raw_validity_full=("raw_validity_full", "mean"),
            pivot_preservation=("pivot_preserved", "mean"),
            semantic_regret=("semantic_regret", "mean"),
            full_context_pivot_gt_rate=("full_pivot_matches_ground_truth", "mean"),
            compressed_pivot_gt_rate=("pivot_matches_ground_truth", "mean"),
            true_mirage_rate=("true_mirage_flag", "mean"),
            rescue_rate=("rescue_flag", "mean"),
            instability_rate=("instability_flag", "mean"),
            n=("task_id", "count"),
        )
        .sort_values(["model_name", "category"])
    )
    summary_table_path = output_dir / "miragebench_ollama_summary_table.csv"
    summary_table.to_csv(summary_table_path, index=False)

    release_tasks = [
        {
            "task_id": task.task_id,
            "category": task.category,
            "full_context": task.full_context,
            "compressed_context": task.compressed_context,
            "question": task.question,
            "pivot_ground_truth": task.pivot_ground_truth,
            "answer_ground_truth": task.answer_ground_truth,
            "decoy_pivot": task.decoy_pivot,
            "decoy_answer": task.decoy_answer,
            "k": task.k,
        }
        for task in tasks
    ]
    tasks_path = output_dir / "miragebench_v0_1_tasks.json"
    tasks_path.write_text(json.dumps(release_tasks, indent=2), encoding="utf-8")

    print(f"Saved results: {results_path}")
    print(f"Saved level summary: {summary_by_level_path}")
    print(f"Saved leaderboard: {leaderboard_path}")
    print(f"Saved summary table: {summary_table_path}")
    print(f"Saved tasks JSON: {tasks_path}")
    print("\nThree-model comparison:")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
