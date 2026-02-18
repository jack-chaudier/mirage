#!/usr/bin/env python3
"""Run MirageBench against hosted API models (Anthropic/OpenAI/xAI)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from run_miragebench_ollama import (
    _load_notebook_runtime,
    _patch_runtime_with_methodology_fixes,
    _validate_investment_ground_truth,
)

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504, 529}
OUTPUT_COLUMNS = [
    "task_id",
    "model_name",
    "full_answer",
    "compressed_answer",
    "raw_validity",
    "raw_validity_full",
    "raw_validity_compressed",
    "pivot_preserved",
    "semantic_regret",
    "compression_level",
    "category",
    "difficulty_level",
    "full_pivot",
    "compressed_pivot",
    "full_pivot_matches_ground_truth",
    "pivot_matches_ground_truth",
    "pivot_outcome",
    "high_validity_flag",
    "true_mirage_flag",
    "rescue_flag",
    "instability_flag",
    "mirage_flag",
]
OUTCOME_ORDER = (
    "true_mirage",
    "rescue",
    "instability",
    "stable_correct",
    "stable_wrong",
    "unresolved",
)


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MirageBench against Anthropic/OpenAI/xAI APIs with retry and summary outputs.",
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "xai"),
        required=True,
        help="API provider.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID (e.g. claude-haiku-4-5-20251001, grok-4-1-fast, gpt-4o-mini).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key. If omitted, reads ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY based on provider.",
    )
    parser.add_argument(
        "--compression-levels",
        default="0.4,0.5,0.6",
        help="Comma-separated compression levels (default: 0.4,0.5,0.6).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="Maximum tasks (default: 12).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max completion tokens per request (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient errors (default: 3).",
    )
    parser.add_argument(
        "--output",
        default="results/raw/miragebench_api_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Summary CSV path (default: derived from --output).",
    )
    parser.add_argument(
        "--investment-only",
        action="store_true",
        help="Run only investment tasks from the default 12-task suite.",
    )
    parser.add_argument(
        "--investment-difficulty-levels",
        default="",
        help=(
            "Comma-separated investment difficulty levels for sweep mode "
            "(e.g. easy,medium,hard,extreme)."
        ),
    )
    parser.add_argument(
        "--tasks-per-difficulty",
        type=int,
        default=4,
        help="Number of investment tasks to generate per difficulty level (default: 4).",
    )
    return parser.parse_args()


def _extract_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            val = getattr(response, attr, None)
            if isinstance(val, int):
                return val

    return None


def _is_transient_error(exc: Exception) -> bool:
    status = _extract_status_code(exc)
    if status in TRANSIENT_STATUS_CODES:
        return True

    text = str(exc).lower()
    markers = (
        "rate limit",
        "too many requests",
        "overload",
        "temporarily unavailable",
        "timeout",
        "timed out",
    )
    return any(marker in text for marker in markers)


class ResilientGenerator:
    """Wrap a text generator with fixed inter-call delay and retry/backoff."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        delay_seconds: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        self._generate_fn = generate_fn
        self._delay_seconds = max(0.0, float(delay_seconds))
        self._max_retries = max(0, int(max_retries))
        self._last_call_time = 0.0

    def __call__(self, prompt: str) -> str:
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < self._delay_seconds:
            time.sleep(self._delay_seconds - elapsed)

        for attempt in range(self._max_retries + 1):
            try:
                out = self._generate_fn(prompt)
                self._last_call_time = time.monotonic()
                return str(out or "").strip()
            except Exception as exc:
                self._last_call_time = time.monotonic()
                if attempt >= self._max_retries or not _is_transient_error(exc):
                    raise
                sleep_s = self._delay_seconds * (2**attempt)
                status = _extract_status_code(exc)
                print(
                    f"[retry] transient error status={status} "
                    f"attempt={attempt + 1}/{self._max_retries + 1}; sleeping {sleep_s:.2f}s",
                )
                time.sleep(sleep_s)

        raise RuntimeError("Unreachable retry loop state.")


def make_anthropic_generator(
    model: str,
    api_key: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def generate(prompt: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        blocks = getattr(message, "content", [])
        texts: List[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                texts.append(str(text))
        return "".join(texts).strip()

    return generate


def make_openai_generator(
    model: str,
    api_key: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return str(completion.choices[0].message.content or "").strip()

    return generate


def make_xai_generator(
    model: str,
    api_key: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    return make_openai_generator(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url="https://api.x.ai/v1",
    )


def _resolve_api_key(provider: str, explicit_key: str) -> str:
    if explicit_key:
        return explicit_key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY", "").strip()
    if provider == "xai":
        return os.environ.get("XAI_API_KEY", "").strip()
    return ""


def load_runtime_and_tasks(root: Path) -> Tuple[Dict[str, Any], List[Any]]:
    notebook_path = root / "mirage_bench_colab.ipynb"
    runtime = _load_notebook_runtime(notebook_path)
    _patch_runtime_with_methodology_fixes(runtime)
    tasks = runtime["build_miragebench_v01"]()
    _validate_investment_ground_truth(tasks)
    return runtime, tasks


def build_investment_difficulty_tasks(
    runtime: Dict[str, Any],
    levels: Sequence[str],
    tasks_per_level: int,
) -> List[Any]:
    if tasks_per_level <= 0:
        raise ValueError("--tasks-per-difficulty must be > 0")

    presets: Dict[str, Dict[str, float]] = runtime.get("INVESTMENT_DIFFICULTY_PRESETS", {})
    if not presets:
        raise RuntimeError("Investment difficulty presets are not available in runtime.")

    unknown = [lvl for lvl in levels if lvl not in presets]
    if unknown:
        known = ", ".join(sorted(presets))
        bad = ", ".join(unknown)
        raise RuntimeError(
            f"Unknown investment difficulty level(s): {bad}. Known levels: {known}.",
        )

    build_investment_task = runtime["build_investment_task"]
    tasks: List[Any] = []
    task_num = 1
    for difficulty_index, lvl in enumerate(levels):
        for _ in range(tasks_per_level):
            task = build_investment_task(task_num=task_num, difficulty=lvl)
            task.metadata["difficulty_level"] = lvl
            task.metadata["difficulty_index"] = difficulty_index
            tasks.append(task)
            task_num += 1

    return tasks


def run_api_eval(
    runtime: Dict[str, Any],
    tasks: List[Any],
    model_name: str,
    generate_fn: Callable[[str], str],
    compression_levels: Sequence[float],
    max_tasks: Optional[int],
) -> pd.DataFrame:
    make_prompt = runtime["make_prompt"]
    render_compressed_variant = runtime["render_compressed_variant"]
    extract_pivot_id = runtime["extract_pivot_id"]
    raw_validity_score = runtime["raw_validity_score"]
    semantic_regret = runtime["semantic_regret"]
    classify_pivot_outcome = runtime["classify_pivot_outcome"]
    seed = runtime["SEED"]

    rows: List[Dict[str, Any]] = []
    sub_tasks = tasks if max_tasks is None else tasks[:max_tasks]
    total_tasks = len(sub_tasks)

    for index, task in enumerate(sub_tasks, start=1):
        full_prompt = make_prompt(task.full_context, task.question)
        difficulty_level = task.metadata.get("difficulty_level", "")

        try:
            full_answer = generate_fn(full_prompt)
            full_pivot = extract_pivot_id(full_answer, [task.pivot_ground_truth, task.decoy_pivot])
            raw_validity_full = raw_validity_score(full_answer, task)
        except Exception as exc:
            err_msg = str(exc)
            for lvl in compression_levels:
                rows.append(
                    {
                        "task_id": task.task_id,
                        "model_name": model_name,
                        "compression_level": float(lvl),
                        "category": task.category,
                        "difficulty_level": difficulty_level,
                        "error": err_msg,
                    }
                )
            print(f"[error] {model_name} task={task.task_id} stage=full msg={err_msg}")
            continue

        completed_levels = 0
        for lvl in compression_levels:
            try:
                comp_context = render_compressed_variant(task, drop_fraction=lvl, seed=seed)
                comp_prompt = make_prompt(comp_context, task.question)
                compressed_answer = generate_fn(comp_prompt)

                compressed_pivot = extract_pivot_id(
                    compressed_answer,
                    [task.pivot_ground_truth, task.decoy_pivot],
                )
                raw_validity_compressed = raw_validity_score(compressed_answer, task)

                pivot_outcome = classify_pivot_outcome(task, full_pivot, compressed_pivot)
                high_validity = int(raw_validity_compressed >= 0.70)

                row = {
                    "task_id": task.task_id,
                    "model_name": model_name,
                    "full_answer": full_answer,
                    "compressed_answer": compressed_answer,
                    "raw_validity": raw_validity_compressed,
                    "raw_validity_full": raw_validity_full,
                    "raw_validity_compressed": raw_validity_compressed,
                    "pivot_preserved": int(
                        bool(full_pivot and compressed_pivot and full_pivot == compressed_pivot)
                    ),
                    "semantic_regret": semantic_regret(full_answer, compressed_answer),
                    "compression_level": float(lvl),
                    "category": task.category,
                    "difficulty_level": difficulty_level,
                    "full_pivot": full_pivot,
                    "compressed_pivot": compressed_pivot,
                    "full_pivot_matches_ground_truth": int(full_pivot == task.pivot_ground_truth),
                    "pivot_matches_ground_truth": int(compressed_pivot == task.pivot_ground_truth),
                    "pivot_outcome": pivot_outcome,
                    "high_validity_flag": high_validity,
                    "true_mirage_flag": int((pivot_outcome == "true_mirage") and high_validity),
                    "rescue_flag": int((pivot_outcome == "rescue") and high_validity),
                    "instability_flag": int((pivot_outcome == "instability") and high_validity),
                }
                row["mirage_flag"] = row["true_mirage_flag"]
                rows.append(row)
                completed_levels += 1
            except Exception as exc:
                rows.append(
                    {
                        "task_id": task.task_id,
                        "model_name": model_name,
                        "compression_level": float(lvl),
                        "category": task.category,
                        "difficulty_level": difficulty_level,
                        "error": str(exc),
                    }
                )

        print(
            f"[done] model={model_name} task={task.task_id} "
            f"index={index}/{total_tasks} levels={completed_levels}/{len(compression_levels)}",
        )

    return pd.DataFrame(rows)


def _ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    ordered = OUTPUT_COLUMNS + [c for c in out.columns if c not in OUTPUT_COLUMNS]
    return out[ordered]


def _to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _derive_summary_path(output_path: Path, summary_output: str) -> Path:
    if summary_output:
        return Path(summary_output).resolve()

    stem = output_path.stem
    if stem.endswith("_results"):
        summary_stem = stem[: -len("_results")] + "_summary"
    else:
        summary_stem = stem + "_summary"
    return output_path.with_name(summary_stem + output_path.suffix)


def build_summary_df(usable: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    n = int(len(usable))
    overall_full_acc = float(usable["full_pivot_matches_ground_truth"].mean()) if n else np.nan
    rows.append(
        {
            "section": "full_context_accuracy_overall",
            "category": "ALL",
            "outcome": "",
            "metric": "rate",
            "value": overall_full_acc,
            "count": n,
        }
    )

    by_category_acc = (
        usable.groupby("category", as_index=False)["full_pivot_matches_ground_truth"].mean()
        if n
        else pd.DataFrame(columns=["category", "full_pivot_matches_ground_truth"])
    )
    for _, rec in by_category_acc.iterrows():
        rows.append(
            {
                "section": "full_context_accuracy_by_category",
                "category": str(rec["category"]),
                "outcome": "",
                "metric": "rate",
                "value": float(rec["full_pivot_matches_ground_truth"]),
                "count": int((usable["category"] == rec["category"]).sum()),
            }
        )

    outcomes = usable["pivot_outcome"].fillna("unresolved") if n else pd.Series(dtype=str)
    overall_counts = outcomes.value_counts(dropna=False).to_dict()
    for outcome in OUTCOME_ORDER:
        count = int(overall_counts.get(outcome, 0))
        rows.append(
            {
                "section": "outcome_breakdown_overall",
                "category": "ALL",
                "outcome": outcome,
                "metric": "rate",
                "value": (count / n) if n else np.nan,
                "count": count,
            }
        )

    for category, grp in usable.groupby("category"):
        c_n = int(len(grp))
        c_counts = grp["pivot_outcome"].fillna("unresolved").value_counts(dropna=False).to_dict()
        for outcome in OUTCOME_ORDER:
            c_count = int(c_counts.get(outcome, 0))
            rows.append(
                {
                    "section": "outcome_breakdown_by_category",
                    "category": str(category),
                    "outcome": outcome,
                    "metric": "rate",
                    "value": (c_count / c_n) if c_n else np.nan,
                    "count": c_count,
                }
            )

    if "difficulty_level" in usable.columns:
        tmp = usable.copy()
        tmp["difficulty_level"] = tmp["difficulty_level"].fillna("").astype(str)
        tmp = tmp[tmp["difficulty_level"] != ""]
        for difficulty_level, grp in tmp.groupby("difficulty_level"):
            d_n = int(len(grp))
            d_acc = float(grp["full_pivot_matches_ground_truth"].mean()) if d_n else np.nan
            rows.append(
                {
                    "section": "full_context_accuracy_by_difficulty",
                    "category": "investment",
                    "outcome": "",
                    "metric": "rate",
                    "value": d_acc,
                    "count": d_n,
                    "difficulty_level": difficulty_level,
                }
            )

            d_counts = grp["pivot_outcome"].fillna("unresolved").value_counts(dropna=False).to_dict()
            for outcome in OUTCOME_ORDER:
                d_count = int(d_counts.get(outcome, 0))
                rows.append(
                    {
                        "section": "outcome_breakdown_by_difficulty",
                        "category": "investment",
                        "outcome": outcome,
                        "metric": "rate",
                        "value": (d_count / d_n) if d_n else np.nan,
                        "count": d_count,
                        "difficulty_level": difficulty_level,
                    }
                )

    return pd.DataFrame(rows)


def print_summary(usable: pd.DataFrame) -> None:
    if usable.empty:
        print("No successful rows available for summary.")
        return

    overall_acc = float(usable["full_pivot_matches_ground_truth"].mean())
    print("\nFull-context pivot accuracy:")
    print(
        f"  overall: {overall_acc:.3f} "
        f"({int(usable['full_pivot_matches_ground_truth'].sum())}/{len(usable)})",
    )

    by_cat = (
        usable.groupby("category", as_index=False)["full_pivot_matches_ground_truth"]
        .mean()
        .sort_values("category")
    )
    for _, rec in by_cat.iterrows():
        c = rec["category"]
        rate = float(rec["full_pivot_matches_ground_truth"])
        denom = int((usable["category"] == c).sum())
        numer = int(usable.loc[usable["category"] == c, "full_pivot_matches_ground_truth"].sum())
        print(f"  {c}: {rate:.3f} ({numer}/{denom})")

    print("\nOutcome breakdown (overall):")
    vc = usable["pivot_outcome"].fillna("unresolved").value_counts(dropna=False).to_dict()
    for outcome in OUTCOME_ORDER:
        count = int(vc.get(outcome, 0))
        rate = count / max(1, len(usable))
        print(f"  {outcome}: {rate:.3f} ({count}/{len(usable)})")

    print("\nOutcome breakdown (by category):")
    for category, grp in usable.groupby("category"):
        print(f"  {category}:")
        c_n = len(grp)
        c_vc = grp["pivot_outcome"].fillna("unresolved").value_counts(dropna=False).to_dict()
        for outcome in OUTCOME_ORDER:
            count = int(c_vc.get(outcome, 0))
            rate = count / max(1, c_n)
            print(f"    {outcome}: {rate:.3f} ({count}/{c_n})")

    if "difficulty_level" in usable.columns:
        tmp = usable.copy()
        tmp["difficulty_level"] = tmp["difficulty_level"].fillna("").astype(str)
        tmp = tmp[tmp["difficulty_level"] != ""]
        if not tmp.empty:
            print("\nOutcome breakdown (by difficulty):")
            for difficulty_level, grp in tmp.groupby("difficulty_level"):
                d_n = len(grp)
                d_acc = float(grp["full_pivot_matches_ground_truth"].mean())
                print(f"  {difficulty_level} | full-ctx acc: {d_acc:.3f} ({int(grp['full_pivot_matches_ground_truth'].sum())}/{d_n})")
                d_vc = grp["pivot_outcome"].fillna("unresolved").value_counts(dropna=False).to_dict()
                for outcome in OUTCOME_ORDER:
                    count = int(d_vc.get(outcome, 0))
                    rate = count / max(1, d_n)
                    print(f"    {outcome}: {rate:.3f} ({count}/{d_n})")


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
    output_path = Path(args.output).resolve()
    summary_path = _derive_summary_path(output_path, args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = _resolve_api_key(args.provider, args.api_key)
    if not api_key:
        env_name = (
            "ANTHROPIC_API_KEY"
            if args.provider == "anthropic"
            else "OPENAI_API_KEY"
            if args.provider == "openai"
            else "XAI_API_KEY"
        )
        raise RuntimeError(
            f"Missing API key. Pass --api-key or set {env_name}.",
        )

    compression_levels = _parse_csv_floats(args.compression_levels)
    runtime, tasks = load_runtime_and_tasks(here)
    difficulty_levels = _parse_csv_strings(args.investment_difficulty_levels)

    if difficulty_levels:
        tasks = build_investment_difficulty_tasks(
            runtime=runtime,
            levels=difficulty_levels,
            tasks_per_level=args.tasks_per_difficulty,
        )
    elif args.investment_only:
        tasks = [task for task in tasks if task.category == "investment"]

    effective_max_tasks = args.max_tasks
    if difficulty_levels and args.max_tasks == 12:
        # Sweep mode defaults to full generated matrix unless user explicitly overrides.
        effective_max_tasks = len(tasks)
    print(
        f"Running model={args.model} provider={args.provider} "
        f"tasks={len(tasks)} eval_max_tasks={effective_max_tasks} "
        f"compression_levels={compression_levels}",
    )
    if difficulty_levels:
        print(
            f"Investment difficulty sweep enabled: levels={difficulty_levels} "
            f"tasks_per_level={args.tasks_per_difficulty}",
        )

    if args.provider == "anthropic":
        base_gen = make_anthropic_generator(
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.provider == "xai":
        base_gen = make_xai_generator(
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        base_gen = make_openai_generator(
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    resilient_gen = ResilientGenerator(
        generate_fn=base_gen,
        delay_seconds=args.request_delay_seconds,
        max_retries=args.max_retries,
    )

    results_df = run_api_eval(
        runtime=runtime,
        tasks=tasks,
        model_name=args.model,
        generate_fn=resilient_gen,
        compression_levels=compression_levels,
        max_tasks=effective_max_tasks,
    )

    results_df = _ensure_output_schema(results_df)
    results_df.to_csv(output_path, index=False)

    usable = results_df.copy()
    if "error" in usable.columns and usable["error"].notna().any():
        error_count = int(usable["error"].notna().sum())
        print(f"\nWarning: {error_count} row(s) have errors and were excluded from summaries.")
        usable = usable[usable["error"].isna()].copy()

    usable = _to_numeric(
        usable,
        (
            "raw_validity",
            "raw_validity_full",
            "raw_validity_compressed",
            "pivot_preserved",
            "semantic_regret",
            "compression_level",
            "full_pivot_matches_ground_truth",
            "pivot_matches_ground_truth",
            "high_validity_flag",
            "true_mirage_flag",
            "rescue_flag",
            "instability_flag",
            "mirage_flag",
        ),
    )

    summary_df = build_summary_df(usable)
    summary_df.to_csv(summary_path, index=False)

    print_summary(usable)
    print(f"\nSaved results CSV: {output_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
