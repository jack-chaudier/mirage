#!/usr/bin/env python3
"""Evaluate base vs fine-tuned Gemma on mirage-aware behavior."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PIVOT_REGEX = re.compile(r"PIVOT_ID\s*=\s*([A-Z]\d{1,4}-E\d{3})", re.IGNORECASE)
MARKER_REGEX = re.compile(r"([A-Z]\d{1,4}-E\d{3})")
DEGRADED_REGEX = re.compile(r"Evidence assessment:\s*DEGRADED", re.IGNORECASE)
EVIDENCE_REGEX = re.compile(r"Evidence assessment:\s*(DEGRADED|STRONG)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mirage-aware adapter against base model.")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-2-2b-it-4bit",
        help="Base model id/path.",
    )
    parser.add_argument(
        "--adapter-path",
        default="endogenous_context_theory/adapters/mirage_aware_v1",
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--valid-jsonl",
        default="endogenous_context_theory/training_data/valid.jsonl",
        help="Validation JSONL with embedded oracle metadata.",
    )
    parser.add_argument(
        "--out-csv",
        default="endogenous_context_theory/results/mirage_aware_eval_results.csv",
        help="Path for per-example result CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default="endogenous_context_theory/results/mirage_aware_eval_summary.csv",
        help="Path for summary metrics CSV.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=280,
        help="Generation token budget.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=5,
        help="How many full sample outputs to print.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Could not parse {path}:{line_num}: {exc}") from exc
            rows.append(row)
    return rows


def _extract_prompt(messages: List[Dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    if messages:
        return str(messages[0].get("content", ""))
    return ""


def _extract_pivot(text: str, fallback: List[str]) -> str:
    if not text:
        return ""
    match = PIVOT_REGEX.search(text)
    if match:
        return match.group(1)
    markers = MARKER_REGEX.findall(text)
    if not markers:
        return ""
    for candidate in fallback:
        if candidate in markers:
            return candidate
    return markers[0]


def _normalize_generated(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        for key in ("text", "output", "generated_text", "response"):
            if key in output:
                return str(output[key]).strip()
    if isinstance(output, (list, tuple)):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("text", "output", "generated_text", "response"):
                if key in first:
                    return str(first[key]).strip()
        return str(first).strip()
    return str(output).strip()


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    from mlx_lm import generate

    attempts = [
        {"prompt": prompt, "max_tokens": max_new_tokens, "temp": temperature},
        {"prompt": prompt, "max_new_tokens": max_new_tokens, "temp": temperature},
        {"prompt": prompt, "max_tokens": max_new_tokens, "temperature": temperature},
    ]

    for kwargs in attempts:
        try:
            out = generate(model, tokenizer, **kwargs)
            return _normalize_generated(out)
        except TypeError:
            continue

    # Positional fallback.
    try:
        out = generate(model, tokenizer, prompt, max_tokens=max_new_tokens)
        return _normalize_generated(out)
    except TypeError as exc:
        raise RuntimeError(f"Unsupported mlx_lm.generate signature: {exc}") from exc


def _rate_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _summarize_variant(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    pred_col = f"{prefix}_pred_pivot"
    degraded_flag_col = f"{prefix}_flag_degraded"
    format_col = f"{prefix}_format_ok"

    oracle_degraded = df["evidence_degraded"] == True
    oracle_not_degraded = df["evidence_degraded"] == False
    is_full = df["is_full_context"] == True
    is_compressed = df["is_full_context"] == False

    pred_match = df[pred_col] == df["oracle_pivot"]
    silent_mirage = oracle_degraded & (~pred_match) & (~df[degraded_flag_col])

    return {
        "model_variant": prefix,
        "n": int(len(df)),
        "n_full_context": int(is_full.sum()),
        "n_compressed": int(is_compressed.sum()),
        "pivot_accuracy_all": _rate_bool(pred_match),
        "pivot_accuracy_full_context": _rate_bool(pred_match[is_full]),
        "pivot_accuracy_compressed": _rate_bool(pred_match[is_compressed]),
        "explicit_degradation_flag_rate": _rate_bool(df.loc[oracle_degraded, degraded_flag_col]),
        "false_degradation_flag_rate": _rate_bool(df.loc[oracle_not_degraded, degraded_flag_col]),
        "silent_mirage_rate": _rate_bool(silent_mirage[oracle_degraded]),
        "format_adherence_rate": _rate_bool(df[format_col]),
    }


def main() -> None:
    args = parse_args()

    valid_path = Path(args.valid_jsonl).resolve()
    out_csv = Path(args.out_csv).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(valid_path)
    if not rows:
        raise RuntimeError(f"No rows found in {valid_path}")

    from mlx_lm import load

    print("Loading base model:", args.model)
    base_model, base_tokenizer = load(args.model)
    print("Loading fine-tuned adapter:", args.adapter_path)
    ft_model, ft_tokenizer = load(args.model, adapter_path=args.adapter_path)

    result_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages", [])
        prompt = _extract_prompt(messages)
        if not prompt:
            raise RuntimeError(f"Missing user prompt in row {idx}")

        oracle_pivot = str(row.get("oracle_pivot", ""))
        if not oracle_pivot:
            raise RuntimeError(f"Missing oracle_pivot in valid row {idx}")

        gt_pivot = str(row.get("gt_pivot", ""))
        decoy_pivot = str(row.get("decoy_pivot", ""))
        fallback = [p for p in (oracle_pivot, gt_pivot, decoy_pivot) if p]

        evidence_degraded = bool(row.get("evidence_degraded", False))
        prereq_ratio = float(row.get("prereq_ratio", 0.0))

        base_output = _generate_text(
            model=base_model,
            tokenizer=base_tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        ft_output = _generate_text(
            model=ft_model,
            tokenizer=ft_tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        base_pred = _extract_pivot(base_output, fallback=fallback)
        ft_pred = _extract_pivot(ft_output, fallback=fallback)
        base_degraded = bool(DEGRADED_REGEX.search(base_output))
        ft_degraded = bool(DEGRADED_REGEX.search(ft_output))

        base_format_ok = bool(PIVOT_REGEX.search(base_output)) and bool(EVIDENCE_REGEX.search(base_output))
        ft_format_ok = bool(PIVOT_REGEX.search(ft_output)) and bool(EVIDENCE_REGEX.search(ft_output))

        result_rows.append(
            {
                "row_index": idx,
                "task_id": row.get("task_id", ""),
                "difficulty": row.get("difficulty", ""),
                "is_full_context": bool(row.get("is_full_context", False)),
                "compression_level": float(row.get("compression_level", 0.0)),
                "compression_seed": int(row.get("compression_seed", 0)),
                "oracle_pivot": oracle_pivot,
                "evidence_degraded": evidence_degraded,
                "prereq_ratio": prereq_ratio,
                "base_pred_pivot": base_pred,
                "ft_pred_pivot": ft_pred,
                "base_flag_degraded": base_degraded,
                "ft_flag_degraded": ft_degraded,
                "base_format_ok": base_format_ok,
                "ft_format_ok": ft_format_ok,
                "base_output": base_output,
                "ft_output": ft_output,
            }
        )

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(rows)} eval examples...")

    df = pd.DataFrame(result_rows)
    df.to_csv(out_csv, index=False)

    summary_rows = [
        _summarize_variant(df, "base"),
        _summarize_variant(df, "ft"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved per-example results:", out_csv)
    print("Saved summary:", summary_csv)
    print("\nSummary table:")
    print(summary_df.to_string(index=False))

    sample_n = max(0, int(args.print_samples))
    if sample_n > 0:
        print(f"\n=== Manual sample outputs ({sample_n}) ===")
        for row in result_rows[:sample_n]:
            print(
                f"\n--- task_id={row['task_id']} difficulty={row['difficulty']} "
                f"full={row['is_full_context']} level={row['compression_level']} seed={row['compression_seed']} ---"
            )
            print(
                f"oracle_pivot={row['oracle_pivot']} evidence_degraded={row['evidence_degraded']} "
                f"prereq_ratio={row['prereq_ratio']:.3f}"
            )
            print("\n[Base output]")
            print(row["base_output"])
            print("\n[Fine-tuned output]")
            print(row["ft_output"])


if __name__ == "__main__":
    main()
