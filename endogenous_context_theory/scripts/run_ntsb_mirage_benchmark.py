#!/usr/bin/env python3
"""Run NTSB mirage benchmark with naive vs contract compression.

Backends:
- rule: deterministic heuristic (no API calls)
- openai: Chat Completions API with strict structured prompt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PIVOT_LINE_REGEX = re.compile(r"PIVOT_ID\s*=\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
EVENT_ID_REGEX = re.compile(r"\b(e\d+)\b", re.IGNORECASE)
DEGRADED_STRICT_REGEX = re.compile(r"Evidence assessment:\s*DEGRADED", re.IGNORECASE)


@dataclass
class BackendConfig:
    backend: str
    model: str
    api_key: str
    max_tokens: int
    temperature: float
    timeout_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="endogenous_context_theory/data/ntsb/ntsb_event_graphs.cleaned.json",
        help="Input cleaned NTSB JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="endogenous_context_theory/results/ntsb",
        help="Output directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--backend",
        choices=("rule", "openai"),
        default="rule",
        help="Inference backend.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model id for OpenAI backend.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for OpenAI backend. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--budgets",
        default="0.7,0.5,0.3",
        help="Comma-separated retained fractions.",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
        help="Comma-separated compression seeds.",
    )
    parser.add_argument(
        "--methods",
        default="naive,contract",
        help="Comma-separated methods: naive,contract.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    return parser.parse_args()


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _extract_pivot(text: str) -> str:
    if not text:
        return "UNKNOWN"
    match = PIVOT_LINE_REGEX.search(text)
    if match:
        return match.group(1)
    fallback = EVENT_ID_REGEX.search(text)
    return fallback.group(1) if fallback else "UNKNOWN"


def _extract_flagged_degraded(text: str) -> bool:
    return bool(DEGRADED_STRICT_REGEX.search(text or ""))


def _render_context(incident: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    lines = [
        f"Incident: {incident.get('id')}",
        f"Title: {incident.get('title')}",
        f"Date: {incident.get('date')}",
        f"NTSB ID: {incident.get('ntsb_id')}",
        f"Summary: {incident.get('summary')}",
        "",
        "Event timeline:",
    ]
    for e in sorted(events, key=lambda x: x.get("t", 10**9)):
        lines.append(
            f"- {e.get('id')} | t={e.get('t')} | phase={e.get('phase')} | actor={e.get('actor')} | weight={e.get('weight')}: {e.get('desc')}"
        )
    lines.extend(
        [
            "",
            "Instructions:",
            "1) Pick the pivot event ID for the turning point/root cause in this provided context.",
            "2) Emit strict format:",
            "   PIVOT_ID=<event_id>",
            "   Evidence assessment: STRONG|DEGRADED",
        ]
    )
    return "\n".join(lines)


def _predict_with_rule(incident: Dict[str, Any], kept_events: List[Dict[str, Any]], prereq_ids: List[str], full_pivot: str) -> str:
    kept_ids = {str(e.get("id")) for e in kept_events}
    pivot_present = full_pivot in kept_ids
    prereq_present = [pid for pid in prereq_ids if pid in kept_ids]
    prereq_ratio = len(prereq_present) / len(prereq_ids) if prereq_ids else 1.0

    if kept_events:
        predicted = max(kept_events, key=lambda e: float(e.get("weight", -1e9))).get("id", "UNKNOWN")
    else:
        predicted = "UNKNOWN"

    status = "STRONG" if pivot_present and prereq_ratio >= 1.0 else "DEGRADED"
    confidence = "HIGH" if status == "STRONG" else "LOW"

    lines = [
        f"PIVOT_ID={predicted}",
        f"Evidence assessment: {status}",
        f"Confidence: {confidence}",
        "",
        "Prerequisite events:",
    ]
    for pid in prereq_ids:
        state = "confirmed in context" if pid in kept_ids else "not found in context"
        lines.append(f"- {pid}: {state}")
    return "\n".join(lines)


def _predict_with_openai(prompt: str, cfg: BackendConfig) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
    )
    return str(response.choices[0].message.content or "").strip()


def _build_oracle_compressed(kept_events: List[Dict[str, Any]], full_pivot: str) -> str:
    kept_ids = {str(e.get("id")) for e in kept_events}
    if full_pivot in kept_ids:
        return full_pivot
    if not kept_events:
        return "UNKNOWN"
    return str(max(kept_events, key=lambda e: float(e.get("weight", -1e9))).get("id", "UNKNOWN"))


def _compress_naive(events: List[Dict[str, Any]], budget: float, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    n = len(events)
    keep_n = max(1, int(round(n * budget)))
    keep_n = min(keep_n, n)
    picks = rng.sample(range(n), k=keep_n)
    kept = [events[i] for i in sorted(picks, key=lambda idx: events[idx].get("t", 10**9))]
    return kept


def _compress_contract(
    events: List[Dict[str, Any]],
    budget: float,
    seed: int,
    full_pivot: str,
    prereq_ids: List[str],
    k_recommended: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    n = len(events)
    target_n = max(1, int(round(n * budget)))

    id_to_event = {str(e.get("id")): e for e in events}
    mandatory_ids = {full_pivot}

    prereqs = [pid for pid in prereq_ids if pid in id_to_event]
    if k_recommended > 0 and prereqs:
        if len(prereqs) <= k_recommended:
            mandatory_ids.update(prereqs)
        else:
            chosen = rng.sample(prereqs, k=k_recommended)
            mandatory_ids.update(chosen)

    remaining_ids = [eid for eid in id_to_event if eid not in mandatory_ids]

    if len(mandatory_ids) >= target_n:
        keep_ids = set(mandatory_ids)
    else:
        extra_needed = target_n - len(mandatory_ids)
        extra = rng.sample(remaining_ids, k=min(extra_needed, len(remaining_ids)))
        keep_ids = set(mandatory_ids).union(extra)

    kept = [id_to_event[eid] for eid in keep_ids if eid in id_to_event]
    kept.sort(key=lambda e: e.get("t", 10**9))
    return kept


def _bootstrap_ci(values: Sequence[float], iters: int = 2000) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        v = float(arr[0])
        return (v, v)
    rng = np.random.default_rng(42)
    means = []
    n = len(arr)
    for _ in range(iters):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(sample.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _safe_rate(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return float("nan")
    return float(df[col].astype(float).mean())


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    incidents = payload.get("incidents", [])
    if not isinstance(incidents, list) or not incidents:
        raise ValueError("Input JSON has no incidents")

    budgets = _parse_csv_floats(args.budgets)
    seeds = _parse_csv_ints(args.seeds)
    methods = _parse_csv_strs(args.methods)

    if args.backend == "openai":
        api_key = args.api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OpenAI backend selected but no API key provided")
    else:
        api_key = ""

    backend_cfg = BackendConfig(
        backend=args.backend,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )

    full_predictions: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for incident in incidents:
        incident_id = str(incident.get("id"))
        events = list(incident.get("events", []))
        events.sort(key=lambda e: e.get("t", 10**9))
        if not events:
            continue

        full_pivot = str(incident.get("pivot_ground_truth"))
        k_recommended = int(incident.get("k_recommended", 0))

        pivot_event = next((e for e in events if str(e.get("id")) == full_pivot), None)
        if pivot_event is None:
            raise ValueError(f"Incident {incident_id} missing pivot event {full_pivot}")

        pivot_t = int(pivot_event.get("t"))
        prereq_ids = [
            str(e.get("id"))
            for e in events
            if e.get("phase") == "DEVELOPMENT" and isinstance(e.get("t"), int) and int(e.get("t")) < pivot_t
        ]

        full_prompt = _render_context(incident, events)
        if backend_cfg.backend == "rule":
            full_response = _predict_with_rule(incident, events, prereq_ids, full_pivot)
        else:
            full_response = _predict_with_openai(full_prompt, backend_cfg)

        full_pred_pivot = _extract_pivot(full_response)
        full_flagged = _extract_flagged_degraded(full_response)

        full_predictions[incident_id] = {
            "response": full_response,
            "pivot": full_pred_pivot,
            "flagged_degraded": full_flagged,
            "oracle_full_pivot": full_pivot,
        }

        for method in methods:
            if method not in {"naive", "contract"}:
                raise ValueError(f"Unsupported method: {method}")
            for budget in budgets:
                for seed in seeds:
                    if method == "naive":
                        kept_events = _compress_naive(events, budget=budget, seed=seed)
                    else:
                        kept_events = _compress_contract(
                            events,
                            budget=budget,
                            seed=seed,
                            full_pivot=full_pivot,
                            prereq_ids=prereq_ids,
                            k_recommended=k_recommended,
                        )

                    kept_ids = {str(e.get("id")) for e in kept_events}
                    prereq_present = [pid for pid in prereq_ids if pid in kept_ids]
                    prereq_ratio = len(prereq_present) / len(prereq_ids) if prereq_ids else 1.0
                    oracle_degraded = (full_pivot not in kept_ids) or (prereq_ratio < 1.0)

                    oracle_compressed_pivot = _build_oracle_compressed(kept_events, full_pivot)
                    info_shift = oracle_compressed_pivot != full_pivot

                    compressed_prompt = _render_context(incident, kept_events)
                    if backend_cfg.backend == "rule":
                        response = _predict_with_rule(incident, kept_events, prereq_ids, full_pivot)
                    else:
                        response = _predict_with_openai(compressed_prompt, backend_cfg)
                        time.sleep(0.1)

                    predicted_pivot = _extract_pivot(response)
                    flagged_degraded = _extract_flagged_degraded(response)

                    full_pred = full_predictions[incident_id]["pivot"]
                    full_correct = full_pred == full_pivot
                    compressed_correct = predicted_pivot == oracle_compressed_pivot

                    true_mirage = full_correct and (not compressed_correct)
                    rescue = (not full_correct) and compressed_correct
                    instability = (full_pred != predicted_pivot) and (not true_mirage) and (not rescue)

                    model_error = predicted_pivot != oracle_compressed_pivot
                    silent_mirage = oracle_degraded and model_error and (not flagged_degraded)

                    rows.append(
                        {
                            "incident_id": incident_id,
                            "method": method,
                            "budget": budget,
                            "seed": seed,
                            "n_events_total": len(events),
                            "n_events_kept": len(kept_events),
                            "achieved_retention": len(kept_events) / len(events),
                            "k_recommended": k_recommended,
                            "prereq_count": len(prereq_ids),
                            "prereq_ratio": prereq_ratio,
                            "oracle_full_pivot": full_pivot,
                            "oracle_compressed_pivot": oracle_compressed_pivot,
                            "predicted_full_pivot": full_pred,
                            "predicted_pivot": predicted_pivot,
                            "full_correct": full_correct,
                            "pivot_correct": compressed_correct,
                            "pivot_preserved": predicted_pivot == full_pivot,
                            "oracle_degraded": oracle_degraded,
                            "flagged_degraded": flagged_degraded,
                            "false_alarm": (not oracle_degraded) and flagged_degraded,
                            "true_mirage": true_mirage,
                            "rescue": rescue,
                            "instability": instability,
                            "info_shift": info_shift,
                            "model_error": model_error,
                            "silent_mirage": silent_mirage,
                            "raw_response": response,
                            "raw_response_full": full_predictions[incident_id]["response"],
                            "backend": backend_cfg.backend,
                            "model": backend_cfg.model,
                        }
                    )

    per_example = pd.DataFrame(rows)
    if per_example.empty:
        raise RuntimeError("No benchmark rows generated")

    per_example_path = output_dir / "mirage_results_per_example.csv"
    per_example.to_csv(per_example_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    grouped = per_example.groupby(["method", "budget"], as_index=False)

    for _, group in grouped:
        method = str(group["method"].iloc[0])
        budget = float(group["budget"].iloc[0])

        deg = group[group["oracle_degraded"] == True]  # noqa: E712
        strong = group[group["oracle_degraded"] == False]  # noqa: E712
        deg_wrong = deg[deg["pivot_correct"] == False]  # noqa: E712

        def ci_for(df: pd.DataFrame, col: str) -> Tuple[float, float]:
            vals = df[col].astype(float).tolist()
            return _bootstrap_ci(vals, iters=args.bootstrap_iters)

        pivot_acc = _safe_rate(group, "pivot_correct")
        pivot_lo, pivot_hi = ci_for(group, "pivot_correct")

        silent = _safe_rate(deg, "silent_mirage") if not deg.empty else float("nan")
        silent_lo, silent_hi = ci_for(deg, "silent_mirage") if not deg.empty else (float("nan"), float("nan"))

        flag_deg = _safe_rate(deg, "flagged_degraded") if not deg.empty else float("nan")
        flag_deg_lo, flag_deg_hi = ci_for(deg, "flagged_degraded") if not deg.empty else (float("nan"), float("nan"))

        false_alarm = _safe_rate(strong, "flagged_degraded") if not strong.empty else float("nan")
        fgw = _safe_rate(deg_wrong, "flagged_degraded") if not deg_wrong.empty else float("nan")

        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(group),
                "n_degraded": len(deg),
                "n_strong": len(strong),
                "pivot_accuracy": pivot_acc,
                "pivot_accuracy_ci_low": pivot_lo,
                "pivot_accuracy_ci_high": pivot_hi,
                "pivot_preservation": _safe_rate(group, "pivot_preserved"),
                "true_mirage_rate": _safe_rate(group, "true_mirage"),
                "rescue_rate": _safe_rate(group, "rescue"),
                "instability_rate": _safe_rate(group, "instability"),
                "info_shift_rate": _safe_rate(group, "info_shift"),
                "model_error_rate": _safe_rate(group, "model_error"),
                "silent_mirage_rate": silent,
                "silent_mirage_ci_low": silent_lo,
                "silent_mirage_ci_high": silent_hi,
                "flag_degraded_rate": flag_deg,
                "flag_degraded_ci_low": flag_deg_lo,
                "flag_degraded_ci_high": flag_deg_hi,
                "false_alarm_rate": false_alarm,
                "flag_given_wrong": fgw,
                "mean_prereq_ratio": float(group["prereq_ratio"].mean()),
                "mean_achieved_retention": float(group["achieved_retention"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["method", "budget"], kind="stable")
    summary_path = output_dir / "mirage_results_summary.csv"
    summary.to_csv(summary_path, index=False)

    full_preds_path = output_dir / "mirage_full_predictions.json"
    full_preds_path.write_text(json.dumps(full_predictions, indent=2) + "\n", encoding="utf-8")

    print(f"Per-example results: {per_example_path}")
    print(f"Summary results: {summary_path}")
    print(f"Full-context predictions: {full_preds_path}")
    print("\nSummary preview:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
