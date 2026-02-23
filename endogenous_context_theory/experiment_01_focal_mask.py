#!/usr/bin/env python3
"""Validate focal-mask record-gap theorem against empirical streaming traces.

The theorem predicts that for Bernoulli focal mask probability p and grammar
prefix requirement k, the asymptotic trap rate satisfies:

    P(min_gap < k) -> 1 - exp(-Lambda_k(p))

where:
    Lambda_k(p) = -ln(1-p) + sum_{g=1}^{k-1} [1 - (1-p)^g] / g.

This script loads trace data (JSON/CSV), computes empirical min_gap from
record-setting focal events with strict running-max inequality, and compares
observed trap rates to theorem predictions across epsilon slices.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FOCAL_KEYS = (
    "is_focal",
    "focal",
    "is_protagonist",
    "protagonist",
    "focal_event",
    "is_target",
)
WEIGHT_KEYS = (
    "weight",
    "score",
    "importance",
    "event_weight",
    "composite",
    "sev",
    "value",
    "w",
)
POSITION_KEYS = (
    "timestamp",
    "position",
    "pos",
    "index",
    "step",
    "t",
    "eid",
    "id",
)
EPSILON_KEYS = ("epsilon", "eps", "frontload_epsilon", "front_loading_epsilon")
TRACE_ID_KEYS = ("trace_id", "trace", "id", "run_id", "name")


@dataclass
class CanonicalEvent:
    focal: bool
    weight: float
    position: float


@dataclass
class Trace:
    trace_id: str
    epsilon: float
    events: list[CanonicalEvent]
    source: str
    schema_keys: tuple[str, ...]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _first_present_key(row: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in row:
            return key
    return None


def _infer_epsilon_from_text(text: str) -> float | None:
    lowered = text.lower()
    patterns = (
        r"(?:epsilon|eps)[^0-9]*([0-9]+(?:[._p][0-9]+)?)",
        r"(?:^|[_-])e([0-9]+(?:[._p][0-9]+)?)(?:$|[_-])",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        token = match.group(1).replace("p", ".").replace("_", ".")
        try:
            return float(token)
        except ValueError:
            continue
    return None


def _extract_epsilon(
    payload: dict[str, Any] | None,
    trace_id: str,
    source_name: str,
) -> float | None:
    if payload:
        for key in EPSILON_KEYS:
            if key in payload:
                eps = _coerce_float(payload[key])
                if eps is not None:
                    return eps
    for text in (trace_id, source_name):
        eps = _infer_epsilon_from_text(text)
        if eps is not None:
            return eps
    return None


def _event_to_canonical(
    event: dict[str, Any],
    fallback_position: int,
    trace_meta: dict[str, Any] | None,
) -> CanonicalEvent | None:
    focal: bool | None = None
    focal_key = _first_present_key(event, FOCAL_KEYS)
    if focal_key is not None:
        focal = _coerce_bool(event.get(focal_key))

    if focal is None:
        role = event.get("role")
        if isinstance(role, str):
            role_low = role.lower()
            if role_low in {"focal", "pivot", "protagonist"}:
                focal = True
            if role_low in {"candidate", "decoy", "routine", "nonfocal", "non_focal"}:
                focal = False

    if focal is None and trace_meta:
        focal_actor = trace_meta.get("focal_actor") or trace_meta.get("protagonist")
        if isinstance(focal_actor, str):
            actor = event.get("actor")
            actors = event.get("actors")
            if isinstance(actor, str):
                focal = actor == focal_actor
            elif isinstance(actors, list):
                focal = focal_actor in actors

    if focal is None:
        actor = event.get("actor")
        if isinstance(actor, str):
            if actor == "focal":
                focal = True
            elif actor.startswith("dev_"):
                focal = False

    weight_key = _first_present_key(event, WEIGHT_KEYS)
    weight = _coerce_float(event.get(weight_key)) if weight_key is not None else None
    if weight is None or focal is None:
        return None

    position_key = _first_present_key(event, POSITION_KEYS)
    position = (
        _coerce_float(event.get(position_key)) if position_key is not None else float(fallback_position)
    )
    if position is None:
        position = float(fallback_position)

    return CanonicalEvent(focal=bool(focal), weight=float(weight), position=float(position))


def _build_trace_from_event_rows(
    event_rows: Iterable[dict[str, Any]],
    trace_id: str,
    epsilon: float | None,
    source: str,
    trace_meta: dict[str, Any] | None = None,
) -> Trace | None:
    if epsilon is None:
        return None

    canonical_events: list[tuple[int, CanonicalEvent]] = []
    first_schema: tuple[str, ...] = ()
    for idx, row in enumerate(event_rows):
        if idx == 0:
            first_schema = tuple(sorted(row.keys()))
        event = _event_to_canonical(row, fallback_position=idx, trace_meta=trace_meta)
        if event is not None:
            canonical_events.append((idx, event))

    if not canonical_events:
        return None

    canonical_events.sort(key=lambda pair: (pair[1].position, pair[0]))
    ordered_events = [event for _, event in canonical_events]
    return Trace(
        trace_id=trace_id,
        epsilon=float(epsilon),
        events=ordered_events,
        source=source,
        schema_keys=first_schema,
    )


def _load_test09_metadata(path: Path) -> list[Trace]:
    from src.generators import bursty_generator

    header = pd.read_csv(path, nrows=0)
    required = {"n", "n_focal", "epsilon", "seed"}
    if not required.issubset(set(header.columns)):
        return []

    df = pd.read_csv(path)
    traces: list[Trace] = []
    for row in df.itertuples(index=False):
        n = int(getattr(row, "n"))
        n_focal = int(getattr(row, "n_focal"))
        epsilon = float(getattr(row, "epsilon"))
        seed = int(getattr(row, "seed"))
        events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
        traces.append(
            Trace(
                trace_id=f"test09_eps_{epsilon:.3f}_n_{n}_seed_{seed}",
                epsilon=epsilon,
                events=[
                    CanonicalEvent(
                        focal=bool(event.is_focal),
                        weight=float(event.weight),
                        position=float(event.timestamp),
                    )
                    for event in events
                ],
                source=str(path),
                schema_keys=("is_focal", "weight", "timestamp"),
            )
        )
    return traces


def _load_test16_metadata(path: Path) -> list[Trace]:
    from src.generators import bursty_generator

    header = pd.read_csv(path, nrows=0)
    required = {"n", "epsilon", "trials"}
    if not required.issubset(set(header.columns)):
        return []

    df = pd.read_csv(path)
    if "k" in df.columns:
        df = df.drop_duplicates(subset=["n", "epsilon", "trials"]).copy()
    else:
        df = df.drop_duplicates(subset=["n", "epsilon"]).copy()

    traces: list[Trace] = []
    for row in df.itertuples(index=False):
        n = int(getattr(row, "n"))
        epsilon = float(getattr(row, "epsilon"))
        trials = int(getattr(row, "trials"))
        n_focal = max(10, n // 2)
        for seed in range(trials):
            events = bursty_generator(n=n, n_focal=n_focal, epsilon=epsilon, seed=seed)
            traces.append(
                Trace(
                    trace_id=f"test16_eps_{epsilon:.3f}_n_{n}_seed_{seed}",
                    epsilon=epsilon,
                    events=[
                        CanonicalEvent(
                            focal=bool(event.is_focal),
                            weight=float(event.weight),
                            position=float(event.timestamp),
                        )
                        for event in events
                    ],
                    source=str(path),
                    schema_keys=("is_focal", "weight", "timestamp"),
                )
            )
    return traces


def _load_rhun_organic_metadata(path: Path) -> list[Trace]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        return []
    if not isinstance(payload[0], dict):
        return []
    if not {"epsilon", "seed"}.issubset(set(payload[0].keys())):
        return []
    if "events" in payload[0]:
        return []

    repo_root = Path(__file__).resolve().parents[1]
    rhun_path = repo_root / "projects" / "rhun"
    if str(rhun_path) not in sys.path:
        sys.path.insert(0, str(rhun_path))

    from rhun.generators.bursty import BurstyConfig, BurstyGenerator

    generator = BurstyGenerator()
    seen: set[tuple[float, int]] = set()
    traces: list[Trace] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        eps_raw = _coerce_float(row.get("epsilon"))
        seed_raw = _coerce_float(row.get("seed"))
        if eps_raw is None or seed_raw is None:
            continue
        epsilon = float(eps_raw)
        seed = int(seed_raw)
        key = (epsilon, seed)
        if key in seen:
            continue
        seen.add(key)

        graph = generator.generate(
            BurstyConfig(
                n_events=int(row.get("n_events", 200)),
                n_actors=int(row.get("n_actors", 6)),
                seed=seed,
                epsilon=epsilon,
            )
        )
        events = sorted(graph.events, key=lambda event: (float(event.timestamp), str(event.id)))
        traces.append(
            Trace(
                trace_id=f"rhun_eps_{epsilon:.3f}_seed_{seed}",
                epsilon=epsilon,
                events=[
                    CanonicalEvent(
                        focal=("actor_0" in event.actors),
                        weight=float(event.weight),
                        position=float(event.timestamp),
                    )
                    for event in events
                ],
                source=str(path),
                schema_keys=("actors", "weight", "timestamp"),
            )
        )
    return traces


def _load_json_generic(path: Path) -> list[Trace]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    traces: list[Trace] = []

    def append_trace(item: dict[str, Any], fallback_id: str) -> None:
        events = item.get("events")
        if not isinstance(events, list):
            return
        trace_id = fallback_id
        for key in TRACE_ID_KEYS:
            if key in item and item[key] is not None:
                trace_id = str(item[key])
                break
        eps = _extract_epsilon(item, trace_id=trace_id, source_name=path.name)
        trace = _build_trace_from_event_rows(
            event_rows=(row for row in events if isinstance(row, dict)),
            trace_id=trace_id,
            epsilon=eps,
            source=str(path),
            trace_meta=item,
        )
        if trace is not None:
            traces.append(trace)

    if isinstance(payload, dict):
        if isinstance(payload.get("traces"), list):
            for idx, item in enumerate(payload["traces"]):
                if isinstance(item, dict):
                    append_trace(item, fallback_id=f"{path.stem}_trace_{idx}")
        elif isinstance(payload.get("events"), list):
            append_trace(payload, fallback_id=path.stem)
    elif isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and isinstance(payload[0].get("events"), list):
            for idx, item in enumerate(payload):
                if isinstance(item, dict):
                    append_trace(item, fallback_id=f"{path.stem}_trace_{idx}")
        elif payload and isinstance(payload[0], dict):
            eps = _extract_epsilon(None, trace_id=path.stem, source_name=path.name)
            trace = _build_trace_from_event_rows(
                event_rows=(row for row in payload if isinstance(row, dict)),
                trace_id=path.stem,
                epsilon=eps,
                source=str(path),
                trace_meta=None,
            )
            if trace is not None:
                traces.append(trace)
    return traces


def _find_column(columns: Sequence[str], keys: Sequence[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _load_csv_generic(path: Path) -> list[Trace]:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    if not columns:
        return []

    if {"n", "n_focal", "epsilon", "seed"}.issubset(set(columns)):
        return _load_test09_metadata(path)
    if {"n", "epsilon", "trials"}.issubset(set(columns)):
        return _load_test16_metadata(path)

    focal_col = _find_column(columns, FOCAL_KEYS)
    weight_col = _find_column(columns, WEIGHT_KEYS)
    if focal_col is None or weight_col is None:
        return []

    position_col = _find_column(columns, POSITION_KEYS)
    eps_col = _find_column(columns, EPSILON_KEYS)
    trace_id_col = _find_column(columns, TRACE_ID_KEYS)
    focal_actor_col = _find_column(columns, ("focal_actor", "protagonist", "target_actor"))

    df = pd.read_csv(path)
    if trace_id_col is None:
        grouped = [(path.stem, df)]
    else:
        grouped = list(df.groupby(trace_id_col, sort=False, dropna=False))

    traces: list[Trace] = []
    for group_key, group_df in grouped:
        group_records = group_df.to_dict(orient="records")
        trace_id = str(group_key)
        group_meta: dict[str, Any] = {}
        if focal_actor_col is not None and len(group_df) > 0:
            group_meta["focal_actor"] = group_df.iloc[0][focal_actor_col]

        epsilon: float | None = None
        if eps_col is not None and len(group_df) > 0:
            epsilon = _coerce_float(group_df.iloc[0][eps_col])
        if epsilon is None:
            epsilon = _extract_epsilon(group_meta, trace_id=trace_id, source_name=path.name)

        event_rows: list[dict[str, Any]] = []
        for idx, rec in enumerate(group_records):
            row = dict(rec)
            if position_col is None:
                row["index"] = idx
            event_rows.append(row)

        trace = _build_trace_from_event_rows(
            event_rows=event_rows,
            trace_id=trace_id,
            epsilon=epsilon,
            source=str(path),
            trace_meta=group_meta,
        )
        if trace is not None:
            traces.append(trace)
    return traces


def load_traces(data_dir: Path) -> list[Trace]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    traces: list[Trace] = []
    if data_dir.is_file():
        candidates = [data_dir]
    else:
        candidates = sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".csv"}
        )
    for path in candidates:
        try:
            if path.name == "organic_oscillation_raw.json":
                traces.extend(_load_rhun_organic_metadata(path))
                continue
            if path.name == "test_09_record_process_raw.csv":
                traces.extend(_load_test09_metadata(path))
                continue
            if path.name == "test_16_organic_traps.csv":
                traces.extend(_load_test16_metadata(path))
                continue

            if path.suffix.lower() == ".json":
                traces.extend(_load_json_generic(path))
            else:
                traces.extend(_load_csv_generic(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping {path} due to parse error: {exc}")

    deduped: dict[tuple[str, float], Trace] = {}
    for trace in traces:
        deduped[(trace.trace_id, trace.epsilon)] = trace
    return list(deduped.values())


def lambda_k(p: float, k: int) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    base = -math.log1p(-p)
    correction = sum((1.0 - (1.0 - p) ** g) / g for g in range(1, k))
    return float(base + correction)


def predicted_trap_rate(p: float, k: int) -> float:
    lam = lambda_k(p, k)
    if math.isinf(lam):
        return 1.0
    return float(1.0 - math.exp(-lam))


def compute_record_stats(events: Sequence[CanonicalEvent]) -> tuple[int, float]:
    running_max = float("-inf")
    record_positions: list[int] = []

    for idx, event in enumerate(events):
        if event.focal and event.weight > running_max:
            running_max = event.weight
            record_positions.append(idx)

    if len(record_positions) < 2:
        return len(record_positions), float("inf")

    gaps: list[int] = []
    for start, end in zip(record_positions[:-1], record_positions[1:]):
        gap = sum(1 for event in events[start + 1 : end] if not event.focal)
        gaps.append(gap)
    min_gap = float(min(gaps)) if gaps else float("inf")
    return len(record_positions), min_gap


def summarize_loaded_traces(
    traces: Sequence[Trace],
    trace_stats: pd.DataFrame,
    min_records: int,
) -> None:
    if not traces:
        print("No traces loaded.")
        return

    first = traces[0]
    lengths = trace_stats["n_events"].to_numpy(dtype=int)
    record_counts = trace_stats["n_records"].to_numpy(dtype=int)
    eps_values = sorted({trace.epsilon for trace in traces})
    low_record_mask = trace_stats["n_records"] < int(min_records)
    low_record_count = int(low_record_mask.sum())

    print("=== TRACE LOAD SUMMARY ===")
    print(f"Traces loaded: {len(traces)}")
    print(f"Epsilon values: {', '.join(f'{eps:.3f}' for eps in eps_values)}")
    print(
        "Events per trace: "
        f"min={int(lengths.min())}, p25={np.percentile(lengths, 25):.1f}, "
        f"median={np.median(lengths):.1f}, p75={np.percentile(lengths, 75):.1f}, "
        f"max={int(lengths.max())}, mean={lengths.mean():.2f}"
    )
    if len(set(lengths.tolist())) > 1:
        print("Trace lengths vary across the dataset.")
    else:
        print(f"Trace length is constant at {int(lengths[0])} events.")
    print(
        "Record-setting focal events per trace: "
        f"min={int(record_counts.min())}, p25={np.percentile(record_counts, 25):.1f}, "
        f"median={np.median(record_counts):.1f}, p75={np.percentile(record_counts, 75):.1f}, "
        f"max={int(record_counts.max())}, mean={record_counts.mean():.2f}"
    )
    print(
        f"Traces with fewer than {min_records} records (flagged for min_gap reliability): "
        f"{low_record_count}/{len(traces)}"
    )
    if low_record_count > 0:
        low_by_eps = (
            trace_stats.loc[low_record_mask]
            .groupby("epsilon")
            .size()
            .sort_index()
            .to_dict()
        )
        low_by_eps_str = ", ".join(f"{eps:.3f}:{count}" for eps, count in low_by_eps.items())
        print(f"Low-record traces by epsilon: {low_by_eps_str}")

    print("\n=== FIRST TRACE SCHEMA ===")
    print(f"trace_id: {first.trace_id}")
    print(f"source: {first.source}")
    print(f"epsilon: {first.epsilon:.3f}")
    print(f"raw event keys: {', '.join(first.schema_keys) if first.schema_keys else '(unknown)'}")
    print("normalized event fields: focal(bool), weight(float), position(float)")


def compute_metrics(
    trace_stats: pd.DataFrame,
    ks: Sequence[int],
) -> tuple[pd.DataFrame, dict[float, float]]:
    rows: list[dict[str, Any]] = []
    p_hat_per_epsilon: dict[float, float] = {}

    for epsilon, group in trace_stats.groupby("epsilon", sort=True):
        total_events = int(group["n_events"].sum())
        total_focal = int(group["n_focal"].sum())
        p_hat = (total_focal / total_events) if total_events > 0 else 0.0
        p_hat_per_epsilon[float(epsilon)] = p_hat

        per_trace_p = group["p_trace"].to_numpy(dtype=float)
        p_trace_mean = float(np.mean(per_trace_p)) if per_trace_p.size > 0 else float("nan")
        p_trace_std = float(np.std(per_trace_p, ddof=0)) if per_trace_p.size > 0 else float("nan")

        min_gaps = group["min_gap"].to_numpy(dtype=float)

        for k in ks:
            lam = lambda_k(p_hat, k)
            pred = predicted_trap_rate(p_hat, k)
            observed = (
                float(np.mean([gap < float(k) for gap in min_gaps]))
                if min_gaps.size > 0
                else float("nan")
            )
            residual = observed - pred if (not np.isnan(observed)) else float("nan")
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "k": k,
                    "p_hat": p_hat,
                    "p_trace_mean": p_trace_mean,
                    "p_trace_std": p_trace_std,
                    "lambda_k": lam,
                    "predicted_trap_rate": pred,
                    "observed_trap_rate": observed,
                    "residual": residual,
                    "n_traces": int(len(group)),
                }
            )
    comparison = pd.DataFrame(rows).sort_values(["epsilon", "k"]).reset_index(drop=True)
    return comparison, p_hat_per_epsilon


def _plot_pred_vs_obs(df: pd.DataFrame, output_dir: Path, ks: Sequence[int]) -> None:
    for k in ks:
        sub = df[df["k"] == k].sort_values("epsilon")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sub["predicted_trap_rate"], sub["observed_trap_rate"], s=55, color="#1f77b4")
        for row in sub.itertuples(index=False):
            ax.annotate(f"eps={row.epsilon:.2f}", (row.predicted_trap_rate, row.observed_trap_rate), fontsize=8)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted Trap Rate")
        ax.set_ylabel("Observed Trap Rate")
        ax.set_title(f"Focal-Mask Theorem: Predicted vs Observed Trap Rate (k={k})")
        fig.tight_layout()
        fig.savefig(output_dir / f"experiment_01_predicted_vs_observed_k{k}.png", dpi=180)
        plt.close(fig)


def _plot_residual_vs_epsilon(df: pd.DataFrame, output_dir: Path, ks: Sequence[int]) -> None:
    for k in ks:
        sub = df[df["k"] == k].sort_values("epsilon")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(sub["epsilon"], sub["residual"], marker="o", color="#d62728", linewidth=1.2)
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Residual (observed - predicted)")
        ax.set_title(f"Burstiness Deformation: Residual vs epsilon (k={k})")
        fig.tight_layout()
        fig.savefig(output_dir / f"experiment_01_residual_vs_epsilon_k{k}.png", dpi=180)
        plt.close(fig)


def _plot_heatmaps(df: pd.DataFrame, output_dir: Path, ks: Sequence[int]) -> None:
    eps_values = sorted(df["epsilon"].unique().tolist())
    pred = (
        df.pivot(index="epsilon", columns="k", values="predicted_trap_rate")
        .reindex(index=eps_values, columns=list(ks))
        .to_numpy(dtype=float)
    )
    obs = (
        df.pivot(index="epsilon", columns="k", values="observed_trap_rate")
        .reindex(index=eps_values, columns=list(ks))
        .to_numpy(dtype=float)
    )

    finite_values = np.concatenate([pred[np.isfinite(pred)], obs[np.isfinite(obs)]])
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True)
    im0 = axes[0].imshow(pred, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Predicted Trap Rates")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("epsilon")
    axes[0].set_xticks(range(len(ks)))
    axes[0].set_xticklabels([str(k) for k in ks])
    axes[0].set_yticks(range(len(eps_values)))
    axes[0].set_yticklabels([f"{eps:.2f}" for eps in eps_values])

    axes[1].imshow(obs, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Observed Trap Rates")
    axes[1].set_xlabel("k")
    axes[1].set_xticks(range(len(ks)))
    axes[1].set_xticklabels([str(k) for k in ks])

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Trap Rate")
    fig.savefig(output_dir / "experiment_01_heatmap_predicted_vs_observed.png", dpi=180)
    plt.close(fig)


def _interpretation_paragraph(
    mae: float,
    burstiness_gap_per_k: dict[int, float],
    spearman_per_k: dict[int, float],
) -> str:
    if mae < 0.05:
        fit_text = "very close"
    elif mae < 0.12:
        fit_text = "reasonably close"
    else:
        fit_text = "materially different"

    mean_gap = float(np.mean(list(burstiness_gap_per_k.values()))) if burstiness_gap_per_k else float("nan")
    if np.isnan(mean_gap):
        deformation_text = "insufficient"
    elif mean_gap > 0:
        deformation_text = "positive"
    elif mean_gap < 0:
        deformation_text = "negative"
    else:
        deformation_text = "near-zero"

    rho_bits = ", ".join(
        f"k={k}: {rho:.3f}" if np.isfinite(rho) else f"k={k}: nan"
        for k, rho in sorted(spearman_per_k.items())
    )
    return (
        f"The theorem baseline is {fit_text} to empirical trap rates (MAE={mae:.4f}); "
        f"the mean burstiness gap is {mean_gap:.4f}, indicating a {deformation_text} deformation "
        f"relative to the i.i.d. focal-mask prediction, and Spearman residual-vs-epsilon trends are "
        f"[{rho_bits}], which helps quantify whether burst timing systematically perturbs trap risk as posed "
        f"by the Nevzorov deformation open problem."
    )


def _json_safe(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def run(data_dir: Path, output_dir: Path, ks: Sequence[int], min_records: int) -> None:
    traces = load_traces(data_dir)
    if not traces:
        raise RuntimeError(
            f"No usable traces found in {data_dir}. "
            "Provide JSON/CSV traces or metadata such as test_09_record_process_raw.csv."
        )

    trace_rows: list[dict[str, Any]] = []
    for idx, trace in enumerate(traces):
        n_events = len(trace.events)
        n_focal = sum(1 for event in trace.events if event.focal)
        p_trace = (n_focal / n_events) if n_events > 0 else float("nan")
        n_records, min_gap = compute_record_stats(trace.events)
        trace_rows.append(
            {
                "trace_idx": idx,
                "trace_id": trace.trace_id,
                "epsilon": trace.epsilon,
                "n_events": n_events,
                "n_focal": n_focal,
                "p_trace": p_trace,
                "n_records": n_records,
                "min_gap": min_gap,
            }
        )
    trace_stats = pd.DataFrame(trace_rows)

    summarize_loaded_traces(traces, trace_stats=trace_stats, min_records=min_records)

    eligible = trace_stats[trace_stats["n_records"] >= int(min_records)].copy()
    excluded = trace_stats[trace_stats["n_records"] < int(min_records)].copy()
    print(
        f"\nEffective sample size after excluding traces with n_records < {min_records}: "
        f"{len(eligible)}/{len(trace_stats)}"
    )
    if len(eligible) == 0:
        raise RuntimeError(
            f"All traces were excluded by min_records={min_records}. "
            "Lower --min_records or provide richer traces."
        )

    comparison, p_hat_per_epsilon = compute_metrics(eligible, ks=ks)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "experiment_01_focal_mask_validation.csv"
    summary_path = output_dir / "experiment_01_focal_mask_summary.json"
    comparison.to_csv(table_path, index=False)

    print("\n=== COMPARISON TABLE ===")
    print(
        comparison[
            [
                "epsilon",
                "k",
                "p_hat",
                "lambda_k",
                "predicted_trap_rate",
                "observed_trap_rate",
                "residual",
                "n_traces",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    mae = float(np.mean(np.abs(comparison["residual"])))
    spearman_rho_per_k: dict[int, float] = {}
    spearman_p_per_k: dict[int, float] = {}
    for k in ks:
        sub = comparison[comparison["k"] == k].sort_values("epsilon")
        if len(sub) >= 2:
            eps_vals = sub["epsilon"].to_numpy(dtype=float)
            residual_vals = sub["residual"].to_numpy(dtype=float)
            if np.allclose(eps_vals, eps_vals[0]) or np.allclose(residual_vals, residual_vals[0]):
                spearman_rho_per_k[int(k)] = float("nan")
                spearman_p_per_k[int(k)] = float("nan")
            else:
                rho, pval = spearmanr(eps_vals, residual_vals)
                spearman_rho_per_k[int(k)] = float(rho) if np.isfinite(rho) else float("nan")
                spearman_p_per_k[int(k)] = float(pval) if np.isfinite(pval) else float("nan")
        else:
            spearman_rho_per_k[int(k)] = float("nan")
            spearman_p_per_k[int(k)] = float("nan")

    burstiness_gap_per_k = {
        int(k): float(comparison[comparison["k"] == k]["residual"].mean())
        for k in ks
    }

    n_eps = comparison["epsilon"].nunique()
    if n_eps < 3:
        print(
            "\n[NOTE] Fewer than 3 epsilon values were found; epsilon-monotonicity claims are limited. "
            "Comparison still runs across k."
        )

    print(f"\nMean absolute error across cells: {mae:.6f}")
    print("Spearman rho(residual, epsilon) per k:")
    for k in ks:
        rho = spearman_rho_per_k[int(k)]
        pval = spearman_p_per_k[int(k)]
        rho_text = f"{rho:.6f}" if np.isfinite(rho) else "nan"
        pval_text = f"{pval:.6f}" if np.isfinite(pval) else "nan"
        print(f"  k={k}: rho={rho_text}, p={pval_text}")
    print("Burstiness gap per k (mean residual):")
    for k in ks:
        print(f"  k={k}: {burstiness_gap_per_k[int(k)]:.6f}")

    _plot_pred_vs_obs(comparison, output_dir=output_dir, ks=ks)
    _plot_residual_vs_epsilon(comparison, output_dir=output_dir, ks=ks)
    _plot_heatmaps(comparison, output_dir=output_dir, ks=ks)

    summary_payload = {
        "mean_absolute_error": mae,
        "spearman_rho_per_k": {str(k): _json_safe(v) for k, v in spearman_rho_per_k.items()},
        "spearman_pvalue_per_k": {str(k): _json_safe(v) for k, v in spearman_p_per_k.items()},
        "burstiness_gap_per_k": {str(k): _json_safe(v) for k, v in burstiness_gap_per_k.items()},
        "p_hat_per_epsilon": {f"{eps:.6f}": p for eps, p in sorted(p_hat_per_epsilon.items())},
        "n_traces_total": int(len(trace_stats)),
        "n_traces_included": int(len(eligible)),
        "n_traces_excluded_low_records": int(len(excluded)),
        "min_records_threshold": int(min_records),
        "n_epsilon": int(n_eps),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8")

    interpretation = _interpretation_paragraph(
        mae=mae,
        burstiness_gap_per_k=burstiness_gap_per_k,
        spearman_per_k=spearman_rho_per_k,
    )
    print("\n=== INTERPRETATION ===")
    print(interpretation)
    print(f"\nSaved comparison table: {table_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved plots under: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate focal-mask record-gap theorem against empirical traces."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing trace JSON/CSV files (or metadata files to reconstruct traces).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory where comparison CSV/JSON/PNG artifacts will be saved.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Grammar prefix requirements k to evaluate (default: 1 2 3).",
    )
    parser.add_argument(
        "--min_records",
        type=int,
        default=3,
        help="Exclude traces with fewer than this many record-setting focal events (default: 3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(data_dir=args.data_dir, output_dir=args.output_dir, ks=args.k_values, min_records=args.min_records)
