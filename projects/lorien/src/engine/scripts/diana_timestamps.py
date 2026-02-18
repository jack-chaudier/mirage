"""Temporal distribution diagnostic for Diana's strict-invalid arcs.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.diana_timestamps
"""

from __future__ import annotations

import argparse
import json
from statistics import median
from pathlib import Path
from typing import Any

from narrativefield.extraction.rashomon import extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.k_sweep_experiment import _generate_canon_after_b_for_seed, _simulate_story
from scripts.test_goal_evolution import _evolution_profiles

TARGET_AGENTS = ["diana", "thorne", "marcus"]
BEAT_TYPES = ["setup", "complication", "escalation", "turning_point", "consequence"]
SEGMENTS = ["early", "mid", "late"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "diana_timestamps.json"
EPSILON = 1e-9


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _parse_seeds(raw: str) -> list[int]:
    text = raw.strip()
    if "," in text:
        out = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not out:
            raise ValueError("No valid seeds parsed from comma list.")
        return sorted(set(out))
    if "-" in text:
        left, right = text.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            raise ValueError("Seed range must satisfy end >= start.")
        return list(range(start, end + 1))
    return [int(text)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _segment(global_pos: float) -> str:
    if global_pos < 0.25:
        return "early"
    if global_pos <= 0.70:
        return "mid"
    return "late"


def _arc_event_rows(arc: Any, simulation_end_tick: int) -> list[dict[str, Any]]:
    paired = list(zip(arc.events, arc.beats))
    paired.sort(key=lambda pair: (int(pair[0].tick_id), int(pair[0].order_in_tick)))
    if not paired:
        return []

    start_tick = int(paired[0][0].tick_id)
    end_tick = int(paired[-1][0].tick_id)
    arc_span = max(end_tick - start_tick, 1)
    sim_den = max(int(simulation_end_tick), 1)

    out: list[dict[str, Any]] = []
    for event, beat in paired:
        tick = int(event.tick_id)
        global_pos = float(tick / sim_den)
        local_pos = float((tick - start_tick) / arc_span)
        out.append(
            {
                "event_id": str(event.id),
                "tick_id": tick,
                "order_in_tick": int(event.order_in_tick),
                "global_pos": global_pos,
                "arc_local_pos": local_pos,
                "beat": str(beat.value),
                "segment": _segment(global_pos),
            }
        )
    return out


def _event_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    globals_ = [float(row["global_pos"]) for row in rows]
    locals_ = [float(row["arc_local_pos"]) for row in rows]
    segment_counts = {name: 0 for name in SEGMENTS}
    for row in rows:
        segment_counts[str(row["segment"])] += 1
    total = len(rows)
    segment_shares = {
        name: (float(segment_counts[name] / total) if total else 0.0)
        for name in SEGMENTS
    }
    return {
        "n_events": int(total),
        "mean_global_pos": float(_mean(globals_)),
        "median_global_pos": float(median(globals_)) if globals_ else 0.0,
        "mean_arc_local_pos": float(_mean(locals_)),
        "median_arc_local_pos": float(median(locals_)) if locals_ else 0.0,
        "segment_counts": segment_counts,
        "segment_shares": segment_shares,
    }


def _global_histogram(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = [0] * 10
    for row in rows:
        value = float(row["global_pos"])
        clamped = min(max(value, 0.0), 1.0)
        index = min(int(clamped * 10.0), 9)
        counts[index] += 1
    return [
        {
            "bin": f"{i / 10.0:.1f}-{(i + 1) / 10.0:.1f}",
            "count": int(counts[i]),
        }
        for i in range(10)
    ]


def _beats_by_segment(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out = {seg: {beat: 0 for beat in BEAT_TYPES} for seg in SEGMENTS}
    for row in rows:
        seg = str(row["segment"])
        beat = str(row["beat"])
        if seg in out and beat in out[seg]:
            out[seg][beat] += 1
    return out


def _segment_share(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    counts = {name: 0 for name in SEGMENTS}
    for row in rows:
        counts[str(row["segment"])] += 1
    return {name: (float(counts[name] / total) if total else 0.0) for name in SEGMENTS}


def _histogram_line(bin_label: str, count: int, max_count: int) -> str:
    if max_count <= 0:
        bar = ""
    else:
        width = int(round((count / max_count) * 18))
        if count > 0:
            width = max(width, 1)
        bar = "█" * width
    return f"  {bin_label}: {bar:<18} {count} events"


def _verdict(late_share_invalid: float) -> str:
    if late_share_invalid > 0.60:
        return "End-loaded: search is grabbing late-simulation events. Q-metric kinetic bias confirmed."
    if late_share_invalid < 0.40:
        return "Spread: classifier is mislabeling mid-arc events. Beat classifier position heuristic is the issue."
    return "Moderate clustering: partial end-loading with some mid-arc presence."


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _print_summary(
    *,
    diana_valid_arc_count: int,
    diana_invalid_arc_count: int,
    diana_valid_stats: dict[str, Any],
    diana_invalid_stats: dict[str, Any],
    invalid_hist: list[dict[str, Any]],
    invalid_beats_by_segment: dict[str, dict[str, int]],
    comparison_shares: dict[str, dict[str, float]],
    verdict: str,
) -> None:
    print()
    print("=== DIANA ARC EVENT TEMPORAL DISTRIBUTION (full evolution, 50 seeds) ===")
    print()
    print(f"Diana valid arcs (N={diana_valid_arc_count}):")
    valid_share = diana_valid_stats["segment_shares"]
    print(
        "  Global position:  "
        f"early(<0.25): {_fmt_pct(valid_share['early'])}   "
        f"mid(0.25-0.70): {_fmt_pct(valid_share['mid'])}   "
        f"late(>0.70): {_fmt_pct(valid_share['late'])}"
    )
    print(
        "  Mean global pos: "
        f"{float(diana_valid_stats['mean_global_pos']):.2f}   "
        f"Median: {float(diana_valid_stats['median_global_pos']):.2f}"
    )
    print()
    print(f"Diana invalid arcs (N={diana_invalid_arc_count}):")
    invalid_share = diana_invalid_stats["segment_shares"]
    print(
        "  Global position:  "
        f"early(<0.25): {_fmt_pct(invalid_share['early'])}   "
        f"mid(0.25-0.70): {_fmt_pct(invalid_share['mid'])}   "
        f"late(>0.70): {_fmt_pct(invalid_share['late'])}"
    )
    print(
        "  Mean global pos: "
        f"{float(diana_invalid_stats['mean_global_pos']):.2f}   "
        f"Median: {float(diana_invalid_stats['median_global_pos']):.2f}"
    )
    print()
    print("  Global position histogram (invalid arcs):")
    max_count = max((int(row["count"]) for row in invalid_hist), default=0)
    for row in invalid_hist:
        print(_histogram_line(str(row["bin"]), int(row["count"]), max_count))
    print()
    print("  Beat labels by global position (invalid arcs):")
    for seg in SEGMENTS:
        beat_counts = invalid_beats_by_segment[seg]
        print(
            f"  {seg}({ '<0.25' if seg == 'early' else '0.25-0.70' if seg == 'mid' else '>0.70' }):    "
            f"SETUP: {beat_counts['setup']}  "
            f"COMPLICATION: {beat_counts['complication']}  "
            f"ESCALATION: {beat_counts['escalation']}  "
            f"TURNING_POINT: {beat_counts['turning_point']}  "
            f"CONSEQUENCE: {beat_counts['consequence']}"
        )
    print()
    print("=== COMPARISON: EARLY/MID/LATE EVENT SHARE ===")
    print("                  early(<0.25)    mid(0.25-0.70)    late(>0.70)")
    print(
        f"Thorne:           {_fmt_pct(comparison_shares['thorne']['early']):<15}"
        f"{_fmt_pct(comparison_shares['thorne']['mid']):<18}"
        f"{_fmt_pct(comparison_shares['thorne']['late'])}"
    )
    print(
        f"Diana (valid):    {_fmt_pct(comparison_shares['diana_valid']['early']):<15}"
        f"{_fmt_pct(comparison_shares['diana_valid']['mid']):<18}"
        f"{_fmt_pct(comparison_shares['diana_valid']['late'])}"
    )
    print(
        f"Diana (invalid):  {_fmt_pct(comparison_shares['diana_invalid']['early']):<15}"
        f"{_fmt_pct(comparison_shares['diana_invalid']['mid']):<18}"
        f"{_fmt_pct(comparison_shares['diana_invalid']['late'])}"
    )
    print(
        f"Marcus:           {_fmt_pct(comparison_shares['marcus']['early']):<15}"
        f"{_fmt_pct(comparison_shares['marcus']['mid']):<18}"
        f"{_fmt_pct(comparison_shares['marcus']['late'])}"
    )
    print()
    print("=== VERDICT ===")
    print(verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal distribution diagnostic for Diana arcs.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    seeds = _parse_seeds(args.seeds)
    full_evolutions = _evolution_profiles()["full"]

    diana_valid_event_rows: list[dict[str, Any]] = []
    diana_invalid_event_rows: list[dict[str, Any]] = []
    thorne_event_rows: list[dict[str, Any]] = []
    marcus_event_rows: list[dict[str, Any]] = []

    diana_valid_arc_count = 0
    diana_invalid_arc_count = 0

    per_seed: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        print(f"[{i:03d}/{len(seeds):03d}] seed={seed} condition=full_evolution", flush=True)
        canon_after_b = _generate_canon_after_b_for_seed(
            seed=seed,
            event_limit=int(args.event_limit),
            tick_limit=int(args.tick_limit),
        )

        story = _simulate_story(
            label="full_evolution",
            seed=seed,
            loaded_canon=canon_after_b,
            tick_limit=int(args.tick_limit),
            event_limit=int(args.event_limit),
            evolutions=full_evolutions,
        )

        parsed = parse_simulation_output(story.payload)
        metrics_output = run_metrics_pipeline(parsed)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
        simulation_end_tick = max((int(event.tick_id) for event in metrics_output.events), default=1)

        rashomon = extract_rashomon_set(
            events=metrics_output.events,
            seed=seed,
            agents=list(TARGET_AGENTS),
            total_sim_time=total_sim_time,
        )
        arc_by_agent = {str(arc.protagonist): arc for arc in rashomon.arcs}

        seed_row: dict[str, Any] = {"seed": int(seed), "simulation_end_tick": int(simulation_end_tick)}
        for agent in TARGET_AGENTS:
            arc = arc_by_agent[agent]
            event_rows = _arc_event_rows(arc, simulation_end_tick)
            row = {
                "valid": bool(arc.valid),
                "event_count": int(len(event_rows)),
                "event_rows": event_rows,
            }
            seed_row[agent] = row

            if agent == "diana":
                if bool(arc.valid):
                    diana_valid_arc_count += 1
                    diana_valid_event_rows.extend(event_rows)
                else:
                    diana_invalid_arc_count += 1
                    diana_invalid_event_rows.extend(event_rows)
            elif agent == "thorne":
                thorne_event_rows.extend(event_rows)
            elif agent == "marcus":
                marcus_event_rows.extend(event_rows)
        per_seed.append(seed_row)

    diana_valid_stats = _event_stats(diana_valid_event_rows)
    diana_invalid_stats = _event_stats(diana_invalid_event_rows)
    invalid_hist = _global_histogram(diana_invalid_event_rows)
    invalid_beats_by_segment = _beats_by_segment(diana_invalid_event_rows)

    comparison_shares = {
        "thorne": _segment_share(thorne_event_rows),
        "diana_valid": _segment_share(diana_valid_event_rows),
        "diana_invalid": _segment_share(diana_invalid_event_rows),
        "marcus": _segment_share(marcus_event_rows),
    }

    late_share_invalid = float(comparison_shares["diana_invalid"]["late"])
    verdict = _verdict(late_share_invalid)

    payload = {
        "config": {
            "seeds": [int(seed) for seed in seeds],
            "condition": "full_evolution",
            "total_runs": int(len(seeds)),
            "bin_edges": [round(i / 10.0, 1) for i in range(11)],
            "segment_thresholds": {"early_lt": 0.25, "mid_lte": 0.70, "late_gt": 0.70},
        },
        "diana": {
            "valid_arc_count": int(diana_valid_arc_count),
            "invalid_arc_count": int(diana_invalid_arc_count),
            "valid_event_stats": diana_valid_stats,
            "invalid_event_stats": diana_invalid_stats,
            "invalid_global_histogram": invalid_hist,
            "invalid_beats_by_segment": invalid_beats_by_segment,
        },
        "comparison_early_mid_late_share": comparison_shares,
        "verdict": {
            "label": verdict,
            "late_share_invalid": late_share_invalid,
        },
        "per_seed": per_seed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_summary(
        diana_valid_arc_count=diana_valid_arc_count,
        diana_invalid_arc_count=diana_invalid_arc_count,
        diana_valid_stats=diana_valid_stats,
        diana_invalid_stats=diana_invalid_stats,
        invalid_hist=invalid_hist,
        invalid_beats_by_segment=invalid_beats_by_segment,
        comparison_shares=comparison_shares,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
