"""Diagnostic for Diana's strict-invalid arc composition in full evolution runs.

Run:
    cd src/engine && ./.venv/bin/python -m scripts.diana_diagnostic
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from narrativefield.extraction.rashomon import extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from scripts.k_sweep_experiment import (
    _generate_canon_after_b_for_seed,
    _simulate_story,
)
from scripts.test_goal_evolution import _evolution_profiles

TARGET_AGENTS = ["diana", "thorne", "marcus"]
DISPLAY_EVENT_TYPES = [
    "observe",
    "conflict",
    "chat",
    "catastrophe",
    "reveal",
    "confide",
    "physical",
]
ALL_EVENT_TYPES = [
    "chat",
    "observe",
    "social_move",
    "reveal",
    "conflict",
    "internal",
    "physical",
    "confide",
    "lie",
    "catastrophe",
]
BEAT_TYPES = ["setup", "complication", "escalation", "turning_point", "consequence"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "diana_diagnostic.json"
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


def _agent_involved(event, agent_id: str) -> bool:
    return event.source_agent == agent_id or agent_id in event.target_agents


def _row_for_agent(*, seed: int, agent_id: str, full_events: list[Any], arc: Any) -> dict[str, Any]:
    sim_involvement_count = sum(1 for event in full_events if _agent_involved(event, agent_id))
    arc_event_counter = Counter(event.type.value for event in arc.events)
    arc_beat_counter = Counter(beat.value for beat in arc.beats)

    event_counts = {event_type: int(arc_event_counter.get(event_type, 0)) for event_type in ALL_EVENT_TYPES}
    beat_counts = {beat_type: int(arc_beat_counter.get(beat_type, 0)) for beat_type in BEAT_TYPES}
    arc_event_count = int(len(arc.events))
    shown_total = sum(event_counts.get(event_type, 0) for event_type in DISPLAY_EVENT_TYPES)
    other_count = int(max(0, arc_event_count - shown_total))

    return {
        "seed": int(seed),
        "agent": agent_id,
        "sim_involvement_count": int(sim_involvement_count),
        "arc_event_count": arc_event_count,
        "arc_event_type_counts": event_counts,
        "arc_beat_counts": beat_counts,
        "arc_other_event_count": other_count,
        "valid": bool(arc.valid),
        "violations": [str(v) for v in arc.violations],
    }


def _mean_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sim_means = _mean([float(row["sim_involvement_count"]) for row in rows]) if rows else 0.0
    arc_means = _mean([float(row["arc_event_count"]) for row in rows]) if rows else 0.0

    event_means: dict[str, float] = {}
    for event_type in ALL_EVENT_TYPES:
        event_means[event_type] = _mean(
            [float(row["arc_event_type_counts"].get(event_type, 0)) for row in rows]
        ) if rows else 0.0

    beat_means: dict[str, float] = {}
    for beat_type in BEAT_TYPES:
        beat_means[beat_type] = _mean(
            [float(row["arc_beat_counts"].get(beat_type, 0)) for row in rows]
        ) if rows else 0.0

    return {
        "sim_involvement_count": float(sim_means),
        "arc_event_count": float(arc_means),
        "arc_event_type_means": event_means,
        "arc_beat_means": beat_means,
        "arc_other_event_mean": _mean([float(row["arc_other_event_count"]) for row in rows]) if rows else 0.0,
    }


def _comparison_row(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    valid_count = sum(1 for row in rows if bool(row["valid"]))
    mean_involvement = _mean([float(row["sim_involvement_count"]) for row in rows]) if rows else 0.0
    mean_arc_events = _mean([float(row["arc_event_count"]) for row in rows]) if rows else 0.0
    mean_observe = _mean([float(row["arc_event_type_counts"].get("observe", 0)) for row in rows]) if rows else 0.0
    return {
        "mean_sim_involvement": float(mean_involvement),
        "mean_arc_events": float(mean_arc_events),
        "mean_arc_observe_events": float(mean_observe),
        "arc_valid_rate": float(valid_count / n) if n else 0.0,
    }


def _verdict(diana_valid_means: dict[str, Any], diana_invalid_means: dict[str, Any]) -> dict[str, Any]:
    valid_inv = float(diana_valid_means["sim_involvement_count"])
    invalid_inv = float(diana_invalid_means["sim_involvement_count"])
    involvement_gap = invalid_inv / max(valid_inv, EPSILON)

    valid_arc_events = float(diana_valid_means["arc_event_count"])
    invalid_arc_events = float(diana_invalid_means["arc_event_count"])

    valid_observe = float(diana_valid_means["arc_event_type_means"].get("observe", 0.0))
    invalid_observe = float(diana_invalid_means["arc_event_type_means"].get("observe", 0.0))

    observe_ratio_valid = valid_observe / max(valid_arc_events, EPSILON)
    observe_ratio_invalid = invalid_observe / max(invalid_arc_events, EPSILON)

    invalid_dev_beats = float(diana_invalid_means["arc_beat_means"].get("complication", 0.0)) + float(
        diana_invalid_means["arc_beat_means"].get("escalation", 0.0)
    )

    if involvement_gap < 0.75:
        label = "Narrative starvation: evolution concentrates dramatic mass in other agents"
    elif (observe_ratio_invalid > (observe_ratio_valid + 0.15)) and (invalid_dev_beats < 0.5):
        label = "Lexical blindness: beat classifier doesn't recognize epistemic development"
    else:
        label = "Mixed signal: both event supply and beat labeling likely contribute"

    return {
        "label": label,
        "rationale": {
            "involvement_gap": float(involvement_gap),
            "observe_ratio_valid": float(observe_ratio_valid),
            "observe_ratio_invalid": float(observe_ratio_invalid),
            "invalid_dev_beats_mean": float(invalid_dev_beats),
        },
    }


def _print_summary(
    *,
    diana_valid_count: int,
    diana_invalid_count: int,
    diana_valid_means: dict[str, Any],
    diana_invalid_means: dict[str, Any],
    comparison: dict[str, dict[str, float]],
    verdict: dict[str, Any],
) -> None:
    def _fmt_pair(key_path: list[str], digits: int = 1) -> tuple[str, str]:
        left = diana_valid_means
        right = diana_invalid_means
        for key in key_path:
            left = left[key]  # type: ignore[index]
            right = right[key]  # type: ignore[index]
        return (f"{float(left):.{digits}f}", f"{float(right):.{digits}f}")

    print()
    print("=== DIANA EVENT COMPOSITION (full evolution, 50 seeds) ===")
    print()
    print(f"Diana arcs: {diana_valid_count} valid, {diana_invalid_count} invalid")
    print()
    print("                        Valid arcs (mean)    Invalid arcs (mean)")
    a, b = _fmt_pair(["sim_involvement_count"])
    print(f"Total sim events:       {a:<20} {b}")
    a, b = _fmt_pair(["arc_event_count"])
    print(f"Arc events:             {a:<20} {b}")
    for event_type in DISPLAY_EVENT_TYPES:
        a, b = _fmt_pair(["arc_event_type_means", event_type])
        print(f"  {event_type.upper():<20}{a:<20} {b}")
    a, b = _fmt_pair(["arc_other_event_mean"])
    print(f"  {'other'.upper():<20}{a:<20} {b}")
    print()
    print("Beat distribution:")
    for beat_type in BEAT_TYPES:
        a, b = _fmt_pair(["arc_beat_means", beat_type])
        print(f"  {beat_type.upper():<20}{a:<20} {b}")
    print()

    print("=== COMPARISON: THORNE vs DIANA vs MARCUS ===")
    print("                        Thorne (mean)    Diana (mean)     Marcus (mean)")
    print(
        "Total sim involvement:  "
        f"{comparison['thorne']['mean_sim_involvement']:<16.1f}"
        f"{comparison['diana']['mean_sim_involvement']:<16.1f}"
        f"{comparison['marcus']['mean_sim_involvement']:.1f}"
    )
    print(
        "Arc events:             "
        f"{comparison['thorne']['mean_arc_events']:<16.1f}"
        f"{comparison['diana']['mean_arc_events']:<16.1f}"
        f"{comparison['marcus']['mean_arc_events']:.1f}"
    )
    print(
        "Arc OBSERVE events:     "
        f"{comparison['thorne']['mean_arc_observe_events']:<16.1f}"
        f"{comparison['diana']['mean_arc_observe_events']:<16.1f}"
        f"{comparison['marcus']['mean_arc_observe_events']:.1f}"
    )
    print(
        "Arc valid rate:         "
        f"{comparison['thorne']['arc_valid_rate'] * 100.0:>5.1f}%"
        f"{comparison['diana']['arc_valid_rate'] * 100.0:>16.1f}%"
        f"{comparison['marcus']['arc_valid_rate'] * 100.0:>16.1f}%"
    )
    print()
    print("=== VERDICT ===")
    print(verdict["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diana invalid arc event composition diagnostic.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--seeds", type=str, default="1-50")
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    seeds = _parse_seeds(args.seeds)
    evolutions = _evolution_profiles()["full"]

    per_seed_rows: list[dict[str, Any]] = []
    agent_rows: dict[str, list[dict[str, Any]]] = {agent: [] for agent in TARGET_AGENTS}
    diana_violations = Counter()

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index:03d}/{total_runs:03d}] seed={seed} condition=full_evolution", flush=True)
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
            evolutions=evolutions,
        )
        parsed = parse_simulation_output(story.payload)
        metrics_output = run_metrics_pipeline(parsed)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None

        rashomon = extract_rashomon_set(
            events=metrics_output.events,
            seed=seed,
            agents=list(TARGET_AGENTS),
            total_sim_time=total_sim_time,
        )
        arc_by_agent = {str(arc.protagonist): arc for arc in rashomon.arcs}

        per_seed_entry: dict[str, Any] = {"seed": int(seed)}
        for agent in TARGET_AGENTS:
            arc = arc_by_agent[agent]
            row = _row_for_agent(
                seed=seed,
                agent_id=agent,
                full_events=metrics_output.events,
                arc=arc,
            )
            agent_rows[agent].append(row)
            per_seed_entry[agent] = row

            if agent == "diana" and not bool(row["valid"]):
                for violation in row["violations"]:
                    diana_violations[str(violation)] += 1

        per_seed_rows.append(per_seed_entry)

    diana_valid_rows = [row for row in agent_rows["diana"] if bool(row["valid"])]
    diana_invalid_rows = [row for row in agent_rows["diana"] if not bool(row["valid"])]
    diana_valid_means = _mean_block(diana_valid_rows)
    diana_invalid_means = _mean_block(diana_invalid_rows)

    comparison = {
        "thorne": _comparison_row(agent_rows["thorne"]),
        "diana": _comparison_row(agent_rows["diana"]),
        "marcus": _comparison_row(agent_rows["marcus"]),
    }
    verdict = _verdict(diana_valid_means, diana_invalid_means)

    payload = {
        "config": {
            "seeds": [int(seed) for seed in seeds],
            "condition": "full_evolution",
            "total_runs": int(len(seeds)),
            "agents_compared": list(TARGET_AGENTS),
        },
        "diana_summary": {
            "n_valid": int(len(diana_valid_rows)),
            "n_invalid": int(len(diana_invalid_rows)),
            "valid_means": diana_valid_means,
            "invalid_means": diana_invalid_means,
            "invalid_violation_frequencies": {
                key: int(value) for key, value in sorted(diana_violations.items(), key=lambda item: (-item[1], item[0]))
            },
        },
        "comparison": comparison,
        "per_seed": per_seed_rows,
        "verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _print_summary(
        diana_valid_count=len(diana_valid_rows),
        diana_invalid_count=len(diana_invalid_rows),
        diana_valid_means=diana_valid_means,
        diana_invalid_means=diana_invalid_means,
        comparison=comparison,
        verdict=verdict,
    )


if __name__ == "__main__":
    main()
