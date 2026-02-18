"""Run deterministic Rashomon arc extraction across a seed range.

Run:
    cd src/engine && python -m scripts.sweep_rashomon --seed-from 1 --seed-to 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from narrativefield.extraction.rashomon import extract_rashomon_set
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import BeliefState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DINNER_PARTY_AGENTS = ["thorne", "elena", "marcus", "lydia", "diana", "victor"]
BEAT_ORDER = [
    BeatType.SETUP.value,
    BeatType.COMPLICATION.value,
    BeatType.ESCALATION.value,
    BeatType.TURNING_POINT.value,
    BeatType.CONSEQUENCE.value,
]


def _apply_claim_state_overrides(world, canon: WorldCanon) -> None:
    for claim_id, beliefs_by_agent in sorted(canon.claim_states.items()):
        if claim_id not in world.definition.all_claims:
            continue
        for agent_id, belief_state in sorted((beliefs_by_agent or {}).items()):
            agent = world.agents.get(agent_id)
            if agent is None:
                continue
            try:
                agent.beliefs[claim_id] = BeliefState(str(belief_state))
            except ValueError:
                continue


def _build_payload(seed: int, loaded_canon: WorldCanon | None) -> dict[str, Any]:
    world = create_dinner_party_world()
    canon_copy = WorldCanon.from_dict(loaded_canon.to_dict()) if loaded_canon is not None else None
    world.canon = init_canon_from_world(world.definition, canon_copy)
    if canon_copy is not None:
        _apply_claim_state_overrides(world, world.canon)

    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = run_simulation(world, Random(seed), cfg)

    initial_state = snapshots[0] if snapshots else {}
    periodic_snapshots = snapshots[1:] if len(snapshots) > 1 else []
    return {
        "format_version": "1.0.0",
        "metadata": {
            "scenario": "dinner_party",
            "seed": int(seed),
            "total_ticks": int(world.tick_id),
            "total_sim_time": float(world.sim_time),
            "agent_count": int(len(world.agents)),
            "event_count": int(len(events)),
            "snapshot_interval": int(world.definition.snapshot_interval),
            "truncated": bool(world.truncated),
        },
        "initial_state": initial_state,
        "snapshots": periodic_snapshots,
        "events": [event.to_dict() for event in events],
        "secrets": [secret.to_dict() for secret in world.definition.secrets.values()],
        "claims": [claim.to_dict() for claim in world.definition.claims.values()],
        "locations": [location.to_dict() for location in world.definition.locations.values()],
        "world_canon": world.canon.to_dict() if world.canon is not None else WorldCanon().to_dict(),
    }


def _load_canon(path: str | None) -> WorldCanon | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "world_canon" in raw and isinstance(raw.get("world_canon"), dict):
        raw = raw["world_canon"]
    if not isinstance(raw, dict):
        raise ValueError("Canon payload must be an object.")
    return WorldCanon.from_dict(raw)


def _beat_summary(beats: list[BeatType]) -> str:
    counts = Counter(beat.value for beat in beats)
    return " ".join(f"{beat_name}({int(counts.get(beat_name, 0))})" for beat_name in BEAT_ORDER)


def _turning_point_event_id(events: list[Event], beats: list[BeatType]) -> str | None:
    for event, beat in zip(events, beats):
        if beat == BeatType.TURNING_POINT:
            return event.id
    return None


def _compact_event(event: Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "type": event.type.value,
        "source_agent": event.source_agent,
        "target_agents": list(event.target_agents),
        "location_id": event.location_id,
        "description": event.description,
        "tension": float(event.metrics.tension),
        "significance": float(event.metrics.significance),
    }


def _shared_tp_preview(tp_overlap: dict[str, list[str]]) -> str:
    shared = [(event_id, protagonists) for event_id, protagonists in tp_overlap.items() if len(protagonists) >= 2]
    if not shared:
        return "none"
    shared.sort(key=lambda item: (-len(item[1]), item[0]))
    best_event, protagonists = shared[0]
    return f"{best_event}({len(protagonists)} arcs)"


def _resolve_output_path(output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return output_path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_DIR / f"rashomon_sweep_{timestamp}.json"


def _print_summary(payload: dict[str, Any]) -> None:
    seed_from, seed_to = payload["seed_range"]
    summary = payload["summary"]
    total_seeds = int(payload["total_seeds"])

    print()
    print(f"=== RASHOMON SWEEP SUMMARY (seeds {seed_from}-{seed_to}) ===")
    print()
    print("Protagonist validity rates:")
    for protagonist, rate in sorted(
        summary["protagonist_validity_rate"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        valid = int(round(float(rate) * total_seeds))
        print(f"  {protagonist:<9} {valid:>2}/{total_seeds} ({float(rate) * 100.0:>4.0f}%)")
    print()

    print(f"Mean valid arcs per seed: {float(summary['mean_valid_arcs_per_seed']):.2f}")
    print(
        f"Seeds with all 6 valid: {int(summary['seeds_with_all_6_valid'])}/{total_seeds} "
        f"({(int(summary['seeds_with_all_6_valid']) / total_seeds * 100.0):.0f}%)"
    )
    print()

    shared_turning_points = summary["shared_turning_points"]
    print("Top shared turning points (appear in 3+ arcs):")
    if not shared_turning_points:
        print("  (none)")
    else:
        ranked = sorted(
            shared_turning_points.items(),
            key=lambda item: (-int(item[1]["seed_hits"]), item[0]),
        )
        for event_id, info in ranked[:8]:
            protagonists = ", ".join(info["protagonists"])
            print(f"  {event_id}: {protagonists} ({int(info['seed_hits'])} seed(s))")
    print()

    print("Mean pairwise event overlap (Jaccard):")
    overlap_matrix = summary["mean_overlap_matrix"]
    if not overlap_matrix:
        print("  (none)")
    else:
        for pair_key, value in sorted(overlap_matrix.items(), key=lambda item: (-item[1], item[0]))[:12]:
            print(f"  {pair_key:<18} {float(value):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Rashomon arc extraction across deterministic seeds.")
    parser.add_argument("--seed-from", type=int, default=1, help="First seed (inclusive).")
    parser.add_argument("--seed-to", type=int, default=50, help="Last seed (inclusive).")
    parser.add_argument(
        "--canon-path",
        type=str,
        default=None,
        help="Optional path to canon JSON (world_canon object or full sim payload).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    args = parser.parse_args()

    if args.seed_to < args.seed_from:
        raise ValueError("--seed-to must be >= --seed-from")

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_canon = _load_canon(args.canon_path)
    seeds = list(range(args.seed_from, args.seed_to + 1))

    per_seed: list[dict[str, Any]] = []
    valid_counts: list[int] = []
    protagonist_total = Counter()
    protagonist_valid = Counter()
    overlap_sums: dict[str, float] = defaultdict(float)
    overlap_counts: dict[str, int] = defaultdict(int)
    shared_tp_seed_hits: dict[str, int] = defaultdict(int)
    shared_tp_protagonists: dict[str, set[str]] = defaultdict(set)

    for seed in seeds:
        sim_payload = _build_payload(seed, loaded_canon)
        parsed = parse_simulation_output(sim_payload)
        metrics_output = run_metrics_pipeline(parsed)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None

        rashomon_set = extract_rashomon_set(
            events=metrics_output.events,
            seed=seed,
            agents=list(DINNER_PARTY_AGENTS),
            total_sim_time=total_sim_time,
        )
        overlap_matrix = rashomon_set.overlap_matrix
        turning_overlap = rashomon_set.turning_point_overlap()
        valid_counts.append(rashomon_set.valid_count)

        for pair_key, overlap in overlap_matrix.items():
            overlap_sums[pair_key] += float(overlap)
            overlap_counts[pair_key] += 1

        for event_id, protagonists in turning_overlap.items():
            if len(protagonists) < 2:
                continue
            shared_tp_seed_hits[event_id] += 1
            shared_tp_protagonists[event_id].update(protagonists)

        arcs_payload: list[dict[str, Any]] = []
        for arc in rashomon_set.arcs:
            protagonist_total[arc.protagonist] += 1
            if arc.valid:
                protagonist_valid[arc.protagonist] += 1
            arcs_payload.append(
                {
                    "protagonist": arc.protagonist,
                    "valid": arc.valid,
                    "score": float(arc.arc_score.composite) if arc.arc_score else None,
                    "event_count": len(arc.events),
                    "beat_summary": _beat_summary(arc.beats),
                    "turning_point_event_id": _turning_point_event_id(arc.events, arc.beats),
                    "violations": list(arc.violations),
                    "beats": [beat.value for beat in arc.beats],
                    "events": [_compact_event(event) for event in arc.events],
                }
            )

        best_valid = next((arc for arc in rashomon_set.arcs if arc.valid and arc.arc_score), None)
        best_label = (
            f"{best_valid.protagonist}({best_valid.arc_score.composite:.2f})"
            if best_valid is not None and best_valid.arc_score is not None
            else "none"
        )
        print(
            f"Seed {seed:>2}: {rashomon_set.valid_count}/{len(rashomon_set.arcs)} valid | "
            f"best={best_label} | shared TPs: {_shared_tp_preview(turning_overlap)}"
        )

        per_seed.append(
            {
                "seed": int(seed),
                "valid_count": int(rashomon_set.valid_count),
                "arcs": arcs_payload,
                "overlap_matrix": overlap_matrix,
                "turning_point_overlap": turning_overlap,
            }
        )

    total_seeds = len(seeds)
    mean_overlap_matrix = {
        pair_key: (overlap_sums[pair_key] / overlap_counts[pair_key])
        for pair_key in sorted(overlap_sums.keys())
        if overlap_counts[pair_key] > 0
    }
    shared_turning_points = {
        event_id: {
            "protagonists": sorted(shared_tp_protagonists[event_id]),
            "seed_hits": int(shared_tp_seed_hits[event_id]),
        }
        for event_id in sorted(shared_tp_seed_hits.keys())
    }
    summary = {
        "mean_valid_arcs_per_seed": (sum(valid_counts) / total_seeds) if total_seeds else 0.0,
        "min_valid": min(valid_counts) if valid_counts else 0,
        "max_valid": max(valid_counts) if valid_counts else 0,
        "seeds_with_all_6_valid": sum(1 for count in valid_counts if count == 6),
        "protagonist_validity_rate": {
            protagonist: (
                float(protagonist_valid.get(protagonist, 0)) / float(total_seeds)
                if total_seeds
                else 0.0
            )
            for protagonist in DINNER_PARTY_AGENTS
        },
        "mean_overlap_matrix": mean_overlap_matrix,
        "shared_turning_points": shared_turning_points,
    }

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed_range": [int(args.seed_from), int(args.seed_to)],
        "total_seeds": int(total_seeds),
        "summary": summary,
        "per_seed": per_seed,
    }
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _print_summary(output_payload)
    print()
    print(f"Saved Rashomon sweep results: {output_path}")


if __name__ == "__main__":
    main()
