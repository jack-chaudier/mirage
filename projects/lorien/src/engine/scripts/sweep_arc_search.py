"""Run a seed sweep for arc search validity and failure diagnostics.

Run:
    cd src/engine && python -m scripts.sweep_arc_search --seed-from 1 --seed-to 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from statistics import median
from typing import Any

from narrativefield.extraction.arc_search import search_arc
from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline
from narrativefield.schema.agents import BeliefState
from narrativefield.schema.canon import WorldCanon
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


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


def _print_summary(summary: dict[str, Any]) -> None:
    seed_from = int(summary["seed_from"])
    seed_to = int(summary["seed_to"])
    total = int(summary["total_runs"])
    valid = int(summary["valid_runs"])
    invalid = int(summary["invalid_runs"])

    print()
    print(f"=== ARC SEARCH SWEEP: seeds {seed_from}-{seed_to} ===")
    print()
    print(f"Valid arcs: {valid:>3}/{total} ({(valid / total * 100.0):.0f}%)")
    print(f"Fallback:  {invalid:>3}/{total} ({(invalid / total * 100.0):.0f}%)")
    print()

    print("Rule failure frequency (across all invalid runs):")
    rule_presence = summary["rule_presence_by_invalid_run"]
    if rule_presence:
        for rule, count in sorted(rule_presence.items(), key=lambda item: (-item[1], item[0])):
            pct = (float(count) / float(invalid) * 100.0) if invalid else 0.0
            print(f"  {rule:<24} {int(count):>3}/{invalid:<3} ({pct:>3.0f}%)")
    else:
        print("  (none)")
    print()

    print("Near-miss analysis (best candidate violation count):")
    near_miss = summary["near_miss_buckets"]
    print(f"  1 violation:  {int(near_miss['1']):>3} runs")
    print(f"  2 violations: {int(near_miss['2']):>3} runs")
    print(f"  3+ violations:{int(near_miss['3+']):>3} runs")
    print()

    valid_scores = summary["valid_arc_scores"]
    if valid_scores["count"] > 0:
        print("Valid arc scores:")
        print(
            f"  min={valid_scores['min']:.2f}  "
            f"median={valid_scores['median']:.2f}  "
            f"max={valid_scores['max']:.2f}"
        )
    else:
        print("Valid arc scores: (none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep arc search performance across seeds.")
    parser.add_argument("--seed-from", type=int, default=1, help="First seed in the sweep (inclusive).")
    parser.add_argument("--seed-to", type=int, default=20, help="Last seed in the sweep (inclusive).")
    parser.add_argument(
        "--canon-path",
        type=str,
        default=None,
        help="Optional path to canon JSON (either world_canon object or full sim payload with world_canon).",
    )
    args = parser.parse_args()

    if args.seed_to < args.seed_from:
        raise ValueError("--seed-to must be >= --seed-from")

    loaded_canon = _load_canon(args.canon_path)
    seeds = list(range(args.seed_from, args.seed_to + 1))
    results: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds, start=1):
        payload = _build_payload(seed, loaded_canon)
        parsed = parse_simulation_output(payload)
        metrics_output = run_metrics_pipeline(parsed)
        total_sim_time = float(parsed.metadata.get("total_sim_time") or 0.0) or None
        search_result = search_arc(all_events=metrics_output.events, total_sim_time=total_sim_time)

        diagnostics = search_result.diagnostics.to_dict() if search_result.diagnostics else {}
        is_valid = bool(search_result.validation.valid)
        score = float(search_result.arc_score.composite) if search_result.arc_score is not None else None
        result_row = {
            "seed": int(seed),
            "protagonist": search_result.protagonist,
            "arc_valid": is_valid,
            "violations": list(search_result.validation.violations),
            "rule_failure_counts": diagnostics.get("rule_failure_counts", {}),
            "candidates_evaluated": int(diagnostics.get("candidates_evaluated", 0) or 0),
            "best_candidate_violation_count": int(diagnostics.get("best_candidate_violation_count", 0) or 0),
            "best_candidate_violations": diagnostics.get("best_candidate_violations", []),
            "primary_failure": diagnostics.get("primary_failure", ""),
            "arc_score_composite": score,
            "selected_event_count": len(search_result.events),
        }
        results.append(result_row)

        if is_valid:
            print(
                f"[{index}/{len(seeds)}] seed={seed}: VALID "
                f"(score={score:.2f}, protagonist={search_result.protagonist})"
            )
        else:
            print(
                f"[{index}/{len(seeds)}] seed={seed}: INVALID "
                f"(primary_failure={result_row['primary_failure'] or 'unknown'})"
            )

    valid_rows = [row for row in results if row["arc_valid"]]
    invalid_rows = [row for row in results if not row["arc_valid"]]
    rule_presence = Counter()
    for row in invalid_rows:
        counts = row.get("rule_failure_counts", {})
        if not isinstance(counts, dict):
            continue
        for rule, count in counts.items():
            if int(count) > 0:
                rule_presence[str(rule)] += 1

    near_miss = {"1": 0, "2": 0, "3+": 0}
    for row in invalid_rows:
        count = int(row.get("best_candidate_violation_count", 0) or 0)
        if count <= 1:
            near_miss["1"] += 1
        elif count == 2:
            near_miss["2"] += 1
        else:
            near_miss["3+"] += 1

    valid_scores_list = [
        float(row["arc_score_composite"])
        for row in valid_rows
        if row.get("arc_score_composite") is not None
    ]
    if valid_scores_list:
        valid_score_summary = {
            "count": len(valid_scores_list),
            "min": min(valid_scores_list),
            "median": median(valid_scores_list),
            "max": max(valid_scores_list),
        }
    else:
        valid_score_summary = {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "max": 0.0,
        }

    summary = {
        "seed_from": int(args.seed_from),
        "seed_to": int(args.seed_to),
        "total_runs": len(results),
        "valid_runs": len(valid_rows),
        "invalid_runs": len(invalid_rows),
        "rule_presence_by_invalid_run": dict(rule_presence),
        "near_miss_buckets": near_miss,
        "valid_arc_scores": valid_score_summary,
    }
    _print_summary(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = OUTPUT_DIR / f"arc_sweep_{timestamp}.json"
    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canon_path": args.canon_path,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print()
    print(f"Saved raw sweep results: {output_path}")


if __name__ == "__main__":
    main()
