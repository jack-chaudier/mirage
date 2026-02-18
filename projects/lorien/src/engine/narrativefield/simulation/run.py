from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from narrativefield.schema.agents import BeliefState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import Event
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, init_canon_from_world, run_simulation


FORMAT_VERSION = "1.0.0"


@dataclass(frozen=True)
class SimulationMetadata:
    simulation_id: str
    deterministic_id: str
    scenario: str
    total_ticks: int
    total_sim_time: float
    agent_count: int
    event_count: int
    snapshot_interval: int
    timestamp: str
    seed: int = 0
    time_scale: float = 1.0
    truncated: bool = False
    python_version: str = ""
    git_commit: str | None = None
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "deterministic_id": self.deterministic_id,
            "scenario": self.scenario,
            "total_ticks": self.total_ticks,
            "total_sim_time": self.total_sim_time,
            "agent_count": self.agent_count,
            "event_count": self.event_count,
            "snapshot_interval": self.snapshot_interval,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "time_scale": self.time_scale,
            "truncated": self.truncated,
            "python_version": self.python_version,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
        }


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _deterministic_id(*, scenario: str, seed: int) -> str:
    return f"{scenario}_seed_{seed}"


def validate_events(events: list[Event]) -> list[str]:
    errors: list[str] = []
    if not events:
        errors.append("Event log is empty")
        return errors

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if (curr.tick_id, curr.order_in_tick) <= (prev.tick_id, prev.order_in_tick):
            errors.append(f"Event ordering violation at {curr.id}")
            break

    ids = {e.id for e in events}
    for e in events:
        for parent in e.causal_links:
            if parent not in ids:
                errors.append(f"{e.id} references missing causal link {parent}")
                break

    return errors


def _load_canon(path: Path) -> WorldCanon:
    raw = json.loads(path.read_text(encoding="utf-8"))
    world_canon_raw = raw.get("world_canon")
    if not isinstance(world_canon_raw, dict):
        raise SystemExit(f"Canon file {path} is missing 'world_canon'")
    return WorldCanon.from_dict(world_canon_raw)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="narrativefield.simulation.run")
    parser.add_argument("--scenario", default="dinner_party", choices=["dinner_party"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--tick-limit", type=int, default=300)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--canon")
    args = parser.parse_args(argv)

    if args.time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {args.time_scale}")

    if args.scenario != "dinner_party":
        raise SystemExit(f"Unsupported scenario: {args.scenario}")

    world = create_dinner_party_world()
    loaded_canon: WorldCanon | None = None
    if args.canon:
        loaded_canon = _load_canon(Path(args.canon))
        if loaded_canon.world_id and loaded_canon.world_id != world.definition.id:
            raise SystemExit(
                f"Canon world_id mismatch: expected '{world.definition.id}', got '{loaded_canon.world_id}'"
            )

    world.canon = init_canon_from_world(world.definition, loaded_canon)
    if loaded_canon is not None:
        _apply_claim_state_overrides(world, world.canon)

    rng = Random(args.seed)

    cfg = SimulationConfig(
        tick_limit=int(args.tick_limit),
        event_limit=int(args.event_limit),
        max_sim_time=world.definition.sim_duration_minutes * args.time_scale,
        snapshot_interval_events=world.definition.snapshot_interval,
        time_scale=args.time_scale,
    )

    events, snapshots = run_simulation(world, rng, cfg)
    errs = validate_events(events)
    if errs:
        raise SystemExit("Invalid simulation output:\\n" + "\\n".join(f"- {e}" for e in errs))

    timestamp = datetime.now(timezone.utc).isoformat()
    sim_id = f"sim_{timestamp.replace(':', '').replace('-', '').replace('+', 'Z')}"

    cfg_dict = {
        "tick_limit": cfg.tick_limit,
        "event_limit": cfg.event_limit,
        "max_sim_time": cfg.max_sim_time,
        "snapshot_interval_events": cfg.snapshot_interval_events,
        "max_catastrophes_per_tick": cfg.max_catastrophes_per_tick,
        "max_actions_per_tick": cfg.max_actions_per_tick,
        "time_scale": cfg.time_scale,
    }

    metadata = SimulationMetadata(
        simulation_id=sim_id,
        deterministic_id=_deterministic_id(scenario=args.scenario, seed=int(args.seed)),
        scenario=args.scenario,
        total_ticks=world.tick_id,
        total_sim_time=world.sim_time,
        agent_count=len(world.agents),
        event_count=len(events),
        snapshot_interval=world.definition.snapshot_interval,
        timestamp=timestamp,
        seed=args.seed,
        time_scale=args.time_scale,
        truncated=bool(world.truncated),
        python_version=sys.version,
        git_commit=_get_git_commit(),
        config_hash=_config_hash(cfg_dict),
    )

    output: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "metadata": metadata.to_dict(),
        "initial_state": snapshots[0],
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "claims": [c.to_dict() for c in world.definition.claims.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
        "world_canon": world.canon.to_dict() if world.canon is not None else WorldCanon().to_dict(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
