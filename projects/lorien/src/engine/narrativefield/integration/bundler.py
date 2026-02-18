from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from narrativefield.metrics.irony import compute_snapshot_irony
from narrativefield.schema.agents import AgentState, BeliefState
from narrativefield.schema.events import Event
from narrativefield.schema.scenes import Scene
from narrativefield.schema.world import Location, SecretDefinition


FORMAT_VERSION = "1.0.0"
logger = logging.getLogger(__name__)


def _summarize_goals(agent: AgentState) -> str:
    g = agent.goals
    closeness_mean = 0.0
    if g.closeness:
        closeness_mean = sum(float(v) for v in g.closeness.values()) / max(1, len(g.closeness))
    dims = {
        "safety": float(g.safety),
        "status": float(g.status),
        "closeness": float(closeness_mean),
        "secrecy": float(g.secrecy),
        "truth_seeking": float(g.truth_seeking),
        "autonomy": float(g.autonomy),
        "loyalty": float(g.loyalty),
    }
    top = sorted(dims.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return "Top goals: " + ", ".join(f"{k}={v:.2f}" for k, v in top)


def _primary_flaw(agent: AgentState) -> str:
    if not agent.flaws:
        return ""
    return agent.flaws[0].description or agent.flaws[0].flaw_type.value


@dataclass(frozen=True)
class BundleInputs:
    metadata: dict[str, Any]
    initial_agents: dict[str, AgentState]
    snapshots: list[dict[str, Any]]  # includes initial_state + periodic snapshots
    events: list[Event]
    scenes: list[Scene]
    secrets: dict[str, SecretDefinition]
    locations: dict[str, Location]
    world_canon: dict[str, Any] | None = None


def build_belief_snapshots(inputs: BundleInputs) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for snap in inputs.snapshots:
        tick_id = int(snap.get("tick_id", 0))
        sim_time = float(snap.get("sim_time", 0.0))
        raw_agents = snap.get("agents") or {}

        beliefs: dict[str, dict[str, str]] = {}
        typed_beliefs: dict[str, dict[str, BeliefState]] = {}

        for agent_id, raw in raw_agents.items():
            row_raw = (raw.get("beliefs") or {}) if isinstance(raw, dict) else {}
            beliefs[str(agent_id)] = {str(k): str(v) for k, v in row_raw.items()}
            typed_beliefs[str(agent_id)] = {}
            for sid, st in beliefs[str(agent_id)].items():
                try:
                    typed_beliefs[str(agent_id)][sid] = BeliefState(st)
                except ValueError:
                    typed_beliefs[str(agent_id)][sid] = BeliefState.UNKNOWN

        agent_irony, sc = compute_snapshot_irony(beliefs=typed_beliefs, secrets=inputs.secrets)

        out.append(
            {
                "tick_id": tick_id,
                "sim_time": sim_time,
                "beliefs": beliefs,
                "agent_irony": {k: float(v) for k, v in agent_irony.items()},
                "scene_irony": float(sc),
            }
        )
    return out


def bundle_for_renderer(inputs: BundleInputs) -> dict[str, Any]:
    if not inputs.events:
        logger.warning("Renderer bundler received 0 events.")
    if not inputs.initial_agents:
        logger.warning("Renderer bundler received 0 agents in initial_agents.")
    if not inputs.snapshots:
        logger.warning("Renderer bundler received 0 snapshots.")

    belief_snapshots = build_belief_snapshots(inputs)

    agents_manifest = []
    for agent in inputs.initial_agents.values():
        agents_manifest.append(
            {
                "id": agent.id,
                "name": agent.name,
                "initial_location": agent.location,
                "goal_summary": _summarize_goals(agent),
                "primary_flaw": _primary_flaw(agent),
            }
        )

    # Metadata in the renderer payload should describe *this* payload (not the raw sim output).
    # Preserve raw counts in optional fields so downstream tools can reason about compression.
    metadata = dict(inputs.metadata)
    if "event_count" in metadata and "raw_event_count" not in metadata:
        metadata["raw_event_count"] = metadata["event_count"]
    metadata["event_count"] = len(inputs.events)
    metadata["agent_count"] = len(agents_manifest)

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "agents": agents_manifest,
        "locations": [loc.to_dict() for loc in inputs.locations.values()],
        "secrets": [sec.to_dict() for sec in inputs.secrets.values()],
        "events": [e.to_dict() for e in inputs.events],
        "scenes": [s.to_dict() for s in inputs.scenes],
        "belief_snapshots": belief_snapshots,
    }
    if inputs.world_canon is not None:
        payload["world_canon"] = dict(inputs.world_canon)
    return payload
