from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import AgentState
from .world import Location, SecretDefinition


@dataclass
class Scene:
    id: str
    event_ids: list[str]
    location: str
    participants: list[str]
    time_start: float
    time_end: float
    tick_start: int
    tick_end: int

    tension_arc: list[float] = field(default_factory=list)
    tension_peak: float = 0.0
    tension_mean: float = 0.0
    dominant_theme: str = ""
    scene_type: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_ids": list(self.event_ids),
            "location": self.location,
            "participants": list(self.participants),
            "time_start": self.time_start,
            "time_end": self.time_end,
            "tick_start": self.tick_start,
            "tick_end": self.tick_end,
            "tension_arc": list(self.tension_arc),
            "tension_peak": self.tension_peak,
            "tension_mean": self.tension_mean,
            "dominant_theme": self.dominant_theme,
            "scene_type": self.scene_type,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(
            id=str(data["id"]),
            event_ids=[str(x) for x in (data.get("event_ids") or [])],
            location=str(data.get("location", "")),
            participants=[str(x) for x in (data.get("participants") or [])],
            time_start=float(data.get("time_start", 0.0)),
            time_end=float(data.get("time_end", 0.0)),
            tick_start=int(data.get("tick_start", 0)),
            tick_end=int(data.get("tick_end", 0)),
            tension_arc=[float(x) for x in (data.get("tension_arc") or [])],
            tension_peak=float(data.get("tension_peak", 0.0)),
            tension_mean=float(data.get("tension_mean", 0.0)),
            dominant_theme=str(data.get("dominant_theme", "")),
            scene_type=str(data.get("scene_type", "")),
            summary=str(data.get("summary", "")),
        )


@dataclass
class SnapshotState:
    snapshot_id: str
    tick_id: int
    sim_time: float
    event_count: int

    agents: dict[str, AgentState]
    secrets: dict[str, SecretDefinition]
    locations: dict[str, Location]

    global_tension: float = 0.0
    active_scene_id: str = ""
    belief_matrix: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "tick_id": self.tick_id,
            "sim_time": self.sim_time,
            "event_count": self.event_count,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "secrets": {k: v.to_dict() for k, v in self.secrets.items()},
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "global_tension": self.global_tension,
            "active_scene_id": self.active_scene_id,
            "belief_matrix": self.belief_matrix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotState":
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            tick_id=int(data.get("tick_id", 0)),
            sim_time=float(data.get("sim_time", 0.0)),
            event_count=int(data.get("event_count", 0)),
            agents={str(k): AgentState.from_dict(v) for k, v in (data.get("agents") or {}).items()},
            secrets={
                str(k): SecretDefinition.from_dict(v) for k, v in (data.get("secrets") or {}).items()
            },
            locations={
                str(k): Location.from_dict(v) for k, v in (data.get("locations") or {}).items()
            },
            global_tension=float(data.get("global_tension", 0.0)),
            active_scene_id=str(data.get("active_scene_id", "")),
            belief_matrix={
                str(agent): {str(secret): str(state) for secret, state in row.items()}
                for agent, row in (data.get("belief_matrix") or {}).items()
            },
        )


@dataclass(frozen=True)
class CausalNeighborhood:
    event_id: str
    backward: list[str]
    forward: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "backward": list(self.backward), "forward": list(self.forward)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CausalNeighborhood":
        return cls(
            event_id=str(data["event_id"]),
            backward=[str(x) for x in (data.get("backward") or [])],
            forward=[str(x) for x in (data.get("forward") or [])],
        )
