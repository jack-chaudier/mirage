from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from narrativefield.schema.agents import AgentState
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import Event, EventType
from narrativefield.schema.world import Location, SecretDefinition, WorldDefinition


@dataclass
class PerceivedState:
    """An agent's perceived slice of the world (tick-loop.md Phase 1)."""

    visible_agents: list[AgentState] = field(default_factory=list)
    overhearable_locations: list[str] = field(default_factory=list)
    visible_emotions: dict[str, dict[str, float]] = field(default_factory=dict)

    # Simple, optional flags used by the decision engine.
    recent_conflict_at_location: bool = False


@dataclass
class Action:
    """A proposed action from an agent (tick-loop.md Phase 3)."""

    agent_id: str
    action_type: EventType
    target_agents: list[str]
    location_id: str
    utility_score: float
    content: str

    dialogue: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_target_available: bool = True
    priority_class: int = 1
    is_dramatic: bool = False


@dataclass
class WorldState:
    """Materialized world state for the tick loop and decision engine."""

    definition: WorldDefinition
    agents: dict[str, AgentState]

    sim_time: float = 0.0
    tick_id: int = 0

    event_log: list[Event] = field(default_factory=list)
    event_seq: int = 0  # used for evt_{seq:04d} IDs
    # True if the event limit cut off a tick mid-generation.
    truncated: bool = False
    canon: WorldCanon | None = None

    @property
    def locations(self) -> dict[str, Location]:
        return self.definition.locations

    @property
    def secrets(self) -> dict[str, SecretDefinition]:
        return self.definition.secrets

    @property
    def seating(self) -> dict[str, list[str]] | None:
        return self.definition.seating
