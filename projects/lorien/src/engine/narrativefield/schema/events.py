from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union


class EventType(Enum):
    CHAT = "chat"
    OBSERVE = "observe"
    SOCIAL_MOVE = "social_move"
    REVEAL = "reveal"
    CONFLICT = "conflict"
    INTERNAL = "internal"
    PHYSICAL = "physical"
    CONFIDE = "confide"
    LIE = "lie"
    CATASTROPHE = "catastrophe"


class BeatType(Enum):
    SETUP = "setup"
    COMPLICATION = "complication"
    ESCALATION = "escalation"
    TURNING_POINT = "turning_point"
    CONSEQUENCE = "consequence"


class DeltaKind(Enum):
    AGENT_EMOTION = "agent_emotion"
    AGENT_RESOURCE = "agent_resource"
    AGENT_LOCATION = "agent_location"
    RELATIONSHIP = "relationship"
    BELIEF = "belief"
    SECRET_STATE = "secret_state"
    WORLD_RESOURCE = "world_resource"
    COMMITMENT = "commitment"
    PACING = "pacing"
    ARTIFACT_STATE = "artifact_state"
    FACTION_STATE = "faction_state"
    INSTITUTION_STATE = "institution_state"
    LOCATION_MEMORY = "location_memory"


class DeltaOp(Enum):
    SET = "set"
    ADD = "add"


DeltaValue = Union[float, int, str, bool]


@dataclass(frozen=True)
class StateDelta:
    """Typed, validatable state change (specs/schema/events.md Section 5)."""

    kind: DeltaKind
    agent: str
    agent_b: Optional[str] = None
    attribute: str = ""
    op: DeltaOp = DeltaOp.ADD
    value: DeltaValue = 0.0
    reason_code: str = ""
    reason_display: str = ""

    def __post_init__(self) -> None:
        if self.op == DeltaOp.ADD and not isinstance(self.value, (int, float)):
            raise ValueError("DeltaOp.ADD requires numeric value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "agent": self.agent,
            "agent_b": self.agent_b,
            "attribute": self.attribute,
            "op": self.op.value,
            "value": self.value,
            "reason_code": self.reason_code,
            "reason_display": self.reason_display,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateDelta":
        return cls(
            kind=DeltaKind(data["kind"]),
            agent=str(data["agent"]),
            agent_b=data.get("agent_b"),
            attribute=str(data.get("attribute", "")),
            op=DeltaOp(data.get("op", DeltaOp.ADD.value)),
            value=data.get("value", 0.0),
            reason_code=str(data.get("reason_code", "")),
            reason_display=str(data.get("reason_display", "")),
        )


@dataclass(frozen=True)
class CollapsedBelief:
    agent: str
    secret: str
    from_state: str
    to_state: str

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "secret": self.secret, "from": self.from_state, "to": self.to_state}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollapsedBelief":
        return cls(
            agent=str(data["agent"]),
            secret=str(data["secret"]),
            from_state=str(data["from"]),
            to_state=str(data["to"]),
        )


@dataclass(frozen=True)
class IronyCollapseInfo:
    detected: bool
    drop: float
    collapsed_beliefs: list[CollapsedBelief]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "drop": self.drop,
            "collapsed_beliefs": [b.to_dict() for b in self.collapsed_beliefs],
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IronyCollapseInfo":
        return cls(
            detected=bool(data.get("detected", False)),
            drop=float(data.get("drop", 0.0)),
            collapsed_beliefs=[
                CollapsedBelief.from_dict(b) for b in (data.get("collapsed_beliefs") or [])
            ],
            score=float(data.get("score", 0.0)),
        )


@dataclass
class EventMetrics:
    """Typed metrics container (specs/schema/events.md Section 8)."""

    tension: float = 0.0
    irony: float = 0.0
    significance: float = 0.0
    thematic_shift: dict[str, float] = field(default_factory=dict)
    tension_components: dict[str, float] = field(default_factory=dict)
    irony_collapse: Optional[IronyCollapseInfo] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tension": self.tension,
            "irony": self.irony,
            "significance": self.significance,
            "thematic_shift": dict(self.thematic_shift),
            "tension_components": dict(self.tension_components),
            "irony_collapse": self.irony_collapse.to_dict() if self.irony_collapse else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventMetrics":
        irony_collapse = data.get("irony_collapse")
        return cls(
            tension=float(data.get("tension", 0.0)),
            irony=float(data.get("irony", 0.0)),
            significance=float(data.get("significance", 0.0)),
            thematic_shift={str(k): float(v) for k, v in (data.get("thematic_shift") or {}).items()},
            tension_components={
                str(k): float(v) for k, v in (data.get("tension_components") or {}).items()
            },
            irony_collapse=IronyCollapseInfo.from_dict(irony_collapse)
            if isinstance(irony_collapse, dict)
            else None,
        )


@dataclass
class EventEntities:
    """Typed references to non-agent world entities touched by an event."""

    locations: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    factions: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.locations:
            data["locations"] = list(self.locations)
        if self.artifacts:
            data["artifacts"] = list(self.artifacts)
        if self.factions:
            data["factions"] = list(self.factions)
        if self.institutions:
            data["institutions"] = list(self.institutions)
        if self.claims:
            data["claims"] = list(self.claims)
        if self.concepts:
            data["concepts"] = list(self.concepts)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EventEntities":
        if not data:
            return cls()
        return cls(
            locations=[str(x) for x in (data.get("locations") or [])],
            artifacts=[str(x) for x in (data.get("artifacts") or [])],
            factions=[str(x) for x in (data.get("factions") or [])],
            institutions=[str(x) for x in (data.get("institutions") or [])],
            claims=[str(x) for x in (data.get("claims") or [])],
            concepts=[str(x) for x in (data.get("concepts") or [])],
        )

    @property
    def total_refs(self) -> int:
        return (
            len(self.locations)
            + len(self.artifacts)
            + len(self.factions)
            + len(self.institutions)
            + len(self.claims)
            + len(self.concepts)
        )


@dataclass
class Event:
    """A single event node in the event-primary graph (specs/schema/events.md Section 6)."""

    id: str
    sim_time: float
    tick_id: int
    order_in_tick: int
    type: EventType
    source_agent: str
    target_agents: list[str]
    location_id: str
    causal_links: list[str]
    deltas: list[StateDelta]
    description: str
    dialogue: Optional[str] = None
    content_metadata: Optional[dict[str, Any]] = None
    beat_type: Optional[BeatType] = None
    metrics: EventMetrics = field(default_factory=EventMetrics)
    entities: EventEntities = field(default_factory=EventEntities)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "sim_time": self.sim_time,
            "tick_id": self.tick_id,
            "order_in_tick": self.order_in_tick,
            "type": self.type.value,
            "source_agent": self.source_agent,
            "target_agents": list(self.target_agents),
            "location_id": self.location_id,
            "causal_links": list(self.causal_links),
            "deltas": [d.to_dict() for d in self.deltas],
            "description": self.description,
            "dialogue": self.dialogue,
            "content_metadata": self.content_metadata,
            "beat_type": self.beat_type.value if self.beat_type else None,
            "metrics": self.metrics.to_dict(),
        }
        entities = self.entities.to_dict()
        if entities:
            data["entities"] = entities
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            id=str(data["id"]),
            sim_time=float(data["sim_time"]),
            tick_id=int(data["tick_id"]),
            order_in_tick=int(data.get("order_in_tick", 0)),
            type=EventType(data["type"]),
            source_agent=str(data["source_agent"]),
            target_agents=[str(x) for x in (data.get("target_agents") or [])],
            location_id=str(data["location_id"]),
            causal_links=[str(x) for x in (data.get("causal_links") or [])],
            deltas=[StateDelta.from_dict(d) for d in (data.get("deltas") or [])],
            description=str(data.get("description", "")),
            dialogue=data.get("dialogue"),
            content_metadata=data.get("content_metadata"),
            beat_type=BeatType(data["beat_type"]) if data.get("beat_type") else None,
            metrics=EventMetrics.from_dict(data.get("metrics") or {}),
            entities=EventEntities.from_dict(data.get("entities")),
        )
