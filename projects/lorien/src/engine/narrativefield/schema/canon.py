from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocationMemory:
    """Accumulated memory of what happened at a location."""

    tension_residue: float = 0.0
    notable_event_ids: list[str] = field(default_factory=list)
    visit_count: int = 0
    last_event_tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tension_residue": float(self.tension_residue),
            "notable_event_ids": [str(x) for x in self.notable_event_ids],
            "visit_count": int(self.visit_count),
            "last_event_tick": int(self.last_event_tick),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LocationMemory":
        if not data:
            return cls()
        return cls(
            tension_residue=float(data.get("tension_residue", 0.0)),
            notable_event_ids=[str(x) for x in (data.get("notable_event_ids") or [])],
            visit_count=int(data.get("visit_count", 0)),
            last_event_tick=int(data.get("last_event_tick", 0)),
        )


@dataclass
class CanonArtifact:
    """A persistent world object (document, weapon, symbol, etc.)."""

    id: str
    name: str
    artifact_type: str
    state: dict[str, Any] = field(default_factory=dict)
    first_seen_event_id: str = ""
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "artifact_type": self.artifact_type,
            "state": dict(self.state),
            "first_seen_event_id": self.first_seen_event_id,
            "history": [str(x) for x in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CanonArtifact":
        if not data:
            return cls(id="", name="", artifact_type="")
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            artifact_type=str(data.get("artifact_type", "")),
            state=dict(data.get("state") or {}),
            first_seen_event_id=str(data.get("first_seen_event_id", "")),
            history=[str(x) for x in (data.get("history") or [])],
        )


@dataclass
class CanonFaction:
    """A group, alliance, or organization."""

    id: str
    name: str
    state: dict[str, float] = field(default_factory=dict)
    members: list[str] = field(default_factory=list)
    relationships: dict[str, float] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": {str(k): float(v) for k, v in self.state.items()},
            "members": [str(x) for x in self.members],
            "relationships": {str(k): float(v) for k, v in self.relationships.items()},
            "history": [str(x) for x in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CanonFaction":
        if not data:
            return cls(id="", name="")
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            state={str(k): float(v) for k, v in (data.get("state") or {}).items()},
            members=[str(x) for x in (data.get("members") or [])],
            relationships={
                str(k): float(v) for k, v in (data.get("relationships") or {}).items()
            },
            history=[str(x) for x in (data.get("history") or [])],
        )


@dataclass
class CanonInstitution:
    """A persistent structural entity (guild, court, church, market)."""

    id: str
    name: str
    state: dict[str, float] = field(default_factory=dict)
    policies: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": {str(k): float(v) for k, v in self.state.items()},
            "policies": [str(x) for x in self.policies],
            "history": [str(x) for x in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CanonInstitution":
        if not data:
            return cls(id="", name="")
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            state={str(k): float(v) for k, v in (data.get("state") or {}).items()},
            policies=[str(x) for x in (data.get("policies") or [])],
            history=[str(x) for x in (data.get("history") or [])],
        )


@dataclass
class CanonTexture:
    """A validated texture detail committed to world canon."""

    id: str
    statement: str
    entity_refs: list[str]
    detail_type: str
    source_story_id: str = ""
    source_scene_index: int = 0
    committed_at_canon_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "entity_refs": [str(x) for x in self.entity_refs],
            "detail_type": self.detail_type,
            "source_story_id": self.source_story_id,
            "source_scene_index": int(self.source_scene_index),
            "committed_at_canon_version": int(self.committed_at_canon_version),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CanonTexture":
        if not data:
            return cls(id="", statement="", entity_refs=[], detail_type="")
        return cls(
            id=str(data.get("id", "")),
            statement=str(data.get("statement", "")),
            entity_refs=[str(x) for x in (data.get("entity_refs") or [])],
            detail_type=str(data.get("detail_type", "")),
            source_story_id=str(data.get("source_story_id", "")),
            source_scene_index=int(data.get("source_scene_index", 0)),
            committed_at_canon_version=int(data.get("committed_at_canon_version", 0)),
        )


@dataclass
class WorldCanon:
    """Persistent, queryable world state materialized from the event stream."""

    world_id: str = ""
    canon_version: int = 0
    last_tick: int = 0
    last_event_id: str = ""

    location_memory: dict[str, LocationMemory] = field(default_factory=dict)

    artifacts: dict[str, CanonArtifact] = field(default_factory=dict)
    factions: dict[str, CanonFaction] = field(default_factory=dict)
    institutions: dict[str, CanonInstitution] = field(default_factory=dict)
    texture: dict[str, CanonTexture] = field(default_factory=dict)

    claim_states: dict[str, dict[str, str]] = field(default_factory=dict)
    claim_confidence: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "canon_version": int(self.canon_version),
            "last_tick": int(self.last_tick),
            "last_event_id": self.last_event_id,
            "location_memory": {k: v.to_dict() for k, v in self.location_memory.items()},
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "factions": {k: v.to_dict() for k, v in self.factions.items()},
            "institutions": {k: v.to_dict() for k, v in self.institutions.items()},
            "texture": {k: v.to_dict() for k, v in self.texture.items()},
            "claim_states": {
                str(claim_id): {str(agent_id): str(state) for agent_id, state in states.items()}
                for claim_id, states in self.claim_states.items()
            },
            "claim_confidence": {
                str(claim_id): {str(agent_id): float(conf) for agent_id, conf in confs.items()}
                for claim_id, confs in self.claim_confidence.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorldCanon":
        if not data:
            return cls()

        return cls(
            world_id=str(data.get("world_id", "")),
            canon_version=int(data.get("canon_version", 0)),
            last_tick=int(data.get("last_tick", 0)),
            last_event_id=str(data.get("last_event_id", "")),
            location_memory={
                str(loc_id): LocationMemory.from_dict(mem)
                for loc_id, mem in (data.get("location_memory") or {}).items()
            },
            artifacts={
                str(artifact_id): CanonArtifact.from_dict(artifact)
                for artifact_id, artifact in (data.get("artifacts") or {}).items()
            },
            factions={
                str(faction_id): CanonFaction.from_dict(faction)
                for faction_id, faction in (data.get("factions") or {}).items()
            },
            institutions={
                str(inst_id): CanonInstitution.from_dict(inst)
                for inst_id, inst in (data.get("institutions") or {}).items()
            },
            texture={
                str(texture_id): CanonTexture.from_dict(tex)
                for texture_id, tex in (data.get("texture") or {}).items()
            },
            claim_states={
                str(claim_id): {str(agent_id): str(state) for agent_id, state in (states or {}).items()}
                for claim_id, states in (data.get("claim_states") or {}).items()
            },
            claim_confidence={
                str(claim_id): {str(agent_id): float(conf) for agent_id, conf in (confs or {}).items()}
                for claim_id, confs in (data.get("claim_confidence") or {}).items()
            },
        )


def _validate_decay_rate(name: str, value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {rate}")
    return rate


def decay_canon(canon: WorldCanon, tension_decay: float = 0.6, belief_decay: float = 0.85) -> WorldCanon:
    """Apply exponential decay to canon state between chain steps.

    Args:
        canon: The canon to decay (modified in place and returned).
        tension_decay: Multiplier for location tension residue per step.
        belief_decay: Multiplier for belief confidence per step.

    Returns:
        The same canon object, mutated.
    """

    tension_rate = _validate_decay_rate("tension_decay", tension_decay)
    belief_rate = _validate_decay_rate("belief_decay", belief_decay)
    if tension_rate == 1.0 and belief_rate == 1.0:
        return canon

    for memory in canon.location_memory.values():
        decayed_tension = float(memory.tension_residue) * tension_rate
        memory.tension_residue = 0.0 if decayed_tension < 0.05 else decayed_tension

    for claim_id, states_by_agent in canon.claim_states.items():
        confidence_by_agent = canon.claim_confidence.setdefault(claim_id, {})
        for agent_id, belief_state in states_by_agent.items():
            if str(belief_state) == "unknown":
                confidence_by_agent.pop(agent_id, None)
                continue

            current_confidence = float(confidence_by_agent.get(agent_id, 1.0))
            decayed_confidence = current_confidence * belief_rate
            if decayed_confidence < 0.1:
                states_by_agent[agent_id] = "unknown"
                confidence_by_agent.pop(agent_id, None)
                continue

            confidence_by_agent[agent_id] = decayed_confidence

        if not confidence_by_agent:
            canon.claim_confidence.pop(claim_id, None)

    return canon
