from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FlawType(Enum):
    PRIDE = "pride"
    LOYALTY = "loyalty"
    TRAUMA = "trauma"
    AMBITION = "ambition"
    JEALOUSY = "jealousy"
    COWARDICE = "cowardice"
    VANITY = "vanity"
    GUILT = "guilt"
    OBSESSION = "obsession"
    DENIAL = "denial"


@dataclass(frozen=True)
class CharacterFlaw:
    flaw_type: FlawType
    strength: float
    trigger: str
    effect: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "flaw_type": self.flaw_type.value,
            "strength": self.strength,
            "trigger": self.trigger,
            "effect": self.effect,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterFlaw":
        return cls(
            flaw_type=FlawType(data["flaw_type"]),
            strength=float(data.get("strength", 0.0)),
            trigger=str(data.get("trigger", "")),
            effect=str(data.get("effect", "")),
            description=str(data.get("description", "")),
        )


@dataclass
class GoalVector:
    safety: float = 0.5
    status: float = 0.5
    closeness: dict[str, float] = field(default_factory=dict)
    secrecy: float = 0.5
    truth_seeking: float = 0.5
    autonomy: float = 0.5
    loyalty: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety": self.safety,
            "status": self.status,
            "closeness": dict(self.closeness),
            "secrecy": self.secrecy,
            "truth_seeking": self.truth_seeking,
            "autonomy": self.autonomy,
            "loyalty": self.loyalty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalVector":
        return cls(
            safety=float(data.get("safety", 0.5)),
            status=float(data.get("status", 0.5)),
            closeness={str(k): float(v) for k, v in (data.get("closeness") or {}).items()},
            secrecy=float(data.get("secrecy", 0.5)),
            truth_seeking=float(data.get("truth_seeking", 0.5)),
            autonomy=float(data.get("autonomy", 0.5)),
            loyalty=float(data.get("loyalty", 0.5)),
        )


@dataclass
class RelationshipState:
    trust: float = 0.0
    affection: float = 0.0
    obligation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"trust": self.trust, "affection": self.affection, "obligation": self.obligation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RelationshipState":
        return cls(
            trust=float(data.get("trust", 0.0)),
            affection=float(data.get("affection", 0.0)),
            obligation=float(data.get("obligation", 0.0)),
        )


@dataclass
class PacingState:
    dramatic_budget: float = 1.0
    stress: float = 0.0
    composure: float = 1.0
    commitment: float = 0.0
    recovery_timer: int = 0
    suppression_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dramatic_budget": self.dramatic_budget,
            "stress": self.stress,
            "composure": self.composure,
            "commitment": self.commitment,
            "recovery_timer": self.recovery_timer,
            "suppression_count": self.suppression_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PacingState":
        return cls(
            dramatic_budget=float(data.get("dramatic_budget", 1.0)),
            stress=float(data.get("stress", 0.0)),
            composure=float(data.get("composure", 1.0)),
            commitment=float(data.get("commitment", 0.0)),
            recovery_timer=int(data.get("recovery_timer", 0)),
            suppression_count=int(data.get("suppression_count", 0)),
        )


class BeliefState(Enum):
    UNKNOWN = "unknown"
    SUSPECTS = "suspects"
    BELIEVES_TRUE = "believes_true"
    BELIEVES_FALSE = "believes_false"


@dataclass
class AgentState:
    id: str
    name: str
    location: str

    goals: GoalVector
    flaws: list[CharacterFlaw]
    pacing: PacingState
    emotional_state: dict[str, float]

    relationships: dict[str, RelationshipState]
    beliefs: dict[str, BeliefState]

    alcohol_level: float = 0.0
    commitments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "goals": self.goals.to_dict(),
            "flaws": [f.to_dict() for f in self.flaws],
            "pacing": self.pacing.to_dict(),
            "emotional_state": dict(self.emotional_state),
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
            "beliefs": {k: v.value for k, v in self.beliefs.items()},
            "alcohol_level": self.alcohol_level,
            "commitments": list(self.commitments),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            location=str(data.get("location", "")),
            goals=GoalVector.from_dict(data.get("goals") or {}),
            flaws=[CharacterFlaw.from_dict(f) for f in (data.get("flaws") or [])],
            pacing=PacingState.from_dict(data.get("pacing") or {}),
            emotional_state={str(k): float(v) for k, v in (data.get("emotional_state") or {}).items()},
            relationships={
                str(k): RelationshipState.from_dict(v)
                for k, v in (data.get("relationships") or {}).items()
            },
            beliefs={str(k): BeliefState(v) for k, v in (data.get("beliefs") or {}).items()},
            alcohol_level=float(data.get("alcohol_level", 0.0)),
            commitments=[str(x) for x in (data.get("commitments") or [])],
        )

