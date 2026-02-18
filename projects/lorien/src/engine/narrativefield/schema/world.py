from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    privacy: float
    capacity: int
    adjacent: list[str]
    overhear_from: list[str]
    overhear_probability: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "privacy": self.privacy,
            "capacity": self.capacity,
            "adjacent": list(self.adjacent),
            "overhear_from": list(self.overhear_from),
            "overhear_probability": self.overhear_probability,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Location":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            privacy=float(data.get("privacy", 0.0)),
            capacity=int(data.get("capacity", 0)),
            adjacent=[str(x) for x in (data.get("adjacent") or [])],
            overhear_from=[str(x) for x in (data.get("overhear_from") or [])],
            overhear_probability=float(data.get("overhear_probability", 0.0)),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class SecretDefinition:
    id: str
    holder: list[str]
    about: Optional[str]
    content_type: str
    description: str
    truth_value: bool
    initial_knowers: list[str]
    initial_suspecters: list[str]
    dramatic_weight: float
    reveal_consequences: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "holder": list(self.holder),
            "about": self.about,
            "content_type": self.content_type,
            "description": self.description,
            "truth_value": self.truth_value,
            "initial_knowers": list(self.initial_knowers),
            "initial_suspecters": list(self.initial_suspecters),
            "dramatic_weight": self.dramatic_weight,
            "reveal_consequences": self.reveal_consequences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecretDefinition":
        holder = data.get("holder") or []
        if isinstance(holder, str):
            # Keep a bit of backwards compatibility for older JSON examples.
            holder = [holder]
        return cls(
            id=str(data["id"]),
            holder=[str(x) for x in holder],
            about=str(data["about"]) if data.get("about") is not None else None,
            content_type=str(data.get("content_type", "")),
            description=str(data.get("description", "")),
            truth_value=bool(data.get("truth_value", False)),
            initial_knowers=[str(x) for x in (data.get("initial_knowers") or [])],
            initial_suspecters=[str(x) for x in (data.get("initial_suspecters") or [])],
            dramatic_weight=float(data.get("dramatic_weight", 0.0)),
            reveal_consequences=str(data.get("reveal_consequences", "")),
        )

    def to_claim(self) -> "ClaimDefinition":
        """Convert this secret into a claim for unified processing."""
        return ClaimDefinition(
            id=self.id,
            description=self.description,
            truth_status="true" if self.truth_value else "false",
            claim_type="secret",
            scope="private",
            holder=list(self.holder),
            about=self.about,
            content_type=self.content_type,
            source_event_ids=[],
            initial_knowers=list(self.initial_knowers),
            initial_suspecters=list(self.initial_suspecters),
            dramatic_weight=float(self.dramatic_weight),
            reveal_consequences=self.reveal_consequences,
            propagation_rate=0.0,
            decay_rate=0.0,
        )


@dataclass(frozen=True)
class ClaimDefinition:
    id: str
    description: str
    truth_status: str
    claim_type: str
    scope: str
    holder: list[str]
    about: Optional[str]
    content_type: str
    source_event_ids: list[str]
    initial_knowers: list[str]
    initial_suspecters: list[str]
    dramatic_weight: float
    reveal_consequences: str
    propagation_rate: float
    decay_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "truth_status": self.truth_status,
            "claim_type": self.claim_type,
            "scope": self.scope,
            "holder": list(self.holder),
            "about": self.about,
            "content_type": self.content_type,
            "source_event_ids": list(self.source_event_ids),
            "initial_knowers": list(self.initial_knowers),
            "initial_suspecters": list(self.initial_suspecters),
            "dramatic_weight": self.dramatic_weight,
            "reveal_consequences": self.reveal_consequences,
            "propagation_rate": self.propagation_rate,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimDefinition":
        holder = data.get("holder") or []
        if isinstance(holder, str):
            holder = [holder]
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            truth_status=str(data.get("truth_status", "unknown")),
            claim_type=str(data.get("claim_type", "rumor")),
            scope=str(data.get("scope", "private")),
            holder=[str(x) for x in holder],
            about=str(data["about"]) if data.get("about") is not None else None,
            content_type=str(data.get("content_type", "")),
            source_event_ids=[str(x) for x in (data.get("source_event_ids") or [])],
            initial_knowers=[str(x) for x in (data.get("initial_knowers") or [])],
            initial_suspecters=[str(x) for x in (data.get("initial_suspecters") or [])],
            dramatic_weight=float(data.get("dramatic_weight", 0.0)),
            reveal_consequences=str(data.get("reveal_consequences", "")),
            propagation_rate=float(data.get("propagation_rate", 0.0)),
            decay_rate=float(data.get("decay_rate", 0.0)),
        )


@dataclass
class WorldDefinition:
    id: str
    name: str
    description: str
    sim_duration_minutes: float
    ticks_per_minute: float

    locations: dict[str, Location]
    secrets: dict[str, SecretDefinition]
    claims: dict[str, ClaimDefinition] = field(default_factory=dict)
    seating: Optional[dict[str, list[str]]] = None

    primary_themes: list[str] = field(default_factory=list)

    snapshot_interval: int = 20
    catastrophe_threshold: float = 0.35
    composure_gate: float = 0.30
    trust_repair_multiplier: float = 3.0

    @property
    def all_claims(self) -> dict[str, ClaimDefinition]:
        """Unified view of claim-like state: converted secrets + explicit claims."""
        merged = {sid: secret.to_claim() for sid, secret in self.secrets.items()}
        merged.update(self.claims)
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sim_duration_minutes": self.sim_duration_minutes,
            "ticks_per_minute": self.ticks_per_minute,
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "secrets": {k: v.to_dict() for k, v in self.secrets.items()},
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "seating": self.seating,
            "primary_themes": list(self.primary_themes),
            "snapshot_interval": self.snapshot_interval,
            "catastrophe_threshold": self.catastrophe_threshold,
            "composure_gate": self.composure_gate,
            "trust_repair_multiplier": self.trust_repair_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldDefinition":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            sim_duration_minutes=float(data.get("sim_duration_minutes", 0.0)),
            ticks_per_minute=float(data.get("ticks_per_minute", 0.0)),
            locations={
                str(k): Location.from_dict(v) for k, v in (data.get("locations") or {}).items()
            },
            secrets={
                str(k): SecretDefinition.from_dict(v) for k, v in (data.get("secrets") or {}).items()
            },
            claims={
                str(k): ClaimDefinition.from_dict(v) for k, v in (data.get("claims") or {}).items()
            },
            seating=data.get("seating"),
            primary_themes=[str(x) for x in (data.get("primary_themes") or [])],
            snapshot_interval=int(data.get("snapshot_interval", 20)),
            catastrophe_threshold=float(data.get("catastrophe_threshold", 0.35)),
            composure_gate=float(data.get("composure_gate", 0.30)),
            trust_repair_multiplier=float(data.get("trust_repair_multiplier", 3.0)),
        )
