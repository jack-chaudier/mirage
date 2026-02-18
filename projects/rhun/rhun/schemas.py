"""Domain-agnostic schemas for causal event graphs and constrained extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Phase(Enum):
    """Sequential phases for extracted sequences. Order matters."""

    SETUP = auto()
    DEVELOPMENT = auto()  # Equivalent to complication/escalation in narrative
    TURNING_POINT = auto()  # The max-weight element; phase boundary
    RESOLUTION = auto()  # Post-turning-point falling action

    def __lt__(self, other: Phase) -> bool:
        if not isinstance(other, Phase):
            return NotImplemented
        return self.value < other.value


@dataclass(frozen=True)
class Event:
    """A node in the causal event graph."""

    id: str
    timestamp: float  # Position in temporal order (absolute time)
    weight: float  # Importance/tension score (the objective)
    actors: frozenset[str] = frozenset()  # Which actors are involved
    causal_parents: tuple[str, ...] = ()  # IDs of events that caused this one
    metadata: dict[str, Any] = field(default_factory=dict)  # Domain-specific payload


@dataclass(frozen=True)
class CausalGraph:
    """A temporally ordered causal DAG with actor attribution and weights."""

    events: tuple[Event, ...]  # Sorted by timestamp
    actors: frozenset[str]  # All actors in the graph
    seed: int | None = None  # Random seed that generated this graph
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_actors(self) -> int:
        return len(self.actors)

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].timestamp - self.events[0].timestamp

    def events_for_actor(self, actor: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if actor in e.actors)

    def event_by_id(self, event_id: str) -> Event | None:
        for e in self.events:
            if e.id == event_id:
                return e
        return None

    def global_position(self, event: Event) -> float:
        """Normalized position in [0, 1] based on timestamp."""
        if self.duration == 0:
            return 0.5
        return (event.timestamp - self.events[0].timestamp) / self.duration

    def children(self, event_id: str) -> list[Event]:
        """Events that list event_id as a causal parent."""
        return [e for e in self.events if event_id in e.causal_parents]

    def build_adjacency(self) -> dict[str, list[str]]:
        """Forward adjacency: parent_id -> [child_ids]."""
        adj: dict[str, list[str]] = {e.id: [] for e in self.events}
        for e in self.events:
            for pid in e.causal_parents:
                if pid in adj:
                    adj[pid].append(e.id)
        return adj

    def build_reverse_adjacency(self) -> dict[str, list[str]]:
        """Reverse adjacency: child_id -> [parent_ids]."""
        rev: dict[str, list[str]] = {e.id: [] for e in self.events}
        for e in self.events:
            for pid in e.causal_parents:
                rev[e.id].append(pid)
        return rev


@dataclass(frozen=True)
class ExtractedSequence:
    """Result of extraction: a subsequence of events with phase labels."""

    events: tuple[Event, ...]
    phases: tuple[Phase, ...]  # Parallel to events
    focal_actor: str
    score: float = 0.0
    valid: bool = False
    violations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turning_point(self) -> Event | None:
        for e, p in zip(self.events, self.phases):
            if p == Phase.TURNING_POINT:
                return e
        return None

    @property
    def n_development(self) -> int:
        return sum(1 for p in self.phases if p == Phase.DEVELOPMENT)

    @property
    def n_events(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class ExtractionResult:
    """Complete result of multi-actor extraction from one graph."""

    graph: CausalGraph
    sequences: dict[str, ExtractedSequence]  # actor -> extracted sequence
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_count(self) -> int:
        return sum(1 for s in self.sequences.values() if s.valid)

    @property
    def all_valid(self) -> bool:
        return all(s.valid for s in self.sequences.values())

    @property
    def mean_score(self) -> float:
        valid_scores = [s.score for s in self.sequences.values() if s.valid]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    @property
    def validity_adjusted_score(self) -> float:
        n_actors = len(self.sequences)
        if n_actors == 0:
            return 0.0
        return self.mean_score * (self.valid_count / n_actors)
