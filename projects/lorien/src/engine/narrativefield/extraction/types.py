from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from narrativefield.schema.events import BeatType, Event, EventType


@dataclass(frozen=True)
class ArcValidation:
    valid: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "violations": list(self.violations)}


@dataclass(frozen=True)
class BeatClassification:
    event_id: str
    beat_type: BeatType

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "beat_type": self.beat_type.value}


@dataclass(frozen=True)
class ArcScore:
    composite: float
    tension_variance: float
    peak_tension: float
    tension_shape: float
    significance: float
    thematic_coherence: float
    irony_arc: float
    protagonist_dominance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": float(self.composite),
            "tension_variance": float(self.tension_variance),
            "peak_tension": float(self.peak_tension),
            "tension_shape": float(self.tension_shape),
            "significance": float(self.significance),
            "thematic_coherence": float(self.thematic_coherence),
            "irony_arc": float(self.irony_arc),
            "protagonist_dominance": float(self.protagonist_dominance),
        }


@dataclass(frozen=True)
class ArcCandidate:
    """A candidate arc found by region search."""

    protagonist: str
    events: list[Event]
    beats: list[BeatType]
    validation: ArcValidation
    score: ArcScore
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protagonist": self.protagonist,
            "event_ids": [e.id for e in self.events],
            "beat_types": [b.value for b in self.beats],
            "is_valid": self.validation.valid,
            "score": self.score.to_dict(),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class CharacterBrief:
    agent_id: str
    name: str
    role_in_arc: str
    key_goal: str
    key_flaw: str
    key_secret: str | None
    emotional_start: str
    emotional_end: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role_in_arc": self.role_in_arc,
            "key_goal": self.key_goal,
            "key_flaw": self.key_flaw,
            "key_secret": self.key_secret,
            "emotional_start": self.emotional_start,
            "emotional_end": self.emotional_end,
        }


@dataclass(frozen=True)
class Beat:
    beat_type: BeatType
    event_id: str
    event_type: EventType
    scene_id: str | None
    location: str
    participants: list[str]
    description: str
    tension: float
    irony_note: str | None
    key_changes: list[str]
    emotional_states: dict[str, str]
    pov_suggestion: str
    tone_suggestion: str
    pacing_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_type": self.beat_type.value,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "scene_id": self.scene_id,
            "location": self.location,
            "participants": list(self.participants),
            "description": self.description,
            "tension": float(self.tension),
            "irony_note": self.irony_note,
            "key_changes": list(self.key_changes),
            "emotional_states": dict(self.emotional_states),
            "pov_suggestion": self.pov_suggestion,
            "tone_suggestion": self.tone_suggestion,
            "pacing_note": self.pacing_note,
        }


@dataclass(frozen=True)
class BeatSheet:
    arc_id: str
    protagonist: str
    title_suggestion: str
    genre_preset: str
    arc_score: ArcScore
    setting_summary: str
    time_span: str
    characters: list[CharacterBrief]
    beats: list[Beat]
    dominant_theme: str
    thematic_trajectory: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arc_id": self.arc_id,
            "protagonist": self.protagonist,
            "title_suggestion": self.title_suggestion,
            "genre_preset": self.genre_preset,
            "arc_score": self.arc_score.to_dict(),
            "setting_summary": self.setting_summary,
            "time_span": self.time_span,
            "characters": [c.to_dict() for c in self.characters],
            "beats": [b.to_dict() for b in self.beats],
            "dominant_theme": self.dominant_theme,
            "thematic_trajectory": self.thematic_trajectory,
        }
