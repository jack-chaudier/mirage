from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonFact:
    """A simulation-grounded fact phrased for narrative continuity."""

    id: str
    statement: str
    source_event_ids: list[str]
    entity_refs: list[str]
    scene_index: int


@dataclass(frozen=True)
class TextureFact:
    """An LLM-originated detail intended to persist for consistency."""

    id: str
    statement: str
    entity_refs: list[str]
    detail_type: str
    scene_index: int
    confidence: float = 1.0


@dataclass
class SceneLoreUpdates:
    """Lore extracted from one generated scene."""

    scene_index: int
    canon_facts: list[CanonFact] = field(default_factory=list)
    texture_facts: list[TextureFact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": int(self.scene_index),
            "canon_facts": [
                {
                    "id": f.id,
                    "statement": f.statement,
                    "source_event_ids": list(f.source_event_ids),
                    "entity_refs": list(f.entity_refs),
                    "scene_index": int(f.scene_index),
                }
                for f in self.canon_facts
            ],
            "texture_facts": [
                {
                    "id": f.id,
                    "statement": f.statement,
                    "entity_refs": list(f.entity_refs),
                    "detail_type": f.detail_type,
                    "scene_index": int(f.scene_index),
                    "confidence": float(f.confidence),
                }
                for f in self.texture_facts
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SceneLoreUpdates:
        if not data:
            return cls(scene_index=0)

        canon_facts = [
            CanonFact(
                id=str(raw.get("id", "")),
                statement=str(raw.get("statement", "")),
                source_event_ids=[str(x) for x in (raw.get("source_event_ids") or [])],
                entity_refs=[str(x) for x in (raw.get("entity_refs") or [])],
                scene_index=int(raw.get("scene_index", data.get("scene_index", 0)) or 0),
            )
            for raw in (data.get("canon_facts") or [])
            if isinstance(raw, dict)
        ]
        texture_facts = [
            TextureFact(
                id=str(raw.get("id", "")),
                statement=str(raw.get("statement", "")),
                entity_refs=[str(x) for x in (raw.get("entity_refs") or [])],
                detail_type=str(raw.get("detail_type", "")),
                scene_index=int(raw.get("scene_index", data.get("scene_index", 0)) or 0),
                confidence=float(raw.get("confidence", 1.0) or 1.0),
            )
            for raw in (data.get("texture_facts") or [])
            if isinstance(raw, dict)
        ]
        return cls(
            scene_index=int(data.get("scene_index", 0) or 0),
            canon_facts=canon_facts,
            texture_facts=texture_facts,
        )


@dataclass
class StoryLore:
    """Accumulated scene-level lore for one generated story."""

    scene_lore: list[SceneLoreUpdates] = field(default_factory=list)

    @property
    def all_canon_facts(self) -> list[CanonFact]:
        return [fact for scene in self.scene_lore for fact in scene.canon_facts]

    @property
    def all_texture_facts(self) -> list[TextureFact]:
        return [fact for scene in self.scene_lore for fact in scene.texture_facts]

    def to_dict(self) -> dict[str, Any]:
        return {"scene_lore": [scene.to_dict() for scene in self.scene_lore]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StoryLore:
        if not data:
            return cls()
        return cls(
            scene_lore=[
                SceneLoreUpdates.from_dict(raw)
                for raw in (data.get("scene_lore") or [])
                if isinstance(raw, dict)
            ]
        )
