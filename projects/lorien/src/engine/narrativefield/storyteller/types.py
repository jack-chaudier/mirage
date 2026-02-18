from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

from narrativefield.schema.lore import StoryLore


def _xml(tag: str, content: str, *, indent: str = "") -> str:
    return f"{indent}<{tag}>{_xml_escape(content)}</{tag}>"


@dataclass(slots=True)
class CharacterState:
    """Snapshot of a character's state at the current point in the story."""

    agent_id: str
    name: str
    location: str
    emotional_state: str
    current_goal: str
    knowledge: list[str]
    secrets_revealed: list[str]
    secrets_held: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "location": self.location,
            "emotional_state": self.emotional_state,
            "current_goal": self.current_goal,
            "knowledge": list(self.knowledge),
            "secrets_revealed": list(self.secrets_revealed),
            "secrets_held": list(self.secrets_held),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterState:
        return cls(
            agent_id=str(data.get("agent_id", "")),
            name=str(data.get("name", "")),
            location=str(data.get("location", "")),
            emotional_state=str(data.get("emotional_state", "")),
            current_goal=str(data.get("current_goal", "")),
            knowledge=[str(x) for x in (data.get("knowledge", []) or [])],
            secrets_revealed=[str(x) for x in (data.get("secrets_revealed", []) or [])],
            secrets_held=[str(x) for x in (data.get("secrets_held", []) or [])],
        )


@dataclass(slots=True)
class NarrativeThread:
    """An unresolved narrative thread that should be tracked."""

    description: str
    involved_agents: list[str]
    tension_level: float
    introduced_at_scene: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "involved_agents": list(self.involved_agents),
            "tension_level": float(self.tension_level),
            "introduced_at_scene": int(self.introduced_at_scene),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeThread:
        return cls(
            description=str(data.get("description", "")),
            involved_agents=[str(x) for x in (data.get("involved_agents", []) or [])],
            tension_level=float(data.get("tension_level", 0.0) or 0.0),
            introduced_at_scene=int(data.get("introduced_at_scene", 0) or 0),
        )


@dataclass(slots=True)
class NarrativeStateObject:
    """The rolling context passed between sequential generation calls.

    Design constraints (from research):
    - Keep total under 1,000 tokens to minimize per-call overhead
    - Cumulative summary is the primary compression mechanism
    - Last paragraph provides literal text continuity for smooth transitions
    - Character states track what each character KNOWS (for consistency)
    - Narrative plan enables foreshadowing across scene boundaries
    """

    summary_so_far: str
    last_paragraph: str
    current_scene_index: int
    characters: list[CharacterState]
    active_location: str
    unresolved_threads: list[NarrativeThread]
    narrative_plan: list[str]
    total_words_generated: int = 0
    scenes_completed: int = 0

    def to_prompt_xml(self) -> str:
        """Serialize to XML for insertion into Claude prompts.

        Uses XML tags because Anthropic's testing shows XML improves
        Claude's parsing accuracy for structured context.
        """

        lines: list[str] = ["<narrative_state>"]
        lines.append(_xml("summary_so_far", self.summary_so_far, indent="  "))
        lines.append(_xml("last_paragraph", self.last_paragraph, indent="  "))
        lines.append(_xml("current_scene_index", str(self.current_scene_index), indent="  "))
        lines.append(_xml("active_location", self.active_location, indent="  "))
        lines.append(_xml("total_words_generated", str(self.total_words_generated), indent="  "))
        lines.append(_xml("scenes_completed", str(self.scenes_completed), indent="  "))

        # Characters
        lines.append("  <characters>")
        for c in self.characters:
            lines.append("    <character>")
            lines.append(_xml("agent_id", c.agent_id, indent="      "))
            lines.append(_xml("name", c.name, indent="      "))
            lines.append(_xml("location", c.location, indent="      "))
            lines.append(_xml("emotional_state", c.emotional_state, indent="      "))
            lines.append(_xml("current_goal", c.current_goal, indent="      "))

            lines.append("      <knowledge>")
            for item in c.knowledge:
                lines.append(_xml("item", item, indent="        "))
            lines.append("      </knowledge>")

            lines.append("      <secrets_revealed>")
            for item in c.secrets_revealed:
                lines.append(_xml("item", item, indent="        "))
            lines.append("      </secrets_revealed>")

            lines.append("      <secrets_held>")
            for item in c.secrets_held:
                lines.append(_xml("item", item, indent="        "))
            lines.append("      </secrets_held>")

            lines.append("    </character>")
        lines.append("  </characters>")

        # Threads
        lines.append("  <unresolved_threads>")
        for t in self.unresolved_threads:
            lines.append("    <thread>")
            lines.append(_xml("description", t.description, indent="      "))
            lines.append(_xml("tension_level", str(t.tension_level), indent="      "))
            lines.append(_xml("introduced_at_scene", str(t.introduced_at_scene), indent="      "))
            lines.append("      <involved_agents>")
            for a in t.involved_agents:
                lines.append(_xml("agent_id", a, indent="        "))
            lines.append("      </involved_agents>")
            lines.append("    </thread>")
        lines.append("  </unresolved_threads>")

        # Plan
        lines.append("  <narrative_plan>")
        for item in self.narrative_plan:
            lines.append(_xml("item", item, indent="    "))
        lines.append("  </narrative_plan>")

        lines.append("</narrative_state>")
        return "\n".join(lines)

    def estimate_tokens(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""

        return len(self.to_prompt_xml()) // 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_so_far": self.summary_so_far,
            "last_paragraph": self.last_paragraph,
            "current_scene_index": int(self.current_scene_index),
            "characters": [c.to_dict() for c in self.characters],
            "active_location": self.active_location,
            "unresolved_threads": [t.to_dict() for t in self.unresolved_threads],
            "narrative_plan": list(self.narrative_plan),
            "total_words_generated": int(self.total_words_generated),
            "scenes_completed": int(self.scenes_completed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeStateObject:
        return cls(
            summary_so_far=str(data.get("summary_so_far", "")),
            last_paragraph=str(data.get("last_paragraph", "")),
            current_scene_index=int(data.get("current_scene_index", 0) or 0),
            characters=[
                CharacterState.from_dict(x) for x in (data.get("characters", []) or []) if isinstance(x, dict)
            ],
            active_location=str(data.get("active_location", "")),
            unresolved_threads=[
                NarrativeThread.from_dict(x)
                for x in (data.get("unresolved_threads", []) or [])
                if isinstance(x, dict)
            ],
            narrative_plan=[str(x) for x in (data.get("narrative_plan", []) or [])],
            total_words_generated=int(data.get("total_words_generated", 0) or 0),
            scenes_completed=int(data.get("scenes_completed", 0) or 0),
        )


@dataclass(slots=True)
class SceneChunk:
    """A group of events that form a narrative scene."""

    scene_index: int
    events: list[Any]  # list[Event] — use Any to avoid circular import, type-check at runtime
    location: str
    time_start: float
    time_end: float
    characters_present: list[str]
    scene_type: str  # from the existing scene classification
    is_pivotal: bool  # True if contains TURNING_POINT, CATASTROPHE, or major REVEAL
    summary: str = ""  # filled by Phase 1


@dataclass(slots=True)
class SceneOutcome:
    """Diagnostics for a single scene generation attempt."""

    scene_index: int
    status: str  # "ok" | "failed" | "skipped"
    word_count: int
    error_type: str | None  # e.g. "overloaded_error", "BadRequestError"
    retries: int
    generation_time_s: float


@dataclass(slots=True)
class GenerationResult:
    """Output of the full prose generation pipeline."""

    status: str  # "complete" | "partial" | "failed"
    prose: str
    word_count: int
    scenes_generated: int
    scene_outcomes: list[SceneOutcome]
    final_state: NarrativeStateObject
    usage: dict[str, Any]
    generation_time_seconds: float
    checkpoint_path: str | None
    story_lore: StoryLore | None = None
