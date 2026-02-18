from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.agents import AgentState, FlawType, GoalVector, RelationshipState
from narrativefield.schema.world import Location, SecretDefinition, WorldDefinition


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    """Static character information for prompt injection."""

    agent_id: str
    name: str
    description: str  # 2-3 sentence personality description
    primary_flaw: str  # from the FlawType
    flaw_description: str  # narrative description of how the flaw manifests
    voice_notes: str  # 1-2 sentence voice direction for dialogue generation
    key_relationships: dict[str, str]  # agent_id -> "relationship description"
    secrets_held: list[str]  # secret descriptions (not IDs)
    goal_summary: str  # derived from GoalVector for MVP


@dataclass(frozen=True, slots=True)
class LocationProfile:
    """Static location information for prompt injection."""

    location_id: str
    name: str
    description: str  # 1-2 sentence description
    privacy_level: str  # "public", "semi-private", "private"
    narrative_function: str  # "main stage", "escape", "intimate space", etc.


class Lorebook:
    """Context injection system for the prose generation pipeline."""

    def __init__(
        self,
        world_definition: WorldDefinition,
        agents: list[AgentState],
        secrets: list[SecretDefinition],
        canon: WorldCanon | None = None,
    ):
        self._characters: dict[str, CharacterProfile] = {}
        self._locations: dict[str, LocationProfile] = {}
        self._name_by_id: dict[str, str] = {a.id: a.name for a in agents}
        self._canon = canon
        self._build(world_definition, agents, secrets)

    def get_context_for_scene(self, character_ids: list[str], location_id: str) -> str:
        """Return XML-formatted context for characters and location in this scene."""

        ids: list[str] = []
        seen: set[str] = set()
        for cid in character_ids:
            if cid in seen:
                continue
            seen.add(cid)
            if cid in self._characters:
                ids.append(cid)

        lines: list[str] = ["<lorebook>"]

        loc = self._locations.get(location_id)
        if loc is not None:
            loc_text = f"{loc.description} Privacy: {self._privacy_note(loc.privacy_level)} Narrative role: {loc.narrative_function}."
            lines.append(
                f'  <location name="{_xml_attr(loc.location_id)}">{_xml_escape(_truncate_chars(loc_text, 420))}</location>'
            )

        # Budgeting: keep total under ~800 tokens (1 token ~ 4 chars).
        max_total_chars = 800 * 4
        overhead = len("\n".join(lines)) + len("\n</lorebook>")
        available = max(0, max_total_chars - overhead)
        per_char_budget = available // max(1, len(ids)) if ids else 0
        per_char_budget = max(220, min(700, per_char_budget)) if ids else 0

        for cid in ids:
            prof = self._characters[cid]
            text = _render_character_profile(prof, self._name_by_id)
            # Enforce hard per-character constraints.
            text = _truncate_words(text, 150)
            text = _truncate_chars(text, per_char_budget)
            lines.append(
                f'  <character name="{_xml_attr(prof.name)}" id="{_xml_attr(prof.agent_id)}">{_xml_escape(text)}</character>'
            )

        lines.append("</lorebook>")
        return "\n".join(lines)

    def get_canon_context_for_scene(self, character_ids: list[str], location_id: str) -> str:
        """Return XML context from canon state for the active scene.

        Includes location memory, inherited claim beliefs, and established texture details.
        Returns an empty string when no canon context is available.
        """

        if self._canon is None:
            return ""

        location_lines: list[str] = []
        belief_lines: list[str] = []
        texture_lines: list[str] = []

        # 1) Location memory.
        loc_mem = self._canon.location_memory.get(location_id)
        if loc_mem is not None and float(loc_mem.tension_residue) > 0.05:
            hints: list[str] = []
            if float(loc_mem.tension_residue) > 0.60:
                hints.append("High residual tension from recent conflict.")
            elif float(loc_mem.tension_residue) > 0.30:
                hints.append("Recent events still weigh on this room.")
            if loc_mem.notable_event_ids:
                hints.append("People remember what happened here.")
            location_lines.append(
                f'  <location_memory location="{_xml_attr(location_id)}" '
                f'tension_residue="{float(loc_mem.tension_residue):.2f}" '
                f'visit_count="{int(loc_mem.visit_count)}" '
                f'notable_events="{int(len(loc_mem.notable_event_ids))}">'
                f"{_xml_escape(' '.join(hints) or 'This location carries accumulated narrative residue.')}"
                "</location_memory>"
            )

        # 2) Inherited claim beliefs by present characters.
        normalized_ids = [cid for cid in character_ids if cid]
        for char_id in normalized_ids:
            belief_pairs: list[str] = []
            for claim_id in sorted(self._canon.claim_states.keys()):
                state = (self._canon.claim_states.get(claim_id) or {}).get(char_id, "")
                state_l = str(state).strip().lower()
                if not state_l or state_l == "unknown":
                    continue
                belief_pairs.append(f"{claim_id}: {state_l}")
            if not belief_pairs:
                continue
            visible = "; ".join(belief_pairs[:5])
            char_name = self._name_by_id.get(char_id, char_id)
            belief_lines.append(
                f'  <inherited_knowledge character="{_xml_attr(char_id)}" '
                f'name="{_xml_attr(char_name)}">'
                f"{_xml_escape(f'{char_name} already knows: {visible}. This knowledge predates this evening.')}"
                "</inherited_knowledge>"
            )

        # 3) Established texture details for entities in this scene.
        relevant_refs = set(normalized_ids)
        relevant_refs.add(location_id)
        for _, tex in sorted(self._canon.texture.items()):
            if not any(ref in relevant_refs for ref in tex.entity_refs):
                continue
            texture_lines.append(
                f'  <established_detail type="{_xml_attr(tex.detail_type)}">'
                f"{_xml_escape(tex.statement)}"
                "</established_detail>"
            )
            if len(texture_lines) >= 8:
                break

        lines = ["<world_memory>", *location_lines, *belief_lines, *texture_lines, "</world_memory>"]

        if len(lines) <= 2:
            return ""

        # Token budget proxy: ~1600 chars.
        if len("\n".join(lines)) > 1600 and texture_lines:
            while texture_lines and len("\n".join(["<world_memory>", *location_lines, *belief_lines, *texture_lines, "</world_memory>"])) > 1600:
                texture_lines.pop()

        if len("\n".join(["<world_memory>", *location_lines, *belief_lines, *texture_lines, "</world_memory>"])) > 1600 and belief_lines:
            while belief_lines and len("\n".join(["<world_memory>", *location_lines, *belief_lines, *texture_lines, "</world_memory>"])) > 1600:
                belief_lines.pop()

        final_lines = ["<world_memory>", *location_lines, *belief_lines, *texture_lines, "</world_memory>"]
        if len(final_lines) <= 2:
            return ""
        return "\n".join(final_lines)

    def get_full_cast(self) -> str:
        """Return abbreviated profiles for ALL characters (for system prompt)."""

        lines: list[str] = ["<full_cast>"]
        for agent_id in sorted(self._characters.keys()):
            prof = self._characters[agent_id]
            # Very compact: name + flaw + one-sentence trait.
            text = (
                f"Flaw: {prof.primary_flaw}. "
                f"{prof.description.split('.', 1)[0].strip()}. "
                f"Voice: {prof.voice_notes.split('.', 1)[0].strip()}."
            )
            text = _truncate_words(text, 40)
            lines.append(
                f'  <character name="{_xml_attr(prof.name)}" id="{_xml_attr(prof.agent_id)}">{_xml_escape(text)}</character>'
            )
        lines.append("</full_cast>")
        return "\n".join(lines)

    def _build(self, world_definition: WorldDefinition, agents: list[AgentState], secrets: list[SecretDefinition]) -> None:
        # Locations
        for loc_id, loc in sorted(world_definition.locations.items(), key=lambda kv: kv[0]):
            privacy_level = _privacy_level(loc.privacy)
            self._locations[loc_id] = LocationProfile(
                location_id=loc.id,
                name=loc.name,
                description=loc.description.strip() or loc.name,
                privacy_level=privacy_level,
                narrative_function=_narrative_function(loc),
            )

        # Secrets: accept explicit list, but fall back to world definition if absent.
        secrets_list = list(secrets) if secrets else list(world_definition.secrets.values())
        secrets_by_holder: dict[str, list[str]] = {}
        for s in secrets_list:
            for holder in s.holder:
                secrets_by_holder.setdefault(holder, []).append(s.description)

        # Characters
        for a in agents:
            primary = _primary_flaw(a)
            flaw_name = primary.flaw_type.name if primary is not None else FlawType.DENIAL.name
            flaw_desc = (primary.description if primary is not None else "").strip() or "Keeps uncomfortable truths at bay."

            goal_summary = _goal_summary(a.goals, self._name_by_id)

            relationships = _key_relationships(a.relationships)
            description = _character_description(a, flaw_name, flaw_desc, goal_summary)

            self._characters[a.id] = CharacterProfile(
                agent_id=a.id,
                name=a.name,
                description=description,
                primary_flaw=flaw_name,
                flaw_description=flaw_desc,
                voice_notes=_voice_notes_for_agent(a),
                key_relationships=relationships,
                secrets_held=sorted(secrets_by_holder.get(a.id, [])),
                goal_summary=goal_summary,
            )

    def _privacy_note(self, privacy_level: str) -> str:
        if privacy_level == "public":
            return "public - conversations easily overheard."
        if privacy_level == "semi-private":
            return "semi-private - some discretion possible."
        return "private - intimate conversations possible."


def _xml_attr(value: str) -> str:
    return _xml_escape(str(value or ""), {'"': "&quot;"})


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:.") + "..."


def _truncate_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = max_chars - 3
    if cut <= 0:
        return "..."
    snippet = text[:cut]
    # Avoid breaking in the middle of a word if possible.
    if " " in snippet:
        snippet = snippet.rsplit(" ", 1)[0]
    return snippet.rstrip(",;:.") + "..."


def _privacy_level(privacy: float) -> str:
    if privacy <= 0.2:
        return "public"
    if privacy <= 0.5:
        return "semi-private"
    return "private"


def _narrative_function(loc: Location) -> str:
    # Lightweight heuristic: big rooms are "stage"; high privacy is "intimate/escape".
    if loc.capacity >= 5:
        return "main stage"
    if loc.privacy >= 0.8 and loc.capacity <= 1:
        return "escape"
    if loc.privacy > 0.5:
        return "intimate space"
    if loc.privacy > 0.2:
        return "side stage"
    return "transition space"


def _primary_flaw(agent: AgentState) -> Any:
    if not agent.flaws:
        return None
    return max(agent.flaws, key=lambda f: float(getattr(f, "strength", 0.0) or 0.0))


def _goal_summary(goals: GoalVector, name_by_id: dict[str, str]) -> str:
    tags: list[str] = []
    if goals.status >= 0.7:
        tags.append("status")
    if goals.safety >= 0.7:
        tags.append("safety")
    if goals.secrecy >= 0.7:
        tags.append("secrecy")
    if goals.truth_seeking >= 0.7:
        tags.append("truth")
    if goals.autonomy >= 0.7:
        tags.append("autonomy")
    if goals.loyalty >= 0.7:
        tags.append("loyalty")

    if not tags:
        tags.append("balance")

    close_target = ""
    if goals.closeness:
        cid, val = max(goals.closeness.items(), key=lambda kv: float(kv[1]))
        if float(val) > 0.55:
            close_target = name_by_id.get(cid, cid)

    parts: list[str] = [f"Priorities: {', '.join(tags)}"]
    if close_target:
        parts.append(f"seeks closeness with {close_target}")
    if goals.truth_seeking <= 0.3:
        parts.append("avoids digging for truth")
    return "; ".join(parts) + "."


def _key_relationships(relationships: dict[str, RelationshipState]) -> dict[str, str]:
    # Keep this compact: choose top relationships by absolute intensity.
    def score(rel: RelationshipState) -> float:
        return abs(rel.trust) + abs(rel.affection) + abs(rel.obligation)

    ranked = sorted(relationships.items(), key=lambda kv: score(kv[1]), reverse=True)
    top = ranked[:3]

    out: dict[str, str] = {}
    for other_id, rel in top:
        out[other_id] = _relationship_phrase(rel)
    return out


def _relationship_phrase(rel: RelationshipState) -> str:
    trust = float(rel.trust)
    affection = float(rel.affection)
    obligation = float(rel.obligation)

    if trust >= 0.55:
        base = "trusts"
    elif trust <= -0.3:
        base = "mistrusts"
    else:
        base = "is wary of"

    extras: list[str] = []
    if affection >= 0.55:
        extras.append("is close to")
    elif affection <= -0.3:
        extras.append("resents")

    if obligation >= 0.55:
        extras.append("feels indebted to")

    if extras:
        return base + " / " + ", ".join(extras)
    return base


def _character_description(agent: AgentState, flaw_name: str, flaw_desc: str, goal_summary: str) -> str:
    # Two short sentences; keep it prompt-friendly.
    flaw_phrase = flaw_name.lower().replace("_", " ")
    first = f"{agent.name} is shaped by {flaw_phrase}: {flaw_desc.strip()}"
    second = f"{goal_summary.strip()}"
    return f"{first} {second}".strip()


def _voice_notes_for_agent(agent: AgentState) -> str:
    # Keep this deterministic and compact so prompts remain stable across runs.
    voices: dict[str, str] = {
        "thorne": (
            "Thorne speaks in clipped, high-status statements and strategic questions. "
            "He implies pressure more often than he names it directly."
        ),
        "elena": (
            "Elena starts cautiously, hedging and self-correcting when cornered. "
            "When pushed, her sentences fragment before she lands on hard truths."
        ),
        "marcus": (
            "Marcus keeps a measured, practical cadence and avoids emotional flourishes. "
            "He answers narrowly, then redirects to logistics or plausibility."
        ),
        "lydia": (
            "Lydia uses polished social language to de-escalate and preserve decorum. "
            "Under stress, politeness becomes sharper and more controlling."
        ),
        "diana": (
            "Diana speaks with precise observational detail and analytical framing. "
            "She asks clarifying questions that expose contradictions without shouting."
        ),
        "victor": (
            "Victor uses probing, interrogative sentences and strategic pauses. "
            "He advances by implication, forcing others to fill silence with admissions."
        ),
    }
    return voices.get(
        agent.id,
        (
            f"{agent.name} speaks in concise, situation-aware sentences. "
            "Their diction should reflect pressure and social subtext."
        ),
    )


def _render_character_profile(profile: CharacterProfile, name_by_id: dict[str, str]) -> str:
    rel_parts: list[str] = []
    for other_id, rel_desc in profile.key_relationships.items():
        other_name = name_by_id.get(other_id, other_id)
        rel_parts.append(f"{other_name}: {rel_desc}")

    secrets = profile.secrets_held[:2]
    secrets_part = ""
    if secrets:
        secrets_part = " Secrets held: " + " | ".join(secrets) + "."

    rel_part = ""
    if rel_parts:
        rel_part = " Relationships: " + " ; ".join(rel_parts) + "."

    return (
        f"{profile.description.strip()} "
        f"Primary flaw: {profile.primary_flaw} ({profile.flaw_description.strip()}). "
        f"Voice: {profile.voice_notes.strip()} "
        f"{profile.goal_summary.strip()}"
        f"{rel_part}"
        f"{secrets_part}"
    ).strip()
