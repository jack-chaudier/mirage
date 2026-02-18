from __future__ import annotations

from narrativefield.llm.config import PipelineConfig
from narrativefield.schema.lore import TextureFact
from narrativefield.storyteller.event_summarizer import summarize_event
from narrativefield.storyteller.lorebook import Lorebook
from narrativefield.storyteller.types import CharacterState, NarrativeStateObject, SceneChunk


def build_system_prompt(lorebook: Lorebook, config: PipelineConfig) -> str:
    """Build the static system prompt cached across all scene generation calls.

    Target: 1,500-2,500 tokens (~6,000-10,000 chars).
    """

    cast_xml = lorebook.get_full_cast()

    return f"""\
You are a literary fiction author writing a novella set at an intimate dinner party. \
Your prose is third-person limited, following the protagonist's perspective. \
Write in a style that is precise and evocative without being ornamental — \
literary fiction, not purple prose. Every sentence earns its place.

STYLE REQUIREMENTS:
- Show, don't tell. Render emotions through body language, micro-expressions, \
dialogue subtext, and physical sensation. Never name an emotion directly when you \
can show it.
- Dialogue must sound like each specific character. Vocabulary, rhythm, and deflection \
patterns differ per person. People interrupt, trail off, change subjects.
- Ground every scene in sensory detail: the clink of silverware, the taste of wine, \
the hum of conversation in the next room, the weight of a silence.
- Use internal monologue sparingly and only at moments of genuine psychological shift. \
When you do, make it feel involuntary — a thought the character cannot suppress.
- Scene transitions should read like chapter breaks: a beat of white space, then re-anchor \
in a new sensory detail before resuming the narrative.
- Pacing: slow down for pivotal moments (revelations, confrontations, catastrophes). \
Speed through transitional beats. Match prose density to dramatic weight.

CAST:
{cast_xml}

WORLD:
This is a contemporary dinner party at a well-appointed home. The evening spans roughly \
two to three hours. There are multiple rooms — a dining area, kitchen, balcony, foyer, \
and bathroom — each with different privacy levels. Characters move between locations; \
conversations can be overheard in adjacent rooms. The atmosphere begins civilized and \
deteriorates as secrets surface and alliances shift.

Tone: the tension of a Harold Pinter play crossed with the social observation of \
Sally Rooney. Menace lives in what is left unsaid.

OUTPUT FORMAT:
Return your response in exactly this structure:

<prose>
[Your story text here. Paragraphs separated by blank lines. No chapter headings or \
section markers unless the instructions specify a scene transition.]
</prose>
<state_update>
<summary>[Updated running summary of the story so far, incorporating this new scene. \
Keep factual and compressed — plot points, character knowledge changes, emotional arcs. \
Under {config.max_summary_words} words.]</summary>
<character_updates>
<character id="agent_id" emotional_state="current emotion" current_goal="what they want now" />
[One element per character who appeared in this scene.]
</character_updates>
<new_threads>
<thread description="what tension was introduced" involved="agent1,agent2" tension="0.0-1.0" />
[Only if genuinely new narrative threads emerged.]
</new_threads>
<resolved_threads>[Comma-separated indices of threads from the unresolved_threads list \
that were resolved in this scene, e.g. 0,2. Empty if none resolved.]</resolved_threads>
</state_update>
<lore_updates>
<canon_facts>
<fact event_ids="EVT_001,EVT_003">Natural-language restatement grounded in simulation events.</fact>
[2-5 significant facts. Only facts grounded in provided events.]
</canon_facts>
<texture_facts>
<detail type="gesture" entities="agent_id">Persistent invented detail for continuity.</detail>
[0-6 details. Type values: gesture, appearance, backstory, setting, relationship_history, habit, object.]
</texture_facts>
</lore_updates>

The <lore_updates> block is optional. If omitted, generation still succeeds."""


def build_scene_prompt(
    state: NarrativeStateObject,
    scene: SceneChunk,
    lorebook: Lorebook,
    upcoming_scenes: list[SceneChunk],
    config: PipelineConfig,
    accumulated_texture: list[TextureFact] | None = None,
    protagonist_id: str | None = None,
) -> str:
    """Build the dynamic user prompt for a single scene.

    Ordered for U-shaped attention (Liu et al., 2024):
    TOP = narrative state + continuity (high attention)
    MIDDLE = lorebook + upcoming preview (lower attention, reference)
    BOTTOM = events + instructions (high attention)
    """

    sections: list[str] = []

    # --- TOP: narrative state and continuity ---
    sections.append(state.to_prompt_xml())

    if state.summary_so_far.strip():
        sections.append(
            "<previously_established>\n"
            "Previously established facts (carry forward unless events explicitly change them):\n\n"
            f"{state.summary_so_far.strip()}\n"
            "</previously_established>"
        )
    if accumulated_texture:
        lines = [f"- {tf.statement}" for tf in accumulated_texture[:10] if tf.statement.strip()]
        if lines:
            sections.append(
                "<established_details>\n"
                "Details established in earlier scenes (maintain consistency):\n"
                + "\n".join(lines)
                + "\n</established_details>"
            )

    if state.last_paragraph:
        sections.append(
            "<continuity>\n"
            "The previous scene ended with this paragraph. Continue smoothly from here:\n\n"
            f"{state.last_paragraph}\n"
            "</continuity>"
        )

    # --- MIDDLE: lorebook context and upcoming preview ---
    lorebook_xml = lorebook.get_context_for_scene(
        scene.characters_present, scene.location
    )
    sections.append(lorebook_xml)
    canon_ctx = lorebook.get_canon_context_for_scene(
        scene.characters_present, scene.location
    )
    has_world_memory = bool(canon_ctx.strip())
    if has_world_memory:
        sections.append(canon_ctx)

    if upcoming_scenes:
        preview_lines = ["<upcoming>"]
        for upcoming in upcoming_scenes[:3]:
            chars = ", ".join(upcoming.characters_present[:4])
            pivotal_flag = " [PIVOTAL]" if upcoming.is_pivotal else ""
            preview_lines.append(
                f'  <scene index="{upcoming.scene_index}" type="{upcoming.scene_type}" '
                f'location="{upcoming.location}"{pivotal_flag} characters="{chars}" />'
            )
        preview_lines.append("</upcoming>")
        sections.append("\n".join(preview_lines))

    # --- BOTTOM: events and instructions ---
    event_lines = ["<events>"]
    for event in scene.events:
        summary = summarize_event(event)
        parts = [f'  <event id="{event.id}" type="{event.type.value}" time="{event.sim_time:.1f}">']
        parts.append(f"    <summary>{summary}</summary>")
        if event.dialogue:
            parts.append(f"    <dialogue>{event.dialogue}</dialogue>")
        if event.beat_type is not None:
            parts.append(f"    <beat_type>{event.beat_type.value}</beat_type>")
        metrics = event.metrics
        parts.append(
            f"    <metrics tension=\"{metrics.tension:.2f}\" "
            f"irony=\"{metrics.irony:.2f}\" "
            f'significance=\"{metrics.significance:.2f}\" />'
        )
        parts.append("  </event>")
        event_lines.append("\n".join(parts))
    event_lines.append("</events>")
    sections.append("\n".join(event_lines))

    # Compute target word count based on event count and pivotal status.
    base_words = min(config.phase2_max_words_per_chunk, max(500, len(scene.events) * 120))
    if scene.is_pivotal:
        target_words = min(config.phase2_max_words_per_chunk, int(base_words * 1.4))
    else:
        target_words = base_words

    # Build secrets-held list (what MUST NOT be revealed by the prose).
    secrets_held: list[str] = []
    for char in state.characters:
        unrevealed = [s for s in char.secrets_held if s not in char.secrets_revealed]
        if unrevealed:
            secrets_held.append(f"  {char.name}: {'; '.join(unrevealed)}")

    # Build unresolved threads list.
    thread_refs: list[str] = []
    for i, thread in enumerate(state.unresolved_threads):
        thread_refs.append(f"  [{i}] {thread.description} (tension: {thread.tension_level:.2f})")

    instruction_lines = ["<instructions>"]
    instruction_lines.append(f"Scene {scene.scene_index}: {scene.scene_type} at {scene.location}.")
    instruction_lines.append(f"Target word count: {target_words} words.")

    if scene.is_pivotal:
        instruction_lines.append(
            "This is a PIVOTAL scene. Slow down. More interiority, longer beats, "
            "sensory grounding. Let the weight of the moment land."
        )

    # Protagonist emotional arc hint from character states.
    protagonist_chars = [c for c in state.characters if c.agent_id in scene.characters_present]
    designated_pc: CharacterState | None = None
    if protagonist_id:
        for char in state.characters:
            if char.agent_id == protagonist_id:
                designated_pc = char
                break
    if protagonist_chars:
        pc = designated_pc or protagonist_chars[0]
        instruction_lines.append(
            f"Protagonist perspective: {pc.name} ({pc.emotional_state}). "
            f"Current goal: {pc.current_goal}."
        )
        instruction_lines.append(
            f"Write strictly from {pc.name}'s perspective. Do not render other characters' interior thoughts."
        )

    if has_world_memory:
        instruction_lines.append(
            "If world_memory context is present, characters may already know pre-evening facts. "
            "Reference inherited knowledge naturally instead of rediscovery."
        )

    if secrets_held:
        instruction_lines.append("Secrets that MUST NOT be revealed (characters still hold these):")
        instruction_lines.extend(secrets_held)

    if thread_refs:
        instruction_lines.append("Unresolved threads to weave in where natural:")
        instruction_lines.extend(thread_refs)

    instruction_lines.append(
        "Return your prose inside <prose> tags and your state update inside <state_update> tags."
    )
    instruction_lines.append("</instructions>")
    sections.append("\n".join(instruction_lines))

    return "\n\n".join(sections)


def build_summary_compression_prompt(
    old_summary: str,
    new_scene_prose: str,
    max_words: int = 500,
) -> str:
    """Build prompt for Grok to compress the running summary after a new scene."""

    return f"""\
You are a factual summarizer. Merge the existing story summary with the new scene below \
into a single compressed summary. Stay under {max_words} words.

Capture:
- Major plot developments and their consequences
- Each character's current emotional state and what they now know
- Unresolved tensions and open questions
- Physical positions of characters (who is where)

Do NOT use creative or literary language. Write in terse, factual sentences. \
Do NOT add interpretation or speculation. Only state what happened.

<existing_summary>
{old_summary}
</existing_summary>

<new_scene>
{new_scene_prose}
</new_scene>

Write the merged summary now. No preamble, no XML tags, just the summary text."""


def build_continuity_check_prompt(
    prose_chunk: str,
    events: list,
    character_states: list,
) -> str:
    """Build prompt for Grok to verify prose-to-simulation consistency.

    Returns structured JSON with consistency verdict and violations.
    """

    # Build event reference.
    event_lines: list[str] = []
    for event in events:
        summary = summarize_event(event) if hasattr(event, "type") else str(event)
        agents = [event.source_agent] + list(getattr(event, "target_agents", []))
        event_lines.append(
            f"  - [{event.id}] {summary} (location: {event.location_id}, "
            f"agents: {', '.join(agents)})"
        )

    # Build character knowledge reference.
    char_lines: list[str] = []
    for cs in character_states:
        if isinstance(cs, CharacterState):
            knowledge_str = ", ".join(cs.knowledge[:5]) if cs.knowledge else "none"
            secrets_str = ", ".join(cs.secrets_held) if cs.secrets_held else "none"
            revealed_str = ", ".join(cs.secrets_revealed) if cs.secrets_revealed else "none"
            char_lines.append(
                f"  - {cs.name} (at {cs.location}): knows [{knowledge_str}], "
                f"holds secrets [{secrets_str}], revealed [{revealed_str}], "
                f"emotional state: {cs.emotional_state}"
            )
        else:
            char_lines.append(f"  - {cs}")

    events_ref = "\n".join(event_lines) if event_lines else "  (none)"
    chars_ref = "\n".join(char_lines) if char_lines else "  (none)"

    return f"""\
You are a continuity checker for a prose generation pipeline. Compare the generated \
prose against the simulation data and flag any inconsistencies.

Check for these violation types:
1. KNOWLEDGE_VIOLATION: A character references information they should not know yet.
2. LOCATION_ERROR: A character appears in a location they are not at, or location \
details contradict the simulation.
3. TEMPORAL_ERROR: Events appear out of order, or time references are inconsistent.
4. EMOTIONAL_MISMATCH: A character's portrayed emotional state significantly contradicts \
the simulation data.

<prose>
{prose_chunk}
</prose>

<simulation_events>
{events_ref}
</simulation_events>

<character_states>
{chars_ref}
</character_states>

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{"consistent": true_or_false, "violations": [{{"type": "VIOLATION_TYPE", "description": "what is wrong", "severity": "high_or_low"}}]}}

If the prose is consistent, return: {{"consistent": true, "violations": []}}"""
