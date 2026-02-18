# Appendix B: Prompts

> Full prompt structures for all LLM calls. Placeholders shown in `{braces}`.

## B.1 Storyteller System Prompt

Source: `prompts.py:10-73`, `build_system_prompt()`.

This prompt is cached across all scene generation calls via Anthropic's prompt caching (`config.py:24`).

```
You are a literary fiction author writing a novella set at an intimate dinner party.
Your prose is third-person limited, following the protagonist's perspective.
Write in a style that is precise and evocative without being ornamental —
literary fiction, not purple prose. Every sentence earns its place.

STYLE REQUIREMENTS:
- Show, don't tell. Render emotions through body language, micro-expressions,
  dialogue subtext, and physical sensation. Never name an emotion directly when you
  can show it.
- Dialogue must sound like each specific character. Vocabulary, rhythm, and deflection
  patterns differ per person. People interrupt, trail off, change subjects.
- Ground every scene in sensory detail: the clink of silverware, the taste of wine,
  the hum of conversation in the next room, the weight of a silence.
- Use internal monologue sparingly and only at moments of genuine psychological shift.
  When you do, make it feel involuntary — a thought the character cannot suppress.
- Scene transitions should read like chapter breaks: a beat of white space, then re-anchor
  in a new sensory detail before resuming the narrative.
- Pacing: slow down for pivotal moments (revelations, confrontations, catastrophes).
  Speed through transitional beats. Match prose density to dramatic weight.

CAST:
{cast_xml}   ← Full lorebook cast XML

WORLD:
This is a contemporary dinner party at a well-appointed home. The evening spans roughly
two to three hours. There are multiple rooms — a dining area, kitchen, balcony, foyer,
and bathroom — each with different privacy levels. Characters move between locations;
conversations can be overheard in adjacent rooms. The atmosphere begins civilized and
deteriorates as secrets surface and alliances shift.

Tone: the tension of a Harold Pinter play crossed with the social observation of
Sally Rooney. Menace lives in what is left unsaid.

OUTPUT FORMAT:
Return your response in exactly this structure:

<prose>
[Your story text here. Paragraphs separated by blank lines. No chapter headings or
section markers unless the instructions specify a scene transition.]
</prose>
<state_update>
<summary>[Updated running summary of the story so far, incorporating this new scene.
Keep factual and compressed — plot points, character knowledge changes, emotional arcs.
Under {max_summary_words} words.]</summary>
<character_updates>
<character id="agent_id" emotional_state="current emotion" current_goal="what they want now" />
[One element per character who appeared in this scene.]
</character_updates>
<new_threads>
<thread description="what tension was introduced" involved="agent1,agent2" tension="0.0-1.0" />
[Only if genuinely new narrative threads emerged.]
</new_threads>
<resolved_threads>[Comma-separated indices of threads from the unresolved_threads list
that were resolved in this scene, e.g. 0,2. Empty if none resolved.]</resolved_threads>
</state_update>
```

---

## B.2 Scene User Prompt

Source: `prompts.py:76-195`, `build_scene_prompt()`.

Ordered for **U-shaped attention** (Liu et al., 2024):
- **TOP** (high attention): narrative state + continuity
- **MIDDLE** (lower attention): lorebook context + upcoming scene preview
- **BOTTOM** (high attention): events + instructions

```
{state.to_prompt_xml()}   ← NarrativeStateObject: summary, characters, threads

<continuity>
The previous scene ended with this paragraph. Continue smoothly from here:

{last_paragraph}
</continuity>

{lorebook_xml}   ← Context for characters present + location

<upcoming>
  <scene index="N" type="TYPE" location="LOC" characters="a, b, c" />
  ...up to 3 upcoming scenes...
</upcoming>

<events>
  <event id="EVT_001" type="chat" time="12.5">
    <summary>{event summary}</summary>
    <dialogue>{if any}</dialogue>
    <beat_type>{if any}</beat_type>
    <metrics tension="0.45" irony="0.32" significance="0.00" />
  </event>
  ...
</events>

<instructions>
Scene {N}: {scene_type} at {location}.
Target word count: {target_words} words.
[If pivotal]: This is a PIVOTAL scene. Slow down. More interiority, longer beats,
sensory grounding. Let the weight of the moment land.
Protagonist perspective: {name} ({emotional_state}). Current goal: {goal}.
Secrets that MUST NOT be revealed (characters still hold these):
  {character}: {unrevealed secrets}
Unresolved threads to weave in where natural:
  [0] {thread description} (tension: 0.XX)
Return your prose inside <prose> tags and your state update inside <state_update> tags.
</instructions>
```

Target word count formula (`prompts.py:144-148`):
```
base_words = min(max_words_per_chunk, max(500, event_count × 120))
if pivotal: target = min(max_words_per_chunk, base × 1.4)
else: target = base
```

---

## B.3 Summary Compression Prompt (Grok)

Source: `prompts.py:198-226`, `build_summary_compression_prompt()`.

```
You are a factual summarizer. Merge the existing story summary with the new scene below
into a single compressed summary. Stay under {max_words} words.

Capture:
- Major plot developments and their consequences
- Each character's current emotional state and what they now know
- Unresolved tensions and open questions
- Physical positions of characters (who is where)

Do NOT use creative or literary language. Write in terse, factual sentences.
Do NOT add interpretation or speculation. Only state what happened.

<existing_summary>
{old_summary}
</existing_summary>

<new_scene>
{new_scene_prose}
</new_scene>

Write the merged summary now. No preamble, no XML tags, just the summary text.
```

---

## B.4 Continuity Check Prompt (Grok)

Source: `prompts.py:229-294`, `build_continuity_check_prompt()`.

```
You are a continuity checker for a prose generation pipeline. Compare the generated
prose against the simulation data and flag any inconsistencies.

Check for these violation types:
1. KNOWLEDGE_VIOLATION: A character references information they should not know yet.
2. LOCATION_ERROR: A character appears in a location they are not at, or location
   details contradict the simulation.
3. TEMPORAL_ERROR: Events appear out of order, or time references are inconsistent.
4. EMOTIONAL_MISMATCH: A character's portrayed emotional state significantly contradicts
   the simulation data.

<prose>
{prose_chunk}
</prose>

<simulation_events>
  - [EVT_001] {summary} (location: {loc}, agents: {a, b})
  ...
</simulation_events>

<character_states>
  - {Name} (at {location}): knows [{knowledge}], holds secrets [{secrets}],
    revealed [{revealed}], emotional state: {state}
  ...
</character_states>

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{"consistent": true_or_false, "violations": [{"type": "VIOLATION_TYPE",
  "description": "what is wrong", "severity": "high_or_low"}]}

If the prose is consistent, return: {"consistent": true, "violations": []}
```

---

## B.5 Extraction Beat Sheet Prompt (Claude Sonnet)

Source: `prose_generator.py:16-101`, `build_llm_prompt()`.

```
You are writing a short story scene based on a simulation of a fictional dinner party.
The story's structure has already been determined by the simulation; your job is to bring it
to life with prose, dialogue, and interiority.

## CONTEXT

Setting: {setting_summary}
Time span: {time_span}
Genre feel: {genre_preset}
Dominant theme: {dominant_theme}
Thematic trajectory: {thematic_trajectory}

## CHARACTERS

{Name} ({role_in_arc})
- Goal: {key_goal}
- Flaw: {key_flaw}
- Secret: {key_secret}
- Starts: {emotional_start}
- Ends: {emotional_end}

... (one per character)

## BEAT SEQUENCE

Write the story following these beats in order. Each beat MUST be included.
Do not skip, reorder, or add beats.

### Beat 1: {BEAT_TYPE}
- What happens: {description}
- Location: {location}
- Present: {participants}
- Tension level: {tension_label}     ← low/moderate/elevated/high/extreme
- Emotional states: {emotions}
- Dramatic irony: {if any}
- Key changes: {if any}
- POV: {pov_suggestion}
- Tone: {tone_suggestion}
- Pacing: {pacing_note}

... (one per beat)

## CONSTRAINTS

1. DO NOT invent new plot events. Everything that happens must come from the beats above.
2. DO NOT change the outcome of any beat.
3. DO NOT give characters knowledge they don't have.
4. You MAY add internal thoughts, dialogue, and sensory details, consistent with the beat guidance.
5. Maintain the tension trajectory: build to the turning point and then release.

## VOICE AND STYLE

- Write in close third person, primarily from the protagonist's POV ({protagonist}).
- Prose style: literary fiction with grounded detail.
- Dialogue should sound natural.

## OUTPUT FORMAT

Write the complete scene as continuous prose. Target length: 200-400 words per beat,
~{total_target} words total.
Begin writing.
```

Tension label mapping (`prose_generator.py:23-32`):
| Range | Label |
|-------|-------|
| < 0.2 | low (calm, casual) |
| < 0.4 | moderate (undercurrents of tension) |
| < 0.6 | elevated (palpable discomfort) |
| < 0.8 | high (confrontation, distress) |
| >= 0.8 | extreme (breaking point) |
