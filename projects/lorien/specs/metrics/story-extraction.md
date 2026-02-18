# Story Extraction Specification

> **Spec:** `specs/metrics/story-extraction.md`
> **Owner:** metrics-architect
> **Status:** Draft
> **Depends on:** `specs/schema/events.md` (#1), `specs/schema/scenes.md` (#4), `specs/metrics/tension-pipeline.md` (#13), `specs/metrics/scene-segmentation.md` (#15), `specs/metrics/irony-and-beliefs.md` (#14)
> **Blocks:** `specs/integration/data-flow.md` (#17)
> **Doc3 decisions:** #7 (promises as search targets), #14 (arc grammars — hard structure + soft scoring)

---

## 1. Overview

Story extraction is the pipeline that turns a user-selected event path into a readable story. It has four stages:

```
Selected Events (from map interaction or story query)
    → Arc Grammar Validation (hard filter: is this a valid story?)
    → Beat Classification (tag each event with its narrative function)
    → Soft Scoring (rank valid arcs by story quality)
    → Beat Sheet Generation (structured output)
    → LLM Prompt (convert beat sheet to prose)
```

The core principle (Decision #14): **grammar first, scoring second.** Invalid arcs are rejected before scoring, preventing "highlight reel" extractions that lack narrative structure.

---

## 2. Arc Grammar

### 2.1 Beat Types

From doc3.md, the canonical beat types:

```python
class BeatType(Enum):
    SETUP         = "setup"          # introduces character + situation
    COMPLICATION  = "complication"    # new obstacle or information
    ESCALATION    = "escalation"     # tension increases, stakes rise
    TURNING_POINT = "turning_point"  # moment of irreversible change
    CONSEQUENCE   = "consequence"    # aftermath, new equilibrium, closure
```

### 2.2 Grammar Definition (BNF)

A valid arc must match this grammar. The grammar defines the minimum structural requirements for a story.

```
<arc>           ::= <setup_phase> <development_phase> <climax_phase> <aftermath_phase>

<setup_phase>   ::= SETUP+

<development_phase> ::= <development_beat>+
<development_beat>  ::= COMPLICATION | ESCALATION

<climax_phase>  ::= TURNING_POINT

<aftermath_phase> ::= <aftermath_beat>+
<aftermath_beat>   ::= CONSEQUENCE

Constraints:
1. At least 1 SETUP beat
2. At least 1 COMPLICATION or ESCALATION beat
3. Exactly 1 TURNING_POINT beat
4. At least 1 CONSEQUENCE beat
5. Beats must appear in grammar order (no SETUP after TURNING_POINT)
6. Total beats: minimum 4, maximum 20
7. Protagonist consistency: one agent must appear in >= 60% of events
8. Causal connectivity: each event (except first) must have at least one
   causal_link to a prior event in the arc OR share a participant with
   the previous event
9. Minimum time span: arc must span >= 15% of total simulation time
```

### 2.3 Grammar Validation

```python
def validate_arc(events: list[Event], grammar: ArcGrammar) -> ArcValidation:
    """
    Check if a sequence of classified events satisfies the arc grammar.

    Returns ArcValidation with:
    - valid: bool
    - violations: list of specific rule violations (for debugging / UI feedback)
    """
    violations = []
    beat_sequence = [e.beat_type for e in events if e.beat_type is not None]

    # Rule 1: At least 1 SETUP
    if beat_sequence.count(BeatType.SETUP) < 1:
        violations.append("Missing SETUP beat")

    # Rule 2: At least 1 COMPLICATION or ESCALATION
    development = [b for b in beat_sequence if b in {BeatType.COMPLICATION, BeatType.ESCALATION}]
    if len(development) < 1:
        violations.append("Missing COMPLICATION or ESCALATION beat")

    # Rule 3: Exactly 1 TURNING_POINT
    tp_count = beat_sequence.count(BeatType.TURNING_POINT)
    if tp_count != 1:
        violations.append(f"Expected 1 TURNING_POINT, found {tp_count}")

    # Rule 4: At least 1 CONSEQUENCE
    if beat_sequence.count(BeatType.CONSEQUENCE) < 1:
        violations.append("Missing CONSEQUENCE beat")

    # Rule 5: Order constraint
    phase_order = {
        BeatType.SETUP: 0,
        BeatType.COMPLICATION: 1,
        BeatType.ESCALATION: 1,
        BeatType.TURNING_POINT: 2,
        BeatType.CONSEQUENCE: 3,
    }
    for i in range(1, len(beat_sequence)):
        if phase_order.get(beat_sequence[i], 0) < phase_order.get(beat_sequence[i - 1], 0):
            violations.append(
                f"Order violation: {beat_sequence[i].value} after {beat_sequence[i - 1].value}"
            )
            break  # report first violation only

    # Rule 6: Beat count
    if len(beat_sequence) < 4:
        violations.append(f"Too few beats: {len(beat_sequence)} < 4")
    if len(beat_sequence) > 20:
        violations.append(f"Too many beats: {len(beat_sequence)} > 20")

    # Rule 7: Protagonist consistency
    agent_counts = _count_agent_appearances(events)
    total = len(events)
    if total > 0:
        max_agent, max_count = max(agent_counts.items(), key=lambda x: x[1])
        if max_count / total < 0.6:
            violations.append(
                f"No protagonist: most frequent agent '{max_agent}' appears in "
                f"{max_count}/{total} events ({max_count/total:.0%})"
            )

    # Rule 9: Causal connectivity
    arc_event_ids = {e.id for e in events}
    for i, event in enumerate(events[1:], 1):
        has_causal_link = any(link in arc_event_ids for link in event.causal_links)
        shares_participant = bool(
            ({event.source_agent} | set(event.target_agents))
            & ({events[i - 1].source_agent} | set(events[i - 1].target_agents))
        )
        if not has_causal_link and not shares_participant:
            violations.append(
                f"Causal gap at event {event.id}: no causal link or participant overlap"
            )
            break

    # Rule 10: Minimum time span
    if events:
        span = events[-1].sim_time - events[0].sim_time
        min_span = 0.15  # 15% of total — caller must provide total sim time
        # Simplified: check absolute minimum of 10 sim minutes
        if span < 10.0:
            violations.append(f"Arc too short: spans {span:.1f} sim minutes (minimum 10)")

    return ArcValidation(valid=len(violations) == 0, violations=violations)


@dataclass
class ArcValidation:
    valid: bool
    violations: list[str]


def _count_agent_appearances(events: list[Event]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        for agent in {e.source_agent} | set(e.target_agents):
            counts[agent] = counts.get(agent, 0) + 1
    return counts
```

---

## 3. Beat Classification

### 3.1 Classification Algorithm

Each event in a candidate arc is classified into a BeatType based on its properties and position in the sequence.

```python
def classify_beats(events: list[Event], scenes: list[Scene]) -> list[BeatType]:
    """
    Classify each event in the arc into a BeatType.

    Strategy: use a combination of:
    1. Position in the arc (early = SETUP, late = CONSEQUENCE)
    2. Event properties (tension, type, deltas)
    3. Scene context (scene type)

    Returns a list of BeatType, one per event.
    """
    n = len(events)
    if n == 0:
        return []

    classifications = []
    tensions = [e.metrics["tension"] for e in events]
    peak_tension_idx = tensions.index(max(tensions))

    for i, event in enumerate(events):
        position_ratio = i / max(n - 1, 1)  # 0.0 = start, 1.0 = end

        beat = _classify_single_beat(
            event=event,
            position_ratio=position_ratio,
            peak_tension_idx=peak_tension_idx,
            event_index=i,
            total_events=n,
            tensions=tensions,
        )
        classifications.append(beat)

    return classifications


def _classify_single_beat(
    event: Event,
    position_ratio: float,
    peak_tension_idx: int,
    event_index: int,
    total_events: int,
    tensions: list[float],
) -> BeatType:
    """
    Classify a single event. Rules applied in priority order.
    """

    # Rule 1: CATASTROPHE or CONFLICT at tension peak → TURNING_POINT
    if event_index == peak_tension_idx and event.type in {EventType.CATASTROPHE, EventType.CONFLICT}:
        return BeatType.TURNING_POINT

    # Rule 2: REVEAL with irony collapse → TURNING_POINT
    if event.type == EventType.REVEAL:
        collapse = event.metrics.get("irony_collapse", {})
        if collapse.get("detected", False):
            return BeatType.TURNING_POINT

    # Rule 3: Position-based defaults
    if position_ratio < 0.25:
        # Early: likely SETUP or COMPLICATION
        if event.type in {EventType.CHAT, EventType.SOCIAL_MOVE, EventType.OBSERVE}:
            return BeatType.SETUP
        elif event.type in {EventType.CONFIDE, EventType.LIE}:
            return BeatType.COMPLICATION
        else:
            return BeatType.SETUP

    elif position_ratio < 0.70:
        # Middle: development phase
        if event_index < peak_tension_idx:
            # Before the peak: escalation or complication
            if tensions[event_index] > tensions[max(0, event_index - 1)]:
                return BeatType.ESCALATION
            else:
                return BeatType.COMPLICATION
        else:
            # After peak but before 70% mark: still escalation
            return BeatType.ESCALATION

    elif event_index > peak_tension_idx:
        # After the peak and past 70%: consequence (aftermath / closure)
        return BeatType.CONSEQUENCE

    else:
        # Fallback: if we're past 70% but before/at peak
        return BeatType.ESCALATION
```

### 3.2 Classification Heuristics Summary

| Condition | Beat Type |
|---|---|
| Event at tension peak AND type is CATASTROPHE/CONFLICT | TURNING_POINT |
| REVEAL event with irony collapse | TURNING_POINT |
| Position < 25% AND type is CHAT/SOCIAL_MOVE/OBSERVE | SETUP |
| Position < 25% AND type is CONFIDE/LIE | COMPLICATION |
| Position 25-70%, before peak, tension rising | ESCALATION |
| Position 25-70%, before peak, tension flat/falling | COMPLICATION |
| Position > 70%, after peak | CONSEQUENCE |

### 3.3 Post-Classification Validation

After classification, check that the result satisfies the grammar. If it doesn't, attempt corrections:

```python
def fix_classifications(beats: list[BeatType], events: list[Event]) -> list[BeatType]:
    """
    Attempt to fix common classification problems:
    1. No TURNING_POINT: promote the highest-tension event in the middle 50%
    2. No SETUP: reclassify the first event as SETUP
    3. No CONSEQUENCE: reclassify the last event as CONSEQUENCE
    """
    beats = list(beats)

    # Fix missing TURNING_POINT
    if BeatType.TURNING_POINT not in beats:
        tensions = [e.metrics["tension"] for e in events]
        mid_start = len(events) // 4
        mid_end = 3 * len(events) // 4
        mid_tensions = [(tensions[i], i) for i in range(mid_start, mid_end)]
        if mid_tensions:
            _, best_idx = max(mid_tensions)
            beats[best_idx] = BeatType.TURNING_POINT

    # Fix missing SETUP
    if BeatType.SETUP not in beats:
        beats[0] = BeatType.SETUP

    # Fix missing CONSEQUENCE
    if BeatType.CONSEQUENCE not in beats:
        beats[-1] = BeatType.CONSEQUENCE

    return beats
```

---

## 4. Soft Scoring

Soft scoring ranks valid arcs. Only called on arcs that pass grammar validation.

### 4.1 Scoring Function

```python
def score_arc(
    events: list[Event],
    beats: list[BeatType],
    weights: TensionWeights,
    scenes: list[Scene],
) -> ArcScore:
    """
    Score a grammar-valid arc on multiple quality dimensions.
    Returns a composite score and individual component scores.
    """
    tensions = [e.metrics["tension"] for e in events]

    # 1. Tension variance: good stories have dynamic tension, not flat lines
    tension_variance = _variance(tensions)
    tension_variance_score = min(tension_variance / 0.05, 1.0)  # 0.05 variance = excellent

    # 2. Peak tension: the highest point should be genuinely dramatic
    peak_tension = max(tensions) if tensions else 0.0
    peak_tension_score = peak_tension  # already [0, 1]

    # 3. Tension shape: the arc should have a recognizable shape
    #    (rising to peak, then falling)
    tension_shape_score = _evaluate_tension_shape(tensions)

    # 4. Counterfactual impact of turning point
    tp_events = [e for e, b in zip(events, beats) if b == BeatType.TURNING_POINT]
    if tp_events:
        tp_significance = tp_events[0].metrics.get("significance", 0.0)
    else:
        tp_significance = 0.0
    significance_score = tp_significance

    # 5. Thematic coherence: how consistently does the arc move along one thematic axis?
    thematic_coherence_score = _thematic_coherence(events)

    # 6. Irony arc: does irony build and then collapse?
    irony_arc_score = _evaluate_irony_arc(events)

    # 7. Protagonist consistency: how much does the protagonist dominate?
    protagonist_score = _protagonist_dominance(events)

    # Composite: weighted sum
    composite = (
        0.20 * tension_variance_score +
        0.15 * peak_tension_score +
        0.15 * tension_shape_score +
        0.15 * significance_score +
        0.15 * thematic_coherence_score +
        0.10 * irony_arc_score +
        0.10 * protagonist_score
    )

    return ArcScore(
        composite=composite,
        tension_variance=tension_variance_score,
        peak_tension=peak_tension_score,
        tension_shape=tension_shape_score,
        significance=significance_score,
        thematic_coherence=thematic_coherence_score,
        irony_arc=irony_arc_score,
        protagonist_dominance=protagonist_score,
    )


@dataclass
class ArcScore:
    composite: float
    tension_variance: float
    peak_tension: float
    tension_shape: float
    significance: float
    thematic_coherence: float
    irony_arc: float
    protagonist_dominance: float
```

### 4.2 Scoring Sub-Functions

```python
def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _evaluate_tension_shape(tensions: list[float]) -> float:
    """
    Score how well the tension arc matches a classic dramatic shape:
    rise → peak → fall.

    A perfect shape has the peak in the 50-80% position.
    """
    if len(tensions) < 3:
        return 0.0

    peak_idx = tensions.index(max(tensions))
    peak_position = peak_idx / (len(tensions) - 1)

    # Best peak position: 60-75% through the arc
    if 0.5 <= peak_position <= 0.8:
        position_score = 1.0
    elif 0.4 <= peak_position <= 0.9:
        position_score = 0.7
    else:
        position_score = 0.3

    # Check that tension generally rises before peak and falls after
    pre_peak = tensions[:peak_idx + 1]
    post_peak = tensions[peak_idx:]

    rising_count = sum(1 for i in range(1, len(pre_peak)) if pre_peak[i] >= pre_peak[i - 1])
    rising_ratio = rising_count / max(len(pre_peak) - 1, 1)

    falling_count = sum(1 for i in range(1, len(post_peak)) if post_peak[i] <= post_peak[i - 1])
    falling_ratio = falling_count / max(len(post_peak) - 1, 1)

    shape_score = (rising_ratio + falling_ratio) / 2.0

    return position_score * 0.5 + shape_score * 0.5


def _thematic_coherence(events: list[Event]) -> float:
    """
    How consistently does the arc move along thematic axes?
    High coherence = mostly one axis changing. Low = scattered changes across all axes.
    """
    axis_totals: dict[str, float] = {}
    for e in events:
        for axis, delta in e.metrics.get("thematic_shift", {}).items():
            axis_totals[axis] = axis_totals.get(axis, 0.0) + abs(delta)

    if not axis_totals:
        return 0.5  # neutral if no thematic data

    total_shift = sum(axis_totals.values())
    if total_shift == 0:
        return 0.5

    # Coherence = max axis share of total shift
    max_axis_shift = max(axis_totals.values())
    return max_axis_shift / total_shift


def _evaluate_irony_arc(events: list[Event]) -> float:
    """
    Does irony build in the first half and collapse in the second?
    """
    if len(events) < 4:
        return 0.0

    ironies = [e.metrics.get("irony", 0.0) for e in events]
    mid = len(ironies) // 2

    first_half_mean = sum(ironies[:mid]) / max(mid, 1)
    second_half_mean = sum(ironies[mid:]) / max(len(ironies) - mid, 1)

    # Good irony arc: first half > second half (irony collapses)
    if first_half_mean > 0 and second_half_mean < first_half_mean:
        drop_ratio = (first_half_mean - second_half_mean) / first_half_mean
        return min(drop_ratio, 1.0)

    return 0.2  # flat or inverse irony arc still gets partial credit


def _protagonist_dominance(events: list[Event]) -> float:
    """How much does the most frequent agent dominate the arc?"""
    counts = _count_agent_appearances(events)
    if not counts:
        return 0.0
    max_count = max(counts.values())
    return max_count / len(events)
```

---

## 5. Beat Sheet Format

The beat sheet is the structured intermediate representation between "selected events" and "LLM prompt." It provides all the information the LLM needs to generate prose without access to the raw event log.

### 5.1 Data Structure

```python
@dataclass
class BeatSheet:
    """Structured story outline ready for LLM prose generation."""

    # Metadata
    arc_id: str
    protagonist: str                    # agent_id of the focal character
    title_suggestion: str               # auto-generated, LLM can override
    genre_preset: str                   # "thriller", "relationship_drama", "mystery", "default"
    arc_score: ArcScore

    # Setting
    setting_summary: str                # brief description of world/situation
    time_span: str                      # "52.5 - 63.0 sim minutes (evening)"

    # Characters
    characters: list[CharacterBrief]    # key info for each character in the arc

    # Beats (ordered)
    beats: list[Beat]

    # Thematic summary
    dominant_theme: str                 # e.g., "truth_deception"
    thematic_trajectory: str            # e.g., "deception → truth (forced)"


@dataclass
class CharacterBrief:
    """Minimal character info for the LLM."""
    agent_id: str
    name: str
    role_in_arc: str                    # "protagonist", "antagonist", "catalyst", "observer"
    key_goal: str                       # one-sentence goal description
    key_flaw: str                       # one-sentence flaw description
    key_secret: str | None              # relevant secret, if any
    emotional_start: str                # emotional state at arc start
    emotional_end: str                  # emotional state at arc end


@dataclass
class Beat:
    """One dramatic beat in the story."""
    beat_type: BeatType
    event_id: str
    event_type: EventType
    scene_id: str | None                # which scene this belongs to
    location: str
    participants: list[str]

    # Narrative content
    description: str                    # from Event.description
    tension: float
    irony_note: str | None              # e.g., "Thorne doesn't know about the affair"

    # State changes (human-readable)
    key_changes: list[str]              # e.g., ["Trust between Diana and Thorne drops sharply"]

    # Emotional states of key participants
    emotional_states: dict[str, str]    # agent_id → emotion label

    # Writing guidance
    pov_suggestion: str                 # whose perspective to write from
    tone_suggestion: str                # "tense", "intimate", "confrontational", etc.
    pacing_note: str                    # "slow build", "rapid", "lingering"
```

### 5.2 JSON Representation

```json
{
    "arc_id": "arc_thorne_betrayal_001",
    "protagonist": "thorne",
    "title_suggestion": "What Thorne Didn't See",
    "genre_preset": "relationship_drama",
    "arc_score": {
        "composite": 0.72,
        "tension_variance": 0.81,
        "peak_tension": 0.78,
        "tension_shape": 0.85,
        "significance": 0.60,
        "thematic_coherence": 0.74,
        "irony_arc": 0.65,
        "protagonist_dominance": 0.75
    },
    "setting_summary": "An evening dinner party at the Ashford residence. Six guests, multiple secrets, rising alcohol and falling composure.",
    "time_span": "15.0 - 52.5 sim minutes (~37 minutes of story time)",
    "characters": [
        {
            "agent_id": "thorne",
            "name": "Thorne Ashford",
            "role_in_arc": "protagonist",
            "key_goal": "Maintain his social standing and protect his marriage",
            "key_flaw": "Pride prevents him from confronting uncomfortable truths",
            "key_secret": "Doesn't know about the embezzlement (secret_embezzle_01)",
            "emotional_start": "Composed, slightly anxious",
            "emotional_end": "Shattered, accusatory"
        }
    ],
    "beats": [
        {
            "beat_type": "setup",
            "event_id": "E004",
            "event_type": "observe",
            "scene_id": "scene_000",
            "location": "dining_table",
            "participants": ["thorne"],
            "description": "Thorne notices Marcus and Elena exchanging a glance",
            "tension": 0.24,
            "irony_note": "Thorne doesn't know about the affair — this glance is his first clue",
            "key_changes": ["Thorne's suspicion increases by 0.15"],
            "emotional_states": {"thorne": "uneasy, watchful"},
            "pov_suggestion": "thorne",
            "tone_suggestion": "subtle unease",
            "pacing_note": "slow, observational — the reader should notice before Thorne fully registers"
        }
    ],
    "dominant_theme": "truth_deception",
    "thematic_trajectory": "deception → suspicion → forced confrontation with truth"
}
```

---

## 6. LLM Prompt Template

### 6.1 Prompt Structure

The LLM prompt is constructed from the beat sheet. It follows a structured format that constrains the LLM to write within the story's established structure while allowing creative freedom in voice, imagery, and dialogue.

```python
def build_llm_prompt(beat_sheet: BeatSheet) -> str:
    """
    Build the LLM prompt from a beat sheet.

    The prompt has 5 sections:
    1. CONTEXT — world, characters, setting
    2. STRUCTURE — the beat sequence (what must happen)
    3. CONSTRAINTS — what the LLM must NOT do
    4. VOICE — tone, style, POV guidance
    5. OUTPUT FORMAT — what to produce
    """

    prompt = f"""You are writing a short story scene based on a simulation of a fictional dinner party. The story's structure has already been determined by the simulation — your job is to bring it to life with prose, dialogue, and interiority.

## CONTEXT

**Setting:** {beat_sheet.setting_summary}
**Time span:** {beat_sheet.time_span}
**Genre feel:** {beat_sheet.genre_preset}
**Dominant theme:** {beat_sheet.dominant_theme}
**Thematic trajectory:** {beat_sheet.thematic_trajectory}

## CHARACTERS

"""
    for char in beat_sheet.characters:
        prompt += f"""**{char.name}** ({char.role_in_arc})
- Goal: {char.key_goal}
- Flaw: {char.key_flaw}
- Secret: {char.key_secret or "None"}
- Starts: {char.emotional_start}
- Ends: {char.emotional_end}

"""

    prompt += """## BEAT SEQUENCE

Write the story following these beats in order. Each beat MUST be included. Do not skip, reorder, or add beats.

"""
    for i, beat in enumerate(beat_sheet.beats, 1):
        prompt += f"""### Beat {i}: {beat.beat_type.value.upper()}
- **What happens:** {beat.description}
- **Location:** {beat.location}
- **Present:** {', '.join(beat.participants)}
- **Tension level:** {_tension_label(beat.tension)}
- **Emotional states:** {_format_emotions(beat.emotional_states)}
"""
        if beat.irony_note:
            prompt += f"- **Dramatic irony:** {beat.irony_note}\n"
        if beat.key_changes:
            prompt += f"- **Key changes:** {'; '.join(beat.key_changes)}\n"
        prompt += f"- **POV:** {beat.pov_suggestion}\n"
        prompt += f"- **Tone:** {beat.tone_suggestion}\n"
        prompt += f"- **Pacing:** {beat.pacing_note}\n"
        prompt += "\n"

    prompt += """## CONSTRAINTS

1. DO NOT invent new plot events. Everything that happens must come from the beats above.
2. DO NOT change the outcome of any beat. The simulation determined what happened; you determine how it reads.
3. DO NOT give characters knowledge they don't have. Respect the dramatic irony notes.
4. You MAY add:
   - Internal thoughts and feelings (consistent with emotional states listed)
   - Dialogue (consistent with character voice and situation)
   - Sensory details (sight, sound, smell of the dinner party)
   - Body language and micro-expressions
   - Brief flashbacks or memories (to add depth to reactions)
5. Maintain the tension trajectory: the story should feel like it builds to the turning point and then releases.

## VOICE AND STYLE

- Write in close third person, primarily from the protagonist's POV ({beat_sheet.protagonist}).
- Shift POV only at beat boundaries, and only when the POV suggestion changes.
- Prose style: literary fiction. Specific, grounded details. No purple prose.
- Dialogue should sound natural — people at a dinner party, not characters in a novel.
- Show, don't tell emotional states. Use behavior, dialogue, and body language.

## OUTPUT FORMAT

Write the complete scene as continuous prose. Use section breaks (---) between beats only if there's a location change or significant time skip. Target length: 200-400 words per beat, {len(beat_sheet.beats) * 300} words total.

Begin writing.
"""
    return prompt


def _tension_label(tension: float) -> str:
    if tension < 0.2:
        return "low (calm, casual)"
    elif tension < 0.4:
        return "moderate (undercurrents of tension)"
    elif tension < 0.6:
        return "elevated (palpable discomfort)"
    elif tension < 0.8:
        return "high (confrontation, distress)"
    else:
        return "extreme (crisis, breaking point)"


def _format_emotions(emotions: dict[str, str]) -> str:
    return ", ".join(f"{name}: {state}" for name, state in emotions.items())
```

---

## 7. Worked Example: 8 Dinner Party Events

### 7.1 Selected Events

The user selects Thorne's arc from the map, focusing on his discovery of the affair.

| # | Event ID | Type | Tension | Description |
|---|---|---|---|---|
| 1 | E004 | OBSERVE | 0.24 | Thorne notices Marcus and Elena exchanging a glance |
| 2 | E011 | CHAT | 0.12 | Lydia mentions to Diana that Marcus and Elena seem friendly |
| 3 | E014 | CONFLICT | 0.58 | Diana confronts Thorne about the debt on the balcony |
| 4 | E015 | CONFLICT | 0.65 | Thorne defends himself, voices rising |
| 5 | E017 | SOCIAL_MOVE | 0.20 | Thorne returns to dining room, rattled |
| 6 | E018 | INTERNAL | 0.35 | Thorne thinks about what he's noticed tonight |
| 7 | E019 | OBSERVE | 0.42 | Thorne sees Elena and Marcus sitting close, Elena touching his hand |
| 8 | E020 | CATASTROPHE | 0.78 | Thorne snaps: "How long have you and Marcus been so... close?" |

### 7.2 Beat Classification

```
E004 → SETUP       (position 0%, OBSERVE, introduces the problem)
E011 → COMPLICATION (position 14%, CHAT that adds information)
E014 → ESCALATION  (position 29%, CONFLICT, tension rising)
E015 → ESCALATION  (position 43%, CONFLICT, tension still rising)
E017 → COMPLICATION (position 57%, SOCIAL_MOVE, transition beat)
E018 → ESCALATION  (position 71%, INTERNAL, tension building internally)
E019 → ESCALATION  (position 86%, OBSERVE, final straw)
E020 → TURNING_POINT (position 100%, CATASTROPHE at tension peak)
```

**Problem:** No CONSEQUENCE. The arc ends at the turning point.

**Fix options:**
1. Extend the arc by adding E021-E022 (if they exist) showing aftermath.
2. Accept a truncated arc (valid if we relax rule 4 for "cliffhanger" arcs).

For this example, assume the user selected exactly these 8 events. The validator would flag "Missing CONSEQUENCE beat." The system suggests extending the selection:

```
Suggestion: "This arc ends at the turning point. Add 1-2 events after E020 for a complete arc.
Available: E021 (Elena's reaction), E022 (Marcus attempts to leave)."
```

User extends to 10 events:

| 9 | E021 | INTERNAL | 0.55 | Elena freezes, unable to speak |
| 10 | E022 | SOCIAL_MOVE | 0.30 | Marcus quietly stands and heads for the foyer |

Updated classifications:
```
E020 → TURNING_POINT
E021 → CONSEQUENCE  (position 89%, after peak, emotional aftermath)
E022 → CONSEQUENCE  (position 100%, final event, new equilibrium: Marcus leaving)
```

### 7.3 Grammar Validation

```
SETUP(E004) → COMPLICATION(E011) → ESCALATION(E014) → ESCALATION(E015)
→ COMPLICATION(E017) → ESCALATION(E018) → ESCALATION(E019)
→ TURNING_POINT(E020) → CONSEQUENCE(E021) → CONSEQUENCE(E022)
```

Check:
- [x] At least 1 SETUP (E004)
- [x] At least 1 COMPLICATION (E011, E017)
- [x] Exactly 1 TURNING_POINT (E020)
- [x] At least 1 CONSEQUENCE (E021)
- [x] Order correct (SETUP → COMPLICATION → ESCALATION → TURNING_POINT → CONSEQUENCE)
- [x] Beat count: 10 (5 <= 10 <= 20)
- [x] Protagonist: Thorne appears in 8/10 events (80% > 60%)
- [x] Causal connectivity: E011 shares participant (Lydia/Diana overlap), all others connected
- [x] Time span: 15.0 to 55.5 = 40.5 min (> 10 min minimum)

**Result: VALID**

### 7.4 Arc Score

| Component | Score | Reasoning |
|---|---|---|
| tension_variance | 0.81 | High variance (0.12 to 0.78) |
| peak_tension | 0.78 | Strong peak at E020 |
| tension_shape | 0.85 | Peak at position 80%, good rising-falling shape |
| significance | 0.60 | E020 has moderate counterfactual impact (assumed) |
| thematic_coherence | 0.74 | Mostly truth_deception axis |
| irony_arc | 0.65 | Irony builds (Thorne doesn't know) then collapses (E020) |
| protagonist_dominance | 0.80 | Thorne in 8/10 events |

**Composite: 0.20(0.81) + 0.15(0.78) + 0.15(0.85) + 0.15(0.60) + 0.15(0.74) + 0.10(0.65) + 0.10(0.80) = 0.162 + 0.117 + 0.128 + 0.090 + 0.111 + 0.065 + 0.080 = 0.753**

**Arc score: 0.75** (strong arc)

### 7.5 Beat Sheet (Abbreviated)

```json
{
    "arc_id": "arc_thorne_001",
    "protagonist": "thorne",
    "title_suggestion": "The Glance That Broke the Evening",
    "genre_preset": "relationship_drama",
    "arc_score": {"composite": 0.75},
    "setting_summary": "Evening dinner party at the Ashford residence. Six guests gathered around a table laden with good food and bad secrets.",
    "time_span": "15.0 - 55.5 sim minutes",
    "characters": [
        {"agent_id": "thorne", "name": "Thorne Ashford", "role_in_arc": "protagonist", "key_goal": "Host a successful evening, maintain control", "key_flaw": "Pride — would rather not see than face humiliation", "key_secret": "Doesn't know about the embezzlement", "emotional_start": "Composed but tense", "emotional_end": "Publicly shattered"},
        {"agent_id": "elena", "name": "Elena Thorne", "role_in_arc": "catalyst", "key_goal": "Keep the affair hidden", "key_flaw": "Conflict avoidance", "key_secret": "Affair with Marcus", "emotional_start": "Nervous", "emotional_end": "Frozen in terror"},
        {"agent_id": "marcus", "name": "Marcus Webb", "role_in_arc": "catalyst", "key_goal": "Maintain cover story", "key_flaw": "Arrogance — believes he won't get caught", "key_secret": "Affair with Elena + embezzlement", "emotional_start": "Confident", "emotional_end": "Calculating escape"},
        {"agent_id": "diana", "name": "Diana Reeves", "role_in_arc": "supporting", "key_goal": "Protect Elena while managing her own debt to Marcus", "key_flaw": "Loyalty conflict", "key_secret": "Owes Marcus money", "emotional_start": "Watchful", "emotional_end": "Torn"}
    ],
    "beats": ["... (10 beats as defined above)"],
    "dominant_theme": "truth_deception",
    "thematic_trajectory": "suppressed suspicion → forced confrontation → public rupture"
}
```

### 7.6 LLM Prompt (First 2 beats excerpt)

The full prompt would be ~2000 words. Here is how the first two beats render:

```
### Beat 1: SETUP
- **What happens:** Thorne notices Marcus and Elena exchanging a glance
- **Location:** dining_table
- **Present:** Thorne
- **Tension level:** moderate (undercurrents of tension)
- **Emotional states:** thorne: uneasy, watchful
- **Dramatic irony:** Thorne doesn't know about the affair — this glance is his first clue
- **Key changes:** Thorne's suspicion increases
- **POV:** thorne
- **Tone:** subtle unease
- **Pacing:** slow, observational — the reader should notice before Thorne fully registers

### Beat 2: COMPLICATION
- **What happens:** Lydia mentions to Diana that Marcus and Elena seem friendly
- **Location:** dining_table
- **Present:** Lydia, Diana
- **Tension level:** low (calm, casual)
- **Emotional states:** lydia: curious, diana: distracted
- **Dramatic irony:** Lydia is closer to the truth than she realizes
- **Key changes:** The observation enters social circulation — it's no longer just in Thorne's head
- **POV:** thorne (overhearing from nearby)
- **Tone:** casual surface, ominous undertone
- **Pacing:** brief — a passing comment that lands heavier than intended
```

---

## 8. Story Query Interface

When a user makes a natural language query ("Find me a tragedy for Thorne"), the system:

1. Identifies the protagonist (Thorne)
2. Searches the event graph for paths involving Thorne
3. Classifies beats on each candidate path
4. Validates against the arc grammar
5. Scores valid arcs
6. Returns the top-N results, ranked by composite score

```python
@dataclass
class StoryQuery:
    protagonist: str | None = None        # agent_id, or None for "any"
    genre_hint: str | None = None         # "tragedy", "comedy", "mystery", etc.
    thematic_axis: str | None = None      # e.g., "loyalty_betrayal"
    thematic_direction: float | None = None  # -1 = toward betrayal, +1 = toward loyalty
    min_events: int = 5
    max_events: int = 15
    max_results: int = 5


def search_stories(
    query: StoryQuery,
    events: list[Event],
    scenes: list[Scene],
    weights: TensionWeights,
) -> list[tuple[list[Event], BeatSheet, ArcScore]]:
    """
    Search for story arcs matching the query.

    Algorithm:
    1. Filter events by protagonist (if specified)
    2. Generate candidate paths (sliding windows + causal chain following)
    3. Classify beats on each candidate
    4. Validate against grammar
    5. Score valid arcs
    6. Return top-N by score
    """
    # Phase 6 feature — full implementation deferred.
    # For MVP, the user selects events manually on the map.
    ...
```

---

## 9. Edge Cases

### Arc with no clear turning point
If the selected events have a flat tension profile (no peak > 0.4), the beat classifier will still assign a TURNING_POINT to the highest-tension event. The arc will validate but score poorly on `tension_shape` and `peak_tension`.

### Very short arcs (4 events)
Minimum valid arc: 1 SETUP + 1 COMPLICATION + 1 TURNING_POINT + 1 CONSEQUENCE. This is an extremely compressed story — the LLM prompt should note "Write a very tight, focused scene."

### Multiple TURNING_POINTs in selection
The grammar requires exactly 1. If the classifier tags multiple events as TURNING_POINT, keep only the one with the highest tension and reclassify others as ESCALATION (if before) or CONSEQUENCE (if after).

### Events from different characters with no overlap
If the user selects events from completely different storylines, the causal connectivity check (Rule 9) fails. The system reports "These events don't form a connected story" and suggests bridging events.

### LLM refusing to follow structure
The prompt is designed to be highly constraining. If the LLM deviates (skips beats, invents events), the system can:
1. Detect deviations by checking for beat marker adherence
2. Re-prompt with stricter instructions
3. Present the beat sheet directly to the user as a fallback

---

## 10. NOT In Scope

- **Automatic arc discovery (Phase 6):** The story query search engine that finds arcs without user selection. For MVP, users select events on the map.
- **Multi-protagonist arcs:** The grammar assumes one protagonist. Ensemble stories (interleaving multiple POVs) are a post-MVP extension.
- **Genre-specific grammars:** The grammar is fixed (setup → development → turning point → aftermath). Genre-specific grammars (mystery has different beat requirements than romance) are a post-MVP extension.
- **LLM fine-tuning:** The prompt template uses a general-purpose LLM (Claude). Fine-tuning for prose style is out of scope.
- **Interactive editing:** The user cannot modify the beat sheet in MVP. They select events, get a beat sheet, get prose. Beat sheet editing is a Phase 4+ feature.

---

## 11. Dependencies

| Depends On | What It Provides |
|---|---|
| `specs/schema/events.md` | Event schema, EventType, BeatType |
| `specs/schema/scenes.md` | Scene schema |
| `specs/metrics/tension-pipeline.md` | Per-event tension values, TensionWeights |
| `specs/metrics/irony-and-beliefs.md` | Irony scores, irony collapse detection |
| `specs/metrics/scene-segmentation.md` | Scene list for scene_id assignment in beats |

| Depended On By | What It Consumes |
|---|---|
| `specs/integration/data-flow.md` | BeatSheet format, LLM prompt interface |
