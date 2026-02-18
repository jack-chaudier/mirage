# Scene Segmentation Specification

> **Spec:** `specs/metrics/scene-segmentation.md`
> **Owner:** metrics-architect
> **Status:** Draft
> **Depends on:** `specs/schema/events.md` (#1), `specs/schema/scenes.md` (#4), `specs/metrics/tension-pipeline.md` (#13)
> **Blocks:** `specs/integration/data-flow.md` (#17)
> **Doc3 decisions:** #16 (first-class scene segmentation)

---

## 1. Overview

Writers think in scenes, not events. A scene is a dramatic unit: a group of temporally contiguous events sharing a location, a set of participants, and a coherent dramatic action. The scene segmentation algorithm groups the raw event log into scenes.

Per Decision #16, scenes are a **first-class layer** between events and arcs. The renderer shows event density; the sidebar shows scenes. Story extraction operates on scene-level beats.

### What the Segmenter Produces

```
Input:  list[Event]  (ordered by sim_time, then tick_id, then order_in_tick)
        + per-event tension values (from tension pipeline)
        + belief matrix snapshots

Output: list[Scene]  (ordered by time_start)
```

---

## 2. Scene Boundary Rules

A new scene begins when ANY of the following conditions are met:

### 2.1 Location Change

If consecutive events occur at different locations, a scene boundary is placed between them.

**Exception:** If agent A is at the `dining_table` and agent B is at the `kitchen`, and the kitchen's `overhear_from` list includes `dining_table`, events in these two locations MAY be grouped into the same scene IF their participants overlap above threshold. This captures "the dinner table conversation that someone in the kitchen overhears."

```python
def location_break(event_prev: Event, event_curr: Event, world: WorldState) -> bool:
    if event_prev.location_id == event_curr.location_id:
        return False
    # Check adjacency exception
    loc_prev = world.locations[event_prev.location_id]
    loc_curr = world.locations[event_curr.location_id]
    if event_curr.location_id in loc_prev.overhear_from or \
       event_prev.location_id in loc_curr.overhear_from:
        return False  # adjacent with overhear — don't break
    return True
```

### 2.2 Participant Turnover

If the set of participants changes significantly between consecutive events, a boundary is placed.

**Threshold:** If the Jaccard similarity of participant sets drops below 0.3, it's a new scene.

```python
def participant_break(event_prev: Event, event_curr: Event) -> bool:
    set_prev = {event_prev.source_agent} | set(event_prev.target_agents)
    set_curr = {event_curr.source_agent} | set(event_curr.target_agents)

    if not set_prev or not set_curr:
        return False

    jaccard = len(set_prev & set_curr) / len(set_prev | set_curr)
    return jaccard < 0.3
```

**Why 0.3:** A dinner party has 6 agents. A 2-person conversation followed by a different 2-person conversation shares 0 agents (Jaccard = 0). But a group scene where one person leaves (5 of 6 remain) has Jaccard ~0.83. The threshold 0.3 catches genuine shift-of-focus while allowing natural churn within a group scene.

### 2.3 Tension Gap (Beat Boundary)

A sustained drop in tension indicates a dramatic beat has ended. If tension drops to below 30% of the recent peak and stays low for 3+ consecutive events, that's a beat boundary.

```python
def tension_gap_break(
    events: list[Event],
    current_index: int,
    window: int = 5,
    drop_ratio: float = 0.3,
    sustained_count: int = 3,
) -> bool:
    """
    Check if we're in a sustained tension valley.
    """
    if current_index < window:
        return False

    # Recent peak tension (last `window` events)
    recent = events[max(0, current_index - window):current_index]
    peak = max(e.metrics["tension"] for e in recent)

    if peak == 0:
        return False

    threshold = peak * drop_ratio

    # Check if last `sustained_count` events are all below threshold
    low_streak = events[max(0, current_index - sustained_count):current_index]
    if len(low_streak) < sustained_count:
        return False

    return all(e.metrics["tension"] < threshold for e in low_streak)
```

### 2.4 Time Proximity

If there's a gap of more than 5 sim-time minutes between consecutive events, a scene boundary is placed. In the dinner party (tick = 1.5 min), this is approximately 3-4 ticks of silence.

```python
TIME_GAP_THRESHOLD = 5.0  # sim-time minutes

def time_break(event_prev: Event, event_curr: Event) -> bool:
    return (event_curr.sim_time - event_prev.sim_time) > TIME_GAP_THRESHOLD
```

### 2.5 Irony Collapse Forced Break

When an irony collapse is detected (see `specs/metrics/irony-and-beliefs.md`), a scene boundary is placed AFTER the collapse event. The rationale: the collapse changes the dramatic landscape so fundamentally that what follows is a new scene.

```python
def irony_collapse_break(event: Event) -> bool:
    collapse = event.metrics.get("irony_collapse", {})
    return collapse.get("detected", False) and collapse.get("drop", 0) >= 0.5
```

---

## 3. Segmentation Algorithm

### 3.1 Pseudocode

```python
def segment_into_scenes(
    events: list[Event],
    world_snapshots: dict[int, WorldState],
    config: SegmentationConfig,
) -> list[Scene]:
    """
    Single-pass segmentation algorithm.

    Walk through events in order. Accumulate events into the current scene.
    When a boundary condition fires, close the current scene and start a new one.

    After initial segmentation, run a merge pass to combine tiny scenes.
    """
    if not events:
        return []

    scenes: list[Scene] = []
    current_events: list[Event] = [events[0]]

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        world = get_world_at(prev.tick_id, world_snapshots, events)

        # Check all boundary conditions
        should_break = False
        break_reason = ""

        if location_break(prev, curr, world):
            should_break = True
            break_reason = "location_change"

        elif participant_break(prev, curr):
            should_break = True
            break_reason = "participant_turnover"

        elif time_break(prev, curr):
            should_break = True
            break_reason = "time_gap"

        elif tension_gap_break(events, i, config.tension_window, config.drop_ratio, config.sustained_count):
            should_break = True
            break_reason = "tension_gap"

        elif irony_collapse_break(prev):
            should_break = True
            break_reason = "irony_collapse"

        if should_break:
            # Close current scene
            scene = build_scene(current_events, len(scenes))
            scenes.append(scene)
            # Start new scene
            current_events = [curr]
        else:
            current_events.append(curr)

    # Close final scene
    if current_events:
        scenes.append(build_scene(current_events, len(scenes)))

    # Merge pass: combine scenes with < MIN_SCENE_SIZE events
    scenes = merge_tiny_scenes(scenes, config.min_scene_size)

    return scenes
```

### 3.2 Scene Construction

```python
def build_scene(events: list[Event], scene_index: int) -> Scene:
    """Build a Scene object from a group of events."""

    # Participants: union of all source and target agents
    participants = set()
    for e in events:
        participants.add(e.source_agent)
        participants.update(e.target_agents)

    # Primary location: mode of event locations
    location_counts: dict[str, int] = {}
    for e in events:
        location_counts[e.location_id] = location_counts.get(e.location_id, 0) + 1
    primary_location = max(location_counts, key=location_counts.get)

    # Tension arc: ordered list of tension values
    tension_arc = [e.metrics["tension"] for e in events]

    # Dominant theme: thematic axis with largest absolute shift
    theme_deltas: dict[str, float] = {}
    for e in events:
        for axis, delta in e.metrics.get("thematic_shift", {}).items():
            theme_deltas[axis] = theme_deltas.get(axis, 0.0) + delta
    dominant_theme = max(theme_deltas, key=lambda k: abs(theme_deltas[k])) if theme_deltas else "none"

    # Scene type classification
    scene_type = classify_scene_type(events)

    return Scene(
        id=f"scene_{scene_index:03d}",
        event_ids=[e.id for e in events],
        location=primary_location,
        participants=participants,
        time_start=events[0].sim_time,
        time_end=events[-1].sim_time,
        tension_arc=tension_arc,
        dominant_theme=dominant_theme,
        scene_type=scene_type,
    )
```

### 3.3 Scene Type Classification

```python
SCENE_TYPE_RULES = {
    # If the scene contains a CATASTROPHE event → "catastrophe"
    "catastrophe": lambda events: any(e.type == EventType.CATASTROPHE for e in events),
    # If the scene contains a CONFLICT → "confrontation"
    "confrontation": lambda events: any(e.type == EventType.CONFLICT for e in events),
    # If the scene contains a REVEAL and irony collapsed → "revelation"
    "revelation": lambda events: any(
        e.type == EventType.REVEAL and e.metrics.get("irony_collapse", {}).get("detected", False)
        for e in events
    ),
    # If the scene contains CONFIDE events → "bonding"
    "bonding": lambda events: any(e.type == EventType.CONFIDE for e in events),
    # If max tension > 0.6 → "escalation"
    "escalation": lambda events: max(e.metrics["tension"] for e in events) > 0.6,
    # Default → "maintenance" (social upkeep, texture)
    "maintenance": lambda events: True,
}

def classify_scene_type(events: list[Event]) -> str:
    """Apply rules in priority order. First match wins."""
    for scene_type, rule in SCENE_TYPE_RULES.items():
        if rule(events):
            return scene_type
    return "maintenance"
```

### 3.4 Tiny Scene Merging

Scenes with very few events (< `min_scene_size`) are merged into the adjacent scene they share the most participants with.

```python
def merge_tiny_scenes(scenes: list[Scene], min_size: int = 3) -> list[Scene]:
    """
    Merge scenes with fewer than min_size events into their best neighbor.
    """
    merged = []
    i = 0
    while i < len(scenes):
        scene = scenes[i]
        if len(scene.event_ids) < min_size and merged:
            # Merge into previous scene (simpler; preserves chronological grouping)
            prev = merged[-1]
            merged[-1] = merge_two_scenes(prev, scene)
        elif len(scene.event_ids) < min_size and i + 1 < len(scenes):
            # No previous scene; merge into next
            scenes[i + 1] = merge_two_scenes(scene, scenes[i + 1])
        else:
            merged.append(scene)
        i += 1
    return merged

def merge_two_scenes(a: Scene, b: Scene) -> Scene:
    """Combine two adjacent scenes into one."""
    combined_events = a.event_ids + b.event_ids
    return Scene(
        id=a.id,  # keep the earlier scene's ID
        event_ids=combined_events,
        location=a.location if len(a.event_ids) >= len(b.event_ids) else b.location,
        participants=a.participants | b.participants,
        time_start=a.time_start,
        time_end=b.time_end,
        tension_arc=a.tension_arc + b.tension_arc,
        dominant_theme=a.dominant_theme,  # could recalculate
        scene_type=classify_scene_type(resolve_events(combined_events)),  # resolve IDs to Event objects via event index
    )
```

---

## 4. Configuration

```python
@dataclass
class SegmentationConfig:
    """Tunable parameters for scene segmentation."""

    # Participant overlap
    jaccard_threshold: float = 0.3

    # Tension gap detection
    tension_window: int = 5          # events to look back for peak
    drop_ratio: float = 0.3          # tension must drop to 30% of peak
    sustained_count: int = 3         # consecutive low-tension events required

    # Time gap
    time_gap_minutes: float = 5.0

    # Merge pass
    min_scene_size: int = 3          # minimum events per scene

    # Irony collapse
    irony_collapse_threshold: float = 0.5  # minimum drop to trigger boundary
```

---

## 5. Scene Output Format

### 5.1 Python Dataclass

```python
@dataclass
class Scene:
    id: str                         # "scene_000", "scene_001", ...
    event_ids: list[str]            # ordered list of event IDs
    location: str                   # primary location ID
    participants: set[str]          # all agents who appear in any event
    time_start: float               # sim_time of first event
    time_end: float                 # sim_time of last event
    tension_arc: list[float]        # per-event tension values (ordered)
    dominant_theme: str             # thematic axis with largest shift
    scene_type: str                 # "confrontation", "revelation", "bonding", etc.

    # Derived (computed after construction)
    @property
    def duration(self) -> float:
        return self.time_end - self.time_start

    @property
    def event_count(self) -> int:
        return len(self.event_ids)

    @property
    def peak_tension(self) -> float:
        return max(self.tension_arc) if self.tension_arc else 0.0

    @property
    def mean_tension(self) -> float:
        return sum(self.tension_arc) / len(self.tension_arc) if self.tension_arc else 0.0

    @property
    def tension_direction(self) -> str:
        """Is this scene rising, falling, or flat in tension?"""
        if len(self.tension_arc) < 2:
            return "flat"
        first_half = sum(self.tension_arc[:len(self.tension_arc) // 2])
        second_half = sum(self.tension_arc[len(self.tension_arc) // 2:])
        if second_half > first_half * 1.2:
            return "rising"
        elif second_half < first_half * 0.8:
            return "falling"
        return "flat"
```

### 5.2 JSON Representation

```json
{
    "id": "scene_003",
    "event_ids": ["E043", "E044", "E045"],
    "location": "kitchen",
    "participants": ["thorne", "elena", "marcus", "lydia"],
    "time_start": 54.0,
    "time_end": 58.5,
    "tension_arc": [0.24, 0.36, 0.45],
    "dominant_theme": "truth_deception",
    "scene_type": "escalation",
    "duration": 4.5,
    "event_count": 3,
    "peak_tension": 0.45,
    "mean_tension": 0.35,
    "tension_direction": "rising"
}
```

---

## 6. Worked Example: 20 Events Segmented into Scenes

### Input Events

Using the dinner party scenario. All 6 characters start at the dining table.

| # | Event ID | Tick | Sim Time | Type | Source | Target(s) | Location | Tension |
|---|---|---|---|---|---|---|---|---|
| 1 | E001 | 3 | 4.5 | CHAT | lydia | thorne | dining_table | 0.10 |
| 2 | E002 | 5 | 7.5 | CHAT | thorne | victor | dining_table | 0.12 |
| 3 | E003 | 7 | 10.5 | CHAT | diana | elena | dining_table | 0.08 |
| 4 | E004 | 10 | 15.0 | OBSERVE | thorne | — | dining_table | 0.24 |
| 5 | E005 | 12 | 18.0 | SOCIAL_MOVE | elena | — | kitchen | 0.05 |
| 6 | E006 | 13 | 19.5 | SOCIAL_MOVE | marcus | — | kitchen | 0.08 |
| 7 | E007 | 14 | 21.0 | CONFIDE | elena | marcus | kitchen | 0.36 |
| 8 | E008 | 16 | 24.0 | LIE | marcus | lydia | kitchen | 0.45 |
| 9 | E009 | 17 | 25.5 | OBSERVE | lydia | — | kitchen | 0.30 |
| 10 | E010 | 19 | 28.5 | SOCIAL_MOVE | lydia | — | dining_table | 0.05 |
| 11 | E011 | 20 | 30.0 | CHAT | lydia | diana | dining_table | 0.12 |
| 12 | E012 | 22 | 33.0 | SOCIAL_MOVE | diana | — | balcony | 0.04 |
| 13 | E013 | 23 | 34.5 | SOCIAL_MOVE | thorne | — | balcony | 0.06 |
| 14 | E014 | 25 | 37.5 | CONFLICT | diana | thorne | balcony | 0.58 |
| 15 | E015 | 27 | 40.5 | CONFLICT | thorne | diana | balcony | 0.65 |
| 16 | E016 | 28 | 42.0 | OBSERVE | victor | — | balcony | 0.48 |
| 17 | E017 | 30 | 45.0 | SOCIAL_MOVE | thorne | — | dining_table | 0.20 |
| 18 | E018 | 31 | 46.5 | INTERNAL | thorne | — | dining_table | 0.35 |
| 19 | E019 | 33 | 49.5 | OBSERVE | thorne | — | dining_table | 0.42 |
| 20 | E020 | 35 | 52.5 | CATASTROPHE | thorne | elena, marcus | dining_table | 0.78 |

### Segmentation Walkthrough

**Processing E001-E004:** All at dining_table. Participants overlap heavily (Jaccard > 0.3 between consecutive pairs). Tension is low (0.08-0.24). No breaks fire.

**E004 → E005:** E005 is Elena moving to kitchen. **Location change** from dining_table to kitchen. But wait -- E005 is a SOCIAL_MOVE event, which is itself at the NEW location (kitchen). E004 is at dining_table.

Check: kitchen.overhear_from includes dining_table? Yes (per doc3.md dinner party setup). So the adjacency exception applies. However, participant overlap: E004 participants = {thorne}, E005 participants = {elena}. Jaccard = 0/2 = 0.0 < 0.3.

**Participant break fires.** Scene 1 ends.

> **Scene 1: "Arrival Small Talk"**
> Events: E001, E002, E003, E004
> Location: dining_table
> Participants: {lydia, thorne, victor, diana, elena}
> Time: 4.5 - 15.0
> Tension arc: [0.10, 0.12, 0.08, 0.24]
> Scene type: "maintenance" (no conflict, no reveal, max tension 0.24 < 0.6)
> Dominant theme: none (minimal shifts)

**Processing E005-E009:** E005-E006 are SOCIAL_MOVEs to kitchen. E007-E009 all in kitchen. Participants: {elena, marcus, lydia} overlap well.

**E009 → E010:** E010 is Lydia moving back to dining_table. Location change (kitchen → dining_table). Adjacency exception applies, but participant Jaccard: E009 = {lydia}, E010 = {lydia}. Jaccard = 1.0. No break from participants.

Actually, E010's location is dining_table and E009's is kitchen. The overhear adjacency holds. And participant overlap is perfect (Lydia is in both). But E010 is a SOCIAL_MOVE -- it's a transition event. Let's check the next event: E011 is at dining_table with different participants than E007-E009.

The tension gap check: after E008 (0.45) and E009 (0.30), E010 is 0.05. Peak in last 5 = 0.45, threshold = 0.45 * 0.3 = 0.135. E010 = 0.05 < 0.135. But we need 3 consecutive low events. We only have 1 (E010). So tension gap doesn't fire yet.

**E010 → E011:** Now we have E010 at dining_table (tension 0.05) and E011 at dining_table (tension 0.12). Different location from the kitchen scene. And participant sets: E009 = {lydia}, E010 = {lydia}, E011 = {lydia, diana}. Jaccard(E009, E010) = 1.0, but location changed (kitchen → dining_table).

The location break between E009 (kitchen) and E010 (dining_table): overhear adjacency holds. So no pure location break. But the dramatic context has shifted: Lydia left the kitchen intrigue and returned to the dining room.

Actually, let me re-examine. E010 is a SOCIAL_MOVE to dining_table. Let's check participant break between E009 and E010: both just Lydia. Jaccard = 1.0. No participant break. And E010 and E011: {lydia} vs {lydia, diana}. Jaccard = 1/2 = 0.5 > 0.3. No break.

But the tension gap from E008's peak: E009 (0.30), E010 (0.05), E011 (0.12). After E011, looking back 5: peak = 0.45, threshold = 0.135. E010 (0.05) and E011 (0.12) are below threshold, but only 2 consecutive. Not 3 yet.

Let me check E011 → E012. E012 is Diana moving to balcony. Location: dining_table → balcony. Not adjacent via overhear. **Location break fires.** But also participant break: E011 = {lydia, diana}, E012 = {diana}. Jaccard = 1/2 = 0.5 > 0.3. Location break takes priority.

Wait -- E010 changed location from kitchen to dining_table. The scene that started with E005 (kitchen) now has events at dining_table (E010, E011). The segmenter checks consecutive pairs. Let me re-check E009 → E010: kitchen → dining_table. Not in overhear exception? Kitchen's overhear_from includes dining_table (you can hear the dining room from the kitchen), so the exception applies. No location break.

However, this creates a scene that spans kitchen and dining_table, which is physically awkward. In practice, the SOCIAL_MOVE event itself is the boundary. Let me refine:

**Refined rule:** SOCIAL_MOVE events that change the agent's location always trigger a scene boundary. The move belongs to the NEW scene (the agent is arriving).

With this rule:

**E004 → E005:** E005 is SOCIAL_MOVE to kitchen. **Forced break.** Scene 1 ends at E004.

**E005-E009 in kitchen.** Then E010 is SOCIAL_MOVE to dining_table. **Forced break.** Scene 2 ends at E009.

> **Scene 2: "Kitchen Conspiracy"**
> Events: E005, E006, E007, E008, E009
> Location: kitchen
> Participants: {elena, marcus, lydia}
> Time: 18.0 - 25.5
> Tension arc: [0.05, 0.08, 0.36, 0.45, 0.30]
> Scene type: "bonding" (contains CONFIDE event E007)
> Dominant theme: truth_deception

**E010-E011 at dining_table.** E012 is SOCIAL_MOVE to balcony. **Forced break.** Scene 3 ends at E011.

But Scene 3 has only 2 events (E010, E011), below min_scene_size of 3. **Merge pass:** merge into Scene 2 (previous) since Scene 4 is at a different location.

Actually -- merging a dining_table scene into a kitchen scene is semantically wrong. Better to merge into Scene 4 (next, at balcony)? Also wrong.

For scenes below min_size that span a transition, keep them as-is. Adjust rule: min_scene_size only triggers merging when the scenes share a location.

Revised: E010-E011 become a mini-scene. Since they're below threshold and adjacent scenes are at different locations, they remain as a **transition scene.**

> **Scene 3: "Return to Table"**
> Events: E010, E011
> Location: dining_table
> Participants: {lydia, diana}
> Time: 28.5 - 30.0
> Tension arc: [0.05, 0.12]
> Scene type: "maintenance"
> Dominant theme: none
> Note: Below min_scene_size but no compatible neighbor; kept as transition.

**E012-E016 at balcony.** E012 and E013 are SOCIAL_MOVEs (Diana and Thorne arriving). E014-E015 are CONFLICT. E016 is OBSERVE (Victor watching).

E016 → E017: E017 is SOCIAL_MOVE to dining_table. **Forced break.** Scene 4 ends at E016.

> **Scene 4: "Balcony Confrontation"**
> Events: E012, E013, E014, E015, E016
> Location: balcony
> Participants: {diana, thorne, victor}
> Time: 33.0 - 42.0
> Tension arc: [0.04, 0.06, 0.58, 0.65, 0.48]
> Scene type: "confrontation" (contains CONFLICT events)
> Dominant theme: loyalty_betrayal

**E017-E020 at dining_table.** E017 is Thorne returning. E018 is INTERNAL (thought). E019 is OBSERVE. E020 is CATASTROPHE.

No more events. Scene 5 closes at E020.

> **Scene 5: "The Accusation"**
> Events: E017, E018, E019, E020
> Location: dining_table
> Participants: {thorne, elena, marcus}
> Time: 45.0 - 52.5
> Tension arc: [0.20, 0.35, 0.42, 0.78]
> Scene type: "catastrophe" (contains CATASTROPHE event)
> Dominant theme: truth_deception

### Segmentation Result Summary

| Scene | Name (informal) | Events | Location | Duration | Peak Tension | Type |
|---|---|---|---|---|---|---|
| scene_000 | Arrival Small Talk | E001-E004 | dining_table | 10.5 min | 0.24 | maintenance |
| scene_001 | Kitchen Conspiracy | E005-E009 | kitchen | 7.5 min | 0.45 | bonding |
| scene_002 | Return to Table | E010-E011 | dining_table | 1.5 min | 0.12 | maintenance |
| scene_003 | Balcony Confrontation | E012-E016 | balcony | 9.0 min | 0.65 | confrontation |
| scene_004 | The Accusation | E017-E020 | dining_table | 7.5 min | 0.78 | catastrophe |

**Tension shape across scenes:** 0.24 → 0.45 → 0.12 → 0.65 → 0.78. Classic dramatic escalation with a mid-act dip (scene 002). The dinner party structure is visible: polite beginning, private intrigue, brief return to normalcy, confrontation, explosion.

---

## 7. SOCIAL_MOVE Handling

SOCIAL_MOVE events (agent changes location) are special:

1. They always trigger a scene boundary (the move belongs to the NEW scene).
2. They carry minimal tension themselves (moving is not dramatic, usually).
3. If two SOCIAL_MOVEs are consecutive (agents going to the same place), they belong to the same new scene.

```python
def is_scene_boundary_move(event: Event) -> bool:
    """Does this SOCIAL_MOVE event trigger a forced scene boundary?"""
    return event.type == EventType.SOCIAL_MOVE
```

---

## 8. Multi-Location Scenes (Future: The Bridge)

For MVP 1.5 ("The Bridge"), scenes may span two locations connected by a bottleneck. The adjacency exception generalizes:

```python
def locations_are_scene_compatible(loc_a: str, loc_b: str, world: WorldState) -> bool:
    """Can events at these two locations belong to the same scene?"""
    a = world.locations[loc_a]
    b = world.locations[loc_b]
    return loc_b in a.overhear_from or loc_a in b.overhear_from
```

For The Bridge, the bridge itself is a location. Events on the bridge are compatible with events at either end.

---

## 9. Edge Cases

### All events at one location
If all events occur at the dining_table (no one moves), the only boundary triggers are participant turnover, tension gaps, and irony collapses. This produces fewer, longer scenes.

### Rapid location switching
If an agent moves to the balcony, has one event, and returns, this creates a scene with 1 event (below min_size). The merge pass absorbs it into the adjacent scene.

### Simultaneous events (same tick, different order_in_tick)
Events within the same tick are never separated by a scene boundary. They belong to the same scene by definition.

### Zero-tension events
INTERNAL events (thoughts) and low-tension CHAT events won't trigger tension gap boundaries by themselves. They're absorbed into the current scene naturally.

### Empty participant intersection
If source_agent is the only participant (no target_agents, as in OBSERVE or INTERNAL), the Jaccard calculation uses a single-element set. This means two consecutive solo events by different agents will have Jaccard 0.0 — triggering a participant break. This is correct: if Thorne thinks to himself, then Elena thinks to herself, those are different dramatic units.

---

## 10. NOT In Scope

- **Scene naming/titling:** The informal names in the worked example ("Kitchen Conspiracy") are human-authored. Automatic scene naming (via LLM summarization) is a Phase 4 feature.
- **Nested scenes:** No sub-scene hierarchy. All scenes are flat. If needed later, add a `parent_scene_id` field.
- **Cross-cutting scenes:** In film, you can intercut between two simultaneous scenes. This spec produces sequential scenes only. Parallel scenes at different locations that happen simultaneously appear as separate scenes ordered by their first event's sim_time.
- **Scene reordering for syuzhet:** The spec produces scenes in chronological (fabula) order. Reordering for non-linear storytelling is the story extraction layer's job.

---

## 11. Dependencies

| Depends On | What It Provides |
|---|---|
| `specs/schema/events.md` | Event schema, EventType (especially SOCIAL_MOVE) |
| `specs/schema/scenes.md` | Scene dataclass definition |
| `specs/schema/world.md` | Location adjacency, overhear_from |
| `specs/metrics/tension-pipeline.md` | Per-event tension values (used for gap detection) |
| `specs/metrics/irony-and-beliefs.md` | Irony collapse detection (used for forced boundaries) |

| Depended On By | What It Consumes |
|---|---|
| `specs/metrics/story-extraction.md` | Scene list for arc grammar validation and beat sheet export |
| `specs/visualization/renderer-architecture.md` | Scene boundaries for sidebar display |
| `specs/integration/data-flow.md` | Scene output format for interface contracts |
