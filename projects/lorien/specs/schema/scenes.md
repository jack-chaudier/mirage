# Scene & Snapshot Schema Specification

> **Status:** CANONICAL — defines the Scene layer, scene segmentation triggers, SnapshotState, and CQRS index tables.
> **Implements:** Decisions #2 (CQRS read layer / snapshots), #15 (precomputed hover neighborhoods), #16 (first-class scene segmentation)
> **Consumers:** visualization (renderer, interaction-model, thread-layout), metrics (scene-segmentation, tension-pipeline, story-extraction), simulation (tick-loop for snapshots)
> **Depends on:** `specs/schema/events.md`, `specs/schema/agents.md`, `specs/schema/world.md`

---

## Table of Contents

1. [Scene](#1-scene)
2. [Scene Segmentation Triggers](#2-scene-segmentation-triggers)
3. [SnapshotState](#3-snapshotstate)
4. [Index Tables](#4-index-tables)
5. [Precomputed Hover Neighborhoods](#5-precomputed-hover-neighborhoods)
6. [JSON Examples](#6-json-examples)
7. [Validation Rules](#7-validation-rules)
8. [NOT In Scope](#8-not-in-scope)

---

## 1. Scene

A scene is a grouping of events that forms a dramatic unit (Decision #16). Scenes are the intermediate layer between raw events and story arcs. Writers think in scenes, not events.

### Python

```python
from dataclasses import dataclass, field

@dataclass
class Scene:
    """A group of events forming a dramatic unit."""
    id: str                                      # Unique scene ID. Format: "scene_{NNN}". E.g. "scene_001"
    event_ids: list[str]                         # Event IDs in this scene, in chronological order
    location: str                                # Primary location (location where most events occur)
    participants: list[str]                      # Agent IDs present during the scene (union of all event participants)
    time_start: float                            # sim_time of the first event in the scene
    time_end: float                              # sim_time of the last event in the scene
    tick_start: int                              # tick_id of the first event
    tick_end: int                                # tick_id of the last event

    # Computed by metrics pipeline
    tension_arc: list[float]                     # Tension values for each event in the scene, in order.
                                                 # Length == len(event_ids). Each value in [0.0, 1.0].
    tension_peak: float = 0.0                    # Max tension in the scene
    tension_mean: float = 0.0                    # Average tension in the scene
    dominant_theme: str = ""                     # Thematic axis with the largest absolute shift across the scene.
                                                 # E.g. "loyalty_betrayal". Empty if no thematic shift.
    scene_type: str = ""                         # Classification: see Scene Types below.

    # Summary (for beat sheets and story extraction)
    summary: str = ""                            # One-sentence human-readable summary.
                                                 # E.g. "Thorne confronts Marcus on the balcony about missing funds."
```

### TypeScript

```typescript
interface Scene {
    id: string;
    event_ids: string[];
    location: string;
    participants: string[];
    time_start: number;
    time_end: number;
    tick_start: number;
    tick_end: number;

    tension_arc: number[];           // [0.0, 1.0] per event
    tension_peak: number;
    tension_mean: number;
    dominant_theme: string;
    scene_type: string;
    summary: string;
}
```

### Scene Types

| Scene Type | Description | Typical Tension Profile |
|-----------|-------------|------------------------|
| `catastrophe` | Contains a CATASTROPHE event. Highest dramatic intensity. | Highest peak |
| `confrontation` | Direct conflict between characters. CONFLICT events. | Sharp rise to peak |
| `revelation` | A secret is revealed. REVEAL event with irony collapse. | Spike followed by rapid change |
| `bonding` | Characters build/strengthen relationships. CONFIDE events dominate. | Low, flat or gently rising |
| `escalation` | Tension rising above 0.6. Suspicion, information leaks, stress. | Gradually rising |
| `maintenance` | Social upkeep, texture, no dominant dramatic event. Default. | Low, flat |

> **Authority:** Scene types are defined in `scene-segmentation.md` (Decision 21). This table mirrors that spec's `classify_scene_type()` priority order.

Scene type is assigned by the metrics pipeline based on (in priority order):
1. Presence of CATASTROPHE events → `catastrophe`
2. Presence of CONFLICT events → `confrontation`
3. Presence of REVEAL with irony collapse → `revelation`
4. Presence of CONFIDE events → `bonding`
5. Max tension > 0.6 → `escalation`
6. Default → `maintenance`

---

## 2. Scene Segmentation Triggers

Scenes are segmented from the raw event stream. A new scene begins when any of the following triggers fire. The segmentation algorithm runs post-hoc on the complete event log (not during simulation).

### Trigger 1: Location Change

A new scene begins when the primary location of events shifts. "Primary location" = the location where the majority of events in a rolling window are occurring.

```python
def location_change_trigger(
    current_scene_location: str,
    event: Event
) -> bool:
    """New scene if the event is at a different location than the current scene."""
    return event.location_id != current_scene_location
```

**Exception:** If a single agent briefly visits another location (e.g., bathroom trip) and returns within 3 events, do NOT start a new scene. This is a "brief departure," not a scene change. The departure events are included in the current scene.

### Trigger 2: Significant Participant Turnover

A new scene begins when the set of agents present changes substantially.

```python
def participant_turnover_trigger(
    current_participants: set[str],
    new_participants: set[str],
    turnover_threshold: float = 0.3  # Jaccard overlap threshold (Decision 21, per scene-segmentation.md)
) -> bool:
    """
    New scene if the overlap between current and new participant sets
    drops below the threshold.
    """
    if not current_participants or not new_participants:
        return True
    overlap = len(current_participants & new_participants)
    total = len(current_participants | new_participants)
    return (overlap / total) < turnover_threshold
```

### Trigger 3: Tension Gap

A new scene begins if tension drops to near-zero for multiple consecutive ticks, indicating a lull.

```python
def tension_gap_trigger(
    recent_tensions: list[float],
    gap_threshold: float = 0.05,
    gap_duration: int = 4
) -> bool:
    """
    New scene if the last `gap_duration` events all have tension below `gap_threshold`.
    This detects natural breaks in the action.
    """
    if len(recent_tensions) < gap_duration:
        return False
    return all(t < gap_threshold for t in recent_tensions[-gap_duration:])
```

### Trigger 4: Time Gap

A new scene begins if there's a significant gap in sim_time between events (more than 5 sim-minutes with no events).

```python
def time_gap_trigger(
    previous_event_time: float,
    current_event_time: float,
    max_gap_minutes: float = 5.0
) -> bool:
    """New scene if there's a long gap between events."""
    return (current_event_time - previous_event_time) > max_gap_minutes
```

### Trigger 5: Catastrophe Event

A CATASTROPHE event always starts a new scene (if one isn't already starting). Catastrophes are dramatic ruptures that break the current scene's continuity.

```python
def catastrophe_trigger(event: Event) -> bool:
    """CATASTROPHE events always trigger a new scene."""
    return event.type == EventType.CATASTROPHE
```

### Trigger 6: SOCIAL_MOVE Forced Boundary

A SOCIAL_MOVE event (agent changing location) always forces a scene boundary. This ensures location transitions are never buried mid-scene. Defined in `scene-segmentation.md` Section 7.

```python
def social_move_trigger(event: Event) -> bool:
    """SOCIAL_MOVE events always force a scene boundary."""
    return event.type == EventType.SOCIAL_MOVE
```

> **Authority:** This trigger is defined in `scene-segmentation.md` (Decision 21), which is authoritative for all segmentation rules.

### Segmentation Algorithm (Pseudocode)

```python
def segment_into_scenes(events: list[Event]) -> list[Scene]:
    """
    Group a chronologically-ordered event list into scenes.
    Runs post-hoc on the complete event log.
    """
    scenes: list[Scene] = []
    current_events: list[Event] = []
    current_location: str = ""
    current_participants: set[str] = set()

    for event in events:
        # Check triggers
        start_new_scene = False

        if not current_events:
            start_new_scene = True  # First event always starts a scene

        elif catastrophe_trigger(event):
            start_new_scene = True

        elif location_change_trigger(current_location, event):
            # Check brief departure exception
            if not is_brief_departure(event, current_events, events):
                start_new_scene = True

        elif time_gap_trigger(current_events[-1].sim_time, event.sim_time):
            start_new_scene = True

        elif tension_gap_trigger([e.metrics["tension"] for e in current_events]):
            start_new_scene = True

        else:
            # Check participant turnover
            new_participants = get_event_participants(event)
            if participant_turnover_trigger(current_participants, new_participants):
                start_new_scene = True

        # Handle scene boundary
        if start_new_scene and current_events:
            scenes.append(build_scene(current_events, len(scenes)))
            current_events = []
            current_participants = set()

        # Add event to current scene
        current_events.append(event)
        current_location = event.location_id
        current_participants |= get_event_participants(event)

    # Final scene
    if current_events:
        scenes.append(build_scene(current_events, len(scenes)))

    return scenes


def build_scene(events: list[Event], index: int) -> Scene:
    """Construct a Scene from a list of events."""
    all_participants = set()
    for e in events:
        all_participants.add(e.source_agent)
        all_participants.update(e.target_agents)

    # Primary location = most common location
    location_counts: dict[str, int] = {}
    for e in events:
        location_counts[e.location_id] = location_counts.get(e.location_id, 0) + 1
    primary_location = max(location_counts, key=location_counts.get)

    tension_arc = [e.metrics["tension"] for e in events]

    return Scene(
        id=f"scene_{index + 1:03d}",
        event_ids=[e.id for e in events],
        location=primary_location,
        participants=sorted(all_participants),
        time_start=events[0].sim_time,
        time_end=events[-1].sim_time,
        tick_start=events[0].tick_id,
        tick_end=events[-1].tick_id,
        tension_arc=tension_arc,
        tension_peak=max(tension_arc) if tension_arc else 0.0,
        tension_mean=sum(tension_arc) / len(tension_arc) if tension_arc else 0.0,
        # dominant_theme and scene_type computed by metrics pipeline
    )


def get_event_participants(event: Event) -> set[str]:
    """Get all agents involved in an event."""
    participants = {event.source_agent}
    participants.update(event.target_agents)
    return participants


def is_brief_departure(event: Event, current_events: list[Event], all_events: list[Event]) -> bool:
    """
    Check if a location change is a brief departure (agent goes and comes back within 3 events).
    If so, don't start a new scene.
    """
    departure_location = event.location_id
    current_location = current_events[-1].location_id if current_events else ""

    # Look ahead up to 3 events
    event_index = all_events.index(event)
    lookahead = all_events[event_index + 1 : event_index + 4]

    # If any lookahead event returns to the current scene's location, it's brief
    return any(e.location_id == current_location for e in lookahead)
```

### Expected Scene Count

For the dinner party MVP (100-200 events, ~150 minutes):
- Estimated 8-15 scenes
- Average scene length: 10-20 events
- Shortest scenes: 3-5 events (brief transitions, aftermath)
- Longest scenes: 25-35 events (sustained dining table conversation with rising tension)

---

## 3. SnapshotState

Complete world state cached periodically (Decision #2). Used for fast state recovery without replaying from tick 0.

### Python

```python
from dataclasses import dataclass, field

@dataclass
class SnapshotState:
    """Complete world state at a point in time. Cached every N events for CQRS."""
    # === Temporal Position ===
    snapshot_id: str                              # Format: "snap_{NNN}". E.g. "snap_005"
    tick_id: int                                  # Tick at which this snapshot was taken
    sim_time: float                               # Sim-time at this tick
    event_count: int                              # Number of events processed up to this snapshot

    # === Agent States ===
    agents: dict[str, "AgentState"]              # agent_id → complete AgentState at this tick
                                                  # Includes goals, flaws, pacing, emotions,
                                                  # relationships, beliefs, alcohol, commitments

    # === World State ===
    secrets: dict[str, "SecretDefinition"]        # secret_id → SecretDefinition (may have changed)
    locations: dict[str, "Location"]              # location_id → Location (static, but included for completeness)

    # === Derived State ===
    global_tension: float = 0.0                   # Scene-level tension at this point
    active_scene_id: str = ""                     # Which scene is active at this tick
    belief_matrix: dict[str, dict[str, str]] = field(default_factory=dict)
                                                  # Materialized: agent_id → {secret_id → BeliefState.value}
                                                  # Redundant with agents[].beliefs but useful for fast irony queries
```

### TypeScript

```typescript
interface SnapshotState {
    snapshot_id: string;
    tick_id: number;
    sim_time: number;
    event_count: number;

    agents: Record<string, AgentState>;
    secrets: Record<string, SecretDefinition>;
    locations: Record<string, Location>;

    global_tension: number;
    active_scene_id: string;
    belief_matrix: Record<string, Record<string, string>>;
}
```

### Snapshot Frequency

Snapshots are taken every `snapshot_interval` events (default: 20). With 100-200 events in the dinner party, this produces 5-10 snapshots.

```python
def should_snapshot(event_count: int, snapshot_interval: int = 20) -> bool:
    """Check if it's time to take a snapshot."""
    return event_count > 0 and event_count % snapshot_interval == 0
```

### State Recovery

To recover world state at any event:

```python
def recover_state_at(event_id: str, snapshots: list[SnapshotState], events: list[Event]) -> "WorldState":
    """
    Recover the world state just after a specific event was processed.
    Uses nearest prior snapshot + replay.
    """
    target_event = get_event(event_id)
    target_event_index = events.index(target_event)

    # Find nearest snapshot at or before this event
    best_snapshot = None
    for snap in reversed(snapshots):
        if snap.event_count <= target_event_index + 1:
            best_snapshot = snap
            break

    if best_snapshot is None:
        # No snapshot found — replay from initial state
        state = create_initial_world_state()
        replay_start = 0
    else:
        state = restore_from_snapshot(best_snapshot)
        replay_start = best_snapshot.event_count

    # Replay events from snapshot to target
    for i in range(replay_start, target_event_index + 1):
        apply_event(state, events[i])

    return state
```

**Performance:** With snapshots every 20 events, the maximum replay is 19 events. For the dinner party scale, this is instant (<1ms).

---

## 4. Index Tables

Built incrementally as events are appended (Decision #2). These enable O(1) or O(n) lookups for common queries without replaying events.

### Index Definitions

```python
from collections import defaultdict

class EventIndices:
    """CQRS read-optimized index tables, built incrementally."""

    def __init__(self):
        # Primary index: event_id → Event
        self.events: dict[str, Event] = {}

        # Agent timeline: agent_id → list of event_ids (in chronological order)
        # Includes events where agent is source_agent OR in target_agents
        self.agent_timeline: dict[str, list[str]] = defaultdict(list)

        # Location events: location_id → list of event_ids
        self.location_events: dict[str, list[str]] = defaultdict(list)

        # Event participants: event_id → list of agent_ids (all participants)
        self.event_participants: dict[str, list[str]] = {}

        # Secret events: secret_id → list of event_ids that touch this secret
        # (events with BELIEF or SECRET_STATE deltas referencing this secret)
        self.secret_events: dict[str, list[str]] = defaultdict(list)

        # Pair interactions: (agent_a, agent_b) → list of event_ids
        # where both agents are participants. Sorted tuple key ensures
        # (A,B) and (B,A) map to the same list.
        self.pair_interactions: dict[tuple[str, str], list[str]] = defaultdict(list)

        # Causal links (bidirectional):
        # Forward: event_id → list of event_ids it caused
        self.forward_links: dict[str, list[str]] = defaultdict(list)
        # Backward: event_id → list of event_ids that caused it (same as event.causal_links)
        self.backward_links: dict[str, list[str]] = defaultdict(list)

    def index_event(self, event: Event) -> None:
        """Add a single event to all indices. Called when event is appended."""
        eid = event.id

        # Primary
        self.events[eid] = event

        # Agent timeline
        self.agent_timeline[event.source_agent].append(eid)
        for target in event.target_agents:
            if target != event.source_agent:
                self.agent_timeline[target].append(eid)

        # Location
        self.location_events[event.location_id].append(eid)

        # Participants
        participants = [event.source_agent] + [t for t in event.target_agents if t != event.source_agent]
        self.event_participants[eid] = participants

        # Secrets
        for delta in event.deltas:
            if delta.kind in (DeltaKind.BELIEF, DeltaKind.SECRET_STATE):
                secret_id = delta.attribute
                if secret_id and eid not in self.secret_events[secret_id]:
                    self.secret_events[secret_id].append(eid)

        # Pair interactions
        all_agents = set(participants)
        for a in all_agents:
            for b in all_agents:
                if a < b:  # sorted pair
                    self.pair_interactions[(a, b)].append(eid)

        # Causal links
        self.backward_links[eid] = list(event.causal_links)
        for cause_id in event.causal_links:
            self.forward_links[cause_id].append(eid)
```

### TypeScript

```typescript
interface EventIndices {
    events: Record<string, Event>;
    agent_timeline: Record<string, string[]>;
    location_events: Record<string, string[]>;
    event_participants: Record<string, string[]>;
    secret_events: Record<string, string[]>;
    pair_interactions: Record<string, string[]>;  // key: "agent_a|agent_b" (sorted)
    forward_links: Record<string, string[]>;
    backward_links: Record<string, string[]>;
}
```

### Query Examples

```python
# "What events involve Thorne?"
thorne_events = indices.agent_timeline["thorne"]

# "What happened on the balcony?"
balcony_events = indices.location_events["balcony"]

# "Who was involved in event evt_0089?"
participants = indices.event_participants["evt_0089"]

# "What events touched the affair secret?"
affair_events = indices.secret_events["secret_affair_01"]

# "What interactions did Thorne and Marcus have?"
thorne_marcus = indices.pair_interactions[("marcus", "thorne")]  # sorted tuple

# "What caused event evt_0089?"
causes = indices.backward_links["evt_0089"]

# "What did event evt_0027 lead to?"
effects = indices.forward_links["evt_0027"]
```

---

## 5. Precomputed Hover Neighborhoods

At data load time (or after simulation completes), precompute causal neighborhoods for every event (Decision #15). Hover interactions are then pure O(1) lookups.

### Python

```python
@dataclass
class CausalNeighborhood:
    """Precomputed causal cone for an event. Used for hover highlighting."""
    event_id: str
    backward: list[str]              # Events that causally preceded this one (BFS depth 3)
    forward: list[str]               # Events that this one caused (BFS depth 3)

def precompute_neighborhoods(
    indices: EventIndices,
    depth: int = 3
) -> dict[str, CausalNeighborhood]:
    """
    Precompute causal neighborhoods for all events.
    BFS depth of 3 gives a useful causal cone without overwhelming the display.
    """
    neighborhoods: dict[str, CausalNeighborhood] = {}

    for event_id in indices.events:
        backward = bfs(event_id, indices.backward_links, depth)
        forward = bfs(event_id, indices.forward_links, depth)
        neighborhoods[event_id] = CausalNeighborhood(
            event_id=event_id,
            backward=backward,
            forward=forward,
        )

    return neighborhoods


def bfs(start: str, links: dict[str, list[str]], max_depth: int) -> list[str]:
    """Breadth-first search through causal links up to max_depth."""
    visited: set[str] = set()
    result: list[str] = []
    queue: list[tuple[str, int]] = [(start, 0)]

    while queue:
        current, depth = queue.pop(0)
        if current in visited or depth > max_depth:
            continue
        visited.add(current)
        if current != start:
            result.append(current)
        if depth < max_depth:
            for neighbor in links.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    return result
```

### TypeScript

```typescript
interface CausalNeighborhood {
    event_id: string;
    backward: string[];              // causal ancestors (depth 3)
    forward: string[];               // causal descendants (depth 3)
}

// Precomputed map: event_id -> CausalNeighborhood
type HoverIndex = Record<string, CausalNeighborhood>;
```

### Performance

For 200 events with average 2 causal links each:
- BFS depth 3 visits at most ~8 nodes per direction (2^3)
- Total precomputation: 200 events x 16 BFS nodes = ~3200 operations
- Time: <10ms
- Memory: ~200 neighborhoods x ~16 IDs x ~10 bytes = ~32KB

Hover lookup is O(1): just look up `neighborhoods[event_id]`.

---

## 6. JSON Examples

### Example Scene: Balcony Confrontation

```json
{
    "id": "scene_007",
    "event_ids": ["evt_0085", "evt_0086", "evt_0087", "evt_0088", "evt_0089", "evt_0090", "evt_0091", "evt_0092"],
    "location": "balcony",
    "participants": ["thorne", "marcus"],
    "time_start": 58.0,
    "time_end": 66.0,
    "tick_start": 116,
    "tick_end": 132,
    "tension_arc": [0.25, 0.35, 0.42, 0.55, 0.72, 0.68, 0.60, 0.45],
    "tension_peak": 0.72,
    "tension_mean": 0.5025,
    "dominant_theme": "loyalty_betrayal",
    "scene_type": "confrontation",
    "summary": "Thorne corners Marcus on the balcony and accuses him of embezzling from the firm."
}
```

### Example Scene: Quiet Dinner Table Opening

```json
{
    "id": "scene_001",
    "event_ids": ["evt_0001", "evt_0002", "evt_0003", "evt_0004", "evt_0005", "evt_0006", "evt_0007", "evt_0008", "evt_0009", "evt_0010", "evt_0011", "evt_0012"],
    "location": "dining_table",
    "participants": ["thorne", "elena", "marcus", "lydia", "diana", "victor"],
    "time_start": 0.0,
    "time_end": 8.0,
    "tick_start": 0,
    "tick_end": 16,
    "tension_arc": [0.02, 0.03, 0.05, 0.04, 0.06, 0.08, 0.07, 0.10, 0.12, 0.09, 0.11, 0.13],
    "tension_peak": 0.13,
    "tension_mean": 0.075,
    "dominant_theme": "",
    "scene_type": "bonding",
    "summary": "The evening begins with pleasantries around the dinner table. Everyone is polite, but subtle undercurrents hint at tensions beneath the surface."
}
```

### Example Snapshot

```json
{
    "snapshot_id": "snap_003",
    "tick_id": 80,
    "sim_time": 40.0,
    "event_count": 60,
    "agents": {
        "thorne": {
            "id": "thorne",
            "name": "James Thorne",
            "location": "dining_table",
            "goals": { "safety": 0.4, "status": 0.9, "closeness": {"elena": 0.7, "marcus": 0.6, "lydia": 0.3, "diana": 0.4, "victor": 0.5}, "secrecy": 0.3, "truth_seeking": 0.6, "autonomy": 0.7, "loyalty": 0.8 },
            "flaws": [{"flaw_type": "pride", "strength": 0.8, "trigger": "status_threat", "effect": "overweight_status", "description": "..."}],
            "pacing": { "dramatic_budget": 0.75, "stress": 0.25, "composure": 0.82, "commitment": 0.1, "recovery_timer": 0 },
            "emotional_state": { "anger": 0.15, "fear": 0.05, "hope": 0.2, "shame": 0.0, "affection": 0.3, "suspicion": 0.3 },
            "relationships": {
                "elena": { "trust": 0.75, "affection": 0.55, "obligation": 0.3 },
                "marcus": { "trust": 0.55, "affection": 0.35, "obligation": 0.2 },
                "lydia": { "trust": 0.35, "affection": 0.2, "obligation": 0.0 },
                "diana": { "trust": 0.5, "affection": 0.3, "obligation": 0.1 },
                "victor": { "trust": 0.6, "affection": 0.3, "obligation": 0.1 }
            },
            "beliefs": {
                "secret_affair_01": "unknown",
                "secret_embezzle_01": "suspects",
                "secret_diana_debt": "unknown",
                "secret_lydia_knows": "unknown",
                "secret_victor_investigation": "unknown"
            },
            "alcohol_level": 0.15,
            "commitments": []
        }
    },
    "secrets": {},
    "locations": {},
    "global_tension": 0.35,
    "active_scene_id": "scene_004",
    "belief_matrix": {
        "thorne": {"secret_affair_01": "unknown", "secret_embezzle_01": "suspects", "secret_diana_debt": "unknown", "secret_lydia_knows": "unknown", "secret_victor_investigation": "unknown"}
    }
}
```

(Snapshot truncated for brevity — in practice, all agents, all secrets, and all locations would be included.)

### Example Causal Neighborhood

```json
{
    "event_id": "evt_0089",
    "backward": ["evt_0085", "evt_0072", "evt_0070", "evt_0055", "evt_0043", "evt_0027"],
    "forward": ["evt_0090", "evt_0091", "evt_0095", "evt_0098", "evt_0102"]
}
```

This shows that event evt_0089 (Thorne confronting Marcus) was caused by a chain going back through evt_0085, evt_0072, etc., and led to events evt_0090 through evt_0102.

---

## 7. Validation Rules

### Scene Validation

1. `id` must be unique and follow the `scene_{NNN}` format
2. `event_ids` must be non-empty (minimum 1 event per scene)
3. All event IDs in `event_ids` must reference valid events
4. Events in `event_ids` must be in chronological order (by tick_id, then order_in_tick)
5. `time_start <= time_end`
6. `tick_start <= tick_end`
7. `location` must reference a valid location ID
8. `participants` must be non-empty
9. `tension_arc` length must equal `event_ids` length
10. All tension_arc values must be in [0.0, 1.0]
11. `tension_peak` must equal `max(tension_arc)`
12. `scene_type` must be one of the recognized scene types (or empty if not yet classified)
13. Every event in the simulation must belong to exactly one scene (no gaps, no overlaps)
14. Scenes must be in chronological order: `scene_N.time_end <= scene_{N+1}.time_start`

### Snapshot Validation

1. `snapshot_id` must be unique
2. `tick_id` must be non-negative
3. `event_count` must be non-negative
4. `agents` must contain entries for all agents in the simulation
5. `secrets` must contain entries for all secrets
6. Each agent state within the snapshot must pass AgentState validation (see agents.md)
7. `belief_matrix` must be consistent with `agents[].beliefs`
8. Snapshots must be in chronological order by `tick_id`

### Index Table Validation

1. Every indexed event ID must reference a valid event
2. Agent timelines must be in chronological order
3. Location event lists must be in chronological order
4. Forward and backward links must be consistent: if A is in forward_links[B], then B must be in backward_links[A]
5. Pair interaction keys must use sorted agent ID tuples

---

## 8. NOT In Scope

- **Scene segmentation algorithm tuning** — specific threshold values may need adjustment based on testing. The values in Section 2 are starting points. Final tuning is in `specs/metrics/scene-segmentation.md`.
- **Scene-to-arc mapping** — how scenes compose into story arcs. See `specs/metrics/story-extraction.md`.
- **Visualization of scenes** — how scene boundaries are rendered. See `specs/visualization/renderer-architecture.md`.
- **Scene summary generation** — LLM-based summarization of scenes. Phase 4 concern.
- **Snapshot storage format** — whether snapshots are stored in-memory, on disk, or in a database. Implementation detail for the simulation engine.

---

## Dependencies

This spec is upstream of:
- `specs/metrics/scene-segmentation.md` (implements the segmentation algorithm defined here)
- `specs/metrics/tension-pipeline.md` (computes tension_arc values for scenes)
- `specs/metrics/story-extraction.md` (uses scenes as building blocks for arcs)
- `specs/visualization/renderer-architecture.md` (renders scene boundaries)
- `specs/visualization/interaction-model.md` (uses hover neighborhoods)
- `specs/integration/data-flow.md` (defines how indices and snapshots flow between subsystems)

This spec depends on:
- `specs/schema/events.md` (Scene contains event_ids; indices reference Events)
- `specs/schema/agents.md` (SnapshotState contains AgentState)
- `specs/schema/world.md` (SnapshotState contains Locations, SecretDefinitions)
