# Event Schema Specification

> **Status:** CANONICAL — all subsystems depend on these types.
> **Implements:** Decisions #1 (event-primary graph), #3 (typed deltas), #4 (discrete ticks), #14 (arc grammars / BeatType)
> **Consumers:** simulation (tick-loop, decision-engine), visualization (renderer, thread-layout, interaction-model), metrics (tension-pipeline, irony-and-beliefs, scene-segmentation, story-extraction)

---

## Table of Contents

1. [EventType Enum](#1-eventtype-enum)
2. [BeatType Enum](#2-beattype-enum)
3. [DeltaKind Enum](#3-deltakind-enum)
4. [DeltaOp Enum](#4-deltaop-enum)
5. [StateDelta](#5-statedelta)
6. [Event](#6-event)
7. [Causal Links](#7-causal-links)
8. [Metrics Dict](#8-metrics-dict)
9. [JSON Examples](#9-json-examples)
10. [Validation Rules](#10-validation-rules)
11. [NOT In Scope](#11-not-in-scope)

---

## 1. EventType Enum

Every event has exactly one type. The type determines which fields are required and what deltas are expected.

### Python

```python
from enum import Enum

class EventType(Enum):
    CHAT         = "chat"           # Low-tension social maintenance: small talk, greetings
    OBSERVE      = "observe"        # Overhearing, noticing, seeing — passive information gain
    SOCIAL_MOVE  = "social_move"    # Changing location or position (leaving table, going to balcony)
    REVEAL       = "reveal"         # Intentional information transfer (showing evidence, announcing)
    CONFLICT     = "conflict"       # Confrontation, accusation, argument
    INTERNAL     = "internal"       # Thought, decision, realization — invisible to other agents
    PHYSICAL     = "physical"       # Physical action: pour drink, slam door, flip table, hand object
    CONFIDE      = "confide"        # Private sharing of secret or feeling to a trusted party
    LIE          = "lie"            # Deliberate misinformation — source knows it's false
    CATASTROPHE  = "catastrophe"    # Involuntary break: breakdown, blurting, explosion, collapse
```

### TypeScript

```typescript
enum EventType {
    CHAT         = "chat",
    OBSERVE      = "observe",
    SOCIAL_MOVE  = "social_move",
    REVEAL       = "reveal",
    CONFLICT     = "conflict",
    INTERNAL     = "internal",
    PHYSICAL     = "physical",
    CONFIDE      = "confide",
    LIE          = "lie",
    CATASTROPHE  = "catastrophe",
}
```

### Behavioral Notes

| EventType | Visibility | Typical Dramatic Budget Cost | Expected Deltas |
|-----------|-----------|------------------------------|-----------------|
| CHAT | All agents at location | 0.0 | Relationship (small trust/affection shifts) |
| OBSERVE | Source agent only | 0.0 | Belief changes |
| SOCIAL_MOVE | All agents at origin & destination | 0.05 | AGENT_LOCATION |
| REVEAL | Target agents + overhear-eligible | 0.3 | BELIEF, possibly RELATIONSHIP |
| CONFLICT | All agents at location | 0.4-0.6 | RELATIONSHIP (trust/affection drops), AGENT_EMOTION (stress), PACING |
| INTERNAL | Source agent only (invisible) | 0.0 | AGENT_EMOTION, COMMITMENT |
| PHYSICAL | All agents at location | 0.0-0.3 | AGENT_RESOURCE, WORLD_RESOURCE, AGENT_EMOTION |
| CONFIDE | Source + target only | 0.2 | BELIEF, RELATIONSHIP (trust increase) |
| LIE | Target agents | 0.1-0.2 | BELIEF (false belief planted) |
| CATASTROPHE | All agents at location | 0.8-1.0 (drains budget) | Multiple: RELATIONSHIP, AGENT_EMOTION, SECRET_STATE, PACING |

**INTERNAL events** have `target_agents: []` and produce no deltas visible to other agents. They exist for the metrics pipeline (tracking internal state changes) and for story extraction (showing the reader what a character was thinking).

**CATASTROPHE events** are triggered by the pacing physics (Decision #10: cusp catastrophe), not by agent decision logic. They fire when `stress * commitment^2 > threshold AND composure < minimum`. The simulation engine generates these, not the decision engine.

---

## 2. BeatType Enum

Narrative function of an event within an arc. Assigned post-hoc by the metrics pipeline or the story extraction layer — NOT by the simulation. An event may have `beat_type: null` if it hasn't been classified or doesn't belong to a recognized arc.

### Python

```python
class BeatType(Enum):
    SETUP         = "setup"          # Introduces character, situation, or stakes
    COMPLICATION  = "complication"    # New obstacle, information, or constraint
    ESCALATION    = "escalation"     # Tension increases, stakes rise
    TURNING_POINT = "turning_point"  # Moment of irreversible change
    CONSEQUENCE   = "consequence"    # Aftermath, new equilibrium, fallout
```

### TypeScript

```typescript
enum BeatType {
    SETUP         = "setup",
    COMPLICATION  = "complication",
    ESCALATION    = "escalation",
    TURNING_POINT = "turning_point",
    CONSEQUENCE   = "consequence",
}
```

### Classification Heuristics

These are guidelines for the metrics pipeline. Exact implementation is defined in `specs/metrics/story-extraction.md`.

| BeatType | Signal |
|----------|--------|
| SETUP | First N events involving a character, or first event at a new location |
| COMPLICATION | Event introduces new BELIEF delta or new secret entering a scene |
| ESCALATION | Event where tension metric is higher than previous event in the arc |
| TURNING_POINT | Event with highest counterfactual impact in the arc, or a CATASTROPHE |
| CONSEQUENCE | Events after the turning point where tension is decreasing |

A valid arc (per Decision #14) must contain at least one event of each BeatType, in order. Events between required beats are permitted (not every event needs a beat_type).

---

## 3. DeltaKind Enum

The discriminated union tag for StateDelta. Determines which fields on StateDelta are required and how the delta is applied to world state.

### Python

```python
class DeltaKind(Enum):
    AGENT_EMOTION    = "agent_emotion"     # Changes to emotional_state dict
    AGENT_RESOURCE   = "agent_resource"    # Changes to agent resources (alcohol_level, etc.)
    AGENT_LOCATION   = "agent_location"    # Agent moves to a new location
    RELATIONSHIP     = "relationship"      # Changes to trust/affection/obligation between agents
    BELIEF           = "belief"            # Agent's belief about a secret changes
    SECRET_STATE     = "secret_state"      # Secret itself changes (revealed publicly, etc.)
    WORLD_RESOURCE   = "world_resource"    # Changes to world-level resources
    COMMITMENT       = "commitment"        # Agent makes an irreversible choice
    PACING           = "pacing"            # Changes to dramatic_budget, stress, composure, recovery_timer
```

### TypeScript

```typescript
enum DeltaKind {
    AGENT_EMOTION    = "agent_emotion",
    AGENT_RESOURCE   = "agent_resource",
    AGENT_LOCATION   = "agent_location",
    RELATIONSHIP     = "relationship",
    BELIEF           = "belief",
    SECRET_STATE     = "secret_state",
    WORLD_RESOURCE   = "world_resource",
    COMMITMENT       = "commitment",
    PACING           = "pacing",
}
```

### Field Requirements by DeltaKind

| DeltaKind | `agent` | `agent_b` | `attribute` | `value` type | Notes |
|-----------|---------|-----------|-------------|-------------|-------|
| AGENT_EMOTION | required | null | emotion name (e.g. "anger", "fear", "shame") | float | Applied to `agent.emotional_state[attribute]` |
| AGENT_RESOURCE | required | null | resource name (e.g. "alcohol_level") | float | Applied to agent resource fields |
| AGENT_LOCATION | required | null | "" (unused) | string (location_id) | SET only. Moves agent to new location |
| RELATIONSHIP | required | required | dimension (e.g. "trust", "affection", "obligation") | float | Applied to `agent.relationships[agent_b][attribute]` |
| BELIEF | required | null | secret_id | string (BeliefState value) | SET only. Applied to `agent.beliefs[attribute]` |
| SECRET_STATE | required | null | secret_id | string or bool | Changes to the secret object itself (e.g. `publicly_known: true`) |
| WORLD_RESOURCE | required | null | resource name | float | World-level resource changes |
| COMMITMENT | required | null | commitment description | string | Appends to `agent.commitments` |
| PACING | required | null | pacing field (e.g. "stress", "composure", "dramatic_budget", "recovery_timer") | float or int | Applied to `agent.pacing[attribute]` |

---

## 4. DeltaOp Enum

### Python

```python
class DeltaOp(Enum):
    SET = "set"    # Replace the current value entirely
    ADD = "add"    # Add to the current numeric value (float/int only)
```

### TypeScript

```typescript
enum DeltaOp {
    SET = "set",
    ADD = "add",
}
```

**Rules:**
- `ADD` is only valid when `value` is numeric (float or int).
- `SET` works for all value types (float, string, bool).
- AGENT_LOCATION deltas must use `SET`.
- BELIEF deltas must use `SET` (belief states are enums, not numeric).
- COMMITMENT deltas must use `SET` (appending a commitment string).

---

## 5. StateDelta

The atom of state change. Every Event produces zero or more StateDelta objects that describe exactly what changed in the world.

### Python

```python
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class StateDelta:
    """Typed, validatable, refactor-safe state change."""
    kind: DeltaKind
    agent: str                                   # primary agent affected (agent_id)
    agent_b: Optional[str] = None                # second agent (for RELATIONSHIP, or source of belief)
    attribute: str = ""                           # what's changing (emotion name, resource name, secret_id, etc.)
    op: DeltaOp = DeltaOp.ADD
    value: Union[float, str, bool, int] = 0.0
    reason_code: str = ""                         # machine-readable: "BETRAYAL_OBSERVED", "ALCOHOL_EFFECT"
    reason_display: str = ""                      # human-readable: "Saw Marcus whisper to Elena"
```

### TypeScript

```typescript
interface StateDelta {
    kind: DeltaKind;
    agent: string;                          // agent_id of primary agent affected
    agent_b?: string | null;                // second agent for relationships/beliefs
    attribute: string;                      // what's changing
    op: DeltaOp;                            // SET or ADD
    value: number | string | boolean;
    reason_code: string;                    // machine-readable tag
    reason_display: string;                 // human-readable tooltip text
}
```

### JSON Schema

```json
{
    "type": "object",
    "required": ["kind", "agent", "op", "value"],
    "properties": {
        "kind": { "type": "string", "enum": ["agent_emotion", "agent_resource", "agent_location", "relationship", "belief", "secret_state", "world_resource", "commitment", "pacing"] },
        "agent": { "type": "string" },
        "agent_b": { "type": ["string", "null"] },
        "attribute": { "type": "string" },
        "op": { "type": "string", "enum": ["set", "add"] },
        "value": { "type": ["number", "string", "boolean"] },
        "reason_code": { "type": "string" },
        "reason_display": { "type": "string" }
    }
}
```

---

## 6. Event

The atomic unit of the fabula graph. Events are nodes (Decision #1). World-state is derived by replaying events from snapshots (Decision #2).

### Python

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Event:
    """A single thing that happened. The atomic unit of the fabula graph."""

    # === Identity ===
    id: str                                       # Unique ID. Format: "evt_{zero-padded sequence}". E.g. "evt_0001"

    # === Temporal Position (Decision #4) ===
    sim_time: float                               # Simulation time in minutes from start. E.g. 0.0 = start, 45.5 = 45m30s in
    tick_id: int                                  # Which tick this belongs to. Starts at 0.
    order_in_tick: int                            # Resolution order within the tick. 0-indexed.

    # === Type ===
    type: EventType                               # Discriminator for event behavior

    # === Participants ===
    source_agent: str                             # Agent who initiated/experienced this. agent_id.
    target_agents: list[str]                      # Agents directly addressed/affected. May be empty (INTERNAL, OBSERVE).
    location_id: str                              # Where this happened. Must be a valid Location.id.

    # === Causality (Decision #1: event-primary graph) ===
    causal_links: list[str]                       # Event IDs that causally preceded this event. See Section 7.

    # === State Changes (Decision #3: typed deltas) ===
    deltas: list[StateDelta]                      # All state changes this event produces. Ordered.

    # === Content ===
    description: str                              # Human-readable summary. Used in tooltips and beat sheets.
                                                  # E.g. "Marcus confronts Elena about the missing funds"
    dialogue: Optional[str] = None                # Actual spoken words, if any. Used by story extraction for prose.
                                                  # E.g. '"Where did the money go, Elena?"'
    content_metadata: Optional[dict] = None       # Structured metadata for action context. E.g. {"secret_id": "secret_affair_01"}
                                                  # Used by decision engine to pass structured data without fragile string matching.

    # === Narrative Classification ===
    beat_type: Optional[BeatType] = None          # Assigned by metrics pipeline, not simulation. May be null.

    # === Computed Metrics (Decision #2: CQRS read layer) ===
    metrics: "EventMetrics" = field(default_factory=lambda: EventMetrics())


@dataclass
class EventMetrics:
    """Typed metrics container — populated by the metrics pipeline, not the simulation.
    Python dataclass mirrors the TypeScript EventMetrics interface."""
    tension: float = 0.0                           # [0.0, 1.0] — computed by tension pipeline
    irony: float = 0.0                             # [0.0, +inf) — sum of irony scores for agents present
    significance: float = 0.0                      # [0.0, 1.0] — counterfactual impact (Phase 5, always 0.0 until then)
    thematic_shift: dict = field(default_factory=dict)  # axis_name → delta. E.g. {"loyalty_betrayal": -0.3}
    tension_components: dict = field(default_factory=dict)  # 8 sub-metric name → float. E.g. {"danger": 0.6, ...}
    irony_collapse: Optional[dict] = None          # null unless irony collapse detected at this event
```

### TypeScript

```typescript
interface Event {
    // Identity
    id: string;

    // Temporal position
    sim_time: number;              // minutes from simulation start
    tick_id: number;
    order_in_tick: number;

    // Type
    type: EventType;

    // Participants
    source_agent: string;          // agent_id
    target_agents: string[];       // agent_ids
    location_id: string;

    // Causality
    causal_links: string[];        // event IDs

    // State changes
    deltas: StateDelta[];

    // Content
    description: string;
    dialogue?: string | null;
    content_metadata?: Record<string, unknown> | null;  // structured action context (e.g. {secret_id: "..."})

    // Narrative classification
    beat_type?: BeatType | null;

    // Computed metrics
    metrics: EventMetrics;
}

interface EventMetrics {
    tension: number;               // [0.0, 1.0]
    irony: number;                 // [0.0, +inf)
    significance: number;          // [0.0, 1.0]
    thematic_shift: Record<string, number>;  // axis → delta
    tension_components: Record<string, number>;  // 8 sub-metric name → value
    irony_collapse: IronyCollapseInfo | null;     // null unless collapse detected
}

interface IronyCollapseInfo {
    detected: boolean;
    drop: number;                  // magnitude of irony drop
    collapsed_beliefs: Array<{agent: string; secret: string; from: string; to: string}>;
    score: number;                 // [0.0, 1.0] collapse quality score
}
```

### ID Format

Event IDs follow the pattern `evt_{NNNN}` where NNNN is a zero-padded 4-digit sequence number. This supports up to 9999 events per simulation run, well beyond the MVP target of 100-200.

Examples: `evt_0001`, `evt_0042`, `evt_0137`

### Field Invariants

1. `sim_time >= 0.0` — no negative time
2. `tick_id >= 0` — ticks are non-negative integers
3. `order_in_tick >= 0` — order within a tick is non-negative
4. If `tick_id` of event A < `tick_id` of event B, then `sim_time` of A <= `sim_time` of B
5. `source_agent` must be a valid agent_id in the simulation
6. `target_agents` must not contain `source_agent` (you don't target yourself)
7. `location_id` must be a valid Location.id
8. All event IDs in `causal_links` must refer to events with a lower `tick_id`, or the same `tick_id` with a lower `order_in_tick`
9. `deltas` list may be empty (e.g., a CHAT that doesn't meaningfully change state)
10. `description` must be non-empty
11. For INTERNAL events: `target_agents` must be empty
12. For CATASTROPHE events: `source_agent`'s pacing state must have met catastrophe threshold at event generation time

---

## 7. Causal Links

Causal links are the edges of the event-primary graph (Decision #1). They connect an event to the prior events that caused it.

### What Constitutes a Valid Causal Link

An event B may list event A in its `causal_links` if ANY of the following hold:

1. **Direct response:** B is a verbal or physical response to A (e.g., A is a CONFLICT accusation, B is the response)
2. **Information dependency:** B was only possible because of information revealed/observed in A (e.g., A is an OBSERVE event where agent sees something, B is a CONFLICT where they confront based on what they saw)
3. **Emotional trigger:** A produced a delta that pushed the agent past a threshold that triggered B (e.g., A raised stress, B is a CATASTROPHE triggered by accumulated stress)
4. **Location dependency:** A is a SOCIAL_MOVE that brought agents into proximity needed for B
5. **State dependency:** A produced a delta that B's action scoring depended on (e.g., A changed trust, B's decision to CONFIDE was influenced by trust level)

### Rules

- Every event except the first event in the simulation MUST have at least one causal link.
- Causal links point backward only (to events with earlier tick_id, or same tick_id with earlier order_in_tick).
- Self-links are forbidden (`event.id` must not appear in `event.causal_links`).
- Causal links should be **direct** — link to the proximate cause, not the root cause. If A caused B caused C, then C links to B, not to A. The chain A->B->C is recoverable by traversal.
- Maximum recommended causal links per event: 5. Events rarely have more than 3 direct causes. If an event has >5 links, the simulation should consider whether some are indirect.

### Causal Link Index

The CQRS read layer (Decision #2) builds a bidirectional index:

```python
# Built incrementally as events are appended
forward_links: dict[str, list[str]]   # event_id -> list of events it caused
backward_links: dict[str, list[str]]  # event_id -> list of events that caused it (same as event.causal_links)
```

This supports the precomputed hover neighborhoods (Decision #15): BFS depth-3 in both directions gives the causal cone for each event.

---

## 8. Metrics Dict

The `metrics` field on Event is populated by the metrics pipeline, not the simulation engine. The simulation produces events with default metrics values. The metrics pipeline fills them in as a separate pass.

### Keys and Value Ranges

| Key | Type | Range | Computed By | Description |
|-----|------|-------|-------------|-------------|
| `tension` | float | [0.0, 1.0] | tension-pipeline | Weighted sum of tension sub-metrics, normalized. See `specs/metrics/tension-pipeline.md`. |
| `irony` | float | [0.0, +inf) | irony-and-beliefs | Sum of irony scores for all agents present at the event's location. An irony score of 0.0 means no dramatic irony. Values above 4.0 are exceptional. See `specs/metrics/irony-and-beliefs.md`. |
| `significance` | float | [0.0, 1.0] | counterfactual impact (Decision #9) | JSD between futures-with and futures-without this event, normalized to [0,1]. Expensive to compute — only populated on demand (user click or threshold trigger). Default 0.0 means "not yet computed", not "unimportant". |
| `thematic_shift` | dict[str, float] | each value in [-1.0, 1.0] | tension-pipeline | Per-axis thematic movement. Keys are from THEMATIC_AXES: `loyalty_betrayal`, `freedom_control`, `love_duty`, `innocence_corruption`, `truth_deception`, `order_chaos`. Only axes with nonzero shift are included. Empty dict means no thematic shift. |

### Thematic Axes Reference

From doc3.md:

```python
THEMATIC_AXES = {
    "loyalty_betrayal":     (-1.0, 1.0),   # -1 = betrayal, +1 = loyalty
    "freedom_control":      (-1.0, 1.0),
    "love_duty":            (-1.0, 1.0),
    "innocence_corruption": (-1.0, 1.0),
    "truth_deception":      (-1.0, 1.0),
    "order_chaos":          (-1.0, 1.0),
}
```

A thematic_shift of `{"loyalty_betrayal": -0.3}` means this event moved the narrative 0.3 units toward the "betrayal" end of the loyalty_betrayal axis.

---

## 9. JSON Examples

### Example 1: CHAT event (low-tension small talk)

```json
{
    "id": "evt_0003",
    "sim_time": 2.5,
    "tick_id": 5,
    "order_in_tick": 0,
    "type": "chat",
    "source_agent": "thorne",
    "target_agents": ["elena"],
    "location_id": "dining_table",
    "causal_links": ["evt_0001"],
    "deltas": [
        {
            "kind": "relationship",
            "agent": "elena",
            "agent_b": "thorne",
            "attribute": "affection",
            "op": "add",
            "value": 0.05,
            "reason_code": "PLEASANT_CONVERSATION",
            "reason_display": "Thorne complimented Elena's work at the gallery"
        }
    ],
    "description": "Thorne makes small talk with Elena about her art gallery opening.",
    "dialogue": "\"I heard the opening was a tremendous success. You must be thrilled.\"",
    "beat_type": null,
    "metrics": {
        "tension": 0.05,
        "irony": 0.0,
        "significance": 0.0,
        "thematic_shift": {}
    }
}
```

### Example 2: OBSERVE event (overhearing a secret)

```json
{
    "id": "evt_0027",
    "sim_time": 18.0,
    "tick_id": 36,
    "order_in_tick": 1,
    "type": "observe",
    "source_agent": "lydia",
    "target_agents": [],
    "location_id": "kitchen",
    "causal_links": ["evt_0025", "evt_0026"],
    "deltas": [
        {
            "kind": "belief",
            "agent": "lydia",
            "agent_b": null,
            "attribute": "secret_affair_01",
            "op": "set",
            "value": "suspects",
            "reason_code": "OVERHEARD_WHISPER",
            "reason_display": "Lydia overheard Marcus and Elena whispering intimately in the kitchen"
        },
        {
            "kind": "agent_emotion",
            "agent": "lydia",
            "agent_b": null,
            "attribute": "suspicion",
            "op": "add",
            "value": 0.4,
            "reason_code": "OVERHEARD_WHISPER",
            "reason_display": "Something about their body language felt wrong"
        }
    ],
    "description": "Lydia overhears Marcus and Elena whispering intimately while fetching wine from the kitchen.",
    "dialogue": null,
    "beat_type": "complication",
    "metrics": {
        "tension": 0.35,
        "irony": 2.0,
        "significance": 0.0,
        "thematic_shift": {
            "truth_deception": 0.15
        }
    }
}
```

### Example 3: CONFLICT event (confrontation)

```json
{
    "id": "evt_0089",
    "sim_time": 62.0,
    "tick_id": 124,
    "order_in_tick": 0,
    "type": "conflict",
    "source_agent": "thorne",
    "target_agents": ["marcus"],
    "location_id": "balcony",
    "causal_links": ["evt_0085", "evt_0072"],
    "deltas": [
        {
            "kind": "relationship",
            "agent": "thorne",
            "agent_b": "marcus",
            "attribute": "trust",
            "op": "add",
            "value": -0.4,
            "reason_code": "ACCUSATION_MADE",
            "reason_display": "Thorne accused Marcus of embezzling from the firm"
        },
        {
            "kind": "relationship",
            "agent": "marcus",
            "agent_b": "thorne",
            "attribute": "trust",
            "op": "add",
            "value": -0.3,
            "reason_code": "ACCUSED_UNFAIRLY",
            "reason_display": "Marcus feels betrayed by Thorne's public accusation"
        },
        {
            "kind": "agent_emotion",
            "agent": "thorne",
            "agent_b": null,
            "attribute": "anger",
            "op": "add",
            "value": 0.3,
            "reason_code": "CONFRONTATION_INITIATED",
            "reason_display": "Months of suspicion boiling over"
        },
        {
            "kind": "agent_emotion",
            "agent": "marcus",
            "agent_b": null,
            "attribute": "fear",
            "op": "add",
            "value": 0.5,
            "reason_code": "SECRET_THREATENED",
            "reason_display": "Marcus realizes Thorne may know about the missing money"
        },
        {
            "kind": "pacing",
            "agent": "thorne",
            "agent_b": null,
            "attribute": "dramatic_budget",
            "op": "add",
            "value": -0.4,
            "reason_code": "DRAMATIC_ACTION_COST",
            "reason_display": "Confrontation drained Thorne's dramatic budget"
        },
        {
            "kind": "pacing",
            "agent": "marcus",
            "agent_b": null,
            "attribute": "stress",
            "op": "add",
            "value": 0.3,
            "reason_code": "CONFLICT_EXPOSURE",
            "reason_display": "Being confronted raised Marcus's stress"
        }
    ],
    "description": "Thorne corners Marcus on the balcony and accuses him of embezzling from their shared business.",
    "dialogue": "\"I've seen the ledger, Marcus. Where did the money go?\"",
    "beat_type": "escalation",
    "metrics": {
        "tension": 0.72,
        "irony": 3.5,
        "significance": 0.0,
        "thematic_shift": {
            "loyalty_betrayal": -0.25,
            "truth_deception": 0.2
        }
    }
}
```

### Example 4: CATASTROPHE event (involuntary breakdown)

```json
{
    "id": "evt_0142",
    "sim_time": 98.0,
    "tick_id": 196,
    "order_in_tick": 0,
    "type": "catastrophe",
    "source_agent": "elena",
    "target_agents": ["thorne", "marcus", "lydia", "diana", "victor"],
    "location_id": "dining_table",
    "causal_links": ["evt_0139", "evt_0140", "evt_0141"],
    "deltas": [
        {
            "kind": "secret_state",
            "agent": "elena",
            "agent_b": null,
            "attribute": "secret_affair_01",
            "op": "set",
            "value": "publicly_known",
            "reason_code": "CATASTROPHE_REVEAL",
            "reason_display": "Elena blurted out the truth about her affair with Marcus"
        },
        {
            "kind": "belief",
            "agent": "thorne",
            "agent_b": null,
            "attribute": "secret_affair_01",
            "op": "set",
            "value": "believes_true",
            "reason_code": "DIRECT_CONFESSION",
            "reason_display": "Elena confessed in front of everyone"
        },
        {
            "kind": "belief",
            "agent": "diana",
            "agent_b": null,
            "attribute": "secret_affair_01",
            "op": "set",
            "value": "believes_true",
            "reason_code": "DIRECT_CONFESSION",
            "reason_display": "Elena confessed in front of everyone"
        },
        {
            "kind": "belief",
            "agent": "victor",
            "agent_b": null,
            "attribute": "secret_affair_01",
            "op": "set",
            "value": "believes_true",
            "reason_code": "DIRECT_CONFESSION",
            "reason_display": "Elena confessed in front of everyone"
        },
        {
            "kind": "relationship",
            "agent": "thorne",
            "agent_b": "elena",
            "attribute": "trust",
            "op": "set",
            "value": -0.8,
            "reason_code": "BETRAYAL_REVEALED",
            "reason_display": "Thorne's wife was having an affair with his business partner"
        },
        {
            "kind": "relationship",
            "agent": "thorne",
            "agent_b": "marcus",
            "attribute": "trust",
            "op": "set",
            "value": -1.0,
            "reason_code": "DOUBLE_BETRAYAL",
            "reason_display": "Marcus was both embezzling and having an affair with Thorne's wife"
        },
        {
            "kind": "agent_emotion",
            "agent": "elena",
            "agent_b": null,
            "attribute": "shame",
            "op": "add",
            "value": 0.8,
            "reason_code": "PUBLIC_SHAME",
            "reason_display": "The weight of the secret finally broke her"
        },
        {
            "kind": "pacing",
            "agent": "elena",
            "agent_b": null,
            "attribute": "dramatic_budget",
            "op": "set",
            "value": 0.0,
            "reason_code": "CATASTROPHE_DRAIN",
            "reason_display": "Catastrophe event fully depleted dramatic budget"
        },
        {
            "kind": "pacing",
            "agent": "elena",
            "agent_b": null,
            "attribute": "composure",
            "op": "set",
            "value": 0.0,
            "reason_code": "COMPOSURE_COLLAPSE",
            "reason_display": "Complete breakdown of social facade"
        }
    ],
    "description": "Elena breaks down at the dinner table and confesses her affair with Marcus in front of everyone.",
    "dialogue": "\"I can't do this anymore! Thorne, I'm sorry — Marcus and I... we've been...\"",
    "beat_type": "turning_point",
    "metrics": {
        "tension": 0.95,
        "irony": 0.5,
        "significance": 0.0,
        "thematic_shift": {
            "loyalty_betrayal": -0.6,
            "truth_deception": 0.5,
            "innocence_corruption": -0.3
        }
    }
}
```

### Example 5: INTERNAL event (invisible thought)

```json
{
    "id": "evt_0045",
    "sim_time": 30.0,
    "tick_id": 60,
    "order_in_tick": 2,
    "type": "internal",
    "source_agent": "victor",
    "target_agents": [],
    "location_id": "dining_table",
    "causal_links": ["evt_0043"],
    "deltas": [
        {
            "kind": "agent_emotion",
            "agent": "victor",
            "agent_b": null,
            "attribute": "suspicion",
            "op": "add",
            "value": 0.2,
            "reason_code": "PATTERN_NOTICED",
            "reason_display": "Victor noticed Marcus checking his phone nervously"
        },
        {
            "kind": "commitment",
            "agent": "victor",
            "agent_b": null,
            "attribute": "",
            "op": "set",
            "value": "investigate_marcus_phone",
            "reason_code": "CURIOSITY_TRIGGERED",
            "reason_display": "Victor decided to keep an eye on Marcus"
        }
    ],
    "description": "Victor notices Marcus's nervous phone-checking and decides to watch him more closely.",
    "dialogue": null,
    "beat_type": null,
    "metrics": {
        "tension": 0.15,
        "irony": 1.0,
        "significance": 0.0,
        "thematic_shift": {
            "truth_deception": 0.1
        }
    }
}
```

### Example 6: LIE event (deliberate misinformation)

```json
{
    "id": "evt_0058",
    "sim_time": 40.0,
    "tick_id": 80,
    "order_in_tick": 0,
    "type": "lie",
    "source_agent": "marcus",
    "target_agents": ["thorne"],
    "location_id": "foyer",
    "causal_links": ["evt_0055"],
    "deltas": [
        {
            "kind": "belief",
            "agent": "thorne",
            "agent_b": null,
            "attribute": "secret_embezzle_01",
            "op": "set",
            "value": "believes_false",
            "reason_code": "DELIBERATE_MISDIRECTION",
            "reason_display": "Marcus convinced Thorne the accounting discrepancy was a clerical error"
        },
        {
            "kind": "relationship",
            "agent": "thorne",
            "agent_b": "marcus",
            "attribute": "trust",
            "op": "add",
            "value": 0.1,
            "reason_code": "BELIEVED_EXPLANATION",
            "reason_display": "Thorne accepted Marcus's explanation"
        },
        {
            "kind": "pacing",
            "agent": "marcus",
            "agent_b": null,
            "attribute": "stress",
            "op": "add",
            "value": 0.15,
            "reason_code": "LYING_STRESS",
            "reason_display": "The act of lying increased Marcus's internal stress"
        },
        {
            "kind": "pacing",
            "agent": "marcus",
            "agent_b": null,
            "attribute": "commitment",
            "op": "add",
            "value": 0.2,
            "reason_code": "DEEPER_INTO_DECEPTION",
            "reason_display": "Each lie commits Marcus further to maintaining the facade"
        }
    ],
    "description": "Marcus tells Thorne the financial discrepancy was a simple accounting error, deflecting suspicion.",
    "dialogue": "\"It was just a bookkeeping mix-up. I already spoke to the accountant — she's correcting it Monday.\"",
    "beat_type": "escalation",
    "metrics": {
        "tension": 0.45,
        "irony": 4.0,
        "significance": 0.0,
        "thematic_shift": {
            "truth_deception": -0.3,
            "loyalty_betrayal": -0.15
        }
    }
}
```

### Example 7: SOCIAL_MOVE event (location change)

```json
{
    "id": "evt_0070",
    "sim_time": 48.5,
    "tick_id": 97,
    "order_in_tick": 0,
    "type": "social_move",
    "source_agent": "diana",
    "target_agents": [],
    "location_id": "dining_table",
    "causal_links": ["evt_0068"],
    "deltas": [
        {
            "kind": "agent_location",
            "agent": "diana",
            "agent_b": null,
            "attribute": "",
            "op": "set",
            "value": "balcony",
            "reason_code": "SEEKING_PRIVACY",
            "reason_display": "Diana excused herself to get some air"
        },
        {
            "kind": "pacing",
            "agent": "diana",
            "agent_b": null,
            "attribute": "dramatic_budget",
            "op": "add",
            "value": -0.05,
            "reason_code": "SOCIAL_MOVE_COST",
            "reason_display": "Minor cost of leaving the table"
        }
    ],
    "description": "Diana excuses herself from the table and steps out onto the balcony.",
    "dialogue": "\"Excuse me, I need a moment of fresh air.\"",
    "beat_type": null,
    "metrics": {
        "tension": 0.2,
        "irony": 0.0,
        "significance": 0.0,
        "thematic_shift": {}
    }
}
```

**Note on SOCIAL_MOVE:** The `location_id` field on the Event records where the agent WAS (origin). The AGENT_LOCATION delta records where they're GOING (destination). This lets the event be indexed under both locations.

### Example 8: CONFIDE event (private trust)

```json
{
    "id": "evt_0034",
    "sim_time": 22.0,
    "tick_id": 44,
    "order_in_tick": 0,
    "type": "confide",
    "source_agent": "elena",
    "target_agents": ["diana"],
    "location_id": "bathroom",
    "causal_links": ["evt_0030"],
    "deltas": [
        {
            "kind": "belief",
            "agent": "diana",
            "agent_b": null,
            "attribute": "secret_affair_01",
            "op": "set",
            "value": "believes_true",
            "reason_code": "CONFIDED_SECRET",
            "reason_display": "Elena told Diana about her affair with Marcus"
        },
        {
            "kind": "relationship",
            "agent": "diana",
            "agent_b": "elena",
            "attribute": "trust",
            "op": "add",
            "value": 0.15,
            "reason_code": "TRUST_THROUGH_VULNERABILITY",
            "reason_display": "Elena's vulnerability deepened Diana's sense of trust"
        },
        {
            "kind": "relationship",
            "agent": "elena",
            "agent_b": "diana",
            "attribute": "trust",
            "op": "add",
            "value": 0.1,
            "reason_code": "CONFIDING_BOND",
            "reason_display": "The act of confiding creates reciprocal trust"
        },
        {
            "kind": "pacing",
            "agent": "elena",
            "agent_b": null,
            "attribute": "stress",
            "op": "add",
            "value": -0.1,
            "reason_code": "STRESS_RELIEF_CONFIDING",
            "reason_display": "Sharing the burden temporarily reduced Elena's stress"
        }
    ],
    "description": "Elena confides in Diana about her affair with Marcus while they're alone in the bathroom.",
    "dialogue": "\"Diana, I need to tell someone. Marcus and I... it's been going on for months.\"",
    "beat_type": "complication",
    "metrics": {
        "tension": 0.4,
        "irony": 1.5,
        "significance": 0.0,
        "thematic_shift": {
            "truth_deception": 0.2,
            "loyalty_betrayal": 0.1
        }
    }
}
```

---

## 10. Validation Rules

These rules must be enforced by the simulation engine when producing events and by any data loader when ingesting event logs.

### Event-Level Validation

1. **ID uniqueness:** No two events may share an `id`.
2. **Temporal monotonicity:** Events must be appendable in `(tick_id, order_in_tick)` order. Appending an event with a lower `(tick_id, order_in_tick)` than the last appended event is an error.
3. **Causal link validity:** Every ID in `causal_links` must reference an already-appended event.
4. **Location consistency:** `location_id` must reference a valid location in the world definition.
5. **Agent consistency:** `source_agent` and all entries in `target_agents` must reference valid agents.
6. **Type-specific rules:**
    - INTERNAL: `target_agents` must be empty.
    - CATASTROPHE: simulation must verify catastrophe threshold was met.
    - SOCIAL_MOVE: must contain exactly one AGENT_LOCATION delta.
    - LIE: must contain at least one BELIEF delta that plants a false belief.

### Delta-Level Validation

1. **Kind-attribute match:** The `attribute` must be valid for the `kind`. See the Field Requirements table in Section 3.
2. **Op-value match:** If `op` is `ADD`, `value` must be numeric. If `op` is `SET`, `value` can be any type.
3. **Agent existence:** `agent` (and `agent_b` if present) must reference valid agents.
4. **Range constraints:**
    - Emotional state values: clamped to [0.0, 1.0] after application
    - Relationship dimensions (trust, affection): clamped to [-1.0, 1.0] after application
    - Relationship obligation: clamped to [0.0, 1.0] after application
    - Pacing dramatic_budget: clamped to [0.0, 1.0] after application
    - Pacing stress: clamped to [0.0, 1.0] after application
    - Pacing composure: clamped to [0.0, 1.0] after application
    - Pacing commitment: clamped to [0.0, 1.0] after application
    - Pacing recovery_timer: non-negative integer
    - Alcohol level: clamped to [0.0, 1.0] after application

### Event Log Integrity

The full event log must satisfy:
1. **First event has no causal links** (or links to a synthetic "simulation_start" event).
2. **Every agent that appears in any event exists in the initial world state.**
3. **Snapshot consistency:** Replaying all events from tick 0 (or from the nearest snapshot) must produce the same world state as the cached snapshot at any given tick.

---

## 11. NOT In Scope

The following are explicitly outside this spec and handled elsewhere:

- **Agent decision logic** — how agents choose actions. See `specs/simulation/decision-engine.md`.
- **Tick loop mechanics** — how ticks advance and actions resolve. See `specs/simulation/tick-loop.md`.
- **Scene segmentation** — how events are grouped into scenes. See `specs/schema/scenes.md` and `specs/metrics/scene-segmentation.md`.
- **Tension computation formulas** — how metrics.tension is calculated. See `specs/metrics/tension-pipeline.md`.
- **Irony computation formulas** — how metrics.irony is calculated. See `specs/metrics/irony-and-beliefs.md`.
- **Counterfactual impact computation** — how metrics.significance is calculated. See Decision #9 in doc3.md.
- **Visualization rendering** — how events are displayed. See `specs/visualization/`.
- **Story extraction and arc validation** — how beat_type is assigned and arcs are validated. See `specs/metrics/story-extraction.md`.
- **Promise system** — post-hoc search targets. See Decision #7 in doc3.md.

---

## Dependencies

This spec is upstream of:
- `specs/schema/agents.md` (uses StateDelta to modify agent state)
- `specs/schema/scenes.md` (groups Events into Scenes)
- `specs/simulation/tick-loop.md` (produces Events)
- `specs/simulation/decision-engine.md` (produces Events)
- `specs/visualization/renderer-architecture.md` (renders Events)
- `specs/visualization/thread-layout.md` (positions Events)
- `specs/visualization/interaction-model.md` (responds to Event hover/click)
- `specs/metrics/tension-pipeline.md` (computes Event.metrics)
- `specs/metrics/irony-and-beliefs.md` (computes Event.metrics.irony)
- `specs/metrics/scene-segmentation.md` (groups Events)
- `specs/metrics/story-extraction.md` (classifies Event.beat_type)
- `specs/integration/data-flow.md` (defines how Events flow between subsystems)

This spec depends on:
- doc3.md (architectural decisions — all resolved)
- `specs/schema/agents.md` (for valid agent_id values and state fields — circular dependency resolved by defining field names here, ranges in agents.md)
- `specs/schema/world.md` (for valid location_id values)
