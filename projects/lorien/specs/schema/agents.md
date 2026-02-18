# Agent State Schema Specification

> **Status:** CANONICAL — defines the complete agent state model.
> **Implements:** Decisions #5 (pacing physics), #6 (vectorized feature scoring), #8 (finite proposition catalog / belief matrix), #10 (two-parameter catastrophe)
> **Consumers:** simulation (decision-engine, tick-loop, pacing-physics, dinner-party-config), visualization (thread-layout), metrics (tension-pipeline, irony-and-beliefs)
> **Depends on:** `specs/schema/events.md` (StateDelta applies to these structures)

---

## Table of Contents

1. [GoalVector](#1-goalvector)
2. [CharacterFlaw](#2-characterflaw)
3. [RelationshipState](#3-relationshipstate)
4. [PacingState](#4-pacingstate)
5. [BeliefState Enum & Belief Matrix](#5-beliefstate-enum--belief-matrix)
6. [Emotional State](#6-emotional-state)
7. [AgentState](#7-agentstate)
8. [Update Rules & Formulas](#8-update-rules--formulas)
9. [Dinner Party Characters](#9-dinner-party-characters---worked-examples)
10. [Validation Rules](#10-validation-rules)
11. [NOT In Scope](#11-not-in-scope)

---

## 1. GoalVector

An agent's desires in feature space (Decision #6). Used by the decision engine to score candidate actions via weighted dot product. The conflict between two agents is measured by cosine distance between their goal vectors.

### Python

```python
from dataclasses import dataclass, field

@dataclass
class GoalVector:
    """An agent's desires in feature space. Each dimension is [0.0, 1.0]."""
    safety: float = 0.5            # desire to avoid danger, exposure, vulnerability
    status: float = 0.5            # desire for social position, respect, dominance
    closeness: dict[str, float] = field(default_factory=dict)
                                    # {agent_id: desire_for_closeness} each in [-1.0, 1.0]
                                    # positive = wants closeness, negative = wants distance
    secrecy: float = 0.5           # desire to keep own secrets hidden
    truth_seeking: float = 0.5     # desire for honesty, uncovering truth, transparency
    autonomy: float = 0.5          # desire for independence, not being controlled
    loyalty: float = 0.5           # desire to honor commitments, stand by allies
```

### TypeScript

```typescript
interface GoalVector {
    safety: number;                 // [0.0, 1.0]
    status: number;                 // [0.0, 1.0]
    closeness: Record<string, number>;  // agent_id -> [-1.0, 1.0]
    secrecy: number;                // [0.0, 1.0]
    truth_seeking: number;          // [0.0, 1.0]
    autonomy: number;               // [0.0, 1.0]
    loyalty: number;                // [0.0, 1.0]
}
```

### Scoring

The decision engine scores an action by:

```python
def score_action(agent: AgentState, action: Action, world: WorldState) -> float:
    """
    Compute how well an action aligns with the agent's goals.
    Higher score = agent more likely to choose this action.
    """
    predicted_deltas = estimate_action_effects(action, world)
    goal_alignment = dot_product(predicted_deltas, agent.goals)
    flaw_bias = compute_flaw_bias(agent.flaws, action, world)
    return goal_alignment + flaw_bias
```

The `closeness` dict is handled specially: for actions involving another agent, the closeness desire toward that specific agent is used as a weight.

### Conflict Score

```python
import numpy as np

def conflict_score(a: GoalVector, b: GoalVector) -> float:
    """
    Cosine distance between goal vectors.
    0.0 = perfectly aligned, 1.0 = orthogonal, 2.0 = perfectly opposed.
    """
    vec_a = np.array([a.safety, a.status, a.secrecy, a.truth_seeking, a.autonomy, a.loyalty])
    vec_b = np.array([b.safety, b.status, b.secrecy, b.truth_seeking, b.autonomy, b.loyalty])
    cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)
    return 1.0 - cosine_sim  # cosine distance
```

Note: `closeness` is excluded from the global conflict score since it's agent-pair-specific. It's used in pairwise interaction scoring instead.

### Goal Vector Dynamics

Goal vectors are **mostly static** within a single simulation run. They represent deep character traits, not moment-to-moment desires. However, extreme events CAN shift them:

- A CATASTROPHE event may shift `safety` upward (the character becomes more cautious).
- A betrayal may shift `loyalty` downward.
- These shifts are rare (at most 1-2 per simulation run) and are explicitly modeled as StateDelta with `kind: AGENT_RESOURCE` and `attribute: "goal_safety"` etc.

For the MVP dinner party, goal vectors are fixed at initialization.

---

## 2. CharacterFlaw

Irrational biases that make agents story-shaped rather than optimal (Decision #6, doc2.md Force 3). Flaws distort the action scoring function, causing agents to make suboptimal but dramatically interesting choices.

### Python

```python
from enum import Enum

class FlawType(Enum):
    PRIDE        = "pride"          # overestimates own capability, defends ego
    LOYALTY      = "loyalty"        # continues alliances past self-interest
    TRAUMA       = "trauma"         # avoids situations resembling past pain
    AMBITION     = "ambition"       # overweights high-risk high-reward paths
    JEALOUSY     = "jealousy"       # perceives threats to relationships that don't exist
    COWARDICE    = "cowardice"      # avoids confrontation even when necessary
    VANITY       = "vanity"         # prioritizes appearance/perception over substance
    GUILT        = "guilt"          # punishes self, makes concessions to atone
    OBSESSION    = "obsession"      # fixates on one goal to exclusion of others
    DENIAL       = "denial"         # refuses to update beliefs when evidence contradicts them

@dataclass
class CharacterFlaw:
    """A character flaw that biases the agent's decision-making."""
    flaw_type: FlawType
    strength: float                  # [0.0, 1.0] how much this distorts decisions
    trigger: str                     # what activates it (machine-readable tag)
                                     # e.g. "status_threat", "betrayal_detected", "secret_exposure"
    effect: str                      # what it does (machine-readable tag)
                                     # e.g. "overweight_status", "avoid_confrontation", "deny_evidence"
    description: str                 # human-readable description for tooltips
```

### TypeScript

```typescript
enum FlawType {
    PRIDE        = "pride",
    LOYALTY      = "loyalty",
    TRAUMA       = "trauma",
    AMBITION     = "ambition",
    JEALOUSY     = "jealousy",
    COWARDICE    = "cowardice",
    VANITY       = "vanity",
    GUILT        = "guilt",
    OBSESSION    = "obsession",
    DENIAL       = "denial",
}

interface CharacterFlaw {
    flaw_type: FlawType;
    strength: number;               // [0.0, 1.0]
    trigger: string;
    effect: string;
    description: string;
}
```

### Bias Mechanics

The decision engine applies flaw bias as follows:

```python
def compute_flaw_bias(flaws: list[CharacterFlaw], action: Action, world: WorldState) -> float:
    """
    Compute total flaw-induced bias on an action score.
    Positive = flaw pushes toward this action. Negative = flaw pushes away.
    """
    total_bias = 0.0
    for flaw in flaws:
        if trigger_matches(flaw.trigger, action, world):
            # Flaw is active — apply its effect
            if flaw.effect == "overweight_status":
                total_bias += flaw.strength * action.status_impact
            elif flaw.effect == "avoid_confrontation":
                if action.type in (EventType.CONFLICT, EventType.REVEAL):
                    total_bias -= flaw.strength * 0.5
            elif flaw.effect == "deny_evidence":
                if action.type == EventType.OBSERVE:
                    total_bias -= flaw.strength * 0.3  # less likely to accept new info
            # ... etc for each effect type
    return total_bias
```

### Trigger Tags

Standard trigger tags (extensible):

| Trigger | Fires When |
|---------|-----------|
| `status_threat` | Another agent's action would lower this agent's status |
| `betrayal_detected` | Agent discovers a trusted ally acted against them |
| `secret_exposure` | One of the agent's secrets is at risk of being revealed |
| `rejection` | Agent's social bid (closeness, alliance) is rebuffed |
| `authority_challenge` | Agent's position or decisions are questioned |
| `intimacy_offered` | Another agent offers closeness or vulnerability |
| `conflict_nearby` | Conflict is happening in the same location |
| `loss_imminent` | Agent is about to lose something valued |

### Effect Tags

Standard effect tags (extensible):

| Effect | Behavior |
|--------|----------|
| `overweight_status` | Actions that increase status score higher |
| `avoid_confrontation` | CONFLICT and REVEAL actions score lower |
| `deny_evidence` | OBSERVE events less likely to update beliefs |
| `escalate_conflict` | CONFLICT actions score higher |
| `seek_validation` | CHAT actions with high-status agents score higher |
| `self_sacrifice` | Actions that hurt self but help allies score higher |
| `fixate_on_target` | Actions involving the obsession target score higher |
| `overcommit` | High-risk actions score disproportionately higher |

---

## 3. RelationshipState

Tracks the state of the relationship between two agents. Stored as a dict on each agent: `relationships[other_agent_id] -> RelationshipState`.

Relationships are **asymmetric**: Agent A's trust toward B may differ from B's trust toward A.

### Python

```python
@dataclass
class RelationshipState:
    """State of one agent's relationship toward another agent."""
    trust: float = 0.0              # [-1.0, 1.0] — -1 = active distrust, 0 = neutral, 1 = full trust
    affection: float = 0.0          # [-1.0, 1.0] — -1 = hatred, 0 = indifference, 1 = love
    obligation: float = 0.0         # [0.0, 1.0] — 0 = no obligation, 1 = deep debt/duty
```

### TypeScript

```typescript
interface RelationshipState {
    trust: number;                  // [-1.0, 1.0]
    affection: number;              // [-1.0, 1.0]
    obligation: number;             // [0.0, 1.0]
}
```

### Relationship Dynamics

**Hysteresis (Decision #5 rule 6):** Repairing trust costs 3x more than breaking it.

```python
def apply_relationship_delta(current: RelationshipState, delta: StateDelta) -> RelationshipState:
    """Apply a relationship delta with hysteresis."""
    attr = delta.attribute  # "trust", "affection", or "obligation"
    current_val = getattr(current, attr)

    if delta.op == DeltaOp.SET:
        new_val = delta.value
    else:  # ADD
        raw_change = delta.value
        # Hysteresis: positive changes to trust are 3x harder than negative
        if attr == "trust" and raw_change > 0:
            raw_change *= (1.0 / 3.0)
        new_val = current_val + raw_change

    # Clamp to valid range
    if attr == "obligation":
        new_val = max(0.0, min(1.0, new_val))
    else:
        new_val = max(-1.0, min(1.0, new_val))

    setattr(current, attr, new_val)
    return current
```

### Social Masking (Decision #5 rule 7)

Agents suppress conflict in public settings. The decision engine applies a social masking modifier:

```python
def social_masking_modifier(agent: AgentState, action: Action, location: Location) -> float:
    """
    Reduce score of dramatic actions in public locations.
    Returns a negative modifier for conflict/reveal actions in low-privacy locations.
    """
    if action.type in (EventType.CONFLICT, EventType.REVEAL, EventType.CATASTROPHE):
        # Higher privacy = less suppression
        suppression = (1.0 - location.privacy) * agent.pacing.composure * 0.5
        return -suppression
    return 0.0
```

This means confrontations are more likely on the balcony (privacy 0.7) than at the dining table (privacy 0.1).

---

## 4. PacingState

Per-agent drama pacing (Decision #5). Prevents constant chaos by imposing resource constraints on dramatic actions.

> **Authority note:** The PacingState dataclass is defined here. For authoritative update rules, constants, thresholds, and catastrophe mechanics, see `specs/simulation/pacing-physics.md`. The formulas below are illustrative.

### Python

```python
@dataclass
class PacingState:
    """Per-agent pacing physics. The drama budget system."""
    dramatic_budget: float = 1.0     # [0.0, 1.0] — fuel for dramatic actions, replenishes in quiet beats
    stress: float = 0.0              # [0.0, 1.0] — accumulates with conflict exposure (catastrophe param 1)
    composure: float = 1.0           # [0.0, 1.0] — ability to mask true state, social facade strength
    commitment: float = 0.0          # [0.0, 1.0] — investment in current path (catastrophe param 2)
    recovery_timer: int = 0          # non-negative int — ticks until next dramatic action allowed (0 = ready)
    suppression_count: int = 0       # [0, ...] — consecutive ticks of suppressed high stress
                                     # Increments when stress > 0.5 and composure > 0.5 (masking).
                                     # Resets to 0 when stress drops below 0.5 or composure breaks.
                                     # Feeds into catastrophe potential formula.
```

### TypeScript

```typescript
interface PacingState {
    dramatic_budget: number;         // [0.0, 1.0]
    stress: number;                  // [0.0, 1.0]
    composure: number;               // [0.0, 1.0]
    commitment: number;              // [0.0, 1.0]
    recovery_timer: number;          // non-negative integer
    suppression_count: number;       // non-negative integer — consecutive ticks of suppressed high stress
}
```

### Update Rules

These formulas are applied at the end of each tick, after all events for the tick have been processed.

#### Dramatic Budget

```python
def update_dramatic_budget(pacing: PacingState, events_this_tick: list[Event], agent_id: str) -> None:
    """
    Dramatic budget is spent by dramatic actions and replenished by quiet beats.
    """
    # Cost: applied by event deltas (PACING kind, attribute "dramatic_budget", op ADD, negative value)
    # Already applied during event processing.

    # Replenishment: if no dramatic events this tick, budget recovers
    agent_events = [e for e in events_this_tick if e.source_agent == agent_id]
    dramatic_types = {EventType.CONFLICT, EventType.REVEAL, EventType.CATASTROPHE, EventType.CONFIDE}

    had_dramatic_event = any(e.type in dramatic_types for e in agent_events)

    if not had_dramatic_event:
        # Replenish at rate of 0.05 per tick during quiet beats
        pacing.dramatic_budget = min(1.0, pacing.dramatic_budget + 0.05)

    pacing.dramatic_budget = max(0.0, min(1.0, pacing.dramatic_budget))
```

#### Stress

```python
def update_stress(pacing: PacingState, events_this_tick: list[Event], agent_id: str, location_id: str) -> None:
    """
    Stress accumulates from conflict exposure (even witnessing) and decays slowly.
    """
    # Accumulation: applied by event deltas during processing.

    # Passive decay: stress decreases by 0.02 per tick if agent is in a private location
    # and 0.01 per tick if in a public location
    from_location = get_location(location_id)
    if from_location.privacy > 0.5:
        pacing.stress = max(0.0, pacing.stress - 0.02)
    else:
        pacing.stress = max(0.0, pacing.stress - 0.01)

    pacing.stress = max(0.0, min(1.0, pacing.stress))
```

#### Composure

```python
def update_composure(pacing: PacingState, agent: AgentState) -> None:
    """
    Composure degrades with stress and alcohol. Recovers slowly.
    """
    # Composure loss from stress: high stress erodes composure
    stress_erosion = pacing.stress * 0.03  # per tick

    # Composure loss from alcohol
    alcohol_erosion = agent.alcohol_level * 0.02  # per tick

    # Natural recovery (small)
    natural_recovery = 0.01  # per tick

    composure_change = natural_recovery - stress_erosion - alcohol_erosion
    pacing.composure = max(0.0, min(1.0, pacing.composure + composure_change))
```

#### Commitment

```python
def update_commitment(pacing: PacingState) -> None:
    """
    Commitment doesn't decay naturally. It only increases through COMMITMENT deltas
    (irreversible choices) and can only decrease through a CATASTROPHE event
    (which resets it to 0).
    """
    # No passive update. Commitment changes only through event deltas.
    pass
```

#### Recovery Timer

```python
def update_recovery_timer(pacing: PacingState) -> None:
    """
    Recovery timer counts down each tick. At 0, agent can take dramatic actions again.
    """
    if pacing.recovery_timer > 0:
        pacing.recovery_timer -= 1
```

### Catastrophe Threshold (Decision #10)

> **Authoritative spec:** `specs/simulation/pacing-physics.md` owns the exact catastrophe formula and aftermath. The summary below is for quick reference.

The cusp catastrophe fires when both control parameters are high and composure is low. Suppression count adds directly to catastrophe potential, modeling the "pressure cooker" dynamic:

```python
def check_catastrophe(pacing: PacingState, agent: AgentState) -> bool:
    """
    Two-parameter cusp catastrophe check.
    Returns True if the agent should experience an involuntary catastrophe event.
    See specs/simulation/pacing-physics.md for authoritative constants.
    """
    catastrophe_potential = (pacing.stress * (pacing.commitment ** 2)
                             + pacing.suppression_count * 0.03)

    CATASTROPHE_THRESHOLD = 0.35
    COMPOSURE_GATE = 0.30

    return (catastrophe_potential >= CATASTROPHE_THRESHOLD
            and pacing.composure < COMPOSURE_GATE
            and pacing.recovery_timer == 0)
```

When a catastrophe fires:
- `dramatic_budget` is set to 0.0
- `stress` is halved (not zeroed — the relief is partial)
- `composure` is set to 0.30 (not 0.0 — agent retains minimal social function)
- `commitment` increases (the catastrophe is itself an irreversible act that deepens commitment)
- `suppression_count` is reset to 0
- `recovery_timer` is set to 10 ticks (agent is in recovery, can't take dramatic actions)
- The catastrophe type depends on the agent's dominant flaw (see Decision #10 in doc3.md)

See `specs/simulation/pacing-physics.md` for the full aftermath rules, recovery dynamics, and edge cases.

---

## 5. BeliefState Enum & Belief Matrix

The finite proposition catalog (Decision #8). Each agent has a belief about each secret.

### BeliefState Enum

```python
class BeliefState(Enum):
    UNKNOWN        = "unknown"          # hasn't encountered this proposition
    SUSPECTS       = "suspects"         # has partial evidence, not sure
    BELIEVES_TRUE  = "believes_true"    # confident it's true
    BELIEVES_FALSE = "believes_false"   # confident it's false (may be wrong!)
```

```typescript
enum BeliefState {
    UNKNOWN        = "unknown",
    SUSPECTS       = "suspects",
    BELIEVES_TRUE  = "believes_true",
    BELIEVES_FALSE = "believes_false",
}
```

### Belief Matrix

Stored per agent as `beliefs: dict[str, BeliefState]` — mapping `secret_id -> BeliefState`.

At the world level, the belief matrix is the cross-product of all agents and all secrets:

```python
# Conceptual: belief_matrix[agent_id][secret_id] -> BeliefState
# In practice, stored on each AgentState.beliefs dict

def get_belief_matrix(agents: dict[str, AgentState]) -> dict[str, dict[str, BeliefState]]:
    """Materialize the full belief matrix from agent states."""
    return {agent_id: agent.beliefs for agent_id, agent in agents.items()}
```

### Belief Transition Rules

Beliefs transition based on events. Not all transitions are valid:

```
UNKNOWN → SUSPECTS       (partial evidence: overhearing, noticing suspicious behavior)
UNKNOWN → BELIEVES_TRUE  (direct evidence: confession, witnessing, shown proof)
UNKNOWN → BELIEVES_FALSE (told a lie, given false evidence)

SUSPECTS → BELIEVES_TRUE (confirming evidence received)
SUSPECTS → BELIEVES_FALSE (compelling counter-evidence or denial)
SUSPECTS → UNKNOWN       (NOT ALLOWED — you can't un-suspect)

BELIEVES_TRUE → BELIEVES_FALSE   (RARE — only with very compelling counter-evidence + DENIAL flaw NOT active)
BELIEVES_FALSE → BELIEVES_TRUE   (RARE — requires direct, undeniable proof)
BELIEVES_FALSE → SUSPECTS        (some counter-evidence to their false belief)
BELIEVES_TRUE → SUSPECTS         (NOT ALLOWED — you can't become less sure of something you believe)

Any state → BELIEVES_TRUE        (if publicly revealed — CATASTROPHE or public REVEAL sets all present to BELIEVES_TRUE)
```

### Irony Computation (from Decision #8)

```python
def irony_score(agent_id: str, beliefs: dict[str, BeliefState], secrets: dict[str, "Secret"]) -> float:
    """
    Count of false beliefs + critical unknowns for this agent.
    Higher = more dramatic irony (the audience knows something the character doesn't).
    """
    score = 0.0
    for secret_id, secret in secrets.items():
        belief = beliefs.get(secret_id, BeliefState.UNKNOWN)
        if secret.truth_value and belief == BeliefState.BELIEVES_FALSE:
            score += 2.0   # actively wrong about something true = highest irony
        elif not secret.truth_value and belief == BeliefState.BELIEVES_TRUE:
            score += 2.0   # believes something false is true = also high irony
        elif secret.truth_value and belief == BeliefState.UNKNOWN:
            # Doesn't know something true — only ironic if it's relevant to them
            if secret.about == agent_id or secret.holder == agent_id:
                score += 1.5  # the secret is about them and they don't know!
            else:
                score += 0.5  # general ignorance
        elif belief == BeliefState.SUSPECTS:
            score += 0.25  # mild irony — they suspect but aren't sure
    return score
```

---

## 6. Emotional State

A dict of named emotions, each a float in [0.0, 1.0]. These are modified by AGENT_EMOTION deltas.

### Standard Emotions

```python
STANDARD_EMOTIONS = {
    "anger":      0.0,    # [0.0, 1.0] — frustration, rage
    "fear":       0.0,    # [0.0, 1.0] — anxiety, dread
    "hope":       0.0,    # [0.0, 1.0] — optimism, anticipation of good outcome
    "shame":      0.0,    # [0.0, 1.0] — embarrassment, guilt-adjacent
    "affection":  0.0,    # [0.0, 1.0] — warmth, love (general, not toward a specific agent)
    "suspicion":  0.0,    # [0.0, 1.0] — wariness, distrust (general)
}
```

```typescript
interface EmotionalState {
    anger: number;        // [0.0, 1.0]
    fear: number;         // [0.0, 1.0]
    hope: number;         // [0.0, 1.0]
    shame: number;        // [0.0, 1.0]
    affection: number;    // [0.0, 1.0]
    suspicion: number;    // [0.0, 1.0]
}
```

### Emotional Decay

Emotions decay toward baseline (0.0) each tick unless reinforced:

```python
EMOTION_DECAY_RATE = 0.02  # per tick

def decay_emotions(emotional_state: dict[str, float]) -> None:
    """Emotions decay toward 0.0 each tick."""
    for emotion in emotional_state:
        if emotional_state[emotion] > 0.0:
            emotional_state[emotion] = max(0.0, emotional_state[emotion] - EMOTION_DECAY_RATE)
```

---

## 7. AgentState

Complete state of one agent at a point in time.

### Python

```python
@dataclass
class AgentState:
    """Complete state of one agent at a point in time."""
    # === Identity ===
    id: str                                       # Unique agent ID. E.g. "thorne"
    name: str                                     # Display name. E.g. "James Thorne"

    # === Position ===
    location: str                                 # Current location_id

    # === Psychology ===
    goals: GoalVector                             # Deep desires in feature space
    flaws: list[CharacterFlaw]                    # Irrational biases
    pacing: PacingState                           # Drama budget system
    emotional_state: dict[str, float] = field(default_factory=lambda: {
        "anger": 0.0, "fear": 0.0, "hope": 0.0,
        "shame": 0.0, "affection": 0.0, "suspicion": 0.0
    })

    # === Social ===
    relationships: dict[str, RelationshipState] = field(default_factory=dict)
    # {agent_id: RelationshipState}

    # === Knowledge ===
    beliefs: dict[str, BeliefState] = field(default_factory=dict)
    # {secret_id: BeliefState}

    # === Resources ===
    alcohol_level: float = 0.0                    # [0.0, 1.0] — degrades composure

    # === Commitments ===
    commitments: list[str] = field(default_factory=list)
    # List of irreversible choice descriptions
```

### TypeScript

```typescript
interface AgentState {
    id: string;
    name: string;
    location: string;

    goals: GoalVector;
    flaws: CharacterFlaw[];
    pacing: PacingState;
    emotional_state: Record<string, number>;

    relationships: Record<string, RelationshipState>;
    beliefs: Record<string, BeliefState>;

    alcohol_level: number;           // [0.0, 1.0]
    commitments: string[];
}
```

### JSON Schema

```json
{
    "type": "object",
    "required": ["id", "name", "location", "goals", "flaws", "pacing", "emotional_state", "relationships", "beliefs", "alcohol_level", "commitments"],
    "properties": {
        "id": { "type": "string" },
        "name": { "type": "string" },
        "location": { "type": "string" },
        "goals": { "$ref": "#/definitions/GoalVector" },
        "flaws": { "type": "array", "items": { "$ref": "#/definitions/CharacterFlaw" } },
        "pacing": { "$ref": "#/definitions/PacingState" },
        "emotional_state": {
            "type": "object",
            "additionalProperties": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
        },
        "relationships": {
            "type": "object",
            "additionalProperties": { "$ref": "#/definitions/RelationshipState" }
        },
        "beliefs": {
            "type": "object",
            "additionalProperties": { "type": "string", "enum": ["unknown", "suspects", "believes_true", "believes_false"] }
        },
        "alcohol_level": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "commitments": { "type": "array", "items": { "type": "string" } }
    }
}
```

---

## 8. Update Rules & Formulas

### Per-Tick Update Order

After all events for a tick have been processed (deltas applied), the following updates run in order:

1. **Decay emotions** — all emotions move toward 0.0 (Section 6)
2. **Update composure** — eroded by stress and alcohol (Section 4)
3. **Update stress** — passive decay based on location privacy (Section 4)
4. **Replenish dramatic budget** — if no dramatic events this tick (Section 4)
5. **Decrement recovery timer** (Section 4)
6. **Check catastrophe threshold** — if met, generate CATASTROPHE event for next tick (Section 4)

### Delta Application

When a StateDelta is applied to an AgentState:

```python
def apply_delta(agent: AgentState, delta: StateDelta) -> None:
    """Apply a single StateDelta to an AgentState."""
    match delta.kind:
        case DeltaKind.AGENT_EMOTION:
            current = agent.emotional_state.get(delta.attribute, 0.0)
            if delta.op == DeltaOp.ADD:
                agent.emotional_state[delta.attribute] = clamp(current + delta.value, 0.0, 1.0)
            else:
                agent.emotional_state[delta.attribute] = clamp(delta.value, 0.0, 1.0)

        case DeltaKind.AGENT_RESOURCE:
            if delta.attribute == "alcohol_level":
                if delta.op == DeltaOp.ADD:
                    agent.alcohol_level = clamp(agent.alcohol_level + delta.value, 0.0, 1.0)
                else:
                    agent.alcohol_level = clamp(delta.value, 0.0, 1.0)

        case DeltaKind.AGENT_LOCATION:
            agent.location = delta.value  # always SET

        case DeltaKind.RELATIONSHIP:
            if delta.agent_b not in agent.relationships:
                agent.relationships[delta.agent_b] = RelationshipState()
            apply_relationship_delta(agent.relationships[delta.agent_b], delta)

        case DeltaKind.BELIEF:
            # delta.attribute = secret_id, delta.value = BeliefState string
            agent.beliefs[delta.attribute] = BeliefState(delta.value)

        case DeltaKind.COMMITMENT:
            agent.commitments.append(delta.value)

        case DeltaKind.PACING:
            current = getattr(agent.pacing, delta.attribute)
            if delta.op == DeltaOp.ADD:
                new_val = current + delta.value
            else:
                new_val = delta.value
            # Clamp based on attribute
            if delta.attribute in ("recovery_timer", "suppression_count"):
                new_val = max(0, int(new_val))
            else:
                new_val = clamp(new_val, 0.0, 1.0)
            setattr(agent.pacing, delta.attribute, new_val)

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))
```

---

## 9. Dinner Party Characters — Worked Examples

Six agents for the MVP "Dinner Party Protocol". These are complete initial states that the simulation starts with.

### Character 1: James Thorne (host)

**Role:** The host. Wealthy businessman. Married to Elena. Business partner with Marcus. Doesn't know his wife is having an affair with his business partner or that Marcus has been embezzling.

```json
{
    "id": "thorne",
    "name": "James Thorne",
    "location": "dining_table",
    "goals": {
        "safety": 0.4,
        "status": 0.9,
        "closeness": {
            "elena": 0.7,
            "marcus": 0.6,
            "lydia": 0.3,
            "diana": 0.4,
            "victor": 0.5
        },
        "secrecy": 0.3,
        "truth_seeking": 0.6,
        "autonomy": 0.7,
        "loyalty": 0.8
    },
    "flaws": [
        {
            "flaw_type": "pride",
            "strength": 0.8,
            "trigger": "status_threat",
            "effect": "overweight_status",
            "description": "Thorne's self-image as a successful, respected man is his armor. Anything that threatens this image triggers an outsized reaction."
        }
    ],
    "pacing": {
        "dramatic_budget": 1.0,
        "stress": 0.1,
        "composure": 0.9,
        "commitment": 0.0,
        "recovery_timer": 0,
        "suppression_count": 0
    },
    "emotional_state": {
        "anger": 0.0,
        "fear": 0.0,
        "hope": 0.3,
        "shame": 0.0,
        "affection": 0.4,
        "suspicion": 0.1
    },
    "relationships": {
        "elena": { "trust": 0.8, "affection": 0.6, "obligation": 0.3 },
        "marcus": { "trust": 0.7, "affection": 0.4, "obligation": 0.2 },
        "lydia": { "trust": 0.3, "affection": 0.2, "obligation": 0.0 },
        "diana": { "trust": 0.5, "affection": 0.3, "obligation": 0.1 },
        "victor": { "trust": 0.6, "affection": 0.3, "obligation": 0.1 }
    },
    "beliefs": {
        "secret_affair_01": "unknown",
        "secret_embezzle_01": "unknown",
        "secret_diana_debt": "unknown",
        "secret_lydia_knows": "unknown",
        "secret_victor_investigation": "unknown"
    },
    "alcohol_level": 0.0,
    "commitments": []
}
```

### Character 2: Elena Thorne (host's wife)

**Role:** Thorne's wife. Having a secret affair with Marcus. Feels trapped in her marriage. Confided in Diana about the affair.

```json
{
    "id": "elena",
    "name": "Elena Thorne",
    "location": "dining_table",
    "goals": {
        "safety": 0.7,
        "status": 0.4,
        "closeness": {
            "thorne": 0.1,
            "marcus": 0.8,
            "lydia": 0.2,
            "diana": 0.6,
            "victor": 0.2
        },
        "secrecy": 0.9,
        "truth_seeking": 0.3,
        "autonomy": 0.8,
        "loyalty": 0.3
    },
    "flaws": [
        {
            "flaw_type": "guilt",
            "strength": 0.7,
            "trigger": "intimacy_offered",
            "effect": "self_sacrifice",
            "description": "Elena's guilt about the affair makes her overly accommodating to Thorne, which paradoxically increases her stress."
        },
        {
            "flaw_type": "cowardice",
            "strength": 0.5,
            "trigger": "conflict_nearby",
            "effect": "avoid_confrontation",
            "description": "Elena will go to great lengths to avoid direct conflict, preferring to flee or deflect."
        }
    ],
    "pacing": {
        "dramatic_budget": 1.0,
        "stress": 0.3,
        "composure": 0.7,
        "commitment": 0.4,
        "recovery_timer": 0,
        "suppression_count": 0
    },
    "emotional_state": {
        "anger": 0.0,
        "fear": 0.2,
        "hope": 0.2,
        "shame": 0.3,
        "affection": 0.3,
        "suspicion": 0.0
    },
    "relationships": {
        "thorne": { "trust": 0.3, "affection": 0.1, "obligation": 0.6 },
        "marcus": { "trust": 0.7, "affection": 0.8, "obligation": 0.2 },
        "lydia": { "trust": 0.2, "affection": 0.1, "obligation": 0.0 },
        "diana": { "trust": 0.7, "affection": 0.5, "obligation": 0.1 },
        "victor": { "trust": 0.3, "affection": 0.1, "obligation": 0.0 }
    },
    "beliefs": {
        "secret_affair_01": "believes_true",
        "secret_embezzle_01": "unknown",
        "secret_diana_debt": "suspects",
        "secret_lydia_knows": "unknown",
        "secret_victor_investigation": "unknown"
    },
    "alcohol_level": 0.0,
    "commitments": ["maintain_affair_secrecy"]
}
```

### Character 3: Marcus Webb (business partner)

**Role:** Thorne's business partner. Having an affair with Elena. Embezzling from the firm. Charming, manipulative. The most secrets to keep.

```json
{
    "id": "marcus",
    "name": "Marcus Webb",
    "location": "dining_table",
    "goals": {
        "safety": 0.8,
        "status": 0.7,
        "closeness": {
            "thorne": 0.3,
            "elena": 0.6,
            "lydia": 0.1,
            "diana": 0.2,
            "victor": -0.3
        },
        "secrecy": 1.0,
        "truth_seeking": 0.1,
        "autonomy": 0.6,
        "loyalty": 0.2
    },
    "flaws": [
        {
            "flaw_type": "ambition",
            "strength": 0.8,
            "trigger": "loss_imminent",
            "effect": "overcommit",
            "description": "Marcus doubles down when cornered. Rather than cutting losses, he escalates deceptions."
        },
        {
            "flaw_type": "denial",
            "strength": 0.6,
            "trigger": "betrayal_detected",
            "effect": "deny_evidence",
            "description": "Marcus refuses to acknowledge the harm his actions cause. He rationalizes everything."
        }
    ],
    "pacing": {
        "dramatic_budget": 1.0,
        "stress": 0.4,
        "composure": 0.8,
        "commitment": 0.6,
        "recovery_timer": 0,
        "suppression_count": 0
    },
    "emotional_state": {
        "anger": 0.0,
        "fear": 0.3,
        "hope": 0.1,
        "shame": 0.1,
        "affection": 0.2,
        "suspicion": 0.2
    },
    "relationships": {
        "thorne": { "trust": 0.2, "affection": 0.1, "obligation": 0.5 },
        "elena": { "trust": 0.5, "affection": 0.5, "obligation": 0.3 },
        "lydia": { "trust": 0.1, "affection": 0.0, "obligation": 0.0 },
        "diana": { "trust": 0.2, "affection": 0.1, "obligation": 0.0 },
        "victor": { "trust": -0.3, "affection": -0.2, "obligation": 0.0 }
    },
    "beliefs": {
        "secret_affair_01": "believes_true",
        "secret_embezzle_01": "believes_true",
        "secret_diana_debt": "unknown",
        "secret_lydia_knows": "unknown",
        "secret_victor_investigation": "suspects"
    },
    "alcohol_level": 0.0,
    "commitments": ["maintain_affair_secrecy", "cover_embezzlement"]
}
```

### Character 4: Lydia Cross (the observer)

**Role:** Thorne's assistant. Observant and quiet. Has noticed discrepancies in the firm's books. Suspects (but isn't sure about) the affair. Loyal to Thorne but afraid of Marcus.

```json
{
    "id": "lydia",
    "name": "Lydia Cross",
    "location": "dining_table",
    "goals": {
        "safety": 0.8,
        "status": 0.3,
        "closeness": {
            "thorne": 0.5,
            "elena": 0.1,
            "marcus": -0.2,
            "diana": 0.3,
            "victor": 0.4
        },
        "secrecy": 0.4,
        "truth_seeking": 0.8,
        "autonomy": 0.5,
        "loyalty": 0.9
    },
    "flaws": [
        {
            "flaw_type": "cowardice",
            "strength": 0.7,
            "trigger": "conflict_nearby",
            "effect": "avoid_confrontation",
            "description": "Lydia knows things that could blow the evening apart, but her fear of conflict keeps her silent."
        },
        {
            "flaw_type": "loyalty",
            "strength": 0.6,
            "trigger": "betrayal_detected",
            "effect": "self_sacrifice",
            "description": "Lydia's loyalty to Thorne means she'll endure personal risk to protect him, even when telling him the truth would cause a scene."
        }
    ],
    "pacing": {
        "dramatic_budget": 1.0,
        "stress": 0.2,
        "composure": 0.8,
        "commitment": 0.3,
        "recovery_timer": 0,
        "suppression_count": 0
    },
    "emotional_state": {
        "anger": 0.1,
        "fear": 0.2,
        "hope": 0.1,
        "shame": 0.0,
        "affection": 0.2,
        "suspicion": 0.5
    },
    "relationships": {
        "thorne": { "trust": 0.7, "affection": 0.4, "obligation": 0.6 },
        "elena": { "trust": 0.2, "affection": 0.1, "obligation": 0.0 },
        "marcus": { "trust": -0.2, "affection": -0.1, "obligation": 0.0 },
        "diana": { "trust": 0.4, "affection": 0.3, "obligation": 0.0 },
        "victor": { "trust": 0.5, "affection": 0.2, "obligation": 0.0 }
    },
    "beliefs": {
        "secret_affair_01": "suspects",
        "secret_embezzle_01": "suspects",
        "secret_diana_debt": "unknown",
        "secret_lydia_knows": "believes_true",
        "secret_victor_investigation": "unknown"
    },
    "alcohol_level": 0.0,
    "commitments": []
}
```

### Character 5: Diana Forrest (the confidante)

**Role:** Elena's old friend. Knows about the affair (Elena confided in her). Has her own secret: she owes Marcus a large debt. This creates a loyalty conflict — she wants to help Elena but can't afford to cross Marcus.

```json
{
    "id": "diana",
    "name": "Diana Forrest",
    "location": "dining_table",
    "goals": {
        "safety": 0.6,
        "status": 0.5,
        "closeness": {
            "thorne": 0.3,
            "elena": 0.7,
            "marcus": -0.1,
            "lydia": 0.3,
            "victor": 0.2
        },
        "secrecy": 0.7,
        "truth_seeking": 0.4,
        "autonomy": 0.6,
        "loyalty": 0.7
    },
    "flaws": [
        {
            "flaw_type": "guilt",
            "strength": 0.6,
            "trigger": "intimacy_offered",
            "effect": "self_sacrifice",
            "description": "Diana feels guilty about her debt to Marcus and about keeping Elena's secret from Thorne. This guilt makes her over-accommodate everyone."
        },
        {
            "flaw_type": "jealousy",
            "strength": 0.4,
            "trigger": "rejection",
            "effect": "escalate_conflict",
            "description": "Diana envies Elena's comfortable life and Thorne's wealth, even as she sympathizes with Elena's unhappiness."
        }
    ],
    "pacing": {
        "dramatic_budget": 1.0,
        "stress": 0.15,
        "composure": 0.85,
        "commitment": 0.3,
        "recovery_timer": 0,
        "suppression_count": 0
    },
    "emotional_state": {
        "anger": 0.0,
        "fear": 0.1,
        "hope": 0.2,
        "shame": 0.2,
        "affection": 0.3,
        "suspicion": 0.1
    },
    "relationships": {
        "thorne": { "trust": 0.4, "affection": 0.3, "obligation": 0.1 },
        "elena": { "trust": 0.7, "affection": 0.6, "obligation": 0.2 },
        "marcus": { "trust": 0.1, "affection": -0.1, "obligation": 0.7 },
        "lydia": { "trust": 0.3, "affection": 0.2, "obligation": 0.0 },
        "victor": { "trust": 0.3, "affection": 0.2, "obligation": 0.0 }
    },
    "beliefs": {
        "secret_affair_01": "believes_true",
        "secret_embezzle_01": "unknown",
        "secret_diana_debt": "believes_true",
        "secret_lydia_knows": "unknown",
        "secret_victor_investigation": "unknown"
    },
    "alcohol_level": 0.0,
    "commitments": ["keep_elena_secret"]
}
```

### Character 6: Victor Hale (the investigator)

**Role:** A journalist and old college friend of Thorne's. Invited as a guest. Secretly investigating Marcus's business dealings for an article. Doesn't know about the affair. Sharp, observant, low empathy — sees people as sources.

```json
{
    "id": "victor",
    "name": "Victor Hale",
    "location": "dining_table",
    "goals": {
        "safety": 0.3,
        "status": 0.6,
        "closeness": {
            "thorne": 0.5,
            "elena": 0.1,
            "marcus": -0.2,
            "lydia": 0.3,
            "diana": 0.1
        },
        "secrecy": 0.6,
        "truth_seeking": 1.0,
        "autonomy": 0.8,
        "loyalty": 0.4
    },
    "flaws": [
        {
            "flaw_type": "obsession",
            "strength": 0.8,
            "trigger": "secret_exposure",
            "effect": "fixate_on_target",
            "description": "Victor's journalist instinct takes over when he smells a story. He'll push too hard, burn relationships, and ignore social cues in pursuit of the truth."
        },
        {
            "flaw_type": "vanity",
            "strength": 0.4,
            "trigger": "authority_challenge",
            "effect": "seek_validation",
            "description": "Victor values being seen as the smartest person in the room. He'll drop hints about what he knows to show off, even when discretion would serve him better."
        }
    ],
    "pacing": {
        "dramatic_budget": 1.0,
        "stress": 0.1,
        "composure": 0.9,
        "commitment": 0.2,
        "recovery_timer": 0,
        "suppression_count": 0
    },
    "emotional_state": {
        "anger": 0.0,
        "fear": 0.0,
        "hope": 0.3,
        "shame": 0.0,
        "affection": 0.1,
        "suspicion": 0.3
    },
    "relationships": {
        "thorne": { "trust": 0.5, "affection": 0.4, "obligation": 0.1 },
        "elena": { "trust": 0.3, "affection": 0.1, "obligation": 0.0 },
        "marcus": { "trust": -0.1, "affection": 0.0, "obligation": 0.0 },
        "lydia": { "trust": 0.3, "affection": 0.1, "obligation": 0.0 },
        "diana": { "trust": 0.2, "affection": 0.1, "obligation": 0.0 }
    },
    "beliefs": {
        "secret_affair_01": "unknown",
        "secret_embezzle_01": "suspects",
        "secret_diana_debt": "unknown",
        "secret_lydia_knows": "unknown",
        "secret_victor_investigation": "believes_true"
    },
    "alcohol_level": 0.0,
    "commitments": ["write_expose_on_marcus"]
}
```

### Initial Belief Matrix (Summary)

| Agent \ Secret | affair_01 | embezzle_01 | diana_debt | lydia_knows | victor_investigation |
|----------------|-----------|-------------|------------|-------------|---------------------|
| thorne | unknown | unknown | unknown | unknown | unknown |
| elena | believes_true | unknown | suspects | unknown | unknown |
| marcus | believes_true | believes_true | unknown | unknown | suspects |
| lydia | suspects | suspects | unknown | believes_true | unknown |
| diana | believes_true | unknown | believes_true | unknown | unknown |
| victor | unknown | suspects | unknown | unknown | believes_true |

This matrix shows the rich information asymmetry that drives the dinner party drama. Key observations:
- Thorne knows nothing — maximum irony potential
- Marcus has the most secrets to protect — maximum stress
- Lydia suspects the most but lacks courage to act — maximum internal tension
- Diana knows about the affair but is constrained by her debt to Marcus — maximum loyalty conflict
- Victor is investigating but doesn't know about the affair — his investigation could trigger unexpected reveals

---

## 10. Validation Rules

### AgentState Validation

1. `id` must be non-empty and unique among all agents
2. `name` must be non-empty
3. `location` must reference a valid Location.id
4. All emotional_state values must be in [0.0, 1.0]
5. `alcohol_level` must be in [0.0, 1.0]
6. All RelationshipState trust/affection values must be in [-1.0, 1.0]
7. All RelationshipState obligation values must be in [0.0, 1.0]
8. All GoalVector scalar values must be in [0.0, 1.0]
9. All GoalVector closeness values must be in [-1.0, 1.0]
10. All CharacterFlaw strength values must be in [0.0, 1.0]
11. All PacingState values must be in their defined ranges (see Section 4)
12. All BeliefState values must be valid enum values
13. An agent must have a relationship entry for every other agent in the simulation
14. An agent must have a belief entry for every secret in the simulation (default: UNKNOWN)

### GoalVector Validation

1. At least one scalar dimension must be > 0.5 (the agent must want something strongly)
2. `closeness` dict must contain entries for all other agents

### CharacterFlaw Validation

1. Each agent must have at least one flaw (flawless agents are boring)
2. Each agent should have at most 3 flaws (more becomes unmanageable)
3. `flaw_type` must be a valid FlawType enum value
4. `trigger` must be a recognized trigger tag
5. `effect` must be a recognized effect tag

---

## 11. NOT In Scope

- **How agents choose actions** — decision engine logic. See `specs/simulation/decision-engine.md`.
- **Specific dinner party backstories** — narrative design detail. See `specs/simulation/dinner-party-config.md`.
- **How goal vectors are used in scoring** — implementation detail. See `specs/simulation/decision-engine.md`.
- **Tension computation from agent state** — metrics concern. See `specs/metrics/tension-pipeline.md`.
- **Visualization of agent state** — rendering concern. See `specs/visualization/`.
- **Secret definitions** — world schema concern. See `specs/schema/world.md`.

---

## Dependencies

This spec is upstream of:
- `specs/simulation/decision-engine.md` (uses GoalVector, CharacterFlaw for action scoring)
- `specs/simulation/pacing-physics.md` (uses PacingState, catastrophe threshold)
- `specs/simulation/tick-loop.md` (applies deltas to AgentState)
- `specs/simulation/dinner-party-config.md` (instantiates 6 AgentStates)
- `specs/metrics/tension-pipeline.md` (reads emotional_state, relationships, pacing for sub-metrics)
- `specs/metrics/irony-and-beliefs.md` (reads belief matrix)
- `specs/visualization/thread-layout.md` (positions threads by agent)

This spec depends on:
- `specs/schema/events.md` (StateDelta, DeltaKind, DeltaOp — how state changes are expressed)
- `specs/schema/world.md` (Location ids, Secret definitions — circular dependency resolved by referencing IDs only)
