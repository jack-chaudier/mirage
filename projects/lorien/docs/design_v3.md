# NarrativeField v3: The Engineering Specification

## What This Document Is

v1 was the vision. v2 was the refined plan. This is the **engineering spec** — every open question from four rounds of critique is resolved with a concrete decision, a justification, and (where relevant) the data structure that implements it.

The 17-point adversarial critique was the most valuable input so far. It found real structural problems that would have caused painful rewrites later. This document addresses every one.

---

## Decision Log: The 17 Resolved Questions

### Decision 1: Event-Primary Graph (Not Bipartite)

**The question:** Should nodes be world-states, events, or a bipartite mix?

**Decision:** Event-primary. Events are nodes. World-state is a **cached materialized view** derived by replaying events from the nearest snapshot.

**Why:** The bipartite approach is theoretically clean but doubles graph size and complicates every query. For the MVP scope (dinner party = ~100-200 events), the overhead isn't justified. Event-primary matches how writers think ("what happened") and how the visualizer works (each dot on the map is an event, not a state).

**State access:** Periodic snapshots every 20 events. To get world-state at event N, find nearest snapshot ≤ N, replay forward. Index tables (below) handle the fast-path queries.

**If this breaks later:** Migrating to bipartite is straightforward — you're just promoting snapshots to first-class nodes and adding edges. The event schema doesn't change.

### Decision 2: CQRS Read Layer (Snapshots + Indices)

**The question:** "Replay everything" will kill interactivity.

**Decision:** Event log is source of truth. Add a read-optimized layer:

```
Snapshots:
    Every 20 events (or every 5 sim-minutes), persist full WorldState.

Index Tables (built incrementally as events are appended):
    event_id       → Event
    agent_id       → List[event_id]          (agent's timeline)
    location_id    → List[event_id]          (events at location)
    event_id       → List[agent_id]          (participants)
    secret_id      → List[event_id]          (events touching this secret)
    (agent_a, agent_b) → List[event_id]      (shared interaction history)

Materialized Views (recomputed on demand, cached):
    relationship_graph_at(t)  → adjacency matrix of trust/affection
    tension_components_at(t)  → per-agent tension sub-metric vector
    belief_matrix_at(t)       → agent × secret → belief_state
    cluster_assignments_at(t) → community detection output
```

This is event sourcing + CQRS without the enterprise bloat. Hover tooltips read from indices. Drag-to-resimulate replays from nearest snapshot.

### Decision 3: Typed Deltas (Not String Paths)

**The question:** `StateDelta.target: str` like `"relationship[A,B].trust"` is a footgun.

**Decision:** Typed delta discriminated union.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Union

class DeltaKind(Enum):
    AGENT_EMOTION    = "agent_emotion"
    AGENT_RESOURCE   = "agent_resource"
    AGENT_LOCATION   = "agent_location"
    RELATIONSHIP     = "relationship"
    BELIEF           = "belief"
    SECRET_STATE     = "secret_state"
    WORLD_RESOURCE   = "world_resource"
    COMMITMENT       = "commitment"

class DeltaOp(Enum):
    SET = "set"
    ADD = "add"

@dataclass
class StateDelta:
    kind: DeltaKind
    agent: str                          # primary agent affected
    agent_b: str | None = None          # for relationships / beliefs about
    attribute: str = ""                 # e.g., "trust", "anger", "gold"
    op: DeltaOp = DeltaOp.ADD
    value: float | str | bool = 0.0
    reason_code: str = ""               # machine-readable: "BETRAYAL_OBSERVED"
    reason_display: str = ""            # human-readable: "Saw the affair"
```

This is validatable, type-checkable, refactor-safe, and fast to apply. The `reason_code` enables analytics ("how many BETRAYAL_OBSERVED deltas in this run?"). The `reason_display` goes in tooltips.

### Decision 4: Discrete Ticks + Event Queue (Time Model)

**The question:** Turn-based ticks? Continuous time? Concurrent events?

**Decision:** Discrete ticks with simultaneous action resolution.

```
Each tick:
    1. Every agent proposes an action (based on current perceived state)
    2. Conflict resolution:
       - If actions are compatible → all execute
       - If actions conflict → priority rules + randomness
       - "Simultaneous" actions share a tick_id
    3. Generate Event objects with:
       - sim_time: float (the tick's time value)
       - tick_id: int (which tick this belongs to)
       - order_in_tick: int (resolution order within the tick)
    4. Apply deltas, update indices, maybe snapshot
```

**Why ticks over continuous:** Continuous scheduling (priority queues of next-event times) is elegant but adds complexity that doesn't pay off at dinner-party scale. Ticks are debuggable, reproducible, and easy to reason about. The `order_in_tick` field preserves causal ordering within a tick.

**Concurrency:** Two people CAN talk simultaneously (same tick, different order_in_tick). Someone CAN overhear (if `location_id` matches and they're not the target). This is enough for the dinner party without building a full concurrent event system.

**If this breaks later:** Migrating to continuous time means replacing the tick loop with a priority queue and adding duration fields to events. The event schema barely changes — `sim_time` already exists.

### Decision 5: Pacing Physics (The Drama Budget)

**The question:** "Narrative gravity" pulling agents toward drama will produce constant chaos — melodrama generators, betrayal machines, emotionally flat.

**Decision:** Add explicit pacing forces that create the tension-release-tension rhythm stories actually have.

```python
@dataclass
class AgentPacingState:
    dramatic_budget: float = 1.0     # 0-1, replenishes in quiet beats
    stress: float = 0.0              # accumulates with conflict exposure
    composure: float = 1.0           # ability to mask true feelings
    recovery_timer: int = 0          # ticks until next dramatic action allowed

# Pacing rules:
# 1. Dramatic actions (confront, reveal, betray) COST dramatic_budget
# 2. Budget replenishes slowly during non-dramatic ticks (chat, eat, drink)
# 3. Stress rises with conflict exposure (even witnessing conflict)
# 4. High stress + low composure → involuntary actions (blurting, crying, leaving)
# 5. High stress + high composure → SUPPRESSION (builds toward bigger catastrophe)
# 6. Hysteresis: repairing trust costs 3x more than breaking it
# 7. Social masking: agents suppress conflict in public settings
#    (drama more likely in private spaces — kitchen, balcony)
```

This produces:
- **Escalation** (stress accumulates across small events)
- **Cooldown** (budget depletion forces quiet beats)
- **Bigger payoff** (suppressed tension leads to larger catastrophes when composure breaks)
- **Spatial dynamics** (agents seek private spaces for confrontation)

The dinner party naturally pressures this system: forced proximity prevents escape, alcohol degrades composure, and social norms enforce masking — until they don't.

### Decision 6: Vectorized Feature Scoring (Not Continuous Gradients)

**The question:** ∇U and cross products are math cosplay if utilities are discrete heuristics.

**Decision:** Keep the gradient *language* for conceptual communication, but implement as vectorized feature scoring.

```python
# Each agent's "desire vector" in feature space
@dataclass
class GoalVector:
    safety: float       # desire to avoid danger
    status: float       # desire for social position
    closeness: dict     # {agent_id: desire_for_closeness}
    secrecy: float      # desire to keep secrets hidden
    truth: float        # desire for honesty / revealing truth
    autonomy: float     # desire for independence
    loyalty: float      # desire to honor commitments

# "Conflict intensity" between agents:
def conflict_score(a: GoalVector, b: GoalVector) -> float:
    """Cosine distance between goal vectors = how opposed their desires are."""
    # High score = goals point in opposite directions = conflict
    ...

# "Best action" for an agent:
def score_action(agent, action, world_state) -> float:
    """Weighted dot product of action's predicted effects with agent's goal vector."""
    predicted_deltas = simulate_action_effects(action, world_state)
    return dot(predicted_deltas, agent.goal_vector) + bias_from_flaws(agent)
```

The `bias_from_flaws` function is where irrationality enters — pride overweights status, loyalty overweights closeness-to-allies, trauma avoids certain action types regardless of utility.

### Decision 7: Promises as Search Targets (Not Physics Wells)

**The question:** Soft attractors will secretly become hard rails.

**Decision:** Promises are *not* part of the simulation physics. They're post-hoc search targets.

```
Workflow:
1. User defines promises: "A betrayal must occur by midpoint"
2. Simulation runs WITHOUT the promise influencing agent decisions
3. System checks: "Did the promise occur naturally?"
4. If YES → highlight it on the map
5. If NO → system reports:
   a. "Closest approach to this promise" (when did it almost happen?)
   b. "Suggested interventions" — diegetic pressure events that could nudge:
      - A rumor spreads
      - A deadline appears
      - New information enters
      - An external shock occurs
   c. User can inject one intervention and re-simulate
6. System also reports MULTIPLE paths to satisfy the promise:
   "Revolution could occur via military coup, popular uprising, or political scandal"
```

This preserves agent autonomy. The world responds to authored pressure rather than being secretly pulled. Writers get control without railroading.

### Decision 8: Finite Proposition Catalog (Beliefs & Irony)

**The question:** The irony math needs propositions with probabilities, but building full epistemic logic will kill you.

**Decision:** Finite, enumerable propositions. No probability distributions — just a belief state enum.

```python
@dataclass
class Secret:
    id: str
    holder: str              # who generated/holds this secret
    about: str | None        # which agent it's about (if any)
    content_type: str        # "affair", "debt", "identity", "plan"
    truth_value: bool        # ground truth

class BeliefState(Enum):
    UNKNOWN    = "unknown"         # hasn't encountered this proposition
    SUSPECTS   = "suspects"        # has partial evidence
    BELIEVES   = "believes_true"   # confident it's true
    DENIES     = "believes_false"  # confident it's false (possibly wrong)

# Belief matrix: agent × secret → BeliefState
# Stored as: Dict[str, Dict[str, BeliefState]]
# beliefs["Thorne"]["secret_affair_01"] = BeliefState.UNKNOWN

# Irony computation:
def irony_score(agent_id: str, beliefs: dict, secrets: dict) -> float:
    """Count of false beliefs + critical unknowns for this agent."""
    score = 0.0
    for secret_id, secret in secrets.items():
        belief = beliefs[agent_id].get(secret_id, BeliefState.UNKNOWN)
        if secret.truth_value and belief == BeliefState.DENIES:
            score += 2.0   # actively wrong = high irony
        elif secret.truth_value and belief == BeliefState.UNKNOWN:
            if secret.about == agent_id or relevance(agent_id, secret) > 0.5:
                score += 1.0  # doesn't know something they should
    return score

def scene_irony(agents_present: list, beliefs: dict, secrets: dict) -> float:
    """Sum of irony across all agents in the scene."""
    return sum(irony_score(a, beliefs, secrets) for a in agents_present)
```

This is cheap, robust, and produces meaningful numbers. The moment `scene_irony` drops sharply (a reveal event sets beliefs to match truth), you have a detectable "irony collapse" — the plot twist.

### Decision 9: Defined Future Summary Space (Counterfactual Impact)

**The question:** "Compare distributions over futures" — futures of *what*?

**Decision:** Three canonical future summaries, computed at a fixed horizon H (for dinner party: H = 20 ticks forward).

```python
@dataclass
class FutureSummary:
    # 1. Relationship outcome matrix
    #    For each pair: allied / neutral / hostile at horizon
    relationship_outcomes: Dict[tuple, str]  # ("A","B") → "hostile"
    
    # 2. Knowledge state
    #    For each secret: who knows it at horizon
    knowledge_distribution: Dict[str, List[str]]  # secret_id → [agents who know]
    
    # 3. Social outcome
    #    For each agent: status delta, still_present (or left), emotional_state
    agent_outcomes: Dict[str, AgentOutcome]

@dataclass
class AgentOutcome:
    still_present: bool
    status_delta: float      # gained or lost social standing
    emotional_state: str     # "humiliated", "vindicated", "oblivious", etc.

# Impact computation:
def counterfactual_impact(event: Event, world: WorldState, n_samples: int = 50) -> float:
    """
    Simulate n_samples futures WITH the event, n_samples WITHOUT.
    Compare distributions over FutureSummary using Jensen-Shannon divergence.
    """
    futures_with = [simulate_forward(world.after(event), H) for _ in range(n_samples)]
    futures_without = [simulate_forward(world.without(event), H) for _ in range(n_samples)]
    
    summaries_with = [summarize(f) for f in futures_with]
    summaries_without = [summarize(f) for f in futures_without]
    
    return jensen_shannon_divergence(summaries_with, summaries_without)
```

**Why Jensen-Shannon over KL:** JSD is symmetric (the "distance" between A and B is the same as B to A) and always finite. KL divergence can blow up when one distribution assigns zero probability to something the other doesn't. JSD is better behaved for this use case.

**Cost:** 100 forward simulations per event queried. Expensive — but you don't compute this for every event. Only for:
- Events the user clicks on ("how important was this?")
- Events that cross a tension threshold (automatic flagging)
- Story query results (ranking candidate arcs)

### Decision 10: Two-Parameter Catastrophe (Not Just Thresholding)

**The question:** "If stress > threshold → betray" is a trigger, not catastrophe theory.

**Decision:** Implement a genuine cusp catastrophe with two control parameters.

```
Control Parameter A: Stress (accumulates continuously)
Control Parameter B: Commitment / Lock-in (how much the agent has invested in the current path)

The cusp surface:
- Low stress, low commitment → stable equilibrium (polite behavior)
- High stress, low commitment → gradual shift (agent withdraws, avoids)
- Low stress, high commitment → stable equilibrium (loyal behavior)
- High stress, HIGH commitment → CATASTROPHE ZONE
  The agent can't withdraw (too committed) and can't endure (too stressed)
  → sudden discontinuous transition: betrayal, breakdown, explosion

Implementation:
    catastrophe_potential = stress * commitment^2
    if catastrophe_potential > threshold AND composure < minimum:
        → trigger catastrophe event (type depends on agent personality)
        → betrayal if dominant trait is ambition
        → breakdown if dominant trait is loyalty
        → explosion if dominant trait is pride
```

This produces RARE, meaningful cliffs — not constant drama. The two-parameter requirement means an agent needs to be BOTH highly stressed AND deeply committed before a catastrophe occurs. One without the other produces gradual changes (withdrawal or loyal endurance), not sudden snaps.

### Decision 11: TDA-Lite (Union-Find, Not Full Vietoris-Rips)

**The question:** Persistent homology is computationally brutal at scale.

**Decision:** Union-find for H₀ (connected components). Graph cycle heuristics for loop detection. No full Vietoris-Rips complex in MVP.

```python
# H₀: Connected components at varying interaction thresholds
def narrative_components(events: list, thresholds: list[float]) -> dict:
    """
    At each threshold t, two agents are "connected" if they share 
    an interaction with tension > t in the last N ticks.
    
    Returns: {threshold: number_of_components}
    This answers: "How many disconnected storylines exist at each drama level?"
    """
    results = {}
    for t in thresholds:
        uf = UnionFind(all_agents)
        for event in events:
            if event.metrics["tension"] >= t:
                for pair in combinations(event.participants, 2):
                    uf.union(*pair)
        results[t] = uf.num_components()
    return results

# Loop detection: graph cycle heuristics (not full H₁)
def detect_narrative_cycles(events: list, window: int = 30) -> list:
    """
    Find agents who return to similar states after a sequence of changes.
    Flag as "possible cyclical dynamic" — not automatic "theme."
    """
    # Compare agent state at tick T with state at tick T-window
    # If cosine similarity > 0.8, flag as potential cycle
    ...
```

Full persistent homology is a Phase 4+ feature. For MVP, union-find H₀ gives you the most valuable structural insight (storyline connectivity) at trivial computational cost.

### Decision 12: Flexible X-Axis API (Pinned Time as Default)

**The question:** Pinning x = time bakes in chronological worldview, breaks for flashbacks/non-linear.

**Decision:** Pin time for MVP, but design the axis API to be swappable.

```typescript
// Frontend axis configuration
interface AxisConfig {
    mode: "sim_time" | "scene_index" | "reveal_order" | "reader_knowledge" | "custom";
    label: string;
    mapper: (event: Event) => number;  // maps event to x-position
}

// Default: chronological
const defaultX: AxisConfig = {
    mode: "sim_time",
    label: "Time",
    mapper: (e) => e.sim_time
};

// Future: scene-based (non-linear storytelling)
const sceneX: AxisConfig = {
    mode: "scene_index",
    label: "Scene",
    mapper: (e) => e.scene_id  // requires scene segmentation layer
};
```

This costs nothing to implement now but prevents hardwiring assumptions into the renderer.

### Decision 13: Noisy Fake Data

**The question:** Handcrafted data will be too clean; real sim data will be messy.

**Decision:** The fake data generator produces three tiers of events.

```
Tier 1: Story-critical events (the "real" dinner party plot)
    ~20 events with clear causal chains, high tension, meaningful

Tier 2: Texture events (realistic but low-importance)
    ~40 events: small talk, pouring wine, passing dishes, going to bathroom
    These test: does the visualizer handle noise without becoming unreadable?

Tier 3: Ambiguity events (deliberately messy)
    ~10 events: simultaneous conversations, misheard dialogue,
    contradictory signals (someone smiles while angry)
    These test: does the system degrade gracefully with imperfect data?
```

Total: ~70 events. The renderer must be legible with all three tiers present, not just the clean Tier 1 events.

### Decision 14: Arc Grammars for Story Queries

**The question:** Path-search optimizers will find "arc hacks" — highlight reels without setup, exploiting metric loopholes.

**Decision:** Hard structural constraints + soft scoring.

```python
@dataclass
class ArcGrammar:
    """Minimum beat structure that a valid story arc must satisfy."""
    required_beats: list  # ordered list of beat types
    
    # Default: classic 5-beat structure
    # ["setup", "complication", "escalation", "turning_point", "consequence"]

class BeatType(Enum):
    SETUP        = "setup"        # introduces character + situation
    COMPLICATION = "complication"  # new obstacle or information
    ESCALATION   = "escalation"   # tension increases
    TURNING_POINT = "turning"     # moment of irreversible change
    CONSEQUENCE  = "consequence"  # aftermath / new equilibrium

def validate_arc(events: list[Event], grammar: ArcGrammar) -> bool:
    """Check that the event sequence satisfies the beat grammar."""
    # 1. Must be causally connected (each event links to a prior one)
    # 2. Must contain at least one event matching each required beat
    # 3. Beats must appear in grammar order
    # 4. Must span minimum time duration (no instant stories)
    # 5. Must involve consistent character set (protagonist appears in >60% of events)
    ...

def score_arc(events: list[Event], weights: TensionWeights) -> float:
    """Soft scoring: higher = better story. Only called on grammar-valid arcs."""
    # Weighted sum of: total tension variance, peak tension,
    # counterfactual impact of turning point, thematic coherence,
    # irony peak-to-trough range
    ...
```

Story queries first filter by grammar validity, then rank by score. This prevents highlight-reel arcs.

### Decision 15: Precomputed Hover Neighborhoods

**The question:** Real-time pathfinding on hover will melt frame rate.

**Decision:** Precompute causal neighborhoods at load time. Hover is pure lookup.

```python
# At data load time, precompute:
causal_graph = {}  # event_id → {backward: set[event_id], forward: set[event_id]}

for event in all_events:
    # Backward: events in this event's causal_links (depth 2-3)
    causal_graph[event.id]["backward"] = bfs_backward(event, depth=3)
    
    # Forward: events that list this event in their causal_links (depth 2-3)
    causal_graph[event.id]["forward"] = bfs_forward(event, depth=3)

# At hover time: O(1) lookup
def on_hover(event_id):
    highlight(causal_graph[event_id]["backward"], opacity=0.6)
    highlight(causal_graph[event_id]["forward"], opacity=0.8)
    highlight([event_id], opacity=1.0)
```

For the dinner party (~70-200 events), this precomputation takes milliseconds. BFS depth of 3 gives a useful "causal cone" without overwhelming the display. Full pathfinding happens on click, not hover.

### Decision 16: Scene Segmentation as First-Class Layer

**The question:** Writers think in scenes, not events.

**Decision:** Add an explicit scene layer between events and arcs.

```python
@dataclass
class Scene:
    id: str
    events: list[str]           # event IDs in this scene
    location: str               # primary location
    participants: set[str]      # agents present
    time_start: float
    time_end: float
    tension_arc: list[float]    # tension values across the scene
    dominant_theme: str         # highest-delta thematic axis
    scene_type: str             # "confrontation", "revelation", "bonding", etc.

def segment_into_scenes(events: list[Event]) -> list[Scene]:
    """
    Group events into scenes based on:
    1. Location continuity (same location = same scene)
    2. Participant overlap (>50% shared participants)  
    3. Tension continuity (no long gaps of zero activity)
    4. Time proximity (events within N ticks of each other)
    
    Scene boundary = location change OR significant participant turnover 
                     OR tension drop to near-zero for multiple ticks.
    """
    ...
```

The map shows event density. The sidebar shows scenes. Selecting an arc auto-selects the best scene cuts. Exporting a beat sheet exports scene summaries, not raw events.

### Decision 17: MVP 1.5 — "The Bridge" (Two-Location Stress Test)

**The question:** The dinner party is too convergent — it hides the hard problem of disconnected storylines.

**Decision:** Keep Dinner Party as MVP 1. Plan a deliberate MVP 1.5:

```
"The Bridge"
- 8 agents
- 2 locations: a tavern and a manor house
- 1 bottleneck: a bridge (travel takes 3 ticks, can be blocked)
- Some agents start at tavern, some at manor
- A secret must travel between locations via an agent who crosses

This stress-tests:
- Disconnected narrative components (H₀ > 1)
- Convergence timing (when do the two storylines collide?)
- Information propagation (how do secrets travel between communities?)
- The Y-axis embedding (must show spatial separation clearly)
```

---

## The Revised Schema (Incorporating All Decisions)

```python
"""
NarrativeField Core Schema v3
All 17 architectural decisions implemented.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# ============================================================
# ENUMS
# ============================================================

class EventType(Enum):
    CHAT         = "chat"           # Low-tension social maintenance
    OBSERVE      = "observe"        # Overhearing, noticing, watching
    SOCIAL_MOVE  = "social_move"    # Changing location, seating
    REVEAL       = "reveal"         # Information transfer (intentional)
    CONFLICT     = "conflict"       # Confrontation, accusation
    INTERNAL     = "internal"       # Thought, decision (invisible to others)
    PHYSICAL     = "physical"       # Action: pour drink, flip table
    CONFIDE      = "confide"        # Private sharing of secret/feeling
    LIE          = "lie"            # Deliberate misinformation
    CATASTROPHE  = "catastrophe"    # Involuntary break: breakdown, explosion

class DeltaKind(Enum):
    AGENT_EMOTION    = "agent_emotion"
    AGENT_RESOURCE   = "agent_resource"
    AGENT_LOCATION   = "agent_location"
    RELATIONSHIP     = "relationship"
    BELIEF           = "belief"
    SECRET_STATE     = "secret_state"
    WORLD_RESOURCE   = "world_resource"
    COMMITMENT       = "commitment"
    PACING           = "pacing"

class DeltaOp(Enum):
    SET = "set"
    ADD = "add"

class BeliefState(Enum):
    UNKNOWN       = "unknown"
    SUSPECTS      = "suspects"
    BELIEVES_TRUE = "believes_true"
    BELIEVES_FALSE = "believes_false"

class BeatType(Enum):
    SETUP         = "setup"
    COMPLICATION  = "complication"
    ESCALATION    = "escalation"
    TURNING_POINT = "turning_point"
    CONSEQUENCE   = "consequence"


# ============================================================
# CORE DATA STRUCTURES
# ============================================================

@dataclass
class StateDelta:
    """The atom of change. Typed, validatable, refactor-safe."""
    kind: DeltaKind
    agent: str
    agent_b: Optional[str] = None       # for relationships / beliefs about
    attribute: str = ""                  # e.g., "trust", "anger", "gold"
    op: DeltaOp = DeltaOp.ADD
    value: float | str | bool = 0.0
    reason_code: str = ""               # machine-readable
    reason_display: str = ""            # human-readable (tooltip)


@dataclass
class Event:
    """A single thing that happened. The atomic unit of the fabula graph."""
    id: str
    sim_time: float
    tick_id: int
    order_in_tick: int                  # resolution order within simultaneous tick
    type: EventType
    
    # Participants
    source_agent: str
    target_agents: list[str]
    location_id: str
    
    # Causality (the "crystallize" data)
    causal_links: list[str]             # IDs of events that caused this
    
    # State changes
    deltas: list[StateDelta]
    
    # Human-readable
    description: str
    beat_type: Optional[BeatType] = None  # narrative function of this event
    
    # Computed metrics (populated by metrics pipeline, not simulation)
    metrics: dict = field(default_factory=lambda: {
        "tension": 0.0,
        "irony": 0.0,
        "significance": 0.0,
        "thematic_shift": {},           # axis_name → delta
    })


@dataclass
class Secret:
    """A piece of information that can be known, unknown, or misbelieved."""
    id: str
    holder: str                         # who generated this secret
    about: Optional[str] = None         # which agent it's about
    content_type: str = ""              # "affair", "debt", "identity", "plan"
    content_display: str = ""           # human-readable description
    truth_value: bool = True            # ground truth


@dataclass
class GoalVector:
    """An agent's desires in feature space. Used for action scoring."""
    safety: float = 0.5
    status: float = 0.5
    closeness: dict = field(default_factory=dict)   # agent_id → desire
    secrecy: float = 0.5
    truth_seeking: float = 0.5
    autonomy: float = 0.5
    loyalty: float = 0.5


@dataclass
class PacingState:
    """Per-agent drama pacing. Prevents constant chaos."""
    dramatic_budget: float = 1.0        # 0-1, replenishes in quiet beats
    stress: float = 0.0                 # accumulates with conflict exposure
    composure: float = 1.0             # ability to mask true state
    commitment: float = 0.0            # investment in current path (catastrophe param 2)
    recovery_timer: int = 0            # ticks until next dramatic action allowed


@dataclass
class CharacterFlaw:
    """Irrational bias that makes agents story-shaped, not optimal."""
    name: str                           # "pride", "loyalty", "trauma", etc.
    strength: float                     # 0-1, how much this distorts decisions
    trigger: str                        # what activates it: "status_threat", "betrayal", etc.
    effect: str                         # "overestimate_self", "avoid_confrontation", etc.


@dataclass
class AgentState:
    """Complete state of one agent at a point in time."""
    id: str
    name: str
    location: str
    
    # Psychology
    goals: GoalVector
    flaws: list[CharacterFlaw]
    pacing: PacingState
    emotional_state: dict = field(default_factory=lambda: {
        "anger": 0.0, "fear": 0.0, "hope": 0.0,
        "shame": 0.0, "affection": 0.0, "suspicion": 0.0
    })
    
    # Social
    relationships: dict = field(default_factory=dict)
    # {agent_id: {"trust": 0.7, "affection": 0.3, "obligation": 0.1}}
    
    # Knowledge
    beliefs: dict = field(default_factory=dict)
    # {secret_id: BeliefState}
    
    # Resources
    alcohol_level: float = 0.0
    
    # Commitments (irreversible choices)
    commitments: list[str] = field(default_factory=list)


@dataclass
class Location:
    """A place where events can occur."""
    id: str
    name: str
    privacy: float                      # 0 = fully public, 1 = fully private
    capacity: int                       # max agents
    adjacent: list[str]                 # connected location IDs
    overhear_from: list[str]            # locations you can eavesdrop on from here


@dataclass
class Scene:
    """A group of events forming a dramatic unit. First-class layer."""
    id: str
    event_ids: list[str]
    location: str
    participants: set[str]
    time_start: float
    time_end: float
    tension_arc: list[float]
    dominant_theme: str
    scene_type: str                     # "confrontation", "revelation", "bonding", etc.


@dataclass
class WorldState:
    """Complete snapshot of the world. Cached periodically for CQRS."""
    tick_id: int
    sim_time: float
    agents: dict[str, AgentState]       # agent_id → AgentState
    secrets: dict[str, Secret]          # secret_id → Secret
    locations: dict[str, Location]      # location_id → Location
    global_tension: float = 0.0
    active_scenes: list[str] = field(default_factory=list)


# ============================================================
# TENSION WEIGHTS (User-Tunable)
# ============================================================

@dataclass
class TensionWeights:
    """User controls what 'tension' means for their story type."""
    danger: float = 1.0
    time_pressure: float = 1.0
    goal_frustration: float = 1.0
    relationship_volatility: float = 1.0
    information_gap: float = 1.0
    resource_scarcity: float = 1.0
    moral_cost: float = 1.0
    irony_density: float = 1.0

    # Presets
    @classmethod
    def thriller(cls):
        return cls(danger=2.0, time_pressure=2.0, information_gap=1.5)
    
    @classmethod
    def relationship_drama(cls):
        return cls(relationship_volatility=2.0, moral_cost=2.0, irony_density=1.5)
    
    @classmethod
    def mystery(cls):
        return cls(information_gap=2.5, irony_density=2.0)


# ============================================================
# THEMATIC AXES
# ============================================================

THEMATIC_AXES = {
    "loyalty_betrayal":     (-1.0, 1.0),   # -1 = betrayal, +1 = loyalty
    "freedom_control":      (-1.0, 1.0),
    "love_duty":            (-1.0, 1.0),
    "innocence_corruption": (-1.0, 1.0),
    "truth_deception":      (-1.0, 1.0),
    "order_chaos":          (-1.0, 1.0),
}
```

---

## The Revised Build Order

### Phase 1: The Renderer (Weeks 1-3)
Build the visualizer with fake data (3 tiers: story-critical + texture + ambiguity).

Deliverable: An interactive 2D map (time × character-lanes) with:
- Tension heatmap
- Character arc curves (using spring algorithm for Y-positioning)
- Hover → precomputed causal neighborhood highlight
- Click → full arc crystallization
- Scene boundaries marked
- Legible even with noisy Tier 2/3 events

### Phase 2: Toy Simulation — Dinner Party (Weeks 4-7)
Build the agent simulation with pacing physics, typed deltas, tick-based time, and finite belief catalog.

Deliverable: A simulation that produces 100-200 events for 6 agents at a dinner party, feeds into the Phase 1 renderer, and produces recognizable narrative structure.

### Phase 3: Metrics Pipeline (Weeks 8-10)
Compute tension (weighted sub-metrics), irony (belief matrix), thematic shifts, and scene segmentation on the event log.

Deliverable: The renderer shows computed metrics instead of hand-assigned ones. Tension presets (thriller/drama/mystery) produce visibly different maps from the same event log.

### Phase 4: Story Extraction (Weeks 11-13)
Arc grammar validation + soft scoring. Beat sheet export. 1-2 LLM-written scenes.

Deliverable: Select a region on the map → get a structured beat sheet → get sample prose.

### Phase 5: Counterfactual Impact (Weeks 14-16)
Branch simulation. JSD over future summaries. Impact scores on events.

Deliverable: Click an event → see "how much this mattered" as a number with explanation.

### Phase 6: Story Queries (Weeks 17-20)
Natural language → path search with grammar constraints + scoring.

Deliverable: "Find me a tragedy for Character X" → highlighted arc on map.

### Phase 7: MVP 1.5 — The Bridge (Weeks 21-24)
Two-location world. Tests disconnected components, information propagation, convergence timing.

Deliverable: Proof that the system works beyond a single room.

---

## The Creative Provocation: Braids, Not Terrain?

Document 4 asked: what if the visualization isn't terrain at all, but a **braid of threads**?

This is worth taking seriously. A braid/subway-map/river-delta metaphor has advantages:

- **More writer-native.** Writers think in "threads" and "weaving," not "topological manifolds."
- **Better for the core interaction.** Hover-to-select a thread is more intuitive than hover-to-select a region of terrain.
- **Still supports all the math.** Tension can be thread thickness or color intensity. Convergence is threads touching. Catastrophe is a thread snapping or color-shifting.
- **Solves the 2D/3D debate.** Braids are naturally 2D (time × character-space) with optional 3D (tension as vertical displacement of threads).

**Recommendation:** Build the Phase 1 renderer as a braid/thread visualization. The "terrain" view can be a secondary mode that interprets the same data differently. Test with users which metaphor they reach for first.

The math doesn't change. The data structures don't change. Only the rendering metaphor changes. And "a loom of narrative threads that crystallize when you touch them" is a *better pitch* than "a topological manifold with scalar fields."

---

## The Elevator Pitch (v3)

*"NarrativeField simulates fictional worlds — agents with desires, secrets, and flaws, colliding under pressure — and weaves the results into an interactive tapestry of narrative threads. Each thread is a character's journey. Where threads cross, characters interact. Where threads glow hot, tension peaks. Hover, and the causal chains crystallize: you see exactly what led to this moment and what follows from it. Select a thread, and you've selected a story — with structure guaranteed by the simulation, meaning measured by the math, and prose filled in by AI. It's not AI writing stories. It's AI helping you find the stories that were always in the world."*
