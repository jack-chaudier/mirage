# Tick Loop Specification

> **Status:** Approved (with 5 fixes applied)
> **Author:** sim-designer
> **Dependencies:** specs/schema/events.md (#1) — DONE, specs/schema/agents.md (#2) — DONE
> **Dependents:** specs/integration/data-flow.md (#17)
> **Doc3 Decisions:** #1 (event-primary graph), #2 (CQRS read layer), #4 (discrete ticks + event queue), #10 (two-parameter catastrophe), #16 (scene segmentation)

---

## 1. Purpose

This spec defines the complete lifecycle of a single simulation tick: how agents propose actions, how conflicts are resolved, how events are generated, how state is updated, and how indices are maintained. It is the main loop of the simulation engine.

**Core contract:** Each tick transforms a `WorldState` into a new `WorldState` by producing zero or more `Event` objects. The event log is append-only. World state is a materialized view derived from the event stream.

**NOT in scope:**
- How agents decide what to do (see decision-engine.md)
- How pacing variables update (see pacing-physics.md)
- Character definitions and scenario setup (see dinner-party-config.md)
- Post-hoc metric computation (see tension-pipeline.md, irony-and-beliefs.md)

---

## 2. Simulation Loop Overview

```
SimulationLoop:
    world = initialize_world(scenario_config)
    tick_id = 0

    while not termination_condition(world, tick_id):
        events = execute_tick(world, tick_id)
        for event in events:
            append_to_event_log(event)
            apply_deltas(world, event.deltas)
            update_indices(event)
        if should_snapshot(world.total_event_count):
            save_snapshot(world, tick_id)
        tick_id += 1

    return event_log
```

### 2.1 Termination Conditions

The simulation ends when ANY of these are true:

```python
def termination_condition(world: WorldState, tick_id: int) -> bool:
    # Hard time limit: dinner party is ~2-3 hours = 120-180 sim minutes
    if world.sim_time >= MAX_SIM_TIME:  # default: 150.0 minutes
        return True

    # Hard tick limit: safety valve
    if tick_id >= MAX_TICKS:  # default: 300
        return True

    # Party breaks up: fewer than 2 agents remaining at any location
    active_agents = [a for a in world.agents.values() if a.location != "departed"]
    if len(active_agents) < 2:
        return True

    # All agents in cooldown with no pending actions: stalemate
    if all(a.pacing.recovery_timer > 0 for a in active_agents):
        if all(a.pacing.stress < 0.2 for a in active_agents):
            return True  # everyone's calm and cooling down — evening is winding down

    return False
```

### 2.2 Time Progression

Each tick advances sim_time by a variable amount based on the dominant event type in the tick:

```python
SIM_TIME_PER_TICK = {
    "calm":      0.5,   # minutes — small talk, eating
    "moderate":  0.75,  # minutes — meaningful conversation, movement
    "dramatic":  1.0,   # minutes — confrontation, revelation
    "catastrophe": 1.5, # minutes — breakdown, explosion (time dilates for drama)
}

def tick_duration(events: list[Event]) -> float:
    """The most dramatic event in the tick determines the tick's time step."""
    if any(e.type == EventType.CATASTROPHE for e in events):
        return SIM_TIME_PER_TICK["catastrophe"]
    if any(e.type in (EventType.CONFLICT, EventType.REVEAL) for e in events):
        return SIM_TIME_PER_TICK["dramatic"]
    if any(e.type in (EventType.CONFIDE, EventType.LIE, EventType.SOCIAL_MOVE) for e in events):
        return SIM_TIME_PER_TICK["moderate"]
    return SIM_TIME_PER_TICK["calm"]
```

This creates a natural "time dilation" effect: dramatic moments take more narrative time (and feel more significant in the event log) while quiet moments compress.

---

## 3. Tick Lifecycle (execute_tick)

A single tick has 6 phases, executed in strict order.

```
Phase 1: Perception     — Each agent builds their perceived state
Phase 2: Catastrophe    — Check involuntary breaks before voluntary action
Phase 3: Proposal       — Each non-catastrophe agent proposes 0-N candidate actions
Phase 4: Resolution     — Resolve conflicts when multiple agents act simultaneously
Phase 5: Event Gen      — Convert resolved actions into Event objects
Phase 6: State Update   — Apply deltas, update pacing, update indices, maybe snapshot
```

### Phase 1: Perception

Each agent constructs their **perceived world state** — what they think is happening, filtered through their beliefs and location.

```python
def build_perception(agent: AgentState, world: WorldState) -> PerceivedState:
    """
    An agent's perception is the world state filtered by:
    1. Location: can only see agents at same location (or overhear adjacent)
    2. Beliefs: beliefs may be wrong (false beliefs replace ground truth)
    3. Emotional state: high stress distorts perception (optional future feature)
    """
    perceived = PerceivedState()

    # Agents I can see
    my_location = world.locations[agent.location]
    perceived.visible_agents = [
        a for a in world.agents.values()
        if a.location == agent.location and a.id != agent.id
    ]
    perceived.overhearable_locations = my_location.overhear_from

    # Relationships (my view of them)
    perceived.relationships = agent.relationships

    # Beliefs about secrets (may be wrong!)
    perceived.beliefs = agent.beliefs

    # My own state (always accurate)
    perceived.self_state = agent

    # Other agents' emotional states: visible only if composure < masking threshold
    perceived.visible_emotions = {}
    for other in perceived.visible_agents:
        if other.pacing.composure < COMPOSURE_MIN_FOR_MASKING:
            # Their mask has slipped — I can see their true emotional state
            perceived.visible_emotions[other.id] = other.emotional_state
        else:
            # They're masking — I see a neutral/pleasant facade
            perceived.visible_emotions[other.id] = {"pleasant": 0.5}

    return perceived
```

**Key design choice:** Agents act on *perceived* state, not true state. An agent with a false belief (BeliefState.BELIEVES_FALSE) will make decisions based on that false belief. This is how dramatic irony enters the simulation — the audience (metrics pipeline) can see the truth, but the agent cannot.

### Phase 2: Catastrophe Check

Before any voluntary action, check if any agent triggers a catastrophe. Catastrophes preempt voluntary action — an agent who catastrophes this tick does NOT propose voluntary actions.

```python
def check_catastrophes(world: WorldState) -> list[tuple[str, str]]:
    """
    Returns list of (agent_id, catastrophe_subtype) for agents who catastrophe this tick.
    Maximum 2 per tick (see pacing-physics.md Section 13.1).
    """
    catastrophes = []
    for agent in world.agents.values():
        if check_catastrophe(agent):
            subtype = select_catastrophe_type(agent)
            catastrophes.append((agent.id, subtype))

    # Cap at 2: sort by catastrophe_potential (highest first), take top 2
    catastrophes.sort(
        key=lambda c: catastrophe_potential(world.agents[c[0]]),
        reverse=True
    )
    return catastrophes[:2]
```

Agents who catastrophe are added to the `catastrophe_agents` set and excluded from Phase 3.

### Phase 3: Proposal

Each non-catastrophe agent proposes 0-N candidate actions based on their perceived state. The decision engine (see decision-engine.md) scores all available actions and returns the top candidate.

```python
def propose_actions(world: WorldState, catastrophe_agents: set[str]) -> dict[str, Action]:
    """
    Each agent proposes their best action for this tick.
    Returns: {agent_id: Action} for agents who want to act.
    Agents with recovery_timer > 0 can only propose non-dramatic actions.
    """
    proposals = {}

    for agent in world.agents.values():
        if agent.id in catastrophe_agents:
            continue  # catastrophe agents don't propose
        if agent.location == "departed":
            continue  # agents who left the party don't act

        perception = build_perception(agent, world)
        action = decision_engine.select_action(agent, perception, world)

        if action is not None:
            proposals[agent.id] = action

    return proposals
```

#### The Action Type

```python
@dataclass
class Action:
    """A proposed action from an agent. Not yet an Event."""
    agent_id: str                    # who's proposing this
    action_type: EventType           # what kind of action
    target_agents: list[str]         # who it's directed at
    location_id: str                 # where it happens
    utility_score: float             # how much the agent wants to do this
    content: str                     # human-readable description of what they'd do
    dialogue: str | None = None      # what they'd say, if applicable
    metadata: dict[str, Any] = field(default_factory=dict)  # action-specific context (secret_id, topic, etc.)
    requires_target_available: bool = True  # can this be blocked by target being busy?

    # Metadata for conflict resolution
    priority_class: int = 1          # 1=normal, 2=urgent, 3=reactive (responding to prior event)
    is_dramatic: bool = False        # does this cost dramatic budget?
```

#### Action Availability by State

| Agent State | Available Action Types |
|---|---|
| `recovery_timer > 0` | CHAT, OBSERVE, SOCIAL_MOVE, INTERNAL, PHYSICAL (non-dramatic only) |
| `dramatic_budget < BUDGET_MINIMUM` | CHAT, OBSERVE, SOCIAL_MOVE, INTERNAL, PHYSICAL (non-dramatic only) |
| `location == "departed"` | None (agent has left) |
| Normal state | All action types |

### Phase 4: Conflict Resolution

When multiple agents propose actions for the same tick, some may conflict. Resolution determines which actions succeed and in what order.

#### 4.1 Conflict Types

```python
def detect_conflicts(proposals: dict[str, Action]) -> list[Conflict]:
    """Identify conflicting action pairs."""
    conflicts = []

    for (a_id, a_action), (b_id, b_action) in combinations(proposals.items(), 2):
        # Target contention: two agents trying to engage the same target
        if (a_action.requires_target_available
                and b_action.requires_target_available
                and set(a_action.target_agents) & set(b_action.target_agents)):
            conflicts.append(Conflict(
                type="target_contention",
                agents=[a_id, b_id],
                contested_resource=list(set(a_action.target_agents) & set(b_action.target_agents)),
            ))

        # Location capacity: both trying to move to a full location
        if (a_action.action_type == EventType.SOCIAL_MOVE
                and b_action.action_type == EventType.SOCIAL_MOVE
                and get_destination(a_action) == get_destination(b_action)):
            dest = get_destination(a_action)
            if location_near_capacity(dest):
                conflicts.append(Conflict(
                    type="location_capacity",
                    agents=[a_id, b_id],
                    contested_resource=[dest],
                ))

        # Incompatible actions: one agent confiding while another is confronting the same person
        if is_incompatible_pair(a_action, b_action):
            conflicts.append(Conflict(
                type="incompatible_actions",
                agents=[a_id, b_id],
                contested_resource=[],
            ))

    return conflicts
```

#### 4.2 Resolution Algorithm

```python
def resolve_conflicts(proposals: dict[str, Action],
                       conflicts: list[Conflict],
                       rng: Random) -> list[Action]:
    """
    Resolve conflicts and return ordered list of actions that execute this tick.

    Priority rules:
    1. Higher priority_class wins (reactive > urgent > normal)
    2. Within same class: higher utility_score wins
    3. Ties broken by randomness (seeded for reproducibility)

    Losers of target_contention:
    - Action is downgraded, not cancelled
    - CONFIDE → becomes INTERNAL (agent wanted to confide but target was busy)
    - CONFLICT → becomes INTERNAL (agent wanted to confront but target was engaged)
    - CHAT → becomes OBSERVE (agent watches instead of talking)
    """
    resolved = []
    blocked_agents = set()

    # Sort conflicts by severity (incompatible > target > location)
    for conflict in sorted(conflicts, key=lambda c: conflict_severity(c), reverse=True):
        if any(a in blocked_agents for a in conflict.agents):
            continue  # already resolved

        # Determine winner
        candidates = [(a, proposals[a]) for a in conflict.agents if a not in blocked_agents]
        if len(candidates) < 2:
            continue

        # Sort by priority_class DESC, then utility_score DESC, then random tiebreak
        candidates.sort(key=lambda c: (c[1].priority_class, c[1].utility_score, rng.random()), reverse=True)

        winner_id, winner_action = candidates[0]

        # Losers get downgraded
        for loser_id, loser_action in candidates[1:]:
            downgraded = downgrade_action(loser_action)
            proposals[loser_id] = downgraded

    # Build final ordered list: catastrophes first, then by priority, then by utility
    for agent_id, action in proposals.items():
        resolved.append(action)

    resolved.sort(key=lambda a: (a.priority_class, a.utility_score), reverse=True)
    return resolved


def downgrade_action(action: Action) -> Action:
    """Convert a blocked action into a weaker alternative."""
    downgrades = {
        EventType.CONFIDE:  EventType.INTERNAL,    # wanted to confide, couldn't
        EventType.CONFLICT: EventType.INTERNAL,    # wanted to confront, couldn't
        EventType.REVEAL:   EventType.INTERNAL,    # wanted to reveal, couldn't
        EventType.CHAT:     EventType.OBSERVE,     # wanted to talk, target busy
        EventType.LIE:      EventType.INTERNAL,    # wanted to lie, couldn't
    }
    new_type = downgrades.get(action.action_type, EventType.INTERNAL)
    return Action(
        agent_id=action.agent_id,
        action_type=new_type,
        target_agents=[],
        location_id=action.location_id,
        utility_score=action.utility_score * 0.5,
        content=f"[Blocked] {action.content}",
        is_dramatic=False,
    )
```

#### 4.3 Compatible Actions (No Conflict)

Many simultaneous actions are compatible and all execute:
- Two agents CHATting with different targets: both execute
- Agent A CHATs at table while Agent B CONFIDEs on balcony: both execute (different locations)
- Agent A OBSERVEs while Agent B does anything: OBSERVE is passive, always compatible
- Agent A has INTERNAL thought while anything else happens: INTERNAL is invisible, always compatible
- Agent A POURs a drink while Agent B CHATs: physical actions are usually compatible with social ones

### Phase 5: Event Generation

Convert resolved actions into Event objects. This is where actions become part of the permanent event log.

```python
def generate_events(resolved_actions: list[Action],
                     catastrophes: list[tuple[str, str]],
                     world: WorldState,
                     tick_id: int) -> list[Event]:
    """
    Convert resolved actions and catastrophes into Event objects.
    Assigns event IDs, sim_time, order_in_tick, causal links, and deltas.
    """
    events = []
    order = 0

    # Catastrophe events first (they are involuntary and take priority)
    for agent_id, subtype in catastrophes:
        agent = world.agents[agent_id]
        event = generate_catastrophe_event(agent, subtype, world, tick_id, order)
        events.append(event)
        order += 1

    # Then resolved voluntary actions
    for action in resolved_actions:
        event = action_to_event(action, world, tick_id, order)
        events.append(event)
        order += 1

    return events


def action_to_event(action: Action, world: WorldState,
                     tick_id: int, order: int) -> Event:
    """Convert a resolved Action into an Event with deltas."""
    event_id = generate_event_id()

    # Generate deltas based on action type
    deltas = generate_deltas_for_action(action, world)

    # Generate causal links
    causal_links = find_causal_links(action, world)

    return Event(
        id=event_id,
        sim_time=world.sim_time,  # updated after all events generated
        tick_id=tick_id,
        order_in_tick=order,
        type=action.action_type,
        source_agent=action.agent_id,
        target_agents=action.target_agents,
        location_id=action.location_id,
        causal_links=causal_links,
        deltas=deltas,
        description=action.content,
        dialogue=action.dialogue,
    )
```

#### 5.1 Delta Generation by Event Type

```python
def generate_deltas_for_action(action: Action, world: WorldState) -> list[StateDelta]:
    """
    Generate state deltas based on the action type.
    This is the bridge between the decision engine and the event log.
    """
    deltas = []
    agent = world.agents[action.agent_id]

    match action.action_type:

        case EventType.CHAT:
            # Small relationship shifts based on content
            for target_id in action.target_agents:
                deltas.append(StateDelta(
                    kind=DeltaKind.RELATIONSHIP,
                    agent=target_id,
                    agent_b=action.agent_id,
                    attribute="affection",
                    op=DeltaOp.ADD,
                    value=random_in_range(0.02, 0.08),  # mild positive
                    reason_code="PLEASANT_CONVERSATION",
                    reason_display=f"Chatting with {agent.name}",
                ))

        case EventType.OBSERVE:
            # Belief changes based on what was observed
            observed_info = determine_observable_info(agent, world)
            for info in observed_info:
                deltas.extend(info_to_belief_deltas(agent, info))

        case EventType.SOCIAL_MOVE:
            destination = get_destination(action)
            deltas.append(StateDelta(
                kind=DeltaKind.AGENT_LOCATION,
                agent=action.agent_id,
                attribute="",
                op=DeltaOp.SET,
                value=destination,
                reason_code="LOCATION_CHANGE",
                reason_display=f"{agent.name} moves to {destination}",
            ))
            # Minor budget cost for leaving a social situation
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="dramatic_budget",
                op=DeltaOp.ADD,
                value=-0.05,
                reason_code="SOCIAL_MOVE_COST",
                reason_display="Cost of excusing oneself",
            ))

        case EventType.REVEAL:
            # Secret becomes known to targets
            secret_id = action.metadata.get("secret_id")
            for target_id in action.target_agents:
                deltas.append(StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent=target_id,
                    attribute=secret_id,
                    op=DeltaOp.SET,
                    value=BeliefState.BELIEVES_TRUE.value,
                    reason_code="DIRECT_REVEAL",
                    reason_display=f"{agent.name} revealed the truth",
                ))
            # Budget cost
            budget_cost = BUDGET_COST_MAJOR if is_major_reveal(action) else BUDGET_COST_MINOR
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="dramatic_budget",
                op=DeltaOp.ADD,
                value=-budget_cost,
                reason_code="DRAMATIC_ACTION_COST",
                reason_display="Revealing information drains dramatic budget",
            ))

        case EventType.CONFLICT:
            # Trust drops for both parties
            for target_id in action.target_agents:
                deltas.append(StateDelta(
                    kind=DeltaKind.RELATIONSHIP,
                    agent=action.agent_id,
                    agent_b=target_id,
                    attribute="trust",
                    op=DeltaOp.ADD,
                    value=random_in_range(-0.2, -0.4),
                    reason_code="CONFRONTATION",
                    reason_display=f"Confronted {world.agents[target_id].name}",
                ))
                deltas.append(StateDelta(
                    kind=DeltaKind.RELATIONSHIP,
                    agent=target_id,
                    agent_b=action.agent_id,
                    attribute="trust",
                    op=DeltaOp.ADD,
                    value=random_in_range(-0.2, -0.3),
                    reason_code="ACCUSED_BY",
                    reason_display=f"Accused by {agent.name}",
                ))
            # Emotional changes
            deltas.append(StateDelta(
                kind=DeltaKind.AGENT_EMOTION,
                agent=action.agent_id,
                attribute="anger",
                op=DeltaOp.ADD,
                value=random_in_range(0.1, 0.3),
                reason_code="CONFRONTATION_INITIATED",
                reason_display="Anger from confrontation",
            ))
            # Stress for targets
            for target_id in action.target_agents:
                deltas.append(StateDelta(
                    kind=DeltaKind.PACING,
                    agent=target_id,
                    attribute="stress",
                    op=DeltaOp.ADD,
                    value=STRESS_GAIN_DIRECT,
                    reason_code="CONFLICT_EXPOSURE",
                    reason_display="Stress from being confronted",
                ))
            # Budget cost
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="dramatic_budget",
                op=DeltaOp.ADD,
                value=-BUDGET_COST_MAJOR,
                reason_code="DRAMATIC_ACTION_COST",
                reason_display="Confrontation drains dramatic budget",
            ))

        case EventType.INTERNAL:
            # Only self-directed emotional/commitment changes
            # Specific deltas depend on what the agent was thinking about
            pass  # Handled by decision engine's content generation

        case EventType.PHYSICAL:
            # Context-dependent: pouring wine, handing something, etc.
            if "drink" in action.content.lower():
                deltas.append(StateDelta(
                    kind=DeltaKind.AGENT_RESOURCE,
                    agent=action.agent_id,
                    attribute="alcohol_level",
                    op=DeltaOp.ADD,
                    value=0.15,
                    reason_code="ALCOHOL_CONSUMED",
                    reason_display=f"{agent.name} took a drink",
                ))

        case EventType.CONFIDE:
            # Secret shared with target, trust increases
            secret_id = action.metadata.get("secret_id")
            for target_id in action.target_agents:
                deltas.append(StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent=target_id,
                    attribute=secret_id,
                    op=DeltaOp.SET,
                    value=BeliefState.BELIEVES_TRUE.value,
                    reason_code="CONFIDED_SECRET",
                    reason_display=f"{agent.name} confided a secret",
                ))
                # Trust boost (bidirectional)
                deltas.append(StateDelta(
                    kind=DeltaKind.RELATIONSHIP,
                    agent=target_id,
                    agent_b=action.agent_id,
                    attribute="trust",
                    op=DeltaOp.ADD,
                    value=0.15,
                    reason_code="TRUST_THROUGH_VULNERABILITY",
                    reason_display="Vulnerability deepened trust",
                ))
                deltas.append(StateDelta(
                    kind=DeltaKind.RELATIONSHIP,
                    agent=action.agent_id,
                    agent_b=target_id,
                    attribute="trust",
                    op=DeltaOp.ADD,
                    value=0.10,
                    reason_code="CONFIDING_BOND",
                    reason_display="Confiding creates reciprocal trust",
                ))
            # Stress relief from sharing
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="stress",
                op=DeltaOp.ADD,
                value=-0.10,
                reason_code="STRESS_RELIEF_CONFIDING",
                reason_display="Sharing the burden reduced stress",
            ))
            # Budget cost
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="dramatic_budget",
                op=DeltaOp.ADD,
                value=-BUDGET_COST_MINOR,
                reason_code="DRAMATIC_ACTION_COST",
                reason_display="Confiding drains dramatic budget",
            ))

        case EventType.LIE:
            # Plant false belief in target
            secret_id = action.metadata.get("secret_id")
            for target_id in action.target_agents:
                deltas.append(StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent=target_id,
                    attribute=secret_id,
                    op=DeltaOp.SET,
                    value=BeliefState.BELIEVES_FALSE.value,
                    reason_code="DELIBERATE_MISDIRECTION",
                    reason_display=f"{agent.name} lied about {secret_id}",
                ))
                # Temporary trust boost (they believed the lie)
                deltas.append(StateDelta(
                    kind=DeltaKind.RELATIONSHIP,
                    agent=target_id,
                    agent_b=action.agent_id,
                    attribute="trust",
                    op=DeltaOp.ADD,
                    value=0.05,
                    reason_code="BELIEVED_EXPLANATION",
                    reason_display="Believed the explanation",
                ))
            # Liar gains stress and commitment
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="stress",
                op=DeltaOp.ADD,
                value=0.15,
                reason_code="LYING_STRESS",
                reason_display="Lying increases internal stress",
            ))
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="commitment",
                op=DeltaOp.ADD,
                value=0.20,
                reason_code="DEEPER_INTO_DECEPTION",
                reason_display="Each lie commits further to the facade",
            ))
            # Budget cost
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=action.agent_id,
                attribute="dramatic_budget",
                op=DeltaOp.ADD,
                value=-BUDGET_COST_MAJOR,
                reason_code="DRAMATIC_ACTION_COST",
                reason_display="Lying drains dramatic budget",
            ))

    return deltas
```

#### 5.2 Causal Link Generation

```python
def find_causal_links(action: Action, world: WorldState) -> list[str]:
    """
    Determine which prior events caused this action.
    Uses a priority-ordered search:
    1. Direct response events (was this agent targeted by a recent event?)
    2. Information dependency (did the agent learn something that triggered this?)
    3. Location dependency (did a move enable this interaction?)
    4. State dependency (did a relationship/emotion change influence this?)
    """
    links = []
    agent = world.agents[action.agent_id]

    # 1. Direct response: find the most recent event targeting this agent
    recent_targeting = find_recent_events_targeting(agent.id, max_ticks_back=3)
    if recent_targeting:
        links.append(recent_targeting[0].id)

    # 2. Information dependency: find the event that gave the agent relevant knowledge
    if action.action_type in (EventType.CONFLICT, EventType.REVEAL, EventType.LIE):
        knowledge_source = find_knowledge_source(agent, action)
        if knowledge_source:
            links.append(knowledge_source.id)

    # 3. Location dependency: if agent recently moved, link to the move
    if agent_recently_moved(agent.id, max_ticks_back=2):
        move_event = find_recent_move(agent.id)
        if move_event:
            links.append(move_event.id)

    # 4. State dependency: link to events that changed relationships affecting this action
    if action.target_agents:
        for target_id in action.target_agents:
            rel_change = find_recent_relationship_change(agent.id, target_id, max_ticks_back=5)
            if rel_change:
                links.append(rel_change.id)

    # Deduplicate and cap at 5
    links = list(dict.fromkeys(links))[:5]

    # Every event except the very first must have at least one causal link
    if not links and world.event_count > 0:
        # Fallback: link to the most recent event in the same location
        fallback = find_most_recent_event_at_location(action.location_id)
        if fallback:
            links.append(fallback.id)

    return links
```

#### 5.3 Witness and Overhear Events

After generating primary events, check if any non-participating agents should receive passive OBSERVE events.

```python
def generate_witness_events(primary_events: list[Event], world: WorldState,
                             tick_id: int, start_order: int) -> list[Event]:
    """
    Generate OBSERVE events for agents who witness or overhear primary events.
    Only for events that are visible/audible (not INTERNAL).
    """
    witness_events = []
    order = start_order

    for event in primary_events:
        if event.type == EventType.INTERNAL:
            continue  # internal thoughts are invisible

        location = world.locations[event.location_id]
        participants = {event.source_agent} | set(event.target_agents)

        # Agents at the same location who aren't participants
        witnesses = [
            a for a in world.agents.values()
            if a.location == event.location_id
            and a.id not in participants
        ]

        # Agents at overhearable locations (for loud events)
        if event.type in (EventType.CONFLICT, EventType.CATASTROPHE, EventType.REVEAL):
            # Check all agents at locations that can overhear this location
            for other_loc in world.locations.values():
                if event.location_id in other_loc.overhear_from:
                    overhearers = [
                        a for a in world.agents.values()
                        if a.location == other_loc.id
                        and a.id not in participants
                    ]
                    for agent in overhearers:
                        observe_event = generate_overhear_event(
                            agent, event, world, tick_id, order
                        )
                        witness_events.append(observe_event)
                        order += 1

        # Direct witnesses at same location
        for agent in witnesses:
            if event.type in (EventType.CONFLICT, EventType.CATASTROPHE, EventType.REVEAL):
                # Witnessing conflict: stress gain, possible belief change
                observe_event = generate_witness_event(
                    agent, event, world, tick_id, order
                )
                witness_events.append(observe_event)
                order += 1

    return witness_events
```

### Phase 6: State Update

After all events are generated, apply their effects to the world state.

```python
def apply_tick_updates(world: WorldState, events: list[Event], tick_id: int):
    """
    Apply all events from this tick to the world state.
    Order matters: deltas are applied in event order.
    """
    # Step 1: Apply deltas from each event
    for event in events:
        for delta in event.deltas:
            apply_delta(world, delta)

    # Step 2: Update pacing state for ALL agents (not just those who acted)
    for agent in world.agents.values():
        if agent.location == "departed":
            continue
        agent_events = [e for e in events if involves_agent(e, agent.id)]
        location = world.locations[agent.location]
        old_pacing = agent.pacing
        agent.pacing = update_pacing(agent, agent_events, location)

        # Generate pacing deltas for the event log (attached to last event involving agent)
        pacing_deltas = generate_pacing_deltas(agent.id, old_pacing, agent.pacing)
        if pacing_deltas and agent_events:
            agent_events[-1].deltas.extend(pacing_deltas)

    # Step 3: Update sim_time
    # (Hysteresis for trust repair is now handled inline in apply_delta)
    duration = tick_duration(events)
    world.sim_time += duration

    # Step 4: Update tick_id
    world.tick_id = tick_id

    # Step 5: Update indices
    for event in events:
        update_indices(event)

    # Step 6: Snapshot check (based on event count, not tick count — per doc3.md Decision #2)
    if should_snapshot(world.total_event_count):
        save_snapshot(world, tick_id)


def apply_delta(world: WorldState, delta: StateDelta):
    """Apply a single delta to the world state with clamping."""
    agent = world.agents[delta.agent]

    match delta.kind:
        case DeltaKind.AGENT_EMOTION:
            current = agent.emotional_state.get(delta.attribute, 0.0)
            if delta.op == DeltaOp.ADD:
                agent.emotional_state[delta.attribute] = clamp(current + delta.value, 0.0, 1.0)
            else:
                agent.emotional_state[delta.attribute] = clamp(delta.value, 0.0, 1.0)

        case DeltaKind.AGENT_RESOURCE:
            current = getattr(agent, delta.attribute, 0.0)
            if delta.op == DeltaOp.ADD:
                setattr(agent, delta.attribute, clamp(current + delta.value, 0.0, 1.0))
            else:
                setattr(agent, delta.attribute, clamp(delta.value, 0.0, 1.0))

        case DeltaKind.AGENT_LOCATION:
            agent.location = delta.value

        case DeltaKind.RELATIONSHIP:
            if delta.agent_b not in agent.relationships:
                agent.relationships[delta.agent_b] = {"trust": 0.0, "affection": 0.0, "obligation": 0.0}
            rel = agent.relationships[delta.agent_b]
            current = rel.get(delta.attribute, 0.0)
            effective_value = delta.value
            # Hysteresis: trust repair costs 3x (see pacing-physics.md Section 6.1)
            if delta.attribute == "trust" and delta.op == DeltaOp.ADD and delta.value > 0:
                effective_value = delta.value / 3.0
            if delta.op == DeltaOp.ADD:
                rel[delta.attribute] = clamp(current + effective_value, -1.0, 1.0)
            else:
                rel[delta.attribute] = clamp(effective_value, -1.0, 1.0)

        case DeltaKind.BELIEF:
            agent.beliefs[delta.attribute] = delta.value

        case DeltaKind.SECRET_STATE:
            secret = world.secrets.get(delta.attribute)
            if secret:
                # e.g., marking as publicly known
                setattr(secret, "state", delta.value)

        case DeltaKind.COMMITMENT:
            agent.commitments.append(delta.value)

        case DeltaKind.PACING:
            current = getattr(agent.pacing, delta.attribute, 0.0)
            if delta.op == DeltaOp.ADD:
                new_val = current + delta.value
            else:
                new_val = delta.value
            setattr(agent.pacing, delta.attribute, new_val)
            # Clamping handled by pacing update rules

        case DeltaKind.WORLD_RESOURCE:
            # World-level resource change
            pass  # MVP: no world resources beyond locations
```

#### 6.1 Index Updates

```python
def update_indices(event: Event):
    """
    Update CQRS read-layer indices (Decision #2).
    Called after each event is appended.
    """
    # event_id → Event
    event_index[event.id] = event

    # agent_id → List[event_id] (agent's timeline)
    agent_timeline[event.source_agent].append(event.id)
    for target in event.target_agents:
        agent_timeline[target].append(event.id)

    # location_id → List[event_id]
    location_events[event.location_id].append(event.id)

    # event_id → List[agent_id] (participants)
    event_participants[event.id] = [event.source_agent] + event.target_agents

    # secret_id → List[event_id] (events touching this secret)
    for delta in event.deltas:
        if delta.kind in (DeltaKind.BELIEF, DeltaKind.SECRET_STATE):
            secret_events[delta.attribute].append(event.id)

    # (agent_a, agent_b) → List[event_id] (shared interaction history)
    for target in event.target_agents:
        pair = tuple(sorted([event.source_agent, target]))
        interaction_history[pair].append(event.id)

    # Bidirectional causal graph (for hover neighborhoods, Decision #15)
    for cause_id in event.causal_links:
        forward_links[cause_id].append(event.id)
    backward_links[event.id] = event.causal_links
```

#### 6.2 Snapshot Policy

```python
def should_snapshot(total_event_count: int) -> bool:
    """
    Snapshot every 20 events OR every 5 sim-minutes (Decision #2).
    """
    events_since_last = total_event_count - last_snapshot_event_count
    return events_since_last >= 20

def save_snapshot(world: WorldState, tick_id: int):
    """Deep-copy the world state and store it keyed by tick_id."""
    snapshot = deep_copy(world)
    snapshots[tick_id] = snapshot
    last_snapshot_tick = tick_id
```

---

## 4. Putting It All Together: execute_tick

```python
def execute_tick(world: WorldState, tick_id: int, rng: Random) -> list[Event]:
    """
    The complete tick lifecycle. Returns all events generated this tick.
    """
    all_events = []

    # Phase 1: Perception (implicit — built on demand in Phase 3)

    # Phase 2: Catastrophe check
    catastrophes = check_catastrophes(world)
    catastrophe_agents = {c[0] for c in catastrophes}

    # Generate catastrophe events
    order = 0
    for agent_id, subtype in catastrophes:
        agent = world.agents[agent_id]
        event = generate_catastrophe_event(agent, subtype, world, tick_id, order)
        all_events.append(event)
        order += 1

    # Phase 3: Action proposal (excluding catastrophe agents)
    proposals = propose_actions(world, catastrophe_agents)

    # Phase 4: Conflict resolution
    conflicts = detect_conflicts(proposals)
    resolved_actions = resolve_conflicts(proposals, conflicts, rng)

    # Phase 5: Event generation
    for action in resolved_actions:
        event = action_to_event(action, world, tick_id, order)
        all_events.append(event)
        order += 1

    # Phase 5b: Witness/overhear events
    witness_events = generate_witness_events(all_events, world, tick_id, order)
    all_events.extend(witness_events)

    # Phase 6: State update (applied by caller in main loop)
    # This is done OUTSIDE execute_tick so the caller controls it

    return all_events
```

---

## 5. Worked Examples

### 5.1 Calm Tick (Small Talk at the Table)

**World state at tick 15:**
- All 6 agents at dining_table (privacy=0.1)
- No agent has stress > 0.3
- No agent has recovery_timer > 0
- Appetizers are being served

**Phase 2: Catastrophe check**
```
No agent passes catastrophe check.
(Highest potential: Marcus at 0.05 * 0.1^2 + 0 = 0.0005 — nowhere near 0.35)
catastrophes = []
```

**Phase 3: Proposals**
```
Thorne → CHAT with Elena (utility: 0.72) — "Compliment her gallery opening"
Elena  → CHAT with Diana (utility: 0.65) — "Ask about the new restaurant"
Marcus → CHAT with Victor (utility: 0.58) — "Discuss the market"
Diana  → PHYSICAL (utility: 0.45) — "Pour more wine for the table"
Lydia  → OBSERVE (utility: 0.40) — "Watch the room dynamics"
Victor → CHAT with Thorne (utility: 0.55) — "Bring up the partnership deal"
```

**Phase 4: Conflict resolution**
```
Check: Victor wants to CHAT with Thorne, but Thorne is targeting Elena.
→ No conflict! Thorne is talking TO Elena, not FROM Elena. Victor can still
  address Thorne. Multiple incoming CHATs to the same person are fine.

Check: No target contention. No location conflicts.
→ All proposals are compatible. All execute.
```

**Phase 5: Events generated**
```
evt_0031: CHAT (Thorne → Elena) at dining_table
  deltas: [relationship: Elena→Thorne affection +0.05]
  causal_links: [evt_0028]  (Thorne sat next to Elena earlier)

evt_0032: CHAT (Elena → Diana) at dining_table
  deltas: [relationship: Diana→Elena affection +0.03]
  causal_links: [evt_0029]

evt_0033: CHAT (Marcus → Victor) at dining_table
  deltas: [relationship: Victor→Marcus affection +0.04]
  causal_links: [evt_0027]

evt_0034: PHYSICAL (Diana pours wine) at dining_table
  deltas: [agent_resource: Diana alcohol_level +0.15,
           agent_resource: Thorne alcohol_level +0.15,
           agent_resource: Elena alcohol_level +0.15]
  causal_links: [evt_0030]

evt_0035: OBSERVE (Lydia watches room) at dining_table
  deltas: []  (nothing notable to observe yet)
  causal_links: [evt_0031]  (she's watching Thorne and Elena chat)

evt_0036: CHAT (Victor → Thorne) at dining_table
  deltas: [relationship: Thorne→Victor affection +0.03]
  causal_links: [evt_0033]  (following up on Marcus's market talk)
```

**Phase 6: State update**
```
- All deltas applied (small relationship boosts, alcohol levels up)
- Pacing updates: all agents get budget recharge (+0.08 each, no private bonus)
- Stress decay for any above 0: minimal
- Composure hit from alcohol for those who drank (Thorne, Elena, Diana: -0.06 each)
- sim_time += 0.5 (calm tick)
- No snapshot (total_event_count not yet at 20-event boundary)
```

**Total: 6 events, 0 conflicts, 0 catastrophes. Calm evening so far.**

### 5.2 Contested Tick (Two Agents Trying to Confide in the Same Person)

**World state at tick 44:**
- Dining table: Thorne, Victor, Lydia
- Kitchen: Marcus, Elena (both went to "get more wine")
- Balcony: Diana (stepped out for air)
- Marcus and Elena are alone in the kitchen — both want to confide in the other about different secrets

**Phase 2: Catastrophe check**
```
No catastrophes. Highest potential: Thorne at 0.30 * 0.35^2 + 2*0.03 = 0.097 — below threshold.
```

**Phase 3: Proposals**
```
Marcus → CONFIDE to Elena (utility: 0.85) — wants to confide about the embezzlement
Elena  → CONFIDE to Marcus (utility: 0.82) — wants to confide about her feelings for him
Thorne → CHAT with Lydia (utility: 0.50) — small talk at table
Victor → OBSERVE (utility: 0.45) — watching Thorne's body language
Lydia  → CHAT with Thorne (utility: 0.48) — responding to Thorne
Diana  → INTERNAL (utility: 0.40) — thinking about what she learned from Elena earlier
```

**Phase 4: Conflict resolution**
```
Conflict detected: Marcus CONFIDE → Elena AND Elena CONFIDE → Marcus
  Both target the same person, and CONFIDE requires target_available.

Resolution:
  Marcus: priority_class=1, utility=0.85
  Elena:  priority_class=1, utility=0.82
  Winner: Marcus (higher utility)

Elena's action downgraded:
  CONFIDE → INTERNAL ("Elena wanted to confide in Marcus but he spoke first")
  Elena still receives Marcus's confession (as target of his CONFIDE)
```

**Phase 5: Events generated**
```
evt_0088: CONFIDE (Marcus → Elena) at kitchen
  deltas: [
    belief: Elena learns secret_embezzle_01 → believes_true,
    relationship: Elena→Marcus trust +0.15 (vulnerability),
    relationship: Marcus→Elena trust +0.10 (confiding bond),
    pacing: Marcus stress -0.10 (relief),
    pacing: Marcus dramatic_budget -0.15 (CONFIDE cost),
  ]
  dialogue: "Elena, I need to tell you something. I've been... moving money from the firm."
  causal_links: [evt_0072, evt_0080]

evt_0089: INTERNAL (Elena — blocked confide) at kitchen
  deltas: [
    agent_emotion: Elena hope +0.2 (Marcus trusts her enough to confide),
    commitment: Elena "protect_marcus" (internally commits to keeping his secret),
  ]
  description: "Elena listens to Marcus's confession, her own confession dying on her lips."
  causal_links: [evt_0088]  (directly caused by Marcus speaking first)

evt_0090: CHAT (Thorne → Lydia) at dining_table
  deltas: [relationship: Lydia→Thorne affection +0.04]
  causal_links: [evt_0085]

evt_0091: OBSERVE (Victor watches Thorne) at dining_table
  deltas: [agent_emotion: Victor suspicion +0.1]
  description: "Victor notices Thorne keeps glancing toward the kitchen."
  causal_links: [evt_0090]

evt_0092: CHAT (Lydia → Thorne) at dining_table
  deltas: [relationship: Thorne→Lydia affection +0.03]
  causal_links: [evt_0090]

evt_0093: INTERNAL (Diana thinks on balcony) at balcony
  deltas: [agent_emotion: Diana suspicion +0.15]
  description: "Diana replays Elena's confession in her mind. Something doesn't add up."
  causal_links: [evt_0034]  (Elena's earlier confide)
```

**Key observation:** The conflict resolution turned what would have been two competing CONFIDE events into a richer sequence: Marcus confides, Elena's intention is blocked and becomes an internal moment, which actually creates a more interesting narrative beat (she chose to listen instead of speak).

### 5.3 Catastrophe Tick (Stress x Commitment Threshold Crossed)

**World state at tick 78:**
- Dining table: Thorne, Elena, Marcus, Victor, Lydia (Diana is on balcony)
- The evening has been escalating. Thorne has confronted Marcus twice. Elena is stressed.
- Victor has been lying about the business deal.

**Key agent states:**
```
Victor:
  stress: 0.72, commitment: 0.78, composure: 0.28, suppression_count: 12
  catastrophe_potential = 0.72 * 0.78^2 + (12 * 0.03) = 0.438 + 0.36 = 0.798
  composure (0.28) < gate (0.30) ✓
  recovery_timer: 0 ✓
  → CATASTROPHE FIRES

Thorne:
  stress: 0.55, commitment: 0.60, composure: 0.42
  catastrophe_potential = 0.55 * 0.60^2 + (5 * 0.03) = 0.198 + 0.15 = 0.348
  composure (0.42) > gate (0.30) ✗
  → No catastrophe (composure still holds)

Elena:
  stress: 0.48, commitment: 0.45, composure: 0.35
  catastrophe_potential = 0.48 * 0.45^2 + (3 * 0.03) = 0.097 + 0.09 = 0.187
  → No catastrophe (potential below threshold)
```

**Phase 2: Catastrophe check**
```
Victor triggers catastrophe!
  Dominant flaw: ambition. Peak emotion: anger (0.6).
  → catastrophe_subtype: "explosion"

catastrophes = [("victor", "explosion")]
catastrophe_agents = {"victor"}
```

**Phase 3: Proposals (excluding Victor)**
```
Thorne → CONFLICT with Victor (utility: 0.78) — "Demand answers about the deal"
  → BUT Victor is having a catastrophe this tick. Target unavailable.
  → Downgraded: CONFLICT → INTERNAL ("Thorne was about to confront Victor when...")

Elena → SOCIAL_MOVE to balcony (utility: 0.65) — "Escape the tension"
Marcus → CHAT with Lydia (utility: 0.50) — "Try to appear calm"
Lydia → OBSERVE (utility: 0.55) — "Watch Victor carefully"
Diana → [on balcony, not involved in proposals targeting table agents]
  → INTERNAL (utility: 0.30) — "Enjoying the quiet"
```

**Note:** Thorne's proposed CONFLICT targets Victor, but Victor is catastrophe-ing. Since Victor can't be a willing participant in a conflict HE didn't initiate, Thorne's action is downgraded to INTERNAL. This creates a narrative beat: Thorne was about to confront Victor, but Victor broke first.

**Phase 5: Events generated**
```
evt_0156: CATASTROPHE (Victor → [Thorne, Marcus]) at dining_table
  catastrophe_subtype: "explosion"
  deltas: [
    -- Victor's aftermath --
    pacing: Victor stress SET 0.36 (halved from 0.72),
    pacing: Victor composure SET 0.30 (catastrophe reset),
    pacing: Victor recovery_timer SET 8,
    pacing: Victor suppression_count SET 0,
    pacing: Victor dramatic_budget ADD -0.50,
    pacing: Victor commitment ADD +0.10 → 0.88,
    agent_emotion: Victor anger SET 0.9,
    agent_emotion: Victor shame +0.3,

    -- Targets --
    pacing: Thorne stress ADD +0.12,
    pacing: Marcus stress ADD +0.12,
    agent_emotion: Thorne anger +0.2,
    agent_emotion: Marcus fear +0.4,

    -- Secret revealed in outburst --
    secret_state: secret_deal_01 → "publicly_known",
    belief: Thorne secret_deal_01 → believes_true,
    belief: Elena secret_deal_01 → believes_true,
    belief: Marcus secret_deal_01 → believes_true,
    belief: Lydia secret_deal_01 → believes_true,

    -- Relationship damage --
    relationship: Thorne→Victor trust ADD -0.4,
    relationship: Marcus→Victor trust ADD -0.3,
  ]
  description: "Victor slams both hands on the table, sending glasses trembling.
    'You want the truth? FINE. The deal was never going to close — I used the deposit
    to cover my own debts. Every single one of you would have done the same!'"
  dialogue: "\"You want the truth? FINE. The deal was never going to close...\""
  causal_links: [evt_0150, evt_0153, evt_0155]  (the pressure events)

evt_0157: INTERNAL (Thorne — blocked conflict) at dining_table
  deltas: [
    agent_emotion: Thorne anger +0.3,
    commitment: Thorne "destroy_victor" (new commitment),
    pacing: Thorne commitment ADD +0.15,
  ]
  description: "Thorne was about to confront Victor when Victor exploded first.
    Thorne's jaw tightens. Now he knows everything."
  causal_links: [evt_0156]

evt_0158: SOCIAL_MOVE (Elena → balcony) at dining_table
  deltas: [
    agent_location: Elena SET "balcony",
    pacing: Elena dramatic_budget ADD -0.05,
  ]
  description: "Elena pushes her chair back and hurries to the balcony, hands shaking."
  causal_links: [evt_0156]

evt_0159: CHAT (Marcus → Lydia) at dining_table
  deltas: [
    relationship: Lydia→Marcus affection +0.02,
  ]
  description: "Marcus tries to change the subject with Lydia, his voice unsteady."
  dialogue: "\"So, Lydia, how's the... the new project going?\""
  causal_links: [evt_0156]
```

**Witness events:**
```
evt_0160: OBSERVE (Lydia witnesses catastrophe) at dining_table
  deltas: [
    pacing: Lydia stress ADD +0.05,
    agent_emotion: Lydia fear +0.2,
  ]
  description: "Lydia freezes, glass halfway to her lips, as Victor erupts."
  causal_links: [evt_0156]

evt_0161: OBSERVE (Diana overhears from balcony) at balcony
  deltas: [
    pacing: Diana stress ADD +0.03,
    belief: Diana secret_deal_01 → suspects,  (overheard, not full clarity)
  ]
  description: "Diana hears shouting from inside. She catches fragments — something about
    a deal, debts, money."
  causal_links: [evt_0156]
```

**Phase 6: State update**
```
All deltas applied:
  - Victor: stress 0.36, composure 0.30, recovery_timer 8, suppression 0, budget ~0.0
  - Thorne: stress 0.67 (was 0.55, +0.12 from target), commitment 0.75
  - Marcus: stress 0.50 (was 0.38, +0.12), fear spiking
  - Elena: moved to balcony, stress 0.48
  - Lydia: stress 0.25, fear elevated
  - Diana: stress 0.18, suspects the deal secret

Pacing updates for all agents...
sim_time += 1.5 (catastrophe tick)

Index updates:
  - secret_deal_01 now indexed to evt_0156
  - Forward links from evt_0150, evt_0153, evt_0155 now include evt_0156
  - Victor's timeline includes evt_0156
  - dining_table events include evt_0156 through evt_0160
  - balcony events include evt_0158, evt_0161
```

**Aftermath analysis:** This single catastrophe tick:
- Revealed a secret to 4 agents simultaneously
- Damaged 2 relationships
- Created 3 new emotional shifts
- Triggered a location change (Elena fleeing)
- Generated a blocked-action narrative beat (Thorne's internal moment)
- Produced 7 events total (1 catastrophe + 4 voluntary + 2 witness)
- Advanced the evening by 1.5 sim-minutes (dramatic time dilation)

---

## 6. Reproducibility

### 6.1 Deterministic Seeding

The simulation must be fully reproducible given the same:
1. Initial world state (scenario config)
2. Random seed

```python
def run_simulation(config: ScenarioConfig, seed: int = 42) -> EventLog:
    rng = Random(seed)
    world = initialize_world(config)
    # ...all randomness flows from rng...
```

Every random decision uses the same `rng` instance:
- Action scoring noise
- Conflict resolution tiebreaking
- Delta value ranges (e.g., `random_in_range(0.02, 0.08)`)
- Catastrophe subtype selection (when multiple are equally valid)

### 6.2 Debug Mode

In debug mode, each tick emits a TickDebugRecord:

```python
@dataclass
class TickDebugRecord:
    tick_id: int
    perceptions: dict[str, PerceivedState]      # what each agent saw
    proposals: dict[str, Action]                 # what each agent wanted to do
    conflicts: list[Conflict]                    # detected conflicts
    resolutions: list[tuple[str, str, str]]      # (agent, original, resolved_to)
    catastrophes: list[tuple[str, str, float]]   # (agent, subtype, potential)
    events: list[Event]                          # final events
    pacing_before: dict[str, PacingState]        # pacing state before update
    pacing_after: dict[str, PacingState]         # pacing state after update
```

---

## 7. Performance Characteristics

For the dinner party MVP (6 agents, ~100-200 events, ~60-150 ticks):

| Phase | Complexity | Expected Time |
|---|---|---|
| Perception | O(A * A) where A = agents | < 1ms |
| Catastrophe check | O(A) | < 1ms |
| Proposal | O(A * actions) | < 10ms (depends on decision engine) |
| Conflict resolution | O(A^2) | < 1ms |
| Event generation | O(events * delta_count) | < 5ms |
| State update | O(events * delta_count) | < 5ms |
| Index update | O(events) | < 1ms |
| **Total per tick** | | **< 25ms** |
| **Full simulation** | ~100 ticks | **< 2.5s** |

This leaves ample headroom for decision engine complexity. Even if the decision engine takes 50ms per agent (300ms total), the full simulation completes in under 30 seconds.

---

## 8. Edge Cases

### 8.1 Empty Tick

If all agents are on cooldown (recovery_timer > 0) and no catastrophes fire, the tick produces zero events. The tick still advances sim_time (by the "calm" rate) and pacing states still update (stress decays, budget recharges, timer decrements). This represents "awkward silence."

Empty ticks are valid and expected after major dramatic moments. They give the pacing system time to recover.

### 8.2 All Agents at Same Location

When all 6 agents are at the dining table:
- Every CONFLICT or CATASTROPHE generates witness OBSERVE events for all non-participants (up to 4 witness events per dramatic event)
- Masking is strongly enforced (privacy=0.1)
- Event count per tick can be high (6 voluntary + several witness events)

### 8.3 Agent Alone

If an agent is alone at a location (e.g., bathroom):
- Only INTERNAL, PHYSICAL, and SOCIAL_MOVE actions are available
- No target_agents for any action
- No witness events generated
- Budget recharges at higher rate (private location bonus)
- Stress decays at higher rate (private location bonus)

### 8.4 First Tick

The first tick (tick_id=0) is special:
- No causal links exist yet (first events link to nothing, or to a synthetic "simulation_start" sentinel)
- Initial world state is the scenario config's starting state
- All agents are at their starting locations
- All pacing states are at their initial values

### 8.5 Last Tick

When termination fires:
- The current tick completes fully (no partial execution)
- A synthetic "simulation_end" event is generated with metadata about why the simulation stopped
- Final snapshot is saved regardless of the normal snapshot schedule

---

## 9. Relationship to Other Specs

| Spec | Relationship |
|---|---|
| **events.md** | Defines the Event, StateDelta, EventType etc. that this spec produces. |
| **decision-engine.md** | Provides the `select_action()` function called in Phase 3. |
| **pacing-physics.md** | Provides `check_catastrophe()`, `update_pacing()`, and all pacing constants. |
| **dinner-party-config.md** | Provides the initial WorldState for `initialize_world()`. |
| **agents.md** | Defines the AgentState, PacingState, GoalVector etc. read and modified each tick. |
| **world.md** | Defines Location, privacy, adjacency, overhear rules. |
| **tension-pipeline.md** | Consumes the event log AFTER simulation. No interaction during ticks. |
| **scene-segmentation.md** | Consumes the event log AFTER simulation. No interaction during ticks. |

---

## 10. Resolved Questions

1. **Should agents react within the same tick to a catastrophe?** **No.** Agents react next tick. Simpler and more realistic (shock response delay). Catastrophes resolve in Phase 2; other agents' proposals based on pre-catastrophe state play out as planned.

2. **Should downgraded actions keep their original causal links?** **Yes.** The intention was real, so causal links are preserved. A CONFIDE downgraded to INTERNAL still links back to the events that motivated the confide attempt.

3. **How many OBSERVE/witness events per tick is too many?** **No throttling for MVP.** Generate all witness events. With 6 agents this produces at most 4-5 witness events per dramatic action, which is acceptable.
