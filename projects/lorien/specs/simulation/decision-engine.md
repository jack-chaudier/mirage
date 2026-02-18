# Decision Engine Specification

> **Status:** Draft
> **Author:** sim-designer
> **Dependencies:** specs/schema/events.md (#1) — DONE, specs/schema/agents.md (#2) — DONE, specs/schema/world.md (#3) — DONE
> **Dependents:** specs/simulation/tick-loop.md (#5), specs/integration/data-flow.md (#17)
> **Doc3 Decisions:** #6 (vectorized feature scoring), #5 (pacing physics / social masking), #8 (finite proposition catalog)

---

## 1. Purpose

The decision engine is the "brain" of each agent. Every tick, it examines the agent's perceived state and selects the single best action to propose. The tick loop (tick-loop.md) calls `select_action()` once per agent per tick and uses the returned Action for conflict resolution and event generation.

**Core contract:** Given an agent's state, their perception of the world, and the world state, return the Action that this agent would choose — or None if the agent does nothing this tick.

**Design philosophy:** Agents should make decisions that are *psychologically plausible but dramatically interesting*. This means:
1. Mostly rational (goal-aligned) behavior...
2. ...distorted by character flaws (irrational biases)...
3. ...modulated by pacing state (budget, stress, composure, recovery)...
4. ...with noise to prevent deterministic loops.

**NOT in scope:**
- Tick lifecycle / execution order (see tick-loop.md)
- Pacing state update rules (see pacing-physics.md)
- Character definitions (see dinner-party-config.md, agents.md)
- Post-hoc metrics (see tension-pipeline.md)

---

## 2. The Action Space

### 2.1 Available Actions

At any tick, an agent can choose from the following action types. The specific *instances* depend on location, present agents, and known information.

| Action Type | Requires Target | Requires Location | Dramatic? | Description |
|---|---|---|---|---|
| CHAT | Yes (1+ agents at location) | Same location | No | Social maintenance, small talk, asking questions |
| OBSERVE | No | Any | No | Watch someone, scan the room, listen |
| SOCIAL_MOVE | No (destination) | Anywhere | No | Move to a different location |
| REVEAL | Yes (1+ agents at location) | Same location | Yes | Intentionally share true information |
| CONFLICT | Yes (1+ agents at location) | Same location | Yes | Confrontation, accusation, argument |
| INTERNAL | No | Any | No | Think, plan, process emotion |
| PHYSICAL | Optional | Same location | No* | Pour drink, hand object, slam door, etc. |
| CONFIDE | Yes (1 agent at location) | Same location | Yes | Privately share secret/feeling |
| LIE | Yes (1+ agents at location) | Same location | Yes | Deliberately plant false information |

*PHYSICAL can be dramatic (slam a door, flip a table) but is usually not.

### 2.2 Action Generation

For each tick, the engine generates all *valid* action instances, then scores them.

```python
def generate_candidate_actions(agent: AgentState, perception: PerceivedState,
                                world: WorldState) -> list[Action]:
    """
    Generate all valid action candidates for this agent this tick.
    Returns a list of potential actions to be scored.
    """
    candidates = []
    location = world.locations[agent.location]
    present_agents = perception.visible_agents

    # --- CHAT candidates: one per present agent, plus group chat ---
    for other in present_agents:
        candidates.append(Action(
            agent_id=agent.id,
            action_type=EventType.CHAT,
            target_agents=[other.id],
            location_id=agent.location,
            content=generate_chat_content(agent, other, perception),
            is_dramatic=False,
        ))
    if len(present_agents) >= 2:
        # Group chat option
        candidates.append(Action(
            agent_id=agent.id,
            action_type=EventType.CHAT,
            target_agents=[a.id for a in present_agents],
            location_id=agent.location,
            content=generate_group_chat_content(agent, present_agents, perception),
            is_dramatic=False,
        ))

    # --- OBSERVE candidates: one per present agent, plus general ---
    for other in present_agents:
        candidates.append(Action(
            agent_id=agent.id,
            action_type=EventType.OBSERVE,
            target_agents=[],
            location_id=agent.location,
            content=f"Observe {other.name}",
            is_dramatic=False,
        ))
    candidates.append(Action(
        agent_id=agent.id,
        action_type=EventType.OBSERVE,
        target_agents=[],
        location_id=agent.location,
        content="Scan the room",
        is_dramatic=False,
    ))

    # --- SOCIAL_MOVE candidates: one per adjacent location with capacity ---
    for adj_loc_id in location.adjacent:
        adj_loc = world.locations[adj_loc_id]
        agents_there = count_agents_at(world, adj_loc_id)
        if agents_there < adj_loc.capacity:
            candidates.append(Action(
                agent_id=agent.id,
                action_type=EventType.SOCIAL_MOVE,
                target_agents=[],
                location_id=agent.location,
                content=f"Move to {adj_loc.name}",
                is_dramatic=False,
            ))

    # --- INTERNAL candidate: always available ---
    candidates.append(Action(
        agent_id=agent.id,
        action_type=EventType.INTERNAL,
        target_agents=[],
        location_id=agent.location,
        content=generate_internal_thought(agent, perception),
        is_dramatic=False,
    ))

    # --- PHYSICAL candidates: context-dependent ---
    physical_options = generate_physical_options(agent, location, world)
    candidates.extend(physical_options)

    # --- Dramatic actions: only if budget and recovery allow ---
    if (agent.pacing.dramatic_budget >= BUDGET_MINIMUM_FOR_ACTION
            and agent.pacing.recovery_timer == 0):

        # REVEAL: for each secret the agent knows, to each present agent
        for secret_id, belief in agent.beliefs.items():
            if belief == BeliefState.BELIEVES_TRUE:
                secret = world.secrets[secret_id]
                for other in present_agents:
                    # Only reveal if the other doesn't already know
                    other_belief = world.agents[other.id].beliefs.get(secret_id)
                    if other_belief != BeliefState.BELIEVES_TRUE:
                        candidates.append(Action(
                            agent_id=agent.id,
                            action_type=EventType.REVEAL,
                            target_agents=[other.id],
                            location_id=agent.location,
                            content=f"Reveal {secret.description} to {other.name}",
                            content_metadata={"secret_id": secret_id},
                            is_dramatic=True,
                        ))

        # CONFLICT: for each present agent the agent has grievance with
        for other in present_agents:
            rel = agent.relationships.get(other.id)
            if rel and (rel.trust < 0.0 or agent.emotional_state.get("anger", 0) > 0.3):
                candidates.append(Action(
                    agent_id=agent.id,
                    action_type=EventType.CONFLICT,
                    target_agents=[other.id],
                    location_id=agent.location,
                    content=f"Confront {other.name}",
                    is_dramatic=True,
                ))

        # CONFIDE: for each present agent with high trust (1-on-1 preferred)
        for other in present_agents:
            rel = agent.relationships.get(other.id)
            if rel and rel.trust > 0.4:
                for secret_id, belief in agent.beliefs.items():
                    if belief == BeliefState.BELIEVES_TRUE:
                        other_belief = world.agents[other.id].beliefs.get(secret_id)
                        if other_belief != BeliefState.BELIEVES_TRUE:
                            candidates.append(Action(
                                agent_id=agent.id,
                                action_type=EventType.CONFIDE,
                                target_agents=[other.id],
                                location_id=agent.location,
                                content=f"Confide about {secret_id} to {other.name}",
                                content_metadata={"secret_id": secret_id},
                                is_dramatic=True,
                                requires_target_available=True,
                            ))

        # LIE: for each present agent, about each secret
        for other in present_agents:
            for secret_id, belief in agent.beliefs.items():
                if belief == BeliefState.BELIEVES_TRUE:
                    secret = world.secrets[secret_id]
                    # Only lie if: agent wants to protect this secret
                    # AND target might be asking/curious about it
                    if should_consider_lying(agent, other, secret, perception):
                        candidates.append(Action(
                            agent_id=agent.id,
                            action_type=EventType.LIE,
                            target_agents=[other.id],
                            location_id=agent.location,
                            content=f"Lie to {other.name} about {secret_id}",
                            content_metadata={"secret_id": secret_id},
                            is_dramatic=True,
                        ))

    return candidates
```

### 2.3 Action Pruning

Before scoring, prune obviously invalid or irrelevant actions:

```python
def prune_candidates(candidates: list[Action], agent: AgentState) -> list[Action]:
    """Remove actions that are invalid given the current state."""
    pruned = []
    for action in candidates:
        # Can't target yourself
        if agent.id in action.target_agents:
            continue
        # Can't confide/reveal what you don't know
        if action.action_type in (EventType.REVEAL, EventType.CONFIDE, EventType.LIE):
            secret_id = action.content_metadata.get("secret_id")
            if not secret_id or agent.beliefs.get(secret_id) != BeliefState.BELIEVES_TRUE:
                continue
        # Can't move to current location
        if action.action_type == EventType.SOCIAL_MOVE:
            dest = get_destination(action)
            if dest == agent.location:
                continue
        pruned.append(action)
    return pruned
```

---

## 3. The Scoring Formula

Each candidate action receives a total score. The agent proposes the action with the highest score (subject to noise).

```
total_score = base_utility(goal_vector, action)
            + flaw_bias(flaws, action, context)
            + pacing_modifier(pacing_state, action, location)
            + relationship_modifier(relationships, action)
            + noise(rng)
```

### 3.1 Base Utility: Goal-Action Alignment

The core scoring mechanism (Decision #6). Each action is projected into the same feature space as the agent's goal vector, and the dot product measures alignment.

```python
def base_utility(agent: AgentState, action: Action, world: WorldState) -> float:
    """
    Score how well this action aligns with the agent's goals.
    Projects the action's predicted effects into goal-vector space
    and computes weighted dot product.
    """
    goals = agent.goals
    effects = estimate_action_effects(action, world)

    score = 0.0

    # Safety: actions that reduce exposure/risk score higher for safety-oriented agents
    score += goals.safety * effects.safety_impact

    # Status: actions that increase social standing
    score += goals.status * effects.status_impact

    # Closeness: actions involving agents the actor wants to be close to
    if action.target_agents:
        for target_id in action.target_agents:
            closeness_desire = goals.closeness.get(target_id, 0.0)
            score += closeness_desire * effects.closeness_impact

    # Secrecy: actions that protect secrets score higher for secretive agents
    score += goals.secrecy * effects.secrecy_impact

    # Truth-seeking: actions that reveal information
    score += goals.truth_seeking * effects.truth_impact

    # Autonomy: actions that assert independence
    score += goals.autonomy * effects.autonomy_impact

    # Loyalty: actions that honor commitments/allies
    score += goals.loyalty * effects.loyalty_impact

    return score
```

#### 3.1.1 Action Effect Estimation

Each action type has a characteristic effect profile:

```python
@dataclass
class ActionEffects:
    """Estimated impact of an action on each goal dimension."""
    safety_impact: float = 0.0       # [-1, 1] negative = risky
    status_impact: float = 0.0       # [-1, 1] negative = status-lowering
    closeness_impact: float = 0.0    # [-1, 1] positive = brings closer
    secrecy_impact: float = 0.0      # [-1, 1] negative = exposes secrets
    truth_impact: float = 0.0        # [-1, 1] positive = reveals truth
    autonomy_impact: float = 0.0     # [-1, 1] positive = asserts independence
    loyalty_impact: float = 0.0      # [-1, 1] positive = honors commitments


# Base profiles by action type (modified by context)
ACTION_EFFECT_PROFILES: dict[EventType, ActionEffects] = {
    EventType.CHAT: ActionEffects(
        safety_impact=0.1,       # safe
        status_impact=0.1,       # mild status maintenance
        closeness_impact=0.2,    # builds connection
        secrecy_impact=0.0,      # neutral
        truth_impact=0.0,        # neutral
        autonomy_impact=0.0,     # neutral
        loyalty_impact=0.1,      # mild social duty
    ),
    EventType.OBSERVE: ActionEffects(
        safety_impact=0.2,       # safe — information gathering
        status_impact=0.0,       # neutral
        closeness_impact=0.0,    # neutral
        secrecy_impact=0.1,      # may detect threats to secrets
        truth_impact=0.3,        # information gain
        autonomy_impact=0.1,     # independent action
        loyalty_impact=0.0,      # neutral
    ),
    EventType.SOCIAL_MOVE: ActionEffects(
        safety_impact=0.0,       # depends on destination
        status_impact=-0.05,     # slight cost of leaving
        closeness_impact=0.0,    # depends on who's at destination
        secrecy_impact=0.1,      # can seek privacy
        truth_impact=0.0,        # neutral
        autonomy_impact=0.2,     # asserting independence
        loyalty_impact=0.0,      # neutral
    ),
    EventType.REVEAL: ActionEffects(
        safety_impact=-0.3,      # risky — exposing information
        status_impact=0.2,       # can be power move
        closeness_impact=0.1,    # sharing creates bond
        secrecy_impact=-0.8,     # actively revealing
        truth_impact=0.8,        # major truth event
        autonomy_impact=0.3,     # taking decisive action
        loyalty_impact=-0.2,     # may betray someone's trust
    ),
    EventType.CONFLICT: ActionEffects(
        safety_impact=-0.4,      # risky — confrontation
        status_impact=0.1,       # can assert dominance
        closeness_impact=-0.3,   # damages relationship
        secrecy_impact=-0.2,     # may expose information in argument
        truth_impact=0.4,        # may force truth out
        autonomy_impact=0.4,     # standing up for yourself
        loyalty_impact=0.0,      # context-dependent
    ),
    EventType.INTERNAL: ActionEffects(
        safety_impact=0.3,       # safe — no exposure
        status_impact=0.0,       # invisible
        closeness_impact=0.0,    # invisible
        secrecy_impact=0.3,      # protects secrets (doing nothing)
        truth_impact=0.1,        # processing information
        autonomy_impact=0.2,     # independent thought
        loyalty_impact=0.0,      # neutral
    ),
    EventType.PHYSICAL: ActionEffects(
        safety_impact=0.1,       # usually safe
        status_impact=0.05,      # hosting/serving behavior
        closeness_impact=0.1,    # social lubrication
        secrecy_impact=0.0,      # neutral
        truth_impact=0.0,        # neutral
        autonomy_impact=0.0,     # neutral
        loyalty_impact=0.1,      # mild social duty
    ),
    EventType.CONFIDE: ActionEffects(
        safety_impact=-0.2,      # moderate risk — trusting someone
        status_impact=-0.1,      # vulnerability costs status
        closeness_impact=0.5,    # major closeness builder
        secrecy_impact=-0.5,     # sharing a secret
        truth_impact=0.3,        # sharing truth
        autonomy_impact=0.1,     # choosing to open up
        loyalty_impact=0.3,      # honoring the relationship
    ),
    EventType.LIE: ActionEffects(
        safety_impact=0.1,       # short-term safety (deflecting)
        status_impact=0.0,       # neutral (if not caught)
        closeness_impact=-0.1,   # false closeness
        secrecy_impact=0.5,      # protects secrets
        truth_impact=-0.8,       # actively suppressing truth
        autonomy_impact=-0.1,    # constrained by the lie
        loyalty_impact=-0.4,     # betraying trust
    ),
}


def estimate_action_effects(action: Action, world: WorldState) -> ActionEffects:
    """
    Get the effect profile for an action, adjusted for context.
    Starts with the base profile and modifies based on specifics.
    """
    base = ACTION_EFFECT_PROFILES[action.action_type]
    effects = ActionEffects(**vars(base))  # copy

    # Context adjustments:

    # Moving to a private location increases secrecy_impact
    if action.action_type == EventType.SOCIAL_MOVE:
        dest = world.locations[get_destination(action)]
        effects.secrecy_impact += dest.privacy * 0.3
        effects.safety_impact += dest.privacy * 0.2

    # Revealing a high-weight secret increases truth_impact
    if action.action_type in (EventType.REVEAL, EventType.CONFIDE):
        secret_id = action.content_metadata.get("secret_id")
        if secret_id:
            secret = world.secrets[secret_id]
            effects.truth_impact *= (1.0 + secret.dramatic_weight)
            effects.secrecy_impact *= (1.0 + secret.dramatic_weight)

    # Confronting someone with low trust amplifies conflict effects
    if action.action_type == EventType.CONFLICT and action.target_agents:
        target_id = action.target_agents[0]
        agent = world.agents[action.agent_id]
        rel = agent.relationships.get(target_id)
        if rel and rel.trust < -0.3:
            effects.truth_impact *= 1.5  # more motivated
            effects.closeness_impact *= 1.5  # more destructive

    return effects
```

### 3.2 Flaw Bias: Irrational Distortion

Character flaws distort the base utility by adding bonuses or penalties to specific action types. This is where agents become story-shaped rather than optimal.

```python
def flaw_bias(agent: AgentState, action: Action, perception: PerceivedState,
              world: WorldState) -> float:
    """
    Compute total flaw-induced bias on action score.
    Positive = flaw pushes toward this action.
    Negative = flaw pushes away.
    """
    total_bias = 0.0

    for flaw in agent.flaws:
        if not trigger_active(flaw, agent, action, perception, world):
            continue

        bias = apply_flaw_effect(flaw, action, agent, world)
        total_bias += bias

    return total_bias


def trigger_active(flaw: CharacterFlaw, agent: AgentState, action: Action,
                    perception: PerceivedState, world: WorldState) -> bool:
    """Check if a flaw's trigger condition is met in the current context."""
    match flaw.trigger:
        case "status_threat":
            # Someone's action would lower this agent's status
            return any(
                perceived_as_status_threat(agent, a, perception)
                for a in perception.visible_agents
            )
        case "betrayal_detected":
            # Agent has evidence of betrayal
            return agent.emotional_state.get("suspicion", 0) > 0.4
        case "secret_exposure":
            # One of agent's secrets is at risk
            return any(
                secret_at_risk(agent, s, world) for s in world.secrets.values()
            )
        case "rejection":
            # Recent social rebuff
            return agent.emotional_state.get("shame", 0) > 0.3
        case "authority_challenge":
            # Someone questioned agent's position
            return any(
                challenged_by(agent, a, perception) for a in perception.visible_agents
            )
        case "intimacy_offered":
            # Someone is being vulnerable with agent
            return action.action_type in (EventType.CONFIDE, EventType.CHAT) and action.target_agents
        case "conflict_nearby":
            # Conflict in same location
            return perception.recent_conflict_at_location
        case "loss_imminent":
            # Agent about to lose something valued
            return agent.pacing.stress > 0.5 and agent.pacing.commitment > 0.4
        case _:
            return False


def apply_flaw_effect(flaw: CharacterFlaw, action: Action, agent: AgentState,
                       world: WorldState) -> float:
    """Apply the flaw's effect to generate a bias value."""
    strength = flaw.strength

    match flaw.effect:
        case "overweight_status":
            # Actions that increase status score disproportionately higher
            effects = estimate_action_effects(action, world)
            return strength * effects.status_impact * 0.5

        case "avoid_confrontation":
            # CONFLICT and REVEAL score lower
            if action.action_type in (EventType.CONFLICT, EventType.REVEAL):
                return -strength * 0.5
            # SOCIAL_MOVE (fleeing) scores higher
            if action.action_type == EventType.SOCIAL_MOVE:
                return strength * 0.3
            return 0.0

        case "deny_evidence":
            # OBSERVE scores lower (don't want to see truth)
            if action.action_type == EventType.OBSERVE:
                return -strength * 0.3
            # LIE scores higher (maintain the fiction)
            if action.action_type == EventType.LIE:
                return strength * 0.3
            return 0.0

        case "escalate_conflict":
            # CONFLICT scores higher
            if action.action_type == EventType.CONFLICT:
                return strength * 0.4
            return 0.0

        case "seek_validation":
            # CHAT with high-status agents scores higher
            if action.action_type == EventType.CHAT and action.target_agents:
                target = world.agents[action.target_agents[0]]
                if target.goals.status > 0.6:
                    return strength * 0.3
            return 0.0

        case "self_sacrifice":
            # Actions that help allies at own cost score higher
            if action.action_type in (EventType.CONFIDE, EventType.REVEAL):
                return strength * 0.3
            return 0.0

        case "fixate_on_target":
            # Actions involving obsession target score higher
            # (target derived from flaw context, e.g. "marcus" for Victor)
            if action.target_agents and is_obsession_target(agent, action.target_agents[0]):
                return strength * 0.5
            return 0.0

        case "overcommit":
            # High-risk dramatic actions score higher
            if action.is_dramatic:
                return strength * 0.4
            return 0.0

        case _:
            return 0.0
```

### 3.3 Pacing Modifier

The pacing system (pacing-physics.md) modulates action scores to enforce dramatic rhythm.

```python
def pacing_modifier(agent: AgentState, action: Action, location: Location) -> float:
    """
    Modify action score based on pacing state.
    Enforces dramatic budget, recovery, social masking, and stress-driven behavior.
    """
    modifier = 0.0
    pacing = agent.pacing

    # --- Hard gates (these return -infinity to block the action) ---

    # Recovery timer blocks dramatic actions
    if action.is_dramatic and pacing.recovery_timer > 0:
        return -1000.0  # effectively blocks this action

    # Budget minimum blocks dramatic actions
    if action.is_dramatic and pacing.dramatic_budget < BUDGET_MINIMUM_FOR_ACTION:
        return -1000.0

    # --- Social masking (Decision #5, rule 7) ---
    if action.is_dramatic and location.privacy < 0.3:
        # Public location: suppress dramatic behavior if composure allows
        if pacing.composure >= COMPOSURE_MIN_FOR_MASKING:
            modifier -= (1.0 - location.privacy) * pacing.composure * MASKING_STRESS_SUPPRESSION
        # If composure is below threshold, no masking — the mask has cracked

    # --- Stress-driven behavior ---

    # High stress increases utility of stress-relief actions
    if pacing.stress > 0.5:
        # Moving to private space: relief
        if action.action_type == EventType.SOCIAL_MOVE:
            dest = get_destination_location(action, world)
            if dest and dest.privacy > 0.5:
                modifier += pacing.stress * 0.3  # stressed agents want privacy

        # Confiding: relief
        if action.action_type == EventType.CONFIDE:
            modifier += pacing.stress * 0.2

        # Drinking: relief (PHYSICAL action)
        if action.action_type == EventType.PHYSICAL and "drink" in action.content.lower():
            modifier += pacing.stress * 0.15

    # --- High commitment locks agent into consistent behavior ---
    if pacing.commitment > 0.5:
        # Actions consistent with commitments score higher
        if is_consistent_with_commitments(action, agent):
            modifier += pacing.commitment * 0.2
        # Backtracking actions score lower
        if contradicts_commitments(action, agent):
            modifier -= pacing.commitment * 0.4

    # --- Low budget encourages quiet behavior ---
    if pacing.dramatic_budget < 0.4 and not action.is_dramatic:
        modifier += 0.1  # mild bonus for quiet actions when budget is low

    return modifier
```

### 3.4 Relationship Modifier

Relationships affect action scoring beyond what the goal vector captures.

```python
def relationship_modifier(agent: AgentState, action: Action, world: WorldState) -> float:
    """
    Modify action score based on relationship with target(s).
    """
    if not action.target_agents:
        return 0.0

    modifier = 0.0

    for target_id in action.target_agents:
        rel = agent.relationships.get(target_id)
        if not rel:
            continue

        match action.action_type:
            case EventType.CHAT:
                # Prefer chatting with liked people
                modifier += rel.affection * 0.2

            case EventType.CONFIDE:
                # Only confide in trusted people
                modifier += rel.trust * 0.5
                # Obligation increases confide motivation
                modifier += rel.obligation * 0.2

            case EventType.CONFLICT:
                # More likely to confront distrusted people
                modifier -= rel.trust * 0.3  # negative trust = positive modifier
                # Less likely to confront people with high affection
                modifier -= rel.affection * 0.2

            case EventType.REVEAL:
                # Reveal TO trusted people (sharing information)
                modifier += rel.trust * 0.3
                # But reveal ABOUT distrusted people
                # (handled by checking if the secret is about the target)

            case EventType.LIE:
                # More willing to lie to distrusted people
                modifier -= rel.trust * 0.2  # negative trust = easier to lie
                # Guilt about lying to close people
                modifier -= rel.affection * 0.3

    return modifier
```

### 3.5 Noise

Noise prevents deterministic behavior and creates variation between simulation runs.

```python
def action_noise(rng: Random) -> float:
    """
    Add Gaussian noise to action score.
    Mean 0, standard deviation 0.1.
    This means noise can swing a score by ~0.2 in either direction (2 sigma).
    """
    return rng.gauss(0.0, 0.1)
```

The noise magnitude (sigma=0.1) is calibrated so that:
- Between two clearly differentiated actions (score gap > 0.3), noise rarely changes the winner
- Between two close actions (score gap < 0.15), noise frequently changes the winner
- This creates meaningful variation without chaos

---

## 4. The Complete Scoring Pipeline

```python
def select_action(agent: AgentState, perception: PerceivedState,
                   world: WorldState, rng: Random) -> Action | None:
    """
    The main decision engine entry point.
    Called once per agent per tick by the tick loop.
    """
    # Step 1: Generate candidates
    candidates = generate_candidate_actions(agent, perception, world)
    candidates = prune_candidates(candidates, agent)

    if not candidates:
        return None  # Agent does nothing (rare — INTERNAL is always available)

    # Step 2: Score each candidate
    scored = []
    for action in candidates:
        score = (
            base_utility(agent, action, world)
            + flaw_bias(agent, action, perception, world)
            + pacing_modifier(agent, action, world.locations[agent.location])
            + relationship_modifier(agent, action, world)
            + action_noise(rng)
        )
        scored.append((score, action))

    # Step 3: Select highest-scoring action
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_action = scored[0]

    # Step 4: Set metadata
    best_action.utility_score = best_score
    best_action.priority_class = determine_priority_class(best_action, agent)

    return best_action


def determine_priority_class(action: Action, agent: AgentState) -> int:
    """
    Assign priority class for conflict resolution.
    Higher priority wins in conflict resolution.
    """
    if action.action_type == EventType.CONFLICT and agent.pacing.stress > 0.6:
        return 3  # reactive — responding to pressure
    if action.is_dramatic:
        return 2  # urgent — dramatic actions take priority
    return 1  # normal
```

---

## 5. Worked Decision Traces

### 5.1 Rational Agent: Victor Probes for Information

**Context (tick 25):**
- Victor is at the dining_table, seated next to Thorne and Lydia
- Present: all 6 agents
- Victor's state: stress=0.15, composure=0.88, budget=0.92, recovery=0, commitment=0.2
- Victor suspects the embezzlement (SUSPECTS) and wants to investigate Marcus
- Victor's flaws: obsession (strength=0.8, fixate_on_target), vanity (strength=0.4, seek_validation)

**Candidate actions and scores:**

```
ACTION                              BASE    FLAW    PACING  REL     NOISE   TOTAL
─────────────────────────────────────────────────────────────────────────────────
CHAT with Thorne                    0.42    +0.12   0.00    +0.10   +0.03   0.67
  (seek_validation: Thorne is high-status → +0.12)
  (relationship: affection 0.4 * 0.2 = +0.08, trust 0.5 * 0 = 0)

OBSERVE Marcus                      0.55    +0.40   0.00    0.00    -0.05   0.90
  (base: truth_seeking=1.0 * 0.3 = 0.30, safety=0.3*0.2 = 0.06 ...)
  (obsession fixate: Marcus is target → +0.40)

CHAT with Marcus                    0.35    +0.40   0.00    -0.06   +0.08   0.77
  (obsession fixate: +0.40)
  (relationship: trust -0.1, affection 0.0 → -0.06)

CHAT with Lydia                     0.38    0.00    0.00    +0.06   -0.02   0.42
  (no flaw trigger)
  (relationship: mild positive)

SOCIAL_MOVE to balcony              0.25    0.00    0.00    0.00    +0.01   0.26
  (no motivation to leave — low stress)

OBSERVE room (general)              0.45    0.00    0.00    0.00    +0.04   0.49
  (truth_seeking high, but no obsession focus)

CONFLICT with Marcus                0.30    +0.40   -0.35   -0.02   -0.07   0.26
  (obsession: +0.40)
  (pacing: public masking → composure 0.88 * (1-0.1) * 0.5 = -0.40, but
   composure is high so masking penalty is strong: -(1-0.1) * 0.88 * 0.5 = -0.40)
  Wait — let me recalculate masking:
  modifier = -(1.0 - privacy) * composure * MASKING = -(0.9) * 0.88 * 0.5 = -0.396 ≈ -0.40
  But net dramatic: -0.40 + small stress bonus = -0.35 net

REVEAL embezzlement to Thorne       0.48    +0.40   -0.35   +0.15   +0.02   0.70
  (obsession: +0.40, but masking penalty is strong at table)
  (relationship: trust Thorne 0.5 → +0.15)

INTERNAL (think about evidence)     0.35    0.00    0.00    0.00    +0.06   0.41
```

**Winner: OBSERVE Marcus (score 0.90)**

Victor's obsession flaw (fixate_on_target, strength=0.8) gives a massive +0.40 bonus to any action involving Marcus. Combined with his high truth_seeking (1.0), OBSERVE Marcus scores highest. The REVEAL to Thorne is tempting (0.70) but the masking penalty at the public dining table (-0.35) suppresses it. Victor will watch Marcus carefully this tick — gathering information before making a move.

**Narrative result:** "Victor's eyes track Marcus across the table, watching him refill his wine glass with slightly trembling hands. Something's off."

### 5.2 Irrational Agent: Elena Under Pressure

**Context (tick 52):**
- Elena is at the dining_table
- Present: Thorne, Marcus, Victor, Lydia (Diana is on the balcony)
- Elena's state: stress=0.55, composure=0.42, budget=0.60, recovery=0, commitment=0.45
- Elena knows about the affair (BELIEVES_TRUE) and suspects Diana's debt (SUSPECTS)
- Thorne just asked Elena a pointed question about her recent schedule (creating tension)
- Elena's flaws: guilt (strength=0.7, self_sacrifice), cowardice (strength=0.5, avoid_confrontation)
- Elena's emotional state: fear=0.4, shame=0.5, affection=0.3 (toward Marcus)

**Candidate actions and scores:**

```
ACTION                              BASE    FLAW    PACING  REL     NOISE   TOTAL
─────────────────────────────────────────────────────────────────────────────────
CHAT with Thorne (deflect)          0.32    +0.21   0.00    -0.04   +0.05   0.54
  (guilt self_sacrifice: +0.21 — accommodating Thorne)
  (relationship: obligation 0.6 boosts, but low affection 0.1)

SOCIAL_MOVE to balcony              0.28    +0.25   +0.17   0.00    +0.02   0.72 ★
  (cowardice avoid_confrontation: +0.25 — flee the scene)
  (pacing: stress 0.55 > 0.5 → privacy bonus: 0.55 * 0.3 = +0.17)
  (balcony privacy=0.7 → secrecy bonus in base)

SOCIAL_MOVE to bathroom             0.30    +0.25   +0.17   0.00    -0.03   0.69
  (same cowardice bonus, bathroom more private but further)

CHAT with Marcus (reassure)         0.38    +0.21   0.00    +0.22   +0.01   0.82 ★★
  (guilt self_sacrifice: +0.21)
  (relationship: affection 0.8, trust 0.7 → +0.22)
  Wait — but she's at the dining table, talking to Marcus in front of Thorne

  Hmm, but CHAT isn't dramatic, so no masking penalty applies.
  The problem: this is irrational. She SHOULD avoid Marcus right now (Thorne is
  watching). But her guilt flaw makes her over-accommodate, and high affection
  for Marcus pulls her toward him.

CONFIDE to Marcus (about stress)    0.35    +0.21   -0.27   +0.35   -0.02   0.62
  (guilt: +0.21)
  (pacing: masking penalty at table, but composure 0.42 > 0.40 threshold,
   so masking IS still active: -(0.9) * 0.42 * 0.5 = -0.19,
   plus stress-relief: 0.55*0.2 = +0.11 → net -0.08?
   Hmm wait, dramatic budget check: 0.60 > 0.20 ✓, recovery 0 ✓
   Masking: -(1-0.1)*0.42*0.5 = -0.19; stress relief: +0.11; net pacing: -0.08
   Wait, let me recalculate: -0.19 + 0.11 = -0.08? That's too mild.
   Actually the masking applies to dramatic actions specifically:
   modifier = -0.19 for masking
   modifier += 0.55 * 0.2 for stress-relief confide
   net = -0.19 + 0.11 = -0.08)
  (relationship: trust 0.7 * 0.5 + obligation 0.2 * 0.2 = +0.39)
  Hmm, total: 0.35 + 0.21 + (-0.08) + 0.39 + (-0.02) = 0.85?

  Wait. Let me re-examine. Confide is dramatic, and masking is:
  -(1.0 - 0.1) * 0.42 * 0.5 = -0.189
  stress relief: 0.55 * 0.2 = 0.11
  net pacing: -0.189 + 0.11 = -0.079
  relationship: trust 0.7 → 0.7 * 0.5 = 0.35; obligation 0.2 * 0.2 = 0.04; total rel: 0.39
  But also commitment consistency: maintaining affair secrecy → confiding ABOUT the affair
  is contradicting commitment... hmm, she's confiding about stress, not the affair directly.
  Let's say no commitment contradiction.

  Total: 0.35 + 0.21 + (-0.08) + 0.39 + (-0.02) = 0.85

  This is high! But it would be confiding in Marcus AT THE TABLE in front of Thorne.
  The masking penalty is modest because her composure is borderline.

LIE to Thorne (about schedule)     0.28    0.00    -0.19   -0.12   +0.04   0.01
  (no flaw trigger for lying, cowardice doesn't push toward lies)
  (masking penalty: dramatic at table)
  (relationship: guilt about lying to husband → affection penalty)
  (base: secrecy=0.9 * 0.5 = +0.45, but truth_seeking=0.3 * -0.8 = -0.24 → net 0.21
   plus other dims...)

OBSERVE Thorne                      0.30    0.00    0.00    0.00    -0.01   0.29
  (no flaw trigger, mild information gathering)

INTERNAL (process emotions)         0.40    0.00    0.00    0.00    +0.03   0.43
  (safe, processes shame/fear internally)

PHYSICAL (drink wine)               0.25    0.00    +0.08   0.00    +0.01   0.34
  (stress relief: 0.55 * 0.15 = +0.08)
```

**Let me re-sort with corrected values:**

```
CONFIDE to Marcus (about stress):  0.85  ← IRRATIONAL WINNER
CHAT with Marcus (reassure):       0.82
SOCIAL_MOVE to balcony:            0.72
SOCIAL_MOVE to bathroom:           0.69
CONFIDE re-scored... wait.

Actually the big question: is "confide about stress to Marcus at the table" really the
highest? Let me reconsider. The confide is dramatic → masking applies. But the
relationship modifier for Marcus is very high (+0.39) and guilt flaw adds +0.21.

But CHAT with Marcus is non-dramatic → no masking → and still gets guilt (+0.21) and
relationship (+0.22). CHAT scores 0.82.

And CONFIDE scores 0.85 — just barely higher due to the relationship trust modifier
being 0.5 * 0.7 = 0.35 for confide vs 0.2 * 0.8 = 0.16 for chat affection.
```

**Winner: CONFIDE to Marcus (score 0.85) — but let's examine the irrationality**

Elena's rational move is to flee (SOCIAL_MOVE to balcony, 0.72). The OBSERVE or INTERNAL options are safest. But her flaws and relationships override rationality:

1. **Guilt** (self_sacrifice) pushes her toward accommodating behavior — talking to Marcus, even when it's risky
2. **High affection** for Marcus (0.8) makes any action involving him score higher
3. **Cowardice** (avoid_confrontation) pushes away from conflict, but toward flight OR toward "appeasing" behavior (chatting, confiding)
4. The CONFIDE scores highest because the trust+affection modifier for Marcus is so strong, and confiding offers stress relief

**But this is a terrible idea.** Confiding in Marcus at the dining table, in front of Thorne, even about "stress," risks drawing attention to their closeness. An optimal agent would flee to the balcony. Elena's guilt and affection for Marcus override her rational self-interest.

**Actual winner after noise consideration:** With noise sigma=0.1, there's about a 40% chance CHAT with Marcus wins instead (gap is only 0.03). And about a 20% chance the flee response wins (gap is 0.13). The noise creates genuine uncertainty about whether Elena succumbs to her irrational pull or manages to escape.

Let's say noise gives CHAT with Marcus the edge this tick:

**Narrative result:** "Elena turns to Marcus with too-bright eyes. 'How's the wine? I think it might be the 2019.' Her voice is a shade too casual. Across the table, Lydia notices."

This is exactly the kind of "suboptimal but dramatically rich" behavior the decision engine should produce. Elena's guilt and attachment to Marcus make her gravitate toward him even when every strategic instinct says she should be putting distance between them.

---

## 6. Special Decision Logic

### 6.1 Alcohol-Influenced Decisions

Alcohol (tracked as `alcohol_level` on AgentState) affects the decision engine:

```python
def alcohol_modifier(agent: AgentState, action: Action) -> float:
    """
    Alcohol lowers inhibitions — dramatic actions become more attractive.
    Applied as a small positive modifier to dramatic actions.
    """
    if agent.alcohol_level < 0.15:
        return 0.0  # below threshold — no effect

    if action.is_dramatic:
        # Alcohol makes dramatic actions more appealing
        return agent.alcohol_level * 0.3
    return 0.0
```

This is deliberately simple. Alcohol's main effect is through composure degradation (pacing-physics.md), which lowers the masking penalty. The direct decision modifier is a secondary "lowered inhibitions" effect.

### 6.2 Movement as Strategic Action

Agents don't just move randomly. The decision engine scores SOCIAL_MOVE with strategic intelligence:

```python
def score_move(agent: AgentState, destination: Location, world: WorldState) -> float:
    """Additional scoring for SOCIAL_MOVE actions based on strategic value."""
    score = 0.0

    # Who's at the destination?
    agents_at_dest = get_agents_at(world, destination.id)

    # Move toward desired interactions
    for other in agents_at_dest:
        closeness_desire = agent.goals.closeness.get(other.id, 0.0)
        if closeness_desire > 0.3:
            score += closeness_desire * 0.3  # want to be near them

    # Move toward privacy when stressed
    if agent.pacing.stress > 0.5:
        score += destination.privacy * 0.2

    # Move toward privacy when wanting to confide/reveal
    if has_pending_dramatic_action(agent, world):
        score += destination.privacy * 0.3

    # Follow someone who just left (if motivated)
    recent_movers = get_agents_who_recently_moved(world, destination.id)
    for mover in recent_movers:
        rel = agent.relationships.get(mover.id)
        if rel and (rel.trust > 0.5 or agent.emotional_state.get("suspicion", 0) > 0.4):
            score += 0.2  # follow them

    return score
```

### 6.3 The "Do Nothing" Bias

Every tick, the INTERNAL action serves as the "do nothing meaningful" baseline. It has a moderate base score (0.35-0.45) that serves as the threshold other actions must beat. If all actions score below the INTERNAL threshold, the agent effectively does nothing interesting — they think, observe passively, or sit quietly.

This prevents agents from acting just because they can. In early ticks when stress is low and there's no motivation, agents will frequently choose INTERNAL or low-key CHAT — producing the calm opening that makes later drama meaningful.

---

## 7. Tuning Guide

### 7.1 Score Ranges

Expected total score ranges for a well-tuned system:

| Score Range | Meaning |
|---|---|
| < 0.0 | Blocked by pacing or extremely misaligned |
| 0.0 - 0.3 | Weak — unlikely to be selected |
| 0.3 - 0.5 | Moderate — "do nothing" territory (INTERNAL, OBSERVE) |
| 0.5 - 0.7 | Solid — motivated CHAT, strategic SOCIAL_MOVE |
| 0.7 - 0.9 | Strong — flaw-driven or relationship-driven dramatic action |
| > 0.9 | Dominant — the agent strongly wants this (usually flaw + relationship + stress stacking) |

### 7.2 Common Tuning Problems

| Problem | Symptom | Fix |
|---|---|---|
| Agents never act dramatically | All scores < 0.5 | Increase flaw strength, reduce masking penalty |
| Agents always choose the same action | Score gaps too large | Increase noise sigma (0.1 → 0.15) |
| Agents ignore relationships | Relationship modifier too weak | Increase relationship coefficients (0.2 → 0.4) |
| Agents always flee | SOCIAL_MOVE scores too high | Reduce privacy bonus, increase status cost of leaving |
| Flaws dominate everything | Flaw bias > 1.0 | Cap flaw bias at 0.5 per flaw, or reduce strength values |

---

## 8. Edge Cases

### 8.1 No Agents Present

If an agent is alone (bathroom, balcony), only INTERNAL, PHYSICAL, SOCIAL_MOVE, and OBSERVE are available. The agent will typically either think, drink, or move back to the group.

### 8.2 All Actions Blocked

If recovery_timer > 0 AND dramatic_budget < minimum, the agent can only choose non-dramatic actions. This is intentional — it forces quiet behavior after dramatic episodes.

### 8.3 Conflicting Flaws

If an agent has two flaws that push in opposite directions (e.g., loyalty toward confrontation and cowardice away from it), the net bias is the sum. This can produce paralysis (net bias near zero, noise decides) or oscillation (one tick the loyalty wins, next tick the cowardice wins). Both are narratively interesting — they represent genuine internal conflict.

### 8.4 Perfect Information Problem

The perception system (tick-loop.md Phase 1) ensures agents don't have perfect information. An agent acts on beliefs, not truth. If Elena believes Marcus hasn't embezzled (UNKNOWN), she won't confront him about it — even though the player (audience) knows it's true. This is critical for dramatic irony to work.

---

## 9. Relationship to Other Specs

| Spec | Relationship |
|---|---|
| **tick-loop.md** | Calls `select_action()` in Phase 3. Uses returned Action for conflict resolution. |
| **pacing-physics.md** | Pacing state is INPUT to `pacing_modifier()`. Decision engine does not update pacing. |
| **agents.md** | GoalVector, CharacterFlaw, RelationshipState are the core data structures read by scoring. |
| **events.md** | EventType enum determines action categorization. StateDelta is output of event generation. |
| **world.md** | Location privacy, adjacency, overhear rules inform action generation and scoring. |
| **dinner-party-config.md** | Character designs (goals, flaws, relationships) are the input that makes scoring produce interesting behavior. |

---

## 10. Open Questions

1. **Should there be memory of recent actions?** Currently, agents don't track "I chatted with Thorne last tick." Adding short-term memory could prevent repetitive behavior (chatting with the same person 5 ticks in a row). Implementation: a small recency penalty for repeating the same (type, target) pair within 3 ticks.

2. **Should agents model other agents' likely actions?** Currently, agents score actions based on their own state without predicting what others will do. Adding Theory of Mind (ToM) would let agents anticipate reactions — e.g., "If I confront Marcus, Thorne will likely side with me." This is complex and probably Phase 3+.

3. **Content generation for actions.** The `content` and `dialogue` fields on Action need to be filled with human-readable text. For MVP, these can be template-based. For production, they could be LLM-generated from the structured action data. This spec doesn't define the content generation approach — it's a separate concern.
