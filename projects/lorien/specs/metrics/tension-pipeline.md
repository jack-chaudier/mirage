# Tension Pipeline Specification

> **Spec:** `specs/metrics/tension-pipeline.md`
> **Owner:** metrics-architect
> **Status:** Draft
> **Depends on:** `specs/schema/events.md` (#1), `specs/schema/agents.md` (#2)
> **Blocks:** `specs/integration/data-flow.md` (#17)
> **Doc3 decisions:** #5 (pacing physics), #6 (vectorized feature scoring), #8 (finite belief catalog), #14 (arc grammars — soft scoring uses tension)

---

## 1. Overview

The tension pipeline computes a composite **tension score** for every event in the event log. Tension is the primary scalar field that drives the visualization heatmap, story extraction scoring, and the reader's sense of "something is at stake."

Tension is NOT a single formula. It is a **weighted sum of 8 interpretable sub-metrics**, each normalized to [0.0, 1.0]. Users tune the weights via `TensionWeights` to shift the lens between genres (thriller, relationship drama, mystery).

### Pipeline Summary

```
EventLog + WorldState snapshots
    → per-event sub-metric computation (8 floats)
    → weighted aggregation (TensionWeights → 1 float)
    → normalization (global min-max over the run)
    → attachment to Event.metrics["tension"] and Event.metrics["tension_components"]
```

### Timing

The tension pipeline runs as a **post-processing pass** over the completed event log. It is NOT computed during simulation. The simulation produces events and deltas; the metrics pipeline reads them afterward.

For interactive use (re-simulation), the pipeline must be fast enough to recompute on ~200 events in <100ms on commodity hardware. All sub-metrics are O(1) per event given indexed access to world state.

---

## 2. Sub-Metric Definitions

Each sub-metric is a function:

```
sub_metric(event: Event, world_before: WorldState, world_after: WorldState, context: MetricsContext) -> float
```

Where `MetricsContext` provides:
- The full event log (for lookups)
- Index tables (agent timelines, location events, secret events, interaction pairs)
- The secrets registry (ground truth)
- The belief matrix at the current tick
- Global normalization constants (computed in a first pass)

All sub-metrics return values in [0.0, 1.0].

### 2.1 danger

**What it measures:** Proximity to physical or social harm for the agents involved in this event.

**Formula:**

```python
def danger(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    Combines physical threat and social threat.
    Physical: event type is CONFLICT or CATASTROPHE, or agent is in a location
              with an ongoing conflict.
    Social:   an agent's reputation/status is under direct attack (accusation,
              public humiliation, exposure of secret).
    """
    physical = 0.0
    social = 0.0

    # Physical threat: event types that imply direct confrontation
    DANGER_TYPES = {EventType.CONFLICT: 0.8, EventType.CATASTROPHE: 1.0}
    if event.type in DANGER_TYPES:
        physical = DANGER_TYPES[event.type]

    # Social threat: any delta that reduces status or trust sharply
    for delta in event.deltas:
        if delta.kind == DeltaKind.RELATIONSHIP and delta.attribute == "trust":
            if delta.op == DeltaOp.ADD and isinstance(delta.value, (int, float)):
                if delta.value < -0.3:
                    social = max(social, min(abs(delta.value), 1.0))
        if delta.kind == DeltaKind.SECRET_STATE:
            # A secret being exposed publicly is social danger for the holder
            secret = context.secrets[delta.attribute]  # delta.attribute holds the secret_id for SECRET_STATE deltas
            if secret.about is not None:
                # The person the secret is about faces social danger
                social = max(social, 0.7)

    # Combine: take the max channel, with a small bonus for both being present
    combined = max(physical, social) + 0.15 * min(physical, social)
    return min(combined, 1.0)
```

**Edge cases:**
- INTERNAL events have danger = 0.0 (no one is threatened by a thought).
- OBSERVE events can have danger > 0 if the observer witnesses a conflict (they inherit partial threat from the observed event's danger).

### 2.2 time_pressure

**What it measures:** How close agents are to deadlines, irreversible thresholds, or forced decisions.

**Formula:**

```python
def time_pressure(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    In the dinner party context, time pressure comes from:
    1. The evening ending (fixed endpoint: sim_time approaching max_time)
    2. An agent's recovery_timer being active (they're suppressing, clock ticking)
    3. Secrets that are "about to be revealed" (multiple agents SUSPECT)
    4. Alcohol degrading composure over time (irreversible within the evening)
    """
    scores = []

    # 1. Evening progression: pressure increases as the night goes on
    progress = event.sim_time / context.max_sim_time  # 0.0 to 1.0
    # Use a quadratic ramp: pressure accelerates in the final third
    evening_pressure = progress ** 2
    scores.append(evening_pressure)

    # 2. Recovery timer pressure: agents with active timers are "holding back"
    source = world_before.agents[event.source_agent]
    if source.pacing.recovery_timer > 0:
        # Pressure from suppression: higher when timer is almost up
        timer_ratio = 1.0 - (source.pacing.recovery_timer / context.max_recovery_timer)
        scores.append(timer_ratio * 0.6)

    # 3. Secret convergence: count agents who SUSPECT a secret relevant to this event
    for secret_id in context.secrets_relevant_to_event(event):
        suspect_count = sum(
            1 for agent_id in world_before.agents
            if world_before.agents[agent_id].beliefs.get(secret_id) == BeliefState.SUSPECTS
        )
        if suspect_count >= 2:
            scores.append(min(suspect_count / 4.0, 1.0))  # caps at 4 suspectors

    # 4. Alcohol-driven composure loss (for source agent)
    composure_loss = 1.0 - source.pacing.composure
    if composure_loss > 0.3:
        scores.append(composure_loss * 0.5)

    return min(max(scores) if scores else 0.0, 1.0)
```

**Dinner party specifics:** The `max_sim_time` is approximately 180 minutes (3 hours). The quadratic ramp means pressure is low for the first hour, moderate for the second, and high for the third -- matching a natural evening arc.

### 2.3 goal_frustration

**What it measures:** How far the source agent is from satisfying their goals, using the GoalVector cosine distance framework (Decision #6).

**Formula:**

```python
def goal_frustration(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    Compute the distance between the agent's goal vector and the "satisfaction vector"
    representing how well their current world state satisfies each goal dimension.

    Frustration = 1 - cosine_similarity(goal_vector, satisfaction_vector)
    """
    agent = world_before.agents[event.source_agent]
    goal_vec = _goal_to_array(agent.goals)        # [safety, status, secrecy, truth, autonomy, loyalty]
    sat_vec = _satisfaction_to_array(agent, world_before, context)

    # Cosine similarity
    dot = sum(g * s for g, s in zip(goal_vec, sat_vec))
    norm_g = sqrt(sum(g ** 2 for g in goal_vec))
    norm_s = sqrt(sum(s ** 2 for s in sat_vec))

    if norm_g == 0 or norm_s == 0:
        return 0.5  # neutral if goals or satisfaction are zero vectors

    cosine_sim = dot / (norm_g * norm_s)
    frustration = (1.0 - cosine_sim) / 2.0  # normalize from [-1,1] to [0,1]
    return frustration


def _satisfaction_to_array(agent: AgentState, world: WorldState, ctx: MetricsContext) -> list[float]:
    """
    Map each goal dimension to a satisfaction score [0, 1]:
    - safety: 1.0 if no conflict in agent's location, scales down with danger
    - status: agent's average trust received from all other agents
    - secrecy: fraction of agent's secrets that remain UNKNOWN to others
    - truth_seeking: fraction of secrets the agent BELIEVES correctly
    - autonomy: 1.0 - commitment level (more commitments = less autonomy)
    - loyalty: fraction of commitments the agent has honored (not broken)
    """
    safety = 1.0 - ctx.location_danger(agent.location, world)

    all_trusts = [
        world.agents[other].relationships.get(agent.id, {}).get("trust", 0.0)
        for other in world.agents if other != agent.id
    ]
    status = (sum(all_trusts) / len(all_trusts) + 1.0) / 2.0 if all_trusts else 0.5

    own_secrets = [s for s in ctx.secrets.values() if s.holder == agent.id]
    if own_secrets:
        hidden = sum(
            1 for s in own_secrets
            if all(
                world.agents[a].beliefs.get(s.id, BeliefState.UNKNOWN) == BeliefState.UNKNOWN
                for a in world.agents if a != agent.id
            )
        )
        secrecy = hidden / len(own_secrets)
    else:
        secrecy = 1.0

    all_secrets = list(ctx.secrets.values())
    if all_secrets:
        correct = sum(
            1 for s in all_secrets
            if (agent.beliefs.get(s.id) == BeliefState.BELIEVES_TRUE and s.truth_value)
            or (agent.beliefs.get(s.id) == BeliefState.BELIEVES_FALSE and not s.truth_value)
        )
        truth_seeking = correct / len(all_secrets)
    else:
        truth_seeking = 1.0

    autonomy = 1.0 - min(len(agent.commitments) / 5.0, 1.0)

    # Loyalty: simplified — assume honored if commitment is still active
    loyalty = 1.0  # default; reduced by BETRAYAL_OBSERVED reason codes in deltas

    return [safety, status, secrecy, truth_seeking, autonomy, loyalty]
```

**Normalization note:** The GoalVector uses `closeness: dict` for per-agent closeness desires. For the array conversion, closeness is averaged into the `status` dimension. A more refined version could add per-target closeness dimensions, but for the dinner party MVP, 6 dimensions suffice.

### 2.4 relationship_volatility

**What it measures:** The rate of change of relationship states among event participants.

**Formula:**

```python
def relationship_volatility(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    Look at the relationship deltas produced by this event and the last N events
    involving these participants. High rate of change = high volatility.
    """
    LOOKBACK_WINDOW = 5  # events

    # Collect relationship deltas from this event
    current_deltas = [
        abs(d.value) for d in event.deltas
        if d.kind == DeltaKind.RELATIONSHIP
        and isinstance(d.value, (int, float))
    ]

    # Collect relationship deltas from recent events involving same participants
    participants = {event.source_agent} | set(event.target_agents)
    recent_events = context.recent_events_for_agents(participants, LOOKBACK_WINDOW)
    recent_deltas = []
    for re in recent_events:
        recent_deltas.extend(
            abs(d.value) for d in re.deltas
            if d.kind == DeltaKind.RELATIONSHIP
            and isinstance(d.value, (int, float))
        )

    all_deltas = current_deltas + recent_deltas
    if not all_deltas:
        return 0.0

    # Volatility = mean absolute delta, scaled to [0, 1]
    # A delta of 0.5 (half the trust range) in one event is very high
    mean_delta = sum(all_deltas) / len(all_deltas)
    volatility = min(mean_delta / 0.5, 1.0)

    # Bonus for oscillation: if deltas alternate sign, add instability bonus
    signed_deltas = [
        d.value for e in [event] + recent_events for d in e.deltas
        if d.kind == DeltaKind.RELATIONSHIP and isinstance(d.value, (int, float))
    ]
    if len(signed_deltas) >= 2:
        sign_changes = sum(
            1 for i in range(1, len(signed_deltas))
            if (signed_deltas[i] > 0) != (signed_deltas[i - 1] > 0)
        )
        oscillation_bonus = min(sign_changes / 3.0, 0.3)
        volatility = min(volatility + oscillation_bonus, 1.0)

    return volatility
```

### 2.5 information_gap

**What it measures:** The gap between what the audience (omniscient reader) knows and what the characters in this event know. This is the structural basis of suspense.

**Formula:**

```python
def information_gap(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    For each secret relevant to this event's participants and location:
    - If the audience knows it (truth_value is defined) but participant(s) don't:
      that's a gap.
    - Weighted by relevance: a secret ABOUT a participant scores higher than
      a secret about someone absent.
    """
    participants = {event.source_agent} | set(event.target_agents)
    total_gap = 0.0
    relevant_secrets = 0

    for secret_id, secret in context.secrets.items():
        # Determine relevance to this event
        relevance = _secret_relevance(secret, participants, event.location_id)
        if relevance < 0.1:
            continue
        relevant_secrets += 1

        # Compute gap: how many participants are ignorant or wrong?
        gap_score = 0.0
        for agent_id in participants:
            belief = world_before.agents[agent_id].beliefs.get(secret_id, BeliefState.UNKNOWN)
            if belief == BeliefState.UNKNOWN:
                gap_score += 0.5 * relevance
            elif (belief == BeliefState.BELIEVES_TRUE and not secret.truth_value) or \
                 (belief == BeliefState.BELIEVES_FALSE and secret.truth_value):
                gap_score += 1.0 * relevance  # actively wrong = bigger gap
            # SUSPECTS gets partial credit (smaller gap)
            elif belief == BeliefState.SUSPECTS:
                gap_score += 0.25 * relevance

        # Normalize by number of participants
        if participants:
            gap_score /= len(participants)
        total_gap += gap_score

    if relevant_secrets == 0:
        return 0.0

    # Normalize by number of relevant secrets, cap at 1.0
    return min(total_gap / relevant_secrets, 1.0)


def _secret_relevance(secret: Secret, participants: set[str], location_id: str) -> float:
    """
    How relevant is this secret to the current event?
    - 1.0 if the secret is ABOUT a participant
    - 0.7 if the secret is HELD BY a participant
    - 0.4 if the secret's content_type matches the event theme
    - 0.1 otherwise (background knowledge)
    """
    if secret.about in participants:
        return 1.0
    if secret.holder in participants:
        return 0.7
    # Content-type matching is a simplification; in a full system
    # this would check thematic relevance to the event
    return 0.1
```

**Relationship to irony:** `information_gap` and `irony_density` are correlated but distinct. `information_gap` measures audience-vs-character knowledge asymmetry (suspense). `irony_density` measures character-vs-truth alignment (irony). A scene can have high information_gap but low irony if characters simply don't know things, vs. high irony if they actively believe the wrong thing.

### 2.6 resource_scarcity

**What it measures:** Competition for limited resources -- in the dinner party context: alcohol supply, exit availability (can someone leave?), social capital (alliances), and conversational access (who can talk to whom given seating/location).

**Formula:**

```python
def resource_scarcity(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    Dinner-party-specific resource scarcity.
    Generalizable to other scenarios by swapping resource definitions.
    """
    scores = []

    # 1. Social capital scarcity: does the source agent have allies present?
    agent = world_before.agents[event.source_agent]
    present_agents = context.agents_at_location(event.location_id, world_before)
    allies = [
        a for a in present_agents
        if a != event.source_agent
        and agent.relationships.get(a, {}).get("trust", 0.0) > 0.3
    ]
    enemies = [
        a for a in present_agents
        if a != event.source_agent
        and agent.relationships.get(a, {}).get("trust", 0.0) < -0.3
    ]
    if present_agents:
        isolation = 1.0 - (len(allies) / max(len(present_agents) - 1, 1))
        enemy_ratio = len(enemies) / max(len(present_agents) - 1, 1)
        social_scarcity = (isolation + enemy_ratio) / 2.0
        scores.append(social_scarcity)

    # 2. Exit scarcity: is the agent in a location they can't easily leave?
    location = world_before.locations[event.location_id]
    if location.privacy < 0.3:  # public location = harder to escape
        crowd = len(present_agents)
        if crowd >= location.capacity - 1:
            scores.append(0.7)  # nearly full, hard to move

    # 3. Privacy scarcity: does the agent need a private conversation but is in public?
    if event.type in {EventType.CONFIDE, EventType.REVEAL, EventType.CONFLICT}:
        if location.privacy < 0.5:
            scores.append(0.6)  # wants privacy, doesn't have it

    return max(scores) if scores else 0.0
```

### 2.7 moral_cost

**What it measures:** The ethical price of the best available action for the source agent. High when all options are bad.

**Formula:**

```python
def moral_cost(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    Moral cost is high when the event involves:
    1. Betraying an ally (breaking trust with someone who trusts you)
    2. Lying to someone who trusts you
    3. Revealing a confidence (betraying a confide)
    4. Choosing between conflicting loyalties
    5. Sacrificing one relationship to protect another
    """
    cost = 0.0
    agent = world_before.agents[event.source_agent]

    # Betrayal cost: event harms someone who trusts the source
    for target_id in event.target_agents:
        trust_from_target = world_before.agents[target_id].relationships.get(
            event.source_agent, {}
        ).get("trust", 0.0)

        # Check if this event damages the target
        harm_to_target = sum(
            abs(d.value) for d in event.deltas
            if d.kind == DeltaKind.RELATIONSHIP
            and d.agent_b == target_id
            and isinstance(d.value, (int, float))
            and d.value < 0
        )

        if trust_from_target > 0.3 and harm_to_target > 0:
            # Betrayal: harming someone who trusts you
            cost = max(cost, trust_from_target * harm_to_target * 2.0)

    # Lie cost: proportional to the trust the target has in the source
    if event.type == EventType.LIE:
        for target_id in event.target_agents:
            trust = world_before.agents[target_id].relationships.get(
                event.source_agent, {}
            ).get("trust", 0.0)
            cost = max(cost, max(trust, 0.0) * 0.7)

    # Confidence betrayal: revealing a secret that was confided
    if event.type == EventType.REVEAL:
        for delta in event.deltas:
            if delta.kind == DeltaKind.SECRET_STATE:
                secret = context.secrets.get(delta.agent)
                if secret and secret.holder != event.source_agent:
                    # Revealing someone else's secret
                    cost = max(cost, 0.8)

    # Loyalty conflict: agent has high loyalty goal but event damages a commitment
    if agent.goals.loyalty > 0.6:
        for delta in event.deltas:
            if delta.kind == DeltaKind.COMMITMENT:
                cost = max(cost, agent.goals.loyalty * 0.6)

    return min(cost, 1.0)
```

### 2.8 irony_density

**What it measures:** The concentration of dramatic irony among the agents present in this event. Computed from the belief matrix (Decision #8).

**Formula:**

```python
def irony_density(event: Event, world_before: WorldState, context: MetricsContext) -> float:
    """
    Sum of irony scores for all agents present in this event,
    normalized by the number of agents and relevant secrets.

    See specs/metrics/irony-and-beliefs.md for the full irony computation.
    """
    participants = {event.source_agent} | set(event.target_agents)
    # Also include observers at the same location
    observers = context.agents_at_location(event.location_id, world_before)
    all_present = participants | set(observers)

    total_irony = 0.0
    relevant_count = 0

    for agent_id in all_present:
        for secret_id, secret in context.secrets.items():
            relevance = _secret_relevance(secret, {agent_id}, event.location_id)
            if relevance < 0.1:
                continue
            relevant_count += 1

            belief = world_before.agents[agent_id].beliefs.get(
                secret_id, BeliefState.UNKNOWN
            )
            if secret.truth_value and belief == BeliefState.BELIEVES_FALSE:
                total_irony += 2.0 * relevance  # actively wrong
            elif not secret.truth_value and belief == BeliefState.BELIEVES_TRUE:
                total_irony += 2.0 * relevance  # believes a lie
            elif belief == BeliefState.UNKNOWN and relevance >= 0.5:
                total_irony += 1.0 * relevance  # critical ignorance

    if relevant_count == 0:
        return 0.0

    # Normalize: max possible irony per (agent, secret) pair is 2.0
    max_possible = relevant_count * 2.0
    return min(total_irony / max_possible, 1.0)
```

---

## 3. TensionWeights and Aggregation

### 3.1 The TensionWeights Struct

```python
@dataclass
class TensionWeights:
    """
    User-tunable weights for the 8 tension sub-metrics.
    Default: all weights = 1.0 (equal contribution).
    Genre presets modify these weights.
    Weights are NOT required to sum to 1.0 — they are relative importance multipliers.
    """
    danger: float = 1.0
    time_pressure: float = 1.0
    goal_frustration: float = 1.0
    relationship_volatility: float = 1.0
    information_gap: float = 1.0
    resource_scarcity: float = 1.0
    moral_cost: float = 1.0
    irony_density: float = 1.0
```

### 3.2 Aggregation Formula

```python
def compute_tension(
    components: dict[str, float],
    weights: TensionWeights
) -> float:
    """
    Weighted mean of sub-metrics.

    tension = (sum of w_i * c_i) / (sum of w_i)

    This ensures the output is in [0.0, 1.0] since all components are in [0.0, 1.0].
    """
    w = [
        weights.danger, weights.time_pressure, weights.goal_frustration,
        weights.relationship_volatility, weights.information_gap,
        weights.resource_scarcity, weights.moral_cost, weights.irony_density,
    ]
    c = [
        components["danger"], components["time_pressure"],
        components["goal_frustration"], components["relationship_volatility"],
        components["information_gap"], components["resource_scarcity"],
        components["moral_cost"], components["irony_density"],
    ]
    total_weight = sum(w)
    if total_weight == 0:
        return 0.0
    return sum(wi * ci for wi, ci in zip(w, c)) / total_weight
```

### 3.3 Genre Presets (Exact Weight Vectors)

```python
# Preset 1: Thriller
THRILLER_WEIGHTS = TensionWeights(
    danger=2.5,
    time_pressure=2.0,
    goal_frustration=1.0,
    relationship_volatility=0.5,
    information_gap=1.5,
    resource_scarcity=1.5,
    moral_cost=0.5,
    irony_density=1.0,
)
# Emphasizes: physical danger, racing clock, hidden information
# De-emphasizes: relationship drama, moral dilemmas

# Preset 2: Relationship Drama
RELATIONSHIP_DRAMA_WEIGHTS = TensionWeights(
    danger=0.3,
    time_pressure=0.5,
    goal_frustration=1.5,
    relationship_volatility=2.5,
    information_gap=1.0,
    resource_scarcity=0.5,
    moral_cost=2.0,
    irony_density=1.5,
)
# Emphasizes: relationship instability, moral anguish, ironic misunderstandings
# De-emphasizes: physical danger, resource competition

# Preset 3: Mystery
MYSTERY_WEIGHTS = TensionWeights(
    danger=1.0,
    time_pressure=1.0,
    goal_frustration=0.5,
    relationship_volatility=0.5,
    information_gap=2.5,
    resource_scarcity=0.3,
    moral_cost=1.0,
    irony_density=2.0,
)
# Emphasizes: what the audience knows vs. characters, irony from false beliefs
# De-emphasizes: goal pursuit, resource competition
```

### 3.4 Weight Vector Summary Table

| Sub-metric | Default | Thriller | Rel. Drama | Mystery |
|---|---|---|---|---|
| danger | 1.0 | **2.5** | 0.3 | 1.0 |
| time_pressure | 1.0 | **2.0** | 0.5 | 1.0 |
| goal_frustration | 1.0 | 1.0 | 1.5 | 0.5 |
| relationship_volatility | 1.0 | 0.5 | **2.5** | 0.5 |
| information_gap | 1.0 | 1.5 | 1.0 | **2.5** |
| resource_scarcity | 1.0 | 1.5 | 0.5 | 0.3 |
| moral_cost | 1.0 | 0.5 | **2.0** | 1.0 |
| irony_density | 1.0 | 1.0 | 1.5 | **2.0** |

---

## 4. Output Format

### 4.1 MetricsPayload (Per-Event)

```python
@dataclass
class TensionPayload:
    """Attached to Event.metrics after the tension pipeline runs."""
    tension: float                    # composite score, [0.0, 1.0]
    tension_components: dict[str, float]  # 8 sub-metric values
    # Keys: "danger", "time_pressure", "goal_frustration",
    #        "relationship_volatility", "information_gap",
    #        "resource_scarcity", "moral_cost", "irony_density"
```

### 4.2 JSON Representation

After pipeline execution, each event's `metrics` dict contains:

```json
{
    "tension": 0.63,
    "tension_components": {
        "danger": 0.0,
        "time_pressure": 0.25,
        "goal_frustration": 0.71,
        "relationship_volatility": 0.85,
        "information_gap": 0.60,
        "resource_scarcity": 0.30,
        "moral_cost": 0.90,
        "irony_density": 0.45
    },
    "irony": 0.45,
    "significance": 0.0,
    "thematic_shift": {
        "loyalty_betrayal": -0.2,
        "truth_deception": 0.1
    }
}
```

---

## 5. Pipeline Execution Order

```python
def run_tension_pipeline(
    events: list[Event],
    snapshots: dict[int, WorldState],
    secrets: dict[str, Secret],
    weights: TensionWeights,
    index_tables: IndexTables,
) -> None:
    """
    Mutates each event's metrics dict in place.

    Two-pass algorithm:
    Pass 1: Compute raw sub-metrics for every event.
    Pass 2: Global min-max normalization (optional — only if we want
            tension to be relative to this specific run).
    """
    context = MetricsContext(
        events=events,
        snapshots=snapshots,
        secrets=secrets,
        index_tables=index_tables,
        max_sim_time=events[-1].sim_time if events else 1.0,
        max_recovery_timer=10,  # configurable
    )

    # Pass 1: compute raw sub-metrics
    for event in events:
        world_before = context.world_state_before(event)
        world_after = context.world_state_after(event)

        components = {
            "danger": danger(event, world_before, context),
            "time_pressure": time_pressure(event, world_before, context),
            "goal_frustration": goal_frustration(event, world_before, context),
            "relationship_volatility": relationship_volatility(event, world_before, context),
            "information_gap": information_gap(event, world_before, context),
            "resource_scarcity": resource_scarcity(event, world_before, context),
            "moral_cost": moral_cost(event, world_before, context),
            "irony_density": irony_density(event, world_before, context),
        }

        event.metrics["tension_components"] = components
        event.metrics["tension"] = compute_tension(components, weights)

    # Pass 2 (optional): global normalization
    # Only needed if we want tension relative to the run rather than absolute.
    # For MVP, skip this — absolute [0,1] values from sub-metrics are sufficient.
    # If enabled:
    # all_tensions = [e.metrics["tension"] for e in events]
    # t_min, t_max = min(all_tensions), max(all_tensions)
    # for e in events:
    #     e.metrics["tension"] = (e.metrics["tension"] - t_min) / (t_max - t_min + 1e-9)
```

### MetricsContext Helper

```python
@dataclass
class MetricsContext:
    events: list[Event]
    snapshots: dict[int, WorldState]    # tick_id → WorldState
    secrets: dict[str, Secret]
    index_tables: IndexTables
    max_sim_time: float
    max_recovery_timer: int

    def world_state_before(self, event: Event) -> WorldState:
        """Get world state just before this event by replaying from nearest snapshot."""
        nearest_tick = max(t for t in self.snapshots if t <= event.tick_id)
        state = deepcopy(self.snapshots[nearest_tick])
        # Replay events from snapshot up to (but not including) this event
        for e in self.events:
            if e.tick_id > nearest_tick and e.tick_id < event.tick_id:
                apply_deltas(state, e.deltas)
            elif e.tick_id == event.tick_id and e.order_in_tick < event.order_in_tick:
                apply_deltas(state, e.deltas)
        return state

    def world_state_after(self, event: Event) -> WorldState:
        """Get world state just after this event."""
        state = self.world_state_before(event)
        apply_deltas(state, event.deltas)
        return state

    def agents_at_location(self, location_id: str, world: WorldState) -> list[str]:
        return [a_id for a_id, a in world.agents.items() if a.location == location_id]

    def recent_events_for_agents(self, agents: set[str], n: int) -> list[Event]:
        """Last n events involving any of the given agents."""
        result = []
        for e in reversed(self.events):
            if len(result) >= n:
                break
            event_agents = {e.source_agent} | set(e.target_agents)
            if event_agents & agents:
                result.append(e)
        return result

    def secrets_relevant_to_event(self, event: Event) -> list[str]:
        """Secret IDs relevant to this event's participants."""
        participants = {event.source_agent} | set(event.target_agents)
        return [
            s_id for s_id, s in self.secrets.items()
            if s.about in participants or s.holder in participants
        ]
```

---

## 6. Worked Example: 5 Consecutive Dinner Party Events

### Setup

**Characters present:** Thorne (host), Elena (Thorne's wife), Marcus (Elena's secret lover / embezzler), Lydia (colleague who suspects Marcus), Diana (knows affair, owes Marcus money), Victor (journalist investigating Marcus).

> **Note:** Character names and secrets from canonical `specs/schema/agents.md` and `specs/schema/world.md`.

**Secrets (relevant subset for this example):**
- `secret_affair_01`: Elena and Marcus are having an affair. Truth: true. Holders: Elena, Marcus.
- `secret_embezzle_01`: Marcus is embezzling from the business. Truth: true. Holder: Marcus.
- `secret_diana_debt`: Diana owes Marcus money. Truth: true. Holders: Diana, Marcus.

**Initial belief matrix (relevant subset):**

| Agent | affair_01 | embezzle_01 | diana_debt |
|---|---|---|---|
| Thorne | UNKNOWN | UNKNOWN | UNKNOWN |
| Elena | BELIEVES_TRUE | UNKNOWN | SUSPECTS |
| Marcus | BELIEVES_TRUE | BELIEVES_TRUE | UNKNOWN |
| Lydia | SUSPECTS | SUSPECTS | UNKNOWN |
| Diana | BELIEVES_TRUE | UNKNOWN | BELIEVES_TRUE |
| Victor | UNKNOWN | SUSPECTS | UNKNOWN |

**Weights used:** RELATIONSHIP_DRAMA preset.

### Event Sequence

**Event E042** — tick 35, sim_time 52.5 min
```
Type: CHAT
Source: Lydia → Target: [Marcus]
Location: dining_table
Description: "Lydia asks Marcus about the Hartwell account numbers."
Deltas: []
Causal links: []
```

**Event E043** — tick 36, sim_time 54.0 min
```
Type: OBSERVE
Source: Thorne → Target: []
Location: dining_table
Description: "Thorne notices Marcus and Elena exchanging a glance."
Deltas: [
    {kind: AGENT_EMOTION, agent: "thorne", attribute: "suspicion", op: ADD, value: 0.15}
]
Causal links: []
```

**Event E044** — tick 37, sim_time 55.5 min
```
Type: CONFIDE
Source: Elena → Target: [Marcus]
Location: kitchen
Description: "Elena whispers to Marcus that Thorne seems suspicious tonight."
Deltas: [
    {kind: RELATIONSHIP, agent: "elena", agent_b: "marcus", attribute: "trust", op: ADD, value: 0.1},
    {kind: AGENT_EMOTION, agent: "elena", attribute: "fear", op: ADD, value: 0.2},
]
Causal links: [E043]
```

**Event E045** — tick 38, sim_time 57.0 min
```
Type: LIE
Source: Marcus → Target: [Lydia]
Location: kitchen
Description: "Marcus tells Lydia the account discrepancy was a clerical error, already fixed."
Deltas: [
    {kind: BELIEF, agent: "lydia", agent_b: "marcus", attribute: "secret_embezzle_01", op: SET, value: "believes_false"},
]
Causal links: [E042, E044]
```

**Event E046** — tick 39, sim_time 58.5 min
```
Type: CONFLICT
Source: Diana → Target: [Marcus]
Location: balcony
Description: "Diana confronts Marcus about the money she owes him, asking for more time."
Deltas: [
    {kind: RELATIONSHIP, agent: "diana", agent_b: "marcus", attribute: "trust", op: ADD, value: -0.25},
    {kind: RELATIONSHIP, agent: "marcus", agent_b: "diana", attribute: "trust", op: ADD, value: -0.15},
    {kind: AGENT_EMOTION, agent: "diana", attribute: "shame", op: ADD, value: 0.3},
    {kind: AGENT_EMOTION, agent: "marcus", attribute: "anger", op: ADD, value: 0.2},
    {kind: PACING, agent: "diana", attribute: "stress", op: ADD, value: 0.2},
    {kind: PACING, agent: "marcus", attribute: "dramatic_budget", op: ADD, value: -0.3},
]
Causal links: [E031]
```

### Sub-Metric Computation

#### E042: Lydia asks Marcus about the Hartwell account (CHAT)

| Sub-metric | Value | Reasoning |
|---|---|---|
| danger | 0.00 | CHAT event, no deltas |
| time_pressure | 0.09 | evening is 29% over (52.5/180); 0.29^2 = 0.08 |
| goal_frustration | 0.35 | Lydia has moderate truth_seeking goal; satisfaction is middling since Marcus deflects |
| relationship_volatility | 0.00 | No relationship deltas in this or recent events for Lydia-Marcus pair |
| information_gap | 0.42 | Lydia SUSPECTS embezzle_01 (about Marcus = relevance 1.0); SUSPECTS affair_01 |
| resource_scarcity | 0.00 | Dining table, no resource pressure |
| moral_cost | 0.00 | Innocent question |
| irony_density | 0.35 | Lydia is probing the embezzlement while Marcus BELIEVES_TRUE on it; she only SUSPECTS |

**Tension (relationship_drama weights):**
```
= (0.3*0.00 + 0.5*0.09 + 1.5*0.35 + 2.5*0.00 + 1.0*0.42 + 0.5*0.00 + 2.0*0.00 + 1.5*0.35)
  / (0.3 + 0.5 + 1.5 + 2.5 + 1.0 + 0.5 + 2.0 + 1.5)
= (0 + 0.045 + 0.525 + 0 + 0.42 + 0 + 0 + 0.525) / 9.8
= 1.515 / 9.8
= 0.155
```

**Tension = 0.15** (low -- an innocent question, but the irony and information gap are non-zero).

---

#### E043: Thorne notices the glance (OBSERVE)

| Sub-metric | Value | Reasoning |
|---|---|---|
| danger | 0.00 | OBSERVE, no physical/social harm yet |
| time_pressure | 0.09 | similar evening progression |
| goal_frustration | 0.45 | Thorne's suspicion delta increases fear; secrecy satisfaction drops since he senses something |
| relationship_volatility | 0.10 | Small suspicion delta (0.15 emotion), not relationship but hints at future volatility |
| information_gap | 0.58 | Thorne is UNKNOWN on secret_affair_01 (relevance 1.0 -- it's about his wife). He doesn't know what he saw means. |
| resource_scarcity | 0.00 | No resource pressure |
| moral_cost | 0.00 | Observation has no moral weight |
| irony_density | 0.52 | Thorne is UNKNOWN on secret_affair_01 while watching the two conspirators. High irony. |

**Tension (relationship_drama):**
```
= (0.3*0.00 + 0.5*0.09 + 1.5*0.45 + 2.5*0.10 + 1.0*0.58 + 0.5*0.00 + 2.0*0.00 + 1.5*0.52)
  / 9.8
= (0 + 0.045 + 0.675 + 0.25 + 0.58 + 0 + 0 + 0.78) / 9.8
= 2.33 / 9.8
= 0.238
```

**Tension = 0.24** (moderate -- the dramatic irony is building).

---

#### E044: Elena confides in Marcus (CONFIDE)

| Sub-metric | Value | Reasoning |
|---|---|---|
| danger | 0.15 | Social danger -- confiding in the kitchen could be overheard (kitchen has overhear_from adjacency) |
| time_pressure | 0.12 | Rising fear in Elena (+0.2) acts as internal pressure |
| goal_frustration | 0.55 | Elena wants secrecy (high secrecy goal = 0.9) but senses it slipping; frustration high |
| relationship_volatility | 0.20 | Trust delta (+0.1) between Elena-Marcus; recent events building |
| information_gap | 0.55 | Elena BELIEVES_TRUE on affair (correct), but is UNKNOWN on embezzle_01 (Marcus's other secret). Main gap: audience knows Thorne is getting suspicious, Elena only suspects. |
| resource_scarcity | 0.60 | Privacy scarcity: CONFIDE in kitchen (privacy ~0.5, threshold crossed) |
| moral_cost | 0.25 | Confiding itself isn't costly, but Elena is deepening the conspiracy |
| irony_density | 0.48 | Elena doesn't know Thorne is already suspicious (she only worries; audience saw E043) |

**Tension (relationship_drama):**
```
= (0.3*0.15 + 0.5*0.12 + 1.5*0.55 + 2.5*0.20 + 1.0*0.55 + 0.5*0.60 + 2.0*0.25 + 1.5*0.48)
  / 9.8
= (0.045 + 0.06 + 0.825 + 0.50 + 0.55 + 0.30 + 0.50 + 0.72) / 9.8
= 3.50 / 9.8
= 0.357
```

**Tension = 0.36** (rising -- secrecy under pressure, privacy scarce).

---

#### E045: Marcus lies to Lydia about the accounts (LIE)

| Sub-metric | Value | Reasoning |
|---|---|---|
| danger | 0.20 | Social danger: if the lie is discovered, Marcus's financial fraud is exposed |
| time_pressure | 0.10 | Evening progression, no special urgency |
| goal_frustration | 0.50 | Marcus wants secrecy (high) but Lydia's probing threatens it; tension between secrecy and maintaining trust |
| relationship_volatility | 0.15 | Lydia's belief just changed (SET to believes_false on embezzle_01) |
| information_gap | 0.72 | Lydia now BELIEVES_FALSE on embezzle_01 (truth = true). Maximum ironic gap. |
| resource_scarcity | 0.30 | In kitchen, moderate privacy |
| moral_cost | 0.65 | LIE to Lydia who is a colleague. Trust_from_target is moderate (~0.5). |
| irony_density | 0.70 | Lydia actively believes the wrong thing about the embezzlement. Marcus knows the truth. Peak irony for this dyad. |

**Tension (relationship_drama):**
```
= (0.3*0.20 + 0.5*0.10 + 1.5*0.50 + 2.5*0.15 + 1.0*0.72 + 0.5*0.30 + 2.0*0.65 + 1.5*0.70)
  / 9.8
= (0.06 + 0.05 + 0.75 + 0.375 + 0.72 + 0.15 + 1.30 + 1.05) / 9.8
= 4.455 / 9.8
= 0.455
```

**Tension = 0.45** (elevated -- the lie generates moral cost and irony).

---

#### E046: Diana confronts Marcus about her debt (CONFLICT)

| Sub-metric | Value | Reasoning |
|---|---|---|
| danger | 0.80 | CONFLICT event type (0.8 base); trust dropping sharply (-0.25, -0.15) |
| time_pressure | 0.15 | Evening 32% over; Diana's stress is rising (pacing delta) |
| goal_frustration | 0.70 | Diana's safety and autonomy goals are threatened by the debt; Marcus's secrecy goal frustrated by the confrontation |
| relationship_volatility | 0.80 | Two trust deltas (-0.25, -0.15); bidirectional damage; recent events had minimal volatility so this is a spike |
| information_gap | 0.50 | Both know about the debt; but others don't know about diana_debt -- gap remains |
| resource_scarcity | 0.55 | Balcony is semi-private but adjacent to dining room -- could be overheard |
| moral_cost | 0.45 | Diana pays a moral cost for confrontation (breaking social norms); the debt constrains her loyalty to Elena |
| irony_density | 0.40 | Moderate -- Diana knows about the affair but can't act because of the debt. Marcus doesn't know Victor is investigating him. Multiple unknowns. |

**Tension (relationship_drama):**
```
= (0.3*0.80 + 0.5*0.15 + 1.5*0.70 + 2.5*0.80 + 1.0*0.50 + 0.5*0.55 + 2.0*0.45 + 1.5*0.40)
  / 9.8
= (0.24 + 0.075 + 1.05 + 2.00 + 0.50 + 0.275 + 0.90 + 0.60) / 9.8
= 5.64 / 9.8
= 0.576
```

**Tension = 0.58** (high -- open conflict with bidirectional trust damage).

### Summary Table

| Event | Type | Tension (Rel. Drama) | Highest Sub-metric |
|---|---|---|---|
| E042 | CHAT | 0.15 | information_gap (0.42) |
| E043 | OBSERVE | 0.24 | information_gap (0.58) |
| E044 | CONFIDE | 0.36 | resource_scarcity (0.60) |
| E045 | LIE | 0.45 | information_gap (0.72) |
| E046 | CONFLICT | 0.58 | relationship_volatility (0.80) |

The tension curve across these 5 events shows a clear rising trajectory: 0.15 → 0.24 → 0.36 → 0.45 → 0.58. The dominant driver shifts from information_gap (suspense from secrets) to relationship_volatility (open conflict). This matches the expected dramatic pattern: quiet observation builds to confrontation.

### Same Events Under Thriller Weights

For comparison, the same 5 events scored with THRILLER_WEIGHTS:

| Event | Tension (Thriller) | Highest Sub-metric |
|---|---|---|
| E042 | 0.13 | information_gap (0.42) |
| E043 | 0.19 | information_gap (0.58) |
| E044 | 0.22 | resource_scarcity (0.60) |
| E045 | 0.30 | information_gap (0.72) |
| E046 | 0.55 | danger (0.80) |

The thriller lens compresses the middle events (less weight on moral_cost and relationship_volatility) but spikes sharply on E046 due to the danger weight. The shape is the same (rising) but flatter in the middle -- a thriller would need physical danger to sustain tension between E042-E045.

---

## 7. Edge Cases

### Events with no participants (system events)
If a future scenario includes environment events (weather, time passage), all agent-dependent sub-metrics return 0.0. Only `time_pressure` (from evening progression) and `resource_scarcity` (from environment changes) would be nonzero.

### Events with all agents present
When all 6 agents share a location (dining table at the start), irony_density can spike because every agent's gaps contribute. This is correct: a dinner table scene with everyone present and multiple active secrets should feel tense.

### Zero-weight sub-metrics
If a user sets a weight to 0.0, that sub-metric is excluded from the weighted mean. The denominator decreases accordingly. All weights at 0.0 returns tension = 0.0.

### Very short runs (<10 events)
The lookback window for relationship_volatility (5 events) may not have enough data. The function handles this gracefully by returning 0.0 when no deltas are found.

---

## 8. NOT In Scope

- **Counterfactual impact (significance):** Computed separately via branch simulation. See Decision #9 in doc3.md. Not part of the tension pipeline.
- **Thematic shift computation:** Computed by a separate pipeline. The `irony_density` sub-metric reads from the belief matrix but does not compute thematic axes.
- **Scene-level tension arcs:** Computed by the scene segmentation pipeline after per-event tension is available. See `specs/metrics/scene-segmentation.md`.
- **Real-time tension during simulation:** The pipeline is post-hoc. For real-time display during simulation, a lightweight approximation (last 3 sub-metrics cached, others estimated) could be added later.
- **Custom user sub-metrics:** Users can adjust weights but cannot add new sub-metrics in MVP. Extensibility point for post-MVP.

---

## 9. Dependencies

| Depends On | What It Provides |
|---|---|
| `specs/schema/events.md` | Event schema, EventType, StateDelta, DeltaKind |
| `specs/schema/agents.md` | AgentState, GoalVector, PacingState, BeliefState |
| `specs/schema/world.md` | Location (privacy, adjacency, overhear_from) |
| `specs/metrics/irony-and-beliefs.md` | Full irony computation; irony_density sub-metric is a simplified version |

| Depended On By | What It Consumes |
|---|---|
| `specs/metrics/scene-segmentation.md` | Per-event tension values for beat boundary detection |
| `specs/metrics/story-extraction.md` | Tension for arc scoring (tension variance, peak tension) |
| `specs/visualization/renderer-architecture.md` | Tension values for heatmap rendering |
| `specs/integration/data-flow.md` | TensionPayload format for interface contracts |
