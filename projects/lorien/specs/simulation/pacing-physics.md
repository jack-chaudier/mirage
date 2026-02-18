# Pacing Physics Specification

> **Status:** Draft
> **Author:** sim-designer
> **Dependencies:** specs/schema/agents.md (#2) — for PacingState field names
> **Dependents:** specs/simulation/tick-loop.md (#5), specs/simulation/decision-engine.md (#6)
> **Doc3 Decisions:** #5 (Pacing Physics), #10 (Two-Parameter Catastrophe)

---

## 1. Purpose

The pacing system prevents the simulation from degenerating into either constant melodrama or eternal small talk. It enforces the tension-release-tension rhythm that distinguishes narrative from noise.

**Core principle:** Agents are not free to act dramatically whenever they want. Dramatic actions are *expensive* — they drain a resource that recovers slowly. This creates natural escalation (stress builds), natural cooldowns (budget depletes), and natural catastrophes (suppressed stress eventually breaks through).

**NOT in scope:**
- Tension *metrics* computed post-hoc by the metrics pipeline. Pacing state is the *simulation-side* drama regulator; tension scores are the *analytics-side* summary. They are related but distinct.
- Agent decision scoring formulas (see decision-engine.md). This spec defines the pacing *state* and its *update rules*; the decision engine reads pacing state as input to action scoring.

---

## 2. State Variables

Each agent carries a `PacingState` dataclass updated every tick.

```python
@dataclass
class PacingState:
    dramatic_budget: float = 1.0     # [0.0, 1.0] — resource spent on dramatic actions
    stress: float = 0.0              # [0.0, 1.0] — accumulated conflict exposure
    composure: float = 1.0           # [0.0, 1.0] — ability to mask/suppress true state
    commitment: float = 0.0          # [0.0, 1.0] — investment in current path (catastrophe param 2)
    recovery_timer: int = 0          # [0, ...] — ticks until dramatic actions re-allowed
    suppression_count: int = 0       # [0, ...] — consecutive ticks of suppressed high stress
```

### 2.1 Variable Semantics

| Variable | What It Represents | Increases When | Decreases When |
|---|---|---|---|
| `dramatic_budget` | Capacity for dramatic action | Quiet ticks pass (recharge) | Agent performs a dramatic action (drain) |
| `stress` | Accumulated emotional pressure | Agent witnesses/participates in conflict, receives bad news, is lied to, overhears secrets | Calm ticks pass (slow decay), agent leaves to private space |
| `composure` | Social mask / self-control | Rest in private space (slow recovery), time away from stressors | Alcohol consumption, prolonged high stress, exhaustion |
| `commitment` | Irreversible investment in a course of action | Agent makes public statements, confronts someone, reveals a secret, takes a side | Rarely — commitment is a ratchet (see Section 4.4) |
| `recovery_timer` | Cooldown after dramatic action | Set to N after a dramatic event fires | Decrements by 1 each tick (floor 0) |
| `suppression_count` | How long agent has been "holding it in" | Increments each tick where stress > STRESS_HIGH_THRESHOLD and agent did NOT act dramatically | Resets to 0 when agent performs any dramatic action or catastrophe fires |

---

## 3. Constants and Thresholds

All constants are defined centrally for tuning. Values below are defaults for the Dinner Party scenario.

```python
@dataclass
class PacingConstants:
    # --- Dramatic Budget ---
    BUDGET_RECHARGE_RATE: float = 0.08       # per quiet tick
    BUDGET_RECHARGE_BONUS_PRIVATE: float = 0.04  # extra recharge in private location
    BUDGET_COST_MINOR: float = 0.15          # CONFIDE, minor REVEAL
    BUDGET_COST_MAJOR: float = 0.35          # CONFLICT, LIE, major REVEAL
    BUDGET_COST_CATASTROPHE: float = 0.50    # CATASTROPHE events
    BUDGET_MINIMUM_FOR_ACTION: float = 0.20  # can't initiate dramatic action below this

    # --- Stress ---
    STRESS_GAIN_DIRECT: float = 0.12         # participating in conflict
    STRESS_GAIN_WITNESS: float = 0.05        # witnessing conflict in same location
    STRESS_GAIN_OVERHEAR: float = 0.03       # overhearing from adjacent location
    STRESS_GAIN_SECRET_LEARNED: float = 0.08 # learning a disturbing secret
    STRESS_GAIN_LIED_TO: float = 0.10        # discovering you were lied to
    STRESS_GAIN_BETRAYAL: float = 0.15       # trust broken by someone close
    STRESS_DECAY_RATE: float = 0.03          # per calm tick
    STRESS_DECAY_PRIVATE_BONUS: float = 0.02 # extra decay in private location
    STRESS_HIGH_THRESHOLD: float = 0.60      # above this, suppression tracking begins

    # --- Composure ---
    COMPOSURE_ALCOHOL_PENALTY: float = 0.06  # per drink consumed
    COMPOSURE_STRESS_EROSION: float = 0.02   # per tick while stress > 0.5
    COMPOSURE_RECOVERY_RATE: float = 0.01    # per tick in private space, stress < 0.3
    COMPOSURE_MIN_FOR_MASKING: float = 0.40  # below this, agent can't suppress in public
    COMPOSURE_FLOOR: float = 0.05            # never reaches exactly 0

    # --- Commitment ---
    COMMITMENT_GAIN_PUBLIC_STATEMENT: float = 0.10  # saying something in front of others
    COMMITMENT_GAIN_CONFRONTATION: float = 0.15     # directly confronting someone
    COMMITMENT_GAIN_REVEAL_SECRET: float = 0.20     # revealing a secret (can't un-tell)
    COMMITMENT_GAIN_TAKE_SIDE: float = 0.12         # publicly supporting one party
    COMMITMENT_DECAY_RATE: float = 0.01              # per tick (very slow — ratchet behavior)
    COMMITMENT_DECAY_BLOCKED_ABOVE: float = 0.50    # no decay above this threshold

    # --- Recovery Timer ---
    RECOVERY_TICKS_MINOR: int = 2            # cooldown after minor dramatic action
    RECOVERY_TICKS_MAJOR: int = 4            # cooldown after major dramatic action
    RECOVERY_TICKS_CATASTROPHE: int = 6      # cooldown after catastrophe

    # --- Catastrophe ---
    CATASTROPHE_POTENTIAL_THRESHOLD: float = 0.35  # stress * commitment^2 must exceed this
    CATASTROPHE_COMPOSURE_GATE: float = 0.30       # composure must be BELOW this
    CATASTROPHE_SUPPRESSION_BONUS: float = 0.03    # added to potential per suppression_count tick
    CATASTROPHE_COOLDOWN_TICKS: int = 8            # minimum ticks between catastrophes for same agent

    # --- Hysteresis ---
    TRUST_REPAIR_MULTIPLIER: float = 3.0     # repairing trust costs 3x breaking it
    COMPOSURE_REBUILD_AFTER_CATASTROPHE: float = 0.30  # composure resets to this after catastrophe

    # --- Social Masking ---
    PUBLIC_PRIVACY_THRESHOLD: float = 0.3    # location.privacy below this = "public"
    MASKING_STRESS_SUPPRESSION: float = 0.5  # in public, dramatic action utility multiplied by this
```

---

## 4. Update Rules

Pacing state is updated at the END of each tick, AFTER all events for that tick have been resolved and deltas applied. The update function receives the agent's current state, the list of events from this tick, and the location metadata.

### 4.1 Dramatic Budget

```python
def update_dramatic_budget(agent: AgentState, tick_events: list[Event], location: Location) -> float:
    budget = agent.pacing.dramatic_budget

    # --- Drain: did the agent perform dramatic actions this tick? ---
    for event in tick_events:
        if event.source_agent != agent.id:
            continue
        if event.type in (EventType.CONFLICT,):
            budget -= BUDGET_COST_MAJOR
        elif event.type in (EventType.CONFIDE, EventType.REVEAL):
            # Minor vs major based on secret significance
            if is_major_reveal(event):
                budget -= BUDGET_COST_MAJOR
            else:
                budget -= BUDGET_COST_MINOR
        elif event.type == EventType.LIE:
            budget -= BUDGET_COST_MAJOR
        elif event.type == EventType.CATASTROPHE:
            budget -= BUDGET_COST_CATASTROPHE

    # --- Recharge: quiet ticks replenish ---
    agent_acted_dramatically = any(
        e.source_agent == agent.id and e.type in DRAMATIC_EVENT_TYPES
        for e in tick_events
    )
    if not agent_acted_dramatically:
        budget += BUDGET_RECHARGE_RATE
        if location.privacy >= PUBLIC_PRIVACY_THRESHOLD:
            budget += BUDGET_RECHARGE_BONUS_PRIVATE

    return clamp(budget, 0.0, 1.0)
```

**Dramatic event types** (for budget and recovery purposes):
```python
DRAMATIC_EVENT_TYPES = {
    EventType.CONFLICT,
    EventType.REVEAL,
    EventType.CONFIDE,
    EventType.LIE,
    EventType.CATASTROPHE,
}
```

**Design note:** CHAT, OBSERVE, SOCIAL_MOVE, INTERNAL, and PHYSICAL are non-dramatic. An agent can chat, move rooms, eat, drink, or think without spending budget.

### 4.2 Stress

```python
def update_stress(agent: AgentState, tick_events: list[Event], location: Location) -> float:
    stress = agent.pacing.stress

    for event in tick_events:
        # Direct participation in conflict
        if event.type == EventType.CONFLICT and agent.id in event.participants():
            stress += STRESS_GAIN_DIRECT

        # Witnessing conflict in same location
        elif event.type == EventType.CONFLICT and event.location_id == agent.location:
            stress += STRESS_GAIN_WITNESS

        # Overhearing conflict from adjacent location
        elif (event.type == EventType.CONFLICT
              and event.location_id in location.overhear_from):
            stress += STRESS_GAIN_OVERHEAR

        # Learning a secret (about self or someone close)
        if event.type in (EventType.REVEAL, EventType.CONFIDE):
            if agent.id in event.target_agents:
                secret = get_secret_from_event(event)
                if secret and is_disturbing_to(agent, secret):
                    stress += STRESS_GAIN_SECRET_LEARNED

        # Discovering a lie
        if is_lie_discovery_event(event, agent):
            stress += STRESS_GAIN_LIED_TO

        # Betrayal (trust broken by someone with trust > 0.5)
        if is_betrayal_event(event, agent):
            stress += STRESS_GAIN_BETRAYAL

    # --- Decay: calm ticks reduce stress ---
    agent_in_conflict = any(
        e.type in (EventType.CONFLICT, EventType.CATASTROPHE)
        and agent.id in e.participants()
        for e in tick_events
    )
    if not agent_in_conflict:
        stress -= STRESS_DECAY_RATE
        if location.privacy >= PUBLIC_PRIVACY_THRESHOLD:
            stress -= STRESS_DECAY_PRIVATE_BONUS

    return clamp(stress, 0.0, 1.0)
```

**Key behavior:** Stress accumulates from BOTH direct and indirect exposure. Witnessing someone else's fight still raises your stress. This means the dinner party as a whole trends toward higher stress over time, even for bystanders. Stress decays slowly — it takes many calm ticks to fully de-stress.

### 4.3 Composure

```python
def update_composure(agent: AgentState, tick_events: list[Event], location: Location) -> float:
    composure = agent.pacing.composure

    # --- Alcohol degrades composure ---
    drinks_this_tick = count_drinks(agent, tick_events)
    composure -= COMPOSURE_ALCOHOL_PENALTY * drinks_this_tick

    # --- Sustained stress erodes composure ---
    if agent.pacing.stress > 0.5:
        composure -= COMPOSURE_STRESS_EROSION

    # --- Recovery in private, low-stress situations ---
    if (location.privacy >= PUBLIC_PRIVACY_THRESHOLD
            and agent.pacing.stress < 0.3
            and not any_conflict_this_tick(agent, tick_events)):
        composure += COMPOSURE_RECOVERY_RATE

    return clamp(composure, COMPOSURE_FLOOR, 1.0)
```

**Key behavior:** Composure is the "slow fuse." It degrades from alcohol and sustained stress but recovers only in safe conditions. At the dinner party, alcohol is flowing and stress is accumulating, so composure tends to ratchet downward over the evening. This creates a ticking clock: early in the evening, agents can mask their feelings. Later, they can't.

**Social masking rule:** When `composure >= COMPOSURE_MIN_FOR_MASKING` and `location.privacy < PUBLIC_PRIVACY_THRESHOLD` (i.e., public space), the agent's dramatic action utility is multiplied by `MASKING_STRESS_SUPPRESSION` (0.5). This means agents are half as likely to start drama in public — but once composure drops below 0.40, the mask cracks and they can't help themselves.

### 4.4 Commitment

```python
def update_commitment(agent: AgentState, tick_events: list[Event]) -> float:
    commitment = agent.pacing.commitment

    for event in tick_events:
        if event.source_agent != agent.id:
            continue

        # Public statements lock you in
        if event.type == EventType.CHAT and is_public_location(event.location_id):
            if contains_opinion_or_position(event):
                commitment += COMMITMENT_GAIN_PUBLIC_STATEMENT

        # Confrontation commits you to a stance
        if event.type == EventType.CONFLICT:
            commitment += COMMITMENT_GAIN_CONFRONTATION

        # Revealing a secret is irreversible
        if event.type == EventType.REVEAL:
            commitment += COMMITMENT_GAIN_REVEAL_SECRET

        # Taking sides in someone else's conflict
        if event.type in (EventType.CHAT, EventType.CONFLICT):
            if is_taking_sides(event):
                commitment += COMMITMENT_GAIN_TAKE_SIDE

    # --- Slow decay (ratchet behavior) ---
    # Commitment barely decays, and above 0.50 it doesn't decay at all
    if commitment <= COMMITMENT_DECAY_BLOCKED_ABOVE:
        commitment -= COMMITMENT_DECAY_RATE

    return clamp(commitment, 0.0, 1.0)
```

**Key behavior:** Commitment is a ratchet. It goes up easily but comes down very slowly, and above 0.50 it doesn't decay at all. This models the psychological reality that once you've publicly committed to a position, backed someone in an argument, or revealed a secret, you can't easily walk it back. The dinner party naturally drives commitment upward: people make toasts, express opinions, take sides in conversations.

**Why commitment matters for catastrophe:** High commitment means the agent CAN'T withdraw. They've said too much, promised too much, invested too much. Combined with high stress, this creates the "trapped animal" condition that triggers catastrophic breaks.

### 4.5 Recovery Timer

```python
def update_recovery_timer(agent: AgentState, tick_events: list[Event]) -> int:
    timer = agent.pacing.recovery_timer

    # Set timer based on what the agent did this tick
    for event in tick_events:
        if event.source_agent != agent.id:
            continue
        if event.type == EventType.CATASTROPHE:
            timer = max(timer, RECOVERY_TICKS_CATASTROPHE)
        elif event.type in (EventType.CONFLICT, EventType.LIE):
            timer = max(timer, RECOVERY_TICKS_MAJOR)
        elif event.type in (EventType.REVEAL, EventType.CONFIDE):
            timer = max(timer, RECOVERY_TICKS_MINOR)

    # Decrement
    if timer > 0:
        timer -= 1

    return timer
```

**Key behavior:** After a dramatic action, the agent is on cooldown. During cooldown, the decision engine will NOT select dramatic actions (the recovery_timer check happens before action scoring). This creates natural "breathing room" — after a confrontation, the involved agents default to non-dramatic behavior for a few ticks, allowing other characters to react and the audience to process.

### 4.6 Suppression Count

```python
def update_suppression_count(agent: AgentState, tick_events: list[Event]) -> int:
    count = agent.pacing.suppression_count

    acted_dramatically = any(
        e.source_agent == agent.id and e.type in DRAMATIC_EVENT_TYPES
        for e in tick_events
    )

    if acted_dramatically:
        # Reset — agent released pressure
        return 0

    if agent.pacing.stress >= STRESS_HIGH_THRESHOLD:
        # Agent is stressed but held it together this tick
        return count + 1

    # Stress is below threshold — no suppression happening
    return count
```

**Key behavior:** Suppression count tracks how many ticks an agent has been "holding it in." Each tick of suppression adds a small bonus to catastrophe potential (see Section 5). An agent who has been suppressing for 10 ticks is a pressure cooker — the catastrophe threshold effectively gets lower with each tick of suppression.

---

## 5. The Two-Parameter Catastrophe

> **Doc3 Decision #10:** Implement a genuine cusp catastrophe with two control parameters (stress and commitment).

### 5.1 The Cusp Model

The catastrophe is NOT a simple threshold trigger. It's a two-parameter system where the interaction between stress and commitment creates qualitatively different behavior:

| Stress | Commitment | Behavior |
|---|---|---|
| Low | Low | **Stable equilibrium.** Polite behavior. Agent follows social norms. |
| High | Low | **Gradual withdrawal.** Agent disengages — moves to another room, falls silent, avoids eye contact. No sudden break. |
| Low | High | **Stable loyalty.** Agent is deeply invested but not pressured. Maintains course confidently. |
| **High** | **High** | **CATASTROPHE ZONE.** Agent can't withdraw (too committed) and can't endure (too stressed). Sudden discontinuous break. |

This matches real human behavior: you only snap when you're BOTH trapped AND pressured. Pressure alone lets you flee. Entrapment alone is endurable. Together, they create the cliff.

### 5.2 Catastrophe Potential Formula

```python
def catastrophe_potential(agent: AgentState) -> float:
    """
    Compute the agent's catastrophe potential.
    This is checked every tick. If it exceeds the threshold
    AND composure is below the gate, a catastrophe fires.
    """
    base = agent.pacing.stress * (agent.pacing.commitment ** 2)
    suppression_bonus = agent.pacing.suppression_count * CATASTROPHE_SUPPRESSION_BONUS
    return base + suppression_bonus
```

**Why `commitment ** 2`?** The square creates the cusp geometry. At low commitment, even high stress produces a small potential (0.8 * 0.2^2 = 0.032 — far below threshold). At high commitment, moderate stress is enough (0.5 * 0.8^2 = 0.32 — near threshold). This means catastrophes are RARE without high commitment, which prevents random blowups.

**Why suppression bonus?** An agent who has been suppressing for many ticks is more volatile. After 10 ticks of suppression, the bonus is 0.30, which can push a borderline agent over the edge. This creates the "last straw" dynamic — the trivial event that finally breaks the dam.

### 5.3 Catastrophe Check

```python
def check_catastrophe(agent: AgentState) -> bool:
    """Called every tick. Returns True if a catastrophe event should fire."""
    # Cooldown check: no rapid-fire catastrophes
    if agent.pacing.recovery_timer > 0:
        return False

    potential = catastrophe_potential(agent)

    # Both conditions must be met
    if (potential >= CATASTROPHE_POTENTIAL_THRESHOLD
            and agent.pacing.composure < CATASTROPHE_COMPOSURE_GATE):
        return True

    return False
```

**The composure gate:** Even if catastrophe potential is sky-high, an agent with composure >= 0.30 can still hold it together. This means early in the evening (composure high), catastrophes are nearly impossible. As alcohol and stress erode composure, the gate opens. The composure gate is what makes catastrophes happen LATE in the evening, not randomly throughout.

### 5.4 Catastrophe Event Generation

When `check_catastrophe()` returns True, the simulation generates an involuntary CATASTROPHE event. The agent does NOT choose this — it happens TO them. The type of catastrophe depends on the agent's personality:

```python
def generate_catastrophe(agent: AgentState, world: WorldState) -> Event:
    """
    Generate an involuntary catastrophe event.
    The catastrophe type is determined by the agent's dominant flaw
    and current emotional state.
    """
    catastrophe_subtype = select_catastrophe_type(agent)

    # Generate the event
    event = Event(
        id=generate_id(),
        sim_time=world.sim_time,
        tick_id=world.tick_id,
        order_in_tick=CATASTROPHE_ORDER,  # catastrophes resolve LAST in a tick
        type=EventType.CATASTROPHE,
        source_agent=agent.id,
        target_agents=determine_catastrophe_targets(agent, catastrophe_subtype, world),
        location_id=agent.location,
        causal_links=gather_stress_sources(agent),  # what events caused this
        deltas=generate_catastrophe_deltas(agent, catastrophe_subtype, world),
        description=generate_catastrophe_description(agent, catastrophe_subtype),
    )

    return event


def select_catastrophe_type(agent: AgentState) -> str:
    """
    The catastrophe type depends on the agent's dominant flaw and emotional peak.
    """
    dominant_flaw = max(agent.flaws, key=lambda f: f.strength)
    peak_emotion = max(agent.emotional_state, key=agent.emotional_state.get)

    # Mapping from (flaw, emotion) to catastrophe subtype
    if dominant_flaw.flaw_type == "pride" or peak_emotion == "anger":
        return "explosion"          # Shouts, insults, throws something, flips table
    elif dominant_flaw.flaw_type == "loyalty" or peak_emotion == "shame":
        return "breakdown"          # Crying, fleeing the room, confession under duress
    elif dominant_flaw.flaw_type == "ambition" or peak_emotion == "fear":
        return "desperate_gambit"   # Reckless accusation, blurted secret, power play
    elif dominant_flaw.flaw_type == "avoidance" or peak_emotion == "fear":
        return "flight"             # Agent abruptly leaves (the room or the party)
    elif peak_emotion == "suspicion":
        return "accusation"         # Public accusation, demands answers
    else:
        return "breakdown"          # Default: emotional breakdown
```

### 5.5 Catastrophe Aftermath

After a catastrophe fires, the following state changes occur:

```python
def apply_catastrophe_aftermath(agent: AgentState) -> list[StateDelta]:
    """
    State changes applied to the catastrophe source agent AFTER the event.
    """
    deltas = []

    # Stress partially releases (the pressure valve opened)
    deltas.append(StateDelta(
        kind=DeltaKind.PACING,
        agent=agent.id,
        attribute="stress",
        op=DeltaOp.SET,
        value=agent.pacing.stress * 0.5,  # halved, not zeroed
        reason_code="CATASTROPHE_RELEASE",
    ))

    # Composure crashes — the mask is off
    deltas.append(StateDelta(
        kind=DeltaKind.PACING,
        agent=agent.id,
        attribute="composure",
        op=DeltaOp.SET,
        value=COMPOSURE_REBUILD_AFTER_CATASTROPHE,  # 0.30
        reason_code="CATASTROPHE_COMPOSURE_RESET",
    ))

    # Commitment may increase (you've now REALLY committed)
    # or reset if the catastrophe was flight
    if agent.pacing.commitment < 0.80:
        deltas.append(StateDelta(
            kind=DeltaKind.PACING,
            agent=agent.id,
            attribute="commitment",
            op=DeltaOp.ADD,
            value=0.10,
            reason_code="CATASTROPHE_DEEPENED_COMMITMENT",
        ))

    # Recovery timer set
    deltas.append(StateDelta(
        kind=DeltaKind.PACING,
        agent=agent.id,
        attribute="recovery_timer",
        op=DeltaOp.SET,
        value=CATASTROPHE_COOLDOWN_TICKS,  # 8
        reason_code="CATASTROPHE_COOLDOWN",
    ))

    # Suppression count resets
    deltas.append(StateDelta(
        kind=DeltaKind.PACING,
        agent=agent.id,
        attribute="suppression_count",
        op=DeltaOp.SET,
        value=0,
        reason_code="CATASTROPHE_SUPPRESSION_RESET",
    ))

    return deltas
```

**Aftermath on bystanders:** Everyone present at the catastrophe location gains stress:
- Direct target(s): `+STRESS_GAIN_DIRECT` (0.12)
- Others in same location: `+STRESS_GAIN_WITNESS` (0.05)
- Adjacent locations (overhearing): `+STRESS_GAIN_OVERHEAR` (0.03)

This cascade effect means one catastrophe can push other agents closer to their own thresholds, potentially triggering a chain reaction (rare but spectacular when it happens).

---

## 6. Hysteresis: Trust Repair Asymmetry

> **Doc3 Decision #5:** "Hysteresis: repairing trust costs 3x more than breaking it."

### 6.1 The Rule

When a relationship delta involves trust:

```python
def apply_trust_delta(agent_a: str, agent_b: str, delta_value: float,
                       current_trust: float) -> float:
    """
    Trust changes are asymmetric: breaking is fast, repairing is slow.
    """
    if delta_value > 0:
        # Repairing trust: apply at 1/3 strength
        effective_delta = delta_value / TRUST_REPAIR_MULTIPLIER
    else:
        # Breaking trust: apply at full strength
        effective_delta = delta_value

    new_trust = clamp(current_trust + effective_delta, -1.0, 1.0)
    return new_trust
```

### 6.2 Narrative Purpose

This creates irreversibility pressure. A single betrayal can destroy trust quickly, but rebuilding it takes 3x as many positive interactions. This means:
- **Early betrayals have lasting consequences.** A lie discovered in the appetizer course poisons the entire evening.
- **Redemption arcs are slow and effortful.** An agent who betrayed someone must work hard to rebuild — and might not succeed in the timeframe of one dinner party.
- **Multiple betrayals are devastating.** Two trust-breaking events in sequence create a hole that's nearly impossible to climb out of in the remaining simulation time.

### 6.3 Composure Rebuild After Catastrophe

Similarly, composure recovery after a catastrophe is penalized:

```python
def composure_after_catastrophe(agent: AgentState) -> float:
    """
    After a catastrophe, composure is set to a low value.
    Subsequent recovery is at half the normal rate for RECOVERY_TICKS_CATASTROPHE ticks.
    """
    return COMPOSURE_REBUILD_AFTER_CATASTROPHE  # 0.30
```

The normal composure recovery rate (0.01/tick in private, low-stress conditions) means it takes ~70 ticks to fully rebuild composure from 0.30 to 1.0 — far longer than the dinner party's typical 60-100 tick runtime. A catastrophe permanently marks the agent for the rest of the evening.

---

## 7. Social Masking

### 7.1 Public vs Private Spaces

Each location has a `privacy` field (0.0 = fully public, 1.0 = fully private). For the dinner party:

| Location | Privacy | Effect |
|---|---|---|
| Dining table | 0.1 | Strongly public — masking enforced |
| Foyer | 0.2 | Semi-public — masking enforced |
| Kitchen | 0.5 | Semi-private — masking relaxed |
| Balcony | 0.7 | Private — masking minimal |
| Bathroom | 0.9 | Very private — no masking |

### 7.2 Masking Mechanics

```python
def masking_modifier(agent: AgentState, location: Location) -> float:
    """
    Returns a multiplier [0.0, 1.0] applied to the utility of dramatic actions.
    Low value = agent suppresses dramatic behavior.
    """
    if location.privacy >= PUBLIC_PRIVACY_THRESHOLD:
        # Private space: no suppression
        return 1.0

    if agent.pacing.composure < COMPOSURE_MIN_FOR_MASKING:
        # Composure too low: can't mask anymore even in public
        return 1.0

    # In public with composure: suppress dramatic action utility
    return MASKING_STRESS_SUPPRESSION  # 0.5
```

### 7.3 Spatial Dynamics

The masking system creates emergent spatial behavior:
- **Agents seek private spaces for confrontation.** When an agent wants to confront someone, the decision engine scores the action higher if they first move to a private location (balcony, kitchen).
- **Public catastrophes are more shocking.** A catastrophe in a public space (dining table) means composure failed entirely — the agent couldn't hold it together despite social pressure. These events are more dramatic because they violate the masking norm.
- **Private spaces become pressure valves.** Agents naturally drift toward the kitchen or balcony when stressed, creating natural "scene cuts" as characters leave and return to the main space.
- **Overhearing creates complications.** The kitchen is semi-private but adjacent to the dining table (overhear_from). A "private" conversation in the kitchen can be overheard by someone at the table, creating information leaks.

---

## 8. Complete Tick Update Sequence

The pacing update happens in a specific order within the tick lifecycle. This order matters because some updates depend on others.

```python
def update_pacing(agent: AgentState, tick_events: list[Event],
                   location: Location) -> PacingState:
    """
    Update all pacing variables for this agent at the end of a tick.
    Called AFTER event resolution and delta application.
    Order: stress -> composure -> commitment -> budget -> recovery -> suppression
    """
    new_pacing = PacingState()

    # 1. Stress first (other updates read it)
    new_pacing.stress = update_stress(agent, tick_events, location)

    # 2. Composure (reads stress)
    new_pacing.composure = update_composure(agent, tick_events, location)

    # 3. Commitment (independent of stress/composure)
    new_pacing.commitment = update_commitment(agent, tick_events)

    # 4. Dramatic budget (reads whether dramatic events occurred)
    new_pacing.dramatic_budget = update_dramatic_budget(agent, tick_events, location)

    # 5. Recovery timer
    new_pacing.recovery_timer = update_recovery_timer(agent, tick_events)

    # 6. Suppression count (reads stress threshold)
    new_pacing.suppression_count = update_suppression_count(agent, tick_events)

    return new_pacing
```

---

## 9. Worked Examples

### 9.1 Example: Slow Burn — Elena's Evening

Elena starts the dinner party with default pacing state.

**Tick 1-5: Appetizer course (calm)**
```
Tick 1: CHAT at dining table. Stress +0, Budget recharge +0.08 → budget=1.0, stress=0.0
Tick 3: OBSERVE — notices Marco and Diana exchanging glances. Stress +0.
Tick 5: CHAT with Marco. Pleasant. Budget=1.0, stress=0.0, composure=1.0, commitment=0.0
```

**Tick 8: Elena learns a secret**
```
CONFIDE: Diana tells Elena (in kitchen, privacy=0.5) that she's been seeing Marco's ex.
- Stress: +0.08 (secret_learned) → stress=0.08
- Budget: -0.15 (CONFIDE received doesn't cost budget — only source pays)
- Commitment: +0.0 (Elena didn't act, just listened)
- Composure: 1.0 (unchanged)
```

**Tick 12-18: Tension builds**
```
Tick 12: CONFLICT between Marco and Victor at table. Elena witnesses.
  - Stress: +0.05 (witness) → stress=0.10 (after decay)
Tick 15: Elena overhears Victor insulting Marco from kitchen.
  - Stress: +0.03 (overhear) → stress=0.10
Tick 18: Elena takes a second glass of wine.
  - Composure: -0.06 → composure=0.88
  - Stress: decayed to ~0.06 by now (calm ticks)
```

**Tick 25: Elena takes a side**
```
CHAT at dining table: Elena publicly agrees with Marco's position.
- Commitment: +0.10 (public statement) + 0.12 (taking sides) → commitment=0.22
- Budget: unchanged (CHAT is not dramatic)
- Stress: 0.04 (decaying)
```

**Tick 30-40: Escalation**
```
Tick 30: CONFLICT — Victor confronts Marco about the business deal. Elena present.
  - Stress: +0.05 (witness) → stress=0.07
Tick 33: Elena defends Marco directly. CONFLICT with Victor.
  - Stress: +0.12 (direct conflict) → stress=0.17
  - Commitment: +0.15 (confrontation) → commitment=0.37
  - Budget: -0.35 (major) → budget=0.65
  - Recovery timer: 4
Tick 34-36: Recovery timer counting down. Elena CHATs calmly.
  - Budget: +0.08/tick → budget=0.89
  - Stress: -0.03/tick → stress=0.08
Tick 38: Elena discovers Victor lied about the contract. REVEAL from Sophia.
  - Stress: +0.10 (lied_to) → stress=0.15
Tick 40: Third glass of wine.
  - Composure: -0.06 → composure=0.76
```

**Tick 45-55: The pressure cooker**
```
Tick 45: CONFLICT — Marco confronts Victor with proof. Elena as witness.
  - Stress: +0.05 → stress=0.17
Tick 48: Elena publicly accuses Victor of fraud. CONFLICT.
  - Stress: +0.12 → stress=0.26
  - Commitment: +0.15 → commitment=0.52 (above decay threshold!)
  - Budget: -0.35 → budget=0.62
  - Recovery timer: 4
Tick 50: Victor retaliates, mocks Elena's naivete. CONFLICT targeting Elena.
  - Stress: +0.12 → stress=0.35
  - (Recovery timer still active — Elena can't act dramatically)
Tick 53: Elena steps away to balcony.
  - Stress decaying: -0.05/tick (calm + private bonus) → stress=0.20
  - Budget recharging: +0.12/tick → budget=0.98
```

**State at tick 55:**
```
dramatic_budget: 0.98
stress: 0.15
composure: 0.70 (eroded by alcohol and stress)
commitment: 0.52 (locked above decay threshold)
recovery_timer: 0
suppression_count: 0

Catastrophe potential: 0.15 * 0.52^2 = 0.041 — far below 0.35 threshold.
Elena is tense but stable. No catastrophe.
```

### 9.2 Example: Catastrophe — Victor's Breaking Point

Victor starts with high ambition (flaw) and enters the dinner party already stressed from a business setback.

**Initial state (modified):**
```
dramatic_budget: 1.0
stress: 0.20 (pre-existing)
composure: 0.85
commitment: 0.15 (has already committed to a position on the deal)
suppression_count: 0
```

**Tick 10-25: Victor debates and commits**
```
Tick 10: Victor makes a toast defending his business decision. Public.
  - Commitment: +0.10 → 0.25
Tick 15: Victor argues with Marco. CONFLICT.
  - Stress: +0.12 → 0.29
  - Commitment: +0.15 → 0.40
  - Budget: -0.35 → 0.65
Tick 20: Victor lies to Sophia about the contract. LIE.
  - Commitment: +0.20 → 0.60 (above decay threshold!)
  - Budget: -0.35 → 0.38
  - Recovery timer: 4
Tick 22-25: Calm ticks. Wine.
  - Budget: +0.08/tick → 0.62
  - Stress: -0.03/tick → 0.20
  - Composure: -0.06 (wine) → 0.73
```

**Tick 30-40: Stress accumulation**
```
Tick 30: Elena accuses Victor publicly. CONFLICT (target).
  - Stress: +0.12 → 0.29
Tick 33: Marco presents evidence of Victor's lie. REVEAL.
  - Stress: +0.08 (disturbing secret about himself exposed) → 0.34
  - This is also a betrayal of his deception → +0.15 → stress=0.46
Tick 35: Victor tries to defend himself. CONFLICT.
  - Stress: +0.12 → 0.55
  - Commitment: +0.15 → 0.75
  - Budget: -0.35 → 0.35
Tick 36-39: Victor suppresses at dining table (public, composure still above 0.40).
  - Masking active: dramatic action utility * 0.5
  - Suppression count: 1, 2, 3, 4
  - Stress: not decaying (still above 0.5, conflict echoes)
  - Actually stress: 0.55, gaining +0.02 composure_stress_erosion
  - Composure: -0.02/tick → 0.73, 0.71, 0.69, 0.67
```

**Tick 40: Fourth glass of wine**
```
  Composure: -0.06 → 0.61
  Stress: 0.52 (slow decay)
  Commitment: 0.75
  Suppression count: 5
```

**Tick 42-48: The final build**
```
Tick 42: Sophia mentions the contract discrepancy again. CHAT (not even conflict!)
  - Stress: +0.05 (contextual — agent perceives it as threatening) → 0.54
  - Suppression count: 7
Tick 44: Another wine.
  - Composure: -0.06 → 0.49
Tick 46: Diana asks Victor directly "Is it true?" CHAT.
  - Stress: +0.08 → 0.59
  - Suppression count: 9
Tick 47: Stress erosion on composure.
  - Composure: -0.02 → 0.47
  - Suppression count: 10
```

**Tick 48: CATASTROPHE CHECK**
```
stress = 0.59
commitment = 0.75
composure = 0.45
suppression_count = 10

catastrophe_potential = 0.59 * 0.75^2 + (10 * 0.03)
                      = 0.59 * 0.5625 + 0.30
                      = 0.332 + 0.30
                      = 0.632

Threshold: 0.35 → EXCEEDED
Composure gate: 0.30 → composure is 0.45, NOT below gate.

NO CATASTROPHE YET. Composure is still too high!
```

**Tick 50: One more wine. Elena asks a pointed question.**
```
  Composure: -0.06 → 0.39
  Stress: +0.05 → 0.61
  Still above composure gate (0.30). Close.

  Composure stress erosion: -0.02 → 0.37
```

**Tick 52: Marco says "Just admit it, Victor."**
```
  Stress: +0.12 (direct conflict) → 0.70
  Composure: -0.02 (stress erosion) → 0.35
  Suppression count: 14

  catastrophe_potential = 0.70 * 0.75^2 + (14 * 0.03)
                        = 0.70 * 0.5625 + 0.42
                        = 0.394 + 0.42
                        = 0.814

  Composure: 0.35 → still above gate (0.30)! Close!
```

**Tick 53: Sophia accidentally knocks over a glass. Startles Victor.**
```
  Composure: -0.02 → 0.33
  (One more erosion tick...)
```

**Tick 54: Elena looks at Victor with visible disgust. OBSERVE event (not even conflict).**
```
  Composure: -0.02 → 0.31
  Stress: 0.67 (slight decay)

  Still above 0.30 gate. One more tick...
```

**Tick 55: Composure erosion ticks again.**
```
  Composure: -0.02 → 0.29

  catastrophe_potential = 0.67 * 0.75^2 + (17 * 0.03)
                        = 0.377 + 0.51
                        = 0.887

  Threshold: 0.35 → EXCEEDED (0.887)
  Composure gate: 0.30 → composure is 0.29 → BELOW GATE

  *** CATASTROPHE FIRES ***
```

**Catastrophe event:**
```
Victor's dominant flaw: ambition. Peak emotion: anger.
→ catastrophe_subtype = "explosion"

Event generated:
  type: CATASTROPHE
  source_agent: "victor"
  target_agents: ["marco", "elena"] (the two who pressured him)
  location: "dining_table"
  description: "Victor slams his fist on the table, knocking over glasses.
                'You have NO IDEA what I've sacrificed for this deal!
                Every one of you would have done the same!'"

Aftermath:
  Victor: stress 0.67 → 0.34 (halved), composure → 0.30, recovery_timer=8,
          suppression_count → 0, commitment += 0.10 → 0.85
  Marco: stress += 0.12 (target)
  Elena: stress += 0.12 (target)
  Others at table: stress += 0.05 (witnesses)
  Diana (in kitchen): stress += 0.03 (overhear)
```

### 9.3 Example: Suppression Without Catastrophe — Sophia's Quiet Evening

Sophia has high composure (personality trait) and avoidance flaw. She witnesses conflict but never commits.

```
Tick 1-55 summary:
  - Witnesses 4 conflicts
  - Stress rises to 0.35
  - Commitment stays at 0.05 (never takes sides, never reveals anything)
  - Composure: 0.72 (only 2 drinks, naturally composed)
  - Suppression count: 8 (stress was above 0.6 briefly)

  catastrophe_potential = 0.35 * 0.05^2 + (8 * 0.03)
                        = 0.35 * 0.0025 + 0.24
                        = 0.001 + 0.24
                        = 0.241

  Threshold: 0.35 → NOT exceeded.
  Composure: 0.72 → Well above gate.

  NO CATASTROPHE. Sophia survives the evening intact.

  Instead: stress=0.35 with low commitment → GRADUAL WITHDRAWAL.
  Sophia quietly slips to the balcony, watches the night, says nothing.
  This is the non-catastrophe high-stress path — withdrawal, not explosion.
```

This demonstrates the cusp model: without commitment, high stress produces quiet retreat, not dramatic explosion.

---

## 10. Pacing State as Decision Engine Input

> Detailed in specs/simulation/decision-engine.md. Summary here for completeness.

The decision engine reads pacing state through these channels:

| Pacing Variable | Effect on Decision |
|---|---|
| `dramatic_budget < BUDGET_MINIMUM_FOR_ACTION` | Agent CANNOT initiate dramatic actions this tick |
| `recovery_timer > 0` | Agent CANNOT initiate dramatic actions this tick |
| `stress > STRESS_HIGH_THRESHOLD` | Increases utility of stress-relief actions (leave room, drink, confide) |
| `composure < COMPOSURE_MIN_FOR_MASKING` | Removes public masking penalty — agent may act dramatically in public |
| `masking_modifier()` | Multiplier on dramatic action utility in public spaces |
| `commitment` | Increases utility of actions consistent with committed path; decreases utility of backtracking |

**The decision engine does NOT directly read:** `catastrophe_potential`, `suppression_count`. These are internal to the catastrophe check. Catastrophes are involuntary — the decision engine never "decides" to have one.

---

## 11. Generating Pacing Deltas

Every pacing state change is recorded as a StateDelta on the event that caused it. This makes pacing changes visible in the event log and auditable.

```python
def generate_pacing_deltas(agent_id: str, old: PacingState, new: PacingState) -> list[StateDelta]:
    """Generate deltas for any pacing state changes this tick."""
    deltas = []
    for field in ["dramatic_budget", "stress", "composure", "commitment",
                   "recovery_timer", "suppression_count"]:
        old_val = getattr(old, field)
        new_val = getattr(new, field)
        if old_val != new_val:
            deltas.append(StateDelta(
                kind=DeltaKind.PACING,
                agent=agent_id,
                attribute=field,
                op=DeltaOp.SET,
                value=new_val,
                reason_code=f"PACING_UPDATE_{field.upper()}",
            ))
    return deltas
```

---

## 12. Tuning Guide

### 12.1 "Too Much Drama" (Melodrama)

If catastrophes happen too frequently:
- Increase `CATASTROPHE_POTENTIAL_THRESHOLD` (e.g., 0.35 → 0.50)
- Decrease `CATASTROPHE_SUPPRESSION_BONUS` (e.g., 0.03 → 0.01)
- Increase `BUDGET_COST_MAJOR` to drain budget faster, forcing longer cooldowns
- Decrease `STRESS_GAIN_*` values across the board

### 12.2 "Too Boring" (Flatline)

If no dramatic events occur:
- Increase initial stress levels for characters with existing conflicts
- Decrease `COMPOSURE_ALCOHOL_PENALTY` threshold sensitivity (more composure loss per drink)
- Increase `COMMITMENT_GAIN_*` values so agents lock in faster
- Decrease `BUDGET_COST_*` values to let agents act more freely

### 12.3 "Drama Bunching" (All At Once)

If all catastrophes happen in the same tick window:
- Increase `CATASTROPHE_COOLDOWN_TICKS` to spread them out
- Add variance to initial pacing states so agents are on different timers
- Ensure character designs have varying stress sensitivity (via flaw system)

### 12.4 "No Spatial Variety" (Everyone Stays at Table)

If agents don't use private spaces:
- Increase `MASKING_STRESS_SUPPRESSION` penalty (e.g., 0.5 → 0.3)
- Ensure the decision engine gives high utility to "move to private space" when stress is high
- Make private space stress decay noticeably faster than public

---

## 13. Edge Cases

### 13.1 Multiple Catastrophes in One Tick

If two agents both pass the catastrophe check in the same tick:
- Both catastrophes fire, ordered by `catastrophe_potential` (highest first).
- The second agent's stress is updated with witness/target bonuses from the first catastrophe BEFORE their catastrophe resolves.
- This can create a "chain reaction" — Agent A explodes, which pushes Agent B over the edge.
- Maximum catastrophes per tick: capped at 2 to prevent degeneracy.

### 13.2 Agent Leaves the Party

If a catastrophe subtype is "flight" and the agent leaves:
- They are removed from all location participant lists.
- Their pacing state continues to update (stress decays, composure recovers).
- They cannot be targeted by other agents' actions.
- They may return after `CATASTROPHE_COOLDOWN_TICKS` if their stress drops below 0.3.

### 13.3 All Agents at Maximum Stress

If the simulation enters a state where all agents are highly stressed:
- The recovery timer and budget system prevent simultaneous action.
- The system will naturally produce a sequence of catastrophes separated by cooldown periods, rather than a single chaotic explosion.
- If ALL agents catastrophe within a 10-tick window, the simulation may end early (the party breaks up).

### 13.4 Zero-Conflict Runs

Possible if character designs don't create goal conflicts. The pacing system allows this — it doesn't *force* drama, it *enables* it. A dinner party where everyone gets along is a valid (if boring) simulation output. The dinner-party-config.md ensures this doesn't happen by designing characters with inherent conflicts.

---

## 14. Relationship to Other Specs

| Spec | Relationship |
|---|---|
| **decision-engine.md** | Reads pacing state to score actions. Masking modifier applied there. |
| **tick-loop.md** | Calls `update_pacing()` at step 4 of each tick. Calls `check_catastrophe()` at step 2. |
| **dinner-party-config.md** | Initial pacing states per character. Tuning constants may be adjusted per scenario. |
| **events.md** | CATASTROPHE EventType defined there. Pacing deltas use DeltaKind.PACING. |
| **agents.md** | PacingState dataclass defined there. This spec defines its update semantics. |
| **tension-pipeline.md** | Pacing state (stress, composure) is INPUT to post-hoc tension computation, but tension is NOT input to pacing updates. No circular dependency. |

---

## 15. Open Questions

1. **Should composure recovery be faster after voluntary dramatic action vs catastrophe?** Current spec treats them the same (composure just follows its update rules). Could differentiate: voluntary action = mild composure cost, catastrophe = severe.

2. **Should commitment decay below 0.5 be faster?** Current: 0.01/tick below 0.5, zero above. Alternative: 0.02/tick below 0.3, 0.01/tick 0.3-0.5, zero above. This would make low-commitment agents "bounce back" to neutral faster.

3. **Per-character pacing constants?** Current spec uses global constants. Could allow per-character overrides (e.g., Victor has STRESS_GAIN_DIRECT = 0.15 instead of 0.12 because he's more volatile). This adds expressiveness at the cost of tuning complexity.
