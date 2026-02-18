from __future__ import annotations

from dataclasses import dataclass

from narrativefield.schema.agents import AgentState, PacingState
from narrativefield.schema.events import DeltaKind, DeltaOp, Event, EventType, StateDelta
from narrativefield.schema.world import Location, SecretDefinition


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class PacingConstants:
    """Pacing physics constants.

    Source of truth: specs/simulation/pacing-physics.md Section 3.
    """

    # --- Dramatic Budget ---
    BUDGET_RECHARGE_RATE: float = 0.08
    BUDGET_RECHARGE_BONUS_PRIVATE: float = 0.04
    BUDGET_COST_MINOR: float = 0.15
    BUDGET_COST_MAJOR: float = 0.35
    BUDGET_COST_CATASTROPHE: float = 0.50
    BUDGET_MINIMUM_FOR_ACTION: float = 0.20

    # --- Stress ---
    STRESS_GAIN_DIRECT: float = 0.12
    STRESS_GAIN_WITNESS: float = 0.05
    STRESS_GAIN_OVERHEAR: float = 0.03
    STRESS_GAIN_SECRET_LEARNED: float = 0.08
    STRESS_GAIN_LIED_TO: float = 0.10
    STRESS_GAIN_BETRAYAL: float = 0.15
    STRESS_DECAY_RATE: float = 0.03
    STRESS_DECAY_PRIVATE_BONUS: float = 0.02
    STRESS_HIGH_THRESHOLD: float = 0.60

    # --- Composure ---
    COMPOSURE_ALCOHOL_PENALTY: float = 0.06
    COMPOSURE_STRESS_EROSION: float = 0.02
    COMPOSURE_RECOVERY_RATE: float = 0.01
    COMPOSURE_MIN_FOR_MASKING: float = 0.40
    COMPOSURE_FLOOR: float = 0.05

    # --- Commitment ---
    COMMITMENT_GAIN_PUBLIC_STATEMENT: float = 0.10
    COMMITMENT_GAIN_CONFRONTATION: float = 0.15
    COMMITMENT_GAIN_REVEAL_SECRET: float = 0.20
    COMMITMENT_GAIN_TAKE_SIDE: float = 0.12
    COMMITMENT_DECAY_RATE: float = 0.01
    COMMITMENT_DECAY_BLOCKED_ABOVE: float = 0.50

    # --- Recovery Timer ---
    RECOVERY_TICKS_MINOR: int = 2
    RECOVERY_TICKS_MAJOR: int = 4
    RECOVERY_TICKS_CATASTROPHE: int = 6

    # --- Catastrophe ---
    CATASTROPHE_POTENTIAL_THRESHOLD: float = 0.35
    CATASTROPHE_COMPOSURE_GATE: float = 0.30
    CATASTROPHE_SUPPRESSION_BONUS: float = 0.03
    CATASTROPHE_COOLDOWN_TICKS: int = 8

    # --- Hysteresis ---
    TRUST_REPAIR_MULTIPLIER: float = 3.0
    COMPOSURE_REBUILD_AFTER_CATASTROPHE: float = 0.30

    # --- Social Masking ---
    PUBLIC_PRIVACY_THRESHOLD: float = 0.3
    MASKING_STRESS_SUPPRESSION: float = 0.5


DEFAULT_CONSTANTS = PacingConstants()


DRAMATIC_EVENT_TYPES: set[EventType] = {
    EventType.CONFLICT,
    EventType.REVEAL,
    EventType.CONFIDE,
    EventType.LIE,
    EventType.CATASTROPHE,
}


def participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def catastrophe_potential(agent: AgentState, c: PacingConstants = DEFAULT_CONSTANTS) -> float:
    """Compute stress * commitment^2 + suppression bonus."""

    base = agent.pacing.stress * (agent.pacing.commitment**2)
    suppression_bonus = agent.pacing.suppression_count * c.CATASTROPHE_SUPPRESSION_BONUS
    return base + suppression_bonus


def check_catastrophe(
    agent: AgentState,
    c: PacingConstants = DEFAULT_CONSTANTS,
    *,
    catastrophe_threshold: float | None = None,
    composure_gate: float | None = None,
) -> bool:
    """Return True if a catastrophe event should fire for this agent this tick."""

    # Cooldown check: no rapid-fire catastrophes.
    if agent.pacing.recovery_timer > 0:
        return False

    potential = catastrophe_potential(agent, c)
    threshold = float(catastrophe_threshold) if catastrophe_threshold is not None else c.CATASTROPHE_POTENTIAL_THRESHOLD
    gate = float(composure_gate) if composure_gate is not None else c.CATASTROPHE_COMPOSURE_GATE
    return potential >= threshold and agent.pacing.composure < gate


def apply_social_masking(
    utility: float, agent: AgentState, location: Location, c: PacingConstants = DEFAULT_CONSTANTS
) -> float:
    """Suppress dramatic utility in public spaces when composure allows masking."""

    if location.privacy < c.PUBLIC_PRIVACY_THRESHOLD and agent.pacing.composure >= c.COMPOSURE_MIN_FOR_MASKING:
        return utility * c.MASKING_STRESS_SUPPRESSION
    return utility


def _is_major_reveal(event: Event, secrets: dict[str, SecretDefinition] | None) -> bool:
    # Heuristic: high-weight secrets count as major.
    if not secrets:
        return False
    secret_id = (event.content_metadata or {}).get("secret_id")
    if not secret_id:
        return False
    secret = secrets.get(str(secret_id))
    if not secret:
        return False
    return secret.dramatic_weight >= 0.8


def update_budget(
    agent: AgentState,
    tick_events: list[Event],
    location: Location,
    *,
    secrets: dict[str, SecretDefinition] | None = None,
    c: PacingConstants = DEFAULT_CONSTANTS,
) -> float:
    """Update dramatic_budget per pacing-physics.md Section 4.1."""

    budget = agent.pacing.dramatic_budget

    for event in tick_events:
        if event.source_agent != agent.id:
            continue
        if event.type == EventType.CONFLICT:
            budget -= c.BUDGET_COST_MAJOR
        elif event.type == EventType.CONFIDE:
            budget -= c.BUDGET_COST_MINOR
        elif event.type == EventType.REVEAL:
            budget -= c.BUDGET_COST_MAJOR if _is_major_reveal(event, secrets) else c.BUDGET_COST_MINOR
        elif event.type == EventType.LIE:
            budget -= c.BUDGET_COST_MAJOR
        elif event.type == EventType.CATASTROPHE:
            budget -= c.BUDGET_COST_CATASTROPHE

    acted_dramatically = any(
        e.source_agent == agent.id and e.type in DRAMATIC_EVENT_TYPES for e in tick_events
    )
    if not acted_dramatically:
        budget += c.BUDGET_RECHARGE_RATE
        if location.privacy >= c.PUBLIC_PRIVACY_THRESHOLD:
            budget += c.BUDGET_RECHARGE_BONUS_PRIVATE

    return clamp(budget, 0.0, 1.0)


def _count_drinks(agent: AgentState, tick_events: list[Event]) -> int:
    # Count *consumption* rather than who performed the physical action.
    # In the tick loop, "pour wine" may increase multiple agents' alcohol_level via deltas.
    drinks = 0
    for event in tick_events:
        for delta in event.deltas:
            if delta.kind != DeltaKind.AGENT_RESOURCE:
                continue
            if delta.agent != agent.id:
                continue
            if delta.attribute != "alcohol_level":
                continue
            if delta.op != DeltaOp.ADD:
                continue
            if float(delta.value) > 0:
                drinks += 1
    return drinks


def update_composure(
    agent: AgentState, tick_events: list[Event], location: Location, *, c: PacingConstants = DEFAULT_CONSTANTS
) -> float:
    """Update composure per pacing-physics.md Section 4.3."""

    composure = agent.pacing.composure
    composure -= c.COMPOSURE_ALCOHOL_PENALTY * _count_drinks(agent, tick_events)

    if agent.pacing.stress > 0.5:
        composure -= c.COMPOSURE_STRESS_EROSION

    any_conflict = any(e.type in (EventType.CONFLICT, EventType.CATASTROPHE) for e in tick_events)
    if location.privacy >= c.PUBLIC_PRIVACY_THRESHOLD and agent.pacing.stress < 0.3 and not any_conflict:
        composure += c.COMPOSURE_RECOVERY_RATE

    return clamp(composure, c.COMPOSURE_FLOOR, 1.0)


def _get_secret_from_event(event: Event, secrets: dict[str, SecretDefinition] | None) -> SecretDefinition | None:
    if not secrets:
        return None
    secret_id = (event.content_metadata or {}).get("secret_id")
    if not secret_id:
        return None
    return secrets.get(str(secret_id))


def _is_disturbing_to(agent: AgentState, secret: SecretDefinition) -> bool:
    # MVP heuristic: secrets about the agent or with high dramatic weight are disturbing.
    if secret.about == agent.id:
        return True
    return secret.dramatic_weight >= 0.5


def update_stress(
    agent: AgentState,
    tick_events: list[Event],
    location: Location,
    *,
    secrets: dict[str, SecretDefinition] | None = None,
    c: PacingConstants = DEFAULT_CONSTANTS,
) -> float:
    """Update stress per pacing-physics.md Section 4.2."""

    stress = agent.pacing.stress

    for event in tick_events:
        # Conflict exposure.
        if event.type == EventType.CONFLICT and agent.id in participants(event):
            stress += c.STRESS_GAIN_DIRECT
        elif event.type == EventType.CONFLICT and event.location_id == agent.location:
            stress += c.STRESS_GAIN_WITNESS
        elif event.type == EventType.CONFLICT and event.location_id in location.overhear_from:
            stress += c.STRESS_GAIN_OVERHEAR

        # Catastrophes are also stressful (mirror conflict behavior).
        if event.type == EventType.CATASTROPHE and agent.id in participants(event):
            stress += c.STRESS_GAIN_DIRECT
        elif event.type == EventType.CATASTROPHE and event.location_id == agent.location:
            stress += c.STRESS_GAIN_WITNESS
        elif event.type == EventType.CATASTROPHE and event.location_id in location.overhear_from:
            stress += c.STRESS_GAIN_OVERHEAR

        # Learning disturbing secrets.
        if event.type in (EventType.REVEAL, EventType.CONFIDE) and agent.id in event.target_agents:
            secret = _get_secret_from_event(event, secrets)
            if secret and _is_disturbing_to(agent, secret):
                stress += c.STRESS_GAIN_SECRET_LEARNED

        # Being lied to is stressful (pacing-physics.md Section 4.2).
        if event.type == EventType.LIE and agent.id in participants(event) and agent.id in event.target_agents:
            stress += c.STRESS_GAIN_LIED_TO

        # Placeholder: betrayal detection is a future refinement.

    agent_in_conflict = any(
        e.type in (EventType.CONFLICT, EventType.CATASTROPHE) and agent.id in participants(e)
        for e in tick_events
    )
    if not agent_in_conflict:
        stress -= c.STRESS_DECAY_RATE
        if location.privacy >= c.PUBLIC_PRIVACY_THRESHOLD:
            stress -= c.STRESS_DECAY_PRIVATE_BONUS

    return clamp(stress, 0.0, 1.0)


def _contains_opinion_or_position(event: Event) -> bool:
    meta = event.content_metadata or {}
    if meta.get("public_statement") is True:
        return True
    return len(event.description) >= 40


def update_commitment(
    agent: AgentState,
    tick_events: list[Event],
    location: Location,
    *,
    c: PacingConstants = DEFAULT_CONSTANTS,
) -> float:
    """Update commitment per pacing-physics.md Section 4.4."""

    commitment = agent.pacing.commitment

    for event in tick_events:
        if event.source_agent != agent.id:
            continue

        if event.type == EventType.CHAT and location.privacy < c.PUBLIC_PRIVACY_THRESHOLD:
            if _contains_opinion_or_position(event):
                commitment += c.COMMITMENT_GAIN_PUBLIC_STATEMENT

        if event.type == EventType.CONFLICT:
            commitment += c.COMMITMENT_GAIN_CONFRONTATION

        if event.type == EventType.REVEAL:
            commitment += c.COMMITMENT_GAIN_REVEAL_SECRET

        # Future: taking-sides detection would allow catastrophe triggers from
        # alliance shifts (not yet implemented in MVP).

    if commitment <= c.COMMITMENT_DECAY_BLOCKED_ABOVE:
        commitment -= c.COMMITMENT_DECAY_RATE

    return clamp(commitment, 0.0, 1.0)


def update_recovery_timer(agent: AgentState, tick_events: list[Event], *, c: PacingConstants = DEFAULT_CONSTANTS) -> int:
    """Update recovery_timer per pacing-physics.md Section 4.5."""

    timer = int(agent.pacing.recovery_timer)

    for event in tick_events:
        if event.source_agent != agent.id:
            continue
        if event.type == EventType.CATASTROPHE:
            timer = max(timer, c.RECOVERY_TICKS_CATASTROPHE)
        elif event.type in (EventType.CONFLICT, EventType.LIE):
            timer = max(timer, c.RECOVERY_TICKS_MAJOR)
        elif event.type in (EventType.REVEAL, EventType.CONFIDE):
            timer = max(timer, c.RECOVERY_TICKS_MINOR)

    if timer > 0:
        timer -= 1
    return max(timer, 0)


def update_suppression_count(
    agent: AgentState, tick_events: list[Event], *, c: PacingConstants = DEFAULT_CONSTANTS
) -> int:
    """Update suppression_count per pacing-physics.md Section 4.6."""

    count = int(agent.pacing.suppression_count)
    acted_dramatically = any(
        e.source_agent == agent.id and e.type in DRAMATIC_EVENT_TYPES for e in tick_events
    )
    if acted_dramatically:
        return 0
    if agent.pacing.stress >= c.STRESS_HIGH_THRESHOLD:
        return count + 1
    return count


def apply_catastrophe_aftermath(agent: AgentState, c: PacingConstants = DEFAULT_CONSTANTS) -> list[StateDelta]:
    """Pacing deltas applied to the catastrophe source agent (pacing-physics.md Section 5.5)."""

    deltas: list[StateDelta] = []

    deltas.append(
        StateDelta(
            kind=DeltaKind.PACING,
            agent=agent.id,
            attribute="stress",
            op=DeltaOp.SET,
            value=float(agent.pacing.stress) * 0.5,
            reason_code="CATASTROPHE_RELEASE",
            reason_display="Stress partially releases after catastrophe.",
        )
    )
    deltas.append(
        StateDelta(
            kind=DeltaKind.PACING,
            agent=agent.id,
            attribute="composure",
            op=DeltaOp.SET,
            value=c.COMPOSURE_REBUILD_AFTER_CATASTROPHE,
            reason_code="CATASTROPHE_COMPOSURE_RESET",
            reason_display="Composure crashes after catastrophe.",
        )
    )
    if agent.pacing.commitment < 0.80:
        deltas.append(
            StateDelta(
                kind=DeltaKind.PACING,
                agent=agent.id,
                attribute="commitment",
                op=DeltaOp.ADD,
                value=0.10,
                reason_code="CATASTROPHE_DEEPENED_COMMITMENT",
                reason_display="Commitment deepens after catastrophe.",
            )
        )
    deltas.append(
        StateDelta(
            kind=DeltaKind.PACING,
            agent=agent.id,
            attribute="recovery_timer",
            op=DeltaOp.SET,
            value=c.CATASTROPHE_COOLDOWN_TICKS,
            reason_code="CATASTROPHE_COOLDOWN",
            reason_display="Cooldown after catastrophe.",
        )
    )
    deltas.append(
        StateDelta(
            kind=DeltaKind.PACING,
            agent=agent.id,
            attribute="suppression_count",
            op=DeltaOp.SET,
            value=0,
            reason_code="CATASTROPHE_SUPPRESSION_RESET",
            reason_display="Suppression resets after catastrophe.",
        )
    )

    return deltas


def end_of_tick_update(
    agent: AgentState,
    tick_events: list[Event],
    location: Location,
    *,
    secrets: dict[str, SecretDefinition] | None = None,
    c: PacingConstants = DEFAULT_CONSTANTS,
) -> PacingState:
    """Compute next pacing state after the tick.

    Note: Catastrophe aftermath is applied as an override so the resulting end-of-tick
    state matches pacing-physics.md Section 5.5 (no decrement on the 8-tick cooldown).
    """

    next_pacing = PacingState(
        dramatic_budget=update_budget(agent, tick_events, location, secrets=secrets, c=c),
        stress=update_stress(agent, tick_events, location, secrets=secrets, c=c),
        composure=update_composure(agent, tick_events, location, c=c),
        commitment=update_commitment(agent, tick_events, location, c=c),
        recovery_timer=update_recovery_timer(agent, tick_events, c=c),
        suppression_count=update_suppression_count(agent, tick_events, c=c),
    )

    if any(e.type == EventType.CATASTROPHE and e.source_agent == agent.id for e in tick_events):
        # Apply aftermath on top of the computed values.
        for delta in apply_catastrophe_aftermath(agent, c=c):
            if delta.kind != DeltaKind.PACING or delta.agent != agent.id:
                continue
            if delta.op == DeltaOp.SET:
                setattr(next_pacing, delta.attribute, delta.value)
            else:  # ADD
                current = getattr(next_pacing, delta.attribute)
                setattr(next_pacing, delta.attribute, current + delta.value)

        next_pacing.dramatic_budget = clamp(next_pacing.dramatic_budget, 0.0, 1.0)
        next_pacing.stress = clamp(next_pacing.stress, 0.0, 1.0)
        next_pacing.composure = clamp(next_pacing.composure, c.COMPOSURE_FLOOR, 1.0)
        next_pacing.commitment = clamp(next_pacing.commitment, 0.0, 1.0)
        next_pacing.recovery_timer = max(int(next_pacing.recovery_timer), 0)
        next_pacing.suppression_count = max(int(next_pacing.suppression_count), 0)

    return next_pacing
