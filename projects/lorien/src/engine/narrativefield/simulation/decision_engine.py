from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional

from narrativefield.schema.agents import AgentState, BeliefState, CharacterFlaw
from narrativefield.schema.events import EventType
from narrativefield.schema.world import Location, SecretDefinition
from narrativefield.simulation import pacing
from narrativefield.simulation.types import Action, PerceivedState, WorldState


@dataclass
class ActionEffects:
    """Estimated impact of an action on each goal dimension (decision-engine.md Section 3.1.1)."""

    safety_impact: float = 0.0
    status_impact: float = 0.0
    closeness_impact: float = 0.0
    secrecy_impact: float = 0.0
    truth_impact: float = 0.0
    autonomy_impact: float = 0.0
    loyalty_impact: float = 0.0


ACTION_EFFECT_PROFILES: dict[EventType, ActionEffects] = {
    EventType.CHAT: ActionEffects(
        safety_impact=0.1,
        status_impact=0.1,
        closeness_impact=0.2,
        secrecy_impact=0.0,
        truth_impact=0.0,
        autonomy_impact=0.0,
        loyalty_impact=0.1,
    ),
    EventType.OBSERVE: ActionEffects(
        safety_impact=0.2,
        status_impact=0.0,
        closeness_impact=0.0,
        secrecy_impact=0.1,
        truth_impact=0.3,
        autonomy_impact=0.1,
        loyalty_impact=0.0,
    ),
    EventType.SOCIAL_MOVE: ActionEffects(
        safety_impact=0.0,  # context dependent
        status_impact=-0.05,
        closeness_impact=0.0,  # context dependent
        secrecy_impact=0.1,
        truth_impact=0.0,
        autonomy_impact=0.2,
        loyalty_impact=0.0,
    ),
    EventType.REVEAL: ActionEffects(
        safety_impact=-0.3,
        status_impact=0.2,
        closeness_impact=0.1,
        secrecy_impact=-0.8,
        truth_impact=0.8,
        autonomy_impact=0.3,
        loyalty_impact=-0.2,
    ),
    EventType.CONFLICT: ActionEffects(
        safety_impact=-0.4,
        status_impact=0.1,
        closeness_impact=-0.3,
        secrecy_impact=-0.2,
        truth_impact=0.4,
        autonomy_impact=0.4,
        loyalty_impact=0.0,
    ),
    EventType.INTERNAL: ActionEffects(
        safety_impact=0.3,
        status_impact=0.0,
        closeness_impact=0.0,
        secrecy_impact=0.3,
        truth_impact=0.1,
        autonomy_impact=0.2,
        loyalty_impact=0.0,
    ),
    EventType.PHYSICAL: ActionEffects(
        safety_impact=0.1,
        status_impact=0.05,
        closeness_impact=0.1,
        secrecy_impact=0.0,
        truth_impact=0.0,
        autonomy_impact=0.0,
        loyalty_impact=0.1,
    ),
    EventType.CONFIDE: ActionEffects(
        safety_impact=-0.2,
        status_impact=-0.1,
        closeness_impact=0.5,
        secrecy_impact=-0.5,
        truth_impact=0.3,
        autonomy_impact=0.1,
        loyalty_impact=0.3,
    ),
    EventType.LIE: ActionEffects(
        safety_impact=0.1,
        status_impact=0.0,
        closeness_impact=-0.1,
        secrecy_impact=0.5,
        truth_impact=-0.8,
        autonomy_impact=-0.1,
        loyalty_impact=-0.4,
    ),
}


def _destination(action: Action) -> Optional[str]:
    dest = action.metadata.get("destination")
    return str(dest) if isinstance(dest, str) else None


def estimate_action_effects(action: Action, world: WorldState) -> ActionEffects:
    base = ACTION_EFFECT_PROFILES.get(action.action_type, ActionEffects())
    effects = ActionEffects(**vars(base))

    if action.action_type == EventType.SOCIAL_MOVE:
        dest_id = _destination(action)
        if dest_id and dest_id in world.locations:
            dest = world.locations[dest_id]
            effects.secrecy_impact += dest.privacy * 0.3
            effects.safety_impact += dest.privacy * 0.2

            # Moving toward desired people slightly increases closeness impact.
            agents_there = [a for a in world.agents.values() if a.location == dest_id]
            if agents_there:
                desire = 0.0
                for other in agents_there:
                    desire += world.agents[action.agent_id].goals.closeness.get(other.id, 0.0)
                effects.closeness_impact += max(min(desire / max(len(agents_there), 1), 1.0), -1.0) * 0.1

    # PHYSICAL actions are context-dependent (decision-engine.md). We treat hosting/serving
    # behaviors as more socially meaningful than a neutral physical action.
    if action.action_type == EventType.PHYSICAL:
        content = (action.content or "").lower()
        if "pour" in content or "serve" in content or "refill" in content:
            effects.status_impact += 0.10
            effects.closeness_impact += 0.20
            effects.loyalty_impact += 0.20

    if action.action_type in (EventType.REVEAL, EventType.CONFIDE):
        secret_id = action.metadata.get("secret_id")
        if isinstance(secret_id, str) and secret_id in world.secrets:
            secret = world.secrets[secret_id]
            effects.truth_impact *= 1.0 + secret.dramatic_weight
            effects.secrecy_impact *= 1.0 + secret.dramatic_weight

    if action.action_type == EventType.LIE:
        secret_id = action.metadata.get("secret_id")
        if isinstance(secret_id, str) and secret_id in world.secrets and action.target_agents:
            # Lying is more \"useful\" when someone is actively probing.
            for tid in action.target_agents:
                t = world.agents.get(tid)
                if not t:
                    continue
                if t.goals.truth_seeking > 0.6 or t.beliefs.get(secret_id) == BeliefState.SUSPECTS:
                    effects.secrecy_impact += 0.20
                    effects.safety_impact += 0.05
                    break

    if action.action_type == EventType.CONFLICT and action.target_agents:
        target_id = action.target_agents[0]
        rel = world.agents[action.agent_id].relationships.get(target_id)
        if rel and rel.trust < -0.3:
            effects.truth_impact *= 1.5
            effects.closeness_impact *= 1.5

        # If the source suspects secrets about the target, confrontation is more truth-y (seeking answers).
        source = world.agents.get(action.agent_id)
        if source:
            for secret in sorted(world.secrets.values(), key=lambda s: s.id):
                if secret.about != target_id:
                    continue
                if source.beliefs.get(secret.id) == BeliefState.SUSPECTS:
                    effects.truth_impact += 0.20
                    effects.autonomy_impact += 0.10
                    break

    return effects


def base_utility(agent: AgentState, action: Action, world: WorldState) -> float:
    goals = agent.goals
    effects = estimate_action_effects(action, world)

    score = 0.0
    score += goals.safety * effects.safety_impact
    score += goals.status * effects.status_impact

    if action.target_agents:
        # Normalize closeness contributions so group actions don't dominate purely by target count.
        closeness_sum = 0.0
        for target_id in action.target_agents:
            desire = goals.closeness.get(target_id, 0.0)
            closeness_sum += desire * effects.closeness_impact
        score += closeness_sum / max(len(action.target_agents), 1)

    score += goals.secrecy * effects.secrecy_impact
    score += goals.truth_seeking * effects.truth_impact
    score += goals.autonomy * effects.autonomy_impact
    score += goals.loyalty * effects.loyalty_impact

    if action.action_type == EventType.SOCIAL_MOVE:
        dest_id = _destination(action)
        if dest_id and dest_id in world.locations:
            score += score_move(agent, world.locations[dest_id], world)

    # INTERNAL is the "do nothing meaningful" baseline. Keep it moderate so it
    # doesn't dominate dramatic choices (decision-engine.md Section 4 note).
    if action.action_type == EventType.INTERNAL:
        return min(score, 0.45)

    return score


def has_pending_dramatic_action(agent: AgentState, world: WorldState) -> bool:
    """Heuristic: does the agent know something (or needs to protect something) that could drive drama?"""

    # If secrecy is high and others present suspect/know a secret the agent knows, the agent may seek privacy.
    if agent.goals.secrecy > 0.6:
        for secret_id, belief in sorted(agent.beliefs.items()):
            if belief != BeliefState.BELIEVES_TRUE:
                continue
            for other in sorted(world.agents.values(), key=lambda a: a.id):
                if other.id == agent.id or other.location != agent.location:
                    continue
                b = other.beliefs.get(secret_id, BeliefState.UNKNOWN)
                if b in (BeliefState.SUSPECTS, BeliefState.BELIEVES_TRUE):
                    return True

    return False


def score_move(agent: AgentState, destination: Location, world: WorldState) -> float:
    """Extra scoring for SOCIAL_MOVE actions based on strategic value (decision-engine.md Section 6.2)."""

    score = 0.0
    agents_at_dest = sorted(
        (a for a in world.agents.values() if a.location == destination.id and a.id != agent.id),
        key=lambda a: a.id,
    )
    agents_at_current = sorted(
        (a for a in world.agents.values() if a.location == agent.location and a.id != agent.id),
        key=lambda a: a.id,
    )

    for other in agents_at_dest:
        desire = agent.goals.closeness.get(other.id, 0.0)
        if desire > 0.3:
            score += desire * 0.3

        # Truth-seekers move toward the person they suspect is involved in a secret.
        if agent.goals.truth_seeking > 0.7:
            for secret in sorted(world.secrets.values(), key=lambda s: s.id):
                if secret.about != other.id:
                    continue
                if agent.beliefs.get(secret.id) == BeliefState.SUSPECTS:
                    score += 0.20

    # Mild social gravity: being where people are is usually better than being alone.
    score += min(0.25, 0.08 * len(agents_at_dest))
    if not agents_at_current and agents_at_dest:
        score += 0.15

    # Social inertia: leaving a populated dining table has social cost. This reduces
    # SOCIAL_MOVE spam and helps scenes feel less fragmented.
    if agent.location == "dining_table" and destination.id != "dining_table":
        agents_at_table = len(agents_at_current) + 1  # +1 for self

        inertia = 0.06 + 0.03 * max(0, agents_at_table - 3)
        inertia *= 1.0 + agent.goals.status * 0.5

        # If the agent has a strong reason to seek privacy, reduce (but do not eliminate)
        # the social cost of leaving. This prevents the table from collapsing immediately
        # while still allowing secrets to pull characters away.
        if has_pending_dramatic_action(agent, world):
            inertia *= 0.7

        # High stress makes leaving more plausible.
        if agent.pacing.stress >= 0.60:
            inertia *= 0.4
        elif agent.pacing.stress >= 0.45:
            inertia *= 0.6

        score -= inertia if agents_at_table >= 4 else inertia * 0.6

    # Conversely, returning to the table when it's active is socially rewarding.
    if destination.id == "dining_table" and agents_at_dest:
        score += 0.10 + agent.goals.status * 0.14 + agent.goals.loyalty * 0.12

    # Location affordances (scenario-tuned, minimal): increase kitchen/foyer activity,
    # discourage repetitive bathroom trips. These are intentionally small so the
    # dining_table remains the default gravity well.
    if destination.id == "kitchen":
        if agent.location == "dining_table":
            # Kitchen is a plausible pretext for leaving the table (wine/snacks). We want it
            # to be used sometimes, but not as a new "gravity well" that the whole party
            # migrates to. Penalize pile-ons to avoid group-collapse dynamics.
            score += 0.03 + agent.goals.autonomy * 0.04
            if agent.alcohol_level < 0.50:
                score += 0.06
            if len(agents_at_dest) >= 2:
                score -= 0.10 * (len(agents_at_dest) - 1)

    if destination.id == "foyer":
        if agent.location == "dining_table":
            # Foyer is semi-public, but it offers an "escape hatch" and a vantage point
            # for agents who value autonomy/truth-seeking.
            score += 0.02 + agent.goals.autonomy * 0.04
            # Encourage limited foyer usage (phone call / hallway reset) without triggering
            # a "follow the leader" group migration.
            if agent.goals.truth_seeking > 0.65:
                score += 0.10
            if agent.pacing.stress > 0.60:
                score += 0.04
            if agents_at_dest:
                score -= 0.15

    if destination.id == "bathroom":
        # Privacy is attractive (handled below), but repeated bathroom trips read as avoidance.
        recent_bathroom = sum(
            1
            for ev in world.event_log[-30:]
            if ev.type == EventType.SOCIAL_MOVE
            and ev.source_agent == agent.id
            and (ev.content_metadata or {}).get("destination") == "bathroom"
        )
        score -= 0.08 + agent.goals.status * 0.03
        if agent.pacing.stress < 0.5:
            score -= 0.04
        if recent_bathroom:
            score -= 0.10 * recent_bathroom
        if agent.pacing.stress > 0.75:
            score += 0.08

    effective_privacy = destination.privacy
    if destination.id == "foyer":
        # Foyers are transitional but still offer enough privacy for short side conversations.
        effective_privacy = min(effective_privacy, 0.55)
    if destination.id == "bathroom":
        # Bathrooms are private, but they're also suspicious; do not let raw privacy
        # dominate move choice for secretive agents.
        effective_privacy = min(effective_privacy, 0.75)

    if agent.pacing.stress > 0.5:
        score += effective_privacy * 0.2

    if has_pending_dramatic_action(agent, world):
        score += effective_privacy * 0.3

    # Secretive agents prefer privacy even when not stressed.
    if agent.goals.secrecy > 0.7:
        score += effective_privacy * 0.15

    return score


def _action_involves_agent(action: Action, agent_id: str) -> bool:
    if agent_id in action.target_agents:
        return True
    # OBSERVE actions usually encode the focus in metadata.
    focus = action.metadata.get("observed_agent")
    return isinstance(focus, str) and focus == agent_id


def _obsession_target(agent: AgentState) -> Optional[str]:
    # MVP heuristic: Victor is obsessed with Marcus.
    if agent.id == "victor":
        return "marcus"
    return None


def _secret_at_risk(agent: AgentState, secret: SecretDefinition, world: WorldState) -> bool:
    # A secret is "at risk" if someone present is suspecting it or knows it.
    for other in sorted(world.agents.values(), key=lambda a: a.id):
        if other.id == agent.id:
            continue
        if other.location != agent.location:
            continue
        b = other.beliefs.get(secret.id, BeliefState.UNKNOWN)
        if b in (BeliefState.SUSPECTS, BeliefState.BELIEVES_TRUE):
            return True
    return False


def _trigger_active(
    flaw: CharacterFlaw, agent: AgentState, action: Action, perception: PerceivedState, world: WorldState
) -> bool:
    match flaw.trigger:
        case "status_threat":
            return agent.emotional_state.get("shame", 0.0) > 0.2 or agent.emotional_state.get("anger", 0.0) > 0.2
        case "betrayal_detected":
            return agent.emotional_state.get("suspicion", 0.0) > 0.4
        case "secret_exposure":
            return any(
                agent.beliefs.get(sid) == BeliefState.BELIEVES_TRUE and _secret_at_risk(agent, s, world)
                for sid, s in sorted(world.secrets.items())
            )
        case "rejection":
            return agent.emotional_state.get("shame", 0.0) > 0.3
        case "authority_challenge":
            # Vanity tends to be persistently active in social settings.
            return bool(perception.visible_agents)
        case "intimacy_offered":
            return action.action_type in (EventType.CONFIDE, EventType.CHAT) and bool(action.target_agents)
        case "conflict_nearby":
            return perception.recent_conflict_at_location or agent.pacing.stress > 0.5
        case "loss_imminent":
            return agent.pacing.stress > 0.5 and agent.pacing.commitment > 0.4
        case _:
            return False


def _apply_flaw_effect(flaw: CharacterFlaw, action: Action, agent: AgentState, world: WorldState) -> float:
    strength = flaw.strength

    match flaw.effect:
        case "overweight_status":
            effects = estimate_action_effects(action, world)
            return strength * effects.status_impact * 0.5

        case "avoid_confrontation":
            if action.action_type in (EventType.CONFLICT, EventType.REVEAL):
                return -strength * 0.5
            if action.action_type == EventType.SOCIAL_MOVE:
                return strength * 0.3
            return 0.0

        case "deny_evidence":
            if action.action_type == EventType.OBSERVE:
                return -strength * 0.3
            if action.action_type == EventType.LIE:
                return strength * 0.3
            return 0.0

        case "escalate_conflict":
            if action.action_type == EventType.CONFLICT:
                return strength * 0.4
            return 0.0

        case "seek_validation":
            if action.action_type == EventType.CHAT and action.target_agents:
                target = world.agents.get(action.target_agents[0])
                if target and target.goals.status > 0.6:
                    return strength * 0.3
            return 0.0

        case "self_sacrifice":
            if action.action_type in (EventType.CONFIDE, EventType.REVEAL):
                return strength * 0.3
            return 0.0

        case "fixate_on_target":
            target = _obsession_target(agent)
            if target and _action_involves_agent(action, target):
                return strength * 0.5
            return 0.0

        case "overcommit":
            if action.is_dramatic:
                return strength * 0.4
            return 0.0

        case _:
            return 0.0


def flaw_bias(agent: AgentState, action: Action, perception: PerceivedState, world: WorldState) -> float:
    total = 0.0
    for flaw in agent.flaws:
        if _trigger_active(flaw, agent, action, perception, world):
            total += _apply_flaw_effect(flaw, action, agent, world)
    return total


def _is_consistent_with_commitments(action: Action, agent: AgentState) -> bool:
    secret_id = action.metadata.get("secret_id")
    if not isinstance(secret_id, str):
        secret_id = None

    # Secrecy commitments: consistent with LIE, inconsistent with REVEAL/CONFIDE.
    if "maintain_affair_secrecy" in agent.commitments and secret_id == "secret_affair_01":
        return action.action_type == EventType.LIE
    if "cover_embezzlement" in agent.commitments and secret_id == "secret_embezzle_01":
        return action.action_type == EventType.LIE
    return True


def _contradicts_commitments(action: Action, agent: AgentState) -> bool:
    secret_id = action.metadata.get("secret_id")
    if not isinstance(secret_id, str):
        return False

    if "maintain_affair_secrecy" in agent.commitments and secret_id == "secret_affair_01":
        return action.action_type in (EventType.REVEAL, EventType.CONFIDE)
    if "cover_embezzlement" in agent.commitments and secret_id == "secret_embezzle_01":
        return action.action_type in (EventType.REVEAL, EventType.CONFIDE)
    return False


def pacing_modifier(agent: AgentState, action: Action, world: WorldState, location: Location) -> float:
    c = pacing.DEFAULT_CONSTANTS
    p = agent.pacing
    modifier = 0.0

    if action.is_dramatic and p.recovery_timer > 0:
        return -1000.0
    if action.is_dramatic and p.dramatic_budget < c.BUDGET_MINIMUM_FOR_ACTION:
        return -1000.0

    # Social masking.
    if action.is_dramatic and location.privacy < c.PUBLIC_PRIVACY_THRESHOLD and p.composure >= c.COMPOSURE_MIN_FOR_MASKING:
        modifier -= (1.0 - location.privacy) * p.composure * c.MASKING_STRESS_SUPPRESSION

    # Stress-driven relief.
    if p.stress > 0.5:
        if action.action_type == EventType.SOCIAL_MOVE:
            dest_id = _destination(action)
            if dest_id and dest_id in world.locations and world.locations[dest_id].privacy > 0.5:
                modifier += p.stress * 0.3
        if action.action_type == EventType.CONFIDE:
            modifier += p.stress * 0.2
        if action.action_type == EventType.PHYSICAL:
            content = (action.content or "").lower()
            if "drink" in content:
                modifier += p.stress * 0.25
            if "pour" in content or "refill" in content or "serve" in content:
                modifier += p.stress * 0.12
        if action.action_type == EventType.LIE:
            modifier += p.stress * 0.15

    # Stress escalation: as stress rises, agents are more likely to confront or blurt out a reveal.
    # Keep this modest; it mainly serves to ensure non-Victor agents can "tip" into drama.
    if p.stress > 0.55 and action.action_type in (EventType.CONFLICT, EventType.REVEAL):
        modifier += 0.03 + (p.stress - 0.55) * 0.6

    # Commitment consistency.
    if p.commitment > 0.5:
        if _is_consistent_with_commitments(action, agent):
            modifier += p.commitment * 0.2
        if _contradicts_commitments(action, agent):
            modifier -= p.commitment * 0.4

    # Low budget encourages quiet.
    if p.dramatic_budget < 0.4 and not action.is_dramatic:
        modifier += 0.1

    # Alcohol: lowered inhibitions.
    if agent.alcohol_level >= 0.15 and action.is_dramatic:
        modifier += agent.alcohol_level * 0.3

    return modifier


def relationship_modifier(agent: AgentState, action: Action, world: WorldState) -> float:
    if not action.target_agents:
        return 0.0

    modifier = 0.0
    for target_id in action.target_agents:
        rel = agent.relationships.get(target_id)
        if not rel:
            continue

        match action.action_type:
            case EventType.CHAT:
                modifier += rel.affection * 0.2
            case EventType.CONFIDE:
                modifier += rel.trust * 0.5
                modifier += rel.obligation * 0.2
            case EventType.CONFLICT:
                modifier -= rel.trust * 0.3
                modifier -= rel.affection * 0.2
            case EventType.REVEAL:
                modifier += rel.trust * 0.3
            case EventType.LIE:
                modifier -= rel.trust * 0.2
                modifier -= rel.affection * 0.3

    # Similar normalization: group targets shouldn't multiply relationship effects.
    return modifier / max(len(action.target_agents), 1)


def action_noise(rng: Random) -> float:
    return rng.gauss(0.0, 0.1)


def _action_signature(action: Action) -> tuple:
    """A stable signature for recency penalties (decision-engine.md Open Question 10.1)."""

    dest = _destination(action) if action.action_type == EventType.SOCIAL_MOVE else None
    observed = action.metadata.get("observed_agent") if action.action_type == EventType.OBSERVE else None
    secret_id = action.metadata.get("secret_id") if action.action_type in (EventType.REVEAL, EventType.CONFIDE, EventType.LIE) else None

    return (
        action.action_type.value,
        tuple(sorted(action.target_agents)),
        str(dest) if dest else None,
        str(observed) if isinstance(observed, str) else None,
        str(secret_id) if isinstance(secret_id, str) else None,
    )


def _event_signature(event) -> tuple:
    dest = (event.content_metadata or {}).get("destination") if event.type == EventType.SOCIAL_MOVE else None
    observed = (event.content_metadata or {}).get("observed_agent") if event.type == EventType.OBSERVE else None
    secret_id = (event.content_metadata or {}).get("secret_id") if event.type in (EventType.REVEAL, EventType.CONFIDE, EventType.LIE) else None
    return (
        event.type.value,
        tuple(sorted(event.target_agents)),
        str(dest) if isinstance(dest, str) else None,
        str(observed) if isinstance(observed, str) else None,
        str(secret_id) if isinstance(secret_id, str) else None,
    )


def recency_penalty(agent: AgentState, action: Action, world: WorldState) -> float:
    """Penalize repeating the same (type, target, metadata) action within a short window."""

    lookback_ticks = 3
    # Repetition is worse for low-information actions (INTERNAL/OBSERVE) than for drama.
    base = 0.15 if not action.is_dramatic else 0.08
    sig = _action_signature(action)
    current_tick = int(getattr(world, "tick_id", 0))

    penalty = 0.0
    for ev in reversed(world.event_log[-50:]):
        if ev.source_agent != agent.id:
            continue
        dt = current_tick - int(ev.tick_id)
        if dt <= 0:
            continue
        if dt > lookback_ticks:
            break
        if _event_signature(ev) != sig:
            continue
        # More recent repeats are discouraged more strongly.
        penalty += base * (1.0 + (lookback_ticks - dt) * 0.35)

    # Extra discouragement for jittery pacing: moving every tick tends to fragment scenes.
    if action.action_type == EventType.SOCIAL_MOVE:
        moved_recently = False
        for ev in reversed(world.event_log[-30:]):
            if ev.source_agent != agent.id:
                continue
            dt = current_tick - int(ev.tick_id)
            if dt <= 0:
                continue
            if dt > 3:
                break
            if ev.type == EventType.SOCIAL_MOVE:
                moved_recently = True
                break
        if moved_recently:
            penalty += 0.12

    return penalty


def determine_priority_class(action: Action, agent: AgentState) -> int:
    if action.action_type == EventType.CONFLICT and agent.pacing.stress > 0.6:
        return 3
    if action.is_dramatic:
        return 2
    return 1


def generate_candidate_actions(agent: AgentState, perception: PerceivedState, world: WorldState) -> list[Action]:
    candidates: list[Action] = []
    location = world.locations[agent.location]
    present_agents = perception.visible_agents

    # CHAT: one per present agent, plus group.
    for other in present_agents:
        # At the dining table, even 1:1 chats are semi-public; use a long-form
        # description so pacing.commitment can tick up via "public statement".
        # (pacing-physics.md Section 4.4, implemented as len(description) >= 40.)
        if location.id == "dining_table":
            content = f"Make polite conversation with {other.name} about the company and the evening."
        else:
            content = f"Chat with {other.name}"
        candidates.append(
            Action(
                agent_id=agent.id,
                action_type=EventType.CHAT,
                target_agents=[other.id],
                location_id=agent.location,
                utility_score=0.0,
                content=content,
                is_dramatic=False,
                requires_target_available=False,
            )
        )
    if len(present_agents) >= 2:
        candidates.append(
            Action(
                agent_id=agent.id,
                action_type=EventType.CHAT,
                target_agents=[a.id for a in present_agents],
                location_id=agent.location,
                utility_score=0.0,
                content="Make small talk with the group over dinner.",
                is_dramatic=False,
                requires_target_available=False,
            )
        )
        # Public statement: long-form chat that increases commitment (pacing-physics.md).
        if location.id == "dining_table":
            candidates.append(
                Action(
                    agent_id=agent.id,
                    action_type=EventType.CHAT,
                    target_agents=[a.id for a in present_agents],
                    location_id=agent.location,
                    utility_score=0.0,
                    content="Make a toast about loyalty, success, and the future of the company.",
                    is_dramatic=False,
                    requires_target_available=False,
                )
            )

    # OBSERVE: one per present agent, plus general scan.
    for other in present_agents:
        candidates.append(
            Action(
                agent_id=agent.id,
                action_type=EventType.OBSERVE,
                target_agents=[],
                location_id=agent.location,
                utility_score=0.0,
                content=f"Observe {other.name}",
                metadata={"observed_agent": other.id},
                is_dramatic=False,
                requires_target_available=False,
            )
        )
    candidates.append(
        Action(
            agent_id=agent.id,
            action_type=EventType.OBSERVE,
            target_agents=[],
            location_id=agent.location,
            utility_score=0.0,
            content="Scan the room",
            is_dramatic=False,
            requires_target_available=False,
        )
    )

    # SOCIAL_MOVE: adjacent locations with capacity.
    for adj_id in location.adjacent:
        adj = world.locations.get(adj_id)
        if not adj:
            continue
        agents_there = sum(1 for a in world.agents.values() if a.location == adj_id)
        if agents_there < adj.capacity:
            candidates.append(
                Action(
                    agent_id=agent.id,
                    action_type=EventType.SOCIAL_MOVE,
                    target_agents=[],
                    location_id=agent.location,
                    utility_score=0.0,
                    content=f"Move to {adj.name}",
                    metadata={"destination": adj_id},
                    is_dramatic=False,
                    requires_target_available=False,
                )
            )

    # INTERNAL: always.
    candidates.append(
        Action(
            agent_id=agent.id,
            action_type=EventType.INTERNAL,
            target_agents=[],
            location_id=agent.location,
            utility_score=0.0,
            content="Think",
            is_dramatic=False,
            requires_target_available=False,
        )
    )

    # PHYSICAL: simple option to drink.
    if agent.alcohol_level < 0.95:
        candidates.append(
            Action(
                agent_id=agent.id,
                action_type=EventType.PHYSICAL,
                target_agents=[],
                location_id=agent.location,
                utility_score=0.0,
                content="Drink wine",
                is_dramatic=False,
                requires_target_available=False,
            )
        )

    # PHYSICAL: hosting/serving behavior (more likely at the table).
    if location.id == "dining_table" and present_agents:
        candidates.append(
            Action(
                agent_id=agent.id,
                action_type=EventType.PHYSICAL,
                target_agents=[a.id for a in present_agents],
                location_id=agent.location,
                utility_score=0.0,
                content="Pour wine for the table",
                is_dramatic=False,
                requires_target_available=False,
            )
        )

    # Location affordance actions (scenario-specific, but schema-neutral).
    if location.id == "kitchen":
        table_agents = sorted(
            (a for a in world.agents.values() if a.location == "dining_table" and a.id != agent.id),
            key=lambda a: a.id,
        )
        if table_agents:
            candidates.append(
                Action(
                    agent_id=agent.id,
                    action_type=EventType.PHYSICAL,
                    target_agents=[a.id for a in table_agents],
                    location_id=agent.location,
                    utility_score=0.0,
                    content="Refill drinks for the table",
                    is_dramatic=False,
                    requires_target_available=False,
                )
            )

    if location.id == "foyer":
        candidates.append(
            Action(
                agent_id=agent.id,
                action_type=EventType.INTERNAL,
                target_agents=[],
                location_id=agent.location,
                utility_score=0.0,
                content="Check phone",
                metadata={"affordance": "check_phone"},
                is_dramatic=False,
                requires_target_available=False,
            )
        )

    # Dramatic actions only if budget + recovery allow.
    if agent.pacing.dramatic_budget >= pacing.DEFAULT_CONSTANTS.BUDGET_MINIMUM_FOR_ACTION and agent.pacing.recovery_timer == 0:
        # REVEAL: reveal known secrets.
        for secret_id, belief in sorted(agent.beliefs.items()):
            if belief != BeliefState.BELIEVES_TRUE:
                continue
            for other in present_agents:
                other_belief = world.agents[other.id].beliefs.get(secret_id, BeliefState.UNKNOWN)
                if other_belief == BeliefState.BELIEVES_TRUE:
                    continue
                candidates.append(
                    Action(
                        agent_id=agent.id,
                        action_type=EventType.REVEAL,
                        target_agents=[other.id],
                        location_id=agent.location,
                        utility_score=0.0,
                        content=f"Reveal {secret_id} to {other.name}",
                        metadata={"secret_id": secret_id},
                        is_dramatic=True,
                    )
                )

        # CONFLICT: confront distrusted people or when angry.
        for other in present_agents:
            rel = agent.relationships.get(other.id)
            if not rel:
                continue
            suspicion = agent.emotional_state.get("suspicion", 0.0)
            desire = agent.goals.closeness.get(other.id, 0.0)
            if (
                rel.trust < 0.0
                or agent.emotional_state.get("anger", 0.0) > 0.3
                or (suspicion > 0.4 and rel.trust < 0.2)
                or (suspicion > 0.55 and abs(desire) > 0.2)
                or _is_probe_threat(agent, other)
            ):
                candidates.append(
                    Action(
                        agent_id=agent.id,
                        action_type=EventType.CONFLICT,
                        target_agents=[other.id],
                        location_id=agent.location,
                        utility_score=0.0,
                        content=f"Confront {other.name}",
                        is_dramatic=True,
                    )
                )

        # CONFIDE: share known secrets with trusted people.
        for other in present_agents:
            rel = agent.relationships.get(other.id)
            if not rel or rel.trust <= 0.4:
                continue
            for secret_id, belief in sorted(agent.beliefs.items()):
                if belief != BeliefState.BELIEVES_TRUE:
                    continue
                other_belief = world.agents[other.id].beliefs.get(secret_id, BeliefState.UNKNOWN)
                if other_belief == BeliefState.BELIEVES_TRUE:
                    continue
                candidates.append(
                    Action(
                        agent_id=agent.id,
                        action_type=EventType.CONFIDE,
                        target_agents=[other.id],
                        location_id=agent.location,
                        utility_score=0.0,
                        content=f"Confide about {secret_id} to {other.name}",
                        metadata={"secret_id": secret_id},
                        is_dramatic=True,
                        requires_target_available=True,
                    )
                )

        # LIE: protect secrets to curious targets.
        for other in present_agents:
            for secret_id, belief in sorted(agent.beliefs.items()):
                if belief != BeliefState.BELIEVES_TRUE:
                    continue
                if should_consider_lying(agent, other.id, secret_id, world):
                    candidates.append(
                        Action(
                            agent_id=agent.id,
                            action_type=EventType.LIE,
                            target_agents=[other.id],
                            location_id=agent.location,
                            utility_score=0.0,
                            content=f"Lie to {other.name} about {secret_id}",
                            metadata={"secret_id": secret_id},
                            is_dramatic=True,
                        )
                    )

    return candidates


def should_consider_lying(agent: AgentState, target_id: str, secret_id: str, world: WorldState) -> bool:
    # If secrecy goal is high and the target seems suspicious/curious, lying is on the table.
    if agent.goals.secrecy < 0.6:
        return False
    target = world.agents.get(target_id)
    if not target:
        return False
    if target.goals.truth_seeking > 0.6:
        return True
    return target.beliefs.get(secret_id, BeliefState.UNKNOWN) == BeliefState.SUSPECTS


def _is_probe_threat(agent: AgentState, other: AgentState) -> bool:
    """Return True if 'other' is likely probing secrets the agent is trying to protect."""

    if agent.goals.secrecy < 0.8:
        return False
    if other.goals.truth_seeking <= 0.6:
        return False

    for secret_id, belief in sorted(agent.beliefs.items()):
        if belief != BeliefState.BELIEVES_TRUE:
            continue
        if other.beliefs.get(secret_id, BeliefState.UNKNOWN) == BeliefState.SUSPECTS:
            return True
    return False


def prune_candidates(candidates: list[Action], agent: AgentState) -> list[Action]:
    pruned: list[Action] = []
    for action in candidates:
        if agent.id in action.target_agents:
            continue
        if action.action_type == EventType.SOCIAL_MOVE:
            dest = _destination(action)
            if not dest or dest == agent.location:
                continue
        if action.action_type in (EventType.REVEAL, EventType.CONFIDE, EventType.LIE):
            secret_id = action.metadata.get("secret_id")
            if not isinstance(secret_id, str):
                continue
            if agent.beliefs.get(secret_id) != BeliefState.BELIEVES_TRUE:
                continue
        pruned.append(action)
    return pruned


def score_action_no_noise(agent: AgentState, action: Action, perception: PerceivedState, world: WorldState) -> float:
    loc = world.locations[agent.location]
    return (
        base_utility(agent, action, world)
        + flaw_bias(agent, action, perception, world)
        + pacing_modifier(agent, action, world, loc)
        + relationship_modifier(agent, action, world)
    )


def _action_tiebreak_key(action: Action) -> tuple[str, str, tuple[str, ...], str, str, str, str]:
    """
    Deterministic tie-breaker for action selection.

    Scores can (rarely) tie due to symmetrical state or identical candidate actions.
    We prefer a stable, content-based ordering over relying on candidate generation order.
    """
    observed_agent = action.metadata.get("observed_agent")
    destination = action.metadata.get("destination")
    secret_id = action.metadata.get("secret_id")
    return (
        action.action_type.value,
        action.location_id,
        tuple(sorted(action.target_agents)),
        str(observed_agent) if observed_agent is not None else "",
        str(destination) if destination is not None else "",
        str(secret_id) if secret_id is not None else "",
        action.content,
    )


def select_action(agent: AgentState, perception: PerceivedState, world: WorldState, rng: Random) -> Optional[Action]:
    candidates = generate_candidate_actions(agent, perception, world)
    candidates = prune_candidates(candidates, agent)
    if not candidates:
        return None

    scored: list[tuple[float, Action]] = []
    loc = world.locations[agent.location]
    for action in candidates:
        score = (
            base_utility(agent, action, world)
            + flaw_bias(agent, action, perception, world)
            + pacing_modifier(agent, action, world, loc)
            + relationship_modifier(agent, action, world)
            - recency_penalty(agent, action, world)
            + action_noise(rng)
        )
        if action.action_type == EventType.SOCIAL_MOVE:
            score -= 0.10
        scored.append((score, action))

    # Deterministic ordering for ties: sort by score then stable action key.
    scored.sort(key=lambda sa: (-sa[0], _action_tiebreak_key(sa[1])))
    best_score, best_action = scored[0]
    best_action.utility_score = best_score
    best_action.priority_class = determine_priority_class(best_action, agent)
    return best_action
