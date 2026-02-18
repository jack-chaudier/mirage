from __future__ import annotations

from dataclasses import dataclass

from narrativefield.schema.agents import AgentState, BeliefState
from narrativefield.schema.events import CollapsedBelief, DeltaKind, Event, IronyCollapseInfo
from narrativefield.schema.world import ClaimDefinition, SecretDefinition


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _relevance(about: str | None, holder: list[str], agent_id: str, agents_present: set[str]) -> float:
    if about == agent_id:
        return 1.0
    if agent_id in holder:
        return 0.7
    if about is not None and about in agents_present:
        return 0.5
    return 0.2


def secret_relevance(secret: SecretDefinition, agent_id: str, agents_present: set[str]) -> float:
    """
    How relevant is this secret to this agent in the current context?

    Source: specs/metrics/irony-and-beliefs.md Section 3.2.
    """
    return _relevance(secret.about, secret.holder, agent_id, agents_present)


def claim_relevance(claim: ClaimDefinition, agent_id: str, agents_present: set[str]) -> float:
    return _relevance(claim.about, claim.holder, agent_id, agents_present)


def agent_irony(
    agent_id: str,
    beliefs: dict[str, dict[str, BeliefState]],
    secrets: dict[str, SecretDefinition],
    agents_present: set[str],
) -> float:
    """
    Per-agent irony score.

    Scoring (specs/metrics/irony-and-beliefs.md Section 3.1):
    - actively wrong: 2.0
    - relevant unknown: 1.5
    - general unknown: 0.5
    - suspects (true secret): 0.25
    - correct: 0.0
    Weighted by secret relevance.
    """
    score = 0.0
    row = beliefs.get(agent_id, {})

    for secret_id, secret in secrets.items():
        relevance = secret_relevance(secret, agent_id, agents_present)
        if relevance < 0.1:
            continue

        belief = row.get(secret_id, BeliefState.UNKNOWN)

        # Actively wrong belief: maximum irony.
        if belief == BeliefState.BELIEVES_TRUE and not secret.truth_value:
            score += 2.0 * relevance
            continue
        if belief == BeliefState.BELIEVES_FALSE and secret.truth_value:
            score += 2.0 * relevance
            continue

        if belief == BeliefState.UNKNOWN:
            if secret.about == agent_id or agent_id in secret.holder:
                score += 1.5 * relevance
            else:
                score += 0.5 * relevance
            continue

        if belief == BeliefState.SUSPECTS and secret.truth_value:
            score += 0.25 * relevance

    return float(score)


def scene_irony(
    agents_present: set[str],
    beliefs: dict[str, dict[str, BeliefState]],
    secrets: dict[str, SecretDefinition],
) -> float:
    if not agents_present:
        return 0.0
    total = sum(agent_irony(a, beliefs, secrets, agents_present) for a in agents_present)
    return float(total) / float(len(agents_present))


def agent_irony_claims(
    agent_id: str,
    beliefs: dict[str, dict[str, BeliefState]],
    claims: dict[str, ClaimDefinition],
    agents_present: set[str],
) -> float:
    score = 0.0
    row = beliefs.get(agent_id, {})

    for claim_id, claim in claims.items():
        relevance = claim_relevance(claim, agent_id, agents_present)
        if relevance < 0.1:
            continue

        belief = row.get(claim_id, BeliefState.UNKNOWN)
        truth = str(claim.truth_status).strip().lower()

        if truth == "unknown":
            continue

        if truth == "contested":
            if belief in (BeliefState.BELIEVES_TRUE, BeliefState.BELIEVES_FALSE):
                score += 0.5 * relevance
            continue

        if truth not in ("true", "false"):
            continue

        truth_value = truth == "true"

        if belief == BeliefState.BELIEVES_TRUE and not truth_value:
            score += 2.0 * relevance
            continue
        if belief == BeliefState.BELIEVES_FALSE and truth_value:
            score += 2.0 * relevance
            continue

        if belief == BeliefState.UNKNOWN:
            if claim.about == agent_id or agent_id in claim.holder:
                score += 1.5 * relevance
            else:
                score += 0.5 * relevance
            continue

        if belief == BeliefState.SUSPECTS and truth_value:
            score += 0.25 * relevance

    return float(score)


def scene_irony_claims(
    agents_present: set[str],
    beliefs: dict[str, dict[str, BeliefState]],
    claims: dict[str, ClaimDefinition],
) -> float:
    if not agents_present:
        return 0.0
    total = sum(agent_irony_claims(a, beliefs, claims, agents_present) for a in agents_present)
    return float(total) / float(len(agents_present))


@dataclass(frozen=True)
class IronyState:
    beliefs: dict[str, dict[str, BeliefState]]
    agent_locations: dict[str, str]


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _agents_at_location(agent_locations: dict[str, str], location_id: str) -> set[str]:
    return {aid for aid, loc in agent_locations.items() if loc == location_id}


def _clone_beliefs(initial_agents: dict[str, AgentState]) -> dict[str, dict[str, BeliefState]]:
    return {aid: dict(a.beliefs) for aid, a in initial_agents.items()}


def _clone_locations(initial_agents: dict[str, AgentState]) -> dict[str, str]:
    return {aid: a.location for aid, a in initial_agents.items()}


def _claims_in_play(
    claims: dict[str, ClaimDefinition],
    beliefs: dict[str, dict[str, BeliefState]],
) -> dict[str, ClaimDefinition]:
    active_ids = {claim_id for row in beliefs.values() for claim_id in row.keys()}
    if not active_ids:
        return {}
    return {claim_id: claim for claim_id, claim in claims.items() if claim_id in active_ids}


def run_irony_pipeline(
    events: list[Event],
    initial_agents: dict[str, AgentState],
    secrets: dict[str, SecretDefinition],
    *,
    claims: dict[str, ClaimDefinition] | None = None,
) -> IronyState:
    """
    Enrich events with:
    - event.metrics.irony (scene-level irony at that event)
    - event.metrics.irony_collapse (if detected)

    This pass also returns the final belief matrix and agent locations, which
    are convenient for downstream computations and bundling.
    """
    beliefs = _clone_beliefs(initial_agents)
    agent_locations = _clone_locations(initial_agents)

    prev_scene_irony: float | None = None

    for event in events:
        # Apply location deltas first so SOCIAL_MOVE events are evaluated at the destination.
        for d in event.deltas:
            if d.kind != DeltaKind.AGENT_LOCATION:
                continue
            agent_locations[d.agent] = str(d.value)

        collapsed_beliefs: list[CollapsedBelief] = []
        for d in event.deltas:
            if d.kind != DeltaKind.BELIEF:
                continue
            agent_id = d.agent
            secret_id = d.attribute
            if not agent_id or not secret_id:
                continue
            old = beliefs.setdefault(agent_id, {}).get(secret_id, BeliefState.UNKNOWN).value
            new_raw = str(d.value)
            try:
                new = BeliefState(new_raw).value
            except ValueError:
                continue
            beliefs.setdefault(agent_id, {})[secret_id] = BeliefState(new)
            collapsed_beliefs.append(
                CollapsedBelief(agent=agent_id, secret=secret_id, from_state=old, to_state=new)
            )

        agents_present = _agents_at_location(agent_locations, event.location_id) | _participants(event)

        if claims:
            curr_scene_irony = scene_irony_claims(agents_present, beliefs, _claims_in_play(claims, beliefs))
        else:
            curr_scene_irony = scene_irony(agents_present, beliefs, secrets)
        event.metrics.irony = curr_scene_irony

        # Irony collapse: scene_irony drop >= 0.5 between consecutive events.
        collapse: IronyCollapseInfo | None = None
        if prev_scene_irony is not None:
            drop = prev_scene_irony - curr_scene_irony
            if drop >= 0.5:
                collapse = IronyCollapseInfo(
                    detected=True,
                    drop=float(drop),
                    collapsed_beliefs=collapsed_beliefs,
                    score=float(drop),
                )

        event.metrics.irony_collapse = collapse
        prev_scene_irony = curr_scene_irony

    return IronyState(beliefs=beliefs, agent_locations=agent_locations)


def compute_snapshot_irony(
    *,
    beliefs: dict[str, dict[str, BeliefState]],
    secrets: dict[str, SecretDefinition],
    claims: dict[str, ClaimDefinition] | None = None,
) -> tuple[dict[str, float], float]:
    """
    Compute per-agent irony + scene-level irony for a snapshot belief matrix.
    """
    agents_present = set(beliefs.keys())
    if claims:
        active_claims = _claims_in_play(claims, beliefs)
        per_agent = {aid: agent_irony_claims(aid, beliefs, active_claims, agents_present) for aid in agents_present}
        sc = scene_irony_claims(agents_present, beliefs, active_claims)
    else:
        per_agent = {aid: agent_irony(aid, beliefs, secrets, agents_present) for aid in agents_present}
        sc = scene_irony(agents_present, beliefs, secrets)
    return per_agent, sc
