from __future__ import annotations

from dataclasses import dataclass

from narrativefield.schema.agents import AgentState, BeliefState, RelationshipState
from narrativefield.schema.events import DeltaKind, Event, EventType
from narrativefield.schema.index_tables import IndexTables
from narrativefield.schema.world import ClaimDefinition, SecretDefinition


TENSION_KEYS: tuple[str, ...] = (
    "danger",
    "time_pressure",
    "goal_frustration",
    "relationship_volatility",
    "information_gap",
    "resource_scarcity",
    "moral_cost",
    "irony_density",
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _merge_claims(
    secrets: dict[str, SecretDefinition],
    claims: dict[str, ClaimDefinition] | None,
) -> dict[str, ClaimDefinition]:
    merged = {secret_id: secret.to_claim() for secret_id, secret in secrets.items()}
    if claims:
        merged.update(claims)
    return merged


def _claims_in_play(
    claims: dict[str, ClaimDefinition],
    all_agents: dict[str, AgentState],
) -> dict[str, ClaimDefinition]:
    active_ids = {claim_id for a in all_agents.values() for claim_id in a.beliefs.keys()}
    if not active_ids:
        return {}
    return {claim_id: claim for claim_id, claim in claims.items() if claim_id in active_ids}


@dataclass(frozen=True)
class TensionWeights:
    danger: float = 0.125
    time_pressure: float = 0.125
    goal_frustration: float = 0.125
    relationship_volatility: float = 0.125
    information_gap: float = 0.125
    resource_scarcity: float = 0.125
    moral_cost: float = 0.125
    irony_density: float = 0.125

    def as_dict(self) -> dict[str, float]:
        return {
            "danger": self.danger,
            "time_pressure": self.time_pressure,
            "goal_frustration": self.goal_frustration,
            "relationship_volatility": self.relationship_volatility,
            "information_gap": self.information_gap,
            "resource_scarcity": self.resource_scarcity,
            "moral_cost": self.moral_cost,
            "irony_density": self.irony_density,
        }


DEFAULT_WEIGHTS = TensionWeights()


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 1e-9:
        return {k: 1.0 / len(TENSION_KEYS) for k in TENSION_KEYS}
    return {k: max(0.0, float(weights.get(k, 0.0))) / total for k in TENSION_KEYS}


def _danger(event: Event) -> float:
    physical = 0.0
    if event.type == EventType.CONFLICT:
        physical = 0.8
    elif event.type == EventType.CATASTROPHE:
        physical = 1.0

    # OBSERVE events inherit partial danger from what was observed/overheard.
    if event.type == EventType.OBSERVE and event.content_metadata:
        t = event.content_metadata.get("observed_event_type") or event.content_metadata.get("overheard_event_type")
        if isinstance(t, str):
            if t == EventType.CONFLICT.value:
                physical = max(physical, 0.45)
            elif t == EventType.CATASTROPHE.value:
                physical = max(physical, 0.65)

    social = 0.0
    for d in event.deltas:
        if d.kind == DeltaKind.RELATIONSHIP and d.attribute == "trust" and isinstance(d.value, (int, float)):
            if float(d.value) < -0.3:
                social = max(social, min(abs(float(d.value)), 1.0))

    if event.type == EventType.REVEAL:
        social = max(social, 0.7)
    elif event.type == EventType.LIE:
        social = max(social, 0.5)
    elif event.type == EventType.CONFIDE:
        social = max(social, 0.3)

    combined = max(physical, social) + 0.15 * min(physical, social)
    return clamp(combined, 0.0, 1.0)


def _time_pressure(
    *,
    event: Event,
    max_sim_time: float,
    source_agent: AgentState | None,
    all_agents: dict[str, AgentState],
    claims: dict[str, ClaimDefinition],
) -> float:
    scores: list[float] = []

    # Evening progression ramps up quadratically.
    progress = event.sim_time / max(1e-6, max_sim_time)
    scores.append(clamp(progress**2, 0.0, 1.0))

    if source_agent is not None and source_agent.pacing.recovery_timer > 0:
        max_timer = 8.0
        timer_ratio = 1.0 - (float(source_agent.pacing.recovery_timer) / max_timer)
        scores.append(clamp(timer_ratio * 0.6, 0.0, 1.0))

    # Claim convergence: if multiple agents suspect claims relevant to participants.
    participants = _participants(event)
    for claim_id, claim in claims.items():
        if claim.about in participants or any(h in participants for h in claim.holder):
            suspect_count = sum(
                1 for a in all_agents.values() if a.beliefs.get(claim_id, BeliefState.UNKNOWN) == BeliefState.SUSPECTS
            )
            if suspect_count >= 2:
                scores.append(clamp(suspect_count / 4.0, 0.0, 1.0))

    if source_agent is not None:
        composure_loss = 1.0 - float(source_agent.pacing.composure)
        if composure_loss > 0.3:
            scores.append(clamp(composure_loss * 0.5, 0.0, 1.0))

    return clamp(max(scores) if scores else 0.0, 0.0, 1.0)


def _goal_frustration(source_agent: AgentState | None) -> float:
    if source_agent is None:
        return 0.0
    # MVP heuristic: uses pacing proxies in [0,1].
    # Spec target formula uses cosine distance between current state vector and
    # per-agent goal vectors; those vectors are not yet materialized in MVP data.
    return clamp(
        0.6 * float(source_agent.pacing.stress) + 0.4 * (1.0 - float(source_agent.pacing.dramatic_budget)),
        0.0,
        1.0,
    )


def _relationship_volatility(event: Event, recent_events: list[Event]) -> float:
    current_abs = 0.0
    for d in event.deltas:
        if d.kind == DeltaKind.RELATIONSHIP and isinstance(d.value, (int, float)):
            current_abs += abs(float(d.value))
    current_scaled = clamp(current_abs / 0.6, 0.0, 1.0)

    recent_abs = 0.0
    for re in recent_events:
        for d in re.deltas:
            if d.kind == DeltaKind.RELATIONSHIP and isinstance(d.value, (int, float)):
                recent_abs += abs(float(d.value))
    recent_scaled = clamp(recent_abs / (0.6 * max(1, len(recent_events))), 0.0, 1.0)

    return max(current_scaled, recent_scaled)


def _information_gap(
    *,
    event: Event,
    agents_present: set[str],
    all_agents: dict[str, AgentState],
    claims: dict[str, ClaimDefinition],
) -> float:
    if not agents_present or not claims:
        return 0.0

    num = 0.0
    den = 0.0
    for claim_id, claim in claims.items():
        weight = clamp(float(claim.dramatic_weight), 0.0, 1.0) or 0.1
        scope = str(claim.scope).strip().lower()

        if scope == "public":
            den += weight
            continue

        audience = {a for a in agents_present if a in all_agents}
        if scope == "local":
            audience = {a for a in audience if all_agents[a].location != event.location_id}
        elif scope.startswith("location:"):
            scope_location = scope.split(":", 1)[1].strip()
            audience = {a for a in audience if all_agents[a].location != scope_location}

        states = {
            all_agents[a].beliefs.get(claim_id, BeliefState.UNKNOWN).value
            for a in audience
        }
        if len(states) <= 1:
            div = 0.0
        else:
            # 4 possible states, so 2→0.33, 3→0.67, 4→1.0
            div = clamp((len(states) - 1) / 3.0, 0.0, 1.0)
        num += div * weight
        den += weight

    return clamp(num / max(1e-9, den), 0.0, 1.0)


def _resource_scarcity(source_agent: AgentState | None) -> float:
    if source_agent is None:
        return 0.0
    # MVP heuristic: pacing-derived scarcity proxy in [0,1].
    # Spec target formula uses explicit world-state resource features, which the
    # current simulation schema does not expose as normalized per-agent vectors.
    budget_scarcity = 1.0 - float(source_agent.pacing.dramatic_budget)
    composure_loss = 1.0 - float(source_agent.pacing.composure)
    timer = clamp(float(source_agent.pacing.recovery_timer) / 8.0, 0.0, 1.0)
    return clamp(max(budget_scarcity, composure_loss) * 0.85 + timer * 0.15, 0.0, 1.0)


def _moral_cost(event: Event, source_agent: AgentState | None) -> float:
    if source_agent is None:
        return 0.0
    truth = float(source_agent.goals.truth_seeking)
    secrecy = float(source_agent.goals.secrecy)
    loyalty = float(source_agent.goals.loyalty)

    if event.type == EventType.CATASTROPHE:
        return 0.9
    if event.type == EventType.LIE:
        return clamp(0.5 + 0.5 * truth, 0.0, 1.0)
    if event.type == EventType.CONFLICT:
        return clamp(0.35 + 0.25 * loyalty, 0.0, 1.0)
    if event.type == EventType.REVEAL:
        return clamp(0.25 + 0.5 * secrecy, 0.0, 1.0)
    if event.type == EventType.CONFIDE:
        return 0.2
    return 0.0


def _irony_density(event: Event) -> float:
    # event.metrics.irony is unbounded-ish; normalize into [0,1] for tension components.
    return clamp(float(event.metrics.irony) / 2.0, 0.0, 1.0)


def _apply_delta(agents: dict[str, AgentState], delta: object, *, trust_repair_multiplier: float = 3.0) -> None:
    """
    Minimal, metrics-side delta application so we can compute features against evolving state.
    Mirrors narrativefield.simulation.tick_loop.apply_delta().
    """
    # Import here to keep this module standalone from the tick loop implementation.
    from narrativefield.schema.events import DeltaOp, StateDelta

    if not isinstance(delta, StateDelta):
        return
    if delta.agent not in agents:
        return
    agent = agents[delta.agent]

    match delta.kind:
        case DeltaKind.AGENT_EMOTION:
            current = float(agent.emotional_state.get(delta.attribute, 0.0))
            if delta.op == DeltaOp.ADD:
                agent.emotional_state[delta.attribute] = clamp(current + float(delta.value), 0.0, 1.0)
            else:
                agent.emotional_state[delta.attribute] = clamp(float(delta.value), 0.0, 1.0)

        case DeltaKind.AGENT_RESOURCE:
            current = float(getattr(agent, delta.attribute, 0.0))
            if delta.op == DeltaOp.ADD:
                setattr(agent, delta.attribute, clamp(current + float(delta.value), 0.0, 1.0))
            else:
                setattr(agent, delta.attribute, clamp(float(delta.value), 0.0, 1.0))

        case DeltaKind.AGENT_LOCATION:
            agent.location = str(delta.value)

        case DeltaKind.RELATIONSHIP:
            if not delta.agent_b:
                return
            rel = agent.relationships.get(delta.agent_b)
            if rel is None:
                rel = RelationshipState()
                agent.relationships[delta.agent_b] = rel

            attr = delta.attribute
            if attr not in ("trust", "affection", "obligation"):
                return

            current = float(getattr(rel, attr))
            if delta.op == DeltaOp.SET:
                new_val = float(delta.value)
            else:
                change = float(delta.value)
                # Trust repair hysteresis: positive trust changes are 3x harder.
                if attr == "trust" and change > 0:
                    change = change / trust_repair_multiplier
                new_val = current + change

            if attr == "obligation":
                new_val = clamp(new_val, 0.0, 1.0)
            else:
                new_val = clamp(new_val, -1.0, 1.0)
            setattr(rel, attr, new_val)

        case DeltaKind.BELIEF:
            secret_id = delta.attribute
            if not secret_id:
                return
            val = str(delta.value)
            try:
                agent.beliefs[secret_id] = BeliefState(val)
            except ValueError:
                return

        case DeltaKind.COMMITMENT:
            agent.commitments.append(str(delta.value))

        case DeltaKind.PACING:
            attr = delta.attribute
            current = getattr(agent.pacing, attr)
            if delta.op == DeltaOp.SET:
                new_val = delta.value
            else:
                new_val = float(current) + float(delta.value)
            setattr(agent.pacing, attr, new_val)

        case _:
            # WORLD_RESOURCE and SECRET_STATE are not used in the MVP metrics.
            return


def run_tension_pipeline(
    events: list[Event],
    *,
    agents: dict[str, AgentState],
    secrets: dict[str, SecretDefinition],
    claims: dict[str, ClaimDefinition] | None = None,
    weights: dict[str, float] | None = None,
    index_tables: IndexTables | None = None,
    max_sim_time: float | None = None,
) -> None:
    """
    Populate per-event tension components + final tension score.

    Outputs:
    - event.metrics.tension_components (8 keys)
    - event.metrics.tension (weighted sum)
    """
    if max_sim_time is None:
        max_sim_time = max((e.sim_time for e in events), default=0.0) or 1.0
    w = _normalize_weights((weights or DEFAULT_WEIGHTS.as_dict()))
    merged_claims = _merge_claims(secrets, claims)

    # For relationship volatility we want a fast "recent events involving these participants" lookup.
    if index_tables is None:
        index_tables = IndexTables(
            event_by_id={e.id: e for e in events},
            events_by_agent={},
            events_by_location={},
            participants_by_event={},
            events_by_secret={},
            events_by_pair={},
            forward_links={},
        )

    # Maintain a rolling window for quick recent lookup.
    recent_window: list[Event] = []
    RECENT_MAX = 30

    for event in events:
        source = agents.get(event.source_agent)
        agents_present = {a.id for a in agents.values() if a.location == event.location_id} | _participants(event)
        claims_in_play = _claims_in_play(merged_claims, agents)

        # Recent events involving any participant.
        participants = _participants(event)
        recent_events = [e for e in recent_window if participants & _participants(e)][-5:]

        comps: dict[str, float] = {
            "danger": _danger(event),
            "time_pressure": _time_pressure(
                event=event,
                max_sim_time=max_sim_time,
                source_agent=source,
                all_agents=agents,
                claims=claims_in_play,
            ),
            "goal_frustration": _goal_frustration(source),
            "relationship_volatility": _relationship_volatility(event, recent_events),
            "information_gap": _information_gap(
                event=event,
                agents_present=agents_present,
                all_agents=agents,
                claims=claims_in_play,
            ),
            "resource_scarcity": _resource_scarcity(source),
            "moral_cost": _moral_cost(event, source),
            "irony_density": _irony_density(event),
        }

        # Ensure canonical keys exist, clamp each channel to [0,1].
        comps = {k: clamp(float(comps.get(k, 0.0)), 0.0, 1.0) for k in TENSION_KEYS}

        tension = 0.0
        for k in TENSION_KEYS:
            tension += w[k] * comps[k]
        tension = clamp(tension, 0.0, 1.0)

        event.metrics.tension_components = comps
        event.metrics.tension = tension

        recent_window.append(event)
        if len(recent_window) > RECENT_MAX:
            recent_window = recent_window[-RECENT_MAX:]

        # Advance the evolving world state for subsequent events.
        for d in event.deltas:
            _apply_delta(agents, d)
