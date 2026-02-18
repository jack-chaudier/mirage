from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from random import Random
from typing import Any, Optional

from narrativefield.schema.agents import AgentState, BeliefState, PacingState, RelationshipState
from narrativefield.schema.canon import (
    CanonArtifact,
    CanonFaction,
    CanonInstitution,
    LocationMemory,
    WorldCanon,
)
from narrativefield.schema.events import (
    DeltaKind,
    DeltaOp,
    Event,
    EventEntities,
    EventMetrics,
    EventType,
    StateDelta,
)
from narrativefield.schema.scenes import SnapshotState
from narrativefield.schema.world import WorldDefinition
from narrativefield.simulation import decision_engine, pacing
from narrativefield.simulation.types import Action, PerceivedState, WorldState


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


SIM_TIME_PER_TICK: dict[str, float] = {
    "calm": 0.5,
    "moderate": 0.75,
    "dramatic": 1.0,
    "catastrophe": 1.5,
}

CONTENT_TYPE_TO_CONCEPTS: dict[str, list[str]] = {
    "affair": ["betrayal", "romance"],
    "financial": ["money", "trust"],
    "investigation": ["truth", "exposure"],
}

LOCATION_STAIN_BY_EVENT_TYPE: dict[EventType, float] = {
    EventType.CATASTROPHE: 0.8,
    EventType.CONFLICT: 0.4,
}

NOTABLE_EVENT_PROXY_BY_TYPE: dict[EventType, float] = {
    EventType.CATASTROPHE: 1.0,
    EventType.CONFLICT: 0.8,
    EventType.REVEAL: 0.6,
    EventType.CONFIDE: 0.5,
    EventType.LIE: 0.45,
    EventType.PHYSICAL: 0.3,
    EventType.SOCIAL_MOVE: 0.25,
    EventType.OBSERVE: 0.2,
    EventType.CHAT: 0.2,
    EventType.INTERNAL: 0.1,
}


@dataclass(frozen=True)
class SimulationConfig:
    tick_limit: int = 300
    event_limit: int = 200
    max_sim_time: float = 150.0
    snapshot_interval_events: int = 20
    max_catastrophes_per_tick: int = 2
    max_actions_per_tick: int = 6
    time_scale: float = 1.0


def init_canon_from_world(
    world_def: WorldDefinition,
    loaded_canon: WorldCanon | None = None,
) -> WorldCanon:
    canon = loaded_canon if loaded_canon is not None else WorldCanon()
    canon.world_id = world_def.id
    for loc_id in world_def.locations:
        canon.location_memory.setdefault(loc_id, LocationMemory())
    return canon


def _tick_duration(events: list[Event], time_scale: float = 1.0) -> float:
    # MVP note: time_scale only stretches/compresses sim_time progression and
    # time-based termination. It does not alter decision dynamics or pacing math.
    if any(e.type == EventType.CATASTROPHE for e in events):
        base = SIM_TIME_PER_TICK["catastrophe"]
    elif any(e.type in (EventType.CONFLICT, EventType.REVEAL) for e in events):
        base = SIM_TIME_PER_TICK["dramatic"]
    elif any(e.type in (EventType.CONFIDE, EventType.LIE, EventType.SOCIAL_MOVE) for e in events):
        base = SIM_TIME_PER_TICK["moderate"]
    else:
        base = SIM_TIME_PER_TICK["calm"]
    return base * time_scale


def _generate_event_id(world: WorldState) -> str:
    world.event_seq += 1
    return f"evt_{world.event_seq:04d}"


def _recent_conflict_at_location(world: WorldState, location_id: str, lookback: int = 6) -> bool:
    for e in world.event_log[-lookback:]:
        if e.location_id == location_id and e.type in (EventType.CONFLICT, EventType.CATASTROPHE):
            return True
    return False


def build_perception(agent: AgentState, world: WorldState) -> PerceivedState:
    loc = world.locations[agent.location]
    visible = [a for a in world.agents.values() if a.location == agent.location and a.id != agent.id]
    perceived = PerceivedState(
        visible_agents=visible,
        overhearable_locations=list(loc.overhear_from),
        recent_conflict_at_location=_recent_conflict_at_location(world, agent.location),
    )

    # Visible emotions: if the other agent's mask has slipped, show real emotions.
    for other in visible:
        if other.pacing.composure < pacing.DEFAULT_CONSTANTS.COMPOSURE_MIN_FOR_MASKING:
            perceived.visible_emotions[other.id] = dict(other.emotional_state)
        else:
            perceived.visible_emotions[other.id] = {"pleasant": 0.5}

    return perceived


def _select_catastrophe_subtype(agent: AgentState) -> str:
    if not agent.flaws:
        return "breakdown"
    dominant = max(agent.flaws, key=lambda f: f.strength)
    match dominant.flaw_type.value:
        case "pride":
            return "explosion"
        case "ambition":
            return "desperate_gambit"
        case "cowardice":
            return "flight"
        case _:
            return "breakdown"


def check_catastrophes(world: WorldState, cfg: SimulationConfig) -> list[tuple[str, str]]:
    catastrophes: list[tuple[str, str]] = []
    for agent in world.agents.values():
        if pacing.check_catastrophe(
            agent,
            catastrophe_threshold=world.definition.catastrophe_threshold,
            composure_gate=world.definition.composure_gate,
        ):
            catastrophes.append((agent.id, _select_catastrophe_subtype(agent)))

    catastrophes.sort(key=lambda t: pacing.catastrophe_potential(world.agents[t[0]]), reverse=True)
    return catastrophes[: cfg.max_catastrophes_per_tick]


def propose_actions(world: WorldState, catastrophe_agents: set[str], rng: Random) -> dict[str, Action]:
    proposals: dict[str, Action] = {}

    # Sort agents by id for deterministic RNG consumption order across processes.
    for agent in sorted(world.agents.values(), key=lambda a: a.id):
        if agent.id in catastrophe_agents:
            continue
        if agent.location == "departed":
            continue

        perception = build_perception(agent, world)
        action = decision_engine.select_action(agent, perception, world, rng)
        if action is None:
            # INTERNAL is always available; this is a safety valve.
            action = Action(
                agent_id=agent.id,
                action_type=EventType.INTERNAL,
                target_agents=[],
                location_id=agent.location,
                utility_score=0.0,
                content="Think",
                is_dramatic=False,
                requires_target_available=False,
            )

        proposals[agent.id] = action

    return proposals


@dataclass(frozen=True)
class Conflict:
    type: str
    agents: list[str]
    contested: list[str]


def detect_conflicts(proposals: dict[str, Action]) -> list[Conflict]:
    conflicts: list[Conflict] = []

    for (a_id, a), (b_id, b) in combinations(sorted(proposals.items()), 2):
        # Target contention: shared target(s).
        if (
            a.requires_target_available
            and b.requires_target_available
            and set(a.target_agents)
            and set(b.target_agents)
            and (set(a.target_agents) & set(b.target_agents))
        ):
            contested = sorted(set(a.target_agents) & set(b.target_agents))
            conflicts.append(Conflict(type="target_contention", agents=[a_id, b_id], contested=contested))

        # Incompatible actions: confrontation blocks confide/reveal on same target.
        if set(a.target_agents) & set(b.target_agents):
            types = {a.action_type, b.action_type}
            if EventType.CONFLICT in types and (EventType.CONFIDE in types or EventType.REVEAL in types):
                conflicts.append(Conflict(type="incompatible_actions", agents=[a_id, b_id], contested=[]))

    return conflicts


def downgrade_action(action: Action) -> Action:
    downgrades = {
        EventType.CONFIDE: EventType.INTERNAL,
        EventType.CONFLICT: EventType.INTERNAL,
        EventType.REVEAL: EventType.INTERNAL,
        EventType.CHAT: EventType.OBSERVE,
        EventType.LIE: EventType.INTERNAL,
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
        requires_target_available=False,
    )


def resolve_conflicts(proposals: dict[str, Action], conflicts: list[Conflict], rng: Random) -> list[Action]:
    # Resolve conflicts in descending severity.
    severity = {"incompatible_actions": 3, "target_contention": 2, "location_capacity": 1}
    for conflict in sorted(conflicts, key=lambda c: severity.get(c.type, 0), reverse=True):
        a_id, b_id = conflict.agents
        if a_id not in proposals or b_id not in proposals:
            continue
        a = proposals[a_id]
        b = proposals[b_id]

        cand = [(a_id, a), (b_id, b)]
        cand.sort(key=lambda t: (t[1].priority_class, t[1].utility_score, rng.random()), reverse=True)
        winner_id, _ = cand[0]
        losers = [x for x in (a_id, b_id) if x != winner_id]
        for loser in losers:
            proposals[loser] = downgrade_action(proposals[loser])

    resolved = list(proposals.values())
    resolved.sort(key=lambda a: (a.priority_class, a.utility_score), reverse=True)
    return resolved


def _weighted_choice(weights: list[float], rng: Random) -> int:
    total = float(sum(weights))
    if total <= 0:
        return int(rng.randrange(len(weights)))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if r <= acc:
            return i
    return len(weights) - 1


def choose_actions(resolved_actions: list[Action], rng: Random, max_actions: int) -> list[Action]:
    """Pick up to max_actions actions to execute this tick.

    We sample (without replacement by agent) to avoid one agent dominating every tick.
    INTERNAL actions are downweighted so the simulation doesn't collapse into pure introspection.
    """

    if not resolved_actions or max_actions <= 0:
        return []

    chosen: list[Action] = []
    pool = list(resolved_actions)
    used_agents: set[str] = set()

    # Softmax temperature: lower = more greedy.
    temperature = 0.4

    while pool and len(chosen) < max_actions:
        scores: list[float] = []
        for a in pool:
            s = float(a.utility_score)
            if a.action_type == EventType.INTERNAL:
                s -= 0.35
            if a.action_type == EventType.OBSERVE:
                s -= 0.10
            if a.action_type == EventType.SOCIAL_MOVE:
                s -= 0.20
            scores.append(s)

        m = max(scores)
        weights = [math.exp((s - m) / max(temperature, 1e-6)) for s in scores]
        idx = _weighted_choice(weights, rng)
        act = pool.pop(idx)
        chosen.append(act)
        used_agents.add(act.agent_id)
        pool = [a for a in pool if a.agent_id not in used_agents]

    chosen.sort(key=lambda a: (a.priority_class, a.utility_score), reverse=True)
    return chosen


def _global_tension_proxy(world: WorldState) -> float:
    agents = list(world.agents.values())
    if not agents:
        return 0.0
    return sum(float(a.pacing.stress) for a in agents) / float(len(agents))


def _snapshot_dict(world: WorldState, *, tick_id: int, snapshot_index: int) -> dict[str, Any]:
    # SnapshotState is our canonical typed representation, but we emit dicts for downstream
    # consumers (integration/data-flow.md).
    snap = SnapshotState(
        snapshot_id=f"snap_{snapshot_index:04d}",
        tick_id=tick_id,
        sim_time=world.sim_time,
        event_count=len(world.event_log),
        agents=world.agents,
        secrets=world.secrets,
        locations=world.locations,
        global_tension=_global_tension_proxy(world),
    )
    return snap.to_dict()


def _rand_range(rng: Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _event_participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _find_causal_links(world: WorldState, location_id: str) -> list[str]:
    if not world.event_log:
        return []
    # Prefer linking within the same location; fall back to the latest event.
    for e in reversed(world.event_log[-20:]):
        if e.location_id == location_id:
            return [e.id]
    return [world.event_log[-1].id]


def _dedupe_strs(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _event_content_type(event: Event, world: WorldState) -> str | None:
    metadata = event.content_metadata or {}
    direct = metadata.get("content_type")
    if isinstance(direct, str) and direct:
        return direct

    claim_like_id = metadata.get("secret_id") or metadata.get("propagated_claim_id")
    if isinstance(claim_like_id, str) and claim_like_id:
        claim = world.definition.all_claims.get(claim_like_id)
        if claim and claim.content_type:
            return claim.content_type

    for delta in event.deltas:
        if delta.kind == DeltaKind.BELIEF and delta.attribute:
            claim = world.definition.all_claims.get(delta.attribute)
            if claim and claim.content_type:
                return claim.content_type
    return None


def _populate_event_entities(event: Event, world: WorldState) -> None:
    entities = event.entities if event.entities is not None else EventEntities()

    locations = [event.location_id]
    if event.type == EventType.SOCIAL_MOVE:
        for delta in event.deltas:
            if delta.kind == DeltaKind.AGENT_LOCATION and delta.value:
                locations.append(str(delta.value))
                break

    claims: list[str] = []
    for delta in event.deltas:
        if delta.kind in (DeltaKind.BELIEF, DeltaKind.SECRET_STATE):
            claim_id = str(delta.attribute or "")
            if claim_id:
                claims.append(claim_id)

    concepts: list[str] = []
    content_type = _event_content_type(event, world)
    if content_type:
        concepts.extend(CONTENT_TYPE_TO_CONCEPTS.get(content_type, []))

    entities.locations = _dedupe_strs([*entities.locations, *locations])
    entities.claims = _dedupe_strs([*entities.claims, *claims])
    entities.concepts = _dedupe_strs([*entities.concepts, *concepts])
    entities.artifacts = _dedupe_strs(list(entities.artifacts))
    entities.factions = _dedupe_strs(list(entities.factions))
    entities.institutions = _dedupe_strs(list(entities.institutions))
    event.entities = entities


def _event_notable_proxy(event: Event) -> float:
    return NOTABLE_EVENT_PROXY_BY_TYPE.get(event.type, 0.1)


def _event_by_id(
    world: WorldState,
    event_id: str,
    in_tick_events: dict[str, Event],
) -> Event | None:
    if event_id in in_tick_events:
        return in_tick_events[event_id]
    for prior in reversed(world.event_log):
        if prior.id == event_id:
            return prior
    return None


def _update_notable_event_ids(
    world: WorldState,
    memory: LocationMemory,
    event: Event,
    in_tick_events: dict[str, Event],
) -> None:
    def _rank_key(event_id: str) -> tuple[float, int, int]:
        found = _event_by_id(world, event_id, in_tick_events)
        if found is None:
            return (0.0, -1, -1)
        return (_event_notable_proxy(found), found.tick_id, found.order_in_tick)

    event_ids = [eid for eid in memory.notable_event_ids if eid != event.id]
    event_ids.append(event.id)
    ranked = sorted(event_ids, key=_rank_key, reverse=True)
    memory.notable_event_ids = ranked[:10]


def _apply_artifact_delta(canon: WorldCanon, delta: StateDelta, event: Event) -> None:
    artifact_id = str(delta.attribute or "")
    if not artifact_id:
        return
    artifact = canon.artifacts.get(artifact_id)
    if artifact is None:
        artifact = CanonArtifact(id=artifact_id, name=artifact_id, artifact_type="unknown")
        canon.artifacts[artifact_id] = artifact
    artifact.state[str(delta.agent)] = delta.value
    if not artifact.first_seen_event_id:
        artifact.first_seen_event_id = event.id
    artifact.history = [*artifact.history[-19:], event.id]


def _apply_faction_delta(canon: WorldCanon, delta: StateDelta, event: Event) -> None:
    faction_id = str(delta.attribute or "")
    if not faction_id:
        return
    faction = canon.factions.get(faction_id)
    if faction is None:
        faction = CanonFaction(id=faction_id, name=faction_id)
        canon.factions[faction_id] = faction
    if isinstance(delta.value, (int, float)):
        faction.state[str(delta.agent)] = float(delta.value)
    faction.history = [*faction.history[-19:], event.id]


def _apply_institution_delta(canon: WorldCanon, delta: StateDelta, event: Event) -> None:
    institution_id = str(delta.attribute or "")
    if not institution_id:
        return
    institution = canon.institutions.get(institution_id)
    if institution is None:
        institution = CanonInstitution(id=institution_id, name=institution_id)
        canon.institutions[institution_id] = institution
    if isinstance(delta.value, (int, float)):
        institution.state[str(delta.agent)] = float(delta.value)
    institution.history = [*institution.history[-19:], event.id]


def _update_canon_for_event(
    world: WorldState,
    event: Event,
    in_tick_events: dict[str, Event],
) -> None:
    if world.canon is None:
        world.canon = init_canon_from_world(world.definition)
    canon = world.canon

    location_id = event.location_id
    memory = canon.location_memory.setdefault(location_id, LocationMemory())
    memory.visit_count += 1
    memory.last_event_tick = event.tick_id

    stain_amount = LOCATION_STAIN_BY_EVENT_TYPE.get(event.type, 0.0)
    if stain_amount > 0.0:
        new_residue = min(1.0, float(memory.tension_residue) + stain_amount)
        memory.tension_residue = new_residue
        _update_notable_event_ids(world, memory, event, in_tick_events)
        event.deltas.append(
            StateDelta(
                kind=DeltaKind.LOCATION_MEMORY,
                agent=event.source_agent,
                attribute=location_id,
                op=DeltaOp.SET,
                value=new_residue,
                reason_code="LOCATION_TENSION_STAIN",
                reason_display=f"Tension residue updated at {location_id}",
            )
        )

    for delta in event.deltas:
        if delta.kind == DeltaKind.ARTIFACT_STATE:
            _apply_artifact_delta(canon, delta, event)
        elif delta.kind == DeltaKind.FACTION_STATE:
            _apply_faction_delta(canon, delta, event)
        elif delta.kind == DeltaKind.INSTITUTION_STATE:
            _apply_institution_delta(canon, delta, event)
        elif delta.kind == DeltaKind.LOCATION_MEMORY:
            target_loc = str(delta.attribute or event.location_id)
            target_memory = canon.location_memory.setdefault(target_loc, LocationMemory())
            if isinstance(delta.value, (int, float)):
                target_memory.tension_residue = clamp(float(delta.value), 0.0, 1.0)

    canon.last_event_id = event.id


def _decay_location_memory(canon: WorldCanon) -> None:
    for location_id in sorted(canon.location_memory):
        memory = canon.location_memory[location_id]
        memory.tension_residue = max(0.0, float(memory.tension_residue) * 0.97)


def _snapshot_claim_states(world: WorldState) -> None:
    if world.canon is None:
        world.canon = init_canon_from_world(world.definition)

    claim_states: dict[str, dict[str, str]] = {}
    for claim_id in sorted(world.definition.all_claims):
        claim_states[claim_id] = {
            agent_id: world.agents[agent_id].beliefs.get(claim_id, BeliefState.UNKNOWN).value
            for agent_id in sorted(world.agents)
        }
    world.canon.claim_states = claim_states


def _generate_observe_deltas(source: AgentState, observed_agent_id: Optional[str], rng: Random) -> list[StateDelta]:
    deltas: list[StateDelta] = []

    # Minimal, story-shaped heuristics for the dinner party:
    # observing Elena or Marcus can generate suspicion of the affair.
    if observed_agent_id in ("elena", "marcus") and source.beliefs.get("secret_affair_01") == BeliefState.UNKNOWN:
        if rng.random() < 0.35:
            deltas.append(
                StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent=source.id,
                    agent_b=observed_agent_id,
                    attribute="secret_affair_01",
                    op=DeltaOp.SET,
                    value=BeliefState.SUSPECTS.value,
                    reason_code="SUSPICIOUS_BODY_LANGUAGE",
                    reason_display="Noticed familiarity that didn't match the story.",
                )
            )
            deltas.append(
                StateDelta(
                    kind=DeltaKind.AGENT_EMOTION,
                    agent=source.id,
                    attribute="suspicion",
                    op=DeltaOp.ADD,
                    value=0.2,
                    reason_code="SUSPICION_RISES",
                    reason_display="A small detail didn't add up.",
                )
            )

    return deltas


def action_to_event(action: Action, world: WorldState, tick_id: int, order: int, rng: Random) -> Event:
    event_id = _generate_event_id(world)
    causal_links = _find_causal_links(world, action.location_id)
    deltas: list[StateDelta] = []
    content_metadata: dict[str, Any] | None = None

    match action.action_type:
        case EventType.CHAT:
            for target_id in action.target_agents:
                deltas.append(
                    StateDelta(
                        kind=DeltaKind.RELATIONSHIP,
                        agent=target_id,
                        agent_b=action.agent_id,
                        attribute="affection",
                        op=DeltaOp.ADD,
                        value=_rand_range(rng, 0.02, 0.08),
                        reason_code="PLEASANT_CONVERSATION",
                        reason_display=f"Chatting with {world.agents[action.agent_id].name}",
                    )
                )

        case EventType.OBSERVE:
            observed = action.metadata.get("observed_agent")
            observed_id = str(observed) if isinstance(observed, str) else None
            content_metadata = {"observed_agent": observed_id} if observed_id else None
            deltas.extend(_generate_observe_deltas(world.agents[action.agent_id], observed_id, rng))

            witnessed_type = action.metadata.get("observed_event_type") or action.metadata.get("overheard_event_type")
            witnessed_id = action.metadata.get("observed_event") or action.metadata.get("overheard_event")
            witnessed_source = action.metadata.get("observed_source_agent") or action.metadata.get("overheard_source_agent")
            if isinstance(witnessed_type, str):
                content_metadata = {
                    **(content_metadata or {}),
                    "witnessed_event": str(witnessed_id) if isinstance(witnessed_id, str) else None,
                    "witnessed_event_type": witnessed_type,
                    "witnessed_source_agent": str(witnessed_source) if isinstance(witnessed_source, str) else None,
                }

                # Witnessing drama changes emotions, which drives future actions.
                if witnessed_type in ("conflict", "catastrophe"):
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.AGENT_EMOTION,
                            agent=action.agent_id,
                            attribute="suspicion",
                            op=DeltaOp.ADD,
                            value=0.15,
                            reason_code="WITNESSED_DRAMA",
                            reason_display="Tension rose after witnessing conflict.",
                        )
                    )
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.AGENT_EMOTION,
                            agent=action.agent_id,
                            attribute="anger",
                            op=DeltaOp.ADD,
                            value=0.10,
                            reason_code="WITNESSED_CONFRONTATION",
                            reason_display="Witnessing confrontation stirred anger.",
                        )
                    )
                elif witnessed_type == "reveal":
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.AGENT_EMOTION,
                            agent=action.agent_id,
                            attribute="suspicion",
                            op=DeltaOp.ADD,
                            value=0.10,
                            reason_code="WITNESSED_REVEAL",
                            reason_display="A revelation made motives feel uncertain.",
                        )
                    )

        case EventType.SOCIAL_MOVE:
            dest = action.metadata.get("destination")
            dest_id = str(dest) if isinstance(dest, str) else ""
            deltas.append(
                StateDelta(
                    kind=DeltaKind.AGENT_LOCATION,
                    agent=action.agent_id,
                    attribute="",
                    op=DeltaOp.SET,
                    value=dest_id,
                    reason_code="LOCATION_CHANGE",
                    reason_display=f"{world.agents[action.agent_id].name} moves to {dest_id}",
                )
            )
            content_metadata = {"destination": dest_id}

        case EventType.REVEAL:
            secret_id = action.metadata.get("secret_id")
            if isinstance(secret_id, str):
                content_metadata = {"secret_id": secret_id}
                for target_id in action.target_agents:
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.BELIEF,
                            agent=target_id,
                            agent_b=action.agent_id,
                            attribute=secret_id,
                            op=DeltaOp.SET,
                            value=BeliefState.BELIEVES_TRUE.value,
                            reason_code="DIRECT_REVEAL",
                            reason_display=f"{world.agents[action.agent_id].name} revealed the truth",
                        )
                    )

        case EventType.CONFLICT:
            if action.target_agents:
                target_id = action.target_agents[0]
                deltas.append(
                    StateDelta(
                        kind=DeltaKind.RELATIONSHIP,
                        agent=action.agent_id,
                        agent_b=target_id,
                        attribute="trust",
                        op=DeltaOp.ADD,
                        value=_rand_range(rng, -0.4, -0.2),
                        reason_code="CONFRONTATION",
                        reason_display=f"Confronted {world.agents[target_id].name}",
                    )
                )
                deltas.append(
                    StateDelta(
                        kind=DeltaKind.RELATIONSHIP,
                        agent=target_id,
                        agent_b=action.agent_id,
                        attribute="trust",
                        op=DeltaOp.ADD,
                        value=_rand_range(rng, -0.3, -0.2),
                        reason_code="ACCUSED_BY",
                        reason_display=f"Accused by {world.agents[action.agent_id].name}",
                    )
                )
            deltas.append(
                StateDelta(
                    kind=DeltaKind.AGENT_EMOTION,
                    agent=action.agent_id,
                    attribute="anger",
                    op=DeltaOp.ADD,
                    value=_rand_range(rng, 0.1, 0.3),
                    reason_code="CONFRONTATION_INITIATED",
                    reason_display="Anger from confrontation",
                )
            )

        case EventType.PHYSICAL:
            content = (action.content or "").lower()
            if "pour" in content or "refill" in content or "serve" in content:
                # Pouring/refilling is a social action: multiple people may drink.
                recipients = sorted({action.agent_id, *action.target_agents})
                for rid in recipients:
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.AGENT_RESOURCE,
                            agent=rid,
                            attribute="alcohol_level",
                            op=DeltaOp.ADD,
                            value=0.15,
                            reason_code="ALCOHOL_SERVED",
                            reason_display="Wine was poured",
                        )
                    )

            if "drink" in content:
                deltas.append(
                    StateDelta(
                        kind=DeltaKind.AGENT_RESOURCE,
                        agent=action.agent_id,
                        attribute="alcohol_level",
                        op=DeltaOp.ADD,
                        value=0.15,
                        reason_code="ALCOHOL_CONSUMED",
                        reason_display=f"{world.agents[action.agent_id].name} took a drink",
                    )
                )

        case EventType.CONFIDE:
            secret_id = action.metadata.get("secret_id")
            if isinstance(secret_id, str):
                content_metadata = {"secret_id": secret_id}
                for target_id in action.target_agents:
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.BELIEF,
                            agent=target_id,
                            agent_b=action.agent_id,
                            attribute=secret_id,
                            op=DeltaOp.SET,
                            value=BeliefState.BELIEVES_TRUE.value,
                            reason_code="CONFIDED_SECRET",
                            reason_display=f"{world.agents[action.agent_id].name} confided a secret",
                        )
                    )
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.RELATIONSHIP,
                            agent=target_id,
                            agent_b=action.agent_id,
                            attribute="trust",
                            op=DeltaOp.ADD,
                            value=0.15,
                            reason_code="TRUST_THROUGH_VULNERABILITY",
                            reason_display="Vulnerability deepened trust",
                        )
                    )
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.RELATIONSHIP,
                            agent=action.agent_id,
                            agent_b=target_id,
                            attribute="trust",
                            op=DeltaOp.ADD,
                            value=0.10,
                            reason_code="CONFIDING_BOND",
                            reason_display="Confiding creates reciprocal trust",
                        )
                    )

        case EventType.LIE:
            secret_id = action.metadata.get("secret_id")
            if isinstance(secret_id, str):
                content_metadata = {"secret_id": secret_id}
                for target_id in action.target_agents:
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.BELIEF,
                            agent=target_id,
                            agent_b=action.agent_id,
                            attribute=secret_id,
                            op=DeltaOp.SET,
                            value=BeliefState.BELIEVES_FALSE.value,
                            reason_code="DELIBERATE_MISDIRECTION",
                            reason_display=f"{world.agents[action.agent_id].name} lied about {secret_id}",
                        )
                    )
                    deltas.append(
                        StateDelta(
                            kind=DeltaKind.RELATIONSHIP,
                            agent=target_id,
                            agent_b=action.agent_id,
                            attribute="trust",
                            op=DeltaOp.ADD,
                            value=0.05,
                            reason_code="BELIEVED_EXPLANATION",
                            reason_display="Believed the explanation",
                        )
                    )

        case EventType.INTERNAL:
            # INTERNAL actions are usually silent, but allow schema-neutral metadata.
            if action.metadata:
                content_metadata = dict(action.metadata)

        case _:
            pass

    event = Event(
        id=event_id,
        sim_time=world.sim_time,
        tick_id=tick_id,
        order_in_tick=order,
        type=action.action_type,
        source_agent=action.agent_id,
        target_agents=list(action.target_agents),
        location_id=action.location_id,
        causal_links=causal_links,
        deltas=deltas,
        description=action.content,
        content_metadata=content_metadata,
        metrics=EventMetrics(),
    )
    _populate_event_entities(event, world)
    return event


def generate_catastrophe_event(agent: AgentState, subtype: str, world: WorldState, tick_id: int, order: int, rng: Random) -> Event:
    event_id = _generate_event_id(world)
    causal_links = _find_causal_links(world, agent.location)

    # Targets: everyone else at location (sorted for determinism).
    targets = sorted(a.id for a in world.agents.values() if a.location == agent.location and a.id != agent.id)

    deltas: list[StateDelta] = []

    # Catastrophes often rupture relationships.
    for t in targets:
        deltas.append(
            StateDelta(
                kind=DeltaKind.RELATIONSHIP,
                agent=agent.id,
                agent_b=t,
                attribute="trust",
                op=DeltaOp.ADD,
                value=_rand_range(rng, -0.5, -0.25),
                reason_code="CATASTROPHE_OUTBURST",
                reason_display="Said something that couldn't be taken back.",
            )
        )

    # Involuntary confession: reveal one of the agent's true beliefs to everyone present.
    revealable = sorted(sid for sid, b in agent.beliefs.items() if b == BeliefState.BELIEVES_TRUE)
    secret_id = revealable[0] if revealable else None
    if secret_id:
        for t in targets:
            deltas.append(
                StateDelta(
                    kind=DeltaKind.BELIEF,
                    agent=t,
                    agent_b=agent.id,
                    attribute=secret_id,
                    op=DeltaOp.SET,
                    value=BeliefState.BELIEVES_TRUE.value,
                    reason_code="CATASTROPHE_CONFESSION",
                    reason_display="A truth slipped out under pressure.",
                )
            )

    event = Event(
        id=event_id,
        sim_time=world.sim_time,
        tick_id=tick_id,
        order_in_tick=order,
        type=EventType.CATASTROPHE,
        source_agent=agent.id,
        target_agents=targets,
        location_id=agent.location,
        causal_links=causal_links,
        deltas=deltas,
        description=f"Catastrophe ({subtype})",
        content_metadata={"subtype": subtype, "secret_id": secret_id} if secret_id else {"subtype": subtype},
        metrics=EventMetrics(),
    )
    _populate_event_entities(event, world)
    return event


def generate_witness_events(primary_events: list[Event], world: WorldState, tick_id: int, start_order: int, rng: Random) -> list[Event]:
    witness_events: list[Event] = []
    order = start_order

    for event in primary_events:
        if event.type == EventType.INTERNAL:
            continue

        participants = _event_participants(event)

        if event.type in (EventType.CONFLICT, EventType.CATASTROPHE, EventType.REVEAL):
            # Same-location witnesses: sample at most 1 per primary event to keep event rates realistic.
            witnesses = sorted(
                (a for a in world.agents.values() if a.location == event.location_id and a.id not in participants),
                key=lambda a: a.id,
            )
            if witnesses and rng.random() <= 0.6:
                agent = rng.choice(witnesses)
                observe_action = Action(
                    agent_id=agent.id,
                    action_type=EventType.OBSERVE,
                    target_agents=[],
                    location_id=agent.location,
                    utility_score=0.0,
                    content=f"Witnessed {event.type.value}",
                    metadata={
                        "observed_event": event.id,
                        "observed_event_type": event.type.value,
                        "observed_source_agent": event.source_agent,
                    },
                    is_dramatic=False,
                    requires_target_available=False,
                )
                obs = action_to_event(observe_action, world, tick_id, order, rng)
                obs.causal_links = [event.id]
                witness_events.append(obs)
                order += 1

            # Overhear events from adjacent locations that list this location in overhear_from.
            for listener_loc in sorted(world.locations.values(), key=lambda loc: loc.id):
                if event.location_id not in listener_loc.overhear_from:
                    continue
                overhearers = sorted(
                    (a for a in world.agents.values() if a.location == listener_loc.id and a.id not in participants),
                    key=lambda a: a.id,
                )
                if not overhearers:
                    continue
                if rng.random() >= listener_loc.overhear_probability:
                    continue
                agent = rng.choice(overhearers)
                observe_action = Action(
                    agent_id=agent.id,
                    action_type=EventType.OBSERVE,
                    target_agents=[],
                    location_id=agent.location,
                    utility_score=0.0,
                    content=f"Overheard {event.type.value}",
                    metadata={
                        "overheard_event": event.id,
                        "overheard_event_type": event.type.value,
                        "overheard_source_agent": event.source_agent,
                    },
                    is_dramatic=False,
                    requires_target_available=False,
                )
                obs = action_to_event(observe_action, world, tick_id, order, rng)
                obs.causal_links = [event.id]
                witness_events.append(obs)
                order += 1

        if event.type in (EventType.CHAT, EventType.SOCIAL_MOVE):
            claim_refs = sorted(
                {
                    d.attribute
                    for d in event.deltas
                    if d.kind == DeltaKind.BELIEF
                    and d.attribute
                    and d.attribute in world.definition.claims
                    and world.definition.claims[d.attribute].claim_type != "secret"
                }
            )
            if not claim_refs:
                continue

            nearby = sorted(
                (a for a in world.agents.values() if a.location == event.location_id and a.id not in participants),
                key=lambda a: a.id,
            )
            if not nearby:
                continue

            for claim_id in claim_refs:
                claim = world.definition.claims.get(claim_id)
                if claim is None:
                    continue
                spread_chance = clamp(float(claim.propagation_rate), 0.0, 1.0)
                if spread_chance <= 0.0:
                    continue

                eligible = [a for a in nearby if a.beliefs.get(claim_id, BeliefState.UNKNOWN) == BeliefState.UNKNOWN]
                if not eligible:
                    continue
                if rng.random() > spread_chance:
                    continue

                observer = rng.choice(eligible)
                rumor_event = Event(
                        id=_generate_event_id(world),
                        sim_time=world.sim_time,
                        tick_id=tick_id,
                        order_in_tick=order,
                        type=EventType.OBSERVE,
                        source_agent=observer.id,
                        target_agents=[],
                        location_id=observer.location,
                        causal_links=[event.id],
                        deltas=[
                            StateDelta(
                                kind=DeltaKind.BELIEF,
                                agent=observer.id,
                                agent_b=event.source_agent,
                                attribute=claim_id,
                                op=DeltaOp.SET,
                                value=BeliefState.SUSPECTS.value,
                                reason_code="RUMOR_OVERHEARD",
                                reason_display="A nearby conversation seeded a new suspicion.",
                            )
                        ],
                        description=f"Overheard rumor ({claim_id})",
                        content_metadata={
                            "overheard_event": event.id,
                            "overheard_event_type": event.type.value,
                            "overheard_source_agent": event.source_agent,
                            "propagated_claim_id": claim_id,
                        },
                        metrics=EventMetrics(),
                    )
                _populate_event_entities(rumor_event, world)
                witness_events.append(rumor_event)
                order += 1

    return witness_events


def apply_delta(world: WorldState, delta: StateDelta) -> None:
    match delta.kind:
        case DeltaKind.AGENT_EMOTION:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            current = float(agent.emotional_state.get(delta.attribute, 0.0))
            if delta.op == DeltaOp.ADD:
                agent.emotional_state[delta.attribute] = clamp(current + float(delta.value), 0.0, 1.0)
            else:
                agent.emotional_state[delta.attribute] = clamp(float(delta.value), 0.0, 1.0)

        case DeltaKind.AGENT_RESOURCE:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            current = float(getattr(agent, delta.attribute, 0.0))
            if delta.op == DeltaOp.ADD:
                setattr(agent, delta.attribute, clamp(current + float(delta.value), 0.0, 1.0))
            else:
                setattr(agent, delta.attribute, clamp(float(delta.value), 0.0, 1.0))

        case DeltaKind.AGENT_LOCATION:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            agent.location = str(delta.value)

        case DeltaKind.RELATIONSHIP:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            if not delta.agent_b:
                return
            rel = agent.relationships.get(delta.agent_b)
            if not rel:
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
                # Trust repair hysteresis (pacing-physics.md): positive trust changes are 3x harder.
                if attr == "trust" and change > 0:
                    change = change / world.definition.trust_repair_multiplier
                new_val = current + change

            if attr == "obligation":
                new_val = clamp(new_val, 0.0, 1.0)
            else:
                new_val = clamp(new_val, -1.0, 1.0)

            setattr(rel, attr, new_val)

        case DeltaKind.BELIEF:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            secret_id = delta.attribute
            if not secret_id:
                return
            val = str(delta.value)
            try:
                agent.beliefs[secret_id] = BeliefState(val)
            except ValueError:
                # Leave as-is if invalid.
                return

        case DeltaKind.COMMITMENT:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            agent.commitments.append(str(delta.value))

        case DeltaKind.PACING:
            agent = world.agents.get(delta.agent)
            if agent is None:
                return
            attr = delta.attribute
            current = getattr(agent.pacing, attr)
            if delta.op == DeltaOp.SET:
                new_val = float(delta.value)
            else:
                new_val = float(current) + float(delta.value)
            # Keep pacing dimensions normalized for downstream catastrophe checks.
            if attr != "recovery_timer":
                new_val = clamp(float(new_val), 0.0, 1.0)
            setattr(agent.pacing, attr, new_val)

        case DeltaKind.ARTIFACT_STATE | DeltaKind.FACTION_STATE | DeltaKind.INSTITUTION_STATE:
            # Canon deltas are handled in _update_canon_for_event.
            return

        case DeltaKind.LOCATION_MEMORY:
            # LOCATION_MEMORY is a materialized-canon delta, not an agent-state delta.
            return

        case _:
            # WORLD_RESOURCE and SECRET_STATE are out of scope in MVP.
            return


def _pacing_deltas(old: PacingState, new: PacingState, agent_id: str) -> list[StateDelta]:
    deltas: list[StateDelta] = []
    for field_name in ("dramatic_budget", "stress", "composure", "commitment", "recovery_timer", "suppression_count"):
        old_val = getattr(old, field_name)
        new_val = getattr(new, field_name)
        if old_val == new_val:
            continue
        deltas.append(
            StateDelta(
                kind=DeltaKind.PACING,
                agent=agent_id,
                attribute=field_name,
                op=DeltaOp.SET,
                value=new_val,
                reason_code="PACING_UPDATE",
                reason_display="End-of-tick pacing update",
            )
        )
    return deltas


def apply_tick_updates(world: WorldState, tick_events: list[Event], cfg: SimulationConfig | None = None) -> None:
    if world.canon is None:
        world.canon = init_canon_from_world(world.definition)

    # Step 1: Apply event deltas and materialize canon updates in order.
    in_tick_events: dict[str, Event] = {}
    for event in tick_events:
        in_tick_events[event.id] = event
        for delta in event.deltas:
            apply_delta(world, delta)
        _update_canon_for_event(world, event, in_tick_events)

    # Step 2: Update pacing for all agents, collecting pending event deltas.
    pending_event_deltas: dict[str, list[StateDelta]] = {}
    for agent in sorted(world.agents.values(), key=lambda a: a.id):
        if agent.location == "departed":
            continue
        loc = world.locations[agent.location]
        old = PacingState(**vars(agent.pacing))
        new = pacing.end_of_tick_update(agent, tick_events, loc, secrets=world.secrets)

        deltas = _pacing_deltas(old, new, agent.id)
        if deltas:
            # Attach pacing deltas only to events sourced by the same agent.
            # If the agent did not source an event this tick, keep the update on
            # agent state only (no cross-agent event attribution).
            target_event: Optional[Event] = None
            for e in reversed(tick_events):
                if e.source_agent == agent.id:
                    target_event = e
                    break
            if target_event is not None:
                pending_event_deltas.setdefault(target_event.id, []).extend(deltas)

        # Apply to world state.
        agent.pacing = new

    # Step 3: Event finalization (single merge point for derived deltas).
    if pending_event_deltas:
        finalized_events: list[Event] = []
        for event in tick_events:
            derived = pending_event_deltas.get(event.id)
            if not derived:
                finalized_events.append(event)
                continue
            finalized_events.append(
                Event(
                    id=event.id,
                    sim_time=event.sim_time,
                    tick_id=event.tick_id,
                    order_in_tick=event.order_in_tick,
                    type=event.type,
                    source_agent=event.source_agent,
                    target_agents=event.target_agents,
                    location_id=event.location_id,
                    causal_links=event.causal_links,
                    deltas=[*event.deltas, *derived],
                    description=event.description,
                    dialogue=event.dialogue,
                    content_metadata=event.content_metadata,
                    beat_type=event.beat_type,
                    metrics=event.metrics,
                    entities=event.entities,
                )
            )
        tick_events[:] = finalized_events

    # Step 4: End-of-tick canon decay and metadata updates.
    if world.canon is not None:
        _decay_location_memory(world.canon)
        world.canon.canon_version += 1
        world.canon.last_tick = tick_events[-1].tick_id if tick_events else world.tick_id
        if tick_events:
            world.canon.last_event_id = tick_events[-1].id

    # Step 5: Advance sim time.
    time_scale = cfg.time_scale if cfg is not None else 1.0
    world.sim_time += _tick_duration(tick_events, time_scale)


def execute_tick(world: WorldState, tick_id: int, rng: Random, cfg: SimulationConfig) -> list[Event]:
    all_events: list[Event] = []

    # NOTE: Deliberate spec divergence: we intentionally generate catastrophe events first in the
    # tick (before normal action selection). A pacing-physics.md comment says "resolve last", but
    # catastrophe-first yields cleaner chain reactions because fallout is propagated once across all
    # generated events during apply_tick_updates.
    catastrophes = check_catastrophes(world, cfg)
    catastrophe_agents = {a for a, _ in catastrophes}

    order = 0
    for agent_id, subtype in catastrophes:
        event = generate_catastrophe_event(world.agents[agent_id], subtype, world, tick_id, order, rng)
        all_events.append(event)
        order += 1

    proposals = propose_actions(world, catastrophe_agents, rng)
    conflicts = detect_conflicts(proposals)
    resolved_actions = resolve_conflicts(proposals, conflicts, rng)

    # Execute only a small number of actions per tick (see tick-loop.md Section 2.2 tuning note).
    actions_to_execute = choose_actions(resolved_actions, rng, cfg.max_actions_per_tick)
    for action in actions_to_execute:
        event = action_to_event(action, world, tick_id, order, rng)
        all_events.append(event)
        order += 1

    witness = generate_witness_events(all_events, world, tick_id, order, rng)
    all_events.extend(witness)

    return all_events


def termination_condition(world: WorldState, tick_id: int, cfg: SimulationConfig) -> bool:
    if world.sim_time >= cfg.max_sim_time:
        return True
    if tick_id >= cfg.tick_limit:
        return True

    active = [a for a in world.agents.values() if a.location != "departed"]
    if len(active) < 2:
        return True

    # Stalemate: everyone is cooling down and calm.
    if active and all(a.pacing.recovery_timer > 0 for a in active) and all(a.pacing.stress < 0.2 for a in active):
        return True

    if len(world.event_log) >= cfg.event_limit:
        return True

    return False


def run_simulation(world: WorldState, rng: Random, cfg: SimulationConfig) -> tuple[list[Event], list[dict[str, Any]]]:
    """Run the simulation, returning (events, snapshots).

    Snapshots are dictionaries produced from schema.scenes.SnapshotState.
    Snapshot policy: every cfg.snapshot_interval_events events.
    """

    snapshots: list[dict[str, Any]] = []
    world.truncated = False
    world.canon = init_canon_from_world(world.definition, world.canon)

    # Initial snapshot at tick 0.
    snapshots.append(_snapshot_dict(world, tick_id=0, snapshot_index=len(snapshots)))

    last_snapshot_event_count = 0

    tick_id = 0
    while not termination_condition(world, tick_id, cfg):
        tick_events = execute_tick(world, tick_id, rng, cfg)
        remaining = cfg.event_limit - len(world.event_log)
        if remaining <= 0:
            break
        if len(tick_events) > remaining:
            world.truncated = True
            tick_events = tick_events[:remaining]

        apply_tick_updates(world, tick_events, cfg)
        # Ensure monotonic sim_time: events generated at the current world.sim_time.
        world.event_log.extend(tick_events)

        # Snapshot based on event count (Decision #2 in data-flow.md).
        if len(world.event_log) - last_snapshot_event_count >= cfg.snapshot_interval_events:
            snapshots.append(_snapshot_dict(world, tick_id=tick_id, snapshot_index=len(snapshots)))
            last_snapshot_event_count = len(world.event_log)

        tick_id += 1
        world.tick_id = tick_id

        if len(world.event_log) >= cfg.event_limit:
            break

    _snapshot_claim_states(world)
    return world.event_log, snapshots
