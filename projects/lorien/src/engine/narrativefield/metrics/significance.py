from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from narrativefield.schema.events import DeltaKind, DeltaOp, Event, EventType


_BELIEF_UNKNOWN = "unknown"
_BELIEF_SUSPECTS = "suspects"
_BELIEF_TRUE = "believes_true"
_BELIEF_FALSE = "believes_false"
_KNOWN_BELIEF_STATES = {_BELIEF_SUSPECTS, _BELIEF_TRUE, _BELIEF_FALSE}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ordered_events(all_events: list[Event]) -> list[Event]:
    return sorted(
        all_events,
        key=lambda e: (float(e.sim_time), int(e.tick_id), int(e.order_in_tick), str(e.id)),
    )


def _incoming_degree(events: list[Event]) -> dict[str, int]:
    incoming: dict[str, int] = defaultdict(int)
    for event in events:
        for parent_id in event.causal_links:
            incoming[str(parent_id)] += 1
    return dict(incoming)


def _first_event_by_location(events: list[Event]) -> dict[str, str]:
    first: dict[str, str] = {}
    for event in events:
        first.setdefault(event.location_id, event.id)
    return first


def _parse_belief_state(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in {_BELIEF_UNKNOWN, _BELIEF_SUSPECTS, _BELIEF_TRUE, _BELIEF_FALSE}:
        return v
    return None


def _belief_transition_weight(old_state: str, new_state: str) -> float:
    if old_state == _BELIEF_UNKNOWN and new_state in {_BELIEF_TRUE, _BELIEF_FALSE}:
        return 1.0
    if old_state == _BELIEF_SUSPECTS and new_state == _BELIEF_TRUE:
        return 0.5
    if old_state != new_state:
        return 0.3
    return 0.0


def _delta_magnitude_and_reveal_flag(
    event: Event,
    belief_state: dict[tuple[str, str], str],
    relationship_state: dict[tuple[str, str, str], float],
) -> tuple[float, bool]:
    weighted_sum = 0.0
    has_unknown_to_known_belief = False

    for delta in event.deltas:
        if delta.kind == DeltaKind.BELIEF:
            secret_id = str(delta.attribute or "")
            if not secret_id:
                continue
            new_state = _parse_belief_state(delta.value)
            if new_state is None:
                continue

            key = (str(delta.agent), secret_id)
            old_state = belief_state.get(key, _BELIEF_UNKNOWN)
            weighted_sum += _belief_transition_weight(old_state, new_state)

            if old_state == _BELIEF_UNKNOWN and new_state in _KNOWN_BELIEF_STATES:
                has_unknown_to_known_belief = True

            belief_state[key] = new_state
            continue

        if delta.kind == DeltaKind.RELATIONSHIP:
            if not isinstance(delta.value, (int, float)):
                continue
            rel_key = (str(delta.agent), str(delta.agent_b or ""), str(delta.attribute or ""))
            if delta.op == DeltaOp.ADD:
                change = float(delta.value)
                weighted_sum += abs(change)
                relationship_state[rel_key] = relationship_state.get(rel_key, 0.0) + change
                continue
            if delta.op == DeltaOp.SET:
                new_val = float(delta.value)
                old_val = relationship_state.get(rel_key, 0.0)
                weighted_sum += abs(new_val - old_val)
                relationship_state[rel_key] = new_val
                continue

        if delta.kind == DeltaKind.COMMITMENT:
            weighted_sum += 0.6
            continue
        if delta.kind == DeltaKind.AGENT_EMOTION:
            weighted_sum += 0.3
            continue
        if delta.kind == DeltaKind.PACING:
            weighted_sum += 0.2
            continue
        if delta.kind == DeltaKind.WORLD_RESOURCE:
            weighted_sum += 0.8
            continue
        if delta.kind == DeltaKind.ARTIFACT_STATE:
            weighted_sum += 0.8
            continue
        if delta.kind == DeltaKind.FACTION_STATE:
            weighted_sum += 0.7
            continue
        if delta.kind == DeltaKind.INSTITUTION_STATE:
            weighted_sum += 0.7
            continue
        if delta.kind == DeltaKind.LOCATION_MEMORY:
            weighted_sum += 0.3
            continue

    return min(1.0, weighted_sum / 2.0), has_unknown_to_known_belief


def _breadth(event: Event) -> float:
    participants = {str(agent_id) for agent_id in event.target_agents if str(agent_id)}
    if event.source_agent:
        participants.add(str(event.source_agent))
    agent_breadth = len(participants)
    entity_breadth = event.entities.total_refs if event.entities else 0
    return min(1.0, (agent_breadth + entity_breadth * 0.5) / 5.0)


def _causal_centrality(event: Event, incoming_degree: dict[str, int]) -> float:
    out_degree = len(event.causal_links)
    in_degree = int(incoming_degree.get(event.id, 0))
    return min(1.0, (in_degree + out_degree) / 6.0)


def _novelty(
    event: Event,
    *,
    first_event_by_location: dict[str, str],
    has_unknown_to_known_belief: bool,
) -> float:
    flags = [
        1.0 if first_event_by_location.get(event.location_id) == event.id else 0.0,
        1.0 if has_unknown_to_known_belief else 0.0,
        1.0 if event.type == EventType.CATASTROPHE else 0.0,
        1.0 if len(event.target_agents) >= 3 else 0.0,
    ]
    return sum(flags) / 4.0


@dataclass(frozen=True)
class _ScoreContext:
    ordered_events: list[Event]
    incoming_degree: dict[str, int]
    first_event_by_location: dict[str, str]


def _build_context(all_events: list[Event]) -> _ScoreContext:
    ordered = _ordered_events(all_events)
    return _ScoreContext(
        ordered_events=ordered,
        incoming_degree=_incoming_degree(ordered),
        first_event_by_location=_first_event_by_location(ordered),
    )


def _score_events(context: _ScoreContext) -> dict[int, float]:
    belief_state: dict[tuple[str, str], str] = {}
    relationship_state: dict[tuple[str, str, str], float] = {}
    score_by_object_id: dict[int, float] = {}

    for event in context.ordered_events:
        delta_magnitude, has_unknown_to_known_belief = _delta_magnitude_and_reveal_flag(
            event,
            belief_state=belief_state,
            relationship_state=relationship_state,
        )
        causal_centrality = _causal_centrality(event, context.incoming_degree)
        novelty = _novelty(
            event,
            first_event_by_location=context.first_event_by_location,
            has_unknown_to_known_belief=has_unknown_to_known_belief,
        )
        breadth = _breadth(event)
        significance = _clamp01(
            0.40 * delta_magnitude
            + 0.25 * causal_centrality
            + 0.20 * novelty
            + 0.15 * breadth
        )
        score_by_object_id[id(event)] = significance

    return score_by_object_id


def compute_significance(event: Event, all_events: list[Event]) -> float:
    """
    Compute significance for a single event.

    Deterministic over the full event list and canonical event ordering.
    """
    context = _build_context(all_events)
    scores = _score_events(context)
    direct = scores.get(id(event))
    if direct is not None:
        return float(direct)

    # Fallback for callers that pass an equivalent-but-distinct Event object.
    for candidate in context.ordered_events:
        if candidate.id == event.id:
            return float(scores.get(id(candidate), 0.0))
    return 0.0


def populate_significance(events: list[Event]) -> None:
    """Compute and assign significance for every event in-place."""
    context = _build_context(events)
    scores = _score_events(context)
    for event in events:
        event.metrics.significance = float(scores.get(id(event), 0.0))
