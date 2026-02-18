from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .events import DeltaKind, Event


@dataclass
class IndexTables:
    """
    Read-optimized indices over the event log.
    Mirrors specs/integration/data-flow.md Section 4.
    """

    event_by_id: dict[str, Event]
    events_by_agent: dict[str, list[str]]
    events_by_location: dict[str, list[str]]
    participants_by_event: dict[str, list[str]]
    events_by_secret: dict[str, list[str]]
    events_by_pair: dict[tuple[str, str], list[str]]
    forward_links: dict[str, list[str]]


def build_index_tables(events: list[Event]) -> IndexTables:
    """Build all index tables in a single pass over the event log."""
    tables = IndexTables(
        event_by_id={},
        events_by_agent=defaultdict(list),
        events_by_location=defaultdict(list),
        participants_by_event={},
        events_by_secret=defaultdict(list),
        events_by_pair=defaultdict(list),
        forward_links=defaultdict(list),
    )

    for event in events:
        tables.event_by_id[event.id] = event

        participants = [event.source_agent] + [t for t in event.target_agents if t != event.source_agent]
        tables.participants_by_event[event.id] = participants

        # Agent timelines
        for agent_id in participants:
            tables.events_by_agent[agent_id].append(event.id)

        # Location timeline
        tables.events_by_location[event.location_id].append(event.id)

        # Secret touches
        for delta in event.deltas:
            if delta.kind in (DeltaKind.BELIEF, DeltaKind.SECRET_STATE):
                secret_id = delta.attribute
                if secret_id:
                    tables.events_by_secret[secret_id].append(event.id)

        # Pair interactions
        for i, a in enumerate(participants):
            for b in participants[i + 1 :]:
                key = tuple(sorted((a, b)))
                tables.events_by_pair[key].append(event.id)

        # Forward causal links
        for parent_id in event.causal_links:
            tables.forward_links[parent_id].append(event.id)

    # Convert defaultdicts to plain dicts for serialization friendliness.
    tables.events_by_agent = dict(tables.events_by_agent)
    tables.events_by_location = dict(tables.events_by_location)
    tables.events_by_secret = dict(tables.events_by_secret)
    tables.events_by_pair = dict(tables.events_by_pair)
    tables.forward_links = dict(tables.forward_links)

    return tables

