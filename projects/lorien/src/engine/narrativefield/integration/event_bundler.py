"""Narrative event bundler: compresses micro-events for viz/extraction.

Sits between the metrics pipeline and scene segmentation.
By design, SOCIAL_MOVE events are preserved so segmentation can detect
location boundaries from explicit movement events.

Current bundling behavior only attaches low-significance OBSERVE events
as witness metadata on nearby meaningful events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from narrativefield.schema.events import DeltaKind, Event, EventType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundlerConfig:
    """Tunable thresholds for event bundling."""

    # Legacy no-op fields kept for backwards-compatible config wiring.
    move_squash_dt: float = 2.0

    # Legacy no-op fields kept for backwards-compatible config wiring.
    move_attach_dt: float = 6.0

    # observe attach: max sim-time gap (minutes) for attaching a low-importance
    # observe to a nearby meaningful event in the same location.
    observe_attach_dt: float = 2.0

    # observe events with significance above this are kept as-is.
    observe_significance_threshold: float = 0.3


DEFAULT_BUNDLER_CONFIG = BundlerConfig()


@dataclass
class BundleStats:
    """Bookkeeping for what the bundler did."""

    input_count: int = 0
    output_count: int = 0
    moves_squashed: int = 0
    moves_attached: int = 0
    observes_attached: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "moves_squashed": self.moves_squashed,
            "moves_attached": self.moves_attached,
            "observes_attached": self.observes_attached,
        }


@dataclass
class BundleResult:
    events: list[Event]
    stats: BundleStats
    dropped_ids: set[str] = field(default_factory=set)


def _has_meaningful_deltas(event: Event) -> bool:
    return any(d.kind in {DeltaKind.BELIEF, DeltaKind.RELATIONSHIP, DeltaKind.SECRET_STATE} for d in event.deltas)


# ---------------------------------------------------------------------------
# Step 1: Attach low-importance observe events as witness metadata
# ---------------------------------------------------------------------------


def _attach_observes(
    events: list[Event],
    config: BundlerConfig,
) -> tuple[list[Event], set[str], dict[str, str], int]:
    """Attach low-importance observe events as witnesses[] on nearby events.

    An observe is "low-importance" when it has no belief/relationship deltas
    and its significance is below the threshold.

    The observe is attached to the nearest *next* non-observe event in the
    same location within Δt.  If no suitable host is found, the observe is
    kept as-is.

    Returns (result_events, dropped_ids, absorbed_by, attach_count).
    """
    if not events:
        return [], set(), {}, 0

    # Identify low-importance observes.
    low_observe_ids: set[str] = set()
    for e in events:
        if e.type != EventType.OBSERVE:
            continue
        if _has_meaningful_deltas(e):
            continue
        sig = float(e.metrics.significance) if e.metrics else 0.0
        if sig >= config.observe_significance_threshold:
            continue
        low_observe_ids.add(e.id)

    if not low_observe_ids:
        return list(events), set(), {}, 0

    # For each low-importance observe, find the nearest next non-observe event
    # in the same location within Δt.
    host_map: dict[str, list[Event]] = {}  # host_event_id -> [observe_events]
    absorbed_by: dict[str, str] = {}
    attached_ids: set[str] = set()

    for i, event in enumerate(events):
        if event.id not in low_observe_ids:
            continue
        # Search forward for a host.
        for j in range(i + 1, len(events)):
            candidate = events[j]
            if (candidate.sim_time - event.sim_time) > config.observe_attach_dt:
                break
            if candidate.id in low_observe_ids:
                continue
            if candidate.type in {EventType.OBSERVE, EventType.SOCIAL_MOVE}:
                continue
            if candidate.location_id != event.location_id:
                continue
            # Found a host.
            host_map.setdefault(candidate.id, []).append(event)
            attached_ids.add(event.id)
            absorbed_by[event.id] = candidate.id
            break

    # Build result: skip attached observes, augment hosts with witnesses[].
    result: list[Event] = []
    for event in events:
        if event.id in attached_ids:
            continue
        if event.id in host_map:
            meta = dict(event.content_metadata) if event.content_metadata else {}
            existing = meta.get("witnesses")
            witnesses_list: list[dict[str, Any]] = list(existing) if isinstance(existing, list) else []
            for obs in host_map[event.id]:
                witnesses_list.append(
                    {
                        "observer": obs.source_agent,
                        "observed_event": obs.id,
                        "description": obs.description,
                        "sim_time": obs.sim_time,
                    }
                )
            meta["witnesses"] = witnesses_list
            # Merge causal parents from attached observes.
            merged_parents = set(event.causal_links)
            for obs in host_map[event.id]:
                merged_parents.update(obs.causal_links)
            merged_parents -= attached_ids
            result.append(Event(
                id=event.id,
                sim_time=event.sim_time,
                tick_id=event.tick_id,
                order_in_tick=event.order_in_tick,
                type=event.type,
                source_agent=event.source_agent,
                target_agents=event.target_agents,
                location_id=event.location_id,
                causal_links=sorted(merged_parents),
                deltas=event.deltas,
                description=event.description,
                dialogue=event.dialogue,
                content_metadata=meta,
                beat_type=event.beat_type,
                metrics=event.metrics,
            ))
        else:
            result.append(event)

    return result, attached_ids, absorbed_by, len(attached_ids)


# ---------------------------------------------------------------------------
# Step 2: Rewire causal links for any remaining dropped events
# ---------------------------------------------------------------------------


def _rewire_causal_links(
    *,
    events: list[Event],
    all_dropped_ids: set[str],
    absorbed_by: dict[str, str],
    original_parents: dict[str, list[str]],
) -> list[Event]:
    """Rewire causal_links so they skip over dropped events.

    For each event, replace any causal_link pointing to a dropped event with
    the kept event that absorbed it, or (if none) the nearest kept ancestors.
    """
    if not all_dropped_ids:
        return events

    kept_ids = {e.id for e in events}
    kept_by_id = {e.id: e for e in events}

    def _event_key(e: Event) -> tuple[float, int, int]:
        return (float(e.sim_time), int(e.tick_id), int(e.order_in_tick))

    def _follow_absorbed(eid: str) -> str | None:
        cur = eid
        # Follow chains defensively (should be short).
        for _ in range(8):
            nxt = absorbed_by.get(cur)
            if not nxt:
                break
            cur = nxt
        return cur if cur in kept_ids else None

    def _resolve_to_kept(eid: str, *, seen: set[str], max_key: tuple[float, int, int]) -> set[str]:
        # If this id was absorbed into a kept event, link to that kept event.
        absorbed = _follow_absorbed(eid)
        if absorbed and _event_key(kept_by_id[absorbed]) <= max_key:
            return {absorbed}
        if eid in kept_ids and _event_key(kept_by_id[eid]) <= max_key:
            return {eid}
        if eid not in all_dropped_ids:
            return set()
        # Fall back to ancestors from the original event stream.
        out: set[str] = set()
        for parent in original_parents.get(eid, []):
            if parent in seen:
                continue
            seen.add(parent)
            out |= _resolve_to_kept(parent, seen=seen, max_key=max_key)
        return out

    result: list[Event] = []
    for event in events:
        max_key = _event_key(event)
        new_links: set[str] = set()
        for link in event.causal_links:
            if link == event.id:
                continue
            if link in all_dropped_ids:
                new_links |= _resolve_to_kept(link, seen={link}, max_key=max_key)
            else:
                # If a causal link somehow points forward in time, drop it.
                if link in kept_by_id and _event_key(kept_by_id[link]) <= max_key:
                    new_links.add(link)

        # Final filter: remove self and any non-kept ids.
        new_links.discard(event.id)
        new_links = {link_id for link_id in new_links if link_id in kept_ids}

        if set(new_links) != set(event.causal_links):
            result.append(Event(
                id=event.id,
                sim_time=event.sim_time,
                tick_id=event.tick_id,
                order_in_tick=event.order_in_tick,
                type=event.type,
                source_agent=event.source_agent,
                target_agents=event.target_agents,
                location_id=event.location_id,
                causal_links=sorted(new_links),
                deltas=event.deltas,
                description=event.description,
                dialogue=event.dialogue,
                content_metadata=event.content_metadata,
                beat_type=event.beat_type,
                metrics=event.metrics,
            ))
        else:
            result.append(event)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bundle_events(
    events: list[Event],
    config: BundlerConfig = DEFAULT_BUNDLER_CONFIG,
) -> BundleResult:
    """Compress an event list for visualization and extraction.

    Pipeline: attach low-importance observes -> rewire.
    """
    if not events:
        logger.warning("Event bundler received an empty event list; returning empty output.")
        return BundleResult(events=[], stats=BundleStats(input_count=0, output_count=0), dropped_ids=set())

    # Ensure deterministic behavior regardless of caller ordering.
    events = sorted(events, key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
    stats = BundleStats(input_count=len(events))

    # Keep a parent lookup for causal rewiring.
    original_parents: dict[str, list[str]] = {e.id: list(e.causal_links) for e in events}

    # SOCIAL_MOVE events are intentionally preserved so segmentation can
    # identify location boundaries from explicit movement events.
    result = list(events)
    dropped_moves: set[str] = set()
    dropped_attached_moves: set[str] = set()
    absorbed_moves: dict[str, str] = {}
    absorbed_attached_moves: dict[str, str] = {}
    stats.moves_squashed = 0
    stats.moves_attached = 0

    # Step 1: attach low-importance observes.
    result, dropped_obs, absorbed_obs, attach_count = _attach_observes(result, config)
    stats.observes_attached = attach_count

    all_dropped = dropped_moves | dropped_attached_moves | dropped_obs
    absorbed_by: dict[str, str] = {}
    absorbed_by.update(absorbed_moves)
    absorbed_by.update(absorbed_attached_moves)
    absorbed_by.update(absorbed_obs)

    # Step 2: rewire causal links that still reference dropped events.
    result = _rewire_causal_links(
        events=result,
        all_dropped_ids=all_dropped,
        absorbed_by=absorbed_by,
        original_parents=original_parents,
    )

    stats.output_count = len(result)
    return BundleResult(events=result, stats=stats, dropped_ids=all_dropped)
