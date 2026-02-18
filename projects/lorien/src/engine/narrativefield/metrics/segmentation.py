from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from narrativefield.schema.events import Event, EventType
from narrativefield.schema.scenes import Scene
from narrativefield.schema.world import Location


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class SegmentationConfig:
    # Decision 21: Jaccard threshold is 0.3.
    participant_jaccard_threshold: float = 0.3

    time_gap_minutes: float = 5.0

    # Tension gap: sustained valley below peak*drop_ratio.
    tension_window: int = 5
    drop_ratio: float = 0.3
    sustained_count: int = 3

    min_scene_size: int = 3


DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()


def locations_are_scene_compatible(loc_a: str, loc_b: str, locations: dict[str, Location]) -> bool:
    """Adjacency exception: allow grouping when overhearing is possible."""
    a = locations.get(loc_a)
    b = locations.get(loc_b)
    if not a or not b:
        return False
    return (loc_b in a.overhear_from) or (loc_a in b.overhear_from)


def location_break(prev: Event, curr: Event, locations: dict[str, Location]) -> bool:
    prev_loc = _scene_location_id(prev)
    curr_loc = _scene_location_id(curr)
    if prev_loc == curr_loc:
        return False
    if locations_are_scene_compatible(prev_loc, curr_loc, locations):
        return False
    return True


def _scene_location_id(event: Event) -> str:
    # SOCIAL_MOVE stores source in location_id and destination in content_metadata.
    if event.type == EventType.SOCIAL_MOVE:
        metadata = event.content_metadata or {}
        destination = metadata.get("destination")
        if isinstance(destination, str) and destination:
            return destination
    return event.location_id


def participant_break(prev: Event, curr: Event, *, threshold: float) -> bool:
    return _jaccard(_participants(prev), _participants(curr)) < threshold


def time_break(prev: Event, curr: Event, *, threshold_minutes: float) -> bool:
    return (curr.sim_time - prev.sim_time) > threshold_minutes


def tension_gap_break(
    events: list[Event], current_index: int, *, window: int, drop_ratio: float, sustained_count: int
) -> bool:
    if current_index < window:
        return False

    recent = events[max(0, current_index - window) : current_index]
    peak = max((e.metrics.tension for e in recent), default=0.0)
    if peak <= 1e-9:
        return False
    threshold = peak * drop_ratio

    low_streak = events[max(0, current_index - sustained_count) : current_index]
    if len(low_streak) < sustained_count:
        return False
    return all(e.metrics.tension < threshold for e in low_streak)


def irony_collapse_break(prev: Event) -> bool:
    collapse = prev.metrics.irony_collapse
    return bool(collapse and collapse.detected and float(collapse.drop) >= 0.5)


def _scene_type(events: list[Event], tension_peak: float) -> str:
    if any(e.type == EventType.CATASTROPHE for e in events):
        return "catastrophe"
    if any(e.type == EventType.CONFLICT for e in events):
        return "confrontation"
    if any(
        e.type == EventType.REVEAL and e.metrics.irony_collapse and e.metrics.irony_collapse.detected
        for e in events
    ):
        return "revelation"
    if any(e.type == EventType.CONFIDE for e in events):
        return "bonding"
    if tension_peak > 0.6:
        return "escalation"
    return "maintenance"


def _dominant_theme(events: list[Event]) -> str:
    totals: dict[str, float] = {}
    for e in events:
        for axis, delta in (e.metrics.thematic_shift or {}).items():
            totals[axis] = totals.get(axis, 0.0) + float(delta)
    if not totals:
        return ""
    return max(totals, key=lambda k: abs(totals[k]))


def _build_scene(events: list[Event], idx: int) -> Scene:
    location_counts: dict[str, int] = {}
    participants: set[str] = set()
    for e in events:
        participants |= _participants(e)
        scene_loc = _scene_location_id(e)
        location_counts[scene_loc] = location_counts.get(scene_loc, 0) + 1

    primary_location = max(location_counts, key=location_counts.get) if location_counts else ""
    tension_arc = [clamp(float(e.metrics.tension), 0.0, 1.0) for e in events]
    peak = max(tension_arc, default=0.0)
    avg = float(mean(tension_arc)) if tension_arc else 0.0

    return Scene(
        # Stable for a fixed input ordering/configuration; used as a deterministic
        # scene label rather than a globally unique identifier.
        id=f"scene_{idx:03d}",
        event_ids=[e.id for e in events],
        location=primary_location,
        participants=sorted(participants),
        time_start=float(events[0].sim_time),
        time_end=float(events[-1].sim_time),
        tick_start=int(events[0].tick_id),
        tick_end=int(events[-1].tick_id),
        tension_arc=tension_arc,
        tension_peak=float(peak),
        tension_mean=float(avg),
        dominant_theme=_dominant_theme(events),
        scene_type=_scene_type(events, peak),
        summary="",
    )


def _merge_tiny_scene_groups(groups: list[list[Event]], min_size: int) -> list[list[Event]]:
    if not groups:
        return groups

    def tension_mean(group: list[Event]) -> float:
        vals = [float(e.metrics.tension) for e in group]
        return float(mean(vals)) if vals else 0.0

    merged: list[list[Event]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        if len(g) >= min_size or len(groups) == 1:
            merged.append(g)
            i += 1
            continue

        # Choose neighbor with closest tension.
        prev_group = merged[-1] if merged else None
        next_group = groups[i + 1] if i + 1 < len(groups) else None

        if prev_group is None and next_group is None:
            merged.append(g)
            i += 1
            continue

        if prev_group is None:
            groups[i + 1] = g + next_group  # preserve order
            i += 1
            continue

        if next_group is None:
            prev_group.extend(g)
            i += 1
            continue

        t = tension_mean(g)
        if abs(tension_mean(prev_group) - t) <= abs(tension_mean(next_group) - t):
            prev_group.extend(g)
        else:
            groups[i + 1] = g + next_group
        i += 1

    # Repeat until stable (tiny scenes can chain-merge).
    if any(len(g) < min_size for g in merged) and len(merged) != len(groups):
        return _merge_tiny_scene_groups(merged, min_size)
    return merged


def segment_into_scenes(
    events: list[Event],
    *,
    locations: dict[str, Location],
    config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
) -> list[Scene]:
    """
    Segment an ordered event log into scenes.

    Authority: specs/metrics/scene-segmentation.md (Decision 21).
    """
    if not events:
        return []

    groups: list[list[Event]] = [[events[0]]]

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]

        # Events in the same tick are never separated.
        if curr.tick_id == prev.tick_id:
            groups[-1].append(curr)
            continue

        should_break = False

        # SOCIAL_MOVE events always force a boundary, but consecutive SOCIAL_MOVEs belong together.
        if curr.type == EventType.SOCIAL_MOVE and prev.type != EventType.SOCIAL_MOVE:
            should_break = True

        # Catastrophes are dramatic ruptures.
        if curr.type == EventType.CATASTROPHE and prev.type != EventType.CATASTROPHE:
            should_break = True

        if not should_break and location_break(prev, curr, locations):
            should_break = True
        elif not should_break and participant_break(prev, curr, threshold=config.participant_jaccard_threshold):
            should_break = True
        elif not should_break and time_break(prev, curr, threshold_minutes=config.time_gap_minutes):
            should_break = True
        elif not should_break and tension_gap_break(
            events,
            i,
            window=config.tension_window,
            drop_ratio=config.drop_ratio,
            sustained_count=config.sustained_count,
        ):
            should_break = True
        elif not should_break and irony_collapse_break(prev):
            should_break = True

        if should_break:
            groups.append([curr])
        else:
            groups[-1].append(curr)

    groups = _merge_tiny_scene_groups(groups, config.min_scene_size)
    scenes = [_build_scene(g, idx) for idx, g in enumerate(groups)]
    return scenes
