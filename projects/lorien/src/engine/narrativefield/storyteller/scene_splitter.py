from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from narrativefield.schema.events import BeatType, Event, EventType
from narrativefield.storyteller.types import SceneChunk


@dataclass(frozen=True)
class _Boundary:
    index: int  # split occurs BETWEEN index and index+1 (i.e., after events[index])
    reason: str  # "location" | "time_gap" | "beat" | "characters"
    priority: int  # lower is higher priority


def split_into_scenes(
    events: list[Event],
    target_chunk_size: int = 10,  # aim for 8-12 events per chunk
    min_chunk_size: int = 3,
) -> list[SceneChunk]:
    """Split arc events into narrative scene chunks.

    Splitting priority (checked in this order):
    1. Location change
    2. Time gap (>10 sim-minutes)
    3. Beat type transition (TURNING_POINT->CONSEQUENCE)
    4. Character set change (Jaccard < 0.5)
    5. Mathematical midpoint (only if a chunk exceeds 2x target and there were no natural boundaries)

    Notes:
    - Events are assumed to already be sorted; we do not re-sort.
    - The TURNING_POINT beat is never split *before* the TP event (it stays with rising action).
    """

    if not events:
        return []
    if len(events) == 1:
        return [_build_chunk(scene_index=0, events=events)]

    # Precompute natural boundaries between consecutive events.
    boundaries = _compute_boundaries(events)

    # Greedy assembly with size guards.
    chunks: list[list[Event]] = []
    start = 0
    scene_index = 0

    i = 0
    natural_boundary_seen_in_chunk = False
    while i < len(events) - 1:
        next_event = events[i + 1]

        # Never split *before* a turning point.
        if _is_turning_point(next_event):
            # Still mark that we *saw* a potential boundary if one exists here; it matters for midpoint logic.
            if boundaries.get(i) is not None:
                natural_boundary_seen_in_chunk = True
            i += 1
            continue

        b = boundaries.get(i)
        if b is not None:
            natural_boundary_seen_in_chunk = True

        current_len = i - start + 1
        remaining = len(events) - (i + 1)

        should_split = False
        if b is not None:
            should_split = _should_split_at_boundary(
                b,
                current_len=current_len,
                remaining=remaining,
                target_chunk_size=target_chunk_size,
                min_chunk_size=min_chunk_size,
            )

        # Midpoint fallback: only if we have no natural boundaries at all and we are ballooning.
        if (
            not should_split
            and not natural_boundary_seen_in_chunk
            and current_len >= (2 * target_chunk_size)
            and current_len >= (min_chunk_size * 2)
        ):
            split_after = _safe_midpoint_split_index(events, start, i)
            if split_after is not None:
                chunks.append(events[start : split_after + 1])
                start = split_after + 1
                scene_index += 1
                natural_boundary_seen_in_chunk = False
                i = start
                continue

        if should_split:
            chunks.append(events[start : i + 1])
            start = i + 1
            scene_index += 1
            natural_boundary_seen_in_chunk = False

        i += 1

    # Final chunk.
    if start < len(events):
        chunks.append(events[start:])

    return [_build_chunk(scene_index=idx, events=chunk) for idx, chunk in enumerate(chunks)]


def _is_turning_point(event: Event) -> bool:
    return event.beat_type == BeatType.TURNING_POINT


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _compute_boundaries(events: list[Event]) -> dict[int, _Boundary]:
    """Return candidate boundaries keyed by index i meaning boundary after events[i]."""

    out: dict[int, _Boundary] = {}
    for i in range(len(events) - 1):
        prev = events[i]
        nxt = events[i + 1]

        candidates: list[_Boundary] = []

        # 1) Location change
        if prev.location_id != nxt.location_id:
            candidates.append(_Boundary(index=i, reason="location", priority=1))

        # 2) Time gap
        if (nxt.sim_time - prev.sim_time) > 10.0:
            candidates.append(_Boundary(index=i, reason="time_gap", priority=2))

        # 3) Beat transition
        if prev.beat_type == BeatType.TURNING_POINT and nxt.beat_type == BeatType.CONSEQUENCE:
            candidates.append(_Boundary(index=i, reason="beat", priority=3))

        # 4) Character set change
        if _jaccard(_participants(prev), _participants(nxt)) < 0.5:
            candidates.append(_Boundary(index=i, reason="characters", priority=4))

        if not candidates:
            continue

        # Choose highest priority candidate.
        best = min(candidates, key=lambda b: b.priority)
        out[i] = best

    return out


def _should_split_at_boundary(
    boundary: _Boundary,
    *,
    current_len: int,
    remaining: int,
    target_chunk_size: int,
    min_chunk_size: int,
) -> bool:
    """Decide whether to actually split at a natural boundary.

    We avoid creating tiny chunks unless we are at the tail end of the arc.
    """

    # Never split if it would make the current chunk too small.
    if current_len < min_chunk_size:
        return False

    # Prefer to avoid leaving a tiny trailing chunk. Allow it only for higher-priority reasons.
    leaving_tiny_tail = remaining < min_chunk_size
    if leaving_tiny_tail and boundary.reason not in ("beat", "time_gap", "location"):
        return False

    # Character-change boundaries are "soft": only split if we're already at/over target size.
    if boundary.reason == "characters":
        return current_len >= target_chunk_size

    # Otherwise, split for hard boundaries.
    return True


def _safe_midpoint_split_index(events: list[Event], start: int, end_inclusive: int) -> int | None:
    """Find a safe split point near the mathematical midpoint.

    Returns an index `k` where we split after events[k]. Ensures we don't split immediately
    before a TURNING_POINT event.
    """

    length = end_inclusive - start + 1
    if length < 2:
        return None

    mid = start + (length // 2) - 1
    mid = max(start, min(mid, end_inclusive - 1))

    # Search outward for a safe spot.
    for offset in range(0, length):
        for cand in (mid - offset, mid + offset):
            if cand < start or cand >= end_inclusive:
                continue
            if _is_turning_point(events[cand + 1]):
                continue
            return cand
    return None


def _build_chunk(*, scene_index: int, events: list[Event]) -> SceneChunk:
    if not events:
        # Should not happen, but keep it safe.
        raise ValueError("SceneChunk cannot be built from an empty event list.")

    location = _mode_location(events)
    time_start = float(events[0].sim_time)
    time_end = float(events[-1].sim_time)
    characters_present = sorted({p for e in events for p in _participants(e)})
    scene_type = _classify_scene_type(events)
    is_pivotal = _is_pivotal_chunk(events)

    return SceneChunk(
        scene_index=int(scene_index),
        events=list(events),
        location=location,
        time_start=time_start,
        time_end=time_end,
        characters_present=characters_present,
        scene_type=scene_type,
        is_pivotal=is_pivotal,
        summary="",
    )


def _mode_location(events: list[Event]) -> str:
    counts = Counter(e.location_id for e in events)
    if not counts:
        return ""
    # Deterministic tie-break: pick the first location encountered among tied modes.
    max_count = max(counts.values())
    tied = {loc for loc, c in counts.items() if c == max_count}
    for e in events:
        if e.location_id in tied:
            return e.location_id
    return events[0].location_id


def _classify_scene_type(events: list[Event]) -> str:
    total = len(events)
    if total == 0:
        return "escalation"

    counts = Counter(e.type for e in events)

    def _ratio(types: set[EventType]) -> float:
        return sum(counts.get(t, 0) for t in types) / total

    if _ratio({EventType.CONFLICT, EventType.CATASTROPHE}) > 0.5:
        return "confrontation"
    if _ratio({EventType.REVEAL, EventType.CONFIDE}) > 0.5:
        return "revelation"
    if _ratio({EventType.CHAT, EventType.OBSERVE}) > 0.5:
        return "conversation"
    return "escalation"


def _is_pivotal_chunk(events: list[Event]) -> bool:
    for e in events:
        if e.beat_type == BeatType.TURNING_POINT:
            return True
        if e.type == EventType.CATASTROPHE:
            return True
        if e.type == EventType.REVEAL and e.metrics.irony_collapse is not None and e.metrics.irony_collapse.score > 0.5:
            return True
    return False
