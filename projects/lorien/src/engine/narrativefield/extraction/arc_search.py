from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_validator import GrammarConfig, validate_arc
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.extraction.types import ArcCandidate, ArcScore, ArcValidation
from narrativefield.schema.events import BeatType, DeltaKind, Event, EventType

EVENT_TYPE_WEIGHTS: dict[EventType, float] = {
    EventType.CATASTROPHE: 5.0,
    EventType.CONFLICT: 4.0,
    EventType.LIE: 3.6,
    EventType.REVEAL: 3.4,
    EventType.CONFIDE: 3.0,
    EventType.PHYSICAL: 2.0,
    EventType.CHAT: 1.0,
    EventType.INTERNAL: 0.8,
    EventType.OBSERVE: 0.5,
    EventType.SOCIAL_MOVE: 0.3,
}

EVENT_TYPE_DRAMA_ORDER: dict[EventType, int] = {
    EventType.CATASTROPHE: 0,
    EventType.CONFLICT: 1,
    EventType.REVEAL: 2,
    EventType.CONFIDE: 3,
    EventType.LIE: 4,
    EventType.SOCIAL_MOVE: 5,
    EventType.OBSERVE: 6,
    EventType.CHAT: 7,
    EventType.INTERNAL: 8,
}

PHASE_ORDER: dict[BeatType, int] = {
    BeatType.SETUP: 0,
    BeatType.COMPLICATION: 1,
    BeatType.ESCALATION: 1,
    BeatType.TURNING_POINT: 2,
    BeatType.CONSEQUENCE: 3,
}

# BeatTypes allowed at each phase level, used during monotonic enforcement.
_PHASE_TO_BEATS: dict[int, list[BeatType]] = {
    0: [BeatType.SETUP],
    1: [BeatType.COMPLICATION, BeatType.ESCALATION],
    2: [BeatType.TURNING_POINT],
    3: [BeatType.CONSEQUENCE],
}


@dataclass
class ArcSearchDiagnostics:
    violations: list[str] = field(default_factory=list)
    suggested_protagonist: str = ""
    suggested_time_window: tuple[float, float] | None = None
    suggested_keep_ids: list[str] = field(default_factory=list)
    suggested_drop_ids: list[str] = field(default_factory=list)
    primary_failure: str = ""
    rule_failure_counts: dict[str, int] = field(default_factory=dict)
    best_candidate_violation_count: int = 0
    # `search_arc` can early-return a valid candidate before diagnostics are built.
    # In that common path this remains 0; it is primarily meaningful for fallback analysis.
    candidates_evaluated: int = 0
    best_candidate_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "violations": list(self.violations),
            "suggested_protagonist": self.suggested_protagonist,
            "suggested_time_window": (
                list(self.suggested_time_window) if self.suggested_time_window else None
            ),
            "suggested_keep_ids": list(self.suggested_keep_ids),
            "suggested_drop_ids": list(self.suggested_drop_ids),
            "primary_failure": self.primary_failure,
            "rule_failure_counts": dict(self.rule_failure_counts),
            "best_candidate_violation_count": int(self.best_candidate_violation_count),
            "candidates_evaluated": int(self.candidates_evaluated),
            "best_candidate_violations": list(self.best_candidate_violations),
        }


@dataclass
class ArcSearchResult:
    events: list[Event]
    beats: list[BeatType]
    protagonist: str
    validation: ArcValidation
    arc_score: ArcScore | None = None
    diagnostics: ArcSearchDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "beats": [b.value for b in self.beats],
            "protagonist": self.protagonist,
            "validation": self.validation.to_dict(),
            "arc_score": self.arc_score.to_dict() if self.arc_score else None,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }


def _delta_signal(event: Event) -> float:
    """
    Fallback importance signal derived from non-pacing deltas.

    Notes:
    - Many belief/secret deltas are categorical; treat their presence as signal.
    - For numeric deltas, add absolute magnitude.
    """
    score = 0.0
    for d in event.deltas:
        if d.kind == DeltaKind.PACING:
            continue
        if d.kind in {DeltaKind.BELIEF, DeltaKind.RELATIONSHIP, DeltaKind.SECRET_STATE}:
            score += 1.0
        v = d.value
        if isinstance(v, bool):
            score += 0.25 if v else 0.0
        elif isinstance(v, (int, float)):
            score += abs(float(v))
    return score


def _event_importance(event: Event) -> float:
    """
    Compute per-event importance.

    Prefer `event.metrics.significance` when it's non-zero; otherwise fall back to
    a type-weighted + delta-signal heuristic.
    """
    sig = float(getattr(event.metrics, "significance", 0.0) or 0.0)
    if sig > 0:
        return sig
    weight = EVENT_TYPE_WEIGHTS.get(event.type, 0.5)
    return weight + 0.5 * _delta_signal(event)


def fallback_event_sort_key(
    event: Event,
    protagonist: str | None = None,
) -> tuple[float, int, int, float, int, int, str]:
    """Deterministic fallback ranking key for significance tie-breaks."""
    significance = -float(getattr(event.metrics, "significance", 0.0) or 0.0)
    protagonist_rank = 0 if protagonist and _involves(event, protagonist) else 1
    drama_rank = EVENT_TYPE_DRAMA_ORDER.get(event.type, 9)
    return (
        significance,
        protagonist_rank,
        drama_rank,
        float(event.sim_time),
        int(event.tick_id),
        int(event.order_in_tick),
        str(event.id),
    )


def _normalize_violation(violation: str) -> str:
    """Normalize validator messages into stable rule IDs for aggregation."""
    normalized = violation.lower()

    if "consequence" in normalized and "missing" in normalized:
        return "missing_consequence"
    if "setup" in normalized and "missing" in normalized:
        return "missing_setup"
    if "complication" in normalized and "missing" in normalized:
        return "missing_development"
    if "turning_point" in normalized:
        if "found 0" in normalized or "missing" in normalized:
            return "missing_turning_point"
        return "duplicate_turning_point"
    if "protagonist" in normalized:
        return "no_protagonist"
    if "order violation" in normalized:
        return "phase_order_violation"
    if "too few" in normalized:
        return "too_few_beats"
    if "too many" in normalized:
        return "too_many_beats"
    if "causal gap" in normalized:
        return "causal_gap"
    if "too short" in normalized:
        return "arc_too_short"
    return "unknown"


def _score_agents(events: list[Event]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for e in events:
        s = _event_importance(e)
        scores[e.source_agent] += s
        for t in e.target_agents:
            scores[t] += s
    return dict(scores)


def _has_meaningful_deltas(event: Event) -> bool:
    return any(d.kind in {DeltaKind.BELIEF, DeltaKind.RELATIONSHIP} for d in event.deltas)


def _participants(event: Event) -> set[str]:
    return {event.source_agent, *event.target_agents}


def _involves(event: Event, agent_id: str) -> bool:
    return event.source_agent == agent_id or agent_id in event.target_agents


def _build_reverse_links(events: list[Event]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = defaultdict(list)
    for e in events:
        for parent in e.causal_links:
            reverse[parent].append(e.id)
    # Deterministic traversal order.
    for k in list(reverse.keys()):
        reverse[k].sort()
    return dict(reverse)


def _causal_neighborhood(
    *,
    anchor_id: str,
    by_id: dict[str, Event],
    reverse_links: dict[str, list[str]],
    allowed_ids: set[str],
    max_depth: int = 3,
) -> set[str]:
    """BFS up to `max_depth` both backward and forward in the causal graph."""
    visited: set[str] = {anchor_id}
    frontier: list[tuple[str, int]] = [(anchor_id, 0)]
    out: set[str] = set()

    while frontier:
        cur_id, depth = frontier.pop(0)
        if depth >= max_depth:
            continue

        cur = by_id.get(cur_id)
        if not cur:
            continue

        # Backward edges: parents.
        for parent_id in sorted(cur.causal_links):
            if parent_id in visited or parent_id not in allowed_ids:
                continue
            visited.add(parent_id)
            out.add(parent_id)
            frontier.append((parent_id, depth + 1))

        # Forward edges: children.
        for child_id in reverse_links.get(cur_id, []):
            if child_id in visited or child_id not in allowed_ids:
                continue
            visited.add(child_id)
            out.add(child_id)
            frontier.append((child_id, depth + 1))

    return out


@dataclass(frozen=True)
class _Candidate:
    events: list[Event]
    beats: list[BeatType]
    validation: ArcValidation
    protagonist: str
    arc_score: ArcScore | None
    importance_sum: float
    anchor_id: str


def _select_beat_for_phase(
    *,
    phase_level: int,
    index: int,
    current_beats: list[BeatType],
    events: list[Event],
) -> BeatType:
    """
    Select the canonical beat for a phase level during monotonic repair.

    Selection criteria:
    - phase 0 -> SETUP
    - phase 1 -> COMPLICATION/ESCALATION based on local tension direction
    - phase 2 -> TURNING_POINT
    - phase 3 -> CONSEQUENCE
    """
    candidates = _PHASE_TO_BEATS[phase_level]
    if phase_level == 1 and len(candidates) > 1:
        prev_tension = events[index - 1].metrics.tension if index > 0 else 0.0
        cur_tension = events[index].metrics.tension
        return BeatType.ESCALATION if cur_tension > prev_tension else BeatType.COMPLICATION
    return candidates[0]


def _has_post_peak_consequence(events: list[Event], beats: list[BeatType]) -> bool:
    if not events or len(events) != len(beats):
        return False
    peak_idx = max(range(len(events)), key=lambda j: events[j].metrics.tension)
    return any(b == BeatType.CONSEQUENCE for b in beats[peak_idx + 1 :])


def _extend_candidate_with_post_peak_consequence(
    *,
    candidate: _Candidate,
    window: list[Event],
    protagonist: str,
    max_events: int,
    total_sim_time: float | None,
    grammar_config: GrammarConfig | None,
) -> _Candidate | None:
    """
    When no valid candidate contains post-peak consequences, widen with
    post-peak events and accept slightly lower-scoring but complete arcs.
    """
    selected_ids = {e.id for e in candidate.events}
    peak_event = max(candidate.events, key=lambda e: e.metrics.tension, default=None)
    if peak_event is None:
        return None

    post_peak_pool = [
        e for e in window if e.id not in selected_ids and e.sim_time > peak_event.sim_time
    ]
    if not post_peak_pool:
        return None

    post_peak_pool.sort(
        key=lambda e: (
            0 if _involves(e, protagonist) else 1,
            -_event_importance(e),
            e.sim_time,
            e.id,
        )
    )

    trial = list(candidate.events)
    for extra in post_peak_pool:
        if len(trial) >= max_events:
            break
        trial.append(extra)
        trial.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))

        beats = classify_beats(trial)
        if grammar_config is None:
            beats = _enforce_monotonic_beats(trial, beats)
        else:
            beats = _enforce_monotonic_beats(trial, beats, grammar_config=grammar_config)
        if grammar_config is None:
            validation = validate_arc(events=trial, beats=beats, total_sim_time=total_sim_time)
        else:
            validation = validate_arc(
                events=trial,
                beats=beats,
                total_sim_time=total_sim_time,
                grammar_config=grammar_config,
            )
        if not validation.valid:
            continue
        if not _has_post_peak_consequence(trial, beats):
            continue

        arc_score = score_arc(trial, beats)
        return _Candidate(
            events=trial,
            beats=beats,
            validation=validation,
            protagonist=protagonist,
            arc_score=arc_score,
            importance_sum=sum(_event_importance(e) for e in trial),
            anchor_id=candidate.anchor_id,
        )

    return None


def search_arc(
    all_events: list[Event],
    time_start: float | None = None,
    time_end: float | None = None,
    agent_ids: list[str] | None = None,
    protagonist: str | None = None,
    max_events: int = 20,
    total_sim_time: float | None = None,
    grammar_config: GrammarConfig | None = None,
) -> ArcSearchResult:
    """Search for a coherent protagonist-centric arc within a time region."""

    # Step 1: Filter to time window.
    window = all_events
    if time_start is not None:
        window = [e for e in window if e.sim_time >= time_start]
    if time_end is not None:
        window = [e for e in window if e.sim_time <= time_end]
    if agent_ids:
        agent_set = set(agent_ids)
        window = [
            e for e in window
            if e.source_agent in agent_set or bool(set(e.target_agents) & agent_set)
        ]
    window.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))

    if not window:
        return ArcSearchResult(
            events=[],
            beats=[],
            protagonist=protagonist or "",
            validation=ArcValidation(valid=False, violations=("No events in time window",)),
            diagnostics=ArcSearchDiagnostics(
                violations=["No events in time window"],
                primary_failure="No events in time window",
            ),
        )

    # Step 2: Score agents to find protagonist.
    agent_scores = _score_agents(window)
    if not protagonist:
        protagonist = max(agent_scores, key=agent_scores.get)  # type: ignore[arg-type]

    # Index all events for causal expansion.
    by_id: dict[str, Event] = {e.id: e for e in all_events}
    reverse_links = _build_reverse_links(all_events)
    window_ids = {e.id for e in window}

    # Select anchor events: top-K important events in the region.
    anchor_scored: list[tuple[float, Event]] = []
    for e in window:
        if e.type == EventType.INTERNAL:
            continue
        # Exclude SOCIAL_MOVE/OBSERVE unless they contain belief/relationship deltas.
        if e.type in {EventType.SOCIAL_MOVE, EventType.OBSERVE} and not _has_meaningful_deltas(e):
            continue
        anchor_scored.append((_event_importance(e), e))
    anchor_scored.sort(key=lambda se: (-se[0], se[1].sim_time, se[1].id))

    anchors = [e for _, e in anchor_scored[: max(1, min(8, len(anchor_scored)))]]
    # Prefer anchors involving the protagonist; fall back to any anchor.
    proto_anchors = [e for e in anchors if _involves(e, protagonist)]
    if proto_anchors:
        anchors = proto_anchors

    proto_events_sorted = sorted(
        [e for e in window if _involves(e, protagonist) and e.type != EventType.INTERNAL],
        key=lambda e: (-_event_importance(e), e.sim_time, e.id),
    )
    proto_keep_ids = {e.id for e in proto_events_sorted[: max_events * 2]}
    # Ensure we include some early/late context so the arc can span >=10 sim minutes.
    proto_by_time = sorted(
        [e for e in window if _involves(e, protagonist) and e.type != EventType.INTERNAL],
        key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick),
    )
    for e in proto_by_time[:3]:
        proto_keep_ids.add(e.id)
    for e in proto_by_time[-3:]:
        proto_keep_ids.add(e.id)

    candidates: list[_Candidate] = []
    for anchor in anchors:
        neighborhood = _causal_neighborhood(
            anchor_id=anchor.id,
            by_id=by_id,
            reverse_links=reverse_links,
            allowed_ids=window_ids,
            max_depth=3,
        )

        pool_ids = {anchor.id} | neighborhood | proto_keep_ids
        pool = [by_id[i] for i in pool_ids if i in by_id]
        pool.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))

        selected = _downsample_preserving_continuity(
            pool=pool,
            bridge_pool=window,
            protagonist=protagonist,
            anchor_id=anchor.id,
            max_events=max_events,
        )

        beats = classify_beats(selected)
        if grammar_config is None:
            beats = _enforce_monotonic_beats(selected, beats)
        else:
            beats = _enforce_monotonic_beats(selected, beats, grammar_config=grammar_config)
        if grammar_config is None:
            validation = validate_arc(events=selected, beats=beats, total_sim_time=total_sim_time)
        else:
            validation = validate_arc(
                events=selected,
                beats=beats,
                total_sim_time=total_sim_time,
                grammar_config=grammar_config,
            )

        arc_score: ArcScore | None = score_arc(selected, beats) if validation.valid else None
        candidates.append(
            _Candidate(
                events=selected,
                beats=beats,
                validation=validation,
                protagonist=protagonist,
                arc_score=arc_score,
                importance_sum=sum(_event_importance(e) for e in selected),
                anchor_id=anchor.id,
            )
        )

    # Pick best: valid arcs first by composite score, then importance; otherwise fewest violations.
    valid = [c for c in candidates if c.validation.valid]
    if valid:
        valid_with_aftermath = [c for c in valid if _has_post_peak_consequence(c.events, c.beats)]
        if valid_with_aftermath:
            best = max(
                valid_with_aftermath,
                key=lambda c: (float(c.arc_score.composite) if c.arc_score else 0.0, c.importance_sum),
            )
        else:
            best = max(
                valid,
                key=lambda c: (float(c.arc_score.composite) if c.arc_score else 0.0, c.importance_sum),
            )
            widened = _extend_candidate_with_post_peak_consequence(
                candidate=best,
                window=window,
                protagonist=protagonist,
                max_events=max_events,
                total_sim_time=total_sim_time,
                grammar_config=grammar_config,
            )
            if widened is not None:
                best = widened
        return ArcSearchResult(
            events=best.events,
            beats=best.beats,
            protagonist=best.protagonist,
            validation=best.validation,
            arc_score=best.arc_score,
            diagnostics=None,
        )

    # If none are valid, choose the "least bad" and return diagnostics.
    # Prefer candidates that still include post-peak consequences so fallback arcs
    # preserve causal aftermath when possible.
    aftermath_candidates = [c for c in candidates if _has_post_peak_consequence(c.events, c.beats)]
    candidate_pool = aftermath_candidates if aftermath_candidates else candidates
    best = min(candidate_pool, key=lambda c: (len(c.validation.violations), -c.importance_sum))
    rule_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        for violation in candidate.validation.violations:
            rule_counts[_normalize_violation(violation)] += 1
    keep_ids = [e.id for e in best.events]
    drop_ids = [e.id for e in window if e.id not in set(keep_ids)]
    diagnostics = ArcSearchDiagnostics(
        violations=list(best.validation.violations),
        suggested_protagonist=protagonist,
        suggested_time_window=(best.events[0].sim_time, best.events[-1].sim_time) if best.events else None,
        suggested_keep_ids=keep_ids,
        suggested_drop_ids=drop_ids[:200],
        primary_failure=(best.validation.violations[0] if best.validation.violations else ""),
        rule_failure_counts=dict(rule_counts),
        best_candidate_violation_count=len(best.validation.violations),
        candidates_evaluated=len(candidates),
        best_candidate_violations=list(best.validation.violations),
    )
    return ArcSearchResult(
        events=best.events,
        beats=best.beats,
        protagonist=best.protagonist,
        validation=best.validation,
        arc_score=None,
        diagnostics=diagnostics,
    )


def _repair_causal_continuity(
    selected: list[Event],
    pool: list[Event],
    max_events: int,
) -> list[Event]:
    """Insert bridge events from pool when consecutive selected events lack connectivity."""
    if len(selected) < 2:
        return selected

    selected_ids = {e.id for e in selected}
    pool_by_id = {e.id: e for e in pool if e.id not in selected_ids}

    result = list(selected)
    i = 0
    while i < len(result) - 1:
        curr = result[i]
        nxt = result[i + 1]
        has_link = any(link == curr.id for link in nxt.causal_links)
        shares = bool(_participants(curr) & _participants(nxt))
        if not has_link and not shares:
            # Try to find a bridge event between them in pool.
            bridge = _find_bridge(curr, nxt, pool_by_id)
            if bridge and len(result) < max_events:
                result.insert(i + 1, bridge)
                selected_ids.add(bridge.id)
                del pool_by_id[bridge.id]
                # Re-check from the same position.
                continue
        i += 1

    return result


def _find_bridge(
    before: Event,
    after: Event,
    pool: dict[str, Event],
) -> Event | None:
    """Find a pool event that connects before and after via causal link or participant overlap."""
    before_parts = _participants(before)
    after_parts = _participants(after)
    candidates: list[tuple[float, Event]] = []

    for e in pool.values():
        if e.sim_time < before.sim_time or e.sim_time > after.sim_time:
            continue
        e_parts = _participants(e)
        connects_before = (
            any(link == before.id for link in e.causal_links)
            or bool(e_parts & before_parts)
        )
        connects_after = (
            any(link == e.id for link in after.causal_links)
            or bool(e_parts & after_parts)
        )
        if connects_before and connects_after:
            candidates.append((_event_importance(e), e))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _enforce_monotonic_beats(
    events: list[Event],
    beats: list[BeatType],
    grammar_config: GrammarConfig | None = None,
) -> list[BeatType]:
    """Left-to-right monotonic phase enforcement with single TURNING_POINT guarantee."""
    if len(beats) < 2:
        return beats

    if grammar_config is not None:
        return _enforce_monotonic_beats_configured(events, beats, grammar_config)

    result = list(beats)

    # First pass: enforce monotonic phase ordering.
    max_phase = PHASE_ORDER[result[0]]
    for i in range(1, len(result)):
        cur_phase = PHASE_ORDER[result[i]]
        if cur_phase < max_phase:
            # Promote to minimum valid phase.
            for phase_level in sorted(_PHASE_TO_BEATS.keys()):
                if phase_level >= max_phase:
                    result[i] = _select_beat_for_phase(
                        phase_level=phase_level,
                        index=i,
                        current_beats=result,
                        events=events,
                    )
                    break
        max_phase = max(max_phase, PHASE_ORDER[result[i]])

    # Ensure exactly 1 TURNING_POINT.
    tp_indices = [i for i, b in enumerate(result) if b == BeatType.TURNING_POINT]
    if len(tp_indices) == 0:
        # Pick highest-tension event in middle 50%.
        n = len(events)
        mid_start = n // 4
        mid_end = 3 * n // 4
        if mid_start < mid_end:
            best_idx = max(
                range(mid_start, mid_end),
                key=lambda j: events[j].metrics.tension,
            )
            result[best_idx] = BeatType.TURNING_POINT
            # Ensure everything after is CONSEQUENCE.
            for j in range(best_idx + 1, len(result)):
                if PHASE_ORDER[result[j]] < PHASE_ORDER[BeatType.CONSEQUENCE]:
                    result[j] = _select_beat_for_phase(
                        phase_level=PHASE_ORDER[BeatType.CONSEQUENCE],
                        index=j,
                        current_beats=result,
                        events=events,
                    )
    elif len(tp_indices) > 1:
        # Keep the one with highest tension and demote duplicates by position.
        best = max(tp_indices, key=lambda j: events[j].metrics.tension)
        for idx in tp_indices:
            if idx != best:
                if idx < best:
                    result[idx] = BeatType.ESCALATION
                else:
                    result[idx] = BeatType.CONSEQUENCE

        # Ensure everything after the surviving turning point is phase 3.
        for j in range(best + 1, len(result)):
            if PHASE_ORDER[result[j]] < PHASE_ORDER[BeatType.CONSEQUENCE]:
                result[j] = _select_beat_for_phase(
                    phase_level=PHASE_ORDER[BeatType.CONSEQUENCE],
                    index=j,
                    current_beats=result,
                    events=events,
                )

    return result


def _enforce_monotonic_beats_configured(
    events: list[Event],
    beats: list[BeatType],
    grammar_config: GrammarConfig,
) -> list[BeatType]:
    """Configurable beat enforcement used during grammar-path sweeps."""
    if len(beats) < 2:
        return beats

    result = list(beats)
    max_phase_regressions = max(0, int(grammar_config.max_phase_regressions))
    regression_count = 0

    # First pass: allow up to N regressions, repair any beyond that budget.
    prev_phase = PHASE_ORDER[result[0]]
    for i in range(1, len(result)):
        cur_phase = PHASE_ORDER[result[i]]
        if cur_phase < prev_phase:
            regression_count += 1
            if regression_count > max_phase_regressions:
                result[i] = _select_beat_for_phase(
                    phase_level=prev_phase,
                    index=i,
                    current_beats=result,
                    events=events,
                )
                cur_phase = PHASE_ORDER[result[i]]
        prev_phase = cur_phase

    # Turning-point budget.
    max_turning_points = max(1, int(grammar_config.max_turning_points))
    tp_indices = [i for i, beat in enumerate(result) if beat == BeatType.TURNING_POINT]
    if len(tp_indices) == 0:
        n = len(events)
        mid_start = n // 4
        mid_end = 3 * n // 4
        if mid_start < mid_end:
            best_idx = max(range(mid_start, mid_end), key=lambda j: events[j].metrics.tension)
            result[best_idx] = BeatType.TURNING_POINT
            tp_indices = [best_idx]
    elif len(tp_indices) > max_turning_points:
        keep = sorted(tp_indices, key=lambda j: events[j].metrics.tension, reverse=True)[:max_turning_points]
        keep_set = set(keep)
        keep_sorted = sorted(keep)
        first_keep = keep_sorted[0]
        last_keep = keep_sorted[-1]

        for idx in tp_indices:
            if idx in keep_set:
                continue
            if idx < first_keep:
                result[idx] = BeatType.ESCALATION
            elif idx > last_keep:
                result[idx] = BeatType.CONSEQUENCE
            else:
                result[idx] = BeatType.ESCALATION
        tp_indices = keep_sorted

    # Strict-style aftermath sweep only when TP budget is singular.
    if tp_indices and max_turning_points <= 1:
        last_tp = max(tp_indices)
        for j in range(last_tp + 1, len(result)):
            if PHASE_ORDER[result[j]] < PHASE_ORDER[BeatType.CONSEQUENCE]:
                result[j] = _select_beat_for_phase(
                    phase_level=PHASE_ORDER[BeatType.CONSEQUENCE],
                    index=j,
                    current_beats=result,
                    events=events,
                )

    return result


def _downsample_preserving_continuity(
    *,
    pool: list[Event],
    bridge_pool: list[Event],
    protagonist: str,
    anchor_id: str,
    max_events: int,
) -> list[Event]:
    """
    Downsample to <= max_events with deterministic selection:
    - keep endpoints + anchor
    - prefer protagonist-involving events by importance
    - repair causal continuity with bridge events from `bridge_pool`
    """
    if not pool:
        return []
    if len(pool) <= max_events:
        return pool

    must_keep: set[str] = {pool[0].id, pool[-1].id, anchor_id}
    selected_ids: set[str] = set(must_keep)

    remaining = [e for e in pool if e.id not in selected_ids]
    remaining.sort(
        key=lambda e: (
            0 if _involves(e, protagonist) else 1,
            -_event_importance(e),
            e.sim_time,
            e.id,
        )
    )

    for e in remaining:
        if len(selected_ids) >= max_events:
            break
        selected_ids.add(e.id)

    selected = [e for e in pool if e.id in selected_ids]
    selected.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))

    # Ensure post-peak coverage for CONSEQUENCE beat candidates.
    if len(selected) >= 4:
        peak_idx = max(range(len(selected)), key=lambda idx: selected[idx].metrics.tension)
        has_post_peak = any(
            idx > peak_idx for idx in range(len(selected))
            if selected[idx].id not in must_keep
        )

        if not has_post_peak and peak_idx < (len(selected) - 1):
            peak_time = float(selected[peak_idx].sim_time)
            selected_id_set = {e.id for e in selected}
            post_peak_pool = [
                e for e in pool
                if e.id not in selected_id_set
                and float(e.sim_time) > peak_time
                and _involves(e, protagonist)
            ]
            if not post_peak_pool:
                post_peak_pool = [
                    e for e in pool
                    if e.id not in selected_id_set
                    and float(e.sim_time) > peak_time
                ]

            if post_peak_pool:
                best_post = max(post_peak_pool, key=_event_importance)
                swappable = [
                    (idx, e)
                    for idx, e in enumerate(selected)
                    if e.id not in must_keep and float(e.sim_time) < peak_time
                ]
                if not swappable:
                    swappable = [
                        (idx, e)
                        for idx, e in enumerate(selected)
                        if e.id not in must_keep and float(e.sim_time) <= peak_time
                    ]

                if swappable:
                    worst_idx, _worst_event = min(swappable, key=lambda item: _event_importance(item[1]))
                    selected[worst_idx] = best_post
                    selected.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
                    selected_ids = {e.id for e in selected}

    # Ensure minimum span (>=10 sim minutes) when possible by widening to edges.
    if selected and (selected[-1].sim_time - selected[0].sim_time) < 10.0:
        for e in pool:
            if len(selected) >= max_events:
                break
            if e.sim_time < selected[0].sim_time and _involves(e, protagonist):
                selected.insert(0, e)
                selected_ids.add(e.id)
        for e in reversed(pool):
            if len(selected) >= max_events:
                break
            if e.sim_time > selected[-1].sim_time and _involves(e, protagonist):
                selected.append(e)
                selected_ids.add(e.id)

    selected = _repair_causal_continuity(selected, bridge_pool, max_events)
    selected.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
    return selected


def search_region_for_arcs(
    events: list[Event],
    time_start: float,
    time_end: float,
    agent_filter: str | None = None,
    max_candidates: int = 5,
    total_sim_time: float | None = None,
    grammar_config: GrammarConfig | None = None,
) -> list[ArcCandidate]:
    """Search a time region for valid protagonist-centric arcs.

    Instead of validating the whole region as an arc, this treats the region
    as a *search space* and finds the best arc subsets within it.
    """
    # 1. Filter to region.
    region_events = [
        e for e in events
        if time_start <= e.sim_time <= time_end
    ]
    region_events.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))

    if not region_events:
        return []

    # 2. Identify protagonist candidates.
    if agent_filter:
        protagonist_candidates = [agent_filter]
    else:
        agent_scores = _score_agents(region_events)
        ranked = sorted(agent_scores.items(), key=lambda kv: -kv[1])
        protagonist_candidates = [a for a, _ in ranked[:5]]

    if not protagonist_candidates:
        return []

    # 3. Run arc search for each protagonist candidate.
    results: list[ArcCandidate] = []
    for protagonist in protagonist_candidates:
        if grammar_config is None:
            search_result = search_arc(
                all_events=events,
                time_start=time_start,
                time_end=time_end,
                protagonist=protagonist,
                max_events=20,
                total_sim_time=total_sim_time,
            )
        else:
            search_result = search_arc(
                all_events=events,
                time_start=time_start,
                time_end=time_end,
                protagonist=protagonist,
                max_events=20,
                total_sim_time=total_sim_time,
                grammar_config=grammar_config,
            )

        if not search_result.events:
            continue

        # Score even invalid arcs so we can rank them.
        arc_score = search_result.arc_score or score_arc(
            search_result.events, search_result.beats,
        )

        if not search_result.validation.valid:
            continue

        # Build explanation.
        peak_event = max(search_result.events, key=lambda e: e.metrics.tension)
        beat_counts: dict[str, int] = defaultdict(int)
        for b in search_result.beats:
            beat_counts[b.value] += 1
        beat_summary = ", ".join(
            f"{bt}({ct})" for bt, ct in sorted(beat_counts.items())
        )
        explanation = (
            f"{protagonist}'s arc: a {arc_score.composite:.2f}-quality "
            f"{len(search_result.events)}-beat arc centered on "
            f"\"{peak_event.description[:60]}\". "
            f"Beat structure: {beat_summary}."
        )

        results.append(
            ArcCandidate(
                protagonist=protagonist,
                events=search_result.events,
                beats=search_result.beats,
                validation=search_result.validation,
                score=arc_score,
                explanation=explanation,
            )
        )

    # 4. Sort by composite score descending, return top candidates.
    results.sort(key=lambda c: -c.score.composite)
    return results[:max_candidates]
