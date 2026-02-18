from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from narrativefield.extraction.arc_scorer import score_arc
from narrativefield.extraction.arc_search import ArcSearchDiagnostics, ArcSearchResult, search_arc
from narrativefield.extraction.arc_validator import GrammarConfig
from narrativefield.extraction.types import ArcScore, ArcValidation
from narrativefield.schema.events import BeatType, Event


def _pair_key(agent_a: str, agent_b: str) -> str:
    left, right = sorted((str(agent_a), str(agent_b)))
    return f"{left}-{right}"


def _arc_score_from_dict(raw: dict[str, Any] | None) -> ArcScore | None:
    if not isinstance(raw, dict):
        return None
    return ArcScore(
        composite=float(raw.get("composite", 0.0)),
        tension_variance=float(raw.get("tension_variance", 0.0)),
        peak_tension=float(raw.get("peak_tension", 0.0)),
        tension_shape=float(raw.get("tension_shape", 0.0)),
        significance=float(raw.get("significance", 0.0)),
        thematic_coherence=float(raw.get("thematic_coherence", 0.0)),
        irony_arc=float(raw.get("irony_arc", 0.0)),
        protagonist_dominance=float(raw.get("protagonist_dominance", 0.0)),
    )


def _validation_from_dict(raw: dict[str, Any] | None) -> ArcValidation:
    payload = raw or {}
    return ArcValidation(
        valid=bool(payload.get("valid", False)),
        violations=tuple(str(v) for v in (payload.get("violations") or [])),
    )


def _diagnostics_from_dict(raw: dict[str, Any] | None) -> ArcSearchDiagnostics | None:
    if not isinstance(raw, dict):
        return None
    time_window = raw.get("suggested_time_window")
    parsed_time_window: tuple[float, float] | None = None
    if isinstance(time_window, list) and len(time_window) == 2:
        parsed_time_window = (float(time_window[0]), float(time_window[1]))
    return ArcSearchDiagnostics(
        violations=[str(v) for v in (raw.get("violations") or [])],
        suggested_protagonist=str(raw.get("suggested_protagonist") or ""),
        suggested_time_window=parsed_time_window,
        suggested_keep_ids=[str(v) for v in (raw.get("suggested_keep_ids") or [])],
        suggested_drop_ids=[str(v) for v in (raw.get("suggested_drop_ids") or [])],
        primary_failure=str(raw.get("primary_failure") or ""),
        rule_failure_counts={str(k): int(v) for k, v in (raw.get("rule_failure_counts") or {}).items()},
        best_candidate_violation_count=int(raw.get("best_candidate_violation_count", 0) or 0),
        candidates_evaluated=int(raw.get("candidates_evaluated", 0) or 0),
        best_candidate_violations=[str(v) for v in (raw.get("best_candidate_violations") or [])],
    )


def _search_result_from_dict(raw: dict[str, Any] | None) -> ArcSearchResult:
    payload = raw or {}
    return ArcSearchResult(
        events=[Event.from_dict(e) for e in (payload.get("events") or [])],
        beats=[BeatType(b) for b in (payload.get("beats") or [])],
        protagonist=str(payload.get("protagonist") or ""),
        validation=_validation_from_dict(payload.get("validation")),
        arc_score=_arc_score_from_dict(payload.get("arc_score")),
        diagnostics=_diagnostics_from_dict(payload.get("diagnostics")),
    )


@dataclass
class RashomonArc:
    """One protagonist's arc from a shared event log."""

    protagonist: str
    search_result: ArcSearchResult
    arc_score: ArcScore | None
    events: list[Event]
    beats: list[BeatType]
    valid: bool
    violation_count: int
    violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protagonist": self.protagonist,
            "search_result": self.search_result.to_dict(),
            "arc_score": self.arc_score.to_dict() if self.arc_score else None,
            "events": [event.to_dict() for event in self.events],
            "beats": [beat.value for beat in self.beats],
            "valid": bool(self.valid),
            "violation_count": int(self.violation_count),
            "violations": list(self.violations),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RashomonArc:
        search_result = _search_result_from_dict(raw.get("search_result"))
        events = [Event.from_dict(e) for e in (raw.get("events") or [])]
        beats = [BeatType(b) for b in (raw.get("beats") or [])]
        violations = [str(v) for v in (raw.get("violations") or [])]
        valid = bool(raw.get("valid", False))
        return cls(
            protagonist=str(raw.get("protagonist") or ""),
            search_result=search_result,
            arc_score=_arc_score_from_dict(raw.get("arc_score")),
            events=events,
            beats=beats,
            valid=valid,
            violation_count=int(raw.get("violation_count", len(violations))),
            violations=violations,
        )


@dataclass
class RashomonSet:
    """All protagonist arcs from a single simulation."""

    seed: int
    total_events: int
    arcs: list[RashomonArc]

    @property
    def valid_count(self) -> int:
        return sum(1 for arc in self.arcs if arc.valid)

    @property
    def overlap_matrix(self) -> dict[str, float]:
        """Jaccard overlap across valid arcs using canonical pair keys: A-B where A<B."""
        matrix: dict[str, float] = {}
        valid_arcs = [arc for arc in self.arcs if arc.valid]
        for i, arc_a in enumerate(valid_arcs):
            for arc_b in valid_arcs[i + 1 :]:
                set_a = {event.id for event in arc_a.events}
                set_b = {event.id for event in arc_b.events}
                union = set_a | set_b
                overlap = len(set_a & set_b) / len(union) if union else 0.0
                matrix[_pair_key(arc_a.protagonist, arc_b.protagonist)] = overlap
        return matrix

    def turning_point_overlap(self) -> dict[str, list[str]]:
        """Event IDs that serve as turning points across protagonists."""
        per_event: dict[str, set[str]] = {}
        for arc in self.arcs:
            if not arc.valid:
                continue
            for event, beat in zip(arc.events, arc.beats):
                if beat != BeatType.TURNING_POINT:
                    continue
                per_event.setdefault(event.id, set()).add(arc.protagonist)
        return {event_id: sorted(protagonists) for event_id, protagonists in sorted(per_event.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "total_events": int(self.total_events),
            "valid_count": int(self.valid_count),
            "overlap_matrix": dict(self.overlap_matrix),
            "turning_point_overlap": self.turning_point_overlap(),
            "arcs": [arc.to_dict() for arc in self.arcs],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RashomonSet:
        arcs = [RashomonArc.from_dict(item) for item in (raw.get("arcs") or [])]
        total_events = int(raw.get("total_events", 0) or 0)
        if total_events <= 0 and arcs:
            ids = {event.id for arc in arcs for event in arc.events}
            total_events = len(ids)
        return cls(
            seed=int(raw.get("seed", 0) or 0),
            total_events=total_events,
            arcs=arcs,
        )


def extract_rashomon_set(
    events: list[Event],
    seed: int,
    agents: list[str] | None = None,
    max_events_per_arc: int = 20,
    total_sim_time: float | None = None,
    grammar_config: GrammarConfig | None = None,
) -> RashomonSet:
    """Extract one protagonist arc per agent from a shared event list."""
    if agents is None:
        discovered_agents: set[str] = set()
        for event in events:
            discovered_agents.add(event.source_agent)
            discovered_agents.update(event.target_agents)
        agents = sorted(discovered_agents)

    arcs: list[RashomonArc] = []
    for agent in agents:
        if grammar_config is None:
            search_result = search_arc(
                all_events=events,
                protagonist=agent,
                max_events=max_events_per_arc,
                total_sim_time=total_sim_time,
            )
        else:
            search_result = search_arc(
                all_events=events,
                protagonist=agent,
                max_events=max_events_per_arc,
                total_sim_time=total_sim_time,
                grammar_config=grammar_config,
            )
        arc_score: ArcScore | None = None
        if search_result.validation.valid:
            arc_score = search_result.arc_score or score_arc(search_result.events, search_result.beats)

        arcs.append(
            RashomonArc(
                protagonist=agent,
                search_result=search_result,
                arc_score=arc_score,
                events=search_result.events,
                beats=search_result.beats,
                valid=search_result.validation.valid,
                violation_count=len(search_result.validation.violations),
                violations=list(search_result.validation.violations),
            )
        )

    arcs.sort(
        key=lambda arc: (
            0 if arc.valid else 1,
            -(arc.arc_score.composite if arc.arc_score else 0.0),
            arc.violation_count,
            arc.protagonist,
        )
    )
    return RashomonSet(seed=seed, total_events=len(events), arcs=arcs)
