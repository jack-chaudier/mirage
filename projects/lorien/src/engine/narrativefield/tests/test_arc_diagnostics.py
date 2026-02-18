from __future__ import annotations

import narrativefield.extraction.arc_search as arc_search
from narrativefield.extraction.arc_search import ArcSearchDiagnostics, _normalize_violation, search_arc
from narrativefield.extraction.types import ArcValidation
from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType


def _evt(
    event_id: str,
    *,
    sim_time: float,
    significance: float = 0.1,
    source: str = "alice",
) -> Event:
    return Event(
        id=event_id,
        sim_time=sim_time,
        tick_id=int(sim_time),
        order_in_tick=0,
        type=EventType.CHAT,
        source_agent=source,
        target_agents=["bob"],
        location_id="dining_table",
        causal_links=[],
        deltas=[],
        description=f"event {event_id}",
        metrics=EventMetrics(tension=0.2 + sim_time * 0.01, irony=0.1, significance=significance),
    )


def test_normalize_violation_coverage() -> None:
    cases = [
        ("Missing CONSEQUENCE beat", "missing_consequence"),
        ("Missing SETUP beat", "missing_setup"),
        ("Missing COMPLICATION or ESCALATION beat", "missing_development"),
        ("Expected 1 TURNING_POINT, found 0", "missing_turning_point"),
        ("Expected 1 TURNING_POINT, found 3", "duplicate_turning_point"),
        ("No protagonist: most frequent agent 'thorne' appears in 3/10 events (30%)", "no_protagonist"),
        ("Order violation: CONSEQUENCE after SETUP", "phase_order_violation"),
        ("Too few beats: 3 < 4", "too_few_beats"),
        ("Too many beats: 25 > 20", "too_many_beats"),
        ("Causal gap at event EVT_005: no causal link or participant overlap", "causal_gap"),
        ("Arc too short: spans 5.0 sim minutes (minimum 10)", "arc_too_short"),
        ("Some unknown violation", "unknown"),
    ]
    for violation, expected in cases:
        assert _normalize_violation(violation) == expected


def test_diagnostics_rule_counts_aggregated(monkeypatch) -> None:
    events = [
        _evt("e0", sim_time=0.0, significance=1.0),
        _evt("e1", sim_time=1.0, significance=0.9),
        _evt("e2", sim_time=2.0, significance=0.8),
        _evt("e3", sim_time=3.0, significance=0.7),
    ]
    by_id = {event.id: event for event in events}
    violations_by_anchor = {
        "e0": ("Missing CONSEQUENCE beat", "No protagonist: most frequent agent 'alice' appears in 2/3 events (66%)"),
        "e1": ("Missing CONSEQUENCE beat",),
        "e2": ("Causal gap at event EVT_005: no causal link or participant overlap",),
        "e3": ("Arc too short: spans 5.0 sim minutes (minimum 10)",),
    }

    def fake_downsample(*, anchor_id: str, **_kwargs):
        return [by_id[anchor_id], by_id["e0"], by_id["e3"]]

    def fake_validate(*, events: list[Event], **_kwargs) -> ArcValidation:
        anchor_id = events[0].id
        return ArcValidation(valid=False, violations=violations_by_anchor[anchor_id])

    monkeypatch.setattr(arc_search, "_downsample_preserving_continuity", fake_downsample)
    monkeypatch.setattr(arc_search, "classify_beats", lambda events: [BeatType.SETUP for _ in events])
    monkeypatch.setattr(arc_search, "_enforce_monotonic_beats", lambda events, beats: beats)
    monkeypatch.setattr(arc_search, "validate_arc", fake_validate)

    result = search_arc(events, max_events=3, total_sim_time=120.0)
    assert not result.validation.valid
    assert result.diagnostics is not None
    assert result.diagnostics.candidates_evaluated == 4
    assert result.diagnostics.rule_failure_counts["missing_consequence"] == 2
    assert result.diagnostics.rule_failure_counts["no_protagonist"] == 1
    assert result.diagnostics.rule_failure_counts["causal_gap"] == 1
    assert result.diagnostics.rule_failure_counts["arc_too_short"] == 1


def test_diagnostics_best_candidate_selected(monkeypatch) -> None:
    events = [
        _evt("e0", sim_time=0.0, significance=1.0),
        _evt("e1", sim_time=1.0, significance=0.9),
        _evt("e2", sim_time=2.0, significance=0.8),
    ]
    by_id = {event.id: event for event in events}
    violations_by_anchor = {
        "e0": ("Missing CONSEQUENCE beat", "Causal gap at event EVT_005: no causal link or participant overlap"),
        "e1": ("Missing CONSEQUENCE beat",),
        "e2": ("Missing CONSEQUENCE beat", "Arc too short: spans 5.0 sim minutes (minimum 10)", "Too few beats: 3 < 4"),
    }

    def fake_downsample(*, anchor_id: str, **_kwargs):
        return [by_id[anchor_id], by_id["e0"], by_id["e2"]]

    def fake_validate(*, events: list[Event], **_kwargs) -> ArcValidation:
        return ArcValidation(valid=False, violations=violations_by_anchor[events[0].id])

    monkeypatch.setattr(arc_search, "_downsample_preserving_continuity", fake_downsample)
    monkeypatch.setattr(arc_search, "classify_beats", lambda events: [BeatType.SETUP for _ in events])
    monkeypatch.setattr(arc_search, "_enforce_monotonic_beats", lambda events, beats: beats)
    monkeypatch.setattr(arc_search, "validate_arc", fake_validate)

    result = search_arc(events, max_events=3, total_sim_time=120.0)
    assert result.diagnostics is not None
    assert result.diagnostics.best_candidate_violation_count == 1
    assert result.diagnostics.best_candidate_violations == ["Missing CONSEQUENCE beat"]


def test_diagnostics_to_dict_roundtrip() -> None:
    diagnostics = ArcSearchDiagnostics(
        violations=["Missing CONSEQUENCE beat"],
        suggested_protagonist="alice",
        suggested_time_window=(1.0, 22.0),
        suggested_keep_ids=["e1", "e2"],
        suggested_drop_ids=["e3"],
        primary_failure="Missing CONSEQUENCE beat",
        rule_failure_counts={"missing_consequence": 3},
        best_candidate_violation_count=1,
        candidates_evaluated=8,
        best_candidate_violations=["Missing CONSEQUENCE beat"],
    )

    payload = diagnostics.to_dict()
    reconstructed = ArcSearchDiagnostics(
        violations=list(payload["violations"]),
        suggested_protagonist=str(payload["suggested_protagonist"]),
        suggested_time_window=tuple(payload["suggested_time_window"]) if payload["suggested_time_window"] else None,
        suggested_keep_ids=list(payload["suggested_keep_ids"]),
        suggested_drop_ids=list(payload["suggested_drop_ids"]),
        primary_failure=str(payload["primary_failure"]),
        rule_failure_counts=dict(payload["rule_failure_counts"]),
        best_candidate_violation_count=int(payload["best_candidate_violation_count"]),
        candidates_evaluated=int(payload["candidates_evaluated"]),
        best_candidate_violations=list(payload["best_candidate_violations"]),
    )
    assert reconstructed.to_dict() == payload


def test_diagnostics_backward_compat_defaults() -> None:
    diagnostics = ArcSearchDiagnostics()
    assert diagnostics.rule_failure_counts == {}
    assert diagnostics.best_candidate_violation_count == 0
    assert diagnostics.candidates_evaluated == 0
    assert diagnostics.best_candidate_violations == []
