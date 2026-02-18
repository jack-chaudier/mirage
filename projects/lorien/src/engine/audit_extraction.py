#!/usr/bin/env python3
"""
Comprehensive end-to-end audit of the NarrativeField extraction pipeline.

Runs simulation with seed 42, metrics pipeline, arc search, and validates
every component of the extraction subsystem against the spec.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from random import Random
from typing import Any

FAKE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fake-dinner-party.nf-viz.json"


# ---------------------------------------------------------------------------
# 1. Simulation run (seed 42)
# ---------------------------------------------------------------------------

def run_simulation(seed: int = 42) -> dict[str, Any]:
    """Run the simulation and return the raw JSON-serializable result dict."""
    from narrativefield.simulation.scenarios import create_dinner_party_world
    from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation as _run_sim

    world = create_dinner_party_world()
    rng = Random(seed)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=world.definition.sim_duration_minutes,
        snapshot_interval_events=world.definition.snapshot_interval,
    )
    events, snapshots = _run_sim(world, rng, cfg)
    return {
        "format_version": "1.0.0",
        "metadata": {
            "scenario": "dinner_party",
            "total_ticks": world.tick_id,
            "total_sim_time": world.sim_time,
            "agent_count": len(world.agents),
            "event_count": len(events),
            "seed": seed,
        },
        "initial_state": snapshots[0] if snapshots else {},
        "snapshots": snapshots[1:],
        "events": [e.to_dict() for e in events],
        "secrets": [s.to_dict() for s in world.definition.secrets.values()],
        "locations": [loc.to_dict() for loc in world.definition.locations.values()],
    }


# ---------------------------------------------------------------------------
# 2. Metrics pipeline
# ---------------------------------------------------------------------------

def run_metrics(sim_result: dict[str, Any]) -> dict[str, Any]:
    """Run metrics pipeline and return the renderer payload."""
    from narrativefield.integration.bundler import BundleInputs, bundle_for_renderer
    from narrativefield.metrics.pipeline import parse_simulation_output, run_metrics_pipeline

    parsed = parse_simulation_output(sim_result)
    out = run_metrics_pipeline(parsed)
    payload = bundle_for_renderer(
        BundleInputs(
            metadata=parsed.metadata,
            initial_agents=parsed.initial_agents,
            snapshots=out.belief_snapshots,
            events=out.events,
            scenes=out.scenes,
            secrets=parsed.secrets,
            locations=parsed.locations,
        )
    )
    return payload


# ---------------------------------------------------------------------------
# 3. Arc search
# ---------------------------------------------------------------------------

def run_arc_search(payload: dict[str, Any]):
    """Run arc search on the renderer payload."""
    from narrativefield.extraction.arc_search import search_arc
    from narrativefield.schema.events import Event

    events = [Event.from_dict(e) for e in payload["events"]]
    events.sort(key=lambda e: (e.sim_time, e.tick_id, e.order_in_tick))
    total_sim_time = payload.get("metadata", {}).get("total_sim_time", 120.0)

    result = search_arc(
        all_events=events,
        time_start=None,
        time_end=None,
        max_events=20,
        total_sim_time=float(total_sim_time),
    )
    return result


# ---------------------------------------------------------------------------
# 4. Beat classification validation
# ---------------------------------------------------------------------------

def validate_beat_classification(result) -> list[str]:
    """Validate beats follow SETUP->COMPLICATION->ESCALATION->TURNING_POINT->CONSEQUENCE."""
    from narrativefield.schema.events import BeatType

    findings: list[str] = []
    beats = result.beats

    if not beats:
        findings.append("CRITICAL: No beats produced by arc search")
        return findings

    tp_count = sum(1 for b in beats if b == BeatType.TURNING_POINT)
    if tp_count != 1:
        findings.append(f"FAIL: Expected exactly 1 TURNING_POINT, got {tp_count}")
    else:
        findings.append("PASS: Exactly 1 TURNING_POINT found")

    setup_count = sum(1 for b in beats if b == BeatType.SETUP)
    if setup_count < 1:
        findings.append("FAIL: Missing SETUP beat")
    else:
        findings.append(f"PASS: {setup_count} SETUP beat(s)")

    dev_count = sum(1 for b in beats if b in {BeatType.COMPLICATION, BeatType.ESCALATION})
    if dev_count < 1:
        findings.append("FAIL: Missing COMPLICATION/ESCALATION beat")
    else:
        findings.append(f"PASS: {dev_count} development beat(s)")

    conseq_count = sum(1 for b in beats if b == BeatType.CONSEQUENCE)
    if conseq_count < 1:
        findings.append("FAIL: Missing CONSEQUENCE beat")
    else:
        findings.append(f"PASS: {conseq_count} CONSEQUENCE beat(s)")

    phase_order = {
        BeatType.SETUP: 0,
        BeatType.COMPLICATION: 1,
        BeatType.ESCALATION: 1,
        BeatType.TURNING_POINT: 2,
        BeatType.CONSEQUENCE: 3,
    }
    monotonic_ok = True
    for i in range(1, len(beats)):
        if phase_order[beats[i]] < phase_order[beats[i - 1]]:
            findings.append(f"FAIL: Phase regression at index {i}: {beats[i].value} after {beats[i-1].value}")
            monotonic_ok = False
            break
    if monotonic_ok:
        findings.append("PASS: Monotonic phase ordering holds")

    seq_str = " -> ".join(b.value for b in beats)
    findings.append(f"INFO: Beat sequence: {seq_str}")

    return findings


# ---------------------------------------------------------------------------
# 5. Arc validation
# ---------------------------------------------------------------------------

def validate_arc_validation_catches() -> list[str]:
    """Test that arc_validator correctly rejects bad inputs."""
    from narrativefield.extraction.arc_validator import validate_arc
    from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType

    findings: list[str] = []

    def make_evt(i, t, et=EventType.CHAT, tension=0.3, src="a"):
        return Event(
            id=f"test-{i}", sim_time=t, tick_id=i, order_in_tick=0,
            type=et, source_agent=src,
            target_agents=["b"] if et != EventType.INTERNAL else [],
            location_id="dining_table",
            causal_links=[f"test-{i-1}"] if i > 0 else [],
            deltas=[], description=f"test event {i}",
            metrics=EventMetrics(tension=tension),
        )

    # Test 1: Too few beats
    short_events = [make_evt(i, float(i)) for i in range(3)]
    short_beats = [BeatType.SETUP, BeatType.TURNING_POINT, BeatType.CONSEQUENCE]
    v = validate_arc(events=short_events, beats=short_beats)
    if not v.valid and any("Too few" in viol for viol in v.violations):
        findings.append("PASS: Validator rejects too-few-beats (<4)")
    else:
        findings.append(f"FAIL: Validator did not reject 3 beats. valid={v.valid}, violations={v.violations}")

    # Test 2: Missing TURNING_POINT
    evts = [make_evt(i, float(i) * 3) for i in range(5)]
    beats_no_tp = [BeatType.SETUP, BeatType.COMPLICATION, BeatType.ESCALATION, BeatType.CONSEQUENCE, BeatType.CONSEQUENCE]
    v2 = validate_arc(events=evts, beats=beats_no_tp)
    if not v2.valid and any("TURNING_POINT" in viol for viol in v2.violations):
        findings.append("PASS: Validator catches missing TURNING_POINT")
    else:
        findings.append(f"FAIL: Validator did not catch missing TP. valid={v2.valid}, violations={v2.violations}")

    # Test 3: Phase order violation
    evts3 = [make_evt(i, float(i) * 3) for i in range(5)]
    beats_bad_order = [BeatType.CONSEQUENCE, BeatType.SETUP, BeatType.COMPLICATION, BeatType.TURNING_POINT, BeatType.CONSEQUENCE]
    v3 = validate_arc(events=evts3, beats=beats_bad_order)
    if not v3.valid and any("Order violation" in viol for viol in v3.violations):
        findings.append("PASS: Validator catches phase order violation")
    else:
        findings.append(f"FAIL: Validator did not catch order violation. valid={v3.valid}, violations={v3.violations}")

    # Test 4: Arc too short (time span < 10 min)
    short_time_evts = [make_evt(i, float(i) * 2) for i in range(5)]
    short_time_beats = [BeatType.SETUP, BeatType.COMPLICATION, BeatType.ESCALATION, BeatType.TURNING_POINT, BeatType.CONSEQUENCE]
    v4 = validate_arc(events=short_time_evts, beats=short_time_beats)
    if not v4.valid and any("too short" in viol.lower() for viol in v4.violations):
        findings.append("PASS: Validator catches too-short time span (<10 min)")
    else:
        findings.append(f"FAIL: Validator did not catch short span. valid={v4.valid}, violations={v4.violations}")

    # Test 5: No protagonist
    agents = ["a", "b", "c", "d", "e"]
    mixed_evts = []
    for i in range(10):
        src = agents[i % 5]
        tgt = agents[(i + 1) % 5]
        mixed_evts.append(Event(
            id=f"mix-{i}", sim_time=float(i) * 3, tick_id=i, order_in_tick=0,
            type=EventType.CHAT, source_agent=src, target_agents=[tgt],
            location_id="dining_table",
            causal_links=[f"mix-{i-1}"] if i > 0 else [],
            deltas=[], description=f"mixed {i}",
            metrics=EventMetrics(tension=0.3),
        ))
    mixed_beats = [BeatType.SETUP, BeatType.SETUP, BeatType.COMPLICATION, BeatType.ESCALATION,
                   BeatType.ESCALATION, BeatType.ESCALATION, BeatType.ESCALATION,
                   BeatType.TURNING_POINT, BeatType.CONSEQUENCE, BeatType.CONSEQUENCE]
    v5 = validate_arc(events=mixed_evts, beats=mixed_beats)
    if not v5.valid and any("protagonist" in viol.lower() for viol in v5.violations):
        findings.append("PASS: Validator catches missing protagonist (no 60% dominance)")
    else:
        findings.append(f"INFO: Protagonist check: valid={v5.valid}, violations={v5.violations}")

    # Test 6: Causal gap
    gap_evts = [make_evt(0, 0.0), make_evt(1, 5.0),
                Event(id="test-2", sim_time=10.0, tick_id=2, order_in_tick=0,
                      type=EventType.CHAT, source_agent="z", target_agents=["y"],
                      location_id="dining_table", causal_links=[], deltas=[],
                      description="isolated", metrics=EventMetrics(tension=0.3)),
                make_evt(3, 15.0), make_evt(4, 20.0)]
    gap_beats = [BeatType.SETUP, BeatType.COMPLICATION, BeatType.ESCALATION, BeatType.TURNING_POINT, BeatType.CONSEQUENCE]
    v6 = validate_arc(events=gap_evts, beats=gap_beats)
    if not v6.valid and any("Causal gap" in viol for viol in v6.violations):
        findings.append("PASS: Validator catches causal gap")
    else:
        findings.append(f"FAIL: Validator did not catch causal gap. valid={v6.valid}, violations={v6.violations}")

    return findings


# ---------------------------------------------------------------------------
# 6. Arc scoring validation
# ---------------------------------------------------------------------------

def validate_arc_scoring(result) -> list[str]:
    """Check arc scoring components and composite bounds."""
    findings: list[str] = []

    if result.arc_score is None:
        if result.validation.valid:
            findings.append("FAIL: Arc is valid but arc_score is None")
        else:
            findings.append("INFO: Arc not valid, no score computed (expected)")
        return findings

    score = result.arc_score

    if 0.0 <= score.composite <= 1.0:
        findings.append(f"PASS: Composite score {score.composite:.4f} in [0, 1]")
    else:
        findings.append(f"FAIL: Composite score {score.composite:.4f} out of [0, 1]")

    components = {
        "tension_variance": score.tension_variance,
        "peak_tension": score.peak_tension,
        "tension_shape": score.tension_shape,
        "significance": score.significance,
        "thematic_coherence": score.thematic_coherence,
        "irony_arc": score.irony_arc,
        "protagonist_dominance": score.protagonist_dominance,
    }
    for name, val in components.items():
        if 0.0 <= val <= 1.0:
            findings.append(f"PASS: {name} = {val:.4f} in [0, 1]")
        else:
            findings.append(f"FAIL: {name} = {val:.4f} OUT OF [0, 1]")

    expected = (
        0.20 * score.tension_variance
        + 0.15 * score.peak_tension
        + 0.15 * score.tension_shape
        + 0.15 * score.significance
        + 0.15 * score.thematic_coherence
        + 0.10 * score.irony_arc
        + 0.10 * score.protagonist_dominance
    )
    if abs(expected - score.composite) < 1e-9:
        findings.append(f"PASS: Composite matches weighted sum (delta={abs(expected - score.composite):.2e})")
    else:
        findings.append(f"FAIL: Composite {score.composite:.6f} != weighted sum {expected:.6f}")

    weight_sum = 0.20 + 0.15 + 0.15 + 0.15 + 0.15 + 0.10 + 0.10
    if abs(weight_sum - 1.0) < 1e-9:
        findings.append("PASS: Scoring weights sum to 1.0")
    else:
        findings.append(f"FAIL: Scoring weights sum to {weight_sum}")

    return findings


# ---------------------------------------------------------------------------
# 7. Beat sheet generation
# ---------------------------------------------------------------------------

def validate_beat_sheet(result) -> list[str]:
    """Test beat sheet generation."""
    from narrativefield.extraction.beat_sheet import build_beat_sheet
    from narrativefield.extraction.prose_generator import build_llm_prompt

    findings: list[str] = []

    if not result.events or not result.validation.valid or result.arc_score is None:
        findings.append("SKIP: Cannot build beat sheet - arc not valid")
        return findings

    beat_sheet = build_beat_sheet(
        events=result.events,
        beats=result.beats,
        protagonist=result.protagonist,
        genre_preset="default",
        arc_score=result.arc_score,
    )

    if beat_sheet.protagonist == result.protagonist:
        findings.append(f"PASS: Protagonist matches: {beat_sheet.protagonist}")
    else:
        findings.append("FAIL: Protagonist mismatch")

    if beat_sheet.beats:
        findings.append(f"PASS: Beat sheet has {len(beat_sheet.beats)} beats")
    else:
        findings.append("FAIL: Beat sheet has no beats")

    if len(beat_sheet.beats) == len(result.events):
        findings.append("PASS: Beat count matches event count")
    else:
        findings.append(f"FAIL: Beat count {len(beat_sheet.beats)} != event count {len(result.events)}")

    for i, beat in enumerate(beat_sheet.beats):
        if not beat.event_id:
            findings.append(f"FAIL: Beat {i} missing event_id")
        if not beat.description:
            findings.append(f"FAIL: Beat {i} missing description")
        if not beat.location:
            findings.append(f"FAIL: Beat {i} missing location")

    if beat_sheet.characters:
        findings.append(f"PASS: {len(beat_sheet.characters)} character brief(s)")
        proto_chars = [c for c in beat_sheet.characters if c.agent_id == result.protagonist]
        if proto_chars:
            findings.append("PASS: Protagonist in character briefs")
        else:
            findings.append("FAIL: Protagonist missing from character briefs")
    else:
        findings.append("FAIL: No character briefs")

    try:
        d = beat_sheet.to_dict()
        assert isinstance(d, dict) and "beats" in d and "characters" in d
        findings.append("PASS: BeatSheet.to_dict() works")
    except Exception as exc:
        findings.append(f"FAIL: BeatSheet.to_dict() error: {exc}")

    prompt = build_llm_prompt(beat_sheet)
    if "BEAT SEQUENCE" in prompt:
        findings.append("PASS: LLM prompt contains BEAT SEQUENCE")
    else:
        findings.append("FAIL: LLM prompt missing BEAT SEQUENCE")
    if "CONSTRAINTS" in prompt:
        findings.append("PASS: LLM prompt contains CONSTRAINTS")
    else:
        findings.append("FAIL: LLM prompt missing CONSTRAINTS")
    if beat_sheet.protagonist in prompt:
        findings.append("PASS: Protagonist in LLM prompt")
    else:
        findings.append("FAIL: Protagonist missing from LLM prompt")

    # Edge case: empty events
    try:
        empty_sheet = build_beat_sheet(
            events=[], beats=[], protagonist="nobody",
            genre_preset="default", arc_score=result.arc_score,
        )
        findings.append(f"PASS: Empty beat sheet builds (arc_id={empty_sheet.arc_id})")
    except Exception as exc:
        findings.append(f"FAIL: Empty beat sheet crashes: {exc}")

    return findings


# ---------------------------------------------------------------------------
# 8. API server audit
# ---------------------------------------------------------------------------

def validate_api_server() -> list[str]:
    """Check API server endpoints, schemas, deprecation warnings."""
    import inspect

    findings: list[str] = []

    from narrativefield.extraction import api_server

    source = inspect.getsource(api_server)
    if "on_event" in source:
        findings.append(
            "FINDING [MEDIUM]: api_server.py uses deprecated @app.on_event('startup') "
            "(FastAPI 0.103+ deprecation). Should migrate to lifespan context manager. "
            "File: narrativefield/extraction/api_server.py:115"
        )

    from narrativefield.extraction.api_server import app
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    if "/extract" in routes:
        findings.append("PASS: /extract endpoint exists")
    else:
        findings.append("FAIL: /extract endpoint missing")

    cors_found = any("CORS" in str(m) for m in app.user_middleware)
    findings.append("PASS: CORS middleware configured" if cors_found else "INFO: No CORS middleware")

    from narrativefield.extraction.api_server import StoryExtractionRequestModel
    model_fields = list(StoryExtractionRequestModel.model_fields.keys())
    for f in ["selection_type", "event_ids", "protagonist_agent_id", "genre_preset"]:
        findings.append(f"PASS: Request model has '{f}'" if f in model_fields else f"FAIL: Request model missing '{f}'")

    sel_field = StoryExtractionRequestModel.model_fields.get("selection_type")
    if sel_field and "search" in str(sel_field.annotation):
        findings.append("PASS: selection_type supports 'search' mode")

    findings.append(
        "FINDING [LOW]: CORS allows all origins (allow_origins=['*']). "
        "Consider restricting in production. File: narrativefield/extraction/api_server.py:106-111"
    )

    from narrativefield.extraction.api_server import _default_payload_path
    path = _default_payload_path()
    findings.append(f"INFO: Default payload path: {path}")
    findings.append("PASS: Default payload file exists" if path.exists() else "INFO: Default payload file not found")

    # Check that the /extract endpoint auto-prefilter is wired correctly
    if "len(req.event_ids) > 50" in source and "len(req.selected_events) > 50" in source:
        findings.append("PASS: Auto-prefilter threshold (>50 events) present")
    else:
        findings.append("INFO: Could not confirm auto-prefilter threshold in source")

    # Check _pick_protagonist exists and returns max by count
    if "_pick_protagonist" in source:
        findings.append("PASS: _pick_protagonist helper exists")
    else:
        findings.append("FAIL: _pick_protagonist helper missing")

    return findings


# ---------------------------------------------------------------------------
# 9. Spec vs implementation divergence
# ---------------------------------------------------------------------------

def validate_spec_divergence() -> list[str]:
    findings: list[str] = []

    findings.append(
        "FINDING [LOW]: beat_classifier.classify_beats signature diverges from spec. "
        "Spec: classify_beats(events, scenes). Impl: classify_beats(events). "
        "No scene context is used. File: narrativefield/extraction/beat_classifier.py:6"
    )

    findings.append(
        "FINDING [MEDIUM]: beat_classifier._classify_single_beat diverges from spec position "
        "thresholds. Spec: 0.25/0.70 boundaries. Impl: uses event_index < peak_tension_idx "
        "for all events past 25%, ignoring the 0.70 boundary. Events past 70% but before "
        "peak are COMPLICATION/ESCALATION in impl vs ESCALATION-only in spec. "
        "File: narrativefield/extraction/beat_classifier.py:61-69"
    )

    findings.append(
        "FINDING [MEDIUM]: Duplicate TURNING_POINT resolution differs. "
        "repair_beats keeps FIRST TP (line 93). _enforce_monotonic_beats keeps "
        "HIGHEST-TENSION TP (line 467). Second pass can override first. "
        "File: beat_classifier.py:91-96 vs arc_search.py:465-470"
    )

    findings.append(
        "FINDING [LOW]: arc_scorer.score_arc signature omits weights and scenes "
        "parameters from spec. File: narrativefield/extraction/arc_scorer.py:7"
    )

    findings.append(
        "FINDING [LOW]: arc_validator.validate_arc signature omits ArcGrammar parameter. "
        "Grammar is hardcoded. Fine for MVP. File: narrativefield/extraction/arc_validator.py:7"
    )

    findings.append(
        "FINDING [LOW]: Spec rule numbering skips Rule 8 (causal connectivity is Rule 9, "
        "time span is Rule 10). Implementation uses Rules 8 and 9. "
        "File: narrativefield/extraction/arc_validator.py:72-90"
    )

    findings.append(
        "FINDING [LOW]: tension_variance normalization threshold 0.05 is low. "
        "Most arcs score 1.0 on this component, reducing discrimination. "
        "File: narrativefield/extraction/arc_scorer.py:17"
    )

    findings.append(
        "FINDING [MEDIUM]: _enforce_monotonic_beats always promotes regressions to "
        "COMPLICATION at phase 1 (_PHASE_TO_BEATS[level][0]), never ESCALATION. "
        "File: narrativefield/extraction/arc_search.py:442-444"
    )

    findings.append(
        "FINDING [LOW]: ArcValidation is frozen=True with mutable list[str] 'violations'. "
        "Safe in practice but semantically inconsistent. "
        "File: narrativefield/extraction/types.py:9-15"
    )

    # Check spec says REVEAL + CONFIDE/LIE in early position -> COMPLICATION
    # but impl also classifies REVEAL there as COMPLICATION (confirmed in line 57)
    findings.append(
        "FINDING [LOW]: Spec says early CONFIDE/LIE -> COMPLICATION. Impl also maps early "
        "REVEAL -> COMPLICATION (line 57). Spec says early REVEAL should be... unclear. "
        "The spec table (Section 3.2) does not cover REVEAL in position < 0.25. "
        "Implementation adds REVEAL to the COMPLICATION bucket at line 57. "
        "File: narrativefield/extraction/beat_classifier.py:57"
    )

    return findings


# ---------------------------------------------------------------------------
# 10. Edge case tests
# ---------------------------------------------------------------------------

def validate_edge_cases() -> list[str]:
    from narrativefield.extraction.arc_scorer import score_arc
    from narrativefield.extraction.beat_classifier import classify_beats
    from narrativefield.extraction.arc_search import ArcSearchResult
    from narrativefield.extraction.types import ArcValidation
    from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType, IronyCollapseInfo

    findings: list[str] = []

    def make_evt(i, t, et=EventType.CHAT, tension=0.3, src="a"):
        return Event(
            id=f"edge-{i}", sim_time=t, tick_id=i, order_in_tick=0,
            type=et, source_agent=src, target_agents=["b"],
            location_id="dining_table",
            causal_links=[f"edge-{i-1}"] if i > 0 else [],
            deltas=[], description=f"edge event {i}",
            metrics=EventMetrics(tension=tension, irony=0.3),
        )

    # Empty events
    try:
        assert classify_beats([]) == []
        findings.append("PASS: classify_beats([]) returns []")
    except Exception as exc:
        findings.append(f"FAIL: classify_beats([]) crashes: {exc}")

    # Single event
    try:
        beats = classify_beats([make_evt(0, 0.0)])
        assert len(beats) == 1
        findings.append(f"PASS: classify_beats(1 event) -> {beats[0].value}")
    except Exception as exc:
        findings.append(f"FAIL: classify_beats(1 event) crashes: {exc}")

    # Flat tension
    flat = [make_evt(i, float(i) * 3, tension=0.5) for i in range(8)]
    beats = classify_beats(flat)
    findings.append("PASS: Flat tension -> TP (repair)" if BeatType.TURNING_POINT in beats
                     else "FAIL: Flat tension no TURNING_POINT")

    # All-zero tension
    zero = [make_evt(i, float(i) * 3, tension=0.0) for i in range(8)]
    zbeats = classify_beats(zero)
    try:
        score = score_arc(zero, zbeats)
        findings.append(f"PASS: Zero-tension peak_tension={score.peak_tension:.2f}" if score.peak_tension == 0.0
                         else f"FAIL: peak_tension={score.peak_tension}")
    except Exception as exc:
        findings.append(f"FAIL: score_arc crashes on zeros: {exc}")

    # to_dict round trip
    try:
        r = ArcSearchResult(events=[], beats=[], protagonist="test",
                            validation=ArcValidation(valid=False, violations=["test"]))
        d = r.to_dict()
        assert d["protagonist"] == "test" and not d["validation"]["valid"]
        findings.append("PASS: ArcSearchResult.to_dict() for empty result")
    except Exception as exc:
        findings.append(f"FAIL: to_dict error: {exc}")

    # REVEAL + irony collapse
    reveal = make_evt(5, 15.0, et=EventType.REVEAL, tension=0.7)
    reveal.metrics.irony_collapse = IronyCollapseInfo(detected=True, drop=0.4, collapsed_beliefs=[], score=0.8)
    mixed = [make_evt(0, 0.0), make_evt(1, 3.0), make_evt(2, 6.0),
             make_evt(3, 9.0), make_evt(4, 12.0), reveal,
             make_evt(6, 18.0), make_evt(7, 21.0)]
    beats = classify_beats(mixed)
    tp_idx = [i for i, b in enumerate(beats) if b == BeatType.TURNING_POINT]
    findings.append(f"PASS: REVEAL + irony collapse -> TP at {tp_idx}" if tp_idx
                     else "FAIL: REVEAL + irony collapse no TP")

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_findings: dict[str, list[str]] = {}
    errors: list[str] = []

    print("=" * 60)
    print("NarrativeField Extraction Pipeline Audit")
    print("=" * 60)

    # Step 1: Simulation
    print("\n[1/9] Running simulation (seed=42)...")
    try:
        sim_result = run_simulation(42)
        event_count = len(sim_result.get("events", []))
        print(f"  -> {event_count} events generated")
        all_findings["simulation"] = [f"PASS: Simulation generated {event_count} events"]
    except Exception as exc:
        print(f"  -> FAILED: {exc}")
        traceback.print_exc()
        errors.append(f"Simulation: {exc}")
        all_findings["simulation"] = [f"FAIL: {exc}"]
        sim_result = None

    # Step 2: Metrics
    print("\n[2/9] Running metrics pipeline...")
    payload = None
    if sim_result:
        try:
            payload = run_metrics(sim_result)
            event_count = len(payload.get("events", []))
            print(f"  -> Metrics computed for {event_count} events")
            all_findings["metrics"] = [f"PASS: Metrics pipeline processed {event_count} events"]
        except Exception as exc:
            print(f"  -> FAILED: {exc}")
            traceback.print_exc()
            errors.append(f"Metrics: {exc}")
            all_findings["metrics"] = [f"FAIL: {exc}"]

    # Fallback: use fake data file
    if payload is None and FAKE_DATA_PATH.exists():
        print("  -> Falling back to fake data file")
        payload = json.loads(FAKE_DATA_PATH.read_text(encoding="utf-8"))
        all_findings.setdefault("metrics", []).append(f"INFO: Used fake data from {FAKE_DATA_PATH}")

    # Step 3: Arc search
    print("\n[3/9] Running arc search...")
    arc_result = None
    if payload:
        try:
            arc_result = run_arc_search(payload)
            print(f"  -> {len(arc_result.events)} events, protagonist={arc_result.protagonist}, valid={arc_result.validation.valid}")
            if arc_result.validation.violations:
                for v in arc_result.validation.violations:
                    print(f"     violation: {v}")
            all_findings["arc_search"] = [
                f"PASS: Arc search selected {len(arc_result.events)} events",
                f"INFO: Protagonist: {arc_result.protagonist}",
                f"INFO: Valid: {arc_result.validation.valid}",
            ]
            if arc_result.arc_score:
                all_findings["arc_search"].append(f"INFO: Composite: {arc_result.arc_score.composite:.4f}")
        except Exception as exc:
            print(f"  -> FAILED: {exc}")
            traceback.print_exc()
            errors.append(f"Arc search: {exc}")
            all_findings["arc_search"] = [f"FAIL: {exc}"]
    else:
        all_findings["arc_search"] = ["SKIP: No data"]

    # Step 4: Beat classification
    print("\n[4/9] Validating beat classification...")
    if arc_result:
        bf = validate_beat_classification(arc_result)
        all_findings["beat_classification"] = bf
        for f in bf:
            print(f"  {f}")
    else:
        all_findings["beat_classification"] = ["SKIP: No arc result"]

    # Step 5: Arc validation
    print("\n[5/9] Validating arc_validator catches...")
    vf = validate_arc_validation_catches()
    all_findings["arc_validation"] = vf
    for f in vf:
        print(f"  {f}")

    # Step 6: Arc scoring
    print("\n[6/9] Validating arc scoring...")
    if arc_result:
        sf = validate_arc_scoring(arc_result)
        all_findings["arc_scoring"] = sf
        for f in sf:
            print(f"  {f}")
    else:
        all_findings["arc_scoring"] = ["SKIP: No arc result"]

    # Step 7: Beat sheet
    print("\n[7/9] Validating beat sheet generation...")
    if arc_result:
        bsf = validate_beat_sheet(arc_result)
        all_findings["beat_sheet"] = bsf
        for f in bsf:
            print(f"  {f}")
    else:
        all_findings["beat_sheet"] = ["SKIP: No arc result"]

    # Step 8: API server
    print("\n[8/9] Auditing API server...")
    af = validate_api_server()
    all_findings["api_server"] = af
    for f in af:
        print(f"  {f}")

    # Step 9: Spec divergence
    print("\n[9/9] Spec vs implementation divergence...")
    spf = validate_spec_divergence()
    all_findings["spec_divergence"] = spf
    for f in spf:
        print(f"  {f}")

    # Bonus: Edge cases
    print("\n[Bonus] Edge case tests...")
    ef = validate_edge_cases()
    all_findings["edge_cases"] = ef
    for f in ef:
        print(f"  {f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_pass = total_fail = total_findings = total_info = 0
    for section, findings in all_findings.items():
        p = sum(1 for f in findings if f.startswith("PASS"))
        fl = sum(1 for f in findings if f.startswith("FAIL"))
        fi = sum(1 for f in findings if f.startswith("FINDING"))
        inf = sum(1 for f in findings if f.startswith(("INFO", "SKIP")))
        total_pass += p
        total_fail += fl
        total_findings += fi
        total_info += inf
        status = "OK" if fl == 0 else "ISSUES"
        print(f"  [{status}] {section}: {p} pass, {fl} fail, {fi} finding(s)")

    print(f"\nTotals: {total_pass} pass, {total_fail} fail, {total_findings} findings, {total_info} info")
    if errors:
        print(f"\nCritical errors: {len(errors)}")
        for e in errors:
            print(f"  - {e}")

    return all_findings


if __name__ == "__main__":
    main()
