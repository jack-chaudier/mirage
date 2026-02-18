# Extraction & API Subsystem Audit Findings

**Auditor:** extraction-auditor
**Date:** 2026-02-08
**Scope:** `narrativefield/extraction/` (arc_search, beat_classifier, arc_validator, arc_scorer, beat_sheet, prose_generator, api_server, types)
**Spec Reference:** `specs/metrics/story-extraction.md`

---

## Executive Summary

The extraction subsystem is **functionally solid**. The end-to-end pipeline (simulation seed=42 -> metrics -> arc search -> beat classification -> validation -> scoring -> beat sheet -> LLM prompt) completes successfully with no crashes and produces a valid 20-event arc. All 50 automated checks pass. There are 12 findings (3 MEDIUM, 9 LOW), none CRITICAL or HIGH. The main risks are spec divergences in beat classification logic that could produce unexpected results with edge-case tension profiles.

**Test Results:** 50 pass, 0 fail, 12 findings, 5 info

---

## Findings

### F-E01 [MEDIUM]: FastAPI `on_event` Deprecation
**File:** `narrativefield/extraction/api_server.py:115`
**Description:** Uses `@app.on_event("startup")` which was deprecated in FastAPI 0.103+. Should migrate to the `lifespan` context manager pattern.
**Impact:** Future FastAPI upgrades will produce deprecation warnings, eventually breaking.
**Fix:** Replace with:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global _loaded_payload
    path = _default_payload_path()
    if path.exists():
        _loaded_payload = _load_payload(path)
    yield
    # shutdown (no-op)

app = FastAPI(title="...", version="1.0.0", lifespan=lifespan)
```

### F-E02 [MEDIUM]: Beat Classification Position Threshold Divergence
**File:** `narrativefield/extraction/beat_classifier.py:61-69`
**Spec Section:** 3.1 (`_classify_single_beat`)
**Description:** The spec defines three position zones: `< 0.25` (setup/complication), `0.25-0.70` (development), `>= 0.70` (consequence after peak). The implementation ignores the 0.70 boundary and instead uses `event_index < peak_tension_idx` for ALL events past 25%. This means events positioned past 70% but before the tension peak are classified as COMPLICATION/ESCALATION instead of ESCALATION-only (spec) or CONSEQUENCE (if we follow the spec's >=0.70 + after-peak rule).
**Impact:** When the tension peak occurs late in the arc (>70%), the impl correctly avoids premature CONSEQUENCE labeling. This is arguably a better heuristic. But it diverges from the spec's stated algorithm.
**Recommendation:** Update spec Section 3.1 to match the implementation's `event_index < peak_tension_idx` approach, which handles late-peak arcs better.

### F-E03 [MEDIUM]: Duplicate TURNING_POINT Resolution Conflict
**File:** `beat_classifier.py:91-96` vs `arc_search.py:465-470`
**Description:** Two separate passes handle duplicate TURNING_POINTs differently:
- `repair_beats()` (beat_classifier.py:93): Keeps the **first** TURNING_POINT, downgrades later ones to ESCALATION.
- `_enforce_monotonic_beats()` (arc_search.py:467): Keeps the **highest-tension** TURNING_POINT, downgrades others to ESCALATION.

Since `arc_search.search_arc` calls `classify_beats` (which includes `repair_beats`) and then calls `_enforce_monotonic_beats`, the second pass can override the first. If `repair_beats` kept the first TP and `_enforce_monotonic_beats` prefers a different (higher-tension) one, the result changes.
**Impact:** In practice both passes usually agree because `classify_beats` tends to assign TP to the peak-tension event. But with irony-collapse TPs (Rule 2), the two could disagree.
**Recommendation:** Remove duplicate-TP handling from `repair_beats` since `_enforce_monotonic_beats` runs afterward and is more principled (highest tension wins).

### F-E04 [MEDIUM]: Monotonic Beat Promotion Always Chooses COMPLICATION
**File:** `narrativefield/extraction/arc_search.py:442-444`
**Description:** `_enforce_monotonic_beats` promotes phase regressions by picking `_PHASE_TO_BEATS[phase_level][0]`. For phase level 1, this is always `COMPLICATION` (never `ESCALATION`), because `_PHASE_TO_BEATS = {1: [BeatType.COMPLICATION, BeatType.ESCALATION]}` and index [0] is COMPLICATION.
**Impact:** Events that should be ESCALATION (tension rising relative to previous event) are always demoted to COMPLICATION when monotonic repair kicks in. This affects arc scoring indirectly since the beat types are used for context.
**Recommendation:** Use tension comparison to choose between COMPLICATION and ESCALATION when promoting to phase 1.

### F-E05 [LOW]: CORS Allows All Origins
**File:** `narrativefield/extraction/api_server.py:106-111`
**Description:** `allow_origins=["*"]` permits any domain to make requests.
**Impact:** Fine for development. Should be restricted to the renderer's origin in production.

### F-E06 [LOW]: `classify_beats` Signature Diverges from Spec
**File:** `narrativefield/extraction/beat_classifier.py:6`
**Spec Section:** 3.1
**Description:** Spec: `classify_beats(events, scenes)`. Impl: `classify_beats(events)`. Scene context is not used in classification.
**Impact:** Low. Scene context could improve classification (e.g., location changes), but isn't needed for the current heuristic-based approach.

### F-E07 [LOW]: `score_arc` Signature Omits Spec Parameters
**File:** `narrativefield/extraction/arc_scorer.py:7`
**Spec Section:** 4.1
**Description:** Spec: `score_arc(events, beats, weights, scenes)`. Impl: `score_arc(events, beats)`. TensionWeights and scenes are not used.
**Impact:** Low. Custom scoring weights are a future extensibility feature.

### F-E08 [LOW]: `validate_arc` Signature Omits ArcGrammar
**File:** `narrativefield/extraction/arc_validator.py:7`
**Spec Section:** 2.3
**Description:** Spec passes an `ArcGrammar` parameter; impl hardcodes the grammar. Fine for MVP with a single grammar.

### F-E09 [LOW]: Spec Rule Numbering Mismatch
**File:** `narrativefield/extraction/arc_validator.py:72-90`
**Spec Section:** 2.2
**Description:** Spec numbers causal connectivity as "Rule 9" and time span as "Rule 10" (skipping Rule 8). Implementation comments use "Rule 8" and "Rule 9". The rules themselves are correctly implemented.

### F-E10 [LOW]: Tension Variance Normalization Threshold Too Low
**File:** `narrativefield/extraction/arc_scorer.py:17`
**Spec Section:** 4.1
**Description:** `tension_variance_score = min(tension_variance / 0.05, 1.0)`. Since tension is in [0,1], a variance of 0.05 is easily achieved with moderate spread (e.g., values ranging from 0.2 to 0.7). Most real arcs will score 1.0 on this component, reducing its discriminative power.
**Recommendation:** Raise threshold to 0.10 or use a sigmoid curve for smoother scoring.

### F-E11 [LOW]: ArcValidation Frozen with Mutable Field
**File:** `narrativefield/extraction/types.py:9-15`
**Description:** `ArcValidation` is `frozen=True` but contains `violations: list[str]`. Frozen dataclasses prevent attribute reassignment but mutable fields can still be mutated in-place via `v.violations.append(...)`.
**Impact:** Safe in current usage since violations are only set at creation.

### F-E12 [LOW]: Spec Gap for REVEAL in Early Position
**File:** `narrativefield/extraction/beat_classifier.py:57`
**Spec Section:** 3.2 (Heuristics Summary Table)
**Description:** The spec table covers early CONFIDE/LIE -> COMPLICATION and early CHAT/SOCIAL_MOVE/OBSERVE -> SETUP, but does not explicitly cover REVEAL in position < 0.25. Implementation classifies early REVEAL as COMPLICATION (bundled with CONFIDE/LIE at line 57). This is a reasonable choice but not documented in the spec.

---

## End-to-End Pipeline Results (seed=42)

| Step | Result |
|------|--------|
| Simulation | 200 events generated |
| Metrics pipeline | 124 events after bundling |
| Arc search | 20 events selected, protagonist=marcus |
| Arc validation | VALID (0 violations) |
| Beat sequence | setup(3) -> complication/escalation(12) -> turning_point(1) -> consequence(4) |
| Arc score composite | 0.4138 |
| Beat sheet | 20 beats, 6 character briefs, all fields populated |
| LLM prompt | Contains all 5 required sections |

### Scoring Breakdown

| Component | Score | Notes |
|-----------|-------|-------|
| tension_variance | 0.2141 | Low -- arc tension range is narrow |
| peak_tension | 0.6833 | Good peak |
| tension_shape | 0.7417 | Good rise-fall shape |
| significance | 0.0000 | No significance computed yet (Phase 5 feature) |
| thematic_coherence | 0.3738 | Scattered thematic shifts |
| irony_arc | 0.0122 | Nearly flat irony -- no meaningful build/collapse |
| protagonist_dominance | 1.0000 | Marcus appears in all 20 selected events |
| **Composite** | **0.4138** | Moderate quality arc |

### Observations
- The low irony_arc (0.01) and zero significance score drag the composite down
- significance=0.0 is expected since counterfactual impact (Phase 5) is not yet implemented
- protagonist_dominance at 1.0 means the arc search is heavily protagonist-centric
- thematic_coherence at 0.37 suggests multiple thematic axes are active

---

## Validation Coverage

| Check | Count | Status |
|-------|-------|--------|
| Arc validator rejects: too few beats | 1 | PASS |
| Arc validator rejects: missing TURNING_POINT | 1 | PASS |
| Arc validator rejects: phase order violation | 1 | PASS |
| Arc validator rejects: too-short time span | 1 | PASS |
| Arc validator rejects: no protagonist | 1 | PASS |
| Arc validator rejects: causal gap | 1 | PASS |
| Edge case: empty events | 1 | PASS |
| Edge case: single event | 1 | PASS |
| Edge case: flat tension | 1 | PASS |
| Edge case: zero tension | 1 | PASS |
| Edge case: REVEAL + irony collapse | 1 | PASS |
| Beat sheet: empty events | 1 | PASS |
| to_dict round-trips | 2 | PASS |
| API endpoint existence | 1 | PASS |
| API request model fields | 4 | PASS |
| Composite score bounds | 1 | PASS |
| All component scores in [0,1] | 7 | PASS |
| Composite = weighted sum | 1 | PASS |
| Weight sum = 1.0 | 1 | PASS |
| LLM prompt sections | 3 | PASS |

**Total: 50 pass, 0 fail**

---

## Recommendations (Priority Order)

1. **F-E01**: Migrate `on_event` to `lifespan` context manager (easy fix, prevents future breakage)
2. **F-E03**: Remove duplicate-TP handling from `repair_beats` to avoid conflict with `_enforce_monotonic_beats`
3. **F-E02**: Update spec to match implementation's peak-relative classification (or decide which is canonical)
4. **F-E04**: Use tension-relative logic when promoting regressions to phase 1 (COMPLICATION vs ESCALATION)
5. **F-E10**: Raise tension_variance threshold from 0.05 to 0.10

---

## Audit Script

The audit script is at `$PROJECT_ROOT/src/engine/audit_extraction.py`. Run with:
```bash
cd $PROJECT_ROOT/src/engine && source .venv/bin/activate && python audit_extraction.py
```
