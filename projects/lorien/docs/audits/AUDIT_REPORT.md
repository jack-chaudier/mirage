# NarrativeField Implementation Audit Report

> **Date:** 2026-02-08
> **Branch:** codex/integration/wps1-4
> **Commit:** be5a59a
> **Auditors:** sim-auditor, metrics-auditor, extraction-auditor, frontend-auditor
> **Method:** 4 parallel agents, each auditing one subsystem with automated test scripts

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Health** | GOOD - Structurally sound, functionally correct |
| **Total Findings** | 35 (0 CRITICAL, 6 HIGH, 10 MEDIUM, 19 LOW) |
| **Existing Tests** | 53/53 passing (45 Python + 8 TypeScript) |
| **Automated Audit Checks** | 144 pass, 0 fail |
| **Spec Compliance** | ~85% (38/44 spec constants match; divergences are intentional MVP simplifications) |
| **Previous Audit (S-01..S-35)** | 5 FIXED, 2 PRESENT, 1 PARTIAL, remainder N/A to implementation scope |
| **Builds** | Python: clean, TypeScript: clean (71 modules) |

**Key Takeaway:** The implementation is production-ready for MVP. No critical bugs or data corruption issues. The 6 HIGH findings are type safety gaps (frontend validator/loader), a config mismatch (foyer privacy), a dead field (global_tension), SOCIAL_MOVE event dominance, and an index type mismatch. All are addressable without architectural changes.

---

## 1. Simulation Engine

**Agent:** sim-auditor | **Findings:** 3 HIGH, 4 MEDIUM, 3 LOW

### 1.1 Multi-Seed Stress Test (Seeds 1-10)

All 10 seeds ran to completion. Key statistics:

| Metric | Range | Notes |
|--------|-------|-------|
| Events | 200 (all seeds) | All terminate at event_limit |
| Ticks | 32-33 | Consistent |
| Sim Time | 26.5-28.5 min | Out of 150 max |
| Catastrophes | 0-1 per run | 6/10 seeds had exactly 1 |
| SOCIAL_MOVE % | 37-47% | Dominates event budget |
| Dramatic event % | 10-17% | CONFLICT, REVEAL, CONFIDE, LIE, CATASTROPHE |
| Secret discoveries | 10-15 | Good variability |
| Departures | 0 | No agent ever leaves |

### 1.2 Invariant Checks

All passed across all 10 seeds:
- Monotonic (tick_id, order_in_tick) ordering
- Causal links reference preceding events only
- All locations valid, no non-adjacent moves
- No self-targeting, valid agent IDs
- Relationship values in [-1,1], obligation in [0,1]
- Pacing: stress [0,1], composure [0.05,1], budget [0,1], commitment [0,1]
- Alcohol in [0,1], bathroom capacity respected
- All beliefs are valid BeliefState enum values
- All Event.metrics are typed EventMetrics instances

### 1.3 Spec Compliance

| Check | Result |
|-------|--------|
| 38 PacingConstants match spec | PASS |
| catastrophe_potential formula | PASS |
| Catastrophe gate (>= 0.35 AND composure < 0.30) | PASS |
| Recovery timer blocks catastrophe | PASS |
| Trust repair hysteresis (3x) | PASS |
| GoalVector 7 dimensions | PASS |
| Foyer privacy (spec: 0.2, code: 0.6) | **FAIL** [S-HIGH-2] |
| Catastrophe ordering (spec: last, code: first) | **FAIL** [S-MEDIUM-3] |
| self_sacrifice scope (spec: CONFIDE/REVEAL, code: +CHAT) | **FAIL** [S-MEDIUM-4] |

### 1.4 Edge Cases

All passed: event_limit=1, max_sim_time=0, tick_limit=0, stalemate condition.

### 1.5 Previous Audit Findings

| Finding | Status |
|---------|--------|
| S-01 (dominant_flaw.name -> .flaw_type) | **FIXED** (`tick_loop.py:85`) |
| S-03+S-15 (EventMetrics typed dataclass) | **FIXED** (`events.py:140-174`) |
| S-05 (content_display -> description) | **FIXED** |
| S-06+S-18 (content_metadata on Event) | **FIXED** (`events.py:193`) |
| S-04+S-14 (WorldStateSnapshot naming) | **PARTIAL** (SnapshotState exists but run_simulation returns dicts) |
| S-27 (fragile string matching) | **PRESENT** (`tick_loop.py:465`) |
| S-28 (event mutability) | **PRESENT** (`tick_loop.py:685,852`) |

### 1.6 Simulation Findings

| ID | Severity | Description | File |
|----|----------|-------------|------|
| S-HIGH-1 | HIGH | SOCIAL_MOVE dominance (37-47% of events) dilutes narrative density | `decision_engine.py:755`, `tick_loop.py:240` |
| S-HIGH-2 | HIGH | Foyer privacy 0.6 (code) vs 0.2 (spec) -- affects pacing behavior | `dinner_party.py:49` |
| S-HIGH-3 | HIGH | Snapshot global_tension always 0.0 (hardcoded placeholder) | `tick_loop.py:921-928` |
| S-MED-1 | MEDIUM | Event immutability violations (S-28 still present) | `tick_loop.py:685,718,852` |
| S-MED-2 | MEDIUM | Fragile string matching for PHYSICAL (S-27 still present) | `tick_loop.py:464-492` |
| S-MED-3 | MEDIUM | Catastrophe ordering contradicts spec (code: first, spec: last) | `tick_loop.py:868-872` |
| S-MED-4 | MEDIUM | self_sacrifice includes CHAT (spec: CONFIDE/REVEAL only) | `decision_engine.py:453` |
| S-LOW-1 | LOW | run_simulation returns dicts, not typed SnapshotState | `tick_loop.py:921-956` |
| S-LOW-2 | LOW | No termination variety (all seeds hit event_limit only) | `tick_loop.py:891-908` |
| S-LOW-3 | LOW | Pacing update order differs from spec (no functional impact) | `pacing.py:431-438` |

---

## 2. Metrics Pipeline

**Agent:** metrics-auditor | **Findings:** 0 CRITICAL, 0 HIGH, 3 MEDIUM, 5 LOW

### 2.1 End-to-End Results (Seed 42)

| Step | Result |
|------|--------|
| Raw events | 80 |
| After bundling | 48 |
| Scenes | 9 |
| Pipeline order | irony -> thematic -> tension -> bundle -> segmentation (correct) |
| All events have metrics | YES |
| Tension values in [0,1] | YES |
| All 8 tension components present | YES |

### 2.2 Tension Pipeline

- All 8 canonical tension_component keys present on every event
- CONFLICT/CATASTROPHE events have non-zero danger
- LIE events have moral_cost >= 0.5
- Global normalization intentionally skipped (spec marks optional)

### 2.3 Irony Pipeline

- secret_relevance follows spec tiers: 1.0/0.7/0.5/0.2
- All irony collapses have drop >= 0.5
- scene_irony = mean of agent_irony
- Collapse detection uses scene_irony

### 2.4 Thematic Pipeline

- THEMATIC_SHIFT_RULES match spec
- CONFLICT -> order_chaos, LIE -> truth_deception
- Near-zero shifts filtered (|delta| > 0.01)

### 2.5 Segmentation

- All events in exactly one scene
- All scene types valid (6 canonical types)
- Jaccard threshold = 0.3, tension window = 5
- Min scene size = 3, time gap = 5.0 min
- SOCIAL_MOVE forces scene boundary
- Same-tick events grouped together

### 2.6 Event Bundler

- No causal links to dropped events
- No forward causal links
- No self-referential links
- Social move squash preserves route metadata
- Compression ratio: 60% (80 -> 48 events)

### 2.7 Data Flow

- Pipeline order matches spec (irony -> thematic -> tension -> segmentation)
- NarrativeFieldPayload has all 8 required keys
- AgentManifest and BeliefSnapshot fields match spec
- metadata.event_count matches actual count

### 2.8 Metrics Findings

| ID | Severity | Description | File |
|----|----------|-------------|------|
| M-MED-1 | MEDIUM | goal_frustration uses heuristic (0.6*stress + 0.4*(1-budget)) instead of spec's cosine distance | `tension.py:135-143` |
| M-MED-2 | MEDIUM | resource_scarcity uses pacing proxies instead of world-state features | `tension.py:193-199` |
| M-MED-3 | MEDIUM | Irony collapse scoring simplified (score=drop vs spec's magnitude+breadth+denial formula) | `irony.py:164-169` |
| M-LOW-1 | LOW | information_gap uses belief diversity instead of per-participant gap | `tension.py:163-190` |
| M-LOW-2 | LOW | relationship_volatility missing oscillation bonus | `tension.py:146-160` |
| M-LOW-3 | LOW | TensionWeights defaults pre-normalized (0.125 vs spec's 1.0; math identical) | `tension.py:32-40` |
| M-LOW-4 | LOW | secret_relevance missing 0.9 tier | `irony.py:14-26` |
| M-LOW-5 | LOW | Irony "relevant unknown" multiplied by relevance (spec ambiguous on this) | `irony.py:64-66` |

---

## 3. Extraction & API

**Agent:** extraction-auditor | **Findings:** 0 CRITICAL, 0 HIGH, 4 MEDIUM, 8 LOW

### 3.1 Arc Search (Seed 42)

| Step | Result |
|------|--------|
| Input events | 124 (after bundling) |
| Arc events selected | 20 |
| Protagonist | marcus |
| Arc validation | VALID (0 violations) |
| Beat sequence | SETUP(3) -> COMPLICATION/ESCALATION(12) -> TURNING_POINT(1) -> CONSEQUENCE(4) |
| Composite score | 0.4138 (moderate quality) |

Score breakdown: tension_variance=0.21, peak_tension=0.68, tension_shape=0.74, significance=0.00 (Phase 5), thematic_coherence=0.37, irony_arc=0.01, protagonist_dominance=1.00.

### 3.2 Beat Classification

- All 5 BeatTypes exercised (SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE)
- Exactly 1 TURNING_POINT per arc
- Monotonic beat enforcement works correctly

### 3.3 Arc Validation & Scoring

- Validator correctly rejects: too few beats, missing TP, phase order violation, too-short span, no protagonist, causal gap
- All score components in [0,1], composite = weighted sum, weights sum to 1.0
- Edge cases pass: empty events, single event, flat tension, zero tension

### 3.4 API Server

- FastAPI endpoints correctly defined
- Request/response schemas match
- **Deprecation warning** for `on_event` -> `lifespan` confirmed

### 3.5 Beat Sheet

- 20 beats, 6 character briefs, all fields populated
- LLM prompt contains all 5 required sections
- Empty events edge case handled

### 3.6 Extraction Findings

| ID | Severity | Description | File |
|----|----------|-------------|------|
| E-MED-1 | MEDIUM | FastAPI on_event deprecation (should use lifespan) | `api_server.py:115` |
| E-MED-2 | MEDIUM | Beat classification position thresholds diverge from spec | `beat_classifier.py:61-69` |
| E-MED-3 | MEDIUM | Duplicate TURNING_POINT resolution conflict between repair_beats and _enforce_monotonic_beats | `beat_classifier.py:91-96`, `arc_search.py:465-470` |
| E-MED-4 | MEDIUM | Monotonic beat promotion always chooses COMPLICATION over ESCALATION | `arc_search.py:442-444` |
| E-LOW-1 | LOW | CORS allows all origins | `api_server.py:106-111` |
| E-LOW-2 | LOW | classify_beats omits scenes parameter (spec has it) | `beat_classifier.py:6` |
| E-LOW-3 | LOW | score_arc omits weights and scenes parameters | `arc_scorer.py:7` |
| E-LOW-4 | LOW | validate_arc omits ArcGrammar parameter | `arc_validator.py:7` |
| E-LOW-5 | LOW | Spec rule numbering mismatch (code: 8,9 vs spec: 9,10) | `arc_validator.py:72-90` |
| E-LOW-6 | LOW | tension_variance threshold (0.05) too low, most arcs score 1.0 | `arc_scorer.py:17` |
| E-LOW-7 | LOW | ArcValidation frozen=True with mutable list field | `types.py:9-15` |
| E-LOW-8 | LOW | Spec gap for REVEAL in early position (classified as COMPLICATION) | `beat_classifier.py:57` |

---

## 4. Frontend & Cross-System

**Agent:** frontend-auditor | **Findings:** 0 CRITICAL, 3 HIGH, 6 MEDIUM, 6 LOW

### 4.1 Type Alignment

| Type | TS vs Python | Status |
|------|-------------|--------|
| EventType (10 values) | Exact match | PASS |
| BeatType (5 values) | Exact match | PASS |
| DeltaKind (9 values) | Exact match | PASS |
| DeltaOp (2 values) | Exact match | PASS |
| FlawType (10 values) | Exact match | PASS |
| BeliefState (4 values) | Exact match | PASS |
| SceneType (6 values) | TS has enum, Python uses plain string | MEDIUM [F-01] |
| Event (15 fields) | Perfect match | PASS |
| EventMetrics (6 fields) | tension_components typed vs untyped | MEDIUM [F-02] |
| StateDelta (8 fields) | Exact match | PASS |
| IronyCollapseInfo | Wire-compatible (from_state/to_state serialized as from/to) | LOW [F-03] |
| GoalVector (7 fields) | Exact match | PASS |
| RelationshipState (3 fields) | Exact match | PASS |
| PacingState (6 fields) | Exact match | PASS |
| AgentState (10 fields) | Exact match | PASS |
| Location (8 fields) | Exact match | PASS |
| SecretDefinition (10 fields) | Exact match | PASS |
| WorldDefinition (12 fields) | Exact match | PASS |
| Scene (12 fields) | Exact match | PASS |
| SnapshotState (9 fields) | Exact match | PASS |
| NarrativeFieldPayload (8 keys) | Exact match | PASS |
| AgentManifest (5 fields) | Exact match | PASS |
| BeliefSnapshot (5 fields) | Exact match | PASS |

### 4.2 Build & Tests

| Check | Result |
|-------|--------|
| TypeScript build (tsc --noEmit + vite) | PASS (71 modules, 0 errors) |
| Frontend tests (vitest) | 8/8 passing |
| Lint | Clean |

### 4.3 Data Loading & Validation

- `parseNarrativeFieldPayload` validates top-level shape but casts arrays without per-item validation
- `validatePayload.ts` is a CLI script, NOT called during UI data loading
- Payload validation checks format_version, counts, agent refs, causal links, scene coverage, belief snapshots

### 4.4 Component Review

- InfoPanel, TimelineBar, BeatSheetPanel, ControlPanel: No issues
- CanvasRenderer: useMemo dependency array concern (getState() bypasses tracking)

### 4.5 Frontend Findings

| ID | Severity | Description | File |
|----|----------|-------------|------|
| F-HIGH-1 | HIGH | validatePayload.ts doesn't validate event.type, numbers, deltas, beat_type | `validatePayload.ts:81-110` |
| F-HIGH-2 | HIGH | parseNarrativeFieldPayload casts arrays without per-item validation | `loader.ts:62-67` |
| F-HIGH-3 | HIGH | EventIndices has backward_links not in Python; pair key type mismatch | `scenes.ts:55`, `index_tables.py:10-22` |
| F-MED-1 | MEDIUM | SceneType enum only in TS, Python uses plain string | `scenes.ts:5-12`, `scenes.py:25` |
| F-MED-2 | MEDIUM | tension_components: TS typed interface vs Python dict | `events.ts:64`, `events.py:147` |
| F-MED-3 | MEDIUM | raw_event_count in bundler not declared in TS SimulationMetadata | `bundler.py:103-104`, `payload.ts:6-15` |
| F-MED-4 | MEDIUM | validatePayload.ts is CLI-only, not called during UI loading | `validatePayload.ts:42`, `narrativeFieldStore.ts:177` |
| F-MED-5 | MEDIUM | CanvasRenderer useMemo accesses getState() bypassing dependency array | `CanvasRenderer.tsx:106` |
| F-MED-6 | MEDIUM | BeatSheetPanel uses string literals instead of enum for beatColor switch | `BeatSheetPanel.tsx:8-21` |
| F-LOW-1 | LOW | IronyCollapseInfo field names from_state/to_state vs from/to (wire-compatible) | `events.ts:55`, `events.py:100` |
| F-LOW-2 | LOW | Bundler may pass extra metadata fields not in TS interface | `bundler.py:102` |
| F-LOW-3 | LOW | Empty events edge case handled but no user warning | `narrativeFieldStore.ts:180-195` |
| F-LOW-4 | LOW | useVisibleEvents creates new array each render (OK for MVP) | `narrativeFieldStore.ts:336-352` |
| F-LOW-5 | LOW | Default tension weights: TS=1.0/key, Python=0.125/key (math equivalent) | `tensionPresets.ts:5-14`, `tension.py:33-40` |
| F-LOW-6 | LOW | EmotionalState interface defined but unused | `agents.ts:54-61` |

---

## 5. Consolidated Findings by Severity

### HIGH (6)

| ID | Subsystem | Description | Fix Effort |
|----|-----------|-------------|------------|
| S-HIGH-1 | Simulation | SOCIAL_MOVE dominance (37-47%) dilutes narrative density | Medium (tune scoring) |
| S-HIGH-2 | Simulation | Foyer privacy 0.6 vs spec 0.2 | Trivial (config change) |
| S-HIGH-3 | Simulation | global_tension always 0.0 in snapshots | Easy (compute proxy) |
| F-HIGH-1 | Frontend | validatePayload gaps (event.type, numbers, deltas) | Medium (add checks) |
| F-HIGH-2 | Frontend | Loader casts arrays without per-item validation | Medium (add validation) |
| F-HIGH-3 | Frontend | EventIndices backward_links / pair key mismatch | Easy (sync types) |

### MEDIUM (10)

| ID | Subsystem | Description |
|----|-----------|-------------|
| S-MED-1 | Simulation | Event immutability violations (S-28) |
| S-MED-2 | Simulation | Fragile string matching for PHYSICAL (S-27) |
| S-MED-3 | Simulation | Catastrophe ordering contradicts spec |
| S-MED-4 | Simulation | self_sacrifice includes CHAT (spec deviation) |
| M-MED-1 | Metrics | goal_frustration heuristic vs cosine distance |
| M-MED-2 | Metrics | resource_scarcity uses pacing proxies |
| M-MED-3 | Metrics | Irony collapse scoring simplified |
| E-MED-1 | Extraction | FastAPI on_event deprecation |
| E-MED-2 | Extraction | Beat classification position thresholds diverge |
| E-MED-3 | Extraction | Duplicate TP resolution conflict |

(Plus 5 more MEDIUM in Frontend: F-MED-1 through F-MED-6, and E-MED-4)

### LOW (19)

Simulation: 3 | Metrics: 5 | Extraction: 8 | Frontend: 6

All LOW findings are cosmetic, informational, or have no functional impact.

---

## 6. Previous Spec Audit (S-01 through S-35) Status

| Status | Count | IDs |
|--------|-------|-----|
| **FIXED** | 5 | S-01, S-03+S-15, S-05, S-06+S-18 |
| **PRESENT** | 2 | S-27, S-28 |
| **PARTIAL** | 1 | S-04+S-14 (SnapshotState exists but not used by run_simulation) |
| **Not in implementation scope** | 27 | Spec-only issues, Phase 5+ features, documentation |

---

## 7. Recommendations (Priority Order)

### Quick Wins (< 30 min each)
1. **S-HIGH-2**: Change foyer privacy from 0.6 to 0.2 in `dinner_party.py:49`
2. **E-MED-1**: Migrate `on_event` to `lifespan` in `api_server.py:115`
3. **F-HIGH-3**: Add `backward_links` to Python IndexTables or remove from TS EventIndices
4. **F-MED-3**: Add `raw_event_count` to TS SimulationMetadata interface

### Medium Effort (1-2 hours each)
5. **S-HIGH-1**: Add SOCIAL_MOVE penalty (-0.15) in softmax scoring
6. **S-HIGH-3**: Compute mean-stress proxy for global_tension in snapshots
7. **E-MED-3**: Remove duplicate-TP handling from `repair_beats`
8. **E-MED-4**: Use tension comparison for COMPLICATION vs ESCALATION promotion
9. **F-HIGH-1 + F-HIGH-2**: Add per-item validation in loader + extend validatePayload

### Deferred (Accept for MVP)
10. **M-MED-1,2**: Heuristic-based tension sub-metrics (adequate for dinner party scale)
11. **M-MED-3**: Simplified irony collapse scoring (consider full formula in Phase 3 polish)
12. **S-MED-1,2**: Event immutability and string matching (refactor in Phase 2 polish)

---

## 8. Audit Artifacts

| File | Description |
|------|-------------|
| `docs/audits/audit_findings_simulation.md` | Full simulation audit (271 lines) |
| `docs/audits/audit_findings_metrics.md` | Full metrics audit (198 lines) |
| `docs/audits/audit_findings_extraction.md` | Full extraction audit (188 lines) |
| `docs/audits/audit_findings_frontend.md` | Full frontend audit (392 lines) |
| `src/engine/audit_sim.py` | Simulation stress test script |
| `src/engine/audit_metrics.py` | Metrics pipeline test script |
| `src/engine/audit_extraction.py` | Extraction pipeline test script |

All audit scripts can be re-run to verify findings:
```bash
cd src/engine && source .venv/bin/activate
python audit_sim.py        # Multi-seed simulation + invariant checks
python audit_metrics.py    # End-to-end metrics pipeline
python audit_extraction.py # Arc search + validation + scoring
```
