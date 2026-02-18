# NarrativeField Second-Pass Audit Report

> **Date (local):** 2026-02-10 21:56 EST
> **Date (UTC):** 2026-02-11 02:56 UTC
> **Branch:** codex/threads-topology-polish
> **Commit:** ecb6463
> **Environment:** Python 3.14.2, Node v25.4.0, macOS Darwin 25.2.0
> **First audit cross-ref:** `docs/audits/AUDIT_REPORT.md` (2026-02-10, same commit)
> **Auditors (multi-agent):** sim-stress, contract-tracer, viz-build, extract-story, lead
> **Method:** adversarial sim runs (14 seeds, 4 time-scale values) + field-by-field contract tracing (5 fields) + npm/pip/ruff audits + extraction/storyteller script audits + golden story prose analysis + portfolio readiness review
> **Scope:** Unhappy paths, boundary conditions, and specific risks flagged by first audit

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Overall Health | GOOD (unchanged from first audit) |
| Engine Tests | 80/80 passing |
| Visualization Tests | 59/59 passing (10 test files) |
| Determinism | CONFIRMED across 50 seeds, 2 PYTHONHASHSEED values, truncation-stable |
| Total Findings | **59** (1 CRITICAL, 12 HIGH, 25 MEDIUM, 21 LOW) |
| Positive Confirmations | 3 (determinism, exception hygiene, non-determinism isolation) |

**What changed since first audit:** Nothing — same commit. This second pass attacks unhappy paths, boundary conditions, and coupling risks the first audit flagged but didn't fully resolve.

**Updated risk assessment:** The system's correctness core remains strong. The most impactful new discoveries are:
1. **40% of the world is dead space** (foyer + bathroom never visited, kitchen barely used) — limits narrative variety
2. **`significance` field is never computed** but downstream code depends on it — silent contract violation affecting bundler filtering
3. **Creative model is Haiku 4.5** — the cheapest Claude model as the prose engine for a literary product
4. **Python deps completely unpinned** — builds are non-reproducible

No CRITICAL data corruption or crash risks were found in the simulation core.

---

## Severity Definitions

- **CRITICAL:** Architectural mismatch or quality ceiling that undermines the product's core value proposition
- **HIGH:** Incorrect output, silent contract violation, or significant gap in common scenarios
- **MEDIUM:** Spec drift, tech debt, or quality issue affecting some paths; moderate security/dependency concerns
- **LOW:** Cosmetic, minor tech debt, intentional simplification, or narrow edge case

---

## Section 1: Adversarial Input Results

| Input | Outcome | Finding |
|-------|---------|---------|
| `--time-scale 0` | Empty event log, unhelpful error message | FU-ADV-MED-01 |
| `--time-scale -1` | Empty event log, same unhelpful error | FU-ADV-MED-01 |
| `--time-scale 0.001` | Full sim, identical events to time_scale=1 | FU-ADV-MED-02 |
| `--time-scale 1000` | Full sim, identical events to time_scale=1 | FU-ADV-MED-02 |
| `--seed 0` | Normal run, no crash | Pass |
| `--seed 999` | Normal run, no crash | Pass |
| `--seed 2147483647` | Normal run, no crash | Pass |
| 5 random large seeds | Normal runs, no crashes | Pass |
| 0 events to bundler | Accepted without warning | FU-ADV-LOW-06 |
| Duplicate event IDs | Silently overwritten in dict | FU-ADV-LOW-05 |
| Broken causal_links | Accepted without warning | FU-ADV-LOW-05 |

### FU-ADV-MED-01: time_scale=0 and negative values produce unhelpful error
**Severity:** MEDIUM
**Files:** `src/engine/narrativefield/simulation/run.py:105-106,117-118`, `tick_loop.py:921-923`
**Extends:** New finding
**Description:** No input validation on `--time-scale`. When `time_scale=0`, `max_sim_time` computes to `0.0`, triggering immediate termination. When `time_scale=-1`, `max_sim_time=-150.0`, also causing immediate exit. Both produce empty event logs surfaced as `"Event log is empty"` rather than `"time_scale must be positive"`.
**Reproduction:** `python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/out.json --time-scale 0`

### FU-ADV-MED-02: time_scale is cosmetic — does not affect simulation behavior
**Severity:** MEDIUM
**Files:** `src/engine/narrativefield/simulation/tick_loop.py:39-48,884-885`
**Extends:** New finding
**Description:** `time_scale` only controls `sim_time` advancement and time-based termination. It does NOT influence decision-making, pacing, stress/composure, or any simulation mechanics. Runs with `time_scale=0.001` and `time_scale=1000` produce byte-for-byte identical event sequences. The tension metric's `time_pressure` component uses `sim_time / max_sim_time`, so time_scale cancels out. If time_scale is meant to slow/accelerate decision dynamics, it currently doesn't.

### FU-ADV-HIGH-01: Foyer and bathroom are NEVER selected — 40% of the world is dead space
**Severity:** HIGH
**Files:** `src/engine/narrativefield/simulation/decision_engine.py:297-358,754-773`, `scenarios/dinner_party.py:46-65`
**Extends:** ENG-LOW-03
**Description:** Across 14 tested seeds, foyer and bathroom receive exactly zero events. Location distribution: dining_table 71-86%, balcony 11-29%, kitchen 0-2.5%, foyer 0%, bathroom 0%. Root causes: foyer has low privacy (0.2, strictly dominated by balcony at 0.7); effective privacy cap at 0.35; social inertia from dining_table not offset by foyer's tiny bonus; global SOCIAL_MOVE penalty of -0.20; bathroom is 2 hops from dining_table via foyer (permanently unreachable). The "bathroom escape" and "foyer phone call" spec patterns never emerge.

### FU-ADV-MED-03: WorldDefinition catastrophe_threshold and composure_gate are dead config
**Severity:** MEDIUM
**Files:** `src/engine/narrativefield/schema/world.py:106-107`, `pacing.py:61-62,99-107`
**Extends:** ENG-LOW-02
**Description:** `WorldDefinition` exposes configurable `catastrophe_threshold` and `composure_gate`. The dinner party scenario sets them explicitly. However, `pacing.check_catastrophe()` uses hardcoded `DEFAULT_CONSTANTS` — the WorldDefinition values are never read. This is a config API lie: changing these values has no effect.

### FU-ADV-MED-04: Victor triggers 50% of catastrophes; Elena/Diana never trigger
**Severity:** MEDIUM
**Files:** `scenarios/dinner_party.py:415-468`, `pacing.py:91-107`
**Extends:** New finding
**Description:** Across seeds 1-10: Victor 17/34 catastrophes (50%), Marcus 10 (29%), Lydia 6 (18%), Thorne 1 (3%), Elena 0, Diana 0. Victor's obsession flaw and truth_seeking=0.9 drive confrontation; Elena's cowardice flaw keeps her from commitment-building acts. 33% of characters never reach catastrophe, limiting narrative variety.

---

## Section 2: Determinism Deep Dive

### FU-DET-POS-01: Determinism CONFIRMED across seeds and PYTHONHASHSEED values (Positive)
**Files:** `scripts/science_harness.py`, simulation code
**Description:** Identical seeds produce byte-identical event streams across `PYTHONHASHSEED=12345` vs `67890`. SHA256 hashes match. Truncation stability confirmed: first 50 events of limit=50 are field-identical to first 50 events of limit=200. The codebase uses `sorted()` for all agent iterations.

### FU-DET-POS-02: Non-determinism sources cleanly isolated from simulation (Positive)
**Files:** Various
**Description:** All `datetime.now`, `uuid4` calls are outside simulation-critical paths (metadata timestamps after sim, storyteller, API server, test scripts). Simulation exclusively uses seeded `Random` instance.

### FU-DET-LOW-01: Partial ticks invisible to downstream consumers
**Severity:** LOW
**Files:** `src/engine/narrativefield/simulation/tick_loop.py:956-978`
**Extends:** DET-LOW-01
**Description:** When `event_limit` is hit mid-tick, witness events are dropped first (reasonable). However, no metadata flag indicates truncation occurred. The output JSON's last tick may be partial with no way for downstream consumers to detect this.

---

## Section 3: Bundling/Segmentation Coupling Evidence

### FU-BUN-HIGH-01: Segmentation uses SOCIAL_MOVE source location instead of destination
**Severity:** HIGH
**Files:** `src/engine/narrativefield/metrics/segmentation.py:52-57,224-226`, `integration/event_bundler.py:226-227`
**Extends:** CON-MED-02
**Description:** SOCIAL_MOVE events store origin in `location_id` and destination in `content_metadata.destination`. Segmentation's `location_break()` compares `prev.location_id` to `curr.location_id`. Surviving SOCIAL_MOVEs get grouped into **source** location scenes, not destination scenes. A move from dining_table to balcony gets grouped into the dining_table scene.
**Evidence:** All 11 surviving SOCIAL_MOVE events in seed 1 show `location_id=dining_table` even when destination is balcony/kitchen.

### FU-BUN-MED-01: SOCIAL_MOVE boundary logic designed for dense pre-bundle stream but runs post-bundle
**Severity:** MEDIUM
**Files:** `segmentation.py:224-226`, `pipeline.py:126-127`
**Extends:** CON-MED-01
**Description:** Pipeline order: metrics -> bundle -> segment. Post-bundle, 68-81% of SOCIAL_MOVEs are removed (attached to host events). The surviving moves are the most isolated, least connected. Forcing scene boundaries on these creates fragment scenes. The min_scene_size merge step partially compensates.
**Evidence:** Seed 1: 46 SM pre-bundle -> 11 post-bundle (76% removed). Seed 2: 57 -> 11 (81%). Seed 3: 53 -> 17 (68%).

### FU-BUN-MED-02: Consecutive SOCIAL_MOVEs grouped by type alone, not by destination
**Severity:** MEDIUM
**Files:** `segmentation.py:224-226`
**Extends:** CON-MED-02
**Description:** Segmentation only checks `curr.type == EventType.SOCIAL_MOVE`, not destination. Two consecutive moves to different locations (kitchen vs balcony) are placed in the same scene.

### FU-BUN-LOW-01: Social_move squashing is dead code (0 squashes across all seeds)
**Severity:** LOW
**Files:** `integration/event_bundler.py:95-190`
**Description:** The squash step (consecutive same-agent moves within 2 minutes) reports `moves_squashed=0` for all tested seeds. The sim never generates consecutive same-agent SOCIAL_MOVEs within the time window. All reduction comes from the attach step.

---

## Section 4: Contract Trace Results

Five fields traced end-to-end from producer through intermediate transformations to consumer.

| Field | Producer | Consumer | Status |
|-------|----------|----------|--------|
| tension | tension.py → bundler → loader.ts → tensionComputer.ts | TensionTerrainLayer | Clean — no mismatch |
| causal_links | tick_loop.py → event_bundler.py → loader.ts → causalIndex.ts | CausalIndex | Clean — 0 dangling refs |
| scene_id | segmentation.py → bundler → loader.ts | TimelineBar | String, index-based, stable per config |
| irony_collapse | irony.py → bundler → loader.ts | **No consumer** | Dead weight |
| belief_snapshots | bundler.py → loader.ts → store | **No consumer** | Dead weight |
| significance | **Never computed** | bundler, loader, InfoPanel, arc_search | Silent contract violation |

### FU-CON-HIGH-01: significance field is never populated — downstream consumers get perpetual zero
**Severity:** HIGH
**Files:** `metrics/pipeline.py:120-121`, `schema/events.py:145`, `integration/event_bundler.py:341-342`, `extraction/arc_search.py:114`
**Extends:** S-03/S-15 (EventMetrics incomplete)
**Description:** No pipeline step writes `event.metrics.significance`. It defaults to 0.0. The bundler's `observe_significance_threshold` (0.3) check always fails (0.0 < 0.3), causing ALL low-delta observe events to be attached regardless of importance. The InfoPanel shows `significance: 0.000`. Arc search's turning point detection always gets 0.0.
**Evidence:** `grep -r "\.significance\s*=" src/engine/narrativefield/metrics/` returns zero results.

### FU-CON-HIGH-02: Pipeline comment falsely claims significance is available for filtering
**Severity:** HIGH
**Files:** `metrics/pipeline.py:120-121`
**Extends:** FU-CON-HIGH-01
**Description:** The pipeline comment says "importance/significance are available for filtering" — but significance is never computed. This misleads developers into trusting a zero-value field.

### FU-CON-HIGH-03: Loader does not validate irony_collapse field shape
**Severity:** HIGH
**Files:** `src/visualization/src/data/loader.ts:199-227`, `types/events.ts:52-57`
**Extends:** New finding
**Description:** The loader validates tension, irony, significance, and tension_components metrics but NOT irony_collapse. A payload with `irony_collapse: { detected: "yes", drop: "not a number" }` would pass validation silently.

### FU-CON-MED-01: irony_collapse produced by Python but zero viz components consume it
**Severity:** MEDIUM
**Files:** `metrics/irony.py:160-172`, `visualization/src/types/events.ts:52-57`
**Extends:** New finding
**Description:** irony_collapse is computed, serialized, and typed in TypeScript, but no component reads it. The only consumer is `segmentation.py:86-88` (Python-side). In the viz, it's dead weight.

### FU-CON-MED-02: belief_snapshots loaded into store but never consumed by any component
**Severity:** MEDIUM
**Files:** `integration/bundler.py:51-82`, `store/narrativeFieldStore.ts:249`
**Extends:** New finding
**Description:** `beliefSnapshots` is populated in the Zustand store but no component, selector, or canvas layer reads it. The planned belief matrix heatmap is not yet implemented.

### FU-CON-MED-03: SceneType enum/string mismatch between Python and TypeScript
**Severity:** MEDIUM
**Files:** `schema/scenes.py:25`, `types/scenes.ts:5-12,28`
**Extends:** New finding
**Description:** Python produces scene_type as a plain string. TypeScript defines a SceneType enum but declares the field as `SceneType | string`. The loader validates it's a string but not that it's a valid enum value. Adding a new scene type in Python would silently pass through without TypeScript compile-time checks.

### FU-CON-LOW-01: scene_id is index-based — not stable across bundler config changes
**Severity:** LOW
**Files:** `segmentation.py:131`
**Description:** Scene IDs are sequential strings ("scene_000", "scene_001"). Deterministic per input, but change if bundler config changes. Acceptable for MVP.

### FU-CON-LOW-02: MetricsPipelineOutput.scenes typed as list[Any] instead of list[Scene]
**Severity:** LOW
**Files:** `metrics/pipeline.py:84`
**Description:** `scenes: list[Any]  # Scene` — the comment reveals the intended type, but `Any` weakens type safety.

### FU-CON-LOW-03: thematic_shift keys are implicitly shared vocabulary, not validated
**Severity:** LOW
**Files:** `metrics/thematic.py:6-12`, `types/events.ts:63`
**Description:** Python produces dynamic axis keys ("order_chaos", etc.). TypeScript declares `Record<string, number>` which accepts anything. The axis vocabulary is implicitly shared.

---

## Section 5: Visualization Stress Results

### FU-VIZ-HIGH-01: TensionTerrainLayer has 16 `as any` casts bypassing OffscreenCanvas type safety
**Severity:** HIGH
**Files:** `src/visualization/src/canvas/layers/TensionTerrainLayer.ts` (lines 499, 911, 919, 929-931, 938, 946, 957, 965, 975, 1147, 1150, 1196)
**Extends:** First audit code quality observations
**Description:** 16 `as any` casts in production code, all for `OffscreenCanvas | HTMLCanvasElement` to `drawImage()` and Canvas `filter`. The cast at line 1150 is particularly risky — `coastGlowBitmap` could be `null` and `as any` silences the null check.

### FU-VIZ-HIGH-02: ESLint fails with 18 errors — lint gate has no CI enforcement
**Severity:** HIGH
**Files:** `TensionTerrainLayer.ts`, `CanvasRenderer.tsx`
**Extends:** First audit observations
**Description:** `npm run lint` exits with 19 problems (18 errors, 1 warning). Includes 14 `no-explicit-any`, 2 `no-constant-condition`, 1 unused variable, 1 unused function. No CI or pre-commit hook prevents merging.

### FU-VIZ-MED-01: TensionTerrainLayer O(E * G) computation — may degrade at 200+ events
**Severity:** MEDIUM
**Files:** `TensionTerrainLayer.ts:407-432,522-597,658-719`
**Extends:** FE-MED-03
**Description:** Terrain field: ~O(E * sigma^2/cellPx^2). At 200 events with default settings: ~1M iterations. Contour tracing: O(cols * rows * levels) = ~1.4M iterations. `buildPolylines` uses `line.unshift()` which is O(n) per call, making backward extension O(n^2). Throttle (100ms) and caching mitigate, but resize/zoom triggers full recomputation.

### FU-VIZ-MED-02: TimelineBar assumes events are pre-sorted by sim_time
**Severity:** MEDIUM
**Files:** `src/visualization/src/components/TimelineBar.tsx:12-13`
**Extends:** FE-MED-01
**Description:** Reads `events[0].sim_time` as min and `events[events.length - 1].sim_time` as max. The store does not sort events. If payload provides out-of-order events, domain is wrong.

### FU-VIZ-MED-03: threadLayout ignores agent initial_location from manifest
**Severity:** MEDIUM
**Files:** `src/visualization/src/layout/threadLayout.ts:292-293`
**Extends:** FE-MED-02
**Description:** Hardcodes `initialLocations.set(a, 'dining_table')` for all agents. The `ThreadLayoutInput` interface accepts agents as strings, not manifest objects, so there is no pathway for `initial_location` to reach the layout.

### FU-VIZ-MED-04: loader.ts uses 6 `as unknown as` casts on validated-but-untyped data
**Severity:** MEDIUM
**Files:** `src/visualization/src/data/loader.ts:364-369`
**Extends:** FE-MED-03
**Description:** After field-by-field validation, arrays are cast via `as unknown as`. TypeScript trusts the runtime validation is complete. If validation misses a new required field, cast silently passes `undefined` through. Extra JSON fields pass through silently.

### FU-VIZ-MED-05: CanvasRenderer useEffect has missing React Hook dependencies
**Severity:** MEDIUM
**Files:** `src/visualization/src/components/CanvasRenderer.tsx:441`
**Description:** Draw effect depends on `viewport.y` and `viewport.height` but dependency array only includes `viewport.scale`. Vertical panning won't trigger redraw.

### FU-VIZ-MED-06: Zustand store useVisibleEvents creates new array on every call
**Severity:** MEDIUM
**Files:** `src/visualization/src/store/narrativeFieldStore.ts:520-536`
**Description:** `useVisibleEvents` runs a filter producing a new array reference on every store change that affects its dependencies. This causes a cascade: pan -> setViewport -> new array -> baseModel recomputes -> drawModel recomputes -> draw fires. With 200 events and frequent panning, this creates unnecessary re-renders.

### FU-VIZ-LOW-01: Unused `domainSpan` variable in TensionTerrainLayer draw()
**Severity:** LOW
**Files:** `TensionTerrainLayer.ts:1016`
**Description:** `domainSpan` computed but never read. Dead code from refactor.

### FU-VIZ-LOW-02: Empty-state handling adequate but not systematically tested
**Severity:** LOW
**Files:** `CanvasRenderer.tsx:151`, `TensionTerrainLayer.ts`, `threadLayout.ts:64`
**Description:** Guards exist for 0 events, empty scenes, empty belief_snapshots. But edge cases (empty scenes + non-empty events) are not explicitly tested.

---

## Section 6: Extraction Semantic Correctness

### FU-EXT-HIGH-01: Duplicate TURNING_POINT resolution inconsistency between two passes
**Severity:** HIGH
**Files:** `extraction/beat_classifier.py:91-93`, `extraction/arc_search.py:472-477`
**Extends:** Extraction audit spec_divergence
**Description:** `repair_beats()` has a comment saying it defers TP resolution to `_enforce_monotonic_beats`. The latter keeps the highest-tension TP and downgrades others to ESCALATION. The dual-resolution logic is confusing and risks divergent behavior if call order changes. The comment in `repair_beats` implies deferred resolution while the actual dedup lives only in `_enforce_monotonic_beats`.

### FU-EXT-HIGH-02: _enforce_monotonic_beats phase promotion is fragile and hard to audit
**Severity:** HIGH
**Files:** `extraction/arc_search.py:440-452`
**Extends:** Extraction audit spec_divergence
**Description:** Nested loop + conditional distinguishes COMPLICATION vs ESCALATION at phase 1 by tension direction, but the default path at other phase levels always picks `candidates[0]`. The logic is correct but fragile and hard to follow. Refactoring into a dedicated `_promote_to_phase()` function is needed.

### FU-EXT-MED-01: Beat position thresholds diverge from spec (0.25/0.70 vs peak-index-based)
**Severity:** MEDIUM
**Files:** `extraction/beat_classifier.py:61-69`
**Extends:** Extraction audit spec_divergence
**Description:** Spec defines position boundaries at 0.25 and 0.70. Implementation uses `event_index < peak_tension_idx` for events past 0.25, ignoring the 0.70 threshold. Events at position 0.80 before the peak get classified as development beats rather than escalation-only.

### FU-EXT-MED-02: tension_variance normalization threshold too low (0.10) — nearly all arcs score 1.0
**Severity:** MEDIUM
**Files:** `extraction/arc_scorer.py:17`
**Extends:** Extraction audit
**Description:** `tension_variance_score = min(variance / 0.10, 1.0)`. The 0.10 threshold is so low that nearly any arc with non-trivial variation saturates at 1.0 (audit shows variance=0.1224, already at max). This component provides no discriminative power at 20% weight in the composite score.

### FU-EXT-LOW-01: ArcValidation is frozen=True with mutable list[str] violations field
**Severity:** LOW
**Files:** `extraction/types.py:9-15`
**Description:** `frozen=True` prevents reassignment but not mutation of the list contents. `violations.append(...)` would not error. Safe in practice but semantically inconsistent. Use `tuple[str, ...]` for true immutability.

### FU-EXT-LOW-02: classify_beats omits scene context parameter from spec
**Severity:** LOW
**Files:** `extraction/beat_classifier.py:6`
**Description:** Spec defines `classify_beats(events, scenes)` but implementation uses `classify_beats(events)` with no scene context. Beat classification operates purely on tension values and event types. Acceptable for MVP.

---

## Section 7: Prose Quality Taxonomy + Repetition Guard Audit

### FU-STO-CRIT-01: Creative model is Haiku 4.5 — cheapest Claude model as prose engine
**Severity:** CRITICAL
**Files:** `src/engine/narrativefield/llm/config.py:10`, `llm/gateway.py:29`
**Extends:** ST-MED-01
**Description:** `creative_model = "claude-haiku-4-5-20251001"`. Haiku is designed for classification and simple tasks, not literary prose. For a system whose entire value proposition is high-quality literary prose, using the cheapest model is a quality ceiling. The golden stories are surprisingly good for Haiku but a model upgrade would produce substantially better prose with more distinct character voices and richer interiority. The model is already configurable via PipelineConfig — only the default needs updating.

### FU-STO-HIGH-01: Seed 51 story ends at dramatic peak — no narrative resolution
**Severity:** HIGH
**Files:** `examples/seed_51_story.txt`, `examples/seed_51_meta.json`
**Extends:** NARR-MED-01
**Description:** Seed 51 ends mid-scene with Victor saying "Ask her what she noticed" — a cliffhanger with no resolution. Only 3 scenes (vs 4 for seed 42), 2416 words, final scene only 491 words. Root cause: arc_search selects events clustering around the confrontation peak with insufficient CONSEQUENCE beats. The scene_splitter creates a tiny final chunk that reads as a cliffhanger.

### FU-STO-MED-01: Repetition guard returns 0 removals despite detectable patterns
**Severity:** MEDIUM
**Files:** `storyteller/repetition_guard.py`, `examples/seed_42_meta.json:47`, `seed_51_meta.json:35`
**Extends:** First audit
**Description:** Both golden stories report `repetitions_removed: 0` but contain: structural repetition ("Diana watched X" — 8 times in seed_42), sentence-starter repetition (character name + possessive), metaphor reuse ("the kind of X that Y" — 5+ times), semantic repetition (two "silence had weight" passages). The guard only detects exact N-gram matches. LLM repetition is typically semantic/structural.

### FU-STO-MED-02: Scene continuity relies solely on last_paragraph — no validation
**Severity:** MEDIUM
**Files:** `storyteller/narrator.py:96-101,509-510`
**Description:** Scene-to-scene continuity uses `last_paragraph` text and a rolling summary. No validation that the LLM actually continues coherently. If scene N contradicts scene N-1 (character who left is back), the error propagates through all subsequent scenes. The continuity check runs post-generation only. Acceptable for MVP.

### FU-STO-LOW-01: Golden stories exhibit character voice collapse
**Severity:** LOW
**Files:** `examples/seed_42_story.txt`, `seed_51_story.txt`
**Description:** All six characters share a similar narrative voice. Interior observation style is uniform: "the kind of X that Y", physical micro-tells for every character, simile-as-insight pattern for 4+ characters. Partially a Haiku limitation, partially a prompt issue (single narrative voice instruction). Adding per-character voice notes to lorebook character entries would help.

### FU-STO-LOW-02: LLM gateway Grok model fallback is silent
**Severity:** LOW
**Files:** `llm/gateway.py:272-318`
**Description:** `_generate_grok` falls back from `grok-4-1-fast` to `grok-beta` silently. The fallback model may have different capabilities/pricing. No metadata about the selected model is returned to the caller.

---

## Section 8: Error Handling Inventory

### FU-ERR-POS-01: Zero exception handlers in simulation code (Positive)
**Files:** `src/engine/narrativefield/simulation/`
**Description:** Simulation code does not swallow any exceptions. All errors propagate. Correct behavior for a deterministic simulation. The only handler is in `run.py` for `_get_git_commit()` metadata (appropriate).

### FU-ERR-MED-01: Checkpoint save swallows all exceptions — partial writes can corrupt state
**Severity:** MEDIUM
**Files:** `storyteller/checkpoint.py:33-60`
**Extends:** New finding
**Description:** `save()` writes 3 artifacts non-atomically (state JSON, prose chunks, manifest). A crash between writes leaves inconsistent state. The method catches all exceptions and logs but does not raise, meaning the narrator continues without knowing the checkpoint is broken. No atomic writes (write-to-temp-then-rename).

### FU-ERR-LOW-01: apply_delta PACING case lacks clamping
**Severity:** LOW
**Files:** `tick_loop.py:815-822`
**Description:** Pacing deltas applied via `setattr` without clamping to [0,1]. Currently safe because callers pre-clamp. If future ADD ops lack clamping, values could exceed bounds.

### FU-ERR-LOW-02: Metrics pipeline accepts fully degenerate inputs without warning
**Severity:** LOW
**Files:** `metrics/pipeline.py`
**Description:** Pipeline processes 0 agents, 0 secrets, 0 locations, events referencing non-existent agents/locations — all without error. Produces valid but meaningless output.

---

## Section 9: Dependency & Build Hygiene

### FU-DEP-HIGH-01: Python dependencies completely unpinned
**Severity:** HIGH
**Files:** `src/engine/pyproject.toml:9-14`
**Extends:** SEC-MED-01
**Description:** All four runtime dependencies (`numpy`, `anthropic`, `fastapi`, `uvicorn`) have no version constraints. Dev dependencies (`pytest`, `ruff`) also unpinned. No `requirements.txt` or lock file committed. Builds are non-reproducible. The `anthropic` SDK in particular has frequent breaking changes.

### FU-DEP-MED-01: openai package used but not declared in pyproject.toml
**Severity:** MEDIUM
**Files:** `pyproject.toml`, `llm/gateway.py:17,216`
**Extends:** New finding
**Description:** `openai` is imported in gateway.py for Grok/xAI calls. README quickstart instructs `pip install -e ".[dev]" openai` but `openai` is not in pyproject.toml. `pip install -e .` alone will fail at runtime when the Grok tier is called.

### FU-DEP-MED-02: Python engine has 82 ruff lint errors (47 unused imports)
**Severity:** MEDIUM
**Files:** `src/engine/` (across multiple files)
**Extends:** First audit ruff observations
**Description:** 82 lint errors: 47 unused imports (F401), 19 E402, 5 unused vars (F841), 4 F541, 3 E702, 2 E741, 2 F811. 51 are auto-fixable. Linter configured but not enforced.

### FU-DEP-LOW-01: Node dependencies use caret ranges; 5 moderate dev-only npm audit vulns
**Severity:** LOW
**Files:** `src/visualization/package.json`, `package-lock.json`
**Description:** All Node deps use `^` ranges (standard npm practice). Lock file committed and up-to-date. `npm audit` reports 5 moderate vulns in dev-only `esbuild -> vite -> vitest` chain (GHSA-67mh cross-origin request). `npm audit --omit=dev` reports 0 vulns.

---

## Section 10: Portfolio Readiness Scorecard

| Criterion | Status | Finding |
|-----------|--------|---------|
| README explains what/why/how | **GOOD** — clear value prop, sample output, architecture diagram | — |
| Cold-start < 2 min (viz) | **GOOD** — `npm install && npm run dev` works | — |
| Cold-start (full pipeline) | **FAIR** — requires 2 API keys + undeclared `openai` dep | FU-POR-MED-03 |
| Screenshots/recordings of viz | **MISSING** — zero visual assets | FU-POR-HIGH-01 |
| $0.06/story highlighted | **GOOD** — in README first paragraph and production table | — |
| Architecture readable in 10 min | **GOOD** — `docs/ARCHITECTURE.md` is well-structured | — |
| LICENSE file | **MISSING** | FU-POR-MED-01 |
| Demo path clarity | **FAIR** — README leads with LLM path before zero-dep viz path | FU-POR-LOW-05 |
| Commit history quality | **GOOD** — conventional commits, clear descriptions | — |
| Embarrassing artifacts | **MINOR** — 1 TODO, 1 console.log, hardcoded paths in audit docs | FU-POR-LOW-01, FU-POR-LOW-02 |
| CI/CD pipeline | **MISSING** | FU-POR-MED-02 |

### FU-POR-HIGH-01: No visual assets for a visualization-centric project
**Severity:** HIGH
**Files:** `README.md`
**Description:** The project's core differentiator is an interactive 2D thread topology visualization, but README contains zero screenshots, GIFs, or video recordings. Reviewers will likely never install locally. For a portfolio project, this is a significant gap.

### FU-POR-MED-01: No LICENSE file at repository root
**Severity:** MEDIUM
**Files:** Repository root
**Description:** No LICENSE file. Project is legally "all rights reserved" by default. Even for a private portfolio project, including a license is standard practice.

### FU-POR-MED-02: No CI/CD pipeline — all quality checks are manual
**Severity:** MEDIUM
**Files:** `.github/` (missing)
**Description:** No GitHub Actions. 80 engine tests and 59 viz tests run manually. Even a basic CI workflow would demonstrate engineering maturity.

### FU-POR-MED-03: No .env.example file for required API keys
**Severity:** MEDIUM
**Files:** Repository root
**Description:** README requires `ANTHROPIC_API_KEY` and `XAI_API_KEY` but no `.env.example` with placeholder values. Standard practice for documenting required environment variables.

### FU-POR-LOW-01: Hardcoded local paths in audit documentation
**Severity:** LOW
**Files:** `docs/audits/AUDIT_REPORT.md:77`, `audit_findings_simulation.md:265,267`, `audit_findings_extraction.md:184,186`
**Description:** `$PROJECT_ROOT` appears in 4 audit documents. Leaks developer username and directory structure.

### FU-POR-LOW-02: Version 0.0.0 in both pyproject.toml and package.json
**Severity:** LOW
**Files:** `pyproject.toml:7`, `package.json:4`
**Description:** Undersells project maturity given working end-to-end pipeline with 6 successful story generations.

### FU-POR-LOW-03: No unified task runner (Makefile or similar)
**Severity:** LOW
**Files:** Repository root
**Description:** Common tasks require changing directories. No Makefile, justfile, or workspace config for `test-all` or `lint-all` from root.

### FU-POR-LOW-04: Single TODO in production code
**Severity:** LOW
**Files:** `pacing.py:308`
**Description:** `# TODO: taking-sides detection.` — only TODO/FIXME/HACK/XXX in the entire `src/` tree. Codebase is remarkably clean.

### FU-POR-LOW-05: README quickstart leads with LLM-dependent path before zero-dep viz path
**Severity:** LOW
**Files:** `README.md:72-94`
**Description:** "Generate A Story" (requires 2 API keys) comes before "Explore The Visualization" (zero API keys). Reordering would improve cold-start experience for reviewers.

---

## Appendix A: Commands Run

### sim-stress agent
```bash
# Adversarial time-scale
cd src/engine
python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/audit_ts0.json --time-scale 0
python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/audit_tsn1.json --time-scale -1
python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/audit_ts001.json --time-scale 0.001
python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/audit_ts1000.json --time-scale 1000

# Boundary seeds
python -m narrativefield.simulation.run --scenario dinner_party --seed 0 --output /tmp/audit_s0.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 999 --output /tmp/audit_s999.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 2147483647 --output /tmp/audit_smax.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 12345 --output /tmp/audit_slarge1.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 99999 --output /tmp/audit_slarge2.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 314159 --output /tmp/audit_slarge3.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 777777 --output /tmp/audit_slarge4.json
python -m narrativefield.simulation.run --scenario dinner_party --seed 1000000 --output /tmp/audit_slarge5.json

# Truncation stability
python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/audit_e50.json --event-limit 50
python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output /tmp/audit_e200.json --event-limit 200

# Extended determinism
cd $PROJECT_ROOT
python scripts/science_harness.py --seeds 1-50 --output /tmp/science_50.json --skip-metrics

# Non-determinism scan
rg -n "datetime\.now|time\.time|uuid4|uuid\.uuid4|os\.urandom|random\.random\(\)" src/engine
rg -n "except:|except Exception" src/engine/narrativefield/simulation
```

### contract-tracer agent
```bash
# Bundle/segment coupling quantification
cd src/engine
for seed in 1 2 3 4 5 6 7 8 9 10; do
  python -m narrativefield.simulation.run --scenario dinner_party --seed $seed --output /tmp/audit_bundle_s${seed}.json
done

# Field tracing: code inspection + grep for field usage
rg -n "\.significance" src/engine/narrativefield/metrics/
rg -n "irony_collapse" src/visualization/src/
rg -n "beliefSnapshots" src/visualization/src/
```

### viz-build agent
```bash
cd src/visualization
npm test
npm run build
npm run lint
npm run validate:fake
npm audit
npm audit --omit=dev

cd src/engine
ruff check . --statistics

rg "as any|as unknown|@ts-ignore|@ts-expect-error" src/visualization/src --glob "!*.test.*" -n
rg "console\.(log|warn|error)" src/visualization/src --glob "!*.test.*" -n
```

### extract-story agent
```bash
cd src/engine
python audit_extraction.py
python audit_storyteller.py
rg -n "except:|except Exception" src/engine/narrativefield/extraction src/engine/narrativefield/storyteller
```

### lead (Section 10 + verification)
```bash
git log --oneline -30
rg "TODO|FIXME|HACK|XXX" src/ docs/ scripts/
rg "/path/to/|/home/" src/ docs/ scripts/
rg "sk-|api[_-]key|password|secret" src/
rg "console\.log|print\(" src/visualization/src --glob "!*.test.*"
cd src/engine && pytest -q
cd src/visualization && npm test
```

---

## Appendix B: Finding ID Index

| ID | Severity | Section | Title |
|----|----------|---------|-------|
| FU-ADV-MED-01 | MEDIUM | 1 | time_scale=0 and negative values produce unhelpful error |
| FU-ADV-MED-02 | MEDIUM | 1 | time_scale is cosmetic — does not affect sim behavior |
| FU-ADV-HIGH-01 | HIGH | 1 | Foyer and bathroom never selected — 40% dead space |
| FU-ADV-MED-03 | MEDIUM | 1 | catastrophe_threshold/composure_gate are dead config |
| FU-ADV-MED-04 | MEDIUM | 1 | Victor 50% catastrophe dominance; Elena/Diana never trigger |
| FU-ADV-LOW-05 | LOW | 1 | Metrics pipeline accepts duplicate IDs silently |
| FU-ADV-LOW-06 | LOW | 1 | Bundler accepts degenerate inputs without warning |
| FU-DET-LOW-01 | LOW | 2 | Partial ticks invisible to downstream consumers |
| FU-BUN-HIGH-01 | HIGH | 3 | Segmentation uses SOCIAL_MOVE source instead of destination |
| FU-BUN-MED-01 | MEDIUM | 3 | SOCIAL_MOVE boundary logic runs on post-bundle events |
| FU-BUN-MED-02 | MEDIUM | 3 | Consecutive SOCIAL_MOVEs grouped by type, not destination |
| FU-BUN-LOW-01 | LOW | 3 | Social_move squashing is dead code |
| FU-CON-HIGH-01 | HIGH | 4 | significance never populated — perpetual zero |
| FU-CON-HIGH-02 | HIGH | 4 | Pipeline comment falsely claims significance available |
| FU-CON-HIGH-03 | HIGH | 4 | Loader does not validate irony_collapse shape |
| FU-CON-MED-01 | MEDIUM | 4 | irony_collapse produced but not consumed by viz |
| FU-CON-MED-02 | MEDIUM | 4 | belief_snapshots loaded but never consumed |
| FU-CON-MED-03 | MEDIUM | 4 | SceneType enum/string mismatch Python vs TypeScript |
| FU-CON-LOW-01 | LOW | 4 | scene_id index-based, not stable across config changes |
| FU-CON-LOW-02 | LOW | 4 | pipeline.scenes typed as list[Any] |
| FU-CON-LOW-03 | LOW | 4 | thematic_shift keys implicitly shared, not validated |
| FU-VIZ-HIGH-01 | HIGH | 5 | 16 `as any` casts in TensionTerrainLayer |
| FU-VIZ-HIGH-02 | HIGH | 5 | ESLint fails with 18 errors, no CI enforcement |
| FU-VIZ-MED-01 | MEDIUM | 5 | TensionTerrainLayer O(E*G) may degrade at 200+ events |
| FU-VIZ-MED-02 | MEDIUM | 5 | TimelineBar assumes pre-sorted events |
| FU-VIZ-MED-03 | MEDIUM | 5 | threadLayout ignores agent initial_location |
| FU-VIZ-MED-04 | MEDIUM | 5 | loader.ts 6 `as unknown as` casts |
| FU-VIZ-MED-05 | MEDIUM | 5 | CanvasRenderer useEffect missing dependencies |
| FU-VIZ-MED-06 | MEDIUM | 5 | Zustand useVisibleEvents creates new array every call |
| FU-VIZ-LOW-01 | LOW | 5 | Unused domainSpan variable |
| FU-VIZ-LOW-02 | LOW | 5 | Empty-state handling not systematically tested |
| FU-EXT-HIGH-01 | HIGH | 6 | Duplicate TURNING_POINT resolution inconsistency |
| FU-EXT-HIGH-02 | HIGH | 6 | _enforce_monotonic_beats phase promotion is fragile |
| FU-EXT-MED-01 | MEDIUM | 6 | Beat position thresholds diverge from spec |
| FU-EXT-MED-02 | MEDIUM | 6 | tension_variance threshold too low — no discrimination |
| FU-EXT-LOW-01 | LOW | 6 | ArcValidation frozen with mutable list |
| FU-EXT-LOW-02 | LOW | 6 | classify_beats omits scene context from spec |
| FU-STO-CRIT-01 | CRITICAL | 7 | Creative model is Haiku 4.5 |
| FU-STO-HIGH-01 | HIGH | 7 | Seed 51 ends at dramatic peak — no resolution |
| FU-STO-MED-01 | MEDIUM | 7 | Repetition guard returns 0 removals despite patterns |
| FU-STO-MED-02 | MEDIUM | 7 | Scene continuity relies on last_paragraph only |
| FU-STO-LOW-01 | LOW | 7 | Character voice collapse in golden stories |
| FU-STO-LOW-02 | LOW | 7 | Grok model fallback is silent |
| FU-ERR-MED-01 | MEDIUM | 8 | Checkpoint save swallows exceptions, non-atomic writes |
| FU-ERR-LOW-01 | LOW | 8 | apply_delta PACING lacks clamping |
| FU-ERR-LOW-02 | LOW | 8 | Metrics pipeline accepts degenerate inputs |
| FU-DEP-HIGH-01 | HIGH | 9 | Python dependencies completely unpinned |
| FU-DEP-MED-01 | MEDIUM | 9 | openai not declared in pyproject.toml |
| FU-DEP-MED-02 | MEDIUM | 9 | 82 ruff lint errors (47 unused imports) |
| FU-DEP-LOW-01 | LOW | 9 | 5 moderate dev-only npm audit vulns |
| FU-POR-HIGH-01 | HIGH | 10 | No visual assets for viz project |
| FU-POR-MED-01 | MEDIUM | 10 | No LICENSE file |
| FU-POR-MED-02 | MEDIUM | 10 | No CI/CD pipeline |
| FU-POR-MED-03 | MEDIUM | 10 | No .env.example for API keys |
| FU-POR-LOW-01 | LOW | 10 | Hardcoded local paths in audit docs |
| FU-POR-LOW-02 | LOW | 10 | Version 0.0.0 in both packages |
| FU-POR-LOW-03 | LOW | 10 | No unified task runner |
| FU-POR-LOW-04 | LOW | 10 | Single TODO in production code |
| FU-POR-LOW-05 | LOW | 10 | README quickstart ordering |

**Total: 59 findings** (1 CRITICAL, 12 HIGH, 25 MEDIUM, 21 LOW) + 3 positive confirmations

---

## Appendix C: Cross-Reference to First Audit

| First Audit ID | Follow-up Finding(s) | Status |
|----------------|---------------------|--------|
| ENG-LOW-02 | FU-ADV-MED-03 | Config proven dead; upgraded to MEDIUM |
| ENG-LOW-03 | FU-ADV-HIGH-01 | Location bias proven with 14 seeds; upgraded to HIGH |
| DET-LOW-01 | FU-DET-LOW-01 | Confirmed — partial tick metadata missing |
| CON-MED-01 | FU-BUN-MED-01 | Quantified: 68-81% SOCIAL_MOVE removal pre-segment |
| CON-MED-02 | FU-BUN-HIGH-01, FU-BUN-MED-02 | Source-vs-destination proven; upgraded to HIGH |
| FE-MED-01 | FU-VIZ-MED-02 | Confirmed — sort assumption still present |
| FE-MED-02 | FU-VIZ-MED-03 | Confirmed — initial_location still ignored |
| FE-MED-03 | FU-VIZ-MED-04, FU-VIZ-HIGH-01 | Expanded: 16 `as any` casts + 6 `as unknown as` |
| SEC-MED-01 | FU-DEP-HIGH-01 | Upgraded to HIGH: completely unpinned deps |
| ST-MED-01 | FU-STO-CRIT-01 | Upgraded to CRITICAL: Haiku as creative engine confirmed |
| NARR-MED-01 | FU-STO-HIGH-01 | Root-caused: arc search endpoint selection issue |
| S-03/S-15 | FU-CON-HIGH-01, FU-CON-HIGH-02 | New: significance never computed, contract lie |
