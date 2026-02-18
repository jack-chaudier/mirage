# Implementation Risk Report

**Author:** Implementation Risk Analyst
**Date:** 2026-02-06
**Scope:** All 17 specifications + MASTER_PLAN.md + doc3.md
**Status:** Complete

---

## Executive Summary

The NarrativeField specification suite is unusually thorough for a planning phase. The 17 specs are well-cross-referenced, include worked examples, and resolve most ambiguities. However, implementation will encounter approximately **12 concrete bugs**, **8 performance concerns**, **6 underspecified behaviors**, and **4 known dead-code placeholders** that must be addressed before or during coding.

The highest-risk area is the **simulation layer** (Phases 2-3), where pacing physics, decision engine, and tick loop contain field-name mismatches, fragile string matching, and post-hoc event mutation. The visualization layer (Phase 1) is the cleanest and lowest-risk. The metrics layer has a known unimplemented feature (`denial_bonus`) and a potential danger sub-metric bug.

---

## 1. Risk Matrix by Implementation Phase

### Phase 1: Renderer with Fake Data (LOW RISK)

| Risk | Severity | Likelihood | Spec | Notes |
|------|----------|------------|------|-------|
| BFS uses `queue.shift()` (O(n) per pop) | Low | Certain | renderer-architecture.md:770 | TypeScript array.shift() is O(n). Use a proper queue or accept it at 200 events. |
| `Set<string>` in Zustand store may cause re-render issues | Medium | Moderate | renderer-architecture.md:314 | Zustand selectors use referential equality; `Set` objects don't serialize cleanly. Use sorted arrays or `Immutable.Set`. |
| Hit canvas color-encoding limited to ~16M unique events | Low | Negligible | renderer-architecture.md:640-644 | RGB encoding gives 16.7M unique IDs. Irrelevant for 200 events. |
| `agents` variable referenced but not in scope in `generateSplines` | Medium | Certain | thread-layout.md:285 | `CHARACTER_COLORS[agents.indexOf(agentId)]` -- `agents` is not a parameter. Must pass agent list or use a lookup map. |
| Fake data spec describes 70 events but only defines ~20 in detail | Low | Certain | fake-data-visual-spec.md:62-88 | Tier 2 and Tier 3 events are described narratively, not as JSON. The coding agent must generate them. |

### Phase 2: Dinner Party Simulation (HIGH RISK)

| Risk | Severity | Likelihood | Spec | Notes |
|------|----------|------------|------|-------|
| `select_catastrophe_type` uses `dominant_flaw.name` but field is `flaw_type` | High | Certain | pacing-physics.md | AttributeError at runtime. Must use `dominant_flaw.flaw_type`. |
| `content_metadata` vs `metadata` field name inconsistency | High | Certain | decision-engine.md | Action dataclass uses `metadata`; code references `content_metadata` in some contexts. Pick one. |
| PHYSICAL delta generation uses fragile string matching | Medium | Likely | tick-loop.md | `"drink" in action.content.lower()` -- any action containing "drink" triggers alcohol_level delta. Needs structured metadata. |
| Pacing deltas appended to `agent_events[-1].deltas` post-creation | Medium | Likely | tick-loop.md | Mutating events after creation violates event immutability. Should generate pacing events separately or include pacing deltas during event creation. |
| Worked examples use non-canonical character names | Medium | Certain | pacing-physics.md | References "Marco" and "Sophia" instead of canonical names. Already flagged in MASTER_PLAN but must be fixed before implementation. |
| O(agents x secrets) candidate action generation | Medium | Moderate | decision-engine.md | For each agent: REVEAL/CONFIDE/LIE candidates per secret. 6 agents x 5 secrets = 30 candidates per agent type. Manageable for MVP but scales poorly. |
| `is_brief_departure` uses `all_events.index(event)` (O(n)) | Medium | Certain | scenes.md | Linear scan to find event index. Should use a dict lookup. |
| BFS in scenes.md uses `queue.pop(0)` (O(n) per pop) | Low | Certain | scenes.md | Python `list.pop(0)` is O(n). Use `collections.deque.popleft()`. |

### Phase 3: Metrics Pipeline (MEDIUM RISK)

| Risk | Severity | Likelihood | Spec | Notes |
|------|----------|------------|------|-------|
| `danger` sub-metric accesses `context.secrets[delta.agent]` | High | Likely | tension-pipeline.md:84 | `delta.agent` is the agent whose state changed, NOT a secret ID. For SECRET_STATE deltas, the secret ID is in `delta.attribute`. This is a bug. |
| `denial_bonus = 0.0` never computed | Medium | Certain | irony-and-beliefs.md:360 | `irony_collapse_score` has a `denial_bonus` term weighted at 0.2 but it's hardcoded to 0.0 with a comment "computed when we have access to before/after states." Must be implemented or the weight redistributed. |
| `world_state_before()` uses deepcopy + replay (expensive) | High | Likely | tension-pipeline.md:746-754 | Called once per event, per sub-metric. For 200 events, this means ~200 deepcopy operations of the full WorldState plus delta replays. Could be O(n^2) overall. |
| `significance = 0.0` placeholder | Low | Certain | story-extraction.md (via events.md) | Counterfactual impact is a Phase 5 feature. Score component weighted at 0.15 in arc scoring but always zero. |
| `_secret_relevance` in tension-pipeline.md differs from irony-and-beliefs.md | Medium | Moderate | tension-pipeline.md:338-353 vs irony-and-beliefs.md:154-182 | Different relevance tiers (1.0/0.7/0.4/0.1 vs 1.0/0.9/0.7/0.5/0.2/0.0). Should be unified or documented as intentionally different. |
| Irony collapse threshold inconsistency | Low | Moderate | irony-and-beliefs.md:301 vs scene-segmentation.md:127 | Both use 0.5, but scene-segmentation also gates on `collapse.get("drop", 0) >= 0.5` which measures a different quantity than the `drop >= 0.5` in irony-and-beliefs. Verify these align. |
| `event.metrics["tension"]` accessed as dict key | Medium | Likely | scene-segmentation.md:93,105,263 | Events use dict-style access (`e.metrics["tension"]`), but renderer-architecture.md defines `metrics` as an `EventMetrics` object with named fields. Must be consistent: either dict or typed object. |

### Phase 4: Story Extraction (LOW-MEDIUM RISK)

| Risk | Severity | Likelihood | Spec | Notes |
|------|----------|------------|------|-------|
| `classify_scene_type_from_ids` called but never defined | Medium | Certain | scene-segmentation.md:313 | `merge_two_scenes` calls `classify_scene_type_from_ids(combined_events)` but only `classify_scene_type(events)` is defined (takes Event objects, not IDs). |
| Grammar rule 8 (causal connectivity) only reports first violation | Low | Moderate | story-extraction.md:151 | `break` after first gap means multi-gap arcs only report one error. Minor UX issue. |
| LLM prompt template uses f-string with `{len(beat_sheet.beats) * 300}` | Low | Certain | story-extraction.md:724 | Token count estimate hardcoded. Acceptable for MVP. |

### Phase 5-6: Counterfactual & Story Queries (NOT YET SPECIFIED)

These phases are explicitly deferred. The `significance` field and `StoryQuery.search_stories()` function are stubs.

---

## 2. Algorithm Complexity Analysis

### 2.1 Critical Path: Metrics Pipeline

The metrics pipeline is the most computationally intensive post-simulation step.

| Operation | Complexity | For 200 Events | Concern |
|-----------|-----------|-----------------|---------|
| `world_state_before()` per event | O(n) replay from nearest snapshot | 200 x ~10 event replays = 2000 delta applications + 200 deepcopy calls | **Major concern.** deepcopy of WorldState (6 agents, relationships, beliefs) is expensive. |
| `danger()` per event | O(d) where d = delta count | 200 x ~5 deltas = 1000 | Acceptable |
| `time_pressure()` per event | O(s) where s = secret count | 200 x 5 x 6 = 6000 | Acceptable |
| `goal_frustration()` per event | O(a + s) agents + secrets | 200 x (6 + 5) = 2200 | Acceptable |
| `relationship_volatility()` per event | O(w) lookback window | 200 x 5 = 1000 | Acceptable |
| `information_gap()` per event | O(s x p) secrets x participants | 200 x 5 x 3 = 3000 | Acceptable |
| `resource_scarcity()` per event | O(a) agents at location | 200 x 6 = 1200 | Acceptable |
| `moral_cost()` per event | O(t x d) targets x deltas | 200 x 3 x 5 = 3000 | Acceptable |
| `irony_density()` per event | O(a x s) agents x secrets | 200 x 6 x 5 = 6000 | Acceptable |
| **Total sub-metric computation** | | ~20K operations | Acceptable |
| **Bottleneck: `world_state_before()`** | O(n) per call, O(n^2) total | 200 x deepcopy + replay | **Must optimize.** Cache the forward-replayed world state. |

**Recommendation:** Instead of replaying from snapshots for each event, maintain a running WorldState that applies deltas forward in a single pass. This reduces `world_state_before()` from O(n^2) to O(n) total.

### 2.2 Thread Layout Engine

| Operation | Complexity | For MVP | Concern |
|-----------|-----------|---------|---------|
| Force computation per time sample | O(a^2) per iteration, O(a^2 x i) per sample | 6^2 x 50 = 1800 per sample | Acceptable |
| Total across all samples | O(s x a^2 x i) | 100 x 1800 = 180K | Acceptable (~55ms) |
| Spline generation | O(a x s) | 6 x 100 = 600 | Trivial |

Layout engine is well within budget.

### 2.3 Causal Index (BFS-3)

| Operation | Complexity | For MVP | Concern |
|-----------|-----------|---------|---------|
| BFS-3 per event | O(k^3) where k = avg causal links | 2^3 = 8 nodes per BFS | Trivial |
| Total index build | O(n x k^3) | 200 x 8 = 1600 | Trivial (~5ms) |

No concern. BFS-3 on sparse causal graphs is fast.

### 2.4 Scene Segmentation

| Operation | Complexity | For MVP | Concern |
|-----------|-----------|---------|---------|
| Main loop | O(n) single pass | 200 events | Trivial |
| Tension gap check (per event) | O(w) where w = window size | 200 x 5 = 1000 | Trivial |
| Merge pass | O(s) where s = scene count | ~10-15 scenes | Trivial |

No concern.

---

## 3. Underspecified Behaviors

### 3.1 SOCIAL_MOVE Forced Scene Boundary

**Status:** Partially specified.

`scene-segmentation.md` Section 7 defines `is_scene_boundary_move()` which forces a boundary on SOCIAL_MOVE events. However, `scenes.md` (the schema spec) does **not** mention this rule in its 5 segmentation triggers. The schema spec lists: location_change, participant_turnover, tension_gap, time_proximity, irony_collapse. SOCIAL_MOVE forced boundary is a 6th trigger that exists only in scene-segmentation.md.

**Impact:** Schema spec and algorithm spec disagree on the number of triggers. Implementer reading only scenes.md will miss the SOCIAL_MOVE rule.

**Recommendation:** Add SOCIAL_MOVE forced boundary to scenes.md Section 2 (boundary conditions).

### 3.2 Thematic Shift Computation

**Status:** Underspecified.

`data-flow.md` Section 7 defines `THEMATIC_SHIFT_RULES` and `compute_thematic_shift()`, but there is no dedicated spec for thematic computation. The thematic axes themselves (order_chaos, truth_deception, loyalty_betrayal, innocence_corruption, freedom_control) are mentioned but never formally enumerated as a closed set.

Questions:
- Are these axes the complete list?
- What are the initial thematic positions (0.0 for all)?
- Is thematic_shift cumulative or per-event?
- How does the renderer display accumulated thematic trajectory?

**Impact:** The story extraction pipeline uses `thematic_coherence` as a 0.15-weight scoring component, but the thematic data feeding it is loosely specified.

**Recommendation:** Either write a minimal thematic spec or inline the axis definitions in events.md.

### 3.3 Witness/Overhear Event Generation Details

**Status:** Partially specified.

`tick-loop.md` describes witness events for agents who observe but don't participate. The rules for OBSERVE vs overhear are described narratively but the exact threshold for when an overhear event is generated (vs. the agent simply being unaware) is not formalized.

Key missing detail: What is the probability/threshold for overhearing? Is it deterministic (always overhear if in adjacent location) or probabilistic? The `overhear_from` list in world.md implies deterministic adjacency, but no probability is assigned.

**Recommendation:** Make overhearing deterministic for MVP (if in overhear_from location, always generate a witness OBSERVE event).

### 3.4 Action Conflict Resolution Priority Classes

**Status:** Specified but incomplete.

`tick-loop.md` defines 3 priority classes (1=catastrophe, 2=conflict/reveal, 3=everything else) but doesn't specify what happens when two agents both attempt CONFLICT against each other in the same tick. Both get priority class 2. The tiebreaker is `utility_score` then random, but the downgrade rules (Section 5 in tick-loop) don't cover the case of mutual CONFLICT.

**Recommendation:** Specify that mutual CONFLICT results in a single CONFLICT event with both agents as participants (source = higher utility, target = other).

### 3.5 GoalVector `closeness` Dict

**Status:** Underspecified at the metrics level.

`agents.md` defines `closeness: dict[str, float]` as per-agent closeness desires. `tension-pipeline.md` notes this is "averaged into the status dimension" for simplicity. However, the decision engine uses closeness for action scoring (wanting to be near specific people), and it's unclear how the closeness dict interacts with the 6-dimensional array conversion in `_goal_to_array()`.

**Recommendation:** Document that closeness is excluded from the tension pipeline's goal vector and only used in the decision engine.

### 3.6 Recovery Timer and Suppression Count Reset

**Status:** Specified in pacing-physics.md but interplay with catastrophe unclear.

After a catastrophe: `recovery_timer = 8`, `suppression_count = 0`. But if the same agent is involved in a CONFLICT event on the very next tick (while recovery_timer > 0), the recovery timer blocks further stress accumulation (stress changes are dampened). The interaction between "just had a catastrophe" and "immediately in conflict again" needs clarification.

**Recommendation:** Add a post-catastrophe immunity window (e.g., 2 ticks) where the agent cannot trigger another catastrophe, even if conditions re-align.

---

## 4. Dead Code and Placeholder Inventory

| Item | Location | Status | Phase Required |
|------|----------|--------|----------------|
| `significance = 0.0` | events.md, story-extraction.md | Placeholder | Phase 5 (counterfactual impact) |
| `denial_bonus = 0.0` | irony-and-beliefs.md:360 | Placeholder (never computed) | Phase 3 (irony pipeline) |
| `search_stories()` | story-extraction.md:930 | Stub (`...` body) | Phase 6 (story queries) |
| `buildLocationLookup()` / `buildInteractionLookup()` | thread-layout.md:238-249 | Stub (pseudocode, `...` body) | Phase 1 (thread layout) |

---

## 5. Testing Strategy Recommendations

### 5.1 Phase 1: Renderer

| Test Area | Strategy | Priority |
|-----------|----------|----------|
| JSON parsing/validation | Property-based tests: generate random valid/invalid Event JSON, verify parser accepts/rejects correctly | High |
| Causal index BFS-3 | Unit test with known graph topologies: linear chain, diamond, star, disconnected. Verify backward/forward cones. | High |
| Thread layout convergence | Integration test: verify all 6 threads maintain minSeparation at every time sample. Test edge case: all agents at same location for full simulation. | Medium |
| Hit canvas accuracy | Manual/visual test: verify click on overlapping events returns highest-significance event. | Medium |
| Tension recompute performance | Benchmark: slider drag must recompute 200 events in < 16ms. | Medium |
| Zoom level transitions | Visual regression test: capture screenshots at each zoom level, verify correct elements appear/hide. | Low |

### 5.2 Phase 2: Simulation

| Test Area | Strategy | Priority |
|-----------|----------|----------|
| Pacing physics update order | Unit test: verify stress -> composure -> commitment -> budget -> recovery -> suppression order produces correct intermediate values. Test with pacing-physics.md worked examples (after fixing character names). | Critical |
| Catastrophe trigger | Property-based test: generate random PacingState values, verify catastrophe fires if and only if `stress * commitment^2 + suppression_count * 0.03 >= 0.35 AND composure < 0.30`. | Critical |
| Decision engine scoring | Unit test: reproduce the Victor and Elena worked traces from decision-engine.md. Verify noise doesn't flip decisions when gap > 0.3. | High |
| Conflict resolution | Unit test: submit conflicting actions from 2 agents. Verify priority class ordering, downgrade rules, and single-event generation. | High |
| Event immutability | Integration test: verify no event is modified after creation (no post-hoc delta appending). | Medium |
| Termination conditions | Integration test: verify simulation terminates when max_sim_time reached, < 2 agents remain, or all calm+cooldown. | Medium |

### 5.3 Phase 3: Metrics

| Test Area | Strategy | Priority |
|-----------|----------|----------|
| Tension pipeline end-to-end | Integration test: run the 5-event worked example from tension-pipeline.md and verify all 8 sub-metrics match within 0.05 tolerance. | Critical |
| Irony computation | Unit test: reproduce the 10-event trace from irony-and-beliefs.md. Verify scene_irony values at each step. | Critical |
| Scene segmentation | Integration test: run the 20-event worked example from scene-segmentation.md. Verify 5 scenes are produced with correct boundaries. | High |
| World state reconstruction | Performance test: verify `world_state_before()` optimization. Time 200 calls with and without caching. Target: < 100ms total. | High |
| Danger sub-metric bug | Unit test: create an event with SECRET_STATE delta and verify the correct secret is looked up via `delta.attribute`, not `delta.agent`. | Critical |

### 5.4 Phase 4: Story Extraction

| Test Area | Strategy | Priority |
|-----------|----------|----------|
| Arc grammar validation | Unit test: construct minimal valid and invalid beat sequences. Test all 9 rules independently. | High |
| Beat classification | Unit test: reproduce the 8-event worked example from story-extraction.md. Verify classifications match. | Medium |
| Fix classification edge cases | Unit test: arc with no turning point, arc with multiple turning points, 4-event minimal arc. | Medium |
| LLM prompt construction | Snapshot test: verify prompt output matches expected format for a known beat sheet. | Low |

---

## 6. Cross-Spec Consistency Issues

### 6.1 TensionWeights Sub-metric Names

`doc3.md` TensionWeights use names: `danger`, `time_pressure`, `goal_frustration`, `relationship_volatility`, `information_gap`, `resource_scarcity`, `moral_cost`, `irony_density`.

`tension-pipeline.md` uses the same names. `renderer-architecture.md` TypeScript interfaces also match.

**Status:** ALIGNED. No issue here (my earlier concern from reading doc3.md was unfounded).

### 6.2 Participant Turnover Threshold

`scenes.md` (schema spec) uses threshold 0.5 for participant_turnover.
`scene-segmentation.md` (algorithm spec) uses threshold 0.3.

Decision 21 says scene-segmentation.md is authoritative.

**Status:** CONFLICT. scenes.md must be updated from 0.5 to 0.3.

### 6.3 Scene Dataclass Fields

`scenes.md` defines `Scene` with fields: `id, event_ids, location, participants, time_start, time_end, tension_arc, dominant_theme, scene_type`.

`scene-segmentation.md` defines the same fields but adds derived properties: `duration`, `event_count`, `peak_tension`, `mean_tension`, `tension_direction`.

The renderer's TypeScript `Scene` interface matches the base fields.

**Status:** ALIGNED (derived properties are computed, not stored).

### 6.4 Event.metrics Structure

Python specs use dict-style access: `event.metrics["tension"]`, `event.metrics.get("irony_collapse", {})`.

TypeScript renderer defines `EventMetrics` as a typed object with named fields.

The JSON serialization bridges them, but the Python code should also use a typed dataclass for metrics, not raw dicts, to catch key errors at development time.

**Status:** MINOR INCONSISTENCY. Recommend Python `@dataclass` for EventMetrics.

### 6.5 Secret.holder vs Secret.about

In `world.md`, each secret has `holder` (who knows/is responsible) and `about` (who it concerns).

For `secret_affair_01`: `holder = "elena"` but the affair is between Elena AND Marcus. `irony-and-beliefs.md` Section 5.2 lists holders as "elena, marcus" (plural), but `world.md` SecretDefinition has `holder: str` (singular).

**Status:** CONFLICT. SecretDefinition.holder is typed as a single string, but some secrets have multiple holders. Either make `holder` a list or add a `co_holders` field. This affects `moral_cost` and `information_gap` computations.

### 6.6 Data Flow Pipeline Ordering

`data-flow.md` specifies execution order: irony -> thematic -> tension -> segmentation.

But `tension-pipeline.md` Section 2.8 (irony_density) says it reads from the belief matrix directly, not from pre-computed irony values. If irony_density re-computes irony inline, the separate irony pipeline run is redundant. If irony_density reads the pre-computed `event.metrics.irony` value, then the ordering matters and is correct.

**Status:** AMBIGUOUS. The irony_density sub-metric appears to independently compute irony from the belief matrix, duplicating work done by the irony pipeline. Clarify: should irony_density read from pre-computed irony values (faster, requires pipeline ordering) or compute independently (more self-contained)?

---

## 7. Noise and Calibration Concerns

### 7.1 Decision Engine Noise (sigma = 0.1)

The decision engine adds Gaussian noise (sigma = 0.1) to action scores. The spec notes that a gap > 0.3 is "rarely flipped" and a gap < 0.15 is "frequently flipped."

**Risk:** The noise distribution is unbounded Gaussian. A sigma-0.1 Gaussian has ~0.3% probability of exceeding 0.3 and ~0.003% of exceeding 0.4. For 200 events x 10 candidate actions = 2000 noise samples per run, the 0.3% tail means ~6 unexpected flips per simulation.

**Recommendation:** Consider clipping noise to [-0.3, 0.3] to prevent extreme outliers that could produce narratively incoherent decisions.

### 7.2 Tension Sub-metric Calibration

Each sub-metric is independently normalized to [0, 1], but their semantic scales differ. `danger` returns 0.8 for a single CONFLICT event; `time_pressure` returns 0.09 at 29% evening completion. With default weights (all 1.0), danger dominates any event containing a CONFLICT.

The genre presets partially address this by re-weighting, but the base calibration of sub-metrics is uneven.

**Recommendation:** After implementing the pipeline, run it on the 70-event fake dataset and check the distribution of each sub-metric. If any sub-metric is consistently > 0.8 or < 0.1, adjust its internal scaling.

### 7.3 Catastrophe Threshold Sensitivity

The catastrophe formula `stress * commitment^2 + suppression_count * 0.03 >= 0.35 AND composure < 0.30` is highly sensitive to the 0.35 threshold. From the worked example in pacing-physics.md:

- stress=0.72, commitment=0.65, suppression=3: 0.72 * 0.4225 + 0.09 = 0.394 >= 0.35 (fires)
- stress=0.60, commitment=0.65, suppression=3: 0.60 * 0.4225 + 0.09 = 0.344 < 0.35 (doesn't fire)

A 0.12 difference in stress separates catastrophe from no-catastrophe. This is by design (cusp catastrophe is supposed to be abrupt), but it means small calibration errors in stress update rules can dramatically change simulation outcomes.

**Recommendation:** Log catastrophe near-misses (formula value in [0.30, 0.35) with composure < 0.30) during development to understand how often agents approach but don't cross the threshold.

---

## 8. Phase 1 (Renderer) Specific Risks

Phase 1 is the first to be implemented and has the most self-contained spec. Key risks:

1. **Fake data must be hand-crafted or scripted.** The fake-data-visual-spec describes 70 events qualitatively but provides JSON-level detail for only ~20. Someone must write the remaining 50 events with proper causal_links, deltas, and metrics values.

2. **Thread layout `agents` variable out of scope.** In `thread-layout.md:285`, `generateSplines()` references `agents` (the agent ID list) but it's not a parameter of the function. This will cause a compile error.

3. **Tension heatmap rendering is not specified algorithmically.** The fake-data-visual-spec describes a "Gaussian-blurred color field" (sigma = 30px x, 60px y) but no code is provided. The renderer team must implement Gaussian blur in Canvas, which typically requires a two-pass box blur approximation.

4. **Causal link arrows at Detail zoom.** The interaction-model describes thin directed arrows between causally linked events at Detail zoom, but the renderer-architecture's `drawFrame()` function doesn't include an arrow-drawing step. Must be added to the EventNodeLayer or a dedicated CausalArrowLayer.

---

## 9. Recommendations Summary (Ordered by Priority)

### Must Fix Before Implementation

1. **Fix `select_catastrophe_type` field name:** `dominant_flaw.name` -> `dominant_flaw.flaw_type` (pacing-physics.md)
2. **Fix `danger` sub-metric secret lookup:** `context.secrets[delta.agent]` -> `context.secrets[delta.attribute]` (tension-pipeline.md:84)
3. **Fix `content_metadata` vs `metadata` inconsistency:** Choose one field name (decision-engine.md)
4. **Fix `classify_scene_type_from_ids`:** Define the function or change call to pass Event objects (scene-segmentation.md:313)
5. **Update scenes.md participant_turnover threshold:** 0.5 -> 0.3 (per Decision 21)
6. **Add SOCIAL_MOVE forced boundary to scenes.md** (per scene-segmentation.md Section 7)
7. **Fix `generateSplines` scope bug:** Pass `agents` parameter (thread-layout.md:285)
8. **Resolve `Secret.holder` type:** `str` vs `list[str]` (world.md vs irony-and-beliefs.md)

### Should Fix During Implementation

9. **Optimize `world_state_before()`:** Use forward-replaying cached state instead of deepcopy + replay
10. **Implement `denial_bonus`:** Or redistribute its 0.2 weight to other irony_collapse_score components
11. **Replace `queue.pop(0)` with deque:** In scenes.md BFS and renderer BFS
12. **Replace fragile string matching:** `"drink" in action.content.lower()` -> structured action metadata
13. **Prevent post-creation event mutation:** Generate pacing deltas during event creation, not appended after
14. **Unify `_secret_relevance`:** Use one function across tension-pipeline.md and irony-and-beliefs.md, or document the intentional difference
15. **Define thematic axes formally:** Enumerate the closed set of axes and initial values

### Can Defer

16. **Clip decision engine noise** to prevent extreme outliers
17. **Log catastrophe near-misses** for calibration
18. **Sub-metric calibration pass** after initial implementation
19. **Fix non-canonical character names** in pacing-physics.md worked examples

---

## 10. File-by-File Risk Summary

| File | Risk Level | Key Issues |
|------|-----------|------------|
| events.md | Low | Clean schema. `significance=0.0` placeholder. |
| scenes.md | Medium | Threshold conflict (0.5 vs 0.3). Missing SOCIAL_MOVE trigger. O(n) operations. |
| world.md | Low | `Secret.holder` type issue (str vs list). |
| agents.md | Low | Clean. Defers to pacing-physics for update rules. |
| pacing-physics.md | High | Field name bug (`name` vs `flaw_type`). Non-canonical names. |
| decision-engine.md | High | Field name inconsistency. Unbounded noise. |
| dinner-party-config.md | Low | Clean reference spec. All reconciliation items resolved. |
| tick-loop.md | Medium | Fragile string matching. Post-creation event mutation. |
| renderer-architecture.md | Low | BFS shift() O(n). Set serialization concern. |
| interaction-model.md | Low | Clean. Well-specified edge cases. |
| thread-layout.md | Low | Scope bug in `generateSplines`. Stub functions. |
| fake-data-visual-spec.md | Low | 50 events need generation. Visual target doc only. |
| irony-and-beliefs.md | Medium | `denial_bonus` placeholder. Thorough otherwise. |
| scene-segmentation.md | Medium | Undefined function call. SOCIAL_MOVE rule not in schema. |
| story-extraction.md | Low | `significance` always 0.0. `search_stories` stub. |
| tension-pipeline.md | High | Danger sub-metric bug. Expensive state reconstruction. |
| data-flow.md | Low | Clean integration spec. Pipeline ordering ambiguity. |
| MASTER_PLAN.md | N/A | Reference doc. Correctly identifies most issues. |
| doc3.md | N/A | Canonical design doc. All decisions well-resolved. |
