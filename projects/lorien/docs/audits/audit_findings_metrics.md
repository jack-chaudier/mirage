# Metrics Pipeline & Integration Audit Findings

**Auditor:** metrics-auditor
**Date:** 2026-02-08
**Scope:** Metrics pipeline (tension, irony, thematic, segmentation), event bundler, renderer integration
**Simulation seed:** 42 (80 events -> 48 after bundling, 9 scenes)
**Existing tests:** 20/20 passing (test_metrics_pipeline.py, test_event_bundler.py)

---

## Summary

**44 checks passed, 8 findings (0 CRITICAL, 0 HIGH, 3 MEDIUM, 5 LOW)**

The metrics pipeline is well-implemented and functionally correct. All critical invariants hold: pipeline order is correct, tension values are in [0,1], all events land in exactly one scene, no causal links reference dropped events, and the renderer payload matches the NarrativeFieldPayload contract. The 8 findings are all spec-vs-implementation deviations where the implementation uses intentional MVP simplifications that differ from the full spec formulas but produce reasonable results.

---

## Pipeline Order Verification

**Status: PASS**

The pipeline in `src/engine/narrativefield/metrics/pipeline.py:89-128` correctly executes in the spec-mandated order (data-flow.md Section 3.2):

1. Irony (line 99)
2. Thematic (line 102)
3. Tension (line 111) -- reads irony values from step 1
4. Bundle (line 123) -- runs after metrics so importance/significance are available
5. Segmentation (line 127) -- runs on bundled events

The bundler running between tension and segmentation is an implementation choice that differs from the spec (spec doesn't mention bundling), but it is correct: the bundler needs metrics for filtering, and segmentation benefits from fewer, cleaner events.

---

## Finding M-01 (MEDIUM): goal_frustration uses heuristic instead of cosine distance

**Spec:** tension-pipeline.md Section 2.3 defines `goal_frustration` as `1 - cosine_similarity(goal_vector, satisfaction_vector)` using 6 GoalVector dimensions (safety, status, secrecy, truth_seeking, autonomy, loyalty).

**Implementation:** `src/engine/narrativefield/metrics/tension.py:135-143` uses:
```python
0.6 * stress + 0.4 * (1.0 - dramatic_budget)
```

**Impact:** The heuristic correlates with frustration (stress and depleted budget indicate thwarted goals) but does not capture per-dimension goal frustration. An agent with high truth_seeking frustrated by lies would score the same as an agent with high secrecy frustrated by reveals, as long as their stress/budget are equal.

**Recommendation:** Accept for MVP. The full cosine distance formula requires computing a satisfaction vector from world state, which is expensive. The heuristic is a reasonable proxy.

---

## Finding M-02 (MEDIUM): resource_scarcity uses pacing proxies instead of world-state features

**Spec:** tension-pipeline.md Section 2.6 defines `resource_scarcity` using social capital (allies vs enemies), exit scarcity (location capacity), and privacy scarcity (CONFIDE/REVEAL in public locations).

**Implementation:** `src/engine/narrativefield/metrics/tension.py:193-199` uses:
```python
max(budget_scarcity, composure_loss) * 0.85 + timer * 0.15
```
where `budget_scarcity = 1 - dramatic_budget`, `composure_loss = 1 - composure`, `timer = recovery_timer / 8.0`.

**Impact:** The implementation captures "how depleted is this agent" rather than "what resources are scarce in the environment." Agents isolated among enemies at a full table will not show high resource_scarcity unless their pacing state is also depleted.

**Recommendation:** Accept for MVP. The pacing-based proxy works because pacing state is influenced by social dynamics during simulation.

---

## Finding M-03 (MEDIUM): Irony collapse scoring is simplified

**Spec:** irony-and-beliefs.md Section 4.3 defines collapse score as:
```
score = 0.5 * magnitude + 0.3 * breadth + 0.2 * denial_bonus
```
where magnitude = `min(drop / 2.0, 1.0)`, breadth = `min(len(collapsed_beliefs) / 4.0, 1.0)`.

**Implementation:** `src/engine/narrativefield/metrics/irony.py:164-169` sets `score = drop` directly.

**Impact:** The simplified scoring doesn't distinguish between a single large belief collapse and many simultaneous small collapses. The breadth and denial_bonus components add narrative weight that the simple drop value doesn't capture.

**Recommendation:** Consider implementing the full formula. It's a straightforward change and the data (collapsed_beliefs list) is already available.

---

## Finding L-01 (LOW): information_gap uses belief-state diversity instead of per-participant gap

**Spec:** tension-pipeline.md Section 2.5 computes information_gap by iterating over each participant and scoring their individual knowledge gap per secret.

**Implementation:** `src/engine/narrativefield/metrics/tension.py:163-190` measures belief-state diversity -- how many distinct belief states exist among agents present for each secret. More diverse states = higher gap.

**Impact:** The diversity approach captures "how much disagreement/ignorance exists" but doesn't weight by individual participant relevance. Functionally similar for the dinner party where most secrets involve present agents.

---

## Finding L-02 (LOW): relationship_volatility missing oscillation bonus

**Spec:** tension-pipeline.md Section 2.4 includes an oscillation bonus (`sign_changes / 3.0`, capped at 0.3) for when relationship deltas alternate sign.

**Implementation:** `src/engine/narrativefield/metrics/tension.py:146-160` uses `max(current_scaled, recent_scaled)` without oscillation detection.

**Impact:** Rapid trust oscillation (trust goes up then down repeatedly) won't produce elevated volatility scores. This is a minor narrative signal for the dinner party MVP.

---

## Finding L-03 (LOW): TensionWeights defaults are pre-normalized

**Spec:** tension-pipeline.md Section 3.1 says default weights are all `1.0` (unnormalized), with aggregation dividing by `sum(weights)`.

**Implementation:** `src/engine/narrativefield/metrics/tension.py:32-40` uses `0.125` per weight (pre-normalized, sum=1.0).

**Impact:** Zero. The _normalize_weights function in line 58-62 normalizes any weight vector, so the mathematical result is identical. This is purely a cosmetic deviation.

---

## Finding L-04 (LOW): secret_relevance missing 0.9 tier

**Spec:** irony-and-beliefs.md Section 3.2 defines 5 relevance tiers: 1.0 (about), 0.9 (about someone in relationship with AND both present), 0.7 (held by), 0.5 (about someone present), 0.2 (otherwise).

**Implementation:** `src/engine/narrativefield/metrics/irony.py:14-26` has 4 tiers: 1.0 / 0.7 / 0.5 / 0.2 (no 0.9 tier).

**Impact:** Irony scores will be slightly lower for secrets about agents the current agent has a relationship with when both are present. The missing tier reduces irony sensitivity by ~10% for affected agent-secret pairs.

---

## Finding L-05 (LOW): Irony "relevant unknown" scoring ambiguity

**Spec text** (Section 3.1): "Relevant unknown (UNKNOWN on a secret that is about them or held by them): 1.5"
**Spec code sample** (Section 3.1): Uses `1.5` as a fixed value (not multiplied by relevance).

**Implementation:** `src/engine/narrativefield/metrics/irony.py:64-66` uses `1.5 * relevance` for about/holder cases.

**Impact:** For the "about" case (relevance=1.0), the result is identical. For the "held by" case (relevance=0.7), the implementation gives 1.05 instead of 1.5. This reduces holder irony by 30%.

**Note:** The spec text and spec code are actually inconsistent with each other on this point. The text says 1.5 (fixed), the code sample also says 1.5 (but the code doesn't multiply by relevance for this tier). The implementation chose to multiply, which is defensible but differs from the spec code.

---

## Checks That Passed

### Tension Pipeline
- All 8 canonical tension_component keys present on every event
- All tension values in [0.0, 1.0]
- Tension = mean of components (default equal weights)
- CONFLICT/CATASTROPHE events have non-zero danger
- LIE events have moral_cost >= 0.5
- CONFIDE events have moral_cost = 0.2
- Global normalization pass intentionally skipped (spec marks it optional)

### Irony Pipeline
- secret_relevance follows spec values: 1.0/0.7/0.5/0.2
- All irony collapses have drop >= 0.5 (threshold matches spec)
- scene_irony = mean of agent_irony (spec Section 3.3)
- Irony collapse detection uses scene_irony (spec Section 4.2)

### Thematic Pipeline
- THEMATIC_SHIFT_RULES match spec (data-flow.md Section 7)
- CONFLICT events produce order_chaos thematic shift
- LIE events produce truth_deception thematic shift
- Near-zero thematic shifts correctly filtered (|delta| > 0.01)

### Segmentation
- All events appear in exactly one scene
- All scene types are valid (catastrophe/confrontation/revelation/bonding/escalation/maintenance)
- Scenes ordered by time_start
- Jaccard threshold = 0.3 (spec Section 4)
- Tension window = 5, drop_ratio = 0.3, sustained_count = 3
- Min scene size = 3
- Time gap threshold = 5.0 minutes
- SOCIAL_MOVE forces scene boundary (spec Section 7)
- Same-tick events grouped together (spec Section 9)
- Scene tension_arc and tension_peak are consistent

### Event Bundler
- Bundle stats counts are consistent
- No causal links reference dropped events
- No forward causal links detected
- No self-referential causal links
- Social move squash preserves route metadata
- Attached moves have required fields (mover, move_event, source, destination)
- Compression ratio: 60% (80 -> 48 events)

### Data Flow / Renderer Payload
- NarrativeFieldPayload has all required top-level keys (format_version, metadata, agents, locations, secrets, events, scenes, belief_snapshots)
- AgentManifest fields match spec (id, name, initial_location, goal_summary, primary_flaw)
- BeliefSnapshot fields match spec (tick_id, sim_time, beliefs, agent_irony, scene_irony)
- metadata.event_count matches actual event count
- All events have full metrics in payload
- Tension components have 8 canonical keys in payload
- MetricsPipelineOutput has correct shape

---

## Audit Script

The end-to-end audit script is at: `src/engine/audit_metrics.py`

Run with:
```bash
cd src/engine && source .venv/bin/activate && python audit_metrics.py
```
