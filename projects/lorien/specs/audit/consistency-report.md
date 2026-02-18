# NarrativeField Spec Consistency Audit

> **Auditor:** Team Lead (taking over from spec-auditor)
> **Date:** 2026-02-06
> **Scope:** All 17 specs + MASTER_PLAN.md + doc3.md
> **Status:** Complete

---

## Summary

**27 findings total:** 5 CRITICAL, 9 HIGH, 8 MEDIUM, 5 LOW

All 8 pre-verified issues confirmed. 19 additional cross-spec conflicts discovered.

---

## Verified Issues (Mandatory Checks)

| # | Issue | Location | Verified? | Current Value | Correct Value |
|---|-------|----------|-----------|---------------|---------------|
| 1 | `participant_turnover` threshold | scenes.md:126 | YES | `0.5` | `0.3` (Decision 21, scene-segmentation.md authoritative) |
| 2 | SOCIAL_MOVE forced boundary | scenes.md (absent) | YES | Not documented | Must add — scene-segmentation.md Section 7 defines `is_scene_boundary_move()` |
| 3 | Scene type enum divergence | scenes.md:84-90 vs scene-segmentation.md:251-265 | YES | scenes.md has 7 types: bonding, tension_building, confrontation, revelation, aftermath, transition, climax. scene-segmentation.md has 6: catastrophe, confrontation, revelation, bonding, escalation, maintenance | Must reconcile — scene-segmentation.md is authoritative (Decision 21) |
| 4 | `snapshot_interval` wrong unit | data-flow.md:162,621 | YES | "ticks between snapshots" | "events between snapshots" (tick-loop.md and world.md confirm events) |
| 5 | MASTER_PLAN tension sub-metric names | MASTER_PLAN.md:201 | YES | stakes, conflict_proximity, information_asymmetry, irony_density, relationship_strain, constraint_pressure, pacing_momentum, uncertainty | danger, time_pressure, goal_frustration, relationship_volatility, information_gap, resource_scarcity, moral_cost, irony_density (7 of 8 stale; only irony_density matches) |
| 6 | `danger` sub-metric bug | tension-pipeline.md:84 | YES | `context.secrets[delta.agent]` | `context.secrets[delta.attribute]` (delta.agent is the affected agent, not the secret ID; delta.attribute holds the secret_id for SECRET_STATE deltas) |
| 7 | `denial_bonus = 0.0` never computed | irony-and-beliefs.md:360 | YES | Hardcoded `0.0` with comment "computed when we have access to before/after states" | Dead code — weighted at 0.2 in irony_collapse_score but never contributes |
| 8 | `significance = 0.0` always zero | events.md:315, story-extraction.md:377 | YES | Phase 5 stub; weighted at 0.15 in arc scoring | BY DESIGN but affects soft scoring now — all arcs get 0 from this component |

---

## New Findings

### CRITICAL (blocks implementation)

| # | Issue | Specs Involved | Details | Resolution |
|---|-------|---------------|---------|------------|
| C1 | `select_catastrophe_type` uses `dominant_flaw.name` but field is `flaw_type` | pacing-physics.md:433 vs agents.md:135 | `CharacterFlaw` dataclass has `flaw_type: FlawType`, not `name`. Lines 433-439 all use `.name`. AttributeError at runtime. | Change `dominant_flaw.name` to `dominant_flaw.flaw_type` in all 4 branches |
| C2 | EventMetrics missing `tension_components` and `irony_collapse` | events.md:312-317 vs data-flow.md, renderer-architecture.md | Events.md default factory and TS EventMetrics interface omit these two fields that data-flow.md and renderer expect. Blocks Phase 1 fake data generation. | Add to events.md: `tension_components: {}` and `irony_collapse: null` |
| C3 | Three names for WorldState snapshot | data-flow.md (WorldStateSnapshot, 4 fields), scenes.md (SnapshotState, 9 fields), tension-pipeline.md (WorldState) | Same concept, 3 names, 3 field sets. Causes confusion in every cross-spec reference. | Adopt one canonical name + scenes.md field set (superset). Update all 3 specs. |
| C4 | `content_display` vs `description` in SecretDefinition | data-flow.md vs world.md:329+ | data-flow.md uses `content_display`, world.md uses `description`. Same concept, different names. | Pick one (recommend `description` since world.md is the producer). Update data-flow.md. |
| C5 | `content_metadata` field absent from Event schema | decision-engine.md:152,183,202,223 vs events.md | decision-engine.md reads/writes `action.content_metadata` (e.g., `{"secret_id": secret_id}`), but Event schema in events.md has no such field. | Either add `content_metadata: dict` to Event or route secret_id through existing `deltas`. |

### HIGH (causes incorrect behavior)

| # | Issue | Specs Involved | Details | Resolution |
|---|-------|---------------|---------|------------|
| H1 | Scene type enum mismatch (verified #3) | scenes.md:84-90 vs scene-segmentation.md:251-265 | scenes.md: 7 types (bonding, tension_building, confrontation, revelation, aftermath, transition, climax). scene-segmentation.md: 6 types (catastrophe, confrontation, revelation, bonding, escalation, maintenance). Only 3 overlap (confrontation, revelation, bonding). | scene-segmentation.md is authoritative. Update scenes.md to match its 6 types. |
| H2 | `Secret.holder` typed as `str` but some secrets need multiple holders | world.md:329 vs irony-and-beliefs.md | `secret_affair_01` involves both Elena and Marcus but `holder: str` is singular. irony-and-beliefs.md Section 5.2 references plural holders. | Make `holder: list[str]` or add `co_holders: list[str]`. |
| H3 | SecretDefinition missing 4 fields in data-flow.md | data-flow.md vs world.md | data-flow.md omits `initial_knowers`, `initial_suspecters`, `dramatic_weight`, `reveal_consequences`. These are needed by irony pipeline. | Update data-flow.md to include full world.md SecretDefinition. |
| H4 | LocationDefinition missing `overhear_probability` in data-flow.md | data-flow.md vs world.md | Functionally critical — without it, overhearing mechanics fail. | Add `overhear_probability: float` and `description: str` to data-flow.md LocationDefinition. |
| H5 | IronyCollapse dataclass vs JSON format mismatch | irony-and-beliefs.md:333 vs Section 7.1 JSON | Dataclass: `collapsed_beliefs: list[tuple[str, str]]`. JSON: `[{agent, secret, from, to}]`. Dataclass also missing `detected` and `score` fields. | Update dataclass to match JSON output format. |
| H6 | `classify_scene_type_from_ids` called but undefined | scene-segmentation.md:313 | `merge_two_scenes` calls `classify_scene_type_from_ids(combined_events)` but only `classify_scene_type(events: list[Event])` exists. | Either define the function or change call to pass Event objects instead of IDs. |
| H7 | `_secret_relevance` has different tiers across specs | tension-pipeline.md:338-353 vs irony-and-beliefs.md:154-182 | tension-pipeline: 4 tiers (1.0/0.7/0.4/0.1). irony-and-beliefs: 6 tiers (1.0/0.9/0.7/0.5/0.2/0.0). Same function, different outputs. | Unify into one canonical function or document as intentionally different. |
| H8 | `event.metrics` dict-style vs typed-object access | scene-segmentation.md:93,263 vs renderer-architecture.md TS | Python specs use `e.metrics["tension"]`; TS renderer defines EventMetrics as typed object. Will cause key-error vs property-access confusion. | Define Python `@dataclass EventMetrics` to match TS interface. |
| H9 | MASTER_PLAN tension sub-metric names 7/8 stale (verified #5) | MASTER_PLAN.md:201 vs tension-pipeline.md | Only `irony_density` matches. All others renamed since MASTER_PLAN was written. | Update MASTER_PLAN.md Section 2.4 to use canonical names from tension-pipeline.md. |

### MEDIUM (causes confusion, not bugs)

| # | Issue | Specs Involved | Details | Resolution |
|---|-------|---------------|---------|------------|
| M1 | `participant_turnover` threshold stale (verified #1) | scenes.md:126 vs scene-segmentation.md | 0.5 in scenes.md, should be 0.3. | Update scenes.md:126 from 0.5 to 0.3. |
| M2 | SOCIAL_MOVE forced boundary missing (verified #2) | scenes.md vs scene-segmentation.md | scenes.md lists 5 triggers; SOCIAL_MOVE is a 6th trigger only in scene-segmentation.md. | Add to scenes.md Section 2. |
| M3 | `generateSplines` references out-of-scope `agents` | thread-layout.md:285 | `CHARACTER_COLORS[agents.indexOf(agentId)]` but `agents` is not a function parameter. | Pass `agents: string[]` as parameter to `generateSplines`. |
| M4 | renderer-architecture.md NarrativeFieldPayload missing fields | renderer-architecture.md vs data-flow.md | Omits `metadata` and `belief_snapshots`. | Update renderer-architecture.md to match data-flow.md. |
| M5 | AgentManifest field count mismatch | data-flow.md (5 fields) vs renderer-architecture.md (3 fields) | data-flow.md adds `goal_summary` and `primary_flaw`. `summarize_goals()` function undefined. | Update renderer-architecture.md; define `summarize_goals()`. |
| M6 | BeliefSnapshot field asymmetry | data-flow.md vs irony-and-beliefs.md | data-flow.md has `sim_time` (producer lacks it). irony-and-beliefs.md has `pairwise_irony` (data-flow.md lacks it). | Add both fields to canonical BeliefSnapshot. |
| M7 | `classify_beats` return type mismatch | data-flow.md vs story-extraction.md | data-flow.md expects `BeatClassification[]`. story-extraction.md returns `list[BeatType]`. | Have classify_beats return event_id + beat_type pairs. |
| M8 | Thematic axes never formally enumerated | data-flow.md Section 7 (scattered) | 5 axes mentioned (order_chaos, truth_deception, loyalty_betrayal, innocence_corruption, freedom_control) but no closed enum. | Add ThematicAxis enum to events.md or a new thematic spec. |

### LOW (cosmetic / documentation)

| # | Issue | Specs Involved | Details | Resolution |
|---|-------|---------------|---------|------------|
| L1 | `format_version` declared but never produced | data-flow.md Section 11 | No interface includes it; no producer generates it. | Add to SimulationOutput and NarrativeFieldPayload interfaces. |
| L2 | `simulation_id` and `timestamp` have no producer | data-flow.md:156,163 | Integration-layer concern, not spec bug. | Document as "populated by simulation harness." |
| L3 | Non-canonical character names in pacing-physics.md | pacing-physics.md worked examples | References "Marco" and "Sophia" instead of canonical names. | Update to canonical names from dinner-party-config.md. |
| L4 | `denial_bonus` dead weight (verified #7) | irony-and-beliefs.md:360-362 | Weighted at 0.2 but always 0.0. Slightly distorts irony_collapse_score. | Either implement or redistribute weight to other components. |
| L5 | `significance` always 0.0 (verified #8) | events.md:315, story-extraction.md:377 | Phase 5 stub. Weighted at 0.15 in arc scoring. | Document in story-extraction.md that significance score is intentionally zero until Phase 5. |

---

## Enum Alignment Matrix

| Enum | events.md | scenes.md | scene-segmentation.md | story-extraction.md | tension-pipeline.md | Verdict |
|------|-----------|-----------|----------------------|--------------------|--------------------|---------|
| **EventType** (10 values) | CHAT, OBSERVE, SOCIAL_MOVE, REVEAL, CONFLICT, INTERNAL, PHYSICAL, CONFIDE, LIE, CATASTROPHE | Same (references events.md) | Same | Same | Same | ALIGNED |
| **BeatType** (5 values) | SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE | — | — | Same 5 values | — | ALIGNED |
| **DeltaKind** (9 values) | AGENT_EMOTION, AGENT_RESOURCE, AGENT_LOCATION, RELATIONSHIP, BELIEF, SECRET_STATE, WORLD_RESOURCE, COMMITMENT, PACING | — | — | — | References same set | ALIGNED |
| **DeltaOp** (2 values) | SET, ADD | — | — | — | References same set | ALIGNED |
| **SceneType** | — | 7 types: bonding, tension_building, confrontation, revelation, aftermath, transition, climax | 6 types: catastrophe, confrontation, revelation, bonding, escalation, maintenance | — | — | **DIVERGENT** (H1) |
| **FlawType** | — | — | — | — | — | agents.md defines: pride, guilt, cowardice, ambition, denial, loyalty, jealousy, obsession, vanity (ALIGNED across agents.md + dinner-party-config.md) |
| **BeliefState** | — | — | — | — | — | agents.md defines: UNKNOWN, SUSPECTS, BELIEVES_TRUE, BELIEVES_FALSE (ALIGNED across irony-and-beliefs.md) |

---

## Constant Alignment Matrix

| Constant | Authoritative Source | scenes.md | scene-segmentation.md | pacing-physics.md | Other Refs | Verdict |
|----------|---------------------|-----------|----------------------|-------------------|------------|---------|
| `participant_turnover` threshold | scene-segmentation.md (Decision 21) | **0.5** | 0.3 | — | MASTER_PLAN.md says 0.3 | **STALE in scenes.md** (M1) |
| `tension_gap_threshold` | scene-segmentation.md | 0.3 | 0.3 | — | — | ALIGNED |
| `time_gap_threshold` (minutes) | scene-segmentation.md | 5.0 | 5.0 | — | — | ALIGNED |
| `min_scene_size` (events) | scene-segmentation.md | 3 | 3 | — | — | ALIGNED |
| `irony_collapse_threshold` | irony-and-beliefs.md | — | 0.5 | — | irony-and-beliefs.md: 0.5 | ALIGNED |
| `snapshot_interval` | world.md | — | — | — | data-flow.md: 20 (unit wrong), world.md: 20 events | **UNIT WRONG in data-flow.md** (H-verified #4) |
| `catastrophe_threshold` | pacing-physics.md (Decision 19) | — | — | 0.35 | doc3.md: 0.35 | ALIGNED |
| `composure_minimum` | pacing-physics.md | — | — | 0.30 | doc3.md: 0.30 | ALIGNED |
| `stress_decay` | pacing-physics.md | — | — | 0.05 | agents.md: 0.05 | ALIGNED |
| `composure_recovery` | pacing-physics.md | — | — | 0.03 | agents.md: 0.03 | ALIGNED |
| `decision_noise_sigma` | decision-engine.md | — | — | — | 0.1 | No cross-ref | ALIGNED |
| TensionWeights (all 1.0 default) | tension-pipeline.md | — | — | — | doc3.md: all 1.0 | ALIGNED |

---

## Appendix: Authority Chain Applied

When conflicts were found, the following authority chain was used:

1. **Pacing constants**: pacing-physics.md (Decision 19) — all pacing values confirmed aligned with agents.md and doc3.md
2. **Segmentation algorithm**: scene-segmentation.md (Decision 21) — scenes.md has 2 stale values and missing trigger
3. **Character definitions**: dinner-party-config.md — pacing-physics.md has non-canonical names in worked examples
4. **Pipeline order**: irony -> thematic -> tension -> scenes (Decision 20) — confirmed in data-flow.md and MASTER_PLAN.md
5. **BeatType enum**: 5 types only (Decision 18) — confirmed consistent across events.md and story-extraction.md
