# NarrativeField Spec Audit Synthesis

> **Author:** Team Lead
> **Date:** 2026-02-06
> **Sources:** consistency-report.md, risk-report.md, contract-report.md
> **Status:** Complete

---

## Executive Summary

Three independent audits (spec consistency, implementation risk, interface contracts) examined all 17 specs + MASTER_PLAN.md + doc3.md. After deduplication and cross-referencing, the unified findings are:

| Severity | Count | Meaning |
|----------|-------|---------|
| **CRITICAL** | 7 | Blocks implementation or causes runtime failures |
| **HIGH** | 11 | Causes incorrect behavior or silent data loss |
| **MEDIUM** | 10 | Causes confusion, has workarounds |
| **LOW** | 7 | Cosmetic, documentation, or deferred-phase issues |
| **Total** | **35** | |

**Highest-risk phase:** Phase 2 (Simulation) — 4 CRITICAL + 5 HIGH issues
**Lowest-risk phase:** Phase 1 (Renderer) — 1 CRITICAL (EventMetrics gap) + 2 MEDIUM

---

## Unified Issue List (Deduplicated, Severity-Ranked)

### CRITICAL — Must fix before implementation begins

| ID | Issue | Specs | Source Reports | Phase | Owner | Resolution |
|----|-------|-------|---------------|-------|-------|------------|
| **S-01** | `select_catastrophe_type` uses `dominant_flaw.name`; field is `flaw_type` | pacing-physics.md:433 vs agents.md:135 | risk, consistency | P2 | Sim | Change `.name` → `.flaw_type` in all 4 branches |
| **S-02** | `danger` sub-metric: `context.secrets[delta.agent]` should be `context.secrets[delta.attribute]` | tension-pipeline.md:84 | risk, consistency | P3 | Metrics | Fix lookup key. `delta.agent` is the affected agent; `delta.attribute` holds the secret_id for SECRET_STATE deltas. |
| **S-03** | EventMetrics missing `tension_components` and `irony_collapse` | events.md:312-317 vs data-flow.md, renderer-architecture.md | contract (X-2, C2-2) | P1 | Schema | Add `"tension_components": {}` and `"irony_collapse": null` to Python default factory and TS EventMetrics |
| **S-04** | Three names for WorldState snapshot type (WorldStateSnapshot / SnapshotState / WorldState) | data-flow.md, scenes.md, tension-pipeline.md | contract (X-3, C2-1) | P2,P3 | Schema | Adopt `WorldStateSnapshot` name + scenes.md field set (superset). Update all 3 specs. |
| **S-05** | `content_display` vs `description` in SecretDefinition | data-flow.md vs world.md | contract (C1-3), consistency (C4) | P2 | Schema | Use `description` (world.md is producer). Update data-flow.md. |
| **S-06** | `content_metadata` field absent from Event schema | decision-engine.md:152,183,202,223 vs events.md | consistency (C5) | P2 | Schema | Add `content_metadata: Optional[dict] = None` to Event dataclass, or route through deltas. |
| **S-07** | `classify_scene_type_from_ids` called but never defined | scene-segmentation.md:313 | risk, consistency (H6) | P3 | Metrics | Define the function or change `merge_two_scenes` to pass Event objects. |

### HIGH — Causes incorrect behavior

| ID | Issue | Specs | Source Reports | Phase | Owner | Resolution |
|----|-------|-------|---------------|-------|-------|------------|
| **S-08** | Scene type enum: 7 types (scenes.md) vs 6 types (scene-segmentation.md) | scenes.md:84-90 vs scene-segmentation.md:251-265 | consistency (H1) | P3 | Schema | scene-segmentation.md is authoritative (Decision 21). Update scenes.md to its 6 types: catastrophe, confrontation, revelation, bonding, escalation, maintenance. |
| **S-09** | `Secret.holder: str` but some secrets need multiple holders | world.md:329 vs irony-and-beliefs.md | risk, consistency (H2) | P2 | Schema | Change to `holder: list[str]` or add `co_holders: list[str]`. Affects moral_cost and information_gap computations. |
| **S-10** | SecretDefinition in data-flow.md missing 4 world.md fields | data-flow.md vs world.md | contract (C1-4) | P2 | Integration | Add `initial_knowers`, `initial_suspecters`, `dramatic_weight`, `reveal_consequences` to data-flow.md. |
| **S-11** | LocationDefinition missing `overhear_probability` in data-flow.md | data-flow.md vs world.md | contract (C1-5) | P2 | Integration | Add `overhear_probability: float` and `description: str`. |
| **S-12** | IronyCollapse dataclass vs JSON format disagree | irony-and-beliefs.md:333 vs Section 7.1 | contract (C2-4), consistency (H5) | P3 | Metrics | Update dataclass: add `detected: bool`, `score: float`; change `collapsed_beliefs` to `list[dict]` with `{agent, secret, from, to}`. |
| **S-13** | `_secret_relevance` different tiers across specs | tension-pipeline.md:338-353 vs irony-and-beliefs.md:154-182 | risk, consistency (H7) | P3 | Metrics | Unify into one canonical function (recommend irony-and-beliefs.md's 6-tier version). |
| **S-14** | `snapshot_interval` says "ticks" in 2 locations | data-flow.md:162,621 | contract (X-1, C1-1) | P2 | Integration | Change "ticks" → "events" in both lines. |
| **S-15** | `event.metrics` dict-style (Python) vs typed object (TS) | scene-segmentation.md, tension-pipeline.md vs renderer-architecture.md | risk, consistency (H8) | P3 | Schema | Define Python `@dataclass EventMetrics` matching TS interface. |
| **S-16** | `world_state_before()` O(n²) via deepcopy + replay | tension-pipeline.md:746-754 | risk | P3 | Metrics | Replace with forward-replaying cached state. Single pass O(n). |
| **S-17** | MASTER_PLAN tension sub-metric names 7/8 stale | MASTER_PLAN.md:201 | consistency (H9) | — | Doc | Update to canonical: danger, time_pressure, goal_frustration, relationship_volatility, information_gap, resource_scarcity, moral_cost, irony_density. |
| **S-18** | `content_metadata` vs `metadata` field name inconsistency in decision-engine.md | decision-engine.md internal | risk | P2 | Sim | Pick one name consistently. Relates to S-06. |

### MEDIUM — Causes confusion, has workarounds

| ID | Issue | Specs | Source Reports | Phase | Owner | Resolution |
|----|-------|-------|---------------|-------|-------|------------|
| **S-19** | `participant_turnover` threshold 0.5 → 0.3 | scenes.md:126 | consistency (M1), risk | P3 | Schema | Update scenes.md:126. |
| **S-20** | SOCIAL_MOVE forced boundary missing from scenes.md | scenes.md vs scene-segmentation.md | consistency (M2), risk | P3 | Schema | Add as 6th trigger in scenes.md Section 2. |
| **S-21** | `generateSplines` references out-of-scope `agents` variable | thread-layout.md:285 | risk, consistency (M3) | P1 | Viz | Pass `agents: string[]` parameter. |
| **S-22** | renderer-architecture.md NarrativeFieldPayload missing `metadata`, `belief_snapshots` | renderer-architecture.md vs data-flow.md | contract (C3-1) | P1 | Viz | Update renderer-architecture.md. |
| **S-23** | AgentManifest 5 fields (data-flow.md) vs 3 fields (renderer) | data-flow.md vs renderer-architecture.md | contract (C3-2) | P1 | Viz | Update renderer-architecture.md; define `summarize_goals()`. |
| **S-24** | BeliefSnapshot: `sim_time` in data-flow.md only; `pairwise_irony` in irony spec only | data-flow.md vs irony-and-beliefs.md | contract (C2-3) | P3 | Metrics | Add both fields to canonical BeliefSnapshot. |
| **S-25** | `classify_beats` return type: `list[BeatType]` vs `BeatClassification[]` | story-extraction.md vs data-flow.md | contract (C4-1) | P4 | Extraction | Return `BeatClassification[]` (richer). |
| **S-26** | Thematic axes never formally enumerated | data-flow.md Section 7 (scattered) | risk, consistency (M8) | P3 | Schema | Add ThematicAxis enum or inline definitions. |
| **S-27** | PHYSICAL delta uses fragile string matching (`"drink" in content.lower()`) | tick-loop.md | risk | P2 | Sim | Use structured `content_metadata` (ties to S-06). |
| **S-28** | Pacing deltas appended post-creation (violates event immutability) | tick-loop.md | risk | P2 | Sim | Generate pacing deltas during event creation. |

### LOW — Cosmetic, documentation, deferred

| ID | Issue | Specs | Source Reports | Phase | Owner | Resolution |
|----|-------|-------|---------------|-------|-------|------------|
| **S-29** | `format_version` declared but never produced | data-flow.md Section 11 | contract (X-4, C3-3) | — | Integration | Add to interfaces. Low urgency. |
| **S-30** | `simulation_id` and `timestamp` have no producer spec | data-flow.md:156,163 | contract (C1-2) | P2 | Integration | Document as "populated by simulation harness." |
| **S-31** | Non-canonical character names in pacing-physics.md examples | pacing-physics.md | risk, consistency (L3) | P2 | Sim | Update to canonical names. |
| **S-32** | `denial_bonus = 0.0` dead weight (0.2 of irony_collapse_score) | irony-and-beliefs.md:360 | consistency (L4) | P3 | Metrics | Implement or redistribute weight. |
| **S-33** | `significance = 0.0` always zero (Phase 5 stub) | events.md, story-extraction.md | consistency (L5) | P5 | — | Document in story-extraction.md as intentional. |
| **S-34** | BFS uses `queue.shift()` / `queue.pop(0)` — O(n) | renderer-architecture.md, scenes.md | risk | P1,P3 | Various | Use proper deque. Acceptable at 200 events but good practice. |
| **S-35** | Irony pipeline duplication: irony_density re-computes vs pre-computed values | tension-pipeline.md:476-513 vs data-flow.md pipeline order | risk | P3 | Metrics | Clarify: should irony_density read pre-computed values or compute independently? |

---

## Compound Issues (Cross-Referencing)

Several issues compound each other:

1. **S-06 + S-18 + S-27:** The `content_metadata` absence from Event schema (S-06) causes the decision engine naming confusion (S-18) and forces the tick-loop to use fragile string matching (S-27). Fixing S-06 (adding the field) resolves all three.

2. **S-03 + S-15:** EventMetrics missing fields (S-03) and dict-vs-typed access (S-15) are the same root problem — the Python Event.metrics is a raw dict but should be a typed dataclass matching the TS interface.

3. **S-04 + S-14:** WorldState snapshot naming (S-04) and snapshot_interval units (S-14) are both data-flow.md problems that should be fixed together.

4. **S-08 + S-19 + S-20:** Scene type enum mismatch (S-08), stale turnover threshold (S-19), and missing SOCIAL_MOVE trigger (S-20) are all scenes.md issues. Fix scenes.md once to resolve all three.

---

## Fix Priority Order

**Pre-implementation (fix in specs before any code):**

| Priority | IDs | Description | Effort |
|----------|-----|-------------|--------|
| 1 | S-03 + S-15 | Add EventMetrics fields to events.md; create Python dataclass | 30 min |
| 2 | S-04 + S-14 | Unify WorldStateSnapshot naming + fix units | 30 min |
| 3 | S-01 | Fix `dominant_flaw.name` → `.flaw_type` | 5 min |
| 4 | S-02 | Fix danger secret lookup | 5 min |
| 5 | S-05 | Resolve `content_display` → `description` | 10 min |
| 6 | S-06 + S-18 | Add `content_metadata` to Event schema | 15 min |
| 7 | S-08 + S-19 + S-20 | Fix scenes.md: enum, threshold, SOCIAL_MOVE | 20 min |
| 8 | S-07 | Define or fix `classify_scene_type_from_ids` | 10 min |
| 9 | S-09 | Fix Secret.holder type | 15 min |
| 10 | S-10 + S-11 | Add missing fields to data-flow.md contracts | 15 min |
| 11 | S-12 | Fix IronyCollapse dataclass | 15 min |
| 12 | S-17 | Update MASTER_PLAN tension sub-metric names | 5 min |

**Total pre-implementation fix time: ~3 hours**

**During implementation:**

| Priority | IDs | Description | Phase |
|----------|-----|-------------|-------|
| 13 | S-16 | Optimize world_state_before() | P3 |
| 14 | S-13 | Unify _secret_relevance tiers | P3 |
| 15 | S-21 + S-22 + S-23 | Fix renderer spec gaps | P1 |
| 16 | S-27 + S-28 | Fix tick-loop string matching + immutability | P2 |
| 17 | S-24 + S-25 + S-26 | Fix BeliefSnapshot, classify_beats, thematic axes | P3-P4 |

---

## Phase Impact Summary

| Phase | CRITICAL | HIGH | MEDIUM | LOW | Top Concern |
|-------|----------|------|--------|-----|-------------|
| Phase 1 (Renderer) | 1 (S-03) | 0 | 3 (S-21,S-22,S-23) | 1 (S-34) | EventMetrics schema gap blocks fake data |
| Phase 2 (Simulation) | 3 (S-01,S-05,S-06) | 5 (S-09,S-10,S-11,S-14,S-18) | 2 (S-27,S-28) | 2 (S-30,S-31) | Field name bugs cause AttributeError |
| Phase 3 (Metrics) | 2 (S-02,S-07) | 5 (S-08,S-12,S-13,S-15,S-16) | 4 (S-19,S-20,S-24,S-26) | 2 (S-32,S-35) | O(n²) world_state_before bottleneck |
| Phase 4 (Extraction) | 0 | 0 | 1 (S-25) | 1 (S-33) | Cleanest phase |
| Cross-phase | 1 (S-04) | 1 (S-17) | 0 | 1 (S-29) | WorldState snapshot naming confusion |
