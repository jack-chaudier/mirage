# NarrativeField — Master Plan

> **Status:** FINAL (Planning Sprint Output)
> **Covers:** All 17 specifications produced during the planning sprint
> **Implements:** CLAUDE.md build order, doc3.md architectural decisions

---

## 1. Executive Summary

NarrativeField simulates fictional worlds and extracts the stories that emerge. This document summarizes the 17 engineering specifications produced during the planning sprint, documents decisions made beyond doc3.md, defines the implementation order, and catalogs all open questions and cross-spec reconciliation items that must be resolved before coding begins.

The specifications cover four subsystems:

| Subsystem | Specs | Key Responsibility |
|-----------|-------|--------------------|
| **Schema** | 4 specs | Canonical data structures (events, agents, world, scenes) |
| **Simulation** | 4 specs | Tick loop, decision engine, pacing physics, scenario config |
| **Visualization** | 4 specs | Renderer architecture, interactions, thread layout, fake data |
| **Metrics** | 4 specs | Tension pipeline, irony/beliefs, scene segmentation, story extraction |
| **Integration** | 1 spec | Data flow and interface contracts between all subsystems |

---

## 2. Specification Summaries

### 2.1 Schema Layer

#### `specs/schema/events.md` — Event Schema

**Decision coverage:** #1 (event-primary graph), #3 (typed deltas), #4 (discrete ticks), #14 (arc grammars)

Defines the foundational `Event` type that all subsystems consume. Includes:
- **EventType enum** (10 types): CHAT, OBSERVE, SOCIAL_MOVE, REVEAL, CONFLICT, INTERNAL, PHYSICAL, CONFIDE, LIE, CATASTROPHE
- **BeatType enum** (5 types): SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE
- **StateDelta** with DeltaKind discriminated union (9 kinds) and DeltaOp enum
- Behavioral notes table: visibility, dramatic budget cost, and expected deltas per event type
- Causal link structure (backward/forward references between events)
- Metrics attachment point (`Event.metrics: Dict[str, Any]`)
- Validation rules and JSON examples in both Python and TypeScript

#### `specs/schema/agents.md` — Agent State Schema

**Decision coverage:** #5 (pacing physics), #6 (vectorized scoring), #8 (belief catalog), #10 (catastrophe)

Defines the complete agent state model:
- **GoalVector** — 7 dimensions: safety, status, closeness, secrecy, truth_seeking, autonomy, loyalty
- **CharacterFlaw** — type, strength (0.0-1.0), trigger conditions
- **RelationshipState** — per-pair trust, affection, respect, familiarity
- **PacingState** — dramatic_budget, stress, composure, commitment, recovery_timer, suppression_count
- **BeliefState enum** — UNKNOWN, SUSPECTS, BELIEVES_TRUE, BELIEVES_FALSE
- **AgentState** — composite of all above plus location, emotional state
- All 6 dinner party characters as fully instantiated JSON examples (Section 9)
- Update rules and formulas for state transitions

#### `specs/schema/world.md` — World Schema

**Decision coverage:** #17 (MVP dinner party setting)

Defines the spatial and informational substrate:
- **Location** — id, name, privacy (0.0-1.0), capacity, adjacent locations, overhear_from, overhear_probability
- **Dinner party map** — 5 rooms with adjacency graph and overhear rules
- **Seating arrangement** at the dining table
- **Movement rules** — adjacency constraints, movement cost
- **SecretDefinition** — id, description, truth_value, holder, about, content_type, initial_knowers, initial_suspecters, dramatic_weight, reveal_consequences
- **Dinner party secrets** — 5 secrets with full metadata
- **WorldDefinition** — top-level container for locations, secrets, agents
- Validation rules

#### `specs/schema/scenes.md` — Scene & Snapshot Schema

**Decision coverage:** #2 (CQRS read layer), #15 (precomputed hover), #16 (scene segmentation)

Defines the intermediate narrative layer and CQRS infrastructure:
- **Scene** dataclass — event range, participants, location, tension_arc, tension_peak, tension_mean, dominant_theme, scene_type (7 types: bonding, tension_building, confrontation, revelation, aftermath, transition, climax)
- **Scene segmentation triggers** — location change, participant turnover, tension gap, time gap, irony collapse
- **SnapshotState** — complete world state cached every 20 events for CQRS reads
- **Index tables** (7 indices) — event_by_id, agent_timeline, location_events, event_participants, secret_events, pair_interactions, forward/backward causal links
- **CausalNeighborhood** — precomputed BFS depth-3 for O(1) hover lookup
- Validation rules: 13 for scenes, 8 for snapshots, 5 for indices

### 2.2 Simulation Layer

#### `specs/simulation/tick-loop.md` — Tick Loop

**Decision coverage:** #1 (event-primary), #2 (CQRS), #4 (discrete ticks), #10 (catastrophe), #16 (scenes)

Defines the complete lifecycle of a single simulation tick:
- **6-phase tick lifecycle:** (1) Perceive, (2) Decide, (3) Resolve conflicts, (4) Generate events, (5) Apply deltas, (6) Update indices/snapshots
- **Termination conditions** — tick limit, event limit, all-secrets-revealed, all-agents-departed, dramatic budget exhausted
- **Conflict resolution** — simultaneous action proposal, priority-based resolution for conflicting actions
- Snapshot creation every 20 events
- Worked examples for 3 different tick scenarios

#### `specs/simulation/decision-engine.md` — Decision Engine

**Decision coverage:** #6 (vectorized feature scoring), #5 (pacing physics interaction)

Defines how agents choose actions:
- **Action space** — 9 action types with generation and pruning logic
- **5-component scoring formula:** `total_score = base_utility + flaw_bias + pacing_modifier + relationship_modifier + noise`
- **base_utility** — dot product of predicted deltas (from ACTION_EFFECT_PROFILES) with agent's GoalVector
- **flaw_bias** — 8 flaw effects (overweight_status, avoid_confrontation, deny_evidence, escalate_conflict, seek_validation, self_sacrifice, fixate_on_target, overcommit) with 8 trigger conditions
- **pacing_modifier** — hard gates (budget=0 → -1000), social masking formula, stress-driven escalation, commitment consistency
- **relationship_modifier** — trust/affection scaling on cooperative vs antagonistic actions
- **noise** — Gaussian with sigma=0.1
- **metadata** field on Action for LLM-generated dialogue/description (renamed from content_metadata)
- 2 fully worked decision traces (Victor OBSERVE, Elena CONFIDE)

#### `specs/simulation/pacing-physics.md` — Pacing Physics

**Decision coverage:** #5 (pacing physics), #10 (two-parameter catastrophe)

Defines the drama regulation system:
- **6 state variables** per agent: dramatic_budget, stress, composure, commitment, recovery_timer, suppression_count
- **Update rules** — per-tick decay/recovery formulas for each variable
- **Catastrophe trigger:** `stress * commitment^2 + suppression_bonus > 0.35 AND composure < 0.30`
- **Social masking:** composure >= 0.40 in public spaces → dramatic utility halved
- **Budget drain/recharge:** dramatic actions cost budget (0.1-1.0 based on EventType), recharge at 0.08/tick
- **Stress accumulation** from conflict exposure, overhearing secrets, being lied to
- **Composure erosion** from alcohol, prolonged high stress, exhaustion
- **Commitment ratchet** — increases easily, decays only slowly below 0.50
- Hysteresis and recovery mechanics
- Worked traces showing pacing evolution across 40+ ticks

#### `specs/simulation/dinner-party-config.md` — Dinner Party Configuration

**Decision coverage:** #17 (MVP), #13 (3-tier fake data)

The narrative design document for the MVP scenario:
- **6 characters:** James Thorne (host), Elena Thorne (wife), Marcus Webb (partner), Lydia Cross (assistant), Diana Forrest (friend), Victor Hale (journalist)
- **Backstories and design rationale** for each character's role in generating drama
- **5 conflict arcs:** A (Betrayal Triangle: Thorne/Elena/Marcus), B (Investigation: Victor→Marcus), C (Loyalty Test: Lydia), D (Alliance: Diana/Marcus debt, potential), E (Thorne's Discovery, meta-arc)
- **5 secrets:** affair, embezzlement, diana_debt, lydia_knows, victor_investigation
- **Trust matrix** (6x6) with key asymmetries noted
- **Affection matrix** (6x6)
- **Initial pacing states** with rationale (Marcus starts at highest stress=0.4, commitment=0.6)
- **5 expected narrative phases:** Calm Surface, Undercurrents, Cracks Appear, The Break, Aftermath
- **Reconciliation notes** (Section 11) — documents 7 constant conflicts between agents.md and pacing-physics.md, with resolution principle: pacing-physics.md is authoritative
- **Success criteria** — 100-200 events, at least 1 catastrophe, escalation pattern, spatial variety

### 2.3 Visualization Layer

#### `specs/visualization/renderer-architecture.md` — Renderer Architecture

**Decision coverage:** #12 (flexible x-axis), #15 (precomputed hover), #16 (scenes)

Defines the frontend technical architecture:
- **Tech stack:** React + D3.js + Canvas (2D primary), Zustand state management
- **NarrativeFieldStore** — Zustand store with selectors for events, scenes, agents, tension weights, camera, selection state
- **Component tree:** App → ControlPanel + CanvasRenderer + InfoPanel + TimelineBar + BeatSheetPanel
- **4 canvas layers:** Background (scene bands), Thread (character paths), EventNode (dots), Highlight (hover/selection overlays)
- **Hit detection** via offscreen canvas with unique RGB color per event — O(1) mouse detection
- **Event type color palette** and character thread colors (Wong palette, colorblind-safe)
- **Client-side tension recomputation:** frontend receives 8 sub-metric vectors, applies TensionWeights in real-time via sliders
- **Project file structure** under src/ with store/, data/, layout/, canvas/, components/, types/, constants/, utils/
- All TypeScript interfaces defined (Event, Scene, TensionWeights, RenderedEvent, ThreadPath, CausalIndex)

#### `specs/visualization/interaction-model.md` — Interaction Model

**Decision coverage:** #15 (precomputed hover neighborhoods)

Defines all user interactions:
- **Hover** — BFS-3 causal cone highlight with backward (blue) and forward (amber) tinting, <16ms budget from precomputed neighborhoods
- **Click** — Arc crystallization sequence: 350ms total across 5 phases (ripple, solidify, relate, annotate, settle)
- **Zoom levels:** Cloud (0.1-0.4), Threads (0.4-1.5), Detail (1.5-5.0) with progressive disclosure
- **Region select** — Shift+drag, amber overlay, activates BeatSheetPanel
- **Interaction state machine:** IDLE → HOVERING → CRYSTALLIZED / SELECTING → REGION_SELECTED
- **Filter controls** — toggle character threads, event types, tension threshold
- **Keyboard shortcuts:** 1/2/3 for zoom presets, Tab for cycle characters, F for fit-all, Escape for deselect

#### `specs/visualization/thread-layout.md` — Thread Layout Algorithm

Defines the Y-axis positioning of character threads:
- **Spring-force algorithm** with 5 phases: Initialize lanes → Compute time samples → Forces per sample → Location/interaction lookup → Generate cubic splines
- **4 force types:** attraction (co-location strength=0.3), repulsion (separation strength=0.2), interaction bonus (0.5 spike at event time), lane spring (home lane pull strength=0.1)
- **Inertia** factor (0.7) for temporal smoothness
- **Minimum separation** constraint (20px)
- 3 worked examples: two characters meet, character leaves table, all-together edge case
- Performance target: ~55ms for 6 agents at 100 time samples

#### `specs/visualization/fake-data-visual-spec.md` — Fake Data Visual Spec

**Decision coverage:** #13 (3-tier fake data)

Describes what the user sees when the Phase 1 renderer loads:
- **70-event fake dataset** structured as 3 tiers: ~20 story-critical events, ~30 texture events, ~20 ambiguity events
- **Full event table** for all 70 events with time, type, characters, location, description, tension
- **Visual annotations** for each major scene: thread convergence/divergence, tension coloring, scene boundaries
- **Zoom-level snapshots:** what appears at Cloud, Threads, and Detail levels
- **Hover/click behavior** examples with specific events
- Uses canonical character names from dinner-party-config.md

### 2.4 Metrics Layer

#### `specs/metrics/tension-pipeline.md` — Tension Pipeline

**Decision coverage:** #5 (pacing physics), #6 (vectorized scoring), #8 (belief catalog), #14 (arc grammars)

Defines the post-processing tension computation:
- **8 sub-metrics** (each [0.0, 1.0]): danger, time_pressure, goal_frustration, relationship_volatility, information_gap, resource_scarcity, moral_cost, irony_density
- **TensionWeights** — user-adjustable 8-element weight vector for genre tuning
- **3 genre presets:** thriller (high danger, time_pressure, moral_cost), relationship_drama (high relationship_volatility, irony_density, information_gap), mystery (high information_gap, resource_scarcity, irony_density)
- **Computation:** post-processing pass over completed event log, O(1) per event with indexed access
- **Performance target:** <100ms for 200 events
- **Output:** `Event.metrics["tension"]` (scalar) and `Event.metrics["tension_components"]` (8-element dict)
- Worked example: 10 events with full sub-metric breakdown

#### `specs/metrics/irony-and-beliefs.md` — Irony and Beliefs

**Decision coverage:** #8 (finite proposition catalog), #5 (pacing physics — composure masks beliefs)

Defines the dramatic irony computation system:
- **Belief matrix:** `Dict[agent_id, Dict[secret_id, BeliefState]]`
- **BeliefState enum:** UNKNOWN, SUSPECTS, BELIEVES_TRUE, BELIEVES_FALSE
- **Belief transition rules** per event type (valid and invalid transitions)
- **agent_irony():** actively wrong=2.0, critical ignorance=1.0, suspects=0.25, correct=0.0
- **secret_relevance():** 1.0 (about agent) → 0.9 (relationship partner present) → 0.7 (held by agent) → 0.5 (present) → 0.2 (no connection)
- **scene_irony():** normalized sum of weighted agent irony across present agents
- **pairwise_irony():** asymmetric knowledge, contradictory beliefs, mutual ignorance
- **IronyCollapse detection:** scene_irony drop >= 0.5
- **10-event trace** showing belief matrix evolution from tick 3 to tick 42
- Scene irony evolution example: 0.95 → 0.62 (collapse at catastrophe)

#### `specs/metrics/scene-segmentation.md` — Scene Segmentation

**Decision coverage:** #16 (first-class scene segmentation)

Defines the algorithm that groups events into scenes:
- **5 boundary rules:** location change, participant turnover (Jaccard < 0.3), tension gap (>0.3 absolute change), time gap (>5 min simulated), irony collapse (>=0.5 drop)
- **SOCIAL_MOVE forced boundary** — always triggers scene break
- **Merge pass** for tiny scenes (<3 events)
- **SegmentationConfig** dataclass with tunable thresholds
- **Metrics pipeline execution order:** Irony → Thematic shifts → Tension → Scene segmentation (segmentation depends on tension and irony, runs last)
- Worked example: 20 events segmented into 5 scenes (Arrival Small Talk, Kitchen Conspiracy, Return to Table, Balcony Confrontation, The Accusation)

#### `specs/metrics/story-extraction.md` — Story Extraction

**Decision coverage:** #7 (promises as search targets), #14 (arc grammars)

Defines the pipeline from event selection to readable story:
- **4-stage pipeline:** Arc grammar validation → Beat classification → Soft scoring → Beat sheet generation → LLM prompt
- **Arc grammar (BNF):** `<arc> ::= <setup_phase> <development_phase> <climax_phase> <aftermath_phase>` with hard constraints on beat ordering
- **Beat classification rules** mapping EventType + context to BeatType
- **Soft scoring:** 5 quality dimensions (tension_arc_quality, irony_utilization, character_development, causal_density, thematic_coherence)
- **Beat sheet format:** structured intermediate representation consumed by LLM prompt
- **LLM prompt template** for Claude API — converts beat sheet to prose narrative
- **CONSEQUENCE covers closure:** The canonical 5-type BeatType enum is preserved; CONSEQUENCE handles both aftermath and resolution beats (minimum arc: 4 events)

### 2.5 Integration Layer

#### `specs/integration/data-flow.md` — Data Flow and Interface Contracts

**Decision coverage:** #1 (event-primary graph), #2 (CQRS read layer), #16 (scenes)

Defines all interfaces between subsystems:
- **System architecture diagram** (Mermaid) showing all data flows
- **5 interface contracts:**
  1. SimulationOutput → EventLog (simulation metadata, initial state, snapshots, events, secrets, locations)
  2. EventLog → MetricsPipeline (pipeline execution order: irony → thematic → tension → scenes)
  3. EventLog + Metrics → NarrativeFieldPayload (resolves renderer's open question about input shape)
  4. UserSelection → StoryExtractor (StoryExtractionRequest/Response)
  5. StoryExtractor → LLM (LLMStoryRequest/Response with prompt template)
- **IndexTables** built in a single pass from event log (7 indices)
- **Thematic shift computation rules** per EventType
- **Error handling:** validation, graceful degradation, beat sheet fallback
- **File formats:** `.nf-sim.json` (simulation output), `.nf-viz.json` (renderer payload), `.nf-beats.json` (beat sheet export)
- **format_version** field for forward compatibility
- **Performance characteristics** table for each subsystem

---

## 3. New Decisions Made During the Sprint

These decisions were made to resolve ambiguities in doc3.md or to fill gaps discovered during specification.

### Decision 18: CONSEQUENCE Covers Closure (No RESOLUTION BeatType)

**Context:** story-extraction.md initially proposed adding RESOLUTION as a 6th BeatType for aftermath/closure beats. After review, this was reverted.

**Resolution:** The canonical 5-type BeatType enum (SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE) is preserved unchanged. CONSEQUENCE covers both aftermath and closure beats. The arc grammar's aftermath phase uses CONSEQUENCE alone. Minimum valid arc is 4 events (SETUP + COMPLICATION + TURNING_POINT + CONSEQUENCE).

**Affected specs:** None — story-extraction.md was updated to remove RESOLUTION; events.md remains unchanged.

### Decision 19: Pacing-Physics Authority Principle

**Context:** agents.md and pacing-physics.md define overlapping pacing constants with different values (7 conflicts documented in dinner-party-config.md Section 11).

**Resolution:** pacing-physics.md is authoritative for ALL pacing-related constants, update rules, and behavioral formulas. agents.md defines data structure shapes only. Where they conflict, pacing-physics.md prevails.

**Specific conflicts resolved (pacing-physics.md values win):**
| Constant | agents.md/world.md | pacing-physics.md (AUTHORITATIVE) |
|---|---|---|
| Catastrophe threshold | 0.6 | 0.35 |
| Composure minimum for masking | 0.2 | 0.40 |
| Composure gate for catastrophe | 0.2 | 0.30 |
| Budget recharge rate | 0.05/tick | 0.08/tick |
| Commitment passive decay | None | 0.01/tick below 0.50 |
| Catastrophe aftermath (commitment) | Reset to 0 | +0.10 |
| Recovery timer on catastrophe | 10 ticks | 8 ticks |

### Decision 20: Metrics Pipeline Execution Order

**Context:** Multiple metrics specs depend on each other's outputs. data-flow.md needed a definitive ordering.

**Resolution:** Pipeline runs in this fixed order:
1. **Irony & Beliefs** — updates belief matrix, computes irony scores (no dependencies on other metrics)
2. **Thematic shifts** — uses event types and deltas (no metric dependencies)
3. **Tension pipeline** — consumes irony_density from step 1, thematic shifts from step 2
4. **Scene segmentation** — consumes tension scores from step 3 and irony scores from step 1

### Decision 21: Scene Segmentation Owns Algorithm Details

**Context:** scenes.md (schema spec) and scene-segmentation.md (metrics spec) both define segmentation triggers with conflicting thresholds.

**Resolution:** scenes.md defines the Scene data structure and declares that segmentation triggers exist. scene-segmentation.md is authoritative for the algorithm, thresholds, and implementation. Where they conflict:
- Participant turnover threshold: scenes.md says 0.5, **scene-segmentation.md says 0.3** (authoritative)
- SOCIAL_MOVE forced boundary: only in scene-segmentation.md (authoritative)

### Decision 22: Content Generation is LLM-Mediated

**Context:** decision-engine.md leaves open how `content_metadata` (dialogue text, action descriptions) is generated.

**Resolution:** For MVP, the decision engine produces structured action descriptions (who, what, where, why). An LLM call converts these to natural dialogue/descriptions during event generation (tick-loop Phase 4). This is a simulation-time LLM call, distinct from the story extraction LLM call. Performance budget: <500ms per LLM call, batched where possible.

### Decision 23: File Format Versioning

**Context:** data-flow.md introduces three file formats (.nf-sim.json, .nf-viz.json, .nf-beats.json).

**Resolution:** All file formats include a `format_version: string` field (semver). Readers must check version compatibility. Breaking changes increment major version. This enables offline sharing and replay of simulation runs.

---

## 4. Implementation Order

Per CLAUDE.md build order, refined with spec dependencies:

### Phase 1: Renderer with Fake Data (Visualization-first)

**Goal:** Interactive 2D thread visualization using the 70-event fake dataset.

**Prerequisite specs:** renderer-architecture.md, interaction-model.md, thread-layout.md, fake-data-visual-spec.md

**Implementation steps:**
1. Set up React + TypeScript project with Zustand store
2. Implement NarrativeFieldPayload parser (from data-flow.md Interface Contract 3)
3. Implement thread layout algorithm (spring-force Y-axis from thread-layout.md)
4. Implement 4-layer Canvas renderer (Background, Thread, EventNode, Highlight)
5. Implement hit detection via offscreen color-coded canvas
6. Build ControlPanel (tension weight sliders, character toggles, zoom)
7. Implement hover (precomputed BFS-3 causal cone highlight)
8. Implement click (arc crystallization animation, 350ms)
9. Implement zoom levels (Cloud, Threads, Detail) with progressive disclosure
10. Implement region select (Shift+drag → BeatSheetPanel)
11. Build InfoPanel (event details, character summary, scene info)
12. Create fake-data-visual-spec.md's 70-event dataset as `.nf-viz.json`
13. Integration test: load fake data → visual matches fake-data-visual-spec.md

**Key deliverable:** Working interactive visualization that matches the visual spec.

### Phase 2: Dinner Party Simulation

**Goal:** Agent simulation feeding real events to the Phase 1 renderer.

**Prerequisite specs:** tick-loop.md, decision-engine.md, pacing-physics.md, dinner-party-config.md, events.md, agents.md, world.md

**Implementation steps:**
1. Implement data model (events.md, agents.md, world.md, scenes.md dataclasses)
2. Implement PacingState update rules from pacing-physics.md
3. Implement catastrophe trigger logic (stress * commitment^2 + suppression_bonus > 0.35 AND composure < 0.30)
4. Implement decision engine — action generation, pruning, 5-component scoring
5. Implement tick loop — 6-phase lifecycle, conflict resolution, event generation
6. Implement content generation LLM integration (Decision 22)
7. Implement termination conditions
8. Implement CQRS infrastructure — snapshots every 20 events, 7 index tables
9. Implement dinner party scenario initialization from dinner-party-config.md
10. Output `.nf-sim.json` format per data-flow.md Interface Contract 1
11. Integration test: simulation → .nf-sim.json → verify 100-200 events, escalation pattern, spatial variety

**Key deliverable:** Dinner party simulation producing realistic event logs.

### Phase 3: Metrics Pipeline

**Goal:** Tension, irony, thematic shift, and scene segmentation computation.

**Prerequisite specs:** tension-pipeline.md, irony-and-beliefs.md, scene-segmentation.md

**Implementation steps:**
1. Implement belief matrix and transition rules from irony-and-beliefs.md
2. Implement irony scoring (agent_irony, secret_relevance, scene_irony, pairwise_irony)
3. Implement irony collapse detection
4. Implement thematic shift computation rules per EventType
5. Implement 8 tension sub-metrics from tension-pipeline.md
6. Implement TensionWeights aggregation and global normalization
7. Implement scene segmentation algorithm (5 boundary rules + merge pass)
8. Wire pipeline in execution order (Decision 20): irony → thematic → tension → scenes
9. Transform .nf-sim.json → .nf-viz.json (data-flow.md Interface Contract 2→3)
10. Integration test: load Phase 2 output → compute all metrics → verify tension arc shows escalation

**Key deliverable:** Complete metrics pipeline transforming simulation output into renderer-ready data.

### Phase 4: Story Extraction

**Goal:** Arc grammar validation → beat sheet → LLM prose.

**Prerequisite specs:** story-extraction.md, data-flow.md (Interface Contracts 4, 5)

**Implementation steps:**
1. Implement BeatType classifier (event context → beat type)
2. Implement arc grammar validator (BNF rules, hard constraints)
3. Implement soft scoring (5 quality dimensions)
4. Implement beat sheet generator (structured intermediate format)
5. Implement LLM prompt template and Claude API integration
6. Wire StoryExtractionRequest/Response interface (data-flow.md Contract 4)
7. Output `.nf-beats.json` format
8. Integration test: select events from Phase 3 output → extract story → verify grammar validity

**Key deliverable:** Story extraction pipeline producing readable narratives from event selections.

### Phase 5: Counterfactual Impact (Future)

**Goal:** Branch simulation and JSD scoring (Decision #9).

Not yet specified. Requires:
- Branch simulation infrastructure (re-run from snapshot with modified state)
- JSD computation over defined future summaries
- UI for comparing branched outcomes

### Phase 6: Story Queries (Future)

**Goal:** Natural language → path search with grammar constraints.

Not yet specified. Requires:
- NL query parser
- Path search algorithm over event graph
- Grammar-constrained search optimization

---

## 5. Cross-Spec Reconciliation Items

These MUST be resolved before implementation begins.

### 5.1 ~~RESOLVED: Character Name Inconsistency~~

**Status:** RESOLVED — metrics-architect updated all 4 metrics specs + data-flow.md:
- Character names aligned: Elise→Elena, Clara→Lydia, Diego→Diana, Noor→Victor
- Secret IDs aligned with dinner-party-config.md canonical IDs
- irony-and-beliefs.md: initial belief matrix rewritten, 10-event trace rewritten, irony scoring weights updated (relevant unknown 1.0→1.5 per agents.md)
- tension-pipeline.md: worked examples updated with canonical characters
- scene-segmentation.md, story-extraction.md: character references updated
- All computed values recalculated; math unchanged, only narrative framing updated

### 5.2 ~~RESOLVED: BeatType Enum~~

**Status:** RESOLVED — story-extraction.md was updated to use the canonical 5-type BeatType enum. CONSEQUENCE covers closure. No changes needed to events.md or renderer-architecture.md.

### 5.3 ~~RESOLVED: Pacing Constants in agents.md and world.md~~

**Status:** RESOLVED — schema-architect applied all fixes:
- agents.md: added `suppression_count`, updated catastrophe threshold to 0.35, composure gate to 0.30, added authority note deferring to pacing-physics.md
- world.md: kitchen privacy 0.4→0.5, foyer privacy 0.3→0.2, catastrophe_threshold default 0.6→0.35, composure_gate (renamed from composure_minimum) 0.2→0.30

### 5.4 MEDIUM: Scene Segmentation Threshold Conflict

**Severity:** MEDIUM — two specs disagree on participant_turnover threshold.

Per Decision 21, scene-segmentation.md (Jaccard < 0.3) is authoritative over scenes.md (0.5). scenes.md Section 2 must be updated.

### 5.5 MEDIUM: SOCIAL_MOVE Forced Boundary

**Severity:** MEDIUM — scenes.md doesn't mention this rule.

scene-segmentation.md adds SOCIAL_MOVE as a forced scene boundary. scenes.md should reference this (or at minimum not contradict it).

### 5.6 LOW: world.md Uses "Elena" in Description Text

**Severity:** LOW — minor naming inconsistency in SecretDefinition description field (line ~327). Should read the canonical character name format used elsewhere.

### 5.7 LOW: tension-pipeline.md `danger` Sub-Metric Bug

**Severity:** LOW — reported during review, may or may not have been fixed.

The `danger` sub-metric formula was flagged for a potential issue. Verify during implementation.

---

## 6. Open Questions

Collected from all specs. These need resolution during implementation (or before, if they affect architecture).

### From decision-engine.md

1. **Recency penalty** — Should recently-taken action types be penalized to prevent repetition? (e.g., 3 CHATs in a row feels robotic). Suggested: multiply by 0.7 for same-type repeat, 0.5 for triple repeat.

2. **Theory of Mind** — Should agents model other agents' likely actions? For MVP, agents are "socially aware but not strategic" — they read current emotional state but don't predict future moves. Post-MVP could add a ToM layer.

3. **Content generation** — How does `content_metadata` get populated? Decision 22 above resolves this for MVP (LLM-mediated), but the exact prompt template and batching strategy need specification.

### From tick-loop.md

4. **Same-tick reaction** — Can an agent react to another agent's action in the same tick? Resolution: No. Actions are proposed based on pre-tick state. Reactions happen next tick. This preserves simultaneity semantics.

5. **Downgraded causal links** — When a proposed action is modified by conflict resolution, is the causal link to the triggering event preserved? Resolution: Yes, but marked as `downgraded: true`.

6. **Witness throttling** — Should there be a limit on OBSERVE events per tick to prevent "everyone notices everything"? Resolution: Not for MVP. The 6-agent, 5-location setup naturally limits observations. Revisit if event counts are too high.

### From renderer-architecture.md

7. **NarrativeFieldPayload shape** — Resolved by data-flow.md Interface Contract 3. No longer open.

### From story-extraction.md

8. **Multi-character arc extraction** — How to handle arcs that interleave multiple character perspectives? The grammar is per-arc, but scoring should reward arcs that naturally weave perspectives. Needs work during Phase 4.

9. **LLM prompt temperature** — What temperature for prose generation? Suggested: 0.7 for narrative prose, 0.3 for summary/analysis. Needs experimentation.

### From scene-segmentation.md

10. **Adaptive thresholds** — Should segmentation thresholds adapt based on the overall tension level of the run? (A high-tension run might need higher tension_gap thresholds to avoid over-segmentation.) Deferred to post-MVP.

### From data-flow.md

11. **Streaming vs batch** — For Phase 1-3, batch processing is sufficient. Real-time visualization of an ongoing simulation would require streaming events to the renderer. Deferred to post-MVP.

12. **Replay controls** — data-flow.md mentions that .nf-sim.json enables replay, but no spec defines rewind/step-forward UI. Deferred to post-MVP.

---

## 7. Dependency Graph

```
Schema Layer (must implement first):
  events.md ← agents.md ← world.md ← scenes.md

Simulation Layer (depends on schema):
  pacing-physics.md ← decision-engine.md ← tick-loop.md
  dinner-party-config.md (depends on agents.md + world.md)

Metrics Layer (depends on schema + events):
  irony-and-beliefs.md ← tension-pipeline.md ← scene-segmentation.md
  story-extraction.md (depends on scenes.md + tension + irony + segmentation)

Visualization Layer (depends on schema):
  renderer-architecture.md ← thread-layout.md + interaction-model.md
  fake-data-visual-spec.md (depends on renderer + dinner-party-config)

Integration Layer (depends on all):
  data-flow.md (depends on simulation + metrics + visualization)
```

---

## 8. Tech Stack Summary

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Simulation engine | Python | 3.12+ | Dataclasses, type hints, event sourcing |
| Metrics pipeline | Python + NumPy + SciPy + NetworkX | Latest stable | Sub-metric computation, graph analysis |
| Visualization | TypeScript + React + D3.js | React 18+, D3 v7+ | 2D interactive thread visualization |
| State management | Zustand | Latest stable | Minimal frontend state store |
| Canvas rendering | HTML5 Canvas API | N/A | 4-layer rendering + hit detection |
| Content generation | Claude API | Latest | Dialogue/description generation (sim-time) |
| Story generation | Claude API | Latest | Beat sheet → prose (extraction-time) |
| Data format | JSON | N/A | .nf-sim.json, .nf-viz.json, .nf-beats.json |
| Optional 3D | Three.js | Latest stable | Future 3D visualization mode |

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Decision engine produces boring/repetitive behavior | HIGH | Flaw system + pacing constraints + noise. Tune via recency penalty (Open Question 1). |
| Catastrophe never triggers | HIGH | dinner-party-config.md carefully tunes initial pacing states. Marcus starts at stress=0.4, commitment=0.6. Adjust threshold if needed. |
| Catastrophe triggers too early | MEDIUM | Composure gate (0.30) prevents early catastrophes. Social masking in public spaces adds further delay. |
| LLM content generation too slow | MEDIUM | Batch LLM calls. Fallback to template-based content for non-critical events. |
| Thread layout produces visual collisions | LOW | Minimum separation constraint (20px) + lane springs. Tune force constants if needed. |
| 200 events overwhelms renderer | LOW | Canvas rendering with O(1) hit detection. Zoom levels provide progressive disclosure. Performance target validated in thread-layout.md (~55ms for 6 agents). |
| Character name inconsistency causes bugs | HIGH | MUST resolve before implementation (Section 5.1). |

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **Event** | The atomic unit of narrative. Something that happened at a point in time. |
| **Tick** | A discrete time step in the simulation. Each tick produces 0+ events. |
| **Scene** | A group of related events forming a dramatic unit. Intermediate between events and arcs. |
| **Arc** | A sequence of events/scenes that forms a complete narrative structure (setup → crisis → resolution). |
| **Beat** | An event tagged with its narrative function (BeatType). |
| **Beat sheet** | A structured summary of an arc's beats, ready for LLM prose generation. |
| **Dramatic budget** | A per-agent resource that limits dramatic actions. Drains on use, recharges over time. |
| **Stress** | Accumulated emotional pressure from conflict exposure. |
| **Composure** | Social mask / self-control. When it drops below threshold, catastrophes become possible. |
| **Commitment** | Irreversible investment in a course of action. Ratchets upward. |
| **Catastrophe** | An involuntary dramatic break (breakdown, blurting, explosion) triggered when stress + commitment overwhelm composure. |
| **Belief matrix** | A 2D structure mapping (agent, secret) → BeliefState. Tracks what each agent believes about each secret. |
| **Dramatic irony** | The gap between what the audience knows (ground truth) and what a character believes. |
| **Irony collapse** | The moment when a character learns the truth, reducing dramatic irony sharply. |
| **Tension** | A composite metric (8 sub-metrics, weighted sum) representing narrative intensity. |
| **TensionWeights** | User-adjustable weights that shift the genre lens (thriller, relationship drama, mystery). |
| **Causal cone** | The BFS-3 neighborhood of an event in the causal graph. Used for hover highlighting. |
| **NarrativeFieldPayload** | The top-level JSON structure consumed by the renderer (agents, events, scenes, metrics). |

---

## 11. Next Steps

1. ~~Resolve character name inconsistency~~ — DONE (metrics-architect updated all specs)
2. ~~Update events.md BeatType~~ — DONE (stays at 5 types, no change needed)
3. ~~Update agents.md pacing constants~~ — DONE (schema-architect applied all fixes)
4. **Update scenes.md** — Align participant_turnover threshold with scene-segmentation.md (Section 5.4)
5. **Update scenes.md** — Add SOCIAL_MOVE forced boundary reference (Section 5.5)
6. **Begin Phase 1 implementation** — Renderer with fake data
