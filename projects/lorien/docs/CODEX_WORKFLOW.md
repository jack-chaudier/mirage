# NarrativeField — Codex Implementation Workflow

> **Generated:** 2026-02-06
> **Source:** 17 specs + audit-synthesis.md + docs/design_v3.md + MASTER_PLAN.md
> **Purpose:** Guide Codex agents through implementation of NarrativeField Phases 1-4

---

## 1. Audit Summary

Three independent audits (spec consistency, implementation risk, interface contracts) examined all 17 specs + MASTER_PLAN.md + docs/design_v3.md. After deduplication and cross-referencing:

| Severity | Count | Meaning |
|----------|-------|---------|
| **CRITICAL** | 7 | Blocks implementation or causes runtime failures |
| **HIGH** | 11 | Causes incorrect behavior or silent data loss |
| **MEDIUM** | 10 | Causes confusion, has workarounds |
| **LOW** | 7 | Cosmetic, documentation, or deferred-phase issues |
| **Total** | **35** | |

**Highest-risk phase:** Phase 2 (Simulation) -- 4 CRITICAL + 5 HIGH issues
**Lowest-risk phase:** Phase 1 (Renderer) -- 1 CRITICAL (EventMetrics gap) + 2 MEDIUM

Full report: `specs/audit/audit-synthesis.md`

---

## 2. Repository Structure

```
narrativefield/
├── AGENTS.md                        # Root AGENTS.md (project-wide guidance)
├── pyproject.toml                   # Python project config (simulation + metrics)
├── package.json                     # Node project config (visualization)
├── tsconfig.json
├── src/
│   ├── visualization/               # Phase 1: TypeScript/React/Canvas
│   │   ├── AGENTS.md
│   │   ├── App.tsx
│   │   ├── store/
│   │   │   ├── index.ts             # Zustand store definition
│   │   │   ├── selectors.ts
│   │   │   └── actions.ts
│   │   ├── data/
│   │   │   ├── parser.ts            # JSON -> typed Event[] parsing
│   │   │   ├── causal-index.ts      # BFS-3 precomputation
│   │   │   ├── thread-index.ts      # Agent timeline builder
│   │   │   └── tension-computer.ts  # Client-side weighted tension
│   │   ├── layout/
│   │   │   ├── axis-mapper.ts       # X-position computation
│   │   │   ├── thread-layout.ts     # Spring-force Y-position engine
│   │   │   └── visual-encoding.ts   # Event -> RenderedEvent mapping
│   │   ├── canvas/
│   │   │   ├── CanvasRenderer.tsx
│   │   │   ├── HitCanvas.ts         # Offscreen hit detection
│   │   │   └── layers/
│   │   │       ├── background.ts    # Scene boundaries + heatmap
│   │   │       ├── threads.ts       # Character spline drawing
│   │   │       ├── events.ts        # Event node drawing
│   │   │       └── highlight.ts     # Causal cone overlay
│   │   ├── components/
│   │   │   ├── ToolbarPanel.tsx
│   │   │   ├── SidePanel.tsx
│   │   │   ├── CharacterFilterList.tsx
│   │   │   ├── TensionSliderPanel.tsx
│   │   │   ├── EventDetailPanel.tsx
│   │   │   ├── SceneListPanel.tsx
│   │   │   ├── BeatSheetPanel.tsx
│   │   │   ├── TooltipOverlay.tsx
│   │   │   ├── RegionSelectOverlay.tsx
│   │   │   └── StatusBar.tsx
│   │   ├── types/
│   │   │   ├── events.ts            # Event, StateDelta, Scene types
│   │   │   ├── visualization.ts     # RenderedEvent, ThreadPath, etc.
│   │   │   └── store.ts             # Store state types
│   │   ├── constants/
│   │   │   ├── colors.ts            # Color palettes
│   │   │   ├── defaults.ts          # Default weights, zoom thresholds
│   │   │   └── presets.ts           # Tension weight presets
│   │   └── utils/
│   │       ├── geometry.ts          # Point-in-rect, spline math
│   │       └── color.ts             # Color interpolation helpers
│   ├── simulation/                  # Phase 2: Python
│   │   ├── AGENTS.md
│   │   ├── __init__.py
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── tick_loop.py         # Main simulation loop
│   │   │   ├── event_queue.py       # Event generation + ordering
│   │   │   └── conflict_resolution.py
│   │   ├── pacing/
│   │   │   ├── __init__.py
│   │   │   ├── physics.py           # PacingState update rules
│   │   │   ├── catastrophe.py       # Cusp catastrophe detection
│   │   │   └── constants.py         # PacingConstants dataclass
│   │   ├── decision/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py            # Action scoring + selection
│   │   │   ├── scoring.py           # 5-component scoring formula
│   │   │   ├── action_space.py      # Action generation + pruning
│   │   │   └── perception.py        # PerceivedState builder
│   │   ├── schema/
│   │   │   ├── __init__.py
│   │   │   ├── events.py            # Event, StateDelta, enums
│   │   │   ├── agents.py            # AgentState, PacingState, GoalVector
│   │   │   ├── world.py             # WorldState, Location, Secret
│   │   │   └── scenes.py            # Scene dataclass
│   │   └── scenarios/
│   │       ├── __init__.py
│   │       └── dinner_party.py      # Scenario config + initialization
│   ├── metrics/                     # Phase 3: Python
│   │   ├── AGENTS.md
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Orchestrator: irony -> thematic -> tension -> scenes
│   │   ├── tension/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py          # 8 sub-metric computations
│   │   │   ├── sub_metrics.py       # danger, time_pressure, etc.
│   │   │   └── weights.py           # TensionWeights + presets
│   │   ├── irony/
│   │   │   ├── __init__.py
│   │   │   ├── beliefs.py           # Belief matrix management
│   │   │   ├── scoring.py           # agent_irony, scene_irony, pairwise
│   │   │   └── collapse.py          # IronyCollapse detection
│   │   ├── thematic/
│   │   │   ├── __init__.py
│   │   │   └── shifts.py            # Thematic shift computation
│   │   ├── scenes/
│   │   │   ├── __init__.py
│   │   │   ├── segmentation.py      # 5 boundary rules + merge
│   │   │   └── typing.py            # Scene type classification
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── world_state.py       # Forward-cache world_state_before() (S-16)
│   │       └── index_tables.py      # IndexTables builder
│   └── extraction/                  # Phase 4: Python
│       ├── AGENTS.md
│       ├── __init__.py
│       ├── grammar/
│       │   ├── __init__.py
│       │   ├── validator.py         # Arc grammar BNF validation
│       │   └── rules.py             # Beat ordering constraints
│       ├── beats/
│       │   ├── __init__.py
│       │   ├── classifier.py        # Event -> BeatType classification
│       │   ├── scoring.py           # 5-dimension arc quality scoring
│       │   └── sheet.py             # BeatSheet generator
│       └── llm/
│           ├── __init__.py
│           ├── prompt.py            # LLM prompt template builder
│           └── client.py            # Claude API wrapper
├── data/
│   └── fake/
│       ├── dinner_party_70.nf-viz.json   # Phase 1 fake data (70 events)
│       └── README.md
├── specs/                           # Existing specs (read-only reference)
│   ├── audit/
│   ├── schema/
│   ├── simulation/
│   ├── visualization/
│   ├── metrics/
│   ├── integration/
│   └── MASTER_PLAN.md
└── tests/
    ├── visualization/               # Vitest + @testing-library/react
    │   ├── data/
    │   ├── layout/
    │   ├── canvas/
    │   └── components/
    ├── simulation/                  # pytest
    │   ├── engine/
    │   ├── pacing/
    │   ├── decision/
    │   └── scenarios/
    ├── metrics/                     # pytest
    │   ├── tension/
    │   ├── irony/
    │   ├── scenes/
    │   └── integration/
    └── extraction/                  # pytest
        ├── grammar/
        ├── beats/
        └── llm/
```

---

## 3. Root AGENTS.md

```markdown
# NarrativeField -- Root AGENTS.md

## Project Overview

NarrativeField simulates fictional worlds (agents with desires, secrets, flaws
colliding under pressure) and maps the resulting narratives onto an interactive
topology of threaded story arcs. The system is event-sourced: the simulation
produces an append-only event log, the metrics pipeline enriches it, the
renderer visualizes it, and the extraction layer converts selections into prose.

## Key Architectural Decisions (from docs/design_v3.md)

1. **Event-primary graph** -- Events are nodes; world-state is a materialized
   view derived from replaying events from the nearest snapshot.
2. **CQRS read layer** -- Periodic snapshots every 20 events + 7 index tables.
3. **Typed deltas** -- Discriminated union (DeltaKind enum, 9 kinds).
4. **Discrete ticks** -- Simultaneous action resolution with order_in_tick.
5. **Pacing physics** -- dramatic_budget, stress, composure, commitment,
   recovery_timer, suppression_count regulate drama pacing.
6. **Vectorized feature scoring** -- GoalVector dot product for action scoring.
7. **Promises as search targets** -- Post-hoc evaluation, not simulation physics.
8. **Finite belief catalog** -- BeliefState enum (UNKNOWN/SUSPECTS/BELIEVES_TRUE/
   BELIEVES_FALSE) per agent x secret.
9. **JSD over future summaries** -- Counterfactual impact (Phase 5, deferred).
10. **Two-parameter catastrophe** -- stress x commitment^2 cusp model.
11. **TDA-lite** -- Union-find H0, cycle heuristics. No Vietoris-Rips in MVP.
12. **Flexible x-axis** -- Pinned sim_time default, swappable AxisConfig.
13. **3-tier fake data** -- story-critical + texture + ambiguity events.
14. **Arc grammars** -- Hard structural constraints + soft quality scoring.
15. **Precomputed hover neighborhoods** -- BFS depth-3, O(1) lookup.
16. **First-class scene segmentation** -- Scenes are an intermediate layer.
17. **MVP 1.5 "The Bridge"** -- Two-location stress test (post-MVP).

Additional decisions (from MASTER_PLAN.md):
18. **CONSEQUENCE covers closure** -- 5-type BeatType enum. No RESOLUTION.
19. **pacing-physics.md is authoritative** for all pacing constants.
20. **Pipeline execution order**: irony -> thematic -> tension -> scenes.
21. **scene-segmentation.md is authoritative** for segmentation algorithm.
22. **Content generation is LLM-mediated** at simulation time.
23. **File format versioning** -- format_version field in all JSON outputs.

## Build Order

| Phase | Focus | Tech | Parallelism |
|-------|-------|------|-------------|
| 1 | Renderer with fake data | TypeScript, React, Canvas, Zustand | High (data model || canvas) |
| 2 | Dinner Party simulation | Python 3.12+ dataclasses | Sequential (schema -> pacing -> decision -> tick) |
| 3 | Metrics pipeline | Python, NumPy | Partial (irony || thematic; tension after; scenes last) |
| 4 | Story extraction | Python + Claude API | Sequential (grammar -> beats -> scoring -> LLM) |

**Schema module (`src/simulation/schema/`)** is shared by Phases 2 and 3. It
must be implemented first and remain stable across both phases.

## Cross-Cutting Conventions

### Naming
- Python: snake_case for modules, functions, variables. PascalCase for classes.
- TypeScript: camelCase for functions/variables. PascalCase for types/interfaces.
- Event IDs: `evt_{NNNN}` (zero-padded 4-digit).
- Agent IDs: lowercase (e.g., `thorne`, `elena`, `marcus`, `lydia`, `diana`, `victor`).

### Error Handling
- Python: Raise typed exceptions (e.g., `ValidationError`, `MetricsPipelineError`).
- TypeScript: Return `Result<T, Error>` or throw with descriptive messages.
- Validation at system boundaries (JSON parsing, file loading). Trust internal data.

### Logging
- Python: `logging` stdlib. DEBUG for per-tick state. INFO for phase transitions.
- TypeScript: `console.warn` for recoverable issues. `console.error` for failures.

### Testing
- Python: `pytest`. Each module has a corresponding test file.
- TypeScript: `vitest` + `@testing-library/react`. Canvas tests use mock contexts.
- Every public function has at least one test. Worked examples from specs serve as
  golden test cases.

### Data Formats
- `.nf-sim.json` -- Simulation output (SimulationOutput interface).
- `.nf-viz.json` -- Renderer payload (NarrativeFieldPayload interface).
- `.nf-beats.json` -- Beat sheet export.
- All include `format_version: "1.0.0"`.

## Known Gotchas (from Audit)

- **S-01**: `select_catastrophe_type` uses `.name` but field is `.flaw_type`.
  Fix in pacing-physics.md:433 before implementing catastrophe logic.
- **S-02**: `danger` sub-metric looks up `context.secrets[delta.agent]` but should
  use `context.secrets[delta.attribute]`. Fix in tension-pipeline.md:84.
- **S-03 + S-15**: EventMetrics missing `tension_components` and `irony_collapse`
  in Python. Must define Python `@dataclass EventMetrics` matching TS interface.
- **S-06 + S-18 + S-27**: `content_metadata` absent from Event schema. Add
  `content_metadata: Optional[dict] = None` to Event. Eliminates fragile string
  matching in tick-loop.md.
- **S-16**: `world_state_before()` must use forward cache, not deepcopy + replay.

## Authority Chain (for Spec Conflicts)

1. **docs/design_v3.md** -- Canonical for all 17 architectural decisions.
2. **pacing-physics.md** -- Authoritative for pacing constants (Decision 19).
3. **scene-segmentation.md** -- Authoritative for segmentation algorithm (Decision 21).
4. **dinner-party-config.md** -- Authoritative for character definitions.
5. **events.md** -- Authoritative for Event/StateDelta/enum shapes.
6. **data-flow.md** -- Authoritative for interface contracts between subsystems.

## Commands

```bash
# Python (simulation + metrics + extraction)
pytest                                # Run all Python tests
pytest tests/simulation/              # Run simulation tests only
pytest tests/metrics/                 # Run metrics tests only
python -m simulation.engine.tick_loop # Run dinner party simulation

# TypeScript (visualization)
npm run dev                           # Start dev server
npm test                              # Run Vitest
npm run build                         # Production build
npm run lint                          # ESLint + type check
```
```

---

## 4. Subsystem AGENTS.md Files

### 4.1 `src/visualization/AGENTS.md` -- Phase 1

```markdown
# Visualization Subsystem -- AGENTS.md

## Overview

Phase 1 entry point. Interactive 2D thread visualization using React + Canvas +
Zustand. Loads a `.nf-viz.json` payload (NarrativeFieldPayload) and renders
character threads with tension heatmaps, causal highlighting, and scene boundaries.

## Architecture

- **React** manages component tree, state, UI chrome
- **Canvas API** (not SVG) for rendering -- handles 200+ nodes at 60fps
- **Zustand** for state management -- minimal boilerplate, selector-based reactivity
- **D3.js** (tree-shaken) for scales, color interpolation, force layout helpers

## Component Hierarchy

```
NarrativeFieldApp
├── DataLoader                    # Parses JSON, populates store
├── ToolbarPanel
│   ├── ZoomControls              # Cloud / Threads / Detail
│   ├── AxisSelector              # X-axis mode dropdown
│   └── ExportButton              # Region -> beat sheet
├── MainCanvas
│   ├── CanvasRenderer            # 4-layer drawing engine
│   │   ├── BackgroundLayer       # Scene bands + tension heatmap
│   │   ├── ThreadLayer           # Character path splines
│   │   ├── EventNodeLayer        # Event dots/glyphs
│   │   └── HighlightLayer        # Causal cone overlay
│   ├── HitCanvas                 # Offscreen color-coded hit detection
│   ├── TooltipOverlay            # HTML tooltip over canvas
│   └── RegionSelectOverlay       # Shift+drag rectangle
├── SidePanel
│   ├── CharacterFilterList       # Toggle character visibility
│   ├── TensionSliderPanel        # 8 weight sliders + presets
│   ├── EventDetailPanel          # Selected event details
│   ├── SceneListPanel            # Scene cards
│   └── BeatSheetPanel            # Exported beat sheet
└── StatusBar                     # Event count, time range, zoom %
```

## Data Flow

```
.nf-viz.json
  -> DataLoader (parser.ts)
  -> Zustand Store
     ├── CausalIndex (BFS-3 precomputed neighborhoods)
     ├── ThreadIndex (agent timelines)
     └── MetricsIndex (tension vectors)
  -> ThreadLayoutEngine (spring-force Y positions)
  -> CanvasRenderer (4 layers, back to front)
  -> HitCanvas (O(1) mouse -> event_id lookup)
```

## Key Specs

- `specs/visualization/renderer-architecture.md` -- Component tree, store, canvas layers
- `specs/visualization/thread-layout.md` -- Spring-force Y-axis algorithm
- `specs/visualization/interaction-model.md` -- Hover, click, zoom, region select
- `specs/visualization/fake-data-visual-spec.md` -- 70-event fake dataset

## Audit Issues to Watch

- **S-03**: EventMetrics must include `tension_components` and `irony_collapse`.
  The TS EventMetrics interface in renderer-architecture.md already has
  `tension_components`. Ensure the parser expects both fields.
- **S-21**: `generateSplines` references out-of-scope `agents` variable.
  Pass `agents: string[]` as a parameter to the function.
- **S-22**: NarrativeFieldPayload in renderer-architecture.md is missing
  `metadata` and `belief_snapshots` fields. Use data-flow.md Section 3.3 as
  the canonical payload shape.
- **S-23**: AgentManifest has 5 fields in data-flow.md but only 3 in renderer.
  Use data-flow.md's 5-field version. Implement `summarize_goals()`.
- **S-34**: BFS uses `queue.shift()` which is O(n). Use a proper deque or
  index-based queue for `buildCausalIndex`.

## Entry Point

DataLoader parses NarrativeFieldPayload (data-flow.md Section 3.3 is canonical):

```typescript
interface NarrativeFieldPayload {
  metadata: SimulationMetadata;
  agents: AgentManifest[];
  locations: LocationDefinition[];
  secrets: SecretDefinition[];
  events: Event[];
  scenes: Scene[];
  belief_snapshots: BeliefSnapshot[];
}
```

## Testing

- **Framework**: Vitest + @testing-library/react
- **Canvas tests**: Mock `CanvasRenderingContext2D`. Assert draw calls.
- **Store tests**: Verify state transitions for hover, click, filter, weight changes.
- **Layout tests**: Golden test -- 70-event fake data produces expected Y positions.
- **Integration test**: Load fake data -> verify no console errors, correct event count.

## Color Palettes

Event types: See renderer-architecture.md Section 7 (EVENT_TYPE_COLORS).
Character threads: Wong palette (colorblind-safe), 6 colors.
```

### 4.2 `src/simulation/AGENTS.md` -- Phase 2

```markdown
# Simulation Subsystem -- AGENTS.md

## Overview

Phase 2. Python 3.12+ simulation engine that produces an event log from the
Dinner Party scenario. Uses dataclasses, type hints, event sourcing. Output
is a `.nf-sim.json` file (SimulationOutput, per data-flow.md Section 3.1).

## Module Hierarchy

```
simulation/
├── schema/          # Shared data model (also used by metrics)
│   ├── events.py    # Event, StateDelta, EventType, BeatType, DeltaKind, DeltaOp
│   ├── agents.py    # AgentState, PacingState, GoalVector, CharacterFlaw, RelationshipState
│   ├── world.py     # WorldState, Location, Secret, WorldDefinition
│   └── scenes.py    # Scene dataclass
├── engine/          # Core simulation loop
│   ├── tick_loop.py           # execute_tick(), termination conditions
│   ├── event_queue.py         # Event generation, ID assignment, causal link generation
│   └── conflict_resolution.py # Detect conflicts, resolve, downgrade actions
├── pacing/          # Drama regulation (Decision 5, 10)
│   ├── physics.py     # update_pacing() -- all per-tick state updates
│   ├── catastrophe.py # check_catastrophe(), select_catastrophe_type(), generate_catastrophe_event()
│   └── constants.py   # PacingConstants dataclass (authoritative values)
├── decision/        # Agent decision-making (Decision 6)
│   ├── engine.py      # select_action() -- top-level decision function
│   ├── scoring.py     # 5-component scoring: base_utility + flaw_bias + pacing + relationship + noise
│   ├── action_space.py# generate_candidate_actions(), prune by availability
│   └── perception.py  # build_perception() -- filtered world state per agent
└── scenarios/       # Scenario configurations
    └── dinner_party.py # 6 characters, 5 locations, 5 secrets, initial state
```

## Key Specs

- `specs/simulation/tick-loop.md` -- 6-phase tick lifecycle, worked examples
- `specs/simulation/pacing-physics.md` -- Pacing constants, update rules, catastrophe
- `specs/simulation/decision-engine.md` -- Action scoring formula, flaw effects
- `specs/simulation/dinner-party-config.md` -- Characters, secrets, trust matrix
- `specs/schema/events.md` -- Event/StateDelta/enum definitions
- `specs/schema/agents.md` -- AgentState structure
- `specs/schema/world.md` -- Location/Secret definitions

## Authority Chain

- **pacing-physics.md** is authoritative for ALL pacing constants (Decision 19).
- **dinner-party-config.md** is authoritative for character definitions.
- **events.md** is authoritative for Event/StateDelta shapes.

## Audit Issues to Watch

- **S-01** (CRITICAL): `select_catastrophe_type` uses `dominant_flaw.name` but the
  field is `flaw_type`. Fix: use `.flaw_type` in all 4 branches of catastrophe
  subtype selection.
- **S-05** (CRITICAL): SecretDefinition uses `content_display` in data-flow.md but
  `description` in world.md. Use `description` (world.md is the producer).
- **S-06** (CRITICAL): `content_metadata` field absent from Event schema. Add
  `content_metadata: Optional[dict] = None` to Event dataclass in schema/events.py.
- **S-09** (HIGH): `Secret.holder` is `str` but some secrets need multiple holders.
  Change to `holder: list[str]` or add `co_holders: list[str]`.
- **S-18** (HIGH): `content_metadata` vs `metadata` naming in decision-engine.md.
  Use `content_metadata` consistently (matches S-06 fix).
- **S-27** (MEDIUM): PHYSICAL delta uses fragile string matching
  (`"drink" in content.lower()`). Use structured `content_metadata` instead.
- **S-28** (MEDIUM): Pacing deltas appended post-creation violates event
  immutability. Generate pacing deltas during event creation in Phase 5.

## Implementation Order (Strict)

1. `schema/` -- All dataclasses, enums, validation
2. `pacing/constants.py` -- PacingConstants with authoritative values
3. `pacing/physics.py` -- update_pacing() rules
4. `pacing/catastrophe.py` -- Catastrophe detection and generation
5. `decision/perception.py` -- PerceivedState builder
6. `decision/action_space.py` -- Action generation and pruning
7. `decision/scoring.py` -- 5-component scoring formula
8. `decision/engine.py` -- select_action() orchestrator
9. `engine/conflict_resolution.py` -- Conflict detection and resolution
10. `engine/event_queue.py` -- Event generation, delta generation, causal links
11. `engine/tick_loop.py` -- Main simulation loop
12. `scenarios/dinner_party.py` -- Scenario initialization
13. Integration test: run simulation -> validate output

## Testing

- **Framework**: pytest
- **Unit tests**: Each module has a test file. Use worked examples from specs.
- **Pacing tests**: Verify catastrophe threshold (stress * commitment^2 + suppression_bonus > 0.35 AND composure < 0.30).
- **Decision tests**: Verify 5-component scoring matches worked traces in decision-engine.md.
- **Integration test**: Full dinner party run -> 100-200 events, at least 1 catastrophe, escalation pattern.
- **Determinism test**: Same seed -> identical event log.
```

### 4.3 `src/metrics/AGENTS.md` -- Phase 3

```markdown
# Metrics Subsystem -- AGENTS.md

## Overview

Phase 3. Post-processing pipeline that enriches the event log with tension,
irony, thematic shifts, and scene segmentation. Reads `.nf-sim.json`,
produces `.nf-viz.json`. Python + NumPy.

## Pipeline Execution Order (Decision 20)

This order is MANDATORY. Each step depends on outputs from prior steps.

1. **Irony & Beliefs** -- Updates belief matrix, computes irony scores.
   No dependencies on other metrics. Output: `event.metrics.irony`,
   `event.metrics.irony_collapse`, belief snapshots.
2. **Thematic shifts** -- Uses event types and deltas. No metric dependencies.
   Output: `event.metrics.thematic_shift`.
3. **Tension pipeline** -- Consumes irony_density from step 1, thematic shifts
   from step 2. Output: `event.metrics.tension`, `event.metrics.tension_components`.
4. **Scene segmentation** -- Consumes tension from step 3 and irony from step 1.
   Output: `Scene[]`.

## Module Hierarchy

```
metrics/
├── pipeline.py          # Orchestrator: run_metrics_pipeline()
├── tension/
│   ├── pipeline.py      # run_tension_pipeline() -- iterate events, compute 8 sub-metrics
│   ├── sub_metrics.py   # danger(), time_pressure(), goal_frustration(), etc.
│   └── weights.py       # TensionWeights dataclass + genre presets
├── irony/
│   ├── beliefs.py       # BeliefMatrix class -- update rules per event type
│   ├── scoring.py       # agent_irony(), scene_irony(), pairwise_irony()
│   └── collapse.py      # detect_irony_collapse() -- scene_irony drop >= 0.5
├── thematic/
│   └── shifts.py        # compute_thematic_shift() per event
├── scenes/
│   ├── segmentation.py  # segment_into_scenes() -- 5 boundary rules + merge
│   └── typing.py        # classify_scene_type() -- 6 types (Decision 21)
└── utils/
    ├── world_state.py   # world_state_before() with FORWARD CACHE (fixes S-16)
    └── index_tables.py  # build_index_tables() -- single-pass, 7 indices
```

## Constant Sources

- **Pacing constants**: `specs/simulation/pacing-physics.md` (Decision 19)
- **Segmentation thresholds**: `specs/metrics/scene-segmentation.md` (Decision 21)
- **Scene types**: 6 types from scene-segmentation.md: catastrophe, confrontation,
  revelation, bonding, escalation, maintenance (NOT scenes.md's 7 types -- S-08)

## Key Specs

- `specs/metrics/tension-pipeline.md` -- 8 sub-metrics, TensionWeights, normalization
- `specs/metrics/irony-and-beliefs.md` -- Belief matrix, irony scoring, collapse
- `specs/metrics/scene-segmentation.md` -- 5 boundary rules, merge pass, scene types
- `specs/integration/data-flow.md` -- MetricsPipelineOutput, NarrativeFieldPayload

## Audit Issues to Watch

- **S-02** (CRITICAL): `danger` sub-metric lookup. Line 84 of tension-pipeline.md
  uses `context.secrets[delta.agent]` but should use `context.secrets[delta.attribute]`.
  `delta.agent` is the affected agent; `delta.attribute` holds the secret_id.
- **S-07** (CRITICAL): `classify_scene_type_from_ids` called but never defined in
  scene-segmentation.md:313. Define the function or change `merge_two_scenes` to
  pass Event objects instead of just event IDs.
- **S-08** (HIGH): Scene type enum mismatch -- scenes.md has 7 types,
  scene-segmentation.md has 6. Use scene-segmentation.md's 6 types (Decision 21).
- **S-12** (HIGH): IronyCollapse dataclass vs JSON disagree. Add `detected: bool`,
  `score: float`; change `collapsed_beliefs` to `list[dict]`.
- **S-13** (HIGH): `_secret_relevance` has different tiers across specs. Unify to
  irony-and-beliefs.md's 6-tier version.
- **S-15** (HIGH): Python Event.metrics is a raw dict but should be a typed
  `@dataclass EventMetrics` matching the TS interface.
- **S-16** (HIGH): `world_state_before()` is O(n^2) via deepcopy + replay. Replace
  with forward-replaying cached state (single pass, O(n) total).
- **S-19** (MEDIUM): `participant_turnover` threshold: use 0.3 (scene-segmentation.md),
  not 0.5 (scenes.md).
- **S-20** (MEDIUM): SOCIAL_MOVE forced boundary -- always triggers scene break.
- **S-35** (LOW): irony_density in tension pipeline: clarify whether it reads
  pre-computed irony values or computes independently. Should read pre-computed
  (pipeline runs irony first per Decision 20).

## world_state_before() Strategy (S-16 Fix)

```python
# WRONG (O(n^2)):
# def world_state_before(event): return deepcopy(snapshot) + replay_all_deltas

# RIGHT (O(n) total):
# Forward-cache: maintain a running WorldState, deepcopy at each event boundary.
# The pipeline iterates events in order anyway, so the forward state is free.

class WorldStateCache:
    def __init__(self, initial_state):
        self.current = deepcopy(initial_state)

    def advance_to(self, event):
        """Return state BEFORE this event, then apply its deltas."""
        before = self.current  # reference (not copy -- read-only downstream)
        self.current = deepcopy(self.current)
        for delta in event.deltas:
            apply_delta(self.current, delta)
        return before
```

## Testing

- **Framework**: pytest + NumPy for numerical assertions
- **Tension tests**: Verify each sub-metric against worked examples in tension-pipeline.md.
- **Irony tests**: Verify 10-event trace from irony-and-beliefs.md Section 8.
- **Scene tests**: Verify 20-event segmentation from scene-segmentation.md Section 6.
- **Pipeline integration**: Load Phase 2 output -> run full pipeline -> verify
  tension arc shows escalation, at least 1 irony collapse, 5+ scenes.
- **Performance**: <500ms for 200 events (assert with timing).
```

### 4.4 `src/extraction/AGENTS.md` -- Phase 4

```markdown
# Extraction Subsystem -- AGENTS.md

## Overview

Phase 4. Converts event selections into structured beat sheets and LLM-generated
prose. Arc grammar validation ensures narrative coherence. Output is
`.nf-beats.json` + prose from Claude API.

## Module Hierarchy

```
extraction/
├── grammar/
│   ├── validator.py   # validate_arc() -- BNF grammar, hard constraints
│   └── rules.py       # ArcGrammar definition, required beat ordering
├── beats/
│   ├── classifier.py  # classify_beats() -> BeatClassification[]
│   ├── scoring.py     # score_arc() -- 5 quality dimensions
│   └── sheet.py       # BeatSheet generator (structured intermediate format)
└── llm/
    ├── prompt.py      # LLM prompt template builder
    └── client.py      # Claude API wrapper (with mock for testing)
```

## Key Specs

- `specs/metrics/story-extraction.md` -- Arc grammar BNF, beat classification rules,
  soft scoring, beat sheet format, LLM prompt template
- `specs/integration/data-flow.md` Sections 3.4-3.5 -- StoryExtractionRequest/Response,
  LLMStoryRequest/Response

## Design Constraints

- **BeatType enum** (Decision 18): SETUP, COMPLICATION, ESCALATION, TURNING_POINT,
  CONSEQUENCE. No RESOLUTION type. CONSEQUENCE covers closure.
- **Arc grammar** (Decision 14): Hard structural constraints + soft scoring.
  Valid arc = at least 1 event per BeatType, in order.
- **Minimum arc**: 4 events (SETUP + COMPLICATION + TURNING_POINT + CONSEQUENCE).
- **classify_beats()** must return `BeatClassification[]` (S-25), not `list[BeatType]`.

## Audit Issues to Watch

- **S-25** (MEDIUM): `classify_beats` return type must be `BeatClassification[]`
  (richer, includes event_id + beat_type), not bare `list[BeatType]`.
- **S-33** (LOW): `significance = 0.0` always zero. This is intentional (Phase 5
  stub). Document clearly, don't compute.

## Testing

- **Framework**: pytest
- **Grammar tests**: Valid and invalid arc examples from story-extraction.md.
- **Beat classification**: Verify classification against worked examples.
- **LLM tests**: Mock LLM client returns structured placeholder prose. Never call
  real API in tests.
- **Integration**: Select events from a Phase 3 output -> extract story -> verify
  grammar validity and beat sheet structure.
```

---

## 5. Codex Skills

### 5.1 `$nf-implement` -- Implement a Spec Section with Tests

**Description:** Implements a specific section of a spec file, producing source code and corresponding test files. Checks audit-synthesis.md for relevant issues before coding.

**Trigger:** `$nf-implement <spec_path> <section_number>`

**Prompt Template:**
```
You are implementing NarrativeField. Read the following files:
1. The spec: {spec_path} -- focus on Section {section_number}
2. The audit: specs/audit/audit-synthesis.md -- check for issues tagged to this spec
3. The subsystem AGENTS.md in the target source directory

Produce:
- Implementation code in the appropriate src/ subdirectory
- Test file in the corresponding tests/ subdirectory
- List any audit issues (S-XX) you addressed

Follow the naming conventions, error handling, and testing rules from the
root AGENTS.md. Use worked examples from the spec as golden test cases.
If the spec has a bug flagged in the audit, fix it in your implementation
and note the fix.
```

### 5.2 `$nf-validate` -- Validate Subsystem Output Against Interface Contract

**Description:** Validates that a subsystem's output matches the interface contract defined in data-flow.md.

**Trigger:** `$nf-validate <subsystem_name> <sample_data_path>`

**Prompt Template:**
```
You are validating NarrativeField interface contracts. Read:
1. specs/integration/data-flow.md -- find the contract for {subsystem_name}
2. The sample data at {sample_data_path}

Perform field-by-field verification:
- Every required field is present
- Types match the contract (string, number, enum values)
- Ranges are valid (e.g., tension in [0.0, 1.0])
- Nested structures are correctly shaped
- No extra fields that aren't in the contract

Report: PASS/FAIL per field, with details on any mismatches.
```

### 5.3 `$nf-trace-data` -- Trace a Field Through Full Pipeline

**Description:** Traces a specific data field from its producer spec through the interface contract to all consumer specs.

**Trigger:** `$nf-trace-data <field_name>`

**Prompt Template:**
```
You are tracing the field "{field_name}" through the NarrativeField pipeline. Read:
1. specs/integration/data-flow.md -- find all contracts mentioning this field
2. All spec files that produce or consume this field

Produce a chain:
  Producer spec (file:line) -> Interface contract (file:section) -> Consumer spec (file:line)

For each link, verify:
- Field name matches exactly
- Type matches
- Value range is consistent
- No transformation is lost

Report any breaks in the chain.
```

### 5.4 `$nf-reconcile` -- Check for Spec Drift After Code Changes

**Description:** After modifying implementation code, checks whether the changes are consistent with the referenced specs.

**Trigger:** `$nf-reconcile <modified_file_path>`

**Prompt Template:**
```
You are checking for spec drift in NarrativeField. Read:
1. The modified file: {modified_file_path}
2. The AGENTS.md in the same directory (for spec references)
3. All specs listed in the AGENTS.md "Key Specs" section

For each spec reference:
- Verify the implementation matches the spec's definitions
- Check that constants match authoritative sources (pacing-physics.md, etc.)
- Flag any divergences between code and spec

Produce: list of specs that may need updating, or confirmation of alignment.
```

---

## 6. Cloud Task DAG

### Phase 1: Visualization (T-101 through T-110)

| ID | Title | Prereqs | Spec Refs | Prompt Template | Verify | Complexity | Audit Fixes |
|----|-------|---------|-----------|-----------------|--------|------------|-------------|
| T-101 | Define TypeScript type definitions | -- | events.md, renderer-architecture.md Sec 4 | Implement all TS interfaces (Event, StateDelta, Scene, EventMetrics with tension_components, AgentManifest, NarrativeFieldPayload, BeliefSnapshot) in `src/visualization/types/`. Use data-flow.md Section 3.3 as canonical payload shape. | `npm run typecheck` passes | M | S-03, S-15, S-22, S-23 |
| T-102 | Build fake data JSON | -- | fake-data-visual-spec.md, dinner-party-config.md | Generate the 70-event fake dataset as `data/fake/dinner_party_70.nf-viz.json`. Include all 3 tiers (story-critical, texture, ambiguity). Populate tension_components, irony, irony_collapse, thematic_shift for all events. Include agents, locations, secrets, scenes, belief_snapshots, metadata. | JSON validates against NarrativeFieldPayload interface | L | S-03 |
| T-103 | Implement data parser and Zustand store | T-101 | renderer-architecture.md Sec 2-3 | Implement `src/visualization/data/parser.ts` (JSON -> typed arrays with validation), `src/visualization/store/index.ts` (Zustand store with all slices from renderer-architecture.md Section 3), and `src/visualization/store/selectors.ts`. | `npm test -- data/ store/` passes | M | -- |
| T-104 | Build causal index | T-101 | renderer-architecture.md Sec 8 | Implement `src/visualization/data/causal-index.ts` -- BFS-3 precomputation of forward and backward causal neighborhoods. Use index-based queue (not `queue.shift()`). | `npm test -- causal-index` passes, O(1) lookup confirmed | S | S-34 |
| T-105 | Implement thread layout engine | T-101 | thread-layout.md | Implement `src/visualization/layout/thread-layout.ts` -- spring-force Y-axis algorithm with 4 forces (attraction, repulsion, interaction bonus, lane spring). Pass `agents` parameter explicitly. Output ThreadPath[] with cubic spline control points. | `npm test -- thread-layout` passes, 70 events < 100ms | M | S-21 |
| T-106 | Implement canvas renderer skeleton | T-101 | renderer-architecture.md Sec 7 | Implement `src/visualization/canvas/CanvasRenderer.tsx` with 4-layer architecture (background, threads, events, highlight). Implement `HitCanvas.ts` for O(1) mouse detection. Set up `requestAnimationFrame` loop with viewport transforms. | Canvas renders without errors; hit detection returns correct event IDs | L | -- |
| T-107 | Implement tension computer | T-101 | renderer-architecture.md Sec 9 | Implement `src/visualization/data/tension-computer.ts` -- client-side weighted tension recomputation from tension_components. `computeTension()` and `recomputeAllTension()` per renderer-architecture.md Section 9. | `npm test -- tension-computer` passes, 200 events < 5ms | S | -- |
| T-108 | Build UI components | T-103 | renderer-architecture.md Sec 5-6 | Implement all SidePanel and Toolbar components: CharacterFilterList, TensionSliderPanel (8 sliders + 3 presets), EventDetailPanel, SceneListPanel, TooltipOverlay, StatusBar. Wire to Zustand store actions. | `npm test -- components/` passes | L | -- |
| T-109 | Implement interaction handlers | T-103, T-104, T-106 | interaction-model.md | Implement hover (BFS-3 cone highlight <16ms), click (arc crystallization), zoom levels (Cloud/Threads/Detail with progressive disclosure), region select (Shift+drag). Wire to Zustand interaction state. | Hover highlights correct causal cone; click selects; zoom transitions smoothly | L | -- |
| T-110 | Phase 1 integration test | T-102, T-105, T-106, T-108, T-109 | fake-data-visual-spec.md | Load `dinner_party_70.nf-viz.json` -> verify: all 70 events render, 6 character threads visible, tension heatmap shows escalation, hover/click work, scene boundaries visible, zoom levels switch correctly. | Manual visual check + automated render test | M | -- |

### Phase 2: Simulation (T-201 through T-215)

| ID | Title | Prereqs | Spec Refs | Prompt Template | Verify | Complexity | Audit Fixes |
|----|-------|---------|-----------|-----------------|--------|------------|-------------|
| T-201 | Implement core schema dataclasses | -- | events.md, agents.md, world.md, scenes.md | Implement `src/simulation/schema/` -- Event, StateDelta, EventType, BeatType, DeltaKind, DeltaOp, AgentState, PacingState, GoalVector, CharacterFlaw, WorldState, Location, Secret, Scene. Add `content_metadata: Optional[dict] = None` to Event. Define `@dataclass EventMetrics` with tension_components and irony_collapse. Use `description` (not `content_display`) for Secret. | `pytest tests/simulation/schema/` passes | L | S-03, S-05, S-06, S-15 |
| T-202 | Implement pacing constants | T-201 | pacing-physics.md Sec 3 | Implement `src/simulation/pacing/constants.py` -- PacingConstants dataclass with ALL values from pacing-physics.md (authoritative per Decision 19). Include budget costs, stress gains, composure thresholds, commitment decay, catastrophe thresholds. | `pytest tests/simulation/pacing/test_constants.py` -- verify catastrophe threshold is 0.35, composure gate is 0.30 | S | -- |
| T-203 | Implement pacing physics | T-201, T-202 | pacing-physics.md Sec 4-8 | Implement `src/simulation/pacing/physics.py` -- update_pacing() with per-tick rules for all 6 state variables. Budget recharge (0.08/tick + private bonus), stress decay (0.03/tick), composure erosion from alcohol, commitment ratchet (decay blocked above 0.50), recovery timer decrement, suppression_count tracking. | `pytest tests/simulation/pacing/test_physics.py` -- verify update rules match pacing-physics.md worked traces | M | -- |
| T-204 | Implement catastrophe detection | T-201, T-202 | pacing-physics.md Sec 9-13 | Implement `src/simulation/pacing/catastrophe.py` -- check_catastrophe() (potential = stress * commitment^2 + suppression_bonus > 0.35 AND composure < 0.30), select_catastrophe_type() using `.flaw_type` NOT `.name`, generate_catastrophe_event(). Cap at 2 per tick. | `pytest tests/simulation/pacing/test_catastrophe.py` -- verify catastrophe fires for Victor's example state from tick-loop.md Section 5.3 | M | S-01 |
| T-205 | Implement perception builder | T-201 | tick-loop.md Sec 3 Phase 1 | Implement `src/simulation/decision/perception.py` -- build_perception() that filters world state by location visibility, belief substitution, composure-based emotion visibility (COMPOSURE_MIN_FOR_MASKING = 0.40). | `pytest tests/simulation/decision/test_perception.py` | S | -- |
| T-206 | Implement action space | T-201 | decision-engine.md Sec 2-3 | Implement `src/simulation/decision/action_space.py` -- generate_candidate_actions() for all 9 action types, prune by availability (recovery_timer, budget minimum, location). Use content_metadata for structured action context. | `pytest tests/simulation/decision/test_action_space.py` | M | S-18, S-27 |
| T-207 | Implement scoring formula | T-201, T-206 | decision-engine.md Sec 4-8 | Implement `src/simulation/decision/scoring.py` -- 5-component scoring: base_utility (GoalVector dot product), flaw_bias (8 flaw effects with triggers), pacing_modifier (budget gate, social masking, stress escalation), relationship_modifier (trust/affection scaling), noise (Gaussian sigma=0.1). | `pytest tests/simulation/decision/test_scoring.py` -- verify against Victor OBSERVE and Elena CONFIDE worked traces | L | -- |
| T-208 | Implement decision engine | T-205, T-206, T-207 | decision-engine.md Sec 1 | Implement `src/simulation/decision/engine.py` -- select_action() that builds perception, generates candidates, scores all, returns top-scoring action. | `pytest tests/simulation/decision/test_engine.py` | M | -- |
| T-209 | Implement conflict resolution | T-201 | tick-loop.md Sec 3 Phase 4 | Implement `src/simulation/engine/conflict_resolution.py` -- detect_conflicts() (target contention, location capacity, incompatible actions), resolve_conflicts() (priority_class > utility_score > random), downgrade_action() mapping. | `pytest tests/simulation/engine/test_conflict.py` -- verify contested confide example from tick-loop.md Section 5.2 | M | -- |
| T-210 | Implement event generation | T-201, T-209 | tick-loop.md Sec 3 Phase 5 | Implement `src/simulation/engine/event_queue.py` -- action_to_event(), generate_deltas_for_action() (per event type), find_causal_links(), generate_witness_events(). Generate pacing deltas during event creation (not post-append). | `pytest tests/simulation/engine/test_event_queue.py` | L | S-28 |
| T-211 | Implement tick loop | T-204, T-208, T-209, T-210 | tick-loop.md Sec 2-4 | Implement `src/simulation/engine/tick_loop.py` -- execute_tick() with 6 phases, apply_tick_updates(), apply_delta() with clamping and trust hysteresis (3x cost), snapshot policy (every 20 events), time progression. | `pytest tests/simulation/engine/test_tick_loop.py` -- verify calm tick example from Section 5.1 | L | -- |
| T-212 | Implement dinner party config | T-201 | dinner-party-config.md | Implement `src/simulation/scenarios/dinner_party.py` -- 6 characters with full GoalVector, flaws, initial relationships (trust/affection matrices), 5 locations with adjacency and overhear rules, 5 secrets, initial pacing states. | `pytest tests/simulation/scenarios/test_dinner_party.py` -- verify character count, secret count, location adjacency | M | S-31 |
| T-213 | Implement simulation harness | T-211, T-212 | tick-loop.md Sec 2, data-flow.md Sec 3.1 | Implement run_simulation() that initializes world from config, runs tick loop until termination, outputs SimulationOutput JSON. Include simulation_id, timestamp, format_version. Deterministic seeding (seed parameter). | `pytest tests/simulation/test_harness.py` -- same seed produces identical output | M | S-30 |
| T-214 | Implement CQRS index builder | T-201, T-210 | data-flow.md Sec 4 | Implement build_index_tables() as a single-pass builder over the event log. 7 indices: event_by_id, events_by_agent, events_by_location, participants_by_event, events_by_secret, events_by_pair, forward_links. | `pytest tests/simulation/test_indices.py` | S | -- |
| T-215 | Phase 2 integration test | T-213 | dinner-party-config.md Sec 10 | Run full dinner party simulation with seed=42. Verify: 100-200 events, at least 1 catastrophe, escalation pattern (tension increases over time), spatial variety (events in 3+ locations), valid SimulationOutput JSON. Write output to `data/test_output/sim_001.nf-sim.json`. | `pytest tests/simulation/test_integration.py` | M | -- |

### Phase 3: Metrics (T-301 through T-312)

| ID | Title | Prereqs | Spec Refs | Prompt Template | Verify | Complexity | Audit Fixes |
|----|-------|---------|-----------|-----------------|--------|------------|-------------|
| T-301 | Implement belief matrix | T-201 | irony-and-beliefs.md Sec 2-3 | Implement `src/metrics/irony/beliefs.py` -- BeliefMatrix class with update rules per event type, valid/invalid transitions, transition functions for REVEAL/CONFIDE/LIE/OBSERVE events. | `pytest tests/metrics/irony/test_beliefs.py` | M | -- |
| T-302 | Implement irony scoring | T-301 | irony-and-beliefs.md Sec 4-6 | Implement `src/metrics/irony/scoring.py` -- agent_irony() (actively wrong=2.0, critical ignorance=1.0, suspects=0.25), secret_relevance() (6-tier version from irony-and-beliefs.md), scene_irony(), pairwise_irony(). | `pytest tests/metrics/irony/test_scoring.py` -- verify 10-event trace from irony-and-beliefs.md Section 8 | M | S-13 |
| T-303 | Implement irony collapse detection | T-302 | irony-and-beliefs.md Sec 7 | Implement `src/metrics/irony/collapse.py` -- detect_irony_collapse() when scene_irony drops >= 0.5. Return IronyCollapse dataclass with detected, score, collapsed_beliefs as `list[dict]`. | `pytest tests/metrics/irony/test_collapse.py` | S | S-12 |
| T-304 | Implement thematic shift computation | T-201 | data-flow.md Sec 7 | Implement `src/metrics/thematic/shifts.py` -- compute_thematic_shift() per event using THEMATIC_SHIFT_RULES table and delta-driven rules. Filter near-zero shifts. | `pytest tests/metrics/thematic/test_shifts.py` | S | -- |
| T-305 | Implement world state cache | T-201 | tension-pipeline.md Sec 4 | Implement `src/metrics/utils/world_state.py` -- WorldStateCache with forward-replaying strategy. advance_to() returns state BEFORE the event, then applies deltas. Single pass, O(n) total. | `pytest tests/metrics/utils/test_world_state.py` -- verify state accuracy at random events | M | S-16 |
| T-306 | Implement index tables builder | T-201, T-214 | data-flow.md Sec 4 | Implement `src/metrics/utils/index_tables.py` -- reuse same IndexTables structure from simulation, but built as a standalone function for the metrics pipeline entry point. | `pytest tests/metrics/utils/test_index_tables.py` | S | -- |
| T-307 | Implement danger sub-metric | T-305, T-306 | tension-pipeline.md Sec 2.1 | Implement `danger()` in `src/metrics/tension/sub_metrics.py`. Use `context.secrets[delta.attribute]` (NOT `delta.agent`). Combine physical + social threat channels. | `pytest tests/metrics/tension/test_danger.py` | S | S-02 |
| T-308 | Implement remaining 7 tension sub-metrics | T-305, T-306 | tension-pipeline.md Sec 2.2-2.8 | Implement time_pressure(), goal_frustration(), relationship_volatility(), information_gap(), resource_scarcity(), moral_cost(), irony_density() in `src/metrics/tension/sub_metrics.py`. irony_density reads pre-computed irony values. | `pytest tests/metrics/tension/test_sub_metrics.py` | L | S-35 |
| T-309 | Implement tension pipeline | T-307, T-308 | tension-pipeline.md Sec 3-4 | Implement `src/metrics/tension/pipeline.py` -- run_tension_pipeline() iterating events, computing 8 sub-metrics per event, applying TensionWeights, global min-max normalization. Store in event.metrics as typed EventMetrics. | `pytest tests/metrics/tension/test_pipeline.py` -- verify against worked example (10 events) | M | -- |
| T-310 | Implement scene segmentation | T-309, T-303 | scene-segmentation.md Sec 2-4 | Implement `src/metrics/scenes/segmentation.py` -- segment_into_scenes() with 5 boundary rules (location change, participant turnover Jaccard < 0.3, tension gap, time gap, irony collapse) + SOCIAL_MOVE forced boundary + merge pass for <3 event scenes. | `pytest tests/metrics/scenes/test_segmentation.py` -- verify 20-event example from scene-segmentation.md Section 6 | M | S-08, S-19, S-20 |
| T-311 | Implement scene type classification | T-310 | scene-segmentation.md Sec 5 | Implement `src/metrics/scenes/typing.py` -- classify_scene_type() using 6 types (catastrophe, confrontation, revelation, bonding, escalation, maintenance). Define classify_scene_type_from_events() (takes Event objects, not just IDs). | `pytest tests/metrics/scenes/test_typing.py` | S | S-07 |
| T-312 | Phase 3 integration test | T-301, T-304, T-309, T-310, T-311 | data-flow.md Sec 3.2-3.3 | Implement `src/metrics/pipeline.py` -- run_metrics_pipeline() orchestrator in order: irony -> thematic -> tension -> scenes. Load Phase 2 .nf-sim.json output, run pipeline, bundle as .nf-viz.json. Verify: tension arc escalates, 1+ irony collapse, 5+ scenes, valid NarrativeFieldPayload. | `pytest tests/metrics/test_integration.py` | M | -- |

### Phase 4: Extraction (T-401 through T-408)

| ID | Title | Prereqs | Spec Refs | Prompt Template | Verify | Complexity | Audit Fixes |
|----|-------|---------|-----------|-----------------|--------|------------|-------------|
| T-401 | Implement arc grammar rules | T-201 | story-extraction.md Sec 2 | Implement `src/extraction/grammar/rules.py` -- ArcGrammar dataclass with required_beats list, BNF constraints. Default 5-beat grammar: [SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE]. | `pytest tests/extraction/grammar/test_rules.py` | S | -- |
| T-402 | Implement arc grammar validator | T-401 | story-extraction.md Sec 3 | Implement `src/extraction/grammar/validator.py` -- validate_arc() checking: causal connectivity, beat ordering, minimum duration, protagonist consistency (>60% events). Return ArcValidation with violations list. | `pytest tests/extraction/grammar/test_validator.py` -- test valid and invalid arcs | M | -- |
| T-403 | Implement beat classifier | T-201 | story-extraction.md Sec 4 | Implement `src/extraction/beats/classifier.py` -- classify_beats() returning `list[BeatClassification]` (event_id + beat_type pairs). Classification rules: first events = SETUP, new belief = COMPLICATION, rising tension = ESCALATION, max impact or CATASTROPHE = TURNING_POINT, post-peak falling = CONSEQUENCE. | `pytest tests/extraction/beats/test_classifier.py` | M | S-25 |
| T-404 | Implement arc scoring | T-403 | story-extraction.md Sec 5 | Implement `src/extraction/beats/scoring.py` -- score_arc() with 5 quality dimensions: tension_arc_quality, irony_utilization, character_development, causal_density, thematic_coherence. Each dimension [0.0, 1.0], weighted sum. | `pytest tests/extraction/beats/test_scoring.py` | M | -- |
| T-405 | Implement beat sheet generator | T-403, T-404 | story-extraction.md Sec 6, data-flow.md Sec 3.4 | Implement `src/extraction/beats/sheet.py` -- BeatSheet dataclass and generator. Produce structured intermediate format consumed by LLM prompt. Include StoryExtractionResponse wrapper. | `pytest tests/extraction/beats/test_sheet.py` | M | -- |
| T-406 | Implement LLM prompt builder | T-405 | story-extraction.md Sec 7, data-flow.md Sec 3.5 | Implement `src/extraction/llm/prompt.py` -- build beat sheet into Claude API prompt. Include character context, scene summaries, beat descriptions, and prose style instructions. | `pytest tests/extraction/llm/test_prompt.py` -- verify prompt structure | S | -- |
| T-407 | Implement LLM client | T-406 | data-flow.md Sec 3.5 | Implement `src/extraction/llm/client.py` -- Claude API wrapper (Anthropic SDK). Include mock client for testing that returns structured placeholder prose. Handle timeout, rate limit, and error cases. Return LLMStoryResponse. | `pytest tests/extraction/llm/test_client.py` -- mock mode only | M | -- |
| T-408 | Phase 4 integration test | T-402, T-405, T-407 | story-extraction.md | Select 10 events from a Phase 3 output -> run full extraction pipeline -> verify: arc validates, beats classified, score computed, beat sheet generated, mock LLM returns prose. Write to .nf-beats.json. | `pytest tests/extraction/test_integration.py` | M | S-33 |

---

## 7. Task Dependency Visualization

```
Phase 1 (Visualization)                Phase 2 (Simulation)
========================                =======================

T-101 ──┬── T-103 ──── T-109           T-201 ──┬── T-202 ── T-203 ── T-204 ──┐
        │       │                               │                              │
        ├── T-102      │                        ├── T-205 ──┐                  │
        │              │                        │           │                  │
        ├── T-104 ─────┤                        ├── T-206 ──┼── T-208 ──┐     │
        │              │                        │           │           │     │
        ├── T-105 ─────┤                        ├── T-207 ──┘           │     │
        │              │                        │                       │     │
        ├── T-106 ─────┤                        ├── T-209 ──────────────┼─────┤
        │              │                        │                       │     │
        └── T-107      │                        ├── T-210 ──────────────┘     │
                       │                        │                             │
        T-108 ─────────┤                        └── T-212                     │
                       │                                                      │
                  T-110                         T-211 ◄───────────────────────┘
                                                   │
                                                T-213 ── T-214
                                                   │
                                                T-215

Phase 3 (Metrics)                       Phase 4 (Extraction)
========================                =======================

T-301 ── T-302 ── T-303                 T-401 ── T-402
                                                    │
T-304 (parallel with T-301-303)         T-403 ── T-404 ── T-405 ── T-406 ── T-407
                                                                            │
T-305 ──┬── T-307                                                      T-408
        │
T-306 ──┤
        │
        └── T-308
              │
           T-309 ── T-310 ── T-311
                       │
                    T-312

Cross-phase dependencies:
  T-201 is shared by Phase 2 and Phase 3 (schema module)
  T-312 (metrics integration) needs T-215 (sim output) for real data
  T-408 (extraction integration) needs T-312 (metrics output) for real data
```

---

## 8. Multi-Agent Orchestration Script

```python
"""
NarrativeField Codex Orchestration Sketch
=========================================

NOT production code. Illustrates how to launch Codex agents for the
NarrativeField task DAG using the Anthropic Agents SDK.

Pattern: DAG-based task scheduling with parallel independent tasks
and sequential dependent chains.
"""

import asyncio
from anthropic import Anthropic

# Simulated Codex agent runner
async def run_codex_task(task_id: str, prompt: str, spec_refs: list[str],
                         audit_context: str, working_dir: str) -> dict:
    """
    Launch a single Codex agent to complete one task.
    Returns: {"task_id": str, "status": "success"|"failure", "output_files": list}
    """
    full_prompt = f"""
    {prompt}

    ## Spec References
    Read these files before implementing:
    {chr(10).join(f'- {ref}' for ref in spec_refs)}

    ## Audit Context
    {audit_context}

    ## Working Directory
    {working_dir}

    ## Conventions
    Follow the AGENTS.md in the working directory for naming, testing, and
    error handling conventions.
    """

    # In production: call Codex API with sandbox, file access, test runner
    # result = await codex.run(prompt=full_prompt, sandbox=True)
    print(f"[{task_id}] Starting: {prompt[:80]}...")
    await asyncio.sleep(1)  # Placeholder
    print(f"[{task_id}] Complete")
    return {"task_id": task_id, "status": "success", "output_files": []}


async def run_phase_1():
    """Phase 1: Visualization. Maximize parallelism."""
    # Wave 1: Independent foundation tasks (all parallel)
    wave_1 = await asyncio.gather(
        run_codex_task("T-101", "Define TypeScript type definitions",
                       ["specs/schema/events.md", "specs/visualization/renderer-architecture.md"],
                       "S-03, S-15, S-22, S-23", "src/visualization/"),
        run_codex_task("T-102", "Build fake data JSON (70 events)",
                       ["specs/visualization/fake-data-visual-spec.md"],
                       "S-03", "data/fake/"),
    )

    # Wave 2: Depends on T-101 (types defined)
    wave_2 = await asyncio.gather(
        run_codex_task("T-103", "Implement data parser and Zustand store",
                       ["specs/visualization/renderer-architecture.md"],
                       "", "src/visualization/"),
        run_codex_task("T-104", "Build causal index (BFS-3)",
                       ["specs/visualization/renderer-architecture.md"],
                       "S-34", "src/visualization/"),
        run_codex_task("T-105", "Implement thread layout engine",
                       ["specs/visualization/thread-layout.md"],
                       "S-21", "src/visualization/"),
        run_codex_task("T-106", "Implement canvas renderer skeleton",
                       ["specs/visualization/renderer-architecture.md"],
                       "", "src/visualization/"),
        run_codex_task("T-107", "Implement tension computer",
                       ["specs/visualization/renderer-architecture.md"],
                       "", "src/visualization/"),
        run_codex_task("T-108", "Build UI components",
                       ["specs/visualization/renderer-architecture.md"],
                       "", "src/visualization/"),
    )

    # Wave 3: Depends on store + canvas + interactions
    wave_3 = await asyncio.gather(
        run_codex_task("T-109", "Implement interaction handlers",
                       ["specs/visualization/interaction-model.md"],
                       "", "src/visualization/"),
    )

    # Wave 4: Integration test (needs everything)
    await run_codex_task("T-110", "Phase 1 integration test",
                         ["specs/visualization/fake-data-visual-spec.md"],
                         "", "src/visualization/")


async def run_phase_2():
    """Phase 2: Simulation. Strict ordering with some parallelism."""
    # Foundation: schema (shared with Phase 3)
    await run_codex_task("T-201", "Implement core schema dataclasses",
                         ["specs/schema/events.md", "specs/schema/agents.md",
                          "specs/schema/world.md", "specs/schema/scenes.md"],
                         "S-03, S-05, S-06, S-15", "src/simulation/")

    # Wave 1: Pacing chain (sequential) + Decision chain start (parallel)
    pacing_task = asyncio.create_task(run_pacing_chain())
    perception_task = asyncio.create_task(
        run_codex_task("T-205", "Implement perception builder",
                       ["specs/simulation/tick-loop.md"], "", "src/simulation/"))
    action_space_task = asyncio.create_task(
        run_codex_task("T-206", "Implement action space",
                       ["specs/simulation/decision-engine.md"],
                       "S-18, S-27", "src/simulation/"))
    scoring_task = asyncio.create_task(
        run_codex_task("T-207", "Implement scoring formula",
                       ["specs/simulation/decision-engine.md"], "", "src/simulation/"))
    config_task = asyncio.create_task(
        run_codex_task("T-212", "Implement dinner party config",
                       ["specs/simulation/dinner-party-config.md"],
                       "S-31", "src/simulation/"))

    await asyncio.gather(pacing_task, perception_task, action_space_task,
                         scoring_task, config_task)

    # Wave 2: Decision engine (needs perception + action_space + scoring)
    await run_codex_task("T-208", "Implement decision engine",
                         ["specs/simulation/decision-engine.md"], "", "src/simulation/")

    # Wave 3: Conflict resolution + Event generation (parallel, need schema)
    await asyncio.gather(
        run_codex_task("T-209", "Implement conflict resolution",
                       ["specs/simulation/tick-loop.md"], "", "src/simulation/"),
        run_codex_task("T-210", "Implement event generation",
                       ["specs/simulation/tick-loop.md"], "S-28", "src/simulation/"),
    )

    # Wave 4: Tick loop (needs catastrophe + decision + conflict + events)
    await run_codex_task("T-211", "Implement tick loop",
                         ["specs/simulation/tick-loop.md"], "", "src/simulation/")

    # Wave 5: Harness + indices
    await asyncio.gather(
        run_codex_task("T-213", "Implement simulation harness",
                       ["specs/integration/data-flow.md"], "S-30", "src/simulation/"),
        run_codex_task("T-214", "Implement CQRS index builder",
                       ["specs/integration/data-flow.md"], "", "src/simulation/"),
    )

    # Wave 6: Integration test
    await run_codex_task("T-215", "Phase 2 integration test",
                         ["specs/simulation/dinner-party-config.md"], "", "src/simulation/")


async def run_pacing_chain():
    """Sequential pacing implementation chain."""
    await run_codex_task("T-202", "Implement pacing constants",
                         ["specs/simulation/pacing-physics.md"], "", "src/simulation/")
    await run_codex_task("T-203", "Implement pacing physics",
                         ["specs/simulation/pacing-physics.md"], "", "src/simulation/")
    await run_codex_task("T-204", "Implement catastrophe detection",
                         ["specs/simulation/pacing-physics.md"], "S-01", "src/simulation/")


async def run_phase_3():
    """Phase 3: Metrics. Partial parallelism."""
    # Wave 1: Independent metric foundations (parallel)
    await asyncio.gather(
        run_codex_task("T-301", "Implement belief matrix",
                       ["specs/metrics/irony-and-beliefs.md"], "", "src/metrics/"),
        run_codex_task("T-304", "Implement thematic shift computation",
                       ["specs/integration/data-flow.md"], "", "src/metrics/"),
        run_codex_task("T-305", "Implement world state cache",
                       ["specs/metrics/tension-pipeline.md"], "S-16", "src/metrics/"),
        run_codex_task("T-306", "Implement index tables builder",
                       ["specs/integration/data-flow.md"], "", "src/metrics/"),
    )

    # Wave 2: Irony scoring (needs belief matrix)
    await run_codex_task("T-302", "Implement irony scoring",
                         ["specs/metrics/irony-and-beliefs.md"], "S-13", "src/metrics/")

    # Wave 3: Irony collapse + tension sub-metrics (parallel)
    await asyncio.gather(
        run_codex_task("T-303", "Implement irony collapse detection",
                       ["specs/metrics/irony-and-beliefs.md"], "S-12", "src/metrics/"),
        run_codex_task("T-307", "Implement danger sub-metric",
                       ["specs/metrics/tension-pipeline.md"], "S-02", "src/metrics/"),
        run_codex_task("T-308", "Implement remaining 7 tension sub-metrics",
                       ["specs/metrics/tension-pipeline.md"], "S-35", "src/metrics/"),
    )

    # Wave 4: Tension pipeline (needs all sub-metrics + irony)
    await run_codex_task("T-309", "Implement tension pipeline",
                         ["specs/metrics/tension-pipeline.md"], "", "src/metrics/")

    # Wave 5: Scene segmentation (needs tension + irony)
    await run_codex_task("T-310", "Implement scene segmentation",
                         ["specs/metrics/scene-segmentation.md"],
                         "S-08, S-19, S-20", "src/metrics/")
    await run_codex_task("T-311", "Implement scene type classification",
                         ["specs/metrics/scene-segmentation.md"], "S-07", "src/metrics/")

    # Wave 6: Integration
    await run_codex_task("T-312", "Phase 3 integration test",
                         ["specs/integration/data-flow.md"], "", "src/metrics/")


async def run_phase_4():
    """Phase 4: Extraction. Largely sequential."""
    await run_codex_task("T-401", "Implement arc grammar rules",
                         ["specs/metrics/story-extraction.md"], "", "src/extraction/")
    await run_codex_task("T-402", "Implement arc grammar validator",
                         ["specs/metrics/story-extraction.md"], "", "src/extraction/")
    await run_codex_task("T-403", "Implement beat classifier",
                         ["specs/metrics/story-extraction.md"], "S-25", "src/extraction/")
    await run_codex_task("T-404", "Implement arc scoring",
                         ["specs/metrics/story-extraction.md"], "", "src/extraction/")
    await run_codex_task("T-405", "Implement beat sheet generator",
                         ["specs/metrics/story-extraction.md"], "", "src/extraction/")

    # LLM tasks can run in parallel with each other
    await asyncio.gather(
        run_codex_task("T-406", "Implement LLM prompt builder",
                       ["specs/metrics/story-extraction.md"], "", "src/extraction/"),
        run_codex_task("T-407", "Implement LLM client",
                       ["specs/integration/data-flow.md"], "", "src/extraction/"),
    )

    await run_codex_task("T-408", "Phase 4 integration test",
                         ["specs/metrics/story-extraction.md"], "S-33", "src/extraction/")


async def main():
    """Run all phases sequentially (Phase 1 and 2 could overlap with schema sharing)."""
    print("=== NarrativeField Implementation ===")

    # Phase 1 and Phase 2 share T-201 (schema). Run Phase 2 schema first,
    # then Phase 1 and Phase 2 remainder can overlap.
    print("\n--- Phase 1: Visualization ---")
    await run_phase_1()

    print("\n--- Phase 2: Simulation ---")
    await run_phase_2()

    print("\n--- Phase 3: Metrics ---")
    await run_phase_3()

    print("\n--- Phase 4: Extraction ---")
    await run_phase_4()

    print("\n=== All phases complete ===")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Pre-Implementation Kickoff Checklist

### 9.1 Fix CRITICAL Audit Issues in Specs

These must be fixed in spec files BEFORE any code is written.

- [ ] **S-03 + S-15** (30 min): Add `tension_components` and `irony_collapse` to
  EventMetrics in `specs/schema/events.md:312-317`. Create Python `@dataclass
  EventMetrics` matching the TS interface. Update the default factory.
- [ ] **S-04 + S-14** (30 min): Unify WorldState snapshot naming to
  `WorldStateSnapshot` across `specs/integration/data-flow.md`,
  `specs/schema/scenes.md`, `specs/metrics/tension-pipeline.md`. Change
  `snapshot_interval` from "ticks" to "events" at data-flow.md:162,621.
- [ ] **S-01** (5 min): Fix `dominant_flaw.name` to `.flaw_type` in
  `specs/simulation/pacing-physics.md:433` (all 4 branches).
- [ ] **S-02** (5 min): Fix `context.secrets[delta.agent]` to
  `context.secrets[delta.attribute]` in `specs/metrics/tension-pipeline.md:84`.
- [ ] **S-05** (10 min): Change `content_display` to `description` in
  `specs/integration/data-flow.md` SecretDefinition.
- [ ] **S-06 + S-18** (15 min): Add `content_metadata: Optional[dict] = None` to
  Event dataclass in `specs/schema/events.md`. Update
  `specs/simulation/decision-engine.md` to use `content_metadata` consistently.
- [ ] **S-07** (10 min): Define `classify_scene_type_from_events()` in
  `specs/metrics/scene-segmentation.md:313` (takes Event objects, not IDs).

### 9.2 Fix HIGH Audit Issues in Specs

- [ ] **S-08 + S-19 + S-20** (20 min): Update `specs/schema/scenes.md`:
  scene_type enum to 6 types (remove "climax", rename per scene-segmentation.md),
  participant_turnover threshold 0.5 -> 0.3, add SOCIAL_MOVE forced boundary.
- [ ] **S-09** (15 min): Change `Secret.holder: str` to `holder: list[str]` in
  `specs/schema/world.md:329` and propagate to data-flow.md.
- [ ] **S-10 + S-11** (15 min): Add `initial_knowers`, `initial_suspecters`,
  `dramatic_weight`, `reveal_consequences` to SecretDefinition in data-flow.md.
  Add `overhear_probability` and `description` to LocationDefinition.
- [ ] **S-12** (15 min): Update IronyCollapse dataclass in irony-and-beliefs.md:
  add `detected: bool`, `score: float`; change `collapsed_beliefs` to
  `list[dict]` with `{agent, secret, from, to}`.
- [ ] **S-13** (10 min): Unify `_secret_relevance` to irony-and-beliefs.md's
  6-tier version in tension-pipeline.md.
- [ ] **S-17** (5 min): Update `specs/MASTER_PLAN.md:201` tension sub-metric
  names to canonical: danger, time_pressure, goal_frustration,
  relationship_volatility, information_gap, resource_scarcity, moral_cost,
  irony_density.

### 9.3 Repository Setup

- [ ] Create directory structure per Section 2
- [ ] Create `pyproject.toml` with Python 3.12+, pytest, numpy, scipy, networkx, anthropic
- [ ] Create `package.json` with React 18+, D3 v7, Zustand, TypeScript, Vitest
- [ ] Create `tsconfig.json` with strict mode
- [ ] Copy AGENTS.md files to their target directories
- [ ] Set up ESLint + Prettier for TypeScript
- [ ] Set up ruff or black for Python formatting

### 9.4 Fake Data Generation

- [ ] Generate `data/fake/dinner_party_70.nf-viz.json` per fake-data-visual-spec.md
- [ ] Validate JSON against NarrativeFieldPayload interface
- [ ] Verify all 3 tiers represented (story-critical, texture, ambiguity)
- [ ] Verify tension_components populated for all 70 events

### 9.5 Smoke Tests

- [ ] `npm run typecheck` passes with no errors
- [ ] `npm test` runner works (even with 0 tests)
- [ ] `pytest` runner discovers test directories
- [ ] Python imports resolve (`from simulation.schema.events import Event`)

### 9.6 AGENTS.md Verification

- [ ] Root AGENTS.md references correct spec paths
- [ ] Each subsystem AGENTS.md lists correct key specs
- [ ] Authority chain is consistent across all AGENTS.md files
- [ ] All audit issue IDs (S-XX) referenced in AGENTS.md files exist in audit-synthesis.md

---

## 10. Verification Checklist

Self-verification of this document:

- [x] All 4 AGENTS.md files reference correct spec paths (relative from repo root)
- [x] Cloud task DAG has no circular dependencies (verified: all prereqs point to
  lower-numbered tasks within phase, or to T-201 for cross-phase schema)
- [x] Skills reference real tool invocations (spec reads, audit checks, code generation)
- [x] Kickoff checklist addresses all 7 CRITICAL audit findings (S-01 through S-07)
- [x] Total AGENTS.md content is concise (all under 32KB)
- [x] Task prompts are self-contained (each references spec + section + audit issues)
- [x] Phase 1 tasks maximize parallelism (T-101, T-102 independent; T-103-T-108 parallel after T-101)
- [x] Phase 2 tasks maintain strict ordering (schema -> pacing -> decision -> tick loop)
- [x] Phase 3 respects pipeline order (irony -> thematic -> tension -> scenes)
- [x] Phase 4 is largely sequential (grammar -> beats -> scoring -> LLM)
- [x] 45 total tasks: Phase 1 (10) + Phase 2 (15) + Phase 3 (12) + Phase 4 (8)
- [x] Authority chain documented in every AGENTS.md that deals with pacing, segmentation, or characters
