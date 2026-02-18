# NarrativeField Architecture

NarrativeField is a simulation-first storytelling system. Instead of prompting an LLM to "write a story", it:

1. Simulates a deterministic multi-agent world into an event log.
2. Computes narrative metrics (tension, irony) over the log.
3. Extracts coherent arcs using grammar + scoring.
4. Narrates those arcs into multi-scene prose with tracked cost/timing artifacts.

This document is the engineering deep dive. For authoritative schemas and algorithms, see the specification set in `specs/` (read-only).

---

## 1) Design Philosophy

### "Architectural" vs "Episodic" storytelling

Most AI writing tools are episodic: a prompt produces the next paragraph, then the next, with structure and continuity enforced primarily by prompt engineering. NarrativeField is architectural: it builds a world with agents and constraints, simulates what happens, and only then turns the resulting structure into prose.

This inversion has concrete engineering consequences:
- The system's ground truth is an **inspectable event log**, not a prompt transcript.
- Structure is enforced upstream by **dynamics + metrics + extraction**, not downstream by "please write a plot twist".
- Outputs are reproducible. If seed + scenario are the same, the simulation and extracted arc are the same, enabling debugging and regression tests.

### Constraint satisfaction vs sequential generation

Sequential generation is good at local coherence, but it struggles to satisfy global constraints (pacing, rising tension, irony payoffs) without heavy scaffolding. NarrativeField reframes story generation as a selection problem:
- Generate many events (the world "does things").
- Score them (mathematical tension and belief/irony state).
- Select a subset that forms a valid arc (grammar constraints + soft ranking).
- Narrate from a structured beat sheet with rich world context.

### Determinism as a feature

Deterministic simulation is not just for reproducibility; it enables:
- A/B comparisons (different metrics weights, different extraction grammar parameters) against the same ground truth.
- Debuggable failure cases (a specific seed becomes a regression fixture).
- Measurable production validation (success rate, retries, cost, timing) that is independent of "writerly taste".

---

## 2) Simulation Engine

**Specs:** `specs/simulation/tick-loop.md`, `specs/simulation/decision-engine.md`, `specs/simulation/pacing-physics.md`, `specs/simulation/dinner-party-config.md`  
**Schemas:** `specs/schema/events.md`, `specs/schema/agents.md`, `specs/schema/world.md`  
**Implementation:** `src/engine/narrativefield/simulation/`

### Event-sourced world model

The simulation produces an append-only list of `Event` objects. World state is treated as a materialized view derived from that stream (with periodic snapshots for fast replay). This is the foundation for every later stage: metrics are computed over events; visualization renders the resulting structure; extraction and narration operate over selected event subsets.

### Discrete ticks with simultaneous action resolution

The time model is discrete. Each tick:
1. Agents construct a perceived state (location-filtered, belief-filtered).
2. Catastrophes may fire (involuntary breaks) before voluntary action.
3. Agents propose actions; conflicts are resolved deterministically with tie-breaking randomness from a seeded RNG.
4. Events are emitted with ordering (`tick_id`, `order_in_tick`) and `sim_time` progression.

See `specs/simulation/tick-loop.md` for the contract and ordering guarantees.

### Agent decision engine inputs

Agents are not generic "chatbots". They are stateful entities with:
- Goals/desires and flaws (biasing action scoring toward story-shaped decisions).
- Secrets and beliefs (finite belief catalog; belief updates are events/deltas).
- Relationships and social context.
- Pacing state (dramatic budget, stress, composure, commitment).

The decision engine reads these values as inputs and proposes actions; it does not directly "write plot". See `specs/simulation/decision-engine.md` and `specs/schema/agents.md`.

### Pacing physics (drama budget + hysteresis)

The pacing system prevents a simulation from collapsing into either endless small talk or constant melodrama. The core idea is that dramatic action is expensive and bounded:
- `dramatic_budget` is spent on dramatic actions and recharges on quiet ticks.
- `stress` accumulates via conflict exposure and decays slowly.
- `composure` enables masking; low composure makes emotion and conflict leak into public scenes.
- `commitment` captures irreversible investment in a path (a second catastrophe parameter).
- Hysteresis is explicit: repairing trust is harder than breaking it; recovery is slower than escalation.

Catastrophes are modeled via a simple cusp-like trigger that combines state variables:
- Catastrophe potential is based on a function of `stress` and `commitment` (plus suppression history).
- A composure gate prevents constant breaks; catastrophes fire when stress is high and composure is low.

See `specs/simulation/pacing-physics.md` for the state variables, constants, and update rules.

### Affordances and location rules

Locations are not just labels. They define interaction constraints:
- Privacy (public vs private actions, masking behavior).
- Adjacency and overhear rules (events can be witnessed indirectly).
- Capacity and movement affordances.

These constraints are part of the simulation dynamics and also feed downstream segmentation (scene boundaries often align with location + participant set shifts). See `specs/schema/world.md` and `specs/simulation/dinner-party-config.md`.

### Deterministic seeding

RNG seeding is explicit and testable. The live generation harness uses `random.Random(seed)` and passes it through simulation, ensuring reproducible event logs per scenario+seed. See `src/engine/scripts/test_live_generation.py`.

---

## 3) Metrics Pipeline

**Specs:** `specs/metrics/tension-pipeline.md`, `specs/metrics/irony-and-beliefs.md`, `specs/metrics/scene-segmentation.md`  
**Integration contract:** `specs/integration/data-flow.md`  
**Implementation:** `src/engine/narrativefield/metrics/`

Metrics are computed post-hoc over the completed event log and snapshots, then attached back onto events. This separation matters:
- Simulation produces "what happened" and state deltas.
- Metrics compute "what it means" in a narratological sense (tension/irony/structure) without affecting simulation dynamics.

### Tension scoring (mathematical, interpretable)

Each event receives a composite tension score derived from a weighted sum of normalized sub-metrics (eight channels, each in [0,1]). Sub-metrics are defined to be interpretable and computable with indexed state access (O(1) per event given snapshots/indices).

See `specs/metrics/tension-pipeline.md` and implementation in `src/engine/narrativefield/metrics/tension.py`.

### Beliefs + irony

The system tracks a belief matrix: (agent x secret) -> BeliefState. Dramatic irony is computed as a function of relevance-weighted belief gaps between ground truth secrets and character beliefs.

This enables:
- Quantified irony density per event/scene.
- Explicit irony collapse detection when a reveal event closes the belief gap.

See `specs/metrics/irony-and-beliefs.md` and implementation in `src/engine/narrativefield/metrics/irony.py`.

### Scene segmentation

Scenes are first-class units between events and arcs. The segmenter groups events into scenes using explicit boundary rules:
- Location change (with an adjacency/overhear exception).
- Participant turnover (Jaccard similarity threshold).
- Time gaps.
- Sustained tension valleys (beat boundary).
- Forced breaks after irony collapse.

See `specs/metrics/scene-segmentation.md` and implementation in `src/engine/narrativefield/metrics/segmentation.py`.

### How metrics feed extraction

Extraction uses scenes and event metrics in two ways:
- As inputs to beat classification and arc validation constraints.
- As soft scoring signals (prefer arcs with coherent escalation and meaningful turning points).

---

## 4) Story Extraction

**Spec:** `specs/metrics/story-extraction.md`  
**Implementation:** `src/engine/narrativefield/extraction/`

Story extraction turns a large, noisy event log into a small, structured arc suitable for narration.

### Arc grammar + beat sheets

Extraction is grammar-first:
1. Classify candidate events into BeatTypes (SETUP, COMPLICATION, ESCALATION, TURNING_POINT, CONSEQUENCE).
2. Validate that a sequence satisfies an arc grammar (ordering constraints, minimum phases, causal connectivity, protagonist consistency, time span).
3. Score valid candidates and select the best arc.
4. Emit a beat sheet (structured representation) that narration can consume deterministically.

See `specs/metrics/story-extraction.md` and implementation in:
- `src/engine/narrativefield/extraction/arc_search.py`
- `src/engine/narrativefield/extraction/arc_validator.py`
- `src/engine/narrativefield/extraction/arc_scorer.py`
- `src/engine/narrativefield/extraction/beat_classifier.py`
- `src/engine/narrativefield/extraction/beat_sheet.py`

### Causal and thread indexing (selection, not guessing)

Arc search uses the event log's causal links and participant overlap to keep arcs coherent. It is not a highlight reel; it explicitly rejects sequences that do not form a story-shaped structure.

---

## 5) Narration Pipeline

**Implementation:** `src/engine/narrativefield/storyteller/` and `src/engine/narrativefield/llm/`

Narration converts an extracted arc into multi-scene prose with measurable reliability and per-scene artifacts.

### Scene splitting and narrative state

Given arc events, the narrator:
- Splits events into scene chunks (`scene_splitter.py`).
- Maintains a running narrative state object (summary-so-far, character emotional state/goals, unresolved threads) that is updated after each scene (`narrator.py`, `types.py`).
- Compresses state when it grows too large to preserve token budgets (`prompts.py`, structural summarization calls).

### Lorebook context injection

Before generating prose, the narrator constructs a lorebook:
- World definition (locations, scenario framing).
- Agent backstories and relationships.
- Secret registry and current world state context.

This makes the prose generation grounded: the creative model is not inventing the world from scratch, it is rendering a specific simulated situation. See `lorebook.py`.

### Dual-LLM strategy (structural vs creative)

Narration uses two tiers of model calls via `src/engine/narrativefield/llm/gateway.py`:
- **STRUCTURAL (Grok 4.1 Fast):** continuity checks, compression, validation-like tasks.
- **CREATIVE (Claude Haiku 4.5):** prose generation from beat sheets + lorebook.
- **CREATIVE_DEEP:** the same creative model with extended thinking enabled for pivotal scenes when configured.

This separation reduces cost while keeping creative output quality high where it matters.

### Extended thinking tier for pivotal scenes

When a scene is marked pivotal and `phase2_use_extended_thinking_for_pivotal` is enabled, the gateway enables extended thinking for that scene (CREATIVE_DEEP). If transient failures occur, the gateway retries with exponential backoff and can downgrade CREATIVE_DEEP to CREATIVE after retry exhaustion. See `llm/gateway.py` and `storyteller/narrator.py`.

### Reliability: retry armor + repetition guard + continuity checks

The narration pipeline includes:
- Exponential backoff retry with jitter for retryable provider failures (`llm/gateway.py`).
- Repetition detection for common sequential-generation failure modes (sentence/paragraph repeats) (`repetition_guard.py`).
- Post-generation continuity checks (structural tier) to detect inconsistencies between prose and the underlying events (`postprocessor.py`).

### Per-scene artifact tracking (meta.json)

Runs emit:
- Prose output (`.txt`).
- A `*_meta.json` file including per-scene outcomes (status, retries, word counts, timing) plus total tokens and estimated cost.

The live harness that writes these artifacts is `src/engine/scripts/test_live_generation.py`. Golden outputs are copied into `examples/`.

---

## 6) Cost Architecture

The cost model is an explicit design constraint: generate story-shaped prose at low marginal cost.

Key mechanisms:
- **Tiering:** use a cheap structural model for compression/continuity tasks and reserve creative spend for prose.
- **Prompt caching (Anthropic):** cacheable system prompts reduce repeated input token cost for sequential scene generation when extended thinking is not enabled (`llm/gateway.py`).
- **Budgeting:** the pipeline has explicit caps for max tokens per creative scene and for narrative-state size (`src/engine/narrativefield/llm/config.py`).
- **Concurrency controls:** structured batch calls are concurrency-limited to avoid provider throttling while maintaining throughput.

The result is measurable: production validation runs can report word counts, per-seed costs, and retry counts in `*_meta.json` and in `examples/`.

---

## 7) Visualization (Frontend)

**Specs:** `specs/visualization/renderer-architecture.md`, `specs/visualization/thread-layout.md`, `specs/visualization/interaction-model.md`  
**Implementation:** `src/visualization/src/`

The visualization is an interactive exploration layer over the event+metrics bundle:
- React + TypeScript state management (Zustand).
- Canvas renderer with multiple layers (background, nodes, threads, highlights) and hit detection for hover/click.
- Thread layout computed by a spring-force style algorithm to make narrative threads readable.
- Timeline navigation and "hover neighborhoods" that emphasize causal/participant proximity.

The UI consumes `.nf-viz.json` bundles produced by the metrics pipeline (see `specs/integration/data-flow.md` for interface contracts).

