# NarrativeField Methodology

> Evidence-first documentation of the NarrativeField engine and visualization system.
> Every claim cites `file:line`. Test results captured 2026-02-11.

## Table of Contents

1. [Overview](#1-overview)
2. [Simulation Engine](#2-simulation-engine)
3. [Pacing Physics](#3-pacing-physics)
4. [Metrics Pipeline](#4-metrics-pipeline)
5. [Visualization](#5-visualization)
6. [LLM Integration](#6-llm-integration)
7. [Scenario Design](#7-scenario-design)
8. [Seeding & Determinism](#8-seeding--determinism)
9. [Test Status](#9-test-status)
10. [Spec vs Implementation Delta Table](#10-spec-vs-implementation-delta-table)
- [Appendix A: Formulas](methodology/FORMULAS.md)
- [Appendix B: Prompts](methodology/PROMPTS.md)
- [Appendix C: Data Schemas](methodology/DATA_SCHEMAS.md)

---

## 1. Overview

NarrativeField simulates fictional worlds and extracts narratives from the emergent dynamics. The core inversion: **don't write the story — simulate the world, extract the stories that emerge.**

### Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Simulation  │───▶│   Metrics   │───▶│Visualization│    │  Extraction │
│   Engine     │    │  Pipeline   │    │  (Canvas 2D)│    │  (LLM Prose)│
│  (Python)    │    │  (Python)   │    │ (TypeScript) │    │  (Python)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   Event log         Enriched           Rendered           Beat sheets
   (.nf-sim.json)    (.nf-viz.json)     thread map         → story prose
```

### Data Flow

1. **Simulation** produces a JSON event log with ~100-200 events per run
2. **Metrics pipeline** enriches events with tension, irony, thematic shift scores, then bundles and segments into scenes
3. **Visualization** renders an interactive 2D thread map from the enriched payload
4. **Extraction** selects character arcs, classifies beats, builds beat sheets, and generates prose via LLM

### Tech Stack

| Component | Stack |
|-----------|-------|
| Simulation | Python 3.12+, dataclasses, event sourcing |
| Metrics | Python, NumPy-free (pure dataclass math) |
| Visualization | TypeScript, React, Canvas 2D, Zustand |
| Prose generation | Claude API (Anthropic SDK), Grok API (OpenAI SDK) |
| Data format | JSON event logs |

---

## 2. Simulation Engine

The simulation uses an **event-sourced world model** (Decision #1): events are the primary nodes, world-state is a materialized view rebuilt from deltas.

### Tick Loop Pipeline

Each tick executes these steps in `execute_tick()` (`tick_loop.py:888-918`):

```
1. Check catastrophes     →  check_catastrophes()       (tick_loop.py:895)
2. Propose actions        →  propose_actions()          (tick_loop.py:904)
3. Detect conflicts       →  detect_conflicts()         (tick_loop.py:905)
4. Resolve conflicts      →  resolve_conflicts()        (tick_loop.py:906)
5. Choose actions         →  choose_actions()           (tick_loop.py:909)
6. Convert to events      →  action_to_event()          (tick_loop.py:910-913)
7. Generate witnesses     →  generate_witness_events()  (tick_loop.py:915-916)
```

After all tick events are generated, `apply_tick_updates()` (`tick_loop.py:850-886`) runs:
1. Apply event deltas in order
2. Update pacing state for all agents via `pacing.end_of_tick_update()`
3. Advance sim time

### Decision Engine (Per-Agent)

Each agent proposes one action via argmax over candidates (`decision_engine.py:1034-1054`):

```
Score = base_utility
      + flaw_bias
      + pacing_modifier
      + relationship_modifier
      - recency_penalty
      + action_noise
```

SOCIAL_MOVE actions receive a -0.20 penalty (`decision_engine.py:1045-1046`). Ties are broken deterministically by a stable action key (`decision_engine.py:1049-1050`). This is **argmax selection**, not softmax.

### Cross-Agent Action Selection

After per-agent proposals are conflict-resolved, `choose_actions()` (`tick_loop.py:224-262`) uses **softmax sampling** with temperature=0.4 (`tick_loop.py:239`) to select which actions execute this tick. Score penalties before softmax:

| Action Type | Penalty |
|-------------|---------|
| INTERNAL | -0.35 (`tick_loop.py:246`) |
| SOCIAL_MOVE | -0.20 (`tick_loop.py:249-250`) |
| OBSERVE | -0.10 (`tick_loop.py:247-248`) |

Sampling is without replacement by agent — each agent acts at most once per tick.

### Conflict Detection

`detect_conflicts()` (`tick_loop.py:144-165`) checks two rules:

1. **Target contention**: Two actions with `requires_target_available=True` share a target agent (`tick_loop.py:148-157`). Uses set intersection, not Jaccard.
2. **Incompatible actions**: CONFLICT blocks CONFIDE/REVEAL on the same target (`tick_loop.py:159-163`).

### Conflict Resolution

`resolve_conflicts()` (`tick_loop.py:189-208`) resolves in descending severity order:

- `incompatible_actions` (severity 3) > `target_contention` (2) > `location_capacity` (1)
- Winner chosen by: `priority_class` → `utility_score` → random tiebreak
- Losers are downgraded (`tick_loop.py:168-186`):
  - CONFIDE/CONFLICT/REVEAL/LIE → INTERNAL
  - CHAT → OBSERVE

### Termination Conditions (`tick_loop.py:921-938`)

Simulation ends when any of: max sim time reached, tick limit hit, <2 active agents, all agents in cooldown with low stress, or event limit reached.

See [Appendix A](methodology/FORMULAS.md) for the full utility computation formulas.

---

## 3. Pacing Physics

Six state variables per agent (`pacing.py:14-73`, `PacingConstants` dataclass):

| Variable | Range | Key Constants |
|----------|-------|---------------|
| `dramatic_budget` | [0, 1] | Recharge 0.08/tick, cost 0.15-0.50 per action |
| `stress` | [0, 1] | Gain 0.03-0.15 per trigger, decay 0.03/tick |
| `composure` | [0.05, 1] | Alcohol -0.06, stress erosion -0.02, recovery +0.01 |
| `commitment` | [0, 1] | Public statement +0.10, confrontation +0.15, reveal +0.20 |
| `recovery_timer` | [0, ∞) | Set to 2-6 ticks after dramatic events, decrements each tick |
| `suppression_count` | [0, ∞) | Increments when stress ≥0.60 and no dramatic action; resets on dramatic action |

### Catastrophe Cusp

The catastrophe check (`pacing.py:91-107`):

```
potential = stress × commitment² + suppression_count × 0.03
```

Catastrophe fires when:
- `potential >= 0.35` (threshold, `pacing.py:61`)
- `composure < 0.30` (gate, `pacing.py:62`)
- `recovery_timer == 0` (cooldown check, `pacing.py:103`)

After catastrophe (`pacing.py:352-414`):
- stress halved
- composure reset to 0.30
- commitment +0.10 (if <0.80)
- recovery_timer set to 8 ticks
- suppression_count reset to 0

### Social Masking

In public locations (privacy < 0.3), agents with composure ≥ 0.40 have dramatic utility suppressed by 0.5× (`pacing.py:110-117`).

### Hysteresis

Trust repair is 3× harder than trust damage (`pacing.py:67`, applied in `tension.py:277-278` during metrics and in `tick_loop.py` during simulation).

See [Appendix A](methodology/FORMULAS.md) for complete update rules with all constants.

---

## 4. Metrics Pipeline

The pipeline runs in `run_metrics_pipeline()` (`pipeline.py:89-136`):

| Step | What | Code |
|------|------|------|
| 1 | **Irony** — belief-state scoring per event | `pipeline.py:98-99` |
| 2 | **Thematic** — rules-based shift across 5 axes | `pipeline.py:101-102` |
| 3 | **Tension** — 8-component weighted score | `pipeline.py:104-118` |
| 4 | **Bundling** — compress micro-events for viz/extraction | `pipeline.py:120-124` |
| 5 | **Segmentation** — split bundled events into scenes | `pipeline.py:126-127` |

**Note:** Specs (Decision #20) say "irony → thematic → tension → scenes". Implementation adds a bundling step between tension and segmentation. There is no separate significance step — `EventMetrics.significance` defaults to 0.0 and is not currently computed.

### Tension Model (8 Components)

Defined in `tension.py:11-20`, default weight 1/8 each (`tension.py:32-40`):

| Component | Source | Key Logic |
|-----------|--------|-----------|
| `danger` | Event type + trust deltas | CATASTROPHE=1.0, CONFLICT=0.8, REVEAL=0.7 (`tension.py:65-95`) |
| `time_pressure` | Evening progression + secret convergence | Quadratic ramp + recovery timer + composure loss (`tension.py:98-132`) |
| `goal_frustration` | Agent pacing state | 0.6×stress + 0.4×(1-budget) (`tension.py:135-143`) |
| `relationship_volatility` | Relationship deltas | Current + recent window, scaled by 0.6 (`tension.py:146-160`) |
| `information_gap` | Belief divergence per secret | Weighted by dramatic_weight, diversity across agents present (`tension.py:163-190`) |
| `resource_scarcity` | Budget + composure depletion | max(budget_scarcity, composure_loss)×0.85 + timer×0.15 (`tension.py:193-199`) |
| `moral_cost` | Event type × agent goals | CATASTROPHE=0.9, LIE=0.5+0.5×truth, etc. (`tension.py:202-219`) |
| `irony_density` | Pre-computed irony score | irony/2.0, clamped to [0,1] (`tension.py:222-224`) |

Final tension = weighted sum, clamped to [0, 1]. The tension pipeline also applies deltas to advance evolving state for subsequent events (`tension.py:396-397`).

### Irony Scoring

Per-agent irony uses a belief-state scoring table (`irony.py:29-74`):

| Belief State | Score | Condition |
|-------------|-------|-----------|
| Actively wrong | 2.0 × relevance | Believes opposite of truth |
| Relevant unknown | 1.5 × relevance | Secret is about/held by agent |
| General unknown | 0.5 × relevance | Standard unknown |
| Suspects (true) | 0.25 × relevance | On the right track |
| Correct | 0.0 | No irony |

Relevance weights (`irony.py:14-26`): about_self=1.0, holder=0.7, about_present=0.5, other=0.2.

Scene irony = mean of per-agent irony scores for agents present. Irony collapse detected when scene irony drops ≥0.5 between consecutive events (`irony.py:159-169`).

### Thematic Shift

Rules-based + delta-driven across 5 axes (`thematic.py:6-12`):

| Axis | Triggers |
|------|----------|
| `order_chaos` | CONFLICT -0.15, CATASTROPHE -0.30 |
| `truth_deception` | REVEAL +0.2, LIE -0.2, CONFIDE +0.1 |
| `loyalty_betrayal` | CONFIDE +0.1, trust delta < -0.2 → -0.1, trust > +0.2 → +0.05 |
| `innocence_corruption` | CATASTROPHE -0.15 |
| `freedom_control` | Commitment delta → -0.1 |

### Segmentation

Boundary rules in `SegmentationConfig` (`segmentation.py:25-38`):

| Rule | Threshold | Code |
|------|-----------|------|
| Participant Jaccard | < 0.3 | `segmentation.py:28` |
| Time gap | > 5.0 minutes | `segmentation.py:30` |
| Tension valley window | 5 events | `segmentation.py:33` |
| Tension drop ratio | 0.3 | `segmentation.py:34` |
| Sustained count | 3 | `segmentation.py:35` |
| Minimum scene size | 3 events | `segmentation.py:37` |

See [Appendix A](methodology/FORMULAS.md) for full formulas and [Appendix C](methodology/DATA_SCHEMAS.md) for the delta table comparing metrics segmentation vs storyteller scene splitting.

---

## 5. Visualization

### Architecture: 4-Layer Canvas 2D

The renderer uses stacked HTML Canvas elements, **not** D3/SVG (D3 packages are installed but unused for rendering). Each layer is a separate `<canvas>` with independent dirty-flag redraws (`LayerManager.ts:24-187`):

| Layer | Canvas | Content |
|-------|--------|---------|
| `background` | `BackgroundLayer.ts` | Scene bands (alternating opacity) + atmospheric heatmap |
| `threads` | `ThreadLayer.ts` | Per-agent thread paths as Bezier curves |
| `events` | `EventNodeLayer.ts` | Event nodes as circles + HitCanvas for click detection |
| `highlight` | `HighlightLayer.ts` | Hover/selection rings, causal graph highlights |

The top (highlight) layer receives pointer events; others have `pointerEvents: 'none'` (`LayerManager.ts:73`).

### Force-Directed Thread Layout

`computeThreadLayout()` (`threadLayout.ts:283-350`) runs a force simulation with 5 forces:

| Force | Default | Effect |
|-------|---------|--------|
| `attractionStrength` | 0.3 | Pulls agents at same location together |
| `repulsionStrength` | 0.2 | Pushes agents at different locations apart |
| `interactionBonus` | 0.5 | Extra attraction when agents interact (within 1.0 min window) |
| `laneSpringStrength` | 0.1 | Pulls agents back toward their base lane |
| `inertia` | 0.7 | Resistance to change from previous position |

Additional parameters: `minSeparation=20px`, `lanePadding=40px`, `iterations=50`, `convergenceThreshold=0.5`, `timeResolution=0.5` minutes (`threadLayout.ts:33-46`).

Agent positions are computed per time-sample along the x-axis, producing smooth thread paths.

### Visual Encoding

| Channel | Mapping | Code |
|---------|---------|------|
| X position | sim_time → viewport pixels | `renderModel.ts:6-12` |
| Y position | Force-directed layout, interpolated per agent | `renderModel.ts:14-46` |
| Node shape | Circle | `EventNodeLayer.ts` |
| Node color | Event type → color map (10 types) | `colors.ts:13-24` |
| Node radius | `baseRadius + significance × 2` (base: 4/8/12 by zoom) | `renderModel.ts:78-86` |
| Node opacity | `0.25 + tension × 0.85` | `renderModel.ts:100` |
| Glow color | Tension bands: <0.2 none, <0.4 blue, <0.6 amber, <0.8 orange, ≥0.8 red | `renderModel.ts:48-54` |
| Thread thickness | `2 + avgTension × 4` (pixels) | `threadLayout.ts:336` |
| Thread color | Per-character color palette | `colors.ts:3-10` |
| Background heat | Per-event glow intensity → tension heat color (5-stop ramp) | `BackgroundLayer.ts:39-45`, `colors.ts:26-33` |

Character colors: Thorne=#E69F00, Elena=#56B4E9, Marcus=#009E73, Lydia=#F0E442, Diana=#0072B2, Victor=#D55E00 (`colors.ts:3-10`).

### Interaction

- **Hover**: Highlights causal graph — backward links in blue (`rgba(68,136,204,0.65)`), forward links in amber (`rgba(204,136,68,0.65)`) (`HighlightLayer.ts:28-38`, `colors.ts:35-36`)
- **Click**: Selects event with bright white ring (`HighlightLayer.ts:49-52`)
- **Arc selection**: Highlights all events for selected agent's arc (`HighlightLayer.ts:41-47`)
- **Tension recomputation**: Weight sliders + genre presets recompute tension in real-time via `recomputeTensionMap()` (`narrativeFieldStore.ts:68-70`)
- **Pan/zoom**: Viewport state with zoom levels: CLOUD, THREADS, DETAIL (`narrativeFieldStore.ts:46`)

### Topology Mode

Not yet implemented. The background layer uses a simple atmospheric heatmap, not topographic terrain. The spec calls for a Wendland C2 RBF → topographic terrain rendering, but this is deferred.

---

## 6. LLM Integration

Two separate LLM-powered pipelines exist with independent model configurations.

### 6a. Storyteller Pipeline

Scene-by-scene narrative generation via `SequentialNarrator`.

**STRUCTURAL tier**: `grok-4-1-fast` (XAI, via OpenAI SDK). Fallback: `grok-beta` (`config.py:9`, `gateway.py:278-284`).
- Used for: summary compression, continuity checks
- Called via `LLMGateway.generate(ModelTier.STRUCTURAL, ...)` → `_generate_grok()` (`gateway.py:93,272`) with retry + fallback

**CREATIVE tier**: `claude-haiku-4-5-20251001` (Anthropic) (`config.py:10`, `gateway.py:27-30`).
- Used for: prose generation per scene chunk
- Prompt caching: system prompt cached across all scenes (`config.py:24`, `gateway.py:386-393`)
- Extended thinking: optional for pivotal scenes, 10k budget tokens (`gateway.py:380-382`)

**Model tier enum** (`gateway.py:27-30`):
- `STRUCTURAL` — Grok 4.1 Fast
- `CREATIVE` — Claude Haiku 4.5
- `CREATIVE_DEEP` — Claude Haiku 4.5 + extended thinking

### 6b. Extraction Prose Generator

Beat sheet → story via `generate_prose()` (`prose_generator.py:122-164`).

- **Model**: `claude-sonnet-4-5-20250929` — hardcoded (`prose_generator.py:12`)
- **Temperature**: 0.8 (`prose_generator.py:13`)
- Synchronous Anthropic SDK call (not async gateway)
- No prompt caching, no extended thinking
- **Drift**: This pipeline does NOT use `PipelineConfig` or `LLMGateway`. It is an independent code path with its own model constant.

### Model Summary

| Pipeline | Model | Provider | Async | Caching | Extended Thinking |
|----------|-------|----------|-------|---------|-------------------|
| Storyteller structural | `grok-4-1-fast` | XAI/OpenAI SDK | Yes | No | No |
| Storyteller creative | `claude-haiku-4-5-20251001` | Anthropic SDK | Yes | Yes | Optional (pivotal) |
| Extraction | `claude-sonnet-4-5-20250929` | Anthropic SDK | No | No | No |

See [Appendix B](methodology/PROMPTS.md) for full prompt text.

---

## 7. Scenario Design

### The Dinner Party (`dinner_party.py:8-142`)

6 agents, 1 evening (~150 sim-minutes), 5 locations.

**Characters**:

| Agent | Name | Key Flaw | Primary Goal |
|-------|------|----------|-------------|
| `thorne` | James Thorne | Pride (strength 0.8) | Status (0.9) |
| `elena` | Elena Thorne | Guilt (0.7) + Cowardice (0.5) | Secrecy (0.9) |
| `marcus` | Marcus Webb | Ambition (0.8) + Denial (0.6) | Secrecy (1.0) |
| `lydia` | Lydia Cross | Cowardice (0.7) + Loyalty (0.6) | Loyalty (0.9) |
| `diana` | Diana Forrest | Guilt (0.6) + Jealousy (0.4) | Closeness/Elena (0.7) |
| `victor` | Victor Hale | Obsession (0.6) + Vanity (0.4) | Truth-seeking (0.9) |

**Secrets** (`dinner_party.py:67-127`):

| Secret | Weight | Initial Knowers |
|--------|--------|-----------------|
| Elena-Marcus affair | 1.0 | elena, marcus, diana; lydia suspects |
| Marcus embezzlement | 0.9 | marcus; lydia, victor suspect |
| Diana's debt to Marcus | 0.6 | diana, marcus |
| Lydia's knowledge of discrepancies | 0.5 | lydia |
| Victor's investigation of Marcus | 0.7 | victor; marcus suspects |

**Locations** (`dinner_party.py:14-65`):

| Location | Privacy | Capacity | Overhears From |
|----------|---------|----------|----------------|
| Dining Table | 0.1 | 6 | Kitchen |
| Kitchen | 0.5 | 3 | — |
| Balcony | 0.7 | 3 | — |
| Foyer | 0.2 | 4 | — |
| Bathroom | 0.9 | 1 | — |

**Seating** (circular): Thorne–Elena–Marcus–Diana–Lydia–Victor (`dinner_party.py:129-136`).

**Conflict Arcs**: The three primary tension lines are the affair (Elena/Marcus vs Thorne), the embezzlement (Marcus vs Victor's investigation), and the loyalty web (Lydia/Diana caught between sides).

---

## 8. Seeding & Determinism

### Random Threading

All stochastic decisions are routed through `random.Random(seed)`, threaded from `run_simulation()` through every function that needs randomness. No use of `random.random()` global state.

### Science Harness (`scripts/science_harness.py`)

Multi-seed batch runner for reproducibility analysis:

- **Event stream fingerprinting**: SHA-256 over determinism-critical fields (id, type, source_agent, target_agents, location_id, causal_links, deltas, tick_id, order_in_tick, sim_time) (`science_harness.py:17-29`, `science_harness.py:66-74`)
- **Cross-process determinism probe**: Runs the same seed with two different `PYTHONHASHSEED` values (12345, 67890) and compares event stream fingerprints (`science_harness.py:209-244`)
- **Batch execution**: Seeds configurable via `--seed-from` / `--seed-to`, each run through simulation + optional metrics pipeline (`science_harness.py:247-298`)
- **Per-run metrics**: catastrophe count, catastrophes by agent, location distribution, event stream SHA-256 (`science_harness.py:274-295`)

### Determinism Guarantees

The in-process test (`test_determinism.py`) and cross-process probe verify that:
1. Same seed → identical event stream (byte-identical SHA-256)
2. `PYTHONHASHSEED` variation does not affect output
3. Time scale changes preserve event count

---

## 9. Test Status

Captured 2026-02-11 on branch `docs/methodology`, macOS Darwin 25.2.0, Python 3.14.2.

| Suite | Result | Details |
|-------|--------|---------|
| Engine tests | **80 passed** in 2.24s | `pytest -v`, zero failures |
| Engine lint | 82 errors (51 auto-fixable) | Mostly unused imports in audit scripts, not core code |
| Viz tests | **8 passed** (4 test files) | Vitest v2.1.9, 366ms |
| Viz lint | **Clean** (0 warnings) | ESLint `--max-warnings=0` |
| Viz build | **Pass** | 71 modules, 260.97 kB (74.25 kB gzip) |

Lint errors are concentrated in audit scripts (`audit_sim.py`, `audit_metrics.py`, `audit_storyteller.py`) and test files (`test_narrator_e2e.py`), but also include a few core modules: `api_server.py` (unused `asyncio` import), `narrator.py` (unused `asyncio` import), and `prompts.py` (unused `BeatType` import).

---

## 10. Spec vs Implementation Delta Table

| Area | Spec Says | Code Does | Spec File | Code File:Line |
|------|-----------|-----------|-----------|----------------|
| Metrics pipeline order | irony → thematic → tension → scenes (Decision #20) | irony → thematic → tension → **bundle** → segmentation (5 steps) | `data-flow.md` | `pipeline.py:98-127` |
| Significance metric | Computed per event | Defaults to 0.0 (not computed) | `events.md` | `events.py:145` |
| Metrics segmentation | Jaccard <0.3, time gap >5min, tension valley | Same | `scene-segmentation.md` | `segmentation.py:27-35` |
| Storyteller scene splitting | N/A (storyteller-specific) | Jaccard <0.5, time gap >10min, beat transition, midpoint fallback | — | `scene_splitter.py:28-29,140,148` |
| Decision engine | "softmax" (per prior exploration) | **Argmax** + noise per agent, deterministic tiebreak | `decision-engine.md` | `decision_engine.py:1049-1050` |
| Cross-agent selection | — | Softmax temp=0.4 over resolved actions | — | `tick_loop.py:239` |
| Conflict detection | — | Target contention + incompatible action rules | `tick-loop.md` | `tick_loop.py:144-165` |
| Catastrophe order | Catastrophes resolve last (pacing-physics.md) | Catastrophes generated **first** in tick | `pacing-physics.md` | `tick_loop.py:891-894` |
| Extraction model | Should use pipeline config | Hardcoded `claude-sonnet-4-5-20250929` | — | `prose_generator.py:12` |
| Storyteller creative model | — | `claude-haiku-4-5-20251001` | — | `config.py:10` |
| Topology visualization | Wendland C2 RBF → topographic terrain | Atmospheric heatmap (simple tension bands) | `fake-data-visual-spec.md` | `BackgroundLayer.ts:39-45` |
