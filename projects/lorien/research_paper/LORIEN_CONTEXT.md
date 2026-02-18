# Lorien / NarrativeField — Complete Project Context

## What This Document Is

This is a comprehensive context document for continuing development of Lorien, a simulation-first narrative engine. It captures the architecture, mathematical foundations, implementation state, experimental findings, and current development frontier as of February 2026. Any AI assistant reading this should be able to understand exactly what the system does, how it works, what has been proven, and what comes next.

---

## The Core Thesis

Current LLMs generate fiction by sequential token prediction — each sentence follows from the last. This produces "episodic" storytelling where events feel locally coherent but globally arbitrary. Great fiction (ASOIAF, The Wire, etc.) feels like a "cathedral" because it's built from deliberate structural relationships: promises made early and paid off late, characters whose persistent goals force specific outcomes, locations seeded before becoming plot-relevant, and parallel events that create thematic resonance.

Lorien inverts the generation paradigm. Instead of generating stories token-by-token, it:

1. **Simulates** a deterministic multi-agent world where characters with goals, beliefs, and constraints interact
2. **Scores** the resulting events using mathematical tension/significance metrics
3. **Extracts** the most compelling protagonist arc using a grammar-constrained search
4. **Narrates** that arc into prose using an LLM as a rendering engine

The LLM touches only step 4. Steps 1-3 are deterministic, reproducible, and model-independent. The structural quality of stories is entirely determined by agent definitions, seed, and inherited world state — not by which language model renders the prose.

This is the "crystallization" paradigm: narrative elements gain resolution uniformly across the story world (like crystals forming in supersaturated solution), enabling non-sequential generation where climaxes can be created before beginnings, and global coherence emerges through constraint satisfaction rather than local token prediction.

---

## Architecture

### Pipeline

```
Seed + Canon → Simulation → Events → Metrics → Arc Extraction → Narration → Prose + Lore
                  │                     │            │                │
                  │                     │            │                └─ LLM (Haiku/Grok)
                  │                     │            └─ Grammar search (deterministic)
                  │                     └─ Math (tension, irony, significance)
                  └─ Deterministic Python (agents with goals/beliefs/constraints)
```

### The Simulation Layer

Deterministic multi-agent simulation written in Python. Given a seed and optional canon (inherited world state), produces a sequence of ~200 events. Each event has:

- `event_type`: CHAT, CONFLICT, REVEAL, LIE, CONFIDE, CATASTROPHE, PHYSICAL, OBSERVE
- `source_agent`: who initiated
- `target_agents`: who was affected
- `location_id`: where it happened (dining_table, foyer, kitchen, balcony, bathroom)
- Timestamps, belief changes, tension contributions

The simulation is fully deterministic — same seed + same canon = same events, every time. Canon persistence changes the simulation from event 0 onward (verified: inherited beliefs and tension residue alter early action selection immediately).

**In-simulation tension decay**: Each tick multiplies tension by 0.97 (within a single run).

### The Dinner Party Scenario

The current (and only) scenario is a dinner party with 6 agents:

| Agent | Role | Goal | Archetype |
|-------|------|------|-----------|
| **Thorne** | Host/husband | Maintain normalcy | The Unknowing |
| **Elena** | Wife (having affair) | Conceal the affair | The Guilty |
| **Marcus** | Financial advisor (affair partner) | Conceal both affair and financial fraud | The Double Concealer |
| **Victor** | Journalist | Expose the truth | The Investigator |
| **Lydia** | Mediator/friend | Keep the peace | The Mediator |
| **Diana** | Observer (knows the secret) | Protect herself | The Witness |

5 locations: dining_table, foyer, kitchen, balcony, bathroom.

The scenario is a pressure-cooker: six people with directly opposed goals confined in a small space. This design is intentional — it produces maximum structural variety from a minimal setup, enabling deep analysis of one scenario rather than shallow analysis of many.

### The Metrics Layer

Pure math, no LLM. Computes per-event scores:

- **Tension**: Accumulated pressure from conflict, deception, revelation
- **Irony**: Gaps between character beliefs and world truth (dramatic irony)
- **Significance**: Combined metric weighting tension, irony, and structural importance
- **Segmentation**: Scene boundaries detected from tension dynamics

Location tension residue persists in canon between stories.

### Arc Extraction (Grammar-Constrained Search)

The arc extractor searches for the best protagonist arc through the event sequence. It uses a beat grammar that classifies events into narrative beats:

- **SETUP**: Establishing character, situation, relationships
- **COMPLICATION**: New information or obstacles
- **ESCALATION**: Tension increasing, stakes rising
- **TURNING_POINT**: The moment everything changes (highest-scored single event)
- **CONSEQUENCE**: Aftermath and resolution

The grammar enforces structural constraints:
- Setup must precede complication
- Escalation must precede turning point
- Turning point is singular (one per arc)
- Consequence follows turning point
- Minimum event counts per beat type

An arc is **valid** if it satisfies all grammar constraints. An arc **score** is computed from event significance, beat distribution quality, and structural coherence.

**Proven reliability**: 50/50 seeds produce valid arcs (100% after Phase 4 fixes). Previously 33/50 before instrumentation and targeted bug fixes.

### Rashomon Extraction

Extension of arc extraction: extract one protagonist arc per agent (6 total) from a single simulation. Each agent's arc selects different events from the same simulation, producing genuinely different stories from the same world.

**Results across 50-seed sweep**:
- 294/300 valid arcs (98%)
- Mean 5.88 valid arcs per seed (out of 6)
- 44/50 seeds produce all 6 valid (88%)

**Per-agent validity and mean scores**:

| Agent | Validity | Mean Score | Notes |
|-------|----------|------------|-------|
| Lydia | 98% | 0.694 (highest) | Mediator containment→failure arc is structurally cleanest |
| Victor | 100% | 0.686 | Investigation arc always valid |
| Marcus | 98% | 0.684 | Concealment arc |
| Thorne | 100% | 0.674 | Discovery arc |
| Diana | 96% | 0.663 | Observer arc |
| Elena | 96% | 0.644 | Guilty party arc (most fragile) |

Key finding: the mediator (Lydia) produces the highest-scoring arcs. Containment failure is maximally dramatic — the character whose role is *holding tension in* produces the cleanest structural arc when that containment breaks.

### Overlap Matrix

Measures how much two protagonists' arcs share events (Jaccard similarity). Uses flat unique-pair keys: `"A-B"` where A<B lexicographically (Jaccard is symmetric; storing both directions is redundant).

**Highest overlap** (most entangled arcs):
- thorne-victor: 0.349 (investigation partners)
- lydia-thorne: 0.346 (emotional explosion hub)
- marcus-victor: 0.316 (opposed goals)

**Lowest overlap** (most independent perspectives):
- elena-marcus: 0.137 (co-conspirators experience the party through different events)
- diana-marcus: 0.138

### Wound Patterns (Structural Attractors)

"Wounds" are recurring structural patterns — specific agent-pair × location × beat-type tuples that appear as escalation or turning points across multiple seeds. Discovered through statistical analysis of the 50-seed Rashomon sweep.

**Invariant wounds** (fire in virtually every simulation):

| Pattern | Frequency | Why |
|---------|-----------|-----|
| marcus-victor @ dining_table | 94% | Investigation wound: opposed goals (conceal vs. expose) |
| diana-thorne @ dining_table | 90% | Dramatic irony wound: belief asymmetry (Diana knows, Thorne doesn't) |
| thorne-victor @ dining_table | 90% | Discovery wound: investigation partnership |
| elena-thorne @ dining_table | 86% | Secret-keeper/husband wound |

**Notable pattern**: elena-marcus @ dining_table fires only 38% — the co-conspirators don't produce escalation *with each other*, only through interactions with others. They are catalysts, not opponents.

**(none)-thorne @ dining_table | observe**: 30% — Thorne alone, observing. Internal detective-fiction beat emerging without programming.

### Narration Layer

Two-model architecture:
- **Creative tier**: Claude Haiku 4.5 — prose generation, subtext, interiority
- **Structural tier**: Grok 4.1 Fast — JSON validation, scene boundary enforcement, lore extraction

Cost per story: ~$0.06. Five-story chain with full narration: ~$0.30.

### Canon and Texture

**Canon (WorldCanon)** carries state between stories:
- `claim_states`: Per-agent beliefs about world facts (categorical strings)
- `location_memory[loc].tension_residue`: Accumulated tension per location
- `texture`: Dictionary of `CanonTexture` facts (specific details invented during narration)

**Two accumulation channels** (critical finding — see Experimental Results):
- **Simulation state** (beliefs + tension): Feeds back into simulation. Changes event generation from step 0.
- **Texture** (narrator-invented details): Feeds into narration prompts only. Does NOT feed back into simulation.

These channels are currently **orthogonal** — texture does not influence structure. See findings below.

### Lore Loop

After narration, the system extracts:
- **Canon facts**: World-truth claims (who revealed what, what was confirmed)
- **Texture facts**: Specific invented details (wine labels, physical gestures, room descriptions)

Texture facts are committed to `WorldCanon.texture` with canonical commit IDs (`{generation_id}__{tf_id}`) to prevent cross-story key collisions. These facts are injected into subsequent narration prompts, making each successive story's prose richer and more specific.

---

## Tech Stack

- **Runtime**: Python (Bun for frontend), `.venv` virtual environment
- **Simulation**: Pure Python, deterministic
- **Metrics**: Pure Python math
- **Visualization**: React, TypeScript, HTML Canvas, D3.js, Zustand
- **Narration**: Anthropic API (Claude Haiku 4.5), xAI API (Grok 4.1 Fast)
- **Testing**: pytest (177+ tests as of Phase 2.5)
- **Frontend**: Tailwind CSS, shadcn/ui

### Key Files

```
src/engine/
├── narrativefield/
│   ├── schema/
│   │   └── canon.py              # WorldCanon, CanonTexture, claim_states
│   ├── extraction/
│   │   ├── arc_search.py         # Single-protagonist arc extraction (DO NOT MODIFY — proven 50/50)
│   │   └── rashomon.py           # Multi-protagonist extraction (RashomonArc, RashomonSet)
│   ├── simulation/               # Deterministic agent simulation
│   ├── storyteller/
│   │   └── narrator.py           # SequentialNarrator.generate()
│   └── tests/
│       ├── test_rashomon.py      # 7 tests
│       └── test_canon_decay.py   # 6 tests (being implemented)
├── scripts/
│   ├── demo_lore_loop.py         # A→B→C chain with lore extraction
│   ├── demo_canon_persistence.py # Canon persistence A/B demo
│   ├── sweep_rashomon.py         # 50-seed Rashomon sweep
│   ├── analyze_wounds.py         # Statistical wound analysis
│   ├── research_chain.py         # 5-story research chain with Rashomon at each step
│   └── output/                   # Artifacts (.gitignore'd)
│       ├── rashomon_sweep_1_50.json
│       ├── wound_analysis_1_50.json
│       ├── research_chain.json          # Skip-narration baseline
│       ├── research_chain_full.json     # Full narration run
│       ├── research_chain_reverse.json  # Seed-order experiment
│       ├── research_chain_seed7first.json
│       └── research_chain_seed7last.json
```

---

## Experimental Results

### Phase 4: Lore Loop (50-seed sweep)

- **Arc validity**: 50/50 seeds (100%), up from 33/50 before instrumentation fixes
- **Texture compounding**: 0→9→24→36 facts across A→B→C→D chain (steady, no saturation)
- **Canon divergence**: Inherited state changes simulation from event 0 (verified by A/B comparison — same seed, different canon produces different story from first event)
- **Cost**: $0.05-0.07 per story

### Phase 2: Rashomon + Wounds (50-seed sweep)

- **Rashomon validity**: 294/300 (98%), mean 5.88/6 valid per seed
- **10 wound patterns** above 25% threshold
- **3 invariant wounds**: marcus-victor, thorne-victor, elena-thorne (all fire 5/5 in every chain ordering tested)
- **1 null wound**: elena-marcus never fires (0/5 in chains, 38% in population) — co-conspirators are catalysts, not opponents

### Phase 2.5: Research Chain (5-story chain with Rashomon)

**Skip-narration vs. full narration**: All structural measurements are **identical**. Same validity counts, same scores to full precision, same overlap matrices, same wound persistence. Texture accumulation (0→15→28→42→58→69) has zero effect on structure.

**Implication**: Texture and simulation state are orthogonal channels. Texture enriches prose but does not influence the simulation that generates events.

**The U-Shape**: Mean scores across the 5-story chain show:
```
Story A: 0.697 → Story B: 0.672 → Story C: 0.631 → Story D: 0.626 → Story E: 0.706
```

Initially interpreted as an emergent "compression → recovery" dynamic (possibly 5-act structure). This interpretation was **killed** by the seed-order experiment.

### Seed-Order Experiment (4 chains, same 5 seeds, different orderings)

Ran chains: Original (42,51,7,13,29), Reverse (29,13,7,51,42), S7-First (7,42,51,13,29), S7-Last (42,51,13,29,7). All skip-narration, zero cost.

**Key finding: The U-shape was an artifact of seed ordering, not emergent structure.**

Position means across all 4 chains:
```
Position A (depth 0): mean 0.696
Position B (depth 1): mean 0.679
Position C (depth 2): mean 0.644
Position D (depth 3): mean 0.650
Position E (depth 4): mean 0.637
```

**Accumulated canon monotonically suppresses arc scores.** Degradation rate: ~0.015 per chain position. A 12-episode season would degrade to ~0.52 by episode 8-9, approaching the validity threshold.

**Seed 7 analysis**: Not inherently weak. Scores 0.721 at depth 0 (highest single measurement across all 20 data points). Scores 0.614-0.631 at depth 2. The interaction between seed and accumulated state determines score — neither seed nor position alone explains it.

**Seed 51**: Most robust. Spread of only 0.025 across all positions. Some seeds are constraint-tolerant; others are constraint-fragile.

**Hypothesis classification**:
- ~~Seed property~~ (seed 7 always low): **Disproven** — scores 0.721 at depth 0
- ~~Position property~~ (mid-chain always low): **Partially supported** — general downward trend, but variance is high
- **Interaction** (seed × canon depth): **Confirmed** — the dominant effect is how a particular seed's event patterns interact with the specific accumulated belief state

---

### Canon Decay Experiment (Failed — Revealed Architectural Gap)

**Branch**: `feature/canon-decay`

Implemented `decay_canon()` in `canon.py` with two parameters:
- `tension_decay` (float, default 1.0): Multiplier for location tension residue between stories
- `belief_decay` (float, default 0.85): Multiplier for belief confidence between stories (added `claim_confidence` dict to WorldCanon, backward-compatible)

CLI args added to `research_chain.py` (`--tension-decay`, `--belief-decay`). 6 new tests in `test_canon_decay.py`. 176 tests passing.

**Result: Zero effect.** All structural measurements identical to no-decay runs — same scores to full precision, same validity, same overlap, same wound persistence. Decay at tension=0.6, belief=0.85 produced:

```
NO DECAY:    0.696 → 0.679 → 0.644 → 0.650 → 0.637  (degradation: -0.059)
WITH DECAY:  0.696 → 0.679 → 0.644 → 0.650 → 0.637  (degradation: -0.059)
```

**Root cause**: The decay function is mathematically correct but causally disconnected from the simulation's decision path. Three specific gaps:

1. **Belief confidence vs. categorical states**: The simulation reads `claim_states` — categorical strings like "known", "suspected", "unknown." Decay operates on `claim_confidence` (a new float field), but the simulation doesn't read confidence. Decaying 1.0 → 0.85 → 0.72 → 0.61 → 0.52 never crosses the 0.1 pruning threshold, so no beliefs get removed, and the categorical states the simulation actually consumes are unchanged. Inherited beliefs stay at 26-27 across all stories regardless of decay.

2. **Tension residue vs. simulation dynamics**: Between-story tension residue may not be the field the simulation reads when making event decisions, or the within-run tension decay (×0.97 per tick) dominates so completely that starting residue is irrelevant by the time structurally important events occur.

3. **Pattern consistency**: This is the third instance of the same architectural pattern — canon stores data that narration and reporting consume, but the simulation's decision path is narrower than the canon schema suggests. Texture was structurally inert (narration-only). Belief confidence is structurally inert (reporting-only). The simulation's actual input interface is narrower than the canon object.

**Critical prerequisite before further decay work**: Trace the simulation's actual read path. Specifically:
- When `run_story(seed, canon)` executes, which exact fields from WorldCanon does the tick loop consume when deciding what event to generate?
- Are those fields the ones being decayed, or are they reconstructed from scratch each run?
- What would it take to change the categorical `claim_states` values (e.g., resetting low-confidence beliefs to "unknown") such that the simulation generates different events?

Meaningful decay requires either changing the categorical states themselves, reducing the belief count by actually removing entries, or modifying the simulation's input interface to read confidence. The current approach decays a field nobody reads.

---

## Current Development Frontier

### The Simulation Input Interface Problem

The central finding from the research chain, seed-order experiment, and decay experiment is that **the simulation's decision engine is more isolated from canon than the architecture diagrams suggest.** Canon carries rich state between stories, but the simulation may only consume a narrow slice of it.

Three layers of evidence:
1. **Texture**: 69 accumulated facts, zero structural effect — texture feeds narration only
2. **Belief confidence**: Decayed correctly, zero structural effect — simulation reads categorical states, not confidence floats
3. **Tension residue**: Decayed correctly, zero structural effect — within-run dynamics likely dominate starting values

Before building more sophisticated decay, governance, or feedback mechanisms, the immediate engineering task is **mapping the simulation's actual input interface**: which specific canon fields does the tick loop read, and what changes to those fields produce measurable changes in event generation?

### Future Architectural Moves (Not Yet Started)

1. **Texture → Simulation feedback**: Inject selected texture facts into agent context before simulation. Currently texture is structurally inert — it enriches prose but doesn't change what events agents generate. Closing this loop would make accumulated world-specificity change simulation behavior, not just narration quality.

2. **Second scenario**: Different setting, different agent configurations, different secret topology. Tests whether the grammar, wound detection, and Rashomon extraction generalize beyond the dinner party. Should be designed after texture feedback is implemented, so the scenario can exploit texture-informed simulation.

3. **Season orchestration**: 12-story chain with Rashomon rotation (different protagonist each episode), accumulated canon, and aggregate visualization showing the terrain of the whole season.

4. **Canon governance**: Contradiction detection, belief reinforcement (beliefs that recur across simulations maintain/increase confidence), selective pruning based on narrative relevance.

---

## Key Principles

### Depth Over Breadth
Stay with one scenario and deepen analysis. Testing 6 dramatically different protagonist archetypes within one scenario provides stronger evidence of grammar generality than testing 5 shallow scenarios. Depth produces discoveries; breadth produces demos.

### Cheap Experiments First
$0.06/story and $0.00/structural-experiment means you can run 50 experiments for the price of one expensive model call. The seed-order experiment that killed the "emergent 5-act structure" narrative cost $0.00 and took 6 minutes. Always run `--skip-narration` before committing to full narration runs.

### Structure is Model-Independent
The simulation, metrics, arc extraction, and Rashomon analysis don't use any LLM. Swapping Claude for GPT or Gemini or a local model changes prose quality but produces identical structural measurements. The value is in the structure, not the rendering.

### Don't Build Governance Before Observing Breakdowns
The research chain was designed to observe where canon actually breaks down rather than engineering solutions to unobserved problems. The seed-order experiment proved the degradation is real and measured its rate. Now canon decay is being built against known data, not speculative concerns.

### One Parameter, Then Measure
Don't build three interacting systems when one parameter sweep on a simple function will tell you how much complexity you actually need. The measurement infrastructure (research chain + seed-order comparison) exists and is free to re-run.

---

## Cost Profile

| Operation | Cost | Time |
|-----------|------|------|
| Single story (full narration) | ~$0.06 | ~3 min |
| 5-story chain (full narration) | ~$0.30 | ~15 min |
| 5-story chain (skip-narration) | $0.00 | ~2 min |
| 50-seed Rashomon sweep | $0.00 | ~10 min |
| Wound analysis | $0.00 | seconds |
| 4-chain seed-order experiment | $0.00 | ~8 min |

---

## Test Counts

- **177+ tests** passing (pytest)
- 170 baseline (Phase 4) + 7 Rashomon tests + canon decay tests (in progress)
- Arc extraction: 50/50 seeds valid
- Rashomon: 294/300 arcs valid across 50-seed sweep

---

## What Has Been Disproven

- ~~"The U-shape is emergent 5-act structure"~~ — It's seed × canon interaction, not a structural phase transition
- ~~"Texture accumulation enriches narrative structure"~~ — Texture and structure are orthogonal; texture enriches prose only
- ~~"Constraint accumulation improves arc quality over time"~~ — It degrades it, at ~0.015 per chain step
- ~~"Lydia (the mediator) would produce weak arcs"~~ — She produces the highest-scoring arcs of all 6 agents
- ~~"Simple exponential decay on canon fields will flatten score degradation"~~ — Decay operates on fields the simulation doesn't read; the simulation's input interface is narrower than the canon schema

## What Has Been Proven

- Deterministic simulation + grammar-constrained extraction produces valid dramatic structure (100% at single-protagonist, 98% at 6-protagonist)
- Six fundamentally different character archetypes produce valid arcs from the same grammar — the grammar captures something universal about dramatic structure
- Wound patterns are real structural properties of agent configurations, invariant across seed orderings
- Canon persistence changes simulation from event 0 — inherited state is causally active, not cosmetic
- Texture accumulation is healthy (steady rate, no saturation at 69 facts) but structurally inert
- The simulation's decision engine is more isolated from canon than the architecture suggests — three independent experiments (texture, confidence decay, tension decay) all produced zero structural effect
- The system produces 13,900 words of narrated fiction across 5 chained stories for $0.30 in 15 minutes
