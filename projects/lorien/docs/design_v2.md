# NarrativeField v2: The Synthesized Plan

## What Changed (And Why)

Two rounds of critique converged on the same core insight: **the original plan had the right vision but the wrong build order**. The math was beautiful but floated above the engineering. This revision stitches them together — keeping every idea that earned its place, killing the ones that were "math cosplay," and adding the genuinely new concepts that emerged from the critique.

The three biggest shifts:

1. **Graph-first, manifold-second.** The ground truth is a directed event graph (the *fabula*). The topological surface is a *lens* over that graph (the *syuzhet*), not the world itself. This prevents you from confusing a visualization artifact with a narrative structure.

2. **"Narrative Physics" instead of real physics.** Rational agents produce boring simulations. Your agents need to be pulled toward *drama*, not stability. The simulation isn't modeling reality — it's modeling *stories about* reality.

3. **Build the visualizer first, with fake data.** The "crystallizing map" is the product. If it doesn't feel magical with hand-crafted data, no amount of simulation sophistication will save it. Phase 1 is the renderer, not the engine.

---

## The Architecture (Two Layers, Clean Separation)

### Layer 1: The Fabula Graph (What Happens)

A directed multigraph that serves as ground truth for the entire system.

**Nodes:** World-states or event-states. Each node is a snapshot:

```
WorldState {
    timestamp: float
    agents: Map<AgentID, AgentState>
    world: GlobalState
    active_tensions: List<Tension>
}

AgentState {
    id: string
    position: LocationID
    goals: Vec<Goal>            // ranked desires
    relationships: Map<AgentID, RelationshipState>
    beliefs: Map<AgentID, BeliefSet>  // what they THINK is true
    resources: Map<string, float>
    commitments: List<Commitment>     // irreversible choices made
    emotional_state: EmotionalVector  // anger, fear, hope, etc.
    secrets: List<Secret>             // things others don't know
}

RelationshipState {
    trust: float        // -1 to 1
    affection: float    // -1 to 1
    obligation: float   // 0 to 1
    history: List<EventID>  // shared events
}
```

**Edges:** Events with causes, participants, and structured consequences.

```
Event {
    id: EventID
    timestamp: float
    type: EventType     // trade, betray, confess, threaten, ally, travel, etc.
    participants: List<AgentID>
    location: LocationID
    trigger: CausalChain        // what caused this
    deltas: List<StateDelta>    // explicit, structured changes
    counterfactual_impact: float  // computed later (see Meaning section)
}

StateDelta {
    target: string              // e.g., "relationship[A,B].trust"
    operation: "set" | "add"
    value: any
    reason: string              // human-readable cause
}
```

**Why event sourcing matters:** Every metric, every visualization, every story extraction is *derived* from this log. You can replay, branch, diff two worlds, and do "what if" comparisons cleanly. The event log is your database. Everything else is a view.

### Layer 2: The Syuzhet Field (How It's Presented)

This is where the topological map lives. It doesn't *replace* the graph — it *renders* it for human navigation.

The syuzhet field takes the fabula graph and produces:
- An embedded 2D/3D space (the "terrain")
- Scalar fields (tension, meaning, thematic coordinates)
- Character trajectories (curves through the embedded space)
- Structural annotations (convergence points, turning points, loops)

**Critical principle from Critique 1:** Topology comes from your *metric*, not your *plot*. Compute topological features (persistent homology, connected components) from distances in the original feature space — then visualize them on the embedded surface. Never compute topology from the UMAP output.

**Critical principle from Critique 2:** Use a *pinned manifold*, not free embedding. The x-axis is always time. This prevents the "blob problem" where UMAP destroys the arrow of time and produces unreadable clusters.

---

## The Simulation Engine: "Narrative Physics"

### The Dwarf Fortress Problem (And the Fix)

If you simulate rational agents, you get *The Sims*. Agents eat, sleep, avoid danger, accumulate resources, and nothing interesting happens. This is because rational agents minimize risk and maximize utility — they converge to equilibrium.

Stories require the opposite: **agents must be pulled toward disequilibrium.**

The fix is to replace "real physics" with "narrative physics" — a set of forces that are psychologically plausible but dramatically productive.

### Narrative Gravity: Forces That Create Stories

In physical simulation, particles minimize potential energy. In narrative simulation, agents should (unconsciously) maximize **dramatic potential**. Here's how:

**Force 1: Goal Tension (The Gradient)**

Each agent has a utility function U_i over world-states. The gradient ∇U_i points toward their desires. This is standard.

But the key insight: **conflicting gradients create stories**. When Agent A's gradient points directly opposite to Agent B's, you get a force collision. The magnitude of the cross-product ‖∇U_A × ∇U_B‖ at their intersection is a computable measure of *conflict intensity*.

**Force 2: Dramatic Attractors (The Warps)**

Beyond individual goals, add *narrative attractors* — regions in state space that pull agents toward dramatically productive configurations:

- **Proximity to secrets being revealed** (information asymmetry approaching zero)
- **Approach to irreversible thresholds** (points of no return)
- **Convergence of antagonists** (agents with opposed goals moving into shared space)

These attractors warp the "terrain" agents navigate. They don't force outcomes — they make certain configurations more likely, the way a valley in a landscape makes water likely to flow there without dictating which specific path it takes.

**Force 3: Irrational Pulls (The Flaws)**

Rational agents are boring. Add character-specific biases:

- **Pride:** overestimates own capability, underestimates threats
- **Loyalty:** continues alliances past the point of self-interest
- **Trauma:** avoids situations resembling past pain, even when avoidance is costly
- **Ambition:** overweights long-shot high-reward paths
- **Love:** biases perception of another agent's trustworthiness

These aren't "bugs" in the decision engine — they're the *source* of story. Hamlet doesn't act because of character flaw. Macbeth acts too much because of character flaw. Both are stories *because* the agent is irrational.

### User-Seeded Attractors: "Promises"

Writers don't want a pure slot machine. They want to set up constraints:

- "There will be a revolution by the midpoint"
- "Character X and Y must meet"
- "A betrayal must occur"

These become **soft constraints** — potential wells in the simulation's state space. The simulation naturally flows toward them without being forced. If the constraint is impossible given the current agent configurations, the system reports this as "unreachable" — which is itself useful information ("your world doesn't support this story yet, here's why").

In the field metaphor: attractors warp the terrain, but agents still choose paths through it.

---

## The Math of Meaning: Three Computable Primitives

The original plan used entropy as the primary meaning metric. Both critiques agreed: entropy captures *uncertainty*, but meaning in stories comes from something deeper. Here are three primitives that are both mathematically precise and narratively resonant.

### Primitive 1: Counterfactual Impact (The Best One)

**Meaning ≈ "How much does this event change the future?"**

Formally: simulate forward N times from the state *before* event e, and N times from the state *after* event e. Compare the distributions over futures.

```
Impact(e) = D_KL( P(future | after_e) || P(future | before_e) )
```

Or use Wasserstein distance for a more geometrically meaningful measure.

**Intuition:** A meaningless event is one where the future barely changes whether it happened or not. A meaningful event creates a new basin of outcomes — it "changes everything." This is exactly what we mean when we say a moment in a story "mattered."

**Implementation:** This requires running the simulation forward multiple times from branch points. Expensive, but parallelizable, and you don't need to do it for every event — only for events that cross certain tension thresholds or that the user specifically queries.

### Primitive 2: Catastrophe / Discontinuity (Plot Twists)

**From Critique 2 — Catastrophe Theory (René Thom):**

Smooth calculus handles smooth changes. But the most meaningful story moments are *discontinuous* — sudden betrayals, realizations, reversals.

The **cusp catastrophe** models this: imagine a surface that folds over itself. An agent moves along the smooth top layer as stress slowly increases. At some point they hit the fold edge and *drop* instantaneously to the bottom layer — a phase transition, a "snap."

**Application to the visualization:** Your topological map shouldn't just be rolling hills. It needs **cliffs**. The height (tension) builds smoothly, then drops sharply at catastrophe points. When a user hovers over a cliff edge, they see the mounting tension; the fall is the plot twist.

**Computable:** Track state variables that are approaching bifurcation thresholds. When a continuous increase in one variable (stress, distrust, resource depletion) causes a *discontinuous* change in behavior (alliance flipping to war, trust collapsing to betrayal), that's a catastrophe point. Flag it.

### Primitive 3: Cohomology Failure / Dramatic Irony

**From Critique 2 — Sheaf Theory applied practically:**

Meaning often comes from the gap between what a character *thinks* is true and what *is* true. This is dramatic irony — the audience knows the bomb is under the table, but the characters don't.

**Formalization:**

Each agent has a *belief set* (their "local section" — what they perceive of the world). The actual world state is the "global section." Dramatic irony is the divergence between them.

```
Irony(agent_i, t) = distance( beliefs_i(t), truth(t) )
```

More precisely, for any proposition p:

```
irony_score(i, p) = |P_i(p) - P_world(p)|
```

Where P_i(p) is agent i's credence in proposition p and P_world(p) is the ground truth (0 or 1).

**Scene-level irony** = the sum of irony scores across all agents in a scene. High scene irony = the audience knows something the characters don't = suspense.

**Application:** Color-code the map by irony density. Areas of high cohomology failure (many agents holding false beliefs) are scenes of high suspense potential. The moment irony collapses to zero (the reveal) is a peak event.

### Bonus Primitive: Irreversible Commitment

Track when agents cross points of no return:

- A promise made publicly
- A bridge burned (relationship destroyed)
- A resource spent that can't be recovered
- A secret revealed that can't be un-known

**Commitment score** = the count/weight of irreversible choices made. Meaning spikes when characters give up something they can't get back. This is measurable in a rule-based world without any statistical machinery.

---

## Tension as a Tunable Lens (Not a Decree)

**From Critique 1:** If you hardcode a single tension formula, you'll argue with your own tool. Different stories have different tension signatures.

**Fix:** Tension is a *weighted bundle of interpretable sub-metrics*, and users can tune the weights:

```
Tension(state, weights) = Σ wᵢ × metric_i(state)
```

Sub-metrics:

| Metric | What It Captures |
|--------|-----------------|
| `danger` | Physical threat to focal agent |
| `time_pressure` | Approaching deadlines or irreversible thresholds |
| `goal_frustration` | Distance between desired and actual state |
| `relationship_volatility` | Rate of change of trust/affection |
| `information_gap` | How much the agent doesn't know (but should) |
| `resource_scarcity` | Competition for limited things |
| `moral_cost` | The price of the available choices |
| `irony_density` | Cohomology failure (see above) |

**Default profile:** Equal weights. But a user writing a thriller would crank up `danger` and `time_pressure`. A user writing a relationship drama would crank up `relationship_volatility` and `moral_cost`. A user writing a mystery would crank up `information_gap` and `irony_density`.

This makes tension a *slider-defined lens*, not a fixed formula.

---

## Thematic Fields: What People Actually Mean by "Meaning"

**From Critique 1:** Entropy measures uncertainty. But meaning in stories is often about *thematic drift* — the way a story slowly shifts from hope to despair, or from innocence to corruption.

Add a small set of **thematic axes** (hand-designed initially):

- **Loyalty ↔ Betrayal**
- **Freedom ↔ Control**
- **Love ↔ Duty**
- **Innocence ↔ Corruption**
- **Truth ↔ Deception**
- **Order ↔ Chaos**

Each event nudges these coordinates based on what happened:

```
ThematicShift {
    event_id: EventID
    axis: "loyalty_betrayal"
    direction: -0.3    // toward betrayal
    agent: AgentID
}
```

The map can then show not just tension peaks but **thematic trajectories** — how the story drifts across these dimensions over time. A story that starts in "loyalty/innocence" territory and ends in "betrayal/corruption" territory has a recognizable shape (tragedy). One that starts in "control/deception" and ends in "freedom/truth" is a liberation arc.

This gives users a way to *search by shape*: "Find me a path that moves from loyalty to betrayal" is a query with a well-defined geometric answer.

---

## The Visualization: "Wavefunction Collapse" Interface

### The Concept

**From Critique 2, refined with Critique 1's pragmatism:**

The visualization has three states, like quantum mechanics:

**1. The Cloud (Possibility) — Zoomed Out**
When zoomed out, don't draw individual lines. Render a **volumetric fog** or density heatmap. The density represents the *probability of interesting events* at that time/space coordinate. This is the "whole world" view — you see where stories cluster without seeing specific stories.

**2. The Hover (Observation) — Mid Zoom**
As the cursor approaches a region, the system performs real-time pathfinding. Bright **streamlines** shoot out from the cursor position — these are the specific causal chains (backward: "what led to this?", forward: "what follows from this?") connected to the hovered event. Other paths dim.

**3. The Crystallization (Selection) — Clicked**
When you click, the selected character arc "crystallizes" — it becomes a solid, high-contrast curve on the surface. Related arcs (characters who interact with the selected one) glow at lower intensity. Unrelated arcs fade to near-transparency. You're now looking at *one story* in the world.

**4. The Drag (Interaction) — Power Feature**
Grab a character's line and drag it over a tension peak. The system re-simulates from that decision point to find how other characters' paths warp in response. This is your "what if" engine — and it's visually stunning because you literally see the ripples propagate.

### 2D Default, 3D Delight

**From Critique 1:** 3D is seductive but hard to read, has occlusion problems, and risks being a "cool demo" instead of a daily tool.

**Ship 2D as default:**
- x-axis: time (always pinned)
- y-axis: character lanes / faction clusters / thematic position
- color/intensity: tension (heatmap)
- curves: character arcs as paths through events

**Optional 3D mode:**
- x-axis: time
- y-axis: character/faction space
- z-axis: tension (the terrain height)
- This is for exploration, marketing, and the "wow" moment

This mirrors how scientific visualization tools work: 2D for comprehension, 3D for exploration.

### The Pinned Manifold (Solving the Blob Problem)

**From Critique 2:** Standard UMAP/t-SNE destroys the arrow of time. You get clusters, not paths.

**Fix:** The x-axis is *always time*. Only the y-axis (and z in 3D) comes from embedding. This guarantees:
- Flow lines always move left-to-right (forward in time)
- The map is readable as a timeline
- Convergence/divergence of characters is visible as curves approaching/separating in the y-direction

For the y-axis embedding, use spectral methods on the interaction graph (at each time slice, how tightly clustered are different characters?) rather than generic dimensionality reduction.

---

## Story Queries: The Real Killer Feature

**From Critique 1:** Instead of only visual exploration, let users *ask* for stories.

This turns the map from a visualization into a **story search engine**:

- "Find me a redemption arc for Character X"
- "Show me arcs where two rivals become allies"
- "Give me the most 'inevitable' tragedy"
- "Find convergences where three factions collide"
- "Show me arcs where Character X's loyalty flips under pressure"
- "Give me a tragedy route, not a hero route"

**Implementation:** These are path-search queries over the event graph, optimizing a weighted combination of:
- Counterfactual impact (the path includes meaningful events)
- Tension profile (matches the requested shape — rising action, climax, resolution)
- Thematic trajectory (moves in the requested direction)
- Coherence (the events are causally connected, not cherry-picked)

The map becomes the *explanation layer*: "here's the arc I found, and here's *why* it scores high — you can see the tension peak here, the turning point here, the thematic shift here."

---

## Topology: Aimed Correctly

### What TDA Actually Buys You (For the MVP)

**Persistent H₀ (connected components):**
- "How many major story clusters exist at different interaction thresholds?"
- "When do isolated arcs connect?"
- This directly answers structural questions writers care about.

**Persistent H₁ (loops):**
- Treat with caution early. Loops can mean genuine cyclical structure, but can also mean simulation artifacts (agents dithering) or metric artifacts.
- Use as a *flag* ("possible cyclical dynamic"), not an automatic "theme."

**Morse Theory (critical points):**
- Apply to the tension scalar field on the manifold.
- Maxima = crisis points. Minima = rest points. Saddle points = where storylines merge or split.
- This is genuinely useful for automatic structural annotation ("this is where Act 2 begins").

### Computing Topology Correctly

**From Critique 1:** Never compute topology from the 2D/3D embedding. Compute it from a well-defined distance metric in the original feature space.

Define distance between two world-states as:

```
d(S₁, S₂) = α × agent_state_distance(S₁, S₂)
           + β × relationship_graph_distance(S₁, S₂)
           + γ × resource_distance(S₁, S₂)
           + δ × belief_divergence(S₁, S₂)
```

With weights tunable. Then run persistent homology on the Vietoris-Rips complex built from this distance. The resulting persistence diagrams are trustworthy structural features — they don't depend on how you happen to project the data for visualization.

---

## The MVP: "The Dinner Party Protocol"

**From Critique 2, refined with Critique 1's event-sourcing discipline.**

Do not start by simulating a kingdom. Start by simulating a room. This is constraint as liberation — a dinner party has enough social complexity to produce real narrative structure, but small enough scope that you can hand-verify every output.

### Scope

- **Setting:** 1 dining room, 1 evening (2-3 hours of simulation time)
- **Characters:** 6 agents
- **State dimensions per agent:** Affection, Jealousy/Envy, Trust, Alcohol Level, Secret Knowledge, Goal Urgency
- **Event types:** Speak (to whom, about what), Confide, Accuse, Flirt, Lie, Reveal, Leave the Room, Confront
- **Location detail:** Table seating (who's next to whom matters), a balcony (private conversation space), a kitchen (overhearing space)

### Why This Works

A dinner party naturally produces:
- **Convergence:** Everyone is in the same room
- **Information asymmetry:** Characters have secrets, partial knowledge
- **Social pressure:** Can't easily leave, forced proximity
- **Escalation:** Alcohol + accumulated tension → catastrophe points
- **Multiple valid stories:** The same dinner can be a comedy, a tragedy, a mystery, or a thriller depending on which character you follow

### The Build Order

**Phase 1: The Renderer (Weeks 1-3)**
*Ignore the simulation entirely.* Write a script that generates fake data — a JSON list of 6 characters with pre-scripted events. Build the visualizer:
- The pinned manifold (time × character-space)
- The tension heatmap
- The crystallization hover effect
- Arc selection and highlighting

**Prove the visualization is magical with hand-crafted data.** This is your product demo, your pitch deck, your "oh wow" moment. If it doesn't feel magical with perfect data, no simulation will save it.

**Phase 2: The Toy Simulation (Weeks 4-6)**
Build a simple agent simulation:
- Agents have goals, relationships, secrets, and biases
- Events fire based on proximity + goal activation + randomness + narrative gravity
- Each event produces structured StateDelta objects
- Feed real simulation data into the visualizer from Phase 1

Validate: Does the simulation produce event logs that, when visualized, show recognizable narrative structure? If not, tune the narrative physics, not the renderer.

**Phase 3: The Metrics Layer (Weeks 7-9)**
Compute the meaning/tension/theme metrics on the event log:
- Tension (weighted sub-metrics with default profile)
- Counterfactual impact (branch simulations, expensive but revealing)
- Irony density (belief vs truth divergence)
- Thematic coordinates (per-event shifts along thematic axes)

Feed these into the renderer as color/height/annotation data.

**Phase 4: Story Extraction (Weeks 10-12)**
Select a path on the map. Export:
- Beat sheet (structured event list with tension annotations)
- Scene outlines (event clusters → scene descriptions)
- 1-2 fully written scenes via LLM (proving the structure→prose pipeline)

**Phase 5: Story Queries (Weeks 13-16)**
Implement path search:
- "Find me the most dramatic arc for Character X"
- "Show me where Characters A and B's stories intersect"
- "Give me a tragedy"
The map highlights the results. This is when NarrativeField becomes a *tool*, not a demo.

---

## The Math Stack (Reordered for Reality)

### Tier 1: Required for MVP

| Topic | Why You Need It | Where to Learn |
|-------|----------------|----------------|
| Graph theory (dynamic graphs) | The fabula graph IS the product | Any algorithms textbook + NetworkX docs |
| Probability basics | Sampling futures, computing surprise | EECS 492 review |
| Linear algebra | Embeddings, spectral clustering, PCA | EECS 281/review |
| Basic information theory | Entropy, mutual information | Cover & Thomas Ch. 1-3 |

### Tier 2: High Leverage After MVP

| Topic | Why You Need It | Where to Learn |
|-------|----------------|----------------|
| Markov decision processes | Making agents purposeful, not random | Sutton & Barto Ch. 1-4 |
| Causal inference / counterfactuals | "Meaning = impact" computation | Pearl's "Book of Why" (accessible) |
| Dimensionality reduction | Diffusion maps for the embedding | Coifman & Lafon paper |
| Bifurcation / catastrophe basics | Detecting phase transitions = plot twists | Strogatz Ch. 3, 8 |

### Tier 3: When Metrics Are Stable

| Topic | Why You Need It | Where to Learn |
|-------|----------------|----------------|
| Persistent homology (H₀, H₁) | Structural diagnostics of narrative | Edelsbrunner & Harer Ch. 1-4 |
| Morse theory | Critical points of tension = act boundaries | Matsumoto "An Introduction to Morse Theory" |
| Optimal transport | Wasserstein distance for world comparison | Peyré & Cuturi "Computational OT" |

### Tier 4: Don't Touch Until Shipping

| Topic | Why It Can Wait |
|-------|----------------|
| Sheaf theory / category theory | Gorgeous but won't rescue a weak sim or unclear UI |
| Full differential geometry | The pinned manifold sidesteps most of this |
| Advanced TDA (H₂+, spectral sequences) | Overkill for narrative structure |

---

## Engineering Principles (Non-Negotiable)

1. **Event log is source of truth.** Everything is derived from the append-only event log. This enables replay, branching, diffing, and "what-if."

2. **State deltas are explicit.** Every event produces structured changes: `relationship[A,B].trust += -0.3`, `resource[factionX] -= 2`, `belief[A about B] = "traitor"`. This is what makes visualization and summarization possible.

3. **Never let the LLM drive the simulation.** Use it for summarizing event windows, generating scene prose after selection, writing character voice. But the decision loop is rule-based until the rule world is rock-solid.

4. **Metrics are user-tunable.** Tension weights, thematic axis definitions, "meaning" formula — all configurable. The tool serves the writer's intent, not a hardcoded theory of narrative.

5. **Topology comes from the metric, not the projection.** Compute structural features from well-defined distances in the original feature space. Visualize them on the embedded surface. Never compute topology from the UMAP output.

---

## The Elevator Pitch (Revised)

*"Writers don't need another text generator. They need a way to see the shape of a story before they write it. NarrativeField simulates fictional worlds — agents with desires, secrets, and flaws interacting under pressure — and maps the resulting narratives onto an interactive surface. Peaks are crises. Valleys are calm. The curves snaking across the terrain are character arcs. Hover and the map crystallizes: causal chains shoot out, showing you which events connect. Click, and you've selected a story — with mathematically guaranteed structure, turning points, and payoff. Then an AI writes the prose, filling in texture around a skeleton that already works. It's not AI generating stories. It's AI helping you find the stories that were always there."*

---

## Next Concrete Step

Draft two things:

1. **The Event Schema** — the exact JSON structure of an event, with 5-10 example events from a dinner party scenario. This forces every vague concept to become a data structure.

2. **A Fake Data Script** — 50 hand-written events for 6 characters at a dinner party. This becomes the test data for Phase 1 of the renderer. Writing it by hand will teach you more about what the simulation needs to produce than any amount of architecture planning.

When those two artifacts exist, everything else has a foundation to build on.
