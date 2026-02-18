# NarrativeField: A Topological Approach to Story Generation

## The Core Thesis

Great stories don't emerge from character sheets and plot outlines — they emerge from **worlds rich enough to contain many stories**, and the craft of selecting the most meaningful paths through that world. Think of Tolkien: he didn't write *The Lord of the Rings* and then build Middle-earth. He built Middle-earth, and the stories crystallized out of its internal logic.

The problem is that **worldbuilding is computationally and cognitively expensive**. Writers burn out trying to keep track of how 30 characters with different motivations would realistically interact across time, space, and causality. And when they skip the worldbuilding, stories feel thin — there's no sense that the characters exist when the camera isn't pointed at them.

**NarrativeField** is a tool that inverts the process. You build the world — its agents, constraints, geography, factions, tensions — and the tool simulates it forward, producing a rich topological space of possible narratives. Then you explore that space visually: a 3D surface where each point represents a world-state, and the curves snaking across it are the story arcs of individual characters. Where curves converge, characters interact. Where the surface warps, tension accumulates. You hover, you select, you extract the story that matters.

---

## What You Already Know (And Why It Matters)

From your ASOIAF deep dive, you mapped how character arcs weave through a shared world — Arya's arc touches Tywin's in Harrenhal, diverges entirely from Dany's until late, and intertwines with the Hound's in a way that transforms both. You noticed that Martin's genius is that **each POV character lives in the same world but experiences a different story**, and the richness comes from how those stories interfere with each other.

From *The Weapon's Heart*, you built a mythological scaffold (Persephone/seasonal descent/return) that gave structure to a character's emotional journey through a pre-existing world. The story bible approach — defining the world's rules, then letting the character move through them — is exactly the worldbuilding-first philosophy.

From your systems engineering background (C++, game engines, OS work), you have the instincts for the computational side: state machines, concurrent processes, event-driven architecture.

This project sits at the intersection of all of that.

---

## The Mathematical Framework

### Layer 1: The World as a Dynamical System

A narrative world at its most fundamental level is a **state space** with **evolution rules**.

**State Vector:** At any moment in time, the world has a state:

```
S(t) = (A₁(t), A₂(t), ..., Aₙ(t), W(t))
```

Where each `Aᵢ(t)` is an agent state (position, knowledge, relationships, emotional state, resources, goals) and `W(t)` is the world state (political power structures, resource distribution, active conflicts, environmental conditions).

**Evolution:** The world evolves according to:

```
dS/dt = F(S, t) + ε(t)
```

Where `F` encodes the deterministic rules (physics, social dynamics, economic pressures) and `ε` is stochastic noise (chance encounters, natural disasters, human irrationality). This is just a dynamical system — the kind you'd study in EECS 460 or a nonlinear dynamics course.

**Key insight:** You don't need to solve this analytically. You simulate it — like an N-body simulation, but for narrative agents.

### Layer 2: Character Trajectories as Curves in State Space

Each character traces a **curve** (a 1-dimensional path) through the high-dimensional state space. If the total state space is ℝⁿ, a character's arc is a mapping:

```
γᵢ: [0, T] → ℝⁿ
    t ↦ projection of S(t) onto character i's relevant subspace
```

This is where the Calc 3 / differential geometry intuition kicks in. Think of each character's trajectory as a parametric curve in space. The "story" of that character is the shape of that curve — its curvature, its tangent vectors, its inflection points.

**Tangent vector** `γᵢ'(t)` = the character's current direction/momentum (what they're doing, where they're headed).

**Curvature** `κᵢ(t)` = how sharply the character's trajectory is bending. High curvature = a turning point. A betrayal, a revelation, a death — these are moments of extreme curvature.

**Torsion** = the trajectory twisting out of its current plane. This maps to genre-level shifts: a political drama suddenly becoming a survival story, or a romance turning into a tragedy.

### Layer 3: The Narrative Surface (Your Topological Map)

Here's where it gets genuinely novel. Rather than visualizing character arcs as independent curves, we construct a **surface** that captures the narrative structure of the entire world.

**The Narrative Manifold:** Define a 2D manifold `M` embedded in 3D where:
- **x-axis** = time (or a more abstract notion of narrative progression)
- **y-axis** = "character/perspective space" (which character's story we're following)
- **z-axis (height)** = **narrative tension** (a scalar field computed from the state)

The surface undulates — peaks are moments of high tension (battles, reveals, crises), valleys are periods of calm. Character arcs are curves *on this surface*, like contour-following paths on a terrain map.

**Tension as a scalar field:** Tension `T(x, y)` at a point can be computed from:
- Goal frustration: `‖desired_state - current_state‖` for the focal character
- Proximity to danger or conflict
- Information asymmetry (what the character doesn't know but the reader does)
- Rate of change of relationships
- Entropy of the character's situation (how many possible outcomes exist)

This gives you a function `T: M → ℝ` that you can visualize as a height map.

### Layer 4: The Vector Field (Story Momentum)

On this surface, define a **vector field** `V: M → TM` (a vector at every point on the manifold) that represents **narrative momentum**:

```
V(x, y) = (dx/dt, dy/dt)
```

Where:
- The x-component is the forward drive of time/plot
- The y-component is the "pull" toward or away from other characters

**Flow lines** of this vector field are the story arcs themselves. This is exactly the Calc 3 concept of a flow field — if you drop a particle at a point, the vector field carries it along a path. That path is a story.

**Key properties of the vector field:**
- **Divergence** `∇·V` > 0 at points where storylines scatter (the fellowship breaking). `∇·V` < 0 where storylines converge (characters gathering for a final battle).
- **Curl** `∇×V` ≠ 0 where there's cyclical narrative structure — repeating patterns, spiraling tension, characters orbiting each other without resolution.
- **Fixed points** where `V = 0`: these are narrative equilibria — stable situations that resist change (a peaceful kingdom) or unstable equilibria (a powder keg waiting to explode). The unstable ones are the most narratively interesting.

### Layer 5: Topological Features — The Shape of the Story

This is where algebraic topology gives you tools to talk about the *structure* of a narrative in ways that transcend any individual plot point.

**Homology groups** detect "holes" in the narrative:
- **H₀** counts connected components — how many disconnected storylines exist. In early ASOIAF, H₀ is large (Dany in Essos is disconnected from everything in Westeros). Over time, H₀ should decrease as storylines connect.
- **H₁** detects loops — cyclical narrative structures, recurring themes, characters whose arcs form closed paths. "The wheel of time" is literally a narrative with nontrivial H₁.
- **H₂** detects enclosed volumes — rare in narrative, but could represent self-contained sub-worlds (a dream sequence, a flashback arc that's fully encapsulated).

**Persistent homology** (from topological data analysis) is particularly powerful here. As you vary a threshold parameter (like "how closely must characters interact to be considered part of the same storyline?"), topological features appear and disappear. Features that persist across many scales are the *important* structural elements of your narrative.

**Morse theory** connects the topology to the scalar field (tension). Critical points of the tension function — maxima, minima, saddle points — correspond to topological changes in the narrative. A saddle point is where two storylines merge or split. The index of each critical point tells you what kind of narrative transition is happening.

### Layer 6: Information Theory — Measuring "Meaning"

This is the bridge from math to *meaning*. Shannon entropy and its variants give you a way to quantify how "interesting" or "meaningful" a narrative path is.

**Surprise:** The information content of an event is `I(e) = -log₂(P(e))`. Low-probability events carry more information. A character dying in a world where death is rare is more surprising (more meaningful) than in a grimdark world.

**Narrative entropy:** At any point, the entropy `H = -Σ pᵢ log₂(pᵢ)` over possible next states measures uncertainty. High entropy = the reader doesn't know what will happen = tension. A good story oscillates between high and low entropy — building tension, then resolving it, then building again.

**Mutual information** between character arcs: `I(γᵢ; γⱼ)` measures how much knowing one character's trajectory tells you about another's. High mutual information = deeply intertwined arcs. Low = independent stories that happen to share a world.

**KL-divergence** from genre expectations: You can define a "prior" distribution (what a reader expects based on genre conventions) and measure how much the actual narrative diverges from it. Too little divergence = predictable. Too much = incoherent. The sweet spot is where stories feel both surprising and inevitable.

---

## Architecture: How This Becomes Software

### The Simulation Engine (Backend)

```
WorldState → [Agent Decision Engine] → WorldState'
                    ↓
            [Event Log / History]
                    ↓
            [Metric Computation]
                    ↓
        [Narrative Manifold Construction]
```

**Tech stack candidates:**
- **Simulation core:** Rust or C++ (you have the chops). Agent-based modeling with an ECS-like architecture. Each agent has a state, a decision function, and perception of the world.
- **Decision engine:** Could start rule-based (if-then), evolve to utility-based (agents maximize expected utility), and eventually incorporate LLM-based reasoning for more naturalistic behavior.
- **Event log:** Append-only log of (time, agent, action, consequences). This is your raw data.
- **Metric computation:** Compute tension, entropy, curvature, information content from the event log. This can be a Python/NumPy pipeline.

### The Manifold Constructor (Math Engine)

Takes the event log and computed metrics and constructs the visual manifold:

1. **Dimensionality reduction:** The raw state space is high-dimensional. Use UMAP, t-SNE, or (better) diffusion maps to project onto a 2D manifold while preserving local structure. Diffusion maps are particularly good because they respect the *dynamics* — nearby points in the embedding are points the system can transition between easily.

2. **Tension surface:** Interpolate the scalar tension field onto the reduced manifold using Gaussian process regression or radial basis functions.

3. **Vector field computation:** Compute the narrative momentum field from the projected dynamics. Smooth with Helmholtz decomposition (separate into irrotational + solenoidal components) for cleaner visualization.

4. **Topological feature extraction:** Use persistent homology (libraries: Ripser, GUDHI, giotto-tda) to identify the structural features.

### The Visualization (Frontend — Your MVP)

An interactive 3D topological map. This is the product surface.

**Core interaction:**
- A terrain-like surface where height = tension, colored by some narrative property (which character is focal, what genre mode is active, etc.)
- **Curves** snaking across the surface = character arcs
- **Hover** over any point → see a tooltip with the world state, key events, active characters
- **Click a curve** → it lights up, showing the full arc of that character. Related curves (characters they interact with) glow dimly.
- **Drag to select a region** → extract a "story window" — a contiguous narrative with a beginning, rising action, climax, falling action

**Tech:**
- Three.js or React Three Fiber for 3D rendering
- WebGPU for compute shaders (surface generation, flow visualization)
- D3.js for 2D projections / fallback views
- Custom shaders for the "crystallizing" effect (the surface resolving from noise into structure as you zoom in)

### The Story Extractor (The AI Layer)

Once a user selects a path or region on the manifold, an LLM generates the actual prose:

- Input: The sequence of events along the selected path, character states, key interactions
- Output: A readable narrative, with the *shape* already determined by the topology (the LLM fills in the texture, dialogue, sensory detail — it doesn't decide what happens)

This is the key philosophical move: **the math determines structure, the AI fills in prose**. This prevents the common failure mode of AI-generated stories (meandering, no structure, no payoff) because the structure is mathematically guaranteed to have tension arcs, turning points, and convergence.

---

## Math You'll Need to Study

### Essential (start here)
1. **Multivariable Calculus (Calc 3 review):** Vector fields, divergence, curl, flow lines, surface integrals. You know this — just refresh the intuitions.
2. **Linear Algebra:** Eigenvalues/eigenvectors (for stability analysis of equilibria), SVD (for dimensionality reduction), matrix exponentials (for system dynamics).
3. **Probability & Information Theory:** Entropy, KL-divergence, mutual information, Bayesian updating. EECS 492 covered some of this.
4. **Graph Theory:** Character interaction graphs, centrality measures, community detection, spectral graph theory.

### Important (second pass)
5. **Dynamical Systems & ODEs:** Phase portraits, stability analysis, bifurcation theory (how small parameter changes cause qualitative shifts in behavior — perfect for "what if" scenario exploration).
6. **Differential Geometry:** Curves, surfaces, curvature, geodesics (shortest paths on the manifold = the most "economical" stories), Riemannian metrics.
7. **Topological Data Analysis (TDA):** Persistent homology, Betti numbers, simplicial complexes, filtrations. This is the field that directly enables your MVP visualization.

### Advanced (when you're ready to publish a paper)
8. **Sheaf Theory / Category Theory:** Sheaves formalize "local-to-global" consistency — perfect for ensuring narrative coherence (local character motivations must be globally consistent). Category theory gives you morphisms between narrative structures (how one story relates to another).
9. **Morse Theory:** The deep connection between critical points of scalar functions and the topology of manifolds. This is literally the math of "turning points change the shape of the story."
10. **Optimal Transport:** Wasserstein distances between narrative distributions. How "far apart" are two versions of the same story? This enables meaningful comparison and variation.

---

## Development Roadmap

### Phase 0: Foundations (Weeks 1-3)
- [ ] Refresh Calc 3, linear algebra, dynamical systems basics
- [ ] Read: "Computational Topology" by Edelsbrunner & Harer (Ch. 1-4)
- [ ] Read: "Topological Data Analysis" tutorials (giotto-tda docs are excellent)
- [ ] Implement a toy agent-based simulation (5 agents, simple rules, 2D grid world)
- [ ] Log events, compute basic metrics (encounter frequency, goal progress)

### Phase 1: Proof of Concept (Weeks 4-7)
- [ ] Build a richer simulation (10-20 agents, factions, resources, relationships)
- [ ] Implement tension/entropy computation pipeline
- [ ] Apply UMAP/diffusion maps to project state histories onto 2D
- [ ] Generate a static 3D surface visualization (Python + matplotlib first)
- [ ] Plot character trajectories on the surface
- [ ] Run persistent homology, identify structural features

### Phase 2: Interactive MVP (Weeks 8-14)
- [ ] Port visualization to Three.js / React Three Fiber
- [ ] Implement hover/click/select interactions
- [ ] Build the "crystallizing" rendering effect
- [ ] Add curve selection (click a character's path, see related arcs)
- [ ] Region selection → event sequence extraction
- [ ] Basic LLM integration for prose generation from selected paths

### Phase 3: Polish & Differentiate (Weeks 15-20)
- [ ] Vector field visualization (animated flow particles on the surface)
- [ ] Topological feature annotations (mark convergence points, turning points, loops)
- [ ] "What if" branching (modify a decision, re-simulate, see how the manifold changes)
- [ ] Story quality scoring (information-theoretic metrics for selected paths)
- [ ] User-facing worldbuilding interface (define agents, rules, constraints)

### Phase 4: Product (Weeks 20+)
- [ ] Multi-user collaboration on shared worlds
- [ ] Template worlds (fantasy, sci-fi, contemporary)
- [ ] Export to standard story formats
- [ ] API for programmatic world generation

---

## Why This Is Actually New

Existing AI story tools (NovelAI, Sudowrite, ChatGPT) work by **generating text forward** — they predict the next token. This means they have no global structure, no awareness of where the story is going, and no way to ensure payoff for setup.

Existing worldbuilding tools (World Anvil, Campfire, etc.) are **databases** — they organize information but don't simulate, analyze, or generate.

Existing computational narrative research (interactive fiction, story planners like DPOCL) works at the **symbolic planning level** — they find action sequences that achieve goals. But they don't produce the *feel* of a story, and they don't give you a visual, exploratory interface.

**NarrativeField combines:**
1. Agent-based simulation (from computational social science)
2. Topological data analysis (from pure math / data science)
3. Information-theoretic quality metrics (from information theory)
4. Interactive 3D visualization (from scientific visualization)
5. LLM prose generation (from modern AI)

Nobody is doing all five together. The mathematical framework — treating narrative as a dynamical system on a manifold with computable topological invariants — is, as far as I can tell, genuinely novel as an approach to *story generation tools*.

---

## Recommended Reading & Resources

**Books:**
- "Nonlinear Dynamics and Chaos" — Strogatz (dynamical systems intuition, beautifully written)
- "Computational Topology" — Edelsbrunner & Harer (TDA foundations)
- "The Art of Fiction" — John Gardner (what makes narrative *work* from a craft perspective)
- "Story" — Robert McKee (structural principles of narrative, useful for defining your metrics)
- "Gödel, Escher, Bach" — Hofstadter (the philosophy of meaning and formal systems)

**Papers:**
- "Topological Data Analysis" — Carlsson (2009, foundational survey)
- "Persistent Homology — a Survey" — Edelsbrunner & Harer
- "Diffusion Maps" — Coifman & Lafon (dimensionality reduction that respects dynamics)
- "Narrative Planning" — Mark Riedl's work at Georgia Tech (computational narrative)

**Tools:**
- giotto-tda (Python TDA library)
- Ripser (fast persistent homology)
- GUDHI (comprehensive computational topology)
- Three.js / React Three Fiber (3D web visualization)
- Mesa (Python agent-based modeling framework)

---

## The Elevator Pitch

*"What if you could see the shape of a story before you wrote it? NarrativeField simulates entire fictional worlds, then maps the resulting narratives onto an interactive topological surface — a terrain where peaks are moments of crisis, valleys are calm, and the paths snaking across it are the lives of characters. You explore the surface, find the stories worth telling, and extract them. It's worldbuilding meets mathematics meets AI: structure from simulation, meaning from topology, prose from language models."*
