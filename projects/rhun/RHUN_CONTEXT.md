# Rhun — Research Context Document

**Last updated:** February 13, 2026
**Purpose:** Complete context for AI assistants and collaborators working on this research program. Read this before doing anything.

---

## 1. What This Is

Rhun is a domain-agnostic research framework for studying **constrained extraction of structured subsequences from causal event graphs.** The core research question:

> Under what conditions does greedy extraction from causal DAGs with sequential phase constraints provably fail?

The system generates synthetic causal directed acyclic graphs (DAGs), extracts structured subsequences using greedy and oracle search strategies, validates them against a parameterized sequential grammar, and measures when and why extraction fails.

### Origin

Rhun was born from findings in a predecessor project called **Lorien/NarrativeField** (`~/lorien/`), a narrative generation system that simulates multi-agent worlds and extracts story arcs. During large-scale experimentation on that system (3,250+ deterministic runs), we discovered that:

1. A single sequential constraint (requiring ≥1 development-phase element before a turning point) accounts for 100% of the regularization effect in constrained extraction
2. The failure mode is purely temporal — it depends on when high-weight events occur in the timeline
3. The failure is invariant to how the importance function is decomposed across agents
4. The failure is invariant to anchor selection strategy
5. Relaxing constraints can make extraction *worse* (a non-monotonic "valley of death")

These findings are domain-agnostic properties of greedy search under sequential constraints, not narrative-specific phenomena. Rhun strips away the narrative scaffolding to study them as pure combinatorial optimization.

### Relationship to Lorien

- `~/lorien/` — the narrative AI system. Has a paper under submission to ICIDS (computational narratology conference). Still functional, still valuable as a realistic DAG generator and as the source of all empirical findings that motivated this research.
- `~/rhun/` — the research framework. Domain-agnostic. Designed to prove theorems, validate them on synthetic graphs, and generalize findings across domains.

Lorien is NOT a dependency of Rhun. Rhun has its own graph generators, extraction engine, and grammar system. The narrative system is one possible adapter (see `rhun/adapters/narrative.py`) but the core framework has no narrative-specific language or concepts.

---

## 2. The Core Theorem

### Prefix-Constraint Impossibility Theorem (Informal)

Let G be a temporally ordered causal graph with n events. Let w: Events → ℝ⁺ be a weight function. Let GREEDY be a search that selects the max-w event as the turning point and injects it into every candidate pool. Let GRAMMAR require k ≥ 1 development-phase elements before the turning point (the "prefix constraint").

**If the max-w event occurs at temporal position j (0-indexed) where j < k, then GREEDY produces zero valid sequences.**

The search enters an **absorbing state**: once the max-w event is classified as the turning point, the monotonic phase rule prevents any return to the development phase. If fewer than k events precede the turning point in the development phase, no sequence of future event additions can satisfy the prefix constraint. The search is deterministically trapped.

### Key Properties (Empirically Validated)

- **Shear invariance:** The failure is independent of how w is decomposed into per-actor components. Tested by computing per-agent importance and filtering injection sets by agent-relative scores. Result: zero focal failures fixed (r = 0.033 correlation between agent-relative shear and extraction failure).

- **Search invariance:** The failure is independent of anchor selection strategy. Tested by forcing all anchors to mid-timeline positions. Result: 0/9 failures fixed; byte-identical invalid arcs produced regardless of anchor strategy.

- **Parameter coupling invariance:** The only effective control parameter is continuous goal intensity (which shifts the temporal distribution of high-weight events). Categorical parameter coupling is vacuous for the tested agent. Tested via a 2×11 factorial decoupling sweep. Result: all three conditions produced numerically identical results.

### Current Status of the Theorem

The position sweep experiment (first experiment on synthetic graphs) shows:

| Metric | Value |
|---|---|
| Mean theorem accuracy | 0.918 across 21 epsilon values |
| False positive rate | **0.000** (theorem never predicts failure when extraction succeeds) |
| False negative rate | Increases with epsilon (0.05 at ε=0.60, rising to 0.35 at ε=0.95) |

**Interpretation:** The theorem captures one precisely characterized failure mode (absorbing state from premature TP assignment) with zero false alarms. At extreme front-loading (ε > 0.80), additional failure modes emerge that the theorem doesn't cover — likely insufficient focal actor coverage, timespan violations, or causal connectivity breakdown in severely front-loaded graphs. The theorem states a *sufficient* condition for failure, not a necessary one.

**Open question:** What are the additional failure modes at high ε? Characterizing these would extend the theorem from "one precisely understood trap" to "a complete taxonomy of greedy extraction failures under sequential constraints."

---

## 3. Architecture

```
~/rhun/
├── rhun/                              # Main package
│   ├── schemas.py                     # Core data structures (CausalGraph, Event, Phase, ExtractedSequence)
│   ├── generators/                    # DAG generators
│   │   ├── base.py                    # Abstract generator interface
│   │   ├── uniform.py                 # Uniform random DAGs (null model — no front-loading)
│   │   └── bursty.py                  # Temporal preferential attachment with front-loading parameter ε
│   ├── extraction/                    # Extraction engine
│   │   ├── grammar.py                 # Parameterized sequential phase grammar (GrammarConfig)
│   │   ├── phase_classifier.py        # Assigns phase labels based on weight/position
│   │   ├── pool_construction.py       # Pool building: BFS, injection, filtered injection
│   │   ├── search.py                  # Greedy search + oracle (brute-force) search
│   │   ├── scoring.py                 # Weight-sum + TP-weighted scoring
│   │   └── validator.py               # Grammar validation with violation reporting
│   ├── theory/                        # Theorem + automated verification
│   │   ├── theorem.py                 # check_precondition, verify_prediction, diagnose_absorption
│   │   └── counterexamples.py         # Boundary conditions
│   ├── experiments/                   # Experiment implementations
│   │   ├── runner.py                  # Common infrastructure
│   │   ├── position_sweep.py          # Sweep ε (front-loading) vs validity
│   │   ├── kj_boundary.py            # Joint sweep of k (prefix requirement) and j (max-weight position)
│   │   └── invariance.py             # Graph density, actor count, weight distribution sweeps
│   └── adapters/                      # Domain-specific adapters
│       ├── narrative.py               # Adapter for Lorien dinner party DAGs
│       └── incident.py                # Stub for incident response DAGs
├── tests/                             # 19 tests, all passing
├── experiments/                       # Experiment runner scripts + output/
│   ├── run_position_sweep.py
│   ├── run_kj_boundary.py
│   └── output/                        # JSON + markdown results
└── paper/                             # Workshop paper (stub)
```

### Key Interfaces

**Graph generation:**
```python
from rhun.generators.bursty import BurstyGenerator, BurstyConfig

config = BurstyConfig(n_events=200, n_actors=6, seed=42, epsilon=0.5)
graph = BurstyGenerator().generate(config)
```

**Extraction:**
```python
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.grammar import GrammarConfig

grammar = GrammarConfig(min_prefix_elements=1)  # k=1
result = greedy_extract(graph, focal_actor="actor_0", grammar=grammar)
print(result.valid, result.score, result.n_development)
```

**Theorem verification:**
```python
from rhun.theory.theorem import check_precondition, verify_prediction, diagnose_absorption

prediction = check_precondition(graph, "actor_0", grammar)
print(prediction["predicted_failure"])

verification = verify_prediction(graph, "actor_0", grammar, result)
print(verification["prediction_correct"])

absorption = diagnose_absorption(result, grammar)
print(absorption["absorbed"], absorption["absorption_step"])
```

### Running Experiments

```bash
cd ~/rhun
.venv/bin/python experiments/run_position_sweep.py
.venv/bin/python experiments/run_kj_boundary.py
```

All experiments are deterministic, reproducible from fixed seeds, and cost $0.00 (no LLM calls, no GPU required for current experiments).

---

## 4. Key Terminology

Use these terms precisely. They are chosen to be domain-agnostic and mathematically accurate.

| Term | Meaning | NOT this |
|---|---|---|
| **Event** | A node in the causal DAG with timestamp, weight, and actor attribution | Not "beat," "scene," or "incident" |
| **Phase** | One of: SETUP, DEVELOPMENT, TURNING_POINT, RESOLUTION | Not "act," "beat type" |
| **Turning point** | The max-weight event in an extracted sequence; the phase boundary | Not "climax," "crisis" |
| **Prefix constraint** | Grammar requirement for k ≥ 1 DEVELOPMENT events before TP | Not "development beat constraint" (that's the Lorien term) |
| **Front-loading (ε)** | Fraction of high-weight events in the first ε of the timeline | Not "burstiness" (too vague) |
| **Absorbing state** | A search state from which no valid completion exists | Not "dead end" (too informal) or "phase transition" (wrong mathematical framework) |
| **Pool contamination** | When injection heuristics force high-weight early events into every candidate pool | Not "proto_keep_ids" (that's the Lorien implementation detail) |
| **Focal actor** | The actor whose perspective the extraction targets | Not "protagonist" (narrative-specific) |

---

## 5. What Has Been Proven

### In Lorien (predecessor, narrative domain)

| Finding | Experiment | Status |
|---|---|---|
| One sequential constraint accounts for 100% of regularization | Feasible volume sweep (8 levels, 50 seeds) | Confirmed |
| The regularization cliff is search-algorithmic, not environment-dependent | Depth-0 vs depth-2 development beat sweep | Confirmed (32pp vs 36pp) |
| Candidate pools are entirely degenerate in failing cases | Search instrumentation (33/33 candidates have zero development) | Confirmed |
| Anchor selection doesn't matter | Anchor diversification (3 strategies, 0/9 fixed) | Confirmed |
| Pool injection is dual-function (necessary AND contaminating) | Proto-keep ablation (0/50 valid without it; temporal filter fixes 7/9) | Confirmed |
| Importance decomposition doesn't matter (shear invariance) | Shear-filtered injection (r=0.033, 0/9 fixed) | Confirmed |
| α=0.5 peak is pure continuous tuning (no categorical coupling) | Decoupled α sweep (3 conditions numerically identical) | Confirmed |
| The operative variable is temporal position of high-weight events | α sweep + Diana TP position data (0.51 → 0.716 at α=0.5) | Confirmed |

### In Rhun (current, synthetic domain)

| Finding | Experiment | Status |
|---|---|---|
| Validity decreases with front-loading (ε) | Position sweep (0.927 → 0.487) | Confirmed |
| Absorption rate increases with front-loading | Position sweep (0.073 → 0.440) | Confirmed |
| Theorem has zero false positives | Position sweep error analysis | Confirmed |
| Theorem has increasing false negatives at high ε | Position sweep error analysis (FN rate 0.05 → 0.35) | Confirmed |
| Additional failure modes exist beyond the theorem's scope | Position sweep at ε > 0.80 | Identified, not yet characterized |

### What Has Been Disproven

| Hypothesis | Test | Result |
|---|---|---|
| "Importance decomposition (shear) determines contamination" | Shear filter experiment | Dead (r=0.033, 0/9 fixed) |
| "α=0.5 peak is continuous-categorical coupling" | Decoupled α sweep | Dead (Thorne has no categorical commitments) |
| "The failure is a topological/geometric phenomenon" | Shear invariance + α coupling null | Dead for current formulation |
| "Relaxing constraints monotonically improves extraction" | Feasible volume sweep | Dead (valley of death at intermediate relaxation) |

---

## 6. What's Next

### Immediate (This Session / Next Session)

1. **Run the k-j boundary sweep.** Joint sweep of k (grammar.min_prefix_elements, 0–5) and j (temporal index of max-weight event). The theorem predicts a diagonal boundary: failure when j < k. This is the second core figure for the workshop paper.

2. **Characterize the false negative failure modes.** At ε ≥ 0.80, extractions fail for reasons outside the theorem. Diagnose these: are they coverage violations? Timespan violations? Causal connectivity breakdown? Each additional failure mode is a potential extension of the theorem.

3. **Evaluate whether the bursty generator's ε parameter needs recalibration.** The mid-range plateau (ε 0.15–0.55 all near 0.97) suggests ε isn't linearly controlling the max-weight position. Verify by plotting mean_max_weight_position vs. ε. If the relationship is non-linear, the generator may need adjustment or the experiment should sweep max-weight position directly rather than ε.

### Near-Term (Weeks 2–4)

4. **Invariance tests.** Sweep graph density (sparse → dense causal links), actor count (2 → 50), and weight distribution (uniform, power-law, bimodal). Each invariance confirmed is a corollary of the theorem.

5. **Beam search comparison.** Implement beam search as an alternative to greedy. Test whether beam width > 1 escapes the absorbing state trap. If so, characterize the minimum beam width needed as a function of ε and k. This is the constructive result: "greedy fails, but beam search of width w escapes."

6. **Write the theorem formally.** Clean combinatorial statement with proof. The absorbing-state characterization should be the proof mechanism: once the search enters the absorbing region of the state machine, the proof shows no valid completion exists.

### Medium-Term (Weeks 4–8)

7. **Workshop paper draft.** Target: NeurIPS or ICML 2026 workshop on constrained optimization, structured generation, or safe AI. Working structure:
   - §1: Problem — greedy extraction from causal DAGs with sequential phase constraints
   - §2: The prefix-constraint impossibility result (theorem + proof)
   - §3: Absorbing state characterization (the mechanism)
   - §4: Empirical validation — position sweep + k-j boundary on synthetic graphs
   - §5: Null results as invariance proofs (shear, coupling — from Lorien)
   - §6: Domain applications (narrative extraction as case study, incident response as future work)
   - §7: Constructive escape via beam search (if results are ready)

8. **Cross-domain validation.** Build the incident response DAG generator (synthetic distributed system outage traces). Demonstrate the same failure mode and the same theorem applicability in a non-narrative domain.

9. **GPU scaling (Great Lakes cluster).** Vectorize the extraction search for batch execution over thousands of synthetic graphs. The k-j boundary plot across 10,000+ instances with varying topology is the signature figure.

### Longer-Term (Months 3–6)

10. **Extended theorem.** Characterize the additional failure modes discovered at high ε. Aim for a complete taxonomy: "greedy extraction fails if and only if one of these N conditions holds."

11. **Formal paper (full conference).** Extend the workshop paper with the complete taxonomy, beam search constructive result, and cross-domain validation.

12. **Connections to optimization theory.** The non-monotonic regularization curve (valley of death) may connect to known results about constraint relaxation in non-convex optimization. The absorbing state characterization may connect to the theory of greedy algorithms on non-matroidal independence systems (this is an underexplored area in combinatorial optimization).

---

## 7. Empirical Methodology

All experiments in this project follow strict methodological principles:

1. **Verification-first:** Every experiment runs a baseline condition first and confirms it matches expected anchors before proceeding to experimental conditions. If baseline doesn't reproduce, the experiment stops.

2. **Fully deterministic:** All randomness is seeded. Same seed → same graph → same extraction → same result. No stochastic components.

3. **$0.00 compute cost:** No LLM calls, no API fees. All computation is local CPU. (GPU may be used for batch scaling in future, but the experiments themselves remain deterministic.)

4. **Null results are findings.** The shear experiment (r=0.033, 0/9 fixed) and the decoupled α sweep (identical arrays) are published as invariance proofs, not buried as failures.

5. **Error decomposition:** When a prediction fails, break it into TP/FP/FN/TN. Understand *which direction* the error goes. The position sweep's "zero false positives, increasing false negatives" is a much more informative statement than "accuracy drops to 0.65."

---

## 8. What NOT To Do

Patterns that have been tried and failed, or that lead to unproductive directions:

- **Do NOT use physics analogies (phase transitions, spontaneous symmetry breaking, cosmic censorship, geodesic incompleteness) without rigorous mathematical justification.** These were explored extensively and found to be analogies without mathematical content applicable to this system. The operative mathematics is combinatorial, not geometric or thermodynamic.

- **Do NOT assume the narrative domain is the interesting part.** The narrative system was the scaffolding that led to the discovery. The research contribution is the theorem about greedy extraction on DAGs. The narrative is one case study.

- **Do NOT train neural models to approximate the existing scoring function.** The scoring function has known biases (kinetic bias in the narrative domain). Training a neural net on it just learns the bias faster. Fix the metric first, then consider learned approaches.

- **Do NOT modify `~/lorien/` source code from this project.** Lorien is frozen as a validated instrument. If you need narrative DAGs, use the adapter in `rhun/adapters/narrative.py`.

- **Do NOT call the prefix constraint a "topological prior."** Topology has a precise mathematical meaning that doesn't apply here. It's a sequential structural constraint. Call it that.

- **Do NOT claim universal critical exponents or power-law scaling** unless you can demonstrate it rigorously across multiple system sizes with proper finite-size scaling analysis. The system is finite and discrete. A sigmoid fit is almost certainly more appropriate than a power law.

---

## 9. Open Research Questions

These are genuinely open — we don't know the answers. They're ordered roughly by tractability.

1. **What causes the false negatives at high ε?** The theorem predicts failure when max-weight is early, but at ε ≥ 0.80, 35% of failures happen for other reasons. What are they? Can they be formalized as additional sufficient conditions?

2. **Is the k-j boundary actually diagonal?** The theorem predicts it should be. The k-j boundary experiment will answer this directly. If it's not diagonal, what shape is it and why?

3. **Can beam search escape the absorbing state?** If greedy enters the absorbing state deterministically, does beam search with width w > 1 maintain at least one non-absorbed candidate? What's the minimum w as a function of graph properties?

4. **Does the non-monotonic regularization curve (valley of death) generalize beyond the narrative domain?** The Lorien feasible volume sweep showed that partial constraint relaxation makes extraction worse. Does this happen on synthetic graphs with different topologies? Under what conditions does relaxation help vs. hurt?

5. **Is there a natural family of sequential constraints beyond the prefix constraint?** The prefix constraint (k elements before TP) is one member of a family of sequential ordering requirements. What about suffix constraints (m elements after TP)? Balanced constraints (at least p% of elements in each phase)? Which of these create absorbing states and which don't?

6. **What is the right formal framework for "absorbing states in greedy search under non-hereditary constraints"?** This bridges combinatorial optimization and automata theory. The grammar is a finite state machine. The prefix constraint creates absorbing states. Is there a general theory of when sequential constraints create absorbing states in greedy search? This could be the deepest theoretical contribution.

7. **Does the incident response domain exhibit the same failure mode?** Root-cause analysis tools notoriously anchor on loud symptoms (the server crash) rather than quiet precursors (the config change 2 hours earlier). Is this literally the same prefix-constraint absorbing state, just in a different domain?

---

## 10. Multi-AI Workflow

This project uses multiple AI assistants:

- **Claude (Anthropic):** Primary analysis, experiment design, theoretical hygiene, codex prompt generation. Conservative about claims. Pushes back on overclaiming and physics analogies. Strong on error decomposition and precise mathematical framing.
- **Gemini (Google):** Strategic framing, creative hypothesis generation, enthusiasm. Generates many ideas, ~20% of which survive scrutiny. Good at identifying connections to other fields but inflates terminology. Must be calibrated against Claude's analysis.
- **Codex agents (Cursor/Claude Code):** Implementation. Receive detailed prompts with verification criteria, expected outputs, and commit messages.

**Calibration rule:** If an AI says "this shatters/revolutionizes X" or invokes physics (phase transitions, symmetry breaking, geodesics) without proving the mathematical structure actually applies — downgrade to "this is a clean empirical demonstration of a known phenomenon in a novel domain with precise mechanistic explanation." The findings are genuinely interesting. They don't need inflation.

---

## 11. Running the Codebase

```bash
# Setup
cd ~/rhun
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest -v  # 19 tests, all should pass

# Experiments
python experiments/run_position_sweep.py
python experiments/run_kj_boundary.py

# Quick smoke test
python -c "
from rhun.generators.bursty import BurstyGenerator, BurstyConfig
from rhun.extraction.search import greedy_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.theory.theorem import check_precondition, diagnose_absorption

g = BurstyGenerator().generate(BurstyConfig(seed=42, epsilon=0.8))
r = greedy_extract(g, 'actor_0', GrammarConfig(min_prefix_elements=1))
print(f'Valid: {r.valid}, Score: {r.score:.3f}, Dev: {r.n_development}')
p = check_precondition(g, 'actor_0', GrammarConfig(min_prefix_elements=1))
print(f'Predicted failure: {p[\"predicted_failure\"]}')
a = diagnose_absorption(r, GrammarConfig(min_prefix_elements=1))
print(f'Absorbed: {a[\"absorbed\"]}')
"
```
