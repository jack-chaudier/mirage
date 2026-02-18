# Rhun — Research Context Document (v4)

**Last updated:** February 15, 2026 (evening)
**Previous versions:** RHUN_CONTEXT.md (Feb 13), v2 (Feb 14 morning), v3 (Feb 14 evening)
**Purpose:** Complete context for AI assistants and collaborators working on this research. Read this before doing anything. This document is self-contained — you do not need to read prior versions.

---

## 1. What This Project Is

Rhun is a domain-agnostic research framework for studying **constrained extraction of structured subsequences from causal event graphs.** The core research question:

> Under what conditions does greedy extraction from causal DAGs with sequential phase constraints provably fail, and what is the complete constructive algorithm hierarchy that fixes it?

The system generates synthetic causal directed acyclic graphs (DAGs), extracts structured subsequences using greedy, viability-aware, DP-based, and oracle strategies, validates them against a parameterized sequential grammar, and measures when and why extraction fails.

### Origin

Rhun was born from findings in **Lorien/NarrativeField** (`~/lorien/`), a narrative generation system that simulates multi-agent worlds and extracts story arcs. During 3,250+ deterministic runs, we discovered that a single sequential constraint — requiring ≥1 development-phase element before a turning point — accounts for 100% of the regularization effect. The failure mode is purely temporal, invariant to importance decomposition, anchor selection, and categorical parameter coupling. These are domain-agnostic properties of greedy search. Rhun strips away the narrative scaffolding to study them as pure combinatorial optimization.

**Lorien is NOT a dependency of Rhun.** The codebase is at `~/rhun/`. Lorien (`~/lorien/`) is frozen as a validated instrument. Rhun has its own generators, extraction engine, grammar system, and theory verification.

---

## 2. The Core Theorem

### Prefix-Constraint Impossibility Theorem

Let G be a temporally ordered causal graph with n events. Let w: Events → ℝ⁺ be a weight function. Let GREEDY select the max-w focal-actor event e* as the turning point (TP) and inject it into every candidate pool. Let GRAMMAR require k ≥ 1 development-phase elements before the TP (the "prefix constraint"). Let j_dev be the number of development-eligible events (from ANY actor) with timestamps strictly before e* in the candidate pool.

**If j_dev < k, GREEDY produces zero valid sequences.**

The search enters an **absorbing state**: the monotonic phase rule prevents return to the development phase after the TP is assigned. If fewer than k events can be classified as DEVELOPMENT before the TP, no future additions can satisfy the prefix constraint. The search is deterministically trapped.

### Formal Proof (Delivered by GPT, verified)

The proof uses the product automaton P = EventGraph × GrammarDFA:
1. The injection mechanism forces greedy to select e* (max-weight focal event) as TP
2. Once e* is consumed as TURNING_POINT, no DFA transition to DEVELOPMENT exists (monotonic phase rule)
3. If j_dev < k, fewer than k events can be classified as DEVELOPMENT in the pre-TP portion
4. The post-TP DFA region is unreachable from states with insufficient DEVELOPMENT count
5. Conclusion: greedy produces zero valid sequences when j_dev < k

**The converse is NOT guaranteed:** j_dev ≥ k does not imply greedy succeeds (false negatives from Layers 2-4 exist).

### Empirical Status

**Zero false positives across all tested conditions.** 11,400 instances (k=0-5, ε=0.05-0.95, 100 seeds per ε), plus invariance across actor count (2-50), event density (50-1000), weight distributions (uniform, power-law, bimodal), and multi-burst topology. The theorem never predicts failure when greedy succeeds.

Key measurement discovery: j_dev counts development-eligible events from ALL actors before the TP, not just focal-actor events. Using focal-only (j_focal) produced 43 false positives where non-focal events filled DEVELOPMENT slots. Using j_dev: 0 FPs at k≥2, 8 FPs at k=1 (all timestamp-tie artifacts on a single seed). Below-diagonal validity for k≥3 is exactly 0.000 across every cell in the k-j boundary heatmap.

### Theoretical Grounding

The feasible family under prefix constraints violates both the hereditary and exchange axioms of matroids and greedoids (confirmed on 40/40 tested graphs, n_events=20, n_actors=3). This places the system outside the Korte-Lovász greedoid framework, in the regime of general independence systems where greedy has no optimality guarantees. The absorbing state is the specific mechanism: **TP coupling**, where modifying any element triggers global phase reassignment, invalidating the entire sequence.

---

## 3. The Complete Algorithm Hierarchy

This is the main constructive contribution. Six levels of algorithm, each addressing a specific failure mode, forming a **partial order** (not a total order — span-VAG and gap-VAG branch).

| Level | Algorithm | Search Type | TP Selection | Candidate Pool | Multi-burst+gap | Bursty+gap | Complexity | Median Runtime |
|---|---|---|---|---|---:|---:|---|---:|
| 0 | Myopic greedy | Forward, myopic | Endogenous (max-w) | BFS + injection | 8% | 54.7% | O(n²) | 1.3ms |
| 1 | Span-VAG | Forward, 1-step span | Endogenous | BFS + injection | **0%** (adversarial!) | 62.7% | O(n²) | 9ms |
| 2 | Gap-aware VAG | Forward, depth-1 bridge | Endogenous | BFS + injection | 60% | 81.3% | O(n² × pool) | 10ms |
| 2.5 | BVAG | Forward, global budget | Endogenous | BFS + injection | 60% | 82.0% | O(n² × q) | 10ms |
| 3 | TP-solver (M=25) | Multi-pass DP | Outer loop (top-M) | BFS + injection | 94% | 96.0% | O(M×L×n²×d) | 2.3s / 5.7s |
| ∞ | Exact oracle | Exhaustive DP | All focal events | All events | 100% | 99.3% | O(n²×S) | 4.4s / 3.1s |

### Key Transitions

**Greedy → Span-VAG:** Adds "can I still reach minimum timespan?" Fixes assembly compression (Layer 3, 55/57 cases). Cost: ~8× slower.

**Span-VAG → Gap-VAG:** Adds "does a bridge exist for each gap?" This is a **branch**, not a chain — span-VAG is adversarial on multi-burst+gap (0%!) because span viability pushes toward temporally spread events that skip the valley. Gap-VAG drops span awareness and adds gap awareness. Cost: ~1.5× slower.

**Gap-VAG → BVAG:** Adds "is the total bridge budget feasible?" Marginal improvement (3 cases on bursty, 0 on multi-burst). BVAG is algebraically redundant when all gaps need only 1 bridge (max gap ≤ 2G). Cost: ~1×.

**BVAG → TP-Solver:** The fundamental architectural change. Factors out endogenous TP coupling into an outer loop. Converts the non-hereditary, non-Markov search into a clean RCSPP with fixed phase labels. Replaces forward selection with label-setting DP maintaining all non-dominated partial sequences simultaneously. Cost: ~150-400× slower (still under 6 seconds).

**TP-Solver → Exact Oracle:** Expands search space (all events vs BFS pool, all TPs vs top-M). The remaining gap is: (a) M too small — oracle's optimal TP not in top-M candidates (3 residual cases at M=25), (b) pool construction boundary — solver's BFS pool doesn't contain span-extending events at temporal extremes that oracle uses (5 residual cases).

### TP-Conditioned RCSPP Solver Details

**File:** `rhun/extraction/tp_conditioned_solver.py`

**Algorithm:**
1. **Outer loop 1:** Enumerate top-M focal-actor events by weight as TP candidates
2. **Outer loop 2:** For each TP τ, enumerate n_pre ∈ {0..L-1} (pre-TP event count)
3. **Phase assignment:** With τ and n_pre fixed, every event gets a deterministic phase (SETUP/DEVELOPMENT if before τ, TURNING_POINT if τ, RESOLUTION if after)
4. **Inner solve:** Label-setting DP. Labels = (score, slots_used, first_t). Bucket key = (last_event_id, dfa_state). Dominance: A dominates B if A.score ≥ B.score AND A.slots_used ≤ B.slots_used AND A.first_t ≤ B.first_t
5. **Optional:** Backward gap heuristic (min_hops_to_end) for admissible pruning
6. **Accept:** Best valid sequence across all (τ, n_pre)

**Empirical performance at M=25:**
- Labels generated: 536K median (multi-burst), 1.54M median (bursty)
- Frontier peaks: 1557 (multi-burst), 673 (bursty)
- Recovery: 17/20 multi-burst BVAG failures, 21/26 bursty BVAG failures
- Zero regressions

**Runtime warning:** At M=25, the solver is SLOWER than exact oracle on bursty (5.65s vs 3.05s median). GPT predicted break-even at M≈14 for bursty — confirmed empirically. **Adaptive M is required for practical deployment.** M=10 is faster than oracle on both topologies; M=25 exceeds oracle runtime on bursty.

---

## 4. Failure Class Taxonomy (Complete)

### Class A: Absorbing State (Theorem Domain)
j_dev < k → greedy deterministically trapped. Zero false positives. Captured by Prefix-Constraint Impossibility Theorem. Fraction depends on ε and k.

### Class B: Pipeline Coupling
Phase classifier assigns too many events to SETUP, leaving zero DEVELOPMENT. Fixed by grammar-aware classifier. 63 cases in original bursty analysis. All recovered.

### Class C: Commitment Timing (The Frontier)
Bridges exist in pool but greedy commits to high-weight core events before budget becomes critical. Two subtypes:
- **C1: Pool construction boundary.** Events oracle uses are not in the BFS-reachable pool. 5 bursty residual cases at M=25 — all fail with `insufficient_timespan`, not gap violations. The missing events are span-extending events at temporal extremes (very early or very late timestamps) that belong to non-focal actors with no causal path from the focal actor.
- **C2: Dimensional knapsack exhaustion.** Bridge events present but selected too late. Sequence-length dimension exhausted before spatial routing complete. **Solved by TP-conditioned solver** (17/20 multi-burst, 21/26 bursty recoveries).

### Class D: Assembly Compression
Weight-maximizing assembly compresses timespan. Solved by span-only VAG (55/57). The 2 unsolved are edge cases requiring complete reassembly.

---

## 5. Pool Construction Boundary (Newly Characterized)

The TP-conditioned solver's residual failures at M=25 were diagnosed via `experiments/run_tp_solver_pool_diagnosis.py`. All 8 cases (5 bursty, 3 multi-burst "M too small") have exact_sequence_in_tp_pool=False — the oracle uses events outside the solver's BFS pool.

**Key finding:** The missing events are NOT gap-bridging events in valleys. They are **span-extending events at temporal boundaries** — events at the far ends of the timeline (e.g., e0186 at t=0.97 or e0010 at t=0.02) from non-focal actors that the oracle uses to satisfy the minimum timespan constraint. The solver's best attempts all fail with `insufficient_timespan`.

**Everyone was wrong about the mechanism:**
- Grok predicted "causally disconnected bridge events" → Wrong (it's span, not gap)
- Gemini predicted "drop BFS, unrestricted fixes all" → Right prescription, wrong reason
- GPT predicted "could be pool OR transition, don't conflate" → Correctly cautious

**For the paper:** The pool boundary is a named, characterized result:
> The TP-conditioned solver is exact within the BFS-reachable candidate set. Residual failures arise from candidate set restriction, not algorithmic incompleteness. The choice of candidate set is a domain modeling decision: causal locality (BFS) for narrative/incident applications; unrestricted for pure temporal extraction.

---

## 6. Quality When Algorithms Succeed

Sharp bimodal behavior across all topologies and algorithms:

| Topology | Approx Ratio Mean | Approx Ratio Min | Cases < 0.90 |
|---|---:|---:|---:|
| Bursty (no gap) | 0.996 | 0.975 | 0 |
| Multi-burst (no gap) | 0.9905 | 0.9512 | 0 |
| Bursty (with gap) | 1.007 (VAG/exact) | 0.881 | 1 |
| Multi-burst (with gap) | 1.054 (VAG/exact) | 1.002 | 0 |

The problem is **feasibility-dominated**: surviving constraints is hard; any surviving sequence is near-optimal. This is characteristic of non-hereditary constraint systems where the feasible region is narrow.

---

## 7. Invariance Results

The theorem holds structurally across all tested parameter variations. Sweep at k=2, ε=0.1-0.9, 50 seeds per ε.

**Actor count (N=2,4,6,10,20,50):** 0 FPs everywhere. Mean j_dev_pool = 48.264 **constant** across all N. Disproves Gemini's density-suppression corollary — j_dev depends on total event count (fixed at 200), not actor count. Adding actors subdivides the same events, doesn't add new ones.

**Event density (n=50,100,200,500,1000):** 0 FPs everywhere. Mean j_dev_pool scales linearly: 10.6, 21.4, 48.3, 101.3, 201.0. Validity improves from 0.73 (n=50) to 0.93 (n=200+). The 7-8% failure floor at large n comes from non-absorbing-state failure modes (Layers 2-4).

**Weight distribution (uniform, power_law, bimodal):** 5 FPs total (0.67%). All are tp_match=False — weight replacement destroys native weight-timestamp correlation, causing greedy to select a different TP than j_dev_pool predicted. j_dev_output = 0 FPs everywhere. These are measurement artifacts of the weight replacement procedure, not theorem failures.

**Multi-burst topology:** 0 FPs. Theorem accuracy 0.948.

---

## 8. What Has Been Disproven

| Hypothesis | Test | Result |
|---|---|---|
| Shear (importance decomposition) determines contamination | Lorien shear filter | Dead (r=0.033, 0/9 fixed) |
| α=0.5 peak is continuous-categorical coupling | Lorien decoupled α sweep | Dead (identical arrays) |
| Failure is topological/geometric | Shear + α coupling null | Dead |
| Relaxing constraints monotonically improves extraction | Lorien feasible volume sweep | Dead on Lorien; monotonic on synthetic |
| Prefix dominance explains false negatives | Prefix dominance test | Dead (6/138 = 4.3%) |
| Beam search recovers assembly compression | Beam sweep w=2..16 | Dead (saturates at w=2, 0/57) |
| 1-opt repair fixes timespan compression | Repair sweep | Dead (0/57) |
| Causal depth is the operative variable | Oracle diff analysis | Dead (process properties, not graph properties) |
| Budget reservation solves gap constraint | BVAG evaluation | Dead on multi-burst (0/20), marginal on bursty (3/33) |
| Valley of death is universal | Bursty synthetic sweep | Dead (topology-dependent) |
| Absorbing state probability decays with actor count | Invariance actor sweep | Dead (j_dev=48.264 constant across N=2-50) |
| Pool mismatches are causally disconnected bridge events | Pool diagnosis | Dead (mechanism is span-extending events, not bridges) |
| Span-VAG monotonically improves over greedy | Multi-burst + gap | Dead (0% adversarial catastrophe) |

---

## 9. Adversarial Viability & Constraint Antagonism

**Span-VAG → 0% on multi-burst+gap** is a key structural finding. When optimizing for one constraint (span) actively worsens another (gap), this is **constraint antagonism under greedy composition**.

Formalized by Gemini: constraints C₁ and C₂ are k-antagonistic under policy π at state S if (a) a jointly feasible completion exists, but (b) the top-k candidates by π's objective that preserve C₁ viability all violate C₂ viability.

**TP-conditioning partially resolves antagonism.** Fixing the TP destination eliminates span as an online routing objective — it becomes a static acceptance check. The inner DP explores the joint constraint space globally rather than making myopic choices. Residual antagonism manifests as increased label count (1.5M bursty vs 536K multi-burst) rather than validity failures. Gemini overclaimed "α_TP = 0" — if antagonism were truly zero, the solver would hit 100%.

---

## 10. Experiments (Complete List, 32 total)

### Phase 1: Failure Taxonomy (Experiments 1-17, Feb 13-14 morning)

| # | Experiment | Key Finding |
|---|---|---|
| 1 | Position sweep | Validity decreases with ε; theorem 0 FP, increasing FN |
| 2 | Oracle diff | 138/138 FN solvable (100% search-tricked) |
| 3 | FN divergence | 97.1% different TP; greedy median 0 dev events |
| 4 | TP misselection | Greedy picks heavier (+13%) but earlier (5×) TP |
| 5 | Prefix dominance | Condition holds in only 6/138 (dead) |
| 6 | Extraction internals | 63 classifier failures, 67 timespan failures |
| 7 | Population A violations | 67/67 are insufficient_timespan |
| 8 | Phase classifier analysis | ceil(0.2 × 1) = 1 deterministic interaction |
| 9 | k-j boundary (single ε) | Clean diagonal at ε=0.50 |
| 10 | Multi-ε k-j boundary | Diagonal holds at all ε |
| 11 | Classifier fix + re-eval | 63 cases recovered; below-diagonal unchanged |
| 12 | Beam search sweep | 18/75 at w=2; saturates completely |
| 13 | k-j beam search | Modest above-diagonal improvement at ε=0.90 |
| 14 | Pool bottleneck | 98.6% oracle events in pool; not bottleneck |
| 15 | Valley of death | No valley; monotonic on all curves |
| 16 | Approximation ratio | Ratio ≥ 0.95 everywhere; bimodal |
| 17 | 1-opt repair | 0/57 recovered |

### Phase 2: Constructive Algorithms (Experiments 18-27, Feb 14 afternoon)

| # | Experiment | Key Finding |
|---|---|---|
| 18 | Exact DP oracle | True optimum; bimodal quality confirmed |
| 19 | Greedoid axiom test | Both axioms fail; non-viable fraction rises at TP |
| 20 | Layer 3 anti-correlation | r = -0.491 weight-timestamp in selected pool |
| 21 | Swap distance | 54/57 need exactly 1 swap |
| 22 | VAG evaluation | 55/57 L3, 18/18 L2 recovered; 0 regressions |
| 23 | Gap adversarial (bursty) | Gap-aware VAG 78%; oracle 100% |
| 24 | Multi-burst generator | r = 0.014; greedy 94%; span-VAG 100% |
| 25 | Multi-burst comparison | Bimodal quality (mean 0.9905) |
| 26 | Multi-burst + gap | Span-VAG 0% (!); gap-aware VAG 60%; oracle 100% |
| 27 | BVAG evaluation | +3/33 bursty, 0/20 multi-burst. Null result. |

### Phase 3: Boundary Mapping (Experiments 28-29, Feb 14 evening)

| # | Experiment | Key Finding |
|---|---|---|
| 28 | k-j boundary (formal) | j-definition mismatch; 8 FP; needs re-run |
| 29 | k-j companion | j_focal outperforms j_theorem |

### Phase 4: TP-Solver & Validation (Experiments 30-32, Feb 15)

| # | Experiment | Key Finding |
|---|---|---|
| 30 | FP8 + FP43 diagnosis | All FPs are measurement artifacts (timestamp ties + non-focal prefix filling). Theorem airtight. |
| 31 | Final k-j boundary | j_dev_output: 0 FPs. j_dev_pool: 0 FPs for k≥2. Razor-sharp diagonal. Core Figure #2 locked. |
| 32 | Invariance suite | 0 FPs across actor count, density, topology. 5 FPs in weight variants (TP identity shift under non-native weights). |
| 33 | Weight FP diagnosis | All 5 are tp_match=False (weight replacement destroys correlation). |
| 34 | TP-solver evaluation (M=10) | Multi-burst 80%, bursty 94%, 0 regressions. 10/20 + 18/26 recoveries. |
| 35 | TP-solver evaluation (M=25) | Multi-burst 94%, bursty 96%, 0 regressions. 17/20 + 21/26 recoveries. |
| 36 | Pool mismatch diagnosis | All 8 residuals are pool_construction_difference. Mechanism: span-extending events at temporal boundaries, not gap bridges. |

---

## 11. Architecture

```
~/rhun/
├── rhun/                              # Main package
│   ├── schemas.py                     # CausalGraph, Event, Phase, ExtractedSequence
│   ├── generators/
│   │   ├── base.py                    # Abstract generator interface
│   │   ├── uniform.py                 # Uniform random DAGs (null model)
│   │   ├── bursty.py                  # Temporal preferential attachment, ε parameter
│   │   └── multiburst.py             # Two-burst generator with sparse valley
│   ├── extraction/
│   │   ├── grammar.py                 # GrammarConfig: min_prefix_elements, min_timespan_fraction, max_temporal_gap
│   │   ├── phase_classifier.py        # Phase assignment (grammar-aware fix applied)
│   │   ├── pool_construction.py       # Pool building: BFS, injection, filtered injection
│   │   ├── search.py                  # greedy_extract, oracle_extract, beam_search_extract
│   │   ├── exact_oracle.py            # Exhaustive DP oracle (true optimum)
│   │   ├── viability_greedy.py        # VAG: span-only, gap-aware, budget-aware (BVAG)
│   │   ├── tp_conditioned_solver.py   # TP-conditioned RCSPP (the constructive breakthrough)
│   │   ├── scoring.py                 # Weight-sum + TP-weighted scoring
│   │   └── validator.py               # Grammar validation with violation reporting
│   ├── theory/
│   │   ├── theorem.py                 # check_precondition, verify_prediction, diagnose_absorption
│   │   └── counterexamples.py         # Boundary conditions
│   ├── experiments/
│   │   ├── runner.py                  # ExperimentMetadata, ExperimentTimer, save_results
│   │   ├── position_sweep.py
│   │   ├── kj_boundary.py
│   │   └── invariance.py
│   └── adapters/
│       ├── narrative.py               # Adapter for Lorien dinner party DAGs
│       └── incident.py                # Stub for incident response DAGs
├── tests/                             # 38 tests, all passing
├── experiments/                       # Runner scripts + output/
│   ├── run_position_sweep.py
│   ├── run_kj_boundary.py
│   ├── run_kj_boundary_final.py       # Core Figure #2 (j_dev, zero FP)
│   ├── run_bvag_evaluation.py
│   ├── run_multiburst_gap.py
│   ├── run_tp_solver_evaluation.py    # M=10 and M=25 configurations
│   ├── run_tp_solver_pool_diagnosis.py
│   ├── run_invariance_suite.py
│   ├── verify_bvag.py
│   └── output/                        # JSON + markdown results (36 experiments)
└── paper/                             # Workshop paper (stub)
```

### Key Interfaces

```python
# Graph generation
from rhun.generators.bursty import BurstyGenerator, BurstyConfig
from rhun.generators.multiburst import MultiBurstGenerator, MultiBurstConfig

graph = BurstyGenerator().generate(BurstyConfig(seed=42, epsilon=0.5))
graph = MultiBurstGenerator().generate(MultiBurstConfig(seed=42))

# Extraction (full algorithm suite)
from rhun.extraction.search import greedy_extract
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.extraction.tp_conditioned_solver import tp_conditioned_solve
from rhun.extraction.grammar import GrammarConfig

grammar = GrammarConfig(min_prefix_elements=1, min_timespan_fraction=0.3, max_temporal_gap=0.14)
result = greedy_extract(graph, "actor_0", grammar)
exact, meta = exact_oracle_extract(graph, "actor_0", grammar)
vag, vag_meta = viability_aware_greedy_extract(graph, "actor_0", grammar,
    gap_aware=True, budget_aware=True)
tp_result, tp_meta = tp_conditioned_solve(graph, "actor_0", grammar, M=25)

# Theorem verification
from rhun.theory.theorem import check_precondition, verify_prediction, diagnose_absorption
pred = check_precondition(graph, "actor_0", grammar)
```

### Running

```bash
cd ~/rhun
source .venv/bin/activate
pytest -v                                    # 38 tests
python experiments/run_position_sweep.py      # ~2 min
python experiments/run_kj_boundary_final.py   # ~13 min
python experiments/run_invariance_suite.py    # ~10 min
python experiments/run_tp_solver_evaluation.py --M 25  # ~38 min
```

All experiments are deterministic, reproducible from fixed seeds, $0.00 compute cost.

---

## 12. Key Terminology

| Term | Meaning | NOT this |
|---|---|---|
| **Event** | Node in causal DAG with timestamp, weight, actor | Not "beat," "scene" |
| **Phase** | SETUP, DEVELOPMENT, TURNING_POINT, RESOLUTION | Not "act" |
| **Turning point (TP)** | Max-weight focal event in extracted sequence; phase boundary | Not "climax" |
| **Prefix constraint** | Grammar requires k ≥ 1 DEVELOPMENT events before TP | Not "development beat constraint" |
| **Front-loading (ε)** | Fraction of high-weight events in first ε of timeline | Not "burstiness" |
| **Absorbing state** | Search state from which no valid completion exists | Not "dead end" or "phase transition" |
| **j_dev** | Development-eligible events (any actor) before TP | Not j_focal or j_theorem |
| **Focal actor** | Actor whose perspective extraction targets | Not "protagonist" |
| **Bridge event** | Low-weight event placed to satisfy max_temporal_gap | — |
| **Resonance catastrophe** | G ≈ valley_width regime where gap is maximally binding | — |
| **Commitment timing** | Greedy must choose bridge events early but can't anticipate need | — |
| **TP coupling** | Modifying any element triggers global phase reassignment | — |
| **B_lb** | Lower bound on bridge events needed: Σ max(0, ceil(gap/G) - 1) | — |
| **Adversarial viability** | Optimizing for one constraint worsens another | — |
| **Pool construction boundary** | Events oracle uses that aren't in BFS-reachable pool | — |

---

## 13. The Paper

### Status: Experimental phase near-complete. Pool boundary diagnosed. Paper draft can begin.

**Paper-ready components:**
- §1-2 (Introduction + Formalism): Complete. Theorem statement with j_dev, zero FPs verified.
- §3 (Impossibility Result): Complete. Formal proof via product automaton, k-j boundary heatmap (Core Figure #2), zero FPs for k≥2.
- §4 (Failure Taxonomy): Complete. Four layers fully characterized with fixes or impossibility proofs.
- §5 (Constructive Hierarchy): Complete pending pool boundary resolution. VAG → gap-aware → BVAG → TP-solver with recovery rates and empirical strict dominance.
- §6 (Bimodal Quality): Complete. Approximation ratio ≥0.95 when feasible, feasibility-dominated landscape.
- §7 (Discussion): Needs writing. Valley of death topology-dependent, forward search boundary at commitment-timing, connections to process mining/scheduling.

### Target
NeurIPS or ICML 2026 workshop on constrained optimization, structured generation, safe AI, or ML systems. 6-8 pages.

### Core Figures
1. Position sweep (validity vs ε) — done
2. k-j boundary heatmap (j_dev_pool, razor-sharp diagonal) — done
3. Algorithm hierarchy comparison table (greedy → TP-solver → oracle) — done
4. Multi-burst + gap resonance catastrophe (span-VAG at 0%) — done
5. TP-solver recovery analysis (M=10 vs M=25, residual diagnosis) — done

---

## 14. Multi-AI Workflow

This project uses multiple AI assistants with calibrated roles and honest prediction scoring.

### Roles

- **Claude (Anthropic):** Team leader. Primary analysis, experiment design, theoretical hygiene, codex prompt generation, researcher calibration. Conservative about claims. Pushes back on overclaiming. Identified: BVAG algebraic redundancy, j-definition mismatch, control-flow ordering for multi-burst null, pool boundary mechanism (span-extending, not gap-bridging). Writes all codex prompts with verification criteria.

- **GPT (OpenAI):** Heavy proofs, detailed specifications, careful complexity analysis. Best contributions: formal impossibility proof, TP-conditioned solver specification (directly implemented), h=1 lower bound lemma (depth-1 viability provably fails on adversarial instance), A* heuristic admissibility correction (max not sum), complexity analysis predicting M*≈14 break-even correctly. Most reliable predictions. Correctly cautioned "don't assume pool = disconnected bridges" — vindicated by diagnosis.

- **Gemini (Google):** Strategic framing, creative hypothesis generation. Best contributions: greedoid/matroid framing, RCSPP connection, gap adversarial experiment design, multi-burst generator concept, adversarial viability naming, backward DP heuristic idea, TP feasibility pre-filter proposal (validated by M=25 data). Weaknesses: inflates terminology, overclaims (α_TP=0, density-suppression corollary disproven), predicts with false confidence. ~30-40% of ideas survive scrutiny.

- **Grok (X.AI):** Verification, hierarchy formalization, tightness analysis. Best contributions: partial order observation for hierarchy theorem (span/gap branch), depth-d recovery theorem refinement, B_lb tightness analysis (near-exact for disjoint coverage). Restricted-pool oracle experiment design (correct but superseded by pool diagnosis).

- **Codex agents (Cursor/Claude Code):** Implementation. Receive detailed prompts with verification criteria, expected outputs, and commit messages. Executed 36 experiments with zero implementation failures. 38/38 tests passing.

### Prediction Scorecards

**BVAG Round (Experiment 27):**
| Predictor | Bursty | Actual | Multi-burst | Actual |
|---|---|---:|---|---:|
| Claude | 88-95% | 80.0% | 75-90% | 60.0% |
| Gemini | 31/33 exact | 3/33 | 20/20 exact | 0/20 |
| GPT | "much closer" | +3 only | "much closer" | +0 |

**TP-Solver Round (Experiment 34, M=10):**
| Predictor | Multi-burst | Actual | Bursty | Actual |
|---|---|---:|---|---:|
| Claude | 17-19/20 | 10/20 | 25-30/33 | 18/26 |
| Gemini | 20/20 | 10/20 | — | — |

**FP43 Diagnosis Prediction:**
| Predictor | Prediction | Actual |
|---|---|---|
| Claude | TP divergence or classifier | Non-focal prefix filling |
| Gemini | TP divergence | Wrong (tp_match=True 51/51) |
| Grok | TP divergence (pool dynamics) | Wrong |
| GPT | (not asked) | N/A |

**Pool Boundary Prediction:**
| Predictor | Prediction | Actual |
|---|---|---|
| GPT | "Could be pool OR transition, don't conflate" | Correct (all pool, zero transition) |
| Grok | "Causally disconnected bridge events" | Wrong sub-mechanism (span, not gap) |
| Gemini | "Drop BFS, unrestricted fixes all" | Right fix, wrong reason |

**Calibration rule:** If an AI says "this shatters/revolutionizes X" or invokes physics (phase transitions, symmetry breaking, geodesics) without proving the mathematical structure applies — downgrade to the precise empirical statement.

### Workflow Pattern

1. **Claude designs experiments** → writes codex prompts with verification criteria, expected outputs, commit messages
2. **Codex agents implement and run** → deterministic results, zero failures across 36 experiments
3. **Claude analyzes results** → routes findings to researchers with calibrated tasks
4. **Researchers (GPT/Gemini/Grok) produce theory/analysis** → Claude evaluates, cross-pollinates, scores predictions
5. **Repeat** with next experiment informed by all findings

Key insight: the multi-AI workflow works because each system has complementary strengths. GPT for proofs and specs, Gemini for creative constructions and literature connections, Grok for verification and formalization, Claude for experiment design and calibration. The prediction scorecard keeps everyone honest.

---

## 15. Methodology

1. **Verification-first:** Baseline conditions confirmed before experimental conditions.
2. **Fully deterministic:** All randomness seeded. Same seed → same result.
3. **$0.00 compute:** No LLM calls, no API fees. Local CPU only.
4. **Null results are findings.** BVAG null (0/20), repair null (0/57), valley of death (monotonic), prefix dominance (4.3%), density-suppression (constant j_dev) — all published as structural results.
5. **Error decomposition:** TP/FP/FN/TN breakdown. Directional analysis, not aggregate accuracy.
6. **Prediction scoring:** All AI researchers make quantitative predictions before results. Scored honestly. Nobody gets credit for post-hoc explanations.

---

## 16. What NOT To Do

Everything from previous versions still applies, plus:

- **Do NOT assume pool mismatches are disconnected bridge events.** Pool diagnosis proved the mechanism is span-extending events at temporal extremes, not gap bridges in valleys.
- **Do NOT use M=25 on bursty without adaptive M.** Solver exceeds oracle runtime (5.65s vs 3.05s). Use M=10 for practical deployment, M=25 only for coverage experiments.
- **Do NOT claim the hierarchy is a total order.** It's a partial order — span-VAG and gap-VAG branch (span-VAG is adversarial on multi-burst+gap).
- **Do NOT assume BVAG adds value on multi-burst.** BVAG is algebraically redundant when max gap ≤ 2G (0/20 recoveries, same as gap-VAG).
- **Do NOT use j_focal or j_theorem for the k-j boundary.** j_dev (development-eligible events from any actor) is the correct variable. j_focal produces 43 FPs. j_theorem produces 8 FPs but misses most real failures.
- **Do NOT assume Gemini's density-suppression corollary holds.** Actor count has zero effect on j_dev (flat at 48.264 for N=2-50). The suppression works through event density (n), not actor count (N).
- **Do NOT build deeper forward filters to solve commitment timing.** BVAG proved this is bounded. The remaining cases need TP-conditioned DP.
- **Do NOT use physics analogies** without rigorous mathematical justification.
- **Do NOT train neural models to approximate the scoring function.**
- **Do NOT modify `~/lorien/` source code from this project.**
- **Do NOT assume the 57-case assembly gap is a pool construction problem.** Pool diagnosis proved 98.6% event coverage. The bottleneck is scoring/assembly.
- **Do NOT assume beam search at higher widths will help beyond w=2.** Saturates completely.
- **Do NOT conflate "oracle" with "global optimum."** oracle_extract is TP-exhaustive but heuristic in assembly. exact_oracle_extract is the true optimum.

---

## 17. What's Next

### Immediate (Next Session)

1. **Decide on pool construction for the paper.** GPT raised the right question: is Rhun "extraction from a causal graph" (BFS pool) or "temporal subsequence selection" (unrestricted pool)? The grammar constraints are purely temporal — they don't reference causal edges. The BFS pool is a Lorien artifact. Options: (a) drop BFS for the solver, use unrestricted pool, close the 5 bursty residuals; (b) keep BFS, name the pool boundary as a characterized result. Either way, state the design decision explicitly.

2. **Implement Gemini's TP feasibility pre-filter.** Before ranking focal events by weight, filter out events where j_dev < k (absorbing state) or temporal_coverage < span_threshold (span infeasible). This reduces effective M without losing coverage — recovering M=25 results with M=5-10 runtime.

3. **Implement GPT's admissible A* heuristic.** h = max(min_hops_to_target, min_steps_to_accept_DFA). Prune labels where h > remaining_slots. GPT correctly identified that sum (not max) is inadmissible when one event can satisfy multiple constraints. Expected: significant label reduction → runtime improvement.

### Near-Term (Weeks 2-3)

4. **Paper draft.** All empirical results are in. Structure: impossibility theorem → failure taxonomy → VAG hierarchy → TP-conditioned solver → bimodal quality → pool boundary. Target: NeurIPS/ICML 2026 workshop.

5. **Formal proof of forward filtering ceiling.** GPT proved h=1 lower bound (depth-1 viability fails on adversarial instance). Open question: does the depth hierarchy collapse at d=2 (depth-2 sufficient for all temporal DAGs), or is it strict (depth-d < depth-(d+1) for all d)?

6. **TP-Decoupling Theorem.** If the TP is fixed exogenously (not max-weight of selected set), the feasible family becomes hereditary. Prove: exogenous TP → greedy achieves (1-1/e) approximation. Endogenous TP → greedy can achieve ratio 0. This dichotomy explains why TP-conditioning works.

### Deferred (Parking Lot)

7. **2D (G × valley_width) phase transition sweep.** Maps the resonance catastrophe boundary.
8. **Cross-domain validation.** Incident response adapter — synthetic distributed system outage traces.
9. **GPU scaling (Great Lakes cluster).** Vectorize extraction for batch execution over 10,000+ graphs.
10. **Connections to process mining.** van der Aalst alignment heuristics as TP-solver inner heuristic.
11. **Suffix/balanced constraint variants.** Does the absorbing state generalize beyond prefix constraints?

---

## 18. Key Results Files

| File | Content |
|---|---|
| `experiments/output/kj_boundary_final.json` | Core Figure #2 data (j_dev_pool, zero FP) |
| `experiments/output/kj_boundary_final_heatmap.png` | Publication heatmap |
| `experiments/output/invariance_suite.json` | Actor/density/weight/topology sweeps |
| `experiments/output/tp_solver_evaluation.json` | M=10 results |
| `experiments/output/tp_solver_evaluation_m25.json` | M=25 results |
| `experiments/output/tp_solver_evaluation_m25_summary.md` | Recovery analysis + timing |
| `experiments/output/tp_solver_pool_diagnosis.md` | 8-case pool boundary diagnosis |
| `experiments/output/fp43_diagnosis.json` | Non-tie FP diagnosis (non-focal prefix) |
| `RHUN_CONTEXT.md` | Original context (project origin) |
| `RHUN_CONTEXT_v2.md` | Phase 1 taxonomy |
| `RHUN_CONTEXT_v3.md` | Through Experiment 29 |
| This file | Current complete state |
