# Rhun — Research Context Document (v3)

**Last updated:** February 14, 2026 (evening)
**Previous versions:** RHUN_CONTEXT.md (Feb 13), RHUN_CONTEXT_v2.md (Feb 14 morning)
**Purpose:** Complete context for AI assistants and collaborators working on this research. Read this before doing anything. This document is self-contained — you do not need to read v1 or v2 (though they're available for historical detail).

---

## 1. What This Project Is

Rhun is a domain-agnostic research framework for studying **constrained extraction of structured subsequences from causal event graphs.** The core research question:

> Under what conditions does greedy extraction from causal DAGs with sequential phase constraints provably fail, and what is the boundary of constructive algorithmic fixes?

The system generates synthetic causal directed acyclic graphs (DAGs), extracts structured subsequences using greedy, viability-aware, and oracle strategies, validates them against a parameterized sequential grammar, and measures when and why extraction fails.

### Origin

Rhun was born from findings in a predecessor project called **Lorien/NarrativeField** (`~/lorien/`), a narrative generation system that simulates multi-agent worlds and extracts story arcs. During large-scale experimentation (3,250+ deterministic runs), we discovered that a single sequential constraint — requiring ≥1 development-phase element before a turning point — accounts for 100% of the regularization effect. The failure mode is purely temporal, invariant to importance decomposition, anchor selection, and categorical parameter coupling. These are domain-agnostic properties of greedy search. Rhun strips away the narrative scaffolding to study them as pure combinatorial optimization.

**Lorien is NOT a dependency of Rhun.** The codebase is at `~/rhun/`. Lorien (`~/lorien/`) is frozen as a validated instrument. Rhun has its own generators, extraction engine, grammar system, and theory verification.

---

## 2. The Core Theorem

### Prefix-Constraint Impossibility Theorem

Let G be a temporally ordered causal graph with n events. Let w: Events → ℝ⁺ be a weight function. Let GREEDY select the max-w event as the turning point (TP) and inject it into every candidate pool. Let GRAMMAR require k ≥ 1 development-phase elements before the TP (the "prefix constraint").

**If the max-w event occurs at temporal position j (0-indexed) where j < k, GREEDY produces zero valid sequences.**

The search enters an **absorbing state**: the monotonic phase rule prevents return to the development phase after the TP is assigned. If fewer than k events precede the TP in the development phase, no future additions can satisfy the prefix constraint. The search is deterministically trapped.

### Empirical Status

**Zero false positives across all tested conditions.** The theorem never predicts failure when greedy succeeds.

The theorem states a *sufficient* condition for failure, not a necessary one. Additional failure modes exist (see §4) that the theorem doesn't cover — these show up as false negatives (greedy fails but the theorem didn't predict it).

### Theoretical Grounding

The feasible family under prefix constraints violates both the hereditary and exchange (augmentation) axioms of matroids and greedoids. Formally tested on small graphs (n_events=20, n_actors=3):

- **Hereditary axiom:** FAILS. Removing an element from a feasible set can destroy feasibility (removing a DEVELOPMENT event below the TP violates k ≥ 1). Counterexample rate: 100% of tested graphs.
- **Exchange/augmentation axiom:** FAILS. Larger feasible sets cannot always donate elements to smaller ones while maintaining feasibility.

This places the constraint system outside the Korte-Lovász greedoid framework, in the regime of general independence systems where greedy has no optimality guarantees. The absorbing state is the specific mechanism: it creates **TP coupling**, where modifying any element can trigger global phase reassignment, invalidating the entire sequence.

---

## 3. What Has Been Discovered (Complete Arc)

Research proceeded in three phases across Feb 13-14, 2026.

### Phase 1: Failure Taxonomy (Experiments 1-17, Feb 13-14 morning)

We ran 17 deterministic experiments that fully characterized why greedy extraction fails. Every false negative (greedy fails on a graph the oracle solves) was traced to a specific root cause. The taxonomy has four layers:

**Layer 1: Phase Classifier Coupling (63/138 FN cases, 45.7%)**
The phase classifier assigns `n_setup = ceil(0.2 × n_before_tp)`. When greedy has only 1 event before TP, this forces it to SETUP, leaving 0 DEVELOPMENT. Fix: grammar-aware classifier that preserves min_development slots (2-line change). All 63 cases recovered. Below-diagonal predictions unchanged.

**Layer 2: Turning Point Misselection (18/138 cases, 13.0%)**
Greedy picks a heavier-but-earlier TP than oracle. Beam search w=2 recovers all 18 cases. Beam search saturates completely at w=2 — widths 4, 8, 16 add zero recoveries.

**Layer 3: Sequence Assembly Compression (57/138 cases, 41.3%)**
Greedy selects temporally clustered high-weight events, compressing timespan below the grammar minimum. 98.6% of oracle events are in greedy's pool — the bottleneck is scoring, not search. Attempted fixes: beam search (0/57), 1-opt repair (0/57). This is a Resource-Constrained Shortest Path Problem (RCSPP) where weight and timespan are negatively correlated.

**Layer 4: Rare edge cases (~9% combined)**
Prefix dominance (6 cases), same-TP compression (4), selection artifacts (2), reclassification (1).

### Phase 2: Constructive Algorithms (Experiments 18-27, Feb 14 afternoon)

After characterizing failures, we built and evaluated constructive fixes. This arc is now **complete** — we've mapped the full capability envelope of forward viability filtering.

**Exact DP Oracle** (Experiment 18)
Implemented exhaustive search over all valid subsequences. Provides the true optimum for comparison. Key finding: greedy/exact ratio is bimodal (≥ 0.95 when valid, complete failure otherwise). The problem is feasibility-dominated.

**Greedoid Axiom Test** (Experiment 19)
Confirmed hereditary and exchange axioms both fail. Non-viable fraction rises sharply with sequence length. First non-viable step correlates with TP assignment.

**Layer 3 Anti-Correlation Diagnostic** (Experiment 20)
56/57 Layer 3 cases have top-m-by-weight span below threshold. Mean Pearson r between weight and timestamp in selected pool: -0.491. Weight and temporal spread are anti-correlated — this is the mechanism behind assembly compression.

**Swap Distance Verification** (Experiment 21)
54/57 Layer 3 cases need exactly 1 improving swap for span feasibility. The swap-in events are low-weight (mean rank 26.86) temporal endpoints. Confirms the "weight sacrifice for span" tradeoff is minimal but greedy can't find it.

**VAG Evaluation** (Experiment 22)
Viability-Aware Greedy (VAG) with span-only filtering: recovers 55/57 Layer 3 cases and all 18 Layer 2 cases. Zero regressions. Mean approximation ratio 1.0055. VAG is the primary constructive fix for span constraints.

**Gap Adversarial on Bursty** (Experiment 23)
Added `max_temporal_gap` constraint to grammar. Calibrated G = 0.1402 (p60 of greedy max-gaps). Results with gap constraint: greedy 57.3%, span-VAG 59.3%, gap-aware VAG 78.0%, exact oracle 100%. Gap-aware VAG uses depth-1 bridge existence checks.

**Multi-Burst Generator** (Experiment 24)
New generator with two temporal bursts and a sparse valley (r ≈ 0.014 weight-timestamp correlation, vs r ≈ -0.5 for bursty). Designed to stress-test VAG. Results: greedy 94%, span-only VAG 100%, exact oracle 100%. Three failures are Layer 1/2 hybrids (classifier + TP position in burst1).

**Multi-Burst Diagnostic** (Experiment 25)
Confirmed bimodal quality (mean ratio 0.9905, min 0.9512, zero below 0.90). Max-weight events split 42/58 between burst1/burst2. Zero inter-burst max-weight events.

**Multi-Burst + Gap Adversarial** (Experiment 26)
The critical experiment. Calibrated G = 0.4000 (clamped from raw p60 of 0.4434). Results:

| Algorithm | No Gap | With Gap | Delta |
|---|---:|---:|---:|
| Greedy | 94.0% | 8.0% | -86.0% |
| Span-only VAG | 100.0% | **0.0%** | -100.0% |
| Gap-aware VAG | 100.0% | 60.0% | -40.0% |
| Exact oracle | 100.0% | 100.0% | 0.0% |

**Span-only VAG at 0%** is the key finding: viability filtering for global constraints (span) is *adversarial* for local constraints (gap) because it selects temporally spread events that skip the valley. This is "adversarial viability" — optimizing for one constraint worsens another.

**Resonance catastrophe:** G ≈ valley_width creates a regime where the gap constraint is maximally binding. Every sequence crossing the valley needs ≥1 bridge event, but bridge events are sparse, low-weight valley residents that greedy ignores.

**BVAG Evaluation** (Experiment 27)
Budget-Reservation VAG: added bridge-budget lower bound B_lb(S) = Σ max(0, ceil(gap/G) - 1). At each step, verify B_lb ≤ remaining_slots. Results:

| Topology | Gap-aware VAG | BVAG | Recovered |
|---|---:|---:|---:|
| Bursty (G=0.14, 150 seeds) | 78.0% | 80.0% | 3/33 |
| Multi-burst (G=0.40, 50 seeds) | 60.0% | 60.0% | 0/20 |

**Null result on multi-burst.** The budget filter never activates. Verified empirically: all 20 failures have B_lb = 1, and the gap-bridge check (which counts gaps, not total bridges) already handles single-bridge cases identically. BVAG is algebraically redundant when max gap ≤ 2G.

The 3 bursty recoveries occur where a single gap needs 2+ bridges (gap > 2G), so B_lb = 2 blocks the greedy's last core event at step 18, forcing bridge selection. First budget block at step 18, B_lb = 2, remaining_slots = 2.

**Bridge existence verified:** 19/20 multi-burst failures have bridge events in the pool. The bridges exist but can't be reached by greedy's selection order. This is the commitment-timing problem: the algorithm needs to select bridge events at steps 5-10, but no single-step lookahead can anticipate this need.

### Phase 3: k-j Boundary Sweep (Experiment 28, Feb 14 evening)

**This experiment revealed a measurement mismatch, now resolved by the companion analysis (Exp 29).**

Sweep: k = 0-5, ε = 0.05-0.95 (19 values), 100 seeds per ε. All continuous constraints disabled (span = 0.0, gap = inf) to isolate the pure prefix-constraint absorbing state.

Results:
- Oracle feasible: 1900/1900 (100%)
- k=0: 100% valid, theorem accuracy 1.000

**Two j definitions were compared:**

| Definition | What it counts | Accuracy | FP | FN |
|---|---|---:|---:|---:|
| j_focal | Focal-actor events before max-weight focal event | 0.958 | 51 | 425 |
| j_theorem | All events (any actor) before max-weight focal event in full graph | 0.937 | 8 | 714 |

**j_focal wins on accuracy.** j_theorem counts non-focal events (mean |j_focal - j_theorem| = 34.3), making it systematically too large. This makes j_theorem < k rarely satisfied → fewer predicted failures → fewer FPs but far more FNs. The theorem becomes vacuously correct on the failure side but misses real failures.

**Below-diagonal behavior (j < k): Near-zero validity.** The theorem's core prediction holds. At k≥2, all below-diagonal cells show 0.0% greedy validity with j_theorem. The single anomaly is k=1, j_theorem=0 at 9.6% (7/73 instances succeed — the 8 FPs across all k values). These 8 FPs need diagnosis.

**Above-diagonal behavior (j ≥ k): Substantial degradation, worsening with k.** This is NOT a theorem failure — it's Layer 1 (classifier coupling) contaminating the signal. The unfixed classifier steals DEVELOPMENT slots via `ceil(0.2 × n_before_tp)`, which gets worse as k increases. Examples: k=3 j=3 at 64.9%, k=4 j=4 at 46.2%. These would recover with the grammar-aware classifier fix.

**Action needed for paper figure:** Re-run with (a) j_focal as binning variable and (b) grammar-aware classifier fix active. This should produce: near-0% below diagonal, near-100% above diagonal, clean sharp boundary.

---

## 4. The Viability Filtering Hierarchy (Complete)

This is the main constructive contribution. Forward viability filtering has a hierarchy of capability with a hard ceiling:

| Level | Algorithm | Constraint Class | Recovery Rate | Complexity |
|---|---|---|---:|---|
| 0 | Myopic greedy | None | Baseline | O(n²) |
| 1 | Span-only VAG | Global metric (timespan) | 55/57 (96.5%) | O(n²) |
| 2 | Gap-aware VAG | Local metric, depth-1 | 78% bursty, 60% multi-burst | O(n² × pool) |
| 2.5 | BVAG | Local metric + budget | 80% bursty, 60% multi-burst | O(n² × q) |
| ∞ | Exact DP oracle | All | 100% | O(n² × S) |

**The gap between Level 2.5 and ∞ is the fundamental boundary.** Forward viability filtering (any single-step lookahead) cannot solve cases requiring early commitment to bridge events. The remaining failures need global optimization (DP).

Key insight: the hierarchy has **sharp transitions**. VAG doesn't partially solve Layer 3 — it solves 55/57 completely. Gap-aware VAG doesn't gradually improve gap cases — it either bridges the gap or it can't. BVAG adds 3 cases on bursty and 0 on multi-burst. There is no smooth degradation.

---

## 5. Failure Class Taxonomy (Updated)

Across all tested topologies and constraint configurations:

### Class A: Absorbing State (Theorem Domain)
j < k → greedy deterministically trapped. Zero false positives. Captured by Prefix-Constraint Impossibility Theorem. Fraction depends on ε and k.

### Class B: Pipeline Coupling
Phase classifier assigns too many events to SETUP, leaving zero DEVELOPMENT. Fixed by grammar-aware classifier. 63 cases in original bursty analysis.

### Class C: Commitment Timing (The Frontier)
Bridges exist in pool but greedy commits to high-weight core events before budget becomes critical. Two subtypes:

- **C1: Spatial Edge Fragmentation (2.2% of bursty gap failures).** Bridge events genuinely absent from pool. Unreachable in product graph. No algorithm fixes without pool changes.
- **C2: Dimensional Knapsack Exhaustion (97.8% bursty, 100% multi-burst).** Bridge events present but selected too late. Sequence-length dimension exhausted before spatial routing complete. BVAG addresses 9.1% of C2 on bursty, 0% on multi-burst. Remaining C2 requires DP.

### Class D: Assembly Compression (Layer 3 from v2)
Weight-maximizing assembly compresses timespan. Solved by span-only VAG (55/57). The 2 unsolved are edge cases requiring complete reassembly.

---

## 6. Quality When Greedy Succeeds

Sharp bimodal behavior across all topologies:

| Topology | Approx Ratio Mean | Approx Ratio Min | Cases < 0.90 |
|---|---:|---:|---:|
| Bursty (no gap) | 0.996 | 0.975 | 0 |
| Multi-burst (no gap) | 0.9905 | 0.9512 | 0 |
| Bursty (with gap) | 1.0073 (VAG/exact) | 0.8807 | 1 |
| Multi-burst (with gap) | 1.0541 (VAG/exact) | 1.0015 | 0 |

The problem is feasibility-dominated: surviving constraints is hard; any surviving sequence is near-optimal.

---

## 7. What Has Been Disproven

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
| Valley of death is universal | Bursty synthetic sweep | Dead (topology-dependent; requires modular graph structure) |

---

## 8. Experiments (Complete List)

All experiments in `~/rhun/experiments/` with outputs in `experiments/output/`.

### Phase 1: Failure Taxonomy (Feb 13-14 morning)

| # | Experiment | Script | Key Finding |
|---|---|---|---|
| 1 | Position sweep | run_position_sweep.py | Validity decreases with ε; theorem 0 FP, increasing FN |
| 2 | Oracle diff | run_oracle_diff.py | 138/138 FN solvable (100% search-tricked) |
| 3 | FN divergence | run_fn_divergence_analysis.py | 97.1% different TP; greedy median 0 dev events |
| 4 | TP misselection | run_tp_misselection_analysis.py | Greedy picks heavier (+13%) but earlier (5×) TP |
| 5 | Prefix dominance | run_prefix_dominance_test.py | Condition holds in only 6/138 (dead) |
| 6 | Extraction internals | run_extraction_internals_diagnosis.py | 63 classifier failures, 67 timespan failures |
| 7 | Population A violations | run_population_a_violations.py | 67/67 are insufficient_timespan |
| 8 | Phase classifier analysis | (document only) | ceil(0.2 × 1) = 1 deterministic interaction |
| 9 | k-j boundary (single ε) | run_kj_boundary.py | Clean diagonal at ε=0.50 |
| 10 | Multi-ε k-j boundary | run_kj_boundary_multi_epsilon.py | Diagonal holds at all ε; above-diagonal scales with k and ε |
| 11 | Classifier fix + re-eval | run_kj_boundary_multi_epsilon.py (fixed) | 63 cases recovered; below-diagonal unchanged |
| 12 | Beam search sweep | run_beam_search_sweep.py | 18/75 at w=2; saturates completely |
| 13 | k-j beam search | (in beam sweep) | Modest above-diagonal improvement at ε=0.90 |
| 14 | Pool bottleneck | run_pool_bottleneck_diagnosis.py | 98.6% oracle events in pool; not the bottleneck |
| 15 | Valley of death | run_valley_of_death.py | No valley; monotonic on all curves |
| 16 | Approximation ratio | run_approximation_ratio.py | Ratio ≥ 0.95 everywhere; bimodal behavior |
| 17 | 1-opt repair | run_repair_sweep.py | 0/57 recovered; local repair insufficient |

### Phase 2: Constructive Algorithms (Feb 14 afternoon)

| # | Experiment | Script | Key Finding |
|---|---|---|---|
| 18 | Exact DP oracle | (in exact_oracle.py) | True optimum baseline; bimodal quality confirmed |
| 19 | Greedoid axiom test | run_greedoid_axiom_test.py | Both axioms fail; non-viable fraction rises at TP |
| 20 | Layer 3 anti-correlation | run_layer3_anticorrelation.py | 56/57 top-m span below threshold; r = -0.491 |
| 21 | Swap distance | run_swap_distance.py | 54/57 need 1 swap; swap-ins are low-weight endpoints |
| 22 | VAG evaluation | run_vag_evaluation.py | 55/57 L3, 18/18 L2 recovered; 0 regressions |
| 23 | Gap adversarial (bursty) | run_gap_adversarial.py | Gap-aware VAG 78%; exact oracle 100% |
| 24 | Multi-burst generator | run_multiburst_diagnostic.py | r = 0.014; greedy 94%; span-VAG 100% |
| 25 | Multi-burst comparison | run_multiburst_comparison.py | Bimodal quality confirmed (mean 0.9905) |
| 26 | Multi-burst + gap | run_multiburst_gap.py | Span-VAG 0% (!); gap-aware VAG 60%; oracle 100% |
| 27 | BVAG evaluation | run_bvag_evaluation.py | +3/33 bursty, 0/20 multi-burst. Null result. |

### Phase 3: Boundary Mapping (Feb 14 evening)

| # | Experiment | Script | Key Finding |
|---|---|---|---|
| 28 | k-j boundary (formal) | run_kj_boundary.py | j-definition mismatch; 8 FP; needs re-run |
| 29 | k-j companion | run_kj_boundary_companion.py | j_focal outperforms j_theorem; above-diagonal = Layer 1 |

---

## 9. Architecture

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
│   ├── run_kj_boundary.py             # Formal k-j boundary (Exp 28)
│   ├── run_bvag_evaluation.py
│   ├── run_multiburst_gap.py
│   ├── verify_bvag.py                 # BVAG correctness verification
│   └── output/                        # JSON + markdown results
└── paper/                             # Workshop paper (stub)
```

### Key Interfaces

```python
# Graph generation
from rhun.generators.bursty import BurstyGenerator, BurstyConfig
from rhun.generators.multiburst import MultiBurstGenerator, MultiBurstConfig

graph = BurstyGenerator().generate(BurstyConfig(seed=42, epsilon=0.5))
graph = MultiBurstGenerator().generate(MultiBurstConfig(seed=42))

# Extraction
from rhun.extraction.search import greedy_extract
from rhun.extraction.exact_oracle import exact_oracle_extract
from rhun.extraction.viability_greedy import viability_aware_greedy_extract
from rhun.extraction.grammar import GrammarConfig

grammar = GrammarConfig(min_prefix_elements=1, min_timespan_fraction=0.3, max_temporal_gap=0.14)
result = greedy_extract(graph, "actor_0", grammar)
exact, meta = exact_oracle_extract(graph, "actor_0", grammar)
vag, vag_meta = viability_aware_greedy_extract(graph, "actor_0", grammar,
    gap_aware=True, budget_aware=True)

# Theorem verification
from rhun.theory.theorem import check_precondition, verify_prediction, diagnose_absorption
pred = check_precondition(graph, "actor_0", grammar)
```

### Running

```bash
cd ~/rhun
source .venv/bin/activate
pytest -v                              # 38 tests
python experiments/run_position_sweep.py
python experiments/run_kj_boundary.py   # ~13 min
python experiments/run_bvag_evaluation.py  # ~23 min
```

All experiments are deterministic, reproducible from fixed seeds, $0.00 compute cost.

---

## 10. Key Terminology

| Term | Meaning | NOT this |
|---|---|---|
| **Event** | Node in causal DAG with timestamp, weight, actor | Not "beat," "scene" |
| **Phase** | SETUP, DEVELOPMENT, TURNING_POINT, RESOLUTION | Not "act" |
| **Turning point (TP)** | Max-weight event in extracted sequence; phase boundary | Not "climax" |
| **Prefix constraint** | Grammar requires k ≥ 1 DEVELOPMENT events before TP | Not "development beat constraint" |
| **Front-loading (ε)** | Fraction of high-weight events in first ε of timeline | Not "burstiness" |
| **Absorbing state** | Search state from which no valid completion exists | Not "dead end" or "phase transition" |
| **Pool contamination** | Injection heuristics force high-weight early events into pools | Not "proto_keep_ids" |
| **Focal actor** | Actor whose perspective extraction targets | Not "protagonist" |
| **Bridge event** | Low-weight event placed to satisfy max_temporal_gap | — |
| **Resonance catastrophe** | G ≈ valley_width regime where gap is maximally binding | — |
| **Commitment timing** | Greedy must choose bridge events early but can't anticipate need | — |
| **TP coupling** | Modifying any element triggers global phase reassignment | — |
| **B_lb** | Lower bound on bridge events needed: Σ max(0, ceil(gap/G) - 1) | — |

---

## 11. The Paper

### Status: Two threads remain before writing.

1. **k-j boundary figure needs the j-definition companion analysis** (§3 above). Once j_theorem is extracted and the matrix recomputed, the sharp diagonal should appear. This is Core Figure #2.

2. **The constructive algorithm hierarchy is complete** and ready to write up as a section.

### Target

NeurIPS or ICML 2026 workshop on constrained optimization, structured generation, safe AI, or ML systems. 6-8 pages.

### Proposed Structure

- **§1 Introduction:** Greedy extraction from causal DAGs under sequential phase constraints.
- **§2 Formalism:** Graph model, grammar (DFA), greedy search, prefix constraint. Non-hereditary constraint space (greedoid connection).
- **§3 Prefix-Constraint Impossibility Result:** Theorem + proof sketch. k-j boundary figure. Zero false positives.
- **§4 Failure Taxonomy:** Classifier coupling (63 cases, fix). TP misselection (18, beam w=2). Assembly compression (57, RCSPP connection).
- **§5 Constructive Fixes:** VAG hierarchy (span → gap-aware → BVAG). Capability envelope. BVAG null result as boundary proof.
- **§6 Bimodal Quality:** Approximation ratio ≥ 0.95. Feasibility-dominated landscape.
- **§7 Discussion:** Valley of death null (topology-dependent). Forward search boundary. Connections to process mining, scheduling.

### Core Figures

1. Position sweep (validity vs ε)
2. k-j boundary heatmap (diagonal boundary, requires j-definition fix)
3. VAG hierarchy comparison (span-VAG, gap-aware VAG, BVAG, exact oracle)
4. Multi-burst + gap resonance catastrophe (span-VAG at 0%)

---

## 12. What's Next

### Immediate (Next Session)

1. **Diagnose the 8 j_theorem FPs.** Cases where j_theorem < k but greedy succeeds. Quick investigation — may reveal an edge case in `check_precondition` or a classifier interaction.

2. **Re-run k-j boundary with classifier fix + j_focal.** The grammar-aware classifier (already implemented in `phase_classifier.py`) should eliminate above-diagonal degradation. j_focal should give the sharp diagonal. This produces Core Figure #2.

3. **Clean k-j boundary figure.** Publication-quality heatmap panel (k rows, j columns, greedy success rate as color). Below-diagonal = theorem domain (~0%), above-diagonal = near-100% with fixed classifier.

### Near-Term (Weeks 2-4)

3. **Formal proof of the theorem.** Combinatorial statement with proof via absorbing-state characterization. Formalize the DFA, show no transition from post-TP to DEV, prove no valid completion exists.

4. **Invariance tests.** Sweep graph density (sparse → dense), actor count (2 → 50), weight distribution (uniform, power-law, bimodal). Each invariance confirmed is a corollary.

5. **Cross-domain validation.** Build `adapters/incident.py` — synthetic distributed system outage traces. Demonstrate same failure mode and theorem applicability in a non-narrative domain.

### Deferred (Parking Lot)

6. **2D (G × valley_width) phase transition sweep.** Maps the resonance catastrophe boundary. 12,480 instances. Park until after paper draft.

7. **Valley of death on modular graphs.** Build a stochastic block model generator. The valley appeared in Lorien (modular narrative DAGs) but not bursty synthetics. What topological motif triggers it?

8. **Depth-d bridge viability.** Current gap-aware VAG checks depth-1 (does a bridge exist for each gap independently). Depth-d would check whether a consistent set of bridges exists for all gaps simultaneously. Complexity concern.

9. **Beam search for multi-burst gap.** Does beam width > 1 help with commitment timing? The single-bridge case might benefit from keeping an alternative trajectory alive.

---

## 13. Multi-AI Workflow

This project uses multiple AI assistants with calibrated roles:

- **Claude (Anthropic):** Primary analysis, experiment design, theoretical hygiene, codex prompt generation. Conservative about claims. Pushes back on overclaiming. Strong on error decomposition, code-level verification, experiment sequencing. Identified the BVAG algebraic redundancy mechanism, the j-definition mismatch in k-j boundary, and the control-flow ordering explanation for the multi-burst null.

- **Gemini (Google):** Strategic framing, creative hypothesis generation, literature connections. Best contributions: greedoid/matroid framing, RCSPP connection, gap adversarial experiment design, multi-burst generator concept. Weaknesses: inflates terminology ("this shatters X"), predicts outcomes with false confidence. Predicted BVAG would recover all 20 multi-burst cases (actual: 0). ~30-40% of ideas survive scrutiny.

- **GPT (OpenAI):** Practical implementation guidance, seed-level failure characterization. Correctly predicted multi-burst bimodal quality and identified the "switch to DP" insight for commitment-timing failures. Predicted BVAG would get "much closer to oracle" (actual: no movement). Useful for concrete code-level suggestions.

- **Codex agents (Cursor/Claude Code):** Implementation. Receive detailed prompts with verification criteria, expected outputs, and commit messages. Executed 28 experiments with zero implementation failures.

**Prediction scorecard (BVAG, the last round):**
| Predictor | Bursty prediction | Actual | Multi-burst prediction | Actual |
|---|---|---:|---|---:|
| Claude | 88-95% | 80.0% | 75-90% | 60.0% |
| Gemini | 31/33 exact | 3/33 | 20/20 exact | 0/20 |
| GPT | "much closer to oracle" | +3 only | "much closer to oracle" | +0 |

**Calibration rule:** If an AI says "this shatters/revolutionizes X" or invokes physics (phase transitions, symmetry breaking, geodesics) without proving the mathematical structure applies — downgrade to the precise empirical statement. The findings are genuinely interesting. They don't need inflation.

---

## 14. Methodology

1. **Verification-first:** Baseline conditions confirmed before experimental conditions.
2. **Fully deterministic:** All randomness seeded. Same seed → same result.
3. **$0.00 compute:** No LLM calls, no API fees. Local CPU only.
4. **Null results are findings.** BVAG null (0/20), repair null (0/57), valley of death (monotonic), prefix dominance (4.3%) — all published as structural results.
5. **Error decomposition:** TP/FP/FN/TN breakdown. Directional analysis, not aggregate accuracy.
6. **Prediction scoring:** All AI researchers make quantitative predictions before results. Scored honestly.

---

## 15. What NOT To Do

Everything from previous versions still applies, plus:

- **Do NOT assume the 57-case assembly gap is a pool construction problem.** Pool diagnosis proved 98.6% event coverage. The bottleneck is scoring/assembly.
- **Do NOT assume beam search at higher widths will help beyond w=2.** It saturates completely.
- **Do NOT assume local repair (1-opt, 2-opt) will fix timespan compression.** 0/57 recovered.
- **Do NOT conflate "oracle" with "global optimum."** oracle_extract is TP-exhaustive but heuristic in assembly. exact_oracle_extract is the true optimum.
- **Do NOT claim the valley of death is universal.** Topology-dependent (modular DAGs only).
- **Do NOT build deeper forward filters to solve the commitment-timing frontier.** BVAG proved this is bounded. The remaining cases need DP/global optimization.
- **Do NOT use j_theorem (full-graph index) for the k-j boundary figure.** j_theorem counts all events (any actor) before the max-weight focal event. With 200 events and 6 actors, j_theorem ≈ j_focal × 6. This makes j_theorem < k almost never true, producing 0.937 accuracy vs j_focal's 0.958. Use j_focal (focal-actor temporal rank) — it's closer to the actual absorbing state mechanism. The correct theoretical j is "development-eligible focal events before TP," which j_focal approximates.
- **Do NOT use physics analogies** (phase transitions, geodesics, symmetry breaking) without rigorous mathematical justification. The operative mathematics is combinatorial.
- **Do NOT train neural models to approximate the scoring function.** The scoring function has known biases. Fix the metric first.
- **Do NOT modify `~/lorien/` source code from this project.** Frozen as validated instrument.

---

## 16. Key Files

| File | Purpose |
|---|---|
| `rhun/extraction/viability_greedy.py` | VAG + gap-aware + BVAG (the constructive algorithm hierarchy) |
| `rhun/extraction/exact_oracle.py` | True optimal baseline |
| `rhun/extraction/grammar.py` | GrammarConfig: min_prefix_elements, min_timespan_fraction, max_temporal_gap |
| `rhun/extraction/phase_classifier.py` | Phase classification (grammar-aware fix applied) |
| `rhun/extraction/search.py` | greedy_extract, oracle_extract, beam_search_extract |
| `rhun/theory/theorem.py` | check_precondition, verify_prediction, diagnose_absorption |
| `rhun/generators/bursty.py` | BurstyGenerator with ε front-loading |
| `rhun/generators/multiburst.py` | MultiBurstGenerator with two bursts + sparse valley |
| `experiments/output/` | All experiment results as JSON + markdown summaries |
| `experiments/verify_bvag.py` | BVAG correctness verification (confirms redundancy) |
| `RHUN_CONTEXT.md` | Original context (project origin, Lorien relationship) |
| `RHUN_CONTEXT_v2.md` | Phase 1 failure taxonomy (experiments 1-17) |
| This file | Current complete state |
