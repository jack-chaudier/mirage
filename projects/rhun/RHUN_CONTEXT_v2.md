# Rhun — Research Context Document (v2)

**Last updated:** February 14, 2026
**Previous version:** RHUN_CONTEXT.md (February 13, 2026)
**Purpose:** Complete context for AI assistants and collaborators. Read this before doing anything. The original RHUN_CONTEXT.md covers the project's origin, architecture, and codebase — read that first if you haven't. This document covers what has been discovered since and where the research stands now.

---

## 1. What Happened

On the evening of February 13–14, 2026, we ran a sequence of 12 deterministic experiments that fully characterized why greedy extraction fails on causal DAGs under sequential phase constraints. Every experiment is reproducible from fixed seeds at $0.00 compute cost. All code is in `~/rhun/`.

The research question was:

> Under what conditions does greedy extraction from causal DAGs with sequential phase constraints provably fail?

We now have a complete answer for the tested system.

---

## 2. The Core Theorem (Confirmed)

### Prefix-Constraint Impossibility Theorem

If the max-weight event among the focal actor's events occurs at temporal position j < k (where k = grammar.min_prefix_elements), greedy extraction produces zero valid sequences. The search enters an absorbing state: the monotonic phase rule prevents any return to the development phase after the turning point is assigned.

**Status: Zero false positives across all tested conditions.**

- Tested at ε = {0.30, 0.50, 0.70, 0.90}, k = {0, 1, 2, 3, 4, 5}, 50 seeds per condition
- Below-diagonal (j < k) success rate: **0.000 everywhere sampled**
- The theorem states a sufficient condition for failure, not a necessary one

The k-j boundary figure (multi-ε panel) is the visual centerpiece: a clean diagonal boundary where the theorem predicts failure, with above-diagonal degradation from secondary failure modes.

### Theoretical Grounding

The prefix constraint violates the hereditary property of matroids (if a sequence is valid, subsets need not be valid). This places the constraint system in the domain of greedoids (Korte & Lovász, 1984), where greedy algorithms lose optimality guarantees. The absorbing state is the specific mechanism by which greedy produces maximal-but-not-maximum independent sets in this non-hereditary system.

---

## 3. The Complete Failure Taxonomy

We traced every false negative (cases where greedy fails on graphs that oracle can solve) to a specific, deterministic root cause. The taxonomy has four layers, discovered in sequence through targeted experiments.

### Layer 1: Phase Classifier Coupling (63/138 cases, 45.7%)

**Mechanism:** The phase classifier assigns phases after sequence construction using `n_setup = ceil(0.2 × n_before_tp)`. When greedy constructs a sequence with only one event before the TP (common under front-loading), `ceil(0.2 × 1) = 1` forces the sole pre-TP event to SETUP, leaving zero development events. The grammar requires k ≥ 1 development events, so validation fails.

**Root cause:** A pipeline coupling failure. The search succeeds at its local objective but silently violates the downstream classifier's hardcoded assumption that `n_before_tp ≥ 5`. This is an implicit contract violation under covariate shift (Sculley's "Hidden Technical Debt" framework).

**Fix implemented:** Grammar-aware classifier that caps setup count to preserve min_development slots. Two-line change in `phase_classifier.py`. Result: all 63 cases flip to valid. Below-diagonal theorem predictions unchanged.

**Experiment chain:** oracle_diff → fn_divergence → tp_misselection → prefix_dominance_test → extraction_internals_diagnosis → phase_classifier_analysis

### Layer 2: Turning Point Misselection (18/138 cases, 13.0%)

**Mechanism:** Greedy selects a heavier-but-earlier TP than oracle (greedy TP weight 0.90 vs oracle 0.78, greedy position index 6 vs oracle 32). This is classical greedy suboptimality — maximizing the local objective (TP weight) at the cost of global constraint satisfaction.

**Fix:** Beam search with width 2 keeps an alternative TP candidate alive. Recovers all 18 cases. **Beam search saturates completely at w=2** — widths 4, 8, 16 add zero recoveries. The valid path for these cases requires only one alternative to survive.

**Oracle TP rank:** Median rank 4 in the weight ordering (oracle typically picks the 4th-heaviest event). However, oracle rank does not predict minimum beam width — the relationship is not mechanistic.

**Experiment chain:** oracle_diff → fn_divergence → tp_misselection → beam_search_sweep

### Layer 3: Sequence Assembly Compression (57/138 cases, 41.3%)

**Mechanism:** Greedy has access to the right events (98.6% of oracle's events are in greedy's candidate pools; 52/57 cases have the complete oracle event set in a single pool). Pool timespan exceeds the grammar threshold in all 57 cases. But greedy's weight-maximizing assembly selects temporally clustered high-weight events, compressing the sequence timespan below the grammar's minimum.

Oracle succeeds on all 57 by selecting lower-weight but temporally spread events from the same pool. The bottleneck is not pool construction or search width — it's the scoring function's single-criterion weight maximization conflicting with the continuous timespan constraint. This is a Resource-Constrained Shortest Path Problem (RCSPP) where the objective gradient is negatively correlated with constraint feasibility.

**Attempted fixes that failed:**
- Beam search (w=2..16): saturates at w=2, 0/57 recovered. Wider beams explore more weight-maximizing variants from the same compressed subspace.
- 1-opt swap repair: 0/57 recovered. "No improving swap available" in 54/57 cases — pool alternatives are similarly compressed. The valid path requires complete reassembly, not local perturbation.

**Theoretical fix (not implemented):** Lexicographic Phase I/Phase II assembly — build a minimum-timespan-satisfying sequence first (feasibility), then optimize weight within that constraint (optimality). Or Pareto label-setting that maintains a frontier of non-dominated (weight, timespan) candidates.

**Experiment chain:** beam_search_sweep → pool_bottleneck_diagnosis → repair_sweep

### Layer 4: Rare Edge Cases (6 prefix dominance + 4 same-TP timespan + 2 selection + 1 reclassification = ~9% combined)

Small populations with distinct mechanisms. The 4 same-TP cases (correct TP but extreme temporal compression) are qualitatively different — greedy and oracle agree on the TP but greedy builds a worse prefix. The 6 prefix dominance cases are the only ones matching the "early distractor" hypothesis (max early weight > max viable weight).

---

## 4. Quality on Successes: Bimodal Behavior

When greedy succeeds, it's near-optimal:

| ε | k | Ratio Mean | Ratio Min | Cases < 0.90 |
|---|---|---|---|---|
| 0.30 | 0 | 0.996 | 0.975 | 0 |
| 0.90 | 3 | 1.001 | 0.979 | 0 |

Zero cases below 0.90 across all 16 (ε, k) combinations. The approximation ratio does not degrade with front-loading — ε only increases failure rate, not quality of successful extractions.

**Finding:** Greedy exhibits sharp bimodal behavior. Near-optimal or complete failure, with no graceful degradation. The problem is feasibility-dominated: surviving the constraints is the hard part, and any sequence that survives is structurally guaranteed to be high-weight.

**Side note:** Greedy outscores oracle in some cases (ratio > 1.00), confirming that `oracle_extract` is a strong but not globally optimal baseline. It exhaustively searches TPs but uses a heuristic assembly around each TP that occasionally sacrifices weight for feasibility.

---

## 5. What Did Not Reproduce

### Valley of Death (Non-Monotonic Regularization)

In Lorien's narrative DAGs, partially relaxing constraints made extraction *worse* before making it better. We tested this on synthetic graphs:

- Swept prefix constraint k = 0..10 at ε = {0.50, 0.90}
- Swept timespan constraint 0.00..0.50 at ε = {0.50, 0.90}
- **Result: Monotonic on all four curves. No valley.**

The valley of death is topology-dependent — it requires specific graph motifs (likely the modular structure of narrative DAGs with dense scenes connected by sparse bridges) that the bursty generator doesn't produce. This is a publishable negative result: the non-monotonic regularization discovered in Lorien is not a universal property of greedy search under sequential constraints.

### Prefix Dominance Condition

The hypothesis that greedy fails when `max_weight(early events) > max_weight(viable events)` was tested on all 138 FN cases. It held in only 6/138 (4.3%). The failure mechanism is downstream of TP positioning — it's about how the sequence is assembled and classified, not where the TP sits.

### Causal Depth as Operative Variable

Proposed as a replacement for temporal index in the theorem statement. Never tested because the oracle diff and divergence analysis showed the failures are process properties (classifier + assembly), not static graph properties. Causal depth may still be relevant for other purposes but is not the right variable for explaining the observed false negatives.

---

## 6. Experiments (Complete List)

All experiments are in `~/rhun/experiments/` with outputs in `experiments/output/`.

| # | Experiment | Script | Key Finding |
|---|---|---|---|
| 1 | Position sweep | run_position_sweep.py | Validity decreases with ε; theorem has 0 FP, increasing FN |
| 2 | Oracle diff | run_oracle_diff.py | 138/138 FN are solvable (100% search-tricked) |
| 3 | FN divergence | run_fn_divergence_analysis.py | 97.1% different TP; greedy median 0 dev events |
| 4 | TP misselection | run_tp_misselection_analysis.py | Greedy picks heavier (+13%) but earlier (5×) TP |
| 5 | Prefix dominance | run_prefix_dominance_test.py | Condition holds in only 6/138 (dead) |
| 6 | Extraction internals | run_extraction_internals_diagnosis.py | 63 classifier failures, 67 timespan failures |
| 7 | Population A violations | run_population_a_violations.py | 67/67 are insufficient_timespan |
| 8 | Phase classifier analysis | (document only) | ceil(0.2 × 1) = 1 deterministic interaction |
| 9 | k-j boundary | run_kj_boundary.py | Clean diagonal at ε=0.50 |
| 10 | Multi-ε k-j boundary | run_kj_boundary_multi_epsilon.py | Diagonal holds at all ε; above-diagonal scales with k and ε |
| 11 | Classifier fix + re-eval | run_kj_boundary_multi_epsilon.py (fixed) | 63 cases recovered; below-diagonal unchanged |
| 12 | Beam search sweep | run_beam_search_sweep.py | 18/75 at w=2; saturates completely |
| 13 | k-j beam search | (in beam sweep) | Modest above-diagonal improvement at ε=0.90 |
| 14 | Pool bottleneck | run_pool_bottleneck_diagnosis.py | 98.6% oracle events in pool; not the bottleneck |
| 15 | Valley of death | run_valley_of_death.py | No valley; monotonic on all curves |
| 16 | Approximation ratio | run_approximation_ratio.py | Ratio ≥ 0.95 everywhere; bimodal behavior |
| 17 | 1-opt repair | run_repair_sweep.py | 0/57 recovered; local repair insufficient |

---

## 7. The Paper

### Status: Ready to write.

Target: NeurIPS or ICML 2026 workshop on constrained optimization, structured generation, safe AI, or ML systems. 6–8 pages.

### Proposed Structure

- **§1 Introduction:** Greedy extraction from causal DAGs under sequential phase constraints. The question: when and why does greedy fail, and what breaks first?
- **§2 Formalism:** Graph model, grammar (DFA), greedy search, prefix constraint. Non-hereditary constraint space (greedoid connection, Korte & Lovász 1984).
- **§3 The Prefix-Constraint Impossibility Result:** Theorem + proof sketch. Multi-ε k-j boundary figure. Zero false positives.
- **§4 Failure Taxonomy:** Three secondary failure modes beyond the theorem. Classifier coupling (63 cases, fix, before/after). TP misselection (18 cases, beam saturation). Assembly bottleneck (57 cases, pool diagnosis, repair null, oracle gap). RCSPP connection.
- **§5 Bimodal Quality:** Approximation ratio ≥ 0.95. Feasibility-dominated landscape. No graceful degradation.
- **§6 Discussion & Future Work:** Valley of death null (topology-dependent). Phase I/II assembly for layer 3. Cross-domain validation. Connections to process mining, scheduling.

### Core Figures

1. Position sweep (validity vs ε)
2. Multi-ε k-j boundary panel (4 heatmaps showing diagonal boundary + above-diagonal degradation)
3. Classifier fix before/after (above-diagonal improvement across ε and k)
4. Approximation ratio or beam search saturation

---

## 8. Open Research Directions

These are genuinely open. They're ordered by a mix of tractability and potential impact. Future collaborators should feel free to challenge, reorder, or add to this list.

### Near-Term (Weeks)

1. **Solve the 57-case assembly bottleneck.** The pool contains everything oracle needs. The scoring function is the problem. Candidate approaches: lexicographic Phase I/II assembly (feasibility-first, then optimize), Pareto label-setting (multi-objective frontier), dynamic Lagrangian penalty on timespan. Any of these that cracks 40+ of the 57 cases is a paper result.

2. **Improve oracle_extract.** Oracle outscores greedy in most cases but greedy occasionally wins (ratio > 1.00). Oracle exhaustively searches TPs but uses heuristic assembly. A true ILP-based oracle would give a tight upper bound and might reveal whether the 57-case gap is closer to the theoretical optimum than we think.

3. **Formal proof of the theorem.** Currently empirical with zero false positives. For a theory-track submission, needs a combinatorial proof. The absorbing state characterization is the proof mechanism: formalize the DFA, show no transition exists from post-TP state back to DEV, prove no valid completion exists.

4. **Invariance tests.** Sweep graph density (sparse → dense causal links), actor count (2 → 50), weight distribution (uniform, power-law, bimodal). Each invariance confirmed is a corollary of the theorem.

### Medium-Term (Months)

5. **Cross-domain validation: incident response.** Build the `adapters/incident.py` generator — synthetic distributed system outage traces where cascading alerts mask quiet root-cause events. If the same four-layer taxonomy appears, the result generalizes beyond the bursty generator.

6. **The constraint family.** The prefix constraint (k elements before TP) is one member of a family: suffix constraints (m elements after TP), balanced constraints (≥p% per phase), ordering constraints. Which create absorbing states and which don't? A general characterization of which sequential constraints are "greedy-safe" would be the deepest theoretical contribution.

7. **Hunt the valley of death.** It appeared in Lorien's modular narrative DAGs but not in bursty synthetic graphs. What topological motif triggers it? Candidate: build a modular DAG generator (stochastic block model with dense clusters and sparse bridges). If the valley emerges on modular graphs, non-monotonic regularization is a property of modular constraint landscapes — connecting narrative structure theory to combinatorial optimization.

8. **Process mining connection.** The conformance checking community (ICPM, BPM conferences) extracts execution traces from noisy concurrent event logs and checks them against normative process models. They face "concept drift" (analogous to ε temporal shift). Our taxonomy maps to their pain points. Framing paper for that community.

### Longer-Term

9. **The assembly bottleneck as a general result.** The bimodal quality finding (near-optimal or catastrophic, no middle ground) and the RCSPP connection suggest a general phenomenon: single-criterion greedy on feasibility-dominated problems with negatively correlated objectives and constraints. Is there a theorem here about when greedy exhibits bimodal behavior?

10. **Connections to planning and scheduling.** Job-shop scheduling has sequential precedence constraints on DAGs and greedy dispatching rules. The absorbing state may be a known failure mode in different language.

11. **GPU scaling for the k-j boundary.** The signature figure (k-j boundary across 10,000+ instances with varying topology) requires vectorized batch execution. Great Lakes cluster.

---

## 9. Methodology (Unchanged)

All experiments follow the principles from the original RHUN_CONTEXT.md:

1. **Verification-first:** Baseline conditions checked before experimental conditions.
2. **Fully deterministic:** All randomness seeded. Same seed → same result.
3. **$0.00 compute:** No LLM calls, no API fees. Local CPU only.
4. **Null results are findings.** Prefix dominance (4.3%), repair null (0/57), valley of death (monotonic) — all published as structural results.
5. **Error decomposition:** TP/FP/FN/TN breakdown. Directional error analysis, not aggregate accuracy.

---

## 10. What NOT To Do (Updated)

Everything from the original RHUN_CONTEXT.md still applies, plus:

- **Do NOT assume the 57-case assembly gap is a pool construction problem.** Pool diagnosis proved 98.6% event coverage. The bottleneck is scoring/assembly.
- **Do NOT assume beam search at higher widths will help.** It saturates at w=2. The failure is in the assembly framework, not search depth.
- **Do NOT assume local repair (1-opt, 2-opt) will fix timespan compression.** Tested and confirmed null. The weight-optimal skeleton is structurally alien to the feasibility-optimal skeleton.
- **Do NOT conflate "oracle" with "global optimum."** oracle_extract is exhaustive over TPs but heuristic in assembly. Greedy sometimes outscores it.
- **Do NOT claim the valley of death is universal.** It's topology-dependent. It appeared in Lorien, not in synthetic bursty graphs.
- **Do NOT use the prefix dominance condition** (max early weight > max viable weight) as a theorem extension. It explains only 4.3% of false negatives.

---

## 11. Multi-AI Workflow (Updated)

This project uses multiple AI assistants with calibrated roles:

- **Claude (Anthropic):** Primary analysis, experiment design, theoretical hygiene, codex prompt generation. Conservative about claims. Pushes back on overclaiming. Strong on error decomposition, precise mathematical framing, and experiment sequencing. Killed the prefix dominance hypothesis, the causal depth proposal, and the "beam search will fix everything" expectation through targeted empirical tests.
- **Gemini (Google):** Strategic framing, creative hypothesis generation, literature connections. Best contributions: greedoid/matroid framing, RCSPP connection, pipeline fragility under covariate shift, feasibility-dominated landscape concept, Phase I/II assembly suggestion, process mining community identification. Weaknesses: inflates terminology, predicts empirical outcomes with false confidence. ~40% of Gemini's ideas survived scrutiny in this session (improved from the historical ~20%).
- **Codex agents (Cursor/Claude Code):** Implementation. Received detailed prompts with pre-check requirements, verification conditions, expected outputs, and commit messages. Executed 17 experiments with zero implementation failures.

**Calibration rule (unchanged):** If an AI says "this shatters/revolutionizes X" or invokes physics without proving the mathematical structure applies — downgrade to the precise empirical statement. The findings are genuinely interesting. They don't need inflation.

---

## 12. Running the Codebase

```bash
# Setup
cd ~/rhun
source .venv/bin/activate

# Tests (19 passing, includes fixed classifier behavior)
pytest -v

# Core experiments
python experiments/run_position_sweep.py
python experiments/run_kj_boundary_multi_epsilon.py
python experiments/run_oracle_diff.py
python experiments/run_beam_search_sweep.py
python experiments/run_approximation_ratio.py
python experiments/run_valley_of_death.py

# Quick smoke test (with fixed classifier)
python -c "
from rhun.generators.bursty import BurstyGenerator, BurstyConfig
from rhun.extraction.search import greedy_extract, oracle_extract
from rhun.extraction.grammar import GrammarConfig
from rhun.theory.theorem import check_precondition

g = BurstyGenerator().generate(BurstyConfig(seed=42, epsilon=0.8))
grammar = GrammarConfig(min_prefix_elements=1)
greedy = greedy_extract(g, 'actor_0', grammar)
oracle, _ = oracle_extract(g, 'actor_0', grammar)
pred = check_precondition(g, 'actor_0', grammar)
print(f'Greedy valid: {greedy.valid}, Oracle valid: {oracle.valid}')
print(f'Greedy score: {greedy.score:.3f}, Oracle score: {oracle.score:.3f}')
print(f'Theorem predicts failure: {pred[\"predicted_failure\"]}')
if greedy.valid and oracle.valid:
    print(f'Approximation ratio: {greedy.score / oracle.score:.4f}')
"
```

---

## 13. Key Files

| File | Purpose |
|---|---|
| `rhun/extraction/phase_classifier.py` | Phase classification — now with `min_development` parameter (the classifier fix) |
| `rhun/extraction/search.py` | greedy_extract, oracle_extract, beam_search_extract, repair_timespan |
| `rhun/extraction/grammar.py` | GrammarConfig with min_prefix_elements, min_timespan_fraction |
| `rhun/theory/theorem.py` | check_precondition, verify_prediction, diagnose_absorption |
| `rhun/generators/bursty.py` | BurstyGenerator with ε front-loading parameter |
| `experiments/output/` | All experiment results as JSON + markdown summaries |
| `RHUN_CONTEXT.md` | Original context (architecture, Lorien relationship, terminology) |
| This file | Current research state |
