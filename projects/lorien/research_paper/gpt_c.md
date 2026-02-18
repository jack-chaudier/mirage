Below is (1) a **condensed “context pack”** you can hand to Claude Code so it doesn’t miss any of the key claims/results, and (2) a **single copy‑paste Claude Code prompt** that instructs it to generate a **full LaTeX research paper draft** inside `~/lorien`, pulling exact numbers from your repo artifacts (JSON, scripts, etc.) and generating tables/figures automatically. 

---

## 1) Context pack for the paper draft

### Core thesis (positioning)

* **Problem framing:** Sequential token prediction yields locally coherent but globally arbitrary fiction; “cathedral” storytelling requires long-range structural constraint satisfaction (promises/payoffs, goal persistence, thematic resonance).
* **Lorien’s inversion:** generate structure first via deterministic simulation + scoring + grammar-constrained arc extraction; use an LLM only for final prose rendering (“crystallization”).
* **Architectural constraint:** *LLM must not drive structural decisions.* Structural pipeline steps 1–3 are deterministic and reproducible. 

### System summary (what exists)

* Deterministic multi-agent simulation in Python; each run yields ~200 events with types like `CHAT, CONFLICT, REVEAL, LIE, CONFIDE, CATASTROPHE, PHYSICAL, OBSERVE` in locations like dining_table/foyer/kitchen/balcony/bathroom.
* **Metrics:** tension, irony (belief gap), significance; scene segmentation by tension dynamics.
* **Arc extraction:** grammar-constrained beat sequence: `SETUP → COMPLICATION → ESCALATION → TURNING_POINT → CONSEQUENCE` with structural constraints; arc validity defined by satisfying grammar.
* **Rashomon extraction:** one arc per agent from the same objective event trace. 

### Scenario (the “Dinner Party”)

* 6 agents: Thorne (host), Elena (wife having affair), Marcus (affair partner + fraud), Victor (journalist), Lydia (mediator), Diana (witness).
* Pressure-cooker topology: opposed goals, small location graph; intentionally designed for maximal structural diversity from minimal content. 

### Previously established baseline results to cite in paper

* Single-protagonist arc validity: **50/50 seeds valid (100%)** after Phase 4 fixes (prior 33/50).
* Rashomon sweep: **294/300 valid arcs (98%)**, mean **5.88/6** valid arcs per seed; 44/50 seeds yield all 6 valid arcs (88%).
* Per-agent mean scores (population sweep): Lydia highest, Elena lowest; mediator containment→failure is structurally clean. 

### Wounds + overlap (structure diagnostics)

* Overlap matrix is Jaccard similarity of events shared across two protagonists’ arcs; highest overlaps (e.g., Thorne–Victor, Lydia–Thorne) indicate entanglement hubs.
* “Wounds” are recurring (agent-pair × location × beat-type) patterns that frequently appear as escalation/turning points; some are near-invariant attractors (e.g., Marcus–Victor at dining_table). 

### Canon & lore loop (critical negative result)

* WorldCanon persists between stories:

  * `claim_states` categorical belief states,
  * location tension residue,
  * narrator-invented `texture` facts.
* **Orthogonality finding:** texture accumulates and enriches prose but does **not** affect structure; skip-narration vs full narration yields *identical structural measurements.*
* **Failed decay experiment:** decaying `claim_confidence` floats and tension residue had **zero** effect because the simulation reads **categorical** `claim_states`, not the decayed floats. 

### Chain degradation (the “heat death” measurement)

* Seed-order experiment across multiple chains showed the apparent U-shape was **seed-order artifact**; the true effect is **monotonic suppression** of mean arc score with depth.
* Degradation rate: ~**0.015 mean score per chain step**, implying a 12-episode season would approach validity boundary by ~episode 8–9 without intervention. 

---

## 2) Two new experiments that define the paper’s contribution

### Experiment 1: Epiphany bypass (categorical belief perturbation)

**Goal:** Prove the sim is causally sensitive to `claim_states`, and test whether a single belief mutation can recover degraded structure mid-chain.

* Setup: chained stories with `--skip-narration` (zero LLM calls), using seed schedule designed for isolation (e.g., 7,7,7).
* Method: save canon after Story B; mutate one *central* claim_state; run Story C baseline vs mutated.
* Results (from the conversation’s artifacts):

  * **Event divergence at tick 0** for the central mutation → causal sensitivity is immediate.
  * **Control mutation** (peripheral belief) yields near-zero change and/or no meaningful improvement.
  * **Score did not recover**; central mutation produced a **negative delta** in mean score (baseline ≈ 0.668, mutated ≈ 0.628; Δ ≈ −0.040).
  * Trace shows the sim reads `claim_states` heavily (order of ~13k reads/story).
    **Interpretation:** belief perturbations are causal, but single-belief changes cannot rescue a system whose dramatic fuel (information asymmetry) has already been spent; also, arbitrary suspicion can destabilize protagonist coherence.

### Experiment 2: Goal evolution (utility vector adaptation)

**Goal:** Test whether **updating agent objectives** (not beliefs) to match post-revelation canon restores structural quality.

* Setup: hold exhausted canon fixed (after Story B), same seed for Story C; vary agent goal vectors/commitments by condition.
* Conditions:

  * Depth-0 fresh reference (seed 7, empty canon) — mean ≈ **0.721**
  * Depth-2 baseline (exhausted canon) — mean ≈ **0.668**
  * Thorne-only evolution — mean ≈ **0.641**, and **Marcus arc invalid** (per JSON: `valid_arcs=5`, Marcus score null)
  * Targeted evolution (Thorne + Elena) — mean ≈ **0.683** (+0.015 vs baseline)
  * Full evolution (all agents) — mean ≈ **0.715** (+0.047 vs baseline; only −0.006 vs depth-0)
* Event distribution shifts under full evolution (qualitative “genre shift”): fewer REVEAL/CHAT; more CONFLICT/CATASTROPHE/OBSERVE (exact counts should be taken from `goal_evolution_experiment.json`).
  **Interpretation:** chain degradation is largely an **objective-function mismatch** problem: agents pursue stale goals after world truths become common knowledge. Coordinated multi-agent goal evolution restores structural quality near depth-0 and can change the dominant dramatic mode from revelation-driven to consequence-driven.

### Architectural takeaway (future work hook)

* **Stance Matrix pattern:** don’t let an LLM output floats; instead define discrete, tested stance profiles per agent (e.g., CONCEALER → SURVIVOR → BETRAYED) as immutable parameter packages. Between episodes, choose stance keys deterministically or via a constrained classifier (LLM-as-classifier is optional future work).

---

# 3) Copy-paste Claude Code prompt (writes full LaTeX paper in `~/lorien`)

Paste everything below into Claude Code **verbatim**.

```text
You are Claude Code running inside the repository at ~/lorien. Your task is to write a full, compile-ready LaTeX research paper draft (not an outline) describing the Lorien/NarrativeField system and two new experiments:
(1) Epiphany bypass (categorical claim_state mutation), and
(2) Goal evolution (agent goal-vector/commitment adaptation).

CRITICAL CONSTRAINTS / STYLE:
- Accuracy over rhetoric. Do NOT claim universal narrative laws. Frame findings as “in this system / in this scenario / in this architecture.”
- Preserve the core thesis: the LLM is not used in the structural pipeline; experiments must be skip-narration/deterministic.
- The paper must read like a submission-ready AIIDE/ICIDS paper draft: clear methods, ablations/controls, reproducibility details, limitations.
- Use exact numbers from repository artifacts. Do not “round-trip” from memory; parse JSON outputs.
- Do not modify engine code. You may add paper files and figure-generation scripts under a new /paper directory.

DELIVERABLES:
1) Create directory: ~/lorien/paper/
2) Write LaTeX: ~/lorien/paper/main.tex (full paper)
3) Write BibTeX: ~/lorien/paper/refs.bib (minimal but real references)
4) Create figures from data: ~/lorien/paper/figures/*.pdf (or .png) + a script to generate them
5) Add a Makefile or latexmkrc so `latexmk -pdf main.tex` works in /paper
6) Produce a compiled PDF if toolchain exists; otherwise ensure main.tex is compile-ready.

PAPER TARGET LENGTH:
- 8–10 pages in standard article style (not including appendix), but prioritize completeness over strict length.

STEP 0 — DISCOVER CONTEXT FILES + ARTIFACTS
Search the repo for the main context doc and experiment artifacts. Likely locations:
- A context document containing “Lorien / NarrativeField — Complete Project Context”
- src/engine/scripts/output/epiphany_experiment.json
- src/engine/scripts/output/sim_read_path_trace.txt
- src/engine/scripts/output/goal_evolution_experiment.json
Also locate earlier baseline sweep artifacts:
- src/engine/scripts/output/rashomon_sweep_1_50.json
- src/engine/scripts/output/wound_analysis_1_50.json
- research chain artifacts (research_chain*.json)

If any are missing, locate the scripts that generate them (e.g., src/engine/scripts/test_epiphany_bypass.py, test_goal_evolution.py, sweep_rashomon.py, analyze_wounds.py, research_chain.py). Prefer reading existing JSON; only re-run scripts if an artifact is missing.

STEP 1 — EXTRACT ALL KEY NUMBERS PROGRAMMATICALLY
Write a small Python helper script under ~/lorien/paper/scripts/extract_results.py that:
- Loads the JSON artifacts (epiphany_experiment.json, goal_evolution_experiment.json, rashomon_sweep_1_50.json, wound_analysis_1_50.json, and the seed-order/chain artifacts if available).
- Emits a machine-readable “results_summary.json” and a human-readable “results_summary.md” with:
  A) Baseline system validity and sweep stats (single-protagonist validity, Rashomon validity, per-agent means, overlap highlights, invariant wounds)
  B) Chain degradation measurements (mean score vs depth; estimate slope ~0.015 per step; confirm from artifacts)
  C) Epiphany bypass: baseline vs mutated mean score; delta; per-agent scores; valid_arcs; first_divergence_tick; control mutation effect; number of CANON_READ lines/reads if recorded
  D) Goal evolution: depth-0 mean, depth-2 baseline mean, thorne-only mean + valid_arcs and Marcus null, targeted mean, full evolution mean; full-vs-baseline delta; full-vs-depth0 delta
  E) Event-type distributions for baseline vs full evolution (and depth-0 if available): counts per event_type and deltas.

All figures/tables in the paper must be sourced from this script output or direct JSON parsing.

STEP 2 — GENERATE FIGURES (AUTOMATED)
Create ~/lorien/paper/scripts/make_figures.py that reads results_summary.json and generates:
Figure 1: System architecture diagram (simple box/arrow figure). This can be drawn in LaTeX TikZ OR as a generated SVG/PDF. Keep it clean.
Figure 2: Chain degradation plot: mean arc score vs depth position (A–E), including the seed-order averaged means.
Figure 3: Experiment 1 results: baseline vs mutated mean score bar chart (plus control mutation bar).
Figure 4: Experiment 2 results: bars for depth-0, baseline depth-2, thorne-only, targeted, full evolution (include error bars only if data supports).
Figure 5: Event type distribution shift (baseline vs full evolution), as grouped bars.

Save figures into ~/lorien/paper/figures/ and reference them from LaTeX with captions.

Use matplotlib (no seaborn). Do not hardcode colors; default is fine.

STEP 3 — WRITE THE PAPER (main.tex)
Write a complete LaTeX paper with the following structure (adjust as needed):

Title (choose a precise one):
Option A: “Restoring Structural Quality in Deterministic Narrative Simulation via Coordinated Goal Evolution”
Option B: “Objective-Function Adaptation for Long-Form Deterministic Narrative Simulation”

Abstract:
- 150–250 words.
- State: deterministic simulation + grammar-constrained arc extraction yields valid structure; chained canon causes monotonic degradation; belief perturbations are causal but insufficient; coordinated goal evolution restores scores near depth-0; propose stance-matrix as future work.

1. Introduction
- Cathedral vs token prediction framing.
- Lorien approach (simulation → metrics → grammar search → optional narration).
- The long-form problem: chained episodes degrade as information equilibrates / objectives become stale.
- Contributions bullet list (3–6 bullets).

2. Related Work
Keep it short but real. Include citations in refs.bib to:
- Interactive drama / drama management (e.g., Mateas & Stern “Façade”)
- Narrative planning / story generation (e.g., Riedl & Young)
- Multi-agent simulation in narrative (whichever you can responsibly cite)
- Grammar-based plot / story structure (e.g., narrative grammars / plot units)
Do NOT overclaim novelty; be precise: our novelty is the combination of deterministic simulation + grammar-constrained extraction + measured chain degradation + goal evolution intervention.

3. System: Lorien / NarrativeField
- Deterministic simulation layer (event types, agents, locations, determinism).
- Canon state (claim_states, tension residue, texture) and lore loop; emphasize texture is narration-only and structurally orthogonal in current system.
- Metrics: tension/irony/significance.
- Arc extraction grammar and Rashomon multi-protagonist extraction.
Include Figure 1 architecture diagram.

4. Baseline Empirical Properties
- Single-protagonist validity (50/50) and Rashomon sweep (294/300) with per-agent means table.
- Wounds and overlap matrix: define them; summarize invariant wounds and key overlaps.
- Chain degradation measurement and seed-order falsification of the “U-shape” story.
Include Figure 2 and a table of position means.

5. Experiment 1: Epiphany Bypass (Claim-State Perturbation)
- Hypothesis: if claim_states are causal levers, single categorical mutation should alter events; may recover degraded score.
- Method: seeds (e.g., 7,7,7), extract canon after B, mutate central claim, run C baseline/mutated + control mutation; skip-narration.
- Results: first divergence tick; mean score change; per-agent changes; control condition; trace evidence of claim_state reads.
- Interpretation: causal sensitivity confirmed; score recovery not achieved; motivates need for objective-function adaptation.

Include Figure 3.

6. Experiment 2: Coordinated Goal Evolution
- Hypothesis: degraded chains reflect objective-function mismatch after revelations; adapting goal vectors/commitments to post-revelation canon restores structural quality.
- Method: fixed exhausted canon; conditions (depth-0, baseline, thorne-only, targeted, full); how goal vectors were evolved (template copy + minimal edits), no new mechanics.
- Results: report mean scores, deltas, validity (including Marcus null in thorne-only), and event-type distribution shift; emphasize coordinated evolution > unilateral.
- Interpretation: goal evolution restores near depth-0 quality; long-form viability via episodic stance transitions.

Include Figure 4 and Figure 5, plus a table.

7. Discussion
- What this suggests about long-form simulation: managing information equilibrium is necessary but not sufficient; objectives must evolve.
- Determinism & reproducibility advantages (no prompt sensitivity in experiments).
- Limitations: one scenario; small agent cast; goal evolution currently hand-designed; event taxonomy constraints.
- Negative results as evidence: texture orthogonality; confidence decay ineffectiveness; belief perturbation not sufficient.

8. Future Work
- Stance Matrix (discrete tested stance profiles; deterministic application; optional LLM as classifier constrained to stance keys).
- Affordance loop (Evidence/Commitment/Leverage) as causal state; must be in sim read path.
- Season orchestrator / search over seeds using structural metrics.
- “Dinner party as cosmology” (genre lens via different significance profiles) as additional validation experiment.

9. Conclusion
- Summarize main empirical findings and architectural contribution.

Appendix (optional but helpful)
- Reproducibility: exact commands to regenerate artifacts
- Data schema snippet for claim_states and goal vectors (from code)
- Additional tables (per-agent scores for all conditions)

STEP 4 — VALIDATE AGAINST CODE + ARTIFACTS
Before finalizing the paper text, confirm:
- Claim_states structure and semantics from src/engine/narrativefield/schema/canon.py
- GoalVector fields and commitments from the simulation code (locate definitions)
- Arc grammar constraints from src/engine/narrativefield/extraction/arc_search.py (DO NOT EDIT)
- Rashomon extraction behavior from rashomon.py (DO NOT EDIT)

STEP 5 — WRITE refs.bib (MINIMUM SET)
Create refs.bib with at least 8 references with correct BibTeX fields. Use known canonical works (Façade, narrative planning, drama management, narrative grammars). If you can’t verify some details, choose fewer but correct ones. Prefer correctness.

STEP 6 — COMPILE
Try to compile:
cd ~/lorien/paper && latexmk -pdf main.tex
If latexmk is missing, still ensure main.tex is compile-ready and mention the compile command in a comment.

OUTPUT CHECKLIST
- main.tex compiles (or is compile-ready)
- tables have exact values from JSON
- figures are generated from scripts, not manual
- no overclaims
- negative results included
- methods are reproducible and precise

Proceed now.
```

---

If you want, I can also produce a **short “reviewer-proof” contributions list** and **a ready-made abstract** that matches the exact narrative of the two experiments—so Claude Code can paste it in with minimal rework. 
