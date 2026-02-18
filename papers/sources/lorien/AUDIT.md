# Paper Audit: Goal Evolution in Deterministic Narrative Simulation

**Audited:** 2026-02-12T08:33:30Z
**Paper file:** `goal_evolution_paper.tex`
**Data sources verified:**
- `src/engine/scripts/output/epiphany_experiment.json`
- `src/engine/scripts/output/goal_evolution_experiment.json`
- `src/engine/scripts/output/canon_after_B.json`
- `src/engine/scripts/output/wound_analysis_1_50.json`
- `src/engine/scripts/output/research_chain.json`
- `src/engine/scripts/output/research_chain_reverse.json`
- `src/engine/scripts/output/research_chain_seed7first.json`
- `src/engine/scripts/output/research_chain_seed7last.json`
- `src/engine/scripts/output/rashomon_sweep_1_50.json`

## Critical Issues (must fix before submission)

### Numerical Errors
- [ ] `Section 1 / line 45` and `Table 1 caption / line 127` — text says “decline monotonically,” but corrected position means are `0.696 -> 0.679 -> 0.644 -> 0.650 -> 0.637` (rebound at position 3). Source: computed from `research_chain*.json.chain_summary.mean_score_progression` across the 4 orderings (position means: `[0.695765, 0.679033, 0.643728, 0.650132, 0.637219]`). Fix: say “overall downward trend with a local rebound at position 3.”
- [ ] `Section 4.1 / line 142` — “all four orderings show negative slope from position 0 to 4” is incorrect. `research_chain.json` has positive first-to-last delta `+0.009602` (`0.696877 -> 0.706479`). Fix: “3 of 4 orderings show negative first-to-last slope; position-averaged means still decline overall.”
- [ ] `Section 4.1 / line 142` — single-chain last value is written as `0.707`; source is `0.706479` (rounds to `0.706` at 3 decimals). Source: `research_chain.json.chain_summary.mean_score_progression[4]`. Fix table/text value.
- [ ] `Section 3.1 / line 100` — half-life approximations are off. For `alpha_t=0.6`, half-life is `ln(0.5)/ln(0.6)=1.357` episodes, not `~1.5`; for `alpha_b=0.85`, half-life is `4.265`, not `~4.6`. Source: `src/engine/narrativefield/schema/canon.py:269` and formula. Fix numbers or remove half-life parentheticals.
- [ ] `Section 5.2 / line 186` — “mutated field appeared in top-20 most-accessed beliefs” is unsupported. `epiphany_experiment.json.trace_summary.top_fields` does not include `thorne x secret_embezzle_01` (not in top 30). Fix: remove top-20 claim or report exact rank/count from full field counts.
- [ ] `Section 6.2 / line 254` — “ranking is strictly monotonic with agent count” is false as written: 0-agent baseline (`0.668`) > 1-agent thorne-only (`0.641`). Source: `goal_evolution_experiment.json.conditions.*.mean_valid_arc_score`. Fix wording to “nonlinear / superlinear collective pattern.”

### Factual Errors
- [ ] `Section 5.1 / lines 157-160` — method omits key fallback detail: intended target `thorne.secret_affair_01` is not mutable (`believes_true`), so script falls back to `thorne.secret_embezzle_01`. Source: `test_epiphany_bypass.py:_select_primary_mutation` (`src/engine/scripts/test_epiphany_bypass.py:391`) and `epiphany_experiment.json.mutations.primary.reason="fallback_ranked"`. Fix: explicitly describe fallback.
- [ ] `Section 3.1 / Eq. 1` — utility equation is incomplete relative to implementation: `SOCIAL_MOVE` gets an extra fixed `-0.10` penalty at selection time. Source: `src/engine/narrativefield/simulation/decision_engine.py:1045`. Fix: add this term as a special-case note.
- [ ] `Section 3.3 / lines 108-112` — grammar is described as strict `setup -> complication -> escalation -> turning_point -> consequence`, but validator/search allow `complication OR escalation`, enforce exactly one TP, and add extra constraints (4–20 beats, protagonist share >=60%, causal continuity, minimum span). Source: `src/engine/narrativefield/extraction/arc_validator.py:22-90`, `src/engine/narrativefield/extraction/arc_search.py:614-683`. Fix: present actual grammar/constraints.
- [ ] `Section 3.1 / line 100` — paper implies between-episode decay is active in the reported chain experiments, but default chain runner args are no decay (`--tension-decay 1.0`, `--belief-decay 1.0`) unless explicitly set. Source: `src/engine/scripts/research_chain.py:625-634`. Fix: clarify “engine supports decay; experiment config specifies whether enabled.”

### Overclaiming
- [ ] `Abstract / line 34` — “confirming that agent beliefs causally drive story structure” -> Suggested: “providing evidence, in this system, that belief-state perturbations causally affect extracted arc structure.”
- [ ] `Introduction contribution 2 / line 49` — “narrative structure is causally driven by belief propositions” -> Suggested: “our perturbation results indicate strong causal sensitivity to belief propositions.”
- [ ] `Section 4.1 / line 142` — “ruling out seed-quality artifacts” -> Suggested: “substantially reducing the likelihood that observed degradation is only a seed-order artifact.”
- [ ] `Section 5.3 / line 192` — “phase-transition-like sensitivity” -> Suggested: “a sharp, threshold-like sensitivity pattern in this scenario.”
- [ ] `Section 6.2 / line 284` — “demonstrates that narrative quality is a property of interaction topology, not individual agents” -> Suggested: “supports the hypothesis that interaction topology is a major driver of quality in this setup.”
- [ ] `Conclusion / line 349` — “fundamentally a property” -> Suggested: “appears to be strongly shaped by the multi-agent interaction topology.”
- [ ] `Conclusion / line 351` — “across arbitrarily many episodes” -> Suggested: “across longer episode sequences than fixed-goal baselines.”

### Compilation Errors
- [ ] No hard compile failures found (`make clean && make` succeeds), but persistent layout warnings remain: overfull boxes at `goal_evolution_paper.tex` lines 95, 111, 171-182, 224-235, 260-274 (from `goal_evolution_paper.log`). Fix with equation/table width adjustments.
- [ ] First `pdflatex` pass shows transient undefined citation/ref warnings (expected before bibtex/rerun), then resolves by final pass. Fix workflow note: always run full `make` before checking warnings.

## Recommended Additions

### Missing Tables
1. **Overlap Shift Table (Baseline vs Targeted/Full/Thorne-only)** — Add top positive/negative Jaccard shifts with pair-level deltas. Data: `goal_evolution_experiment.json.comparisons.largest_overlap_shifts_vs_baseline`.
2. **Wound Topology Binary Matrix** — Rows = wound patterns, cols = conditions, cells = present/absent (+frequency annotation). Data: `goal_evolution_experiment.json.comparisons.wound_topology`.
3. **Canon State Summary After Story B** — Claims x agents belief-state grid to support “information equilibrium” diagnosis. Data: `canon_after_B.json.world_canon.claim_states`.
4. **Mutation Diff Table (Field-level)** — Condensed “before -> after” rows for Thorne + Elena (and maybe Marcus) to make intervention reproducible. Data: `goal_evolution_experiment.json.conditions.*.mutation_details`.
5. **Failed Intervention Summary Table** — If texture/confidence/tension interventions were run elsewhere, add explicit null-effect rows (with seeds and deltas). Currently absent from this paper and the cited experiment sections.

### Missing Figures
1. **Condition Mean Score Bar Chart** — Grouped bars for `Depth-0`, `Baseline`, `Thorne-only`, `Targeted`, `Full`; include deltas to baseline. Data: `goal_evolution_experiment.json.conditions.*.mean_valid_arc_score`. Feasible in `pgfplots`.
2. **Per-agent Heatmap Across Conditions** — 6x5 matrix showing agent score shifts and null arc cell for Marcus in thorne-only. Data: `goal_evolution_experiment.json.comparisons.per_agent_table`. Feasible in `pgfplots`.
3. **Chain Position Trend Plot** — Line plot of corrected means by position plus faint lines for each ordering. Data: all `research_chain*.json.chain_summary.mean_score_progression`. Feasible in `pgfplots`.
4. **Event-Type Distribution Chart** — Stacked or grouped bars to visualize genre/interaction shift. Data: `goal_evolution_experiment.json.comparisons.event_type_distribution_rows`. Feasible in `pgfplots`.
5. **System Pipeline Diagram** — Simulation -> Metrics -> Rashomon -> (optional) LLM prose, explicitly marking deterministic vs LLM stage. Feasible in `tikz`.
6. **Canon Equilibrium Diagram** — Simple matrix or sankey-like view showing universally known vs mixed claims after Story B. Data: `canon_after_B.json.world_canon.claim_states`. Feasible in `tikz`.
7. **Wound Transition Diagram** — Baseline-to-full edge presence changes (appeared/disappeared). Data: `goal_evolution_experiment.json.comparisons.wound_topology`. Feasible in `tikz`.

### Thin Sections
1. **Abstract** — Add explicit super-linearity and genre-shift language (currently implied but not named).
2. **Chain Degradation Problem** — Add explicit seed-order protocol table (4 orderings, same 5 seeds) and note one ordering slopes up first->last.
3. **Chain Degradation Problem** — Add explicit failed-intervention narrative (texture/confidence/tension) if those experiments exist.
4. **Experiment 1 Method** — Add the intended-target fallback (`secret_affair_01` -> `secret_embezzle_01`) and why.
5. **System Description** — Add formal metric definitions/weights (currently only names are listed, no equations/weights).
6. **Discussion** — Add concrete claim-level evidence for “information equilibrium” (which claims are universally known).
7. **Future Work** — Add stance-matrix/discretized-goal evolution path, Structural Gradient Ascent, and affordance-model roadmap if these are core to your planned architecture.

## Formatting Improvements

### Tables
- [ ] `Table 1` column spec mismatch: `\begin{tabular}{lccc}` but only 3 columns are used; change to `lcc` (or add intended column). Location: `goal_evolution_paper.tex:129`.
- [ ] Add decimal alignment via `siunitx` (`S` columns) for all numeric tables.
- [ ] Use consistent condition labels (`Depth-0` vs `Depth 0`, `Baseline (depth~2)` vs `Baseline (depth 2)`).
- [ ] Add footnote that means are over **valid arcs only** in conditions where invalid arcs exist (thorne-only).
- [ ] Consider splitting dense per-agent table or rotating to improve readability in two-column layout.

### Typography
- [ ] Switch to a conference template (`aaai`, `acmart`, etc.) instead of `article + geometry` for submission realism. Current header: `goal_evolution_paper.tex:1-4`.
- [ ] Fix overwide utility and grammar math lines (`goal_evolution_paper.tex:95`, `goal_evolution_paper.tex:111`).
- [ ] Reduce manual hyphenation pressure from narrow columns (many underfull boxes) via rephrasing or `microtype` if available.
- [ ] Standardize math/text notation for deltas and percentages throughout narrative and captions.

### Layout
- [ ] Add at least one key figure in first 2 pages (currently figure-less, table-heavy opening).
- [ ] Move dense result interpretation closer to each table/figure for scanability.
- [ ] Break long analytical paragraphs in Sections 6-7 into shorter claim-evidence chunks.

## Minor Issues

### Citation/Reference
- [ ] No missing `\cite{}` keys vs `references.bib` were found, but several related-work items in the prompt are still not actually cited in text (e.g., DOME entry exists but uncited). Source: `references.bib` contains `wang2025dome`; paper text has no `\cite{wang2025dome}`.
- [ ] Add POCL/IPOCL-primary references explicitly if claiming planning lineage beyond high-level narrative-planning surveys.
- [ ] Consider adding/confirming `Awash` and CGR references if these are intended comparison points.
- [ ] Clean up BibTeX entry types/fields for consistency (`journal` used in some `@inproceedings` entries).
- [ ] Ensure all cited “2025” works are final bibliographic forms (venue/proceedings details) before submission.

### Style/Tone
- [ ] Replace “disproved” phrasing for single-chain U-shape with “did not replicate under seed-order controls.”
- [ ] Lowercase inconsistency: “thorne-only” in prose vs “Thorne-only” in table.
- [ ] “Strictly monotonic with agent count” language should be changed to “nonlinear/superlinear collective effect.”
- [ ] Keep claims explicitly scoped to “this system/scenario” in discussion and conclusion.

### Render Check (.jpg)
- [ ] First-page JPG render is legible and structurally correct (title/columns/tables appear), but raster output is soft/low-resolution; use PDF for quality judgments and submission.

## Summary

- Critical issues: 19
- Recommended additions: 19
- Formatting improvements: 12
- Minor issues: 10
- Overall assessment: The core experiments and most reported table numbers are strong and mostly accurate, but several high-visibility claims overstate the data or contradict the corrected chain results. With targeted factual edits, stronger hedging, and a few key visual/table additions, this can become submission-ready.
