# Changelog

All notable changes to the NarrativeField project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Paper2: Section 7.8 "Candidate Pool Contamination" tracing all 9 Diana failures to importance-driven event injection
- Paper2: Section 6 subsections "Constraint Dimensionality" (min_development_beats accounts for 100% of regularization) and "Depth Invariance" (cliff is search-algorithmic, not environment-dependent)
- Paper2: Two new future work items (temporal injection filtering, generalization beyond narrative extraction)

### Fixed
- Paper2: Rewrote VA formula (Eq. 1) to match per-seed arc-level computation (code was correct, formula was wrong)
- Paper2: Regenerated k-sweep Figure 1 VA coordinates from artifact data (VA curve is U-shaped, not endpoint-collapse)
- Paper2: Corrected seed autopsy metric margin (0.017 → 0.098, "mild" → "substantial" misalignment)
- Paper2: Fixed alpha threshold (> 0.5 → ≥ 0.5) and added categorical discontinuity acknowledgment
- Paper2: Corrected Q-metric weighting claim (not 50/50, actually ~60/40 tension-irony/structural)
- Paper2: Added search algorithm description for standalone readers
- Paper2: Completed grammar recap with missing validator constraints (beat count, time span)
- Paper2: Clarified "highest-leverage agent" in abstract (quality vs validity leverage)
- Paper2: Softened generalization claims in discussion and conclusion
- Paper2: Added paper2 Makefile target
- Paper2: Fixed α=0.5 categorical direction in Section 5.1 (commitments adopt evolved values, not revert to default)
- Paper2: Clarified VA averaging convention (valid arcs only, no double-penalty)
- Paper2: Updated contribution list to reflect single-root-cause reframing

### Changed
- Updated README with paper2 reference
- Paper2: Rewrote Discussion 8.4 (pool contamination cross-reference), 8.5 (three fix directions + two-regime reframing), and 8.6 (single diagnostic agent limitation)
- Paper2: Rewrote Conclusion with full mechanistic chain (pool contamination → turning-point anchoring → phase collapse)
- Paper2: Updated abstract to reflect deeper mechanistic story
- Paper2: Added Figure 1 VA annotations at k=0 and k=3, moved legend to top-left
- Paper2: Added noise acknowledgment to Figure 2 caption
- Paper2: Added bridging sentence between Table 1 all-valid rate and Figure 1 VA curve

### Added
- Parameterized extraction grammar support via `GrammarConfig` in `src/engine/narrativefield/extraction/arc_validator.py`, with optional grammar injection in arc search/Rashomon extraction while preserving strict default behavior
- Feasible-volume regularization experiment runner `src/engine/scripts/feasible_volume_sweep.py` plus generated artifacts `src/engine/scripts/output/feasible_volume_sweep.json` and `src/engine/scripts/output/feasible_volume_summary.md`
- Depth-0 vs depth-2 dev-beat sweep runner `src/engine/scripts/depth0_devbeat_sweep.py` plus generated artifacts `src/engine/scripts/output/depth0_devbeat_sweep.json` and `src/engine/scripts/output/depth0_devbeat_summary.md`
- Search candidate pool instrumentation via `src/engine/scripts/search_candidate_logger.py` plus SVG/HTML plot renderer `src/engine/scripts/plot_candidate_scatter.py` and output artifacts under `src/engine/scripts/output/`
- Anchor diversification experiment runner `src/engine/scripts/anchor_diversification_experiment.py` comparing default, diversified, and all-midrange Diana anchor strategies with depth-2 full-extraction impact metrics and summary artifacts
- Proto-keep ablation runner `src/engine/scripts/proto_keep_ablation.py` comparing injection/causal pool modes for all-midrange Diana anchors with verification outputs and mechanism summary artifacts
- Rashomon multi-protagonist extraction module `src/engine/narrativefield/extraction/rashomon.py` with `RashomonArc`, `RashomonSet`, and `extract_rashomon_set` for six-arc per-seed extraction
- Rashomon regression test suite `src/engine/narrativefield/tests/test_rashomon.py` including seed-42 determinism coverage, overlap semantics, turning-point overlap, and roundtrip serialization
- Seed sweep + wound analysis scripts: `src/engine/scripts/sweep_rashomon.py` and `src/engine/scripts/analyze_wounds.py`, with output artifacts written under `src/engine/scripts/output/`
- Phase-2 experiment runner `src/engine/scripts/phase2_experiment.py` implementing per-agent alpha sweeps, Thorne-locked coordinate descent, A3 reference validation with exact k-sweep key reuse, and 2x2+full amplifier/repair factorial analysis
- Engine test coverage for phase-2 experiment utilities in `src/engine/narrativefield/tests/test_phase2_experiment.py` (interpolation, validity-adjusted scoring, and k-sweep lookup-key matching)
- Lore loop chain demo (`scripts/demo_lore_loop.py`) with A→B→C three-story texture compounding (0→9→24→36 facts)
- Pipeline architecture SVG (`assets/pipeline-architecture.svg`) showing five-stage pipeline with lore loop feedback
- Canon divergence SVG (`assets/canon-divergence.svg`) showing fresh vs canon event stream comparison
- Texture compounding SVG (`assets/texture-compounding.svg`) showing three-story texture accumulation
- Lore loop chain summary and Story C artifacts in `examples/`

### Changed
- Arc-search diagnostics docs now clarify that `candidates_evaluated` can remain `0` when `search_arc` returns a valid candidate before fallback candidate enumeration
- Updated checkpoint overview SVG (`assets/checkpoint-overview-2026-02.svg`) with current metrics: 170 engine tests, 68 viz tests, 50/50 arc validity, three-story chain evidence
- Rewrote README with current evidence, visual assets, canon test comparison, and lore loop results

### Previously Added
- Arc-search diagnostics instrumentation (`rule_failure_counts`, `best_candidate_violation_count`, `candidates_evaluated`, `best_candidate_violations`) plus normalization keys for failure-frequency analysis
- Arc-search seed sweep harness `src/engine/scripts/sweep_arc_search.py` for deterministic multi-seed validity/fallback reporting
- Phase 4 lore loop foundation in engine storyteller: structured lore schema (`CanonFact`, `TextureFact`, `SceneLoreUpdates`, `StoryLore`), optional `<lore_updates>` parsing, and per-run `GenerationResult.story_lore`
- Canon texture persistence support via `WorldCanon.texture` + `CanonTexture` with backward-compatible serialization
- Canon-aware lorebook context injection (`<world_memory>`) and storyteller lore tests (`test_lore.py`, `test_lore_extraction.py`, `test_lorebook_canon.py`, `test_narrator_lore.py`)
- Engine demo script `src/engine/scripts/demo_canon_stories.py` that runs Story A/B canon simulations and generates side-by-side live prose (Story B fresh vs canon-loaded) with metadata, comparison summary, and copied `examples/` artifacts
- Phase 3 checkpoint report at `docs/checkpoints/2026-02-phase3-checkpoint.md`
- README updated with WorldCanon pipeline stage, world building section, canon persistence validation, and current test counts (124 + 68 = 192)
- WorldCanon materialized state schema (`schema/canon.py`) with location memory, canonical entity containers, and claim-state snapshots
- Event entity references (`Event.entities`) with additive typed refs for locations, artifacts, factions, institutions, claims, and concepts
- Canon-focused engine test suites: `test_canon.py` (unit coverage) and `test_canon_integration.py` (simulation + persistence coverage)
- Engine claim schema: new `ClaimDefinition` plus `WorldDefinition.claims` and unified `WorldDefinition.all_claims`
- Secret-to-claim compatibility bridge via `SecretDefinition.to_claim()` and world-level claim serialization support
- Claims test suite (`test_claims.py`) covering schema roundtrips, irony/tension behavior, rumor propagation, and seed-42 regression compatibility
- METHODOLOGY.md with appendices (Formulas, Prompts, Data Schemas) for research team onboarding
- Narrative Topology View mode (tension terrain + glowing character arcs + topology markers)
- Topology View visual polish: bolder contours with value labels, summit markers, curated event annotations, clearer character arcs, tension legend, and subtle time gridlines
- Threads View visual polish: tension-responsive scene washes, tension-encoded arc thickness/opacity, smart annotations, interaction links, endpoint markers, and improved event node styling
- February 2026 version checkpoint report at `docs/checkpoints/2026-02-version-checkpoint.md` with completed quality-gate and seed-sweep outcomes
- Checkpoint science artifact at `data/checkpoints/2026-02-checkpoint-science.json` for deterministic 12-seed validation (`1-10,42,51`)
- Checkpoint SVG renderer script `scripts/render_checkpoint_graphic.py` and generated overview graphic `assets/checkpoint-overview-2026-02.svg`
- Audit documentation index at `docs/audits/README.md` classifying authoritative reports versus archived historical material
- Historical audits archived under `docs/audits/archive/` for cleaner top-level audit navigation
- Topology View map viewport: bounded narrative world, cursor-anchored proportional zoom (X+Y), vertical panning, fit-to-world, and a narrative-island ocean/coastline treatment
- Visualization test suite: 49 tests across 9 files (store, renderModel, HitCanvas, loader, layout, terrain, topology layers)
- UX review with fixes: tooltip contrast, marker sizing, scene label overlap, zoom button states
- Audit reports for all subsystems (simulation, metrics, extraction, frontend)
- Storyteller audit report with prose generation quality findings
- Repetition guard module for storyteller prose deduplication
- Tests for repetition guard
- CHANGELOG.md project history
- Documentation updates (README, CLAUDE.md, AGENTS.md files)
- Repository `LICENSE` (MIT), root `.env.example`, and GitHub Actions CI workflow for engine + visualization checks
- Timeline domain regression test for unsorted events (`TimelineBar`)
- Thread layout regression test covering manifest `initial_location` handling
- Root `Makefile` with unified `lint`, `test`, `build`, and `all` task targets
- Shared thematic-axis constants in Python (`schema/thematic_axes.py`) and TypeScript (`types/thematic.ts`)
- Determinism metadata on simulation outputs: `metadata.deterministic_id` and `metadata.truncated`
- Visualization store regression coverage for empty payload handling (0 events / 0 scenes)
- Comprehensive post-fix verification audit report (`docs/audits/AUDIT_VERIFICATION.md`) with full quality-gate outputs and finding-status matrix
- Engine significance scoring test suite (`test_significance.py`) plus seed-42 acceptance assertions for catastrophe/belief coverage and score spread

### Changed
- Story prompts now include optional `<lore_updates>` output instructions plus strict protagonist POV constraints (no non-protagonist interiority)
- Sequential narrator now supports optional pre-run canon injection and commits texture facts to canon with collision-safe keys `{generation_id}__{tf.id}`
- Canon story demo now performs lore seeding from Story A narration, includes fallback caveats when arc search falls back, and reports lore extraction/canonization stats
- Visualization payload loader now supports canon-era `.nf-viz` metadata fallback (`simulation_id` from `deterministic_id`, `timestamp` default), accepts `location_memory` delta kinds, and preserves unknown future delta kinds with warnings instead of hard rejection
- Visualization payload CLI validator now allows belief snapshot keys beyond `secrets` (for claim-state compatibility)
- Simulation tick loop now materializes and updates `WorldCanon` deterministically (location tension staining/decay, visit memory, claim-state snapshots) and emits `LOCATION_MEMORY` deltas for catastrophe/conflict events
- Simulation runner now supports optional `--canon` loading from prior outputs and writes `world_canon` in simulation JSON
- Metrics significance breadth now accounts for event entity refs and supports world-canon delta kind weights (`artifact_state`, `faction_state`, `institution_state`, `location_memory`)
- Metrics pipeline and renderer bundler now parse and pass through optional `world_canon` payloads
- Metrics pipeline now parses optional `claims` payloads and evaluates irony/tension over unified claim space while preserving secret-only compatibility
- Dinner party scenario now includes additive example claims (`claim_thorne_health`, `claim_guild_pressure`) without changing baseline secret-driven outcomes
- Witness generation now supports claim rumor spread from `CHAT`/`SOCIAL_MOVE` belief deltas using per-claim propagation rates
- Storyteller default creative model upgraded to `claude-haiku-4-5-20251001` (while preserving structural model defaults)
- Canon story prose demo now caps invalid arc-search fallback to the top 15 events by significance (instead of all events) to keep narrator prompts within budget
- Metrics pipeline now computes event significance before bundling, enabling real significance-aware OBSERVE filtering
- Scene segmentation now treats `SOCIAL_MOVE` destination as the scene location (instead of source location)
- Bundler now preserves `SOCIAL_MOVE` events so segmentation can enforce location-change boundaries
- Dinner party movement tuning updated: foyer privacy increased, bathroom directly reachable from dining table, and SOCIAL_MOVE penalty reduced for more varied location usage
- Beat repair logic now centralizes TURNING_POINT resolution in monotonic enforcement with deterministic duplicate handling
- Arc search now prefers arcs that include post-peak CONSEQUENCE beats and widens candidate windows when needed
- Visualization canvas layer typing cleaned up (`TensionTerrainLayer`) with explicit canvas image/context handling and no `as any` casts
- Engine dependency ranges are now pinned in `pyproject.toml` with `openai` declared, plus a generated `requirements.lock`
- README "What It Looks Like" now references the checkpoint overview artifact (`assets/checkpoint-overview-2026-02.svg`) and adds a compact checkpoint evidence section
- Science harness now accepts explicit seed lists via `--seeds` while preserving range mode (`--seed-from/--seed-to`)
- Simulation now validates `time_scale > 0` at CLI entry and documents current `time_scale` semantics explicitly
- Tick-loop pacing updates now finalize deltas at a single end-of-tick merge point and avoid cross-agent event attribution
- Catastrophe gates now honor scenario-provided `catastrophe_threshold` and `composure_gate` values
- Dinner party catastrophe tuning rebalanced per-agent pacing/goals so catastrophe ownership is distributed across the cast
- Beat classification now uses spec position boundaries (`0.25`, `0.70`) with monotonic TP enforcement selecting phase beats deterministically
- Arc scoring tension-variance normalization retuned from seed 1-10 median variance for better discriminative power
- Story prompts now include a `previously_established` continuity section, and repetition guard now flags structural/starter repetition patterns
- Timeline domain computation now scans event min/max directly instead of assuming sorted input
- Thread layout now accepts agent manifests and uses scenario `initial_location` values rather than hardcoded `dining_table`
- Payload loader now builds typed arrays via runtime guards instead of unsafe `as unknown as` coercions
- Topology tension terrain field generation now trims low-contribution kernel tails for better scaling at 200+ events
- Story extraction API CORS origins now read from `CORS_ORIGINS` (default `http://localhost:5173`) with explicit dev-only note
- Engine audit/test helper files cleaned up to satisfy `ruff check .` without suppressing core checks
- Metrics pipeline now warns on duplicate event IDs (deduplicated) and dangling causal-link references
- Event bundler simplified by removing dead SOCIAL_MOVE squash/attach implementation paths
- Storyteller lorebook character entries now include per-character voice notes injected into prompts
- Grok structural fallback now logs explicit fallback messages and records model-used metadata in API responses
- Package versions bumped to `0.1.0` for both engine and visualization
- README quickstart now leads with visualization exploration before API-key-required story generation
- Audit docs now use `$PROJECT_ROOT` instead of hardcoded local filesystem paths
- Visualization dependency ranges are now pinned to exact versions (no caret ranges) for reproducible installs
- Metrics significance computation upgraded to v0 "reachable future change" scoring (delta magnitude + causal centrality + novelty + breadth) and moved into dedicated `metrics/significance.py`

### Removed
- Cleaned up 13 stale branches (local + remote) that were merged or squash-merged to main

### Fixed
- Fixed arc-search TURNING_POINT deduplication to demote post-TP duplicates to CONSEQUENCE and enforce post-TP phase-3 sweep ordering
- Fixed beat-classifier setup repair to force the first beat to SETUP by position (not by presence anywhere in sequence)
- Fixed arc-search monotonic repair to route post-TP promotions through phase-aware beat selection helpers
- Fixed arc-search downsampling to enforce post-peak non-essential coverage when available, improving consequence candidate retention
- Fixed demo fallback event ranking tie-breakers to prefer protagonist-involving and higher-drama event types on equal significance
- Fixed storyteller `/api/generate-prose` world-data wiring so both inline and loaded-payload paths now build Lorebook-compatible context (world definition, agents, secrets) instead of falling back to generic prompts
- Fixed `/api/generate-prose` config override hardening by enforcing an explicit allowlist with type/range validation and warning logs for rejected keys
- Fixed unbounded prose result retention by capping `_prose_results` to 50 entries with oldest-first eviction
- Fixed incorrect scene boundary behavior for back-to-back `SOCIAL_MOVE` events to different destinations
- Added regression coverage for social-move destination segmentation and significance-driven OBSERVE bundling behavior
- Fixed payload loader validation gap for `metrics.irony_collapse` shape
- Fixed visualization lint failures (`eslint`) including hook dependency warnings and unused symbols
- Fixed stale catastrophe config wiring by passing world-defined thresholds into pacing catastrophe checks
- Fixed repetition guard blind spots where structural repeated templates were previously undetected
- Fixed `useVisibleEvents` churn by memoizing visible-event derivation with shallow store selection
- Fixed checkpoint writes to use atomic temp-file replacement and to raise on save failures
- Fixed visualization lint warning from non-component exports in `TimelineBar.tsx`
- Fixed `DeltaKind.PACING` state application to clamp non-timer pacing values into `[0, 1]`
- Fixed extraction `ArcValidation` immutability by switching violations from mutable lists to tuples
- Fixed payload validation to sanity-check thematic shift keys/values against canonical axes
- Fixed arc-search fallback selection to prefer post-peak consequence candidates when no fully valid arc is found

## [2026-02-09]

### Added
- Per-scene outcome tracking and run artifact reporting
- Retry with exponential backoff to LLM gateway
- Complete storyteller prose generation pipeline
- Integration improvements for extraction, simulation, and visualization
- Audit scripts and region search test

### Fixed
- Mid-word scene boundary truncation
- Timestamped story output filenames for iteration history preservation
- Story output now saved to scripts/output/
- Extended thinking params corrected; switched to Haiku 4.5

## [2026-02-08]

### Added
- Micro-event bundling for cleaner nf-viz streams (PR #2, #3)
- Auto search region selection for valid arcs in extraction pipeline
- Dinner party affordances and catastrophe tuning (PR #4)
- Determinism and time scale support for simulation engine (PR #1)

### Fixed
- Social moves now attached using destination metadata in metrics pipeline
- Renderer metadata counts updated correctly after bundling

### Added (Tests)
- Test coverage for social-move attachment and forward causal rewire prevention

## [2026-02-07]

### Added
- Initial commit: NarrativeField MVP
  - Event-sourced simulation engine with dinner party scenario
  - Metrics pipeline (tension, irony, thematic shift computation)
  - Extraction pipeline with arc grammar and beat sheet generation
  - Visualization renderer integration
  - Storyteller prose generation via Claude API
  - Complete spec suite in specs/ directory
