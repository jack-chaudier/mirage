# NarrativeField Post-Fix Independent Verification Audit

- Date: 2026-02-11
- Auditor: Codex (independent trust-but-verify pass)
- Branch: `codex/threads-topology-polish`
- HEAD: `45a47b2`
- Environment: macOS, Python 3.14.2, Node/Vite local toolchain
- Prior references:
  - `docs/audits/AUDIT_REPORT.md` (baseline)
  - `docs/audits/AUDIT_REPORT_FOLLOWUP.md` (59 findings)
  - `docs/audits/AUDIT_VERIFICATION.md` (self-reported fix verification)

## Executive Summary

- Health rating: **RISKY**
- Follow-up fix success rate: **38/59 VERIFIED FIXED**
  - 13/59 PARTIALLY FIXED
  - 6/59 NOT FIXED
  - 2/59 REGRESSION
- Regressions found: **3 total**
  - 1 CRITICAL: creative model ID configured to an unavailable Anthropic model, blocking prose generation
  - 1 MEDIUM: extraction tension variance normalization moved in the wrong direction (0.10 -> 0.013)
  - 1 MEDIUM: `classify_beats()` flat-tension edge now fails to guarantee a turning point
- New issues found (post-fix, not explicitly tracked in follow-up): **3**
  - 1 MEDIUM: SOCIAL_MOVE destination boundary still misses same-tick edge cases (seed 42 has 3 mismatches)
  - 2 LOW: audit harness drift/staleness in extraction and storyteller checks
- Model availability status: **BLOCKED**
  - Configured model `claude-sonnet-4-5-20250514` returns Anthropic 404 NotFound.
  - Verified callable alternatives: `claude-sonnet-4-5-20250929`, `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001`.
- Portfolio readiness: **NOT READY**

## 1) Working Tree Hygiene (Part 0)

### Pre-audit state (required check)

Commands run:
- `git status`
- `git log --oneline -10`

Result at start:
- Only known uncommitted files were present:
  - `docs/audits/AUDIT_REPORT.md` (modified)
  - `docs/audits/AUDIT_REPORT_2026-02-08.md` (untracked)
  - `docs/audits/AUDIT_REPORT_FOLLOWUP.md` (untracked)
- No other pre-existing unstaged code changes were found.

### Test-file churn in fix commits (`HEAD~7..HEAD`)

- New test files:
  - `src/engine/narrativefield/tests/test_segmentation.py`
  - `src/engine/narrativefield/tests/test_tick_loop_updates.py`
  - `src/visualization/src/components/TimelineBar.test.ts`
- Modified test files: 10 additional test files modified.
- Removed test files: none.
- Coverage caveat: `src/engine/narrativefield/tests/test_event_bundler.py` had heavy rewrites (355-line diff, net deletion-heavy). Some old assertions were removed while new preservation-oriented behavior was added.

## 2) Fix Verification Table (all 59 FU findings)

| Finding ID | Claimed Status | Verified Status | Evidence | Notes |
|---|---|---|---|---|
| FU-ADV-MED-01 | FIXED | VERIFIED FIXED | `src/engine/narrativefield/simulation/run.py:116-117`; CLI now raises explicit ValueError for <=0 time_scale. | Helpful error message now explicit. |
| FU-ADV-MED-02 | FIXED | PARTIALLY FIXED | `src/engine/narrativefield/simulation/tick_loop.py:39-41`; time_scale still only scales sim_time/termination, not decision dynamics. | Behavioral dynamics unchanged; mostly semantic/UX clarification. |
| FU-ADV-HIGH-01 | FIXED | VERIFIED FIXED | Seeds 1-10 run: foyer 10/10, bathroom 10/10, dining_table still primary 63.5%-81.0%. | Dead-space resolved; watch balcony variance (seed9=25.5%). |
| FU-ADV-MED-03 | FIXED | VERIFIED FIXED | `src/engine/narrativefield/simulation/tick_loop.py:102-106` now uses world catastrophe_threshold/composure_gate. | Config now live in catastrophe checks. |
| FU-ADV-MED-04 | FIXED | PARTIALLY FIXED | Catastrophe totals seeds 1-10: Victor 13, Thorne 10, Marcus 9, Lydia 4, Diana 3, Elena 2. | No 50% monopolist now, but Victor still highest trigger. |
| FU-ADV-LOW-05 | FIXED | VERIFIED FIXED | `src/engine/narrativefield/metrics/pipeline.py:164-179` dedupes duplicate IDs + warning; `test_metrics_pipeline.py:208+`. | Duplicate handling no longer silent overwrite. |
| FU-ADV-LOW-06 | FIXED | VERIFIED FIXED | `src/engine/narrativefield/integration/event_bundler.py:294-296` warns on empty list. | Degenerate input now flagged. |
| FU-DET-LOW-01 | FIXED | VERIFIED FIXED | `src/engine/narrativefield/simulation/tick_loop.py:998-1000` + `run.py:163` truncated metadata exposed. | Downstream now sees truncation state. |
| FU-BUN-HIGH-01 | FIXED | PARTIALLY FIXED | `src/engine/narrativefield/metrics/segmentation.py:62-69`; destination used. But seed42 still 3 destination mismatches (same-tick grouping). | Destination fix exists but same-tick rule can still mask intended boundary. |
| FU-BUN-MED-01 | FIXED | PARTIALLY FIXED | Pipeline still bundles before segmentation (`pipeline.py:245-251`), mitigated by preserving SOCIAL_MOVE in bundler (`event_bundler.py:305-307`). | Core ordering risk remains by design; mitigations help. |
| FU-BUN-MED-02 | FIXED | PARTIALLY FIXED | `test_segmentation.py:77-106` covers destination split, but seed42 has same-tick edge mismatches. | Improved but not fully robust in all temporal edge cases. |
| FU-BUN-LOW-01 | FIXED | PARTIALLY FIXED | SOCIAL_MOVE squashing removed from behavior; legacy config fields remain no-op (`event_bundler.py:27-31`). | Legacy dead knobs retained for compatibility. |
| FU-CON-HIGH-01 | FIXED | VERIFIED FIXED | Significance assignment present (`pipeline.py:127-157`); seed42 significance non-zero across event types. | Significance no longer perpetually zero. |
| FU-CON-HIGH-02 | FIXED | VERIFIED FIXED | Significance computed before bundling (`pipeline.py:242-247`); comment now accurate. | Comment/implementation now aligned. |
| FU-CON-HIGH-03 | FIXED | VERIFIED FIXED | Loader validates irony_collapse.detected/drop (`src/visualization/src/data/loader.ts:221-231`); malformed payload rejected. | Bad irony payloads are caught. |
| FU-CON-MED-01 | FIXED | NOT FIXED | `rg irony_collapse` in viz shows parse/store only; no rendering/interaction consumption. | Contract field remains unused by UX. |
| FU-CON-MED-02 | FIXED | NOT FIXED | `beliefSnapshots` stored (`narrativeFieldStore.ts:255`) but unused by UI components. | Loaded but still dead in rendering flow. |
| FU-CON-MED-03 | FIXED | VERIFIED FIXED | Frontend scene_type now `SceneType | string` (`src/visualization/src/types/scenes.ts:26`). | String/enum mismatch risk materially reduced. |
| FU-CON-LOW-01 | FIXED | NOT FIXED | Scene IDs still index-based (`src/engine/narrativefield/metrics/segmentation.py:146`). | ID stability issue still present across config changes. |
| FU-CON-LOW-02 | FIXED | VERIFIED FIXED | `MetricsPipelineOutput.scenes` typed `list[Scene]` (`pipeline.py:87-90`). | Type debt resolved. |
| FU-CON-LOW-03 | FIXED | PARTIALLY FIXED | Unknown thematic axes now warned in loader (`loader.ts:256-259`) but not hard-validated. | Forward-compatible warning is weaker than strict validation. |
| FU-VIZ-HIGH-01 | FIXED | PARTIALLY FIXED | `as any` removed; one `as unknown as` remains (`TensionTerrainLayer.ts:222`). | Major safety improvement, minor unsafe cast remains. |
| FU-VIZ-HIGH-02 | FIXED | VERIFIED FIXED | `npm run lint` exits 0; 4 `eslint-disable` comments only in `validatePayload.ts`. | Lint gate is now enforceable. |
| FU-VIZ-MED-01 | FIXED | VERIFIED FIXED | Terrain layer now cached/incremental (`TensionTerrainLayer.ts` cache + recompute pipeline). Build handles 79 modules. | Performance risk reduced; still monitor at higher E. |
| FU-VIZ-MED-02 | FIXED | VERIFIED FIXED | Domain now computed unsorted via helper + test (`timelineDomain.ts`, `TimelineBar.test.ts`). | No pre-sort assumption now. |
| FU-VIZ-MED-03 | FIXED | VERIFIED FIXED | threadLayout uses manifest initial_location (`threadLayout.ts:46,103` + test). | Scenario manifest now affects layout init. |
| FU-VIZ-MED-04 | FIXED | VERIFIED FIXED | No `as unknown as` casts in `loader.ts`; stronger runtime validation present. | Typed validation improved in loader. |
| FU-VIZ-MED-05 | FIXED | VERIFIED FIXED | React-hook dependency lint now clean with `eslint --max-warnings=0`. | Hook deps no longer lint-failing. |
| FU-VIZ-MED-06 | FIXED | VERIFIED FIXED | `useVisibleEvents` memoized selector (`narrativeFieldStore.ts:526-553`). | Selector churn reduced. |
| FU-VIZ-LOW-01 | FIXED | VERIFIED FIXED | Lint clean; unused `domainSpan` removed from layer implementation. | Resolved via cleanup + lint gate. |
| FU-VIZ-LOW-02 | FIXED | VERIFIED FIXED | Empty-state coverage added in store tests (`narrativeFieldStore.test.ts` loadEventLog empty case). | Targeted empty-state test exists. |
| FU-EXT-HIGH-01 | FIXED | VERIFIED FIXED | TP dedup centralized in `_enforce_monotonic_beats` (`arc_search.py:534+`), test added (`test_extraction.py:226+`). | Audit harness text is partly stale on this point. |
| FU-EXT-HIGH-02 | FIXED | PARTIALLY FIXED | Phase promotion now tension-aware (`arc_search.py:206-227`) but flat-tension edge still fails in `audit_extraction.py`. | Better than baseline but edge behavior still rough. |
| FU-EXT-MED-01 | FIXED | PARTIALLY FIXED | Classifier now uses 0.25/0.70 windows (`beat_classifier.py:58-72`) but still diverges from full spec behavior. | Improved, not spec-complete. |
| FU-EXT-MED-02 | FIXED | REGRESSION | `TENSION_VARIANCE_NORMALIZATION` changed from 0.10 to 0.013 (`arc_scorer.py:6`), increasing score saturation. | Applied opposite direction of intended fix request. |
| FU-EXT-LOW-01 | FIXED | VERIFIED FIXED | ArcValidation violations now immutable tuple (`extraction/types.py`). | Immutability issue closed. |
| FU-EXT-LOW-02 | FIXED | NOT FIXED | `classify_beats(events)` still has no scene-context parameter (`beat_classifier.py:6-15`). | Spec/API parity still missing. |
| FU-STO-CRIT-01 | FIXED | REGRESSION | Creative model set to `claude-sonnet-4-5-20250514` (`llm/config.py:10`); live probe returns 404 NotFoundError. | Most important regression: creative generation broken at runtime. |
| FU-STO-HIGH-01 | FIXED | VERIFIED FIXED | Seed51 arc includes post-peak consequence beats (indices 18-19) and validation passes. | Structural fix exists even though prose regeneration failed. |
| FU-STO-MED-01 | FIXED | PARTIALLY FIXED | Guard enhanced (`repetition_guard.py:250+`), but seed_42_story flags only 1 pattern and removes 0 chars. | Improved heuristics, limited practical catch rate on golden text. |
| FU-STO-MED-02 | FIXED | VERIFIED FIXED | Prompt includes rolling summary + last paragraph continuity (`prompts.py:95-109`); narrator maintains summary (`narrator.py:507-529`). | Continuity context path materially improved. |
| FU-STO-LOW-01 | FIXED | NOT FIXED | Existing golden stories unchanged (e.g., `examples/seed_42_meta.json` timestamp 2026-02-09) and still show old voice issues. | Requires regeneration to demonstrate improvement. |
| FU-STO-LOW-02 | FIXED | VERIFIED FIXED | Grok fallback now logged + surfaced in metadata (`gateway.py:318-321`, `response_metadata.fallback_used`). | Fallback no longer silent. |
| FU-ERR-MED-01 | FIXED | VERIFIED FIXED | Atomic write + re-raise implemented (`checkpoint.py:17-33`, `checkpoint.py:63-87`). | Atomicity + error visibility both improved. |
| FU-ERR-LOW-01 | FIXED | VERIFIED FIXED | Pacing delta clamping present in tick-loop apply_delta path (`tick_loop.py` PACING branch). | Clamp behavior restored. |
| FU-ERR-LOW-02 | FIXED | VERIFIED FIXED | Pipeline now rejects degenerate/invalid metric inputs (`pipeline.py:65-83`, sanitization warnings). | Degenerate input handling strengthened. |
| FU-DEP-HIGH-01 | FIXED | VERIFIED FIXED | Python dependencies now bounded in `src/engine/pyproject.toml`. | Bounded ranges improve reproducibility/security posture. |
| FU-DEP-MED-01 | FIXED | VERIFIED FIXED | `openai>=2.17,<3.0` declared in `pyproject.toml`. | Runtime dependency declared. |
| FU-DEP-MED-02 | FIXED | VERIFIED FIXED | `ruff check .` returns clean; 82-error state resolved. | Lint debt cleared. |
| FU-DEP-LOW-01 | PARTIAL | PARTIALLY FIXED | Exact Node versions set, but `npm audit` still reports 3 moderate dev vulns (vite/tsx/esbuild chain). | Semver-major security upgrades still deferred. |
| FU-POR-HIGH-01 | FIXED | NOT FIXED | No committed screenshots/videos; only placeholder `assets/README.md`. | Portfolio still lacks visual proof artifacts. |
| FU-POR-MED-01 | FIXED | VERIFIED FIXED | `LICENSE` present at repo root. | Resolved. |
| FU-POR-MED-02 | FIXED | VERIFIED FIXED | CI workflow exists (`.github/workflows/ci.yml`) and local `make all` passes. | Workflow exists; no hosted run inspected here. |
| FU-POR-MED-03 | FIXED | VERIFIED FIXED | `.env.example` includes Anthropic + xAI keys. | Resolved. |
| FU-POR-LOW-01 | PARTIAL | PARTIALLY FIXED | 4 local-path refs remain in audit docs (`AUDIT_REPORT_FOLLOWUP.md`, `AUDIT_VERIFICATION.md`). | Self-reported partial is accurate. |
| FU-POR-LOW-02 | FIXED | VERIFIED FIXED | Versions bumped to `0.1.0` in engine + visualization manifests. | Resolved. |
| FU-POR-LOW-03 | FIXED | VERIFIED FIXED | `Makefile` with `all` target added and working. | Resolved. |
| FU-POR-LOW-04 | FIXED | VERIFIED FIXED | No TODO/FIXME/HACK/XXX left in `src/`; only README TODO comment remains. | Production TODO removed. |
| FU-POR-LOW-05 | FIXED | VERIFIED FIXED | README quickstart now starts with visualization path before LLM generation. | Ordering now portfolio-friendly. |
### Supplemental carry-over IDs explicitly re-checked

(These were called out in your instructions and/or remain active in audit harnesses.)

| Finding ID | Claimed Status | Verified Status | Evidence | Notes |
|---|---|---|---|---|
| CON-MED-01 | FIXED | PARTIALLY FIXED | `src/engine/narrativefield/metrics/pipeline.py:245-251`; bundling still before segmentation; SOCIAL_MOVE preservation mitigates loss. | Architectural ordering concern remains; behavior improved. |
| CON-MED-02 | FIXED | PARTIALLY FIXED | `src/engine/narrativefield/metrics/segmentation.py:239-248`; destination-aware checks added but same-tick grouping can mask destination boundaries. | Better than baseline, not fully resolved in all edge cases. |
| FU-EXT-MED-01 (priority contradiction) | FIXED | PARTIALLY FIXED | `src/engine/narrativefield/extraction/beat_classifier.py:58-72`; `audit_extraction.py` still flags divergence. | Threshold window improved, full spec parity still missing. |
| MET-MED-01 | FIXED | NOT FIXED | `src/engine/narrativefield/metrics/tension.py` + `audit_metrics.py` finding #1 (goal_frustration heuristic divergence). | Audit still fails this check. |
| MET-MED-02 | FIXED | NOT FIXED | `src/engine/narrativefield/metrics/tension.py` + `audit_metrics.py` finding #2 (resource_scarcity proxy divergence). | Audit still fails this check. |

## 3) Regression Detection Results (Part 2)

### 3a) Full test suites

Commands:
- `cd src/engine && pytest -q -v`
- `cd src/visualization && npm test -- --reporter=verbose`

Results:
- Engine: **84 passed** (18 test files)
- Visualization: **62 passed** (11 test files)

Expected delta checks:
- Engine pre-fix 80 -> now 84: confirmed.
- Viz pre-fix 59 -> now 62: confirmed.
- New viz test file: `src/visualization/src/components/TimelineBar.test.ts` confirmed.
- No test files removed.
- Build module count: **79 modules transformed** (3 new modules since pre-fix):
  - `src/visualization/src/components/TimelineBar.test.ts`
  - `src/visualization/src/components/timelineDomain.ts`
  - `src/visualization/src/types/thematic.ts`

### 3b) Audit scripts

Commands:
- `cd src/engine && python audit_sim.py`
- `cd src/engine && python audit_metrics.py`
- `cd src/engine && python audit_extraction.py`
- `cd src/engine && python audit_storyteller.py`

Results:
- `audit_sim.py`: exits 0; 0 invariant violations; same 2 spec deviations called out.
- `audit_metrics.py`: **44 pass / 8 fail** (3 MEDIUM, 5 LOW)
- `audit_extraction.py`: **49 pass / 1 fail / 11 findings**
- `audit_storyteller.py`: **94 pass / 0 fail / 1 MEDIUM finding** (`creative_model unexpected`)

Required root-cause callouts:

1. `audit_metrics.py` 7 -> 8 failures
- 8th failure is LOW irony check: "Relevant unknown score follows spec exactly".
- Root cause: the check exists as an explicit audit fail condition tied to text-vs-code ambiguity in spec language, not a runtime crash/regression.
- Classification: **new/stricter audit interpretation (coverage/accounting delta), not a core behavioral regression**.

2. `audit_extraction.py` 0 -> 1 failure (`Flat tension no TURNING_POINT`)
- Root cause: `classify_beats()` no longer creates a fallback TURNING_POINT in `repair_beats` (logic moved to `_enforce_monotonic_beats` in arc search path).
- Evidence:
  - Current: `src/engine/narrativefield/extraction/beat_classifier.py:84-86` (TP creation removed from repair)
  - Pre-fix reference (before fix commits): TP fallback existed in repair pass.
- Classification: **behavior regression in `classify_beats()` API edge case** (while `arc_search` path may still produce valid arcs).

### 3c) Determinism (critical)

Command pattern:
- `PYTHONHASHSEED=<A|B> python -m narrativefield.simulation.run --scenario dinner_party --seed <seed> --output <path>`

Seeds verified: 42, 51, 9.

Outcome:
- Normalized event streams were identical across hash seeds for all 3 seeds.
- Determinism status: **PASS**.

### 3d) Build + lint

Commands:
- `cd src/engine && ruff check .`
- `cd src/visualization && npm run lint && npm run build`

Outcome:
- `ruff`: pass
- `eslint`: pass
- `vite build`: pass (`79 modules transformed`)

### 3e) Contract stability

Fresh payload flow:
- Simulate seed 42 -> run metrics pipeline -> bundle renderer payload
- Validate with `cd src/visualization && npx tsx src/data/validatePayload.ts <payload>`

Outcome:
- Validation returned `ok: true` on freshly generated payload.
- Payload stats (seed 42, event_limit=200):
  - raw events: 200
  - bundled events: 154
  - scenes: 26
  - bundler stats: `observes_attached=46`, `moves_squashed=0`, `moves_attached=0`

### 3f) Behavioral regression check: location rebalance

Seeds 1-10 independent run:
- foyer appears in 10/10 seeds: **yes**
- bathroom appears in 10/10 seeds: **yes**
- dining_table share range: **63.5% to 81.0%** (still primary)
- balcony range: **8.0% to 25.5%** (seed 9 outlier)
- kitchen: seed 10 has **0%**

Catastrophes:
- per-seed range: **2 to 8**
- per-agent total over seeds 1-10:
  - victor 13, thorne 10, marcus 9, lydia 4, diana 3, elena 2

Assessment:
- Dead-space fix succeeded.
- Victor dominance improved vs prior 50%, but he remains top trigger.

### 3g) Bundler/segmentation coupling

Seed 42 (event_limit=200):
- raw SOCIAL_MOVE: 66
- post-bundle SOCIAL_MOVE: 66 (preserved)
- scene count: 26
- scenes beginning with SOCIAL_MOVE: 9
- destination mismatch cases: 3 (`scene_007`, `scene_016`, `scene_024`)

Assessment:
- Destination-based logic exists and tests were added.
- Same-tick grouping rule still allows destination/location mismatches in live runs.

## 4) New Issues Discovered (Part 3)

### 4a) Significance computation correctness

- Significance is now computed and non-trivial.
- Seed 42 event-type means (bundled):
  - observe 0.2285
  - chat 0.2022
  - reveal 0.5070
  - conflict 0.4258
  - catastrophe 0.5902
  - social_move 0.2260
- OBSERVE filtering is active (`observes_attached=46`), but net bundling compression remains lighter than pre-fix due social-move preservation and live significance.

### 4b) Beat threshold + scoring changes

- Beat window now uses 0.25/0.70 (improved), but extraction still diverges from full spec behavior.
- Tension variance normalization changed from 0.10 to 0.013; many arcs still saturate near 1.0.
- This is opposite the direction requested in follow-up prompt and is treated as a regression.

### 4c) Repetition guard effectiveness

- Enhanced guard logic exists (sentence starters + structural templates).
- Running guard on `examples/seed_42_story.txt` found only one repeated starter (`he was`, count=4), removed 0 chars.
- Improvement exists but remains limited for broader semantic repetition patterns.

### 4d) Model availability and prose generation (most important)

- Configured creative model: `claude-sonnet-4-5-20250514`
- Live Anthropic probe result: **404 not found**
- Official current Anthropic model list uses Sonnet 4.5 ID `claude-sonnet-4-5-20250929`.
- Local live probes confirm callable alternatives:
  - `claude-sonnet-4-5-20250929` -> OK
  - `claude-sonnet-4-20250514` -> OK
  - `claude-haiku-4-5-20251001` -> OK

Source references:
- https://docs.anthropic.com/en/docs/models-overview
- https://docs.anthropic.com/en/docs/about-claude/models/migrating-to-claude-4

### 4e) CI correctness

- `.github/workflows/ci.yml` exists with Python 3.12 and Node 22 jobs.
- Local equivalent gate via `make all` passed (ruff, pytest, vitest, eslint, build).
- No secrets are required for those CI checks.

### 4f) Makefile correctness

- `make all` executed successfully end-to-end.

## 5) Portfolio Readiness Scorecard (Part 4)

| Criterion | Grade | Evidence |
|---|---|---|
| README explains what/why/how in <2 min | PARTIAL | Good structure, but includes outdated model narrative and screenshot TODO. |
| Zero-dep viz demo works on clone | PARTIAL | Viz works without API keys after `npm install`; no prebuilt zero-install demo artifact. |
| Full pipeline demo works with API keys | FAIL | Creative model ID in config is invalid (404). |
| Creative model ID is valid and callable | FAIL | `claude-sonnet-4-5-20250514` returns Anthropic NotFound 404. |
| Grok routing unchanged (gateway still uses xAI) | PASS | Structural tier still routes to xAI (`grok-4-1-fast` with fallback). |
| Screenshots/recordings exist | FAIL | No committed screenshots/videos; only `assets/README.md` guidance. |
| Architecture doc readable in 10 min | PASS | `docs/ARCHITECTURE.md` is structured and scan-friendly. |
| All tests pass (84 engine + 62 viz) | PASS | Verified locally. |
| Both linters pass (ruff + eslint exit 0) | PASS | Verified locally. |
| Build succeeds | PASS | `npm run build` succeeds (79 modules). |
| No hardcoded local paths | PARTIAL | 4 local-path refs remain in audit docs. |
| No leaked credentials | PASS | Placeholder-only key references; no real secrets found. |
| LICENSE file present | PASS | `LICENSE` exists. |
| CI pipeline present and would pass | PASS | Workflow exists; local gate passes. |
| .env.example present with both API keys | PASS | Present with Anthropic + xAI placeholders. |
| Version > 0.0.0 | PASS | Engine + viz are `0.1.0`. |
| Commit history clean | PASS | Cohesive fix-commit sequence with scoped messages. |
| No uncommitted code changes | PASS | Only audit-document deltas present before this post-fix report. |
| Determinism proven post-fix | PASS | Seeds 42/51/9 deterministic across PYTHONHASHSEED values. |
| Location variety (foyer+bathroom used) | PASS | Both appear in 10/10 seeds. |
| Significance computed (not zero) | PASS | Significance populated with meaningful spread by type. |
| Scene boundaries match location changes | PARTIAL | Destination fix exists; 3 same-tick mismatch cases in seed 42. |
| Golden stories have resolution (seed 51) | FAIL | Existing `examples/seed_51_story.txt` still ends at peak; stories not regenerated. |
| $X/story cost documented | PASS | README documents per-story cost and run table. |
| No embarrassing debug artifacts | PARTIAL | README screenshot TODO + validator console logs + residual local paths in audit docs. |

## 6) Recommended Next Steps (priority order)

1. **Fix creative model ID immediately (blocker).**
   - Update `src/engine/narrativefield/llm/config.py` creative model from `claude-sonnet-4-5-20250514` to a valid ID (currently `claude-sonnet-4-5-20250929`).
   - Re-run a minimal live probe before any further portfolio claims.

2. **Regenerate golden stories after model fix.**
   - Recreate `examples/seed_42_story.txt`, `examples/seed_51_story.txt`, and metadata.
   - Confirm seed 51 now includes visible post-peak narrative resolution.

3. **Address extraction regressions.**
   - Restore a TP guarantee for `classify_beats()` flat-tension edge (or explicitly narrow API contract and update audit harness/spec links).
   - Revisit `TENSION_VARIANCE_NORMALIZATION` so the score no longer saturates near 1.0.

4. **Close segmentation edge case.**
   - Resolve same-tick SOCIAL_MOVE destination mismatch behavior so scene location assignment is consistent with movement destination semantics.

5. **Complete portfolio assets.**
   - Add real screenshots/recordings under `assets/` and remove README TODO marker.

6. **Finish documentation polish.**
   - Remove remaining hardcoded local paths in audit docs.
   - Align README model narrative with actual configured model strategy.

## 7) Appendix: Commands Run + Raw Highlights

### Working tree and history

```bash
cd ~/lorien
git status
git log --oneline -10
```

Highlights:
- Pre-audit uncommitted state matched expected known files only.

### Core verification and quality gates

```bash
cd src/engine && pytest -q -v
cd src/visualization && npm test -- --reporter=verbose
cd src/engine && python audit_sim.py
cd src/engine && python audit_metrics.py
cd src/engine && python audit_extraction.py
cd src/engine && python audit_storyteller.py
cd src/engine && ruff check .
cd src/visualization && npm run lint && npm run build
cd ~/lorien && make all
```

Highlights:
- Engine tests: 84 passed
- Viz tests: 62 passed / 11 files
- `audit_metrics.py`: 44 pass / 8 fail
- `audit_extraction.py`: 49 pass / 1 fail
- Lint/build gates pass

### Determinism and contract checks

```bash
PYTHONHASHSEED=12345 python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output run_a.json
PYTHONHASHSEED=67890 python -m narrativefield.simulation.run --scenario dinner_party --seed 42 --output run_b.json
# repeated for seeds 51 and 9

# fresh payload validation
npx tsx src/data/validatePayload.ts <fresh_payload.nf-viz.json>
```

Highlights:
- Determinism: identical normalized streams across hash seeds for 42/51/9
- Fresh payload validates successfully (`ok: true`)

### Critical model checks

```bash
# configured model probe
python - <<'PY'
# anthropic probe for claude-sonnet-4-5-20250514
PY

# candidate valid model probes
python - <<'PY'
# probes for claude-sonnet-4-5-20250929, claude-sonnet-4-20250514, claude-haiku-4-5-20251001
PY
```

Highlights:
- Configured model returns 404 not found.
- Alternative current IDs above return successful responses.

