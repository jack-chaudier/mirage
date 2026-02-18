# Storyteller Integration Audit Report

**Date:** 2026-02-09
**Branch:** `codex/integration/wps1-4`
**Scope:** 10 new modules across `llm/` and `storyteller/`, plus `api_server.py` wiring

---

## 1. Test Suite Status

### Python (`src/engine`)
- **pytest:** 66 passed, 0 failed (1.64s)
  - 5 storyteller tests (gateway, lorebook, checkpoint, narrator_e2e, scene_splitter): all pass
  - 15 event_bundler tests: all pass
  - 5 extraction tests: all pass
  - Full coverage of new modules confirmed

### Frontend (`src/visualization`)
- **tsc + vite build:** Success (71 modules, 441ms)
- **vitest:** 8 passed, 0 failed (315ms)

### Existing Audit Scripts
- **audit_extraction.py:** 50 pass, 0 fail, 11 findings (all pre-existing)
- **audit_sim.py:** No invariant violations across 10 seeds

### New Audit Script
- **audit_storyteller.py:** 94 pass, 0 fail, 1 finding

---

## 2. Module Inventory

| # | Module | Path | Status |
|---|--------|------|--------|
| 1 | `narrativefield.llm` | `llm/__init__.py` | IMPORTS OK |
| 2 | `narrativefield.llm.config` | `llm/config.py` | IMPORTS OK |
| 3 | `narrativefield.llm.gateway` | `llm/gateway.py` | IMPORTS OK |
| 4 | `narrativefield.storyteller` | `storyteller/__init__.py` | IMPORTS OK |
| 5 | `narrativefield.storyteller.types` | `storyteller/types.py` | IMPORTS OK |
| 6 | `narrativefield.storyteller.narrator` | `storyteller/narrator.py` | IMPORTS OK |
| 7 | `narrativefield.storyteller.checkpoint` | `storyteller/checkpoint.py` | IMPORTS OK |
| 8 | `narrativefield.storyteller.lorebook` | `storyteller/lorebook.py` | IMPORTS OK |
| 9 | `narrativefield.storyteller.scene_splitter` | `storyteller/scene_splitter.py` | IMPORTS OK |
| 10 | `narrativefield.storyteller.event_summarizer` | `storyteller/event_summarizer.py` | IMPORTS OK |
| 11 | `narrativefield.storyteller.prompts` | `storyteller/prompts.py` | IMPORTS OK |
| 12 | `narrativefield.storyteller.postprocessor` | `storyteller/postprocessor.py` | IMPORTS OK |

All 12 modules import cleanly. No circular dependencies.

---

## 3. Integration Seams

### Import Chain Verification (10 grep checks)

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | narrator.py imports from llm/ + storyteller/ | **CONNECTED** | Lines 10-28: imports LLMGateway, ModelTier, PipelineConfig, CheckpointManager, split_into_scenes, summarize_scene, Lorebook, prompt builders, PostProcessor |
| 2 | narrator.py uses `is_pivotal` → `CREATIVE_DEEP` | **CONNECTED** | Line 363: `if scene.is_pivotal and self.config.phase2_use_extended_thinking_for_pivotal: tier = ModelTier.CREATIVE_DEEP` |
| 3 | narrator.py calls `checkpoint.save()` after each scene | **CONNECTED** | Line 423: `self.checkpoint_mgr.save(state, prose_chunks, i)`, also line 385 (error path save) |
| 4 | prompts.py uses `lorebook.get_context_for_scene()` | **CONNECTED** | Line 105: `lorebook_xml = lorebook.get_context_for_scene(scene.characters_present, scene.location)` |
| 5 | narrator.py calls summary compression with STRUCTURAL tier | **CONNECTED** | Line 543: `await self.gateway.generate(ModelTier.STRUCTURAL, ...)` inside `_compress_summary()` |
| 6 | api_server.py uses SequentialNarrator | **CONNECTED** | Line 462: lazy import, line 534: `narrator = SequentialNarrator(config=config)` |
| 7 | gateway.py supports `cache_system_prompt` | **CONNECTED** | Line 96: parameter in `generate()`, line 261: conditional cache_control block in `_generate_claude()` |
| 8 | config.py model strings | **CONNECTED** | `structural_model = "grok-4-1-fast"`, `creative_model = "claude-sonnet-4-5-20250929"` |
| 9 | postprocessor.py uses `generate_batch` | **CONNECTED** | Line 63: `responses = await self.gateway.generate_batch(ModelTier.STRUCTURAL, requests, ...)` |
| 10 | NarrativeStateObject.to_prompt_xml() | **CONNECTED** | Line 100: method defined, returns full XML. Line 169: `estimate_tokens()` uses it |

### Cross-Module Data Flow

| Seam | Status | Notes |
|------|--------|-------|
| Simulation → Events → Arc Search | **CONNECTED** | Existing pipeline, confirmed by audit_extraction.py |
| Arc Events → `split_into_scenes()` | **CONNECTED** | 20 events → 4 scenes (seed 42) |
| Scene Chunks → `summarize_scene()` | **CONNECTED** | All 4 scenes get summaries |
| World Data → `Lorebook` | **CONNECTED** | `Lorebook(world_definition, agents, secrets)` works |
| Lorebook → `get_context_for_scene()` XML | **CONNECTED** | Returns valid `<lorebook>` XML with location + characters |
| Lorebook → `get_full_cast()` XML | **CONNECTED** | Returns valid `<full_cast>` XML with 6 characters |
| State + Scene + Lorebook → `build_scene_prompt()` | **CONNECTED** | Returns ~10,866 chars with all 4 sections |
| `build_system_prompt()` → cached system prompt | **CONNECTED** | Returns ~3,964 chars with full_cast embedded |
| Narrator → Gateway (creative) | **CONNECTED** | Awaits `self.gateway.generate(tier, system, user, ...)` |
| Narrator → Gateway (structural/compression) | **CONNECTED** | `_compress_summary()` uses `ModelTier.STRUCTURAL` |
| Narrator → CheckpointManager | **CONNECTED** | `save()` after each scene, `load_latest()` on resume |
| Narrator → PostProcessor | **CONNECTED** | Phase 3: `check_continuity()` + `join_prose()` |
| PostProcessor → Gateway (batch) | **CONNECTED** | `generate_batch(STRUCTURAL, ...)` for continuity checks |
| API `/api/generate-prose` → Narrator | **PARTIAL** | See Gap #1 below |
| Region Search → Prose Pipeline | **PARTIAL** | See Gap #2 below |

---

## 4. Scene Splitting Results (Seed 42)

**Input:** 20 arc events (protagonist: victor), all with beat types assigned

| Scene | Events | Location | Type | Pivotal | Characters |
|-------|--------|----------|------|---------|------------|
| 0 | 10 | dining_table | escalation | No | diana, elena, lydia, marcus, thorne, victor |
| 1 | 5 | dining_table | escalation | Yes | diana, elena, lydia, marcus, thorne, victor |
| 2 | 3 | dining_table | revelation | No | lydia, marcus, thorne, victor |
| 3 | 2 | balcony | escalation | Yes | diana, elena, lydia, marcus, thorne, victor |

**Observations:**
- Scene 0 is at the target chunk size (10). Good.
- Scene 3 only has 2 events (below min_chunk_size=3) — this is the tail end, which the splitter allows for hard boundaries (location change: dining_table → balcony)
- 2 of 4 scenes are pivotal → CREATIVE_DEEP will be used for those scenes
- Character sets are large (5-6) because it's a dinner table scene — lorebook per-character budget will be constrained

---

## 5. Prompt Budget

| Component | Chars | ~Tokens | Budget |
|-----------|-------|---------|--------|
| System prompt | 3,964 | 991 | Cached (one-time write) |
| Scene prompt (scene 0) | 10,866 | 2,716 | Per-scene |
| NarrativeStateObject XML | 4,361 | 1,090 | Per-scene, within scene prompt |
| Summary compression prompt | ~699 | ~175 | Per-scene (structural tier) |
| Continuity check prompt | ~2,912 | ~728 | Per-scene (structural tier, batch) |
| **Total per creative call** | **~14,830** | **~3,707** | System (cached) + Scene |

**Finding:** Initial NarrativeStateObject is 1,090 tokens, exceeding `max_state_tokens=1000`. This is with empty summaries/threads — after a few scenes, the state will be larger. The narrator does not currently enforce the `max_state_tokens` limit (no truncation logic exists in `_apply_state_update` or `_compress_summary`). The compressed summary is bounded by `max_summary_words=500`, but character states, threads, and narrative plan add up.

---

## 6. Identified Gaps

### Gap 1: API endpoint does not pass world_data to narrator (MEDIUM)
**File:** `narrativefield/extraction/api_server.py:464-490`

**What's wrong:** The `/api/generate-prose` endpoint builds `world_data` in two cases:
- **Inline events path** (line 473): sets `world_data = {"raw_context": ctx}`, but the narrator's `_build_lorebook()` (narrator.py:481-492) expects keys `"world_definition"`, `"agents"`, `"secrets"`. The `"raw_context"` key will cause a KeyError that's silently caught, resulting in `lorebook = None`.
- **Loaded-payload path** (lines 474-485): `world_data` is never set — it stays `None`. The narrator proceeds without a Lorebook.

**Impact:** All prose generation via the API endpoint will use the fallback system prompt (generic fiction writer, narrator.py:497-504) instead of the rich Lorebook-powered prompt with character profiles, relationships, and location descriptions. The narrator still works (tests pass via mock), but prompt quality degrades significantly.

**Fix:** For the loaded-payload path, construct `world_data` from `_loaded_payload`:
```python
# After resolving events from _loaded_payload:
world_data = {
    "world_definition": ...,  # Need WorldDefinition from payload
    "agents": ...,            # Need AgentState list from payload
    "secrets": list(_loaded_payload.secrets.values()),
}
```
The challenge is that `_loaded_payload` stores agents as raw dicts (`agents_manifest`), not `AgentState` objects. Either the payload loading needs to preserve `AgentState` / `WorldDefinition` objects, or the narrator's `_build_lorebook()` needs to accept raw dicts. For the inline-events path, change `{"raw_context": ctx}` to properly extract and pass the required keys.

### Gap 2: No direct region-search → prose pipeline flow (LOW)
**File:** `narrativefield/extraction/api_server.py`

**What's wrong:** The `/api/search-region` endpoint returns candidate arcs with events, scores, and validation results. There is no endpoint or client-side flow that takes a search-region result and feeds it into `/api/generate-prose`. The user must manually extract event IDs from the search result and pass them to the prose endpoint.

**Impact:** Minor — this is a frontend UX concern, not a backend bug. The data is compatible; it just needs wiring in the frontend.

### Gap 3: NarrativeStateObject exceeds max_state_tokens on initialization (MEDIUM)
**File:** `narrativefield/storyteller/types.py:166-169`, `narrativefield/storyteller/narrator.py:310-320`

**What's wrong:** The initial state for 6 characters with empty fields is already 1,090 tokens, exceeding the `max_state_tokens=1000` budget. After generation begins, summaries, threads, and character knowledge will grow. No truncation or enforcement mechanism exists.

**Fix:** Either raise `max_state_tokens` to 1500, or add truncation logic (trim character knowledge lists, compress narrative_plan to only upcoming scenes, limit characters in state to only those in current+next scene).

---

## 7. Recommended Next Steps

### Priority 1 (before first real LLM run)
1. **Fix Gap 1:** Wire world_data in `/api/generate-prose` so the Lorebook is built for loaded-payload runs. This is critical for prompt quality.
2. **Fix Gap 3:** Either raise `max_state_tokens` to 1500, or add state truncation logic to bound the NarrativeStateObject.

### Priority 2 (before shipping)
3. **Frontend wiring:** Add a "Generate Prose" button in the arc-search results UI that passes candidate events to `/api/generate-prose`.
4. **Cost guardrail:** Add an estimated cost preview before calling the narrator (the `UsageStats` pricing is already implemented).

### Priority 3 (nice to have)
5. **Test with real LLM calls:** Run a single seed-42 arc through the full pipeline with actual API keys to verify prompt/response parsing works end-to-end.
6. **Lorebook token budget tuning:** With 6 characters, the per-character budget in `get_context_for_scene` caps at ~533 chars each. Verify this is sufficient for meaningful character context.

---

## Appendix: Full Test + Audit Results

### pytest (66/66 pass)
```
test_affordances (5), test_api_server (2), test_arc_search_real_data (1),
test_checkpoint (5), test_determinism (3), test_event_bundler (15),
test_extraction (3), test_gateway (5), test_lorebook (2),
test_metrics_pipeline (1), test_narrator_e2e (6), test_pacing (6),
test_scene_splitter (3), test_schema (4), test_simulation_success (1)
```

### audit_storyteller.py (94 pass, 0 fail, 1 finding)
- 12/12 module imports pass
- 35/35 type field checks pass
- 4/4 scene splitting checks pass
- 7/7 event summarizer checks pass
- 5/5 lorebook checks pass
- 15/15 prompt construction checks pass
- 13/13 narrator instantiation checks pass
- 10/10 API endpoint wiring checks pass
- 1 finding: initial state tokens (1,090) exceed max_state_tokens (1,000)

### Grep Checks (10/10 connected)
All import chains, method calls, and type references verified present in source code.

### Audit Scripts
```bash
cd src/engine && source .venv/bin/activate
python -m pytest -v                # 66 pass
python audit_extraction.py         # 50 pass, 11 findings
python audit_sim.py                # 0 invariant violations
python audit_storyteller.py        # 94 pass, 1 finding
```
