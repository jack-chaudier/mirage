"""
NarrativeField Storyteller Integration Audit
=============================================
Verifies all 10 modules import, types are compatible, and the full
pipeline (scene splitting → summarization → prompt construction →
narrator instantiation → API endpoint wiring) works on real sim data.

NO LLM API calls are made. NO code is fixed.
"""
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import inspect
import traceback

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
_pass = 0
_fail = 0
_findings: list[str] = []
_info: list[str] = []


def _ok(msg: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS: {msg}")


def _bad(msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  FAIL: {msg}")


def _finding(severity: str, msg: str) -> None:
    _findings.append(f"[{severity}] {msg}")
    print(f"  FINDING [{severity}]: {msg}")


def _note(msg: str) -> None:
    _info.append(msg)
    print(f"  INFO: {msg}")


# ===================================================================
# Section 1: Module imports
# ===================================================================
print("=" * 72)
print("NarrativeField Storyteller Integration Audit")
print("=" * 72)
print()

print("[1/8] Module imports...")

_import_results: dict[str, bool] = {}

MODULES = [
    "narrativefield.llm",
    "narrativefield.llm.config",
    "narrativefield.llm.gateway",
    "narrativefield.storyteller",
    "narrativefield.storyteller.types",
    "narrativefield.storyteller.narrator",
    "narrativefield.storyteller.checkpoint",
    "narrativefield.storyteller.lorebook",
    "narrativefield.storyteller.scene_splitter",
    "narrativefield.storyteller.event_summarizer",
    "narrativefield.storyteller.prompts",
    "narrativefield.storyteller.postprocessor",
]

for mod_name in MODULES:
    try:
        __import__(mod_name)
        _import_results[mod_name] = True
        _ok(f"import {mod_name}")
    except Exception as exc:
        _import_results[mod_name] = False
        _bad(f"import {mod_name}: {exc}")

print()

# ===================================================================
# Section 2: Type field compatibility checks
# ===================================================================
print("[2/8] Type field compatibility checks...")

from narrativefield.storyteller.types import (
    GenerationResult,
    NarrativeStateObject,
    SceneChunk,
)
from narrativefield.llm.config import PipelineConfig
from narrativefield.llm.gateway import LLMGateway, ModelTier

# NarrativeStateObject fields
for field in ["summary_so_far", "last_paragraph", "current_scene_index",
              "characters", "active_location", "unresolved_threads",
              "narrative_plan", "total_words_generated", "scenes_completed"]:
    if field in NarrativeStateObject.__dataclass_fields__:
        _ok(f"NarrativeStateObject has field '{field}'")
    else:
        _bad(f"NarrativeStateObject missing field '{field}'")

# Methods
for method_name in ["to_prompt_xml", "estimate_tokens", "to_dict", "from_dict"]:
    if hasattr(NarrativeStateObject, method_name):
        _ok(f"NarrativeStateObject has method '{method_name}'")
    else:
        _bad(f"NarrativeStateObject missing method '{method_name}'")

# SceneChunk fields
for field in ["scene_index", "events", "location", "time_start", "time_end",
              "characters_present", "scene_type", "is_pivotal", "summary"]:
    if field in SceneChunk.__dataclass_fields__:
        _ok(f"SceneChunk has field '{field}'")
    else:
        _bad(f"SceneChunk missing field '{field}'")

# GenerationResult fields
for field in ["prose", "word_count", "scenes_generated", "final_state",
              "usage", "generation_time_seconds", "checkpoint_path"]:
    if field in GenerationResult.__dataclass_fields__:
        _ok(f"GenerationResult has field '{field}'")
    else:
        _bad(f"GenerationResult missing field '{field}'")

# ModelTier enum members
for member in ["STRUCTURAL", "CREATIVE", "CREATIVE_DEEP"]:
    if hasattr(ModelTier, member):
        _ok(f"ModelTier has '{member}'")
    else:
        _bad(f"ModelTier missing '{member}'")

# PipelineConfig defaults
config = PipelineConfig()
if config.structural_model == "grok-4-1-fast":
    _ok(f"PipelineConfig.structural_model = {config.structural_model}")
else:
    _finding("MEDIUM", f"PipelineConfig.structural_model unexpected: {config.structural_model}")

if config.creative_model == "claude-sonnet-4-5-20250929":
    _ok(f"PipelineConfig.creative_model = {config.creative_model}")
else:
    _finding("MEDIUM", f"PipelineConfig.creative_model unexpected: {config.creative_model}")

if config.checkpoint_enabled:
    _ok("PipelineConfig.checkpoint_enabled = True (default)")
else:
    _finding("LOW", "PipelineConfig.checkpoint_enabled default is False; expected True")

print()

# ===================================================================
# Section 3: Run sim → arc → split_into_scenes on seed-42
# ===================================================================
print("[3/8] Scene splitting on seed-42 simulation data...")

from random import Random
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation as _run_sim
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.extraction.beat_classifier import classify_beats
from narrativefield.extraction.arc_search import search_arc
from narrativefield.storyteller.scene_splitter import split_into_scenes

world = create_dinner_party_world()
rng = Random(42)
cfg = SimulationConfig(
    tick_limit=300,
    event_limit=200,
    max_sim_time=world.definition.sim_duration_minutes,
    snapshot_interval_events=world.definition.snapshot_interval,
)
events, _snapshots = _run_sim(world, rng, cfg)
_note(f"Simulation produced {len(events)} events")

# Keep references to agents and secrets for lorebook building
agents = list(world.agents.values())
secrets = world.definition.secrets

# Classify beats
beats = classify_beats(events)
for ev, bt in zip(events, beats):
    ev.beat_type = bt
_ok(f"Beat classification: {len(beats)} beats assigned")

# Arc search for protagonist
search_result = search_arc(
    all_events=events,
    time_start=None,
    time_end=None,
    agent_ids=None,
    protagonist=None,
    max_events=20,
)
arc_events = search_result.events
protagonist = search_result.protagonist
_note(f"Arc search: {len(arc_events)} events, protagonist={protagonist}, valid={search_result.validation.valid}")

# Re-classify beats for arc
arc_beats = classify_beats(arc_events)
for ev, bt in zip(arc_events, arc_beats):
    ev.beat_type = bt

# Split into scenes
scenes = split_into_scenes(arc_events, target_chunk_size=10)
_ok(f"split_into_scenes returned {len(scenes)} scenes")
for i, sc in enumerate(scenes):
    _note(
        f"  Scene {i}: {len(sc.events)} events, location={sc.location}, "
        f"type={sc.scene_type}, pivotal={sc.is_pivotal}, "
        f"characters={sc.characters_present}"
    )

# Validate SceneChunk fields are populated
for i, sc in enumerate(scenes):
    if not sc.characters_present:
        _finding("HIGH", f"Scene {i} has empty characters_present list")
    if sc.time_end < sc.time_start:
        _finding("HIGH", f"Scene {i} has time_end < time_start")

# Ensure at least one pivotal scene
pivotal_count = sum(1 for s in scenes if s.is_pivotal)
if pivotal_count > 0:
    _ok(f"{pivotal_count} pivotal scene(s) found")
else:
    _finding("MEDIUM", "No pivotal scenes found in arc — CREATIVE_DEEP tier will never be used")

print()

# ===================================================================
# Section 4: Event summarizer on real events
# ===================================================================
print("[4/8] Event summarizer on real events...")

from narrativefield.storyteller.event_summarizer import summarize_event, summarize_scene

# Test on individual events
for ev in arc_events[:5]:
    summary = summarize_event(ev)
    if summary and len(summary) > 0:
        _ok(f"summarize_event({ev.id[:16]}...) = '{summary[:60]}...'")
    else:
        _bad(f"summarize_event({ev.id[:16]}...) returned empty string")

# Test scene summary
if scenes:
    scene_summary = summarize_scene(scenes[0].events)
    if scene_summary and len(scene_summary) > 10:
        _ok(f"summarize_scene(scene_0) = '{scene_summary[:80]}...'")
    else:
        _bad(f"summarize_scene(scene_0) returned too-short: '{scene_summary}'")

# Fill summaries on all scenes (needed for prompt building)
for sc in scenes:
    sc.summary = summarize_scene(sc.events)
_ok(f"All {len(scenes)} scene summaries filled")

print()

# ===================================================================
# Section 5: Lorebook construction from world data
# ===================================================================
print("[5/8] Lorebook construction...")

from narrativefield.storyteller.lorebook import Lorebook

agent_states = list(agents)

try:
    lorebook = Lorebook(world.definition, agent_states, list(secrets.values()))
    _ok("Lorebook constructed from (world, agents, secrets)")
except Exception as exc:
    _bad(f"Lorebook construction failed: {exc}")
    traceback.print_exc()
    lorebook = None

if lorebook is not None:
    # Test get_context_for_scene
    if scenes:
        ctx_xml = lorebook.get_context_for_scene(
            scenes[0].characters_present, scenes[0].location
        )
        if "<lorebook>" in ctx_xml and "</lorebook>" in ctx_xml:
            _ok(f"get_context_for_scene XML valid ({len(ctx_xml)} chars)")
        else:
            _bad("get_context_for_scene XML missing root tags")
        _note(f"  Lorebook context sample (first 200 chars):\n    {ctx_xml[:200]}...")

    # Test get_full_cast
    cast_xml = lorebook.get_full_cast()
    if "<full_cast>" in cast_xml and "</full_cast>" in cast_xml:
        _ok(f"get_full_cast XML valid ({len(cast_xml)} chars)")
    else:
        _bad("get_full_cast XML missing root tags")

    # Count characters in cast
    char_count = cast_xml.count("<character ")
    _note(f"  Full cast has {char_count} character entries")
    if char_count != len(agent_states):
        _finding("LOW", f"Full cast has {char_count} entries but {len(agent_states)} agents exist")

print()

# ===================================================================
# Section 6: Prompt construction (all 4 types)
# ===================================================================
print("[6/8] Prompt construction (4 types)...")

from narrativefield.storyteller.prompts import (
    build_system_prompt,
    build_scene_prompt,
    build_summary_compression_prompt,
    build_continuity_check_prompt,
)
from narrativefield.storyteller.narrator import _build_initial_characters

if lorebook is not None and scenes:
    # 1. System prompt
    sys_prompt = build_system_prompt(lorebook, config)
    sys_tokens = len(sys_prompt) // 4
    _ok(f"build_system_prompt: {len(sys_prompt)} chars (~{sys_tokens} tokens)")
    if "OUTPUT FORMAT" in sys_prompt:
        _ok("System prompt contains OUTPUT FORMAT section")
    else:
        _finding("MEDIUM", "System prompt missing OUTPUT FORMAT section")
    if "<full_cast>" in sys_prompt:
        _ok("System prompt embeds full_cast XML")
    else:
        _finding("MEDIUM", "System prompt does not embed full_cast XML")

    # 2. Scene prompt
    # Build initial state
    initial_chars = _build_initial_characters(arc_events)
    state = NarrativeStateObject(
        summary_so_far="",
        last_paragraph="",
        current_scene_index=0,
        characters=initial_chars,
        active_location=arc_events[0].location_id if arc_events else "",
        unresolved_threads=[],
        narrative_plan=[sc.summary for sc in scenes],
        total_words_generated=0,
        scenes_completed=0,
    )

    scene_prompt = build_scene_prompt(
        state, scenes[0], lorebook, scenes[1:3], config
    )
    scene_tokens = len(scene_prompt) // 4
    _ok(f"build_scene_prompt: {len(scene_prompt)} chars (~{scene_tokens} tokens)")
    if "<narrative_state>" in scene_prompt:
        _ok("Scene prompt contains narrative_state XML")
    else:
        _finding("HIGH", "Scene prompt missing narrative_state XML block")
    if "<events>" in scene_prompt:
        _ok("Scene prompt contains events XML")
    else:
        _finding("HIGH", "Scene prompt missing events XML block")
    if "<lorebook>" in scene_prompt:
        _ok("Scene prompt contains lorebook XML")
    else:
        _finding("MEDIUM", "Scene prompt missing lorebook XML block")
    if "<instructions>" in scene_prompt:
        _ok("Scene prompt contains instructions block")
    else:
        _finding("MEDIUM", "Scene prompt missing instructions block")

    # 3. Summary compression prompt
    comp_prompt = build_summary_compression_prompt("Victor went to the kitchen.", "He found the knife.", 500)
    if "existing_summary" in comp_prompt and "new_scene" in comp_prompt:
        _ok(f"build_summary_compression_prompt: {len(comp_prompt)} chars")
    else:
        _bad("Summary compression prompt missing expected XML sections")

    # 4. Continuity check prompt
    cont_prompt = build_continuity_check_prompt(
        "Victor walked into the kitchen.", scenes[0].events, initial_chars
    )
    if "simulation_events" in cont_prompt and "character_states" in cont_prompt:
        _ok(f"build_continuity_check_prompt: {len(cont_prompt)} chars")
    else:
        _bad("Continuity check prompt missing expected sections")

    # Token budget estimates
    state_xml = state.to_prompt_xml()
    state_tokens = state.estimate_tokens()
    _note(f"  NarrativeStateObject.to_prompt_xml(): {len(state_xml)} chars, ~{state_tokens} tokens")
    if state_tokens > config.max_state_tokens:
        _finding("MEDIUM", f"Initial state already exceeds max_state_tokens ({state_tokens} > {config.max_state_tokens})")
    else:
        _ok(f"Initial state tokens ({state_tokens}) within budget ({config.max_state_tokens})")

    total_prompt_tokens = sys_tokens + scene_tokens + state_tokens
    _note(f"  Total estimated prompt: ~{total_prompt_tokens} tokens (system={sys_tokens} + scene={scene_tokens} + state={state_tokens})")

print()

# ===================================================================
# Section 7: SequentialNarrator instantiation + method signatures
# ===================================================================
print("[7/8] SequentialNarrator instantiation + method inspection...")

from narrativefield.storyteller.narrator import SequentialNarrator
from narrativefield.storyteller.postprocessor import PostProcessor
from narrativefield.storyteller.checkpoint import CheckpointManager

# Instantiate narrator
narrator = SequentialNarrator(config=config)
if isinstance(narrator.gateway, LLMGateway):
    _ok("SequentialNarrator.gateway is LLMGateway")
else:
    _bad(f"SequentialNarrator.gateway type: {type(narrator.gateway)}")

if narrator.config is config:
    _ok("SequentialNarrator.config passed through correctly")
else:
    _bad("SequentialNarrator.config mismatch")

# Check generate signature
gen_sig = inspect.signature(narrator.generate)
expected_params = ["events", "beat_sheet", "world_data", "run_id", "resume"]
for param in expected_params:
    if param in gen_sig.parameters:
        _ok(f"SequentialNarrator.generate has param '{param}'")
    else:
        _bad(f"SequentialNarrator.generate missing param '{param}'")

# Check narrator.generate is async
if asyncio.iscoroutinefunction(narrator.generate):
    _ok("SequentialNarrator.generate is async")
else:
    _bad("SequentialNarrator.generate is NOT async")

# PostProcessor
pp = PostProcessor(narrator.gateway, config)
if asyncio.iscoroutinefunction(pp.check_continuity):
    _ok("PostProcessor.check_continuity is async")
else:
    _bad("PostProcessor.check_continuity is NOT async")

if callable(getattr(pp, "join_prose", None)):
    _ok("PostProcessor.join_prose exists and is callable")
else:
    _bad("PostProcessor.join_prose missing")

# Test join_prose with sample chunks
joined = pp.join_prose(["First scene prose.", "Second scene prose."])
if "* * *" in joined:
    _ok("PostProcessor.join_prose inserts scene breaks")
else:
    _finding("LOW", "PostProcessor.join_prose does not insert scene breaks")

# Test XML artifact stripping
joined_dirty = pp.join_prose(["<prose>Clean text</prose><state_update><summary>ignore</summary></state_update>"])
if "<prose>" not in joined_dirty and "<state_update>" not in joined_dirty:
    _ok("PostProcessor.join_prose strips XML artifacts")
else:
    _bad(f"PostProcessor.join_prose did not strip XML: '{joined_dirty[:100]}'")

# CheckpointManager
ckpt = CheckpointManager("/tmp/test_audit_ckpt", "test_run")
if ckpt.run_id == "test_run":
    _ok("CheckpointManager instantiation works")
else:
    _bad("CheckpointManager run_id mismatch")

print()

# ===================================================================
# Section 8: API endpoint wiring
# ===================================================================
print("[8/8] API endpoint wiring...")

from narrativefield.extraction.api_server import app

# Check routes
route_paths = [route.path for route in app.routes]
_note(f"  Registered routes: {route_paths}")

expected_routes = ["/extract", "/api/generate-prose", "/api/prose-status/{run_id}", "/api/search-region"]
for route_path in expected_routes:
    if route_path in route_paths:
        _ok(f"Route {route_path} registered")
    else:
        _bad(f"Route {route_path} NOT registered")

# Check that /api/generate-prose handler imports are correct
# (lazy imports inside the function — verify they resolve)
try:
    from narrativefield.llm.config import PipelineConfig as PC2
    from narrativefield.storyteller.narrator import SequentialNarrator as SN2
    _ = (PC2, SN2)
    _ok("Lazy imports in generate_prose_endpoint resolve correctly")
except ImportError as exc:
    _bad(f"Lazy imports in generate_prose_endpoint fail: {exc}")

# Check ProseGenerationRequestModel
from narrativefield.extraction.api_server import ProseGenerationRequestModel
model_fields = set(ProseGenerationRequestModel.model_fields.keys())
for field in ["event_ids", "protagonist_agent_id", "resume_run_id", "config_overrides", "selected_events", "context"]:
    if field in model_fields:
        _ok(f"ProseGenerationRequestModel has field '{field}'")
    else:
        _bad(f"ProseGenerationRequestModel missing field '{field}'")

print()

# ===================================================================
# Summary
# ===================================================================
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  Total: {_pass} pass, {_fail} fail, {len(_findings)} finding(s), {len(_info)} info")
print()

if _findings:
    print("FINDINGS:")
    for f in _findings:
        print(f"  {f}")
    print()

if _info:
    print("INFO:")
    for i in _info:
        print(f"  {i}")
    print()

if _fail > 0:
    print(f"RESULT: {_fail} FAILURE(S) — integration issues detected")
else:
    print("RESULT: ALL CHECKS PASSED — storyteller pipeline is fully wired")
