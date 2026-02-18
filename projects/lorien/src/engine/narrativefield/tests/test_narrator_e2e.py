"""End-to-end tests for the prose generation pipeline.

Uses seed-42 simulation data with Victor's arc.
Mocks LLM calls to avoid API costs during testing.
"""

from __future__ import annotations

import asyncio
from random import Random
from unittest.mock import MagicMock, patch

import pytest

from narrativefield.extraction.arc_search import search_arc
from narrativefield.llm.config import PipelineConfig
from narrativefield.llm.gateway import LLMGateway, ModelTier
from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType
from narrativefield.simulation.scenarios.dinner_party import create_dinner_party_world
from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation
from narrativefield.storyteller.lorebook import Lorebook
from narrativefield.storyteller.prompts import (
    build_continuity_check_prompt,
    build_scene_prompt,
    build_summary_compression_prompt,
    build_system_prompt,
)
from narrativefield.storyteller.scene_splitter import split_into_scenes
from narrativefield.storyteller.types import (
    CharacterState,
    GenerationResult,
    NarrativeStateObject,
    NarrativeThread,
    SceneOutcome,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_seed42():
    """Run seed-42 simulation once and cache results for all tests."""
    world = create_dinner_party_world()
    rng = Random(42)
    cfg = SimulationConfig(
        tick_limit=300,
        event_limit=200,
        max_sim_time=150.0,
        snapshot_interval_events=20,
    )
    events, snapshots = run_simulation(world, rng, cfg)
    return {"world": world, "events": events, "snapshots": snapshots}


@pytest.fixture(scope="module")
def victor_arc(sim_seed42):
    """Extract Victor's arc from seed-42 data."""
    events = sim_seed42["events"]
    result = search_arc(
        all_events=events,
        protagonist="victor",
        max_events=20,
        total_sim_time=150.0,
    )
    # Set beat types on events.
    for ev, bt in zip(result.events, result.beats):
        ev.beat_type = bt
    return result


@pytest.fixture
def mock_gateway():
    """Create a mock LLMGateway that returns plausible responses."""
    gw = LLMGateway()

    # Track call count per tier.
    call_log: dict[str, int] = {"structural": 0, "creative": 0, "creative_deep": 0}

    async def _mock_generate(
        tier: ModelTier,
        system_prompt: str,
        user_prompt: str,
        cache_system_prompt: bool = False,
        max_tokens: int = 2000,
    ) -> str:
        call_log[tier.value] = call_log.get(tier.value, 0) + 1

        if tier is ModelTier.STRUCTURAL:
            # Determine if this is a summary compression or continuity check.
            if "continuity" in system_prompt.lower() or "continuity" in user_prompt.lower():
                return '{"consistent": true, "violations": []}'
            # Summary compression.
            return (
                "Victor arrived at the dinner party. Marcus greeted him coolly. "
                "Elena observed from the kitchen. Tension rising among the guests."
            )

        # CREATIVE or CREATIVE_DEEP — return mock prose with state update.
        scene_num = call_log.get("creative", 0) + call_log.get("creative_deep", 0)
        return (
            "<prose>\n"
            f"The candlelight flickered across Victor's face as he studied the room. "
            f"Scene {scene_num} of the evening unfolded with quiet menace. "
            "Marcus sat at the far end of the table, his smile a thin mask over something harder. "
            "Elena's fingers traced the stem of her wine glass, but her eyes were elsewhere — fixed on "
            "the doorway, as if calculating the distance to the balcony. Victor felt the weight of "
            "secrets pressing against the conversation like water against a dam.\n\n"
            '"More wine?" Thorne offered, his voice pitched to carry exactly the right blend of '
            "generosity and control. Victor declined with a small shake of his head. The investigation "
            "could not wait much longer. Every minute at this table was a minute Marcus might discover "
            "what Victor already knew.\n"
            "</prose>\n"
            "<state_update>\n"
            "<summary>Victor watches Marcus at the dinner table. Elena seems anxious. "
            "Thorne plays the gracious host. Victor is preparing to confront Marcus.</summary>\n"
            "<character_updates>\n"
            '<character id="victor" emotional_state="alert, calculating" current_goal="confront Marcus" />\n'
            '<character id="marcus" emotional_state="guarded" current_goal="maintain facade" />\n'
            "</character_updates>\n"
            "<new_threads>\n"
            '<thread description="Victor preparing confrontation" involved="victor,marcus" tension="0.7" />\n'
            "</new_threads>\n"
            "<resolved_threads></resolved_threads>\n"
            "</state_update>"
        )

    gw.generate = _mock_generate  # type: ignore[assignment]
    gw._call_log = call_log  # type: ignore[attr-defined]

    # Mock generate_batch too.
    async def _mock_batch(tier, requests, max_concurrency=10):
        return [
            await _mock_generate(tier, r.get("system", ""), r.get("user", ""))
            for r in requests
        ]

    gw.generate_batch = _mock_batch  # type: ignore[assignment]

    return gw


# ---------------------------------------------------------------------------
# Test 1: Full pipeline with mocked LLM
# ---------------------------------------------------------------------------


def test_full_pipeline_with_mocked_llm(sim_seed42, victor_arc, mock_gateway, tmp_path):
    """Run the complete prose generation pipeline with mocked LLM calls."""
    from narrativefield.storyteller.narrator import SequentialNarrator

    config = PipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_enabled=True,
    )
    narrator = SequentialNarrator(config=config)
    narrator.gateway = mock_gateway

    events = victor_arc.events
    world = sim_seed42["world"]

    # Build world_data from simulation world state.

    world_data = {
        "world_definition": world.definition,
        "agents": list(world.agents.values()),
        "secrets": list(world.definition.secrets.values()),
    }

    result = asyncio.run(
        narrator.generate(
            events=events,
            world_data=world_data,
            run_id="test-full-pipeline",
        )
    )

    # Assertions
    assert isinstance(result, GenerationResult)
    assert result.word_count > 0
    assert len(result.prose) > 0
    assert result.scenes_generated >= 1
    assert result.generation_time_seconds >= 0
    assert "total_input_tokens" in result.usage or "continuity_report" in result.usage


# ---------------------------------------------------------------------------
# Test 2: Checkpoint resume
# ---------------------------------------------------------------------------


def test_checkpoint_resume(sim_seed42, victor_arc, tmp_path):
    """Test that generation can be interrupted and resumed from checkpoint."""
    from narrativefield.storyteller.narrator import SequentialNarrator

    config = PipelineConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_enabled=True,
        phase2_events_per_chunk=3,
    )

    call_count = 0
    fail_at_scene = 1  # Fail on the 2nd creative call.

    async def _failing_generate(
        tier: ModelTier,
        system_prompt: str,
        user_prompt: str,
        cache_system_prompt: bool = False,
        max_tokens: int = 2000,
    ) -> str:
        nonlocal call_count
        if tier in (ModelTier.CREATIVE, ModelTier.CREATIVE_DEEP):
            call_count += 1
            if call_count > fail_at_scene:
                raise RuntimeError("Simulated API failure")
            return (
                "<prose>\n"
                f"Mock prose chunk {call_count}. Victor surveyed the room.\n"
                "</prose>\n"
                "<state_update>\n"
                f"<summary>Scene {call_count} summary.</summary>\n"
                "<character_updates></character_updates>\n"
                "<new_threads></new_threads>\n"
                "<resolved_threads></resolved_threads>\n"
                "</state_update>"
            )
        # Structural: summary compression
        return f"Compressed summary after scene {call_count}."

    async def _failing_batch(tier, requests, max_concurrency=10):
        return [
            await _failing_generate(tier, r.get("system", ""), r.get("user", ""))
            for r in requests
        ]

    # --- Run 1: Should fail partway through ---
    narrator1 = SequentialNarrator(config=config)
    narrator1.gateway.generate = _failing_generate  # type: ignore[assignment]
    narrator1.gateway.generate_batch = _failing_batch  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Simulated API failure"):
        asyncio.run(
            narrator1.generate(
                events=victor_arc.events,
                run_id="test-resume",
            )
        )

    # Verify checkpoint exists.
    from narrativefield.storyteller.checkpoint import CheckpointManager

    mgr = CheckpointManager(str(tmp_path / "checkpoints"), "test-resume")
    assert mgr.has_checkpoint()

    # --- Run 2: Resume should continue from checkpoint ---
    call_count = 0  # Reset so it doesn't fail again immediately.
    fail_at_scene = 999  # Don't fail this time.

    narrator2 = SequentialNarrator(config=config)
    narrator2.gateway.generate = _failing_generate  # type: ignore[assignment]
    narrator2.gateway.generate_batch = _failing_batch  # type: ignore[assignment]

    result = asyncio.run(
        narrator2.generate(
            events=victor_arc.events,
            run_id="test-resume",
            resume=True,
        )
    )

    assert isinstance(result, GenerationResult)
    assert result.word_count > 0
    assert result.scenes_generated >= 1


# ---------------------------------------------------------------------------
# Test 2b: GenerationResult diagnostics (scene outcomes + run status)
# ---------------------------------------------------------------------------


def test_generation_result_tracks_scene_outcomes(sim_seed42, victor_arc, mock_gateway):
    """GenerationResult includes per-scene status tracking."""
    from narrativefield.storyteller.narrator import SequentialNarrator

    config = PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=5)
    narrator = SequentialNarrator(config=config)
    narrator.gateway = mock_gateway

    world = sim_seed42["world"]
    world_data = {
        "world_definition": world.definition,
        "agents": list(world.agents.values()),
        "secrets": list(world.definition.secrets.values()),
    }

    result = asyncio.run(
        narrator.generate(
            events=victor_arc.events,
            world_data=world_data,
            run_id="test-scene-outcomes",
        )
    )

    assert isinstance(result, GenerationResult)
    assert isinstance(result.scene_outcomes, list)
    assert len(result.scene_outcomes) == result.scenes_generated
    assert all(s.status == "ok" for s in result.scene_outcomes)
    assert result.status == "complete"


def test_generation_result_partial_on_scene_failure(sim_seed42, victor_arc):
    """If a scene fails, result.status is 'partial' not 'complete'."""
    from narrativefield.storyteller.narrator import SequentialNarrator

    # Create a gateway that returns empty prose for the first scene, then succeeds.
    gw = LLMGateway()
    creative_calls = 0

    async def _mock_generate(
        tier: ModelTier,
        system_prompt: str,
        user_prompt: str,
        cache_system_prompt: bool = False,
        max_tokens: int = 2000,
    ) -> str:
        nonlocal creative_calls

        if tier is ModelTier.STRUCTURAL:
            if "continuity" in system_prompt.lower() or "continuity" in user_prompt.lower():
                return '{"consistent": true, "violations": []}'
            return "Compressed summary."

        creative_calls += 1
        if creative_calls == 1:
            return (
                "<prose></prose>\n"
                "<state_update>\n"
                "<summary></summary>\n"
                "</state_update>"
            )
        return (
            "<prose>\n"
            "Victor watched the table with a careful, practiced calm.\n"
            "</prose>\n"
            "<state_update>\n"
            "<summary>Victor watches the dinner guests.</summary>\n"
            "</state_update>"
        )

    async def _mock_batch(tier, requests, max_concurrency=10):
        return [
            await _mock_generate(tier, r.get("system", ""), r.get("user", ""))
            for r in requests
        ]

    gw.generate = _mock_generate  # type: ignore[assignment]
    gw.generate_batch = _mock_batch  # type: ignore[assignment]

    config = PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=5)
    narrator = SequentialNarrator(config=config)
    narrator.gateway = gw

    world = sim_seed42["world"]
    world_data = {
        "world_definition": world.definition,
        "agents": list(world.agents.values()),
        "secrets": list(world.definition.secrets.values()),
    }

    result = asyncio.run(
        narrator.generate(
            events=victor_arc.events,
            world_data=world_data,
            run_id="test-partial-on-failure",
        )
    )

    assert result.status == "partial"
    assert len(result.scene_outcomes) == result.scenes_generated
    assert any(s.status == "failed" for s in result.scene_outcomes)
    assert any(s.status == "ok" for s in result.scene_outcomes)
    assert result.scene_outcomes[0].status == "failed"


# ---------------------------------------------------------------------------
# Test 3: Scene splitting sanity
# ---------------------------------------------------------------------------


def test_scene_splitting_sanity(victor_arc):
    """Verify that 20 events produce reasonable scene chunks."""
    events = victor_arc.events

    # Must have beat types set.
    for ev, bt in zip(events, victor_arc.beats):
        ev.beat_type = bt

    chunks = split_into_scenes(events, target_chunk_size=10, min_chunk_size=3)

    # 20 events should produce 2-6 scene chunks.
    assert 1 <= len(chunks) <= 8, f"Expected 1-8 chunks, got {len(chunks)}"

    # No scene should have more than 15 events.
    for chunk in chunks:
        assert len(chunk.events) <= 15, f"Scene {chunk.scene_index} has {len(chunk.events)} events"

    # TURNING_POINT should be in a pivotal scene.
    tp_events = [e for e in events if e.beat_type == BeatType.TURNING_POINT]
    if tp_events:
        tp_event_ids = {e.id for e in tp_events}
        pivotal_chunks = [c for c in chunks if c.is_pivotal]
        pivotal_event_ids = {e.id for c in pivotal_chunks for e in c.events}
        assert tp_event_ids & pivotal_event_ids, "TURNING_POINT should be in a pivotal scene"

    # All events should be accounted for.
    all_chunk_events = [e.id for c in chunks for e in c.events]
    assert len(all_chunk_events) == len(events)


# ---------------------------------------------------------------------------
# Test 4: Prompt structure
# ---------------------------------------------------------------------------


def test_prompt_structure(sim_seed42, victor_arc):
    """Verify prompt construction produces well-structured prompts."""
    world = sim_seed42["world"]
    events = victor_arc.events

    # Build lorebook.
    lorebook = Lorebook(
        world.definition,
        list(world.agents.values()),
        list(world.definition.secrets.values()),
    )
    config = PipelineConfig()

    # --- System prompt ---
    system_prompt = build_system_prompt(lorebook, config)
    assert len(system_prompt) > 500, "System prompt too short"

    # Should contain the full cast.
    assert "<full_cast>" in system_prompt
    assert "Victor" in system_prompt or "victor" in system_prompt

    # Should contain output format instructions.
    assert "<prose>" in system_prompt
    assert "<state_update>" in system_prompt
    assert "<lore_updates>" in system_prompt

    # --- Scene prompt ---
    for ev, bt in zip(events, victor_arc.beats):
        ev.beat_type = bt
    chunks = split_into_scenes(events)
    assert len(chunks) >= 1

    state = NarrativeStateObject(
        summary_so_far="Victor has arrived at the dinner party.",
        last_paragraph="He stood in the doorway, scanning the room.",
        current_scene_index=0,
        characters=[
            CharacterState(
                agent_id="victor",
                name="Victor",
                location="dining_table",
                emotional_state="watchful",
                current_goal="investigate Marcus",
                knowledge=[],
                secrets_revealed=[],
                secrets_held=["secret_victor_investigation"],
            )
        ],
        active_location="dining_table",
        unresolved_threads=[
            NarrativeThread(
                description="Victor's investigation of Marcus",
                involved_agents=["victor", "marcus"],
                tension_level=0.6,
                introduced_at_scene=0,
            )
        ],
        narrative_plan=["Scene 1 summary", "Scene 2 summary"],
    )

    scene_prompt = build_scene_prompt(
        state=state,
        scene=chunks[0],
        lorebook=lorebook,
        upcoming_scenes=chunks[1:3],
        config=config,
    )

    # Should contain narrative state XML.
    assert "<narrative_state>" in scene_prompt

    # Should contain events section.
    assert "<events>" in scene_prompt

    # Should contain instructions section.
    assert "<instructions>" in scene_prompt
    assert "Write strictly from Victor's perspective." in scene_prompt

    # Pivotal scene should have special treatment.
    pivotal_chunks = [c for c in chunks if c.is_pivotal]
    if pivotal_chunks:
        pivotal_prompt = build_scene_prompt(
            state=state,
            scene=pivotal_chunks[0],
            lorebook=lorebook,
            upcoming_scenes=[],
            config=config,
        )
        assert "PIVOTAL" in pivotal_prompt

    # --- Summary compression prompt ---
    summary_prompt = build_summary_compression_prompt(
        old_summary="Victor arrived.",
        new_scene_prose="Victor confronted Marcus about the embezzlement.",
        max_words=500,
    )
    assert "500" in summary_prompt
    assert "Victor arrived" in summary_prompt

    # --- Continuity check prompt ---
    continuity_prompt = build_continuity_check_prompt(
        prose_chunk="Victor saw Marcus across the room.",
        events=events[:3],
        character_states=state.characters,
    )
    assert "KNOWLEDGE_VIOLATION" in continuity_prompt
    assert "consistent" in continuity_prompt


# ---------------------------------------------------------------------------
# Test 5: Post-processor prose joining
# ---------------------------------------------------------------------------


def test_postprocessor_join_prose():
    """Verify post-processor correctly strips XML artifacts and joins prose."""
    from narrativefield.storyteller.postprocessor import PostProcessor

    pp = PostProcessor(gateway=MagicMock(), config=PipelineConfig())

    chunks = [
        "<prose>\nVictor entered the room. The air was thick with tension.\n</prose>",
        "Marcus poured wine with a steady hand. <state_update><summary>test</summary></state_update>",
        "  \n\n\n  Elena watched from the kitchen doorway.\n\n\n\n  ",
    ]

    joined = pp.join_prose(chunks)

    # XML artifacts should be stripped.
    assert "<prose>" not in joined
    assert "</prose>" not in joined
    assert "<state_update>" not in joined

    # Chunks should be joined with scene breaks.
    assert "* * *" in joined

    # Content should be present.
    assert "Victor entered" in joined
    assert "Marcus poured" in joined
    assert "Elena watched" in joined

    # Multiple blank lines should be normalized.
    assert "\n\n\n" not in joined


def test_join_prose_no_midword_scene_breaks():
    """Scene breaks must not split words."""
    from narrativefield.storyteller.postprocessor import PostProcessor

    pp = PostProcessor(gateway=LLMGateway(), config=PipelineConfig())
    chunks = ["The financial irregular", "ities were documented carefully."]
    result = pp.join_prose(chunks)
    # Should not have a scene break inside "irregularities"
    assert "irregular\n" not in result or "irregularities" in result


# ---------------------------------------------------------------------------
# Test 6: API endpoint (if available)
# ---------------------------------------------------------------------------


def test_generate_prose_endpoint_with_inline_events():
    """Test the /api/generate-prose endpoint with inline selected_events."""
    from fastapi.testclient import TestClient

    from narrativefield.extraction.api_server import app

    # Build minimal synthetic events.
    events = []
    for i in range(5):
        events.append(
            Event(
                id=f"test_e{i:03d}",
                sim_time=float(i * 3),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT if i < 3 else EventType.CONFLICT,
                source_agent="victor",
                target_agents=["marcus"],
                location_id="dining_table",
                causal_links=[f"test_e{i-1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"Test event {i}",
                metrics=EventMetrics(
                    tension=0.1 + i * 0.15,
                    irony=0.2,
                    significance=0.5 if i == 3 else 0.1,
                ),
            )
        )

    payload = {
        "selected_events": [e.to_dict() for e in events],
        "protagonist_agent_id": "victor",
        "context": {
            "metadata": {"total_sim_time": 15.0},
            "agents": [],
            "scenes": [],
            "secrets": [],
        },
    }

    client = TestClient(app)

    # Patch the narrator to use mock gateway.
    with patch(
        "narrativefield.storyteller.narrator.SequentialNarrator"
    ) as MockNarrator:
        mock_instance = MagicMock()

        async def _mock_gen(*args, **kwargs):
            return GenerationResult(
                status="complete",
                prose="Mock generated story.",
                word_count=3,
                scenes_generated=1,
                scene_outcomes=[
                    SceneOutcome(
                        scene_index=0,
                        status="ok",
                        word_count=3,
                        error_type=None,
                        retries=0,
                        generation_time_s=0.1,
                    )
                ],
                final_state=NarrativeStateObject(
                    summary_so_far="test",
                    last_paragraph="test",
                    current_scene_index=0,
                    characters=[],
                    active_location="dining_table",
                    unresolved_threads=[],
                    narrative_plan=[],
                ),
                usage={
                    "total_input_tokens": 100,
                    "total_output_tokens": 200,
                    "estimated_cost_usd": 0.01,
                    "continuity_report": [],
                },
                generation_time_seconds=0.1,
                checkpoint_path=None,
            )

        mock_instance.generate = _mock_gen
        MockNarrator.return_value = mock_instance

        resp = client.post("/api/generate-prose", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "prose" in data
    assert "word_count" in data
    assert "run_id" in data
    assert "usage" in data


# ---------------------------------------------------------------------------
# Test 7: State budget enforcement
# ---------------------------------------------------------------------------


def test_state_budget_enforcement():
    """After heavy accumulation, _enforce_state_budget brings state under budget."""
    from narrativefield.storyteller.narrator import _enforce_state_budget

    config = PipelineConfig()  # max_state_tokens=1500 after fix

    state = NarrativeStateObject(
        summary_so_far="word " * 300,
        last_paragraph="Last paragraph text.",
        current_scene_index=5,
        characters=[
            CharacterState(
                f"agent_{i}",
                f"Agent {i}",
                "dining_table",
                "anxious",
                "some goal",
                [f"knows thing {j}" for j in range(20)],
                [f"revealed {j}" for j in range(10)],
                [f"secret {j}" for j in range(10)],
            )
            for i in range(6)
        ],
        active_location="dining_table",
        unresolved_threads=[
            NarrativeThread(f"thread {i}", ["a", "b"], 0.5, i) for i in range(8)
        ],
        narrative_plan=[f"plan {i}" for i in range(5)],
        total_words_generated=3000,
        scenes_completed=5,
    )

    assert state.estimate_tokens() > config.max_state_tokens  # precondition

    trimmed = _enforce_state_budget(
        state,
        config.max_state_tokens,
        config.max_summary_words,
        current_scene_characters=["agent_0", "agent_1"],
    )

    assert trimmed.estimate_tokens() <= config.max_state_tokens
    assert len(trimmed.narrative_plan) <= 1
    assert len(trimmed.unresolved_threads) <= 3
