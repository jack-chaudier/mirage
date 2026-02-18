from __future__ import annotations

import asyncio

from narrativefield.llm.config import PipelineConfig
from narrativefield.llm.gateway import LLMGateway, ModelTier
from narrativefield.schema.canon import WorldCanon
from narrativefield.schema.events import BeatType, Event, EventMetrics, EventType
from narrativefield.simulation.scenarios.dinner_party import create_dinner_party_world
from narrativefield.storyteller.narrator import SequentialNarrator


def _world_data() -> dict:
    world = create_dinner_party_world()
    return {
        "world_definition": world.definition,
        "agents": list(world.agents.values()),
        "secrets": list(world.definition.secrets.values()),
    }


def _events_three_scenes() -> list[Event]:
    locations = ["dining_table"] * 3 + ["kitchen"] * 3 + ["balcony"] * 3
    out: list[Event] = []
    for i, loc in enumerate(locations):
        out.append(
            Event(
                id=f"evt_{i:03d}",
                sim_time=float(i),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT if i < 6 else EventType.CONFLICT,
                source_agent="victor",
                target_agents=["marcus"],
                location_id=loc,
                causal_links=[f"evt_{i-1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"event {i}",
                metrics=EventMetrics(tension=0.2 + (i * 0.05), irony=0.1, significance=0.3),
                beat_type=BeatType.SETUP,
            )
        )
    return out


def _events_single_scene() -> list[Event]:
    out: list[Event] = []
    for i in range(3):
        out.append(
            Event(
                id=f"single_{i:03d}",
                sim_time=float(i),
                tick_id=i,
                order_in_tick=0,
                type=EventType.CHAT,
                source_agent="victor",
                target_agents=["marcus"],
                location_id="dining_table",
                causal_links=[f"single_{i-1:03d}"] if i > 0 else [],
                deltas=[],
                description=f"single event {i}",
                metrics=EventMetrics(tension=0.2, irony=0.1, significance=0.2),
                beat_type=BeatType.SETUP,
            )
        )
    return out


def _install_gateway(
    narrator: SequentialNarrator,
    *,
    with_lore: bool,
    first_scene_texture_statement: str = "Victor taps his ring finger before pressing a question.",
) -> list[str]:
    gw = LLMGateway()
    creative_prompts: list[str] = []
    creative_calls = 0

    async def _mock_generate(
        tier: ModelTier,
        system_prompt: str,
        user_prompt: str,
        cache_system_prompt: bool = False,
        max_tokens: int = 2000,
    ) -> str:
        nonlocal creative_calls
        if tier in (ModelTier.CREATIVE, ModelTier.CREATIVE_DEEP):
            idx = creative_calls
            creative_calls += 1
            creative_prompts.append(user_prompt)

            lore_block = ""
            if with_lore:
                if idx == 0:
                    lore_block = (
                        "<lore_updates>\n"
                        "<canon_facts>\n"
                        '<fact event_ids="evt_000">Victor clocked Marcus\'s hesitation.</fact>\n'
                        "</canon_facts>\n"
                        "<texture_facts>\n"
                        f'<detail type="gesture" entities="victor">{first_scene_texture_statement}</detail>\n'
                        "</texture_facts>\n"
                        "</lore_updates>"
                    )
                else:
                    lore_block = (
                        "<lore_updates>\n"
                        "<canon_facts>\n"
                        '<fact event_ids="evt_001">The tension keeps rising.</fact>\n'
                        "</canon_facts>\n"
                        "<texture_facts></texture_facts>\n"
                        "</lore_updates>"
                    )

            return (
                "<prose>\n"
                f"Scene {idx} prose from Victor perspective.\n"
                "</prose>\n"
                "<state_update>\n"
                f"<summary>Summary {idx}.</summary>\n"
                "<character_updates>\n"
                '<character id="victor" emotional_state="focused" current_goal="pressure Marcus" />\n'
                "</character_updates>\n"
                "<new_threads></new_threads>\n"
                "<resolved_threads></resolved_threads>\n"
                "</state_update>\n"
                f"{lore_block}"
            )

        # STRUCTURAL
        if "<new_details>" in user_prompt:
            return "[]"
        return "Compressed factual summary."

    async def _mock_batch(tier, requests, max_concurrency=10):
        return ['{"consistent": true, "violations": []}' for _ in requests]

    gw.generate = _mock_generate  # type: ignore[assignment]
    gw.generate_batch = _mock_batch  # type: ignore[assignment]
    narrator.gateway = gw
    return creative_prompts


def test_narrator_with_canon_parameter() -> None:
    narrator = SequentialNarrator(config=PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=3))
    _install_gateway(narrator, with_lore=True)
    canon = WorldCanon()

    result = asyncio.run(
        narrator.generate(
            events=_events_three_scenes(),
            world_data=_world_data(),
            run_id="narrator_with_canon",
            canon=canon,
        )
    )
    assert result.status in {"complete", "partial"}
    assert result.story_lore is not None


def test_narrator_lore_in_result() -> None:
    narrator = SequentialNarrator(config=PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=3))
    _install_gateway(narrator, with_lore=True)

    result = asyncio.run(
        narrator.generate(
            events=_events_three_scenes(),
            world_data=_world_data(),
            run_id="narrator_lore_result",
        )
    )

    assert result.story_lore is not None
    assert len(result.story_lore.scene_lore) == result.scenes_generated
    assert len(result.story_lore.all_canon_facts) >= 1
    assert len(result.story_lore.all_texture_facts) >= 1


def test_narrator_texture_accumulation() -> None:
    narrator = SequentialNarrator(config=PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=3))
    statement = "Victor taps his ring finger before pressing a question."
    creative_prompts = _install_gateway(
        narrator,
        with_lore=True,
        first_scene_texture_statement=statement,
    )

    asyncio.run(
        narrator.generate(
            events=_events_three_scenes(),
            world_data=_world_data(),
            run_id="narrator_texture_accumulation",
        )
    )

    assert len(creative_prompts) >= 3
    assert "<established_details>" in creative_prompts[2]
    assert statement in creative_prompts[2]


def test_narrator_backward_compat_no_lore() -> None:
    narrator = SequentialNarrator(config=PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=3))
    _install_gateway(narrator, with_lore=False)

    result = asyncio.run(
        narrator.generate(
            events=_events_three_scenes(),
            world_data=_world_data(),
            run_id="narrator_no_lore",
        )
    )

    assert result.story_lore is not None
    assert len(result.story_lore.scene_lore) == result.scenes_generated
    assert all(len(scene.canon_facts) == 0 for scene in result.story_lore.scene_lore)
    assert all(len(scene.texture_facts) == 0 for scene in result.story_lore.scene_lore)


def test_texture_commit_keys_do_not_collide_across_runs() -> None:
    narrator = SequentialNarrator(config=PipelineConfig(checkpoint_enabled=False, phase2_events_per_chunk=3))
    _install_gateway(narrator, with_lore=True)
    canon = WorldCanon()

    asyncio.run(
        narrator.generate(
            events=_events_single_scene(),
            world_data=_world_data(),
            run_id="run_a",
            canon=canon,
        )
    )
    _install_gateway(narrator, with_lore=True)
    asyncio.run(
        narrator.generate(
            events=_events_single_scene(),
            world_data=_world_data(),
            run_id="run_b",
            canon=canon,
        )
    )

    assert "run_a__tf_0_0" in canon.texture
    assert "run_b__tf_0_0" in canon.texture
    assert canon.texture["run_a__tf_0_0"].statement == canon.texture["run_b__tf_0_0"].statement
