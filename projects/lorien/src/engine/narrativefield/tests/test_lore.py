from __future__ import annotations

from narrativefield.schema.canon import CanonTexture, WorldCanon
from narrativefield.schema.lore import CanonFact, SceneLoreUpdates, StoryLore, TextureFact


def test_canon_fact_creation() -> None:
    fact = CanonFact(
        id="cf_0_0",
        statement="Victor learned the secret at dinner.",
        source_event_ids=["evt_001"],
        entity_refs=["victor", "claim_secret_01"],
        scene_index=0,
    )
    assert fact.id == "cf_0_0"
    assert fact.source_event_ids == ["evt_001"]


def test_texture_fact_creation() -> None:
    fact = TextureFact(
        id="tf_0_0",
        statement="Victor taps his ring finger when cornered.",
        entity_refs=["victor"],
        detail_type="gesture",
        scene_index=0,
        confidence=0.9,
    )
    assert fact.detail_type == "gesture"
    assert fact.confidence == 0.9


def test_scene_lore_updates_roundtrip() -> None:
    scene = SceneLoreUpdates(
        scene_index=2,
        canon_facts=[
            CanonFact(
                id="cf_2_0",
                statement="Thorne confronted Marcus.",
                source_event_ids=["evt_010"],
                entity_refs=["thorne", "marcus"],
                scene_index=2,
            )
        ],
        texture_facts=[
            TextureFact(
                id="tf_2_0",
                statement="The chandelier leaves the far table edge in shadow.",
                entity_refs=["dining_table"],
                detail_type="setting",
                scene_index=2,
            )
        ],
    )
    parsed = SceneLoreUpdates.from_dict(scene.to_dict())
    assert parsed.to_dict() == scene.to_dict()


def test_story_lore_roundtrip() -> None:
    story = StoryLore(
        scene_lore=[
            SceneLoreUpdates(
                scene_index=0,
                canon_facts=[
                    CanonFact(
                        id="cf_0_0",
                        statement="Opening scene fact.",
                        source_event_ids=["evt_001"],
                        entity_refs=["victor"],
                        scene_index=0,
                    )
                ],
                texture_facts=[],
            )
        ]
    )
    parsed = StoryLore.from_dict(story.to_dict())
    assert parsed.to_dict() == story.to_dict()


def test_story_lore_aggregation() -> None:
    story = StoryLore(
        scene_lore=[
            SceneLoreUpdates(
                scene_index=0,
                canon_facts=[
                    CanonFact(
                        id="cf_0_0",
                        statement="Fact A",
                        source_event_ids=["evt_a"],
                        entity_refs=[],
                        scene_index=0,
                    )
                ],
                texture_facts=[
                    TextureFact(
                        id="tf_0_0",
                        statement="Texture A",
                        entity_refs=[],
                        detail_type="habit",
                        scene_index=0,
                    )
                ],
            ),
            SceneLoreUpdates(
                scene_index=1,
                canon_facts=[
                    CanonFact(
                        id="cf_1_0",
                        statement="Fact B",
                        source_event_ids=["evt_b"],
                        entity_refs=[],
                        scene_index=1,
                    )
                ],
                texture_facts=[],
            ),
        ]
    )
    assert len(story.all_canon_facts) == 2
    assert len(story.all_texture_facts) == 1


def test_canon_texture_roundtrip() -> None:
    canon = WorldCanon(
        texture={
            "run_abc__tf_0_0": CanonTexture(
                id="run_abc__tf_0_0",
                statement="Elena aligns the silverware before difficult topics.",
                entity_refs=["elena", "dining_table"],
                detail_type="habit",
                source_story_id="run_abc",
                source_scene_index=0,
                committed_at_canon_version=3,
            )
        }
    )
    parsed = WorldCanon.from_dict(canon.to_dict())
    assert parsed.to_dict() == canon.to_dict()


def test_world_canon_backward_compat_without_texture() -> None:
    legacy = {
        "world_id": "legacy_world",
        "canon_version": 1,
        "last_tick": 10,
        "last_event_id": "evt_010",
        "location_memory": {},
        "artifacts": {},
        "factions": {},
        "institutions": {},
        "claim_states": {},
    }
    parsed = WorldCanon.from_dict(legacy)
    assert parsed.texture == {}
