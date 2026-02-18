from __future__ import annotations

import xml.etree.ElementTree as ET

from narrativefield.schema.agents import (
    AgentState,
    BeliefState,
    CharacterFlaw,
    FlawType,
    GoalVector,
    PacingState,
    RelationshipState,
)
from narrativefield.schema.world import Location, SecretDefinition, WorldDefinition
from narrativefield.storyteller.lorebook import Lorebook


def _mock_world() -> tuple[WorldDefinition, list[AgentState], list[SecretDefinition]]:
    loc = Location(
        id="dining_table",
        name="Dining Table",
        privacy=0.1,
        capacity=6,
        adjacent=[],
        overhear_from=[],
        overhear_probability=0.0,
        description="The social center of the evening.",
    )

    secret = SecretDefinition(
        id="secret_victor_01",
        holder=["victor"],
        about="victor",
        content_type="investigation",
        description="Victor is secretly investigating Marcus.",
        truth_value=True,
        initial_knowers=["victor"],
        initial_suspecters=[],
        dramatic_weight=0.7,
        reveal_consequences="Marcus reacts defensively.",
    )

    world = WorldDefinition(
        id="w1",
        name="Test World",
        description="A tiny mock world for lorebook tests.",
        sim_duration_minutes=60.0,
        ticks_per_minute=2.0,
        locations={"dining_table": loc},
        secrets={"secret_victor_01": secret},
    )

    victor = AgentState(
        id="victor",
        name="Victor",
        location="dining_table",
        goals=GoalVector(status=0.6, secrecy=0.9, truth_seeking=0.8, closeness={"diana": 0.4}),
        flaws=[
            CharacterFlaw(
                flaw_type=FlawType.OBSESSION,
                strength=0.9,
                trigger="uncertainty",
                effect="overanalyze",
                description="He fixates on hidden patterns and can't let go.",
            )
        ],
        pacing=PacingState(),
        emotional_state={"anger": 0.2, "suspicion": 0.5},
        relationships={"diana": RelationshipState(trust=0.6, affection=0.2, obligation=0.0)},
        beliefs={"secret_victor_01": BeliefState.BELIEVES_TRUE},
        alcohol_level=0.0,
        commitments=[],
    )

    diana = AgentState(
        id="diana",
        name="Diana",
        location="dining_table",
        goals=GoalVector(safety=0.7, loyalty=0.7, closeness={"victor": 0.5}),
        flaws=[
            CharacterFlaw(
                flaw_type=FlawType.LOYALTY,
                strength=0.7,
                trigger="conflict",
                effect="protect_others",
                description="She protects others even when it costs her.",
            )
        ],
        pacing=PacingState(),
        emotional_state={"fear": 0.2},
        relationships={"victor": RelationshipState(trust=0.4, affection=0.5, obligation=0.2)},
        beliefs={"secret_victor_01": BeliefState.UNKNOWN},
        alcohol_level=0.0,
        commitments=[],
    )

    return world, [victor, diana], [secret]


def test_get_context_for_scene_xml_includes_only_requested_characters() -> None:
    world, agents, secrets = _mock_world()
    lb = Lorebook(world, agents=agents, secrets=secrets)

    xml = lb.get_context_for_scene(character_ids=["victor"], location_id="dining_table")
    root = ET.fromstring(xml)
    assert root.tag == "lorebook"

    chars = root.findall("character")
    assert len(chars) == 1
    assert chars[0].attrib.get("id") == "victor"

    # Ensure a non-requested character is excluded.
    assert all(c.attrib.get("id") != "diana" for c in chars)

    # Budget check: rough estimate 1 token ~= 4 chars.
    assert (len(xml) // 4) < 800


def test_get_full_cast_returns_all_characters_and_is_parseable() -> None:
    world, agents, secrets = _mock_world()
    lb = Lorebook(world, agents=agents, secrets=secrets)

    cast_xml = lb.get_full_cast()
    root = ET.fromstring(cast_xml)
    assert root.tag == "full_cast"

    chars = root.findall("character")
    ids = {c.attrib.get("id") for c in chars}
    assert ids == {"victor", "diana"}
