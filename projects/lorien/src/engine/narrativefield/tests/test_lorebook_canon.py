from __future__ import annotations

from narrativefield.schema.agents import (
    AgentState,
    BeliefState,
    CharacterFlaw,
    FlawType,
    GoalVector,
    PacingState,
    RelationshipState,
)
from narrativefield.schema.canon import CanonTexture, LocationMemory, WorldCanon
from narrativefield.schema.world import Location, SecretDefinition, WorldDefinition
from narrativefield.storyteller.lorebook import Lorebook


def _world_agents_secrets() -> tuple[WorldDefinition, list[AgentState], list[SecretDefinition]]:
    world = WorldDefinition(
        id="w",
        name="World",
        description="test",
        sim_duration_minutes=120.0,
        ticks_per_minute=2.0,
        locations={
            "dining_table": Location(
                id="dining_table",
                name="Dining Table",
                privacy=0.1,
                capacity=6,
                adjacent=[],
                overhear_from=[],
                overhear_probability=0.0,
                description="The center of conversation.",
            )
        },
        secrets={
            "secret_01": SecretDefinition(
                id="secret_01",
                holder=["victor"],
                about="victor",
                content_type="investigation",
                description="Victor is investigating Marcus.",
                truth_value=True,
                initial_knowers=["victor"],
                initial_suspecters=[],
                dramatic_weight=0.7,
                reveal_consequences="Conflict rises.",
            )
        },
    )
    victor = AgentState(
        id="victor",
        name="Victor",
        location="dining_table",
        goals=GoalVector(truth_seeking=0.9),
        flaws=[
            CharacterFlaw(
                flaw_type=FlawType.OBSESSION,
                strength=0.8,
                trigger="ambiguity",
                effect="fixate",
                description="Fixates on unresolved details.",
            )
        ],
        pacing=PacingState(),
        emotional_state={},
        relationships={"diana": RelationshipState(trust=0.3)},
        beliefs={"claim_affair": BeliefState.BELIEVES_TRUE},
        alcohol_level=0.0,
        commitments=[],
    )
    diana = AgentState(
        id="diana",
        name="Diana",
        location="dining_table",
        goals=GoalVector(safety=0.7),
        flaws=[
            CharacterFlaw(
                flaw_type=FlawType.DENIAL,
                strength=0.6,
                trigger="pressure",
                effect="deflect",
                description="Avoids direct admission.",
            )
        ],
        pacing=PacingState(),
        emotional_state={},
        relationships={"victor": RelationshipState(trust=0.4)},
        beliefs={"claim_affair": BeliefState.UNKNOWN},
        alcohol_level=0.0,
        commitments=[],
    )
    return world, [victor, diana], list(world.secrets.values())


def test_lorebook_without_canon() -> None:
    world, agents, secrets = _world_agents_secrets()
    lorebook = Lorebook(world, agents, secrets)
    assert lorebook.get_canon_context_for_scene(["victor"], "dining_table") == ""


def test_lorebook_with_canon() -> None:
    world, agents, secrets = _world_agents_secrets()
    canon = WorldCanon()
    lorebook = Lorebook(world, agents, secrets, canon=canon)
    assert lorebook is not None


def test_canon_context_with_residue() -> None:
    world, agents, secrets = _world_agents_secrets()
    canon = WorldCanon(
        location_memory={
            "dining_table": LocationMemory(
                tension_residue=0.8,
                notable_event_ids=["evt_1"],
                visit_count=4,
                last_event_tick=9,
            )
        }
    )
    lorebook = Lorebook(world, agents, secrets, canon=canon)
    xml = lorebook.get_canon_context_for_scene(["victor"], "dining_table")
    assert "<world_memory>" in xml
    assert "<location_memory " in xml
    assert 'tension_residue="0.80"' in xml


def test_canon_context_with_beliefs() -> None:
    world, agents, secrets = _world_agents_secrets()
    canon = WorldCanon(
        claim_states={
            "claim_affair": {
                "victor": "believes_true",
                "diana": "unknown",
            }
        }
    )
    lorebook = Lorebook(world, agents, secrets, canon=canon)
    xml = lorebook.get_canon_context_for_scene(["victor", "diana"], "dining_table")
    assert "<inherited_knowledge " in xml
    assert "claim_affair: believes_true" in xml
    assert "diana already knows" not in xml.lower()


def test_canon_context_with_texture() -> None:
    world, agents, secrets = _world_agents_secrets()
    canon = WorldCanon(
        texture={
            "run_a__tf_0_0": CanonTexture(
                id="run_a__tf_0_0",
                statement="Victor straightens his cuff before difficult questions.",
                entity_refs=["victor"],
                detail_type="habit",
            ),
            "run_a__tf_0_1": CanonTexture(
                id="run_a__tf_0_1",
                statement="The dining table runner is frayed at one corner.",
                entity_refs=["dining_table"],
                detail_type="setting",
            ),
        }
    )
    lorebook = Lorebook(world, agents, secrets, canon=canon)
    xml = lorebook.get_canon_context_for_scene(["victor"], "dining_table")
    assert "<established_detail " in xml
    assert "frayed at one corner" in xml


def test_canon_context_empty_canon() -> None:
    world, agents, secrets = _world_agents_secrets()
    lorebook = Lorebook(world, agents, secrets, canon=WorldCanon())
    assert lorebook.get_canon_context_for_scene(["victor"], "dining_table") == ""


def test_canon_context_low_residue_excluded() -> None:
    world, agents, secrets = _world_agents_secrets()
    canon = WorldCanon(
        location_memory={
            "dining_table": LocationMemory(
                tension_residue=0.02,
                notable_event_ids=["evt_1"],
                visit_count=2,
                last_event_tick=2,
            )
        }
    )
    lorebook = Lorebook(world, agents, secrets, canon=canon)
    xml = lorebook.get_canon_context_for_scene(["victor"], "dining_table")
    assert "<location_memory " not in xml
