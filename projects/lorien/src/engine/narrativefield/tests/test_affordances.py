from __future__ import annotations

from narrativefield.schema.events import Event, EventMetrics, EventType
from narrativefield.simulation import decision_engine
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.tick_loop import build_perception


def test_kitchen_refill_action_generated() -> None:
    world = create_dinner_party_world()
    agent = world.agents["lydia"]
    agent.location = "kitchen"

    perception = build_perception(agent, world)
    candidates = decision_engine.generate_candidate_actions(agent, perception, world)

    assert any(a.action_type == EventType.PHYSICAL and "refill" in a.content.lower() for a in candidates)


def test_foyer_check_phone_action_generated() -> None:
    world = create_dinner_party_world()
    agent = world.agents["victor"]
    agent.location = "foyer"

    perception = build_perception(agent, world)
    candidates = decision_engine.generate_candidate_actions(agent, perception, world)

    assert any(a.action_type == EventType.INTERNAL and a.content == "Check phone" for a in candidates)


def test_social_inertia_discourages_leaving_populated_table() -> None:
    world = create_dinner_party_world()
    agent = world.agents["thorne"]

    # Make the agent "not secretive" for this test so pending-drama doesn't dominate.
    agent.goals.secrecy = 0.0
    agent.pacing.stress = 0.0

    kitchen = world.locations["kitchen"]
    score_full_table = decision_engine.score_move(agent, kitchen, world)

    # If everyone else is gone, inertia should be lower and leaving should score higher.
    for aid, other in world.agents.items():
        if aid != agent.id:
            other.location = "departed"

    score_empty_table = decision_engine.score_move(agent, kitchen, world)

    assert score_empty_table > score_full_table


def test_kitchen_and_foyer_outscore_bathroom_in_low_stress_case() -> None:
    world = create_dinner_party_world()
    agent = world.agents["lydia"]

    agent.goals.secrecy = 0.0
    agent.goals.autonomy = 1.0
    agent.pacing.stress = 0.0
    agent.alcohol_level = 0.0

    kitchen = world.locations["kitchen"]
    foyer = world.locations["foyer"]
    bathroom = world.locations["bathroom"]

    score_kitchen = decision_engine.score_move(agent, kitchen, world)
    score_foyer = decision_engine.score_move(agent, foyer, world)
    score_bathroom = decision_engine.score_move(agent, bathroom, world)

    assert score_kitchen > score_bathroom
    assert score_foyer > score_bathroom


def test_bathroom_penalty_increases_with_recent_visits() -> None:
    world = create_dinner_party_world()
    agent = world.agents["diana"]

    agent.goals.secrecy = 0.0
    agent.pacing.stress = 0.0

    bathroom = world.locations["bathroom"]
    base_score = decision_engine.score_move(agent, bathroom, world)

    # Simulate repeated bathroom trips in the recent event log.
    for i in range(3):
        world.event_log.append(
            Event(
                id=f"m{i}",
                sim_time=float(i),
                tick_id=i,
                order_in_tick=0,
                type=EventType.SOCIAL_MOVE,
                source_agent=agent.id,
                target_agents=[],
                location_id="dining_table",
                causal_links=[],
                deltas=[],
                description="Move",
                content_metadata={"destination": "bathroom"},
                metrics=EventMetrics(),
            )
        )

    penalized_score = decision_engine.score_move(agent, bathroom, world)

    assert penalized_score < base_score
