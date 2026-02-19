from __future__ import annotations

from dataclasses import dataclass

from narrativefield.schema.agents import AgentState, BeliefState
from narrativefield.schema.world import Location
from narrativefield.simulation.scenarios import create_dinner_party_world
from narrativefield.simulation.types import WorldState


ROLE_MAP = {
    "thorne": "Host, investor, and keeper of the evening's fragile order.",
    "elena": "Host and social architect, elegant but carefully guarded.",
    "marcus": "Business partner with practiced charm and concealed pressure.",
    "diana": "Family friend with divided loyalties and private debt.",
    "lydia": "Analytical observer with a talent for reading tension.",
    "victor": "Journalist guest, attentive to cracks beneath the performance.",
}


@dataclass(slots=True)
class ScenarioCharacter:
    id: str
    name: str
    role: str
    location: str
    alive: bool
    known_secrets: set[str]


@dataclass(slots=True)
class ScenarioBundle:
    scenario_id: str
    world_state: WorldState
    characters: dict[str, ScenarioCharacter]
    locations: dict[str, Location]
    adjacency: dict[str, set[str]]
    all_secret_ids: set[str]


def _known_secret_ids(agent: AgentState) -> set[str]:
    return {
        secret_id
        for secret_id, belief in agent.beliefs.items()
        if belief == BeliefState.BELIEVES_TRUE
    }


def load_dinner_party_scenario() -> ScenarioBundle:
    world = create_dinner_party_world()

    characters: dict[str, ScenarioCharacter] = {}
    for agent_id, agent in world.agents.items():
        characters[agent_id] = ScenarioCharacter(
            id=agent_id,
            name=agent.name,
            role=ROLE_MAP.get(agent_id, "Guest at the dinner party."),
            location=agent.location,
            alive=True,
            known_secrets=_known_secret_ids(agent),
        )

    adjacency: dict[str, set[str]] = {}
    for loc_id, location in world.definition.locations.items():
        adjacency[loc_id] = set(location.adjacent)

    return ScenarioBundle(
        scenario_id=world.definition.id,
        world_state=world,
        characters=characters,
        locations=world.definition.locations,
        adjacency=adjacency,
        all_secret_ids=set(world.definition.secrets.keys()),
    )
