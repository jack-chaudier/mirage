from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from narrativefield.simulation.tick_loop import SimulationConfig, run_simulation

from .schemas import Character, GuardStats, LocationInfo, SceneProposal, SceneResponse
from .scenarios.dinner_party import ScenarioBundle, ScenarioCharacter, load_dinner_party_scenario

if TYPE_CHECKING:
    from .schemas import Violation


@dataclass(slots=True)
class RuntimeCharacter:
    id: str
    name: str
    role: str
    location: str
    alive: bool = True
    known_secrets: set[str] = field(default_factory=set)


@dataclass(slots=True)
class RuntimeWorldState:
    scenario_id: str
    world_state: object
    characters: dict[str, RuntimeCharacter]
    locations: dict[str, object]
    adjacency: dict[str, set[str]]
    all_secret_ids: set[str]
    revealed_secret_ids: set[str] = field(default_factory=set)
    recent_events: list[str] = field(default_factory=list)
    active_pivot_actor: str | None = None
    current_scene_id: str = "scene_0"
    current_location: str = "dining_table"
    turn: int = 0


class NarrativeEngine:
    """Simulation-backed game driver for the interactive narrative demo."""

    def __init__(self, scenario: ScenarioBundle):
        self._rng = Random(42)
        characters = {
            c.id: RuntimeCharacter(
                id=c.id,
                name=c.name,
                role=c.role,
                location=c.location,
                alive=c.alive,
                known_secrets=set(c.known_secrets),
            )
            for c in scenario.characters.values()
        }
        self.state = RuntimeWorldState(
            scenario_id=scenario.scenario_id,
            world_state=scenario.world_state,
            characters=characters,
            locations=dict(scenario.locations),
            adjacency=dict(scenario.adjacency),
            all_secret_ids=set(scenario.all_secret_ids),
            revealed_secret_ids=set(),
            current_location="dining_table",
        )

    @classmethod
    def from_dinner_party(cls) -> "NarrativeEngine":
        return cls(load_dinner_party_scenario())

    def get_character(self, character_id: str) -> RuntimeCharacter | None:
        return self.state.characters.get(character_id)

    def character_ids_at_location(self, location_id: str) -> list[str]:
        return [c.id for c in self.state.characters.values() if c.location == location_id and c.alive]

    def are_adjacent(self, location_a: str, location_b: str) -> bool:
        if location_a == location_b:
            return True
        return location_b in self.state.adjacency.get(location_a, set())

    def is_secret_revealed(self, secret_id: str) -> bool:
        return secret_id in self.state.revealed_secret_ids

    def get_world_context(self) -> str:
        lines: list[str] = []
        lines.append(f"Scenario: {self.state.scenario_id}")
        lines.append(f"Turn: {self.state.turn}")
        lines.append(f"Current scene: {self.state.current_scene_id}")
        lines.append(f"Current location focus: {self.state.current_location}")
        lines.append(f"Active pivot actor: {self.state.active_pivot_actor or 'none'}")
        revealed = sorted(self.state.revealed_secret_ids)
        lines.append(f"Revealed secrets: {', '.join(revealed) if revealed else 'none'}")

        lines.append("\nCharacter states:")
        for character in sorted(self.state.characters.values(), key=lambda c: c.id):
            knows = ", ".join(sorted(character.known_secrets)) if character.known_secrets else "none"
            lines.append(
                f"- {character.id} ({character.name}) | alive={character.alive} | "
                f"location={character.location} | knows={knows}"
            )

        lines.append("\nLocation occupancy:")
        for location_id in sorted(self.state.locations.keys()):
            present = self.character_ids_at_location(location_id)
            lines.append(f"- {location_id}: {', '.join(present) if present else 'none'}")

        lines.append("\nRecent events:")
        recent = self.state.recent_events[-8:]
        if not recent:
            lines.append("- none")
        else:
            for event_line in recent:
                lines.append(f"- {event_line}")

        lines.append(
            "\nOutput requirements: return structured JSON scene events followed by literary prose. "
            "Do not reference secrets a character does not know."
        )
        return "\n".join(lines)

    def validate_proposal(self, proposal: SceneProposal) -> list["Violation"]:
        from .guard import MirageGuard

        guard = MirageGuard()
        result = guard.validate_scene(proposal=proposal, engine=self)
        return result.violations

    def advance_simulation(self, ticks: int = 2) -> None:
        world = self.state.world_state
        start_events = len(world.event_log)
        ticks = max(1, int(ticks))
        cfg = SimulationConfig(
            tick_limit=ticks,
            event_limit=max(start_events + (ticks * 6), start_events + 1),
            max_sim_time=world.definition.sim_duration_minutes,
            snapshot_interval_events=world.definition.snapshot_interval,
        )
        run_simulation(world=world, rng=self._rng, cfg=cfg)
        new_events = world.event_log[start_events:]

        # Sync projected character locations from the Lorien world state.
        for character_id, runtime_char in self.state.characters.items():
            sim_agent = world.agents.get(character_id)
            if sim_agent is not None:
                runtime_char.location = sim_agent.location

        for event in new_events[:4]:
            summary = (
                f"{event.source_agent} {event.type.value} at {event.location_id}"
                f" (tick={event.tick_id})"
            )
            self.state.recent_events.append(summary)

    def _apply_event(self, event) -> None:
        actor = self.get_character(event.actor_id)
        if actor is None:
            return

        resolved_location = event.location_id or actor.location
        lower_action = event.action.lower()

        is_move = event.event_type.lower() in {"move", "physical"}
        if not is_move and resolved_location != actor.location:
            move_verbs = ("move", "go to", "goes to", "walk", "step", "head to",
                          "heads to", "exit", "leave", "enter", "retreat", "follow")
            is_move = any(verb in lower_action for verb in move_verbs)
        if is_move:
            actor.location = resolved_location

        for secret_id in event.reveals_secret_ids:
            self.state.revealed_secret_ids.add(secret_id)
            for character in self.state.characters.values():
                character.known_secrets.add(secret_id)

        if event.pivot_actor_id:
            self.state.active_pivot_actor = event.pivot_actor_id

        summary = (
            f"{actor.name} ({actor.id}) {event.action}"
            f" @ {resolved_location}"
            f" targets={','.join(event.target_ids) if event.target_ids else 'none'}"
        )
        self.state.recent_events.append(summary)

    def commit_scene(self, proposal: SceneProposal, turn: int, guard_stats: GuardStats) -> SceneResponse:
        self.advance_simulation(ticks=2)

        # Set characters_present to the scene's focal location FIRST,
        # then apply events — physical/move events override the default.
        if proposal.location:
            self.state.current_location = proposal.location
            for char_id in proposal.characters_present:
                character = self.get_character(char_id)
                if character and character.alive:
                    character.location = proposal.location

        for event in proposal.events:
            self._apply_event(event)

        self.state.current_scene_id = proposal.scene_id
        self.state.turn = turn

        character_names = [
            self.state.characters[cid].name
            for cid in proposal.characters_present
            if cid in self.state.characters
        ]

        return SceneResponse(
            scene_id=proposal.scene_id,
            prose=proposal.prose,
            location=proposal.location,
            characters_present=character_names,
            choices=proposal.choices,
            turn=turn,
            guard_stats=guard_stats.model_copy(deep=True),
        )

    def public_characters(self) -> list[Character]:
        return [
            Character(
                id=character.id,
                name=character.name,
                role=character.role,
                location=character.location,
                alive=character.alive,
            )
            for character in sorted(self.state.characters.values(), key=lambda c: c.name)
        ]

    def public_locations(self) -> list[LocationInfo]:
        infos: list[LocationInfo] = []
        for location_id in sorted(self.state.locations.keys()):
            location = self.state.locations[location_id]
            present = [
                character.name
                for character in sorted(self.state.characters.values(), key=lambda c: c.name)
                if character.location == location_id and character.alive
            ]
            infos.append(
                LocationInfo(
                    id=location_id,
                    name=getattr(location, "name", location_id),
                    description=getattr(location, "description", ""),
                    characters_present=present,
                )
            )
        return infos
