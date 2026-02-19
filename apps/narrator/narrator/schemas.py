from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field


class Character(BaseModel):
    id: str
    name: str
    role: str
    location: str
    alive: bool = True


class LocationInfo(BaseModel):
    id: str
    name: str
    description: str
    characters_present: list[str]


class Choice(BaseModel):
    id: str
    label: str
    description: str


class Violation(BaseModel):
    type: str
    description: str
    severity: Literal["hard", "soft"] = "hard"
    event_index: int | None = None
    event_id: str | None = None


class GuardResult(BaseModel):
    valid: bool
    violations: list[Violation] = Field(default_factory=list)
    repaired: bool = False


class ProposalEvent(BaseModel):
    id: str | None = None
    actor_id: str
    action: str
    event_type: str = "social_move"
    target_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    reveals_secret_ids: list[str] = Field(default_factory=list)
    required_knowledge_ids: list[str] = Field(default_factory=list)
    regresses_secret_ids: list[str] = Field(default_factory=list)
    pivot_actor_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SceneProposal(BaseModel):
    scene_id: str
    prose: str
    location: str
    characters_present: list[str] = Field(default_factory=list)
    choices: list[Choice] = Field(default_factory=list)
    events: list[ProposalEvent] = Field(default_factory=list)
    transition_note: str | None = None


class GuardStats(BaseModel):
    proposals_generated: int = 0
    violations_caught: int = 0
    violations_by_type: dict[str, int] = Field(default_factory=dict)
    repairs_applied: int = 0

    def add_violations(self, violations: list[Violation]) -> None:
        if not violations:
            return
        self.violations_caught += len(violations)
        counts = defaultdict(int, self.violations_by_type)
        for violation in violations:
            counts[violation.type] += 1
        self.violations_by_type = dict(counts)


class SceneResponse(BaseModel):
    scene_id: str
    prose: str
    location: str
    characters_present: list[str]
    choices: list[Choice]
    turn: int
    guard_stats: GuardStats


class GameState(BaseModel):
    session_id: str
    turn: int
    characters: list[Character]
    locations: list[LocationInfo]
    guard_stats: GuardStats
    scenes_so_far: int


class NewGameRequest(BaseModel):
    mode: Literal["mock", "live"] | None = None


class ChooseRequest(BaseModel):
    choice_id: str


class NewGameResponse(BaseModel):
    session_id: str
    state: GameState
    scene: SceneResponse


class LogEntry(BaseModel):
    turn: int
    attempt: int
    mode: str
    choice_id: str
    proposal_scene_id: str
    valid: bool
    repaired: bool
    violations: list[Violation] = Field(default_factory=list)
    note: str | None = None


class GameLogResponse(BaseModel):
    session_id: str
    entries: list[LogEntry]
