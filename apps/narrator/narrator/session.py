from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from uuid import uuid4

from .engine import NarrativeEngine
from .generator import SceneGenerator
from .guard import MirageGuard
from .schemas import (
    GameLogResponse,
    GameState,
    GuardStats,
    LogEntry,
    NewGameResponse,
    SceneResponse,
)


@dataclass(slots=True)
class SessionState:
    session_id: str
    mode: str
    engine: NarrativeEngine
    generator: SceneGenerator
    guard: MirageGuard
    guard_stats: GuardStats = field(default_factory=GuardStats)
    turn: int = 0
    scene_history: list[SceneResponse] = field(default_factory=list)
    log_entries: list[LogEntry] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionManager:
    def __init__(self, *, default_mode: str = "mock", max_repairs: int = 2):
        self.default_mode = default_mode
        self.max_repairs = max_repairs
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, mode: str | None = None) -> NewGameResponse:
        chosen_mode = mode or self.default_mode
        session_id = str(uuid4())
        state = SessionState(
            session_id=session_id,
            mode=chosen_mode,
            engine=NarrativeEngine.from_dinner_party(),
            generator=SceneGenerator(mode=chosen_mode),
            guard=MirageGuard(),
        )

        async with self._lock:
            self._sessions[session_id] = state

        initial_scene = await self._run_turn(state=state, choice_id="start")
        return NewGameResponse(
            session_id=session_id,
            state=self._to_game_state(state),
            scene=initial_scene,
        )

    async def choose(self, session_id: str, choice_id: str) -> SceneResponse:
        state = self._get_or_raise(session_id)
        async with state.lock:
            if not state.scene_history:
                raise ValueError("Session has no active scene.")
            current_scene = state.scene_history[-1]
            choice_ids = {choice.id for choice in current_scene.choices}
            if not choice_ids:
                raise ValueError("This scene has ended. Start a new game.")
            if choice_id not in choice_ids:
                raise ValueError(f"Unknown choice_id '{choice_id}' for current scene.")
            return await self._run_turn(state=state, choice_id=choice_id)

    def get_state(self, session_id: str) -> GameState:
        state = self._get_or_raise(session_id)
        return self._to_game_state(state)

    def get_log(self, session_id: str) -> GameLogResponse:
        state = self._get_or_raise(session_id)
        return GameLogResponse(session_id=session_id, entries=list(state.log_entries))

    def _get_or_raise(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"Unknown session_id '{session_id}'.")
        return state

    async def _run_turn(self, *, state: SessionState, choice_id: str) -> SceneResponse:
        next_turn = state.turn + 1
        accepted = None
        max_attempts = max(1, self.max_repairs + 1)

        for attempt in range(1, max_attempts + 1):
            proposal = await state.generator.generate_scene(
                engine=state.engine,
                current_scene_id=state.engine.state.current_scene_id,
                choice_id=choice_id,
                turn=next_turn,
                attempt=attempt,
            )

            state.guard_stats.proposals_generated += 1
            guard_result = state.guard.validate_scene(proposal=proposal, engine=state.engine)
            state.guard_stats.add_violations(guard_result.violations)

            state.log_entries.append(
                LogEntry(
                    turn=next_turn,
                    attempt=attempt,
                    mode=state.mode,
                    choice_id=choice_id,
                    proposal_scene_id=proposal.scene_id,
                    valid=guard_result.valid,
                    repaired=False,
                    violations=list(guard_result.violations),
                )
            )

            if guard_result.valid:
                accepted = proposal
                break

            repaired, repaired_result = state.guard.attempt_repair(proposal=proposal, engine=state.engine)
            if repaired_result.valid and repaired_result.repaired:
                state.guard_stats.repairs_applied += 1
                state.log_entries.append(
                    LogEntry(
                        turn=next_turn,
                        attempt=attempt,
                        mode=state.mode,
                        choice_id=choice_id,
                        proposal_scene_id=repaired.scene_id,
                        valid=True,
                        repaired=True,
                        violations=list(guard_result.violations),
                        note="auto_repair_applied",
                    )
                )
                accepted = repaired
                break

        if accepted is None:
            accepted = state.generator.safe_fallback_scene(
                engine=state.engine,
                current_scene_id=state.engine.state.current_scene_id,
                turn=next_turn,
            )
            state.log_entries.append(
                LogEntry(
                    turn=next_turn,
                    attempt=max_attempts,
                    mode=state.mode,
                    choice_id=choice_id,
                    proposal_scene_id=accepted.scene_id,
                    valid=True,
                    repaired=True,
                    violations=[],
                    note="fallback_scene_used",
                )
            )

        response = state.engine.commit_scene(
            proposal=accepted,
            turn=next_turn,
            guard_stats=state.guard_stats,
        )

        state.turn = next_turn
        state.scene_history.append(response)
        return response

    def _to_game_state(self, state: SessionState) -> GameState:
        return GameState(
            session_id=state.session_id,
            turn=state.turn,
            characters=state.engine.public_characters(),
            locations=state.engine.public_locations(),
            guard_stats=state.guard_stats.model_copy(deep=True),
            scenes_so_far=len(state.scene_history),
        )


def build_session_manager() -> SessionManager:
    mode = os.getenv("LLM_MODE", "mock").strip().lower() or "mock"
    max_repairs = int(os.getenv("NARRATOR_MAX_REPAIRS", "2"))
    return SessionManager(default_mode=mode, max_repairs=max_repairs)
