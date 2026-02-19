from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import ChooseRequest, GameLogResponse, GameState, NewGameRequest, NewGameResponse, SceneResponse
from .session import build_session_manager


def create_app() -> FastAPI:
    app = FastAPI(title="Narrator: Mirage-Guarded Interactive Fiction", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = build_session_manager()

    @app.post("/api/game/new", response_model=NewGameResponse)
    async def new_game(payload: NewGameRequest | None = Body(default=None)) -> NewGameResponse:
        mode = payload.mode if payload is not None else None
        try:
            return await manager.create_session(mode=mode)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to create game session: {exc}") from exc

    @app.post("/api/game/{session_id}/choose", response_model=SceneResponse)
    async def choose(session_id: str, payload: ChooseRequest) -> SceneResponse:
        try:
            return await manager.choose(session_id=session_id, choice_id=payload.choice_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Choice processing failed: {exc}") from exc

    @app.get("/api/game/{session_id}/state", response_model=GameState)
    async def state(session_id: str) -> GameState:
        try:
            return manager.get_state(session_id=session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/game/{session_id}/log", response_model=GameLogResponse)
    async def log(session_id: str) -> GameLogResponse:
        try:
            return manager.get_log(session_id=session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("narrator.server:app", host="0.0.0.0", port=8000, reload=False)
