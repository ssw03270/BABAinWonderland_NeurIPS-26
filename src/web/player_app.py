from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.web.common import html_file_response, mount_shared_static
from src.web.player_sessions import DEFAULT_RESET_SEED, PlayerSession


class PlayerActionRequest(BaseModel):
    action: str


class PlayerResetRequest(BaseModel):
    seed: int = DEFAULT_RESET_SEED


class PlayerSelectMapRequest(BaseModel):
    scenarioType: str | None = None


class PlayerRecordingRequest(BaseModel):
    enabled: bool


def create_player_app(
    *,
    env_id: str | None,
    scenario_type: str | None,
    seed: int,
    autosave_enabled: bool,
) -> FastAPI:
    session = PlayerSession.create(
        requested_env_id=env_id,
        requested_scenario_type=scenario_type,
        seed=int(seed),
        autosave_enabled=bool(autosave_enabled),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            session.close()

    app = FastAPI(title="BABA Player Web UI", lifespan=lifespan)
    app.state.session = session
    mount_shared_static(app)

    @app.get("/")
    def index():
        return html_file_response("player.html")

    @app.get("/api/session")
    def get_session():
        return app.state.session.to_payload()

    @app.post("/api/session/actions")
    def step_session(request: PlayerActionRequest):
        try:
            app.state.session.step(action_name=request.action)
            return app.state.session.to_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/reset")
    def reset_session(request: PlayerResetRequest):
        app.state.session.reset(seed=int(request.seed), scenario_type=app.state.session.scenario_type)
        return app.state.session.to_payload()

    @app.post("/api/session/next-scenario")
    def next_scenario():
        try:
            app.state.session.advance_to_next_scenario()
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/select-map")
    def select_map(request: PlayerSelectMapRequest):
        try:
            app.state.session.select_scenario(scenario_type=request.scenarioType)
            return app.state.session.to_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/undo")
    def undo_session():
        try:
            app.state.session.undo_last_action()
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/recording")
    def set_recording(request: PlayerRecordingRequest):
        app.state.session.set_recording_enabled(bool(request.enabled))
        return app.state.session.to_payload()

    @app.post("/api/session/save")
    def save_session():
        try:
            return app.state.session.save_current_frame()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
