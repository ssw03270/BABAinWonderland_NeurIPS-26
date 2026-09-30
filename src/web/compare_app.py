from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.web.common import html_file_response, mount_shared_static
from src.web.compare_sessions import CompareSession


class CompareActionRequest(BaseModel):
    action: str


class CompareResetRequest(BaseModel):
    seed: int


class CompareSelectMapRequest(BaseModel):
    scenarioType: str | None = None


def create_compare_app(
    *,
    experiment: str | None,
    version: str | None,
    program_file: str | None,
    env_config: str,
    experiment_config: str,
    output_dir: str | None,
    seed: int,
) -> FastAPI:
    session = CompareSession.create(
        experiment=experiment,
        version=version,
        program_file=program_file,
        env_config=env_config,
        experiment_config=experiment_config,
        seed=int(seed),
        output_dir=output_dir,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            session.close()

    app = FastAPI(title="BABA Compare Web UI", lifespan=lifespan)
    app.state.session = session
    mount_shared_static(app)

    @app.get("/")
    def index():
        return html_file_response("compare.html")

    @app.get("/api/compare/session")
    def get_session():
        return app.state.session.to_payload()

    @app.post("/api/compare/session/actions")
    def step_session(request: CompareActionRequest):
        try:
            app.state.session.step(action_name=request.action)
            return app.state.session.to_payload()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/reset")
    def reset_session(request: CompareResetRequest):
        app.state.session.reset(seed=int(request.seed))
        return app.state.session.to_payload()

    @app.post("/api/compare/session/undo")
    def undo_session():
        try:
            app.state.session.undo_last_action()
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/next-scenario")
    def next_scenario():
        try:
            app.state.session.advance_to_next_scenario()
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/select-map")
    def select_map(request: CompareSelectMapRequest):
        try:
            app.state.session.select_scenario(scenario_type=request.scenarioType)
            return app.state.session.to_payload()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/save")
    def save_session():
        try:
            return app.state.session.save_current_frame()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
