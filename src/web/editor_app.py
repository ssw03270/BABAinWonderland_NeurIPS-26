from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.web.common import html_file_response, mount_shared_static
from src.web.editor_sessions import EditorSession, build_editor_catalog_payload
from src.environments.custom_map_spec import resolve_custom_map_path


class EditCellRequest(BaseModel):
    x: int
    y: int
    mode: str = "append"
    token: Optional[str] = None
    direction: Optional[int] = None


class OpenSessionRequest(BaseModel):
    mapPath: str


class CreateSessionRequest(BaseModel):
    difficulty: str = "easy"
    width: Optional[int] = None
    height: Optional[int] = None
    scenarioName: Optional[str] = None
    displayName: Optional[str] = None


def create_editor_app(
    *,
    map_path: Path,
    spec: Dict[str, Any],
    dirty: bool,
    status: str,
) -> FastAPI:
    session = EditorSession.from_spec(
        map_path=map_path,
        spec=spec,
        dirty=bool(dirty),
        status=status,
    )

    app = FastAPI(title="BABA Editor Web UI")
    app.state.session = session
    mount_shared_static(app)

    @app.get("/")
    def index():
        return html_file_response("editor.html")

    @app.get("/api/session")
    def get_session():
        return app.state.session.to_payload()

    @app.get("/api/catalog")
    def get_catalog():
        return build_editor_catalog_payload(current_map_path=app.state.session.map_path)

    @app.post("/api/session/cells")
    def edit_cell(request: EditCellRequest):
        try:
            mode = str(request.mode or "append").strip().lower()
            if mode == "append":
                if not isinstance(request.token, str) or not request.token:
                    raise ValueError("`token` is required for append mode.")
                app.state.session.append_token(
                    x=int(request.x),
                    y=int(request.y),
                    token=request.token,
                    direction=request.direction,
                )
            elif mode == "pop":
                app.state.session.pop_token(x=int(request.x), y=int(request.y))
            elif mode == "clear":
                app.state.session.clear_cell(x=int(request.x), y=int(request.y))
            else:
                raise ValueError(f"Unsupported editor mode `{request.mode}`.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return app.state.session.to_payload()

    @app.post("/api/session/clear-interior")
    def clear_interior():
        app.state.session.clear_interior()
        return app.state.session.to_payload()

    @app.post("/api/session/open")
    def open_session(request: OpenSessionRequest):
        try:
            resolved_path = resolve_custom_map_path(request.mapPath)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail=f"Custom map file not found: {resolved_path}")
        app.state.session = EditorSession.create(map_file=str(resolved_path))
        return app.state.session.to_payload()

    @app.post("/api/session/new")
    def create_session(request: CreateSessionRequest):
        try:
            app.state.session = EditorSession.create(
                difficulty=request.difficulty,
                width=request.width,
                height=request.height,
                scenario_name=request.scenarioName,
                display_name=request.displayName,
                create_new=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return app.state.session.to_payload()

    @app.post("/api/session/save")
    def save_session():
        app.state.session.save()
        return app.state.session.to_payload()

    return app
