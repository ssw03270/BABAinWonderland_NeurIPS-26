from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from src.web.compare_sessions import CompareSession
from src.web.common import NO_CACHE_HEADERS, PROJECT_ROOT, html_file_response, mount_shared_static
from src.web.discovery_registry import (
    can_shutdown_discovery_dashboard_server,
    delete_discovery_run,
    DiscoveryRunDeleteError,
    list_discovery_run_index_payloads,
    load_discovery_run_payload,
    request_discovery_dashboard_server_shutdown,
    resolve_run_artifact_path,
    update_discovery_viewer_state,
)
from src.web.offline_eval_service import (
    DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
    DEFAULT_OFFLINE_EVAL_SAMPLE_SEED,
    DEFAULT_OFFLINE_EVAL_RESULT_PAGE_SIZE,
    DEFAULT_OFFLINE_EVAL_WORKERS,
    OfflineEvalService,
)


class OfflineEvalRunRequest(BaseModel):
    runId: str | None = None
    runMode: str = "accuracy"
    source: str = Field(min_length=1)
    program: dict = Field(default_factory=dict)
    datasetRoot: str
    discoveryJson: str | None = None
    scenarioSplit: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT
    sampleSeed: int = DEFAULT_OFFLINE_EVAL_SAMPLE_SEED
    workers: int = Field(default=DEFAULT_OFFLINE_EVAL_WORKERS, ge=1)


class OfflineEvalExportRequest(BaseModel):
    scope: str
    failureId: str | None = None
    destinationDir: str | None = None
    preferredDir: str | None = None


class DiscoveryRunDeleteRequest(BaseModel):
    confirmText: str = Field(min_length=1)


class CompareLaunchRequest(BaseModel):
    versionKey: str = Field(min_length=1)
    seed: int = 42


class CompareActionRequest(BaseModel):
    action: str = Field(min_length=1)


class CompareResetRequest(BaseModel):
    seed: int


class CompareSelectMapRequest(BaseModel):
    scenarioType: str | None = None


class DiscoveryViewerStateRequest(BaseModel):
    clientId: str = Field(min_length=1)
    viewMode: str = "live"
    projectionMode: str = "live"
    projectionExcludeSparseClasses: bool = False


def create_discovery_app() -> FastAPI:
    eval_service = OfflineEvalService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            compare_session = getattr(app.state, "compare_session", None)
            if compare_session is not None:
                compare_session.close()
            eval_service.close()

    app = FastAPI(title="BABA Discovery Web Dashboard", lifespan=lifespan)
    app.state.eval_service = eval_service
    app.state.compare_session = None
    mount_shared_static(app)

    def require_compare_session() -> CompareSession:
        session = getattr(app.state, "compare_session", None)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Compare session unavailable. Launch compare from Version Navigator first.",
            )
        return session

    def replace_compare_session(next_session: CompareSession) -> CompareSession:
        previous = getattr(app.state, "compare_session", None)
        app.state.compare_session = next_session
        if previous is not None:
            previous.close()
        return next_session

    @app.get("/")
    def index():
        return html_file_response("discovery.html")

    @app.get("/compare")
    def compare_index():
        return html_file_response("compare.html")

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "api": "discovery",
            "version": 1,
            "projectRoot": str(PROJECT_ROOT.resolve()),
        }

    @app.get("/api/runs")
    def list_runs():
        runs = list_discovery_run_index_payloads()
        return {
            "runs": runs,
            "activeRunId": runs[0].get("runId") if runs else None,
            "canShutdownServer": can_shutdown_discovery_dashboard_server(),
        }

    @app.post("/api/server/shutdown")
    def shutdown_server(background_tasks: BackgroundTasks):
        if not can_shutdown_discovery_dashboard_server():
            raise HTTPException(
                status_code=409,
                detail="Discovery runs are still live. Wait for them to finish before shutting down the dashboard server.",
            )
        background_tasks.add_task(
            request_discovery_dashboard_server_shutdown,
            delay_sec=0.25,
        )
        return {
            "ok": True,
            "shutdownScheduled": True,
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        payload = load_discovery_run_payload(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Unknown discovery run.")
        return payload

    @app.post("/api/runs/{run_id}/delete")
    def delete_run(run_id: str, request: DiscoveryRunDeleteRequest):
        try:
            return delete_discovery_run(
                run_id,
                confirm_text=request.confirmText,
            )
        except DiscoveryRunDeleteError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_run_artifact(run_id: str, artifact_path: str):
        resolved_path = resolve_run_artifact_path(run_id, artifact_path)
        if resolved_path is None:
            raise HTTPException(status_code=404, detail="Unknown run artifact.")
        return FileResponse(resolved_path, headers=dict(NO_CACHE_HEADERS))

    @app.post("/api/runs/{run_id}/viewer-state")
    def set_run_viewer_state(run_id: str, body: DiscoveryViewerStateRequest):
        updated = update_discovery_viewer_state(
            run_id,
            client_id=body.clientId,
            view_mode=body.viewMode,
            projection_mode=body.projectionMode,
            projection_exclude_sparse_classes=body.projectionExcludeSparseClasses,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Unknown discovery run.")
        return {"ok": True}

    @app.post("/api/runs/{run_id}/compare")
    def launch_compare_session(run_id: str, request: CompareLaunchRequest):
        payload = load_discovery_run_payload(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Unknown discovery run.")

        run_output_dir = str(payload.get("runOutputDir") or "").strip()
        if not run_output_dir:
            raise HTTPException(
                status_code=409,
                detail="Discovery run does not expose a compare-compatible run output directory.",
            )

        requested_version = str(request.versionKey or "").strip()
        program_versions = payload.get("programVersions")
        if not isinstance(program_versions, list):
            program_versions = (payload.get("dashboard") or {}).get("programVersions")
        available_versions = {
            str(entry.get("versionKey") or "").strip()
            for entry in (program_versions or [])
            if isinstance(entry, dict) and str(entry.get("versionKey") or "").strip()
        }
        if requested_version not in available_versions:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown discovery program version: {requested_version}",
            )

        try:
            session = CompareSession.create(
                experiment=run_output_dir,
                version=requested_version,
                program_file=None,
                env_config="configs/env_config.yaml",
                experiment_config="configs/experiment_config_online.yaml",
                seed=int(request.seed),
                output_dir=None,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        replace_compare_session(session)
        return {
            "ok": True,
            "url": "/compare",
            "versionKey": session.version_tag,
            "seed": session.current_seed,
        }

    @app.get("/api/compare/session")
    def get_compare_session():
        return require_compare_session().to_payload()

    @app.post("/api/compare/session/actions")
    def step_compare_session(request: CompareActionRequest):
        session = require_compare_session()
        try:
            session.step(action_name=request.action)
            return session.to_payload()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/reset")
    def reset_compare_session(request: CompareResetRequest):
        session = require_compare_session()
        session.reset(seed=int(request.seed))
        return session.to_payload()

    @app.post("/api/compare/session/undo")
    def undo_compare_session():
        session = require_compare_session()
        try:
            session.undo_last_action()
            return session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/next-scenario")
    def next_compare_scenario():
        session = require_compare_session()
        try:
            session.advance_to_next_scenario()
            return session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/select-map")
    def select_compare_map(request: CompareSelectMapRequest):
        session = require_compare_session()
        try:
            session.select_scenario(scenario_type=request.scenarioType)
            return session.to_payload()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/compare/session/save")
    def save_compare_session():
        session = require_compare_session()
        try:
            return session.save_current_frame()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/eval/defaults")
    def get_eval_defaults():
        return app.state.eval_service.get_defaults()

    @app.get("/api/eval/catalog")
    def get_eval_catalog():
        return app.state.eval_service.get_catalog()

    @app.get("/api/eval/runs")
    def list_eval_runs():
        return {
            "runs": app.state.eval_service.list_runs(),
        }

    @app.post("/api/eval/runs")
    def create_eval_run(request: OfflineEvalRunRequest):
        run_output_dir = None
        program_context = None
        safe_run_id = str(request.runId or "").strip()
        if safe_run_id:
            payload = load_discovery_run_payload(safe_run_id)
            if payload is not None:
                run_output_dir = str(payload.get("runOutputDir") or "").strip() or None
                dashboard = payload.get("dashboard")
                if isinstance(dashboard, dict) and isinstance(dashboard.get("programContext"), dict):
                    program_context = dict(dashboard.get("programContext") or {})
        return app.state.eval_service.create_run(
            source=request.source,
            program=request.program,
            dataset_root=request.datasetRoot,
            discovery_json=request.discoveryJson,
            run_output_dir=run_output_dir,
            run_mode=request.runMode,
            program_context=program_context,
            allow_imports=True,
            sample_seed=int(request.sampleSeed),
            scenario_split=request.scenarioSplit,
            workers=int(request.workers),
        )

    @app.get("/api/eval/runs/{run_id}")
    def get_eval_run(run_id: str):
        return app.state.eval_service.get_run(run_id)

    @app.get("/api/eval/runs/{run_id}/classes")
    def get_eval_classes(
        run_id: str,
        offset: int = 0,
        limit: int = DEFAULT_OFFLINE_EVAL_RESULT_PAGE_SIZE,
        status: str = "all",
        sort: str = "desc",
    ):
        return app.state.eval_service.get_class_results(
            run_id,
            offset=int(offset),
            limit=int(limit),
            status_filter=status,
            sort=sort,
        )

    @app.get("/api/eval/runs/{run_id}/failures")
    def get_eval_failures(run_id: str):
        return app.state.eval_service.get_failures(run_id)

    @app.get("/api/eval/runs/{run_id}/failures/{failure_id}")
    def get_eval_failure(run_id: str, failure_id: str):
        return app.state.eval_service.get_failure(run_id, failure_id)

    @app.get("/api/eval/runs/{run_id}/failures/{failure_id}/render/{kind}")
    def render_eval_failure(run_id: str, failure_id: str, kind: str):
        png_bytes = app.state.eval_service.render_failure_kind(run_id, failure_id, kind)
        return Response(content=png_bytes, media_type="image/png", headers=dict(NO_CACHE_HEADERS))

    @app.get("/api/eval/runs/{run_id}/class-representatives/{representative_id}")
    def get_eval_class_representative(run_id: str, representative_id: str):
        return app.state.eval_service.get_class_representative(run_id, representative_id)

    @app.get("/api/eval/runs/{run_id}/class-representatives/{representative_id}/render/{kind}")
    def render_eval_class_representative(run_id: str, representative_id: str, kind: str):
        png_bytes = app.state.eval_service.render_class_representative_kind(run_id, representative_id, kind)
        return Response(content=png_bytes, media_type="image/png", headers=dict(NO_CACHE_HEADERS))

    @app.post("/api/eval/runs/{run_id}/export")
    def export_eval_failures(run_id: str, request: OfflineEvalExportRequest):
        return app.state.eval_service.export_failures(
            run_id=run_id,
            scope=request.scope,
            failure_id=request.failureId,
            destination_dir=request.destinationDir,
        )

    @app.post("/api/eval/runs/{run_id}/export/pick-directory")
    def export_eval_failures_pick_directory(run_id: str, request: OfflineEvalExportRequest):
        picked = app.state.eval_service.pick_export_directory(
            run_id=run_id,
            preferred_dir=request.preferredDir,
        )
        if picked is None:
            return {
                "cancelled": True,
            }
        return picked

    @app.post("/api/eval/runs/{run_id}/export/start")
    def start_eval_export(run_id: str, request: OfflineEvalExportRequest):
        return app.state.eval_service.start_export(
            run_id=run_id,
            scope=request.scope,
            failure_id=request.failureId,
            destination_dir=request.destinationDir,
        )

    @app.get("/api/eval/runs/{run_id}/export/status")
    def get_eval_export_status(run_id: str):
        return app.state.eval_service.get_export_status(run_id)

    @app.post("/api/eval/runs/{run_id}/export/cancel")
    def cancel_eval_export(run_id: str):
        return app.state.eval_service.cancel_export(run_id)

    return app
