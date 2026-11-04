from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBUI_ROOT = PROJECT_ROOT / "webui"
ASSET_ROOT = PROJECT_ROOT / "assets"

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheStaticFiles(StaticFiles):
    def file_response(
        self,
        full_path: str | Path,
        stat_result,
        scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(
            full_path=full_path,
            stat_result=stat_result,
            scope=scope,
            status_code=status_code,
        )
        response.headers.update(NO_CACHE_HEADERS)
        return response


def mount_shared_static(app: FastAPI) -> None:
    app.mount("/assets", NoCacheStaticFiles(directory=ASSET_ROOT), name="assets")
    app.mount("/static", NoCacheStaticFiles(directory=WEBUI_ROOT), name="static")


def html_file_response(filename: str) -> FileResponse:
    return FileResponse(WEBUI_ROOT / filename, headers=dict(NO_CACHE_HEADERS))
