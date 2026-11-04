from __future__ import annotations

from threading import Timer
import webbrowser

EDITOR_WEB_PORT = 8003


def build_server_url(*, host: str, port: int) -> str:
    return f"http://{host}:{int(port)}/"


def run_app_server(
    *,
    app,
    label: str,
    host: str,
    port: int,
    reload: bool = False,
    open_browser: bool = True,
) -> None:
    import uvicorn

    url = build_server_url(host=host, port=int(port))
    print(f"Launching {label} at {url}")
    if reload and open_browser:
        print("Browser auto-open is disabled while --reload is enabled.")
    elif open_browser:
        Timer(0.8, lambda: webbrowser.open(url, new=2)).start()
    uvicorn.run(app, host=host, port=int(port), reload=bool(reload))

def run_editor_web_ui(
    *,
    host: str,
    port: int,
    reload: bool,
    open_browser: bool,
    map_path,
    spec,
    dirty: bool,
    status: str,
) -> None:
    from src.web.editor_app import create_editor_app

    app = create_editor_app(
        map_path=map_path,
        spec=spec,
        dirty=bool(dirty),
        status=status,
    )
    run_app_server(
        app=app,
        label="BABA Editor Web UI",
        host=host,
        port=int(port),
        reload=bool(reload),
        open_browser=bool(open_browser),
    )
