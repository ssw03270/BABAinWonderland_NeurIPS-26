from __future__ import annotations

from pathlib import Path
from threading import Timer
from typing import Optional
import webbrowser

PLAYER_WEB_PORT = 8001
COMPARE_WEB_PORT = 8002
EDITOR_WEB_PORT = 8003
DISCOVERY_WEB_PORT = 8000


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


def run_player_web_ui(
    *,
    host: str,
    port: int,
    reload: bool,
    open_browser: bool,
    env_id: Optional[str],
    scenario_type: Optional[str],
    seed: int,
    autosave_enabled: bool,
) -> None:
    from src.web.player_app import create_player_app

    app = create_player_app(
        env_id=env_id,
        scenario_type=scenario_type,
        seed=int(seed),
        autosave_enabled=bool(autosave_enabled),
    )
    run_app_server(
        app=app,
        label="BABA Player Web UI",
        host=host,
        port=int(port),
        reload=bool(reload),
        open_browser=bool(open_browser),
    )


def run_compare_web_ui(
    *,
    host: str,
    port: int,
    reload: bool,
    open_browser: bool,
    experiment: Optional[str],
    version: Optional[str],
    program_file: Optional[str],
    env_config: str,
    experiment_config: str,
    output_dir: Optional[str],
    seed: int,
) -> None:
    from src.web.compare_app import create_compare_app

    resolved_program_file = (
        str(Path(program_file).resolve()) if isinstance(program_file, str) and program_file.strip() else None
    )
    app = create_compare_app(
        experiment=experiment,
        version=version,
        program_file=resolved_program_file,
        env_config=env_config,
        experiment_config=experiment_config,
        output_dir=output_dir,
        seed=int(seed),
    )
    run_app_server(
        app=app,
        label="BABA Compare Web UI",
        host=host,
        port=int(port),
        reload=bool(reload),
        open_browser=bool(open_browser),
    )


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


def run_discovery_dashboard_server(
    *,
    host: str,
    port: int,
    reload: bool,
    open_browser: bool,
) -> None:
    from src.web.discovery_app import create_discovery_app

    app = create_discovery_app()
    run_app_server(
        app=app,
        label="BABA Discovery Web Dashboard",
        host=host,
        port=int(port),
        reload=bool(reload),
        open_browser=bool(open_browser),
    )
