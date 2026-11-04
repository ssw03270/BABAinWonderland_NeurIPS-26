"""Interactive ASCII map editor entrypoint for Baba custom maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure direct script execution resolves the project-local `src` package.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.custom_map_spec import CUSTOM_MAP_DIFFICULTIES
from src.web.editor_sessions import EditorSession
from src.web.launcher import EDITOR_WEB_PORT, run_editor_web_ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ASCII map editor for Baba custom maps.")
    parser.add_argument(
        "--map-file",
        default=None,
        help="Optional custom map JSON path. If omitted, a new map session is created.",
    )
    parser.add_argument("--width", type=int, default=None, help="Width for a new map file.")
    parser.add_argument("--height", type=int, default=None, help="Height for a new map file.")
    parser.add_argument(
        "--difficulty",
        choices=tuple(CUSTOM_MAP_DIFFICULTIES.keys()),
        default=None,
        help="Difficulty bucket for a new map session.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Optional internal scenario id for a new map.",
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="Optional human-readable display name for a map.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the web UI.")
    parser.add_argument(
        "--port",
        type=int,
        default=EDITOR_WEB_PORT,
        help=f"Bind port for the web UI (default: {EDITOR_WEB_PORT}).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local web UI development.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = EditorSession.create(
        map_file=args.map_file,
        difficulty=args.difficulty,
        width=args.width,
        height=args.height,
        scenario_name=args.scenario_name,
        display_name=args.display_name,
    )

    print("Controls move to the dedicated editor web UI.")
    print("Hotkeys: arrows move cursor, space/enter place, backspace pop, s save, c clear.")

    run_editor_web_ui(
        host=getattr(args, "host", "127.0.0.1"),
        port=int(getattr(args, "port", EDITOR_WEB_PORT)),
        reload=bool(getattr(args, "reload", False)),
        open_browser=not bool(getattr(args, "no_browser", False)),
        map_path=session.map_path,
        spec=session.spec,
        dirty=bool(session.dirty),
        status=session.status,
    )


if __name__ == "__main__":
    main()
