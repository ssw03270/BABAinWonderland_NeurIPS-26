"""
Manual Baba environment player entrypoint.

Examples:
  python ./scripts/play_baba_env.py
  python ./scripts/play_baba_env.py --env env/baba_in_wonderland_baseline_hard
  python ./scripts/play_baba_env.py --env env/baba_in_wonderland_baseline_hard --scenario shift_make_key_you_then_door_win
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure direct script execution resolves the project-local `src` package.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.player_runtime import (
    AUTOSAVE_EACH_STEP,
    ENV_IDS,
    RESET_SEED,
    SELECTED_ENV_INDEX,
    resolve_env_selection,
)
from src.web.launcher import PLAYER_WEB_PORT, run_player_web_ui


def print_env_index_table() -> None:
    print("Available environments:")
    for idx, env_id in enumerate(ENV_IDS):
        marker = " <==" if idx == SELECTED_ENV_INDEX else ""
        print(f"  [{idx:02d}] {env_id}{marker}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual Baba environment player")
    parser.add_argument(
        "--env",
        dest="env_id",
        help="Environment id to launch, e.g. env/baba_in_wonderland_baseline_hard",
    )
    parser.add_argument(
        "--scenario",
        dest="scenario_type",
        help="Optional scenario_type for environments that support fixed scenarios",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="Print the available environment ids and exit",
    )
    parser.add_argument("--seed", type=int, default=RESET_SEED, help="Reset seed used for the player session.")
    parser.add_argument(
        "--autosave",
        action="store_true",
        default=AUTOSAVE_EACH_STEP,
        help="Autosave rendered frames after each action.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the web UI.")
    parser.add_argument(
        "--port",
        type=int,
        default=PLAYER_WEB_PORT,
        help=f"Bind port for the web UI (default: {PLAYER_WEB_PORT}).",
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
    if args.list_envs:
        print_env_index_table()
        return

    env_index, env_id = resolve_env_selection(args.env_id)
    print_env_index_table()
    print()
    print(f"Launching: env[{env_index}] = {env_id}")
    if args.scenario_type:
        print(f"Requested scenario: {args.scenario_type}")
    print("Controls move to the dedicated player web UI. Hotkeys: arrows, space, r, t, s.")

    run_player_web_ui(
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        open_browser=not bool(args.no_browser),
        env_id=args.env_id,
        scenario_type=args.scenario_type,
        seed=int(args.seed),
        autosave_enabled=bool(args.autosave),
    )


if __name__ == "__main__":
    main()
