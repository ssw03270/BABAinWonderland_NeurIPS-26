from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BABA discovery dashboard server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host for the dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the dashboard.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for dashboard development.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.web.launcher import run_discovery_dashboard_server

    run_discovery_dashboard_server(
        host=args.host,
        port=int(args.port),
        reload=bool(args.reload),
        open_browser=not bool(args.no_browser),
    )


if __name__ == "__main__":
    main()
