"""
Live transition comparer for Baba in Wonderland.

- left: ground-truth environment transition
- right: program rollout transition driven by prior predicted states
"""

from __future__ import annotations

import argparse

from src.compare_runtime import normalize_version_tag, resolve_path
from src.web.launcher import COMPARE_WEB_PORT, run_compare_web_ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live transition comparer for BABA experiment programs."
    )
    parser.add_argument(
        "--experiment",
        required=False,
        type=str,
        help="Experiment directory name under the project experiments directory or absolute path.",
    )
    parser.add_argument(
        "--version",
        required=False,
        type=str,
        help="Program version tag (e.g., v006, 6, final).",
    )
    parser.add_argument(
        "--program-file",
        default=None,
        type=str,
        help="Optional path to a standalone compare-compatible Python program file.",
    )
    parser.add_argument(
        "--env-config",
        default="configs/env_config.yaml",
        type=str,
        help="Path to environment YAML config.",
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment_config_online.yaml",
        type=str,
        help="Path to experiment YAML config (used for sandbox limits).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Optional output directory for saved PNG/JSON frames.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Reset seed used for the environment.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the web UI.")
    parser.add_argument(
        "--port",
        type=int,
        default=COMPARE_WEB_PORT,
        help=f"Bind port for the web UI (default: {COMPARE_WEB_PORT}).",
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
    args = parser.parse_args()

    if not args.experiment and not args.program_file:
        raise ValueError("Provide either --experiment or --program-file.")

    resolved_experiment = None
    resolved_version = None
    resolved_program_file = None
    if args.program_file:
        program_path = resolve_path(args.program_file)
        if not program_path.exists():
            raise FileNotFoundError(f"Program file not found: {program_path}")
        resolved_program_file = str(program_path)
        if args.version:
            try:
                resolved_version = normalize_version_tag(args.version)
            except ValueError:
                resolved_version = str(args.version).strip()
        else:
            resolved_version = program_path.stem
    else:
        resolved_experiment = args.experiment
        if not args.version:
            raise ValueError("--version is required when loading a program from an experiment.")
        resolved_version = normalize_version_tag(args.version)

    print("Controls move to the dedicated compare web UI. Hotkeys: arrows, space, r, t, s.")

    run_compare_web_ui(
        host=args.host,
        port=int(args.port),
        reload=bool(args.reload),
        open_browser=not bool(args.no_browser),
        experiment=resolved_experiment,
        version=resolved_version,
        program_file=resolved_program_file,
        env_config=args.env_config,
        experiment_config=args.experiment_config,
        output_dir=args.output_dir,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
