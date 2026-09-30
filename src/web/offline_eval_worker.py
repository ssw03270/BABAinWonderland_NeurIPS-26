from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.web.offline_eval_service import _run_eval_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single offline heuristic eval worker job.")
    parser.add_argument(
        "--job-file",
        type=Path,
        required=True,
        help="Path to the offline eval worker job JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_path = Path(args.job_file).resolve()
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping job payload at {job_path}")
    _run_eval_worker(
        run_id=str(payload["run_id"]),
        run_dir=str(payload["run_dir"]),
        dataset_root=str(payload["dataset_root"]),
        discovery_json=(
            None
            if payload.get("discovery_json") in {None, ""}
            else str(payload["discovery_json"])
        ),
        source=str(payload["source"]),
        program=dict(payload.get("program") or {}),
        run_output_dir=(
            None
            if payload.get("run_output_dir") in {None, ""}
            else str(payload["run_output_dir"])
        ),
        word_aliases=dict(payload.get("word_aliases") or {}),
        word_aliases_source_path=(
            None
            if payload.get("word_aliases_source_path") in {None, ""}
            else str(payload["word_aliases_source_path"])
        ),
        run_mode=str(payload.get("run_mode") or "accuracy"),
        program_context=(
            dict(payload.get("program_context") or {})
            if isinstance(payload.get("program_context"), dict)
            else None
        ),
        allow_imports=bool(payload.get("allow_imports", False)),
        sample_seed=int(payload.get("sample_seed", 42)),
        scenario_split=str(payload.get("scenario_split") or "all"),
        workers=int(payload.get("workers", 16)),
        retention_days=int(payload["retention_days"]),
        max_runs=int(payload["max_runs"]),
    )


if __name__ == "__main__":
    main()
