from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collect.data_coverage import (  # noqa: E402
    archive_state_input_to_canonical_text,
    archive_state_input_to_state_obj,
)
from scripts.data_collect.evaluate_offline_dataset import (  # noqa: E402
    _load_state_archive_inputs,
    _load_word_aliases,
    _load_yaml,
    load_offline_dataset,
)
from src.data.transition_artifacts import build_manual_transition_payload  # noqa: E402


DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "configs" / "custom_map_splits_dot_test.yaml"


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _load_split_scenarios(
    *,
    split_manifest: Path,
    split: str,
    difficulty: str,
) -> list[str]:
    payload = _load_yaml(split_manifest)
    difficulty_payload = payload.get(difficulty)
    if not isinstance(difficulty_payload, Mapping):
        raise ValueError(f"Difficulty {difficulty!r} not found in {split_manifest}")

    normalized_split = str(split).strip()
    split_key = {
        "train": "all_train",
        "test": "all_test",
    }.get(normalized_split, normalized_split)
    scenarios = difficulty_payload.get(split_key)
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError(f"Split {split_key!r} not found in {split_manifest}")
    return [str(value).strip() for value in scenarios if str(value).strip()]


def _read_transition_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path} line {line_index + 1}")
            rows.append(row)
    return rows


def _safe_clear_output_dir(path: Path) -> None:
    resolved = path.resolve()
    artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
    if artifacts_root not in resolved.parents and resolved != artifacts_root:
        raise ValueError(
            "Refusing to clear output outside the project artifacts directory. "
            f"Requested: {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def export_manual_transitions(
    *,
    dataset_root: Path,
    output_dir: Path,
    scenarios: Sequence[str],
    word_aliases: Mapping[str, Mapping[str, str]],
    world_setting: str,
    experiment_config: Path,
    env_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite:
        _safe_clear_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_offline_dataset(
        dataset_root=dataset_root,
        allowed_scenarios=scenarios,
        show_progress=False,
    )
    selected_scenarios = set(str(value).strip() for value in scenarios)
    global_index = 0
    bundle_summaries: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles, start=1):
        state_inputs = _load_state_archive_inputs(bundle.states_path)
        state_texts = [
            archive_state_input_to_canonical_text(
                state_input,
                word_aliases=word_aliases,
            )
            for state_input in state_inputs
        ]
        state_objs = [
            archive_state_input_to_state_obj(
                state_input,
                word_aliases=word_aliases,
            )
            for state_input in state_inputs
        ]
        rows = _read_transition_rows(bundle.transitions_path)
        scenario_dir = output_dir / str(bundle.scenario_type)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        written_for_bundle = 0
        for row in rows:
            transition_index = int(row.get("transition_index", written_for_bundle))
            state_index = int(row["state_index"])
            next_state_index = int(row["next_state_index"])
            payload = build_manual_transition_payload(
                env_id=env_id,
                env_index=bundle_index,
                scenario_type=str(bundle.scenario_type),
                step_index=transition_index,
                action_name=str(row.get("action", "")).strip(),
                action_id=-1,
                reward=float(row.get("reward", 0.0) or 0.0),
                terminated=bool(row.get("done", False)),
                truncated=False,
                previous_state_raw=state_texts[state_index],
                previous_state_obj=state_objs[state_index],
                next_state_raw=state_texts[next_state_index],
                next_state_obj=state_objs[next_state_index],
            )
            artifact_name = (
                f"{global_index:08d}_{bundle.scenario_type}_"
                f"{transition_index:06d}.json"
            )
            (scenario_dir / artifact_name).write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            global_index += 1
            written_for_bundle += 1

        bundle_summaries.append(
            {
                "scenario_type": str(bundle.scenario_type),
                "artifact_stem": str(bundle.artifact_stem),
                "transition_count": int(written_for_bundle),
                "transitions_path": str(bundle.transitions_path),
                "states_path": str(bundle.states_path),
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "experiment_config": str(experiment_config),
        "scenario_count": len(bundle_summaries),
        "transition_count": int(global_index),
        "requested_scenario_count": len(selected_scenarios),
        "world_setting": str(world_setting),
        "word_aliases_enabled": bool(word_aliases),
        "scenarios": bundle_summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export BABA offline transition datasets as per-transition manual "
            "artifacts consumable by ManualTransitionExplorer."
        )
    )
    parser.add_argument("--dataset-kind", default="solution", choices=("solution", "coverage"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--difficulty", default="original")
    parser.add_argument(
        "--world-setting",
        default="wonderland",
        choices=(
            "wonderland",
            "baba-in-wonderland",
            "alice",
            "default",
            "default-world",
            "original",
        ),
        help="Word setting used when serializing archived states. The project default is Wonderland.",
    )
    parser.add_argument(
        "--split-manifest",
        default=str(DEFAULT_SPLIT_MANIFEST),
    )
    parser.add_argument(
        "--experiment-config",
        default=str(PROJECT_ROOT / "configs" / "experiment_config_offline.yaml"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-id", default="env/baba_custom_ascii_original")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = PROJECT_ROOT / "test_dataset" / str(args.dataset_kind)
    split_manifest = _resolve_project_path(args.split_manifest)
    experiment_config = _resolve_project_path(args.experiment_config)
    output_dir = _resolve_project_path(args.output_dir)
    scenarios = _load_split_scenarios(
        split_manifest=split_manifest,
        split=str(args.split),
        difficulty=str(args.difficulty),
    )
    word_aliases = _load_word_aliases(
        world_setting=str(args.world_setting),
        experiment_config=experiment_config,
    )
    summary = export_manual_transitions(
        dataset_root=dataset_root,
        output_dir=output_dir,
        scenarios=scenarios,
        word_aliases=word_aliases,
        world_setting=str(args.world_setting),
        experiment_config=experiment_config,
        env_id=str(args.env_id),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
