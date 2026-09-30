from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collect.data_coverage import (  # noqa: E402
    archive_state_input_to_canonical_text,
    archive_state_input_to_state_obj,
)
from scripts.data_collect.evaluate_offline_dataset import (  # noqa: E402
    _load_state_archive_inputs,
    load_offline_dataset,
)
from src.data.transition_artifacts import build_manual_transition_payload  # noqa: E402


DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "configs" / "custom_map_splits_dot_test.yaml"


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _load_config_with_extends(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    active_seen = set(seen or set())
    if resolved in active_seen:
        chain = " -> ".join(str(item) for item in [*active_seen, resolved])
        raise ValueError(f"Cyclic config inheritance detected: {chain}")
    active_seen.add(resolved)

    payload = _load_yaml(resolved)
    base_value = payload.get("base_config", payload.get("extends"))
    if not base_value:
        return payload
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"`base_config` must be a non-empty string in {resolved}")

    base_path = Path(base_value.strip())
    if not base_path.is_absolute():
        base_path = (resolved.parent / base_path).resolve()
    base_payload = _load_config_with_extends(base_path, seen=active_seen)
    return _deep_merge_dicts(base_payload, payload)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"base_config", "extends"}:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_word_aliases(
    *,
    world_setting: str,
    experiment_config: Path,
) -> dict[str, dict[str, str]]:
    normalized = str(world_setting or "default").strip().lower().replace("_", "-")
    if normalized in {"default", "default-world", "original"}:
        return {}
    if normalized not in {"wonderland", "baba-in-wonderland", "alice"}:
        raise ValueError(f"Unsupported world setting: {world_setting}")

    payload = _load_config_with_extends(experiment_config)
    serialization = payload.get("serialization")
    if not isinstance(serialization, Mapping):
        return {}
    raw_aliases = serialization.get("word_aliases")
    if not isinstance(raw_aliases, Mapping):
        return {}

    aliases: dict[str, dict[str, str]] = {}
    for raw_type, raw_mapping in raw_aliases.items():
        if not isinstance(raw_type, str) or not isinstance(raw_mapping, Mapping):
            continue
        type_key = raw_type.strip().lower()
        type_aliases: dict[str, str] = {}
        for raw_word, raw_alias in raw_mapping.items():
            if not isinstance(raw_word, str) or not isinstance(raw_alias, str):
                continue
            word = raw_word.strip().lower()
            alias = raw_alias.strip().lower()
            if word and alias:
                type_aliases[word] = alias
        if type_key and type_aliases:
            aliases[type_key] = type_aliases
    return aliases


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
