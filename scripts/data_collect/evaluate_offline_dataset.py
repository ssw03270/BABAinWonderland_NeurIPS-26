from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collect.data_coverage import (  # noqa: E402
    _resolve_project_path,
    archive_state_input_to_canonical_text,
)
from src.data.transition_buffer import Transition  # noqa: E402
from src.dynamics_examples import load_program_sources  # noqa: E402
from src.program_model import ProgramEvaluator, SandboxConfig  # noqa: E402


DEFAULT_DATASET_ROOT = Path("test_dataset") / "coverage"
DEFAULT_HEURISTIC_DISCOVERY_FILENAME = "heuristic_dynamics_discovery.json"
DEFAULT_HEURISTIC_SAMPLE_SEED = 42


@dataclass(frozen=True)
class OfflineScenarioDataset:
    scenario_type: str
    artifact_stem: str
    transitions_path: Path
    states_path: Path
    summary_path: Optional[Path]
    transition_count: int
    selected_transition_indices: Optional[frozenset[int]] = None
    source_transition_count: Optional[int] = None


_WORKER_EVALUATORS: Dict[tuple[str, bool], ProgramEvaluator] = {}


def _build_program_evaluator(*, allow_imports: bool) -> ProgramEvaluator:
    return ProgramEvaluator(
        sandbox_config=SandboxConfig(
            allow_imports=bool(allow_imports),
        )
    )


def _progress_iter(
    items: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    total: Optional[int] = None,
    leave: bool = False,
):
    if not enabled or _tqdm is None:
        return items
    return _tqdm(
        items,
        total=total,
        desc=desc,
        unit="item",
        leave=leave,
    )


def _count_transition_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return int(count)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON at {path}")
    return payload


def _load_state_archive_inputs(states_path: Path) -> List[Dict[str, Any]]:
    with np.load(states_path, allow_pickle=False) as payload:
        grid_sizes = np.asarray(payload["grid_sizes"])
        events = np.asarray(payload["events"])
        rewards = np.asarray(payload["rewards"])
        terminated = np.asarray(payload["terminated"])
        truncated = np.asarray(payload["truncated"])
        state_sources = np.asarray(payload["state_sources"])
        object_offsets = np.asarray(payload["object_offsets"])
        object_types = np.asarray(payload["object_types"])
        object_words = np.asarray(payload["object_words"])
        object_x = np.asarray(payload["object_x"])
        object_y = np.asarray(payload["object_y"])
        object_has_direction = np.asarray(payload["object_has_direction"])
        object_directions = np.asarray(payload["object_directions"])
        raw_state_count = payload.get("state_count")

        if raw_state_count is not None:
            state_count = int(np.asarray(raw_state_count).reshape(-1)[0])
        else:
            state_count = int(len(grid_sizes))

        state_payloads: List[Dict[str, Any]] = []
        for state_index in range(max(0, state_count)):
            start = int(object_offsets[state_index]) if state_index < len(object_offsets) else 0
            end = (
                int(object_offsets[state_index + 1])
                if state_index + 1 < len(object_offsets)
                else int(len(object_types))
            )
            objects: List[Dict[str, Any]] = []
            for object_index in range(max(0, start), max(0, end)):
                row = {
                    "type": str(object_types[object_index]),
                    "word": str(object_words[object_index]),
                    "position": [
                        int(object_x[object_index]),
                        int(object_y[object_index]),
                    ],
                }
                has_direction = bool(object_has_direction[object_index])
                if has_direction:
                    row["direction"] = str(object_directions[object_index])
                objects.append(row)

            raw_grid_size = grid_sizes[state_index] if state_index < len(grid_sizes) else [0, 0]
            if len(raw_grid_size) >= 2:
                grid_size = [int(raw_grid_size[0]), int(raw_grid_size[1])]
            else:
                grid_size = [0, 0]

            state_payloads.append(
                {
                    "source": (
                        str(state_sources[state_index]).strip()
                        if state_index < len(state_sources)
                        else ""
                    ),
                    "grid_size": grid_size,
                    "objects": objects,
                    "event": str(events[state_index]).strip() if state_index < len(events) else "",
                    "reward": float(rewards[state_index]) if state_index < len(rewards) else 0.0,
                    "terminated": bool(terminated[state_index]) if state_index < len(terminated) else False,
                    "truncated": bool(truncated[state_index]) if state_index < len(truncated) else False,
                }
            )
    return state_payloads


def _load_transitions_from_bundle(
    *,
    states_path: Path,
    transitions_path: Path,
    selected_transition_indices: Optional[Sequence[int]] = None,
) -> List[Transition]:
    state_payloads = _load_state_archive_inputs(states_path)
    state_json_by_index = [
        archive_state_input_to_canonical_text(state_payload)
        for state_payload in state_payloads
    ]

    allowed_transition_indices = None
    if selected_transition_indices is not None:
        allowed_transition_indices = {
            int(value)
            for value in selected_transition_indices
        }

    transitions: List[Transition] = []
    line_index = 0
    with transitions_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object row in {transitions_path}")
            transition_index = int(row.get("transition_index", line_index))
            if (
                allowed_transition_indices is not None
                and transition_index not in allowed_transition_indices
                and line_index not in allowed_transition_indices
            ):
                line_index += 1
                continue
            state_index = int(row["state_index"])
            next_state_index = int(row["next_state_index"])
            if not (0 <= state_index < len(state_json_by_index)):
                raise IndexError(
                    f"state_index {state_index} is out of bounds for {transitions_path}"
                )
            if not (0 <= next_state_index < len(state_json_by_index)):
                raise IndexError(
                    f"next_state_index {next_state_index} is out of bounds for {transitions_path}"
                )
            transitions.append(
                Transition(
                    state=state_json_by_index[state_index],
                    action=str(row.get("action", "")).strip(),
                    next_state=state_json_by_index[next_state_index],
                    reward=float(row.get("reward", 0.0) or 0.0),
                    done=bool(row.get("done", False)),
                )
            )
            line_index += 1
    return transitions


def _serialize_compile_errors(errors: Sequence[Any]) -> List[Dict[str, str]]:
    return [
        {
            "phase": str(getattr(error, "phase", "")).strip(),
            "message": str(getattr(error, "message", "")).strip(),
        }
        for error in errors
    ]


def _get_worker_evaluator(source: str, *, allow_imports: bool) -> ProgramEvaluator:
    safe_source = str(source)
    cache_key = (safe_source, bool(allow_imports))
    evaluator = _WORKER_EVALUATORS.get(cache_key)
    if evaluator is None:
        evaluator = _build_program_evaluator(allow_imports=bool(allow_imports))
        _WORKER_EVALUATORS[cache_key] = evaluator
    return evaluator


def _evaluate_bundle_task(
    *,
    bundle_payload: Mapping[str, Any],
    program_payloads: Sequence[Mapping[str, Any]],
    allow_imports: bool,
) -> Dict[str, Any]:
    transitions_path = Path(str(bundle_payload["transitions_path"]))
    states_path = Path(str(bundle_payload["states_path"]))
    transitions = _load_transitions_from_bundle(
        states_path=states_path,
        transitions_path=transitions_path,
        selected_transition_indices=bundle_payload.get("selected_transition_indices"),
    )

    program_results: List[Dict[str, Any]] = []
    for program_payload in program_payloads:
        source = str(program_payload["source"])
        evaluator = _get_worker_evaluator(source, allow_imports=allow_imports)
        evaluation = evaluator.evaluate_source(
            source=source,
            transitions=transitions,
            collect_records=False,
        )
        evaluator.clear_runtime_caches()
        program_results.append(
            {
                "version_id": str(program_payload["version_id"]),
                "correct_count": int(evaluation.correct_count),
                "total_count": int(evaluation.total_count),
                "accuracy": float(evaluation.accuracy),
                "runtime_error_count": int(evaluation.runtime_error_count),
                "compile_errors": _serialize_compile_errors(evaluation.compile_errors),
            }
        )

    return {
        "scenario_type": str(bundle_payload["scenario_type"]),
        "artifact_stem": str(bundle_payload["artifact_stem"]),
        "transition_count": int(bundle_payload["transition_count"]),
        "source_transition_count": int(
            bundle_payload.get("source_transition_count", bundle_payload["transition_count"])
        ),
        "transitions_path": str(transitions_path),
        "states_path": str(states_path),
        "program_results": program_results,
    }


def _bundle_from_summary_row(
    *,
    dataset_root: Path,
    row: Mapping[str, Any],
) -> Optional[OfflineScenarioDataset]:
    transitions_value = row.get("transitions_path")
    states_value = row.get("states_path")
    if not isinstance(transitions_value, str) or not transitions_value.strip():
        return None
    if not isinstance(states_value, str) or not states_value.strip():
        return None

    transitions_path = _resolve_project_path(transitions_value)
    states_path = _resolve_project_path(states_value)
    if not transitions_path.exists() or not states_path.exists():
        return None

    scenario_type = str(row.get("scenario_type", "")).strip()
    artifact_stem = str(row.get("artifact_stem", "")).strip()
    if not artifact_stem:
        artifact_stem = transitions_path.stem.removesuffix("_transitions")
    if not scenario_type:
        scenario_type = artifact_stem
    transition_count = max(0, int(row.get("transition_count", 0) or 0))
    if transition_count <= 0:
        transition_count = _count_transition_rows(transitions_path)

    summary_path = dataset_root / f"{artifact_stem}.json"
    return OfflineScenarioDataset(
        scenario_type=scenario_type,
        artifact_stem=artifact_stem,
        transitions_path=transitions_path,
        states_path=states_path,
        summary_path=(summary_path if summary_path.exists() else None),
        transition_count=int(transition_count),
    )


def _scan_dataset_root(dataset_root: Path) -> List[OfflineScenarioDataset]:
    bundles: List[OfflineScenarioDataset] = []
    for transitions_path in sorted(dataset_root.glob("*_transitions.jsonl")):
        artifact_stem = transitions_path.stem.removesuffix("_transitions")
        states_path = dataset_root / f"{artifact_stem}_states.npz"
        if not states_path.exists():
            continue
        summary_path = dataset_root / f"{artifact_stem}.json"
        scenario_type = artifact_stem
        if summary_path.exists():
            try:
                summary_payload = _read_json(summary_path)
                scenario_value = summary_payload.get("scenario_type")
                if isinstance(scenario_value, str) and scenario_value.strip():
                    scenario_type = scenario_value.strip()
            except Exception:
                pass
        bundles.append(
            OfflineScenarioDataset(
                scenario_type=scenario_type,
                artifact_stem=artifact_stem,
                transitions_path=transitions_path,
                states_path=states_path,
                summary_path=(summary_path if summary_path.exists() else None),
                transition_count=_count_transition_rows(transitions_path),
            )
        )
    return bundles


def _parse_boolish_flag(raw_value: Any) -> Optional[bool]:
    if isinstance(raw_value, bool):
        return bool(raw_value)
    if raw_value is None:
        return None
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return None


def _resolve_heuristic_discovery_request(
    dataset_root: Path,
    raw_value: Any,
) -> Optional[Path]:
    boolish = _parse_boolish_flag(raw_value)
    if boolish is False:
        return None
    if boolish is True:
        return dataset_root / DEFAULT_HEURISTIC_DISCOVERY_FILENAME
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return _resolve_project_path(text)


def _append_bundle_lookup(
    lookup: Dict[str, List[OfflineScenarioDataset]],
    key: str,
    bundle: OfflineScenarioDataset,
) -> None:
    normalized_key = str(key).strip()
    if not normalized_key:
        return
    lookup.setdefault(normalized_key, []).append(bundle)


def _resolve_unique_bundle_lookup(
    lookup: Mapping[str, Sequence[OfflineScenarioDataset]],
    key: Any,
) -> Optional[OfflineScenarioDataset]:
    if not isinstance(key, str):
        return None
    normalized_key = key.strip()
    if not normalized_key:
        return None
    matches = lookup.get(normalized_key)
    if not matches or len(matches) != 1:
        return None
    return matches[0]


def _resolve_heuristic_mapping_bundle(
    *,
    row: Mapping[str, Any],
    bundle_by_transition_path: Mapping[str, OfflineScenarioDataset],
    bundles_by_transition_name: Mapping[str, Sequence[OfflineScenarioDataset]],
    bundles_by_artifact_stem: Mapping[str, Sequence[OfflineScenarioDataset]],
    bundles_by_scenario_type: Mapping[str, Sequence[OfflineScenarioDataset]],
) -> Optional[OfflineScenarioDataset]:
    transitions_value = row.get("transitions_path")
    if isinstance(transitions_value, str) and transitions_value.strip():
        resolved_path = str(_resolve_project_path(transitions_value))
        exact_match = bundle_by_transition_path.get(resolved_path)
        if exact_match is not None:
            return exact_match

        transition_name = Path(transitions_value).name
        named_match = _resolve_unique_bundle_lookup(
            bundles_by_transition_name,
            transition_name,
        )
        if named_match is not None:
            return named_match

        transition_stem = Path(transitions_value).stem.removesuffix("_transitions")
        stem_match = _resolve_unique_bundle_lookup(
            bundles_by_artifact_stem,
            transition_stem,
        )
        if stem_match is not None:
            return stem_match

    artifact_match = _resolve_unique_bundle_lookup(
        bundles_by_artifact_stem,
        row.get("artifact_stem"),
    )
    if artifact_match is not None:
        return artifact_match

    return _resolve_unique_bundle_lookup(
        bundles_by_scenario_type,
        row.get("scenario_type"),
    )


def _sample_bundles_by_heuristic_class(
    *,
    bundles: Sequence[OfflineScenarioDataset],
    heuristic_discovery_json: Path,
    seed: int,
) -> tuple[List[OfflineScenarioDataset], Dict[str, Any]]:
    payload = _read_json(heuristic_discovery_json)
    mappings = payload.get("transition_class_mapping")
    if not isinstance(mappings, list):
        raise ValueError(
            f"Heuristic discovery JSON is missing `transition_class_mapping`: {heuristic_discovery_json}"
        )

    bundle_by_transition_path = {
        str(bundle.transitions_path.resolve()): bundle
        for bundle in bundles
    }
    bundles_by_transition_name: Dict[str, List[OfflineScenarioDataset]] = {}
    bundles_by_artifact_stem: Dict[str, List[OfflineScenarioDataset]] = {}
    bundles_by_scenario_type: Dict[str, List[OfflineScenarioDataset]] = {}
    for bundle in bundles:
        _append_bundle_lookup(
            bundles_by_transition_name,
            bundle.transitions_path.name,
            bundle,
        )
        _append_bundle_lookup(
            bundles_by_artifact_stem,
            bundle.artifact_stem,
            bundle,
        )
        _append_bundle_lookup(
            bundles_by_scenario_type,
            bundle.scenario_type,
            bundle,
        )

    class_to_refs: Dict[int, List[tuple[OfflineScenarioDataset, int]]] = {}
    matched_bundle_count = 0
    skipped_mapping_rows = 0
    for row in mappings:
        if not isinstance(row, Mapping):
            skipped_mapping_rows += 1
            continue
        transitions_value = row.get("transitions_path")
        transition_to_class = row.get("transition_to_class")
        if not isinstance(transitions_value, str) or not isinstance(transition_to_class, Mapping):
            skipped_mapping_rows += 1
            continue
        bundle = _resolve_heuristic_mapping_bundle(
            row=row,
            bundle_by_transition_path=bundle_by_transition_path,
            bundles_by_transition_name=bundles_by_transition_name,
            bundles_by_artifact_stem=bundles_by_artifact_stem,
            bundles_by_scenario_type=bundles_by_scenario_type,
        )
        if bundle is None:
            continue
        matched_bundle_count += 1
        for raw_transition_index, raw_class_idx in transition_to_class.items():
            try:
                transition_index = int(raw_transition_index)
                class_idx = int(raw_class_idx)
            except (TypeError, ValueError):
                continue
            class_to_refs.setdefault(class_idx, []).append((bundle, transition_index))

    if not class_to_refs:
        raise ValueError(
            "No heuristic-dynamics classes matched the selected evaluation bundles. "
            f"Source JSON: {heuristic_discovery_json}"
        )

    rng = random.Random(int(seed))
    selected_indices_by_bundle_path: Dict[str, set[int]] = {}
    sampled_class_count = 0
    for class_idx in sorted(class_to_refs.keys()):
        refs = class_to_refs[class_idx]
        ordered_refs = sorted(
            refs,
            key=lambda item: (
                item[0].scenario_type,
                item[0].artifact_stem,
                str(item[0].transitions_path),
                int(item[1]),
            ),
        )
        chosen_bundle, chosen_transition_index = rng.choice(ordered_refs)
        selected_indices_by_bundle_path.setdefault(
            str(chosen_bundle.transitions_path.resolve()),
            set(),
        ).add(int(chosen_transition_index))
        sampled_class_count += 1

    sampled_bundles: List[OfflineScenarioDataset] = []
    sampled_transition_total = 0
    for bundle in bundles:
        selected_indices = selected_indices_by_bundle_path.get(str(bundle.transitions_path.resolve()))
        if not selected_indices:
            continue
        sampled_transition_count = len(selected_indices)
        sampled_transition_total += int(sampled_transition_count)
        sampled_bundles.append(
            OfflineScenarioDataset(
                scenario_type=bundle.scenario_type,
                artifact_stem=bundle.artifact_stem,
                transitions_path=bundle.transitions_path,
                states_path=bundle.states_path,
                summary_path=bundle.summary_path,
                transition_count=int(sampled_transition_count),
                selected_transition_indices=frozenset(sorted(selected_indices)),
                source_transition_count=int(bundle.transition_count),
            )
        )

    if not sampled_bundles:
        raise ValueError(
            "Heuristic-dynamics sampling selected zero transitions for evaluation. "
            f"Source JSON: {heuristic_discovery_json}"
        )

    metadata = {
        "enabled": True,
        "source_json": str(heuristic_discovery_json),
        "sample_seed": int(seed),
        "class_count": int(sampled_class_count),
        "sampled_transition_count": int(sampled_transition_total),
        "matched_bundle_count": int(matched_bundle_count),
        "sampled_bundle_count": int(len(sampled_bundles)),
        "skipped_mapping_rows": int(skipped_mapping_rows),
    }
    return sampled_bundles, metadata


def load_offline_dataset(
    *,
    dataset_root: str | Path,
    allowed_scenarios: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> List[OfflineScenarioDataset]:
    resolved_dataset_root = _resolve_project_path(dataset_root)
    if not resolved_dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {resolved_dataset_root}")

    requested_scenarios = {
        str(value).strip()
        for value in (allowed_scenarios or ())
        if isinstance(value, str) and value.strip()
    }

    bundles: List[OfflineScenarioDataset] = []
    batch_summary_path = resolved_dataset_root / "batch_summary.json"
    if batch_summary_path.exists():
        batch_payload = _read_json(batch_summary_path)
        results = batch_payload.get("results")
        if isinstance(results, list):
            filtered_results: List[Mapping[str, Any]] = []
            for row in results:
                if not isinstance(row, Mapping):
                    continue
                if requested_scenarios:
                    row_scenario = str(row.get("scenario_type", "")).strip()
                    row_artifact_stem = str(row.get("artifact_stem", "")).strip()
                    candidate_names = {
                        value
                        for value in (row_scenario, row_artifact_stem)
                        if value
                    }
                    if candidate_names and requested_scenarios.isdisjoint(candidate_names):
                        continue
                filtered_results.append(row)
            loading_bar = None
            if show_progress and _tqdm is not None:
                total_transitions = sum(
                    max(0, int(row.get("transition_count", 0) or 0))
                    for row in filtered_results
                )
                use_transition_units = total_transitions > 0
                loading_bar = _tqdm(
                    total=total_transitions if use_transition_units else len(filtered_results),
                    desc="Indexing transitions",
                    unit="transition" if use_transition_units else "bundle",
                    leave=False,
                )
            for row in filtered_results:
                bundle = _bundle_from_summary_row(
                    dataset_root=resolved_dataset_root,
                    row=row,
                )
                if bundle is None:
                    continue
                bundles.append(bundle)
                if loading_bar is not None:
                    increment = int(bundle.transition_count) if use_transition_units else 1
                    loading_bar.update(max(1, increment))
            if loading_bar is not None:
                if loading_bar.total is not None and loading_bar.n < int(loading_bar.total):
                    loading_bar.total = int(loading_bar.n)
                loading_bar.close()

    if bundles:
        loaded_transition_paths = {
            str(Path(bundle.transitions_path).resolve())
            for bundle in bundles
        }
        for transitions_path in sorted(resolved_dataset_root.glob("*_transitions.jsonl")):
            resolved_transitions_path = transitions_path.resolve()
            if str(resolved_transitions_path) in loaded_transition_paths:
                continue
            artifact_stem = transitions_path.stem.removesuffix("_transitions")
            states_path = resolved_dataset_root / f"{artifact_stem}_states.npz"
            if not states_path.exists():
                continue
            summary_path = resolved_dataset_root / f"{artifact_stem}.json"
            scenario_type = artifact_stem
            if summary_path.exists():
                try:
                    summary_payload = _read_json(summary_path)
                    scenario_value = summary_payload.get("scenario_type")
                    if isinstance(scenario_value, str) and scenario_value.strip():
                        scenario_type = scenario_value.strip()
                except Exception:
                    pass
            if requested_scenarios and requested_scenarios.isdisjoint({scenario_type, artifact_stem}):
                continue
            bundles.append(
                OfflineScenarioDataset(
                    scenario_type=scenario_type,
                    artifact_stem=artifact_stem,
                    transitions_path=resolved_transitions_path,
                    states_path=states_path,
                    summary_path=(summary_path if summary_path.exists() else None),
                    transition_count=_count_transition_rows(transitions_path),
                )
            )

    if not bundles:
        scanned_paths = sorted(resolved_dataset_root.glob("*_transitions.jsonl"))
        if show_progress and _tqdm is not None:
            total_transitions = sum(_count_transition_rows(path) for path in scanned_paths)
            loading_bar = _tqdm(
                total=total_transitions if total_transitions > 0 else len(scanned_paths),
                desc="Indexing transitions",
                unit="transition" if total_transitions > 0 else "bundle",
                leave=False,
            )
            scanned: List[OfflineScenarioDataset] = []
            for transitions_path in scanned_paths:
                artifact_stem = transitions_path.stem.removesuffix("_transitions")
                states_path = resolved_dataset_root / f"{artifact_stem}_states.npz"
                if not states_path.exists():
                    continue
                summary_path = resolved_dataset_root / f"{artifact_stem}.json"
                scenario_type = artifact_stem
                if summary_path.exists():
                    try:
                        summary_payload = _read_json(summary_path)
                        scenario_value = summary_payload.get("scenario_type")
                        if isinstance(scenario_value, str) and scenario_value.strip():
                            scenario_type = scenario_value.strip()
                    except Exception:
                        pass
                transition_count = _count_transition_rows(transitions_path)
                scanned.append(
                    OfflineScenarioDataset(
                        scenario_type=scenario_type,
                        artifact_stem=artifact_stem,
                        transitions_path=transitions_path,
                        states_path=states_path,
                        summary_path=(summary_path if summary_path.exists() else None),
                        transition_count=transition_count,
                    )
                )
                loading_bar.update(
                    max(1, int(transition_count))
                    if total_transitions <= 0
                    else int(transition_count)
                )
            if loading_bar.total is not None and loading_bar.n < int(loading_bar.total):
                loading_bar.total = int(loading_bar.n)
            loading_bar.close()
        else:
            scanned = _scan_dataset_root(resolved_dataset_root)
        if requested_scenarios:
            scanned = [
                bundle
                for bundle in scanned
                if bundle.scenario_type in requested_scenarios
            ]
        bundles = scanned

    if not bundles:
        raise ValueError(f"No offline transition bundles found under {resolved_dataset_root}")

    discovered = {bundle.scenario_type for bundle in bundles}
    missing = sorted(requested_scenarios - discovered)
    if missing:
        raise ValueError(
            "Requested scenarios are missing from the dataset root: "
            + ", ".join(missing)
        )

    return sorted(bundles, key=lambda bundle: (bundle.scenario_type, bundle.artifact_stem))


def evaluate_offline_dataset(
    *,
    bundles: Sequence[OfflineScenarioDataset],
    program_sources: Sequence[Any],
    workers: int = 1,
    allow_imports: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    if len(bundles) == 0:
        raise ValueError("At least one dataset bundle is required.")
    if len(program_sources) == 0:
        raise ValueError("At least one program source is required.")

    program_rows: List[Dict[str, Any]] = [
        {
            "program": program.to_dict(),
            "accuracy": 0.0,
            "correct_count": 0,
            "total_count": 0,
            "runtime_error_count": 0,
            "compile_errors": [],
            "scenarios": [],
        }
        for program in program_sources
    ]
    version_row_by_id = {
        str(row["program"]["version_id"]): row
        for row in program_rows
    }

    total_transitions = sum(int(bundle.transition_count) for bundle in bundles)
    total_evaluations = int(total_transitions * len(program_sources))
    normalized_workers = max(1, int(workers))

    eval_bar = None
    if show_progress and _tqdm is not None:
        eval_bar = _tqdm(
            total=total_evaluations,
            desc="Evaluating transitions",
            unit="transition",
            leave=True,
        )

    def _accumulate_bundle_result(bundle_result: Mapping[str, Any]) -> None:
        scenario_type = str(bundle_result["scenario_type"])
        artifact_stem = str(bundle_result["artifact_stem"])
        transition_count = int(bundle_result["transition_count"])
        source_transition_count = int(
            bundle_result.get("source_transition_count", bundle_result["transition_count"])
        )
        transitions_path = str(bundle_result["transitions_path"])
        states_path = str(bundle_result["states_path"])
        for program_result in bundle_result["program_results"]:
            version_id = str(program_result["version_id"])
            version_row = version_row_by_id[version_id]
            compile_rows = list(program_result.get("compile_errors", []))
            if compile_rows and not version_row["compile_errors"]:
                version_row["compile_errors"] = compile_rows
            correct_count = int(program_result["correct_count"])
            total_count = int(program_result["total_count"])
            runtime_error_count = int(program_result["runtime_error_count"])
            version_row["correct_count"] += correct_count
            version_row["total_count"] += total_count
            version_row["runtime_error_count"] += runtime_error_count
            version_row["scenarios"].append(
                {
                    "scenario_type": scenario_type,
                    "artifact_stem": artifact_stem,
                    "transition_count": transition_count,
                    "source_transition_count": source_transition_count,
                    "correct_count": correct_count,
                    "accuracy": float(program_result["accuracy"]),
                    "runtime_error_count": runtime_error_count,
                    "transitions_path": transitions_path,
                    "states_path": states_path,
                }
            )

    if normalized_workers <= 1:
        program_evaluators = [
            _build_program_evaluator(allow_imports=allow_imports)
            for _ in program_sources
        ]
        for bundle in bundles:
            transitions = _load_transitions_from_bundle(
                states_path=bundle.states_path,
                transitions_path=bundle.transitions_path,
                selected_transition_indices=bundle.selected_transition_indices,
            )
            for program, evaluator, version_row in zip(program_sources, program_evaluators, program_rows):
                progress_state = {"evaluated_count": 0}
                if eval_bar is not None:
                    eval_bar.set_description(f"Eval {program.version_id} / {bundle.scenario_type}")

                def _on_progress(payload: Dict[str, Any]) -> None:
                    if eval_bar is None:
                        return
                    current = int(payload.get("evaluated_count", 0) or 0)
                    delta = max(0, current - int(progress_state["evaluated_count"]))
                    if delta > 0:
                        eval_bar.update(delta)
                        progress_state["evaluated_count"] = current

                evaluation = evaluator.evaluate_source(
                    source=program.source,
                    transitions=transitions,
                    progress_callback=_on_progress if eval_bar is not None else None,
                    collect_records=False,
                )
                evaluator.clear_runtime_caches()
                if eval_bar is not None:
                    remaining = max(
                        0,
                        int(bundle.transition_count) - int(progress_state["evaluated_count"]),
                    )
                    if remaining > 0:
                        eval_bar.update(remaining)

                compile_rows = _serialize_compile_errors(evaluation.compile_errors)
                if compile_rows and not version_row["compile_errors"]:
                    version_row["compile_errors"] = compile_rows

                version_row["correct_count"] += int(evaluation.correct_count)
                version_row["total_count"] += int(evaluation.total_count)
                version_row["runtime_error_count"] += int(evaluation.runtime_error_count)
                version_row["scenarios"].append(
                    {
                        "scenario_type": bundle.scenario_type,
                        "artifact_stem": bundle.artifact_stem,
                        "transition_count": int(bundle.transition_count),
                        "source_transition_count": int(bundle.source_transition_count or bundle.transition_count),
                        "correct_count": int(evaluation.correct_count),
                        "accuracy": float(evaluation.accuracy),
                        "runtime_error_count": int(evaluation.runtime_error_count),
                        "transitions_path": str(bundle.transitions_path),
                        "states_path": str(bundle.states_path),
                    }
                )
    else:
        program_payloads = [
            {
                "version_id": str(program.version_id),
                "source": str(program.source),
            }
            for program in program_sources
        ]
        bundle_payloads = [
            {
                "scenario_type": bundle.scenario_type,
                "artifact_stem": bundle.artifact_stem,
                "transition_count": int(bundle.transition_count),
                "source_transition_count": int(bundle.source_transition_count or bundle.transition_count),
                "selected_transition_indices": (
                    sorted(int(value) for value in bundle.selected_transition_indices)
                    if bundle.selected_transition_indices is not None
                    else None
                ),
                "transitions_path": str(bundle.transitions_path),
                "states_path": str(bundle.states_path),
            }
            for bundle in bundles
        ]
        with ProcessPoolExecutor(max_workers=normalized_workers) as executor:
            future_to_bundle = {
                executor.submit(
                    _evaluate_bundle_task,
                    bundle_payload=bundle_payload,
                    program_payloads=program_payloads,
                    allow_imports=bool(allow_imports),
                ): bundle_payload
                for bundle_payload in bundle_payloads
            }
            for future in as_completed(future_to_bundle):
                bundle_payload = future_to_bundle[future]
                bundle_result = future.result()
                _accumulate_bundle_result(bundle_result)
                if eval_bar is not None:
                    eval_bar.set_description(
                        f"Eval {bundle_result['scenario_type']} ({len(program_sources)} programs)"
                    )
                    eval_bar.update(
                        int(bundle_payload["transition_count"]) * int(len(program_sources))
                    )

    for version_row in program_rows:
        total_count = int(version_row["total_count"])
        correct_count = int(version_row["correct_count"])
        version_row["accuracy"] = (
            float(correct_count) / float(total_count)
            if total_count > 0
            else 0.0
        )
        version_row["scenarios"].sort(key=lambda row: (row["scenario_type"], row["artifact_stem"]))

    if eval_bar is not None:
        eval_bar.close()

    return {
        "dataset": {
            "scenario_count": int(len(bundles)),
            "transition_count": int(total_transitions),
            "workers": int(normalized_workers),
            "scenarios": [
                {
                    "scenario_type": bundle.scenario_type,
                    "artifact_stem": bundle.artifact_stem,
                    "transition_count": int(bundle.transition_count),
                    "source_transition_count": int(bundle.source_transition_count or bundle.transition_count),
                    "transitions_path": str(bundle.transitions_path),
                    "states_path": str(bundle.states_path),
                    "summary_path": (str(bundle.summary_path) if bundle.summary_path is not None else None),
                }
                for bundle in bundles
            ],
        },
        "versions": program_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained Baba program versions against offline transition datasets "
            "collected by data_coverage.py or data_solution.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset directory containing batch_summary.json plus *_states.npz / *_transitions.jsonl.",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        default=[],
        help="Restrict evaluation to one scenario. Can be repeated.",
    )
    parser.add_argument(
        "--heuristic-dynamics-discovery",
        "--heuristic_dynamics_discovery",
        dest="heuristic_dynamics_discovery",
        nargs="?",
        default="false",
        const="true",
        help=(
            "If enabled, evaluate one random transition per heuristic dynamics class using a fixed seed. "
            "Pass `true` to read <dataset-root>/heuristic_dynamics_discovery.json, "
            "`false` to disable, or provide an explicit JSON path."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment root directory containing program_versions/.",
    )
    parser.add_argument(
        "--program-dir",
        type=Path,
        default=None,
        help="Directory containing v*.py program versions.",
    )
    parser.add_argument(
        "--program-file",
        dest="program_files",
        type=Path,
        action="append",
        default=[],
        help="Additional standalone program file to evaluate. Can be repeated.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the machine-readable report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report to stdout.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during dataset loading and evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for bundle-level parallel evaluation.",
    )
    parser.add_argument(
        "--disallow-imports",
        action="store_true",
        help=(
            "Disable Python imports in evaluated programs. "
            "Keep disabled only for strict sandboxing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.experiment_dir is None
        and args.program_dir is None
        and len(args.program_files) == 0
    ):
        raise ValueError("Provide at least one of --experiment-dir, --program-dir, or --program-file.")

    bundles = load_offline_dataset(
        dataset_root=args.dataset_root,
        allowed_scenarios=args.scenarios,
        show_progress=(not bool(args.no_progress)),
    )
    heuristic_sampling = {
        "enabled": False,
    }
    heuristic_discovery_path = _resolve_heuristic_discovery_request(
        _resolve_project_path(args.dataset_root),
        args.heuristic_dynamics_discovery,
    )
    if heuristic_discovery_path is not None:
        if not heuristic_discovery_path.exists():
            raise FileNotFoundError(
                f"Heuristic dynamics discovery JSON not found: {heuristic_discovery_path}"
            )
        bundles, heuristic_sampling = _sample_bundles_by_heuristic_class(
            bundles=bundles,
            heuristic_discovery_json=heuristic_discovery_path,
            seed=DEFAULT_HEURISTIC_SAMPLE_SEED,
        )
    program_sources = load_program_sources(
        experiment_dir=args.experiment_dir,
        program_dir=args.program_dir,
        program_files=args.program_files,
    )
    report = evaluate_offline_dataset(
        bundles=bundles,
        program_sources=program_sources,
        workers=max(1, int(args.workers)),
        allow_imports=(not bool(args.disallow_imports)),
        show_progress=(not bool(args.no_progress)),
    )
    report["dataset"]["dataset_root"] = str(_resolve_project_path(args.dataset_root))
    report["dataset"]["heuristic_dynamics_discovery"] = heuristic_sampling

    if args.output_json is not None:
        output_path = _resolve_project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    dataset_payload = report["dataset"]
    print(f"Dataset root: {dataset_payload['dataset_root']}")
    print(f"Scenarios: {dataset_payload['scenario_count']}")
    print(f"Transitions: {dataset_payload['transition_count']}")
    if dataset_payload.get("heuristic_dynamics_discovery", {}).get("enabled"):
        heuristic_payload = dataset_payload["heuristic_dynamics_discovery"]
        print(
            "Heuristic dynamics sampling: "
            f"{heuristic_payload['class_count']} classes -> "
            f"{heuristic_payload['sampled_transition_count']} sampled transitions "
            f"(seed={heuristic_payload['sample_seed']})"
        )
    print(f"Workers: {dataset_payload['workers']}")
    for version_row in report["versions"]:
        program = version_row["program"]
        print(
            f"{program['version_id']} | acc={version_row['accuracy']:.4f} "
            f"| correct={version_row['correct_count']}/{version_row['total_count']} "
            f"| runtime_errors={version_row['runtime_error_count']} "
            f"| compile_errors={len(version_row['compile_errors'])}"
        )
    if args.output_json is not None:
        print(f"JSON report: {_resolve_project_path(args.output_json)}")


if __name__ == "__main__":
    main()
