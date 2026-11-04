from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collect.data_coverage import _resolve_project_path  # noqa: E402


DEFAULT_DATASET_ROOT = Path("test_dataset") / "coverage"
DEFAULT_OUTPUT_JSON = Path("test_dataset") / "heuristic_dynamics_discovery.json"
DEFAULT_OUTPUT_FILENAME = "heuristic_dynamics_discovery.json"

ObjectToken = Tuple[str, str, int, int, str]


@dataclass(frozen=True)
class TransitionBundle:
    dataset_root: Path
    dataset_label: str
    scenario_type: str
    artifact_stem: str
    transitions_path: Path
    states_path: Path
    summary_path: Optional[Path]
    transition_count: int


@dataclass(frozen=True)
class ParsedState:
    objects: Dict[ObjectToken, int]
    grid_size: Tuple[int, int]
    terminated: bool
    truncated: bool


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON at {path}")
    return payload


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return int(count)


def _normalize_action(raw_action: Any) -> str:
    return str(raw_action).strip().lower()


def _normalize_text(raw: Any) -> str:
    if raw is None:
        return ""
    value = str(raw).strip()
    return value.lower()


def _normalize_direction(raw: Any) -> str:
    value = _normalize_text(raw)
    if not value:
        return ""
    # Keep deterministic empty direction for non-directional objects.
    return value


def _normalize_object_token(
    *,
    obj_type: Any,
    word: Any,
    position: Any,
    direction: Any,
    has_direction: bool,
) -> Optional[ObjectToken]:
    if (
        not isinstance(position, (list, tuple))
        or len(position) != 2
        or not isinstance(position[0], (int, np.integer))
        or not isinstance(position[1], (int, np.integer))
    ):
        return None
    token_type = _normalize_text(obj_type)
    token_word = _normalize_text(word)
    token_direction = _normalize_direction(direction) if bool(has_direction) else ""
    return (
        token_type,
        token_word,
        int(position[0]),
        int(position[1]),
        token_direction,
    )


def _collect_bundle_dataset(
    *,
    dataset_root: Path,
    dataset_label: str,
) -> List[TransitionBundle]:
    bundles: List[TransitionBundle] = []
    for transitions_path in sorted(dataset_root.glob("*_transitions.jsonl")):
        artifact_stem = transitions_path.stem.removesuffix("_transitions")
        states_path = dataset_root / f"{artifact_stem}_states.npz"
        if not states_path.exists():
            continue

        summary_path: Optional[Path] = dataset_root / f"{artifact_stem}.json"
        scenario_type = artifact_stem
        transition_count = _count_jsonl_rows(transitions_path)

        if summary_path.exists():
            try:
                summary_payload = _read_json(summary_path)
                scenario_value = summary_payload.get("scenario_type")
                if isinstance(scenario_value, str) and scenario_value.strip():
                    scenario_type = scenario_value.strip()
                summary_count = summary_payload.get("transition_count")
                if isinstance(summary_count, int) and summary_count > 0:
                    transition_count = int(summary_count)
            except Exception:
                pass

        bundles.append(
            TransitionBundle(
                dataset_root=dataset_root,
                dataset_label=dataset_label,
                scenario_type=scenario_type,
                artifact_stem=artifact_stem,
                transitions_path=transitions_path.resolve(),
                states_path=states_path.resolve(),
                summary_path=(summary_path if summary_path.exists() else None),
                transition_count=transition_count,
            )
        )
    return bundles


def _discover_dataset_bundles(
    *,
    dataset_roots: Sequence[Path],
) -> List[TransitionBundle]:
    if not dataset_roots:
        raise ValueError("At least one dataset root is required.")

    all_bundles: List[TransitionBundle] = []
    for dataset_root in dataset_roots:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        bundles = _collect_bundle_dataset(
            dataset_root=dataset_root,
            dataset_label=dataset_root.name,
        )
        if not bundles:
            raise ValueError(f"No transition/state bundles found in {dataset_root}")
        all_bundles.extend(bundles)

    return sorted(
        all_bundles,
        key=lambda bundle: (str(bundle.dataset_root), bundle.scenario_type, bundle.artifact_stem),
    )


def _read_state_npz(*, path: Path) -> List[ParsedState]:
    with np.load(path, allow_pickle=False) as payload:
        grid_sizes = np.asarray(payload["grid_sizes"])
        terminated = np.asarray(payload["terminated"])
        truncated = np.asarray(payload["truncated"])
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

        states: List[ParsedState] = []
        for state_index in range(max(0, state_count)):
            raw_grid_size = (
                grid_sizes[state_index] if state_index < len(grid_sizes) else [0, 0]
            )
            if (
                isinstance(raw_grid_size, (list, tuple, np.ndarray))
                and len(raw_grid_size) >= 2
            ):
                try:
                    width = int(raw_grid_size[0])
                    height = int(raw_grid_size[1])
                except (TypeError, ValueError):
                    width = 0
                    height = 0
            else:
                width = 0
                height = 0

            start = int(object_offsets[state_index]) if state_index < len(object_offsets) else 0
            end = (
                int(object_offsets[state_index + 1])
                if state_index + 1 < len(object_offsets)
                else int(len(object_types))
            )
            if end < start:
                end = start

            object_counter: Dict[ObjectToken, int] = {}
            for object_index in range(start, end):
                token = _normalize_object_token(
                    obj_type=object_types[object_index],
                    word=object_words[object_index],
                    position=(int(object_x[object_index]), int(object_y[object_index])),
                    direction=object_directions[object_index],
                    has_direction=bool(object_has_direction[object_index]),
                )
                if token is None:
                    continue
                object_counter[token] = object_counter.get(token, 0) + 1

            states.append(
                ParsedState(
                    objects=object_counter,
                    grid_size=(max(0, width), max(0, height)),
                    terminated=bool(terminated[state_index]) if state_index < len(terminated) else False,
                    truncated=bool(truncated[state_index]) if state_index < len(truncated) else False,
                )
            )
        return states


def _counter_items_to_payload(
    counter: Mapping[ObjectToken, int],
) -> List[List[Any]]:
    return [
        [int(count), token[0], token[1], int(token[2]), int(token[3]), token[4]]
        for token, count in sorted(counter.items(), key=lambda item: item[0])
    ]


def _group_object_positions_by_signature(
    *,
    state_counter: Mapping[ObjectToken, int],
) -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
    grouped: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for token, count in state_counter.items():
        obj_type, obj_word, x, y, direction = token
        if count <= 0:
            continue
        key = (obj_type, obj_word, direction)
        grouped[key].extend([(int(x), int(y)) for _ in range(int(count))])
    return grouped


def _counter_to_payload(
    entries: Mapping[Tuple[str, str, int, int, str, str], int],
) -> List[List[Any]]:
    # entries key format: (kind, obj_type, obj_word, dx, dy, direction)
    payload: List[List[Any]] = []
    for (kind, obj_type, obj_word, dx, dy, direction), count in sorted(
        entries.items(), key=lambda item: item[0]
    ):
        payload.append(
            [
                int(count),
                str(kind),
                str(obj_type),
                str(obj_word),
                int(dx),
                int(dy),
                str(direction),
            ]
        )
    return payload


def _build_position_deltas(
    *,
    current_state: ParsedState,
    next_state: ParsedState,
) -> Tuple[Dict[Tuple[str, str, int, int, str, str], int], Dict[Tuple[str, str, int, int, str, str], int]]:
    current_counter = Counter(current_state.objects)
    next_counter = Counter(next_state.objects)
    removed = current_counter - next_counter
    added = next_counter - current_counter

    removed_by_sig = _group_object_positions_by_signature(state_counter=removed)
    added_by_sig = _group_object_positions_by_signature(state_counter=added)

    added_delta_counts: Dict[Tuple[str, str, int, int, str, str], int] = defaultdict(int)
    removed_delta_counts: Dict[Tuple[str, str, int, int, str, str], int] = defaultdict(int)

    object_signatures = set(removed_by_sig.keys()) | set(added_by_sig.keys())
    for sig in object_signatures:
        obj_type, obj_word, direction = sig
        removed_positions = sorted(removed_by_sig.get(sig, []))
        added_positions = sorted(added_by_sig.get(sig, []))

        removed_counter = Counter(removed_positions)
        added_counter = Counter(added_positions)
        unchanged = removed_counter & added_counter

        # Remove unchanged items first so we only describe actual changes.
        for position, count in unchanged.items():
            if count <= 0:
                continue
            removed_counter[position] -= count
            if removed_counter[position] <= 0:
                del removed_counter[position]
            added_counter[position] -= count
            if added_counter[position] <= 0:
                del added_counter[position]

        remained_removed = [
            position
            for position, count in sorted(removed_counter.items(), key=lambda item: item[0])
            for _ in range(int(count))
        ]
        remained_added = [
            position
            for position, count in sorted(added_counter.items(), key=lambda item: item[0])
            for _ in range(int(count))
        ]

        paired_count = min(len(remained_removed), len(remained_added))
        for idx in range(paired_count):
            removed_x, removed_y = remained_removed[idx]
            added_x, added_y = remained_added[idx]
            dx = int(added_x - removed_x)
            dy = int(added_y - removed_y)
            if dx == 0 and dy == 0:
                continue
            key = ("move", obj_type, obj_word, dx, dy, direction)
            added_delta_counts[key] += 1
            removed_delta_counts[key] += 1

        for idx in range(paired_count, len(remained_removed)):
            removed_x, removed_y = remained_removed[idx]
            key = ("remove", obj_type, obj_word, 0, 0, direction)
            removed_delta_counts[key] += 1

        for idx in range(paired_count, len(remained_added)):
            added_x, added_y = remained_added[idx]
            key = ("add", obj_type, obj_word, 0, 0, direction)
            added_delta_counts[key] += 1

    return added_delta_counts, removed_delta_counts


def _build_state_difference(
    current_state: ParsedState,
    next_state: ParsedState,
) -> Dict[str, Any]:
    added_delta_counts, removed_delta_counts = _build_position_deltas(
        current_state=current_state,
        next_state=next_state,
    )

    diff: Dict[str, Any] = {
        "added_objects": _counter_to_payload(added_delta_counts),
        "removed_objects": _counter_to_payload(removed_delta_counts),
    }
    if current_state.grid_size != next_state.grid_size:
        diff["grid_size"] = {
            "from": [current_state.grid_size[0], current_state.grid_size[1]],
            "to": [next_state.grid_size[0], next_state.grid_size[1]],
        }
    if current_state.terminated != next_state.terminated:
        diff["terminated"] = {
            "from": bool(current_state.terminated),
            "to": bool(next_state.terminated),
        }
    if current_state.truncated != next_state.truncated:
        diff["truncated"] = {
            "from": bool(current_state.truncated),
            "to": bool(next_state.truncated),
        }
    return diff


def _state_diff_to_key_payload(diff: Mapping[str, Any]) -> str:
    return json.dumps(
        {"state_diff": diff},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def discover_heuristic_dynamics(
    *,
    dataset_roots: Sequence[Path],
    show_progress: bool,
) -> Dict[str, Any]:
    resolved_roots = [_resolve_project_path(root) for root in dataset_roots]
    bundles = _discover_dataset_bundles(dataset_roots=resolved_roots)

    total_expected_transitions = sum(int(bundle.transition_count) for bundle in bundles)
    transition_iter = _tqdm(
        total=total_expected_transitions,
        desc="Classifying transitions",
        unit="transition",
        disable=(not show_progress) or _tqdm is None,
        leave=False,
    )

    class_id_by_signature: Dict[Tuple[str, str], int] = {}
    class_key_by_id: Dict[int, str] = {}
    class_counts: Dict[int, int] = defaultdict(int)
    class_state_diff_by_id: Dict[int, Dict[str, Any]] = {}
    class_sample_by_id: Dict[int, Dict[str, Any]] = {}
    bundle_mappings: List[Dict[str, Any]] = []

    transition_total_seen = 0
    valid_transitions = 0
    failed_rows = 0
    missing_refs = 0
    invalid_bundles: List[str] = []

    for bundle in bundles:
        try:
            states = _read_state_npz(path=bundle.states_path)
        except Exception:
            invalid_bundles.append(str(bundle.transitions_path))
            if transition_iter is not None:
                transition_iter.total = max(
                    0,
                    int((transition_iter.total or 0) - int(bundle.transition_count)),
                )
            continue

        transition_mapping: Dict[str, int] = {}
        line_index = 0

        with bundle.transitions_path.open("r", encoding="utf-8") as handle:
            for raw_row in handle:
                row_text = raw_row.strip()
                if not row_text:
                    continue
                transition_total_seen += 1
                try:
                    row = json.loads(row_text)
                except Exception:
                    failed_rows += 1
                    if transition_iter is not None:
                        transition_iter.update(1)
                    continue

                if not isinstance(row, Mapping):
                    failed_rows += 1
                    if transition_iter is not None:
                        transition_iter.update(1)
                    continue

                try:
                    transition_index = int(row.get("transition_index", line_index))
                    state_index = int(row["state_index"])
                    next_state_index = int(row["next_state_index"])
                    action = _normalize_action(row.get("action", ""))
                except Exception:
                    failed_rows += 1
                    if transition_iter is not None:
                        transition_iter.update(1)
                    continue

                line_index += 1
                if (
                    state_index < 0
                    or next_state_index < 0
                    or state_index >= len(states)
                    or next_state_index >= len(states)
                ):
                    missing_refs += 1
                    failed_rows += 1
                    if transition_iter is not None:
                        transition_iter.update(1)
                    continue

                state_diff = _build_state_difference(
                    current_state=states[state_index],
                    next_state=states[next_state_index],
                )
                signature_key = _state_diff_to_key_payload(state_diff)
                class_key = (action, signature_key)

                class_idx = class_id_by_signature.get(class_key)
                if class_idx is None:
                    class_idx = len(class_id_by_signature)
                    class_id_by_signature[class_key] = int(class_idx)
                    class_key_by_id[int(class_idx)] = action
                    class_state_diff_by_id[int(class_idx)] = state_diff
                    class_sample_by_id[int(class_idx)] = {
                        "dataset_root": str(bundle.dataset_root),
                        "artifact_stem": bundle.artifact_stem,
                        "scenario_type": bundle.scenario_type,
                        "transition_index": int(transition_index),
                        "transitions_path": str(bundle.transitions_path),
                        "states_path": str(bundle.states_path),
                    }

                class_counts[int(class_idx)] += 1
                transition_mapping[str(int(transition_index))] = int(class_idx)
                valid_transitions += 1

                if transition_iter is not None:
                    transition_iter.update(1)

        bundle_mappings.append(
            {
                "dataset_label": bundle.dataset_label,
                "dataset_root": str(bundle.dataset_root),
                "artifact_stem": bundle.artifact_stem,
                "scenario_type": bundle.scenario_type,
                "transitions_path": str(bundle.transitions_path),
                "states_path": str(bundle.states_path),
                "transition_count": int(bundle.transition_count),
                "transition_to_class": transition_mapping,
            }
        )

    if transition_iter is not None:
        if transition_iter.total is not None and transition_iter.n < int(transition_iter.total):
            transition_iter.total = int(transition_iter.n)
        transition_iter.close()

    class_metadata = [
        {
            "class_idx": int(class_idx),
            "action": str(class_key_by_id.get(int(class_idx), "")),
            "transition_count": int(class_counts.get(int(class_idx), 0)),
            "sample_transition": class_sample_by_id.get(int(class_idx)),
            "state_diff": class_state_diff_by_id.get(int(class_idx), {}),
        }
        for class_idx in sorted(class_counts.keys())
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": {
            "mode": "canonical_state_difference_plus_action",
            "signature": "transition state-diff computed from canonical object multiset delta + state flags",
        },
        "stats": {
            "dataset_root_count": int(len({str(bundle.dataset_root) for bundle in bundles})),
            "bundle_count": int(len(bundles)),
            "transition_rows_seen": int(transition_total_seen),
            "valid_transitions": int(valid_transitions),
            "failed_rows": int(failed_rows),
            "missing_state_references": int(missing_refs),
            "invalid_bundles": int(len(invalid_bundles)),
            "num_classes": int(len(class_counts)),
            "num_unique_class_signatures": int(len(class_id_by_signature)),
        },
        "inputs": [
            {
                "dataset_root": str(bundle.dataset_root),
                "dataset_label": bundle.dataset_label,
                "transition_count": int(bundle.transition_count),
            }
            for bundle in bundles
        ],
        "datasets": [
            {
                "dataset_root": str(bundle.dataset_root),
                "dataset_label": bundle.dataset_label,
                "artifact_stem": bundle.artifact_stem,
                "scenario_type": bundle.scenario_type,
                "transition_count": int(bundle.transition_count),
                "transitions_path": str(bundle.transitions_path),
                "states_path": str(bundle.states_path),
                "summary_path": (
                    str(bundle.summary_path) if bundle.summary_path is not None else None
                ),
            }
            for bundle in bundles
        ],
        "class_metadata": class_metadata,
        "class_counts": {str(int(k)): int(v) for k, v in sorted(class_counts.items())},
        "transition_class_mapping": bundle_mappings,
        "invalid_bundles": invalid_bundles,
        "class_signature_notes": [
            "difference_and_action_to_class = key(action, added_objects, removed_objects, grid_size, terminated, truncated)",
            "class index starts from 0 in order of first sighting.",
        ],
    }


def _resolve_output_json_path(
    *,
    dataset_roots: Sequence[Path],
    output_json: Optional[Path],
) -> Path:
    if output_json is not None:
        return _resolve_project_path(output_json)
    if len(dataset_roots) == 1:
        return _resolve_project_path(dataset_roots[0]) / DEFAULT_OUTPUT_FILENAME
    return _resolve_project_path(DEFAULT_OUTPUT_JSON)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover heuristic dynamics classes from offline transition datasets using "
            "canonical state difference + action."
        )
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        type=str,
        help=(
            "Dataset directory containing *_transitions.jsonl and *_states.npz. "
            "Can be repeated to combine coverage/solution datasets."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output JSON report path. By default, a single dataset root writes to "
            "<dataset-root>/heuristic_dynamics_discovery.json; multiple roots fall back to "
            "test_dataset/heuristic_dynamics_discovery.json."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable transition-level progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root:
        dataset_roots = [_resolve_project_path(root) for root in args.dataset_root]
    else:
        dataset_roots = [_resolve_project_path(DEFAULT_DATASET_ROOT)]

    report = discover_heuristic_dynamics(
        dataset_roots=dataset_roots,
        show_progress=(not bool(args.no_progress)),
    )

    output_path = _resolve_output_json_path(
        dataset_roots=dataset_roots,
        output_json=args.output_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Scanned dataset roots: {len(dataset_roots)}")
    print(f"Bundles: {report['stats']['bundle_count']}")
    print(f"Classes: {report['stats']['num_classes']}")
    print(f"Processed transitions: {report['stats']['valid_transitions']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
