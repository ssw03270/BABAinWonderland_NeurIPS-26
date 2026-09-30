from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
import json
import multiprocessing
from numbers import Integral
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import Transition
from src.environments.state_serializer import StateSerializer
from src.environments.custom_map_spec import load_custom_map_catalog, load_custom_map_spec
from src.program_model.state_codec import canonicalize_state_json, parse_state_json

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


DEFAULT_ACTION_ORDER = (1, 2, 3, 4, 0)
ACTION_NAME_TO_ID = {
    "idle": 0,
    "up": 1,
    "right": 2,
    "down": 3,
    "left": 4,
}
LARGE_HEURISTIC = 10_000
CARDINAL_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DEFAULT_COVERAGE_DIFFICULTY = "original"
DEFAULT_COVERAGE_SCENARIO_SPLIT = "all_test"
DEFAULT_COVERAGE_SPLIT_MANIFEST = Path("configs") / "custom_map_splits_dot_test.yaml"
DEFAULT_COVERAGE_OUTPUT_ROOT = Path("test_dataset") / "coverage"
DEFAULT_COVERAGE_MAX_TRANSITIONS = 100_000
DEFAULT_COVERAGE_ENV_MAX_STEPS = 100


def _normalized_word(obj: Mapping[str, Any]) -> Optional[str]:
    raw = obj.get("word")
    if not isinstance(raw, str):
        return None
    word = raw.strip().lower()
    return word or None


def _normalized_position(obj: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    raw = obj.get("position")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not isinstance(raw[0], int)
        or not isinstance(raw[1], int)
    ):
        return None
    return int(raw[0]), int(raw[1])


def _index_objects_by_position(objects: Iterable[Mapping[str, Any]]) -> Dict[Tuple[int, int], List[Mapping[str, Any]]]:
    indexed: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for obj in objects:
        position = _normalized_position(obj)
        if position is None:
            continue
        indexed[position].append(obj)
    return indexed


def _words_at(objects: Iterable[Mapping[str, Any]], allowed_types: Sequence[str]) -> List[str]:
    allowed = {item.strip().lower() for item in allowed_types}
    words: List[str] = []
    seen: set[str] = set()
    for obj in objects:
        obj_type = str(obj.get("type", "")).strip().lower()
        if obj_type not in allowed:
            continue
        word = _normalized_word(obj)
        if word is None or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def _first_rule_token(
    objects: Iterable[Mapping[str, Any]],
    allowed_types: Sequence[str],
) -> Optional[Tuple[str, str]]:
    allowed = {item.strip().lower() for item in allowed_types}
    for obj in objects:
        obj_type = str(obj.get("type", "")).strip().lower()
        if obj_type not in allowed:
            continue
        word = _normalized_word(obj)
        if word is None:
            continue
        return obj_type, word
    return None


def _collect_subject_segment(
    indexed: Mapping[Tuple[int, int], List[Mapping[str, Any]]],
    start: Tuple[int, int],
    delta: Tuple[int, int],
) -> List[Tuple[str, str]]:
    x, y = int(start[0]), int(start[1])
    dx, dy = int(delta[0]), int(delta[1])
    tokens: List[Tuple[str, str]] = []
    seen_noun = False
    expect_noun = True
    while True:
        token = _first_rule_token(
            indexed.get((x, y), ()),
            ("rule_noun", "rule_property", "rule_operator"),
        )
        if token is None:
            break
        token_type, token_word = token
        if expect_noun:
            if token_type == "rule_operator" and token_word == "and" and not seen_noun:
                tokens.append(token)
            elif token_type == "rule_noun":
                tokens.append(token)
                seen_noun = True
                expect_noun = False
            else:
                break
        else:
            if token_type != "rule_operator" or token_word != "and":
                break
            tokens.append(token)
            expect_noun = True
        x += dx
        y += dy
    tokens.reverse()
    return tokens


def _collect_predicate_segment(
    indexed: Mapping[Tuple[int, int], List[Mapping[str, Any]]],
    start: Tuple[int, int],
    delta: Tuple[int, int],
) -> List[Tuple[str, str]]:
    x, y = int(start[0]), int(start[1])
    dx, dy = int(delta[0]), int(delta[1])
    tokens: List[Tuple[str, str]] = []
    seen_item = False
    expect_item = True
    while True:
        token = _first_rule_token(
            indexed.get((x, y), ()),
            ("rule_noun", "rule_property", "rule_operator"),
        )
        if token is None:
            break
        token_type, token_word = token
        if expect_item:
            if token_type == "rule_operator" and token_word == "and" and not seen_item:
                tokens.append(token)
            elif token_type in {"rule_noun", "rule_property"}:
                tokens.append(token)
                seen_item = True
                expect_item = False
            else:
                break
        else:
            if token_type != "rule_operator" or token_word != "and":
                break
            tokens.append(token)
            expect_item = True
        x += dx
        y += dy
    return tokens


def _parse_conjoined_rule_words(
    tokens: Sequence[Tuple[str, str]],
    allowed_item_types: Sequence[str],
) -> Optional[Tuple[str, ...]]:
    trimmed = [token for token in tokens if token is not None]
    while trimmed and trimmed[0] == ("rule_operator", "and"):
        trimmed = trimmed[1:]
    while trimmed and trimmed[-1] == ("rule_operator", "and"):
        trimmed = trimmed[:-1]
    if not trimmed:
        return None

    allowed = {item.strip().lower() for item in allowed_item_types}
    words: List[str] = []
    expect_item = True
    for token_type, token_word in trimmed:
        if expect_item:
            if token_type not in allowed:
                return None
            words.append(token_word)
        else:
            if token_type != "rule_operator" or token_word != "and":
                return None
        expect_item = not expect_item
    if expect_item:
        return None
    return tuple(words)


def extract_active_rules(state_obj: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    objects = state_obj.get("objects", [])
    if not isinstance(objects, list):
        return ()

    indexed = _index_objects_by_position(objects)
    rules: set[Tuple[str, str]] = set()
    for (x, y), cell_objects in indexed.items():
        if "is" not in _words_at(cell_objects, ("rule_operator",)):
            continue
        for dx, dy in ((1, 0), (0, 1)):
            subject_words = _parse_conjoined_rule_words(
                _collect_subject_segment(indexed, (x - dx, y - dy), (-dx, -dy)),
                ("rule_noun",),
            )
            predicate_words = _parse_conjoined_rule_words(
                _collect_predicate_segment(indexed, (x + dx, y + dy), (dx, dy)),
                ("rule_property", "rule_noun"),
            )
            if not subject_words or not predicate_words:
                continue
            for subject_word in subject_words:
                for predicate_word in predicate_words:
                    rules.add((subject_word, predicate_word))
    return tuple(sorted(rules))


def _collect_text_positions(
    state_obj: Mapping[str, Any],
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[Tuple[int, int]]], List[Tuple[int, int]]]:
    noun_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    property_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    is_positions: List[Tuple[int, int]] = []
    objects = state_obj.get("objects", [])
    if not isinstance(objects, list):
        return noun_positions, property_positions, is_positions
    for obj in objects:
        obj_type = str(obj.get("type", "")).strip().lower()
        word = _normalized_word(obj)
        position = _normalized_position(obj)
        if word is None or position is None:
            continue
        if obj_type == "rule_noun":
            noun_positions[word].append(position)
        elif obj_type == "rule_property":
            property_positions[word].append(position)
        elif obj_type == "rule_operator" and word == "is":
            is_positions.append(position)
    return noun_positions, property_positions, is_positions


def _collect_world_positions(state_obj: Mapping[str, Any]) -> Dict[str, List[Tuple[int, int]]]:
    world_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    objects = state_obj.get("objects", [])
    if not isinstance(objects, list):
        return world_positions
    for obj in objects:
        if str(obj.get("type", "")).strip().lower() != "world_object":
            continue
        word = _normalized_word(obj)
        position = _normalized_position(obj)
        if word is None or position is None:
            continue
        world_positions[word].append(position)
    return world_positions


def analyze_state(state_obj: Mapping[str, Any]) -> Dict[str, Any]:
    active_rules = extract_active_rules(state_obj)
    world_positions = _collect_world_positions(state_obj)
    noun_positions, property_positions, is_positions = _collect_text_positions(state_obj)
    raw_grid_size = state_obj.get("grid_size", [])
    if (
        isinstance(raw_grid_size, list)
        and len(raw_grid_size) == 2
        and isinstance(raw_grid_size[0], int)
        and isinstance(raw_grid_size[1], int)
    ):
        grid_size = (int(raw_grid_size[0]), int(raw_grid_size[1]))
    else:
        grid_size = (0, 0)

    subjects_by_property: Dict[str, List[str]] = defaultdict(list)
    for subject, predicate in active_rules:
        subjects_by_property[predicate].append(subject)
    you_subjects = sorted(subject for subject, predicate in active_rules if predicate == "you")
    win_subjects = sorted(subject for subject, predicate in active_rules if predicate == "win")
    you_positions: List[Tuple[int, int]] = []
    win_positions: List[Tuple[int, int]] = []
    for subject in you_subjects:
        you_positions.extend(world_positions.get(subject, ()))
    for subject in win_subjects:
        win_positions.extend(world_positions.get(subject, ()))
    overlap = sorted(set(you_positions) & set(win_positions))
    return {
        "grid_size": grid_size,
        "active_rules": tuple(f"{subject} is {predicate}" for subject, predicate in active_rules),
        "raw_rules": active_rules,
        "you_positions": tuple(sorted(you_positions)),
        "win_positions": tuple(sorted(win_positions)),
        "goal_overlap": tuple(overlap),
        "subjects_by_property": {key: tuple(sorted(set(value))) for key, value in subjects_by_property.items()},
        "noun_positions": {key: tuple(value) for key, value in noun_positions.items()},
        "property_positions": {key: tuple(value) for key, value in property_positions.items()},
        "is_positions": tuple(is_positions),
        "world_positions": {key: tuple(value) for key, value in world_positions.items()},
    }


def _valid_grid_position(pos: Tuple[int, int], *, width: int, height: int) -> bool:
    return 0 <= int(pos[0]) < int(width) and 0 <= int(pos[1]) < int(height)


def _collect_property_position_set(analysis: Mapping[str, Any], property_name: str) -> set[Tuple[int, int]]:
    world_positions = analysis.get("world_positions", {})
    subjects_by_property = analysis.get("subjects_by_property", {})
    if not isinstance(world_positions, Mapping) or not isinstance(subjects_by_property, Mapping):
        return set()
    positions: set[Tuple[int, int]] = set()
    for subject in subjects_by_property.get(property_name, ()):
        for raw_pos in world_positions.get(subject, ()):
            if isinstance(raw_pos, tuple) and len(raw_pos) == 2:
                positions.add((int(raw_pos[0]), int(raw_pos[1])))
    return positions


def _estimate_goal_route_cost(analysis: Mapping[str, Any]) -> int:
    raw_grid_size = analysis.get("grid_size", ())
    if (
        not isinstance(raw_grid_size, (list, tuple))
        or len(raw_grid_size) != 2
        or not isinstance(raw_grid_size[0], int)
        or not isinstance(raw_grid_size[1], int)
    ):
        return LARGE_HEURISTIC
    width = int(raw_grid_size[0])
    height = int(raw_grid_size[1])
    you_positions = tuple(pos for pos in analysis.get("you_positions", ()) if isinstance(pos, tuple) and len(pos) == 2)
    win_positions = tuple(pos for pos in analysis.get("win_positions", ()) if isinstance(pos, tuple) and len(pos) == 2)
    if width <= 0 or height <= 0 or not you_positions or not win_positions:
        return LARGE_HEURISTIC

    frontier: List[Tuple[int, Tuple[int, int]]] = [(0, (int(pos[0]), int(pos[1]))) for pos in you_positions]
    best_cost = {node: cost for cost, node in frontier}
    goal_cells = {(int(pos[0]), int(pos[1])) for pos in win_positions if _valid_grid_position((int(pos[0]), int(pos[1])), width=width, height=height)}
    stop_cells = _collect_property_position_set(analysis, "stop")
    push_cells = _collect_property_position_set(analysis, "push")
    hazard_cells = (
        _collect_property_position_set(analysis, "sink")
        | _collect_property_position_set(analysis, "defeat")
        | _collect_property_position_set(analysis, "hot")
    )
    while frontier:
        frontier.sort(key=lambda item: item[0])
        cost, pos = frontier.pop(0)
        if cost > best_cost.get(pos, LARGE_HEURISTIC):
            continue
        if pos in goal_cells:
            return int(cost)
        for dx, dy in CARDINAL_DELTAS:
            nxt = (int(pos[0]) + dx, int(pos[1]) + dy)
            if not _valid_grid_position(nxt, width=width, height=height) or nxt in stop_cells:
                continue
            step_cost = 1 + (2 if nxt in push_cells else 0) + (4 if nxt in hazard_cells and nxt not in goal_cells else 0)
            next_cost = int(cost) + int(step_cost)
            if next_cost >= best_cost.get(nxt, LARGE_HEURISTIC):
                continue
            best_cost[nxt] = next_cost
            frontier.append((next_cost, nxt))
    return LARGE_HEURISTIC


def estimate_analysis_heuristic(analysis: Mapping[str, Any]) -> int:
    if tuple(analysis.get("goal_overlap", ())):
        return 0
    return _estimate_goal_route_cost(analysis)


def _freeze_state_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value.keys(), key=lambda item: str(item)):
            items.append((str(key), _freeze_state_value(value[key])))
        return tuple(items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_state_value(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return repr(value)


def _state_key(state_obj: Mapping[str, Any]) -> Any:
    raw_grid_size = state_obj.get("grid_size", [])
    if (
        isinstance(raw_grid_size, (list, tuple))
        and len(raw_grid_size) == 2
        and isinstance(raw_grid_size[0], Integral)
        and isinstance(raw_grid_size[1], Integral)
    ):
        grid_size = (int(raw_grid_size[0]), int(raw_grid_size[1]))
    else:
        grid_size = (0, 0)

    raw_step = state_obj.get("step", {})
    terminated = bool(raw_step.get("terminated", False)) if isinstance(raw_step, Mapping) else False

    object_rows: List[Tuple[str, str, int, int, str]] = []
    raw_objects = state_obj.get("objects", [])
    if isinstance(raw_objects, list):
        for raw_obj in raw_objects:
            if not isinstance(raw_obj, Mapping):
                continue
            raw_position = raw_obj.get("position", [])
            if (
                isinstance(raw_position, (list, tuple))
                and len(raw_position) == 2
                and isinstance(raw_position[0], Integral)
                and isinstance(raw_position[1], Integral)
            ):
                x = int(raw_position[0])
                y = int(raw_position[1])
            else:
                x = 0
                y = 0
            object_rows.append(
                (
                    str(raw_obj.get("type", "")),
                    str(raw_obj.get("word", "")),
                    x,
                    y,
                    str(raw_obj.get("direction", "")),
                )
            )
    object_rows.sort(key=lambda row: (row[3], row[2], row[0], row[4], row[1]))

    extras = {
        key: value
        for key, value in state_obj.items()
        if key not in {"grid_size", "step", "objects"}
    }
    return (
        grid_size[0],
        grid_size[1],
        terminated,
        tuple(object_rows),
        _freeze_state_value(extras),
    )


def _state_key_from_archive_input(raw_payload: Mapping[str, Any]) -> Any:
    raw_grid_size = raw_payload.get("grid_size", [])
    if (
        isinstance(raw_grid_size, (list, tuple))
        and len(raw_grid_size) == 2
        and isinstance(raw_grid_size[0], Integral)
        and isinstance(raw_grid_size[1], Integral)
    ):
        raw_width = int(raw_grid_size[0])
        raw_height = int(raw_grid_size[1])
    else:
        raw_width = 0
        raw_height = 0

    width = max(0, raw_width - 2)
    height = max(0, raw_height - 2)
    terminated = bool(raw_payload.get("terminated", False))

    object_rows: List[Tuple[str, str, int, int, str]] = []
    rows_sorted = True
    previous_sort_key: Optional[Tuple[int, int, str, str, str]] = None
    raw_objects = raw_payload.get("objects", [])
    if isinstance(raw_objects, list):
        for raw_obj in raw_objects:
            if not isinstance(raw_obj, Mapping):
                continue
            raw_position = raw_obj.get("position", [])
            if (
                not isinstance(raw_position, (list, tuple))
                or len(raw_position) != 2
                or not isinstance(raw_position[0], Integral)
                or not isinstance(raw_position[1], Integral)
            ):
                continue
            raw_x = int(raw_position[0])
            raw_y = int(raw_position[1])
            if (
                raw_x <= 0
                or raw_y <= 0
                or raw_x >= raw_width - 1
                or raw_y >= raw_height - 1
            ):
                continue

            raw_direction = raw_obj.get("direction")
            if isinstance(raw_direction, str):
                direction = raw_direction.strip()
            elif isinstance(raw_direction, Integral):
                direction = StateSerializer.DIRECTION_NAMES.get(int(raw_direction), "unknown")
            else:
                direction = ""

            row = (
                str(raw_obj.get("type", "")),
                str(raw_obj.get("word", "")),
                raw_x - 1,
                raw_y - 1,
                direction,
            )
            sort_key = (row[3], row[2], row[0], row[4], row[1])
            if previous_sort_key is not None and sort_key < previous_sort_key:
                rows_sorted = False
            previous_sort_key = sort_key
            object_rows.append(row)

    if not rows_sorted:
        object_rows.sort(key=lambda row: (row[3], row[2], row[0], row[4], row[1]))
    return (width, height, terminated, tuple(object_rows), ())


def _infer_num_actions(env: Any) -> int:
    raw_num_actions = getattr(env, "num_actions", None)
    if isinstance(raw_num_actions, Integral):
        return int(raw_num_actions)
    action_space = getattr(env, "action_space", None)
    if action_space is not None and isinstance(getattr(action_space, "n", None), Integral):
        return int(action_space.n)
    return len(DEFAULT_ACTION_ORDER)


def _safe_get_action_name(env: Any, action: int) -> str:
    fn = getattr(env, "get_action_name", None)
    if callable(fn):
        try:
            value = fn(int(action))
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    inverse = {value: key for key, value in ACTION_NAME_TO_ID.items()}
    return inverse.get(int(action), f"action_{int(action)}")


def _format_action_path(action_names: Sequence[str]) -> str:
    return " -> ".join(str(name) for name in action_names) if action_names else "<empty>"


def _format_duration(seconds: float) -> str:
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if total_seconds < 0:
        total_seconds = 0.0
    rounded_seconds = int(round(total_seconds))
    if rounded_seconds >= 3600:
        hours = rounded_seconds // 3600
        minutes = (rounded_seconds % 3600) // 60
        secs = rounded_seconds % 60
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if rounded_seconds >= 60:
        minutes = rounded_seconds // 60
        secs = rounded_seconds % 60
        return f"{minutes}m{secs:02d}s"
    if total_seconds >= 10:
        return f"{rounded_seconds}s"
    return f"{total_seconds:.1f}s"


def _sanitize_name(value: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "_" for char in str(value)]
    normalized = "".join(cleaned).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "unknown"


def _build_run_output_dir(*, label: str, search_mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "solver_artifacts" / f"{_sanitize_name(label)}_{_sanitize_name(search_mode)}_{timestamp}"


def _write_solver_summary(*, path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _normalize_archive_object_input(raw_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_obj, Mapping):
        return None

    raw_position = raw_obj.get("position", [])
    if (
        isinstance(raw_position, (list, tuple))
        and len(raw_position) == 2
        and isinstance(raw_position[0], Integral)
        and isinstance(raw_position[1], Integral)
    ):
        position = [int(raw_position[0]), int(raw_position[1])]
    else:
        position = [0, 0]

    row: Dict[str, Any] = {
        "type": str(raw_obj.get("type", "")),
        "word": str(raw_obj.get("word", "")),
        "position": position,
    }
    raw_direction = raw_obj.get("direction")
    if isinstance(raw_direction, str) and raw_direction.strip():
        row["direction"] = raw_direction.strip()
    elif isinstance(raw_direction, Integral):
        row["direction"] = StateSerializer.DIRECTION_NAMES.get(int(raw_direction), "unknown")
    return row


def _normalize_state_archive_input(
    raw_payload: Mapping[str, Any],
    *,
    default_source: str,
) -> Dict[str, Any]:
    raw_grid_size = raw_payload.get("grid_size", [])
    if (
        isinstance(raw_grid_size, (list, tuple))
        and len(raw_grid_size) == 2
        and isinstance(raw_grid_size[0], Integral)
        and isinstance(raw_grid_size[1], Integral)
    ):
        grid_size = [int(raw_grid_size[0]), int(raw_grid_size[1])]
    else:
        grid_size = [0, 0]

    objects: List[Dict[str, Any]] = []
    raw_objects = raw_payload.get("objects", [])
    if isinstance(raw_objects, list):
        for raw_obj in raw_objects:
            normalized = _normalize_archive_object_input(raw_obj)
            if normalized is not None:
                objects.append(normalized)

    raw_event = raw_payload.get("event", "")
    event = str(raw_event).strip() if isinstance(raw_event, str) else ""

    try:
        reward = float(raw_payload.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        reward = 0.0

    raw_source = raw_payload.get("source")
    if isinstance(raw_source, str) and raw_source.strip():
        source = raw_source.strip()
    else:
        source = str(default_source).strip() or "unknown"

    return {
        "source": source,
        "grid_size": grid_size,
        "objects": objects,
        "event": event,
        "reward": float(reward),
        "terminated": bool(raw_payload.get("terminated", False)),
        "truncated": bool(raw_payload.get("truncated", False)),
    }


def _state_archive_input_from_state_obj(state_obj: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(state_obj)
    raw_grid_size = [2, 2]
    canonical_grid_size = payload.get("grid_size", [])
    if (
        isinstance(canonical_grid_size, (list, tuple))
        and len(canonical_grid_size) == 2
        and isinstance(canonical_grid_size[0], Integral)
        and isinstance(canonical_grid_size[1], Integral)
    ):
        raw_grid_size = [
            max(0, int(canonical_grid_size[0])) + 2,
            max(0, int(canonical_grid_size[1])) + 2,
        ]

    raw_objects: List[Dict[str, Any]] = []
    objects = payload.get("objects", [])
    if isinstance(objects, list):
        for raw_obj in objects:
            normalized = _normalize_archive_object_input(raw_obj)
            if normalized is None:
                continue
            normalized["position"] = [
                int(normalized["position"][0]) + 1,
                int(normalized["position"][1]) + 1,
            ]
            raw_objects.append(normalized)

    step = payload.get("step", {})
    terminated = bool(step.get("terminated", False)) if isinstance(step, Mapping) else False
    return _normalize_state_archive_input(
        {
            "source": "canonical_state_fallback",
            "grid_size": raw_grid_size,
            "objects": raw_objects,
            "event": "",
            "reward": 0.0,
            "terminated": terminated,
            "truncated": False,
        },
        default_source="canonical_state_fallback",
    )


def archive_state_input_to_state_obj(
    state_input: Mapping[str, Any],
    *,
    normalize_input: bool = True,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if normalize_input:
        payload = _normalize_state_archive_input(state_input, default_source="unknown")
    elif isinstance(state_input, dict):
        payload = state_input
    else:
        payload = dict(state_input)
    serializer = StateSerializer(format_type="json", word_aliases=word_aliases)
    raw_width = int(payload["grid_size"][0])
    raw_height = int(payload["grid_size"][1])
    width, height = serializer._trim_grid_size(raw_width, raw_height)
    objects = serializer._normalize_objects(
        list(payload["objects"]),
        raw_width=raw_width,
        raw_height=raw_height,
    )
    return {
        "grid_size": [int(width), int(height)],
        "step": {"terminated": bool(payload["terminated"])},
        "objects": objects,
    }


def archive_state_input_to_canonical_text(
    state_input: Mapping[str, Any],
    *,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    return canonicalize_state_json(
        archive_state_input_to_state_obj(
            state_input,
            word_aliases=word_aliases,
        )
    )


def _state_obj_from_state_value(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        payload = state
    elif isinstance(state, Mapping):
        payload = dict(state)
    elif isinstance(state, str):
        payload = parse_state_json(state)
    else:
        payload = parse_state_json(
            json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    if not isinstance(payload.get("objects", []), list):
        payload["objects"] = []
    if not isinstance(payload.get("step", {}), Mapping):
        payload["step"] = {"terminated": False}
    return payload


@dataclass(slots=True)
class StatePacket:
    state_key: Any
    state_obj: Optional[Dict[str, Any]]
    state_archive_input: Dict[str, Any]


def _build_transition_archive_payload(
    transition_records: Sequence["TransitionRecord"],
    state_archive_inputs: Optional[Mapping[Any, Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    state_index_by_key: Dict[Any, int] = {}
    state_payloads: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []

    def get_state_index(state_key: Any) -> int:
        existing = state_index_by_key.get(state_key)
        if existing is not None:
            return int(existing)
        state_index = len(state_payloads)
        state_index_by_key[state_key] = state_index
        archive_payload = None
        if state_archive_inputs is not None:
            archive_payload = state_archive_inputs.get(state_key)
        if isinstance(archive_payload, dict):
            normalized_payload = archive_payload
        elif isinstance(archive_payload, Mapping):
            normalized_payload = dict(archive_payload)
        else:
            normalized_payload = _normalize_state_archive_input(
                {
                    "source": "missing_archive_input",
                    "grid_size": [0, 0],
                    "objects": [],
                    "event": "",
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                },
                default_source="missing_archive_input",
            )
        state_payloads.append(normalized_payload)
        return int(state_index)

    for transition_index, transition in enumerate(transition_records):
        state_index = get_state_index(transition.state_key)
        next_state_index = get_state_index(transition.next_state_key)
        transition_rows.append(
            {
                "transition_index": int(transition_index),
                "state_index": int(state_index),
                "next_state_index": int(next_state_index),
                "action": str(transition.action),
                "reward": float(transition.reward),
                "done": bool(transition.done),
            }
        )

    return state_payloads, transition_rows


def _write_state_archive_npz(*, path: Path, state_payloads: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_sizes: List[Tuple[int, int]] = []
    events: List[str] = []
    rewards: List[float] = []
    terminated_flags: List[bool] = []
    truncated_flags: List[bool] = []
    state_sources: List[str] = []
    object_offsets: List[int] = [0]
    object_types: List[str] = []
    object_words: List[str] = []
    object_x: List[int] = []
    object_y: List[int] = []
    object_has_direction: List[bool] = []
    object_directions: List[str] = []

    for payload in state_payloads:
        raw_grid_size = payload.get("grid_size", [])
        if (
            isinstance(raw_grid_size, list)
            and len(raw_grid_size) == 2
            and isinstance(raw_grid_size[0], int)
            and isinstance(raw_grid_size[1], int)
        ):
            grid_sizes.append((int(raw_grid_size[0]), int(raw_grid_size[1])))
        else:
            grid_sizes.append((0, 0))

        raw_event = payload.get("event", "")
        events.append(str(raw_event).strip() if isinstance(raw_event, str) else "")
        try:
            rewards.append(float(payload.get("reward", 0.0) or 0.0))
        except (TypeError, ValueError):
            rewards.append(0.0)
        terminated_flags.append(bool(payload.get("terminated", False)))
        truncated_flags.append(bool(payload.get("truncated", False)))
        raw_source = payload.get("source", "")
        state_sources.append(str(raw_source).strip() if isinstance(raw_source, str) else "")

        raw_objects = payload.get("objects", [])
        if isinstance(raw_objects, list):
            for raw_obj in raw_objects:
                if not isinstance(raw_obj, Mapping):
                    continue
                object_types.append(str(raw_obj.get("type", "")))
                object_words.append(str(raw_obj.get("word", "")))
                raw_position = raw_obj.get("position", [])
                if (
                    isinstance(raw_position, list)
                    and len(raw_position) == 2
                    and isinstance(raw_position[0], int)
                    and isinstance(raw_position[1], int)
                ):
                    object_x.append(int(raw_position[0]))
                    object_y.append(int(raw_position[1]))
                else:
                    object_x.append(0)
                    object_y.append(0)
                raw_direction = raw_obj.get("direction")
                has_direction = isinstance(raw_direction, str) and bool(raw_direction.strip())
                object_has_direction.append(bool(has_direction))
                object_directions.append(str(raw_direction).strip() if has_direction else "")
        object_offsets.append(len(object_types))

    np.savez_compressed(
        path,
        format_version=np.asarray([2], dtype=np.int32),
        state_count=np.asarray([len(state_payloads)], dtype=np.int64),
        object_count=np.asarray([len(object_types)], dtype=np.int64),
        grid_sizes=np.asarray(grid_sizes, dtype=np.int32),
        events=np.asarray(events, dtype=np.str_),
        rewards=np.asarray(rewards, dtype=np.float64),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        state_sources=np.asarray(state_sources, dtype=np.str_),
        object_offsets=np.asarray(object_offsets, dtype=np.int64),
        object_types=np.asarray(object_types, dtype=np.str_),
        object_words=np.asarray(object_words, dtype=np.str_),
        object_x=np.asarray(object_x, dtype=np.int32),
        object_y=np.asarray(object_y, dtype=np.int32),
        object_has_direction=np.asarray(object_has_direction, dtype=np.bool_),
        object_directions=np.asarray(object_directions, dtype=np.str_),
    )
    return path


def _write_transition_rows(*, path: Path, transition_rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in transition_rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")
    return path


def _resolve_project_path(path_or_str: str | Path) -> Path:
    path = Path(path_or_str)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def _resolve_custom_map_folder(path_or_str: str | Path) -> Path:
    return _resolve_project_path(path_or_str)


def _load_existing_coverage_summary(
    *,
    artifact_root: Path,
    artifact_stem: str,
) -> Optional[Dict[str, Any]]:
    summary_path = artifact_root / f"{artifact_stem}.json"
    states_path = artifact_root / f"{artifact_stem}_states.npz"
    transitions_path = artifact_root / f"{artifact_stem}_transitions.jsonl"
    if not summary_path.exists() or not states_path.exists() or not transitions_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    normalized = dict(payload)
    normalized["states_path"] = str(states_path)
    normalized["transitions_path"] = str(transitions_path)
    return normalized


def _build_existing_coverage_result(
    *,
    summary_payload: Mapping[str, Any],
    scenario_type: str,
    difficulty: str,
    map_path: str,
    artifact_stem: str,
) -> Dict[str, Any]:
    stats_payload = summary_payload.get("stats") if isinstance(summary_payload.get("stats"), Mapping) else {}
    diagnostics_payload = (
        summary_payload.get("diagnostics")
        if isinstance(summary_payload.get("diagnostics"), Mapping)
        else {}
    )
    elapsed_seconds = float(
        stats_payload.get(
            "elapsed_seconds",
            summary_payload.get("elapsed_seconds", 0.0),
        )
        or 0.0
    )
    return {
        "scenario_type": str(scenario_type),
        "difficulty": str(summary_payload.get("difficulty", difficulty)),
        "map_path": str(summary_payload.get("map_path", map_path)),
        "artifact_stem": str(summary_payload.get("artifact_stem", artifact_stem)),
        "solved": bool(summary_payload.get("solved", False)),
        "transition_count": int(summary_payload.get("transition_count", 0) or 0),
        "visited_states": int(
            summary_payload.get(
                "visited_states",
                summary_payload.get("archived_state_count", 0),
            )
            or 0
        ),
        "expanded_states": int(stats_payload.get("expanded_states", 0) or 0),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": str(stats_payload.get("elapsed_human", _format_duration(elapsed_seconds))),
        "transitions_path": summary_payload.get("transitions_path"),
        "states_path": summary_payload.get("states_path"),
        "stop_reason": "skipped_existing",
        "previous_stop_reason": diagnostics_payload.get(
            "stop_reason",
            summary_payload.get("stop_reason"),
        ),
        "skipped_existing": True,
    }


def _discover_custom_map_files(folder: str | Path) -> List[Path]:
    resolved_folder = _resolve_custom_map_folder(folder)
    return sorted(path.resolve() for path in resolved_folder.glob("*.json") if path.is_file())


def _build_map_artifact_stem(*, folder_name: str, map_path: Path) -> str:
    return f"{_sanitize_name(folder_name)}_{_sanitize_name(map_path.stem)}"


def _resolve_env_from_args(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    return str(args.env).strip(), (str(args.scenario).strip() if isinstance(args.scenario, str) and args.scenario.strip() else None)


@dataclass(slots=True)
class SearchNode:
    state_key: Any
    parent_index: int
    action_from_parent: int
    depth: int
    snapshot: Any = None
    process_owner: int = -1
    process_handle: int = -1


@dataclass(slots=True)
class WorkerNodeRequest:
    node_index: int
    state_key: Any
    action_path: Optional[Tuple[int, ...]]
    snapshot: Any = None


@dataclass(slots=True)
class WorkerTransition:
    action_id: int
    action_name: str
    next_state_key: Any
    next_state_obj: Optional[Dict[str, Any]]
    next_state_archive_input: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    is_win: bool
    next_state_snapshot: Any = None
    next_state_handle: int = -1


@dataclass(slots=True)
class WorkerNodeResult:
    node_index: int
    transitions: List[WorkerTransition]
    replay_steps: int = 0
    snapshot_restores: int = 0


@dataclass(slots=True)
class TransitionRecord:
    state_key: Any
    action: str
    next_state_key: Any
    reward: float = 0.0
    done: bool = False


@dataclass(frozen=True, slots=True)
class ProcessWorkerInit:
    env_factory: Callable[[], Any]
    seed: int
    action_order: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProcessExpandRequest:
    node_index: int
    state_key: Any
    handle: int


@dataclass(frozen=True, slots=True)
class ProcessMaterializeRequest:
    node_index: int
    state_key: Any
    action_path: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProcessMaterializeResult:
    node_index: int
    handle: int
    state_key: Any


@dataclass(frozen=True, slots=True)
class ProcessResetResponse:
    worker_id: int
    handle: int
    state_key: Any
    state_obj: Optional[Dict[str, Any]]
    state_archive_input: Dict[str, Any]


@dataclass(slots=True)
class PendingProcessMigration:
    state_key: Any
    parent_index: int
    action_from_parent: int
    depth: int
    source_worker: int
    source_handle: int
    target_worker: int


class _ProcessActorPool:
    def __init__(self, *, init_payload: ProcessWorkerInit, workers: int) -> None:
        self.workers = max(1, int(workers))
        self._ctx = multiprocessing.get_context("spawn")
        self._request_queues: List[Any] = []
        self._response_queues: List[Any] = []
        self._processes: List[Any] = []
        self._closed = False

        for worker_id in range(self.workers):
            request_queue = self._ctx.Queue()
            response_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_process_actor_worker_main,
                args=(int(worker_id), request_queue, response_queue, init_payload),
                daemon=True,
            )
            process.start()
            self._request_queues.append(request_queue)
            self._response_queues.append(response_queue)
            self._processes.append(process)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker_id in range(self.workers):
            try:
                self._request_queues[worker_id].put(("shutdown", None))
            except Exception:
                pass
        for process in self._processes:
            try:
                process.join(timeout=5.0)
            except Exception:
                pass
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for queue_obj in (*self._request_queues, *self._response_queues):
            try:
                queue_obj.close()
            except Exception:
                pass

    def _submit(self, worker_id: int, command: str, payload: Any) -> None:
        self._request_queues[int(worker_id)].put((str(command), payload))

    def _recv(self, worker_id: int, expected_command: str) -> Any:
        command, payload = self._response_queues[int(worker_id)].get()
        if str(command) != str(expected_command):
            raise RuntimeError(
                f"Unexpected process worker response from worker {int(worker_id)}: "
                f"expected={expected_command} got={command}"
            )
        return payload

    def reset_root(self, *, seed: int, worker_id: int = 0) -> ProcessResetResponse:
        worker_id = int(worker_id)
        self._submit(worker_id, "reset", {"seed": int(seed)})
        payload = self._recv(worker_id, "reset")
        if not isinstance(payload, ProcessResetResponse):
            raise RuntimeError("Process worker returned an invalid reset payload.")
        return payload

    def expand(self, grouped_requests: Mapping[int, Sequence[ProcessExpandRequest]]) -> Dict[int, List[WorkerNodeResult]]:
        active_workers = [int(worker_id) for worker_id, requests in grouped_requests.items() if requests]
        for worker_id in active_workers:
            self._submit(worker_id, "expand", tuple(grouped_requests[worker_id]))
        results: Dict[int, List[WorkerNodeResult]] = {}
        for worker_id in active_workers:
            payload = self._recv(worker_id, "expand")
            if not isinstance(payload, list):
                raise RuntimeError("Process worker returned an invalid expand payload.")
            results[worker_id] = payload
        return results

    def materialize_paths(
        self,
        grouped_requests: Mapping[int, Sequence[ProcessMaterializeRequest]],
    ) -> Dict[int, List[ProcessMaterializeResult]]:
        active_workers = [int(worker_id) for worker_id, requests in grouped_requests.items() if requests]
        for worker_id in active_workers:
            self._submit(worker_id, "materialize_paths", tuple(grouped_requests[worker_id]))
        results: Dict[int, List[ProcessMaterializeResult]] = {}
        for worker_id in active_workers:
            payload = self._recv(worker_id, "materialize_paths")
            if not isinstance(payload, list):
                raise RuntimeError("Process worker returned an invalid materialize_paths payload.")
            results[worker_id] = payload
        return results

    def export_handles(self, grouped_handles: Mapping[int, Sequence[int]]) -> Dict[int, List[Any]]:
        active_workers = [int(worker_id) for worker_id, handles in grouped_handles.items() if handles]
        for worker_id in active_workers:
            self._submit(worker_id, "export", tuple(int(handle) for handle in grouped_handles[worker_id]))
        results: Dict[int, List[Any]] = {}
        for worker_id in active_workers:
            payload = self._recv(worker_id, "export")
            if not isinstance(payload, list):
                raise RuntimeError("Process worker returned an invalid export payload.")
            results[worker_id] = payload
        return results

    def import_snapshots(self, grouped_snapshots: Mapping[int, Sequence[Any]]) -> Dict[int, List[int]]:
        active_workers = [int(worker_id) for worker_id, snapshots in grouped_snapshots.items() if snapshots]
        for worker_id in active_workers:
            self._submit(worker_id, "import", list(grouped_snapshots[worker_id]))
        results: Dict[int, List[int]] = {}
        for worker_id in active_workers:
            payload = self._recv(worker_id, "import")
            if not isinstance(payload, list):
                raise RuntimeError("Process worker returned an invalid import payload.")
            results[worker_id] = [int(handle) for handle in payload]
        return results

    def release_handles(self, grouped_handles: Mapping[int, Sequence[int]]) -> None:
        for worker_id, handles in grouped_handles.items():
            normalized = [int(handle) for handle in handles if int(handle) >= 0]
            if not normalized:
                continue
            self._submit(int(worker_id), "release", tuple(normalized))


def _process_actor_worker_main(
    worker_id: int,
    request_queue: Any,
    response_queue: Any,
    init_payload: ProcessWorkerInit,
) -> None:
    worker_env = init_payload.env_factory()
    solver = BabaSolver(
        env=worker_env,
        seed=int(init_payload.seed),
        search_mode="bfs",
        action_order=tuple(int(action) for action in init_payload.action_order),
        workers=1,
        parallel_backend="sequential",
        chunk_size=1,
        return_child_snapshots=True,
        show_progress=False,
        progress_leave=False,
    )
    solver._thread_local.env = worker_env
    snapshot_store: Dict[int, Any] = {}
    next_handle = 0

    def allocate(snapshot: Any) -> int:
        nonlocal next_handle
        next_handle += 1
        snapshot_store[int(next_handle)] = snapshot
        return int(next_handle)

    try:
        while True:
            command, payload = request_queue.get()
            if str(command) == "shutdown":
                break

            if str(command) == "reset":
                snapshot_store.clear()
                next_handle = 0
                reset_seed = int(payload.get("seed", init_payload.seed)) if isinstance(payload, Mapping) else int(init_payload.seed)
                state_packet = solver._reset_env(worker_env, seed=reset_seed)
                root_snapshot = solver._capture_local_snapshot(worker_env)
                if root_snapshot is None:
                    raise RuntimeError("Process backend requires environment snapshot support.")
                root_handle = allocate(root_snapshot)
                response_queue.put(
                    (
                        "reset",
                        ProcessResetResponse(
                            worker_id=int(worker_id),
                            handle=int(root_handle),
                            state_key=state_packet.state_key,
                            state_obj=state_packet.state_obj,
                            state_archive_input=state_packet.state_archive_input,
                        ),
                    )
                )
                continue

            if str(command) == "expand":
                requests = tuple(payload or ())
                results: List[WorkerNodeResult] = []
                for request in requests:
                    if not isinstance(request, ProcessExpandRequest):
                        raise RuntimeError("Invalid process expand request payload.")
                    snapshot = snapshot_store.get(int(request.handle))
                    if snapshot is None:
                        raise RuntimeError(
                            f"Process worker {int(worker_id)} is missing snapshot handle {int(request.handle)}."
                        )
                    base_result = solver._expand_node(
                        worker_env,
                        WorkerNodeRequest(
                            node_index=int(request.node_index),
                            state_key=request.state_key,
                            action_path=None,
                            snapshot=snapshot,
                        ),
                    )
                    converted: List[WorkerTransition] = []
                    for produced in base_result.transitions:
                        child_handle = -1
                        if produced.next_state_snapshot is not None:
                            child_handle = allocate(produced.next_state_snapshot)
                        converted.append(
                            WorkerTransition(
                                action_id=int(produced.action_id),
                                action_name=str(produced.action_name),
                                next_state_key=produced.next_state_key,
                                next_state_obj=produced.next_state_obj,
                                next_state_archive_input=produced.next_state_archive_input,
                                reward=float(produced.reward),
                                terminated=bool(produced.terminated),
                                truncated=bool(produced.truncated),
                                done=bool(produced.done),
                                is_win=bool(produced.is_win),
                                next_state_snapshot=None,
                                next_state_handle=int(child_handle),
                            )
                        )
                    results.append(
                        WorkerNodeResult(
                            node_index=int(base_result.node_index),
                            transitions=converted,
                            replay_steps=int(base_result.replay_steps),
                            snapshot_restores=int(base_result.snapshot_restores),
                        )
                    )
                response_queue.put(("expand", results))
                continue

            if str(command) == "materialize_paths":
                requests = tuple(payload or ())
                results: List[ProcessMaterializeResult] = []
                for request in requests:
                    if not isinstance(request, ProcessMaterializeRequest):
                        raise RuntimeError("Invalid process materialize request payload.")
                    state_packet = solver._reset_env(worker_env, seed=int(init_payload.seed))
                    for action in tuple(int(item) for item in request.action_path):
                        state_packet, _reward, terminated, truncated, _info = solver._step_env(worker_env, action)
                        if terminated or truncated:
                            break
                    local_snapshot = solver._capture_local_snapshot(worker_env)
                    if local_snapshot is None:
                        raise RuntimeError("Process backend requires environment snapshot support.")
                    results.append(
                        ProcessMaterializeResult(
                            node_index=int(request.node_index),
                            handle=int(allocate(local_snapshot)),
                            state_key=state_packet.state_key,
                        )
                    )
                response_queue.put(("materialize_paths", results))
                continue

            if str(command) == "export":
                handles = tuple(int(handle) for handle in (payload or ()))
                snapshots: List[Any] = []
                for handle in handles:
                    snapshot = snapshot_store.get(int(handle))
                    if snapshot is None:
                        raise RuntimeError(
                            f"Process worker {int(worker_id)} cannot export missing handle {int(handle)}."
                        )
                    if isinstance(snapshot, Mapping) and str(snapshot.get("snapshot_format", "")).strip() == "serialized_v1":
                        snapshots.append(snapshot)
                        continue
                    if not solver._restore_snapshot(worker_env, snapshot):
                        raise RuntimeError(
                            f"Process worker {int(worker_id)} could not restore handle {int(handle)} for export."
                        )
                    transfer_snapshot = solver._capture_transfer_snapshot(worker_env)
                    if transfer_snapshot is None:
                        raise RuntimeError(
                            f"Process worker {int(worker_id)} could not serialize handle {int(handle)} for export."
                        )
                    snapshots.append(transfer_snapshot)
                response_queue.put(("export", snapshots))
                continue

            if str(command) == "import":
                snapshots = list(payload or [])
                handles: List[int] = []
                for snapshot in snapshots:
                    handles.append(int(allocate(snapshot)))
                response_queue.put(("import", handles))
                continue

            if str(command) == "release":
                for handle in tuple(int(item) for item in (payload or ())):
                    snapshot_store.pop(int(handle), None)
                continue

            raise RuntimeError(f"Unsupported process worker command: {command}")
    finally:
        close_fn = getattr(worker_env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _materialize_canonical_transitions(
    transition_records: Sequence[TransitionRecord],
    state_objects: Mapping[Any, Mapping[str, Any]],
) -> List[Transition]:
    rows: List[Transition] = []
    for transition in transition_records:
        state_obj = state_objects.get(transition.state_key)
        next_state_obj = state_objects.get(transition.next_state_key)
        if not isinstance(state_obj, Mapping) or not isinstance(next_state_obj, Mapping):
            continue
        rows.append(
            Transition(
                state=canonicalize_state_json(dict(state_obj)),
                action=str(transition.action),
                next_state=canonicalize_state_json(dict(next_state_obj)),
                reward=float(transition.reward),
                done=bool(transition.done),
            )
        )
    return rows


def materialize_canonical_transitions(
    transition_records: Sequence[TransitionRecord],
    state_objects: Mapping[Any, Mapping[str, Any]],
) -> List[Transition]:
    return _materialize_canonical_transitions(
        transition_records=transition_records,
        state_objects=state_objects,
    )


@dataclass
class SolverStats:
    expanded_states: int = 0
    generated_states: int = 0
    duplicate_states: int = 0
    terminal_pruned: int = 0
    depth_pruned: int = 0
    elapsed_seconds: float = 0.0
    max_depth_reached: int = 0
    transition_duplicates: int = 0
    win_transitions: int = 0
    truncated_transitions: int = 0
    replay_steps: int = 0
    snapshot_restores: int = 0
    layers_completed: int = 0
    worker_count: int = 1


@dataclass
class SolverResult:
    solved: bool
    action_path: Tuple[int, ...]
    action_names: Tuple[str, ...]
    search_mode: str
    stats: SolverStats
    visited_states: int
    frontier_remaining: int
    goal_state: Optional[str] = None
    goal_rules: Tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    transition_records: List[TransitionRecord] = field(default_factory=list)
    state_archive_inputs: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    state_objects: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    transition_count: int = 0
    exhausted: bool = False

    def to_dict(self, *, include_transitions: bool = False) -> Dict[str, Any]:
        payload = {
            "solved": bool(self.solved),
            "action_path": list(self.action_path),
            "action_names": list(self.action_names),
            "action_sequence": _format_action_path(self.action_names),
            "search_mode": str(self.search_mode),
            "visited_states": int(self.visited_states),
            "frontier_remaining": int(self.frontier_remaining),
            "goal_state": self.goal_state,
            "goal_rules": list(self.goal_rules),
            "transition_count": int(self.transition_count),
            "exhausted": bool(self.exhausted),
            "stats": {
                "expanded_states": int(self.stats.expanded_states),
                "generated_states": int(self.stats.generated_states),
                "duplicate_states": int(self.stats.duplicate_states),
                "terminal_pruned": int(self.stats.terminal_pruned),
                "depth_pruned": int(self.stats.depth_pruned),
                "elapsed_seconds": float(self.stats.elapsed_seconds),
                "elapsed_human": _format_duration(self.stats.elapsed_seconds),
                "max_depth_reached": int(self.stats.max_depth_reached),
                "transition_duplicates": int(self.stats.transition_duplicates),
                "win_transitions": int(self.stats.win_transitions),
                "truncated_transitions": int(self.stats.truncated_transitions),
                "replay_steps": int(self.stats.replay_steps),
                "snapshot_restores": int(self.stats.snapshot_restores),
                "layers_completed": int(self.stats.layers_completed),
                "worker_count": int(self.stats.worker_count),
            },
            "diagnostics": dict(self.diagnostics),
        }
        if include_transitions:
            payload["canonical_transitions"] = [
                transition.to_dict()
                for transition in materialize_canonical_transitions(
                    self.transition_records,
                    self.state_objects,
                )
            ]
        return payload


class BabaSolver:
    """Parallel BFS collector for canonical transitions on a single seeded map."""

    def __init__(
        self,
        env: Any | None = None,
        *,
        env_factory: Optional[Callable[[], Any]] = None,
        seed: int = 42,
        search_mode: str = "bfs",
        max_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_states: Optional[int] = None,
        max_transitions: Optional[int] = None,
        action_order: Optional[Sequence[int]] = None,
        workers: Optional[int] = None,
        parallel_backend: str = "thread",
        chunk_size: int = 32,
        macro_action_repeat: int = 1,
        return_child_snapshots: bool = True,
        progress_label: Optional[str] = None,
        show_progress: Optional[bool] = None,
        progress_leave: bool = True,
    ) -> None:
        if env is None and env_factory is None:
            raise ValueError("Either env or env_factory must be provided.")

        self.env_factory = env_factory
        self._owns_primary_env = False
        if env is None:
            env = env_factory()
            self._owns_primary_env = True
        self.env = env

        normalized_mode = str(search_mode).strip().lower() or "bfs"
        if normalized_mode not in {"bfs", "astar"}:
            raise ValueError("search_mode must be one of: bfs, astar")
        self.search_mode_requested = normalized_mode
        self.search_mode = "bfs"
        self.seed = int(seed)
        self.max_depth = None if max_depth is None else max(0, int(max_depth))
        self.max_nodes = None if max_nodes is None else max(0, int(max_nodes))
        self.max_states = None if max_states is None else max(1, int(max_states))
        self.max_transitions = None if max_transitions is None else max(0, int(max_transitions))
        self.chunk_size = max(1, int(chunk_size))
        self.macro_action_repeat = max(1, int(macro_action_repeat))
        self.return_child_snapshots = bool(return_child_snapshots)
        self.progress_label = str(progress_label).strip() if isinstance(progress_label, str) and progress_label.strip() else "env"
        if show_progress is None:
            self.show_progress = bool(getattr(sys.stderr, "isatty", lambda: False)())
        else:
            self.show_progress = bool(show_progress)
        self.progress_leave = bool(progress_leave)
        self.num_actions = max(1, _infer_num_actions(self.env))
        self.action_names = tuple(_safe_get_action_name(self.env, action) for action in range(self.num_actions))
        self.action_order = self._resolve_action_order(action_order)
        normalized_backend = str(parallel_backend).strip().lower() or "thread"
        if normalized_backend not in {"sequential", "thread", "process"}:
            raise ValueError("parallel_backend must be one of: sequential, thread, process")
        if workers is None:
            resolved_workers = min(8, os.cpu_count() or 1) if env_factory is not None else 1
        else:
            resolved_workers = max(1, int(workers))
        if normalized_backend == "sequential":
            resolved_workers = 1
        if env_factory is None:
            if normalized_backend == "process" and resolved_workers > 1:
                raise ValueError("parallel_backend='process' requires env_factory so workers can recreate the environment.")
            resolved_workers = 1
        self.workers = max(1, int(resolved_workers))
        if self.workers <= 1:
            self.parallel_backend = "sequential"
        elif normalized_backend == "process":
            if not self._is_picklable_for_process(env_factory):
                raise ValueError(
                    "parallel_backend='process' requires a picklable env_factory. "
                    "Use a top-level function or functools.partial instead of a lambda or local closure."
                )
            self.parallel_backend = "process"
        else:
            self.parallel_backend = "thread"
        self.parallel_backend_requested = normalized_backend
        self._transport_snapshots_across_processes = False
        self._process_seed_depth = 2 if self.parallel_backend == "process" and self.workers > 1 else 0

        self._analysis_cache: Dict[Any, Dict[str, Any]] = {}
        self._heuristic_cache: Dict[Any, int] = {}
        self._wrapper_fast_path = self._detect_wrapper_fast_path(self.env)
        self._thread_local = threading.local()
        self._worker_envs: List[Any] = []
        self._worker_env_lock = threading.Lock()
        self._executor: Optional[Any] = None
        self._process_pool: Optional[_ProcessActorPool] = None

    def __enter__(self) -> "BabaSolver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        seen_ids: set[int] = set()
        envs_to_close: List[Any] = []
        if self._owns_primary_env and self.env is not None:
            envs_to_close.append(self.env)
        for worker_env in self._worker_envs:
            if worker_env is None or worker_env is self.env:
                continue
            envs_to_close.append(worker_env)
        for item in envs_to_close:
            item_id = id(item)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            close_fn = getattr(item, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._process_pool is not None:
            self._process_pool.close()
            self._process_pool = None

    def _resolve_action_order(self, action_order: Optional[Sequence[int]]) -> Tuple[int, ...]:
        if action_order is not None:
            resolved = []
            for raw_action in action_order:
                action = int(raw_action)
                if action < 0 or action >= self.num_actions:
                    raise ValueError(f"Invalid action in action_order: {raw_action}")
                if action not in resolved:
                    resolved.append(action)
            if not resolved:
                raise ValueError("action_order must not be empty")
            return tuple(resolved)

        preferred = [action for action in DEFAULT_ACTION_ORDER if action < self.num_actions]
        preferred.extend(action for action in range(self.num_actions) if action not in preferred)
        return tuple(preferred)

    def _is_picklable_for_process(self, value: Any) -> bool:
        if value is None:
            return False
        try:
            pickle.dumps(value)
        except Exception:
            return False
        return True

    def _analysis_for(self, state_key: Any, state_obj: Mapping[str, Any]) -> Dict[str, Any]:
        cached = self._analysis_cache.get(state_key)
        if cached is not None:
            return cached
        analysis = analyze_state(state_obj)
        self._analysis_cache[state_key] = analysis
        return analysis

    def _heuristic_for(self, state_key: Any, state_obj: Mapping[str, Any]) -> int:
        cached = self._heuristic_cache.get(state_key)
        if cached is not None:
            return cached
        heuristic = estimate_analysis_heuristic(self._analysis_for(state_key, state_obj))
        self._heuristic_cache[state_key] = heuristic
        return heuristic

    def _is_goal_state(
        self,
        state_key: Any,
        state_obj: Mapping[str, Any],
        info: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if isinstance(info, Mapping) and bool(info.get("is_win", False)):
            return True
        return bool(self._analysis_for(state_key, state_obj).get("goal_overlap"))

    def _action_candidates(self, _state_key: Any, _state_obj: Mapping[str, Any]) -> Tuple[int, ...]:
        return tuple(int(action) for action in self.action_order)

    def solve(self) -> SolverResult:
        return self.collect()

    def _create_progress_bar(self):
        if not self.show_progress or _tqdm is None:
            return None
        total = None
        if self.max_transitions is not None and int(self.max_transitions) >= 0:
            total = int(self.max_transitions)
        progress_bar = _tqdm(
            total=total,
            desc=f"{self.progress_label} canonical",
            unit="tr",
            leave=self.progress_leave,
            dynamic_ncols=True,
            file=sys.stdout,
            miniters=1,
            mininterval=0.1,
        )
        progress_bar.refresh()
        return progress_bar

    def collect(self) -> SolverResult:
        start_time = time.perf_counter()
        stats = SolverStats(worker_count=int(self.workers))
        transition_records: List[TransitionRecord] = []
        transition_keys: set[Tuple[Any, int, Any]] = set()
        visited_keys: set[Any] = set()
        nodes: List[SearchNode] = []
        state_archive_inputs: Dict[Any, Dict[str, Any]] = {}
        state_objects: Dict[Any, Dict[str, Any]] = {}
        frontier: List[int] = []
        stop_reason = "frontier_exhausted"
        first_win_state_key: Optional[Any] = None
        progress_bar = self._create_progress_bar()

        def ensure_state_obj(
            state_key: Any,
            *,
            candidate_state_obj: Optional[Dict[str, Any]] = None,
            candidate_archive_input: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            existing = state_objects.get(state_key)
            if existing is not None:
                return existing
            if candidate_state_obj is not None:
                state_objects[state_key] = candidate_state_obj
                return candidate_state_obj
            archive_input = candidate_archive_input
            if archive_input is None:
                archive_input = state_archive_inputs.get(state_key)
            if archive_input is None:
                materialized = archive_state_input_to_state_obj(
                    {
                        "source": "missing_archive_input",
                        "grid_size": [0, 0],
                        "objects": [],
                        "event": "",
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                    },
                    normalize_input=False,
                )
            else:
                materialized = archive_state_input_to_state_obj(
                    archive_input,
                    normalize_input=False,
                )
            state_objects[state_key] = materialized
            return materialized

        try:
            use_process_actors = bool(self.parallel_backend == "process" and self.workers > 1)
            root_packet = self._reset_env(self.env, seed=self.seed)
            root_snapshot = self._capture_snapshot(self.env)
            state_archive_inputs[root_packet.state_key] = root_packet.state_archive_input
            ensure_state_obj(
                root_packet.state_key,
                candidate_state_obj=root_packet.state_obj,
                candidate_archive_input=root_packet.state_archive_input,
            )
            nodes.append(
                SearchNode(
                    state_key=root_packet.state_key,
                    parent_index=-1,
                    action_from_parent=-1,
                    depth=0,
                    snapshot=root_snapshot,
                )
            )
            visited_keys.add(root_packet.state_key)
            frontier = [0]

            if self.max_transitions == 0:
                stop_reason = "max_transitions"

            while frontier and stop_reason == "frontier_exhausted":
                remaining_expand_budget = (
                    None
                    if self.max_nodes is None
                    else max(0, int(self.max_nodes) - int(stats.expanded_states))
                )
                if remaining_expand_budget is not None and remaining_expand_budget <= 0:
                    stop_reason = "max_nodes"
                    break

                current_frontier = frontier
                frontier = []
                expandable: List[int] = []
                deferred: List[int] = []
                for node_index in current_frontier:
                    node = nodes[node_index]
                    stats.max_depth_reached = max(stats.max_depth_reached, int(node.depth))
                    if self.max_depth is not None and node.depth >= int(self.max_depth):
                        stats.depth_pruned += 1
                        continue
                    if remaining_expand_budget is None or len(expandable) < int(remaining_expand_budget):
                        expandable.append(node_index)
                    else:
                        deferred.append(node_index)

                if not expandable:
                    frontier = deferred
                    if deferred:
                        stop_reason = "max_nodes"
                    continue

                process_layer = bool(
                    use_process_actors
                    and expandable
                    and all(int(nodes[int(node_index)].depth) >= int(self._process_seed_depth) for node_index in expandable)
                )

                if process_layer:
                    self._materialize_process_frontier(expandable, nodes)
                    results_by_node = self._expand_process_layer(expandable, nodes)
                    layer_children = []
                    stop_mid_layer = False
                    release_by_worker: Dict[int, List[int]] = defaultdict(list)

                    for node_index in expandable:
                        result = results_by_node.get(int(node_index))
                        if result is None:
                            raise RuntimeError(
                                f"Process worker results are missing node {int(node_index)}."
                            )
                        stats.expanded_states += 1
                        stats.replay_steps += int(result.replay_steps)
                        stats.snapshot_restores += int(result.snapshot_restores)
                        node = nodes[result.node_index]
                        source_worker = int(node.process_owner)
                        parent_handle = int(node.process_handle)
                        if source_worker >= 0 and parent_handle >= 0:
                            release_by_worker[source_worker].append(parent_handle)
                        node.process_handle = -1
                        added_transition_count = 0

                        for produced in result.transitions:
                            child_handle = int(produced.next_state_handle)
                            if stop_mid_layer or (
                                self.max_transitions is not None
                                and len(transition_records) >= int(self.max_transitions)
                            ):
                                stop_reason = "max_transitions"
                                stop_mid_layer = True
                                if child_handle >= 0:
                                    release_by_worker[source_worker].append(child_handle)
                                continue

                            stats.generated_states += 1
                            state_archive_inputs.setdefault(
                                produced.next_state_key,
                                produced.next_state_archive_input,
                            )
                            ensure_state_obj(
                                produced.next_state_key,
                                candidate_state_obj=produced.next_state_obj,
                                candidate_archive_input=produced.next_state_archive_input,
                            )
                            transition_key = (node.state_key, int(produced.action_id), produced.next_state_key)
                            if transition_key in transition_keys:
                                stats.transition_duplicates += 1
                            else:
                                transition_keys.add(transition_key)
                                transition_records.append(
                                    TransitionRecord(
                                        state_key=node.state_key,
                                        action=str(produced.action_name),
                                        next_state_key=produced.next_state_key,
                                        reward=float(produced.reward),
                                        done=bool(produced.done),
                                    )
                                )
                                added_transition_count += 1

                            if produced.is_win:
                                stats.win_transitions += 1
                                if first_win_state_key is None:
                                    first_win_state_key = produced.next_state_key
                            if produced.truncated:
                                stats.truncated_transitions += 1
                            if produced.done:
                                stats.terminal_pruned += 1
                                if child_handle >= 0:
                                    release_by_worker[source_worker].append(child_handle)
                                continue
                            if produced.next_state_key in visited_keys:
                                stats.duplicate_states += 1
                                if child_handle >= 0:
                                    release_by_worker[source_worker].append(child_handle)
                                continue
                            if self.max_states is not None and (
                                len(nodes) >= int(self.max_states)
                            ):
                                stop_reason = "max_states"
                                if child_handle >= 0:
                                    release_by_worker[source_worker].append(child_handle)
                                continue
                            if child_handle < 0:
                                raise RuntimeError(
                                    "Process backend requires child snapshots for non-terminal states."
                                )

                            visited_keys.add(produced.next_state_key)
                            nodes.append(
                                SearchNode(
                                    state_key=produced.next_state_key,
                                    parent_index=result.node_index,
                                    action_from_parent=int(produced.action_id),
                                    depth=int(node.depth + 1),
                                    snapshot=None,
                                    process_owner=int(source_worker),
                                    process_handle=int(child_handle),
                                )
                            )
                            layer_children.append(len(nodes) - 1)

                        if progress_bar is not None and added_transition_count > 0:
                            progress_bar.update(int(added_transition_count))
                            quota_text = (
                                str(int(self.max_transitions))
                                if self.max_transitions is not None
                                else "?"
                            )
                            progress_bar.set_postfix_str(
                                f"canonical={len(transition_records)}/{quota_text}",
                                refresh=False,
                            )

                    self._release_process_handles(release_by_worker)
                    stats.layers_completed += 1
                    frontier = deferred + layer_children
                    if stop_mid_layer:
                        break
                else:
                    requests = self._build_requests(expandable, nodes)
                    chunk_results = self._expand_layer(requests)
                    layer_children = []
                    stop_mid_layer = False
                    merged_node_count = 0

                    for node_results in chunk_results:
                        for result in node_results:
                            merged_node_count += 1
                            stats.expanded_states += 1
                            stats.replay_steps += int(result.replay_steps)
                            stats.snapshot_restores += int(result.snapshot_restores)
                            node = nodes[result.node_index]
                            node.snapshot = None
                            added_transition_count = 0
                            for produced in result.transitions:
                                if self.max_transitions is not None and len(transition_records) >= int(self.max_transitions):
                                    stop_reason = "max_transitions"
                                    stop_mid_layer = True
                                    break

                                stats.generated_states += 1
                                state_archive_inputs.setdefault(
                                    produced.next_state_key,
                                    produced.next_state_archive_input,
                                )
                                ensure_state_obj(
                                    produced.next_state_key,
                                    candidate_state_obj=produced.next_state_obj,
                                    candidate_archive_input=produced.next_state_archive_input,
                                )
                                transition_key = (node.state_key, int(produced.action_id), produced.next_state_key)
                                if transition_key in transition_keys:
                                    stats.transition_duplicates += 1
                                else:
                                    transition_keys.add(transition_key)
                                    transition_records.append(
                                        TransitionRecord(
                                            state_key=node.state_key,
                                            action=str(produced.action_name),
                                            next_state_key=produced.next_state_key,
                                            reward=float(produced.reward),
                                            done=bool(produced.done),
                                        )
                                    )
                                    added_transition_count += 1

                                if produced.is_win:
                                    stats.win_transitions += 1
                                    if first_win_state_key is None:
                                        first_win_state_key = produced.next_state_key
                                if produced.truncated:
                                    stats.truncated_transitions += 1
                                if produced.done:
                                    stats.terminal_pruned += 1
                                    continue
                                if produced.next_state_key in visited_keys:
                                    stats.duplicate_states += 1
                                    continue
                                if self.max_states is not None and len(nodes) >= int(self.max_states):
                                    stop_reason = "max_states"
                                    continue

                                nodes.append(
                                    SearchNode(
                                        state_key=produced.next_state_key,
                                        parent_index=result.node_index,
                                        action_from_parent=int(produced.action_id),
                                        depth=int(node.depth + 1),
                                        snapshot=produced.next_state_snapshot,
                                    )
                                )
                                visited_keys.add(produced.next_state_key)
                                layer_children.append(len(nodes) - 1)

                            if progress_bar is not None and added_transition_count > 0:
                                progress_bar.update(int(added_transition_count))
                                quota_text = (
                                    str(int(self.max_transitions))
                                    if self.max_transitions is not None
                                    else "?"
                                )
                                progress_bar.set_postfix_str(
                                    f"canonical={len(transition_records)}/{quota_text}",
                                    refresh=False,
                                )
                            if stop_mid_layer:
                                break
                        if stop_mid_layer:
                            break

                    stats.layers_completed += 1
                    if stop_mid_layer:
                        unmerged_expandable = expandable[merged_node_count:]
                        frontier = unmerged_expandable + deferred + layer_children
                        break
                    frontier = deferred + layer_children

            stats.elapsed_seconds = time.perf_counter() - start_time
            exhausted = bool(stop_reason == "frontier_exhausted" and not frontier)
            goal_rules: Tuple[str, ...] = ()
            if first_win_state_key is not None:
                goal_rules = tuple(
                    analyze_state(ensure_state_obj(first_win_state_key)).get("active_rules", ())
                )
            return SolverResult(
                solved=first_win_state_key is not None,
                action_path=(),
                action_names=(),
                search_mode=self.search_mode,
                stats=stats,
                visited_states=len(nodes),
                frontier_remaining=len(frontier),
                goal_state=None,
                goal_rules=goal_rules,
                diagnostics={
                    "seed": int(self.seed),
                    "requested_search_mode": str(self.search_mode_requested),
                    "effective_search_mode": str(self.search_mode),
                    "action_order": list(self.action_order),
                    "max_depth": (int(self.max_depth) if self.max_depth is not None else None),
                    "max_nodes": (int(self.max_nodes) if self.max_nodes is not None else None),
                    "max_states": (int(self.max_states) if self.max_states is not None else None),
                    "max_transitions": (int(self.max_transitions) if self.max_transitions is not None else None),
                    "max_transitions_per_scenario": (
                        int(self.max_transitions) if self.max_transitions is not None else None
                    ),
                    "workers": int(self.workers),
                    "chunk_size": int(self.chunk_size),
                    "parallel_backend_requested": str(self.parallel_backend_requested),
                    "parallel_backend": str(self.parallel_backend),
                    "parallel_enabled": bool(self.parallel_backend != "sequential"),
                    "persistent_worker_pool": bool(self.parallel_backend in {"thread", "process"}),
                    "persistent_thread_pool": bool(self.parallel_backend == "thread"),
                    "persistent_process_pool": bool(self.parallel_backend == "process"),
                    "frontier_snapshot_reuse": bool(self.return_child_snapshots),
                    "process_seed_depth": int(self._process_seed_depth),
                    "process_sticky_subtrees": bool(self.parallel_backend == "process" and self.workers > 1),
                    "process_frontier_materialization": (
                        "replay_from_root_prefix"
                        if self.parallel_backend == "process" and self.workers > 1
                        else "not_used"
                    ),
                    "state_representation": "raw_frozen_state_obj",
                    "lazy_state_obj_materialization": True,
                    "eager_text_transition_materialization": False,
                    "eager_goal_state_materialization": False,
                    "stop_reason": str(stop_reason),
                    "wrapper_fast_path": bool(self._wrapper_fast_path),
                },
                transition_records=transition_records,
                state_archive_inputs=state_archive_inputs,
                state_objects=state_objects,
                transition_count=len(transition_records),
                exhausted=exhausted,
            )
        finally:
            stats.elapsed_seconds = time.perf_counter() - start_time
            if progress_bar is not None:
                if progress_bar.total is None:
                    progress_bar.total = int(len(transition_records))
                progress_bar.n = int(len(transition_records))
                progress_bar.set_postfix_str(
                    f"canonical={len(transition_records)}/{int(progress_bar.total)} stop={stop_reason}",
                    refresh=False,
                )
                progress_bar.refresh()
                progress_bar.close()

    def _resolve_action_paths(
        self,
        frontier: Sequence[int],
        nodes: Sequence[SearchNode],
    ) -> Dict[int, Tuple[int, ...]]:
        path_cache: Dict[int, Tuple[int, ...]] = {}

        def resolve_path(node_index: int) -> Tuple[int, ...]:
            cached = path_cache.get(node_index)
            if cached is not None:
                return cached
            node = nodes[node_index]
            if node.parent_index < 0:
                path = ()
            else:
                path = resolve_path(node.parent_index) + (int(node.action_from_parent),)
            path_cache[node_index] = path
            return path

        return {int(node_index): resolve_path(int(node_index)) for node_index in frontier}

    def _build_requests(
        self,
        frontier: Sequence[int],
        nodes: Sequence[SearchNode],
    ) -> List[WorkerNodeRequest]:
        needs_paths = any(nodes[node_index].snapshot is None for node_index in frontier)
        resolved_paths = self._resolve_action_paths(frontier, nodes) if needs_paths else {}
        requests: List[WorkerNodeRequest] = []
        for node_index in frontier:
            node = nodes[node_index]
            requests.append(
                WorkerNodeRequest(
                    node_index=int(node_index),
                    state_key=node.state_key,
                    action_path=(resolved_paths.get(int(node_index)) if needs_paths and node.snapshot is None else None),
                    snapshot=node.snapshot,
                )
            )
        return requests

    def _ensure_process_pool(self) -> _ProcessActorPool:
        if self._process_pool is not None:
            return self._process_pool
        if self.env_factory is None:
            raise RuntimeError("parallel_backend='process' requires env_factory.")
        init_payload = ProcessWorkerInit(
            env_factory=self.env_factory,
            seed=int(self.seed),
            action_order=tuple(int(action) for action in self.action_order),
        )
        self._process_pool = _ProcessActorPool(
            init_payload=init_payload,
            workers=int(self.workers),
        )
        return self._process_pool

    def _select_process_owner(self, state_key: Any) -> int:
        if self.workers <= 1:
            return 0
        return int((hash(state_key) & 0x7FFFFFFF) % int(self.workers))

    def _expand_process_layer(
        self,
        frontier: Sequence[int],
        nodes: Sequence[SearchNode],
    ) -> Dict[int, WorkerNodeResult]:
        grouped_requests: Dict[int, List[ProcessExpandRequest]] = defaultdict(list)
        for node_index in frontier:
            node = nodes[int(node_index)]
            if int(node.process_owner) < 0 or int(node.process_handle) < 0:
                raise RuntimeError(
                    f"Process frontier node {int(node_index)} is missing a worker handle."
                )
            grouped_requests[int(node.process_owner)].append(
                ProcessExpandRequest(
                    node_index=int(node_index),
                    state_key=node.state_key,
                    handle=int(node.process_handle),
                )
            )
        results_by_node: Dict[int, WorkerNodeResult] = {}
        for worker_results in self._ensure_process_pool().expand(grouped_requests).values():
            for result in worker_results:
                results_by_node[int(result.node_index)] = result
        return results_by_node

    def _materialize_process_frontier(
        self,
        frontier: Sequence[int],
        nodes: Sequence[SearchNode],
    ) -> None:
        unresolved = [
            int(node_index)
            for node_index in frontier
            if int(nodes[int(node_index)].process_handle) < 0
        ]
        if not unresolved:
            return
        resolved_paths = self._resolve_action_paths(unresolved, nodes)
        grouped_requests: Dict[int, List[ProcessMaterializeRequest]] = defaultdict(list)
        for node_index in unresolved:
            node = nodes[int(node_index)]
            target_worker = self._select_process_owner(node.state_key)
            grouped_requests[int(target_worker)].append(
                ProcessMaterializeRequest(
                    node_index=int(node_index),
                    state_key=node.state_key,
                    action_path=tuple(int(action) for action in resolved_paths[int(node_index)]),
                )
            )
        materialized_by_worker = self._ensure_process_pool().materialize_paths(grouped_requests)
        materialized_by_node: Dict[int, ProcessMaterializeResult] = {}
        for worker_results in materialized_by_worker.values():
            for result in worker_results:
                materialized_by_node[int(result.node_index)] = result
        for node_index in unresolved:
            result = materialized_by_node.get(int(node_index))
            if result is None:
                raise RuntimeError(f"Process worker failed to materialize node {int(node_index)}.")
            if result.state_key != nodes[int(node_index)].state_key:
                raise RuntimeError(
                    "Materialized process node does not match the stored BFS key. "
                    f"expected={nodes[int(node_index)].state_key} got={result.state_key}"
                )
            nodes[int(node_index)].process_owner = self._select_process_owner(nodes[int(node_index)].state_key)
            nodes[int(node_index)].process_handle = int(result.handle)
            nodes[int(node_index)].snapshot = None

    def _migrate_process_nodes(
        self,
        pending_migrations: Sequence[PendingProcessMigration],
    ) -> Dict[int, int]:
        if not pending_migrations:
            return {}
        pool = self._ensure_process_pool()
        exports_by_worker: Dict[int, List[int]] = defaultdict(list)
        for migration in pending_migrations:
            exports_by_worker[int(migration.source_worker)].append(int(migration.source_handle))
        exported_snapshots = pool.export_handles(exports_by_worker)

        exported_iterators: Dict[int, Iterable[Any]] = {
            int(worker_id): iter(payloads)
            for worker_id, payloads in exported_snapshots.items()
        }
        imports_by_worker: Dict[int, List[Any]] = defaultdict(list)
        migration_target_order: Dict[int, List[int]] = defaultdict(list)
        for migration_index, migration in enumerate(pending_migrations):
            try:
                snapshot_payload = next(exported_iterators[int(migration.source_worker)])
            except StopIteration as exc:
                raise RuntimeError("Process snapshot export ordering mismatch.") from exc
            imports_by_worker[int(migration.target_worker)].append(snapshot_payload)
            migration_target_order[int(migration.target_worker)].append(int(migration_index))

        imported_handles = pool.import_snapshots(imports_by_worker)
        migrated_handles: Dict[int, int] = {}
        for target_worker, migration_indices in migration_target_order.items():
            handles = imported_handles.get(int(target_worker), [])
            if len(handles) != len(migration_indices):
                raise RuntimeError("Process snapshot import ordering mismatch.")
            for migration_index, handle in zip(migration_indices, handles):
                migrated_handles[int(migration_index)] = int(handle)
        return migrated_handles

    def _release_process_handles(self, release_by_worker: Mapping[int, Sequence[int]]) -> None:
        if not release_by_worker:
            return
        self._ensure_process_pool().release_handles(release_by_worker)

    def _expand_layer(self, requests: Sequence[WorkerNodeRequest]) -> List[List[WorkerNodeResult]]:
        if not requests:
            return []
        if self.parallel_backend == "sequential" or self.workers <= 1 or len(requests) <= 1:
            return [self._expand_chunk(tuple(requests))]

        chunk_size = max(1, int(self.chunk_size))
        chunks = [
            tuple(requests[index:index + chunk_size])
            for index in range(0, len(requests), chunk_size)
        ]
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.workers)
        return list(self._executor.map(self._expand_chunk, chunks))

    def _get_worker_env(self) -> Any:
        current_env = getattr(self._thread_local, "env", None)
        if current_env is not None:
            return current_env
        if self.env_factory is None:
            self._thread_local.env = self.env
            return self.env
        worker_env = self.env_factory()
        self._thread_local.env = worker_env
        with self._worker_env_lock:
            self._worker_envs.append(worker_env)
        return worker_env

    def _make_state_packet(
        self,
        *,
        state_obj: Mapping[str, Any],
        state_archive_input: Mapping[str, Any],
        normalize_state_obj: bool = True,
        normalize_state_archive_input: bool = True,
    ) -> StatePacket:
        normalized_state_obj = (
            _state_obj_from_state_value(state_obj)
            if normalize_state_obj
            else dict(state_obj)
        )
        normalized_archive_input = (
            _normalize_state_archive_input(
                state_archive_input,
                default_source="state_packet",
            )
            if normalize_state_archive_input
            else dict(state_archive_input)
        )
        return StatePacket(
            state_key=_state_key(normalized_state_obj),
            state_obj=normalized_state_obj,
            state_archive_input=normalized_archive_input,
        )

    def _state_packet_from_state(self, state: Any) -> StatePacket:
        state_obj = _state_obj_from_state_value(state)
        return self._make_state_packet(
            state_obj=state_obj,
            state_archive_input=_state_archive_input_from_state_obj(state_obj),
            normalize_state_obj=False,
            normalize_state_archive_input=False,
        )

    def _state_packet_from_wrapper_cache(
        self,
        env: Any,
        *,
        terminated: bool,
        materialize_state_obj: bool = False,
    ) -> StatePacket:
        state_archive_input = self._capture_state_archive_input(env)
        if materialize_state_obj:
            return self._make_state_packet(
                state_obj=self._state_obj_from_wrapper_cache(env, terminated=terminated),
                state_archive_input=state_archive_input,
                normalize_state_obj=False,
                normalize_state_archive_input=False,
            )
        return StatePacket(
            state_key=_state_key_from_archive_input(state_archive_input),
            state_obj=None,
            state_archive_input=state_archive_input,
        )

    def _reset_env(self, env: Any, *, seed: int) -> StatePacket:
        if self._detect_wrapper_fast_path(env):
            reset_raw_fn = getattr(env, "reset_raw", None)
            if callable(reset_raw_fn):
                reset_raw_fn(seed=int(seed))
            else:
                env.reset(seed=int(seed))
            return self._state_packet_from_wrapper_cache(
                env,
                terminated=bool(getattr(env, "last_terminated", False)),
                materialize_state_obj=False,
            )
        state = env.reset(seed=int(seed))
        return self._state_packet_from_state(state)

    def _capture_local_snapshot(self, env: Any) -> Optional[Any]:
        snapshot_fn = getattr(env, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        try:
            return snapshot_fn()
        except Exception:
            return None

    def _capture_transfer_snapshot(self, env: Any) -> Optional[Any]:
        snapshot_serializable_fn = getattr(env, "snapshot_serializable", None)
        if callable(snapshot_serializable_fn):
            try:
                return snapshot_serializable_fn()
            except Exception:
                pass
        return self._capture_local_snapshot(env)

    def _capture_snapshot(self, env: Any) -> Optional[Any]:
        if self._transport_snapshots_across_processes:
            return self._capture_transfer_snapshot(env)
        return self._capture_local_snapshot(env)

    def _restore_snapshot(self, env: Any, snapshot: Any) -> bool:
        if isinstance(snapshot, Mapping) and str(snapshot.get("snapshot_format", "")).strip() == "serialized_v1":
            restore_serializable_fn = getattr(env, "restore_serializable", None)
            if callable(restore_serializable_fn):
                try:
                    restore_serializable_fn(snapshot, refresh_cache=False)
                    return True
                except TypeError:
                    try:
                        restore_serializable_fn(snapshot)
                        return True
                    except Exception:
                        return False
                except Exception:
                    return False
        restore_fn = getattr(env, "restore", None)
        if not callable(restore_fn):
            return False
        try:
            restore_fn(snapshot, refresh_cache=False)
            return True
        except TypeError:
            try:
                restore_fn(snapshot)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _detect_wrapper_fast_path(self, env: Any) -> bool:
        cached = getattr(env, "_baba_solver_fast_path", None)
        if isinstance(cached, bool):
            return cached
        serializer = getattr(env, "serializer", None)
        value = bool(
            serializer is not None
            and callable(getattr(env, "_update_state", None))
            and callable(getattr(env, "_infer_event", None))
            and hasattr(env, "current_objects")
            and hasattr(env, "grid_size")
            and hasattr(env, "env")
            and callable(getattr(serializer, "_normalize_objects", None))
            and callable(getattr(serializer, "_trim_grid_size", None))
        )
        try:
            setattr(env, "_baba_solver_fast_path", value)
        except Exception:
            pass
        return value

    def _state_obj_from_wrapper_cache(
        self,
        env: Any,
        *,
        terminated: bool,
    ) -> Dict[str, Any]:
        raw_width, raw_height = getattr(env, "grid_size", (0, 0))
        if not isinstance(raw_width, Integral) or not isinstance(raw_height, Integral):
            raw_width, raw_height = 0, 0
        serializer = env.serializer
        width, height = serializer._trim_grid_size(int(raw_width), int(raw_height))
        current_objects = getattr(env, "current_objects", [])
        objects = serializer._normalize_objects(
            current_objects if isinstance(current_objects, list) else [],
            raw_width=int(raw_width),
            raw_height=int(raw_height),
        )
        return {
            "grid_size": [int(width), int(height)],
            "step": {"terminated": bool(terminated)},
            "objects": objects,
        }

    def _capture_state_archive_input(
        self,
        env: Any,
    ) -> Dict[str, Any]:
        raw_grid_size = getattr(env, "grid_size", None)
        raw_objects = getattr(env, "current_objects", None)
        if (
            isinstance(raw_grid_size, (tuple, list))
            and len(raw_grid_size) == 2
            and isinstance(raw_grid_size[0], Integral)
            and isinstance(raw_grid_size[1], Integral)
            and isinstance(raw_objects, list)
        ):
            return {
                "source": "wrapper_cache",
                "grid_size": [int(raw_grid_size[0]), int(raw_grid_size[1])],
                "objects": raw_objects,
                "event": getattr(env, "last_event", ""),
                "reward": getattr(env, "last_reward", 0.0),
                "terminated": getattr(env, "last_terminated", False),
                "truncated": getattr(env, "last_truncated", False),
            }
        return {
            "source": "missing_wrapper_cache",
            "grid_size": [0, 0],
            "objects": [],
            "event": "",
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    def _step_env(
        self,
        env: Any,
        action: int,
    ) -> Tuple[StatePacket, float, bool, bool, Dict[str, Any]]:
        if self._detect_wrapper_fast_path(env):
            return self._step_env_fast(env, int(action))

        step_fn = getattr(env, "step", None)
        if not callable(step_fn):
            raise RuntimeError("Environment does not provide a usable step() method.")

        state, reward, terminated, truncated, info = step_fn(int(action))
        return (
            self._state_packet_from_state(state),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info) if isinstance(info, dict) else {},
        )

    def _step_env_fast(
        self,
        env: Any,
        action: int,
    ) -> Tuple[StatePacket, float, bool, bool, Dict[str, Any]]:
        try:
            result = env.env.step(int(action), emit_obs=False)
        except TypeError:
            result = env.env.step(int(action))
        if not isinstance(result, tuple):
            raise RuntimeError("Unexpected step return type from BABA environment.")

        if len(result) == 5:
            _obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            _obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(f"Unexpected step return length from BABA environment: {len(result)}")

        env._update_state()

        is_win = bool(getattr(env.env, "is_win", False))
        is_defeat = bool(getattr(env.env, "is_defeat", False))
        reached_limit = bool(
            done
            and not is_win
            and not is_defeat
            and isinstance(getattr(env.env, "step_count", None), Integral)
            and isinstance(getattr(env.env, "max_steps", None), Integral)
            and int(env.env.step_count) >= int(env.env.max_steps)
        )
        terminated = bool(done and not reached_limit)
        truncated = bool(done and reached_limit)

        env.last_reward = float(reward)
        env.last_terminated = terminated
        env.last_truncated = truncated
        env.last_event = env._infer_event(
            action=int(action),
            terminated=terminated,
            truncated=truncated,
            is_win=is_win,
            is_defeat=is_defeat,
        )

        info_dict: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        info_dict.setdefault("is_win", is_win)
        info_dict.setdefault("is_defeat", is_defeat)
        return (
            self._state_packet_from_wrapper_cache(
                env,
                terminated=terminated,
                materialize_state_obj=False,
            ),
            float(reward),
            bool(terminated),
            bool(truncated),
            info_dict,
        )

    def _restore_request_state(
        self,
        env: Any,
        request: WorkerNodeRequest,
    ) -> int:
        replay_steps = 0
        if request.snapshot is not None and self._restore_snapshot(env, request.snapshot):
            return replay_steps

        state_packet = self._reset_env(env, seed=self.seed)
        for action in request.action_path or ():
            state_packet, _reward, terminated, truncated, _info = self._step_env(env, int(action))
            replay_steps += 1
            if terminated or truncated:
                raise RuntimeError("Stored BFS path reached a terminal state before restoration completed.")
        if state_packet.state_key != request.state_key:
            raise RuntimeError(
                "Restored state does not match the stored BFS node. "
                f"expected={request.state_key} got={state_packet.state_key}"
            )
        return replay_steps

    def _expand_node(
        self,
        env: Any,
        request: WorkerNodeRequest,
    ) -> WorkerNodeResult:
        result = WorkerNodeResult(node_index=int(request.node_index), transitions=[])
        result.replay_steps += int(self._restore_request_state(env, request))
        snapshot = self._capture_snapshot(env)

        for action_index, action in enumerate(self.action_order):
            if action_index > 0:
                if snapshot is not None and self._restore_snapshot(env, snapshot):
                    result.snapshot_restores += 1
                else:
                    result.replay_steps += int(self._restore_request_state(env, request))

            next_state_packet, reward, terminated, truncated, info = self._step_env(env, int(action))
            next_state_snapshot = None
            if self.return_child_snapshots and not bool(terminated or truncated):
                next_state_snapshot = self._capture_snapshot(env)
            result.transitions.append(
                WorkerTransition(
                    action_id=int(action),
                    action_name=str(self.action_names[int(action)]),
                    next_state_key=next_state_packet.state_key,
                    next_state_obj=next_state_packet.state_obj,
                    next_state_archive_input=next_state_packet.state_archive_input,
                    next_state_snapshot=next_state_snapshot,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    done=bool(terminated or truncated),
                    is_win=bool(info.get("is_win", False)),
                )
            )
        return result

    def _expand_chunk(
        self,
        requests: Sequence[WorkerNodeRequest],
    ) -> List[WorkerNodeResult]:
        env = self._get_worker_env()
        return [self._expand_node(env, request) for request in requests]


def create_baba_wrapper(
    *,
    env_name: str,
    scenario_type: Optional[str],
    seed: int,
    max_steps: int,
) -> Any:
    from src.environments import BabaWrapper

    env_kwargs: Dict[str, Any] = {}
    if scenario_type is not None:
        env_kwargs["scenario_type"] = scenario_type
    return BabaWrapper(
        env_name=env_name,
        max_steps=int(max_steps),
        render_mode=None,
        state_format="json",
        seed=int(seed),
        env_kwargs=env_kwargs,
    )


def _resolve_effective_baba_scenario_type(
    *,
    env_name: str,
    scenario_type: Optional[str],
    seed: int,
    max_steps: int,
) -> Optional[str]:
    normalized = str(scenario_type).strip() if isinstance(scenario_type, str) else ""
    if normalized:
        return normalized

    probe_env = create_baba_wrapper(
        env_name=env_name,
        scenario_type=None,
        seed=int(seed),
        max_steps=int(max_steps),
    )
    try:
        reset_raw_fn = getattr(probe_env, "reset_raw", None)
        if callable(reset_raw_fn):
            reset_raw_fn(seed=int(seed))
        else:
            probe_env.reset(seed=int(seed))
        resolved = getattr(getattr(probe_env, "env", None), "scenario_type", None)
        if isinstance(resolved, str) and resolved.strip():
            return str(resolved).strip()
        return None
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _discover_env_scenarios(
    *,
    env_name: str,
    seed: int,
    max_steps: int,
) -> Tuple[str, ...]:
    probe_env = create_baba_wrapper(
        env_name=env_name,
        scenario_type=None,
        seed=int(seed),
        max_steps=int(max_steps),
    )
    try:
        inner_env = getattr(probe_env, "env", None)
        custom_catalog = getattr(inner_env, "custom_catalog", None)
        if isinstance(custom_catalog, Mapping) and custom_catalog:
            return tuple(str(name).strip() for name in custom_catalog.keys() if str(name).strip())

        raw_scenarios = getattr(inner_env, "SCENARIO_TYPES", None)
        if isinstance(raw_scenarios, (list, tuple)):
            return tuple(str(name).strip() for name in raw_scenarios if isinstance(name, str) and name.strip())
        return ()
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def solve_baba_env(
    *,
    artifact_root: Path,
    artifact_stem: str,
    env_name: str,
    scenario_type: Optional[str],
    seed: int,
    search_mode: str,
    max_depth: Optional[int],
    max_nodes: Optional[int],
    max_steps: int,
    max_states: Optional[int] = None,
    max_transitions: Optional[int] = None,
    workers: Optional[int] = None,
    parallel_backend: str = "thread",
    chunk_size: int = 32,
    show_progress: Optional[bool] = None,
) -> SolverResult:
    effective_scenario_type = _resolve_effective_baba_scenario_type(
        env_name=env_name,
        scenario_type=scenario_type,
        seed=int(seed),
        max_steps=int(max_steps),
    )

    env_factory = partial(
        create_baba_wrapper,
        env_name=env_name,
        scenario_type=effective_scenario_type,
        seed=seed,
        max_steps=max_steps,
    )
    solver = BabaSolver(
        env_factory=env_factory,
        seed=seed,
        search_mode=search_mode,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_states=max_states,
        max_transitions=max_transitions,
        workers=workers,
        parallel_backend=parallel_backend,
        chunk_size=chunk_size,
        progress_label=(effective_scenario_type or env_name),
        show_progress=show_progress,
        progress_leave=True,
    )
    try:
        result = solver.collect()
        artifact_root.mkdir(parents=True, exist_ok=True)
        state_payloads, transition_rows = _build_transition_archive_payload(
            result.transition_records,
            state_archive_inputs=result.state_archive_inputs,
        )
        states_path = _write_state_archive_npz(
            path=artifact_root / f"{artifact_stem}_states.npz",
            state_payloads=state_payloads,
        )
        transitions_path = _write_transition_rows(
            path=artifact_root / f"{artifact_stem}_transitions.jsonl",
            transition_rows=transition_rows,
        )
        result.diagnostics["artifact_root"] = str(artifact_root)
        result.diagnostics["transitions_path"] = str(transitions_path)
        result.diagnostics["states_path"] = str(states_path)
        result.diagnostics["archived_state_count"] = int(len(state_payloads))

        summary_payload = result.to_dict()
        summary_payload["env_name"] = env_name
        summary_payload["scenario_type"] = effective_scenario_type
        summary_payload["seed"] = int(seed)
        summary_payload["artifact_stem"] = artifact_stem
        summary_payload["max_transitions_per_scenario"] = (
            int(max_transitions) if max_transitions is not None else None
        )
        summary_payload["transitions_path"] = str(transitions_path)
        summary_payload["states_path"] = str(states_path)
        summary_payload["archived_state_count"] = int(len(state_payloads))
        _write_solver_summary(path=artifact_root / f"{artifact_stem}.json", payload=summary_payload)
        result.diagnostics["effective_scenario_type"] = effective_scenario_type
        return result
    finally:
        solver.close()


def solve_custom_map_folder(
    *,
    custom_map_folder: str | Path,
    seed: int,
    search_mode: str,
    max_depth: Optional[int],
    max_nodes: Optional[int],
    max_steps: int,
    max_states: Optional[int] = None,
    max_transitions: Optional[int] = None,
    workers: Optional[int] = None,
    parallel_backend: str = "thread",
    chunk_size: int = 32,
    show_progress: Optional[bool] = None,
) -> Dict[str, Any]:
    map_files = _discover_custom_map_files(custom_map_folder)
    if not map_files:
        raise ValueError(f"No JSON custom maps found under {custom_map_folder}")

    resolved_folder = _resolve_custom_map_folder(custom_map_folder)
    folder_name = resolved_folder.name
    artifact_root = _build_run_output_dir(label=f"{folder_name}_batch", search_mode="bfs")
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    batch_start = time.perf_counter()
    solved_count = 0

    for map_path in map_files:
        spec = load_custom_map_spec(map_path)
        env_name = f"env/baba_custom_ascii_{str(spec['difficulty']).strip().lower()}"
        scenario_type = str(spec["scenario_name"]).strip()
        artifact_stem = _build_map_artifact_stem(folder_name=folder_name, map_path=map_path)
        result = solve_baba_env(
            artifact_root=artifact_root,
            artifact_stem=artifact_stem,
            env_name=env_name,
            scenario_type=scenario_type,
            seed=seed,
            search_mode=search_mode,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_steps=max_steps,
            max_states=max_states,
            max_transitions=max_transitions,
            workers=workers,
            parallel_backend=parallel_backend,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
        solved_count += int(result.solved)
        results.append(
            {
                "map_path": str(map_path),
                "scenario_type": scenario_type,
                "difficulty": str(spec["difficulty"]),
                "artifact_stem": artifact_stem,
                "solved": bool(result.solved),
                "transition_count": int(result.transition_count),
                "visited_states": int(result.visited_states),
                "expanded_states": int(result.stats.expanded_states),
                "elapsed_seconds": float(result.stats.elapsed_seconds),
                "elapsed_human": _format_duration(result.stats.elapsed_seconds),
                "transitions_path": result.diagnostics.get("transitions_path"),
                "states_path": result.diagnostics.get("states_path"),
                "stop_reason": result.diagnostics.get("stop_reason"),
            }
        )

    batch_elapsed_seconds = time.perf_counter() - batch_start
    batch_payload = {
        "custom_map_folder": str(resolved_folder),
        "folder_name": folder_name,
        "artifact_root": str(artifact_root),
        "search_mode": "bfs",
        "seed": int(seed),
        "max_transitions_per_scenario": (int(max_transitions) if max_transitions is not None else None),
        "map_count": len(map_files),
        "solved_count": int(solved_count),
        "failed_count": int(len(map_files) - solved_count),
        "elapsed_seconds": batch_elapsed_seconds,
        "elapsed_human": _format_duration(batch_elapsed_seconds),
        "results": results,
    }
    _write_solver_summary(path=artifact_root / "batch_summary.json", payload=batch_payload)
    return batch_payload


def solve_env_scenario_batch(
    *,
    env_name: str,
    scenario_types: Sequence[str],
    seed: int,
    search_mode: str,
    max_depth: Optional[int],
    max_nodes: Optional[int],
    max_steps: int,
    max_states: Optional[int] = None,
    max_transitions: Optional[int] = None,
    workers: Optional[int] = None,
    parallel_backend: str = "thread",
    chunk_size: int = 32,
    show_progress: Optional[bool] = None,
) -> Dict[str, Any]:
    normalized_scenarios = tuple(
        str(scenario).strip()
        for scenario in scenario_types
        if isinstance(scenario, str) and scenario.strip()
    )
    if not normalized_scenarios:
        raise ValueError(f"No scenarios discovered for {env_name}")

    label = f"{_sanitize_name(env_name.replace('/', '__'))}_batch"
    artifact_root = _build_run_output_dir(label=label, search_mode="bfs")
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    batch_start = time.perf_counter()
    solved_count = 0

    for scenario_type in normalized_scenarios:
        artifact_stem = _sanitize_name(scenario_type)
        result = solve_baba_env(
            artifact_root=artifact_root,
            artifact_stem=artifact_stem,
            env_name=env_name,
            scenario_type=scenario_type,
            seed=seed,
            search_mode=search_mode,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_steps=max_steps,
            max_states=max_states,
            max_transitions=max_transitions,
            workers=workers,
            parallel_backend=parallel_backend,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
        solved_count += int(result.solved)
        results.append(
            {
                "scenario_type": scenario_type,
                "artifact_stem": artifact_stem,
                "solved": bool(result.solved),
                "transition_count": int(result.transition_count),
                "visited_states": int(result.visited_states),
                "expanded_states": int(result.stats.expanded_states),
                "elapsed_seconds": float(result.stats.elapsed_seconds),
                "elapsed_human": _format_duration(result.stats.elapsed_seconds),
                "transitions_path": result.diagnostics.get("transitions_path"),
                "states_path": result.diagnostics.get("states_path"),
                "stop_reason": result.diagnostics.get("stop_reason"),
            }
        )

    batch_elapsed_seconds = time.perf_counter() - batch_start
    batch_payload = {
        "env_name": str(env_name),
        "artifact_root": str(artifact_root),
        "search_mode": "bfs",
        "seed": int(seed),
        "max_transitions_per_scenario": (int(max_transitions) if max_transitions is not None else None),
        "scenario_count": int(len(normalized_scenarios)),
        "solved_count": int(solved_count),
        "failed_count": int(len(normalized_scenarios) - solved_count),
        "elapsed_seconds": batch_elapsed_seconds,
        "elapsed_human": _format_duration(batch_elapsed_seconds),
        "results": results,
    }
    _write_solver_summary(path=artifact_root / "batch_summary.json", payload=batch_payload)
    return batch_payload


def solve_test_coverage_batch(
    *,
    difficulty: str = DEFAULT_COVERAGE_DIFFICULTY,
    scenario_split: str = DEFAULT_COVERAGE_SCENARIO_SPLIT,
    split_manifest_path: str | Path = DEFAULT_COVERAGE_SPLIT_MANIFEST,
    output_root: str | Path = DEFAULT_COVERAGE_OUTPUT_ROOT,
    scenario_type: Optional[str] = None,
    skip_existing: bool = True,
    seed: int,
    max_depth: Optional[int],
    max_nodes: Optional[int],
    max_steps: int,
    max_states: Optional[int] = None,
    max_transitions: Optional[int] = DEFAULT_COVERAGE_MAX_TRANSITIONS,
    workers: Optional[int] = None,
    parallel_backend: str = "thread",
    chunk_size: int = 32,
    show_progress: Optional[bool] = None,
) -> Dict[str, Any]:
    requested_scenarios = None
    if isinstance(scenario_type, str) and scenario_type.strip():
        requested_scenarios = [str(scenario_type).strip()]

    catalog = load_custom_map_catalog(
        difficulty,
        scenario_split=scenario_split,
        allowed_scenario_types=requested_scenarios,
        split_manifest_path=split_manifest_path,
    )
    if not catalog:
        requested_label = str(scenario_type).strip() if isinstance(scenario_type, str) else ""
        if requested_label:
            raise ValueError(
                f"No custom maps found for difficulty `{difficulty}` split `{scenario_split}` "
                f"matching scenario `{requested_label}`."
            )
        raise ValueError(f"No custom maps found for difficulty `{difficulty}` split `{scenario_split}`.")

    resolved_output_root = _resolve_project_path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path = _resolve_project_path(split_manifest_path)

    results: List[Dict[str, Any]] = []
    batch_start = time.perf_counter()
    solved_count = 0
    skipped_count = 0

    for map_name, entry in catalog.items():
        spec = entry["spec"]
        env_name = f"env/baba_custom_ascii_{str(spec['difficulty']).strip().lower()}"
        artifact_stem = _sanitize_name(map_name)
        if skip_existing:
            existing_summary = _load_existing_coverage_summary(
                artifact_root=resolved_output_root,
                artifact_stem=artifact_stem,
            )
            if existing_summary is not None:
                solved_count += int(bool(existing_summary.get("solved", False)))
                skipped_count += 1
                results.append(
                    _build_existing_coverage_result(
                        summary_payload=existing_summary,
                        scenario_type=str(map_name),
                        difficulty=str(spec["difficulty"]),
                        map_path=str(entry["path"]),
                        artifact_stem=artifact_stem,
                    )
                )
                continue
        result = solve_baba_env(
            artifact_root=resolved_output_root,
            artifact_stem=artifact_stem,
            env_name=env_name,
            scenario_type=str(map_name),
            seed=seed,
            search_mode="bfs",
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_steps=max_steps,
            max_states=max_states,
            max_transitions=max_transitions,
            workers=workers,
            parallel_backend=parallel_backend,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
        solved_count += int(result.solved)
        results.append(
            {
                "scenario_type": str(map_name),
                "difficulty": str(spec["difficulty"]),
                "map_path": str(entry["path"]),
                "artifact_stem": artifact_stem,
                "solved": bool(result.solved),
                "transition_count": int(result.transition_count),
                "visited_states": int(result.visited_states),
                "expanded_states": int(result.stats.expanded_states),
                "elapsed_seconds": float(result.stats.elapsed_seconds),
                "elapsed_human": _format_duration(result.stats.elapsed_seconds),
                "transitions_path": result.diagnostics.get("transitions_path"),
                "states_path": result.diagnostics.get("states_path"),
                "stop_reason": result.diagnostics.get("stop_reason"),
                "skipped_existing": False,
            }
        )

    batch_elapsed_seconds = time.perf_counter() - batch_start
    batch_payload = {
        "difficulty": str(difficulty),
        "scenario_split": str(scenario_split),
        "split_manifest_path": str(resolved_manifest_path),
        "output_root": str(resolved_output_root),
        "search_mode": "bfs",
        "seed": int(seed),
        "skip_existing": bool(skip_existing),
        "skipped_count": int(skipped_count),
        "max_transitions_per_scenario": (int(max_transitions) if max_transitions is not None else None),
        "scenario_count": int(len(results)),
        "solved_count": int(solved_count),
        "failed_count": int(len(results) - solved_count),
        "elapsed_seconds": batch_elapsed_seconds,
        "elapsed_human": _format_duration(batch_elapsed_seconds),
        "results": results,
    }
    if requested_scenarios is not None:
        batch_payload["requested_scenario_type"] = requested_scenarios[0]
    _write_solver_summary(path=resolved_output_root / "batch_summary.json", payload=batch_payload)
    return batch_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect BFS coverage transitions for Baba custom-map test scenarios.")
    parser.add_argument("--difficulty", default=DEFAULT_COVERAGE_DIFFICULTY)
    parser.add_argument("--scenario-split", default=DEFAULT_COVERAGE_SCENARIO_SPLIT)
    parser.add_argument("--split-manifest", default=str(DEFAULT_COVERAGE_SPLIT_MANIFEST))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_COVERAGE_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum BFS depth per scenario; omit for no explicit limit",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum expanded states per scenario; omit for no explicit limit",
    )
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=DEFAULT_COVERAGE_MAX_TRANSITIONS,
        help="Maximum canonical transitions to collect per scenario.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("thread", "process", "sequential"), default="thread")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--env-max-steps", type=int, default=DEFAULT_COVERAGE_ENV_MAX_STEPS)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute scenarios even when coverage artifacts already exist.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_max_steps = (
        int(args.env_max_steps)
        if args.env_max_steps is not None
        else DEFAULT_COVERAGE_ENV_MAX_STEPS
    )
    batch_summary = solve_test_coverage_batch(
        difficulty=str(args.difficulty),
        scenario_split=str(args.scenario_split),
        split_manifest_path=args.split_manifest,
        output_root=args.output_root,
        scenario_type=(str(args.scenario).strip() if isinstance(args.scenario, str) and args.scenario.strip() else None),
        skip_existing=(not bool(args.no_skip_existing)),
        seed=int(args.seed),
        max_depth=(int(args.max_depth) if args.max_depth is not None else None),
        max_nodes=(int(args.max_nodes) if args.max_nodes is not None else None),
        max_steps=env_max_steps,
        max_states=(int(args.max_states) if args.max_states is not None else None),
        max_transitions=(int(args.max_transitions) if args.max_transitions is not None else None),
        workers=(int(args.workers) if args.workers is not None else None),
        parallel_backend=str(args.backend),
        chunk_size=int(args.chunk_size),
        show_progress=(not bool(args.no_progress)),
    )

    if args.json:
        print(json.dumps(batch_summary, ensure_ascii=False, indent=2))
        return

    print(f"Difficulty: {batch_summary['difficulty']}")
    print(f"Split: {batch_summary['scenario_split']}")
    if "requested_scenario_type" in batch_summary:
        print(f"Scenario: {batch_summary['requested_scenario_type']}")
    print(f"Output root: {batch_summary['output_root']}")
    print(f"Processed {batch_summary['scenario_count']} scenarios in {batch_summary['elapsed_human']}")
    for item in batch_summary["results"]:
        print(
            f"{item['scenario_type']} | transitions={item['transition_count']} "
            f"| states={item['visited_states']} | expanded={item['expanded_states']} "
            f"| elapsed={item['elapsed_human']} | stop={item['stop_reason']} "
            f"| out={item['transitions_path']} | raw={item['states_path']}"
        )


if __name__ == "__main__":
    main()
