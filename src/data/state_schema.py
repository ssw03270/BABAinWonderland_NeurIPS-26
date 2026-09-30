from __future__ import annotations

from collections import Counter
import json
import struct
from numbers import Integral
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


StateLike = Union[str, Dict[str, Any]]

STATE_KEY_PREFIX = "ps:"

DIRECTION_NAMES = {
    0: "facing right",
    1: "facing down",
    2: "facing left",
    3: "facing up",
}

_RAW_DIRECTION_ALIASES = {
    "right": "facing right",
    "down": "facing down",
    "left": "facing left",
    "up": "facing up",
}

_TYPE_ALIASES = {
    "a": "rule_noun",
    "b": "rule_operator",
    "c": "rule_property",
    "d": "world_object",
    "rule_noun": "rule_noun",
    "rule_operator": "rule_operator",
    "rule_property": "rule_property",
    "world_object": "world_object",
    "noun": "rule_noun",
    "operator": "rule_operator",
    "property": "rule_property",
    "object": "world_object",
}

_STATE_VERSION = 1
_STATE_HAS_GRID_SIZE = 1 << 0
_STATE_HAS_STEP = 1 << 1
_STATE_HAS_OBJECTS = 1 << 2
_STEP_HAS_TERMINATED = 1 << 0
_OBJECT_HAS_TYPE = 1 << 0
_OBJECT_HAS_WORD = 1 << 1
_OBJECT_HAS_POSITION = 1 << 2
_OBJECT_HAS_DIRECTION = 1 << 3
_OBJECT_HAS_COLOR = 1 << 4


def _normalize_object_type_fallback(raw_type: Any) -> str:
    value = str(raw_type or "unknown").strip().lower() or "unknown"
    if value.startswith("f") and len(value) > 1 and value != "floor":
        return value[1:]
    return value


def normalize_state_direction(raw_direction: Any) -> Optional[str]:
    if isinstance(raw_direction, Integral):
        return DIRECTION_NAMES.get(int(raw_direction), "unknown")
    if not isinstance(raw_direction, str):
        return None
    direction = raw_direction.strip().lower()
    if not direction:
        return None
    if direction in DIRECTION_NAMES.values():
        return direction
    return _RAW_DIRECTION_ALIASES.get(direction, "unknown")


def normalize_raw_baba_object_schema(*, obj: Any, raw_type: Any) -> Tuple[str, str]:
    normalized_type = str(raw_type or getattr(obj, "type", "unknown")).strip().lower() or "unknown"
    raw_word = getattr(obj, "name", None)
    if isinstance(raw_word, str):
        word = raw_word.strip().lower()
        if word:
            resolved_word = "lava" if word == "flava" else word
        else:
            resolved_word = _normalize_object_type_fallback(normalized_type) or "unknown"
    else:
        resolved_word = _normalize_object_type_fallback(normalized_type) or "unknown"

    if normalized_type == "rule_object":
        return "rule_noun", resolved_word
    if normalized_type == "rule_is":
        return "rule_operator", "is"
    if normalized_type == "rule_and":
        return "rule_operator", "and"
    if normalized_type in {"rule_property", "rule_color"}:
        return "rule_property", resolved_word
    return "world_object", resolved_word


def normalize_state_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("State must be a dictionary.")

    normalized = dict(state)
    raw_step = normalized.get("step")
    if isinstance(raw_step, dict):
        step = dict(raw_step)
        step.pop("truncated", None)
        normalized["step"] = step
    return normalized


def parse_state_json(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid state JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError("State JSON must decode to a dictionary.")
    return normalize_state_payload(parsed)


def dump_state_json(state: Dict[str, Any]) -> str:
    try:
        return json.dumps(
            normalize_state_payload(state),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"State is not JSON-serializable: {e}") from e


def canonicalize_state_obj(state: Dict[str, Any]) -> str:
    return dump_state_json(state)


def canonicalize_state_json(state: StateLike) -> str:
    if isinstance(state, dict):
        return canonicalize_state_obj(state)
    if isinstance(state, str):
        return canonicalize_state_obj(parse_state_json(state))
    raise ValueError("state must be either dict or JSON string.")


def _canonical_token(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def states_equivalent(expected: Any, predicted: Any, path: str = "$") -> bool:
    if type(expected) is not type(predicted):
        return False

    if isinstance(expected, dict):
        if set(expected.keys()) != set(predicted.keys()):
            return False
        for key in expected.keys():
            child_path = f"{path}.{key}"
            if not states_equivalent(expected[key], predicted[key], path=child_path):
                return False
        return True

    if isinstance(expected, list):
        if _is_unordered_object_list(path, expected, predicted):
            return _unordered_list_equal(expected, predicted)
        if len(expected) != len(predicted):
            return False
        for idx, (exp_item, pred_item) in enumerate(zip(expected, predicted)):
            child_path = f"{path}[{idx}]"
            if not states_equivalent(exp_item, pred_item, path=child_path):
                return False
        return True

    return expected == predicted


def _is_unordered_object_list(path: str, expected: List[Any], predicted: List[Any]) -> bool:
    if not path.endswith(".objects"):
        return False
    return all(isinstance(item, dict) for item in expected) and all(
        isinstance(item, dict) for item in predicted
    )


def _unordered_list_equal(expected: List[Any], predicted: List[Any]) -> bool:
    expected_tokens = Counter(_canonical_token(item) for item in expected)
    predicted_tokens = Counter(_canonical_token(item) for item in predicted)
    return expected_tokens == predicted_tokens


def _normalize_grid_size(grid_size: Any) -> Tuple[int, int]:
    if (
        isinstance(grid_size, (tuple, list))
        and len(grid_size) == 2
        and isinstance(grid_size[0], Integral)
        and isinstance(grid_size[1], Integral)
    ):
        return int(grid_size[0]), int(grid_size[1])
    return 0, 0


def _trim_grid_size(raw_width: int, raw_height: int) -> Tuple[int, int]:
    return max(0, int(raw_width) - 2), max(0, int(raw_height) - 2)


def _is_outer_border_position(x: int, y: int, raw_width: int, raw_height: int) -> bool:
    return x == 0 or y == 0 or x == raw_width - 1 or y == raw_height - 1


def _normalize_word_aliases(
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    if not isinstance(word_aliases, Mapping):
        return {}

    normalized: Dict[str, Dict[str, str]] = {}
    for raw_type, raw_mapping in word_aliases.items():
        if not isinstance(raw_type, str) or not isinstance(raw_mapping, Mapping):
            continue
        obj_type = str(raw_type).strip().lower()
        if not obj_type:
            continue
        alias_map: Dict[str, str] = {}
        for raw_word, raw_alias in raw_mapping.items():
            if not isinstance(raw_word, str) or not isinstance(raw_alias, str):
                continue
            word = raw_word.strip().lower()
            alias = raw_alias.strip()
            if not word or not alias:
                continue
            alias_map[word] = alias
        if alias_map:
            normalized[obj_type] = alias_map
    return normalized


def build_state_payload(
    *,
    grid_size: Tuple[int, int],
    objects: Iterable[Mapping[str, Any]],
    terminated: bool,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    raw_width, raw_height = _normalize_grid_size(grid_size)
    width, height = _trim_grid_size(raw_width, raw_height)
    normalized_aliases = _normalize_word_aliases(word_aliases)
    rows: List[Dict[str, Any]] = []

    for raw in objects:
        if not isinstance(raw, Mapping):
            continue

        pos = raw.get("position")
        if (
            not isinstance(pos, list)
            or len(pos) != 2
            or not isinstance(pos[0], Integral)
            or not isinstance(pos[1], Integral)
        ):
            continue

        x = int(pos[0])
        y = int(pos[1])
        if x < 0 or y < 0:
            continue
        if raw_width > 0 and raw_height > 0:
            if x >= raw_width or y >= raw_height:
                continue
            if _is_outer_border_position(x=x, y=y, raw_width=raw_width, raw_height=raw_height):
                continue

        raw_type = str(raw.get("type", "object")).strip().lower() or "object"
        obj_type = _TYPE_ALIASES.get(raw_type, "object")
        raw_word = raw.get("word")
        original_word = (
            str(raw_word).strip().lower()
            if isinstance(raw_word, str) and raw_word.strip()
            else obj_type
        )
        word_value = normalized_aliases.get(obj_type, {}).get(original_word, original_word)
        row = {
            "type": obj_type,
            "word": word_value,
            "position": [x - 1, y - 1],
        }
        raw_color = raw.get("color")
        if isinstance(raw_color, str):
            color = raw_color.strip().lower()
            if color:
                row["color"] = color
        direction_name = normalize_state_direction(raw.get("direction"))
        if direction_name is not None:
            row["direction"] = direction_name
        rows.append(row)

    return {
        "grid_size": [width, height],
        "step": {
            "terminated": bool(terminated),
        },
        "objects": rows,
    }


def _pack_text(buffer: bytearray, text: str) -> None:
    encoded = str(text).encode("utf-8")
    buffer.extend(struct.pack("<I", len(encoded)))
    buffer.extend(encoded)


def _pack_optional_text(buffer: bytearray, value: Optional[str]) -> None:
    if value is None:
        buffer.extend(b"\x00")
        return
    buffer.extend(b"\x01")
    _pack_text(buffer, value)


def pack_state_obj(state: Dict[str, Any]) -> bytes:
    normalized = normalize_state_payload(state)
    raw_grid_size = normalized.get("grid_size")
    state_flags = 0
    width = 0
    height = 0
    if (
        "grid_size" in normalized
        and
        isinstance(raw_grid_size, list)
        and len(raw_grid_size) == 2
        and isinstance(raw_grid_size[0], Integral)
        and isinstance(raw_grid_size[1], Integral)
    ):
        width = int(raw_grid_size[0])
        height = int(raw_grid_size[1])
        state_flags |= _STATE_HAS_GRID_SIZE

    raw_step = normalized.get("step")
    step_flags = 0
    terminated = None
    step_extras: Dict[str, Any] = {}
    if isinstance(raw_step, Mapping):
        state_flags |= _STATE_HAS_STEP
        if "terminated" in raw_step:
            terminated = bool(raw_step.get("terminated"))
            step_flags |= _STEP_HAS_TERMINATED
        step_extras = {
            str(key): value
            for key, value in raw_step.items()
            if str(key) not in {"terminated", "truncated"}
        }

    objects = normalized.get("objects")
    object_rows = list(objects) if isinstance(objects, list) else None
    if isinstance(object_rows, list):
        state_flags |= _STATE_HAS_OBJECTS

    excluded_top_level_keys = set()
    if state_flags & _STATE_HAS_GRID_SIZE:
        excluded_top_level_keys.add("grid_size")
    if state_flags & _STATE_HAS_STEP:
        excluded_top_level_keys.add("step")
    if state_flags & _STATE_HAS_OBJECTS:
        excluded_top_level_keys.add("objects")
    top_extras = {
        str(key): value
        for key, value in normalized.items()
        if str(key) not in excluded_top_level_keys
    }

    buffer = bytearray()
    buffer.extend(struct.pack("<BB", _STATE_VERSION, int(state_flags)))
    if state_flags & _STATE_HAS_GRID_SIZE:
        buffer.extend(struct.pack("<ii", int(width), int(height)))
    buffer.extend(struct.pack("<B", int(step_flags)))
    if step_flags & _STEP_HAS_TERMINATED:
        buffer.extend(struct.pack("<?", bool(terminated)))
    _pack_text(buffer, _canonical_token(step_extras))
    _pack_text(buffer, _canonical_token(top_extras))
    if not (state_flags & _STATE_HAS_OBJECTS):
        return bytes(buffer)

    buffer.extend(struct.pack("<I", len(object_rows or [])))

    for raw in object_rows or []:
        row = raw if isinstance(raw, Mapping) else {}
        object_flags = 0
        excluded_object_keys = set()

        raw_type = row.get("type")
        type_text: Optional[str] = None
        if "type" in row and isinstance(raw_type, str):
            type_text = str(raw_type)
            object_flags |= _OBJECT_HAS_TYPE
            excluded_object_keys.add("type")

        raw_word = row.get("word")
        word_text: Optional[str] = None
        if "word" in row and isinstance(raw_word, str):
            word_text = str(raw_word)
            object_flags |= _OBJECT_HAS_WORD
            excluded_object_keys.add("word")

        position = row.get("position")
        x = 0
        y = 0
        if (
            "position" in row
            and
            isinstance(position, list)
            and len(position) == 2
            and isinstance(position[0], Integral)
            and isinstance(position[1], Integral)
        ):
            x = int(position[0])
            y = int(position[1])
            object_flags |= _OBJECT_HAS_POSITION
            excluded_object_keys.add("position")

        direction_value = row.get("direction")
        direction_text: Optional[str] = None
        if "direction" in row and isinstance(direction_value, str):
            direction_text = str(direction_value)
            object_flags |= _OBJECT_HAS_DIRECTION
            excluded_object_keys.add("direction")

        color_value = row.get("color")
        color_text: Optional[str] = None
        if "color" in row and isinstance(color_value, str):
            color_text = str(color_value)
            object_flags |= _OBJECT_HAS_COLOR
            excluded_object_keys.add("color")

        object_extras = {
            str(key): value
            for key, value in row.items()
            if str(key) not in excluded_object_keys
        }
        buffer.extend(struct.pack("<B", int(object_flags)))
        _pack_optional_text(buffer, type_text)
        _pack_optional_text(buffer, word_text)
        if object_flags & _OBJECT_HAS_POSITION:
            buffer.extend(struct.pack("<ii", int(x), int(y)))
        _pack_optional_text(buffer, direction_text)
        _pack_optional_text(buffer, color_text)
        _pack_text(buffer, _canonical_token(object_extras))
    return bytes(buffer)
