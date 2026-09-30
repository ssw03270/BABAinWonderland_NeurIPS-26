from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_MAP_ROOT = PROJECT_ROOT / "custom_maps"
DEFAULT_CUSTOM_MAP_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "configs" / "custom_map_splits_dot_test.yaml"
CUSTOM_MAP_DIFFICULTIES: Dict[str, Optional[Tuple[int, int]]] = {
    "easy": (8, 6),
    "medium": (11, 8),
    "hard": (14, 10),
    "original": None,
}

CANONICAL_ASCII_OBJECTS: Dict[str, str] = {
    "b": "baba",
    "w": "wall",
    "f": "flag",
    "k": "key",
    "d": "door",
    "l": "lava",
    "a": "algae",
    "c": "crab",
    "g": "grass",
    "j": "jelly",
    "e": "keke",
    "v": "love",
    "p": "pillar",
    "r": "rock",
    "s": "skull",
    "t": "star",
    "n": "tile",
    "u": "water",
    "m": "flower",
    "h": "brick",
    "q": "hedge",
    "y": "bubble",
    "x": "ice",
    "z": "cog",
    "i": "pipe",
    ";": "robot",
    ",": "bolt",
    ":": "bog",
    "'": "reed",
}

LEGACY_ASCII_OBJECT_ALIASES: Dict[str, str] = {
    "o": "crab",
}

ASCII_OBJECTS: Dict[str, str] = {
    **CANONICAL_ASCII_OBJECTS,
    **LEGACY_ASCII_OBJECT_ALIASES,
}

CANONICAL_ASCII_RULE_OBJECTS: Dict[str, str] = {
    "B": "baba",
    "W": "wall",
    "Z": "flag",
    "K": "key",
    "D": "door",
    "L": "lava",
    "A": "algae",
    "C": "crab",
    "G": "grass",
    "J": "jelly",
    "Q": "keke",
    "V": "love",
    "U": "pillar",
    "R": "rock",
    "F": "skull",
    "*": "star",
    "]": "tile",
    "~": "water",
    "{": "flower",
    "}": "brick",
    "=": "hedge",
    ")": "bubble",
    "(": "ice",
    "^": "cog",
    "/": "pipe",
    "&": "robot",
    "%": "bolt",
    "<": "bog",
    ">": "reed",
    "@": "text",
}

LEGACY_ASCII_RULE_OBJECT_ALIASES: Dict[str, str] = {
    "O": "crab",
}

ASCII_RULE_OBJECTS: Dict[str, str] = {
    **CANONICAL_ASCII_RULE_OBJECTS,
    **LEGACY_ASCII_RULE_OBJECT_ALIASES,
}

ASCII_RULE_OPERATORS: Dict[str, str] = {
    "I": "is",
    "+": "and",
}

ASCII_RULE_PROPERTIES: Dict[str, str] = {
    "Y": "you",
    "N": "win",
    "S": "stop",
    "M": "move",
    "H": "shift",
    "[": "sink",
    "!": "open",
    "P": "push",
    "?": "shut",
    "T": "hot",
    "E": "melt",
    "X": "defeat",
    "$": "float",
}

ASCII_BORDER_TOKEN = "#"
ASCII_EMPTY_TOKEN = "."

VALID_ASCII_TOKENS = (
    {ASCII_BORDER_TOKEN, ASCII_EMPTY_TOKEN}
    | set(ASCII_OBJECTS.keys())
    | set(ASCII_RULE_OBJECTS.keys())
    | set(ASCII_RULE_OPERATORS.keys())
    | set(ASCII_RULE_PROPERTIES.keys())
)
STACKABLE_ASCII_TOKENS = VALID_ASCII_TOKENS - {ASCII_BORDER_TOKEN, ASCII_EMPTY_TOKEN}

DEFAULT_EDITOR_PALETTE: List[str] = [
    ASCII_EMPTY_TOKEN,
    ASCII_BORDER_TOKEN,
    *CANONICAL_ASCII_OBJECTS.keys(),
    *CANONICAL_ASCII_RULE_OBJECTS.keys(),
    *ASCII_RULE_OPERATORS.keys(),
    *ASCII_RULE_PROPERTIES.keys(),
]

EDITOR_PALETTE_LABELS: Dict[str, str] = {
    ASCII_EMPTY_TOKEN: "empty",
    ASCII_BORDER_TOKEN: "border",
}

for _token, _name in ASCII_RULE_OPERATORS.items():
    EDITOR_PALETTE_LABELS[_token] = f"op {_name}"

for _token, _name in ASCII_OBJECTS.items():
    EDITOR_PALETTE_LABELS[_token] = f"world {_name}"

for _token, _name in ASCII_RULE_OBJECTS.items():
    EDITOR_PALETTE_LABELS[_token] = f"noun {_name}"

for _token, _name in ASCII_RULE_PROPERTIES.items():
    EDITOR_PALETTE_LABELS[_token] = f"prop {_name}"


CANONICAL_TOKEN_ALIASES: Dict[str, str] = {
    "o": "c",
    "O": "C",
}


def _canonicalize_layout_row_tokens(row: str) -> str:
    return "".join(CANONICAL_TOKEN_ALIASES.get(token, token) for token in row)


def normalize_custom_map_difficulty(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized not in CUSTOM_MAP_DIFFICULTIES:
        expected = ", ".join(sorted(CUSTOM_MAP_DIFFICULTIES))
        raise ValueError(f"Unsupported custom map difficulty `{raw_value}`. Expected one of: {expected}.")
    return normalized


def get_custom_map_storage_root(difficulty: str) -> Path:
    normalized = normalize_custom_map_difficulty(difficulty)
    if normalized == "original":
        return CUSTOM_MAP_ROOT
    return CUSTOM_MAP_ROOT / normalized


def custom_map_has_fixed_dimensions(difficulty: str) -> bool:
    normalized = normalize_custom_map_difficulty(difficulty)
    return CUSTOM_MAP_DIFFICULTIES[normalized] is not None


def get_custom_map_dimensions(difficulty: str) -> Tuple[int, int]:
    normalized = normalize_custom_map_difficulty(difficulty)
    dims = CUSTOM_MAP_DIFFICULTIES[normalized]
    if dims is None:
        raise ValueError(
            f"Custom map difficulty `{normalized}` uses variable map sizes. "
            "Read width/height from the map spec instead."
        )
    return dims


def infer_custom_map_difficulty(
    *,
    width: int,
    height: int,
    path: str | Path | None = None,
) -> str:
    if path is not None:
        path_obj = Path(path)
        parent_name = str(path_obj.parent.name).strip().lower()
        if parent_name in CUSTOM_MAP_DIFFICULTIES:
            return parent_name
    for difficulty, dims in CUSTOM_MAP_DIFFICULTIES.items():
        if dims is None:
            continue
        if dims == (int(width), int(height)):
            return difficulty
    if path is not None:
        path_obj = Path(path)
        try:
            parent = path_obj.resolve().parent
        except OSError:
            parent = path_obj.parent.resolve()
        try:
            custom_root = CUSTOM_MAP_ROOT.resolve()
        except OSError:
            custom_root = CUSTOM_MAP_ROOT
        if parent == custom_root:
            return "original"
    raise ValueError(
        f"Could not infer custom map difficulty from size {(width, height)}. "
        "Provide `difficulty` explicitly or store the map directly under custom_maps."
    )


def resolve_custom_map_path(
    path_or_str: str | Path,
    *,
    difficulty: str | None = None,
) -> Path:
    path = Path(path_or_str)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    root_candidate = CUSTOM_MAP_ROOT / path
    if root_candidate.exists():
        return root_candidate.resolve()
    if difficulty is not None:
        normalized = normalize_custom_map_difficulty(difficulty)
        return (get_custom_map_storage_root(normalized) / path).resolve()
    matches = [
        (get_custom_map_storage_root(difficulty_name) / path)
        for difficulty_name in CUSTOM_MAP_DIFFICULTIES
        if (get_custom_map_storage_root(difficulty_name) / path).exists()
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(
            f"Custom map path `{path}` is ambiguous across difficulty folders. "
            "Specify `difficulty` or include the folder name."
        )
    return root_candidate.resolve()


def resolve_custom_map_split_manifest_path(
    path_or_str: str | Path | None = None,
) -> Path:
    if path_or_str is None or not str(path_or_str).strip():
        return DEFAULT_CUSTOM_MAP_SPLIT_MANIFEST_PATH.resolve()
    path = Path(path_or_str)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def _normalize_scenario_name_sequence(
    raw_value: Any,
    *,
    field_name: str,
) -> List[str]:
    raw_items: List[Any] = []
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        raw_items = [part.strip() for part in raw_value.replace("\n", ",").split(",")]
    elif isinstance(raw_value, Iterable) and not isinstance(raw_value, Mapping):
        raw_items = list(raw_value)
    else:
        raise ValueError(f"`{field_name}` must be a string or sequence of scenario names.")

    normalized: List[str] = []
    seen = set()
    for raw_item in raw_items:
        item = str(raw_item or "").strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized


def load_custom_map_split_manifest(
    path_or_str: str | Path | None = None,
) -> Dict[str, Dict[str, List[str]]]:
    path = resolve_custom_map_split_manifest_path(path_or_str)
    if not path.exists():
        raise FileNotFoundError(f"Custom map split manifest not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Custom map split manifest must be a mapping: {path}")

    manifest: Dict[str, Dict[str, List[str]]] = {}
    for raw_difficulty, raw_splits in payload.items():
        difficulty = normalize_custom_map_difficulty(raw_difficulty)
        if not isinstance(raw_splits, Mapping):
            raise ValueError(
                f"Custom map split manifest entry `{difficulty}` must map split names to scenario lists."
            )
        normalized_splits: Dict[str, List[str]] = {}
        for raw_split_name, raw_scenario_names in raw_splits.items():
            split_name = str(raw_split_name or "").strip()
            if not split_name:
                raise ValueError(
                    f"Custom map split manifest `{path}` contains an empty split name for `{difficulty}`."
                )
            normalized_splits[split_name] = _normalize_scenario_name_sequence(
                raw_scenario_names,
                field_name=f"{difficulty}.{split_name}",
            )
        manifest[difficulty] = normalized_splits
    return manifest


def resolve_custom_map_scenario_names(
    difficulty: str,
    *,
    scenario_split: str | None = None,
    allowed_scenario_types: Any = None,
    split_manifest_path: str | Path | None = None,
) -> Optional[List[str]]:
    normalized_difficulty = normalize_custom_map_difficulty(difficulty)
    selected_names: List[str] = []

    split_name = str(scenario_split or "").strip()
    if split_name:
        manifest = load_custom_map_split_manifest(split_manifest_path)
        difficulty_splits = manifest.get(normalized_difficulty, {})
        if split_name not in difficulty_splits:
            available = ", ".join(sorted(difficulty_splits)) or "<none>"
            raise ValueError(
                f"Unknown custom map split `{split_name}` for difficulty `{normalized_difficulty}`. "
                f"Available splits: {available}."
            )
        selected_names = list(difficulty_splits[split_name])

    allowed_names = _normalize_scenario_name_sequence(
        allowed_scenario_types,
        field_name="allowed_scenario_types",
    )
    if allowed_names:
        if selected_names:
            allowed_set = set(allowed_names)
            selected_names = [name for name in selected_names if name in allowed_set]
            if not selected_names:
                raise ValueError(
                    "Custom map selection produced an empty scenario set after intersecting "
                    f"`scenario_split={split_name}` with `allowed_scenario_types`."
                )
        else:
            selected_names = allowed_names

    if not selected_names:
        return None
    return selected_names


def build_empty_custom_map_spec(
    *,
    width: int | None = None,
    height: int | None = None,
    difficulty: str = "easy",
    scenario_name: str = "custom_map",
    display_name: str | None = None,
) -> Dict[str, Any]:
    normalized_difficulty = normalize_custom_map_difficulty(difficulty)
    fixed_dims = CUSTOM_MAP_DIFFICULTIES[normalized_difficulty]
    if fixed_dims is not None:
        default_width, default_height = fixed_dims
        if width is None:
            width = default_width
        if height is None:
            height = default_height
    elif width is None or height is None:
        raise ValueError(
            f"Difficulty `{normalized_difficulty}` requires explicit width and height."
        )
    width = int(width)
    height = int(height)
    if fixed_dims is not None and (width, height) != fixed_dims:
        raise ValueError(
            f"Difficulty `{normalized_difficulty}` requires fixed size "
            f"{fixed_dims}, got {(width, height)}."
        )
    if width < 3 or height < 3:
        raise ValueError("width and height must be at least 3 to include a border.")
    layout = []
    for y in range(height):
        if y in {0, height - 1}:
            layout.append(ASCII_BORDER_TOKEN * width)
        else:
            layout.append(
                ASCII_BORDER_TOKEN
                + (ASCII_EMPTY_TOKEN * (width - 2))
                + ASCII_BORDER_TOKEN
            )
    return {
        "scenario_name": scenario_name,
        "display_name": str(display_name or "").strip(),
        "difficulty": normalized_difficulty,
        "width": int(width),
        "height": int(height),
        "layout": layout,
        "dynamic_directions": {},
        "cell_stacks": {},
        "inactive_rules": [],
        "target_plan": "",
    }


def _normalize_dynamic_directions(
    raw_value: Any,
) -> Dict[Tuple[int, int], int]:
    normalized: Dict[Tuple[int, int], int] = {}
    if isinstance(raw_value, Mapping):
        items: Iterable[Tuple[Any, Any]] = raw_value.items()
    elif isinstance(raw_value, list):
        items = []
        for row in raw_value:
            if not isinstance(row, Mapping):
                continue
            position = row.get("position")
            direction = row.get("direction")
            items = list(items) + [(position, direction)]
    else:
        return normalized

    for raw_key, raw_direction in items:
        position: Optional[Tuple[int, int]] = None
        if isinstance(raw_key, str):
            parts = [part.strip() for part in raw_key.split(",")]
            if len(parts) == 2 and parts[0] and parts[1]:
                try:
                    position = (int(parts[0]), int(parts[1]))
                except ValueError:
                    position = None
        elif (
            isinstance(raw_key, (list, tuple))
            and len(raw_key) == 2
            and isinstance(raw_key[0], int)
            and isinstance(raw_key[1], int)
        ):
            position = (int(raw_key[0]), int(raw_key[1]))

        if position is None:
            continue

        try:
            direction = int(raw_direction)
        except (TypeError, ValueError):
            continue
        if direction not in {0, 1, 2, 3}:
            continue
        normalized[position] = direction
    return normalized


def _serialize_dynamic_directions(
    dynamic_directions: Mapping[Tuple[int, int], int],
) -> Dict[str, int]:
    serialized: Dict[str, int] = {}
    for position, direction in dynamic_directions.items():
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not isinstance(position[0], int)
            or not isinstance(position[1], int)
        ):
            continue
        try:
            normalized_direction = int(direction)
        except (TypeError, ValueError):
            continue
        serialized[f"{position[0]},{position[1]}"] = normalized_direction
    return dict(sorted(serialized.items()))


def _parse_stack_position(raw_key: Any) -> Tuple[int, int] | None:
    if isinstance(raw_key, str):
        parts = [part.strip() for part in raw_key.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None
    if (
        isinstance(raw_key, (list, tuple))
        and len(raw_key) == 2
        and isinstance(raw_key[0], int)
        and isinstance(raw_key[1], int)
    ):
        return int(raw_key[0]), int(raw_key[1])
    return None


def _normalize_stack_token(raw_token: Any) -> str:
    token = str(raw_token or "").strip()
    if len(token) != 1:
        raise ValueError(f"Stack token must be a single ASCII token, got `{raw_token}`.")
    token = CANONICAL_TOKEN_ALIASES.get(token, token)
    if token not in STACKABLE_ASCII_TOKENS:
        raise ValueError(f"Unsupported stacked token `{token}`.")
    return token


def _normalize_cell_stacks(
    raw_value: Any,
    *,
    width: int,
    height: int,
    layout: List[str],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    normalized: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    if raw_value is None or raw_value == "":
        return normalized
    if not isinstance(raw_value, Mapping):
        raise ValueError("Custom map spec `cell_stacks` must be a mapping of positions to token lists.")

    for raw_key, raw_entries in raw_value.items():
        position = _parse_stack_position(raw_key)
        if position is None:
            raise ValueError(f"Invalid stacked-cell position `{raw_key}`.")
        x, y = position
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Stacked-cell position {(x, y)} is outside the map bounds {(width, height)}.")
        if x in {0, width - 1} or y in {0, height - 1}:
            raise ValueError(f"Custom map border cells cannot define `cell_stacks`: {(x, y)}.")
        base_token = layout[y][x]
        if base_token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
            raise ValueError(
                f"Custom map `cell_stacks` at {(x, y)} requires a non-empty base token in `layout`, "
                f"found `{base_token}`."
            )
        if not isinstance(raw_entries, list):
            raise ValueError(f"Custom map `cell_stacks` entry at {(x, y)} must be a list.")

        normalized_entries: List[Dict[str, Any]] = []
        for index, raw_entry in enumerate(raw_entries):
            direction: int | None = None
            if isinstance(raw_entry, str):
                token = _normalize_stack_token(raw_entry)
            elif isinstance(raw_entry, Mapping):
                token = _normalize_stack_token(raw_entry.get("token"))
                raw_direction = raw_entry.get("direction")
                if raw_direction is not None:
                    try:
                        direction = int(raw_direction)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid stacked direction `{raw_direction}` at {(x, y)}[{index}]."
                        ) from exc
                    if direction not in {0, 1, 2, 3}:
                        raise ValueError(
                            f"Stacked direction at {(x, y)}[{index}] must be one of 0, 1, 2, 3."
                        )
                    if token not in ASCII_OBJECTS:
                        raise ValueError(
                            f"Only world-object stacked tokens can carry a direction at {(x, y)}[{index}]."
                        )
            else:
                raise ValueError(
                    f"Custom map `cell_stacks` entry at {(x, y)}[{index}] must be a token string or mapping."
                )

            entry: Dict[str, Any] = {"token": token}
            if direction is not None:
                entry["direction"] = direction
            normalized_entries.append(entry)

        if normalized_entries:
            normalized[position] = normalized_entries

    return normalized


def _serialize_cell_stacks(
    cell_stacks: Mapping[Tuple[int, int], List[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    serialized: Dict[str, List[Dict[str, Any]]] = {}
    for position, entries in cell_stacks.items():
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not isinstance(position[0], int)
            or not isinstance(position[1], int)
        ):
            continue
        serialized_entries: List[Dict[str, Any]] = []
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                continue
            try:
                token = _normalize_stack_token(raw_entry.get("token"))
            except ValueError:
                continue
            entry: Dict[str, Any] = {"token": token}
            raw_direction = raw_entry.get("direction")
            if raw_direction is not None:
                try:
                    direction = int(raw_direction)
                except (TypeError, ValueError):
                    direction = None
                if direction in {0, 1, 2, 3} and token in ASCII_OBJECTS:
                    entry["direction"] = direction
            serialized_entries.append(entry)
        if serialized_entries:
            serialized[f"{position[0]},{position[1]}"] = serialized_entries
    return dict(sorted(serialized.items()))


def normalize_custom_map_spec(payload: Mapping[str, Any]) -> Dict[str, Any]:
    scenario_name = str(payload.get("scenario_name") or "custom_map").strip() or "custom_map"
    display_name = str(payload.get("display_name") or "").strip()

    width = payload.get("width")
    height = payload.get("height")
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("Custom map spec must provide integer width and height.") from exc
    if width < 3 or height < 3:
        raise ValueError("Custom map width and height must be at least 3.")

    difficulty = payload.get("difficulty")
    if difficulty is None:
        difficulty = infer_custom_map_difficulty(
            width=width,
            height=height,
            path=payload.get("source_path"),
        )
    difficulty = normalize_custom_map_difficulty(difficulty)
    fixed_dims = CUSTOM_MAP_DIFFICULTIES[difficulty]
    if fixed_dims is not None and (width, height) != fixed_dims:
        raise ValueError(
            f"Custom map difficulty `{difficulty}` requires size "
            f"{fixed_dims}, got {(width, height)}."
        )

    raw_layout = payload.get("layout")
    if not isinstance(raw_layout, list):
        raise ValueError("Custom map spec `layout` must be a list of strings.")
    if len(raw_layout) != height:
        raise ValueError(
            f"Custom map layout height mismatch: expected {height}, got {len(raw_layout)}."
        )

    layout: List[str] = []
    for y, row in enumerate(raw_layout):
        if not isinstance(row, str):
            raise ValueError(f"Custom map layout row {y} must be a string.")
        row = _canonicalize_layout_row_tokens(row)
        if len(row) != width:
            raise ValueError(
                f"Custom map layout row {y} width mismatch: expected {width}, got {len(row)}."
            )
        for x, token in enumerate(row):
            if token not in VALID_ASCII_TOKENS:
                raise ValueError(
                    f"Unsupported token `{token}` at {(x, y)} in custom map layout."
                )
            is_border = x in {0, width - 1} or y in {0, height - 1}
            if is_border and token != ASCII_BORDER_TOKEN:
                raise ValueError(
                    f"Custom map border must use `{ASCII_BORDER_TOKEN}` at {(x, y)}, found `{token}`."
                )
            if not is_border and token == ASCII_BORDER_TOKEN:
                raise ValueError(
                    f"Custom map uses `{ASCII_BORDER_TOKEN}` inside the playable area at {(x, y)}."
                )
        layout.append(row)

    dynamic_directions = _normalize_dynamic_directions(
        payload.get("dynamic_directions") or {}
    )
    cell_stacks = _normalize_cell_stacks(
        payload.get("cell_stacks") or {},
        width=width,
        height=height,
        layout=layout,
    )
    inactive_rules = payload.get("inactive_rules") or []
    if not isinstance(inactive_rules, list):
        inactive_rules = []
    normalized_inactive_rules = []
    for row in inactive_rules:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            continue
        normalized_inactive_rules.append(list(row))

    return {
        "scenario_name": scenario_name,
        "display_name": display_name,
        "difficulty": difficulty,
        "width": width,
        "height": height,
        "layout": layout,
        "dynamic_directions": dynamic_directions,
        "cell_stacks": cell_stacks,
        "inactive_rules": normalized_inactive_rules,
        "target_plan": str(payload.get("target_plan") or ""),
    }


def load_custom_map_spec(path_or_str: str | Path) -> Dict[str, Any]:
    path = resolve_custom_map_path(path_or_str)
    if not path.exists():
        raise FileNotFoundError(f"Custom map file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Custom map root must be a mapping: {path}")
    payload = dict(payload)
    payload.setdefault("source_path", str(path))
    spec = normalize_custom_map_spec(payload)
    spec["source_path"] = str(path)
    return spec


def save_custom_map_spec(spec: Mapping[str, Any], path_or_str: str | Path) -> Path:
    normalized = normalize_custom_map_spec(spec)
    path = resolve_custom_map_path(
        path_or_str,
        difficulty=normalized["difficulty"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = dict(normalized)
    if not serialized.get("display_name"):
        serialized.pop("display_name", None)
    serialized["dynamic_directions"] = _serialize_dynamic_directions(
        normalized["dynamic_directions"]
    )
    serialized["cell_stacks"] = _serialize_cell_stacks(normalized["cell_stacks"])
    path.write_text(
        json.dumps(serialized, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return path


def list_custom_map_paths(difficulty: str) -> List[Path]:
    normalized = normalize_custom_map_difficulty(difficulty)
    root = get_custom_map_storage_root(normalized)
    paths: List[Path] = []
    if root.exists():
        paths.extend(path for path in root.glob("*.json") if path.is_file())
    unique_paths = {path.resolve(): path.resolve() for path in paths}
    return sorted(unique_paths.values())


def load_custom_map_catalog(
    difficulty: str,
    *,
    scenario_split: str | None = None,
    allowed_scenario_types: Any = None,
    split_manifest_path: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    normalized = normalize_custom_map_difficulty(difficulty)
    catalog: Dict[str, Dict[str, Any]] = {}
    for path in list_custom_map_paths(normalized):
        spec = load_custom_map_spec(path)
        scenario_name = str(spec.get("scenario_name") or path.stem)
        if spec["difficulty"] != normalized:
            raise ValueError(
                f"Custom map `{path}` declares difficulty `{spec['difficulty']}` "
                f"but is stored under `{normalized}`."
            )
        if scenario_name in catalog:
            if normalized != "original":
                raise ValueError(
                    f"Duplicate custom map scenario_name `{scenario_name}` found for difficulty `{normalized}`."
                )
            fallback_name = path.stem
            scenario_name = fallback_name
            suffix = 2
            while scenario_name in catalog:
                scenario_name = f"{fallback_name}_{suffix}"
                suffix += 1
            spec = dict(spec)
            spec["scenario_name"] = scenario_name
        catalog[scenario_name] = {
            "path": str(path),
            "spec": spec,
        }

    selected_scenario_names = resolve_custom_map_scenario_names(
        normalized,
        scenario_split=scenario_split,
        allowed_scenario_types=allowed_scenario_types,
        split_manifest_path=split_manifest_path,
    )
    if selected_scenario_names is None:
        return catalog

    missing = [name for name in selected_scenario_names if name not in catalog]
    if missing:
        available = ", ".join(sorted(catalog)) or "<none>"
        raise ValueError(
            f"Custom map selection for difficulty `{normalized}` references unknown scenario(s): "
            f"{', '.join(missing)}. Available scenarios: {available}."
        )

    return {
        scenario_name: catalog[scenario_name]
        for scenario_name in selected_scenario_names
    }
