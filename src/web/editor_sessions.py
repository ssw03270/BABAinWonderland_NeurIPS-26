from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid
from typing import Any, Dict, Optional

from src.editor_runtime import (
    append_token_to_cell_stack,
    build_next_custom_map_path,
    clear_cell_stack,
    describe_editor_token_name,
    describe_editor_token_for_sprite,
    get_cell_stack_entries,
    get_visible_cell_token,
    is_border,
    pop_top_token_from_cell_stack,
    set_tile,
)
from src.environments.custom_map_spec import (
    ASCII_BORDER_TOKEN,
    ASCII_EMPTY_TOKEN,
    CUSTOM_MAP_DIFFICULTIES,
    DEFAULT_EDITOR_PALETTE,
    EDITOR_PALETTE_LABELS,
    build_empty_custom_map_spec,
    load_custom_map_catalog,
    load_custom_map_spec,
    normalize_custom_map_difficulty,
    resolve_custom_map_path,
    save_custom_map_spec,
)
from src.ui.unified_scene import build_custom_map_board_scene
from src.web.map_display_names import (
    get_custom_map_display_name_index,
    refresh_custom_map_display_name_index,
    resolve_custom_map_spec_display_name,
)


DEFAULT_ORIGINAL_EDITOR_SIZE = (12, 8)


def _resolve_editor_display_name(
    *,
    spec: Dict[str, Any],
    map_path: Path,
    index: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    return resolve_custom_map_spec_display_name(
        spec,
        map_path,
        index=index,
    )


def _palette_payload() -> list[Dict[str, Any]]:
    return [
        {
            "token": token,
            "label": EDITOR_PALETTE_LABELS.get(token, token),
            "sprite": describe_editor_token_for_sprite(token),
        }
        for token in DEFAULT_EDITOR_PALETTE
    ]


def _default_dimensions_for_difficulty(difficulty: str) -> tuple[int, int]:
    normalized = normalize_custom_map_difficulty(difficulty)
    dims = CUSTOM_MAP_DIFFICULTIES[normalized]
    if dims is not None:
        return int(dims[0]), int(dims[1])
    return DEFAULT_ORIGINAL_EDITOR_SIZE


def build_editor_catalog_payload(*, current_map_path: Optional[Path] = None) -> Dict[str, Any]:
    difficulties: list[Dict[str, Any]] = []
    total_maps = 0
    current_path_text = None
    if current_map_path is not None:
        current_path_text = str(Path(current_map_path).resolve())
    display_name_index = get_custom_map_display_name_index()
    for difficulty in CUSTOM_MAP_DIFFICULTIES:
        normalized = normalize_custom_map_difficulty(difficulty)
        default_width, default_height = _default_dimensions_for_difficulty(normalized)
        maps: list[Dict[str, Any]] = []
        for scenario_name, entry in load_custom_map_catalog(normalized).items():
            spec = dict(entry["spec"])
            map_path = Path(entry["path"]).resolve()
            display_name = _resolve_editor_display_name(
                spec=spec,
                map_path=map_path,
                index=display_name_index,
            )
            maps.append(
                {
                    "difficulty": normalized,
                    "scenarioName": str(spec.get("scenario_name") or scenario_name),
                    "displayName": display_name,
                    "label": display_name or str(spec.get("scenario_name") or scenario_name),
                    "mapPath": str(map_path),
                    "mapFileName": map_path.name,
                    "width": int(spec.get("width", 0) or 0),
                    "height": int(spec.get("height", 0) or 0),
                    "isCurrent": current_path_text == str(map_path),
                }
            )
        total_maps += len(maps)
        fixed_dims = CUSTOM_MAP_DIFFICULTIES[normalized]
        difficulties.append(
            {
                "id": normalized,
                "label": normalized,
                "fixedSize": fixed_dims is not None,
                "defaultWidth": default_width,
                "defaultHeight": default_height,
                "maps": maps,
                "mapCount": len(maps),
            }
        )
    return {
        "difficulties": difficulties,
        "totalMaps": total_maps,
        "currentMapPath": current_path_text,
    }


@dataclass
class EditorSession:
    session_id: str
    map_path: Path
    spec: Dict[str, Any]
    dirty: bool = False
    status: str = ""

    @classmethod
    def from_spec(
        cls,
        *,
        map_path: Path,
        spec: Dict[str, Any],
        dirty: bool,
        status: str,
    ) -> "EditorSession":
        return cls(
            session_id=uuid.uuid4().hex,
            map_path=map_path,
            spec=spec,
            dirty=bool(dirty),
            status=str(status),
        )

    @classmethod
    def create(
        cls,
        *,
        map_file: Optional[str] = None,
        difficulty: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scenario_name: Optional[str] = None,
        display_name: Optional[str] = None,
        create_new: bool = False,
    ) -> "EditorSession":
        session_id = uuid.uuid4().hex
        requested_display_name = (
            str(display_name).strip()
            if isinstance(display_name, str) and display_name.strip()
            else ""
        )
        if isinstance(map_file, str) and map_file.strip():
            resolved_path = resolve_custom_map_path(map_file, difficulty=difficulty)
            if resolved_path.exists() and not create_new:
                spec = load_custom_map_spec(resolved_path)
                if requested_display_name:
                    spec["display_name"] = requested_display_name
                resolved_display_name = _resolve_editor_display_name(
                    spec=spec,
                    map_path=resolved_path,
                )
                return cls(
                    session_id=session_id,
                    map_path=resolved_path,
                    spec=spec,
                    dirty=False,
                    status=f"Loaded {resolved_display_name or resolved_path.name}",
                )

        normalized_difficulty = normalize_custom_map_difficulty(difficulty or "easy")
        if isinstance(map_file, str) and map_file.strip():
            map_path = resolve_custom_map_path(map_file, difficulty=normalized_difficulty)
        else:
            map_path = build_next_custom_map_path(normalized_difficulty)

        default_width, default_height = _default_dimensions_for_difficulty(normalized_difficulty)
        resolved_width = int(width) if width is not None else default_width
        resolved_height = int(height) if height is not None else default_height
        scenario_id = (
            str(scenario_name).strip()
            if isinstance(scenario_name, str) and scenario_name.strip()
            else map_path.stem
        )
        spec = build_empty_custom_map_spec(
            width=resolved_width,
            height=resolved_height,
            difficulty=normalized_difficulty,
            scenario_name=scenario_id,
            display_name=requested_display_name,
        )
        return cls(
            session_id=session_id,
            map_path=map_path,
            spec=spec,
            dirty=True,
            status=f"Created new {normalized_difficulty} map session",
        )

    def _validate_cell(self, x: int, y: int) -> None:
        if not (0 <= int(x) < int(self.spec["width"]) and 0 <= int(y) < int(self.spec["height"])):
            raise ValueError(f"Cell out of range: {(x, y)}")

    def append_token(
        self,
        *,
        x: int,
        y: int,
        token: str,
        direction: Optional[int] = None,
    ) -> None:
        self._validate_cell(x, y)
        if is_border(self.spec, x, y):
            set_tile(self.spec, x, y, ASCII_BORDER_TOKEN)
            self.status = f"Border fixed at {(x, y)}"
            return
        if token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
            clear_cell_stack(self.spec, x, y)
            self.dirty = True
            self.status = f"Cleared {(x, y)}"
            return
        append_token_to_cell_stack(
            self.spec,
            x,
            y,
            str(token),
            direction=direction,
        )
        self.dirty = True
        stack_depth = len(get_cell_stack_entries(self.spec, x, y))
        token_name = describe_editor_token_name(str(token))
        self.status = f"Added `{token_name}` to {(x, y)}"
        if direction is not None:
            self.status += f" dir={int(direction)}"
        if stack_depth > 1:
            self.status += f" | stack={stack_depth}"

    def pop_token(self, *, x: int, y: int) -> None:
        self._validate_cell(x, y)
        if is_border(self.spec, x, y):
            set_tile(self.spec, x, y, ASCII_BORDER_TOKEN)
            self.status = f"Border fixed at {(x, y)}"
            return
        removed = pop_top_token_from_cell_stack(self.spec, x, y)
        if removed is None:
            self.status = f"Cell {(x, y)} is already empty"
            return
        self.dirty = True
        removed_name = describe_editor_token_name(str(removed.get("token") or ""))
        remaining = len(get_cell_stack_entries(self.spec, x, y))
        self.status = f"Removed top `{removed_name}` from {(x, y)} | stack={remaining}"

    def clear_cell(self, *, x: int, y: int) -> None:
        self._validate_cell(x, y)
        if is_border(self.spec, x, y):
            set_tile(self.spec, x, y, ASCII_BORDER_TOKEN)
            self.status = f"Border fixed at {(x, y)}"
            return
        clear_cell_stack(self.spec, x, y)
        self.dirty = True
        self.status = f"Cleared {(x, y)}"

    def clear_interior(self) -> None:
        for y in range(1, int(self.spec["height"]) - 1):
            for x in range(1, int(self.spec["width"]) - 1):
                clear_cell_stack(self.spec, x, y)
        self.spec["dynamic_directions"] = {}
        self.spec["cell_stacks"] = {}
        self.dirty = True
        self.status = "Cleared interior cells"

    def save(self) -> None:
        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        save_custom_map_spec(self.spec, self.map_path)
        refresh_custom_map_display_name_index()
        self.dirty = False
        display_name = _resolve_editor_display_name(
            spec=self.spec,
            map_path=self.map_path,
        )
        self.status = f"Saved {display_name or self.map_path.name}"

    def to_payload(self) -> Dict[str, Any]:
        display_name = _resolve_editor_display_name(
            spec=self.spec,
            map_path=self.map_path,
        )
        visible_top_left = None
        if int(self.spec.get("width", 0) or 0) > 1 and int(self.spec.get("height", 0) or 0) > 1:
            visible_top_left = get_visible_cell_token(self.spec, 1, 1)
        return {
            "sessionId": self.session_id,
            "mapPath": str(self.map_path),
            "mapFileName": self.map_path.name,
            "difficulty": str(self.spec.get("difficulty") or self.map_path.parent.name),
            "scenarioName": str(self.spec.get("scenario_name") or self.map_path.stem),
            "displayName": display_name,
            "width": int(self.spec.get("width", 0) or 0),
            "height": int(self.spec.get("height", 0) or 0),
            "dirty": bool(self.dirty),
            "status": self.status,
            "visibleTopLeft": visible_top_left,
            "palette": _palette_payload(),
            "scene": build_custom_map_board_scene(self.spec).to_payload(),
        }
