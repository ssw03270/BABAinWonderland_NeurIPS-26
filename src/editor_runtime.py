from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.environments.custom_map_spec import (
    ASCII_BORDER_TOKEN,
    ASCII_EMPTY_TOKEN,
    ASCII_OBJECTS,
    ASCII_RULE_OPERATORS,
    ASCII_RULE_OBJECTS,
    ASCII_RULE_PROPERTIES,
    DEFAULT_EDITOR_PALETTE,
    EDITOR_PALETTE_LABELS,
    get_custom_map_storage_root,
    normalize_custom_map_difficulty,
)


PALETTE_COLUMNS = 4
MAX_PALETTE_COLUMNS = 9
PALETTE_GAP_PX = 8
DIRECTION_COLUMNS = 3
DIRECTION_GAP_PX = 8
DIRECTION_BUTTON_WIDTH = 72
DIRECTION_BUTTON_HEIGHT = 40
BASE_CELL_PX = 44
MIN_CELL_PX = 12
BASE_FONT_SIZE = 20
BASE_SMALL_FONT_SIZE = 15
BASE_MARGIN_PX = 16
BASE_TITLE_GAP_PX = 36
BASE_SECTION_GAP_PX = 24
BASE_INFO_LINE_HEIGHT = 20
MIN_PALETTE_BUTTON_WIDTH = 88
MIN_PALETTE_BUTTON_HEIGHT = 28
MIN_DIRECTION_BUTTON_WIDTH = 56
MIN_DIRECTION_BUTTON_HEIGHT = 28
MIN_WINDOW_WIDTH = 640
MIN_WINDOW_HEIGHT = 480
DISPLAY_PADDING_PX = 80
FALLBACK_DISPLAY_WIDTH = 1600
FALLBACK_DISPLAY_HEIGHT = 960
MIN_AXIS_LEFT_WIDTH = 24
MIN_AXIS_BOTTOM_HEIGHT = 22
AXIS_TICK_LENGTH_PX = 5
SPRITE_PREVIEW_PADDING_PX = 6


def build_next_custom_map_path(difficulty: str) -> Path:
    normalized = normalize_custom_map_difficulty(difficulty)
    root = get_custom_map_storage_root(normalized)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / f"custom_map_{index:02d}.json"
        if not candidate.exists():
            return candidate
        index += 1


def set_tile(spec: Dict, x: int, y: int, token: str) -> None:
    row = list(spec["layout"][y])
    row[x] = token
    spec["layout"][y] = "".join(row)


def get_tile(spec: Dict, x: int, y: int) -> str:
    return str(spec["layout"][y][x])


def _build_stack_entry(token: str, direction: Optional[int] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"token": str(token)}
    if token in ASCII_OBJECTS and direction in {0, 1, 2, 3}:
        entry["direction"] = int(direction)
    return entry


def get_cell_stack_entries(spec: Dict, x: int, y: int) -> List[Dict[str, Any]]:
    base_token = get_tile(spec, x, y)
    if base_token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
        return []

    dynamic_directions = spec.get("dynamic_directions", {})
    entries = [
        _build_stack_entry(
            base_token,
            dynamic_directions.get((x, y)),
        )
    ]
    raw_stacks = spec.get("cell_stacks", {})
    if isinstance(raw_stacks, dict):
        for raw_entry in raw_stacks.get((x, y), []):
            if not isinstance(raw_entry, dict):
                continue
            token = str(raw_entry.get("token") or "").strip()
            if not token:
                continue
            direction = raw_entry.get("direction")
            direction = int(direction) if direction in {0, 1, 2, 3} else None
            entries.append(_build_stack_entry(token, direction))
    return entries


def set_cell_stack_entries(spec: Dict, x: int, y: int, entries: List[Dict[str, Any]]) -> None:
    spec.setdefault("dynamic_directions", {})
    spec.setdefault("cell_stacks", {})

    cleaned: List[Dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        token = str(raw_entry.get("token") or "").strip()
        if token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
            continue
        if token not in ASCII_OBJECTS and token not in ASCII_RULE_OPERATORS and token not in ASCII_RULE_OBJECTS and token not in ASCII_RULE_PROPERTIES:
            continue
        cleaned.append(_build_stack_entry(token, raw_entry.get("direction")))

    if not cleaned:
        set_tile(spec, x, y, ASCII_EMPTY_TOKEN)
        spec["dynamic_directions"].pop((x, y), None)
        spec["cell_stacks"].pop((x, y), None)
        return

    base_entry = cleaned[0]
    set_tile(spec, x, y, str(base_entry["token"]))
    if "direction" in base_entry:
        spec["dynamic_directions"][(x, y)] = int(base_entry["direction"])
    else:
        spec["dynamic_directions"].pop((x, y), None)

    extra_entries = cleaned[1:]
    if extra_entries:
        spec["cell_stacks"][(x, y)] = extra_entries
    else:
        spec["cell_stacks"].pop((x, y), None)


def append_token_to_cell_stack(
    spec: Dict,
    x: int,
    y: int,
    token: str,
    *,
    direction: Optional[int] = None,
) -> None:
    entries = get_cell_stack_entries(spec, x, y)
    entries.append(_build_stack_entry(token, direction))
    set_cell_stack_entries(spec, x, y, entries)


def pop_top_token_from_cell_stack(spec: Dict, x: int, y: int) -> Optional[Dict[str, Any]]:
    entries = get_cell_stack_entries(spec, x, y)
    if not entries:
        return None
    removed = entries.pop()
    set_cell_stack_entries(spec, x, y, entries)
    return removed


def clear_cell_stack(spec: Dict, x: int, y: int) -> None:
    set_cell_stack_entries(spec, x, y, [])


def get_visible_cell_token(spec: Dict, x: int, y: int) -> str:
    token = get_tile(spec, x, y)
    if token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
        return token
    entries = get_cell_stack_entries(spec, x, y)
    if not entries:
        return token
    return str(entries[-1]["token"])


def get_visible_cell_direction(spec: Dict, x: int, y: int) -> Optional[int]:
    entries = get_cell_stack_entries(spec, x, y)
    if not entries:
        return None
    direction = entries[-1].get("direction")
    return int(direction) if direction in {0, 1, 2, 3} else None


def is_border(spec: Dict, x: int, y: int) -> bool:
    return x in {0, spec["width"] - 1} or y in {0, spec["height"] - 1}


def classify_editor_token(token: str) -> str:
    if token == ASCII_BORDER_TOKEN:
        return "border"
    if token == ASCII_EMPTY_TOKEN:
        return "empty"
    if token in ASCII_OBJECTS:
        return "world"
    if token in ASCII_RULE_OPERATORS or token in ASCII_RULE_OBJECTS or token in ASCII_RULE_PROPERTIES:
        return "rule"
    return "unknown"


def describe_editor_token_for_sprite(token: str) -> Optional[Dict[str, str]]:
    if token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
        return None
    if token in ASCII_OBJECTS:
        return {"kind": "world", "word": ASCII_OBJECTS[token]}
    if token in ASCII_RULE_OPERATORS:
        return {"kind": "rule", "word": ASCII_RULE_OPERATORS[token], "obj_type": "rule_operator"}
    if token in ASCII_RULE_OBJECTS:
        return {
            "kind": "rule",
            "word": ASCII_RULE_OBJECTS[token],
            "obj_type": "rule_noun",
        }
    if token in ASCII_RULE_PROPERTIES:
        return {
            "kind": "rule",
            "word": ASCII_RULE_PROPERTIES[token],
            "obj_type": "rule_property",
        }
    return None


def describe_editor_token_name(token: str) -> str:
    if token == ASCII_EMPTY_TOKEN:
        return "empty"
    if token == ASCII_BORDER_TOKEN:
        return "border"
    if token in ASCII_OBJECTS:
        return ASCII_OBJECTS[token]
    if token in ASCII_RULE_OPERATORS:
        return ASCII_RULE_OPERATORS[token]
    if token in ASCII_RULE_OBJECTS:
        return ASCII_RULE_OBJECTS[token]
    if token in ASCII_RULE_PROPERTIES:
        return ASCII_RULE_PROPERTIES[token]
    return token


def compute_grid_rows(item_count: int, columns: int) -> int:
    if item_count <= 0:
        return 0
    return ((item_count - 1) // columns) + 1


def scale_dimension(base_value: int, cell_px: int, *, min_value: int) -> int:
    scaled = round(base_value * float(cell_px) / float(BASE_CELL_PX))
    return max(min_value, int(scaled))


def get_palette_button_dimensions(cell_px: int) -> tuple[int, int]:
    width = max(
        scale_dimension(120, cell_px, min_value=MIN_PALETTE_BUTTON_WIDTH),
        cell_px * 2 + 16,
    )
    height = max(
        scale_dimension(40, cell_px, min_value=MIN_PALETTE_BUTTON_HEIGHT),
        cell_px - 4,
    )
    return width, height


def get_direction_button_dimensions(cell_px: int) -> tuple[int, int]:
    width = scale_dimension(
        DIRECTION_BUTTON_WIDTH,
        cell_px,
        min_value=MIN_DIRECTION_BUTTON_WIDTH,
    )
    height = scale_dimension(
        DIRECTION_BUTTON_HEIGHT,
        cell_px,
        min_value=MIN_DIRECTION_BUTTON_HEIGHT,
    )
    return width, height


def estimate_axis_left_width(map_height: int, small_font_size: int, cell_px: int) -> int:
    digits = max(1, len(str(max(0, int(map_height) - 1))))
    estimated_width = int(round(digits * small_font_size * 0.7)) + 12
    return max(
        MIN_AXIS_LEFT_WIDTH,
        estimated_width,
        scale_dimension(28, cell_px, min_value=MIN_AXIS_LEFT_WIDTH),
    )


def estimate_axis_bottom_height(small_font_size: int, cell_px: int) -> int:
    estimated_height = small_font_size + 10
    return max(
        MIN_AXIS_BOTTOM_HEIGHT,
        estimated_height,
        scale_dimension(26, cell_px, min_value=MIN_AXIS_BOTTOM_HEIGHT),
    )


def compute_axis_label_step(
    *,
    cell_px: int,
    font_size: int,
    max_index: int,
    axis: str,
) -> int:
    if axis == "x":
        digits = max(1, len(str(max(0, int(max_index)))))
        label_span_px = int(round(digits * font_size * 0.7)) + 8
    else:
        label_span_px = font_size + 6
    return max(1, int(math.ceil(label_span_px / max(1, int(cell_px)))))


def build_axis_label_indices(length: int, step: int) -> List[int]:
    if length <= 0:
        return []
    resolved_step = max(1, int(step))
    indices = list(range(0, int(length), resolved_step))
    last_index = int(length) - 1
    if last_index not in indices:
        indices.append(last_index)
    return indices


def make_rect(x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
    return (int(x), int(y), int(width), int(height))


def point_in_rect(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    px, py = point
    x, y, width, height = rect
    return x <= px < x + width and y <= py < y + height


def build_palette_button_specs(
    *,
    panel_left: int,
    panel_top: int,
    cell_px: int,
    columns: int = PALETTE_COLUMNS,
    gap_px: int = PALETTE_GAP_PX,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    button_width, button_height = get_palette_button_dimensions(cell_px)
    for index, token in enumerate(DEFAULT_EDITOR_PALETTE):
        row = index // columns
        col = index % columns
        x = panel_left + col * (button_width + gap_px)
        y = panel_top + row * (button_height + gap_px)
        specs.append(
            {
                "token": token,
                "label": EDITOR_PALETTE_LABELS.get(token, token),
                "rect": make_rect(x, y, button_width, button_height),
            }
        )
    return specs


def measure_palette_panel_height(
    *,
    item_count: int,
    cell_px: int,
    columns: int = PALETTE_COLUMNS,
    gap_px: int = PALETTE_GAP_PX,
) -> int:
    _, button_height = get_palette_button_dimensions(cell_px)
    rows = compute_grid_rows(item_count, columns)
    if rows <= 0:
        return 0
    return rows * button_height + max(0, rows - 1) * gap_px


def build_direction_button_specs(
    *,
    panel_left: int,
    panel_top: int,
    button_width: int | None = None,
    button_height: int | None = None,
    cell_px: int = BASE_CELL_PX,
    gap_px: int = DIRECTION_GAP_PX,
    columns: int = DIRECTION_COLUMNS,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    if button_width is None or button_height is None:
        button_width, button_height = get_direction_button_dimensions(cell_px)
    labels = [
        (">", "set_0"),
        ("^", "set_1"),
        ("<", "set_2"),
        ("v", "set_3"),
        ("none", "clear"),
    ]
    for index, (label, action) in enumerate(labels):
        row = index // columns
        col = index % columns
        x = panel_left + col * (button_width + gap_px)
        y = panel_top + row * (button_height + gap_px)
        specs.append(
            {
                "label": label,
                "action": action,
                "rect": make_rect(x, y, button_width, button_height),
            }
        )
    return specs


def measure_direction_panel_height(
    *,
    item_count: int,
    button_height: int = DIRECTION_BUTTON_HEIGHT,
    columns: int = DIRECTION_COLUMNS,
    gap_px: int = DIRECTION_GAP_PX,
) -> int:
    rows = compute_grid_rows(item_count, columns)
    if rows <= 0:
        return 0
    return rows * button_height + max(0, rows - 1) * gap_px


def find_palette_token_at(
    point: Tuple[int, int], button_specs: List[Dict[str, object]]
) -> Optional[str]:
    for spec in button_specs:
        rect = spec.get("rect")
        token = spec.get("token")
        if (
            isinstance(rect, tuple)
            and len(rect) == 4
            and isinstance(token, str)
            and point_in_rect(point, rect)
        ):
            return token
    return None


def find_direction_action_at(
    point: Tuple[int, int], button_specs: List[Dict[str, object]]
) -> Optional[str]:
    for spec in button_specs:
        rect = spec.get("rect")
        action = spec.get("action")
        if (
            isinstance(rect, tuple)
            and len(rect) == 4
            and isinstance(action, str)
            and point_in_rect(point, rect)
        ):
            return action
    return None


def measure_panel_width(
    *,
    item_width: int,
    columns: int,
    gap_px: int,
) -> int:
    if columns <= 0:
        return 0
    return columns * item_width + max(0, columns - 1) * gap_px


def compute_editor_layout(
    *,
    map_width: int,
    map_height: int,
    max_window_width: Optional[int] = None,
    max_window_height: Optional[int] = None,
) -> Dict[str, int]:
    resolved_max_width = int(max_window_width or FALLBACK_DISPLAY_WIDTH)
    resolved_max_height = int(max_window_height or FALLBACK_DISPLAY_HEIGHT)
    best_layout: Optional[Dict[str, int]] = None
    best_score: Optional[tuple[int, int, int]] = None

    for cell_px in range(BASE_CELL_PX, MIN_CELL_PX - 1, -1):
        font_size = scale_dimension(BASE_FONT_SIZE, cell_px, min_value=10)
        small_font_size = scale_dimension(BASE_SMALL_FONT_SIZE, cell_px, min_value=8)
        margin_px = scale_dimension(BASE_MARGIN_PX, cell_px, min_value=8)
        title_gap_px = scale_dimension(BASE_TITLE_GAP_PX, cell_px, min_value=20)
        section_gap_px = scale_dimension(BASE_SECTION_GAP_PX, cell_px, min_value=12)
        info_line_height = scale_dimension(BASE_INFO_LINE_HEIGHT, cell_px, min_value=14)
        axis_left_width = estimate_axis_left_width(map_height, small_font_size, cell_px)
        axis_bottom_height = estimate_axis_bottom_height(small_font_size, cell_px)
        grid_width_px = int(map_width) * cell_px
        grid_height_px = int(map_height) * cell_px
        direction_button_width, direction_button_height = get_direction_button_dimensions(cell_px)
        direction_height = measure_direction_panel_height(
            item_count=5,
            button_height=direction_button_height,
        )

        max_columns_for_scale = min(MAX_PALETTE_COLUMNS, len(DEFAULT_EDITOR_PALETTE))
        for palette_columns in range(PALETTE_COLUMNS, max_columns_for_scale + 1):
            palette_button_width, _palette_button_height = get_palette_button_dimensions(cell_px)
            sidebar_width = measure_panel_width(
                item_width=palette_button_width,
                columns=palette_columns,
                gap_px=PALETTE_GAP_PX,
            )
            palette_height = measure_palette_panel_height(
                item_count=len(DEFAULT_EDITOR_PALETTE),
                cell_px=cell_px,
                columns=palette_columns,
            )
            info_top = (
                margin_px
                + title_gap_px
                + palette_height
                + section_gap_px
                + title_gap_px
                + direction_height
                + section_gap_px
            )
            sidebar_px = sidebar_width
            window_width_px = (
                margin_px * 3
                + sidebar_px
                + axis_left_width
                + grid_width_px
            )
            window_height_px = max(
                margin_px * 2 + grid_height_px + axis_bottom_height,
                info_top + info_line_height * 6 + margin_px,
            )

            layout = {
                "cell_px": cell_px,
                "font_size": font_size,
                "small_font_size": small_font_size,
                "margin_px": margin_px,
                "title_gap_px": title_gap_px,
                "section_gap_px": section_gap_px,
                "info_line_height": info_line_height,
                "axis_left_width": axis_left_width,
                "axis_bottom_height": axis_bottom_height,
                "palette_columns": palette_columns,
                "palette_button_width": palette_button_width,
                "direction_button_width": direction_button_width,
                "direction_button_height": direction_button_height,
                "sidebar_px": sidebar_px,
                "grid_width_px": grid_width_px,
                "grid_height_px": grid_height_px,
                "palette_height": palette_height,
                "direction_height": direction_height,
                "info_top": info_top,
                "window_width_px": window_width_px,
                "window_height_px": window_height_px,
            }
            overflow = max(0, window_width_px - resolved_max_width) + max(
                0,
                window_height_px - resolved_max_height,
            )
            score = (overflow, -cell_px, palette_columns)
            if best_score is None or score < best_score:
                best_layout = layout
                best_score = score
            if overflow == 0:
                best_layout = layout
                best_score = score
                break

        if best_score is not None and best_score[0] == 0:
            break

    assert best_layout is not None
    best_layout["sidebar_x"] = best_layout["margin_px"]
    best_layout["axis_left_x"] = best_layout["sidebar_px"] + best_layout["margin_px"] * 2
    best_layout["grid_origin_x"] = best_layout["axis_left_x"] + best_layout["axis_left_width"]
    best_layout["grid_origin_y"] = best_layout["margin_px"]
    best_layout["palette_title_y"] = best_layout["margin_px"]
    best_layout["palette_top"] = best_layout["palette_title_y"] + best_layout["title_gap_px"]
    best_layout["direction_title_y"] = (
        best_layout["palette_top"]
        + best_layout["palette_height"]
        + best_layout["section_gap_px"]
    )
    best_layout["direction_top"] = (
        best_layout["direction_title_y"] + best_layout["title_gap_px"]
    )
    best_layout["axis_bottom_y"] = (
        best_layout["grid_origin_y"] + best_layout["grid_height_px"]
    )
    return best_layout
