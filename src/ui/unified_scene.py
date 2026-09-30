"""Shared board-scene helpers for image and web renderers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.environments.custom_map_spec import (
    ASCII_BORDER_TOKEN,
    ASCII_EMPTY_TOKEN,
    ASCII_OBJECTS,
    ASCII_RULE_OBJECTS,
    ASCII_RULE_OPERATORS,
    ASCII_RULE_PROPERTIES,
)
from src.visualization import canonicalize_state_for_visualization


RULE_OBJECT_TYPES = {"rule_noun", "rule_operator", "rule_property"}
RULE_END_TYPES = {"rule_noun", "rule_property"}
HAZARD_PROPERTIES = {"defeat", "hot", "sink", "melt"}
MAP_DIRECTION_NAMES = {
    0: "facing right",
    1: "facing up",
    2: "facing left",
    3: "facing down",
}


@lru_cache(maxsize=1)
def _sprite_manifest() -> Dict[str, set[str]]:
    root = Path(__file__).resolve().parents[2] / "assets" / "babagui_sprites"
    manifest: Dict[str, set[str]] = {"icon": set(), "text": set()}
    for group in tuple(manifest.keys()):
        directory = root / group
        if not directory.exists():
            continue
        manifest[group] = {path.stem.strip().lower() for path in directory.glob("*.gif")}
    return manifest


def normalize_word(word: Any) -> str:
    if not isinstance(word, str):
        return "unknown"
    value = word.strip().lower()
    return value or "unknown"


def extract_grid_size(state: Dict[str, Any]) -> Tuple[int, int]:
    raw = state.get("grid_size")
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and isinstance(raw[0], int)
        and isinstance(raw[1], int)
    ):
        return int(raw[0]), int(raw[1])

    max_x = -1
    max_y = -1
    for obj in state.get("objects", []):
        if not isinstance(obj, dict):
            continue
        pos = obj.get("position")
        if (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and isinstance(pos[0], int)
            and isinstance(pos[1], int)
        ):
            max_x = max(max_x, int(pos[0]))
            max_y = max(max_y, int(pos[1]))

    agent = state.get("agent")
    if isinstance(agent, dict):
        pos = agent.get("position")
        if (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and isinstance(pos[0], int)
            and isinstance(pos[1], int)
        ):
            max_x = max(max_x, int(pos[0]))
            max_y = max(max_y, int(pos[1]))

    return max_x + 1, max_y + 1


def extract_rule_triples(state: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    by_pos: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
    for obj in state.get("objects", []):
        if not isinstance(obj, dict):
            continue
        obj_type = normalize_word(obj.get("type"))
        if obj_type not in RULE_OBJECT_TYPES:
            continue
        pos = obj.get("position")
        if not (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and isinstance(pos[0], int)
            and isinstance(pos[1], int)
        ):
            continue
        word = normalize_word(obj.get("word") or obj.get("text"))
        by_pos[(int(pos[0]), int(pos[1]))].append((obj_type, word))

    def collect_subject_segment(
        start: Tuple[int, int],
        delta: Tuple[int, int],
    ) -> List[Tuple[str, str]]:
        x, y = int(start[0]), int(start[1])
        dx, dy = int(delta[0]), int(delta[1])
        tokens: List[Tuple[str, str]] = []
        seen_noun = False
        expect_noun = True
        while True:
            options = by_pos.get((x, y), [])
            token = next(
                (
                    (obj_type, word)
                    for obj_type, word in options
                    if obj_type in {"rule_noun", "rule_property", "rule_operator"}
                ),
                None,
            )
            if token is None:
                break
            obj_type, word = token
            if expect_noun:
                if obj_type == "rule_operator" and word == "and" and not seen_noun:
                    tokens.append(token)
                elif obj_type == "rule_noun":
                    tokens.append(token)
                    seen_noun = True
                    expect_noun = False
                else:
                    break
            else:
                if obj_type != "rule_operator" or word != "and":
                    break
                tokens.append(token)
                expect_noun = True
            x += dx
            y += dy
        tokens.reverse()
        return tokens

    def collect_predicate_segment(
        start: Tuple[int, int],
        delta: Tuple[int, int],
    ) -> List[Tuple[str, str]]:
        x, y = int(start[0]), int(start[1])
        dx, dy = int(delta[0]), int(delta[1])
        tokens: List[Tuple[str, str]] = []
        seen_item = False
        expect_item = True
        while True:
            options = by_pos.get((x, y), [])
            token = next(
                (
                    (obj_type, word)
                    for obj_type, word in options
                    if obj_type in {"rule_noun", "rule_property", "rule_operator"}
                ),
                None,
            )
            if token is None:
                break
            obj_type, word = token
            if expect_item:
                if obj_type == "rule_operator" and word == "and" and not seen_item:
                    tokens.append(token)
                elif obj_type in {"rule_noun", "rule_property"}:
                    tokens.append(token)
                    seen_item = True
                    expect_item = False
                else:
                    break
            else:
                if obj_type != "rule_operator" or word != "and":
                    break
                tokens.append(token)
                expect_item = True
            x += dx
            y += dy
        return tokens

    def parse_conjoined_words(
        tokens: Sequence[Tuple[str, str]],
        allowed_item_types: Sequence[str],
    ) -> Optional[Tuple[str, ...]]:
        if len(tokens) == 0:
            return None
        trimmed = list(tokens)
        while trimmed and trimmed[0] == ("rule_operator", "and"):
            trimmed = trimmed[1:]
        while trimmed and trimmed[-1] == ("rule_operator", "and"):
            trimmed = trimmed[:-1]
        if len(trimmed) == 0:
            return None
        allowed = {normalize_word(item) for item in allowed_item_types}
        words: List[str] = []
        expect_item = True
        for obj_type, word in trimmed:
            if expect_item:
                if obj_type not in allowed:
                    return None
                words.append(word)
            else:
                if obj_type != "rule_operator" or word != "and":
                    return None
            expect_item = not expect_item
        if expect_item:
            return None
        return tuple(words)

    triples: List[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for (x, y), operators in by_pos.items():
        if not any(obj_type == "rule_operator" and word == "is" for obj_type, word in operators):
            continue
        for dx, dy in ((1, 0), (0, 1)):
            subjects = parse_conjoined_words(
                collect_subject_segment((x - dx, y - dy), (-dx, -dy)),
                ("rule_noun",),
            )
            if not subjects:
                continue
            predicate_tokens = collect_predicate_segment((x + dx, y + dy), (dx, dy))
            if len(predicate_tokens) == 0:
                continue
            predicates = parse_conjoined_words(predicate_tokens, RULE_END_TYPES)
            if not predicates:
                continue
            for subject in subjects:
                for predicate in predicates:
                    triple = (subject, "is", predicate)
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)
    return triples


def extract_active_property_map(state: Dict[str, Any]) -> Dict[str, set[str]]:
    active: Dict[str, set[str]] = defaultdict(set)
    for lhs, _, rhs in extract_rule_triples(state):
        active[lhs].add(rhs)
    return active


def determine_highlight_kind(properties: set[str]) -> Optional[str]:
    if "you" in properties:
        return "you"
    if "win" in properties:
        return "win"
    if properties.intersection(HAZARD_PROPERTIES):
        return "danger"
    return None


def _normalize_direction(direction: Any) -> Optional[str]:
    if isinstance(direction, str):
        normalized = direction.strip().lower()
        return normalized or None
    if isinstance(direction, int):
        return MAP_DIRECTION_NAMES.get(int(direction))
    return None


def _normalize_position_key(raw_key: Any) -> Optional[Tuple[int, int]]:
    if (
        isinstance(raw_key, tuple)
        and len(raw_key) == 2
        and isinstance(raw_key[0], int)
        and isinstance(raw_key[1], int)
    ):
        return int(raw_key[0]), int(raw_key[1])
    if (
        isinstance(raw_key, list)
        and len(raw_key) == 2
        and isinstance(raw_key[0], int)
        and isinstance(raw_key[1], int)
    ):
        return int(raw_key[0]), int(raw_key[1])
    if isinstance(raw_key, str):
        parts = [part.strip() for part in raw_key.split(",")]
        if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
            return int(parts[0]), int(parts[1])
    return None


@dataclass(frozen=True)
class SceneObject:
    kind: str
    word: str
    obj_type: str
    sprite_key: Optional[str] = None
    direction: Optional[str] = None
    highlight_kind: Optional[str] = None

    def resolved_sprite_key(self) -> str:
        return normalize_word(self.sprite_key or self.word)

    def sprite_path(self) -> str:
        group = "text" if self.kind == "rule" else "icon"
        return f"/assets/babagui_sprites/{group}/{self.resolved_sprite_key().upper()}.gif"

    def sprite_available(self) -> bool:
        group = "text" if self.kind == "rule" else "icon"
        return self.resolved_sprite_key() in _sprite_manifest().get(group, set())

    def to_payload(self) -> Dict[str, Any]:
        sprite_available = self.sprite_available()
        payload: Dict[str, Any] = {
            "kind": self.kind,
            "word": self.word,
            "objType": self.obj_type,
            "spriteKey": self.resolved_sprite_key(),
            "spritePath": self.sprite_path() if sprite_available else None,
            "spriteAvailable": bool(sprite_available),
        }
        if self.direction:
            payload["direction"] = self.direction
        if self.highlight_kind:
            payload["highlightKind"] = self.highlight_kind
        return payload


@dataclass(frozen=True)
class SceneCell:
    x: int
    y: int
    objects: Tuple[SceneObject, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "stackCount": len(self.objects),
            "objects": [obj.to_payload() for obj in self.objects],
        }


@dataclass(frozen=True)
class BoardScene:
    width: int
    height: int
    cells: Tuple[SceneCell, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "gridSize": [int(self.width), int(self.height)],
            "cells": [cell.to_payload() for cell in self.cells],
        }


def _build_scene_object_from_state(
    obj: Dict[str, Any],
    *,
    property_map: Mapping[str, set[str]],
) -> Optional[SceneObject]:
    obj_type = normalize_word(obj.get("type"))
    word = normalize_word(obj.get("word") or obj.get("text"))
    direction = _normalize_direction(obj.get("direction"))
    if obj_type in RULE_OBJECT_TYPES:
        return SceneObject(
            kind="rule",
            word=word,
            obj_type=obj_type,
            sprite_key=word,
            direction=direction,
        )
    return SceneObject(
        kind="world",
        word=word,
        obj_type=obj_type or "world_object",
        sprite_key=word,
        direction=direction,
        highlight_kind=determine_highlight_kind(property_map.get(word, set())),
    )


def _scene_object_count_key(scene_obj: SceneObject) -> Tuple[str, str, str]:
    return (
        str(scene_obj.kind),
        str(scene_obj.obj_type),
        str(scene_obj.word),
    )


def _order_cell_objects_by_map_rarity(
    cells_by_position: Mapping[Tuple[int, int], Sequence[SceneObject]],
) -> Tuple[SceneCell, ...]:
    count_by_object = Counter(
        _scene_object_count_key(scene_obj)
        for objects_in_cell in cells_by_position.values()
        for scene_obj in objects_in_cell
    )
    ordered_cells: List[SceneCell] = []
    for (x, y), objects_in_cell in sorted(
        cells_by_position.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        resolved_objects = sorted(
            list(objects_in_cell),
            key=lambda scene_obj: -int(count_by_object[_scene_object_count_key(scene_obj)]),
        )
        if resolved_objects:
            ordered_cells.append(
                SceneCell(x=int(x), y=int(y), objects=tuple(resolved_objects))
            )
    return tuple(ordered_cells)


def build_state_board_scene(
    state: Dict[str, Any],
    *,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> BoardScene:
    visual_state = canonicalize_state_for_visualization(state, visual_config=visual_config)
    width, height = extract_grid_size(visual_state)
    property_map = extract_active_property_map(visual_state)
    by_cell: Dict[Tuple[int, int], List[SceneObject]] = defaultdict(list)
    objects = visual_state.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            pos = obj.get("position")
            if not (
                isinstance(pos, (list, tuple))
                and len(pos) == 2
                and isinstance(pos[0], int)
                and isinstance(pos[1], int)
            ):
                continue
            x = int(pos[0])
            y = int(pos[1])
            if not (0 <= x < width and 0 <= y < height):
                continue
            scene_obj = _build_scene_object_from_state(obj, property_map=property_map)
            if scene_obj is None:
                continue
            by_cell[(x, y)].append(scene_obj)

    cells = _order_cell_objects_by_map_rarity(by_cell)
    return BoardScene(width=max(0, int(width)), height=max(0, int(height)), cells=cells)


def _token_to_scene_object(token: str, *, direction: Optional[str]) -> Optional[SceneObject]:
    if token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
        return None
    if token in ASCII_OBJECTS:
        return SceneObject(
            kind="world",
            word=ASCII_OBJECTS[token],
            obj_type="world_object",
            sprite_key=ASCII_OBJECTS[token],
            direction=direction,
        )
    if token in ASCII_RULE_OPERATORS:
        return SceneObject(
            kind="rule",
            word=ASCII_RULE_OPERATORS[token],
            obj_type="rule_operator",
            sprite_key=ASCII_RULE_OPERATORS[token],
            direction=direction,
        )
    if token in ASCII_RULE_OBJECTS:
        return SceneObject(
            kind="rule",
            word=ASCII_RULE_OBJECTS[token],
            obj_type="rule_noun",
            sprite_key=ASCII_RULE_OBJECTS[token],
            direction=direction,
        )
    if token in ASCII_RULE_PROPERTIES:
        return SceneObject(
            kind="rule",
            word=ASCII_RULE_PROPERTIES[token],
            obj_type="rule_property",
            sprite_key=ASCII_RULE_PROPERTIES[token],
            direction=direction,
        )
    return None


def build_custom_map_board_scene(spec: Mapping[str, Any]) -> BoardScene:
    width = int(spec.get("width", 0) or 0)
    height = int(spec.get("height", 0) or 0)
    layout = spec.get("layout")
    if not isinstance(layout, list):
        return BoardScene(width=width, height=height, cells=())

    dynamic_directions_raw = spec.get("dynamic_directions") or {}
    cell_stacks_raw = spec.get("cell_stacks") or {}
    dynamic_directions: Dict[Tuple[int, int], str] = {}
    cell_stacks: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}

    if isinstance(dynamic_directions_raw, Mapping):
        for raw_key, raw_direction in dynamic_directions_raw.items():
            position = _normalize_position_key(raw_key)
            direction = _normalize_direction(raw_direction)
            if position is not None and direction is not None:
                dynamic_directions[position] = direction

    if isinstance(cell_stacks_raw, Mapping):
        for raw_key, raw_entries in cell_stacks_raw.items():
            position = _normalize_position_key(raw_key)
            if position is None or not isinstance(raw_entries, list):
                continue
            normalized_entries: List[Mapping[str, Any]] = []
            for raw_entry in raw_entries:
                if isinstance(raw_entry, Mapping):
                    normalized_entries.append(raw_entry)
            if normalized_entries:
                cell_stacks[position] = normalized_entries

    by_cell: Dict[Tuple[int, int], List[SceneObject]] = defaultdict(list)
    for y, row in enumerate(layout):
        if not isinstance(row, str):
            continue
        for x, token in enumerate(row):
            if token in {ASCII_EMPTY_TOKEN, ASCII_BORDER_TOKEN}:
                continue
            base_direction = dynamic_directions.get((x, y))
            base_obj = _token_to_scene_object(token, direction=base_direction)
            if base_obj is not None:
                by_cell[(x, y)].append(base_obj)
            for raw_entry in cell_stacks.get((x, y), []):
                entry_token = str(raw_entry.get("token") or "").strip()
                entry_direction = _normalize_direction(raw_entry.get("direction"))
                scene_obj = _token_to_scene_object(entry_token, direction=entry_direction)
                if scene_obj is not None:
                    by_cell[(x, y)].append(scene_obj)
    cells = _order_cell_objects_by_map_rarity(by_cell)
    return BoardScene(width=width, height=height, cells=tuple(cells))
