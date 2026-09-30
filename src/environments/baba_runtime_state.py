from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from baba_in_wonderland.grid import BabaIsYouGrid
from baba_in_wonderland.rule import extract_ruleset
from baba_in_wonderland.world_object import (
    RuleAnd,
    RuleColor,
    RuleIs,
    RuleObject,
    RuleProperty,
    Ruleset,
    Wall,
    make_obj,
)
from src.data import RuntimeObjectRow, RuntimeStatePacket, RuntimeStateVocab


_NO_DIRECTION = -1
_DIRECTION_BY_NAME = {
    "facing right": 0,
    "facing down": 1,
    "facing left": 2,
    "facing up": 3,
}


def _vocab_lookup(values: Iterable[str]) -> Dict[str, int]:
    return {str(value): int(index) for index, value in enumerate(values)}


def _direction_name_to_id(value: Any) -> int:
    if not isinstance(value, str):
        return _NO_DIRECTION
    return int(_DIRECTION_BY_NAME.get(value.strip().lower(), _NO_DIRECTION))


def _resolve_vocab(values: Tuple[str, ...], index: int, *, kind: str) -> str:
    resolved_index = int(index)
    if resolved_index < 0 or resolved_index >= len(values):
        raise KeyError(f"Runtime state packet references unknown {kind}_id={resolved_index}.")
    value = str(values[resolved_index]).strip()
    if not value:
        raise KeyError(f"Runtime state packet references empty {kind}_id={resolved_index}.")
    return value


def _raw_grid_size(packet: RuntimeStatePacket) -> Tuple[int, int]:
    return max(3, int(packet.width) + 2), max(3, int(packet.height) + 2)


def _make_grid(existing_grid: Any, width: int, height: int) -> Any:
    if (
        existing_grid is not None
        and int(getattr(existing_grid, "width", 0) or 0) == int(width)
        and int(getattr(existing_grid, "height", 0) or 0) == int(height)
    ):
        grid = existing_grid
        debug = bool(getattr(grid, "debug", False))
    else:
        grid_cls = existing_grid.__class__ if existing_grid is not None else BabaIsYouGrid
        debug = bool(getattr(existing_grid, "debug", False)) if existing_grid is not None else False
        grid = grid_cls(int(width), int(height), debug=debug)
    grid.grid = [[None] for _ in range(int(width) * int(height))]
    grid._occupied_indices = set()
    grid._occupied_indices_cache = None
    return grid


def _append_object(grid: Any, occupied: set[int], obj: Any, x: int, y: int) -> None:
    idx = int(y) * int(grid.width) + int(x)
    stack = grid.grid[idx]
    if not isinstance(stack, list):
        stack = [None]
        grid.grid[idx] = stack
    elif not stack:
        stack.append(None)
    stack.append(obj)
    occupied.add(idx)
    if hasattr(obj, "init_pos"):
        obj.init_pos = (int(x), int(y))
    if hasattr(obj, "cur_pos"):
        obj.cur_pos = (int(x), int(y))


def _append_border_walls(grid: Any, occupied: set[int]) -> None:
    width = int(grid.width)
    height = int(grid.height)
    for x in range(width):
        _append_object(grid, occupied, Wall(), x, 0)
        _append_object(grid, occupied, Wall(), x, height - 1)
    for y in range(1, height - 1):
        _append_object(grid, occupied, Wall(), 0, y)
        _append_object(grid, occupied, Wall(), width - 1, y)


def _make_runtime_object(type_text: str, word_text: str) -> Any:
    if type_text == "world_object":
        return make_obj(word_text)
    if type_text == "rule_noun":
        return RuleObject(word_text)
    if type_text == "rule_operator":
        if word_text == "is":
            return RuleIs()
        if word_text == "and":
            return RuleAnd()
        raise ValueError(f"Unsupported rule operator in runtime state: {word_text}")
    if type_text == "rule_property":
        return RuleProperty(word_text)
    if type_text == "rule_color":
        return RuleColor(word_text)
    return make_obj(word_text)


def _set_ruleset_references(env: Any, grid: Any, ruleset: Any) -> None:
    grid._ruleset = ruleset
    for idx in grid._iter_occupied_indices():
        for obj in grid.grid[idx]:
            if obj is not None:
                obj.set_ruleset(ruleset)
    env._ruleset = ruleset


def restore_runtime_packet(
    wrapper: Any,
    packet: RuntimeStatePacket,
    vocab: RuntimeStateVocab,
    *,
    step_count: int,
) -> None:
    env = wrapper.env
    raw_width, raw_height = _raw_grid_size(packet)
    grid = _make_grid(getattr(env, "grid", None), raw_width, raw_height)
    occupied: set[int] = set()
    _append_border_walls(grid, occupied)

    type_texts = tuple(vocab.type_texts)
    word_texts = tuple(vocab.word_texts)
    reverse_word_aliases = getattr(wrapper, "_runtime_reverse_word_aliases", {})
    for row in packet.objects:
        type_text = _resolve_vocab(type_texts, int(row.type_id), kind="type")
        word_text = _resolve_vocab(word_texts, int(row.word_id), kind="word")
        if reverse_word_aliases:
            type_aliases = reverse_word_aliases.get(type_text)
            if type_aliases is not None:
                word_text = type_aliases.get(word_text, word_text)
        obj = _make_runtime_object(type_text, word_text)
        if int(row.direction) != _NO_DIRECTION and hasattr(obj, "dir"):
            obj.dir = int(row.direction)
        _append_object(grid, occupied, obj, int(row.x) + 1, int(row.y) + 1)

    grid._occupied_indices = occupied
    grid._occupied_indices_cache = None
    if hasattr(grid, "encoding_level") and hasattr(env, "encoding_level"):
        grid.encoding_level = int(getattr(env, "encoding_level", 1) or 1)
    env.grid = grid

    sync_dimensions = getattr(env, "_sync_dimensions", None)
    if callable(sync_dimensions):
        sync_dimensions(int(raw_width), int(raw_height))
    else:
        env.width = int(raw_width)
        env.height = int(raw_height)

    ruleset = Ruleset(extract_ruleset(grid, default_ruleset=getattr(env, "default_ruleset", {})))
    _set_ruleset_references(env, grid, ruleset)

    sync_agent = getattr(env, "_sync_primary_agent_state", None)
    if callable(sync_agent):
        sync_agent()
    env.carrying = None
    env.step_count = int(step_count)
    env.is_win = False
    env.is_defeat = False

    wrapper.grid_size = (int(raw_width), int(raw_height))
    wrapper.current_objects = []
    wrapper.current_state_id = None
    wrapper.last_reward = 0.0
    wrapper.last_event = "restore"
    wrapper.last_terminated = bool(packet.terminated)
    wrapper.last_truncated = False


def capture_runtime_packet(wrapper: Any, vocab: RuntimeStateVocab) -> RuntimeStatePacket:
    env = wrapper.env
    grid = getattr(env, "grid", None)
    raw_width = int(getattr(grid, "width", 0) or 0)
    raw_height = int(getattr(grid, "height", 0) or 0)
    if grid is None or raw_width <= 0 or raw_height <= 0:
        return RuntimeStatePacket(width=0, height=0, terminated=bool(wrapper.last_terminated), objects=())

    type_texts = tuple(vocab.type_texts)
    word_texts = tuple(vocab.word_texts)
    type_lookup = _vocab_lookup(type_texts)
    word_lookup = _vocab_lookup(word_texts)
    rows: list[RuntimeObjectRow] = []
    raw_cells = getattr(grid, "grid", None)
    occupied_indices = (
        tuple(grid._iter_occupied_indices())
        if callable(getattr(grid, "_iter_occupied_indices", None))
        else range(raw_width * raw_height)
    )
    normalize_object_type = wrapper._normalize_object_type
    normalize_schema = wrapper._normalize_state_object_schema
    extract_direction = wrapper._extract_object_direction
    word_aliases = getattr(wrapper, "_runtime_word_aliases", {})

    for idx in occupied_indices:
        x = int(idx % raw_width)
        y = int(idx // raw_width)
        if x == 0 or y == 0 or x == raw_width - 1 or y == raw_height - 1:
            continue
        if isinstance(raw_cells, list) and idx < len(raw_cells):
            stack = raw_cells[idx]
        else:
            stack = wrapper._get_cell_stack(grid=grid, x=x, y=y)
        iterable_stack = stack if isinstance(stack, list) else [stack]
        for obj in iterable_stack:
            if obj is None:
                continue
            raw_type = normalize_object_type(str(getattr(obj, "type", "unknown")))
            obj_type, word = normalize_schema(obj=obj, raw_type=raw_type)
            if word_aliases:
                type_aliases = word_aliases.get(obj_type)
                if type_aliases is not None:
                    word = type_aliases.get(word, word)
            if obj_type not in type_lookup:
                raise KeyError(f"Runtime capture found unknown state type: {obj_type}")
            if word not in word_lookup:
                raise KeyError(f"Runtime capture found unknown state word: {word}")
            rows.append(
                RuntimeObjectRow(
                    x=int(x - 1),
                    y=int(y - 1),
                    type_id=int(type_lookup[obj_type]),
                    word_id=int(word_lookup[word]),
                    direction=int(_direction_name_to_id(extract_direction(obj))),
                )
            )
    return RuntimeStatePacket(
        width=max(0, int(raw_width) - 2),
        height=max(0, int(raw_height) - 2),
        terminated=bool(wrapper.last_terminated),
        objects=tuple(rows),
    )
