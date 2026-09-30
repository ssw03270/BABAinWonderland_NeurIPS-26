"""Baba in Wonderland environment wrapper."""

from __future__ import annotations

import copy
import importlib
from numbers import Integral
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .state_serializer import StateSerializer
from src.data import StateStore, canonical_state_key
from src.data.state_schema import build_state_payload
from src.visualization import build_visualization_config, normalize_visual_object_type
from src.web.map_display_names import resolve_map_display_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baba_in_wonderland  # type: ignore
from baba_in_wonderland.world_object import Ruleset as BabaRuleset, WorldObj as BabaWorldObj  # type: ignore

_SIMPLE_BABA_CLONE_TYPES = (type(None), bool, int, float, str, tuple)


def _clone_plain_structure(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: _clone_plain_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_plain_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_plain_structure(item) for item in value)
    if isinstance(value, set):
        return {_clone_plain_structure(item) for item in value}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def _clone_baba_object(
    obj: Any,
    *,
    ruleset: Any = None,
    memo: Optional[Dict[int, Any]] = None,
) -> Any:
    if obj is None or BabaWorldObj is None or not isinstance(obj, BabaWorldObj):
        return _clone_plain_structure(obj)

    if memo is None:
        memo = {}
    cached = memo.get(id(obj))
    if cached is not None:
        return cached

    cloned = obj.__class__.__new__(obj.__class__)
    memo[id(obj)] = cloned

    cloned_state = dict(obj.__dict__)
    cloned_state.pop("_ruleset", None)
    fast_path_compatible = True
    for key, value in cloned_state.items():
        if key == "img" and isinstance(value, np.ndarray):
            continue
        if isinstance(value, _SIMPLE_BABA_CLONE_TYPES):
            continue
        fast_path_compatible = False
        break
    if fast_path_compatible:
        cloned.__dict__.update(cloned_state)
        if ruleset is not None:
            cloned.__dict__["_ruleset"] = ruleset
        return cloned

    for key, value in tuple(cloned_state.items()):
        if key == "img" and isinstance(value, np.ndarray):
            # Render bitmaps are immutable for our search usage; sharing avoids a huge copy cost.
            continue
        if BabaWorldObj is not None and isinstance(value, BabaWorldObj):
            cloned_state[key] = _clone_baba_object(value, ruleset=ruleset, memo=memo)
            continue
        if isinstance(value, np.ndarray):
            cloned_state[key] = np.array(value, copy=True)
            continue
        if isinstance(value, (dict, list, set)):
            cloned_state[key] = _clone_plain_structure(value)
            continue
    cloned.__dict__.update(cloned_state)

    if ruleset is not None:
        cloned.__dict__["_ruleset"] = ruleset
    return cloned


def _clone_baba_grid(
    grid: Any,
    *,
    ruleset: Any = None,
) -> Any:
    if grid is None:
        return None

    try:
        cloned_grid = grid.__class__(
            int(getattr(grid, "width", 0)),
            int(getattr(grid, "height", 0)),
            debug=bool(getattr(grid, "debug", False)),
        )
    except Exception:
        return copy.deepcopy(grid)

    if hasattr(grid, "encoding_level"):
        try:
            cloned_grid.encoding_level = int(getattr(grid, "encoding_level"))
        except Exception:
            pass
    if ruleset is not None:
        try:
            cloned_grid._ruleset = ruleset
        except Exception:
            pass

    memo: Dict[int, Any] = {}
    clone_object = _clone_baba_object
    raw_cells = getattr(grid, "grid", [])
    cloned_cells: List[List[Any]] = [[None] for _ in range(len(raw_cells))]
    iter_occupied = getattr(grid, "_iter_occupied_indices", None)
    if callable(iter_occupied):
        occupied_indices = tuple(iter_occupied())
    else:
        occupied_indices = tuple(
            idx
            for idx, stack in enumerate(raw_cells)
            if isinstance(stack, list) and any(item is not None for item in stack)
        )
    for idx in occupied_indices:
        stack = raw_cells[idx]
        if isinstance(stack, list):
            if len(stack) == 2 and stack[0] is None and stack[1] is not None:
                cloned_cells[idx] = [None, clone_object(stack[1], ruleset=ruleset, memo=memo)]
                continue
            source_stack = stack
        else:
            source_stack = [stack]

        cloned_stack: List[Any] = []
        append_cloned = cloned_stack.append
        for obj in source_stack:
            append_cloned(clone_object(obj, ruleset=ruleset, memo=memo) if obj is not None else None)
        if any(item is not None for item in cloned_stack):
            cloned_cells[idx] = cloned_stack
    cloned_grid.grid = cloned_cells
    if hasattr(cloned_grid, "_occupied_indices"):
        try:
            cloned_grid._occupied_indices = set(occupied_indices)
            cloned_grid._occupied_indices_cache = occupied_indices
        except Exception:
            rebuild_occupied = getattr(cloned_grid, "_rebuild_occupied_indices", None)
            if callable(rebuild_occupied):
                rebuild_occupied()
    return cloned_grid


def _serialize_baba_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if BabaWorldObj is not None and isinstance(value, BabaWorldObj):
        return _serialize_baba_object(value)
    if isinstance(value, tuple):
        return {"__tuple__": [_serialize_baba_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_baba_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_baba_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return {"__set__": [_serialize_baba_value(item) for item in value]}
    return value


def _normalize_scenario_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _scenario_display_label(scenario_type: Any) -> Optional[str]:
    normalized = _normalize_scenario_name(scenario_type)
    if normalized is None:
        return None
    return resolve_map_display_name(normalized)


def _serialize_baba_object(obj: Any) -> Any:
    if obj is None or BabaWorldObj is None or not isinstance(obj, BabaWorldObj):
        return _serialize_baba_value(obj)

    attrs: Dict[str, Any] = {}
    for key, value in obj.__dict__.items():
        if key == "_ruleset":
            continue
        if key == "img" and isinstance(value, np.ndarray):
            continue
        attrs[str(key)] = _serialize_baba_value(value)
    return {
        "__baba_object__": True,
        "module": str(obj.__class__.__module__),
        "class": str(obj.__class__.__name__),
        "attrs": attrs,
    }


def _deserialize_baba_value(value: Any, *, ruleset: Any = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_deserialize_baba_value(item, ruleset=ruleset) for item in value]
    if not isinstance(value, dict):
        return value
    if bool(value.get("__baba_object__", False)):
        return _deserialize_baba_object(value, ruleset=ruleset)
    if bool(value.get("__ndarray__", False)):
        return np.array(value.get("data", []), dtype=np.dtype(str(value.get("dtype", "float64"))))
    if "__tuple__" in value:
        raw_items = value.get("__tuple__", [])
        return tuple(_deserialize_baba_value(item, ruleset=ruleset) for item in raw_items)
    if "__set__" in value:
        raw_items = value.get("__set__", [])
        return {_deserialize_baba_value(item, ruleset=ruleset) for item in raw_items}
    return {key: _deserialize_baba_value(item, ruleset=ruleset) for key, item in value.items()}


def _deserialize_baba_object(payload: Any, *, ruleset: Any = None) -> Any:
    if not isinstance(payload, dict) or not bool(payload.get("__baba_object__", False)):
        return _deserialize_baba_value(payload, ruleset=ruleset)

    module = importlib.import_module(str(payload.get("module", "baba_in_wonderland.world_object")))
    cls = getattr(module, str(payload.get("class", "")))
    obj = cls.__new__(cls)
    raw_attrs = payload.get("attrs", {})
    attrs = {
        str(key): _deserialize_baba_value(value, ruleset=ruleset)
        for key, value in dict(raw_attrs).items()
    }
    obj.__dict__.update(attrs)
    if ruleset is not None and BabaWorldObj is not None and isinstance(obj, BabaWorldObj):
        obj.__dict__["_ruleset"] = ruleset
    return obj


def _serialize_baba_grid(grid: Any) -> Any:
    if grid is None:
        return None

    raw_cells = getattr(grid, "grid", [])
    iter_occupied = getattr(grid, "_iter_occupied_indices", None)
    if callable(iter_occupied):
        occupied_indices = tuple(int(idx) for idx in iter_occupied())
    else:
        occupied_indices = tuple(
            int(idx)
            for idx, stack in enumerate(raw_cells)
            if isinstance(stack, list) and any(item is not None for item in stack)
        )

    occupied_cells: List[Dict[str, Any]] = []
    for idx in occupied_indices:
        stack = raw_cells[idx]
        if isinstance(stack, list) and len(stack) == 2 and stack[0] is None and stack[1] is not None:
            occupied_cells.append(
                {
                    "index": int(idx),
                    "kind": "single_top",
                    "object": _serialize_baba_object(stack[1]),
                }
            )
            continue
        iterable_stack = stack if isinstance(stack, list) else [stack]
        occupied_cells.append(
            {
                "index": int(idx),
                "kind": "stack",
                "stack": [_serialize_baba_value(item) for item in iterable_stack],
            }
        )

    return {
        "grid_module": str(grid.__class__.__module__),
        "grid_class": str(grid.__class__.__name__),
        "width": int(getattr(grid, "width", 0)),
        "height": int(getattr(grid, "height", 0)),
        "debug": bool(getattr(grid, "debug", False)),
        "encoding_level": int(getattr(grid, "encoding_level", 1) or 1),
        "occupied_cells": occupied_cells,
    }


def _deserialize_baba_grid(payload: Any, *, ruleset: Any = None) -> Any:
    if not isinstance(payload, Mapping):
        return None

    module = importlib.import_module(str(payload.get("grid_module", "baba_in_wonderland.grid")))
    cls = getattr(module, str(payload.get("grid_class", "BabaIsYouGrid")))
    restored_grid = cls(
        int(payload.get("width", 0) or 0),
        int(payload.get("height", 0) or 0),
        debug=bool(payload.get("debug", False)),
    )
    if hasattr(restored_grid, "encoding_level"):
        try:
            restored_grid.encoding_level = int(payload.get("encoding_level", 1) or 1)
        except Exception:
            pass
    if ruleset is not None:
        try:
            restored_grid._ruleset = ruleset
        except Exception:
            pass

    raw_cells: List[List[Any]] = [[None] for _ in range(len(getattr(restored_grid, "grid", [])))]
    occupied_indices: List[int] = []
    for cell_payload in payload.get("occupied_cells", []):
        if not isinstance(cell_payload, Mapping):
            continue
        idx = int(cell_payload.get("index", -1))
        if idx < 0 or idx >= len(raw_cells):
            continue
        kind = str(cell_payload.get("kind", "")).strip().lower()
        if kind == "single_top":
            raw_cells[idx] = [None, _deserialize_baba_object(cell_payload.get("object"), ruleset=ruleset)]
        else:
            raw_stack = cell_payload.get("stack", [])
            iterable_stack = raw_stack if isinstance(raw_stack, list) else [raw_stack]
            raw_cells[idx] = [
                _deserialize_baba_value(item, ruleset=ruleset)
                for item in iterable_stack
            ]
        occupied_indices.append(idx)
    restored_grid.grid = raw_cells
    if hasattr(restored_grid, "_occupied_indices"):
        try:
            restored_grid._occupied_indices = set(occupied_indices)
            restored_grid._occupied_indices_cache = tuple(occupied_indices)
        except Exception:
            rebuild_occupied = getattr(restored_grid, "_rebuild_occupied_indices", None)
            if callable(rebuild_occupied):
                rebuild_occupied()
    return restored_grid


class BabaWrapper:
    """Baba in Wonderland environment wrapper."""

    ACTION_NAMES = {
        0: "idle",
        1: "up",
        2: "right",
        3: "down",
        4: "left",
    }

    @staticmethod
    def _normalize_runtime_word_aliases(
        word_aliases: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> Dict[str, Dict[str, str]]:
        if not isinstance(word_aliases, Mapping):
            return {}
        normalized: Dict[str, Dict[str, str]] = {}
        for raw_type, raw_mapping in word_aliases.items():
            if not isinstance(raw_mapping, Mapping):
                continue
            obj_type = normalize_visual_object_type(raw_type)
            alias_map: Dict[str, str] = {}
            for raw_word, raw_alias in raw_mapping.items():
                if not isinstance(raw_word, str) or not isinstance(raw_alias, str):
                    continue
                word = raw_word.strip().lower()
                alias = raw_alias.strip()
                if word and alias and word != alias:
                    alias_map[word] = alias
            if alias_map:
                normalized[obj_type] = alias_map
        return normalized

    @staticmethod
    def _reverse_unique_runtime_word_aliases(
        word_aliases: Mapping[str, Mapping[str, str]],
    ) -> Dict[str, Dict[str, str]]:
        reversed_aliases: Dict[str, Dict[str, str]] = {}
        collisions: Dict[str, set[str]] = {}
        for obj_type, raw_mapping in word_aliases.items():
            if not isinstance(raw_mapping, Mapping):
                continue
            type_aliases = reversed_aliases.setdefault(str(obj_type), {})
            type_collisions = collisions.setdefault(str(obj_type), set())
            for word, alias in raw_mapping.items():
                existing = type_aliases.get(str(alias))
                if existing is None:
                    type_aliases[str(alias)] = str(word)
                elif existing != str(word):
                    type_collisions.add(str(alias))
        for obj_type, aliases in collisions.items():
            for alias in aliases:
                reversed_aliases.get(obj_type, {}).pop(alias, None)
        return {
            str(obj_type): dict(alias_map)
            for obj_type, alias_map in reversed_aliases.items()
            if alias_map
        }

    def __init__(
        self,
        env_name: str = "env/two_room-break_stop-make_win",
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        state_format: str = "json",
        seed: Optional[int] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        kwargs = dict(env_kwargs or {})
        kwargs.setdefault("max_steps", int(max_steps))
        if render_mode is not None:
            kwargs.setdefault("render_mode", render_mode)

        maybe_env = baba_in_wonderland.make(env_name, **kwargs)
        if isinstance(maybe_env, dict):
            raise ValueError(
                f"`{env_name}` matched multiple environments. "
                "Provide a single concrete env id (e.g., env/two_room-break_stop-make_win)."
            )

        self.env_name = str(env_name)
        self.env = maybe_env
        self._runtime_word_aliases = self._normalize_runtime_word_aliases(word_aliases)
        self._runtime_reverse_word_aliases = self._reverse_unique_runtime_word_aliases(
            self._runtime_word_aliases
        )
        self.visualization_config = build_visualization_config(word_aliases)
        self.serializer = StateSerializer(
            format_type=state_format,
            word_aliases=word_aliases,
        )
        self.base_seed = int(seed) if seed is not None else None
        self._reset_count = 0
        self.last_reset_seed: Optional[int] = None

        if self.base_seed is not None:
            if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "seed"):
                self.env.action_space.seed(self.base_seed)
            if hasattr(self.env, "observation_space") and hasattr(
                self.env.observation_space, "seed"
            ):
                self.env.observation_space.seed(self.base_seed)

        self.grid_size: Tuple[int, int] = (0, 0)
        self.current_objects: List[Dict[str, Any]] = []
        self._state_store: Optional[StateStore] = None
        self.current_state_id: Optional[int] = None
        self.last_event = "reset"
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self._fixed_scenario_type = _normalize_scenario_name(
            getattr(self.env, "requested_scenario_type", None)
        )
        raw_scenario_types = getattr(self.env, "SCENARIO_TYPES", None)
        if isinstance(raw_scenario_types, (list, tuple)):
            self._scenario_types = tuple(
                str(item).strip()
                for item in raw_scenario_types
                if isinstance(item, str) and item.strip()
            )
        else:
            self._scenario_types = ()
        self._scenario_cycle_index = 0

    @property
    def state_store(self) -> StateStore:
        if self._state_store is None:
            self._state_store = StateStore()
        return self._state_store

    @state_store.setter
    def state_store(self, state_store: StateStore) -> None:
        if not isinstance(state_store, StateStore):
            raise TypeError("BabaWrapper.state_store must be a StateStore instance.")
        self._state_store = state_store

    def set_state_store(self, state_store: Any) -> None:
        if not isinstance(state_store, StateStore):
            return
        previous_store = self._state_store
        previous_state_id = getattr(self, "current_state_id", None)
        self._state_store = state_store
        if (
            isinstance(previous_store, StateStore)
            and isinstance(previous_state_id, int)
            and int(previous_state_id) > 0
        ):
            state_obj = previous_store.state_obj(int(previous_state_id))
            self.current_state_id = int(self.state_store.intern_state_obj(state_obj))
        else:
            self.current_state_id = None

    def _next_reset_scenario_type(self) -> Optional[str]:
        if self._fixed_scenario_type is not None:
            return self._fixed_scenario_type
        if len(self._scenario_types) <= 1:
            return None
        scenario_type = self._scenario_types[self._scenario_cycle_index % len(self._scenario_types)]
        self._scenario_cycle_index += 1
        return scenario_type

    def _perform_reset(
        self,
        seed: Optional[int] = None,
        *,
        scenario_type: Optional[str] = None,
    ) -> None:
        if seed is not None:
            reset_seed = int(seed)
        elif self.base_seed is not None:
            reset_seed = self.base_seed + self._reset_count
        else:
            reset_seed = None

        resolved_scenario_type = _normalize_scenario_name(scenario_type)
        if resolved_scenario_type is None:
            resolved_scenario_type = self._next_reset_scenario_type()
        if hasattr(self.env, "requested_scenario_type"):
            setattr(self.env, "requested_scenario_type", resolved_scenario_type)

        if reset_seed is None:
            _ = self.env.reset()
        else:
            # The Baba environment samples maps from global numpy randomness.
            np.random.seed(reset_seed)
            random.seed(reset_seed)
            _ = self.env.reset(seed=reset_seed)

        self._reset_count += 1
        self.last_reset_seed = reset_seed
        self.last_event = "reset"
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self._update_state()

    def reset_raw(self, seed: Optional[int] = None) -> None:
        self._perform_reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> str:
        self._perform_reset(seed=seed)
        return self._serialize_state()

    def list_graph_search_worlds(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], ...]:
        if seed is not None:
            resolved_seed = int(seed)
        elif self.base_seed is not None:
            resolved_seed = int(self.base_seed)
        else:
            resolved_seed = 0

        if self._fixed_scenario_type is not None:
            return (
                {
                    "seed": int(resolved_seed),
                    "scenario_type": str(self._fixed_scenario_type),
                    "label": str(
                        _scenario_display_label(self._fixed_scenario_type)
                        or self._fixed_scenario_type
                    ),
                },
            )

        if self._scenario_types:
            return tuple(
                {
                    "seed": int(resolved_seed),
                    "scenario_type": str(scenario_type),
                    "label": str(_scenario_display_label(scenario_type) or scenario_type),
                }
                for scenario_type in self._scenario_types
            )

        return (
            {
                "seed": int(resolved_seed),
                "scenario_type": None,
                "label": f"seed={int(resolved_seed)}",
            },
        )

    def reset_for_graph_search(
        self,
        *,
        seed: int,
        scenario_type: Optional[str] = None,
    ) -> str:
        self._perform_reset(
            seed=int(seed),
            scenario_type=_normalize_scenario_name(scenario_type),
        )
        return self._serialize_state()

    def step(self, action: int) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        try:
            result = self.env.step(int(action), emit_obs=False)
        except TypeError:
            result = self.env.step(int(action))
        if not isinstance(result, tuple):
            raise RuntimeError("Unexpected step return type from BABA environment.")

        if len(result) == 5:
            _, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            _, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(
                f"Unexpected step return length from BABA environment: {len(result)}"
            )

        is_win = bool(getattr(self.env, "is_win", False))
        is_defeat = bool(getattr(self.env, "is_defeat", False))
        reached_limit = bool(
            done
            and not is_win
            and not is_defeat
            and isinstance(getattr(self.env, "step_count", None), Integral)
            and isinstance(getattr(self.env, "max_steps", None), Integral)
            and int(self.env.step_count) >= int(self.env.max_steps)
        )
        terminated = bool(done and not reached_limit)
        truncated = bool(done and reached_limit)

        self.last_reward = float(reward)
        self.last_terminated = terminated
        self.last_truncated = truncated
        self.last_event = self._infer_event(
            action=int(action),
            terminated=terminated,
            truncated=truncated,
            is_win=is_win,
            is_defeat=is_defeat,
        )
        self._update_state()

        text_state = self._serialize_state()

        info_dict: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        info_dict.setdefault("is_win", is_win)
        info_dict.setdefault("is_defeat", is_defeat)
        return text_state, float(reward), terminated, truncated, info_dict

    def restore_runtime_packet(
        self,
        packet: Any,
        vocab: Any,
        *,
        step_count: int = 0,
    ) -> None:
        from .baba_runtime_state import restore_runtime_packet

        restore_runtime_packet(self, packet, vocab, step_count=int(step_count))

    def capture_runtime_packet(self, vocab: Any) -> Any:
        from .baba_runtime_state import capture_runtime_packet

        return capture_runtime_packet(self, vocab)

    def step_runtime_packet(
        self,
        action: int,
        vocab: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        try:
            result = self.env.step(int(action), emit_obs=False)
        except TypeError:
            result = self.env.step(int(action))
        if not isinstance(result, tuple):
            raise RuntimeError("Unexpected step return type from BABA environment.")

        if len(result) == 5:
            _, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            _, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(
                f"Unexpected step return length from BABA environment: {len(result)}"
            )

        is_win = bool(getattr(self.env, "is_win", False))
        is_defeat = bool(getattr(self.env, "is_defeat", False))
        reached_limit = bool(
            done
            and not is_win
            and not is_defeat
            and isinstance(getattr(self.env, "step_count", None), Integral)
            and isinstance(getattr(self.env, "max_steps", None), Integral)
            and int(self.env.step_count) >= int(self.env.max_steps)
        )
        terminated = bool(done and not reached_limit)
        truncated = bool(done and reached_limit)

        self.last_reward = float(reward)
        self.last_terminated = terminated
        self.last_truncated = truncated
        self.last_event = self._infer_event(
            action=int(action),
            terminated=terminated,
            truncated=truncated,
            is_win=is_win,
            is_defeat=is_defeat,
        )
        packet = self.capture_runtime_packet(vocab)

        info_dict: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        info_dict.setdefault("is_win", is_win)
        info_dict.setdefault("is_defeat", is_defeat)
        return packet, float(reward), terminated, truncated, info_dict

    def snapshot(self) -> Dict[str, Any]:
        """Capture an environment snapshot optimized for repeated solver restores."""

        agent_pos = getattr(self.env, "agent_pos", None)
        if agent_pos is None:
            encoded_agent_pos = None
        else:
            encoded_agent_pos = (int(agent_pos[0]), int(agent_pos[1]))

        ruleset_obj = getattr(self.env, "_ruleset", None)
        if BabaRuleset is not None and isinstance(ruleset_obj, BabaRuleset):
            ruleset_payload = _clone_plain_structure(ruleset_obj.ruleset_dict)
        else:
            ruleset_payload = copy.deepcopy(ruleset_obj)

        return {
            "snapshot_format": "compact_clone_v1",
            "width": int(getattr(self.env, "width", 0) or 0),
            "height": int(getattr(self.env, "height", 0) or 0),
            "grid": _clone_baba_grid(getattr(self.env, "grid", None), ruleset=None),
            "ruleset_dict": ruleset_payload,
            "requested_scenario_type": _normalize_scenario_name(
                getattr(self.env, "requested_scenario_type", None)
            ),
            "scenario_type": _normalize_scenario_name(
                getattr(self.env, "scenario_type", None)
            ),
            "agent_pos": encoded_agent_pos,
            "agent_dir": getattr(self.env, "agent_dir", None),
            "carrying": _clone_baba_object(getattr(self.env, "carrying", None), ruleset=None),
            "step_count": int(getattr(self.env, "step_count", 0) or 0),
            "is_win": bool(getattr(self.env, "is_win", False)),
            "is_defeat": bool(getattr(self.env, "is_defeat", False)),
            "last_event": str(self.last_event),
            "last_reward": float(self.last_reward),
            "last_terminated": bool(self.last_terminated),
            "last_truncated": bool(self.last_truncated),
        }

    def snapshot_serializable(self) -> Dict[str, Any]:
        """Capture a process-safe snapshot payload that can cross process boundaries."""

        agent_pos = getattr(self.env, "agent_pos", None)
        if agent_pos is None:
            encoded_agent_pos = None
        else:
            encoded_agent_pos = (int(agent_pos[0]), int(agent_pos[1]))

        ruleset_obj = getattr(self.env, "_ruleset", None)
        if BabaRuleset is not None and isinstance(ruleset_obj, BabaRuleset):
            ruleset_payload = _clone_plain_structure(ruleset_obj.ruleset_dict)
        else:
            ruleset_payload = copy.deepcopy(ruleset_obj)

        return {
            "snapshot_format": "serialized_v1",
            "width": int(getattr(self.env, "width", 0) or 0),
            "height": int(getattr(self.env, "height", 0) or 0),
            "grid": _serialize_baba_grid(getattr(self.env, "grid", None)),
            "ruleset_dict": ruleset_payload,
            "requested_scenario_type": _normalize_scenario_name(
                getattr(self.env, "requested_scenario_type", None)
            ),
            "scenario_type": _normalize_scenario_name(
                getattr(self.env, "scenario_type", None)
            ),
            "agent_pos": encoded_agent_pos,
            "agent_dir": getattr(self.env, "agent_dir", None),
            "carrying": _serialize_baba_object(getattr(self.env, "carrying", None)),
            "step_count": int(getattr(self.env, "step_count", 0) or 0),
            "is_win": bool(getattr(self.env, "is_win", False)),
            "is_defeat": bool(getattr(self.env, "is_defeat", False)),
            "last_event": str(self.last_event),
            "last_reward": float(self.last_reward),
            "last_terminated": bool(self.last_terminated),
            "last_truncated": bool(self.last_truncated),
        }

    def restore(
        self,
        snapshot: Mapping[str, Any],
        *,
        refresh_cache: bool = True,
    ) -> None:
        """Restore a previously captured snapshot."""

        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be a mapping.")

        snapshot_format = str(snapshot.get("snapshot_format", "")).strip()
        if snapshot_format == "serialized_v1":
            snapshot_ruleset_dict = snapshot.get("ruleset_dict", {})
            if BabaRuleset is not None:
                restored_ruleset = BabaRuleset(_clone_plain_structure(snapshot_ruleset_dict))
            else:
                restored_ruleset = copy.deepcopy(snapshot_ruleset_dict)
            self.env.grid = _deserialize_baba_grid(snapshot.get("grid"), ruleset=restored_ruleset)
            self.env._ruleset = restored_ruleset
            self.env.carrying = _deserialize_baba_object(snapshot.get("carrying"), ruleset=restored_ruleset)
        elif snapshot_format == "compact_clone_v1":
            snapshot_ruleset_dict = snapshot.get("ruleset_dict", {})
            if BabaRuleset is not None:
                restored_ruleset = BabaRuleset(_clone_plain_structure(snapshot_ruleset_dict))
            else:
                restored_ruleset = copy.deepcopy(snapshot_ruleset_dict)
            restored_grid = _clone_baba_grid(snapshot.get("grid"), ruleset=restored_ruleset)
            self.env.grid = restored_grid
            self.env._ruleset = restored_ruleset
            self.env.carrying = _clone_baba_object(snapshot.get("carrying"), ruleset=restored_ruleset)
        else:
            self.env.grid = copy.deepcopy(snapshot.get("grid"))
            self.env._ruleset = copy.deepcopy(snapshot.get("ruleset"))
            self.env.carrying = copy.deepcopy(snapshot.get("carrying"))

        restored_width = int(snapshot.get("width", 0) or 0)
        restored_height = int(snapshot.get("height", 0) or 0)
        sync_dimensions = getattr(self.env, "_sync_dimensions", None)
        if (
            callable(sync_dimensions)
            and restored_width > 0
            and restored_height > 0
        ):
            sync_dimensions(restored_width, restored_height)
        else:
            if restored_width > 0:
                self.env.width = restored_width
            if restored_height > 0:
                self.env.height = restored_height

        restored_agent_pos = snapshot.get("agent_pos")
        if restored_agent_pos is None:
            self.env.agent_pos = None
        else:
            self.env.agent_pos = np.array(restored_agent_pos, copy=True)
        restored_requested_scenario = _normalize_scenario_name(
            snapshot.get("requested_scenario_type")
        )
        restored_scenario_type = _normalize_scenario_name(
            snapshot.get("scenario_type")
        )
        if hasattr(self.env, "requested_scenario_type"):
            self.env.requested_scenario_type = restored_requested_scenario
        if hasattr(self.env, "scenario_type"):
            self.env.scenario_type = restored_scenario_type
        self.env.agent_dir = snapshot.get("agent_dir")
        self.env.step_count = int(snapshot.get("step_count", 0) or 0)
        self.env.is_win = bool(snapshot.get("is_win", False))
        self.env.is_defeat = bool(snapshot.get("is_defeat", False))

        self.last_event = str(snapshot.get("last_event", "restore"))
        self.last_reward = float(snapshot.get("last_reward", 0.0) or 0.0)
        self.last_terminated = bool(snapshot.get("last_terminated", False))
        self.last_truncated = bool(snapshot.get("last_truncated", False))

        if refresh_cache:
            self._update_state()

    def restore_serializable(
        self,
        snapshot: Mapping[str, Any],
        *,
        refresh_cache: bool = True,
    ) -> None:
        self.restore(snapshot, refresh_cache=refresh_cache)

    def _update_state(self) -> None:
        width = getattr(self.env, "width", None)
        height = getattr(self.env, "height", None)
        if isinstance(width, Integral) and isinstance(height, Integral):
            self.grid_size = (int(width), int(height))
        else:
            obs_shape = getattr(getattr(self.env, "observation_space", None), "shape", None)
            if (
                isinstance(obs_shape, tuple)
                and len(obs_shape) >= 2
                and isinstance(obs_shape[0], Integral)
                and isinstance(obs_shape[1], Integral)
            ):
                self.grid_size = (int(obs_shape[0]), int(obs_shape[1]))
            else:
                self.grid_size = (0, 0)

        logical_objects = self._extract_object_snapshots()
        state_obj = self._build_state_payload(logical_objects)
        self.current_objects = logical_objects
        self.current_state_id = int(self.state_store.intern_state_obj(state_obj))

    def _extract_object_snapshots(self) -> List[Dict[str, Any]]:
        grid = getattr(self.env, "grid", None)
        width, height = self.grid_size
        if grid is None or width <= 0 or height <= 0:
            return []

        logical_objects: List[Dict[str, Any]] = []
        raw_cells = getattr(grid, "grid", None)
        iter_occupied = getattr(grid, "_iter_occupied_indices", None)
        if callable(iter_occupied):
            occupied_indices = iter_occupied()
        else:
            occupied_indices = range(width * height)

        normalize_object_type = self._normalize_object_type
        normalize_schema = self._normalize_state_object_schema
        extract_direction = self._extract_object_direction

        for idx in occupied_indices:
            x = int(idx % width)
            y = int(idx // width)
            if isinstance(raw_cells, list) and idx < len(raw_cells):
                stack = raw_cells[idx]
            else:
                stack = self._get_cell_stack(grid=grid, x=x, y=y)

            if isinstance(stack, list) and len(stack) == 2 and stack[0] is None and stack[1] is not None:
                obj = stack[1]
                raw_type = normalize_object_type(str(getattr(obj, "type", "unknown")))
                obj_type, word = normalize_schema(obj=obj, raw_type=raw_type)
                row = {"type": obj_type, "word": word, "position": [x, y]}
                direction = extract_direction(obj)
                if direction is not None:
                    row["direction"] = direction
                logical_objects.append(dict(row))
                continue

            iterable_stack = stack if isinstance(stack, list) else [stack]
            non_none_count = 0
            for obj in iterable_stack:
                if obj is None:
                    continue
                non_none_count += 1
                raw_type = normalize_object_type(str(getattr(obj, "type", "unknown")))
                obj_type, word = normalize_schema(obj=obj, raw_type=raw_type)
                row = {
                    "type": obj_type,
                    "word": word,
                    "position": [x, y],
                }
                direction = extract_direction(obj)
                if direction is not None:
                    row["direction"] = direction
                logical_objects.append(dict(row))
        return logical_objects

    def _build_state_payload(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        return build_state_payload(
            grid_size=self.grid_size,
            objects=objects,
            terminated=bool(self.last_terminated),
            word_aliases=self.serializer.word_aliases,
        )

    def get_visualization_state(self) -> Dict[str, Any]:
        if not isinstance(self.current_state_id, int) or int(self.current_state_id) <= 0:
            return {}
        return self.state_store.state_obj(int(self.current_state_id))

    def get_canonical_state_key(self, state_json: Optional[str] = None) -> Optional[str]:
        if not isinstance(self.current_state_id, int) or int(self.current_state_id) <= 0:
            return None
        current_state_key = self.state_store.state_key(int(self.current_state_id))
        if not isinstance(current_state_key, str) or not current_state_key:
            return None
        if state_json is None:
            return current_state_key
        resolved_state_json = str(state_json).strip()
        if not resolved_state_json:
            return None
        return canonical_state_key(resolved_state_json)

    def _get_cell_stack(self, grid: Any, x: int, y: int) -> List[Any]:
        try:
            stack = grid.get(x, y, "all")
        except TypeError:
            stack = grid.get(x, y)
        except Exception:
            return []

        if isinstance(stack, list):
            return stack
        return [stack]

    def _normalize_object_type(self, raw_type: str) -> str:
        value = raw_type.strip().lower() or "unknown"
        if value.startswith("f") and len(value) > 1 and value != "floor":
            return value[1:]
        return value

    def _normalize_state_object_schema(self, obj: Any, raw_type: str) -> Tuple[str, str]:
        if raw_type == "rule_object":
            return "rule_noun", self._extract_object_word(obj, fallback=raw_type)
        if raw_type == "rule_is":
            return "rule_operator", "is"
        if raw_type == "rule_and":
            return "rule_operator", "and"
        if raw_type in {"rule_property", "rule_color"}:
            return "rule_property", self._extract_object_word(obj, fallback=raw_type)
        return "world_object", self._extract_object_word(obj, fallback=raw_type)

    def _extract_object_word(self, obj: Any, fallback: str) -> str:
        raw_word = getattr(obj, "name", None)
        if isinstance(raw_word, str):
            word = raw_word.strip().lower()
            if word:
                if word == "flava":
                    return "lava"
                return word
        normalized_fallback = self._normalize_object_type(str(fallback or "unknown"))
        return normalized_fallback or "unknown"

    def _extract_object_direction(self, obj: Any) -> Optional[str]:
        raw_direction = getattr(obj, "dir", None)
        if not isinstance(raw_direction, Integral):
            return None
        return StateSerializer.DIRECTION_NAMES.get(int(raw_direction), "unknown")

    def _infer_event(
        self,
        action: int,
        terminated: bool,
        truncated: bool,
        is_win: bool,
        is_defeat: bool,
    ) -> str:
        if terminated:
            if is_win:
                return "reached_goal"
            if is_defeat:
                return "defeated"
            return "terminated"
        if truncated:
            return "time_limit"
        if action == 0:
            return "idle"
        return "none"

    def _serialize_state(self) -> str:
        if not isinstance(self.current_state_id, int) or int(self.current_state_id) <= 0:
            return ""
        return self.state_store.state_json(int(self.current_state_id))

    def get_action_name(self, action: int) -> str:
        return self.ACTION_NAMES.get(action, f"unknown action {action}")

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def num_actions(self) -> int:
        return self.env.action_space.n

    def close(self):
        self.env.close()

    def get_visualization_config(self) -> Dict[str, Any]:
        return dict(self.visualization_config)


