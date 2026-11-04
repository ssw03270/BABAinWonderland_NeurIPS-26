from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from .state_schema import (
    STATE_KEY_PREFIX,
    canonicalize_state_json,
    dump_state_json,
    normalize_state_payload,
    pack_state_obj,
    parse_state_json,
)


_CACHE_MAX_SIZE = 4096
_EMPTY_JSON_OBJECT = "{}"
_EMPTY_JSON_ARRAY = "[]"
_NULL_SENTINEL = 0xFFFF
_STATE_HAS_GRID_SIZE = 1 << 0
_STATE_HAS_STEP = 1 << 1
_STATE_HAS_OBJECTS = 1 << 2
_STEP_HAS_TERMINATED = 1 << 0
_OBJECT_HAS_TYPE = 1 << 0
_OBJECT_HAS_WORD = 1 << 1
_OBJECT_HAS_POSITION = 1 << 2
_OBJECT_HAS_DIRECTION = 1 << 3
_OBJECT_HAS_COLOR = 1 << 4
_OBJECT_FULL_FLAGS = (
    _OBJECT_HAS_TYPE | _OBJECT_HAS_WORD | _OBJECT_HAS_POSITION | _OBJECT_HAS_DIRECTION
)
_STATE_STORE_FORMAT_VERSION = 1
_NO_RUNTIME_DIRECTION = -1

# ---------------------------------------------------------------------------
# Append-only snapshot format (format version 2)
#
# ``save_directory`` rewrites every column, which costs O(total states) even
# when only a handful of new states were interned. The append-only snapshot
# keeps the same columns as raw, header-less binary files so a writer can
# extend them in place, and commits the new lengths by atomically replacing a
# single manifest.
#
# The invariant that makes this safe: *no committed byte is ever rewritten*.
# Row ``i`` of every column is written exactly once, when state id ``i + 1`` is
# first snapshotted, and the manifest that publishes it is replaced atomically
# after the column bytes are fsynced. A reader maps exactly the number of rows
# the manifest declares, so it can only ever see fully-durable bytes.
#
# ``.npy`` cannot be used here: its shape lives in a 128-byte header that an
# append would have to rewrite, which is precisely the one window in which a
# reader could observe a torn file.
# ---------------------------------------------------------------------------
_APPEND_SNAPSHOT_FORMAT_VERSION = 2
_APPEND_SNAPSHOT_MANIFEST_NAME = "manifest.json"
_APPEND_SNAPSHOT_MANIFEST_TEMP_NAME = "manifest.json.tmp"
# The state-key column is the one column whose row stride can change: numpy's
# fixed-width unicode dtype truncates silently, so a key wider than the
# committed itemsize needs the whole column restrided. Restriding it in place
# would break the invariant above - the bytes are rewritten while the live
# manifest still declares the old width, and a reader indexing at the old
# stride lands mid-key. So the width goes in the FILE NAME and a restride
# writes a brand-new file that no committed manifest points at; the manifest
# swap then publishes the new name atomically, exactly like every other change.
_APPEND_SNAPSHOT_STATE_KEY_NAME_PREFIX = "state_keys_w"
_APPEND_SNAPSHOT_STATE_KEY_NAME_SUFFIX = ".bin"


def _append_snapshot_state_key_name(width: int) -> str:
    resolved = max(1, int(width))
    return (
        f"{_APPEND_SNAPSHOT_STATE_KEY_NAME_PREFIX}{resolved}"
        f"{_APPEND_SNAPSHOT_STATE_KEY_NAME_SUFFIX}"
    )
_APPEND_SNAPSHOT_REPLACE_ATTEMPTS = 50
_APPEND_SNAPSHOT_REPLACE_SLEEP_SEC = 0.02

# (attribute, file name, dtype) for the columns indexed by ``state_id - 1``.
_APPEND_SNAPSHOT_STATE_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("_widths", "widths.bin", "<i4"),
    ("_heights", "heights.bin", "<i4"),
    ("_state_flags", "state_flags.bin", "|u1"),
    ("_step_flags", "step_flags.bin", "|u1"),
    ("_terminated", "terminated.bin", "|i1"),
    ("_object_offsets", "object_offsets.bin", "<u4"),
    ("_object_counts", "object_counts.bin", "<u4"),
    ("_step_extra_ids", "step_extra_ids.bin", "<u4"),
    ("_state_extra_ids", "state_extra_ids.bin", "<u4"),
)

# (attribute, file name, dtype) for the columns indexed by object row.
_APPEND_SNAPSHOT_OBJECT_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("_object_flags", "object_flags.bin", "|u1"),
    ("_object_type_ids", "object_type_ids.bin", "<u2"),
    ("_object_word_ids", "object_word_ids.bin", "<u2"),
    ("_object_x", "object_x.bin", "<i4"),
    ("_object_y", "object_y.bin", "<i4"),
    ("_object_direction_ids", "object_direction_ids.bin", "<u2"),
    ("_object_color_ids", "object_color_ids.bin", "<u2"),
    ("_object_extra_ids", "object_extra_ids.bin", "<u4"),
)

# (attribute, manifest key) for the five text vocabularies. They are tiny and are
# carried inside the manifest so that the lengths a reader maps and the vocab
# it resolves ids against are published by the same atomic replace.
_APPEND_SNAPSHOT_VOCABS: Tuple[Tuple[str, str], ...] = (
    ("_type_text_by_id", "type"),
    ("_word_text_by_id", "word"),
    ("_direction_text_by_id", "direction"),
    ("_color_text_by_id", "color"),
    ("_json_text_by_id", "json"),
)

_store_token_counter = itertools.count(1)

_DIRECTION_TEXT_BY_RUNTIME_ID = (
    "facing right",
    "facing down",
    "facing left",
    "facing up",
)
_RUNTIME_ID_BY_DIRECTION_TEXT = {
    text: index for index, text in enumerate(_DIRECTION_TEXT_BY_RUNTIME_ID)
}


@dataclass(frozen=True, slots=True)
class RuntimeStateVocab:
    type_texts: Tuple[str, ...]
    word_texts: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeObjectRow:
    x: int
    y: int
    type_id: int
    word_id: int
    direction: int = _NO_RUNTIME_DIRECTION


@dataclass(frozen=True, slots=True)
class RuntimeStatePacket:
    width: int
    height: int
    terminated: bool
    objects: Tuple[RuntimeObjectRow, ...]


@dataclass(frozen=True, slots=True)
class StateTokenView:
    width: int
    height: int
    object_offset: int
    object_count: int
    object_flags: Any
    object_type_ids: Any
    object_word_ids: Any
    object_x: Any
    object_y: Any
    object_direction_ids: Any
    type_texts: List[str]
    word_texts: List[str]
    direction_texts: List[str]
    required_object_flags: int
    direction_object_flag: int
    null_sentinel: int


def _clone_plain_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_plain_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_plain_structure(item) for item in value]
    return value


def _is_state_key_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text.startswith(STATE_KEY_PREFIX):
        return False
    digest = text[len(STATE_KEY_PREFIX) :]
    return bool(digest) and all(char in "0123456789abcdef" for char in digest.lower())


def canonical_state_key(state: Any) -> str:
    if isinstance(state, str):
        text = state.strip()
        if not text:
            return ""
        if _is_state_key_text(text):
            return text
        try:
            digest_source = pack_state_obj(parse_state_json(text))
        except ValueError:
            digest_source = text.encode("utf-8")
        return f"{STATE_KEY_PREFIX}{hashlib.sha1(digest_source).hexdigest()}"
    if isinstance(state, dict):
        digest_source = pack_state_obj(normalize_state_payload(state))
        return f"{STATE_KEY_PREFIX}{hashlib.sha1(digest_source).hexdigest()}"
    try:
        return canonical_state_key(canonicalize_state_json(state))
    except ValueError:
        return f"{STATE_KEY_PREFIX}{hashlib.sha1(str(state).encode('utf-8')).hexdigest()}"


def _runtime_vocab_text(values: Tuple[str, ...], index: int) -> str:
    resolved_index = int(index)
    if resolved_index < 0 or resolved_index >= len(values):
        return ""
    return str(values[resolved_index])


def runtime_state_packet_obj(
    packet: RuntimeStatePacket,
    vocab: RuntimeStateVocab,
) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    type_texts = tuple(str(value) for value in vocab.type_texts)
    word_texts = tuple(str(value) for value in vocab.word_texts)
    for row in packet.objects:
        type_text = _runtime_vocab_text(type_texts, int(row.type_id))
        word_text = _runtime_vocab_text(word_texts, int(row.word_id))
        if not type_text or not word_text:
            raise KeyError(
                "Runtime state packet references an unknown state-store vocab id."
            )
        payload: Dict[str, Any] = {
            "type": str(type_text),
            "word": str(word_text),
            "position": [int(row.x), int(row.y)],
        }
        if int(row.direction) != _NO_RUNTIME_DIRECTION:
            direction_index = int(row.direction)
            if direction_index < 0 or direction_index >= len(_DIRECTION_TEXT_BY_RUNTIME_ID):
                raise KeyError(
                    f"Runtime state packet references an unknown direction id: {direction_index}"
                )
            payload["direction"] = _DIRECTION_TEXT_BY_RUNTIME_ID[direction_index]
        objects.append(payload)
    return {
        "grid_size": [int(packet.width), int(packet.height)],
        "step": {"terminated": bool(packet.terminated)},
        "objects": objects,
    }


def runtime_state_packet_key(
    packet: RuntimeStatePacket,
    vocab: RuntimeStateVocab,
) -> str:
    return canonical_state_key(runtime_state_packet_obj(packet, vocab))


class _ViewCache:
    def __init__(self, max_size: int = _CACHE_MAX_SIZE) -> None:
        self.max_size = max(1, int(max_size))
        self._data: OrderedDict[int, Any] = OrderedDict()

    def clear(self) -> None:
        self._data.clear()

    def get(self, key: int) -> Any:
        resolved_key = int(key)
        if resolved_key not in self._data:
            return None
        value = self._data.pop(resolved_key)
        self._data[resolved_key] = value
        return value

    def put(self, key: int, value: Any) -> Any:
        resolved_key = int(key)
        if resolved_key in self._data:
            self._data.pop(resolved_key)
        self._data[resolved_key] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)
        return value


class StateStore:
    """
    Discovery-local immutable packed state store.

    Source of truth is an integer-coded in-memory column store.
    JSON/dict views are reconstructed lazily on demand.
    """

    def __init__(
        self,
        *,
        state_obj_cache_max_size: int = _CACHE_MAX_SIZE,
        state_json_cache_max_size: int = _CACHE_MAX_SIZE,
        runtime_packet_cache_max_size: int = _CACHE_MAX_SIZE,
    ) -> None:
        self._state_id_by_key: Dict[str, int] = {}
        self._state_key_by_id: Dict[int, str] = {}
        self._state_keys_array: Optional[np.ndarray] = None
        self._readonly = False
        # Identity for append-only snapshot caches. ``id(store)`` cannot be
        # used: CPython reuses the address of a freed object, so a new store
        # allocated at a dead store's address would look like the same store
        # that a cached snapshot was built from. This counter never repeats.
        self._store_token = int(next(_store_token_counter))

        self._widths = array("i")
        self._heights = array("i")
        self._state_flags = array("B")
        self._step_flags = array("B")
        self._terminated = array("b")
        self._object_offsets = array("I")
        self._object_counts = array("I")
        self._step_extra_ids = array("I")
        self._state_extra_ids = array("I")

        self._object_flags = array("B")
        self._object_type_ids = array("H")
        self._object_word_ids = array("H")
        self._object_x = array("i")
        self._object_y = array("i")
        self._object_direction_ids = array("H")
        self._object_color_ids = array("H")
        self._object_extra_ids = array("I")

        self._type_id_by_text: Dict[str, int] = {}
        self._type_text_by_id: List[str] = []
        self._word_id_by_text: Dict[str, int] = {}
        self._word_text_by_id: List[str] = []
        self._direction_id_by_text: Dict[str, int] = {}
        self._direction_text_by_id: List[str] = []
        self._color_id_by_text: Dict[str, int] = {}
        self._color_text_by_id: List[str] = []
        self._json_id_by_text: Dict[str, int] = {_EMPTY_JSON_OBJECT: 0, _EMPTY_JSON_ARRAY: 1}
        self._json_text_by_id: List[str] = [_EMPTY_JSON_OBJECT, _EMPTY_JSON_ARRAY]

        self._state_obj_cache = _ViewCache(state_obj_cache_max_size)
        self._state_json_cache = _ViewCache(state_json_cache_max_size)
        self._runtime_packet_cache = _ViewCache(runtime_packet_cache_max_size)

    def __len__(self) -> int:
        return int(len(self._widths))

    @property
    def store_token(self) -> int:
        """Process-unique identity of this store's *current* id assignment.

        State ids are append-only for a fixed token: a snapshot taken while the
        token was ``t`` stays correct for every id it already covers. ``clear``
        renumbers, so it takes a fresh token.
        """
        return int(self._store_token)

    def clear(self) -> None:
        self.close()
        self._state_id_by_key.clear()
        self._state_key_by_id.clear()
        self._state_keys_array = None
        self._readonly = False
        self._store_token = int(next(_store_token_counter))

        self._widths = array("i")
        self._heights = array("i")
        self._state_flags = array("B")
        self._step_flags = array("B")
        self._terminated = array("b")
        self._object_offsets = array("I")
        self._object_counts = array("I")
        self._step_extra_ids = array("I")
        self._state_extra_ids = array("I")

        self._object_flags = array("B")
        self._object_type_ids = array("H")
        self._object_word_ids = array("H")
        self._object_x = array("i")
        self._object_y = array("i")
        self._object_direction_ids = array("H")
        self._object_color_ids = array("H")
        self._object_extra_ids = array("I")

        self._type_id_by_text.clear()
        self._type_text_by_id.clear()
        self._word_id_by_text.clear()
        self._word_text_by_id.clear()
        self._direction_id_by_text.clear()
        self._direction_text_by_id.clear()
        self._color_id_by_text.clear()
        self._color_text_by_id.clear()
        self._json_id_by_text = {_EMPTY_JSON_OBJECT: 0, _EMPTY_JSON_ARRAY: 1}
        self._json_text_by_id = [_EMPTY_JSON_OBJECT, _EMPTY_JSON_ARRAY]

        self._state_obj_cache.clear()
        self._state_json_cache.clear()
        self._runtime_packet_cache.clear()

    @staticmethod
    def _close_array_storage(value: Any) -> None:
        current = value
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            mmap_handle = getattr(current, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
                return
            current = getattr(current, "base", None)

    def close(self) -> None:
        for value in (
            self._state_keys_array,
            self._widths,
            self._heights,
            self._state_flags,
            self._step_flags,
            self._terminated,
            self._object_offsets,
            self._object_counts,
            self._step_extra_ids,
            self._state_extra_ids,
            self._object_flags,
            self._object_type_ids,
            self._object_word_ids,
            self._object_x,
            self._object_y,
            self._object_direction_ids,
            self._object_color_ids,
            self._object_extra_ids,
        ):
            if value is not None:
                self._close_array_storage(value)

    def _ensure_state_key_indexes(self) -> None:
        if self._state_id_by_key and self._state_key_by_id:
            return
        if self._state_keys_array is None:
            return
        self._state_key_by_id = {
            int(index) + 1: str(key)
            for index, key in enumerate(self._state_keys_array)
        }
        self._state_id_by_key = {
            key: int(index)
            for index, key in self._state_key_by_id.items()
        }

    def _ensure_vocab_indexes(self) -> None:
        if not self._type_id_by_text and self._type_text_by_id:
            self._type_id_by_text = {value: index for index, value in enumerate(self._type_text_by_id)}
        if not self._word_id_by_text and self._word_text_by_id:
            self._word_id_by_text = {value: index for index, value in enumerate(self._word_text_by_id)}
        if not self._direction_id_by_text and self._direction_text_by_id:
            self._direction_id_by_text = {value: index for index, value in enumerate(self._direction_text_by_id)}
        if not self._color_id_by_text and self._color_text_by_id:
            self._color_id_by_text = {value: index for index, value in enumerate(self._color_text_by_id)}
        if not self._json_id_by_text and self._json_text_by_id:
            self._json_id_by_text = {value: index for index, value in enumerate(self._json_text_by_id)}

    def _ensure_mutable_storage(self) -> None:
        if not self._readonly:
            return
        self._ensure_state_key_indexes()
        self._ensure_vocab_indexes()
        self._widths = array("i", np.asarray(self._widths, dtype=np.int32).tolist())
        self._heights = array("i", np.asarray(self._heights, dtype=np.int32).tolist())
        self._state_flags = array("B", np.asarray(self._state_flags, dtype=np.uint8).tolist())
        self._step_flags = array("B", np.asarray(self._step_flags, dtype=np.uint8).tolist())
        self._terminated = array("b", np.asarray(self._terminated, dtype=np.int8).tolist())
        self._object_offsets = array("I", np.asarray(self._object_offsets, dtype=np.uint32).tolist())
        self._object_counts = array("I", np.asarray(self._object_counts, dtype=np.uint32).tolist())
        self._step_extra_ids = array("I", np.asarray(self._step_extra_ids, dtype=np.uint32).tolist())
        self._state_extra_ids = array("I", np.asarray(self._state_extra_ids, dtype=np.uint32).tolist())
        self._object_flags = array("B", np.asarray(self._object_flags, dtype=np.uint8).tolist())
        self._object_type_ids = array("H", np.asarray(self._object_type_ids, dtype=np.uint16).tolist())
        self._object_word_ids = array("H", np.asarray(self._object_word_ids, dtype=np.uint16).tolist())
        self._object_x = array("i", np.asarray(self._object_x, dtype=np.int32).tolist())
        self._object_y = array("i", np.asarray(self._object_y, dtype=np.int32).tolist())
        self._object_direction_ids = array(
            "H",
            np.asarray(self._object_direction_ids, dtype=np.uint16).tolist(),
        )
        self._object_color_ids = array(
            "H",
            np.asarray(self._object_color_ids, dtype=np.uint16).tolist(),
        )
        self._object_extra_ids = array("I", np.asarray(self._object_extra_ids, dtype=np.uint32).tolist())
        self._state_keys_array = None
        self._readonly = False

    def save_directory(self, root: str | Path) -> Path:
        return self._save_directory_range(
            root,
            first_state_id=1,
            last_state_id=int(len(self._widths)),
        )

    def save_directory_slice(
        self,
        root: str | Path,
        *,
        first_state_id: int,
        last_state_id: int,
    ) -> Path:
        return self._save_directory_range(
            root,
            first_state_id=int(first_state_id),
            last_state_id=int(last_state_id),
        )

    def _save_directory_range(
        self,
        root: str | Path,
        *,
        first_state_id: int,
        last_state_id: int,
    ) -> Path:
        output_root = Path(root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        total_state_count = int(len(self._widths))
        first_id = max(1, int(first_state_id))
        last_id = int(last_state_id)
        if last_id < first_id:
            state_start = 0
            state_end = 0
            state_count = 0
        else:
            if last_id > total_state_count:
                raise KeyError(
                    f"StateStore slice end {last_id} exceeds state count {total_state_count}."
                )
            state_start = int(first_id - 1)
            state_end = int(last_id)
            state_count = int(state_end - state_start)
        if state_count:
            object_start = int(self._object_offsets[state_start])
            last_object_index = int(state_end - 1)
            object_end = int(
                self._object_offsets[last_object_index]
                + self._object_counts[last_object_index]
            )
            object_offsets = (
                np.asarray(
                    self._object_offsets[state_start:state_end],
                    dtype=np.uint32,
                )
                - np.uint32(object_start)
            )
        else:
            object_start = 0
            object_end = 0
            object_offsets = np.asarray([], dtype=np.uint32)

        manifest = {
            "formatVersion": int(_STATE_STORE_FORMAT_VERSION),
            "stateCount": int(state_count),
            "firstStateId": int(first_id if state_count else 1),
            "lastStateId": int(last_id if state_count else 0),
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        np.save(output_root / "widths.npy", np.asarray(self._widths, dtype=np.int32)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "heights.npy", np.asarray(self._heights, dtype=np.int32)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "state_flags.npy", np.asarray(self._state_flags, dtype=np.uint8)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "step_flags.npy", np.asarray(self._step_flags, dtype=np.uint8)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "terminated.npy", np.asarray(self._terminated, dtype=np.int8)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "object_offsets.npy", object_offsets, allow_pickle=False)
        np.save(output_root / "object_counts.npy", np.asarray(self._object_counts, dtype=np.uint32)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "step_extra_ids.npy", np.asarray(self._step_extra_ids, dtype=np.uint32)[state_start:state_end], allow_pickle=False)
        np.save(output_root / "state_extra_ids.npy", np.asarray(self._state_extra_ids, dtype=np.uint32)[state_start:state_end], allow_pickle=False)

        np.save(output_root / "object_flags.npy", np.asarray(self._object_flags, dtype=np.uint8)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_type_ids.npy", np.asarray(self._object_type_ids, dtype=np.uint16)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_word_ids.npy", np.asarray(self._object_word_ids, dtype=np.uint16)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_x.npy", np.asarray(self._object_x, dtype=np.int32)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_y.npy", np.asarray(self._object_y, dtype=np.int32)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_direction_ids.npy", np.asarray(self._object_direction_ids, dtype=np.uint16)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_color_ids.npy", np.asarray(self._object_color_ids, dtype=np.uint16)[object_start:object_end], allow_pickle=False)
        np.save(output_root / "object_extra_ids.npy", np.asarray(self._object_extra_ids, dtype=np.uint32)[object_start:object_end], allow_pickle=False)

        if self._state_keys_array is not None:
            state_keys = np.asarray(self._state_keys_array, dtype=np.str_)[
                state_start:state_end
            ]
        else:
            state_keys = np.asarray(
                [
                    self._state_key_by_id.get(int(state_id), "")
                    for state_id in range(first_id, last_id + 1)
                ],
                dtype=np.str_,
            )

        np.save(
            output_root / "state_keys.npy",
            state_keys,
            allow_pickle=False,
        )
        np.save(output_root / "type_vocab.npy", np.asarray(self._type_text_by_id, dtype=np.str_), allow_pickle=False)
        np.save(output_root / "word_vocab.npy", np.asarray(self._word_text_by_id, dtype=np.str_), allow_pickle=False)
        np.save(
            output_root / "direction_vocab.npy",
            np.asarray(self._direction_text_by_id, dtype=np.str_),
            allow_pickle=False,
        )
        np.save(output_root / "color_vocab.npy", np.asarray(self._color_text_by_id, dtype=np.str_), allow_pickle=False)
        np.save(output_root / "json_vocab.npy", np.asarray(self._json_text_by_id, dtype=np.str_), allow_pickle=False)
        return output_root

    @classmethod
    def load_directory(
        cls,
        root: str | Path,
        *,
        state_obj_cache_max_size: int = _CACHE_MAX_SIZE,
        state_json_cache_max_size: int = _CACHE_MAX_SIZE,
        runtime_packet_cache_max_size: int = _CACHE_MAX_SIZE,
    ) -> "StateStore":
        input_root = Path(root).resolve()
        manifest_path = input_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid state store manifest: {manifest_path}")
        if int(manifest.get("formatVersion", 0) or 0) != _STATE_STORE_FORMAT_VERSION:
            raise ValueError(f"Unsupported state store format version: {manifest_path}")

        store = cls(
            state_obj_cache_max_size=state_obj_cache_max_size,
            state_json_cache_max_size=state_json_cache_max_size,
            runtime_packet_cache_max_size=runtime_packet_cache_max_size,
        )
        store._readonly = True
        store._widths = np.load(input_root / "widths.npy", allow_pickle=False, mmap_mode="r")
        store._heights = np.load(input_root / "heights.npy", allow_pickle=False, mmap_mode="r")
        store._state_flags = np.load(input_root / "state_flags.npy", allow_pickle=False, mmap_mode="r")
        store._step_flags = np.load(input_root / "step_flags.npy", allow_pickle=False, mmap_mode="r")
        store._terminated = np.load(input_root / "terminated.npy", allow_pickle=False, mmap_mode="r")
        store._object_offsets = np.load(input_root / "object_offsets.npy", allow_pickle=False, mmap_mode="r")
        store._object_counts = np.load(input_root / "object_counts.npy", allow_pickle=False, mmap_mode="r")
        store._step_extra_ids = np.load(input_root / "step_extra_ids.npy", allow_pickle=False, mmap_mode="r")
        store._state_extra_ids = np.load(input_root / "state_extra_ids.npy", allow_pickle=False, mmap_mode="r")

        store._object_flags = np.load(input_root / "object_flags.npy", allow_pickle=False, mmap_mode="r")
        store._object_type_ids = np.load(input_root / "object_type_ids.npy", allow_pickle=False, mmap_mode="r")
        store._object_word_ids = np.load(input_root / "object_word_ids.npy", allow_pickle=False, mmap_mode="r")
        store._object_x = np.load(input_root / "object_x.npy", allow_pickle=False, mmap_mode="r")
        store._object_y = np.load(input_root / "object_y.npy", allow_pickle=False, mmap_mode="r")
        store._object_direction_ids = np.load(
            input_root / "object_direction_ids.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        store._object_color_ids = np.load(
            input_root / "object_color_ids.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        store._object_extra_ids = np.load(input_root / "object_extra_ids.npy", allow_pickle=False, mmap_mode="r")
        store._state_keys_array = np.load(input_root / "state_keys.npy", allow_pickle=False, mmap_mode="r")

        store._type_text_by_id = [str(value) for value in np.load(input_root / "type_vocab.npy", allow_pickle=False).tolist()]
        store._word_text_by_id = [str(value) for value in np.load(input_root / "word_vocab.npy", allow_pickle=False).tolist()]
        store._direction_text_by_id = [
            str(value) for value in np.load(input_root / "direction_vocab.npy", allow_pickle=False).tolist()
        ]
        store._color_text_by_id = [str(value) for value in np.load(input_root / "color_vocab.npy", allow_pickle=False).tolist()]
        store._json_text_by_id = [str(value) for value in np.load(input_root / "json_vocab.npy", allow_pickle=False).tolist()]

        store._type_id_by_text = {}
        store._word_id_by_text = {}
        store._direction_id_by_text = {}
        store._color_id_by_text = {}
        store._json_id_by_text = {}

        store._state_obj_cache.clear()
        store._state_json_cache.clear()
        store._runtime_packet_cache.clear()
        return store

    # ------------------------------------------------------------------
    # Append-only snapshot (format version 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_append_snapshot_manifest(root: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(
                (root / _APPEND_SNAPSHOT_MANIFEST_NAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("formatVersion", 0) or 0) != _APPEND_SNAPSHOT_FORMAT_VERSION:
            return None
        return payload

    @staticmethod
    def _commit_append_snapshot_manifest(root: Path, manifest: Dict[str, Any]) -> None:
        temp_path = root / _APPEND_SNAPSHOT_MANIFEST_TEMP_NAME
        with open(temp_path, "wb") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
        final_path = root / _APPEND_SNAPSHOT_MANIFEST_NAME
        # os.replace is atomic, so a reader observes either the whole previous
        # manifest or the whole new one. On Windows it can still fail with a
        # sharing violation while a reader holds the destination open; readers
        # keep it open only long enough to parse a couple of kilobytes.
        for attempt in range(_APPEND_SNAPSHOT_REPLACE_ATTEMPTS):
            try:
                os.replace(temp_path, final_path)
                return
            except PermissionError:
                if attempt == _APPEND_SNAPSHOT_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_APPEND_SNAPSHOT_REPLACE_SLEEP_SEC)

    @staticmethod
    def _append_snapshot_rows(
        path: Path,
        *,
        dtype: str,
        values: Any,
        first_row: int,
    ) -> None:
        payload = np.asarray(values, dtype=dtype)
        if not payload.size:
            return
        descriptor = os.open(
            str(path),
            os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0),
        )
        with os.fdopen(descriptor, "r+b") as handle:
            # Seek to the last committed row rather than opening in append
            # mode: bytes past the committed length can only be residue from a
            # crashed write, and overwriting them keeps the file consistent
            # with the manifest that is about to publish it.
            handle.seek(int(first_row) * int(payload.dtype.itemsize))
            handle.write(payload.tobytes())
        # Deliberately not fsynced. A reader in another process sees these
        # bytes as soon as write() returns -- cross-process visibility comes
        # from the page cache, not from durability -- and the manifest that
        # publishes them is only replaced afterwards. fsync would only protect
        # the snapshot across a power loss, and the snapshot is a temp cache
        # that dies with the process that owns it.

    @staticmethod
    def _append_snapshot_columns_intact(
        root: Path,
        *,
        state_count: int,
        object_count: int,
        state_key_width: int,
    ) -> bool:
        expectations = [
            (file_name, int(state_count) * int(np.dtype(dtype).itemsize))
            for _attribute, file_name, dtype in _APPEND_SNAPSHOT_STATE_COLUMNS
        ]
        expectations.extend(
            (file_name, int(object_count) * int(np.dtype(dtype).itemsize))
            for _attribute, file_name, dtype in _APPEND_SNAPSHOT_OBJECT_COLUMNS
        )
        expectations.append(
            (
                _append_snapshot_state_key_name(state_key_width),
                int(state_count) * int(np.dtype(f"<U{max(1, int(state_key_width))}").itemsize),
            )
        )
        for file_name, expected_size in expectations:
            if expected_size <= 0:
                continue
            try:
                actual_size = int((root / file_name).stat().st_size)
            except OSError:
                return False
            # A longer file is fine: the surplus is either crash residue or the
            # tail of a superseded state-key width, and the manifest bounds
            # every read to the committed prefix.
            if actual_size < expected_size:
                return False
        return True

    def append_snapshot_directory(self, root: str | Path) -> int:
        """Extend an append-only snapshot of this store at ``root``.

        Writes only the rows interned since the directory's committed manifest
        and returns the committed state count. Creates the snapshot from
        scratch when ``root`` holds no usable manifest, so a deleted or
        half-written directory heals itself on the next call.
        """
        output_root = Path(root)
        output_root.mkdir(parents=True, exist_ok=True)
        state_count = int(len(self._widths))
        object_count = int(len(self._object_type_ids))

        manifest = self._read_append_snapshot_manifest(output_root)
        committed_states = 0
        committed_objects = 0
        state_key_width = 0
        if manifest is not None:
            committed_states = max(0, int(manifest.get("stateCount", 0) or 0))
            committed_objects = max(0, int(manifest.get("objectCount", 0) or 0))
            state_key_width = max(0, int(manifest.get("stateKeyWidth", 0) or 0))
            # Identity check first. Row i of a snapshot means "state id i+1 of
            # the store that wrote it", so appending to another store's
            # directory silently republishes its rows under this store's ids.
            # The count comparison below only catches the case where the other
            # store was AHEAD; a shorter one appends cleanly and reports
            # success. A missing token means a manifest written before this
            # field existed, which cannot be attributed either way, so it is
            # rebuilt rather than trusted.
            committed_token = manifest.get("storeToken")
            if not isinstance(committed_token, int) or int(committed_token) != int(
                self._store_token
            ):
                committed_states = 0
                committed_objects = 0
                state_key_width = 0
                manifest = None
            elif committed_states > state_count or committed_objects > object_count:
                raise ValueError(
                    f"Append-only snapshot {output_root} already covers "
                    f"{committed_states} states / {committed_objects} object rows, "
                    f"more than this store's {state_count} / {object_count}. "
                    "The directory does not belong to this store's id assignment."
                )
        if manifest is not None:
            if not self._append_snapshot_columns_intact(
                output_root,
                state_count=committed_states,
                object_count=committed_objects,
                state_key_width=state_key_width,
            ):
                committed_states = 0
                committed_objects = 0
                state_key_width = 0
            elif committed_states == state_count and committed_objects == object_count:
                # Nothing to publish: the committed manifest already describes
                # this store exactly. This is the path every protected-set
                # evaluation takes, since it interns nothing.
                return int(state_count)

        appended_keys = [
            self.state_key(state_id)
            for state_id in range(committed_states + 1, state_count + 1)
        ]
        if max((len(key) for key in appended_keys), default=0) > state_key_width:
            # Restride into a NEW file rather than rewriting the committed one:
            # the live manifest still names the old-width file until the commit
            # below, so a concurrent reader keeps indexing bytes nobody touched.
            # Rare in practice because state keys have a fixed length, but
            # `intern` accepts caller-supplied keys, so the branch is reachable.
            state_key_first_row = 0
            state_keys = [
                self.state_key(state_id) for state_id in range(1, state_count + 1)
            ]
            state_key_width = max((len(key) for key in state_keys), default=1)
        else:
            state_key_first_row = committed_states
            state_keys = appended_keys
        state_key_width = max(1, int(state_key_width))

        for attribute, file_name, dtype in _APPEND_SNAPSHOT_STATE_COLUMNS:
            self._append_snapshot_rows(
                output_root / file_name,
                dtype=dtype,
                values=getattr(self, attribute)[committed_states:state_count],
                first_row=committed_states,
            )
        for attribute, file_name, dtype in _APPEND_SNAPSHOT_OBJECT_COLUMNS:
            self._append_snapshot_rows(
                output_root / file_name,
                dtype=dtype,
                values=getattr(self, attribute)[committed_objects:object_count],
                first_row=committed_objects,
            )
        self._append_snapshot_rows(
            output_root / _append_snapshot_state_key_name(state_key_width),
            dtype=f"<U{state_key_width}",
            values=state_keys,
            first_row=state_key_first_row,
        )

        self._commit_append_snapshot_manifest(
            output_root,
            {
                "formatVersion": int(_APPEND_SNAPSHOT_FORMAT_VERSION),
                "storeToken": int(self._store_token),
                "stateCount": int(state_count),
                "objectCount": int(object_count),
                "stateKeyWidth": int(state_key_width),
                "vocab": {
                    manifest_key: [str(value) for value in getattr(self, attribute)]
                    for attribute, manifest_key in _APPEND_SNAPSHOT_VOCABS
                },
            },
        )
        return int(state_count)

    @staticmethod
    def _map_append_snapshot_column(path: Path, *, dtype: str, count: int) -> np.ndarray:
        resolved_count = max(0, int(count))
        if resolved_count == 0:
            return np.empty(0, dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r", shape=(resolved_count,))

    @classmethod
    def load_snapshot_directory(
        cls,
        root: str | Path,
        *,
        state_obj_cache_max_size: int = _CACHE_MAX_SIZE,
        state_json_cache_max_size: int = _CACHE_MAX_SIZE,
        runtime_packet_cache_max_size: int = _CACHE_MAX_SIZE,
    ) -> "StateStore":
        input_root = Path(root).resolve()
        manifest = cls._read_append_snapshot_manifest(input_root)
        if manifest is None:
            raise ValueError(
                f"Unsupported or missing append-only state store manifest: {input_root}"
            )
        state_count = max(0, int(manifest.get("stateCount", 0) or 0))
        object_count = max(0, int(manifest.get("objectCount", 0) or 0))
        state_key_width = max(1, int(manifest.get("stateKeyWidth", 1) or 1))
        vocab = manifest.get("vocab")
        if not isinstance(vocab, dict):
            vocab = {}

        store = cls(
            state_obj_cache_max_size=state_obj_cache_max_size,
            state_json_cache_max_size=state_json_cache_max_size,
            runtime_packet_cache_max_size=runtime_packet_cache_max_size,
        )
        store._readonly = True
        for attribute, file_name, dtype in _APPEND_SNAPSHOT_STATE_COLUMNS:
            setattr(
                store,
                attribute,
                cls._map_append_snapshot_column(
                    input_root / file_name,
                    dtype=dtype,
                    count=state_count,
                ),
            )
        for attribute, file_name, dtype in _APPEND_SNAPSHOT_OBJECT_COLUMNS:
            setattr(
                store,
                attribute,
                cls._map_append_snapshot_column(
                    input_root / file_name,
                    dtype=dtype,
                    count=object_count,
                ),
            )
        # The width is part of the file name, so a reader that mapped an older
        # manifest keeps its own file even after a restride publishes a new one.
        store._state_keys_array = cls._map_append_snapshot_column(
            input_root / _append_snapshot_state_key_name(state_key_width),
            dtype=f"<U{state_key_width}",
            count=state_count,
        )

        for attribute, manifest_key in _APPEND_SNAPSHOT_VOCABS:
            setattr(
                store,
                attribute,
                [str(value) for value in (vocab.get(manifest_key) or [])],
            )
        store._type_id_by_text = {}
        store._word_id_by_text = {}
        store._direction_id_by_text = {}
        store._color_id_by_text = {}
        store._json_id_by_text = {}

        store._state_obj_cache.clear()
        store._state_json_cache.clear()
        store._runtime_packet_cache.clear()
        return store

    def intern(
        self,
        state_json: str,
        *,
        state_key: Optional[str] = None,
    ) -> int:
        self._ensure_mutable_storage()
        resolved_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and str(state_key).strip()
            else canonical_state_key(state_json)
        )
        existing_state_id = self._state_id_by_key.get(resolved_key)
        if existing_state_id is not None:
            return int(existing_state_id)
        state_obj = parse_state_json(state_json)
        return self.intern_state_obj(state_obj, state_key=resolved_key)

    def intern_state_obj(
        self,
        state_obj: Dict[str, Any],
        *,
        state_key: Optional[str] = None,
    ) -> int:
        self._ensure_mutable_storage()
        normalized = normalize_state_payload(state_obj)
        resolved_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and str(state_key).strip()
            else canonical_state_key(normalized)
        )
        existing_state_id = self._state_id_by_key.get(resolved_key)
        if existing_state_id is not None:
            return int(existing_state_id)

        header, encoded_objects = self._encode_state_obj(normalized)
        state_id = int(len(self._widths) + 1)
        self._state_id_by_key[resolved_key] = state_id
        self._state_key_by_id[state_id] = resolved_key
        self._append_state(header=header, encoded_objects=encoded_objects)
        return state_id

    def lookup_state_id(self, state: Any) -> Optional[int]:
        self._ensure_state_key_indexes()
        if isinstance(state, str):
            direct_key = state.strip()
            if direct_key in self._state_id_by_key:
                return self._state_id_by_key[direct_key]
        return self._state_id_by_key.get(canonical_state_key(state))

    def state_key(self, state_id: int) -> str:
        resolved_state_id = int(state_id)
        stored = self._state_key_by_id.get(resolved_state_id)
        if stored is None and self._state_keys_array is not None:
            index = resolved_state_id - 1
            if 0 <= index < len(self._state_keys_array):
                stored = str(self._state_keys_array[index])
        return str(stored) if isinstance(stored, str) else ""

    def state_json(self, state_id: int) -> str:
        resolved_state_id = int(state_id)
        cached = self._state_json_cache.get(resolved_state_id)
        if isinstance(cached, str):
            return cached
        state_json = dump_state_json(self.state_obj(resolved_state_id, copy=False))
        return str(self._state_json_cache.put(resolved_state_id, state_json))

    def state_obj(self, state_id: int, *, copy: bool = True) -> Dict[str, Any]:
        resolved_state_id = int(state_id)
        cached = self._state_obj_cache.get(resolved_state_id)
        if isinstance(cached, dict):
            return _clone_plain_structure(cached) if copy else cached
        state_obj = self._decode_state_obj(resolved_state_id)
        self._state_obj_cache.put(resolved_state_id, state_obj)
        return _clone_plain_structure(state_obj) if copy else state_obj

    def state_token_view(self, state_id: int) -> StateTokenView:
        index = int(state_id) - 1
        if index < 0 or index >= len(self._widths):
            raise KeyError(f"Unknown state_id: {int(state_id)}")
        state_flags = int(self._state_flags[index])
        return StateTokenView(
            width=max(1, int(self._widths[index])) if state_flags & _STATE_HAS_GRID_SIZE else 1,
            height=max(1, int(self._heights[index])) if state_flags & _STATE_HAS_GRID_SIZE else 1,
            object_offset=int(self._object_offsets[index]),
            object_count=int(self._object_counts[index]),
            object_flags=self._object_flags,
            object_type_ids=self._object_type_ids,
            object_word_ids=self._object_word_ids,
            object_x=self._object_x,
            object_y=self._object_y,
            object_direction_ids=self._object_direction_ids,
            type_texts=self._type_text_by_id,
            word_texts=self._word_text_by_id,
            direction_texts=self._direction_text_by_id,
            required_object_flags=(
                _OBJECT_HAS_TYPE | _OBJECT_HAS_WORD | _OBJECT_HAS_POSITION
            ),
            direction_object_flag=_OBJECT_HAS_DIRECTION,
            null_sentinel=_NULL_SENTINEL,
        )

    def runtime_state_vocab(self) -> RuntimeStateVocab:
        return RuntimeStateVocab(
            type_texts=tuple(str(value) for value in self._type_text_by_id),
            word_texts=tuple(str(value) for value in self._word_text_by_id),
        )

    def runtime_state_packet(self, state_id: int) -> RuntimeStatePacket:
        index = int(state_id) - 1
        if index < 0 or index >= len(self._widths):
            raise KeyError(f"Unknown state_id: {int(state_id)}")
        cached = self._runtime_packet_cache.get(int(state_id))
        if isinstance(cached, RuntimeStatePacket):
            return cached

        state_flags = int(self._state_flags[index])
        step_flags = int(self._step_flags[index])
        object_offset = int(self._object_offsets[index])
        object_count = int(self._object_counts[index])
        rows: List[RuntimeObjectRow] = []
        for object_index in range(object_offset, object_offset + object_count):
            object_flags = int(self._object_flags[object_index])
            if not (
                object_flags & _OBJECT_HAS_TYPE
                and object_flags & _OBJECT_HAS_WORD
                and object_flags & _OBJECT_HAS_POSITION
            ):
                continue
            direction = _NO_RUNTIME_DIRECTION
            direction_id = int(self._object_direction_ids[object_index])
            if object_flags & _OBJECT_HAS_DIRECTION and direction_id != _NULL_SENTINEL:
                direction_text = self._resolve_text_vocab(
                    self._direction_text_by_id,
                    direction_id,
                )
                direction = int(
                    _RUNTIME_ID_BY_DIRECTION_TEXT.get(
                        str(direction_text),
                        _NO_RUNTIME_DIRECTION,
                    )
                )
            rows.append(
                RuntimeObjectRow(
                    x=int(self._object_x[object_index]),
                    y=int(self._object_y[object_index]),
                    type_id=int(self._object_type_ids[object_index]),
                    word_id=int(self._object_word_ids[object_index]),
                    direction=int(direction),
                )
            )
        packet = RuntimeStatePacket(
            width=int(self._widths[index]) if state_flags & _STATE_HAS_GRID_SIZE else 0,
            height=int(self._heights[index]) if state_flags & _STATE_HAS_GRID_SIZE else 0,
            terminated=bool(self._terminated[index])
            if step_flags & _STEP_HAS_TERMINATED
            else False,
            objects=tuple(rows),
        )
        return self._runtime_packet_cache.put(int(state_id), packet)

    def _runtime_state_packet_obj(self, packet: RuntimeStatePacket) -> Dict[str, Any]:
        return runtime_state_packet_obj(packet, self.runtime_state_vocab())

    def intern_runtime_state_packet(
        self,
        packet: RuntimeStatePacket,
        *,
        state_key: Optional[str] = None,
    ) -> int:
        self._ensure_mutable_storage()
        resolved_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and str(state_key).strip()
            else ""
        )
        if not resolved_key:
            resolved_key = runtime_state_packet_key(packet, self.runtime_state_vocab())
        existing_state_id = self._state_id_by_key.get(resolved_key)
        if existing_state_id is not None:
            return int(existing_state_id)

        encoded_objects: List[Tuple[int, int, int, int, int, int, int, int]] = []
        for row in packet.objects:
            if int(row.type_id) < 0 or int(row.type_id) >= len(self._type_text_by_id):
                raise KeyError(
                    "Runtime state packet references an unknown state-store vocab id."
                )
            if int(row.word_id) < 0 or int(row.word_id) >= len(self._word_text_by_id):
                raise KeyError(
                    "Runtime state packet references an unknown state-store vocab id."
                )
            object_flags = _OBJECT_HAS_TYPE | _OBJECT_HAS_WORD | _OBJECT_HAS_POSITION
            direction_id = _NULL_SENTINEL
            if int(row.direction) != _NO_RUNTIME_DIRECTION:
                direction_index = int(row.direction)
                if direction_index < 0 or direction_index >= len(_DIRECTION_TEXT_BY_RUNTIME_ID):
                    raise KeyError(
                        f"Runtime state packet references an unknown direction id: {direction_index}"
                    )
                direction_id = self._intern_text_vocab(
                    _DIRECTION_TEXT_BY_RUNTIME_ID[direction_index],
                    mapping=self._direction_id_by_text,
                    values=self._direction_text_by_id,
                )
                object_flags |= _OBJECT_HAS_DIRECTION
            encoded_objects.append(
                (
                    int(object_flags),
                    int(row.type_id),
                    int(row.word_id),
                    int(row.x),
                    int(row.y),
                    int(direction_id),
                    int(_NULL_SENTINEL),
                    0,
                )
            )

        state_id = int(len(self._widths) + 1)
        self._state_id_by_key[resolved_key] = state_id
        self._state_key_by_id[state_id] = resolved_key
        self._append_state(
            header=(
                int(packet.width),
                int(packet.height),
                int(_STATE_HAS_GRID_SIZE | _STATE_HAS_STEP | _STATE_HAS_OBJECTS),
                int(_STEP_HAS_TERMINATED),
                bool(packet.terminated),
                0,
                0,
            ),
            encoded_objects=encoded_objects,
        )
        self._runtime_packet_cache.put(int(state_id), packet)
        return state_id

    def _encode_state_obj(
        self,
        state_obj: Mapping[str, Any],
    ) -> Tuple[
        Tuple[int, int, int, int, bool, int, int],
        List[Tuple[int, int, int, int, int, int, int, int]],
    ]:
        raw_grid_size = state_obj.get("grid_size")
        state_flags = 0
        width = 0
        height = 0
        if (
            "grid_size" in state_obj
            and
            isinstance(raw_grid_size, list)
            and len(raw_grid_size) == 2
            and isinstance(raw_grid_size[0], int)
            and isinstance(raw_grid_size[1], int)
        ):
            width = int(raw_grid_size[0])
            height = int(raw_grid_size[1])
            state_flags |= _STATE_HAS_GRID_SIZE

        raw_step = state_obj.get("step")
        step_flags = 0
        terminated = False
        step_extra_id = 0
        handled_top_level_keys = set()
        if state_flags & _STATE_HAS_GRID_SIZE:
            handled_top_level_keys.add("grid_size")
        if isinstance(raw_step, Mapping):
            state_flags |= _STATE_HAS_STEP
            handled_top_level_keys.add("step")
            if "terminated" in raw_step:
                terminated = bool(raw_step.get("terminated", False))
                step_flags |= _STEP_HAS_TERMINATED
            step_extras = {
                str(key): value
                for key, value in raw_step.items()
                if str(key) not in {"terminated", "truncated"}
            }
            step_extra_id = self._intern_json_blob(step_extras)

        state_extras = {
            str(key): value
            for key, value in state_obj.items()
            if str(key) not in handled_top_level_keys
        }
        raw_objects = state_obj.get("objects")
        if isinstance(raw_objects, list):
            state_flags |= _STATE_HAS_OBJECTS
            handled_top_level_keys.add("objects")
            state_extras = {
                str(key): value
                for key, value in state_obj.items()
                if str(key) not in handled_top_level_keys
            }
        state_extra_id = self._intern_json_blob(state_extras)

        encoded_objects: List[Tuple[int, int, int, int, int, int, int]] = []
        if isinstance(raw_objects, list):
            for raw_object in raw_objects:
                row = raw_object if isinstance(raw_object, Mapping) else {}
                object_flags = 0
                handled_object_keys = set()

                raw_type = row.get("type")
                type_id = 0
                if "type" in row and isinstance(raw_type, str):
                    type_id = self._intern_text_vocab(
                        str(raw_type),
                        mapping=self._type_id_by_text,
                        values=self._type_text_by_id,
                    )
                    object_flags |= _OBJECT_HAS_TYPE
                    handled_object_keys.add("type")

                raw_word = row.get("word")
                word_id = 0
                if "word" in row and isinstance(raw_word, str):
                    word_id = self._intern_text_vocab(
                        str(raw_word),
                        mapping=self._word_id_by_text,
                        values=self._word_text_by_id,
                    )
                    object_flags |= _OBJECT_HAS_WORD
                    handled_object_keys.add("word")

                position = row.get("position")
                x = 0
                y = 0
                if (
                    "position" in row
                    and
                    isinstance(position, list)
                    and len(position) == 2
                    and isinstance(position[0], int)
                    and isinstance(position[1], int)
                ):
                    x = int(position[0])
                    y = int(position[1])
                    object_flags |= _OBJECT_HAS_POSITION
                    handled_object_keys.add("position")
                direction_id = _NULL_SENTINEL
                if "direction" in row and isinstance(row.get("direction"), str):
                    direction_id = self._intern_text_vocab(
                        str(row.get("direction")),
                        mapping=self._direction_id_by_text,
                        values=self._direction_text_by_id,
                    )
                    object_flags |= _OBJECT_HAS_DIRECTION
                    handled_object_keys.add("direction")
                color_id = _NULL_SENTINEL
                if "color" in row and isinstance(row.get("color"), str):
                    color_id = self._intern_text_vocab(
                        str(row.get("color")),
                        mapping=self._color_id_by_text,
                        values=self._color_text_by_id,
                    )
                    object_flags |= _OBJECT_HAS_COLOR
                    handled_object_keys.add("color")
                object_extras = {
                    str(key): value
                    for key, value in row.items()
                    if str(key) not in handled_object_keys
                }
                encoded_objects.append(
                    (
                        int(object_flags),
                        int(type_id),
                        int(word_id),
                        int(x),
                        int(y),
                        int(direction_id),
                        int(color_id),
                        self._intern_json_blob(object_extras),
                    )
                )

        header = (
            int(width),
            int(height),
            int(state_flags),
            int(step_flags),
            bool(terminated),
            int(step_extra_id),
            int(state_extra_id),
        )
        return header, encoded_objects

    def _append_state(
        self,
        *,
        header: Tuple[int, int, int, int, bool, int, int],
        encoded_objects: Iterable[Tuple[int, int, int, int, int, int, int, int]],
    ) -> None:
        width, height, state_flags, step_flags, terminated, step_extra_id, state_extra_id = header
        object_offset = int(len(self._object_type_ids))
        object_count = 0
        for (
            object_flags,
            type_id,
            word_id,
            x,
            y,
            direction_id,
            color_id,
            object_extra_id,
        ) in encoded_objects:
            self._object_flags.append(int(object_flags))
            self._object_type_ids.append(int(type_id))
            self._object_word_ids.append(int(word_id))
            self._object_x.append(int(x))
            self._object_y.append(int(y))
            self._object_direction_ids.append(int(direction_id))
            self._object_color_ids.append(int(color_id))
            self._object_extra_ids.append(int(object_extra_id))
            object_count += 1

        self._widths.append(int(width))
        self._heights.append(int(height))
        self._state_flags.append(int(state_flags))
        self._step_flags.append(int(step_flags))
        self._terminated.append(1 if bool(terminated) else 0)
        self._object_offsets.append(int(object_offset))
        self._object_counts.append(int(object_count))
        self._step_extra_ids.append(int(step_extra_id))
        self._state_extra_ids.append(int(state_extra_id))

    def _decode_state_obj(self, state_id: int) -> Dict[str, Any]:
        index = int(state_id) - 1
        if index < 0 or index >= len(self._widths):
            raise KeyError(f"Unknown state_id: {int(state_id)}")

        state: Dict[str, Any] = self._decode_json_blob(int(self._state_extra_ids[index]))
        state_flags = int(self._state_flags[index])
        if state_flags & _STATE_HAS_GRID_SIZE:
            state["grid_size"] = [int(self._widths[index]), int(self._heights[index])]
        if state_flags & _STATE_HAS_STEP:
            step_payload = self._decode_json_blob(int(self._step_extra_ids[index]))
            if int(self._step_flags[index]) & _STEP_HAS_TERMINATED:
                step_payload["terminated"] = bool(self._terminated[index])
            state["step"] = step_payload

        if state_flags & _STATE_HAS_OBJECTS:
            state["objects"] = self._decode_object_rows(
                int(self._object_offsets[index]),
                int(self._object_counts[index]),
            )
        return state

    def _decode_object_rows(
        self,
        object_offset: int,
        object_count: int,
    ) -> List[Dict[str, Any]]:
        """Decode one state's object rows from bulk column slices.

        Slicing each column once and materializing it with ``tolist()`` replaces
        eight per-row scalar reads (each of which allocates a numpy scalar under
        memmap) with eight vectorized reads per state.
        """
        if object_count <= 0:
            return []
        stop = object_offset + object_count
        flags_column = self._object_flags[object_offset:stop].tolist()
        extra_column = self._object_extra_ids[object_offset:stop].tolist()
        type_column = self._object_type_ids[object_offset:stop].tolist()
        word_column = self._object_word_ids[object_offset:stop].tolist()
        x_column = self._object_x[object_offset:stop].tolist()
        y_column = self._object_y[object_offset:stop].tolist()
        direction_column = self._object_direction_ids[object_offset:stop].tolist()
        color_column = self._object_color_ids[object_offset:stop].tolist()

        type_texts = self._type_text_by_id
        word_texts = self._word_text_by_id
        direction_texts = self._direction_text_by_id
        color_texts = self._color_text_by_id
        type_count = len(type_texts)
        word_count = len(word_texts)
        direction_count = len(direction_texts)
        color_count = len(color_texts)
        blob_zero_is_empty = bool(self._json_text_by_id) and (
            self._json_text_by_id[0] == _EMPTY_JSON_OBJECT
        )

        rows: List[Dict[str, Any]] = []
        append_row = rows.append
        for object_flags, extra_id, type_id, word_id, x, y, direction_id, color_id in zip(
            flags_column,
            extra_column,
            type_column,
            word_column,
            x_column,
            y_column,
            direction_column,
            color_column,
        ):
            if (
                object_flags == _OBJECT_FULL_FLAGS
                and extra_id == 0
                and blob_zero_is_empty
                and type_id < type_count
                and word_id < word_count
                and direction_id != _NULL_SENTINEL
                and direction_id < direction_count
            ):
                append_row(
                    {
                        "type": type_texts[type_id],
                        "word": word_texts[word_id],
                        "position": [x, y],
                        "direction": direction_texts[direction_id],
                    }
                )
                continue
            row = self._decode_json_blob(extra_id)
            if object_flags & _OBJECT_HAS_TYPE:
                row["type"] = type_texts[type_id] if type_id < type_count else ""
            if object_flags & _OBJECT_HAS_WORD:
                row["word"] = word_texts[word_id] if word_id < word_count else ""
            if object_flags & _OBJECT_HAS_POSITION:
                row["position"] = [x, y]
            if object_flags & _OBJECT_HAS_DIRECTION and direction_id != _NULL_SENTINEL:
                row["direction"] = (
                    direction_texts[direction_id] if direction_id < direction_count else ""
                )
            if object_flags & _OBJECT_HAS_COLOR and color_id != _NULL_SENTINEL:
                row["color"] = color_texts[color_id] if color_id < color_count else ""
            append_row(row)
        return rows

    def _intern_text_vocab(
        self,
        text: str,
        *,
        mapping: MutableMapping[str, int],
        values: List[str],
    ) -> int:
        resolved_text = str(text)
        existing = mapping.get(resolved_text)
        if existing is not None:
            return int(existing)
        next_id = int(len(values))
        mapping[resolved_text] = next_id
        values.append(resolved_text)
        return next_id

    def _resolve_text_vocab(self, values: List[str], index: int) -> str:
        resolved_index = int(index)
        if resolved_index < 0 or resolved_index >= len(values):
            return ""
        return str(values[resolved_index])

    def _intern_json_blob(self, payload: Mapping[str, Any]) -> int:
        if not payload:
            return 0
        text = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = self._json_id_by_text.get(text)
        if existing is not None:
            return int(existing)
        next_id = int(len(self._json_text_by_id))
        self._json_id_by_text[text] = next_id
        self._json_text_by_id.append(text)
        return next_id

    def _decode_json_blob(self, blob_id: int) -> Dict[str, Any]:
        resolved_blob_id = int(blob_id)
        if resolved_blob_id < 0 or resolved_blob_id >= len(self._json_text_by_id):
            return {}
        text = self._json_text_by_id[resolved_blob_id]
        if text == _EMPTY_JSON_OBJECT:
            return {}
        decoded = json.loads(text)
        return dict(decoded) if isinstance(decoded, dict) else {}
