"""
Transition Buffer

Transition data (state, action, next_state) management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from numbers import Integral
import random
from typing import Any, Dict, List, Optional, Tuple

from .state_store import StateStore, _clone_plain_structure, canonical_state_key
from .state_schema import dump_state_json, normalize_state_payload, parse_state_json


def _positive_integral(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    resolved = int(value)
    return resolved if resolved > 0 else None


@dataclass(slots=True, init=False, eq=False)
class Transition:
    """Single transition record."""

    action: str
    reward: float = 0.0
    done: bool = False
    world_index: Optional[int] = None
    map_name: Optional[str] = None
    state_store: Optional[StateStore] = field(default=None, repr=False)
    state_id: Optional[int] = field(default=None, repr=False)
    next_state_id: Optional[int] = field(default=None, repr=False)
    _state_json: Optional[str] = field(default=None, repr=False)
    _next_state_json: Optional[str] = field(default=None, repr=False)
    _state_obj: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _next_state_obj: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _state_key: Optional[str] = field(default=None, repr=False)
    _next_state_key: Optional[str] = field(default=None, repr=False)

    def __init__(
        self,
        state: str,
        action: str,
        next_state: str,
        reward: float = 0.0,
        done: bool = False,
        world_index: Optional[int] = None,
        map_name: Optional[str] = None,
        *,
        state_store: Optional[StateStore] = None,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
    ) -> None:
        resolved_store = state_store if isinstance(state_store, StateStore) else None
        resolved_state_id = _positive_integral(state_id)
        resolved_next_state_id = _positive_integral(next_state_id)

        state_text = str(state).strip()
        next_state_text = str(next_state).strip()
        if resolved_store is not None and resolved_state_id is None and state_text:
            resolved_state_id = int(resolved_store.intern(state_text))
        if resolved_store is not None and resolved_next_state_id is None and next_state_text:
            resolved_next_state_id = int(resolved_store.intern(next_state_text))

        self.action = str(action)
        self.reward = float(reward)
        self.done = bool(done)
        self.world_index = _positive_integral(world_index)
        self.map_name = (
            str(map_name).strip()
            if isinstance(map_name, str) and str(map_name).strip()
            else None
        )
        self.state_store = resolved_store
        self.state_id = resolved_state_id
        self.next_state_id = resolved_next_state_id
        self._state_json = None if resolved_state_id is not None else state_text
        self._next_state_json = None if resolved_next_state_id is not None else next_state_text
        self._state_obj = None
        self._next_state_obj = None
        self._state_key = None
        self._next_state_key = None

    @classmethod
    def from_state_ids(
        cls,
        *,
        state_store: StateStore,
        state_id: int,
        action: str,
        next_state_id: int,
        reward: float = 0.0,
        done: bool = False,
        world_index: Optional[int] = None,
        map_name: Optional[str] = None,
    ) -> "Transition":
        return cls(
            state="",
            action=action,
            next_state="",
            reward=reward,
            done=done,
            world_index=world_index,
            map_name=map_name,
            state_store=state_store,
            state_id=int(state_id),
            next_state_id=int(next_state_id),
        )

    @classmethod
    def from_state_objects(
        cls,
        *,
        state_obj: Dict[str, Any],
        action: str,
        next_state_obj: Dict[str, Any],
        reward: float = 0.0,
        done: bool = False,
        world_index: Optional[int] = None,
        map_name: Optional[str] = None,
        state_key: Optional[str] = None,
        next_state_key: Optional[str] = None,
    ) -> "Transition":
        transition = cls(
            state="",
            action=action,
            next_state="",
            reward=reward,
            done=done,
            world_index=world_index,
            map_name=map_name,
        )
        transition._state_obj = normalize_state_payload(state_obj)
        transition._next_state_obj = normalize_state_payload(next_state_obj)
        transition._state_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and str(state_key).strip()
            else canonical_state_key(transition._state_obj)
        )
        transition._next_state_key = (
            str(next_state_key).strip()
            if isinstance(next_state_key, str) and str(next_state_key).strip()
            else canonical_state_key(transition._next_state_obj)
        )
        return transition

    @property
    def state(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.state_id, int):
            resolved = self.state_store.state_json(int(self.state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        if isinstance(self._state_obj, dict):
            if not isinstance(self._state_json, str) or not self._state_json:
                self._state_json = dump_state_json(self._state_obj)
            return str(self._state_json)
        return str(self._state_json) if isinstance(self._state_json, str) else ""

    @property
    def next_state(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.next_state_id, int):
            resolved = self.state_store.state_json(int(self.next_state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        if isinstance(self._next_state_obj, dict):
            if not isinstance(self._next_state_json, str) or not self._next_state_json:
                self._next_state_json = dump_state_json(self._next_state_obj)
            return str(self._next_state_json)
        return str(self._next_state_json) if isinstance(self._next_state_json, str) else ""

    @property
    def state_key(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.state_id, int):
            resolved = self.state_store.state_key(int(self.state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        if isinstance(self._state_key, str) and self._state_key:
            return self._state_key
        return canonical_state_key(self.state)

    @property
    def next_state_key(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.next_state_id, int):
            resolved = self.state_store.state_key(int(self.next_state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        if isinstance(self._next_state_key, str) and self._next_state_key:
            return self._next_state_key
        return canonical_state_key(self.next_state)

    def state_obj(self) -> Dict[str, Any]:
        if isinstance(self.state_store, StateStore) and isinstance(self.state_id, int):
            return self.state_store.state_obj(int(self.state_id))
        if isinstance(self._state_obj, dict):
            return _clone_plain_structure(self._state_obj)
        return parse_state_json(self.state)

    def next_state_obj(self) -> Dict[str, Any]:
        if isinstance(self.state_store, StateStore) and isinstance(self.next_state_id, int):
            return self.state_store.state_obj(int(self.next_state_id))
        if isinstance(self._next_state_obj, dict):
            return _clone_plain_structure(self._next_state_obj)
        return parse_state_json(self.next_state)

    def to_dict(self) -> dict:
        payload = {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
        }
        if _positive_integral(self.world_index) is not None:
            payload["world_index"] = int(self.world_index)
        if isinstance(self.map_name, str) and self.map_name.strip():
            payload["map_name"] = str(self.map_name).strip()
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "Transition":
        return cls(**data)


GraphEdgeIdentityFields = Tuple[str, str, str]


def canonical_graph_world_scope(
    *,
    world_index: Any = None,
    map_name: Any = None,
    world_identity: Any = None,
) -> str:
    if isinstance(world_identity, str) and world_identity.strip():
        return str(world_identity).strip()
    resolved_world_index = _positive_integral(world_index)
    if resolved_world_index is not None:
        return f"world:{int(resolved_world_index)}"
    if isinstance(map_name, str) and map_name.strip():
        return f"map:{map_name.strip()}"
    return "world:unknown"


def canonical_graph_edge_identity_fields(
    *,
    world_index: Any = None,
    map_name: Any = None,
    world_identity: Any = None,
    state: Any,
    action: Any,
) -> GraphEdgeIdentityFields:
    return (
        canonical_graph_world_scope(
            world_index=world_index,
            map_name=map_name,
            world_identity=world_identity,
        ),
        canonical_state_key(state),
        str(action),
    )


def canonical_graph_edge_identity_key_from_fields(
    *,
    world_index: Any = None,
    map_name: Any = None,
    world_identity: Any = None,
    state: Any,
    action: Any,
) -> str:
    world_text, state_text, action_text = canonical_graph_edge_identity_fields(
        world_index=world_index,
        map_name=map_name,
        world_identity=world_identity,
        state=state,
        action=action,
    )
    raw = f"{world_text}|{state_text}|{action_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def canonical_graph_edge_identity_key(transition: Transition) -> str:
    return canonical_graph_edge_identity_key_from_fields(
        world_index=transition.world_index,
        map_name=transition.map_name,
        state=transition.state_key,
        action=transition.action,
    )


class TransitionBuffer:
    """Transition data buffer."""

    def __init__(self, max_size: Optional[int] = None):
        self.buffer: List[Transition] = []
        self.max_size = max_size

    def add(self, transition: Transition) -> None:
        if self.max_size is not None and len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def extend(self, transitions: List[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, n: int) -> List[Transition]:
        n = min(max(0, int(n)), len(self.buffer))
        return random.sample(self.buffer, n)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def get_all(self) -> List[Transition]:
        return list(self.buffer)
