"""
Canonical Dataset

Canonical transition store used for rule evaluation.
State JSON is interned in the shared StateStore, and transitions are tracked by map-local graph edge identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .state_store import StateStore
from .transition_buffer import (
    Transition,
    TransitionBuffer,
    canonical_graph_edge_identity_key,
)


@dataclass
class _StoredTransition:
    state_id: int
    action: str
    next_state_id: int
    reward: float
    done: bool
    world_index: Optional[int] = None
    map_name: Optional[str] = None


class CanonicalDataset(TransitionBuffer):
    """Canonical D: map-local graph-edge dataset backed by interned state JSON."""

    def __init__(
        self,
        max_size: Optional[int] = None,
        *,
        state_store: Optional[StateStore] = None,
        on_change: Optional[Callable[[], None]] = None,
    ):
        super().__init__(max_size=max_size)
        self._keys: Set[str] = set()
        self._ordered_keys: List[str] = []
        self._index_by_key: Dict[str, int] = {}
        self._records_by_key: Dict[str, _StoredTransition] = {}
        self._transition_views_by_key: Dict[str, Transition] = {}
        self._owns_state_store = state_store is None
        self._state_store = state_store if state_store is not None else StateStore()
        self._on_change = on_change

    @property
    def state_store(self) -> StateStore:
        return self._state_store

    def _compute_key(self, transition: Transition) -> str:
        return canonical_graph_edge_identity_key(transition)

    def transition_keys(self) -> List[str]:
        return list(self._ordered_keys)

    def contains_key(self, transition_key: str) -> bool:
        return str(transition_key) in self._keys

    def transition_for_key(self, transition_key: str) -> Optional[Transition]:
        safe_key = str(transition_key)
        if safe_key not in self._keys:
            return None
        return self._transition_views_by_key.get(safe_key)

    def _emit_change(self) -> None:
        callback = self._on_change
        if callable(callback):
            callback()

    def _add_with_details_and_view(
        self,
        transition: Transition,
        *,
        emit_change: bool,
    ) -> Tuple[bool, Optional[Transition], Optional[Transition]]:
        key = self._compute_key(transition)
        if key in self._keys:
            self._validate_existing_edge_outcome(key=key, transition=transition)
            self._merge_transition_artifacts(key=key, transition=transition)
            return False, None, None

        state_id = self._resolve_transition_state_id(transition, next_state=False)
        next_state_id = self._resolve_transition_state_id(transition, next_state=True)
        record = _StoredTransition(
            state_id=state_id,
            action=str(transition.action),
            next_state_id=next_state_id,
            reward=float(transition.reward),
            done=bool(transition.done),
            world_index=(
                int(transition.world_index)
                if isinstance(transition.world_index, int) and int(transition.world_index) > 0
                else None
            ),
            map_name=(
                str(transition.map_name).strip()
                if isinstance(transition.map_name, str) and str(transition.map_name).strip()
                else None
            ),
        )
        self._records_by_key[key] = record
        view = self._build_transition_view(record)
        self._transition_views_by_key[key] = view
        self._keys.add(key)
        self._ordered_keys.append(key)
        self.buffer.append(view)
        self._index_by_key[key] = len(self._ordered_keys) - 1

        removed_transition: Optional[Transition] = None
        if self.max_size and len(self.buffer) > self.max_size:
            removed_key = self._ordered_keys.pop(0)
            removed_transition = self.buffer.pop(0)
            self._keys.discard(removed_key)
            self._index_by_key.pop(removed_key, None)
            self._drop_record(removed_key)
            self._reindex_from(0)

        if emit_change:
            self._emit_change()
        return True, removed_transition, view

    def add_with_details(self, transition: Transition) -> Tuple[bool, Optional[Transition]]:
        added, removed_transition, _view = self._add_with_details_and_view(
            transition,
            emit_change=True,
        )
        return added, removed_transition

    def add(self, transition: Transition) -> bool:
        added, _removed = self.add_with_details(transition)
        return bool(added)

    def merge(self, transitions: List[Transition]) -> int:
        added = 0
        for transition in transitions:
            if self.add(transition):
                added += 1
        return added

    def merge_with_details(
        self,
        transitions: List[Transition],
    ) -> Tuple[int, List[Transition], List[Transition]]:
        added_transitions: List[Transition] = []
        removed_transitions: List[Transition] = []
        for transition in transitions:
            added, removed, view = self._add_with_details_and_view(
                transition,
                emit_change=False,
            )
            if not added:
                continue
            if view is not None:
                added_transitions.append(view)
            if removed is not None:
                removed_transitions.append(removed)
        if added_transitions or removed_transitions:
            self._emit_change()
        return len(added_transitions), added_transitions, removed_transitions

    def discard(self, transition: Transition) -> bool:
        target_key = self._compute_key(transition)
        if target_key not in self._keys:
            return False
        index = self._index_by_key.get(target_key)
        if index is None:
            return False
        self._ordered_keys.pop(index)
        self.buffer.pop(index)
        self._keys.discard(target_key)
        self._index_by_key.pop(target_key, None)
        self._drop_record(target_key)
        self._reindex_from(index)
        self._emit_change()
        return True

    def clear(self) -> None:
        had_items = bool(self._keys)
        super().clear()
        self._keys.clear()
        self._ordered_keys.clear()
        self._index_by_key.clear()
        self._records_by_key.clear()
        self._transition_views_by_key.clear()
        if self._owns_state_store:
            self._state_store = StateStore()
        if had_items:
            self._emit_change()

    def sample(self, n: int) -> List[Transition]:
        n = min(max(0, int(n)), len(self.buffer))
        return random.sample(self.buffer, n)

    def get_all(self) -> List[Transition]:
        return list(self.buffer)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.clear()
        for payload in data:
            if isinstance(payload, dict):
                self.add(Transition.from_dict(payload))

    def _drop_record(self, transition_key: str) -> None:
        self._records_by_key.pop(str(transition_key), None)
        self._transition_views_by_key.pop(str(transition_key), None)

    def _merge_transition_artifacts(self, *, key: str, transition: Transition) -> None:
        record = self._records_by_key.get(str(key))
        if record is None:
            return
        changed = False
        if (
            record.world_index is None
            and isinstance(transition.world_index, int)
            and int(transition.world_index) > 0
        ):
            record.world_index = int(transition.world_index)
            changed = True
        if (
            record.map_name is None
            and isinstance(transition.map_name, str)
            and str(transition.map_name).strip()
        ):
            record.map_name = str(transition.map_name).strip()
            changed = True
        if not changed:
            return
        self._transition_views_by_key[str(key)] = self._build_transition_view(record)
        index = self._index_by_key.get(str(key))
        if index is None:
            return
        self.buffer[index] = self._transition_views_by_key[str(key)]

    def _validate_existing_edge_outcome(self, *, key: str, transition: Transition) -> None:
        record = self._records_by_key.get(str(key))
        if record is None:
            return
        existing_next_key = self._state_store.state_key(int(record.next_state_id))
        incoming_next_key = transition.next_state_key
        if existing_next_key != incoming_next_key or bool(record.done) != bool(transition.done):
            raise ValueError(
                "CanonicalDataset graph edge conflict: the same map-local "
                "state/action produced a different outcome."
            )

    def _reindex_from(self, start_index: int) -> None:
        for index in range(max(0, int(start_index)), len(self._ordered_keys)):
            self._index_by_key[self._ordered_keys[index]] = index

    def _resolve_transition_state_id(self, transition: Transition, *, next_state: bool) -> int:
        state_id = transition.next_state_id if next_state else transition.state_id
        if (
            isinstance(transition.state_store, StateStore)
            and transition.state_store is self._state_store
            and isinstance(state_id, int)
            and int(state_id) > 0
        ):
            return int(state_id)
        state_json = transition.next_state if next_state else transition.state
        state_key = transition.next_state_key if next_state else transition.state_key
        return int(self._state_store.intern(state_json, state_key=state_key))

    def _build_transition_view(self, record: _StoredTransition) -> Transition:
        return Transition.from_state_ids(
            state_store=self._state_store,
            state_id=int(record.state_id),
            action=str(record.action),
            next_state_id=int(record.next_state_id),
            reward=float(record.reward),
            done=bool(record.done),
            world_index=(
                int(record.world_index)
                if isinstance(record.world_index, int) and int(record.world_index) > 0
                else None
            ),
            map_name=(
                str(record.map_name).strip()
                if isinstance(record.map_name, str) and str(record.map_name).strip()
                else None
            ),
        )
