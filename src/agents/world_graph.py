"""Map-local world graph primitives for graph-contrastive discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True, slots=True)
class EdgeRef:
    world_index: int
    edge_id: int


@dataclass(slots=True)
class GraphEdge:
    ref: EdgeRef
    source_state_id: int
    action: int
    next_state_id: int
    done: bool
    class_id: Optional[int] = None
    rh: float = 0.0
    rz: float = 0.0
    rtotal: float = 0.0
    trainable: bool = True
    canonical: bool = False
    protected: bool = False


@dataclass(slots=True)
class WorldGraph:
    world_index: int
    world_seed: int
    world_label: str = ""
    state_id_by_key: Dict[str, int] = field(default_factory=dict)
    state_key_by_id: Dict[int, str] = field(default_factory=dict)
    edges: Dict[int, GraphEdge] = field(default_factory=dict)
    edge_ids_by_source_state: Dict[int, Dict[int, int]] = field(default_factory=dict)
    trainable_edge_ids: list[int] = field(default_factory=list)
    canonical_edge_ids: set[int] = field(default_factory=set)
    protected_edge_ids: set[int] = field(default_factory=set)
    _next_edge_id: int = 1

    def register_state(self, *, state_id: int, state_key: str) -> int:
        resolved_state_id = int(state_id)
        resolved_state_key = str(state_key)
        existing_state_id = self.state_id_by_key.get(resolved_state_key)
        if existing_state_id is not None:
            if int(existing_state_id) != resolved_state_id:
                raise ValueError(
                    "WorldGraph state conflict: one map-local state key was "
                    "assigned to multiple state ids."
                )
            return int(existing_state_id)
        existing_state_key = self.state_key_by_id.get(resolved_state_id)
        if existing_state_key is not None and existing_state_key != resolved_state_key:
            raise ValueError(
                "WorldGraph state conflict: one state id was assigned to "
                "multiple map-local state keys."
            )
        self.state_id_by_key[resolved_state_key] = resolved_state_id
        self.state_key_by_id[resolved_state_id] = resolved_state_key
        return resolved_state_id

    def register_edge(
        self,
        *,
        source_state_id: int,
        action: int,
        next_state_id: int,
        done: bool,
        class_id: Optional[int] = None,
        rh: float = 0.0,
        rz: float = 0.0,
        rtotal: float = 0.0,
        trainable: bool = True,
    ) -> EdgeRef:
        source_id = int(source_state_id)
        action_id = int(action)
        next_id = int(next_state_id)
        edge_ids_by_action = self.edge_ids_by_source_state.get(source_id)
        existing_edge_id = (
            edge_ids_by_action.get(action_id)
            if edge_ids_by_action is not None
            else None
        )
        if existing_edge_id is not None:
            edge = self.edges[int(existing_edge_id)]
            if int(edge.next_state_id) != next_id or bool(edge.done) != bool(done):
                raise ValueError(
                    "WorldGraph edge conflict: one map-local state/action "
                    "produced multiple outcomes."
                )
            was_trainable = bool(edge.trainable)
            self._merge_edge_metadata(
                edge=edge,
                class_id=class_id,
                rh=rh,
                rz=rz,
                rtotal=rtotal,
                trainable=trainable,
            )
            if bool(edge.trainable) and not was_trainable:
                self.trainable_edge_ids.append(int(existing_edge_id))
            return edge.ref

        edge_id = int(self._next_edge_id)
        self._next_edge_id += 1
        ref = EdgeRef(world_index=int(self.world_index), edge_id=edge_id)
        edge = GraphEdge(
            ref=ref,
            source_state_id=source_id,
            action=action_id,
            next_state_id=next_id,
            done=bool(done),
            class_id=class_id if isinstance(class_id, int) and int(class_id) > 0 else None,
            rh=float(rh),
            rz=float(rz),
            rtotal=float(rtotal),
            trainable=bool(trainable),
        )
        self.edges[edge_id] = edge
        self.edge_ids_by_source_state.setdefault(source_id, {})[action_id] = edge_id
        if edge.trainable:
            self.trainable_edge_ids.append(edge_id)
        return ref

    def outgoing_edges(self, source_state_id: int) -> Tuple[Tuple[int, GraphEdge], ...]:
        edge_ids_by_action = self.edge_ids_by_source_state.get(int(source_state_id))
        if not edge_ids_by_action:
            return ()
        return tuple(
            (int(action), self.edges[int(edge_id)])
            for action, edge_id in edge_ids_by_action.items()
        )

    def mark_canonical(self, ref: EdgeRef) -> None:
        self._require_local_ref(ref)
        edge = self.edges[int(ref.edge_id)]
        edge.canonical = True
        self.canonical_edge_ids.add(int(ref.edge_id))

    def mark_protected(self, ref: EdgeRef) -> None:
        self._require_local_ref(ref)
        edge = self.edges[int(ref.edge_id)]
        edge.protected = True
        self.protected_edge_ids.add(int(ref.edge_id))

    @property
    def edge_count(self) -> int:
        return int(len(self.edges))

    @property
    def trainable_edge_count(self) -> int:
        return int(len(self.trainable_edge_ids))

    def _require_local_ref(self, ref: EdgeRef) -> None:
        if int(ref.world_index) != int(self.world_index) or int(ref.edge_id) not in self.edges:
            raise KeyError(
                f"EdgeRef does not belong to world={int(self.world_index)}: {ref}"
            )

    @staticmethod
    def _merge_edge_metadata(
        *,
        edge: GraphEdge,
        class_id: Optional[int],
        rh: float,
        rz: float,
        rtotal: float,
        trainable: bool,
    ) -> None:
        if isinstance(class_id, int) and int(class_id) > 0:
            edge.class_id = int(class_id)
        edge.rh = float(rh)
        edge.rz = float(rz)
        edge.rtotal = float(rtotal)
        edge.trainable = bool(edge.trainable or trainable)
