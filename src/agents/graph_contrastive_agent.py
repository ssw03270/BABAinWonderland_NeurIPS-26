"""Graph-based frontier search contrastive explorer."""

from __future__ import annotations

import heapq
import math
import multiprocessing
import os
import pickle
import queue
import random
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .graph_collect_actor import (
    ActorDoneEvent,
    ActorErrorEvent,
    ActorShutdown,
    ActorStateRef,
    ActorStartCollect,
    ActorStepWorkItem,
    ActorTransitionBatchEvent,
    ActorTransitionEvent,
    run_map_collect_actor,
)
from .graph_search_agent import (
    SearchNode,
    WorldSpec,
    shuffle_round_robin_sessions,
)
from .contrastive_core import (
    ContrastiveEntropyContext,
    GraphContrastiveBase,
    ContrastiveSample,
    ContrastiveSampleStore,
    _clone_if_inference_tensor,
    _distributed_step_command_should_share_tensors,
    _share_cpu_tensor_if_requested,
)
from .world_graph import EdgeRef, WorldGraph
from src.data import (
    StateStore,
    canonical_state_key,
    runtime_state_packet_key,
    runtime_state_packet_obj,
)
from src.data.transition_buffer import Transition
from .analysis_archive import AnalysisArchiveWriter
from src.program_model import ProgramEvaluationTask
from src.web.map_display_names import resolve_map_display_name


@dataclass(slots=True)
class FrontierCandidate:
    state_id: int
    state_key: str
    action: int
    score: float
    rh: float
    rz_pred: float
    order: int


@dataclass(frozen=True, slots=True)
class _FrontierTransitionCommit:
    node: SearchNode
    next_state_id: int
    next_key: str
    candidate_depth: int
    action_name: str
    map_name: Optional[str]
    transition_result: Dict[str, Any]
    transition_env_reward: float
    transition_rh: float
    transition_rz: float
    transition_intrinsic_reward: float
    display_step: int


@dataclass(slots=True)
class _CollectActorWorkerHandle:
    worker_id: int
    command_queue: Any
    runner: Any


@dataclass(slots=True)
class _PreparedActorTransition:
    event: ActorTransitionEvent
    session: "GraphContrastiveWorldSession"
    candidate: FrontierCandidate
    transition_key: str
    next_state_id: int
    transition: Transition
    assessment_transition: Transition


@dataclass(slots=True)
class _FrontierCandidateIndex:
    state_ids: List[int] = field(default_factory=list)
    state_index_by_id: Dict[int, int] = field(default_factory=dict)
    masks_by_state_id: Dict[int, int] = field(default_factory=dict)
    candidate_counts: List[int] = field(default_factory=list)
    tree_capacity: int = 1
    tree: List[int] = field(default_factory=lambda: [0, 0])
    total_count: int = 0

    def set_mask(self, *, state_id: int, mask: int, action_count: int) -> None:
        resolved_state_id = int(state_id)
        resolved_mask = int(mask)
        resolved_count = max(0, int(action_count))
        if resolved_mask <= 0 or resolved_count <= 0:
            self.remove_state(resolved_state_id)
            return

        row_index = self.state_index_by_id.get(resolved_state_id)
        if row_index is None:
            row_index = self._insert_state_id(resolved_state_id)
        self.masks_by_state_id[resolved_state_id] = resolved_mask
        self._set_count(row_index, resolved_count)

    def remove_state(self, state_id: int) -> None:
        resolved_state_id = int(state_id)
        row_index = self.state_index_by_id.get(resolved_state_id)
        self.masks_by_state_id.pop(resolved_state_id, None)
        if row_index is not None:
            self._set_count(row_index, 0)

    def resolve(self, local_position: int, action_order: Sequence[int]) -> Tuple[int, int]:
        if not 0 <= int(local_position) < int(self.total_count):
            raise IndexError(
                f"frontier local_position out of range: {int(local_position)}"
            )
        remaining = int(local_position)
        node = 1
        while node < int(self.tree_capacity):
            left = node * 2
            left_count = int(self.tree[left])
            if remaining < left_count:
                node = left
            else:
                remaining -= left_count
                node = left + 1
        row_index = node - int(self.tree_capacity)
        state_id = int(self.state_ids[row_index])
        mask = int(self.masks_by_state_id[state_id])
        for action in action_order:
            resolved_action = int(action)
            if mask & (1 << resolved_action):
                if remaining == 0:
                    return state_id, resolved_action
                remaining -= 1
        raise RuntimeError(
            "Frontier candidate index is inconsistent with its action mask."
        )

    def _insert_state_id(self, state_id: int) -> int:
        resolved_state_id = int(state_id)
        if not self.state_ids or resolved_state_id > int(self.state_ids[-1]):
            row_index = len(self.state_ids)
            self.state_ids.append(resolved_state_id)
            self.candidate_counts.append(0)
            self.state_index_by_id[resolved_state_id] = row_index
            self._ensure_tree_capacity(len(self.state_ids))
            return row_index

        row_index = bisect_left(self.state_ids, resolved_state_id)
        if (
            row_index < len(self.state_ids)
            and int(self.state_ids[row_index]) == resolved_state_id
        ):
            self.state_index_by_id[resolved_state_id] = row_index
            return row_index
        self.state_ids.insert(row_index, resolved_state_id)
        self.candidate_counts.insert(row_index, 0)
        self.state_index_by_id = {
            int(existing_state_id): int(index)
            for index, existing_state_id in enumerate(self.state_ids)
        }
        self._rebuild_tree()
        return row_index

    def _set_count(self, row_index: int, count: int) -> None:
        old_count = int(self.candidate_counts[int(row_index)])
        new_count = max(0, int(count))
        if old_count == new_count:
            return
        self.candidate_counts[int(row_index)] = new_count
        delta = new_count - old_count
        self.total_count += int(delta)
        self._add_to_tree(int(row_index), int(delta))

    def _ensure_tree_capacity(self, row_count: int) -> None:
        if int(row_count) <= int(self.tree_capacity):
            return
        while int(self.tree_capacity) < int(row_count):
            self.tree_capacity *= 2
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self._ensure_minimum_tree_capacity()
        self.tree = [0 for _ in range(int(self.tree_capacity) * 2)]
        for index, count in enumerate(self.candidate_counts):
            self.tree[int(self.tree_capacity) + int(index)] = int(count)
        for index in range(int(self.tree_capacity) - 1, 0, -1):
            self.tree[index] = self.tree[index * 2] + self.tree[index * 2 + 1]
        self.total_count = int(self.tree[1]) if len(self.tree) > 1 else 0

    def _ensure_minimum_tree_capacity(self) -> None:
        required = max(1, len(self.candidate_counts))
        while int(self.tree_capacity) < required:
            self.tree_capacity *= 2

    def _add_to_tree(self, row_index: int, delta: int) -> None:
        if int(delta) == 0:
            return
        index = int(self.tree_capacity) + int(row_index)
        while index > 0:
            self.tree[index] += int(delta)
            index //= 2


@dataclass(slots=True)
class GraphContrastiveWorldSession:
    spec: WorldSpec
    root_state_id: Optional[int] = None
    world_graph: WorldGraph = field(init=False)
    state_nodes: Dict[int, SearchNode] = field(default_factory=dict)
    expanded_state_ids: set[int] = field(default_factory=set)
    pending_action_masks_by_state: Dict[int, int] = field(default_factory=dict)
    frontier_index: _FrontierCandidateIndex = field(default_factory=_FrontierCandidateIndex)
    world_transition_count: int = 0
    warmup_edge_queue: Deque[tuple[int, int]] = field(default_factory=deque)
    frontier_version: int = 0

    def __post_init__(self) -> None:
        self.world_graph = WorldGraph(
            world_index=int(self.spec.world_index),
            world_seed=int(self.spec.world_seed),
            world_label=str(self.spec.world_label),
        )

    @property
    def world_index(self) -> int:
        return int(self.spec.world_index)

    @property
    def world_seed(self) -> int:
        return int(self.spec.world_seed)

    @property
    def scenario_type(self) -> Optional[str]:
        return self.spec.scenario_type

    @property
    def world_label(self) -> str:
        return self.spec.world_label


@dataclass(frozen=True, slots=True)
class _FrontierStateActionBatch:
    session: GraphContrastiveWorldSession
    node: SearchNode
    actions: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _FrontierRefreshSelection:
    state_action_entries: Tuple[_FrontierStateActionBatch, ...]
    total_candidates: int
    selected_candidates: int


class GraphContrastiveAgent(GraphContrastiveBase):
    strategy_name = "graph_contrastive"
    collection_topology = "graph"
    transition_batch_scope = "world"
    _FRONTIER_REBUILD_TARGET_CANDIDATES = 10240
    _FRONTIER_REBUILD_MAX_STATE_BATCH = 2048
    _FRONTIER_REBUILD_MIN_STATE_BATCH = 32
    _FRONTIER_REBUILD_CANDIDATE_CHUNK = 32768
    _SUPPORTED_FRONTIER_SAMPLING_MODES = (
        "global_uniform",
        "map_balanced",
    )
    _ITERATION_SUMMARY_METRIC_KEYS = (
        "train_iter",
        "contrastive_loss",
        "prototype_top1_accuracy",
        "sample_store_size",
        "active_dynamics_classes",
        "known_dynamics_classes",
        "dynamics_class_transition_counts",
    )
    _ITERATION_METRIC_KEYS = (
        "global_step",
        "iter",
        "train_iter",
        "program_version",
        "iteration_end_program_version",
        "llm_calls",
        "contrastive_loss",
        "prototype_top1_accuracy",
        "rh",
        "rz",
        "rtotal",
        "frontier_size",
        "live_map_frontier_size",
        "sample_store_size",
        "world_index",
        "map_name",
        "step",
        "action",
        "done",
        "current_dynamics_class",
        "active_dynamics_classes",
        "known_dynamics_classes",
        "dynamics_class_transition_counts",
    )

    def __init__(
        self,
        env: Any,
        seed: int = 42,
        max_step_depth_per_world: Optional[int] = None,
        collect_workers: Any = "auto",
        collect_env_factory: Optional[Callable[[], Any]] = None,
        action_order: Optional[Sequence[int]] = None,
        replay_cap: Optional[int] = None,
        frontier_cap: Optional[int] = None,
        frontier_sampling_mode: str = "global_uniform",
        analysis_archive_enabled: bool = False,
        analysis_archive_local_dir: Optional[str] = None,
        analysis_archive_copy_interval_sec: float = 5.0,
        **kwargs: Any,
    ) -> None:
        self._state_store: Optional[StateStore] = None
        self.action_order: Tuple[int, ...] = ()
        self._all_action_mask: int = 0
        self._world_specs: Tuple[WorldSpec, ...] = ()
        self._round_robin_world_indices: Tuple[int, ...] = ()
        self._active_worlds: Deque[GraphContrastiveWorldSession] = deque()
        self._all_world_sessions: Dict[int, GraphContrastiveWorldSession] = {}
        self._world_sessions_initialized = False
        self.total_restore_steps = 0
        self.total_collect_calls = 0
        self.total_transitions_collected = 0
        self.total_worlds_completed = 0
        self._frontier_push_order = 0
        self._last_visual_board_state_id: Optional[int] = None
        self._iteration_step_metrics: List[Dict[str, Any]] = []
        self._iteration_summary_metrics: Dict[str, Any] = {}
        self._last_recorded_iteration_metric: Optional[int] = None
        self._analysis_archive_enabled = bool(analysis_archive_enabled)
        self._analysis_archive_local_dir = (
            str(analysis_archive_local_dir).strip()
            if isinstance(analysis_archive_local_dir, str)
            and str(analysis_archive_local_dir).strip()
            else None
        )
        self._analysis_archive_copy_interval_sec = max(
            0.1,
            float(analysis_archive_copy_interval_sec),
        )
        self._analysis_archive_writer: Optional[AnalysisArchiveWriter] = None
        self._analysis_archive_output_dir: Optional[Path] = None
        self._analysis_node_rows: List[Dict[str, Any]] = []
        self._analysis_edge_rows: List[Dict[str, Any]] = []
        self.replay_cap = self._resolve_optional_refresh_cap(replay_cap, "replay_cap")
        self.frontier_cap = self._resolve_optional_refresh_cap(frontier_cap, "frontier_cap")
        self.frontier_sampling_mode = self._resolve_frontier_sampling_mode(
            frontier_sampling_mode
        )
        self._last_collect_stats: Dict[str, Any] = {}
        self._last_completed_world_summary: Dict[str, Any] = {}
        self.collect_workers_auto, self.collect_workers = self._resolve_collect_workers(
            collect_workers
        )
        self.collect_env_factory = collect_env_factory
        self._collect_actor_event_queue: Optional[Any] = None
        self._collect_actor_workers: Dict[int, _CollectActorWorkerHandle] = {}
        self._collect_actor_worker_count: Optional[int] = None
        self._collect_actor_world_worker: Dict[int, int] = {}
        self._collect_actor_dispatch_ids_by_worker: Dict[int, int] = {}
        super().__init__(env=env, seed=seed, **kwargs)
        self._validate_prototype_sample_cap_for_replay()

        state_format = str(self.env.serializer.format_type)
        if state_format != "json":
            raise ValueError("GraphContrastiveAgent requires JSON state serialization.")
        self.max_step_depth_per_world = self._resolve_optional_step_depth_limit(
            max_step_depth_per_world
        )
        self.action_order = self._resolve_action_order(action_order)
        self._all_action_mask = self._action_mask(self.action_order)
        self._world_specs = self._resolve_world_specs()
        self._round_robin_world_indices = tuple(
            int(spec.world_index)
            for spec in shuffle_round_robin_sessions(self._world_specs, seed=self.seed)
        )
        if not self._world_specs:
            raise ValueError("GraphContrastiveAgent requires at least one target world.")
        self.reset()
        self._load_initial_contrastive_checkpoint_if_configured(force=True)

    @property
    def state_store(self) -> StateStore:
        return self._ensure_state_store()

    def close(self) -> None:
        self._close_collect_actor_workers()
        self._close_analysis_archive_writer()
        super().close()

    def _close_analysis_archive_writer(self) -> None:
        writer = self._analysis_archive_writer
        if writer is not None:
            writer.close()
        self._analysis_archive_writer = None
        self._analysis_archive_output_dir = None

    def _ensure_analysis_archive_writer(
        self,
        output_dir: str | Path,
    ) -> Optional[AnalysisArchiveWriter]:
        if not self._analysis_archive_enabled:
            return None
        output_root = Path(output_dir).absolute()
        exp_root = (output_root / "analysis_archive").absolute()
        if (
            self._analysis_archive_writer is None
            or self._analysis_archive_output_dir != exp_root
        ):
            self._close_analysis_archive_writer()
            local_root = None
            if self._analysis_archive_local_dir is not None:
                local_root = (
                    Path(self._analysis_archive_local_dir).resolve()
                    / output_root.name
                    / "analysis_archive"
                )
            self._analysis_archive_writer = AnalysisArchiveWriter(
                exp_root=exp_root,
                local_root=local_root,
                copy_interval_sec=self._analysis_archive_copy_interval_sec,
            )
            self._analysis_archive_output_dir = exp_root
        return self._analysis_archive_writer

    def _close_collect_actor_workers(self) -> None:
        workers = dict(self._collect_actor_workers)
        if not workers:
            self._collect_actor_event_queue = None
            self._collect_actor_workers = {}
            self._collect_actor_worker_count = None
            self._collect_actor_world_worker.clear()
            self._collect_actor_dispatch_ids_by_worker.clear()
            return
        for handle in workers.values():
            handle.command_queue.put(ActorShutdown())
        for handle in workers.values():
            handle.runner.join(timeout=5.0)
            if handle.runner.is_alive() and hasattr(handle.runner, "terminate"):
                handle.runner.terminate()
                handle.runner.join(timeout=5.0)
        self._collect_actor_workers = {}
        self._collect_actor_event_queue = None
        self._collect_actor_worker_count = None
        self._collect_actor_world_worker.clear()
        self._collect_actor_dispatch_ids_by_worker.clear()

    def _spawn_collect_actor_worker(self, *, worker_id: int) -> _CollectActorWorkerHandle:
        context = multiprocessing.get_context("spawn")
        if self._collect_actor_event_queue is None:
            self._collect_actor_event_queue = context.Queue()
        command_queue = context.Queue()
        kwargs = {
            "worker_id": int(worker_id),
            "env_factory": self.collect_env_factory,
            "command_queue": command_queue,
            "event_queue": self._collect_actor_event_queue,
        }
        runner = context.Process(
            target=run_map_collect_actor,
            kwargs=kwargs,
            daemon=False,
        )
        runner.start()
        handle = _CollectActorWorkerHandle(
            worker_id=int(worker_id),
            command_queue=command_queue,
            runner=runner,
        )
        self._collect_actor_workers[int(worker_id)] = handle
        self._collect_actor_dispatch_ids_by_worker.setdefault(int(worker_id), 0)
        return handle

    def _close_collect_actor_worker(self, *, worker_id: int) -> None:
        handle = self._collect_actor_workers.pop(int(worker_id), None)
        if handle is None:
            return
        try:
            handle.command_queue.put(ActorShutdown())
        except (RuntimeError, ValueError, OSError):
            pass
        handle.runner.join(timeout=5.0)
        if handle.runner.is_alive() and hasattr(handle.runner, "terminate"):
            handle.runner.terminate()
            handle.runner.join(timeout=5.0)

    def _respawn_collect_actor_worker(self, *, worker_id: int) -> _CollectActorWorkerHandle:
        self._close_collect_actor_worker(worker_id=int(worker_id))
        return self._spawn_collect_actor_worker(worker_id=int(worker_id))

    def _validate_process_collect_runtime(self) -> None:
        if not callable(self.collect_env_factory):
            raise ValueError(
                "collect_env_factory is required for graph contrastive actor collect."
            )
        try:
            pickle.dumps(self.collect_env_factory)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ValueError(
                "collect_env_factory must be picklable for process parallel collect."
            ) from exc

    def _effective_collect_workers(self, *, active_map_count: int) -> int:
        del active_map_count
        return max(1, int(self.collect_workers))

    def _collect_workers_config_value(self) -> Any:
        if bool(self.collect_workers_auto):
            return "auto"
        return int(self.collect_workers)

    @staticmethod
    def _resolve_optional_refresh_cap(value: Optional[int], name: str) -> Optional[int]:
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} must be positive when provided.")
        return int(parsed)

    def _validate_prototype_sample_cap_for_replay(self) -> None:
        if self.prototype_sample_cap is None or self.replay_cap is None:
            return
        if int(self.prototype_sample_cap) >= int(self.replay_cap):
            return
        raise ValueError(
            "prototype_sample_cap must be greater than or equal to replay_cap "
            "because frontier replay tau is sampled from the capped prototype tau pool. "
            f"Got prototype_sample_cap={int(self.prototype_sample_cap)} and "
            f"replay_cap={int(self.replay_cap)}. Increase prototype_sample_cap or "
            "lower replay_cap."
        )

    @classmethod
    def _resolve_frontier_sampling_mode(cls, raw_mode: Any) -> str:
        mode = str(raw_mode).strip()
        if mode in cls._SUPPORTED_FRONTIER_SAMPLING_MODES:
            return mode
        supported_modes = ", ".join(cls._SUPPORTED_FRONTIER_SAMPLING_MODES)
        raise ValueError(
            f"frontier_sampling_mode must be one of {supported_modes}, got {raw_mode!r}."
        )

    def _ensure_collect_actor_workers(self, *, worker_count: int) -> None:
        self._validate_process_collect_runtime()
        resolved_worker_count = max(1, int(worker_count))
        if self._collect_actor_worker_count != resolved_worker_count:
            self._close_collect_actor_workers()
        if self._collect_actor_event_queue is None:
            context = multiprocessing.get_context("spawn")
            self._collect_actor_event_queue = context.Queue()
        for worker_id in range(int(resolved_worker_count)):
            handle = self._collect_actor_workers.get(int(worker_id))
            if handle is None:
                self._spawn_collect_actor_worker(worker_id=int(worker_id))
            elif not handle.runner.is_alive():
                self._respawn_collect_actor_worker(worker_id=int(worker_id))
        self._collect_actor_worker_count = int(resolved_worker_count)

    def set_state_store(self, state_store: Any) -> None:
        resolved_state_store = self._require_state_store(
            state_store,
            owner="GraphContrastiveAgent",
        )
        if self._state_store is resolved_state_store:
            self._bind_env_state_store(resolved_state_store)
            return
        if self._world_sessions_initialized or self._active_worlds:
            self.reset()
        else:
            self._shared_state_token_cache.clear()
        self._state_store = resolved_state_store
        self._bind_env_state_store(resolved_state_store)

    def reset(self) -> None:
        self._close_collect_actor_workers()
        super().reset()
        self.total_restore_steps = 0
        self.total_collect_calls = 0
        self.total_transitions_collected = 0
        self.total_worlds_completed = 0
        self._active_worlds = deque()
        self._round_robin_world_indices = tuple(
            int(spec.world_index)
            for spec in shuffle_round_robin_sessions(self._world_specs, seed=self.seed)
        )
        self._all_world_sessions = {}
        self._world_sessions_initialized = False
        self._frontier_push_order = 0
        self._last_collect_stats = {}
        self._last_completed_world_summary = {}
        self._last_visual_board_state_id = None
        self._iteration_step_metrics = []
        self._iteration_summary_metrics = {}
        self._last_recorded_iteration_metric = None
        self._analysis_node_rows = []
        self._analysis_edge_rows = []

    def restore_iteration_boundary_state(self, resume_state: Any) -> Dict[str, Any]:
        state_store = self._require_state_store(
            getattr(resume_state, "state_store", None),
            owner="GraphContrastiveAgent",
        )
        sample_store = getattr(resume_state, "sample_store", None)
        if not isinstance(sample_store, ContrastiveSampleStore):
            raise TypeError("resume_state.sample_store must be a ContrastiveSampleStore.")

        self._close_collect_actor_workers()
        self._state_store = state_store
        self._bind_env_state_store(state_store)
        self.sample_store = sample_store
        self._shared_state_token_cache.clear()
        self._sample_storage_token_cache.clear()
        self._invalidate_sample_store_tau_cache()
        self._mark_sample_storage_changed()
        self._mark_sample_labels_changed()

        cutoff_global_step = int(getattr(resume_state, "cutoff_global_step", 0) or 0)
        checkpoint_total_steps = getattr(resume_state, "checkpoint_total_steps", None)
        restored_total_steps = max(
            int(getattr(self, "total_steps", 0) or 0),
            cutoff_global_step,
            int(checkpoint_total_steps)
            if isinstance(checkpoint_total_steps, int)
            else 0,
        )
        self.total_steps = int(restored_total_steps)
        self.total_transitions_collected = int(getattr(resume_state, "edge_count", 0) or 0)
        self.total_collect_calls = 0
        self.total_restore_steps = 0
        self.total_worlds_completed = 0
        self._last_collect_stats = {}
        self._last_completed_world_summary = {}
        self._last_visual_board_state_id = None
        self._iteration_step_metrics = []
        self._iteration_summary_metrics = {}
        self._last_recorded_iteration_metric = None
        self._analysis_node_rows = []
        self._analysis_edge_rows = []

        final_nodes = self._restore_iteration_boundary_nodes(
            getattr(resume_state, "node_rows", ())
        )
        sessions = self._restore_iteration_boundary_sessions(final_nodes)
        self._restore_iteration_boundary_edges(
            sessions=sessions,
            edge_rows=getattr(resume_state, "final_edge_rows", ()),
        )
        self._restore_iteration_boundary_frontiers(sessions=sessions)
        self._restore_iteration_boundary_dashboard_metrics(resume_state)

        ordered_sessions = shuffle_round_robin_sessions(
            list(sessions.values()),
            seed=self.seed,
        )
        self._round_robin_world_indices = tuple(
            int(session.world_index) for session in ordered_sessions
        )
        self._active_worlds = deque(
            session for session in ordered_sessions if self._session_is_active(session)
        )
        self._all_world_sessions = dict(sessions)
        self._world_sessions_initialized = True
        self.total_worlds_completed = sum(
            1 for session in sessions.values() if not self._session_is_active(session)
        )
        self._frontier_push_order = int(self._global_frontier_size())

        return {
            "world_count": int(len(sessions)),
            "active_world_count": int(len(self._active_worlds)),
            "frontier_size": int(self._global_frontier_size()),
            "edge_count": int(self._graph_edge_count()),
            "sample_store_size": int(len(self.sample_store)),
            "total_steps": int(self.total_steps),
        }

    def _restore_iteration_boundary_dashboard_metrics(self, resume_state: Any) -> None:
        history = getattr(resume_state, "dashboard_history", None)
        if not isinstance(history, Mapping):
            return
        metrics = history.get("metrics")
        if not isinstance(metrics, Mapping):
            return

        def _restore_window(attr_name: str, key: str) -> None:
            window = getattr(self, attr_name, None)
            raw_values = metrics.get(key)
            if window is None or not isinstance(raw_values, (list, tuple)):
                return
            clear = getattr(window, "clear", None)
            append = getattr(window, "append", None)
            if not callable(clear) or not callable(append):
                return
            clear()
            for raw_value in list(raw_values)[-int(self.dashboard_history_limit):]:
                append(raw_value)

        _restore_window("_rolling_contrastive_loss", "contrastive_loss")
        _restore_window("_rolling_prototype_top1_accuracy", "prototype_top1_accuracy")
        _restore_window("_rolling_rh", "rh")
        _restore_window("_rolling_rz", "rz")
        _restore_window("_rolling_rtotal", "rtotal")

        last_train_stats = dict(getattr(self, "_last_train_stats", {}) or {})
        for key in ("contrastive_loss", "prototype_top1_accuracy"):
            raw_values = metrics.get(key)
            if not isinstance(raw_values, (list, tuple)):
                continue
            for raw_value in reversed(list(raw_values)):
                try:
                    parsed = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    last_train_stats[key] = float(parsed)
                    break
        self._last_train_stats = last_train_stats

        visualizer = getattr(self, "visualizer", None)
        restore_history = getattr(visualizer, "restore_dashboard_history", None)
        if callable(restore_history):
            restore_history(history)

    def _restore_iteration_boundary_nodes(
        self,
        node_rows: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Dict[int, SearchNode]]:
        nodes_by_world: Dict[int, Dict[int, SearchNode]] = {}
        for row in node_rows:
            if not isinstance(row, Mapping):
                continue
            try:
                world_index = int(row.get("world_index"))
                state_id = int(row.get("state_id"))
                depth = int(row.get("depth", 0) or 0)
            except (TypeError, ValueError):
                continue
            raw_parent_state_id = row.get("parent_state_id")
            raw_parent_action = row.get("parent_action")
            parent_state_id = (
                int(raw_parent_state_id)
                if isinstance(raw_parent_state_id, int)
                and int(raw_parent_state_id) > 0
                else None
            )
            parent_action = (
                int(raw_parent_action)
                if isinstance(raw_parent_action, int)
                and int(raw_parent_action) >= 0
                else None
            )
            state_key = row.get("state_key")
            if not isinstance(state_key, str) or not state_key.strip():
                state_key = self._ensure_state_store().state_key(state_id)
            nodes_by_world.setdefault(world_index, {})[state_id] = SearchNode(
                state_id=state_id,
                key=str(state_key),
                depth=depth,
                parent_state_id=parent_state_id,
                parent_action=parent_action,
            )
        return nodes_by_world

    def _restore_iteration_boundary_sessions(
        self,
        nodes_by_world: Mapping[int, Mapping[int, SearchNode]],
    ) -> Dict[int, GraphContrastiveWorldSession]:
        sessions: Dict[int, GraphContrastiveWorldSession] = {}
        for spec in self._world_specs:
            world_index = int(spec.world_index)
            world_nodes = dict(nodes_by_world.get(world_index, {}))
            if not world_nodes:
                raise ValueError(
                    "Resume archive is missing graph nodes for "
                    f"world_index={world_index}."
                )
            root_nodes = [
                node for node in world_nodes.values() if int(node.depth) == 0
            ]
            if not root_nodes:
                raise ValueError(
                    "Resume archive is missing a root node for "
                    f"world_index={world_index}."
                )
            root_node = min(root_nodes, key=lambda node: int(node.state_id))
            session = GraphContrastiveWorldSession(
                spec=spec,
                root_state_id=int(root_node.state_id),
            )
            for node in world_nodes.values():
                session.world_graph.register_state(
                    state_id=int(node.state_id),
                    state_key=str(node.key),
                )
                session.state_nodes[int(node.state_id)] = node
            sessions[world_index] = session
        return sessions

    def _restore_iteration_boundary_edges(
        self,
        *,
        sessions: Mapping[int, GraphContrastiveWorldSession],
        edge_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        for row in sorted(
            (dict(item) for item in edge_rows if isinstance(item, Mapping)),
            key=lambda item: (
                int(item.get("world_index", 0) or 0),
                int(item.get("edge_id", 0) or 0),
            ),
        ):
            world_index = int(row.get("world_index"))
            session = sessions.get(world_index)
            if session is None:
                raise ValueError(f"Resume edge references unknown world_index={world_index}.")
            class_id = (
                int(row.get("class_id"))
                if isinstance(row.get("class_id"), int)
                and int(row.get("class_id")) > 0
                else None
            )
            edge_ref = session.world_graph.register_edge(
                source_state_id=int(row.get("source_state_id")),
                action=int(row.get("action")),
                next_state_id=int(row.get("next_state_id")),
                done=bool(row.get("done")),
                class_id=class_id,
                rh=float(row.get("rh", 0.0) or 0.0),
                rz=float(row.get("rz", 0.0) or 0.0),
                rtotal=float(row.get("rtotal", 0.0) or 0.0),
                trainable=bool(row.get("trainable", True)),
            )
            expected_edge_id = row.get("edge_id")
            if isinstance(expected_edge_id, int) and int(edge_ref.edge_id) != int(expected_edge_id):
                raise RuntimeError(
                    "Restored world graph edge order does not match archive: "
                    f"world={world_index} expected={int(expected_edge_id)} "
                    f"got={int(edge_ref.edge_id)}."
                )
            session.world_graph.mark_canonical(edge_ref)
            session.world_graph.mark_protected(edge_ref)
            session.world_transition_count += 1

    def _restore_iteration_boundary_frontiers(
        self,
        *,
        sessions: Mapping[int, GraphContrastiveWorldSession],
    ) -> None:
        use_scored_frontier = self._should_use_frontier_scoring()
        for session in sessions.values():
            for node in session.state_nodes.values():
                if not self._node_can_expand(node):
                    continue
                session.expanded_state_ids.add(int(node.state_id))
                executed_mask = 0
                for action, _edge in session.world_graph.outgoing_edges(int(node.state_id)):
                    executed_mask |= 1 << int(action)
                pending_mask = int(self._all_action_mask) & ~int(executed_mask)
                if pending_mask <= 0:
                    continue
                self._set_pending_action_mask(
                    session=session,
                    state_id=int(node.state_id),
                    mask=int(pending_mask),
                )
                if not use_scored_frontier:
                    for action in self._actions_from_mask(pending_mask):
                        session.warmup_edge_queue.append((int(node.state_id), int(action)))
            self._mark_frontier_changed(session)

    def _resolve_action_order(
        self,
        action_order: Optional[Sequence[int]],
    ) -> Tuple[int, ...]:
        if action_order is None:
            return tuple(range(self.num_actions))
        resolved: List[int] = []
        for raw_action in action_order:
            action = int(raw_action)
            if action < 0 or action >= self.num_actions:
                raise ValueError(
                    f"Invalid action id in action_order: {action} "
                    f"(valid range: 0..{self.num_actions - 1})"
                )
            resolved.append(action)
        if len(set(resolved)) != len(resolved):
            raise ValueError("action_order must not contain duplicates.")
        if len(resolved) != self.num_actions:
            missing = sorted(set(range(self.num_actions)) - set(resolved))
            raise ValueError(
                "action_order must cover every action exactly once. "
                f"Missing={missing}"
            )
        return tuple(resolved)

    def _resolve_optional_step_depth_limit(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError(
                "max_step_depth_per_world must be >= 0 when provided, "
                f"got {value!r}."
            )
        return int(parsed)

    @staticmethod
    def _auto_collect_worker_cap() -> int:
        return max(2, int(os.cpu_count() or 1))

    @classmethod
    def _resolve_collect_workers(cls, value: Any) -> tuple[bool, int]:
        if value is None:
            return True, cls._auto_collect_worker_cap()
        if isinstance(value, str) and value.strip().lower() == "auto":
            return True, cls._auto_collect_worker_cap()
        resolved = int(value)
        if resolved < 2:
            raise ValueError("collect_workers must be an integer >= 2.")
        return False, resolved

    @staticmethod
    def _resolve_positive_int(value: Any, *, name: str) -> int:
        resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{name} must be an integer > 0, got {value!r}.")
        return resolved

    def _action_mask(self, actions: Sequence[int]) -> int:
        mask = 0
        for raw_action in actions:
            mask |= 1 << int(raw_action)
        return int(mask)

    def _actions_from_mask(self, mask: int) -> List[int]:
        resolved_mask = int(mask)
        if resolved_mask <= 0:
            return []
        return [
            int(action)
            for action in self.action_order
            if resolved_mask & (1 << int(action))
        ]

    def _action_count_from_mask(self, mask: int) -> int:
        resolved_mask = int(mask)
        if resolved_mask <= 0:
            return 0
        all_action_mask = getattr(self, "_all_action_mask", 0)
        if isinstance(all_action_mask, int) and (resolved_mask & ~int(all_action_mask)) == 0:
            return int(resolved_mask.bit_count())
        return sum(
            1
            for action in self.action_order
            if resolved_mask & (1 << int(action))
        )

    def _set_pending_action_mask(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        mask: int,
    ) -> None:
        resolved_state_id = int(state_id)
        resolved_mask = int(mask)
        action_count = self._action_count_from_mask(resolved_mask)
        if resolved_mask <= 0 or action_count <= 0:
            self._clear_pending_action_mask(
                session=session,
                state_id=resolved_state_id,
            )
            return
        session.pending_action_masks_by_state[resolved_state_id] = resolved_mask
        session.frontier_index.set_mask(
            state_id=resolved_state_id,
            mask=resolved_mask,
            action_count=action_count,
        )

    @staticmethod
    def _clear_pending_action_mask(
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
    ) -> None:
        resolved_state_id = int(state_id)
        session.pending_action_masks_by_state.pop(resolved_state_id, None)
        session.frontier_index.remove_state(resolved_state_id)

    def _resolve_world_specs(self) -> Tuple[WorldSpec, ...]:
        normalized = self._normalize_world_specs(
            self.env.list_graph_search_worlds(seed=int(self.seed))
        )
        if normalized:
            return normalized
        raise ValueError("GraphContrastiveAgent requires at least one target world.")

    def _normalize_world_specs(self, raw_worlds: Any) -> Tuple[WorldSpec, ...]:
        if not isinstance(raw_worlds, (list, tuple)):
            return ()
        normalized: List[WorldSpec] = []
        for index, raw_world in enumerate(raw_worlds, start=1):
            if isinstance(raw_world, Mapping):
                world_seed = int(raw_world.get("seed", self.seed))
                raw_scenario_type = raw_world.get("scenario_type")
                scenario_type = (
                    str(raw_scenario_type).strip()
                    if isinstance(raw_scenario_type, str) and raw_scenario_type.strip()
                    else None
                )
                raw_label = raw_world.get("label")
                label = str(
                    resolve_map_display_name(raw_label, scenario_type) or ""
                ).strip()
            else:
                world_seed = int(raw_world)
                scenario_type = None
                label = ""
            normalized.append(
                WorldSpec(
                    world_index=index,
                    world_seed=int(world_seed),
                    scenario_type=scenario_type,
                    label=label,
                )
            )
        return tuple(normalized)

    def _ensure_world_sessions_initialized(self) -> int:
        if self._world_sessions_initialized:
            return 0
        sessions = [self._create_world_session(spec) for spec in self._world_specs]
        ordered_sessions = shuffle_round_robin_sessions(
            sessions,
            seed=self.seed,
        )
        self._active_worlds = deque(ordered_sessions)
        self._round_robin_world_indices = tuple(
            int(session.world_index) for session in ordered_sessions
        )
        self._all_world_sessions = {
            session.world_index: session
            for session in sessions
        }
        self._world_sessions_initialized = True
        return int(len(sessions))

    def _create_world_session(self, spec: WorldSpec) -> GraphContrastiveWorldSession:
        root_state = self._reset_world(spec)
        self._map_reset_count += 1
        root_state_id, root_key = self._intern_state_json(root_state)
        root_node = SearchNode(
            state_id=root_state_id,
            key=root_key,
            depth=0,
        )
        session = GraphContrastiveWorldSession(
            spec=spec,
            root_state_id=int(root_state_id),
        )
        session.world_graph.register_state(
            state_id=int(root_state_id),
            state_key=str(root_key),
        )
        session.state_nodes[root_state_id] = root_node
        self._prepare_state_frontier(session=session, node=root_node)
        self._append_analysis_node_row(
            iteration_event="add",
            session=session,
            node=root_node,
        )
        return session

    def _reset_world(self, spec: WorldSpec) -> str:
        return self.env.reset_for_graph_search(
            seed=int(spec.world_seed),
            scenario_type=spec.scenario_type,
        )

    def _stamp_sample_world_metadata(
        self,
        sample: ContrastiveSample,
        *,
        session: GraphContrastiveWorldSession,
    ) -> ContrastiveSample:
        sample.source_world_index = int(session.world_index)
        sample.source_world_seed = int(session.world_seed)
        return sample

    def _trusted_state_key(self, state_json: str) -> Optional[str]:
        getter = getattr(self.env, "get_canonical_state_key", None)
        if not callable(getter):
            return None
        try:
            resolved = getter(state_json)
        except TypeError:
            resolved = getter()
        if isinstance(resolved, str) and resolved.strip():
            return str(resolved).strip()
        return None

    def _state_key_from_json(self, state_json: str) -> str:
        trusted_state_key = self._trusted_state_key(state_json)
        if isinstance(trusted_state_key, str) and trusted_state_key:
            return trusted_state_key
        return canonical_state_key(state_json)

    def _intern_state_json(
        self,
        state_json: str,
        *,
        state_key: Optional[str] = None,
    ) -> tuple[int, str]:
        resolved_state_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and str(state_key).strip()
            else self._state_key_from_json(state_json)
        )
        state_store = self._ensure_state_store()
        state_id = state_store.intern(
            state_json,
            state_key=resolved_state_key,
        )
        return int(state_id), str(resolved_state_key)

    def _node_state_json(self, node: SearchNode) -> str:
        state_json = self._ensure_state_store().state_json(node.state_id)
        if isinstance(state_json, str) and state_json:
            return state_json
        raise RuntimeError(f"Missing state_json for node state_id={int(node.state_id)}")

    def _session_frontier_size(self, session: GraphContrastiveWorldSession) -> int:
        return int(session.frontier_index.total_count)

    def _global_frontier_size(self) -> int:
        return int(
            sum(
                self._session_frontier_size(session)
                for session in self._all_world_sessions.values()
            )
        )

    def _graph_edge_count(self) -> int:
        return int(
            sum(
                int(session.world_graph.edge_count)
                for session in self._all_world_sessions.values()
            )
        )

    def _graph_trainable_edge_count(self) -> int:
        return int(
            sum(
                int(session.world_graph.trainable_edge_count)
                for session in self._all_world_sessions.values()
            )
        )

    def _session_is_active(self, session: GraphContrastiveWorldSession) -> bool:
        return self._session_frontier_size(session) > 0

    def _build_world_summary(self, session: GraphContrastiveWorldSession) -> Dict[str, Any]:
        return {
            "world_seed": int(session.world_seed),
            "world_index": int(session.world_index),
            "world_label": str(session.world_label),
            "scenario_type": (
                str(session.scenario_type)
                if isinstance(session.scenario_type, str)
                else None
            ),
            "visit_count": 0,
            "transitions_collected": 0,
            "edge_count": int(session.world_graph.edge_count),
            "trainable_edge_count": int(session.world_graph.trainable_edge_count),
            "unique_states_discovered": int(len(session.state_nodes)),
            "states_expanded": int(len(session.expanded_state_ids)),
            "restore_steps": 0,
            "max_depth": 0,
            "resume_pending": bool(self._session_is_active(session)),
        }

    def _update_world_summary_after_visit(
        self,
        *,
        session: GraphContrastiveWorldSession,
        world_summary: Dict[str, Any],
        transition_count_before: int,
        restore_steps_before: int,
    ) -> None:
        world_summary["visit_count"] = int(world_summary["visit_count"]) + 1
        world_summary["transitions_collected"] = int(world_summary["transitions_collected"]) + max(
            0,
            int(session.world_transition_count) - int(transition_count_before),
        )
        world_summary["unique_states_discovered"] = int(len(session.state_nodes))
        world_summary["states_expanded"] = int(len(session.expanded_state_ids))
        world_summary["edge_count"] = int(session.world_graph.edge_count)
        world_summary["trainable_edge_count"] = int(session.world_graph.trainable_edge_count)
        world_summary["restore_steps"] = int(world_summary["restore_steps"]) + max(
            0,
            int(self.total_restore_steps) - int(restore_steps_before),
        )
        world_summary["max_depth"] = max(
            (int(node.depth) for node in session.state_nodes.values()),
            default=0,
        )
        world_summary["resume_pending"] = bool(self._session_is_active(session))

    def _current_world_step(self, session: GraphContrastiveWorldSession) -> int:
        return int(session.world_transition_count) + 1

    def _depth_limit_allows_state(self, depth: int) -> bool:
        if self.max_step_depth_per_world is None:
            return True
        return int(depth) <= int(self.max_step_depth_per_world)

    def _node_can_expand(self, node: SearchNode) -> bool:
        if self.max_step_depth_per_world is None:
            return True
        return int(node.depth) < int(self.max_step_depth_per_world)

    def _state_tokens(
        self,
        session: GraphContrastiveWorldSession,
        state_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        resolved_state_id = int(state_id)
        node = session.state_nodes.get(resolved_state_id)
        if node is None:
            raise KeyError(f"Unknown state_id in frontier: {resolved_state_id}")
        return self._cached_state_tokens_by_id(resolved_state_id)

    def _frontier_rebuild_state_batch_size(self) -> int:
        action_count = max(1, int(len(self.action_order)))
        target_candidates = max(1, int(self._FRONTIER_REBUILD_TARGET_CANDIDATES))
        state_batch_size = max(1, target_candidates // action_count)
        state_batch_size = min(state_batch_size, int(self._FRONTIER_REBUILD_MAX_STATE_BATCH))
        state_batch_size = max(state_batch_size, int(self._FRONTIER_REBUILD_MIN_STATE_BATCH))
        return int(state_batch_size)

    def _frontier_rebuild_candidate_chunk_size(
        self,
        *,
        replay_count: int,
    ) -> int:
        del replay_count
        return int(self._FRONTIER_REBUILD_CANDIDATE_CHUNK)

    def _frontier_refresh_sample_positions(
        self,
        *,
        total_count: int,
        cap: Optional[int],
        tag: str,
        signature_parts: Sequence[Any],
    ) -> Tuple[int, ...]:
        total = max(0, int(total_count))
        if total <= 0:
            return ()
        if cap is None or int(cap) >= total:
            return tuple(range(total))
        rng = random.Random(
            "|".join(
                str(part)
                for part in (
                    int(self.seed),
                    str(tag),
                    int(cap),
                    int(total),
                    *signature_parts,
                )
            )
        )
        return tuple(sorted(int(index) for index in rng.sample(range(total), int(cap))))

    def _frontier_refresh_sample_candidates(
        self,
        *,
        candidates: Sequence[int],
        cap: Optional[int],
        tag: str,
        signature_parts: Sequence[Any],
    ) -> Tuple[int, ...]:
        resolved_candidates = tuple(int(index) for index in candidates)
        total = len(resolved_candidates)
        if total <= 0:
            return ()
        if cap is None or int(cap) >= total:
            return resolved_candidates
        rng = random.Random(
            "|".join(
                str(part)
                for part in (
                    int(self.seed),
                    str(tag),
                    int(cap),
                    int(total),
                    *signature_parts,
                )
            )
        )
        selected_positions = sorted(int(index) for index in rng.sample(range(total), int(cap)))
        return tuple(int(resolved_candidates[position]) for position in selected_positions)

    def _frontier_refresh_replay_candidate_indices(self) -> Tuple[int, ...]:
        if self.prototype_sample_cap is None:
            return ()
        if (
            not self._active_prototype_sample_indices
            and not self._prototype_learning_bootstrap_completed
            and self.total_steps >= self.learning_starts
        ):
            self._maybe_finalize_learning_start_prototypes()
        if self._prototype_rebuild_pending:
            self._ensure_prototypes_current()
        return tuple(int(index) for index in self._active_prototype_sample_indices)

    def _frontier_refresh_replay_indices_from_candidates(
        self,
        replay_candidate_indices: Sequence[int],
    ) -> Tuple[int, ...]:
        return self._frontier_refresh_sample_candidates(
            candidates=replay_candidate_indices,
            cap=self.replay_cap,
            tag="replay",
            signature_parts=(
                int(self._sample_storage_version),
                int(self._sample_label_version),
                int(self.prototype_sample_cap) if self.prototype_sample_cap is not None else None,
                int(self._prototype_bootstrap_threshold()),
            ),
        )

    def _frontier_refresh_replay_indices(self) -> Tuple[int, ...]:
        if self.prototype_sample_cap is None:
            return self._frontier_refresh_sample_positions(
                total_count=int(len(self.sample_store.storage)),
                cap=self.replay_cap,
                tag="replay",
                signature_parts=(
                    int(self._sample_storage_version),
                    int(self._sample_label_version),
                ),
            )
        candidate_indices = self._frontier_refresh_replay_candidate_indices()
        return self._frontier_refresh_replay_indices_from_candidates(candidate_indices)

    def _encode_frontier_refresh_replay_tau(
        self,
        replay_indices: Sequence[int],
        *,
        replay_candidate_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        resolved_indices = tuple(int(index) for index in replay_indices)
        sample_store_size = int(len(self.sample_store.storage))
        if sample_store_size <= 0 or not resolved_indices:
            return torch.zeros((0, int(self.contrastive_dim)), device=self.device, dtype=torch.float32)
        if self.prototype_sample_cap is not None:
            prototype_indices = (
                tuple(int(index) for index in replay_candidate_indices)
                if replay_candidate_indices is not None
                else self._frontier_refresh_replay_candidate_indices()
            )
            if tuple(resolved_indices) == tuple(prototype_indices):
                prototype_tau = self._encode_prototype_sample_tau_embeddings(
                    prototype_indices
                )
                return _clone_if_inference_tensor(prototype_tau.detach())
            prototype_index_positions = {
                int(storage_index): int(position)
                for position, storage_index in enumerate(prototype_indices)
            }
            replay_positions: List[int] = []
            for storage_index in resolved_indices:
                position = prototype_index_positions.get(int(storage_index))
                if position is None:
                    raise ValueError(
                        "frontier replay indices must be sampled from the same "
                        "prototype replay candidate pool used to encode replay tau."
                    )
                replay_positions.append(int(position))
            prototype_tau = self._encode_prototype_sample_tau_embeddings(
                prototype_indices
            )
            if not replay_positions:
                return torch.zeros(
                    (0, int(self.contrastive_dim)),
                    device=self.device,
                    dtype=torch.float32,
                )
            return _clone_if_inference_tensor(
                prototype_tau.index_select(
                    0,
                    torch.as_tensor(
                        replay_positions,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ).detach()
            )
        full_tau_signature = (
            int(self._dynamics_parameter_version),
            int(sample_store_size),
        )
        cached_full_tau = (
            self._sample_store_tau_cache
            if self._sample_store_tau_cache_signature == full_tau_signature
            else None
        )
        if cached_full_tau is not None:
            return _clone_if_inference_tensor(
                cached_full_tau.index_select(
                    0,
                    torch.as_tensor(
                        resolved_indices,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ).detach()
            )
        samples = [
            self.sample_store.storage[int(index)]
            for index in resolved_indices
        ]
        actions = torch.as_tensor(
            [int(item.action) for item in samples],
            device=self.device,
            dtype=torch.long,
        )
        buckets = tuple(
            self._pack_sample_state_buckets(
                samples,
                next_state=False,
            )
        )
        with torch.no_grad():
            return _clone_if_inference_tensor(
                self._encode_bucketed_state_action_embeddings(
                    buckets=buckets,
                    actions=actions,
                    total_count=len(samples),
                ).detach()
            )

    def _score_tau_embeddings_with_model(
        self,
        model: Any,
        tau_embeddings: torch.Tensor,
        *,
        sample_tau: torch.Tensor,
        context: Optional[ContrastiveEntropyContext],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(tau_embeddings.size(0)) if tau_embeddings.ndim > 0 else 0
        zero_reward = torch.zeros(
            (batch_size,),
            device=tau_embeddings.device,
            dtype=torch.float32,
        )
        if self._frontier_uses_intrinsic_reward() and int(sample_tau.size(0)) > 0:
            tau_embeddings_fp32 = tau_embeddings.to(dtype=torch.float32)
            particle_entropy_reward = self._particle_entropy_intrinsic_reward(
                tau_embeddings_fp32,
                target_keys=sample_tau,
                exclude_self=False,
            )
        else:
            particle_entropy_reward = zero_reward

        if not self._frontier_uses_prototype_entropy_reward(context):
            prototype_entropy_reward = zero_reward
        else:
            logits = model.contrastive_logits(
                keys=tau_embeddings,
                class_indices=(context.class_indices - 1),
            )
            prototype_entropy_reward = self._expected_prototype_entropy_reward_from_logits(
                logits,
                context=context,
            )
        rh = particle_entropy_reward.reshape(-1).to(dtype=torch.float32)
        rz = prototype_entropy_reward.reshape(-1).to(device=rh.device, dtype=torch.float32)
        rh = rh.mul(float(self.intrinsic_reward_scale))
        rz = rz.mul(float(self.prototype_entropy_scale))
        rh = torch.nan_to_num(rh, nan=0.0, posinf=0.0, neginf=0.0)
        rz = torch.nan_to_num(rz, nan=0.0, posinf=0.0, neginf=0.0)
        rtotal = torch.nan_to_num(rh + rz, nan=0.0, posinf=0.0, neginf=0.0)
        return rh, rz, rtotal

    def _frontier_uses_intrinsic_reward(self) -> bool:
        return float(self.intrinsic_reward_scale) != 0.0

    def _frontier_uses_prototype_entropy_reward(
        self,
        context: Optional[ContrastiveEntropyContext],
    ) -> bool:
        if float(self.prototype_entropy_scale) <= 0.0:
            return False
        if context is None or int(context.class_indices.numel()) <= 0:
            return False
        bonus = context.bonus_by_position
        return bool(
            int(bonus.numel()) > 0
            and bool(torch.count_nonzero(bonus.detach()).item())
        )

    def _empty_frontier_sample_tau(self) -> torch.Tensor:
        return torch.zeros(
            (0, int(self.contrastive_dim)),
            device=self.device,
            dtype=torch.float32,
        )

    def _score_tau_embeddings(
        self,
        tau_embeddings: torch.Tensor,
        *,
        sample_tau: torch.Tensor,
        context: Optional[ContrastiveEntropyContext],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._score_tau_embeddings_with_model(
            self.dynamics,
            tau_embeddings,
            sample_tau=sample_tau,
            context=context,
        )

    def _frontier_state_action_token_batches(
        self,
        state_action_entries: Sequence[_FrontierStateActionBatch],
    ) -> Iterable[Tuple[_FrontierStateActionBatch, ...]]:
        state_batch_limit = max(1, int(self._frontier_rebuild_state_batch_size()))
        max_padded_tokens = max(1, int(self._ENCODE_BUCKET_MAX_PADDED_TOKENS))
        max_padding_ratio = max(1.0, float(self._ENCODE_BUCKET_MAX_PADDING_RATIO))
        current_entries: List[_FrontierStateActionBatch] = []
        current_token_sum = 0
        current_max_tokens = 0

        for entry in state_action_entries:
            resolved_actions = tuple(int(action) for action in entry.actions)
            if not resolved_actions:
                continue
            state_tokens, _state_mask = self._cached_state_tokens_by_id(
                int(entry.node.state_id)
            )
            token_count = max(1, int(state_tokens.size(0)))
            candidate_count = len(current_entries) + 1
            candidate_token_sum = current_token_sum + token_count
            candidate_max_tokens = max(current_max_tokens, token_count)
            candidate_padded_tokens = candidate_count * candidate_max_tokens
            candidate_padding_ratio = float(candidate_padded_tokens) / float(
                max(1, candidate_token_sum)
            )
            should_flush = bool(
                current_entries
                and (
                    candidate_count > state_batch_limit
                    or candidate_padded_tokens > max_padded_tokens
                    or candidate_padding_ratio > max_padding_ratio
                )
            )
            if should_flush:
                yield tuple(current_entries)
                current_entries = []
                current_token_sum = 0
                current_max_tokens = 0

            current_entries.append(
                _FrontierStateActionBatch(
                    session=entry.session,
                    node=entry.node,
                    actions=resolved_actions,
                )
            )
            current_token_sum += token_count
            current_max_tokens = max(current_max_tokens, token_count)

        if current_entries:
            yield tuple(current_entries)

    def _score_frontier_state_action_batches(
        self,
        *,
        state_action_entries: Sequence[_FrontierStateActionBatch],
        sample_tau: torch.Tensor,
        context: Optional[ContrastiveEntropyContext] = None,
    ) -> List[
        Tuple[GraphContrastiveWorldSession, SearchNode, int, float, float, float]
    ]:
        resolved_entries: List[_FrontierStateActionBatch] = []
        for entry in state_action_entries:
            resolved_actions = tuple(int(action) for action in entry.actions)
            if not resolved_actions:
                continue
            resolved_entries.append(
                _FrontierStateActionBatch(
                    session=entry.session,
                    node=entry.node,
                    actions=resolved_actions,
                )
            )
        if not resolved_entries:
            return []

        encoded_states = [
            self._cached_state_tokens_by_id(int(entry.node.state_id))
            for entry in resolved_entries
        ]
        state_tokens_tensor, state_mask_tensor = self._pack_cached_state_batch(encoded_states)
        candidate_state_rows: List[int] = []
        candidate_actions: List[int] = []
        for state_row, entry in enumerate(resolved_entries):
            for action in entry.actions:
                candidate_state_rows.append(int(state_row))
                candidate_actions.append(int(action))
        if not candidate_actions:
            return []

        resolved_context = (
            context
            if context is not None
            else self._build_prototype_entropy_context()
        )
        candidate_rows_tensor = torch.as_tensor(
            candidate_state_rows,
            device=self.device,
            dtype=torch.long,
        )
        candidate_actions_tensor = torch.as_tensor(
            candidate_actions,
            device=self.device,
            dtype=torch.long,
        )
        candidate_chunk_size = self._frontier_rebuild_candidate_chunk_size(
            replay_count=int(sample_tau.size(0))
        )
        scored_entries: List[
            Tuple[GraphContrastiveWorldSession, SearchNode, int, float, float, float]
        ] = []

        with torch.inference_mode():
            with self._autocast_context():
                state_context = self.dynamics.encode_state_context(
                    state_tokens_tensor,
                    state_mask_tensor,
                )
            total_candidates = int(candidate_actions_tensor.numel())
            for start_index in range(0, total_candidates, candidate_chunk_size):
                end_index = min(total_candidates, start_index + candidate_chunk_size)
                chunk_rows = candidate_rows_tensor[start_index:end_index]
                chunk_actions = candidate_actions_tensor[start_index:end_index]
                chunk_mask = state_mask_tensor.index_select(0, chunk_rows)
                with self._autocast_context():
                    tau_embeddings = self.dynamics.encode_state_action_from_context(
                        state_context.index_select(0, chunk_rows),
                        chunk_mask,
                        chunk_actions,
                    )
                rh, rz, rtotal = self._score_tau_embeddings(
                    tau_embeddings,
                    sample_tau=sample_tau,
                    context=resolved_context,
                )
                score_rows = (
                    torch.stack(
                        (
                            rh.reshape(-1),
                            rz.reshape(-1),
                            rtotal.reshape(-1),
                        ),
                        dim=1,
                    )
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .tolist()
                )
                for local_index in range(start_index, end_index):
                    state_row = int(candidate_state_rows[local_index])
                    entry = resolved_entries[state_row]
                    chunk_offset = local_index - start_index
                    rh_value, rz_value, score_value = score_rows[int(chunk_offset)]
                    scored_entries.append(
                        (
                            entry.session,
                            entry.node,
                            int(candidate_actions[local_index]),
                            float(rh_value),
                            float(rz_value),
                            float(score_value),
                        )
                    )
        return scored_entries

    def _can_use_distributed_frontier_scoring(
        self,
        *,
        replay_count: int,
        candidate_count: int,
    ) -> bool:
        if self._uses_frozen_initial_representation():
            return False
        if self.device.type != "cuda" or os.name == "nt":
            return False
        if not torch.cuda.is_available():
            return False
        if self._frontier_uses_intrinsic_reward() and int(replay_count) <= 0:
            return False
        if int(candidate_count) <= 0:
            return False
        world_size = int(len(self.parallel_device_ids))
        return (
            world_size > 1
            and int(candidate_count) >= world_size
            and getattr(self, "_distributed_trainer", None) is not None
        )

    def _frontier_score_payload_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        resolved = tensor.detach()
        if resolved.device.type != "cpu":
            resolved = resolved.to(device="cpu")
        if not resolved.is_contiguous():
            resolved = resolved.contiguous()
        return _share_cpu_tensor_if_requested(
            resolved,
            share_memory=_distributed_step_command_should_share_tensors(),
        )

    def _empty_frontier_score_payload(self) -> Dict[str, Any]:
        state_tokens, state_mask = self._empty_host_packed_state_batch()
        return {
            "state_tokens": self._frontier_score_payload_tensor(state_tokens),
            "state_mask": self._frontier_score_payload_tensor(state_mask),
            "candidate_state_rows": self._frontier_score_payload_tensor(
                torch.zeros((0,), device="cpu", dtype=torch.long)
            ),
            "candidate_actions": self._frontier_score_payload_tensor(
                torch.zeros((0,), device="cpu", dtype=torch.long)
            ),
            "candidate_world_indices": self._frontier_score_payload_tensor(
                torch.zeros((0,), device="cpu", dtype=torch.long)
            ),
            "candidate_state_ids": self._frontier_score_payload_tensor(
                torch.zeros((0,), device="cpu", dtype=torch.long)
            ),
            "candidate_orders": self._frontier_score_payload_tensor(
                torch.zeros((0,), device="cpu", dtype=torch.long)
            ),
            "candidate_state_keys": [],
        }

    def _frontier_score_context_payload(
        self,
        context: Optional[ContrastiveEntropyContext],
    ) -> Dict[str, Any]:
        if context is None or int(context.class_indices.numel()) <= 0:
            return {}
        return {
            "class_indices": self._frontier_score_payload_tensor(
                context.class_indices.detach().to(device="cpu", dtype=torch.long)
            ),
            "bonus_by_position": self._frontier_score_payload_tensor(
                context.bonus_by_position.detach().to(device="cpu", dtype=torch.float32)
            ),
        }

    def _frontier_score_config(self, *, replay_count: int) -> Dict[str, Any]:
        candidate_chunk_size = self._frontier_rebuild_candidate_chunk_size(
            replay_count=int(replay_count)
        )
        return {
            "knn_k": int(self.knn_k),
            "knn_avg": bool(self.knn_avg),
            "knn_clip": float(self.knn_clip),
            "intrinsic_reward_scale": float(self.intrinsic_reward_scale),
            "prototype_entropy_scale": float(self.prototype_entropy_scale),
            "prototype_entropy_temperature": float(self.prototype_entropy_temperature),
            "contrastive_temperature": float(self.contrastive_temperature),
            "candidate_chunk_size": int(candidate_chunk_size),
            "state_batch_size": int(self._frontier_rebuild_state_batch_size()),
        }

    def _frontier_score_ordered_candidates(
        self,
        state_action_entries: Sequence[_FrontierStateActionBatch],
    ) -> List[Tuple[GraphContrastiveWorldSession, SearchNode, int, int]]:
        candidates: List[Tuple[GraphContrastiveWorldSession, SearchNode, int, int]] = []
        next_order = int(self._frontier_push_order)
        for state_action_batch in self._frontier_state_action_token_batches(
            state_action_entries
        ):
            for entry in state_action_batch:
                for action in entry.actions:
                    candidates.append(
                        (
                            entry.session,
                            entry.node,
                            int(action),
                            int(next_order),
                        )
                    )
                    next_order += 1
        self._frontier_push_order = int(next_order)
        return candidates

    def _frontier_score_rank_payloads(
        self,
        candidates: Sequence[Tuple[GraphContrastiveWorldSession, SearchNode, int, int]],
        *,
        world_size: int,
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        total_count = int(len(candidates))
        for rank in range(int(world_size)):
            start = (total_count * int(rank)) // int(world_size)
            end = (total_count * (int(rank) + 1)) // int(world_size)
            shard = list(candidates[start:end])
            if not shard:
                payloads.append(self._empty_frontier_score_payload())
                continue

            local_state_rows: Dict[int, int] = {}
            encoded_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            candidate_state_rows: List[int] = []
            candidate_actions: List[int] = []
            candidate_world_indices: List[int] = []
            candidate_state_ids: List[int] = []
            candidate_orders: List[int] = []
            candidate_state_keys: List[str] = []
            for session, node, action, order in shard:
                state_id = int(node.state_id)
                local_row = local_state_rows.get(state_id)
                if local_row is None:
                    local_row = len(encoded_states)
                    local_state_rows[state_id] = int(local_row)
                    encoded_states.append(self._cached_state_tokens_by_id(state_id))
                candidate_state_rows.append(int(local_row))
                candidate_actions.append(int(action))
                candidate_world_indices.append(int(session.world_index))
                candidate_state_ids.append(int(state_id))
                candidate_orders.append(int(order))
                candidate_state_keys.append(str(node.key))

            state_tokens, state_mask = self._pack_cached_state_batch_to_host(
                encoded_states
            )
            payloads.append(
                {
                    "state_tokens": self._frontier_score_payload_tensor(state_tokens),
                    "state_mask": self._frontier_score_payload_tensor(state_mask),
                    "candidate_state_rows": self._frontier_score_payload_tensor(
                        torch.as_tensor(candidate_state_rows, device="cpu", dtype=torch.long)
                    ),
                    "candidate_actions": self._frontier_score_payload_tensor(
                        torch.as_tensor(candidate_actions, device="cpu", dtype=torch.long)
                    ),
                    "candidate_world_indices": self._frontier_score_payload_tensor(
                        torch.as_tensor(candidate_world_indices, device="cpu", dtype=torch.long)
                    ),
                    "candidate_state_ids": self._frontier_score_payload_tensor(
                        torch.as_tensor(candidate_state_ids, device="cpu", dtype=torch.long)
                    ),
                    "candidate_orders": self._frontier_score_payload_tensor(
                        torch.as_tensor(candidate_orders, device="cpu", dtype=torch.long)
                    ),
                    "candidate_state_keys": candidate_state_keys,
                }
            )
        return payloads

    def _merge_distributed_frontier_score_items(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        scored_items: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        worklist_heaps: Dict[
            int,
            List[Tuple[Tuple[float, float, float, int], int, FrontierCandidate]],
        ] = {
            int(session.world_index): []
            for session in sessions
        }

        for item in scored_items:
            world_index = int(item["world_index"])
            budget = max(0, int(budgets.get(world_index, 0)))
            if budget <= 0:
                continue
            order = int(item["order"])
            rh = float(item["rh"])
            rz_pred = float(item["rz_pred"])
            score = float(item["score"])
            priority = (score, rz_pred, rh, -int(order))
            work_item = FrontierCandidate(
                state_id=int(item["state_id"]),
                state_key=str(item["state_key"]),
                action=int(item["action"]),
                rh=float(rh),
                rz_pred=float(rz_pred),
                score=float(score),
                order=int(order),
            )
            heap = worklist_heaps.setdefault(world_index, [])
            heap_entry = (priority, int(order), work_item)
            if len(heap) < budget:
                heapq.heappush(heap, heap_entry)
            elif priority > heap[0][0]:
                heapq.heapreplace(heap, heap_entry)

        return {
            int(session.world_index): tuple(
                heap_entry[2]
                for heap_entry in sorted(
                    worklist_heaps.get(int(session.world_index), ()),
                    key=lambda heap_entry: heap_entry[0],
                    reverse=True,
                )
            )
            for session in sessions
        }

    def _score_frontier_entries_distributed_for_actor_worklists(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        state_action_entries: Sequence[_FrontierStateActionBatch],
        replay_indices: Sequence[int],
        replay_candidate_indices: Sequence[int],
        context: Optional[ContrastiveEntropyContext],
    ) -> Tuple[Dict[int, Tuple[FrontierCandidate, ...]], int]:
        candidates = self._frontier_score_ordered_candidates(state_action_entries)
        if not candidates:
            return (
                {int(session.world_index): () for session in sessions},
                0,
            )
        trainer = self._get_distributed_trainer()
        sample_tau = (
            self._encode_frontier_refresh_replay_tau(
                replay_indices,
                replay_candidate_indices=replay_candidate_indices,
            )
            if replay_indices
            else self._empty_frontier_sample_tau()
        )
        rank_payloads = self._frontier_score_rank_payloads(
            candidates,
            world_size=int(trainer.world_size),
        )
        try:
            result = trainer.score_frontier(
                sample_tau=sample_tau,
                rank_payloads=rank_payloads,
                context=self._frontier_score_context_payload(context),
                score_config=self._frontier_score_config(replay_count=len(replay_indices)),
                budgets={int(key): int(value) for key, value in budgets.items()},
            )
        except BaseException:
            self._close_distributed_trainer()
            raise
        raw_items = result.get("items") or []
        scored_items = [item for item in raw_items if isinstance(item, Mapping)]
        worklists = self._merge_distributed_frontier_score_items(
            sessions=sessions,
            budgets=budgets,
            scored_items=scored_items,
        )
        return worklists, int(result.get("scored_count", len(candidates)))

    def _should_use_frontier_scoring(self) -> bool:
        return bool(self.total_steps >= self.learning_starts and len(self.sample_store) > 0)

    def _train_schedule_name(self) -> str:
        return "post_verify_batch"

    @staticmethod
    def _mark_frontier_changed(session: GraphContrastiveWorldSession) -> None:
        session.frontier_version += 1

    def _frontier_round_signature(self) -> Tuple[Any, ...]:
        return (
            int(self._sample_storage_version),
            int(self._sample_label_version),
            int(self._dynamics_parameter_version),
            int(self._prototype_summary_version),
            tuple(self._active_class_indices()),
            int(self.dynamics.num_classes),
            float(self.intrinsic_reward_scale),
            float(self.prototype_entropy_scale),
            int(self.replay_cap) if self.replay_cap is not None else None,
            int(self.frontier_cap) if self.frontier_cap is not None else None,
            int(self.prototype_sample_cap) if self.prototype_sample_cap is not None else None,
            (
                int(self.prototype_sample_min_per_class)
                if self.prototype_sample_min_per_class is not None
                else None
            ),
            str(self.frontier_sampling_mode),
        )

    def _select_frontier_refresh_state_action_batches(
        self,
        sessions: Sequence[GraphContrastiveWorldSession],
    ) -> _FrontierRefreshSelection:
        candidate_positions_by_world: Dict[int, range] = {}
        selectable_position = 0
        for session in sessions:
            session.warmup_edge_queue.clear()
            world_index = int(session.world_index)
            world_start = int(selectable_position)
            selectable_position += int(session.frontier_index.total_count)
            if selectable_position > world_start:
                candidate_positions_by_world[world_index] = range(
                    world_start,
                    selectable_position,
                )

        selection_signature = self._frontier_refresh_selection_signature(
            sessions
        )
        selected_positions = self._select_frontier_refresh_candidate_positions(
            candidate_positions_by_world=candidate_positions_by_world,
            signature_parts=selection_signature,
        )
        selected_offsets_by_world: Dict[int, Tuple[int, ...]] = {}
        selected_index = 0
        selected_count = len(selected_positions)
        for session in sessions:
            if selected_index >= selected_count:
                break
            world_index = int(session.world_index)
            world_range = candidate_positions_by_world.get(world_index)
            if world_range is None:
                continue
            world_start = int(world_range.start)
            world_stop = int(world_range.stop)
            while (
                selected_index < selected_count
                and int(selected_positions[selected_index]) < world_start
            ):
                selected_index += 1
            offsets: List[int] = []
            while (
                selected_index < selected_count
                and int(selected_positions[selected_index]) < world_stop
            ):
                offsets.append(int(selected_positions[selected_index]) - world_start)
                selected_index += 1
            if offsets:
                selected_offsets_by_world[world_index] = tuple(offsets)

        grouped: Dict[
            Tuple[int, int],
            Tuple[GraphContrastiveWorldSession, SearchNode, List[int]],
        ] = {}
        if selected_offsets_by_world:
            for session in sessions:
                world_index = int(session.world_index)
                selected_offsets = selected_offsets_by_world.get(world_index)
                if not selected_offsets:
                    continue
                for local_position in selected_offsets:
                    state_id, action = session.frontier_index.resolve(
                        int(local_position),
                        self.action_order,
                    )
                    node = session.state_nodes.get(int(state_id))
                    if node is None:
                        raise RuntimeError(
                            "Frontier candidate index referenced a missing SearchNode: "
                            f"world_index={world_index} state_id={int(state_id)}"
                        )
                    group_key = (world_index, int(node.state_id))
                    group = grouped.get(group_key)
                    if group is None:
                        grouped[group_key] = (session, node, [int(action)])
                    else:
                        group[2].append(int(action))

        entries = tuple(
            _FrontierStateActionBatch(
                session=session,
                node=node,
                actions=tuple(actions),
            )
            for session, node, actions in grouped.values()
        )
        return _FrontierRefreshSelection(
            state_action_entries=entries,
            total_candidates=int(selectable_position),
            selected_candidates=len(selected_positions),
        )

    def _frontier_refresh_selection_signature(
        self,
        sessions: Sequence[GraphContrastiveWorldSession],
    ) -> Tuple[Any, ...]:
        return (
            self._frontier_round_signature(),
            tuple(
                (int(session.world_index), int(session.frontier_version))
                for session in sessions
            ),
        )

    def _select_frontier_refresh_candidate_positions(
        self,
        *,
        candidate_positions_by_world: Mapping[int, Sequence[int]],
        signature_parts: Sequence[Any],
    ) -> Tuple[int, ...]:
        missing_candidate_count = sum(
            len(positions)
            for positions in candidate_positions_by_world.values()
        )
        if int(missing_candidate_count) <= 0:
            return ()
        if self.frontier_sampling_mode == "global_uniform":
            return self._frontier_refresh_sample_positions(
                total_count=int(missing_candidate_count),
                cap=self.frontier_cap,
                tag="frontier",
                signature_parts=signature_parts,
            )
        return self._select_map_balanced_frontier_refresh_positions(
            candidate_positions_by_world=candidate_positions_by_world,
            missing_candidate_count=int(missing_candidate_count),
            signature_parts=signature_parts,
        )

    def _select_map_balanced_frontier_refresh_positions(
        self,
        *,
        candidate_positions_by_world: Mapping[int, Sequence[int]],
        missing_candidate_count: int,
        signature_parts: Sequence[Any],
    ) -> Tuple[int, ...]:
        if self.frontier_cap is None or int(self.frontier_cap) >= int(missing_candidate_count):
            return tuple(range(int(missing_candidate_count)))
        cap = max(0, int(self.frontier_cap))
        world_indices = sorted(
            int(world_index)
            for world_index, positions in candidate_positions_by_world.items()
            if len(positions) > 0
        )
        if not world_indices:
            return ()

        rng = random.Random(
            "|".join(
                str(part)
                for part in (
                    int(self.seed),
                    "frontier_maps",
                    int(cap),
                    int(missing_candidate_count),
                    *signature_parts,
                )
            )
        )
        rng.shuffle(world_indices)

        base_quota = cap // len(world_indices)
        remainder = cap % len(world_indices)
        selected: set[int] = set()
        for order_index, world_index in enumerate(world_indices):
            positions = candidate_positions_by_world[int(world_index)]
            quota = int(base_quota + (1 if order_index < remainder else 0))
            quota = min(quota, len(positions))
            if quota <= 0:
                continue
            local_offsets = self._frontier_refresh_sample_positions(
                total_count=len(positions),
                cap=quota,
                tag=f"frontier_map:{int(world_index)}",
                signature_parts=signature_parts,
            )
            selected.update(int(positions[int(offset)]) for offset in local_offsets)

        leftover = cap - len(selected)
        if leftover > 0:
            remaining_count = max(0, int(missing_candidate_count) - len(selected))
            leftover_offsets = self._frontier_refresh_sample_positions(
                total_count=remaining_count,
                cap=leftover,
                tag="frontier_leftover",
                signature_parts=signature_parts,
            )
            leftover_offset_set = {int(offset) for offset in leftover_offsets}
            remaining_offset = 0
            for world_index in sorted(candidate_positions_by_world):
                for position in candidate_positions_by_world[int(world_index)]:
                    resolved_position = int(position)
                    if resolved_position in selected:
                        continue
                    if int(remaining_offset) in leftover_offset_set:
                        selected.add(resolved_position)
                    remaining_offset += 1
        return tuple(sorted(selected))

    def _emit_frontier_refresh_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        *,
        phase: str,
        replay_total: int,
        replay_sampled: int,
        frontier_total: int,
        frontier_sampled: int,
        scored: Optional[int] = None,
        elapsed_sec: Optional[float] = None,
        select_elapsed_sec: Optional[float] = None,
        score_elapsed_sec: Optional[float] = None,
    ) -> None:
        if not callable(progress_callback):
            return
        payload: Dict[str, Any] = {
            "progress_kind": "frontier_refresh",
            "frontier_refresh_phase": str(phase),
            "frontier_replay_total_samples": max(0, int(replay_total)),
            "frontier_replay_sampled_samples": max(0, int(replay_sampled)),
            "frontier_total_candidates": max(0, int(frontier_total)),
            "frontier_sampled_candidates": max(0, int(frontier_sampled)),
        }
        if scored is not None:
            payload["frontier_scored_candidates"] = max(0, int(scored))
        if select_elapsed_sec is not None:
            payload["frontier_select_elapsed_sec"] = max(
                0.0,
                float(select_elapsed_sec),
            )
        if score_elapsed_sec is not None:
            payload["frontier_score_elapsed_sec"] = max(
                0.0,
                float(score_elapsed_sec),
            )
        if elapsed_sec is not None:
            payload["frontier_refresh_elapsed_sec"] = max(0.0, float(elapsed_sec))
        progress_callback(payload)

    def _score_frontier_entries_for_actor_worklists(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        state_action_entries: Sequence[_FrontierStateActionBatch],
        sample_tau: torch.Tensor,
        context: Optional[ContrastiveEntropyContext],
    ) -> Tuple[Dict[int, Tuple[FrontierCandidate, ...]], int]:
        worklist_heaps: Dict[
            int,
            List[Tuple[Tuple[float, float, float, int], int, FrontierCandidate]],
        ] = {
            int(session.world_index): []
            for session in sessions
        }
        scored_count = 0

        def maybe_keep_work_item(
            *,
            session: GraphContrastiveWorldSession,
            node: SearchNode,
            action: int,
            rh: float,
            rz_pred: float,
            score: float,
            order: int,
        ) -> None:
            world_index = int(session.world_index)
            budget = max(0, int(budgets.get(world_index, 0)))
            if budget <= 0:
                return
            priority = (
                float(score),
                float(rz_pred),
                float(rh),
                -int(order),
            )
            heap = worklist_heaps.setdefault(world_index, [])
            if len(heap) >= budget and priority <= heap[0][0]:
                return
            work_item = FrontierCandidate(
                state_id=int(node.state_id),
                state_key=str(node.key),
                action=int(action),
                rh=float(rh),
                rz_pred=float(rz_pred),
                score=float(score),
                order=int(order),
            )
            heap_entry = (priority, int(order), work_item)
            if len(heap) < budget:
                heapq.heappush(heap, heap_entry)
            else:
                heapq.heapreplace(heap, heap_entry)

        for state_action_batch in self._frontier_state_action_token_batches(
            state_action_entries
        ):
            scored_entries = self._score_frontier_state_action_batches(
                state_action_entries=state_action_batch,
                sample_tau=sample_tau,
                context=context,
            )
            scored_count += len(scored_entries)
            for session, node, action, rh, rz_pred, score in scored_entries:
                order = int(self._frontier_push_order)
                self._frontier_push_order += 1
                maybe_keep_work_item(
                    session=session,
                    node=node,
                    action=int(action),
                    rh=float(rh),
                    rz_pred=float(rz_pred),
                    score=float(score),
                    order=int(order),
                )
        worklists = {
            int(session.world_index): tuple(
                heap_entry[2]
                for heap_entry in sorted(
                    worklist_heaps.get(int(session.world_index), ()),
                    key=lambda heap_entry: heap_entry[0],
                    reverse=True,
                )
            )
            for session in sessions
        }
        return worklists, int(scored_count)

    def _build_zero_score_worklists_for_collect(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        state_action_entries: Sequence[_FrontierStateActionBatch],
    ) -> Tuple[Dict[int, Tuple[FrontierCandidate, ...]], int]:
        worklist_heaps: Dict[
            int,
            List[Tuple[Tuple[float, float, float, int], int, FrontierCandidate]],
        ] = {
            int(session.world_index): []
            for session in sessions
        }
        scored_count = 0
        for entry in state_action_entries:
            budget = max(0, int(budgets.get(int(entry.session.world_index), 0)))
            for action in entry.actions:
                order = int(self._frontier_push_order)
                self._frontier_push_order += 1
                scored_count += 1
                if budget <= 0:
                    continue
                priority = (0.0, 0.0, 0.0, -int(order))
                work_item = FrontierCandidate(
                    state_id=int(entry.node.state_id),
                    state_key=str(entry.node.key),
                    action=int(action),
                    rh=0.0,
                    rz_pred=0.0,
                    score=0.0,
                    order=int(order),
                )
                heap = worklist_heaps.setdefault(int(entry.session.world_index), [])
                heap_entry = (priority, int(order), work_item)
                if len(heap) < budget:
                    heapq.heappush(heap, heap_entry)
                elif priority > heap[0][0]:
                    heapq.heapreplace(heap, heap_entry)
        worklists = {
            int(session.world_index): tuple(
                heap_entry[2]
                for heap_entry in sorted(
                    worklist_heaps.get(int(session.world_index), ()),
                    key=lambda heap_entry: heap_entry[0],
                    reverse=True,
                )
            )
            for session in sessions
        }
        return worklists, int(scored_count)

    def _build_scored_worklists_for_collect(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        worklists: Dict[int, Tuple[FrontierCandidate, ...]] = {
            int(session.world_index): ()
            for session in sessions
        }
        if not self._should_use_frontier_scoring():
            return worklists

        refresh_started_at = perf_counter()
        uses_intrinsic_reward = self._frontier_uses_intrinsic_reward()
        replay_candidate_indices = (
            self._frontier_refresh_replay_candidate_indices()
            if uses_intrinsic_reward and self.prototype_sample_cap is not None
            else ()
        )
        replay_total = (
            len(replay_candidate_indices)
            if uses_intrinsic_reward and self.prototype_sample_cap is not None
            else int(len(self.sample_store))
        )
        replay_indices = (
            (
                self._frontier_refresh_replay_indices_from_candidates(replay_candidate_indices)
                if self.prototype_sample_cap is not None
                else self._frontier_refresh_replay_indices()
            )
            if uses_intrinsic_reward
            else ()
        )
        select_started_at = perf_counter()
        selection = self._select_frontier_refresh_state_action_batches(sessions)
        select_elapsed_sec = perf_counter() - select_started_at
        score_started_at = perf_counter()
        self._emit_frontier_refresh_progress(
            progress_callback,
            phase="start",
            replay_total=replay_total,
            replay_sampled=len(replay_indices),
            frontier_total=selection.total_candidates,
            frontier_sampled=selection.selected_candidates,
        )
        scored_count = 0
        if selection.state_action_entries:
            context = self._build_prototype_entropy_context()
            uses_prototype_entropy_reward = self._frontier_uses_prototype_entropy_reward(
                context
            )
            if not uses_intrinsic_reward and not uses_prototype_entropy_reward:
                worklists, scored_count = self._build_zero_score_worklists_for_collect(
                    sessions=sessions,
                    budgets=budgets,
                    state_action_entries=selection.state_action_entries,
                )
            elif self._can_use_distributed_frontier_scoring(
                replay_count=len(replay_indices),
                candidate_count=selection.selected_candidates,
            ):
                worklists, scored_count = self._score_frontier_entries_distributed_for_actor_worklists(
                    sessions=sessions,
                    budgets=budgets,
                    state_action_entries=selection.state_action_entries,
                    replay_indices=replay_indices,
                    replay_candidate_indices=replay_candidate_indices,
                    context=context,
                )
            else:
                worklists, scored_count = self._score_frontier_entries_for_actor_worklists(
                    sessions=sessions,
                    budgets=budgets,
                    state_action_entries=selection.state_action_entries,
                    sample_tau=(
                        self._encode_frontier_refresh_replay_tau(
                            replay_indices,
                            replay_candidate_indices=replay_candidate_indices,
                        )
                        if uses_intrinsic_reward
                        else self._empty_frontier_sample_tau()
                    ),
                    context=context,
                )
        self._emit_frontier_refresh_progress(
            progress_callback,
            phase="end",
            replay_total=replay_total,
            replay_sampled=len(replay_indices),
            frontier_total=selection.total_candidates,
            frontier_sampled=selection.selected_candidates,
            scored=scored_count,
            elapsed_sec=perf_counter() - refresh_started_at,
            select_elapsed_sec=select_elapsed_sec,
            score_elapsed_sec=perf_counter() - score_started_at,
        )
        return worklists

    def _build_warmup_worklists_for_collect(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        worklists: Dict[int, Tuple[FrontierCandidate, ...]] = {}
        for session in sessions:
            world_index = int(session.world_index)
            budget = max(0, int(budgets.get(world_index, 0)))
            candidates: List[FrontierCandidate] = []
            while len(candidates) < budget and session.warmup_edge_queue:
                state_id, action = session.warmup_edge_queue.popleft()
                pending_mask = int(
                    session.pending_action_masks_by_state.get(int(state_id), 0)
                )
                if not bool(pending_mask & (1 << int(action))):
                    continue
                node = session.state_nodes.get(int(state_id))
                if node is None:
                    continue
                candidates.append(
                    FrontierCandidate(
                        state_id=int(node.state_id),
                        state_key=str(node.key),
                        action=int(action),
                        score=0.0,
                        rh=0.0,
                        rz_pred=0.0,
                        order=-1,
                    )
                )
            worklists[world_index] = tuple(candidates)
        return worklists

    def _build_candidate_worklists_for_collect(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        if self._should_use_frontier_scoring():
            return self._build_scored_worklists_for_collect(
                sessions=sessions,
                budgets=budgets,
                progress_callback=progress_callback,
            )
        return self._build_warmup_worklists_for_collect(
            sessions=sessions,
            budgets=budgets,
        )

    def _prepare_state_frontier(
        self,
        *,
        session: GraphContrastiveWorldSession,
        node: SearchNode,
    ) -> None:
        if int(node.state_id) in session.expanded_state_ids:
            return
        if not self._node_can_expand(node):
            return
        session.expanded_state_ids.add(int(node.state_id))
        pending_mask = int(self._all_action_mask)
        if pending_mask <= 0:
            return
        self._set_pending_action_mask(
            session=session,
            state_id=int(node.state_id),
            mask=pending_mask,
        )
        self._mark_frontier_changed(session)
        if self._should_use_frontier_scoring():
            return
        for action in self.action_order:
            session.warmup_edge_queue.append((int(node.state_id), int(action)))

    def _remove_active_world(
        self,
        session: GraphContrastiveWorldSession,
    ) -> None:
        world_index = int(session.world_index)
        self._active_worlds = deque(
            active_session
            for active_session in self._active_worlds
            if int(active_session.world_index) != world_index
        )

    def _mark_executed_state_action(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        action: int,
    ) -> None:
        resolved_state_id = int(state_id)
        resolved_action = int(action)
        pending_mask = session.pending_action_masks_by_state.get(resolved_state_id)
        if pending_mask is None:
            return
        remaining_mask = int(pending_mask) & ~(1 << resolved_action)
        if remaining_mask > 0:
            self._set_pending_action_mask(
                session=session,
                state_id=resolved_state_id,
                mask=remaining_mask,
            )
            self._mark_frontier_changed(session)
            return
        self._clear_pending_action_mask(
            session=session,
            state_id=resolved_state_id,
        )
        self._mark_frontier_changed(session)

    def _record_known_edge(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        action: int,
        next_state_id: int,
        done: bool,
        class_id: Optional[int] = None,
        rh: float = 0.0,
        rz: float = 0.0,
        rtotal: float = 0.0,
    ) -> EdgeRef:
        existing_edge_id = session.world_graph.edge_ids_by_source_state.get(
            int(state_id),
            {},
        ).get(int(action))
        edge_ref = session.world_graph.register_edge(
            source_state_id=int(state_id),
            action=int(action),
            next_state_id=int(next_state_id),
            done=bool(done),
            class_id=class_id,
            rh=float(rh),
            rz=float(rz),
            rtotal=float(rtotal),
            trainable=True,
        )
        self._append_analysis_edge_row(
            iteration_event="add" if existing_edge_id is None else "update",
            session=session,
            edge_ref=edge_ref,
        )
        return edge_ref

    def _propagate_shorter_depths(
        self,
        *,
        session: GraphContrastiveWorldSession,
        start_state_id: int,
    ) -> None:
        queue: Deque[int] = deque([int(start_state_id)])
        while queue:
            state_id = queue.popleft()
            node = session.state_nodes.get(int(state_id))
            if node is None:
                continue
            for action, edge in session.world_graph.outgoing_edges(int(state_id)):
                if bool(edge.done):
                    continue
                child_node = session.state_nodes.get(int(edge.next_state_id))
                if child_node is None:
                    continue
                candidate_depth = int(node.depth) + 1
                if not self._depth_limit_allows_state(candidate_depth):
                    continue
                if candidate_depth >= int(child_node.depth):
                    continue
                updated_child = SearchNode(
                    state_id=int(child_node.state_id),
                    key=str(child_node.key),
                    depth=int(candidate_depth),
                    parent_state_id=int(state_id),
                    parent_action=int(action),
                )
                session.state_nodes[int(child_node.state_id)] = updated_child
                if (
                    int(updated_child.state_id) not in session.expanded_state_ids
                    and self._node_can_expand(updated_child)
                ):
                    self._prepare_state_frontier(
                        session=session,
                        node=updated_child,
                    )
                self._append_analysis_node_row(
                    iteration_event="update_depth",
                    session=session,
                    node=updated_child,
                )
                queue.append(int(child_node.state_id))

    def _relax_discovered_state(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        state_key: str,
        depth: int,
        parent_state_id: int,
        parent_action: int,
    ) -> Optional[SearchNode]:
        if not self._depth_limit_allows_state(depth):
            return None

        existing = session.state_nodes.get(int(state_id))
        if existing is None:
            session.world_graph.register_state(
                state_id=int(state_id),
                state_key=str(state_key),
            )
            node = SearchNode(
                state_id=int(state_id),
                key=str(state_key),
                depth=int(depth),
                parent_state_id=int(parent_state_id),
                parent_action=int(parent_action),
            )
            session.state_nodes[int(state_id)] = node
            self._prepare_state_frontier(
                session=session,
                node=node,
            )
            self._append_analysis_node_row(
                iteration_event="add",
                session=session,
                node=node,
            )
            return node

        session.world_graph.register_state(
            state_id=int(state_id),
            state_key=str(state_key),
        )

        if int(depth) >= int(existing.depth):
            return existing

        updated = SearchNode(
            state_id=int(existing.state_id),
            key=str(existing.key),
            depth=int(depth),
            parent_state_id=int(parent_state_id),
            parent_action=int(parent_action),
        )
        session.state_nodes[int(state_id)] = updated
        if (
            int(updated.state_id) not in session.expanded_state_ids
            and self._node_can_expand(updated)
        ):
            self._prepare_state_frontier(
                session=session,
                node=updated,
            )
        self._append_analysis_node_row(
            iteration_event="update_depth",
            session=session,
            node=updated,
        )
        self._propagate_shorter_depths(
            session=session,
            start_state_id=int(state_id),
        )
        return updated

    def _actor_collect_budgets(
        self,
        *,
        max_transitions: int,
        sessions: Sequence[GraphContrastiveWorldSession],
    ) -> Dict[int, int]:
        session_count = max(1, int(len(sessions)))
        base_budget = int(max_transitions) // session_count
        remainder = int(max_transitions) % session_count
        return {
            int(session.world_index): int(base_budget + (1 if index < remainder else 0))
            for index, session in enumerate(sessions)
        }

    def _empty_collect_transition_assessment(
        self,
        transition_key: str,
    ) -> Dict[str, Any]:
        return {
            "transition_key": str(transition_key),
            "current_version_id": self._current_program_version_id(),
            "current_source_digest": self._current_program_source_digest(),
            "current_explains": False,
            "predicted_next_state_json": None,
            "prediction_error": None,
            "assigned_class_id": 0,
            "assigned_group_id": None,
            "assignment_status": "unknown",
        }

    def _assessment_assignment_fields(
        self,
        assessment: Mapping[str, Any],
    ) -> Tuple[Optional[int], Optional[str], str]:
        if assessment.get("current_version_id") != self._current_program_version_id():
            return None, None, "unknown"
        if assessment.get("current_source_digest") != self._current_program_source_digest():
            return None, None, "unknown"
        if assessment.get("current_explains") is not True:
            return None, None, "unknown"
        class_id = (
            int(assessment.get("assigned_class_id"))
            if isinstance(assessment.get("assigned_class_id"), int)
            and int(assessment.get("assigned_class_id")) > 0
            else None
        )
        group_id = (
            str(assessment.get("assigned_group_id")).strip()
            if isinstance(assessment.get("assigned_group_id"), str)
            and str(assessment.get("assigned_group_id")).strip()
            else None
        )
        status = (
            str(assessment.get("assignment_status"))
            if isinstance(assessment.get("assignment_status"), str)
            and str(assessment.get("assignment_status")).strip()
            else "unknown"
        )
        if class_id is None or group_id is None:
            return None, None, status
        return int(class_id), str(group_id), status

    def _record_collect_transition_assessment(
        self,
        assessment: Mapping[str, Any],
    ) -> None:
        transition_key = assessment.get("transition_key")
        if not isinstance(transition_key, str) or not transition_key:
            return
        if assessment.get("current_version_id") != self._current_program_version_id():
            return
        if assessment.get("current_source_digest") != self._current_program_source_digest():
            return
        self._current_source_transition_assessments[str(transition_key)] = dict(assessment)

    def _prepared_actor_transition_key(
        self,
        prepared: _PreparedActorTransition,
    ) -> str:
        return str(prepared.transition_key)

    def _evaluate_collect_transition_assessments(
        self,
        prepared_transitions: Sequence[_PreparedActorTransition],
    ) -> List[Dict[str, Any]]:
        assessments = [
            self._empty_collect_transition_assessment(
                self._prepared_actor_transition_key(prepared)
            )
            for prepared in prepared_transitions
        ]
        if not prepared_transitions:
            return assessments
        current_source = getattr(self, "_current_program_source", None)
        if not isinstance(current_source, str) or not current_source.strip():
            return assessments

        transitions = [
            prepared.assessment_transition for prepared in prepared_transitions
        ]
        evaluation = self._program_evaluator.evaluate_programs(
            programs=[
                ProgramEvaluationTask(
                    label="current",
                    source=current_source,
                )
            ],
            transitions=transitions,
            collect_records=True,
        ).first().evaluation
        explained_indices: List[int] = []
        for record in list(getattr(evaluation, "records", []) or []):
            index = int(getattr(record, "index", -1))
            if index < 0 or index >= len(assessments):
                continue
            predicted = getattr(record, "predicted_canonical", None)
            is_correct = bool(getattr(record, "is_correct", False))
            assessment = dict(assessments[index])
            assessment["current_explains"] = (
                bool(is_correct)
                and getattr(record, "error", None) is None
            )
            assessment["predicted_next_state_json"] = (
                str(predicted) if isinstance(predicted, str) else None
            )
            assessment["prediction_error"] = self._clone_sandbox_error(
                getattr(record, "error", None)
            )
            if assessment["current_explains"] is True:
                explained_indices.append(index)
            assessments[index] = assessment

        if explained_indices:
            assignments = self._group_classifier.classify_transitions(
                transitions=[transitions[index] for index in explained_indices],
                include_rows=False,
                known_explaining_version_id=(
                    self._group_classifier.snapshot.version_snapshot.current_version_id
                ),
            )
            for index, (assignment, _rows) in zip(explained_indices, assignments):
                assessment = dict(assessments[index])
                if (
                    isinstance(getattr(assignment, "class_id", None), int)
                    and int(assignment.class_id) > 0
                    and isinstance(getattr(assignment, "group_id", None), str)
                    and str(assignment.group_id).strip()
                ):
                    assessment["assigned_class_id"] = int(assignment.class_id)
                    assessment["assigned_group_id"] = str(assignment.group_id)
                    assessment["assignment_status"] = (
                        str(assignment.status)
                        if isinstance(getattr(assignment, "status", None), str)
                        else "assigned"
                    )
                else:
                    assessment["assigned_class_id"] = 0
                    assessment["assigned_group_id"] = None
                    assessment["assignment_status"] = (
                        str(assignment.status)
                        if isinstance(getattr(assignment, "status", None), str)
                        else "unknown"
                    )
                assessments[index] = assessment
        return assessments

    def _materialize_actor_transition_event(
        self,
        *,
        event: ActorTransitionEvent,
        session: GraphContrastiveWorldSession,
        candidate: FrontierCandidate,
    ) -> _PreparedActorTransition:
        source_node = session.state_nodes.get(int(candidate.state_id))
        if source_node is None:
            raise RuntimeError(
                "Actor transition referenced a state id missing from the main mirror. "
                f"world_index={int(session.world_index)} state_id={int(candidate.state_id)}"
            )
        if int(event.source_state_id) != int(candidate.state_id):
            raise RuntimeError(
                "Actor transition source metadata does not match the pending request. "
                f"request_id={int(event.request_id)} actor={int(event.source_state_id)} "
                f"main={int(candidate.state_id)}"
            )
        if str(event.source_state_key) != str(source_node.key):
            raise RuntimeError(
                "Actor transition source key does not match the main mirror. "
                f"request_id={int(event.request_id)}"
            )
        if int(event.source_depth) < int(source_node.depth):
            raise RuntimeError(
                "Actor transition source depth does not match the main mirror. "
                f"request_id={int(event.request_id)} actor={int(event.source_depth)} "
                f"main={int(source_node.depth)}"
            )
        if int(event.candidate_depth) != int(event.source_depth) + 1:
            raise RuntimeError(
                "Actor transition candidate depth is not one step after source depth. "
                f"request_id={int(event.request_id)}"
            )
        if int(event.action) != int(candidate.action):
            raise RuntimeError(
                "Actor transition action metadata does not match the pending request. "
                f"request_id={int(event.request_id)} actor={int(event.action)} "
                f"main={int(candidate.action)}"
            )
        expected_transition_key = self._transition_key_from_fields(
            world_index=int(session.world_index),
            map_name=str(session.world_label).strip() or None,
            state_identity=str(event.source_state_key),
            action_name=str(event.action_name),
            next_state_identity=str(event.next_state_key),
        )
        if str(event.transition_key) != str(expected_transition_key):
            raise RuntimeError(
                "Actor transition key does not match the main graph identity key. "
                f"request_id={int(event.request_id)}"
            )
        state_store = self._ensure_state_store()
        next_state_id = state_store.intern_runtime_state_packet(
            event.next_state_packet,
            state_key=str(event.next_state_key),
        )
        transition = Transition.from_state_ids(
            state_store=state_store,
            state_id=int(candidate.state_id),
            action=str(event.action_name),
            next_state_id=int(next_state_id),
            reward=float(event.reward),
            done=bool(event.done),
            world_index=int(session.world_index),
            map_name=(
                str(event.map_name).strip()
                if isinstance(event.map_name, str) and str(event.map_name).strip()
                else (str(session.world_label).strip() or None)
            ),
        )
        assessment_transition = self._build_actor_assessment_transition(
            event=event,
            candidate=candidate,
            next_state_id=int(next_state_id),
            transition=transition,
        )
        canonical_transition_key = self._transition_key_from_fields(
            world_index=transition.world_index,
            map_name=transition.map_name,
            state_identity=transition.state_key,
            action_name=transition.action,
            next_state_identity=transition.next_state_key,
        )
        return _PreparedActorTransition(
            event=event,
            session=session,
            candidate=candidate,
            transition_key=str(canonical_transition_key),
            next_state_id=int(next_state_id),
            transition=transition,
            assessment_transition=assessment_transition,
        )

    def _build_actor_assessment_transition(
        self,
        *,
        event: ActorTransitionEvent,
        candidate: FrontierCandidate,
        next_state_id: int,
        transition: Transition,
    ) -> Transition:
        state_store = self._ensure_state_store()
        source_state_id = int(candidate.state_id)
        source_packet = state_store.runtime_state_packet(source_state_id)
        vocab = state_store.runtime_state_vocab()
        source_runtime_key = runtime_state_packet_key(source_packet, vocab)
        source_store_key = state_store.state_key(source_state_id)
        next_store_key = state_store.state_key(int(next_state_id))
        if (
            str(source_runtime_key) == str(source_store_key)
            and str(event.next_state_key) == str(next_store_key)
        ):
            return transition
        return Transition.from_state_objects(
            state_obj=runtime_state_packet_obj(source_packet, vocab),
            action=str(event.action_name),
            next_state_obj=runtime_state_packet_obj(event.next_state_packet, vocab),
            reward=float(event.reward),
            done=bool(event.done),
            world_index=transition.world_index,
            map_name=transition.map_name,
            state_key=str(source_runtime_key),
            next_state_key=str(event.next_state_key),
        )

    def _commit_prepared_actor_transition_event(
        self,
        *,
        prepared: _PreparedActorTransition,
        assessment: Mapping[str, Any],
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        event = prepared.event
        session = prepared.session
        candidate = prepared.candidate
        source_node = session.state_nodes.get(int(candidate.state_id))
        if source_node is None:
            raise RuntimeError(
                "Actor transition could not be committed because its source node is missing."
            )
        canonical_transition_key = self._prepared_actor_transition_key(prepared)
        self._current_collect_transition_key = str(canonical_transition_key)
        self._last_visual_board_state_id = int(prepared.next_state_id)
        assigned_class_id, assigned_group_id, assignment_status = (
            self._assessment_assignment_fields(assessment)
        )
        commit = self._prepare_frontier_transition_commit(
            session=session,
            candidate=candidate,
            next_state_id=int(prepared.next_state_id),
            candidate_depth=int(source_node.depth) + 1,
            action_name=str(event.action_name),
            map_name=(
                str(event.map_name).strip()
                if isinstance(event.map_name, str) and str(event.map_name).strip()
                else (str(session.world_label).strip() or None)
            ),
            env_reward=float(event.reward),
            done=bool(event.done),
            assigned_class_id=assigned_class_id,
            assigned_group_id=assigned_group_id,
            assignment_status=assignment_status,
        )
        if commit is None:
            raise RuntimeError(
                "Actor transition could not be committed because its source node is missing."
            )
        self._record_collect_transition_assessment(assessment)
        self._commit_frontier_transition_record(
            session=session,
            candidate=candidate,
            done=bool(event.done),
            transitions=transitions,
            commit=commit,
        )
        self._apply_frontier_transition_graph_update(
            session=session,
            candidate=candidate,
            done=bool(event.done),
            commit=commit,
        )
        self.total_restore_steps += int(event.restore_steps)
        self._publish_frontier_transition_feedback(
            session=session,
            commit=commit,
            done=bool(event.done),
            transitions=transitions,
            progress_callback=progress_callback,
            board_state_id=int(prepared.next_state_id),
        )

    def _commit_actor_transition_event(
        self,
        *,
        event: ActorTransitionEvent,
        session: GraphContrastiveWorldSession,
        candidate: FrontierCandidate,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        prepared = self._materialize_actor_transition_event(
            event=event,
            session=session,
            candidate=candidate,
        )
        self._commit_prepared_actor_transition_event(
            prepared=prepared,
            assessment=self._empty_collect_transition_assessment(
                self._prepared_actor_transition_key(prepared)
            ),
            transitions=transitions,
            progress_callback=progress_callback,
        )

    def _fair_collect_base_map_budget(
        self,
        *,
        max_transitions: int,
        map_count: int,
    ) -> int:
        safe_map_count = max(1, int(map_count))
        safe_budget = max(1, int(max_transitions))
        return max(1, safe_budget // safe_map_count)

    def _collect_fair_map_actor_slices(
        self,
        *,
        max_transitions: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        prototype_refresh_elapsed_sec: float = 0.0,
    ) -> List[Transition]:
        transitions: List[Transition] = []
        collect_prep_started_at = perf_counter()
        self.total_collect_calls += 1
        self._current_collect_transition_key = None
        self._current_collect_question_mark_count = 0
        self._has_seen_question_mark_transition = False

        if int(max_transitions) <= 0:
            self._last_collect_stats = {
                "transitions_collected": 0,
                "stopped_by_max_transitions": False,
                "worlds_started": 0,
                "worlds_processed": 0,
                "world_visits": 0,
                "worlds_completed": 0,
                "worlds_completed_total": int(self.total_worlds_completed),
                "resume_pending": bool(self._active_worlds),
                "warmup_active": bool(not self._should_use_frontier_scoring()),
                "parallel_collect_enabled": True,
                "parallel_collect_actor_enabled": True,
                "parallel_collect_map_slices": 0,
                "prototype_refresh_elapsed_sec": max(
                    0.0,
                    float(prototype_refresh_elapsed_sec),
                ),
                "collect_setup_elapsed_sec": 0.0,
                "collect_prep_elapsed_sec": max(
                    0.0,
                    float(prototype_refresh_elapsed_sec),
                ),
                "world_summaries": [],
            }
            return transitions

        transition_budget = int(max_transitions)
        worlds_started = self._ensure_world_sessions_initialized()
        active_sessions = [
            session
            for session in list(self._active_worlds)
            if self._session_is_active(session)
        ]
        if len(active_sessions) > transition_budget:
            active_sessions = active_sessions[:transition_budget]
        if not active_sessions:
            worker_count = self._effective_collect_workers(active_map_count=0)
            collect_setup_elapsed_sec = max(0.0, perf_counter() - collect_prep_started_at)
            collect_prep_elapsed_sec = max(
                0.0,
                float(prototype_refresh_elapsed_sec),
            ) + float(collect_setup_elapsed_sec)
            self._last_collect_stats = {
                "worlds_started": int(worlds_started),
                "worlds_processed": 0,
                "world_visits": 0,
                "worlds_completed": 0,
                "transitions_collected": 0,
                "stopped_by_max_transitions": False,
                "worlds_completed_total": int(self.total_worlds_completed),
                "resume_pending": bool(len(self._active_worlds) > 0),
                "warmup_active": bool(not self._should_use_frontier_scoring()),
                "parallel_collect_enabled": True,
                "parallel_collect_actor_enabled": True,
                "parallel_collect_workers": int(worker_count),
                "parallel_collect_worker_cap": int(self.collect_workers),
                "parallel_collect_workers_auto": bool(self.collect_workers_auto),
                "parallel_collect_map_slices": 0,
                "prototype_refresh_elapsed_sec": max(
                    0.0,
                    float(prototype_refresh_elapsed_sec),
                ),
                "collect_setup_elapsed_sec": float(collect_setup_elapsed_sec),
                "collect_prep_elapsed_sec": float(collect_prep_elapsed_sec),
                "world_summaries": [],
            }
            raise RuntimeError(
                "Collect invariant violated: positive collect budget but no active "
                "frontier sessions. "
                f"worlds_started={int(worlds_started)} "
                f"active_worlds={int(len(self._active_worlds))} "
                f"global_frontier_size={int(self._global_frontier_size())} "
                f"total_worlds_completed={int(self.total_worlds_completed)}"
            )

        budgets = self._actor_collect_budgets(
            max_transitions=transition_budget,
            sessions=active_sessions,
        )
        candidate_worklists_by_world = self._build_candidate_worklists_for_collect(
            sessions=active_sessions,
            budgets=budgets,
            progress_callback=progress_callback,
        )
        dispatch_sessions = [
            session
            for session in active_sessions
            if candidate_worklists_by_world.get(int(session.world_index), ())
        ]
        world_summaries: Dict[int, Dict[str, Any]] = {
            int(session.world_index): self._build_world_summary(session)
            for session in active_sessions
        }
        worker_count = self._effective_collect_workers(
            active_map_count=len(dispatch_sessions)
        )
        if not dispatch_sessions:
            collect_setup_elapsed_sec = max(0.0, perf_counter() - collect_prep_started_at)
            collect_prep_elapsed_sec = max(
                0.0,
                float(prototype_refresh_elapsed_sec),
            ) + float(collect_setup_elapsed_sec)
            self._last_collect_stats = {
                "worlds_started": int(worlds_started),
                "worlds_processed": int(len(world_summaries)),
                "world_visits": 0,
                "worlds_completed": 0,
                "transitions_collected": 0,
                "stopped_by_max_transitions": False,
                "worlds_completed_total": int(self.total_worlds_completed),
                "resume_pending": bool(len(self._active_worlds) > 0),
                "warmup_active": bool(not self._should_use_frontier_scoring()),
                "parallel_collect_enabled": True,
                "parallel_collect_actor_enabled": True,
                "parallel_collect_workers": int(worker_count),
                "parallel_collect_worker_cap": int(self.collect_workers),
                "parallel_collect_workers_auto": bool(self.collect_workers_auto),
                "parallel_collect_map_slices": 0,
                "prototype_refresh_elapsed_sec": max(
                    0.0,
                    float(prototype_refresh_elapsed_sec),
                ),
                "collect_setup_elapsed_sec": float(collect_setup_elapsed_sec),
                "collect_prep_elapsed_sec": float(collect_prep_elapsed_sec),
                "world_summaries": list(world_summaries.values()),
            }
            raise RuntimeError(
                "Collect invariant violated: active frontier sessions produced no "
                "actor work items. "
                f"active_sessions={int(len(active_sessions))} "
                f"global_frontier_size={int(self._global_frontier_size())} "
                f"warmup_active={bool(not self._should_use_frontier_scoring())} "
                f"world_summaries={list(world_summaries.values())}"
            )

        self._ensure_collect_actor_workers(worker_count=worker_count)
        event_queue = self._collect_actor_event_queue
        if event_queue is None:
            raise RuntimeError("Collect actor event queue was not initialized.")
        pending_by_worker: Dict[
            int,
            Deque[Tuple[GraphContrastiveWorldSession, Tuple[FrontierCandidate, ...]]],
        ] = {
            worker_id: deque()
            for worker_id in range(int(worker_count))
        }
        for session in dispatch_sessions:
            world_index = int(session.world_index)
            worker_id = int(
                self._collect_actor_world_worker.get(
                    world_index,
                    world_index % worker_count,
                )
            )
            worker_id = int(worker_id % worker_count)
            self._collect_actor_world_worker[world_index] = worker_id
            pending_by_worker[worker_id].append(
                (
                    session,
                    tuple(candidate_worklists_by_world.get(int(world_index), ())),
                )
            )

        active_worker_world: Dict[int, int] = {}
        active_worker_task: Dict[
            int,
            Tuple[GraphContrastiveWorldSession, Tuple[FrontierCandidate, ...]],
        ] = {}
        pending_requests: Dict[int, Tuple[GraphContrastiveWorldSession, FrontierCandidate]] = {}
        request_counter = 0
        state_store = self._ensure_state_store()
        state_vocab = state_store.runtime_state_vocab()

        def _start_next(worker_id: int) -> None:
            nonlocal request_counter
            queue_for_worker = pending_by_worker[int(worker_id)]
            if not queue_for_worker:
                active_worker_world.pop(int(worker_id), None)
                active_worker_task.pop(int(worker_id), None)
                return
            session, candidates = queue_for_worker.popleft()
            world_index = int(session.world_index)
            handle = self._collect_actor_workers[int(worker_id)]
            dispatch_id = int(
                self._collect_actor_dispatch_ids_by_worker.get(int(worker_id), 0) + 1
            )
            self._collect_actor_dispatch_ids_by_worker[int(worker_id)] = int(dispatch_id)
            state_ref_by_id: Dict[int, int] = {}
            states: List[ActorStateRef] = []
            items: List[ActorStepWorkItem] = []
            for candidate in candidates:
                node = session.state_nodes.get(int(candidate.state_id))
                if node is None:
                    raise RuntimeError(
                        "Actor worklist references a state missing from the main mirror. "
                        f"world_index={int(world_index)} state_id={int(candidate.state_id)}"
                    )
                state_ref = state_ref_by_id.get(int(candidate.state_id))
                if state_ref is None:
                    state_ref = len(states)
                    state_ref_by_id[int(candidate.state_id)] = int(state_ref)
                    states.append(
                        ActorStateRef(
                            state_ref=int(state_ref),
                            state_id=int(candidate.state_id),
                            state_key=str(node.key),
                            source_depth=int(node.depth),
                            packet=state_store.runtime_state_packet(
                                int(candidate.state_id)
                            ),
                        )
                    )
                request_id = int(request_counter)
                request_counter += 1
                pending_requests[request_id] = (session, candidate)
                items.append(
                    ActorStepWorkItem(
                        request_id=int(request_id),
                        state_ref=int(state_ref),
                        action=int(candidate.action),
                        action_name=str(self.action_names[int(candidate.action)]),
                    )
                )
            handle.command_queue.put(
                ActorStartCollect(
                    dispatch_id=int(dispatch_id),
                    world_index=int(world_index),
                    state_vocab=state_vocab,
                    states=tuple(states),
                    items=tuple(items),
                    map_name=str(session.world_label).strip() or None,
                )
            )
            active_worker_world[int(worker_id)] = world_index
            active_worker_task[int(worker_id)] = (session, candidates)

        for worker_id in range(int(worker_count)):
            _start_next(worker_id)

        collect_setup_elapsed_sec = max(0.0, perf_counter() - collect_prep_started_at)
        collect_prep_elapsed_sec = max(
            0.0,
            float(prototype_refresh_elapsed_sec),
        ) + float(collect_setup_elapsed_sec)
        if callable(progress_callback):
            progress_callback(
                {
                    "progress_kind": "collect_prep",
                    "collect_prep_phase": "end",
                    "prototype_refresh_elapsed_sec": max(
                        0.0,
                        float(prototype_refresh_elapsed_sec),
                    ),
                    "collect_setup_elapsed_sec": float(collect_setup_elapsed_sec),
                    "collect_prep_elapsed_sec": float(collect_prep_elapsed_sec),
                    "collect_prep_active_maps": int(len(active_sessions)),
                    "collect_prep_dispatch_maps": int(len(dispatch_sessions)),
                    "collect_prep_workers": int(worker_count),
                    "collect_prep_budget": int(transition_budget),
                }
            )

        worlds_completed = 0
        world_visits = 0

        pending_prepared_transitions: List[_PreparedActorTransition] = []

        def _record_transition_batch_event(event: ActorTransitionBatchEvent) -> None:
            nonlocal world_visits
            for transition_event in event.transitions:
                request_context = pending_requests.pop(
                    int(transition_event.request_id),
                    None,
                )
                if request_context is None:
                    raise RuntimeError(
                        "Collect actor returned an unknown transition request id: "
                        f"{int(transition_event.request_id)}."
                    )
                session_for_event, candidate = request_context
                pending_prepared_transitions.append(
                    self._materialize_actor_transition_event(
                        event=transition_event,
                        session=session_for_event,
                        candidate=candidate,
                    )
                )
                world_visits += 1

        def _requeue_active_worker_task(worker_id: int) -> None:
            task = active_worker_task.pop(int(worker_id), None)
            active_worker_world.pop(int(worker_id), None)
            if task is None:
                return
            session, candidates = task
            pending_candidate_ids = {
                id(candidate)
                for request_session, candidate in pending_requests.values()
                if request_session is session
            }
            remaining_candidates = tuple(
                candidate for candidate in candidates if id(candidate) in pending_candidate_ids
            )
            if not remaining_candidates:
                return
            stale_request_ids = [
                request_id
                for request_id, (request_session, _candidate) in pending_requests.items()
                if request_session is session
            ]
            for request_id in stale_request_ids:
                pending_requests.pop(int(request_id), None)
            pending_by_worker[int(worker_id)].appendleft((session, remaining_candidates))

        while active_worker_world:
            timeout = 0.005
            try:
                event = event_queue.get(timeout=timeout)
            except queue.Empty:
                for worker_id in list(active_worker_world.keys()):
                    handle = self._collect_actor_workers[int(worker_id)]
                    if not handle.runner.is_alive():
                        _requeue_active_worker_task(int(worker_id))
                        self._respawn_collect_actor_worker(worker_id=int(worker_id))
                        _start_next(int(worker_id))
                event = None

            if isinstance(event, ActorTransitionBatchEvent):
                active_dispatch_id = self._collect_actor_dispatch_ids_by_worker.get(
                    int(event.worker_id)
                )
                if int(event.dispatch_id) != int(active_dispatch_id or 0):
                    continue
                _record_transition_batch_event(event)
            elif isinstance(event, ActorDoneEvent):
                active_dispatch_id = self._collect_actor_dispatch_ids_by_worker.get(
                    int(event.worker_id)
                )
                if int(event.dispatch_id) != int(active_dispatch_id or 0):
                    continue
                active_worker_world.pop(int(event.worker_id), None)
                active_worker_task.pop(int(event.worker_id), None)
                _start_next(int(event.worker_id))
            elif isinstance(event, ActorErrorEvent):
                active_dispatch_id = self._collect_actor_dispatch_ids_by_worker.get(
                    int(event.worker_id)
                )
                if (
                    isinstance(event.dispatch_id, int)
                    and int(event.dispatch_id) != int(active_dispatch_id or 0)
                ):
                    continue
                raise RuntimeError(
                    "Collect actor failed "
                    f"worker={int(event.worker_id)} world={event.world_index}: "
                    f"{event.message}\n{event.traceback_text}"
                )
            elif event is not None:
                raise RuntimeError(f"Unexpected collect actor event: {type(event).__name__}.")

        if pending_requests:
            raise RuntimeError(
                f"Collect actor finished with {len(pending_requests)} unreturned transition(s)."
            )
        assessments = self._evaluate_collect_transition_assessments(
            pending_prepared_transitions
        )
        for prepared, assessment in zip(pending_prepared_transitions, assessments):
            self._commit_prepared_actor_transition_event(
                prepared=prepared,
                assessment=assessment,
                transitions=transitions,
                progress_callback=progress_callback,
            )
            summary = world_summaries.get(int(prepared.session.world_index))
            if isinstance(summary, dict):
                summary["visit_count"] = int(summary["visit_count"]) + 1
                summary["restore_steps"] = int(summary["restore_steps"]) + int(
                    prepared.event.restore_steps
                )

        for session in active_sessions:
            summary = world_summaries.get(int(session.world_index))
            if isinstance(summary, dict):
                summary["transitions_collected"] = int(session.world_transition_count)
                summary["unique_states_discovered"] = int(len(session.state_nodes))
                summary["states_expanded"] = int(len(session.expanded_state_ids))
                summary["edge_count"] = int(session.world_graph.edge_count)
                summary["trainable_edge_count"] = int(
                    session.world_graph.trainable_edge_count
                )
                summary["max_depth"] = max(
                    (int(node.depth) for node in session.state_nodes.values()),
                    default=0,
                )
                summary["resume_pending"] = self._session_frontier_size(session) > 0
            if self._session_is_active(session) and self._session_frontier_size(session) <= 0:
                self._remove_active_world(session)
                worlds_completed += 1
                self.total_worlds_completed += 1
                if isinstance(summary, dict):
                    self._last_completed_world_summary = dict(summary)

        next_active_worlds: Deque[GraphContrastiveWorldSession] = deque()
        for session in self._active_worlds:
            if self._session_is_active(session):
                next_active_worlds.append(session)
        self._active_worlds = next_active_worlds
        stopped_by_max_transitions = len(transitions) >= int(transition_budget)
        base_map_budget = self._fair_collect_base_map_budget(
            max_transitions=transition_budget,
            map_count=len(active_sessions),
        )
        remainder_map_count = int(transition_budget) % max(1, int(len(active_sessions)))
        self._last_collect_stats = {
            "worlds_started": int(worlds_started),
            "worlds_processed": int(len(world_summaries)),
            "world_visits": int(world_visits),
            "worlds_completed": int(worlds_completed),
            "transitions_collected": int(len(transitions)),
            "stopped_by_max_transitions": bool(stopped_by_max_transitions),
            "worlds_completed_total": int(self.total_worlds_completed),
            "resume_pending": bool(len(self._active_worlds) > 0),
            "warmup_active": bool(not self._should_use_frontier_scoring()),
            "parallel_collect_enabled": True,
            "parallel_collect_actor_enabled": True,
            "parallel_collect_workers": int(worker_count),
            "parallel_collect_worker_cap": int(self.collect_workers),
            "parallel_collect_workers_auto": bool(self.collect_workers_auto),
            "fair_collect_base_map_budget": int(base_map_budget),
            "fair_collect_remainder_map_count": int(remainder_map_count),
            "parallel_collect_map_slices": int(len(dispatch_sessions)),
            "prototype_refresh_elapsed_sec": max(
                0.0,
                float(prototype_refresh_elapsed_sec),
            ),
            "collect_setup_elapsed_sec": float(collect_setup_elapsed_sec),
            "collect_prep_elapsed_sec": float(collect_prep_elapsed_sec),
            "collect_prep_active_maps": int(len(active_sessions)),
            "collect_prep_dispatch_maps": int(len(dispatch_sessions)),
            "collect_prep_workers": int(worker_count),
            "collect_prep_budget": int(transition_budget),
            "world_summaries": list(world_summaries.values()),
        }
        return transitions

    def collect(
        self,
        max_transitions: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_on_unknown_transition: bool = False,
    ) -> List[Transition]:
        del stop_on_unknown_transition

        if not isinstance(max_transitions, int) or int(max_transitions) <= 0:
            raise ValueError(
                "GraphContrastiveAgent.collect requires a positive max_transitions "
                "global budget."
            )
        prototype_refresh_started_at = perf_counter()
        self._ensure_prototypes_current()
        prototype_refresh_elapsed_sec = perf_counter() - prototype_refresh_started_at
        return self._collect_fair_map_actor_slices(
            max_transitions=int(max_transitions),
            progress_callback=progress_callback,
            prototype_refresh_elapsed_sec=float(prototype_refresh_elapsed_sec),
        )

    def _prepare_frontier_transition_commit(
        self,
        *,
        session: GraphContrastiveWorldSession,
        candidate: FrontierCandidate,
        next_state_id: int,
        candidate_depth: int,
        action_name: str,
        map_name: Optional[str],
        env_reward: float,
        done: bool,
        assigned_class_id: Optional[int] = None,
        assigned_group_id: Optional[str] = None,
        assignment_status: Optional[str] = None,
    ) -> Optional[_FrontierTransitionCommit]:
        node = session.state_nodes.get(int(candidate.state_id))
        if node is None:
            return None

        expected_action_name = self.action_names[int(candidate.action)]
        if str(action_name) != str(expected_action_name):
            raise RuntimeError(
                "Collect actor action metadata does not match main action names. "
                f"action={int(candidate.action)} actor={action_name} main={expected_action_name}"
            )
        resolved_next_state_id = int(next_state_id)
        next_key = self._ensure_state_store().state_key(resolved_next_state_id)
        sample = self._build_contrastive_sample_from_ids(
            state_id=int(candidate.state_id),
            action=int(candidate.action),
            done=bool(done),
            next_state_id=resolved_next_state_id,
            source_world_index=int(session.world_index),
            source_world_seed=int(session.world_seed),
        )
        sample = self._stamp_sample_world_metadata(
            sample,
            session=session,
        )
        if isinstance(assigned_class_id, int) and int(assigned_class_id) > 0:
            sample.class_id = int(assigned_class_id)
            sample.leaf_group_id = (
                str(assigned_group_id).strip()
                if isinstance(assigned_group_id, str)
                and str(assigned_group_id).strip()
                else None
            )
            sample.assignment_status = (
                str(assignment_status)
                if isinstance(assignment_status, str)
                and str(assignment_status).strip()
                else "assigned"
            )
        self._record_observed_sample(sample)
        display_class_id = (
            int(assigned_class_id)
            if isinstance(assigned_class_id, int)
            and int(assigned_class_id) > 0
            else 0
        )
        transition_rh = float(candidate.rh)
        transition_rz = float(candidate.rz_pred)
        transition_intrinsic_reward = float(candidate.score)
        transition_result = {
            "display_class_id": int(display_class_id),
            "unknown_transition": bool(display_class_id <= 0),
            "class_rows": None,
            "state_action_duplicate_count": None,
            "rh": float(transition_rh),
            "rz": float(transition_rz),
            "rtotal": float(transition_intrinsic_reward),
        }

        display_step = self._current_world_step(session)
        return _FrontierTransitionCommit(
            node=node,
            next_state_id=resolved_next_state_id,
            next_key=str(next_key),
            candidate_depth=int(candidate_depth),
            action_name=str(action_name),
            map_name=map_name,
            transition_result=transition_result,
            transition_env_reward=float(env_reward),
            transition_rh=float(transition_rh),
            transition_rz=float(transition_rz),
            transition_intrinsic_reward=float(transition_intrinsic_reward),
            display_step=int(display_step),
        )

    def _commit_frontier_transition_record(
        self,
        *,
        session: GraphContrastiveWorldSession,
        candidate: FrontierCandidate,
        done: bool,
        transitions: List[Transition],
        commit: _FrontierTransitionCommit,
    ) -> None:
        session.world_transition_count += 1
        transitions.append(
            Transition.from_state_ids(
                state_store=self._ensure_state_store(),
                state_id=int(candidate.state_id),
                action=str(commit.action_name),
                next_state_id=int(commit.next_state_id),
                reward=float(commit.transition_env_reward),
                done=bool(done),
                world_index=int(session.world_index),
                map_name=commit.map_name,
            )
        )
        self.total_steps += 1
        self.total_transitions_collected += 1
        self._last_observed_class_index = (
            int(commit.transition_result.get("display_class_id"))
            if isinstance(commit.transition_result.get("display_class_id"), int)
            and int(commit.transition_result.get("display_class_id")) > 0
            else None
        )

    def _apply_frontier_transition_graph_update(
        self,
        *,
        session: GraphContrastiveWorldSession,
        candidate: FrontierCandidate,
        done: bool,
        commit: _FrontierTransitionCommit,
    ) -> None:
        self._mark_executed_state_action(
            session=session,
            state_id=int(candidate.state_id),
            action=int(candidate.action),
        )
        self._record_known_edge(
            session=session,
            state_id=int(candidate.state_id),
            action=int(candidate.action),
            next_state_id=int(commit.next_state_id),
            done=bool(done),
            class_id=(
                int(commit.transition_result.get("display_class_id"))
                if isinstance(commit.transition_result.get("display_class_id"), int)
                and int(commit.transition_result.get("display_class_id")) > 0
                else None
            ),
            rh=float(commit.transition_rh),
            rz=float(commit.transition_rz),
            rtotal=float(commit.transition_intrinsic_reward),
        )
        if not bool(done):
            self._relax_discovered_state(
                session=session,
                state_id=int(commit.next_state_id),
                state_key=str(commit.next_key),
                depth=int(commit.candidate_depth),
                parent_state_id=int(commit.node.state_id),
                parent_action=int(candidate.action),
            )

    def _publish_frontier_transition_feedback(
        self,
        *,
        session: GraphContrastiveWorldSession,
        commit: _FrontierTransitionCommit,
        done: bool,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        board_state_id: Optional[int] = None,
    ) -> None:
        live_map_frontier_size = int(self._session_frontier_size(session))
        total_frontier_size = int(self._global_frontier_size())

        self._render_collect_transition(
            episode_index=int(session.world_index),
            step_index=int(commit.display_step),
            action_name=str(commit.action_name),
            display_class_id=(
                int(commit.transition_result.get("display_class_id"))
                if isinstance(commit.transition_result.get("display_class_id"), int)
                else None
            ),
            state_action_duplicate_count=None,
            rh=float(commit.transition_rh),
            rz=float(commit.transition_rz),
            rtotal=float(commit.transition_intrinsic_reward),
            done=bool(done),
            class_rows=commit.transition_result.get("class_rows"),
            frontier_size=int(total_frontier_size),
            live_map_frontier_size=int(live_map_frontier_size),
            map_name=commit.map_name,
            board_state_id=board_state_id,
        )

        if callable(progress_callback):
            progress_callback(
                {
                    "transitions_collected": int(len(transitions)),
                    "source": self.strategy_name,
                    "world_index": int(session.world_index),
                    "world_seed": int(session.world_seed),
                    "world_label": str(session.world_label),
                    "frontier_size": int(self._session_frontier_size(session)),
                    "global_frontier_size": int(total_frontier_size),
                    "live_map_frontier_size": int(live_map_frontier_size),
                    "known_class_count": int(self.known_class_count),
                    "active_class_count": int(self.active_class_count),
                    "unknown_transition": bool(
                        commit.transition_result.get("unknown_transition", False)
                    ),
                }
            )

    def _restore_node(
        self,
        session: GraphContrastiveWorldSession,
        node: SearchNode,
    ) -> str:
        restore_runtime_packet = getattr(self.env, "restore_runtime_packet", None)
        capture_runtime_packet = getattr(self.env, "capture_runtime_packet", None)
        if not callable(restore_runtime_packet) or not callable(capture_runtime_packet):
            raise RuntimeError(
                "GraphContrastiveAgent environment must provide runtime packet restore methods."
            )

        state_store = self._ensure_state_store()
        vocab = state_store.runtime_state_vocab()
        restore_runtime_packet(
            state_store.runtime_state_packet(int(node.state_id)),
            vocab,
            step_count=int(node.depth),
        )
        self.total_restore_steps += 1

        restored_packet = capture_runtime_packet(vocab)
        restored_state_id = state_store.intern_runtime_state_packet(restored_packet)
        restored_state_key = state_store.state_key(int(restored_state_id))
        if restored_state_key != node.key:
            raise RuntimeError(
                "Restored state does not match the stored frontier node. "
                f"expected={node.key} got={restored_state_key}"
        )
        return state_store.state_json(int(restored_state_id))

    def _build_visualization_metrics(
        self,
        *,
        episode_index: int,
        step_index: int,
        action_name: str,
        dynamics_class_index: Optional[int],
        state_action_duplicate_count: Optional[int],
        rh: float,
        rz: float,
        rtotal: float,
        done: bool,
        frontier_size: int,
        live_map_frontier_size: int,
        world_index: Optional[int] = None,
        map_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        del episode_index
        del state_action_duplicate_count
        self._record_live_reward_metric_sample(
            rh=float(rh),
            rz=float(rz),
            rtotal=float(rtotal),
        )
        live_reward_stats = self._live_reward_metric_stats()
        contrastive_loss = float(self._last_train_stats.get("contrastive_loss", 0.0))
        top1_accuracy = float(self._last_train_stats.get("prototype_top1_accuracy", 0.0))
        mean_positive_logit = float(self._last_train_stats.get("mean_positive_logit", 0.0))
        mean_max_negative_logit = float(self._last_train_stats.get("mean_max_negative_logit", 0.0))
        rh_mean = float(live_reward_stats["rh_mean"])
        rh_std = float(live_reward_stats["rh_std"])
        rz_mean = float(live_reward_stats["rz_mean"])
        rz_std = float(live_reward_stats["rz_std"])
        rtotal_mean = float(live_reward_stats["rtotal_mean"])
        rtotal_std = float(live_reward_stats["rtotal_std"])
        metrics = {
            "world_index": (
                int(world_index)
                if isinstance(world_index, int) and int(world_index) > 0
                else None
            ),
            "map_name": (
                str(map_name).strip()
                if isinstance(map_name, str) and str(map_name).strip()
                else None
            ),
            "step": int(step_index),
            "global_step": int(self.total_steps),
            "program_version": self._current_program_version_id(),
            "action": str(action_name),
            "progress_phase": "collect",
            "frontier_size": max(0, int(frontier_size)),
            "live_map_frontier_size": max(0, int(live_map_frontier_size)),
            "current_dynamics_class": (
                int(dynamics_class_index)
                if isinstance(dynamics_class_index, int) and int(dynamics_class_index) > 0
                else None
            ),
            "active_dynamics_classes": int(self.active_class_count),
            "known_dynamics_classes": int(self.known_class_count),
            "sample_store_size": int(len(self.sample_store)),
            "edge_count": int(self._graph_edge_count()),
            "trainable_edge_count": int(self._graph_trainable_edge_count()),
            "sample_duplicate_skips": int(self.sample_store.duplicate_skips),
            "contrastive_loss": float(contrastive_loss),
            "prototype_top1_accuracy": float(top1_accuracy),
            "mean_positive_logit": float(mean_positive_logit),
            "mean_max_negative_logit": float(mean_max_negative_logit),
            "rh": float(rh),
            "rz": float(rz),
            "rtotal": float(rtotal),
            "rh_mean": float(rh_mean),
            "rh_std": float(rh_std),
            "rz_mean": float(rz_mean),
            "rz_std": float(rz_std),
            "rtotal_mean": float(rtotal_mean),
            "rtotal_std": float(rtotal_std),
            "train_phase": self._current_train_phase(),
            "train_schedule": self._train_schedule_name(),
            "train_iter": int(self.total_updates),
            "learning_starts": int(self.learning_starts),
            "learning_starts_unit": "transitions",
            "learning_progress": int(self.total_steps),
            "done": bool(done),
        }
        self._append_iteration_step_metric(metrics)
        return metrics

    @staticmethod
    def _resolve_live_map_name(
        session: Optional[GraphContrastiveWorldSession],
    ) -> Optional[str]:
        if session is None:
            return None
        label = str(session.world_label or "").strip()
        return label or None

    def _append_iteration_step_metric(self, metrics: Mapping[str, Any]) -> None:
        self._iteration_step_metrics.append(dict(metrics))

    def _append_analysis_node_row(
        self,
        *,
        iteration_event: str,
        session: GraphContrastiveWorldSession,
        node: SearchNode,
    ) -> None:
        if not self._analysis_archive_enabled:
            return
        parent_state_id = (
            int(node.parent_state_id)
            if isinstance(node.parent_state_id, int)
            and int(node.parent_state_id) > 0
            else None
        )
        parent_action = (
            int(node.parent_action)
            if isinstance(node.parent_action, int)
            and int(node.parent_action) >= 0
            else None
        )
        self._analysis_node_rows.append(
            {
                "row_type": "node",
                "event": str(iteration_event),
                "world_index": int(session.world_index),
                "state_id": int(node.state_id),
                "state_key": str(node.key),
                "depth": int(node.depth),
                "parent_state_id": parent_state_id,
                "parent_action": parent_action,
            }
        )

    def _append_analysis_edge_row(
        self,
        *,
        iteration_event: str,
        session: GraphContrastiveWorldSession,
        edge_ref: EdgeRef,
    ) -> None:
        if not self._analysis_archive_enabled:
            return
        edge = session.world_graph.edges.get(int(edge_ref.edge_id))
        if edge is None:
            return
        action_name = self.action_names[int(edge.action)]
        self._analysis_edge_rows.append(
            {
                "row_type": "edge",
                "event": str(iteration_event),
                "world_index": int(edge.ref.world_index),
                "edge_id": int(edge.ref.edge_id),
                "source_state_id": int(edge.source_state_id),
                "action": int(edge.action),
                "action_name": str(action_name),
                "next_state_id": int(edge.next_state_id),
                "done": bool(edge.done),
                "class_id": (
                    int(edge.class_id)
                    if isinstance(edge.class_id, int) and int(edge.class_id) > 0
                    else None
                ),
                "rh": float(edge.rh),
                "rz": float(edge.rz),
                "rtotal": float(edge.rtotal),
                "trainable": bool(edge.trainable),
            }
        )

    def _merge_iteration_summary_metric_source(self, metrics: Mapping[str, Any]) -> None:
        for key in self._ITERATION_SUMMARY_METRIC_KEYS:
            if key in metrics:
                self._iteration_summary_metrics[key] = metrics.get(key)

    @staticmethod
    def _summarize_dynamics_class_transition_counts(
        class_rows: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for row in class_rows:
            if not isinstance(row, Mapping):
                continue
            raw_class_id = row.get("version_index")
            if not isinstance(raw_class_id, int) or int(raw_class_id) <= 0:
                continue
            raw_count = row.get("transition_count")
            transition_count = (
                int(raw_count)
                if isinstance(raw_count, int) and int(raw_count) >= 0
                else 0
            )
            display_id = (
                str(row.get("display_id")).strip()
                if isinstance(row.get("display_id"), str)
                and str(row.get("display_id")).strip()
                else None
            )
            group_id = (
                str(row.get("group_id")).strip()
                if isinstance(row.get("group_id"), str)
                and str(row.get("group_id")).strip()
                else None
            )
            commit_version = (
                str(row.get("commit_version")).strip()
                if isinstance(row.get("commit_version"), str)
                and str(row.get("commit_version")).strip()
                else None
            )
            summaries.append(
                {
                    "class_id": int(raw_class_id),
                    "class_label": display_id or group_id or str(raw_class_id),
                    "group_id": group_id,
                    "display_id": display_id,
                    "commit_version": commit_version,
                    "transition_count": int(transition_count),
                }
            )
        return summaries

    def _build_iteration_metric_rows(
        self,
        iteration_summary: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        iteration_program_version = iteration_summary.get("program_version")
        for step_metrics in self._iteration_step_metrics:
            metrics = dict(step_metrics)
            for key, value in self._iteration_summary_metrics.items():
                metrics[key] = value
            row: Dict[str, Any] = {
                "global_step": metrics.get("global_step"),
                "iter": iteration_summary.get("iteration"),
                "train_iter": metrics.get("train_iter"),
                "program_version": metrics.get("program_version"),
                "iteration_end_program_version": iteration_program_version,
                "llm_calls": iteration_summary.get("llm_calls"),
                "contrastive_loss": metrics.get("contrastive_loss"),
                "prototype_top1_accuracy": metrics.get("prototype_top1_accuracy"),
                "rh": metrics.get("rh"),
                "rz": metrics.get("rz"),
                "rtotal": metrics.get("rtotal"),
                "frontier_size": metrics.get("frontier_size"),
                "live_map_frontier_size": metrics.get("live_map_frontier_size"),
                "sample_store_size": metrics.get("sample_store_size"),
                "world_index": metrics.get("world_index"),
                "map_name": metrics.get("map_name"),
                "step": metrics.get("step"),
                "action": metrics.get("action"),
                "done": metrics.get("done"),
                "current_dynamics_class": metrics.get("current_dynamics_class"),
                "active_dynamics_classes": metrics.get("active_dynamics_classes"),
                "known_dynamics_classes": metrics.get("known_dynamics_classes"),
                "dynamics_class_transition_counts": metrics.get(
                    "dynamics_class_transition_counts"
                ),
            }
            rows.append(
                {
                    key: row.get(key)
                    for key in self._ITERATION_METRIC_KEYS
                }
            )
        return rows

    def record_iteration_metric_snapshot(
        self,
        *,
        output_dir: str | Path,
        iteration_summary: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        raw_iteration = iteration_summary.get("iteration")
        if not isinstance(raw_iteration, int):
            return None
        iteration = int(raw_iteration)
        if self._last_recorded_iteration_metric == iteration:
            return None
        rows = self._build_iteration_metric_rows(iteration_summary)
        node_rows = list(self._analysis_node_rows)
        edge_rows = list(self._analysis_edge_rows)
        self._iteration_step_metrics = []
        self._iteration_summary_metrics = {}
        self._analysis_node_rows = []
        self._analysis_edge_rows = []
        if not rows and not node_rows and not edge_rows and not self._analysis_archive_enabled:
            return None
        archive_writer = self._ensure_analysis_archive_writer(output_dir)
        result: Dict[str, Any] = {}
        if archive_writer is not None:
            result["analysis_archive"] = archive_writer.record_iteration(
                iteration=iteration,
                metric_rows=rows,
                iteration_summary=iteration_summary,
                node_rows=node_rows,
                edge_rows=edge_rows,
                state_store=self._state_store,
            )
        self._last_recorded_iteration_metric = int(iteration)
        return result or None

    def _should_prepare_live_visual_payload(self) -> bool:
        visualizer = getattr(self, "visualizer", None)
        checker = getattr(visualizer, "should_prepare_snapshot_payload", None)
        if callable(checker):
            return bool(checker())
        return bool(getattr(visualizer, "enabled", False))

    def _render_collect_transition(
        self,
        *,
        episode_index: int,
        step_index: int,
        action_name: str,
        display_class_id: Optional[int],
        state_action_duplicate_count: Optional[int],
        rh: float,
        rz: float,
        rtotal: float,
        done: bool,
        class_rows: Optional[List[Dict[str, Any]]],
        frontier_size: int,
        live_map_frontier_size: int,
        map_name: Optional[str] = None,
        board_state_id: Optional[int] = None,
    ) -> None:
        step_metrics = self._build_visualization_metrics(
            episode_index=episode_index,
            step_index=step_index,
            action_name=action_name,
            dynamics_class_index=(
                int(display_class_id)
                if isinstance(display_class_id, int) and int(display_class_id) > 0
                else None
            ),
            state_action_duplicate_count=state_action_duplicate_count,
            rh=float(rh),
            rz=float(rz),
            rtotal=float(rtotal),
            done=done,
            frontier_size=int(frontier_size),
            live_map_frontier_size=int(live_map_frontier_size),
            world_index=int(episode_index),
            map_name=map_name,
        )
        prepare_live_payload = self._should_prepare_live_visual_payload()
        step_projection_payload = (
            self._maybe_build_transition_projection_payload()
            if prepare_live_payload
            else self._last_projection_payload
        )
        board_state = (
            self._ensure_state_store().state_obj(int(board_state_id))
            if prepare_live_payload
            and isinstance(board_state_id, int)
            and int(board_state_id) > 0
            else None
        )
        update_payload = getattr(self.visualizer, "update_payload_only", None)
        if callable(update_payload):
            payload_update: Dict[str, Any] = {
                "caption": (
                    f"{self.strategy_name} | world={episode_index} "
                    f"step={step_index} action={action_name} "
                    f"rh={float(rh):.3f} rz={float(rz):.3f} rtotal={float(rtotal):.3f} done={done}"
                ),
                "metrics": step_metrics,
                "visitation_heatmap": step_projection_payload,
                "board_state": board_state,
            }
            if class_rows is not None:
                payload_update["class_rows"] = class_rows
            update_payload(**payload_update)
            return

        self.visualizer.render(
            caption=(
                f"{self.strategy_name} | world={episode_index} "
                f"step={step_index} action={action_name} "
                f"rh={float(rh):.3f} rz={float(rz):.3f} rtotal={float(rtotal):.3f} done={done}"
            ),
            metrics=step_metrics,
            visitation_heatmap=step_projection_payload,
            class_rows=class_rows,
            board_state=board_state,
        )

    def finalize_explained_transition_batch(
        self,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        del transitions
        if self.total_steps < self.learning_starts:
            return {
                "train_updates_completed": 0,
                "train_schedule": self._train_schedule_name(),
                "train_phase": self._current_train_phase(),
            }
        updates_completed = self._run_train_update_batch(
            progress_callback=progress_callback,
            desc="Train iter",
        )
        if updates_completed > 0:
            self._publish_latest_train_metrics_to_visualizer(
                progress_phase="verify",
            )
        return {
            "train_updates_completed": int(updates_completed),
            "train_schedule": self._train_schedule_name(),
            "train_phase": self._current_train_phase(),
        }

    def set_collection_feedback(self, summary: Optional[Dict[str, Any]]) -> None:
        if not isinstance(summary, dict):
            return
        raw_added_count = summary.get("added_count")
        if isinstance(raw_added_count, int):
            self._last_added_count = int(raw_added_count)
        self._group_classifier.update_canonical_assignment_stats(
            canonical_class_counts=summary.get("canonical_class_counts"),
            canonical_leaf_group_counts=summary.get("canonical_leaf_group_counts"),
            canonical_classified_count=summary.get("canonical_classified_count"),
            canonical_unassigned_count=summary.get("canonical_unassigned_count"),
        )
        feedback_payload = self._build_collection_feedback_visualizer_payload(
            progress_phase="verify",
        )
        self._merge_iteration_summary_metric_source(feedback_payload["metrics"])
        self.visualizer.update_payload_only(
            caption=(f"{self.strategy_name} | pre_patch samples={len(self.sample_store)}"),
            metrics=feedback_payload["metrics"],
            visitation_heatmap=feedback_payload["visitation_heatmap"],
            class_rows=feedback_payload["class_rows"],
            merge_metrics=True,
            update_dashboard=False,
        )

    def _build_collection_feedback_visualizer_payload(
        self,
        *,
        force_projection_recompute: bool = False,
        progress_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        if force_projection_recompute:
            projection_payload = self._build_transition_projection_payload()
        else:
            projection_payload = self._maybe_build_transition_projection_payload()
        if projection_payload is not None:
            self._last_projection_payload = projection_payload
        predicted_class_index = (
            int(projection_payload.get("current_predicted_class_index"))
            if isinstance(projection_payload, dict)
            and isinstance(projection_payload.get("current_predicted_class_index"), int)
            and int(projection_payload.get("current_predicted_class_index")) > 0
            else None
        )
        predicted_class_confidence = (
            float(projection_payload.get("current_predicted_class_probability"))
            if isinstance(projection_payload, dict)
            and isinstance(projection_payload.get("current_predicted_class_probability"), (int, float))
            else None
        )
        class_rows = self._annotate_class_counts(
            self._build_class_rows(visible_count=max(self.active_class_count, self.known_class_count))
        )
        class_rows = self._decorate_visualization_class_rows(class_rows)
        queue_head_session = self._active_worlds[0] if self._active_worlds else None
        global_frontier_size = int(self._global_frontier_size())
        live_map_frontier_size = (
            int(self._session_frontier_size(queue_head_session))
            if queue_head_session is not None
            else 0
        )
        live_reward_stats = self._live_reward_metric_stats()
        metrics: Dict[str, Any] = {
            "world_index": (
                int(queue_head_session.world_index)
                if queue_head_session is not None
                else None
            ),
            "map_name": self._resolve_live_map_name(queue_head_session),
            "global_step": int(self.total_steps),
            "frontier_size": int(global_frontier_size),
            "live_map_frontier_size": int(live_map_frontier_size),
            "sample_store_size": int(len(self.sample_store)),
            "edge_count": int(self._graph_edge_count()),
            "trainable_edge_count": int(self._graph_trainable_edge_count()),
            "sample_duplicate_skips": int(self.sample_store.duplicate_skips),
            "active_class_count": int(self.active_class_count),
            "active_dynamics_classes": int(self.active_class_count),
            "known_dynamics_classes": int(self.known_class_count),
            "predicted_dynamics_class": predicted_class_index,
            "predicted_dynamics_confidence": predicted_class_confidence,
            "softmax_temperature": float(self.contrastive_temperature),
            "added_count": int(self._last_added_count),
            "train_phase": self._current_train_phase(),
            "train_iter": int(self.total_updates),
            "contrastive_loss": float(self._last_train_stats.get("contrastive_loss", 0.0)),
            "prototype_top1_accuracy": float(self._last_train_stats.get("prototype_top1_accuracy", 0.0)),
            "mean_positive_logit": float(self._last_train_stats.get("mean_positive_logit", 0.0)),
            "mean_max_negative_logit": float(self._last_train_stats.get("mean_max_negative_logit", 0.0)),
            "rh_mean": float(live_reward_stats["rh_mean"]),
            "rh_std": float(live_reward_stats["rh_std"]),
            "rz_mean": float(live_reward_stats["rz_mean"]),
            "rz_std": float(live_reward_stats["rz_std"]),
            "rtotal_mean": float(live_reward_stats["rtotal_mean"]),
            "rtotal_std": float(live_reward_stats["rtotal_std"]),
            "tsne_points": (
                int(projection_payload.get("sampled_sample_store_size", 0))
                if isinstance(projection_payload, dict)
                else 0
            ),
        }
        if isinstance(progress_phase, str) and progress_phase.strip():
            metrics["progress_phase"] = str(progress_phase).strip().lower()
        metrics["dynamics_class_transition_counts"] = (
            self._summarize_dynamics_class_transition_counts(class_rows)
        )
        return {
            "metrics": metrics,
            "visitation_heatmap": projection_payload,
            "class_rows": class_rows,
        }

    def prepare_final_dashboard_snapshot(self) -> None:
        visualizer = getattr(self, "visualizer", None)
        if visualizer is None:
            return
        update_payload_only = getattr(visualizer, "update_payload_only", None)
        if not callable(update_payload_only):
            return
        feedback_payload = self._build_collection_feedback_visualizer_payload(
            force_projection_recompute=True,
        )
        board_state = (
            self._ensure_state_store().state_obj(int(self._last_visual_board_state_id))
            if isinstance(self._last_visual_board_state_id, int)
            and int(self._last_visual_board_state_id) > 0
            else None
        )
        update_payload_only(
            metrics=feedback_payload["metrics"],
            visitation_heatmap=feedback_payload["visitation_heatmap"],
            class_rows=feedback_payload["class_rows"],
            board_state=board_state,
            merge_metrics=True,
            update_dashboard=False,
        )

    def _publish_latest_train_metrics_to_visualizer(
        self,
        *,
        progress_phase: Optional[str] = None,
    ) -> None:
        visualizer = getattr(self, "visualizer", None)
        if visualizer is None:
            return
        update_payload_only = getattr(visualizer, "update_payload_only", None)
        if not callable(update_payload_only):
            return
        feedback_payload = self._build_collection_feedback_visualizer_payload(
            progress_phase=progress_phase,
        )
        self._merge_iteration_summary_metric_source(feedback_payload["metrics"])
        update_payload_only(
            metrics=feedback_payload["metrics"],
            visitation_heatmap=feedback_payload["visitation_heatmap"],
            class_rows=feedback_payload["class_rows"],
            merge_metrics=True,
            update_dashboard=True,
            force_snapshot=True,
            bypass_snapshot_delivery_gate=True,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        frontier_size = int(self._global_frontier_size())
        queue_head_session = self._active_worlds[0] if self._active_worlds else None
        live_map_frontier_size = (
            int(self._session_frontier_size(queue_head_session))
            if queue_head_session is not None
            else 0
        )
        last_train = dict(self._last_train_stats)
        return {
            "name": self.strategy_name,
            "collection_topology": self.collection_topology,
            "transition_batch_scope": self.transition_batch_scope,
            "current_version_id": self._current_program_version_id(),
            "seed": int(self.seed),
            "device": str(self.device),
            "dynamics_encoder_embed_dim": int(self.dynamics_encoder_embed_dim),
            "dynamics_encoder_num_heads": int(self.dynamics_encoder_num_heads),
            "dynamics_encoder_num_blocks": int(self.dynamics_encoder_num_blocks),
            "dynamics_pool_seeds": int(self.dynamics_pool_seeds),
            "dynamics_encoder_dropout": float(self.dynamics_encoder_dropout),
            "world_count": int(len(self._world_specs)),
            "max_step_depth_per_world": (
                int(self.max_step_depth_per_world)
                if isinstance(self.max_step_depth_per_world, int)
                else None
            ),
            "collect_workers": self._collect_workers_config_value(),
            "collect_worker_cap": int(self.collect_workers),
            "parallel_collect_enabled": True,
            "action_order": [int(action) for action in self.action_order],
            "total_steps": int(self.total_steps),
            "total_updates": int(self.total_updates),
            "total_restore_steps": int(self.total_restore_steps),
            "total_transitions_collected": int(self.total_transitions_collected),
            "total_collect_calls": int(self.total_collect_calls),
            "total_worlds_completed": int(self.total_worlds_completed),
            "learning_starts": int(self.learning_starts),
            "learning_starts_unit": "transitions",
            "learning_progress": int(self.total_steps),
            "train_schedule": self._train_schedule_name(),
            "contrastive_batch_size": int(self.contrastive_batch_size),
            "contrastive_train_mode": str(self.contrastive_train_mode),
            "contrastive_representation_mode": str(self.contrastive_representation_mode),
            "encode_bucket_mode": str(self.encode_bucket_mode),
            "replay_cap": int(self.replay_cap) if self.replay_cap is not None else None,
            "frontier_cap": int(self.frontier_cap) if self.frontier_cap is not None else None,
            "frontier_sampling_mode": str(self.frontier_sampling_mode),
            "contrastive_relation_learning_enabled": not bool(
                self._uses_frozen_initial_representation()
            ),
            "contrastive_sampler": "class_balanced_pairs",
            "train_phase": self._current_train_phase(),
            "map_reset_count": int(self._map_reset_count),
            "sample_store_size": int(len(self.sample_store)),
            "edge_count": int(self._graph_edge_count()),
            "trainable_edge_count": int(self._graph_trainable_edge_count()),
            "sample_store_mode": "append_only",
            "sample_deduplicate_exact": bool(self.sample_store.deduplicate_exact),
            "sample_duplicate_skips": int(self.sample_store.duplicate_skips),
            "num_dynamics_classes": int(self.num_dynamics_classes),
            "contrastive_dim": int(self.contrastive_dim),
            "max_prototypes_per_class": int(self.max_prototypes_per_class),
            "prototype_split_base_count": int(self.prototype_split_base_count),
            "prototype_split_min_cluster_occupancy": int(self.prototype_split_min_cluster_occupancy),
            "prototype_sample_cap": int(self.prototype_sample_cap) if self.prototype_sample_cap is not None else None,
            "prototype_sample_min_per_class": (
                int(self.prototype_sample_min_per_class)
                if self.prototype_sample_min_per_class is not None
                else None
            ),
            "num_dynamics_prototypes": int(self.dynamics.total_active_prototype_count()),
            "contrastive_temperature": float(self.contrastive_temperature),
            "contrastive_min_class_count": int(self.contrastive_min_class_count),
            "intrinsic_reward_scale": float(self.intrinsic_reward_scale),
            "prototype_entropy_scale": float(self.prototype_entropy_scale),
            "prototype_entropy_temperature": float(self.prototype_entropy_temperature),
            "prototype_entropy_min_class_count": int(self.prototype_entropy_min_class_count),
            "prototype_entropy_eps": float(self.prototype_entropy_eps),
            "knn_k": int(self.knn_k),
            "knn_avg": bool(self.knn_avg),
            "knn_clip": float(self.knn_clip),
            "knn_exclude_self": bool(self.knn_exclude_self),
            "dynamics_action_conditioning": "multi_seed_plus_action_delta",
            "learning_rate": float(self.learning_rate),
            "contrastive_lr": float(self.contrastive_lr),
            "known_class_count": int(self.known_class_count),
            "active_class_count": int(self.active_class_count),
            "canonical_dynamics_classes": int(self._group_classifier.canonical_class_count),
            "canonical_classified_transitions": int(self._group_classifier.canonical_classified_count),
            "canonical_unassigned_transitions": int(self._group_classifier.canonical_unassigned_count),
            "trainable_contrastive_class_count": int(len(self._trainable_contrastive_class_indices())),
            "provisional_contrastive_class_count": int(len(self._provisional_contrastive_class_indices())),
            "last_added_count": int(self._last_added_count),
            "current_dynamics_class": (
                int(self._last_observed_class_index)
                if isinstance(self._last_observed_class_index, int) and int(self._last_observed_class_index) > 0
                else None
            ),
            "predicted_dynamics_class": (
                int(self._last_predicted_class_index)
                if isinstance(self._last_predicted_class_index, int) and int(self._last_predicted_class_index) > 0
                else None
            ),
            "predicted_dynamics_confidence": (
                float(self._last_predicted_class_confidence)
                if isinstance(self._last_predicted_class_confidence, (int, float))
                else None
            ),
            "loaded_contrastive_checkpoint_path": self._loaded_contrastive_checkpoint_path,
            "loaded_contrastive_checkpoint_program_version_id": (
                self._loaded_contrastive_checkpoint_program_version_id
            ),
            "last_saved_training_artifacts": dict(self._last_saved_training_artifacts),
            "last_version_training_snapshot_path": self._last_version_training_snapshot_path,
            "frontier_size": int(frontier_size),
            "live_map_frontier_size": int(live_map_frontier_size),
            "active_world_count": int(len(self._active_worlds)),
            "active_world_indices": [int(session.world_index) for session in self._active_worlds],
            "target_worlds": self._build_target_world_payloads(),
            "warmup_active": bool(not self._should_use_frontier_scoring()),
            "last_collect": dict(self._last_collect_stats),
            "last_completed_world": dict(self._last_completed_world_summary),
            "last_train": last_train,
            "visualization": self.visualizer.get_diagnostics(),
        }

    def _build_target_world_payloads(self) -> List[Dict[str, Any]]:
        active_world_indices = {int(session.world_index) for session in self._active_worlds}
        queue_head_session = self._active_worlds[0] if self._active_worlds else None
        queue_head_world_index = (
            int(queue_head_session.world_index)
            if queue_head_session is not None
            else None
        )
        round_robin_order_by_world_index = {
            int(world_index): int(order)
            for order, world_index in enumerate(self._round_robin_world_indices, start=1)
        }
        ordered_specs = sorted(
            self._world_specs,
            key=lambda spec: (
                round_robin_order_by_world_index[int(spec.world_index)],
                int(spec.world_index),
            ),
        )
        payloads: List[Dict[str, Any]] = []
        for spec in ordered_specs:
            session = self._all_world_sessions.get(int(spec.world_index))
            preview_state_json = None
            if (
                session is not None
                and isinstance(session.root_state_id, int)
                and int(session.root_state_id) > 0
            ):
                preview_state_json = self._ensure_state_store().state_json(int(session.root_state_id))
            map_name = (
                self._resolve_live_map_name(session)
                if session is not None
                else str(spec.world_label).strip()
            )
            payloads.append(
                {
                    "world_index": int(spec.world_index),
                    "round_robin_order": round_robin_order_by_world_index.get(
                        int(spec.world_index)
                    ),
                    "world_seed": int(spec.world_seed),
                    "world_label": str(spec.world_label),
                    "scenario_type": (
                        str(spec.scenario_type)
                        if isinstance(spec.scenario_type, str) and str(spec.scenario_type).strip()
                        else None
                    ),
                    "map_name": map_name,
                    "transition_count": (
                        int(session.world_transition_count)
                        if session is not None
                        else 0
                    ),
                    "is_active": int(spec.world_index) in active_world_indices,
                    "is_current": (
                        queue_head_world_index is not None
                        and int(spec.world_index) == int(queue_head_world_index)
                    ),
                    "resume_pending": (
                        bool(session is not None and self._session_frontier_size(session) > 0)
                    ),
                    "preview_state_json": (
                        str(preview_state_json)
                        if isinstance(preview_state_json, str) and preview_state_json
                        else None
                    ),
                }
            )
        return payloads
