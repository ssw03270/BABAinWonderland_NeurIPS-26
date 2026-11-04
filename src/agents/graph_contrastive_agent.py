"""Graph-based frontier search contrastive explorer."""

from __future__ import annotations

import hashlib
import heapq
import math
import multiprocessing
import os
import pickle
import queue
import random
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
    _padded_cost_equalized_block_bounds,
    _share_cpu_tensor_if_requested,
    _token_budget_state_bucket_rows,
)
from .world_graph import EdgeRef, WorldGraph
from src.data import (
    StateStore,
    canonical_state_key,
    runtime_state_packet_key,
    runtime_state_packet_obj,
)
from src.data.transition_buffer import Transition
from .analysis_archive import (
    DEFAULT_ANALYSIS_ARCHIVE_COMPRESS_LEVEL,
    AnalysisArchiveWriter,
    normalize_compress_level,
)
from src.program_model import ProgramEvaluationTask
from src.environments.map_display_names import resolve_map_display_name


# Frozen ON PURPOSE. ``order`` is the candidate's refresh-local ring position and
# breaks exact score ties in FIFO order. A ranked candidate must keep that identity
# unchanged from pruning through actor dispatch.
@dataclass(frozen=True, slots=True)
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
class _UniformFrontierTreeNode:
    sort_key: Tuple[str, int]
    state_id: int
    priority: int
    candidate_count: int
    subtree_candidate_count: int
    left: Optional["_UniformFrontierTreeNode"] = None
    right: Optional["_UniformFrontierTreeNode"] = None


@dataclass(slots=True)
class _UniformFrontierIndex:
    """Order-statistic tree keyed by stable state content, not actor arrival order."""

    root: Optional[_UniformFrontierTreeNode] = None
    nodes_by_state_id: Dict[int, _UniformFrontierTreeNode] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return self._subtree_count(self.root)

    @staticmethod
    def _subtree_count(node: Optional[_UniformFrontierTreeNode]) -> int:
        return int(node.subtree_candidate_count) if node is not None else 0

    @classmethod
    def _refresh_node(cls, node: _UniformFrontierTreeNode) -> None:
        node.subtree_candidate_count = (
            int(node.candidate_count)
            + cls._subtree_count(node.left)
            + cls._subtree_count(node.right)
        )

    @staticmethod
    def _stable_priority(state_key: str) -> int:
        digest = hashlib.blake2b(str(state_key).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    @staticmethod
    def _heap_key(node: _UniformFrontierTreeNode) -> Tuple[int, Tuple[str, int]]:
        return int(node.priority), node.sort_key

    @classmethod
    def _rotate_left(cls, node: _UniformFrontierTreeNode) -> _UniformFrontierTreeNode:
        replacement = node.right
        if replacement is None:
            raise RuntimeError("Uniform frontier treap cannot rotate left without a child.")
        node.right = replacement.left
        replacement.left = node
        cls._refresh_node(node)
        cls._refresh_node(replacement)
        return replacement

    @classmethod
    def _rotate_right(cls, node: _UniformFrontierTreeNode) -> _UniformFrontierTreeNode:
        replacement = node.left
        if replacement is None:
            raise RuntimeError("Uniform frontier treap cannot rotate right without a child.")
        node.left = replacement.right
        replacement.right = node
        cls._refresh_node(node)
        cls._refresh_node(replacement)
        return replacement

    @classmethod
    def _insert_node(
        cls,
        root: Optional[_UniformFrontierTreeNode],
        inserted: _UniformFrontierTreeNode,
    ) -> _UniformFrontierTreeNode:
        if root is None:
            return inserted
        if inserted.sort_key < root.sort_key:
            root.left = cls._insert_node(root.left, inserted)
            if cls._heap_key(root.left) < cls._heap_key(root):
                root = cls._rotate_right(root)
        else:
            root.right = cls._insert_node(root.right, inserted)
            if cls._heap_key(root.right) < cls._heap_key(root):
                root = cls._rotate_left(root)
        cls._refresh_node(root)
        return root

    @classmethod
    def _update_count(
        cls,
        root: Optional[_UniformFrontierTreeNode],
        *,
        sort_key: Tuple[str, int],
        count: int,
    ) -> _UniformFrontierTreeNode:
        if root is None:
            raise RuntimeError("Uniform frontier state disappeared from its index.")
        if sort_key < root.sort_key:
            root.left = cls._update_count(root.left, sort_key=sort_key, count=count)
        elif sort_key > root.sort_key:
            root.right = cls._update_count(root.right, sort_key=sort_key, count=count)
        else:
            root.candidate_count = max(0, int(count))
        cls._refresh_node(root)
        return root

    def set_count(self, *, state_id: int, state_key: str, action_count: int) -> None:
        resolved_state_id = int(state_id)
        sort_key = (str(state_key), resolved_state_id)
        resolved_count = max(0, int(action_count))
        existing = self.nodes_by_state_id.get(resolved_state_id)
        if existing is None:
            if resolved_count <= 0:
                return
            inserted = _UniformFrontierTreeNode(
                sort_key=sort_key,
                state_id=resolved_state_id,
                priority=self._stable_priority(sort_key[0]),
                candidate_count=resolved_count,
                subtree_candidate_count=resolved_count,
            )
            self.root = self._insert_node(self.root, inserted)
            self.nodes_by_state_id[resolved_state_id] = inserted
            return
        if existing.sort_key != sort_key:
            raise RuntimeError(
                "Uniform frontier state identity changed after insertion: "
                f"state_id={resolved_state_id}."
            )
        if int(existing.candidate_count) == resolved_count:
            return
        self.root = self._update_count(
            self.root,
            sort_key=sort_key,
            count=resolved_count,
        )

    def remove_state(self, state_id: int) -> None:
        node = self.nodes_by_state_id.get(int(state_id))
        if node is None or int(node.candidate_count) <= 0:
            return
        self.root = self._update_count(
            self.root,
            sort_key=node.sort_key,
            count=0,
        )

    def resolve(
        self,
        local_position: int,
        *,
        pending_masks_by_state: Mapping[int, int],
        action_order: Sequence[int],
    ) -> Tuple[int, int]:
        if not 0 <= int(local_position) < int(self.total_count):
            raise IndexError(
                f"frontier local_position out of range: {int(local_position)}"
            )
        remaining = int(local_position)
        node = self.root
        while node is not None:
            left_count = self._subtree_count(node.left)
            if remaining < left_count:
                node = node.left
                continue
            remaining -= left_count
            if remaining < int(node.candidate_count):
                state_id = int(node.state_id)
                mask = int(pending_masks_by_state.get(state_id, 0))
                for action in action_order:
                    resolved_action = int(action)
                    if mask & (1 << resolved_action):
                        if remaining == 0:
                            return state_id, resolved_action
                        remaining -= 1
                break
            remaining -= int(node.candidate_count)
            node = node.right
        raise RuntimeError("Uniform frontier index diverged from the pending masks.")


@dataclass(slots=True)
class GraphContrastiveWorldSession:
    spec: WorldSpec
    root_state_id: Optional[int] = None
    world_graph: WorldGraph = field(init=False)
    state_nodes: Dict[int, SearchNode] = field(default_factory=dict)
    expanded_state_ids: set[int] = field(default_factory=set)
    pending_action_masks_by_state: Dict[int, int] = field(default_factory=dict)
    frontier_size: int = 0
    frontier_revision: int = 0
    uniform_frontier_index: _UniformFrontierIndex = field(
        default_factory=_UniformFrontierIndex
    )
    world_transition_count: int = 0
    warmup_edge_queue: Deque[Tuple[int, int]] = field(default_factory=deque)
    # Used only by the score_topk_per_map policy.
    # FIFO of (state_id, action). Entries whose pending bit is already cleared are
    # tombstones: they are skipped on pop and dropped by compaction. The invariant
    # "pending bit set <=> live entry in this ring" keeps frontier_size exact.
    frontier_ring: Deque[Tuple[int, int]] = field(default_factory=deque)
    # (state_key, state_id, action) staged during the arrival-ordered commit loop and
    # sorted before being pushed, so ring contents never depend on worker arrival order.
    frontier_ring_staged: List[Tuple[str, int, int]] = field(default_factory=list)
    frontier_pruned_count: int = 0

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
class _FrontierEncodeGroup:
    """One padded encode bucket of frontier entries.

    ``order_offsets[i]`` is the 0-based candidate position of ``entries[i]``'s
    first action inside the *arrival-ordered* candidate stream. Bucketing sorts
    entries by entity count, so this offset (not the emission position) is what
    restores each candidate's original ``order``.
    """

    entries: Tuple[_FrontierStateActionBatch, ...]
    order_offsets: Tuple[int, ...]


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
    _SUPPORTED_FRONTIER_POLICIES = (
        "full_map_uniform",
        "score_topk_per_map",
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
    )
    # Iteration-scoped fields are archived once on the iteration_summary row instead
    # of being duplicated across every step row.
    _ITERATION_SCOPED_ARCHIVE_METRIC_KEYS = (
        "dynamics_class_transition_counts",
    )
    # Per-collect wall-clock, summed over the iteration's collect calls and archived
    # once on the iteration_summary row. ``setup`` is inclusive of prototype and
    # frontier refresh. The dispatch path is split far enough to distinguish slow
    # actor/IPC work from main-process assessment and commit work:
    #   actor wait + IPC = actor_event_loop - materialize
    #   frontier replay sampling = frontier_refresh - frontier_prepare - frontier_score
    #   worklist packing + spawn = setup  - prototype_refresh - frontier_refresh
    # A collect call that returns early (no active worlds) contributes nothing, so
    # `collect_phase_calls` counts the calls the other sums cover.
    _COLLECT_PHASE_TIMING_KEYS = (
        "collect_phase_total_sec",
        "collect_phase_setup_sec",
        "collect_phase_prototype_refresh_sec",
        "collect_phase_frontier_refresh_sec",
        "collect_phase_frontier_prepare_sec",
        "collect_phase_frontier_score_sec",
        "collect_phase_actor_event_loop_sec",
        "collect_phase_materialize_sec",
        "collect_phase_assessment_sec",
        "collect_phase_commit_sec",
        "collect_phase_finalize_sec",
    )

    def __init__(
        self,
        env: Any,
        seed: int = 42,
        collect_workers: Any = "auto",
        collect_env_factory: Optional[Callable[[], Any]] = None,
        action_order: Optional[Sequence[int]] = None,
        replay_cap: Optional[int] = None,
        frontier_policy: str = "score_topk_per_map",
        frontier_capacity_per_map: int = 4000,
        analysis_archive_enabled: bool = False,
        analysis_archive_local_dir: Optional[str] = None,
        analysis_archive_copy_interval_sec: float = 5.0,
        analysis_archive_compress_level: int = DEFAULT_ANALYSIS_ARCHIVE_COMPRESS_LEVEL,
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
        self.total_worlds_policy_exhausted = 0
        self._frontier_scoring_ready = False
        self._iteration_step_metrics: List[Dict[str, Any]] = []
        self._iteration_summary_metrics: Dict[str, Any] = {}
        self._iteration_collect_phase_calls: int = 0
        self._iteration_collect_phase_totals: Dict[str, float] = {}
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
        self._analysis_archive_compress_level = normalize_compress_level(
            analysis_archive_compress_level
        )
        self._analysis_archive_writer: Optional[AnalysisArchiveWriter] = None
        self._analysis_archive_output_dir: Optional[Path] = None
        self._analysis_node_rows: List[Dict[str, Any]] = []
        self._analysis_edge_rows: List[Dict[str, Any]] = []
        self.replay_cap = self._resolve_optional_refresh_cap(replay_cap, "replay_cap")
        self.frontier_policy = self._resolve_frontier_policy(frontier_policy)
        self.frontier_capacity_per_map = self._resolve_required_refresh_cap(
            frontier_capacity_per_map,
            "frontier_capacity_per_map",
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
            outcome = writer.close()
            # Shutdown gives up on a copy worker that has stopped making
            # progress. Anything still queued at that point never reaches
            # exp_root, so say so: a run that ends short of data must not look
            # like a run that ended complete.
            if isinstance(outcome, dict) and outcome.get("timed_out"):
                print(
                    "Analysis archive WARNING: copy worker stalled during shutdown; "
                    f"{int(outcome.get('abandoned_files', 0))} file(s) / "
                    f"{int(outcome.get('abandoned_bytes', 0))} byte(s) were not copied "
                    "to the experiment root."
                )
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
                compress_level=self._analysis_archive_compress_level,
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

    @staticmethod
    def _resolve_required_refresh_cap(value: int, name: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} must be positive.")
        return int(parsed)

    @classmethod
    def _resolve_frontier_policy(cls, value: Any) -> str:
        policy = str(value).strip()
        if policy in cls._SUPPORTED_FRONTIER_POLICIES:
            return policy
        supported = ", ".join(cls._SUPPORTED_FRONTIER_POLICIES)
        raise ValueError(
            f"frontier_policy must be one of {supported}, got {value!r}."
        )

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
            self._clear_shared_state_token_cache()
        self._state_store = resolved_state_store
        self._bind_env_state_store(resolved_state_store)

    def reset(self) -> None:
        self._close_collect_actor_workers()
        super().reset()
        self.total_restore_steps = 0
        self.total_collect_calls = 0
        self.total_transitions_collected = 0
        self.total_worlds_completed = 0
        self.total_worlds_policy_exhausted = 0
        self._frontier_scoring_ready = False
        self._active_worlds = deque()
        self._round_robin_world_indices = tuple(
            int(spec.world_index)
            for spec in shuffle_round_robin_sessions(self._world_specs, seed=self.seed)
        )
        self._all_world_sessions = {}
        self._world_sessions_initialized = False
        self._last_collect_stats = {}
        self._last_completed_world_summary = {}
        self._iteration_step_metrics = []
        self._iteration_summary_metrics = {}
        self._iteration_collect_phase_calls = 0
        self._iteration_collect_phase_totals = {}
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
        self._clear_shared_state_token_cache()
        self._sample_storage_token_cache.clear()
        self._invalidate_sample_store_tau_cache()
        self._mark_sample_storage_changed()
        self._mark_sample_labels_changed()
        self._maybe_finalize_initial_prototypes()
        self._frontier_scoring_ready = bool(len(self.sample_store) > 0)

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
        self.total_worlds_policy_exhausted = 0
        self._last_collect_stats = {}
        self._last_completed_world_summary = {}
        self._iteration_step_metrics = []
        self._iteration_summary_metrics = {}
        self._iteration_collect_phase_calls = 0
        self._iteration_collect_phase_totals = {}
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
        frontier_restore_mode = self._restore_iteration_boundary_frontiers(
            sessions=sessions,
            frontier_rows=getattr(resume_state, "frontier_rows", ()) or (),
        )

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
        self.total_worlds_policy_exhausted = sum(
            1
            for session in sessions.values()
            if not self._session_is_active(session)
            and self._world_completion_kind(session) == "policy_exhausted"
        )

        summary: Dict[str, Any] = {
            "world_count": int(len(sessions)),
            "active_world_count": int(len(self._active_worlds)),
            "frontier_size": int(self._global_frontier_size()),
            "edge_count": int(self._graph_edge_count()),
            "sample_store_size": int(len(self.sample_store)),
            "total_steps": int(self.total_steps),
        }
        summary["frontier_restore_mode"] = str(frontier_restore_mode)
        summary["frontier_policy"] = str(self.frontier_policy)
        summary["frontier_capacity_per_map"] = int(self.frontier_capacity_per_map)
        summary["frontier_pruned_total"] = int(
            self._global_frontier_pruned_count()
        )
        return summary

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
                group_id=(
                    str(row.get("group_id")).strip()
                    if isinstance(row.get("group_id"), str)
                    and str(row.get("group_id")).strip()
                    else None
                ),
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
        frontier_rows: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        """Restore the canonical frontier state for the configured policy."""
        rows = [row for row in frontier_rows if isinstance(row, Mapping)]
        if not rows:
            raise ValueError("The resume archive has no frontier snapshot rows.")
        entries_by_world: Dict[int, List[Tuple[int, int, int]]] = {}
        policy_by_world: Dict[int, str] = {}
        capacity_by_world: Dict[int, int] = {}
        live_count_by_world: Dict[int, int] = {}
        revision_by_world: Dict[int, int] = {}
        pruned_by_world: Dict[int, int] = {}
        runtime_versions_by_world: Dict[int, Dict[str, int]] = {}
        prototype_rebuild_by_world: Dict[
            int,
            Tuple[bool, Optional[Tuple[int, ...]]],
        ] = {}
        scoring_ready_by_world: Dict[int, bool] = {}
        for row in rows:
            row_type = str(row.get("row_type", "")).strip()
            try:
                world_index = int(row.get("world_index"))
            except (TypeError, ValueError):
                continue
            if row_type == "frontier_world":
                if world_index in policy_by_world:
                    raise ValueError(
                        "Resume frontier snapshot contains duplicate world headers: "
                        f"world_index={world_index}."
                    )
                policy_by_world[world_index] = str(row.get("policy", "")).strip()
                capacity_by_world[world_index] = int(row.get("capacity_per_map"))
                live_count_by_world[world_index] = max(
                    0,
                    int(row.get("live_count", 0) or 0),
                )
                revision_by_world[world_index] = max(
                    0,
                    int(row.get("frontier_revision", 0) or 0),
                )
                pruned_by_world[world_index] = max(
                    0,
                    int(row.get("pruned_count", 0) or 0),
                )
                runtime_versions_by_world[world_index] = (
                    self._resolve_runtime_version_state(
                        row.get("runtime_versions"),
                        owner=(
                            "resume frontier snapshot "
                            f"world_index={world_index}"
                        ),
                    )
                )
                prototype_rebuild_by_world[world_index] = (
                    self._resolve_prototype_rebuild_state(
                        row.get("prototype_rebuild_state"),
                        owner=(
                            "resume frontier snapshot "
                            f"world_index={world_index}"
                        ),
                    )
                )
                scoring_ready = row.get("frontier_scoring_ready")
                if not isinstance(scoring_ready, bool):
                    raise ValueError(
                        "Resume frontier snapshot has an invalid "
                        "frontier_scoring_ready value: "
                        f"world_index={world_index}."
                    )
                scoring_ready_by_world[world_index] = scoring_ready
                continue
            if row_type != "frontier_candidate":
                continue
            try:
                entries_by_world.setdefault(world_index, []).append(
                    (
                        int(row.get("seq")),
                        int(row.get("state_id")),
                        int(row.get("action")),
                    )
                )
            except (TypeError, ValueError):
                continue

        expected_world_indices = {int(world_index) for world_index in sessions}
        if set(policy_by_world) != expected_world_indices:
            raise ValueError(
                "Resume frontier snapshot does not contain exactly one world header "
                f"per target map: expected={sorted(expected_world_indices)} "
                f"actual={sorted(policy_by_world)}."
            )
        mismatched_policies = {
            int(world_index): str(policy)
            for world_index, policy in policy_by_world.items()
            if str(policy) != str(self.frontier_policy)
        }
        if mismatched_policies:
            raise ValueError(
                "Resume frontier policy differs from the current configuration: "
                f"current={self.frontier_policy!r} archive={mismatched_policies}."
            )
        mismatched_capacities = {
            int(world_index): int(capacity)
            for world_index, capacity in capacity_by_world.items()
            if int(capacity) != int(self.frontier_capacity_per_map)
        }
        if mismatched_capacities:
            raise ValueError(
                "Resume frontier capacity differs from the current configuration: "
                f"current={int(self.frontier_capacity_per_map)} "
                f"archive={mismatched_capacities}."
            )
        runtime_version_states = {
            tuple(int(values[name]) for name in self._RUNTIME_VERSION_FIELDS)
            for values in runtime_versions_by_world.values()
        }
        if len(runtime_version_states) != 1:
            raise ValueError(
                "Resume frontier snapshot has inconsistent runtime_versions across maps."
            )
        runtime_version_state = dict(
            zip(self._RUNTIME_VERSION_FIELDS, next(iter(runtime_version_states)))
        )
        prototype_rebuild_states = set(prototype_rebuild_by_world.values())
        if len(prototype_rebuild_states) != 1:
            raise ValueError(
                "Resume frontier snapshot has inconsistent prototype rebuild state "
                "across maps."
            )
        prototype_rebuild_state = next(iter(prototype_rebuild_states))
        scoring_ready_states = set(scoring_ready_by_world.values())
        if len(scoring_ready_states) != 1:
            raise ValueError(
                "Resume frontier snapshot has inconsistent frontier_scoring_ready "
                "across maps."
            )
        frontier_scoring_ready = next(iter(scoring_ready_states))

        if self.frontier_policy == "full_map_uniform":
            if any(entries_by_world.values()):
                raise ValueError(
                    "A full_map_uniform frontier snapshot must not contain candidate rows."
                )
            if any(int(value) > 0 for value in pruned_by_world.values()):
                raise ValueError("full_map_uniform cannot resume with pruned candidates.")
            self._restore_iteration_boundary_frontiers_from_graph(
                sessions=sessions,
                live_count_by_world=live_count_by_world,
                revision_by_world=revision_by_world,
            )
            restore_mode = "graph_recompute"
        else:
            self._restore_iteration_boundary_frontiers_from_candidates(
                sessions=sessions,
                entries_by_world=entries_by_world,
                live_count_by_world=live_count_by_world,
                revision_by_world=revision_by_world,
                pruned_by_world=pruned_by_world,
            )
            restore_mode = "candidate_snapshot"
        self._restore_runtime_version_state(
            runtime_version_state,
            owner="resume frontier snapshot",
        )
        self._restore_prototype_rebuild_state(
            {
                "pending": prototype_rebuild_state[0],
                "class_ids": prototype_rebuild_state[1],
            },
            owner="resume frontier snapshot",
        )
        self._frontier_scoring_ready = bool(frontier_scoring_ready)
        return restore_mode

    def _restore_iteration_boundary_frontiers_from_graph(
        self,
        *,
        sessions: Mapping[int, GraphContrastiveWorldSession],
        live_count_by_world: Mapping[int, int],
        revision_by_world: Mapping[int, int],
    ) -> None:
        for world_index, session in sessions.items():
            for node in sorted(
                session.state_nodes.values(),
                key=lambda item: int(item.state_id),
            ):
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
                if not self._should_use_frontier_scoring():
                    session.warmup_edge_queue.extend(
                        (int(node.state_id), int(action))
                        for action in self._actions_from_mask(pending_mask)
                    )
            expected_live = int(live_count_by_world.get(int(world_index), -1))
            if int(session.frontier_size) != expected_live:
                raise ValueError(
                    "Graph-rebuilt frontier size differs from the archive header: "
                    f"world_index={int(world_index)} expected={expected_live} "
                    f"actual={int(session.frontier_size)}."
                )
            session.frontier_revision = int(revision_by_world[int(world_index)])

    def _restore_iteration_boundary_frontiers_from_candidates(
        self,
        *,
        sessions: Mapping[int, GraphContrastiveWorldSession],
        entries_by_world: Mapping[int, Sequence[Tuple[int, int, int]]],
        live_count_by_world: Mapping[int, int],
        revision_by_world: Mapping[int, int],
        pruned_by_world: Mapping[int, int],
    ) -> None:

        for world_index, session in sessions.items():
            for node in session.state_nodes.values():
                session.expanded_state_ids.add(int(node.state_id))
            session.frontier_pruned_count = int(pruned_by_world[int(world_index)])
            session.frontier_ring = deque()
            session.frontier_ring_staged = []
            pending_by_state: Dict[int, int] = {}
            seen_sequences: set[int] = set()
            seen_identities: set[Tuple[int, int]] = set()
            for seq, state_id, action in sorted(
                entries_by_world.get(int(world_index), ())
            ):
                identity = (int(state_id), int(action))
                if int(seq) in seen_sequences or identity in seen_identities:
                    raise ValueError(
                        "Resume frontier snapshot contains duplicate candidate rows: "
                        f"world_index={int(world_index)} seq={int(seq)} identity={identity}."
                    )
                seen_sequences.add(int(seq))
                seen_identities.add(identity)
                node = session.state_nodes.get(int(state_id))
                if node is None:
                    raise ValueError(
                        "Resume frontier candidate references a missing SearchNode: "
                        f"world_index={int(world_index)} state_id={int(state_id)}."
                    )
                executed_mask = 0
                for executed_action, _edge in session.world_graph.outgoing_edges(
                    int(state_id)
                ):
                    executed_mask |= 1 << int(executed_action)
                allowed_mask = int(self._all_action_mask) & ~int(executed_mask)
                if not (allowed_mask & (1 << int(action))):
                    raise ValueError(
                        "Resume frontier candidate was already executed or has an invalid action: "
                        f"world_index={int(world_index)} identity={identity}."
                    )
                pending_by_state[int(state_id)] = int(
                    pending_by_state.get(int(state_id), 0)
                ) | (1 << int(action))
                session.frontier_ring.append((int(state_id), int(action)))
            for state_id, pending_mask in pending_by_state.items():
                self._set_pending_action_mask(
                    session=session,
                    state_id=int(state_id),
                    mask=int(pending_mask),
                )
            expected_live = int(live_count_by_world[int(world_index)])
            if int(session.frontier_size) != expected_live:
                raise ValueError(
                    "Restored score frontier size differs from the archive header: "
                    f"world_index={int(world_index)} expected={expected_live} "
                    f"actual={int(session.frontier_size)}."
                )
            session.frontier_revision = int(revision_by_world[int(world_index)])

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
        previous_mask = int(
            session.pending_action_masks_by_state.get(resolved_state_id, 0)
        )
        session.pending_action_masks_by_state[resolved_state_id] = resolved_mask
        previous_count = self._action_count_from_mask(previous_mask)
        session.frontier_size += int(action_count) - int(previous_count)
        if self.frontier_policy == "full_map_uniform":
            node = session.state_nodes.get(resolved_state_id)
            if node is None:
                raise RuntimeError(
                    "Uniform frontier cannot index a missing SearchNode: "
                    f"world_index={int(session.world_index)} "
                    f"state_id={resolved_state_id}."
                )
            session.uniform_frontier_index.set_count(
                state_id=resolved_state_id,
                state_key=str(node.key),
                action_count=int(action_count),
            )
        if int(previous_mask) != int(resolved_mask):
            session.frontier_revision += 1

    def _clear_pending_action_mask(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
    ) -> None:
        resolved_state_id = int(state_id)
        previous_mask = int(
            session.pending_action_masks_by_state.pop(resolved_state_id, 0)
        )
        session.frontier_size -= self._action_count_from_mask(previous_mask)
        if self.frontier_policy == "full_map_uniform":
            session.uniform_frontier_index.remove_state(resolved_state_id)
        if int(previous_mask) > 0:
            session.frontier_revision += 1
        if session.frontier_size < 0:
            raise RuntimeError("Frontier size became negative while clearing a mask.")

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
        return int(session.frontier_size)

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
        model: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._score_tau_embeddings_with_model(
            self.dynamics if model is None else model,
            tau_embeddings,
            sample_tau=sample_tau,
            context=context,
        )

    def _frontier_resolved_state_action_entries(
        self,
        state_action_entries: Sequence[_FrontierStateActionBatch],
    ) -> Tuple[List[_FrontierStateActionBatch], List[int], List[int]]:
        """Arrival-ordered entries with their entity counts and candidate offsets.

        The returned lists are positionally aligned: ``entity_sizes[i]`` is the
        state token count of ``entries[i]`` and ``order_offsets[i]`` is the
        0-based position of that entry's first action in the arrival-ordered
        candidate stream. Entries without actions are dropped, exactly as the
        scoring path does.
        """
        resolved_entries: List[_FrontierStateActionBatch] = []
        entity_sizes: List[int] = []
        order_offsets: List[int] = []
        next_offset = 0
        for entry in state_action_entries:
            resolved_actions = tuple(int(action) for action in entry.actions)
            if not resolved_actions:
                continue
            state_tokens, _state_mask = self._cached_state_tokens_by_id(
                int(entry.node.state_id)
            )
            resolved_entries.append(
                _FrontierStateActionBatch(
                    session=entry.session,
                    node=entry.node,
                    actions=resolved_actions,
                )
            )
            entity_sizes.append(max(1, int(state_tokens.size(0))))
            order_offsets.append(int(next_offset))
            next_offset += len(resolved_actions)
        return resolved_entries, entity_sizes, order_offsets

    def _frontier_state_action_token_batches(
        self,
        state_action_entries: Sequence[_FrontierStateActionBatch],
    ) -> Iterable[_FrontierEncodeGroup]:
        """Group frontier entries into padded encode buckets.

        Uses the same sorted-greedy bucketing as the training encode path
        (`_token_budget_state_bucket_rows`), so an arrival-ordered (that is,
        world-clustered and entity-count-shuffled) frontier stream still packs
        into a handful of near-uniform buckets instead of thousands of tiny
        ones. Each group carries the arrival-order offsets needed to restore
        every candidate's original position.
        """
        (
            resolved_entries,
            entity_sizes,
            order_offsets,
        ) = self._frontier_resolved_state_action_entries(state_action_entries)
        if not resolved_entries:
            return
        for row_indices in _token_budget_state_bucket_rows(
            entity_sizes,
            max_padded_tokens=int(self._ENCODE_BUCKET_MAX_PADDED_TOKENS),
            max_padding_ratio=float(self._ENCODE_BUCKET_MAX_PADDING_RATIO),
            max_rows=int(self._frontier_rebuild_state_batch_size()),
        ):
            yield _FrontierEncodeGroup(
                entries=tuple(
                    resolved_entries[int(row_index)] for row_index in row_indices
                ),
                order_offsets=tuple(
                    int(order_offsets[int(row_index)]) for row_index in row_indices
                ),
            )

    def _score_frontier_state_action_batches(
        self,
        *,
        group: _FrontierEncodeGroup,
        base_order: int,
        sample_tau: torch.Tensor,
        context: Optional[ContrastiveEntropyContext] = None,
        # None => the live network. The incremental path passes the frozen snapshot
        # copy so a window's candidates are all ranked by the same function; nothing
        # else may pass anything here.
        model: Optional[Any] = None,
    ) -> List[
        Tuple[GraphContrastiveWorldSession, SearchNode, int, int, float, float, float]
    ]:
        resolved_entries = list(group.entries)
        if not resolved_entries:
            return []

        encoded_states = [
            self._cached_state_tokens_by_id(int(entry.node.state_id))
            for entry in resolved_entries
        ]
        state_tokens_tensor, state_mask_tensor = self._pack_cached_state_batch(encoded_states)
        candidate_state_rows: List[int] = []
        candidate_actions: List[int] = []
        candidate_orders: List[int] = []
        for state_row, entry in enumerate(resolved_entries):
            entry_base_order = int(base_order) + int(group.order_offsets[state_row])
            for action_offset, action in enumerate(entry.actions):
                candidate_state_rows.append(int(state_row))
                candidate_actions.append(int(action))
                candidate_orders.append(int(entry_base_order) + int(action_offset))
        if not candidate_actions:
            return []

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
            Tuple[GraphContrastiveWorldSession, SearchNode, int, int, float, float, float]
        ] = []

        scoring_model = self.dynamics if model is None else model
        with torch.inference_mode():
            with self._autocast_context():
                state_context = scoring_model.encode_state_context(
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
                    tau_embeddings = scoring_model.encode_state_action_from_context(
                        state_context.index_select(0, chunk_rows),
                        chunk_mask,
                        chunk_actions,
                    )
                rh, rz, rtotal = self._score_tau_embeddings(
                    tau_embeddings,
                    sample_tau=sample_tau,
                    # Passed straight through, never re-derived. `None` is a real
                    # value here (prototype entropy off, or no eligible class), and
                    # falling back to `_build_prototype_entropy_context()` on it
                    # would score the incremental pass against the LIVE class counts
                    # while the rest of the window used the frozen ones.
                    context=context,
                    model=scoring_model,
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
                            int(candidate_orders[local_index]),
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
        state_count: int,
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
            # Shards are cut on whole states, never mid-state, so a rank can only
            # be fed if there is a state left for it. Candidate count alone is not
            # enough: 3 states x 5 actions clears `candidate_count >= 8` while
            # leaving 5 of 8 ranks with an empty payload, which is slower than
            # scoring the whole thing on one device.
            and int(state_count) >= world_size
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
        next_order = 0
        # Arrival order, deliberately. Each rank shard is packed as one padded
        # block, so its cost is set by the shard's largest state; as long as the
        # shards hold equal row counts, sorting here only moves the large states
        # together without lowering the slowest rank's padded token count.
        # `_frontier_score_rank_payloads` therefore does its own sort downstream
        # and pairs it with UNEQUAL, padded-cost-equalized shard sizes, which is
        # what makes the sort pay. It reads the `order` values emitted here but
        # never moves them. Do not route this
        # function through `_frontier_state_action_token_batches`: `order` must
        # stay derived from arrival position.
        for entry in state_action_entries:
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
        return candidates

    def _frontier_score_state_runs(
        self,
        candidates: Sequence[Tuple[GraphContrastiveWorldSession, SearchNode, int, int]],
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Cut the arrival-ordered candidate list into contiguous per-state runs.

        `_frontier_score_ordered_candidates` emits one `_FrontierStateActionBatch`
        at a time and all of that entry's actions consecutively, and both
        selection paths group candidates on `(world_index, state_id)` before
        building entries, so a state's candidates are always one contiguous run.
        Cutting on a key change therefore reproduces the entries exactly.

        Returns `(entity_size, start, stop)` per run in arrival order, plus the
        matching cached `(tokens, mask)` pair so the packer does not have to
        resolve the state key a second time.
        """
        runs: List[Tuple[int, int, int]] = []
        encoded_runs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        run_key: Optional[Tuple[int, int]] = None
        for position, (session, node, _action, _order) in enumerate(candidates):
            candidate_key = (int(session.world_index), int(node.state_id))
            if run_key is None or candidate_key != run_key:
                tokens, mask = self._cached_state_tokens_by_id(int(node.state_id))
                runs.append((max(1, int(tokens.size(0))), int(position), int(position) + 1))
                encoded_runs.append((tokens, mask))
                run_key = candidate_key
            else:
                entity_size, start, _stop = runs[-1]
                runs[-1] = (int(entity_size), int(start), int(position) + 1)
        return runs, encoded_runs

    def _frontier_score_rank_payloads(
        self,
        candidates: Sequence[Tuple[GraphContrastiveWorldSession, SearchNode, int, int]],
        *,
        world_size: int,
    ) -> List[Dict[str, Any]]:
        """Split the arrival-ordered candidates into one padded block per rank.

        The shards are deliberately UNEQUAL. Each shard is encoded as a single
        `[rows, max_entities, token_dim]` tensor, so a shard's cost is
        `rows * max_entities` and the refresh's wall time is the max over
        shards. Slicing the arrival order into equal candidate counts hands the
        rank that owns the largest world just as many rows as the rank that owns
        the smallest one; sorting by entity count and *still* cutting equal rows
        does not help either. Sorting and then cutting on equalized padded cost
        lowers both the max-over-ranks and the total padded token count.

        Row order is restored by IDENTITY, not by index arithmetic: every
        candidate carries its own `world_index / state_id / state_key / action /
        order` in the payload, `_ddp_score_frontier_payload` copies those columns
        onto each returned item, and `_merge_distributed_frontier_score_items`
        keys purely on them. There is no positional channel to scatter back, so
        reordering shards - and giving them different lengths - cannot land a
        score on the wrong candidate. `candidate_state_rows` is local to its own
        shard and is built together with that shard's rows.

        Unequal shard lengths are safe on the consuming side too: the frontier
        forward runs on `ddp_model.module` under `torch.inference_mode()`, so it
        reaches no collective and no gradient bucket. Results come back as plain
        dicts over the result queue and only the payload *count* is checked.

        KNOWN GAP: the split is equalized on `states x max_entities`, but the
        dominant loop in `_ddp_score_frontier_payload` is per candidate, i.e.
        `candidates x max_entities`. The two agree only while actions-per-state
        is uncorrelated with entity size, which holds across many worlds. With
        only a handful of worlds the correlation is not diluted and the candidate
        term can regress slightly. Equalizing on candidate
        count instead would fix that, but boundaries must stay on state runs or
        the same state gets encoded on two ranks, so it is not a free swap.
        """
        resolved_world_size = int(world_size)
        runs, encoded_runs = self._frontier_score_state_runs(candidates)
        # Total order on content (entity size) with a positional tiebreak (the
        # run's arrival index), so the split never depends on dict/set iteration
        # or on which worker answered first.
        sorted_runs = sorted(
            range(len(runs)),
            key=lambda run_index: (int(runs[run_index][0]), int(run_index)),
        )
        block_bounds = _padded_cost_equalized_block_bounds(
            [int(runs[run_index][0]) for run_index in sorted_runs],
            block_count=resolved_world_size,
        )

        payloads: List[Dict[str, Any]] = []
        for rank in range(resolved_world_size):
            block = sorted_runs[int(block_bounds[rank]):int(block_bounds[rank + 1])]
            if not block:
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
            # Whole runs, so a state can no longer straddle a shard boundary and
            # be encoded twice. The dedup stays for the case where two worlds
            # resolve to the same `state_id`: the tokens are looked up by
            # `state_id` alone, so sharing one row is exact.
            for run_index in block:
                _entity_size, run_start, run_stop = runs[int(run_index)]
                for position in range(int(run_start), int(run_stop)):
                    session, node, action, order = candidates[int(position)]
                    state_id = int(node.state_id)
                    local_row = local_state_rows.get(state_id)
                    if local_row is None:
                        local_row = len(encoded_states)
                        local_state_rows[state_id] = int(local_row)
                        encoded_states.append(encoded_runs[int(run_index)])
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
        retention_budgets: Mapping[int, int],
        scored_items: Sequence[Mapping[str, Any]],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        retained_heaps: Dict[
            int,
            List[Tuple[Tuple[float, float, float, int], int, FrontierCandidate]],
        ] = {
            int(session.world_index): []
            for session in sessions
        }

        for item in scored_items:
            world_index = int(item["world_index"])
            capacity = max(0, int(retention_budgets.get(world_index, 0)))
            if capacity <= 0:
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
            heap = retained_heaps.setdefault(world_index, [])
            heap_entry = (priority, int(order), work_item)
            if len(heap) < capacity:
                heapq.heappush(heap, heap_entry)
            elif priority > heap[0][0]:
                heapq.heapreplace(heap, heap_entry)

        return {
            int(session.world_index): tuple(
                heap_entry[2]
                for heap_entry in sorted(
                    retained_heaps.get(int(session.world_index), ()),
                    key=lambda heap_entry: heap_entry[0],
                    reverse=True,
                )
            )
            for session in sessions
        }

    def _score_frontier_entries_distributed(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        retention_budgets: Mapping[int, int],
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
                budgets={
                    int(session.world_index): max(
                        0,
                        int(retention_budgets.get(int(session.world_index), 0)),
                    )
                    for session in sessions
                },
            )
        except BaseException:
            self._close_distributed_trainer()
            raise
        raw_items = result.get("items") or []
        scored_items = [item for item in raw_items if isinstance(item, Mapping)]
        retained = self._merge_distributed_frontier_score_items(
            sessions=sessions,
            retention_budgets=retention_budgets,
            scored_items=scored_items,
        )
        return retained, int(result.get("scored_count", len(candidates)))

    def _should_use_frontier_scoring(self) -> bool:
        return bool(
            len(self.sample_store) > 0
            and self._frontier_scoring_ready
        )

    def _train_schedule_name(self) -> str:
        return "post_verify_batch"

    def _select_frontier_refresh_state_action_batches(
        self,
        sessions: Sequence[GraphContrastiveWorldSession],
    ) -> _FrontierRefreshSelection:
        if self.frontier_policy == "full_map_uniform":
            return self._select_uniform_frontier_refresh_batches(sessions)
        return self._select_score_topk_frontier_refresh_batches(sessions)

    def _frontier_uniform_selection_signature(
        self,
        session: GraphContrastiveWorldSession,
    ) -> Tuple[Any, ...]:
        return (
            int(self.total_steps),
            int(session.frontier_revision),
            int(self._sample_storage_version),
            int(self._sample_label_version),
            int(self._dynamics_parameter_version),
            int(self._prototype_summary_version),
            tuple(self._active_class_indices()),
            int(self.dynamics.num_classes),
            float(self.intrinsic_reward_scale),
            float(self.prototype_entropy_scale),
        )

    def _select_uniform_frontier_refresh_batches(
        self,
        sessions: Sequence[GraphContrastiveWorldSession],
    ) -> _FrontierRefreshSelection:
        """Uniformly sample at most K live candidates from each map without pruning."""
        entries: List[_FrontierStateActionBatch] = []
        total_candidates = 0
        selected_candidates = 0
        for session in sessions:
            live_count = int(session.frontier_size)
            index = session.uniform_frontier_index
            if int(index.total_count) != live_count:
                raise RuntimeError(
                    "Uniform frontier index and pending masks diverged: "
                    f"world_index={int(session.world_index)} "
                    f"index={int(index.total_count)} pending={live_count}."
                )
            total_candidates += live_count
            positions = self._frontier_refresh_sample_positions(
                total_count=live_count,
                cap=int(self.frontier_capacity_per_map),
                tag=f"frontier_map:{int(session.world_index)}",
                signature_parts=self._frontier_uniform_selection_signature(session),
            )
            grouped: Dict[int, List[int]] = {}
            for position in positions:
                state_id, action = index.resolve(
                    int(position),
                    pending_masks_by_state=session.pending_action_masks_by_state,
                    action_order=self.action_order,
                )
                grouped.setdefault(int(state_id), []).append(int(action))
            for state_id, actions in grouped.items():
                node = session.state_nodes.get(int(state_id))
                if node is None:
                    raise RuntimeError(
                        "Uniform frontier index referenced a missing SearchNode: "
                        f"world_index={int(session.world_index)} state_id={int(state_id)}"
                    )
                entries.append(
                    _FrontierStateActionBatch(
                        session=session,
                        node=node,
                        actions=tuple(actions),
                    )
                )
            selected_candidates += len(positions)
        return _FrontierRefreshSelection(
            state_action_entries=tuple(entries),
            total_candidates=int(total_candidates),
            selected_candidates=int(selected_candidates),
        )

    def _select_score_topk_frontier_refresh_batches(
        self,
        sessions: Sequence[GraphContrastiveWorldSession],
    ) -> _FrontierRefreshSelection:
        """Materialize every live score-policy candidate, including new overflow."""
        entries: List[_FrontierStateActionBatch] = []
        total_candidates = 0
        selected_candidates = 0
        for session in sessions:
            total_candidates += int(session.frontier_size)
            pending_masks = session.pending_action_masks_by_state
            grouped: Dict[int, List[int]] = {}
            for state_id, action in session.frontier_ring:
                if not (
                    int(pending_masks.get(int(state_id), 0)) & (1 << int(action))
                ):
                    continue
                grouped.setdefault(int(state_id), []).append(int(action))
            for state_id, actions in grouped.items():
                node = session.state_nodes.get(int(state_id))
                if node is None:
                    raise RuntimeError(
                        "Frontier ring referenced a missing SearchNode: "
                        f"world_index={int(session.world_index)} state_id={int(state_id)}"
                    )
                entries.append(
                    _FrontierStateActionBatch(
                        session=session,
                        node=node,
                        actions=tuple(actions),
                    )
                )
                selected_candidates += len(actions)
        if int(selected_candidates) != int(total_candidates):
            raise RuntimeError(
                "Score frontier ring and pending masks diverged: "
                f"ring={int(selected_candidates)} pending={int(total_candidates)}."
            )
        return _FrontierRefreshSelection(
            state_action_entries=tuple(entries),
            total_candidates=int(total_candidates),
            selected_candidates=int(selected_candidates),
        )

    def _emit_frontier_refresh_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        *,
        phase: str,
        replay_total: int,
        replay_sampled: int,
        frontier_total: int,
        frontier_selected: int,
        scored: Optional[int] = None,
        elapsed_sec: Optional[float] = None,
        prepare_elapsed_sec: Optional[float] = None,
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
            "frontier_selected_candidates": max(0, int(frontier_selected)),
        }
        if scored is not None:
            payload["frontier_scored_candidates"] = max(0, int(scored))
        if prepare_elapsed_sec is not None:
            payload["frontier_prepare_elapsed_sec"] = max(
                0.0,
                float(prepare_elapsed_sec),
            )
        if score_elapsed_sec is not None:
            payload["frontier_score_elapsed_sec"] = max(
                0.0,
                float(score_elapsed_sec),
            )
        if elapsed_sec is not None:
            payload["frontier_refresh_elapsed_sec"] = max(0.0, float(elapsed_sec))
        progress_callback(payload)

    def _score_frontier_entries(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        retention_budgets: Mapping[int, int],
        state_action_entries: Sequence[_FrontierStateActionBatch],
        sample_tau: torch.Tensor,
        context: Optional[ContrastiveEntropyContext],
    ) -> Tuple[Dict[int, Tuple[FrontierCandidate, ...]], int]:
        retained_heaps: Dict[
            int,
            List[Tuple[Tuple[float, float, float, int], int, FrontierCandidate]],
        ] = {
            int(session.world_index): []
            for session in sessions
        }
        scored_count = 0

        def maybe_retain_candidate(
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
            capacity = max(0, int(retention_budgets.get(world_index, 0)))
            if capacity <= 0:
                return
            priority = (
                float(score),
                float(rz_pred),
                float(rh),
                -int(order),
            )
            heap = retained_heaps.setdefault(world_index, [])
            if len(heap) >= capacity and priority <= heap[0][0]:
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
            if len(heap) < capacity:
                heapq.heappush(heap, heap_entry)
            else:
                heapq.heapreplace(heap, heap_entry)

        # `order` is derived from each candidate's position in the arrival stream,
        # not from the order buckets happen to be emitted in. Bucketing sorts by
        # entity count, so deriving it from emission would renumber every
        # candidate and change the `-order` tiebreak in `priority`.
        base_order = 0
        for group in self._frontier_state_action_token_batches(state_action_entries):
            scored_entries = self._score_frontier_state_action_batches(
                group=group,
                base_order=base_order,
                sample_tau=sample_tau,
                context=context,
            )
            expected_count = sum(len(entry.actions) for entry in group.entries)
            if len(scored_entries) != expected_count:
                raise RuntimeError(
                    "Frontier scoring returned a different number of candidates than "
                    f"the encode bucket contained: {len(scored_entries)} != {expected_count}."
                )
            scored_count += len(scored_entries)
            for session, node, action, order, rh, rz_pred, score in scored_entries:
                maybe_retain_candidate(
                    session=session,
                    node=node,
                    action=int(action),
                    rh=float(rh),
                    rz_pred=float(rz_pred),
                    score=float(score),
                    order=int(order),
                )
        retained = {
            int(session.world_index): tuple(
                heap_entry[2]
                for heap_entry in sorted(
                    retained_heaps.get(int(session.world_index), ()),
                    key=lambda heap_entry: heap_entry[0],
                    reverse=True,
                )
            )
            for session in sessions
        }
        return retained, int(scored_count)

    def _rank_zero_score_frontier_entries(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        retention_budgets: Mapping[int, int],
        state_action_entries: Sequence[_FrontierStateActionBatch],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        retained_heaps: Dict[
            int,
            List[Tuple[Tuple[float, float, float, int], int, FrontierCandidate]],
        ] = {
            int(session.world_index): []
            for session in sessions
        }
        order = 0
        for entry in state_action_entries:
            capacity = max(
                0,
                int(retention_budgets.get(int(entry.session.world_index), 0)),
            )
            for action in entry.actions:
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
                heap = retained_heaps.setdefault(int(entry.session.world_index), [])
                heap_entry = (priority, int(order), work_item)
                if capacity <= 0:
                    order += 1
                    continue
                if len(heap) < capacity:
                    heapq.heappush(heap, heap_entry)
                elif priority > heap[0][0]:
                    heapq.heapreplace(heap, heap_entry)
                order += 1
        retained = {
            int(session.world_index): tuple(
                heap_entry[2]
                for heap_entry in sorted(
                    retained_heaps.get(int(session.world_index), ()),
                    key=lambda heap_entry: heap_entry[0],
                    reverse=True,
                )
            )
            for session in sessions
        }
        return retained

    def _prune_frontiers_to_ranked_candidates(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        ranked_by_world: Mapping[int, Sequence[FrontierCandidate]],
    ) -> None:
        """Keep each map's top candidates while preserving their ring order."""
        if self.frontier_policy != "score_topk_per_map":
            raise RuntimeError("Only score_topk_per_map may prune frontier candidates.")
        capacity = int(self.frontier_capacity_per_map)
        for session in sessions:
            world_index = int(session.world_index)
            live_before = int(session.frontier_size)
            expected_kept = min(capacity, live_before)
            ranked = tuple(ranked_by_world.get(world_index, ()))
            if len(ranked) != expected_kept:
                raise RuntimeError(
                    "Frontier scoring did not return the required per-map top-K: "
                    f"world_index={world_index} expected={expected_kept} "
                    f"actual={len(ranked)}."
                )
            kept_identities = {
                (int(candidate.state_id), int(candidate.action))
                for candidate in ranked
            }
            if len(kept_identities) != len(ranked):
                raise RuntimeError(
                    "Frontier scoring returned duplicate candidate identities for "
                    f"world_index={world_index}."
                )

            pending_masks = session.pending_action_masks_by_state
            compacted: Deque[Tuple[int, int]] = deque()
            pruned_count = 0
            for state_id, action in session.frontier_ring:
                identity = (int(state_id), int(action))
                if not (
                    int(pending_masks.get(identity[0], 0)) & (1 << identity[1])
                ):
                    continue
                if identity in kept_identities:
                    compacted.append(identity)
                    continue
                self._retire_pending_state_action(
                    session=session,
                    state_id=identity[0],
                    action=identity[1],
                )
                pruned_count += 1
            session.frontier_ring = compacted
            session.frontier_pruned_count += int(pruned_count)
            if int(session.frontier_size) != expected_kept or len(compacted) != expected_kept:
                raise RuntimeError(
                    "Frontier ring and pending masks diverged after top-K pruning: "
                    f"world_index={world_index} expected={expected_kept} "
                    f"frontier_size={int(session.frontier_size)} ring_size={len(compacted)}."
                )

    def _refresh_frontiers_for_collect(
        self,
        *,
        sessions: Sequence[GraphContrastiveWorldSession],
        budgets: Mapping[int, int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Dict[int, Tuple[FrontierCandidate, ...]]:
        refresh_started_at = perf_counter()
        uses_frontier_scoring = self._should_use_frontier_scoring()
        retention_budgets = (
            {
                int(session.world_index): int(self.frontier_capacity_per_map)
                for session in sessions
            }
            if self.frontier_policy == "score_topk_per_map"
            else {
                int(session.world_index): max(
                    0,
                    int(budgets.get(int(session.world_index), 0)),
                )
                for session in sessions
            }
        )
        uses_intrinsic_reward = bool(
            uses_frontier_scoring and self._frontier_uses_intrinsic_reward()
        )
        replay_candidate_indices = (
            self._frontier_refresh_replay_candidate_indices()
            if uses_intrinsic_reward and self.prototype_sample_cap is not None
            else ()
        )
        replay_total = (
            (
                len(replay_candidate_indices)
                if self.prototype_sample_cap is not None
                else int(len(self.sample_store))
            )
            if uses_intrinsic_reward
            else 0
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
        prepare_started_at = perf_counter()
        selection = self._select_frontier_refresh_state_action_batches(sessions)
        prepare_elapsed_sec = perf_counter() - prepare_started_at
        score_started_at = perf_counter()
        self._emit_frontier_refresh_progress(
            progress_callback,
            phase="start",
            replay_total=replay_total,
            replay_sampled=len(replay_indices),
            frontier_total=selection.total_candidates,
            frontier_selected=selection.selected_candidates,
        )
        scored_count = 0
        retained: Dict[int, Tuple[FrontierCandidate, ...]] = {
            int(session.world_index): ()
            for session in sessions
        }
        if selection.state_action_entries:
            context = (
                self._build_prototype_entropy_context()
                if uses_frontier_scoring
                else None
            )
            uses_prototype_entropy_reward = bool(
                uses_frontier_scoring
                and self._frontier_uses_prototype_entropy_reward(context)
            )
            if not uses_frontier_scoring or (
                not uses_intrinsic_reward and not uses_prototype_entropy_reward
            ):
                retained = self._rank_zero_score_frontier_entries(
                    sessions=sessions,
                    retention_budgets=retention_budgets,
                    state_action_entries=selection.state_action_entries,
                )
            elif self._can_use_distributed_frontier_scoring(
                replay_count=len(replay_indices),
                candidate_count=selection.selected_candidates,
                state_count=len(selection.state_action_entries),
            ):
                retained, scored_count = self._score_frontier_entries_distributed(
                    sessions=sessions,
                    retention_budgets=retention_budgets,
                    state_action_entries=selection.state_action_entries,
                    replay_indices=replay_indices,
                    replay_candidate_indices=replay_candidate_indices,
                    context=context,
                )
            else:
                refresh_sample_tau = (
                    self._encode_frontier_refresh_replay_tau(
                        replay_indices,
                        replay_candidate_indices=replay_candidate_indices,
                    )
                    if uses_intrinsic_reward
                    else self._empty_frontier_sample_tau()
                )
                retained, scored_count = self._score_frontier_entries(
                    sessions=sessions,
                    retention_budgets=retention_budgets,
                    state_action_entries=selection.state_action_entries,
                    sample_tau=refresh_sample_tau,
                    context=context,
                )
        if self.frontier_policy == "score_topk_per_map":
            self._prune_frontiers_to_ranked_candidates(
                sessions=sessions,
                ranked_by_world=retained,
            )
        worklists = {
            int(session.world_index): tuple(
                retained.get(int(session.world_index), ())[
                    : max(0, int(budgets.get(int(session.world_index), 0)))
                ]
            )
            for session in sessions
        }
        refresh_ended_at = perf_counter()
        score_elapsed_sec = refresh_ended_at - score_started_at
        refresh_elapsed_sec = refresh_ended_at - refresh_started_at
        self._accumulate_collect_phase_timings(
            frontier_refresh_sec=refresh_elapsed_sec,
            frontier_prepare_sec=prepare_elapsed_sec,
            frontier_score_sec=score_elapsed_sec,
        )
        self._emit_frontier_refresh_progress(
            progress_callback,
            phase="end",
            replay_total=replay_total,
            replay_sampled=len(replay_indices),
            frontier_total=selection.total_candidates,
            frontier_selected=selection.selected_candidates,
            scored=scored_count,
            elapsed_sec=refresh_elapsed_sec,
            prepare_elapsed_sec=prepare_elapsed_sec,
            score_elapsed_sec=score_elapsed_sec,
        )
        return worklists

    def _build_uniform_warmup_worklists_for_collect(
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
                if not (
                    int(session.pending_action_masks_by_state.get(int(state_id), 0))
                    & (1 << int(action))
                ):
                    continue
                node = session.state_nodes.get(int(state_id))
                if node is None:
                    raise RuntimeError(
                        "Warmup frontier referenced a missing SearchNode: "
                        f"world_index={world_index} state_id={int(state_id)}"
                    )
                candidates.append(
                    FrontierCandidate(
                        state_id=int(state_id),
                        state_key=str(node.key),
                        action=int(action),
                        rh=0.0,
                        rz_pred=0.0,
                        score=0.0,
                        order=len(candidates),
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
        # Catch-all for candidates staged outside the collect commit loop (root seeds
        # from _create_world_session, resume). Idempotent; no-op if nothing is staged.
        self._flush_frontier_ring_pushes(sessions)
        if self.frontier_policy == "full_map_uniform":
            if not self._should_use_frontier_scoring():
                return self._build_uniform_warmup_worklists_for_collect(
                    sessions=sessions,
                    budgets=budgets,
                )
            for session in sessions:
                session.warmup_edge_queue.clear()
        return self._refresh_frontiers_for_collect(
            sessions=sessions,
            budgets=budgets,
            progress_callback=progress_callback,
        )

    def _prepare_state_frontier(
        self,
        *,
        session: GraphContrastiveWorldSession,
        node: SearchNode,
    ) -> None:
        if int(node.state_id) in session.expanded_state_ids:
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
        if self.frontier_policy == "score_topk_per_map":
            # Sort staged entries before pushing so actor arrival order cannot change
            # the FIFO tiebreak used by the score policy.
            session.frontier_ring_staged.extend(
                (str(node.key), int(node.state_id), int(action))
                for action in self.action_order
            )
        elif not self._should_use_frontier_scoring():
            session.warmup_edge_queue.extend(
                (int(node.state_id), int(action))
                for action in self.action_order
            )

    def _flush_frontier_ring_pushes(
        self,
        sessions: Iterable[GraphContrastiveWorldSession],
    ) -> None:
        """Append newly discovered candidates without applying the capacity yet.

        Ordering: staged tuples are `(state_key, state_id, action)` and are sorted as
        such. `state_key` is a sha1 over the packed state content, so the resulting
        push order is a function of WHAT was discovered, never of which worker
        reported it first. (Sorting by `state_id` would NOT achieve that:
        `StateStore.intern_runtime_state_packet` assigns ids from a single monotone
        counter inside the arrival-ordered commit loop, so state ids are themselves
        arrival-order artefacts. See the report.)

        The ring may exceed ``frontier_capacity_per_map`` between collect calls.
        The next refresh scores every live entry and applies the cap by score.
        """
        if self.frontier_policy != "score_topk_per_map":
            return
        for session in sessions:
            staged = session.frontier_ring_staged
            if not staged:
                continue
            session.frontier_ring_staged = []
            staged.sort()
            for _state_key, state_id, action in staged:
                session.frontier_ring.append((int(state_id), int(action)))

    @staticmethod
    def _world_completion_kind(session: GraphContrastiveWorldSession) -> str:
        """Why did this world's frontier hit zero?

        `fully_explored` means every reachable (state, action) was executed.
        `policy_exhausted` means score-topK pruning removed candidates, so the map is
        not known to be exhaustively explored.
        """
        if int(session.frontier_pruned_count) > 0:
            return "policy_exhausted"
        return "fully_explored"

    def _global_frontier_pruned_count(self) -> int:
        return int(
            sum(
                int(session.frontier_pruned_count)
                for session in self._all_world_sessions.values()
            )
        )

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

    def _retire_pending_state_action(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        action: int,
    ) -> None:
        """Clear one pending (state, action) bit; drop the state when nothing is left.

        Two callers: execution (`_mark_executed_state_action`) and score-topK pruning.
        Both mean "this candidate will never be expanded", which is exactly what a
        cleared pending bit encodes, so ``frontier_size`` remains the number of
        candidates the agent still intends to visit.
        """
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
            return
        self._clear_pending_action_mask(
            session=session,
            state_id=resolved_state_id,
        )

    def _mark_executed_state_action(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        action: int,
    ) -> None:
        self._retire_pending_state_action(
            session=session,
            state_id=int(state_id),
            action=int(action),
        )

    def _record_known_edge(
        self,
        *,
        session: GraphContrastiveWorldSession,
        state_id: int,
        action: int,
        next_state_id: int,
        done: bool,
        class_id: Optional[int] = None,
        group_id: Optional[str] = None,
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
            group_id=group_id,
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
                if int(updated_child.state_id) not in session.expanded_state_ids:
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
    ) -> SearchNode:
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
        if int(updated.state_id) not in session.expanded_state_ids:
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
    ) -> List[Transition]:
        transitions: List[Transition] = []
        # Stays 0.0 on the early-return paths below, which bail out before the rebuild.
        prototype_refresh_elapsed_sec = 0.0
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
        if not active_sessions:
            worker_count = self._effective_collect_workers(active_map_count=0)
            collect_setup_elapsed_sec = max(0.0, perf_counter() - collect_prep_started_at)
            collect_prep_elapsed_sec = float(collect_setup_elapsed_sec)
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
                **self._frontier_collect_stats(),
            }
            self._last_collect_stats["frontier_exhausted"] = True
            self._last_collect_stats["frontier_exhausted_kind"] = (
                "policy_exhausted"
                if int(self._global_frontier_pruned_count()) > 0
                else "fully_explored"
            )
            return transitions

        budgets = self._actor_collect_budgets(
            max_transitions=transition_budget,
            sessions=active_sessions,
        )
        # The prototype memory pool (_ensure_prototypes_current) and the frontier
        # candidates (_refresh_frontiers_for_collect) must be encoded with the
        # same dynamics parameters, so the rebuild is issued here, immediately before
        # the scoring pass and after nothing else can advance the optimizer.
        prototype_refresh_started_at = perf_counter()
        self._ensure_prototypes_current()
        prototype_refresh_elapsed_sec = perf_counter() - prototype_refresh_started_at
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
            collect_prep_elapsed_sec = float(collect_setup_elapsed_sec)
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
        collect_prep_elapsed_sec = float(collect_setup_elapsed_sec)
        actor_event_loop_started_at = perf_counter()
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
        materialize_elapsed_sec = 0.0

        def _record_transition_batch_event(event: ActorTransitionBatchEvent) -> None:
            nonlocal materialize_elapsed_sec, world_visits
            materialize_started_at = perf_counter()
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
            materialize_elapsed_sec += perf_counter() - materialize_started_at

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
        actor_event_loop_elapsed_sec = perf_counter() - actor_event_loop_started_at
        assessment_started_at = perf_counter()
        assessments = self._evaluate_collect_transition_assessments(
            pending_prepared_transitions
        )
        assessment_elapsed_sec = perf_counter() - assessment_started_at
        commit_started_at = perf_counter()
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
        commit_elapsed_sec = perf_counter() - commit_started_at

        # Push this cycle's discoveries into the rings BEFORE the completion sweep, so
        # a world is only ever declared finished against a settled frontier.
        finalize_started_at = perf_counter()
        self._flush_frontier_ring_pushes(active_sessions)

        worlds_policy_exhausted = 0
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
                summary["frontier_pruned"] = int(session.frontier_pruned_count)
            # Every session in active_sessions was active when this collect
            # started, so an empty frontier now means it just became exhausted.
            # (Testing _session_is_active here as well was always false, since
            # that helper is itself defined as frontier_size > 0.)
            if self._session_frontier_size(session) <= 0:
                completion_kind = self._world_completion_kind(session)
                self._remove_active_world(session)
                worlds_completed += 1
                self.total_worlds_completed += 1
                if completion_kind == "policy_exhausted":
                    worlds_policy_exhausted += 1
                    self.total_worlds_policy_exhausted += 1
                if isinstance(summary, dict):
                    summary["completion_kind"] = str(completion_kind)
                    self._last_completed_world_summary = dict(summary)

        next_active_worlds: Deque[GraphContrastiveWorldSession] = deque()
        for session in self._active_worlds:
            if self._session_is_active(session):
                next_active_worlds.append(session)
        self._active_worlds = next_active_worlds
        finalize_elapsed_sec = perf_counter() - finalize_started_at
        stopped_by_max_transitions = len(transitions) >= int(transition_budget)
        base_map_budget = self._fair_collect_base_map_budget(
            max_transitions=transition_budget,
            map_count=len(active_sessions),
        )
        remainder_map_count = int(transition_budget) % max(1, int(len(active_sessions)))
        # One timestamp per collect call (~1 per 4096 transitions), taken once the
        # actors have returned. It is what makes the dispatch phase attributable:
        # everything between the end of setup and here is actor wall time, which no
        # other counter covers.
        self._iteration_collect_phase_calls += 1
        self._accumulate_collect_phase_timings(
            total_sec=perf_counter() - collect_prep_started_at,
            setup_sec=collect_setup_elapsed_sec,
            prototype_refresh_sec=prototype_refresh_elapsed_sec,
            actor_event_loop_sec=actor_event_loop_elapsed_sec,
            materialize_sec=materialize_elapsed_sec,
            assessment_sec=assessment_elapsed_sec,
            commit_sec=commit_elapsed_sec,
            finalize_sec=finalize_elapsed_sec,
        )
        self._last_collect_stats = {
            "worlds_started": int(worlds_started),
            "worlds_processed": int(len(world_summaries)),
            "world_visits": int(world_visits),
            "worlds_completed": int(worlds_completed),
            "transitions_collected": int(len(transitions)),
            "stopped_by_max_transitions": bool(stopped_by_max_transitions),
            "worlds_completed_total": int(self.total_worlds_completed),
            "resume_pending": bool(len(self._active_worlds) > 0),
            "worlds_policy_exhausted": int(worlds_policy_exhausted),
            "worlds_policy_exhausted_total": int(
                self.total_worlds_policy_exhausted
            ),
            **self._frontier_collect_stats(),
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
            "collect_actor_event_loop_elapsed_sec": float(
                actor_event_loop_elapsed_sec
            ),
            "collect_materialize_elapsed_sec": float(materialize_elapsed_sec),
            "collect_assessment_elapsed_sec": float(assessment_elapsed_sec),
            "collect_commit_elapsed_sec": float(commit_elapsed_sec),
            "collect_finalize_elapsed_sec": float(finalize_elapsed_sec),
            "collect_prep_active_maps": int(len(active_sessions)),
            "collect_prep_dispatch_maps": int(len(dispatch_sessions)),
            "collect_prep_workers": int(worker_count),
            "collect_prep_budget": int(transition_budget),
            "world_summaries": list(world_summaries.values()),
        }
        return transitions

    def _frontier_collect_stats(self) -> Dict[str, Any]:
        """Current capacity-policy stats for ``_last_collect_stats``."""
        return {
            "frontier_policy": str(self.frontier_policy),
            "frontier_capacity_per_map": int(self.frontier_capacity_per_map),
            "frontier_pruned_total": int(self._global_frontier_pruned_count()),
            "frontier_live_total": int(self._global_frontier_size()),
        }

    def _accumulate_collect_phase_timings(self, **elapsed_by_phase: float) -> None:
        """Add one collect call's phase timings to the running iteration totals.

        Every value here is read off a `perf_counter` the collect path already took
        for its own log line; nothing new is timed, and nothing is timed per
        transition or per candidate. The totals are floats in a dict of fixed size,
        so an iteration with 40 collect calls costs the same as one with 1.
        """
        totals = self._iteration_collect_phase_totals
        for phase, elapsed_sec in elapsed_by_phase.items():
            key = f"collect_phase_{phase}"
            totals[key] = float(totals.get(key, 0.0)) + max(0.0, float(elapsed_sec))

    def _iteration_collect_phase_summary(self) -> Dict[str, Any]:
        """The iteration's collect timing totals, for the archived summary row.

        Empty when no collect call reported one, so an iteration that only trained
        writes exactly the row it wrote before these fields existed.
        """
        if not self._iteration_collect_phase_totals:
            return {}
        summary: Dict[str, Any] = {
            "collect_phase_calls": int(self._iteration_collect_phase_calls),
        }
        for key in self._COLLECT_PHASE_TIMING_KEYS:
            summary[key] = float(self._iteration_collect_phase_totals.get(key, 0.0))
        return summary

    def _frontier_diagnostics(self) -> Dict[str, Any]:
        """Frontier capacity-policy diagnostics."""
        return {
            "frontier_policy": str(self.frontier_policy),
            "frontier_capacity_per_map": int(self.frontier_capacity_per_map),
            "frontier_pruned_total": int(self._global_frontier_pruned_count()),
            "total_worlds_policy_exhausted": int(self.total_worlds_policy_exhausted),
            "frontier_exhausted": bool(
                self._last_collect_stats.get("frontier_exhausted") is True
            ),
        }

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
        # The prototype rebuild is issued inside _collect_fair_map_actor_slices, so it
        # lands immediately before the frontier scoring pass it must share parameters
        # with.
        return self._collect_fair_map_actor_slices(
            max_transitions=int(max_transitions),
            progress_callback=progress_callback,
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
            "group_id": sample.leaf_group_id,
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
            group_id=(
                str(commit.transition_result.get("group_id")).strip()
                if isinstance(commit.transition_result.get("group_id"), str)
                and str(commit.transition_result.get("group_id")).strip()
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
    ) -> None:
        live_map_frontier_size = int(self._session_frontier_size(session))
        total_frontier_size = int(self._global_frontier_size())

        self._record_transition_metrics(
            episode_index=int(session.world_index),
            step_index=int(commit.display_step),
            action_name=str(commit.action_name),
            dynamics_class_index=(
                int(commit.transition_result.get("display_class_id"))
                if isinstance(commit.transition_result.get("display_class_id"), int)
                else None
            ),
            state_action_duplicate_count=None,
            rh=float(commit.transition_rh),
            rz=float(commit.transition_rz),
            rtotal=float(commit.transition_intrinsic_reward),
            done=bool(done),
            frontier_size=int(total_frontier_size),
            live_map_frontier_size=int(live_map_frontier_size),
            world_index=int(session.world_index),
            map_name=commit.map_name,
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

    def _record_transition_metrics(
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
                "group_id": edge.group_id,
                "rh": float(edge.rh),
                "rz": float(edge.rz),
                "rtotal": float(edge.rtotal),
                "trainable": bool(edge.trainable),
            }
        )

    def _build_analysis_frontier_rows(self) -> List[Dict[str, Any]]:
        """Write compact policy headers and exact score-topK candidate rows."""
        if not self._analysis_archive_enabled:
            return []
        self._flush_frontier_ring_pushes(self._all_world_sessions.values())
        rows: List[Dict[str, Any]] = []
        for world_index in sorted(self._all_world_sessions):
            session = self._all_world_sessions[int(world_index)]
            rows.append(
                {
                    "row_type": "frontier_world",
                    "world_index": int(world_index),
                    "policy": str(self.frontier_policy),
                    "capacity_per_map": int(self.frontier_capacity_per_map),
                    "live_count": int(session.frontier_size),
                    "frontier_revision": int(session.frontier_revision),
                    "pruned_count": int(session.frontier_pruned_count),
                    "runtime_versions": self._runtime_version_state(),
                    "prototype_rebuild_state": self._prototype_rebuild_state(),
                    "frontier_scoring_ready": bool(self._frontier_scoring_ready),
                }
            )
            if self.frontier_policy == "full_map_uniform":
                if int(session.uniform_frontier_index.total_count) != int(
                    session.frontier_size
                ):
                    raise RuntimeError(
                        "Uniform frontier index diverged before archive: "
                        f"world_index={int(world_index)}."
                    )
                continue
            pending_masks = session.pending_action_masks_by_state
            seq = 0
            for state_id, action in session.frontier_ring:
                if not (
                    int(pending_masks.get(int(state_id), 0)) & (1 << int(action))
                ):
                    continue
                node = session.state_nodes.get(int(state_id))
                if node is None:
                    raise RuntimeError(
                        "Score frontier ring referenced a missing SearchNode while archiving: "
                        f"world_index={int(world_index)} state_id={int(state_id)}."
                    )
                rows.append(
                    {
                        "row_type": "frontier_candidate",
                        "world_index": int(world_index),
                        "seq": int(seq),
                        "state_id": int(state_id),
                        "state_key": str(node.key),
                        "action": int(action),
                    }
                )
                seq += 1
            if int(seq) != int(session.frontier_size):
                raise RuntimeError(
                    "Score frontier ring diverged before archive: "
                    f"world_index={int(world_index)} ring={int(seq)} "
                    f"pending={int(session.frontier_size)}."
                )
        return rows

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
        archive_summary = dict(iteration_summary)
        for key in self._ITERATION_SCOPED_ARCHIVE_METRIC_KEYS:
            # Membership test, not `.get(...)`: these keys moved off every step
            # row and now live only here, so writing a `None` placeholder for an
            # iteration that never merged them would erase the value the caller
            # already put in `iteration_summary` and leave resume with no
            # fallback to read (the step rows no longer carry it).
            if key in self._iteration_summary_metrics:
                archive_summary[key] = self._iteration_summary_metrics[key]
        archive_summary.update(self._iteration_collect_phase_summary())
        node_rows = list(self._analysis_node_rows)
        edge_rows = list(self._analysis_edge_rows)
        frontier_rows = self._build_analysis_frontier_rows()
        self._iteration_step_metrics = []
        self._iteration_summary_metrics = {}
        self._iteration_collect_phase_calls = 0
        self._iteration_collect_phase_totals = {}
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
                iteration_summary=archive_summary,
                node_rows=node_rows,
                edge_rows=edge_rows,
                state_store=self._state_store,
                frontier_rows=frontier_rows,
            )
        self._last_recorded_iteration_metric = int(iteration)
        return result or None

    def finalize_explained_transition_batch(
        self,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        del transitions
        updates_completed = self._run_train_update_batch(
            progress_callback=progress_callback,
            desc="Train iter",
        )
        self._frontier_scoring_ready = True
        if updates_completed > 0:
            self._merge_iteration_summary_metric_source(
                self._build_iteration_summary_metrics(progress_phase="verify")
            )
        result: Dict[str, Any] = {
            "train_updates_completed": int(updates_completed),
            "train_schedule": self._train_schedule_name(),
            "train_phase": self._current_train_phase(),
        }
        if self._last_train_phase_timing:
            result["train_phase_timing"] = dict(self._last_train_phase_timing)
        return result

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
        self._merge_iteration_summary_metric_source(
            self._build_iteration_summary_metrics(progress_phase="verify")
        )

    def _build_iteration_summary_metrics(
        self,
        *,
        progress_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        class_rows = self._annotate_class_counts(
            self._build_class_rows(visible_count=max(self.active_class_count, self.known_class_count))
        )
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
        }
        if isinstance(progress_phase, str) and progress_phase.strip():
            metrics["progress_phase"] = str(progress_phase).strip().lower()
        metrics["dynamics_class_transition_counts"] = (
            self._summarize_dynamics_class_transition_counts(class_rows)
        )
        return metrics

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
            "train_schedule": self._train_schedule_name(),
            "contrastive_batch_size": int(self.contrastive_batch_size),
            "contrastive_train_mode": str(self.contrastive_train_mode),
            "contrastive_representation_mode": str(self.contrastive_representation_mode),
            "encode_bucket_mode": str(self.encode_bucket_mode),
            "compile_encoder": bool(self.compile_encoder),
            "compile_enabled": bool(self.compile_enabled),
            "compile_disabled_reason": self.compile_disabled_reason,
            "compile_worker_disabled_reasons": {
                int(rank): str(reason)
                for rank, reason in self._worker_compile_disabled_reasons().items()
            },
            "replay_cap": int(self.replay_cap) if self.replay_cap is not None else None,
            "frontier_scoring_ready": bool(self._frontier_scoring_ready),
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
            **self._frontier_diagnostics(),
            "last_collect": dict(self._last_collect_stats),
            "last_completed_world": dict(self._last_completed_world_summary),
            "last_train": last_train,
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
