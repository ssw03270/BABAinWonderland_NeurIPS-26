"""Exhaustive graph-search explorers for Baba worlds."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

from .base_agent import BaseExplorer
from .dynamics_class_table import DynamicsClassTableSupport
from .exploration_visualizer import ExplorationVisualizer
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
from src.data import StateStore, canonical_graph_edge_identity_key, canonical_state_key, runtime_state_packet_key
from src.data.transition_buffer import Transition
from src.program_model import PredictionRecord, ProgramEvaluator, SandboxConfig, SandboxError, TransitionGroupClassifier
from src.web.map_display_names import resolve_map_display_name

if TYPE_CHECKING:
    from src.environments import BabaWrapper


_TSession = TypeVar("_TSession")


def shuffle_round_robin_sessions(
    sessions: Sequence[_TSession],
    *,
    seed: int,
) -> List[_TSession]:
    """Pick one seeded initial round-robin order and keep it for the run."""

    ordered = list(sessions)
    if len(ordered) <= 1:
        return ordered
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    return ordered


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """One fixed target world for graph search."""

    world_index: int
    world_seed: int
    scenario_type: Optional[str] = None
    label: str = ""

    @property
    def world_label(self) -> str:
        if isinstance(self.label, str) and self.label.strip():
            return str(self.label).strip()
        if isinstance(self.scenario_type, str) and self.scenario_type.strip():
            return str(self.scenario_type).strip()
        return f"seed={int(self.world_seed)}"


@dataclass(frozen=True, slots=True)
class SearchNode:
    """One reachable non-terminal state in a single world graph."""

    state_id: int
    key: str
    depth: int
    parent_state_id: Optional[int] = None
    parent_action: Optional[int] = None


@dataclass(slots=True)
class PendingExpansion:
    """Resume a partially expanded node without changing action order."""

    node: SearchNode
    next_action_index: int = 0
    discovered_children: List[SearchNode] = field(default_factory=list)
    scheduled_state_ids: set[int] = field(default_factory=set)


@dataclass(slots=True)
class WorldSession:
    """Mutable search state for one fixed world."""

    spec: WorldSpec
    root_state_id: Optional[int] = None
    frontier: Deque[SearchNode] = field(default_factory=deque)
    queued_state_ids: set[int] = field(default_factory=set)
    pending_expansion: Optional[PendingExpansion] = None
    state_nodes: Dict[int, SearchNode] = field(default_factory=dict)
    expanded_state_ids: set[int] = field(default_factory=set)
    world_transition_count: int = 0
    world_batch_transition_count: int = 0

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


@dataclass(slots=True)
class _CollectActorWorkerHandle:
    worker_id: int
    command_queue: Any
    runner: Any


@dataclass(frozen=True, slots=True)
class _BFSActorCandidate:
    request_id: int
    order: int
    session: WorldSession
    pending: PendingExpansion
    node: SearchNode
    action_index: int
    action: int
    action_name: str
    completes_node: bool


class BFSExplorer(BaseExplorer):
    """Breadth-first exhaustive transition explorer for fixed Baba worlds."""

    strategy_name = "bfs"
    collection_topology = "graph"
    transition_batch_scope = "world"

    def __init__(
        self,
        env: BabaWrapper,
        seed: int = 42,
        world_transition_batch_limit: Optional[int] = None,
        max_step_depth_per_world: Optional[int] = None,
        action_order: Optional[Sequence[int]] = None,
        collect_workers: Any = "auto",
        collect_env_factory: Optional[Callable[[], Any]] = None,
        evaluator: Optional[ProgramEvaluator] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        dashboard_context_lines: Optional[Sequence[str]] = None,
        dashboard_history_limit: int = 240,
        dashboard_enabled: bool = True,
    ):
        self.env = env
        self.seed = int(seed)
        self._state_store: Optional[StateStore] = None
        self.world_transition_batch_limit = self._resolve_positive_transition_limit(
            world_transition_batch_limit
        )
        self.max_step_depth_per_world = self._resolve_optional_step_depth_limit(
            max_step_depth_per_world
        )
        self.collect_workers_auto, self.collect_workers = self._resolve_collect_workers(
            collect_workers
        )
        self.collect_env_factory = collect_env_factory
        self._collect_actor_event_queue: Optional[Any] = None
        self._collect_actor_workers: Dict[int, _CollectActorWorkerHandle] = {}
        self._collect_actor_worker_count: Optional[int] = None
        self._collect_actor_dispatch_ids_by_worker: Dict[int, int] = {}
        self.num_actions = int(self.env.num_actions)
        self.action_names = tuple(
            self.env.get_action_name(action) for action in range(self.num_actions)
        )

        state_format = str(self.env.serializer.format_type)
        if state_format != "json":
            raise ValueError("BFSExplorer requires JSON state serialization.")
        self._require_runtime_packet_api()

        self._program_evaluator = evaluator or ProgramEvaluator(
            sandbox_config=sandbox_config or SandboxConfig()
        )
        self._current_source_transition_assessments: Dict[str, Dict[str, Any]] = {}
        self._group_classifier = TransitionGroupClassifier(self._program_evaluator)
        self._class_table = DynamicsClassTableSupport(
            evaluator=self._program_evaluator,
            group_classifier=self._group_classifier,
        )
        self.action_order = self._resolve_action_order(action_order)
        self.visualizer = ExplorationVisualizer(
            env=self.env,
            dashboard_history_limit=dashboard_history_limit,
            dashboard_context_lines=dashboard_context_lines,
            enabled=dashboard_enabled,
        )
        self._world_specs = self._resolve_world_specs()
        self._round_robin_world_indices = tuple(
            int(spec.world_index)
            for spec in shuffle_round_robin_sessions(self._world_specs, seed=self.seed)
        )
        if not self._world_specs:
            raise ValueError("BFSExplorer requires at least one target world.")
        self.reset()

    @property
    def known_class_count(self) -> int:
        return int(self._class_table.known_class_count)

    @property
    def active_class_count(self) -> int:
        return int(self._class_table.active_class_count)

    @property
    def state_store(self) -> StateStore:
        return self._ensure_state_store()

    def _ensure_state_store(self) -> StateStore:
        state_store = self._state_store
        if isinstance(state_store, StateStore):
            return state_store
        state_store = StateStore()
        self._state_store = state_store
        self._bind_env_state_store(state_store)
        return state_store

    def _resolve_positive_transition_limit(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None

    def _require_runtime_packet_api(self) -> None:
        required = (
            "restore_runtime_packet",
            "capture_runtime_packet",
            "step_runtime_packet",
        )
        missing = [
            name
            for name in required
            if not callable(getattr(self.env, name, None))
        ]
        if missing:
            raise RuntimeError(
                "BFSExplorer requires runtime packet environment methods: "
                + ", ".join(missing)
            )

    @staticmethod
    def _auto_collect_worker_cap() -> int:
        return max(2, min(8, int(os.cpu_count() or 1)))

    def _actor_candidate_wave_transition_limit(self) -> int:
        action_count = max(1, len(self.action_order))
        worker_count = max(1, int(self.collect_workers))
        return max(action_count, action_count * worker_count * 8)

    @classmethod
    def _resolve_collect_workers(cls, value: Any) -> tuple[bool, int]:
        if value is None:
            return True, cls._auto_collect_worker_cap()
        if isinstance(value, str) and value.strip().lower() == "auto":
            return True, cls._auto_collect_worker_cap()
        resolved = int(value)
        if resolved < 2:
            raise ValueError("collect_workers must be an integer >= 2.")
        return False, int(resolved)

    def _collect_workers_config_value(self) -> Any:
        if bool(self.collect_workers_auto):
            return "auto"
        return int(self.collect_workers)

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

    def set_state_store(self, state_store: Any) -> None:
        resolved_state_store = self._require_state_store(
            state_store,
            owner="BFSExplorer",
        )
        if self._state_store is resolved_state_store:
            self._bind_env_state_store(resolved_state_store)
            return
        if self._world_sessions_initialized or self._active_worlds:
            self.reset()
        self._state_store = resolved_state_store
        self._bind_env_state_store(resolved_state_store)

    def close(self) -> None:
        self._close_collect_actor_workers()

    def _close_collect_actor_workers(self) -> None:
        workers = dict(self._collect_actor_workers)
        if not workers:
            self._collect_actor_event_queue = None
            self._collect_actor_workers = {}
            self._collect_actor_worker_count = None
            self._collect_actor_dispatch_ids_by_worker.clear()
            return
        for handle in workers.values():
            try:
                handle.command_queue.put(ActorShutdown())
            except (RuntimeError, ValueError, OSError):
                pass
        for handle in workers.values():
            handle.runner.join(timeout=5.0)
            if handle.runner.is_alive() and hasattr(handle.runner, "terminate"):
                handle.runner.terminate()
                handle.runner.join(timeout=5.0)
        self._collect_actor_workers = {}
        self._collect_actor_event_queue = None
        self._collect_actor_worker_count = None
        self._collect_actor_dispatch_ids_by_worker.clear()

    def _spawn_collect_actor_worker(self, *, worker_id: int) -> _CollectActorWorkerHandle:
        context = multiprocessing.get_context("spawn")
        if self._collect_actor_event_queue is None:
            self._collect_actor_event_queue = context.Queue()
        command_queue = context.Queue()
        runner = context.Process(
            target=run_map_collect_actor,
            kwargs={
                "worker_id": int(worker_id),
                "env_factory": self.collect_env_factory,
                "command_queue": command_queue,
                "event_queue": self._collect_actor_event_queue,
            },
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
            raise ValueError("collect_env_factory is required for BFS actor collect.")
        try:
            pickle.dumps(self.collect_env_factory)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ValueError(
                "collect_env_factory must be picklable for BFS process collect."
            ) from exc

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

    def _prewarm_collect_actor_workers(self) -> None:
        worker_count = int(self._collect_actor_worker_count or 0)
        event_queue = self._collect_actor_event_queue
        if worker_count <= 0 or event_queue is None:
            return
        state_vocab = StateStore().runtime_state_vocab()
        active_dispatch_ids: Dict[int, int] = {}
        for worker_id in range(worker_count):
            handle = self._collect_actor_workers[int(worker_id)]
            dispatch_id = int(
                self._collect_actor_dispatch_ids_by_worker.get(int(worker_id), 0) + 1
            )
            self._collect_actor_dispatch_ids_by_worker[int(worker_id)] = int(dispatch_id)
            active_dispatch_ids[int(worker_id)] = int(dispatch_id)
            handle.command_queue.put(
                ActorStartCollect(
                    dispatch_id=int(dispatch_id),
                    world_index=0,
                    state_vocab=state_vocab,
                    states=(),
                    items=(),
                    map_name=None,
                )
            )
        active_workers = set(active_dispatch_ids.keys())
        while active_workers:
            try:
                event = event_queue.get(timeout=0.005)
            except queue.Empty:
                for worker_id in tuple(active_workers):
                    handle = self._collect_actor_workers[int(worker_id)]
                    if not handle.runner.is_alive():
                        raise RuntimeError(
                            "BFS collect actor exited during prewarm "
                            f"worker={int(worker_id)}."
                        )
                continue
            if isinstance(event, ActorDoneEvent):
                active_dispatch_id = active_dispatch_ids.get(int(event.worker_id))
                if int(event.dispatch_id) == int(active_dispatch_id or 0):
                    active_workers.discard(int(event.worker_id))
            elif isinstance(event, ActorErrorEvent):
                raise RuntimeError(
                    "BFS collect actor prewarm failed "
                    f"worker={int(event.worker_id)} world={event.world_index}: "
                    f"{event.message}\n{event.traceback_text}"
                )

    def _current_assessment_identity(self) -> tuple[Optional[str], Optional[str]]:
        return (
            self._class_table.current_version_id(),
            self._class_table.current_source_digest(),
        )

    def set_program_context(self, context: Optional[Dict[str, Any]]) -> None:
        previous_identity = self._current_assessment_identity()
        self._class_table.set_program_context(
            context=context,
            max_program_count=None,
        )
        if self._current_assessment_identity() != previous_identity:
            self._current_source_transition_assessments.clear()
        self._refresh_class_table_visualizer_payload()

    def set_collection_feedback(self, summary: Optional[Dict[str, Any]]) -> None:
        if not isinstance(summary, dict):
            return
        raw_added_count = summary.get("added_count")
        raw_dataset_size = summary.get("dataset_size")
        raw_pending_count = summary.get("pending_count")
        if isinstance(raw_added_count, int):
            self._last_added_count = int(raw_added_count)
        if isinstance(raw_dataset_size, int):
            self._last_dataset_size = int(raw_dataset_size)
        if isinstance(raw_pending_count, int):
            self._last_pending_count = int(raw_pending_count)
        self._class_table.update_canonical_assignment_stats(summary)
        frontier_size = self._active_frontier_size()
        queue_head_session = self._queue_head_session()

        self.visualizer.update_payload_only(
            caption=(
                f"{self.strategy_name} | "
                f"pre_patch canonical={self._last_dataset_size}"
            ),
            metrics={
                **self._build_visualization_metrics(
                    step_index=0,
                    action_name="WAIT",
                    dynamics_class_index=self._last_observed_class_index,
                    done=False,
                    frontier_size=frontier_size,
                    world_index=(
                        int(queue_head_session.world_index)
                        if queue_head_session is not None
                        else None
                    ),
                    map_name=self._resolve_live_map_name(queue_head_session),
                    progress_phase="verify",
                    resume_pending=bool(frontier_size > 0 or self._active_worlds),
                ),
            },
            class_rows=self._class_table.decorate_visualization_rows(
                self._class_table.build_class_rows(
                    visible_count=max(self.active_class_count, self.known_class_count)
                )
            ),
            display=self._build_display_config(),
            merge_metrics=True,
            update_dashboard=False,
        )

    def _refresh_class_table_visualizer_payload(self) -> None:
        has_payload = getattr(self.visualizer, "has_payload", None)
        if callable(has_payload) and not bool(has_payload()):
            return
        update_payload_only = getattr(self.visualizer, "update_payload_only", None)
        if not callable(update_payload_only):
            return
        update_payload_only(
            metrics={
                "known_dynamics_classes": int(self.known_class_count),
                "active_class_count": int(self.active_class_count),
                "active_dynamics_classes": int(self.active_class_count),
                "canonical_dynamics_classes": int(
                    self._group_classifier.canonical_class_count
                ),
                "canonical_classified_transitions": int(
                    self._group_classifier.canonical_classified_count
                ),
                "canonical_unassigned_transitions": int(
                    self._group_classifier.canonical_unassigned_count
                ),
            },
            class_rows=self._class_table.decorate_visualization_rows(
                self._class_table.build_class_rows(
                    visible_count=max(self.active_class_count, self.known_class_count)
                )
            ),
            merge_metrics=True,
            update_dashboard=False,
        )

    def _current_visualization_class_rows(self) -> List[Dict[str, Any]]:
        return self._class_table.decorate_visualization_rows(
            self._class_table.build_class_rows(
                visible_count=max(self.active_class_count, self.known_class_count)
            )
        )

    def _should_update_live_visualizer(self) -> bool:
        visualizer = getattr(self, "visualizer", None)
        checker = getattr(visualizer, "is_snapshot_delivery_enabled", None)
        if callable(checker):
            return bool(checker())
        return bool(getattr(visualizer, "enabled", False))

    def reset(self) -> None:
        self._close_collect_actor_workers()
        self.total_steps = 0
        self.total_restore_steps = 0
        self.total_worlds_completed = 0
        self.total_collect_calls = 0
        self.total_transitions_collected = 0
        self._active_worlds: Deque[WorldSession] = deque()
        self._all_world_sessions: Dict[int, WorldSession] = {}
        self._world_sessions_initialized = False
        self._round_robin_world_indices = tuple(
            int(spec.world_index)
            for spec in shuffle_round_robin_sessions(self._world_specs, seed=self.seed)
        )
        self._last_collect_stats: Dict[str, Any] = {}
        self._last_completed_world_summary: Dict[str, Any] = {}
        self._last_added_count = 0
        self._last_dataset_size = 0
        self._last_pending_count = 0
        self._last_observed_class_index: Optional[int] = None
        self._current_source_transition_assessments.clear()
        self._class_table.reset_collect_state()
        self.visualizer.reset()
        if callable(self.collect_env_factory):
            self._ensure_collect_actor_workers(worker_count=max(1, int(self.collect_workers)))
            self._prewarm_collect_actor_workers()

    def collect(
        self,
        max_transitions: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_on_unknown_transition: bool = False,
    ) -> List[Transition]:
        del stop_on_unknown_transition
        transitions: List[Transition] = []
        worlds_completed = 0
        world_visits = 0
        stopped_by_max_transitions = False

        self.total_collect_calls += 1
        self._class_table.reset_collect_state()

        if isinstance(max_transitions, int) and int(max_transitions) <= 0:
            self._last_collect_stats = {
                "transitions_collected": 0,
                "stopped_by_max_transitions": False,
                "worlds_started": 0,
                "worlds_processed": 0,
                "world_visits": 0,
                "worlds_completed": 0,
                "worlds_completed_total": int(self.total_worlds_completed),
                "resume_pending": bool(self._active_worlds),
                "parallel_collect_actor_enabled": bool(callable(self.collect_env_factory)),
                "world_summaries": [],
            }
            return transitions

        transition_budget = int(max_transitions) if isinstance(max_transitions, int) else None
        if self._should_use_actor_collect():
            transitions = self._collect_with_actor_workers(
                max_transitions=transition_budget,
                progress_callback=progress_callback,
            )
            self._finalize_collect_batch_assessments(transitions)
            return transitions
        worlds_started = self._ensure_world_sessions_initialized()
        world_summaries: Dict[int, Dict[str, Any]] = {}
        fair_session_transition_limit = self._fair_session_transition_limit(
            max_transitions=transition_budget,
        )
        fair_session_visits = 0

        while self._active_worlds:
            if (
                transition_budget is not None
                and len(transitions) >= int(transition_budget)
            ):
                stopped_by_max_transitions = True
                break

            session = self._active_worlds.popleft()
            world_visits += 1
            world_summary = world_summaries.get(session.world_index)
            if world_summary is None:
                world_summary = self._build_world_summary(session)
                world_summaries[session.world_index] = world_summary

            transition_count_before = int(session.world_transition_count)
            restore_steps_before = self.total_restore_steps
            expanded_before = len(session.expanded_state_ids)
            session_transition_limit = (
                self._fair_session_visit_transition_limit(
                    base_limit=fair_session_transition_limit,
                    visit_index=fair_session_visits,
                    max_transitions=transition_budget,
                )
                if fair_session_transition_limit is not None
                else None
            )
            fair_session_visits += 1

            pause_reason = self._advance_session_batch(
                session=session,
                transitions=transitions,
                max_transitions=transition_budget,
                session_transition_limit=session_transition_limit,
                progress_callback=progress_callback,
            )

            world_summary["visit_count"] = int(world_summary["visit_count"]) + 1
            world_summary["transitions_collected"] = int(
                world_summary["transitions_collected"]
            ) + max(
                0,
                int(session.world_transition_count) - transition_count_before,
            )
            world_summary["states_expanded"] = int(world_summary["states_expanded"]) + max(
                0,
                len(session.expanded_state_ids) - expanded_before,
            )
            world_summary["restore_steps"] = int(world_summary["restore_steps"]) + max(
                0,
                self.total_restore_steps - restore_steps_before,
            )
            world_summary["unique_states_discovered"] = int(len(session.state_nodes))
            world_summary["max_depth"] = max(
                (int(node.depth) for node in session.state_nodes.values()),
                default=0,
            )

            session_active = self._session_is_active(session)
            world_summary["resume_pending"] = bool(session_active)

            if session_active:
                if (
                    pause_reason == "collect_budget"
                    and fair_session_transition_limit is None
                ):
                    self._active_worlds.appendleft(session)
                else:
                    if pause_reason == "world_batch":
                        session.world_batch_transition_count = 0
                    self._active_worlds.append(session)
            else:
                worlds_completed += 1
                self.total_worlds_completed += 1
                self._last_completed_world_summary = dict(world_summary)

            if (
                transition_budget is not None
                and len(transitions) >= int(transition_budget)
            ):
                stopped_by_max_transitions = True
                break

            if pause_reason == "collect_budget":
                break

        self._last_collect_stats = {
            "worlds_started": int(worlds_started),
            "worlds_processed": int(len(world_summaries)),
            "world_visits": int(world_visits),
            "worlds_completed": int(worlds_completed),
            "transitions_collected": int(len(transitions)),
            "stopped_by_max_transitions": bool(stopped_by_max_transitions),
            "worlds_completed_total": int(self.total_worlds_completed),
            "resume_pending": bool(len(self._active_worlds) > 0),
            "world_summaries": list(world_summaries.values()),
        }
        self._finalize_collect_batch_assessments(transitions)
        return transitions

    def _should_use_actor_collect(self) -> bool:
        return bool(callable(self.collect_env_factory))

    def _collect_with_actor_workers(
        self,
        *,
        max_transitions: Optional[int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> List[Transition]:
        transitions: List[Transition] = []
        worlds_started = self._ensure_world_sessions_initialized()
        world_summaries: Dict[int, Dict[str, Any]] = {}
        worlds_completed = 0
        world_visits = 0
        stopped_by_max_transitions = False
        fair_session_transition_limit = self._fair_session_transition_limit(
            max_transitions=max_transitions,
        )
        fair_session_visits = 0

        while self._active_worlds:
            if (
                max_transitions is not None
                and len(transitions) >= int(max_transitions)
            ):
                stopped_by_max_transitions = True
                break

            session = self._active_worlds.popleft()
            world_visits += 1
            world_summary = world_summaries.get(session.world_index)
            if world_summary is None:
                world_summary = self._build_world_summary(session)
                world_summaries[session.world_index] = world_summary

            transition_count_before = int(session.world_transition_count)
            restore_steps_before = self.total_restore_steps
            expanded_before = len(session.expanded_state_ids)
            session_transition_limit = (
                self._fair_session_visit_transition_limit(
                    base_limit=fair_session_transition_limit,
                    visit_index=fair_session_visits,
                    max_transitions=max_transitions,
                )
                if fair_session_transition_limit is not None
                else None
            )
            fair_session_visits += 1

            pause_reason = self._advance_session_batch_actor(
                session=session,
                transitions=transitions,
                max_transitions=max_transitions,
                session_transition_limit=session_transition_limit,
                progress_callback=progress_callback,
            )

            world_summary["visit_count"] = int(world_summary["visit_count"]) + 1
            world_summary["transitions_collected"] = int(
                world_summary["transitions_collected"]
            ) + max(
                0,
                int(session.world_transition_count) - transition_count_before,
            )
            world_summary["states_expanded"] = int(world_summary["states_expanded"]) + max(
                0,
                len(session.expanded_state_ids) - expanded_before,
            )
            world_summary["restore_steps"] = int(world_summary["restore_steps"]) + max(
                0,
                self.total_restore_steps - restore_steps_before,
            )
            world_summary["unique_states_discovered"] = int(len(session.state_nodes))
            world_summary["max_depth"] = max(
                (int(node.depth) for node in session.state_nodes.values()),
                default=0,
            )
            session_active = self._session_is_active(session)
            world_summary["resume_pending"] = bool(session_active)

            if session_active:
                if (
                    pause_reason == "collect_budget"
                    and fair_session_transition_limit is None
                ):
                    self._active_worlds.appendleft(session)
                else:
                    if pause_reason == "world_batch":
                        session.world_batch_transition_count = 0
                    self._active_worlds.append(session)
            else:
                worlds_completed += 1
                self.total_worlds_completed += 1
                self._last_completed_world_summary = dict(world_summary)

            if (
                max_transitions is not None
                and len(transitions) >= int(max_transitions)
            ):
                stopped_by_max_transitions = True
                break
            if pause_reason == "collect_budget":
                break

        self._last_collect_stats = {
            "worlds_started": int(worlds_started),
            "worlds_processed": int(len(world_summaries)),
            "world_visits": int(world_visits),
            "worlds_completed": int(worlds_completed),
            "transitions_collected": int(len(transitions)),
            "stopped_by_max_transitions": bool(stopped_by_max_transitions),
            "worlds_completed_total": int(self.total_worlds_completed),
            "resume_pending": bool(len(self._active_worlds) > 0),
            "parallel_collect_actor_enabled": True,
            "parallel_collect_workers": int(self._collect_actor_worker_count or 0),
            "parallel_collect_worker_cap": int(self.collect_workers),
            "parallel_collect_workers_auto": bool(self.collect_workers_auto),
            "world_summaries": list(world_summaries.values()),
        }
        return transitions

    def _frontier_size_for_session(self, session: WorldSession) -> int:
        pending = session.pending_expansion
        pending_children = (
            len(pending.discovered_children)
            if pending is not None
            else 0
        )
        return int(len(session.frontier) + pending_children)

    def _active_frontier_size(self) -> int:
        return int(
            sum(
                self._frontier_size_for_session(session)
                for session in self._active_worlds
            )
        )

    def _fair_session_transition_limit(
        self,
        *,
        max_transitions: Optional[int],
    ) -> Optional[int]:
        if self.world_transition_batch_limit is not None:
            return None
        if not isinstance(max_transitions, int) or int(max_transitions) <= 0:
            return None
        active_count = int(len(self._active_worlds))
        if active_count <= 1:
            return None
        return max(1, int(max_transitions) // active_count)

    def _fair_session_visit_transition_limit(
        self,
        *,
        base_limit: Optional[int],
        visit_index: int,
        max_transitions: Optional[int],
    ) -> Optional[int]:
        if not isinstance(base_limit, int) or int(base_limit) <= 0:
            return None
        if not isinstance(max_transitions, int) or int(max_transitions) <= 0:
            return None
        active_count = max(1, int(len(self._active_worlds)) + 1)
        remainder = int(max_transitions) % active_count
        return int(base_limit) + (1 if int(visit_index) < remainder else 0)

    def _build_display_config(self) -> Dict[str, Any]:
        return {
            "hud_title": "STATUS",
            "hud_sections": [
                {
                    "title": "Canonical Dataset",
                    "chips": [
                        ("dataset_size",),
                    ],
                },
                {
                    "title": "Search Frontier",
                    "chips": [
                        ("frontier_size",),
                    ],
                },
            ],
            "projection": {
                "enabled": False,
            },
            "class_probability": {
                "enabled": False,
            },
            "dashboard": {
                "top_left": {
                    "title": "Canonical Dataset",
                    "left_keys": ("dataset_size",),
                    "left_label": "canonical size",
                    "left_color": "#0f766e",
                    "left_ylabel": "size",
                    "left_smoothing": False,
                    "right_keys": ("added_count",),
                    "right_label": "added count",
                    "right_color": "#1D4ED8",
                    "right_ylabel": "added",
                    "right_smoothing": False,
                },
                "bottom_left": {
                    "enabled": False,
                },
            },
        }

    def _board_state_for_state_id(self, state_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if isinstance(state_id, int) and int(state_id) > 0:
            return self._ensure_state_store().state_obj(int(state_id))
        return None

    @staticmethod
    def _resolve_live_map_name(session: Optional[WorldSession]) -> Optional[str]:
        if session is None:
            return None
        label = str(session.world_label or "").strip()
        return label or None

    def _build_visualization_metrics(
        self,
        *,
        step_index: int,
        action_name: str,
        dynamics_class_index: Optional[int],
        done: bool,
        frontier_size: int,
        world_index: Optional[int] = None,
        map_name: Optional[str] = None,
        progress_phase: str = "collect",
        resume_pending: Optional[bool] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        resolved_global_step = (
            int(global_step)
            if isinstance(global_step, int)
            else int(self.total_steps)
        )
        resolved_resume_pending = (
            bool(resume_pending)
            if isinstance(resume_pending, bool)
            else bool(frontier_size > 0 or self._active_worlds)
        )
        return {
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
            "global_step": int(resolved_global_step),
            "action": str(action_name),
            "done": bool(done),
            "frontier_size": int(frontier_size),
            "added_count": int(self._last_added_count),
            "dataset_size": int(self._last_dataset_size),
            "pending_count": int(self._last_pending_count),
            "resume_pending": bool(resolved_resume_pending),
            "progress_phase": str(progress_phase),
            "known_dynamics_classes": int(self.known_class_count),
            "active_class_count": int(self.active_class_count),
            "active_dynamics_classes": int(self.active_class_count),
            "canonical_dynamics_classes": int(
                self._group_classifier.canonical_class_count
            ),
            "canonical_classified_transitions": int(
                self._group_classifier.canonical_classified_count
            ),
            "canonical_unassigned_transitions": int(
                self._group_classifier.canonical_unassigned_count
            ),
            "current_dynamics_class": (
                int(dynamics_class_index)
                if isinstance(dynamics_class_index, int) and int(dynamics_class_index) > 0
                else None
            ),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        queue_head_session = self._queue_head_session()
        all_sessions = tuple(self._all_world_sessions.values())
        frontier_size = self._active_frontier_size()
        discovered_state_count = sum(len(session.state_nodes) for session in all_sessions)
        expanded_state_count = sum(len(session.expanded_state_ids) for session in all_sessions)
        world_transition_count = sum(int(session.world_transition_count) for session in all_sessions)
        return {
            "name": self.strategy_name,
            "collection_topology": self.collection_topology,
            "seed": int(self.seed),
            "world_count": int(len(self._world_specs)),
            "action_order": list(self.action_order),
            "total_steps": int(self.total_steps),
            "total_restore_steps": int(self.total_restore_steps),
            "total_transitions_collected": int(self.total_transitions_collected),
            "total_collect_calls": int(self.total_collect_calls),
            "total_worlds_completed": int(self.total_worlds_completed),
            "last_added_count": int(self._last_added_count),
            "last_dataset_size": int(self._last_dataset_size),
            "world_transition_batch_limit": (
                int(self.world_transition_batch_limit)
                if isinstance(self.world_transition_batch_limit, int)
                else None
            ),
            "collect_workers": self._collect_workers_config_value(),
            "collect_worker_cap": int(self.collect_workers),
            "collect_workers_auto": bool(self.collect_workers_auto),
            "collect_actor_enabled": bool(callable(self.collect_env_factory)),
            "max_step_depth_per_world": (
                int(self.max_step_depth_per_world)
                if isinstance(self.max_step_depth_per_world, int)
                else None
            ),
            "queue_head_world_seed": (
                int(queue_head_session.world_seed)
                if queue_head_session is not None
                else None
            ),
            "queue_head_world_index": (
                int(queue_head_session.world_index)
                if queue_head_session is not None
                else None
            ),
            "queue_head_world_label": (
                str(queue_head_session.world_label)
                if queue_head_session is not None
                else None
            ),
            "frontier_size": int(frontier_size),
            "queue_head_pending_expansion": (
                {
                    "state_key": queue_head_session.pending_expansion.node.key,
                    "next_action_index": int(
                        queue_head_session.pending_expansion.next_action_index
                    ),
                    "discovered_children": len(
                        queue_head_session.pending_expansion.discovered_children
                    ),
                }
                if (
                    queue_head_session is not None
                    and queue_head_session.pending_expansion is not None
                )
                else None
            ),
            "active_world_count": int(len(self._active_worlds)),
            "active_world_indices": [int(session.world_index) for session in self._active_worlds],
            "active_world_labels": [str(session.world_label) for session in self._active_worlds],
            "target_worlds": self._build_target_world_payloads(),
            "discovered_state_count": int(discovered_state_count),
            "expanded_state_count": int(expanded_state_count),
            "world_transition_count": int(world_transition_count),
            "visualization": self.visualizer.get_diagnostics(),
            "last_collect": dict(self._last_collect_stats),
            "last_completed_world": dict(self._last_completed_world_summary),
            "known_class_count": int(self.known_class_count),
            "active_class_count": int(self.active_class_count),
            "current_dynamics_class": self._last_observed_class_index,
        }

    def _queue_head_session(self) -> Optional[WorldSession]:
        if not self._active_worlds:
            return None
        return self._active_worlds[0]

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

    def _resolve_world_specs(self) -> Tuple[WorldSpec, ...]:
        normalized = self._normalize_world_specs(
            self.env.list_graph_search_worlds(seed=int(self.seed))
        )
        if normalized:
            return normalized
        raise ValueError("BFSExplorer requires at least one target world.")

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

        if self._should_update_live_visualizer():
            frontier_size = self._active_frontier_size()
            lead_session = ordered_sessions[0] if ordered_sessions else None
            self.visualizer.render(
                caption=(
                    f"{self.strategy_name} | worlds={int(len(sessions))} step=0 action=START"
                ),
                metrics=self._build_visualization_metrics(
                    step_index=0,
                    action_name="START",
                    dynamics_class_index=None,
                    done=False,
                    frontier_size=frontier_size,
                    world_index=(
                        int(lead_session.world_index)
                        if lead_session is not None
                        else None
                    ),
                    map_name=self._resolve_live_map_name(lead_session),
                    resume_pending=bool(frontier_size > 0),
                ),
                class_rows=self._class_table.decorate_visualization_rows(
                    self._class_table.build_class_rows(visible_count=self.active_class_count)
                ),
                display=self._build_display_config(),
                board_state=self._board_state_for_state_id(
                    int(lead_session.root_state_id)
                    if lead_session is not None
                    and isinstance(lead_session.root_state_id, int)
                    else None
                ),
            )
        return int(len(sessions))

    def _create_world_session(self, spec: WorldSpec) -> WorldSession:
        root_state = self._reset_world(spec)
        root_state_id, root_key = self._intern_state_json(root_state)
        root_node = SearchNode(
            state_id=root_state_id,
            key=root_key,
            depth=0,
        )
        session = WorldSession(spec=spec, root_state_id=int(root_state_id))
        session.state_nodes[root_state_id] = root_node
        self._push_nodes(session, [root_node])
        return session

    def _reset_world(self, spec: WorldSpec) -> str:
        return self.env.reset_for_graph_search(
            seed=int(spec.world_seed),
            scenario_type=spec.scenario_type,
        )

    def _build_world_summary(self, session: WorldSession) -> Dict[str, Any]:
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
            "unique_states_discovered": int(len(session.state_nodes)),
            "states_expanded": 0,
            "restore_steps": 0,
            "max_depth": 0,
            "resume_pending": bool(self._session_is_active(session)),
        }

    def _build_target_world_payloads(self) -> List[Dict[str, Any]]:
        active_world_indices = {int(session.world_index) for session in self._active_worlds}
        queue_head_session = self._queue_head_session()
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
                preview_state_json = self._ensure_state_store().state_json(
                    int(session.root_state_id)
                )
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
                        self._session_is_active(session)
                        if session is not None
                        else False
                    ),
                    "preview_state_json": (
                        str(preview_state_json)
                        if isinstance(preview_state_json, str) and preview_state_json
                        else None
                    ),
                }
            )
        return payloads

    def _current_world_batch_step(self, session: WorldSession) -> int:
        return int(session.world_batch_transition_count) + 1

    def _session_is_active(self, session: WorldSession) -> bool:
        return bool(session.pending_expansion is not None or session.frontier)

    def _depth_limit_allows_state(self, depth: int) -> bool:
        if self.max_step_depth_per_world is None:
            return True
        return int(depth) <= int(self.max_step_depth_per_world)

    def _node_can_expand(self, node: SearchNode) -> bool:
        if self.max_step_depth_per_world is None:
            return True
        return int(node.depth) < int(self.max_step_depth_per_world)

    def _push_nodes(self, session: WorldSession, nodes: Iterable[SearchNode]) -> None:
        materialized = list(nodes)
        if not materialized:
            return
        for node in materialized:
            session.queued_state_ids.add(int(node.state_id))
            session.frontier.append(node)

    def _pop_frontier(self, session: WorldSession) -> SearchNode:
        node = session.frontier.popleft()
        session.queued_state_ids.discard(int(node.state_id))
        return node

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

    def _intern_state_json(self, state_json: str) -> tuple[int, str]:
        state_store = self._ensure_state_store()
        state_key = self._state_key_from_json(state_json)
        state_id = state_store.intern(
            state_json,
            state_key=state_key,
        )
        runtime_key = runtime_state_packet_key(
            state_store.runtime_state_packet(int(state_id)),
            state_store.runtime_state_vocab(),
        )
        if str(runtime_key) == str(state_key):
            return int(state_id), str(state_key)
        runtime_state_id = state_store.intern(
            state_json,
            state_key=str(runtime_key),
        )
        return int(runtime_state_id), str(runtime_key)

    def _node_state_json(self, node: SearchNode) -> str:
        state_json = self._ensure_state_store().state_json(node.state_id)
        if isinstance(state_json, str) and state_json:
            return state_json
        raise RuntimeError(f"Missing state_json for node state_id={int(node.state_id)}")

    def _schedule_node_for_expansion(
        self,
        *,
        session: WorldSession,
        node: SearchNode,
        pending: Optional[PendingExpansion],
    ) -> int:
        state_id = int(node.state_id)
        if not self._node_can_expand(node):
            return 0
        if state_id in session.expanded_state_ids or state_id in session.queued_state_ids:
            return 0
        if pending is not None:
            if state_id == int(pending.node.state_id) or state_id in pending.scheduled_state_ids:
                return 0
            pending.discovered_children.append(node)
            pending.scheduled_state_ids.add(state_id)
            return 1
        self._push_nodes(session, [node])
        return 1

    def _relax_discovered_state(
        self,
        *,
        session: WorldSession,
        pending: Optional[PendingExpansion],
        state_id: int,
        state_key: str,
        depth: int,
        parent_state_id: int,
        parent_action: int,
    ) -> tuple[Optional[SearchNode], int]:
        if not self._depth_limit_allows_state(depth):
            return None, 0

        existing = session.state_nodes.get(int(state_id))
        if existing is None:
            node = SearchNode(
                state_id=int(state_id),
                key=str(state_key),
                depth=int(depth),
                parent_state_id=int(parent_state_id),
                parent_action=int(parent_action),
            )
            session.state_nodes[int(state_id)] = node
            scheduled_count = self._schedule_node_for_expansion(
                session=session,
                node=node,
                pending=pending,
            )
            return node, int(scheduled_count)

        if int(depth) >= int(existing.depth):
            return existing, 0

        raise RuntimeError(
            "BFS depth invariant violated: a shorter path was discovered after "
            f"state_id={int(state_id)} was already stored at depth={int(existing.depth)} "
            f"but candidate depth={int(depth)}."
        )

    def _advance_session_batch_actor(
        self,
        *,
        session: WorldSession,
        transitions: List[Transition],
        max_transitions: Optional[int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        session_transition_limit: Optional[int] = None,
    ) -> str:
        world_batch_limit = self.world_transition_batch_limit
        session_transition_start = int(len(transitions))

        while True:
            if (
                max_transitions is not None
                and len(transitions) >= int(max_transitions)
            ):
                return "collect_budget"
            if (
                session_transition_limit is not None
                and int(session_transition_limit) > 0
                and int(len(transitions)) - int(session_transition_start)
                >= int(session_transition_limit)
            ):
                return "fair_world_batch"
            if (
                world_batch_limit is not None
                and int(session.world_batch_transition_count) >= int(world_batch_limit)
            ):
                return "world_batch"
            if session.pending_expansion is None:
                while session.frontier:
                    node = self._pop_frontier(session)
                    if int(node.state_id) in session.expanded_state_ids:
                        continue
                    session.expanded_state_ids.add(int(node.state_id))
                    session.pending_expansion = PendingExpansion(node=node)
                    break
                if session.pending_expansion is None:
                    return "session_completed"

            remaining_limits = [self._actor_candidate_wave_transition_limit()]
            if isinstance(max_transitions, int):
                remaining_limits.append(max(0, int(max_transitions) - int(len(transitions))))
            if isinstance(session_transition_limit, int) and int(session_transition_limit) > 0:
                remaining_limits.append(
                    max(
                        0,
                        int(session_transition_limit)
                        - (int(len(transitions)) - int(session_transition_start)),
                    )
                )
            if isinstance(world_batch_limit, int) and int(world_batch_limit) > 0:
                remaining_limits.append(
                    max(
                        0,
                        int(world_batch_limit) - int(session.world_batch_transition_count),
                    )
                )
            max_items = min(limit for limit in remaining_limits if int(limit) > 0)
            candidates = self._build_actor_candidate_wave(
                max_items=max_items,
                target_session=session,
            )
            if not candidates:
                continue
            self._ensure_collect_actor_workers(worker_count=max(1, int(self.collect_workers)))
            worker_count = min(
                max(1, len(candidates)),
                max(1, int(self._collect_actor_worker_count or self.collect_workers)),
            )
            events_by_request = self._dispatch_actor_candidate_wave(
                candidates=candidates,
                worker_count=worker_count,
            )
            for candidate in candidates:
                event = events_by_request.get(int(candidate.request_id))
                if event is None:
                    raise RuntimeError(
                        "BFS collect actor did not return request_id="
                        f"{int(candidate.request_id)}."
                    )
                self._commit_actor_transition_event(
                    candidate=candidate,
                    event=event,
                    transitions=transitions,
                    progress_callback=progress_callback,
                )

    def _build_actor_candidate_wave(
        self,
        *,
        max_items: int,
        target_session: Optional[WorldSession] = None,
    ) -> List[_BFSActorCandidate]:
        item_limit = max(1, int(max_items))
        candidates: List[_BFSActorCandidate] = []
        request_id = 0
        order = 0
        action_count = len(self.action_order)

        sessions = (target_session,) if target_session is not None else tuple(self._active_worlds)
        for session in sessions:
            if session is None:
                continue
            if len(candidates) >= item_limit:
                break
            if not self._session_is_active(session):
                continue
            while len(candidates) < item_limit:
                if (
                    self.world_transition_batch_limit is not None
                    and int(session.world_batch_transition_count)
                    >= int(self.world_transition_batch_limit)
                ):
                    break
                if session.pending_expansion is None:
                    while session.frontier:
                        node = self._pop_frontier(session)
                        if int(node.state_id) in session.expanded_state_ids:
                            continue
                        session.expanded_state_ids.add(int(node.state_id))
                        session.pending_expansion = PendingExpansion(node=node)
                        break
                    if session.pending_expansion is None:
                        break

                pending = session.pending_expansion
                if pending is None:
                    break
                if pending.next_action_index >= action_count:
                    self._push_nodes(session, pending.discovered_children)
                    session.pending_expansion = None
                    continue

                action_index = int(pending.next_action_index)
                action = int(self.action_order[action_index])
                completes_node = action_index + 1 >= action_count
                candidates.append(
                    _BFSActorCandidate(
                        request_id=int(request_id),
                        order=int(order),
                        session=session,
                        pending=pending,
                        node=pending.node,
                        action_index=int(action_index),
                        action=int(action),
                        action_name=str(self.action_names[action]),
                        completes_node=bool(completes_node),
                    )
                )
                request_id += 1
                order += 1
                pending.next_action_index += 1
                if completes_node:
                    session.pending_expansion = None

        return candidates

    def _dispatch_actor_candidate_wave(
        self,
        *,
        candidates: Sequence[_BFSActorCandidate],
        worker_count: int,
    ) -> Dict[int, ActorTransitionEvent]:
        event_queue = self._collect_actor_event_queue
        if event_queue is None:
            raise RuntimeError("BFS collect actor event queue was not initialized.")

        state_store = self._ensure_state_store()
        state_vocab = state_store.runtime_state_vocab()
        world_groups: List[List[_BFSActorCandidate]] = []
        group_index_by_world: Dict[int, int] = {}
        for candidate in candidates:
            world_index = int(candidate.session.world_index)
            group_index = group_index_by_world.get(world_index)
            if group_index is None:
                group_index = len(world_groups)
                group_index_by_world[world_index] = int(group_index)
                world_groups.append([])
            world_groups[int(group_index)].append(candidate)

        command_groups: List[List[_BFSActorCandidate]] = []
        for group in world_groups:
            shard_count = min(max(1, int(worker_count)), max(1, len(group)))
            shards: List[List[_BFSActorCandidate]] = [
                []
                for _index in range(shard_count)
            ]
            for index, candidate in enumerate(group):
                shards[int(index) % int(shard_count)].append(candidate)
            command_groups.extend(shard for shard in shards if shard)

        pending_by_worker: Dict[int, Deque[List[_BFSActorCandidate]]] = {
            worker_id: deque()
            for worker_id in range(int(worker_count))
        }
        for group_index, group in enumerate(command_groups):
            pending_by_worker[int(group_index) % int(worker_count)].append(group)

        active_dispatch_ids: Dict[int, int] = {}

        def _start_next(worker_id: int) -> None:
            queue_for_worker = pending_by_worker[int(worker_id)]
            if not queue_for_worker:
                active_dispatch_ids.pop(int(worker_id), None)
                return
            worker_candidates = queue_for_worker.popleft()
            handle = self._collect_actor_workers[int(worker_id)]
            dispatch_id = int(
                self._collect_actor_dispatch_ids_by_worker.get(int(worker_id), 0) + 1
            )
            self._collect_actor_dispatch_ids_by_worker[int(worker_id)] = int(dispatch_id)
            active_dispatch_ids[int(worker_id)] = int(dispatch_id)

            state_ref_by_node: Dict[Tuple[int, int], int] = {}
            states: List[ActorStateRef] = []
            items: List[ActorStepWorkItem] = []
            for candidate in worker_candidates:
                state_ref_key = (int(candidate.node.state_id), int(candidate.node.depth))
                state_ref = state_ref_by_node.get(state_ref_key)
                if state_ref is None:
                    state_ref = len(states)
                    state_ref_by_node[state_ref_key] = int(state_ref)
                    states.append(
                        ActorStateRef(
                            state_ref=int(state_ref),
                            state_id=int(candidate.node.state_id),
                            state_key=str(candidate.node.key),
                            source_depth=int(candidate.node.depth),
                            packet=state_store.runtime_state_packet(
                                int(candidate.node.state_id)
                            ),
                        )
                    )
                items.append(
                    ActorStepWorkItem(
                        request_id=int(candidate.request_id),
                        state_ref=int(state_ref),
                        action=int(candidate.action),
                        action_name=str(candidate.action_name),
                    )
                )

            handle.command_queue.put(
                ActorStartCollect(
                    dispatch_id=int(dispatch_id),
                    world_index=int(worker_candidates[0].session.world_index),
                    state_vocab=state_vocab,
                    states=tuple(states),
                    items=tuple(items),
                    map_name=self._resolve_live_map_name(worker_candidates[0].session),
                )
            )
            active_dispatch_ids[int(worker_id)] = int(dispatch_id)

        for worker_id in range(int(worker_count)):
            _start_next(int(worker_id))

        events_by_request: Dict[int, ActorTransitionEvent] = {}
        active_workers = set(active_dispatch_ids.keys())
        while active_workers:
            try:
                event = event_queue.get(timeout=0.005)
            except queue.Empty:
                for worker_id in tuple(active_workers):
                    handle = self._collect_actor_workers[int(worker_id)]
                    if not handle.runner.is_alive():
                        raise RuntimeError(
                            f"BFS collect actor exited unexpectedly worker={int(worker_id)}."
                        )
                continue

            if isinstance(event, ActorTransitionBatchEvent):
                active_dispatch_id = active_dispatch_ids.get(int(event.worker_id))
                if int(event.dispatch_id) != int(active_dispatch_id or 0):
                    continue
                for transition_event in event.transitions:
                    events_by_request[int(transition_event.request_id)] = transition_event
            elif isinstance(event, ActorDoneEvent):
                active_dispatch_id = active_dispatch_ids.get(int(event.worker_id))
                if int(event.dispatch_id) != int(active_dispatch_id or 0):
                    continue
                _start_next(int(event.worker_id))
                active_workers = set(active_dispatch_ids.keys())
            elif isinstance(event, ActorErrorEvent):
                active_dispatch_id = active_dispatch_ids.get(int(event.worker_id))
                if (
                    isinstance(event.dispatch_id, int)
                    and int(event.dispatch_id) != int(active_dispatch_id or 0)
                ):
                    continue
                raise RuntimeError(
                    "BFS collect actor failed "
                    f"worker={int(event.worker_id)} world={event.world_index}: "
                    f"{event.message}\n{event.traceback_text}"
                )
            elif event is not None:
                raise RuntimeError(f"Unexpected BFS collect actor event: {type(event).__name__}.")

        return events_by_request

    def _commit_actor_transition_event(
        self,
        *,
        candidate: _BFSActorCandidate,
        event: ActorTransitionEvent,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if int(event.source_state_id) != int(candidate.node.state_id):
            raise RuntimeError("BFS actor source state metadata mismatch.")
        if str(event.source_state_key) != str(candidate.node.key):
            raise RuntimeError("BFS actor source key metadata mismatch.")
        if int(event.source_depth) != int(candidate.node.depth):
            raise RuntimeError("BFS actor source depth metadata mismatch.")
        if int(event.action) != int(candidate.action):
            raise RuntimeError("BFS actor action metadata mismatch.")

        session = candidate.session
        state_store = self._ensure_state_store()
        next_state_id = state_store.intern_runtime_state_packet(
            event.next_state_packet,
            state_key=str(event.next_state_key),
        )
        transition = Transition.from_state_ids(
            state_store=state_store,
            state_id=int(candidate.node.state_id),
            action=str(candidate.action_name),
            next_state_id=int(next_state_id),
            reward=float(event.reward),
            done=bool(event.done),
            world_index=int(session.world_index),
            map_name=self._resolve_live_map_name(session),
        )
        if not bool(event.done):
            self._relax_discovered_state(
                session=session,
                pending=candidate.pending,
                state_id=int(next_state_id),
                state_key=str(event.next_state_key),
                depth=int(candidate.node.depth) + 1,
                parent_state_id=int(candidate.node.state_id),
                parent_action=int(candidate.action),
            )

        if bool(candidate.completes_node):
            self._push_nodes(session, candidate.pending.discovered_children)

        next_frontier_size = self._active_frontier_size()
        if self._should_update_live_visualizer():
            display_step = self._current_world_batch_step(session)
            self.visualizer.render(
                caption=(
                    f"{self.strategy_name} | world={session.world_label} "
                    f"step={display_step} action={candidate.action_name} done={bool(event.done)}"
                ),
                metrics=self._build_visualization_metrics(
                    step_index=display_step,
                    action_name=str(candidate.action_name),
                    dynamics_class_index=self._last_observed_class_index,
                    done=bool(event.done),
                    frontier_size=next_frontier_size,
                    world_index=int(session.world_index),
                    map_name=self._resolve_live_map_name(session),
                    resume_pending=bool(next_frontier_size > 0),
                    global_step=int(self.total_steps) + 1,
                ),
                class_rows=self._current_visualization_class_rows(),
                display=self._build_display_config(),
                board_state=self._board_state_for_state_id(int(next_state_id)),
            )

        session.world_transition_count += 1
        session.world_batch_transition_count += 1
        transitions.append(transition)
        self.total_steps += 1
        self.total_transitions_collected += 1
        self.total_restore_steps += int(event.restore_steps)

        if callable(progress_callback):
            try:
                progress_callback(
                    {
                        "transitions_collected": len(transitions),
                        "source": self.strategy_name,
                        "world_index": int(session.world_index),
                        "world_seed": int(session.world_seed),
                        "world_label": str(session.world_label),
                        "frontier_size": int(next_frontier_size),
                        "known_class_count": int(self.known_class_count),
                        "active_class_count": int(self.active_class_count),
                    }
                )
            except Exception:
                pass

    def _advance_session_batch(
        self,
        *,
        session: WorldSession,
        transitions: List[Transition],
        max_transitions: Optional[int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        session_transition_limit: Optional[int] = None,
    ) -> str:
        world_batch_limit = self.world_transition_batch_limit
        session_transition_start = int(len(transitions))

        while True:
            if (
                max_transitions is not None
                and max_transitions > 0
                and len(transitions) >= int(max_transitions)
            ):
                return "collect_budget"
            if (
                session_transition_limit is not None
                and int(session_transition_limit) > 0
                and int(len(transitions)) - int(session_transition_start)
                >= int(session_transition_limit)
            ):
                return "fair_world_batch"
            if (
                world_batch_limit is not None
                and int(session.world_batch_transition_count) >= int(world_batch_limit)
            ):
                return "world_batch"
            if session.pending_expansion is None:
                while session.frontier:
                    node = self._pop_frontier(session)
                    if int(node.state_id) in session.expanded_state_ids:
                        continue
                    session.expanded_state_ids.add(int(node.state_id))
                    session.pending_expansion = PendingExpansion(node=node)
                    break
                if session.pending_expansion is None:
                    return "session_completed"

            pause_reason = self._advance_pending_expansion(
                session=session,
                transitions=transitions,
                max_transitions=max_transitions,
                session_transition_start=session_transition_start,
                session_transition_limit=session_transition_limit,
                progress_callback=progress_callback,
            )
            if pause_reason is not None:
                return pause_reason

    def _project_global_frontier_size(
        self,
        session: WorldSession,
        *,
        additional_children: int = 0,
    ) -> int:
        projected = self._active_frontier_size()
        projected += self._frontier_size_for_session(session)
        projected += max(0, int(additional_children))
        return int(projected)

    def _advance_pending_expansion(
        self,
        session: WorldSession,
        transitions: List[Transition],
        max_transitions: Optional[int],
        session_transition_start: int,
        session_transition_limit: Optional[int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Optional[str]:
        pending = session.pending_expansion
        if pending is None:
            return None

        action_count = len(self.action_order)
        state_store = self._ensure_state_store()
        vocab = state_store.runtime_state_vocab()
        source_packet = state_store.runtime_state_packet(int(pending.node.state_id))

        while pending.next_action_index < action_count:
            action = self.action_order[pending.next_action_index]
            action_name = self.action_names[action]
            (
                next_state_id,
                next_key,
                reward,
                terminated,
                truncated,
            ) = self._step_node_action_packet(
                node=pending.node,
                action=int(action),
                state_store=state_store,
                vocab=vocab,
                source_packet=source_packet,
            )
            done = bool(terminated or truncated)
            transition = Transition.from_state_ids(
                state_store=self._ensure_state_store(),
                state_id=int(pending.node.state_id),
                action=action_name,
                next_state_id=int(next_state_id),
                reward=float(reward),
                done=done,
                world_index=int(session.world_index),
                map_name=self._resolve_live_map_name(session),
            )
            if not done:
                self._relax_discovered_state(
                    session=session,
                    pending=pending,
                    state_id=int(next_state_id),
                    state_key=str(next_key),
                    depth=int(pending.node.depth) + 1,
                    parent_state_id=int(pending.node.state_id),
                    parent_action=int(action),
                )
            next_frontier_size = self._project_global_frontier_size(
                session,
            )
            if self._should_update_live_visualizer():
                display_step = self._current_world_batch_step(session)
                self.visualizer.render(
                    caption=(
                        f"{self.strategy_name} | world={session.world_label} "
                        f"step={display_step} action={action_name} done={done}"
                    ),
                    metrics=self._build_visualization_metrics(
                        step_index=display_step,
                        action_name=action_name,
                        dynamics_class_index=self._last_observed_class_index,
                        done=done,
                        frontier_size=next_frontier_size,
                        world_index=int(session.world_index),
                        map_name=self._resolve_live_map_name(session),
                        resume_pending=bool(next_frontier_size > 0),
                        global_step=int(self.total_steps) + 1,
                    ),
                    class_rows=self._current_visualization_class_rows(),
                    display=self._build_display_config(),
                    board_state=self._board_state_for_state_id(int(next_state_id)),
                )

            session.world_transition_count += 1
            session.world_batch_transition_count += 1
            transitions.append(transition)
            self.total_steps += 1
            self.total_transitions_collected += 1

            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "transitions_collected": len(transitions),
                            "source": self.strategy_name,
                            "world_index": int(session.world_index),
                            "world_seed": int(session.world_seed),
                            "world_label": str(session.world_label),
                            "frontier_size": int(next_frontier_size),
                            "known_class_count": int(self.known_class_count),
                            "active_class_count": int(self.active_class_count),
                        }
                    )
                except Exception:
                    pass

            pending.next_action_index += 1

            if (
                max_transitions is not None
                and max_transitions > 0
                and len(transitions) >= int(max_transitions)
                and pending.next_action_index < action_count
            ):
                return "collect_budget"
            if (
                self.world_transition_batch_limit is not None
                and int(session.world_batch_transition_count)
                >= int(self.world_transition_batch_limit)
                and pending.next_action_index < action_count
            ):
                return "world_batch"
            if (
                session_transition_limit is not None
                and int(session_transition_limit) > 0
                and int(len(transitions)) - int(session_transition_start)
                >= int(session_transition_limit)
                and pending.next_action_index < action_count
            ):
                return "fair_world_batch"

        self._push_nodes(session, pending.discovered_children)
        session.pending_expansion = None

        restored_state_key = self._ensure_state_store().state_key(
            int(pending.node.state_id)
        )
        if not isinstance(restored_state_key, str) or not restored_state_key:
            restored_state_key = self._state_key_from_json(self._node_state_json(pending.node))
        if restored_state_key != pending.node.key:
            raise RuntimeError("State restoration drift detected during graph search.")
        return None

    def _step_node_action_packet(
        self,
        *,
        node: SearchNode,
        action: int,
        state_store: StateStore,
        vocab: Any,
        source_packet: Any,
    ) -> tuple[int, str, float, bool, bool]:
        self.env.restore_runtime_packet(
            source_packet,
            vocab,
            step_count=int(node.depth),
        )
        self.total_restore_steps += 1
        next_packet, reward, terminated, truncated, _info = self.env.step_runtime_packet(
            int(action),
            vocab,
        )
        next_key = runtime_state_packet_key(next_packet, vocab)
        next_state_id = state_store.intern_runtime_state_packet(
            next_packet,
            state_key=str(next_key),
        )
        return (
            int(next_state_id),
            str(next_key),
            float(reward),
            bool(terminated),
            bool(truncated),
        )

    def _restore_node(self, session: WorldSession, node: SearchNode) -> str:
        state_store = self._ensure_state_store()
        vocab = state_store.runtime_state_vocab()
        self.env.restore_runtime_packet(
            state_store.runtime_state_packet(int(node.state_id)),
            vocab,
            step_count=int(node.depth),
        )
        self.total_restore_steps += 1
        restored_packet = self.env.capture_runtime_packet(vocab)
        restored_state_id = state_store.intern_runtime_state_packet(restored_packet)
        restored_state_key = state_store.state_key(int(restored_state_id))
        if restored_state_key != node.key:
            raise RuntimeError(
                "Restored state does not match the stored frontier node. "
                f"expected={node.key} got={restored_state_key}"
            )
        return state_store.state_json(int(restored_state_id))

    @staticmethod
    def _clone_sandbox_error(error: Optional[SandboxError]) -> Optional[SandboxError]:
        if error is None:
            return None
        return SandboxError(
            phase=str(error.phase),
            message=str(error.message),
            exception_type=(
                str(error.exception_type)
                if isinstance(error.exception_type, str)
                else None
            ),
            traceback_text=(
                str(error.traceback_text)
                if isinstance(error.traceback_text, str)
                else None
            ),
        )

    def _empty_current_source_transition_assessment(
        self,
        *,
        transition_key: str,
    ) -> Dict[str, Any]:
        return {
            "transition_key": str(transition_key),
            "current_version_id": self._class_table.current_version_id(),
            "current_source_digest": self._class_table.current_source_digest(),
            "current_explains": False,
            "predicted_next_state_json": None,
            "prediction_error": None,
        }

    def _store_current_source_transition_assessment(
        self,
        *,
        transition: Transition,
        class_id: Optional[int],
        assignment: Any,
        current_explains: Optional[bool],
        current_record: Optional[PredictionRecord],
    ) -> None:
        transition_key = canonical_graph_edge_identity_key(transition)
        assessment = self._empty_current_source_transition_assessment(
            transition_key=transition_key,
        )
        if current_record is not None:
            assessment["current_explains"] = (
                bool(current_record.is_correct) and current_record.error is None
            )
            assessment["predicted_next_state_json"] = current_record.predicted_canonical
            assessment["prediction_error"] = self._clone_sandbox_error(current_record.error)
        elif isinstance(current_explains, bool):
            assessment["current_explains"] = bool(current_explains)

        if assessment.get("current_explains") is not True:
            assessment["assigned_class_id"] = 0
            assessment["assigned_group_id"] = None
            assessment["assigned_group_label"] = None
            assessment["assigned_class_count"] = None
            assessment["assignment_status"] = "unknown"
            assessment["is_new_dynamics_class"] = True
        else:
            metadata = self._class_table.resolve_transition_display_metadata(
                class_id=class_id,
                assignment=assignment,
            )
            assessment["assigned_class_id"] = int(metadata.get("class_id") or 0)
            assessment["assigned_group_id"] = metadata.get("group_id")
            assessment["assigned_group_label"] = metadata.get("group_label")
            assessment["assigned_class_count"] = metadata.get("class_count")
            assessment["assignment_status"] = (
                str(assignment.status)
                if isinstance(getattr(assignment, "status", None), str)
                else None
            )
            assessment["is_new_dynamics_class"] = bool(
                metadata.get("is_new_dynamics_class")
            )
        self._current_source_transition_assessments[transition_key] = dict(assessment)

    def _finalize_collect_batch_assessments(
        self,
        transitions: Sequence[Transition],
        *,
        update_visualizer: bool = True,
    ) -> None:
        safe_transitions = [
            transition
            for transition in list(transitions or [])
            if isinstance(transition, Transition)
        ]
        if not safe_transitions:
            return
        should_update_visualizer = bool(update_visualizer and self._should_update_live_visualizer())
        current_records = self._class_table.current_program_transition_records(
            transitions=safe_transitions,
        )
        classified = self._class_table.classify_transitions_with_current_records(
            transitions=safe_transitions,
            current_records=current_records,
            include_rows=should_update_visualizer,
        )
        for transition, result in zip(safe_transitions, classified):
            (
                class_id,
                assignment,
                _rows,
                current_explains,
                current_record,
            ) = result
            self._store_current_source_transition_assessment(
                transition=transition,
                class_id=class_id,
                assignment=assignment,
                current_explains=current_explains,
                current_record=current_record,
            )
        if should_update_visualizer:
            self._publish_collect_batch_classification(classified)

    def _publish_collect_batch_classification(
        self,
        classified: Sequence[
            tuple[Optional[int], Any, List[Dict[str, Any]], Optional[bool], Optional[PredictionRecord]]
        ],
    ) -> None:
        if not classified:
            return
        class_id, _assignment, rows, _current_explains, _current_record = classified[-1]
        self._last_observed_class_index = (
            int(class_id)
            if isinstance(class_id, int) and int(class_id) > 0
            else None
        )
        update_payload_only = getattr(self.visualizer, "update_payload_only", None)
        if not callable(update_payload_only):
            return
        update_payload_only(
            metrics={
                "current_dynamics_class": self._last_observed_class_index,
                "known_dynamics_classes": int(self.known_class_count),
                "active_class_count": int(self.active_class_count),
                "active_dynamics_classes": int(self.active_class_count),
                "canonical_dynamics_classes": int(
                    self._group_classifier.canonical_class_count
                ),
                "canonical_classified_transitions": int(
                    self._group_classifier.canonical_classified_count
                ),
                "canonical_unassigned_transitions": int(
                    self._group_classifier.canonical_unassigned_count
                ),
            },
            class_rows=list(rows),
            merge_metrics=True,
            update_dashboard=False,
        )

    def refresh_current_source_transition_assessments(
        self,
        transitions: Optional[List[Transition]] = None,
    ) -> None:
        if not isinstance(transitions, list):
            return
        self._finalize_collect_batch_assessments(
            transitions,
            update_visualizer=False,
        )

    def consume_current_source_transition_assessments(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(key): dict(value)
            for key, value in self._current_source_transition_assessments.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
