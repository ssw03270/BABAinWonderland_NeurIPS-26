"""Process-local map actors for graph contrastive collection."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data import (
    RuntimeStatePacket,
    RuntimeStateVocab,
    canonical_state_key,
    canonical_graph_edge_identity_key_from_fields,
    runtime_state_packet_obj,
)


@dataclass(frozen=True, slots=True)
class ActorStateRef:
    state_ref: int
    state_id: int
    state_key: str
    source_depth: int
    packet: RuntimeStatePacket


@dataclass(frozen=True, slots=True)
class ActorStepWorkItem:
    request_id: int
    state_ref: int
    action: int
    action_name: str


@dataclass(frozen=True, slots=True)
class ActorStartCollect:
    dispatch_id: int
    world_index: int
    state_vocab: RuntimeStateVocab
    states: Tuple[ActorStateRef, ...]
    items: Tuple[ActorStepWorkItem, ...]
    map_name: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ActorShutdown:
    """Stop the actor worker after the current queue receive."""


@dataclass(frozen=True, slots=True)
class ActorTransitionEvent:
    request_id: int
    source_state_id: int
    source_state_key: str
    source_depth: int
    action: int
    action_name: str
    next_state_key: str
    transition_key: str
    candidate_depth: int
    next_state_packet: RuntimeStatePacket
    reward: float
    done: bool
    restore_steps: int
    map_name: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ActorTransitionBatchEvent:
    dispatch_id: int
    worker_id: int
    world_index: int
    transitions: Tuple[ActorTransitionEvent, ...]


@dataclass(frozen=True, slots=True)
class ActorDoneEvent:
    dispatch_id: int
    worker_id: int
    world_index: int
    transitions_collected: int


@dataclass(frozen=True, slots=True)
class ActorErrorEvent:
    dispatch_id: int | None
    worker_id: int
    world_index: int | None
    message: str
    traceback_text: str


class _MapCollectActorRunner:
    def __init__(
        self,
        *,
        worker_id: int,
        env: Any,
        command_queue: Any,
        event_queue: Any,
    ) -> None:
        self.worker_id = int(worker_id)
        self.env = env
        self.command_queue = command_queue
        self.event_queue = event_queue

    def close(self) -> None:
        return None

    def run_collect(self, command: ActorStartCollect) -> None:
        state_by_ref: Dict[int, ActorStateRef] = {
            int(state.state_ref): state
            for state in command.states
        }
        transitions_collected = 0
        transition_events: list[ActorTransitionEvent] = []
        transition_rows: List[Dict[str, Any]] = []

        def flush_transition_events() -> None:
            nonlocal transition_events
            if not transition_events:
                return
            self.event_queue.put(
                ActorTransitionBatchEvent(
                    dispatch_id=int(command.dispatch_id),
                    worker_id=int(self.worker_id),
                    world_index=int(command.world_index),
                    transitions=tuple(transition_events),
                )
            )
            transition_events = []

        restore_runtime_packet = getattr(self.env, "restore_runtime_packet", None)
        step_runtime_packet = getattr(self.env, "step_runtime_packet", None)
        if not callable(restore_runtime_packet) or not callable(step_runtime_packet):
            raise RuntimeError(
                "Collect actor environment must provide runtime packet restore and step methods."
            )

        for item in command.items:
            state_ref = int(item.state_ref)
            state = state_by_ref.get(state_ref)
            if state is None:
                raise RuntimeError(
                    f"Actor work item references missing state_ref={state_ref}."
                )

            restore_runtime_packet(
                state.packet,
                command.state_vocab,
                step_count=int(state.source_depth),
            )
            next_packet, reward, terminated, truncated, _info = step_runtime_packet(
                int(item.action),
                command.state_vocab,
            )
            action_name = str(item.action_name)
            next_state_obj = runtime_state_packet_obj(next_packet, command.state_vocab)
            next_state_key = canonical_state_key(next_state_obj)
            transition_key = canonical_graph_edge_identity_key_from_fields(
                world_index=int(command.world_index),
                map_name=command.map_name,
                state=str(state.state_key),
                action=action_name,
            )
            transition_rows.append(
                {
                    "item": item,
                    "state": state,
                    "next_packet": next_packet,
                    "next_state_key": str(next_state_key),
                    "transition_key": str(transition_key),
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                }
            )
            transitions_collected += 1

        for row in transition_rows:
            item = row["item"]
            state = row["state"]
            transition_events.append(
                ActorTransitionEvent(
                    request_id=int(item.request_id),
                    source_state_id=int(state.state_id),
                    source_state_key=str(state.state_key),
                    source_depth=int(state.source_depth),
                    action=int(item.action),
                    action_name=str(item.action_name),
                    next_state_key=str(row["next_state_key"]),
                    transition_key=str(row["transition_key"]),
                    candidate_depth=int(state.source_depth) + 1,
                    next_state_packet=row["next_packet"],
                    reward=float(row["reward"]),
                    done=bool(row["done"]),
                    restore_steps=1,
                    map_name=command.map_name,
                )
            )

        flush_transition_events()
        self.event_queue.put(
            ActorDoneEvent(
                dispatch_id=int(command.dispatch_id),
                worker_id=int(self.worker_id),
                world_index=int(command.world_index),
                transitions_collected=int(transitions_collected),
            )
        )


def run_map_collect_actor(
    *,
    worker_id: int,
    env_factory: Callable[[], Any],
    command_queue: Any,
    event_queue: Any,
) -> None:
    env = env_factory()
    runner = _MapCollectActorRunner(
        worker_id=int(worker_id),
        env=env,
        command_queue=command_queue,
        event_queue=event_queue,
    )
    try:
        while True:
            command = command_queue.get()
            if isinstance(command, ActorShutdown):
                return
            if not isinstance(command, ActorStartCollect):
                event_queue.put(
                    ActorErrorEvent(
                        dispatch_id=None,
                        worker_id=int(worker_id),
                        world_index=None,
                        message=f"Unexpected actor command: {type(command).__name__}.",
                        traceback_text="",
                    )
                )
                continue
            try:
                runner.run_collect(command)
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                event_queue.put(
                    ActorErrorEvent(
                        dispatch_id=int(command.dispatch_id),
                        worker_id=int(worker_id),
                        world_index=int(command.world_index),
                        message=str(exc) or exc.__class__.__name__,
                        traceback_text=traceback.format_exc(),
                    )
                )
    finally:
        runner.close()
