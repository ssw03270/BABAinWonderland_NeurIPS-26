"""Shared graph-contrastive support for Baba."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import heapq
import math
from numbers import Integral
import os
from pathlib import Path
from queue import Empty, Full
import random
import shutil
import socket
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as torch_mp
from torch import Tensor, nn
from torch.nn import functional as F

from .base_agent import BaseExplorer
from .shared_nn import TOKEN_DIM, EntityTokenCodec, SetStateEncoder
from src.data import (
    StateStore,
    Transition,
    canonical_graph_edge_identity_key_from_fields,
    canonical_state_key,
)
from src.program_model import (
    GroupClassMigration,
    ProgramEvaluator,
    SandboxConfig,
    SandboxError,
    TransitionGroupClassifier,
    TransitionGroupContextSnapshot,
)


def _positive_integral(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    resolved = int(value)
    return resolved if resolved > 0 else None


def _world_bucket(value: Any) -> int:
    resolved = _positive_integral(value)
    return int(resolved) if resolved is not None else 0


@dataclass
class ContrastiveSample:
    state_json: Optional[str]
    action: int
    next_state_json: Optional[str]
    done: float
    class_id: Optional[int]
    state_id: Optional[int] = None
    next_state_id: Optional[int] = None
    state_store: Optional[StateStore] = None
    source_env_name: Optional[str] = None
    source_world_index: Optional[int] = None
    source_world_seed: Optional[int] = None
    leaf_group_id: Optional[str] = None
    assignment_status: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.state_store, StateStore):
            return
        if _positive_integral(self.state_id) is None:
            if isinstance(self.state_json, str) and self.state_json:
                self.state_id = int(self.state_store.intern(self.state_json))
        if _positive_integral(self.next_state_id) is None:
            if isinstance(self.next_state_json, str) and self.next_state_json:
                self.next_state_id = int(self.state_store.intern(self.next_state_json))
        if _positive_integral(self.state_id) is not None:
            self.state_id = int(self.state_id)
            self.state_json = None
        if _positive_integral(self.next_state_id) is not None:
            self.next_state_id = int(self.next_state_id)
            self.next_state_json = None

    def resolved_state_json(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.state_id, int):
            resolved = self.state_store.state_json(int(self.state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        return str(self.state_json) if isinstance(self.state_json, str) else ""

    def resolved_next_state_json(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.next_state_id, int):
            resolved = self.state_store.state_json(int(self.next_state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        return str(self.next_state_json) if isinstance(self.next_state_json, str) else ""

    def resolved_state_key(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.state_id, int):
            resolved = self.state_store.state_key(int(self.state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        return canonical_state_key(self.resolved_state_json())

    def resolved_next_state_key(self) -> str:
        if isinstance(self.state_store, StateStore) and isinstance(self.next_state_id, int):
            resolved = self.state_store.state_key(int(self.next_state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        return canonical_state_key(self.resolved_next_state_json())


@dataclass(frozen=True)
class SampleObservation:
    state_action_count: int
    exact_duplicate_index: Optional[int] = None
    dedup_key: Optional["SampleDedupKey"] = None


@dataclass
class ContrastiveEntropyContext:
    label_to_position: Tensor
    bonus_by_position: Tensor
    class_indices: Tensor
    class_counts: Tensor


@dataclass(frozen=True)
class PrototypeSampleSelection:
    storage_indices: Tuple[int, ...]
    class_index_ranges: Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class PackedStateBucket:
    row_indices: Tensor
    state_tokens: Tensor
    state_mask: Tensor


@dataclass(frozen=True)
class HostPackedStateBucket:
    row_indices: Tensor
    state_tokens: Tensor
    state_mask: Tensor


@dataclass(frozen=True)
class SupconBatchMetrics:
    """SupCon batch metrics with the float scalars still on the device.

    The pair counts are already on the host because the loss needs them for
    control flow anyway; ``similarity_means`` stays on the device so the caller
    can fold it into the single stats transfer at the end of the update.
    """

    active_anchor_count: int
    positive_pair_count: int
    negative_pair_count: int
    similarity_means: Tensor



SampleDedupKey = Tuple[int, str, int]
SampleStateActionKey = Tuple[int, str, int]

def _configure_torch_cuda_backends(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _resolve_parallel_cuda_device_ids(
    *,
    device: torch.device,
    visible_cuda_device_count: int,
) -> Tuple[int, ...]:
    if device.type != "cuda":
        return ()
    visible_count = max(0, int(visible_cuda_device_count))
    if visible_count <= 1:
        return ()
    primary_index = 0 if device.index is None else int(device.index)
    if primary_index < 0 or primary_index >= visible_count:
        return ()
    resolved = tuple(range(primary_index, visible_count))
    return resolved if len(resolved) > 1 else ()


_BATCH_PIN_TRACKING_ENABLED = False
_BATCH_PIN_CALLS = 0
_BATCH_PIN_SECONDS = 0.0


def _enable_batch_pin_tracking() -> None:
    global _BATCH_PIN_TRACKING_ENABLED
    _BATCH_PIN_TRACKING_ENABLED = True


def _batch_pin_totals() -> Tuple[int, float]:
    return int(_BATCH_PIN_CALLS), float(_BATCH_PIN_SECONDS)


def _pin_cpu_tensor_if_requested(tensor: Tensor, *, pin_memory: bool) -> Tensor:
    global _BATCH_PIN_CALLS, _BATCH_PIN_SECONDS
    if not bool(pin_memory):
        return tensor
    if tensor.device.type != "cpu":
        return tensor
    if _cpu_tensor_is_shared(tensor):
        return tensor
    if tensor.is_pinned():
        return tensor
    if not torch.cuda.is_available():
        return tensor
    if not _BATCH_PIN_TRACKING_ENABLED:
        return tensor.pin_memory()
    started_at = perf_counter()
    pinned = tensor.pin_memory()
    _BATCH_PIN_CALLS += 1
    _BATCH_PIN_SECONDS += perf_counter() - started_at
    return pinned


def _cpu_tensor_is_shared(tensor: Tensor) -> bool:
    if tensor.device.type != "cpu":
        return False
    is_shared = getattr(tensor, "is_shared", None)
    if not callable(is_shared):
        return False
    return bool(is_shared())


def _share_cpu_tensor_if_requested(tensor: Tensor, *, share_memory: bool) -> Tensor:
    tensor = _clone_if_inference_tensor(tensor)
    if not bool(share_memory):
        return tensor
    if tensor.device.type != "cpu":
        return tensor
    if _cpu_tensor_is_shared(tensor):
        return tensor
    if tensor.is_pinned():
        return tensor
    return tensor.share_memory_()


def _clone_if_inference_tensor(tensor: Tensor) -> Tensor:
    is_inference = getattr(torch, "is_inference", None)
    if not callable(is_inference):
        return tensor
    try:
        if bool(is_inference(tensor)):
            return tensor.clone()
    except RuntimeError:
        return tensor
    return tensor


def _synchronize_cuda_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _distributed_step_command_should_share_tensors() -> bool:
    return os.name != "nt"


_MAX_DISTRIBUTED_TRAIN_STEPS_PER_REQUEST = 20


_PSUTIL_PROCESS: Optional[Any] = None
_PSUTIL_DISABLED = False


def _process_rss_bytes() -> Optional[int]:
    """Resident set size of this process, or None when psutil is unavailable."""
    global _PSUTIL_PROCESS, _PSUTIL_DISABLED
    if _PSUTIL_DISABLED:
        return None
    if _PSUTIL_PROCESS is None:
        try:
            import psutil
        except ImportError:
            _PSUTIL_DISABLED = True
            return None
        _PSUTIL_PROCESS = psutil.Process(os.getpid())
    try:
        return int(_PSUTIL_PROCESS.memory_info().rss)
    except Exception:  # noqa: BLE001 - diagnostics must never break training.
        _PSUTIL_DISABLED = True
        return None


def _cuda_host_allocated_bytes() -> Optional[int]:
    """Bytes currently held by the CUDA pinned-host caching allocator, if exposed."""
    host_memory_stats = getattr(torch.cuda, "host_memory_stats", None)
    if not callable(host_memory_stats):
        return None
    try:
        stats = host_memory_stats()
    except Exception:  # noqa: BLE001 - stat APIs vary across torch builds.
        return None
    if not isinstance(stats, Mapping):
        return None
    value = stats.get("allocated_bytes.all.current")
    return int(value) if isinstance(value, (int, float)) else None


def _tensor_host_bytes(tensor: Tensor) -> int:
    return int(tensor.element_size()) * int(tensor.numel())


class TrainPhaseTimer:
    """Wall-clock accumulator that splits one training step into named phases.

    The timer only records while a training step has opened a window through
    ``begin_step``, so encoder calls made by other phases (prototype rebuild,
    frontier scoring) never leak into the training attribution. When the feature
    is disabled ``begin_step`` is a no-op, which makes every ``mark``/``record``
    call an attribute lookup plus a boolean test and nothing else.

    ``cuda_sync`` inserts ``torch.cuda.synchronize`` at phase boundaries. That is
    the only way to attribute asynchronous GPU work to the phase that issued it,
    but it also serializes the pipeline, so it is opt-in on top of the timer.
    """

    __slots__ = ("enabled", "cuda_sync", "device", "active", "totals", "counts")

    def __init__(
        self,
        *,
        enabled: bool,
        cuda_sync: bool,
        device: torch.device,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.cuda_sync = bool(self.enabled and cuda_sync and device.type == "cuda")
        self.active = False
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        if self.enabled:
            _enable_batch_pin_tracking()

    def reset_cycle(self) -> None:
        self.totals = {}
        self.counts = {}

    def begin_step(self) -> None:
        self.active = self.enabled

    def end_step(self) -> None:
        self.active = False

    def mark(self) -> float:
        if not self.active:
            return 0.0
        if self.cuda_sync:
            torch.cuda.synchronize(self.device)
        return perf_counter()

    def record(self, phase: str, started_at: float) -> None:
        if not self.active:
            return
        if self.cuda_sync:
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - float(started_at)
        self.totals[phase] = self.totals.get(phase, 0.0) + elapsed
        self.counts[phase] = self.counts.get(phase, 0) + 1

    def snapshot(self) -> Dict[str, float]:
        return {phase: float(seconds) for phase, seconds in self.totals.items()}


def _pack_encoded_state_batch_to_host(
    encoded_items: Sequence[Tuple[Tensor, Tensor]],
    *,
    token_dim: int,
    pin_memory: bool,
) -> Tuple[Tensor, Tensor]:
    feature_dim = max(1, int(token_dim))
    if not encoded_items:
        return (
            torch.zeros((0, 1, feature_dim), device="cpu", dtype=torch.float32),
            torch.zeros((0, 1), device="cpu", dtype=torch.bool),
        )

    entity_sizes = [int(tokens.size(0)) for tokens, _mask in encoded_items]
    if len(set(entity_sizes)) == 1:
        host_tokens = torch.stack(
            [
                tokens.detach().to(device="cpu", dtype=torch.float32)
                for tokens, _mask in encoded_items
            ],
            dim=0,
        ).contiguous()
        host_mask = torch.stack(
            [
                mask.detach().to(device="cpu", dtype=torch.bool)
                for _tokens, mask in encoded_items
            ],
            dim=0,
        ).contiguous()
        return (
            _pin_cpu_tensor_if_requested(host_tokens, pin_memory=pin_memory),
            _pin_cpu_tensor_if_requested(host_mask, pin_memory=pin_memory),
        )

    max_entities = max(1, max(entity_sizes))
    resolved_feature_dim = max(
        feature_dim,
        max(
            int(tokens.size(1))
            for tokens, _mask in encoded_items
            if int(tokens.ndim) >= 2
        ),
    )
    host_tokens = torch.zeros(
        (len(encoded_items), max_entities, resolved_feature_dim),
        dtype=torch.float32,
        pin_memory=bool(pin_memory and torch.cuda.is_available()),
    )
    host_mask = torch.zeros(
        (len(encoded_items), max_entities),
        dtype=torch.bool,
        pin_memory=bool(pin_memory and torch.cuda.is_available()),
    )
    for row_index, (tokens, mask) in enumerate(encoded_items):
        count = int(tokens.size(0))
        host_tokens[row_index, :count].copy_(
            tokens.detach().to(device="cpu", dtype=torch.float32)
        )
        host_mask[row_index, :count].copy_(
            mask.detach().to(device="cpu", dtype=torch.bool)
        )
    return host_tokens, host_mask


def _token_budget_state_bucket_rows(
    entity_sizes: Sequence[int],
    *,
    max_padded_tokens: int,
    max_padding_ratio: float,
    max_rows: Optional[int] = None,
) -> List[List[int]]:
    """Group rows into padded encode buckets, smallest entity count first.

    Rows are visited in ``(entity_size, row_index)`` order so that similar
    entity counts land in the same bucket; the greedy padding-ratio constraint
    then keeps buckets large instead of fragmenting on every size change. The
    returned indices always refer to the caller's original row positions, so
    results must be scattered back with ``index_copy_`` (or an equivalent
    position-preserving write) to keep each row's output on its own row.
    """
    if not entity_sizes:
        return []
    sorted_rows = sorted(
        range(len(entity_sizes)),
        key=lambda row_index: (int(entity_sizes[row_index]), int(row_index)),
    )
    buckets: List[List[int]] = []
    current_rows: List[int] = []
    current_entity_sum = 0
    current_max_entities = 0
    resolved_max_padded_tokens = max(1, int(max_padded_tokens))
    resolved_max_padding_ratio = max(1.0, float(max_padding_ratio))
    resolved_max_rows = max(1, int(max_rows)) if max_rows is not None else None

    for row_index in sorted_rows:
        row_size = max(1, int(entity_sizes[int(row_index)]))
        candidate_count = len(current_rows) + 1
        candidate_entity_sum = current_entity_sum + row_size
        # `sorted_rows` is ascending, so the running max equals the max over
        # `current_rows + [row_index]` without rescanning the bucket.
        candidate_max_entities = max(
            int(current_max_entities),
            int(entity_sizes[int(row_index)]),
        )
        candidate_padded_tokens = int(candidate_count * candidate_max_entities)
        candidate_padding_ratio = float(candidate_padded_tokens) / float(
            max(1, candidate_entity_sum)
        )
        should_flush = bool(
            current_rows
            and (
                candidate_padded_tokens > resolved_max_padded_tokens
                or candidate_padding_ratio > resolved_max_padding_ratio
                or (
                    resolved_max_rows is not None
                    and candidate_count > resolved_max_rows
                )
            )
        )
        if should_flush:
            buckets.append(current_rows)
            current_rows = [int(row_index)]
            current_entity_sum = row_size
            current_max_entities = int(entity_sizes[int(row_index)])
        else:
            current_rows.append(int(row_index))
            current_entity_sum = candidate_entity_sum
            current_max_entities = candidate_max_entities

    if current_rows:
        buckets.append(current_rows)
    return buckets


def _padded_cost_equalized_block_bounds(
    sorted_entity_sizes: Sequence[int],
    *,
    block_count: int,
) -> List[int]:
    """Cut an ASCENDING entity-size array into contiguous cost-equalized blocks.

    This is for consumers that pack a whole block into ONE padded
    ``[rows, max_entities, token_dim]`` tensor (the DDP frontier score payload),
    where block ``[i, j)`` therefore costs ``(j - i) * sizes[j - 1]`` padded
    tokens. Wall time is the max over blocks, so the blocks deliberately come
    out UNEQUAL: a block holding the largest states must receive fewer rows to
    match the padded cost of a block holding small ones. Equal row counts leave
    the slowest block untouched and buy nothing.

    ``cost(i, j)`` is non-decreasing in ``j`` (both factors are), which makes
    the classic "minimize the largest part" search exact: binary-search the
    bound ``T``, and for each ``T`` grow maximal blocks greedily. Everything is
    integer arithmetic on the caller's sizes, so the split is bit-identical
    across runs, platforms and process counts - no floats, no RNG, no iteration
    over an unordered container.

    Returns ``block_count + 1`` ascending bounds into ``sorted_entity_sizes``.
    Trailing bounds repeat ``len(sorted_entity_sizes)`` when there are fewer
    rows than blocks, which yields empty (but still valid) trailing blocks.
    """
    resolved_block_count = max(1, int(block_count))
    row_count = len(sorted_entity_sizes)
    if row_count <= 0:
        return [0] * (resolved_block_count + 1)
    sizes = [max(1, int(value)) for value in sorted_entity_sizes]

    def block_cost(start: int, stop: int) -> int:
        return int(stop - start) * int(sizes[int(stop) - 1])

    def greedy_bounds(bound: int) -> Optional[List[int]]:
        bounds = [0]
        start = 0
        while start < row_count:
            # `sizes` is ascending, so cost(start, start + 1) == sizes[start]
            # <= sizes[-1] <= bound: the block always advances by at least one
            # row and this loop always terminates.
            low = start + 1
            high = row_count
            while low < high:
                middle = (low + high + 1) // 2
                if block_cost(start, middle) <= bound:
                    low = middle
                else:
                    high = middle - 1
            start = low
            bounds.append(int(start))
            if len(bounds) > resolved_block_count + 1:
                return None
        return bounds

    low_bound = int(sizes[-1])
    high_bound = block_cost(0, row_count)
    while low_bound < high_bound:
        middle = (low_bound + high_bound) // 2
        if greedy_bounds(middle) is None:
            low_bound = middle + 1
        else:
            high_bound = middle
    bounds = greedy_bounds(low_bound)
    if bounds is None:
        raise RuntimeError("Padded-cost block search failed to converge.")

    # The bound search already hit the optimum, but the greedy pass can use
    # fewer blocks than there are ranks (few rows, or one dominant size). Spare
    # ranks then split the costliest block at its cost-balancing point, which
    # lowers the realized max for free.
    while len(bounds) - 1 < resolved_block_count:
        target_index = -1
        target_cost = -1
        for index in range(len(bounds) - 1):
            if int(bounds[index + 1]) - int(bounds[index]) <= 1:
                continue
            cost = block_cost(int(bounds[index]), int(bounds[index + 1]))
            if cost > target_cost:
                target_cost = int(cost)
                target_index = int(index)
        if target_index < 0:
            break
        start = int(bounds[target_index])
        stop = int(bounds[target_index + 1])
        low = start + 1
        high = stop - 1
        split = start + 1
        split_cost: Optional[int] = None
        while low <= high:
            middle = (low + high) // 2
            left_cost = block_cost(start, middle)
            right_cost = block_cost(middle, stop)
            worst = left_cost if left_cost > right_cost else right_cost
            if split_cost is None or worst < split_cost:
                split_cost = int(worst)
                split = int(middle)
            if left_cost < right_cost:
                low = middle + 1
            else:
                high = middle - 1
        bounds.insert(target_index + 1, int(split))
    while len(bounds) - 1 < resolved_block_count:
        bounds.append(int(row_count))
    return bounds



class RollingScalarWindow:
    def __init__(self, maxlen: int):
        self.values = deque(maxlen=max(1, int(maxlen)))

    def clear(self) -> None:
        self.values.clear()

    def append(self, value: Any) -> None:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(resolved):
            self.values.append(float(resolved))

    def mean(self, default: float = 0.0) -> float:
        if not self.values:
            return float(default)
        return float(sum(self.values) / float(len(self.values)))

    def std(self, default: float = 0.0) -> float:
        if not self.values:
            return float(default)
        mean_value = self.mean()
        variance = sum((float(value) - mean_value) ** 2 for value in self.values) / float(
            len(self.values)
        )
        return float(math.sqrt(max(0.0, variance)))


class ContrastiveSampleStore:
    def __init__(self, *, deduplicate_exact: bool = True):
        self.deduplicate_exact = bool(deduplicate_exact)
        self.storage: List[ContrastiveSample] = []
        self._key_to_index: Dict[SampleDedupKey, int] = {}
        self._class_to_indices: Dict[int, List[int]] = {}
        self._index_to_class: List[int] = []
        self._index_to_class_position: List[int] = []
        self._state_action_counts: Dict[SampleStateActionKey, int] = {}
        self.last_insert_index: Optional[int] = None
        self.duplicate_skips = 0

    def observe(self, item: ContrastiveSample) -> SampleObservation:
        state_action_count = self._increment_state_action_count(item)
        if not self.deduplicate_exact:
            return SampleObservation(state_action_count=state_action_count)
        key = self._dedup_key(item)
        existing_index = self._key_to_index.get(key)
        return SampleObservation(
            state_action_count=state_action_count,
            exact_duplicate_index=(
                int(existing_index) if existing_index is not None else None
            ),
            dedup_key=key,
        )

    def add(
        self,
        item: ContrastiveSample,
        *,
        observation: Optional[SampleObservation] = None,
    ) -> bool:
        resolved_observation = observation or self.observe(item)
        key = resolved_observation.dedup_key
        existing_index = resolved_observation.exact_duplicate_index
        if self.deduplicate_exact and existing_index is not None:
            existing_item = self.storage[existing_index]
            self._validate_existing_edge_outcome(existing_item, item)
            previous_class_index = self._sample_class_index(existing_item)
            self._merge_item(existing_item, item)
            merged_class_index = self._sample_class_index(existing_item)
            if merged_class_index != previous_class_index:
                self._assign_index_class(existing_index, merged_class_index)
            self.last_insert_index = int(existing_index)
            self.duplicate_skips += 1
            return False

        self.storage.append(item)
        insert_index = len(self.storage) - 1
        self._index_to_class.append(0)
        self._index_to_class_position.append(-1)
        self._assign_index_class(insert_index, self._sample_class_index(item))
        if self.deduplicate_exact:
            resolved_key = key or self._dedup_key(item)
            self._key_to_index[resolved_key] = int(insert_index)
        self.last_insert_index = int(insert_index)
        return True

    def merge_exact_without_observe(self, item: ContrastiveSample) -> Optional[int]:
        if not self.deduplicate_exact:
            return None
        existing_index = self._key_to_index.get(self._dedup_key(item))
        if existing_index is None:
            return None
        existing_item = self.storage[int(existing_index)]
        self._validate_existing_edge_outcome(existing_item, item)
        previous_class_index = self._sample_class_index(existing_item)
        self._merge_item(existing_item, item)
        merged_class_index = self._sample_class_index(existing_item)
        if merged_class_index != previous_class_index:
            self._assign_index_class(int(existing_index), merged_class_index)
        self.last_insert_index = int(existing_index)
        return int(existing_index)

    def sample_class_balanced_pairs_labeled(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        minimum_class_count: int = 2,
    ) -> List[ContrastiveSample]:
        return [
            self.storage[index]
            for index in self.sample_class_balanced_pair_indices_labeled(
                batch_size,
                rng,
                minimum_class_count=minimum_class_count,
            )
        ]

    def sample_class_balanced_pair_indices_labeled(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        minimum_class_count: int = 2,
    ) -> List[int]:
        sample_size = min(len(self.storage), max(2, int(batch_size)))
        if sample_size <= 1:
            return []
        eligible_classes = self._eligible_labeled_classes(
            minimum_class_count=max(2, int(minimum_class_count))
        )
        if not eligible_classes:
            return []

        sampled_indices: List[int] = []
        sampled_class_counts: Dict[int, int] = {}
        class_cycle = list(eligible_classes)
        rng.shuffle(class_cycle)
        class_cursor = 0

        while len(sampled_indices) + 1 < sample_size:
            if class_cursor >= len(class_cycle):
                class_cycle = list(eligible_classes)
                rng.shuffle(class_cycle)
                class_cursor = 0
            class_index = int(class_cycle[class_cursor])
            class_cursor += 1
            class_indices = self._class_to_indices.get(class_index, [])
            if len(class_indices) < 2:
                continue
            first_index, second_index = rng.sample(class_indices, 2)
            sampled_indices.extend([int(first_index), int(second_index)])
            sampled_class_counts[class_index] = int(
                sampled_class_counts.get(class_index, 0) + 2
            )

        if len(sampled_indices) < sample_size:
            represented_classes = [
                int(class_index)
                for class_index, count in sampled_class_counts.items()
                if int(count) > 0
            ]
            fallback_classes = represented_classes or list(eligible_classes)
            class_index = int(fallback_classes[rng.randrange(len(fallback_classes))])
            class_indices = self._class_to_indices.get(class_index, [])
            if class_indices:
                sampled_indices.append(int(class_indices[rng.randrange(len(class_indices))]))

        return [int(index) for index in sampled_indices[:sample_size]]

    def iter_class_balanced_pair_batches(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        minimum_class_count: int = 2,
    ) -> List[List[ContrastiveSample]]:
        return [
            [self.storage[int(index)] for index in batch_indices]
            for batch_indices in self.iter_class_balanced_pair_index_batches(
                batch_size,
                rng,
                minimum_class_count=minimum_class_count,
            )
        ]

    def iter_class_balanced_pair_index_batches(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        minimum_class_count: int = 2,
    ) -> List[List[int]]:
        eligible_classes = self._eligible_labeled_classes(
            minimum_class_count=max(2, int(minimum_class_count))
        )
        if not eligible_classes:
            return []

        pair_groups: List[List[int]] = []
        for class_index in eligible_classes:
            class_indices = list(self._class_to_indices.get(int(class_index), []))
            if len(class_indices) < 2:
                continue
            rng.shuffle(class_indices)
            pair_group: List[int] = []
            for index in class_indices:
                pair_group.append(int(index))
                if len(pair_group) == 2:
                    pair_groups.append(pair_group)
                    pair_group = []
            if pair_group:
                pair_group.append(int(class_indices[rng.randrange(len(class_indices))]))
                pair_groups.append(pair_group)
        if not pair_groups:
            return []

        rng.shuffle(pair_groups)
        ordered_indices = [int(index) for pair in pair_groups for index in pair]
        resolved_batch_size = max(2, int(batch_size))
        if resolved_batch_size % 2 != 0:
            resolved_batch_size += 1
        batches: List[List[int]] = []
        for start in range(0, len(ordered_indices), resolved_batch_size):
            batch_indices = ordered_indices[start : start + resolved_batch_size]
            if len(batch_indices) < 2:
                continue
            batches.append([int(index) for index in batch_indices])
        return batches

    def _eligible_labeled_classes(
        self,
        *,
        minimum_class_count: int,
    ) -> List[int]:
        resolved_minimum_class_count = max(1, int(minimum_class_count))
        return [
            int(class_index)
            for class_index, class_indices in self._class_to_indices.items()
            if int(class_index) > 0
            and len(class_indices) >= resolved_minimum_class_count
        ]

    def state_action_count(self, item: ContrastiveSample) -> int:
        key = self._state_action_key(item)
        return int(self._state_action_counts.get(key, 0))

    def state_action_count_by_key(
        self,
        state_key: str,
        action: int,
        *,
        world_index: Any = None,
    ) -> int:
        return int(
            self._state_action_counts.get(
                (
                    _world_bucket(world_index),
                    str(state_key),
                    int(action),
                ),
                0,
            )
        )

    def class_count(self, class_index: int) -> int:
        return int(len(self._class_to_indices.get(int(class_index), [])))

    def class_counts(self, *, include_zero: bool = True) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for raw_class_index, class_indices in self._class_to_indices.items():
            class_index = int(raw_class_index)
            if not include_zero and class_index <= 0:
                continue
            count = int(len(class_indices))
            if count <= 0:
                continue
            counts[class_index] = count
        return counts

    def clear(self) -> None:
        self.storage.clear()
        self._key_to_index.clear()
        self._class_to_indices.clear()
        self._index_to_class.clear()
        self._index_to_class_position.clear()
        self._state_action_counts.clear()
        self.last_insert_index = None
        self.duplicate_skips = 0

    def __iter__(self):
        return iter(self.storage)

    def __len__(self) -> int:
        return len(self.storage)

    def _dedup_key(self, item: ContrastiveSample) -> SampleDedupKey:
        return (
            _world_bucket(item.source_world_index),
            item.resolved_state_key(),
            int(item.action),
        )

    @staticmethod
    def _validate_existing_edge_outcome(current: ContrastiveSample, incoming: ContrastiveSample) -> None:
        if (
            current.resolved_next_state_key() != incoming.resolved_next_state_key()
            or bool(float(current.done)) != bool(float(incoming.done))
        ):
            raise ValueError(
                "ContrastiveSampleStore graph edge conflict: the same map-local "
                "state/action produced a different outcome."
            )

    def _merge_item(self, current: ContrastiveSample, incoming: ContrastiveSample) -> None:
        if current.state_store is None and incoming.state_store is not None:
            current.state_store = incoming.state_store
        if current.state_id is None and incoming.state_id is not None:
            current.state_id = int(incoming.state_id)
        if current.next_state_id is None and incoming.next_state_id is not None:
            current.next_state_id = int(incoming.next_state_id)
        if current.state_json is None and incoming.state_json is not None:
            current.state_json = str(incoming.state_json)
        if current.next_state_json is None and incoming.next_state_json is not None:
            current.next_state_json = str(incoming.next_state_json)
        if incoming.class_id is not None:
            current.class_id = int(incoming.class_id)
        if incoming.leaf_group_id is not None:
            current.leaf_group_id = str(incoming.leaf_group_id)
        if incoming.assignment_status is not None:
            current.assignment_status = str(incoming.assignment_status)

    def rebuild_class_index(self) -> None:
        self._class_to_indices.clear()
        self._index_to_class = [0] * len(self.storage)
        self._index_to_class_position = [-1] * len(self.storage)
        for storage_index, item in enumerate(self.storage):
            self._assign_index_class(storage_index, self._sample_class_index(item))

    def _sample_class_index(self, item: ContrastiveSample) -> int:
        raw_class_index = item.class_id
        if isinstance(raw_class_index, bool):
            return 0
        if isinstance(raw_class_index, int):
            return int(raw_class_index)
        return 0

    def _assign_index_class(self, storage_index: int, class_index: int) -> None:
        if storage_index < 0 or storage_index >= len(self.storage):
            return
        if storage_index >= len(self._index_to_class):
            return

        previous_position = int(self._index_to_class_position[storage_index])
        previous_class_index = int(self._index_to_class[storage_index])
        if previous_position >= 0:
            previous_bucket = self._class_to_indices.get(previous_class_index)
            if previous_bucket is not None and previous_position < len(previous_bucket):
                tail_storage_index = int(previous_bucket[-1])
                previous_bucket[previous_position] = tail_storage_index
                self._index_to_class_position[tail_storage_index] = previous_position
                previous_bucket.pop()
                if not previous_bucket:
                    self._class_to_indices.pop(previous_class_index, None)

        class_bucket = self._class_to_indices.setdefault(int(class_index), [])
        self._index_to_class[storage_index] = int(class_index)
        self._index_to_class_position[storage_index] = len(class_bucket)
        class_bucket.append(int(storage_index))

    def _state_action_key(self, item: ContrastiveSample) -> SampleStateActionKey:
        return (
            _world_bucket(item.source_world_index),
            item.resolved_state_key(),
            int(item.action),
        )

    def _increment_state_action_count(self, item: ContrastiveSample) -> int:
        key = self._state_action_key(item)
        next_count = int(self._state_action_counts.get(key, 0) + 1)
        self._state_action_counts[key] = next_count
        return next_count


def _encoder_compile_block_reason(device: torch.device) -> Optional[str]:
    """
    Return why ``torch.compile`` cannot be installed on this box, or ``None``
    when it can. Mirrors the ``compile_disabled_reason`` convention used by the
    dreamer baseline trainer, so an operator can read one string and know why a
    run did not get the compiled kernels.

    This only covers what is knowable *before* the first call. ``torch.compile``
    is lazy: it never raises at wrap time, so the real toolchain failures
    (ptxas, host compiler, unwritable/corrupt inductor cache, an unexpected
    graph break under ``fullgraph=True``) surface on the first invocation and
    are caught by the one-shot latch in ``ContrastiveDynamicsNetwork``.
    """
    if not hasattr(torch, "compile"):
        return "torch_compile_unavailable"
    if device.type != "cuda":
        # Inductor's CPU backend needs a host C++ compiler, and the speedup
        # targets CUDA.
        return "cpu_device"
    try:
        import triton  # type: ignore  # noqa: F401
    except Exception:
        return "triton_unavailable"
    try:
        # `from torch import _inductor`, not `import torch._inductor`: the
        # latter would bind the name `torch` in this function's local scope and
        # turn the `hasattr(torch, "compile")` check above into an
        # UnboundLocalError.
        from torch import _inductor  # type: ignore  # noqa: F401
    except Exception:
        return "inductor_unavailable"
    return None


def _install_encoder_compile(
    dynamics: "ContrastiveDynamicsNetwork",
    *,
    enabled: bool,
    device: torch.device,
) -> None:
    """
    Point the dynamics network's two encoder leaves at compiled callables.

    Only the leaves are compiled, never the module. ``torch.compile(module)``
    returns an ``OptimizedModule`` whose ``state_dict()`` keys gain an
    ``_orig_mod.`` prefix; that would break every ``load_state_dict(...,
    strict=True)`` on the DDP path and make compiled-run checkpoints unreadable
    by eager runs. Compiling bound methods leaves ``state_dict()`` byte-identical.

    ``mode`` is deliberately left at the default:
      * ``"reduce-overhead"`` (CUDA Graphs) is unsafe here. ``_ddp_score_frontier_payload``
        and ``_score_frontier_state_action_batches`` keep one ``state_context``
        alive across the whole candidate-chunk loop, which collides with CUDA
        Graph output-buffer reuse and would silently produce wrong embeddings.
      * ``"max-autotune"`` gave no speedup.

    ``dynamic=True`` compiles symbolic-shape kernels from the first call so the
    ~37 distinct entity-count bucket shapes share a handful of graphs instead of
    recompiling per shape. ``fullgraph=True`` turns any graph break into a
    first-call exception, which the latch converts into a recorded reason plus a
    permanent eager fallback - far easier to diagnose than a silent partial
    compile that quietly gives back the speedup.
    """
    if not bool(enabled):
        dynamics.compile_enabled = False
        dynamics.compile_disabled_reason = None
        return
    block_reason = _encoder_compile_block_reason(device)
    if block_reason is not None:
        dynamics.compile_enabled = False
        dynamics.compile_disabled_reason = block_reason
        return
    try:
        compiled_encode_context = torch.compile(
            dynamics.encode_state_context_eager,
            dynamic=True,
            fullgraph=True,
        )
        compiled_pool_state_action = torch.compile(
            dynamics.pool_state_action_eager,
            dynamic=True,
            fullgraph=True,
        )
    except Exception as exc:
        dynamics.compile_enabled = False
        dynamics.compile_disabled_reason = f"compile_setup_failed:{type(exc).__name__}"
        return
    dynamics.encode_context_impl = compiled_encode_context
    dynamics.pool_state_action_impl = compiled_pool_state_action
    dynamics.compile_enabled = True
    dynamics.compile_disabled_reason = None


class ContrastiveDynamicsNetwork(nn.Module):
    def __init__(
        self,
        *,
        state_encoder: SetStateEncoder,
        num_actions: int,
        state_embed_dim: int,
        contrastive_dim: int,
        num_classes: int,
        max_prototypes_per_class: int = 1,
        temperature: float = 0.1,
        num_pool_seeds: int = 1,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.state_embed_dim = max(1, int(state_embed_dim))
        self.num_pool_seeds = max(1, int(num_pool_seeds))
        self.pooled_state_dim = int(self.state_embed_dim) * int(self.num_pool_seeds)
        self.num_classes = max(1, int(num_classes))
        self.max_prototypes_per_class = max(1, int(max_prototypes_per_class))
        self.temperature = max(1e-4, float(temperature))
        self.action_pool_delta = nn.Embedding(num_actions, self.pooled_state_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.pooled_state_dim, max(1, int(contrastive_dim))),
            nn.LayerNorm(max(1, int(contrastive_dim))),
            nn.SiLU(),
            nn.Linear(max(1, int(contrastive_dim)), max(1, int(contrastive_dim))),
        )
        initial_prototypes = F.normalize(
            torch.randn(
                self._prototype_row_count(self.num_classes),
                max(1, int(contrastive_dim)),
            ),
            dim=-1,
            eps=1e-6,
        )
        self.register_buffer(
            "prototype_table",
            initial_prototypes,
        )
        self.register_buffer(
            "prototype_active_counts",
            torch.ones((self.num_classes,), dtype=torch.long),
        )
        self.forward_use_autocast = False
        self.forward_amp_dtype: Optional[torch.dtype] = None
        # torch.compile handles for the two encoder leaves; None means eager.
        # These are plain instance attributes: nn.Module.__setattr__ only
        # intercepts Parameter/Module/Tensor values, so callables land in
        # __dict__ and state_dict() stays identical to an eager run.
        self.encode_context_impl: Optional[Callable[..., Tensor]] = None
        self.pool_state_action_impl: Optional[Callable[..., Tensor]] = None
        self.compile_enabled = False
        self.compile_disabled_reason: Optional[str] = None

    def _disable_encoder_compile(self, exc: BaseException) -> None:
        """
        One-shot latch: drop back to eager permanently and record why.

        Retrying a failed compile on every batch would make the failure path
        slower than plain eager, and both leaves are dropped together because
        every realistic failure here is environmental (toolchain, cache,
        unsupported graph) and would hit the other leaf too.
        """
        self.encode_context_impl = None
        self.pool_state_action_impl = None
        self.compile_enabled = False
        self.compile_disabled_reason = f"compile_raised:{type(exc).__name__}"

    def _prototype_row_count(self, class_count: int) -> int:
        return max(0, int(class_count)) * int(self.max_prototypes_per_class)

    def ensure_prototype_capacity(self, required_count: int) -> bool:
        target_count = max(1, int(required_count))
        if target_count <= int(self.num_classes):
            return False
        with torch.no_grad():
            old_table = self.prototype_table.detach()
            old_active_counts = self.prototype_active_counts.detach()
            new_table = F.normalize(
                torch.randn(
                    self._prototype_row_count(target_count),
                    int(old_table.shape[1]),
                    device=old_table.device,
                    dtype=old_table.dtype,
                ),
                dim=-1,
                eps=1e-6,
            )
            new_table[: int(old_table.shape[0])].copy_(old_table)
            new_active_counts = torch.ones(
                (target_count,),
                device=old_active_counts.device,
                dtype=old_active_counts.dtype,
            )
            new_active_counts[: int(old_active_counts.shape[0])].copy_(old_active_counts)
        self.prototype_table = new_table
        self.prototype_active_counts = new_active_counts
        self.num_classes = int(target_count)
        return True

    def _resolve_class_indices(
        self,
        class_indices: Optional[Sequence[int] | Tensor] = None,
    ) -> Tensor:
        if class_indices is None:
            index_tensor = torch.arange(
                int(self.num_classes),
                device=self.prototype_table.device,
                dtype=torch.long,
            )
        elif isinstance(class_indices, Tensor):
            index_tensor = class_indices.to(
                device=self.prototype_table.device,
                dtype=torch.long,
            ).reshape(-1)
        else:
            index_tensor = torch.as_tensor(
                [int(index) for index in class_indices],
                device=self.prototype_table.device,
                dtype=torch.long,
            )
        if index_tensor.numel() <= 0:
            return index_tensor
        return index_tensor.clamp(min=0, max=max(0, int(self.num_classes) - 1))

    def _prototype_row_indices_for_class_indices(
        self,
        class_indices: Sequence[int] | Tensor,
    ) -> Tensor:
        class_index_tensor = self._resolve_class_indices(class_indices)
        if class_index_tensor.numel() <= 0:
            return torch.zeros(
                (0, int(self.max_prototypes_per_class)),
                device=self.prototype_table.device,
                dtype=torch.long,
            )
        prototype_offsets = torch.arange(
            int(self.max_prototypes_per_class),
            device=self.prototype_table.device,
            dtype=torch.long,
        )
        return (
            class_index_tensor.unsqueeze(1) * int(self.max_prototypes_per_class)
            + prototype_offsets.unsqueeze(0)
        )

    def set_prototype_vectors(
        self,
        target_class_index: int,
        prototype_vectors: Tensor,
    ) -> bool:
        target_index = int(target_class_index) - 1
        if target_index < 0:
            return False
        self.ensure_prototype_capacity(target_index + 1)
        if target_index >= int(self.num_classes):
            return False
        vectors = _clone_if_inference_tensor(prototype_vectors.detach()).to(
            device=self.prototype_table.device,
            dtype=self.prototype_table.dtype,
        )
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.ndim != 2 or int(vectors.shape[1]) != int(self.prototype_table.shape[1]):
            return False
        prototype_count = min(max(1, int(vectors.shape[0])), int(self.max_prototypes_per_class))
        vectors = vectors[:prototype_count]
        normalized_vectors = F.normalize(vectors, dim=-1, eps=1e-6)
        if not bool(torch.isfinite(normalized_vectors).all().item()):
            return False
        row_start = target_index * int(self.max_prototypes_per_class)
        row_end = row_start + int(self.max_prototypes_per_class)
        with torch.no_grad():
            self.prototype_table[row_start:row_end].zero_()
            self.prototype_table[row_start : row_start + prototype_count].copy_(normalized_vectors)
            self.prototype_active_counts[target_index] = int(prototype_count)
        return True

    def encode_state_context_eager(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Leaf A: the set-transformer encode. Pure tensor code, no host branches."""
        return self.state_encoder.encode_context(tokens, mask)

    def pool_state_action_eager(
        self,
        state_context: Tensor,
        state_mask: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """
        Leaf B: action-conditioned pooling + fusion. Pure tensor code.

        Batch extents are expressed as ``-1`` rather than ``int(actions.size(0))``
        so nothing here specializes on a concrete batch size under ``dynamic=True``.
        ``pool.seed`` is ``[1, num_seeds, embed_dim]`` and broadcasts over the
        batch, which is exactly what an explicit ``expand`` would have produced.
        """
        delta_query = self.action_pool_delta(actions).reshape(
            -1,
            self.num_pool_seeds,
            self.state_embed_dim,
        )
        pooled_state = self.state_encoder.pool_context(
            state_context,
            state_mask,
            queries=self.state_encoder.pool.seed + delta_query,
        )
        if pooled_state.ndim == 3:
            pooled_state = pooled_state.reshape(-1, self.pooled_state_dim)
        return F.normalize(self.fusion(pooled_state), dim=-1, eps=1e-6)

    def encode_state_context(self, tokens: Tensor, mask: Tensor) -> Tensor:
        compiled = self.encode_context_impl
        if compiled is None:
            return self.encode_state_context_eager(tokens, mask)
        try:
            return compiled(tokens, mask)
        except torch.cuda.OutOfMemoryError:
            # Out of memory says nothing about the compiled kernels, and eager
            # needs at least as much memory. Latching here would permanently
            # slow the run down for a transient allocation failure, and would
            # record a misleading reason, so let it propagate.
            raise
        except Exception as exc:
            # torch.compile only fails on first invocation. Latch to eager and
            # re-run: a genuine model error re-raises from the eager path, a
            # compile/toolchain error does not.
            self._disable_encoder_compile(exc)
            return self.encode_state_context_eager(tokens, mask)

    def encode_state_action_from_context(
        self,
        state_context: Tensor,
        state_mask: Tensor,
        actions: Tensor,
    ) -> Tensor:
        if actions.ndim == 2:
            actions = actions.squeeze(1)
        if actions.ndim != 1:
            raise ValueError("Actions must have shape [batch] for action-conditioned pooling.")
        action_count = int(actions.size(0))
        if state_context.ndim != 3:
            raise ValueError("State context must have shape [batch, num_tokens, embed_dim].")
        if state_mask.ndim != 2:
            raise ValueError("State mask must have shape [batch, num_tokens].")
        if int(state_context.size(0)) != action_count:
            if int(state_context.size(0)) == 1 and action_count > 1:
                state_context = state_context.expand(action_count, -1, -1)
                state_mask = state_mask.expand(action_count, -1)
            else:
                raise ValueError("State context batch size must match action batch size.")
        compiled = self.pool_state_action_impl
        if compiled is None:
            return self.pool_state_action_eager(state_context, state_mask, actions)
        try:
            return compiled(state_context, state_mask, actions)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            self._disable_encoder_compile(exc)
            return self.pool_state_action_eager(state_context, state_mask, actions)

    def encode_state_action(self, tokens: Tensor, mask: Tensor, actions: Tensor) -> Tensor:
        state_context = self.encode_state_context(tokens, mask)
        return self.encode_state_action_from_context(state_context, mask, actions)

    def _forward_autocast_context(self, tokens: Tensor):
        if (
            not self.forward_use_autocast
            or self.forward_amp_dtype is None
            or tokens.device.type != "cuda"
        ):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.forward_amp_dtype)

    def forward(self, tokens: Tensor, mask: Tensor, actions: Tensor) -> Tensor:
        with self._forward_autocast_context(tokens):
            return self.encode_state_action(tokens, mask, actions)

    def prototype_sets_by_class_index(
        self,
        class_indices: Sequence[int] | Tensor,
    ) -> Tensor:
        row_indices = self._prototype_row_indices_for_class_indices(class_indices)
        if row_indices.numel() <= 0:
            return self.prototype_table[:0].reshape(
                0,
                int(self.max_prototypes_per_class),
                int(self.prototype_table.shape[1]),
            )
        flat_rows = self.prototype_table.index_select(0, row_indices.reshape(-1))
        return flat_rows.reshape(
            int(row_indices.shape[0]),
            int(self.max_prototypes_per_class),
            int(self.prototype_table.shape[1]),
        )

    def active_prototype_counts_by_class_index(
        self,
        class_indices: Sequence[int] | Tensor,
    ) -> Tensor:
        class_index_tensor = self._resolve_class_indices(class_indices)
        if class_index_tensor.numel() <= 0:
            return torch.zeros((0,), device=self.prototype_table.device, dtype=torch.long)
        counts = self.prototype_active_counts.index_select(0, class_index_tensor)
        return counts.clamp(min=1, max=int(self.max_prototypes_per_class))

    def total_active_prototype_count(self) -> int:
        return int(
            self.prototype_active_counts.clamp(
                min=1,
                max=int(self.max_prototypes_per_class),
            ).sum().item()
        )

    def prototype_centroids_by_class_index(
        self,
        class_indices: Sequence[int] | Tensor,
    ) -> Tensor:
        prototype_sets = self.prototype_sets_by_class_index(class_indices)
        if prototype_sets.numel() <= 0:
            return prototype_sets.reshape(0, int(self.prototype_table.shape[1]))
        active_counts = self.active_prototype_counts_by_class_index(class_indices).to(
            device=prototype_sets.device,
            dtype=torch.float32,
        )
        active_mask = (
            torch.arange(
                int(self.max_prototypes_per_class),
                device=prototype_sets.device,
                dtype=torch.long,
            ).unsqueeze(0)
            < active_counts.to(dtype=torch.long).unsqueeze(1)
        ).to(dtype=prototype_sets.dtype)
        class_centroids = (
            prototype_sets * active_mask.unsqueeze(-1)
        ).sum(dim=1) / active_counts.unsqueeze(1).clamp_min(1.0)
        return F.normalize(class_centroids, dim=-1, eps=1e-6)

    def contrastive_logits(
        self,
        *,
        keys: Tensor,
        class_indices: Optional[Sequence[int] | Tensor] = None,
    ) -> Tensor:
        class_index_tensor = self._resolve_class_indices(class_indices)
        prototype_sets = self.prototype_sets_by_class_index(class_index_tensor)
        active_counts = self.active_prototype_counts_by_class_index(class_index_tensor)
        compute_device = prototype_sets.device
        compute_dtype = torch.promote_types(keys.dtype, prototype_sets.dtype)
        normalized_keys = keys.to(device=compute_device, dtype=compute_dtype)
        prototype_sets_for_logits = prototype_sets.to(
            device=compute_device,
            dtype=compute_dtype,
        )
        if prototype_sets_for_logits.numel() <= 0:
            return torch.zeros(
                (int(normalized_keys.shape[0]), 0),
                device=compute_device,
                dtype=compute_dtype,
            )
        prototype_logits = torch.einsum(
            "bd,ckd->bck",
            normalized_keys,
            prototype_sets_for_logits,
        ) / self.temperature
        active_counts = active_counts.to(device=compute_device, dtype=torch.long).clamp(
            min=1,
            max=int(self.max_prototypes_per_class),
        )
        active_mask = (
            torch.arange(
                int(self.max_prototypes_per_class),
                device=compute_device,
                dtype=torch.long,
            ).unsqueeze(0)
            < active_counts.unsqueeze(1)
        )
        masked_logits = prototype_logits.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        return torch.logsumexp(masked_logits, dim=-1) - torch.log(
            active_counts.to(device=compute_device, dtype=compute_dtype)
        ).unsqueeze(0)


def _amp_dtype_name(dtype: Optional[torch.dtype]) -> Optional[str]:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    return None


def _amp_dtype_from_name(name: Optional[str]) -> Optional[torch.dtype]:
    normalized = str(name or "").strip().lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    return None


def _tensor_tree_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _clone_if_inference_tensor(value.detach().to(device="cpu"))
    if isinstance(value, dict):
        return {key: _tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(item) for item in value)
    return value


def _find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_worker_dynamics(
    *,
    model_config: Dict[str, Any],
    device: torch.device,
    forward_amp_dtype: Optional[torch.dtype],
    forward_use_autocast: bool,
) -> ContrastiveDynamicsNetwork:
    state_encoder = SetStateEncoder(
        token_dim=TOKEN_DIM,
        embed_dim=int(model_config["dynamics_encoder_embed_dim"]),
        num_heads=int(model_config["dynamics_encoder_num_heads"]),
        num_blocks=int(model_config["dynamics_encoder_num_blocks"]),
        dropout=float(model_config["dynamics_encoder_dropout"]),
        num_pool_seeds=int(model_config["dynamics_pool_seeds"]),
    ).to(device)
    dynamics = ContrastiveDynamicsNetwork(
        state_encoder=state_encoder,
        num_actions=int(model_config["num_actions"]),
        state_embed_dim=int(model_config["dynamics_encoder_embed_dim"]),
        contrastive_dim=int(model_config["contrastive_dim"]),
        num_classes=int(model_config["num_dynamics_classes"]),
        max_prototypes_per_class=int(model_config["max_prototypes_per_class"]),
        temperature=float(model_config["contrastive_temperature"]),
        num_pool_seeds=int(model_config["dynamics_pool_seeds"]),
    ).to(device)
    dynamics.forward_use_autocast = bool(forward_use_autocast)
    dynamics.forward_amp_dtype = forward_amp_dtype
    # Install before the caller wraps this module in DDP. The compiled region
    # sits *inside* DDP's forward, so DDP's autograd hooks and the inductor
    # backward meet at the same AccumulateGrad nodes as in eager.
    _install_encoder_compile(
        dynamics,
        enabled=bool(model_config.get("compile_encoder", False)),
        device=device,
    )
    return dynamics


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def _ddp_gather_with_grad(local_tensor: Tensor) -> Tuple[Tensor, Tensor]:
    from torch.distributed.nn.functional import all_gather
    import torch.distributed as dist

    local_count = torch.as_tensor(
        [int(local_tensor.size(0))],
        device=local_tensor.device,
        dtype=torch.long,
    )
    gathered_counts = [
        torch.zeros_like(local_count)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_counts, local_count)
    counts = torch.cat(gathered_counts, dim=0)
    max_count = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_count <= 0:
        return local_tensor[:0], counts
    if int(local_tensor.size(0)) < max_count:
        pad_shape = (
            max_count - int(local_tensor.size(0)),
            *tuple(int(dim) for dim in local_tensor.shape[1:]),
        )
        padding = torch.zeros(
            pad_shape,
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        padded = torch.cat([local_tensor, padding], dim=0)
    else:
        padded = local_tensor
    gathered = all_gather(padded)
    pieces = [
        tensor[: int(count.item())]
        for tensor, count in zip(tuple(gathered), counts)
        if int(count.item()) > 0
    ]
    if not pieces:
        return local_tensor[:0], counts
    return torch.cat(pieces, dim=0), counts


def _ddp_gather_no_grad(local_tensor: Tensor, counts: Tensor) -> Tensor:
    import torch.distributed as dist

    max_count = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_count <= 0:
        return local_tensor[:0]
    if int(local_tensor.size(0)) < max_count:
        pad_shape = (
            max_count - int(local_tensor.size(0)),
            *tuple(int(dim) for dim in local_tensor.shape[1:]),
        )
        padding = torch.zeros(
            pad_shape,
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        padded = torch.cat([local_tensor, padding], dim=0)
    else:
        padded = local_tensor
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    pieces = [
        tensor[: int(count.item())]
        for tensor, count in zip(gathered, counts)
        if int(count.item()) > 0
    ]
    if not pieces:
        return local_tensor[:0]
    return torch.cat(pieces, dim=0)


def _row_sharded_supcon_loss(
    *,
    local_embeddings: Tensor,
    local_labels: Tensor,
    global_embeddings: Tensor,
    global_labels: Tensor,
    global_counts: Tensor,
    rank: int,
    world_size: int,
    temperature: float,
) -> Tuple[Tensor, Dict[str, float]]:
    import torch.distributed as dist

    local_count = int(local_embeddings.size(0))
    global_count = int(global_embeddings.size(0))
    if global_count <= 1:
        zero = local_embeddings.sum() * 0.0
        return zero, {
            "contrastive_loss": 0.0,
            "supcon_active_anchor_count": 0.0,
            "supcon_positive_pair_count": 0.0,
            "supcon_mean_positive_similarity": 0.0,
            "supcon_mean_negative_similarity": 0.0,
        }

    with torch.autocast(
        device_type="cuda",
        dtype=local_embeddings.dtype,
        enabled=local_embeddings.device.type == "cuda"
        and local_embeddings.dtype in (torch.float16, torch.bfloat16),
    ):
        similarity = torch.matmul(local_embeddings, global_embeddings.transpose(0, 1))
    similarity = similarity.to(dtype=torch.float32)
    logits = similarity / float(temperature)

    local_rows = torch.arange(local_count, device=logits.device, dtype=torch.long)
    global_start = int(global_counts[: int(rank)].sum().item())
    self_columns = torch.arange(
        global_start,
        global_start + local_count,
        device=logits.device,
        dtype=torch.long,
    )
    if local_count > 0:
        logits[local_rows, self_columns] = float("-inf")

    positive_mask = local_labels.reshape(-1, 1).eq(global_labels.reshape(1, -1))
    if local_count > 0:
        positive_mask[local_rows, self_columns] = False
    positive_counts = positive_mask.sum(dim=1)
    active_mask = positive_counts > 0
    if bool(active_mask.any().item()):
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_log_prob_sum = log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        local_loss_sum = (
            -positive_log_prob_sum[active_mask]
            / positive_counts[active_mask].clamp_min(1).to(dtype=torch.float32)
        ).sum()
    else:
        local_loss_sum = logits.sum() * 0.0

    local_active = active_mask.sum().to(device=logits.device, dtype=torch.float32)
    global_active = local_active.clone()
    dist.all_reduce(global_active, op=dist.ReduceOp.SUM)
    if float(global_active.item()) <= 0.0:
        zero = logits.sum() * 0.0
        return zero, {
            "contrastive_loss": 0.0,
            "supcon_active_anchor_count": 0.0,
            "supcon_positive_pair_count": 0.0,
            "supcon_mean_positive_similarity": 0.0,
            "supcon_mean_negative_similarity": 0.0,
        }
    loss = local_loss_sum * float(world_size) / global_active.detach().clamp_min(1.0)
    global_loss_sum = local_loss_sum.detach().clone()
    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
    contrastive_loss_value = float(
        (global_loss_sum / global_active.detach().clamp_min(1.0)).item()
    )

    local_positive_pair_count = positive_mask.sum().to(device=logits.device, dtype=torch.float32)
    global_positive_pair_count = local_positive_pair_count.clone()
    dist.all_reduce(global_positive_pair_count, op=dist.ReduceOp.SUM)
    positive_similarity_sum = similarity.masked_fill(~positive_mask, 0.0).sum()
    global_positive_similarity_sum = positive_similarity_sum.detach().clone()
    dist.all_reduce(global_positive_similarity_sum, op=dist.ReduceOp.SUM)
    mean_positive = (
        float((global_positive_similarity_sum / global_positive_pair_count.clamp_min(1.0)).item())
        if float(global_positive_pair_count.item()) > 0.0
        else 0.0
    )

    return loss, {
        "contrastive_loss": float(contrastive_loss_value),
        "supcon_active_anchor_count": float(global_active.item()),
        "supcon_positive_pair_count": float(global_positive_pair_count.item()),
        "supcon_mean_positive_similarity": float(mean_positive),
        "supcon_mean_negative_similarity": 0.0,
    }


def _ddp_batch_prototype_metrics(
    *,
    dynamics: ContrastiveDynamicsNetwork,
    local_embeddings: Tensor,
    local_labels: Tensor,
) -> Dict[str, float]:
    import torch.distributed as dist

    totals = torch.zeros((5,), device=local_embeddings.device, dtype=torch.float64)
    if local_embeddings.numel() > 0 and local_labels.numel() > 0:
        with torch.no_grad():
            logits = dynamics.contrastive_logits(keys=local_embeddings.detach()).to(dtype=torch.float32)
            labels_zero_based = local_labels.to(dtype=torch.long) - 1
            valid = (labels_zero_based >= 0) & (labels_zero_based < int(logits.size(1)))
            if bool(valid.any().item()):
                valid_logits = logits[valid]
                valid_labels = labels_zero_based[valid]
                rows = torch.arange(valid_labels.numel(), device=valid_labels.device)
                predictions = valid_logits.argmax(dim=-1)
                correct = (predictions == valid_labels).sum().to(dtype=torch.float64)
                positive_logits = valid_logits[rows, valid_labels].sum().to(dtype=torch.float64)
                if int(valid_logits.size(1)) > 1:
                    negative_logits = valid_logits.clone()
                    negative_logits[rows, valid_labels] = float("-inf")
                    max_negative = negative_logits.max(dim=1).values
                    finite_negative = torch.isfinite(max_negative)
                    max_negative_sum = max_negative[finite_negative].sum().to(dtype=torch.float64)
                    max_negative_count = finite_negative.sum().to(dtype=torch.float64)
                else:
                    max_negative_sum = torch.zeros((), device=valid_logits.device, dtype=torch.float64)
                    max_negative_count = torch.zeros((), device=valid_logits.device, dtype=torch.float64)
                totals = torch.stack(
                    [
                        valid_labels.numel()
                        * torch.ones((), device=valid_logits.device, dtype=torch.float64),
                        correct,
                        positive_logits,
                        max_negative_sum,
                        max_negative_count,
                    ]
                )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    sample_count = float(totals[0].item())
    if sample_count <= 0.0:
        return {
            "prototype_top1_accuracy": 0.0,
            "mean_positive_logit": 0.0,
            "mean_max_negative_logit": 0.0,
        }
    negative_count = float(totals[4].item())
    return {
        "prototype_top1_accuracy": float(totals[1].item() / sample_count),
        "mean_positive_logit": float(totals[2].item() / sample_count),
        "mean_max_negative_logit": (
            float(totals[3].item() / negative_count) if negative_count > 0.0 else 0.0
        ),
    }


def _ddp_average_step_stats(step_stats: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not step_stats:
        return {}
    metric_names = sorted(
        {
            str(key)
            for stats in step_stats
            for key, value in stats.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        key: float(
            sum(float(stats.get(key, 0.0)) for stats in step_stats)
            / float(len(step_stats))
        )
        for key in metric_names
    }


def _ddp_run_contrastive_train_step(
    *,
    ddp_model: nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step_command: Dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
    model_config: Dict[str, Any],
    amp_dtype: Optional[torch.dtype],
    use_grad_scaler: bool,
) -> Optional[Dict[str, float]]:
    import torch.distributed as dist

    local_labels = step_command["labels"].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    local_actions = step_command["actions"].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    support_counts = step_command["support_counts"].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    buckets = step_command.get("buckets") or []
    local_count = int(local_labels.numel())
    embedding_dtype = amp_dtype if amp_dtype is not None else torch.float32
    local_embeddings = torch.empty(
        (local_count, int(model_config["contrastive_dim"])),
        device=device,
        dtype=embedding_dtype,
    )
    for bucket in buckets:
        row_indices = bucket["row_indices"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        if int(row_indices.numel()) <= 0:
            continue
        tokens = bucket["state_tokens"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        mask = bucket["state_mask"].to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
        bucket_actions = local_actions.index_select(0, row_indices)
        bucket_embeddings = ddp_model(tokens, mask, bucket_actions)
        local_embeddings.index_copy_(
            0,
            row_indices,
            bucket_embeddings.to(dtype=embedding_dtype),
        )

    eligible_mask = (local_labels > 0) & (
        support_counts >= max(2, int(model_config["contrastive_min_class_count"]))
    )
    provisional_mask = (local_labels > 0) & ~eligible_mask
    eligible_embeddings = local_embeddings[eligible_mask]
    eligible_labels = local_labels[eligible_mask]
    global_embeddings, global_counts = _ddp_gather_with_grad(eligible_embeddings)
    global_labels = _ddp_gather_no_grad(eligible_labels, global_counts)
    if int(global_labels.numel()) > 0:
        loss, supcon_metrics = _row_sharded_supcon_loss(
            local_embeddings=eligible_embeddings,
            local_labels=eligible_labels,
            global_embeddings=global_embeddings,
            global_labels=global_labels,
            global_counts=global_counts,
            rank=int(rank),
            world_size=int(world_size),
            temperature=float(model_config["contrastive_temperature"]),
        )
        loss = loss + local_embeddings.sum() * 0.0
        optimizer.zero_grad(set_to_none=True)
        if bool(use_grad_scaler):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        prototype_metrics = _ddp_batch_prototype_metrics(
            dynamics=ddp_model.module,
            local_embeddings=eligible_embeddings,
            local_labels=eligible_labels,
        )
        optimizer.zero_grad(set_to_none=True)
    else:
        supcon_metrics = {
            "contrastive_loss": 0.0,
            "supcon_active_anchor_count": 0.0,
            "supcon_positive_pair_count": 0.0,
            "supcon_mean_positive_similarity": 0.0,
            "supcon_mean_negative_similarity": 0.0,
        }
        prototype_metrics = {
            "prototype_top1_accuracy": 0.0,
            "mean_positive_logit": 0.0,
            "mean_max_negative_logit": 0.0,
        }

    counts = torch.stack(
        [
            eligible_mask.sum().to(device=device, dtype=torch.float64),
            provisional_mask.sum().to(device=device, dtype=torch.float64),
        ]
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if int(rank) != 0:
        return None
    return {
        "contrastive_loss": float(supcon_metrics.get("contrastive_loss", 0.0)),
        "contrastive_eligible_sample_count": float(counts[0].item()),
        "contrastive_provisional_sample_count": float(counts[1].item()),
        **supcon_metrics,
        **prototype_metrics,
    }


def _ddp_run_contrastive_train_steps(
    *,
    ddp_model: nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step_commands: Sequence[Dict[str, Any]],
    rank: int,
    world_size: int,
    device: torch.device,
    model_config: Dict[str, Any],
    amp_dtype: Optional[torch.dtype],
    use_grad_scaler: bool,
) -> List[Dict[str, float]]:
    step_stats: List[Dict[str, float]] = []
    for step_command in step_commands:
        stats = _ddp_run_contrastive_train_step(
            ddp_model=ddp_model,
            optimizer=optimizer,
            scaler=scaler,
            step_command=step_command,
            rank=int(rank),
            world_size=int(world_size),
            device=device,
            model_config=model_config,
            amp_dtype=amp_dtype,
            use_grad_scaler=bool(use_grad_scaler),
        )
        if stats is not None:
            step_stats.append(stats)
    return step_stats


def _frontier_autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _frontier_empty_tau(model_config: Mapping[str, Any], device: torch.device) -> Tensor:
    return torch.zeros(
        (0, int(model_config["contrastive_dim"])),
        device=device,
        dtype=torch.float32,
    )


def _frontier_reward_from_neighbor_distances(
    neighbor_distances: Tensor,
    *,
    knn_avg: bool,
    knn_clip: float,
) -> Tensor:
    if not bool(knn_avg):
        reward = neighbor_distances[:, -1].reshape(-1, 1)
        if float(knn_clip) >= 0.0:
            reward = torch.maximum(reward - float(knn_clip), torch.zeros_like(reward))
    else:
        reward = neighbor_distances.reshape(-1, 1)
        if float(knn_clip) >= 0.0:
            reward = torch.maximum(reward - float(knn_clip), torch.zeros_like(reward))
        reward = reward.reshape(
            int(neighbor_distances.size(0)),
            int(neighbor_distances.size(1)),
        ).mean(dim=1, keepdim=True)
    reward = torch.log(reward + 1.0).reshape(-1)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def _normalized_particle_entropy_intrinsic_reward(
    source_keys: Tensor,
    *,
    target_keys: Tensor,
    knn_k: int,
    knn_avg: bool,
    knn_clip: float,
    chunk_size: int,
) -> Tensor:
    source_size = int(source_keys.shape[0]) if source_keys.ndim > 0 else 0
    if source_keys.ndim != 2 or source_size <= 0:
        return torch.zeros((source_size,), device=source_keys.device, dtype=torch.float32)
    if target_keys.ndim != 2 or int(target_keys.size(0)) <= 0:
        return torch.zeros((source_size,), device=source_keys.device, dtype=torch.float32)
    if int(target_keys.size(1)) != int(source_keys.size(1)):
        return torch.zeros((source_size,), device=source_keys.device, dtype=torch.float32)

    target = target_keys.to(device=source_keys.device, dtype=torch.float32)
    target_size = int(target.size(0))
    k_eff = min(max(1, int(knn_k)), target_size)
    resolved_chunk_size = max(1, int(chunk_size))
    target_t = target.transpose(0, 1).contiguous()
    rewards: List[Tensor] = []
    for start_index in range(0, source_size, resolved_chunk_size):
        end_index = min(source_size, start_index + resolved_chunk_size)
        source = source_keys[start_index:end_index].to(dtype=torch.float32)
        similarity = source @ target_t
        neighbor_similarity = torch.topk(
            similarity,
            k=k_eff,
            dim=1,
            largest=True,
            sorted=True,
        ).values
        neighbor_distances = torch.sqrt(
            torch.clamp(2.0 - 2.0 * neighbor_similarity, min=0.0)
        )
        rewards.append(
            _frontier_reward_from_neighbor_distances(
                neighbor_distances,
                knn_avg=bool(knn_avg),
                knn_clip=float(knn_clip),
            )
        )
    return torch.cat(rewards, dim=0) if rewards else source_keys.new_zeros((0,))


def _frontier_score_tau_embeddings(
    *,
    dynamics: ContrastiveDynamicsNetwork,
    tau_embeddings: Tensor,
    sample_tau: Tensor,
    class_indices: Tensor,
    bonus_by_position: Tensor,
    score_config: Mapping[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size = int(tau_embeddings.size(0)) if tau_embeddings.ndim > 0 else 0
    zero_reward = torch.zeros(
        (batch_size,),
        device=tau_embeddings.device,
        dtype=torch.float32,
    )
    intrinsic_scale = float(score_config["intrinsic_reward_scale"])
    prototype_scale = float(score_config["prototype_entropy_scale"])
    if intrinsic_scale != 0.0 and int(sample_tau.size(0)) > 0:
        tau_embeddings_fp32 = tau_embeddings.to(dtype=torch.float32)
        rh = _normalized_particle_entropy_intrinsic_reward(
            tau_embeddings_fp32,
            target_keys=sample_tau,
            knn_k=int(score_config["knn_k"]),
            knn_avg=bool(score_config["knn_avg"]),
            knn_clip=float(score_config["knn_clip"]),
            chunk_size=int(score_config["candidate_chunk_size"]),
        )
    else:
        rh = zero_reward

    if (
        int(class_indices.numel()) <= 0
        or prototype_scale <= 0.0
        or int(bonus_by_position.numel()) <= 0
        or not bool(torch.count_nonzero(bonus_by_position.detach()).item())
    ):
        rz = zero_reward
    else:
        logits = dynamics.contrastive_logits(
            keys=tau_embeddings,
            class_indices=(class_indices - 1),
        )
        logits_fp32 = logits.to(dtype=torch.float32)
        temperature_ratio = max(
            1e-4,
            float(score_config["prototype_entropy_temperature"])
            / max(1e-4, float(score_config["contrastive_temperature"])),
        )
        probs = torch.softmax(logits_fp32 / float(temperature_ratio), dim=-1)
        rz = torch.matmul(
            probs,
            bonus_by_position.to(device=logits_fp32.device, dtype=torch.float32),
        )

    rh = torch.nan_to_num(
        rh.reshape(-1).to(dtype=torch.float32) * intrinsic_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    rz = torch.nan_to_num(
        rz.reshape(-1).to(device=rh.device, dtype=torch.float32)
        * prototype_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    rtotal = torch.nan_to_num(rh + rz, nan=0.0, posinf=0.0, neginf=0.0)
    return rh, rz, rtotal



def _ddp_encode_sample_step_command(
    *,
    dynamics: ContrastiveDynamicsNetwork,
    command: Mapping[str, Any],
    device: torch.device,
    model_config: Mapping[str, Any],
    amp_dtype: Optional[torch.dtype],
) -> Tensor:
    actions = command["actions"].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    sample_count = int(actions.numel())
    if sample_count <= 0:
        return _frontier_empty_tau(model_config, device)
    encoded = torch.empty(
        (sample_count, int(model_config["contrastive_dim"])),
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for bucket in command.get("buckets") or []:
            row_indices = bucket["row_indices"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            if int(row_indices.numel()) <= 0:
                continue
            tokens = bucket["state_tokens"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            mask = bucket["state_mask"].to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )
            bucket_actions = actions.index_select(0, row_indices)
            with _frontier_autocast_context(device, amp_dtype):
                bucket_embeddings = dynamics.encode_state_action(
                    tokens,
                    mask,
                    bucket_actions,
                )
            encoded.index_copy_(0, row_indices, bucket_embeddings.to(dtype=torch.float32))
    return _clone_if_inference_tensor(encoded.detach())


def _frontier_payload_tensor(
    payload: Mapping[str, Any],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    value = payload[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"Frontier score payload field {key!r} must be a tensor.")
    return value.to(device=device, dtype=dtype, non_blocking=True).reshape(-1)


def _frontier_context_tensors(
    context_payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    class_indices = context_payload.get("class_indices")
    bonus_by_position = context_payload.get("bonus_by_position")
    if not isinstance(class_indices, Tensor) or not isinstance(bonus_by_position, Tensor):
        return (
            torch.zeros((0,), device=device, dtype=torch.long),
            torch.zeros((0,), device=device, dtype=torch.float32),
        )
    return (
        class_indices.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1),
        bonus_by_position.to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1),
    )


def _frontier_heap_push(
    heaps: Dict[int, List[Tuple[Tuple[float, float, float, int], int, Dict[str, Any]]]],
    *,
    budgets: Mapping[int, int],
    item: Dict[str, Any],
) -> None:
    world_index = int(item["world_index"])
    budget = max(0, int(budgets.get(world_index, 0)))
    if budget <= 0:
        return
    order = int(item["order"])
    priority = (
        float(item["score"]),
        float(item["rz_pred"]),
        float(item["rh"]),
        -int(order),
    )
    heap = heaps.setdefault(world_index, [])
    heap_entry = (priority, int(order), item)
    if len(heap) < budget:
        heapq.heappush(heap, heap_entry)
    elif priority > heap[0][0]:
        heapq.heapreplace(heap, heap_entry)


def _ddp_score_frontier_payload(
    *,
    dynamics: ContrastiveDynamicsNetwork,
    payload: Mapping[str, Any],
    sample_tau: Tensor,
    context_payload: Mapping[str, Any],
    score_config: Mapping[str, Any],
    budgets: Mapping[int, int],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Tuple[List[Dict[str, Any]], int]:
    candidate_actions = _frontier_payload_tensor(
        payload,
        "candidate_actions",
        device=device,
        dtype=torch.long,
    )
    candidate_count = int(candidate_actions.numel())
    if candidate_count <= 0:
        return [], 0

    state_tokens_value = payload["state_tokens"]
    state_mask_value = payload["state_mask"]
    if not isinstance(state_tokens_value, Tensor) or not isinstance(state_mask_value, Tensor):
        raise TypeError("Frontier score payload state tensors must be tensors.")
    state_tokens = state_tokens_value.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    state_mask = state_mask_value.to(
        device=device,
        dtype=torch.bool,
        non_blocking=True,
    )
    candidate_state_rows = _frontier_payload_tensor(
        payload,
        "candidate_state_rows",
        device=device,
        dtype=torch.long,
    )
    if int(candidate_state_rows.numel()) != candidate_count:
        raise ValueError("Frontier candidate row/action counts do not match.")

    world_indices = payload["candidate_world_indices"]
    state_ids = payload["candidate_state_ids"]
    orders = payload["candidate_orders"]
    if not all(isinstance(tensor, Tensor) for tensor in (world_indices, state_ids, orders)):
        raise TypeError("Frontier candidate metadata must be tensors.")
    world_list = [
        int(value)
        for value in world_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    ]
    state_id_list = [
        int(value)
        for value in state_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    ]
    order_list = [
        int(value)
        for value in orders.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    ]
    action_list = [
        int(value)
        for value in candidate_actions.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    ]
    state_keys = payload.get("candidate_state_keys")
    if not isinstance(state_keys, list):
        raise TypeError("Frontier candidate_state_keys must be a list.")
    if not (
        len(world_list)
        == len(state_id_list)
        == len(order_list)
        == len(action_list)
        == len(state_keys)
        == candidate_count
    ):
        raise ValueError("Frontier candidate metadata lengths do not match.")

    class_indices, bonus_by_position = _frontier_context_tensors(
        context_payload,
        device=device,
    )
    state_batch_size = max(1, int(score_config["state_batch_size"]))
    candidate_chunk_size = max(1, int(score_config["candidate_chunk_size"]))
    heaps: Dict[int, List[Tuple[Tuple[float, float, float, int], int, Dict[str, Any]]]] = {}

    with torch.inference_mode():
        state_count = int(state_tokens.size(0))
        for state_start in range(0, state_count, state_batch_size):
            state_end = min(state_count, state_start + state_batch_size)
            row_mask = (candidate_state_rows >= int(state_start)) & (
                candidate_state_rows < int(state_end)
            )
            candidate_indices = torch.nonzero(row_mask, as_tuple=False).reshape(-1)
            if int(candidate_indices.numel()) <= 0:
                continue
            batch_tokens = state_tokens[state_start:state_end]
            batch_mask = state_mask[state_start:state_end]
            with _frontier_autocast_context(device, amp_dtype):
                state_context = dynamics.encode_state_context(batch_tokens, batch_mask)
            for chunk_start in range(0, int(candidate_indices.numel()), candidate_chunk_size):
                chunk_end = min(int(candidate_indices.numel()), chunk_start + candidate_chunk_size)
                chunk_indices = candidate_indices[chunk_start:chunk_end]
                chunk_rows = candidate_state_rows.index_select(0, chunk_indices) - int(state_start)
                chunk_actions = candidate_actions.index_select(0, chunk_indices)
                chunk_mask = batch_mask.index_select(0, chunk_rows)
                with _frontier_autocast_context(device, amp_dtype):
                    tau_embeddings = dynamics.encode_state_action_from_context(
                        state_context.index_select(0, chunk_rows),
                        chunk_mask,
                        chunk_actions,
                    )
                rh, rz, rtotal = _frontier_score_tau_embeddings(
                    dynamics=dynamics,
                    tau_embeddings=tau_embeddings,
                    sample_tau=sample_tau,
                    class_indices=class_indices,
                    bonus_by_position=bonus_by_position,
                    score_config=score_config,
                )
                score_rows = (
                    torch.stack((rh.reshape(-1), rz.reshape(-1), rtotal.reshape(-1)), dim=1)
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .tolist()
                )
                chunk_index_list = [
                    int(index)
                    for index in chunk_indices.detach().to(device="cpu", dtype=torch.long).tolist()
                ]
                for offset, candidate_index in enumerate(chunk_index_list):
                    rh_value, rz_value, score_value = score_rows[int(offset)]
                    item = {
                        "world_index": int(world_list[int(candidate_index)]),
                        "state_id": int(state_id_list[int(candidate_index)]),
                        "state_key": str(state_keys[int(candidate_index)]),
                        "action": int(action_list[int(candidate_index)]),
                        "rh": float(rh_value),
                        "rz_pred": float(rz_value),
                        "score": float(score_value),
                        "order": int(order_list[int(candidate_index)]),
                    }
                    _frontier_heap_push(heaps, budgets=budgets, item=item)

    items: List[Dict[str, Any]] = []
    for heap in heaps.values():
        items.extend(
            heap_entry[2]
            for heap_entry in sorted(heap, key=lambda heap_entry: heap_entry[0], reverse=True)
        )
    return items, candidate_count


def _ddp_run_frontier_score(
    *,
    ddp_model: nn.parallel.DistributedDataParallel,
    command: Mapping[str, Any],
    rank: int,
    device: torch.device,
    model_config: Mapping[str, Any],
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    sample_tau_tensor = command["sample_tau"]
    if not isinstance(sample_tau_tensor, Tensor):
        raise TypeError("Distributed frontier score requires tensor sample_tau.")
    sample_tau = sample_tau_tensor.detach().to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    raw_budgets = command.get("budgets") or {}
    if not isinstance(raw_budgets, Mapping):
        raise TypeError("Distributed frontier score budgets must be a mapping.")
    budgets = {int(key): int(value) for key, value in raw_budgets.items()}
    raw_payload = command.get("frontier_payload") or {}
    if not isinstance(raw_payload, Mapping):
        raise TypeError("Distributed frontier score payload must be a mapping.")
    raw_context = command.get("context") or {}
    if not isinstance(raw_context, Mapping):
        raise TypeError("Distributed frontier score context must be a mapping.")
    raw_score_config = command.get("score_config") or {}
    if not isinstance(raw_score_config, Mapping):
        raise TypeError("Distributed frontier score config must be a mapping.")
    items, scored_count = _ddp_score_frontier_payload(
        dynamics=ddp_model.module,
        payload=raw_payload,
        sample_tau=sample_tau,
        context_payload=raw_context,
        score_config=raw_score_config,
        budgets=budgets,
        device=device,
        amp_dtype=amp_dtype,
    )
    return {
        "op": "score_frontier",
        "request_id": int(command["request_id"]),
        "rank": int(rank),
        "scored_count": int(scored_count),
        "items": items,
    }



def _ddp_run_sample_tau_step_command(
    *,
    ddp_model: nn.parallel.DistributedDataParallel,
    command: Mapping[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
    model_config: Mapping[str, Any],
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    del world_size
    raw_sample_command = command.get("sample_command") or {}
    if not isinstance(raw_sample_command, Mapping):
        raise TypeError("Distributed sample tau command requires sample_command mapping.")
    local_tau = _ddp_encode_sample_step_command(
        dynamics=ddp_model.module,
        command=raw_sample_command,
        device=device,
        model_config=model_config,
        amp_dtype=amp_dtype,
    )
    encoded_tau = _share_cpu_tensor_if_requested(
        _clone_if_inference_tensor(
            local_tau.detach().to(device="cpu", dtype=torch.float32).contiguous()
        ),
        share_memory=_distributed_step_command_should_share_tensors(),
    )
    result: Dict[str, Any] = {
        "op": "encode_sample_tau_step_command",
        "request_id": int(command["request_id"]),
        "rank": int(rank),
        "encoded_count": int(encoded_tau.size(0)),
        "tau": encoded_tau,
    }
    return result


def _ddp_contrastive_worker(
    rank: int,
    world_size: int,
    device_ids: Sequence[int],
    master_addr: str,
    master_port: int,
    model_config: Dict[str, Any],
    initial_state_dict: Dict[str, Tensor],
    optimizer_state_dict: Optional[Dict[str, Any]],
    scaler_state_dict: Optional[Dict[str, Any]],
    contrastive_lr: float,
    amp_dtype_name: Optional[str],
    use_grad_scaler: bool,
    command_queues: Sequence[Any],
    result_queue: Any,
) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(int(master_port))
    os.environ["RANK"] = str(int(rank))
    os.environ["WORLD_SIZE"] = str(int(world_size))

    device_id = int(device_ids[int(rank)])
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    dist.init_process_group(
        backend="nccl",
        rank=int(rank),
        world_size=int(world_size),
        timeout=timedelta(seconds=1800),
    )
    amp_dtype = _amp_dtype_from_name(amp_dtype_name)
    dynamics = _build_worker_dynamics(
        model_config=model_config,
        device=device,
        forward_amp_dtype=amp_dtype,
        forward_use_autocast=amp_dtype is not None,
    )
    dynamics.load_state_dict(initial_state_dict, strict=True)
    ddp_model = nn.parallel.DistributedDataParallel(
        dynamics,
        device_ids=[device_id],
        output_device=device_id,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    optimizer = torch.optim.AdamW(ddp_model.module.parameters(), lr=float(contrastive_lr))
    if isinstance(optimizer_state_dict, dict):
        optimizer.load_state_dict(optimizer_state_dict)
        _move_optimizer_state_to_device(optimizer, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_grad_scaler))
    if bool(use_grad_scaler) and isinstance(scaler_state_dict, dict):
        scaler.load_state_dict(scaler_state_dict)

    command_queue = command_queues[int(rank)]
    try:
        while True:
            command = command_queue.get()
            op = str(command.get("op", ""))
            if op == "shutdown":
                break
            if op == "load_state":
                ddp_model.module.load_state_dict(command["state_dict"], strict=True)
                request_id = command.get("request_id")
                if request_id is not None:
                    result_queue.put(
                        {
                            "op": "load_state",
                            "request_id": int(request_id),
                            "rank": int(rank),
                        }
                    )
                continue
            if op == "expand_prototype_capacity":
                required_count = max(1, int(command["required_count"]))
                module = ddp_model.module
                changed = module.ensure_prototype_capacity(required_count)
                if changed:
                    # Re-wrapping the SAME module in DDP keeps the compiled leaf
                    # handles alive - do not re-install them here, that would
                    # throw away the warm graphs and pay cold compile again.
                    # prototype_table is not read by either leaf, so growing it
                    # cannot invalidate them either.
                    ddp_model = nn.parallel.DistributedDataParallel(
                        module,
                        device_ids=[device_id],
                        output_device=device_id,
                        gradient_as_bucket_view=True,
                        broadcast_buffers=False,
                    )
                model_config["num_dynamics_classes"] = max(
                    int(model_config["num_dynamics_classes"]),
                    int(module.num_classes),
                )
                result_queue.put(
                    {
                        "op": "expand_prototype_capacity",
                        "request_id": int(command["request_id"]),
                        "rank": int(rank),
                        "num_classes": int(module.num_classes),
                        "changed": bool(changed),
                    }
                )
                continue
            if op == "score_frontier":
                result = _ddp_run_frontier_score(
                    ddp_model=ddp_model,
                    command=command,
                    rank=int(rank),
                    device=device,
                    model_config=model_config,
                    amp_dtype=amp_dtype,
                )
                _synchronize_cuda_device(device)
                result_queue.put(result)
                continue
            if op == "encode_sample_tau_step_command":
                result = _ddp_run_sample_tau_step_command(
                    ddp_model=ddp_model,
                    command=command,
                    rank=int(rank),
                    world_size=int(world_size),
                    device=device,
                    model_config=model_config,
                    amp_dtype=amp_dtype,
                )
                _synchronize_cuda_device(device)
                result_queue.put(result)
                continue
            if op == "train_many_step_commands":
                request_id = int(command["request_id"])
                step_commands = [
                    dict(step)
                    for step in (command.get("steps") or [])
                    if isinstance(step, dict)
                ]
                step_stats = _ddp_run_contrastive_train_steps(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step_commands=step_commands,
                    rank=int(rank),
                    world_size=int(world_size),
                    device=device,
                    model_config=model_config,
                    amp_dtype=amp_dtype,
                    use_grad_scaler=bool(use_grad_scaler),
                )
                _synchronize_cuda_device(device)
                result_queue.put(
                    {
                        "request_id": request_id,
                        "rank": int(rank),
                        "updates_completed": int(len(step_stats)),
                        "step_stats": step_stats,
                        "stats": _ddp_average_step_stats(step_stats),
                        # Reported after the steps ran, so a first-call compile
                        # failure inside this child is already latched here.
                        # Ranks can disagree without breaking the math - the
                        # kernels are semantically identical - but a silent
                        # per-rank disagreement would show up only as a
                        # mysteriously smaller speedup.
                        "compile_enabled": bool(ddp_model.module.compile_enabled),
                        "compile_disabled_reason": ddp_model.module.compile_disabled_reason,
                    }
                )
                continue
    finally:
        dist.destroy_process_group()


class DistributedContrastiveTrainer:
    def __init__(
        self,
        *,
        device_ids: Sequence[int],
        model_config: Dict[str, Any],
        rank0_model: ContrastiveDynamicsNetwork,
        rank0_optimizer: torch.optim.Optimizer,
        rank0_scaler: torch.amp.GradScaler,
        contrastive_lr: float,
        amp_dtype: Optional[torch.dtype],
        use_grad_scaler: bool,
    ) -> None:
        import torch.distributed as dist

        self.device_ids = tuple(int(device_id) for device_id in device_ids)
        self.world_size = int(len(self.device_ids))
        self._request_id = 0
        self._closed = False
        self._rank0_device_id = int(self.device_ids[0])
        self._rank0_model = rank0_model
        self._rank0_optimizer = rank0_optimizer
        self._rank0_scaler = rank0_scaler
        self._rank0_ddp_model: Optional[nn.parallel.DistributedDataParallel] = None
        self.worker_compile_disabled_reasons: Dict[int, str] = {}
        self._dist = dist
        self._model_config = dict(model_config)
        self._amp_dtype = amp_dtype
        self._use_grad_scaler = bool(use_grad_scaler)
        self._previous_dist_env = {
            name: os.environ.get(name)
            for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        ctx = torch_mp.get_context("spawn")
        self._command_queues = [ctx.Queue(maxsize=2) for _ in range(self.world_size)]
        self._result_queue = ctx.Queue(maxsize=max(1, self.world_size * 2))
        master_port = _find_free_loopback_port()
        initial_state_dict = _tensor_tree_to_cpu(rank0_model.state_dict())
        optimizer_state_dict = _tensor_tree_to_cpu(rank0_optimizer.state_dict())
        scaler_state_dict = (
            _tensor_tree_to_cpu(rank0_scaler.state_dict())
            if bool(use_grad_scaler)
            else None
        )
        self._processes = []
        for rank in range(1, self.world_size):
            process = ctx.Process(
                target=_ddp_contrastive_worker,
                args=(
                    int(rank),
                    self.world_size,
                    self.device_ids,
                    "127.0.0.1",
                    int(master_port),
                    dict(model_config),
                    initial_state_dict,
                    optimizer_state_dict,
                    scaler_state_dict,
                    float(contrastive_lr),
                    _amp_dtype_name(amp_dtype),
                    bool(use_grad_scaler),
                    self._command_queues,
                    self._result_queue,
                ),
            )
            process.daemon = True
            process.start()
            self._processes.append(process)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(int(master_port))
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(int(self.world_size))
        torch.cuda.set_device(self._rank0_device_id)
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=int(self.world_size),
            timeout=timedelta(seconds=1800),
        )
        self._rank0_ddp_model = nn.parallel.DistributedDataParallel(
            self._rank0_model,
            device_ids=[self._rank0_device_id],
            output_device=self._rank0_device_id,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rank in range(1, self.world_size):
            process = self._processes[int(rank) - 1]
            if process.exitcode is not None:
                continue
            try:
                self._command_queues[int(rank)].put({"op": "shutdown"}, timeout=1.0)
            except (Full, OSError, ValueError):
                pass
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        self._rank0_ddp_model = None
        if self._dist.is_initialized():
            self._dist.destroy_process_group()
        for name, previous_value in self._previous_dist_env.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        request_id = self._next_request_id()
        payload = _tensor_tree_to_cpu(state_dict)
        for rank in range(1, self.world_size):
            self._command_queues[int(rank)].put(
                {
                    "op": "load_state",
                    "request_id": int(request_id),
                    "state_dict": payload,
                }
            )
        completed_ranks: set[int] = set()
        while len(completed_ranks) < max(0, self.world_size - 1):
            result = self._get_child_result()
            if str(result.get("op", "")) != "load_state":
                continue
            if int(result.get("request_id", -1)) != int(request_id):
                continue
            rank = int(result.get("rank", -1))
            if rank > 0:
                completed_ranks.add(rank)

    def expand_prototype_capacity(self, required_count: int) -> None:
        safe_required_count = max(1, int(required_count))
        self._model_config["num_dynamics_classes"] = max(
            int(self._model_config["num_dynamics_classes"]),
            int(safe_required_count),
        )
        request_id = self._next_request_id()
        for rank in range(1, self.world_size):
            self._command_queues[int(rank)].put(
                {
                    "op": "expand_prototype_capacity",
                    "request_id": int(request_id),
                    "required_count": int(safe_required_count),
                }
            )
        if self._rank0_ddp_model is not None:
            module = self._rank0_ddp_model.module
            module.ensure_prototype_capacity(safe_required_count)
            # Same module object, so the compiled leaf handles survive the
            # re-wrap. Do not re-install compile here.
            self._rank0_ddp_model = nn.parallel.DistributedDataParallel(
                module,
                device_ids=[self._rank0_device_id],
                output_device=self._rank0_device_id,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )
        completed_ranks: set[int] = set()
        while len(completed_ranks) < max(0, self.world_size - 1):
            result = self._get_child_result()
            if str(result.get("op", "")) != "expand_prototype_capacity":
                continue
            if int(result.get("request_id", -1)) != int(request_id):
                continue
            rank = int(result.get("rank", -1))
            if rank <= 0:
                continue
            child_num_classes = int(result.get("num_classes", 0))
            if child_num_classes < safe_required_count:
                raise RuntimeError(
                    "Distributed worker failed to expand prototype capacity: "
                    f"rank={rank} required={safe_required_count} got={child_num_classes}."
                )
            completed_ranks.add(rank)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return int(self._request_id)

    def _raise_if_child_process_exited(self) -> None:
        for process in self._processes:
            if process.exitcode is not None:
                raise RuntimeError(
                    "Distributed contrastive worker exited before completing "
                    f"the request: pid={process.pid} exitcode={process.exitcode}."
                )

    def _get_child_result(self) -> Dict[str, Any]:
        while True:
            try:
                result = self._result_queue.get(timeout=1.0)
            except Empty:
                self._raise_if_child_process_exited()
                continue
            if not isinstance(result, dict):
                raise RuntimeError("Distributed contrastive worker returned a non-dict result.")
            return result

    def _send_child_train_step_commands(
        self,
        *,
        request_id: int,
        rank_step_commands: Sequence[Sequence[Dict[str, Any]]],
    ) -> None:
        for rank in range(1, self.world_size):
            self._command_queues[int(rank)].put(
                {
                    "op": "train_many_step_commands",
                    "request_id": int(request_id),
                    "steps": [
                        dict(step)
                        for step in rank_step_commands[int(rank)]
                    ],
                }
            )

    def _record_worker_compile_state(self, result: Mapping[str, Any]) -> None:
        rank = int(result.get("rank", -1))
        if rank <= 0:
            return
        reason = result.get("compile_disabled_reason")
        if isinstance(reason, str) and reason:
            self.worker_compile_disabled_reasons[rank] = reason
        else:
            self.worker_compile_disabled_reasons.pop(rank, None)

    def _wait_for_child_train_steps(self, request_id: int) -> None:
        completed_ranks: set[int] = set()
        while len(completed_ranks) < max(0, self.world_size - 1):
            result = self._get_child_result()
            if int(result.get("request_id", -1)) != int(request_id):
                continue
            rank = int(result.get("rank", -1))
            if rank > 0:
                completed_ranks.add(rank)
                self._record_worker_compile_state(result)

    def _wait_for_child_frontier_scores(self, request_id: int) -> List[Dict[str, Any]]:
        completed_ranks: set[int] = set()
        results: List[Dict[str, Any]] = []
        while len(completed_ranks) < max(0, self.world_size - 1):
            result = self._get_child_result()
            if str(result.get("op", "")) != "score_frontier":
                continue
            if int(result.get("request_id", -1)) != int(request_id):
                continue
            rank = int(result.get("rank", -1))
            if rank <= 0:
                continue
            completed_ranks.add(rank)
            results.append(result)
        return results

    def _sample_tau_step_command_common(
        self,
        *,
        request_id: int,
        sample_command: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "op": "encode_sample_tau_step_command",
            "request_id": int(request_id),
            "sample_command": dict(sample_command),
        }

    def _send_child_sample_tau_step_commands(
        self,
        *,
        request_id: int,
        rank_sample_commands: Sequence[Dict[str, Any]],
    ) -> None:
        for rank in range(1, self.world_size):
            self._command_queues[int(rank)].put(
                self._sample_tau_step_command_common(
                    request_id=int(request_id),
                    sample_command=dict(rank_sample_commands[int(rank)]),
                )
            )

    def _run_rank0_sample_tau_step_command(
        self,
        *,
        request_id: int,
        rank_sample_commands: Sequence[Dict[str, Any]],
        model_config: Dict[str, Any],
    ) -> Tensor:
        if self._rank0_ddp_model is None:
            raise RuntimeError("Distributed contrastive trainer is closed.")
        rank0_result = _ddp_run_sample_tau_step_command(
            ddp_model=self._rank0_ddp_model,
            command=self._sample_tau_step_command_common(
                request_id=int(request_id),
                sample_command=dict(rank_sample_commands[0]),
            ),
            rank=0,
            world_size=int(self.world_size),
            device=torch.device(f"cuda:{self._rank0_device_id}"),
            model_config=model_config,
            amp_dtype=getattr(self, "_amp_dtype", None),
        )
        _synchronize_cuda_device(torch.device(f"cuda:{self._rank0_device_id}"))
        child_results = self._wait_for_child_sample_tau_step_commands(int(request_id))
        tau_by_rank: Dict[int, Tensor] = {}
        for result in (rank0_result, *child_results):
            rank = int(result.get("rank", -1))
            encoded = result.get("tau")
            if rank < 0 or not isinstance(encoded, Tensor):
                raise RuntimeError(
                    "Distributed sample tau encode did not return a tensor for every rank."
                )
            tau_by_rank[int(rank)] = encoded.detach().to(device="cpu", dtype=torch.float32)
        missing_ranks = [
            int(rank)
            for rank in range(int(self.world_size))
            if int(rank) not in tau_by_rank
        ]
        if missing_ranks:
            raise RuntimeError(
                "Distributed sample tau encode missed rank results: "
                f"{missing_ranks}."
            )
        encoded = torch.cat(
            [tau_by_rank[int(rank)] for rank in range(int(self.world_size))],
            dim=0,
        )
        expected_count = sum(
            int(command.get("actions").numel())
            for command in rank_sample_commands
            if isinstance(command.get("actions"), Tensor)
        )
        if int(encoded.size(0)) != int(expected_count):
            raise RuntimeError(
                "Distributed sample tau encode returned the wrong row count: "
                f"expected={expected_count} got={int(encoded.size(0))}."
            )
        return encoded

    def _wait_for_child_sample_tau_step_commands(self, request_id: int) -> List[Dict[str, Any]]:
        completed_ranks: set[int] = set()
        results: List[Dict[str, Any]] = []
        while len(completed_ranks) < max(0, self.world_size - 1):
            result = self._get_child_result()
            if str(result.get("op", "")) != "encode_sample_tau_step_command":
                continue
            if int(result.get("request_id", -1)) != int(request_id):
                continue
            rank = int(result.get("rank", -1))
            if rank <= 0:
                continue
            completed_ranks.add(rank)
            results.append(result)
        return results

    def _frontier_score_common_command(
        self,
        *,
        request_id: int,
        sample_tau: Tensor,
        context: Dict[str, Any],
        score_config: Dict[str, Any],
        budgets: Mapping[int, int],
    ) -> Dict[str, Any]:
        replay_tau = sample_tau.detach()
        if replay_tau.device.type != "cpu":
            replay_tau = replay_tau.to(device="cpu", dtype=torch.float32)
        else:
            replay_tau = replay_tau.to(dtype=torch.float32)
        replay_tau = _share_cpu_tensor_if_requested(
            replay_tau.contiguous(),
            share_memory=_distributed_step_command_should_share_tensors(),
        )
        return {
            "op": "score_frontier",
            "request_id": int(request_id),
            "sample_tau": replay_tau,
            "context": dict(context),
            "score_config": dict(score_config),
            "budgets": {int(key): int(value) for key, value in budgets.items()},
        }

    def _send_child_frontier_score(
        self,
        *,
        request_id: int,
        sample_tau: Tensor,
        rank_payloads: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        score_config: Dict[str, Any],
        budgets: Mapping[int, int],
    ) -> None:
        common = self._frontier_score_common_command(
            request_id=int(request_id),
            sample_tau=sample_tau,
            context=context,
            score_config=score_config,
            budgets=budgets,
        )
        for rank in range(1, self.world_size):
            self._command_queues[int(rank)].put(
                {
                    **common,
                    "frontier_payload": dict(rank_payloads[int(rank)]),
                }
            )

    def _run_rank0_frontier_score(
        self,
        *,
        request_id: int,
        sample_tau: Tensor,
        rank_payloads: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        score_config: Dict[str, Any],
        budgets: Mapping[int, int],
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._rank0_ddp_model is None:
            raise RuntimeError("Distributed contrastive trainer is closed.")
        rank0_command = {
            **self._frontier_score_common_command(
                request_id=int(request_id),
                sample_tau=sample_tau,
                context=context,
                score_config=score_config,
                budgets=budgets,
            ),
            "frontier_payload": dict(rank_payloads[0]),
        }
        rank0_result = _ddp_run_frontier_score(
            ddp_model=self._rank0_ddp_model,
            command=rank0_command,
            rank=0,
            device=torch.device(f"cuda:{self._rank0_device_id}"),
            model_config=model_config,
            amp_dtype=getattr(self, "_amp_dtype", None),
        )
        _synchronize_cuda_device(torch.device(f"cuda:{self._rank0_device_id}"))
        child_results = self._wait_for_child_frontier_scores(int(request_id))
        items: List[Dict[str, Any]] = []
        scored_count = int(rank0_result.get("scored_count", 0))
        raw_items = rank0_result.get("items") or []
        if isinstance(raw_items, list):
            items.extend(item for item in raw_items if isinstance(item, dict))
        for result in child_results:
            scored_count += int(result.get("scored_count", 0))
            result_items = result.get("items") or []
            if isinstance(result_items, list):
                items.extend(item for item in result_items if isinstance(item, dict))
        return {
            "request_id": int(request_id),
            "items": items,
            "scored_count": int(scored_count),
        }

    def _run_rank0_train_step_commands(
        self,
        *,
        request_id: int,
        rank_step_commands: Sequence[Sequence[Dict[str, Any]]],
        amp_dtype: Optional[torch.dtype],
        use_grad_scaler: bool,
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._rank0_ddp_model is None:
            raise RuntimeError("Distributed contrastive trainer is closed.")
        step_commands = [dict(step) for step in rank_step_commands[0]]
        step_stats = _ddp_run_contrastive_train_steps(
            ddp_model=self._rank0_ddp_model,
            optimizer=self._rank0_optimizer,
            scaler=self._rank0_scaler,
            step_commands=step_commands,
            rank=0,
            world_size=int(self.world_size),
            device=torch.device(f"cuda:{self._rank0_device_id}"),
            model_config=model_config,
            amp_dtype=amp_dtype,
            use_grad_scaler=bool(use_grad_scaler),
        )
        _synchronize_cuda_device(torch.device(f"cuda:{self._rank0_device_id}"))
        self._wait_for_child_train_steps(int(request_id))
        return {
            "request_id": int(request_id),
            "rank": 0,
            "updates_completed": int(len(step_stats)),
            "step_stats": step_stats,
            "stats": _ddp_average_step_stats(step_stats),
        }

    def train_step_commands(
        self,
        rank_step_commands: Sequence[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if len(rank_step_commands) != self.world_size:
            raise ValueError("rank_step_commands must match DDP world size.")
        step_count = len(rank_step_commands[0]) if rank_step_commands else 0
        if step_count <= 0:
            raise ValueError("rank_step_commands must contain at least one train step.")
        for rank_commands in rank_step_commands:
            if len(rank_commands) != step_count:
                raise ValueError("each rank must receive the same number of train steps.")
        request_id = self._next_request_id()
        model_config = dict(getattr(self, "_model_config", {}) or {})
        if not model_config:
            raise RuntimeError("Distributed contrastive trainer is missing model_config.")
        self._send_child_train_step_commands(
            request_id=int(request_id),
            rank_step_commands=rank_step_commands,
        )
        return self._run_rank0_train_step_commands(
            request_id=int(request_id),
            rank_step_commands=rank_step_commands,
            amp_dtype=getattr(self, "_amp_dtype", None),
            use_grad_scaler=bool(getattr(self, "_use_grad_scaler", False)),
            model_config=model_config,
        )

    def encode_sample_tau_step_commands(
        self,
        rank_sample_commands: Sequence[Dict[str, Any]],
    ) -> Tensor:
        if len(rank_sample_commands) != self.world_size:
            raise ValueError("rank_sample_commands must match DDP world size.")
        request_id = self._next_request_id()
        model_config = dict(getattr(self, "_model_config", {}) or {})
        if not model_config:
            raise RuntimeError("Distributed contrastive trainer is missing model_config.")
        self._send_child_sample_tau_step_commands(
            request_id=int(request_id),
            rank_sample_commands=rank_sample_commands,
        )
        return self._run_rank0_sample_tau_step_command(
            request_id=int(request_id),
            rank_sample_commands=rank_sample_commands,
            model_config=model_config,
        )

    def score_frontier(
        self,
        *,
        sample_tau: Tensor,
        rank_payloads: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
        score_config: Dict[str, Any],
        budgets: Mapping[int, int],
    ) -> Dict[str, Any]:
        if len(rank_payloads) != self.world_size:
            raise ValueError("rank_payloads must match DDP world size.")
        request_id = self._next_request_id()
        model_config = dict(getattr(self, "_model_config", {}) or {})
        if not model_config:
            raise RuntimeError("Distributed contrastive trainer is missing model_config.")
        self._send_child_frontier_score(
            request_id=int(request_id),
            sample_tau=sample_tau,
            rank_payloads=rank_payloads,
            context=context,
            score_config=score_config,
            budgets=budgets,
        )
        return self._run_rank0_frontier_score(
            request_id=int(request_id),
            sample_tau=sample_tau,
            rank_payloads=rank_payloads,
            context=context,
            score_config=score_config,
            budgets=budgets,
            model_config=model_config,
        )


class GraphContrastiveBase(BaseExplorer):
    strategy_name = "graph_contrastive"
    _SUPPORTED_CONTRASTIVE_TRAIN_MODES = (
        "sample_batch",
        "full_sample_epoch",
    )
    _SUPPORTED_CONTRASTIVE_REPRESENTATION_MODES = (
        "trained",
        "frozen_initial",
    )
    _SUPPORTED_ENCODE_BUCKET_MODES = (
        "none",
        "token_budget",
    )
    # Rows are sorted by entity count before bucketing, so buckets are near-uniform
    # and most of them sit at this cap. Lowering it trades kernel-launch count
    # against per-bucket activation memory; raising it increases activation memory.
    _ENCODE_BUCKET_MAX_PADDED_TOKENS = 131_072
    # Entity counts are multi-modal, so this ratio decides how many distinct map sizes
    # share one padded bucket. It trades padded-away compute against per-bucket kernel
    # launch overhead; padding is masked out, so it cannot change what is computed
    # (bf16 rounding only). Tuned for 8-GPU DDP with a global batch of 4096, i.e. each
    # rank buckets a 512-sample shard.
    _ENCODE_BUCKET_MAX_PADDING_RATIO = 1.5
    # Rows per tau encode step command. NOT a batch-size tuning knob: it bounds the
    # host memory one command pins, because `_sample_step_command_from_indices` packs
    # and pins every bucket of a command before returning. The uniform-bucket fast path
    # in `_pack_encoded_state_batch_to_host` allocates a second full-size buffer for
    # `pin_memory()`, so the transient peak is about twice the committed size.
    _TAU_REPLAY_ENCODE_MAX_SAMPLES_PER_BATCH = 1024
    _DISTRIBUTED_TAU_ENCODE_MAX_SAMPLES_PER_CHUNK = 8_192
    # Fraction of the state-token cache dropped per eviction once the configured
    # bound is exceeded. Batching the drop amortises the O(n) key scan and stops
    # the cache from evicting on every insert while it sits on the limit.
    _STATE_TOKEN_CACHE_EVICT_PERCENT = 10
    _RUNTIME_VERSION_FIELDS = (
        "sample_storage_version",
        "sample_label_version",
        "dynamics_parameter_version",
        "prototype_summary_version",
    )

    def __init__(
        self,
        env: Any,
        seed: int = 42,
        device: str = "auto",
        dynamics_encoder_embed_dim: int = 128,
        dynamics_encoder_num_heads: int = 4,
        dynamics_encoder_num_blocks: int = 2,
        dynamics_pool_seeds: int = 1,
        contrastive_batch_size: int = 64,
        contrastive_train_mode: str = "sample_batch",
        contrastive_representation_mode: str = "trained",
        train_updates_per_step: int = 1,
        learning_rate: float = 3e-4,
        contrastive_lr: Optional[float] = None,
        contrastive_dim: Optional[int] = None,
        max_prototypes_per_class: int = 1,
        prototype_split_base_count: int = 64,
        prototype_split_min_cluster_occupancy: int = 16,
        prototype_sample_cap: Optional[int] = None,
        prototype_sample_min_per_class: Optional[int] = None,
        contrastive_temperature: Optional[float] = None,
        contrastive_min_class_count: int = 1,
        intrinsic_reward_scale: float = 1.0,
        prototype_entropy_scale: float = 0.0,
        prototype_entropy_temperature: Optional[float] = None,
        prototype_entropy_min_class_count: Optional[int] = None,
        prototype_entropy_eps: float = 1e-6,
        knn_k: int = 16,
        knn_avg: bool = True,
        knn_clip: float = 5e-4,
        knn_exclude_self: bool = True,
        dynamics_encoder_dropout: float = 0.0,
        evaluator: Optional[ProgramEvaluator] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        contrastive_checkpoint_path: Optional[str] = None,
        encode_bucket_mode: str = "token_budget",
        train_phase_timing: bool = False,
        train_phase_timing_cuda_sync: bool = False,
        state_token_cache_max_entries: int = 0,
        compile_encoder: bool = False,
    ):
        self.env = env
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        # PyTorch module constructors draw initial weights from the global RNG.
        torch.manual_seed(self.seed)
        self.device = self._resolve_device(device)
        _configure_torch_cuda_backends(self.device)
        self.visible_cuda_device_count = (
            int(torch.cuda.device_count()) if self.device.type == "cuda" else 0
        )
        self.parallel_device_ids = _resolve_parallel_cuda_device_ids(
            device=self.device,
            visible_cuda_device_count=self.visible_cuda_device_count,
        )
        self.use_mixed_precision = bool(self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype()
        self._use_grad_scaler = bool(
            self.use_mixed_precision
            and self.amp_dtype == torch.float16
            and self.device.type == "cuda"
        )

        self.dynamics_encoder_embed_dim = max(16, int(dynamics_encoder_embed_dim))
        self.dynamics_encoder_num_heads = max(1, int(dynamics_encoder_num_heads))
        self.dynamics_encoder_num_blocks = max(1, int(dynamics_encoder_num_blocks))
        self.dynamics_pool_seeds = max(1, int(dynamics_pool_seeds))
        self.contrastive_batch_size = max(1, int(contrastive_batch_size))
        self.contrastive_train_mode = self._resolve_contrastive_train_mode(
            contrastive_train_mode
        )
        self.contrastive_representation_mode = (
            self._resolve_contrastive_representation_mode(
                contrastive_representation_mode
            )
        )
        self.encode_bucket_mode = self._resolve_encode_bucket_mode(encode_bucket_mode)
        self.compile_encoder = bool(compile_encoder)
        self._train_phase_timer = TrainPhaseTimer(
            enabled=bool(train_phase_timing),
            cuda_sync=bool(train_phase_timing_cuda_sync),
            device=self.device,
        )
        self._pin_host_tensor_calls = 0
        self._pin_host_tensor_seconds = 0.0
        self._last_train_phase_timing: Dict[str, Any] = {}
        self.state_token_cache_max_entries = max(0, int(state_token_cache_max_entries))
        # Eviction victims are drawn from a private stream so that bounding the
        # cache cannot perturb `self.rng`, which drives sampling order.
        self._state_token_cache_rng = random.Random(self.seed ^ 0x5CA1AB1E)
        self.train_updates_per_step = max(1, int(train_updates_per_step))
        self.learning_rate = float(learning_rate)
        self.contrastive_lr = float(contrastive_lr) if contrastive_lr is not None else float(self.learning_rate)
        self.num_dynamics_classes = 1
        self.max_prototypes_per_class = max(1, int(max_prototypes_per_class))
        self.prototype_split_base_count = max(2, int(prototype_split_base_count))
        self.prototype_split_min_cluster_occupancy = max(
            2,
            int(prototype_split_min_cluster_occupancy),
        )
        self.prototype_sample_cap = self._resolve_optional_positive_count(
            prototype_sample_cap,
            "prototype_sample_cap",
        )
        self.prototype_sample_min_per_class = (
            None
            if prototype_sample_min_per_class is None
            else max(1, int(prototype_sample_min_per_class))
        )
        requested_temperature = float(contrastive_temperature) if contrastive_temperature is not None else 0.1
        self.contrastive_temperature = max(1e-4, float(requested_temperature))
        self.contrastive_min_class_count = max(1, int(contrastive_min_class_count))
        self.prototype_entropy_scale = max(0.0, float(prototype_entropy_scale))
        requested_prototype_entropy_temperature = (
            float(prototype_entropy_temperature)
            if prototype_entropy_temperature is not None
            else float(self.contrastive_temperature)
        )
        self.prototype_entropy_temperature = max(1e-4, float(requested_prototype_entropy_temperature))
        requested_prototype_entropy_min_class_count = (
            int(prototype_entropy_min_class_count)
            if prototype_entropy_min_class_count is not None
            else max(
                5,
                int(self.contrastive_min_class_count),
                int(self.prototype_sample_min_per_class or 0),
            )
        )
        self.prototype_entropy_min_class_count = max(
            1,
            int(requested_prototype_entropy_min_class_count),
        )
        self.prototype_entropy_eps = max(1e-12, float(prototype_entropy_eps))
        requested_contrastive_dim = (
            self.dynamics_encoder_embed_dim if contrastive_dim is None else int(contrastive_dim)
        )
        self.contrastive_dim = max(8, int(requested_contrastive_dim))
        self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        self.knn_k = max(1, int(knn_k))
        self.knn_avg = bool(knn_avg)
        self.knn_clip = float(knn_clip)
        self.knn_exclude_self = bool(knn_exclude_self)
        self.dynamics_encoder_dropout = max(0.0, float(dynamics_encoder_dropout))
        self.num_actions = int(
            getattr(
                self.env,
                "num_actions",
                getattr(getattr(self.env, "action_space", None), "n", 0),
            )
        )
        if self.num_actions <= 0:
            raise ValueError("GraphContrastiveBase requires env.num_actions > 0.")
        self.action_names = tuple(self.env.get_action_name(index) for index in range(self.num_actions))

        state_projection_config = None
        get_state_projection_config = getattr(self.env, "get_state_projection_config", None)
        if callable(get_state_projection_config):
            state_projection_config = get_state_projection_config()
        else:
            state_projection_config = getattr(self.env, "state_projection_config", None)
        self.codec = EntityTokenCodec(
            state_projection_config=state_projection_config
        )
        self._program_evaluator = evaluator or ProgramEvaluator(
            sandbox_config=sandbox_config or SandboxConfig()
        )
        existing_state_store = getattr(self, "_state_store", None)
        self._state_store = (
            existing_state_store if isinstance(existing_state_store, StateStore) else None
        )
        self._group_classifier = TransitionGroupClassifier(self._program_evaluator)
        self._group_context_snapshot = TransitionGroupContextSnapshot()
        self._active_class_id_set: frozenset = frozenset()
        self._current_program_source: Optional[str] = None
        self._current_source_transition_assessments: Dict[str, Dict[str, Any]] = {}
        self._sync_program_context_snapshot(self._group_context_snapshot)
        self._configured_contrastive_checkpoint_path = (
            str(contrastive_checkpoint_path).strip()
            if isinstance(contrastive_checkpoint_path, str)
            and str(contrastive_checkpoint_path).strip()
            else None
        )
        if (
            self._uses_frozen_initial_representation()
            and self._configured_contrastive_checkpoint_path is not None
        ):
            raise ValueError(
                "contrastive_checkpoint_path is incompatible with "
                "contrastive_representation_mode='frozen_initial'."
            )
        self._loaded_contrastive_checkpoint_path: Optional[str] = None
        self._loaded_contrastive_checkpoint_program_version_id: Optional[str] = None
        self._last_saved_training_artifacts: Dict[str, str] = {}
        self._last_version_training_snapshot_path: Optional[str] = None

        self._build_modules()
        self.reset()
        self._load_initial_contrastive_checkpoint_if_configured()

    def _ensure_state_store(self) -> StateStore:
        state_store = getattr(self, "_state_store", None)
        if isinstance(state_store, StateStore):
            return state_store
        state_store = StateStore()
        self._state_store = state_store
        self._bind_env_state_store(state_store)
        return state_store

    @property
    def known_class_count(self) -> int:
        return int(self._group_context_snapshot.known_class_count)

    @property
    def active_class_count(self) -> int:
        return int(self._group_context_snapshot.active_class_count)

    def _sync_program_context_snapshot(
        self,
        snapshot: TransitionGroupContextSnapshot,
    ) -> None:
        self._group_context_snapshot = snapshot
        # ``active_class_ids`` is an immutable tuple on a frozen snapshot, so the
        # membership set can be built once here instead of on every
        # ``_normalize_active_class_id`` call (~40k calls per training cycle).
        self._active_class_id_set = frozenset(snapshot.active_class_ids)

    def _resolve_device(self, device: str) -> torch.device:
        choice = str(device or "auto").strip().lower()
        if choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(choice)

    def _resolve_positive_transition_limit(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None

    def _resolve_amp_dtype(self) -> Optional[torch.dtype]:
        if self.device.type != "cuda" or not self.use_mixed_precision:
            return None
        supports_bf16 = bool(
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        return torch.bfloat16 if supports_bf16 else torch.float16

    def _autocast_context(self):
        if self.device.type != "cuda" or not self.use_mixed_precision or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _optimizer_backward_step(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        loss: Tensor,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        if self._use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    @classmethod
    def _resolve_contrastive_representation_mode(cls, raw_mode: Any) -> str:
        normalized_mode = str(raw_mode).strip().lower()
        if normalized_mode in cls._SUPPORTED_CONTRASTIVE_REPRESENTATION_MODES:
            return normalized_mode
        supported_modes = ", ".join(cls._SUPPORTED_CONTRASTIVE_REPRESENTATION_MODES)
        raise ValueError(
            "contrastive_representation_mode must be one of "
            f"{supported_modes}, got {raw_mode!r}."
        )

    @classmethod
    def _resolve_encode_bucket_mode(cls, raw_mode: Any) -> str:
        normalized_mode = str(raw_mode).strip().lower()
        if normalized_mode in cls._SUPPORTED_ENCODE_BUCKET_MODES:
            return normalized_mode
        supported_modes = ", ".join(cls._SUPPORTED_ENCODE_BUCKET_MODES)
        raise ValueError(
            f"encode_bucket_mode must be one of {supported_modes}, got {raw_mode!r}."
        )

    def _uses_frozen_initial_representation(self) -> bool:
        return str(self.contrastive_representation_mode) == "frozen_initial"

    @property
    def compile_enabled(self) -> bool:
        """
        Whether this process is actually running compiled encoder leaves.

        Read through to the network rather than mirroring it: the first-call
        latch can flip the network back to eager long after ``_build_modules``,
        and a stale mirrored copy would report a speedup that is not happening.
        """
        return bool(self.dynamics.compile_enabled)

    @property
    def compile_disabled_reason(self) -> Optional[str]:
        return self.dynamics.compile_disabled_reason

    def _worker_compile_disabled_reasons(self) -> Dict[int, str]:
        """Per-rank compile failures reported by the DDP children, if any."""
        trainer = self._distributed_trainer
        if trainer is None:
            return {}
        return dict(trainer.worker_compile_disabled_reasons)

    def _configure_representation_mode_modules(self) -> None:
        if not self._uses_frozen_initial_representation():
            return
        self.dynamics.eval()
        for parameter in self.dynamics.parameters():
            parameter.requires_grad_(False)

    def _build_modules(self) -> None:
        self.dynamics_state_encoder = SetStateEncoder(
            token_dim=TOKEN_DIM,
            embed_dim=self.dynamics_encoder_embed_dim,
            num_heads=self.dynamics_encoder_num_heads,
            num_blocks=self.dynamics_encoder_num_blocks,
            dropout=self.dynamics_encoder_dropout,
            num_pool_seeds=self.dynamics_pool_seeds,
        ).to(self.device)
        self.dynamics = ContrastiveDynamicsNetwork(
            state_encoder=self.dynamics_state_encoder,
            num_actions=self.num_actions,
            state_embed_dim=self.dynamics_encoder_embed_dim,
            contrastive_dim=self.contrastive_dim,
            num_classes=self.num_dynamics_classes,
            max_prototypes_per_class=self.max_prototypes_per_class,
            temperature=self.contrastive_temperature,
            num_pool_seeds=self.dynamics_pool_seeds,
        ).to(self.device)
        self.dynamics.forward_use_autocast = bool(
            self.use_mixed_precision and self.device.type == "cuda"
        )
        self.dynamics.forward_amp_dtype = self.amp_dtype
        _install_encoder_compile(
            self.dynamics,
            enabled=bool(self.compile_encoder),
            device=self.device,
        )
        self.contrastive_optimizer = torch.optim.AdamW(self.dynamics.parameters(), lr=self.contrastive_lr)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_grad_scaler)
        self._distributed_trainer: Optional[DistributedContrastiveTrainer] = None
        self._configure_representation_mode_modules()

    def reset(self) -> None:
        self._close_distributed_trainer()
        self.sample_store = ContrastiveSampleStore(deduplicate_exact=True)
        # Sole owner of the encoded per-state tensors. Every other cache stores a
        # key into this dict, so bounding it actually frees the tensors.
        self._shared_state_token_cache: Dict[str, Tuple[Tensor, Tensor]] = {}
        # Maintained incrementally on insert/evict; never recomputed by walking the cache.
        self._shared_state_token_cache_bytes = 0
        # storage index -> cache key. Saves the key resolution (a state_store lookup,
        # or a canonical_state_key hash of the state JSON), not the tensors.
        self._sample_storage_token_cache: Dict[int, str] = {}
        self._state_token_cache_evictions = 0
        self._state_token_cache_evicted_entries = 0
        self.total_steps = 0
        self.total_updates = 0
        self._sample_storage_version = 0
        self._sample_label_version = 0
        self._dynamics_parameter_version = 0
        self._prototype_summary_version = 0
        self._prototype_rebuild_pending_class_ids: Optional[set[int]] = set()
        self._sample_store_tau_cache_signature: Optional[Tuple[int, int]] = None
        self._sample_store_tau_cache: Optional[Tensor] = None
        self._prototype_sample_tau_cache_signature: Optional[Tuple[Any, ...]] = None
        self._prototype_sample_tau_cache_indices: Tuple[int, ...] = ()
        self._prototype_sample_tau_cache: Optional[Tensor] = None
        self._active_prototype_sample_indices_by_class: Dict[int, Tuple[int, ...]] = {}
        self._active_prototype_sample_indices: Tuple[int, ...] = ()
        self._prototype_entropy_context_cache_signature: Optional[Tuple[Any, ...]] = None
        self._prototype_entropy_context_cache: Optional[ContrastiveEntropyContext] = None
        self._map_reset_count = 0
        self._last_train_stats: Dict[str, float] = {}
        self._last_added_count = 0
        metric_history_limit = 240
        self._rolling_contrastive_loss = RollingScalarWindow(metric_history_limit)
        self._rolling_prototype_top1_accuracy = RollingScalarWindow(metric_history_limit)
        self._rolling_rh = RollingScalarWindow(metric_history_limit)
        self._rolling_rz = RollingScalarWindow(metric_history_limit)
        self._rolling_rtotal = RollingScalarWindow(metric_history_limit)
        self._last_observed_class_index: Optional[int] = None
        self._prototype_learning_bootstrap_completed = False
        self._prototype_rebuild_pending = False
        self._mature_prototype_class_ids: set[int] = set()
        self._current_collect_transition_key: Optional[str] = None
        self._current_collect_question_mark_count = 0
        self._has_seen_question_mark_transition = False
        self._current_source_transition_assessments.clear()

    def _load_initial_contrastive_checkpoint_if_configured(
        self,
        *,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        checkpoint_path = self._configured_contrastive_checkpoint_path
        if checkpoint_path is None:
            return None
        if not force and self._loaded_contrastive_checkpoint_path is not None:
            return None
        return self.load_training_artifacts(checkpoint_path=checkpoint_path)

    def _close_distributed_trainer(self) -> None:
        trainer = getattr(self, "_distributed_trainer", None)
        self._distributed_trainer = None
        if trainer is None:
            return
        try:
            trainer.close()
        except BaseException:
            pass

    def close(self) -> None:
        self._close_distributed_trainer()

    def __del__(self) -> None:
        trainer = getattr(self, "_distributed_trainer", None)
        if trainer is not None:
            trainer.close()

    @staticmethod
    def _resolve_optional_positive_count(value: Optional[int], name: str) -> Optional[int]:
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} must be positive when provided.")
        return int(parsed)

    def _distributed_model_config(self) -> Dict[str, Any]:
        return {
            "num_actions": int(self.num_actions),
            "dynamics_encoder_embed_dim": int(self.dynamics_encoder_embed_dim),
            "dynamics_encoder_num_heads": int(self.dynamics_encoder_num_heads),
            "dynamics_encoder_num_blocks": int(self.dynamics_encoder_num_blocks),
            "dynamics_pool_seeds": int(self.dynamics_pool_seeds),
            "dynamics_encoder_dropout": float(self.dynamics_encoder_dropout),
            "contrastive_dim": int(self.contrastive_dim),
            "max_prototypes_per_class": int(self.max_prototypes_per_class),
            "num_dynamics_classes": int(self.num_dynamics_classes),
            "contrastive_temperature": float(self.contrastive_temperature),
            "contrastive_min_class_count": int(self.contrastive_min_class_count),
            # Execution strategy, not architecture: this rides along to the DDP
            # children but is deliberately absent from the checkpoint's
            # model_config, which _validate_checkpoint_model_config matches
            # exactly. A compiled run and an eager run share checkpoints.
            "compile_encoder": bool(self.compile_encoder),
        }

    def _can_use_distributed_contrastive_training(self, batch_size: int) -> bool:
        if self._uses_frozen_initial_representation():
            return False
        if self.device.type != "cuda" or os.name == "nt":
            return False
        if not torch.cuda.is_available():
            return False
        if int(batch_size) < 2:
            return False
        return len(self.parallel_device_ids) > 1 and int(batch_size) >= len(self.parallel_device_ids)

    def _can_use_distributed_tau_encoding(self, sample_count: int) -> bool:
        if self._uses_frozen_initial_representation():
            return False
        if self.device.type != "cuda" or os.name == "nt":
            return False
        if not torch.cuda.is_available():
            return False
        if len(self.parallel_device_ids) <= 1:
            return False
        resolved_count = int(sample_count)
        if resolved_count <= 0:
            return False
        return True

    @classmethod
    def _distributed_tau_encode_chunk_size(cls, sample_count: int) -> int:
        resolved_count = max(0, int(sample_count))
        if resolved_count <= 0:
            return 1
        return min(
            resolved_count,
            max(1, int(cls._DISTRIBUTED_TAU_ENCODE_MAX_SAMPLES_PER_CHUNK)),
        )

    def _get_distributed_trainer(self) -> DistributedContrastiveTrainer:
        trainer = self._distributed_trainer
        if trainer is not None:
            return trainer
        trainer = DistributedContrastiveTrainer(
            device_ids=self.parallel_device_ids,
            model_config=self._distributed_model_config(),
            rank0_model=self.dynamics,
            rank0_optimizer=self.contrastive_optimizer,
            rank0_scaler=self.grad_scaler,
            contrastive_lr=float(self.contrastive_lr),
            amp_dtype=self.amp_dtype,
            use_grad_scaler=bool(self._use_grad_scaler),
        )
        self._distributed_trainer = trainer
        return trainer

    def _sync_distributed_trainer_from_main(self) -> None:
        trainer = getattr(self, "_distributed_trainer", None)
        if trainer is None:
            return
        try:
            trainer.load_state_dict(self.dynamics.state_dict())
        except BaseException:
            self._close_distributed_trainer()
            raise

    def _local_sample_step_command(
        self,
        storage_indices: Sequence[int],
    ) -> Dict[str, Any]:
        return self._sample_step_command_from_indices(
            storage_indices,
            share_memory=False,
            pin_memory=bool(self.device.type == "cuda"),
        )

    def _encode_samples_as_keys_from_indices(
        self,
        storage_indices: Sequence[int],
    ) -> Tensor:
        keys, _command = self._encode_sample_keys_and_command_from_indices(storage_indices)
        return keys

    def _encode_sample_keys_and_command_from_indices(
        self,
        storage_indices: Sequence[int],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Encode ``storage_indices`` and hand back the host-side step command.

        ``_sample_step_command_from_indices`` already resolves per-row labels and
        class support counts while it walks the sample store. Returning the
        command lets the caller reuse those host tensors instead of recomputing
        the labels and round-tripping them through the device.
        """
        timer = self._train_phase_timer
        command = self._local_sample_step_command(storage_indices)
        h2d_started_at = timer.mark()
        actions = command["actions"].to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        timer.record("h2d", h2d_started_at)
        total_count = int(actions.numel())
        if total_count <= 0:
            return (
                torch.zeros(
                    (0, int(self.contrastive_dim)),
                    device=self.device,
                    dtype=torch.float32,
                ),
                command,
            )
        encoded: Optional[Tensor] = None
        for bucket in command.get("buckets") or []:
            h2d_started_at = timer.mark()
            row_indices = bucket["row_indices"].to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
            if int(row_indices.numel()) <= 0:
                timer.record("h2d", h2d_started_at)
                continue
            batch_tokens, batch_mask = self._move_host_state_batch_to_device(
                tokens=bucket["state_tokens"],
                mask=bucket["state_mask"],
            )
            timer.record("h2d", h2d_started_at)
            forward_started_at = timer.mark()
            bucket_actions = actions.index_select(0, row_indices)
            bucket_encoded = self._encode_state_action_batch(
                batch_tokens,
                batch_mask,
                bucket_actions,
            )
            if encoded is None:
                encoded = torch.empty(
                    (total_count, int(bucket_encoded.size(-1))),
                    device=bucket_encoded.device,
                    dtype=bucket_encoded.dtype,
                )
            encoded.index_copy_(0, row_indices, bucket_encoded)
            timer.record("fwd", forward_started_at)
        if encoded is None:
            return (
                torch.zeros(
                    (0, int(self.contrastive_dim)),
                    device=self.device,
                    dtype=torch.float32,
                ),
                command,
            )
        return encoded, command

    def _sample_contrastive_batch_indices(self) -> List[int]:
        return self.sample_store.sample_class_balanced_pair_indices_labeled(
            self.contrastive_batch_size,
            self.rng,
            minimum_class_count=max(2, int(self.contrastive_min_class_count)),
        )

    def _distributed_rank_index_payloads(
        self,
        contrastive_indices: Sequence[int],
    ) -> Tuple[List[Tensor], int]:
        world_size = int(len(self.parallel_device_ids))
        if world_size <= 1:
            return [], 0
        usable_count = (int(len(contrastive_indices)) // world_size) * world_size
        if usable_count <= 0:
            return [], 0
        selected_indices = [int(index) for index in contrastive_indices[:usable_count]]
        shard_size = usable_count // world_size
        rank_indices: List[Tensor] = []
        for rank in range(world_size):
            start_index = int(rank) * shard_size
            shard_indices = selected_indices[start_index : start_index + shard_size]
            rank_indices.append(
                torch.as_tensor(shard_indices, device="cpu", dtype=torch.long)
            )
        return rank_indices, usable_count

    def _distributed_sample_train_indices(self) -> Tuple[List[Tensor], int]:
        contrastive_indices = self._sample_contrastive_batch_indices()
        if not contrastive_indices:
            return [], 0
        return self._distributed_rank_index_payloads(contrastive_indices)

    def _distributed_train_step_commands(
        self,
        index_batches: Sequence[Sequence[Tensor]],
    ) -> List[List[Dict[str, Any]]]:
        world_size = int(len(self.parallel_device_ids))
        rank_commands: List[List[Dict[str, Any]]] = [
            [] for _rank in range(max(0, world_size))
        ]
        if world_size <= 0:
            return rank_commands
        share_memory = _distributed_step_command_should_share_tensors()
        for rank_indices in index_batches:
            if len(rank_indices) != world_size:
                raise ValueError("distributed train index batch does not match world size.")
            for rank in range(world_size):
                raw_indices = rank_indices[int(rank)]
                storage_indices = [
                    int(index)
                    for index in raw_indices.detach().to(
                        device="cpu",
                        dtype=torch.long,
                    ).reshape(-1).tolist()
                ]
                rank_commands[int(rank)].append(
                    self._sample_step_command_from_indices(
                        storage_indices,
                        share_memory=bool(share_memory and int(rank) > 0),
                        pin_memory=bool(self.device.type == "cuda" and int(rank) == 0),
                    )
                )
        return rank_commands

    def _apply_distributed_train_result(
        self,
        result: Dict[str, Any],
        *,
        usable_count: int,
    ) -> Dict[str, float]:
        self._mark_dynamics_parameters_changed()
        raw_stats = result.get("stats")
        return self._distributed_train_stats_with_metadata(raw_stats, usable_count=usable_count)

    def _distributed_train_stats_with_metadata(
        self,
        raw_stats: Any,
        *,
        usable_count: int,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = (
            dict(raw_stats)
            if isinstance(raw_stats, dict)
            else {}
        )
        stats.update(
            {
                "contrastive_min_class_count": float(self.contrastive_min_class_count),
                "contrastive_trainable_class_count": float(
                    len(self._trainable_contrastive_class_indices())
                ),
                "contrastive_provisional_class_count": float(
                    len(self._provisional_contrastive_class_indices())
                ),
                "contrastive_batch_size": float(usable_count),
                "contrastive_epoch_batch_count": 1.0,
                "contrastive_epoch_sample_count": float(usable_count),
                "sample_store_size": float(len(self.sample_store)),
                "distributed_world_size": float(len(self.parallel_device_ids)),
            }
        )
        return {
            str(key): float(value)
            for key, value in stats.items()
            if isinstance(value, (int, float))
        }

    def _build_training_checkpoint_payload(
        self,
        *,
        current_version_id: Optional[str],
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        scaler_state = None
        if hasattr(self, "grad_scaler") and self.grad_scaler is not None:
            scaler_state = self.grad_scaler.state_dict()
        return {
            "format_version": 2,
            "strategy_name": str(self.strategy_name),
            # Discovery cycle this checkpoint was written at. Iteration-boundary
            # resume reads this to report any gap between checkpointed weights
            # and the restored analysis archive.
            "iteration": _positive_integral(iteration),
            "current_version_id": (
                str(current_version_id).strip()
                if isinstance(current_version_id, str) and str(current_version_id).strip()
                else self._current_program_version_id()
            ),
            "model_config": {
                "num_actions": int(self.num_actions),
                "dynamics_encoder_embed_dim": int(self.dynamics_encoder_embed_dim),
                "dynamics_encoder_num_heads": int(self.dynamics_encoder_num_heads),
                "dynamics_encoder_num_blocks": int(self.dynamics_encoder_num_blocks),
                "dynamics_pool_seeds": int(self.dynamics_pool_seeds),
                "dynamics_encoder_dropout": float(self.dynamics_encoder_dropout),
                "contrastive_dim": int(self.contrastive_dim),
                "max_prototypes_per_class": int(self.max_prototypes_per_class),
            },
            "num_dynamics_classes": int(self.num_dynamics_classes),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "contrastive_optimizer_state_dict": self.contrastive_optimizer.state_dict(),
            "grad_scaler_state_dict": scaler_state,
            "total_steps": int(self.total_steps),
            "total_updates": int(self.total_updates),
            "runtime_versions": self._runtime_version_state(),
            "prototype_rebuild_state": self._prototype_rebuild_state(),
            "frontier_scoring_ready": bool(self._frontier_scoring_ready),
            "last_train_stats": dict(self._last_train_stats),
            "prototype_learning_bootstrap_completed": bool(
                self._prototype_learning_bootstrap_completed
            ),
            "mature_prototype_class_ids": sorted(
                int(class_id) for class_id in self._mature_prototype_class_ids
            ),
        }

    @staticmethod
    def _safe_checkpoint_tag(value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return None
        text = str(value).strip()
        if not text:
            return None
        sanitized = "".join(
            char if (char.isalnum() or char in "._-") else "_"
            for char in text
        ).strip("._-")
        return sanitized or None

    def _checkpoint_dir(self, output_dir: str | Path) -> Path:
        checkpoint_dir = Path(output_dir).resolve() / "contrastive_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @staticmethod
    def _replace_checkpoint_from_file(
        *,
        source_path: Path,
        destination_path: Path,
    ) -> None:
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.unlink(missing_ok=True)
        try:
            try:
                os.link(source_path, temporary_path)
            except OSError:
                shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _replace_checkpoint_from_payload(
        *,
        payload: Dict[str, Any],
        destination_path: Path,
    ) -> None:
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.unlink(missing_ok=True)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _validate_checkpoint_model_config(self, payload: Dict[str, Any]) -> None:
        model_config = payload.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("Contrastive checkpoint is missing model_config.")
        expected = {
            "num_actions": int(self.num_actions),
            "dynamics_encoder_embed_dim": int(self.dynamics_encoder_embed_dim),
            "dynamics_encoder_num_heads": int(self.dynamics_encoder_num_heads),
            "dynamics_encoder_num_blocks": int(self.dynamics_encoder_num_blocks),
            "dynamics_pool_seeds": int(self.dynamics_pool_seeds),
            "dynamics_encoder_dropout": float(self.dynamics_encoder_dropout),
            "contrastive_dim": int(self.contrastive_dim),
            "max_prototypes_per_class": int(self.max_prototypes_per_class),
        }
        mismatched: List[str] = []
        for key, expected_value in expected.items():
            loaded_value = model_config.get(key)
            if loaded_value != expected_value:
                mismatched.append(
                    f"{key}: expected={expected_value!r} loaded={loaded_value!r}"
                )
        if mismatched:
            raise ValueError(
                "Contrastive checkpoint model_config does not match the current explorer: "
                + ", ".join(mismatched)
            )

    def load_training_artifacts(
        self,
        *,
        checkpoint_path: str | Path,
    ) -> Dict[str, Any]:
        if self._uses_frozen_initial_representation():
            raise ValueError(
                "Cannot load contrastive checkpoints when "
                "contrastive_representation_mode='frozen_initial'."
            )
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Contrastive checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid contrastive checkpoint payload: {path}")
        if payload.get("format_version") != 2:
            raise ValueError(
                "Unsupported contrastive checkpoint format_version: "
                f"{payload.get('format_version')!r}; expected 2."
            )
        self._validate_checkpoint_model_config(payload)

        required_count = payload.get("num_dynamics_classes", 1)
        try:
            resolved_required_count = max(1, int(required_count))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid num_dynamics_classes in checkpoint: {required_count!r}"
            ) from exc
        self._ensure_dynamics_class_capacity(resolved_required_count)

        dynamics_state_dict = payload.get("dynamics_state_dict")
        if not isinstance(dynamics_state_dict, dict):
            raise ValueError("Contrastive checkpoint is missing dynamics_state_dict.")
        self.dynamics.load_state_dict(dynamics_state_dict, strict=True)

        optimizer_state = payload.get("contrastive_optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            self.contrastive_optimizer.load_state_dict(optimizer_state)

        scaler_state = payload.get("grad_scaler_state_dict")
        if (
            isinstance(scaler_state, dict)
            and hasattr(self, "grad_scaler")
            and self.grad_scaler is not None
        ):
            self.grad_scaler.load_state_dict(scaler_state)

        self.num_dynamics_classes = max(
            int(self.num_dynamics_classes),
            resolved_required_count,
        )
        self.total_steps = max(0, int(payload.get("total_steps", 0) or 0))
        self.total_updates = max(0, int(payload.get("total_updates", 0) or 0))
        self._restore_runtime_version_state(
            payload.get("runtime_versions"),
            owner=f"contrastive checkpoint {path}",
        )
        self._restore_prototype_rebuild_state(
            payload.get("prototype_rebuild_state"),
            owner=f"contrastive checkpoint {path}",
        )
        frontier_scoring_ready = payload.get("frontier_scoring_ready")
        if not isinstance(frontier_scoring_ready, bool):
            raise ValueError(
                f"contrastive checkpoint {path} has an invalid "
                "frontier_scoring_ready value."
            )
        self._frontier_scoring_ready = frontier_scoring_ready
        last_train_stats = payload.get("last_train_stats")
        self._last_train_stats = (
            dict(last_train_stats) if isinstance(last_train_stats, dict) else {}
        )
        self._prototype_learning_bootstrap_completed = bool(
            payload.get("prototype_learning_bootstrap_completed", False)
        )
        mature_ids = payload.get("mature_prototype_class_ids")
        if isinstance(mature_ids, (list, tuple)):
            self._mature_prototype_class_ids = {
                int(class_id)
                for class_id in mature_ids
                if isinstance(class_id, int) and int(class_id) > 0
            }
        else:
            self._mature_prototype_class_ids = set()
        self._loaded_contrastive_checkpoint_path = str(path)
        loaded_version_id = payload.get("current_version_id")
        self._loaded_contrastive_checkpoint_program_version_id = (
            str(loaded_version_id).strip()
            if isinstance(loaded_version_id, str) and str(loaded_version_id).strip()
            else None
        )
        return {
            "checkpoint_path": str(path),
            "current_version_id": self._loaded_contrastive_checkpoint_program_version_id,
            "num_dynamics_classes": int(self.num_dynamics_classes),
            "total_steps": int(self.total_steps),
            "total_updates": int(self.total_updates),
        }

    def save_training_artifacts(
        self,
        *,
        output_dir: str | Path,
        current_version_id: Optional[str] = None,
        iteration: Optional[int] = None,
        record_history: bool = True,
    ) -> Optional[Dict[str, Any]]:
        checkpoint_dir = self._checkpoint_dir(output_dir)
        resolved_version_id = (
            str(current_version_id).strip()
            if isinstance(current_version_id, str) and str(current_version_id).strip()
            else self._current_program_version_id()
        )
        payload = self._build_training_checkpoint_payload(
            current_version_id=resolved_version_id,
            iteration=iteration,
        )
        resolved_iteration = max(0, int(iteration or 0))
        resolved_update = max(0, int(self.total_updates))
        history_path = checkpoint_dir / (
            f"train_iter_{resolved_iteration:06d}_"
            f"update_{resolved_update:09d}.pt"
        )
        latest_path = checkpoint_dir / "latest.pt"
        history_was_recorded = False
        if bool(record_history) and not history_path.exists():
            self._replace_checkpoint_from_payload(
                payload=payload,
                destination_path=history_path,
            )
            self._replace_checkpoint_from_file(
                source_path=history_path,
                destination_path=latest_path,
            )
            history_was_recorded = True
        else:
            # Metadata such as the accepted program version can change without
            # another optimizer update. Keep latest.pt current without changing
            # the already-recorded training history.
            self._replace_checkpoint_from_payload(
                payload=payload,
                destination_path=latest_path,
            )

        saved_paths: Dict[str, Any] = {
            "latest_checkpoint_path": str(latest_path),
        }
        if bool(record_history):
            saved_paths["history_checkpoint_path"] = str(history_path)
            saved_paths["history_checkpoint_recorded"] = history_was_recorded
        self._last_saved_training_artifacts = dict(saved_paths)
        return dict(saved_paths)

    def snapshot_training_artifacts_for_version(
        self,
        *,
        output_dir: str | Path,
        current_version_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        checkpoint_dir = self._checkpoint_dir(output_dir)
        latest_path = checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return None
        version_tag = self._safe_checkpoint_tag(current_version_id)
        if version_tag is None:
            return None
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payload["current_version_id"] = str(current_version_id).strip()
            version_path = checkpoint_dir / f"{version_tag}.pt"
            torch.save(payload, version_path)
        else:
            version_path = checkpoint_dir / f"{version_tag}.pt"
            shutil.copy2(latest_path, version_path)
        self._last_version_training_snapshot_path = str(version_path)
        return {
            "latest_checkpoint_path": str(latest_path),
            "version_checkpoint_path": str(version_path),
        }

    def _clear_shared_state_token_cache(self) -> None:
        """Drop the cached state tokens and the byte gauge that tracks them."""
        self._shared_state_token_cache.clear()
        self._shared_state_token_cache_bytes = 0

    def _store_state_token_entry(
        self,
        cache_key: str,
        entry: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Insert one encoded state, update the byte gauge, then enforce the bound.

        These tensors are deliberately left pageable. Their only consumer is
        ``_pack_encoded_state_batch_to_host``, which allocates a fresh destination
        buffer in both branches and pins *that*; a per-state pin would page-lock
        memory that never participates in a DMA.
        """
        tokens, mask = entry
        self._shared_state_token_cache[cache_key] = entry
        self._shared_state_token_cache_bytes += _tensor_host_bytes(
            tokens
        ) + _tensor_host_bytes(mask)
        self._evict_state_token_cache_if_needed()
        return entry

    def _evict_state_token_cache_if_needed(self) -> None:
        """Drop a random batch of entries once the cache exceeds its bound.

        Random batch eviction, not LRU. The training path samples uniformly inside
        each class, where every policy yields the same hit rate but LRU adds
        per-hit bookkeeping; the frontier/prototype path scans a re-drawn set
        cyclically, where LRU is the worst possible choice because it evicts
        exactly what the next pass needs. Evicting a batch (not one entry) keeps
        the cache from oscillating on the limit and re-running this scan per insert.
        """
        max_entries = int(self.state_token_cache_max_entries)
        if max_entries <= 0:
            return
        cache = self._shared_state_token_cache
        if len(cache) <= max_entries:
            return
        victim_count = max(1, (len(cache) * self._STATE_TOKEN_CACHE_EVICT_PERCENT) // 100)
        victim_count = min(victim_count, len(cache))
        victims = self._state_token_cache_rng.sample(list(cache.keys()), victim_count)
        freed_bytes = 0
        for victim_key in victims:
            evicted = cache.pop(victim_key, None)
            if evicted is None:
                continue
            evicted_tokens, evicted_mask = evicted
            freed_bytes += _tensor_host_bytes(evicted_tokens) + _tensor_host_bytes(
                evicted_mask
            )
        self._shared_state_token_cache_bytes = max(
            0,
            int(self._shared_state_token_cache_bytes) - int(freed_bytes),
        )
        self._state_token_cache_evictions += 1
        self._state_token_cache_evicted_entries += len(victims)

    def _cached_state_tokens_entry(
        self,
        *,
        state_key: str,
        state_json: str,
    ) -> Tuple[str, Tuple[Tensor, Tensor]]:
        resolved_state_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and state_key.strip()
            else canonical_state_key(state_json)
        )
        cached = self._shared_state_token_cache.get(resolved_state_key)
        if cached is not None:
            return resolved_state_key, cached
        tokens, mask = self.codec.encode_state(state_json)
        entry = (
            torch.from_numpy(tokens).to(dtype=torch.float32),
            torch.from_numpy(mask).to(dtype=torch.bool),
        )
        return resolved_state_key, self._store_state_token_entry(
            resolved_state_key,
            entry,
        )

    def _cached_state_tokens(
        self,
        *,
        state_key: str,
        state_json: str,
    ) -> Tuple[Tensor, Tensor]:
        return self._cached_state_tokens_entry(
            state_key=state_key,
            state_json=state_json,
        )[1]

    def _cached_state_tokens_by_id_entry(
        self,
        state_id: int,
    ) -> Tuple[str, Tuple[Tensor, Tensor]]:
        resolved_state_id = int(state_id)
        state_store = self._ensure_state_store()
        state_key = state_store.state_key(resolved_state_id)
        cache_key = (
            str(state_key).strip()
            if isinstance(state_key, str) and state_key.strip()
            else f"state_id:{resolved_state_id}"
        )
        cached = self._shared_state_token_cache.get(cache_key)
        if cached is not None:
            return cache_key, cached
        tokens, mask = self.codec.encode_state_id(state_store, resolved_state_id)
        entry = (
            torch.from_numpy(tokens).to(dtype=torch.float32),
            torch.from_numpy(mask).to(dtype=torch.bool),
        )
        return cache_key, self._store_state_token_entry(cache_key, entry)

    def _cached_state_tokens_by_id(
        self,
        state_id: int,
    ) -> Tuple[Tensor, Tensor]:
        return self._cached_state_tokens_by_id_entry(state_id)[1]

    def _sample_state_tokens_entry(
        self,
        item: ContrastiveSample,
        *,
        next_state: bool = False,
    ) -> Tuple[str, Tuple[Tensor, Tensor]]:
        if next_state:
            next_state_id = _positive_integral(item.next_state_id)
            if next_state_id is not None:
                return self._cached_state_tokens_by_id_entry(int(next_state_id))
            return self._cached_state_tokens_entry(
                state_key=item.resolved_next_state_key(),
                state_json=item.resolved_next_state_json(),
            )
        state_id = _positive_integral(item.state_id)
        if state_id is not None:
            return self._cached_state_tokens_by_id_entry(int(state_id))
        return self._cached_state_tokens_entry(
            state_key=item.resolved_state_key(),
            state_json=item.resolved_state_json(),
        )

    def _sample_state_tokens(
        self,
        item: ContrastiveSample,
        *,
        next_state: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        return self._sample_state_tokens_entry(item, next_state=next_state)[1]

    def _sample_state_tokens_for_storage_index(
        self,
        storage_index: int,
        item: ContrastiveSample,
    ) -> Tuple[Tensor, Tensor]:
        """Resolve one stored sample's tokens, skipping the key resolution on a hit.

        The storage map holds only the cache key, so an entry evicted from
        ``_shared_state_token_cache`` degrades to a recompute rather than a stale
        or missing tensor.
        """
        resolved_index = int(storage_index)
        cache_key = self._sample_storage_token_cache.get(resolved_index)
        if cache_key is not None:
            cached = self._shared_state_token_cache.get(cache_key)
            if cached is not None:
                return cached
        resolved_key, encoded = self._sample_state_tokens_entry(item, next_state=False)
        self._sample_storage_token_cache[resolved_index] = resolved_key
        return encoded

    def _invalidate_sample_store_tau_cache(self) -> None:
        self._sample_store_tau_cache_signature = None
        self._sample_store_tau_cache = None
        self._prototype_sample_tau_cache_signature = None
        self._prototype_sample_tau_cache_indices = ()
        self._prototype_sample_tau_cache = None

    def _invalidate_prototype_entropy_context_cache(self) -> None:
        self._prototype_entropy_context_cache_signature = None
        self._prototype_entropy_context_cache = None

    def _maybe_pin_host_tensor(self, tensor: Tensor) -> Tensor:
        if self.device.type != "cuda":
            return tensor
        if tensor.device.type != "cpu":
            return tensor
        if tensor.is_pinned():
            return tensor
        if not self._train_phase_timer.enabled:
            return tensor.pin_memory()
        started_at = perf_counter()
        pinned = tensor.pin_memory()
        self._pin_host_tensor_calls += 1
        self._pin_host_tensor_seconds += perf_counter() - started_at
        return pinned

    def _move_host_state_batch_to_device(
        self,
        *,
        tokens: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.device.type == "cuda":
            return (
                tokens.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=bool(tokens.is_pinned()),
                ),
                mask.to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=bool(mask.is_pinned()),
                ),
            )
        return (
            tokens.to(device=self.device, dtype=torch.float32),
            mask.to(device=self.device, dtype=torch.bool),
        )

    def _empty_packed_state_batch(self) -> Tuple[Tensor, Tensor]:
        feature_dim = int(getattr(self.codec, "token_dim", TOKEN_DIM))
        return (
            torch.zeros((0, 1, feature_dim), device=self.device, dtype=torch.float32),
            torch.zeros((0, 1), device=self.device, dtype=torch.bool),
        )

    def _empty_host_packed_state_batch(self) -> Tuple[Tensor, Tensor]:
        feature_dim = int(getattr(self.codec, "token_dim", TOKEN_DIM))
        return (
            torch.zeros((0, 1, feature_dim), device="cpu", dtype=torch.float32),
            torch.zeros((0, 1), device="cpu", dtype=torch.bool),
        )

    def _pack_cached_state_batch_to_host(
        self,
        encoded_items: Sequence[Tuple[Tensor, Tensor]],
        *,
        pin_memory: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        return _pack_encoded_state_batch_to_host(
            encoded_items,
            token_dim=int(getattr(self.codec, "token_dim", TOKEN_DIM)),
            pin_memory=bool(self.device.type == "cuda") if pin_memory is None else bool(pin_memory),
        )

    def _pack_cached_state_batch(
        self,
        encoded_items: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        host_tokens, host_mask = self._pack_cached_state_batch_to_host(encoded_items)
        return self._move_host_state_batch_to_device(
            tokens=host_tokens,
            mask=host_mask,
        )

    def _pack_sample_state_buckets(
        self,
        items: Sequence[ContrastiveSample],
        *,
        next_state: bool = False,
    ) -> List[PackedStateBucket]:
        host_buckets = self._pack_sample_state_host_buckets(
            items,
            next_state=next_state,
        )
        packed_batches: List[PackedStateBucket] = []
        for bucket in host_buckets:
            batch_tokens, batch_mask = self._move_host_state_batch_to_device(
                tokens=bucket.state_tokens,
                mask=bucket.state_mask,
            )
            packed_batches.append(
                PackedStateBucket(
                    row_indices=bucket.row_indices.to(
                        device=self.device,
                        dtype=torch.long,
                        non_blocking=bool(bucket.row_indices.is_pinned()),
                    ),
                    state_tokens=batch_tokens,
                    state_mask=batch_mask,
                )
            )
        return packed_batches

    def _pack_sample_state_host_buckets(
        self,
        items: Sequence[ContrastiveSample],
        *,
        next_state: bool = False,
        mode: Optional[str] = None,
        pin_memory: Optional[bool] = None,
    ) -> List[HostPackedStateBucket]:
        if not items:
            return []
        encoded_items: List[Tuple[Tensor, Tensor]] = []
        entity_sizes: List[int] = []
        for row_index, item in enumerate(items):
            current_tokens, current_mask = self._sample_state_tokens(
                item,
                next_state=next_state,
            )
            encoded_items.append((current_tokens, current_mask))
            entity_sizes.append(max(1, int(current_tokens.size(0))))

        resolved_mode = self._resolve_encode_bucket_mode(mode or self.encode_bucket_mode)
        if resolved_mode == "none":
            bucket_rows = [list(range(len(encoded_items)))]
        else:
            bucket_rows = self._token_budget_state_bucket_rows(entity_sizes)

        packed_batches: List[HostPackedStateBucket] = []
        for row_indices in bucket_rows:
            batch_tokens, batch_mask = self._pack_cached_state_batch_to_host(
                [encoded_items[row_index] for row_index in row_indices],
                pin_memory=pin_memory,
            )
            row_index_tensor = torch.as_tensor(row_indices, device="cpu", dtype=torch.long)
            if pin_memory is None or bool(pin_memory):
                row_index_tensor = self._maybe_pin_host_tensor(row_index_tensor)
            packed_batches.append(
                HostPackedStateBucket(
                    row_indices=row_index_tensor,
                    state_tokens=batch_tokens,
                    state_mask=batch_mask,
                )
            )
        return packed_batches

    def _token_budget_state_bucket_rows(
        self,
        entity_sizes: Sequence[int],
    ) -> List[List[int]]:
        return _token_budget_state_bucket_rows(
            entity_sizes,
            max_padded_tokens=int(self._ENCODE_BUCKET_MAX_PADDED_TOKENS),
            max_padding_ratio=float(self._ENCODE_BUCKET_MAX_PADDING_RATIO),
        )

    def _sample_command_tensor(
        self,
        tensor: Tensor,
        *,
        share_memory: bool,
        pin_memory: bool,
    ) -> Tensor:
        resolved = tensor.detach()
        if not resolved.is_contiguous():
            resolved = resolved.contiguous()
        if bool(share_memory):
            return _share_cpu_tensor_if_requested(
                resolved,
                share_memory=True,
            )
        return _pin_cpu_tensor_if_requested(
            resolved,
            pin_memory=bool(pin_memory),
        )

    def _sample_step_command_from_indices(
        self,
        storage_indices: Sequence[int],
        *,
        share_memory: bool,
        pin_memory: bool,
    ) -> Dict[str, Any]:
        resolved_indices = [
            int(index)
            for index in storage_indices
            if 0 <= int(index) < len(self.sample_store.storage)
        ]
        timer = self._train_phase_timer
        encode_started_at = timer.mark()
        labels: List[int] = []
        actions: List[int] = []
        support_counts: List[int] = []
        encoded_items: List[Tuple[Tensor, Tensor]] = []
        entity_sizes: List[int] = []
        for storage_index in resolved_indices:
            item = self.sample_store.storage[int(storage_index)]
            label = int(self._resolve_item_class_id(item))
            labels.append(label)
            actions.append(int(item.action))
            support_counts.append(
                int(self.sample_store.class_count(label)) if label > 0 else 0
            )
            tokens, mask = self._sample_state_tokens_for_storage_index(
                int(storage_index),
                item,
            )
            encoded_items.append((tokens, mask))
            entity_sizes.append(max(1, int(tokens.size(0))))
        timer.record("enc", encode_started_at)

        pack_started_at = timer.mark()
        labels_tensor = torch.as_tensor(labels, device="cpu", dtype=torch.long)
        actions_tensor = torch.as_tensor(actions, device="cpu", dtype=torch.long)
        support_counts_tensor = torch.as_tensor(
            support_counts,
            device="cpu",
            dtype=torch.long,
        )
        buckets: List[Dict[str, Tensor]] = []
        if str(self.encode_bucket_mode) == "none":
            bucket_rows = [list(range(len(encoded_items)))]
        else:
            bucket_rows = self._token_budget_state_bucket_rows(entity_sizes)
        for row_indices in bucket_rows:
            state_tokens, state_mask = _pack_encoded_state_batch_to_host(
                [encoded_items[int(row_index)] for row_index in row_indices],
                token_dim=int(getattr(self.codec, "token_dim", TOKEN_DIM)),
                pin_memory=False if share_memory else bool(pin_memory),
            )
            row_index_tensor = torch.as_tensor(
                row_indices,
                device="cpu",
                dtype=torch.long,
            )
            buckets.append(
                {
                    "row_indices": self._sample_command_tensor(
                        row_index_tensor,
                        share_memory=bool(share_memory),
                        pin_memory=bool(pin_memory),
                    ),
                    "state_tokens": self._sample_command_tensor(
                        state_tokens,
                        share_memory=bool(share_memory),
                        pin_memory=bool(pin_memory),
                    ),
                    "state_mask": self._sample_command_tensor(
                        state_mask,
                        share_memory=bool(share_memory),
                        pin_memory=bool(pin_memory),
                    ),
                }
            )
        command = {
            "labels": self._sample_command_tensor(
                labels_tensor,
                share_memory=bool(share_memory),
                pin_memory=bool(pin_memory),
            ),
            "actions": self._sample_command_tensor(
                actions_tensor,
                share_memory=bool(share_memory),
                pin_memory=bool(pin_memory),
            ),
            "support_counts": self._sample_command_tensor(
                support_counts_tensor,
                share_memory=bool(share_memory),
                pin_memory=bool(pin_memory),
            ),
            "buckets": buckets,
        }
        timer.record("pack", pack_started_at)
        return command

    def _encode_bucketed_state_action_embeddings(
        self,
        *,
        buckets: Sequence[PackedStateBucket],
        actions: Tensor,
        total_count: int,
    ) -> Tensor:
        if total_count <= 0:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        encoded: Optional[Tensor] = None
        for bucket in buckets:
            row_indices = bucket.row_indices
            if row_indices.numel() <= 0:
                continue
            bucket_actions = actions.index_select(0, row_indices)
            bucket_encoded = self._encode_state_action_batch(
                bucket.state_tokens,
                bucket.state_mask,
                bucket_actions,
            )
            if encoded is None:
                encoded = torch.empty(
                    (int(total_count), int(bucket_encoded.size(-1))),
                    device=bucket_encoded.device,
                    dtype=bucket_encoded.dtype,
                )
            encoded.index_copy_(0, row_indices, bucket_encoded)
        if encoded is None:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        return encoded

    def _encode_state_action_batch(
        self,
        tokens: Tensor,
        mask: Tensor,
        actions: Tensor,
    ) -> Tensor:
        if actions.numel() <= 0:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        with self._autocast_context():
            return self.dynamics.encode_state_action(tokens, mask, actions)

    def _runtime_version_state(self) -> Dict[str, int]:
        return {
            "sample_storage_version": int(self._sample_storage_version),
            "sample_label_version": int(self._sample_label_version),
            "dynamics_parameter_version": int(self._dynamics_parameter_version),
            "prototype_summary_version": int(self._prototype_summary_version),
        }

    @classmethod
    def _resolve_runtime_version_state(
        cls,
        values: Any,
        *,
        owner: str,
    ) -> Dict[str, int]:
        if not isinstance(values, Mapping):
            raise ValueError(f"{owner} is missing runtime_versions.")
        resolved: Dict[str, int] = {}
        for name in cls._RUNTIME_VERSION_FIELDS:
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{owner} has an invalid runtime version {name!r}.")
            parsed = int(value)
            if parsed < 0:
                raise ValueError(f"{owner} has a negative runtime version {name!r}.")
            resolved[name] = parsed
        return resolved

    def _restore_runtime_version_state(
        self,
        values: Any,
        *,
        owner: str,
    ) -> None:
        resolved = self._resolve_runtime_version_state(values, owner=owner)
        self._sample_storage_version = resolved["sample_storage_version"]
        self._sample_label_version = resolved["sample_label_version"]
        self._dynamics_parameter_version = resolved["dynamics_parameter_version"]
        self._prototype_summary_version = resolved["prototype_summary_version"]
        self._invalidate_sample_store_tau_cache()
        self._invalidate_prototype_entropy_context_cache()

    def _prototype_rebuild_state(self) -> Dict[str, Any]:
        class_ids = self._prototype_rebuild_pending_class_ids
        return {
            "pending": bool(self._prototype_rebuild_pending),
            "class_ids": (
                None
                if class_ids is None
                else sorted(int(class_id) for class_id in class_ids)
            ),
        }

    @staticmethod
    def _resolve_prototype_rebuild_state(
        values: Any,
        *,
        owner: str,
    ) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        if not isinstance(values, Mapping):
            raise ValueError(f"{owner} is missing prototype_rebuild_state.")
        pending = values.get("pending")
        if not isinstance(pending, bool):
            raise ValueError(f"{owner} has an invalid prototype rebuild pending flag.")
        raw_class_ids = values.get("class_ids")
        if raw_class_ids is None:
            class_ids = None
        elif isinstance(raw_class_ids, (list, tuple)):
            resolved_class_ids: set[int] = set()
            for value in raw_class_ids:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(
                        f"{owner} has an invalid prototype rebuild class id."
                    )
                class_id = int(value)
                if class_id <= 0:
                    raise ValueError(
                        f"{owner} has a non-positive prototype rebuild class id."
                    )
                resolved_class_ids.add(class_id)
            class_ids = tuple(sorted(resolved_class_ids))
        else:
            raise ValueError(f"{owner} has invalid prototype rebuild class_ids.")
        return bool(pending), class_ids

    def _restore_prototype_rebuild_state(
        self,
        values: Any,
        *,
        owner: str,
    ) -> None:
        pending, class_ids = self._resolve_prototype_rebuild_state(
            values,
            owner=owner,
        )
        self._prototype_rebuild_pending = bool(pending)
        self._prototype_rebuild_pending_class_ids = (
            None if class_ids is None else set(class_ids)
        )

    def _mark_sample_storage_changed(self) -> None:
        self._sample_storage_version += 1

    def _mark_sample_labels_changed(
        self,
        *,
        indices: Optional[Sequence[int]] = None,
        class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        del indices
        self._sample_label_version += 1
        self._invalidate_prototype_entropy_context_cache()
        if self._prototype_learning_bootstrap_completed:
            self._mark_prototype_rebuild_pending(class_ids=class_ids)

    def _mark_dynamics_parameters_changed(self) -> None:
        self._dynamics_parameter_version += 1
        self._invalidate_sample_store_tau_cache()
        if self._prototype_learning_bootstrap_completed:
            self._mark_prototype_rebuild_pending()

    def _mark_prototype_summary_changed(self) -> None:
        self._prototype_summary_version += 1

    def _mark_prototype_rebuild_pending(
        self,
        *,
        class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if self._uses_frozen_initial_representation():
            return
        if class_ids is not None:
            resolved_class_ids = {
                int(class_id) for class_id in class_ids if int(class_id) > 0
            }
            if not resolved_class_ids:
                return
            if (
                bool(self._prototype_rebuild_pending)
                and self._prototype_rebuild_pending_class_ids is None
            ):
                return
            pending_class_ids = self._prototype_rebuild_pending_class_ids
            if pending_class_ids is None:
                pending_class_ids = set()
            pending_class_ids.update(resolved_class_ids)
            self._prototype_rebuild_pending_class_ids = pending_class_ids
            self._prototype_rebuild_pending = True
            return
        self._prototype_rebuild_pending_class_ids = None
        self._prototype_rebuild_pending = True

    def _ensure_prototypes_current(self) -> bool:
        if not bool(self._prototype_rebuild_pending):
            return False
        pending_class_ids = self._prototype_rebuild_pending_class_ids
        class_ids = (
            None
            if pending_class_ids is None
            else sorted(int(class_id) for class_id in pending_class_ids)
        )
        if class_ids is None:
            rebuilt = self._rebuild_prototypes_from_samples()
        else:
            rebuilt = self._rebuild_prototypes_from_samples(class_ids=class_ids)
        self._prototype_rebuild_pending = False
        self._prototype_rebuild_pending_class_ids = set()
        return rebuilt

    def _prototype_bootstrap_threshold(self) -> int:
        threshold = max(1, int(self.contrastive_min_class_count))
        if self.prototype_sample_min_per_class is not None:
            threshold = max(threshold, int(self.prototype_sample_min_per_class))
        return int(threshold)

    @staticmethod
    def _storage_indices_signature(storage_indices: Sequence[int]) -> Tuple[int, str]:
        hasher = hashlib.blake2b(digest_size=16)
        count = 0
        for raw_index in storage_indices:
            index = max(0, int(raw_index))
            hasher.update(index.to_bytes(8, byteorder="little", signed=False))
            count += 1
        return int(count), hasher.hexdigest()

    def _prototype_sample_rng(
        self,
        *,
        tag: str,
        target_class_ids: Sequence[int],
        class_sizes: Mapping[int, int],
    ) -> random.Random:
        signature_parts = (
            int(self.seed),
            str(tag),
            int(self._sample_storage_version),
            int(self._sample_label_version),
            int(len(self.sample_store.storage)),
            int(self.prototype_sample_cap) if self.prototype_sample_cap is not None else None,
            int(self._prototype_bootstrap_threshold()),
            tuple(int(class_index) for class_index in target_class_ids),
            tuple(
                (int(class_index), int(class_sizes.get(int(class_index), 0)))
                for class_index in target_class_ids
            ),
        )
        return random.Random("|".join(str(part) for part in signature_parts))

    def _select_prototype_rebuild_samples(
        self,
        *,
        target_class_ids: Optional[Sequence[int]] = None,
        class_counts: Optional[Mapping[int, int]] = None,
    ) -> PrototypeSampleSelection:
        resolved_class_counts = (
            {
                int(class_index): int(count)
                for class_index, count in class_counts.items()
            }
            if class_counts is not None
            else self.sample_store.class_counts(include_zero=False)
        )
        if target_class_ids is None:
            candidate_class_ids = sorted(
                int(class_index)
                for class_index, count in resolved_class_counts.items()
                if int(class_index) > 0 and int(count) > 0
            )
        else:
            candidate_class_ids = sorted(
                {
                    int(class_index)
                    for class_index in target_class_ids
                    if int(class_index) > 0
                }
            )
        if not candidate_class_ids:
            return PrototypeSampleSelection(storage_indices=(), class_index_ranges=())

        minimum_count = int(self._prototype_bootstrap_threshold())
        class_to_indices: Dict[int, Tuple[int, ...]] = {}
        for class_index in candidate_class_ids:
            if int(resolved_class_counts.get(int(class_index), 0)) < minimum_count:
                continue
            valid_indices: List[int] = []
            for storage_index in self.sample_store._class_to_indices.get(int(class_index), []):
                resolved_index = int(storage_index)
                if 0 <= resolved_index < len(self.sample_store.storage):
                    valid_indices.append(resolved_index)
            if len(valid_indices) < minimum_count:
                continue
            class_to_indices[int(class_index)] = tuple(valid_indices)

        if not class_to_indices:
            return PrototypeSampleSelection(storage_indices=(), class_index_ranges=())

        if self.prototype_sample_cap is None:
            selected_by_class: Dict[int, set[int]] = {
                int(class_index): set(indices)
                for class_index, indices in class_to_indices.items()
            }
        else:
            cap = int(self.prototype_sample_cap)
            required_floor = int(minimum_count) * len(class_to_indices)
            if cap < required_floor:
                raise ValueError(
                    "prototype_sample_cap is too small to guarantee "
                    f"{minimum_count} samples for each eligible prototype class: "
                    f"prototype_sample_cap={cap}, eligible_class_count={len(class_to_indices)}, "
                    f"minimum_required={required_floor}. Increase prototype_sample_cap "
                    "or lower prototype_sample_min_per_class."
                )
            total_eligible_samples = sum(len(indices) for indices in class_to_indices.values())
            if cap >= total_eligible_samples:
                selected_by_class = {
                    int(class_index): set(indices)
                    for class_index, indices in class_to_indices.items()
                }
            else:
                rng = self._prototype_sample_rng(
                    tag="prototype_rebuild",
                    target_class_ids=tuple(class_to_indices),
                    class_sizes={
                        int(class_index): len(indices)
                        for class_index, indices in class_to_indices.items()
                    },
                )
                selected_by_class = {int(class_index): set() for class_index in class_to_indices}
                selected_indices: set[int] = set()
                for class_index in sorted(class_to_indices):
                    class_indices = list(class_to_indices[int(class_index)])
                    class_floor = min(len(class_indices), minimum_count)
                    class_selected = rng.sample(class_indices, class_floor)
                    selected_by_class[int(class_index)].update(int(index) for index in class_selected)
                    selected_indices.update(int(index) for index in class_selected)

                remaining_slots = max(0, cap - len(selected_indices))
                if remaining_slots > 0:
                    remaining_pool: List[Tuple[int, int]] = []
                    for class_index in sorted(class_to_indices):
                        for storage_index in class_to_indices[int(class_index)]:
                            resolved_index = int(storage_index)
                            if resolved_index not in selected_indices:
                                remaining_pool.append((int(class_index), resolved_index))
                    if remaining_slots >= len(remaining_pool):
                        extra_items = remaining_pool
                    else:
                        extra_items = rng.sample(remaining_pool, remaining_slots)
                    for class_index, storage_index in extra_items:
                        selected_by_class[int(class_index)].add(int(storage_index))

        storage_indices: List[int] = []
        class_index_ranges: List[Tuple[int, int, int]] = []
        for class_index in sorted(selected_by_class):
            selected_indices_for_class = sorted(int(index) for index in selected_by_class[int(class_index)])
            if not selected_indices_for_class:
                continue
            class_start = len(storage_indices)
            storage_indices.extend(selected_indices_for_class)
            class_end = len(storage_indices)
            class_index_ranges.append((int(class_index), int(class_start), int(class_end)))
        return PrototypeSampleSelection(
            storage_indices=tuple(storage_indices),
            class_index_ranges=tuple(class_index_ranges),
        )

    @staticmethod
    def _prototype_selection_indices_by_class(
        selection: PrototypeSampleSelection,
    ) -> Dict[int, Tuple[int, ...]]:
        selected_by_class: Dict[int, Tuple[int, ...]] = {}
        storage_indices = tuple(int(index) for index in selection.storage_indices)
        for class_index, class_start, class_end in selection.class_index_ranges:
            start = max(0, int(class_start))
            end = max(start, int(class_end))
            selected_by_class[int(class_index)] = tuple(storage_indices[start:end])
        return selected_by_class

    def _refresh_active_prototype_sample_indices(self) -> None:
        flattened: List[int] = []
        for class_index in sorted(self._active_prototype_sample_indices_by_class):
            flattened.extend(
                int(index)
                for index in self._active_prototype_sample_indices_by_class[int(class_index)]
            )
        self._active_prototype_sample_indices = tuple(flattened)

    def _record_active_prototype_sample_selection(
        self,
        *,
        selected_by_class: Mapping[int, Sequence[int]],
        target_class_ids: Sequence[int],
        full_rebuild: bool,
    ) -> None:
        if self.prototype_sample_cap is None:
            return
        if full_rebuild:
            active: Dict[int, Tuple[int, ...]] = {}
        else:
            active = dict(self._active_prototype_sample_indices_by_class)
            for class_index in target_class_ids:
                active.pop(int(class_index), None)
        for class_index, storage_indices in selected_by_class.items():
            resolved_indices = tuple(int(index) for index in storage_indices)
            if resolved_indices:
                active[int(class_index)] = resolved_indices
        self._active_prototype_sample_indices_by_class = active
        self._refresh_active_prototype_sample_indices()

    def _sync_mature_prototype_classes(
        self,
        *,
        class_counts: Dict[int, int],
        target_class_ids: Optional[Sequence[int]] = None,
    ) -> None:
        threshold = self._prototype_bootstrap_threshold()
        if target_class_ids is None:
            if not self._prototype_learning_bootstrap_completed:
                self._mature_prototype_class_ids.clear()
                return
            self._mature_prototype_class_ids = {
                int(class_index)
                for class_index, count in class_counts.items()
                if int(class_index) > 0 and int(count) >= threshold
            }
            return
        targeted = {
            int(class_index)
            for class_index in target_class_ids
            if int(class_index) > 0
        }
        for class_index in targeted:
            count = int(class_counts.get(int(class_index), 0))
            if self._prototype_learning_bootstrap_completed and count >= threshold:
                self._mature_prototype_class_ids.add(int(class_index))
            else:
                self._mature_prototype_class_ids.discard(int(class_index))

    def _rebuild_prototypes_from_samples(
        self,
        *,
        class_ids: Optional[Sequence[int]] = None,
        mark_learning_bootstrap_complete: bool = False,
    ) -> bool:
        if self.prototype_sample_cap is not None and class_ids is not None:
            class_ids = None
        full_rebuild = class_ids is None
        if self._uses_frozen_initial_representation():
            if mark_learning_bootstrap_complete:
                self._prototype_learning_bootstrap_completed = True
            if full_rebuild:
                self._prototype_rebuild_pending = False
                self._prototype_rebuild_pending_class_ids = set()
            return False
        class_counts = self.sample_store.class_counts(include_zero=False)
        positive_class_counts = {
            int(class_index): int(count)
            for class_index, count in class_counts.items()
            if int(class_index) > 0 and int(count) > 0
        }
        if mark_learning_bootstrap_complete:
            self._prototype_learning_bootstrap_completed = True

        if class_ids is None:
            target_class_ids = sorted(positive_class_counts)
            self._sync_mature_prototype_classes(class_counts=positive_class_counts)
        else:
            target_class_ids = sorted(
                {
                    int(class_index)
                    for class_index in class_ids
                    if int(class_index) > 0
                }
            )
            self._sync_mature_prototype_classes(
                class_counts=positive_class_counts,
                target_class_ids=target_class_ids,
            )
            target_class_ids = [
                int(class_index)
                for class_index in target_class_ids
                if int(positive_class_counts.get(int(class_index), 0)) > 0
            ]
        if not target_class_ids or len(self.sample_store) <= 0:
            self._record_active_prototype_sample_selection(
                selected_by_class={},
                target_class_ids=target_class_ids,
                full_rebuild=full_rebuild,
            )
            if full_rebuild:
                self._prototype_rebuild_pending = False
                self._prototype_rebuild_pending_class_ids = set()
            return False

        self._ensure_dynamics_class_capacity(int(max(target_class_ids)))
        selection = self._select_prototype_rebuild_samples(
            target_class_ids=target_class_ids,
            class_counts=positive_class_counts,
        )
        storage_indices = list(selection.storage_indices)
        class_index_ranges = list(selection.class_index_ranges)
        if not storage_indices:
            self._record_active_prototype_sample_selection(
                selected_by_class={},
                target_class_ids=target_class_ids,
                full_rebuild=full_rebuild,
            )
            if full_rebuild:
                self._prototype_rebuild_pending = False
                self._prototype_rebuild_pending_class_ids = set()
            return False

        with torch.no_grad():
            if self.prototype_sample_cap is not None:
                tau_embeddings = self._encode_prototype_sample_tau_embeddings(
                    storage_indices
                ).to(dtype=torch.float32)
            else:
                full_tau_signature = (
                    int(self._dynamics_parameter_version),
                    int(len(self.sample_store.storage)),
                )
                cached_full_tau = (
                    self._sample_store_tau_cache
                    if self._sample_store_tau_cache_signature == full_tau_signature
                    else None
                )
                if full_rebuild:
                    cached_full_tau = self._encode_current_sample_store_tau_embeddings()
                if cached_full_tau is not None:
                    tau_embeddings = cached_full_tau.index_select(
                        0,
                        torch.as_tensor(
                            storage_indices,
                            device=self.device,
                            dtype=torch.long,
                        ),
                    ).to(dtype=torch.float32)
                else:
                    tau_embeddings = self._encode_tau_embeddings_for_indices(
                        storage_indices
                    ).to(dtype=torch.float32)
            tau_embeddings = _clone_if_inference_tensor(tau_embeddings.detach())
        if tau_embeddings.ndim != 2 or int(tau_embeddings.size(0)) != int(len(storage_indices)):
            if full_rebuild:
                self._prototype_rebuild_pending = False
                self._prototype_rebuild_pending_class_ids = set()
            return False

        updated = False
        selected_by_class = self._prototype_selection_indices_by_class(selection)
        applied_selected_by_class: Dict[int, Tuple[int, ...]] = {}
        for class_index, class_start, class_end in class_index_ranges:
            class_vectors = tau_embeddings[int(class_start) : int(class_end)]
            if int(class_vectors.size(0)) <= 0:
                continue
            prototype_vectors = self._estimate_class_prototype_vectors(
                class_vectors
            )
            if self.dynamics.set_prototype_vectors(int(class_index), prototype_vectors):
                updated = True
                applied_selected_by_class[int(class_index)] = tuple(
                    selected_by_class.get(int(class_index), ())
                )
        self._record_active_prototype_sample_selection(
            selected_by_class=applied_selected_by_class,
            target_class_ids=target_class_ids,
            full_rebuild=full_rebuild,
        )
        if updated:
            self._mark_prototype_summary_changed()
            self._sync_distributed_trainer_from_main()
        if full_rebuild:
            self._prototype_rebuild_pending = False
            self._prototype_rebuild_pending_class_ids = set()
        return updated

    def _target_prototype_count_for_support(self, support_count: int) -> int:
        resolved_support_count = max(1, int(support_count))
        if resolved_support_count < int(self.prototype_split_base_count):
            scheduled_count = 1
        else:
            scheduled_count = 2 + int(
                math.floor(
                    math.log2(
                        float(resolved_support_count)
                        / float(self.prototype_split_base_count)
                    )
                )
            )
        occupancy_limited_count = max(
            1,
            resolved_support_count // int(self.prototype_split_min_cluster_occupancy),
        )
        return min(
            int(self.max_prototypes_per_class),
            max(1, int(scheduled_count)),
            int(occupancy_limited_count),
        )

    def _estimate_class_prototype_vectors(
        self,
        class_vectors: Tensor,
    ) -> Tensor:
        vectors = class_vectors.to(dtype=torch.float32)
        if vectors.ndim != 2 or int(vectors.size(0)) <= 0:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        normalized_vectors = F.normalize(vectors, dim=-1, eps=1e-6)
        prototype_count = self._target_prototype_count_for_support(
            int(normalized_vectors.size(0))
        )
        if prototype_count == 1:
            return normalized_vectors.mean(dim=0, keepdim=True)
        sample_count = int(normalized_vectors.size(0))
        if sample_count <= prototype_count:
            if sample_count == prototype_count:
                return normalized_vectors
            class_mean = normalized_vectors.mean(dim=0, keepdim=True)
            padding = class_mean.expand(prototype_count - sample_count, -1)
            return torch.cat([normalized_vectors, padding], dim=0)

        centers = self._initialize_spherical_kmeans_centers(
            normalized_vectors,
            prototype_count=prototype_count,
        )
        for _ in range(8):
            similarities = torch.matmul(normalized_vectors, centers.t())
            assignments = similarities.argmax(dim=1)
            updated_centers = centers.clone()
            for center_index in range(prototype_count):
                assigned_mask = assignments == int(center_index)
                if bool(assigned_mask.any().item()):
                    updated_centers[center_index] = F.normalize(
                        normalized_vectors[assigned_mask].mean(dim=0, keepdim=True),
                        dim=-1,
                        eps=1e-6,
                    ).squeeze(0)
            centers = updated_centers
        similarities = torch.matmul(normalized_vectors, centers.t())
        assignments = similarities.argmax(dim=1)
        occupancy_threshold = max(1, int(self.prototype_split_min_cluster_occupancy))
        kept_centers: List[Tensor] = []
        for center_index in range(prototype_count):
            assigned_mask = assignments == int(center_index)
            assigned_count = int(assigned_mask.sum().item())
            if assigned_count >= occupancy_threshold:
                kept_centers.append(centers[center_index])
        if kept_centers:
            return torch.stack(kept_centers, dim=0)
        densest_center_index = int(torch.bincount(assignments, minlength=prototype_count).argmax().item())
        return centers[densest_center_index].unsqueeze(0)

    def _initialize_spherical_kmeans_centers(
        self,
        normalized_vectors: Tensor,
        *,
        prototype_count: int,
    ) -> Tensor:
        sample_count = int(normalized_vectors.size(0))
        resolved_count = min(sample_count, max(1, int(prototype_count)))
        mean_vector = F.normalize(
            normalized_vectors.mean(dim=0, keepdim=True),
            dim=-1,
            eps=1e-6,
        )
        similarities_to_mean = torch.matmul(
            normalized_vectors,
            mean_vector.squeeze(0),
        )
        first_index = int(similarities_to_mean.argmin().item())
        selected_indices = [first_index]
        while len(selected_indices) < resolved_count:
            selected_vectors = normalized_vectors.index_select(
                0,
                torch.as_tensor(
                    selected_indices,
                    device=normalized_vectors.device,
                    dtype=torch.long,
                ),
            )
            max_selected_similarity = torch.matmul(
                normalized_vectors,
                selected_vectors.t(),
            ).max(dim=1).values
            max_selected_similarity[
                torch.as_tensor(
                    selected_indices,
                    device=normalized_vectors.device,
                    dtype=torch.long,
                )
            ] = 1.0
            next_index = int(max_selected_similarity.argmin().item())
            if next_index in selected_indices:
                break
            selected_indices.append(next_index)
        centers = normalized_vectors.index_select(
            0,
            torch.as_tensor(
                selected_indices,
                device=normalized_vectors.device,
                dtype=torch.long,
            ),
        )
        if int(centers.size(0)) < int(prototype_count):
            padding = centers[-1:].expand(int(prototype_count) - int(centers.size(0)), -1)
            centers = torch.cat([centers, padding], dim=0)
        return centers[:prototype_count]

    def _maybe_refresh_bootstrap_prototype_for_class(self, class_index: int) -> bool:
        resolved_class_index = int(class_index)
        if resolved_class_index <= 0:
            return False
        current_count = int(self.sample_store.class_count(resolved_class_index))
        if current_count <= 0:
            self._mature_prototype_class_ids.discard(resolved_class_index)
            return False
        if resolved_class_index in self._mature_prototype_class_ids:
            return False
        threshold = self._prototype_bootstrap_threshold()
        if self._prototype_learning_bootstrap_completed:
            self._mark_prototype_rebuild_pending(class_ids=[resolved_class_index])
            return False
        if current_count > threshold:
            return False
        return self._rebuild_prototypes_from_samples(class_ids=[resolved_class_index])

    def _maybe_finalize_initial_prototypes(self) -> bool:
        if self._prototype_learning_bootstrap_completed:
            return False
        return self._rebuild_prototypes_from_samples(
            mark_learning_bootstrap_complete=True
        )

    def _encode_current_sample_store_tau_embeddings(self) -> Tensor:
        sample_store_size = int(len(self.sample_store.storage))
        if sample_store_size <= 0:
            return torch.zeros((0, int(self.contrastive_dim)), device=self.device, dtype=torch.float32)

        signature = (
            int(self._dynamics_parameter_version),
            sample_store_size,
        )
        cached = self._sample_store_tau_cache
        if cached is not None and self._sample_store_tau_cache_signature == signature:
            return _clone_if_inference_tensor(cached)

        previous_signature = self._sample_store_tau_cache_signature
        can_append = (
            cached is not None
            and isinstance(previous_signature, tuple)
            and len(previous_signature) == 2
            and int(previous_signature[0]) == int(self._dynamics_parameter_version)
            and int(previous_signature[1]) + 1 == sample_store_size
            and isinstance(self.sample_store.last_insert_index, int)
            and int(self.sample_store.last_insert_index) == sample_store_size - 1
        )
        if can_append:
            tail_embeddings = self._encode_tau_embeddings_for_indices(
                [sample_store_size - 1]
            ).detach()
            tau_embeddings = torch.cat([cached, tail_embeddings], dim=0)
        else:
            storage_indices = list(range(sample_store_size))
            distributed_tau = self._encode_tau_embeddings_for_indices_distributed(
                storage_indices
            )
            if distributed_tau is None:
                tau_embeddings = self._encode_tau_embeddings_for_indices(
                    storage_indices
                ).detach()
            else:
                tau_embeddings = distributed_tau.detach()
        tau_embeddings = _clone_if_inference_tensor(tau_embeddings.detach())
        self._sample_store_tau_cache_signature = signature
        self._sample_store_tau_cache = tau_embeddings
        return tau_embeddings

    def _encode_prototype_sample_tau_embeddings(
        self,
        storage_indices: Sequence[int],
    ) -> Tensor:
        resolved_indices = tuple(int(index) for index in storage_indices)
        if not resolved_indices:
            return torch.zeros((0, int(self.contrastive_dim)), device=self.device, dtype=torch.float32)
        signature = (
            int(self._dynamics_parameter_version),
            int(self._sample_storage_version),
            int(self._sample_label_version),
            int(len(self.sample_store.storage)),
            self._storage_indices_signature(resolved_indices),
        )
        cached = self._prototype_sample_tau_cache
        if (
            cached is not None
            and self._prototype_sample_tau_cache_signature == signature
            and self._prototype_sample_tau_cache_indices == resolved_indices
        ):
            return _clone_if_inference_tensor(cached)

        distributed_tau = self._encode_tau_embeddings_for_indices_distributed(
            resolved_indices
        )
        if distributed_tau is None:
            tau_embeddings = self._encode_tau_embeddings_for_indices(
                resolved_indices
            ).detach()
        else:
            tau_embeddings = distributed_tau.detach()
        tau_embeddings = _clone_if_inference_tensor(tau_embeddings.detach())
        self._prototype_sample_tau_cache_signature = signature
        self._prototype_sample_tau_cache_indices = resolved_indices
        self._prototype_sample_tau_cache = tau_embeddings
        return tau_embeddings

    def _has_current_program_source(self) -> bool:
        return bool(
            isinstance(self._current_program_source, str)
            and self._current_program_source.strip()
        )

    def _action_index_from_name(self, action_name: Any) -> Optional[int]:
        if not isinstance(action_name, str):
            return None
        try:
            return int(self.action_names.index(str(action_name)))
        except ValueError:
            return None

    def _build_contrastive_sample(
        self,
        *,
        state_json: str,
        action: int,
        next_state_json: str,
        done: bool,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        source_world_index: Optional[int] = None,
        source_world_seed: Optional[int] = None,
    ) -> ContrastiveSample:
        # No `_map_reset_count` fallback. That counter is the number of world
        # sessions ever created, not a world index, and `source_world_index` feeds
        # the transition identity key. Leaving it None degrades to `world:unknown`,
        # which misses the protected map and re-derives the assignment - slower,
        # but never mislabelled.
        resolved_source_world_index = _positive_integral(source_world_index)
        resolved_source_world_seed = (
            int(source_world_seed)
            if isinstance(source_world_seed, Integral) and not isinstance(source_world_seed, bool)
            else (
                int(getattr(self.env, "last_reset_seed"))
                if isinstance(getattr(self.env, "last_reset_seed", None), Integral)
                and not isinstance(getattr(self.env, "last_reset_seed", None), bool)
                else None
            )
        )
        resolved_state_id = _positive_integral(state_id)
        resolved_next_state_id = _positive_integral(next_state_id)
        return ContrastiveSample(
            state_json=state_json,
            action=int(action),
            next_state_json=next_state_json,
            done=1.0 if done else 0.0,
            class_id=None,
            state_id=resolved_state_id,
            next_state_id=resolved_next_state_id,
            state_store=self._ensure_state_store(),
            source_env_name=(
                str(getattr(self.env, "env_name", "")).strip() or None
            ),
            source_world_index=resolved_source_world_index,
            source_world_seed=resolved_source_world_seed,
        )

    def _build_contrastive_sample_from_ids(
        self,
        *,
        state_id: int,
        action: int,
        next_state_id: int,
        done: bool,
        source_world_index: Optional[int] = None,
        source_world_seed: Optional[int] = None,
    ) -> ContrastiveSample:
        # No `_map_reset_count` fallback. That counter is the number of world
        # sessions ever created, not a world index, and `source_world_index` feeds
        # the transition identity key. Leaving it None degrades to `world:unknown`,
        # which misses the protected map and re-derives the assignment - slower,
        # but never mislabelled.
        resolved_source_world_index = _positive_integral(source_world_index)
        resolved_source_world_seed = (
            int(source_world_seed)
            if isinstance(source_world_seed, Integral) and not isinstance(source_world_seed, bool)
            else (
                int(getattr(self.env, "last_reset_seed"))
                if isinstance(getattr(self.env, "last_reset_seed", None), Integral)
                and not isinstance(getattr(self.env, "last_reset_seed", None), bool)
                else None
            )
        )
        return ContrastiveSample(
            state_json=None,
            action=int(action),
            next_state_json=None,
            done=1.0 if done else 0.0,
            class_id=None,
            state_id=int(state_id),
            next_state_id=int(next_state_id),
            state_store=self._ensure_state_store(),
            source_env_name=(
                str(getattr(self.env, "env_name", "")).strip() or None
            ),
            source_world_index=resolved_source_world_index,
            source_world_seed=resolved_source_world_seed,
        )

    def _record_observed_sample(self, sample: Optional[ContrastiveSample]) -> None:
        if sample is None:
            return
        state_id = _positive_integral(sample.state_id)
        next_state_id = _positive_integral(sample.next_state_id)
        observed_item = ContrastiveSample(
            state_json=(
                None
                if state_id is not None
                else sample.resolved_state_json()
            ),
            action=int(sample.action),
            next_state_json=(
                None
                if next_state_id is not None
                else sample.resolved_next_state_json()
            ),
            done=float(sample.done),
            class_id=sample.class_id,
            leaf_group_id=sample.leaf_group_id,
            assignment_status=sample.assignment_status,
            state_id=state_id,
            next_state_id=next_state_id,
            state_store=sample.state_store,
            source_env_name=sample.source_env_name,
            source_world_index=sample.source_world_index,
            source_world_seed=sample.source_world_seed,
        )
        if self.sample_store.add(observed_item):
            self._mark_sample_storage_changed()

    def _resolve_state_identity(
        self,
        *,
        state_json: str,
        state_id: Optional[int] = None,
    ) -> str:
        if isinstance(state_id, int) and int(state_id) > 0:
            resolved = self._ensure_state_store().state_key(int(state_id))
            if isinstance(resolved, str) and resolved:
                return resolved
        return str(state_json)

    def _build_transition(
        self,
        *,
        state_json: str,
        action_name: str,
        next_state_json: str,
        reward: float = 0.0,
        done: bool = False,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
    ) -> Transition:
        # `world_index`/`map_name` are what `canonical_graph_world_scope` turns
        # into the `world:N` prefix of the transition identity key. Omitting them
        # yields `world:unknown`, which never matches the pipeline's protected
        # `world:N` keys, so `TransitionGroupClassifier` misses its direct-hit
        # map and falls back to replaying every candidate program per transition.
        if (
            isinstance(state_id, int)
            and int(state_id) > 0
            and isinstance(next_state_id, int)
            and int(next_state_id) > 0
        ):
            return Transition.from_state_ids(
                state_store=self._ensure_state_store(),
                state_id=int(state_id),
                action=action_name,
                next_state_id=int(next_state_id),
                reward=float(reward),
                done=bool(done),
                world_index=world_index,
                map_name=map_name,
            )
        return Transition(
            state=state_json,
            action=action_name,
            next_state=next_state_json,
            reward=float(reward),
            done=bool(done),
            state_store=self._ensure_state_store(),
            state_id=(
                int(state_id)
                if isinstance(state_id, int) and int(state_id) > 0
                else None
            ),
            next_state_id=(
                int(next_state_id)
                if isinstance(next_state_id, int) and int(next_state_id) > 0
                else None
            ),
            world_index=world_index,
            map_name=map_name,
        )

    def _emit_train_progress_event(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        *,
        completed: int,
        total: int,
        updates_completed: int,
        desc: str,
    ) -> None:
        if not callable(progress_callback):
            return
        contrastive_loss = (
            self._last_train_stats.get("contrastive_loss")
            if int(updates_completed) > 0
            else None
        )
        try:
            payload = {
                "progress_kind": "train",
                "train_progress_desc": str(desc).strip() or "Train iter",
                "train_progress_completed": max(0, int(completed)),
                "train_progress_total": max(1, int(total)),
                "train_updates_completed": max(0, int(updates_completed)),
                "train_iter": int(self.total_updates),
                "train_schedule": self._train_schedule_name(),
                "train_phase": self._current_train_phase(),
            }
            if isinstance(contrastive_loss, (int, float)):
                payload["train_contrastive_loss"] = float(contrastive_loss)
            progress_callback(payload)
        except (TypeError, ValueError, RuntimeError, OSError):
            return

    def _apply_live_train_metric_means(
        self,
        stats: Dict[str, float],
    ) -> Dict[str, float]:
        resolved = dict(stats)
        if self._rolling_contrastive_loss.values:
            resolved["contrastive_loss"] = self._rolling_contrastive_loss.mean()
        if self._rolling_prototype_top1_accuracy.values:
            resolved["prototype_top1_accuracy"] = (
                self._rolling_prototype_top1_accuracy.mean()
            )
        return resolved

    def _record_live_train_metric_sample(
        self,
        stats: Dict[str, float],
    ) -> Dict[str, float]:
        if "contrastive_loss" in stats:
            self._rolling_contrastive_loss.append(stats.get("contrastive_loss"))
        if "prototype_top1_accuracy" in stats:
            self._rolling_prototype_top1_accuracy.append(
                stats.get("prototype_top1_accuracy")
            )
        return self._apply_live_train_metric_means(stats)

    def _record_live_reward_metric_sample(
        self,
        *,
        rh: float,
        rz: float,
        rtotal: float,
    ) -> None:
        self._rolling_rh.append(rh)
        self._rolling_rz.append(rz)
        self._rolling_rtotal.append(rtotal)

    def _live_reward_metric_stats(self) -> Dict[str, float]:
        return {
            "rh_mean": self._rolling_rh.mean(),
            "rh_std": self._rolling_rh.std(),
            "rz_mean": self._rolling_rz.mean(),
            "rz_std": self._rolling_rz.std(),
            "rtotal_mean": self._rolling_rtotal.mean(),
            "rtotal_std": self._rolling_rtotal.std(),
        }

    def _run_distributed_train_update_batch(
        self,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        desc: str = "Train iter",
    ) -> Optional[int]:
        if self.contrastive_train_mode != "sample_batch":
            return None
        if not self._can_use_distributed_contrastive_training(
            min(len(self.sample_store), int(self.contrastive_batch_size))
        ):
            return None
        total_requested = max(1, int(self.train_updates_per_step))
        self._emit_train_progress_event(
            progress_callback,
            completed=0,
            total=total_requested,
            updates_completed=0,
            desc=desc,
        )

        index_batches: List[List[Tensor]] = []
        usable_counts: List[int] = []
        for _update_index in range(total_requested):
            rank_indices, usable_count = self._distributed_sample_train_indices()
            if usable_count <= 0 or not rank_indices:
                break
            index_batches.append(rank_indices)
            usable_counts.append(int(usable_count))
        if not index_batches:
            return None
        updates_completed = 0
        last_usable_count = int(usable_counts[0])
        aggregate_samples: List[Dict[str, float]] = []
        try:
            trainer = self._get_distributed_trainer()
            for chunk_start in range(
                0,
                len(index_batches),
                _MAX_DISTRIBUTED_TRAIN_STEPS_PER_REQUEST,
            ):
                chunk_end = min(
                    len(index_batches),
                    chunk_start + _MAX_DISTRIBUTED_TRAIN_STEPS_PER_REQUEST,
                )
                chunk_index_batches = index_batches[chunk_start:chunk_end]
                rank_step_commands = self._distributed_train_step_commands(
                    chunk_index_batches
                )
                result = trainer.train_step_commands(rank_step_commands)
                aggregate_stats = self._apply_distributed_train_result(
                    result,
                    usable_count=int(usable_counts[chunk_end - 1]),
                )
                raw_step_stats = result.get("step_stats")
                step_stats = (
                    list(raw_step_stats)
                    if isinstance(raw_step_stats, list)
                    else []
                )
                reported_updates = max(
                    0,
                    int(result.get("updates_completed", len(step_stats))),
                )
                target_completed = min(
                    len(chunk_index_batches),
                    reported_updates or len(step_stats),
                )
                for chunk_offset in range(target_completed):
                    update_index = chunk_start + chunk_offset
                    raw_stats = (
                        step_stats[chunk_offset]
                        if chunk_offset < len(step_stats)
                        and isinstance(step_stats[chunk_offset], dict)
                        else aggregate_stats
                    )
                    train_stats = self._distributed_train_stats_with_metadata(
                        raw_stats,
                        usable_count=int(usable_counts[update_index]),
                    )
                    if train_stats:
                        self._last_train_stats = self._record_live_train_metric_sample(
                            dict(train_stats)
                        )
                        aggregate_samples.append(dict(raw_stats))
                        last_usable_count = int(usable_counts[update_index])
                        self.total_updates += 1
                        updates_completed += 1

                    self._emit_train_progress_event(
                        progress_callback,
                        completed=update_index + 1,
                        total=total_requested,
                        updates_completed=updates_completed,
                        desc=desc,
                    )
                del rank_step_commands
        except BaseException:
            self._close_distributed_trainer()
            raise

        if updates_completed > 0:
            self._mark_prototype_rebuild_pending()
            aggregate_stats = self._distributed_train_stats_with_metadata(
                _ddp_average_step_stats(aggregate_samples),
                usable_count=int(last_usable_count),
            )
            self._last_train_stats = self._apply_live_train_metric_means(
                aggregate_stats if aggregate_stats else self._last_train_stats
            )
        return int(updates_completed)

    def _run_train_update_batch(
        self,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        desc: str = "Train iter",
    ) -> int:
        self._maybe_finalize_initial_prototypes()
        if self._uses_frozen_initial_representation():
            total_requested = max(1, int(self.train_updates_per_step))
            self._emit_train_progress_event(
                progress_callback,
                completed=0,
                total=total_requested,
                updates_completed=0,
                desc=desc,
            )
            self._last_train_stats = self._frozen_initial_representation_stats()
            self._emit_train_progress_event(
                progress_callback,
                completed=total_requested,
                total=total_requested,
                updates_completed=0,
                desc=desc,
            )
            return 0
        distributed_updates = self._run_distributed_train_update_batch(
            progress_callback=progress_callback,
            desc=desc,
        )
        if distributed_updates is not None:
            return int(distributed_updates)
        self._train_phase_timer.reset_cycle()
        total_requested = max(1, int(self.train_updates_per_step))
        updates_completed = 0
        self._emit_train_progress_event(
            progress_callback,
            completed=0,
            total=total_requested,
            updates_completed=updates_completed,
            desc=desc,
        )
        for update_index in range(total_requested):
            train_stats = self._train_step()
            if train_stats:
                self._last_train_stats = self._record_live_train_metric_sample(
                    dict(train_stats)
                )
                self.total_updates += 1
                updates_completed += 1
            self._emit_train_progress_event(
                progress_callback,
                completed=update_index + 1,
                total=total_requested,
                updates_completed=updates_completed,
                desc=desc,
            )
        if updates_completed > 0:
            self._mark_prototype_rebuild_pending()
            self._last_train_stats = self._apply_live_train_metric_means(
                self._last_train_stats
            )
        self._last_train_phase_timing = self._build_train_phase_timing_payload(
            updates_completed=int(updates_completed),
        )
        return int(updates_completed)

    def _build_train_phase_timing_payload(
        self,
        *,
        updates_completed: int,
    ) -> Dict[str, Any]:
        """Per-cycle phase attribution plus the gauges that test host-pin growth.

        Returns an empty mapping when ``train_phase_timing`` is off, so callers
        can drop the key entirely and nothing reaches the log.
        """
        timer = self._train_phase_timer
        if not timer.enabled:
            return {}
        payload: Dict[str, Any] = {
            "updates": max(0, int(updates_completed)),
            "cuda_sync": bool(timer.cuda_sync),
            "phases": timer.snapshot(),
            # Owner-side gauges: these two now account for every cached state
            # tensor, because the storage-index map holds keys, not tensors.
            "state_token_cache_entries": int(len(self._shared_state_token_cache)),
            "state_token_cache_bytes": int(self._shared_state_token_cache_bytes),
            "state_token_cache_max_entries": int(self.state_token_cache_max_entries),
            "state_token_cache_evictions": int(self._state_token_cache_evictions),
            "state_token_cache_evicted_entries": int(
                self._state_token_cache_evicted_entries
            ),
            # Key-only map; counts resolved storage indices, holds no tensors.
            "sample_storage_token_cache_entries": int(
                len(self._sample_storage_token_cache)
            ),
            # Post-unpin this covers the row-index tensors only, which are the
            # genuine non_blocking H2D sources.
            "pin_memory_calls": int(self._pin_host_tensor_calls),
            "pin_memory_seconds": float(self._pin_host_tensor_seconds),
        }
        batch_pin_calls, batch_pin_seconds = _batch_pin_totals()
        payload["batch_pin_calls"] = int(batch_pin_calls)
        payload["batch_pin_seconds"] = float(batch_pin_seconds)
        if self.device.type == "cuda":
            payload["cuda_memory_allocated_bytes"] = int(
                torch.cuda.memory_allocated(self.device)
            )
            payload["cuda_memory_reserved_bytes"] = int(
                torch.cuda.memory_reserved(self.device)
            )
            host_allocated_bytes = _cuda_host_allocated_bytes()
            if host_allocated_bytes is not None:
                payload["cuda_host_allocated_bytes"] = int(host_allocated_bytes)
        rss_bytes = _process_rss_bytes()
        if rss_bytes is not None:
            payload["process_rss_bytes"] = int(rss_bytes)
        return payload

    def _apply_explained_sample_to_store(
        self,
        sample: ContrastiveSample,
        *,
        assessment: Dict[str, Any],
    ) -> Optional[Tuple[int, Optional[int], bool]]:
        normalized_class_id = (
            int(assessment.get("assigned_class_id"))
            if isinstance(assessment.get("assigned_class_id"), int)
            and int(assessment.get("assigned_class_id")) > 0
            else None
        )
        sample.class_id = (
            int(normalized_class_id)
            if isinstance(normalized_class_id, int) and int(normalized_class_id) > 0
            else None
        )
        sample.leaf_group_id = (
            str(assessment.get("assigned_group_id")).strip()
            if isinstance(assessment.get("assigned_group_id"), str)
            and str(assessment.get("assigned_group_id")).strip()
            else None
        )
        sample.assignment_status = (
            str(assessment.get("assignment_status"))
            if isinstance(assessment.get("assignment_status"), str)
            else ("assigned" if sample.class_id is not None else "unknown")
        )
        existing_sample_index = self.sample_store.merge_exact_without_observe(sample)
        sample_storage_changed = existing_sample_index is None and self.sample_store.add(sample)
        sample_index = self.sample_store.last_insert_index
        if not isinstance(sample_index, int):
            return None
        return (
            int(sample_index),
            (
                int(normalized_class_id)
                if isinstance(normalized_class_id, int)
                and int(normalized_class_id) > 0
                else None
            ),
            bool(sample_storage_changed),
        )

    def _commit_explained_sample(
        self,
        sample: ContrastiveSample,
        *,
        assessment: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        applied = self._apply_explained_sample_to_store(
            sample,
            assessment=assessment,
        )
        if applied is None:
            return None
        sample_index, normalized_class_id, sample_storage_changed = applied
        if sample_storage_changed:
            self._mark_sample_storage_changed()
        if sample_storage_changed and isinstance(sample_index, int):
            self._mark_sample_labels_changed(
                indices=[int(sample_index)],
                class_ids=(
                    [int(normalized_class_id)]
                    if isinstance(normalized_class_id, int)
                    and int(normalized_class_id) > 0
                    else []
                ),
            )
        else:
            self._mark_sample_labels_changed()
        if isinstance(normalized_class_id, int) and int(normalized_class_id) > 0:
            self._maybe_refresh_bootstrap_prototype_for_class(int(normalized_class_id))
        self._last_observed_class_index = (
            int(normalized_class_id)
            if isinstance(normalized_class_id, int) and int(normalized_class_id) > 0
            else None
        )
        return {
            "class_id": (
                int(normalized_class_id)
                if isinstance(normalized_class_id, int) and int(normalized_class_id) > 0
                else 0
            ),
            "assignment": {
                "class_id": (
                    int(normalized_class_id)
                    if isinstance(normalized_class_id, int) and int(normalized_class_id) > 0
                    else 0
                ),
                "group_id": sample.leaf_group_id,
                "status": sample.assignment_status,
            },
            "train_updates_completed": 0,
        }

    def commit_explained_transition(self, transition: Transition) -> Optional[Dict[str, Any]]:
        if not isinstance(transition, Transition):
            return None
        action_index = self._action_index_from_name(transition.action)
        if action_index is None:
            return None
        state_id = _positive_integral(transition.state_id)
        next_state_id = _positive_integral(transition.next_state_id)
        if state_id is not None and next_state_id is not None:
            sample = self._build_contrastive_sample_from_ids(
                state_id=int(state_id),
                action=int(action_index),
                next_state_id=int(next_state_id),
                done=bool(transition.done),
                source_world_index=transition.world_index,
            )
        else:
            sample = self._build_contrastive_sample(
                state_json=transition.state,
                action=int(action_index),
                next_state_json=transition.next_state,
                done=bool(transition.done),
                state_id=state_id,
                next_state_id=next_state_id,
                source_world_index=transition.world_index,
            )
        assessment = self._ensure_sample_current_program_assessment(
            sample,
            action_name=transition.action,
        )
        if not isinstance(assessment, dict):
            return {
                "committed": False,
                "reason": "missing_current_program_assessment",
            }
        if assessment.get("current_explains") is not True:
            return {
                "committed": False,
                "reason": "current_program_does_not_explain",
            }
        commit_result = self._commit_explained_sample(
            sample,
            assessment=assessment,
        )
        if commit_result is None:
            return None
        commit_result["committed"] = True
        return commit_result

    def commit_explained_transitions(
        self,
        transitions: List[Transition],
        *,
        assessments: Mapping[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        committed_count = 0
        skipped_count = 0
        storage_changed = False
        label_indices: List[int] = []
        new_label_indices: List[int] = []
        existing_label_update = False
        label_class_ids: set[int] = set()
        bootstrap_class_ids: List[int] = []
        assessment_by_key = dict(assessments)

        for transition in transitions:
            if not isinstance(transition, Transition):
                skipped_count += 1
                continue
            action_index = self._action_index_from_name(transition.action)
            if action_index is None:
                skipped_count += 1
                continue
            state_id = _positive_integral(transition.state_id)
            next_state_id = _positive_integral(transition.next_state_id)
            if state_id is not None and next_state_id is not None:
                sample = self._build_contrastive_sample_from_ids(
                    state_id=int(state_id),
                    action=int(action_index),
                    next_state_id=int(next_state_id),
                    done=bool(transition.done),
                    source_world_index=transition.world_index,
                )
            else:
                sample = self._build_contrastive_sample(
                    state_json=transition.state,
                    action=int(action_index),
                    next_state_json=transition.next_state,
                    done=bool(transition.done),
                    state_id=state_id,
                    next_state_id=next_state_id,
                    source_world_index=transition.world_index,
                )
            transition_key = self._transition_key_from_fields(
                world_index=transition.world_index,
                map_name=transition.map_name,
                state_identity=sample.resolved_state_key(),
                action_name=transition.action,
                next_state_identity=sample.resolved_next_state_key(),
            )
            assessment = assessment_by_key.get(transition_key)
            if not isinstance(assessment, dict):
                skipped_count += 1
                continue
            if assessment.get("current_explains") is not True:
                skipped_count += 1
                continue
            applied = self._apply_explained_sample_to_store(
                sample,
                assessment=assessment,
            )
            if applied is None:
                skipped_count += 1
                continue
            sample_index, normalized_class_id, sample_storage_changed = applied
            committed_count += 1
            storage_changed = bool(storage_changed or sample_storage_changed)
            label_indices.append(int(sample_index))
            if sample_storage_changed:
                new_label_indices.append(int(sample_index))
            else:
                existing_label_update = True
            if isinstance(normalized_class_id, int) and int(normalized_class_id) > 0:
                label_class_ids.add(int(normalized_class_id))
                bootstrap_class_ids.append(int(normalized_class_id))

        if committed_count <= 0:
            return {
                "committed_count": 0,
                "skipped_count": int(skipped_count),
                "train_updates_completed": 0,
            }
        if storage_changed:
            self._mark_sample_storage_changed()
        self._mark_sample_labels_changed(
            indices=label_indices,
            class_ids=sorted(label_class_ids),
        )
        for class_id in bootstrap_class_ids:
            self._maybe_refresh_bootstrap_prototype_for_class(int(class_id))
        self._last_observed_class_index = (
            int(bootstrap_class_ids[-1]) if bootstrap_class_ids else None
        )
        return {
            "committed_count": int(committed_count),
            "skipped_count": int(skipped_count),
            "train_updates_completed": 0,
        }

    def set_program_context(self, context: Optional[Dict[str, Any]]) -> None:
        previous_version_id = self._current_program_version_id()
        previous_source_digest = self._current_program_source_digest()
        update_result = self._group_classifier.sync_context(
            context=context,
            max_program_count=None,
            previous_snapshot=self._group_context_snapshot,
        )
        self._current_program_source = self._resolve_current_program_source(context)
        self._sync_program_context_snapshot(update_result.snapshot)
        context_changed = bool(
            previous_version_id != self._current_program_version_id()
            or previous_source_digest != self._current_program_source_digest()
        )
        if context_changed or update_result.requires_relabel:
            self._current_source_transition_assessments.clear()
        self._ensure_dynamics_class_capacity(self._required_prototype_capacity())
        relabeled_class_ids: List[int] = []
        if update_result.requires_relabel:
            relabeled_class_ids = self._relabel_samples_after_program_context_update(
                affected_class_ids=update_result.affected_class_ids,
            )
        migrated_class_ids = self._apply_group_class_migrations(update_result.class_migrations)
        if update_result.requires_relabel:
            residual_class_ids = sorted(
                {
                    int(class_index)
                    for class_index in relabeled_class_ids
                    if int(class_index) > 0
                }
                - {
                    int(class_index)
                    for class_index in migrated_class_ids
                    if int(class_index) > 0
                }
            )
            if residual_class_ids:
                self._mark_prototype_rebuild_pending(class_ids=residual_class_ids)

    def _relabel_samples_after_program_context_update(
        self,
        *,
        affected_class_ids: Sequence[int] = (),
    ) -> List[int]:
        if len(self.sample_store) <= 0:
            return []
        affected_class_id_set = {
            int(class_id) for class_id in affected_class_ids if int(class_id) > 0
        }
        if not affected_class_id_set:
            return []
        relabeled_class_id_set: set[int] = set()
        target_indices = sorted(
            {
                int(storage_index)
                for class_id in affected_class_id_set
                for storage_index in self.sample_store._class_to_indices.get(
                    int(class_id),
                    [],
                )
            }
        )
        relabeled_any = False
        for storage_index in target_indices:
            resolved_index = int(storage_index)
            if resolved_index < 0 or resolved_index >= len(self.sample_store.storage):
                continue
            item = self.sample_store.storage[resolved_index]
            current_class_index = self._resolve_item_class_id(item)
            if current_class_index <= 0:
                continue
            if current_class_index not in affected_class_id_set:
                continue
            relabeled_class_id_set.add(int(current_class_index))
            item_state_id = _positive_integral(item.state_id)
            item_next_state_id = _positive_integral(item.next_state_id)
            # Pass the sample's world so the identity key matches the pipeline's
            # protected map. Without it every sample would miss the direct-hit
            # map and re-derive its group by replaying the whole program set.
            #
            # Hitting the map is label-neutral only because the pipeline seeds it
            # from this same re-derivation: `_add_to_protected`
            # (discovery/program_pipeline.py) writes the classifier's own output
            # under the `world:N` key. If that seeding ever stops being the
            # classifier's own output, this stops being a speedup and starts
            # moving labels.
            class_index, assignment, _rows = self._classify_transition(
                state_json=(
                    ""
                    if item_state_id is not None
                    else item.resolved_state_json()
                ),
                action=item.action,
                next_state_json=(
                    ""
                    if item_next_state_id is not None
                    else item.resolved_next_state_json()
                ),
                state_id=item_state_id,
                next_state_id=item_next_state_id,
                world_index=item.source_world_index,
                include_rows=False,
            )
            item.class_id = int(class_index) if int(class_index or 0) > 0 else None
            item.leaf_group_id = assignment.group_id
            item.assignment_status = assignment.status
            if isinstance(item.class_id, int) and int(item.class_id) > 0:
                relabeled_class_id_set.add(int(item.class_id))
            relabeled_any = True
        if relabeled_any:
            self.sample_store.rebuild_class_index()
            self._mark_sample_labels_changed(class_ids=sorted(relabeled_class_id_set))
        return sorted(relabeled_class_id_set)

    def _active_class_indices(self) -> List[int]:
        return sorted(
            int(class_id)
            for class_id in self._group_context_snapshot.active_class_ids
            if int(class_id) > 0
        )

    def _required_prototype_capacity(self) -> int:
        active_class_indices = self._active_class_indices()
        if active_class_indices:
            return int(max(active_class_indices))
        return int(max(1, int(self.known_class_count)))

    def _ensure_dynamics_class_capacity(self, required_count: int) -> None:
        safe_required_count = max(1, int(required_count))
        if safe_required_count <= int(self.dynamics.num_classes):
            self.num_dynamics_classes = max(int(self.dynamics.num_classes), safe_required_count)
            return
        if not self.dynamics.ensure_prototype_capacity(safe_required_count):
            self.num_dynamics_classes = max(int(self.dynamics.num_classes), safe_required_count)
            return
        self.num_dynamics_classes = max(int(self.num_dynamics_classes), int(self.dynamics.num_classes))
        trainer = getattr(self, "_distributed_trainer", None)
        if trainer is not None:
            try:
                trainer.expand_prototype_capacity(int(self.dynamics.num_classes))
                self._sync_distributed_trainer_from_main()
            except BaseException:
                self._close_distributed_trainer()
                raise
        self._mark_prototype_summary_changed()

    def _apply_group_class_migrations(
        self,
        migrations: Sequence[GroupClassMigration],
    ) -> List[int]:
        if not migrations:
            return []
        migrated_class_ids = sorted(
            {
                int(class_id)
                for migration in migrations
                for class_id in (migration.source_class_id, migration.broken_class_id)
                if int(class_id) > 0
            }
        )
        if not migrated_class_ids:
            return []
        self._ensure_dynamics_class_capacity(int(max(migrated_class_ids)))
        self._mark_prototype_rebuild_pending(class_ids=migrated_class_ids)
        return migrated_class_ids

    def _sample_contrastive_batch(self) -> List[ContrastiveSample]:
        return self.sample_store.sample_class_balanced_pairs_labeled(
            self.contrastive_batch_size,
            self.rng,
            minimum_class_count=max(2, int(self.contrastive_min_class_count)),
        )

    @classmethod
    def _resolve_contrastive_train_mode(cls, raw_mode: Any) -> str:
        normalized_mode = str(raw_mode).strip().lower()
        if normalized_mode in cls._SUPPORTED_CONTRASTIVE_TRAIN_MODES:
            return normalized_mode
        supported_modes = ", ".join(cls._SUPPORTED_CONTRASTIVE_TRAIN_MODES)
        raise ValueError(
            "contrastive_train_mode must be one of "
            f"{supported_modes}, got {raw_mode!r}."
        )

    def _full_sample_epoch_contrastive_index_batches(self) -> List[List[int]]:
        return self.sample_store.iter_class_balanced_pair_index_batches(
            self.contrastive_batch_size,
            self.rng,
            minimum_class_count=max(2, int(self.contrastive_min_class_count)),
        )

    def _aggregate_full_sample_epoch_stats(
        self,
        batch_stats: Sequence[Dict[str, float]],
    ) -> Dict[str, float]:
        valid_batch_stats = [dict(stats) for stats in batch_stats if stats]
        if not valid_batch_stats:
            return {}
        batch_weights = [
            max(1.0, float(stats.get("contrastive_epoch_sample_count", 0.0)))
            for stats in valid_batch_stats
        ]
        total_weight = float(sum(batch_weights))
        if total_weight <= 0.0:
            return {}
        additive_keys = {
            "contrastive_eligible_sample_count",
            "contrastive_provisional_sample_count",
            "contrastive_epoch_batch_count",
            "contrastive_epoch_sample_count",
        }
        last_value_keys = {
            "contrastive_batch_size",
            "sample_store_size",
        }
        aggregated: Dict[str, float] = {}
        for key in valid_batch_stats[0].keys():
            if key in additive_keys:
                aggregated[key] = float(
                    sum(float(stats.get(key, 0.0)) for stats in valid_batch_stats)
                )
                continue
            if key == "contrastive_batch_size":
                aggregated[key] = float(self.contrastive_batch_size)
                continue
            if key in last_value_keys:
                aggregated[key] = float(valid_batch_stats[-1].get(key, 0.0))
                continue
            aggregated[key] = float(
                sum(
                    float(stats.get(key, 0.0)) * weight
                    for stats, weight in zip(valid_batch_stats, batch_weights)
                )
                / total_weight
            )
        return aggregated

    def _current_train_phase(self) -> str:
        if self._uses_frozen_initial_representation():
            return "frozen_initial"
        return "train" if self.total_updates > 0 else "warmup"

    def _train_schedule_name(self) -> str:
        return "post_transition_batch"


    def _classify_transition(
        self,
        *,
        state_json: str,
        action: int,
        next_state_json: str,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
        include_rows: bool = True,
    ) -> Tuple[Optional[int], Any, List[Dict[str, Any]]]:
        action_name = self.action_names[int(action)]
        transition = self._build_transition(
            state_json=state_json,
            action_name=action_name,
            next_state_json=next_state_json,
            reward=0.0,
            done=False,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=world_index,
            map_name=map_name,
        )
        assignment, rows = self._group_classifier.classify_transition(
            transition=transition,
            include_rows=include_rows,
        )
        class_index = int(assignment.class_id) if int(assignment.class_id) > 0 else None
        return class_index, assignment, rows

    def _current_program_version_id(self) -> Optional[str]:
        current_version_id = self._group_context_snapshot.version_snapshot.current_version_id
        if isinstance(current_version_id, str) and current_version_id.strip():
            return current_version_id.strip()
        return None

    def _resolve_current_program_source(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(context, dict):
            return None
        raw_current_source = context.get("current_source")
        if isinstance(raw_current_source, str) and raw_current_source.strip():
            return str(raw_current_source)
        current_version_id = context.get("current_version_id")
        raw_versions = context.get("versions")
        if not isinstance(current_version_id, str) or not current_version_id.strip():
            return None
        if not isinstance(raw_versions, list):
            return None
        for raw_version in raw_versions:
            if not isinstance(raw_version, dict):
                continue
            version_id = raw_version.get("version_id")
            version_source = raw_version.get("source")
            if (
                isinstance(version_id, str)
                and version_id.strip() == current_version_id.strip()
                and isinstance(version_source, str)
                and version_source.strip()
            ):
                return str(version_source)
        return None

    def _current_program_source_digest(self) -> Optional[str]:
        if not isinstance(self._current_program_source, str) or not self._current_program_source:
            return None
        return hashlib.sha1(self._current_program_source.encode("utf-8")).hexdigest()

    def _clone_sandbox_error(
        self,
        error: Optional[SandboxError],
    ) -> Optional[SandboxError]:
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

    def _transition_key_from_fields(
        self,
        *,
        world_index: Any = None,
        map_name: Any = None,
        state_identity: str,
        action_name: str,
        next_state_identity: str,
    ) -> str:
        del next_state_identity
        return canonical_graph_edge_identity_key_from_fields(
            world_index=world_index,
            map_name=map_name,
            state=state_identity,
            action=action_name,
        )

    def _current_transition_assessment(
        self,
        *,
        state_json: str,
        action_name: str,
        next_state_json: str,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
    ) -> Optional[Dict[str, Any]]:
        transition_key = self._transition_key_from_fields(
            world_index=world_index,
            map_name=map_name,
            state_identity=self._resolve_state_identity(
                state_json=state_json,
                state_id=state_id,
            ),
            action_name=action_name,
            next_state_identity=self._resolve_state_identity(
                state_json=next_state_json,
                state_id=next_state_id,
            ),
        )
        assessment = self._current_source_transition_assessments.get(transition_key)
        if not isinstance(assessment, dict):
            return None
        if assessment.get("current_version_id") != self._current_program_version_id():
            return None
        if assessment.get("current_source_digest") != self._current_program_source_digest():
            return None
        return assessment

    def _clear_sample_assignment(self, sample: Optional[ContrastiveSample]) -> None:
        if sample is None:
            return
        sample.class_id = None
        sample.leaf_group_id = None
        sample.assignment_status = "unknown"

    def _apply_assignment_to_sample(
        self,
        sample: Optional[ContrastiveSample],
        *,
        class_id: Optional[int],
        assignment: Any,
    ) -> None:
        if sample is None:
            return
        normalized_class_id = (
            int(class_id)
            if isinstance(class_id, int) and int(class_id) > 0
            else None
        )
        if normalized_class_id is None:
            self._clear_sample_assignment(sample)
            if isinstance(getattr(assignment, "status", None), str):
                sample.assignment_status = str(assignment.status)
            return
        sample.class_id = int(normalized_class_id)
        sample.leaf_group_id = (
            str(assignment.group_id).strip()
            if isinstance(getattr(assignment, "group_id", None), str)
            and str(assignment.group_id).strip()
            else None
        )
        sample.assignment_status = (
            str(assignment.status)
            if isinstance(getattr(assignment, "status", None), str)
            else "assigned"
        )

    def _resolve_transition_assignment(
        self,
        *,
        state_json: str,
        action: int,
        next_state_json: str,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
        current_transition_explains: Optional[bool],
        include_rows: bool,
        fallback_rows: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[int], Any, List[Dict[str, Any]]]:
        if isinstance(current_transition_explains, bool) and not bool(
            current_transition_explains
        ):
            return (
                None,
                None,
                [dict(row) for row in fallback_rows] if fallback_rows is not None else [],
            )
        class_id, assignment, rows = self._classify_transition(
            state_json=state_json,
            action=action,
            next_state_json=next_state_json,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=world_index,
            map_name=map_name,
            include_rows=include_rows,
        )
        normalized_class_id = (
            int(class_id) if isinstance(class_id, int) and int(class_id) > 0 else None
        )
        return normalized_class_id, assignment, rows

    def _assess_transition_against_current_program(
        self,
        *,
        state_json: str,
        action: int,
        next_state_json: str,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
    ) -> Dict[str, Any]:
        action_name = self.action_names[int(action)]
        transition_key = self._transition_key_from_fields(
            world_index=world_index,
            map_name=map_name,
            state_identity=self._resolve_state_identity(
                state_json=state_json,
                state_id=state_id,
            ),
            action_name=action_name,
            next_state_identity=self._resolve_state_identity(
                state_json=next_state_json,
                state_id=next_state_id,
            ),
        )
        assessment: Dict[str, Any] = {
            "transition_key": transition_key,
            "current_version_id": self._current_program_version_id(),
            "current_source_digest": self._current_program_source_digest(),
            "current_explains": False,
            "predicted_next_state_json": None,
            "prediction_error": None,
        }
        if not isinstance(self._current_program_source, str) or not self._current_program_source.strip():
            self._current_source_transition_assessments[transition_key] = dict(assessment)
            return assessment

        transition = self._build_transition(
            state_json=state_json,
            action_name=action_name,
            next_state_json=next_state_json,
            reward=0.0,
            done=False,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=world_index,
            map_name=map_name,
        )
        record = self._program_evaluator.evaluate_transition(
            source=self._current_program_source,
            transition=transition,
        )
        assessment["current_explains"] = bool(record.is_correct) and record.error is None
        assessment["predicted_next_state_json"] = record.predicted_canonical
        assessment["prediction_error"] = self._clone_sandbox_error(record.error)
        self._current_source_transition_assessments[transition_key] = dict(assessment)
        return assessment

    def _refresh_transition_current_program_assessment(
        self,
        *,
        state_json: str,
        action: int,
        action_name: str,
        next_state_json: str,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
        sample: Optional[ContrastiveSample] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved_world_index = (
            world_index
            if world_index is not None
            else (sample.source_world_index if sample is not None else None)
        )
        assessment = self._assess_transition_against_current_program(
            state_json=state_json,
            action=action,
            next_state_json=next_state_json,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=resolved_world_index,
            map_name=map_name,
        )
        class_id = None
        assignment = None
        if assessment.get("current_explains") is True:
            class_id, assignment, _rows = self._resolve_transition_assignment(
                state_json=state_json,
                action=action,
                next_state_json=next_state_json,
                state_id=state_id,
                next_state_id=next_state_id,
                world_index=resolved_world_index,
                map_name=map_name,
                current_transition_explains=True,
                include_rows=False,
            )
        merged = self._merge_current_transition_assessment(
            state_json=state_json,
            action_name=action_name,
            next_state_json=next_state_json,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=resolved_world_index,
            map_name=map_name,
            class_id=class_id,
            assignment=assignment,
            rows=None,
        )
        self._apply_assignment_to_sample(
            sample,
            class_id=class_id,
            assignment=assignment,
        )
        return merged if isinstance(merged, dict) else dict(assessment)

    def _ensure_sample_current_program_assessment(
        self,
        sample: ContrastiveSample,
        *,
        action_name: str,
    ) -> Optional[Dict[str, Any]]:
        state_id = _positive_integral(sample.state_id)
        next_state_id = _positive_integral(sample.next_state_id)
        state_json = "" if state_id is not None else sample.resolved_state_json()
        next_state_json = (
            ""
            if next_state_id is not None
            else sample.resolved_next_state_json()
        )
        assessment = self._current_transition_assessment(
            state_json=state_json,
            action_name=action_name,
            next_state_json=next_state_json,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=sample.source_world_index,
        )
        if isinstance(assessment, dict):
            return assessment
        try:
            action_index = int(sample.action)
        except (TypeError, ValueError):
            return None
        return self._refresh_transition_current_program_assessment(
            state_json=state_json,
            action=action_index,
            action_name=action_name,
            next_state_json=next_state_json,
            state_id=state_id,
            next_state_id=next_state_id,
            world_index=sample.source_world_index,
            sample=sample,
        )

    def refresh_current_source_transition_assessments(
        self,
        transitions: Optional[List[Transition]] = None,
    ) -> None:
        if not isinstance(transitions, list):
            return
        for transition in transitions:
            if not isinstance(transition, Transition):
                continue
            action_index = self._action_index_from_name(transition.action)
            if action_index is None:
                continue
            state_id = _positive_integral(transition.state_id)
            next_state_id = _positive_integral(transition.next_state_id)
            refresh_kwargs: Dict[str, Any] = {
                "state_json": "" if state_id is not None else transition.state,
                "action": int(action_index),
                "action_name": transition.action,
                "next_state_json": (
                    ""
                    if next_state_id is not None
                    else transition.next_state
                ),
                "state_id": state_id,
                "next_state_id": next_state_id,
                "sample": None,
            }
            if transition.world_index is not None:
                refresh_kwargs["world_index"] = transition.world_index
            if transition.map_name is not None:
                refresh_kwargs["map_name"] = transition.map_name
            self._refresh_transition_current_program_assessment(**refresh_kwargs)

    def _build_class_rows(
        self,
        *,
        visible_count: int,
        current_class_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._group_classifier.build_class_rows(
            current_class_id=current_class_id,
        )
        limit = max(0, int(visible_count))
        if limit > 0:
            return rows[:limit]
        return rows

    def _annotate_class_counts(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        return [
            self._annotate_class_count(dict(row))
            for row in rows
        ]

    def _annotate_class_count(self, row: Dict[str, Any]) -> Dict[str, Any]:
        annotated = dict(row)
        class_id = (
            int(annotated.get("version_index"))
            if isinstance(annotated.get("version_index"), int)
            and int(annotated.get("version_index")) > 0
            else None
        )
        if class_id is None:
            return annotated
        annotated["transition_count"] = int(self.sample_store.class_count(class_id))
        return annotated

    def _resolve_group_display_metadata(
        self,
        *,
        class_id: Optional[int],
        group_id: Optional[str],
        rows: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        del rows
        return self._group_classifier.resolve_group_display_metadata(
            class_id=(
                int(class_id)
                if isinstance(class_id, int) and int(class_id) > 0
                else None
            ),
            group_id=group_id,
        )

    def _resolve_transition_display_metadata(
        self,
        *,
        class_id: Optional[int],
        assignment: Any,
        rows: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        assignment_group_id = (
            str(assignment.group_id).strip()
            if isinstance(getattr(assignment, "group_id", None), str)
            and str(assignment.group_id).strip()
            else None
        )
        return self._resolve_group_display_metadata(
            class_id=(int(class_id) if isinstance(class_id, int) else None),
            group_id=assignment_group_id,
            rows=rows,
        )

    def _merge_current_transition_assessment(
        self,
        *,
        state_json: str,
        action_name: str,
        next_state_json: str,
        state_id: Optional[int] = None,
        next_state_id: Optional[int] = None,
        world_index: Any = None,
        map_name: Any = None,
        class_id: Optional[int],
        assignment: Any,
        rows: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        transition_key = self._transition_key_from_fields(
            world_index=world_index,
            map_name=map_name,
            state_identity=self._resolve_state_identity(
                state_json=state_json,
                state_id=state_id,
            ),
            action_name=action_name,
            next_state_identity=self._resolve_state_identity(
                state_json=next_state_json,
                state_id=next_state_id,
            ),
        )
        assessment = self._current_source_transition_assessments.get(transition_key)
        if not isinstance(assessment, dict):
            return None

        if assessment.get("current_explains") is False:
            assessment["assigned_class_id"] = 0
            assessment["assigned_group_id"] = None
            assessment["assigned_group_label"] = None
            assessment["assigned_class_count"] = None
            assessment["assignment_status"] = "unknown"
            assessment["is_new_dynamics_class"] = True
            self._current_source_transition_assessments[transition_key] = dict(assessment)
            return dict(assessment)

        metadata = self._resolve_transition_display_metadata(
            class_id=class_id,
            assignment=assignment,
            rows=rows,
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
        return dict(assessment)

    def _contrastive_class_counts(self) -> Dict[int, int]:
        return self.sample_store.class_counts(include_zero=False)

    def _trainable_contrastive_class_indices(self) -> List[int]:
        minimum_count = max(2, int(self.contrastive_min_class_count))
        return sorted(
            int(class_index)
            for class_index, count in self._contrastive_class_counts().items()
            if int(class_index) > 0 and int(count) >= minimum_count
        )

    def _provisional_contrastive_class_indices(self) -> List[int]:
        minimum_count = max(2, int(self.contrastive_min_class_count))
        if minimum_count <= 2:
            return []
        return sorted(
            int(class_index)
            for class_index, count in self._contrastive_class_counts().items()
            if int(class_index) > 0 and 0 < int(count) < minimum_count
        )

    def _count_self_information_for_existing_classes(self, counts: Tensor) -> Tensor:
        resolved_counts = counts.reshape(-1).to(dtype=torch.float32)
        if resolved_counts.numel() <= 0:
            return torch.zeros_like(resolved_counts)
        safe_counts = torch.clamp(resolved_counts, min=0.0)
        total_count = safe_counts.sum()
        if not bool(torch.isfinite(total_count).item()) or float(total_count.item()) <= 0.0:
            return torch.zeros_like(safe_counts)
        probs = safe_counts / total_count
        surprisal = -torch.log(probs.clamp_min(float(self.prototype_entropy_eps)))
        return torch.nan_to_num(surprisal, nan=0.0, posinf=0.0, neginf=0.0)

    def _prototype_entropy_prediction_probs(self, logits: Tensor) -> Tensor:
        logits_fp32 = logits.to(dtype=torch.float32)
        if logits_fp32.ndim != 2 or int(logits_fp32.size(0)) <= 0 or int(logits_fp32.size(1)) <= 0:
            return torch.zeros_like(logits_fp32)
        temperature_ratio = max(
            1e-4,
            float(self.prototype_entropy_temperature) / max(1e-4, float(self.contrastive_temperature)),
        )
        return torch.softmax(logits_fp32 / float(temperature_ratio), dim=-1)

    def _expected_prototype_entropy_reward_from_logits(
        self,
        logits: Tensor,
        *,
        context: ContrastiveEntropyContext,
    ) -> Tensor:
        logits_fp32 = logits.to(dtype=torch.float32)
        batch_size = int(logits_fp32.size(0)) if logits_fp32.ndim > 0 else 0
        if (
            logits_fp32.ndim != 2
            or batch_size <= 0
            or int(logits_fp32.size(1)) != int(context.class_indices.numel())
        ):
            return torch.zeros((batch_size,), device=logits_fp32.device, dtype=torch.float32)
        probs = self._prototype_entropy_prediction_probs(logits_fp32)
        return torch.matmul(
            probs,
            context.bonus_by_position.to(device=logits_fp32.device, dtype=torch.float32),
        )

    def _pack_sample_state_batch(
        self,
        items: Sequence[ContrastiveSample],
        *,
        next_state: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        encoded_items = [
            self._sample_state_tokens(item, next_state=next_state)
            for item in items
        ]
        return self._pack_cached_state_batch(encoded_items)

    def consume_current_source_transition_assessments(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(key): dict(value)
            for key, value in self._current_source_transition_assessments.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def _normalize_active_class_id(self, raw_class_id: Any) -> int:
        if isinstance(raw_class_id, bool):
            return 0
        if not isinstance(raw_class_id, int):
            return 0
        class_index = int(raw_class_id)
        if class_index <= 0:
            return 0
        if class_index not in self._active_class_id_set:
            return 0
        return int(class_index)

    def _resolve_item_class_id(self, item: ContrastiveSample) -> int:
        normalized_class_id = int(self._normalize_active_class_id(item.class_id))
        return int(normalized_class_id) if normalized_class_id > 0 else 0

    def _build_prototype_entropy_context(self) -> Optional[ContrastiveEntropyContext]:
        if self.prototype_entropy_scale <= 0.0:
            return None
        active_class_ids = tuple(
            int(class_id)
            for class_id in self._group_context_snapshot.active_class_ids
            if int(class_id) > 0
        )
        mature_filter_ids: Optional[Tuple[int, ...]] = None
        if (
            self.prototype_sample_min_per_class is not None
            and self._prototype_learning_bootstrap_completed
        ):
            mature_filter_ids = tuple(
                sorted(int(class_id) for class_id in self._mature_prototype_class_ids)
            )
        cache_signature = (
            int(self._sample_label_version),
            active_class_ids,
            mature_filter_ids,
            int(self.dynamics.num_classes),
            float(self.prototype_entropy_scale),
            int(self.prototype_entropy_min_class_count),
        )
        if cache_signature == self._prototype_entropy_context_cache_signature:
            return self._prototype_entropy_context_cache
        class_counts = self.sample_store.class_counts(include_zero=False)
        if not class_counts:
            self._prototype_entropy_context_cache_signature = cache_signature
            self._prototype_entropy_context_cache = None
            return None
        active_class_id_set = set(active_class_ids)
        mature_filter_set = set(mature_filter_ids) if mature_filter_ids is not None else None
        ordered_class_indices = sorted(
            int(class_index)
            for class_index, count in class_counts.items()
            if int(class_index) > 0
            and int(class_index) in active_class_id_set
            and (
                mature_filter_set is None
                or int(class_index) in mature_filter_set
            )
            and int(count) > 0
            and int(class_index) <= int(self.dynamics.num_classes)
        )
        if not ordered_class_indices:
            self._prototype_entropy_context_cache_signature = cache_signature
            self._prototype_entropy_context_cache = None
            return None
        class_index_tensor = torch.as_tensor(
            ordered_class_indices,
            device=self.device,
            dtype=torch.long,
        )
        counts_tensor = torch.as_tensor(
            [int(class_counts.get(int(class_index), 0)) for class_index in ordered_class_indices],
            device=self.device,
            dtype=torch.float32,
        )
        eligible_mask = counts_tensor >= float(max(1, int(self.prototype_entropy_min_class_count)))
        label_to_position = torch.full(
            (int(self.dynamics.num_classes) + 1,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        label_to_position[class_index_tensor] = torch.arange(
            class_index_tensor.numel(),
            device=self.device,
            dtype=torch.long,
        )
        total_count = counts_tensor.sum()
        if not bool(torch.isfinite(total_count).item()) or float(total_count.item()) <= 0.0:
            self._prototype_entropy_context_cache_signature = cache_signature
            self._prototype_entropy_context_cache = None
            return None
        bonus = self._count_self_information_for_existing_classes(counts_tensor)
        bonus = torch.where(
            eligible_mask,
            bonus,
            torch.zeros_like(bonus),
        )
        context = ContrastiveEntropyContext(
            label_to_position=label_to_position,
            bonus_by_position=bonus,
            class_indices=class_index_tensor,
            class_counts=counts_tensor,
        )
        self._prototype_entropy_context_cache_signature = cache_signature
        self._prototype_entropy_context_cache = context
        return context

    def _prototype_entropy_reward_for_labels(
        self,
        labels_one_based: Tensor,
        *,
        context: ContrastiveEntropyContext,
    ) -> Tensor:
        labels = labels_one_based.reshape(-1).to(device=self.device, dtype=torch.long)
        if labels.numel() <= 0 or self.prototype_entropy_scale <= 0.0:
            return torch.zeros((int(labels.numel()),), device=self.device, dtype=torch.float32)
        reward = torch.zeros((int(labels.numel()),), device=self.device, dtype=torch.float32)
        valid_label_mask = (labels > 0) & (labels < context.label_to_position.shape[0])
        if not bool(valid_label_mask.any().item()):
            return reward
        valid_positions = context.label_to_position[labels[valid_label_mask]]
        matched_mask = valid_positions >= 0
        if not bool(matched_mask.any().item()):
            return reward
        valid_positions = valid_positions[matched_mask]
        destination_indices = torch.nonzero(valid_label_mask, as_tuple=False).reshape(-1)[matched_mask]
        reward[destination_indices] = context.bonus_by_position.index_select(0, valid_positions)
        return reward

    def _particle_entropy_intrinsic_reward(
        self,
        source_keys: Tensor,
        *,
        target_keys: Optional[Tensor] = None,
        self_target_indices: Optional[Tensor] = None,
        exclude_self: Optional[bool] = None,
    ) -> Tensor:
        source_size = int(source_keys.shape[0]) if source_keys.ndim > 0 else 0
        if source_keys.ndim != 2 or source_size <= 0:
            return torch.zeros((source_size,), device=source_keys.device, dtype=torch.float32)

        source = source_keys
        target = source if target_keys is None else target_keys
        target_size = int(target.size(0))
        if target.ndim != 2 or target_size <= 0:
            return torch.zeros((source_size,), device=source.device, dtype=torch.float32)
        if int(target.size(1)) != int(source.size(1)):
            return torch.zeros((source_size,), device=source.device, dtype=torch.float32)

        sim_matrix = torch.cdist(source, target, p=2.0)
        exclude_self = self.knn_exclude_self if exclude_self is None else bool(exclude_self)
        self_exclusion_applied = False
        if bool(exclude_self):
            if self_target_indices is None and target is source:
                self_target_indices = torch.arange(source_size, device=sim_matrix.device, dtype=torch.long)
            if self_target_indices is not None and target_size > 0:
                source_row_indices = torch.arange(source_size, device=sim_matrix.device, dtype=torch.long)
                target_col_indices = self_target_indices.to(device=sim_matrix.device, dtype=torch.long).reshape(-1)
                valid_mask = (
                    target_col_indices.shape[0] == source_row_indices.shape[0]
                    and bool((target_col_indices >= 0).all().item())
                    and bool((target_col_indices < target_size).all().item())
                )
                if valid_mask:
                    sim_matrix[source_row_indices, target_col_indices] = float("inf")
                    self_exclusion_applied = True

        neighbor_cap = int(target_size) - (1 if self_exclusion_applied else 0)
        if neighbor_cap <= 0:
            return torch.zeros((source_size,), device=source.device, dtype=torch.float32)
        k_eff = min(max(1, int(self.knn_k)), int(neighbor_cap))
        reward = torch.topk(sim_matrix, k=k_eff, dim=1, largest=False, sorted=True).values
        if not self.knn_avg:
            reward = reward[:, -1].reshape(-1, 1)
            if self.knn_clip >= 0.0:
                reward = torch.maximum(reward - float(self.knn_clip), torch.zeros_like(reward))
        else:
            reward = reward.reshape(-1, 1)
            if self.knn_clip >= 0.0:
                reward = torch.maximum(reward - float(self.knn_clip), torch.zeros_like(reward))
            reward = reward.reshape((source_size, k_eff))
            reward = reward.mean(dim=1, keepdim=True)
        reward = torch.log(reward + 1.0).reshape(-1)
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _train_on_contrastive_indices(
        self,
        contrastive_indices: Sequence[int],
    ) -> Dict[str, float]:
        resolved_indices = [int(index) for index in contrastive_indices]
        if not resolved_indices:
            return {}
        timer = self._train_phase_timer
        keys, command = self._encode_sample_keys_and_command_from_indices(resolved_indices)
        label_started_at = timer.mark()
        # The step command already carries the per-row class labels and class
        # support counts computed from the very same storage rows, so they are
        # reused here instead of walking the sample store a second time.
        host_labels = command["labels"]
        host_support_counts = command["support_counts"]
        labels_one_based = host_labels.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        timer.record("label", label_started_at)
        contrastive_stats = self._contrastive_batch_update(
            keys=keys,
            labels_one_based=labels_one_based,
            host_labels=host_labels,
            host_support_counts=host_support_counts,
        )
        if not contrastive_stats:
            return {}
        merged_stats: Dict[str, float] = dict(contrastive_stats)
        merged_stats["contrastive_batch_size"] = float(len(resolved_indices))
        merged_stats["contrastive_epoch_batch_count"] = 1.0
        merged_stats["contrastive_epoch_sample_count"] = float(len(resolved_indices))
        merged_stats["sample_store_size"] = float(len(self.sample_store))
        return merged_stats

    def _train_step(self) -> Dict[str, float]:
        if self._uses_frozen_initial_representation():
            self._last_train_stats = self._frozen_initial_representation_stats()
            return {}
        if len(self.sample_store) < 2:
            return {}
        timer = self._train_phase_timer
        timer.begin_step()
        step_started_at = timer.mark()
        try:
            return self._run_contrastive_train_step()
        finally:
            timer.record("step", step_started_at)
            timer.end_step()

    def _run_contrastive_train_step(self) -> Dict[str, float]:
        timer = self._train_phase_timer
        if self.contrastive_train_mode == "full_sample_epoch":
            sample_started_at = timer.mark()
            contrastive_index_batches = self._full_sample_epoch_contrastive_index_batches()
            timer.record("sample", sample_started_at)
        else:
            sample_started_at = timer.mark()
            contrastive_indices = self._sample_contrastive_batch_indices()
            timer.record("sample", sample_started_at)
            if contrastive_indices:
                return self._train_on_contrastive_indices(contrastive_indices)
            contrastive_index_batches = []
        if not contrastive_index_batches:
            return {}
        batch_stats = [
            self._train_on_contrastive_indices(batch_indices)
            for batch_indices in contrastive_index_batches
            if batch_indices
        ]
        batch_stats = [stats for stats in batch_stats if stats]
        if not batch_stats:
            return {}
        if len(batch_stats) == 1:
            merged_stats = dict(batch_stats[0])
        else:
            merged_stats = self._aggregate_full_sample_epoch_stats(batch_stats)
        return merged_stats

    def _tau_encode_entity_sorted_rows(
        self,
        storage_indices: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        """Resolve the tau-encode rows and the ascending entity-count visit order.

        Returns ``(resolved_indices, order)``. ``resolved_indices`` drops the
        out-of-range storage indices exactly the way
        ``_sample_step_command_from_indices`` drops them, so one encoded row comes
        back per entry. ``order`` is a permutation of
        ``range(len(resolved_indices))`` sorted by ``(entity_size, position)`` —
        the same total order plus positional tiebreak that
        ``_token_budget_state_bucket_rows`` uses, so it is deterministic and never
        depends on dict or set iteration.

        The tau encode chunks its request before bucketing, which confines every
        bucket to one chunk-sized window of *arrival* order. Feeding the chunk
        loop this order instead widens the window to the whole request: rows of
        similar entity count become adjacent, so a chunk's buckets stay near their
        padding-ratio floor. The chunk size itself is unchanged, which is what
        bounds the host memory one step command pins.

        The caller therefore encodes rows out of order and MUST scatter each
        chunk's output back to its ``order`` positions (``index_copy_``). Skipping
        the scatter does not raise; it silently returns every row's embedding on
        somebody else's row.

        Resolving tokens here is not extra work: the encode pass that follows
        needs the very same tensors, and ``state_token_cache_max_entries`` is 0,
        which makes ``_evict_state_token_cache_if_needed`` a no-op, so nothing
        this pass caches can be evicted before the encode pass reads it.
        """
        storage = self.sample_store.storage
        resolved_indices: List[int] = []
        entity_sizes: List[int] = []
        for raw_index in storage_indices:
            storage_index = int(raw_index)
            if not 0 <= storage_index < len(storage):
                continue
            tokens, _mask = self._sample_state_tokens_for_storage_index(
                storage_index,
                storage[storage_index],
            )
            resolved_indices.append(storage_index)
            entity_sizes.append(max(1, int(tokens.size(0))))
        order = sorted(
            range(len(resolved_indices)),
            key=lambda position: (int(entity_sizes[position]), int(position)),
        )
        return resolved_indices, order

    def _encode_tau_embeddings_for_indices_distributed(
        self,
        storage_indices: Sequence[int],
    ) -> Optional[Tensor]:
        resolved_indices = [int(index) for index in storage_indices]
        if not resolved_indices:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        if not self._can_use_distributed_tau_encoding(len(resolved_indices)):
            return None
        resolved_indices, order = self._tau_encode_entity_sorted_rows(resolved_indices)
        if not resolved_indices:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        encoded: Optional[Tensor] = None
        try:
            trainer = self._get_distributed_trainer()
            chunk_size = self._distributed_tau_encode_chunk_size(len(resolved_indices))
            with torch.no_grad():
                for start_index in range(0, len(order), chunk_size):
                    order_chunk = order[start_index : start_index + chunk_size]
                    if not order_chunk:
                        continue
                    rank_commands = self._distributed_tau_step_commands(
                        [resolved_indices[position] for position in order_chunk],
                    )
                    # Rank shards are contiguous slices of `order_chunk` and come
                    # back concatenated in rank order, so chunk row `i` is
                    # `order_chunk[i]`.
                    chunk_encoded = _clone_if_inference_tensor(
                        trainer.encode_sample_tau_step_commands(rank_commands)
                        .to(device=self.device, dtype=torch.float32)
                        .detach()
                    )
                    if encoded is None:
                        encoded = torch.empty(
                            (len(resolved_indices), int(chunk_encoded.size(-1))),
                            device=chunk_encoded.device,
                            dtype=chunk_encoded.dtype,
                        )
                    encoded.index_copy_(
                        0,
                        torch.as_tensor(
                            order_chunk,
                            device=chunk_encoded.device,
                            dtype=torch.long,
                        ),
                        chunk_encoded,
                    )
        except BaseException:
            self._close_distributed_trainer()
            raise
        if encoded is None:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        return _clone_if_inference_tensor(encoded.detach())

    def _distributed_tau_step_commands(
        self,
        storage_indices: Sequence[int],
    ) -> List[Dict[str, Any]]:
        resolved_indices = [int(index) for index in storage_indices]
        world_size = int(len(self.parallel_device_ids))
        share_memory = _distributed_step_command_should_share_tensors()
        rank_commands: List[Dict[str, Any]] = []
        for rank in range(world_size):
            shard_start = (len(resolved_indices) * int(rank)) // int(world_size)
            shard_end = (len(resolved_indices) * (int(rank) + 1)) // int(world_size)
            shard_indices = resolved_indices[shard_start:shard_end]
            rank_commands.append(
                self._sample_step_command_from_indices(
                    shard_indices,
                    share_memory=bool(share_memory and int(rank) > 0),
                    pin_memory=bool(self.device.type == "cuda" and int(rank) == 0),
                )
            )
        return rank_commands

    def _encode_tau_embeddings_for_indices(
        self,
        storage_indices: Sequence[int],
    ) -> Tensor:
        resolved_indices = [int(index) for index in storage_indices]
        if not resolved_indices:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        resolved_indices, order = self._tau_encode_entity_sorted_rows(resolved_indices)
        if not resolved_indices:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        chunk_size = max(1, int(self._TAU_REPLAY_ENCODE_MAX_SAMPLES_PER_BATCH))
        encoded: Optional[Tensor] = None
        with torch.no_grad():
            for start_index in range(0, len(order), chunk_size):
                order_chunk = order[start_index : start_index + chunk_size]
                if not order_chunk:
                    continue
                chunk_encoded = _clone_if_inference_tensor(
                    self._encode_samples_as_keys_from_indices(
                        [resolved_indices[position] for position in order_chunk],
                    ).detach()
                )
                if encoded is None:
                    encoded = torch.empty(
                        (len(resolved_indices), int(chunk_encoded.size(-1))),
                        device=chunk_encoded.device,
                        dtype=chunk_encoded.dtype,
                    )
                # `chunk_encoded` row `i` is `order_chunk[i]`; this puts it back on
                # the row the caller asked for.
                encoded.index_copy_(
                    0,
                    torch.as_tensor(
                        order_chunk,
                        device=chunk_encoded.device,
                        dtype=torch.long,
                    ),
                    chunk_encoded,
                )
        if encoded is None:
            return torch.zeros(
                (0, int(self.contrastive_dim)),
                device=self.device,
                dtype=torch.float32,
            )
        return _clone_if_inference_tensor(encoded.detach())

    def _epoch_end_train_stats_defaults(self) -> Dict[str, float]:
        return {
            "prototype_top1_accuracy": 0.0,
            "mean_positive_logit": 0.0,
            "mean_max_negative_logit": 0.0,
            "prototype_entropy_active_class_count": 0.0,
            "rh_mean": 0.0,
            "rh_std": 0.0,
            "rz_mean": 0.0,
            "rz_std": 0.0,
            "rtotal_mean": 0.0,
            "rtotal_std": 0.0,
        }

    #: Width of ``SupconBatchMetrics.similarity_means`` inside the batched
    #: stats transfer: mean positive similarity, mean negative similarity.
    _SUPCON_METRIC_WIDTH = 2

    def _supervised_contrastive_loss(
        self,
        embeddings: Tensor,
        labels_one_based: Tensor,
    ) -> Tuple[Tensor, Optional[SupconBatchMetrics]]:
        """Return the SupCon loss plus its batch metrics.

        The two mean-similarity scalars stay on the device so the caller can
        fold them into a single stats transfer. ``None`` means "no active
        anchor", which the caller reports as all-zero metrics.
        """
        matmul_embeddings = (
            embeddings
            if embeddings.device.type == "cuda"
            and embeddings.dtype in (torch.float16, torch.bfloat16)
            else embeddings.to(dtype=torch.float32)
        )
        with torch.autocast(
            device_type="cuda",
            dtype=matmul_embeddings.dtype,
            enabled=matmul_embeddings.device.type == "cuda"
            and matmul_embeddings.dtype in (torch.float16, torch.bfloat16),
        ):
            similarity_matrix = torch.matmul(
                matmul_embeddings,
                matmul_embeddings.transpose(0, 1),
            )
        similarity_matrix = similarity_matrix.to(dtype=torch.float32)
        logits = similarity_matrix / float(self.contrastive_temperature)
        self_mask = torch.eye(
            int(logits.size(0)),
            device=logits.device,
            dtype=torch.bool,
        )
        same_label_mask = labels_one_based.unsqueeze(1).eq(labels_one_based.unsqueeze(0))
        positive_mask = same_label_mask & ~self_mask
        positive_counts = positive_mask.sum(dim=1)
        active_anchor_mask = positive_counts > 0
        # The only synchronization left in this function. It has to happen: the
        # active-anchor count decides whether a real loss tensor (and therefore a
        # real optimizer step) exists at all. The positive-pair count rides along
        # in the same transfer instead of paying for a second one, and stays a
        # Python float divisor so the mean-similarity kernels are unchanged.
        active_anchor_count, positive_pair_count = (
            int(value)
            for value in torch.stack((active_anchor_mask.sum(), positive_mask.sum()))
            .to(device="cpu", dtype=torch.long)
            .tolist()
        )
        if active_anchor_count <= 0:
            return torch.zeros((), device=logits.device, dtype=torch.float32), None

        masked_logits = logits.masked_fill(self_mask, float("-inf"))
        log_prob = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
        positive_log_prob_sum = log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        loss_per_anchor = -positive_log_prob_sum / positive_counts.clamp_min(1).to(dtype=torch.float32)
        loss = loss_per_anchor[active_anchor_mask].mean()

        positive_similarity_sum = similarity_matrix.masked_fill(~positive_mask, 0.0).sum()
        offdiag_similarity_sum = similarity_matrix.masked_fill(self_mask, 0.0).sum()
        offdiag_pair_count = int(logits.size(0)) * max(0, int(logits.size(0)) - 1)
        negative_pair_count = max(0, int(offdiag_pair_count) - int(positive_pair_count))
        negative_similarity_sum = offdiag_similarity_sum - positive_similarity_sum
        similarity_means = torch.stack(
            (
                positive_similarity_sum / float(max(1, positive_pair_count)),
                negative_similarity_sum / float(max(1, negative_pair_count)),
            )
        ).detach()
        return loss, SupconBatchMetrics(
            active_anchor_count=int(active_anchor_count),
            positive_pair_count=int(positive_pair_count),
            negative_pair_count=int(negative_pair_count),
            similarity_means=similarity_means,
        )

    def _prototype_eval_metrics(
        self,
        embeddings: Tensor,
        labels_one_based: Tensor,
        *,
        class_indices_zero_based: Sequence[int],
        trainable_class_indices: Sequence[int],
        chunk_size: Optional[int] = None,
    ) -> Optional[Tuple[Tensor, int]]:
        """Return ``(metric_sums, total_count)`` with the sums still on device.

        The caller divides the sums by ``total_count`` on the host after the
        single batched transfer, which is exactly the arithmetic this function
        used to perform itself. ``None`` means "no eligible row", which the
        caller reports as all-zero prototype metrics.
        """
        if embeddings.numel() <= 0 or labels_one_based.numel() <= 0 or not class_indices_zero_based:
            return None
        label_lookup = torch.full(
            (int(self.dynamics.num_classes) + 1,),
            -1,
            device=labels_one_based.device,
            dtype=torch.long,
        )
        trainable_label_tensor = torch.as_tensor(
            [int(class_index) for class_index in trainable_class_indices],
            device=labels_one_based.device,
            dtype=torch.long,
        )
        label_lookup[trainable_label_tensor] = torch.arange(
            trainable_label_tensor.numel(),
            device=labels_one_based.device,
            dtype=torch.long,
        )
        lookup_size = int(label_lookup.numel())
        valid_label_mask = (labels_one_based > 0) & (labels_one_based < lookup_size)
        # Gather through a clamped index so out-of-range labels stay in bounds;
        # they are discarded by ``valid_label_mask`` right afterwards. Fusing the
        # two staged boolean selections into one index list turns three device
        # synchronizations into the single one that ``nonzero`` needs.
        remapped_labels = label_lookup[labels_one_based.clamp(0, max(0, lookup_size - 1))]
        selected_mask = valid_label_mask & (remapped_labels >= 0)
        selected_rows = torch.nonzero(selected_mask, as_tuple=False).reshape(-1)
        total_count = int(selected_rows.numel())
        if total_count <= 0:
            return None
        selected_embeddings = embeddings.index_select(0, selected_rows)
        labels = remapped_labels.index_select(0, selected_rows)
        resolved_chunk_size = (
            total_count
            if chunk_size is None
            else max(1, int(chunk_size))
        )
        metric_sums = torch.zeros((3,), device=selected_embeddings.device, dtype=torch.float64)
        with torch.no_grad():
            for start_index in range(0, total_count, resolved_chunk_size):
                end_index = min(total_count, start_index + resolved_chunk_size)
                chunk_embeddings = selected_embeddings[start_index:end_index]
                chunk_labels = labels[start_index:end_index]
                logits = self.dynamics.contrastive_logits(
                    keys=chunk_embeddings,
                    class_indices=class_indices_zero_based,
                ).to(dtype=torch.float32)
                predictions = logits.argmax(dim=-1)
                metric_sums[0] += (predictions == chunk_labels).sum().to(dtype=torch.float64)
                positive_logits = logits[
                    torch.arange(chunk_labels.size(0), device=chunk_labels.device),
                    chunk_labels,
                ]
                metric_sums[1] += positive_logits.sum().to(dtype=torch.float64)
                if int(logits.size(1)) > 1:
                    top_values, top_indices = torch.topk(
                        logits,
                        k=2,
                        dim=-1,
                        largest=True,
                        sorted=True,
                    )
                    max_negative_logits = torch.where(
                        top_indices[:, 0] == chunk_labels,
                        top_values[:, 1],
                        top_values[:, 0],
                    )
                    metric_sums[2] += max_negative_logits.sum().to(dtype=torch.float64)
        return metric_sums.detach(), total_count

    def _frozen_initial_representation_stats(self) -> Dict[str, float]:
        stats = self._epoch_end_train_stats_defaults()
        stats.update(
            {
                "contrastive_loss": 0.0,
                "contrastive_min_class_count": float(self.contrastive_min_class_count),
                "contrastive_trainable_class_count": float(
                    len(self._trainable_contrastive_class_indices())
                ),
                "contrastive_provisional_class_count": float(
                    len(self._provisional_contrastive_class_indices())
                ),
                "contrastive_eligible_sample_count": 0.0,
                "contrastive_provisional_sample_count": 0.0,
                "supcon_active_anchor_count": 0.0,
                "supcon_positive_pair_count": 0.0,
                "supcon_mean_positive_similarity": 0.0,
                "supcon_mean_negative_similarity": 0.0,
                "contrastive_training_disabled": 1.0,
            }
        )
        return self._apply_live_train_metric_means(stats)

    def _contrastive_batch_update(
        self,
        *,
        keys: Tensor,
        labels_one_based: Tensor,
        host_labels: Tensor,
        host_support_counts: Tensor,
    ) -> Dict[str, float]:
        timer = self._train_phase_timer
        # ``host_labels``/``host_support_counts`` are the CPU tensors the step
        # command already built, so eligibility is decided entirely on the host:
        # no device round trip and no synchronization to pick the batch rows.
        valid_label_mask = host_labels > 0
        eligible_label_mask = valid_label_mask & (
            host_support_counts >= max(2, int(self.contrastive_min_class_count))
        )
        provisional_label_mask = valid_label_mask & ~eligible_label_mask
        eligible_sample_count = int(eligible_label_mask.sum())
        provisional_sample_count = int(provisional_label_mask.sum())
        trainable_class_indices = self._trainable_contrastive_class_indices()

        if self._uses_frozen_initial_representation():
            return {
                "contrastive_loss": 0.0,
                "contrastive_min_class_count": float(self.contrastive_min_class_count),
                "contrastive_trainable_class_count": float(len(trainable_class_indices)),
                "contrastive_provisional_class_count": float(
                    len(self._provisional_contrastive_class_indices())
                ),
                "contrastive_eligible_sample_count": float(eligible_sample_count),
                "contrastive_provisional_sample_count": float(provisional_sample_count),
                "prototype_top1_accuracy": 0.0,
                "mean_positive_logit": 0.0,
                "mean_max_negative_logit": 0.0,
                "supcon_active_anchor_count": 0.0,
                "supcon_positive_pair_count": 0.0,
                "supcon_mean_positive_similarity": 0.0,
                "supcon_mean_negative_similarity": 0.0,
                "contrastive_training_disabled": 1.0,
            }

        supcon_metrics: Optional[SupconBatchMetrics] = None
        prototype_metric_sums: Optional[Tensor] = None
        prototype_total_count = 0
        if eligible_sample_count > 0:
            h2d_started_at = timer.mark()
            eligible_rows = (
                torch.nonzero(eligible_label_mask, as_tuple=False)
                .reshape(-1)
                .to(device=self.device, dtype=torch.long, non_blocking=True)
            )
            timer.record("h2d", h2d_started_at)
            # ``index_select`` with the host-computed row list selects the same
            # rows in the same order as boolean masking, without the device-side
            # ``nonzero`` that boolean masking would have to synchronize on.
            eligible_keys = keys.index_select(0, eligible_rows)
            eligible_labels = labels_one_based.index_select(0, eligible_rows)
            loss_started_at = timer.mark()
            contrastive_loss, supcon_metrics = self._supervised_contrastive_loss(
                eligible_keys,
                eligible_labels,
            )
            self._optimizer_backward_step(
                optimizer=self.contrastive_optimizer,
                loss=contrastive_loss,
            )
            timer.record("loss", loss_started_at)
            self._mark_dynamics_parameters_changed()
            prototype_started_at = timer.mark()
            prototype_metrics = self._prototype_eval_metrics(
                eligible_keys.detach(),
                eligible_labels,
                class_indices_zero_based=[
                    int(class_index) - 1
                    for class_index in trainable_class_indices
                ],
                trainable_class_indices=trainable_class_indices,
            )
            if prototype_metrics is not None:
                prototype_metric_sums, prototype_total_count = prototype_metrics
            timer.record("proto", prototype_started_at)
        else:
            contrastive_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        # Every remaining reported scalar still lives on the device; concatenate
        # them so the whole update pays exactly one host transfer.
        sync_started_at = timer.mark()
        scalar_parts: List[Tensor] = [
            contrastive_loss.detach().reshape(1).to(dtype=torch.float64)
        ]
        if supcon_metrics is not None:
            scalar_parts.append(supcon_metrics.similarity_means.to(dtype=torch.float64))
        if prototype_metric_sums is not None:
            scalar_parts.append(prototype_metric_sums.to(dtype=torch.float64))
        transferred_scalars: List[float] = (
            torch.cat(scalar_parts).to(device="cpu", dtype=torch.float64).tolist()
        )
        timer.record("sync", sync_started_at)

        cursor = 1
        supcon_active_anchor_count = 0.0
        supcon_positive_pair_count = 0.0
        supcon_mean_positive_similarity = 0.0
        supcon_mean_negative_similarity = 0.0
        if supcon_metrics is not None:
            supcon_active_anchor_count = float(supcon_metrics.active_anchor_count)
            supcon_positive_pair_count = float(supcon_metrics.positive_pair_count)
            if supcon_metrics.positive_pair_count > 0:
                supcon_mean_positive_similarity = float(transferred_scalars[cursor])
            if supcon_metrics.negative_pair_count > 0:
                supcon_mean_negative_similarity = float(transferred_scalars[cursor + 1])
            cursor += int(self._SUPCON_METRIC_WIDTH)

        prototype_top1_accuracy = 0.0
        mean_positive_logit = 0.0
        mean_max_negative_logit = 0.0
        if prototype_metric_sums is not None:
            total_count = float(prototype_total_count)
            prototype_top1_accuracy = float(transferred_scalars[cursor] / total_count)
            mean_positive_logit = float(transferred_scalars[cursor + 1] / total_count)
            mean_max_negative_logit = float(transferred_scalars[cursor + 2] / total_count)

        return {
            "contrastive_loss": float(transferred_scalars[0]),
            "contrastive_min_class_count": float(self.contrastive_min_class_count),
            "contrastive_trainable_class_count": float(len(trainable_class_indices)),
            "contrastive_provisional_class_count": float(len(self._provisional_contrastive_class_indices())),
            "contrastive_eligible_sample_count": float(eligible_sample_count),
            "contrastive_provisional_sample_count": float(provisional_sample_count),
            "prototype_top1_accuracy": prototype_top1_accuracy,
            "mean_positive_logit": mean_positive_logit,
            "mean_max_negative_logit": mean_max_negative_logit,
            "supcon_active_anchor_count": supcon_active_anchor_count,
            "supcon_positive_pair_count": supcon_positive_pair_count,
            "supcon_mean_positive_similarity": supcon_mean_positive_similarity,
            "supcon_mean_negative_similarity": supcon_mean_negative_similarity,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "name": str(self.strategy_name),
            "collection_topology": self.collection_topology,
            "transition_batch_scope": self.transition_batch_scope,
            "current_version_id": self._current_program_version_id(),
            "seed": int(self.seed),
            "device": str(self.device),
            "visible_cuda_device_count": int(self.visible_cuda_device_count),
            "parallel_device_ids": [int(device_id) for device_id in self.parallel_device_ids],
            "encode_bucket_mode": str(self.encode_bucket_mode),
            "compile_encoder": bool(self.compile_encoder),
            "compile_enabled": bool(self.compile_enabled),
            "compile_disabled_reason": self.compile_disabled_reason,
            "compile_worker_disabled_reasons": {
                int(rank): str(reason)
                for rank, reason in self._worker_compile_disabled_reasons().items()
            },
            "dynamics_encoder_embed_dim": int(self.dynamics_encoder_embed_dim),
            "dynamics_encoder_num_heads": int(self.dynamics_encoder_num_heads),
            "dynamics_encoder_num_blocks": int(self.dynamics_encoder_num_blocks),
            "dynamics_pool_seeds": int(self.dynamics_pool_seeds),
            "dynamics_encoder_dropout": float(self.dynamics_encoder_dropout),
            "total_steps": int(self.total_steps),
            "total_updates": int(self.total_updates),
            "runtime_versions": self._runtime_version_state(),
            "train_schedule": self._train_schedule_name(),
            "contrastive_batch_size": int(self.contrastive_batch_size),
            "contrastive_train_mode": str(self.contrastive_train_mode),
            "contrastive_representation_mode": str(self.contrastive_representation_mode),
            "contrastive_relation_learning_enabled": not bool(
                self._uses_frozen_initial_representation()
            ),
            "contrastive_sampler": "class_balanced_pairs",
            "train_phase": self._current_train_phase(),
            "warmup_active": bool(self._current_train_phase() == "warmup"),
            "map_reset_count": int(self._map_reset_count),
            "sample_store_size": int(len(self.sample_store)),
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
            "canonical_dynamics_classes": int(
                self._group_classifier.canonical_class_count
            ),
            "canonical_classified_transitions": int(
                self._group_classifier.canonical_classified_count
            ),
            "canonical_unassigned_transitions": int(
                self._group_classifier.canonical_unassigned_count
            ),
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
            "last_train": dict(self._last_train_stats),
        }
