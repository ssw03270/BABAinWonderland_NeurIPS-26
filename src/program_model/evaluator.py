from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

from src.data import StateStore, Transition

from .predictor import ProgramWorldModelPredictor
from .sandbox import ProgramSandbox, SandboxConfig, SandboxError
from .state_codec import dump_state_json, states_equivalent


_K = TypeVar("_K")
_V = TypeVar("_V")
DEFAULT_PREDICTION_CACHE_MAX_ENTRIES = 16384
DEFAULT_EVALUATION_CACHE_MAX_ENTRIES = 32768
DEFAULT_EXPECTED_PAYLOAD_CACHE_MAX_ENTRIES = 16384
DEFAULT_PROCESS_CHUNK_TIMEOUT_SEC = 1800.0


class _LRUCache(Generic[_K, _V]):
    def __init__(self, max_entries: Optional[int]) -> None:
        self.max_entries = None if max_entries is None else max(0, int(max_entries))
        self._entries: "OrderedDict[_K, _V]" = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def get(self, key: _K) -> Optional[_V]:
        if key not in self._entries:
            return None
        value = self._entries.pop(key)
        self._entries[key] = value
        return value

    def __setitem__(self, key: _K, value: _V) -> None:
        if self.max_entries == 0:
            return
        if key in self._entries:
            self._entries.pop(key)
        self._entries[key] = value
        if self.max_entries is None:
            return
        while len(self._entries) > int(self.max_entries):
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return int(len(self._entries))


@dataclass
class PredictionRecord:
    index: int
    transition: Transition
    expected_canonical: str
    predicted_canonical: Optional[str]
    is_correct: bool
    error: Optional[SandboxError]


@dataclass
class ProgramEvaluation:
    accuracy: float
    correct_count: int
    total_count: int
    runtime_error_count: int
    compile_errors: List[SandboxError] = field(default_factory=list)
    records: List[PredictionRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ProgramEvaluationTask:
    label: str
    source: str


@dataclass
class ProgramEvaluationResult:
    label: str
    evaluation: ProgramEvaluation
    explains: Optional[Tuple[bool, ...]] = None


@dataclass
class ProgramEvaluationBatch:
    labels: Tuple[str, ...]
    by_label: Dict[str, ProgramEvaluationResult]

    def first(self) -> ProgramEvaluationResult:
        if not self.labels:
            raise ValueError("Program evaluation batch is empty.")
        return self.by_label[self.labels[0]]


@dataclass
class ProgramExplainResult:
    label: str
    explains: Tuple[bool, ...]


@dataclass
class ProgramExplainBatch:
    labels: Tuple[str, ...]
    by_label: Dict[str, ProgramExplainResult]

    def first(self) -> ProgramExplainResult:
        if not self.labels:
            raise ValueError("Program explain batch is empty.")
        return self.by_label[self.labels[0]]


@dataclass(frozen=True)
class VersionContextSnapshot:
    version_ids: Tuple[str, ...] = ()
    active_version_count: int = 0
    current_version_id: Optional[str] = None


@dataclass(frozen=True)
class VersionContextUpdateResult:
    snapshot: VersionContextSnapshot
    source_changed: bool
    version_ids_changed: bool
    active_version_count_changed: bool


@dataclass(frozen=True)
class _SourceHandle:
    source_key: str
    source: str
    predictor: ProgramWorldModelPredictor
    compile_errors: Tuple[SandboxError, ...]


@dataclass(frozen=True)
class _ExpectedStatePayload:
    parsed_state: Optional[Dict[str, Any]]
    canonical: str


@dataclass(frozen=True)
class _CachedPrediction:
    predicted_canonical: Optional[str]
    predicted_state: Optional[Dict[str, Any]]
    error: Optional[SandboxError]


@dataclass(frozen=True)
class _CachedEvaluation:
    expected_canonical: str
    predicted_canonical: Optional[str]
    is_correct: bool
    error: Optional[SandboxError]


@dataclass(frozen=True)
class _ProcessChunkRequest:
    source: str
    start_index: int
    collect_records: bool
    explains_only: bool = False
    failed_records_only: bool = False
    required_record_indices: Tuple[int, ...] = ()
    transition_rows: Tuple[Dict[str, Any], ...] = ()
    transition_refs: Tuple["_ProcessTransitionRef", ...] = ()
    transition_indices: Tuple[int, ...] = ()
    state_store_snapshot_dir: Optional[str] = None


@dataclass(frozen=True)
class _ProcessTransitionRef:
    state_id: int
    next_state_id: int
    action: str
    reward: float
    done: bool
    world_index: Optional[int]
    map_name: Optional[str]


@dataclass(frozen=True)
class _ProcessChunkRecord:
    expected_canonical: str
    predicted_canonical: Optional[str]
    is_correct: bool
    error: Optional[SandboxError]
    local_index: Optional[int] = None


@dataclass(frozen=True)
class _ProcessChunkResult:
    start_index: int
    total_count: int
    correct_count: int
    runtime_error_count: int
    compile_errors: Tuple[SandboxError, ...]
    records: Tuple[_ProcessChunkRecord, ...]
    explains: Tuple[bool, ...] = ()
    explain_indices: Tuple[int, ...] = ()


class ProgramEvaluator:
    def __init__(
        self,
        sandbox: Optional[ProgramSandbox] = None,
        *,
        sandbox_config: Optional[SandboxConfig] = None,
        program_eval_workers: Any = "auto",
        prediction_cache_max_entries: Optional[int] = DEFAULT_PREDICTION_CACHE_MAX_ENTRIES,
        evaluation_cache_max_entries: Optional[int] = DEFAULT_EVALUATION_CACHE_MAX_ENTRIES,
        expected_payload_cache_max_entries: Optional[int] = DEFAULT_EXPECTED_PAYLOAD_CACHE_MAX_ENTRIES,
        process_chunk_timeout_sec: Optional[float] = DEFAULT_PROCESS_CHUNK_TIMEOUT_SEC,
    ) -> None:
        if sandbox is not None and sandbox_config is not None:
            raise ValueError("Provide either sandbox or sandbox_config, not both.")
        self.sandbox = sandbox or ProgramSandbox(config=sandbox_config or SandboxConfig())
        self.program_eval_workers = self._resolve_program_eval_workers(
            program_eval_workers
        )
        self.process_chunk_timeout_sec = self._resolve_process_chunk_timeout_sec(
            process_chunk_timeout_sec
        )
        self._source_key_by_source: Dict[str, str] = {}
        self._source_handle_by_key: Dict[str, _SourceHandle] = {}
        self._source_key_by_program_id: Dict[str, str] = {}
        self._prediction_cache: _LRUCache[
            Tuple[str, str, str],
            _CachedPrediction,
        ] = _LRUCache(prediction_cache_max_entries)
        self._evaluation_cache: _LRUCache[
            Tuple[str, str, str, str],
            _CachedEvaluation,
        ] = _LRUCache(evaluation_cache_max_entries)
        self._expected_payload_cache: _LRUCache[
            str,
            _ExpectedStatePayload,
        ] = _LRUCache(expected_payload_cache_max_entries)
        self._process_chunk_executor: Optional[ProcessPoolExecutor] = None
        self._process_chunk_worker_count: Optional[int] = None
        self._process_chunk_snapshot_dir: Optional[str] = None
        self._process_chunk_snapshot_store_id: Optional[int] = None
        self._process_chunk_snapshot_state_count: int = 0
        self._process_chunk_stale_snapshot_dirs: List[str] = []

    @staticmethod
    def _resolve_program_eval_workers(value: Any) -> int:
        if value is None:
            value = "auto"
        if isinstance(value, str):
            text = value.strip().lower()
            if text == "auto":
                return max(1, int(os.cpu_count() or 1))
            value = text
        if isinstance(value, bool):
            raise ValueError("program_eval_workers must be a positive integer or 'auto'.")
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "program_eval_workers must be a positive integer or 'auto'."
            ) from exc
        if resolved <= 0:
            raise ValueError("program_eval_workers must be a positive integer or 'auto'.")
        return int(resolved)

    @staticmethod
    def _resolve_process_chunk_timeout_sec(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("process_chunk_timeout_sec must be a positive number or None.")
        try:
            resolved = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "process_chunk_timeout_sec must be a positive number or None."
            ) from exc
        if resolved <= 0.0:
            raise ValueError("process_chunk_timeout_sec must be a positive number or None.")
        return float(resolved)

    def _register_program(self, program_id: str, source: str) -> bool:
        safe_program_id = str(program_id).strip()
        if not safe_program_id:
            raise ValueError("program_id must be a non-empty string.")
        handle = self._get_or_create_source_handle(str(source))
        existing_key = self._source_key_by_program_id.get(safe_program_id)
        if existing_key == handle.source_key:
            return False
        self._source_key_by_program_id[safe_program_id] = handle.source_key
        return True

    def _unregister_program(self, program_id: str) -> None:
        safe_program_id = str(program_id).strip()
        if not safe_program_id:
            return
        self._source_key_by_program_id.pop(safe_program_id, None)

    def _retain_programs(self, program_ids: Sequence[str]) -> None:
        keep_ids = {
            str(program_id).strip()
            for program_id in program_ids
            if str(program_id).strip()
        }
        for program_id in list(self._source_key_by_program_id.keys()):
            if program_id not in keep_ids:
                self._unregister_program(program_id)

    def _get_registered_source(self, program_id: str) -> Optional[str]:
        handle = self._get_registered_handle(program_id)
        if handle is None:
            return None
        return handle.source

    def evaluate_source(
        self,
        source: str,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        *,
        collect_records: bool = True,
    ) -> ProgramEvaluation:
        batch = self.evaluate_programs(
            programs=[
                ProgramEvaluationTask(
                    label="source",
                    source=str(source),
                )
            ],
            transitions=transitions,
            progress_callback=progress_callback,
            collect_records=collect_records,
        )
        return batch.first().evaluation

    def evaluate_programs(
        self,
        *,
        programs: Sequence[ProgramEvaluationTask],
        transitions: Sequence[Transition],
        transition_chunk_size: Optional[int] = None,
        program_transition_indices: Optional[Mapping[str, Sequence[int]]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        collect_records: bool = True,
        failed_records_only: bool = False,
        required_record_indices: Sequence[int] = (),
    ) -> ProgramEvaluationBatch:
        safe_programs = self._normalize_program_tasks(programs)
        safe_transitions = list(transitions)
        if not safe_programs:
            return ProgramEvaluationBatch(labels=(), by_label={})

        configured_worker_count = max(1, int(self.program_eval_workers))
        labels = tuple(program.label for program in safe_programs)
        indices_by_label = self._normalize_program_transition_indices(
            labels=labels,
            transition_count=len(safe_transitions),
            program_transition_indices=program_transition_indices,
        )
        if not safe_transitions:
            by_label = {
                program.label: ProgramEvaluationResult(
                    label=program.label,
                    evaluation=ProgramEvaluation(
                        accuracy=0.0,
                        correct_count=0,
                        total_count=0,
                        runtime_error_count=0,
                        records=[] if collect_records else [],
                    ),
                    explains=() if collect_records and not failed_records_only else None,
                )
                for program in safe_programs
            }
            return ProgramEvaluationBatch(labels=labels, by_label=by_label)

        evaluation_cell_count = sum(
            len(indices_by_label.get(program.label, ()))
            for program in safe_programs
        )
        use_process_pool = (
            configured_worker_count > 1
            and evaluation_cell_count > 0
        )
        if not use_process_pool:
            return self._evaluate_programs_serial(
                programs=safe_programs,
                transitions=safe_transitions,
                progress_callback=progress_callback,
                collect_records=collect_records,
                failed_records_only=failed_records_only,
                required_record_indices=required_record_indices,
                indices_by_label=indices_by_label,
            )

        resolved_chunk_size = (
            max(1, int(transition_chunk_size))
            if transition_chunk_size is not None
            else self._default_process_chunk_size(
                total_count=evaluation_cell_count,
                worker_count=configured_worker_count,
            )
        )
        return self._evaluate_programs_process_chunks(
            programs=safe_programs,
            transitions=safe_transitions,
            chunk_size=resolved_chunk_size,
            progress_callback=progress_callback,
            collect_records=collect_records,
            failed_records_only=failed_records_only,
            required_record_indices=required_record_indices,
            indices_by_label=indices_by_label,
        )

    def evaluate_program_explains(
        self,
        *,
        programs: Sequence[ProgramEvaluationTask],
        transitions: Sequence[Transition],
        transition_chunk_size: Optional[int] = None,
        program_transition_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> ProgramExplainBatch:
        safe_programs = self._normalize_program_tasks(programs)
        safe_transitions = list(transitions)
        labels = tuple(program.label for program in safe_programs)
        if not safe_programs:
            return ProgramExplainBatch(labels=(), by_label={})

        indices_by_label = self._normalize_program_transition_indices(
            labels=labels,
            transition_count=len(safe_transitions),
            program_transition_indices=program_transition_indices,
        )
        if not safe_transitions:
            return ProgramExplainBatch(
                labels=labels,
                by_label={
                    program.label: ProgramExplainResult(
                        label=program.label,
                        explains=(),
                    )
                    for program in safe_programs
                },
            )

        configured_worker_count = max(1, int(self.program_eval_workers))
        evaluation_cell_count = sum(
            len(indices_by_label.get(program.label, ()))
            for program in safe_programs
        )
        use_process_pool = (
            configured_worker_count > 1
            and evaluation_cell_count > 0
        )
        if not use_process_pool:
            return self._evaluate_program_explains_serial(
                programs=safe_programs,
                transitions=safe_transitions,
                indices_by_label=indices_by_label,
            )

        resolved_chunk_size = (
            max(1, int(transition_chunk_size))
            if transition_chunk_size is not None
            else self._default_process_chunk_size(
                total_count=evaluation_cell_count,
                worker_count=configured_worker_count,
            )
        )
        batch = self._evaluate_programs_process_chunks(
            programs=safe_programs,
            transitions=safe_transitions,
            chunk_size=resolved_chunk_size,
            progress_callback=None,
            collect_records=False,
            failed_records_only=False,
            required_record_indices=(),
            indices_by_label=indices_by_label,
            explains_only=True,
        )
        return ProgramExplainBatch(
            labels=batch.labels,
            by_label={
                label: ProgramExplainResult(
                    label=label,
                    explains=tuple(result.explains or ()),
                )
                for label, result in batch.by_label.items()
            },
        )

    def _normalize_program_tasks(
        self,
        programs: Sequence[ProgramEvaluationTask],
    ) -> Tuple[ProgramEvaluationTask, ...]:
        normalized: List[ProgramEvaluationTask] = []
        seen_labels: set[str] = set()
        for raw_program in programs:
            if not isinstance(raw_program, ProgramEvaluationTask):
                raise TypeError("programs must contain ProgramEvaluationTask entries.")
            label = str(raw_program.label).strip()
            source = raw_program.source
            if not label:
                raise ValueError("program label must be a non-empty string.")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"program {label!r} source must be non-empty.")
            if label in seen_labels:
                raise ValueError(f"duplicate program label: {label}")
            seen_labels.add(label)
            normalized.append(
                ProgramEvaluationTask(
                    label=label,
                    source=str(source),
                )
            )
        return tuple(normalized)

    @staticmethod
    def _normalize_program_transition_indices(
        *,
        labels: Sequence[str],
        transition_count: int,
        program_transition_indices: Optional[Mapping[str, Sequence[int]]],
    ) -> Dict[str, Tuple[int, ...]]:
        all_indices = tuple(range(max(0, int(transition_count))))
        if program_transition_indices is None:
            return {str(label): all_indices for label in labels}

        by_label: Dict[str, Tuple[int, ...]] = {}
        for label in labels:
            raw_indices = program_transition_indices.get(str(label))
            if raw_indices is None:
                by_label[str(label)] = all_indices
                continue
            seen: set[int] = set()
            normalized: List[int] = []
            for raw_index in raw_indices:
                if isinstance(raw_index, bool):
                    raise ValueError("program transition indices must be integers.")
                index = int(raw_index)
                if index < 0 or index >= int(transition_count):
                    raise ValueError(
                        f"program transition index {index} is out of range."
                    )
                if index in seen:
                    continue
                seen.add(index)
                normalized.append(index)
            by_label[str(label)] = tuple(normalized)
        return by_label

    def _evaluate_programs_serial(
        self,
        *,
        programs: Sequence[ProgramEvaluationTask],
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        collect_records: bool,
        failed_records_only: bool,
        required_record_indices: Sequence[int],
        indices_by_label: Mapping[str, Tuple[int, ...]],
    ) -> ProgramEvaluationBatch:
        labels = tuple(program.label for program in programs)
        by_label: Dict[str, ProgramEvaluationResult] = {}
        completed_count = 0
        total_count = sum(
            len(indices_by_label.get(program.label, ()))
            for program in programs
        )
        direct_progress_callback = progress_callback if len(programs) == 1 else None
        for program in programs:
            selected_indices = tuple(indices_by_label.get(program.label, ()))
            if not selected_indices:
                evaluation = ProgramEvaluation(
                    accuracy=0.0,
                    correct_count=0,
                    total_count=0,
                    runtime_error_count=0,
                    records=[],
                )
                by_label[program.label] = ProgramEvaluationResult(
                    label=program.label,
                    evaluation=evaluation,
                    explains=(
                        tuple(False for _transition in transitions)
                        if collect_records and not failed_records_only
                        else None
                    ),
                )
                continue
            handle = self._get_or_create_source_handle(program.source)
            evaluation = self._evaluate_source_key(
                source_key=handle.source_key,
                transitions=[transitions[index] for index in selected_indices],
                progress_callback=direct_progress_callback,
                collect_records=collect_records,
            )
            evaluation = self._remap_evaluation_records(
                evaluation=evaluation,
                transitions=transitions,
                selected_indices=selected_indices,
            )
            self._filter_evaluation_records(
                evaluation=evaluation,
                collect_records=collect_records,
                failed_records_only=failed_records_only,
                required_record_indices=required_record_indices,
            )
            by_label[program.label] = ProgramEvaluationResult(
                label=program.label,
                evaluation=evaluation,
                explains=self._evaluation_explains(
                    evaluation=evaluation,
                    transition_count=len(transitions),
                    failed_records_only=failed_records_only,
                    evaluated_indices=selected_indices,
                ),
            )
            completed_count += len(selected_indices)
            if callable(progress_callback) and direct_progress_callback is None:
                progress_callback(
                    {
                        "evaluated_count": int(completed_count),
                        "total_count": int(total_count),
                        "transition_index": int(max(0, completed_count - 1)),
                    }
                )
        return ProgramEvaluationBatch(labels=labels, by_label=by_label)

    def _evaluate_program_explains_serial(
        self,
        *,
        programs: Sequence[ProgramEvaluationTask],
        transitions: List[Transition],
        indices_by_label: Mapping[str, Tuple[int, ...]],
    ) -> ProgramExplainBatch:
        labels = tuple(program.label for program in programs)
        by_label: Dict[str, ProgramExplainResult] = {}
        for program in programs:
            selected_indices = tuple(indices_by_label.get(program.label, ()))
            explains = [False for _transition in transitions]
            if selected_indices:
                handle = self._get_or_create_source_handle(program.source)
                for index in selected_indices:
                    explains[int(index)] = self._evaluate_source_key_transition_explains(
                        source_key=handle.source_key,
                        transition=transitions[int(index)],
                    )
            by_label[program.label] = ProgramExplainResult(
                label=program.label,
                explains=tuple(explains),
            )
        return ProgramExplainBatch(labels=labels, by_label=by_label)

    def _evaluate_programs_process_chunks(
        self,
        *,
        programs: Sequence[ProgramEvaluationTask],
        transitions: List[Transition],
        chunk_size: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        collect_records: bool,
        failed_records_only: bool,
        required_record_indices: Sequence[int],
        indices_by_label: Mapping[str, Tuple[int, ...]],
        explains_only: bool = False,
    ) -> ProgramEvaluationBatch:
        ref_bundle = self._process_transition_refs(transitions)
        state_store_snapshot_dir: Optional[str] = None
        transition_refs: Tuple[_ProcessTransitionRef, ...] = ()
        if ref_bundle is not None:
            state_store, transition_refs, max_state_id = ref_bundle
            state_store_snapshot_dir = self._ensure_process_chunk_state_store_snapshot(
                state_store=state_store,
                max_state_id=max_state_id,
            )

        def make_chunks(
            selected_indices: Sequence[int],
        ) -> List[
            Tuple[
                int,
                Tuple[int, ...],
                Tuple[Dict[str, Any], ...],
                Tuple[_ProcessTransitionRef, ...],
            ]
        ]:
            chunks: List[
                Tuple[
                    int,
                    Tuple[int, ...],
                    Tuple[Dict[str, Any], ...],
                    Tuple[_ProcessTransitionRef, ...],
                ]
            ] = []
            safe_indices = tuple(int(index) for index in selected_indices)
            for offset in range(0, len(safe_indices), int(chunk_size)):
                chunk_indices = tuple(safe_indices[offset : offset + int(chunk_size)])
                if not chunk_indices:
                    continue
                start_index = int(chunk_indices[0])
                if transition_refs:
                    chunks.append(
                        (
                            start_index,
                            chunk_indices,
                            (),
                            tuple(transition_refs[index] for index in chunk_indices),
                        )
                    )
                else:
                    chunks.append(
                        (
                            start_index,
                            chunk_indices,
                            tuple(transitions[index].to_dict() for index in chunk_indices),
                            (),
                        )
                    )
            return chunks

        all_chunks_by_label = {
            program.label: make_chunks(indices_by_label.get(program.label, ()))
            for program in programs
        }
        if not any(all_chunks_by_label.values()):
            return ProgramEvaluationBatch(
                labels=tuple(program.label for program in programs),
                by_label={
                    program.label: ProgramEvaluationResult(
                        label=program.label,
                        evaluation=ProgramEvaluation(
                            accuracy=0.0,
                            correct_count=0,
                            total_count=0,
                            runtime_error_count=0,
                            records=[],
                        ),
                        explains=(
                            tuple(False for _transition in transitions)
                            if collect_records and not failed_records_only
                            else None
                        ),
                    )
                    for program in programs
                },
            )

        executor = self._ensure_process_chunk_executor()
        futures: Dict[Any, Tuple[str, _ProcessChunkRequest]] = {}
        submitted_at_by_future: Dict[Any, float] = {}
        queued_requests: List[Tuple[str, _ProcessChunkRequest]] = []
        required_indices = tuple(int(index) for index in required_record_indices)
        for program in programs:
            for (
                start_index,
                chunk_indices,
                transition_rows,
                chunk_refs,
            ) in all_chunks_by_label.get(program.label, ()):
                request = _ProcessChunkRequest(
                    source=program.source,
                    start_index=int(start_index),
                    collect_records=bool(collect_records),
                    explains_only=bool(explains_only),
                    failed_records_only=bool(failed_records_only),
                    required_record_indices=required_indices,
                    transition_rows=transition_rows,
                    transition_refs=chunk_refs,
                    transition_indices=chunk_indices,
                    state_store_snapshot_dir=state_store_snapshot_dir,
                )
                queued_requests.append((program.label, request))

        chunk_results_by_label: Dict[str, List[_ProcessChunkResult]] = {
            program.label: []
            for program in programs
        }
        completed_count = 0
        total_count = sum(
            len(indices_by_label.get(program.label, ()))
            for program in programs
        )
        pending = set(futures)
        next_request_index = 0
        max_pending_futures = max(1, int(self.program_eval_workers))
        retry_requests: List[Tuple[str, _ProcessChunkRequest]] = []
        retry_isolation_active = False
        broken_pool_retry_count_by_key: Dict[Tuple[str, int, Tuple[int, ...]], int] = {}

        def request_key(
            label: str,
            request: _ProcessChunkRequest,
        ) -> Tuple[str, int, Tuple[int, ...]]:
            return (
                str(label),
                int(request.start_index),
                tuple(int(index) for index in request.transition_indices),
            )

        def submit_available_requests() -> None:
            nonlocal next_request_index
            nonlocal retry_isolation_active
            if retry_isolation_active and not retry_requests and not pending:
                retry_isolation_active = False
            pending_limit = 1 if retry_isolation_active else max_pending_futures
            while (
                len(pending) < pending_limit
                and (
                    retry_requests
                    or next_request_index < len(queued_requests)
                )
            ):
                if retry_requests:
                    label, request = retry_requests.pop(0)
                else:
                    label, request = queued_requests[next_request_index]
                    next_request_index += 1
                future = executor.submit(_evaluate_source_process_chunk, request)
                futures[future] = (label, request)
                submitted_at_by_future[future] = time.perf_counter()
                pending.add(future)

        def record_chunk_result(label: str, result: _ProcessChunkResult) -> None:
            nonlocal completed_count
            chunk_results_by_label[label].append(result)
            completed_count += int(result.total_count)
            if callable(progress_callback):
                progress_callback(
                    {
                        "evaluated_count": int(completed_count),
                        "total_count": int(total_count),
                        "transition_index": int(max(0, completed_count - 1)),
                    }
                )

        submit_available_requests()
        while pending:
            timeout_sec = self.process_chunk_timeout_sec
            wait_timeout: Optional[float] = None
            if timeout_sec is not None:
                now = time.perf_counter()
                wait_timeout = min(
                    1.0,
                    max(
                        0.0,
                        min(
                            float(timeout_sec) - (now - submitted_at_by_future[future])
                            for future in pending
                        ),
                    ),
                )
            done, pending = wait(
                pending,
                timeout=wait_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                expired = self._expired_process_chunk_futures(
                    pending=pending,
                    submitted_at_by_future=submitted_at_by_future,
                )
                if not expired:
                    continue
                expired_futures = set(expired)
                remaining_requests = [
                    futures[future]
                    for future in pending
                    if future not in expired_futures
                ]
                for future in pending:
                    future.cancel()
                for future in expired_futures:
                    label, request = futures[future]
                    result = self._build_process_chunk_runtime_failure(
                        request=request,
                        transitions=transitions,
                        message=self._process_chunk_timeout_message(
                            request=request,
                            timeout_sec=float(timeout_sec),
                        ),
                        exception_type="TimeoutError",
                    )
                    record_chunk_result(label, result)
                self.close_process_chunk_executor(wait=False, terminate=True)
                if (
                    not remaining_requests
                    and next_request_index >= len(queued_requests)
                ):
                    pending.clear()
                    break
                executor = self._ensure_process_chunk_executor()
                pending = set()
                for label, request in remaining_requests:
                    replacement_future = executor.submit(
                        _evaluate_source_process_chunk,
                        request,
                    )
                    futures[replacement_future] = (label, request)
                    submitted_at_by_future[replacement_future] = time.perf_counter()
                    pending.add(replacement_future)
                submit_available_requests()
                continue

            pool_stopped = False
            done_list = list(done)
            for done_offset, future in enumerate(done_list):
                label, request = futures[future]
                try:
                    result = future.result()
                except BrokenProcessPool as exc:
                    restart_futures = {future}.union(pending).union(
                        done_list[done_offset + 1 :]
                    )
                    restart_requests = [
                        futures[restart_future]
                        for restart_future in restart_futures
                    ]
                    for restart_future in restart_futures:
                        restart_future.cancel()
                    self.close_process_chunk_executor(wait=False, terminate=True)
                    executor = self._ensure_process_chunk_executor()
                    pending = set()
                    for retry_label, retry_request in restart_requests:
                        key = request_key(retry_label, retry_request)
                        retry_count = broken_pool_retry_count_by_key.get(key, 0)
                        if retry_count >= 1:
                            retry_result = self._build_process_chunk_runtime_failure(
                                request=retry_request,
                                transitions=transitions,
                                message=(
                                    "Process evaluation worker pool stopped "
                                    "unexpectedly while retrying this chunk: "
                                    f"{exc}"
                                ),
                                exception_type=type(exc).__name__,
                            )
                            record_chunk_result(retry_label, retry_result)
                            continue
                        broken_pool_retry_count_by_key[key] = retry_count + 1
                        retry_requests.append((retry_label, retry_request))
                    retry_isolation_active = bool(retry_requests)
                    pool_stopped = True
                    break
                except Exception as exc:  # noqa: BLE001
                    result = self._build_process_chunk_runtime_failure(
                        request=request,
                        transitions=transitions,
                        message=(
                            "Process evaluation worker failed while evaluating a "
                            f"chunk: {exc}"
                        ),
                        exception_type=type(exc).__name__,
                    )
                record_chunk_result(label, result)
            if pool_stopped:
                submit_available_requests()
                continue
            submit_available_requests()

        labels = tuple(program.label for program in programs)
        by_label: Dict[str, ProgramEvaluationResult] = {}
        for program in programs:
            selected_indices = tuple(indices_by_label.get(program.label, ()))
            if explains_only:
                explains = self._merge_process_chunk_explains(
                    transition_count=len(transitions),
                    chunk_results=chunk_results_by_label.get(program.label, []),
                )
                by_label[program.label] = ProgramEvaluationResult(
                    label=program.label,
                    evaluation=ProgramEvaluation(
                        accuracy=(
                            sum(1 for index in selected_indices if explains[index])
                            / len(selected_indices)
                        )
                        if selected_indices
                        else 0.0,
                        correct_count=sum(
                            1 for index in selected_indices if explains[index]
                        ),
                        total_count=len(selected_indices),
                        runtime_error_count=0,
                        records=[],
                    ),
                    explains=explains,
                )
                continue
            evaluation = self._merge_process_chunk_results(
                transitions=transitions,
                chunk_results=chunk_results_by_label.get(program.label, []),
                collect_records=collect_records,
            )
            by_label[program.label] = ProgramEvaluationResult(
                label=program.label,
                evaluation=evaluation,
                explains=self._evaluation_explains(
                    evaluation=evaluation,
                    transition_count=len(transitions),
                    failed_records_only=failed_records_only,
                    evaluated_indices=selected_indices,
                ),
            )
        self._cleanup_stale_process_chunk_state_store_snapshots()
        return ProgramEvaluationBatch(labels=labels, by_label=by_label)

    def _expired_process_chunk_futures(
        self,
        *,
        pending: Sequence[Any],
        submitted_at_by_future: Mapping[Any, float],
    ) -> Tuple[Any, ...]:
        timeout_sec = self.process_chunk_timeout_sec
        if timeout_sec is None:
            return ()
        now = time.perf_counter()
        return tuple(
            future
            for future in pending
            if now - float(submitted_at_by_future.get(future, now)) >= float(timeout_sec)
        )

    def _process_chunk_timeout_message(
        self,
        *,
        request: _ProcessChunkRequest,
        timeout_sec: float,
    ) -> str:
        return (
            "Process evaluation chunk exceeded watchdog timeout: "
            f"{len(request.transition_indices)} transition(s) did not finish within "
            f"{timeout_sec:.3f}s."
        )

    def _build_process_chunk_runtime_failure(
        self,
        *,
        request: _ProcessChunkRequest,
        transitions: Sequence[Transition],
        message: str,
        exception_type: str,
    ) -> _ProcessChunkResult:
        error = SandboxError(
            phase="execute",
            message=str(message),
            exception_type=str(exception_type),
        )
        records: Tuple[_ProcessChunkRecord, ...] = ()
        explains: Tuple[bool, ...] = ()
        explain_indices: Tuple[int, ...] = ()
        if request.collect_records:
            records = tuple(
                _ProcessChunkRecord(
                    expected_canonical=self._get_expected_state_payload(
                        next_state_key=transitions[index].next_state_key,
                        next_state_obj=transitions[index].next_state_obj(),
                    ).canonical,
                    predicted_canonical=None,
                    is_correct=False,
                    error=self._clone_error(error),
                    local_index=int(index),
                )
                for index in request.transition_indices
                if 0 <= int(index) < len(transitions)
            )
        if request.explains_only:
            explain_indices = tuple(
                int(index)
                for index in request.transition_indices
                if 0 <= int(index) < len(transitions)
            )
            explains = tuple(False for _index in explain_indices)
        total_count = len(tuple(request.transition_indices))
        return _ProcessChunkResult(
            start_index=int(request.start_index),
            total_count=int(total_count),
            correct_count=0,
            runtime_error_count=int(total_count),
            compile_errors=(),
            records=records,
            explains=explains,
            explain_indices=explain_indices,
        )

    @staticmethod
    def _merge_process_chunk_explains(
        *,
        transition_count: int,
        chunk_results: Sequence[_ProcessChunkResult],
    ) -> Tuple[bool, ...]:
        explains = [False for _transition in range(int(transition_count))]
        for result in sorted(chunk_results, key=lambda row: int(row.start_index)):
            indices = tuple(int(index) for index in result.explain_indices)
            if not indices:
                indices = tuple(
                    int(result.start_index) + int(offset)
                    for offset in range(len(result.explains))
                )
            for index, solved in zip(indices, result.explains):
                if 0 <= int(index) < len(explains):
                    explains[int(index)] = bool(solved)
        return tuple(explains)

    def _merge_process_chunk_results(
        self,
        *,
        transitions: List[Transition],
        chunk_results: Sequence[_ProcessChunkResult],
        collect_records: bool,
    ) -> ProgramEvaluation:
        correct_count = 0
        runtime_error_count = 0
        compile_errors: List[SandboxError] = []
        records_by_index: Dict[int, PredictionRecord] = {}
        for result in sorted(chunk_results, key=lambda row: int(row.start_index)):
            correct_count += int(result.correct_count)
            runtime_error_count += int(result.runtime_error_count)
            if not compile_errors and result.compile_errors:
                compile_errors = self._clone_errors(result.compile_errors)
            if not collect_records:
                continue
            start_index = int(result.start_index)
            for local_index, record_payload in enumerate(result.records):
                payload_local_index = record_payload.local_index
                record_index = (
                    int(payload_local_index)
                    if payload_local_index is not None
                    else start_index + int(local_index)
                )
                if record_index < 0 or record_index >= len(transitions):
                    continue
                records_by_index[record_index] = PredictionRecord(
                    index=int(record_index),
                    transition=transitions[record_index],
                    expected_canonical=record_payload.expected_canonical,
                    predicted_canonical=record_payload.predicted_canonical,
                    is_correct=bool(record_payload.is_correct),
                    error=self._clone_error(record_payload.error),
                )

        total_count = sum(int(result.total_count) for result in chunk_results)
        return ProgramEvaluation(
            accuracy=(correct_count / total_count) if total_count else 0.0,
            correct_count=int(correct_count),
            total_count=int(total_count),
            runtime_error_count=int(runtime_error_count),
            compile_errors=compile_errors,
            records=[
                records_by_index[index]
                for index in sorted(records_by_index)
                if index in records_by_index
            ],
        )

    def _remap_evaluation_records(
        self,
        *,
        evaluation: ProgramEvaluation,
        transitions: List[Transition],
        selected_indices: Sequence[int],
    ) -> ProgramEvaluation:
        selected = tuple(int(index) for index in selected_indices)
        records: List[PredictionRecord] = []
        if evaluation.records:
            for record in evaluation.records:
                local_index = int(record.index)
                if local_index < 0 or local_index >= len(selected):
                    continue
                global_index = int(selected[local_index])
                records.append(
                    PredictionRecord(
                        index=global_index,
                        transition=transitions[global_index],
                        expected_canonical=record.expected_canonical,
                        predicted_canonical=record.predicted_canonical,
                        is_correct=bool(record.is_correct),
                        error=self._clone_error(record.error),
                    )
                )
        return ProgramEvaluation(
            accuracy=evaluation.accuracy,
            correct_count=int(evaluation.correct_count),
            total_count=int(evaluation.total_count),
            runtime_error_count=int(evaluation.runtime_error_count),
            compile_errors=self._clone_errors(evaluation.compile_errors),
            records=records,
        )

    def _filter_evaluation_records(
        self,
        *,
        evaluation: ProgramEvaluation,
        collect_records: bool,
        failed_records_only: bool,
        required_record_indices: Sequence[int],
    ) -> None:
        if not collect_records:
            evaluation.records = []
            return
        required_indices = {int(index) for index in required_record_indices}
        if not failed_records_only and not required_indices:
            return
        evaluation.records = [
            record
            for record in evaluation.records
            if (
                int(record.index) in required_indices
                or not bool(failed_records_only)
                or record.error is not None
                or not record.is_correct
            )
        ]

    @staticmethod
    def _evaluation_explains(
        *,
        evaluation: ProgramEvaluation,
        transition_count: int,
        failed_records_only: bool,
        evaluated_indices: Optional[Sequence[int]] = None,
    ) -> Optional[Tuple[bool, ...]]:
        if failed_records_only:
            return None
        if evaluated_indices is None:
            expected_record_count = int(transition_count)
        else:
            expected_record_count = len(tuple(evaluated_indices))
        if len(evaluation.records) != expected_record_count:
            return None
        explains = [False for _transition in range(int(transition_count))]
        for record in evaluation.records:
            index = int(record.index)
            if 0 <= index < len(explains):
                explains[index] = bool(record.is_correct) and record.error is None
        return tuple(explains)

    def _process_transition_refs(
        self,
        transitions: List[Transition],
    ) -> Optional[Tuple[StateStore, Tuple[_ProcessTransitionRef, ...], int]]:
        if not transitions:
            return None
        state_store = transitions[0].state_store
        if not isinstance(state_store, StateStore):
            return None

        refs: List[_ProcessTransitionRef] = []
        max_state_id = 0
        for transition in transitions:
            if transition.state_store is not state_store:
                return None
            state_id = self._positive_state_id(transition.state_id)
            next_state_id = self._positive_state_id(transition.next_state_id)
            if state_id is None or next_state_id is None:
                return None
            max_state_id = max(max_state_id, int(state_id), int(next_state_id))
            refs.append(
                _ProcessTransitionRef(
                    state_id=int(state_id),
                    next_state_id=int(next_state_id),
                    action=str(transition.action),
                    reward=float(transition.reward),
                    done=bool(transition.done),
                    world_index=(
                        int(transition.world_index)
                        if isinstance(transition.world_index, int)
                        and int(transition.world_index) > 0
                        else None
                    ),
                    map_name=(
                        str(transition.map_name).strip()
                        if isinstance(transition.map_name, str)
                        and str(transition.map_name).strip()
                        else None
                    ),
                )
            )
        return state_store, tuple(refs), int(max_state_id)

    @staticmethod
    def _positive_state_id(value: Any) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        resolved = int(value)
        return resolved if resolved > 0 else None

    def _ensure_process_chunk_state_store_snapshot(
        self,
        *,
        state_store: StateStore,
        max_state_id: int,
    ) -> str:
        store_id = id(state_store)
        state_count = int(len(state_store))
        if state_count < int(max_state_id):
            raise KeyError(
                f"StateStore snapshot requires state_id={int(max_state_id)} "
                f"but store only contains {state_count} states."
            )

        snapshot_dir = self._process_chunk_snapshot_dir
        snapshot_count = int(self._process_chunk_snapshot_state_count)
        if (
            self._process_chunk_snapshot_store_id == store_id
            and isinstance(snapshot_dir, str)
        ):
            if snapshot_count == state_count and self._process_chunk_snapshot_is_complete(
                snapshot_dir,
                min_state_count=state_count,
            ):
                return snapshot_dir

        old_snapshot_dir = snapshot_dir if isinstance(snapshot_dir, str) else None
        rebuilt_dir = tempfile.mkdtemp(prefix="baba_process_state_store_")
        try:
            state_store.save_directory(rebuilt_dir)
        except Exception:
            shutil.rmtree(rebuilt_dir, ignore_errors=True)
            raise
        self._process_chunk_snapshot_dir = str(rebuilt_dir)
        self._process_chunk_snapshot_store_id = store_id
        self._process_chunk_snapshot_state_count = state_count
        self._mark_process_chunk_snapshot_stale(old_snapshot_dir)
        return str(rebuilt_dir)

    @classmethod
    def _process_chunk_snapshot_is_complete(
        cls,
        snapshot_dir: str,
        *,
        min_state_count: int,
    ) -> bool:
        root = Path(snapshot_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        try:
            state_count = int(payload.get("stateCount", 0) or 0)
        except (TypeError, ValueError):
            return False
        if state_count < max(0, int(min_state_count)):
            return False
        required_files = (
            "widths.npy",
            "heights.npy",
            "state_flags.npy",
            "step_flags.npy",
            "terminated.npy",
            "object_offsets.npy",
            "object_counts.npy",
            "step_extra_ids.npy",
            "state_extra_ids.npy",
            "object_flags.npy",
            "object_type_ids.npy",
            "object_word_ids.npy",
            "object_x.npy",
            "object_y.npy",
            "object_direction_ids.npy",
            "object_color_ids.npy",
            "object_extra_ids.npy",
            "state_keys.npy",
            "type_vocab.npy",
            "word_vocab.npy",
            "direction_vocab.npy",
            "color_vocab.npy",
            "json_vocab.npy",
        )
        return all((root / file_name).exists() for file_name in required_files)

    @staticmethod
    def _default_process_chunk_size(
        *,
        total_count: int,
        worker_count: int,
    ) -> int:
        denominator = max(1, int(worker_count) * 4)
        balanced = (max(1, int(total_count)) + denominator - 1) // denominator
        return max(16, min(256, int(balanced)))

    def _ensure_process_chunk_executor(self) -> ProcessPoolExecutor:
        worker_count = max(1, int(self.program_eval_workers))
        if (
            self._process_chunk_executor is not None
            and self._process_chunk_worker_count == worker_count
        ):
            return self._process_chunk_executor
        self.close_process_chunk_executor()
        self._process_chunk_executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_process_chunk_worker,
            initargs=(self.sandbox.config,),
        )
        self._process_chunk_worker_count = worker_count
        return self._process_chunk_executor

    def close_process_chunk_executor(
        self,
        *,
        wait: bool = True,
        terminate: bool = False,
    ) -> None:
        executor = self._process_chunk_executor
        self._process_chunk_executor = None
        self._process_chunk_worker_count = None
        if executor is not None:
            if terminate:
                self._terminate_process_chunk_executor(executor)
            executor.shutdown(wait=bool(wait), cancel_futures=True)

    @staticmethod
    def _terminate_process_chunk_executor(executor: ProcessPoolExecutor) -> None:
        raw_processes = getattr(executor, "_processes", None)
        if isinstance(raw_processes, dict):
            processes = list(raw_processes.values())
        elif raw_processes is None:
            processes = []
        else:
            processes = list(raw_processes)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:  # noqa: BLE001
                continue
        for process in processes:
            try:
                process.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                continue
        for process in processes:
            try:
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
            except Exception:  # noqa: BLE001
                continue
        for process in processes:
            try:
                process.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                continue

    def _remove_process_chunk_state_store_snapshot(self) -> None:
        snapshot_dir = self._process_chunk_snapshot_dir
        stale_snapshot_dirs = list(self._process_chunk_stale_snapshot_dirs)
        self._process_chunk_snapshot_dir = None
        self._process_chunk_snapshot_store_id = None
        self._process_chunk_snapshot_state_count = 0
        self._process_chunk_stale_snapshot_dirs = []
        for removable_dir in [snapshot_dir, *stale_snapshot_dirs]:
            if isinstance(removable_dir, str) and removable_dir:
                shutil.rmtree(removable_dir, ignore_errors=True)

    def _mark_process_chunk_snapshot_stale(
        self,
        snapshot_dir: Optional[str],
    ) -> None:
        if not isinstance(snapshot_dir, str) or not snapshot_dir:
            return
        if snapshot_dir == self._process_chunk_snapshot_dir:
            return
        if snapshot_dir not in self._process_chunk_stale_snapshot_dirs:
            self._process_chunk_stale_snapshot_dirs.append(snapshot_dir)
        self._cleanup_stale_process_chunk_state_store_snapshots()

    def _cleanup_stale_process_chunk_state_store_snapshots(self) -> None:
        current_dir = self._process_chunk_snapshot_dir
        remaining: List[str] = []
        for snapshot_dir in self._process_chunk_stale_snapshot_dirs:
            if not isinstance(snapshot_dir, str) or not snapshot_dir:
                continue
            if snapshot_dir == current_dir:
                remaining.append(snapshot_dir)
                continue
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            if Path(snapshot_dir).exists():
                remaining.append(snapshot_dir)
        self._process_chunk_stale_snapshot_dirs = remaining

    def close(self) -> None:
        self.close_process_chunk_executor()
        self._remove_process_chunk_state_store_snapshot()

    def evaluate_transition(
        self,
        *,
        source: str,
        transition: Transition,
    ) -> PredictionRecord:
        handle = self._get_or_create_source_handle(source)
        return self._evaluate_source_key_transition(
            source_key=handle.source_key,
            transition=transition,
            index=0,
        )

    def sync_version_context(
        self,
        *,
        context: Optional[Dict[str, Any]],
        max_program_count: Optional[int],
        previous_snapshot: Optional[VersionContextSnapshot] = None,
    ) -> VersionContextUpdateResult:
        previous = previous_snapshot or VersionContextSnapshot()
        if not isinstance(context, dict):
            return VersionContextUpdateResult(previous, False, False, False)
        raw_versions = context.get("versions")
        if not isinstance(raw_versions, list):
            return VersionContextUpdateResult(previous, False, False, False)

        old_source_by_id = {
            version_id: self._get_registered_source(version_id)
            for version_id in previous.version_ids
        }
        version_rows: List[Tuple[str, str]] = []
        version_limit = (
            max(0, int(max_program_count))
            if max_program_count is not None
            else None
        )
        for raw in raw_versions:
            if not isinstance(raw, dict):
                continue
            version_id = raw.get("version_id")
            source = raw.get("source")
            if not isinstance(version_id, str) or not version_id.strip():
                continue
            if not isinstance(source, str) or not source.strip():
                continue
            if not self._is_explainable_version(
                version_id.strip(),
                raw_index=raw.get("index"),
            ):
                continue
            if version_limit == 0:
                break
            version_rows.append((version_id.strip(), source))
            if version_limit is not None and len(version_rows) >= version_limit:
                break

        source_changed = False
        for version_id, source in version_rows:
            changed = self._register_program(version_id, source)
            old_source = old_source_by_id.get(version_id)
            if changed and old_source is not None and old_source != source:
                source_changed = True
        self._retain_programs([version_id for version_id, _source in version_rows])

        raw_current = context.get("current_version_id")
        current_version_id = (
            str(raw_current).strip()
            if isinstance(raw_current, str) and raw_current.strip()
            else None
        )
        version_ids = tuple(version_id for version_id, _source in version_rows)
        snapshot = VersionContextSnapshot(
            version_ids=version_ids,
            active_version_count=self.resolve_active_version_count(
                version_ids=version_ids,
                current_version_id=current_version_id,
            ),
            current_version_id=current_version_id,
        )
        return VersionContextUpdateResult(
            snapshot=snapshot,
            source_changed=source_changed,
            version_ids_changed=(previous.version_ids != snapshot.version_ids),
            active_version_count_changed=(
                int(previous.active_version_count) != int(snapshot.active_version_count)
            ),
        )

    def resolve_active_version_count(
        self,
        *,
        version_ids: Sequence[str],
        current_version_id: Optional[str],
    ) -> int:
        total_versions = int(len(version_ids))
        if total_versions <= 0:
            return 0
        current_text = (
            str(current_version_id).strip()
            if isinstance(current_version_id, str)
            else ""
        )
        if current_text:
            current_ordinal = self._parse_version_ordinal(current_text)
            if current_ordinal is not None:
                if current_ordinal <= 0:
                    return 0
                explainable_count = sum(
                    1
                    for version_id in version_ids
                    if (
                        (version_ordinal := self._parse_version_ordinal(version_id))
                        is not None
                        and int(version_ordinal) <= int(current_ordinal)
                    )
                )
                if explainable_count > 0:
                    return int(min(explainable_count, total_versions))
            for version_index, version_id in enumerate(version_ids, start=1):
                if str(version_id) == current_text:
                    return int(version_index)
        return int(total_versions)

    def format_prediction_error(self, error: Optional[SandboxError]) -> Optional[str]:
        if error is None:
            return None
        phase = str(getattr(error, "phase", "")).strip() or "error"
        message = str(getattr(error, "message", "")).strip()
        return f"{phase}: {message}" if message else phase

    def clear_runtime_caches(self) -> None:
        self._prediction_cache.clear()
        self._evaluation_cache.clear()
        self._expected_payload_cache.clear()

    def _get_registered_handle(self, program_id: str) -> Optional[_SourceHandle]:
        safe_program_id = str(program_id).strip()
        if not safe_program_id:
            return None
        source_key = self._source_key_by_program_id.get(safe_program_id)
        if source_key is None:
            return None
        return self._source_handle_by_key.get(source_key)

    def _get_or_create_source_handle(self, source: str) -> _SourceHandle:
        safe_source = str(source)
        cached_key = self._source_key_by_source.get(safe_source)
        if cached_key is not None:
            cached_handle = self._source_handle_by_key.get(cached_key)
            if cached_handle is not None and cached_handle.source == safe_source:
                return cached_handle

        base_key = hashlib.sha1(safe_source.encode("utf-8")).hexdigest()
        source_key = base_key
        suffix = 1
        while True:
            existing = self._source_handle_by_key.get(source_key)
            if existing is None or existing.source == safe_source:
                break
            source_key = f"{base_key}:{suffix}"
            suffix += 1

        if source_key in self._source_handle_by_key:
            handle = self._source_handle_by_key[source_key]
            self._source_key_by_source[safe_source] = source_key
            return handle

        predictor = ProgramWorldModelPredictor(sandbox=self.sandbox)
        compiled = self.sandbox.compile_source(safe_source)
        compile_errors = tuple(
            self._clone_errors(compiled.errors if not compiled.success else [])
        )
        predictor.set_compiled_program(
            source=safe_source,
            namespace=compiled.namespace if compiled.success else None,
            compile_error=compiled.errors[0]
            if compiled.errors
            else SandboxError(
                phase="compile",
                message="Program failed to compile.",
            ),
        )
        handle = _SourceHandle(
            source_key=source_key,
            source=safe_source,
            predictor=predictor,
            compile_errors=compile_errors,
        )
        self._source_key_by_source[safe_source] = source_key
        self._source_handle_by_key[source_key] = handle
        return handle

    def _evaluate_source_key(
        self,
        *,
        source_key: str,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        collect_records: bool = True,
    ) -> ProgramEvaluation:
        handle = self._source_handle_by_key.get(source_key)
        if handle is None:
            return self._build_missing_program_evaluation(
                transitions=transitions,
                compile_error=SandboxError(
                    phase="predictor",
                    message=f"Missing cached source handle `{source_key}`.",
                ),
            )

        records: List[PredictionRecord] = []
        correct_count = 0
        runtime_error_count = 0
        total_count = len(transitions)
        for idx, transition in enumerate(transitions):
            if collect_records:
                record = self._evaluate_source_key_transition(
                    source_key=source_key,
                    transition=transition,
                    index=idx,
                )
                error = record.error
                is_correct = bool(record.is_correct)
                records.append(record)
            else:
                cached = self._evaluate_source_key_transition_cached(
                    source_key=source_key,
                    transition=transition,
                )
                error = cached.error
                is_correct = bool(cached.is_correct)
            if error is not None and not self._is_compile_phase_error(error):
                runtime_error_count += 1
            if is_correct:
                correct_count += 1
            if callable(progress_callback):
                progress_callback(
                    {
                        "evaluated_count": int(idx + 1),
                        "total_count": int(total_count),
                        "transition_index": int(idx),
                    }
                )

        accuracy = (correct_count / total_count) if total_count else 0.0
        return ProgramEvaluation(
            accuracy=accuracy,
            correct_count=correct_count,
            total_count=total_count,
            runtime_error_count=runtime_error_count,
            compile_errors=self._clone_errors(handle.compile_errors),
            records=records,
        )

    def _evaluate_source_key_transition(
        self,
        *,
        source_key: str,
        transition: Transition,
        index: int,
    ) -> PredictionRecord:
        cached = self._evaluate_source_key_transition_cached(
            source_key=source_key,
            transition=transition,
        )
        return PredictionRecord(
            index=index,
            transition=transition,
            expected_canonical=cached.expected_canonical,
            predicted_canonical=cached.predicted_canonical,
            is_correct=bool(cached.is_correct),
            error=self._clone_error(cached.error),
        )

    def _evaluate_source_key_transition_cached(
        self,
        *,
        source_key: str,
        transition: Transition,
    ) -> _CachedEvaluation:
        cache_key = (
            source_key,
            transition.state_key,
            transition.action,
            transition.next_state_key,
        )
        cached = self._evaluation_cache.get(cache_key)
        if cached is None:
            expected_payload = self._get_expected_state_payload(
                next_state_key=transition.next_state_key,
                next_state_obj=transition.next_state_obj(),
            )
            prediction = self._predict_source_key_transition(
                source_key=source_key,
                transition=transition,
            )
            is_correct = False
            if prediction.error is None and prediction.predicted_canonical is not None:
                if prediction.predicted_canonical == expected_payload.canonical:
                    is_correct = True
                elif (
                    expected_payload.parsed_state is not None
                    and prediction.predicted_state is not None
                ):
                    is_correct = states_equivalent(
                        expected_payload.parsed_state,
                        prediction.predicted_state,
                        path="$",
                    )
                else:
                    is_correct = prediction.predicted_canonical == expected_payload.canonical
            cached = _CachedEvaluation(
                expected_canonical=expected_payload.canonical,
                predicted_canonical=prediction.predicted_canonical,
                is_correct=is_correct,
                error=self._clone_error(prediction.error),
            )
            self._evaluation_cache[cache_key] = cached
        return cached

    def _evaluate_source_key_transition_explains(
        self,
        *,
        source_key: str,
        transition: Transition,
    ) -> bool:
        cached = self._evaluate_source_key_transition_cached(
            source_key=source_key,
            transition=transition,
        )
        return bool(cached.is_correct) and cached.error is None

    def _predict_source_key_transition(
        self,
        *,
        source_key: str,
        transition: Transition,
    ) -> _CachedPrediction:
        cache_key = (source_key, transition.state_key, transition.action)
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        handle = self._source_handle_by_key.get(source_key)
        if handle is None:
            cached = _CachedPrediction(
                predicted_canonical=None,
                predicted_state=None,
                error=SandboxError(
                    phase="predictor",
                    message=f"Missing cached source handle `{source_key}`.",
                ),
            )
            self._prediction_cache[cache_key] = cached
            return cached

        namespace = getattr(handle.predictor, "_namespace", None)
        if namespace is None:
            error = (
                handle.compile_errors[0]
                if handle.compile_errors
                else SandboxError(
                    phase="compile",
                    message="Program is not compiled.",
                )
            )
            cached = _CachedPrediction(
                predicted_canonical=None,
                predicted_state=None,
                error=self._clone_error(error),
            )
            self._prediction_cache[cache_key] = cached
            return cached

        try:
            state_obj = transition.state_obj()
            executed = handle.predictor.sandbox.execute_predict(
                namespace=namespace,
                state=state_obj,
                action=transition.action,
            )
            if not executed.success or executed.output_state is None:
                error = executed.errors[0] if executed.errors else None
                cached = _CachedPrediction(
                    predicted_canonical=None,
                    predicted_state=None,
                    error=self._clone_error(error),
                )
                self._prediction_cache[cache_key] = cached
                return cached
            predicted_canonical = executed.output_state_json
            if predicted_canonical is None:
                predicted_canonical = dump_state_json(executed.output_state)
            cached = _CachedPrediction(
                predicted_canonical=predicted_canonical,
                predicted_state=executed.output_state,
                error=None,
            )
            self._prediction_cache[cache_key] = cached
            return cached
        except Exception as exc:  # noqa: BLE001
            cached = _CachedPrediction(
                predicted_canonical=None,
                predicted_state=None,
                error=SandboxError(
                    phase="execute",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                ),
            )
            self._prediction_cache[cache_key] = cached
            return cached

    def _get_expected_state_payload(
        self,
        *,
        next_state_key: str,
        next_state_obj: Dict[str, Any],
    ) -> _ExpectedStatePayload:
        cached = self._expected_payload_cache.get(next_state_key)
        if cached is not None:
            return cached
        parsed_state = next_state_obj
        canonical = dump_state_json(parsed_state)
        cached = _ExpectedStatePayload(parsed_state=parsed_state, canonical=canonical)
        self._expected_payload_cache[next_state_key] = cached
        return cached

    def _build_missing_program_evaluation(
        self,
        *,
        transitions: List[Transition],
        compile_error: SandboxError,
    ) -> ProgramEvaluation:
        records = []
        for idx, transition in enumerate(transitions):
            expected_payload = self._get_expected_state_payload(
                next_state_key=transition.next_state_key,
                next_state_obj=transition.next_state_obj(),
            )
            records.append(
                PredictionRecord(
                    index=idx,
                    transition=transition,
                    expected_canonical=expected_payload.canonical,
                    predicted_canonical=None,
                    is_correct=False,
                    error=self._clone_error(compile_error),
                )
            )
        return ProgramEvaluation(
            accuracy=0.0,
            correct_count=0,
            total_count=len(transitions),
            runtime_error_count=0,
            compile_errors=[self._clone_error(compile_error)],
            records=records,
        )

    def _clone_error(self, error: Optional[SandboxError]) -> Optional[SandboxError]:
        if error is None:
            return None
        return SandboxError(
            phase=error.phase,
            message=error.message,
            exception_type=error.exception_type,
            traceback_text=error.traceback_text,
        )

    def _clone_errors(self, errors: Sequence[SandboxError]) -> List[SandboxError]:
        return [
            cloned
            for cloned in (self._clone_error(error) for error in errors)
            if cloned is not None
        ]

    def _is_compile_phase_error(self, error: Optional[SandboxError]) -> bool:
        if error is None:
            return False
        phase = getattr(error, "phase", None)
        if phase is None:
            return False
        return str(phase) in {"parse", "ast_validate", "compile"}

    def _is_explainable_version(
        self,
        version_id: str,
        *,
        raw_index: Any = None,
    ) -> bool:
        version_ordinal = self._parse_version_ordinal(raw_index)
        if version_ordinal is None:
            version_ordinal = self._parse_version_ordinal(version_id)
        return version_ordinal is None or int(version_ordinal) > 0

    def _parse_version_ordinal(self, raw_value: Any) -> Optional[int]:
        if isinstance(raw_value, bool):
            return None
        if isinstance(raw_value, int):
            return int(raw_value)
        text = str(raw_value).strip() if raw_value is not None else ""
        if not text:
            return None
        if text.lower().startswith("v"):
            text = text[1:]
        return int(text) if text.isdigit() else None


_PROCESS_CHUNK_WORKER_EVALUATOR: Optional[ProgramEvaluator] = None
_PROCESS_CHUNK_WORKER_STATE_STORE_DIR: Optional[str] = None
_PROCESS_CHUNK_WORKER_STATE_STORE: Optional[StateStore] = None


def _init_process_chunk_worker(sandbox_config: SandboxConfig) -> None:
    global _PROCESS_CHUNK_WORKER_EVALUATOR
    _PROCESS_CHUNK_WORKER_EVALUATOR = ProgramEvaluator(
        sandbox_config=sandbox_config,
        program_eval_workers=1,
    )


def _process_chunk_worker_state_store(snapshot_dir: str) -> StateStore:
    global _PROCESS_CHUNK_WORKER_STATE_STORE_DIR
    global _PROCESS_CHUNK_WORKER_STATE_STORE
    resolved_dir = str(snapshot_dir)
    if (
        _PROCESS_CHUNK_WORKER_STATE_STORE is not None
        and _PROCESS_CHUNK_WORKER_STATE_STORE_DIR == resolved_dir
    ):
        return _PROCESS_CHUNK_WORKER_STATE_STORE
    if _PROCESS_CHUNK_WORKER_STATE_STORE is not None:
        _PROCESS_CHUNK_WORKER_STATE_STORE.close()
    _PROCESS_CHUNK_WORKER_STATE_STORE = StateStore.load_directory(resolved_dir)
    _PROCESS_CHUNK_WORKER_STATE_STORE_DIR = resolved_dir
    return _PROCESS_CHUNK_WORKER_STATE_STORE


def _evaluate_source_process_chunk(
    payload: _ProcessChunkRequest,
) -> _ProcessChunkResult:
    if _PROCESS_CHUNK_WORKER_EVALUATOR is None:
        raise RuntimeError("Process chunk evaluator worker was not initialized.")

    if payload.transition_refs:
        snapshot_dir = payload.state_store_snapshot_dir
        if not isinstance(snapshot_dir, str) or not snapshot_dir:
            raise RuntimeError("State-id process chunk is missing a StateStore snapshot.")
        state_store = _process_chunk_worker_state_store(snapshot_dir)
        transitions = [
            Transition.from_state_ids(
                state_store=state_store,
                state_id=int(ref.state_id),
                action=str(ref.action),
                next_state_id=int(ref.next_state_id),
                reward=float(ref.reward),
                done=bool(ref.done),
                world_index=ref.world_index,
                map_name=ref.map_name,
            )
            for ref in payload.transition_refs
        ]
    else:
        transitions = [Transition.from_dict(row) for row in payload.transition_rows]
    if payload.explains_only:
        handle = _PROCESS_CHUNK_WORKER_EVALUATOR._get_or_create_source_handle(
            payload.source
        )
        explains = tuple(
            _PROCESS_CHUNK_WORKER_EVALUATOR._evaluate_source_key_transition_explains(
                source_key=handle.source_key,
                transition=transition,
            )
            for transition in transitions
        )
        return _ProcessChunkResult(
            start_index=int(payload.start_index),
            total_count=len(transitions),
            correct_count=sum(1 for solved in explains if solved),
            runtime_error_count=0,
            compile_errors=tuple(handle.compile_errors),
            records=(),
            explains=explains,
            explain_indices=tuple(int(index) for index in payload.transition_indices),
        )
    evaluation = _PROCESS_CHUNK_WORKER_EVALUATOR.evaluate_source(
        source=payload.source,
        transitions=transitions,
        collect_records=payload.collect_records,
    )
    records = ()
    if payload.collect_records:
        source_records = evaluation.records
        required_record_indices = {
            int(index)
            for index in payload.required_record_indices
        }
        if payload.failed_records_only:
            source_records = [
                record
                for record in source_records
                if (
                    _process_chunk_global_index(payload, int(record.index))
                    in required_record_indices
                    or record.error is not None
                    or not record.is_correct
                )
            ]
        records = tuple(
            _ProcessChunkRecord(
                expected_canonical=record.expected_canonical,
                predicted_canonical=record.predicted_canonical,
                is_correct=bool(record.is_correct),
                error=record.error,
                local_index=_process_chunk_global_index(payload, int(record.index)),
            )
            for record in source_records
        )

    return _ProcessChunkResult(
        start_index=int(payload.start_index),
        total_count=int(evaluation.total_count),
        correct_count=int(evaluation.correct_count),
        runtime_error_count=int(evaluation.runtime_error_count),
        compile_errors=tuple(evaluation.compile_errors),
        records=records,
    )


def _process_chunk_global_index(payload: _ProcessChunkRequest, local_index: int) -> int:
    if 0 <= int(local_index) < len(payload.transition_indices):
        return int(payload.transition_indices[int(local_index)])
    return int(payload.start_index) + int(local_index)
