from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import zipfile

import numpy as np

from src.agents.contrastive_core import ContrastiveSample, ContrastiveSampleStore
from src.data.canonical_dataset import CanonicalDataset
from src.data.state_store import StateStore
from src.data.transition_buffer import (
    Transition,
    canonical_graph_edge_identity_key,
)
from src.discovery.program_resume import (
    ProgramResumeBootstrapResult,
    bootstrap_program_resume_artifacts,
    count_completed_candidate_responses,
    derive_last_program_patch_step,
    load_llm_usage_events_through_call_count,
    normalize_version_id,
)


@dataclass
class IterationBoundaryResumeState:
    source_run_dir: Path
    latest_complete_iteration: int
    cutoff_iteration: int
    cutoff_global_step: int
    cutoff_program_version_id: str
    cutoff_state_count: int
    llm_calls: int
    completed_candidate_responses: int
    llm_usage_events: Tuple[Dict[str, Any], ...]
    state_store: StateStore
    canonical_transitions: List[Transition]
    sample_store: ContrastiveSampleStore
    node_rows: List[Dict[str, Any]]
    edge_rows: List[Dict[str, Any]]
    final_edge_rows: List[Dict[str, Any]]
    # Per-world frontier policy snapshot at the cutoff iteration. Score-topK stores
    # exact candidates; full-uniform stores compact headers and rebuilds from graph.
    frontier_rows: List[Dict[str, Any]] = field(default_factory=list)
    transition_class_id_by_key: Dict[str, int] = field(default_factory=dict)
    transition_group_id_by_key: Dict[str, str] = field(default_factory=dict)
    class_metadata_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    no_new_canonical_rounds: int = 0
    max_no_new_canonical_rounds: Optional[int] = None
    saturation_fail_count: Optional[int] = None
    saturation_max_fail_count: Optional[int] = None
    saturation_last_seen_global_step: Optional[int] = None
    saturation_last_patch_global_step: Optional[int] = None
    saturation_is_saturated: Optional[bool] = None
    frontier_size: int = 0
    checkpoint_path: Optional[Path] = None
    checkpoint_total_steps: Optional[int] = None
    checkpoint_total_updates: Optional[int] = None
    checkpoint_version_id: Optional[str] = None
    # Discovery cycle latest.pt was written at, or None for a checkpoint saved
    # before that field existed. See `checkpoint_iteration_lag`.
    checkpoint_iteration: Optional[int] = None
    program_bootstrap: Optional[ProgramResumeBootstrapResult] = None
    restored_sample_store_count: Optional[int] = None

    @property
    def checkpoint_iteration_lag(self) -> Optional[int]:
        """How many cycles the restored weights trail the restored data.

        The graph, state store, sample store and class metadata all come from
        the archive at ``cutoff_iteration``, but the encoder, optimizer and step
        counters come from ``latest.pt``, which the pipeline rewrites only every
        ``_training_checkpoint_interval`` cycles. A positive lag means that many
        cycles of contrastive training are not in the restored weights. The data
        is still complete, so this is recoverable - but it must be visible, not
        inferred from a training curve that restarts lower than it ended.
        """
        if not isinstance(self.checkpoint_iteration, int):
            return None
        return max(0, int(self.cutoff_iteration) - int(self.checkpoint_iteration))

    @property
    def edge_count(self) -> int:
        return int(len(self.final_edge_rows))

    @property
    def canonical_count(self) -> int:
        return int(len(self.canonical_transitions))

    @property
    def sample_store_count(self) -> int:
        if isinstance(self.restored_sample_store_count, int):
            return int(self.restored_sample_store_count)
        return int(len(self.sample_store))

    def summary(self) -> Dict[str, Any]:
        return {
            "source_run_dir": str(self.source_run_dir),
            "latest_complete_iteration": int(self.latest_complete_iteration),
            "cutoff_iteration": int(self.cutoff_iteration),
            "cutoff_global_step": int(self.cutoff_global_step),
            "cutoff_program_version_id": str(self.cutoff_program_version_id),
            "cutoff_state_count": int(self.cutoff_state_count),
            "edge_count": int(self.edge_count),
            "canonical_count": int(self.canonical_count),
            "sample_store_count": int(self.sample_store_count),
            "frontier_size": int(self.frontier_size),
            "llm_calls": int(self.llm_calls),
            "completed_candidate_responses": int(
                self.completed_candidate_responses
            ),
            "llm_usage_event_count": int(len(self.llm_usage_events)),
            "no_new_canonical_rounds": int(self.no_new_canonical_rounds),
            "max_no_new_canonical_rounds": self.max_no_new_canonical_rounds,
            "saturation_fail_count": self.saturation_fail_count,
            "saturation_max_fail_count": self.saturation_max_fail_count,
            "saturation_last_seen_global_step": self.saturation_last_seen_global_step,
            "saturation_last_patch_global_step": self.saturation_last_patch_global_step,
            "saturation_is_saturated": self.saturation_is_saturated,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "checkpoint_total_steps": self.checkpoint_total_steps,
            "checkpoint_total_updates": self.checkpoint_total_updates,
            "checkpoint_version_id": self.checkpoint_version_id,
            "checkpoint_iteration": self.checkpoint_iteration,
            "checkpoint_iteration_lag": self.checkpoint_iteration_lag,
        }


_ITERATION_FILE_PREFIX = "iter_"
_STATE_ARRAY_NAMES = (
    "widths.npy",
    "heights.npy",
    "state_flags.npy",
    "step_flags.npy",
    "terminated.npy",
    "object_counts.npy",
    "step_extra_ids.npy",
    "state_extra_ids.npy",
    "state_keys.npy",
)

_OBJECT_ARRAY_NAMES = (
    "object_flags.npy",
    "object_type_ids.npy",
    "object_word_ids.npy",
    "object_x.npy",
    "object_y.npy",
    "object_direction_ids.npy",
    "object_color_ids.npy",
    "object_extra_ids.npy",
)
_VOCAB_ARRAY_NAMES = (
    "type_vocab.npy",
    "word_vocab.npy",
    "direction_vocab.npy",
    "color_vocab.npy",
    "json_vocab.npy",
)

def build_iteration_boundary_resume_state(
    source_run_dir: str | Path,
    *,
    rollback_iterations: int = 1,
    through_iteration: Optional[int] = None,
    checkpoint_policy: str = "latest",
) -> IterationBoundaryResumeState:
    source = Path(source_run_dir).resolve()
    archive = source / "analysis_archive"
    if not archive.exists():
        raise FileNotFoundError(f"Missing analysis_archive: {archive}")

    complete_iterations = _complete_archive_iterations(archive)
    if not complete_iterations:
        raise ValueError(f"No complete archive iterations found under {archive}")
    latest_iteration = int(complete_iterations[-1])
    if through_iteration is None:
        rollback = max(0, int(rollback_iterations))
        cutoff_iteration = latest_iteration - rollback
    else:
        cutoff_iteration = int(through_iteration)
    if cutoff_iteration <= 0:
        raise ValueError(
            "Discovery resume needs a positive completed iteration. "
            f"latest={latest_iteration} rollback={rollback_iterations}"
        )
    if cutoff_iteration not in complete_iterations:
        raise ValueError(
            f"Iteration {cutoff_iteration} is not complete in analysis_archive. "
            f"latest_complete_iteration={latest_iteration}"
        )

    node_rows = _read_archive_rows(archive / "nodes", through_iteration=cutoff_iteration)
    edge_rows = _read_archive_rows(archive / "edges", through_iteration=cutoff_iteration)
    cutoff_summary, cutoff_step = _read_cutoff_metric_rows(
        archive / "metrics",
        iteration=cutoff_iteration,
    )
    cutoff_global_step = _resolve_int(
        cutoff_summary.get("end_global_step", cutoff_summary.get("global_step")),
        name="cutoff global_step",
    )
    cutoff_version = normalize_version_id(
        cutoff_summary.get("iteration_end_program_version")
        or cutoff_summary.get("program_version")
    )
    llm_calls = int(cutoff_summary.get("llm_calls", 0) or 0)
    completed_candidate_responses = cutoff_summary.get(
        "completed_candidate_responses"
    )
    if not isinstance(completed_candidate_responses, int):
        completed_candidate_responses = count_completed_candidate_responses(
            source,
            through_iteration=cutoff_iteration,
        )
    llm_usage_events = load_llm_usage_events_through_call_count(source, llm_calls)
    no_new_canonical_rounds = _optional_nonnegative_int(
        cutoff_summary.get("no_new_canonical_rounds")
    ) or 0
    max_no_new_canonical_rounds = _optional_nonnegative_int(
        cutoff_summary.get("max_no_new_canonical_rounds")
    )
    saturation_state = _restore_saturation_state_from_boundary(
        source_run_dir=source,
        cutoff_summary=cutoff_summary,
        cutoff_iteration=cutoff_iteration,
        cutoff_global_step=cutoff_global_step,
        cutoff_program_version_id=cutoff_version,
    )
    class_metadata_by_id = _class_metadata_from_boundary(
        cutoff_summary=cutoff_summary,
        cutoff_step=cutoff_step,
    )
    final_edge_rows = _final_edge_rows(edge_rows)
    cutoff_state_count = _max_referenced_state_id(node_rows=node_rows, edge_rows=final_edge_rows)
    if cutoff_state_count <= 0:
        raise ValueError(f"Iteration {cutoff_iteration} has no restorable states.")

    state_store = _load_state_store_through(
        archive / "state_store",
        cutoff_state_count=cutoff_state_count,
    )
    canonical_transitions, sample_store, class_by_key, group_by_key = (
        _build_transition_stores(
            state_store=state_store,
            edge_rows=final_edge_rows,
            class_metadata_by_id=class_metadata_by_id,
        )
    )
    (
        checkpoint_path,
        checkpoint_total_steps,
        checkpoint_total_updates,
        checkpoint_version_id,
        checkpoint_iteration,
    ) = _resolve_checkpoint_metadata(
        source,
        policy=checkpoint_policy,
    )
    frontier_rows = _read_frontier_snapshot(
        archive / "frontier",
        iteration=cutoff_iteration,
    )
    frontier_size = _compute_frontier_size(
        frontier_rows=frontier_rows,
    )
    return IterationBoundaryResumeState(
        source_run_dir=source,
        latest_complete_iteration=latest_iteration,
        cutoff_iteration=cutoff_iteration,
        cutoff_global_step=cutoff_global_step,
        cutoff_program_version_id=cutoff_version,
        cutoff_state_count=cutoff_state_count,
        llm_calls=llm_calls,
        completed_candidate_responses=int(completed_candidate_responses),
        llm_usage_events=llm_usage_events,
        state_store=state_store,
        canonical_transitions=canonical_transitions,
        sample_store=sample_store,
        node_rows=node_rows,
        edge_rows=edge_rows,
        final_edge_rows=final_edge_rows,
        frontier_rows=frontier_rows,
        transition_class_id_by_key=class_by_key,
        transition_group_id_by_key=group_by_key,
        class_metadata_by_id=class_metadata_by_id,
        no_new_canonical_rounds=no_new_canonical_rounds,
        max_no_new_canonical_rounds=max_no_new_canonical_rounds,
        saturation_fail_count=saturation_state.get("fail_count"),
        saturation_max_fail_count=saturation_state.get("max_fail_count"),
        saturation_last_seen_global_step=saturation_state.get("last_seen_global_step"),
        saturation_last_patch_global_step=saturation_state.get("last_patch_global_step"),
        saturation_is_saturated=saturation_state.get("is_saturated"),
        frontier_size=frontier_size,
        checkpoint_path=checkpoint_path,
        checkpoint_total_steps=checkpoint_total_steps,
        checkpoint_total_updates=checkpoint_total_updates,
        checkpoint_version_id=checkpoint_version_id,
        checkpoint_iteration=checkpoint_iteration,
        restored_sample_store_count=len(sample_store),
    )


def bootstrap_iteration_boundary_resume_artifacts(
    *,
    source_run_dir: str | Path,
    target_run_dir: str | Path,
    rollback_iterations: int = 1,
    through_iteration: Optional[int] = None,
    checkpoint_policy: str = "latest",
) -> IterationBoundaryResumeState:
    state = build_iteration_boundary_resume_state(
        source_run_dir,
        rollback_iterations=rollback_iterations,
        through_iteration=through_iteration,
        checkpoint_policy=checkpoint_policy,
    )
    state.program_bootstrap = bootstrap_program_resume_artifacts(
        source_run_dir=state.source_run_dir,
        target_run_dir=target_run_dir,
        through_version_id=state.cutoff_program_version_id,
        through_iteration=state.cutoff_iteration,
    )
    return state


def _complete_archive_iterations(archive: Path) -> Tuple[int, ...]:
    directory = archive / "complete"
    if not directory.exists():
        raise FileNotFoundError(f"Missing archive directory: {directory}")
    complete: List[int] = []
    for path in sorted(directory.glob(f"{_ITERATION_FILE_PREFIX}*.json")):
        stem = path.name.removeprefix(_ITERATION_FILE_PREFIX).removesuffix(".json")
        if not stem.isdigit():
            continue
        iteration = int(stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid archive completion marker: {path}")
        state_count = payload.get("stateCount")
        if (
            payload.get("formatVersion") != 1
            or payload.get("iteration") != iteration
            or isinstance(state_count, bool)
            or not isinstance(state_count, int)
            or int(state_count) < 0
        ):
            raise ValueError(f"Invalid archive completion marker: {path}")
        complete.append(iteration)
    return tuple(complete)


def _archive_iterations(directory: Path) -> Iterable[int]:
    for path in directory.glob(f"{_ITERATION_FILE_PREFIX}*.jsonl.gz"):
        stem = path.name.removeprefix(_ITERATION_FILE_PREFIX).removesuffix(".jsonl.gz")
        if stem.isdigit():
            yield int(stem)


def _archive_iteration_path(directory: Path, iteration: int) -> Path:
    return directory / f"iter_{int(iteration):06d}.jsonl.gz"


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(dict(payload))
    return rows


def _read_archive_rows(directory: Path, *, through_iteration: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for iteration in _archive_iterations(directory):
        if int(iteration) > int(through_iteration):
            continue
        rows.extend(_read_jsonl_gz(_archive_iteration_path(directory, iteration)))
    return rows


def _read_frontier_snapshot(
    directory: Path,
    *,
    iteration: int,
) -> List[Dict[str, Any]]:
    """Read the frontier policy snapshot for exactly one iteration.

    Unlike nodes/edges this is not cumulative, so only the cutoff file is relevant.
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"Missing required frontier snapshot directory: {directory}"
        )
    path = _archive_iteration_path(directory, iteration)
    if not path.exists():
        raise FileNotFoundError(f"Missing required frontier snapshot: {path}")
    rows = _read_jsonl_gz(path)
    if not rows:
        raise ValueError(f"Frontier snapshot is empty: {path}")
    return rows


def _read_cutoff_metric_rows(
    directory: Path,
    *,
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = _read_jsonl_gz(_archive_iteration_path(directory, iteration))
    summary: Optional[Dict[str, Any]] = None
    last_step: Optional[Dict[str, Any]] = None
    for row in rows:
        row_type = row.get("row_type")
        if row_type == "iteration_summary":
            summary = dict(row)
        elif row_type == "step":
            last_step = dict(row)
    if summary is None:
        raise ValueError(f"Missing iteration_summary row for iteration {iteration}")
    return summary, dict(last_step or {})


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not float(parsed).is_integer():
        return None
    return int(parsed)


def _optional_nonnegative_int(value: Any) -> Optional[int]:
    parsed = _optional_int(value)
    if parsed is None or int(parsed) < 0:
        return None
    return int(parsed)


def _optional_positive_int(value: Any) -> Optional[int]:
    parsed = _optional_int(value)
    if parsed is None or int(parsed) <= 0:
        return None
    return int(parsed)


def _class_metadata_from_boundary(
    *,
    cutoff_summary: Mapping[str, Any],
    cutoff_step: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Recover dynamics-class metadata from the iteration boundary.

    Current archives carry ``dynamics_class_transition_counts`` once on the
    ``iteration_summary`` row. Archives written before that change repeated the
    identical value on every ``step`` row instead, so fall back to the last step
    row whenever the summary row does not carry it.
    """
    metadata = _class_metadata_from_metric_row(cutoff_summary)
    if metadata:
        return metadata
    return _class_metadata_from_metric_row(cutoff_step)


def _class_metadata_from_metric_row(row: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    entries = row.get("dynamics_class_transition_counts")
    if not isinstance(entries, list):
        return {}
    metadata: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw_class_id = entry.get("class_id")
        if not isinstance(raw_class_id, int) or int(raw_class_id) <= 0:
            continue
        class_id = int(raw_class_id)
        payload = dict(entry)
        payload["class_id"] = class_id
        metadata[class_id] = payload
    return metadata


def _restore_saturation_state_from_boundary(
    *,
    source_run_dir: Path,
    cutoff_summary: Mapping[str, Any],
    cutoff_iteration: int,
    cutoff_global_step: int,
    cutoff_program_version_id: str,
) -> Dict[str, Optional[int | bool]]:
    max_fail_count = _optional_positive_int(cutoff_summary.get("saturation_max_fail_count"))
    last_patch = derive_last_program_patch_step(
        source_run_dir=source_run_dir,
        through_version_id=cutoff_program_version_id,
        through_iteration=cutoff_iteration,
    )
    raw_is_saturated = cutoff_summary.get("saturation_is_saturated")
    if raw_is_saturated is None:
        raw_is_saturated = cutoff_summary.get("is_saturated")
    is_saturated = raw_is_saturated if isinstance(raw_is_saturated, bool) else None

    last_seen = int(cutoff_global_step) if last_patch is not None else None
    fail_count = (
        max(0, int(cutoff_global_step) - int(last_patch))
        if last_patch is not None
        else None
    )
    if is_saturated is None and fail_count is not None and max_fail_count is not None:
        is_saturated = int(fail_count) >= int(max_fail_count)

    return {
        "fail_count": fail_count,
        "max_fail_count": max_fail_count,
        "last_seen_global_step": last_seen,
        "last_patch_global_step": last_patch,
        "is_saturated": is_saturated,
    }


def _final_edge_rows(edge_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_identity: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    ordered_identities: List[Tuple[int, int, int]] = []
    for row in edge_rows:
        try:
            identity = (
                int(row.get("world_index")),
                int(row.get("source_state_id")),
                int(row.get("action")),
            )
        except (TypeError, ValueError):
            continue
        if identity not in by_identity:
            ordered_identities.append(identity)
        by_identity[identity] = dict(row)
    return [by_identity[identity] for identity in ordered_identities]


def _max_referenced_state_id(
    *,
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
) -> int:
    max_state_id = 0
    for row in node_rows:
        value = row.get("state_id")
        if isinstance(value, int):
            max_state_id = max(max_state_id, int(value))
    for row in edge_rows:
        for field_name in ("source_state_id", "next_state_id"):
            value = row.get(field_name)
            if isinstance(value, int):
                max_state_id = max(max_state_id, int(value))
    return int(max_state_id)


def _load_state_store_through(
    state_archive_dir: Path,
    *,
    cutoff_state_count: int,
) -> StateStore:
    manifest_path = state_archive_dir / "segments.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Invalid state-store segment manifest: {manifest_path}")
    selected = _select_state_segments(segments, cutoff_state_count=cutoff_state_count)
    with tempfile.TemporaryDirectory(prefix="baba_resume_state_") as tmpdir:
        tmp_root = Path(tmpdir)
        arrays, vocabs = _concatenate_state_segments(
            state_archive_dir=state_archive_dir,
            segments=selected,
            cutoff_state_count=cutoff_state_count,
            temp_root=tmp_root,
        )
        combined = tmp_root / "combined_state_store"
        combined.mkdir(parents=True, exist_ok=True)
        _write_combined_state_store(
            combined,
            arrays=arrays,
            vocabs=vocabs,
            format_version=int(_first_manifest_format_version(selected)),
        )
        store = StateStore.load_directory(combined)
        store._ensure_mutable_storage()
        return store


def _select_state_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    cutoff_state_count: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    expected_first = 1
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        first_state_id = int(segment.get("firstStateId", 0) or 0)
        last_state_id = int(segment.get("lastStateId", 0) or 0)
        if first_state_id != expected_first:
            raise ValueError(
                "State-store segments are not contiguous at "
                f"state_id={expected_first}; got firstStateId={first_state_id}."
            )
        if first_state_id > int(cutoff_state_count):
            break
        selected.append(dict(segment))
        expected_first = int(last_state_id) + 1
        if last_state_id >= int(cutoff_state_count):
            break
    if not selected:
        raise ValueError("No state-store segments selected for resume.")
    if int(selected[-1].get("lastStateId", 0) or 0) < int(cutoff_state_count):
        raise ValueError(
            f"State-store archive ends before cutoff state {int(cutoff_state_count)}."
        )
    return selected


def _first_manifest_format_version(segments: Sequence[Mapping[str, Any]]) -> int:
    for segment in segments:
        manifest = segment.get("_inner_manifest")
        if isinstance(manifest, Mapping):
            value = manifest.get("formatVersion")
            if isinstance(value, int):
                return int(value)
    return 1


def _concatenate_state_segments(
    *,
    state_archive_dir: Path,
    segments: Sequence[Mapping[str, Any]],
    cutoff_state_count: int,
    temp_root: Path,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray]]:
    arrays: Dict[str, List[np.ndarray]] = {
        name: [] for name in (*_STATE_ARRAY_NAMES, "object_offsets.npy", *_OBJECT_ARRAY_NAMES)
    }
    vocabs: Dict[str, np.ndarray] = {}
    object_offset = 0
    loaded_state_count = 0
    for index, segment in enumerate(segments):
        segment_dir = _extract_or_resolve_segment(
            state_archive_dir=state_archive_dir,
            segment=segment,
            temp_root=temp_root / f"segment_{index:06d}",
        )
        inner_manifest = json.loads((segment_dir / "manifest.json").read_text(encoding="utf-8"))
        if isinstance(segment, dict):
            segment["_inner_manifest"] = dict(inner_manifest)
        first_state_id = int(inner_manifest.get("firstStateId", segment.get("firstStateId", 1)) or 1)
        last_state_id = int(inner_manifest.get("lastStateId", segment.get("lastStateId", 0)) or 0)
        if first_state_id != loaded_state_count + 1:
            raise ValueError(
                "State-store segment inner manifest is not contiguous: "
                f"expected firstStateId={loaded_state_count + 1}, got {first_state_id}."
            )
        local_count = min(
            int(last_state_id),
            int(cutoff_state_count),
        ) - int(first_state_id) + 1
        if local_count <= 0:
            continue
        object_counts = np.load(segment_dir / "object_counts.npy", allow_pickle=False)[:local_count]
        raw_object_offsets = np.load(segment_dir / "object_offsets.npy", allow_pickle=False)[:local_count]
        if len(object_counts):
            object_end = int(raw_object_offsets[-1]) + int(object_counts[-1])
        else:
            object_end = 0
        for name in _STATE_ARRAY_NAMES:
            arrays[name].append(np.load(segment_dir / name, allow_pickle=False)[:local_count])
        arrays["object_offsets.npy"].append(raw_object_offsets + np.uint32(object_offset))
        for name in _OBJECT_ARRAY_NAMES:
            arrays[name].append(np.load(segment_dir / name, allow_pickle=False)[:object_end])
        object_offset += int(object_end)
        loaded_state_count += int(local_count)
        for name in _VOCAB_ARRAY_NAMES:
            vocab = np.load(segment_dir / name, allow_pickle=False)
            existing = vocabs.get(name)
            if existing is not None and not _is_compatible_vocab_prefix(existing, vocab):
                raise ValueError(f"State-store vocab changed incompatibly: {name}")
            if existing is None or len(vocab) > len(existing):
                vocabs[name] = vocab
        if loaded_state_count >= int(cutoff_state_count):
            break
    if loaded_state_count != int(cutoff_state_count):
        raise ValueError(
            f"Loaded {loaded_state_count} states, expected {int(cutoff_state_count)}."
        )
    return arrays, vocabs


def _extract_or_resolve_segment(
    *,
    state_archive_dir: Path,
    segment: Mapping[str, Any],
    temp_root: Path,
) -> Path:
    rel_path = segment.get("path")
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError(f"Invalid state-store segment path: {segment!r}")
    path = state_archive_dir / rel_path
    storage = str(segment.get("storage", "") or "").strip().lower()
    if storage == "directory" or path.is_dir():
        return path
    if storage != "zip" and path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported state-store segment storage: {segment!r}")
    temp_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as handle:
        handle.extractall(temp_root)
    manifest_paths = list(temp_root.rglob("manifest.json"))
    if len(manifest_paths) != 1:
        raise ValueError(f"Expected one manifest.json in state-store zip: {path}")
    return manifest_paths[0].parent


def _write_combined_state_store(
    root: Path,
    *,
    arrays: Mapping[str, Sequence[np.ndarray]],
    vocabs: Mapping[str, np.ndarray],
    format_version: int,
) -> None:
    state_count = int(sum(len(chunk) for chunk in arrays["widths.npy"]))
    manifest = {
        "formatVersion": int(format_version),
        "stateCount": int(state_count),
        "firstStateId": 1,
        "lastStateId": int(state_count),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name in (*_STATE_ARRAY_NAMES, "object_offsets.npy", *_OBJECT_ARRAY_NAMES):
        chunks = list(arrays.get(name) or [])
        if not chunks:
            raise ValueError(f"Missing state-store array chunks: {name}")
        np.save(root / name, np.concatenate(chunks), allow_pickle=False)
    for name in _VOCAB_ARRAY_NAMES:
        vocab = vocabs.get(name)
        if vocab is None:
            raise ValueError(f"Missing state-store vocab: {name}")
        np.save(root / name, vocab, allow_pickle=False)


def _is_compatible_vocab_prefix(existing: np.ndarray, candidate: np.ndarray) -> bool:
    shorter, longer = (
        (existing, candidate)
        if len(existing) <= len(candidate)
        else (candidate, existing)
    )
    if len(shorter) == 0:
        return True
    return bool(np.array_equal(shorter, longer[: len(shorter)]))


def _build_transition_stores(
    *,
    state_store: StateStore,
    edge_rows: Sequence[Mapping[str, Any]],
    class_metadata_by_id: Mapping[int, Mapping[str, Any]],
) -> Tuple[List[Transition], ContrastiveSampleStore, Dict[str, int], Dict[str, str]]:
    canonical = CanonicalDataset(state_store=state_store)
    sample_store = ContrastiveSampleStore()
    class_by_key: Dict[str, int] = {}
    group_by_key: Dict[str, str] = {}
    for row in edge_rows:
        transition = _transition_from_edge_row(state_store=state_store, row=row)
        canonical.add(transition)
        transition_key = canonical_graph_edge_identity_key(transition)
        raw_group_id = row.get("group_id")
        group_id = (
            str(raw_group_id).strip()
            if isinstance(raw_group_id, str) and str(raw_group_id).strip()
            else None
        )
        class_id = _positive_class_id(row.get("class_id"))
        if group_id is None and class_id is not None:
            metadata = class_metadata_by_id.get(int(class_id))
            raw_group_id = metadata.get("group_id") if isinstance(metadata, Mapping) else None
            if isinstance(raw_group_id, str) and raw_group_id.strip():
                group_id = raw_group_id.strip()
        if group_id is not None:
            group_by_key[transition_key] = group_id
        if class_id is not None:
            class_by_key[transition_key] = int(class_id)
        sample_store.add(
            ContrastiveSample(
                state_json=None,
                action=int(row.get("action")),
                next_state_json=None,
                done=1.0 if bool(row.get("done")) else 0.0,
                class_id=class_id,
                state_id=int(row.get("source_state_id")),
                next_state_id=int(row.get("next_state_id")),
                state_store=state_store,
                source_world_index=int(row.get("world_index")),
                leaf_group_id=group_id,
                assignment_status="assigned" if class_id is not None else "unknown",
            )
        )
    return canonical.get_all(), sample_store, class_by_key, group_by_key


def _transition_from_edge_row(
    *,
    state_store: StateStore,
    row: Mapping[str, Any],
) -> Transition:
    raw_action_name = row.get("action_name")
    action_name = (
        str(raw_action_name).strip()
        if isinstance(raw_action_name, str) and raw_action_name.strip()
        else str(int(row.get("action")))
    )
    return Transition.from_state_ids(
        state_store=state_store,
        state_id=int(row.get("source_state_id")),
        action=action_name,
        next_state_id=int(row.get("next_state_id")),
        reward=float(row.get("env_reward", row.get("reward", 0.0)) or 0.0),
        done=bool(row.get("done")),
        world_index=int(row.get("world_index")),
        map_name=(
            str(row.get("map_name")).strip()
            if isinstance(row.get("map_name"), str) and str(row.get("map_name")).strip()
            else None
        ),
    )


def _positive_class_id(value: Any) -> Optional[int]:
    if isinstance(value, int) and int(value) > 0:
        return int(value)
    return None


def _compute_frontier_size(
    *,
    frontier_rows: Sequence[Mapping[str, Any]],
) -> int:
    total = 0
    header_count = 0
    for row in frontier_rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("row_type", "")).strip() != "frontier_world":
            continue
        try:
            live_count = int(row.get("live_count"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Frontier snapshot header has an invalid live_count.") from exc
        if live_count < 0:
            raise ValueError("Frontier snapshot live_count must be non-negative.")
        total += live_count
        header_count += 1
    if header_count <= 0:
        raise ValueError("Frontier snapshot contains no frontier_world headers.")
    return int(total)


def _resolve_checkpoint_metadata(
    source_run_dir: Path,
    *,
    policy: str,
) -> Tuple[Optional[Path], Optional[int], Optional[int], Optional[str], Optional[int]]:
    normalized = str(policy or "latest").strip()
    if normalized.lower() in {"", "none", "off", "false"}:
        return None, None, None, None, None
    if normalized.lower() == "latest":
        checkpoint_path = source_run_dir / "contrastive_checkpoints" / "latest.pt"
    else:
        checkpoint_path = Path(normalized)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (source_run_dir / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing contrastive checkpoint: {checkpoint_path}")
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return checkpoint_path, None, None, None, None
    if not isinstance(payload, Mapping):
        return checkpoint_path, None, None, None, None
    total_steps = payload.get("total_steps")
    total_updates = payload.get("total_updates")
    version_id = payload.get("current_version_id")
    # Absent in checkpoints written before latest.pt started carrying it.
    iteration = payload.get("iteration")
    return (
        checkpoint_path,
        int(total_steps) if isinstance(total_steps, int) else None,
        int(total_updates) if isinstance(total_updates, int) else None,
        str(version_id).strip() if isinstance(version_id, str) and version_id.strip() else None,
        int(iteration) if isinstance(iteration, int) and int(iteration) > 0 else None,
    )


def _resolve_int(value: Any, *, name: str) -> int:
    if isinstance(value, int):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Missing or invalid {name}: {value!r}") from None
