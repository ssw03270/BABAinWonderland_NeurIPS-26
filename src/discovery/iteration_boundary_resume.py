from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import re
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
    llm_usage_events: Tuple[Dict[str, Any], ...]
    state_store: StateStore
    canonical_transitions: List[Transition]
    sample_store: ContrastiveSampleStore
    node_rows: List[Dict[str, Any]]
    edge_rows: List[Dict[str, Any]]
    final_edge_rows: List[Dict[str, Any]]
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
    program_bootstrap: Optional[ProgramResumeBootstrapResult] = None
    restored_sample_store_count: Optional[int] = None
    dashboard_history: Dict[str, Any] = field(default_factory=dict)

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
            "dashboard_history_points": int(
                len(self.dashboard_history.get("steps", ()))
                if isinstance(self.dashboard_history, dict)
                else 0
            ),
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
_SATURATION_LOG_PATTERN = re.compile(
    r"\[ITER\s+\d+\].*?\bglobal_step=(?P<global_step>\d+)\b"
    r".*?\bsat_step_hit=(?P<fail_count>\d+)/(?P<max_fail_count>\d+)\b"
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
    dashboard_history_limit: int = 240,
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
    class_metadata_by_id = _class_metadata_from_metric_row(cutoff_step)
    dashboard_group_by_key, dashboard_class_by_group_id = (
        _load_dashboard_transition_assignments(
            source_run_dir=source,
            cutoff_program_version_id=cutoff_version,
        )
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
            assignment_group_by_key=dashboard_group_by_key,
            assignment_class_by_group_id=dashboard_class_by_group_id,
        )
    )
    checkpoint_path, checkpoint_total_steps, checkpoint_total_updates, checkpoint_version_id = (
        _resolve_checkpoint_metadata(
            source,
            policy=checkpoint_policy,
        )
    )
    frontier_size = _compute_frontier_size(
        node_rows=node_rows,
        final_edge_rows=final_edge_rows,
    )
    dashboard_history = _build_resume_dashboard_history(
        source_run_dir=source,
        metric_archive_dir=archive / "metrics",
        through_iteration=cutoff_iteration,
        cutoff_global_step=cutoff_global_step,
        history_limit=dashboard_history_limit,
    )

    return IterationBoundaryResumeState(
        source_run_dir=source,
        latest_complete_iteration=latest_iteration,
        cutoff_iteration=cutoff_iteration,
        cutoff_global_step=cutoff_global_step,
        cutoff_program_version_id=cutoff_version,
        cutoff_state_count=cutoff_state_count,
        llm_calls=llm_calls,
        llm_usage_events=llm_usage_events,
        state_store=state_store,
        canonical_transitions=canonical_transitions,
        sample_store=sample_store,
        node_rows=node_rows,
        edge_rows=edge_rows,
        final_edge_rows=final_edge_rows,
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
        restored_sample_store_count=len(sample_store),
        dashboard_history=dashboard_history,
    )


def bootstrap_iteration_boundary_resume_artifacts(
    *,
    source_run_dir: str | Path,
    target_run_dir: str | Path,
    rollback_iterations: int = 1,
    through_iteration: Optional[int] = None,
    checkpoint_policy: str = "latest",
    dashboard_history_limit: int = 240,
) -> IterationBoundaryResumeState:
    state = build_iteration_boundary_resume_state(
        source_run_dir,
        rollback_iterations=rollback_iterations,
        through_iteration=through_iteration,
        checkpoint_policy=checkpoint_policy,
        dashboard_history_limit=dashboard_history_limit,
    )
    state.program_bootstrap = bootstrap_program_resume_artifacts(
        source_run_dir=state.source_run_dir,
        target_run_dir=target_run_dir,
        through_version_id=state.cutoff_program_version_id,
        through_iteration=state.cutoff_iteration,
    )
    return state


def _complete_archive_iterations(archive: Path) -> Tuple[int, ...]:
    iteration_sets: List[set[int]] = []
    for name in ("metrics", "nodes", "edges"):
        directory = archive / name
        if not directory.exists():
            raise FileNotFoundError(f"Missing archive directory: {directory}")
        iteration_sets.append(set(_archive_iterations(directory)))
    complete = set.intersection(*iteration_sets)
    return tuple(sorted(int(iteration) for iteration in complete))


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


_DASHBOARD_HISTORY_KEYS = (
    "contrastive_loss",
    "prototype_top1_accuracy",
    "rtotal",
    "rh",
    "rz",
    "rtotal_mean",
    "rh_mean",
    "rz_mean",
    "rtotal_std",
    "rh_std",
    "rz_std",
)


def _build_resume_dashboard_history(
    *,
    source_run_dir: Path,
    metric_archive_dir: Path,
    through_iteration: int,
    cutoff_global_step: int,
    history_limit: int,
) -> Dict[str, Any]:
    snapshot_history = _load_dashboard_history_snapshot(
        source_run_dir / ".discovery_web" / "dashboard.json",
        cutoff_global_step=cutoff_global_step,
        history_limit=history_limit,
    )
    if snapshot_history.get("steps"):
        return snapshot_history
    return _build_dashboard_history_from_metric_archive(
        metric_archive_dir,
        through_iteration=through_iteration,
        history_limit=history_limit,
    )


def _load_dashboard_history_snapshot(
    path: Path,
    *,
    cutoff_global_step: int,
    history_limit: int,
) -> Dict[str, Any]:
    limit = max(0, int(history_limit))
    if limit <= 0 or not path.exists():
        return {"steps": [], "metrics": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"steps": [], "metrics": {}}
    if not isinstance(payload, Mapping):
        return {"steps": [], "metrics": {}}
    history = payload.get("history")
    if not isinstance(history, Mapping):
        return {"steps": [], "metrics": {}}
    raw_steps = history.get("steps")
    raw_metrics = history.get("metrics")
    if not isinstance(raw_steps, list) or not isinstance(raw_metrics, Mapping):
        return {"steps": [], "metrics": {}}

    selected_indices: List[int] = []
    selected_steps: List[int] = []
    for index, raw_step in enumerate(raw_steps):
        step = _optional_int(raw_step)
        if step is None or int(step) > int(cutoff_global_step):
            continue
        selected_indices.append(int(index))
        selected_steps.append(int(step))
    if not selected_steps:
        return {"steps": [], "metrics": {}}
    selected_indices = selected_indices[-limit:]
    selected_steps = selected_steps[-limit:]

    metrics: Dict[str, List[Any]] = {}
    for raw_key, raw_values in raw_metrics.items():
        key = str(raw_key).strip()
        if not key or not isinstance(raw_values, list):
            continue
        values: List[Any] = []
        for index in selected_indices:
            values.append(raw_values[index] if index < len(raw_values) else None)
        metrics[key] = values
    return {"steps": selected_steps, "metrics": metrics}


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


def _append_bounded(values: List[Any], value: Any, *, limit: int) -> None:
    values.append(value)
    overflow = len(values) - int(limit)
    if overflow > 0:
        del values[:overflow]


def _rolling_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(value) for value in values) / float(len(values)))


def _rolling_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    mean_value = _rolling_mean(values)
    if mean_value is None:
        return None
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(
        len(values)
    )
    return float(math.sqrt(max(0.0, variance)))


def _build_dashboard_history_from_metric_archive(
    directory: Path,
    *,
    through_iteration: int,
    history_limit: int,
) -> Dict[str, Any]:
    limit = max(0, int(history_limit))
    if limit <= 0 or not directory.exists():
        return {"steps": [], "metrics": {}}

    candidate_rows: List[Dict[str, Any]] = []
    candidate_limit = max(limit, limit * 2)
    for iteration in sorted(
        (int(value) for value in _archive_iterations(directory)),
        reverse=True,
    ):
        if int(iteration) > int(through_iteration):
            continue
        rows = [
            row
            for row in _read_jsonl_gz(_archive_iteration_path(directory, iteration))
            if row.get("row_type") == "step"
        ]
        for row in reversed(rows):
            candidate_rows.append(row)
            if len(candidate_rows) >= candidate_limit:
                break
        if len(candidate_rows) >= candidate_limit:
            break
    candidate_rows.reverse()

    steps: List[int] = []
    metrics: Dict[str, List[Any]] = {key: [] for key in _DASHBOARD_HISTORY_KEYS}
    rolling_rh: List[float] = []
    rolling_rz: List[float] = []
    rolling_rtotal: List[float] = []

    for row in candidate_rows:
        step = (
            _optional_int(row.get("global_step"))
            or _optional_int(row.get("step"))
            or (int(steps[-1]) + 1 if steps else 0)
        )
        rh = _optional_float(row.get("rh"))
        rz = _optional_float(row.get("rz"))
        rtotal = _optional_float(row.get("rtotal"))
        if rh is not None:
            _append_bounded(rolling_rh, rh, limit=limit)
        if rz is not None:
            _append_bounded(rolling_rz, rz, limit=limit)
        if rtotal is not None:
            _append_bounded(rolling_rtotal, rtotal, limit=limit)

        point = {
            "contrastive_loss": _optional_float(row.get("contrastive_loss")),
            "prototype_top1_accuracy": _optional_float(
                row.get("prototype_top1_accuracy")
            ),
            "rtotal": rtotal,
            "rh": rh,
            "rz": rz,
            "rtotal_mean": _optional_float(row.get("rtotal_mean"))
            if row.get("rtotal_mean") is not None
            else _rolling_mean(rolling_rtotal),
            "rh_mean": _optional_float(row.get("rh_mean"))
            if row.get("rh_mean") is not None
            else _rolling_mean(rolling_rh),
            "rz_mean": _optional_float(row.get("rz_mean"))
            if row.get("rz_mean") is not None
            else _rolling_mean(rolling_rz),
            "rtotal_std": _optional_float(row.get("rtotal_std"))
            if row.get("rtotal_std") is not None
            else _rolling_std(rolling_rtotal),
            "rh_std": _optional_float(row.get("rh_std"))
            if row.get("rh_std") is not None
            else _rolling_std(rolling_rh),
            "rz_std": _optional_float(row.get("rz_std"))
            if row.get("rz_std") is not None
            else _rolling_std(rolling_rz),
        }
        _append_bounded(steps, int(step), limit=limit)
        for key in _DASHBOARD_HISTORY_KEYS:
            _append_bounded(metrics[key], point.get(key), limit=limit)

    return {
        "steps": steps,
        "metrics": {key: values for key, values in metrics.items() if values},
    }


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

    log_state: Dict[str, Optional[int | bool]] = {}
    if last_patch is None:
        log_state = _restore_saturation_state_from_log(
            source_run_dir=source_run_dir,
            cutoff_global_step=cutoff_global_step,
        )
        last_patch = _optional_nonnegative_int(log_state.get("last_patch_global_step"))
    max_fail_count = max_fail_count or _optional_positive_int(
        log_state.get("max_fail_count")
    )

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


def _restore_saturation_state_from_log(
    *,
    source_run_dir: Path,
    cutoff_global_step: int,
) -> Dict[str, Optional[int | bool]]:
    log_path = source_run_dir / ".discovery_web" / "log.txt"
    if not log_path.exists():
        return {}
    best: Optional[Dict[str, int]] = None
    try:
        with log_path.open("rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _SATURATION_LOG_PATTERN.search(line)
                if match is None:
                    continue
                global_step = int(match.group("global_step"))
                if global_step != int(cutoff_global_step):
                    continue
                best = {
                    "fail_count": int(match.group("fail_count")),
                    "max_fail_count": int(match.group("max_fail_count")),
                    "last_seen_global_step": int(global_step),
                }
    except OSError:
        return {}
    if best is None:
        return {}
    fail_count = int(best["fail_count"])
    max_fail_count = int(best["max_fail_count"])
    last_seen = int(best["last_seen_global_step"])
    return {
        "fail_count": fail_count,
        "max_fail_count": max_fail_count,
        "last_seen_global_step": last_seen,
        "last_patch_global_step": max(0, last_seen - fail_count),
        "is_saturated": fail_count >= max_fail_count,
    }


def _load_dashboard_transition_assignments(
    *,
    source_run_dir: Path,
    cutoff_program_version_id: str,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    dashboard_path = source_run_dir / ".discovery_web" / "dashboard.json"
    if not dashboard_path.exists():
        return {}, {}
    try:
        payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if not isinstance(payload, Mapping):
        return {}, {}
    program_context = payload.get("programContext")
    if not isinstance(program_context, Mapping):
        return {}, {}

    raw_current_version = program_context.get("current_version_id")
    if raw_current_version is not None:
        try:
            current_version = normalize_version_id(raw_current_version)
        except ValueError:
            return {}, {}
        if current_version != cutoff_program_version_id:
            return {}, {}

    group_context = program_context.get("group_context")
    if not isinstance(group_context, Mapping):
        return {}, {}
    raw_transition_groups = group_context.get("transition_leaf_group_ids")
    raw_leaf_classes = group_context.get("leaf_group_class_ids")
    if not isinstance(raw_transition_groups, Mapping) or not isinstance(
        raw_leaf_classes, Mapping
    ):
        return {}, {}

    group_by_key: Dict[str, str] = {}
    for raw_key, raw_group_id in raw_transition_groups.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            continue
        if not isinstance(raw_group_id, str) or not raw_group_id.strip():
            continue
        group_by_key[raw_key.strip()] = raw_group_id.strip()

    class_by_group_id: Dict[str, int] = {}
    for raw_group_id, raw_class_id in raw_leaf_classes.items():
        if not isinstance(raw_group_id, str) or not raw_group_id.strip():
            continue
        class_id = _positive_class_id(raw_class_id)
        if class_id is None:
            continue
        class_by_group_id[raw_group_id.strip()] = int(class_id)
    return group_by_key, class_by_group_id


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
    assignment_group_by_key: Optional[Mapping[str, str]] = None,
    assignment_class_by_group_id: Optional[Mapping[str, int]] = None,
) -> Tuple[List[Transition], ContrastiveSampleStore, Dict[str, int], Dict[str, str]]:
    canonical = CanonicalDataset(state_store=state_store)
    sample_store = ContrastiveSampleStore()
    class_by_key: Dict[str, int] = {}
    group_by_key: Dict[str, str] = {}
    assignment_group_by_key = assignment_group_by_key or {}
    assignment_class_by_group_id = assignment_class_by_group_id or {}
    for row in edge_rows:
        transition = _transition_from_edge_row(state_store=state_store, row=row)
        canonical.add(transition)
        transition_key = canonical_graph_edge_identity_key(transition)
        group_id = None

        class_id = None
        raw_assignment_group_id = assignment_group_by_key.get(transition_key)
        if isinstance(raw_assignment_group_id, str) and raw_assignment_group_id.strip():
            candidate_group_id = raw_assignment_group_id.strip()
            assigned_class_id = _positive_class_id(
                assignment_class_by_group_id.get(candidate_group_id)
            )
            if assigned_class_id is not None:
                group_id = candidate_group_id
                class_id = int(assigned_class_id)

        if class_id is None:
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
    node_rows: Sequence[Mapping[str, Any]],
    final_edge_rows: Sequence[Mapping[str, Any]],
) -> int:
    final_nodes: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in node_rows:
        try:
            key = (int(row.get("world_index")), int(row.get("state_id")))
        except (TypeError, ValueError):
            continue
        final_nodes[key] = dict(row)
    actions_by_world: Dict[int, set[int]] = {}
    executed_by_state: Dict[Tuple[int, int], set[int]] = {}
    for row in final_edge_rows:
        try:
            world_index = int(row.get("world_index"))
            state_id = int(row.get("source_state_id"))
            action = int(row.get("action"))
        except (TypeError, ValueError):
            continue
        actions_by_world.setdefault(world_index, set()).add(action)
        executed_by_state.setdefault((world_index, state_id), set()).add(action)
    action_count = max(
        (len(actions) for actions in actions_by_world.values()),
        default=0,
    )
    frontier_size = 0
    for key in final_nodes:
        world_index, state_id = key
        action_total = len(actions_by_world.get(world_index, set()))
        frontier_size += max(
            0,
            int(action_total or action_count) - len(executed_by_state.get((world_index, state_id), set())),
        )
    return int(frontier_size)


def _resolve_checkpoint_metadata(
    source_run_dir: Path,
    *,
    policy: str,
) -> Tuple[Optional[Path], Optional[int], Optional[int], Optional[str]]:
    normalized = str(policy or "latest").strip()
    if normalized.lower() in {"", "none", "off", "false"}:
        return None, None, None, None
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
        return checkpoint_path, None, None, None
    if not isinstance(payload, Mapping):
        return checkpoint_path, None, None, None
    total_steps = payload.get("total_steps")
    total_updates = payload.get("total_updates")
    version_id = payload.get("current_version_id")
    return (
        checkpoint_path,
        int(total_steps) if isinstance(total_steps, int) else None,
        int(total_updates) if isinstance(total_updates, int) else None,
        str(version_id).strip() if isinstance(version_id, str) and version_id.strip() else None,
    )


def _resolve_int(value: Any, *, name: str) -> int:
    if isinstance(value, int):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Missing or invalid {name}: {value!r}") from None
