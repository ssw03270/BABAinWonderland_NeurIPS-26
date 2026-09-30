from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.llm.usage_summary import build_summary_from_events


_VERSION_ID_PATTERN = re.compile(r"^v(?P<ordinal>\d+)$", re.IGNORECASE)
_STEP_PATH_PATTERN = re.compile(r"(?:^|/)step(?P<step>\d+)(?:/|$)")


@dataclass(frozen=True)
class ProgramResumeBootstrapResult:
    source_run_dir: Path
    target_run_dir: Path
    through_version_id: str
    through_iteration: Optional[int]
    copied_version_ids: tuple[str, ...]
    copied_patch_attempt_dirs: tuple[str, ...]
    copied_transition_image_dirs: tuple[str, ...]
    llm_usage_events: tuple[Dict[str, Any], ...] = ()


def normalize_version_id(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    matched = _VERSION_ID_PATTERN.match(text)
    if matched is None:
        raise ValueError(f"Invalid version id: {raw_value!r}")
    return f"v{int(matched.group('ordinal')):03d}"


def parse_version_ordinal(raw_value: Any) -> Optional[int]:
    text = str(raw_value or "").strip()
    matched = _VERSION_ID_PATTERN.match(text)
    if matched is None:
        return None
    return int(matched.group("ordinal"))


def derive_last_program_patch_step(
    *,
    source_run_dir: str | Path,
    through_version_id: str,
    through_iteration: Optional[int] = None,
) -> Optional[int]:
    source_run_dir = Path(source_run_dir).resolve()
    target_ordinal = parse_version_ordinal(through_version_id)
    if target_ordinal is None:
        return None
    try:
        rows = _load_program_history_rows(source_run_dir / "program_history.jsonl")
    except FileNotFoundError:
        return None

    latest_version_row: Optional[Dict[str, Any]] = None
    latest_ordinal: Optional[int] = None
    for row in rows:
        if not _history_row_is_within_iteration(
            row,
            through_iteration=through_iteration,
        ):
            continue
        version_ordinal = parse_version_ordinal(row.get("version_id"))
        if (
            version_ordinal is None
            or int(version_ordinal) > int(target_ordinal)
            or not bool(row.get("accepted"))
        ):
            continue
        if latest_ordinal is None or int(version_ordinal) >= int(latest_ordinal):
            latest_ordinal = int(version_ordinal)
            latest_version_row = dict(row)

    if latest_version_row is None:
        return None
    direct_step = _history_row_step_index(latest_version_row)
    if direct_step is not None:
        return int(direct_step)
    patch_digest = latest_version_row.get("patch_digest")
    if not isinstance(patch_digest, str) or not patch_digest.strip():
        return 0

    matched_step: Optional[int] = None
    for row in rows:
        if not _history_row_is_within_iteration(
            row,
            through_iteration=through_iteration,
        ):
            continue
        if row.get("version_id") is not None:
            continue
        if not bool(row.get("accepted")):
            continue
        if row.get("patch_digest") != patch_digest:
            continue
        step = _history_row_step_index(row)
        if step is not None:
            matched_step = int(step)
    return matched_step


def _history_row_is_within_iteration(
    row: Dict[str, Any],
    *,
    through_iteration: Optional[int],
) -> bool:
    if through_iteration is None:
        return True
    raw_iteration = row.get("iteration")
    if isinstance(raw_iteration, bool):
        return True
    if isinstance(raw_iteration, int):
        return int(raw_iteration) <= int(through_iteration)
    return True


def _history_row_step_index(row: Dict[str, Any]) -> Optional[int]:
    for field_name in (
        "target_step_index",
        "step_index",
        "current_patch_step_index",
    ):
        raw_step = row.get(field_name)
        if isinstance(raw_step, bool):
            continue
        if isinstance(raw_step, int) and int(raw_step) > 0:
            return int(raw_step)
    return _extract_step_index_from_value(row.get("new_transition_image_paths"))


def _extract_step_index_from_value(value: Any) -> Optional[int]:
    if isinstance(value, str):
        text = value.replace("\\", "/")
        matched = _STEP_PATH_PATTERN.search(text)
        if matched is None:
            return None
        return int(matched.group("step"))
    if isinstance(value, dict):
        steps = [
            step
            for item in value.values()
            for step in [_extract_step_index_from_value(item)]
            if step is not None
        ]
        return max(steps) if steps else None
    if isinstance(value, list):
        steps = [
            step
            for item in value
            for step in [_extract_step_index_from_value(item)]
            if step is not None
        ]
        return max(steps) if steps else None
    return None


def _load_program_history_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing program history: {path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(dict(payload))
    return rows


def _load_llm_usage_event_rows(source_run_dir: Path) -> List[Dict[str, Any]]:
    usage_path = source_run_dir / "llm_usage_summary.json"
    if not usage_path.exists():
        return []
    payload = json.loads(usage_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return []
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, Mapping)]


def load_llm_usage_events_through_call_count(
    source_run_dir: str | Path,
    llm_calls: int,
) -> tuple[Dict[str, Any], ...]:
    target_count = max(0, int(llm_calls))
    if target_count == 0:
        return ()
    source = Path(source_run_dir).resolve()
    return tuple(_load_llm_usage_event_rows(source)[:target_count])


def _timestamp_sort_key(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.timestamp()


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _history_cutoff_metadata(
    rows: Sequence[Dict[str, Any]],
) -> tuple[Optional[int], Optional[float]]:
    cutoff_iteration: Optional[int] = None
    cutoff_timestamp: Optional[float] = None
    for row in rows:
        row_iteration = _optional_int(row.get("iteration"))
        if row_iteration is not None:
            if cutoff_iteration is None or int(row_iteration) > int(cutoff_iteration):
                cutoff_iteration = int(row_iteration)
        timestamp = _timestamp_sort_key(row.get("timestamp"))
        if timestamp is not None and (
            cutoff_timestamp is None or timestamp > cutoff_timestamp
        ):
            cutoff_timestamp = float(timestamp)
    return cutoff_iteration, cutoff_timestamp


def _event_is_within_program_resume_cutoff(
    event: Mapping[str, Any],
    *,
    cutoff_iteration: Optional[int],
    cutoff_timestamp: Optional[float],
) -> bool:
    trial_index = _optional_int(event.get("trial_index"))
    if trial_index is not None:
        return cutoff_iteration is not None and int(trial_index) <= int(cutoff_iteration)
    event_timestamp = _timestamp_sort_key(event.get("timestamp_utc"))
    return (
        cutoff_timestamp is not None
        and event_timestamp is not None
        and event_timestamp <= cutoff_timestamp
    )


def _select_llm_usage_events_through_history(
    *,
    source_run_dir: Path,
    selected_history_rows: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, Any], ...]:
    events = _load_llm_usage_event_rows(source_run_dir)
    if not events:
        return ()
    cutoff_iteration, cutoff_timestamp = _history_cutoff_metadata(selected_history_rows)
    selected = [
        event
        for event in events
        if _event_is_within_program_resume_cutoff(
            event,
            cutoff_iteration=cutoff_iteration,
            cutoff_timestamp=cutoff_timestamp,
        )
    ]
    return tuple(selected)


def _write_llm_usage_summary(
    *,
    target_run_dir: Path,
    events: Sequence[Mapping[str, Any]],
) -> None:
    if not events:
        return
    summary = build_summary_from_events(events)
    (target_run_dir / "llm_usage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _select_history_rows_through_version(
    rows: Sequence[Dict[str, Any]],
    *,
    through_version_id: str,
    through_iteration: Optional[int] = None,
) -> List[Dict[str, Any]]:
    target_ordinal = parse_version_ordinal(through_version_id)
    if target_ordinal is None:
        raise ValueError(f"Invalid through_version_id: {through_version_id!r}")
    cutoff_iteration = (
        int(through_iteration)
        if through_iteration is not None
        else None
    )

    selected: List[Dict[str, Any]] = []
    target_seen = False
    for row in rows:
        version_id = row.get("version_id")
        version_ordinal = parse_version_ordinal(version_id)
        accepted_version_row = bool(row.get("accepted")) and isinstance(
            version_ordinal,
            int,
        )
        if accepted_version_row and int(version_ordinal) > int(target_ordinal):
            break
        if cutoff_iteration is not None:
            raw_iteration = row.get("iteration")
            row_iteration = int(raw_iteration) if isinstance(raw_iteration, int) else None
            has_split_events = bool(row.get("split_events"))
            if row_iteration is not None and row_iteration > cutoff_iteration:
                continue
            if row_iteration is None and has_split_events:
                raise ValueError(
                    "Discovery resume cannot safely cut program_history.jsonl by "
                    "iteration because a split event row has no iteration field."
                )
        selected.append(dict(row))
        if accepted_version_row and int(version_ordinal) == int(target_ordinal):
            target_seen = True

    if not target_seen:
        raise ValueError(
            f"program_history.jsonl does not contain accepted version {through_version_id}."
        )
    return selected


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst)


def _relative_artifact_subdir(
    path_text: Any,
    *,
    root_name: str,
) -> Optional[PurePosixPath]:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    candidate = PurePosixPath(path_text)
    if candidate.is_absolute():
        return None
    parts = candidate.parts
    if len(parts) < 2 or parts[0] != root_name:
        return None
    return PurePosixPath(parts[0], parts[1])


def _iter_required_patch_attempt_dirs(
    rows: Sequence[Dict[str, Any]],
) -> Iterable[PurePosixPath]:
    for row in rows:
        for field_name in (
            "patch_attempt_artifact_dir",
            "patch_attempt_summary_json",
            "applied_program_path",
        ):
            relative_dir = _relative_artifact_subdir(
                row.get(field_name),
                root_name="patch_attempts",
            )
            if relative_dir is not None:
                yield relative_dir
        generator_attempt_artifacts = row.get("generator_attempt_artifacts")
        if not isinstance(generator_attempt_artifacts, list):
            continue
        for artifact_entry in generator_attempt_artifacts:
            if not isinstance(artifact_entry, dict):
                continue
            for field_name in ("prompt_path", "output_path", "reasoning_path", "error_path"):
                relative_dir = _relative_artifact_subdir(
                    artifact_entry.get(field_name),
                    root_name="patch_attempts",
                )
                if relative_dir is not None:
                    yield relative_dir


def _iter_required_transition_image_dirs(
    rows: Sequence[Dict[str, Any]],
) -> Iterable[PurePosixPath]:
    for row in rows:
        transition_image_paths = row.get("new_transition_image_paths")
        if not isinstance(transition_image_paths, dict):
            continue
        for relative_path in transition_image_paths.values():
            relative_dir = _relative_artifact_subdir(
                relative_path,
                root_name="new_transition_images",
            )
            if relative_dir is not None:
                yield relative_dir


def _copy_program_versions(
    *,
    source_run_dir: Path,
    target_run_dir: Path,
    through_version_id: str,
) -> tuple[str, ...]:
    source_versions_dir = source_run_dir / "program_versions"
    if not source_versions_dir.exists():
        raise FileNotFoundError(f"Missing program_versions directory: {source_versions_dir}")

    target_versions_dir = target_run_dir / "program_versions"
    target_versions_dir.mkdir(parents=True, exist_ok=True)

    max_ordinal = parse_version_ordinal(through_version_id)
    assert max_ordinal is not None

    copied_version_ids: List[str] = []
    for path in sorted(source_versions_dir.glob("v*.py")):
        version_ordinal = parse_version_ordinal(path.stem)
        if version_ordinal is None or int(version_ordinal) > int(max_ordinal):
            continue
        copied_version_ids.append(normalize_version_id(path.stem))
        shutil.copy2(path, target_versions_dir / path.name)

    if through_version_id not in copied_version_ids:
        raise FileNotFoundError(
            f"Missing program source for target version {through_version_id} "
            f"under {source_versions_dir}"
        )
    return tuple(copied_version_ids)


def _copy_required_patch_attempt_dirs(
    *,
    source_run_dir: Path,
    target_run_dir: Path,
    rows: Sequence[Dict[str, Any]],
) -> tuple[str, ...]:
    copied_dirs: List[str] = []
    selected_dirs = {
        rel_dir
        for rel_dir in _iter_required_patch_attempt_dirs(rows)
    }
    for rel_dir in sorted(selected_dirs, key=lambda value: value.as_posix()):
        src_dir = source_run_dir / Path(rel_dir.as_posix())
        if not src_dir.exists() or not src_dir.is_dir():
            raise FileNotFoundError(
                f"Missing patch attempt directory required for resume: {src_dir}"
            )
        dst_dir = target_run_dir / Path(rel_dir.as_posix())
        _copy_tree(src_dir, dst_dir)
        copied_dirs.append(rel_dir.as_posix())
    return tuple(copied_dirs)


def _copy_required_transition_image_dirs(
    *,
    source_run_dir: Path,
    target_run_dir: Path,
    rows: Sequence[Dict[str, Any]],
) -> tuple[str, ...]:
    copied_dirs: List[str] = []
    selected_dirs = {
        rel_dir
        for rel_dir in _iter_required_transition_image_dirs(rows)
    }
    for rel_dir in sorted(selected_dirs, key=lambda value: value.as_posix()):
        src_dir = source_run_dir / Path(rel_dir.as_posix())
        if not src_dir.exists() or not src_dir.is_dir():
            raise FileNotFoundError(
                f"Missing transition image directory required for resume: {src_dir}"
            )
        dst_dir = target_run_dir / Path(rel_dir.as_posix())
        _copy_tree(src_dir, dst_dir)
        copied_dirs.append(rel_dir.as_posix())
    return tuple(copied_dirs)


def bootstrap_program_resume_artifacts(
    *,
    source_run_dir: str | Path,
    target_run_dir: str | Path,
    through_version_id: str,
    through_iteration: Optional[int] = None,
) -> ProgramResumeBootstrapResult:
    source_run_dir = Path(source_run_dir).resolve()
    target_run_dir = Path(target_run_dir).resolve()
    safe_version_id = normalize_version_id(through_version_id)

    history_rows = _load_program_history_rows(source_run_dir / "program_history.jsonl")
    selected_history_rows = _select_history_rows_through_version(
        history_rows,
        through_version_id=safe_version_id,
        through_iteration=through_iteration,
    )
    copied_version_ids = _copy_program_versions(
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        through_version_id=safe_version_id,
    )
    copied_patch_attempt_dirs = _copy_required_patch_attempt_dirs(
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        rows=selected_history_rows,
    )
    copied_transition_image_dirs = _copy_required_transition_image_dirs(
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        rows=selected_history_rows,
    )

    history_path = target_run_dir / "program_history.jsonl"
    history_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected_history_rows)
        + "\n",
        encoding="utf-8",
    )
    llm_usage_events = _select_llm_usage_events_through_history(
        source_run_dir=source_run_dir,
        selected_history_rows=selected_history_rows,
    )
    _write_llm_usage_summary(
        target_run_dir=target_run_dir,
        events=llm_usage_events,
    )

    return ProgramResumeBootstrapResult(
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        through_version_id=safe_version_id,
        through_iteration=through_iteration,
        copied_version_ids=tuple(copied_version_ids),
        copied_patch_attempt_dirs=tuple(copied_patch_attempt_dirs),
        copied_transition_image_dirs=tuple(copied_transition_image_dirs),
        llm_usage_events=tuple(llm_usage_events),
    )
