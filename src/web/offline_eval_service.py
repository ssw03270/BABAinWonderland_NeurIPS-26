from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import unified_diff
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
import uuid

from fastapi import HTTPException
import yaml

from scripts.data_collect.evaluate_offline_dataset import (
    _append_bundle_lookup,
    _resolve_heuristic_mapping_bundle,
    load_offline_dataset,
)
from src.compare_runtime import build_comparison_difference_summary as _build_comparison_difference_summary
from src.data import Transition
from src.data.state_schema import dump_state_json
from src.discovery.artifact_renderer import TransitionArtifactRenderer
from src.environments.config_bundle import load_baba_config_bundle
from src.program_model import ProgramEvaluator, SandboxConfig, TransitionGroupClassifier, parse_state_json
from src.visualization import build_visualization_config
from src.web.common import PROJECT_ROOT
from src.web.offline_eval_dataset import (
    HeuristicEvalDataset,
    iter_discovery_class_metadata_rows,
    iter_discovery_transition_class_mapping_rows,
)
from src.web.offline_eval_package import (
    CompiledOfflineEvalPackage,
    load_or_build_compiled_offline_eval_package,
)


DEFAULT_OFFLINE_EVAL_DATASET_ROOT = PROJECT_ROOT / "test_dataset" / "solution"
DEFAULT_OFFLINE_EVAL_DISCOVERY_JSON = DEFAULT_OFFLINE_EVAL_DATASET_ROOT / "heuristic_dynamics_discovery.json"
DEFAULT_OFFLINE_EVAL_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "offline_eval_runs"
DEFAULT_OFFLINE_EVAL_SAMPLE_SEED = 42
DEFAULT_OFFLINE_EVAL_WORKERS = 16
DEFAULT_OFFLINE_EVAL_RESULT_PAGE_SIZE = 200
DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE = 16
DEFAULT_OFFLINE_EVAL_WORKER_BUNDLE_STATE_CACHE_SIZE = 64
DEFAULT_OFFLINE_EVAL_WORKER_MAX_LOADED_BUNDLES = 4
DEFAULT_OFFLINE_EVAL_CLASS_REPRESENTATIVE_LIMIT = 12
DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT = "all"
OFFLINE_EVAL_RUN_MODE_ACCURACY = "accuracy"
OFFLINE_EVAL_RUN_MODE_CLASS_PURITY = "class_purity"
MAX_OFFLINE_EVAL_RESULT_PAGE_SIZE = 1000
DEFAULT_OFFLINE_EVAL_RETENTION_DAYS = 7
DEFAULT_OFFLINE_EVAL_MAX_RUNS = 1
OFFLINE_EVAL_ATOMIC_REPLACE_RETRY_DELAYS_SEC = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.4,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
)
OFFLINE_EVAL_PROGRESS_WRITE_TARGET_STEPS = 200
OFFLINE_EVAL_PROGRESS_WRITE_MAX_COUNT_INTERVAL = 500
OFFLINE_EVAL_PROGRESS_WRITE_MIN_INTERVAL_SEC = 0.25
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs" / "env_config.yaml"
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "experiment_config_online.yaml"
EXPERIMENT_CONFIG_SNAPSHOT_FILENAMES = (
    "experiment_config.resolved.yaml",
    "experiment_config.yaml",
)

_OFFLINE_EVAL_SCENARIO_SPLIT_LABELS = {
    "all": "All maps",
    "all_train": "Train maps",
    "all_test": "Test maps",
}


class OfflineEvalExportCancelled(RuntimeError):
    """Raised when an offline eval export is cancelled by the user."""


_OFFLINE_EVAL_PROCESS_DATASET: Optional[HeuristicEvalDataset] = None
_OFFLINE_EVAL_PROCESS_EVALUATOR: Optional[ProgramEvaluator] = None
_OFFLINE_EVAL_PROCESS_SOURCE: str = ""

_CLASS_ANALYSIS_PROCESS_PACKAGE: Optional[CompiledOfflineEvalPackage] = None
_CLASS_ANALYSIS_PROCESS_EVALUATOR: Optional[ProgramEvaluator] = None
_CLASS_ANALYSIS_PROCESS_CLASSIFIER: Optional[TransitionGroupClassifier] = None
_CLASS_ANALYSIS_PROCESS_SOURCE: str = ""
_CLASS_ANALYSIS_PROCESS_WORD_ALIASES: Dict[str, Dict[str, str]] = {}


@dataclass(frozen=True)
class _ClassPurityTransitionRef:
    row_id: int
    heuristic_class_idx: int
    transition_index: int
    action: str
    scenario_type: str
    artifact_stem: str


@dataclass
class _OfflineEvalProgressWriteState:
    last_completed_count: int = 0
    last_write_monotonic: float = 0.0


def _class_purity_representative_id(
    *,
    heuristic_class_idx: int,
    row_id: int,
    role: str,
    repair_class_id: Optional[int],
) -> str:
    role_text = str(role or "sample").strip().lower().replace("_", "-") or "sample"
    class_text = f"c{int(repair_class_id)}" if isinstance(repair_class_id, int) and int(repair_class_id) > 0 else role_text
    return f"h{int(heuristic_class_idx)}-{class_text}-row{int(row_id)}-{role_text}"


def _build_class_purity_representative_payload(
    *,
    heuristic_class_idx: int,
    ref: _ClassPurityTransitionRef,
    result: Mapping[str, Any],
    role: str,
) -> Dict[str, Any]:
    raw_repair_class_id = result.get("repairClassId")
    repair_class_id = (
        int(raw_repair_class_id)
        if isinstance(raw_repair_class_id, int) and int(raw_repair_class_id) > 0
        else None
    )
    representative_id = _class_purity_representative_id(
        heuristic_class_idx=int(heuristic_class_idx),
        row_id=int(ref.row_id),
        role=role,
        repair_class_id=repair_class_id,
    )
    return {
        "representativeId": representative_id,
        "role": str(role),
        "rowId": int(ref.row_id),
        "heuristicClassIdx": int(heuristic_class_idx),
        "transitionIndex": int(ref.transition_index),
        "action": str(ref.action),
        "scenarioType": str(ref.scenario_type),
        "artifactStem": str(ref.artifact_stem),
        "explainable": bool(result.get("explainable")),
        "assigned": bool(result.get("assigned")),
        "repairClassId": repair_class_id,
        "repairGroupId": (
            str(result.get("repairGroupId"))
            if isinstance(result.get("repairGroupId"), str) and str(result.get("repairGroupId")).strip()
            else None
        ),
        "error": result.get("error") if isinstance(result.get("error"), Mapping) else None,
    }


def _offline_eval_worker_popen_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if sys.platform != "win32":
        return kwargs
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    kwargs["startupinfo"] = startupinfo
    kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return kwargs


def _pick_directory_via_windows_dialog(*, initial_dir: Optional[Path] = None) -> Optional[Path]:
    if sys.platform != "win32":
        raise HTTPException(
            status_code=501,
            detail="Native export folder picker is only supported on Windows.",
        )
    initial_path = ""
    if initial_dir is not None:
        initial_path = str(Path(initial_dir).expanduser().resolve())
    initial_directory_line = ""
    if initial_path:
        initial_directory_line = f"$dialog.InitialDirectory = {json.dumps(initial_path)}"
    script = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class OfflineEvalFolderPickerNativeMethods
{{
    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool BringWindowToTop(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool SetWindowPos(
        IntPtr hWnd,
        IntPtr hWndInsertAfter,
        int X,
        int Y,
        int cx,
        int cy,
        uint uFlags
    );

    [DllImport("kernel32.dll")]
    public static extern uint GetCurrentThreadId();

    public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
    public const uint SWP_NOSIZE = 0x0001;
    public const uint SWP_NOMOVE = 0x0002;
    public const uint SWP_SHOWWINDOW = 0x0040;
    public const int SW_RESTORE = 9;

    public static void ForceForegroundWindow(IntPtr hWnd)
    {{
        if (hWnd == IntPtr.Zero)
        {{
            return;
        }}
        IntPtr foreground = GetForegroundWindow();
        uint currentThreadId = GetCurrentThreadId();
        uint foregroundProcessId;
        uint foregroundThreadId = foreground == IntPtr.Zero
            ? 0
            : GetWindowThreadProcessId(foreground, out foregroundProcessId);
        bool attached = false;
        if (foregroundThreadId != 0 && foregroundThreadId != currentThreadId)
        {{
            attached = AttachThreadInput(currentThreadId, foregroundThreadId, true);
        }}
        try
        {{
            uint flags = SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW;
            ShowWindow(hWnd, SW_RESTORE);
            SetWindowPos(hWnd, HWND_TOPMOST, 0, 0, 0, 0, flags);
            BringWindowToTop(hWnd);
            SetForegroundWindow(hWnd);
        }}
        finally
        {{
            if (attached)
            {{
                AttachThreadInput(currentThreadId, foregroundThreadId, false);
            }}
        }}
    }}
}}
"@
$dialogTitle = "Choose folder for offline eval exports"
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = $dialogTitle
$dialog.Filter = "Folders|*.folder"
$dialog.CheckFileExists = $false
$dialog.CheckPathExists = $true
$dialog.ValidateNames = $false
$dialog.DereferenceLinks = $true
$dialog.Multiselect = $false
$dialog.AddExtension = $false
$dialog.RestoreDirectory = $false
$dialog.FileName = "Select Folder"
{initial_directory_line}
$owner = New-Object System.Windows.Forms.Form
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedToolWindow
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.TopMost = $true
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 50
$timer.Add_Tick({{
  $dialogHandle = [OfflineEvalFolderPickerNativeMethods]::FindWindow("#32770", $dialogTitle)
  if ($dialogHandle -eq [IntPtr]::Zero) {{
    $dialogHandle = [OfflineEvalFolderPickerNativeMethods]::FindWindow($null, $dialogTitle)
  }}
  if ($dialogHandle -eq [IntPtr]::Zero) {{
    return
  }}
  [OfflineEvalFolderPickerNativeMethods]::ForceForegroundWindow($dialogHandle)
  $timer.Stop()
}})
try {{
  $null = $owner.Show()
  $owner.Activate()
  [OfflineEvalFolderPickerNativeMethods]::ForceForegroundWindow($owner.Handle)
  $timer.Start()
  $result = $dialog.ShowDialog($owner)
  if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    $selectedPath = Split-Path -Parent $dialog.FileName
    if (-not $selectedPath) {{
      $selectedPath = $dialog.InitialDirectory
    }}
    if ($selectedPath) {{
      [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
      Write-Output $selectedPath
    }}
  }}
}}
finally {{
  $timer.Stop()
  $timer.Dispose()
  $dialog.Dispose()
  $owner.Dispose()
}}
"""
    try:
        kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-WindowStyle", "Hidden", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            check=False,
            **kwargs,
        )
    except OSError as error:
        raise HTTPException(status_code=500, detail="Failed to open export folder picker.") from error
    stderr_text = str(completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = stderr_text.splitlines()[-1] if stderr_text else "Failed to open export folder picker."
        raise HTTPException(status_code=500, detail=detail)
    selected = str(completed.stdout or "").strip()
    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _is_missing_path_error(error: BaseException) -> bool:
    return isinstance(error, (FileNotFoundError, NotADirectoryError))


def _path_is_missing(path: Path) -> bool:
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _iter_directory_paths(path: Path, *, reverse: bool = False) -> List[Path]:
    try:
        with os.scandir(path) as scan_iter:
            entries = sorted(
                list(scan_iter),
                key=lambda entry: entry.name.lower(),
                reverse=bool(reverse),
            )
    except OSError as error:
        if _is_missing_path_error(error):
            return []
        raise
    paths: List[Path] = []
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        paths.append(Path(entry.path))
    return paths


def _read_json(path: Path) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(6):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            if _is_missing_path_error(error):
                raise
            last_error = error
            if attempt >= 5:
                break
            time.sleep(0.03)
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping JSON at {path}")
        return payload
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to read JSON payload at {path}")


def _read_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return payload


def _normalize_word_aliases(
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    if not isinstance(word_aliases, Mapping):
        return {}
    normalized: Dict[str, Dict[str, str]] = {}
    for raw_type, raw_mapping in word_aliases.items():
        if not isinstance(raw_type, str) or not isinstance(raw_mapping, Mapping):
            continue
        obj_type = raw_type.strip().lower()
        if not obj_type:
            continue
        alias_map: Dict[str, str] = {}
        for raw_word, raw_alias in raw_mapping.items():
            if not isinstance(raw_word, str) or not isinstance(raw_alias, str):
                continue
            word = raw_word.strip().lower()
            alias = raw_alias.strip()
            if word and alias:
                alias_map[word] = alias
        if alias_map:
            normalized[obj_type] = dict(sorted(alias_map.items()))
    return dict(sorted(normalized.items()))


def _word_alias_count(word_aliases: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(len(mapping) for mapping in word_aliases.values() if isinstance(mapping, Mapping))


def _word_aliases_signature(word_aliases: Optional[Mapping[str, Mapping[str, Any]]]) -> str:
    normalized = _normalize_word_aliases(word_aliases)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_word_aliases_from_run_output_dir(
    run_output_dir: str | Path | None,
) -> tuple[Dict[str, Dict[str, str]], Optional[Path]]:
    if run_output_dir in {None, ""}:
        return {}, None
    try:
        resolved_run_dir = Path(run_output_dir).expanduser().resolve()
    except (OSError, RuntimeError):
        return {}, None
    snapshot_dir = resolved_run_dir / "config_snapshot"
    for snapshot_filename in EXPERIMENT_CONFIG_SNAPSHOT_FILENAMES:
        snapshot_path = snapshot_dir / snapshot_filename
        if not snapshot_path.exists():
            continue
        try:
            payload = _read_yaml(snapshot_path)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            return {}, snapshot_path
        serialization = payload.get("serialization")
        if not isinstance(serialization, Mapping):
            return {}, snapshot_path
        return _normalize_word_aliases(serialization.get("word_aliases")), snapshot_path
    return {}, None


def _version_sort_key(version_id: str) -> tuple[int, str]:
    text = str(version_id or "").strip()
    number_text = text[1:] if text.lower().startswith("v") else text
    if number_text.isdigit():
        return int(number_text), text
    return 10**9, text


def _legacy_group_commit_version(group_id: str, fallback: Optional[str] = None) -> str:
    text = str(group_id or "").strip()
    if ":" in text:
        commit_version = text.split(":", 1)[0].strip()
        if commit_version:
            return commit_version
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return "v_unknown"


def _read_optional_json_mapping(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = _read_json(path)
    except (OSError, UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_legacy_program_versions_from_dir(run_output_dir: Path) -> List[Dict[str, Any]]:
    program_dir = Path(run_output_dir) / "program_versions"
    if not program_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(program_dir.glob("*.py"), key=lambda item: _version_sort_key(item.stem)):
        version_id = str(path.stem).strip()
        if not version_id:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if not source.strip():
            continue
        sort_number, _sort_text = _version_sort_key(version_id)
        rows.append(
            {
                "version_id": version_id,
                "source": source,
                "index": sort_number if sort_number < 10**9 else len(rows),
            }
        )
    return rows


def _legacy_default_group_row(
    *,
    group_id: str,
    commit_version: Optional[str] = None,
) -> Dict[str, Any]:
    safe_group_id = str(group_id).strip()
    return {
        "group_id": safe_group_id,
        "commit_version": _legacy_group_commit_version(safe_group_id, commit_version),
        "parent_group_id": None,
        "child_group_ids": [],
        "member_transition_count": 0,
        "class_id": None,
        "retired_class_id": None,
        "split_broken_child_group_id": None,
        "split_kept_child_group_id": None,
        "split_program_source": None,
    }


def _ensure_legacy_group(
    groups: Dict[str, Dict[str, Any]],
    *,
    group_id: str,
    commit_version: Optional[str] = None,
) -> Dict[str, Any]:
    safe_group_id = str(group_id or "").strip()
    if not safe_group_id:
        raise ValueError("Legacy repair group id is empty.")
    group = groups.get(safe_group_id)
    if group is None:
        group = _legacy_default_group_row(
            group_id=safe_group_id,
            commit_version=commit_version,
        )
        groups[safe_group_id] = group
    elif (
        isinstance(commit_version, str)
        and commit_version.strip()
        and str(group.get("commit_version") or "").strip() in {"", "v_unknown"}
    ):
        group["commit_version"] = commit_version.strip()
    return group


def _append_unique_child_group_id(group: Dict[str, Any], child_group_id: str) -> None:
    child_ids = [
        str(value)
        for value in (group.get("child_group_ids") or [])
        if isinstance(value, str) and value.strip()
    ]
    if child_group_id not in child_ids:
        child_ids.append(child_group_id)
    group["child_group_ids"] = child_ids


def _read_legacy_applied_program_source(
    *,
    run_output_dir: Path,
    row: Mapping[str, Any],
) -> Optional[str]:
    raw_path = row.get("applied_program_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path(run_output_dir) / candidate
    try:
        source = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    return source if source.strip() else None


def _apply_legacy_split_event(
    *,
    groups: Dict[str, Dict[str, Any]],
    event: Mapping[str, Any],
    split_program_source: Optional[str],
) -> bool:
    parent_group_id = str(event.get("split_group_id") or "").strip()
    broken_child_group_id = str(event.get("broken_child_group_id") or "").strip()
    kept_child_group_id = str(event.get("kept_child_group_id") or "").strip()
    if not parent_group_id or not broken_child_group_id or not kept_child_group_id:
        return False
    commit_version = _legacy_group_commit_version(parent_group_id)
    parent = _ensure_legacy_group(
        groups,
        group_id=parent_group_id,
        commit_version=commit_version,
    )
    _append_unique_child_group_id(parent, broken_child_group_id)
    _append_unique_child_group_id(parent, kept_child_group_id)
    parent["class_id"] = None
    try:
        source_class_id = int(event.get("source_class_id"))
    except (TypeError, ValueError):
        source_class_id = 0
    if source_class_id > 0:
        parent["retired_class_id"] = int(source_class_id)
    parent["split_broken_child_group_id"] = broken_child_group_id
    parent["split_kept_child_group_id"] = kept_child_group_id
    if isinstance(split_program_source, str) and split_program_source.strip():
        parent["split_program_source"] = split_program_source

    child_specs = [
        (
            broken_child_group_id,
            event.get("broken_class_id"),
            event.get("broken_transition_keys"),
        ),
        (
            kept_child_group_id,
            event.get("kept_class_id"),
            event.get("kept_transition_keys"),
        ),
    ]
    total_member_count = 0
    for child_group_id, raw_class_id, raw_transition_keys in child_specs:
        transition_keys = [
            str(value)
            for value in (raw_transition_keys or [])
            if isinstance(value, str) and value.strip()
        ]
        total_member_count += len(transition_keys)
        child = _ensure_legacy_group(
            groups,
            group_id=child_group_id,
            commit_version=commit_version,
        )
        child["parent_group_id"] = parent_group_id
        child["member_transition_count"] = max(
            int(child.get("member_transition_count") or 0),
            len(transition_keys),
        )
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError):
            class_id = 0
        if class_id > 0 and not list(child.get("child_group_ids") or []):
            child["class_id"] = int(class_id)
    parent["member_transition_count"] = max(
        int(parent.get("member_transition_count") or 0),
        int(total_member_count),
    )
    return True


def _load_legacy_split_groups_from_history(
    *,
    run_output_dir: Path,
    groups: Dict[str, Dict[str, Any]],
) -> int:
    history_path = Path(run_output_dir) / "program_history.jsonl"
    if not history_path.is_file():
        return 0
    split_count = 0
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping):
                    continue
                split_events = row.get("split_events")
                if not isinstance(split_events, list) or not split_events:
                    continue
                split_program_source = _read_legacy_applied_program_source(
                    run_output_dir=run_output_dir,
                    row=row,
                )
                for event in split_events:
                    if not isinstance(event, Mapping):
                        continue
                    if _apply_legacy_split_event(
                        groups=groups,
                        event=event,
                        split_program_source=split_program_source,
                    ):
                        split_count += 1
    except (OSError, UnicodeError):
        return split_count
    return split_count


def _load_legacy_class_rows_from_dashboard(run_output_dir: Path) -> List[Dict[str, Any]]:
    dashboard_path = Path(run_output_dir) / ".discovery_web" / "dashboard.json"
    dashboard = _read_optional_json_mapping(dashboard_path)
    if not isinstance(dashboard, Mapping):
        return []
    rows = dashboard.get("classRows")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _apply_legacy_class_rows(
    *,
    groups: Dict[str, Dict[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    leaf_group_class_ids: Dict[str, int] = {}
    canonical_class_counts: Dict[str, int] = {}
    canonical_leaf_group_counts: Dict[str, int] = {}
    for row in class_rows:
        group_id = str(row.get("group_id") or row.get("version_id") or "").strip()
        if not group_id:
            continue
        commit_version = (
            str(row.get("commit_version") or "").strip()
            or _legacy_group_commit_version(group_id)
        )
        try:
            class_id = int(row.get("version_index"))
        except (TypeError, ValueError):
            continue
        if class_id <= 0:
            continue
        try:
            transition_count = int(row.get("transition_count") or 0)
        except (TypeError, ValueError):
            transition_count = 0
        group = _ensure_legacy_group(
            groups,
            group_id=group_id,
            commit_version=commit_version,
        )
        if not list(group.get("child_group_ids") or []):
            group["class_id"] = int(class_id)
        group["member_transition_count"] = max(
            int(group.get("member_transition_count") or 0),
            max(0, int(transition_count)),
        )
        leaf_group_class_ids[group_id] = int(class_id)
        canonical_class_counts[str(class_id)] = max(0, int(transition_count))
        canonical_leaf_group_counts[group_id] = max(0, int(transition_count))
    return leaf_group_class_ids, canonical_class_counts, canonical_leaf_group_counts


def _build_legacy_root_group_mapping(
    groups: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    root_group_id_by_commit_version: Dict[str, str] = {}
    for group_id, group in sorted(groups.items()):
        parent_group_id = group.get("parent_group_id")
        if isinstance(parent_group_id, str) and parent_group_id.strip():
            continue
        commit_version = _legacy_group_commit_version(
            str(group_id),
            str(group.get("commit_version") or ""),
        )
        if commit_version and commit_version != "v_unknown":
            root_group_id_by_commit_version.setdefault(commit_version, str(group_id))
    return dict(sorted(root_group_id_by_commit_version.items()))


def _load_legacy_program_context_from_run_output_dir(
    run_output_dir: str | Path | None,
) -> Optional[Dict[str, Any]]:
    if run_output_dir in {None, ""}:
        return None
    try:
        resolved_run_dir = Path(run_output_dir).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    versions = _load_legacy_program_versions_from_dir(resolved_run_dir)
    if not versions:
        return None
    class_rows = _load_legacy_class_rows_from_dashboard(resolved_run_dir)
    if not class_rows:
        return None
    groups: Dict[str, Dict[str, Any]] = {}
    split_count = _load_legacy_split_groups_from_history(
        run_output_dir=resolved_run_dir,
        groups=groups,
    )
    leaf_group_class_ids, canonical_class_counts, canonical_leaf_group_counts = _apply_legacy_class_rows(
        groups=groups,
        class_rows=class_rows,
    )
    if not groups or not leaf_group_class_ids:
        return None
    root_group_id_by_commit_version = _build_legacy_root_group_mapping(groups)
    if not root_group_id_by_commit_version:
        return None
    group_hierarchy = [
        dict(groups[group_id])
        for group_id in sorted(groups.keys())
    ]
    current_version_id = str(versions[-1].get("version_id") or "").strip()
    current_source = str(versions[-1].get("source") or "")
    return {
        "current_version_id": current_version_id,
        "current_source": current_source,
        "versions": versions,
        "group_context": {
            "generation": int(split_count),
            "root_group_id_by_commit_version": root_group_id_by_commit_version,
            "leaf_group_class_ids": dict(sorted(leaf_group_class_ids.items())),
            "active_class_ids": sorted(int(value) for value in leaf_group_class_ids.values()),
            "active_leaf_group_ids": [
                str(group_id)
                for group_id, _class_id in sorted(
                    leaf_group_class_ids.items(),
                    key=lambda item: (int(item[1]), str(item[0])),
                )
            ],
            "group_hierarchy": group_hierarchy,
            "canonical_class_counts": dict(sorted(canonical_class_counts.items(), key=lambda item: int(item[0]))),
            "canonical_leaf_group_counts": dict(sorted(canonical_leaf_group_counts.items())),
            "canonical_classified_count": int(sum(canonical_class_counts.values())),
            "canonical_unassigned_count": 0,
            "legacy_recovered": True,
            "legacy_run_output_dir": str(resolved_run_dir),
        },
    }


def _normalize_offline_eval_scenario_split(value: Optional[str]) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "all", "none"}:
        return "all"
    if text == "train":
        return "all_train"
    if text == "test":
        return "all_test"
    return text


def _resolve_config_path_from_project(value: str, *, base_config_path: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return (Path(base_config_path).resolve().parent / candidate).resolve()


def _manifest_group_has_split_rows(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        isinstance(value.get("all_train"), list)
        or isinstance(value.get("all_test"), list)
    )


@lru_cache(maxsize=1)
def _load_offline_eval_scenario_split_mapping() -> Dict[str, List[str]]:
    try:
        env_payload = _read_yaml(DEFAULT_ENV_CONFIG)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        return {}
    environment = env_payload.get("environment")
    if not isinstance(environment, Mapping):
        return {}
    kwargs = environment.get("kwargs")
    if not isinstance(kwargs, Mapping):
        return {}
    manifest_text = str(kwargs.get("split_manifest_path") or "").strip()
    if not manifest_text:
        return {}
    manifest_path = _resolve_config_path_from_project(
        manifest_text,
        base_config_path=DEFAULT_ENV_CONFIG,
    )
    try:
        manifest = _read_yaml(manifest_path)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        return {}

    split_group: Optional[Mapping[str, Any]] = None
    if _manifest_group_has_split_rows(manifest):
        split_group = manifest
    else:
        env_name = str(environment.get("name") or "").strip().lower()
        preferred_keys: List[str] = []
        if "original" in env_name:
            preferred_keys.append("original")
        preferred_keys.extend(
            sorted(str(key) for key in manifest.keys() if str(key) not in set(preferred_keys))
        )
        for key in preferred_keys:
            candidate = manifest.get(key)
            if _manifest_group_has_split_rows(candidate):
                split_group = candidate
                break
    if split_group is None:
        return {}

    split_mapping: Dict[str, List[str]] = {}
    for split_name in ("all_train", "all_test"):
        values = split_group.get(split_name)
        if not isinstance(values, list):
            continue
        normalized_values = [
            str(value).strip()
            for value in values
            if str(value).strip()
        ]
        if normalized_values:
            split_mapping[split_name] = normalized_values
    return split_mapping


def _offline_eval_scenario_split_options() -> List[Dict[str, Any]]:
    split_mapping = _load_offline_eval_scenario_split_mapping()
    all_scenarios = sorted(
        {
            scenario
            for values in split_mapping.values()
            for scenario in values
            if str(scenario).strip()
        }
    )
    options = [
        {
            "value": "all",
            "label": _OFFLINE_EVAL_SCENARIO_SPLIT_LABELS["all"],
            "scenarioCount": (len(all_scenarios) if all_scenarios else None),
        }
    ]
    for split_name in ("all_train", "all_test"):
        values = split_mapping.get(split_name, [])
        if not values:
            continue
        options.append(
            {
                "value": split_name,
                "label": _OFFLINE_EVAL_SCENARIO_SPLIT_LABELS[split_name],
                "scenarioCount": len(values),
            }
        )
    return options


def _resolve_offline_eval_allowed_scenario_types(
    scenario_split: Optional[str],
    *,
    required: bool = False,
) -> Optional[List[str]]:
    normalized_split = _normalize_offline_eval_scenario_split(scenario_split)
    if normalized_split == "all":
        return None
    split_mapping = _load_offline_eval_scenario_split_mapping()
    values = split_mapping.get(normalized_split)
    if values:
        return list(values)
    if required:
        raise ValueError(
            "Offline eval scenario split is unavailable from the current env config: "
            f"{normalized_split}"
        )
    return None


def _count_discovery_json_classes(
    path: Path,
    *,
    dataset_root: Optional[Path] = None,
    allowed_scenario_types: Optional[Sequence[str]] = None,
) -> int:
    allowed_scenarios = (
        frozenset(str(value).strip() for value in allowed_scenario_types if str(value).strip())
        if allowed_scenario_types is not None
        else None
    )

    if allowed_scenarios is None:
        class_ids: set[int] = set()
        for row in iter_discovery_class_metadata_rows(path):
            if not isinstance(row, Mapping):
                continue
            try:
                class_ids.add(int(row.get("class_idx")))
            except (TypeError, ValueError):
                continue
        if class_ids:
            return int(len(class_ids))

        for row in iter_discovery_transition_class_mapping_rows(path):
            if not isinstance(row, Mapping):
                continue
            transition_to_class = row.get("transition_to_class")
            if not isinstance(transition_to_class, Mapping):
                continue
            for raw_class_idx in transition_to_class.values():
                try:
                    class_ids.add(int(raw_class_idx))
                except (TypeError, ValueError):
                    continue
        if not class_ids:
            raise ValueError(f"No heuristic classes found in discovery JSON: {path}")
        return int(len(class_ids))
    if dataset_root is None:
        class_ids: set[int] = set()
        for row in iter_discovery_transition_class_mapping_rows(path):
            if not isinstance(row, Mapping):
                continue
            scenario_type = str(row.get("scenario_type") or "").strip()
            if allowed_scenarios is not None and scenario_type not in allowed_scenarios:
                continue
            transition_to_class = row.get("transition_to_class")
            if not isinstance(transition_to_class, Mapping):
                continue
            for raw_class_idx in transition_to_class.values():
                try:
                    class_ids.add(int(raw_class_idx))
                except (TypeError, ValueError):
                    continue
        if not class_ids:
            raise ValueError(f"No heuristic classes found in discovery JSON: {path}")
        return int(len(class_ids))

    bundles = load_offline_dataset(
        dataset_root=dataset_root,
        show_progress=False,
    )
    bundle_by_transition_path = {
        str(Path(bundle.transitions_path).resolve()): bundle
        for bundle in bundles
    }
    bundles_by_transition_name: Dict[str, List[Any]] = {}
    bundles_by_artifact_stem: Dict[str, List[Any]] = {}
    bundles_by_scenario_type: Dict[str, List[Any]] = {}
    for bundle in bundles:
        _append_bundle_lookup(bundles_by_transition_name, Path(bundle.transitions_path).name, bundle)
        _append_bundle_lookup(bundles_by_artifact_stem, str(bundle.artifact_stem), bundle)
        _append_bundle_lookup(bundles_by_scenario_type, str(bundle.scenario_type), bundle)

    class_ids: set[int] = set()
    for row in iter_discovery_transition_class_mapping_rows(path):
        if not isinstance(row, Mapping):
            continue
        bundle = _resolve_heuristic_mapping_bundle(
            row=row,
            bundle_by_transition_path=bundle_by_transition_path,
            bundles_by_transition_name=bundles_by_transition_name,
            bundles_by_artifact_stem=bundles_by_artifact_stem,
            bundles_by_scenario_type=bundles_by_scenario_type,
        )
        if bundle is None:
            continue
        if allowed_scenarios is not None and str(bundle.scenario_type) not in allowed_scenarios:
            continue
        transition_to_class = row.get("transition_to_class")
        if not isinstance(transition_to_class, Mapping):
            continue
        for raw_class_idx in transition_to_class.values():
            try:
                class_ids.add(int(raw_class_idx))
            except (TypeError, ValueError):
                continue
    if not class_ids:
        raise ValueError(
            "No heuristic-dynamics classes matched the selected evaluation bundles. "
            f"Source JSON: {path}"
        )
    return int(len(class_ids))


def _count_full_dataset_transitions(
    dataset_root: Path,
    *,
    allowed_scenario_types: Optional[Sequence[str]] = None,
) -> int:
    resolved_dataset_root = Path(dataset_root).resolve()
    bundles = load_offline_dataset(
        dataset_root=resolved_dataset_root,
        allowed_scenarios=allowed_scenario_types,
        show_progress=False,
    )
    bundle_count = sum(max(0, int(bundle.transition_count)) for bundle in bundles)
    if bundle_count <= 0:
        raise ValueError(f"No transitions found under {resolved_dataset_root}")
    return int(bundle_count)


def _is_retryable_atomic_replace_error(error: BaseException) -> bool:
    if not isinstance(error, PermissionError):
        return False
    winerror = getattr(error, "winerror", None)
    if winerror in {5, 32}:
        return True
    return os.name == "nt" and winerror is None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for delay_sec in OFFLINE_EVAL_ATOMIC_REPLACE_RETRY_DELAYS_SEC:
            try:
                temp_path.replace(path)
                return
            except PermissionError as error:
                if not _is_retryable_atomic_replace_error(error):
                    raise
                time.sleep(float(delay_sec))
        temp_path.replace(path)
    finally:
        with suppress(OSError):
            if temp_path.exists():
                temp_path.unlink()


def _offline_eval_progress_write_count_interval(total_count: int) -> int:
    return max(
        1,
        min(
            OFFLINE_EVAL_PROGRESS_WRITE_MAX_COUNT_INTERVAL,
            int(total_count) // OFFLINE_EVAL_PROGRESS_WRITE_TARGET_STEPS or 1,
        ),
    )


def _write_progress_summary_if_due(
    *,
    run_path: Path,
    summary: Mapping[str, Any],
    state: _OfflineEvalProgressWriteState,
    completed_count: int,
    total_count: int,
    count_interval: int,
) -> bool:
    completed = int(completed_count)
    total = int(total_count)
    now = time.monotonic()
    if completed >= total:
        return False
    should_write = False
    if not should_write and completed - int(state.last_completed_count) >= max(1, int(count_interval)):
        should_write = True
    if (
        not should_write
        and completed != int(state.last_completed_count)
        and now - float(state.last_write_monotonic) >= OFFLINE_EVAL_PROGRESS_WRITE_MIN_INTERVAL_SEC
    ):
        should_write = True
    if not should_write:
        return False
    _write_json_atomic(run_path, summary)
    state.last_completed_count = completed
    state.last_write_monotonic = now
    return True


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attempt in range(6):
        try:
            with path.open("r", encoding="utf-8") as handle:
                rows = []
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
            return rows
        except OSError as error:
            if _is_missing_path_error(error):
                return rows
            if attempt >= 5:
                raise
            time.sleep(0.03)
    return rows


def _normalize_offline_eval_result_page_limit(limit: int) -> int:
    return max(1, min(MAX_OFFLINE_EVAL_RESULT_PAGE_SIZE, int(limit)))


def _normalize_offline_eval_result_offset(offset: int) -> int:
    return max(0, int(offset))


def _normalize_offline_eval_status_filter(status_filter: Optional[str]) -> str:
    normalized = str(status_filter or "all").strip().lower()
    allowed = {
        "all",
        "failed",
        "pass",
        "mismatch",
        "runtime_error",
        "compile_error",
    }
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown offline eval status filter: {status_filter}")
    return normalized


def _normalize_offline_eval_sort(sort: Optional[str]) -> str:
    normalized = str(sort or "desc").strip().lower()
    if normalized not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail=f"Unknown offline eval sort order: {sort}")
    return normalized


def _offline_eval_row_matches_status(payload: Mapping[str, Any], status_filter: str) -> bool:
    normalized_filter = _normalize_offline_eval_status_filter(status_filter)
    if normalized_filter == "all":
        return True
    status = str(payload.get("status", "")).strip().lower()
    if normalized_filter == "failed":
        return status not in {"", "pass"}
    return status == normalized_filter


def _offline_eval_filtered_total_count(
    summary: Mapping[str, Any],
    *,
    status_filter: str,
) -> int:
    normalized_filter = _normalize_offline_eval_status_filter(status_filter)
    progress = summary.get("progress") if isinstance(summary.get("progress"), Mapping) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    if _is_class_purity_summary(summary):
        if normalized_filter == "all":
            dataset = summary.get("dataset") if isinstance(summary.get("dataset"), Mapping) else {}
            return max(
                0,
                int(
                    metrics.get("resultRowCount")
                    or dataset.get("heuristicClassCount")
                    or dataset.get("classCount")
                    or 0
                ),
            )
        if normalized_filter == "failed":
            return max(0, int(metrics.get("failedCount", 0) or 0))
        if normalized_filter == "pass":
            return max(0, int(metrics.get("correctCount", 0) or 0))
        if normalized_filter == "mismatch":
            return max(0, int(metrics.get("mismatchCount", 0) or 0))
        if normalized_filter == "runtime_error":
            return max(0, int(metrics.get("runtimeErrorCount", 0) or 0))
        if normalized_filter == "compile_error":
            return 0
    if normalized_filter == "all":
        return max(0, int(progress.get("completedCount", 0) or 0))
    if normalized_filter == "failed":
        return max(0, int(metrics.get("failedCount", 0) or 0))
    if normalized_filter == "pass":
        return max(0, int(metrics.get("correctCount", 0) or 0))
    if normalized_filter == "mismatch":
        return max(0, int(metrics.get("mismatchCount", 0) or 0))
    if normalized_filter == "runtime_error":
        return max(0, int(metrics.get("runtimeErrorCount", 0) or 0))
    if normalized_filter == "compile_error":
        return max(0, int(metrics.get("compileErrorCount", 0) or 0))
    return 0


def _read_jsonl_page(
    path: Path,
    *,
    offset: int,
    limit: int,
    status_filter: str,
    sort: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    normalized_offset = _normalize_offline_eval_result_offset(offset)
    normalized_limit = _normalize_offline_eval_result_page_limit(limit)
    normalized_filter = _normalize_offline_eval_status_filter(status_filter)
    normalized_sort = _normalize_offline_eval_sort(sort)
    window_size = normalized_offset + normalized_limit
    if window_size <= 0:
        return rows

    for attempt in range(6):
        try:
            if normalized_sort == "desc":
                window: deque[Dict[str, Any]] = deque(maxlen=window_size)
                with path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        if not _offline_eval_row_matches_status(payload, normalized_filter):
                            continue
                        window.append(payload)
                rows = list(window)
                rows.reverse()
                return rows[normalized_offset : normalized_offset + normalized_limit]

            seen = 0
            with path.open("r", encoding="utf-8") as handle:
                rows = []
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if not _offline_eval_row_matches_status(payload, normalized_filter):
                        continue
                    if seen < normalized_offset:
                        seen += 1
                        continue
                    rows.append(payload)
                    if len(rows) >= normalized_limit:
                        break
            return rows
        except OSError as error:
            if _is_missing_path_error(error):
                return rows
            if attempt >= 5:
                raise
            time.sleep(0.03)
    return rows


def _text_sha1(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _serialize_error(error: Any) -> Optional[Dict[str, Optional[str]]]:
    if error is None:
        return None
    return {
        "phase": str(getattr(error, "phase", "")).strip() or None,
        "message": str(getattr(error, "message", "")).strip() or None,
        "exceptionType": str(getattr(error, "exception_type", "")).strip() or None,
    }


def _build_diff_summary(expected_canonical: str, predicted_canonical: Optional[str]) -> str:
    expected_lines = str(expected_canonical or "").splitlines()
    predicted_lines = str(predicted_canonical or "").splitlines()
    diff_lines = list(
        unified_diff(
            expected_lines,
            predicted_lines,
            fromfile="expected",
            tofile="predicted",
            lineterm="",
            n=2,
        )
    )
    if not diff_lines:
        return "No textual diff."
    return "\n".join(diff_lines[:48])


def _resolve_available_failure_images(predicted_canonical: Optional[str]) -> Dict[str, bool]:
    predicted_available = False
    if isinstance(predicted_canonical, str) and predicted_canonical.strip():
        with suppress(ValueError):
            parse_state_json(predicted_canonical)
            predicted_available = True
    return {
        "previous": True,
        "expected": True,
        "predicted": predicted_available,
        "comparison": predicted_available,
    }


def _normalize_available_failure_images(payload: Any, predicted_canonical: Optional[str]) -> Dict[str, bool]:
    if isinstance(payload, Mapping):
        return {
            "previous": bool(payload.get("previous")),
            "expected": bool(payload.get("expected")),
            "predicted": bool(payload.get("predicted")),
            "comparison": bool(payload.get("comparison")),
        }
    return _resolve_available_failure_images(predicted_canonical)


def _format_prediction_error_message(error: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(error, Mapping):
        return None
    phase = str(error.get("phase", "")).strip()
    message = str(error.get("message", "")).strip()
    if phase and message:
        return f"{phase}: {message}"
    return message or phase or None


def _resolve_class_status(*, is_correct: bool, error: Optional[Mapping[str, Any]]) -> str:
    if is_correct and error is None:
        return "pass"
    phase = str(error.get("phase", "")).strip().lower() if isinstance(error, Mapping) else ""
    if phase in {"parse", "compile", "contract"}:
        return "compile_error"
    if error is not None:
        return "runtime_error"
    return "mismatch"


def _normalize_offline_eval_workers(workers: int) -> int:
    return max(1, int(workers))


def _iter_fixed_size_chunks(values: Iterable[Any], chunk_size: int) -> Iterable[List[Any]]:
    resolved_chunk_size = max(1, int(chunk_size))
    chunk: List[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= resolved_chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _normalize_offline_eval_run_mode(value: Optional[str]) -> str:
    normalized = str(value or OFFLINE_EVAL_RUN_MODE_ACCURACY).strip().lower().replace("-", "_")
    if normalized in {"", "eval", "accuracy", "offline_eval"}:
        return OFFLINE_EVAL_RUN_MODE_ACCURACY
    if normalized in {"class", "classes", "class_purity", "dynamics_class", "dynamics_classes"}:
        return OFFLINE_EVAL_RUN_MODE_CLASS_PURITY
    raise HTTPException(status_code=400, detail=f"Unknown offline eval run mode: {value}")


def _is_class_purity_summary(summary: Mapping[str, Any]) -> bool:
    run_mode = _normalize_offline_eval_run_mode(summary.get("runMode"))
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), Mapping) else {}
    selection_mode = str(dataset.get("selectionMode") or "").strip().lower()
    return run_mode == OFFLINE_EVAL_RUN_MODE_CLASS_PURITY or selection_mode == OFFLINE_EVAL_RUN_MODE_CLASS_PURITY


def _resolve_class_analysis_context(
    *,
    program_context: Mapping[str, Any],
    version_key: Optional[str],
    source: str,
) -> tuple[Dict[str, Any], str, str]:
    if not isinstance(program_context, Mapping):
        raise ValueError("Class analysis requires a stored discovery program context.")
    group_context = program_context.get("group_context")
    if not isinstance(group_context, Mapping) or not isinstance(group_context.get("group_hierarchy"), list):
        raise ValueError("Class analysis requires a stored repair group context.")
    raw_versions = program_context.get("versions")
    if not isinstance(raw_versions, list) or not raw_versions:
        raise ValueError("Class analysis requires stored program versions.")

    version_rows: List[Dict[str, Any]] = []
    for raw in raw_versions:
        if not isinstance(raw, Mapping):
            continue
        version_id = str(raw.get("version_id") or "").strip()
        version_source = str(raw.get("source") or "")
        if not version_id or not version_source.strip():
            continue
        version_rows.append(
            {
                "version_id": version_id,
                "source": version_source,
                "index": raw.get("index"),
            }
        )
    if not version_rows:
        raise ValueError("Class analysis requires at least one source-bearing program version.")

    requested_version = str(version_key or "").strip()
    current_version = str(program_context.get("current_version_id") or "").strip()
    selected_version = requested_version or current_version or str(version_rows[-1]["version_id"])
    selected_index = None
    for index, row in enumerate(version_rows):
        if str(row.get("version_id") or "").strip() == selected_version:
            selected_index = int(index)
            break
    if selected_index is None:
        raise ValueError(f"Class analysis version is not present in the stored program context: {selected_version}")

    selected_source = str(source or "").strip() or str(version_rows[selected_index].get("source") or "").strip()
    if not selected_source:
        raise ValueError(f"Class analysis version has no source: {selected_version}")

    active_versions = [dict(row) for row in version_rows[: selected_index + 1]]
    active_versions[-1]["source"] = selected_source
    context = {
        "current_version_id": selected_version,
        "current_source": selected_source,
        "versions": active_versions,
        "group_context": dict(group_context),
    }
    return context, selected_source, selected_version


def _state_obj_with_word_aliases(
    *,
    package: CompiledOfflineEvalPackage,
    state_id: int,
    word_aliases: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    state_obj = package.state_store.state_obj(int(state_id))
    normalized_aliases = _normalize_word_aliases(word_aliases)
    if not normalized_aliases:
        return state_obj
    objects: List[Dict[str, Any]] = []
    for raw_object in state_obj.get("objects", []):
        if not isinstance(raw_object, Mapping):
            continue
        row = dict(raw_object)
        obj_type = str(row.get("type", "")).strip().lower()
        word = str(row.get("word", "")).strip().lower()
        alias = normalized_aliases.get(obj_type, {}).get(word)
        if alias:
            row["word"] = alias
        objects.append(row)
    aliased_state = dict(state_obj)
    aliased_state["objects"] = objects
    return aliased_state


def _build_class_analysis_transition(
    *,
    package: CompiledOfflineEvalPackage,
    row_id: int,
    word_aliases: Mapping[str, Mapping[str, str]],
) -> Transition:
    normalized_aliases = _normalize_word_aliases(word_aliases)
    if not normalized_aliases:
        return package.build_transition(int(row_id))
    meta = package.transition_meta(int(row_id))
    previous_state_obj = _state_obj_with_word_aliases(
        package=package,
        state_id=int(meta.state_id),
        word_aliases=normalized_aliases,
    )
    next_state_obj = _state_obj_with_word_aliases(
        package=package,
        state_id=int(meta.next_state_id),
        word_aliases=normalized_aliases,
    )
    return Transition(
        state=dump_state_json(previous_state_obj),
        action=str(meta.action),
        next_state=dump_state_json(next_state_obj),
        reward=float(meta.reward),
        done=bool(meta.done),
        map_name=str(meta.bundle.scenario_type),
    )


def _load_class_purity_metadata_by_id(discovery_json: Path) -> Dict[int, Dict[str, Any]]:
    class_metadata_by_id: Dict[int, Dict[str, Any]] = {}
    for row in iter_discovery_class_metadata_rows(Path(discovery_json)):
        if not isinstance(row, Mapping):
            continue
        try:
            class_idx = int(row.get("class_idx"))
        except (TypeError, ValueError):
            continue
        class_metadata_by_id[int(class_idx)] = dict(row)
    return class_metadata_by_id


def _iter_class_purity_transition_refs(
    *,
    package: CompiledOfflineEvalPackage,
    discovery_json: Path,
    allowed_scenario_types: Optional[Sequence[str]],
    class_metadata_by_id: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Iterable[_ClassPurityTransitionRef]:
    resolved_class_metadata_by_id = (
        class_metadata_by_id
        if isinstance(class_metadata_by_id, Mapping)
        else _load_class_purity_metadata_by_id(Path(discovery_json))
    )
    allowed_scenarios = (
        frozenset(str(value).strip() for value in allowed_scenario_types if str(value).strip())
        if allowed_scenario_types is not None
        else None
    )
    bundle_by_transition_path = {
        str(bundle.transitions_path.resolve()): bundle
        for bundle in package.bundles
    }
    bundles_by_transition_name: Dict[str, List[Any]] = {}
    bundles_by_artifact_stem: Dict[str, List[Any]] = {}
    bundles_by_scenario_type: Dict[str, List[Any]] = {}
    for bundle in package.bundles:
        _append_bundle_lookup(bundles_by_transition_name, bundle.transitions_path.name, bundle)
        _append_bundle_lookup(bundles_by_artifact_stem, str(bundle.artifact_stem), bundle)
        _append_bundle_lookup(bundles_by_scenario_type, str(bundle.scenario_type), bundle)

    for row in iter_discovery_transition_class_mapping_rows(Path(discovery_json)):
        if not isinstance(row, Mapping):
            continue
        bundle = _resolve_heuristic_mapping_bundle(
            row=row,
            bundle_by_transition_path=bundle_by_transition_path,
            bundles_by_transition_name=bundles_by_transition_name,
            bundles_by_artifact_stem=bundles_by_artifact_stem,
            bundles_by_scenario_type=bundles_by_scenario_type,
        )
        if bundle is None:
            continue
        if allowed_scenarios is not None and str(bundle.scenario_type) not in allowed_scenarios:
            continue
        transition_to_class = row.get("transition_to_class")
        if not isinstance(transition_to_class, Mapping):
            continue
        for raw_transition_index, raw_class_idx in transition_to_class.items():
            try:
                transition_index = int(raw_transition_index)
                heuristic_class_idx = int(raw_class_idx)
            except (TypeError, ValueError):
                continue
            row_id = package.resolve_transition_row_id(
                bundle_key=str(bundle.bundle_key),
                transition_index=int(transition_index),
            )
            if row_id is None:
                continue
            meta = package.transition_meta(int(row_id))
            metadata = resolved_class_metadata_by_id.get(int(heuristic_class_idx), {})
            if not isinstance(metadata, Mapping):
                metadata = resolved_class_metadata_by_id.get(str(heuristic_class_idx), {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            action = str(metadata.get("action", meta.action)).strip() or str(meta.action)
            yield _ClassPurityTransitionRef(
                row_id=int(row_id),
                heuristic_class_idx=int(heuristic_class_idx),
                transition_index=int(transition_index),
                action=action,
                scenario_type=str(bundle.scenario_type),
                artifact_stem=str(bundle.artifact_stem),
            )


def _close_class_analysis_process_state() -> None:
    global _CLASS_ANALYSIS_PROCESS_PACKAGE
    global _CLASS_ANALYSIS_PROCESS_EVALUATOR
    global _CLASS_ANALYSIS_PROCESS_CLASSIFIER
    if _CLASS_ANALYSIS_PROCESS_PACKAGE is not None:
        with suppress(Exception):
            _CLASS_ANALYSIS_PROCESS_PACKAGE.close()
    if _CLASS_ANALYSIS_PROCESS_EVALUATOR is not None:
        with suppress(Exception):
            _CLASS_ANALYSIS_PROCESS_EVALUATOR.clear_runtime_caches()
    _CLASS_ANALYSIS_PROCESS_PACKAGE = None
    _CLASS_ANALYSIS_PROCESS_EVALUATOR = None
    _CLASS_ANALYSIS_PROCESS_CLASSIFIER = None


def _init_class_analysis_process(
    dataset_root: str,
    program_context: Mapping[str, Any],
    version_key: Optional[str],
    source: str,
    allow_imports: bool,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    global _CLASS_ANALYSIS_PROCESS_PACKAGE
    global _CLASS_ANALYSIS_PROCESS_EVALUATOR
    global _CLASS_ANALYSIS_PROCESS_CLASSIFIER
    global _CLASS_ANALYSIS_PROCESS_SOURCE
    global _CLASS_ANALYSIS_PROCESS_WORD_ALIASES

    _close_class_analysis_process_state()
    evaluator = ProgramEvaluator(
        sandbox_config=SandboxConfig(
            allow_imports=bool(allow_imports),
        )
    )
    context, selected_source, _selected_version = _resolve_class_analysis_context(
        program_context=program_context,
        version_key=version_key,
        source=str(source),
    )
    classifier = TransitionGroupClassifier(evaluator)
    classifier.sync_context(
        context=context,
        max_program_count=max(1, len(context.get("versions") or [])),
    )
    _CLASS_ANALYSIS_PROCESS_PACKAGE = load_or_build_compiled_offline_eval_package(
        dataset_root=Path(str(dataset_root)),
    )
    _CLASS_ANALYSIS_PROCESS_EVALUATOR = evaluator
    _CLASS_ANALYSIS_PROCESS_CLASSIFIER = classifier
    _CLASS_ANALYSIS_PROCESS_SOURCE = selected_source
    _CLASS_ANALYSIS_PROCESS_WORD_ALIASES = _normalize_word_aliases(word_aliases)


def _evaluate_class_analysis_row(row_id: int) -> Dict[str, Any]:
    if (
        _CLASS_ANALYSIS_PROCESS_PACKAGE is None
        or _CLASS_ANALYSIS_PROCESS_EVALUATOR is None
        or _CLASS_ANALYSIS_PROCESS_CLASSIFIER is None
    ):
        raise RuntimeError("Class analysis worker process was not initialized.")
    transition = _build_class_analysis_transition(
        package=_CLASS_ANALYSIS_PROCESS_PACKAGE,
        row_id=int(row_id),
        word_aliases=_CLASS_ANALYSIS_PROCESS_WORD_ALIASES,
    )
    record = _CLASS_ANALYSIS_PROCESS_EVALUATOR.evaluate_transition(
        source=_CLASS_ANALYSIS_PROCESS_SOURCE,
        transition=transition,
    )
    current_explains = bool(record.is_correct) and record.error is None
    if not current_explains:
        return {
            "rowId": int(row_id),
            "explainable": False,
            "assigned": False,
            "repairClassId": None,
            "repairGroupId": None,
            "error": _serialize_error(record.error),
        }
    assignment, _rows = _CLASS_ANALYSIS_PROCESS_CLASSIFIER.classify_transition(
        transition=transition,
        include_rows=False,
    )
    repair_class_id = (
        int(assignment.class_id)
        if isinstance(getattr(assignment, "class_id", None), int)
        and int(assignment.class_id) > 0
        else None
    )
    repair_group_id = (
        str(assignment.group_id)
        if isinstance(getattr(assignment, "group_id", None), str)
        and str(assignment.group_id).strip()
        else None
    )
    return {
        "rowId": int(row_id),
        "explainable": True,
        "assigned": repair_class_id is not None,
        "repairClassId": repair_class_id,
        "repairGroupId": repair_group_id,
        "error": None,
    }


def _build_offline_eval_case_result(
    *,
    dataset: HeuristicEvalDataset,
    evaluator: ProgramEvaluator,
    source: str,
    class_idx: int,
) -> Dict[str, Any]:
    case = dataset.get_eval_case(int(class_idx))
    record = evaluator.evaluate_transition(
        source=str(source),
        transition=case.transition,
    )
    serialized_error = _serialize_error(record.error)
    status = _resolve_class_status(
        is_correct=bool(record.is_correct),
        error=serialized_error,
    )
    failure_payload = None
    if status != "pass":
        failure_payload = {
            "classIdx": int(case.class_idx),
            "action": str(case.action),
            "transitionCount": int(case.transition_count),
            "transitionIndex": int(case.transition_index),
            "scenarioType": str(case.scenario_type),
            "artifactStem": str(case.artifact_stem),
            "status": status,
            "expectedCanonical": str(record.expected_canonical),
            "predictedCanonical": (
                str(record.predicted_canonical)
                if isinstance(record.predicted_canonical, str)
                else None
            ),
            "error": serialized_error,
            "diffSummary": _build_diff_summary(
                str(record.expected_canonical),
                record.predicted_canonical,
            ),
            "availableImages": _resolve_available_failure_images(record.predicted_canonical),
        }
    return {
        "classPayload": {
            "classIdx": int(case.class_idx),
            "action": str(case.action),
            "transitionCount": int(case.transition_count),
            "transitionIndex": int(case.transition_index),
            "scenarioType": str(case.scenario_type),
            "artifactStem": str(case.artifact_stem),
            "status": status,
            "isCorrect": bool(record.is_correct),
            "failureId": None,
            "error": serialized_error,
        },
        "failurePayload": failure_payload,
    }


def _init_offline_eval_process(
    dataset_root: str,
    discovery_json: Optional[str],
    sample_seed: int,
    source: str,
    allow_imports: bool,
    scenario_split: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    global _OFFLINE_EVAL_PROCESS_DATASET
    global _OFFLINE_EVAL_PROCESS_EVALUATOR
    global _OFFLINE_EVAL_PROCESS_SOURCE

    if _OFFLINE_EVAL_PROCESS_DATASET is not None:
        _OFFLINE_EVAL_PROCESS_DATASET.close()
        _OFFLINE_EVAL_PROCESS_DATASET = None
    resolved_discovery_json = (
        Path(str(discovery_json)).resolve()
        if discovery_json not in {None, ""}
        else None
    )
    allowed_scenario_types = _resolve_offline_eval_allowed_scenario_types(
        scenario_split,
        required=_normalize_offline_eval_scenario_split(scenario_split) != "all",
    )
    dataset = HeuristicEvalDataset(
        dataset_root=Path(str(dataset_root)),
        discovery_json=resolved_discovery_json,
        sample_seed=int(sample_seed),
        word_aliases=_normalize_word_aliases(word_aliases),
        allowed_scenario_types=allowed_scenario_types,
        bundle_state_cache_size=DEFAULT_OFFLINE_EVAL_WORKER_BUNDLE_STATE_CACHE_SIZE,
        max_loaded_bundle_contexts=DEFAULT_OFFLINE_EVAL_WORKER_MAX_LOADED_BUNDLES,
    )
    dataset.load()
    _OFFLINE_EVAL_PROCESS_DATASET = dataset
    _OFFLINE_EVAL_PROCESS_EVALUATOR = ProgramEvaluator(
        sandbox_config=SandboxConfig(
            allow_imports=bool(allow_imports),
        )
    )
    _OFFLINE_EVAL_PROCESS_SOURCE = str(source)


def _evaluate_offline_eval_case(class_idx: int) -> Dict[str, Any]:
    if _OFFLINE_EVAL_PROCESS_DATASET is None or _OFFLINE_EVAL_PROCESS_EVALUATOR is None:
        raise RuntimeError("Offline eval worker process was not initialized.")
    try:
        return _build_offline_eval_case_result(
            dataset=_OFFLINE_EVAL_PROCESS_DATASET,
            evaluator=_OFFLINE_EVAL_PROCESS_EVALUATOR,
            source=_OFFLINE_EVAL_PROCESS_SOURCE,
            class_idx=int(class_idx),
        )
    finally:
        _OFFLINE_EVAL_PROCESS_EVALUATOR.clear_runtime_caches()


def _initial_run_summary(
    *,
    run_id: str,
    program: Mapping[str, Any],
    dataset_root: Path,
    discovery_json: Optional[Path],
    scenario_split: str,
    allowed_scenario_types: Optional[Sequence[str]],
    selection_mode: str,
    class_count: int,
    sample_seed: int,
    workers: int,
    run_dir: Path,
    run_output_dir: Optional[Path] = None,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    word_aliases_source_path: Optional[Path] = None,
    run_mode: str = OFFLINE_EVAL_RUN_MODE_ACCURACY,
) -> Dict[str, Any]:
    started_at = _utc_now_iso()
    normalized_word_aliases = _normalize_word_aliases(word_aliases)
    normalized_scenario_split = _normalize_offline_eval_scenario_split(scenario_split)
    normalized_run_mode = _normalize_offline_eval_run_mode(run_mode)
    return {
        "runId": str(run_id),
        "runMode": normalized_run_mode,
        "status": "queued",
        "startedAt": started_at,
        "finishedAt": None,
        "program": {
            "versionKey": str(program.get("versionKey", "")).strip() or None,
            "label": str(program.get("label", "")).strip() or None,
            "sourcePath": str(program.get("sourcePath", "")).strip() or None,
            "sourceDigest": str(program.get("sourceDigest", "")).strip() or _text_sha1(program.get("source")),
            "sourceLineCount": int(program.get("sourceLineCount", 0) or 0),
            "runOutputDir": str(run_output_dir) if isinstance(run_output_dir, Path) else None,
        },
        "stateAliases": {
            "enabled": bool(normalized_word_aliases),
            "sourcePath": (
                str(word_aliases_source_path)
                if isinstance(word_aliases_source_path, Path)
                else None
            ),
            "aliasCount": int(_word_alias_count(normalized_word_aliases)),
            "wordAliases": normalized_word_aliases,
        },
        "dataset": {
            "datasetRoot": str(dataset_root),
            "discoveryJson": (str(discovery_json) if discovery_json is not None else None),
            "scenarioSplit": normalized_scenario_split,
            "scenarioSplitLabel": _OFFLINE_EVAL_SCENARIO_SPLIT_LABELS.get(
                normalized_scenario_split,
                normalized_scenario_split,
            ),
            "scenarioCount": (
                len(allowed_scenario_types)
                if allowed_scenario_types is not None
                else None
            ),
            "classCount": int(class_count),
            "selectionMode": str(selection_mode),
        },
        "options": {
            "sampleSeed": int(sample_seed),
            "workers": _normalize_offline_eval_workers(workers),
        },
        "progress": {
            "completedCount": 0,
            "totalCount": int(class_count),
        },
        "metrics": {
            "accuracy": 0.0,
            "correctCount": 0,
            "failedCount": 0,
            "mismatchCount": 0,
            "runtimeErrorCount": 0,
            "compileErrorCount": 0,
        },
        "latestFailureId": None,
        "compileErrors": [],
        "message": "",
        "artifacts": {
            "runDir": str(run_dir),
            "runPath": str(run_dir / "run.json"),
            "classesPath": str(run_dir / "class_results.jsonl"),
            "failuresPath": str(run_dir / "failures.jsonl"),
            "exportsDir": str(run_dir / "exports"),
        },
    }


def _load_visual_config() -> Dict[str, Any]:
    bundle = load_baba_config_bundle(
        env_config_path=DEFAULT_ENV_CONFIG,
        experiment_config_path=DEFAULT_EXPERIMENT_CONFIG,
    )
    return build_visualization_config(bundle.get("word_aliases"))


def _write_failure_export(
    *,
    export_dir: Path,
    failure: Mapping[str, Any],
    images: Mapping[str, bytes],
) -> Dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    file_map: Dict[str, str] = {}
    file_names = {
        "previous": "previous_state.png",
        "expected": "expected_next_state.png",
        "predicted": "predicted_next_state.png",
        "comparison": "comparison.png",
    }
    for kind, png_bytes in images.items():
        file_path = export_dir / file_names.get(str(kind), f"{kind}.png")
        file_path.write_bytes(png_bytes)
        file_map[kind] = str(file_path)
    payload_path = export_dir / "failure.json"
    payload_path.write_text(
        json.dumps(dict(failure), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    file_map["failure"] = str(payload_path)
    return file_map


def _cleanup_offline_eval_runs(
    *,
    artifact_root: Path,
    retention_days: int,
    max_runs: int,
) -> None:
    artifact_root = Path(artifact_root).resolve()
    cutoff = _utc_now() - timedelta(days=max(1, int(retention_days)))
    retained_runs: List[tuple[datetime, Path]] = []
    try:
        run_dirs = _iter_directory_paths(artifact_root)
    except OSError:
        return
    for run_dir in run_dirs:
        run_path = run_dir / "run.json"
        if _path_is_missing(run_path):
            shutil.rmtree(run_dir, ignore_errors=True)
            continue
        try:
            payload = _read_json(run_path)
        except ValueError:
            shutil.rmtree(run_dir, ignore_errors=True)
            continue
        except OSError:
            continue
        status = str(payload.get("status", "")).strip()
        finished_at = str(payload.get("finishedAt", "")).strip()
        if status in {"queued", "running"}:
            payload["status"] = "failed"
            payload["finishedAt"] = payload.get("finishedAt") or _utc_now_iso()
            payload["message"] = "Worker was unavailable after a dashboard restart."
            try:
                _write_json_atomic(run_path, payload)
            except OSError:
                continue
            finished_at = str(payload.get("finishedAt", "")).strip()
        if not finished_at:
            continue
        with suppress(ValueError):
            finished_dt = datetime.fromisoformat(finished_at)
            if finished_dt.tzinfo is None:
                finished_dt = finished_dt.replace(tzinfo=timezone.utc)
            if finished_dt < cutoff:
                shutil.rmtree(run_dir, ignore_errors=True)
                continue
            retained_runs.append((finished_dt, run_dir))
    retained_runs.sort(key=lambda item: item[0], reverse=True)
    for _finished_dt, run_dir in retained_runs[max(1, int(max_runs)):]:
        shutil.rmtree(run_dir, ignore_errors=True)


def _run_class_purity_worker(
    *,
    run_id: str,
    run_dir: str,
    dataset_root: str,
    discovery_json: str,
    source: str,
    program: Mapping[str, Any],
    program_context: Mapping[str, Any],
    allow_imports: bool,
    sample_seed: int,
    scenario_split: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
    workers: int = DEFAULT_OFFLINE_EVAL_WORKERS,
    run_output_dir: Optional[str] = None,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    word_aliases_source_path: Optional[str] = None,
    retention_days: int = DEFAULT_OFFLINE_EVAL_RETENTION_DAYS,
    max_runs: int = DEFAULT_OFFLINE_EVAL_MAX_RUNS,
) -> None:
    del sample_seed
    run_dir_path = Path(run_dir)
    run_path = run_dir_path / "run.json"
    classes_path = run_dir_path / "class_results.jsonl"
    failures_path = run_dir_path / "failures.jsonl"
    resolved_dataset_root = Path(dataset_root).resolve()
    resolved_discovery_json = Path(discovery_json).resolve()
    normalized_word_aliases = _normalize_word_aliases(word_aliases)
    normalized_scenario_split = _normalize_offline_eval_scenario_split(scenario_split)
    allowed_scenario_types = _resolve_offline_eval_allowed_scenario_types(
        normalized_scenario_split,
        required=normalized_scenario_split != "all",
    )
    resolved_run_output_dir = (
        Path(run_output_dir).resolve()
        if isinstance(run_output_dir, str) and run_output_dir.strip()
        else None
    )
    resolved_word_aliases_source_path = (
        Path(word_aliases_source_path).resolve()
        if isinstance(word_aliases_source_path, str) and word_aliases_source_path.strip()
        else None
    )
    selected_version_key = str(program.get("versionKey") or "").strip() or None
    context, selected_source, resolved_version_key = _resolve_class_analysis_context(
        program_context=program_context,
        version_key=selected_version_key,
        source=str(source),
    )

    summary = _initial_run_summary(
        run_id=run_id,
        program=program,
        dataset_root=resolved_dataset_root,
        discovery_json=resolved_discovery_json,
        scenario_split=normalized_scenario_split,
        allowed_scenario_types=allowed_scenario_types,
        selection_mode=OFFLINE_EVAL_RUN_MODE_CLASS_PURITY,
        class_count=0,
        sample_seed=DEFAULT_OFFLINE_EVAL_SAMPLE_SEED,
        workers=workers,
        run_dir=run_dir_path,
        run_output_dir=resolved_run_output_dir,
        word_aliases=normalized_word_aliases,
        word_aliases_source_path=resolved_word_aliases_source_path,
        run_mode=OFFLINE_EVAL_RUN_MODE_CLASS_PURITY,
    )
    summary["program"]["versionKey"] = resolved_version_key
    summary["status"] = "running"
    summary["message"] = "Loading offline transitions for dynamics class analysis."
    _write_json_atomic(run_path, summary)

    package: Optional[CompiledOfflineEvalPackage] = None
    try:
        package = load_or_build_compiled_offline_eval_package(
            dataset_root=resolved_dataset_root,
        )
        class_metadata_by_id = _load_class_purity_metadata_by_id(resolved_discovery_json)
        stats_by_heuristic: Dict[int, Dict[str, Any]] = {}
        transition_total = 0
        for ref in _iter_class_purity_transition_refs(
            package=package,
            discovery_json=resolved_discovery_json,
            allowed_scenario_types=allowed_scenario_types,
            class_metadata_by_id=class_metadata_by_id,
        ):
            transition_total += 1
            stats = stats_by_heuristic.setdefault(
                int(ref.heuristic_class_idx),
                {
                    "transitionCount": 0,
                    "explainableCount": 0,
                    "assignedCount": 0,
                    "unexplainableCount": 0,
                    "unassignedCount": 0,
                    "repairCounts": {},
                    "repairRepresentatives": {},
                    "unexplainableRepresentative": None,
                    "unassignedRepresentative": None,
                    "sample": ref,
                },
            )
            stats["transitionCount"] = int(stats.get("transitionCount", 0)) + 1
        if transition_total <= 0:
            raise ValueError(
                "No heuristic-dynamics transitions matched the selected evaluation bundles. "
                f"Source JSON: {resolved_discovery_json}"
            )
        heuristic_class_ids = sorted(stats_by_heuristic.keys())
        heuristic_class_count = int(len(heuristic_class_ids))
        summary["dataset"]["classCount"] = heuristic_class_count
        summary["dataset"]["heuristicClassCount"] = heuristic_class_count
        summary["dataset"]["transitionCount"] = transition_total
        summary["progress"] = {
            "completedCount": 0,
            "totalCount": transition_total,
        }
        summary["message"] = f"0/{transition_total} transitions classified for H->C purity."
        _write_json_atomic(run_path, summary)

        completed_count = 0
        explainable_count = 0
        assigned_count = 0
        unexplainable_count = 0
        unassigned_count = 0
        repair_class_counts: Dict[int, int] = {}
        progress_interval = max(1, min(500, transition_total // 200 or 1))
        ref_chunk_size = max(
            DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE,
            DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE * max(1, int(_normalize_offline_eval_workers(workers))) * 4,
        )

        def _consume_result(ref: _ClassPurityTransitionRef, result: Mapping[str, Any]) -> None:
            nonlocal completed_count
            nonlocal explainable_count
            nonlocal assigned_count
            nonlocal unexplainable_count
            nonlocal unassigned_count

            completed_count += 1
            stats = stats_by_heuristic[int(ref.heuristic_class_idx)]
            if result.get("explainable") is not True:
                unexplainable_count += 1
                stats["unexplainableCount"] = int(stats.get("unexplainableCount", 0)) + 1
                if stats.get("unexplainableRepresentative") is None:
                    stats["unexplainableRepresentative"] = (ref, dict(result))
                return
            explainable_count += 1
            stats["explainableCount"] = int(stats.get("explainableCount", 0)) + 1
            repair_class_id = result.get("repairClassId")
            if not isinstance(repair_class_id, int) or int(repair_class_id) <= 0:
                unassigned_count += 1
                stats["unassignedCount"] = int(stats.get("unassignedCount", 0)) + 1
                if stats.get("unassignedRepresentative") is None:
                    stats["unassignedRepresentative"] = (ref, dict(result))
                return
            assigned_count += 1
            repair_class_id = int(repair_class_id)
            stats["assignedCount"] = int(stats.get("assignedCount", 0)) + 1
            repair_counts = stats.setdefault("repairCounts", {})
            repair_counts[repair_class_id] = int(repair_counts.get(repair_class_id, 0)) + 1
            repair_representatives = stats.setdefault("repairRepresentatives", {})
            repair_representatives.setdefault(repair_class_id, (ref, dict(result)))
            repair_class_counts[repair_class_id] = int(repair_class_counts.get(repair_class_id, 0)) + 1

        normalized_workers = _normalize_offline_eval_workers(workers)
        if transition_total <= 1 or normalized_workers <= 1:
            try:
                _init_class_analysis_process(
                    str(resolved_dataset_root),
                    context,
                    resolved_version_key,
                    selected_source,
                    bool(allow_imports),
                    normalized_word_aliases,
                )
                for ref in _iter_class_purity_transition_refs(
                    package=package,
                    discovery_json=resolved_discovery_json,
                    allowed_scenario_types=allowed_scenario_types,
                    class_metadata_by_id=class_metadata_by_id,
                ):
                    _consume_result(ref, _evaluate_class_analysis_row(int(ref.row_id)))
                    if completed_count % progress_interval == 0 or completed_count >= transition_total:
                        summary["progress"] = {
                            "completedCount": int(completed_count),
                            "totalCount": transition_total,
                        }
                        summary["message"] = (
                            f"{completed_count}/{transition_total} transitions classified for H->C purity."
                        )
                        _write_json_atomic(run_path, summary)
            finally:
                _close_class_analysis_process_state()
        else:
            with ProcessPoolExecutor(
                max_workers=min(normalized_workers, transition_total),
                initializer=_init_class_analysis_process,
                initargs=(
                    str(resolved_dataset_root),
                    context,
                    resolved_version_key,
                    selected_source,
                    bool(allow_imports),
                    normalized_word_aliases,
                ),
            ) as executor:
                for ref_chunk in _iter_fixed_size_chunks(
                    _iter_class_purity_transition_refs(
                        package=package,
                        discovery_json=resolved_discovery_json,
                        allowed_scenario_types=allowed_scenario_types,
                        class_metadata_by_id=class_metadata_by_id,
                    ),
                    ref_chunk_size,
                ):
                    row_ids = [int(ref.row_id) for ref in ref_chunk]
                    for ref, result in zip(
                        ref_chunk,
                        executor.map(
                            _evaluate_class_analysis_row,
                            row_ids,
                            chunksize=DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE,
                        ),
                    ):
                        _consume_result(ref, result)
                        if completed_count % progress_interval == 0 or completed_count >= transition_total:
                            summary["progress"] = {
                                "completedCount": int(completed_count),
                                "totalCount": transition_total,
                            }
                            summary["message"] = (
                                f"{completed_count}/{transition_total} transitions classified for H->C purity."
                            )
                            _write_json_atomic(run_path, summary)

        weighted_majority_sum = 0
        pure_class_count = 0
        split_class_count = 0
        unassigned_class_count = 0
        result_rows: List[Dict[str, Any]] = []
        for heuristic_class_idx in heuristic_class_ids:
            stats = stats_by_heuristic[int(heuristic_class_idx)]
            repair_counts_raw = stats.get("repairCounts") if isinstance(stats.get("repairCounts"), Mapping) else {}
            repair_counts = {
                int(class_id): int(count)
                for class_id, count in repair_counts_raw.items()
                if int(class_id) > 0 and int(count) > 0
            }
            assigned_for_class = int(stats.get("assignedCount", 0) or 0)
            majority_repair_class_id = None
            majority_count = 0
            if repair_counts:
                majority_repair_class_id, majority_count = max(
                    repair_counts.items(),
                    key=lambda item: (int(item[1]), -int(item[0])),
                )
                weighted_majority_sum += int(majority_count)
            purity = (majority_count / assigned_for_class) if assigned_for_class > 0 else 0.0
            if assigned_for_class <= 0:
                status = "runtime_error"
                unassigned_class_count += 1
            elif purity >= 1.0:
                status = "pass"
                pure_class_count += 1
            else:
                status = "mismatch"
                split_class_count += 1
            metadata = class_metadata_by_id.get(int(heuristic_class_idx), {})
            sample_ref = stats.get("sample")
            if not isinstance(sample_ref, _ClassPurityTransitionRef):
                continue
            representative_transitions: List[Dict[str, Any]] = []
            repair_representatives_raw = (
                stats.get("repairRepresentatives")
                if isinstance(stats.get("repairRepresentatives"), Mapping)
                else {}
            )
            representative_limit = max(1, int(DEFAULT_OFFLINE_EVAL_CLASS_REPRESENTATIVE_LIMIT))
            total_representative_candidate_count = 0
            for class_id, _count in sorted(
                repair_counts.items(),
                key=lambda item: (-int(item[1]), int(item[0])),
            ):
                pair = repair_representatives_raw.get(int(class_id))
                if not (
                    isinstance(pair, tuple)
                    and len(pair) == 2
                    and isinstance(pair[0], _ClassPurityTransitionRef)
                    and isinstance(pair[1], Mapping)
                ):
                    continue
                total_representative_candidate_count += 1
                if len(representative_transitions) >= representative_limit:
                    continue
                role = "majority" if int(class_id) == int(majority_repair_class_id or 0) else "minority"
                representative_transitions.append(
                    _build_class_purity_representative_payload(
                        heuristic_class_idx=int(heuristic_class_idx),
                        ref=pair[0],
                        result=pair[1],
                        role=role,
                    )
                )
            for role, key in (
                ("unassigned", "unassignedRepresentative"),
                ("unexplainable", "unexplainableRepresentative"),
            ):
                pair = stats.get(key)
                if not (
                    isinstance(pair, tuple)
                    and len(pair) == 2
                    and isinstance(pair[0], _ClassPurityTransitionRef)
                    and isinstance(pair[1], Mapping)
                ):
                    continue
                total_representative_candidate_count += 1
                if len(representative_transitions) >= representative_limit:
                    continue
                representative_transitions.append(
                    _build_class_purity_representative_payload(
                        heuristic_class_idx=int(heuristic_class_idx),
                        ref=pair[0],
                        result=pair[1],
                        role=role,
                    )
                )
            result_rows.append(
                {
                    "classIdx": int(heuristic_class_idx),
                    "action": str(metadata.get("action") or sample_ref.action or "").strip(),
                    "transitionCount": int(stats.get("transitionCount", 0) or 0),
                    "transitionIndex": int(sample_ref.transition_index),
                    "scenarioType": str(sample_ref.scenario_type),
                    "artifactStem": str(sample_ref.artifact_stem),
                    "status": status,
                    "isCorrect": status == "pass",
                    "failureId": None,
                    "error": None,
                    "heuristicClassIdx": int(heuristic_class_idx),
                    "majorityRepairClassId": (
                        int(majority_repair_class_id)
                        if isinstance(majority_repair_class_id, int)
                        else None
                    ),
                    "majorityRepairTransitionCount": int(majority_count),
                    "purity": float(purity),
                    "explainableTransitionCount": int(stats.get("explainableCount", 0) or 0),
                    "assignedTransitionCount": int(assigned_for_class),
                    "unexplainableTransitionCount": int(stats.get("unexplainableCount", 0) or 0),
                    "unassignedTransitionCount": int(stats.get("unassignedCount", 0) or 0),
                    "representativeTransitions": representative_transitions,
                    "representativeTransitionCount": int(len(representative_transitions)),
                    "hasMoreRepresentativeTransitions": bool(
                        total_representative_candidate_count > len(representative_transitions)
                    ),
                    "repairClassDistribution": [
                        {
                            "repairClassId": int(class_id),
                            "transitionCount": int(count),
                        }
                        for class_id, count in sorted(
                            repair_counts.items(),
                            key=lambda item: (-int(item[1]), int(item[0])),
                        )
                    ],
                }
            )

        if classes_path.exists():
            classes_path.unlink()
        for row in sorted(result_rows, key=lambda item: (float(item.get("purity", 0.0)), int(item.get("classIdx", 0)))):
            _append_jsonl(classes_path, row)
        if failures_path.exists():
            failures_path.unlink()

        purity_h_to_r = (weighted_majority_sum / assigned_count) if assigned_count > 0 else 0.0
        assignment_coverage = (assigned_count / explainable_count) if explainable_count > 0 else 0.0
        explainable_coverage = (explainable_count / transition_total) if transition_total > 0 else 0.0
        summary["status"] = "completed"
        summary["finishedAt"] = _utc_now_iso()
        summary["progress"] = {
            "completedCount": transition_total,
            "totalCount": transition_total,
        }
        summary["metrics"] = {
            "accuracy": float(purity_h_to_r),
            "correctCount": int(pure_class_count),
            "failedCount": int(split_class_count + unassigned_class_count),
            "mismatchCount": int(split_class_count),
            "runtimeErrorCount": int(unassigned_class_count),
            "compileErrorCount": 0,
            "resultRowCount": int(len(result_rows)),
            "purityHToR": float(purity_h_to_r),
            "assignmentCoverage": float(assignment_coverage),
            "explainableCoverage": float(explainable_coverage),
            "transitionCount": int(transition_total),
            "heuristicClassCount": int(heuristic_class_count),
            "repairClassCount": int(len(repair_class_counts)),
            "explainableTransitionCount": int(explainable_count),
            "assignedTransitionCount": int(assigned_count),
            "unexplainableTransitionCount": int(unexplainable_count),
            "unassignedTransitionCount": int(unassigned_count),
            "pureHeuristicClassCount": int(pure_class_count),
            "splitHeuristicClassCount": int(split_class_count),
            "unassignedHeuristicClassCount": int(unassigned_class_count),
        }
        summary["message"] = (
            f"H->C purity {(purity_h_to_r * 100):.2f}% over "
            f"{assigned_count}/{transition_total} assigned transitions."
        )
        _write_json_atomic(run_path, summary)
    except Exception as error:  # noqa: BLE001
        summary["status"] = "failed"
        summary["finishedAt"] = _utc_now_iso()
        summary["message"] = str(error)
        _write_json_atomic(run_path, summary)
        raise
    finally:
        if package is not None:
            with suppress(Exception):
                package.close()
        _cleanup_offline_eval_runs(
            artifact_root=run_dir_path.parent,
            retention_days=int(retention_days),
            max_runs=int(max_runs),
        )


def _run_eval_worker(
    *,
    run_id: str,
    run_dir: str,
    dataset_root: str,
    discovery_json: Optional[str],
    source: str,
    program: Mapping[str, Any],
    allow_imports: bool,
    sample_seed: int,
    scenario_split: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
    workers: int = DEFAULT_OFFLINE_EVAL_WORKERS,
    run_output_dir: Optional[str] = None,
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    word_aliases_source_path: Optional[str] = None,
    run_mode: str = OFFLINE_EVAL_RUN_MODE_ACCURACY,
    program_context: Optional[Mapping[str, Any]] = None,
    retention_days: int,
    max_runs: int,
) -> None:
    normalized_run_mode = _normalize_offline_eval_run_mode(run_mode)
    if normalized_run_mode == OFFLINE_EVAL_RUN_MODE_CLASS_PURITY:
        if discovery_json in {None, ""}:
            raise ValueError("Class analysis requires a heuristic dynamics discovery JSON.")
        if not isinstance(program_context, Mapping):
            raise ValueError("Class analysis requires a stored discovery program context.")
        _run_class_purity_worker(
            run_id=run_id,
            run_dir=run_dir,
            dataset_root=dataset_root,
            discovery_json=str(discovery_json),
            source=source,
            program=program,
            program_context=program_context,
            allow_imports=allow_imports,
            sample_seed=sample_seed,
            scenario_split=scenario_split,
            workers=workers,
            run_output_dir=run_output_dir,
            word_aliases=word_aliases,
            word_aliases_source_path=word_aliases_source_path,
            retention_days=retention_days,
            max_runs=max_runs,
        )
        return

    run_dir_path = Path(run_dir)
    run_path = run_dir_path / "run.json"
    classes_path = run_dir_path / "class_results.jsonl"
    failures_path = run_dir_path / "failures.jsonl"
    resolved_discovery_json = (
        Path(discovery_json).resolve()
        if discovery_json is not None and str(discovery_json).strip()
        else None
    )
    normalized_word_aliases = _normalize_word_aliases(word_aliases)
    resolved_run_output_dir = (
        Path(run_output_dir).resolve()
        if isinstance(run_output_dir, str) and run_output_dir.strip()
        else None
    )
    resolved_word_aliases_source_path = (
        Path(word_aliases_source_path).resolve()
        if isinstance(word_aliases_source_path, str) and word_aliases_source_path.strip()
        else None
    )
    normalized_scenario_split = _normalize_offline_eval_scenario_split(scenario_split)
    allowed_scenario_types = _resolve_offline_eval_allowed_scenario_types(
        normalized_scenario_split,
        required=normalized_scenario_split != "all",
    )
    dataset = HeuristicEvalDataset(
        dataset_root=Path(dataset_root),
        discovery_json=resolved_discovery_json,
        sample_seed=int(sample_seed),
        word_aliases=normalized_word_aliases,
        allowed_scenario_types=allowed_scenario_types,
        bundle_state_cache_size=DEFAULT_OFFLINE_EVAL_WORKER_BUNDLE_STATE_CACHE_SIZE,
        max_loaded_bundle_contexts=DEFAULT_OFFLINE_EVAL_WORKER_MAX_LOADED_BUNDLES,
    )
    dataset.load()
    normalized_workers = _normalize_offline_eval_workers(workers)
    summary = _initial_run_summary(
        run_id=run_id,
        program=program,
        dataset_root=Path(dataset_root),
        discovery_json=resolved_discovery_json,
        scenario_split=normalized_scenario_split,
        allowed_scenario_types=allowed_scenario_types,
        selection_mode=dataset.selection_mode,
        class_count=dataset.class_count,
        sample_seed=int(sample_seed),
        workers=normalized_workers,
        run_dir=run_dir_path,
        run_output_dir=resolved_run_output_dir,
        word_aliases=normalized_word_aliases,
        word_aliases_source_path=resolved_word_aliases_source_path,
    )
    summary["status"] = "running"
    _write_json_atomic(run_path, summary)
    correct_count = 0
    mismatch_count = 0
    runtime_error_count = 0
    compile_error_count = 0
    failed_count = 0
    compile_errors: List[Dict[str, Optional[str]]] = []
    failure_counter = 0
    completed_count = 0
    class_count = int(dataset.class_count)
    progress_write_interval = _offline_eval_progress_write_count_interval(class_count)
    progress_write_state = _OfflineEvalProgressWriteState(
        last_completed_count=0,
        last_write_monotonic=time.monotonic(),
    )
    class_chunk_size = max(
        DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE,
        DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE * max(1, int(normalized_workers)) * 4,
    )

    def _consume_case_result(case_result: Mapping[str, Any]) -> None:
        nonlocal correct_count
        nonlocal mismatch_count
        nonlocal runtime_error_count
        nonlocal compile_error_count
        nonlocal failed_count
        nonlocal failure_counter
        nonlocal completed_count

        total_count = int(class_count)
        class_payload = dict(case_result["classPayload"])
        serialized_error = class_payload.get("error")
        status = str(class_payload["status"])
        failure_payload = case_result.get("failurePayload")
        failure_id = None

        if status == "pass":
            correct_count += 1
        else:
            failed_count += 1
            if status == "mismatch":
                mismatch_count += 1
            elif status == "runtime_error":
                runtime_error_count += 1
            elif status == "compile_error":
                compile_error_count += 1
                if isinstance(serialized_error, Mapping):
                    compile_errors.append(dict(serialized_error))
            if failure_payload is not None:
                failure_counter += 1
                failure_id = f"fail_{failure_counter:04d}"
                failure_row = dict(failure_payload)
                failure_row["failureId"] = failure_id
                _append_jsonl(failures_path, failure_row)
                summary["latestFailureId"] = failure_id

        class_payload["failureId"] = failure_id
        _append_jsonl(classes_path, class_payload)

        completed_count += 1
        summary["progress"] = {
            "completedCount": int(completed_count),
            "totalCount": total_count,
        }
        summary["metrics"] = {
            "accuracy": (correct_count / total_count) if total_count > 0 else 0.0,
            "correctCount": int(correct_count),
            "failedCount": int(failed_count),
            "mismatchCount": int(mismatch_count),
            "runtimeErrorCount": int(runtime_error_count),
            "compileErrorCount": int(compile_error_count),
        }
        summary["compileErrors"] = compile_errors[:16]
        summary["message"] = (
            f"{completed_count}/{total_count} {dataset.result_label_plural} evaluated"
            if total_count > 0
            else f"No {dataset.result_label_plural} to evaluate"
        )
        _write_progress_summary_if_due(
            run_path=run_path,
            summary=summary,
            state=progress_write_state,
            completed_count=completed_count,
            total_count=total_count,
            count_interval=progress_write_interval,
        )

    try:
        if class_count <= 1 or normalized_workers <= 1:
            evaluator = ProgramEvaluator(
                sandbox_config=SandboxConfig(
                    allow_imports=bool(allow_imports),
                )
            )
            try:
                for class_idx in dataset.iter_eval_class_indices():
                    _consume_case_result(
                        _build_offline_eval_case_result(
                            dataset=dataset,
                            evaluator=evaluator,
                            source=str(source),
                            class_idx=int(class_idx),
                        )
                    )
                    evaluator.clear_runtime_caches()
            finally:
                evaluator.clear_runtime_caches()
        else:
            with ProcessPoolExecutor(
                max_workers=min(normalized_workers, class_count),
                initializer=_init_offline_eval_process,
                initargs=(
                    str(dataset_root),
                    (str(resolved_discovery_json) if resolved_discovery_json is not None else None),
                    int(sample_seed),
                    str(source),
                    bool(allow_imports),
                    normalized_scenario_split,
                    normalized_word_aliases,
                ),
            ) as executor:
                for class_chunk in _iter_fixed_size_chunks(
                    dataset.iter_eval_class_indices(),
                    class_chunk_size,
                ):
                    for case_result in executor.map(
                        _evaluate_offline_eval_case,
                        class_chunk,
                        chunksize=DEFAULT_OFFLINE_EVAL_PROCESS_POOL_CHUNKSIZE,
                    ):
                        _consume_case_result(case_result)

        final_status = "completed"
        if compile_error_count > 0 and failed_count == class_count and correct_count == 0:
            final_status = "compile_failed"
        summary["status"] = final_status
        summary["finishedAt"] = _utc_now_iso()
        summary["message"] = f"Completed {class_count} {dataset.result_label_plural}."
        _write_json_atomic(run_path, summary)
    except Exception as error:  # noqa: BLE001
        summary["status"] = "failed"
        summary["finishedAt"] = _utc_now_iso()
        summary["message"] = str(error)
        _write_json_atomic(run_path, summary)
        raise
    finally:
        _cleanup_offline_eval_runs(
            artifact_root=run_dir_path.parent,
            retention_days=retention_days,
            max_runs=max_runs,
        )
        dataset.close()


class OfflineEvalService:
    def __init__(
        self,
        *,
        artifact_root: Path = DEFAULT_OFFLINE_EVAL_ARTIFACT_ROOT,
        default_dataset_root: Path = DEFAULT_OFFLINE_EVAL_DATASET_ROOT,
        default_discovery_json: Path = DEFAULT_OFFLINE_EVAL_DISCOVERY_JSON,
        retention_days: int = DEFAULT_OFFLINE_EVAL_RETENTION_DAYS,
        max_runs: int = DEFAULT_OFFLINE_EVAL_MAX_RUNS,
        visual_config: Optional[Mapping[str, Any]] = None,
        catalog_root: Optional[Path] = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.default_dataset_root = Path(default_dataset_root).resolve()
        self.default_discovery_json = Path(default_discovery_json).resolve()
        self.catalog_root = (
            Path(catalog_root).resolve()
            if catalog_root is not None
            else self.default_dataset_root.parent.resolve()
        )
        self.retention_days = max(1, int(retention_days))
        self.max_runs = max(1, int(max_runs))
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._dataset_cache: Dict[tuple[str, str, str, int, str], HeuristicEvalDataset] = {}
        self._dataset_summary_cache: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        self._last_export_destination: Optional[Path] = None
        self._export_status_by_run: Dict[str, Dict[str, Any]] = {}
        self._export_threads: Dict[str, threading.Thread] = {}
        self._export_cancel_events: Dict[str, threading.Event] = {}
        self._export_lock = threading.Lock()
        self._visual_config = (
            dict(visual_config)
            if isinstance(visual_config, Mapping)
            else _load_visual_config()
        )
        self._renderer = TransitionArtifactRenderer(visual_config=self._visual_config)
        self._cleanup_existing_runs()

    def _normalize_discovery_json_path(
        self,
        discovery_json: Optional[str | Path],
    ) -> Optional[Path]:
        if discovery_json is None:
            return None
        discovery_text = str(discovery_json).strip()
        if not discovery_text:
            return None
        return Path(discovery_text).resolve()

    def _resolve_default_discovery_json(
        self,
        dataset_root: Optional[Path] = None,
    ) -> Optional[Path]:
        candidate = self._normalize_discovery_json_path(self.default_discovery_json)
        if candidate is None or not self._looks_like_discovery_json(candidate):
            return None
        if dataset_root is not None and candidate.parent != Path(dataset_root).resolve():
            return None
        return candidate

    def _display_path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(resolved)

    def _looks_like_dataset_root(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "batch_summary.json").exists():
            return True
        return any(path.glob("*_transitions.jsonl"))

    def _looks_like_discovery_json(self, path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() != ".json":
            return False
        try:
            payload = _read_json(path)
        except Exception:
            return False
        return isinstance(payload.get("transition_class_mapping"), list)

    def _scan_catalog_dataset_roots(self) -> List[Path]:
        roots: List[Path] = []
        if self._looks_like_dataset_root(self.default_dataset_root):
            roots.append(self.default_dataset_root)
        if self.catalog_root.exists():
            for child in sorted(self.catalog_root.iterdir()):
                if self._looks_like_dataset_root(child):
                    roots.append(child.resolve())
        deduped: List[Path] = []
        seen: set[str] = set()
        for path in roots:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path.resolve())
        return deduped

    def _scan_catalog_discovery_jsons(self, dataset_root: Path) -> List[Path]:
        roots = [dataset_root.resolve()]
        paths: List[Path] = []
        for root in roots:
            for candidate in sorted(root.glob("*.json")):
                if self._looks_like_discovery_json(candidate):
                    paths.append(candidate.resolve())
        default_path = self._resolve_default_discovery_json(dataset_root)
        if default_path is not None and default_path not in paths:
            paths.insert(0, default_path)
        deduped: List[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path.resolve())
        return deduped

    def get_catalog(self) -> Dict[str, Any]:
        dataset_rows: List[Dict[str, Any]] = []
        default_dataset_root = str(self.default_dataset_root)
        default_discovery_json_path = self._resolve_default_discovery_json()
        default_discovery_json = (
            str(default_discovery_json_path)
            if default_discovery_json_path is not None
            else ""
        )
        scenario_split_options = _offline_eval_scenario_split_options()
        scenario_split_values = [
            str(option.get("value") or "").strip()
            for option in scenario_split_options
            if str(option.get("value") or "").strip()
        ]
        for dataset_root in self._scan_catalog_dataset_roots():
            discovery_rows: List[Dict[str, Any]] = []
            full_dataset_count_by_split: Dict[str, int] = {}
            for scenario_split in scenario_split_values:
                with suppress(Exception):
                    full_dataset_count_by_split[scenario_split] = int(
                        self._get_dataset_summary(
                            dataset_root=dataset_root,
                            discovery_json=None,
                            scenario_split=scenario_split,
                        )["classCount"]
                    )
            full_dataset_count: Optional[int] = full_dataset_count_by_split.get("all")
            for discovery_json in self._scan_catalog_discovery_jsons(dataset_root):
                class_count_by_split: Dict[str, int] = {}
                for scenario_split in scenario_split_values:
                    with suppress(Exception):
                        class_count_by_split[scenario_split] = int(
                            self._get_dataset_summary(
                                dataset_root=dataset_root,
                                discovery_json=discovery_json,
                                scenario_split=scenario_split,
                            )["classCount"]
                        )
                class_count: Optional[int] = class_count_by_split.get("all")
                discovery_rows.append(
                    {
                        "path": str(discovery_json),
                        "label": discovery_json.name,
                        "displayPath": self._display_path(discovery_json),
                        "classCount": class_count,
                        "classCountBySplit": class_count_by_split,
                        "isDefault": str(discovery_json) == default_discovery_json,
                    }
                )
            dataset_rows.append(
                {
                    "path": str(dataset_root),
                    "label": dataset_root.name,
                    "displayPath": self._display_path(dataset_root),
                    "isDefault": str(dataset_root) == default_dataset_root,
                    "fullDatasetCount": full_dataset_count,
                    "fullDatasetCountBySplit": full_dataset_count_by_split,
                    "discoveryJsons": discovery_rows,
                }
            )
        return {
            "datasetRoots": dataset_rows,
            "defaultDatasetRoot": default_dataset_root,
            "defaultDiscoveryJson": default_discovery_json,
            "defaultScenarioSplit": DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
            "scenarioSplits": scenario_split_options,
            "defaultSampleSeed": int(DEFAULT_OFFLINE_EVAL_SAMPLE_SEED),
        }

    def close(self) -> None:
        for cancel_event in list(self._export_cancel_events.values()):
            cancel_event.set()
        for run_id, process in list(self._processes.items()):
            if process.poll() is not None:
                with suppress(Exception):
                    process.wait(timeout=0.1)
                self._processes.pop(run_id, None)
                continue
            with suppress(Exception):
                process.terminate()
            with suppress(Exception):
                process.wait(timeout=0.5)
        self._processes.clear()
        for run_id, thread in list(self._export_threads.items()):
            if not thread.is_alive():
                self._export_threads.pop(run_id, None)
                continue
            with suppress(Exception):
                thread.join(timeout=0.1)
        for dataset in list(self._dataset_cache.values()):
            with suppress(Exception):
                dataset.close()
        self._dataset_cache.clear()
        self._export_threads.clear()
        self._export_cancel_events.clear()

    def _cleanup_existing_runs(self) -> None:
        _cleanup_offline_eval_runs(
            artifact_root=self.artifact_root,
            retention_days=self.retention_days,
            max_runs=self.max_runs,
        )

    def _run_dir(self, run_id: str) -> Path:
        return self.artifact_root / str(run_id)

    def _run_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _classes_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "class_results.jsonl"

    def _failures_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "failures.jsonl"

    def _job_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "job.json"

    def _exports_root(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "exports"

    def _normalize_export_directory_hint(self, path_hint: Optional[str | Path]) -> Optional[Path]:
        if path_hint is None:
            return None
        source_path_text = str(path_hint).strip()
        if not source_path_text:
            return None
        source_path = Path(source_path_text).expanduser().resolve()
        candidate = source_path if source_path.is_dir() else source_path.parent
        if candidate.name == "program_versions" and candidate.parent.exists():
            candidate = candidate.parent
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
        return None

    def _resolve_program_export_directory(
        self,
        summary: Optional[Mapping[str, Any]],
        *,
        preferred_dir: Optional[str | Path] = None,
    ) -> Optional[Path]:
        preferred = self._normalize_export_directory_hint(preferred_dir)
        if preferred is not None:
            return preferred
        if not isinstance(summary, Mapping):
            return None
        program = summary.get("program")
        if not isinstance(program, Mapping):
            return None
        return self._normalize_export_directory_hint(program.get("sourcePath"))

    def _default_export_destination(
        self,
        *,
        summary: Optional[Mapping[str, Any]] = None,
        preferred_dir: Optional[str | Path] = None,
    ) -> Path:
        program_dir = self._resolve_program_export_directory(summary, preferred_dir=preferred_dir)
        if program_dir is not None:
            return program_dir
        if self._last_export_destination is not None and self._last_export_destination.is_dir():
            return self._last_export_destination
        desktop = Path.home() / "Desktop"
        if desktop.is_dir():
            return desktop.resolve()
        home = Path.home()
        if home.is_dir():
            return home.resolve()
        return self.artifact_root

    def pick_export_directory(
        self,
        *,
        run_id: Optional[str] = None,
        preferred_dir: Optional[str | Path] = None,
    ) -> Optional[Dict[str, str]]:
        summary = self.get_run(run_id) if run_id else None
        picked = _pick_directory_via_windows_dialog(
            initial_dir=self._default_export_destination(summary=summary, preferred_dir=preferred_dir),
        )
        if picked is None:
            return None
        self._last_export_destination = picked
        return {
            "directory": str(picked),
            "displayPath": self._display_path(picked),
        }

    def _sync_process_state(self, run_id: str) -> None:
        process = self._processes.get(str(run_id))
        if process is None:
            return
        exitcode = process.poll()
        if exitcode is None:
            return
        with suppress(Exception):
            process.wait(timeout=0.1)
        self._processes.pop(str(run_id), None)
        run_path = self._run_path(str(run_id))
        try:
            summary = _read_json(run_path)
        except FileNotFoundError:
            return
        except OSError:
            return
        if str(summary.get("status", "")).strip() not in {"queued", "running"}:
            return
        summary["status"] = "failed"
        summary["finishedAt"] = _utc_now_iso()
        summary["message"] = (
            f"Worker exited unexpectedly (exitcode={exitcode})."
            if exitcode is not None
            else "Worker exited unexpectedly."
        )
        _write_json_atomic(run_path, summary)

    def _get_dataset(
        self,
        *,
        dataset_root: str | Path,
        discovery_json: str | Path | None,
        sample_seed: int,
        scenario_split: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
        word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> HeuristicEvalDataset:
        resolved_discovery_json = self._normalize_discovery_json_path(discovery_json)
        normalized_word_aliases = _normalize_word_aliases(word_aliases)
        normalized_scenario_split = _normalize_offline_eval_scenario_split(scenario_split)
        allowed_scenario_types = _resolve_offline_eval_allowed_scenario_types(
            normalized_scenario_split,
            required=normalized_scenario_split != "all",
        )
        key = (
            str(Path(dataset_root).resolve()),
            (str(resolved_discovery_json) if resolved_discovery_json is not None else ""),
            normalized_scenario_split,
            int(sample_seed),
            _word_aliases_signature(normalized_word_aliases),
        )
        cached = self._dataset_cache.get(key)
        if cached is not None:
            return cached
        dataset = HeuristicEvalDataset(
            dataset_root=Path(dataset_root),
            discovery_json=resolved_discovery_json,
            sample_seed=int(sample_seed),
            word_aliases=normalized_word_aliases,
            allowed_scenario_types=allowed_scenario_types,
        )
        dataset.load()
        self._dataset_cache[key] = dataset
        return dataset

    def _get_dataset_summary(
        self,
        *,
        dataset_root: str | Path,
        discovery_json: str | Path | None,
        scenario_split: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
    ) -> Dict[str, Any]:
        resolved_dataset_root = Path(dataset_root).resolve()
        resolved_discovery_json = self._normalize_discovery_json_path(discovery_json)
        normalized_scenario_split = _normalize_offline_eval_scenario_split(scenario_split)
        allowed_scenario_types = _resolve_offline_eval_allowed_scenario_types(
            normalized_scenario_split,
            required=normalized_scenario_split != "all",
        )
        key = (
            str(resolved_dataset_root),
            (str(resolved_discovery_json) if resolved_discovery_json is not None else ""),
            normalized_scenario_split,
        )
        cached = self._dataset_summary_cache.get(key)
        if cached is not None:
            return dict(cached)

        if resolved_discovery_json is not None:
            summary = {
                "selectionMode": "heuristic_class",
                "classCount": int(
                    _count_discovery_json_classes(
                        resolved_discovery_json,
                        dataset_root=resolved_dataset_root,
                        allowed_scenario_types=allowed_scenario_types,
                    )
                ),
            }
        else:
            summary = {
                "selectionMode": "full_dataset",
                "classCount": int(
                    _count_full_dataset_transitions(
                        resolved_dataset_root,
                        allowed_scenario_types=allowed_scenario_types,
                    )
                ),
            }
        self._dataset_summary_cache[key] = dict(summary)
        return dict(summary)

    def get_defaults(self) -> Dict[str, Any]:
        default_discovery_json = self._resolve_default_discovery_json(self.default_dataset_root)
        dataset_summary = self._get_dataset_summary(
            dataset_root=self.default_dataset_root,
            discovery_json=default_discovery_json,
            scenario_split=DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
        )
        catalog = self.get_catalog()
        return {
            "datasetRoot": str(self.default_dataset_root),
            "discoveryJson": (str(default_discovery_json) if default_discovery_json is not None else ""),
            "scenarioSplit": DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
            "sampleSeed": int(DEFAULT_OFFLINE_EVAL_SAMPLE_SEED),
            "workers": int(DEFAULT_OFFLINE_EVAL_WORKERS),
            "classCount": int(dataset_summary["classCount"]),
            "selectionMode": str(dataset_summary["selectionMode"]),
            "catalog": catalog,
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        try:
            run_dirs = _iter_directory_paths(self.artifact_root, reverse=True)
        except OSError:
            return runs
        for run_dir in run_dirs:
            run_path = run_dir / "run.json"
            try:
                run_id = run_dir.name
                self._sync_process_state(run_id)
                runs.append(_read_json(run_path))
            except FileNotFoundError:
                continue
            except ValueError:
                continue
            except OSError:
                continue
        runs.sort(
            key=lambda row: str(row.get("startedAt", "")),
            reverse=True,
        )
        return runs

    def get_run(self, run_id: str) -> Dict[str, Any]:
        run_key = str(run_id).strip()
        if not run_key:
            raise HTTPException(status_code=404, detail="Unknown offline eval run.")
        self._sync_process_state(run_key)
        run_path = self._run_path(run_key)
        try:
            return _read_json(run_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown offline eval run.")
        except OSError as error:
            raise HTTPException(
                status_code=503,
                detail="Offline eval run metadata is temporarily unavailable.",
            ) from error

    def get_class_results(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = DEFAULT_OFFLINE_EVAL_RESULT_PAGE_SIZE,
        status_filter: str = "all",
        sort: str = "desc",
    ) -> Dict[str, Any]:
        summary = self.get_run(run_id)
        normalized_offset = _normalize_offline_eval_result_offset(offset)
        normalized_limit = _normalize_offline_eval_result_page_limit(limit)
        normalized_filter = _normalize_offline_eval_status_filter(status_filter)
        normalized_sort = _normalize_offline_eval_sort(sort)
        page_temporarily_unavailable = False
        page_message: Optional[str] = None
        try:
            rows = _read_jsonl_page(
                self._classes_path(str(run_id)),
                offset=normalized_offset,
                limit=normalized_limit,
                status_filter=normalized_filter,
                sort=normalized_sort,
            )
        except OSError as error:
            page_temporarily_unavailable = True
            page_message = (
                "Eval result rows are temporarily unavailable while the worker is writing files. "
                "Progress updates will continue."
            )
            rows = []
        total_count = _offline_eval_filtered_total_count(summary, status_filter=normalized_filter)
        if not page_temporarily_unavailable:
            total_count = max(total_count, normalized_offset + len(rows))
        return {
            "run": summary,
            "classes": rows,
            "page": {
                "offset": normalized_offset,
                "limit": normalized_limit,
                "returnedCount": int(len(rows)),
                "totalCount": int(total_count),
                "hasMore": (
                    False
                    if page_temporarily_unavailable
                    else bool(normalized_offset + len(rows) < total_count)
                ),
                "statusFilter": normalized_filter,
                "sort": normalized_sort,
                "temporarilyUnavailable": page_temporarily_unavailable,
                "message": page_message,
            },
        }

    def get_failures(self, run_id: str) -> Dict[str, Any]:
        summary = self.get_run(run_id)
        rows = _read_jsonl(self._failures_path(str(run_id)))
        return {
            "run": summary,
            "failures": rows,
        }

    def _find_failure_row(self, run_id: str, failure_id: str) -> Dict[str, Any]:
        run_key = str(run_id).strip()
        failure_key = str(failure_id).strip()
        if not failure_key:
            raise HTTPException(status_code=404, detail="Unknown offline eval failure.")
        for row in _read_jsonl(self._failures_path(run_key)):
            if str(row.get("failureId", "")).strip() == failure_key:
                return dict(row)
        raise HTTPException(status_code=404, detail="Unknown offline eval failure.")

    def _find_class_representative_row(
        self,
        run_id: str,
        representative_id: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        run_key = str(run_id).strip()
        representative_key = str(representative_id).strip()
        if not representative_key:
            raise HTTPException(status_code=404, detail="Unknown class representative transition.")
        for row in _read_jsonl(self._classes_path(run_key)):
            representatives = row.get("representativeTransitions")
            if not isinstance(representatives, list):
                continue
            for representative in representatives:
                if not isinstance(representative, Mapping):
                    continue
                if str(representative.get("representativeId", "")).strip() == representative_key:
                    return dict(row), dict(representative)
        raise HTTPException(status_code=404, detail="Unknown class representative transition.")

    def _class_analysis_source_for_job(self, job: Mapping[str, Any]) -> str:
        source = str(job.get("source") or "").strip()
        program = job.get("program") if isinstance(job.get("program"), Mapping) else {}
        program_context = (
            job.get("program_context")
            if isinstance(job.get("program_context"), Mapping)
            else None
        )
        if isinstance(program_context, Mapping):
            with suppress(ValueError):
                _context, selected_source, _version_key = _resolve_class_analysis_context(
                    program_context=program_context,
                    version_key=str(program.get("versionKey") or "").strip() or None,
                    source=source,
                )
                if selected_source.strip():
                    return selected_source
        return source

    def _build_image_meta_payload(
        self,
        *,
        snapshot_meta: Mapping[str, Any],
        fallback_detail: Optional[str] = None,
    ) -> Dict[str, Any]:
        context_line = str(snapshot_meta.get("contextLine") or "").strip()
        detail_line = str(snapshot_meta.get("detailLine") or "").strip()
        if fallback_detail:
            fallback_text = str(fallback_detail).strip()
            if fallback_text:
                detail_line = f"{detail_line} | {fallback_text}" if detail_line else fallback_text
        return {
            "mapName": snapshot_meta.get("mapName"),
            "action": snapshot_meta.get("action"),
            "terminated": snapshot_meta.get("terminated"),
            "done": snapshot_meta.get("done"),
            "lines": [line for line in (context_line, detail_line) if line],
        }

    def _word_aliases_for_summary(self, summary: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
        state_aliases = summary.get("stateAliases") if isinstance(summary, Mapping) else None
        raw_aliases = (
            state_aliases.get("wordAliases")
            if isinstance(state_aliases, Mapping)
            else None
        )
        return _normalize_word_aliases(raw_aliases)

    def _visual_config_for_summary(self, summary: Mapping[str, Any]) -> Dict[str, Any]:
        word_aliases = self._word_aliases_for_summary(summary)
        if word_aliases:
            return build_visualization_config(word_aliases)
        return dict(self._visual_config)

    def _augment_class_representative_state_for_render(
        self,
        *,
        state_obj: Optional[Mapping[str, Any]],
        meta: Any,
    ) -> Dict[str, Any]:
        payload = dict(state_obj) if isinstance(state_obj, Mapping) else {}
        step_payload = payload.get("step")
        resolved_step = dict(step_payload) if isinstance(step_payload, Mapping) else {}
        if "terminated" not in resolved_step:
            resolved_step["terminated"] = bool(getattr(meta, "done", False))
        payload["step"] = resolved_step

        bundle = getattr(meta, "bundle", None)
        map_text = str(payload.get("map_name", "") or payload.get("mapName", "") or "").strip()
        if not map_text and bundle is not None:
            map_text = str(getattr(bundle, "scenario_type", "") or "").strip()
        if map_text:
            payload["map_name"] = map_text
            payload["scenario_type"] = map_text

        artifact_text = str(getattr(bundle, "artifact_stem", "") if bundle is not None else "").strip()
        if artifact_text:
            payload["artifact_stem"] = artifact_text

        return payload

    def _build_class_representative_render_states(
        self,
        *,
        package: CompiledOfflineEvalPackage,
        row_id: int,
        word_aliases: Mapping[str, Mapping[str, str]],
        predicted_state_obj: Optional[Dict[str, Any]],
    ) -> tuple[Any, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        meta = package.transition_meta(int(row_id))
        previous_state_obj = _state_obj_with_word_aliases(
            package=package,
            state_id=int(meta.state_id),
            word_aliases=word_aliases,
        )
        next_state_obj = _state_obj_with_word_aliases(
            package=package,
            state_id=int(meta.next_state_id),
            word_aliases=word_aliases,
        )
        previous_state = self._augment_class_representative_state_for_render(
            state_obj=previous_state_obj,
            meta=meta,
        )
        expected_state = self._augment_class_representative_state_for_render(
            state_obj=next_state_obj,
            meta=meta,
        )
        predicted_state = None
        if isinstance(predicted_state_obj, dict):
            predicted_state = self._augment_class_representative_state_for_render(
                state_obj=predicted_state_obj,
                meta=meta,
            )
        return meta, previous_state, expected_state, predicted_state

    def _build_class_representative_image_meta(
        self,
        *,
        representative: Mapping[str, Any],
        meta: Any,
        previous_state: Dict[str, Any],
        expected_state: Dict[str, Any],
        predicted_state: Optional[Dict[str, Any]],
        predicted_state_obj: Optional[Dict[str, Any]],
        error: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        action = str(getattr(meta, "action", "") or representative.get("action") or "").strip()
        previous_meta = self._renderer.describe_state_snapshot(
            state=previous_state,
            action_name=action,
            include_world_in_context=False,
        )
        expected_meta = self._renderer.describe_state_snapshot(
            state=expected_state,
            action_name=action,
            include_world_in_context=False,
        )
        predicted_meta_state = predicted_state
        if not isinstance(predicted_meta_state, dict):
            predicted_meta_state = self._augment_class_representative_state_for_render(
                state_obj={},
                meta=meta,
            )
        prediction_error = _format_prediction_error_message(error)
        difference_summary = _build_comparison_difference_summary(
            expected_state=expected_state,
            predicted_state=predicted_state,
            prediction_error=prediction_error,
        )
        predicted_meta = self._renderer.describe_state_snapshot(
            state=predicted_meta_state,
            action_name=action,
            include_world_in_context=False,
        )
        comparison_lines = []
        if previous_meta.get("contextLine"):
            comparison_lines.append(str(previous_meta["contextLine"]))
        if previous_meta.get("action"):
            comparison_lines.append(f"ACTION {previous_meta['action']}")
        comparison_lines.extend(
            [
                "Top: Previous State",
                "Bottom: Expected Next State vs Predicted Next State",
            ]
        )
        if difference_summary:
            comparison_lines.append(difference_summary)
        if prediction_error and not isinstance(predicted_state_obj, dict):
            comparison_lines.append(prediction_error)
        return {
            "previous": self._build_image_meta_payload(snapshot_meta=previous_meta),
            "expected": self._build_image_meta_payload(snapshot_meta=expected_meta),
            "predicted": self._build_image_meta_payload(
                snapshot_meta=predicted_meta,
                fallback_detail=prediction_error if not isinstance(predicted_state_obj, dict) else None,
            ),
            "comparison": {
                "lines": [line for line in comparison_lines if str(line).strip()],
            },
        }

    def _augment_failure_state_for_render(
        self,
        *,
        state_obj: Optional[Mapping[str, Any]],
        state_archive: Optional[Mapping[str, Any]],
        case: Any,
    ) -> Dict[str, Any]:
        payload = dict(state_obj) if isinstance(state_obj, Mapping) else {}
        step_payload = payload.get("step")
        resolved_step = dict(step_payload) if isinstance(step_payload, Mapping) else {}
        if "terminated" not in resolved_step and isinstance(state_archive, Mapping) and "terminated" in state_archive:
            resolved_step["terminated"] = bool(state_archive.get("terminated"))
        payload["step"] = resolved_step

        if isinstance(state_archive, Mapping):
            source_text = str(state_archive.get("source", "") or "").strip()
            if source_text and not str(payload.get("source", "") or "").strip():
                payload["source"] = source_text

        map_text = str(payload.get("map_name", "") or payload.get("mapName", "") or "").strip()
        if not map_text:
            map_text = str(getattr(case, "scenario_type", "") or "").strip()
        if not map_text:
            map_text = str(getattr(case, "artifact_stem", "") or "").strip()
        if map_text:
            payload["map_name"] = map_text
            payload["scenario_type"] = map_text

        artifact_text = str(getattr(case, "artifact_stem", "") or "").strip()
        if artifact_text:
            payload["artifact_stem"] = artifact_text

        return payload

    def _build_failure_render_states(
        self,
        *,
        case: Any,
        predicted_state_obj: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        previous_state = self._augment_failure_state_for_render(
            state_obj=case.previous_state_obj,
            state_archive=case.previous_state_archive,
            case=case,
        )
        expected_state = self._augment_failure_state_for_render(
            state_obj=case.next_state_obj,
            state_archive=case.next_state_archive,
            case=case,
        )
        predicted_state = None
        if isinstance(predicted_state_obj, dict):
            predicted_state = self._augment_failure_state_for_render(
                state_obj=predicted_state_obj,
                state_archive=case.next_state_archive,
                case=case,
            )
        return previous_state, expected_state, predicted_state

    def _build_failure_image_meta(
        self,
        *,
        case: Any,
        predicted_state_obj: Optional[Dict[str, Any]],
        failure: Mapping[str, Any],
    ) -> Dict[str, Any]:
        previous_state, expected_state, predicted_state = self._build_failure_render_states(
            case=case,
            predicted_state_obj=predicted_state_obj,
        )
        previous_meta = self._renderer.describe_state_snapshot(
            state=previous_state,
            action_name=str(case.action),
            include_world_in_context=False,
        )
        expected_meta = self._renderer.describe_state_snapshot(
            state=expected_state,
            action_name=str(case.action),
            include_world_in_context=False,
        )
        prediction_error = _format_prediction_error_message(failure.get("error"))
        predicted_meta_state = predicted_state
        if not isinstance(predicted_meta_state, dict):
            predicted_meta_state = self._augment_failure_state_for_render(
                state_obj={},
                state_archive=case.next_state_archive,
                case=case,
            )
        difference_summary = _build_comparison_difference_summary(
            expected_state=expected_state,
            predicted_state=predicted_state,
            prediction_error=prediction_error,
        )
        predicted_meta = self._renderer.describe_state_snapshot(
            state=predicted_meta_state,
            action_name=str(case.action),
            include_world_in_context=False,
        )
        comparison_lines = []
        if previous_meta.get("contextLine"):
            comparison_lines.append(str(previous_meta["contextLine"]))
        if previous_meta.get("action"):
            comparison_lines.append(f"ACTION {previous_meta['action']}")
        comparison_lines.extend(
            [
                "Top: Previous State",
                "Bottom: Expected Next State vs Predicted Next State",
            ]
        )
        if difference_summary:
            comparison_lines.append(difference_summary)
        if prediction_error and not isinstance(predicted_state_obj, dict):
            comparison_lines.append(prediction_error)
        return {
            "previous": self._build_image_meta_payload(snapshot_meta=previous_meta),
            "expected": self._build_image_meta_payload(snapshot_meta=expected_meta),
            "predicted": self._build_image_meta_payload(
                snapshot_meta=predicted_meta,
                fallback_detail=prediction_error if not isinstance(predicted_state_obj, dict) else None,
            ),
            "comparison": {
                "lines": [line for line in comparison_lines if str(line).strip()],
            },
        }

    def _augment_failure_detail(
        self,
        *,
        summary: Mapping[str, Any],
        failure: Mapping[str, Any],
    ) -> Dict[str, Any]:
        detail = dict(failure)
        dataset = self._get_dataset(
            dataset_root=summary["dataset"]["datasetRoot"],
            discovery_json=summary["dataset"]["discoveryJson"],
            sample_seed=int(summary["options"]["sampleSeed"]),
            scenario_split=str(summary["dataset"].get("scenarioSplit") or DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT),
            word_aliases=self._word_aliases_for_summary(summary),
        )
        case = dataset.get_case(int(detail["classIdx"]))
        predicted_canonical = detail.get("predictedCanonical")
        predicted_state_obj = None
        if isinstance(predicted_canonical, str) and predicted_canonical.strip():
            with suppress(ValueError):
                predicted_state_obj = parse_state_json(predicted_canonical)
        detail["imageMeta"] = self._build_failure_image_meta(
            case=case,
            predicted_state_obj=predicted_state_obj,
            failure=detail,
        )
        return detail

    def _build_class_representative_payload(
        self,
        run_id: str,
        representative_id: str,
    ) -> Dict[str, Any]:
        summary = self.get_run(run_id)
        if not _is_class_purity_summary(summary):
            raise HTTPException(status_code=409, detail="Run is not a dynamics class analysis run.")
        class_row, representative = self._find_class_representative_row(run_id, representative_id)
        try:
            job = _read_json(self._job_path(str(run_id)))
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=409,
                detail="Class representative rendering requires the eval job payload. Re-run CLASS analysis.",
            ) from error
        except OSError as error:
            raise HTTPException(
                status_code=503,
                detail="Eval job payload is temporarily unavailable.",
            ) from error

        source = self._class_analysis_source_for_job(job)
        if not source.strip():
            raise HTTPException(status_code=409, detail="Class representative rendering requires program source.")

        dataset_root = Path(str(summary["dataset"]["datasetRoot"]))
        word_aliases = self._word_aliases_for_summary(summary)
        try:
            row_id = int(representative.get("rowId"))
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=409,
                detail="Class representative row id is missing. Re-run CLASS analysis.",
            ) from error
        package: Optional[CompiledOfflineEvalPackage] = None
        evaluator = ProgramEvaluator(
            sandbox_config=SandboxConfig(
                allow_imports=bool(job.get("allow_imports", True)),
            )
        )
        try:
            package = load_or_build_compiled_offline_eval_package(dataset_root=dataset_root)
            transition = _build_class_analysis_transition(
                package=package,
                row_id=row_id,
                word_aliases=word_aliases,
            )
            record = evaluator.evaluate_transition(
                source=source,
                transition=transition,
            )
            serialized_error = _serialize_error(record.error)
            predicted_canonical = (
                str(record.predicted_canonical)
                if isinstance(record.predicted_canonical, str)
                else None
            )
            predicted_state_obj = None
            if isinstance(predicted_canonical, str) and predicted_canonical.strip():
                with suppress(ValueError):
                    predicted_state_obj = parse_state_json(predicted_canonical)
            meta, previous_state, expected_state, predicted_state = self._build_class_representative_render_states(
                package=package,
                row_id=row_id,
                word_aliases=word_aliases,
                predicted_state_obj=predicted_state_obj,
            )
            available_images = _resolve_available_failure_images(predicted_canonical)
            detail = {
                **dict(representative),
                "classIdx": int(class_row.get("classIdx", representative.get("heuristicClassIdx", 0)) or 0),
                "heuristicClassIdx": int(
                    class_row.get("heuristicClassIdx", representative.get("heuristicClassIdx", 0)) or 0
                ),
                "classStatus": class_row.get("status"),
                "purity": class_row.get("purity"),
                "majorityRepairClassId": class_row.get("majorityRepairClassId"),
                "status": _resolve_class_status(
                    is_correct=bool(record.is_correct),
                    error=serialized_error,
                ),
                "isCorrect": bool(record.is_correct),
                "expectedCanonical": str(record.expected_canonical),
                "predictedCanonical": predicted_canonical,
                "error": serialized_error,
                "diffSummary": _build_diff_summary(str(record.expected_canonical), predicted_canonical),
                "availableImages": available_images,
                "imageMeta": self._build_class_representative_image_meta(
                    representative=representative,
                    meta=meta,
                    previous_state=previous_state,
                    expected_state=expected_state,
                    predicted_state=predicted_state,
                    predicted_state_obj=predicted_state_obj,
                    error=serialized_error,
                ),
            }
            return {
                "summary": summary,
                "classRow": class_row,
                "representative": dict(representative),
                "meta": meta,
                "previousState": previous_state,
                "expectedState": expected_state,
                "predictedState": predicted_state,
                "detail": detail,
            }
        finally:
            evaluator.clear_runtime_caches()
            if package is not None:
                with suppress(Exception):
                    package.close()

    def get_class_representative(self, run_id: str, representative_id: str) -> Dict[str, Any]:
        return dict(self._build_class_representative_payload(run_id, representative_id)["detail"])

    def get_failure(self, run_id: str, failure_id: str) -> Dict[str, Any]:
        run_key = str(run_id).strip()
        summary = self.get_run(run_key)
        row = self._find_failure_row(run_key, failure_id)
        return self._augment_failure_detail(summary=summary, failure=row)

    def create_run(
        self,
        *,
        source: str,
        program: Mapping[str, Any],
        dataset_root: str | Path,
        discovery_json: str | Path | None,
        allow_imports: bool,
        sample_seed: int,
        workers: int = DEFAULT_OFFLINE_EVAL_WORKERS,
        run_output_dir: str | Path | None = None,
        scenario_split: str = DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT,
        run_mode: str = OFFLINE_EVAL_RUN_MODE_ACCURACY,
        program_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        safe_source = str(source or "")
        if not safe_source.strip():
            raise HTTPException(status_code=400, detail="Offline eval requires non-empty Python source.")
        normalized_run_mode = _normalize_offline_eval_run_mode(run_mode)
        self._cleanup_existing_runs()
        resolved_dataset_root = Path(dataset_root).resolve()
        resolved_discovery_json = self._normalize_discovery_json_path(discovery_json)
        if _path_is_missing(resolved_dataset_root):
            raise HTTPException(status_code=400, detail=f"Dataset root not found: {resolved_dataset_root}")
        try:
            if not resolved_dataset_root.is_dir():
                raise HTTPException(status_code=400, detail=f"Dataset root not found: {resolved_dataset_root}")
        except OSError as error:
            raise HTTPException(status_code=503, detail="Dataset root is temporarily unavailable.") from error
        if resolved_discovery_json is not None and _path_is_missing(resolved_discovery_json):
            raise HTTPException(status_code=400, detail=f"Discovery JSON not found: {resolved_discovery_json}")
        if resolved_discovery_json is not None:
            try:
                if not resolved_discovery_json.is_file():
                    raise HTTPException(status_code=400, detail=f"Discovery JSON not found: {resolved_discovery_json}")
            except OSError as error:
                raise HTTPException(status_code=503, detail="Discovery JSON is temporarily unavailable.") from error
        raw_run_output_dir = run_output_dir
        if raw_run_output_dir in {None, ""}:
            raw_run_output_dir = program.get("runOutputDir") if isinstance(program, Mapping) else None
        resolved_run_output_dir = (
            Path(raw_run_output_dir).expanduser().resolve()
            if isinstance(raw_run_output_dir, (str, Path)) and str(raw_run_output_dir).strip()
            else None
        )
        resolved_program_context: Optional[Mapping[str, Any]] = (
            program_context if isinstance(program_context, Mapping) else None
        )
        if normalized_run_mode == OFFLINE_EVAL_RUN_MODE_CLASS_PURITY:
            if resolved_discovery_json is None:
                raise HTTPException(
                    status_code=400,
                    detail="Dynamics class analysis requires a heuristic dynamics discovery JSON.",
                )
            if not isinstance(resolved_program_context, Mapping):
                resolved_program_context = _load_legacy_program_context_from_run_output_dir(
                    resolved_run_output_dir,
                )
            if not isinstance(resolved_program_context, Mapping):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Discovery run does not expose repair group context and legacy artifacts "
                        "could not be recovered. Restart the discovery run so programContext is "
                        "written to dashboard.json."
                    ),
                )
            try:
                _resolve_class_analysis_context(
                    program_context=resolved_program_context,
                    version_key=str(program.get("versionKey") or "").strip() or None,
                    source=safe_source,
                )
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
        word_aliases, word_aliases_source_path = _load_word_aliases_from_run_output_dir(
            resolved_run_output_dir,
        )
        normalized_scenario_split = _normalize_offline_eval_scenario_split(scenario_split)
        try:
            allowed_scenario_types = _resolve_offline_eval_allowed_scenario_types(
                normalized_scenario_split,
                required=normalized_scenario_split != "all",
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if normalized_run_mode == OFFLINE_EVAL_RUN_MODE_CLASS_PURITY:
            dataset_summary = {
                "selectionMode": OFFLINE_EVAL_RUN_MODE_CLASS_PURITY,
                "classCount": 0,
            }
        else:
            dataset_summary = self._get_dataset_summary(
                dataset_root=resolved_dataset_root,
                discovery_json=resolved_discovery_json,
                scenario_split=normalized_scenario_split,
            )
        selection_mode = str(dataset_summary["selectionMode"])
        class_count = int(dataset_summary["classCount"])
        if normalized_run_mode == OFFLINE_EVAL_RUN_MODE_CLASS_PURITY:
            selection_mode = OFFLINE_EVAL_RUN_MODE_CLASS_PURITY
        run_id = uuid.uuid4().hex[:12]
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        summary = _initial_run_summary(
            run_id=run_id,
            program=program,
            dataset_root=resolved_dataset_root,
            discovery_json=resolved_discovery_json,
            scenario_split=normalized_scenario_split,
            allowed_scenario_types=allowed_scenario_types,
            selection_mode=selection_mode,
            class_count=class_count,
            sample_seed=int(sample_seed),
            workers=int(workers),
            run_dir=run_dir,
            run_output_dir=resolved_run_output_dir,
            word_aliases=word_aliases,
            word_aliases_source_path=word_aliases_source_path,
            run_mode=normalized_run_mode,
        )
        _write_json_atomic(run_dir / "run.json", summary)
        job_path = run_dir / "job.json"
        _write_json_atomic(
            job_path,
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "dataset_root": str(resolved_dataset_root),
                "discovery_json": (str(resolved_discovery_json) if resolved_discovery_json is not None else None),
                "scenario_split": normalized_scenario_split,
                "source": safe_source,
                "program": dict(program),
                "run_output_dir": (str(resolved_run_output_dir) if resolved_run_output_dir is not None else None),
                "word_aliases": word_aliases,
                "word_aliases_source_path": (
                    str(word_aliases_source_path)
                    if word_aliases_source_path is not None
                    else None
                ),
                "run_mode": normalized_run_mode,
                "program_context": (
                    dict(resolved_program_context)
                    if isinstance(resolved_program_context, Mapping)
                    else None
                ),
                "allow_imports": bool(allow_imports),
                "sample_seed": int(sample_seed),
                "workers": _normalize_offline_eval_workers(workers),
                "retention_days": int(self.retention_days),
                "max_runs": int(self.max_runs),
            },
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.web.offline_eval_worker",
                "--job-file",
                str(job_path),
            ],
            cwd=str(PROJECT_ROOT),
            **_offline_eval_worker_popen_kwargs(),
        )
        self._processes[run_id] = process
        summary["status"] = "running"
        summary["message"] = "Worker started."
        _write_json_atomic(run_dir / "run.json", summary)
        return summary

    def _image_to_png_bytes(self, image) -> bytes:
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        finally:
            image.close()

    def _placeholder_png_bytes(self, *, title: str, message: str) -> bytes:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return TransitionArtifactRenderer._PLACEHOLDER_PNG_BYTES
        image = Image.new("RGB", (520, 220), "#111111")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((16, 18), str(title), fill="#F3EFE6", font=font)
        draw.text((16, 58), str(message), fill="#D8D8D8", font=font)
        return self._image_to_png_bytes(image)

    def _render_failure_images(
        self,
        run_id: str,
        failure_id: str,
        *,
        export_mode: bool = False,
        include_unavailable: bool = True,
    ) -> Dict[str, bytes]:
        summary = self.get_run(run_id)
        failure = self._find_failure_row(run_id, failure_id)
        dataset = self._get_dataset(
            dataset_root=summary["dataset"]["datasetRoot"],
            discovery_json=summary["dataset"]["discoveryJson"],
            sample_seed=int(summary["options"]["sampleSeed"]),
            scenario_split=str(summary["dataset"].get("scenarioSplit") or DEFAULT_OFFLINE_EVAL_SCENARIO_SPLIT),
            word_aliases=self._word_aliases_for_summary(summary),
        )
        visual_config = self._visual_config_for_summary(summary)
        case = dataset.get_case(int(failure["classIdx"]))
        predicted_canonical = failure.get("predictedCanonical")
        available_images = _normalize_available_failure_images(
            failure.get("availableImages"),
            predicted_canonical if isinstance(predicted_canonical, str) else None,
        )
        predicted_state_obj = None
        if isinstance(predicted_canonical, str) and predicted_canonical.strip():
            with suppress(ValueError):
                predicted_state_obj = parse_state_json(predicted_canonical)
        previous_state, expected_state, predicted_state = self._build_failure_render_states(
            case=case,
            predicted_state_obj=predicted_state_obj,
        )
        prediction_error = _format_prediction_error_message(failure.get("error"))
        difference_summary = _build_comparison_difference_summary(
            expected_state=expected_state,
            predicted_state=predicted_state,
            prediction_error=prediction_error,
        )

        title_prefix = (
            f"C{int(case.class_idx):03d}"
            if export_mode
            else None
        )
        previous_title = (
            f"{title_prefix} Previous State"
            if title_prefix
            else "Previous State"
        )
        expected_title = (
            f"{title_prefix} Expected Next State"
            if title_prefix
            else "Expected Next State"
        )
        predicted_title = (
            f"{title_prefix} Predicted Next State"
            if title_prefix
            else "Predicted Next State"
        )

        previous_image = self._renderer.render_state_snapshot_image(
            state=previous_state,
            title=previous_title,
            action_name=str(case.action),
            include_world_in_context=False,
            visual_config=visual_config,
        )
        expected_image = self._renderer.render_state_snapshot_image(
            state=expected_state,
            title=expected_title,
            action_name=str(case.action),
            include_world_in_context=False,
            visual_config=visual_config,
        )
        predicted_image = None
        if predicted_state is not None:
            predicted_image = self._renderer.render_state_snapshot_image(
                state=predicted_state,
                title=predicted_title,
                action_name=str(case.action),
                include_world_in_context=False,
                visual_config=visual_config,
            )

        comparison_image = None
        if bool(available_images.get("comparison")) or include_unavailable:
            comparison_image = self._renderer.compose_comparison_image(
                previous_state=previous_state,
                actual_next_state=expected_state,
                predicted_next_state=predicted_state,
                action_name=str(case.action),
                experiment_name=str(summary["program"].get("label") or summary["program"].get("sourcePath") or "offline_eval"),
                version_tag=str(summary["program"].get("versionKey") or "selected"),
                prediction_error=prediction_error,
                step_index=int(case.transition_index),
                difference_summary=difference_summary,
                status_lines=[
                    f"class={int(case.class_idx)}",
                    f"scenario={case.scenario_type}",
                    f"artifact={case.artifact_stem}",
                ],
                include_world_in_context=False,
                visual_config=visual_config,
            )

        images: Dict[str, bytes] = {}
        if previous_image is not None:
            images["previous"] = self._image_to_png_bytes(previous_image)
        elif include_unavailable:
            images["previous"] = self._placeholder_png_bytes(title=previous_title, message="image unavailable")

        if expected_image is not None:
            images["expected"] = self._image_to_png_bytes(expected_image)
        elif include_unavailable:
            images["expected"] = self._placeholder_png_bytes(title=expected_title, message="image unavailable")

        if predicted_image is not None:
            images["predicted"] = self._image_to_png_bytes(predicted_image)
        elif include_unavailable:
            images["predicted"] = self._placeholder_png_bytes(
                title=predicted_title,
                message=_format_prediction_error_message(failure.get("error")) or "prediction unavailable",
            )

        if comparison_image is not None:
            images["comparison"] = self._image_to_png_bytes(comparison_image)
        return images

    def _render_class_representative_images(
        self,
        run_id: str,
        representative_id: str,
        *,
        include_unavailable: bool = True,
    ) -> Dict[str, bytes]:
        payload = self._build_class_representative_payload(run_id, representative_id)
        summary = payload["summary"]
        representative = payload["representative"]
        meta = payload["meta"]
        previous_state = payload["previousState"]
        expected_state = payload["expectedState"]
        predicted_state = payload["predictedState"]
        detail = payload["detail"]
        visual_config = self._visual_config_for_summary(summary)
        available_images = _normalize_available_failure_images(
            detail.get("availableImages"),
            detail.get("predictedCanonical") if isinstance(detail.get("predictedCanonical"), str) else None,
        )
        action = str(getattr(meta, "action", "") or representative.get("action") or "").strip()
        heuristic_class_idx = int(representative.get("heuristicClassIdx", detail.get("heuristicClassIdx", 0)) or 0)
        repair_class_id = representative.get("repairClassId")
        title_prefix = f"H{heuristic_class_idx:03d}"
        if isinstance(repair_class_id, int) and int(repair_class_id) > 0:
            title_prefix = f"{title_prefix} C{int(repair_class_id)}"
        previous_title = f"{title_prefix} Previous State"
        expected_title = f"{title_prefix} Expected Next State"
        predicted_title = f"{title_prefix} Predicted Next State"
        prediction_error = _format_prediction_error_message(detail.get("error"))
        difference_summary = _build_comparison_difference_summary(
            expected_state=expected_state,
            predicted_state=predicted_state,
            prediction_error=prediction_error,
        )

        previous_image = self._renderer.render_state_snapshot_image(
            state=previous_state,
            title=previous_title,
            action_name=action,
            include_world_in_context=False,
            visual_config=visual_config,
        )
        expected_image = self._renderer.render_state_snapshot_image(
            state=expected_state,
            title=expected_title,
            action_name=action,
            include_world_in_context=False,
            visual_config=visual_config,
        )
        predicted_image = None
        if predicted_state is not None:
            predicted_image = self._renderer.render_state_snapshot_image(
                state=predicted_state,
                title=predicted_title,
                action_name=action,
                include_world_in_context=False,
                visual_config=visual_config,
            )

        comparison_image = None
        if bool(available_images.get("comparison")) or include_unavailable:
            comparison_image = self._renderer.compose_comparison_image(
                previous_state=previous_state,
                actual_next_state=expected_state,
                predicted_next_state=predicted_state,
                action_name=action,
                experiment_name=str(summary["program"].get("label") or summary["program"].get("sourcePath") or "class_analysis"),
                version_tag=str(summary["program"].get("versionKey") or "selected"),
                prediction_error=prediction_error,
                step_index=int(representative.get("transitionIndex", 0) or 0),
                difference_summary=difference_summary,
                status_lines=[
                    f"heuristic_class={heuristic_class_idx}",
                    (
                        f"dynamics_class={int(repair_class_id)}"
                        if isinstance(repair_class_id, int) and int(repair_class_id) > 0
                        else "dynamics_class=unassigned"
                    ),
                    f"scenario={representative.get('scenarioType') or ''}",
                    f"artifact={representative.get('artifactStem') or ''}",
                ],
                include_world_in_context=False,
                visual_config=visual_config,
            )

        images: Dict[str, bytes] = {}
        if previous_image is not None:
            images["previous"] = self._image_to_png_bytes(previous_image)
        elif include_unavailable:
            images["previous"] = self._placeholder_png_bytes(title=previous_title, message="image unavailable")

        if expected_image is not None:
            images["expected"] = self._image_to_png_bytes(expected_image)
        elif include_unavailable:
            images["expected"] = self._placeholder_png_bytes(title=expected_title, message="image unavailable")

        if predicted_image is not None:
            images["predicted"] = self._image_to_png_bytes(predicted_image)
        elif include_unavailable:
            images["predicted"] = self._placeholder_png_bytes(
                title=predicted_title,
                message=prediction_error or "prediction unavailable",
            )

        if comparison_image is not None:
            images["comparison"] = self._image_to_png_bytes(comparison_image)
        return images

    def render_failure_kind(self, run_id: str, failure_id: str, kind: str) -> bytes:
        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in {"previous", "expected", "predicted", "comparison"}:
            raise HTTPException(status_code=400, detail=f"Unknown render kind: {kind}")
        images = self._render_failure_images(run_id, failure_id)
        return images[normalized_kind]

    def render_class_representative_kind(self, run_id: str, representative_id: str, kind: str) -> bytes:
        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in {"previous", "expected", "predicted", "comparison"}:
            raise HTTPException(status_code=400, detail=f"Unknown render kind: {kind}")
        images = self._render_class_representative_images(run_id, representative_id)
        return images[normalized_kind]

    def _prepare_export_request(
        self,
        *,
        run_id: str,
        scope: str,
        failure_id: Optional[str] = None,
        destination_dir: Optional[str | Path] = None,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], Path, Optional[Path]]:
        summary = self.get_run(run_id)
        if scope == "current_failure" and not failure_id:
            raise HTTPException(status_code=400, detail="failureId is required for current_failure export.")
        if scope not in {"current_failure", "all_failures"}:
            raise HTTPException(status_code=400, detail=f"Unknown export scope: {scope}")
        export_stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        requested_destination: Optional[Path] = None
        if destination_dir is not None:
            requested_destination = Path(destination_dir).expanduser().resolve()
            requested_destination.mkdir(parents=True, exist_ok=True)
            if not requested_destination.is_dir():
                raise HTTPException(status_code=400, detail="destinationDir must be a directory.")
            self._last_export_destination = requested_destination
        failures = (
            [self.get_failure(run_id, failure_id)]
            if scope == "current_failure"
            else self.get_failures(run_id)["failures"]
        )
        if requested_destination is None:
            export_root = self._exports_root(str(run_id)) / f"export_{export_stamp}"
        else:
            version_tag = str(summary.get("program", {}).get("versionKey") or "selected").strip() or "selected"
            export_root = requested_destination / f"offline_eval_{version_tag}_{export_stamp}"
        return summary, failures, export_root, requested_destination

    def _execute_export(
        self,
        *,
        run_id: str,
        scope: str,
        summary: Dict[str, Any],
        failures: List[Dict[str, Any]],
        export_root: Path,
        requested_destination: Optional[Path],
        progress_callback: Optional[Callable[..., None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        exported_cases: List[Dict[str, Any]] = []
        total_count = int(len(failures))
        if progress_callback is not None:
            progress_callback(
                completed_count=0,
                total_count=total_count,
                current_failure_id=None,
                message=f"Saving 0/{total_count} failure image sets...",
            )
        if cancel_event is not None and cancel_event.is_set():
            raise OfflineEvalExportCancelled("Export cancelled before any images were written.")
        for index, failure in enumerate(failures, start=1):
            current_failure_id = str(failure["failureId"])
            if cancel_event is not None and cancel_event.is_set():
                raise OfflineEvalExportCancelled(
                    f"Export cancelled after {len(exported_cases)}/{total_count} failure image set(s).",
                )
            images = self._render_failure_images(
                str(run_id),
                current_failure_id,
                export_mode=True,
                include_unavailable=False,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise OfflineEvalExportCancelled(
                    f"Export cancelled after {len(exported_cases)}/{total_count} failure image set(s).",
                )
            case_dir = export_root / current_failure_id
            file_map = _write_failure_export(
                export_dir=case_dir,
                failure={
                    **failure,
                    "run": {
                        "runId": summary["runId"],
                        "program": summary["program"],
                        "dataset": summary["dataset"],
                    },
                },
                images=images,
            )
            exported_cases.append(
                {
                    "failureId": current_failure_id,
                    "files": file_map,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    completed_count=index,
                    total_count=total_count,
                    current_failure_id=current_failure_id,
                    message=f"Saving {index}/{total_count} failure image sets...",
                )
        return {
            "runId": str(run_id),
            "scope": scope,
            "destinationDir": str(requested_destination) if requested_destination is not None else None,
            "exportDir": str(export_root),
            "caseCount": int(len(exported_cases)),
            "cases": exported_cases,
        }

    def _set_export_status(self, run_id: str, payload: Mapping[str, Any]) -> None:
        with self._export_lock:
            self._export_status_by_run[str(run_id)] = dict(payload)

    def get_export_status(self, run_id: str) -> Dict[str, Any]:
        run_key = str(run_id).strip()
        if not run_key:
            raise HTTPException(status_code=404, detail="Unknown offline eval run.")
        self.get_run(run_key)
        with self._export_lock:
            status = self._export_status_by_run.get(run_key)
            if status is None:
                return {
                    "runId": run_key,
                    "status": "idle",
                    "scope": "all_failures",
                    "completedCount": 0,
                    "totalCount": 0,
                    "message": "",
                    "destinationDir": None,
                    "exportDir": None,
                    "startedAt": None,
                    "finishedAt": None,
                    "currentFailureId": None,
                    "cancelRequested": False,
                }
            return dict(status)

    def cancel_export(self, run_id: str) -> Dict[str, Any]:
        run_key = str(run_id).strip()
        if not run_key:
            raise HTTPException(status_code=404, detail="Unknown offline eval run.")
        self.get_run(run_key)
        with self._export_lock:
            current = dict(self._export_status_by_run.get(run_key) or {})
            thread = self._export_threads.get(run_key)
            cancel_event = self._export_cancel_events.get(run_key)
            if not current:
                return {
                    "runId": run_key,
                    "status": "idle",
                    "scope": "all_failures",
                    "completedCount": 0,
                    "totalCount": 0,
                    "message": "",
                    "destinationDir": None,
                    "exportDir": None,
                    "startedAt": None,
                    "finishedAt": None,
                    "currentFailureId": None,
                    "cancelRequested": False,
                }
            status_key = str(current.get("status") or "").strip().lower()
            if thread is None or not thread.is_alive() or status_key not in {"running", "cancelling"}:
                return current
            if cancel_event is not None:
                cancel_event.set()
            completed_count = int(current.get("completedCount", 0) or 0)
            total_count = int(current.get("totalCount", 0) or 0)
            current["status"] = "cancelling"
            current["cancelRequested"] = True
            current["message"] = (
                f"Cancelling export after current item ({completed_count}/{total_count} saved)..."
            )
            self._export_status_by_run[run_key] = dict(current)
            return dict(current)

    def start_export(
        self,
        *,
        run_id: str,
        scope: str,
        failure_id: Optional[str] = None,
        destination_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        run_key = str(run_id).strip()
        if not run_key:
            raise HTTPException(status_code=404, detail="Unknown offline eval run.")
        summary, failures, export_root, requested_destination = self._prepare_export_request(
            run_id=run_key,
            scope=scope,
            failure_id=failure_id,
            destination_dir=destination_dir,
        )
        with self._export_lock:
            existing_thread = self._export_threads.get(run_key)
            if existing_thread is not None and existing_thread.is_alive():
                existing_status = self._export_status_by_run.get(run_key)
                if existing_status is not None:
                    return dict(existing_status)
            started_at = _utc_now_iso()
            cancel_event = threading.Event()
            status_payload: Dict[str, Any] = {
                "runId": run_key,
                "status": "running",
                "scope": scope,
                "completedCount": 0,
                "totalCount": int(len(failures)),
                "message": f"Saving 0/{len(failures)} failure image sets...",
                "destinationDir": str(requested_destination) if requested_destination is not None else None,
                "exportDir": str(export_root),
                "startedAt": started_at,
                "finishedAt": None,
                "currentFailureId": None,
                "cancelRequested": False,
            }
            self._export_status_by_run[run_key] = dict(status_payload)
            self._export_cancel_events[run_key] = cancel_event

        def _progress_update(
            *,
            completed_count: int,
            total_count: int,
            current_failure_id: Optional[str],
            message: str,
        ) -> None:
            with self._export_lock:
                current = dict(self._export_status_by_run.get(run_key, status_payload))
                current["status"] = "cancelling" if cancel_event.is_set() else "running"
                current["completedCount"] = int(completed_count)
                current["totalCount"] = int(total_count)
                current["currentFailureId"] = current_failure_id
                current["message"] = str(message)
                current["cancelRequested"] = bool(cancel_event.is_set())
                self._export_status_by_run[run_key] = current

        def _export_thread_target() -> None:
            try:
                result = self._execute_export(
                    run_id=run_key,
                    scope=scope,
                    summary=summary,
                    failures=failures,
                    export_root=export_root,
                    requested_destination=requested_destination,
                    progress_callback=_progress_update,
                    cancel_event=cancel_event,
                )
                self._set_export_status(
                    run_key,
                    {
                        "runId": run_key,
                        "status": "completed",
                        "scope": scope,
                        "completedCount": int(result["caseCount"]),
                        "totalCount": int(len(failures)),
                        "message": f"Exported {result['caseCount']} case(s) to {result['exportDir']}",
                        "destinationDir": result["destinationDir"],
                        "exportDir": result["exportDir"],
                        "startedAt": started_at,
                        "finishedAt": _utc_now_iso(),
                        "currentFailureId": None,
                        "cancelRequested": False,
                    },
                )
            except OfflineEvalExportCancelled as error:
                completed_count = 0
                with self._export_lock:
                    completed_count = int(self._export_status_by_run.get(run_key, {}).get("completedCount", 0))
                self._set_export_status(
                    run_key,
                    {
                        "runId": run_key,
                        "status": "cancelled",
                        "scope": scope,
                        "completedCount": completed_count,
                        "totalCount": int(len(failures)),
                        "message": str(error),
                        "destinationDir": str(requested_destination) if requested_destination is not None else None,
                        "exportDir": str(export_root),
                        "startedAt": started_at,
                        "finishedAt": _utc_now_iso(),
                        "currentFailureId": None,
                        "cancelRequested": False,
                    },
                )
            except Exception as error:  # noqa: BLE001
                completed_count = 0
                with self._export_lock:
                    completed_count = int(self._export_status_by_run.get(run_key, {}).get("completedCount", 0))
                self._set_export_status(
                    run_key,
                    {
                        "runId": run_key,
                        "status": "failed",
                        "scope": scope,
                        "completedCount": completed_count,
                        "totalCount": int(len(failures)),
                        "message": str(error),
                        "destinationDir": str(requested_destination) if requested_destination is not None else None,
                        "exportDir": str(export_root),
                        "startedAt": started_at,
                        "finishedAt": _utc_now_iso(),
                        "currentFailureId": None,
                        "cancelRequested": False,
                    },
                )
            finally:
                with self._export_lock:
                    self._export_threads.pop(run_key, None)
                    self._export_cancel_events.pop(run_key, None)

        thread = threading.Thread(
            target=_export_thread_target,
            name=f"offline-eval-export-{run_key}",
            daemon=True,
        )
        with self._export_lock:
            self._export_threads[run_key] = thread
        thread.start()
        return status_payload

    def export_failures(
        self,
        *,
        run_id: str,
        scope: str,
        failure_id: Optional[str] = None,
        destination_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        summary, failures, export_root, requested_destination = self._prepare_export_request(
            run_id=str(run_id),
            scope=scope,
            failure_id=failure_id,
            destination_dir=destination_dir,
        )
        return self._execute_export(
            run_id=str(run_id),
            scope=scope,
            summary=summary,
            failures=failures,
            export_root=export_root,
            requested_destination=requested_destination,
        )
