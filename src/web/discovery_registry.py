from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from collections import deque
from copy import deepcopy
from datetime import datetime
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat as stat_module
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from typing import Any, Callable, Dict, Iterator, Optional

import yaml

from src.visualization import build_visualization_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_RUN_METADATA_DIRNAME = ".discovery_web"
DISCOVERY_DASHBOARD_BOOT_TIMEOUT_SEC = 20.0
DISCOVERY_RUN_HEARTBEAT_INTERVAL_SEC = 5.0
DISCOVERY_RUN_PAUSED_HEARTBEAT_INTERVAL_SEC = 30.0
DISCOVERY_RUN_HEARTBEAT_TIMEOUT_SEC = 4.0
DISCOVERY_RUN_HEARTBEAT_TIMEOUT_MULTIPLIER = 4.0
DISCOVERY_RUN_STATUS_UPDATE_STALE_SEC = 30.0 * 60.0
DISCOVERY_RUN_STATUS_UPDATE_TOUCH_INTERVAL_SEC = 30.0
DISCOVERY_RUN_SCAN_CACHE_TTL_SEC = 0.75
DISCOVERY_VIEWER_STATE_TIMEOUT_SEC = 3.0
DISCOVERY_VIEWER_STATE_CACHE_TTL_SEC = 60.0
DISCOVERY_DASHBOARD_PORT_SEARCH_WINDOW = 200
DISCOVERY_DASHBOARD_START_PORT_ATTEMPTS = 10
DISCOVERY_DASHBOARD_HEALTHCHECK_VERSION = 1
DISCOVERY_ATOMIC_REPLACE_RETRY_DELAYS_SEC = (
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
_DISCOVERY_SCAN_CACHE: Dict[str, Any] = {
    "at": 0.0,
    "paths": (),
}
_DISCOVERY_COMPONENT_CACHE_LOCK = threading.Lock()
_DISCOVERY_LOG_TAIL_CACHE: Dict[str, Dict[str, Any]] = {}
_DISCOVERY_RUN_SUMMARY_CACHE: Dict[str, Dict[str, Any]] = {}
_DISCOVERY_TRANSITION_GALLERY_FILE_CACHE: Dict[str, Dict[str, Any]] = {}
_DISCOVERY_PROGRAM_VERSIONS_CACHE: Dict[str, Dict[str, Any]] = {}
_TRANSITION_STEP_DIR_RE = re.compile(r"^step(?P<step>\d+)$", re.IGNORECASE)
_TRANSITION_FAIL_DIR_RE = re.compile(r"^fail(?P<fail>\d+)$", re.IGNORECASE)
_TRANSITION_VERSION_RE = re.compile(r"(v\d+(?::g\d+)?)", re.IGNORECASE)
DISCOVERY_RUN_DELETE_CONFIRMATION_TEXT = "Yes. Delete this run."
EXPERIMENT_CONFIG_SNAPSHOT_FILENAMES = (
    "experiment_config.resolved.yaml",
    "experiment_config.yaml",
)


class DiscoveryRunDeleteError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = str(detail)
        self.status_code = int(status_code)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seconds_since(timestamp: Any) -> Optional[float]:
    parsed = _parse_iso(timestamp)
    if parsed is None:
        return None
    return max(0.0, (datetime.now() - parsed).total_seconds())


def _heartbeat_timeout_for_interval(interval_sec: float) -> float:
    return max(
        float(DISCOVERY_RUN_HEARTBEAT_TIMEOUT_SEC),
        float(interval_sec) * float(DISCOVERY_RUN_HEARTBEAT_TIMEOUT_MULTIPLIER),
    )


def _pid_appears_alive(pid: Any) -> bool:
    try:
        resolved_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if resolved_pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                resolved_pid,
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return int(exit_code.value) == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(resolved_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _resolve_run_status(state: Dict[str, Any]) -> Dict[str, Any]:
    raw_status = str(state.get("status") or "unknown")
    heartbeat_timeout_sec = float(
        state.get("heartbeatTimeoutSec") or DISCOVERY_RUN_HEARTBEAT_TIMEOUT_SEC
    )
    heartbeat_age_sec = _seconds_since(state.get("lastHeartbeatAt"))
    status_update_stale_sec = float(
        state.get("statusUpdateStaleSec") or DISCOVERY_RUN_STATUS_UPDATE_STALE_SEC
    )
    status_update_age_sec = _seconds_since(state.get("lastStatusUpdateAt"))
    pid = state.get("pid")
    effective_status = raw_status
    is_live = False
    status_reason = None

    if raw_status == "running":
        if (
            status_update_age_sec is not None
            and status_update_age_sec >= status_update_stale_sec
        ):
            effective_status = "interrupted"
            status_reason = f"status update stale ({status_update_age_sec:.1f}s)"
        elif heartbeat_age_sec is None:
            effective_status = "interrupted"
            status_reason = "heartbeat missing"
        elif heartbeat_age_sec <= heartbeat_timeout_sec:
            effective_status = "running"
            is_live = True
            status_reason = "heartbeat fresh"
        elif _pid_appears_alive(pid):
            effective_status = "running"
            is_live = True
            status_reason = f"process alive; heartbeat lagging ({heartbeat_age_sec:.1f}s)"
        else:
            effective_status = "interrupted"
            status_reason = f"heartbeat stale ({heartbeat_age_sec:.1f}s)"
    elif raw_status == "queued":
        status_reason = "waiting to start"
    elif raw_status == "completed":
        status_reason = "finished normally"
    elif raw_status == "failed":
        status_reason = "terminated with error"

    return {
        "rawStatus": raw_status,
        "status": effective_status,
        "effectiveStatus": effective_status,
        "isLive": is_live,
        "lastHeartbeatAt": state.get("lastHeartbeatAt"),
        "heartbeatAgeSec": heartbeat_age_sec,
        "heartbeatIntervalSec": state.get("heartbeatIntervalSec"),
        "heartbeatTimeoutSec": heartbeat_timeout_sec,
        "lastStatusUpdateAt": state.get("lastStatusUpdateAt"),
        "statusUpdateAgeSec": status_update_age_sec,
        "statusUpdateStaleSec": status_update_stale_sec,
        "pid": state.get("pid"),
        "statusReason": status_reason,
    }


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, set):
        return [_safe_json_value(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _safe_json_value(value.item())
        except Exception:
            return str(value)
    return str(value)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(path)
    temp_path.write_text(
        json.dumps(_safe_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _replace_atomic_with_retry(temp_path, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not _path_exists(path):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_text(path: Path, *, max_lines: int = 10000) -> str:
    if not _path_exists(path):
        return ""
    if max_lines <= 0:
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            tail_lines = deque(handle, maxlen=max_lines)
    except OSError:
        return ""
    return "".join(tail_lines)


def _path_stat_signature(path: Path) -> tuple[Any, ...]:
    try:
        stat_result = path.stat()
    except OSError:
        return (False, None, None, None)
    try:
        kind = "dir" if stat_module.S_ISDIR(int(getattr(stat_result, "st_mode", 0))) else "file"
    except (TypeError, ValueError):
        kind = None
    return (
        True,
        kind,
        int(getattr(stat_result, "st_mtime_ns", 0)),
        int(getattr(stat_result, "st_size", 0)),
    )


def _path_exists(path: Path) -> bool:
    return bool(_path_stat_signature(path)[0])


def _path_is_missing(path: Path) -> bool:
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _cache_key_for_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _read_component_cache(
    cache: Dict[str, Dict[str, Any]],
    *,
    cache_key: str,
    signature: tuple[Any, ...],
) -> Optional[Any]:
    with _DISCOVERY_COMPONENT_CACHE_LOCK:
        cached = cache.get(str(cache_key))
        if not isinstance(cached, dict):
            return None
        if tuple(cached.get("signature") or ()) != tuple(signature):
            return None
        return cached.get("value")


def _write_component_cache(
    cache: Dict[str, Dict[str, Any]],
    *,
    cache_key: str,
    signature: tuple[Any, ...],
    value: Any,
) -> Any:
    with _DISCOVERY_COMPONENT_CACHE_LOCK:
        cache[str(cache_key)] = {
            "signature": tuple(signature),
            "value": value,
        }
    return value


def _shallow_tree_digest(root: Path, *, max_depth: int = 2) -> str:
    if _path_is_missing(root):
        return "missing"
    exists, kind, mtime_ns, size = _path_stat_signature(root)
    if not exists:
        return "unavailable"
    if kind != "dir":
        return f"{exists}:{kind}:{mtime_ns}:{size}"

    hasher = hashlib.sha1()

    def _visit(directory: Path, depth: int, prefix: str) -> None:
        try:
            with os.scandir(directory) as scan_iter:
                entries = sorted(
                    list(scan_iter),
                    key=lambda entry: entry.name.lower(),
                )
        except OSError:
            hasher.update(f"ERR:{prefix}".encode("utf-8", errors="replace"))
            return
        for entry in entries:
            relative_path = f"{prefix}{entry.name}"
            try:
                stat_result = entry.stat(follow_symlinks=False)
            except OSError:
                hasher.update(f"MISS:{relative_path}".encode("utf-8", errors="replace"))
                continue
            is_dir = bool(entry.is_dir(follow_symlinks=False))
            entry_parts = (
                relative_path,
                "dir" if is_dir else "file",
                str(int(getattr(stat_result, "st_mtime_ns", 0))),
                str(int(getattr(stat_result, "st_size", 0))),
            )
            hasher.update("|".join(entry_parts).encode("utf-8", errors="replace"))
            if is_dir and depth < int(max_depth):
                _visit(Path(entry.path), depth + 1, f"{relative_path}/")

    _visit(root, 0, "")
    return hasher.hexdigest()


def _cached_tail_text(path: Path, *, max_lines: int = 10000) -> str:
    cache_key = _cache_key_for_path(path)
    signature = (int(max_lines),) + _path_stat_signature(path)
    cached_value = _read_component_cache(
        _DISCOVERY_LOG_TAIL_CACHE,
        cache_key=cache_key,
        signature=signature,
    )
    if isinstance(cached_value, str):
        return cached_value
    resolved = _tail_text(path, max_lines=max_lines)
    return str(
        _write_component_cache(
            _DISCOVERY_LOG_TAIL_CACHE,
            cache_key=cache_key,
            signature=signature,
            value=resolved,
        )
    )


def _run_metadata_dir(run_output_dir: Path) -> Path:
    return run_output_dir / DISCOVERY_RUN_METADATA_DIRNAME


def _viewer_state_path(run_dir: Path) -> Path:
    return run_dir / "viewers.json"


def _load_viewer_clients(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    payload = _read_json(_viewer_state_path(run_dir))
    raw_clients = payload.get("clients") if isinstance(payload, dict) else None
    if not isinstance(raw_clients, dict):
        return {}
    clients: Dict[str, Dict[str, Any]] = {}
    for client_id, entry in raw_clients.items():
        normalized_client_id = str(client_id or "").strip()
        if not normalized_client_id or not isinstance(entry, dict):
            continue
        clients[normalized_client_id] = dict(entry)
    return clients


def _prune_stale_viewer_clients(
    clients: Dict[str, Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    current_time = now or datetime.now()
    fresh_clients: Dict[str, Dict[str, Any]] = {}
    for client_id, entry in clients.items():
        seen_at = _parse_iso(entry.get("lastSeenAt"))
        if seen_at is None:
            continue
        age_sec = max(0.0, (current_time - seen_at).total_seconds())
        if age_sec > DISCOVERY_VIEWER_STATE_TIMEOUT_SEC:
            continue
        fresh_clients[client_id] = entry
    return fresh_clients


def _write_viewer_clients(run_dir: Path, clients: Dict[str, Dict[str, Any]]) -> None:
    _write_json_atomic(
        _viewer_state_path(run_dir),
        {
            "clients": clients,
            "updatedAt": _now_iso(),
        },
    )


def _normalize_client_mode(raw_mode: Any) -> str:
    normalized_mode = str(raw_mode or "live").strip().lower()
    if normalized_mode == "paused":
        return "paused"
    return "live"


def _resolve_effective_client_mode(run_dir: Path, key: str) -> Optional[str]:
    raw_clients = _load_viewer_clients(run_dir)
    if not raw_clients:
        return None
    fresh_clients = _prune_stale_viewer_clients(raw_clients)
    if fresh_clients:
        fresh_modes = {
            _normalize_client_mode(entry.get(key))
            for entry in fresh_clients.values()
            if isinstance(entry, dict)
        }
        if "live" in fresh_modes:
            return "live"
        if "paused" in fresh_modes:
            return "paused"
    latest_seen_at: Optional[datetime] = None
    latest_mode: Optional[str] = None
    for entry in raw_clients.values():
        if not isinstance(entry, dict):
            continue
        seen_at = _parse_iso(entry.get("lastSeenAt"))
        if seen_at is None:
            continue
        mode = _normalize_client_mode(entry.get(key))
        if latest_seen_at is None or seen_at > latest_seen_at:
            latest_seen_at = seen_at
            latest_mode = mode
    return latest_mode


def _resolve_effective_viewer_mode(run_dir: Path) -> Optional[str]:
    return _resolve_effective_client_mode(run_dir, "viewMode")


def _resolve_effective_projection_mode(run_dir: Path) -> Optional[str]:
    return _resolve_effective_client_mode(run_dir, "projectionMode")


def _resolve_effective_client_bool(
    run_dir: Path,
    key: str,
    *,
    default: bool,
) -> bool:
    raw_clients = _load_viewer_clients(run_dir)
    latest_seen_at: Optional[datetime] = None
    latest_value = bool(default)
    for client_pool in (_prune_stale_viewer_clients(raw_clients), raw_clients):
        for entry in client_pool.values():
            if not isinstance(entry, dict):
                continue
            seen_at = _parse_iso(entry.get("lastSeenAt"))
            if seen_at is None:
                continue
            if latest_seen_at is not None and seen_at <= latest_seen_at:
                continue
            latest_seen_at = seen_at
            latest_value = bool(entry.get(key)) if isinstance(entry.get(key), bool) else bool(default)
        if latest_seen_at is not None:
            break
    return bool(latest_value)


def _resolve_effective_projection_sparse_class_filter(run_dir: Path) -> bool:
    return _resolve_effective_client_bool(
        run_dir,
        "projectionExcludeSparseClasses",
        default=False,
    )


def _invalidate_discovery_scan_cache() -> None:
    _DISCOVERY_SCAN_CACHE["at"] = 0.0
    _DISCOVERY_SCAN_CACHE["paths"] = ()
    with _DISCOVERY_COMPONENT_CACHE_LOCK:
        _DISCOVERY_LOG_TAIL_CACHE.clear()
        _DISCOVERY_RUN_SUMMARY_CACHE.clear()
        _DISCOVERY_TRANSITION_GALLERY_FILE_CACHE.clear()
        _DISCOVERY_PROGRAM_VERSIONS_CACHE.clear()


def _persisted_run_output_dir(run_dir: Path, state: Dict[str, Any]) -> Optional[Path]:
    if run_dir.name == DISCOVERY_RUN_METADATA_DIRNAME:
        parent = run_dir.parent
        if not _path_is_missing(parent):
            return parent
        return None
    configured = state.get("runOutputDir")
    if isinstance(configured, str) and configured.strip():
        candidate = Path(configured)
        if not _path_is_missing(candidate):
            return candidate
    return None


def _resolve_existing_path_within_root(root: Path, candidate: Path) -> Optional[Path]:
    try:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved_candidate.exists():
        return None
    return resolved_candidate


def _delete_tree_within_root(
    root: Path,
    candidate: Path,
    *,
    deleted_paths: list[str],
) -> bool:
    resolved_candidate = _resolve_existing_path_within_root(root, candidate)
    if resolved_candidate is None or not resolved_candidate.is_dir():
        return False
    relative_path = resolved_candidate.relative_to(root.resolve()).as_posix()
    try:
        shutil.rmtree(resolved_candidate)
    except OSError:
        return False
    deleted_paths.append(relative_path)
    return True


def _path_is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _experiments_root() -> Path:
    return PROJECT_ROOT / "experiments"


def _scan_metadata_dirs_under_root(root: Path, *, max_depth: int) -> list[Path]:
    if not root.exists():
        return []
    discovered: list[Path] = []
    frontier = [root]
    for _ in range(max(1, int(max_depth))):
        next_frontier: list[Path] = []
        for parent in frontier:
            try:
                children = list(parent.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                metadata_dir = child / DISCOVERY_RUN_METADATA_DIRNAME
                state_path = metadata_dir / "state.json"
                if state_path.exists():
                    discovered.append(metadata_dir)
                next_frontier.append(child)
        frontier = next_frontier
    return discovered


def _scan_discovery_metadata_dirs() -> tuple[Path, ...]:
    now = time.monotonic()
    cached_at = float(_DISCOVERY_SCAN_CACHE.get("at") or 0.0)
    if now - cached_at <= DISCOVERY_RUN_SCAN_CACHE_TTL_SEC:
        cached_paths = _DISCOVERY_SCAN_CACHE.get("paths") or ()
        return tuple(Path(path) for path in cached_paths)

    discovered: list[Path] = []
    for root, depth in ((_experiments_root(), 1),):
        discovered.extend(_scan_metadata_dirs_under_root(root, max_depth=depth))
    discovered.sort(key=lambda path: str(path))
    _DISCOVERY_SCAN_CACHE["at"] = now
    _DISCOVERY_SCAN_CACHE["paths"] = tuple(str(path) for path in discovered)
    return tuple(discovered)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(path)
    temp_path.write_text(text, encoding="utf-8")
    _replace_atomic_with_retry(temp_path, path)


def _atomic_temp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def _is_retryable_atomic_replace_error(exc: BaseException) -> bool:
    if not isinstance(exc, PermissionError):
        return False
    if os.name != "nt":
        return False
    winerror = getattr(exc, "winerror", None)
    return winerror in {5, 32}


def _replace_atomic_with_retry(temp_path: Path, destination_path: Path) -> None:
    delays_sec = DISCOVERY_ATOMIC_REPLACE_RETRY_DELAYS_SEC
    try:
        for attempt_index, delay_sec in enumerate(delays_sec):
            try:
                temp_path.replace(destination_path)
                return
            except PermissionError as exc:
                if not _is_retryable_atomic_replace_error(exc):
                    raise
                if attempt_index >= len(delays_sec) - 1:
                    raise
                time.sleep(float(delay_sec))
        temp_path.replace(destination_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _count_jsonl_lines(path: Path) -> int:
    if not _path_exists(path):
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _allowed_discovery_run_output_roots() -> tuple[Path, ...]:
    return (_experiments_root(),)


def _summarize_llm_config_payload(raw_payload: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(raw_payload, dict):
        payload = dict(raw_payload)
    else:
        return {}
    inference_cfg = payload.get("inference") if isinstance(payload, dict) else None
    if not isinstance(inference_cfg, dict):
        return {}

    provider = str(inference_cfg.get("provider") or "unknown").strip() or "unknown"
    mode = "api"
    model = ""
    if provider == "gemini":
        model = str(inference_cfg.get("gemini_model") or "").strip()
    elif provider == "openai":
        model = str(inference_cfg.get("openai_model") or "").strip()

    thinking_level = ""
    if provider == "gemini":
        thinking_level = str(
            inference_cfg.get("gemini_thinking_level")
            or ""
        ).strip()

    reasoning_effort = ""
    if provider == "openai":
        reasoning_effort = str(
            inference_cfg.get("openai_reasoning_effort")
            or inference_cfg.get("reasoning_effort")
            or ""
        ).strip()

    summary: Dict[str, Any] = {
        "provider": provider,
        "mode": mode,
    }
    if model:
        summary["model"] = model
    if thinking_level:
        summary["thinkingLevel"] = thinking_level
    if reasoning_effort:
        summary["reasoningEffort"] = reasoning_effort
    return summary


def _summarize_llm_config(llm_config_path: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if isinstance(llm_config_path, dict):
        payload = dict(llm_config_path)
    elif isinstance(llm_config_path, str) and llm_config_path.strip():
        raw_value = llm_config_path.strip()
        if raw_value.startswith("{") and raw_value.endswith("}"):
            try:
                literal_payload = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                literal_payload = None
            if isinstance(literal_payload, dict):
                payload = dict(literal_payload)
        if not payload:
            try:
                from src.llm.predictor_factory import load_llm_config
            except Exception:
                return {}
            try:
                payload = load_llm_config(raw_value)
            except Exception:
                return {}
    else:
        return {}
    return _summarize_llm_config_payload(payload)


def _resolve_llm_config_summary(
    state: Dict[str, Any],
    run_output_dir: Optional[Path],
) -> Dict[str, Any]:
    persisted_summary = state.get("llmConfigSummary")
    if isinstance(persisted_summary, dict) and persisted_summary:
        return dict(persisted_summary)

    if run_output_dir is not None:
        snapshot_path = run_output_dir / "config_snapshot" / "llm_config.yaml"
        if snapshot_path.exists():
            snapshot_summary = _summarize_llm_config(str(snapshot_path))
            if snapshot_summary:
                return snapshot_summary

    return _summarize_llm_config(state.get("llmConfigPath"))


def _extract_runtime_llm_identity_from_usage_summary(raw_summary: Any) -> Dict[str, Any]:
    if not isinstance(raw_summary, dict):
        return {}
    events = raw_summary.get("events")
    if not isinstance(events, list):
        return {}

    for raw_event in reversed(events):
        if not isinstance(raw_event, dict):
            continue
        provider = str(raw_event.get("provider") or "").strip()
        model = str(raw_event.get("model") or "").strip()
        if not provider and not model:
            continue
        resolved: Dict[str, Any] = {}
        if provider:
            resolved["provider"] = provider
            resolved["mode"] = "api"
        if model:
            resolved["model"] = model
        return resolved
    return {}


def _filter_llm_runtime_summary_for_provider(summary: Dict[str, Any]) -> Dict[str, Any]:
    provider = str(summary.get("provider") or "").strip()
    if not provider:
        return dict(summary)

    filtered = dict(summary)
    if provider == "gemini":
        filtered.pop("reasoningEffort", None)
    else:
        filtered.pop("thinkingLevel", None)
        if provider != "openai":
            filtered.pop("reasoningEffort", None)
    return filtered


def _resolve_llm_runtime_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    config_summary = summary.get("llmConfigSummary")
    resolved = dict(config_summary) if isinstance(config_summary, dict) else {}

    for candidate_key in ("llmUsageSummary",):
        runtime_identity = _extract_runtime_llm_identity_from_usage_summary(summary.get(candidate_key))
        if runtime_identity:
            resolved.update(runtime_identity)
            break

    if not resolved:
        return {}
    return _filter_llm_runtime_summary_for_provider(resolved)


def _load_dashboard_visual_config_from_run_output(run_output_dir: Path) -> Dict[str, Any]:
    snapshot_dir = run_output_dir / "config_snapshot"
    for snapshot_filename in EXPERIMENT_CONFIG_SNAPSHOT_FILENAMES:
        snapshot_path = snapshot_dir / snapshot_filename
        if not _path_exists(snapshot_path):
            continue
        try:
            payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}
        if not isinstance(payload, dict):
            return {}
        serialization = payload.get("serialization")
        if not isinstance(serialization, dict):
            return {}
        return build_visualization_config(serialization.get("word_aliases"))
    return {}


def _ensure_dashboard_visual_config(
    run_output_dir: Path,
    dashboard_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(dashboard_payload or {})
    if isinstance(payload.get("visualConfig"), dict):
        return payload
    visual_config = _load_dashboard_visual_config_from_run_output(run_output_dir)
    if visual_config:
        payload["visualConfig"] = visual_config
    return payload


def _build_dashboard_offline_eval_source(raw_source: Any) -> Optional[str]:
    source = str(raw_source or "").rstrip()
    if not source.strip():
        return None
    if "def predict_next_state(" in source:
        return source.rstrip() + "\n"
    return None


def _hydrate_dashboard_program_versions_from_paths(
    run_output_dir: Path,
    versions: Any,
) -> tuple[Any, bool]:
    if not isinstance(versions, list):
        return versions, False
    hydrated_versions = []
    changed = False
    for entry in versions:
        if not isinstance(entry, dict):
            hydrated_versions.append(entry)
            continue
        next_entry = dict(entry)
        existing_source = next_entry.get("source")
        if not (isinstance(existing_source, str) and existing_source.strip()):
            source_path = next_entry.get("sourcePath")
            resolved_source_path = _resolve_run_output_relative_path(run_output_dir, source_path)
            if resolved_source_path is not None:
                try:
                    raw_source = resolved_source_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    raw_source = None
                eval_source = _build_dashboard_offline_eval_source(raw_source)
                if isinstance(eval_source, str) and eval_source.strip():
                    next_entry["source"] = eval_source
                    next_entry["sourceDigest"] = _text_sha1(eval_source)
                    next_entry["sourceLineCount"] = len(eval_source.splitlines())
                    changed = True
        hydrated_versions.append(next_entry)
    return (hydrated_versions if changed else versions), changed


def _ensure_dashboard_program_version_sources(
    run_output_dir: Path,
    dashboard_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(dashboard_payload or {})
    program_versions, program_versions_changed = _hydrate_dashboard_program_versions_from_paths(
        run_output_dir,
        payload.get("programVersions"),
    )
    if program_versions_changed:
        payload["programVersions"] = program_versions

    transition_gallery = payload.get("transitionGallery")
    if isinstance(transition_gallery, dict):
        gallery_versions, gallery_versions_changed = _hydrate_dashboard_program_versions_from_paths(
            run_output_dir,
            transition_gallery.get("versions"),
        )
        if gallery_versions_changed:
            next_gallery = dict(transition_gallery)
            next_gallery["versions"] = gallery_versions
            payload["transitionGallery"] = next_gallery
    return payload


def _run_output_summary_signature(run_output_dir: Path) -> tuple[Any, ...]:
    return (
        "program",
        _shallow_tree_digest(run_output_dir / "program_versions", max_depth=1),
        _shallow_tree_digest(run_output_dir / "new_transition_images", max_depth=3),
        _shallow_tree_digest(run_output_dir / "config_snapshot", max_depth=1),
        _path_stat_signature(run_output_dir / "llm_usage_summary.json"),
        _path_stat_signature(run_output_dir / "program_history.jsonl"),
    )


def _summarize_run_output_uncached(run_output_dir: Optional[Path]) -> Dict[str, Any]:
    if run_output_dir is None or not _path_exists(run_output_dir):
        return {}

    program_versions_dir = run_output_dir / "program_versions"
    llm_usage_summary_path = run_output_dir / "llm_usage_summary.json"
    history_path = run_output_dir / "program_history.jsonl"
    new_transition_images_dir = run_output_dir / "new_transition_images"

    latest_version = None
    version_count = 0
    if _path_exists(program_versions_dir):
        versions = sorted(program_versions_dir.glob("v*.py"))
        version_count = len(versions)
        if versions:
            latest_version = versions[-1].name

    llm_usage_summary = None
    if _path_exists(llm_usage_summary_path):
        try:
            llm_usage_summary = json.loads(llm_usage_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            llm_usage_summary = None

    image_bundle_count = 0
    if _path_exists(new_transition_images_dir):
        try:
            image_bundle_count = sum(
                1 for path in new_transition_images_dir.rglob("*.json") if path.is_file()
            )
        except OSError:
            image_bundle_count = 0

    return {
        "runOutputDir": str(run_output_dir),
        "versionCount": int(version_count),
        "latestVersion": latest_version,
        "programHistoryCount": _count_jsonl_lines(history_path),
        "newTransitionBundleCount": int(image_bundle_count),
        "llmUsageSummary": llm_usage_summary,
        "configSnapshotFiles": _load_config_snapshot_files(run_output_dir),
    }


def summarize_run_output(run_output_dir: Optional[Path]) -> Dict[str, Any]:
    if run_output_dir is None or _path_is_missing(run_output_dir):
        return {}
    cache_key = _cache_key_for_path(run_output_dir)
    try:
        signature = _run_output_summary_signature(run_output_dir)
    except OSError:
        return {}
    cached_value = _read_component_cache(
        _DISCOVERY_RUN_SUMMARY_CACHE,
        cache_key=cache_key,
        signature=signature,
    )
    if isinstance(cached_value, dict):
        return dict(cached_value)
    try:
        resolved = _summarize_run_output_uncached(run_output_dir)
    except OSError:
        resolved = {}
    cached_summary = (
        dict(resolved)
        if isinstance(resolved, dict)
        else {}
    )
    return dict(
        _write_component_cache(
            _DISCOVERY_RUN_SUMMARY_CACHE,
            cache_key=cache_key,
            signature=signature,
            value=cached_summary,
        )
    )


def _parse_prefixed_int(value: Any, *, pattern: re.Pattern[str], key: str) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    matched = pattern.match(text)
    if matched is None:
        return None
    try:
        resolved = int(matched.group(key))
    except (IndexError, TypeError, ValueError):
        return None
    return resolved if resolved >= 0 else None


def _normalize_commit_version(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    matched = _TRANSITION_VERSION_RE.search(text)
    if matched is None:
        return None
    version_text = str(matched.group(1) or "").strip().lower()
    if not version_text:
        return None
    if ":" in version_text:
        version_text = version_text.split(":", 1)[0]
    if not re.fullmatch(r"v\d+", version_text):
        return None
    return version_text


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _normalize_group_version_key(value: Any, *, commit_version: Optional[str] = None) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        text = str(commit_version or "").strip().lower()
    if not text:
        return None
    if ":" not in text:
        normalized_commit_version = _normalize_commit_version(text)
        if normalized_commit_version is None:
            return None
        return f"{normalized_commit_version}:g0000"
    version_part, group_part = text.split(":", 1)
    normalized_commit_version = _normalize_commit_version(version_part)
    normalized_group_part = group_part.strip().lower()
    if normalized_commit_version is None:
        return None
    if not re.fullmatch(r"g\d+", normalized_group_part):
        return f"{normalized_commit_version}:g0000"
    return f"{normalized_commit_version}:{normalized_group_part}"


def _display_version_label(value: Any) -> str:
    normalized = str(value or "").strip()
    return normalized.upper() if normalized else "-"


def _commit_version_sort_key(value: Any) -> tuple[int, str]:
    normalized_commit_version = _normalize_commit_version(value)
    if normalized_commit_version is None:
        return (10**9, str(value or "").lower())
    try:
        return (int(normalized_commit_version[1:]), normalized_commit_version)
    except ValueError:
        return (10**9, normalized_commit_version)


def _relative_run_artifact_path(run_output_dir: Path, artifact_path: Path) -> Optional[str]:
    try:
        relative_path = artifact_path.resolve().relative_to(run_output_dir.resolve())
    except (OSError, ValueError):
        return None
    return relative_path.as_posix()


def _resolve_run_output_relative_path(
    run_output_dir: Path,
    relative_path: Any,
) -> Optional[Path]:
    normalized_relative_path = str(relative_path or "").strip().replace("\\", "/")
    if not normalized_relative_path:
        return None
    candidate_path = run_output_dir / Path(normalized_relative_path)
    try:
        resolved_path = candidate_path.resolve()
        resolved_path.relative_to(run_output_dir.resolve())
    except (OSError, ValueError):
        return None
    if not resolved_path.exists() or not resolved_path.is_file():
        return None
    return resolved_path


def _text_sha1(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _build_run_artifact_url(run_id: str, relative_path: str) -> str:
    normalized_relative_path = str(relative_path or "").strip().replace("\\", "/")
    quoted_run_id = urllib.parse.quote(str(run_id or "").strip(), safe="")
    quoted_relative_path = urllib.parse.quote(normalized_relative_path, safe="/-_.")
    return f"/api/runs/{quoted_run_id}/artifacts/{quoted_relative_path}"


def _artifact_payload_for_path(
    run_output_dir: Path,
    run_id: str,
    artifact_path: Optional[Path],
) -> Optional[Dict[str, str]]:
    if artifact_path is None or not artifact_path.exists() or not artifact_path.is_file():
        return None
    relative_path = _relative_run_artifact_path(run_output_dir, artifact_path)
    if not relative_path:
        return None
    return {
        "path": relative_path,
        "url": _build_run_artifact_url(run_id, relative_path),
    }


def _artifact_payload_for_relative_path(
    run_output_dir: Path,
    run_id: str,
    relative_path: Any,
) -> Optional[Dict[str, str]]:
    resolved_path = _resolve_run_output_relative_path(run_output_dir, relative_path)
    return _artifact_payload_for_path(run_output_dir, run_id, resolved_path)


def _load_transition_bundle_metadata(bundle_path: Optional[Path]) -> Dict[str, Any]:
    if bundle_path is None:
        return {}
    payload = _read_json(bundle_path)
    return payload if isinstance(payload, dict) else {}


def _bundle_nested_value(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _bundle_transition_terminated(
    payload: Dict[str, Any],
    *,
    primary_state_key: Optional[str],
    fallback_state_key: Optional[str] = None,
) -> Optional[bool]:
    for state_key in (primary_state_key, fallback_state_key):
        if not state_key:
            continue
        terminated = _bundle_nested_value(payload, state_key, "step", "terminated")
        if isinstance(terminated, bool):
            return terminated
    return None


def _bundle_transition_world_index(payload: Dict[str, Any]) -> Optional[int]:
    for candidate in (
        payload.get("world_index"),
        _bundle_nested_value(payload, "previous_state", "world_index"),
        _bundle_nested_value(payload, "actual_next_state", "world_index"),
        _bundle_nested_value(payload, "predicted_next_state", "world_index"),
        _bundle_nested_value(payload, "previous_state_visual", "world_index"),
        _bundle_nested_value(payload, "actual_next_state_visual", "world_index"),
        _bundle_nested_value(payload, "predicted_next_state_visual", "world_index"),
    ):
        if isinstance(candidate, int) and int(candidate) > 0:
            return int(candidate)
    return None


def _bundle_transition_map_name(payload: Dict[str, Any]) -> Optional[str]:
    for candidate in (
        payload.get("map_name"),
        _bundle_nested_value(payload, "previous_state", "map_name"),
        _bundle_nested_value(payload, "actual_next_state", "map_name"),
        _bundle_nested_value(payload, "predicted_next_state", "map_name"),
        _bundle_nested_value(payload, "previous_state_visual", "map_name"),
        _bundle_nested_value(payload, "actual_next_state_visual", "map_name"),
        _bundle_nested_value(payload, "predicted_next_state_visual", "map_name"),
        _bundle_nested_value(payload, "previous_state", "scenario_type"),
        _bundle_nested_value(payload, "actual_next_state", "scenario_type"),
        _bundle_nested_value(payload, "predicted_next_state", "scenario_type"),
        _bundle_nested_value(payload, "previous_state", "requested_scenario_type"),
        _bundle_nested_value(payload, "actual_next_state", "requested_scenario_type"),
        _bundle_nested_value(payload, "predicted_next_state", "requested_scenario_type"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return str(candidate).strip()
    return None


def _bundle_transition_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "worldIndex": _bundle_transition_world_index(payload),
        "mapName": _bundle_transition_map_name(payload),
    }


def _first_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    try:
        candidates = [
            path
            for path in directory.glob(pattern)
            if path.is_file()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.name.lower())
    return candidates[0]


def _load_patch_attempt_metadata(
    run_output_dir: Path,
    run_id: str,
) -> Dict[tuple[int, int, int], list[Dict[str, Any]]]:
    patch_attempts_dir = run_output_dir / "patch_attempts"
    if not patch_attempts_dir.exists() or not patch_attempts_dir.is_dir():
        return {}
    try:
        summary_paths = sorted(
            patch_attempts_dir.rglob("attempt_summary.json"),
            key=lambda path: str(path).lower(),
        )
    except OSError:
        return {}

    metadata_by_attempt: Dict[tuple[int, int, int], list[Dict[str, Any]]] = {}
    for summary_path in summary_paths:
        payload = _read_json(summary_path)
        try:
            iteration = int(payload.get("iteration") or 0)
            data_index = int(payload.get("data_index") or 0)
            failure_index = int(payload.get("failure_index") or 0)
            unexpected_attempt = int(payload.get("unexpected_attempt") or 1)
        except (TypeError, ValueError):
            continue
        if iteration <= 0 or data_index <= 0 or failure_index <= 0:
            continue
        current_version = _normalize_commit_version(payload.get("current_version"))
        attempt_key = (iteration, data_index, failure_index)
        generator_attempts = payload.get("generator_attempts")
        if not isinstance(generator_attempts, list):
            generator_attempts = []
        generator_attempt_entries: list[Dict[str, Any]] = []
        for generator_attempt in generator_attempts:
            if not isinstance(generator_attempt, dict):
                continue
            generator_attempt_entries.append(
                {
                    "generatorTry": int(generator_attempt.get("generator_try") or 0),
                    "totalAttempts": int(generator_attempt.get("total_attempts") or 0),
                    "status": str(generator_attempt.get("status") or "").strip() or None,
                    "prompt": _artifact_payload_for_relative_path(
                        run_output_dir,
                        run_id,
                        generator_attempt.get("prompt_path"),
                    ),
                    "output": _artifact_payload_for_relative_path(
                        run_output_dir,
                        run_id,
                        generator_attempt.get("output_path"),
                    ),
                    "reasoning": _artifact_payload_for_relative_path(
                        run_output_dir,
                        run_id,
                        generator_attempt.get("reasoning_path"),
                    ),
                    "error": _artifact_payload_for_relative_path(
                        run_output_dir,
                        run_id,
                        generator_attempt.get("error_path"),
                    ),
                }
            )
        generator_attempt_entries.sort(
            key=lambda entry: (
                int(entry.get("generatorTry") or 0),
                int(entry.get("totalAttempts") or 0),
            )
        )
        metadata_by_attempt.setdefault(attempt_key, []).append(
            {
                "unexpectedAttempt": max(1, unexpected_attempt),
                "currentVersion": current_version,
                "status": str(payload.get("status") or "").strip() or None,
                "errorMessage": str(payload.get("error_message") or "").strip() or None,
                "elapsedText": str(payload.get("elapsed_text") or "").strip() or None,
                "patchDigest": str(payload.get("patch_digest") or "").strip() or None,
                "accepted": bool(payload.get("accepted")),
                "summary": _artifact_payload_for_path(run_output_dir, run_id, summary_path),
                "appliedProgram": _artifact_payload_for_relative_path(
                    run_output_dir,
                    run_id,
                    payload.get("applied_program_path"),
                ),
                "generatorAttempts": generator_attempt_entries,
            }
        )
    for attempts in metadata_by_attempt.values():
        attempts.sort(
            key=lambda entry: (
                int(entry.get("unexpectedAttempt") or 0),
                str((entry.get("summary") or {}).get("path") or ""),
            )
        )
    return metadata_by_attempt


def _build_transition_review_payload(
    *,
    run_output_dir: Path,
    run_id: str,
    step_dir: Path,
    fail_metadata_by_attempt: Optional[Dict[tuple[int, int, int], list[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    step_index = _parse_prefixed_int(
        step_dir.name,
        pattern=_TRANSITION_STEP_DIR_RE,
        key="step",
    )
    if step_index is None:
        return None

    previous_image = _artifact_payload_for_path(
        run_output_dir,
        run_id,
        step_dir / "previous_state.png",
    )
    expected_image = _artifact_payload_for_path(
        run_output_dir,
        run_id,
        step_dir / "expected_next_state.png",
    )

    explained_bundle_path = step_dir / "bundle_explained.json"
    explained_bundle = _load_transition_bundle_metadata(explained_bundle_path)
    bundle_action = str(explained_bundle.get("action") or "").strip() or None
    previous_terminated = _bundle_transition_terminated(
        explained_bundle,
        primary_state_key="previous_state",
    )
    expected_terminated = _bundle_transition_terminated(
        explained_bundle,
        primary_state_key="actual_next_state",
    )
    explained_image_path = _first_matching_file(
        step_dir,
        "predicted_next_state_explained*.png",
    )
    explained_image = _artifact_payload_for_path(
        run_output_dir,
        run_id,
        explained_image_path,
    )
    explained_version = _normalize_commit_version(
        explained_image_path.stem if explained_image_path is not None else None
    )
    review_context = _bundle_transition_context(explained_bundle)
    explained_payload = None
    if explained_image is not None:
        explained_payload = {
            "versionTag": explained_version,
            "image": explained_image,
            "bundle": _artifact_payload_for_path(
                run_output_dir,
                run_id,
                explained_bundle_path,
            ),
            "stepIndex": int(explained_bundle.get("step_index") or step_index),
            "failureIndex": int(explained_bundle.get("failure_index") or 0),
            "terminated": _bundle_transition_terminated(
                explained_bundle,
                primary_state_key="predicted_next_state",
                fallback_state_key="actual_next_state",
            ),
            **review_context,
        }

    fail_payloads: list[Dict[str, Any]] = []
    try:
        fail_dirs = [
            path
            for path in step_dir.iterdir()
            if path.is_dir()
            and _parse_prefixed_int(path.name, pattern=_TRANSITION_FAIL_DIR_RE, key="fail") is not None
        ]
    except OSError:
        fail_dirs = []
    fail_dirs.sort(
        key=lambda path: _parse_prefixed_int(
            path.name,
            pattern=_TRANSITION_FAIL_DIR_RE,
            key="fail",
        ) or 0
    )
    for fail_dir in fail_dirs:
        failure_index = _parse_prefixed_int(
            fail_dir.name,
            pattern=_TRANSITION_FAIL_DIR_RE,
            key="fail",
        )
        if failure_index is None:
            continue
        bundle_path = fail_dir / "bundle_unexplained.json"
        bundle_payload = _load_transition_bundle_metadata(bundle_path)
        if bundle_action is None:
            bundle_action = str(bundle_payload.get("action") or "").strip() or None
        if previous_terminated is None:
            previous_terminated = _bundle_transition_terminated(
                bundle_payload,
                primary_state_key="previous_state",
            )
        if expected_terminated is None:
            expected_terminated = _bundle_transition_terminated(
                bundle_payload,
                primary_state_key="actual_next_state",
            )
        fail_context = _bundle_transition_context(bundle_payload)
        if review_context.get("worldIndex") is None and fail_context.get("worldIndex") is not None:
            review_context["worldIndex"] = fail_context.get("worldIndex")
        if review_context.get("mapName") is None and fail_context.get("mapName") is not None:
            review_context["mapName"] = fail_context.get("mapName")
        image_path = _first_matching_file(
            fail_dir,
            "predicted_next_state_unexplained*.png",
        )
        image_payload = _artifact_payload_for_path(
            run_output_dir,
            run_id,
            image_path,
        )
        resolved_failure_index = int(bundle_payload.get("failure_index") or failure_index)
        version_tag = _normalize_commit_version(
            image_path.stem if image_path is not None else None
        )
        attempt_metadata_rows: list[Dict[str, Any]] = []
        if isinstance(fail_metadata_by_attempt, dict):
            try:
                attempt_key = (
                    int(bundle_payload.get("iteration") or 0),
                    int(bundle_payload.get("data_index") or 0),
                    resolved_failure_index,
                )
            except (TypeError, ValueError):
                attempt_key = None
            if attempt_key is not None:
                attempt_metadata_rows = [
                    dict(item)
                    for item in (fail_metadata_by_attempt.get(attempt_key) or [])
                    if isinstance(item, dict)
                ]
        primary_attempt = attempt_metadata_rows[0] if attempt_metadata_rows else None
        if version_tag is None and isinstance(primary_attempt, dict):
            version_tag = _normalize_commit_version(primary_attempt.get("currentVersion"))
        fail_payloads.append(
            {
                "failureId": fail_dir.name.lower(),
                "failureIndex": resolved_failure_index,
                "versionTag": version_tag,
                "attempts": attempt_metadata_rows,
                "image": image_payload,
                "bundle": _artifact_payload_for_path(
                    run_output_dir,
                    run_id,
                    bundle_path,
                ),
                "stepIndex": int(bundle_payload.get("step_index") or step_index),
                "terminated": _bundle_transition_terminated(
                    bundle_payload,
                    primary_state_key="predicted_next_state",
                    fallback_state_key="actual_next_state",
                ),
                **fail_context,
            }
        )
    fail_payloads.sort(
        key=lambda item: (
            int(item.get("failureIndex") or 0),
            str(item.get("failureId") or ""),
        )
    )

    if previous_image is None and expected_image is None and explained_payload is None and not fail_payloads:
        return None

    review_id = step_dir.name.lower()
    return {
        "reviewId": review_id,
        "stepId": step_dir.name,
        "stepIndex": int(step_index),
        "action": bundle_action,
        "previousTerminated": previous_terminated,
        "expectedTerminated": expected_terminated,
        "previous": previous_image,
        "expected": expected_image,
        "explained": explained_payload,
        "fails": fail_payloads,
        "resolved": explained_payload is not None,
        "hasFails": bool(fail_payloads),
        **review_context,
    }


def _transition_gallery_file_signature(run_output_dir: Path) -> tuple[Any, ...]:
    return (
        _shallow_tree_digest(run_output_dir / "new_transition_images", max_depth=3),
        _shallow_tree_digest(run_output_dir / "patch_attempts", max_depth=2),
    )


def _build_transition_gallery_file_data_uncached(
    *,
    run_output_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    gallery_dir = run_output_dir / "new_transition_images"
    if not gallery_dir.exists() or not gallery_dir.is_dir():
        return {
            "revision": "missing",
            "versionEntries": [],
            "reviews": [],
            "latestFailReview": None,
        }
    fail_metadata_by_attempt = _load_patch_attempt_metadata(run_output_dir, run_id)

    try:
        step_dirs = [
            path
            for path in gallery_dir.iterdir()
            if path.is_dir()
            and _parse_prefixed_int(path.name, pattern=_TRANSITION_STEP_DIR_RE, key="step") is not None
        ]
    except OSError:
        return {
            "revision": "error",
            "versionEntries": [],
            "reviews": [],
            "latestFailReview": None,
        }

    reviews: list[Dict[str, Any]] = []
    reviews_by_id: Dict[str, Dict[str, Any]] = {}
    step_dirs.sort(
        key=lambda path: _parse_prefixed_int(
            path.name,
            pattern=_TRANSITION_STEP_DIR_RE,
            key="step",
        ) or 0
    )
    for step_dir in step_dirs:
        review_payload = _build_transition_review_payload(
            run_output_dir=run_output_dir,
            run_id=run_id,
            step_dir=step_dir,
            fail_metadata_by_attempt=fail_metadata_by_attempt,
        )
        if review_payload is None:
            continue
        reviews.append(review_payload)
        reviews_by_id[str(review_payload.get("reviewId"))] = review_payload
    if not reviews:
        return {
            "revision": _shallow_tree_digest(gallery_dir, max_depth=3),
            "versionEntries": [],
            "reviews": [],
            "latestFailReview": None,
        }

    version_entries_by_key: Dict[str, Dict[str, Any]] = {}

    def _ensure_version_entry(raw_commit_version: Any) -> Optional[Dict[str, Any]]:
        normalized_commit_version = _normalize_commit_version(raw_commit_version)
        if normalized_commit_version is None:
            return None
        existing = version_entries_by_key.get(normalized_commit_version)
        if existing is not None:
            return existing
        commit_sort_index, _ = _commit_version_sort_key(normalized_commit_version)
        entry = {
            "versionKey": normalized_commit_version,
            "label": _display_version_label(normalized_commit_version),
            "commitVersion": normalized_commit_version,
            "sortIndex": int(commit_sort_index),
            "isActive": False,
            "isCurrentProgram": False,
            "isCurrentExplainer": False,
            "isLatestActive": False,
            "isExplainer": False,
            "introReviewId": None,
            "failReviews": [],
            "defaultReview": None,
        }
        version_entries_by_key[normalized_commit_version] = entry
        return entry

    latest_fail_review = None
    for review in reviews:
        explained = review.get("explained") if isinstance(review.get("explained"), dict) else None
        explained_commit_version = _normalize_commit_version(
            explained.get("versionTag") if explained is not None else None
        )
        if explained_commit_version is not None:
            entry = _ensure_version_entry(explained_commit_version)
            if entry is not None:
                current_intro_review_id = entry.get("introReviewId")
                current_intro_review = (
                    reviews_by_id.get(str(current_intro_review_id))
                    if isinstance(current_intro_review_id, str)
                    else None
                )
                if (
                    current_intro_review is None
                    or int(review.get("stepIndex") or 0) < int(current_intro_review.get("stepIndex") or 0)
                ):
                    entry["introReviewId"] = review.get("reviewId")

        fails = review.get("fails")
        if not isinstance(fails, list):
            continue
        for fail_payload in fails:
            if not isinstance(fail_payload, dict):
                continue
            fail_ref = {
                "reviewId": review.get("reviewId"),
                "failureId": fail_payload.get("failureId"),
                "failureIndex": int(fail_payload.get("failureIndex") or 0),
                "stepIndex": int(review.get("stepIndex") or 0),
                "label": f"FAIL {int(fail_payload.get('failureIndex') or 0):03d}",
            }
            if latest_fail_review is None or (
                int(fail_ref.get("stepIndex") or 0),
                int(fail_ref.get("failureIndex") or 0),
                str(fail_ref.get("failureId") or ""),
            ) > (
                int(latest_fail_review.get("stepIndex") or 0),
                int(latest_fail_review.get("failureIndex") or 0),
                str(latest_fail_review.get("failureId") or ""),
            ):
                latest_fail_review = dict(fail_ref)
            fail_commit_version = _normalize_commit_version(fail_payload.get("versionTag"))
            if fail_commit_version is None:
                continue
            entry = _ensure_version_entry(fail_commit_version)
            if entry is None:
                continue
            fail_reviews = entry.setdefault("failReviews", [])
            if any(
                str(existing.get("reviewId")) == str(fail_ref.get("reviewId"))
                and str(existing.get("failureId")) == str(fail_ref.get("failureId"))
                for existing in fail_reviews
                if isinstance(existing, dict)
            ):
                continue
            fail_reviews.append(dict(fail_ref))

    version_entries = list(version_entries_by_key.values())
    for entry in version_entries:
        fail_reviews = [
            dict(item)
            for item in (entry.get("failReviews") or [])
            if isinstance(item, dict)
        ]
        fail_reviews.sort(
            key=lambda item: (
                int(item.get("stepIndex") or 0),
                int(item.get("failureIndex") or 0),
                str(item.get("failureId") or ""),
            ),
            reverse=True,
        )
        entry["failReviews"] = fail_reviews
        if fail_reviews:
            entry["defaultReview"] = dict(fail_reviews[0])
        elif isinstance(entry.get("introReviewId"), str) and str(entry.get("introReviewId")).strip():
            entry["defaultReview"] = {
                "reviewId": str(entry.get("introReviewId")),
                "failureId": None,
                "failureIndex": 0,
                "stepIndex": int(
                    (reviews_by_id.get(str(entry.get("introReviewId"))) or {}).get("stepIndex") or 0
                ),
                "label": "INTRO",
            }

    version_entries.sort(
        key=lambda entry: (
            int(entry.get("sortIndex") or 10**9),
            _commit_version_sort_key(entry.get("commitVersion"))[0],
            str(entry.get("versionKey") or ""),
        )
    )
    return {
        "revision": _shallow_tree_digest(gallery_dir, max_depth=3),
        "versionEntries": version_entries,
        "reviews": reviews,
        "latestFailReview": dict(latest_fail_review) if isinstance(latest_fail_review, dict) else None,
    }


def _cached_transition_gallery_file_data(
    *,
    run_output_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    cache_key = _cache_key_for_path(run_output_dir / "new_transition_images")
    signature = _transition_gallery_file_signature(run_output_dir)
    cached_value = _read_component_cache(
        _DISCOVERY_TRANSITION_GALLERY_FILE_CACHE,
        cache_key=cache_key,
        signature=signature,
    )
    if isinstance(cached_value, dict):
        return cached_value
    resolved = _build_transition_gallery_file_data_uncached(
        run_output_dir=run_output_dir,
        run_id=run_id,
    )
    return _write_component_cache(
        _DISCOVERY_TRANSITION_GALLERY_FILE_CACHE,
        cache_key=cache_key,
        signature=signature,
        value=resolved if isinstance(resolved, dict) else {},
    )


def _build_transition_gallery(
    *,
    run_output_dir: Path,
    run_id: str,
    dashboard_payload: Dict[str, Any],
) -> Dict[str, Any]:
    static_gallery = _cached_transition_gallery_file_data(
        run_output_dir=run_output_dir,
        run_id=run_id,
    )
    reviews = static_gallery.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return {
            "revision": str(static_gallery.get("revision") or "missing"),
            "versions": [],
            "reviews": [],
            "live": {},
        }

    version_entries = deepcopy(static_gallery.get("versionEntries") or [])
    version_entries_by_key: Dict[str, Dict[str, Any]] = {
        str(entry.get("versionKey")): entry
        for entry in version_entries
        if isinstance(entry, dict) and str(entry.get("versionKey") or "").strip()
    }

    def _ensure_version_entry(raw_commit_version: Any) -> Optional[Dict[str, Any]]:
        normalized_commit_version = _normalize_commit_version(raw_commit_version)
        if normalized_commit_version is None:
            return None
        existing = version_entries_by_key.get(normalized_commit_version)
        if existing is not None:
            return existing
        commit_sort_index, _ = _commit_version_sort_key(normalized_commit_version)
        entry = {
            "versionKey": normalized_commit_version,
            "label": _display_version_label(normalized_commit_version),
            "commitVersion": normalized_commit_version,
            "sortIndex": int(commit_sort_index),
            "isActive": False,
            "isCurrentProgram": False,
            "isCurrentExplainer": False,
            "isLatestActive": False,
            "isExplainer": False,
            "introReviewId": None,
            "failReviews": [],
            "defaultReview": None,
        }
        version_entries_by_key[normalized_commit_version] = entry
        version_entries.append(entry)
        return entry

    class_rows = dashboard_payload.get("classRows")
    if not isinstance(class_rows, list):
        class_rows = []
    for row in class_rows:
        if not isinstance(row, dict):
            continue
        commit_version = _normalize_commit_version(
            row.get("commit_version") or row.get("group_id") or row.get("version_id") or row.get("display_id")
        )
        entry = _ensure_version_entry(commit_version)
        if entry is None:
            continue
        entry["isActive"] = bool(row.get("is_active")) or bool(entry.get("isActive"))
        entry["isCurrentProgram"] = bool(row.get("is_current")) or bool(entry.get("isCurrentProgram"))
        entry["isCurrentExplainer"] = bool(row.get("explains_current_state")) or bool(entry.get("isCurrentExplainer"))
        entry["isLatestActive"] = bool(row.get("is_latest_active")) or bool(entry.get("isLatestActive"))
        entry["isExplainer"] = bool(row.get("is_explainer")) or bool(entry.get("isExplainer"))

    version_entries.sort(
        key=lambda entry: (
            int(entry.get("sortIndex") or 10**9),
            _commit_version_sort_key(entry.get("commitVersion"))[0],
            str(entry.get("versionKey") or ""),
        )
    )

    heatmap = dashboard_payload.get("visitationHeatmap")
    heatmap = heatmap if isinstance(heatmap, dict) else {}
    metrics_payload = dashboard_payload.get("metrics")
    metrics_payload = metrics_payload if isinstance(metrics_payload, dict) else {}
    current_group_key = _normalize_group_version_key(heatmap.get("current_group_id"))
    progress_phase = str(metrics_payload.get("progress_phase") or "").strip().lower() or None
    current_commit_version = _normalize_commit_version(
        heatmap.get("current_group_id") or heatmap.get("current_group_label")
    )
    current_patch_step_index = _coerce_positive_int(
        metrics_payload.get("current_patch_step_index")
    )
    current_patch_failure_index = _coerce_positive_int(
        metrics_payload.get("current_patch_failure_index")
    )
    current_patch_commit_version = _normalize_commit_version(
        metrics_payload.get("current_patch_current_version")
        or metrics_payload.get("current_patch_version")
        or metrics_payload.get("program_version")
    )
    explicit_patch_fail_review = None
    if (
        progress_phase == "patch"
        and current_patch_step_index is not None
        and current_patch_failure_index is not None
    ):
        for review in reviews:
            if int(review.get("stepIndex") or 0) != int(current_patch_step_index):
                continue
            for fail_payload in (review.get("fails") or []):
                if not isinstance(fail_payload, dict):
                    continue
                if int(fail_payload.get("failureIndex") or 0) != int(current_patch_failure_index):
                    continue
                explicit_patch_fail_review = {
                    "reviewId": review.get("reviewId"),
                    "failureId": fail_payload.get("failureId"),
                    "failureIndex": int(fail_payload.get("failureIndex") or 0),
                    "stepIndex": int(review.get("stepIndex") or 0),
                    "label": f"FAIL {int(fail_payload.get('failureIndex') or 0):03d}",
                }
                if current_patch_commit_version is None:
                    current_patch_commit_version = _normalize_commit_version(
                        fail_payload.get("versionTag")
                    )
                break
            if explicit_patch_fail_review is not None:
                break
    if progress_phase == "patch" and current_patch_commit_version is not None:
        current_commit_version = current_patch_commit_version
    current_row = None
    if current_group_key is not None:
        for row in class_rows:
            if not isinstance(row, dict):
                continue
            row_group_key = _normalize_group_version_key(
                row.get("group_id") or row.get("version_id") or row.get("display_id"),
                commit_version=row.get("commit_version"),
            )
            if row_group_key == current_group_key:
                current_row = row
                break
    if current_row is None:
        current_row = next(
            (
                row
                for row in class_rows
                if isinstance(row, dict)
                and (
                    bool(row.get("is_current"))
                    or bool(row.get("is_latest_active"))
                    or row.get("explains_current_state") is True
                )
            ),
            None,
        )
    if current_group_key is None and isinstance(current_row, dict):
        current_group_key = _normalize_group_version_key(
            current_row.get("group_id") or current_row.get("version_id") or current_row.get("display_id"),
            commit_version=current_row.get("commit_version"),
        )
    if current_commit_version is None and isinstance(current_row, dict):
        current_commit_version = _normalize_commit_version(
            current_row.get("commit_version")
            or current_row.get("group_id")
            or current_row.get("version_id")
            or current_row.get("display_id")
        )
    current_version_entry = None
    if current_version_entry is None and current_commit_version is not None:
        current_version_entry = version_entries_by_key.get(current_commit_version)
    if current_version_entry is None:
        current_program_entry = next(
            (entry for entry in version_entries if bool(entry.get("isCurrentProgram"))),
            None,
        )
        current_version_entry = current_program_entry
    current_class_index = None
    for raw_value in (
        heatmap.get("current_group_class_index"),
        heatmap.get("current_class_index"),
        metrics_payload.get("current_dynamics_class"),
        current_row.get("version_index") if isinstance(current_row, dict) else None,
    ):
        try:
            resolved_value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if resolved_value > 0:
            current_class_index = resolved_value
            break
    current_group_label = str(
        heatmap.get("current_group_label")
        or (
            current_row.get("display_id")
            if isinstance(current_row, dict)
            else None
        )
        or (
            current_row.get("version_id")
            if isinstance(current_row, dict)
            else None
        )
        or (
            current_row.get("group_id")
            if isinstance(current_row, dict)
            else None
        )
        or current_group_key
        or ""
    ).strip() or None

    live_default_review = None
    if current_version_entry is not None and isinstance(current_version_entry.get("defaultReview"), dict):
        live_default_review = dict(current_version_entry.get("defaultReview") or {})
    elif reviews:
        latest_review = max(reviews, key=lambda item: int(item.get("stepIndex") or 0))
        live_default_review = {
            "reviewId": str(latest_review.get("reviewId")),
            "failureId": None,
            "failureIndex": 0,
            "stepIndex": int(latest_review.get("stepIndex") or 0),
            "label": "LATEST",
            }

    class_row_live_fail = False
    if isinstance(class_rows, list):
        for row in class_rows:
            if not isinstance(row, dict):
                continue
            if bool(row.get("is_new_dynamics_class")):
                class_row_live_fail = True
                break
            if row.get("explains_current_state") is False:
                class_row_live_fail = True
                break
    # Collect-phase unknown/new-class signals are expected exploration artifacts,
    # not an active patch target for the CURRENT rail.
    is_live_fail = (
        progress_phase != "collect"
        and (
            heatmap.get("current_transition_explained_by_current_program") is False
            or bool(heatmap.get("current_is_new_dynamics_class"))
            or bool(class_row_live_fail)
        )
    )
    latest_fail_review = static_gallery.get("latestFailReview")
    if explicit_patch_fail_review is not None:
        is_live_fail = True
        live_default_review = dict(explicit_patch_fail_review)
    elif is_live_fail and isinstance(latest_fail_review, dict):
        live_default_review = dict(latest_fail_review)
    return {
        "revision": str(static_gallery.get("revision") or "missing"),
        "versions": version_entries,
        "reviews": reviews,
        "live": {
            "currentVersionKey": (
                str(current_version_entry.get("versionKey"))
                if isinstance(current_version_entry, dict)
                else current_commit_version
            ),
            "currentGroupKey": current_group_key,
            "currentGroupLabel": current_group_label,
            "currentClassIndex": current_class_index,
            "isFail": bool(is_live_fail),
            "defaultReview": live_default_review,
        },
    }


def _program_versions_payload_signature(
    run_output_dir: Path,
    transition_gallery: Dict[str, Any],
) -> tuple[Any, ...]:
    gallery_versions = transition_gallery.get("versions")
    if not isinstance(gallery_versions, list):
        gallery_versions = []
    gallery_reviews = transition_gallery.get("reviews")
    if not isinstance(gallery_reviews, list):
        gallery_reviews = []
    versions_signature = tuple(
        (
            str(entry.get("versionKey") or ""),
            bool(entry.get("isActive")),
            bool(entry.get("isCurrentProgram")),
            bool(entry.get("isCurrentExplainer")),
            bool(entry.get("isLatestActive")),
            bool(entry.get("isExplainer")),
            str(entry.get("introReviewId") or ""),
            str((entry.get("defaultReview") or {}).get("reviewId") or ""),
            str((entry.get("defaultReview") or {}).get("failureId") or ""),
            tuple(
                (
                    str(item.get("reviewId") or ""),
                    str(item.get("failureId") or ""),
                    int(item.get("failureIndex") or 0),
                    int(item.get("stepIndex") or 0),
                )
                for item in (entry.get("failReviews") or [])
                if isinstance(item, dict)
            ),
        )
        for entry in gallery_versions
        if isinstance(entry, dict)
    )
    reviews_signature = tuple(
        (
            str(entry.get("reviewId") or ""),
            int(entry.get("stepIndex") or 0),
        )
        for entry in gallery_reviews
        if isinstance(entry, dict)
    )
    return (
        _shallow_tree_digest(run_output_dir / "program_versions", max_depth=1),
        versions_signature,
        reviews_signature,
    )


def _build_program_versions_uncached(
    *,
    run_output_dir: Path,
    transition_gallery: Dict[str, Any],
) -> list[Dict[str, Any]]:
    program_versions_dir = run_output_dir / "program_versions"
    if not program_versions_dir.exists() or not program_versions_dir.is_dir():
        return []

    try:
        version_paths = [
            path
            for path in program_versions_dir.iterdir()
            if path.is_file()
            and _normalize_commit_version(path.stem) is not None
        ]
    except OSError:
        return []

    version_paths.sort(
        key=lambda path: _commit_version_sort_key(path.stem)[0],
    )

    gallery_versions = transition_gallery.get("versions")
    if not isinstance(gallery_versions, list):
        gallery_versions = []
    gallery_versions_by_key: Dict[str, Dict[str, Any]] = {
        str(entry.get("versionKey")): entry
        for entry in gallery_versions
        if isinstance(entry, dict) and str(entry.get("versionKey") or "").strip()
    }

    gallery_reviews = transition_gallery.get("reviews")
    if not isinstance(gallery_reviews, list):
        gallery_reviews = []
    gallery_reviews_by_id: Dict[str, Dict[str, Any]] = {
        str(entry.get("reviewId")): entry
        for entry in gallery_reviews
        if isinstance(entry, dict) and str(entry.get("reviewId") or "").strip()
    }

    program_versions_by_key: Dict[str, Dict[str, Any]] = {}
    for path in version_paths:
        version_key = _normalize_commit_version(path.stem)
        if version_key is None or version_key == "v000":
            continue
        commit_sort_index, _ = _commit_version_sort_key(version_key)
        try:
            version_source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            version_source = None
        entry = {
            "versionKey": version_key,
            "label": _display_version_label(version_key),
            "commitVersion": version_key,
            "sortIndex": int(commit_sort_index),
            "sourcePath": _relative_run_artifact_path(run_output_dir, path),
            "source": version_source if isinstance(version_source, str) and version_source.strip() else None,
            "sourceDigest": _text_sha1(version_source),
            "sourceLineCount": (
                len(version_source.splitlines())
                if isinstance(version_source, str)
                else 0
            ),
            "isActive": False,
            "isCurrentProgram": False,
            "isCurrentExplainer": False,
            "isLatestActive": False,
            "isExplainer": False,
            "introReviewId": None,
            "failReviews": [],
            "defaultReview": None,
        }
        overlay = gallery_versions_by_key.get(version_key)
        if isinstance(overlay, dict):
            entry["isActive"] = bool(overlay.get("isActive"))
            entry["isCurrentProgram"] = bool(overlay.get("isCurrentProgram"))
            entry["isCurrentExplainer"] = bool(overlay.get("isCurrentExplainer"))
            entry["isLatestActive"] = bool(overlay.get("isLatestActive"))
            entry["isExplainer"] = bool(overlay.get("isExplainer"))
            intro_review_id = str(overlay.get("introReviewId") or "").strip() or None
            if intro_review_id:
                entry["introReviewId"] = intro_review_id
            overlay_default_review = overlay.get("defaultReview")
            if isinstance(overlay_default_review, dict):
                entry["defaultReview"] = deepcopy(overlay_default_review)
            overlay_fail_reviews = overlay.get("failReviews")
            if isinstance(overlay_fail_reviews, list):
                entry["failReviews"] = [
                    deepcopy(item)
                    for item in overlay_fail_reviews
                    if isinstance(item, dict)
                ]
        program_versions_by_key[version_key] = entry

    for review in gallery_reviews:
        if not isinstance(review, dict):
            continue
        explained = review.get("explained") if isinstance(review.get("explained"), dict) else None
        explained_commit_version = _normalize_commit_version(
            explained.get("versionTag") if explained is not None else None
        )
        if explained_commit_version is not None:
            entry = program_versions_by_key.get(explained_commit_version)
            if entry is not None:
                current_intro_review_id = entry.get("introReviewId")
                current_intro_review = (
                    gallery_reviews_by_id.get(str(current_intro_review_id))
                    if isinstance(current_intro_review_id, str)
                    else None
                )
                review_step_index = int(review.get("stepIndex") or 0)
                if (
                    current_intro_review is None
                    or review_step_index < int(current_intro_review.get("stepIndex") or 0)
                ):
                    entry["introReviewId"] = str(review.get("reviewId"))

        fails = review.get("fails")
        if not isinstance(fails, list):
            continue
        for fail_payload in fails:
            if not isinstance(fail_payload, dict):
                continue
            fail_commit_version = _normalize_commit_version(fail_payload.get("versionTag"))
            if fail_commit_version is None:
                continue
            entry = program_versions_by_key.get(fail_commit_version)
            if entry is None:
                continue
            fail_ref = {
                "reviewId": review.get("reviewId"),
                "failureId": fail_payload.get("failureId"),
                "failureIndex": int(fail_payload.get("failureIndex") or 0),
                "stepIndex": int(review.get("stepIndex") or 0),
                "label": f"FAIL {int(fail_payload.get('failureIndex') or 0):03d}",
            }
            fail_reviews = entry.setdefault("failReviews", [])
            if any(
                str(existing.get("reviewId")) == str(fail_ref.get("reviewId"))
                and str(existing.get("failureId")) == str(fail_ref.get("failureId"))
                for existing in fail_reviews
                if isinstance(existing, dict)
            ):
                continue
            fail_reviews.append(fail_ref)

    program_versions = list(program_versions_by_key.values())
    for entry in program_versions:
        fail_reviews = [
            dict(item)
            for item in (entry.get("failReviews") or [])
            if isinstance(item, dict)
        ]
        fail_reviews.sort(
            key=lambda item: (
                int(item.get("stepIndex") or 0),
                int(item.get("failureIndex") or 0),
                str(item.get("failureId") or ""),
            ),
            reverse=True,
        )
        entry["failReviews"] = fail_reviews
        if fail_reviews:
            entry["defaultReview"] = dict(fail_reviews[0])
        elif isinstance(entry.get("introReviewId"), str) and str(entry.get("introReviewId")).strip():
            intro_review_id = str(entry.get("introReviewId")).strip()
            entry["defaultReview"] = {
                "reviewId": intro_review_id,
                "failureId": None,
                "failureIndex": 0,
                "stepIndex": int(
                    (gallery_reviews_by_id.get(intro_review_id) or {}).get("stepIndex") or 0
                ),
                "label": "INTRO",
            }

    program_versions.sort(
        key=lambda entry: (
            int(entry.get("sortIndex") or 10**9),
            _commit_version_sort_key(entry.get("commitVersion"))[0],
            str(entry.get("versionKey") or ""),
        )
    )
    return program_versions


def _build_program_versions(
    *,
    run_output_dir: Path,
    transition_gallery: Dict[str, Any],
) -> list[Dict[str, Any]]:
    program_versions_dir = run_output_dir / "program_versions"
    cache_key = _cache_key_for_path(program_versions_dir)
    signature = _program_versions_payload_signature(run_output_dir, transition_gallery)
    cached_value = _read_component_cache(
        _DISCOVERY_PROGRAM_VERSIONS_CACHE,
        cache_key=cache_key,
        signature=signature,
    )
    if isinstance(cached_value, list):
        return cached_value
    resolved = _build_program_versions_uncached(
        run_output_dir=run_output_dir,
        transition_gallery=transition_gallery,
    )
    return _write_component_cache(
        _DISCOVERY_PROGRAM_VERSIONS_CACHE,
        cache_key=cache_key,
        signature=signature,
        value=resolved,
    )


def _load_config_snapshot_files(run_output_dir: Path) -> list[Dict[str, str]]:
    snapshot_dir = run_output_dir / "config_snapshot"
    if not snapshot_dir.exists():
        return []
    preferred_order = {
        "experiment_config.yaml": 0,
        "llm_config.yaml": 1,
        "env_config.yaml": 2,
        "agent_config.yaml": 3,
    }
    files: list[Path] = []
    try:
        files = [
            path
            for path in snapshot_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        ]
    except OSError:
        return []
    files.sort(key=lambda path: (preferred_order.get(path.name, 99), path.name.lower()))
    snapshots: list[Dict[str, str]] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        snapshots.append(
            {
                "name": path.name,
                "label": path.stem.replace("_", " "),
                "content": content,
            }
        )
    return snapshots


def build_dashboard_url(*, host: str, port: int, run_id: Optional[str] = None) -> str:
    browser_host = host
    if host in {"0.0.0.0", "::"}:
        browser_host = "127.0.0.1"
    base = f"http://{browser_host}:{int(port)}/"
    if isinstance(run_id, str) and run_id.strip():
        return f"{base}?run_id={run_id.strip()}"
    return base


def _can_connect(*, host: str, port: int, timeout_sec: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _can_bind_dashboard_port(*, host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind((str(host), int(port)))
        return True
    except OSError:
        return False


def _find_available_dashboard_port(*, host: str, preferred_port: int) -> int:
    start_port = max(1, int(preferred_port))
    end_port = start_port + int(DISCOVERY_DASHBOARD_PORT_SEARCH_WINDOW)
    for candidate in range(start_port, end_port):
        if _can_bind_dashboard_port(host=host, port=candidate):
            return candidate
    raise RuntimeError(
        f"Could not find a free discovery dashboard port in [{start_port}, {end_port - 1}]."
    )


def _discovery_server_healthcheck(*, host: str, port: int, timeout_sec: float = 0.8) -> bool:
    base_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{base_host}:{int(port)}/api/health"
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            if int(getattr(response, "status", 0) or 0) != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        bool(payload.get("ok"))
        and str(payload.get("api") or "").strip() == "discovery"
        and int(payload.get("version") or 0) == DISCOVERY_DASHBOARD_HEALTHCHECK_VERSION
        and str(payload.get("projectRoot") or "").strip() == str(PROJECT_ROOT.resolve())
    )


def _has_live_discovery_runs(*, ignore_run_id: Optional[str] = None) -> bool:
    ignored = str(ignore_run_id or "").strip()
    for run_dir in _scan_discovery_metadata_dirs():
        state = _read_json(run_dir / "state.json")
        if not state:
            continue
        run_id = str(state.get("runId") or "").strip()
        if ignored and run_id == ignored:
            continue
        status = _resolve_run_status(state)
        if str(status.get("effectiveStatus") or "").strip() == "running":
            return True
    return False


def _terminate_stale_discovery_dashboard_server(*, port: int) -> bool:
    if os.name != "nt":
        return False
    script_path = str((PROJECT_ROOT / "scripts" / "run_discovery_dashboard.py").resolve()).replace("'", "''")
    command = (
        "$target = Get-CimInstance Win32_Process | "
        "Where-Object { "
        "$_.CommandLine -and "
        f"$_.CommandLine -like '*{script_path}*' -and "
        f"$_.CommandLine -match '--port\\s+{int(port)}(\\s|$)' "
        "}; "
        "if ($target) { $target | ForEach-Object { Stop-Process -Id $_.ProcessId -Force } ; exit 0 } "
        "else { exit 1 }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def can_shutdown_discovery_dashboard_server() -> bool:
    return not _has_live_discovery_runs()


def _sleep_then_exit_current_process(
    *,
    delay_sec: float,
    exit_code: int = 0,
    sleep_func: Callable[[float], None] = time.sleep,
    exit_func: Callable[[int], None] = os._exit,
) -> None:
    resolved_delay_sec = max(0.0, float(delay_sec))
    if resolved_delay_sec > 0.0:
        sleep_func(resolved_delay_sec)
    exit_func(int(exit_code))


def request_discovery_dashboard_server_shutdown(
    *,
    delay_sec: float = 0.25,
    exit_code: int = 0,
) -> threading.Thread:
    # The dashboard server runs detached from discovery jobs, so shutting it down
    # requires terminating the current process after the HTTP response is flushed.
    thread = threading.Thread(
        target=_sleep_then_exit_current_process,
        name="discovery-dashboard-shutdown",
        kwargs={
            "delay_sec": delay_sec,
            "exit_code": exit_code,
        },
        daemon=True,
    )
    thread.start()
    return thread


def ensure_discovery_dashboard_server(
    *,
    host: str,
    port: int,
    open_browser: bool,
    run_id: Optional[str] = None,
) -> str:
    requested_port = int(port)
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    resolved_port = requested_port
    server_reachable = _can_connect(host=probe_host, port=resolved_port)
    if server_reachable:
        server_healthy = _discovery_server_healthcheck(host=probe_host, port=resolved_port)
        if not server_healthy:
            terminated = _terminate_stale_discovery_dashboard_server(port=resolved_port)
            if terminated:
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if not _can_connect(host=probe_host, port=resolved_port, timeout_sec=0.2):
                        break
                    time.sleep(0.15)
                server_reachable = _can_connect(host=probe_host, port=resolved_port)
            else:
                resolved_port = _find_available_dashboard_port(
                    host=probe_host,
                    preferred_port=requested_port + 1,
                )
                server_reachable = False

    if not server_reachable and int(resolved_port) == int(requested_port):
        resolved_port = _find_available_dashboard_port(
            host=probe_host,
            preferred_port=resolved_port,
        )
    if not server_reachable:
        attempted_ports: list[int] = []
        booted = False
        last_process_returncode: Optional[int] = None
        for _attempt in range(max(1, int(DISCOVERY_DASHBOARD_START_PORT_ATTEMPTS))):
            attempted_ports.append(int(resolved_port))
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_discovery_dashboard.py"),
                "--host",
                str(host),
                "--port",
                str(int(resolved_port)),
                "--no-browser",
            ]
            detached = getattr(subprocess, "DETACHED_PROCESS", 0)
            new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags = detached | new_group
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
            deadline = time.time() + float(DISCOVERY_DASHBOARD_BOOT_TIMEOUT_SEC)
            while time.time() < deadline:
                if _discovery_server_healthcheck(
                    host=probe_host,
                    port=int(resolved_port),
                    timeout_sec=0.3,
                ):
                    booted = True
                    break
                poll = process.poll()
                if poll is not None:
                    last_process_returncode = int(poll)
                    break
                time.sleep(0.15)
            if booted:
                break
            poll = process.poll()
            if poll is None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            elif last_process_returncode is None:
                last_process_returncode = int(poll)
            try:
                resolved_port = _find_available_dashboard_port(
                    host=probe_host,
                    preferred_port=int(resolved_port) + 1,
                )
            except RuntimeError:
                break
        if not booted:
            attempted_text = ", ".join(str(port) for port in attempted_ports)
            returncode_text = (
                f" last_returncode={last_process_returncode}"
                if last_process_returncode is not None
                else ""
            )
            raise RuntimeError(
                "Failed to start discovery dashboard server on "
                f"{probe_host}. attempted_ports=[{attempted_text}]."
                f"{returncode_text}"
            )

    url = build_dashboard_url(host=host, port=int(resolved_port), run_id=run_id)
    if open_browser:
        webbrowser.open(url, new=2)
    return url


class TeeTextIO(io.TextIOBase):
    def __init__(self, *targets: Any):
        self._targets = [target for target in targets if target is not None]

    def _iter_open_targets(self):
        for target in self._targets:
            if bool(getattr(target, "closed", False)):
                continue
            yield target

    @property
    def encoding(self) -> str:
        for target in self._iter_open_targets():
            value = getattr(target, "encoding", None)
            if isinstance(value, str) and value:
                return value
        return "utf-8"

    def write(self, text: str) -> int:
        wrote_any = False
        for target in self._iter_open_targets():
            try:
                write_fn = getattr(target, "write", None)
                if not callable(write_fn):
                    continue
                write_fn(text)
                flush_fn = getattr(target, "flush", None)
                if callable(flush_fn) and not bool(
                    getattr(target, "defer_flush_after_write", False)
                ):
                    flush_fn()
                wrote_any = True
            except ValueError:
                continue
        return len(text) if wrote_any else 0

    def flush(self) -> None:
        for target in self._iter_open_targets():
            try:
                flush_fn = getattr(target, "flush", None)
                if callable(flush_fn):
                    flush_fn()
            except ValueError:
                continue

    def isatty(self) -> bool:
        for target in self._iter_open_targets():
            try:
                if bool(getattr(target, "isatty", lambda: False)()):
                    return True
            except ValueError:
                continue
        return False


class _BufferedLogTarget:
    defer_flush_after_write = True

    def __init__(
        self,
        target: io.TextIOBase,
        *,
        flush_interval_provider: Callable[[], float],
    ) -> None:
        self._target = target
        self._flush_interval_provider = flush_interval_provider
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._buffer: list[str] = []
        self._closed = False
        self._last_flush_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="discovery-log-buffer",
            daemon=True,
        )
        self._thread.start()

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    @property
    def encoding(self) -> str:
        value = getattr(self._target, "encoding", None)
        return value if isinstance(value, str) and value else "utf-8"

    def write(self, text: str) -> int:
        resolved_text = str(text)
        if not resolved_text:
            return 0
        with self._lock:
            if self._closed:
                raise ValueError("I/O operation on closed log target.")
            self._buffer.append(resolved_text)
        self._wake_event.set()
        return len(resolved_text)

    def flush(self) -> None:
        self._flush_due(force=False)

    def force_flush(self) -> None:
        self._flush_due(force=True)

    def close(self) -> None:
        self.force_flush()
        with self._lock:
            self._closed = True
        self._wake_event.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def _flush_interval_sec(self) -> float:
        try:
            return max(0.0, float(self._flush_interval_provider()))
        except (TypeError, ValueError, RuntimeError, OSError):
            return 60.0

    def _flush_due(self, *, force: bool) -> bool:
        now = time.monotonic()
        interval_sec = self._flush_interval_sec()
        with self._lock:
            if not force and now - float(self._last_flush_at) < interval_sec:
                return False
            chunks = list(self._buffer)
            self._buffer.clear()
            self._last_flush_at = now
        if not chunks and not force:
            return False
        try:
            if chunks:
                self._target.write("".join(chunks))
            self._target.flush()
            return True
        except (OSError, ValueError):
            return False

    def _worker_loop(self) -> None:
        while True:
            interval_sec = self._flush_interval_sec()
            wait_sec = min(max(interval_sec, 0.1), 1.0)
            self._wake_event.wait(timeout=wait_sec)
            self._wake_event.clear()
            self._flush_due(force=False)
            with self._lock:
                if self._closed and not self._buffer:
                    return


class _AsyncLatestJsonWriter:
    """Latest-only JSON writer for non-blocking UI updates."""

    def __init__(
        self,
        *,
        name: str,
        write_callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        self._name = str(name or "discovery-json-writer")
        self._write_callback = write_callback
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._latest_seq = 0
        self._written_seq = 0

    def _ensure_thread_locked(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._stop_requested = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=self._name,
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._ensure_thread_locked()
            self._latest_payload = payload
            self._latest_seq += 1
            self._wake_event.set()
            self._idle.notify_all()

    def flush(self, *, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._lock:
            target_seq = int(self._latest_seq)
            if target_seq <= int(self._written_seq):
                return True
            self._ensure_thread_locked()
            self._wake_event.set()
            while int(self._written_seq) < target_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._idle.wait(timeout=remaining)
            return True

    def close(self, *, timeout_sec: float = 5.0) -> bool:
        flushed = self.flush(timeout_sec=timeout_sec)
        with self._lock:
            self._stop_requested = True
            self._wake_event.set()
            self._idle.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_sec)))
        return bool(flushed)

    def _worker_loop(self) -> None:
        while True:
            self._wake_event.wait()
            while True:
                with self._lock:
                    stop_requested = bool(self._stop_requested)
                    pending_seq = int(self._latest_seq)
                    written_seq = int(self._written_seq)
                    payload = self._latest_payload if pending_seq > written_seq else None
                    if payload is None:
                        self._wake_event.clear()
                        self._idle.notify_all()
                        if stop_requested:
                            return
                        break
                    seq = pending_seq
                try:
                    self._write_callback(payload)
                except Exception:
                    pass
                with self._lock:
                    self._written_seq = max(int(self._written_seq), int(seq))
                    if int(self._written_seq) >= int(self._latest_seq):
                        self._latest_payload = None
                        self._wake_event.clear()
                    self._idle.notify_all()
                    if bool(self._stop_requested) and int(self._written_seq) >= int(self._latest_seq):
                        return


class DiscoveryRunPublisher:
    def __init__(
        self,
        *,
        run_output_dir: Path,
        config_path: str,
        llm_config_path: str,
        llm_config_payload: Optional[Dict[str, Any]],
        env_config_path: str,
        agent_config_path: Optional[str],
        max_iterations: Optional[int],
        host: str,
        port: int,
        open_browser: bool,
    ):
        self.run_id = uuid.uuid4().hex
        self.host = str(host)
        self.port = int(port)
        self._browser_requested = bool(open_browser)
        self._run_output_dir = Path(run_output_dir)
        self._run_dir = _run_metadata_dir(self._run_output_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._viewer_state_cache_at = 0.0
        self._viewer_state_cache_enabled = True
        self._live_projection_cache_at = 0.0
        self._live_projection_cache_enabled = False
        self._live_projection_sparse_filter_enabled = False
        self._effective_viewer_mode: Optional[str] = None
        self._effective_projection_mode: Optional[str] = None
        self._bound_explorer: Optional[Any] = None
        self._bound_visualizer: Optional[Any] = None
        self._final_dashboard_preparer: Optional[Callable[[], None]] = None
        self._active_log_target: Optional[_BufferedLogTarget] = None
        self._dashboard_snapshot_writer = _AsyncLatestJsonWriter(
            name=f"discovery-dashboard-writer-{self.run_id[:8]}",
            write_callback=self._write_dashboard_snapshot,
        )
        self._state_writer = _AsyncLatestJsonWriter(
            name=f"discovery-state-writer-{self.run_id[:8]}",
            write_callback=self._write_state_payload,
        )
        llm_config_summary = _summarize_llm_config_payload(llm_config_payload or {})
        self._state: Dict[str, Any] = {
            "runId": self.run_id,
            "configPath": str(config_path),
            "llmConfigPath": str(llm_config_path),
            "llmConfigSummary": llm_config_summary,
            "envConfigPath": str(env_config_path),
            "agentConfigPath": agent_config_path,
            "maxIterations": max_iterations,
            "status": "queued",
            "createdAt": _now_iso(),
            "startedAt": None,
            "finishedAt": None,
            "lastHeartbeatAt": None,
            "lastStatusUpdateAt": None,
            "heartbeatIntervalSec": DISCOVERY_RUN_HEARTBEAT_INTERVAL_SEC,
            "heartbeatTimeoutSec": _heartbeat_timeout_for_interval(
                DISCOVERY_RUN_HEARTBEAT_INTERVAL_SEC
            ),
            "statusUpdateStaleSec": DISCOVERY_RUN_STATUS_UPDATE_STALE_SEC,
            "pid": None,
            "error": None,
            "runOutputDir": str(self._run_output_dir),
            "dashboardUrl": build_dashboard_url(host=self.host, port=self.port, run_id=self.run_id),
        }
        _write_json_atomic(self.state_path, self._state)
        _invalidate_discovery_scan_cache()

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def state_path(self) -> Path:
        return self._run_dir / "state.json"

    @property
    def dashboard_path(self) -> Path:
        return self._run_dir / "dashboard.json"

    @property
    def log_path(self) -> Path:
        return self._run_dir / "log.txt"

    @property
    def dashboard_url(self) -> str:
        return str(
            self._state.get("dashboardUrl")
            or build_dashboard_url(host=self.host, port=self.port, run_id=self.run_id)
        )

    @property
    def viewer_state_path(self) -> Path:
        return _viewer_state_path(self._run_dir)

    def ensure_dashboard_server(self) -> str:
        url = ensure_discovery_dashboard_server(
            host=self.host,
            port=self.port,
            open_browser=self._browser_requested,
            run_id=self.run_id,
        )
        self._browser_requested = False
        with self._lock:
            self._state["dashboardUrl"] = url
            self._write_state_locked()
        return url

    def _write_state_locked(self, *, required: bool = True) -> bool:
        self._state_writer.submit(dict(self._state))
        if required:
            return self._state_writer.flush(timeout_sec=5.0)
        return True

    def _write_state_payload(self, state: Dict[str, Any]) -> None:
        _write_json_atomic(self.state_path, state)

    def _write_dashboard_snapshot(self, snapshot: Dict[str, Any]) -> None:
        _write_json_atomic(self.dashboard_path, snapshot)

    def flush_dashboard_snapshot(self, *, timeout_sec: float = 5.0) -> bool:
        return self._dashboard_snapshot_writer.flush(timeout_sec=timeout_sec)

    def flush_log(self) -> None:
        log_target = self._active_log_target
        if log_target is None:
            return
        try:
            log_target.force_flush()
        except (OSError, ValueError, RuntimeError):
            return

    def _close_dashboard_snapshot_writer(self, *, timeout_sec: float = 5.0) -> bool:
        return self._dashboard_snapshot_writer.close(timeout_sec=timeout_sec)

    def flush_state(self, *, timeout_sec: float = 5.0) -> bool:
        return self._state_writer.flush(timeout_sec=timeout_sec)

    def _close_state_writer(self, *, timeout_sec: float = 5.0) -> bool:
        return self._state_writer.close(timeout_sec=timeout_sec)

    def mark_status_update(self, *, force: bool = False) -> None:
        with self._lock:
            if not force:
                age_sec = _seconds_since(self._state.get("lastStatusUpdateAt"))
                if age_sec is not None and age_sec < DISCOVERY_RUN_STATUS_UPDATE_TOUCH_INTERVAL_SEC:
                    return
            self._state["lastStatusUpdateAt"] = _now_iso()
            self._write_state_locked(required=False)

    def _current_heartbeat_interval_sec(self) -> float:
        return (
            float(DISCOVERY_RUN_HEARTBEAT_INTERVAL_SEC)
            if bool(self._viewer_state_cache_enabled)
            else float(DISCOVERY_RUN_PAUSED_HEARTBEAT_INTERVAL_SEC)
        )

    def _publish_heartbeat(self) -> None:
        self._refresh_live_delivery_modes(emit_snapshot_on_live_resume=True)
        heartbeat_interval_sec = self._current_heartbeat_interval_sec()
        with self._lock:
            if self._state.get("status") != "running":
                return
            self._state["lastHeartbeatAt"] = _now_iso()
            self._state["heartbeatIntervalSec"] = float(heartbeat_interval_sec)
            self._state["heartbeatTimeoutSec"] = _heartbeat_timeout_for_interval(
                heartbeat_interval_sec
            )
            self._write_state_locked(required=False)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._current_heartbeat_interval_sec()):
            try:
                self._publish_heartbeat()
            except Exception:
                pass

    def _ensure_heartbeat_thread(self) -> None:
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"discovery-heartbeat-{self.run_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=0.2)

    def mark_running(self) -> None:
        with self._lock:
            self._state["status"] = "running"
            self._state["startedAt"] = _now_iso()
            self._state["finishedAt"] = None
            self._state["error"] = None
            self._state["pid"] = os.getpid()
            self._state["lastHeartbeatAt"] = _now_iso()
            self._write_state_locked()
        self._ensure_heartbeat_thread()

    def publish_dashboard_snapshot(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.mark_status_update()
        snapshot = dict(payload)
        snapshot["runId"] = self.run_id
        snapshot["updatedAt"] = _now_iso()
        self._dashboard_snapshot_writer.submit(snapshot)

    def _should_publish_snapshots(self) -> bool:
        now = time.monotonic()
        if now - self._viewer_state_cache_at <= DISCOVERY_VIEWER_STATE_CACHE_TTL_SEC:
            return bool(self._viewer_state_cache_enabled)
        snapshot_enabled, _projection_enabled = self._refresh_live_delivery_modes(
            now=now
        )
        return bool(snapshot_enabled)

    def _should_compute_live_projection(self) -> bool:
        now = time.monotonic()
        if now - self._live_projection_cache_at <= DISCOVERY_VIEWER_STATE_CACHE_TTL_SEC:
            return bool(self._live_projection_cache_enabled)
        _snapshot_enabled, projection_enabled = self._refresh_live_delivery_modes(
            now=now
        )
        return bool(projection_enabled)

    def _should_filter_live_projection_sparse_classes(self) -> bool:
        now = time.monotonic()
        if now - self._live_projection_cache_at > DISCOVERY_VIEWER_STATE_CACHE_TTL_SEC:
            self._refresh_live_delivery_modes(now=now)
        return bool(self._live_projection_sparse_filter_enabled)

    def _log_flush_interval_sec(self) -> float:
        return 0.1 if self._should_publish_snapshots() else 60.0

    def _refresh_live_delivery_modes(
        self,
        *,
        now: Optional[float] = None,
        emit_snapshot_on_live_resume: bool = False,
    ) -> tuple[bool, bool]:
        previous_mode = self._effective_viewer_mode
        resolved_mode = _resolve_effective_viewer_mode(self._run_dir)
        resolved_projection_mode = _resolve_effective_projection_mode(self._run_dir)
        resolved_sparse_filter = _resolve_effective_projection_sparse_class_filter(
            self._run_dir
        )
        snapshot_enabled = bool(resolved_mode == "live")
        projection_enabled = bool(
            snapshot_enabled and resolved_projection_mode != "paused"
        )
        cache_now = time.monotonic() if now is None else float(now)
        self._effective_viewer_mode = resolved_mode
        self._effective_projection_mode = resolved_projection_mode
        self._viewer_state_cache_at = cache_now
        self._viewer_state_cache_enabled = bool(snapshot_enabled)
        self._live_projection_cache_at = cache_now
        self._live_projection_cache_enabled = bool(projection_enabled)
        self._live_projection_sparse_filter_enabled = bool(resolved_sparse_filter)
        if (
            emit_snapshot_on_live_resume
            and resolved_mode == "live"
            and previous_mode != "live"
        ):
            visualizer = self._bound_visualizer
            snapshot_emitter = getattr(visualizer, "emit_snapshot", None)
            if callable(snapshot_emitter):
                try:
                    snapshot_emitter()
                except Exception:
                    pass
        return bool(snapshot_enabled), bool(projection_enabled)

    def _publish_final_dashboard_snapshot(self) -> None:
        finalizer = self._final_dashboard_preparer
        finalizer_failed = False
        if callable(finalizer):
            try:
                finalizer()
            except Exception:
                finalizer_failed = True
        if not callable(finalizer) or finalizer_failed:
            explorer = self._bound_explorer
            finalizer = getattr(explorer, "prepare_final_dashboard_snapshot", None)
            if callable(finalizer):
                try:
                    finalizer()
                except Exception:
                    pass
        self.flush_log()
        visualizer = self._bound_visualizer
        snapshot_builder = getattr(visualizer, "build_web_snapshot", None)
        if not callable(snapshot_builder):
            return
        try:
            snapshot = snapshot_builder()
        except Exception:
            return
        if isinstance(snapshot, dict):
            self.publish_dashboard_snapshot(snapshot)

    def mark_completed(self, result: Dict[str, Any]) -> None:
        self._stop_heartbeat_thread()
        try:
            self.flush_log()
            self._publish_final_dashboard_snapshot()
            self.flush_dashboard_snapshot()
        finally:
            try:
                with self._lock:
                    self._state["status"] = "completed"
                    self._state["finishedAt"] = _now_iso()
                    self._state["lastHeartbeatAt"] = _now_iso()
                    self._state["lastStatusUpdateAt"] = _now_iso()
                    run_output_dir = result.get("run_output_dir")
                    if isinstance(run_output_dir, str) and run_output_dir.strip():
                        self._state["runOutputDir"] = run_output_dir.strip()
                    self._write_state_locked(required=False)
            finally:
                self._close_state_writer()
                self._close_dashboard_snapshot_writer()
                _invalidate_discovery_scan_cache()

    def mark_failed(self, exc: BaseException) -> None:
        self._stop_heartbeat_thread()
        try:
            self.flush_log()
            self._publish_final_dashboard_snapshot()
            self.flush_dashboard_snapshot()
        finally:
            try:
                with self._lock:
                    self._state["status"] = "failed"
                    self._state["finishedAt"] = _now_iso()
                    self._state["lastHeartbeatAt"] = _now_iso()
                    self._state["lastStatusUpdateAt"] = _now_iso()
                    self._state["error"] = str(exc)
                    self._write_state_locked(required=False)
            finally:
                self._close_state_writer()
                self._close_dashboard_snapshot_writer()
                _invalidate_discovery_scan_cache()

    def bind_visualizer(self, explorer: Any) -> None:
        visualizer = getattr(explorer, "visualizer", None)
        if visualizer is None:
            return
        self._bound_explorer = explorer
        self._bound_visualizer = visualizer
        diagnostics_setter = getattr(visualizer, "set_agent_diagnostics_provider", None)
        diagnostics_getter = getattr(explorer, "get_diagnostics", None)
        if callable(diagnostics_setter):
            diagnostics_setter(diagnostics_getter if callable(diagnostics_getter) else None)
        setter = getattr(visualizer, "set_snapshot_callback", None)
        if callable(setter):
            setter(self.publish_dashboard_snapshot)
        progress_setter = getattr(visualizer, "set_progress_callback", None)
        if callable(progress_setter):
            progress_setter(self.mark_status_update)
            self.mark_status_update(force=True)
        snapshot_enabled_setter = getattr(visualizer, "set_snapshot_enabled_provider", None)
        if callable(snapshot_enabled_setter):
            snapshot_enabled_setter(self._should_publish_snapshots)
        projection_enabled_setter = getattr(
            visualizer,
            "set_visitation_heatmap_enabled_provider",
            None,
        )
        if callable(projection_enabled_setter):
            projection_enabled_setter(self._should_compute_live_projection)
        projection_filter_setter = getattr(
            visualizer,
            "set_visitation_heatmap_sparse_class_filter_provider",
            None,
        )
        if callable(projection_filter_setter):
            projection_filter_setter(self._should_filter_live_projection_sparse_classes)
        self._refresh_live_delivery_modes(emit_snapshot_on_live_resume=False)

    def bind_final_dashboard_preparer(self, preparer: Any) -> None:
        self._final_dashboard_preparer = preparer if callable(preparer) else None

    @contextmanager
    def tee_streams(self) -> Iterator[None]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_handle:
            log_target = _BufferedLogTarget(
                log_handle,
                flush_interval_provider=self._log_flush_interval_sec,
            )
            self._active_log_target = log_target
            stdout_tee = TeeTextIO(sys.stdout, log_target)
            stderr_tee = TeeTextIO(sys.stderr, log_target)
            try:
                with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                    yield
            finally:
                log_target.force_flush()
                log_target.close()
                if self._active_log_target is log_target:
                    self._active_log_target = None


def build_discovery_run_payload(run_dir: Path) -> Dict[str, Any]:
    state = _read_json(run_dir / "state.json")
    if not state:
        return {}
    run_output_dir_path = _persisted_run_output_dir(run_dir, state)
    if run_output_dir_path is None or _path_is_missing(run_output_dir_path):
        return {}
    dashboard_payload = _read_json(run_dir / "dashboard.json")
    payload = dict(state)
    payload["runOutputDir"] = str(run_output_dir_path)
    payload.update(_resolve_run_status(state))
    payload["logTail"] = _cached_tail_text(run_dir / "log.txt")
    resolved_dashboard = dashboard_payload if isinstance(dashboard_payload, dict) else {}
    if not _path_exists(run_output_dir_path):
        payload["summary"] = {}
        payload["dashboard"] = resolved_dashboard
        payload["transitionGallery"] = {}
        payload["programVersions"] = []
        return payload

    summary = dict(summarize_run_output(run_output_dir_path))
    try:
        llm_config_summary = _resolve_llm_config_summary(state, run_output_dir_path)
    except OSError:
        llm_config_summary = None
    if llm_config_summary:
        summary["llmConfigSummary"] = llm_config_summary
    llm_runtime_summary = _resolve_llm_runtime_summary(summary)
    if llm_runtime_summary:
        summary["llmRuntimeSummary"] = llm_runtime_summary
    payload["summary"] = summary
    resolved_dashboard = _ensure_dashboard_visual_config(run_output_dir_path, resolved_dashboard)
    resolved_dashboard = _ensure_dashboard_program_version_sources(run_output_dir_path, resolved_dashboard)
    payload["dashboard"] = resolved_dashboard
    is_live_run = bool(payload.get("isLive"))
    dashboard_transition_gallery = (
        resolved_dashboard.get("transitionGallery")
        if (
            not is_live_run
            and isinstance(resolved_dashboard.get("transitionGallery"), dict)
        )
        else None
    )
    if dashboard_transition_gallery is not None:
        payload["transitionGallery"] = dashboard_transition_gallery
    else:
        try:
            payload["transitionGallery"] = _build_transition_gallery(
                run_output_dir=run_output_dir_path,
                run_id=str(state.get("runId") or ""),
                dashboard_payload=payload["dashboard"] if isinstance(payload["dashboard"], dict) else {},
            )
        except OSError:
            payload["transitionGallery"] = {}
    dashboard_program_versions = (
        resolved_dashboard.get("programVersions")
        if not is_live_run
        else None
    )
    if isinstance(dashboard_program_versions, list):
        payload["programVersions"] = dashboard_program_versions
    else:
        try:
            payload["programVersions"] = _build_program_versions(
                run_output_dir=run_output_dir_path,
                transition_gallery=payload["transitionGallery"] if isinstance(payload["transitionGallery"], dict) else {},
            )
        except OSError:
            payload["programVersions"] = []
    return _safe_json_value(payload)


def build_discovery_run_index_payload(run_dir: Path) -> Dict[str, Any]:
    state = _read_json(run_dir / "state.json")
    if not state:
        return {}
    run_output_dir_path = _persisted_run_output_dir(run_dir, state)
    if run_output_dir_path is None or _path_is_missing(run_output_dir_path):
        return {}
    payload = {
        "runId": state.get("runId"),
        "createdAt": state.get("createdAt"),
        "startedAt": state.get("startedAt"),
        "finishedAt": state.get("finishedAt"),
        "runOutputDir": str(run_output_dir_path),
        "dashboardUrl": state.get("dashboardUrl"),
        "error": state.get("error"),
    }
    payload.update(_resolve_run_status(state))
    return _safe_json_value(payload)


def list_discovery_run_payloads() -> list[Dict[str, Any]]:
    payloads: list[Dict[str, Any]] = []
    for run_dir in _scan_discovery_metadata_dirs():
        payload = build_discovery_run_payload(run_dir)
        if payload:
            payloads.append(payload)
    payloads.sort(
        key=lambda item: str(item.get("startedAt") or item.get("createdAt") or ""),
        reverse=True,
    )
    return payloads


def list_discovery_run_index_payloads() -> list[Dict[str, Any]]:
    payloads: list[Dict[str, Any]] = []
    for run_dir in _scan_discovery_metadata_dirs():
        payload = build_discovery_run_index_payload(run_dir)
        if payload:
            payloads.append(payload)
    payloads.sort(
        key=lambda item: str(item.get("startedAt") or item.get("createdAt") or ""),
        reverse=True,
    )
    return payloads


def _find_discovery_run_dir(run_id: str) -> Optional[Path]:
    normalized = str(run_id or "").strip()
    if not normalized:
        return None
    for run_dir in _scan_discovery_metadata_dirs():
        state = _read_json(run_dir / "state.json")
        if str(state.get("runId") or "").strip() == normalized:
            return run_dir
    return None


def load_discovery_run_payload(run_id: str) -> Optional[Dict[str, Any]]:
    run_dir = _find_discovery_run_dir(run_id)
    if run_dir is None:
        return None
    payload = build_discovery_run_payload(run_dir)
    return payload or None


def delete_discovery_run(
    run_id: str,
    *,
    confirm_text: str,
) -> Dict[str, Any]:
    if str(confirm_text) != DISCOVERY_RUN_DELETE_CONFIRMATION_TEXT:
        raise DiscoveryRunDeleteError(
            "Deletion confirmation text did not match exactly.",
            status_code=409,
        )

    run_dir = _find_discovery_run_dir(run_id)
    if run_dir is None:
        raise DiscoveryRunDeleteError(
            "Unknown discovery run.",
            status_code=404,
        )
    payload = build_discovery_run_payload(run_dir)
    if not payload:
        raise DiscoveryRunDeleteError(
            "Unknown discovery run.",
            status_code=404,
        )
    if bool(payload.get("isLive")):
        raise DiscoveryRunDeleteError(
            "Live discovery runs cannot be deleted. Wait for the run to finish first.",
            status_code=409,
        )

    run_output_dir_text = str(payload.get("runOutputDir") or "").strip()
    if not run_output_dir_text:
        raise DiscoveryRunDeleteError(
            "Unknown discovery run.",
            status_code=404,
        )
    run_output_dir = Path(run_output_dir_text).resolve()
    if not run_output_dir.exists():
        raise DiscoveryRunDeleteError(
            "Unknown discovery run.",
            status_code=404,
        )
    if not any(
        _path_is_within_root(run_output_dir, root)
        for root in _allowed_discovery_run_output_roots()
    ):
        raise DiscoveryRunDeleteError(
            "Refusing to delete a discovery run outside the configured experiment roots.",
            status_code=409,
        )

    deleted_directories: list[str] = []
    deleted = _delete_tree_within_root(
        run_output_dir.parent,
        run_output_dir,
        deleted_paths=deleted_directories,
    )
    if not deleted:
        raise DiscoveryRunDeleteError(
            "Failed to delete the discovery run directory.",
            status_code=409,
        )

    _invalidate_discovery_scan_cache()
    return {
        "ok": True,
        "deleted": {
            "runId": str(payload.get("runId") or run_id),
            "runOutputDir": str(run_output_dir),
            "deletedDirectories": deleted_directories,
        },
    }


def resolve_run_artifact_path(run_id: str, artifact_path: str) -> Optional[Path]:
    run_dir = _find_discovery_run_dir(run_id)
    if run_dir is None:
        return None
    state = _read_json(run_dir / "state.json")
    if not state:
        return None
    run_output_dir = _persisted_run_output_dir(run_dir, state)
    if run_output_dir is None or not run_output_dir.exists():
        return None
    normalized_artifact_path = str(artifact_path or "").strip().replace("\\", "/")
    if not normalized_artifact_path:
        return None
    candidate_relative_path = Path(normalized_artifact_path)
    if candidate_relative_path.is_absolute():
        return None
    try:
        resolved_path = (run_output_dir / candidate_relative_path).resolve()
        resolved_path.relative_to(run_output_dir.resolve())
    except (OSError, ValueError):
        return None
    if not resolved_path.exists() or not resolved_path.is_file():
        return None
    return resolved_path


def update_discovery_viewer_state(
    run_id: str,
    *,
    client_id: str,
    view_mode: str,
    projection_mode: str = "live",
    projection_exclude_sparse_classes: bool = False,
) -> bool:
    run_dir = _find_discovery_run_dir(run_id)
    normalized_client_id = str(client_id or "").strip()
    normalized_view_mode = _normalize_client_mode(view_mode)
    normalized_projection_mode = _normalize_client_mode(projection_mode)
    if run_dir is None or not normalized_client_id:
        return False
    clients = _prune_stale_viewer_clients(_load_viewer_clients(run_dir))
    clients[normalized_client_id] = {
        "clientId": normalized_client_id,
        "viewMode": normalized_view_mode,
        "projectionMode": normalized_projection_mode,
        "projectionExcludeSparseClasses": bool(projection_exclude_sparse_classes),
        "lastSeenAt": _now_iso(),
    }
    _write_viewer_clients(run_dir, clients)
    return True
