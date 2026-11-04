from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOTENV_LOADED_PATHS: set[str] = set()


def load_llm_credentials(
    *,
    llm_config_path: Path,
    project_root: Path,
    configured_path: Optional[str] = None,
) -> Dict[str, Any]:
    _auto_load_dotenv(llm_config_path=llm_config_path, project_root=project_root)

    candidate_paths = []
    normalized = (configured_path or "").strip()
    if normalized:
        configured = Path(normalized)
        if configured.is_absolute():
            candidate_paths.append(configured)
        else:
            candidate_paths.append((llm_config_path.parent / configured).resolve())
            candidate_paths.append((project_root / configured).resolve())
    candidate_paths.append((project_root / "configs" / "llm_credentials.yaml").resolve())

    seen = set()
    for path in candidate_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_file():
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(payload, dict):
                raise ValueError(f"Expected mapping in credentials file: {path}")
            return payload
    return {}


def resolve_api_key_from_sources(
    *,
    inference_cfg: Dict[str, Any],
    credentials_payload: Dict[str, Any],
    direct_field: str,
    env_field: str,
    default_env_name: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    direct = inference_cfg.get(direct_field)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    from_credentials = _lookup_credential(credentials_payload, direct_field)
    if isinstance(from_credentials, str) and from_credentials.strip():
        return from_credentials.strip()

    env_name = inference_cfg.get(env_field, default_env_name)
    if isinstance(env_name, str) and env_name.strip():
        env_value = os.getenv(env_name.strip())
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip()

    return fallback


def _lookup_credential(payload: Dict[str, Any], field_name: str) -> Optional[str]:
    direct = payload.get(field_name)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    for section_name in ("api_keys", "credentials"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            value = section.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _auto_load_dotenv(*, llm_config_path: Path, project_root: Path) -> None:
    candidate_paths = []
    for base in (project_root, project_root.parent):
        candidate_paths.append((base / ".env").resolve())
        candidate_paths.append((base / ".env.local").resolve())
    candidate_paths.append((llm_config_path.parent / ".env").resolve())

    seen: set[str] = set()
    for path in candidate_paths:
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        _load_dotenv_file(path)


def _load_dotenv_file(path: Path) -> None:
    path_key = str(path.resolve())
    if path_key in _DOTENV_LOADED_PATHS:
        return
    _DOTENV_LOADED_PATHS.add(path_key)

    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ:
            continue
        os.environ[key] = value


def _parse_dotenv_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    if "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    env_key = key.strip()
    if not _ENV_KEY_PATTERN.match(env_key):
        return None

    value = raw_value.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return env_key, bytes(value[1:-1], encoding="utf-8").decode("unicode_escape")
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return env_key, value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return env_key, value
