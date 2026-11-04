from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"base_config", "extends"}:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str | Path, *, _seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    resolved_path = Path(path).resolve()
    seen = set(_seen or set())
    if resolved_path in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved_path])
        raise ValueError(f"Cyclic config inheritance detected: {chain}")
    seen.add(resolved_path)
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")

    base_value = payload.get("base_config", payload.get("extends"))
    if not base_value:
        return payload
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"`base_config` must be a non-empty string in {path}")

    base_path = Path(base_value.strip())
    if not base_path.is_absolute():
        base_path = (resolved_path.parent / base_path).resolve()
    base_payload = _load_yaml(base_path, _seen=seen)
    return _deep_merge_dicts(base_payload, payload)


def load_baba_config_bundle(
    *,
    env_config_path: str | Path,
    experiment_config_path: str | Path,
) -> Dict[str, Any]:
    env_payload = _load_yaml(env_config_path)
    experiment_payload = _load_yaml(experiment_config_path)

    environment = env_payload.get("environment") or {}
    serialization = env_payload.get("serialization") or {}
    experiment_serialization = experiment_payload.get("serialization") or {}

    return {
        "env_name": str(environment.get("name", "env/baba_in_wonderland_baseline_easy")),
        "max_steps": int(environment.get("max_steps", 100)),
        "render_mode": environment.get("render_mode"),
        "env_kwargs": dict(environment.get("kwargs") or {}),
        "state_format": str(serialization.get("format", "json")),
        "word_aliases": dict(experiment_serialization.get("word_aliases") or {}),
    }
