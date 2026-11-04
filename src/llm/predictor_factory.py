from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .credentials import load_llm_credentials, resolve_api_key_from_sources
from .inference import GenerationConfig, LLMPredictor


_SUPPORTED_PROVIDERS = {"openai", "gemini"}

_SUPPORTED_INFERENCE_KEYS = {
    "provider",
    "credentials_path",
    "max_tokens",
    "stop_sequences",
    "request_max_retries",
    "request_retry_initial_delay_sec",
    "request_retry_max_delay_sec",
    "max_total_llm_calls",
    "program_patch_unexpected_error_max_attempts",
    "reasoning_effort",
    "openai_base_url",
    "openai_api_key",
    "openai_api_key_env",
    "openai_model",
    "openai_reasoning_effort",
    "openai_timeout_sec",
    "gemini_base_url",
    "gemini_api_key",
    "gemini_api_key_env",
    "gemini_model",
    "gemini_timeout_sec",
    "gemini_include_thoughts",
    "gemini_thinking_budget",
    "gemini_thinking_level",
}


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


def load_llm_config(
    llm_config_path: str | Path,
    *,
    _seen: Optional[set[Path]] = None,
) -> Dict[str, Any]:
    path = Path(llm_config_path).resolve()
    seen = set(_seen or set())
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Cyclic LLM config inheritance detected: {chain}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")

    base_value = payload.get("base_config", payload.get("extends"))
    if not base_value:
        return payload
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"`base_config` must be a non-empty string in {path}")

    base_path = Path(base_value.strip())
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    base_payload = load_llm_config(base_path, _seen=seen)
    return _deep_merge_dicts(base_payload, payload)


def build_generation_config(
    *,
    llm_config_path: str | Path,
    project_root: str | Path,
    llm_config_payload: Dict[str, Any] | None = None,
) -> GenerationConfig:
    payload = dict(llm_config_payload or load_llm_config(llm_config_path))
    inference_cfg = payload.get("inference") or {}
    if not isinstance(inference_cfg, dict):
        raise ValueError("llm config must contain an `inference` mapping.")

    unsupported_keys = sorted(
        str(key) for key in inference_cfg if key not in _SUPPORTED_INFERENCE_KEYS
    )
    if unsupported_keys:
        configured = ", ".join(unsupported_keys)
        raise ValueError(f"Unsupported inference config keys: {configured}.")

    provider = str(inference_cfg.get("provider", "openai") or "").strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        allowed = "openai, gemini"
        raise ValueError(f"inference.provider must be one of: {allowed}.")

    resolved_llm_config_path = Path(llm_config_path)
    resolved_project_root = Path(project_root)
    credentials_payload = load_llm_credentials(
        llm_config_path=resolved_llm_config_path,
        project_root=resolved_project_root,
        configured_path=inference_cfg.get("credentials_path"),
    )

    openai_reasoning_effort = inference_cfg.get("openai_reasoning_effort")
    if openai_reasoning_effort is None:
        openai_reasoning_effort = inference_cfg.get("reasoning_effort")

    return GenerationConfig(
        provider=provider,
        max_tokens=int(inference_cfg.get("max_tokens", 512)),
        stop_sequences=inference_cfg.get("stop_sequences"),
        reasoning_effort=openai_reasoning_effort,
        openai_reasoning_effort=openai_reasoning_effort,
        openai_base_url=inference_cfg.get(
            "openai_base_url",
            "https://api.openai.com/v1",
        ),
        openai_api_key=resolve_api_key_from_sources(
            inference_cfg=inference_cfg,
            credentials_payload=credentials_payload,
            direct_field="openai_api_key",
            env_field="openai_api_key_env",
            default_env_name="OPENAI_API_KEY",
        ),
        openai_model=inference_cfg.get("openai_model"),
        openai_timeout_sec=float(inference_cfg.get("openai_timeout_sec", 120.0)),
        gemini_base_url=inference_cfg.get(
            "gemini_base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        gemini_api_key=resolve_api_key_from_sources(
            inference_cfg=inference_cfg,
            credentials_payload=credentials_payload,
            direct_field="gemini_api_key",
            env_field="gemini_api_key_env",
            default_env_name="GEMINI_API_KEY",
        ),
        gemini_model=inference_cfg.get("gemini_model"),
        gemini_timeout_sec=float(inference_cfg.get("gemini_timeout_sec", 120.0)),
        gemini_include_thoughts=inference_cfg.get("gemini_include_thoughts"),
        gemini_thinking_budget=inference_cfg.get("gemini_thinking_budget"),
        gemini_thinking_level=inference_cfg.get("gemini_thinking_level"),
        request_max_retries=int(inference_cfg.get("request_max_retries", 3)),
        request_retry_initial_delay_sec=float(
            inference_cfg.get("request_retry_initial_delay_sec", 1.0)
        ),
        request_retry_max_delay_sec=float(
            inference_cfg.get("request_retry_max_delay_sec", 20.0)
        ),
    )


def build_predictor_from_config(
    *,
    llm_config_path: str | Path,
    project_root: str | Path,
    llm_config_payload: Dict[str, Any] | None = None,
) -> LLMPredictor:
    return LLMPredictor(
        build_generation_config(
            llm_config_path=llm_config_path,
            project_root=project_root,
            llm_config_payload=llm_config_payload,
        )
    )
