"""
LLM inference client for program patch generation.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import re
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request


_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_*.-][A-Za-z0-9_*\-.*]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{8,}"),
    re.compile(r"(?i)(key=)[^&\s\"']+"),
)


def _redact_sensitive_text(value: Any) -> str:
    text = str(value)
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.pattern.startswith("(?i)(key=)"):
            text = pattern.sub(r"\1<redacted>", text)
        else:
            text = pattern.sub("<redacted>", text)
    return text


@dataclass
class GenerationConfig:
    provider: str = "openai"
    max_tokens: int = 512
    stop_sequences: Optional[List[str]] = None
    reasoning_effort: Optional[str] = None
    openai_reasoning_effort: Optional[str] = None

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_timeout_sec: float = 120.0

    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_timeout_sec: float = 120.0
    gemini_include_thoughts: Optional[bool] = None
    gemini_thinking_budget: Optional[int] = None
    gemini_thinking_level: Optional[str] = None

    request_max_retries: int = 3
    request_retry_initial_delay_sec: float = 1.0
    request_retry_max_delay_sec: float = 20.0


@dataclass
class GenerationUsage:
    provider: str
    model: str
    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMPredictor:
    """Text generator for the OpenAI and Gemini APIs."""

    _SUPPORTED_PROVIDERS = {"openai", "gemini"}
    _progress_log_interval_sec = 10.0

    def __init__(self, generation_config: Optional[GenerationConfig] = None):
        self.config = generation_config or GenerationConfig()
        self.last_usage: Optional[GenerationUsage] = None
        self._validate_generation_config()

    def generate_text(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Generate model response for a user prompt."""
        self.last_usage = None
        provider = str(self.config.provider or "").strip().lower()
        with self._request_progress_logger(provider):
            if provider == "openai":
                content, reasoning, usage = self._generate_via_openai(prompt)
                self.last_usage = usage
                return content, reasoning
            if provider == "gemini":
                content, reasoning, usage = self._generate_via_gemini(prompt)
                self.last_usage = usage
                return content, reasoning
        raise RuntimeError(f"Unsupported provider: {provider}")

    def _resolve_active_model_name(self, provider: str) -> str:
        provider_name = str(provider or "").strip().lower()
        if provider_name == "openai":
            return str(self.config.openai_model or "").strip()
        if provider_name == "gemini":
            return str(self.config.gemini_model or "").strip()
        return ""

    def _format_request_progress_message(
        self,
        *,
        status: str,
        provider: str,
        model: str,
        elapsed_sec: float,
    ) -> str:
        provider_text = str(provider or "").strip() or "unknown"
        message = f"[LLM] {status} | provider={provider_text}"
        if str(model or "").strip():
            message += f" | model={str(model).strip()}"
        message += f" | elapsed={float(elapsed_sec):.1f}s"
        return message

    @contextmanager
    def _request_progress_logger(self, provider: str):
        interval_sec = float(getattr(self, "_progress_log_interval_sec", 0.0) or 0.0)
        if interval_sec <= 0:
            yield
            return

        provider_name = str(provider or self.config.provider or "").strip() or "unknown"
        model_name = self._resolve_active_model_name(provider_name)
        started_at = time.monotonic()
        stop_event = threading.Event()
        state = {"heartbeat_count": 0, "max_message_width": 0}

        def _emit(status: str, *, transient: bool = False) -> None:
            message = self._format_request_progress_message(
                status=status,
                provider=provider_name,
                model=model_name,
                elapsed_sec=time.monotonic() - started_at,
            )
            width = max(int(state["max_message_width"]), len(message))
            state["max_message_width"] = width
            print(
                message.ljust(width),
                end="\r" if transient else "\n",
                flush=True,
            )

        def _heartbeat_loop() -> None:
            next_emit_at = started_at + interval_sec
            while True:
                wait_sec = max(0.0, next_emit_at - time.monotonic())
                if stop_event.wait(wait_sec):
                    return
                state["heartbeat_count"] += 1
                _emit("Request in progress", transient=True)
                next_emit_at += interval_sec

        thread = threading.Thread(
            target=_heartbeat_loop,
            name="llm-request-progress",
            daemon=True,
        )
        thread.start()
        try:
            yield
        except Exception:
            elapsed_sec = time.monotonic() - started_at
            if state["heartbeat_count"] > 0 or elapsed_sec >= interval_sec:
                _emit("Request failed")
            raise
        else:
            elapsed_sec = time.monotonic() - started_at
            if state["heartbeat_count"] > 0 or elapsed_sec >= interval_sec:
                _emit("Request completed")
        finally:
            stop_event.set()
            thread.join(timeout=0.1)

    def describe_runtime_status(self) -> Dict[str, str]:
        provider = str(self.config.provider or "").strip().lower() or "unknown"
        status = {
            "provider": provider,
            "transport": "api",
            "state": "ready",
            "auth_status": "api_key",
        }
        model = self._resolve_active_model_name(provider)
        if model:
            status["model"] = model
        runtime_effort = self._resolve_runtime_effort(provider)
        if runtime_effort:
            status.update(runtime_effort)
        return status

    def _validate_generation_config(self) -> None:
        provider = str(self.config.provider or "").strip().lower()
        self.config.provider = provider
        if provider not in self._SUPPORTED_PROVIDERS:
            allowed = "openai, gemini"
            raise ValueError(f"inference.provider must be one of: {allowed}.")

        if provider == "openai":
            if self.config.openai_timeout_sec <= 0:
                raise ValueError("openai_timeout_sec must be > 0.")
            if not self._is_nonempty(self.config.openai_base_url):
                raise ValueError("openai_base_url is required.")
            if not self._is_nonempty(self.config.openai_api_key):
                raise ValueError("openai_api_key is required for provider=openai.")
            if not self._is_nonempty(self.config.openai_model):
                raise ValueError("openai_model is required for provider=openai.")
        elif provider == "gemini":
            if self.config.gemini_timeout_sec <= 0:
                raise ValueError("gemini_timeout_sec must be > 0.")
            if not self._is_nonempty(self.config.gemini_base_url):
                raise ValueError("gemini_base_url is required.")
            if not self._is_nonempty(self.config.gemini_api_key):
                raise ValueError("gemini_api_key is required for provider=gemini.")
            if not self._is_nonempty(self.config.gemini_model):
                raise ValueError("gemini_model is required for provider=gemini.")
        self._validate_retry_config()

    def _validate_retry_config(self) -> None:
        if self.config.request_max_retries < 0:
            raise ValueError("request_max_retries must be >= 0.")
        if self.config.request_retry_initial_delay_sec < 0:
            raise ValueError("request_retry_initial_delay_sec must be >= 0.")
        if self.config.request_retry_max_delay_sec < 0:
            raise ValueError("request_retry_max_delay_sec must be >= 0.")

    def _generate_via_openai(
        self, prompt: str
    ) -> Tuple[str, Optional[str], Optional[GenerationUsage]]:
        return self._generate_via_chat_completions(
            base_url=str(self.config.openai_base_url),
            api_key=str(self.config.openai_api_key),
            model=str(self.config.openai_model),
            timeout_sec=float(self.config.openai_timeout_sec),
            provider_name="OpenAI",
            prompt=prompt,
            token_param_candidates=["max_completion_tokens", "max_tokens"],
        )

    def _generate_via_chat_completions(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float,
        provider_name: str,
        prompt: str,
        token_param_candidates: List[str],
    ) -> Tuple[str, Optional[str], Optional[GenerationUsage]]:
        base = base_url.rstrip("/")
        url = f"{base}/chat/completions"
        base_payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        data: Optional[Dict[str, Any]] = None
        last_error: Optional[RuntimeError] = None
        for idx, token_param in enumerate(token_param_candidates or ["max_tokens"]):
            include_stop = bool(self.config.stop_sequences)
            resolved_reasoning_effort = self._resolved_openai_reasoning_effort()
            include_reasoning_effort = self._is_nonempty(resolved_reasoning_effort)

            while True:
                payload = dict(base_payload)
                if include_stop and self.config.stop_sequences:
                    payload["stop"] = self.config.stop_sequences
                if include_reasoning_effort and resolved_reasoning_effort:
                    payload["reasoning_effort"] = resolved_reasoning_effort
                payload[token_param] = self.config.max_tokens

                req = urllib_request.Request(
                    url=url,
                    data=json.dumps(payload).encode("utf-8"),
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                try:
                    data = self._post_json(
                        req=req,
                        timeout_sec=timeout_sec,
                        provider_name=provider_name,
                    )
                    break
                except RuntimeError as exc:
                    last_error = exc
                    error_text = str(exc)
                    if include_stop and self._is_param_not_accepted_error(
                        error_text=error_text,
                        param_name="stop",
                    ):
                        include_stop = False
                        continue
                    if include_reasoning_effort and self._is_param_not_accepted_error(
                        error_text=error_text,
                        param_name="reasoning_effort",
                    ):
                        include_reasoning_effort = False
                        continue

                    has_next = idx < (len(token_param_candidates) - 1)
                    if has_next and self._is_unsupported_parameter_error(
                        error_text=error_text,
                        param_name=token_param,
                    ):
                        break
                    raise

            if data is not None:
                break

        if data is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"{provider_name} request failed before receiving a response.")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{provider_name} response has no choices.")

        message = choices[0].get("message") or {}
        content = self._normalize_chat_content(message.get("content"))
        reasoning = message.get("reasoning")
        if reasoning is None:
            reasoning = message.get("reasoning_content")
        reasoning_text = self._normalize_chat_content(reasoning)
        usage_payload = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = GenerationUsage(
            provider=str(self.config.provider),
            model=model,
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
            reasoning_tokens=self._extract_reasoning_tokens(usage_payload),
            completion_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_payload.get("total_tokens", 0) or 0),
            raw_usage=dict(usage_payload) if isinstance(usage_payload, dict) else None,
        )
        return content.strip(), (reasoning_text.strip() if reasoning_text else None), usage

    def _generate_via_gemini(
        self, prompt: str
    ) -> Tuple[str, Optional[str], Optional[GenerationUsage]]:
        base = str(self.config.gemini_base_url).rstrip("/")
        model = str(self.config.gemini_model).strip()
        if model.startswith("models/"):
            model = model[len("models/") :]

        api_key = str(self.config.gemini_api_key)
        url = f"{base}/models/{model}:generateContent?key={api_key}"
        include_thinking_config = self._should_include_gemini_thinking_config()
        data: Optional[Dict[str, Any]] = None
        last_error: Optional[RuntimeError] = None
        while True:
            generation_config: Dict[str, Any] = {
                "maxOutputTokens": self.config.max_tokens,
            }
            if self.config.stop_sequences:
                generation_config["stopSequences"] = self.config.stop_sequences
            if include_thinking_config:
                thinking_config = self._build_gemini_thinking_config()
                if thinking_config:
                    generation_config["thinkingConfig"] = thinking_config

            payload: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": generation_config,
            }
            req = urllib_request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                data = self._post_json(
                    req=req,
                    timeout_sec=float(self.config.gemini_timeout_sec),
                    provider_name="Gemini",
                )
                break
            except RuntimeError as exc:
                last_error = exc
                if include_thinking_config and self._is_gemini_thinking_config_error(str(exc)):
                    include_thinking_config = False
                    continue
                raise

        if data is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Gemini request failed before receiving a response.")

        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback") or {}
            blocked_reason = prompt_feedback.get("blockReason")
            if blocked_reason:
                raise RuntimeError(f"Gemini blocked request: {blocked_reason}")
            raise RuntimeError("Gemini response has no candidates.")

        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content_obj = first.get("content") or {}
        parts = content_obj.get("parts") or []
        answer_texts: List[str] = []
        thought_texts: List[str] = []
        for part in parts:
            normalized_text = None
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                normalized_text = part["text"]
            elif part is not None:
                normalized_text = self._normalize_chat_content(part)
            if not normalized_text:
                continue
            if isinstance(part, dict) and bool(part.get("thought")):
                thought_texts.append(normalized_text)
            else:
                answer_texts.append(normalized_text)

        content = "".join(answer_texts).strip()
        reasoning_text = "".join(thought_texts).strip()
        if not content:
            finish_reason = first.get("finishReason")
            if finish_reason:
                raise RuntimeError(f"Gemini returned empty content (finishReason={finish_reason}).")
            raise RuntimeError("Gemini returned empty content.")

        usage_payload = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        usage = GenerationUsage(
            provider=str(self.config.provider),
            model=model,
            prompt_tokens=int(usage_payload.get("promptTokenCount", 0) or 0),
            reasoning_tokens=self._extract_reasoning_tokens(usage_payload),
            completion_tokens=int(usage_payload.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage_payload.get("totalTokenCount", 0) or 0),
            raw_usage=dict(usage_payload) if isinstance(usage_payload, dict) else None,
        )
        return content, (reasoning_text or None), usage

    def _should_include_gemini_thinking_config(self) -> bool:
        return any(
            value is not None
            for value in (
                self.config.gemini_include_thoughts,
                self.config.gemini_thinking_budget,
                self.config.gemini_thinking_level,
            )
        )

    def _resolved_openai_reasoning_effort(self) -> Optional[str]:
        if self._is_nonempty(self.config.openai_reasoning_effort):
            return self.config.openai_reasoning_effort
        if self._is_nonempty(self.config.reasoning_effort):
            return self.config.reasoning_effort
        return None

    def _resolve_runtime_effort(self, provider: str) -> Optional[Dict[str, str]]:
        provider_name = str(provider or "").strip().lower()
        if provider_name == "openai":
            effort = self._resolved_openai_reasoning_effort()
            return {"reasoning_effort": effort} if effort else None
        if provider_name == "gemini":
            level = self._resolved_gemini_reasoning_level()
            return {"thinking_level": level} if level else None
        return None

    def _resolved_gemini_reasoning_level(self) -> Optional[str]:
        if self.config.gemini_thinking_level is not None and self._is_nonempty(
            self.config.gemini_thinking_level
        ):
            return str(self.config.gemini_thinking_level).strip().upper()
        if self.config.gemini_thinking_budget is not None:
            return f"budget={int(self.config.gemini_thinking_budget)}"
        return None

    def _build_gemini_thinking_config(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.config.gemini_include_thoughts is not None:
            payload["includeThoughts"] = bool(self.config.gemini_include_thoughts)
        if self.config.gemini_thinking_budget is not None:
            payload["thinkingBudget"] = int(self.config.gemini_thinking_budget)
        if self._is_nonempty(self.config.gemini_thinking_level):
            payload["thinkingLevel"] = str(self.config.gemini_thinking_level).strip().upper()
        return payload

    def _is_gemini_thinking_config_error(self, error_text: str) -> bool:
        lowered = error_text.lower()
        return any(
            token in lowered
            for token in (
                "thinkingconfig",
                "thinkinglevel",
                "thinkingbudget",
                "includethoughts",
            )
        )

    def _extract_reasoning_tokens(self, usage_payload: Dict[str, Any]) -> int:
        if not isinstance(usage_payload, dict):
            return 0

        reasoning_value = self._find_numeric_usage_value(
            usage_payload,
            candidate_keys=[
                "reasoning_tokens",
                "reasoning_output_tokens",
                "thoughtsTokenCount",
                "thoughtTokenCount",
                "thoughts",
            ],
        )
        return reasoning_value if reasoning_value is not None else 0

    def _extract_usage_count(
        self,
        usage_payload: Dict[str, Any],
        *,
        candidate_keys: List[str],
    ) -> int:
        if not isinstance(usage_payload, dict):
            return 0

        for key in candidate_keys:
            value = usage_payload.get(key)
            if isinstance(value, (int, float)):
                return int(value)

        for nested_key in ("usage", "tokens", "usageMetadata", "aggregatedStats"):
            nested_payload = usage_payload.get(nested_key)
            if isinstance(nested_payload, dict):
                nested_value = self._extract_usage_count(
                    nested_payload,
                    candidate_keys=candidate_keys,
                )
                if nested_value:
                    return nested_value

        return 0

    def _find_numeric_usage_value(
        self,
        payload: Any,
        *,
        candidate_keys: List[str],
    ) -> Optional[int]:
        visited: set[int] = set()

        def _walk(value: Any) -> Optional[int]:
            object_id = id(value)
            if object_id in visited:
                return None
            visited.add(object_id)

            if isinstance(value, dict):
                for key in candidate_keys:
                    candidate = value.get(key)
                    if isinstance(candidate, (int, float)):
                        return int(candidate)
                for nested_value in value.values():
                    nested_result = _walk(nested_value)
                    if nested_result is not None:
                        return nested_result
                return None
            if isinstance(value, list):
                for item in value:
                    nested_result = _walk(item)
                    if nested_result is not None:
                        return nested_result
            return None

        return _walk(payload)

    def _post_json(
        self,
        req: urllib_request.Request,
        timeout_sec: float,
        provider_name: str,
    ) -> Dict[str, Any]:
        total_attempts = max(1, self.config.request_max_retries + 1)
        last_runtime_error: Optional[RuntimeError] = None

        for attempt_idx in range(total_attempts):
            try:
                with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib_error.HTTPError as exc:
                detail = _redact_sensitive_text(exc.read().decode("utf-8", errors="replace"))
                runtime_error = RuntimeError(f"{provider_name} HTTPError {exc.code}: {detail}")
                if self._is_retryable_http_error(exc.code) and attempt_idx < (total_attempts - 1):
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    self._sleep_before_retry(attempt_idx, retry_after_header=retry_after)
                    last_runtime_error = runtime_error
                    continue
                raise runtime_error from exc
            except urllib_error.URLError as exc:
                reason = _redact_sensitive_text(exc.reason)
                runtime_error = RuntimeError(f"{provider_name} URLError: {reason}")
                if attempt_idx < (total_attempts - 1):
                    self._sleep_before_retry(attempt_idx, retry_after_header=None)
                    last_runtime_error = runtime_error
                    continue
                raise runtime_error from exc
            except (TimeoutError, socket.timeout) as exc:
                detail = _redact_sensitive_text(exc)
                runtime_error = RuntimeError(f"{provider_name} TimeoutError: {detail}")
                if attempt_idx < (total_attempts - 1):
                    self._sleep_before_retry(attempt_idx, retry_after_header=None)
                    last_runtime_error = runtime_error
                    continue
                raise runtime_error from exc
        else:
            if last_runtime_error is not None:
                raise last_runtime_error
            raise RuntimeError(f"{provider_name} request failed without response.")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{provider_name} response is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{provider_name} response JSON must be an object.")
        return data

    def _is_retryable_http_error(self, status_code: int) -> bool:
        return status_code in {408, 425, 429, 500, 502, 503, 504}

    def _sleep_before_retry(
        self,
        attempt_idx: int,
        retry_after_header: Optional[str],
    ) -> None:
        retry_after = self._parse_retry_after_seconds(retry_after_header)
        if retry_after is not None:
            delay_sec = retry_after
        else:
            base = float(self.config.request_retry_initial_delay_sec)
            cap = float(self.config.request_retry_max_delay_sec)
            delay_sec = min(cap, base * (2**attempt_idx))
        if delay_sec > 0:
            time.sleep(delay_sec)

    def _parse_retry_after_seconds(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            return None
        if seconds < 0:
            return None
        return seconds

    def _is_param_not_accepted_error(self, error_text: str, param_name: str) -> bool:
        text = (error_text or "").lower()
        param = (param_name or "").lower()
        if not text or not param or param not in text:
            return False
        if "unsupported parameter" in text or "unsupported_parameter" in text:
            return True
        if "unsupported value" in text or "unsupported_value" in text:
            return True
        return False

    def _is_unsupported_parameter_error(self, error_text: str, param_name: str) -> bool:
        text = (error_text or "").lower()
        param = (param_name or "").lower()
        if not text or not param:
            return False
        if "unsupported parameter" not in text and "unsupported_parameter" not in text:
            return False
        return param in text

    def _is_nonempty(self, value: Optional[str]) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _normalize_chat_content(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"]
            return json.dumps(value, ensure_ascii=False)
        return str(value)
