from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


@dataclass
class LLMUsageEvent:
    component: str
    provider: str
    model: str
    prompt_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    total_tokens: int
    outcome: str
    prompt_chars: int
    response_chars: int
    raw_usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    trial_index: Optional[int] = None
    step_index: Optional[int] = None
    alignment_round: Optional[int] = None
    repair_round: Optional[int] = None
    timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LLMUsageEvent":
        raw_usage = payload.get("raw_usage")
        return cls(
            component=str(payload.get("component", "")),
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            prompt_tokens=_coerce_int(payload.get("prompt_tokens"), default=0),
            reasoning_tokens=_coerce_int(payload.get("reasoning_tokens"), default=0),
            completion_tokens=_coerce_int(payload.get("completion_tokens"), default=0),
            total_tokens=_coerce_int(payload.get("total_tokens"), default=0),
            outcome=str(payload.get("outcome", "")),
            prompt_chars=_coerce_int(payload.get("prompt_chars"), default=0),
            response_chars=_coerce_int(payload.get("response_chars"), default=0),
            raw_usage=dict(raw_usage) if isinstance(raw_usage, Mapping) else None,
            error=_coerce_optional_str(payload.get("error")),
            trial_index=_coerce_optional_int(payload.get("trial_index")),
            step_index=_coerce_optional_int(payload.get("step_index")),
            alignment_round=_coerce_optional_int(payload.get("alignment_round")),
            repair_round=_coerce_optional_int(payload.get("repair_round")),
            timestamp_utc=str(payload.get("timestamp_utc", "")),
        )


def _coerce_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


class LLMUsageTracker:
    def __init__(self, on_update: Optional[Callable[[], None]] = None) -> None:
        self.events: List[LLMUsageEvent] = []
        self.on_update = on_update

    def record(
        self,
        *,
        component: str,
        predictor: Any,
        prompt_text: str,
        response_text: Optional[str],
        outcome: str,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage = getattr(predictor, "last_usage", None)
        usage_payload = usage.to_dict() if hasattr(usage, "to_dict") else {}
        context = dict(context or {})
        self.events.append(
            LLMUsageEvent(
                component=component,
                provider=str(usage_payload.get("provider", "")),
                model=str(usage_payload.get("model", "")),
                prompt_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
                reasoning_tokens=int(usage_payload.get("reasoning_tokens", 0) or 0),
                completion_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
                total_tokens=int(usage_payload.get("total_tokens", 0) or 0),
                raw_usage=(
                    dict(usage_payload.get("raw_usage"))
                    if isinstance(usage_payload.get("raw_usage"), dict)
                    else None
                ),
                outcome=str(outcome),
                prompt_chars=len(prompt_text or ""),
                response_chars=len(response_text or ""),
                error=error,
                trial_index=context.get("trial_index"),
                step_index=context.get("step_index"),
                alignment_round=context.get("alignment_round"),
                repair_round=context.get("repair_round"),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        if callable(self.on_update):
            self.on_update()

    def restore_events(
        self,
        events: Iterable[LLMUsageEvent | Mapping[str, Any]],
        *,
        replace: bool = False,
        emit_update: bool = False,
    ) -> None:
        restored: List[LLMUsageEvent] = []
        for event in events:
            if isinstance(event, LLMUsageEvent):
                restored.append(event)
                continue
            if isinstance(event, Mapping):
                restored.append(LLMUsageEvent.from_dict(event))
                continue
            raise TypeError(f"Unsupported LLM usage event payload: {type(event)!r}")
        if replace:
            self.events = restored
        else:
            self.events.extend(restored)
        if emit_update and callable(self.on_update):
            self.on_update()

    def build_summary(self) -> Dict[str, Any]:
        def _empty() -> Dict[str, Any]:
            return {
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "prompt_tokens": 0,
                "reasoning_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        overall = _empty()
        components: Dict[str, Dict[str, Any]] = defaultdict(_empty)
        by_trial: Dict[str, Dict[str, Any]] = {}

        for event in self.events:
            for bucket in (overall, components[event.component]):
                bucket["call_count"] += 1
                bucket["success_count"] += 1 if event.outcome == "success" else 0
                bucket["error_count"] += 1 if event.outcome != "success" else 0
                bucket["prompt_tokens"] += event.prompt_tokens
                bucket["reasoning_tokens"] += event.reasoning_tokens
                bucket["completion_tokens"] += event.completion_tokens
                bucket["total_tokens"] += event.total_tokens

            if event.trial_index is None:
                continue
            trial_key = f"trial_{int(event.trial_index):03d}"
            if trial_key not in by_trial:
                by_trial[trial_key] = {
                    "overall": _empty(),
                    "components": defaultdict(_empty),
                }
            trial_bucket = by_trial[trial_key]
            for bucket in (
                trial_bucket["overall"],
                trial_bucket["components"][event.component],
            ):
                bucket["call_count"] += 1
                bucket["success_count"] += 1 if event.outcome == "success" else 0
                bucket["error_count"] += 1 if event.outcome != "success" else 0
                bucket["prompt_tokens"] += event.prompt_tokens
                bucket["reasoning_tokens"] += event.reasoning_tokens
                bucket["completion_tokens"] += event.completion_tokens
                bucket["total_tokens"] += event.total_tokens

        normalized_trials: Dict[str, Any] = {}
        for trial_key, payload in by_trial.items():
            normalized_trials[trial_key] = {
                "overall": dict(payload["overall"]),
                "components": {
                    component: dict(stats)
                    for component, stats in payload["components"].items()
                },
            }

        return {
            "overall": dict(overall),
            "components": {
                component: dict(stats) for component, stats in components.items()
            },
            "trial_breakdown": normalized_trials,
            "events": [event.to_dict() for event in self.events],
        }


def build_summary_from_events(
    events: Iterable[LLMUsageEvent | Mapping[str, Any]],
) -> Dict[str, Any]:
    tracker = LLMUsageTracker()
    tracker.restore_events(events, replace=True)
    return tracker.build_summary()
