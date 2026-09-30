from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


MANUAL_TRANSITION_ARTIFACT_VERSION = 2


def _as_optional_state_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("state payload must be a string or null.")
    normalized = value.strip()
    return normalized or None


def _as_required_state_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _as_optional_state_object(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("state object payload must be a mapping or null.")
    return dict(value)


def _as_required_state_object(value: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return dict(value)


@dataclass(frozen=True)
class ManualTransitionArtifact:
    env_id: str
    env_index: int
    step_index: int
    action: str
    action_id: int
    reward: float
    terminated: bool
    truncated: bool
    previous_state_raw: Optional[str]
    previous_state: Optional[Dict[str, Any]]
    next_state_raw: str
    next_state: Dict[str, Any]
    scenario_type: Optional[str] = None
    schema_version: int = MANUAL_TRANSITION_ARTIFACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "env_id": self.env_id,
            "env_index": int(self.env_index),
            "scenario_type": self.scenario_type,
            "step_index": int(self.step_index),
            "action": self.action,
            "action_id": int(self.action_id),
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "previous_state_raw": self.previous_state_raw,
            "previous_state": (
                dict(self.previous_state)
                if isinstance(self.previous_state, dict)
                else None
            ),
            "next_state_raw": self.next_state_raw,
            "next_state": dict(self.next_state),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ManualTransitionArtifact":
        if not isinstance(payload, Mapping):
            raise ValueError("manual transition payload must be a mapping.")

        env_id = payload.get("env_id", "env/unknown")
        if not isinstance(env_id, str) or not env_id.strip():
            env_id = "env/unknown"

        env_index = payload.get("env_index", -1)
        try:
            resolved_env_index = int(env_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("env_index must be an integer.") from exc

        step_index = payload.get("step_index", 0)
        try:
            resolved_step_index = int(step_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("step_index must be an integer.") from exc

        action_name = payload.get("action")
        if not isinstance(action_name, str) or not action_name.strip():
            raise ValueError("action must be a non-empty string.")

        action_id = payload.get("action_id", -1)
        try:
            resolved_action_id = int(action_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_id must be an integer.") from exc

        reward = payload.get("reward", 0.0)
        try:
            resolved_reward = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward must be numeric.") from exc

        previous_state_raw = _as_optional_state_text(payload.get("previous_state_raw"))
        previous_state_obj = _as_optional_state_object(payload.get("previous_state"))

        next_state_raw = payload.get("next_state_raw")
        if next_state_raw is None:
            next_state_raw = payload.get("state_raw")
        resolved_next_state_raw = _as_required_state_text(
            next_state_raw,
            field_name="next_state_raw",
        )

        next_state_obj = payload.get("next_state")
        if next_state_obj is None:
            next_state_obj = payload.get("state")
        if next_state_obj is None:
            resolved_next_state_obj = {}
        else:
            resolved_next_state_obj = _as_required_state_object(
                next_state_obj,
                field_name="next_state",
            )

        scenario_type = payload.get("scenario_type")
        if scenario_type is not None and (
            not isinstance(scenario_type, str) or not scenario_type.strip()
        ):
            scenario_type = None

        return cls(
            env_id=env_id.strip(),
            env_index=resolved_env_index,
            scenario_type=(scenario_type.strip() if isinstance(scenario_type, str) else None),
            step_index=resolved_step_index,
            action=action_name.strip(),
            action_id=resolved_action_id,
            reward=resolved_reward,
            terminated=bool(payload.get("terminated", False)),
            truncated=bool(payload.get("truncated", False)),
            previous_state_raw=previous_state_raw,
            previous_state=previous_state_obj,
            next_state_raw=resolved_next_state_raw,
            next_state=resolved_next_state_obj,
            schema_version=int(
                payload.get("schema_version", MANUAL_TRANSITION_ARTIFACT_VERSION)
            ),
        )


def build_manual_transition_payload(
    *,
    env_id: str,
    env_index: int,
    scenario_type: Optional[str],
    step_index: int,
    action_name: str,
    action_id: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    previous_state_raw: Optional[str],
    previous_state_obj: Optional[Dict[str, Any]],
    next_state_raw: str,
    next_state_obj: Dict[str, Any],
) -> Dict[str, Any]:
    artifact = ManualTransitionArtifact(
        env_id=str(env_id).strip() or "env/unknown",
        env_index=int(env_index),
        scenario_type=(
            scenario_type.strip()
            if isinstance(scenario_type, str) and scenario_type.strip()
            else None
        ),
        step_index=int(step_index),
        action=str(action_name).strip(),
        action_id=int(action_id),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        previous_state_raw=(
            previous_state_raw.strip()
            if isinstance(previous_state_raw, str) and previous_state_raw.strip()
            else None
        ),
        previous_state=(
            dict(previous_state_obj)
            if isinstance(previous_state_obj, dict)
            else None
        ),
        next_state_raw=_as_required_state_text(
            next_state_raw,
            field_name="next_state_raw",
        ),
        next_state=_as_required_state_object(
            next_state_obj,
            field_name="next_state",
        ),
    )
    return artifact.to_dict()
