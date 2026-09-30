from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def resolve_predicted_visual_state(
    *,
    predicted_logical_state: Optional[Mapping[str, Any]],
    actual_logical_state: Optional[Mapping[str, Any]] = None,
    actual_visual_state: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(predicted_logical_state, Mapping):
        return None

    if (
        isinstance(actual_logical_state, Mapping)
        and isinstance(actual_visual_state, Mapping)
        and dict(predicted_logical_state) == dict(actual_logical_state)
    ):
        return dict(actual_visual_state)

    return dict(predicted_logical_state)
