from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from src.data.state_schema import (
    DIRECTION_NAMES,
    build_state_payload,
    dump_state_json,
    normalize_raw_baba_object_schema,
    normalize_state_direction,
)


class StateSerializer:
    """Serialize BABA states as JSON text."""

    DIRECTION_NAMES = DIRECTION_NAMES

    def __init__(
        self,
        format_type: str = "json",
        word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        normalized = str(format_type).strip().lower()
        if normalized != "json":
            raise ValueError(
                "ASCII state format is disabled. `serialization.format` must be `json`."
            )
        self.format_type = "json"
        self.word_aliases = word_aliases

    def serialize(
        self,
        grid_size: Tuple[int, int],
        objects,
        agent_pos=None,
        agent_dir=None,
        event=None,
        reward=None,
        terminated: Optional[bool] = None,
        truncated=None,
    ) -> str:
        del agent_pos, agent_dir, event, reward, truncated
        if terminated is None:
            raise ValueError("terminated must be provided.")
        payload = build_state_payload(
            grid_size=grid_size,
            objects=objects,
            terminated=bool(terminated),
            word_aliases=self.word_aliases,
        )
        return dump_state_json(payload)

    def _trim_grid_size(self, raw_width: int, raw_height: int) -> Tuple[int, int]:
        return max(0, int(raw_width) - 2), max(0, int(raw_height) - 2)

    def _normalize_objects(
        self,
        objects,
        *,
        raw_width: int,
        raw_height: int,
    ):
        payload = build_state_payload(
            grid_size=(int(raw_width), int(raw_height)),
            objects=list(objects),
            terminated=False,
            word_aliases=self.word_aliases,
        )
        return list(payload.get("objects", []))


__all__ = [
    "StateSerializer",
    "normalize_raw_baba_object_schema",
    "normalize_state_direction",
]
