"""Static explorer backed by manually saved transition JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base_agent import BaseExplorer
from src.data import ManualTransitionArtifact
from src.data.transition_buffer import Transition


_TIMESTAMP_PATTERN = re.compile(r"_(\d{8}_\d{6})\.json$", re.IGNORECASE)


class ManualTransitionExplorer(BaseExplorer):
    """Expose saved manual transition JSONs through the explorer interface."""

    def __init__(self, source_dir: Path | str, seed: int = 42):
        self.source_dir = Path(source_dir)
        self.seed = int(seed)
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Manual transition directory does not exist: {self.source_dir}"
            )

        self._entries = self._load_entries(self.source_dir)
        if not self._entries:
            raise ValueError(
                f"No usable manual transitions found under: {self.source_dir}"
            )

        self._cursor = 0
        self._last_collect_stats: Dict[str, Any] = {}

    @property
    def total_transitions(self) -> int:
        return len(self._entries)

    def collect(
        self,
        max_transitions: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_on_unknown_transition: bool = False,
    ) -> List[Transition]:
        _ = stop_on_unknown_transition
        if self._cursor >= len(self._entries):
            self._last_collect_stats = {
                "transitions_collected": 0,
                "remaining_after_collect": 0,
                "source_dir": str(self.source_dir),
                "source": "manual",
                "exhausted": True,
            }
            return []

        if isinstance(max_transitions, int) and max_transitions > 0:
            end_index = min(len(self._entries), self._cursor + int(max_transitions))
        else:
            end_index = len(self._entries)

        batch_entries = self._entries[self._cursor:end_index]
        transitions = [entry[1] for entry in batch_entries]
        self._cursor = end_index

        if callable(progress_callback):
            for index, _transition in enumerate(transitions, start=1):
                try:
                    progress_callback(
                        {
                            "transitions_collected": index,
                            "source": "manual",
                        }
                    )
                except Exception:
                    pass

        self._last_collect_stats = {
            "transitions_collected": len(transitions),
            "remaining_after_collect": max(0, len(self._entries) - self._cursor),
            "source_dir": str(self.source_dir),
            "source": "manual",
            "exhausted": False,
            "last_batch_first_file": str(batch_entries[0][0]) if batch_entries else None,
            "last_batch_last_file": str(batch_entries[-1][0]) if batch_entries else None,
        }
        return list(transitions)

    def reset(self) -> None:
        self._cursor = 0
        self._last_collect_stats = {}

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "name": "manual_transition",
            "seed": self.seed,
            "source_dir": str(self.source_dir),
            "total_transitions": len(self._entries),
            "remaining_transitions": max(0, len(self._entries) - self._cursor),
            "exhausted": self._cursor >= len(self._entries),
            "last_collect": dict(self._last_collect_stats),
        }

    @classmethod
    def _load_entries(cls, source_dir: Path) -> List[Tuple[Path, Transition]]:
        entries: List[Tuple[Path, Transition]] = []
        for path in sorted(source_dir.rglob("*.json"), key=cls._sort_key_for_path):
            transition = cls._load_transition_from_file(path)
            if transition is None:
                continue
            entries.append((path, transition))
        return entries

    @classmethod
    def _sort_key_for_path(cls, path: Path) -> Tuple[str, str]:
        timestamp = ""
        match = _TIMESTAMP_PATTERN.search(path.name)
        if match is not None:
            timestamp = match.group(1)
        return (timestamp, str(path).lower())

    @staticmethod
    def _load_transition_from_file(path: Path) -> Optional[Transition]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        try:
            artifact = ManualTransitionArtifact.from_payload(payload)
        except ValueError:
            return None
        if not isinstance(artifact.previous_state_raw, str) or not artifact.previous_state_raw:
            return None
        done = payload.get("done")
        if not isinstance(done, bool):
            done = bool(artifact.terminated) or bool(artifact.truncated)

        return Transition(
            state=artifact.previous_state_raw,
            action=artifact.action,
            next_state=artifact.next_state_raw,
            reward=float(artifact.reward),
            done=bool(done),
            world_index=(
                int(artifact.env_index)
                if isinstance(artifact.env_index, int) and int(artifact.env_index) > 0
                else None
            ),
            map_name=(
                str(artifact.scenario_type).strip()
                if isinstance(artifact.scenario_type, str) and str(artifact.scenario_type).strip()
                else None
            ),
        )
