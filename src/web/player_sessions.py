from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.compare_runtime import save_transition_artifacts
from src.player_runtime import (
    ACTION_TO_ID,
    ENV_IDS,
    MAX_STEPS,
    OUTPUT_ROOT,
    SELECTED_ENV_INDEX,
    build_env_label,
    build_save_payload,
    build_scenario_label,
    render_play_frame,
    resolve_env_selection,
    sanitize_env_id,
)
from src.environments import BabaWrapper
from src.ui.baba_manual_player_ui import ActionLogEntry, BabaManualPlayerRenderer


PLAYER_FRAME_RENDERER = BabaManualPlayerRenderer()
DEFAULT_RESET_SEED = 42


def _resolve_wrapper_board_state(wrapper: BabaWrapper) -> Dict[str, Any]:
    return dict(wrapper.get_visualization_state())


def resolve_default_env_id() -> str:
    return ENV_IDS[int(SELECTED_ENV_INDEX)]


def resolve_initial_scenario_type(
    *,
    env_id: str,
    requested_scenario_type: Optional[str],
    seed: int,
) -> Optional[str]:
    normalized_requested = (
        str(requested_scenario_type).strip()
        if isinstance(requested_scenario_type, str) and requested_scenario_type.strip()
        else None
    )
    if normalized_requested is not None:
        return normalized_requested

    wrapper = BabaWrapper(
        env_name=env_id,
        max_steps=int(MAX_STEPS),
        render_mode=None,
        state_format="json",
        seed=int(seed),
        env_kwargs={},
    )
    try:
        raw_catalog = getattr(wrapper.env, "SCENARIO_TYPES", ())
        if not isinstance(raw_catalog, (list, tuple)):
            return None
        catalog = [str(item).strip() for item in raw_catalog if str(item).strip()]
        return catalog[0] if catalog else None
    finally:
        wrapper.close()


def _resolve_scenario_catalog(wrapper: BabaWrapper) -> tuple[str, ...]:
    raw_catalog = getattr(wrapper.env, "SCENARIO_TYPES", ())
    if not isinstance(raw_catalog, (list, tuple)):
        return ()
    return tuple(str(item) for item in raw_catalog)


def resolve_scenario_display_name(wrapper: BabaWrapper) -> Optional[str]:
    base_env = getattr(wrapper, "env", None)
    if base_env is None:
        return None
    active_scenario = getattr(base_env, "scenario_type", None)
    if not isinstance(active_scenario, str) or not active_scenario.strip():
        return None
    active_scenario = active_scenario.strip()

    for attr_name in ("custom_catalog", "SCENARIO_SPECS", "HARD_SCENARIOS"):
        raw_catalog = getattr(base_env, attr_name, None)
        if not isinstance(raw_catalog, dict):
            continue
        entry = raw_catalog.get(active_scenario)
        if isinstance(entry, dict) and isinstance(entry.get("spec"), dict):
            spec = entry["spec"]
        elif isinstance(entry, dict):
            spec = entry
        else:
            continue
        display_name = str(spec.get("display_name") or "").strip()
        if display_name:
            return display_name
    return None


def resolve_named_scenario_display_name(
    wrapper: BabaWrapper,
    scenario_type: Optional[str],
) -> Optional[str]:
    if not isinstance(scenario_type, str) or not scenario_type.strip():
        return None
    normalized = scenario_type.strip()
    base_env = getattr(wrapper, "env", None)
    if base_env is None:
        return None

    for attr_name in ("custom_catalog", "SCENARIO_SPECS", "HARD_SCENARIOS"):
        raw_catalog = getattr(base_env, attr_name, None)
        if not isinstance(raw_catalog, dict):
            continue
        entry = raw_catalog.get(normalized)
        if isinstance(entry, dict) and isinstance(entry.get("spec"), dict):
            spec = entry["spec"]
        elif isinstance(entry, dict):
            spec = entry
        else:
            continue
        display_name = str(spec.get("display_name") or "").strip()
        if display_name:
            return display_name
    return None


@dataclass(frozen=True)
class PlayerCheckpoint:
    snapshot: Dict[str, Any]
    state_raw: str
    state_obj: Dict[str, Any]
    step_index: int
    last_reward: float
    terminated: bool
    truncated: bool
    last_action_name: str
    last_action_id: int


def _copy_state_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    return copy.deepcopy(payload)


def _serialize_action_log_entry(entry: ActionLogEntry) -> Dict[str, Any]:
    return {
        "stepIndex": int(entry.step_index),
        "actionName": str(entry.action_name),
        "reward": float(entry.reward),
        "terminated": bool(entry.terminated),
        "truncated": bool(entry.truncated),
        "note": str(entry.note),
        "eventType": str(entry.event_type),
        "label": entry.label,
        "statusKey": entry.status_key,
        "rewardText": entry.reward_text,
    }


@dataclass
class PlayerSession:
    env_id: str
    env_index: int
    seed: int
    scenario_type: Optional[str] = None
    autosave_enabled: bool = False
    wrapper: Optional[BabaWrapper] = None
    current_state_raw: str = ""
    current_state_obj: Optional[Dict[str, Any]] = None
    previous_state_raw: Optional[str] = None
    previous_state_obj: Optional[Dict[str, Any]] = None
    step_index: int = 0
    last_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    last_action_name: str = "reset"
    last_action_id: int = -1
    output_dir: Optional[Path] = None
    action_history: list[ActionLogEntry] = field(default_factory=list)
    _timeline: list[PlayerCheckpoint] = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls,
        *,
        requested_env_id: Optional[str],
        requested_scenario_type: Optional[str],
        seed: int,
        autosave_enabled: bool = False,
    ) -> "PlayerSession":
        env_index, env_id = resolve_env_selection(requested_env_id)
        initial_scenario = resolve_initial_scenario_type(
            env_id=env_id,
            requested_scenario_type=requested_scenario_type,
            seed=int(seed),
        )
        session = cls(
            env_id=env_id,
            env_index=env_index,
            seed=int(seed),
            scenario_type=initial_scenario,
            autosave_enabled=bool(autosave_enabled),
        )
        session.reset(seed=int(seed), scenario_type=initial_scenario)
        if (
            isinstance(requested_scenario_type, str)
            and requested_scenario_type.strip()
            and session.scenario_type != requested_scenario_type.strip()
        ):
            session.close()
            raise ValueError(
                f"Requested scenario `{requested_scenario_type}` was not activated. "
                f"Active scenario is `{session.scenario_type}`."
            )
        return session

    def _build_wrapper(self, scenario_type: Optional[str]) -> BabaWrapper:
        env_kwargs: Dict[str, Any] = {}
        if isinstance(scenario_type, str) and scenario_type.strip():
            env_kwargs["scenario_type"] = scenario_type.strip()
        return BabaWrapper(
            env_name=self.env_id,
            max_steps=int(MAX_STEPS),
            render_mode=None,
            state_format="json",
            seed=int(self.seed),
            env_kwargs=env_kwargs,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        scenario_type: Optional[str] = None,
    ) -> None:
        if seed is not None:
            self.seed = int(seed)
        if scenario_type is not None:
            self.scenario_type = str(scenario_type).strip() or None

        if self.wrapper is not None:
            self.wrapper.close()
        self.wrapper = self._build_wrapper(self.scenario_type)
        self.current_state_raw = self.wrapper.reset(seed=int(self.seed))
        self.current_state_obj = _resolve_wrapper_board_state(self.wrapper)
        self.previous_state_raw = None
        self.previous_state_obj = None
        self.step_index = 0
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.last_action_name = "reset"
        self.last_action_id = -1
        self.output_dir = OUTPUT_ROOT / sanitize_env_id(self.env_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        active_scenario = getattr(self.wrapper.env, "scenario_type", None)
        self.scenario_type = str(active_scenario).strip() if isinstance(active_scenario, str) else None
        self.action_history = [
            ActionLogEntry(
                step_index=0,
                action_name="reset",
                reward=0.0,
                event_type="reset",
            )
        ]
        self._timeline = [self._capture_checkpoint()]

    def advance_to_next_scenario(self) -> None:
        if self.wrapper is None:
            raise RuntimeError("session is not initialized")
        scenario_catalog = _resolve_scenario_catalog(self.wrapper)
        if not scenario_catalog:
            raise RuntimeError("this environment does not expose multiple scenarios.")
        current = str(self.scenario_type or "").strip()
        if current in scenario_catalog:
            next_index = (scenario_catalog.index(current) + 1) % len(scenario_catalog)
        else:
            next_index = 0
        self.reset(seed=int(self.seed), scenario_type=scenario_catalog[next_index])

    def select_scenario(self, *, scenario_type: Optional[str]) -> None:
        if self.wrapper is None:
            raise RuntimeError("session is not initialized")

        normalized = (
            str(scenario_type).strip()
            if isinstance(scenario_type, str) and str(scenario_type).strip()
            else None
        )
        scenario_catalog = _resolve_scenario_catalog(self.wrapper)
        if scenario_catalog and normalized not in scenario_catalog:
            available = "\n".join(f"  - {value}" for value in scenario_catalog)
            raise ValueError(
                f"Unknown scenario `{scenario_type}`.\nAvailable scenarios:\n{available}"
            )
        self.reset(seed=int(self.seed), scenario_type=normalized)

    def set_recording_enabled(self, enabled: bool) -> None:
        self.autosave_enabled = bool(enabled)

    def step(self, *, action_name: str) -> None:
        if self.wrapper is None:
            raise RuntimeError("session is not initialized")
        normalized = str(action_name).strip().lower()
        if normalized not in ACTION_TO_ID:
            raise ValueError(f"Unsupported action `{action_name}`.")
        if self.terminated or self.truncated:
            raise RuntimeError("episode already ended; reset the session first.")

        self.previous_state_raw = self.current_state_raw
        self.previous_state_obj = _copy_state_payload(self.current_state_obj)
        next_state_raw, reward, terminated, truncated, _info = self.wrapper.step(
            ACTION_TO_ID[normalized]
        )
        self.current_state_raw = next_state_raw
        self.current_state_obj = _resolve_wrapper_board_state(self.wrapper)
        self.step_index += 1
        self.last_reward = float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.last_action_name = normalized
        self.last_action_id = int(ACTION_TO_ID[normalized])
        self.action_history.append(
            ActionLogEntry(
                step_index=int(self.step_index),
                action_name=normalized,
                reward=float(self.last_reward),
                terminated=bool(self.terminated),
                truncated=bool(self.truncated),
            )
        )
        self._timeline.append(self._capture_checkpoint())
        if self.autosave_enabled:
            self.save_current_frame()

    def undo_last_action(self) -> None:
        if self.wrapper is None:
            raise RuntimeError("session is not initialized")
        if len(self._timeline) <= 1:
            raise RuntimeError("there is no action to undo.")

        self._timeline.pop()
        if self.action_history and self.action_history[-1].event_type == "action":
            self.action_history.pop()

        checkpoint = self._timeline[-1]
        self.wrapper.restore(checkpoint.snapshot, refresh_cache=True)
        self._apply_checkpoint(checkpoint)

    def save_current_frame(self) -> Dict[str, Any]:
        if self.wrapper is None or self.current_state_obj is None:
            raise RuntimeError("session is not initialized")
        if self.output_dir is None:
            self.output_dir = OUTPUT_ROOT / sanitize_env_id(self.env_id)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        image = render_play_frame(
            renderer=PLAYER_FRAME_RENDERER,
            state_obj=self.current_state_obj,
            env_id=self.env_id,
            env_index=self.env_index,
            scenario_type=self.scenario_type,
            scenario_display_name=resolve_scenario_display_name(self.wrapper),
            step_index=int(self.step_index),
            action_name=self.last_action_name,
            reward=float(self.last_reward),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            episode_done=bool(self.terminated or self.truncated),
            output_dir=self.output_dir,
            autosave_enabled=bool(self.autosave_enabled),
            log_entries=self.action_history,
            info_message="Saved from player web UI.",
            animation_ms=0,
            visual_config=self.wrapper.get_visualization_config(),
        )
        payload = build_save_payload(
            env_id=self.env_id,
            env_index=self.env_index,
            scenario_type=self.scenario_type,
            step_index=int(self.step_index),
            action_name=self.last_action_name,
            action_id=int(self.last_action_id),
            reward=float(self.last_reward),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            previous_state_raw=self.previous_state_raw,
            previous_state_obj=self.previous_state_obj,
            next_state_raw=self.current_state_raw,
            next_state_obj=self.current_state_obj,
        )
        try:
            image_path, json_path = save_transition_artifacts(
                output_dir=self.output_dir,
                step_index=int(self.step_index),
                action_name=self.last_action_name,
                image=image,
                payload=payload,
            )
        finally:
            image.close()
        return {
            "imagePath": str(image_path),
            "jsonPath": str(json_path),
        }

    def to_payload(self) -> Dict[str, Any]:
        if self.wrapper is None or self.current_state_obj is None:
            raise RuntimeError("session is not initialized")
        scenario_display_name = resolve_scenario_display_name(self.wrapper)
        maps = self._build_map_catalog()
        return {
            "envId": self.env_id,
            "envIndex": self.env_index,
            "envLabel": build_env_label(env_id=self.env_id, env_index=self.env_index),
            "seed": int(self.seed),
            "scenarioType": self.scenario_type,
            "scenarioDisplayName": scenario_display_name,
            "scenarioLabel": build_scenario_label(
                scenario_type=self.scenario_type,
                display_name=scenario_display_name,
            ),
            "scenarios": list(_resolve_scenario_catalog(self.wrapper)),
            "maps": maps,
            "currentMapLabel": next(
                (
                    str(entry.get("label"))
                    for entry in maps
                    if bool(entry.get("isActive"))
                ),
                scenario_display_name or self.scenario_type or "default/randomized",
            ),
            "stepIndex": int(self.step_index),
            "reward": float(self.last_reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "canUndo": len(self._timeline) > 1,
            "actionHistory": [
                _serialize_action_log_entry(entry)
                for entry in self.action_history
            ],
            "autosaveEnabled": bool(self.autosave_enabled),
            "outputDir": str(self.output_dir) if self.output_dir is not None else None,
            "boardState": dict(self.current_state_obj),
            "visualConfig": self.wrapper.get_visualization_config(),
        }

    def close(self) -> None:
        if self.wrapper is not None:
            self.wrapper.close()
            self.wrapper = None

    def _capture_checkpoint(self) -> PlayerCheckpoint:
        if self.wrapper is None or self.current_state_obj is None:
            raise RuntimeError("session is not initialized")
        return PlayerCheckpoint(
            snapshot=self.wrapper.snapshot(),
            state_raw=str(self.current_state_raw),
            state_obj=copy.deepcopy(self.current_state_obj),
            step_index=int(self.step_index),
            last_reward=float(self.last_reward),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            last_action_name=str(self.last_action_name),
            last_action_id=int(self.last_action_id),
        )

    def _apply_checkpoint(self, checkpoint: PlayerCheckpoint) -> None:
        self.current_state_raw = str(checkpoint.state_raw)
        self.current_state_obj = copy.deepcopy(checkpoint.state_obj)
        self.step_index = int(checkpoint.step_index)
        self.last_reward = float(checkpoint.last_reward)
        self.terminated = bool(checkpoint.terminated)
        self.truncated = bool(checkpoint.truncated)
        self.last_action_name = str(checkpoint.last_action_name)
        self.last_action_id = int(checkpoint.last_action_id)
        active_scenario = getattr(getattr(self.wrapper, "env", None), "scenario_type", None)
        self.scenario_type = (
            str(active_scenario).strip()
            if isinstance(active_scenario, str) and str(active_scenario).strip()
            else None
        )
        if len(self._timeline) > 1:
            previous = self._timeline[-2]
            self.previous_state_raw = str(previous.state_raw)
            self.previous_state_obj = copy.deepcopy(previous.state_obj)
        else:
            self.previous_state_raw = None
            self.previous_state_obj = None

    def _build_map_catalog(self) -> list[Dict[str, Any]]:
        if self.wrapper is None:
            return []

        resolved_active = (
            str(self.scenario_type).strip()
            if isinstance(self.scenario_type, str) and str(self.scenario_type).strip()
            else None
        )
        scenario_catalog = _resolve_scenario_catalog(self.wrapper)
        if scenario_catalog:
            return [
                {
                    "index": int(index),
                    "scenarioType": scenario_type,
                    "label": (
                        resolve_named_scenario_display_name(self.wrapper, scenario_type)
                        or scenario_type
                    ),
                    "isActive": scenario_type == resolved_active,
                }
                for index, scenario_type in enumerate(scenario_catalog)
            ]

        rows: list[Dict[str, Any]] = []
        list_graph_search_worlds = getattr(self.wrapper, "list_graph_search_worlds", None)
        worlds = (
            list_graph_search_worlds(seed=int(self.seed))
            if callable(list_graph_search_worlds)
            else ()
        )
        for index, row in enumerate(worlds):
            raw_scenario_type = row.get("scenario_type") if isinstance(row, dict) else None
            scenario_type = (
                str(raw_scenario_type).strip()
                if isinstance(raw_scenario_type, str) and str(raw_scenario_type).strip()
                else None
            )
            label = (
                str(row.get("label")).strip()
                if isinstance(row, dict) and isinstance(row.get("label"), str)
                else ""
            )
            if not label:
                label = scenario_type or f"map {int(index) + 1}"
            rows.append(
                {
                    "index": int(index),
                    "scenarioType": scenario_type,
                    "label": label,
                    "isActive": scenario_type == resolved_active,
                }
            )
        if rows:
            return rows

        fallback_label = resolve_scenario_display_name(self.wrapper) or self.scenario_type or f"seed={int(self.seed)}"
        return [
            {
                "index": 0,
                "scenarioType": self.scenario_type,
                "label": str(fallback_label),
                "isActive": True,
            }
        ]
