from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.data import build_manual_transition_payload
from src.ui.baba_manual_player_ui import ActionLogEntry


ENV_IDS = [
    "env/you_win",
    "env/you_win-fixed_you",
    "env/make_win-distr_obj_rule",
    "env/goto_win-distr_obj_rule",
    "env/goto_win",
    "env/goto_win-distr_obj",
    "env/goto_win-distr_rule",
    "env/goto_win-distr_obj-irrelevant_rule",
    "env/goto_win-distr_win_rule",
    "env/make_win-distr_obj",
    "env/make_win-distr_rule",
    "env/make_win",
    "env/make_win-distr_obj-irrelevant_rule",
    "env/two_room-goto_win",
    "env/two_room-goto_win-distr_obj_rule",
    "env/two_room-goto_win-distr_rule",
    "env/two_room-goto_win-distr_obj",
    "env/two_room-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-goto_win-distr_win_rule",
    "env/two_room-break_stop-goto_win-distr_obj_rule",
    "env/two_room-break_stop-goto_win-distr_obj",
    "env/two_room-break_stop-goto_win-distr_rule",
    "env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-break_stop-goto_win",
    "env/two_room-maybe_break_stop-goto_win-distr_obj_rule",
    "env/two_room-maybe_break_stop-goto_win",
    "env/two_room-maybe_break_stop-goto_win-distr_obj",
    "env/two_room-maybe_break_stop-goto_win-distr_rule",
    "env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-make_win-distr_obj_rule",
    "env/two_room-make_win-distr_rule",
    "env/two_room-make_win",
    "env/two_room-make_win-distr_obj-irrelevant_rule",
    "env/two_room-make_win-distr_obj",
    "env/two_room-make_win-distr_win_rule",
    "env/two_room-break_stop-make_win-distr_obj_rule",
    "env/two_room-break_stop-make_win-distr_rule",
    "env/two_room-break_stop-make_win",
    "env/two_room-break_stop-make_win-distr_obj-irrelevant_rule",
    "env/two_room-break_stop-make_win-distr_obj",
    "env/two_room-make_you",
    "env/two_room-make_you-make_win",
    "env/two_room-make_wall_win",
    "env/baba_in_wonderland_baseline_easy",
    "env/baba_in_wonderland_baseline_medium",
    "env/baba_in_wonderland_baseline_hard",
    "env/baba_custom_ascii_easy",
    "env/baba_custom_ascii_medium",
    "env/baba_custom_ascii_hard",
    "env/baba_custom_ascii_original",
]

SELECTED_ENV_INDEX = len(ENV_IDS) - 1
RESET_SEED = 42
MAX_STEPS = 1000
AUTOSAVE_EACH_STEP = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "manual_env_play"

ACTION_TO_ID = {
    "idle": 0,
    "up": 1,
    "right": 2,
    "down": 3,
    "left": 4,
}


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "__").replace("#", "_").replace("-", "_")


def build_env_label(*, env_id: str, env_index: int) -> str:
    if int(env_index) < 0:
        return env_id
    return f"env[{int(env_index)}]={env_id}"


def build_scenario_label(
    *,
    scenario_type: Optional[str],
    display_name: Optional[str] = None,
) -> Optional[str]:
    scenario = str(scenario_type or "").strip()
    name = str(display_name or "").strip()
    if scenario and name and name != scenario:
        return f"{scenario} | {name}"
    if scenario:
        return scenario
    if name:
        return name
    return None


def build_save_payload(
    *,
    env_id: str,
    env_index: int,
    scenario_type: Optional[str] = None,
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
    return build_manual_transition_payload(
        env_id=env_id,
        env_index=env_index,
        scenario_type=scenario_type,
        step_index=step_index,
        action_name=action_name,
        action_id=action_id,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        previous_state_raw=previous_state_raw,
        previous_state_obj=previous_state_obj,
        next_state_raw=next_state_raw,
        next_state_obj=next_state_obj,
    )


def build_render_meta_lines(
    *,
    output_dir: Path,
    autosave_enabled: bool,
) -> tuple[str, ...]:
    return (
        f"seed={int(RESET_SEED)}",
        f"autosave={'ON' if autosave_enabled else 'OFF'}",
        f"out={output_dir.name}",
    )


def render_play_frame(
    *,
    renderer: Any,
    state_obj: Dict[str, Any],
    env_id: str,
    env_index: int,
    scenario_type: Optional[str],
    scenario_display_name: Optional[str] = None,
    step_index: int,
    action_name: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    episode_done: bool,
    output_dir: Path,
    autosave_enabled: bool,
    log_entries: Optional[list[ActionLogEntry]] = None,
    info_message: Optional[str] = None,
    animation_ms: int = 0,
    visual_config: Optional[Dict[str, Any]] = None,
):
    return renderer.render(
        state=state_obj,
        env_label=build_env_label(env_id=env_id, env_index=env_index),
        scenario_type=build_scenario_label(
            scenario_type=scenario_type,
            display_name=scenario_display_name,
        ),
        step_index=int(step_index),
        action_name=action_name,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_done=bool(episode_done),
        info_message=info_message,
        log_entries=log_entries,
        animation_ms=int(animation_ms),
        visual_config=visual_config,
        meta_lines=build_render_meta_lines(
            output_dir=output_dir,
            autosave_enabled=autosave_enabled,
        ),
    )


def resolve_env_selection(
    requested_env_id: Optional[str],
    custom_map_path: Optional[str] = None,
) -> tuple[int, str]:
    if isinstance(custom_map_path, str) and custom_map_path.strip():
        return -1, "env/baba_custom_ascii"

    if requested_env_id is None:
        env_index = int(SELECTED_ENV_INDEX)
        if not (0 <= env_index < len(ENV_IDS)):
            env_index = len(ENV_IDS) - 1
        return env_index, ENV_IDS[env_index]

    if requested_env_id not in ENV_IDS:
        available = "\n".join(f"  - {env_id}" for env_id in ENV_IDS)
        raise ValueError(
            f"Unknown --env `{requested_env_id}`.\nAvailable environment ids:\n{available}"
        )

    env_index = ENV_IDS.index(requested_env_id)
    return env_index, requested_env_id
