from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import uuid
from typing import Any, Dict, Optional

from src.compare_runtime import (
    ACTION_TO_ID,
    build_comparison_difference_summary,
    build_sandbox_config,
    build_status_lines,
    compose_comparison_image,
    load_yaml,
    normalize_version_tag,
    resolve_config_from_experiment_snapshot,
    resolve_experiment_dir,
    resolve_path,
    resolve_program_path,
    save_transition_artifacts,
)
from src.environments import BabaWrapper
from src.program_model import ProgramSandbox, ProgramWorldModelPredictor, parse_state_json
from src.ui.baba_manual_player_ui import ActionLogEntry
from src.visualization import resolve_predicted_visual_state


def _resolve_env_board_state(
    env: BabaWrapper,
) -> Dict[str, Any]:
    return dict(env.get_visualization_state())


def _resolve_predicted_board_state(
    *,
    predicted_state_obj: Optional[Dict[str, Any]],
    actual_state_obj: Optional[Dict[str, Any]] = None,
    actual_board_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return resolve_predicted_visual_state(
        predicted_logical_state=predicted_state_obj,
        actual_logical_state=actual_state_obj,
        actual_visual_state=actual_board_state,
    )


def _default_output_dir(
    *,
    experiment_dir: Optional[Path],
    program_path: Path,
) -> Path:
    if experiment_dir is not None:
        return experiment_dir / "manual_transition_images"
    return program_path.parent / "manual_transition_images"


def _normalize_scenario_type(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_scenario_catalog(wrapper: BabaWrapper) -> tuple[str, ...]:
    base_env = getattr(wrapper, "env", None)
    if base_env is None:
        return ()
    raw_catalog = getattr(base_env, "SCENARIO_TYPES", ())
    if not isinstance(raw_catalog, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in raw_catalog if str(item).strip())


def _resolve_scenario_display_name(wrapper: BabaWrapper) -> Optional[str]:
    base_env = getattr(wrapper, "env", None)
    if base_env is None:
        return None
    active_scenario = _normalize_scenario_type(getattr(base_env, "scenario_type", None))
    if active_scenario is None:
        return None
    return _resolve_named_scenario_display_name(wrapper, active_scenario)


def _resolve_named_scenario_display_name(
    wrapper: BabaWrapper,
    scenario_type: Optional[str],
) -> Optional[str]:
    normalized = _normalize_scenario_type(scenario_type)
    if normalized is None:
        return None
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


def _has_prediction_diff(difference_summary: Optional[str]) -> bool:
    normalized = str(difference_summary or "").strip().lower()
    return bool(normalized) and normalized != "diff: no visible state difference"


def _normalize_map_rows(rows: Any) -> list[Dict[str, Any]]:
    normalized_rows: list[Dict[str, Any]] = []
    if not isinstance(rows, (list, tuple)):
        return normalized_rows
    for index, row in enumerate(rows):
        scenario_type = row.get("scenario_type") if isinstance(row, dict) else None
        normalized_scenario = _normalize_scenario_type(scenario_type)
        raw_label = row.get("label") if isinstance(row, dict) else None
        label = str(raw_label).strip() if isinstance(raw_label, str) else ""
        if not label:
            label = normalized_scenario or f"map {int(index) + 1}"
        normalized_rows.append(
            {
                "index": int(index),
                "scenarioType": normalized_scenario,
                "label": label,
            }
        )
    return normalized_rows


def _build_wrapper_map_rows(wrapper: BabaWrapper, *, seed: int) -> list[Dict[str, Any]]:
    list_graph_search_worlds = getattr(wrapper, "list_graph_search_worlds", None)
    rows = (
        _normalize_map_rows(list_graph_search_worlds(seed=int(seed)))
        if callable(list_graph_search_worlds)
        else []
    )
    if rows:
        return rows

    scenario_catalog = _resolve_scenario_catalog(wrapper)
    if scenario_catalog:
        return [
            {
                "index": int(index),
                "scenarioType": scenario_type,
                "label": _resolve_named_scenario_display_name(wrapper, scenario_type) or scenario_type,
            }
            for index, scenario_type in enumerate(scenario_catalog)
        ]

    fallback_label = _resolve_scenario_display_name(wrapper) or f"seed={int(seed)}"
    return [
        {
            "index": 0,
            "scenarioType": _normalize_scenario_type(getattr(getattr(wrapper, "env", None), "scenario_type", None)),
            "label": str(fallback_label),
        }
    ]


def _discover_available_maps(
    *,
    env_name: str,
    max_steps: int,
    render_mode: Optional[str],
    seed: int,
    env_kwargs: Dict[str, Any],
    word_aliases: Optional[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    catalog_wrapper = BabaWrapper(
        env_name=str(env_name),
        max_steps=int(max_steps),
        render_mode=render_mode,
        state_format="json",
        seed=int(seed),
        env_kwargs=dict(env_kwargs),
        word_aliases=word_aliases,
    )
    try:
        return _build_wrapper_map_rows(catalog_wrapper, seed=int(seed))
    finally:
        catalog_wrapper.close()


def _relax_compare_map_filters(
    *,
    env_name: str,
    env_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    relaxed_env_kwargs = dict(env_kwargs)
    normalized_env_name = str(env_name).strip()
    if not normalized_env_name.startswith("env/baba_custom_ascii"):
        return relaxed_env_kwargs

    relaxed_env_kwargs.pop("scenario_split", None)
    relaxed_env_kwargs.pop("allowed_scenario_types", None)
    return relaxed_env_kwargs


def _resolve_initial_compare_scenario_type(
    *,
    env_name: str,
    max_steps: int,
    render_mode: Optional[str],
    seed: int,
    env_kwargs: Dict[str, Any],
    word_aliases: Optional[Dict[str, Any]],
) -> Optional[str]:
    probe_wrapper = BabaWrapper(
        env_name=str(env_name),
        max_steps=int(max_steps),
        render_mode=render_mode,
        state_format="json",
        seed=int(seed),
        env_kwargs=dict(env_kwargs),
        word_aliases=word_aliases,
    )
    try:
        probe_wrapper.reset(seed=int(seed))
        return _normalize_scenario_type(
            getattr(getattr(probe_wrapper, "env", None), "scenario_type", None)
        )
    finally:
        probe_wrapper.close()


@dataclass(frozen=True)
class CompareCheckpoint:
    snapshot: Dict[str, Any]
    scenario_type: Optional[str]
    current_seed: int
    actual_current_state_json: str
    predicted_current_state_json: str
    step_index: int
    episode_done: bool
    last_frame: CompareFrame


@dataclass
class CompareFrame:
    step_index: int
    action_name: str
    reward: float
    terminated: bool
    truncated: bool
    prediction_error: Optional[str]
    previous_state_raw: str
    previous_state: Dict[str, Any]
    actual_next_state_raw: str
    actual_next_state: Dict[str, Any]
    previous_predicted_state_raw: str
    previous_predicted_state: Dict[str, Any]
    predicted_next_state_raw: Optional[str]
    predicted_next_state: Optional[Dict[str, Any]]
    predictor_mode: str
    status_lines: list[str]
    seed: int
    difference_summary: Optional[str]
    has_prediction_diff: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stepIndex": int(self.step_index),
            "action": self.action_name,
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "predictionError": self.prediction_error,
            "predictorMode": self.predictor_mode,
            "statusLines": list(self.status_lines),
            "seed": int(self.seed),
            "differenceSummary": self.difference_summary,
            "hasPredictionDiff": bool(self.has_prediction_diff),
            "gtBeforeState": dict(self.previous_state),
            "gtAfterState": dict(self.actual_next_state),
            "predBeforeState": dict(self.previous_predicted_state),
            "predAfterState": dict(self.predicted_next_state)
            if isinstance(self.predicted_next_state, dict)
            else None,
        }


@dataclass
class CompareSession:
    session_id: str
    experiment_name: str
    version_tag: str
    program_path: Path
    exp_cfg_path: Path
    env_cfg_path: Path
    output_dir: Path
    predictor_mode: str
    current_seed: int
    predictor: ProgramWorldModelPredictor
    env: BabaWrapper
    visual_config: Dict[str, Any]
    env_name: str = ""
    max_steps: int = 0
    render_mode: Optional[str] = None
    base_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    word_aliases: Optional[Dict[str, Any]] = None
    scenario_type: Optional[str] = None
    available_maps: list[Dict[str, Any]] = field(default_factory=list)
    action_history: list[ActionLogEntry] = field(default_factory=list)
    _timeline: list[CompareCheckpoint] = field(default_factory=list, repr=False)
    experiment_dir: Optional[Path] = None
    actual_current_state_json: str = ""
    predicted_current_state_json: str = ""
    step_index: int = 0
    episode_done: bool = False
    last_frame: Optional[CompareFrame] = None

    @classmethod
    def create(
        cls,
        *,
        experiment: Optional[str],
        version: Optional[str],
        program_file: Optional[str],
        env_config: str,
        experiment_config: str,
        seed: int,
        output_dir: Optional[str] = None,
    ) -> "CompareSession":
        exp_dir: Optional[Path] = None
        if experiment:
            exp_dir = resolve_experiment_dir(experiment)

        if program_file:
            program_path = resolve_path(program_file)
            if not program_path.exists():
                raise FileNotFoundError(f"Program file not found: {program_path}")
            if version:
                try:
                    version_tag = normalize_version_tag(version)
                except ValueError:
                    version_tag = str(version).strip()
            else:
                version_tag = program_path.stem
        else:
            if exp_dir is None:
                raise ValueError("Provide `experiment` or `program_file`.")
            if not version:
                raise ValueError("`version` is required when `program_file` is omitted.")
            version_tag = normalize_version_tag(version)
            program_path = resolve_program_path(exp_dir, version_tag=version_tag)

        if exp_dir is not None:
            exp_cfg_path = resolve_config_from_experiment_snapshot(
                exp_dir=exp_dir,
                cli_value=experiment_config,
                snapshot_filename="experiment_config.yaml",
                default_relative_path="configs/experiment_config_online.yaml",
            )
            env_cfg_path = resolve_config_from_experiment_snapshot(
                exp_dir=exp_dir,
                cli_value=env_config,
                snapshot_filename="env_config.yaml",
                default_relative_path="configs/env_config.yaml",
            )
            experiment_name = exp_dir.name
        else:
            exp_cfg_path = resolve_path(experiment_config)
            env_cfg_path = resolve_path(env_config)
            experiment_name = f"standalone:{program_path.parent.name}"

        exp_cfg = load_yaml(exp_cfg_path)
        env_cfg = load_yaml(env_cfg_path)
        output_path = (
            resolve_path(output_dir)
            if isinstance(output_dir, str) and output_dir.strip()
            else _default_output_dir(experiment_dir=exp_dir, program_path=program_path)
        )

        sandbox_config = build_sandbox_config(exp_cfg)
        sandbox = ProgramSandbox(config=sandbox_config)
        predictor = ProgramWorldModelPredictor(sandbox=sandbox)
        predictor.set_program_source(program_path.read_text(encoding="utf-8"))

        predictor_mode = "program_code"

        env_kwargs = env_cfg.get("environment", {}).get("kwargs", {})
        if not isinstance(env_kwargs, dict):
            raise ValueError("environment.kwargs must be a mapping if provided.")
        filtered_env_kwargs = copy.deepcopy(env_kwargs)
        initial_scenario_type = _normalize_scenario_type(filtered_env_kwargs.pop("scenario_type", None))
        resolved_env_name = str(env_cfg["environment"]["name"])
        resolved_max_steps = int(env_cfg["environment"]["max_steps"])
        resolved_render_mode = env_cfg["environment"].get("render_mode")
        word_aliases = exp_cfg.get("serialization", {}).get("word_aliases")
        base_env_kwargs = _relax_compare_map_filters(
            env_name=resolved_env_name,
            env_kwargs=filtered_env_kwargs,
        )
        if initial_scenario_type is None and base_env_kwargs != filtered_env_kwargs:
            initial_scenario_type = _resolve_initial_compare_scenario_type(
                env_name=resolved_env_name,
                max_steps=resolved_max_steps,
                render_mode=resolved_render_mode,
                seed=int(seed),
                env_kwargs=filtered_env_kwargs,
                word_aliases=word_aliases,
            )
        available_maps = _discover_available_maps(
            env_name=resolved_env_name,
            max_steps=resolved_max_steps,
            render_mode=resolved_render_mode,
            seed=int(seed),
            env_kwargs=base_env_kwargs,
            word_aliases=word_aliases,
        )

        initial_env_kwargs = dict(base_env_kwargs)
        if initial_scenario_type is not None:
            initial_env_kwargs["scenario_type"] = initial_scenario_type
        env = BabaWrapper(
            env_name=resolved_env_name,
            max_steps=resolved_max_steps,
            render_mode=resolved_render_mode,
            state_format="json",
            seed=int(seed),
            env_kwargs=initial_env_kwargs,
            word_aliases=word_aliases,
        )

        session = cls(
            session_id=uuid.uuid4().hex,
            experiment_name=experiment_name,
            version_tag=version_tag,
            program_path=program_path,
            exp_cfg_path=exp_cfg_path,
            env_cfg_path=env_cfg_path,
            output_dir=output_path,
            predictor_mode=predictor_mode,
            current_seed=int(seed),
            predictor=predictor,
            env=env,
            visual_config=env.get_visualization_config(),
            env_name=resolved_env_name,
            max_steps=resolved_max_steps,
            render_mode=resolved_render_mode,
            base_env_kwargs=base_env_kwargs,
            word_aliases=word_aliases,
            scenario_type=initial_scenario_type,
            available_maps=available_maps,
            experiment_dir=exp_dir,
        )
        session.reset(seed=int(seed), scenario_type=initial_scenario_type)
        return session

    def _build_env(self, *, scenario_type: Optional[str], seed: int) -> BabaWrapper:
        env_kwargs = dict(self.base_env_kwargs)
        normalized_scenario = _normalize_scenario_type(scenario_type)
        if normalized_scenario is not None:
            env_kwargs["scenario_type"] = normalized_scenario
        return BabaWrapper(
            env_name=str(self.env_name),
            max_steps=int(self.max_steps),
            render_mode=self.render_mode,
            state_format="json",
            seed=int(seed),
            env_kwargs=env_kwargs,
            word_aliases=self.word_aliases,
        )

    def reset(self, *, seed: int, scenario_type: Optional[str] = None) -> None:
        normalized_scenario = self.scenario_type if scenario_type is None else _normalize_scenario_type(scenario_type)
        if self.env is None or normalized_scenario != self.scenario_type:
            if self.env is not None:
                self.env.close()
            self.env = self._build_env(scenario_type=normalized_scenario, seed=int(seed))
        self.scenario_type = normalized_scenario
        self.current_seed = int(seed)
        self.actual_current_state_json = self.env.reset(seed=self.current_seed)
        self.predicted_current_state_json = self.actual_current_state_json
        self.step_index = 0
        self.episode_done = False
        self.visual_config = self.env.get_visualization_config()
        reset_state_obj = _resolve_env_board_state(self.env)
        self.last_frame = CompareFrame(
            step_index=0,
            action_name="reset",
            reward=0.0,
            terminated=False,
            truncated=False,
            prediction_error=None,
            previous_state_raw=self.actual_current_state_json,
            previous_state=reset_state_obj,
            actual_next_state_raw=self.actual_current_state_json,
            actual_next_state=reset_state_obj,
            previous_predicted_state_raw=self.actual_current_state_json,
            previous_predicted_state=dict(reset_state_obj),
            predicted_next_state_raw=self.actual_current_state_json,
            predicted_next_state=dict(reset_state_obj),
            predictor_mode=self.predictor_mode,
            status_lines=[
                "controls: arrows move | space idle | z undo | r reset | t next map | s save",
                "reset complete | web compare session ready",
                f"seed={self.current_seed}",
                f"predictor_mode={self.predictor_mode}",
            ],
            seed=self.current_seed,
            difference_summary="Diff: no visible state difference",
            has_prediction_diff=False,
        )
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
        scenario_catalog = self._selectable_scenario_types()
        if not scenario_catalog:
            raise RuntimeError("this compare session does not expose multiple maps.")
        current = _normalize_scenario_type(self.scenario_type)
        if current in scenario_catalog:
            next_index = (scenario_catalog.index(current) + 1) % len(scenario_catalog)
        else:
            next_index = 0
        self.reset(seed=int(self.current_seed), scenario_type=scenario_catalog[next_index])

    def select_scenario(self, *, scenario_type: Optional[str]) -> None:
        normalized = _normalize_scenario_type(scenario_type)
        scenario_catalog = self._selectable_scenario_types()
        if scenario_catalog and normalized not in scenario_catalog:
            available = "\n".join(f"  - {value}" for value in scenario_catalog)
            raise ValueError(
                f"Unknown scenario `{scenario_type}`.\nAvailable scenarios:\n{available}"
            )
        self.reset(seed=int(self.current_seed), scenario_type=normalized)

    def step(self, *, action_name: str) -> None:
        if self.episode_done:
            raise RuntimeError("episode ended; reset the compare session first.")
        normalized_action = str(action_name).strip().lower()
        if normalized_action not in ACTION_TO_ID:
            raise ValueError(f"Unsupported action `{action_name}`.")

        previous_actual_state_json = self.actual_current_state_json
        previous_predicted_state_json = self.predicted_current_state_json
        previous_actual_board_state = _resolve_env_board_state(self.env)
        previous_actual_obj = parse_state_json(previous_actual_state_json)
        previous_predicted_obj = parse_state_json(previous_predicted_state_json)
        if (
            self.last_frame is not None
            and self.last_frame.predicted_next_state_raw == previous_predicted_state_json
            and isinstance(self.last_frame.predicted_next_state, dict)
        ):
            previous_predicted_display_state = dict(self.last_frame.predicted_next_state)
        else:
            previous_predicted_display_state = _resolve_predicted_board_state(
                predicted_state_obj=previous_predicted_obj,
                actual_state_obj=previous_actual_obj,
                actual_board_state=previous_actual_board_state,
            ) or dict(previous_predicted_obj)

        prediction = self.predictor.predict(previous_predicted_state_json, normalized_action)
        predicted_next_state_raw = (
            prediction.predicted_state_json
            if isinstance(prediction.predicted_state_json, str)
            else None
        )
        predicted_next_state_obj = None
        prediction_error_msg = None
        predictor_extra_lines = [f"predictor_mode={self.predictor_mode}"]
        if predicted_next_state_raw is not None:
            try:
                predicted_next_state_obj = parse_state_json(predicted_next_state_raw)
            except ValueError as exc:
                prediction_error_msg = str(exc)
                predicted_next_state_raw = None
        else:
            prediction_error_msg = (
                prediction.error.message if prediction.error is not None else "unknown prediction error"
            )

        next_state_json, reward, terminated, truncated, info = self.env.step(
            ACTION_TO_ID[normalized_action]
        )
        self.actual_current_state_json = next_state_json
        actual_next_obj = parse_state_json(next_state_json)
        actual_next_board_state = _resolve_env_board_state(self.env)

        if predicted_next_state_raw is not None and predicted_next_state_obj is not None:
            self.predicted_current_state_json = predicted_next_state_raw
        predicted_next_display_state = _resolve_predicted_board_state(
            predicted_state_obj=predicted_next_state_obj,
            actual_state_obj=actual_next_obj,
            actual_board_state=actual_next_board_state,
        )
        difference_summary = build_comparison_difference_summary(
            expected_state=actual_next_obj,
            predicted_state=predicted_next_state_obj,
            prediction_error=prediction_error_msg,
        )

        self.step_index += 1
        self.episode_done = bool(terminated or truncated)
        status_lines = build_status_lines(
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            prediction_error=prediction_error_msg,
            episode_done=self.episode_done,
            extra_lines=predictor_extra_lines,
        )
        self.last_frame = CompareFrame(
            step_index=int(self.step_index),
            action_name=normalized_action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            prediction_error=prediction_error_msg,
            previous_state_raw=previous_actual_state_json,
            previous_state=previous_actual_board_state,
            actual_next_state_raw=next_state_json,
            actual_next_state=actual_next_board_state,
            previous_predicted_state_raw=previous_predicted_state_json,
            previous_predicted_state=previous_predicted_display_state,
            predicted_next_state_raw=predicted_next_state_raw,
            predicted_next_state=predicted_next_display_state,
            predictor_mode=self.predictor_mode,
            status_lines=status_lines,
            seed=self.current_seed,
            difference_summary=difference_summary,
            has_prediction_diff=_has_prediction_diff(difference_summary),
        )
        note = ""
        if prediction_error_msg:
            note = f"prediction: {prediction_error_msg}"
        self.action_history.append(
            ActionLogEntry(
                step_index=int(self.step_index),
                action_name=normalized_action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                note=note,
            )
        )
        self._timeline.append(self._capture_checkpoint())

    def undo_last_action(self) -> None:
        if len(self._timeline) <= 1:
            raise RuntimeError("there is no compare action to undo.")
        self._timeline.pop()
        if self.action_history and self.action_history[-1].event_type == "action":
            self.action_history.pop()
        checkpoint = self._timeline[-1]
        self.env.restore(checkpoint.snapshot, refresh_cache=True)
        self._apply_checkpoint(checkpoint)

    def save_current_frame(self) -> Dict[str, Any]:
        if self.last_frame is None:
            raise RuntimeError("No compare frame available.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image = compose_comparison_image(
            previous_state=self.last_frame.previous_state,
            actual_next_state=self.last_frame.actual_next_state,
            predicted_next_state=self.last_frame.predicted_next_state,
            action_name=self.last_frame.action_name,
            experiment_name=self.experiment_name,
            version_tag=self.version_tag,
            prediction_error=self.last_frame.prediction_error,
            step_index=self.last_frame.step_index,
            status_lines=self.last_frame.status_lines,
            difference_summary=self.last_frame.difference_summary,
            visual_config=self.visual_config,
        )
        try:
            payload = {
                "experiment": self.experiment_name,
                "version": self.version_tag,
                "step_index": self.last_frame.step_index,
                "action": self.last_frame.action_name,
                "reward": self.last_frame.reward,
                "terminated": self.last_frame.terminated,
                "truncated": self.last_frame.truncated,
                "seed": self.last_frame.seed,
                "previous_state_raw": self.last_frame.previous_state_raw,
                "actual_next_state_raw": self.last_frame.actual_next_state_raw,
                "previous_predicted_state_raw": self.last_frame.previous_predicted_state_raw,
                "predicted_next_state_raw": self.last_frame.predicted_next_state_raw,
                "prediction_error": self.last_frame.prediction_error,
                "predictor_mode": self.last_frame.predictor_mode,
            }
            image_path, json_path = save_transition_artifacts(
                output_dir=self.output_dir,
                step_index=self.last_frame.step_index,
                action_name=self.last_frame.action_name,
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
        if self.last_frame is None:
            raise RuntimeError("Compare session is not initialized.")
        scenario_display_name = _resolve_scenario_display_name(self.env)
        maps = self._build_map_catalog()
        return {
            "sessionId": self.session_id,
            "experimentName": self.experiment_name,
            "versionTag": self.version_tag,
            "programPath": str(self.program_path),
            "experimentConfigPath": str(self.exp_cfg_path),
            "envConfigPath": str(self.env_cfg_path),
            "predictorMode": self.predictor_mode,
            "outputDir": str(self.output_dir),
            "episodeDone": bool(self.episode_done),
            "seed": int(self.current_seed),
            "scenarioType": self.scenario_type,
            "scenarioDisplayName": scenario_display_name,
            "maps": maps,
            "predictionDiffSummary": self.last_frame.difference_summary,
            "hasPredictionDiff": bool(self.last_frame.has_prediction_diff),
            "currentMapLabel": next(
                (str(entry.get("label")) for entry in maps if bool(entry.get("isActive"))),
                scenario_display_name or self.scenario_type or f"seed={int(self.current_seed)}",
            ),
            "canUndo": len(self._timeline) > 1,
            "actionHistory": [
                _serialize_action_log_entry(entry)
                for entry in self.action_history
            ],
            "visualConfig": dict(self.visual_config),
            "frame": self.last_frame.to_payload(),
        }

    def close(self) -> None:
        self.env.close()

    def _capture_checkpoint(self) -> CompareCheckpoint:
        if self.last_frame is None:
            raise RuntimeError("Compare session is not initialized.")
        return CompareCheckpoint(
            snapshot=self.env.snapshot(),
            scenario_type=self.scenario_type,
            current_seed=int(self.current_seed),
            actual_current_state_json=str(self.actual_current_state_json),
            predicted_current_state_json=str(self.predicted_current_state_json),
            step_index=int(self.step_index),
            episode_done=bool(self.episode_done),
            last_frame=copy.deepcopy(self.last_frame),
        )

    def _apply_checkpoint(self, checkpoint: CompareCheckpoint) -> None:
        self.scenario_type = checkpoint.scenario_type
        self.current_seed = int(checkpoint.current_seed)
        self.actual_current_state_json = str(checkpoint.actual_current_state_json)
        self.predicted_current_state_json = str(checkpoint.predicted_current_state_json)
        self.step_index = int(checkpoint.step_index)
        self.episode_done = bool(checkpoint.episode_done)
        self.visual_config = self.env.get_visualization_config()
        self.last_frame = copy.deepcopy(checkpoint.last_frame)

    def _build_map_catalog(self) -> list[Dict[str, Any]]:
        resolved_active = _normalize_scenario_type(self.scenario_type)
        source_rows = self.available_maps or _build_wrapper_map_rows(self.env, seed=int(self.current_seed))
        rows: list[Dict[str, Any]] = []
        for index, row in enumerate(source_rows):
            scenario_type = _normalize_scenario_type(row.get("scenarioType") if isinstance(row, dict) else None)
            label = str(row.get("label") if isinstance(row, dict) else "").strip()
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
            if resolved_active is not None and all(row["scenarioType"] != resolved_active for row in rows):
                rows.append(
                    {
                        "index": int(len(rows)),
                        "scenarioType": resolved_active,
                        "label": _resolve_scenario_display_name(self.env) or resolved_active,
                        "isActive": True,
                    }
                )
            return rows

        fallback_label = (
            _resolve_scenario_display_name(self.env)
            or self.scenario_type
            or f"seed={int(self.current_seed)}"
        )
        return [
            {
                "index": 0,
                "scenarioType": self.scenario_type,
                "label": str(fallback_label),
                "isActive": True,
            }
        ]

    def _selectable_scenario_types(self) -> tuple[str, ...]:
        rows = self.available_maps or _build_wrapper_map_rows(self.env, seed=int(self.current_seed))
        normalized = [
            _normalize_scenario_type(row.get("scenarioType") if isinstance(row, dict) else None)
            for row in rows
        ]
        return tuple(
            scenario_type
            for scenario_type in dict.fromkeys(normalized)
            if scenario_type is not None
        )
