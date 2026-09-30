from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collect.data_coverage import (
    DEFAULT_COVERAGE_DIFFICULTY,
    DEFAULT_COVERAGE_SCENARIO_SPLIT,
    DEFAULT_COVERAGE_SPLIT_MANIFEST,
    TransitionRecord,
    _build_transition_archive_payload,
    _format_duration,
    _resolve_project_path,
    _sanitize_name,
    _state_archive_input_from_state_obj,
    _state_key_from_archive_input,
    _write_solver_summary,
    _write_state_archive_npz,
    _write_transition_rows,
)
from src.environments import BabaWrapper
from src.environments.custom_map_spec import load_custom_map_catalog
from src.player_runtime import ACTION_TO_ID, resolve_env_selection
from src.web.common import html_file_response, mount_shared_static
from src.web.launcher import PLAYER_WEB_PORT, run_app_server
from src.web.player_sessions import DEFAULT_RESET_SEED, PlayerSession, resolve_scenario_display_name


DEFAULT_SOLUTION_OUTPUT_ROOT = Path("test_dataset") / "solution"
DEFAULT_SOLUTION_ENV_ID = "env/baba_custom_ascii_original"
UNLIMITED_ENV_MAX_STEPS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class SolutionStepRecord:
    previous_state_raw: str
    previous_state_obj: Dict[str, Any]
    action_name: str
    action_id: int
    reward: float
    terminated: bool
    truncated: bool
    next_state_raw: str
    next_state_obj: Dict[str, Any]


def _resolve_state_obj(
    *,
    state_raw: Optional[str],
    state_obj: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if isinstance(state_obj, Mapping):
        return dict(state_obj)
    raise ValueError("Current episode is missing a usable state object.")


def _solution_state_to_archive_input(
    *,
    state_raw: Optional[str],
    state_obj: Optional[Mapping[str, Any]],
    source: str,
    event: str,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> Dict[str, Any]:
    _ = state_raw
    resolved_state_obj = _resolve_state_obj(state_raw=state_raw, state_obj=state_obj)
    archive_input = _state_archive_input_from_state_obj(resolved_state_obj)
    archive_input["source"] = str(source).strip() or "human_solution_live"
    archive_input["event"] = str(event).strip()
    archive_input["reward"] = float(reward)
    archive_input["terminated"] = bool(terminated)
    archive_input["truncated"] = bool(truncated)
    return archive_input


def _materialize_episode_solution_payload(
    step_records: List[SolutionStepRecord],
) -> tuple[List[TransitionRecord], Dict[Any, Dict[str, Any]]]:
    transition_records: List[TransitionRecord] = []
    state_archive_inputs: Dict[Any, Dict[str, Any]] = {}
    transition_keys: set[tuple[Any, str, Any]] = set()

    for record in step_records:
        previous_archive_input = _solution_state_to_archive_input(
            state_raw=record.previous_state_raw,
            state_obj=record.previous_state_obj,
            source="human_solution_live",
            event="manual_solution_previous",
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        next_archive_input = _solution_state_to_archive_input(
            state_raw=record.next_state_raw,
            state_obj=record.next_state_obj,
            source="human_solution_live",
            event=str(record.action_name),
            reward=float(record.reward),
            terminated=bool(record.terminated),
            truncated=bool(record.truncated),
        )
        state_key = _state_key_from_archive_input(previous_archive_input)
        next_state_key = _state_key_from_archive_input(next_archive_input)
        state_archive_inputs.setdefault(state_key, previous_archive_input)
        state_archive_inputs.setdefault(next_state_key, next_archive_input)

        transition_key = (state_key, str(record.action_name), next_state_key)
        if transition_key in transition_keys:
            continue
        transition_keys.add(transition_key)
        transition_records.append(
            TransitionRecord(
                state_key=state_key,
                action=str(record.action_name),
                next_state_key=next_state_key,
                reward=float(record.reward),
                done=bool(record.terminated or record.truncated),
            )
        )

    return transition_records, state_archive_inputs


class PlayerActionRequest(BaseModel):
    action: str


class PlayerResetRequest(BaseModel):
    seed: int = DEFAULT_RESET_SEED


@dataclass
class SolutionCollectorSession(PlayerSession):
    difficulty: str = DEFAULT_COVERAGE_DIFFICULTY
    scenario_split: str = DEFAULT_COVERAGE_SCENARIO_SPLIT
    split_manifest_path: Path = field(default_factory=lambda: _resolve_project_path(DEFAULT_COVERAGE_SPLIT_MANIFEST))
    output_root: Path = field(default_factory=lambda: _resolve_project_path(DEFAULT_SOLUTION_OUTPUT_ROOT))
    target_catalog: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_scenarios: tuple[str, ...] = ()
    current_target_index: int = 0
    skip_existing: bool = True
    env_max_steps: Optional[int] = None
    current_episode_steps: List[SolutionStepRecord] = field(default_factory=list)
    current_solution_saved: bool = False
    saved_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_solution_info: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        *,
        difficulty: str,
        scenario_split: str,
        split_manifest_path: str | Path,
        requested_scenario_type: Optional[str],
        seed: int,
        output_root: str | Path,
        skip_existing: bool,
        env_max_steps: Optional[int],
    ) -> "SolutionCollectorSession":
        resolved_manifest_path = _resolve_project_path(split_manifest_path)
        resolved_output_root = _resolve_project_path(output_root)
        resolved_output_root.mkdir(parents=True, exist_ok=True)

        requested_scenarios = None
        if isinstance(requested_scenario_type, str) and requested_scenario_type.strip():
            requested_scenarios = [str(requested_scenario_type).strip()]

        catalog = load_custom_map_catalog(
            difficulty,
            scenario_split=scenario_split,
            allowed_scenario_types=requested_scenarios,
            split_manifest_path=resolved_manifest_path,
        )
        if not catalog:
            raise ValueError(f"No custom maps found for difficulty `{difficulty}` split `{scenario_split}`.")

        env_index, env_id = resolve_env_selection(DEFAULT_SOLUTION_ENV_ID)
        session = cls(
            env_id=env_id,
            env_index=env_index,
            seed=int(seed),
            scenario_type=None,
            autosave_enabled=False,
            difficulty=str(difficulty),
            scenario_split=str(scenario_split),
            split_manifest_path=resolved_manifest_path,
            output_root=resolved_output_root,
            target_catalog={str(name): dict(value) for name, value in catalog.items()},
            target_scenarios=tuple(str(name) for name in catalog.keys()),
            skip_existing=bool(skip_existing),
            env_max_steps=(max(1, int(env_max_steps)) if env_max_steps is not None else None),
        )
        session._refresh_saved_results()
        start_scenario = session._resolve_initial_scenario(requested_scenario_type=requested_scenario_type)
        session.current_target_index = session.target_scenarios.index(start_scenario)
        session.reset(seed=int(seed), scenario_type=start_scenario)
        if session.collection_complete:
            session._set_status_message(
                f"All {len(session.target_scenarios)} target scenarios already have saved solutions. "
                "Run with `--no-skip-existing` to replay them."
            )
        else:
            session._set_status_message(
                f"Loaded {len(session.target_scenarios)} target scenarios. "
                "Use arrows or the buttons to play, `S` to save a solved run, and `T` to move on."
            )
        session._write_batch_summary()
        return session

    @property
    def collection_complete(self) -> bool:
        return bool(self.skip_existing) and all(
            str(scenario) in self.saved_results for scenario in self.target_scenarios
        )

    def _set_status_message(self, message: str, **extra: Any) -> None:
        payload = {"message": str(message).strip()}
        for key, value in extra.items():
            payload[str(key)] = value
        self.last_solution_info = payload

    def _build_wrapper(self, scenario_type: Optional[str]) -> BabaWrapper:
        env_kwargs: Dict[str, Any] = {}
        if isinstance(scenario_type, str) and scenario_type.strip():
            env_kwargs["scenario_type"] = scenario_type.strip()
        return BabaWrapper(
            env_name=self.env_id,
            max_steps=int(self._resolved_env_max_steps()),
            render_mode=None,
            state_format="json",
            seed=int(self.seed),
            env_kwargs=env_kwargs,
        )

    def _resolved_env_max_steps(self) -> int:
        if self.env_max_steps is None:
            return int(UNLIMITED_ENV_MAX_STEPS)
        return max(1, int(self.env_max_steps))

    def _refresh_saved_results(self) -> None:
        self.saved_results = {}
        for scenario_type in self.target_scenarios:
            summary_path = self.output_root / f"{_sanitize_name(scenario_type)}.json"
            if not summary_path.exists():
                continue
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, Mapping) and bool(payload.get("solved", False)):
                self.saved_results[str(scenario_type)] = dict(payload)

    def _resolve_initial_scenario(self, *, requested_scenario_type: Optional[str]) -> str:
        normalized_requested = (
            str(requested_scenario_type).strip()
            if isinstance(requested_scenario_type, str) and requested_scenario_type.strip()
            else None
        )
        if normalized_requested is not None:
            if normalized_requested not in self.target_catalog:
                available = ", ".join(self.target_scenarios)
                raise ValueError(
                    f"Requested scenario `{normalized_requested}` is not in the configured solution split. "
                    f"Available scenarios: {available}"
                )
            return normalized_requested
        if self.skip_existing:
            for scenario_type in self.target_scenarios:
                if str(scenario_type) not in self.saved_results:
                    return str(scenario_type)
            return str(self.target_scenarios[-1])
        return str(self.target_scenarios[0])

    def _find_next_target_index(self, *, start: int) -> Optional[int]:
        for index in range(max(0, int(start)), len(self.target_scenarios)):
            scenario_type = str(self.target_scenarios[index])
            if not self.skip_existing or scenario_type not in self.saved_results:
                return int(index)
        return None

    def _current_scenario_type(self) -> str:
        scenario_type = str(self.scenario_type or "").strip()
        if not scenario_type:
            raise RuntimeError("No active target scenario is selected.")
        return scenario_type

    def _current_spec(self) -> Dict[str, Any]:
        return dict(self.target_catalog[self._current_scenario_type()])

    def _is_current_solution_ready(self) -> bool:
        if bool(self.truncated):
            return False
        wrapper_env = getattr(self.wrapper, "env", None) if self.wrapper is not None else None
        if bool(getattr(wrapper_env, "is_win", False)):
            return True
        return bool(self.terminated and float(self.last_reward) > 0.0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        scenario_type: Optional[str] = None,
    ) -> None:
        super().reset(seed=seed, scenario_type=scenario_type)
        self.output_dir = self.output_root
        self.current_episode_steps = []
        self.current_solution_saved = False
        active_scenario = str(self.scenario_type or "").strip()
        if active_scenario in self.target_scenarios:
            self.current_target_index = self.target_scenarios.index(active_scenario)

    def step(self, *, action_name: str) -> None:
        previous_state_raw = self.current_state_raw
        previous_state_obj = dict(self.current_state_obj or {})
        super().step(action_name=action_name)
        current_state_raw = self.current_state_raw
        current_state_obj = dict(self.current_state_obj or {})
        self.current_episode_steps.append(
            SolutionStepRecord(
                previous_state_raw=str(previous_state_raw),
                previous_state_obj=previous_state_obj,
                action_name=str(self.last_action_name),
                action_id=int(self.last_action_id),
                reward=float(self.last_reward),
                terminated=bool(self.terminated),
                truncated=bool(self.truncated),
                next_state_raw=str(current_state_raw),
                next_state_obj=current_state_obj,
            )
        )
        if self._is_current_solution_ready() and not self.current_solution_saved:
            self._set_status_message(
                "Solved. Press `S` to save this scenario, or `T` to save and move to the next one."
            )
        elif bool(self.truncated):
            self._set_status_message("Step limit reached. Reset to retry, or press `T` to skip this scenario.")

    def save_current_solution(self) -> Dict[str, Any]:
        scenario_type = self._current_scenario_type()
        if not self.current_episode_steps:
            raise RuntimeError("No gameplay has been recorded for the current scenario yet.")
        if not self._is_current_solution_ready():
            raise RuntimeError("The current scenario is not solved yet, so there is no solution trajectory to save.")

        transition_records, state_archive_inputs = _materialize_episode_solution_payload(self.current_episode_steps)
        state_payloads, transition_rows = _build_transition_archive_payload(
            transition_records,
            state_archive_inputs=state_archive_inputs,
        )
        artifact_stem = _sanitize_name(scenario_type)
        states_path = _write_state_archive_npz(
            path=self.output_root / f"{artifact_stem}_states.npz",
            state_payloads=state_payloads,
        )
        transitions_path = _write_transition_rows(
            path=self.output_root / f"{artifact_stem}_transitions.jsonl",
            transition_rows=transition_rows,
        )
        spec = self._current_spec()["spec"]
        summary_payload = {
            "source": "human_solution_live",
            "env_name": self.env_id,
            "seed": int(self.seed),
            "difficulty": str(spec["difficulty"]),
            "scenario_split": str(self.scenario_split),
            "scenario_type": scenario_type,
            "map_path": str(self._current_spec()["path"]),
            "artifact_stem": artifact_stem,
            "solved": True,
            "transition_count": int(len(transition_records)),
            "raw_transition_count": int(len(self.current_episode_steps)),
            "visited_states": int(len(state_payloads)),
            "stop_reason": "saved_live_human_solution",
            "transitions_path": str(transitions_path),
            "states_path": str(states_path),
        }
        _write_solver_summary(path=self.output_root / f"{artifact_stem}.json", payload=summary_payload)
        self.saved_results[scenario_type] = dict(summary_payload)
        self.current_solution_saved = True
        self._set_status_message(
            f"Saved solution for `{scenario_type}`.",
            scenarioType=scenario_type,
            transitionsPath=str(transitions_path),
            statesPath=str(states_path),
        )
        self._write_batch_summary()
        return dict(self.last_solution_info or {})

    def _write_batch_summary(self) -> None:
        results: List[Dict[str, Any]] = []
        missing_scenarios: List[str] = []
        for index, scenario_type in enumerate(self.target_scenarios, start=1):
            spec_entry = self.target_catalog[str(scenario_type)]
            spec = spec_entry["spec"]
            saved = self.saved_results.get(str(scenario_type))
            if saved is None:
                missing_scenarios.append(str(scenario_type))
                results.append(
                    {
                        "scenario_index": int(index),
                        "scenario_type": str(scenario_type),
                        "difficulty": str(spec["difficulty"]),
                        "map_path": str(spec_entry["path"]),
                        "artifact_stem": _sanitize_name(str(scenario_type)),
                        "solved": False,
                        "transition_count": 0,
                        "visited_states": 0,
                        "raw_transition_count": 0,
                        "transitions_path": None,
                        "states_path": None,
                        "stop_reason": "not_yet_collected",
                    }
                )
                continue
            results.append(
                {
                    "scenario_index": int(index),
                    "scenario_type": str(scenario_type),
                    "difficulty": str(saved.get("difficulty", spec["difficulty"])),
                    "map_path": str(saved.get("map_path", spec_entry["path"])),
                    "artifact_stem": str(saved.get("artifact_stem", _sanitize_name(str(scenario_type)))),
                    "solved": bool(saved.get("solved", False)),
                    "transition_count": int(saved.get("transition_count", 0) or 0),
                    "visited_states": int(saved.get("visited_states", 0) or 0),
                    "raw_transition_count": int(saved.get("raw_transition_count", 0) or 0),
                    "transitions_path": saved.get("transitions_path"),
                    "states_path": saved.get("states_path"),
                    "stop_reason": str(saved.get("stop_reason", "saved_live_human_solution")),
                }
            )

        batch_payload = {
            "source": "human_solution_live",
            "difficulty": str(self.difficulty),
            "scenario_split": str(self.scenario_split),
            "split_manifest_path": str(self.split_manifest_path),
            "output_root": str(self.output_root),
            "scenario_count": int(len(self.target_scenarios)),
            "solved_count": int(len(self.saved_results)),
            "failed_count": int(len(self.target_scenarios) - len(self.saved_results)),
            "missing_scenarios": missing_scenarios,
            "elapsed_seconds": 0.0,
            "elapsed_human": _format_duration(0.0),
            "results": results,
        }
        _write_solver_summary(path=self.output_root / "batch_summary.json", payload=batch_payload)

    def advance_to_next_scenario(self) -> None:
        if self._is_current_solution_ready() and not self.current_solution_saved:
            self.save_current_solution()

        next_index = self._find_next_target_index(start=int(self.current_target_index) + 1)
        if next_index is None:
            self._set_status_message("All target scenarios are complete. You can close this window.")
            self._write_batch_summary()
            return
        self.current_target_index = int(next_index)
        self.reset(seed=int(self.seed), scenario_type=self.target_scenarios[next_index])
        self._set_status_message(
            f"Moved to scenario {int(next_index) + 1}/{len(self.target_scenarios)}: {self.scenario_type}"
        )

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        current_scenario = str(self.scenario_type or "").strip()
        payload.update(
            {
                "collectorMode": "solution",
                "difficulty": str(self.difficulty),
                "scenarioSplit": str(self.scenario_split),
                "scenarioIndex": int(self.current_target_index + 1),
                "scenarioCount": int(len(self.target_scenarios)),
                "solvedCount": int(len(self.saved_results)),
                "remainingCount": int(max(0, len(self.target_scenarios) - len(self.saved_results))),
                "collectionComplete": bool(self.collection_complete),
                "currentSolutionReady": bool(self._is_current_solution_ready()),
                "currentSolutionSaved": bool(self.current_solution_saved),
                "currentScenarioAlreadySaved": bool(current_scenario in self.saved_results),
                "outputRoot": str(self.output_root),
                "lastSolutionInfo": dict(self.last_solution_info) if isinstance(self.last_solution_info, dict) else None,
                "envMaxSteps": (int(self.env_max_steps) if self.env_max_steps is not None else None),
                "envMaxStepsLabel": (
                    str(int(self.env_max_steps)) if self.env_max_steps is not None else "unlimited"
                ),
                "actionCountRaw": int(len(self.current_episode_steps)),
            }
        )
        return payload


def create_solution_app(
    *,
    difficulty: str,
    scenario_split: str,
    split_manifest_path: str | Path,
    scenario_type: Optional[str],
    seed: int,
    output_root: str | Path,
    skip_existing: bool,
    env_max_steps: Optional[int],
) -> FastAPI:
    session = SolutionCollectorSession.create(
        difficulty=str(difficulty),
        scenario_split=str(scenario_split),
        split_manifest_path=split_manifest_path,
        requested_scenario_type=scenario_type,
        seed=int(seed),
        output_root=output_root,
        skip_existing=bool(skip_existing),
        env_max_steps=(int(env_max_steps) if env_max_steps is not None else None),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            session.close()

    app = FastAPI(title="BABA Solution Collector", lifespan=lifespan)
    app.state.session = session
    mount_shared_static(app)

    @app.get("/")
    def index():
        return html_file_response("solution_player.html")

    @app.get("/api/session")
    def get_session():
        return app.state.session.to_payload()

    @app.post("/api/session/actions")
    def step_session(request: PlayerActionRequest):
        try:
            app.state.session.step(action_name=request.action)
            return app.state.session.to_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/reset")
    def reset_session(request: PlayerResetRequest):
        try:
            app.state.session.reset(
                seed=int(request.seed),
                scenario_type=app.state.session.scenario_type,
            )
            app.state.session._set_status_message("Scenario reset. Play again to record a fresh solution.")
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/save-solution")
    def save_solution():
        try:
            app.state.session.save_current_solution()
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/next-scenario")
    def next_scenario():
        try:
            app.state.session.advance_to_next_scenario()
            return app.state.session.to_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively collect human solution trajectories for Baba test scenarios."
    )
    parser.add_argument("--difficulty", default=DEFAULT_COVERAGE_DIFFICULTY)
    parser.add_argument("--scenario-split", default=DEFAULT_COVERAGE_SCENARIO_SPLIT)
    parser.add_argument("--split-manifest", default=str(DEFAULT_COVERAGE_SPLIT_MANIFEST))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_RESET_SEED)
    parser.add_argument("--output-root", default=str(DEFAULT_SOLUTION_OUTPUT_ROOT))
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=None,
        help="Optional manual step cap for solution play. Defaults to unlimited.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=PLAYER_WEB_PORT)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Start from the requested/first scenario even if a saved solution already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_solution_app(
        difficulty=str(args.difficulty),
        scenario_split=str(args.scenario_split),
        split_manifest_path=args.split_manifest,
        scenario_type=(str(args.scenario).strip() if isinstance(args.scenario, str) and args.scenario.strip() else None),
        seed=int(args.seed),
        output_root=args.output_root,
        skip_existing=(not bool(args.no_skip_existing)),
        env_max_steps=(int(args.env_max_steps) if args.env_max_steps is not None else None),
    )
    session: SolutionCollectorSession = app.state.session

    print("Launching interactive solution collector")
    print(f"Difficulty: {session.difficulty}")
    print(f"Split: {session.scenario_split}")
    print(f"Target scenarios: {len(session.target_scenarios)}")
    print(f"Already saved: {len(session.saved_results)}")
    print(
        "Env max steps: "
        + (str(int(session.env_max_steps)) if session.env_max_steps is not None else "unlimited")
    )
    print(f"Output root: {session.output_root}")
    if session.scenario_type:
        print(
            f"Starting scenario: {session.scenario_type} "
            f"({session.current_target_index + 1}/{len(session.target_scenarios)})"
        )
    run_app_server(
        app=app,
        label="BABA Solution Collector",
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        open_browser=not bool(args.no_browser),
    )


if __name__ == "__main__":
    main()
