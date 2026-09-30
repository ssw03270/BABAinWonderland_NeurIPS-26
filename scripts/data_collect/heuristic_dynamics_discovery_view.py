from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import io
from pathlib import Path
import json
import sys
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

def _resolve_project_root() -> Path:
    """Resolve repository root regardless of invocation location."""
    script_path = Path(__file__).resolve()
    for candidate in script_path.parents:
        if (candidate / "src").exists() and (candidate / "scripts").exists():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "src").exists() and (cwd / "scripts").exists():
        return cwd
    return script_path.parents[2]


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from src.discovery.artifact_renderer import TransitionArtifactRenderer
from src.environments.config_bundle import load_baba_config_bundle
from src.visualization import build_visualization_config
from src.web.common import html_file_response, mount_shared_static
from src.web.launcher import run_app_server

from scripts.data_collect.data_coverage import _resolve_project_path  # noqa: E402


DEFAULT_DISCOVERY_JSON = Path("test_dataset") / "heuristic_dynamics_discovery.json"
DEFAULT_ENV_CONFIG = Path("configs") / "env_config.yaml"
DEFAULT_EXPERIMENT_CONFIG = Path("configs") / "experiment_config_online.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8004
DEFAULT_STATE_CACHE_SIZE = 512
DEFAULT_RENDER_CACHE_SIZE = 256


@dataclass(frozen=True)
class TransitionRef:
    bundle_id: int
    transition_index: int


@dataclass
class TransitionPayload:
    transition_index: int
    state_index: int
    next_state_index: int
    action: str
    reward: float
    done: bool
    terminated: bool
    truncated: bool


@dataclass
class BundleContext:
    bundle_id: int
    dataset_label: str
    dataset_root: str
    artifact_stem: str
    scenario_type: str
    transitions_path: Path
    states_path: Path
    transition_to_class: Dict[int, int]
    state_cache_size: int = DEFAULT_STATE_CACHE_SIZE

    transition_row_cache: Dict[int, TransitionPayload] = field(default_factory=dict)
    state_cache: OrderedDict[int, Dict[str, Any]] = field(default_factory=OrderedDict)

    _state_arrays: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _state_count: Optional[int] = field(default=None, init=False, repr=False)

    def _normalize_int(self, value: Any, *, fallback: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _coerce_bool(self, raw: Any, default: bool = False) -> bool:
        if isinstance(raw, bool):
            return bool(raw)
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            v = raw.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                return True
            if v in {"false", "0", "no", "n"}:
                return False
        return bool(default)

    def _coerce_float(self, raw: Any, default: float = 0.0) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _ensure_state_arrays(self) -> Dict[str, np.ndarray]:
        if self._state_arrays is not None:
            return self._state_arrays
        with np.load(self.states_path, allow_pickle=False) as payload:
            grid_sizes = np.asarray(payload["grid_sizes"])
            object_offsets = np.asarray(payload["object_offsets"])
            object_types = np.asarray(payload["object_types"])
            object_words = np.asarray(payload["object_words"])
            object_x = np.asarray(payload["object_x"])
            object_y = np.asarray(payload["object_y"])
            object_has_direction = np.asarray(payload["object_has_direction"])
            object_directions = np.asarray(payload["object_directions"])
            raw_state_count = payload.get("state_count")

        if raw_state_count is not None:
            state_count = int(np.asarray(raw_state_count).reshape(-1)[0])
        else:
            state_count = int(len(grid_sizes))

        self._state_arrays = {
            "grid_sizes": grid_sizes,
            "object_offsets": object_offsets,
            "object_types": object_types,
            "object_words": object_words,
            "object_x": object_x,
            "object_y": object_y,
            "object_has_direction": object_has_direction,
            "object_directions": object_directions,
        }
        self._state_count = int(max(0, state_count))
        return self._state_arrays

    def _cache_state_payload(self, state_index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.state_cache[state_index] = payload
        self.state_cache.move_to_end(state_index)
        if len(self.state_cache) > max(1, int(self.state_cache_size)):
            self.state_cache.popitem(last=False)
        return payload

    def get_state_payload(self, state_index: int) -> Dict[str, Any]:
        resolved_state_index = self._normalize_int(state_index)
        if resolved_state_index is None or resolved_state_index < 0:
            raise IndexError(f"Invalid state index: {state_index}")

        cached = self.state_cache.get(resolved_state_index)
        if cached is not None:
            self.state_cache.move_to_end(resolved_state_index)
            return cached

        arrays = self._ensure_state_arrays()
        state_count = self._state_count or 0
        if resolved_state_index >= state_count:
            raise IndexError(
                f"State index {resolved_state_index} out of bounds for {self.states_path}"
            )

        grid_sizes = arrays["grid_sizes"]
        object_offsets = arrays["object_offsets"]
        object_types = arrays["object_types"]
        object_words = arrays["object_words"]
        object_x = arrays["object_x"]
        object_y = arrays["object_y"]
        object_has_direction = arrays["object_has_direction"]
        object_directions = arrays["object_directions"]

        raw_grid_size = grid_sizes[resolved_state_index]
        if (
            isinstance(raw_grid_size, (list, tuple, np.ndarray))
            and len(raw_grid_size) >= 2
        ):
            grid_size = [int(raw_grid_size[0]), int(raw_grid_size[1])]
        else:
            grid_size = [0, 0]

        start = int(object_offsets[resolved_state_index]) if resolved_state_index < len(object_offsets) else 0
        end = (
            int(object_offsets[resolved_state_index + 1])
            if resolved_state_index + 1 < len(object_offsets)
            else int(len(object_types))
        )
        if end < start:
            end = start

        objects: List[Dict[str, Any]] = []
        for object_index in range(max(0, start), max(0, end)):
            position = [int(object_x[object_index]), int(object_y[object_index])]
            raw_word = object_words[object_index]
            raw_type = object_types[object_index]
            row = {
                "type": str(raw_type),
                "word": str(raw_word),
                "position": position,
            }
            if bool(object_has_direction[object_index]):
                row["direction"] = str(object_directions[object_index])
            objects.append(row)

        payload: Dict[str, Any] = {
            "grid_size": grid_size,
            "objects": objects,
        }
        return self._cache_state_payload(resolved_state_index, payload)

    def _normalize_transition_payload(
        self,
        payload: Mapping[str, Any],
        fallback_transition_index: int,
    ) -> Optional[TransitionPayload]:
        transition_index = self._normalize_int(payload.get("transition_index"), fallback=fallback_transition_index)
        if transition_index is None:
            return None
        try:
            state_index = int(payload["state_index"])
            next_state_index = int(payload["next_state_index"])
        except (TypeError, ValueError, KeyError):
            return None

        action = str(payload.get("action", "")).strip()
        reward = self._coerce_float(payload.get("reward", 0.0))
        done = self._coerce_bool(payload.get("done", False))
        terminated = self._coerce_bool(payload.get("terminated", False))
        truncated = self._coerce_bool(payload.get("truncated", False))
        return TransitionPayload(
            transition_index=transition_index,
            state_index=state_index,
            next_state_index=next_state_index,
            action=action,
            reward=reward,
            done=done or terminated or truncated,
            terminated=terminated,
            truncated=truncated,
        )

    def _load_transition_row(self, transition_index: int) -> Optional[TransitionPayload]:
        payload = self.transition_row_cache.get(transition_index)
        if payload is not None:
            return payload

        target = int(transition_index)
        with self.transitions_path.open("r", encoding="utf-8") as handle:
            line_index = 0
            for raw_line in handle:
                line_text = raw_line.strip()
                if not line_text:
                    continue
                try:
                    row = json.loads(line_text)
                except Exception:
                    line_index += 1
                    continue
                if not isinstance(row, Mapping):
                    line_index += 1
                    continue

                parsed_transition = self._normalize_transition_payload(row, fallback_transition_index=line_index)
                if parsed_transition is None:
                    line_index += 1
                    continue

                # Support transition_index fields and fallback to line-based indices.
                row_transition_index = parsed_transition.transition_index
                if row_transition_index == target or line_index == target:
                    self.transition_row_cache[target] = parsed_transition
                    if row_transition_index != target and row_transition_index not in self.transition_row_cache:
                        self.transition_row_cache[row_transition_index] = parsed_transition
                    return parsed_transition

                line_index += 1
        return None

    def get_transition_payload(self, transition_index: int) -> TransitionPayload:
        resolved = self._normalize_int(transition_index)
        if resolved is None:
            raise KeyError(f"Invalid transition index: {transition_index}")

        cached = self.transition_row_cache.get(resolved)
        if cached is not None:
            return cached

        parsed = self._load_transition_row(resolved)
        if parsed is None:
            raise KeyError(f"Transition row {resolved} not found in {self.transitions_path}")
        return parsed

    def to_bundle_info_payload(self) -> Dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "dataset_label": self.dataset_label,
            "artifact_stem": self.artifact_stem,
            "scenario_type": self.scenario_type,
            "transitions_path": str(self.transitions_path),
            "states_path": str(self.states_path),
        }


@dataclass
class DynamicsDiscoveryViewData:
    discovery_json: Path
    visual_config: Dict[str, Any] = field(default_factory=dict)
    state_cache_size: int = DEFAULT_STATE_CACHE_SIZE
    render_cache_size: int = DEFAULT_RENDER_CACHE_SIZE

    class_metadata_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    class_refs: Dict[int, List[TransitionRef]] = field(default_factory=lambda: defaultdict(list))
    bundles: List[BundleContext] = field(default_factory=list)
    rendered_png_cache: OrderedDict[tuple[int, int, str], bytes] = field(default_factory=OrderedDict)
    _artifact_renderer: Optional[TransitionArtifactRenderer] = field(default=None, init=False, repr=False)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping JSON at {path}")
        return payload

    def _normalize_str(self, value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    def _normalize_int(self, value: Any, fallback: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _resolve_bundle_path(self, value: Any) -> Optional[Path]:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        return _resolve_project_path(candidate)

    def load(self) -> None:
        payload = self._read_json(_resolve_project_path(self.discovery_json))
        transition_class_mapping = payload.get("transition_class_mapping")
        if not isinstance(transition_class_mapping, list):
            raise ValueError("discovery JSON does not contain `transition_class_mapping`.")

        class_metadata = payload.get("class_metadata")
        if isinstance(class_metadata, list):
            for row in class_metadata:
                if not isinstance(row, Mapping):
                    continue
                class_idx = self._normalize_int(row.get("class_idx"))
                if class_idx is None:
                    continue
                action = self._normalize_str(row.get("action"))
                self.class_metadata_by_id[class_idx] = {
                    "class_idx": int(class_idx),
                    "action": action,
                    "transition_count": int(self._normalize_int(row.get("transition_count"), 0) or 0),
                    "sample_transition": row.get("sample_transition"),
                    "state_diff": row.get("state_diff", {}),
                    "class_signature": row.get("class_signature")
                    or row.get("class_signature_notes")
                    or {
                        "action": action,
                        "state_diff": row.get("state_diff", {}),
                    },
                }

        for bundle_id, row in enumerate(transition_class_mapping):
            if not isinstance(row, Mapping):
                continue
            transitions_to_class = row.get("transition_to_class")
            if not isinstance(transitions_to_class, Mapping):
                continue

            transitions_path = self._resolve_bundle_path(row.get("transitions_path"))
            if transitions_path is None or not transitions_path.exists():
                continue
            states_path = self._resolve_bundle_path(row.get("states_path"))
            if states_path is None or not states_path.exists():
                continue

            dataset_root = self._normalize_str(row.get("dataset_root"), default="")
            dataset_label = self._normalize_str(row.get("dataset_label"), default="dataset")
            artifact_stem = self._normalize_str(row.get("artifact_stem"), default=transitions_path.stem.removesuffix("_transitions"))
            scenario_type = self._normalize_str(row.get("scenario_type"), default=artifact_stem)

            transition_to_class = {}
            for raw_transition_index, raw_class_idx in transitions_to_class.items():
                transition_index = self._normalize_int(raw_transition_index)
                class_idx = self._normalize_int(raw_class_idx)
                if transition_index is None or class_idx is None:
                    continue
                transition_to_class[int(transition_index)] = int(class_idx)
                self.class_refs[int(class_idx)].append(
                    TransitionRef(bundle_id=int(bundle_id), transition_index=int(transition_index))
                )

            self.bundles.append(
                BundleContext(
                    bundle_id=int(bundle_id),
                    dataset_label=dataset_label,
                    dataset_root=dataset_root,
                    artifact_stem=artifact_stem,
                    scenario_type=scenario_type,
                    transitions_path=transitions_path,
                    states_path=states_path,
                    transition_to_class=transition_to_class,
                    state_cache_size=self.state_cache_size,
                )
            )

        if not self.class_refs:
            raise ValueError("No class mappings were found in the discovery JSON.")

        all_class_indices = sorted(set(self.class_refs.keys()) | set(self.class_metadata_by_id.keys()))
        for class_idx in all_class_indices:
            self.class_refs.setdefault(class_idx, [])
            metadata = self.class_metadata_by_id.get(class_idx)
            if metadata is None:
                self.class_metadata_by_id[class_idx] = {
                    "class_idx": int(class_idx),
                    "action": "",
                    "transition_count": int(len(self.class_refs[class_idx])),
                    "sample_transition": None,
                    "state_diff": {},
                    "class_signature": {
                        "action": "",
                        "state_diff": {},
                    },
                }
            else:
                metadata["transition_count"] = int(len(self.class_refs[class_idx]))

    def get_class_list(self) -> List[Dict[str, Any]]:
        class_indices = sorted(self.class_refs.keys())
        classes = []
        for class_idx in class_indices:
            metadata = self.class_metadata_by_id.get(class_idx, {})
            classes.append(
                {
                    "classIdx": int(class_idx),
                    "action": self._normalize_str(metadata.get("action"), default=""),
                    "transitionCount": int(len(self.class_refs[class_idx])),
                    "sampleTransition": metadata.get("sample_transition"),
                }
            )
        return classes

    def get_class_metadata(self, class_idx: int) -> Dict[str, Any]:
        metadata = self.class_metadata_by_id.get(int(class_idx))
        if metadata is None:
            raise KeyError(f"Unknown class index: {class_idx}")
        return {
            "classIdx": int(class_idx),
            "action": self._normalize_str(metadata.get("action"), default=""),
            "transitionCount": int(len(self.class_refs[int(class_idx)])),
            "stateDiff": metadata.get("state_diff", {}),
            "sampleTransition": metadata.get("sample_transition"),
            "classSignature": metadata.get("class_signature"),
        }

    def _resolve_transition_selection(
        self,
        class_idx: int,
        position: int,
    ) -> tuple[List[TransitionRef], TransitionRef, BundleContext, TransitionPayload]:
        if class_idx not in self.class_refs:
            raise KeyError(f"Unknown class index: {class_idx}")
        refs = self.class_refs[class_idx]
        if position < 0 or position >= len(refs):
            raise IndexError(f"Transition position out of range for class {class_idx}: {position}")

        selected_ref = refs[position]
        if selected_ref.bundle_id < 0 or selected_ref.bundle_id >= len(self.bundles):
            raise KeyError(f"Invalid bundle id {selected_ref.bundle_id} for class {class_idx}")

        bundle = self.bundles[selected_ref.bundle_id]
        transition = bundle.get_transition_payload(selected_ref.transition_index)
        return refs, selected_ref, bundle, transition

    def _get_artifact_renderer(self) -> TransitionArtifactRenderer:
        renderer = self._artifact_renderer
        if renderer is None:
            renderer = TransitionArtifactRenderer(visual_config=self.visual_config)
            self._artifact_renderer = renderer
        return renderer

    def _cache_rendered_png(
        self,
        key: tuple[int, int, str],
        png_bytes: bytes,
    ) -> bytes:
        self.rendered_png_cache[key] = png_bytes
        self.rendered_png_cache.move_to_end(key)
        if len(self.rendered_png_cache) > max(1, int(self.render_cache_size)):
            self.rendered_png_cache.popitem(last=False)
        return png_bytes

    def render_state_png(
        self,
        *,
        bundle_id: int,
        state_index: int,
        title: str,
    ) -> bytes:
        cache_key = (int(bundle_id), int(state_index), str(title))
        cached = self.rendered_png_cache.get(cache_key)
        if cached is not None:
            self.rendered_png_cache.move_to_end(cache_key)
            return cached

        if bundle_id < 0 or bundle_id >= len(self.bundles):
            raise KeyError(f"Invalid bundle id: {bundle_id}")
        bundle = self.bundles[bundle_id]
        state_payload = bundle.get_state_payload(state_index)
        image = self._get_artifact_renderer().render_state_snapshot_image(
            state=state_payload,
            title=str(title),
            visual_config=self.visual_config,
        )
        if image is None:
            raise RuntimeError("Failed to render transition state image. Pillow may be unavailable.")
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
        finally:
            image.close()
        return self._cache_rendered_png(cache_key, png_bytes)

    def get_class_transition(self, class_idx: int, position: int) -> Dict[str, Any]:
        refs, selected_ref, bundle, transition = self._resolve_transition_selection(class_idx, position)

        previous_state = bundle.get_state_payload(transition.state_index)
        next_state = bundle.get_state_payload(transition.next_state_index)

        metadata = self.get_class_metadata(class_idx)
        previous_state_path = f"{bundle.to_bundle_info_payload()['transitions_path']} [index {transition.state_index}]"
        next_state_path = f"{bundle.to_bundle_info_payload()['transitions_path']} [index {transition.next_state_index}]"

        return {
            "class": metadata,
            "position": int(position),
            "total": int(len(refs)),
            "bundle": {
                **bundle.to_bundle_info_payload(),
                "index": int(selected_ref.bundle_id),
            },
            "transition": {
                "transitionIndex": int(transition.transition_index),
                "stateIndex": int(transition.state_index),
                "nextStateIndex": int(transition.next_state_index),
                "action": transition.action,
                "reward": float(transition.reward),
                "done": bool(transition.done),
                "terminated": bool(transition.terminated),
                "truncated": bool(transition.truncated),
            },
            "previousState": previous_state,
            "nextState": next_state,
            "previousImage": {
                "url": f"/api/classes/{int(class_idx)}/transition/{int(position)}/render/previous",
                "path": previous_state_path,
            },
            "nextImage": {
                "url": f"/api/classes/{int(class_idx)}/transition/{int(position)}/render/next",
                "path": next_state_path,
            },
            "debug": {
                "previousStateLookup": previous_state_path,
                "nextStateLookup": next_state_path,
            },
            "navigation": {
                "previous": int(position - 1) if position > 0 else None,
                "next": int(position + 1) if position + 1 < len(refs) else None,
            },
        }


def create_heuristic_dynamics_discovery_view_app(
    *,
    discovery_json: Path,
    visual_config: Mapping[str, Any],
    state_cache_size: int,
    render_cache_size: int,
    host: str,
    port: int,
) -> FastAPI:
    dataset = DynamicsDiscoveryViewData(
        discovery_json=discovery_json,
        visual_config=dict(visual_config or {}),
        state_cache_size=max(1, int(state_cache_size)),
        render_cache_size=max(1, int(render_cache_size)),
    )
    dataset.load()

    app = FastAPI(
        title="BABA Heuristic Dynamics Discovery Viewer",
    )
    app.state.dataset = dataset
    app.state.discovery_json = str(_resolve_project_path(discovery_json))
    app.state.visual_config = dict(visual_config or {})
    app.state.host = str(host)
    app.state.port = int(port)
    mount_shared_static(app)

    @app.get("/")
    def index():
        return html_file_response("heuristic_dynamics_discovery.html")

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "discoveryJson": app.state.discovery_json,
            "classes": len(app.state.dataset.class_refs),
            "visualConfigLoaded": bool(app.state.visual_config),
            "host": app.state.host,
            "port": app.state.port,
        }

    @app.get("/api/classes")
    def list_classes():
        return {
            "classes": app.state.dataset.get_class_list(),
        }

    @app.get("/api/classes/{class_idx}")
    def get_class_info(class_idx: int):
        try:
            metadata = app.state.dataset.get_class_metadata(int(class_idx))
            return {
                "class": metadata,
            }
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/classes/{class_idx}/transition/{position}")
    def get_transition(class_idx: int, position: int):
        try:
            return app.state.dataset.get_class_transition(int(class_idx), int(position))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except IndexError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/classes/{class_idx}/transition/{position}/render/{state_kind}")
    def render_transition_state(class_idx: int, position: int, state_kind: str):
        normalized_kind = str(state_kind).strip().lower()
        if normalized_kind not in {"previous", "next"}:
            raise HTTPException(status_code=400, detail=f"Unknown state render kind: {state_kind}")
        try:
            _refs, selected_ref, _bundle, transition = app.state.dataset._resolve_transition_selection(
                int(class_idx),
                int(position),
            )
            state_index = transition.state_index if normalized_kind == "previous" else transition.next_state_index
            title = "Previous State" if normalized_kind == "previous" else "Next State"
            png_bytes = app.state.dataset.render_state_png(
                bundle_id=int(selected_ref.bundle_id),
                state_index=int(state_index),
                title=title,
            )
            return Response(content=png_bytes, media_type="image/png")
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except IndexError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app


def _load_visual_config(
    *,
    env_config_path: Path,
    experiment_config_path: Path,
) -> Dict[str, Any]:
    bundle = load_baba_config_bundle(
        env_config_path=_resolve_project_path(env_config_path),
        experiment_config_path=_resolve_project_path(experiment_config_path),
    )
    return build_visualization_config(bundle.get("word_aliases"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a small web UI to inspect heuristic dynamics discovery classes and "
            "browse transitions per class."
        )
    )
    parser.add_argument(
        "--discovery-json",
        type=Path,
        default=DEFAULT_DISCOVERY_JSON,
        help="Input discovery JSON (from heuristic_dynamics_discovery.py).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Bind host for the web viewer.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Bind port for the web viewer.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=DEFAULT_ENV_CONFIG,
        help="Environment config used to match the project's standard visualization setup.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Experiment config used to load serialization word aliases for visualization.",
    )
    parser.add_argument(
        "--state-cache-size",
        type=int,
        default=DEFAULT_STATE_CACHE_SIZE,
        help="Number of decoded states to cache per dataset bundle.",
    )
    parser.add_argument(
        "--render-cache-size",
        type=int,
        default=DEFAULT_RENDER_CACHE_SIZE,
        help="Number of rendered PNG state snapshots to keep in memory.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visual_config = _load_visual_config(
        env_config_path=args.env_config,
        experiment_config_path=args.experiment_config,
    )
    run_app_server(
        app=create_heuristic_dynamics_discovery_view_app(
            discovery_json=_resolve_project_path(args.discovery_json),
            visual_config=visual_config,
            state_cache_size=max(1, int(args.state_cache_size)),
            render_cache_size=max(1, int(args.render_cache_size)),
            host=args.host,
            port=int(args.port),
        ),
        label="BABA Heuristic Dynamics Discovery Viewer",
        host=args.host,
        port=int(args.port),
        reload=bool(args.reload),
        open_browser=not bool(args.no_browser),
    )


if __name__ == "__main__":
    main()
