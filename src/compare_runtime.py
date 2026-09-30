from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from src.discovery.artifact_renderer import TransitionArtifactRenderer
from src.program_model import SandboxConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ACTION_TO_ID: Dict[str, int] = {
    "idle": 0,
    "up": 1,
    "right": 2,
    "down": 3,
    "left": 4,
}
_ARTIFACT_RENDERER = TransitionArtifactRenderer()


def _normalize_diff_object_label(obj: Mapping[str, Any]) -> str:
    word = str(obj.get("word", "") or "").strip()
    if word:
        return word.upper()
    obj_type = str(obj.get("type", "") or "").strip()
    return obj_type.upper() or "OBJECT"


def _normalize_diff_object_position(obj: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    position = obj.get("position")
    if (
        isinstance(position, (list, tuple))
        and len(position) == 2
        and isinstance(position[0], int)
        and isinstance(position[1], int)
    ):
        return int(position[0]), int(position[1])
    return None


def _normalize_diff_direction(direction: Any) -> str:
    if not isinstance(direction, str):
        return ""
    normalized = direction.strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("facing "):
        normalized = normalized[len("facing "):].strip()
    return normalized


def _format_diff_object_signature(obj: Mapping[str, Any]) -> str:
    label = _normalize_diff_object_label(obj)
    position = _normalize_diff_object_position(obj)
    parts = [label]
    if position is not None:
        parts.append(f"@({position[0]},{position[1]})")
    return "".join(parts)


def _build_state_object_counter(state: Mapping[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    raw_objects = state.get("objects")
    if not isinstance(raw_objects, list):
        return counter
    for raw_obj in raw_objects:
        if not isinstance(raw_obj, Mapping):
            continue
        counter[_format_diff_object_signature(raw_obj)] += 1
    return counter


def _build_state_object_direction_counter(state: Mapping[str, Any]) -> Dict[str, Counter[str]]:
    grouped: Dict[str, Counter[str]] = {}
    raw_objects = state.get("objects")
    if not isinstance(raw_objects, list):
        return grouped
    for raw_obj in raw_objects:
        if not isinstance(raw_obj, Mapping):
            continue
        base_signature = _format_diff_object_signature(raw_obj)
        direction = _normalize_diff_direction(raw_obj.get("direction"))
        if not direction:
            continue
        bucket = grouped.setdefault(base_signature, Counter())
        bucket[direction] += 1
    return grouped


def _summarize_counter_delta(prefix: str, counter: Counter[str], *, limit: int = 2) -> Optional[str]:
    entries: List[str] = []
    remaining = 0
    for label in sorted(counter.keys()):
        count = int(counter[label])
        if count <= 0:
            continue
        rendered = label if count == 1 else f"{label} x{count}"
        if len(entries) < limit:
            entries.append(rendered)
        else:
            remaining += count
    if not entries:
        return None
    summary = f"{prefix} " + ", ".join(entries)
    if remaining > 0:
        summary += f" (+{remaining} more)"
    return summary


def _summarize_direction_differences(
    *,
    expected_state: Mapping[str, Any],
    predicted_state: Mapping[str, Any],
    limit: int = 2,
) -> Optional[str]:
    expected_grouped = _build_state_object_direction_counter(expected_state)
    predicted_grouped = _build_state_object_direction_counter(predicted_state)
    entries: List[str] = []
    remaining = 0
    for base_signature in sorted(set(expected_grouped.keys()) | set(predicted_grouped.keys())):
        expected_counter = expected_grouped.get(base_signature, Counter())
        predicted_counter = predicted_grouped.get(base_signature, Counter())
        if expected_counter == predicted_counter:
            continue
        expected_parts = [
            direction if count == 1 else f"{direction} x{count}"
            for direction, count in sorted(expected_counter.items())
            if int(count) > 0
        ]
        predicted_parts = [
            direction if count == 1 else f"{direction} x{count}"
            for direction, count in sorted(predicted_counter.items())
            if int(count) > 0
        ]
        if not expected_parts or not predicted_parts:
            continue
        rendered = f"{base_signature} {'/'.join(expected_parts)}->{'/'.join(predicted_parts)}"
        if len(entries) < limit:
            entries.append(rendered)
        else:
            remaining += 1
    if not entries:
        return None
    summary = "dir " + ", ".join(entries)
    if remaining > 0:
        summary += f" (+{remaining} more)"
    return summary


def build_comparison_difference_summary(
    *,
    expected_state: Mapping[str, Any],
    predicted_state: Optional[Mapping[str, Any]],
    prediction_error: Optional[str],
) -> Optional[str]:
    if not isinstance(predicted_state, Mapping):
        if prediction_error:
            return f"Diff: prediction unavailable ({prediction_error})"
        return "Diff: prediction unavailable"

    segments: List[str] = []
    expected_done = bool(expected_state.get("step", {}).get("terminated")) if isinstance(expected_state.get("step"), Mapping) else None
    predicted_done = bool(predicted_state.get("step", {}).get("terminated")) if isinstance(predicted_state.get("step"), Mapping) else None
    if expected_done is not None and predicted_done is not None and expected_done != predicted_done:
        segments.append(f"done GT={expected_done}, Pred={predicted_done}")

    expected_grid = expected_state.get("grid_size")
    predicted_grid = predicted_state.get("grid_size")
    if (
        isinstance(expected_grid, list)
        and isinstance(predicted_grid, list)
        and len(expected_grid) == 2
        and len(predicted_grid) == 2
        and expected_grid != predicted_grid
    ):
        segments.append(
            f"grid GT={int(expected_grid[0])}x{int(expected_grid[1])}, Pred={int(predicted_grid[0])}x{int(predicted_grid[1])}"
        )

    expected_counter = _build_state_object_counter(expected_state)
    predicted_counter = _build_state_object_counter(predicted_state)
    missing_summary = _summarize_counter_delta("missing", expected_counter - predicted_counter)
    extra_summary = _summarize_counter_delta("unexpected", predicted_counter - expected_counter)
    direction_summary = _summarize_direction_differences(
        expected_state=expected_state,
        predicted_state=predicted_state,
    )
    if direction_summary:
        segments.append(direction_summary)
    if missing_summary:
        segments.append(missing_summary)
    if extra_summary:
        segments.append(extra_summary)

    if not segments:
        return "Diff: no visible state difference"
    return "Diff: " + "; ".join(segments[:3])


def resolve_path(path_or_str: str | Path) -> Path:
    path = Path(path_or_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def build_sandbox_config(exp_cfg: Dict[str, Any]) -> SandboxConfig:
    defaults = SandboxConfig()
    program_cfg = exp_cfg.get("python_program", {})
    if not isinstance(program_cfg, dict):
        program_cfg = {}
    sandbox_cfg = program_cfg.get("sandbox", {})
    if not isinstance(sandbox_cfg, dict):
        sandbox_cfg = {}

    def to_optional_int(value: Any, default_value: int) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default_value)

    return SandboxConfig(
        max_code_chars=to_optional_int(
            sandbox_cfg.get("max_code_chars"), defaults.max_code_chars
        ),
        max_ast_nodes=to_optional_int(
            sandbox_cfg.get("max_ast_nodes"), defaults.max_ast_nodes
        ),
        line_budget=to_optional_int(
            sandbox_cfg.get("line_budget"), defaults.line_budget
        ),
        timeout_ms=to_optional_int(
            sandbox_cfg.get("timeout_ms"), defaults.timeout_ms
        ),
        static_validation_enabled=False,
    )


def normalize_version_tag(version: str) -> str:
    raw = str(version).strip().lower()
    if not raw:
        raise ValueError("version is empty")
    if raw == "final":
        return "final"
    if raw.startswith("v") and raw[1:].isdigit():
        return f"v{int(raw[1:]):03d}"
    if raw.isdigit():
        return f"v{int(raw):03d}"
    raise ValueError(f"Unsupported version format: {version}")


def resolve_experiment_dir(experiment: str) -> Path:
    candidate = Path(experiment)
    if candidate.is_absolute():
        exp_dir = candidate
    else:
        exp_dir = PROJECT_ROOT / "experiments" / experiment
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    return exp_dir


def resolve_config_from_experiment_snapshot(
    *,
    exp_dir: Path,
    cli_value: str,
    snapshot_filename: str,
    default_relative_path: str,
) -> Path:
    snapshot_dir = exp_dir / "config_snapshot"
    snapshot_path = snapshot_dir / snapshot_filename
    if snapshot_path.exists() and cli_value == default_relative_path:
        snapshot_name = Path(snapshot_filename)
        resolved_snapshot_path = (
            snapshot_dir / f"{snapshot_name.stem}.resolved{snapshot_name.suffix}"
        )
        if resolved_snapshot_path.exists():
            return resolved_snapshot_path
        return snapshot_path
    return resolve_path(cli_value)


def resolve_program_path(exp_dir: Path, version_tag: str) -> Path:
    if version_tag == "final":
        versions_dir = exp_dir / "program_versions"
        candidates = sorted(versions_dir.glob("v*.py"))
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(f"final program not found under {versions_dir}")

    path = exp_dir / "program_versions" / f"{version_tag}.py"
    if path.exists():
        return path
    raise FileNotFoundError(
        "Program file not found for version "
        f"{version_tag}. Tried: {path}"
    )


def extract_grid_size(state: Dict[str, Any]) -> Tuple[int, int]:
    return _ARTIFACT_RENDERER.extract_grid_size(state)


def resolve_visual_object_type_key(obj: Dict[str, Any], obj_type: str) -> str:
    raw_obj_type = str(obj_type or obj.get("type") or "").strip().lower()
    if raw_obj_type in {"rule_property", "property"}:
        return "rule_property"
    if raw_obj_type in {"rule_noun", "noun"}:
        return "rule_noun"
    if raw_obj_type in {"rule_operator", "operator"}:
        return "rule_operator"
    if raw_obj_type:
        return raw_obj_type
    return "object"


def resolve_visual_object_label(
    obj: Dict[str, Any],
    obj_type: str,
    type_symbols: Dict[str, str],
) -> str:
    type_key = resolve_visual_object_type_key(obj, obj_type)
    raw_word = str(obj.get("word") or obj.get("text") or "").strip().lower()
    if type_key in {"rule_property", "rule_noun", "rule_operator"}:
        if raw_word:
            return raw_word.upper()
        symbol = str(type_symbols.get(type_key, type_key)).strip()
        return symbol.upper()
    if raw_word:
        symbol = str(type_symbols.get(raw_word, raw_word)).strip()
        return symbol.upper()
    symbol = str(type_symbols.get(type_key, type_key)).strip()
    return symbol.upper()


def extract_object_direction_name(obj: Dict[str, Any]) -> Optional[str]:
    direction = obj.get("direction")
    if isinstance(direction, str) and direction.strip():
        return direction.strip()
    if isinstance(direction, int):
        mapping = {
            0: "facing right",
            1: "facing up",
            2: "facing left",
            3: "facing down",
        }
        return mapping.get(int(direction))
    return None


def render_state_snapshot_image(
    state: Dict[str, Any],
    title: str,
    *,
    action_name: Optional[str] = None,
    show_context_text: bool = True,
    include_world_in_context: bool = True,
    visual_config: Optional[Dict[str, Any]] = None,
):
    image = _ARTIFACT_RENDERER.render_state_snapshot_image(
        state=state,
        title=title,
        action_name=action_name,
        show_context_text=show_context_text,
        include_world_in_context=include_world_in_context,
        visual_config=visual_config,
    )
    if image is None:
        raise RuntimeError("Unable to render state snapshot image.")
    return image


def compose_comparison_image(
    *,
    previous_state: Dict[str, Any],
    actual_next_state: Dict[str, Any],
    predicted_next_state: Optional[Dict[str, Any]],
    action_name: str,
    experiment_name: str,
    version_tag: str,
    prediction_error: Optional[str],
    step_index: int,
    difference_summary: Optional[str] = None,
    status_lines: Optional[List[str]] = None,
    include_world_in_context: bool = True,
    visual_config: Optional[Dict[str, Any]] = None,
):
    return _ARTIFACT_RENDERER.compose_comparison_image(
        previous_state=previous_state,
        actual_next_state=actual_next_state,
        predicted_next_state=predicted_next_state,
        action_name=action_name,
        experiment_name=experiment_name,
        version_tag=version_tag,
        prediction_error=prediction_error,
        step_index=step_index,
        difference_summary=difference_summary,
        status_lines=status_lines,
        include_world_in_context=include_world_in_context,
        visual_config=visual_config,
    )


def build_status_lines(
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    prediction_error: Optional[str],
    episode_done: bool,
    extra_lines: Optional[List[str]] = None,
) -> List[str]:
    status = [
        f"reward={float(reward):.3f}",
        f"terminated={bool(terminated)}",
        f"truncated={bool(truncated)}",
        f"episode_done={bool(episode_done)}",
    ]
    if isinstance(prediction_error, str) and prediction_error.strip():
        status.append(f"prediction_error={prediction_error.strip()}")
    if isinstance(extra_lines, list):
        status.extend(str(line) for line in extra_lines if str(line).strip())
    return status


def save_transition_artifacts(
    *,
    output_dir: Path,
    step_index: int,
    action_name: str,
    image,
    payload: Dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    stem = f"step_{int(step_index):04d}_{action_name}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    image_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    image.save(image_path, format="PNG")
    enriched_payload = dict(payload)
    enriched_payload["timestamp"] = timestamp.isoformat()
    enriched_payload["image_file"] = image_path.name
    json_path.write_text(
        json.dumps(enriched_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return image_path, json_path
