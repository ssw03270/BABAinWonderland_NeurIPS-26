from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
import re
import sys
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data.transition_buffer import Transition
from .environments.state_serializer import (
    StateSerializer,
    normalize_raw_baba_object_schema,
    normalize_state_direction,
)
from .environments.custom_map_spec import (
    ASCII_OBJECTS,
    ASCII_RULE_OBJECTS,
    ASCII_RULE_OPERATORS,
    ASCII_RULE_PROPERTIES,
    load_custom_map_catalog,
)
from .program_model import ProgramEvaluator
from .program_model.state_codec import parse_state_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWED_CUSTOM_MAP_DIFFICULTY = "original"
DEFAULT_ALLOWED_CUSTOM_MAP_SPLITS: Tuple[str, ...] = ("all_train", "all_test")
DEFAULT_TEST_PATHS: Tuple[Path, ...] = ()


@dataclass(frozen=True)
class MinedTestCase:
    source_path: Path
    test_method_name: str
    env_class_name: str
    action_sequence: Tuple[str, ...]


@dataclass(frozen=True)
class ProgramSource:
    version_id: str
    path: Optional[Path]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "path": str(self.path) if isinstance(self.path, Path) else None,
        }


@dataclass(frozen=True)
class DynamicsExample:
    example_id: str
    concept_name: str
    category: str
    description: str
    source_kind: str
    source_ref: str
    env_class_name: Optional[str]
    setup_actions: Tuple[str, ...]
    focal_action: str
    transition: Transition
    expected_reward: float
    expected_terminated: bool
    expected_truncated: bool
    grid_size: Tuple[int, int]
    object_count: int
    world_object_count: int
    rule_block_count: int
    tags: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "concept_name": self.concept_name,
            "category": self.category,
            "description": self.description,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "env_class_name": self.env_class_name,
            "setup_actions": list(self.setup_actions),
            "focal_action": self.focal_action,
            "transition": self.transition.to_dict(),
            "expected_reward": float(self.expected_reward),
            "expected_terminated": bool(self.expected_terminated),
            "expected_truncated": bool(self.expected_truncated),
            "grid_size": [int(self.grid_size[0]), int(self.grid_size[1])],
            "object_count": int(self.object_count),
            "world_object_count": int(self.world_object_count),
            "rule_block_count": int(self.rule_block_count),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ExampleEvaluationResult:
    example_id: str
    is_correct: bool
    predicted_next_state_json: Optional[str]
    error_phase: Optional[str]
    error_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "is_correct": bool(self.is_correct),
            "predicted_next_state_json": self.predicted_next_state_json,
            "error_phase": self.error_phase,
            "error_message": self.error_message,
        }


def ensure_baba_importable(project_root: Path = PROJECT_ROOT) -> None:
    path_text = str(project_root)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    import baba_in_wonderland  # type: ignore  # noqa: F401


class _TestMethodMiner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.env_class_name: Optional[str] = None
        self.action_sequence: List[str] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if (
            self.env_class_name is None
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "env"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            self.env_class_name = node.value.func.id
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        action_name = _extract_step_action_name(node)
        if action_name is not None:
            self.action_sequence.append(action_name)
        self.generic_visit(node)


def _extract_step_action_name(node: ast.Call) -> Optional[str]:
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "step":
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "env":
        return None
    if len(node.args) != 1:
        return None

    raw_action = node.args[0]
    if not isinstance(raw_action, ast.Attribute):
        return None
    if not isinstance(raw_action.value, ast.Attribute):
        return None
    if raw_action.value.attr != "actions":
        return None
    if not isinstance(raw_action.value.value, ast.Name) or raw_action.value.value.id != "env":
        return None
    return raw_action.attr


@lru_cache(maxsize=None)
def mine_test_cases(test_path_text: str) -> Tuple[MinedTestCase, ...]:
    test_path = Path(test_path_text)
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))

    cases: List[MinedTestCase] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if not item.name.startswith("test_"):
                continue
            miner = _TestMethodMiner()
            miner.visit(item)
            if miner.env_class_name is None or len(miner.action_sequence) == 0:
                continue
            cases.append(
                MinedTestCase(
                    source_path=test_path,
                    test_method_name=item.name,
                    env_class_name=miner.env_class_name,
                    action_sequence=tuple(miner.action_sequence),
                )
            )
    return tuple(cases)


def discover_test_cases(
    test_paths: Sequence[Path] = DEFAULT_TEST_PATHS,
) -> List[MinedTestCase]:
    cases: List[MinedTestCase] = []
    for test_path in test_paths:
        if not Path(test_path).exists():
            continue
        cases.extend(mine_test_cases(str(Path(test_path).resolve())))
    return cases


@lru_cache(maxsize=None)
def _load_module_from_path(path_text: str) -> ModuleType:
    module_path = Path(path_text)
    ensure_baba_importable(module_path.parents[1])

    module_name = (
        f"_dynamics_catalog_{module_path.stem}_"
        f"{abs(hash(str(module_path.resolve())))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _extract_state_objects(env: Any) -> List[Dict[str, Any]]:
    grid = getattr(env, "grid", None)
    width = int(getattr(env, "width", 0) or 0)
    height = int(getattr(env, "height", 0) or 0)
    if grid is None or width <= 0 or height <= 0:
        return []

    objects: List[Dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            stack = grid.get(x, y, "all")
            if not isinstance(stack, list):
                stack = [stack]
            for obj in stack:
                if obj is None:
                    continue
                obj_type, word = normalize_raw_baba_object_schema(
                    obj=obj,
                    raw_type=getattr(obj, "type", "unknown"),
                )
                row = {
                    "type": obj_type,
                    "word": word,
                    "position": [x, y],
                }
                color = getattr(obj, "color", None)
                if isinstance(color, str) and color.strip():
                    row["color"] = color.strip().lower()
                direction = normalize_state_direction(getattr(obj, "dir", None))
                if direction is not None:
                    row["direction"] = direction
                objects.append(row)

    objects.sort(
        key=lambda row: (
            int(row["position"][1]),
            int(row["position"][0]),
            str(row["type"]),
            "none" if row.get("color") is None else str(row.get("color")),
            "none" if row.get("direction") is None else str(row.get("direction")),
            "none" if row.get("word") is None else str(row.get("word")),
        )
    )
    return objects


def serialize_raw_baba_state(
    env: Any,
    *,
    terminated: bool,
    truncated: bool = False,
    serializer: Optional[StateSerializer] = None,
) -> str:
    active_serializer = serializer or StateSerializer(format_type="json")
    width = int(getattr(env, "width", 0) or 0)
    height = int(getattr(env, "height", 0) or 0)
    objects = _extract_state_objects(env)
    return active_serializer.serialize(
        grid_size=(width, height),
        objects=objects,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def _step_env(
    env: Any,
    action_name: str,
    *,
    serializer: Optional[StateSerializer] = None,
) -> Tuple[str, float, bool, bool]:
    action_value = getattr(env.actions, action_name)
    result = env.step(action_value)
    if not isinstance(result, tuple):
        raise RuntimeError("Unexpected step return type from raw BABA env.")

    if len(result) == 5:
        _, reward, terminated, truncated, _info = result
        done = bool(terminated or truncated)
    elif len(result) == 4:
        _, reward, done, _info = result
        is_win = bool(getattr(env, "is_win", False))
        reached_limit = bool(
            done
            and not is_win
            and isinstance(getattr(env, "step_count", None), Integral)
            and isinstance(getattr(env, "max_steps", None), Integral)
            and int(env.step_count) >= int(env.max_steps)
        )
        terminated = bool(done and not reached_limit)
        truncated = bool(done and reached_limit)
    else:
        raise RuntimeError(f"Unexpected step return length from raw BABA env: {len(result)}")

    next_state_json = serialize_raw_baba_state(
        env,
        terminated=bool(terminated),
        truncated=bool(truncated),
        serializer=serializer,
    )
    return next_state_json, float(reward), bool(terminated), bool(truncated)


def _derive_category(tokens: Sequence[str]) -> str:
    token_set = set(tokens)
    if "truncates" in token_set or "truncation" in token_set or "limit" in token_set:
        return "turn_limit"
    if "color" in token_set or "prefix" in token_set:
        return "rule_color_parser"
    if "vertical" in token_set or "parses" in token_set or "parsing" in token_set:
        if (
            "replace" in token_set
            or "replacement" in token_set
            or "transform" in token_set
            or "transforms" in token_set
        ):
            return "replacement_transform"
        return "rule_parser"
    if "stack" in token_set or "buried" in token_set or "top" in token_set:
        return "stack_order"
    if "rule" in token_set and ("block" in token_set or "blocks" in token_set):
        return "rule_block_mobility"
    if "pull" in token_set and ("overlap" in token_set or "entry" in token_set):
        return "pull_overlap"
    if "replacement" in token_set or "replace" in token_set or "transform" in token_set:
        return "replacement_transform"
    if "push" in token_set or "payload" in token_set:
        return "push"
    if "pull" in token_set:
        return "pull"
    if "create" in token_set or "breaking" in token_set or "break" in token_set:
        if "win" in token_set or "goal" in token_set:
            return "win_rule_timing"
        if "defeat" in token_set:
            return "defeat_rule_timing"
        if "hot" in token_set or "melt" in token_set:
            return "hot_melt_rule_timing"
        if "sink" in token_set:
            return "sink_rule_timing"
        return "rule_timing"
    if "open" in token_set or "shut" in token_set:
        return "open_shut"
    if "sink" in token_set:
        return "sink"
    if "defeat" in token_set:
        return "defeat"
    if "hot" in token_set or "melt" in token_set or "lava" in token_set:
        return "hot_melt"
    if "win" in token_set or "goal" in token_set or "terminates" in token_set:
        return "win_termination"
    if "move" in token_set and "shift" in token_set:
        return "move_shift_interaction"
    if "shift" in token_set:
        return "shift"
    if "move" in token_set:
        return "move"
    if "conjunction" in token_set or (
        "and" in token_set
        and any(
            marker in token_set
            for marker in ("subjects", "predicates", "mixed", "leading", "trailing", "multiple")
        )
    ):
        return "rule_conjunction"
    if "adjacent" in token_set or "overlapping" in token_set:
        return "multi_agent_lockstep"
    return "general"


def _humanize_test_method(name: str) -> str:
    text = str(name).strip()
    if text.startswith("test_"):
        text = text[5:]
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def _build_description(
    *,
    base_name: str,
    source_test: str,
    setup_actions: Sequence[str],
    focal_action: str,
    total_steps: int,
    focus_step_index: int,
) -> str:
    setup_text = ", ".join(setup_actions) if setup_actions else "none"
    if total_steps <= 1:
        return (
            f"Minimal focused map from `{source_test}`. "
            f"Evaluate the single transition `{focal_action}` for `{base_name}`."
        )
    return (
        f"Minimal focused map from `{source_test}`. "
        f"Replay setup actions [{setup_text}] and evaluate focal step "
        f"{focus_step_index + 1}/{total_steps} with action `{focal_action}` for `{base_name}`."
    )


def _build_tags(tokens: Sequence[str], category: str) -> Tuple[str, ...]:
    priority = [
        "you",
        "move",
        "shift",
        "win",
        "goal",
        "defeat",
        "sink",
        "hot",
        "melt",
        "open",
        "shut",
        "and",
        "create",
        "break",
        "replace",
        "replacement",
        "transform",
        "transforms",
        "color",
        "prefix",
        "vertical",
        "parse",
        "parses",
        "stack",
        "top",
        "buried",
        "rule",
        "block",
        "blocks",
        "truncates",
        "limit",
        "idle",
        "push",
        "collision",
        "overlapping",
        "adjacent",
    ]
    tags = [category]
    for token in priority:
        if token in tokens:
            tags.append(token)
    return tuple(dict.fromkeys(tags))


def _summarize_state_complexity(state_json: str) -> Tuple[Tuple[int, int], int, int, int]:
    parsed = parse_state_json(state_json)
    raw_grid = parsed.get("grid_size")
    grid_size = (0, 0)
    if (
        isinstance(raw_grid, list)
        and len(raw_grid) == 2
        and isinstance(raw_grid[0], Integral)
        and isinstance(raw_grid[1], Integral)
    ):
        grid_size = (int(raw_grid[0]), int(raw_grid[1]))

    objects = parsed.get("objects")
    if not isinstance(objects, list):
        return grid_size, 0, 0, 0

    world_object_count = 0
    rule_block_count = 0
    for row in objects:
        if not isinstance(row, Mapping):
            continue
        if row.get("type") == "world_object":
            world_object_count += 1
        else:
            rule_block_count += 1
    return grid_size, len(objects), world_object_count, rule_block_count


def _resolve_custom_map_token_words(token: Any) -> Tuple[str, ...]:
    raw_token = str(token or "").strip()
    if not raw_token:
        return ()
    return tuple(
        word
        for word in (
            ASCII_OBJECTS.get(raw_token),
            ASCII_RULE_OBJECTS.get(raw_token),
            ASCII_RULE_OPERATORS.get(raw_token),
            ASCII_RULE_PROPERTIES.get(raw_token),
        )
        if isinstance(word, str) and word.strip()
    )


def _extract_words_from_custom_map_spec(spec: Mapping[str, Any]) -> Tuple[str, ...]:
    words: List[str] = []

    for row in spec.get("layout") or []:
        if not isinstance(row, str):
            continue
        for token in row:
            words.extend(_resolve_custom_map_token_words(token))

    cell_stacks = spec.get("cell_stacks") or {}
    if isinstance(cell_stacks, Mapping):
        for entries in cell_stacks.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                words.extend(_resolve_custom_map_token_words(entry.get("token")))

    inactive_rules = spec.get("inactive_rules") or []
    if isinstance(inactive_rules, list):
        for row in inactive_rules:
            if not isinstance(row, (list, tuple)):
                continue
            for item in row:
                raw_item = str(item or "").strip()
                if not raw_item:
                    continue
                token_words = _resolve_custom_map_token_words(raw_item)
                if token_words:
                    words.extend(token_words)
                else:
                    words.append(raw_item.lower())

    return tuple(dict.fromkeys(word.strip().lower() for word in words if str(word).strip()))


@lru_cache(maxsize=None)
def get_allowed_example_words_from_custom_map_splits(
    difficulty: str = DEFAULT_ALLOWED_CUSTOM_MAP_DIFFICULTY,
    split_names: Tuple[str, ...] = DEFAULT_ALLOWED_CUSTOM_MAP_SPLITS,
) -> Tuple[str, ...]:
    allowed_words: List[str] = []
    for split_name in split_names:
        catalog = load_custom_map_catalog(difficulty, scenario_split=split_name)
        for row in catalog.values():
            if not isinstance(row, Mapping):
                continue
            spec = row.get("spec")
            if not isinstance(spec, Mapping):
                continue
            allowed_words.extend(_extract_words_from_custom_map_spec(spec))
    return tuple(dict.fromkeys(word for word in allowed_words if word))


def _extract_words_from_state_json(state_json: str) -> Tuple[str, ...]:
    parsed = parse_state_json(state_json)
    objects = parsed.get("objects")
    if not isinstance(objects, list):
        return ()
    words: List[str] = []
    for row in objects:
        if not isinstance(row, Mapping):
            continue
        raw_word = row.get("word")
        if not isinstance(raw_word, str):
            continue
        word = raw_word.strip().lower()
        if word:
            words.append(word)
    return tuple(dict.fromkeys(words))


def filter_examples_by_allowed_words(
    examples: Sequence[DynamicsExample],
    allowed_words: Iterable[str],
) -> List[DynamicsExample]:
    allowed_word_set = {
        str(word).strip().lower()
        for word in allowed_words
        if isinstance(word, str) and word.strip()
    }
    if len(allowed_word_set) == 0:
        return list(examples)

    filtered: List[DynamicsExample] = []
    for example in examples:
        example_words = set(_extract_words_from_state_json(example.transition.state))
        example_words.update(_extract_words_from_state_json(example.transition.next_state))
        if example_words.issubset(allowed_word_set):
            filtered.append(example)
    return filtered


def build_test_dynamics_examples(
    test_paths: Sequence[Path] = DEFAULT_TEST_PATHS,
    *,
    allowed_words: Optional[Iterable[str]] = None,
    filter_to_default_map_words: bool = True,
) -> List[DynamicsExample]:
    cases = discover_test_cases(test_paths=test_paths)
    examples: List[DynamicsExample] = []
    serializer = StateSerializer(format_type="json")

    for case in cases:
        module = _load_module_from_path(str(case.source_path.resolve()))
        env_cls = getattr(module, case.env_class_name)
        total_steps = len(case.action_sequence)
        base_name = _humanize_test_method(case.test_method_name)
        tokens = tuple(str(case.test_method_name).removeprefix("test_").split("_"))
        category = _derive_category(tokens)
        tags = _build_tags(tokens, category)

        env = env_cls()
        try:
            env.reset(seed=0)
            current_state_json = serialize_raw_baba_state(
                env,
                terminated=bool(getattr(env, "is_win", False)),
                truncated=False,
                serializer=serializer,
            )
            setup_actions: List[str] = []

            for focus_step_index, action_name in enumerate(case.action_sequence):
                before_state_json = current_state_json
                next_state_json, reward, terminated, truncated = _step_env(
                    env,
                    action_name=action_name,
                    serializer=serializer,
                )

                example_id = (
                    f"{case.source_path.stem}.{case.test_method_name}.step{focus_step_index + 1}"
                )
                concept_name = base_name
                if total_steps > 1:
                    concept_name = f"{base_name} | step {focus_step_index + 1}/{total_steps}"

                grid_size, object_count, world_object_count, rule_block_count = (
                    _summarize_state_complexity(before_state_json)
                )
                examples.append(
                    DynamicsExample(
                        example_id=example_id,
                        concept_name=concept_name,
                        category=category,
                        description=_build_description(
                            base_name=base_name,
                            source_test=case.test_method_name,
                            setup_actions=setup_actions,
                            focal_action=action_name,
                            total_steps=total_steps,
                            focus_step_index=focus_step_index,
                        ),
                        source_kind="test_case",
                        source_ref=f"{case.source_path}:{case.test_method_name}",
                        env_class_name=case.env_class_name,
                        setup_actions=tuple(setup_actions),
                        focal_action=action_name,
                        transition=Transition(
                            state=before_state_json,
                            action=action_name,
                            next_state=next_state_json,
                            reward=reward,
                            done=bool(terminated or truncated),
                        ),
                        expected_reward=reward,
                        expected_terminated=terminated,
                        expected_truncated=truncated,
                        grid_size=grid_size,
                        object_count=object_count,
                        world_object_count=world_object_count,
                        rule_block_count=rule_block_count,
                        tags=tags,
                    )
                )
                current_state_json = next_state_json
                setup_actions.append(action_name)
        finally:
            env.close()

    active_allowed_words = allowed_words
    if active_allowed_words is None and filter_to_default_map_words:
        active_allowed_words = get_allowed_example_words_from_custom_map_splits()
    if active_allowed_words is not None:
        return filter_examples_by_allowed_words(examples, active_allowed_words)
    return examples


def build_catalog_report(*, examples: Sequence[DynamicsExample]) -> Dict[str, Any]:
    category_counter = Counter(example.category for example in examples)
    source_counter = Counter(example.source_kind for example in examples)
    example_rows = [{**example.to_dict(), "solved_by_versions": [], "first_solved_by_version": None} for example in examples]
    return {
        "catalog_summary": {
            "total_examples": len(examples),
            "category_counts": dict(sorted(category_counter.items())),
            "source_kind_counts": dict(sorted(source_counter.items())),
        },
        "examples": example_rows,
        "versions": [],
    }


def load_manual_transition_examples(paths: Sequence[Path]) -> List[DynamicsExample]:
    examples: List[DynamicsExample] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Manual transition file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        previous_state_raw = payload.get("previous_state_raw")
        next_state_raw = payload.get("actual_next_state_raw", payload.get("next_state_raw"))
        action_name = str(payload.get("action", "")).strip()
        if not isinstance(previous_state_raw, str) or not previous_state_raw.strip():
            raise ValueError(f"{path} is missing `previous_state_raw`.")
        if not isinstance(next_state_raw, str) or not next_state_raw.strip():
            raise ValueError(
                f"{path} is missing `actual_next_state_raw` or `next_state_raw`."
            )
        if not action_name:
            raise ValueError(f"{path} is missing `action`.")

        concept_name = re.sub(r"\s+", " ", path.stem.replace("_", " ")).strip()
        grid_size, object_count, world_object_count, rule_block_count = (
            _summarize_state_complexity(previous_state_raw)
        )
        examples.append(
            DynamicsExample(
                example_id=f"manual.{path.stem}",
                concept_name=concept_name,
                category="manual_transition",
                description=(
                    f"Manual transition artifact from `{path.name}`. "
                    f"Evaluate the recorded `{action_name}` transition exactly as captured."
                ),
                source_kind="manual_json",
                source_ref=str(path.resolve()),
                env_class_name=None,
                setup_actions=(),
                focal_action=action_name,
                transition=Transition(
                    state=previous_state_raw,
                    action=action_name,
                    next_state=next_state_raw,
                    reward=float(payload.get("reward", 0.0) or 0.0),
                    done=bool(payload.get("terminated", False) or payload.get("truncated", False)),
                ),
                expected_reward=float(payload.get("reward", 0.0) or 0.0),
                expected_terminated=bool(payload.get("terminated", False)),
                expected_truncated=bool(payload.get("truncated", False)),
                grid_size=grid_size,
                object_count=object_count,
                world_object_count=world_object_count,
                rule_block_count=rule_block_count,
                tags=("manual_transition", action_name),
            )
        )
    return examples


def load_program_sources(
    *,
    experiment_dir: Optional[Path] = None,
    program_dir: Optional[Path] = None,
    program_files: Optional[Sequence[Path]] = None,
) -> List[ProgramSource]:
    resolved: List[ProgramSource] = []

    if experiment_dir is not None:
        candidate_dir = Path(experiment_dir) / "program_versions"
        if not candidate_dir.exists():
            raise FileNotFoundError(f"Program versions dir not found: {candidate_dir}")
        program_dir = candidate_dir

    if program_dir is not None:
        for path in sorted(Path(program_dir).glob("v*.py")):
            resolved.append(
                ProgramSource(
                    version_id=path.stem,
                    path=path.resolve(),
                    source=path.read_text(encoding="utf-8"),
                )
            )

    for path in program_files or ():
        resolved.append(
            ProgramSource(
                version_id=Path(path).stem,
                path=Path(path).resolve(),
                source=Path(path).read_text(encoding="utf-8"),
            )
        )

    deduped: List[ProgramSource] = []
    seen: set[str] = set()
    for row in resolved:
        if row.version_id in seen:
            continue
        deduped.append(row)
        seen.add(row.version_id)

    if len(deduped) == 0:
        raise ValueError("No program sources were resolved.")
    return deduped


def evaluate_examples_against_programs(
    *,
    examples: Sequence[DynamicsExample],
    program_sources: Sequence[ProgramSource],
    evaluator: Optional[ProgramEvaluator] = None,
) -> Dict[str, Any]:
    if len(examples) == 0:
        raise ValueError("At least one dynamics example is required.")

    active_evaluator = evaluator or ProgramEvaluator()
    transitions = [example.transition for example in examples]

    version_rows: List[Dict[str, Any]] = []
    solved_so_far: set[str] = set()
    solved_by_example: Dict[str, List[str]] = {example.example_id: [] for example in examples}

    for program in program_sources:
        evaluation = active_evaluator.evaluate_source(
            source=program.source,
            transitions=transitions,
        )
        example_results: List[ExampleEvaluationResult] = []
        matched_ids: List[str] = []
        matched_by_category: Counter[str] = Counter()

        for example, record in zip(examples, evaluation.records):
            error_phase = getattr(record.error, "phase", None) if record.error is not None else None
            error_message = getattr(record.error, "message", None) if record.error is not None else None
            is_correct = bool(record.is_correct) and record.error is None
            if is_correct:
                matched_ids.append(example.example_id)
                matched_by_category[example.category] += 1
                solved_by_example[example.example_id].append(program.version_id)

            example_results.append(
                ExampleEvaluationResult(
                    example_id=example.example_id,
                    is_correct=is_correct,
                    predicted_next_state_json=record.predicted_canonical,
                    error_phase=(str(error_phase) if isinstance(error_phase, str) else None),
                    error_message=(str(error_message) if isinstance(error_message, str) else None),
                )
            )

        new_matches = sorted(set(matched_ids) - solved_so_far)
        solved_so_far.update(matched_ids)
        compile_errors = [
            {
                "phase": str(getattr(error, "phase", "")).strip(),
                "message": str(getattr(error, "message", "")).strip(),
            }
            for error in evaluation.compile_errors
        ]

        version_rows.append(
            {
                "program": program.to_dict(),
                "matched_example_count": len(matched_ids),
                "matched_example_ratio": (
                    float(len(matched_ids)) / float(len(examples))
                    if len(examples) > 0
                    else 0.0
                ),
                "matched_example_ids": matched_ids,
                "matched_by_category": dict(sorted(matched_by_category.items())),
                "newly_solved_example_ids": new_matches,
                "cumulative_solved_example_count": len(solved_so_far),
                "compile_errors": compile_errors,
                "runtime_error_count": int(evaluation.runtime_error_count),
                "evaluation_results": [row.to_dict() for row in example_results],
            }
        )

    example_rows: List[Dict[str, Any]] = []
    for example in examples:
        solved_versions = solved_by_example.get(example.example_id, [])
        example_rows.append(
            {
                **example.to_dict(),
                "solved_by_versions": solved_versions,
                "first_solved_by_version": solved_versions[0] if solved_versions else None,
            }
        )

    category_counter = Counter(example.category for example in examples)
    source_counter = Counter(example.source_kind for example in examples)

    return {
        "catalog_summary": {
            "total_examples": len(examples),
            "category_counts": dict(sorted(category_counter.items())),
            "source_kind_counts": dict(sorted(source_counter.items())),
        },
        "examples": example_rows,
        "versions": version_rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines: List[str] = ["# Dynamics Coverage Report", ""]

    catalog_summary = report.get("catalog_summary") or {}
    lines.append(f"- total_examples: {int(catalog_summary.get('total_examples', 0) or 0)}")
    category_counts = catalog_summary.get("category_counts") or {}
    source_kind_counts = catalog_summary.get("source_kind_counts") or {}
    lines.append(f"- category_count: {len(category_counts)}")
    lines.append(f"- source_kind_count: {len(source_kind_counts)}")
    lines.append("")

    lines.append("## Category Summary")
    lines.append("")
    lines.append("| category | examples |")
    lines.append("| --- | ---: |")
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {int(count)} |")
    lines.append("")

    lines.append("## Version Coverage")
    lines.append("")
    lines.append("| version | solved | cumulative | new | compile_errors | runtime_errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for version_row in report.get("versions") or []:
        program = version_row.get("program") or {}
        lines.append(
            "| "
            f"{program.get('version_id', 'unknown')} | "
            f"{int(version_row.get('matched_example_count', 0) or 0)} | "
            f"{int(version_row.get('cumulative_solved_example_count', 0) or 0)} | "
            f"{len(version_row.get('newly_solved_example_ids') or [])} | "
            f"{len(version_row.get('compile_errors') or [])} | "
            f"{int(version_row.get('runtime_error_count', 0) or 0)} |"
        )
    lines.append("")

    lines.append("## Example Catalog")
    lines.append("")
    lines.append(
        "| example_id | category | first_version | setup | action | blocks | description |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | --- |")
    for example in report.get("examples") or []:
        setup_actions = ", ".join(example.get("setup_actions") or []) or "-"
        first_version = example.get("first_solved_by_version") or "-"
        description = str(example.get("description", "")).replace("|", "\\|")
        lines.append(
            "| "
            f"{example.get('example_id')} | "
            f"{example.get('category')} | "
            f"{first_version} | "
            f"{setup_actions} | "
            f"{example.get('focal_action')} | "
            f"{int(example.get('object_count', 0) or 0)} | "
            f"{description} |"
        )
    lines.append("")

    return "\n".join(lines)
