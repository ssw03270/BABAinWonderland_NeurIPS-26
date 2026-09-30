from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from scripts.data_collect.evaluate_offline_dataset import (
    _append_bundle_lookup,
    _resolve_heuristic_mapping_bundle,
)
from src.data import Transition
from src.data.state_schema import _normalize_word_aliases, dump_state_json

from .offline_eval_package import (
    CompiledOfflineEvalPackage,
    CompiledOfflineEvalTransitionMeta,
    load_or_build_compiled_offline_eval_package,
)


_DISCOVERY_JSON_READ_CHUNK_SIZE = 1024 * 1024


class _DiscoveryJsonArrayMissing(ValueError):
    pass


@dataclass(frozen=True)
class HeuristicEvalRef:
    class_idx: int
    transition_row_id: int
    bundle_key: str
    transition_index: int
    scenario_type: str
    artifact_stem: str


@dataclass(frozen=True)
class HeuristicEvalEvalCase:
    class_idx: int
    action: str
    transition_count: int
    artifact_stem: str
    scenario_type: str
    transitions_path: Path
    states_path: Path
    transition_index: int
    transition: Transition


@dataclass(frozen=True)
class HeuristicEvalCase(HeuristicEvalEvalCase):
    previous_state_archive: Dict[str, Any]
    next_state_archive: Dict[str, Any]
    previous_state_obj: Dict[str, Any]
    next_state_obj: Dict[str, Any]


def _iter_discovery_json_array_items(
    path: Path | str,
    key: str,
    *,
    required: bool = True,
) -> Iterator[Any]:
    resolved_path = Path(path)
    decoder = json.JSONDecoder()
    token = f'"{key}"'
    with resolved_path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while True:
            token_index = buffer.find(token)
            if token_index >= 0:
                break
            chunk = handle.read(_DISCOVERY_JSON_READ_CHUNK_SIZE)
            if not chunk:
                if required:
                    raise _DiscoveryJsonArrayMissing(
                        f"Heuristic discovery JSON is missing `{key}`: {resolved_path}"
                    )
                return
            buffer += chunk
            token_index = buffer.find(token)
            if token_index >= 0:
                break
            if len(buffer) > len(token) * 2:
                buffer = buffer[-len(token) * 2 :]

        index = int(token_index + len(token))
        while True:
            colon_index = buffer.find(":", index)
            if colon_index >= 0:
                index = int(colon_index + 1)
                break
            chunk = handle.read(_DISCOVERY_JSON_READ_CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"Invalid heuristic discovery JSON near `{key}`: {resolved_path}")
            buffer += chunk

        while True:
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if index < len(buffer):
                break
            chunk = handle.read(_DISCOVERY_JSON_READ_CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"Invalid heuristic discovery JSON near `{key}`: {resolved_path}")
            buffer = chunk
            index = 0
        if buffer[index] != "[":
            raise ValueError(f"Heuristic discovery JSON field `{key}` is not an array: {resolved_path}")
        buffer = buffer[index + 1 :]

        while True:
            index = 0
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index < len(buffer):
                    break
                chunk = handle.read(_DISCOVERY_JSON_READ_CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"Unterminated heuristic discovery JSON array `{key}`: {resolved_path}")
                buffer = chunk
                index = 0
            if buffer[index] == "]":
                return
            if buffer[index] == ",":
                buffer = buffer[index + 1 :]
                continue
            while True:
                try:
                    item, end_index = decoder.raw_decode(buffer, index)
                except json.JSONDecodeError as error:
                    chunk = handle.read(_DISCOVERY_JSON_READ_CHUNK_SIZE)
                    if not chunk:
                        raise ValueError(
                            f"Invalid heuristic discovery JSON array `{key}`: {resolved_path}"
                        ) from error
                    buffer += chunk
                    continue
                yield item
                buffer = buffer[end_index:]
                break


def iter_discovery_class_metadata_rows(path: Path | str) -> Iterator[Any]:
    try:
        yield from _iter_discovery_json_array_items(path, "class_metadata", required=False)
    except _DiscoveryJsonArrayMissing:
        return


def iter_discovery_transition_class_mapping_rows(path: Path | str) -> Iterator[Any]:
    yield from _iter_discovery_json_array_items(
        path,
        "transition_class_mapping",
        required=True,
    )


class HeuristicEvalDataset:
    def __init__(
        self,
        *,
        dataset_root: Path | str,
        discovery_json: Path | str | None,
        sample_seed: int,
        word_aliases: Optional[Mapping[str, Mapping[str, Any]]] = None,
        allowed_scenario_types: Optional[Sequence[str]] = None,
        bundle_state_cache_size: int = 128,
        max_loaded_bundle_contexts: Optional[int] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.discovery_json = (
            Path(discovery_json).resolve()
            if discovery_json is not None and str(discovery_json).strip()
            else None
        )
        self.sample_seed = int(sample_seed)
        self.word_aliases = _normalize_word_aliases(word_aliases)
        self.allowed_scenario_types = (
            frozenset(
                str(value).strip()
                for value in allowed_scenario_types
                if str(value).strip()
            )
            if allowed_scenario_types is not None
            else None
        )
        self.bundle_state_cache_size = max(8, int(bundle_state_cache_size))
        self.max_loaded_bundle_contexts = (
            max(1, int(max_loaded_bundle_contexts))
            if max_loaded_bundle_contexts is not None
            else None
        )
        self.class_metadata_by_id: Dict[int, Dict[str, Any]] = {}
        self.class_transition_counts: Dict[int, int] = {}
        self.selected_refs: Dict[int, HeuristicEvalRef] = {}
        self._full_dataset_class_count = 0
        self._transition_view_cache: OrderedDict[int, Transition] = OrderedDict()
        self._package: Optional[CompiledOfflineEvalPackage] = None
        self._loaded = False

    @property
    def class_count(self) -> int:
        if self.discovery_json is None and self._loaded:
            return int(self._full_dataset_class_count)
        return int(len(self.selected_refs))

    @property
    def class_indices(self) -> List[int]:
        return list(self.iter_eval_class_indices())

    @property
    def eval_class_indices(self) -> List[int]:
        return list(self.iter_eval_class_indices())

    def iter_eval_class_indices(self) -> Iterable[int]:
        self.load()
        if self.discovery_json is None:
            package = self._require_package()
            for row_id in package.iter_transition_row_ids(
                allowed_scenario_types=self.allowed_scenario_types,
            ):
                yield int(row_id) + 1
            return
        for class_idx in sorted(
            self.selected_refs.keys(),
            key=lambda value: (
                str(self.selected_refs[int(value)].bundle_key),
                int(self.selected_refs[int(value)].transition_index),
                int(value),
            ),
        ):
            yield int(class_idx)

    @property
    def selection_mode(self) -> str:
        return "heuristic_class" if self.discovery_json is not None else "full_dataset"

    @property
    def result_label_singular(self) -> str:
        return "class" if self.discovery_json is not None else "transition"

    @property
    def result_label_plural(self) -> str:
        return "classes" if self.discovery_json is not None else "transitions"

    def _require_package(self) -> CompiledOfflineEvalPackage:
        if self._package is None:
            raise RuntimeError("Offline eval dataset package has not been loaded.")
        return self._package

    def _scenario_type_allowed(self, scenario_type: str) -> bool:
        if self.allowed_scenario_types is None:
            return True
        return str(scenario_type).strip() in self.allowed_scenario_types

    def _transition_cache_limit(self) -> int:
        bundle_multiplier = int(self.max_loaded_bundle_contexts or 1)
        return max(16, int(self.bundle_state_cache_size) * max(1, bundle_multiplier))

    def _state_obj_with_aliases(self, state_id: int) -> Dict[str, Any]:
        state_obj = self._require_package().state_store.state_obj(int(state_id))
        if not self.word_aliases:
            return state_obj
        objects: List[Dict[str, Any]] = []
        for raw_object in state_obj.get("objects", []):
            if not isinstance(raw_object, Mapping):
                continue
            row = dict(raw_object)
            obj_type = str(row.get("type", "")).strip().lower()
            word = str(row.get("word", "")).strip().lower()
            alias = self.word_aliases.get(obj_type, {}).get(word)
            if alias:
                row["word"] = alias
            objects.append(row)
        aliased_state = dict(state_obj)
        aliased_state["objects"] = objects
        return aliased_state

    def _build_transition_view(self, transition_row_id: int) -> Transition:
        package = self._require_package()
        if not self.word_aliases:
            return package.build_transition(int(transition_row_id))
        meta = package.transition_meta(int(transition_row_id))
        previous_state_obj = self._state_obj_with_aliases(int(meta.state_id))
        next_state_obj = self._state_obj_with_aliases(int(meta.next_state_id))
        return Transition(
            state=dump_state_json(previous_state_obj),
            action=str(meta.action),
            next_state=dump_state_json(next_state_obj),
            reward=float(meta.reward),
            done=bool(meta.done),
            map_name=str(meta.bundle.scenario_type),
        )

    def _get_transition_view(self, transition_row_id: int) -> Transition:
        resolved_row_id = int(transition_row_id)
        cached = self._transition_view_cache.get(resolved_row_id)
        if cached is not None:
            self._transition_view_cache.move_to_end(resolved_row_id)
            return cached
        transition = self._build_transition_view(resolved_row_id)
        self._transition_view_cache[resolved_row_id] = transition
        self._transition_view_cache.move_to_end(resolved_row_id)
        while len(self._transition_view_cache) > self._transition_cache_limit():
            self._transition_view_cache.popitem(last=False)
        return transition

    def load(self) -> None:
        if self._loaded:
            return
        package = load_or_build_compiled_offline_eval_package(
            dataset_root=self.dataset_root,
        )
        self._package = package
        if package.transition_count <= 0:
            raise ValueError(f"No offline dataset bundles found under {self.dataset_root}")

        if self.discovery_json is None:
            self._full_dataset_class_count = package.count_transition_rows(
                allowed_scenario_types=self.allowed_scenario_types,
            )
            if self._full_dataset_class_count <= 0:
                raise ValueError(f"No transitions found under {self.dataset_root}")
            self._loaded = True
            return

        for row in iter_discovery_class_metadata_rows(self.discovery_json):
            if not isinstance(row, Mapping):
                continue
            try:
                class_idx = int(row.get("class_idx"))
            except (TypeError, ValueError):
                continue
            self.class_metadata_by_id[int(class_idx)] = {
                "classIdx": int(class_idx),
                "action": str(row.get("action", "")).strip(),
                "transitionCount": int(row.get("transition_count", 0) or 0),
                "stateDiff": row.get("state_diff", {}),
                "classSignature": row.get("class_signature") or row.get("class_signature_notes"),
                "sampleTransition": row.get("sample_transition"),
            }

        bundle_by_transition_path = {
            str(bundle.transitions_path.resolve()): bundle
            for bundle in package.bundles
        }
        bundles_by_transition_name: Dict[str, List[Any]] = {}
        bundles_by_artifact_stem: Dict[str, List[Any]] = {}
        bundles_by_scenario_type: Dict[str, List[Any]] = {}
        for bundle in package.bundles:
            _append_bundle_lookup(bundles_by_transition_name, bundle.transitions_path.name, bundle)
            _append_bundle_lookup(bundles_by_artifact_stem, str(bundle.artifact_stem), bundle)
            _append_bundle_lookup(bundles_by_scenario_type, str(bundle.scenario_type), bundle)

        rng = random.Random(int(self.sample_seed))
        for row in iter_discovery_transition_class_mapping_rows(self.discovery_json):
            if not isinstance(row, Mapping):
                continue
            transition_to_class = row.get("transition_to_class")
            if not isinstance(transition_to_class, Mapping):
                continue
            bundle = _resolve_heuristic_mapping_bundle(
                row=row,
                bundle_by_transition_path=bundle_by_transition_path,
                bundles_by_transition_name=bundles_by_transition_name,
                bundles_by_artifact_stem=bundles_by_artifact_stem,
                bundles_by_scenario_type=bundles_by_scenario_type,
            )
            if bundle is None:
                continue
            if not self._scenario_type_allowed(str(bundle.scenario_type)):
                continue
            bundle_key = str(bundle.bundle_key)
            for raw_transition_index, raw_class_idx in transition_to_class.items():
                try:
                    transition_index = int(raw_transition_index)
                    class_idx = int(raw_class_idx)
                except (TypeError, ValueError):
                    continue
                transition_row_id = package.resolve_transition_row_id(
                    bundle_key=bundle_key,
                    transition_index=int(transition_index),
                )
                if transition_row_id is None:
                    continue
                ref = HeuristicEvalRef(
                    class_idx=int(class_idx),
                    transition_row_id=int(transition_row_id),
                    bundle_key=bundle_key,
                    transition_index=int(transition_index),
                    scenario_type=str(bundle.scenario_type),
                    artifact_stem=str(bundle.artifact_stem),
                )
                next_count = int(self.class_transition_counts.get(int(class_idx), 0)) + 1
                self.class_transition_counts[int(class_idx)] = next_count
                if int(class_idx) not in self.selected_refs or rng.randrange(next_count) == 0:
                    self.selected_refs[int(class_idx)] = ref
                metadata = self.class_metadata_by_id.setdefault(
                    int(class_idx),
                    {
                        "classIdx": int(class_idx),
                        "action": "",
                        "transitionCount": 0,
                        "stateDiff": {},
                        "classSignature": None,
                        "sampleTransition": None,
                    },
                )
                metadata["transitionCount"] = next_count

        if not self.selected_refs:
            raise ValueError(
                "No heuristic-dynamics classes matched the selected evaluation bundles. "
                f"Source JSON: {self.discovery_json}"
            )
        self._loaded = True

    def get_class_rows(self) -> List[Dict[str, Any]]:
        self.load()
        rows: List[Dict[str, Any]] = []
        for class_idx in self.class_indices:
            metadata = self.class_metadata_by_id.get(int(class_idx), {})
            ref = self.selected_refs[int(class_idx)]
            rows.append(
                {
                    "classIdx": int(class_idx),
                    "action": str(metadata.get("action", "")).strip(),
                    "transitionCount": int(
                        metadata.get(
                            "transitionCount",
                            self.class_transition_counts.get(int(class_idx), 0),
                        )
                    ),
                    "stateDiff": metadata.get("stateDiff", {}),
                    "scenarioType": str(ref.scenario_type),
                    "artifactStem": str(ref.artifact_stem),
                    "transitionIndex": int(ref.transition_index),
                }
            )
        return rows

    def _resolve_case_meta(
        self,
        class_idx: int,
    ) -> tuple[HeuristicEvalRef, CompiledOfflineEvalTransitionMeta, Dict[str, Any]]:
        self.load()
        resolved_class_idx = int(class_idx)
        if self.discovery_json is None:
            transition_row_id = int(resolved_class_idx) - 1
            meta = self._require_package().transition_meta(int(transition_row_id))
            if not self._scenario_type_allowed(str(meta.bundle.scenario_type)):
                raise KeyError(f"Unknown full-dataset transition index: {resolved_class_idx}")
            selected_ref = HeuristicEvalRef(
                class_idx=int(resolved_class_idx),
                transition_row_id=int(meta.row_id),
                bundle_key=str(meta.bundle.bundle_key),
                transition_index=int(meta.transition_index),
                scenario_type=str(meta.bundle.scenario_type),
                artifact_stem=str(meta.bundle.artifact_stem),
            )
            metadata = {
                "classIdx": int(resolved_class_idx),
                "action": str(meta.action),
                "transitionCount": 1,
                "stateDiff": {},
                "classSignature": None,
                "sampleTransition": None,
            }
            return selected_ref, meta, metadata
        selected_ref = self.selected_refs.get(resolved_class_idx)
        if selected_ref is None:
            raise KeyError(f"Unknown heuristic class index: {resolved_class_idx}")
        meta = self._require_package().transition_meta(int(selected_ref.transition_row_id))
        metadata = self.class_metadata_by_id.get(resolved_class_idx, {})
        return selected_ref, meta, metadata

    def get_eval_case(self, class_idx: int) -> HeuristicEvalEvalCase:
        selected_ref, meta, metadata = self._resolve_case_meta(int(class_idx))
        transition = self._get_transition_view(int(selected_ref.transition_row_id))
        action = str(metadata.get("action", meta.action)).strip() or str(meta.action)
        return HeuristicEvalEvalCase(
            class_idx=int(selected_ref.class_idx),
            action=action,
            transition_count=int(
                metadata.get(
                    "transitionCount",
                    self.class_transition_counts.get(int(selected_ref.class_idx), 0),
                )
            ),
            artifact_stem=str(selected_ref.artifact_stem),
            scenario_type=str(selected_ref.scenario_type),
            transitions_path=Path(meta.bundle.transitions_path),
            states_path=Path(meta.bundle.states_path),
            transition_index=int(selected_ref.transition_index),
            transition=transition,
        )

    def _build_state_archive_hint(
        self,
        *,
        state_obj: Mapping[str, Any],
        scenario_type: str,
    ) -> Dict[str, Any]:
        step_payload = state_obj.get("step") if isinstance(state_obj, Mapping) else {}
        return {
            "source": str(scenario_type),
            "terminated": bool(step_payload.get("terminated", False)) if isinstance(step_payload, Mapping) else False,
        }

    def get_case(self, class_idx: int) -> HeuristicEvalCase:
        eval_case = self.get_eval_case(int(class_idx))
        _selected_ref, meta, _metadata = self._resolve_case_meta(int(class_idx))
        previous_state_obj = self._state_obj_with_aliases(int(meta.state_id))
        next_state_obj = self._state_obj_with_aliases(int(meta.next_state_id))
        previous_state_archive = self._build_state_archive_hint(
            state_obj=previous_state_obj,
            scenario_type=str(eval_case.scenario_type),
        )
        next_state_archive = self._build_state_archive_hint(
            state_obj=next_state_obj,
            scenario_type=str(eval_case.scenario_type),
        )
        return HeuristicEvalCase(
            class_idx=int(eval_case.class_idx),
            action=str(eval_case.action),
            transition_count=int(eval_case.transition_count),
            artifact_stem=str(eval_case.artifact_stem),
            scenario_type=str(eval_case.scenario_type),
            transitions_path=Path(eval_case.transitions_path),
            states_path=Path(eval_case.states_path),
            transition_index=int(eval_case.transition_index),
            transition=eval_case.transition,
            previous_state_archive=previous_state_archive,
            next_state_archive=next_state_archive,
            previous_state_obj=previous_state_obj,
            next_state_obj=next_state_obj,
        )

    def close(self) -> None:
        self._transition_view_cache.clear()
        if self._package is not None:
            self._package.close()
        self._package = None
