from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from scripts.data_collect.data_coverage import archive_state_input_to_state_obj
from scripts.data_collect.evaluate_offline_dataset import (
    OfflineScenarioDataset,
    _load_state_archive_inputs,
    load_offline_dataset,
)
from src.data import StateStore, Transition


_OFFLINE_EVAL_PACKAGE_FORMAT_VERSION = 1
_OFFLINE_EVAL_PACKAGE_DIRNAME = ".offline_eval_package"


@dataclass(frozen=True)
class CompiledOfflineEvalBundle:
    bundle_id: int
    scenario_type: str
    artifact_stem: str
    transitions_path: Path
    states_path: Path
    transition_count: int
    source_transition_count: int

    @property
    def bundle_key(self) -> str:
        return str(self.transitions_path.resolve())


@dataclass(frozen=True)
class CompiledOfflineEvalTransitionMeta:
    row_id: int
    bundle: CompiledOfflineEvalBundle
    transition_index: int
    state_id: int
    next_state_id: int
    action: str
    reward: float
    done: bool


class CompiledOfflineEvalPackage:
    def __init__(
        self,
        *,
        package_root: Path,
        dataset_root: Path,
        signature: str,
        state_store_root: Path,
        state_store: Optional[StateStore],
        bundles: Sequence[CompiledOfflineEvalBundle],
        action_vocab: Sequence[str],
        bundle_ids: np.ndarray,
        transition_indices: np.ndarray,
        state_ids: np.ndarray,
        next_state_ids: np.ndarray,
        action_ids: np.ndarray,
        rewards: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.package_root = Path(package_root).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.signature = str(signature)
        self._state_store_root = Path(state_store_root).resolve()
        self._state_store = state_store
        self.bundles = tuple(bundles)
        self.action_vocab = tuple(str(value) for value in action_vocab)
        self.bundle_ids = bundle_ids
        self.transition_indices = transition_indices
        self.state_ids = state_ids
        self.next_state_ids = next_state_ids
        self.action_ids = action_ids
        self.rewards = rewards
        self.done = done
        self._bundle_by_id = {int(bundle.bundle_id): bundle for bundle in self.bundles}
        self._bundle_by_key = {bundle.bundle_key: bundle for bundle in self.bundles}
        self._bundle_row_ranges: Dict[int, tuple[int, int]] = {}
        row_offset = 0
        total_rows = int(len(self.bundle_ids))
        for bundle in self.bundles:
            start = int(row_offset)
            end = min(total_rows, start + max(0, int(bundle.transition_count)))
            self._bundle_row_ranges[int(bundle.bundle_id)] = (start, end)
            row_offset = end
        self._transition_row_lookup_cache: OrderedDict[int, Dict[int, int]] = OrderedDict()
        self._transition_row_lookup_cache_limit = 4

    @staticmethod
    def _close_mmap_array(value: np.ndarray) -> None:
        current = value
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            mmap_handle = getattr(current, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
                return
            current = getattr(current, "base", None)

    def close(self) -> None:
        if self._state_store is not None:
            self._state_store.close()
            self._state_store = None
        for value in (
            self.bundle_ids,
            self.transition_indices,
            self.state_ids,
            self.next_state_ids,
            self.action_ids,
            self.rewards,
            self.done,
        ):
            self._close_mmap_array(value)

    @property
    def transition_count(self) -> int:
        return int(len(self.bundle_ids))

    @property
    def state_store(self) -> StateStore:
        if self._state_store is None:
            self._state_store = StateStore.load_directory(self._state_store_root)
        return self._state_store

    def resolve_bundle(self, bundle_key: str) -> Optional[CompiledOfflineEvalBundle]:
        return self._bundle_by_key.get(str(bundle_key))

    def bundle_row_range(self, bundle_id: int) -> tuple[int, int]:
        return self._bundle_row_ranges.get(int(bundle_id), (0, 0))

    def count_transition_rows(
        self,
        *,
        allowed_scenario_types: Optional[Sequence[str]] = None,
    ) -> int:
        if allowed_scenario_types is None:
            return int(self.transition_count)
        allowed = {
            str(value).strip()
            for value in allowed_scenario_types
            if str(value).strip()
        }
        if not allowed:
            return int(self.transition_count)
        total = 0
        for bundle in self.bundles:
            if str(bundle.scenario_type).strip() not in allowed:
                continue
            start, end = self.bundle_row_range(int(bundle.bundle_id))
            total += max(0, int(end) - int(start))
        return int(total)

    def iter_transition_row_ids(
        self,
        *,
        allowed_scenario_types: Optional[Sequence[str]] = None,
    ) -> Iterable[int]:
        allowed = (
            {
                str(value).strip()
                for value in allowed_scenario_types
                if str(value).strip()
            }
            if allowed_scenario_types is not None
            else None
        )
        for bundle in self.bundles:
            if allowed is not None and str(bundle.scenario_type).strip() not in allowed:
                continue
            start, end = self.bundle_row_range(int(bundle.bundle_id))
            yield from range(int(start), int(end))

    def _transition_row_lookup_for_bundle(self, bundle_id: int) -> Dict[int, int]:
        resolved_bundle_id = int(bundle_id)
        cached = self._transition_row_lookup_cache.get(resolved_bundle_id)
        if cached is not None:
            self._transition_row_lookup_cache.move_to_end(resolved_bundle_id)
            return cached
        start, end = self.bundle_row_range(resolved_bundle_id)
        lookup: Dict[int, int] = {}
        for row_id in range(int(start), int(end)):
            lookup[int(self.transition_indices[row_id])] = int(row_id)
        self._transition_row_lookup_cache[resolved_bundle_id] = lookup
        self._transition_row_lookup_cache.move_to_end(resolved_bundle_id)
        while len(self._transition_row_lookup_cache) > int(self._transition_row_lookup_cache_limit):
            self._transition_row_lookup_cache.popitem(last=False)
        return lookup

    def resolve_transition_row_id(self, *, bundle_key: str, transition_index: int) -> Optional[int]:
        bundle = self.resolve_bundle(bundle_key)
        if bundle is None:
            return None
        resolved_transition_index = int(transition_index)
        start, end = self.bundle_row_range(int(bundle.bundle_id))
        candidate_row = int(start) + resolved_transition_index
        if (
            int(start) <= candidate_row < int(end)
            and int(self.bundle_ids[candidate_row]) == int(bundle.bundle_id)
            and int(self.transition_indices[candidate_row]) == resolved_transition_index
        ):
            return int(candidate_row)
        return self._transition_row_lookup_for_bundle(int(bundle.bundle_id)).get(
            resolved_transition_index
        )

    def transition_meta(self, row_id: int) -> CompiledOfflineEvalTransitionMeta:
        resolved_row_id = int(row_id)
        if resolved_row_id < 0 or resolved_row_id >= self.transition_count:
            raise IndexError(f"Invalid compiled offline eval transition row: {resolved_row_id}")
        bundle = self._bundle_by_id[int(self.bundle_ids[resolved_row_id])]
        action_id = int(self.action_ids[resolved_row_id])
        action = self.action_vocab[action_id] if 0 <= action_id < len(self.action_vocab) else ""
        return CompiledOfflineEvalTransitionMeta(
            row_id=resolved_row_id,
            bundle=bundle,
            transition_index=int(self.transition_indices[resolved_row_id]),
            state_id=int(self.state_ids[resolved_row_id]),
            next_state_id=int(self.next_state_ids[resolved_row_id]),
            action=str(action),
            reward=float(self.rewards[resolved_row_id]),
            done=bool(self.done[resolved_row_id]),
        )

    def build_transition(self, row_id: int) -> Transition:
        meta = self.transition_meta(int(row_id))
        return Transition.from_state_ids(
            state_store=self.state_store,
            state_id=int(meta.state_id),
            action=str(meta.action),
            next_state_id=int(meta.next_state_id),
            reward=float(meta.reward),
            done=bool(meta.done),
            map_name=str(meta.bundle.scenario_type),
        )


def _package_stat_signature(path: Path) -> Dict[str, Any]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "mtimeNs": int(getattr(stat_result, "st_mtime_ns", 0)),
        "size": int(getattr(stat_result, "st_size", 0)),
    }


def _offline_eval_package_signature(
    *,
    dataset_root: Path,
    bundles: Sequence[OfflineScenarioDataset],
) -> str:
    payload = {
        "formatVersion": int(_OFFLINE_EVAL_PACKAGE_FORMAT_VERSION),
        "datasetRoot": str(Path(dataset_root).resolve()),
        "bundles": [
            {
                "scenarioType": str(bundle.scenario_type),
                "artifactStem": str(bundle.artifact_stem),
                "transitionCount": int(bundle.transition_count),
                "sourceTransitionCount": int(bundle.source_transition_count or bundle.transition_count),
                "transitions": _package_stat_signature(Path(bundle.transitions_path).resolve()),
                "states": _package_stat_signature(Path(bundle.states_path).resolve()),
            }
            for bundle in bundles
        ],
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_transition_rows(
    *,
    bundle: OfflineScenarioDataset,
    state_ids_by_index: Sequence[int],
    action_vocab: List[str],
    action_id_by_text: Dict[str, int],
    bundle_ids: List[int],
    transition_indices: List[int],
    state_ids: List[int],
    next_state_ids: List[int],
    action_ids: List[int],
    rewards: List[float],
    done: List[bool],
    bundle_id: int,
) -> None:
    line_index = 0
    with Path(bundle.transitions_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected JSON object row in {bundle.transitions_path}")
            transition_index = int(row.get("transition_index", line_index))
            state_index = int(row["state_index"])
            next_state_index = int(row["next_state_index"])
            if state_index < 0 or state_index >= len(state_ids_by_index):
                raise IndexError(f"state_index {state_index} out of bounds for {bundle.transitions_path}")
            if next_state_index < 0 or next_state_index >= len(state_ids_by_index):
                raise IndexError(f"next_state_index {next_state_index} out of bounds for {bundle.transitions_path}")
            action = str(row.get("action", "")).strip()
            action_id = action_id_by_text.get(action)
            if action_id is None:
                action_id = len(action_vocab)
                action_id_by_text[action] = int(action_id)
                action_vocab.append(action)

            bundle_ids.append(int(bundle_id))
            transition_indices.append(int(transition_index))
            state_ids.append(int(state_ids_by_index[state_index]))
            next_state_ids.append(int(state_ids_by_index[next_state_index]))
            action_ids.append(int(action_id))
            rewards.append(float(row.get("reward", 0.0) or 0.0))
            done.append(bool(row.get("done", False) or row.get("terminated", False) or row.get("truncated", False)))
            line_index += 1


def _write_compiled_offline_eval_package(
    *,
    package_root: Path,
    dataset_root: Path,
    bundles: Sequence[OfflineScenarioDataset],
    signature: str,
) -> None:
    state_store = StateStore()
    action_vocab: List[str] = []
    action_id_by_text: Dict[str, int] = {}
    bundle_ids: List[int] = []
    transition_indices: List[int] = []
    state_ids: List[int] = []
    next_state_ids: List[int] = []
    action_ids: List[int] = []
    rewards: List[float] = []
    done: List[bool] = []
    bundle_rows: List[Dict[str, Any]] = []

    for bundle_id, bundle in enumerate(bundles):
        state_payloads = _load_state_archive_inputs(Path(bundle.states_path))
        state_ids_by_index: List[int] = []
        for state_payload in state_payloads:
            state_obj = archive_state_input_to_state_obj(state_payload, normalize_input=False)
            state_ids_by_index.append(int(state_store.intern_state_obj(state_obj)))

        _read_transition_rows(
            bundle=bundle,
            state_ids_by_index=state_ids_by_index,
            action_vocab=action_vocab,
            action_id_by_text=action_id_by_text,
            bundle_ids=bundle_ids,
            transition_indices=transition_indices,
            state_ids=state_ids,
            next_state_ids=next_state_ids,
            action_ids=action_ids,
            rewards=rewards,
            done=done,
            bundle_id=int(bundle_id),
        )
        bundle_rows.append(
            {
                "bundleId": int(bundle_id),
                "scenarioType": str(bundle.scenario_type),
                "artifactStem": str(bundle.artifact_stem),
                "transitionsPath": str(Path(bundle.transitions_path).resolve()),
                "statesPath": str(Path(bundle.states_path).resolve()),
                "transitionCount": int(bundle.transition_count),
                "sourceTransitionCount": int(bundle.source_transition_count or bundle.transition_count),
            }
        )

    package_root.mkdir(parents=True, exist_ok=True)
    state_store.save_directory(package_root / "state_store")
    (package_root / "bundles.json").write_text(
        json.dumps(bundle_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(package_root / "action_vocab.npy", np.asarray(action_vocab, dtype=np.str_), allow_pickle=False)
    np.save(package_root / "bundle_ids.npy", np.asarray(bundle_ids, dtype=np.uint32), allow_pickle=False)
    np.save(
        package_root / "transition_indices.npy",
        np.asarray(transition_indices, dtype=np.uint32),
        allow_pickle=False,
    )
    np.save(package_root / "state_ids.npy", np.asarray(state_ids, dtype=np.uint32), allow_pickle=False)
    np.save(
        package_root / "next_state_ids.npy",
        np.asarray(next_state_ids, dtype=np.uint32),
        allow_pickle=False,
    )
    np.save(package_root / "action_ids.npy", np.asarray(action_ids, dtype=np.uint32), allow_pickle=False)
    np.save(package_root / "rewards.npy", np.asarray(rewards, dtype=np.float32), allow_pickle=False)
    np.save(package_root / "done.npy", np.asarray(done, dtype=np.bool_), allow_pickle=False)
    (package_root / "manifest.json").write_text(
        json.dumps(
            {
                "formatVersion": int(_OFFLINE_EVAL_PACKAGE_FORMAT_VERSION),
                "signature": str(signature),
                "datasetRoot": str(Path(dataset_root).resolve()),
                "bundleCount": int(len(bundle_rows)),
                "transitionCount": int(len(bundle_ids)),
                "stateCount": int(len(state_store)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_compiled_offline_eval_package(
    *,
    package_root: Path,
    dataset_root: Path,
    signature: str,
) -> CompiledOfflineEvalPackage:
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Invalid offline eval package manifest: {manifest_path}")
    if int(manifest.get("formatVersion", 0) or 0) != _OFFLINE_EVAL_PACKAGE_FORMAT_VERSION:
        raise ValueError(f"Unsupported offline eval package version: {manifest_path}")
    if str(manifest.get("signature", "")).strip() != str(signature):
        raise ValueError(f"Offline eval package signature mismatch: {package_root}")

    bundle_payload = json.loads((package_root / "bundles.json").read_text(encoding="utf-8"))
    if not isinstance(bundle_payload, list):
        raise ValueError(f"Invalid offline eval package bundles payload: {package_root / 'bundles.json'}")
    bundles: List[CompiledOfflineEvalBundle] = []
    for row in bundle_payload:
        if not isinstance(row, Mapping):
            continue
        bundles.append(
            CompiledOfflineEvalBundle(
                bundle_id=int(row["bundleId"]),
                scenario_type=str(row["scenarioType"]),
                artifact_stem=str(row["artifactStem"]),
                transitions_path=Path(str(row["transitionsPath"])).resolve(),
                states_path=Path(str(row["statesPath"])).resolve(),
                transition_count=int(row["transitionCount"]),
                source_transition_count=int(row["sourceTransitionCount"]),
            )
        )

    return CompiledOfflineEvalPackage(
        package_root=package_root,
        dataset_root=dataset_root,
        signature=signature,
        state_store_root=package_root / "state_store",
        state_store=None,
        bundles=bundles,
        action_vocab=[
            str(value)
            for value in np.load(package_root / "action_vocab.npy", allow_pickle=False).tolist()
        ],
        bundle_ids=np.load(package_root / "bundle_ids.npy", allow_pickle=False, mmap_mode="r"),
        transition_indices=np.load(
            package_root / "transition_indices.npy",
            allow_pickle=False,
            mmap_mode="r",
        ),
        state_ids=np.load(package_root / "state_ids.npy", allow_pickle=False, mmap_mode="r"),
        next_state_ids=np.load(
            package_root / "next_state_ids.npy",
            allow_pickle=False,
            mmap_mode="r",
        ),
        action_ids=np.load(package_root / "action_ids.npy", allow_pickle=False, mmap_mode="r"),
        rewards=np.load(package_root / "rewards.npy", allow_pickle=False, mmap_mode="r"),
        done=np.load(package_root / "done.npy", allow_pickle=False, mmap_mode="r"),
    )


def load_or_build_compiled_offline_eval_package(
    *,
    dataset_root: str | Path,
) -> CompiledOfflineEvalPackage:
    resolved_dataset_root = Path(dataset_root).resolve()
    bundles = load_offline_dataset(
        dataset_root=resolved_dataset_root,
        show_progress=False,
    )
    signature = _offline_eval_package_signature(
        dataset_root=resolved_dataset_root,
        bundles=bundles,
    )
    package_parent = resolved_dataset_root / _OFFLINE_EVAL_PACKAGE_DIRNAME
    package_root = package_parent / str(signature)
    if (package_root / "manifest.json").is_file():
        return _load_compiled_offline_eval_package(
            package_root=package_root,
            dataset_root=resolved_dataset_root,
            signature=signature,
        )

    temp_root = package_parent / f".tmp_{uuid.uuid4().hex}"
    _write_compiled_offline_eval_package(
        package_root=temp_root,
        dataset_root=resolved_dataset_root,
        bundles=bundles,
        signature=signature,
    )
    package_parent.mkdir(parents=True, exist_ok=True)
    try:
        temp_root.replace(package_root)
    except FileExistsError:
        pass
    finally:
        if temp_root.exists():
            for child in sorted(temp_root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            if temp_root.exists():
                temp_root.rmdir()

    return _load_compiled_offline_eval_package(
        package_root=package_root,
        dataset_root=resolved_dataset_root,
        signature=signature,
    )
