from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.data import Transition, canonical_graph_edge_identity_key

from .evaluator import ProgramEvaluationTask, ProgramEvaluator, VersionContextSnapshot


CLASS_ASSIGN_TRANSITION_CHUNK_SIZE = 256
CLASS_ASSIGN_ROOT_PROGRAM_BLOCK_SIZE = 24


@dataclass(frozen=True)
class GroupClassMigration:
    parent_group_id: str
    source_class_id: int
    kept_child_group_id: str
    kept_class_id: int
    broken_child_group_id: str
    broken_class_id: int


@dataclass(frozen=True)
class GroupAssignment:
    class_id: int
    group_id: Optional[str]
    status: str


@dataclass(frozen=True)
class TransitionGroupContextSnapshot:
    version_snapshot: VersionContextSnapshot = field(
        default_factory=VersionContextSnapshot
    )
    generation: int = 0
    active_class_ids: Tuple[int, ...] = ()
    active_leaf_group_ids: Tuple[str, ...] = ()

    @property
    def known_class_count(self) -> int:
        return int(len(self.active_class_ids))

    @property
    def active_class_count(self) -> int:
        return int(len(self.active_class_ids))


@dataclass(frozen=True)
class TransitionGroupContextUpdateResult:
    snapshot: TransitionGroupContextSnapshot
    requires_relabel: bool
    affected_class_ids: Tuple[int, ...] = ()
    class_migrations: Tuple[GroupClassMigration, ...] = ()


class TransitionGroupClassifier:
    def __init__(
        self,
        evaluator: ProgramEvaluator,
    ) -> None:
        self._evaluator = evaluator
        self._snapshot = TransitionGroupContextSnapshot()
        self._root_group_id_by_commit_version: Dict[str, str] = {}
        self._group_by_id: Dict[str, Dict[str, Any]] = {}
        self._leaf_group_id_by_transition_key: Dict[str, str] = {}
        self._class_id_by_leaf_group_id: Dict[str, int] = {}
        self._leaf_group_id_by_class_id: Dict[int, str] = {}
        self._canonical_class_counts: Dict[int, int] = {}
        self._canonical_leaf_group_counts: Dict[str, int] = {}
        self._canonical_classified_count: int = 0
        self._canonical_unassigned_count: int = 0
        self._assignment_cache: Dict[Tuple[int, str], GroupAssignment] = {}
        self._class_rows_cache_signature: Optional[Tuple[Any, ...]] = None
        self._class_rows_cache_by_current_class: Dict[int, Tuple[Dict[str, Any], ...]] = {}

    @property
    def snapshot(self) -> TransitionGroupContextSnapshot:
        return self._snapshot

    @property
    def known_class_count(self) -> int:
        return int(self._snapshot.known_class_count)

    @property
    def active_class_count(self) -> int:
        return int(self._snapshot.active_class_count)

    def sync_context(
        self,
        *,
        context: Optional[Dict[str, Any]],
        max_program_count: Optional[int],
        previous_snapshot: Optional[TransitionGroupContextSnapshot] = None,
    ) -> TransitionGroupContextUpdateResult:
        previous = previous_snapshot or self._snapshot
        version_previous = previous.version_snapshot
        version_update = self._evaluator.sync_version_context(
            context=context,
            max_program_count=max_program_count,
            previous_snapshot=version_previous,
        )

        group_context = (
            context.get("group_context")
            if isinstance(context, dict) and isinstance(context.get("group_context"), dict)
            else None
        )
        if group_context is None:
            self._root_group_id_by_commit_version = {}
            self._group_by_id = {}
            self._leaf_group_id_by_transition_key = {}
            self._class_id_by_leaf_group_id = {}
            self._leaf_group_id_by_class_id = {}
            self._canonical_class_counts = {}
            self._canonical_leaf_group_counts = {}
            self._canonical_classified_count = 0
            self._canonical_unassigned_count = 0
            next_snapshot = TransitionGroupContextSnapshot(
                version_snapshot=version_update.snapshot,
                generation=0,
                active_class_ids=(),
                active_leaf_group_ids=(),
            )
            self._snapshot = next_snapshot
            self._assignment_cache.clear()
            self._class_rows_cache_signature = None
            self._class_rows_cache_by_current_class.clear()
            affected_class_ids = tuple(
                sorted(
                    int(class_id)
                    for class_id in previous.active_class_ids
                    if int(class_id) > 0
                )
            )
            return TransitionGroupContextUpdateResult(
                snapshot=next_snapshot,
                requires_relabel=bool(affected_class_ids),
                affected_class_ids=affected_class_ids,
            )

        group_rows = group_context.get("group_hierarchy") or []
        groups: Dict[str, Dict[str, Any]] = {}
        for row in group_rows:
            if not isinstance(row, dict):
                continue
            group_id = row.get("group_id")
            if not isinstance(group_id, str) or not group_id.strip():
                continue
            groups[group_id] = {
                "group_id": group_id,
                "commit_version": row.get("commit_version"),
                "parent_group_id": row.get("parent_group_id"),
                "child_group_ids": [
                    str(child_id)
                    for child_id in (row.get("child_group_ids") or [])
                    if isinstance(child_id, str) and child_id.strip()
                ],
                "member_transition_count": (
                    int(row.get("member_transition_count"))
                    if isinstance(row.get("member_transition_count"), int)
                    and int(row.get("member_transition_count")) >= 0
                    else 0
                ),
                "class_id": (
                    int(row.get("class_id"))
                    if isinstance(row.get("class_id"), int) and int(row.get("class_id")) > 0
                    else None
                ),
                "retired_class_id": (
                    int(row.get("retired_class_id"))
                    if isinstance(row.get("retired_class_id"), int)
                    and int(row.get("retired_class_id")) > 0
                    else None
                ),
                "split_broken_child_group_id": row.get("split_broken_child_group_id"),
                "split_kept_child_group_id": row.get("split_kept_child_group_id"),
                "split_program_source": row.get("split_program_source"),
            }

        root_group_id_by_commit_version = {
            str(key): str(value)
            for key, value in dict(group_context.get("root_group_id_by_commit_version") or {}).items()
            if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        }
        leaf_group_id_by_transition_key: Dict[str, str] = {}
        for key, value in dict(group_context.get("transition_leaf_group_ids") or {}).items():
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            group_id = value.strip()
            group = groups.get(group_id)
            if group is None:
                continue
            if list(group.get("child_group_ids") or []):
                continue
            leaf_group_id_by_transition_key[key.strip()] = group_id
        class_id_by_leaf_group_id: Dict[str, int] = {}
        for key, value in dict(group_context.get("leaf_group_class_ids") or {}).items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, int)
                or int(value) <= 0
            ):
                continue
            group = groups.get(str(key))
            if group is None:
                continue
            if list(group.get("child_group_ids") or []):
                continue
            if int(group.get("member_transition_count") or 0) <= 0:
                continue
            if not self._is_class_eligible_commit_version(group.get("commit_version")):
                continue
            class_id_by_leaf_group_id[str(key)] = int(value)
        leaf_group_id_by_class_id = {
            int(value): str(key)
            for key, value in class_id_by_leaf_group_id.items()
        }
        canonical_class_counts = {
            int(key): int(value)
            for key, value in dict(group_context.get("canonical_class_counts") or {}).items()
            if isinstance(key, (int, str))
            and str(key).strip()
            and isinstance(value, int)
            and int(value) >= 0
        }
        canonical_leaf_group_counts = {
            str(key): int(value)
            for key, value in dict(group_context.get("canonical_leaf_group_counts") or {}).items()
            if isinstance(key, str)
            and key.strip()
            and isinstance(value, int)
            and int(value) >= 0
        }
        canonical_classified_count = (
            int(group_context.get("canonical_classified_count"))
            if isinstance(group_context.get("canonical_classified_count"), int)
            else int(sum(canonical_class_counts.values()))
        )
        canonical_unassigned_count = (
            int(group_context.get("canonical_unassigned_count"))
            if isinstance(group_context.get("canonical_unassigned_count"), int)
            else 0
        )
        active_class_ids = tuple(sorted(leaf_group_id_by_class_id.keys()))
        active_leaf_group_ids = tuple(
            leaf_group_id_by_class_id[class_id] for class_id in active_class_ids
        )
        generation = (
            int(group_context.get("generation"))
            if isinstance(group_context.get("generation"), int)
            else 0
        )

        old_class_ids = set(previous.active_class_ids)
        class_migrations: List[GroupClassMigration] = []
        for group_id, group in groups.items():
            source_class_id = group.get("retired_class_id")
            kept_child_group_id = group.get("split_kept_child_group_id")
            broken_child_group_id = group.get("split_broken_child_group_id")
            if (
                not isinstance(source_class_id, int)
                or source_class_id <= 0
                or not isinstance(kept_child_group_id, str)
                or not kept_child_group_id
                or not isinstance(broken_child_group_id, str)
                or not broken_child_group_id
            ):
                continue
            kept_class_id = class_id_by_leaf_group_id.get(kept_child_group_id)
            broken_class_id = class_id_by_leaf_group_id.get(broken_child_group_id)
            if (
                not isinstance(kept_class_id, int)
                or kept_class_id <= 0
                or not isinstance(broken_class_id, int)
                or broken_class_id <= 0
                or kept_class_id != source_class_id
                or broken_class_id in old_class_ids
                or source_class_id not in old_class_ids
            ):
                continue
            class_migrations.append(
                GroupClassMigration(
                    parent_group_id=group_id,
                    source_class_id=int(source_class_id),
                    kept_child_group_id=kept_child_group_id,
                    kept_class_id=int(kept_class_id),
                    broken_child_group_id=broken_child_group_id,
                    broken_class_id=int(broken_class_id),
                )
            )

        next_snapshot = TransitionGroupContextSnapshot(
            version_snapshot=version_update.snapshot,
            generation=int(generation),
            active_class_ids=active_class_ids,
            active_leaf_group_ids=active_leaf_group_ids,
        )
        self._root_group_id_by_commit_version = root_group_id_by_commit_version
        self._group_by_id = groups
        previous_leaf_group_id_by_transition_key = self._leaf_group_id_by_transition_key
        self._leaf_group_id_by_transition_key = leaf_group_id_by_transition_key
        self._class_id_by_leaf_group_id = class_id_by_leaf_group_id
        self._leaf_group_id_by_class_id = leaf_group_id_by_class_id
        self._canonical_class_counts = canonical_class_counts
        self._canonical_leaf_group_counts = canonical_leaf_group_counts
        self._canonical_classified_count = max(0, int(canonical_classified_count))
        self._canonical_unassigned_count = max(0, int(canonical_unassigned_count))

        previous_class_ids = set(previous.active_class_ids)
        next_class_ids = set(next_snapshot.active_class_ids)
        previous_leaf_by_class = {
            int(class_id): str(leaf_group_id)
            for class_id, leaf_group_id in zip(
                previous.active_class_ids,
                previous.active_leaf_group_ids,
            )
            if int(class_id) > 0
            and isinstance(leaf_group_id, str)
            and leaf_group_id
        }
        next_leaf_by_class = {
            int(class_id): str(leaf_group_id)
            for class_id, leaf_group_id in zip(
                next_snapshot.active_class_ids,
                next_snapshot.active_leaf_group_ids,
            )
            if int(class_id) > 0
            and isinstance(leaf_group_id, str)
            and leaf_group_id
        }
        removed_class_ids = previous_class_ids - next_class_ids
        changed_existing_class_ids = {
            int(class_id)
            for class_id in previous_class_ids & next_class_ids
            if previous_leaf_by_class.get(int(class_id))
            != next_leaf_by_class.get(int(class_id))
        }
        version_ids_appended = bool(
            version_update.version_ids_changed
            and len(next_snapshot.version_snapshot.version_ids)
            >= len(previous.version_snapshot.version_ids)
            and tuple(
                next_snapshot.version_snapshot.version_ids[
                    : len(previous.version_snapshot.version_ids)
                ]
            )
            == tuple(previous.version_snapshot.version_ids)
        )
        non_append_version_change = bool(
            version_update.version_ids_changed and not version_ids_appended
        )
        context_changed_class_ids = (
            previous_class_ids | next_class_ids
            if version_update.source_changed or non_append_version_change
            else set()
        )
        affected_class_ids = tuple(
            sorted(
                {
                    migration.source_class_id
                    for migration in class_migrations
                }
                | {int(class_id) for class_id in removed_class_ids}
                | {int(class_id) for class_id in changed_existing_class_ids}
                | {int(class_id) for class_id in context_changed_class_ids}
            )
        )
        requires_relabel = bool(affected_class_ids)

        class_rows_signature = self._build_class_rows_cache_signature(
            snapshot=next_snapshot,
            canonical_class_counts=canonical_class_counts,
        )
        if class_rows_signature != self._class_rows_cache_signature:
            self._class_rows_cache_signature = class_rows_signature
            self._class_rows_cache_by_current_class.clear()
        if (
            previous.generation != next_snapshot.generation
            or previous.version_snapshot != next_snapshot.version_snapshot
            or previous.active_class_ids != next_snapshot.active_class_ids
            or previous_leaf_group_id_by_transition_key != leaf_group_id_by_transition_key
        ):
            self._assignment_cache.clear()
        self._snapshot = next_snapshot
        return TransitionGroupContextUpdateResult(
            snapshot=next_snapshot,
            requires_relabel=requires_relabel,
            affected_class_ids=affected_class_ids,
            class_migrations=tuple(class_migrations),
        )

    def classify_transition(
        self,
        *,
        transition: Transition,
        include_rows: bool = True,
        known_explaining_version_id: Optional[str] = None,
    ) -> Tuple[GroupAssignment, List[Dict[str, Any]]]:
        return self.classify_transitions(
            transitions=[transition],
            include_rows=include_rows,
            known_explaining_version_id=known_explaining_version_id,
        )[0]

    def classify_transitions(
        self,
        *,
        transitions: Sequence[Transition],
        include_rows: bool = True,
        known_explaining_version_id: Optional[str] = None,
    ) -> List[Tuple[GroupAssignment, List[Dict[str, Any]]]]:
        safe_transitions = list(transitions)
        if not safe_transitions:
            return []
        if not self._snapshot.active_class_ids:
            unknown_rows = self._resolve_class_rows_for_assignment(
                class_id=None,
                include_rows=include_rows,
            )
            return [
                (
                    GroupAssignment(
                        class_id=0,
                        group_id=None,
                        status="unknown",
                    ),
                    [dict(row) for row in unknown_rows],
                )
                for _transition in safe_transitions
            ]

        resolved_assignments: List[Optional[GroupAssignment]] = [
            None
            for _transition in safe_transitions
        ]
        unresolved_indices: List[int] = []
        for index, transition in enumerate(safe_transitions):
            transition_key = self._transition_key(transition)
            cache_key = (int(self._snapshot.generation), transition_key)
            cached = self._assignment_cache.get(cache_key)
            if cached is not None:
                resolved_assignments[index] = cached
                continue
            direct_leaf_group_id = self._leaf_group_id_by_transition_key.get(transition_key)
            if isinstance(direct_leaf_group_id, str) and direct_leaf_group_id:
                assignment = self._assignment_for_leaf_group(direct_leaf_group_id)
                self._assignment_cache[cache_key] = assignment
                resolved_assignments[index] = assignment
                continue
            unresolved_indices.append(index)

        if unresolved_indices:
            leaf_group_ids = self._resolve_leaf_groups_for_unresolved_transitions(
                transitions=safe_transitions,
                transition_indices=unresolved_indices,
                known_explaining_version_id=known_explaining_version_id,
            )
            for index in unresolved_indices:
                group_id = leaf_group_ids.get(index)
                assignment = self._assignment_for_leaf_group(group_id)
                cache_key = (
                    int(self._snapshot.generation),
                    self._transition_key(safe_transitions[index]),
                )
                self._assignment_cache[cache_key] = assignment
                resolved_assignments[index] = assignment

        resolved: List[Tuple[GroupAssignment, List[Dict[str, Any]]]] = []
        for index in range(len(safe_transitions)):
            assignment = resolved_assignments[index] or GroupAssignment(
                class_id=0,
                group_id=None,
                status="unknown",
            )
            resolved.append(
                (
                    assignment,
                    self._resolve_class_rows_for_assignment(
                        class_id=(
                            int(assignment.class_id)
                            if int(assignment.class_id) > 0
                            else None
                        ),
                        include_rows=include_rows,
                    ),
                )
            )
        return resolved

    def _assignment_for_leaf_group(self, group_id: Optional[str]) -> GroupAssignment:
        resolved_group_id = (
            str(group_id).strip()
            if isinstance(group_id, str) and str(group_id).strip()
            else None
        )
        class_id = int(self._class_id_by_leaf_group_id.get(resolved_group_id, 0) or 0)
        return GroupAssignment(
            class_id=class_id,
            group_id=resolved_group_id,
            status="assigned" if class_id > 0 else "unknown",
        )

    def _transition_key(self, transition: Transition) -> str:
        return canonical_graph_edge_identity_key(transition)

    def _rooted_version_ids(self) -> List[str]:
        return [
            str(version_id)
            for version_id in self._snapshot.version_snapshot.version_ids
            if isinstance(version_id, str)
            and version_id.strip()
            and isinstance(
                self._root_group_id_by_commit_version.get(str(version_id).strip()),
                str,
            )
            and str(
                self._root_group_id_by_commit_version.get(str(version_id).strip())
            ).strip()
            and str(self._root_group_id_by_commit_version.get(str(version_id).strip()))
            in self._group_by_id
        ]

    def _fallback_root_group_id(self) -> Optional[str]:
        unique_root_group_ids = sorted(
            {
                str(group_id)
                for group_id in self._root_group_id_by_commit_version.values()
                if isinstance(group_id, str)
                and group_id.strip()
                and group_id in self._group_by_id
            }
        )
        if unique_root_group_ids:
            return unique_root_group_ids[0]
        return None

    def _selected_root_group_id(
        self,
        *,
        rooted_version_ids: Sequence[str],
        stable_explaining_version_id: Optional[str],
    ) -> Optional[str]:
        selected_version_id: Optional[str] = None
        if (
            isinstance(stable_explaining_version_id, str)
            and stable_explaining_version_id.strip()
        ):
            selected_version_id = stable_explaining_version_id.strip()
        else:
            current_version_id = self._snapshot.version_snapshot.current_version_id
            if (
                isinstance(current_version_id, str)
                and current_version_id.strip()
                and current_version_id.strip() in rooted_version_ids
            ):
                selected_version_id = current_version_id.strip()
            elif rooted_version_ids:
                selected_version_id = rooted_version_ids[-1]

        root_group_id = self._root_group_id_by_commit_version.get(selected_version_id)
        if isinstance(root_group_id, str) and root_group_id.strip():
            return root_group_id.strip()
        return None

    def _split_group_has_router(self, group_id: str) -> bool:
        group = self._group_by_id.get(str(group_id))
        if group is None:
            return False
        broken_child_group_id = group.get("split_broken_child_group_id")
        kept_child_group_id = group.get("split_kept_child_group_id")
        split_program_source = group.get("split_program_source")
        return (
            isinstance(broken_child_group_id, str)
            and bool(broken_child_group_id)
            and isinstance(kept_child_group_id, str)
            and bool(kept_child_group_id)
            and isinstance(split_program_source, str)
            and bool(split_program_source.strip())
        )

    def _split_router_group_ids_under_root(self, root_group_id: str) -> List[str]:
        root = str(root_group_id).strip()
        if not root or root not in self._group_by_id:
            return []

        router_group_ids: List[str] = []
        visited: set[str] = set()
        stack = [root]
        while stack:
            group_id = stack.pop()
            if group_id in visited:
                continue
            visited.add(group_id)
            group = self._group_by_id.get(group_id)
            if group is None:
                continue
            if self._split_group_has_router(group_id):
                router_group_ids.append(group_id)
            child_group_ids = [
                str(child_id)
                for child_id in (group.get("child_group_ids") or [])
                if isinstance(child_id, str) and child_id.strip()
            ]
            stack.extend(reversed(child_group_ids))
        return sorted(router_group_ids)

    def _program_eval_task_for_registered_version(
        self,
        version_id: str,
    ) -> Optional[ProgramEvaluationTask]:
        source = self._evaluator._get_registered_source(version_id)
        if source is None:
            return None
        return ProgramEvaluationTask(label=str(version_id), source=str(source))

    def _program_eval_task_for_split_group(
        self,
        group_id: str,
    ) -> Optional[ProgramEvaluationTask]:
        group = self._group_by_id.get(str(group_id))
        if group is None:
            return None
        source = group.get("split_program_source")
        if not isinstance(source, str) or not source.strip():
            return None
        return ProgramEvaluationTask(label=str(group_id), source=str(source))

    def _batch_explains(
        self,
        batch: Any,
        *,
        label: str,
        local_index: int,
    ) -> bool:
        result = batch.by_label.get(str(label))
        explains = result.explains if result is not None else ()
        if explains is None:
            explains = ()
        return bool(explains[local_index]) if local_index < len(explains) else False

    def _resolve_leaf_groups_for_unresolved_transitions(
        self,
        *,
        transitions: Sequence[Transition],
        transition_indices: Sequence[int],
        known_explaining_version_id: Optional[str] = None,
    ) -> Dict[int, Optional[str]]:
        active_indices = [
            int(index)
            for index in transition_indices
            if 0 <= int(index) < len(transitions)
        ]
        if not active_indices:
            return {}

        active_transitions = [transitions[index] for index in active_indices]
        rooted_version_ids = self._rooted_version_ids()
        root_group_id_by_index = self._resolve_root_group_ids_for_active_transitions(
            active_indices=active_indices,
            active_transitions=active_transitions,
            rooted_version_ids=rooted_version_ids,
            known_explaining_version_id=known_explaining_version_id,
        )
        local_position_by_index = {
            int(index): local_index
            for local_index, index in enumerate(active_indices)
        }

        split_tasks_by_group_id: Dict[str, ProgramEvaluationTask] = {}
        split_transition_indices: Dict[str, List[int]] = {}
        for root_group_id in sorted(
            {
                str(group_id)
                for group_id in root_group_id_by_index.values()
                if isinstance(group_id, str) and group_id.strip()
            }
        ):
            local_indices = [
                local_position_by_index[index]
                for index in active_indices
                if root_group_id_by_index.get(index) == root_group_id
            ]
            if not local_indices:
                continue
            for group_id in self._split_router_group_ids_under_root(root_group_id):
                task = self._program_eval_task_for_split_group(group_id)
                if task is None:
                    continue
                split_tasks_by_group_id[group_id] = task
                split_transition_indices.setdefault(group_id, []).extend(local_indices)

        split_batch = None
        if split_tasks_by_group_id:
            split_batch = self._evaluator.evaluate_program_explains(
                programs=[
                    split_tasks_by_group_id[group_id]
                    for group_id in sorted(split_tasks_by_group_id)
                ],
                transitions=active_transitions,
                transition_chunk_size=CLASS_ASSIGN_TRANSITION_CHUNK_SIZE,
                program_transition_indices=split_transition_indices,
            )

        resolved: Dict[int, Optional[str]] = {}
        for index in active_indices:
            local_index = local_position_by_index[index]
            root_group_id = root_group_id_by_index.get(index)
            current_group_id = root_group_id
            visited_group_ids: set[str] = set()

            while isinstance(current_group_id, str) and current_group_id:
                if current_group_id in visited_group_ids:
                    current_group_id = root_group_id
                    break
                visited_group_ids.add(current_group_id)

                group = self._group_by_id.get(current_group_id)
                if group is None:
                    current_group_id = root_group_id
                    break
                child_group_ids = list(group.get("child_group_ids") or [])
                if not child_group_ids:
                    break

                broken_child_group_id = group.get("split_broken_child_group_id")
                kept_child_group_id = group.get("split_kept_child_group_id")
                split_program_source = group.get("split_program_source")
                if (
                    isinstance(broken_child_group_id, str)
                    and broken_child_group_id
                    and isinstance(kept_child_group_id, str)
                    and kept_child_group_id
                    and isinstance(split_program_source, str)
                    and split_program_source.strip()
                ):
                    solved = (
                        self._batch_explains(
                            split_batch,
                            label=current_group_id,
                            local_index=local_index,
                        )
                        if split_batch is not None
                        else False
                    )
                    current_group_id = (
                        kept_child_group_id if solved else broken_child_group_id
                    )
                    continue

                fallback = self._find_first_leaf_descendant(current_group_id)
                current_group_id = fallback or root_group_id
                break

            resolved[index] = current_group_id
        return resolved

    def _resolve_root_group_ids_for_active_transitions(
        self,
        *,
        active_indices: Sequence[int],
        active_transitions: Sequence[Transition],
        rooted_version_ids: Sequence[str],
        known_explaining_version_id: Optional[str] = None,
    ) -> Dict[int, Optional[str]]:
        if not active_indices:
            return {}

        version_ids = tuple(str(version_id) for version_id in rooted_version_ids)
        known_version_id = (
            str(known_explaining_version_id).strip()
            if isinstance(known_explaining_version_id, str)
            and str(known_explaining_version_id).strip() in version_ids
            else None
        )
        reversed_version_ids = tuple(reversed(version_ids))
        active_local_indices = set(range(len(active_indices)))
        stable_version_by_local_index: Dict[int, str] = {}
        root_group_id_by_index: Dict[int, Optional[str]] = {}

        for offset in range(
            0,
            len(reversed_version_ids),
            CLASS_ASSIGN_ROOT_PROGRAM_BLOCK_SIZE,
        ):
            if not active_local_indices:
                break
            block_version_ids = reversed_version_ids[
                offset : offset + CLASS_ASSIGN_ROOT_PROGRAM_BLOCK_SIZE
            ]
            selected_local_indices = tuple(sorted(active_local_indices))
            tasks: List[ProgramEvaluationTask] = []
            for version_id in block_version_ids:
                if known_version_id is not None and version_id == known_version_id:
                    continue
                task = self._program_eval_task_for_registered_version(version_id)
                if task is not None:
                    tasks.append(task)

            root_batch = None
            if tasks and selected_local_indices:
                root_batch = self._evaluator.evaluate_program_explains(
                    programs=tasks,
                    transitions=active_transitions,
                    transition_chunk_size=CLASS_ASSIGN_TRANSITION_CHUNK_SIZE,
                    program_transition_indices={
                        task.label: selected_local_indices
                        for task in tasks
                    },
                )

            finished_local_indices: List[int] = []
            for local_index in selected_local_indices:
                for version_id in block_version_ids:
                    if known_version_id is not None and version_id == known_version_id:
                        solved = True
                    else:
                        solved = (
                            self._batch_explains(
                                root_batch,
                                label=version_id,
                                local_index=local_index,
                            )
                            if root_batch is not None
                            else False
                        )
                    if solved:
                        stable_version_by_local_index[local_index] = version_id
                        continue
                    if local_index not in stable_version_by_local_index:
                        continue
                    index = int(active_indices[local_index])
                    root_group_id_by_index[index] = self._root_group_id_for_stable_version(
                        rooted_version_ids=version_ids,
                        stable_version_id=stable_version_by_local_index.get(
                            local_index
                        ),
                    )
                    finished_local_indices.append(local_index)
                    break
            for local_index in finished_local_indices:
                active_local_indices.discard(local_index)

        for local_index in sorted(active_local_indices):
            index = int(active_indices[local_index])
            root_group_id_by_index[index] = self._root_group_id_for_stable_version(
                rooted_version_ids=version_ids,
                stable_version_id=stable_version_by_local_index.get(local_index),
            )

        return root_group_id_by_index

    def _root_group_id_for_stable_version(
        self,
        *,
        rooted_version_ids: Sequence[str],
        stable_version_id: Optional[str],
    ) -> Optional[str]:
        root_group_id = self._selected_root_group_id(
            rooted_version_ids=rooted_version_ids,
            stable_explaining_version_id=stable_version_id,
        )
        if root_group_id is None:
            root_group_id = self._fallback_root_group_id()
        return root_group_id

    def _resolve_class_rows_for_assignment(
        self,
        *,
        class_id: Optional[int],
        include_rows: bool,
    ) -> List[Dict[str, Any]]:
        if not include_rows:
            return []
        return self.build_class_rows(
            current_class_id=(
                int(class_id)
                if isinstance(class_id, int) and int(class_id) > 0
                else None
            ),
        )

    def build_class_rows(
        self,
        *,
        current_class_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        safe_current_class_id = (
            int(current_class_id)
            if isinstance(current_class_id, int) and int(current_class_id) > 0
            else 0
        )
        cached_rows = self._class_rows_cache_by_current_class.get(safe_current_class_id)
        if cached_rows is not None:
            return [dict(row) for row in cached_rows]

        rows: List[Dict[str, Any]] = []
        current_version_id = (
            str(self._snapshot.version_snapshot.current_version_id).strip()
            if isinstance(self._snapshot.version_snapshot.current_version_id, str)
            and str(self._snapshot.version_snapshot.current_version_id).strip()
            else None
        )
        for group_id in self._ordered_active_leaf_group_ids():
            class_id = self._class_id_by_leaf_group_id.get(str(group_id))
            if not isinstance(class_id, int) or int(class_id) <= 0:
                continue
            if not isinstance(group_id, str) or not group_id:
                continue
            group = self._group_by_id.get(str(group_id)) or {}
            commit_version = (
                str(group.get("commit_version")).strip()
                if isinstance(group.get("commit_version"), str)
                and str(group.get("commit_version")).strip()
                else None
            )
            rows.append(
                {
                    "version_id": group_id,
                    "version_index": int(class_id),
                    "group_id": str(group_id),
                    "commit_version": commit_version,
                    "display_id": self._build_group_display_id(str(group_id)),
                    "explains_current_state": (
                        True if safe_current_class_id > 0 and int(class_id) == safe_current_class_id else None
                    ),
                    "prediction_status": "success",
                    "prediction_error": None,
                    "is_active": True,
                    "is_current": (
                        bool(
                            current_version_id is not None
                            and commit_version is not None
                            and commit_version == current_version_id
                        )
                        if current_version_id is not None
                        else None
                    ),
                    "is_explainer": False,
                    "transition_count": int(
                        self._canonical_class_counts.get(int(class_id), 0)
                    ),
                }
            )
        self._class_rows_cache_by_current_class[safe_current_class_id] = tuple(
            dict(row) for row in rows
        )
        return [dict(row) for row in rows]

    def resolve_group_display_metadata(
        self,
        *,
        class_id: Optional[int],
        group_id: Optional[str],
    ) -> Dict[str, Any]:
        resolved_class_id = (
            int(class_id)
            if isinstance(class_id, int) and int(class_id) > 0
            else 0
        )
        resolved_group_id = (
            str(group_id).strip()
            if isinstance(group_id, str) and str(group_id).strip()
            else None
        )

        matched_group_id: Optional[str] = None
        if isinstance(resolved_group_id, str) and resolved_group_id in self._group_by_id:
            matched_group_id = resolved_group_id
        elif resolved_class_id > 0:
            candidate_group_id = self._leaf_group_id_by_class_id.get(resolved_class_id)
            if isinstance(candidate_group_id, str) and candidate_group_id.strip():
                matched_group_id = candidate_group_id.strip()
                resolved_group_id = matched_group_id

        group_label: Optional[str] = None
        class_count: Optional[int] = None
        if isinstance(matched_group_id, str) and matched_group_id:
            display_id = self._build_group_display_id(matched_group_id)
            if isinstance(display_id, str) and display_id.strip():
                group_label = display_id.strip()
            matched_class_id = self._class_id_by_leaf_group_id.get(matched_group_id)
            if isinstance(matched_class_id, int) and matched_class_id > 0:
                class_count = int(self._canonical_class_counts.get(int(matched_class_id), 0))

        return {
            "class_id": int(resolved_class_id),
            "group_id": resolved_group_id,
            "group_label": group_label,
            "class_count": class_count,
            "is_new_dynamics_class": bool(resolved_class_id <= 0),
        }

    def _build_class_rows_cache_signature(
        self,
        *,
        snapshot: TransitionGroupContextSnapshot,
        canonical_class_counts: Dict[int, int],
    ) -> Tuple[Any, ...]:
        version_snapshot = snapshot.version_snapshot
        return (
            int(snapshot.generation),
            tuple(int(class_id) for class_id in snapshot.active_class_ids),
            tuple(str(group_id) for group_id in snapshot.active_leaf_group_ids),
            tuple(
                str(version_id)
                for version_id in version_snapshot.version_ids
                if isinstance(version_id, str) and version_id.strip()
            ),
            (
                str(version_snapshot.current_version_id).strip()
                if isinstance(version_snapshot.current_version_id, str)
                and str(version_snapshot.current_version_id).strip()
                else None
            ),
            tuple(
                sorted(
                    (int(class_id), int(count))
                    for class_id, count in canonical_class_counts.items()
                    if int(class_id) > 0 and int(count) >= 0
                )
            ),
        )

    @property
    def canonical_classified_count(self) -> int:
        return int(self._canonical_classified_count)

    @property
    def canonical_unassigned_count(self) -> int:
        return int(self._canonical_unassigned_count)

    @property
    def canonical_class_count(self) -> int:
        return int(
            len(
                [
                    class_id
                    for class_id, count in self._canonical_class_counts.items()
                    if int(class_id) > 0 and int(count) > 0
                ]
            )
        )

    def canonical_class_counts(self) -> Dict[int, int]:
        return {
            int(class_id): int(count)
            for class_id, count in self._canonical_class_counts.items()
            if int(class_id) > 0 and int(count) >= 0
        }

    def update_canonical_assignment_stats(
        self,
        *,
        canonical_class_counts: Optional[Mapping[Any, Any]] = None,
        canonical_leaf_group_counts: Optional[Mapping[Any, Any]] = None,
        canonical_classified_count: Optional[int] = None,
        canonical_unassigned_count: Optional[int] = None,
    ) -> None:
        if canonical_class_counts is not None:
            self._canonical_class_counts = {
                int(class_id): int(count)
                for class_id, count in dict(canonical_class_counts).items()
                if isinstance(class_id, (int, str))
                and str(class_id).strip()
                and isinstance(count, int)
                and int(count) >= 0
            }
        if canonical_leaf_group_counts is not None:
            self._canonical_leaf_group_counts = {
                str(group_id): int(count)
                for group_id, count in dict(canonical_leaf_group_counts).items()
                if isinstance(group_id, str)
                and group_id.strip()
                and isinstance(count, int)
                and int(count) >= 0
            }
        if isinstance(canonical_classified_count, int):
            self._canonical_classified_count = max(
                0,
                int(canonical_classified_count),
            )
        elif canonical_class_counts is not None:
            self._canonical_classified_count = max(
                0,
                int(sum(self._canonical_class_counts.values())),
            )
        if isinstance(canonical_unassigned_count, int):
            self._canonical_unassigned_count = max(
                0,
                int(canonical_unassigned_count),
            )
        self._class_rows_cache_signature = None
        self._class_rows_cache_by_current_class.clear()

    def _find_first_leaf_descendant(self, group_id: str) -> Optional[str]:
        current_group_id = group_id
        visited_group_ids = set()
        while (
            isinstance(current_group_id, str)
            and current_group_id
            and current_group_id not in visited_group_ids
        ):
            visited_group_ids.add(current_group_id)
            group = self._group_by_id.get(current_group_id)
            if group is None:
                return None
            child_group_ids = sorted(
                child_id
                for child_id in (group.get("child_group_ids") or [])
                if isinstance(child_id, str) and child_id
            )
            if not child_group_ids:
                return current_group_id
            current_group_id = child_group_ids[0]
        return None

    def _ordered_active_leaf_group_ids(self) -> List[str]:
        ordered_leaf_group_ids: List[str] = []
        seen_group_ids = set()
        root_group_ids = sorted(
            {
                str(group_id)
                for group_id in self._root_group_id_by_commit_version.values()
                if isinstance(group_id, str)
                and group_id.strip()
                and group_id in self._group_by_id
            }
        )
        for root_group_id in root_group_ids:
            self._append_leaf_group_ids_depth_first(
                group_id=root_group_id,
                ordered_leaf_group_ids=ordered_leaf_group_ids,
                seen_group_ids=seen_group_ids,
            )
        for class_id in self._snapshot.active_class_ids:
            fallback_group_id = self._leaf_group_id_by_class_id.get(int(class_id))
            if (
                isinstance(fallback_group_id, str)
                and fallback_group_id
                and fallback_group_id not in seen_group_ids
            ):
                ordered_leaf_group_ids.append(fallback_group_id)
                seen_group_ids.add(fallback_group_id)
        return ordered_leaf_group_ids

    def _append_leaf_group_ids_depth_first(
        self,
        *,
        group_id: str,
        ordered_leaf_group_ids: List[str],
        seen_group_ids: set[str],
    ) -> None:
        safe_group_id = str(group_id).strip()
        if not safe_group_id or safe_group_id in seen_group_ids:
            return
        group = self._group_by_id.get(safe_group_id)
        if group is None:
            return
        child_group_ids = [
            str(child_group_id)
            for child_group_id in (group.get("child_group_ids") or [])
            if isinstance(child_group_id, str) and child_group_id.strip()
        ]
        if not child_group_ids:
            if safe_group_id in self._class_id_by_leaf_group_id:
                ordered_leaf_group_ids.append(safe_group_id)
                seen_group_ids.add(safe_group_id)
            return

        kept_child_group_id = group.get("split_kept_child_group_id")
        broken_child_group_id = group.get("split_broken_child_group_id")
        traversal_order: List[str] = []
        for candidate_group_id in (kept_child_group_id, broken_child_group_id):
            if (
                isinstance(candidate_group_id, str)
                and candidate_group_id.strip()
                and candidate_group_id in child_group_ids
                and candidate_group_id not in traversal_order
            ):
                traversal_order.append(candidate_group_id)
        for child_group_id in sorted(child_group_ids):
            if child_group_id not in traversal_order:
                traversal_order.append(child_group_id)
        for child_group_id in traversal_order:
            self._append_leaf_group_ids_depth_first(
                group_id=child_group_id,
                ordered_leaf_group_ids=ordered_leaf_group_ids,
                seen_group_ids=seen_group_ids,
            )

    def _build_group_display_id(self, group_id: str) -> str:
        safe_group_id = str(group_id).strip()
        if not safe_group_id:
            return ""
        group = self._group_by_id.get(safe_group_id) or {}
        commit_version = (
            str(group.get("commit_version")).strip()
            if isinstance(group.get("commit_version"), str) and str(group.get("commit_version")).strip()
            else ""
        )
        if not commit_version:
            return safe_group_id
        parent_group_id = group.get("parent_group_id")
        if not isinstance(parent_group_id, str) or not parent_group_id.strip():
            return commit_version
        if safe_group_id.startswith(f"{commit_version}:g"):
            suffix = safe_group_id.split(":g", 1)[1].strip()
            if suffix:
                return f"{commit_version}/g{suffix[-3:]}"
        return safe_group_id

    def _is_class_eligible_commit_version(self, version_id: Any) -> bool:
        text = str(version_id).strip() if version_id is not None else ""
        if not text:
            return False
        if text.lower().startswith("v"):
            text = text[1:]
        if text.isdigit():
            return int(text) > 0
        return True
