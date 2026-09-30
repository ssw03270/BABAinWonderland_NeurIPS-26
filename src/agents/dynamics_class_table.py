"""Shared dynamics-class table decoration for Web GUI explorers."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.data import Transition
from src.program_model import (
    PredictionRecord,
    ProgramEvaluator,
    ProgramEvaluationTask,
    TransitionGroupClassifier,
    TransitionGroupContextSnapshot,
)


class DynamicsClassTableSupport:
    """Keep Web GUI class-table behavior consistent across explorers."""

    def __init__(
        self,
        *,
        evaluator: ProgramEvaluator,
        group_classifier: TransitionGroupClassifier,
    ) -> None:
        self._program_evaluator = evaluator
        self._group_classifier = group_classifier
        self._group_context_snapshot = TransitionGroupContextSnapshot()
        self._current_program_source: Optional[str] = None
        self.reset_collect_state()

    @property
    def snapshot(self) -> TransitionGroupContextSnapshot:
        return self._group_context_snapshot

    @property
    def known_class_count(self) -> int:
        return int(self._group_context_snapshot.known_class_count)

    @property
    def active_class_count(self) -> int:
        return int(self._group_context_snapshot.active_class_count)

    @property
    def current_collect_question_mark_count(self) -> int:
        return int(self._current_collect_question_mark_count)

    def reset_collect_state(self) -> None:
        self._current_collect_question_mark_count = 0
        self._has_seen_question_mark_transition = False
        self._current_collect_class_id: Optional[int] = None
        self._current_collect_group_id: Optional[str] = None
        self._current_collect_explains: Optional[bool] = None

    def set_program_context(
        self,
        context: Optional[Dict[str, Any]],
        *,
        max_program_count: Optional[int] = None,
    ) -> Any:
        update_result = self._group_classifier.sync_context(
            context=context,
            max_program_count=max_program_count,
            previous_snapshot=self._group_context_snapshot,
        )
        self._group_context_snapshot = update_result.snapshot
        self._current_program_source = self._resolve_current_program_source(context)
        return update_result

    def update_canonical_assignment_stats(self, summary: Dict[str, Any]) -> None:
        self._group_classifier.update_canonical_assignment_stats(
            canonical_class_counts=summary.get("canonical_class_counts"),
            canonical_leaf_group_counts=summary.get("canonical_leaf_group_counts"),
            canonical_classified_count=summary.get("canonical_classified_count"),
            canonical_unassigned_count=summary.get("canonical_unassigned_count"),
        )

    def current_version_id(self) -> Optional[str]:
        current_version_id = self._group_context_snapshot.version_snapshot.current_version_id
        if isinstance(current_version_id, str) and current_version_id.strip():
            return current_version_id.strip()
        return None

    def current_source_digest(self) -> Optional[str]:
        if not isinstance(self._current_program_source, str) or not self._current_program_source:
            return None
        return hashlib.sha1(self._current_program_source.encode("utf-8")).hexdigest()

    def current_program_transition_record(
        self,
        *,
        transition: Transition,
    ) -> Optional[PredictionRecord]:
        if (
            not isinstance(self._current_program_source, str)
            or not self._current_program_source.strip()
        ):
            return None
        return self._program_evaluator.evaluate_transition(
            source=self._current_program_source,
            transition=transition,
        )

    def current_program_transition_records(
        self,
        *,
        transitions: Sequence[Transition],
    ) -> List[Optional[PredictionRecord]]:
        safe_transitions = list(transitions or [])
        if not safe_transitions:
            return []
        if (
            not isinstance(self._current_program_source, str)
            or not self._current_program_source.strip()
        ):
            return [None for _transition in safe_transitions]
        batch = self._program_evaluator.evaluate_programs(
            programs=[
                ProgramEvaluationTask(
                    label="current",
                    source=self._current_program_source,
                )
            ],
            transitions=safe_transitions,
            collect_records=True,
        )
        records_by_index = {
            int(record.index): record
            for record in batch.first().evaluation.records
            if isinstance(record, PredictionRecord)
        }
        return [
            records_by_index.get(int(index))
            for index in range(len(safe_transitions))
        ]

    def current_program_explains_transition(
        self,
        *,
        transition: Transition,
    ) -> Optional[bool]:
        record = self.current_program_transition_record(transition=transition)
        if record is None:
            return None
        return bool(record.is_correct) and record.error is None

    def classify_transition_with_current_record(
        self,
        *,
        transition: Transition,
        include_rows: bool = True,
    ) -> Tuple[
        Optional[int],
        Any,
        List[Dict[str, Any]],
        Optional[bool],
        Optional[PredictionRecord],
    ]:
        assignment, rows = self._group_classifier.classify_transition(
            transition=transition,
            include_rows=include_rows,
        )
        class_id = (
            int(assignment.class_id)
            if isinstance(getattr(assignment, "class_id", None), int)
            and int(assignment.class_id) > 0
            else None
        )
        current_record = self.current_program_transition_record(
            transition=transition,
        )
        current_explains = (
            (bool(current_record.is_correct) and current_record.error is None)
            if current_record is not None
            else None
        )
        metadata = self.resolve_transition_display_metadata(
            class_id=class_id,
            assignment=assignment,
        )
        current_is_question_mark = self._should_render_current_transition_as_question_mark(
            current_explains=current_explains,
            metadata=metadata,
        )
        self._remember_current_collect_focus(
            class_id=class_id,
            assignment=assignment,
            current_explains=current_explains,
            current_is_question_mark=current_is_question_mark,
        )
        if include_rows:
            rows = self.decorate_current_transition_rows(
                rows=rows,
                class_id=class_id,
                assignment=assignment,
                current_explains=current_explains,
                metadata=metadata,
            )
        else:
            rows = []
        return class_id, assignment, rows, current_explains, current_record

    def classify_transitions_with_current_records(
        self,
        *,
        transitions: Sequence[Transition],
        current_records: Sequence[Optional[PredictionRecord]],
        include_rows: bool = True,
    ) -> List[
        Tuple[
            Optional[int],
            Any,
            List[Dict[str, Any]],
            Optional[bool],
            Optional[PredictionRecord],
        ]
    ]:
        safe_transitions = list(transitions or [])
        if not safe_transitions:
            return []
        records = list(current_records or [])
        if len(records) < len(safe_transitions):
            records.extend([None] * (len(safe_transitions) - len(records)))
        elif len(records) > len(safe_transitions):
            records = records[: len(safe_transitions)]

        current_explains_by_index = [
            (
                bool(record.is_correct) and record.error is None
                if isinstance(record, PredictionRecord)
                else None
            )
            for record in records
        ]
        known_indices = [
            index
            for index, current_explains in enumerate(current_explains_by_index)
            if current_explains is True
        ]
        classified_by_index: Dict[int, Tuple[Any, List[Dict[str, Any]]]] = {}
        if known_indices:
            classified = self._group_classifier.classify_transitions(
                transitions=[safe_transitions[index] for index in known_indices],
                include_rows=include_rows,
                known_explaining_version_id=(
                    self._group_classifier.snapshot.version_snapshot.current_version_id
                ),
            )
            for local_index, result in enumerate(classified):
                classified_by_index[int(known_indices[int(local_index)])] = result

        results: List[
            Tuple[
                Optional[int],
                Any,
                List[Dict[str, Any]],
                Optional[bool],
                Optional[PredictionRecord],
            ]
        ] = []
        for index, _transition in enumerate(safe_transitions):
            current_record = records[index]
            current_explains = current_explains_by_index[index]
            assignment, rows = classified_by_index.get(
                int(index),
                (
                    SimpleNamespace(class_id=0, group_id=None, status="unknown"),
                    self._group_classifier.build_class_rows(current_class_id=None)
                    if include_rows
                    else [],
                ),
            )
            class_id = (
                int(assignment.class_id)
                if isinstance(getattr(assignment, "class_id", None), int)
                and int(assignment.class_id) > 0
                else None
            )
            metadata = self.resolve_transition_display_metadata(
                class_id=class_id,
                assignment=assignment,
            )
            current_is_question_mark = self._should_render_current_transition_as_question_mark(
                current_explains=current_explains,
                metadata=metadata,
            )
            self._remember_current_collect_focus(
                class_id=class_id,
                assignment=assignment,
                current_explains=current_explains,
                current_is_question_mark=current_is_question_mark,
            )
            rows = (
                self.decorate_current_transition_rows(
                    rows=rows,
                    class_id=class_id,
                    assignment=assignment,
                    current_explains=current_explains,
                    metadata=metadata,
                )
                if include_rows
                else []
            )
            results.append(
                (
                    class_id,
                    assignment,
                    rows,
                    current_explains,
                    current_record if isinstance(current_record, PredictionRecord) else None,
                )
            )
        return results

    def classify_transition(
        self,
        *,
        transition: Transition,
        include_rows: bool = True,
    ) -> Tuple[Optional[int], Any, List[Dict[str, Any]], Optional[bool]]:
        class_id, assignment, rows, current_explains, _record = (
            self.classify_transition_with_current_record(
                transition=transition,
                include_rows=include_rows,
            )
        )
        return class_id, assignment, rows, current_explains

    def build_class_rows(
        self,
        *,
        visible_count: int,
        current_class_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._group_classifier.build_class_rows(
            current_class_id=current_class_id,
        )
        limit = max(0, int(visible_count))
        if limit > 0:
            return rows[:limit]
        return rows

    def decorate_visualization_rows(
        self,
        rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        decorated_rows = [dict(row) for row in rows if isinstance(row, dict)]
        current_group_id = self._current_collect_group_id
        current_assignment = (
            SimpleNamespace(group_id=current_group_id)
            if isinstance(current_group_id, str) and current_group_id.strip()
            else None
        )
        if (
            current_assignment is not None
            or self._current_collect_class_id is not None
            or self._current_collect_explains is False
        ):
            return self.decorate_current_transition_rows(
                rows=decorated_rows,
                class_id=self._current_collect_class_id,
                assignment=current_assignment,
                current_explains=self._current_collect_explains,
            )
        if not self._has_seen_question_mark_transition:
            return decorated_rows
        return [
            *decorated_rows,
            self._build_new_dynamics_class_row(
                current_explains=None,
                is_explainer=False,
            ),
        ]

    def resolve_transition_display_metadata(
        self,
        *,
        class_id: Optional[int],
        assignment: Any,
    ) -> Dict[str, Any]:
        assignment_group_id = (
            str(assignment.group_id).strip()
            if isinstance(getattr(assignment, "group_id", None), str)
            and str(assignment.group_id).strip()
            else None
        )
        return self._group_classifier.resolve_group_display_metadata(
            class_id=(
                int(class_id)
                if isinstance(class_id, int) and int(class_id) > 0
                else None
            ),
            group_id=assignment_group_id,
        )

    def decorate_current_transition_rows(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        class_id: Optional[int],
        assignment: Any,
        current_explains: Optional[bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        decorated_rows = [
            dict(row)
            for row in rows
            if isinstance(row, dict) and not bool(row.get("is_new_dynamics_class"))
        ]
        resolved_metadata = (
            dict(metadata)
            if isinstance(metadata, dict)
            else self.resolve_transition_display_metadata(
                class_id=class_id,
                assignment=assignment,
            )
        )
        for row in decorated_rows:
            row["is_explainer"] = False
        current_is_question_mark = self._should_render_current_transition_as_question_mark(
            current_explains=current_explains,
            metadata=resolved_metadata,
        )
        if current_is_question_mark:
            return [
                *decorated_rows,
                self._build_new_dynamics_class_row(
                    current_explains=current_explains,
                    is_explainer=True,
                ),
            ]

        matched_index: Optional[int] = None
        for index, row in enumerate(decorated_rows):
            row_group_id = row.get("group_id")
            row_class_id = row.get("version_index")
            matches_group = (
                isinstance(resolved_metadata.get("group_id"), str)
                and isinstance(row_group_id, str)
                and row_group_id.strip() == str(resolved_metadata.get("group_id"))
            )
            matches_class = (
                int(resolved_metadata.get("class_id") or 0) > 0
                and isinstance(row_class_id, int)
                and int(row_class_id) == int(resolved_metadata.get("class_id"))
            )
            if matches_group or matches_class:
                matched_index = int(index)

        if matched_index is not None:
            row = dict(decorated_rows[matched_index])
            if isinstance(current_explains, bool):
                row["explains_current_state"] = bool(current_explains)
            row["is_explainer"] = True
            decorated_rows[matched_index] = row
        if self._has_seen_question_mark_transition:
            decorated_rows.append(
                self._build_new_dynamics_class_row(
                    current_explains=None,
                    is_explainer=False,
                )
            )
        return decorated_rows

    def _remember_current_collect_focus(
        self,
        *,
        class_id: Optional[int],
        assignment: Any,
        current_explains: Optional[bool],
        current_is_question_mark: bool,
    ) -> None:
        self._current_collect_class_id = (
            int(class_id)
            if isinstance(class_id, int) and int(class_id) > 0
            else None
        )
        self._current_collect_group_id = (
            str(assignment.group_id).strip()
            if isinstance(getattr(assignment, "group_id", None), str)
            and str(assignment.group_id).strip()
            else None
        )
        self._current_collect_explains = (
            bool(current_explains)
            if isinstance(current_explains, bool)
            else current_explains
        )
        if current_is_question_mark:
            self._current_collect_question_mark_count += 1
            self._has_seen_question_mark_transition = True

    def _resolve_current_program_source(
        self,
        context: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if not isinstance(context, dict):
            return None
        raw_current_source = context.get("current_source")
        if isinstance(raw_current_source, str) and raw_current_source.strip():
            return str(raw_current_source)
        current_version_id = context.get("current_version_id")
        raw_versions = context.get("versions")
        if not isinstance(current_version_id, str) or not current_version_id.strip():
            return None
        if not isinstance(raw_versions, list):
            return None
        for raw_version in raw_versions:
            if not isinstance(raw_version, dict):
                continue
            version_id = raw_version.get("version_id")
            version_source = raw_version.get("source")
            if (
                isinstance(version_id, str)
                and version_id.strip() == current_version_id.strip()
                and isinstance(version_source, str)
                and version_source.strip()
            ):
                return str(version_source)
        return None

    def _should_render_current_transition_as_question_mark(
        self,
        *,
        current_explains: Optional[bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if current_explains is False:
            return True
        if isinstance(metadata, dict) and bool(metadata.get("is_new_dynamics_class")):
            return True
        return False

    def _build_new_dynamics_class_row(
        self,
        *,
        current_explains: Optional[bool],
        is_explainer: bool,
    ) -> Dict[str, Any]:
        return {
            "version_id": "NEW!",
            "version_index": None,
            "group_id": None,
            "display_id": "new dynamics class!",
            "explains_current_state": (
                current_explains if isinstance(current_explains, bool) else None
            ),
            "prediction_status": "missing",
            "prediction_error": None,
            "is_active": True,
            "is_explainer": bool(is_explainer),
            "transition_count": max(0, int(self._current_collect_question_mark_count)),
            "is_current": False,
            "is_new_dynamics_class": True,
        }
