"""Program-aware dynamics-class assignment shared by explorers."""

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


class DynamicsClassAssignmentSupport:
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

    @property
    def snapshot(self) -> TransitionGroupContextSnapshot:
        return self._group_context_snapshot

    @property
    def known_class_count(self) -> int:
        return int(self._group_context_snapshot.known_class_count)

    @property
    def active_class_count(self) -> int:
        return int(self._group_context_snapshot.active_class_count)

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

    def classify_transitions_with_current_records(
        self,
        *,
        transitions: Sequence[Transition],
        current_records: Sequence[Optional[PredictionRecord]],
    ) -> List[
        Tuple[
            Optional[int],
            Any,
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
        classified_by_index: Dict[int, Any] = {}
        if known_indices:
            classified = self._group_classifier.classify_transitions(
                transitions=[safe_transitions[index] for index in known_indices],
                include_rows=False,
                known_explaining_version_id=(
                    self._group_classifier.snapshot.version_snapshot.current_version_id
                ),
            )
            for local_index, result in enumerate(classified):
                assignment, _rows = result
                classified_by_index[int(known_indices[int(local_index)])] = assignment

        results: List[Tuple[Optional[int], Any, Optional[bool], Optional[PredictionRecord]]] = []
        for index, _transition in enumerate(safe_transitions):
            current_record = records[index]
            current_explains = current_explains_by_index[index]
            assignment = classified_by_index.get(
                int(index),
                SimpleNamespace(class_id=0, group_id=None, status="unknown"),
            )
            class_id = (
                int(assignment.class_id)
                if isinstance(getattr(assignment, "class_id", None), int)
                and int(assignment.class_id) > 0
                else None
            )
            results.append(
                (
                    class_id,
                    assignment,
                    current_explains,
                    current_record if isinstance(current_record, PredictionRecord) else None,
                )
            )
        return results

    def resolve_transition_metadata(
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
