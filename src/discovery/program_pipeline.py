from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)
from zoneinfo import ZoneInfo

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None

from src.data import (
    CanonicalDataset,
    StateStore,
    Transition,
    canonical_graph_edge_identity_key_from_fields,
    canonical_graph_world_scope,
)
from src.discovery.artifact_renderer import TransitionArtifactRenderer
from src.discovery.program_resume import derive_last_program_patch_step
from src.discovery.saturation import SaturationChecker
from src.program_model import (
    LLMCallBudgetExceeded,
    ProgramPatchGenerator,
    ProgramPatcher,
    PredictionRecord,
    ProgramEvaluation,
    ProgramEvaluationTask,
    ProgramRepository,
    ProgramEvaluator,
    SandboxError,
    TransitionGroupClassifier,
    SandboxConfig,
    dump_state_json,
    parse_state_json,
)
from src.visualization import resolve_predicted_visual_state

if TYPE_CHECKING:
    from src.agents import BaseExplorer
    from src.llm import LLMPredictor, LLMUsageTracker


@dataclass(frozen=True)
class CollectPlan:
    max_transitions: Optional[int]
    schedule_label: str


@dataclass
class CollectRoundResult:
    should_collect: bool
    collected: List[Transition] = field(default_factory=list)
    added: int = 0
    collect_diag: Dict[str, Any] = field(default_factory=dict)
    termination_reason: Optional[str] = None


@dataclass(frozen=True)
class ProgramContextSnapshot:
    versions: List[Any]
    version_rows: List[Dict[str, Any]]
    current_version_id: Optional[str]
    group_context: Dict[str, Any]
    payload_signature: Tuple[Any, ...]
    canonical_context_signature: Tuple[Any, ...]


class ProgramDiscoveryPipeline:
    """Incremental CEGIS-style pipeline for program world model discovery."""

    def __init__(
        self,
        explorer: "BaseExplorer",
        llm_predictor: "LLMPredictor",
        max_fail_count: int = 3,
        saturation_global_step_window: Optional[int] = None,
        shuffle_collected_transitions: bool = True,
        max_collect_transitions_per_verify: Optional[int] = 1,
        canonical_dataset_max_size: Optional[int] = None,
        max_no_new_canonical_rounds: int = 1,
        save_new_transition_images: bool = False,
        output_dir: str = "experiments/",
        patch_unexpected_error_max_attempts: Optional[int] = None,
        max_total_llm_calls: Optional[int] = None,
        progress_bar_refresh_interval_sec: float = 5.0,
        regression_group_sample_k: int = 0,
        regression_witnesses_per_group: int = 1,
        dynamics_class_mode: str = "full",
        regression_witness_mode: str = "class_aware",
        random_seed: int = 0,
        sandbox_config: Optional[SandboxConfig] = None,
        evaluator: Optional[ProgramEvaluator] = None,
        prompt_additional_instructions: Optional[str] = None,
        patch_format: str = "line",
        resume_program_only: bool = False,
    ):
        if max_fail_count <= 0:
            raise ValueError("max_fail_count must be > 0.")
        if (
            saturation_global_step_window is not None
            and int(saturation_global_step_window) <= 0
        ):
            raise ValueError("saturation_global_step_window must be > 0 when provided.")
        if max_total_llm_calls is not None and int(max_total_llm_calls) <= 0:
            raise ValueError("max_total_llm_calls must be > 0 when provided.")
        try:
            resolved_max_no_new_rounds = int(max_no_new_canonical_rounds)
        except (TypeError, ValueError):
            resolved_max_no_new_rounds = 0
        resolved_max_patch_attempts = int(max_fail_count)
        if saturation_global_step_window is None:
            resolved_saturation_window = int(max_fail_count)
        else:
            resolved_saturation_window = int(saturation_global_step_window)

        self.explorer = explorer
        self._visualization_config = self._resolve_visualization_config(explorer)
        self.shuffle_collected_transitions = bool(shuffle_collected_transitions)
        self.max_collect_transitions_per_verify = max_collect_transitions_per_verify
        self.max_no_new_canonical_rounds = max(0, resolved_max_no_new_rounds)
        self.no_new_canonical_rounds = 0
        self.max_patch_attempts_per_target = max(1, resolved_max_patch_attempts)
        self.saturation_global_step_window = max(1, resolved_saturation_window)
        self.save_new_transition_images = bool(save_new_transition_images)
        if patch_unexpected_error_max_attempts is None:
            resolved_patch_unexpected_error_max_attempts = self.max_patch_attempts_per_target
        else:
            try:
                resolved_patch_unexpected_error_max_attempts = int(
                    patch_unexpected_error_max_attempts
                )
            except (TypeError, ValueError):
                resolved_patch_unexpected_error_max_attempts = self.max_patch_attempts_per_target
        self.patch_unexpected_error_max_attempts = max(
            1,
            resolved_patch_unexpected_error_max_attempts,
        )
        self.max_total_llm_calls = (
            int(max_total_llm_calls) if max_total_llm_calls is not None else None
        )
        try:
            resolved_progress_bar_refresh_interval_sec = float(
                progress_bar_refresh_interval_sec
            )
        except (TypeError, ValueError):
            resolved_progress_bar_refresh_interval_sec = 5.0
        if not math.isfinite(resolved_progress_bar_refresh_interval_sec):
            resolved_progress_bar_refresh_interval_sec = 5.0
        self.progress_bar_refresh_interval_sec = max(
            0.0,
            resolved_progress_bar_refresh_interval_sec,
        )
        try:
            resolved_regression_sample_k = int(regression_group_sample_k)
        except (TypeError, ValueError):
            resolved_regression_sample_k = 0
        self.regression_group_sample_k = max(0, resolved_regression_sample_k)
        try:
            resolved_regression_witnesses_per_group = int(
                regression_witnesses_per_group
            )
        except (TypeError, ValueError):
            resolved_regression_witnesses_per_group = 1
        self.regression_witnesses_per_group = max(
            1,
            resolved_regression_witnesses_per_group,
        )
        self.dynamics_class_mode = self._normalize_dynamics_class_mode(
            dynamics_class_mode
        )
        self.regression_witness_mode = self._normalize_regression_witness_mode(
            regression_witness_mode
        )
        try:
            resolved_random_seed = int(random_seed)
        except (TypeError, ValueError):
            resolved_random_seed = 0
        self.random_seed = resolved_random_seed
        self._prompt_rng = random.Random(self.random_seed)
        self._collection_rng = random.Random(self.random_seed + 1)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.new_transition_images_dir = self.output_dir / "new_transition_images"
        self.patch_attempts_dir = self.output_dir / "patch_attempts"
        if self.save_new_transition_images:
            self.new_transition_images_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_renderer = TransitionArtifactRenderer()

        self.state_store = StateStore()
        self.canonical_d = CanonicalDataset(
            max_size=canonical_dataset_max_size,
            state_store=self.state_store,
            on_change=self._mark_canonical_dataset_changed,
        )
        state_store_setter = getattr(self.explorer, "set_state_store", None)
        if callable(state_store_setter):
            state_store_setter(self.state_store)
        self._assert_explorer_uses_pipeline_state_store()
        self.protected_set: List[Transition] = []
        self._protected_keys = set()
        self._protected_transition_by_key: Dict[str, Transition] = {}
        self._protected_commit_version_by_key: Dict[str, str] = {}
        self._protected_leaf_group_id_by_key: Dict[str, str] = {}
        self._protected_group_metadata_by_id: Dict[str, Dict[str, Any]] = {}
        self._protected_group_member_key_sets_by_id: Dict[str, Set[Any]] = {}
        self._protected_root_group_id_by_commit_version: Dict[str, str] = {}
        self._protected_ingress_leaf_group_id_by_commit_version: Dict[str, Optional[str]] = {}
        self._next_group_index_by_commit_version: Dict[str, int] = {}
        self._group_class_id_by_group_id: Dict[str, int] = {}
        self._leaf_group_id_by_class_id: Dict[int, str] = {}
        self._next_group_class_id = 1
        self._group_context_generation = 0
        self._canonical_dataset_generation = 0
        self._last_program_context_payload_signature: Optional[Tuple[Any, ...]] = None
        self._feedback_by_transition: Dict[str, Dict] = {}
        self._current_source_transition_assessments: Dict[str, Dict[str, Any]] = {}
        self._transition_key_cache: Dict[Tuple[str, str, str], str] = {}
        self._requested_termination_reason: Optional[str] = None
        self._requested_termination_detail: Optional[str] = None
        self._last_termination_reason: Optional[str] = None
        self._last_termination_detail: Optional[str] = None

        self.saturation = SaturationChecker(
            max_fail_count=self.saturation_global_step_window
        )
        if evaluator is not None and sandbox_config is not None:
            raise ValueError("Provide either evaluator or sandbox_config, not both.")
        self.evaluator = evaluator or ProgramEvaluator(sandbox_config=sandbox_config)
        self.sandbox = self.evaluator.sandbox
        self._context_group_classifier = TransitionGroupClassifier(
            self.evaluator,
        )
        self._context_group_snapshot = None
        self._canonical_leaf_group_id_by_key: Dict[str, str] = {}
        self._canonical_class_id_by_key: Dict[str, int] = {}
        self._canonical_class_counts: Dict[int, int] = {}
        self._canonical_leaf_group_counts: Dict[str, int] = {}
        self._canonical_transition_keys_by_leaf_group_id: Dict[str, Set[str]] = {}
        self._canonical_classified_count = 0
        self._canonical_assignment_context_signature: Optional[Tuple[Any, ...]] = None
        self.patcher = ProgramPatcher(patch_format=patch_format)
        self.generator = ProgramPatchGenerator(
            llm_predictor=llm_predictor,
            patcher=self.patcher,
            additional_instructions=prompt_additional_instructions,
            usage_tracker=self._build_llm_usage_tracker(llm_predictor),
        )
        self.patch_format = self.patcher.patch_format
        self._configure_patch_generation_predictor(llm_predictor)
        self.llm_usage_tracker = self.generator.usage_tracker

        self.repository = ProgramRepository(root_dir=str(self.output_dir))
        self.current_version = self.repository.ensure_baseline()
        self.current_source = self.current_version.source
        if resume_program_only:
            self._restore_program_only_llm_usage_state()
            self._restore_program_only_state()
        self._last_synced_program_version_id: Optional[str] = None
        self._active_progress_bar: Optional[Any] = None
        self._sync_explorer_program_context(emit_log=False)

        self.iteration = 0
        self._data_index_by_iteration_and_key: Dict[int, Dict[str, int]] = {}
        self._next_data_index_by_iteration: Dict[int, int] = {}
        self._transition_step_index_by_key: Dict[str, int] = {}
        self._next_fallback_transition_step_index = 0
        self._failure_index_by_iteration_and_key: Dict[int, Dict[str, int]] = {}
        self._formal_failure_attempt_state_by_transition_key: Dict[
            str, Dict[str, Any]
        ] = {}
        self._active_collect_batch_queue: List[str] = []
        self._epoch_index = 0
        self._epoch_budget_size = 0
        self._epoch_budget_remaining = 0
        self._last_emitted_cli_header_iteration: Optional[int] = None
        self._patch_elapsed_log_interval_sec = 1.0
        self._progress_bar_last_refresh_at_by_id: Dict[int, float] = {}
        self._live_console_line_width = 0
        self._last_console_line_blank = True
        self._verify_pending_progress_state: Optional[Dict[str, Any]] = None
        self._verify_pending_live_status: Optional[str] = None
        self._deferred_explained_transition_batch: List[Transition] = []
        self._deferred_explorer_context_sync_needed = False
        self._program_started_at: Optional[float] = None
        self._current_iteration_started_at: Optional[float] = None
        self._current_collect_started_at: Optional[float] = None
        self._current_verify_started_at: Optional[float] = None

    def restore_iteration_boundary_state(self, resume_state: Any) -> Dict[str, Any]:
        state_store = getattr(resume_state, "state_store", None)
        if not isinstance(state_store, StateStore):
            raise TypeError("resume_state.state_store must be a StateStore.")
        cutoff_version = self._normalize_commit_version(
            getattr(resume_state, "cutoff_program_version_id", None)
        )
        current_version = self._normalize_commit_version(
            getattr(self.current_version, "version_id", None)
        )
        if cutoff_version != current_version:
            raise ValueError(
                "Discovery resume program version mismatch: "
                f"archive={cutoff_version} repository={current_version}. "
                "Copy program artifacts through the cutoff version before restoring."
            )

        old_max_size = getattr(self.canonical_d, "max_size", None)
        self.state_store = state_store
        self.canonical_d = CanonicalDataset(
            max_size=old_max_size,
            state_store=self.state_store,
            on_change=self._mark_canonical_dataset_changed,
        )
        state_store_setter = getattr(self.explorer, "set_state_store", None)
        if callable(state_store_setter):
            state_store_setter(self.state_store)
        self._assert_explorer_uses_pipeline_state_store()

        transitions = list(getattr(resume_state, "canonical_transitions", ()) or ())
        added, _added_transitions, _removed_transitions = self.canonical_d.merge_with_details(
            transitions
        )
        if int(added) != len(transitions):
            raise RuntimeError(
                "Discovery resume canonical restore lost transitions: "
                f"added={int(added)} expected={len(transitions)}."
            )

        self._transition_key_cache.clear()
        self._feedback_by_transition.clear()
        self._current_source_transition_assessments.clear()
        self._formal_failure_attempt_state_by_transition_key.clear()
        self._data_index_by_iteration_and_key.clear()
        self._next_data_index_by_iteration.clear()
        self._failure_index_by_iteration_and_key.clear()
        self._active_collect_batch_queue.clear()
        self._deferred_explained_transition_batch.clear()
        self._deferred_explorer_context_sync_needed = False

        self._restore_iteration_boundary_protected_state(resume_state)
        explorer_restore = getattr(self.explorer, "restore_iteration_boundary_state", None)
        explorer_summary = (
            explorer_restore(resume_state) if callable(explorer_restore) else None
        )

        self.iteration = int(getattr(resume_state, "cutoff_iteration", 0) or 0)
        cutoff_global_step = int(getattr(resume_state, "cutoff_global_step", 0) or 0)
        self._next_fallback_transition_step_index = max(0, cutoff_global_step)
        self._transition_step_index_by_key = {}
        self._epoch_index = 0
        self._epoch_budget_size = 0
        self._epoch_budget_remaining = 0
        self._restore_iteration_boundary_saturation_state(
            resume_state,
            cutoff_global_step=cutoff_global_step,
        )
        self._restore_iteration_boundary_llm_usage(resume_state)
        self._last_synced_program_version_id = None
        self._sync_explorer_program_context(emit_log=False)
        visualizer = getattr(self.explorer, "visualizer", None)
        emit_snapshot = getattr(visualizer, "emit_snapshot", None)
        if callable(emit_snapshot):
            emit_snapshot(force=True, bypass_delivery_gate=True)

        return {
            "iteration": int(self.iteration),
            "global_step": int(cutoff_global_step),
            "program_version": current_version,
            "canonical_count": int(len(self.canonical_d)),
            "protected_count": int(len(self.protected_set)),
            "explorer": explorer_summary,
        }

    def _restore_iteration_boundary_protected_state(self, resume_state: Any) -> None:
        self._restore_program_only_state()
        self._canonical_leaf_group_id_by_key = {}
        self._canonical_class_id_by_key = {}
        self._canonical_class_counts = {}
        self._canonical_leaf_group_counts = {}
        self._canonical_transition_keys_by_leaf_group_id = {}
        self._canonical_classified_count = 0

        class_metadata_by_id = getattr(resume_state, "class_metadata_by_id", {}) or {}
        for raw_class_id, raw_metadata in sorted(class_metadata_by_id.items()):
            if not isinstance(raw_class_id, int) or int(raw_class_id) <= 0:
                continue
            if not isinstance(raw_metadata, dict):
                continue
            raw_group_id = raw_metadata.get("group_id")
            if not isinstance(raw_group_id, str) or not raw_group_id.strip():
                continue
            group_id = raw_group_id.strip()
            commit_version = self._normalize_commit_version(
                raw_metadata.get("commit_version")
                or self._commit_version_from_group_id(group_id)
            )
            group = self._upsert_program_resume_group(
                group_id=group_id,
                commit_version=commit_version,
            )
            group["class_id"] = int(raw_class_id)
            transition_count = raw_metadata.get("transition_count")
            if isinstance(transition_count, int) and int(transition_count) > 0:
                group["restored_member_transition_count"] = int(transition_count)
            self._group_class_id_by_group_id[group_id] = int(raw_class_id)
            self._leaf_group_id_by_class_id[int(raw_class_id)] = group_id

        transitions = self.canonical_d.get_all()
        keys = self.canonical_d.transition_keys()
        class_by_key = getattr(resume_state, "transition_class_id_by_key", {}) or {}
        group_by_key = getattr(resume_state, "transition_group_id_by_key", {}) or {}
        cutoff_version = self._normalize_commit_version(
            getattr(resume_state, "cutoff_program_version_id", None)
        )

        self.protected_set = list(transitions)
        self._protected_keys = set(keys)
        self._protected_transition_by_key = {
            str(key): transition for key, transition in zip(keys, transitions)
        }
        for key in keys:
            safe_key = str(key)
            class_id = class_by_key.get(safe_key)
            group_id = group_by_key.get(safe_key)
            commit_version = cutoff_version
            if (
                isinstance(class_id, int)
                and int(class_id) > 0
                and isinstance(group_id, str)
                and group_id in self._protected_group_metadata_by_id
            ):
                group = self._protected_group_metadata_by_id[group_id]
                commit_version = self._normalize_commit_version(
                    group.get("commit_version")
                )
                self._protected_leaf_group_id_by_key[safe_key] = group_id
                self._append_transition_key_to_group_lineage(group_id, safe_key)
                self._set_canonical_group_assignment(
                    safe_key,
                    class_id=int(class_id),
                    group_id=group_id,
                )
            self._protected_commit_version_by_key[safe_key] = commit_version

        if self._group_class_id_by_group_id:
            self._next_group_class_id = max(self._group_class_id_by_group_id.values()) + 1
        self._group_context_generation += 1
        self._sanitize_protected_group_registry()

    def _restore_iteration_boundary_saturation_state(
        self,
        resume_state: Any,
        *,
        cutoff_global_step: int,
    ) -> None:
        self.no_new_canonical_rounds = max(
            0,
            int(getattr(resume_state, "no_new_canonical_rounds", 0) or 0),
        )
        self.saturation.reset()
        last_seen = getattr(resume_state, "saturation_last_seen_global_step", None)
        if not isinstance(last_seen, int) or int(last_seen) < 0:
            last_seen = int(cutoff_global_step)
        last_patch = getattr(resume_state, "saturation_last_patch_global_step", None)
        if isinstance(last_patch, int) and int(last_patch) >= 0:
            self.saturation.last_patch_global_step = min(int(last_patch), int(last_seen))
            self.saturation.observe(global_step=int(last_seen))
            return

        fail_count = getattr(resume_state, "saturation_fail_count", None)
        if isinstance(fail_count, int) and int(fail_count) >= 0:
            self.saturation.last_patch_global_step = max(
                0,
                int(last_seen) - int(fail_count),
            )
            self.saturation.observe(global_step=int(last_seen))
            return

        self.saturation.mark_successful_patch(global_step=int(cutoff_global_step))

    def _restore_program_only_llm_usage_state(self) -> None:
        usage_path = self.output_dir / "llm_usage_summary.json"
        if not usage_path.exists():
            return
        try:
            usage_summary = json.loads(usage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid LLM usage summary: {usage_path}") from exc
        if not isinstance(usage_summary, dict):
            return
        events = usage_summary.get("events")
        if not isinstance(events, list):
            return
        self._restore_llm_usage_events(events, replace=True)

    def _restore_iteration_boundary_llm_usage(self, resume_state: Any) -> None:
        llm_calls = int(getattr(resume_state, "llm_calls", 0) or 0)
        event_payloads = getattr(resume_state, "llm_usage_events", ()) or ()
        self._restore_llm_usage_events(
            event_payloads,
            target_count=llm_calls,
            replace=True,
        )

    def _restore_llm_usage_events(
        self,
        event_payloads: Sequence[Mapping[str, Any]],
        *,
        target_count: Optional[int] = None,
        replace: bool,
    ) -> None:
        payloads = list(event_payloads)
        if target_count is not None:
            payloads = payloads[: max(0, int(target_count))]
        restore_events = getattr(self.llm_usage_tracker, "restore_events", None)
        if callable(restore_events):
            restore_events(payloads, replace=replace, emit_update=False)
        else:
            events = getattr(self.llm_usage_tracker, "events", None)
            if isinstance(events, list):
                if replace:
                    events.clear()
                from src.llm.usage_summary import LLMUsageEvent

                events.extend(LLMUsageEvent.from_dict(payload) for payload in payloads)
        if target_count is not None:
            self._pad_llm_usage_event_count(target_count)
        self._write_incremental_llm_usage_summary()

    def _restore_iteration_boundary_llm_usage_count(self, llm_calls: int) -> None:
        self._restore_llm_usage_events((), target_count=llm_calls, replace=False)

    def _pad_llm_usage_event_count(self, llm_calls: int) -> None:
        target_count = max(0, int(llm_calls))
        events = getattr(self.llm_usage_tracker, "events", None)
        if not isinstance(events, list) or len(events) >= target_count:
            return
        from src.llm.usage_summary import LLMUsageEvent

        while len(events) < target_count:
            events.append(
                LLMUsageEvent(
                    component="resume",
                    provider="resume",
                    model="resume",
                    prompt_tokens=0,
                    reasoning_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    outcome="success",
                    prompt_chars=0,
                    response_chars=0,
                    timestamp_utc=datetime.utcnow().isoformat() + "Z",
                )
            )

    def close(self) -> None:
        evaluator_close = getattr(self.evaluator, "close", None)
        if callable(evaluator_close):
            evaluator_close()

    def _normalize_dynamics_class_mode(self, value: Any) -> str:
        normalized = str(value or "full").strip().lower().replace("-", "_")
        aliases = {
            "ours": "full",
            "full": "full",
            "repair_induced": "full",
            "version": "version_only",
            "version_class": "version_only",
            "version_only": "version_only",
            "coarse": "version_only",
            "none": "none",
            "no_class": "none",
            "no_classes": "none",
        }
        if normalized not in aliases:
            supported = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unsupported dynamics_class_mode `{value}`. Supported: {supported}"
            )
        return aliases[normalized]

    def _normalize_regression_witness_mode(self, value: Any) -> str:
        normalized = str(value or "class_aware").strip().lower().replace("-", "_")
        aliases = {
            "class": "class_aware",
            "class_aware": "class_aware",
            "grouped": "class_aware",
            "random": "random",
            "none": "none",
        }
        if normalized not in aliases:
            supported = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unsupported regression_witness_mode `{value}`. Supported: {supported}"
            )
        return aliases[normalized]

    def _uses_dynamics_classes(self) -> bool:
        return self.dynamics_class_mode != "none"

    def _allows_dynamics_class_refinement(self) -> bool:
        return self.dynamics_class_mode == "full"

    def _resolve_positive_int_config(self, value: Any, *, field_name: str) -> int:
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{field_name} must be a positive integer.")
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a positive integer.") from None
        if resolved <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")
        return int(resolved)

    def _resolve_optional_positive_int_config(
        self,
        value: Any,
        *,
        field_name: str,
    ) -> Optional[int]:
        if value is None:
            return None
        return self._resolve_positive_int_config(value, field_name=field_name)

    def _configure_patch_generation_predictor(self, llm_predictor: "LLMPredictor") -> None:
        setattr(llm_predictor, "_progress_log_interval_sec", 0.0)

    def _build_llm_usage_tracker(self, llm_predictor: "LLMPredictor") -> "LLMUsageTracker":
        from src.llm import LLMUsageTracker

        del llm_predictor
        return LLMUsageTracker(on_update=self._write_incremental_llm_usage_summary)

    def _current_program_revision_count(self) -> int:
        current_index = getattr(self.current_version, "index", None)
        if isinstance(current_index, int) and int(current_index) >= 0:
            return int(current_index) + 1
        version_id = getattr(self.current_version, "version_id", None)
        if isinstance(version_id, str) and version_id.startswith("v"):
            suffix = version_id[1:]
            if suffix.isdigit():
                return int(suffix) + 1
        return 0

    def _current_active_class_count(self) -> int:
        if not self._uses_dynamics_classes():
            return 1 if self._no_class_singleton_context_is_active() else 0
        return int(len(self._leaf_group_id_by_class_id))

    def _write_llm_usage_summary_files(self, llm_usage_summary: Dict[str, Any]) -> None:
        with open(self.output_dir / "llm_usage_summary.json", "w", encoding="utf-8") as f:
            json.dump(llm_usage_summary, f, indent=2, ensure_ascii=False)

    def _write_incremental_llm_usage_summary(self) -> None:
        llm_usage_summary = self.llm_usage_tracker.build_summary()
        llm_usage_summary["limits"] = {
            "max_total_llm_calls": self.max_total_llm_calls,
            "remaining_llm_calls": self._remaining_llm_call_budget(),
        }
        self._write_llm_usage_summary_files(llm_usage_summary)

    def _total_llm_call_count(self) -> int:
        return len(getattr(self.llm_usage_tracker, "events", []))

    def _remaining_llm_call_budget(self) -> Optional[int]:
        if self.max_total_llm_calls is None:
            return None
        return max(0, self.max_total_llm_calls - self._total_llm_call_count())

    def _has_reached_llm_call_limit(self) -> bool:
        if self.max_total_llm_calls is None:
            return False
        return self._total_llm_call_count() >= self.max_total_llm_calls

    def _should_stop_for_saturation(self) -> bool:
        if not self.saturation.is_saturated:
            return False
        self._sync_active_collect_batch_pending()
        return not bool(self._active_collect_batch_queue)

    def _resolve_positive_collect_limit(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _resolve_collect_plan(self) -> CollectPlan:
        max_transitions = self._resolve_positive_collect_limit(
            self.max_collect_transitions_per_verify
        )
        if max_transitions is None:
            raise ValueError(
                "ProgramDiscoveryPipeline requires max_collect_transitions_per_verify "
                "to define the collect-to-verify transition budget."
            )
        return CollectPlan(
            max_transitions=max_transitions,
            schedule_label="verify_budget",
        )

    def _format_collect_schedule_line(self, plan: CollectPlan) -> str:
        total_cap_text = (
            f"total_cap={int(plan.max_transitions)}"
            if isinstance(plan.max_transitions, int)
            else "total_cap=unbounded"
        )
        return (
            "Collect/verify schedule: keep collecting transitions "
            f"({total_cap_text}), "
            "then move pending targets into verify until the pending batch is exhausted."
        )

    def _collect_transitions_until_budget(
        self,
        collect_plan: CollectPlan,
    ) -> Tuple[List[Transition], Dict[str, Any]]:
        transition_budget = collect_plan.max_transitions
        if transition_budget is None:
            raise ValueError(
                "Collect plan must define max_transitions before collection starts."
            )

        raw_collected: List[Transition] = []
        collect_calls = 0
        last_collect_diag: Dict[str, Any] = {}
        collect_call_stats: List[Dict[str, Any]] = []
        previous_progress_bar = getattr(self, "_active_progress_bar", None)
        progress_bar: Optional[Any] = None
        collect_started_at: Optional[float] = None
        collected_before_current_call = 0
        self._current_collect_started_at = None
        train_progress_handler, close_train_progress = self._make_train_progress_handler()

        try:
            def _emit_collect_start_once() -> None:
                nonlocal progress_bar, collect_started_at
                if collect_started_at is not None:
                    return
                collect_started_at = perf_counter()
                self._current_collect_started_at = float(collect_started_at)
                self._emit_collect_phase_start(collect_plan=collect_plan)
                progress_bar = self._create_collect_progress_bar(
                    total=int(transition_budget)
                )

            def _progress_callback(
                payload: Any,
                *,
                _transition_budget: int = int(transition_budget),
            ) -> None:
                if not isinstance(payload, dict):
                    return
                if payload.get("progress_kind") == "frontier_refresh":
                    self._emit_frontier_refresh_progress(payload)
                    phase = (
                        str(payload.get("frontier_refresh_phase") or "")
                        .strip()
                        .lower()
                    )
                    if phase == "end":
                        _emit_collect_start_once()
                    return
                if payload.get("progress_kind") == "collect_prep":
                    _emit_collect_start_once()
                    self._emit_collect_prep_progress(payload)
                    return
                _emit_collect_start_once()
                current_count = payload.get("transitions_collected")
                try:
                    collected_count = int(current_count)
                except (TypeError, ValueError):
                    collected_count = None
                if isinstance(collected_count, int) and progress_bar is not None:
                    self._update_collect_progress_bar(
                        progress_bar,
                        collected=max(
                            0,
                            int(collected_before_current_call)
                            + int(collected_count),
                        ),
                        total=_transition_budget,
                    )
                train_progress_handler(payload)

            def _collect_resume_pending(diag: Dict[str, Any]) -> Optional[bool]:
                last_collect = diag.get("last_collect")
                if isinstance(last_collect, dict):
                    value = last_collect.get("resume_pending")
                    if isinstance(value, bool):
                        return bool(value)
                value = diag.get("resume_pending")
                if isinstance(value, bool):
                    return bool(value)
                return None

            progress_callback = _progress_callback

            should_defer_collect_start = False
            explorer = getattr(self, "explorer", None)
            should_use_frontier_scoring = getattr(
                explorer,
                "_should_use_frontier_scoring",
                None,
            )
            if callable(should_use_frontier_scoring):
                should_defer_collect_start = bool(should_use_frontier_scoring())
            if not should_defer_collect_start:
                _emit_collect_start_once()
            while len(raw_collected) < int(transition_budget):
                remaining_budget = int(transition_budget) - int(len(raw_collected))
                if remaining_budget <= 0:
                    break
                collected_before_current_call = int(len(raw_collected))
                collect_calls += 1
                collected_batch = self._collect_with_optional_progress(
                    max_transitions=int(remaining_budget),
                    progress_callback=progress_callback,
                )
                _emit_collect_start_once()
                last_collect_diag = self._collect_explorer_diagnostics()
                last_collect = last_collect_diag.get("last_collect")
                if isinstance(last_collect, dict):
                    collect_call_stats.append(dict(last_collect))
                if collected_batch:
                    raw_collected.extend(collected_batch)
                self._update_collect_progress_bar(
                    progress_bar,
                    collected=len(raw_collected),
                    total=int(transition_budget),
                )
                if len(raw_collected) >= int(transition_budget):
                    break
                if not collected_batch:
                    break
                if _collect_resume_pending(last_collect_diag) is not True:
                    break
        finally:
            close_train_progress()
            self._close_collect_progress_bar(
                progress_bar,
                previous_pbar=previous_progress_bar,
            )

        if not isinstance(last_collect_diag, dict):
            last_collect_diag = {}
        else:
            last_collect_diag = dict(last_collect_diag)

        last_collect = last_collect_diag.get("last_collect")
        last_collect_payload = dict(last_collect) if isinstance(last_collect, dict) else {}
        for key in (
            "worlds_started",
            "worlds_processed",
            "world_visits",
            "worlds_completed",
        ):
            values = [
                int(row[key])
                for row in collect_call_stats
                if isinstance(row.get(key), int)
            ]
            if values:
                last_collect_payload[key] = int(sum(values))
        for key in (
            "prototype_refresh_elapsed_sec",
            "collect_setup_elapsed_sec",
            "collect_prep_elapsed_sec",
        ):
            values = [
                float(row[key])
                for row in collect_call_stats
                if isinstance(row.get(key), (int, float))
            ]
            if values:
                last_collect_payload[key] = float(sum(values))
        last_collect_payload["transitions_collected"] = int(len(raw_collected))
        last_collect_payload["collect_calls"] = int(collect_calls)
        last_collect_payload["round_transition_budget"] = int(transition_budget)
        last_collect_payload["stopped_by_max_transitions"] = bool(
            len(raw_collected) >= int(transition_budget)
        )
        resume_pending = None
        raw_resume_pending = last_collect_payload.get("resume_pending")
        if isinstance(raw_resume_pending, bool):
            resume_pending = bool(raw_resume_pending)
            last_collect_diag["resume_pending"] = bool(raw_resume_pending)
        if isinstance(collect_started_at, (int, float)):
            collect_elapsed_sec = max(0.0, perf_counter() - float(collect_started_at))
            last_collect_payload["collect_elapsed_sec"] = float(collect_elapsed_sec)
            last_collect_diag["collect_elapsed_sec"] = float(collect_elapsed_sec)
        last_collect_diag["last_collect"] = last_collect_payload
        if resume_pending is not None:
            last_collect_diag["resume_pending"] = bool(resume_pending)
        last_collect_diag["round_collect_calls"] = int(collect_calls)
        last_collect_diag["round_transition_budget"] = int(transition_budget)
        return raw_collected, last_collect_diag

    def _is_manual_transition_source_exhausted(
        self,
        collect_diag: Mapping[str, Any],
    ) -> bool:
        if not isinstance(collect_diag, Mapping):
            return False
        explorer_name = str(collect_diag.get("name", "")).strip().lower()
        if explorer_name != "manual_transition":
            return False

        last_collect = collect_diag.get("last_collect")
        last_collect_payload = (
            last_collect if isinstance(last_collect, Mapping) else {}
        )
        if collect_diag.get("exhausted") is True:
            return True
        if last_collect_payload.get("exhausted") is True:
            return True

        try:
            remaining = int(collect_diag.get("remaining_transitions"))
        except (TypeError, ValueError):
            remaining = None
        try:
            collected = int(last_collect_payload.get("transitions_collected"))
        except (TypeError, ValueError):
            collected = None
        return remaining == 0 and collected == 0

    def _clear_requested_termination(self) -> None:
        self._requested_termination_reason = None
        self._requested_termination_detail = None

    def _request_termination(
        self,
        *,
        reason: str,
        detail: Optional[str] = None,
    ) -> None:
        self._requested_termination_reason = str(reason).strip() or "manual"
        if isinstance(detail, str) and detail.strip():
            self._requested_termination_detail = detail.strip()
        else:
            self._requested_termination_detail = None

    def _update_saturation_from_collection(
        self,
        *,
        collect_diag: Dict[str, Any],
    ) -> bool:
        global_step = self._resolve_explorer_global_step(collect_diag)
        is_saturated = self.saturation.observe(global_step=global_step)
        if is_saturated:
            self._request_termination(
                reason="saturation",
                detail=self._format_saturation_termination_message(),
            )
        elif self._requested_termination_reason == "saturation":
            self._clear_requested_termination()
        return is_saturated

    def _emit_collect_phase_start(
        self,
        *,
        collect_plan: CollectPlan,
    ) -> None:
        if isinstance(collect_plan.max_transitions, int):
            message = f"budget={int(collect_plan.max_transitions)}"
        else:
            message = "budget=unbounded"
        self._emit_elapsed_log_event("COLLECT", message, max_len=260)

    def _emit_collect_phase_end(
        self,
        *,
        should_collect: bool,
        collect_diag: Dict[str, Any],
        collected_count: int,
        added_count: int,
        pending_count: int,
    ) -> None:
        if not should_collect:
            return
        parts = [
            f"collected={max(0, int(collected_count))}",
        ]
        elapsed_sec = collect_diag.get("collect_elapsed_sec")
        del added_count
        del pending_count
        if isinstance(elapsed_sec, (int, float)):
            parts.append(f"elapsed={self._format_patch_elapsed(float(elapsed_sec))}")
        self._emit_elapsed_log_event(
            "COLLECT_END",
            " ".join(parts),
            max_len=260,
        )

    def _emit_collect_prep_progress(self, payload: Dict[str, Any]) -> None:
        prototype_elapsed_sec = payload.get("prototype_refresh_elapsed_sec")
        setup_elapsed_sec = payload.get("collect_setup_elapsed_sec")
        total_elapsed_sec = payload.get("collect_prep_elapsed_sec")
        if not isinstance(total_elapsed_sec, (int, float)):
            total_elapsed_sec = 0.0
            if isinstance(prototype_elapsed_sec, (int, float)):
                total_elapsed_sec += max(0.0, float(prototype_elapsed_sec))
            if isinstance(setup_elapsed_sec, (int, float)):
                total_elapsed_sec += max(0.0, float(setup_elapsed_sec))
        parts: List[str] = []
        if isinstance(prototype_elapsed_sec, (int, float)):
            parts.append(
                f"prototype={self._format_patch_elapsed(float(prototype_elapsed_sec))}"
            )
        if isinstance(setup_elapsed_sec, (int, float)):
            parts.append(f"setup={self._format_patch_elapsed(float(setup_elapsed_sec))}")
        active_maps = payload.get("collect_prep_active_maps")
        dispatch_maps = payload.get("collect_prep_dispatch_maps")
        if isinstance(active_maps, int) and isinstance(dispatch_maps, int):
            parts.append(
                f"maps={max(0, int(dispatch_maps))}/{max(0, int(active_maps))}"
            )
        self._emit_timing_log(
            "COLLECT_PREP_END",
            parts,
            elapsed_sec=float(total_elapsed_sec),
            elapsed_insert_index=2,
            max_len=260,
        )

    @staticmethod
    def _format_frontier_refresh_count(sampled: int, total: int) -> str:
        return f"{max(0, int(sampled))}/{max(0, int(total))}"

    def _emit_frontier_refresh_progress(self, payload: Dict[str, Any]) -> None:
        phase = str(payload.get("frontier_refresh_phase") or "").strip().lower()
        replay_count = self._format_frontier_refresh_count(
            payload.get("frontier_replay_sampled_samples"),
            payload.get("frontier_replay_total_samples"),
        )
        frontier_count = self._format_frontier_refresh_count(
            payload.get("frontier_sampled_candidates"),
            payload.get("frontier_total_candidates"),
        )
        parts = [
            f"replay={replay_count}",
            f"frontier={frontier_count}",
        ]
        if phase == "end":
            parts = []
            scored = payload.get("frontier_scored_candidates")
            if isinstance(scored, int):
                parts.append(f"scored={max(0, int(scored))}")
            select_elapsed_sec = payload.get("frontier_select_elapsed_sec")
            if isinstance(select_elapsed_sec, (int, float)):
                parts.append(
                    f"select_elapsed={self._format_patch_elapsed(float(select_elapsed_sec))}"
                )
            score_elapsed_sec = payload.get("frontier_score_elapsed_sec")
            if isinstance(score_elapsed_sec, (int, float)):
                parts.append(
                    f"score_elapsed={self._format_patch_elapsed(float(score_elapsed_sec))}"
                )
            elapsed_sec = payload.get("frontier_refresh_elapsed_sec")
            if isinstance(elapsed_sec, (int, float)):
                parts.append(f"elapsed={self._format_patch_elapsed(float(elapsed_sec))}")
            self._emit_elapsed_log_event(
                "FRONTIER_REFRESH_END",
                " ".join(parts),
                max_len=260,
            )
            return
        self._emit_elapsed_log_event(
            "FRONTIER_REFRESH",
            " ".join(parts),
            max_len=260,
        )

    def _verify_accuracy_text(self) -> str:
        canonical_total = len(self.canonical_d)
        if canonical_total <= 0:
            accuracy = 1.0
        else:
            accuracy = len(self.protected_set) / float(canonical_total)
        return f"{accuracy:.4f}"

    def _emit_verify_phase_start(
        self,
        *,
        epoch_index: int,
        pending_count: int,
    ) -> None:
        protected_count = len(self.protected_set)
        canonical_count = len(self.canonical_d)
        parts = [
            f"epoch={int(epoch_index)}",
            f"pending={max(0, int(pending_count))}",
            f"protected={protected_count}/{canonical_count}",
            f"acc={self._verify_accuracy_text()}",
        ]
        self._current_verify_started_at = perf_counter()
        self._emit_elapsed_log_event("VERIFY", " ".join(parts), max_len=260)

    def _emit_verify_phase_end(
        self,
        *,
        epoch_index: int,
        verified_count: int,
        total_count: int,
    ) -> None:
        protected_count = len(self.protected_set)
        canonical_count = len(self.canonical_d)
        parts = [
            f"epoch={int(epoch_index)}",
            f"verified={max(0, int(verified_count))}/{max(0, int(total_count))}",
            f"protected={protected_count}/{canonical_count}",
            f"acc={self._verify_accuracy_text()}",
        ]
        verify_started_at = self._current_verify_started_at
        if isinstance(verify_started_at, (int, float)):
            parts.append(
                "elapsed="
                + self._format_patch_elapsed(perf_counter() - float(verify_started_at))
            )
            self._current_verify_started_at = None
        self._emit_elapsed_log_event(
            "VERIFY_END",
            " ".join(parts),
            max_len=260,
        )

    def _build_train_summary_payload(
        self,
        finalize_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = self._collect_explorer_diagnostics()
        if not isinstance(payload, dict):
            payload = {}
        else:
            payload = dict(payload)
        if isinstance(finalize_result, dict):
            payload.update(finalize_result)
        return payload

    def _emit_train_phase_start(self, transitions: List[Transition]) -> None:
        payload = self._build_train_summary_payload()
        summary = self._format_explorer_train_summary(payload, include_state=False)
        parts = [f"verified={len(transitions)}"]
        if isinstance(summary, str) and summary:
            parts.append(summary)
        self._emit_elapsed_log_event("TRAIN", " ".join(parts), max_len=260)

    def _emit_train_phase_end(
        self,
        *,
        finalize_result: Optional[Dict[str, Any]],
        elapsed_text: str,
        failed: bool,
    ) -> None:
        payload = self._build_train_summary_payload(finalize_result)
        summary = self._format_explorer_train_summary(
            payload,
            include_metrics=True,
            include_state=False,
        )
        parts: List[str] = []
        if failed:
            parts.append("status=failed")
        if isinstance(summary, str) and summary:
            parts.append(summary)
        if isinstance(elapsed_text, str) and elapsed_text.strip():
            parts.append(f"elapsed={elapsed_text.strip()}")
        self._emit_elapsed_log_event(
            "TRAIN_END",
            " ".join(parts),
            max_len=260,
        )

    def _collect_round(
        self,
        *,
        max_iterations: Optional[int],
        collect_plan: Optional[CollectPlan] = None,
    ) -> CollectRoundResult:
        self._sync_active_collect_batch_pending()
        if self._active_collect_batch_queue:
            return CollectRoundResult(should_collect=False)
        if max_iterations is not None and self.iteration >= max_iterations:
            return CollectRoundResult(
                should_collect=True,
                termination_reason="max_iterations",
            )

        resolved_collect_plan = collect_plan or self._resolve_collect_plan()
        self.iteration += 1
        self._current_iteration_started_at = perf_counter()
        self._emit_iteration_banner(None, max_iterations=max_iterations)
        self._sync_explorer_program_context()

        raw_collected, collect_diag = self._collect_transitions_until_budget(
            resolved_collect_plan
        )
        if not raw_collected:
            if self._is_manual_transition_source_exhausted(collect_diag):
                source_dir = ""
                last_collect = collect_diag.get("last_collect")
                if isinstance(last_collect, Mapping):
                    source_dir = str(last_collect.get("source_dir", "")).strip()
                if not source_dir:
                    source_dir = str(collect_diag.get("source_dir", "")).strip()
                detail = (
                    "Manual transition source exhausted; no more offline "
                    "transitions are available to collect."
                )
                if source_dir:
                    detail += f" Source: {source_dir}"
                self._request_termination(
                    reason="data_exhausted",
                    detail=detail,
                )
                return CollectRoundResult(
                    should_collect=True,
                    collect_diag=collect_diag,
                    termination_reason="data_exhausted",
                )
            raise RuntimeError(
                "Collect invariant violated: explorer returned zero transitions "
                f"for positive collect budget={int(resolved_collect_plan.max_transitions)}. "
                f"collect_diag={collect_diag}"
            )
        collect_post_started_at = perf_counter()
        self._remember_collected_transition_step_indices(
            transitions=raw_collected,
            collect_diag=collect_diag,
        )
        collected = self._prepare_collected_transitions(raw_collected)

        added, added_transitions, removed_transitions = self.canonical_d.merge_with_details(
            collected
        )
        self._active_collect_batch_queue = self._resolve_collect_batch_queue(
            transitions=added_transitions
        )
        self._consume_explorer_current_source_transition_assessments()
        self._update_saturation_from_collection(
            collect_diag=collect_diag,
        )
        if added_transitions:
            self._refresh_canonical_group_assignments_from_cached_assessments(
                transitions=added_transitions,
                removed_transitions=removed_transitions,
                allow_classification=False,
            )
        self._publish_collection_feedback(added_count=added)
        if collected:
            collected_transition = collected[0]
            collected_transition_key = self._transition_key(collected_transition)
            collected_assessment = self._ensure_current_source_transition_assessments(
                [collected_transition]
            ).get(collected_transition_key)
            if (
                collected_transition_key in self._protected_keys
                and isinstance(collected_assessment, dict)
                and collected_assessment.get("current_explains") is True
            ):
                self._finalize_explained_transitions(
                    [collected_transition],
                )
        collect_post_elapsed_sec = perf_counter() - collect_post_started_at
        self._emit_timing_log(
            "COLLECT_POST_END",
            [
                f"added={int(added)}",
            ],
            elapsed_sec=collect_post_elapsed_sec,
            max_len=260,
        )
        return CollectRoundResult(
            should_collect=True,
            collected=collected,
            added=int(added),
            collect_diag=collect_diag,
        )

    def _sync_explorer_program_context(
        self,
        emit_log: bool = True,
    ) -> None:
        context_setter = getattr(self.explorer, "set_program_context", None)
        if not callable(context_setter):
            return
        previous_version = self._last_synced_program_version_id
        current_version = getattr(self.current_version, "version_id", None)
        current_version_text = (
            str(current_version).strip() if isinstance(current_version, str) else ""
        )
        try:
            snapshot = self._build_program_context_snapshot()
            if snapshot.payload_signature == self._last_program_context_payload_signature:
                return
            program_context_payload = self._build_program_context_payload_from_snapshot(
                snapshot=snapshot
            )
            context_setter(program_context_payload)
            visualizer = getattr(self.explorer, "visualizer", None)
            update_payload_only = getattr(visualizer, "update_payload_only", None)
            if callable(update_payload_only):
                update_payload_only(
                    program_context=program_context_payload,
                    update_dashboard=False,
                )
            self._last_program_context_payload_signature = snapshot.payload_signature
            self._deferred_explorer_context_sync_needed = False
        except Exception as e:  # noqa: BLE001
            self._emit_plain_line(f"  [WARN] explorer.set_program_context failed: {e}")
            return

        if current_version_text == previous_version:
            return
        self._last_synced_program_version_id = current_version_text
        self._current_source_transition_assessments = {}
        prev_text = previous_version or "-"
        curr_text = current_version_text or "-"
        revision_count = self._current_program_revision_count()
        active_class_count = self._current_active_class_count()
        if emit_log:
            self._emit_log_event(
                "PROGRAM_CONTEXT",
                (
                    f"rev={prev_text}->{curr_text} "
                    f"revs={revision_count} "
                    f"classes={active_class_count}"
                ),
                indent_level=1,
                max_len=260,
            )

    def _emit_program_context_transition_log(
        self,
        *,
        previous_version_id: Optional[str],
    ) -> None:
        prev_text = (
            str(previous_version_id).strip()
            if isinstance(previous_version_id, str) and previous_version_id.strip()
            else "-"
        )
        current_version = getattr(self.current_version, "version_id", None)
        curr_text = (
            str(current_version).strip()
            if isinstance(current_version, str) and current_version.strip()
            else "-"
        )
        revision_count = self._current_program_revision_count()
        active_class_count = self._current_active_class_count()
        self._emit_log_event(
            "PROGRAM_CONTEXT",
            (
                f"rev={prev_text}->{curr_text} "
                f"revs={revision_count} "
                f"classes={active_class_count}"
            ),
            indent_level=1,
            max_len=260,
        )

    def _commit_explained_transitions_to_explorer(
        self,
        transitions: List[Transition],
    ) -> None:
        committer = getattr(self.explorer, "commit_explained_transitions", None)
        if not callable(committer):
            return
        try:
            committer(
                transitions,
                assessments=self._build_explorer_commit_assessments(transitions),
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_plain_line(
                f"  [WARN] explorer.commit_explained_transitions failed: {exc}"
            )

    def _build_explorer_commit_assessments(
        self,
        transitions: List[Transition],
    ) -> Dict[str, Dict[str, Any]]:
        current_version_id = self._normalized_current_version_id()
        current_source_digest = self._current_source_digest()
        assessments: Dict[str, Dict[str, Any]] = {}
        for transition in transitions:
            transition_key = self._transition_key(transition)
            existing = self._current_source_transition_assessments.get(transition_key)
            assessment = (
                dict(existing)
                if self._assessment_matches_current_source(existing)
                else {}
            )
            leaf_group_id = self._protected_leaf_group_id_by_key.get(transition_key)
            class_id = (
                self._group_class_id_by_group_id.get(leaf_group_id)
                if isinstance(leaf_group_id, str)
                else None
            )
            if not isinstance(class_id, int) or int(class_id) <= 0:
                class_id = assessment.get("assigned_class_id")
            if not isinstance(class_id, int) or int(class_id) <= 0:
                class_id = 0
            if not isinstance(leaf_group_id, str) or not leaf_group_id:
                raw_group_id = assessment.get("assigned_group_id")
                leaf_group_id = (
                    raw_group_id.strip()
                    if isinstance(raw_group_id, str) and raw_group_id.strip()
                    else None
                )
            assessment.update(
                {
                    "transition_key": transition_key,
                    "current_version_id": current_version_id,
                    "current_source_digest": current_source_digest,
                    "current_explains": True,
                    "assigned_class_id": int(class_id),
                    "assigned_group_id": (
                        str(leaf_group_id)
                        if isinstance(leaf_group_id, str) and leaf_group_id
                        else None
                    ),
                    "assignment_status": (
                        "assigned"
                        if int(class_id) > 0
                        and isinstance(leaf_group_id, str)
                        and leaf_group_id
                        else "unknown"
                    ),
                }
            )
            assessments[transition_key] = assessment
        return assessments

    def _finalize_explained_transition_batch_in_explorer(
        self,
        transitions: List[Transition],
    ) -> None:
        finalizer = getattr(self.explorer, "finalize_explained_transition_batch", None)
        if not callable(finalizer):
            return
        finalize_result: Optional[Dict[str, Any]] = None
        finalize_error: Optional[Exception] = None
        train_started_at = perf_counter()
        train_failed = False
        self._emit_train_phase_start(transitions)
        train_progress_handler, close_train_progress = self._make_train_progress_handler()
        try:
            raw_result = finalizer(
                transitions,
                progress_callback=train_progress_handler,
            )
            finalize_result = (
                dict(raw_result)
                if isinstance(raw_result, dict)
                else None
            )
        except Exception as exc:  # noqa: BLE001
            train_failed = True
            finalize_error = exc
            self._emit_plain_line(
                f"  [ERROR] explorer.finalize_explained_transition_batch failed: {exc}"
            )
            self._emit_plain_line(traceback.format_exc().rstrip())
        finally:
            close_train_progress()
        self._emit_train_phase_end(
            finalize_result=finalize_result,
            elapsed_text=self._format_patch_elapsed(perf_counter() - train_started_at),
            failed=train_failed,
        )
        if finalize_error is not None:
            raise finalize_error
        self._save_explorer_training_artifacts_if_needed(finalize_result)

    def _defer_explained_transition_batch_in_explorer(
        self,
        transitions: List[Transition],
    ) -> None:
        if not transitions:
            return
        self._deferred_explained_transition_batch.extend(
            transition
            for transition in transitions
            if isinstance(transition, Transition)
        )

    def _request_deferred_explorer_context_sync(self) -> None:
        self._deferred_explorer_context_sync_needed = True

    def _flush_deferred_explorer_context_sync(self) -> bool:
        if not self._deferred_explorer_context_sync_needed:
            return False
        self._force_sync_explorer_program_context()
        return True

    def _flush_deferred_explained_transition_batch_in_explorer(self) -> int:
        if not self._deferred_explained_transition_batch:
            return 0
        deferred = list(self._deferred_explained_transition_batch)
        self._deferred_explained_transition_batch.clear()
        self._finalize_explained_transition_batch_in_explorer(deferred)
        return int(len(deferred))

    def _flush_deferred_explained_transition_batch_if_verify_batch_complete(
        self,
    ) -> int:
        self._sync_active_collect_batch_pending()
        if self._active_collect_batch_queue:
            return 0
        self._flush_deferred_explorer_context_sync()
        return self._flush_deferred_explained_transition_batch_in_explorer()

    def _save_explorer_training_artifacts_if_needed(
        self,
        finalize_result: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(finalize_result, dict):
            return
        raw_updates_completed = finalize_result.get("train_updates_completed")
        if not isinstance(raw_updates_completed, int) or int(raw_updates_completed) <= 0:
            return
        saver = getattr(self.explorer, "save_training_artifacts", None)
        if not callable(saver):
            return
        saved = saver(
            output_dir=self.output_dir,
            current_version_id=self._normalized_current_version_id(),
        )
        if not isinstance(saved, dict):
            return
        latest_path = saved.get("latest_checkpoint_path")
        artifact_parts: List[str] = []
        if isinstance(latest_path, str) and latest_path.strip():
            artifact_parts.append(f"latest_ckpt={Path(latest_path).name}")
        if artifact_parts:
            self._emit_log_event(
                "TRAIN_SAVE",
                " ".join(artifact_parts),
                indent_level=1,
                max_len=260,
            )

    def _snapshot_explorer_training_artifacts_for_current_version(self) -> None:
        snapshotter = getattr(
            self.explorer,
            "snapshot_training_artifacts_for_version",
            None,
        )
        if not callable(snapshotter):
            return
        saved = snapshotter(
            output_dir=self.output_dir,
            current_version_id=self._normalized_current_version_id(),
        )
        if not isinstance(saved, dict):
            return
        version_path = saved.get("version_checkpoint_path")
        if not isinstance(version_path, str) or not version_path.strip():
            return
        self._emit_log_event(
            "TRAIN_SNAPSHOT",
            f"version_ckpt={Path(version_path).name}",
            indent_level=1,
            max_len=260,
        )

    def _finalize_explained_transitions(
        self,
        transitions: List[Transition],
        *,
        commit_version_override: Optional[str] = None,
        refresh_canonical_groups: bool = True,
        defer_explorer_batch_finalize: bool = False,
        defer_explorer_context_sync: bool = False,
    ) -> int:
        finalized_transitions: List[Transition] = []
        finalized_transition_keys: List[str] = []
        previous_group_generation = int(self._group_context_generation)
        previous_active_class_ids = tuple(
            sorted(int(class_id) for class_id in self._leaf_group_id_by_class_id.keys())
        )
        for transition in transitions:
            transition_key = self._transition_key(transition)
            if transition_key not in self._protected_keys:
                self._add_to_protected(
                    transition,
                    commit_version_override=commit_version_override,
                )
            self._sync_canonical_group_assignment_from_protected(transition_key)
            finalized_transition_keys.append(transition_key)
            self._clear_formal_failure_attempt(transition_key)
            self._feedback_by_transition.pop(transition_key, None)
            finalized_transitions.append(transition)
        if not finalized_transitions:
            return 0
        self._mark_collect_batch_processed_many(finalized_transition_keys)
        current_active_class_ids = tuple(
            sorted(int(class_id) for class_id in self._leaf_group_id_by_class_id.keys())
        )
        group_structure_changed = (
            int(self._group_context_generation) != previous_group_generation
            or current_active_class_ids != previous_active_class_ids
        )
        should_sync_before_commit = (
            not defer_explorer_context_sync or group_structure_changed
        )
        if should_sync_before_commit:
            self._sync_explorer_program_context(
                emit_log=False,
            )
        self._commit_explained_transitions_to_explorer(finalized_transitions)
        if refresh_canonical_groups:
            if defer_explorer_context_sync:
                if group_structure_changed:
                    self._request_deferred_explorer_context_sync()
            elif group_structure_changed:
                self._force_sync_explorer_program_context()
        elif defer_explorer_context_sync:
            self._request_deferred_explorer_context_sync()
        if defer_explorer_batch_finalize:
            self._defer_explained_transition_batch_in_explorer(finalized_transitions)
        else:
            self._finalize_explained_transition_batch_in_explorer(finalized_transitions)
        return len(finalized_transitions)

    def _build_program_context_payload(self, *, versions: List[Any]) -> Dict[str, Any]:
        snapshot = self._build_program_context_snapshot(versions=versions)
        return self._build_program_context_payload_from_snapshot(snapshot=snapshot)

    def _build_program_context_payload_from_snapshot(
        self,
        *,
        snapshot: ProgramContextSnapshot,
    ) -> Dict[str, Any]:
        group_context = dict(snapshot.group_context)
        group_context.update(
            self._build_canonical_group_assignment_payload(snapshot=snapshot)
        )
        return {
            "current_version_id": snapshot.current_version_id,
            "current_source": self.current_source,
            "versions": snapshot.version_rows,
            "group_context": group_context,
        }

    def _normalized_current_version_id(self) -> Optional[str]:
        current_version_id = getattr(self.current_version, "version_id", None)
        if not isinstance(current_version_id, str):
            return None
        normalized = str(current_version_id).strip()
        return normalized or None

    def _build_version_rows(self, versions: List[Any]) -> List[Dict[str, Any]]:
        return [
            {
                "version_id": getattr(version, "version_id", None),
                "source": getattr(version, "source", None),
                "index": getattr(version, "index", None),
            }
            for version in versions
        ]

    def _build_program_context_snapshot(
        self,
        *,
        versions: Optional[List[Any]] = None,
    ) -> ProgramContextSnapshot:
        resolved_versions = (
            versions if versions is not None else self.repository.list_versions()
        )
        version_rows = self._build_version_rows(resolved_versions)
        self._sanitize_protected_group_registry()
        current_version_id = self._normalized_current_version_id()
        group_context = self._build_group_context_payload()
        payload_signature = self._program_context_payload_signature(
            versions=resolved_versions
        )
        canonical_context_signature = self._canonical_assignment_context_signature_for(
            current_version_id=current_version_id,
            version_rows=version_rows,
            group_context=group_context,
        )
        return ProgramContextSnapshot(
            versions=resolved_versions,
            version_rows=version_rows,
            current_version_id=current_version_id,
            group_context=group_context,
            payload_signature=payload_signature,
            canonical_context_signature=canonical_context_signature,
        )

    def _program_context_payload_signature(
        self,
        *,
        versions: List[Any],
    ) -> Tuple[Any, ...]:
        version_ids = tuple(
            str(getattr(version, "version_id", "")).strip()
            for version in versions
            if isinstance(getattr(version, "version_id", None), str)
            and str(getattr(version, "version_id", None)).strip()
        )
        current_version = getattr(self.current_version, "version_id", None)
        current_version_text = (
            str(current_version).strip() if isinstance(current_version, str) and str(current_version).strip() else None
        )
        current_source_digest = self._current_source_digest()
        return (
            current_version_text,
            current_source_digest,
            version_ids,
            int(self._group_context_generation),
        )

    def _mark_canonical_dataset_changed(self) -> None:
        self._canonical_dataset_generation += 1

    def _build_group_context_payload(self) -> Dict[str, Any]:
        if not self._uses_dynamics_classes():
            return self._build_no_class_singleton_group_context_payload()

        group_rows: List[Dict[str, Any]] = []
        for group_id in sorted(self._protected_group_metadata_by_id.keys()):
            group = self._protected_group_metadata_by_id[group_id]
            split_program_source = self._load_split_program_source_for_group(group_id)
            group_rows.append(
                {
                    "group_id": group_id,
                    "commit_version": group.get("commit_version"),
                    "parent_group_id": group.get("parent_group_id"),
                    "child_group_ids": list(group.get("child_group_ids") or []),
                    "member_transition_count": int(
                        max(
                            len(group.get("member_transition_keys") or []),
                            int(group.get("restored_member_transition_count") or 0),
                        )
                    ),
                    "class_id": (
                        int(group.get("class_id"))
                        if isinstance(group.get("class_id"), int)
                        and int(group.get("class_id")) > 0
                        else None
                    ),
                    "retired_class_id": (
                        int(group.get("retired_class_id"))
                        if isinstance(group.get("retired_class_id"), int)
                        and int(group.get("retired_class_id")) > 0
                        else None
                    ),
                    "split_broken_child_group_id": group.get(
                        "split_broken_child_group_id"
                    ),
                    "split_kept_child_group_id": group.get(
                        "split_kept_child_group_id"
                    ),
                    "split_program_source": (
                        split_program_source
                        if isinstance(split_program_source, str)
                        and split_program_source.strip()
                        else None
                    ),
                }
            )
        active_class_ids = tuple(sorted(self._leaf_group_id_by_class_id.keys()))
        active_leaf_group_ids = tuple(
            self._leaf_group_id_by_class_id[class_id] for class_id in active_class_ids
        )
        return {
            "mode": self.dynamics_class_mode,
            "generation": int(self._group_context_generation),
            "root_group_id_by_commit_version": dict(
                sorted(self._protected_root_group_id_by_commit_version.items())
            ),
            "transition_leaf_group_ids": dict(
                sorted(self._protected_leaf_group_id_by_key.items())
            ),
            "leaf_group_class_ids": dict(
                sorted(
                    (str(group_id), int(class_id))
                    for group_id, class_id in self._group_class_id_by_group_id.items()
                )
            ),
            "active_class_ids": [int(class_id) for class_id in active_class_ids],
            "active_leaf_group_ids": [str(group_id) for group_id in active_leaf_group_ids],
            "group_hierarchy": group_rows,
        }

    def _empty_group_context_payload(self) -> Dict[str, Any]:
        return {
            "mode": self.dynamics_class_mode,
            "generation": int(self._group_context_generation),
            "root_group_id_by_commit_version": {},
            "transition_leaf_group_ids": {},
            "leaf_group_class_ids": {},
            "active_class_ids": [],
            "active_leaf_group_ids": [],
            "group_hierarchy": [],
        }

    def _no_class_singleton_group_id(self) -> str:
        return "__no_class_singleton__"

    def _no_class_singleton_root_versions(self) -> Tuple[str, ...]:
        version_ids: List[str] = []
        repository = getattr(self, "repository", None)
        list_versions = getattr(repository, "list_versions", None)
        if not callable(list_versions):
            return ()
        for version in list_versions():
            version_id = self._normalize_commit_version(
                getattr(version, "version_id", None)
            )
            if not self._is_class_eligible_commit_version(version_id):
                continue
            version_ids.append(version_id)
        return tuple(dict.fromkeys(version_ids))

    def _no_class_singleton_context_is_active(self) -> bool:
        if self._uses_dynamics_classes():
            return False
        if not getattr(self, "protected_set", None):
            return False
        return bool(self._no_class_singleton_root_versions())

    def _build_no_class_singleton_group_context_payload(self) -> Dict[str, Any]:
        if not self._no_class_singleton_context_is_active():
            return self._empty_group_context_payload()

        group_id = self._no_class_singleton_group_id()
        root_versions = self._no_class_singleton_root_versions()
        commit_version = root_versions[-1]
        protected_keys = sorted(str(key) for key in self._protected_keys)
        protected_count = max(len(protected_keys), len(self.protected_set))
        return {
            "mode": self.dynamics_class_mode,
            "generation": int(self._group_context_generation),
            "root_group_id_by_commit_version": {
                str(version_id): group_id for version_id in root_versions
            },
            "transition_leaf_group_ids": {
                str(transition_key): group_id for transition_key in protected_keys
            },
            "leaf_group_class_ids": {group_id: 1},
            "active_class_ids": [1],
            "active_leaf_group_ids": [group_id],
            "canonical_class_counts": {1: int(protected_count)},
            "canonical_leaf_group_counts": {group_id: int(protected_count)},
            "canonical_classified_count": int(protected_count),
            "canonical_unassigned_count": max(
                0,
                int(len(self.canonical_d)) - int(protected_count),
            ),
            "group_hierarchy": [
                {
                    "group_id": group_id,
                    "commit_version": commit_version,
                    "parent_group_id": None,
                    "child_group_ids": [],
                    "member_transition_count": int(protected_count),
                    "class_id": 1,
                    "retired_class_id": None,
                    "split_broken_child_group_id": None,
                    "split_kept_child_group_id": None,
                    "split_program_source": None,
                    "is_no_class_singleton": True,
                }
            ],
        }

    def _canonical_assignment_context_signature_for(
        self,
        *,
        current_version_id: Optional[str],
        version_rows: List[Dict[str, Any]],
        group_context: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        return (
            int(group_context.get("generation"))
            if isinstance(group_context.get("generation"), int)
            else 0,
            tuple(
                str(row.get("version_id"))
                for row in version_rows
                if isinstance(row.get("version_id"), str)
                and str(row.get("version_id")).strip()
            ),
            (
                str(current_version_id).strip()
                if isinstance(current_version_id, str) and str(current_version_id).strip()
                else None
            ),
        )

    def _refresh_canonical_group_assignments_from_cached_assessments(
        self,
        *,
        transitions: Optional[List[Transition]] = None,
        removed_transitions: Optional[List[Transition]] = None,
        clear_existing: bool = False,
        allow_classification: bool = True,
        ignore_cached_assignments: bool = False,
        snapshot: Optional[ProgramContextSnapshot] = None,
    ) -> None:
        canonical_dataset = self.canonical_d.buffer
        source_dataset = transitions if transitions is not None else canonical_dataset
        current_transition_keys = (
            set(self.canonical_d.transition_keys()) if transitions is None else None
        )
        removed_transition_keys = (
            {self._transition_key(transition) for transition in removed_transitions}
            if removed_transitions
            else None
        )
        self._prepare_canonical_assignment_maps(
            snapshot=snapshot,
            current_transition_keys=current_transition_keys,
            removed_transition_keys=removed_transition_keys,
            clear_existing=clear_existing,
        )

        assessments = self._ensure_current_source_transition_assessments(
            list(source_dataset)
        )
        pending_classifications: List[Tuple[str, Transition, Dict[str, Any]]] = []
        for transition in source_dataset:
            transition_key = self._transition_key(transition)
            assessment = assessments.get(transition_key)
            if not self._assessment_matches_current_source(assessment):
                self._clear_canonical_group_assignment(transition_key)
                continue
            if assessment.get("current_explains") is not True:
                self._clear_canonical_group_assignment(transition_key)
                continue
            cached_assignment = (
                None
                if ignore_cached_assignments
                else self._current_assessment_group_assignment(assessment)
            )
            if cached_assignment is not None:
                class_id, group_id = cached_assignment
                self._set_canonical_group_assignment(
                    transition_key,
                    class_id=int(class_id),
                    group_id=str(group_id),
                )
                continue
            if allow_classification:
                pending_classifications.append(
                    (transition_key, transition, dict(assessment))
                )
            else:
                self._clear_canonical_group_assignment(transition_key)

        if pending_classifications:
            resolved_snapshot = snapshot or self._build_program_context_snapshot()
            self._prepare_canonical_assignment_maps(
                snapshot=resolved_snapshot,
                current_transition_keys=None,
                removed_transition_keys=None,
                clear_existing=False,
            )
            update_result = self._context_group_classifier.sync_context(
                context={
                    "current_version_id": resolved_snapshot.current_version_id,
                    "current_source": self.current_source,
                    "versions": resolved_snapshot.version_rows,
                    "group_context": resolved_snapshot.group_context,
                },
                max_program_count=max(1, int(len(resolved_snapshot.version_rows))),
                previous_snapshot=self._context_group_snapshot,
            )
            self._context_group_snapshot = update_result.snapshot
            assignments = self._context_group_classifier.classify_transitions(
                transitions=[
                    transition
                    for _transition_key, transition, _assessment in pending_classifications
                ],
                include_rows=False,
            )
            for (
                transition_key,
                _transition,
                assessment,
            ), (assignment, _rows) in zip(pending_classifications, assignments):
                if (
                    isinstance(assignment.class_id, int)
                    and int(assignment.class_id) > 0
                    and isinstance(assignment.group_id, str)
                    and assignment.group_id
                ):
                    updated_assessment = dict(assessment)
                    updated_assessment["assigned_class_id"] = int(assignment.class_id)
                    updated_assessment["assigned_group_id"] = str(assignment.group_id)
                    updated_assessment["assignment_status"] = (
                        str(assignment.status)
                        if isinstance(getattr(assignment, "status", None), str)
                        else "assigned"
                    )
                    self._current_source_transition_assessments[transition_key] = dict(
                        updated_assessment
                    )
                    assessments[transition_key] = dict(updated_assessment)
                    self._set_canonical_group_assignment(
                        transition_key,
                        class_id=int(assignment.class_id),
                        group_id=str(assignment.group_id),
                    )
                    continue
                updated_assessment = dict(assessment)
                updated_assessment["assigned_class_id"] = 0
                updated_assessment["assigned_group_id"] = None
                updated_assessment["assignment_status"] = (
                    str(getattr(assignment, "status"))
                    if isinstance(getattr(assignment, "status", None), str)
                    else "unknown"
                )
                self._current_source_transition_assessments[transition_key] = dict(
                    updated_assessment
                )
                assessments[transition_key] = dict(updated_assessment)
                self._clear_canonical_group_assignment(transition_key)

    def _refresh_canonical_group_assignments_for_split_events(
        self,
        split_events: List[Dict[str, Any]],
    ) -> None:
        affected_group_ids = {
            str(event.get("split_group_id")).strip()
            for event in split_events
            if isinstance(event, dict)
            and isinstance(event.get("split_group_id"), str)
            and str(event.get("split_group_id")).strip()
        }
        if not affected_group_ids:
            return
        affected_keys: Set[str] = set()
        for group_id in affected_group_ids:
            affected_keys.update(
                str(key)
                for key in self._canonical_transition_keys_by_leaf_group_id.get(
                    group_id,
                    set(),
                )
                if isinstance(key, str) and key
            )
        if not affected_keys:
            return

        affected_transitions: List[Transition] = []
        for transition_key in sorted(affected_keys):
            self._clear_cached_assessment_group_assignment(transition_key)
            transition = self.canonical_d.transition_for_key(transition_key)
            if isinstance(transition, Transition):
                affected_transitions.append(transition)
            else:
                self._clear_canonical_group_assignment(transition_key)

        if not affected_transitions:
            return
        self._refresh_canonical_group_assignments_from_cached_assessments(
            transitions=affected_transitions,
            allow_classification=True,
            ignore_cached_assignments=True,
        )

    def _clear_cached_assessment_group_assignment(self, transition_key: str) -> None:
        safe_key = str(transition_key).strip()
        if not safe_key:
            return
        assessment = self._current_source_transition_assessments.get(safe_key)
        if not isinstance(assessment, dict):
            return
        updated_assessment = dict(assessment)
        updated_assessment.pop("assigned_class_id", None)
        updated_assessment.pop("assigned_group_id", None)
        updated_assessment.pop("assignment_status", None)
        self._current_source_transition_assessments[safe_key] = updated_assessment

    def _set_canonical_group_assignment(
        self,
        transition_key: str,
        *,
        class_id: int,
        group_id: str,
    ) -> None:
        safe_key = str(transition_key)
        safe_class_id = int(class_id)
        safe_group_id = str(group_id)
        old_class_id = self._canonical_class_id_by_key.get(safe_key)
        old_group_id = self._canonical_leaf_group_id_by_key.get(safe_key)
        if old_class_id == safe_class_id and old_group_id == safe_group_id:
            return
        if old_class_id is None and old_group_id is None:
            self._canonical_class_id_by_key[safe_key] = safe_class_id
            self._canonical_leaf_group_id_by_key[safe_key] = safe_group_id
            self._canonical_class_counts[safe_class_id] = (
                int(self._canonical_class_counts.get(safe_class_id, 0)) + 1
            )
            self._canonical_leaf_group_counts[safe_group_id] = (
                int(self._canonical_leaf_group_counts.get(safe_group_id, 0)) + 1
            )
            self._canonical_transition_keys_by_leaf_group_id.setdefault(
                safe_group_id,
                set(),
            ).add(safe_key)
            self._canonical_classified_count += 1
            return
        self._clear_canonical_group_assignment(safe_key)
        self._canonical_class_id_by_key[safe_key] = safe_class_id
        self._canonical_leaf_group_id_by_key[safe_key] = safe_group_id
        self._canonical_class_counts[safe_class_id] = (
            int(self._canonical_class_counts.get(safe_class_id, 0)) + 1
        )
        self._canonical_leaf_group_counts[safe_group_id] = (
            int(self._canonical_leaf_group_counts.get(safe_group_id, 0)) + 1
        )
        self._canonical_transition_keys_by_leaf_group_id.setdefault(
            safe_group_id,
            set(),
        ).add(safe_key)
        self._canonical_classified_count += 1

    def _clear_canonical_group_assignment(self, transition_key: str) -> None:
        safe_key = str(transition_key)
        old_class_id = self._canonical_class_id_by_key.pop(safe_key, None)
        old_group_id = self._canonical_leaf_group_id_by_key.pop(safe_key, None)
        if isinstance(old_class_id, int):
            next_count = int(self._canonical_class_counts.get(old_class_id, 0)) - 1
            if next_count > 0:
                self._canonical_class_counts[old_class_id] = next_count
            else:
                self._canonical_class_counts.pop(old_class_id, None)
        if isinstance(old_group_id, str):
            old_group_keys = self._canonical_transition_keys_by_leaf_group_id.get(
                old_group_id
            )
            if old_group_keys is not None:
                old_group_keys.discard(safe_key)
                if not old_group_keys:
                    self._canonical_transition_keys_by_leaf_group_id.pop(
                        old_group_id,
                        None,
                    )
            next_count = int(self._canonical_leaf_group_counts.get(old_group_id, 0)) - 1
            if next_count > 0:
                self._canonical_leaf_group_counts[old_group_id] = next_count
            else:
                self._canonical_leaf_group_counts.pop(old_group_id, None)
        if isinstance(old_class_id, int) or isinstance(old_group_id, str):
            self._canonical_classified_count = max(
                0,
                int(self._canonical_classified_count) - 1,
            )

    def _current_assessment_group_assignment(
        self,
        assessment: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, str]]:
        if not self._assessment_matches_current_source(assessment):
            return None
        if assessment.get("current_explains") is not True:
            return None
        raw_class_id = assessment.get("assigned_class_id")
        if not isinstance(raw_class_id, int) or int(raw_class_id) <= 0:
            return None
        raw_group_id = assessment.get("assigned_group_id")
        if not isinstance(raw_group_id, str) or not raw_group_id.strip():
            return None
        return int(raw_class_id), raw_group_id.strip()

    def _force_sync_explorer_program_context(self) -> None:
        self._last_program_context_payload_signature = None
        self._sync_explorer_program_context(emit_log=False)

    def prepare_final_dashboard_snapshot(self) -> None:
        self._force_sync_explorer_program_context()
        finalizer = getattr(self.explorer, "prepare_final_dashboard_snapshot", None)
        if callable(finalizer):
            finalizer()

    def _build_canonical_group_assignment_payload(
        self,
        *,
        snapshot: ProgramContextSnapshot,
    ) -> Dict[str, Any]:
        if not isinstance(snapshot.group_context, dict):
            return {
                "canonical_class_counts": {},
                "canonical_leaf_group_counts": {},
                "canonical_classified_count": 0,
                "canonical_unassigned_count": 0,
            }

        self._prepare_canonical_assignment_maps(
            snapshot=snapshot,
            current_transition_keys=None,
            removed_transition_keys=None,
            clear_existing=False,
        )
        canonical_class_counts = dict(self._canonical_class_counts)
        canonical_leaf_group_counts = dict(self._canonical_leaf_group_counts)
        canonical_classified_count = int(self._canonical_classified_count)
        canonical_unassigned_count = max(
            0,
            int(len(self.canonical_d) - canonical_classified_count),
        )
        return {
            "canonical_class_counts": {
                str(class_id): int(count)
                for class_id, count in sorted(canonical_class_counts.items())
            },
            "canonical_leaf_group_counts": dict(
                sorted(canonical_leaf_group_counts.items())
            ),
            "canonical_classified_count": int(canonical_classified_count),
            "canonical_unassigned_count": int(canonical_unassigned_count),
        }

    def _build_canonical_assignment_stats_payload(self) -> Dict[str, Any]:
        canonical_classified_count = int(self._canonical_classified_count)
        canonical_unassigned_count = max(
            0,
            int(len(self.canonical_d) - canonical_classified_count),
        )
        return {
            "canonical_class_counts": {
                str(int(class_id)): int(count)
                for class_id, count in sorted(self._canonical_class_counts.items())
                if int(class_id) > 0 and int(count) >= 0
            },
            "canonical_leaf_group_counts": dict(
                sorted(self._canonical_leaf_group_counts.items())
            ),
            "canonical_classified_count": int(canonical_classified_count),
            "canonical_unassigned_count": int(canonical_unassigned_count),
        }

    def _publish_collection_feedback(self, *, added_count: int) -> None:
        explorer_feedback = getattr(self.explorer, "set_collection_feedback", None)
        if not callable(explorer_feedback):
            return
        payload = {
            "added_count": int(added_count),
            "dataset_size": int(len(self.canonical_d)),
            "pending_count": int(max(0, len(self.canonical_d) - len(self.protected_set))),
            **self._build_canonical_assignment_stats_payload(),
        }
        try:
            explorer_feedback(payload)
        except Exception:
            pass

    def _prepare_canonical_assignment_maps(
        self,
        *,
        snapshot: Optional[ProgramContextSnapshot],
        current_transition_keys: Optional[set[str]] = None,
        removed_transition_keys: Optional[set[str]] = None,
        clear_existing: bool,
    ) -> None:
        if clear_existing:
            self._canonical_leaf_group_id_by_key = {}
            self._canonical_class_id_by_key = {}
            self._canonical_class_counts = {}
            self._canonical_leaf_group_counts = {}
            self._canonical_transition_keys_by_leaf_group_id = {}
            self._canonical_classified_count = 0

        if (
            snapshot is not None
            and snapshot.canonical_context_signature
            != self._canonical_assignment_context_signature
        ):
            self._canonical_assignment_context_signature = snapshot.canonical_context_signature

        if current_transition_keys is not None:
            for transition_key in list(self._canonical_leaf_group_id_by_key.keys()):
                if transition_key not in current_transition_keys:
                    self._clear_canonical_group_assignment(transition_key)
        if removed_transition_keys:
            for transition_key in removed_transition_keys:
                self._clear_canonical_group_assignment(transition_key)

    def run(self, max_iterations: Optional[int] = None) -> str:
        self._data_index_by_iteration_and_key.clear()
        self._next_data_index_by_iteration.clear()
        self._failure_index_by_iteration_and_key.clear()
        self._active_collect_batch_queue.clear()
        self._deferred_explained_transition_batch.clear()
        self._deferred_explorer_context_sync_needed = False
        self._program_started_at = perf_counter()
        self._current_iteration_started_at = None
        self._current_collect_started_at = None
        self._current_verify_started_at = None

        print("Starting Program Discovery Pipeline (incremental CEGIS)")
        if max_iterations is None:
            print("Max verify cycles: unlimited (epoch-budget mode)")
        else:
            print(f"Max verify cycles: {max_iterations}")
        collect_plan = self._resolve_collect_plan()
        print(self._format_collect_schedule_line(collect_plan))
        print(
            "New transition images: "
            + ("enabled" if self.save_new_transition_images else "disabled")
        )
        if self.max_total_llm_calls is None:
            print("LLM call budget: disabled")
        else:
            print(f"LLM call budget: {self.max_total_llm_calls}")
        print("-" * 50)

        epoch_index = int(self._epoch_index)
        epoch_budget_size = int(self._epoch_budget_size)
        epoch_budget_remaining = int(self._epoch_budget_remaining)
        self._last_emitted_cli_header_iteration = None
        self._clear_requested_termination()
        self._last_termination_reason = None
        self._last_termination_detail = None
        termination_reason = "manual"
        verify_pending_progress_bar: Optional[Any] = None

        try:
            while True:
                if self._should_stop_for_saturation():
                    termination_reason = "saturation"
                    break
                if self._has_reached_llm_call_limit():
                    termination_reason = "llm_call_limit"
                    break
                collect_result = self._collect_round(
                    max_iterations=max_iterations,
                    collect_plan=collect_plan,
                )
                if collect_result.termination_reason is not None:
                    termination_reason = collect_result.termination_reason
                    break
                should_collect = bool(collect_result.should_collect)
                collected = list(collect_result.collected)
                added = int(collect_result.added)
                collect_diag = dict(collect_result.collect_diag)

                if should_collect:
                    batch_pending_count = self._count_active_collect_batch_pending()
                    if added > 0:
                        epoch_index += 1
                        epoch_budget_size = batch_pending_count
                        epoch_budget_remaining = batch_pending_count
                        self.no_new_canonical_rounds = 0
                    else:
                        epoch_budget_size = 0
                        epoch_budget_remaining = 0
                        self.no_new_canonical_rounds += 1
                self._emit_collect_phase_end(
                    should_collect=should_collect,
                    collect_diag=collect_diag,
                    collected_count=len(collected),
                    added_count=added,
                    pending_count=self._count_active_collect_batch_pending(),
                )
                if should_collect and epoch_budget_size > 0:
                    self._emit_verify_phase_start(
                        epoch_index=epoch_index,
                        pending_count=epoch_budget_size,
                    )
                target = self._select_next_unprotected()
                eval_count = len(self.protected_set) + (1 if target is not None else 0)
                current_accuracy = 1.0
                phase = "collect"
                failure_count = 0
                candidate_count = 0
                accepted = False
                delta_acc = 0.0
                best_acc = current_accuracy
                regressions = 0
                selected_patch_digest = None
                decision_reason = ""
                target_action = target.action if target else None
                target_data_index: Optional[int] = None
                target_step_index: Optional[int] = None
                target_failure_index: Optional[int] = None

                verify_pending_progress_bar = self._sync_verify_pending_batch_progress_bar(
                    verify_pending_progress_bar,
                    total=epoch_budget_size,
                    remaining=self._count_active_collect_batch_pending(),
                    phase="verify",
                    current_action=target_action,
                )

                if target is None:
                    self._emit_log_event("VERIFY", "not a new canon data", max_len=260)
                    phase = "no_target"
                    if self._requested_termination_reason == "saturation":
                        decision_reason = (
                            self._requested_termination_detail
                            or self._format_saturation_termination_message()
                        )
                    elif added == 0 and self._is_protected_complete():
                        decision_reason = (
                            "No pending transition and no new canonical transitions "
                            f"collected for {self.no_new_canonical_rounds} "
                            "collection rounds; retrying collection."
                        )
                    else:
                        decision_reason = "No pending transition outside protected set."
                else:
                    target_key = self._transition_key(target)
                    target_data_index = self._resolve_attempt_data_index(
                        iteration=self.iteration,
                        transition_key=target_key,
                    )
                    target_step_index = self._resolve_transition_step_index(
                        transition_key=target_key,
                        fallback_index=target_data_index,
                    )
                    target_failure_serial = self._next_failure_index(
                        iteration=self.iteration,
                        transition_key=target_key,
                    )
                    target_failure_attempt = self._next_formal_failure_attempt(
                        target_key
                    )
                    attempt_retry_fields = self._format_attempt_retry_fields(
                        target_failure_attempt
                    )
                    target_eval = self._evaluate_current_source_target(target)
                    had_prior_feedback = target_key in self._feedback_by_transition
                    target_predicted_canonical = None
                    target_prediction_error = None
                    if target_eval.records:
                        target_record = target_eval.records[0]
                        target_predicted_canonical = target_record.predicted_canonical
                        if target_record.error is not None:
                            error_rows = self._serialize_smoke_errors([target_record.error])
                            if error_rows:
                                target_prediction_error = error_rows[0]
                    target_solved_by_current = self._is_target_solved(target_eval)
                    current_accuracy = self._compute_iteration_accuracy(
                        protected_correct=len(self.protected_set),
                        protected_total=len(self.protected_set),
                        target_correct=1 if target_solved_by_current else 0,
                        target_total=1,
                    )
                    best_acc = current_accuracy
                    unexplained_transition_image_paths: Optional[Dict[str, str]] = None

                    if target_solved_by_current:
                        if had_prior_feedback:
                            self._save_target_transition_images(
                                transition=target,
                                iteration=self.iteration,
                                data_index=target_data_index or 1,
                                failure_index=0,
                                patch_digest=getattr(self.current_version, "version_id", None),
                                step_index=target_step_index,
                                predicted_next_state_json=target_predicted_canonical,
                                target_prediction_error=target_prediction_error,
                                stem_tag="explained",
                            )
                        promotion_dataset = [target]
                        if should_collect and added > 0:
                            batch_transitions = (
                                self._active_collect_batch_unprotected_transitions()
                            )
                            if batch_transitions:
                                promotion_dataset = batch_transitions
                        finalized_count = self._promote_solved_pending_transitions(
                            promotion_dataset,
                            defer_explorer_batch_finalize=True,
                            defer_explorer_context_sync=True,
                        )
                        phase = "protected_add"
                        decision_reason = (
                            "Current program solved target transition; "
                            f"added {max(1, int(finalized_count))} to protected set."
                        )
                        epoch_budget_remaining = self._count_active_collect_batch_pending()
                        verify_pending_progress_bar = self._sync_verify_pending_batch_progress_bar(
                            verify_pending_progress_bar,
                            total=epoch_budget_size,
                            remaining=epoch_budget_remaining,
                            phase="verify",
                            current_action=None,
                        )
                        if epoch_budget_remaining <= 0:
                            self._emit_verify_phase_end(
                                epoch_index=epoch_index,
                                verified_count=epoch_budget_size,
                                total_count=epoch_budget_size,
                            )
                            self._flush_deferred_explained_transition_batch_if_verify_batch_complete()
                        self._epoch_index = int(epoch_index)
                        self._epoch_budget_size = int(epoch_budget_size)
                        self._epoch_budget_remaining = int(epoch_budget_remaining)
                        explorer_diagnostics = self._collect_explorer_diagnostics()
                        self._log_iteration(
                            phase=phase,
                            collected_count=len(collected),
                            added_count=added,
                            eval_count=eval_count,
                            failure_count=failure_count,
                            candidate_count=candidate_count,
                            accepted=accepted,
                            delta_acc=delta_acc,
                            current_acc=current_accuracy,
                            best_acc=best_acc,
                            regressions=regressions,
                            decision_reason=decision_reason,
                            patch_digest=selected_patch_digest,
                            target_action=target_action,
                            epoch_index=epoch_index,
                            epoch_budget_size=epoch_budget_size,
                            epoch_budget_remaining=epoch_budget_remaining,
                            no_new_canonical_rounds=self.no_new_canonical_rounds,
                            max_no_new_canonical_rounds=self.max_no_new_canonical_rounds,
                            explorer_diagnostics=explorer_diagnostics,
                        )
                        self._maybe_emit_iteration_elapsed_line(
                            pending_remaining=epoch_budget_remaining
                        )
                        continue
                    else:
                        patch_elapsed_text: Optional[str] = None
                        max_unexpected_attempts = int(
                            self.patch_unexpected_error_max_attempts
                        )
                        failure_count = 0
                        target_failure_case = self._build_failure_case(target_eval, target)
                        unexplained_transition_image_paths = self._save_target_transition_images(
                            transition=target,
                            iteration=self.iteration,
                            data_index=target_data_index or 1,
                            failure_index=target_failure_serial or 1,
                            patch_digest=getattr(self.current_version, "version_id", None),
                            step_index=target_step_index,
                            predicted_next_state_json=target_predicted_canonical,
                            target_prediction_error=target_prediction_error,
                            stem_tag="unexplained",
                        )
                        prior_feedback = self._feedback_by_transition.get(target_key, {})
                        formal_failure_index = target_failure_attempt or 1
                        artifact_failure_index = target_failure_serial or 1
                        max_formal_attempts = int(self.max_patch_attempts_per_target)
                        emit_patch_header = True

                        while True:
                            attempt_retry_fields = self._format_attempt_retry_fields(
                                formal_failure_index
                            )
                            patch_retry_progress = self._format_patch_retry_progress(
                                generator_try=1,
                                unexpected_retry=1,
                                unexpected_retry_total=max_unexpected_attempts,
                            )
                            self._update_dashboard_progress_phase(
                                "patch",
                                patch_target={
                                    "step_index": target_step_index,
                                    "data_index": target_data_index,
                                    "failure_index": artifact_failure_index,
                                    "current_version": getattr(
                                        self.current_version,
                                        "version_id",
                                        None,
                                    ),
                                    "action": target_action,
                                },
                            )
                            verify_pending_progress_bar = self._sync_verify_pending_batch_progress_bar(
                                verify_pending_progress_bar,
                                total=epoch_budget_size,
                                remaining=self._count_active_collect_batch_pending(),
                                phase="patch",
                                current_action=target_action,
                            )
                            if emit_patch_header:
                                patch_header = self._format_verify_pending_patch_header(
                                    total=epoch_budget_size,
                                    remaining=self._count_active_collect_batch_pending(),
                                    current_action=target_action,
                                )
                                self._emit_group_header_line(
                                    "LLM",
                                    f"[PATCH  ] {patch_header} acc={current_accuracy:.4f}",
                                    max_len=260,
                                )
                                self._emit_llm_runtime_status()
                                self._emit_log_event(
                                    "PATCH",
                                    f"{attempt_retry_fields} {patch_retry_progress}",
                                    indent_level=1,
                                )
                            generated: Optional[Any] = None
                            applied: Optional[Any] = None
                            selected_patch_attempt_artifact: Optional[Dict[str, Any]] = None
                            unexpected_attempt = 0

                            while generated is None or applied is None:
                                if unexpected_attempt > 0:
                                    unexpected_retry_progress = self._format_patch_retry_progress(
                                        generator_try=1,
                                        unexpected_retry=unexpected_attempt,
                                        unexpected_retry_total=max_unexpected_attempts,
                                    )
                                    self._emit_log_event(
                                        "RETRY",
                                        f"{attempt_retry_fields} {unexpected_retry_progress}",
                                        indent_level=1,
                                        max_len=260,
                                    )
                                unexpected_attempt += 1
                                patch_start_time, patch_stop_event, patch_thread = (
                                    self._start_patch_elapsed_logger()
                                )
                                loop_generated: Optional[Any] = None
                                patch_attempt_dir = self._prepare_patch_attempt_artifact_dir(
                                    iteration=self.iteration,
                                    data_index=target_data_index or 1,
                                    failure_index=artifact_failure_index,
                                    unexpected_attempt=unexpected_attempt,
                                )
                                patch_attempt_summary_path = (
                                    patch_attempt_dir / "attempt_summary.json"
                                )
                                patch_attempt_artifact: Dict[str, Any] = {
                                    "iteration": int(self.iteration),
                                    "data_index": int(target_data_index or 1),
                                    "failure_index": int(artifact_failure_index),
                                    "unexpected_attempt": int(unexpected_attempt),
                                    "target_transition_key": str(target_key),
                                    "current_version": getattr(
                                        self.current_version,
                                        "version_id",
                                        None,
                                    ),
                                    "patch_format": getattr(self, "patch_format", None),
                                    "artifact_dir": self._relative_output_path(
                                        patch_attempt_dir
                                    ),
                                    "summary_path": self._relative_output_path(
                                        patch_attempt_summary_path
                                    ),
                                    "generator_attempts": [],
                                    "status": "started",
                                }

                                def _persist_patch_attempt_artifact() -> None:
                                    self._write_json_artifact(
                                        patch_attempt_summary_path,
                                        patch_attempt_artifact,
                                    )

                                def _record_generation_artifact(
                                    generator_try: int,
                                    total_attempts: int,
                                    prompt: str,
                                    response_text: Optional[str],
                                    reasoning: Optional[str],
                                    error: Optional[str],
                                ) -> None:
                                    patch_attempt_artifact["generator_attempts"].append(
                                        self._save_patch_generation_exchange_artifacts(
                                            artifact_dir=patch_attempt_dir,
                                            generator_try=generator_try,
                                            total_attempts=total_attempts,
                                            prompt=prompt,
                                            response_text=response_text,
                                            reasoning=reasoning,
                                            error=error,
                                        )
                                    )
                                    _persist_patch_attempt_artifact()

                                def _emit_generator_attempt_progress(
                                    generator_try: int,
                                    total_attempts: int,
                                ) -> None:
                                    if generator_try <= 1:
                                        return
                                    generator_retry_progress = self._format_patch_retry_progress(
                                        generator_try=generator_try,
                                        total_generator_attempts=total_attempts,
                                        unexpected_retry=unexpected_attempt,
                                        unexpected_retry_total=max_unexpected_attempts,
                                    )
                                    self._emit_log_event(
                                        "RETRY",
                                        f"{attempt_retry_fields} {generator_retry_progress}",
                                        indent_level=1,
                                        max_len=260,
                                    )

                                try:
                                    if self._has_reached_llm_call_limit():
                                        raise LLMCallBudgetExceeded(
                                            "LLM call budget exhausted before another "
                                            "patch generation call."
                                        )
                                    loop_generated = self.generator.generate(
                                        current_source=self.current_source,
                                        failed_cases=[target_failure_case],
                                        recent_errors=[
                                            target_failure_case.get("error")
                                        ]
                                        if target_failure_case.get("error")
                                        else [],
                                        previous_program_source=prior_feedback.get(
                                            "program_source"
                                        ),
                                        previous_failed_case=prior_feedback.get(
                                            "failed_case"
                                        ),
                                        previous_failure_reason=prior_feedback.get(
                                            "reason"
                                        ),
                                        previous_smoke_errors=prior_feedback.get(
                                            "smoke_errors"
                                        ),
                                        regression_witness_cases=prior_feedback.get(
                                            "regression_witness_cases"
                                        ),
                                        previous_selected_regression_group_ids=prior_feedback.get(
                                            "selected_regression_group_ids"
                                        ),
                                        usage_context={
                                            "trial_index": self.iteration,
                                            "step_index": target_data_index or 1,
                                            "alignment_round": formal_failure_index,
                                        },
                                        attempt_callback=_emit_generator_attempt_progress,
                                        generation_record_callback=_record_generation_artifact,
                                        should_stop_callback=self._has_reached_llm_call_limit,
                                    )
                                except LLMCallBudgetExceeded as exc:
                                    patch_elapsed_text = self._stop_patch_elapsed_logger(
                                        start_time=patch_start_time,
                                        stop_event=patch_stop_event,
                                        thread=patch_thread,
                                    )
                                    decision_reason = str(exc).strip() or (
                                        "LLM call budget exhausted."
                                    )
                                    patch_attempt_artifact["status"] = "llm_call_limit"
                                    patch_attempt_artifact["error_message"] = (
                                        decision_reason
                                    )
                                    patch_attempt_artifact["elapsed_text"] = (
                                        patch_elapsed_text
                                    )
                                    _persist_patch_attempt_artifact()
                                    self._request_termination(
                                        reason="llm_call_limit",
                                        detail=decision_reason,
                                    )
                                    self._emit_log_event(
                                        "STOP",
                                        self._append_elapsed_text(
                                            decision_reason,
                                            patch_elapsed_text,
                                        ),
                                        indent_level=1,
                                        max_len=260,
                                    )
                                    break
                                except Exception as exc:
                                    patch_elapsed_text = self._stop_patch_elapsed_logger(
                                        start_time=patch_start_time,
                                        stop_event=patch_stop_event,
                                        thread=patch_thread,
                                    )
                                    error_message = (
                                        str(exc).strip() or exc.__class__.__name__
                                    )
                                    error_summary, error_action = (
                                        self._split_actionable_message(error_message)
                                    )
                                    self._emit_log_event(
                                        "ERROR",
                                        self._append_elapsed_text(
                                            "failed",
                                            patch_elapsed_text,
                                        ),
                                        indent_level=1,
                                        max_len=260,
                                    )
                                    if error_summary:
                                        self._emit_detail_line(
                                            f"reason={error_summary}",
                                            max_len=260,
                                        )
                                    if error_action:
                                        self._emit_detail_line(
                                            f"Do this: {error_action}",
                                            max_len=260,
                                        )
                                    patch_attempt_artifact["status"] = "llm_error"
                                    patch_attempt_artifact["error_message"] = error_message
                                    patch_attempt_artifact["elapsed_text"] = (
                                        patch_elapsed_text
                                    )
                                    _persist_patch_attempt_artifact()
                                    if unexpected_attempt >= max_unexpected_attempts:
                                        raise RuntimeError(
                                            self._format_unexpected_patch_retry_cap_message(
                                                error_message
                                            )
                                        ) from exc
                                    continue

                                patch_elapsed_text = self._stop_patch_elapsed_logger(
                                    start_time=patch_start_time,
                                    stop_event=patch_stop_event,
                                    thread=patch_thread,
                                )
                                candidate_count = 1 if loop_generated is not None else 0

                                if loop_generated is None or not loop_generated.patch_payload:
                                    decision_reason = "Invalid or unparsable patch payload."
                                    self._emit_log_event(
                                        "ERROR",
                                        self._append_elapsed_text(
                                            f"unexpected_patch_error reason={decision_reason}",
                                            patch_elapsed_text,
                                        ),
                                        indent_level=1,
                                        max_len=260,
                                    )
                                    patch_attempt_artifact["status"] = (
                                        "invalid_patch_payload"
                                    )
                                    patch_attempt_artifact["error_message"] = (
                                        decision_reason
                                    )
                                    patch_attempt_artifact["elapsed_text"] = (
                                        patch_elapsed_text
                                    )
                                    _persist_patch_attempt_artifact()
                                    if unexpected_attempt >= max_unexpected_attempts:
                                        raise RuntimeError(
                                            self._format_unexpected_patch_retry_cap_message(
                                                decision_reason
                                            )
                                        )
                                    continue

                                loop_applied = self.patcher.apply_patch(
                                    base_source=self.current_source,
                                    patch_payload=loop_generated.patch_payload,
                                )
                                if (
                                    not loop_applied.success
                                    or loop_applied.updated_source is None
                                ):
                                    decision_reason = (
                                        f"Patch apply failed: {loop_applied.error_message}"
                                        if not loop_applied.success
                                        else "Patch apply failed: empty updated source."
                                    )
                                    self._emit_log_event(
                                        "ERROR",
                                        self._append_elapsed_text(
                                            f"unexpected_patch_error reason={decision_reason}",
                                            patch_elapsed_text,
                                        ),
                                        indent_level=1,
                                        max_len=260,
                                    )
                                    patch_attempt_artifact["status"] = (
                                        "patch_apply_failed"
                                    )
                                    patch_attempt_artifact["error_message"] = (
                                        decision_reason
                                    )
                                    patch_attempt_artifact["elapsed_text"] = (
                                        patch_elapsed_text
                                    )
                                    _persist_patch_attempt_artifact()
                                    if unexpected_attempt >= max_unexpected_attempts:
                                        raise RuntimeError(
                                            self._format_unexpected_patch_retry_cap_message(
                                                decision_reason
                                            )
                                        )
                                    continue

                                patch_attempt_artifact["status"] = "applied"
                                patch_attempt_artifact["elapsed_text"] = patch_elapsed_text
                                patch_attempt_artifact["patch_digest"] = (
                                    loop_applied.patch_digest
                                )
                                patch_attempt_artifact["applied_program_path"] = (
                                    self._save_patch_attempt_applied_program(
                                        artifact_dir=patch_attempt_dir,
                                        source=loop_applied.updated_source,
                                    )
                                )
                                _persist_patch_attempt_artifact()
                                generated = loop_generated
                                applied = loop_applied
                                selected_patch_attempt_artifact = patch_attempt_artifact
                                failure_count = formal_failure_index

                            if self._requested_termination_reason is not None:
                                phase = "no_patch"
                                decision_reason = (
                                    self._requested_termination_detail
                                    or decision_reason
                                    or "Termination requested during patch generation."
                                )
                                break

                            diag = self._diagnose_candidate(
                                source=applied.updated_source,
                                patch_digest=applied.patch_digest,
                                candidate_index=0,
                                target=target,
                                current_accuracy=current_accuracy,
                            )
                            accepted = bool(diag.get("accepted_by_gate"))
                            delta_acc = (
                                diag.get("delta_acc")
                                if isinstance(diag.get("delta_acc"), (int, float))
                                else 0.0
                            )
                            best_acc = (
                                diag.get("accuracy")
                                if isinstance(diag.get("accuracy"), (int, float))
                                else current_accuracy
                            )
                            regressions = int(diag.get("protected_regressions") or 0)
                            selected_patch_digest = applied.patch_digest
                            regression_feedback = {
                                "broken_transition_keys": [],
                                "broken_group_ids": [],
                                "selected_regression_group_ids": [],
                                "regression_witness_cases": [],
                                "fail_groups": [],
                                "split_events": [],
                            }

                            rejection_reasons = diag.get("rejection_reasons") or []
                            if accepted:
                                decision_reason = (
                                    "target + protected set constraints satisfied."
                                )
                                attempt_reason = "Selected patch."
                            elif rejection_reasons:
                                decision_reason = "; ".join(
                                    str(r) for r in rejection_reasons
                                )
                                attempt_reason = decision_reason
                            else:
                                decision_reason = "gate_failed"
                                attempt_reason = decision_reason
                            if (
                                not accepted
                                and bool(diag.get("qualified_regression_reject"))
                            ):
                                regression_feedback = self._prepare_regression_feedback(
                                    target=target,
                                    diag=diag,
                                    patch_digest=selected_patch_digest,
                                    candidate_source=applied.updated_source,
                                )
                            elif not accepted and regressions > 0:
                                regression_feedback = (
                                    self._prepare_display_only_regression_feedback(
                                        diag=diag
                                    )
                                )
                            feedback_failed_case = None
                            patch_rejection_artifacts = None
                            if not accepted:
                                feedback_failed_case = (
                                    self._build_failure_case_from_prediction(
                                        transition=target,
                                        predicted_next_state_json=diag.get(
                                            "target_predicted_canonical"
                                        ),
                                        prediction_error=diag.get(
                                            "target_prediction_error"
                                        ),
                                        transition_key=target_key,
                                    )
                                )
                                if selected_patch_attempt_artifact is not None:
                                    patch_rejection_artifacts = (
                                        self._save_patch_rejection_artifacts(
                                            artifact_dir=self.output_dir
                                            / str(
                                                selected_patch_attempt_artifact.get(
                                                    "artifact_dir"
                                                )
                                                or ""
                                            ),
                                            target=target,
                                            target_failure_case=feedback_failed_case,
                                            regression_witness_cases=list(
                                                regression_feedback.get(
                                                    "regression_witness_cases"
                                                )
                                                or []
                                            ),
                                        )
                                    )
                            new_transition_image_paths = unexplained_transition_image_paths
                            if not accepted:
                                self._attach_rejection_metadata_to_transition_bundle(
                                    transition_image_paths=unexplained_transition_image_paths,
                                    rejection_metadata=(
                                        self._build_transition_bundle_rejection_metadata(
                                            reason=attempt_reason,
                                            rejection_reasons=rejection_reasons,
                                            qualified_regression_reject=bool(
                                                diag.get("qualified_regression_reject")
                                            ),
                                            regression_feedback=regression_feedback,
                                            patch_rejection_artifacts=patch_rejection_artifacts,
                                        )
                                    ),
                                )
                            if selected_patch_attempt_artifact is not None:
                                selected_patch_attempt_artifact["status"] = (
                                    "accepted" if accepted else "rejected"
                                )
                                selected_patch_attempt_artifact["accepted"] = bool(
                                    accepted
                                )
                                selected_patch_attempt_artifact["patch_digest"] = (
                                    selected_patch_digest
                                )
                                selected_patch_attempt_artifact["reason"] = (
                                    attempt_reason
                                )
                                selected_patch_attempt_artifact["accuracy"] = diag.get(
                                    "accuracy"
                                )
                                selected_patch_attempt_artifact["delta_acc"] = diag.get(
                                    "delta_acc"
                                )
                                selected_patch_attempt_artifact[
                                    "qualified_regression_reject"
                                ] = diag.get("qualified_regression_reject")
                                selected_patch_attempt_artifact[
                                    "rejection_reasons"
                                ] = list(rejection_reasons)
                                selected_patch_attempt_artifact[
                                    "broken_group_ids"
                                ] = list(
                                    regression_feedback.get("broken_group_ids") or []
                                )
                                selected_patch_attempt_artifact[
                                    "selected_regression_group_ids"
                                ] = list(
                                    regression_feedback.get(
                                        "selected_regression_group_ids"
                                    )
                                    or []
                                )
                                selected_patch_attempt_artifact["fail_groups"] = (
                                    self._summarize_fail_groups_for_bundle(
                                        regression_feedback
                                    )
                                )
                                selected_patch_attempt_artifact["split_events"] = list(
                                    regression_feedback.get("split_events") or []
                                )
                                if patch_rejection_artifacts is not None:
                                    selected_patch_attempt_artifact[
                                        "rejection_analysis"
                                    ] = patch_rejection_artifacts
                                self._write_json_artifact(
                                    self.output_dir
                                    / str(
                                        selected_patch_attempt_artifact.get(
                                            "summary_path"
                                        )
                                        or ""
                                    ),
                                    selected_patch_attempt_artifact,
                                )

                            retry_same_target = False
                            if accepted:
                                parent_version = self.current_version.version_id
                                self.current_version = self.repository.save_version(
                                    source=applied.updated_source,
                                    parent_version_id=parent_version,
                                    metadata={
                                        "iteration": self.iteration,
                                        "patch_digest": selected_patch_digest,
                                        "delta_acc": delta_acc,
                                        "current_accuracy": current_accuracy,
                                        "best_accuracy": best_acc,
                                        "regressions": regressions,
                                    },
                                )
                                self._snapshot_explorer_training_artifacts_for_current_version()
                                self.current_source = applied.updated_source
                                explained_transition_image_paths = (
                                    self._save_target_transition_images(
                                        transition=target,
                                        iteration=self.iteration,
                                        data_index=target_data_index or 1,
                                        failure_index=artifact_failure_index,
                                        patch_digest=self.current_version.version_id,
                                        step_index=target_step_index,
                                        predicted_next_state_json=diag.get(
                                            "target_predicted_canonical"
                                        ),
                                        target_prediction_error=diag.get(
                                            "target_prediction_error"
                                        ),
                                        stem_tag="explained",
                                    )
                                )
                                new_transition_image_paths = (
                                    self._merge_transition_image_paths(
                                        unexplained_transition_image_paths,
                                        explained_transition_image_paths,
                                    )
                                )
                                self._finalize_explained_transitions(
                                    [target],
                                    commit_version_override=self.current_version.version_id,
                                    refresh_canonical_groups=False,
                                    defer_explorer_batch_finalize=True,
                                )
                                self._refresh_transition_state_after_program_update()
                                self.saturation.mark_successful_patch(
                                    global_step=(
                                        int(target_step_index)
                                        if isinstance(target_step_index, int)
                                        else None
                                    ),
                                )
                                if self._requested_termination_reason == "saturation":
                                    self._clear_requested_termination()
                                phase = "incorporated"
                                self._emit_log_event(
                                    "ACCEPT",
                                    self._append_elapsed_text(
                                        (
                                            f"acc={best_acc:.4f} "
                                            f"delta={delta_acc:.4f} "
                                            f"rev={self.current_version.version_id} "
                                            f"classes={self._current_active_class_count()}"
                                        ),
                                        patch_elapsed_text,
                                    ),
                                    indent_level=1,
                                )
                                self._emit_single_blank_separator()
                                self._emit_program_context_transition_log(
                                    previous_version_id=parent_version,
                                )
                            else:
                                self._remember_feedback(
                                    target_key,
                                    feedback_failed_case,
                                    decision_reason,
                                    program_source=applied.updated_source,
                                    smoke_errors=diag.get("smoke_errors"),
                                    regression_witness_cases=regression_feedback.get(
                                        "regression_witness_cases"
                                    ),
                                    selected_regression_group_ids=regression_feedback.get(
                                        "selected_regression_group_ids"
                                    ),
                                    broken_transition_keys=regression_feedback.get(
                                        "broken_transition_keys"
                                    ),
                                    broken_group_ids=regression_feedback.get(
                                        "broken_group_ids"
                                    ),
                                )
                                self._record_failure_index(
                                    iteration=self.iteration,
                                    transition_key=target_key,
                                    failure_index=artifact_failure_index,
                                )
                                self._record_formal_failure_attempt(
                                    target_key,
                                    formal_failure_index,
                                )
                                reached_target_retry_cap = (
                                    formal_failure_index >= max_formal_attempts
                                )
                                phase = "no_program"
                                if reached_target_retry_cap:
                                    phase = "discovery_failed"
                                    decision_reason = self._format_target_retry_cap_message(
                                        failure_reason=attempt_reason,
                                        failure_index=formal_failure_index,
                                    )
                                    self._request_termination(
                                        reason="discovery_failed",
                                        detail=decision_reason,
                                    )
                                    self._emit_log_event(
                                        "FAILED",
                                        self._append_elapsed_text(
                                            (
                                                "discovery_failed "
                                                f"acc={best_acc:.4f} "
                                                f"delta={delta_acc:.4f} "
                                                f"reason={decision_reason}"
                                            ),
                                            patch_elapsed_text,
                                        ),
                                        indent_level=1,
                                    )
                                    self._emit_blank_line()
                                else:
                                    self._emit_reject_log_event(
                                        self._append_elapsed_text(
                                            (
                                                f"acc={best_acc:.4f} "
                                                f"delta={delta_acc:.4f} "
                                                f"reason={attempt_reason}"
                                            ),
                                            patch_elapsed_text,
                                        ),
                                        indent_level=1,
                                    )
                                    self._emit_log_event(
                                        "RETRY",
                                        (
                                            f"{self._format_attempt_retry_fields(formal_failure_index + 1)} "
                                            f"{patch_retry_progress}"
                                        ),
                                        indent_level=1,
                                        max_len=260,
                                    )
                                    emit_patch_header = False
                                    retry_same_target = True

                            record_reason = (
                                attempt_reason
                                if accepted or retry_same_target
                                else decision_reason
                            )
                            self.repository.record_attempt(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "iteration": self.iteration,
                                    "candidate_index": 0,
                                    "data_index": target_data_index or 1,
                                    "failure_index": artifact_failure_index,
                                    "patch_digest": selected_patch_digest,
                                    "accepted": accepted,
                                    "reason": record_reason,
                                    "delta_acc": diag.get("delta_acc"),
                                    "accuracy": diag.get("accuracy"),
                                    "regressions": diag.get("protected_regressions"),
                                    "smoke_ok": diag.get("smoke_ok"),
                                    "smoke_error_count": diag.get("smoke_error_count"),
                                    "compile_error_count": diag.get(
                                        "compile_error_count"
                                    ),
                                    "runtime_error_count": diag.get(
                                        "runtime_error_count"
                                    ),
                                    "qualified_regression_reject": diag.get(
                                        "qualified_regression_reject"
                                    ),
                                    "broken_transition_keys": regression_feedback.get(
                                        "broken_transition_keys"
                                    ),
                                    "broken_group_ids": regression_feedback.get(
                                        "broken_group_ids"
                                    ),
                                    "selected_regression_group_ids": regression_feedback.get(
                                        "selected_regression_group_ids"
                                    ),
                                    "split_events": regression_feedback.get(
                                        "split_events"
                                    ),
                                    "rejection_reasons": rejection_reasons,
                                    "new_transition_image_paths": new_transition_image_paths,
                                    "patch_attempt_artifact_dir": (
                                        selected_patch_attempt_artifact.get(
                                            "artifact_dir"
                                        )
                                        if isinstance(
                                            selected_patch_attempt_artifact,
                                            dict,
                                        )
                                        else None
                                    ),
                                    "patch_attempt_summary_json": (
                                        selected_patch_attempt_artifact.get(
                                            "summary_path"
                                        )
                                        if isinstance(
                                            selected_patch_attempt_artifact,
                                            dict,
                                        )
                                        else None
                                    ),
                                    "generator_attempt_artifacts": (
                                        selected_patch_attempt_artifact.get(
                                            "generator_attempts"
                                        )
                                        if isinstance(
                                            selected_patch_attempt_artifact,
                                            dict,
                                        )
                                        else None
                                    ),
                                    "applied_program_path": (
                                        selected_patch_attempt_artifact.get(
                                            "applied_program_path"
                                        )
                                        if isinstance(
                                            selected_patch_attempt_artifact,
                                            dict,
                                        )
                                        else None
                                    ),
                                    "patch_rejection_artifacts": (
                                        patch_rejection_artifacts
                                    ),
                                }
                            )
                            if retry_same_target:
                                prior_feedback = self._feedback_by_transition.get(
                                    target_key,
                                    {},
                                )
                                artifact_failure_index = self._next_failure_index(
                                    iteration=self.iteration,
                                    transition_key=target_key,
                                )
                                formal_failure_index = self._next_formal_failure_attempt(
                                    target_key,
                                )
                                unexplained_transition_image_paths = (
                                    self._save_target_transition_images(
                                        transition=target,
                                        iteration=self.iteration,
                                        data_index=target_data_index or 1,
                                        failure_index=artifact_failure_index,
                                        patch_digest=getattr(
                                            self.current_version,
                                            "version_id",
                                            None,
                                        ),
                                        step_index=target_step_index,
                                        predicted_next_state_json=(
                                            target_predicted_canonical
                                        ),
                                        target_prediction_error=(
                                            target_prediction_error
                                        ),
                                        stem_tag="unexplained",
                                    )
                                )
                                continue
                            break

                epoch_budget_remaining = self._count_active_collect_batch_pending()
                self._update_dashboard_progress_phase(
                    "verify",
                    clear_patch_target=True,
                )
                verify_pending_progress_bar = self._sync_verify_pending_batch_progress_bar(
                    verify_pending_progress_bar,
                    total=epoch_budget_size,
                    remaining=epoch_budget_remaining,
                    phase="verify",
                    current_action=None,
                )
                if epoch_budget_remaining <= 0:
                    self._emit_verify_phase_end(
                        epoch_index=epoch_index,
                        verified_count=epoch_budget_size,
                        total_count=epoch_budget_size,
                    )
                    self._flush_deferred_explained_transition_batch_if_verify_batch_complete()

                self._epoch_index = int(epoch_index)
                self._epoch_budget_size = int(epoch_budget_size)
                self._epoch_budget_remaining = int(epoch_budget_remaining)
                explorer_diagnostics = self._collect_explorer_diagnostics()
                self._log_iteration(
                    phase=phase,
                    collected_count=len(collected),
                    added_count=added,
                    eval_count=eval_count,
                    failure_count=failure_count,
                    candidate_count=candidate_count,
                    accepted=accepted,
                    delta_acc=delta_acc,
                    current_acc=current_accuracy,
                    best_acc=best_acc,
                    regressions=regressions,
                    decision_reason=decision_reason,
                    patch_digest=selected_patch_digest,
                    target_action=target_action,
                    epoch_index=epoch_index,
                    epoch_budget_size=epoch_budget_size,
                    epoch_budget_remaining=epoch_budget_remaining,
                    no_new_canonical_rounds=self.no_new_canonical_rounds,
                    max_no_new_canonical_rounds=self.max_no_new_canonical_rounds,
                    explorer_diagnostics=explorer_diagnostics,
                )
                self._maybe_emit_iteration_elapsed_line(
                    pending_remaining=epoch_budget_remaining
                )
                if self._requested_termination_reason is not None:
                    termination_reason = self._requested_termination_reason
                    break
        finally:
            if verify_pending_progress_bar is not None:
                self._close_verify_pending_progress_bar(
                    verify_pending_progress_bar,
                    previous_pbar=None,
                )
        self._last_termination_reason = str(termination_reason)
        self._last_termination_detail = (
            self._requested_termination_detail
            if isinstance(self._requested_termination_detail, str)
            and self._requested_termination_detail.strip()
            else None
        )
        print("\n" + "=" * 50)
        if termination_reason == "discovery_failed":
            print("DISCOVERY FAILED")
        elif termination_reason == "saturation":
            print("SATURATION REACHED")
        elif termination_reason == "llm_call_limit":
            print("LLM CALL LIMIT REACHED")
        elif termination_reason == "max_iterations":
            print("MAX ITERATIONS REACHED")
        elif termination_reason == "data_exhausted":
            print("DATA EXHAUSTED")
        else:
            print("STOPPED")
        if self._last_termination_detail:
            print(self._last_termination_detail)

        self._save_final()
        print("\n=== Final Program Version ===")
        print(self.current_version.version_id)
        return self.current_source

    def _diagnose_candidate(
        self,
        source: str,
        patch_digest: Optional[str],
        candidate_index: Optional[int],
        target: Transition,
        current_accuracy: float,
    ) -> Dict:
        diagnose_started_at = perf_counter()
        reasons: List[str] = []
        target_state = None
        try:
            target_state = parse_state_json(target.state)
        except ValueError:
            target_state = None

        smoke_started_at = perf_counter()
        smoke_ok, smoke_errors = self.sandbox.smoke_test(
            source=source,
            state=target_state,
            action=target.action,
        )
        smoke_elapsed_sec = perf_counter() - smoke_started_at
        smoke_error_rows = self._serialize_smoke_errors(smoke_errors)

        protected_compile = 0
        protected_runtime = 0
        protected_regressions = 0
        protected_count = len(self.protected_set)
        target_eval_elapsed_sec = 0.0
        protected_eval_elapsed_sec = 0.0
        target_compile = 0
        target_runtime = 0
        target_solved = False
        target_predicted_canonical = None
        target_expected_canonical = None
        target_prediction_error = None
        broken_protected_cases: List[Dict[str, Any]] = []
        broken_protected_transition_keys: List[str] = []
        broken_protected_predictions_by_key: Dict[str, str] = {}
        compile_count = 0
        runtime_count = 0
        accuracy = current_accuracy
        delta_acc = 0.0

        if smoke_ok:
            target_eval_started_at = perf_counter()
            target_eval = self.evaluator.evaluate_programs(
                programs=[
                    ProgramEvaluationTask(
                        label="candidate",
                        source=source,
                    )
                ],
                transitions=[target],
                collect_records=True,
            ).first().evaluation
            target_eval_elapsed_sec = perf_counter() - target_eval_started_at
            protected_eval = None
            if protected_count > 0:
                protected_eval_started_at = perf_counter()
                protected_eval = self.evaluator.evaluate_programs(
                    programs=[
                        ProgramEvaluationTask(
                            label="candidate",
                            source=source,
                        )
                    ],
                    transitions=list(self.protected_set),
                    collect_records=True,
                    failed_records_only=True,
                ).first().evaluation
                protected_eval_elapsed_sec = perf_counter() - protected_eval_started_at
            compile_count = max(
                len(target_eval.compile_errors),
                len(protected_eval.compile_errors) if protected_eval is not None else 0,
            )
            runtime_count = int(target_eval.runtime_error_count) + (
                int(protected_eval.runtime_error_count)
                if protected_eval is not None
                else 0
            )
            total_count = int(target_eval.total_count) + (
                int(protected_eval.total_count)
                if protected_eval is not None
                else 0
            )
            correct_count = int(target_eval.correct_count) + (
                int(protected_eval.correct_count)
                if protected_eval is not None
                else 0
            )
            accuracy = (correct_count / total_count) if total_count else current_accuracy
            protected_records = (
                list(protected_eval.records)
                if protected_eval is not None
                else []
            )
            target_record = target_eval.records[0] if target_eval.records else None
            delta_acc = accuracy - current_accuracy

            if self.protected_set:
                protected_compile = (
                    len(protected_eval.compile_errors)
                    if protected_eval is not None
                    else 0
                )
                protected_runtime = sum(
                    1
                    for row in protected_records
                    if self._has_non_compile_error(row.error)
                )
                regression_records = [
                    row
                    for row in protected_records
                    if row.error is None and not row.is_correct
                ]
                protected_regressions = len(regression_records)
                for row in regression_records:
                    transition_key = self._transition_key(row.transition)
                    broken_protected_transition_keys.append(transition_key)
                    if (
                        isinstance(row.predicted_canonical, str)
                        and row.predicted_canonical.strip()
                    ):
                        broken_protected_predictions_by_key[transition_key] = (
                            row.predicted_canonical
                        )
                if protected_compile:
                    reasons.append(f"protected_compile_errors({protected_compile})")
                if protected_runtime:
                    reasons.append(f"protected_runtime_errors({protected_runtime})")
                if protected_regressions:
                    reasons.append(f"protected_regressions({protected_regressions})")

            target_compile = len(target_eval.compile_errors)
            if target_record is not None:
                target_runtime = 1 if self._has_non_compile_error(target_record.error) else 0
                target_solved = (
                    bool(target_record.is_correct)
                    and target_compile == 0
                    and target_runtime == 0
                )
                target_predicted_canonical = target_record.predicted_canonical
                target_expected_canonical = target_record.expected_canonical
                if target_record.error is not None:
                    error_rows = self._serialize_smoke_errors([target_record.error])
                    if error_rows:
                        target_prediction_error = error_rows[0]
            else:
                target_runtime = 1
                target_prediction_error = {
                    "phase": "diagnose",
                    "message": "Missing target record in diagnosis evaluation.",
                    "exception_type": None,
                    "traceback_text": None,
                }

            if target_compile:
                reasons.append(f"target_compile_errors({target_compile})")
            if target_runtime:
                reasons.append(f"target_runtime_errors({target_runtime})")
            if not target_solved and target_compile == 0 and target_runtime == 0:
                reasons.append("target_mismatch")
        else:
            reasons.append("smoke_test_failed")
            compile_count = sum(
                1
                for row in smoke_error_rows
                if row.get("phase") in {"parse", "ast_validate", "compile"}
            )
            runtime_count = max(0, len(smoke_error_rows) - compile_count)

        accepted_by_gate = (
            smoke_ok
            and target_solved
            and protected_compile == 0
            and protected_runtime == 0
            and protected_regressions == 0
        )
        qualified_regression_reject = (
            smoke_ok
            and target_solved
            and protected_compile == 0
            and protected_runtime == 0
            and protected_regressions > 0
        )

        diagnose_elapsed_sec = perf_counter() - diagnose_started_at
        self._emit_timing_log(
            "DIAGNOSE_END",
            [
                "protected_eval="
                f"{self._format_patch_elapsed(protected_eval_elapsed_sec)}",
                f"protected={max(0, int(protected_count))}",
                f"regressions={max(0, int(protected_regressions))}",
            ],
            elapsed_sec=diagnose_elapsed_sec,
            indent_level=1,
            max_len=260,
        )
        return {
            "candidate_index": candidate_index,
            "patch_digest": patch_digest,
            "smoke_ok": smoke_ok,
            "smoke_error_count": len(smoke_error_rows),
            "smoke_errors": smoke_error_rows,
            "target_solved": target_solved,
            "protected_regressions": protected_regressions,
            "protected_compile_error_count": protected_compile,
            "protected_runtime_error_count": protected_runtime,
            "broken_protected_transition_keys": broken_protected_transition_keys,
            "broken_protected_cases": broken_protected_cases,
            "broken_protected_predictions_by_key": broken_protected_predictions_by_key,
            "target_compile_error_count": target_compile,
            "target_runtime_error_count": target_runtime,
            "target_predicted_canonical": target_predicted_canonical,
            "target_expected_canonical": target_expected_canonical,
            "target_prediction_error": target_prediction_error,
            "compile_error_count": compile_count,
            "runtime_error_count": runtime_count,
            "accuracy": accuracy,
            "delta_acc": delta_acc,
            "accepted_by_gate": accepted_by_gate,
            "qualified_regression_reject": qualified_regression_reject,
            "rejection_reasons": reasons,
        }

    def _is_target_solved(self, target_eval) -> bool:
        return (
            target_eval.correct_count == 1
            and not target_eval.compile_errors
            and target_eval.runtime_error_count == 0
        )

    def _compute_iteration_accuracy(
        self,
        protected_correct: int,
        protected_total: int,
        target_correct: int,
        target_total: int,
    ) -> float:
        total = max(0, protected_total) + max(0, target_total)
        if total == 0:
            return 1.0
        correct = max(0, protected_correct) + max(0, target_correct)
        return float(correct) / float(total)

    def _build_failure_case(self, target_eval, transition: Transition) -> Dict:
        if not target_eval.records:
            return self._build_failure_case_from_prediction(
                transition=transition,
                predicted_next_state_json=None,
                prediction_error={"message": "No evaluation record available."},
                transition_key=self._transition_key(transition),
            )
        record = target_eval.records[0]
        prediction_error = None
        if record.error is not None:
            prediction_error = {"message": record.error.message}
        return self._build_failure_case_from_prediction(
            transition=transition,
            predicted_next_state_json=record.predicted_canonical,
            prediction_error=prediction_error,
            transition_key=self._transition_key(transition),
        )

    def _build_failure_case_from_prediction(
        self,
        transition: Transition,
        predicted_next_state_json: Optional[str],
        prediction_error: Optional[Dict[str, Optional[str]]] = None,
        transition_key: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        normalized_state = self._resolve_canonical_state_json(transition.state)
        normalized_expected_next_state = self._resolve_canonical_state_json(
            transition.next_state
        )
        expected_difference = self._build_transition_difference(
            state_json=normalized_state,
            next_state_json=normalized_expected_next_state,
        )
        predicted_difference: Optional[Dict[str, Any]] = None
        mismatch_difference: Optional[Dict[str, Any]] = None
        predicted_next_state = (
            self._resolve_canonical_state_json(predicted_next_state_json)
            if isinstance(predicted_next_state_json, str)
            else None
        )
        if predicted_next_state:
            predicted_difference = self._build_transition_difference(
                state_json=normalized_state,
                next_state_json=predicted_next_state,
            )
            mismatch_difference = self._build_transition_difference(
                state_json=predicted_next_state,
                next_state_json=normalized_expected_next_state,
            )
        error_message = None
        if isinstance(prediction_error, dict):
            raw_error_message = prediction_error.get("message")
            if raw_error_message is not None:
                error_message = str(raw_error_message)
        case = {
            "transition_key": transition_key,
            "state": normalized_state,
            "action": transition.action,
            "expected_next_state": normalized_expected_next_state,
            "predicted_next_state": predicted_next_state,
            "difference": expected_difference,
            "expected_difference": expected_difference,
            "predicted_difference": predicted_difference,
            "mismatch_difference": mismatch_difference,
            "error": error_message,
        }
        if extra_fields:
            case.update(extra_fields)
        return case

    def _build_transition_difference(
        self,
        state_json: str,
        next_state_json: str,
    ) -> Dict[str, Any]:
        try:
            before = parse_state_json(state_json)
            after = parse_state_json(next_state_json)
        except ValueError as e:
            return {
                "changes": [
                    {
                        "path": "$",
                        "op": "replace",
                        "before": state_json,
                        "after": next_state_json,
                    }
                ],
                "total_changes": 1,
                "error": str(e),
            }

        changes: List[Dict[str, Any]] = []
        self._append_diff_changes(before=before, after=after, path="$", out=changes)

        return {
            "changes": changes,
            "total_changes": len(changes),
        }

    def _append_diff_changes(
        self,
        before: Any,
        after: Any,
        path: str,
        out: List[Dict[str, Any]],
    ) -> None:
        if type(before) is not type(after):
            out.append(
                {
                    "path": path,
                    "op": "replace",
                    "before": before,
                    "after": after,
                }
            )
            return

        if isinstance(before, dict):
            before_keys = set(before.keys())
            after_keys = set(after.keys())
            both = sorted(before_keys & after_keys)
            removed = sorted(before_keys - after_keys)
            added = sorted(after_keys - before_keys)

            for key in removed:
                child_path = f"{path}.{key}"
                out.append(
                    {
                        "path": child_path,
                        "op": "remove",
                        "before": before[key],
                    }
                )
            for key in added:
                child_path = f"{path}.{key}"
                out.append(
                    {
                        "path": child_path,
                        "op": "add",
                        "after": after[key],
                    }
                )
            for key in both:
                child_path = f"{path}.{key}"
                self._append_diff_changes(
                    before=before[key],
                    after=after[key],
                    path=child_path,
                    out=out,
                )
            return

        if isinstance(before, list):
            if self._should_diff_as_unordered_object_list(path, before, after):
                self._append_unordered_object_list_changes(
                    path=path,
                    before=before,
                    after=after,
                    out=out,
                )
                return

            common = min(len(before), len(after))
            for idx in range(common):
                child_path = f"{path}[{idx}]"
                self._append_diff_changes(
                    before=before[idx],
                    after=after[idx],
                    path=child_path,
                    out=out,
                )
            for idx in range(common, len(before)):
                child_path = f"{path}[{idx}]"
                out.append(
                    {
                        "path": child_path,
                        "op": "remove",
                        "before": before[idx],
                    }
                )
            for idx in range(common, len(after)):
                child_path = f"{path}[{idx}]"
                out.append(
                    {
                        "path": child_path,
                        "op": "add",
                        "after": after[idx],
                    }
                )
            return

        if before != after:
            out.append(
                {
                    "path": path,
                    "op": "replace",
                    "before": before,
                    "after": after,
                }
            )

    def _should_diff_as_unordered_object_list(
        self,
        path: str,
        before: List[Any],
        after: List[Any],
    ) -> bool:
        if not path.endswith(".objects"):
            return False
        return all(isinstance(item, dict) for item in before) and all(
            isinstance(item, dict) for item in after
        )

    def _append_unordered_object_list_changes(
        self,
        path: str,
        before: List[Any],
        after: List[Any],
        out: List[Dict[str, Any]],
    ) -> None:
        before_tokens = [self._to_canonical_json_text(item) for item in before]
        after_tokens = [self._to_canonical_json_text(item) for item in after]

        before_counts = Counter(before_tokens)
        after_counts = Counter(after_tokens)

        removed = before_counts - after_counts
        added = after_counts - before_counts

        for token in sorted(removed.keys()):
            item = json.loads(token)
            for _ in range(removed[token]):
                out.append(
                    {
                        "path": path,
                        "op": "remove",
                        "before": item,
                    }
                )

        for token in sorted(added.keys()):
            item = json.loads(token)
            for _ in range(added[token]):
                out.append(
                    {
                        "path": path,
                        "op": "add",
                        "after": item,
                    }
                )

    def _to_canonical_json_text(self, value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _is_protected_complete(self) -> bool:
        return len(self._protected_keys) >= len(self.canonical_d)

    def _resolve_collect_batch_queue(
        self,
        transitions: List[Transition],
    ) -> List[str]:
        batch_queue: List[str] = []
        seen_keys: set[str] = set()
        for transition in transitions:
            key = self._transition_key(transition)
            if key in self._protected_keys:
                continue
            if not self.canonical_d.contains_key(key):
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            batch_queue.append(key)
        return batch_queue

    def _sync_active_collect_batch_pending(self) -> None:
        synced_queue: List[str] = []
        seen_keys: set[str] = set()
        for key in self._active_collect_batch_queue:
            safe_key = str(key)
            if safe_key in seen_keys:
                continue
            if not self.canonical_d.contains_key(safe_key):
                continue
            if safe_key in self._protected_keys:
                continue
            seen_keys.add(safe_key)
            synced_queue.append(safe_key)
        self._active_collect_batch_queue = synced_queue

    def _mark_collect_batch_processed_many(
        self,
        transition_keys: List[str],
    ) -> None:
        processed_keys = {str(key) for key in transition_keys}
        if not processed_keys:
            return
        self._active_collect_batch_queue = [
            key for key in self._active_collect_batch_queue if key not in processed_keys
        ]

    def _active_collect_batch_unprotected_transitions(self) -> List[Transition]:
        pending_keys = [
            str(key)
            for key in self._active_collect_batch_queue
            if str(key) not in self._protected_keys
        ]
        if not pending_keys:
            return []
        transitions: List[Transition] = []
        for key in pending_keys:
            transition = self.canonical_d.transition_for_key(key)
            if transition is not None:
                transitions.append(transition)
        return transitions

    def _count_active_collect_batch_pending(self) -> int:
        return sum(
            1
            for key in self._active_collect_batch_queue
            if key not in self._protected_keys
        )

    def _create_verify_pending_progress_bar(self, *, total: int) -> Optional[Any]:
        return self._create_verify_progress_bar(
            total=total,
            desc="Verify pending",
        )

    def _format_verify_pending_progress_text(
        self,
        *,
        total: int,
        remaining: int,
        phase: Optional[str] = None,
        current_action: Optional[str] = None,
    ) -> str:
        safe_total = max(0, int(total))
        safe_remaining = max(0, min(int(remaining), safe_total))
        done = max(0, safe_total - safe_remaining)
        parts = [f"done={done}/{safe_total}", f"rem={safe_remaining}"]
        if isinstance(phase, str) and phase.strip():
            parts.append(f"phase={phase.strip()}")
        if isinstance(current_action, str) and current_action.strip():
            parts.append(f"action={current_action.strip()}")
        return " ".join(parts)

    def _build_verify_pending_postfix_text(
        self,
        *,
        total: int,
        remaining: int,
        phase: Optional[str] = None,
        current_action: Optional[str] = None,
    ) -> str:
        base = self._format_verify_pending_progress_text(
            total=total,
            remaining=remaining,
            phase=phase,
            current_action=current_action,
        )
        live_status = (
            str(self._verify_pending_live_status).strip()
            if isinstance(self._verify_pending_live_status, str)
            else ""
        )
        if live_status:
            return f"{base} {live_status}".strip()
        return base

    def _update_dashboard_progress_phase(
        self,
        phase: Optional[str],
        *,
        patch_target: Optional[Dict[str, Any]] = None,
        clear_patch_target: bool = False,
    ) -> None:
        visualizer = getattr(self.explorer, "visualizer", None)
        if visualizer is None:
            return
        update_payload_only = getattr(visualizer, "update_payload_only", None)
        if not callable(update_payload_only):
            return
        existing_payload = getattr(visualizer, "_displayed_payload", None)
        if not isinstance(existing_payload, dict):
            existing_payload = getattr(visualizer, "_last_payload", None)
        if not isinstance(existing_payload, dict):
            return
        normalized_phase = (
            str(phase).strip().lower()
            if isinstance(phase, str) and str(phase).strip()
            else None
        )
        if normalized_phase is None:
            return
        metrics: Dict[str, Any] = {
            "progress_phase": normalized_phase,
        }
        if isinstance(patch_target, dict):
            metrics["current_patch_step_index"] = (
                int(patch_target["step_index"])
                if isinstance(patch_target.get("step_index"), int)
                and int(patch_target.get("step_index")) > 0
                else None
            )
            metrics["current_patch_data_index"] = (
                int(patch_target["data_index"])
                if isinstance(patch_target.get("data_index"), int)
                and int(patch_target.get("data_index")) > 0
                else None
            )
            metrics["current_patch_failure_index"] = (
                int(patch_target["failure_index"])
                if isinstance(patch_target.get("failure_index"), int)
                and int(patch_target.get("failure_index")) > 0
                else None
            )
            patch_version = patch_target.get("current_version")
            metrics["current_patch_current_version"] = (
                str(patch_version).strip()
                if isinstance(patch_version, str) and str(patch_version).strip()
                else None
            )
            patch_action = patch_target.get("action")
            metrics["current_patch_action"] = (
                str(patch_action).strip()
                if isinstance(patch_action, str) and str(patch_action).strip()
                else None
            )
        elif clear_patch_target:
            metrics.update(
                {
                    "current_patch_step_index": None,
                    "current_patch_data_index": None,
                    "current_patch_failure_index": None,
                    "current_patch_current_version": None,
                    "current_patch_action": None,
                }
            )
        try:
            update_payload_only(
                metrics=metrics,
                merge_metrics=True,
                update_dashboard=False,
                force_snapshot=True,
                bypass_snapshot_delivery_gate=True,
            )
        except Exception:
            return

    def _format_verify_pending_patch_header(
        self,
        *,
        total: int,
        remaining: int,
        current_action: Optional[str] = None,
    ) -> str:
        safe_total = max(0, int(total))
        safe_remaining = max(0, min(int(remaining), safe_total))
        done = max(0, safe_total - safe_remaining)
        current_index = min(safe_total, done + 1) if safe_remaining > 0 else done
        parts = [f"pending={current_index}/{safe_total}", f"rem={safe_remaining}"]
        if isinstance(current_action, str) and current_action.strip():
            parts.append(f"action={current_action.strip()}")
        return " ".join(parts)

    def _update_verify_pending_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        total: int,
        remaining: int,
        phase: Optional[str] = None,
        current_action: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        if progress_bar is None:
            return
        safe_total = max(1, int(total))
        safe_remaining = max(0, min(int(remaining), safe_total))
        done = max(0, safe_total - safe_remaining)
        self._update_verify_progress_bar(
            progress_bar,
            evaluated=done,
            total=safe_total,
            refresh=False,
        )
        postfix = self._build_verify_pending_postfix_text(
            total=safe_total,
            remaining=safe_remaining,
            phase=phase,
            current_action=current_action,
        )
        postfix = self._build_progress_postfix_text(postfix)
        try:
            progress_bar.set_description_str("Verify pending", refresh=False)
        except TypeError:
            try:
                progress_bar.set_description_str("Verify pending")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
        try:
            progress_bar.set_postfix_str(postfix, refresh=False)
        except Exception:  # noqa: BLE001
            pass
        if refresh:
            self._refresh_progress_bar(progress_bar)

    def _emit_iteration_elapsed_line(self) -> None:
        iter_elapsed_text = self._current_iteration_elapsed_text()
        self._current_iteration_started_at = None
        if not isinstance(iter_elapsed_text, str) or not iter_elapsed_text:
            return
        display_iteration = int(self._current_log_iteration())
        parts = [
            f"[ITER {display_iteration:03d} END]",
            f"iter_elapsed={iter_elapsed_text}",
        ]
        total_elapsed_text = self._program_elapsed_text()
        if isinstance(total_elapsed_text, str) and total_elapsed_text:
            parts.append(f"total_elapsed={total_elapsed_text}")
        self._emit_plain_line(" ".join(parts), max_len=260)

    def _maybe_emit_iteration_elapsed_line(
        self,
        *,
        pending_remaining: Optional[int] = None,
    ) -> bool:
        if not isinstance(self._current_iteration_started_at, (int, float)):
            return False
        if pending_remaining is None:
            self._sync_active_collect_batch_pending()
            pending_remaining = self._count_active_collect_batch_pending()
        resolved_pending_remaining = max(0, int(pending_remaining))
        if resolved_pending_remaining > 0:
            return False
        self._emit_iteration_elapsed_line()
        return True

    def _refresh_verify_pending_progress_display(self, *, refresh: bool = True) -> None:
        state = self._verify_pending_progress_state
        progress_bar = getattr(self, "_active_progress_bar", None)
        if progress_bar is None or not isinstance(state, dict):
            return
        self._update_verify_pending_progress_bar(
            progress_bar,
            total=int(state.get("total", 0)),
            remaining=int(state.get("remaining", 0)),
            phase=state.get("phase"),
            current_action=state.get("current_action"),
            refresh=refresh,
        )

    def _set_verify_pending_live_status(self, status_text: Optional[str]) -> None:
        normalized = (
            str(status_text).strip()
            if isinstance(status_text, str) and str(status_text).strip()
            else None
        )
        self._verify_pending_live_status = normalized
        self._refresh_verify_pending_progress_display(refresh=False)

    def _clear_verify_pending_live_status(self) -> None:
        self._verify_pending_live_status = None
        self._refresh_verify_pending_progress_display(refresh=False)

    def _sync_verify_pending_batch_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        total: int,
        remaining: int,
        phase: Optional[str] = None,
        current_action: Optional[str] = None,
    ) -> Optional[Any]:
        safe_total = max(0, int(total))
        safe_remaining = max(0, min(int(remaining), safe_total))
        if safe_total <= 0 or safe_remaining <= 0:
            self._verify_pending_progress_state = None
            self._verify_pending_live_status = None
            if progress_bar is not None:
                self._close_verify_pending_progress_bar(
                    progress_bar,
                    previous_pbar=None,
                )
            return None

        self._verify_pending_progress_state = {
            "total": int(safe_total),
            "remaining": int(safe_remaining),
            "phase": phase,
            "current_action": current_action,
        }

        if progress_bar is None:
            progress_bar = self._create_verify_pending_progress_bar(total=safe_total)
        else:
            current_total = getattr(progress_bar, "total", None)
            if current_total != safe_total:
                self._close_verify_pending_progress_bar(
                    progress_bar,
                    previous_pbar=None,
                )
                progress_bar = self._create_verify_pending_progress_bar(
                    total=safe_total
                )
        self._update_verify_pending_progress_bar(
            progress_bar,
            total=safe_total,
            remaining=safe_remaining,
            phase=phase,
            current_action=current_action,
        )
        return progress_bar

    def _select_next_unprotected(self) -> Optional[Transition]:
        self._sync_active_collect_batch_pending()
        if self._active_collect_batch_queue:
            transition_key = self._active_collect_batch_queue[0]
            transition = self.canonical_d.transition_for_key(str(transition_key))
            if transition is not None:
                return transition
        if self._is_protected_complete():
            return None
        raise RuntimeError(
            "Active verify batch is empty while canonical still has unprotected "
            "transitions. Pending transitions must enter through the active collect "
            "batch queue."
        )

    def _clear_failure_index(
        self,
        *,
        iteration: int,
        transition_key: str,
    ) -> None:
        try:
            iter_index = int(iteration)
        except (TypeError, ValueError):
            return
        per_iter = self._failure_index_by_iteration_and_key.get(iter_index)
        if not isinstance(per_iter, dict):
            return
        per_iter.pop(str(transition_key), None)
        if not per_iter:
            self._failure_index_by_iteration_and_key.pop(iter_index, None)

    def _current_version_id_text(self) -> Optional[str]:
        current_version_id = getattr(self.current_version, "version_id", None)
        if not isinstance(current_version_id, str):
            return None
        normalized = current_version_id.strip()
        return normalized or None

    def _next_formal_failure_attempt(self, transition_key: str) -> int:
        safe_key = str(transition_key)
        current_version_id = self._current_version_id_text()
        state = self._formal_failure_attempt_state_by_transition_key.get(safe_key)
        if not isinstance(state, dict):
            return 1
        if state.get("version_id") != current_version_id:
            return 1
        current_attempt = state.get("attempt")
        if not isinstance(current_attempt, int) or current_attempt <= 0:
            return 1
        return int(current_attempt) + 1

    def _record_formal_failure_attempt(
        self,
        transition_key: str,
        failure_attempt: int,
    ) -> None:
        if not isinstance(failure_attempt, int) or failure_attempt <= 0:
            return
        self._formal_failure_attempt_state_by_transition_key[str(transition_key)] = {
            "version_id": self._current_version_id_text(),
            "attempt": int(failure_attempt),
        }

    def _clear_formal_failure_attempt(self, transition_key: str) -> None:
        self._formal_failure_attempt_state_by_transition_key.pop(
            str(transition_key),
            None,
        )

    def _format_saturation_termination_message(self) -> str:
        gap_steps = max(0, int(self.saturation.fail_count))
        window_steps = max(1, int(self.saturation.max_fail_count))
        current_global_step = self.saturation.last_seen_global_step
        last_patch_global_step = self.saturation.last_patch_global_step
        detail_parts = [f"gap={gap_steps}/{window_steps}"]
        if isinstance(last_patch_global_step, int):
            detail_parts.append(f"last_patch_global_step={last_patch_global_step}")
        if isinstance(current_global_step, int):
            detail_parts.append(f"current_global_step={current_global_step}")
        return (
            "No successful program patch was incorporated for "
            f"{gap_steps} global steps ({' '.join(detail_parts)}); discovery is saturated "
            "and will stop."
        )

    def _format_target_retry_cap_message(
        self,
        *,
        failure_reason: str,
        failure_index: int,
    ) -> str:
        try:
            resolved_failure_index = int(failure_index)
        except (TypeError, ValueError):
            resolved_failure_index = int(self.max_patch_attempts_per_target)
        cap = max(1, min(resolved_failure_index, int(self.max_patch_attempts_per_target)))
        reason_text = str(failure_reason or "").strip() or "Patch rejected."
        llm_label = self._format_llm_runtime_identity()
        return (
            f"{reason_text} Reached per-target retry cap ({cap}). "
            "This cap applies to the current "
            "program revision. "
            f"The current run's LLM ({llm_label}) could not solve this transition, "
            "so discovery failed."
        )

    def _promote_solved_pending_transitions(
        self,
        dataset: List[Transition],
        *,
        defer_explorer_batch_finalize: bool = False,
        defer_explorer_context_sync: bool = False,
    ) -> int:
        pending = [
            transition
            for transition in dataset
            if self._transition_key(transition) not in self._protected_keys
        ]
        if not pending:
            return 0

        pending_assessments = self._ensure_current_source_transition_assessments(pending)
        promoted_transitions: List[Transition] = []
        for transition in pending:
            transition_key = self._transition_key(transition)
            assessment = pending_assessments.get(transition_key)
            if not self._assessment_matches_current_source(assessment):
                continue
            if assessment.get("current_explains") is not True:
                continue
            if transition_key in self._protected_keys:
                continue
            promoted_transitions.append(transition)
        if promoted_transitions:
            self._refresh_canonical_group_assignments_from_cached_assessments(
                transitions=promoted_transitions,
                allow_classification=True,
            )
        return self._finalize_explained_transitions(
            promoted_transitions,
            defer_explorer_batch_finalize=defer_explorer_batch_finalize,
            defer_explorer_context_sync=defer_explorer_context_sync,
        )

    def _add_to_protected(
        self,
        transition: Transition,
        *,
        commit_version_override: Optional[str] = None,
    ) -> None:
        key = self._transition_key(transition)
        if key in self._protected_keys:
            return
        self._protected_keys.add(key)
        self.protected_set.append(transition)
        self._protected_transition_by_key[key] = transition

        normalized_override = self._normalize_commit_version(commit_version_override)
        reused_leaf_group_id: Optional[str] = None
        assessment_assignment = self._current_assessment_group_assignment(
            self._current_source_transition_assessments.get(key)
        )
        canonical_assignment = self._cached_canonical_group_assignment(key)
        if (
            commit_version_override is not None
            and self._is_class_eligible_commit_version(normalized_override)
        ):
            commit_version = normalized_override
            if (
                assessment_assignment is not None
                and assessment_assignment[1] in self._protected_group_metadata_by_id
                and self._normalize_commit_version(
                    self._protected_group_metadata_by_id[assessment_assignment[1]].get(
                        "commit_version"
                    )
                )
                == commit_version
            ):
                reused_leaf_group_id = assessment_assignment[1]
            elif (
                canonical_assignment is not None
                and canonical_assignment[0] == commit_version
            ):
                _canonical_commit_version, reused_leaf_group_id = canonical_assignment
            elif self._uses_dynamics_classes():
                reused_leaf_group_id = (
                    self._resolve_leaf_group_for_commit_version_transition(
                        transition=transition,
                        commit_version=commit_version,
                    )
                )
        elif (
            assessment_assignment is not None
            and assessment_assignment[1] in self._protected_group_metadata_by_id
        ):
            reused_leaf_group_id = assessment_assignment[1]
            group = self._protected_group_metadata_by_id[reused_leaf_group_id]
            commit_version = self._normalize_commit_version(group.get("commit_version"))
            if not self._is_class_eligible_commit_version(commit_version):
                commit_version = self._normalize_commit_version(
                    getattr(self.current_version, "version_id", None)
                )
        elif canonical_assignment is not None:
            commit_version, reused_leaf_group_id = canonical_assignment
        else:
            commit_version = self._normalize_commit_version(
                getattr(self.current_version, "version_id", None)
            )
        self._protected_commit_version_by_key[key] = commit_version
        if not self._uses_dynamics_classes():
            return

        if isinstance(reused_leaf_group_id, str) and reused_leaf_group_id:
            leaf_group_id = reused_leaf_group_id
        else:
            leaf_group_id = self._ensure_protected_root_group(commit_version)
        self._append_transition_key_to_group_lineage(leaf_group_id, key)
        self._ensure_leaf_group_class(leaf_group_id)
        self._protected_leaf_group_id_by_key[key] = leaf_group_id
        self._sync_canonical_group_assignment_from_protected(key)

    def _resolve_leaf_group_for_commit_version_transition(
        self,
        *,
        transition: Transition,
        commit_version: str,
    ) -> Optional[str]:
        root_group_id = self._ensure_protected_root_group(commit_version)
        return (
            self._route_transition_through_protected_group(
                transition=transition,
                root_group_id=root_group_id,
            )
            or root_group_id
        )

    def _route_transition_through_protected_group(
        self,
        *,
        transition: Transition,
        root_group_id: str,
    ) -> Optional[str]:
        current_group_id: Optional[str] = str(root_group_id).strip()
        visited_group_ids: Set[str] = set()
        while (
            isinstance(current_group_id, str)
            and current_group_id
            and current_group_id not in visited_group_ids
        ):
            visited_group_ids.add(current_group_id)
            group = self._protected_group_metadata_by_id.get(current_group_id)
            if group is None:
                return None
            child_group_ids = [
                str(child_group_id)
                for child_group_id in (group.get("child_group_ids") or [])
                if isinstance(child_group_id, str)
                and child_group_id in self._protected_group_metadata_by_id
            ]
            if not child_group_ids:
                return current_group_id

            broken_child_group_id = group.get("split_broken_child_group_id")
            kept_child_group_id = group.get("split_kept_child_group_id")
            split_program_source = self._load_split_program_source_for_group(
                current_group_id
            )
            if (
                isinstance(broken_child_group_id, str)
                and broken_child_group_id in self._protected_group_metadata_by_id
                and isinstance(kept_child_group_id, str)
                and kept_child_group_id in self._protected_group_metadata_by_id
                and isinstance(split_program_source, str)
                and split_program_source.strip()
            ):
                solved_by_split_program = self._program_source_explains_transition(
                    label=current_group_id,
                    source=split_program_source,
                    transition=transition,
                )
                current_group_id = (
                    kept_child_group_id
                    if solved_by_split_program
                    else broken_child_group_id
                )
                continue
            return self._first_leaf_descendant_group_id(current_group_id)
        return None

    def _first_leaf_descendant_group_id(self, group_id: str) -> Optional[str]:
        stack = [str(group_id)]
        visited_group_ids: Set[str] = set()
        while stack:
            current_group_id = stack.pop()
            if current_group_id in visited_group_ids:
                continue
            visited_group_ids.add(current_group_id)
            group = self._protected_group_metadata_by_id.get(current_group_id)
            if group is None:
                continue
            child_group_ids = [
                str(child_group_id)
                for child_group_id in (group.get("child_group_ids") or [])
                if isinstance(child_group_id, str)
                and child_group_id in self._protected_group_metadata_by_id
            ]
            if not child_group_ids:
                return current_group_id
            stack.extend(reversed(child_group_ids))
        return None

    def _program_source_explains_transition(
        self,
        *,
        label: str,
        source: str,
        transition: Transition,
    ) -> bool:
        batch = self.evaluator.evaluate_programs(
            programs=[ProgramEvaluationTask(label=str(label), source=str(source))],
            transitions=[transition],
            collect_records=True,
        )
        result = batch.by_label.get(str(label))
        explains = result.explains if result is not None else None
        return bool(explains[0]) if explains else False

    def _cached_canonical_group_assignment(
        self,
        transition_key: str,
    ) -> Optional[Tuple[str, str]]:
        safe_key = str(transition_key).strip()
        if not safe_key:
            return None
        leaf_group_id = self._canonical_leaf_group_id_by_key.get(safe_key)
        if not isinstance(leaf_group_id, str) or not leaf_group_id:
            return None
        group = self._protected_group_metadata_by_id.get(leaf_group_id)
        if not isinstance(group, dict):
            return None
        commit_version = self._normalize_commit_version(group.get("commit_version"))
        if not self._is_class_eligible_commit_version(commit_version):
            return None
        class_id = self._canonical_class_id_by_key.get(safe_key)
        if not isinstance(class_id, int) or class_id <= 0:
            return None
        return commit_version, leaf_group_id

    def _sync_canonical_group_assignment_from_protected(
        self,
        transition_key: str,
    ) -> bool:
        safe_key = str(transition_key).strip()
        if not safe_key:
            return False
        leaf_group_id = self._protected_leaf_group_id_by_key.get(safe_key)
        if not isinstance(leaf_group_id, str) or not leaf_group_id:
            return False
        class_id = self._group_class_id_by_group_id.get(leaf_group_id)
        if not isinstance(class_id, int) or int(class_id) <= 0:
            return False
        self._set_canonical_group_assignment(
            safe_key,
            class_id=int(class_id),
            group_id=leaf_group_id,
        )
        assessment = self._current_source_transition_assessments.get(safe_key)
        if self._assessment_matches_current_source(assessment):
            updated_assessment = dict(assessment)
            updated_assessment["assigned_class_id"] = int(class_id)
            updated_assessment["assigned_group_id"] = leaf_group_id
            updated_assessment["assignment_status"] = "assigned"
            self._current_source_transition_assessments[safe_key] = updated_assessment
        return True

    def _load_split_program_source_for_group(self, group_id: str) -> Optional[str]:
        group = self._protected_group_metadata_by_id.get(group_id)
        if group is None:
            return None
        inline_source = group.get("split_program_source")
        if isinstance(inline_source, str) and inline_source.strip():
            return inline_source
        return None

    def _append_transition_key_to_group_lineage(
        self,
        leaf_group_id: str,
        transition_key: str,
    ) -> None:
        current_group_id = leaf_group_id
        visited_group_ids = set()
        while (
            isinstance(current_group_id, str)
            and current_group_id
            and current_group_id not in visited_group_ids
        ):
            visited_group_ids.add(current_group_id)
            self._append_transition_key_to_group(current_group_id, transition_key)
            group = self._protected_group_metadata_by_id.get(current_group_id)
            if group is None:
                break
            parent_group_id = group.get("parent_group_id")
            current_group_id = (
                str(parent_group_id).strip()
                if isinstance(parent_group_id, str) and parent_group_id.strip()
                else None
            )

    def _normalize_commit_version(self, version_id: Optional[str]) -> str:
        if isinstance(version_id, str) and version_id.strip():
            return version_id.strip()
        return "v_unknown"

    def _is_class_eligible_commit_version(self, version_id: Optional[str]) -> bool:
        safe_version_id = self._normalize_commit_version(version_id)
        if safe_version_id == "v_unknown":
            return False
        version_text = safe_version_id[1:] if safe_version_id.lower().startswith("v") else safe_version_id
        if version_text.isdigit():
            return int(version_text) > 0
        return True

    def _allocate_group_class_id(self) -> int:
        class_id = max(1, int(self._next_group_class_id))
        self._next_group_class_id = int(class_id) + 1
        return int(class_id)

    def _assign_group_class_id(
        self,
        *,
        group_id: str,
        class_id: int,
    ) -> int:
        safe_group_id = str(group_id).strip()
        safe_class_id = int(class_id)
        if not safe_group_id or safe_class_id <= 0:
            raise ValueError("group_id and class_id must be valid.")
        previous_group_id = self._leaf_group_id_by_class_id.get(safe_class_id)
        if isinstance(previous_group_id, str) and previous_group_id and previous_group_id != safe_group_id:
            self._group_class_id_by_group_id.pop(previous_group_id, None)
            previous_group = self._protected_group_metadata_by_id.get(previous_group_id)
            if previous_group is not None:
                previous_group["class_id"] = None
                previous_group["retired_class_id"] = safe_class_id
        self._group_class_id_by_group_id[safe_group_id] = safe_class_id
        self._leaf_group_id_by_class_id[safe_class_id] = safe_group_id
        group = self._protected_group_metadata_by_id.get(safe_group_id)
        if group is not None:
            group["class_id"] = safe_class_id
        self._group_context_generation += 1
        return int(safe_class_id)

    def _retire_group_class_id(self, group_id: str) -> Optional[int]:
        safe_group_id = str(group_id).strip()
        if not safe_group_id:
            return None
        class_id = self._group_class_id_by_group_id.pop(safe_group_id, None)
        if isinstance(class_id, int) and class_id > 0:
            mapped_group_id = self._leaf_group_id_by_class_id.get(int(class_id))
            if mapped_group_id == safe_group_id:
                self._leaf_group_id_by_class_id.pop(int(class_id), None)
            group = self._protected_group_metadata_by_id.get(safe_group_id)
            if group is not None:
                group["retired_class_id"] = int(class_id)
                group["class_id"] = None
            self._group_context_generation += 1
            return int(class_id)
        return None

    def _ensure_protected_root_group(self, commit_version: str) -> str:
        existing_group_id = self._protected_root_group_id_by_commit_version.get(commit_version)
        if isinstance(existing_group_id, str) and existing_group_id in self._protected_group_metadata_by_id:
            return existing_group_id

        root_group_id = self._create_protected_group(
            commit_version=commit_version,
            parent_group_id=None,
            member_transition_keys=[],
        )
        self._protected_root_group_id_by_commit_version[commit_version] = root_group_id
        self._protected_ingress_leaf_group_id_by_commit_version[commit_version] = root_group_id
        return root_group_id

    def _ensure_leaf_group_class(self, group_id: str) -> Optional[int]:
        safe_group_id = str(group_id).strip()
        if not safe_group_id:
            return None
        group = self._protected_group_metadata_by_id.get(safe_group_id)
        if group is None:
            return None
        if group.get("child_group_ids"):
            return None
        if not list(group.get("member_transition_keys") or []):
            return None
        commit_version = self._normalize_commit_version(group.get("commit_version"))
        if not self._is_class_eligible_commit_version(commit_version):
            return None
        existing_class_id = group.get("class_id")
        if isinstance(existing_class_id, int) and existing_class_id > 0:
            changed = False
            if self._group_class_id_by_group_id.get(safe_group_id) != int(existing_class_id):
                self._group_class_id_by_group_id[safe_group_id] = int(existing_class_id)
                changed = True
            if self._leaf_group_id_by_class_id.get(int(existing_class_id)) != safe_group_id:
                self._leaf_group_id_by_class_id[int(existing_class_id)] = safe_group_id
                changed = True
            if changed:
                self._group_context_generation += 1
            return int(existing_class_id)
        return self._assign_group_class_id(
            group_id=safe_group_id,
            class_id=self._allocate_group_class_id(),
        )

    def _create_protected_group(
        self,
        commit_version: str,
        parent_group_id: Optional[str],
        member_transition_keys: List[str],
    ) -> str:
        next_index = int(self._next_group_index_by_commit_version.get(commit_version, 0))
        self._next_group_index_by_commit_version[commit_version] = next_index + 1
        group_id = f"{commit_version}:g{next_index:04d}"
        self._protected_group_metadata_by_id[group_id] = {
            "group_id": group_id,
            "commit_version": commit_version,
            "parent_group_id": parent_group_id,
            "child_group_ids": [],
            "member_transition_keys": list(member_transition_keys),
            "class_id": None,
            "retired_class_id": None,
        }
        self._protected_group_member_key_sets_by_id[group_id] = set(member_transition_keys)
        if parent_group_id is not None:
            parent_group = self._protected_group_metadata_by_id.get(parent_group_id)
            if parent_group is not None:
                child_group_ids = list(parent_group.get("child_group_ids") or [])
                if group_id not in child_group_ids:
                    child_group_ids.append(group_id)
                    parent_group["child_group_ids"] = child_group_ids
        return group_id

    def _sanitize_protected_group_registry(self) -> None:
        changed = False
        removed_any = True
        while removed_any:
            removed_any = False
            removable_group_ids = [
                group_id
                for group_id, group in self._protected_group_metadata_by_id.items()
                if not list(group.get("child_group_ids") or [])
                and not list(group.get("member_transition_keys") or [])
                and not (
                    isinstance(group.get("class_id"), int)
                    and int(group.get("class_id")) > 0
                )
                and not (
                    isinstance(group.get("retired_class_id"), int)
                    and int(group.get("retired_class_id")) > 0
                )
                and not (
                    not (
                        isinstance(group.get("parent_group_id"), str)
                        and str(group.get("parent_group_id")).strip()
                    )
                    and str(group_id)
                    == str(
                        self._protected_root_group_id_by_commit_version.get(
                            self._normalize_commit_version(group.get("commit_version"))
                        )
                    )
                )
            ]
            if not removable_group_ids:
                break
            for group_id in removable_group_ids:
                group = self._protected_group_metadata_by_id.get(group_id)
                if group is None:
                    continue
                parent_group_id = group.get("parent_group_id")
                if isinstance(parent_group_id, str) and parent_group_id:
                    parent_group = self._protected_group_metadata_by_id.get(parent_group_id)
                    if parent_group is not None:
                        parent_group["child_group_ids"] = [
                            str(child_group_id)
                            for child_group_id in (parent_group.get("child_group_ids") or [])
                            if str(child_group_id) != str(group_id)
                        ]
                self._group_class_id_by_group_id.pop(str(group_id), None)
                for class_id, mapped_group_id in list(self._leaf_group_id_by_class_id.items()):
                    if str(mapped_group_id) == str(group_id):
                        self._leaf_group_id_by_class_id.pop(int(class_id), None)
                for commit_version, root_group_id in list(self._protected_root_group_id_by_commit_version.items()):
                    if str(root_group_id) == str(group_id):
                        self._protected_root_group_id_by_commit_version.pop(str(commit_version), None)
                for commit_version, ingress_group_id in list(self._protected_ingress_leaf_group_id_by_commit_version.items()):
                    if isinstance(ingress_group_id, str) and str(ingress_group_id) == str(group_id):
                        self._protected_ingress_leaf_group_id_by_commit_version[str(commit_version)] = None
                self._protected_group_metadata_by_id.pop(str(group_id), None)
                self._protected_group_member_key_sets_by_id.pop(str(group_id), None)
                removed_any = True
                changed = True

        root_group_by_commit: Dict[str, str] = {}
        for group_id in sorted(self._protected_group_metadata_by_id.keys()):
            group = self._protected_group_metadata_by_id[group_id]
            parent_group_id = group.get("parent_group_id")
            has_parent = (
                isinstance(parent_group_id, str)
                and parent_group_id.strip()
                and parent_group_id in self._protected_group_metadata_by_id
            )
            if has_parent:
                continue
            commit_version = self._normalize_commit_version(group.get("commit_version"))
            root_group_by_commit.setdefault(commit_version, str(group_id))
        if dict(sorted(self._protected_root_group_id_by_commit_version.items())) != dict(sorted(root_group_by_commit.items())):
            self._protected_root_group_id_by_commit_version = root_group_by_commit
            changed = True

        rebuilt_next_group_index_by_commit_version: Dict[str, int] = {}
        for group_id in sorted(self._protected_group_metadata_by_id.keys()):
            group = self._protected_group_metadata_by_id[group_id]
            commit_version = self._normalize_commit_version(group.get("commit_version"))
            prefix = f"{commit_version}:g"
            next_group_index = rebuilt_next_group_index_by_commit_version.get(
                commit_version,
                0,
            )
            if str(group_id).startswith(prefix):
                suffix = str(group_id)[len(prefix) :]
                if suffix.isdigit():
                    next_group_index = max(next_group_index, int(suffix) + 1)
            rebuilt_next_group_index_by_commit_version[commit_version] = int(
                next_group_index
            )
        if (
            self._next_group_index_by_commit_version
            != rebuilt_next_group_index_by_commit_version
        ):
            self._next_group_index_by_commit_version = (
                rebuilt_next_group_index_by_commit_version
            )
            changed = True

        rebuilt_group_class: Dict[str, int] = {}
        rebuilt_leaf_group_by_class: Dict[int, str] = {}
        for group_id in sorted(self._protected_group_metadata_by_id.keys()):
            group = self._protected_group_metadata_by_id[group_id]
            child_group_ids = [
                str(child_group_id)
                for child_group_id in (group.get("child_group_ids") or [])
                if isinstance(child_group_id, str)
                and child_group_id in self._protected_group_metadata_by_id
            ]
            if child_group_ids != list(group.get("child_group_ids") or []):
                group["child_group_ids"] = child_group_ids
                changed = True
            class_id = group.get("class_id")
            is_leaf = not child_group_ids
            has_members = bool(list(group.get("member_transition_keys") or [])) or int(
                group.get("restored_member_transition_count") or 0
            ) > 0
            eligible_commit_version = self._is_class_eligible_commit_version(
                group.get("commit_version")
            )
            if (
                isinstance(class_id, int)
                and class_id > 0
                and is_leaf
                and has_members
                and eligible_commit_version
                and int(class_id) not in rebuilt_leaf_group_by_class
            ):
                rebuilt_group_class[str(group_id)] = int(class_id)
                rebuilt_leaf_group_by_class[int(class_id)] = str(group_id)
                continue
            if group.get("class_id") is not None and not (
                isinstance(class_id, int)
                and class_id > 0
                and is_leaf
                and eligible_commit_version
            ):
                group["class_id"] = None
                changed = True
        if self._group_class_id_by_group_id != rebuilt_group_class:
            self._group_class_id_by_group_id = rebuilt_group_class
            changed = True
        if self._leaf_group_id_by_class_id != rebuilt_leaf_group_by_class:
            self._leaf_group_id_by_class_id = rebuilt_leaf_group_by_class
            changed = True
        known_class_ids = set(int(class_id) for class_id in self._leaf_group_id_by_class_id.keys())
        for group in self._protected_group_metadata_by_id.values():
            for field_name in ("class_id", "retired_class_id"):
                raw_class_id = group.get(field_name)
                if isinstance(raw_class_id, int) and int(raw_class_id) > 0:
                    known_class_ids.add(int(raw_class_id))
        next_group_class_id = max(known_class_ids, default=0) + 1
        if int(self._next_group_class_id) != int(max(1, next_group_class_id)):
            self._next_group_class_id = int(max(1, next_group_class_id))
            changed = True
        if changed:
            self._group_context_generation += 1

    def _group_member_transition_keys(
        self,
        group_id: str,
        group: Dict[str, Any],
    ) -> List[Any]:
        member_transition_keys = group.get("member_transition_keys")
        if isinstance(member_transition_keys, list):
            return member_transition_keys
        member_transition_keys = list(member_transition_keys or [])
        group["member_transition_keys"] = member_transition_keys
        self._protected_group_member_key_sets_by_id[str(group_id)] = set(member_transition_keys)
        return member_transition_keys

    def _group_member_transition_key_set(
        self,
        group_id: str,
        group: Dict[str, Any],
    ) -> Set[Any]:
        safe_group_id = str(group_id)
        member_transition_keys = self._group_member_transition_keys(safe_group_id, group)
        member_transition_key_set = self._protected_group_member_key_sets_by_id.get(
            safe_group_id
        )
        if member_transition_key_set is None:
            member_transition_key_set = set(member_transition_keys)
            self._protected_group_member_key_sets_by_id[safe_group_id] = (
                member_transition_key_set
            )
        return member_transition_key_set

    def _append_transition_key_to_group(self, group_id: str, transition_key: str) -> None:
        group = self._protected_group_metadata_by_id.get(group_id)
        if group is None:
            return
        member_transition_keys = self._group_member_transition_keys(group_id, group)
        member_transition_key_set = self._group_member_transition_key_set(group_id, group)
        if transition_key not in member_transition_key_set:
            member_transition_key_set.add(transition_key)
            member_transition_keys.append(transition_key)

    def _remember_feedback(
        self,
        transition_key: str,
        failed_case: Dict,
        reason: str,
        program_source: Optional[str] = None,
        smoke_errors: Optional[List[Dict[str, Optional[str]]]] = None,
        regression_witness_cases: Optional[List[Dict[str, Any]]] = None,
        selected_regression_group_ids: Optional[List[str]] = None,
        broken_transition_keys: Optional[List[str]] = None,
        broken_group_ids: Optional[List[str]] = None,
    ) -> None:
        self._feedback_by_transition[transition_key] = {
            "program_source": program_source,
            "failed_case": failed_case,
            "reason": reason,
            "smoke_errors": smoke_errors or [],
            "regression_witness_cases": regression_witness_cases or [],
            "selected_regression_group_ids": selected_regression_group_ids or [],
            "broken_transition_keys": broken_transition_keys or [],
            "broken_group_ids": broken_group_ids or [],
        }

    def _apply_no_patch_retry_policy(
        self,
        *,
        transition_key: str,
        failed_case: Dict[str, Any],
        reason: str,
        failure_index: int,
        count_as_failure: bool = False,
    ) -> None:
        if not count_as_failure:
            return
        self._remember_feedback(
            transition_key,
            failed_case,
            reason,
            program_source=None,
        )
        self._record_failure_index(
            iteration=self.iteration,
            transition_key=transition_key,
            failure_index=failure_index,
        )
        self._record_formal_failure_attempt(
            transition_key,
            failure_index,
        )

    def _serialize_smoke_errors(
        self,
        smoke_errors: List[Any],
    ) -> List[Dict[str, Optional[str]]]:
        rows: List[Dict[str, Optional[str]]] = []
        for err in smoke_errors:
            if isinstance(err, dict):
                phase = err.get("phase")
                message = err.get("message")
                exception_type = err.get("exception_type")
                traceback_text = err.get("traceback_text")
            else:
                phase = getattr(err, "phase", None)
                message = getattr(err, "message", None)
                exception_type = getattr(err, "exception_type", None)
                traceback_text = getattr(err, "traceback_text", None)

            rows.append(
                {
                    "phase": str(phase) if phase is not None else None,
                    "message": str(message) if message is not None else None,
                    "exception_type": (
                        str(exception_type) if exception_type is not None else None
                    ),
                    "traceback_text": (
                        str(traceback_text) if traceback_text is not None else None
                    ),
                }
            )
        return rows

    def _has_non_compile_error(self, error: Any) -> bool:
        if error is None:
            return False
        if isinstance(error, dict):
            phase = error.get("phase")
        else:
            phase = getattr(error, "phase", None)
        if phase is None:
            return True
        return str(phase) not in {"parse", "ast_validate", "compile"}

    def _prepare_regression_feedback(
        self,
        *,
        target: Transition,
        diag: Dict[str, Any],
        patch_digest: Optional[str],
        candidate_source: Optional[str],
    ) -> Dict[str, Any]:
        broken_transition_keys = [
            str(key)
            for key in (diag.get("broken_protected_transition_keys") or [])
            if isinstance(key, str) and key.strip()
        ]
        if not broken_transition_keys:
            return {
                "broken_transition_keys": [],
                "broken_group_ids": [],
                "selected_regression_group_ids": [],
                "regression_witness_cases": [],
                "fail_groups": [],
                "split_events": [],
            }

        if (
            not self._uses_dynamics_classes()
            or self.regression_witness_mode in {"random", "none"}
        ):
            witness_cases = (
                []
                if self.regression_witness_mode == "none"
                else self._select_random_regression_witness_cases(
                    broken_transition_keys=broken_transition_keys,
                    broken_cases=diag.get("broken_protected_cases") or [],
                    broken_predictions_by_key=diag.get(
                        "broken_protected_predictions_by_key"
                    )
                    or {},
                )
            )
            return {
                "broken_transition_keys": broken_transition_keys,
                "broken_group_ids": [],
                "selected_regression_group_ids": [],
                "regression_witness_cases": witness_cases,
                "fail_groups": [],
                "split_events": [],
            }

        split_events: List[Dict[str, Any]] = []
        if self._allows_dynamics_class_refinement():
            split_events = self._refine_leaf_groups_from_broken_transitions(
                broken_transition_keys,
                candidate_source=candidate_source,
                patch_digest=patch_digest,
                target_transition_key=self._transition_key(target),
            )
        broken_group_to_transition_keys = self._group_broken_transition_keys_by_leaf(
            broken_transition_keys
        )
        broken_group_ids = sorted(broken_group_to_transition_keys.keys())
        witness_cases = self._select_regression_witness_cases(
            broken_group_to_transition_keys=broken_group_to_transition_keys,
            broken_cases=diag.get("broken_protected_cases") or [],
            broken_predictions_by_key=diag.get("broken_protected_predictions_by_key")
            or {},
        )

        if split_events:
            self._sync_explorer_program_context(emit_log=False)

        return {
            "broken_transition_keys": broken_transition_keys,
            "broken_group_ids": broken_group_ids,
            "selected_regression_group_ids": [
                case.get("group_id")
                for case in witness_cases
                if isinstance(case.get("group_id"), str)
            ],
            "regression_witness_cases": witness_cases,
            "fail_groups": self._summarize_fail_groups(
                broken_group_to_transition_keys=broken_group_to_transition_keys,
            ),
            "split_events": split_events,
        }

    def _prepare_display_only_regression_feedback(
        self,
        *,
        diag: Dict[str, Any],
    ) -> Dict[str, Any]:
        broken_transition_keys = [
            str(key)
            for key in (diag.get("broken_protected_transition_keys") or [])
            if isinstance(key, str) and key.strip()
        ]
        if not broken_transition_keys:
            return {
                "broken_transition_keys": [],
                "broken_group_ids": [],
                "selected_regression_group_ids": [],
                "regression_witness_cases": [],
                "fail_groups": [],
                "split_events": [],
            }
        if not self._uses_dynamics_classes():
            return {
                "broken_transition_keys": broken_transition_keys,
                "broken_group_ids": [],
                "selected_regression_group_ids": [],
                "regression_witness_cases": [],
                "fail_groups": [],
                "split_events": [],
            }
        broken_group_to_transition_keys = self._group_broken_transition_keys_by_leaf(
            broken_transition_keys
        )
        broken_group_ids = sorted(broken_group_to_transition_keys.keys())
        return {
            "broken_transition_keys": broken_transition_keys,
            "broken_group_ids": broken_group_ids,
            "selected_regression_group_ids": [],
            "regression_witness_cases": [],
            "fail_groups": self._summarize_fail_groups(
                broken_group_to_transition_keys=broken_group_to_transition_keys,
            ),
            "split_events": [],
        }

    def _refine_leaf_groups_from_broken_transitions(
        self,
        broken_transition_keys: List[str],
        *,
        candidate_source: Optional[str],
        patch_digest: Optional[str],
        target_transition_key: Optional[str],
    ) -> List[Dict[str, Any]]:
        broken_key_set = {
            str(key)
            for key in broken_transition_keys
            if isinstance(key, str) and key.strip()
        }
        if not broken_key_set:
            return []

        split_events: List[Dict[str, Any]] = []
        candidate_leaf_group_ids = sorted(
            {
                self._protected_leaf_group_id_by_key.get(key)
                for key in broken_key_set
                if isinstance(self._protected_leaf_group_id_by_key.get(key), str)
            }
        )
        for leaf_group_id in candidate_leaf_group_ids:
            if not isinstance(leaf_group_id, str):
                continue
            leaf_group = self._protected_group_metadata_by_id.get(leaf_group_id)
            if leaf_group is None or leaf_group.get("child_group_ids"):
                continue
            member_transition_keys = list(leaf_group.get("member_transition_keys") or [])
            member_set = set(member_transition_keys)
            broken_in_group = sorted(member_set & broken_key_set)
            if not broken_in_group or len(broken_in_group) == len(member_transition_keys):
                continue

            kept_in_group = sorted(member_set - set(broken_in_group))
            if not kept_in_group:
                continue

            commit_version = self._normalize_commit_version(
                leaf_group.get("commit_version")
                if isinstance(leaf_group.get("commit_version"), str)
                else None
            )
            parent_class_id = self._retire_group_class_id(leaf_group_id)
            broken_child_group_id = self._create_protected_group(
                commit_version=commit_version,
                parent_group_id=leaf_group_id,
                member_transition_keys=broken_in_group,
            )
            kept_child_group_id = self._create_protected_group(
                commit_version=commit_version,
                parent_group_id=leaf_group_id,
                member_transition_keys=kept_in_group,
            )
            if isinstance(parent_class_id, int) and parent_class_id > 0:
                kept_class_id = self._assign_group_class_id(
                    group_id=kept_child_group_id,
                    class_id=parent_class_id,
                )
            else:
                kept_class_id = self._assign_group_class_id(
                    group_id=kept_child_group_id,
                    class_id=self._allocate_group_class_id(),
                )
            broken_class_id = self._assign_group_class_id(
                group_id=broken_child_group_id,
                class_id=self._allocate_group_class_id(),
            )
            if (
                self._protected_ingress_leaf_group_id_by_commit_version.get(commit_version)
                == leaf_group_id
            ):
                self._protected_ingress_leaf_group_id_by_commit_version[commit_version] = None

            leaf_group["split_program_source"] = (
                candidate_source
                if isinstance(candidate_source, str) and candidate_source.strip()
                else None
            )
            leaf_group["split_patch_digest"] = patch_digest
            leaf_group["split_target_transition_key"] = target_transition_key
            leaf_group["split_broken_child_group_id"] = broken_child_group_id
            leaf_group["split_kept_child_group_id"] = kept_child_group_id
            leaf_group["split_broken_class_id"] = int(broken_class_id)
            leaf_group["split_kept_class_id"] = int(kept_class_id)

            for transition_key in broken_in_group:
                self._protected_leaf_group_id_by_key[transition_key] = broken_child_group_id
            for transition_key in kept_in_group:
                self._protected_leaf_group_id_by_key[transition_key] = kept_child_group_id

            split_events.append(
                {
                    "split_group_id": leaf_group_id,
                    "source_class_id": int(parent_class_id) if isinstance(parent_class_id, int) and parent_class_id > 0 else None,
                    "broken_child_group_id": broken_child_group_id,
                    "broken_class_id": int(broken_class_id),
                    "kept_child_group_id": kept_child_group_id,
                    "kept_class_id": int(kept_class_id),
                    "broken_transition_keys": broken_in_group,
                    "kept_transition_keys": kept_in_group,
                }
            )
        if split_events:
            self._refresh_canonical_group_assignments_for_split_events(split_events)
        return split_events

    def _group_broken_transition_keys_by_leaf(
        self,
        broken_transition_keys: List[str],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for transition_key in sorted(
            str(key)
            for key in broken_transition_keys
            if isinstance(key, str) and key.strip()
        ):
            group_id = self._protected_leaf_group_id_by_key.get(transition_key)
            if not isinstance(group_id, str) or not group_id:
                continue
            grouped.setdefault(group_id, []).append(transition_key)
        return grouped

    def _select_random_regression_witness_cases(
        self,
        *,
        broken_transition_keys: List[str],
        broken_cases: List[Dict[str, Any]],
        broken_predictions_by_key: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.regression_group_sample_k <= 0:
            return []
        candidate_keys = sorted(
            str(key)
            for key in broken_transition_keys
            if isinstance(key, str) and key.strip()
        )
        if not candidate_keys:
            return []
        sample_size = min(int(self.regression_group_sample_k), len(candidate_keys))
        if sample_size <= 0:
            return []
        if len(candidate_keys) > sample_size:
            selected_keys = self._prompt_rng.sample(candidate_keys, sample_size)
        else:
            selected_keys = list(candidate_keys)

        case_by_transition_key = {
            str(case.get("transition_key")): dict(case)
            for case in broken_cases
            if isinstance(case, dict) and isinstance(case.get("transition_key"), str)
        }
        witness_cases: List[Dict[str, Any]] = []
        for selected_key in selected_keys:
            witness_case = dict(case_by_transition_key.get(selected_key) or {})
            if not witness_case:
                transition = self._protected_transition_by_key.get(selected_key)
                if transition is None:
                    continue
                predicted_next_state_json = broken_predictions_by_key.get(selected_key)
                witness_case = self._build_failure_case_from_prediction(
                    transition=transition,
                    predicted_next_state_json=(
                        predicted_next_state_json
                        if isinstance(predicted_next_state_json, str)
                        else None
                    ),
                    transition_key=selected_key,
                )
            witness_case.pop("group_id", None)
            witness_case.pop("commit_version", None)
            witness_case["case_source"] = "previous_rejected_candidate"
            witness_cases.append(witness_case)
        return witness_cases

    def _select_regression_witness_cases(
        self,
        *,
        broken_group_to_transition_keys: Dict[str, List[str]],
        broken_cases: List[Dict[str, Any]],
        broken_predictions_by_key: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.regression_group_sample_k <= 0 or not broken_group_to_transition_keys:
            return []

        available_group_ids = sorted(
            group_id
            for group_id in broken_group_to_transition_keys.keys()
            if isinstance(group_id, str) and group_id
        )
        if not available_group_ids:
            return []

        group_limit = min(self.regression_group_sample_k, len(available_group_ids))
        if group_limit <= 0:
            return []

        if len(available_group_ids) > group_limit:
            selected_group_ids = self._prompt_rng.sample(available_group_ids, group_limit)
        else:
            selected_group_ids = list(available_group_ids)

        case_by_transition_key = {
            str(case.get("transition_key")): dict(case)
            for case in broken_cases
            if isinstance(case, dict) and isinstance(case.get("transition_key"), str)
        }
        witness_cases: List[Dict[str, Any]] = []
        for group_id in selected_group_ids:
            broken_keys_in_group = sorted(
                str(key)
                for key in (broken_group_to_transition_keys.get(group_id) or [])
                if isinstance(key, str) and key.strip()
            )
            if not broken_keys_in_group:
                continue
            transition_limit = min(
                int(self.regression_witnesses_per_group),
                len(broken_keys_in_group),
            )
            if transition_limit <= 0:
                continue
            if len(broken_keys_in_group) > transition_limit:
                selected_transition_keys = self._prompt_rng.sample(
                    broken_keys_in_group,
                    transition_limit,
                )
            else:
                selected_transition_keys = list(broken_keys_in_group)
            for selected_transition_key in selected_transition_keys:
                witness_case = dict(
                    case_by_transition_key.get(selected_transition_key) or {}
                )
                if not witness_case:
                    transition = self._protected_transition_by_key.get(
                        selected_transition_key
                    )
                    if transition is None:
                        continue
                    predicted_next_state_json = broken_predictions_by_key.get(
                        selected_transition_key
                    )
                    witness_case = self._build_failure_case_from_prediction(
                        transition=transition,
                        predicted_next_state_json=(
                            predicted_next_state_json
                            if isinstance(predicted_next_state_json, str)
                            else None
                        ),
                        transition_key=selected_transition_key,
                    )
                witness_case["group_id"] = group_id
                witness_case["commit_version"] = (
                    self._protected_commit_version_by_key.get(selected_transition_key)
                )
                witness_case["case_source"] = "previous_rejected_candidate"
                witness_cases.append(witness_case)
        return witness_cases

    def _summarize_fail_groups(
        self,
        *,
        broken_group_to_transition_keys: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for group_id in sorted(
            str(raw_group_id)
            for raw_group_id in broken_group_to_transition_keys.keys()
            if isinstance(raw_group_id, str) and raw_group_id.strip()
        ):
            transition_keys = [
                str(key)
                for key in (broken_group_to_transition_keys.get(group_id) or [])
                if isinstance(key, str) and key.strip()
            ]
            if not transition_keys:
                continue
            raw_class_id = self._group_class_id_by_group_id.get(group_id)
            group_metadata = self._protected_group_metadata_by_id.get(group_id) or {}
            raw_commit_version = (
                group_metadata.get("commit_version")
                if isinstance(group_metadata, dict)
                else None
            )
            commit_version = None
            if isinstance(raw_commit_version, str) and raw_commit_version.strip():
                normalized_commit_version = self._normalize_commit_version(
                    raw_commit_version
                )
                if normalized_commit_version != "v_unknown":
                    commit_version = normalized_commit_version
            summaries.append(
                {
                    "group_id": group_id,
                    "commit_version": commit_version,
                    "class_id": (
                        int(raw_class_id)
                        if isinstance(raw_class_id, int) and raw_class_id > 0
                        else None
                    ),
                    "transition_count": len(transition_keys),
                }
            )
        return summaries

    def _relative_output_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.output_dir).as_posix())
        except ValueError:
            return str(path)

    def _write_text_artifact(self, path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return self._relative_output_path(path)

    def _write_json_artifact(self, path: Path, payload: Dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self._relative_output_path(path)

    def _read_json_artifact(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, dict) else None

    def _clear_program_resume_group_state(self) -> None:
        self.protected_set = []
        self._protected_keys = set()
        self._protected_transition_by_key = {}
        self._protected_commit_version_by_key = {}
        self._protected_leaf_group_id_by_key = {}
        self._protected_group_metadata_by_id = {}
        self._protected_group_member_key_sets_by_id = {}
        self._protected_root_group_id_by_commit_version = {}
        self._protected_ingress_leaf_group_id_by_commit_version = {}
        self._next_group_index_by_commit_version = {}
        self._group_class_id_by_group_id = {}
        self._leaf_group_id_by_class_id = {}
        self._next_group_class_id = 1
        self._group_context_generation = 0

    def _restore_program_resume_root_groups(self) -> None:
        for version in self.repository.list_versions():
            commit_version = self._normalize_commit_version(
                getattr(version, "version_id", None)
            )
            if not self._is_class_eligible_commit_version(commit_version):
                continue
            root_group_id = f"{commit_version}:g0000"
            self._upsert_program_resume_group(
                group_id=root_group_id,
                commit_version=commit_version,
                parent_group_id=None,
            )
            self._protected_root_group_id_by_commit_version[commit_version] = (
                root_group_id
            )
            self._protected_ingress_leaf_group_id_by_commit_version.setdefault(
                commit_version,
                root_group_id,
            )

    def _commit_version_from_group_id(self, group_id: str) -> str:
        safe_group_id = str(group_id).strip()
        version_prefix, _separator, _suffix = safe_group_id.partition(":")
        return self._normalize_commit_version(version_prefix or None)

    def _upsert_program_resume_group(
        self,
        *,
        group_id: str,
        commit_version: str,
        parent_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_group_id = str(group_id).strip()
        if not safe_group_id:
            raise ValueError("group_id is required for program resume restore.")
        group = self._protected_group_metadata_by_id.get(safe_group_id)
        if group is None:
            group = {
                "group_id": safe_group_id,
                "commit_version": commit_version,
                "parent_group_id": parent_group_id,
                "child_group_ids": [],
                "member_transition_keys": [],
                "class_id": None,
                "retired_class_id": None,
                "split_program_source": None,
                "split_patch_digest": None,
                "split_target_transition_key": None,
                "split_broken_child_group_id": None,
                "split_kept_child_group_id": None,
                "split_broken_class_id": None,
                "split_kept_class_id": None,
            }
            self._protected_group_metadata_by_id[safe_group_id] = group
            self._protected_group_member_key_sets_by_id[safe_group_id] = set()
        else:
            group["commit_version"] = commit_version
            if parent_group_id is not None:
                group["parent_group_id"] = parent_group_id
        if isinstance(parent_group_id, str) and parent_group_id.strip():
            parent_group = self._protected_group_metadata_by_id.get(parent_group_id)
            if parent_group is None:
                parent_group = self._upsert_program_resume_group(
                    group_id=parent_group_id,
                    commit_version=self._commit_version_from_group_id(parent_group_id),
                )
            if parent_group is not None:
                child_group_ids = list(parent_group.get("child_group_ids") or [])
                if safe_group_id not in child_group_ids:
                    child_group_ids.append(safe_group_id)
                    parent_group["child_group_ids"] = child_group_ids
        return group

    def _apply_program_resume_split_events(
        self,
        *,
        split_events: List[Dict[str, Any]],
        candidate_source: Optional[str],
    ) -> None:
        for split_event in split_events:
            split_group_id = str(split_event.get("split_group_id") or "").strip()
            if not split_group_id:
                continue
            commit_version = self._commit_version_from_group_id(split_group_id)
            parent_group = self._upsert_program_resume_group(
                group_id=split_group_id,
                commit_version=commit_version,
            )
            broken_child_group_id = str(
                split_event.get("broken_child_group_id") or ""
            ).strip()
            kept_child_group_id = str(
                split_event.get("kept_child_group_id") or ""
            ).strip()
            if not broken_child_group_id or not kept_child_group_id:
                continue
            parent_group["split_program_source"] = (
                candidate_source
                if isinstance(candidate_source, str) and candidate_source.strip()
                else None
            )
            parent_group["split_broken_child_group_id"] = broken_child_group_id
            parent_group["split_kept_child_group_id"] = kept_child_group_id
            broken_class_id = split_event.get("broken_class_id")
            parent_group["split_broken_class_id"] = (
                int(broken_class_id)
                if isinstance(broken_class_id, int) and broken_class_id > 0
                else None
            )
            kept_class_id = split_event.get("kept_class_id")
            parent_group["split_kept_class_id"] = (
                int(kept_class_id)
                if isinstance(kept_class_id, int) and kept_class_id > 0
                else None
            )
            parent_group["child_group_ids"] = [
                broken_child_group_id,
                kept_child_group_id,
            ]

            source_class_id = split_event.get("source_class_id")
            if isinstance(source_class_id, int) and source_class_id > 0:
                parent_group["retired_class_id"] = int(source_class_id)
            parent_group["class_id"] = None

            broken_group = self._upsert_program_resume_group(
                group_id=broken_child_group_id,
                commit_version=commit_version,
                parent_group_id=split_group_id,
            )
            kept_group = self._upsert_program_resume_group(
                group_id=kept_child_group_id,
                commit_version=commit_version,
                parent_group_id=split_group_id,
            )
            if isinstance(broken_class_id, int) and broken_class_id > 0:
                broken_group["class_id"] = int(broken_class_id)
            if isinstance(kept_class_id, int) and kept_class_id > 0:
                kept_group["class_id"] = int(kept_class_id)

    def _restore_program_only_state(self) -> None:
        history_path = self.output_dir / "program_history.jsonl"
        if not history_path.exists():
            return

        self._clear_program_resume_group_state()
        self._restore_program_resume_root_groups()

        for line in history_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid program_history.jsonl row during program resume: {text!r}"
                ) from exc
            if not isinstance(row, dict):
                continue

            raw_split_events = row.get("split_events")
            split_events = (
                [dict(item) for item in raw_split_events if isinstance(item, dict)]
                if isinstance(raw_split_events, list)
                else []
            )
            if not split_events:
                continue
            applied_program_path = row.get("applied_program_path")
            if not isinstance(applied_program_path, str) or not applied_program_path.strip():
                raise ValueError(
                    "Program resume requires applied_program_path for every split event row."
                )
            applied_program = self.output_dir / applied_program_path
            if not applied_program.exists():
                raise FileNotFoundError(
                    f"Missing applied_program.py required for resume: {applied_program}"
                )
            self._apply_program_resume_split_events(
                split_events=split_events,
                candidate_source=applied_program.read_text(encoding="utf-8"),
            )

        self._sanitize_protected_group_registry()
        self._restore_program_only_saturation_baseline()

    def _restore_program_only_saturation_baseline(self) -> None:
        last_patch_step = derive_last_program_patch_step(
            source_run_dir=self.output_dir,
            through_version_id=self.current_version.version_id,
        )
        if not isinstance(last_patch_step, int):
            return
        self.saturation.reset()
        self.saturation.last_patch_global_step = max(0, int(last_patch_step))
        self.saturation.observe(global_step=0)

    def _summarize_split_event_for_bundle(
        self,
        split_event: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(split_event, dict):
            return None
        broken_transition_keys = [
            str(key)
            for key in (split_event.get("broken_transition_keys") or [])
            if isinstance(key, str) and key.strip()
        ]
        kept_transition_keys = [
            str(key)
            for key in (split_event.get("kept_transition_keys") or [])
            if isinstance(key, str) and key.strip()
        ]
        return {
            "split_group_id": split_event.get("split_group_id"),
            "source_class_id": split_event.get("source_class_id"),
            "broken_child_group_id": split_event.get("broken_child_group_id"),
            "broken_class_id": split_event.get("broken_class_id"),
            "broken_transition_count": len(broken_transition_keys),
            "kept_child_group_id": split_event.get("kept_child_group_id"),
            "kept_class_id": split_event.get("kept_class_id"),
            "kept_transition_count": len(kept_transition_keys),
        }

    def _summarize_fail_groups_for_bundle(
        self,
        regression_feedback: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(regression_feedback, dict):
            return []
        normalized_summaries: List[Dict[str, Any]] = []
        raw_fail_groups = regression_feedback.get("fail_groups")
        if isinstance(raw_fail_groups, list):
            for raw_summary in raw_fail_groups:
                if not isinstance(raw_summary, dict):
                    continue
                group_id = raw_summary.get("group_id")
                if not isinstance(group_id, str) or not group_id.strip():
                    continue
                raw_commit_version = raw_summary.get("commit_version")
                commit_version = (
                    str(raw_commit_version).strip()
                    if isinstance(raw_commit_version, str)
                    and str(raw_commit_version).strip()
                    else None
                )
                raw_class_id = raw_summary.get("class_id")
                class_id = (
                    int(raw_class_id)
                    if isinstance(raw_class_id, int) and raw_class_id > 0
                    else None
                )
                raw_transition_count = raw_summary.get("transition_count")
                transition_count = (
                    int(raw_transition_count)
                    if isinstance(raw_transition_count, int)
                    and raw_transition_count > 0
                    else 0
                )
                normalized_summaries.append(
                    {
                        "group_id": group_id,
                        "commit_version": commit_version,
                        "class_id": class_id,
                        "transition_count": transition_count,
                    }
                )
        if normalized_summaries:
            return normalized_summaries
        broken_transition_keys = [
            str(key)
            for key in (regression_feedback.get("broken_transition_keys") or [])
            if isinstance(key, str) and key.strip()
        ]
        if not broken_transition_keys:
            return []
        broken_group_to_transition_keys = self._group_broken_transition_keys_by_leaf(
            broken_transition_keys
        )
        return self._summarize_fail_groups(
            broken_group_to_transition_keys=broken_group_to_transition_keys
        )

    def _build_transition_bundle_rejection_metadata(
        self,
        *,
        reason: str,
        rejection_reasons: Optional[List[Any]] = None,
        qualified_regression_reject: bool = False,
        regression_feedback: Optional[Dict[str, Any]] = None,
        patch_rejection_artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        regression_feedback = (
            dict(regression_feedback)
            if isinstance(regression_feedback, dict)
            else {}
        )
        normalized_reasons = [
            str(item)
            for item in (rejection_reasons or [])
            if str(item).strip()
        ]
        selected_witness_artifacts: List[Dict[str, Any]] = []
        if isinstance(patch_rejection_artifacts, dict):
            for raw_artifact in (
                patch_rejection_artifacts.get("selected_regression_witness_artifacts")
                or []
            ):
                if not isinstance(raw_artifact, dict):
                    continue
                selected_witness_artifacts.append(
                    {
                        "case_label": raw_artifact.get("case_label"),
                        "transition_key": raw_artifact.get("transition_key"),
                        "group_id": raw_artifact.get("group_id"),
                        "commit_version": raw_artifact.get("commit_version"),
                        "case_source": raw_artifact.get("case_source"),
                        "world_index": raw_artifact.get("world_index"),
                        "map_name": raw_artifact.get("map_name"),
                        "previous_state_png": raw_artifact.get("previous_state_png"),
                        "actual_next_state_png": raw_artifact.get("actual_next_state_png"),
                        "predicted_next_state_png": raw_artifact.get("predicted_next_state_png"),
                        "transition_bundle_json": raw_artifact.get("transition_bundle_json"),
                    }
                )
        split_events = [
            summarized
            for summarized in (
                self._summarize_split_event_for_bundle(split_event)
                for split_event in (regression_feedback.get("split_events") or [])
            )
            if summarized is not None
        ]
        fail_groups = self._summarize_fail_groups_for_bundle(regression_feedback)
        return {
            "rejection": {
                "reason": str(reason or "").strip(),
                "rejection_reasons": normalized_reasons,
                "qualified_regression_reject": bool(qualified_regression_reject),
                "broken_group_ids": [
                    str(group_id)
                    for group_id in (regression_feedback.get("broken_group_ids") or [])
                    if isinstance(group_id, str) and group_id.strip()
                ],
                "selected_regression_group_ids": [
                    str(group_id)
                    for group_id in (
                        regression_feedback.get("selected_regression_group_ids") or []
                    )
                    if isinstance(group_id, str) and group_id.strip()
                ],
                "fail_groups": fail_groups,
                "split_events": split_events,
                "selected_regression_witness_artifacts": selected_witness_artifacts,
            }
        }

    def _attach_rejection_metadata_to_transition_bundle(
        self,
        *,
        transition_image_paths: Optional[Dict[str, str]],
        rejection_metadata: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(transition_image_paths, dict):
            return
        if not isinstance(rejection_metadata, dict) or not rejection_metadata:
            return
        bundle_relative_path = transition_image_paths.get("transition_bundle_unexplained")
        if not isinstance(bundle_relative_path, str) or not bundle_relative_path.strip():
            return
        bundle_path = self.output_dir / bundle_relative_path
        bundle_payload = self._read_json_artifact(bundle_path)
        if bundle_payload is None:
            return
        bundle_payload.update(rejection_metadata)
        self._write_json_artifact(bundle_path, bundle_payload)

    def _prepare_patch_attempt_artifact_dir(
        self,
        *,
        iteration: int,
        data_index: int,
        failure_index: int,
        unexpected_attempt: int,
    ) -> Path:
        self.patch_attempts_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = self.patch_attempts_dir / (
            f"iter{int(iteration):04d}_"
            f"data{max(1, int(data_index)):03d}_"
            f"fail{max(0, int(failure_index)):03d}_"
            f"u{max(1, int(unexpected_attempt)):02d}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _save_patch_generation_exchange_artifacts(
        self,
        *,
        artifact_dir: Path,
        generator_try: int,
        total_attempts: int,
        prompt: str,
        response_text: Optional[str],
        reasoning: Optional[str],
        error: Optional[str],
    ) -> Dict[str, Any]:
        prefix = f"generator_try_{max(1, int(generator_try)):02d}"
        record: Dict[str, Any] = {
            "generator_try": max(1, int(generator_try)),
            "total_attempts": max(1, int(total_attempts)),
            "status": "error" if error else "success",
            "prompt_path": self._write_text_artifact(
                artifact_dir / f"{prefix}_input_prompt.txt",
                prompt if isinstance(prompt, str) else "",
            ),
        }
        if isinstance(response_text, str):
            record["output_path"] = self._write_text_artifact(
                artifact_dir / f"{prefix}_output.txt",
                response_text,
            )
        if isinstance(reasoning, str) and reasoning.strip():
            record["reasoning_path"] = self._write_text_artifact(
                artifact_dir / f"{prefix}_reasoning.txt",
                reasoning,
            )
        if isinstance(error, str) and error.strip():
            record["error_path"] = self._write_text_artifact(
                artifact_dir / f"{prefix}_error.txt",
                error,
            )
        return record

    def _save_patch_attempt_applied_program(
        self,
        *,
        artifact_dir: Path,
        source: str,
    ) -> str:
        return self._write_text_artifact(
            artifact_dir / "applied_program.py",
            source if isinstance(source, str) else "",
        )

    def _save_transition_artifact_bundle(
        self,
        *,
        artifact_dir: Path,
        basename: str,
        transition: Transition,
        predicted_next_state_json: Optional[str] = None,
        prediction_error: Optional[Dict[str, Optional[str]]] = None,
        previous_title: Optional[str] = None,
        actual_next_title: str = "Actual Next State",
        predicted_next_title: str = "Predicted Next State",
        bundle_extra_fields: Optional[Dict[str, Any]] = None,
        previous_filename: Optional[str] = None,
        actual_next_filename: Optional[str] = None,
        predicted_next_filename: Optional[str] = None,
        bundle_filename: Optional[str] = None,
        previous_outpath: Optional[Path] = None,
        actual_next_outpath: Optional[Path] = None,
        predicted_next_outpath: Optional[Path] = None,
        bundle_outpath: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        resolved_previous_filename = previous_filename or f"{basename}_previous_state.png"
        resolved_actual_next_filename = (
            actual_next_filename or f"{basename}_actual_next_state.png"
        )
        resolved_predicted_next_filename = (
            predicted_next_filename or f"{basename}_predicted_next_state.png"
        )
        resolved_bundle_filename = bundle_filename or f"{basename}_bundle.json"
        try:
            previous_state = parse_state_json(transition.state)
            actual_next_state = parse_state_json(transition.next_state)
        except ValueError as e:
            bundle_payload = {
                "action": transition.action,
                "previous_state_raw": transition.state,
                "actual_next_state_raw": transition.next_state,
                "parse_error": str(e),
            }
            bundle_payload.update(self._build_transition_artifact_context(transition=transition))
            if bundle_extra_fields:
                bundle_payload.update(bundle_extra_fields)
            bundle_path = bundle_outpath or (artifact_dir / resolved_bundle_filename)
            return {
                "transition_bundle_json": self._write_json_artifact(
                    bundle_path,
                    bundle_payload,
                )
            }

        previous_render_state = dict(previous_state)
        actual_next_render_state = dict(actual_next_state)

        predicted_next_state = None
        predicted_next_render_state = None
        predicted_parse_error = None
        if isinstance(predicted_next_state_json, str) and predicted_next_state_json.strip():
            try:
                predicted_next_state = parse_state_json(predicted_next_state_json)
                predicted_next_render_state = resolve_predicted_visual_state(
                    predicted_logical_state=predicted_next_state,
                    actual_logical_state=actual_next_state,
                    actual_visual_state=actual_next_render_state,
                )
            except ValueError as e:
                predicted_parse_error = str(e)

        artifact_context = self._build_transition_artifact_context(
            transition=transition,
            previous_state=previous_state,
            actual_next_state=actual_next_state,
            predicted_next_state=predicted_next_state,
        )
        previous_snapshot_state = self._apply_transition_artifact_context(
            state=previous_render_state,
            context=artifact_context,
        )
        actual_next_snapshot_state = self._apply_transition_artifact_context(
            state=actual_next_render_state,
            context=artifact_context,
        )
        predicted_next_snapshot_state = (
            self._apply_transition_artifact_context(
                state=predicted_next_render_state,
                context=artifact_context,
            )
            if isinstance(predicted_next_render_state, dict)
            else None
        )

        prev_path = previous_outpath or (artifact_dir / resolved_previous_filename)
        actual_next_path = actual_next_outpath or (
            artifact_dir / resolved_actual_next_filename
        )
        predicted_next_path = predicted_next_outpath or (
            artifact_dir / resolved_predicted_next_filename
        )
        bundle_json_path = bundle_outpath or (artifact_dir / resolved_bundle_filename)

        prev_path.parent.mkdir(parents=True, exist_ok=True)
        actual_next_path.parent.mkdir(parents=True, exist_ok=True)
        predicted_next_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_json_path.parent.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, Any] = {}
        if self._render_state_snapshot(
            state=previous_snapshot_state,
            title=previous_title or f"Previous State | action={transition.action}",
            outpath=prev_path,
            action_name=str(transition.action),
        ):
            saved["previous_state_png"] = self._relative_output_path(prev_path)
        if self._render_state_snapshot(
            state=actual_next_snapshot_state,
            title=actual_next_title,
            outpath=actual_next_path,
            action_name=str(transition.action),
        ):
            saved["actual_next_state_png"] = self._relative_output_path(actual_next_path)
        if predicted_next_snapshot_state is not None and self._render_state_snapshot(
            state=predicted_next_snapshot_state,
            title=predicted_next_title,
            outpath=predicted_next_path,
            action_name=str(transition.action),
        ):
            saved["predicted_next_state_png"] = self._relative_output_path(
                predicted_next_path
            )

        bundle_payload = {
            "action": transition.action,
            "previous_state": previous_state,
            "actual_next_state": actual_next_state,
            "predicted_next_state": predicted_next_state,
            "predicted_next_state_raw": predicted_next_state_json,
            "predicted_parse_error": predicted_parse_error,
            "prediction_error": prediction_error,
        }
        if previous_render_state != previous_state:
            bundle_payload["previous_state_visual"] = previous_render_state
        if actual_next_render_state != actual_next_state:
            bundle_payload["actual_next_state_visual"] = actual_next_render_state
        if (
            isinstance(predicted_next_render_state, dict)
            and predicted_next_render_state != predicted_next_state
        ):
            bundle_payload["predicted_next_state_visual"] = predicted_next_render_state
        bundle_payload.update(artifact_context)
        if bundle_extra_fields:
            bundle_payload.update(bundle_extra_fields)
        saved["transition_bundle_json"] = self._write_json_artifact(
            bundle_json_path,
            bundle_payload,
        )
        return saved or None

    def _save_failure_case_artifacts(
        self,
        *,
        artifact_dir: Path,
        basename: str,
        transition: Transition,
        failure_case: Dict[str, Any],
        case_label: str,
        plain_stage_titles: bool = False,
    ) -> Dict[str, Any]:
        error_message = failure_case.get("error")
        prediction_error = None
        if error_message is not None:
            prediction_error = {
                "message": str(error_message),
            }
        previous_title = (
            "Previous State"
            if plain_stage_titles
            else f"{case_label} Previous State | action={transition.action}"
        )
        actual_next_title = (
            "Expected Next State"
            if plain_stage_titles
            else f"{case_label} Actual Next State"
        )
        predicted_next_title = (
            "Predicted Next State"
            if plain_stage_titles
            else f"{case_label} Predicted Next State"
        )
        saved = self._save_transition_artifact_bundle(
            artifact_dir=artifact_dir,
            basename=basename,
            transition=transition,
            predicted_next_state_json=(
                str(failure_case.get("predicted_next_state"))
                if isinstance(failure_case.get("predicted_next_state"), str)
                else None
            ),
            prediction_error=prediction_error,
            previous_title=previous_title,
            actual_next_title=actual_next_title,
            predicted_next_title=predicted_next_title,
            bundle_extra_fields={
                "failure_case": dict(failure_case),
                "case_label": case_label,
            },
        )
        record: Dict[str, Any] = {
            "case_label": case_label,
            "transition_key": failure_case.get("transition_key"),
            "group_id": failure_case.get("group_id"),
            "commit_version": failure_case.get("commit_version"),
            "case_source": failure_case.get("case_source"),
        }
        record.update(self._build_transition_artifact_context(transition=transition))
        if saved:
            record.update(saved)
        return record

    def _save_patch_rejection_artifacts(
        self,
        *,
        artifact_dir: Path,
        target: Transition,
        target_failure_case: Dict[str, Any],
        regression_witness_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        analysis_dir = artifact_dir / "rejection_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        target_artifacts = self._save_failure_case_artifacts(
            artifact_dir=analysis_dir,
            basename="target",
            transition=target,
            failure_case=target_failure_case,
            case_label="Target",
        )

        regression_artifacts: List[Dict[str, Any]] = []
        for index, witness_case in enumerate(regression_witness_cases, 1):
            transition_key = witness_case.get("transition_key")
            transition = (
                self._protected_transition_by_key.get(str(transition_key))
                if transition_key is not None
                else None
            )
            if transition is None:
                regression_artifacts.append(
                    {
                        "case_label": f"Regression Witness {index}",
                        "transition_key": transition_key,
                        "group_id": witness_case.get("group_id"),
                        "commit_version": witness_case.get("commit_version"),
                        "case_source": witness_case.get("case_source"),
                        "world_index": witness_case.get("world_index"),
                        "map_name": witness_case.get("map_name"),
                        "error": "missing_transition_for_witness_case",
                    }
                )
                continue
            regression_artifacts.append(
                self._save_failure_case_artifacts(
                    artifact_dir=analysis_dir,
                    basename=f"regression_{index:02d}",
                    transition=transition,
                    failure_case=witness_case,
                    case_label=f"Regression Witness {index}",
                    plain_stage_titles=True,
                )
            )

        return {
            "analysis_dir": self._relative_output_path(analysis_dir),
            "target_case_artifacts": target_artifacts,
            "selected_regression_witness_artifacts": regression_artifacts,
        }

    def _sanitize_transition_artifact_token(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        safe = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in text
        ).strip("_")
        return safe or None

    def _resolve_transition_artifact_status(self, value: Any) -> Optional[str]:
        token = self._sanitize_transition_artifact_token(value)
        if token is None:
            return None
        if token.startswith("version_context_"):
            token = token[len("version_context_") :]
        return token or None

    def _resolve_transition_artifact_program_version(self, value: Any) -> Optional[str]:
        token = self._sanitize_transition_artifact_token(value)
        if token is None:
            return None
        if not token.lower().startswith("v"):
            return None
        return token

    def _build_target_transition_artifact_prefix(
        self,
        *,
        iteration: int,
        data_index: int,
        failure_index: int,
    ) -> str:
        safe_data_index = max(1, int(data_index))
        safe_failure_index = max(0, int(failure_index))
        return (
            f"iter{int(iteration):04d}_data{safe_data_index:03d}_"
            f"fail{safe_failure_index:03d}"
        )

    def _remember_collected_transition_step_indices(
        self,
        *,
        transitions: List[Transition],
        collect_diag: Dict[str, Any],
    ) -> None:
        if not transitions:
            return
        final_global_step = self._resolve_explorer_global_step(collect_diag)
        can_infer_precise_steps = (
            isinstance(final_global_step, int)
            and final_global_step > 0
            and final_global_step >= len(transitions)
        )
        if can_infer_precise_steps:
            start_step = int(final_global_step) - len(transitions) + 1
        else:
            start_step = None
        for offset, transition in enumerate(transitions):
            transition_key = self._transition_key(transition)
            if transition_key in self._transition_step_index_by_key:
                continue
            if isinstance(start_step, int):
                resolved_step_index = start_step + int(offset)
            else:
                self._next_fallback_transition_step_index += 1
                resolved_step_index = int(self._next_fallback_transition_step_index)
            resolved_step_index = max(1, int(resolved_step_index))
            self._transition_step_index_by_key[transition_key] = resolved_step_index
            self._next_fallback_transition_step_index = max(
                int(self._next_fallback_transition_step_index),
                resolved_step_index,
            )

    def _resolve_transition_step_index(
        self,
        *,
        transition_key: Optional[str],
        fallback_index: Optional[int] = None,
    ) -> int:
        if isinstance(transition_key, str) and transition_key.strip():
            existing = self._transition_step_index_by_key.get(transition_key.strip())
            if isinstance(existing, int) and existing > 0:
                return int(existing)
        if isinstance(fallback_index, int) and fallback_index > 0:
            resolved_fallback = int(fallback_index)
        else:
            self._next_fallback_transition_step_index += 1
            resolved_fallback = int(self._next_fallback_transition_step_index)
        resolved_fallback = max(1, int(resolved_fallback))
        if isinstance(transition_key, str) and transition_key.strip():
            self._transition_step_index_by_key[transition_key.strip()] = resolved_fallback
        self._next_fallback_transition_step_index = max(
            int(self._next_fallback_transition_step_index),
            resolved_fallback,
        )
        return resolved_fallback

    def _build_new_transition_step_artifact_dir(
        self,
        *,
        step_index: int,
    ) -> Path:
        safe_step_index = max(1, int(step_index))
        return self.new_transition_images_dir / f"step{safe_step_index:03d}"

    def _build_new_transition_fail_artifact_dir(
        self,
        *,
        step_index: int,
        failure_index: int,
    ) -> Path:
        step_dir = self._build_new_transition_step_artifact_dir(
            step_index=step_index,
        )
        safe_failure_index = max(0, int(failure_index))
        if safe_failure_index <= 0:
            return step_dir
        return step_dir / f"fail{safe_failure_index:03d}"

    def _merge_transition_image_paths(
        self,
        *path_maps: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        merged: Dict[str, str] = {}
        for path_map in path_maps:
            if not isinstance(path_map, dict):
                continue
            for key, value in path_map.items():
                if isinstance(key, str) and key and isinstance(value, str) and value:
                    merged[key] = value
        return merged or None

    def _save_target_transition_images(
        self,
        transition: Transition,
        iteration: int,
        data_index: int,
        failure_index: int,
        patch_digest: Optional[str],
        step_index: Optional[int] = None,
        predicted_next_state_json: Optional[str] = None,
        target_prediction_error: Optional[Dict[str, Optional[str]]] = None,
        stem_tag: str = "target_mismatch",
    ) -> Optional[Dict[str, str]]:
        if not self.save_new_transition_images:
            return None

        self.new_transition_images_dir.mkdir(parents=True, exist_ok=True)

        stem = self._build_target_transition_artifact_prefix(
            iteration=iteration,
            data_index=data_index,
            failure_index=failure_index,
        )
        safe_data_index = max(1, int(data_index))
        safe_failure_index = max(0, int(failure_index))
        safe_step_index = (
            max(1, int(step_index))
            if isinstance(step_index, int) and int(step_index) > 0
            else safe_data_index
        )
        step_artifact_dir = self._build_new_transition_step_artifact_dir(
            step_index=safe_step_index,
        )
        fail_artifact_dir = self._build_new_transition_fail_artifact_dir(
            step_index=safe_step_index,
            failure_index=safe_failure_index,
        )
        status_tag = self._resolve_transition_artifact_status(stem_tag)
        variant_artifact_dir = (
            step_artifact_dir if status_tag == "explained" else fail_artifact_dir
        )
        version_tag = self._resolve_transition_artifact_program_version(patch_digest)
        predicted_name_parts = ["predicted_next_state"]
        if isinstance(status_tag, str) and status_tag:
            predicted_name_parts.append(status_tag)
        if isinstance(version_tag, str) and version_tag:
            predicted_name_parts.append(version_tag)
        bundle_name_parts = ["bundle"]
        if isinstance(status_tag, str) and status_tag:
            bundle_name_parts.append(status_tag)
        saved = self._save_transition_artifact_bundle(
            artifact_dir=variant_artifact_dir,
            basename=stem,
            transition=transition,
            predicted_next_state_json=predicted_next_state_json,
            prediction_error=target_prediction_error,
            previous_title=f"Previous State | action={transition.action}",
            actual_next_title="Expected Next State",
            predicted_next_title="Predicted Next State",
            bundle_extra_fields={
                "iteration": int(iteration),
                "data_index": safe_data_index,
                "step_index": safe_step_index,
                "failure_index": safe_failure_index,
            },
            previous_filename="previous_state.png",
            actual_next_filename="expected_next_state.png",
            predicted_next_filename="_".join(predicted_name_parts) + ".png",
            bundle_filename="_".join(bundle_name_parts) + ".json",
            previous_outpath=step_artifact_dir / "previous_state.png",
            actual_next_outpath=step_artifact_dir / "expected_next_state.png",
            predicted_next_outpath=variant_artifact_dir
            / ("_".join(predicted_name_parts) + ".png"),
            bundle_outpath=variant_artifact_dir / ("_".join(bundle_name_parts) + ".json"),
        )
        if not saved:
            return None

        normalized_saved: Dict[str, str] = {}
        previous_png = saved.get("previous_state_png")
        if isinstance(previous_png, str):
            normalized_saved["previous_state_png"] = previous_png
        actual_png = saved.get("actual_next_state_png")
        if isinstance(actual_png, str):
            normalized_saved["expected_next_state_png"] = actual_png
        predicted_png = saved.get("predicted_next_state_png")
        if isinstance(predicted_png, str):
            normalized_saved["predicted_next_state_png"] = predicted_png
        bundle_json = saved.get("transition_bundle_json")
        if isinstance(bundle_json, str):
            normalized_saved["transition_bundle_json"] = bundle_json
        if isinstance(status_tag, str) and status_tag:
            if isinstance(predicted_png, str):
                normalized_saved[f"predicted_next_state_{status_tag}"] = predicted_png
            if isinstance(bundle_json, str):
                normalized_saved[f"transition_bundle_{status_tag}"] = bundle_json
        return normalized_saved or None

    def _resolve_unique_transition_artifact_path(
        self,
        filename: str,
        *,
        target_dir: Optional[Path] = None,
    ) -> Path:
        target_dir = target_dir or self.new_transition_images_dir
        candidate = target_dir / filename
        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix
        index = 1
        while True:
            trial = target_dir / f"{stem}_{index:02d}{suffix}"
            if not trial.exists():
                return trial
            index += 1

    def _render_state_snapshot(
        self,
        state: Dict[str, Any],
        title: str,
        outpath: Path,
        *,
        action_name: Optional[str] = None,
    ) -> bool:
        return self._artifact_renderer.render_state_snapshot(
            state=state,
            title=title,
            outpath=outpath,
            action_name=action_name,
            visual_config=self._visualization_config,
        )

    def _build_transition_artifact_context(
        self,
        *,
        transition: Transition,
        previous_state: Optional[Dict[str, Any]] = None,
        actual_next_state: Optional[Dict[str, Any]] = None,
        predicted_next_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if isinstance(transition.world_index, int) and int(transition.world_index) > 0:
            context["world_index"] = int(transition.world_index)
        map_name = self._resolve_transition_artifact_map_name(
            transition=transition,
            previous_state=previous_state,
            actual_next_state=actual_next_state,
            predicted_next_state=predicted_next_state,
        )
        if map_name:
            context["map_name"] = map_name
        return context

    def _resolve_transition_artifact_map_name(
        self,
        *,
        transition: Transition,
        previous_state: Optional[Dict[str, Any]] = None,
        actual_next_state: Optional[Dict[str, Any]] = None,
        predicted_next_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if isinstance(transition.map_name, str) and transition.map_name.strip():
            return str(transition.map_name).strip()
        for state in (previous_state, actual_next_state, predicted_next_state):
            if not isinstance(state, dict):
                continue
            for key in ("map_name", "scenario_type", "requested_scenario_type"):
                value = state.get(key)
                if isinstance(value, str) and value.strip():
                    return str(value).strip()
        return None

    def _apply_transition_artifact_context(
        self,
        *,
        state: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(state, dict):
            return None
        if not context:
            return dict(state)
        enriched = dict(state)
        world_index = context.get("world_index")
        if isinstance(world_index, int) and int(world_index) > 0:
            enriched["world_index"] = int(world_index)
        map_name = context.get("map_name")
        if isinstance(map_name, str) and map_name.strip():
            enriched["map_name"] = str(map_name).strip()
        return enriched

    def _resolve_visualization_config(self, explorer: Any) -> Dict[str, Any]:
        env = getattr(explorer, "env", None)
        getter = getattr(env, "get_visualization_config", None)
        if callable(getter):
            try:
                resolved = getter()
                if isinstance(resolved, dict):
                    return dict(resolved)
            except Exception:
                return {}
        raw_config = getattr(env, "visualization_config", None)
        return dict(raw_config) if isinstance(raw_config, dict) else {}

    def _resolve_attempt_data_index(self, iteration: int, transition_key: str) -> int:
        iter_index = int(iteration)
        per_iter = self._data_index_by_iteration_and_key.setdefault(iter_index, {})
        existing = per_iter.get(transition_key)
        if isinstance(existing, int) and existing > 0:
            return existing

        next_index = int(self._next_data_index_by_iteration.get(iter_index, 0)) + 1
        self._next_data_index_by_iteration[iter_index] = next_index
        per_iter[transition_key] = next_index
        return next_index

    def _next_failure_index(self, iteration: int, transition_key: str) -> int:
        iter_index = int(iteration)
        per_iter = self._failure_index_by_iteration_and_key.setdefault(iter_index, {})
        current = per_iter.get(transition_key)
        current_index = int(current) if isinstance(current, int) and current > 0 else 0
        return current_index + 1

    def _record_failure_index(
        self,
        iteration: int,
        transition_key: str,
        failure_index: int,
    ) -> None:
        if not isinstance(failure_index, int) or failure_index <= 0:
            return
        iter_index = int(iteration)
        per_iter = self._failure_index_by_iteration_and_key.setdefault(iter_index, {})
        current = per_iter.get(transition_key)
        current_index = int(current) if isinstance(current, int) and current > 0 else 0
        per_iter[transition_key] = max(current_index, int(failure_index))

    def _resolve_attempt_retry_counts(self, failure_index: Any) -> Tuple[int, int]:
        if isinstance(failure_index, int) and failure_index > 0:
            attempt = int(failure_index)
        else:
            attempt = 1
        return attempt, max(0, attempt - 1)

    def _format_attempt_retry_fields(self, failure_index: Any) -> str:
        attempt, _ = self._resolve_attempt_retry_counts(failure_index)
        max_attempts = int(self.max_patch_attempts_per_target)
        return f"attempt={attempt}/{max_attempts}"

    def _transition_key(self, transition: Transition) -> str:
        world_scope = canonical_graph_world_scope(
            world_index=transition.world_index,
            map_name=transition.map_name,
        )
        cache_key = (
            str(world_scope),
            str(transition.state_key),
            str(transition.action),
        )
        cached = self._transition_key_cache.get(cache_key)
        if cached is not None:
            return cached
        transition_key = canonical_graph_edge_identity_key_from_fields(
            world_identity=world_scope,
            state=transition.state_key,
            action=transition.action,
        )
        self._transition_key_cache[cache_key] = transition_key
        return transition_key

    def _resolve_canonical_state_json(self, state_json: Any) -> str:
        if not isinstance(state_json, str):
            return ""
        text = state_json.strip()
        if not text:
            return ""
        try:
            return dump_state_json(parse_state_json(text))
        except ValueError:
            state_id = self.state_store.lookup_state_id(text)
            if isinstance(state_id, int):
                return self.state_store.state_json(state_id)
        return text

    def _current_source_digest(self) -> str:
        return hashlib.sha1(str(self.current_source).encode("utf-8")).hexdigest()

    def _consume_explorer_current_source_transition_assessments(self) -> None:
        consumer = getattr(
            self.explorer,
            "consume_current_source_transition_assessments",
            None,
        )
        if not callable(consumer):
            return
        try:
            raw_assessments = consumer()
        except Exception:
            return
        if not isinstance(raw_assessments, dict):
            return
        current_version_id = getattr(self.current_version, "version_id", None)
        current_version_text = (
            str(current_version_id).strip()
            if isinstance(current_version_id, str) and current_version_id.strip()
            else None
        )
        current_source_digest = self._current_source_digest()
        for key, raw_assessment in raw_assessments.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(raw_assessment, dict):
                continue
            if raw_assessment.get("current_version_id") != current_version_text:
                continue
            if raw_assessment.get("current_source_digest") != current_source_digest:
                continue
            self._current_source_transition_assessments[str(key)] = dict(raw_assessment)

    def _assessment_matches_current_source(self, assessment: Any) -> bool:
        if not isinstance(assessment, dict):
            return False
        current_version_id = getattr(self.current_version, "version_id", None)
        current_version_text = (
            str(current_version_id).strip()
            if isinstance(current_version_id, str) and current_version_id.strip()
            else None
        )
        if assessment.get("current_version_id") != current_version_text:
            return False
        if assessment.get("current_source_digest") != self._current_source_digest():
            return False
        return True

    def _clone_sandbox_error(self, error: Optional[SandboxError]) -> Optional[SandboxError]:
        if error is None:
            return None
        return SandboxError(
            phase=str(error.phase),
            message=str(error.message),
            exception_type=(
                str(error.exception_type)
                if isinstance(error.exception_type, str)
                else None
            ),
            traceback_text=(
                str(error.traceback_text)
                if isinstance(error.traceback_text, str)
                else None
            ),
        )

    def _empty_current_source_transition_assessment(
        self,
        transition_key: str,
    ) -> Dict[str, Any]:
        current_version_id = getattr(self.current_version, "version_id", None)
        current_version_text = (
            str(current_version_id).strip()
            if isinstance(current_version_id, str) and current_version_id.strip()
            else None
        )
        assessment: Dict[str, Any] = {
            "transition_key": transition_key,
            "current_version_id": current_version_text,
            "current_source_digest": self._current_source_digest(),
            "current_explains": False,
            "predicted_next_state_json": None,
            "prediction_error": None,
        }
        return assessment

    def _current_source_transition_assessment_from_record(
        self,
        *,
        transition_key: str,
        record: Optional[PredictionRecord],
        fallback_error: Optional[SandboxError] = None,
    ) -> Dict[str, Any]:
        assessment = self._empty_current_source_transition_assessment(transition_key)
        error = record.error if record is not None else fallback_error
        if record is not None:
            assessment["current_explains"] = (
                bool(record.is_correct) and record.error is None
            )
            assessment["predicted_next_state_json"] = record.predicted_canonical
        assessment["prediction_error"] = self._clone_sandbox_error(error)
        return assessment

    def _current_source_transition_assessment_from_loose_record(
        self,
        *,
        transition_key: str,
        record: Any,
        fallback_error: Optional[SandboxError] = None,
    ) -> Dict[str, Any]:
        if isinstance(record, PredictionRecord):
            return self._current_source_transition_assessment_from_record(
                transition_key=transition_key,
                record=record,
                fallback_error=fallback_error,
            )
        assessment = self._empty_current_source_transition_assessment(transition_key)
        if record is not None:
            record_error = getattr(record, "error", None)
            assessment["current_explains"] = (
                bool(getattr(record, "is_correct", False)) and record_error is None
            )
            predicted = getattr(record, "predicted_canonical", None)
            if isinstance(predicted, str):
                assessment["predicted_next_state_json"] = predicted
            if isinstance(record_error, SandboxError):
                assessment["prediction_error"] = self._clone_sandbox_error(record_error)
        elif isinstance(fallback_error, SandboxError):
            assessment["prediction_error"] = self._clone_sandbox_error(fallback_error)
        return assessment

    def _refresh_current_source_transition_assessments_with_evaluator(
        self,
        transitions: List[Transition],
    ) -> Dict[str, Dict[str, Any]]:
        safe_transitions = list(transitions or [])
        if not safe_transitions:
            return {}
        evaluate_programs = getattr(self.evaluator, "evaluate_programs", None)
        if callable(evaluate_programs):
            batch = evaluate_programs(
                programs=[
                    ProgramEvaluationTask(
                        label="current",
                        source=self.current_source,
                    )
                ],
                transitions=safe_transitions,
                collect_records=True,
            )
            evaluation = batch.first().evaluation
        else:
            evaluate_transition = getattr(self.evaluator, "evaluate_transition", None)
            if callable(evaluate_transition):
                records: List[PredictionRecord] = []
                correct_count = 0
                runtime_error_count = 0
                for index, transition in enumerate(safe_transitions):
                    result = evaluate_transition(self.current_source, transition)
                    result_error = getattr(result, "error", None)
                    sandbox_error = (
                        result_error if isinstance(result_error, SandboxError) else None
                    )
                    is_correct = bool(getattr(result, "is_correct", False))
                    if is_correct and sandbox_error is None:
                        correct_count += 1
                    if result_error is not None and sandbox_error is None:
                        runtime_error_count += 1
                    predicted_canonical = getattr(
                        result,
                        "predicted_canonical",
                        None,
                    )
                    records.append(
                        PredictionRecord(
                            index=index,
                            transition=transition,
                            expected_canonical=str(transition.next_state),
                            predicted_canonical=(
                                predicted_canonical
                                if isinstance(predicted_canonical, str)
                                else None
                            ),
                            is_correct=is_correct,
                            error=sandbox_error,
                        )
                    )
                evaluation = ProgramEvaluation(
                    accuracy=(
                        float(correct_count) / float(len(safe_transitions))
                        if safe_transitions
                        else 0.0
                    ),
                    correct_count=correct_count,
                    total_count=len(safe_transitions),
                    runtime_error_count=runtime_error_count,
                    compile_errors=[],
                    records=records,
                )
            else:
                evaluate_source = getattr(self.evaluator, "evaluate_source", None)
                if not callable(evaluate_source):
                    raise AttributeError(
                        "evaluator must provide evaluate_programs, "
                        "evaluate_transition, or evaluate_source."
                    )
                evaluation = evaluate_source(
                    self.current_source,
                    safe_transitions,
                    progress_callback=None,
                )
        records_by_index = {
            int(getattr(record, "index", index)): record
            for index, record in enumerate(getattr(evaluation, "records", []) or [])
            if record is not None
        }
        fallback_error = (
            evaluation.compile_errors[0] if evaluation.compile_errors else None
        )
        refreshed: Dict[str, Dict[str, Any]] = {}
        for index, transition in enumerate(safe_transitions):
            transition_key = self._transition_key(transition)
            assessment = self._current_source_transition_assessment_from_loose_record(
                transition_key=transition_key,
                record=records_by_index.get(int(index)),
                fallback_error=fallback_error,
            )
            self._current_source_transition_assessments[transition_key] = dict(
                assessment
            )
            refreshed[transition_key] = dict(assessment)
        return refreshed

    def _ensure_current_source_transition_assessments(
        self,
        transitions: Optional[List[Transition]] = None,
        *,
        refresh_with_evaluator: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        safe_transitions = list(transitions or [])
        if refresh_with_evaluator:
            return self._refresh_current_source_transition_assessments_with_evaluator(
                safe_transitions
            )
        elif not safe_transitions:
            self._consume_explorer_current_source_transition_assessments()

        transition_pairs = [
            (transition, self._transition_key(transition))
            for transition in safe_transitions
        ]
        resolved: Dict[str, Dict[str, Any]] = {}
        for _transition, transition_key in transition_pairs:
            assessment = self._current_source_transition_assessments.get(transition_key)
            if not self._assessment_matches_current_source(assessment):
                assessment = self._empty_current_source_transition_assessment(
                    transition_key
                )
            resolved[str(transition_key)] = dict(assessment)
        return resolved

    def _refresh_transition_state_after_program_update(self) -> None:
        self._current_source_transition_assessments = {}
        self._last_program_context_payload_signature = None
        self._sync_explorer_program_context(
            emit_log=False,
        )
        self._refresh_active_pending_assessments_after_program_update()

    def _refresh_active_pending_assessments_after_program_update(self) -> None:
        pending_transitions = self._active_collect_batch_unprotected_transitions()
        if not pending_transitions:
            return
        assessments = self._ensure_current_source_transition_assessments(
            pending_transitions,
            refresh_with_evaluator=True,
        )
        explained_transitions: List[Transition] = []
        for transition in pending_transitions:
            transition_key = self._transition_key(transition)
            assessment = assessments.get(transition_key)
            if not self._assessment_matches_current_source(assessment):
                continue
            if assessment.get("current_explains") is not True:
                continue
            explained_transitions.append(transition)
        if explained_transitions:
            self._refresh_canonical_group_assignments_from_cached_assessments(
                transitions=explained_transitions,
                allow_classification=True,
            )

    def _sandbox_error_from_assessment(self, raw_error: Any) -> Optional[SandboxError]:
        if not isinstance(raw_error, SandboxError):
            return None
        return SandboxError(
            phase=str(raw_error.phase),
            message=str(raw_error.message),
            exception_type=(
                str(raw_error.exception_type)
                if isinstance(raw_error.exception_type, str)
                else None
            ),
            traceback_text=(
                str(raw_error.traceback_text)
                if isinstance(raw_error.traceback_text, str)
                else None
            ),
        )

    def _cached_current_source_target_evaluation(
        self,
        target: Transition,
    ) -> Optional[ProgramEvaluation]:
        target_key = self._transition_key(target)
        assessment = self._current_source_transition_assessments.get(target_key)
        if not self._assessment_matches_current_source(assessment):
            self._current_source_transition_assessments.pop(target_key, None)
            return None

        prediction_error = self._sandbox_error_from_assessment(
            assessment.get("prediction_error")
        )
        compile_errors = (
            [prediction_error]
            if prediction_error is not None
            and getattr(prediction_error, "phase", None) in {"parse", "ast_validate", "compile"}
            else []
        )
        runtime_error_count = (
            1 if prediction_error is not None and not compile_errors else 0
        )
        current_explains = bool(assessment.get("current_explains"))
        predicted_next_state_json = assessment.get("predicted_next_state_json")
        record = PredictionRecord(
            index=0,
            transition=target,
            expected_canonical=str(target.next_state),
            predicted_canonical=(
                str(predicted_next_state_json)
                if isinstance(predicted_next_state_json, str)
                else None
            ),
            is_correct=current_explains,
            error=prediction_error,
        )
        return ProgramEvaluation(
            accuracy=1.0 if current_explains else 0.0,
            correct_count=1 if current_explains else 0,
            total_count=1,
            runtime_error_count=runtime_error_count,
            compile_errors=compile_errors,
            records=[record],
        )

    def _missing_current_source_target_evaluation(
        self,
        target: Transition,
    ) -> ProgramEvaluation:
        return ProgramEvaluation(
            accuracy=0.0,
            correct_count=0,
            total_count=1,
            runtime_error_count=0,
            compile_errors=[],
            records=[
                PredictionRecord(
                    index=0,
                    transition=target,
                    expected_canonical=str(target.next_state),
                    predicted_canonical=None,
                    is_correct=False,
                    error=None,
                )
            ],
        )

    def _evaluate_current_source_target(self, target: Transition) -> ProgramEvaluation:
        self._ensure_current_source_transition_assessments([target])
        cached = self._cached_current_source_target_evaluation(target)
        if cached is not None:
            return cached
        self._ensure_current_source_transition_assessments(
            [target],
            refresh_with_evaluator=True,
        )
        cached = self._cached_current_source_target_evaluation(target)
        if cached is not None:
            return cached
        return self._missing_current_source_target_evaluation(target)

    def _collect_explorer_diagnostics(self) -> Dict[str, Any]:
        default_name = self.explorer.__class__.__name__
        getter = getattr(self.explorer, "get_diagnostics", None)
        if not callable(getter):
            return {"name": default_name}

        try:
            raw = getter()
        except Exception as e:  # noqa: BLE001
            return {"name": default_name, "diagnostics_error": str(e)}

        if not isinstance(raw, dict):
            return {"name": default_name, "diagnostics_error": "non_mapping_diagnostics"}

        sanitized = self._sanitize_log_value(raw)
        if not isinstance(sanitized, dict):
            return {"name": default_name, "diagnostics_error": "non_mapping_diagnostics"}

        name = sanitized.get("name")
        if not isinstance(name, str) or not name.strip():
            sanitized["name"] = default_name
        return sanitized

    def _sanitize_log_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, dict):
            return {
                str(key): self._sanitize_log_value(row_value)
                for key, row_value in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_log_value(row_value) for row_value in value]

        scalar = getattr(value, "item", None)
        if callable(scalar):
            try:
                return self._sanitize_log_value(scalar())
            except Exception:  # noqa: BLE001
                pass
        return str(value)

    def _log_iteration(
        self,
        phase: str,
        collected_count: int,
        added_count: int,
        eval_count: int,
        failure_count: int,
        candidate_count: int,
        accepted: bool,
        delta_acc: float,
        current_acc: float,
        best_acc: float,
        regressions: int,
        decision_reason: str,
        patch_digest: Optional[str],
        target_action: Optional[str],
        epoch_index: int,
        epoch_budget_size: int,
        epoch_budget_remaining: int,
        no_new_canonical_rounds: int,
        max_no_new_canonical_rounds: int,
        explorer_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        saturation_status = self.saturation.get_status()
        timestamp = datetime.now().isoformat()
        epoch_done, epoch_total, epoch_ratio = self._compute_epoch_progress(
            epoch_budget_size=epoch_budget_size,
            epoch_budget_remaining=epoch_budget_remaining,
        )
        flow_stages = self._derive_flow_stages(
            phase=phase,
            collected_count=collected_count,
        )
        flow_stage_primary = flow_stages[0]
        display_iteration = int(self._current_cli_iteration())
        entry = {
            "iteration": self.iteration,
            "display_iteration": display_iteration,
            "phase": phase,
            "flow_stage_primary": flow_stage_primary,
            "flow_stages": flow_stages,
            "collected_count": collected_count,
            "added_count": added_count,
            "eval_count": eval_count,
            "failure_count": failure_count,
            "candidate_count": candidate_count,
            "patch_count": candidate_count,
            "accepted": accepted,
            "delta_acc": delta_acc,
            "current_accuracy": current_acc,
            "best_accuracy": best_acc,
            "regressions": regressions,
            "program_version": self.current_version.version_id,
            "program_revision_count": self._current_program_revision_count(),
            "llm_calls": self._total_llm_call_count(),
            "active_class_count": self._current_active_class_count(),
            "global_step": self._resolve_explorer_global_step(explorer_diagnostics),
            "patch_digest": patch_digest,
            "decision_reason": decision_reason,
            "dataset_size": len(self.canonical_d),
            "protected_size": len(self.protected_set),
            "pending_count": max(0, len(self.canonical_d) - len(self.protected_set)),
            "target_action": target_action,
            "epoch_index": epoch_index,
            "epoch_budget_size": epoch_budget_size,
            "epoch_budget_remaining": epoch_budget_remaining,
            "epoch_progress_done": epoch_done,
            "epoch_progress_total": epoch_total,
            "epoch_progress_ratio": epoch_ratio,
            "no_new_canonical_rounds": no_new_canonical_rounds,
            "max_no_new_canonical_rounds": max_no_new_canonical_rounds,
            "saturation_fail_count": saturation_status["fail_count"],
            "saturation_max_fail_count": saturation_status["max_fail_count"],
            "is_saturated": saturation_status["is_saturated"],
            "explorer": explorer_diagnostics or {},
            "timestamp": timestamp,
        }
        self._record_explorer_iteration_metric_snapshot(entry)

    def _record_explorer_iteration_metric_snapshot(
        self,
        entry: Dict[str, Any],
    ) -> None:
        recorder = getattr(self.explorer, "record_iteration_metric_snapshot", None)
        if not callable(recorder):
            return
        recorder(
            output_dir=self.output_dir,
            iteration_summary=entry,
        )

    def _compute_epoch_progress(
        self,
        epoch_budget_size: int,
        epoch_budget_remaining: int,
    ) -> Tuple[int, int, float]:
        total = max(0, int(epoch_budget_size))
        remaining = max(0, int(epoch_budget_remaining))
        effective_total = max(total, remaining)
        if effective_total == 0:
            return 0, 0, 0.0
        done = max(0, effective_total - remaining)
        return done, effective_total, float(done) / float(effective_total)

    def _derive_flow_stages(
        self,
        phase: str,
        collected_count: int,
    ) -> List[str]:
        stages: List[str] = []
        if collected_count > 0:
            stages.append("explore_collect")
        if phase == "protected_add":
            stages.append("verify_pending")
        elif phase in {"no_patch", "no_program", "discovery_failed", "incorporated"}:
            stages.append("llm_patch")
        elif phase == "no_target":
            stages.append("idle_no_target")
        elif phase == "collect":
            stages.append("collect_only")
        else:
            stages.append("other")
        return stages

    def _truncate_text(self, text: Any, max_len: int = 100) -> str:
        raw = str(text).rstrip()
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 3] + "..."

    def _format_runtime_elapsed(self, elapsed_sec: float) -> str:
        total_seconds = max(0, int(float(elapsed_sec)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{seconds:02d}s"
        if minutes > 0:
            return f"{minutes}m{seconds:02d}s"
        return f"{seconds}s"

    def _program_elapsed_text(self) -> Optional[str]:
        started_at = self._program_started_at
        if not isinstance(started_at, (int, float)):
            return None
        return self._format_runtime_elapsed(perf_counter() - float(started_at))

    def _current_iteration_elapsed_text(self) -> Optional[str]:
        started_at = self._current_iteration_started_at
        if not isinstance(started_at, (int, float)):
            return None
        return self._format_runtime_elapsed(perf_counter() - float(started_at))

    def _append_program_elapsed_suffix(self, text: str) -> str:
        line = str(text).rstrip()
        if "total_elapsed=" in line:
            return line
        elapsed_text = self._program_elapsed_text()
        if not isinstance(elapsed_text, str) or not elapsed_text:
            return line
        return f"{line} total_elapsed={elapsed_text}"

    def _build_progress_postfix_text(self, *parts: Optional[str]) -> str:
        resolved_parts: List[str] = []
        for part in parts:
            if isinstance(part, str) and part.strip():
                resolved_parts.append(part.strip())
        elapsed_text = self._program_elapsed_text()
        if isinstance(elapsed_text, str) and elapsed_text:
            resolved_parts.append(f"total_elapsed={elapsed_text}")
        return " ".join(resolved_parts)

    def _emit_plain_line(
        self,
        text: str,
        max_len: int = 220,
    ) -> None:
        self._clear_live_console_line()
        pbar = getattr(self, "_active_progress_bar", None)
        if pbar is None or bool(getattr(pbar, "disable", False)):
            line = self._truncate_text(text, max_len=max_len)
            print(line)
            self._last_console_line_blank = False
            return
        line = self._append_program_elapsed_suffix(text)
        line = self._truncate_text(line, max_len=max_len)
        pbar.write(line)
        self._refresh_progress_bar(pbar, force=True)
        self._last_console_line_blank = False

    def _format_epoch_progress_text(self, done: int, total: int) -> str:
        if total <= 0:
            return "-"
        ratio = max(0.0, min(1.0, float(done) / float(total)))
        return f"{done}/{total} ({ratio * 100.0:.1f}%)"

    def _emit_console_line(
        self,
        pbar: Optional[Any],
        text: str,
        max_len: int = 220,
    ) -> None:
        self._clear_live_console_line()
        if pbar is None:
            pbar = getattr(self, "_active_progress_bar", None)
        if pbar is None or bool(getattr(pbar, "disable", False)):
            line = self._truncate_text(text, max_len=max_len)
            print(line)
            self._last_console_line_blank = False
            return
        line = self._append_program_elapsed_suffix(text)
        line = self._truncate_text(line, max_len=max_len)
        pbar.write(line)
        self._refresh_progress_bar(pbar, force=True)
        self._last_console_line_blank = False

    def _emit_blank_line(self) -> None:
        self._clear_live_console_line()
        pbar = getattr(self, "_active_progress_bar", None)
        if pbar is None or bool(getattr(pbar, "disable", False)):
            print("")
            self._last_console_line_blank = True
            return
        pbar.write("")
        self._refresh_progress_bar(pbar, force=True)
        self._last_console_line_blank = True

    def _emit_single_blank_separator(self) -> None:
        if bool(getattr(self, "_last_console_line_blank", True)):
            return
        self._emit_blank_line()

    def _supports_live_console_line(self) -> bool:
        stream = getattr(sys, "stdout", None)
        if stream is None or not hasattr(stream, "write") or not hasattr(stream, "flush"):
            return False
        if callable(getattr(stream, "getvalue", None)):
            return False
        return True

    def _emit_live_console_line(self, text: str, max_len: int = 220) -> None:
        if not self._supports_live_console_line():
            self._emit_console_line(None, text, max_len=max_len)
            return
        line = self._truncate_text(text, max_len=max_len)
        stream = sys.stdout
        width = max(int(self._live_console_line_width), len(line))
        stream.write("\r" + line.ljust(width))
        stream.flush()
        self._live_console_line_width = width

    def _clear_live_console_line(self) -> None:
        if not self._supports_live_console_line():
            return
        width = int(getattr(self, "_live_console_line_width", 0) or 0)
        if width <= 0:
            return
        stream = sys.stdout
        stream.write("\r" + (" " * width) + "\r")
        stream.flush()
        self._live_console_line_width = 0

    def _current_cli_iteration(self) -> int:
        version = getattr(self, "current_version", None)
        index = getattr(version, "index", None)
        if isinstance(index, int):
            return max(0, int(index))
        version_id = getattr(version, "version_id", None)
        if isinstance(version_id, str):
            text = version_id.strip().lower()
            if text.startswith("v") and text[1:].isdigit():
                return max(0, int(text[1:]))
        return max(0, int(self.iteration))

    def _current_log_iteration(self) -> int:
        if isinstance(self.iteration, int) and self.iteration > 0:
            return int(self.iteration)
        return self._current_cli_iteration()

    def _emit_log_event(
        self,
        category: str,
        message: str,
        indent_level: int = 0,
        max_len: int = 220,
    ) -> None:
        prefix = self._format_log_label(category)
        text = f"{self._indent_prefix(indent_level)}{prefix}"
        if str(message or "").strip():
            text = f"{text} {message}"
        self._emit_console_line(
            None,
            text,
            max_len=max_len,
        )

    def _emit_elapsed_log_event(
        self,
        category: str,
        message: str,
        indent_level: int = 0,
        max_len: int = 220,
    ) -> None:
        self._emit_log_event(
            category,
            self._append_program_elapsed_suffix(message),
            indent_level=indent_level,
            max_len=max_len,
        )

    def _emit_timing_log(
        self,
        category: str,
        message_parts: List[str],
        *,
        elapsed_sec: float,
        elapsed_insert_index: Optional[int] = None,
        indent_level: int = 0,
        max_len: int = 220,
    ) -> None:
        if not isinstance(elapsed_sec, (int, float)):
            return
        elapsed = max(0.0, float(elapsed_sec))
        parts = [str(part).strip() for part in message_parts if str(part).strip()]
        elapsed_part = f"elapsed={self._format_patch_elapsed(elapsed)}"
        if isinstance(elapsed_insert_index, int):
            insert_index = max(0, min(len(parts), int(elapsed_insert_index)))
            parts.insert(insert_index, elapsed_part)
        else:
            parts.append(elapsed_part)
        self._emit_elapsed_log_event(
            category,
            " ".join(parts),
            indent_level=indent_level,
            max_len=max_len,
        )

    def _emit_reject_log_event(
        self,
        message: str,
        indent_level: int = 0,
        max_len: int = 220,
    ) -> None:
        self._emit_log_event(
            "REJECT",
            message,
            indent_level=indent_level,
            max_len=max_len,
        )
        self._emit_blank_line()

    def _emit_group_header_line(self, group: str, message: str, max_len: int = 220) -> None:
        del group
        self._emit_single_blank_separator()
        body = str(message or "").strip()
        if body:
            self._emit_console_line(None, body, max_len=max_len)

    def _emit_detail_line(
        self,
        message: str,
        max_len: int = 220,
        indent_level: int = 1,
    ) -> None:
        self._emit_plain_line(
            f"{self._detail_indent(indent_level)}{str(message).strip()}",
            max_len=max_len,
        )

    def _emit_live_detail_line(
        self,
        message: str,
        max_len: int = 220,
        indent_level: int = 1,
    ) -> None:
        self._emit_live_console_line(
            f"{self._detail_indent(indent_level)}{str(message).strip()}",
            max_len=max_len,
        )

    def _format_log_label(self, category: str) -> str:
        text = str(category).strip().upper()
        exact_labels = {
            "COLLECT": "[COLLECT]",
            "COLLECT_PREP_END": "[COLLECT PREP END]",
            "COLLECT_END": "[COLLECT END]",
            "COLLECT_POST_END": "[COLLECT POST END]",
            "DIAGNOSE_END": "[DIAGNOSE END]",
            "FRONTIER_REFRESH": "[FRONTIER REFRESH]",
            "FRONTIER_REFRESH_END": "[FRONTIER REFRESH END]",
            "VERIFY": "[VERIFY]",
            "VERIFY_END": "[VERIFY END]",
            "TRAIN": "[TRAIN]",
            "TRAIN_END": "[TRAIN END]",
        }
        if text in exact_labels:
            return exact_labels[text]
        if text == "LLM":
            return f"[{text}]"
        return f"[{text:<7}]"

    def _indent_prefix(self, indent_level: int) -> str:
        return " " * (5 * max(0, int(indent_level)))

    def _detail_indent(self, indent_level: int = 3) -> str:
        return self._indent_prefix(indent_level)

    def _llm_provider_label(self) -> str:
        predictor = getattr(self.generator, "llm_predictor", None)
        config = getattr(predictor, "config", None)
        provider = getattr(config, "provider", None)
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
        if predictor is None:
            return "unknown"
        return predictor.__class__.__name__

    def _resolve_llm_model_label(self) -> Optional[str]:
        predictor = getattr(self.generator, "llm_predictor", None)
        config = getattr(predictor, "config", None)
        if config is None:
            return None
        for attr_name in (
            "openai_model",
            "gemini_model",
        ):
            value = getattr(config, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_llm_runtime_status(self) -> Dict[str, str]:
        predictor = getattr(self.generator, "llm_predictor", None)
        provider_label = self._llm_provider_label()
        describe = getattr(predictor, "describe_runtime_status", None)
        if callable(describe):
            try:
                payload = describe()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                normalized: Dict[str, str] = {}
                for key, value in payload.items():
                    if value is None:
                        continue
                    text = str(value).strip()
                    if text:
                        normalized[str(key)] = text
                if "provider" not in normalized:
                    normalized["provider"] = provider_label
                return normalized

        status = {
            "provider": provider_label,
            "transport": "api",
            "state": "ready",
        }
        model_label = self._resolve_llm_model_label()
        if model_label:
            status["model"] = model_label
        return status

    def _format_llm_runtime_identity(self) -> str:
        status = self._resolve_llm_runtime_status()
        parts: List[str] = []
        provider_label = str(status.get("provider") or "").strip()
        if provider_label:
            parts.append(f"llm={provider_label}")
        model_label = str(status.get("model") or "").strip()
        if model_label:
            parts.append(f"model={model_label}")
        thinking_label = str(status.get("thinking_level") or status.get("reasoning_effort") or "").strip()
        if thinking_label:
            parts.append(f"thinking={thinking_label}")
        return " | ".join(parts) if parts else "llm=unknown"

    def _format_llm_runtime_status_line(self) -> str:
        status = self._resolve_llm_runtime_status()
        transport = str(status.get("transport") or "").strip().lower()
        provider = str(status.get("provider") or "").strip().lower()
        parts = ["cli" if transport == "cli" else "llm", status.get("state", "unknown")]
        model_label = status.get("model")
        if model_label:
            parts.append(f"model={model_label}")
        if provider == "gemini":
            thinking_level = status.get("thinking_level")
            if thinking_level:
                parts.append(f"thinking_level={thinking_level}")
        else:
            reasoning_effort = status.get("reasoning_effort")
            if reasoning_effort:
                parts.append(f"reasoning_effort={reasoning_effort}")
        auth_status = status.get("auth_status")
        if auth_status:
            parts.append(f"auth={auth_status}")
        if transport == "cli":
            command_status = status.get("command_status")
            if command_status:
                parts.append(f"exec={command_status}")
        return " ".join(str(part).strip() for part in parts if str(part).strip())

    def _emit_llm_runtime_status(self) -> None:
        self._emit_detail_line(
            self._format_llm_runtime_status_line(),
            max_len=260,
            indent_level=1,
        )

    def _format_patch_elapsed(self, elapsed_sec: float) -> str:
        seconds = max(0.0, float(elapsed_sec))
        if seconds < 10.0:
            return f"{seconds:.2f}s"
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        return f"{minutes}m{secs:02d}s"

    def _start_patch_elapsed_logger(
        self,
    ) -> tuple[float, Event, Thread]:
        start_time = perf_counter()
        stop_event = Event()
        interval = max(0.01, float(getattr(self, "_patch_elapsed_log_interval_sec", 1.0)))
        active_pbar = getattr(self, "_active_progress_bar", None)
        using_verify_pending_bar = (
            active_pbar is not None and isinstance(self._verify_pending_progress_state, dict)
        )
        if using_verify_pending_bar:
            self._set_verify_pending_live_status("llm_run=0.00s")
        else:
            self._emit_live_detail_line("run 0.00s", indent_level=1)

        def _heartbeat() -> None:
            while not stop_event.wait(interval):
                elapsed_text = self._format_patch_elapsed(perf_counter() - start_time)
                if using_verify_pending_bar:
                    self._set_verify_pending_live_status(f"llm_run={elapsed_text}")
                else:
                    self._emit_live_detail_line(f"run {elapsed_text}", indent_level=1)

        thread = Thread(target=_heartbeat, name="llm-patch-elapsed", daemon=True)
        thread.start()
        return start_time, stop_event, thread

    def _stop_patch_elapsed_logger(
        self,
        *,
        start_time: float,
        stop_event: Event,
        thread: Thread,
    ) -> str:
        stop_event.set()
        thread.join(timeout=0.2)
        if isinstance(self._verify_pending_progress_state, dict):
            self._clear_verify_pending_live_status()
        else:
            self._clear_live_console_line()
        return self._format_patch_elapsed(perf_counter() - start_time)

    def _append_elapsed_text(self, message: str, elapsed_text: Optional[str]) -> str:
        if not isinstance(elapsed_text, str) or not elapsed_text.strip():
            return message
        return f"{message} elapsed={elapsed_text.strip()}"

    def _split_actionable_message(self, message: str) -> tuple[str, Optional[str]]:
        text = str(message or "").strip()
        if not text:
            return "", None
        summary = text
        action: Optional[str] = None
        if " Do this: " in text:
            summary, action = text.split(" Do this: ", 1)
        if summary.startswith("[") and "] " in summary:
            summary = summary.split("] ", 1)[1]
        summary = summary.strip()
        if action is not None:
            action = action.strip() or None
        return summary, action

    def _generator_patch_total_attempts(self) -> int:
        return 1

    def _format_patch_retry_progress(
        self,
        *,
        generator_try: int = 1,
        total_generator_attempts: Optional[int] = None,
        unexpected_retry: int = 1,
        unexpected_retry_total: Optional[int] = None,
    ) -> str:
        try:
            resolved_total_attempts = int(total_generator_attempts)
        except (TypeError, ValueError):
            resolved_total_attempts = self._generator_patch_total_attempts()
        total_attempts = max(1, resolved_total_attempts)
        try:
            resolved_try = int(generator_try)
        except (TypeError, ValueError):
            resolved_try = 1
        resolved_try = min(total_attempts, max(1, resolved_try))
        parts = [f"generator_try={resolved_try}/{total_attempts}"]
        if unexpected_retry_total is not None:
            try:
                resolved_total = int(unexpected_retry_total)
            except (TypeError, ValueError):
                resolved_total = 0
            resolved_total = max(0, resolved_total)
            try:
                resolved_unexpected_retry = int(unexpected_retry)
            except (TypeError, ValueError):
                resolved_unexpected_retry = 1
            if resolved_total <= 0:
                resolved_unexpected_retry = 0
            else:
                resolved_unexpected_retry = min(
                    resolved_total,
                    max(1, resolved_unexpected_retry),
                )
            parts.append(
                f"unexpected_retry={resolved_unexpected_retry}/{resolved_total}"
            )
        return " ".join(parts)

    def _format_unexpected_patch_retry_cap_message(self, message: str) -> str:
        summary, action = self._split_actionable_message(message)
        problem = (
            (summary or str(message or "").strip() or "Patch stage failed.")
            + " "
            + "Unexpected patch-stage retry cap reached "
            + f"({self.patch_unexpected_error_max_attempts})."
        )
        if action:
            return f"{problem} Do this: {action}"
        return problem

    def _emit_iteration_banner(
        self,
        pbar: Optional[Any],
        max_iterations: Optional[int],
    ) -> None:
        del max_iterations
        current_cli_iteration = int(self._current_log_iteration())
        if (
            isinstance(self._last_emitted_cli_header_iteration, int)
            and int(self._last_emitted_cli_header_iteration) == current_cli_iteration
        ):
            return
        self._last_emitted_cli_header_iteration = current_cli_iteration
        iter_text = f"{current_cli_iteration:03d}"
        explorer_diag = self._collect_explorer_diagnostics()
        global_step = self._resolve_explorer_global_step(explorer_diag)
        banner_parts = [
            f"[ITER {iter_text}]",
            f"rev={self.current_version.version_id}",
            f"classes={self._current_active_class_count()}",
        ]
        if global_step is not None:
            banner_parts.append(f"global_step={global_step}")
        banner_parts.append(f"canon={len(self.canonical_d)}")
        elapsed_text = self._program_elapsed_text()
        saturation_step_text = self._format_banner_saturation_step_text()
        if isinstance(elapsed_text, str) and elapsed_text:
            banner_parts.append(f"total_elapsed={elapsed_text}")
        banner_parts.append(f"sat_step_hit={saturation_step_text}")
        banner_parts.append(f"timestamp={self._current_kst_timestamp_text()}")
        self._emit_plain_line("")
        self._emit_console_line(pbar, " ".join(banner_parts))
        self._emit_plain_line("-" * 72)

    def _current_kst_timestamp_text(self) -> str:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")

    def _format_banner_saturation_step_text(self) -> str:
        window_steps = max(1, int(self.saturation_global_step_window))
        current_steps = int(self.saturation.fail_count)
        return f"{current_steps}/{window_steps}"

    def _progress_bar_refresh_interval_sec(self) -> float:
        try:
            interval = float(getattr(self, "progress_bar_refresh_interval_sec", 5.0))
        except (TypeError, ValueError):
            return 5.0
        if not math.isfinite(interval):
            return 5.0
        return max(0.0, interval)

    def _progress_bar_refresh_state(self) -> Dict[int, float]:
        state = getattr(self, "_progress_bar_last_refresh_at_by_id", None)
        if not isinstance(state, dict):
            state = {}
            self._progress_bar_last_refresh_at_by_id = state
        return state

    def _refresh_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        force: bool = False,
    ) -> bool:
        if progress_bar is None or bool(getattr(progress_bar, "disable", False)):
            return False
        state = self._progress_bar_refresh_state()
        progress_bar_id = id(progress_bar)
        now = perf_counter()
        if not force:
            interval = self._progress_bar_refresh_interval_sec()
            last_refresh_at = state.get(progress_bar_id)
            if (
                isinstance(last_refresh_at, (int, float))
                and now - float(last_refresh_at) < interval
            ):
                return False
        progress_bar.refresh()
        state[progress_bar_id] = now
        return True

    def _supports_collect_progress_bar(self) -> bool:
        if tqdm is None:
            return False
        stream = getattr(sys, "stdout", None)
        if stream is None or not hasattr(stream, "write") or not hasattr(stream, "flush"):
            return False
        return True

    def _create_collect_progress_bar(self, *, total: int) -> Optional[Any]:
        if not self._supports_collect_progress_bar():
            return None
        safe_total = max(1, int(total))
        progress_bar = tqdm(
            total=safe_total,
            desc="Collect transitions",
            unit="tr",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
            miniters=1,
            mininterval=self._progress_bar_refresh_interval_sec(),
        )
        self._active_progress_bar = progress_bar
        self._update_collect_progress_bar(
            progress_bar,
            collected=0,
            total=safe_total,
        )
        return progress_bar

    def _create_verify_progress_bar(
        self,
        *,
        total: int,
        desc: str = "Verify transitions",
    ) -> Optional[Any]:
        if not self._supports_collect_progress_bar():
            return None
        safe_total = max(1, int(total))
        progress_bar = tqdm(
            total=safe_total,
            desc=str(desc).strip() or "Verify transitions",
            unit="tr",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
            miniters=1,
            mininterval=self._progress_bar_refresh_interval_sec(),
        )
        self._active_progress_bar = progress_bar
        self._update_verify_progress_bar(
            progress_bar,
            evaluated=0,
            total=safe_total,
        )
        return progress_bar

    def _create_train_progress_bar(
        self,
        *,
        total: int,
        desc: str = "Train iter",
    ) -> Optional[Any]:
        if not self._supports_collect_progress_bar():
            return None
        safe_total = max(1, int(total))
        progress_bar = tqdm(
            total=safe_total,
            desc=str(desc).strip() or "Train iter",
            unit="iter",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
            miniters=1,
            mininterval=self._progress_bar_refresh_interval_sec(),
        )
        self._active_progress_bar = progress_bar
        self._update_train_progress_bar(
            progress_bar,
            completed=0,
            total=safe_total,
            refresh=True,
        )
        return progress_bar

    def _update_collect_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        collected: int,
        total: int,
    ) -> None:
        if progress_bar is None:
            return
        safe_total = max(1, int(total))
        safe_collected = min(max(0, int(collected)), safe_total)
        progress_bar.total = safe_total
        progress_bar.n = safe_collected
        postfix = self._build_progress_postfix_text()
        if postfix:
            try:
                progress_bar.set_postfix_str(postfix, refresh=False)
            except Exception:  # noqa: BLE001
                pass
        self._refresh_progress_bar(progress_bar)

    def _update_verify_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        evaluated: int,
        total: int,
        refresh: bool = True,
    ) -> None:
        if progress_bar is None:
            return
        safe_total = max(1, int(total))
        safe_evaluated = min(max(0, int(evaluated)), safe_total)
        progress_bar.total = safe_total
        progress_bar.n = safe_evaluated
        postfix = self._build_progress_postfix_text()
        if postfix:
            try:
                progress_bar.set_postfix_str(postfix, refresh=False)
            except Exception:  # noqa: BLE001
                pass
        if refresh:
            self._refresh_progress_bar(progress_bar)

    def _build_train_progress_postfix_text(
        self,
        *,
        train_iter: Optional[int] = None,
        updates_completed: Optional[int] = None,
        contrastive_loss: Optional[float] = None,
    ) -> str:
        parts: List[str] = []
        if isinstance(train_iter, int) and train_iter >= 0:
            parts.append(f"train_iter={int(train_iter)}")
        if isinstance(updates_completed, int) and updates_completed >= 0:
            parts.append(f"ok={int(updates_completed)}")
        if isinstance(contrastive_loss, (int, float)):
            parts.append(f"cl={float(contrastive_loss):.3f}")
        return " ".join(parts)

    def _update_train_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        completed: int,
        total: int,
        train_iter: Optional[int] = None,
        updates_completed: Optional[int] = None,
        contrastive_loss: Optional[float] = None,
        refresh: bool = True,
    ) -> None:
        if progress_bar is None:
            return
        safe_total = max(1, int(total))
        safe_completed = min(max(0, int(completed)), safe_total)
        progress_bar.total = safe_total
        progress_bar.n = safe_completed
        postfix = self._build_train_progress_postfix_text(
            train_iter=(
                int(train_iter)
                if isinstance(train_iter, int)
                else None
            ),
            updates_completed=(
                int(updates_completed)
                if isinstance(updates_completed, int)
                else None
            ),
            contrastive_loss=(
                float(contrastive_loss)
                if isinstance(contrastive_loss, (int, float))
                else None
            ),
        )
        postfix = self._build_progress_postfix_text(postfix)
        if postfix:
            try:
                progress_bar.set_postfix_str(postfix, refresh=False)
            except Exception:  # noqa: BLE001
                pass
        if refresh:
            self._refresh_progress_bar(progress_bar)

    def _close_collect_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        previous_pbar: Optional[Any] = None,
    ) -> None:
        if progress_bar is None:
            return
        self._clear_live_console_line()
        if getattr(self, "_active_progress_bar", None) is progress_bar:
            self._active_progress_bar = previous_pbar
        self._progress_bar_refresh_state().pop(id(progress_bar), None)
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001
            pass

    def _close_verify_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        previous_pbar: Optional[Any] = None,
    ) -> None:
        if progress_bar is None:
            return
        self._clear_live_console_line()
        if getattr(self, "_active_progress_bar", None) is progress_bar:
            self._active_progress_bar = previous_pbar
        self._progress_bar_refresh_state().pop(id(progress_bar), None)
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001
            pass

    def _close_train_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        previous_pbar: Optional[Any] = None,
    ) -> None:
        if progress_bar is None:
            return
        self._clear_live_console_line()
        if getattr(self, "_active_progress_bar", None) is progress_bar:
            self._active_progress_bar = previous_pbar
        self._progress_bar_refresh_state().pop(id(progress_bar), None)
        try:
            progress_bar.close()
        except Exception:  # noqa: BLE001
            pass

    def _make_train_progress_handler(
        self,
    ) -> Tuple[Callable[[Any], None], Callable[[], None]]:
        state: Dict[str, Any] = {
            "progress_bar": None,
            "previous_pbar": None,
        }

        def _close() -> None:
            progress_bar = state.get("progress_bar")
            previous_pbar = state.get("previous_pbar")
            if progress_bar is None:
                state["previous_pbar"] = None
                return
            self._close_train_progress_bar(
                progress_bar,
                previous_pbar=previous_pbar,
            )
            state["progress_bar"] = None
            state["previous_pbar"] = None

        def _handle(payload: Any) -> None:
            if not isinstance(payload, dict):
                return
            raw_total = payload.get("train_progress_total")
            raw_completed = payload.get("train_progress_completed")
            if not isinstance(raw_total, int) or not isinstance(raw_completed, int):
                return
            safe_total = max(1, int(raw_total))
            safe_completed = min(max(0, int(raw_completed)), safe_total)
            progress_bar = state.get("progress_bar")
            if progress_bar is None:
                previous_pbar = getattr(self, "_active_progress_bar", None)
                progress_bar = self._create_train_progress_bar(
                    total=safe_total,
                    desc=str(payload.get("train_progress_desc") or "Train iter"),
                )
                state["progress_bar"] = progress_bar
                state["previous_pbar"] = previous_pbar
            self._update_train_progress_bar(
                progress_bar,
                completed=safe_completed,
                total=safe_total,
                train_iter=(
                    int(payload.get("train_iter"))
                    if isinstance(payload.get("train_iter"), int)
                    else None
                ),
                updates_completed=(
                    int(payload.get("train_updates_completed"))
                    if isinstance(payload.get("train_updates_completed"), int)
                    else None
                ),
                contrastive_loss=(
                    float(payload.get("train_contrastive_loss"))
                    if isinstance(payload.get("train_contrastive_loss"), (int, float))
                    else None
                ),
                refresh=True,
            )
            if safe_completed >= safe_total:
                _close()

        return _handle, _close

    def _close_verify_pending_progress_bar(
        self,
        progress_bar: Optional[Any],
        *,
        previous_pbar: Optional[Any] = None,
    ) -> None:
        self._verify_pending_progress_state = None
        self._verify_pending_live_status = None
        self._close_verify_progress_bar(
            progress_bar,
            previous_pbar=previous_pbar,
        )

    def _resolve_explorer_global_step(self, explorer: Any) -> Optional[int]:
        if not isinstance(explorer, dict):
            return None
        for container in (
            explorer,
            explorer.get("last_collect"),
            explorer.get("visualization"),
        ):
            if not isinstance(container, dict):
                continue
            for key in ("global_step", "total_steps"):
                value = container.get(key)
                try:
                    resolved = int(value)
                except (TypeError, ValueError):
                    continue
                if resolved >= 0:
                    return resolved
        return None

    def _supports_collect_train_progress(self, explorer: Any) -> bool:
        if not isinstance(explorer, dict):
            return False
        return any(
            key in explorer
            for key in (
                "total_updates",
                "train_schedule",
            )
        )

    def _extract_collect_train_status(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        last_collect = payload.get("last_collect")
        last_collect = last_collect if isinstance(last_collect, dict) else {}
        last_train = payload.get("last_train")
        last_train = last_train if isinstance(last_train, dict) else {}
        status: Dict[str, Any] = {}

        train_schedule = payload.get("train_schedule", last_collect.get("train_schedule"))
        if isinstance(train_schedule, str) and train_schedule.strip():
            status["train_schedule"] = train_schedule.strip()

        train_updates_total = payload.get("train_updates_total", payload.get("total_updates"))
        if isinstance(train_updates_total, int):
            status["train_updates_total"] = int(train_updates_total)

        train_phase = payload.get("train_phase", last_collect.get("train_phase"))
        if isinstance(train_phase, str) and train_phase.strip():
            status["train_phase"] = train_phase.strip()

        train_updates_completed = payload.get(
            "train_updates_completed",
            last_collect.get("train_updates_completed"),
        )
        if isinstance(train_updates_completed, int):
            status["train_updates_completed"] = int(train_updates_completed)

        td_loss = payload.get("train_td_loss")
        if not isinstance(td_loss, (int, float)):
            td_loss = payload.get("train_mean_td_loss")
        if not isinstance(td_loss, (int, float)):
            td_loss = last_train.get("td_loss")
        if not isinstance(td_loss, (int, float)):
            td_loss = last_collect.get("mean_td_loss")
        if isinstance(td_loss, (int, float)):
            status["td_loss"] = float(td_loss)

        mono_loss = payload.get("train_monotonic_loss")
        if not isinstance(mono_loss, (int, float)):
            mono_loss = last_train.get("monotonic_loss")
        if not isinstance(mono_loss, (int, float)):
            mono_loss = last_collect.get("mean_monotonic_loss")
        if isinstance(mono_loss, (int, float)):
            status["mono_loss"] = float(mono_loss)

        mean_prefix_length = payload.get("train_mean_prefix_length")
        if not isinstance(mean_prefix_length, (int, float)):
            mean_prefix_length = last_train.get("mean_prefix_length")
        if isinstance(mean_prefix_length, (int, float)):
            status["mean_prefix_length"] = float(mean_prefix_length)

        expanded_batch_size = payload.get("train_expanded_batch_size")
        if not isinstance(expanded_batch_size, (int, float)):
            expanded_batch_size = last_train.get("expanded_batch_size")
        if isinstance(expanded_batch_size, (int, float)):
            status["expanded_batch_size"] = float(expanded_batch_size)

        contrastive_loss = payload.get("contrastive_loss")
        if not isinstance(contrastive_loss, (int, float)):
            contrastive_loss = payload.get("train_contrastive_loss")
        if not isinstance(contrastive_loss, (int, float)):
            contrastive_loss = last_train.get("contrastive_loss")
        if isinstance(contrastive_loss, (int, float)):
            status["contrastive_loss"] = float(contrastive_loss)

        prototype_top1_accuracy = payload.get("prototype_top1_accuracy")
        if not isinstance(prototype_top1_accuracy, (int, float)):
            prototype_top1_accuracy = payload.get("train_prototype_top1_accuracy")
        if not isinstance(prototype_top1_accuracy, (int, float)):
            prototype_top1_accuracy = last_train.get("prototype_top1_accuracy")
        if isinstance(prototype_top1_accuracy, (int, float)):
            status["prototype_top1_accuracy"] = float(prototype_top1_accuracy)

        sample_store_size = payload.get("sample_store_size")
        sample_store_capacity = payload.get("sample_store_capacity")
        if not isinstance(sample_store_size, int):
            sample_store_size = last_collect.get("sample_store_size")
        if isinstance(sample_store_size, int):
            status["sample_store_size"] = int(sample_store_size)
        if isinstance(sample_store_capacity, int):
            status["sample_store_capacity"] = int(sample_store_capacity)

        learning_starts = payload.get("learning_starts", last_collect.get("learning_starts"))
        if isinstance(learning_starts, int):
            status["learning_starts"] = int(learning_starts)

        learning_starts_unit = payload.get(
            "learning_starts_unit",
            last_collect.get("learning_starts_unit"),
        )
        if isinstance(learning_starts_unit, str) and learning_starts_unit.strip():
            status["learning_starts_unit"] = learning_starts_unit.strip()

        learning_progress = payload.get(
            "learning_progress",
            last_collect.get("learning_progress"),
        )
        if isinstance(learning_progress, int):
            status["learning_progress"] = int(learning_progress)

        total_steps = payload.get("total_steps", last_collect.get("total_steps"))
        if isinstance(total_steps, int):
            status["total_steps"] = int(total_steps)
        return status

    def _format_explorer_train_summary(
        self,
        explorer: Any,
        *,
        include_metrics: bool = False,
        include_state: bool = True,
    ) -> Optional[str]:
        if not self._supports_collect_train_progress(explorer):
            return None
        status = self._extract_collect_train_status(explorer)
        parts: List[str] = []
        train_updates_completed = status.get("train_updates_completed")
        if isinstance(train_updates_completed, int) and train_updates_completed > 0:
            parts.append(f"updates=+{train_updates_completed}")
        train_updates_total = status.get("train_updates_total")
        train_phase = str(status.get("train_phase") or "").strip().lower()
        td_loss = status.get("td_loss")
        if isinstance(td_loss, (int, float)):
            parts.append(f"td={float(td_loss):.3f}")
        mono_loss = status.get("mono_loss")
        if isinstance(mono_loss, (int, float)):
            parts.append(f"mono={float(mono_loss):.3f}")
        mean_prefix_length = status.get("mean_prefix_length")
        if isinstance(mean_prefix_length, (int, float)):
            parts.append(f"pfx={float(mean_prefix_length):.2f}")
        expanded_batch_size = status.get("expanded_batch_size")
        if isinstance(expanded_batch_size, (int, float)):
            parts.append(f"xb={float(expanded_batch_size):.0f}")
        learning_starts = status.get("learning_starts")
        sample_store_size = status.get("sample_store_size")
        total_steps = status.get("total_steps")
        learning_progress = status.get("learning_progress")
        warmup_value: Optional[int] = None
        if isinstance(learning_progress, int):
            warmup_value = int(learning_progress)
        elif isinstance(total_steps, int):
            warmup_value = int(total_steps)
        elif isinstance(sample_store_size, int):
            warmup_value = int(sample_store_size)
        if train_phase == "warmup":
            if include_state:
                parts.append("state=warmup")
            if isinstance(learning_starts, int) and learning_starts > 0 and isinstance(
                warmup_value, int
            ):
                parts.append(f"warmup={warmup_value}/{int(learning_starts)}")
        elif train_phase:
            if include_state:
                parts.append(f"state={train_phase}")
        elif isinstance(learning_starts, int) and learning_starts > 0 and isinstance(
            warmup_value, int
        ) and warmup_value < int(learning_starts):
            if include_state:
                parts.append("state=warmup")
            parts.append(f"warmup={warmup_value}/{int(learning_starts)}")
        else:
            if include_state:
                parts.append("state=train")
        if isinstance(train_updates_total, int):
            parts.append(f"total={train_updates_total}")
        if include_metrics:
            contrastive_loss = status.get("contrastive_loss")
            if isinstance(contrastive_loss, (int, float)):
                parts.append(f"contrastive_loss={float(contrastive_loss):.4f}")
            prototype_top1_accuracy = status.get("prototype_top1_accuracy")
            if isinstance(prototype_top1_accuracy, (int, float)):
                parts.append(f"top1_acc={float(prototype_top1_accuracy):.4f}")
        return " ".join(parts) if parts else None

    def _collect_with_optional_progress(
        self,
        max_transitions: Optional[int],
        progress_callback: Optional[Any] = None,
    ) -> List[Transition]:
        return self.explorer.collect(
            max_transitions=max_transitions,
            progress_callback=progress_callback,
        )

    def _prepare_collected_transitions(
        self,
        transitions: List[Transition],
    ) -> List[Transition]:
        prepared = list(transitions) if isinstance(transitions, list) else list(transitions or [])
        self._require_collected_transitions_use_pipeline_state_store(prepared)
        if self.shuffle_collected_transitions and len(prepared) > 1:
            self._collection_rng.shuffle(prepared)
        return prepared

    def _requires_pipeline_state_store_transitions(self) -> bool:
        return str(getattr(self.explorer, "collection_topology", "")).strip() == "graph"

    def _assert_explorer_uses_pipeline_state_store(self) -> None:
        if not self._requires_pipeline_state_store_transitions():
            return
        explorer_store = getattr(self.explorer, "state_store", None)
        if explorer_store is not self.state_store:
            raise RuntimeError(
                "Graph discovery requires the explorer to use the pipeline StateStore. "
                "The explorer did not keep the StateStore passed through set_state_store()."
            )

    def _require_collected_transitions_use_pipeline_state_store(
        self,
        transitions: List[Transition],
    ) -> None:
        if not self._requires_pipeline_state_store_transitions():
            return
        self._assert_explorer_uses_pipeline_state_store()
        for index, transition in enumerate(transitions):
            if transition.state_store is not self.state_store:
                raise RuntimeError(
                    "Graph discovery collected a transition from a non-pipeline "
                    f"StateStore at batch index {int(index)}."
                )
            if not isinstance(transition.state_id, int) or int(transition.state_id) <= 0:
                raise RuntimeError(
                    "Graph discovery collected a transition without a valid source "
                    f"state id at batch index {int(index)}."
                )
            if (
                not isinstance(transition.next_state_id, int)
                or int(transition.next_state_id) <= 0
            ):
                raise RuntimeError(
                    "Graph discovery collected a transition without a valid next "
                    f"state id at batch index {int(index)}."
                )

    def _format_explorer_summary(self, explorer: Any) -> str:
        if not isinstance(explorer, dict):
            return "agent=-"
        name = explorer.get("name")
        name_text = str(name).strip() if isinstance(name, str) else "-"
        if not name_text:
            name_text = "-"
        parts = [f"agent={name_text}"]

        last_collect = explorer.get("last_collect")
        if isinstance(last_collect, dict):
            transitions_collected = last_collect.get("transitions_collected")
            if isinstance(transitions_collected, int):
                parts.append(f"a_data={transitions_collected}")
            epsilon_end = last_collect.get("epsilon_end")
            if isinstance(epsilon_end, (int, float)):
                parts.append(f"eps={float(epsilon_end):.3f}")
            counterexample_hit_ratio = last_collect.get("counterexample_hit_ratio")
            if isinstance(counterexample_hit_ratio, (int, float)):
                parts.append(f"ce={float(counterexample_hit_ratio):.2f}")
            mean_training_reward = last_collect.get("mean_training_reward")
            if isinstance(mean_training_reward, (int, float)):
                parts.append(f"r_train={float(mean_training_reward):.3f}")
            mean_counterexample_reward = last_collect.get("mean_counterexample_reward")
            if isinstance(mean_counterexample_reward, (int, float)):
                parts.append(f"r_ce={float(mean_counterexample_reward):.3f}")
            mean_lineage_reward = last_collect.get("mean_lineage_reward")
            if isinstance(mean_lineage_reward, (int, float)):
                parts.append(f"r_lin={float(mean_lineage_reward):.3f}")
            mean_td_loss = last_collect.get("mean_td_loss")
            if isinstance(mean_td_loss, (int, float)):
                parts.append(f"td={float(mean_td_loss):.3f}")
            mean_info_gain_raw = last_collect.get("mean_info_gain_raw")
            if isinstance(mean_info_gain_raw, (int, float)):
                parts.append(f"ig={float(mean_info_gain_raw):.3f}")
            mean_info_gain_normalized = last_collect.get("mean_info_gain_normalized")
            if isinstance(mean_info_gain_normalized, (int, float)):
                parts.append(f"ig_n={float(mean_info_gain_normalized):.3f}")
            mean_dynamics_nll = last_collect.get("mean_dynamics_nll")
            if isinstance(mean_dynamics_nll, (int, float)):
                parts.append(f"dyn_nll={float(mean_dynamics_nll):.3f}")
        else:
            epsilon = explorer.get("epsilon")
            if isinstance(epsilon, (int, float)):
                parts.append(f"eps={float(epsilon):.3f}")
        lineage_anchor_count = explorer.get("lineage_anchor_count")
        if isinstance(lineage_anchor_count, int):
            parts.append(f"linA={lineage_anchor_count}")
        last_added_count = explorer.get("last_added_count")
        if isinstance(last_added_count, int):
            parts.append(f"added={last_added_count}")
        active_class_count = explorer.get("active_class_count")
        if isinstance(active_class_count, int):
            parts.append(f"classes={active_class_count}")
        observed_dynamics_classes = explorer.get("observed_dynamics_classes")
        if isinstance(observed_dynamics_classes, int):
            parts.append(f"obs={observed_dynamics_classes}")
        current_dynamics_class = explorer.get("current_dynamics_class")
        if isinstance(current_dynamics_class, int) and current_dynamics_class > 0:
            parts.append(f"cls=C{current_dynamics_class}")
        resume_pending = explorer.get("resume_pending")
        if isinstance(resume_pending, bool):
            parts.append("resume=on" if resume_pending else "resume=off")
        return " ".join(parts)

    def _format_iteration_console_lines(self, entry: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        explorer_summary = self._format_explorer_summary(entry.get("explorer"))
        phase = str(entry.get("phase") or "")
        target = entry.get("target_action") or "-"
        done = int(entry.get("epoch_progress_done") or 0)
        total = int(entry.get("epoch_progress_total") or 0)
        progress_text = self._format_epoch_progress_text(done, total)

        if int(entry.get("collected_count") or 0) > 0:
            lines.append(
                "  [COLLECT]"
                + (f" {explorer_summary}" if explorer_summary else "")
            )
            train_summary = self._format_explorer_train_summary(
                entry.get("explorer"),
                include_state=False,
            )
            if isinstance(train_summary, str) and train_summary:
                lines.append(
                    self._append_program_elapsed_suffix(f"  [TRAIN] {train_summary}")
                )

        if phase in {"no_patch", "no_program", "discovery_failed", "incorporated"}:
            status = {
                "no_patch": "invalid_patch",
                "no_program": "gate_rejected",
                "discovery_failed": "discovery_failed",
                "incorporated": "accepted",
            }.get(phase, phase)
            attempt_retry_fields = self._format_attempt_retry_fields(
                entry.get("failure_count")
            )
            lines.append(
                "  [LLM-PATCH] "
                f"epoch={entry.get('epoch_index', 0)} "
                f"progress={progress_text} "
                f"target={target} "
                f"status={status} "
                f"{attempt_retry_fields} "
                f"patch={entry.get('patch_count', 0)} "
                f"acc={float(entry.get('best_accuracy') or 0.0):.4f} "
                f"delta={float(entry.get('delta_acc') or 0.0):.4f} "
                f"sat={entry.get('saturation_fail_count', 0)}/"
                f"{entry.get('saturation_max_fail_count', 0)} "
                f"reason={self._truncate_text(entry.get('decision_reason') or '-', max_len=120)}"
            )
            return lines

        lines.append(self._format_iteration_summary(entry))
        return lines

    def _format_iteration_summary(self, entry: Dict) -> str:
        target = entry.get("target_action") or "-"
        patch_count = entry.get("patch_count", entry.get("candidate_count", 0))
        explorer_summary = self._format_explorer_summary(entry.get("explorer"))
        display_iteration = int(entry.get("display_iteration") or entry.get("iteration") or 0)
        parts = [
            f"[ITERATION {display_iteration:03d}]",
            f"phase={entry['phase']}",
            f"rev={entry.get('program_version')}",
            f"classes={entry.get('active_class_count', 0)}",
        ]
        global_step = entry.get("global_step")
        if isinstance(global_step, int):
            parts.append(f"global_step={global_step}")
        parts.extend(
            [
                f"data={entry['collected_count']} (+{entry['added_count']})",
                f"target={target}",
                f"patch={patch_count}",
                f"acc={entry['best_accuracy']:.4f}",
                f"delta={entry['delta_acc']:.4f}",
                f"canon={entry['dataset_size']}",
                f"epoch={entry.get('epoch_index', 0)}",
                f"rem={entry.get('epoch_budget_remaining', 0)}/{entry.get('epoch_budget_size', 0)}",
                f"no_new={entry.get('no_new_canonical_rounds', 0)}",
                f"sat={entry['saturation_fail_count']}/{entry['saturation_max_fail_count']}",
            ]
        )
        if explorer_summary:
            parts.append(explorer_summary)
        return " ".join(parts)

    def _save_final(self) -> None:
        llm_usage_summary = self.llm_usage_tracker.build_summary()
        llm_usage_summary["limits"] = {
            "max_total_llm_calls": self.max_total_llm_calls,
            "remaining_llm_calls": self._remaining_llm_call_budget(),
        }
        self._write_llm_usage_summary_files(llm_usage_summary)
