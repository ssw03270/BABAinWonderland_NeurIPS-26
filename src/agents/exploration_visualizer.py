"""Live visualization helper for collect-phase exploration."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
import math
from time import perf_counter
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence

_PAYLOAD_UNSET = object()


class ExplorationVisualizer:
    """Web snapshot/dashboard helper for collect-phase exploration."""

    _HUD_CONTEXT_KEY_GROUPS = (
        ("world_index",),
        ("map_name",),
    )
    _HUD_PRIMARY_KEYS = ("episode", "ep", "step", "global_step", "action")
    _HUD_SECONDARY_KEYS = (
        "eps",
        "epsilon",
        "rh",
        "rz",
        "rtotal",
        "reward",
        "done",
        "unk",
        "unknown_hit_ratio",
    )
    _DASHBOARD_BASE_WIDTH = 15.6
    _DASHBOARD_BASE_HEIGHT = 8.8
    _DASHBOARD_WIDTH_MARGIN_IN = 2.2
    _DASHBOARD_HEIGHT_MARGIN_IN = 2.6
    _DASHBOARD_CONTENT_LEFT = 0.055
    _DASHBOARD_CONTENT_RIGHT = 0.972
    _DASHBOARD_CONTENT_TOP = 0.89
    _DASHBOARD_CONTENT_BOTTOM = 0.085
    _DASHBOARD_COLUMN_GAP = 0.055
    _DASHBOARD_ROW_GAP = 0.075
    _DASHBOARD_LEFT_COLUMN_WIDTH_RATIO = 1.0
    _DASHBOARD_RIGHT_COLUMN_WIDTH_RATIO = 2.42
    _DASHBOARD_TOP_LEFT_HEIGHT_RATIO = 1.0
    _DASHBOARD_BOTTOM_LEFT_HEIGHT_RATIO = 1.0
    _DASHBOARD_PROJECTION_HEIGHT_RATIO = 2.0
    _DASHBOARD_CLASS_PROBABILITY_HEIGHT_RATIO = 1.3
    _DASHBOARD_CLASS_PROBABILITY_LEFT_INSET = 0.0
    _DASHBOARD_RIGHT_AXIS_SPINE_X = 1.0
    _DASHBOARD_LEFT_Y_LABEL_PAD = 0
    _DASHBOARD_RIGHT_Y_LABEL_PAD = 2.0
    _DASHBOARD_RIGHT_Y_TICK_PAD = 1.4
    _DASHBOARD_X_LABEL_PAD = 1.0
    _DASHBOARD_PANEL_TITLE_PAD = 1.5
    _DASHBOARD_CONTEXT_X = 0.988
    _DASHBOARD_CONTEXT_Y = 0.952
    _DASHBOARD_HIDDEN_RECT = (-10.0, -10.0, 0.001, 0.001)

    def __init__(
        self,
        env: Any,
        dashboard_history_limit: int = 240,
        dashboard_context_lines: Optional[Sequence[str]] = None,
        snapshot_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        enabled: bool = True,
    ):
        self.env = env
        self.enabled = bool(enabled)
        self.dashboard_history_limit = max(16, int(dashboard_history_limit))
        self.dashboard_context_lines = self._normalize_dashboard_context_lines(
            dashboard_context_lines
        )
        self.dashboard_error: Optional[str] = None
        self._dashboard = None
        self._dashboard_step_history: Deque[int] = deque(maxlen=self.dashboard_history_limit)
        self._dashboard_info_gain_history: Deque[float] = deque(maxlen=self.dashboard_history_limit)
        self._dashboard_dynamics_nll_history: Deque[float] = deque(maxlen=self.dashboard_history_limit)
        self._dashboard_aux_metric_a_history: Deque[float] = deque(maxlen=self.dashboard_history_limit)
        self._dashboard_aux_metric_b_history: Deque[float] = deque(maxlen=self.dashboard_history_limit)
        self._dashboard_transition_reward_history: Deque[float] = deque(maxlen=self.dashboard_history_limit)
        self._dashboard_metric_histories: Dict[str, Deque[float]] = {}
        self._last_payload: Optional[Dict[str, Any]] = None
        self._displayed_payload: Optional[Dict[str, Any]] = None
        self._snapshot_callback = snapshot_callback
        self._snapshot_enabled_provider: Optional[Callable[[], bool]] = None
        self._visitation_heatmap_enabled_provider: Optional[Callable[[], bool]] = None
        self._visitation_heatmap_sparse_class_filter_provider: Optional[Callable[[], bool]] = None
        self._agent_diagnostics_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._progress_callback: Optional[Callable[[], None]] = None
        self._snapshot_min_interval_sec = 0.1
        self._last_snapshot_emit_at = 0.0

    def reset(self) -> None:
        self.dashboard_error = None
        self._dashboard_step_history.clear()
        self._dashboard_info_gain_history.clear()
        self._dashboard_dynamics_nll_history.clear()
        self._dashboard_aux_metric_a_history.clear()
        self._dashboard_aux_metric_b_history.clear()
        self._dashboard_transition_reward_history.clear()
        self._dashboard_metric_histories.clear()
        self._last_payload = None
        self._displayed_payload = None
        self._dashboard = None
        if not self.enabled:
            return
        self._emit_progress()
        self._emit_snapshot()

    def _normalize_dashboard_context_lines(
        self,
        dashboard_context_lines: Optional[Sequence[str]],
    ) -> tuple[str, ...]:
        if dashboard_context_lines is None:
            return ()
        normalized_lines: list[str] = []
        for raw_line in dashboard_context_lines:
            line = str(raw_line).strip()
            if line:
                normalized_lines.append(line)
        return tuple(normalized_lines)

    def _build_dashboard_context_text(self) -> str:
        if not self.dashboard_context_lines:
            return ""
        return "\n".join(self.dashboard_context_lines)

    def render(
        self,
        caption: Optional[str] = None,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        visitation_heatmap: Optional[Any] = None,
        class_rows: Optional[Sequence[Dict[str, Any]]] = None,
        display: Optional[Dict[str, Any]] = None,
        program_context: Optional[Dict[str, Any]] = None,
        board_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "caption": str(caption or ""),
            "metrics": dict(metrics or {}),
            "visitation_heatmap": visitation_heatmap,
            "class_rows": [
                dict(row)
                for row in (class_rows or ())
                if isinstance(row, dict)
            ],
            "display": dict(display) if isinstance(display, dict) else {},
            "program_context": (
                dict(program_context)
                if isinstance(program_context, dict)
                else None
            ),
            "board_state": dict(board_state) if isinstance(board_state, dict) else None,
        }
        self._last_payload = payload
        self._render_payload(payload=payload)

    def _render_payload(
        self,
        *,
        payload: Dict[str, Any],
    ) -> None:
        self._displayed_payload = payload
        self._update_dashboard(
            metrics=payload.get("metrics"),
            visitation_heatmap=payload.get("visitation_heatmap"),
            display=payload.get("display"),
        )
        self._emit_progress()
        self._emit_snapshot()

    def _build_hud_lines(self, payload: Dict[str, Any]) -> tuple[str, str]:
        raw_caption = str(payload.get("caption") or "").strip()
        metrics = payload.get("metrics")
        title, ordered_fields = self._extract_caption_fields(raw_caption)
        ordered_fields = self._merge_metric_fields(
            ordered_fields=ordered_fields,
            metrics=metrics if isinstance(metrics, dict) else {},
        )

        context_chips = []
        primary_chips = []
        secondary_chips = []
        consumed = set()
        used_labels: set[str] = set()

        def append_unique_chip(target: list[str], *, key: str, value: Any) -> None:
            label = self._format_field_label(key)
            formatted = self._format_field_value(key=key, value=value)
            if not label or not formatted:
                return
            if label in used_labels:
                return
            target.append(f"{label} {formatted}")
            used_labels.add(label)

        for key_group in self._HUD_CONTEXT_KEY_GROUPS:
            lookup_key = next(
                (
                    candidate
                    for candidate in (
                        self._lookup_field_key(ordered_fields, raw_key)
                        for raw_key in key_group
                    )
                    if candidate is not None
                ),
                None,
            )
            if lookup_key is None:
                continue
            consumed.add(lookup_key)
            append_unique_chip(
                context_chips,
                key=lookup_key,
                value=ordered_fields[lookup_key],
            )

        for key in self._HUD_PRIMARY_KEYS:
            lookup_key = self._lookup_field_key(ordered_fields, key)
            if lookup_key is None:
                continue
            consumed.add(lookup_key)
            append_unique_chip(primary_chips, key=lookup_key, value=ordered_fields[lookup_key])

        for key in self._HUD_SECONDARY_KEYS:
            lookup_key = self._lookup_field_key(ordered_fields, key)
            if lookup_key is None:
                continue
            if lookup_key in consumed or self._is_training_metric_key(lookup_key):
                continue
            consumed.add(lookup_key)
            append_unique_chip(secondary_chips, key=lookup_key, value=ordered_fields[lookup_key])

        # Keep the left-panel formatting stable from the first frame by always
        # reserving space for the dynamics-class panel.
        lines: list[str] = []
        if context_chips:
            lines.append("   ".join(context_chips))
            summary_chips = primary_chips + secondary_chips
            if summary_chips:
                lines.append("   ".join(summary_chips))
        else:
            max_chars = self._resolve_metrics_char_capacity()
            max_summary_lines = 2
            lines.extend(
                self._wrap_chips(
                    primary_chips,
                    max_chars=max_chars,
                    max_lines=min(2, max_summary_lines),
                )
            )
            remaining_summary_lines = max(0, max_summary_lines - len(lines))
            if remaining_summary_lines > 0:
                lines.extend(
                    self._wrap_chips(
                        secondary_chips,
                        max_chars=max_chars,
                        max_lines=remaining_summary_lines,
                    )
                )
        display = payload.get("display")
        display_config = display if isinstance(display, dict) else {}
        custom_training_lines = self._build_configured_hud_lines(
            ordered_fields=ordered_fields,
            display=display_config,
        )
        training_lines = custom_training_lines or self._build_training_hud_lines(
            ordered_fields=ordered_fields,
            metrics=metrics if isinstance(metrics, dict) else {},
        )
        if training_lines:
            section_title = "TRAINING"
            if custom_training_lines:
                configured_title = str(display_config.get("hud_title") or "").strip()
                if configured_title:
                    section_title = configured_title
            lines.append("")
            lines.append(section_title)
            lines.extend(training_lines)
        if not lines:
            lines = ["Waiting for exploration updates."]

        return title or "Exploration", "\n".join(lines)

    def _is_training_metric_key(self, key: str) -> bool:
        lookup = str(key).strip().lower()
        return lookup in {
            "dataset_size",
            "pending_count",
            "observed_dynamics_classes",
            "resume_pending",
            "progress_phase",
            "train_phase",
            "learning_phase",
            "train_schedule",
            "train_updates_per_event",
            "train_event_index",
            "train_event_count",
            "train_iter",
            "updates",
            "policy_updates",
            "dynamics_updates",
            "td",
            "mono",
            "pfx",
            "xb",
            "queued",
            "sample_store_size",
            "use_per",
            "per_alpha",
            "per_beta",
            "per_beta_current",
            "per_priority_mean",
            "per_priority_max",
            "per_is_weight_mean",
            "per_is_weight_min",
            "contrastive_loss",
            "prototype_top1_accuracy",
            "mean_positive_logit",
            "mean_max_negative_logit",
            "rh",
            "rz",
            "rtotal",
            "rh_mean",
            "rh_std",
            "rz_mean",
            "rz_std",
            "rtotal_mean",
            "rtotal_std",
            "actor_loss",
            "policy_loss",
            "critic_loss",
            "current_dynamics_class",
            "predicted_dynamics_class",
            "predicted_dynamics_confidence",
            "softmax_temperature",
            "active_dynamics_classes",
            "known_dynamics_classes",
            "sample_duplicate_skips",
            "learning_starts",
            "added_count",
            "tsne_points",
        }

    def _build_configured_hud_lines(
        self,
        *,
        ordered_fields: "OrderedDict[str, str]",
        display: Dict[str, Any],
    ) -> list[str]:
        if not isinstance(display, dict):
            return []
        raw_sections = display.get("hud_sections")
        if not isinstance(raw_sections, (list, tuple)):
            return []

        output_lines: list[str] = []
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                continue
            title = str(raw_section.get("title") or "").strip()
            raw_chips = raw_section.get("chips")
            if not isinstance(raw_chips, (list, tuple)):
                continue
            chips: list[str] = []
            for raw_chip in raw_chips:
                if isinstance(raw_chip, str):
                    candidate_keys = (raw_chip,)
                elif isinstance(raw_chip, (list, tuple)):
                    candidate_keys = tuple(
                        str(key).strip()
                        for key in raw_chip
                        if str(key).strip()
                    )
                else:
                    candidate_keys = ()
                if not candidate_keys:
                    continue
                chip = self._build_field_chip_from_candidates(
                    ordered_fields=ordered_fields,
                    keys=candidate_keys,
                )
                if chip:
                    chips.append(chip)
            if not chips:
                continue
            label = title if title else "Metrics"
            output_lines.append(f"{label}: {'   '.join(chips)}")
        return output_lines

    def _build_training_hud_lines(
        self,
        *,
        ordered_fields: "OrderedDict[str, str]",
        metrics: Dict[str, Any],
    ) -> list[str]:
        safe_metrics = metrics if isinstance(metrics, dict) else {}
        rand_line = self._build_policy_entropy_rand_line(safe_metrics)
        sections: list[tuple[str, tuple[tuple[str, ...], ...]]] = [
            (
                "Optimization",
                (
                    ("train_iter", "updates"),
                    ("sample_store_size",),
                ),
            ),
            (
                "Frontiers",
                (
                    ("frontier_size",),
                    ("live_map_frontier_size",),
                ),
            ),
            (
                "Representation",
                (
                    ("contrastive_loss",),
                    ("prototype_top1_accuracy",),
                ),
            ),
                (
                    "Intrinsic",
                    (
                        ("rtotal",),
                        ("rh",),
                        ("rz",),
                        ("policy_loss", "actor_loss"),
                        ("critic_loss",),
                        ("rtotal_std",),
                    ),
                ),
        ]
        if self._optional_bool(safe_metrics.get("use_per")):
            sections.extend(
                [
                    (
                        "PER",
                        (
                            ("use_per",),
                            ("per_alpha",),
                            ("per_beta_current", "per_beta"),
                        ),
                    ),
                    (
                        "PER PRIORITY",
                        (
                            ("per_priority_mean",),
                            ("per_priority_max",),
                        ),
                    ),
                    (
                        "PER WEIGHT",
                        (
                            ("per_is_weight_mean",),
                            ("per_is_weight_min",),
                        ),
                    ),
                ]
            )
        output_lines: list[str] = []
        for section_title, key_groups in sections:
            if section_title == "Intrinsic":
                primary_chips: list[str] = []
                secondary_chips: list[str] = []
                for group_index, key_group in enumerate(key_groups):
                    chip = self._build_field_chip_from_candidates(
                        ordered_fields=ordered_fields,
                        keys=key_group,
                    )
                    if not chip:
                        continue
                    if group_index < 3 or key_group == ("rtotal_std",):
                        primary_chips.append(chip)
                    else:
                        secondary_chips.append(chip)
                if rand_line:
                    secondary_chips.append(rand_line)
                if primary_chips:
                    output_lines.append(f"{section_title}: {'   '.join(primary_chips)}")
                if secondary_chips:
                    output_lines.append(f"{section_title}: {'   '.join(secondary_chips)}")
                continue
            chips: list[str] = []
            for key_group in key_groups:
                chip = self._build_field_chip_from_candidates(
                    ordered_fields=ordered_fields,
                    keys=key_group,
                )
                if chip:
                    chips.append(chip)
            if not chips:
                continue
            # Keep one-line diagnostics without truncation.
            line = f"{section_title}: {'   '.join(chips)}"
            output_lines.append(line)
        return output_lines

    def _build_policy_entropy_rand_line(self, metrics: Dict[str, Any]) -> str:
        current_choices = self._optional_float(metrics.get("policy_entropy_choices"))
        if current_choices is None:
            policy_entropy_mean = self._optional_float(metrics.get("policy_entropy_mean"))
            if policy_entropy_mean is not None and math.isfinite(policy_entropy_mean):
                current_choices = math.exp(policy_entropy_mean)
        target_choices = self._optional_float(metrics.get("target_entropy_choices"))
        if target_choices is None:
            target_entropy = self._optional_float(metrics.get("target_entropy"))
            if target_entropy is not None and math.isfinite(target_entropy):
                target_choices = math.exp(target_entropy)
        if current_choices is None or target_choices is None:
            return ""
        if not math.isfinite(current_choices) or not math.isfinite(target_choices):
            return ""
        return f"Rand: {float(current_choices):.1f} / {float(target_choices):.1f}"

    def _build_field_chip_from_candidates(
        self,
        *,
        ordered_fields: "OrderedDict[str, str]",
        keys: Sequence[str],
    ) -> str:
        for key in keys:
            if str(key).strip().lower() == "rtotal_std":
                continue
            lookup_key = self._lookup_field_key(ordered_fields, key)
            if lookup_key is None:
                continue
            if str(lookup_key).strip().lower() == "rtotal_std":
                continue
            chip = self._format_field_chip(key=lookup_key, value=ordered_fields[lookup_key])
            if chip:
                return chip
        return ""


    def _resolve_metrics_char_capacity(self) -> int:
        return 46

    def _build_status_text(self) -> str:
        return ""

    def _summarize_prediction_error(self, raw_error: Any) -> str:
        text = str(raw_error or "").strip()
        if not text:
            return ""
        lowered = text.lower()
        if "line budget exceeded" in lowered:
            return "line budget"
        if "timeout" in lowered:
            return "timeout"
        if "must return a dictionary" in lowered:
            return "return dict"
        if "missing callable" in lowered:
            return "missing fn"
        if "signature" in lowered:
            return "bad signature"
        if ":" in text:
            phase, message = text.split(":", 1)
            phase_text = str(phase).strip().lower()
            message_text = " ".join(str(message).strip().split())
            short_message = message_text[:18].rstrip(".,;:")
            if phase_text:
                return f"{phase_text}:{short_message}" if short_message else phase_text
        compact = " ".join(text.split())
        return compact[:20].rstrip(".,;:")


    def _extract_caption_fields(self, caption: str) -> tuple[str, "OrderedDict[str, str]"]:
        title = caption
        body = ""
        if "|" in caption:
            title, body = caption.split("|", 1)
        normalized_title = self._format_agent_title(title)
        fields: "OrderedDict[str, str]" = OrderedDict()
        for raw_part in body.replace("|", " ").split():
            if "=" not in raw_part:
                continue
            key, value = raw_part.split("=", 1)
            key = str(key).strip()
            value = str(value).strip()
            if not key or key in fields:
                continue
            fields[key] = value
        return normalized_title, fields

    def _merge_metric_fields(
        self,
        *,
        ordered_fields: "OrderedDict[str, str]",
        metrics: Dict[str, Any],
    ) -> "OrderedDict[str, str]":
        merged: "OrderedDict[str, str]" = OrderedDict(ordered_fields)
        for key, value in metrics.items():
            normalized_key = str(key).strip()
            if not normalized_key or self._lookup_field_key(merged, normalized_key) is not None:
                continue
            formatted = self._format_metric_value(value)
            if formatted == "":
                continue
            merged[normalized_key] = formatted
        return merged

    def _lookup_field_key(
        self,
        ordered_fields: "OrderedDict[str, str]",
        lookup: str,
    ) -> Optional[str]:
        target = str(lookup).strip().lower()
        for key in ordered_fields:
            if str(key).strip().lower() == target:
                return key
        return None

    def _format_agent_title(self, raw: str) -> str:
        value = str(raw or "").strip().replace("_", " ")
        if not value:
            return "Exploration"

        lowered = value.lower()
        if lowered == "bfs":
            return lowered.upper()
        words = [word for word in value.split() if word]
        if not words:
            return "Exploration"
        return " ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in words)

    def _format_field_chip(self, *, key: str, value: Any) -> str:
        label = self._format_field_label(key)
        formatted_value = self._format_field_value(key=key, value=value)
        if not label or not formatted_value:
            return ""
        return f"{label} {formatted_value}"

    def _format_field_value(self, *, key: str, value: Any) -> str:
        formatted = self._format_metric_value(value)
        if not formatted:
            return ""
        return formatted

    def _format_field_label(self, key: str) -> str:
        lookup = str(key).strip().lower()
        labels = {
            "episode": "EP",
            "ep": "EP",
            "world_index": "WORLD",
            "map_name": "MAP",
            "step": "STEP",
            "global_step": "GLOBAL",
            "action": "ACTION",
            "reward": "REWARD",
            "rh": "RH",
            "rz": "RZ",
            "rtotal": "RTOTAL",
            "train_r": "TRAIN",
            "progress_phase": "PHASE",
            "train_phase": "LEARN",
            "learning_phase": "LEARN",
            "train_schedule": "SCHED",
            "train_updates_per_event": "UPD_EVT",
            "ig_norm": "IG_NORM",
            "epsilon": "EPS",
            "eps": "EPS",
            "td": "TD",
            "mono": "MONO",
            "pfx": "PFX",
            "xb": "XB",
            "q": "Q",
            "unk": "UNKNOWN",
            "unknown_hit_ratio": "UNK_HIT",
            "sample_store_size": "SAMPLES",
            "sample_duplicate_skips": "DEDUP",
            "queued": "QUEUED",
            "updates": "TRAIN_IT",
            "train_iter": "TRAIN_IT",
            "policy_updates": "POLICY_UPD",
            "dynamics_updates": "DYN_UPD",
            "use_per": "ENABLED",
            "per_alpha": "ALPHA",
            "per_beta": "BETA",
            "per_beta_current": "BETA",
            "per_priority_mean": "P_MEAN",
            "per_priority_max": "P_MAX",
            "per_is_weight_mean": "W_MEAN",
            "per_is_weight_min": "W_MIN",
            "contrastive_loss": "CONTRASTIVE",
            "prototype_top1_accuracy": "TOP1_ACC",
            "mean_positive_logit": "POS_LOGIT",
            "mean_max_negative_logit": "MAX_NEG_LOGIT",
            "rh_mean": "RH_MEAN",
            "rh_std": "RH_STD",
            "rz_mean": "RZ_MEAN",
            "rz_std": "RZ_STD",
            "rtotal_mean": "RTOTAL_MEAN",
            "rtotal_std": "RTOTAL_STD",
            "actor_loss": "POLICY",
            "policy_loss": "POLICY",
            "critic_loss": "CRITIC",
            "current_dynamics_class": "CUR_CLASS",
            "predicted_dynamics_class": "PRED_CLASS",
            "predicted_dynamics_confidence": "PRED_CONF",
            "softmax_temperature": "SOFTMAX_T",
            "active_dynamics_classes": "ACTIVE_CLASS",
            "known_dynamics_classes": "KNOWN_CLASS",
            "observed_dynamics_classes": "OBS_CLASS",
            "learning_starts": "WARMUP_AT",
            "tsne_points": "TSNE_N",
            "done": "DONE",
            "added_count": "ADDED",
            "dataset_size": "CANON",
            "frontier_size": "TOTAL_FRONTIER",
            "live_map_frontier_size": "MAP_FRONTIER",
            "pending_count": "PENDING",
            "resume_pending": "RESUME",
        }
        if lookup in labels:
            return labels[lookup]
        return str(key).strip().upper().replace(" ", "_")

    def _wrap_chips(
        self,
        chips: list[str],
        *,
        max_chars: int,
        max_lines: int,
    ) -> list[str]:
        lines: list[str] = []
        current: list[str] = []
        current_length = 0
        for chip in chips:
            if not chip:
                continue
            separator = 3 if current else 0
            projected = current_length + separator + len(chip)
            if current and projected > max_chars:
                lines.append("   ".join(current))
                if len(lines) >= max_lines:
                    break
                current = [chip]
                current_length = len(chip)
                continue
            current.append(chip)
            current_length = projected

        if current and len(lines) < max_lines:
            lines.append("   ".join(current))
        return lines

    def _format_metric_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, int):
            return str(int(value))
        if isinstance(value, float):
            return self._format_metric(value)
        text = str(value).strip()
        if not text:
            return ""
        parsed = self._optional_float(text)
        if parsed is not None and text not in {"nan", "NaN", "inf", "-inf"}:
            return self._format_metric(parsed)
        return text

    def _record_dashboard_history(
        self,
        *,
        current_step: int,
        metrics: Dict[str, Any],
        display: Optional[Dict[str, Any]] = None,
    ) -> None:
        dashboard_spec = self._resolve_dashboard_spec(display=display)
        top_left = dashboard_spec["top_left"]
        bottom_left = dashboard_spec["bottom_left"]
        top_left_enabled = self._is_dashboard_panel_enabled(top_left)
        bottom_left_enabled = self._is_dashboard_panel_enabled(bottom_left)
        contrastive_loss = self._first_optional_float(
            metrics=metrics,
            keys=tuple(top_left.get("left_keys") or ()),
        ) if top_left_enabled else None
        prototype_top1_accuracy = self._first_optional_float(
            metrics=metrics,
            keys=tuple(top_left.get("right_keys") or ()),
        ) if top_left_enabled else None
        left_keys = tuple(bottom_left.get("left_keys") or ())
        shade_keys = tuple(bottom_left.get("shade_keys") or ())
        right_keys = tuple(bottom_left.get("right_keys") or ())
        policy_intrinsic_reward = (
            self._first_optional_float(metrics=metrics, keys=(str(left_keys[0]),))
            if bottom_left_enabled and left_keys
            else None
        )
        bottom_left_shade_value = (
            self._first_optional_float(metrics=metrics, keys=(str(shade_keys[0]),))
            if bottom_left_enabled and shade_keys
            else None
        )
        transition_intrinsic_reward = (
            self._first_optional_float(metrics=metrics, keys=(str(right_keys[0]),))
            if bottom_left_enabled and right_keys
            else None
        )

        self._dashboard_step_history.append(int(current_step))
        self._dashboard_info_gain_history.append(
            float(contrastive_loss) if contrastive_loss is not None else float("nan")
        )
        self._dashboard_dynamics_nll_history.append(
            float(policy_intrinsic_reward) if policy_intrinsic_reward is not None else float("nan")
        )
        self._dashboard_aux_metric_a_history.append(
            float(prototype_top1_accuracy) if prototype_top1_accuracy is not None else float("nan")
        )
        self._dashboard_aux_metric_b_history.append(
            float(bottom_left_shade_value) if bottom_left_shade_value is not None else float("nan")
        )
        self._dashboard_transition_reward_history.append(
            float(transition_intrinsic_reward) if transition_intrinsic_reward is not None else float("nan")
        )
        tracked_keys: list[str] = []
        for panel in (top_left, bottom_left):
            if not self._is_dashboard_panel_enabled(panel):
                continue
            for key_group_name in ("left_keys", "right_keys", "shade_keys"):
                raw_keys = panel.get(key_group_name)
                if not isinstance(raw_keys, (list, tuple)):
                    continue
                for raw_key in raw_keys:
                    key = str(raw_key).strip().lower()
                    if not key or key in tracked_keys:
                        continue
                    tracked_keys.append(key)
                    self._append_dashboard_metric_history(
                        key,
                        self._first_optional_float(metrics=metrics, keys=(key,)),
                    )

    def _append_dashboard_metric_history(self, key: str, value: Optional[float]) -> None:
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            return
        history = self._dashboard_metric_histories.get(normalized_key)
        if history is None:
            history = deque(maxlen=self.dashboard_history_limit)
            self._dashboard_metric_histories[normalized_key] = history
        history.append(float(value) if value is not None else float("nan"))

    def _dashboard_metric_history(self, key: str) -> list[float]:
        history = self._dashboard_metric_histories.get(str(key).strip().lower())
        if history is None:
            return [float("nan")] * int(len(self._dashboard_step_history))
        values = list(history)
        missing = max(0, int(len(self._dashboard_step_history)) - len(values))
        if missing <= 0:
            return values
        return ([float("nan")] * missing) + values

    def restore_dashboard_history(self, history: Mapping[str, Any]) -> int:
        if not self.enabled or not isinstance(history, MappingABC):
            return 0
        raw_steps = history.get("steps")
        raw_metrics = history.get("metrics")
        if not isinstance(raw_steps, SequenceABC) or isinstance(raw_steps, (str, bytes)):
            return 0
        if not isinstance(raw_metrics, MappingABC):
            return 0
        raw_step_values = list(raw_steps)
        if not raw_step_values:
            return 0

        limit = int(self.dashboard_history_limit)
        start = max(0, len(raw_step_values) - limit)
        selected_steps = raw_step_values[start:]
        self._dashboard_step_history.clear()
        self._dashboard_info_gain_history.clear()
        self._dashboard_dynamics_nll_history.clear()
        self._dashboard_aux_metric_a_history.clear()
        self._dashboard_aux_metric_b_history.clear()
        self._dashboard_transition_reward_history.clear()
        self._dashboard_metric_histories.clear()

        last_step = 0
        for index, raw_step in enumerate(selected_steps):
            parsed_step = self._optional_int(raw_step)
            if parsed_step is None:
                parsed_step = last_step + 1 if index > 0 else 0
            last_step = int(parsed_step)
            self._dashboard_step_history.append(int(parsed_step))

        selected_count = len(self._dashboard_step_history)
        for raw_key, raw_values in raw_metrics.items():
            key = str(raw_key).strip().lower()
            if (
                not key
                or not isinstance(raw_values, SequenceABC)
                or isinstance(raw_values, (str, bytes))
            ):
                continue
            values = list(raw_values)
            metric_history = deque(maxlen=limit)
            for source_index in range(start, start + selected_count):
                raw_value = values[source_index] if source_index < len(values) else None
                parsed_value = self._optional_float(raw_value)
                metric_history.append(
                    float(parsed_value) if parsed_value is not None else float("nan")
                )
            self._dashboard_metric_histories[key] = metric_history

        dashboard_spec = self._resolve_dashboard_spec(display=None)
        top_left = dashboard_spec["top_left"]
        bottom_left = dashboard_spec["bottom_left"]

        def extend_legacy_history(target: Deque[float], keys: Sequence[Any]) -> None:
            key = str(keys[0]).strip().lower() if keys else ""
            values = self._dashboard_metric_history(key) if key else []
            for index in range(selected_count):
                value = values[index] if index < len(values) else float("nan")
                target.append(
                    float(value)
                    if self._optional_float(value) is not None
                    else float("nan")
                )

        extend_legacy_history(
            self._dashboard_info_gain_history,
            tuple(top_left.get("left_keys") or ()),
        )
        extend_legacy_history(
            self._dashboard_aux_metric_a_history,
            tuple(top_left.get("right_keys") or ()),
        )
        extend_legacy_history(
            self._dashboard_dynamics_nll_history,
            tuple(bottom_left.get("left_keys") or ()),
        )
        extend_legacy_history(
            self._dashboard_aux_metric_b_history,
            tuple(bottom_left.get("shade_keys") or ()),
        )
        extend_legacy_history(
            self._dashboard_transition_reward_history,
            tuple(bottom_left.get("right_keys") or ()),
        )
        return int(selected_count)


    def _compute_dashboard_layout(
        self,
        *,
        display: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        dashboard_spec = self._resolve_dashboard_spec(display=display)
        projection_panel = self._resolve_display_panel_spec(
            display=display,
            key="projection",
        )
        class_probability_panel = self._resolve_display_panel_spec(
            display=display,
            key="class_probability",
        )

        panel_enabled = {
            "top_left": self._is_dashboard_panel_enabled(dashboard_spec["top_left"]),
            "bottom_left": self._is_dashboard_panel_enabled(dashboard_spec["bottom_left"]),
            "projection": self._is_dashboard_panel_enabled(projection_panel),
            "class_probability": self._is_dashboard_panel_enabled(class_probability_panel),
        }

        left_panels: list[tuple[str, float]] = []
        if panel_enabled["top_left"]:
            left_panels.append(("top_left", float(self._DASHBOARD_TOP_LEFT_HEIGHT_RATIO)))
        if panel_enabled["bottom_left"]:
            left_panels.append(("bottom_left", float(self._DASHBOARD_BOTTOM_LEFT_HEIGHT_RATIO)))

        right_panels: list[tuple[str, float]] = []
        if panel_enabled["projection"]:
            right_panels.append(("projection", float(self._DASHBOARD_PROJECTION_HEIGHT_RATIO)))
        if panel_enabled["class_probability"]:
            right_panels.append(
                ("class_probability", float(self._DASHBOARD_CLASS_PROBABILITY_HEIGHT_RATIO))
            )

        left_column_enabled = bool(left_panels)
        right_column_enabled = bool(right_panels)

        width_units = 0.0
        if left_column_enabled:
            width_units += float(self._DASHBOARD_LEFT_COLUMN_WIDTH_RATIO)
        if right_column_enabled:
            width_units += float(self._DASHBOARD_RIGHT_COLUMN_WIDTH_RATIO)

        left_height_units = sum(weight for _name, weight in left_panels)
        right_height_units = sum(weight for _name, weight in right_panels)
        height_units = max(1.0, float(left_height_units), float(right_height_units))

        full_width_units = (
            float(self._DASHBOARD_LEFT_COLUMN_WIDTH_RATIO)
            + float(self._DASHBOARD_RIGHT_COLUMN_WIDTH_RATIO)
        )
        full_height_units = (
            float(self._DASHBOARD_PROJECTION_HEIGHT_RATIO)
            + float(self._DASHBOARD_CLASS_PROBABILITY_HEIGHT_RATIO)
        )
        width_scale = (
            float(self._DASHBOARD_BASE_WIDTH) - float(self._DASHBOARD_WIDTH_MARGIN_IN)
        ) / max(1.0, full_width_units)
        height_scale = (
            float(self._DASHBOARD_BASE_HEIGHT) - float(self._DASHBOARD_HEIGHT_MARGIN_IN)
        ) / max(1.0, full_height_units)

        figure_width = (
            float(self._DASHBOARD_WIDTH_MARGIN_IN)
            + width_scale * max(1.0, float(width_units))
        )
        figure_height = (
            float(self._DASHBOARD_HEIGHT_MARGIN_IN)
            + height_scale * max(1.0, float(height_units))
        )

        content_left = float(self._DASHBOARD_CONTENT_LEFT)
        content_right = float(self._DASHBOARD_CONTENT_RIGHT)
        content_top = float(self._DASHBOARD_CONTENT_TOP)
        content_bottom = float(self._DASHBOARD_CONTENT_BOTTOM)
        content_width = max(0.1, content_right - content_left)
        content_height = max(0.1, content_top - content_bottom)
        column_gap = float(self._DASHBOARD_COLUMN_GAP)

        panel_rects: Dict[str, tuple[float, float, float, float]] = {
            "top_left": tuple(float(value) for value in self._DASHBOARD_HIDDEN_RECT),
            "bottom_left": tuple(float(value) for value in self._DASHBOARD_HIDDEN_RECT),
            "projection": tuple(float(value) for value in self._DASHBOARD_HIDDEN_RECT),
            "class_probability": tuple(float(value) for value in self._DASHBOARD_HIDDEN_RECT),
        }

        def _stack_column_panels(
            *,
            origin_x: float,
            width: float,
            panels: Sequence[tuple[str, float]],
        ) -> None:
            if not panels:
                return
            if len(panels) == 1:
                name, _weight = panels[0]
                panel_rects[name] = (
                    float(origin_x),
                    float(content_bottom),
                    float(width),
                    float(content_height),
                )
                return
            available_height = max(
                0.08,
                float(content_height) - (float(self._DASHBOARD_ROW_GAP) * float(len(panels) - 1)),
            )
            total_weight = max(1.0, sum(float(weight) for _name, weight in panels))
            current_top = float(content_top)
            for name, weight in panels:
                panel_height = available_height * float(weight) / total_weight
                current_bottom = current_top - panel_height
                panel_rects[name] = (
                    float(origin_x),
                    float(current_bottom),
                    float(width),
                    float(panel_height),
                )
                current_top = current_bottom - float(self._DASHBOARD_ROW_GAP)

        if left_column_enabled and right_column_enabled:
            available_width = max(0.1, content_width - column_gap)
            total_ratio = (
                float(self._DASHBOARD_LEFT_COLUMN_WIDTH_RATIO)
                + float(self._DASHBOARD_RIGHT_COLUMN_WIDTH_RATIO)
            )
            left_width = available_width * float(self._DASHBOARD_LEFT_COLUMN_WIDTH_RATIO) / total_ratio
            right_width = available_width * float(self._DASHBOARD_RIGHT_COLUMN_WIDTH_RATIO) / total_ratio
            _stack_column_panels(
                origin_x=content_left,
                width=left_width,
                panels=left_panels,
            )
            _stack_column_panels(
                origin_x=(content_left + left_width + column_gap),
                width=right_width,
                panels=right_panels,
            )
        elif left_column_enabled:
            _stack_column_panels(
                origin_x=content_left,
                width=content_width,
                panels=left_panels,
            )
        elif right_column_enabled:
            _stack_column_panels(
                origin_x=content_left,
                width=content_width,
                panels=right_panels,
            )

        if (
            left_column_enabled
            and right_column_enabled
            and panel_enabled["bottom_left"]
            and panel_enabled["class_probability"]
        ):
            rect = panel_rects["class_probability"]
            inset = min(
                float(self._DASHBOARD_CLASS_PROBABILITY_LEFT_INSET),
                max(0.0, float(rect[2]) - 0.12),
            )
            if inset > 0.0:
                panel_rects["class_probability"] = (
                    float(rect[0] + inset),
                    float(rect[1]),
                    float(rect[2] - inset),
                    float(rect[3]),
                )

        return {
            "signature": (
                bool(panel_enabled["top_left"]),
                bool(panel_enabled["bottom_left"]),
                bool(panel_enabled["projection"]),
                bool(panel_enabled["class_probability"]),
            ),
            "figure_width": float(figure_width),
            "figure_height": float(figure_height),
            "panel_enabled": panel_enabled,
            "panel_rects": panel_rects,
        }


    def _update_dashboard(
        self,
        *,
        metrics: Optional[Dict[str, Any]],
        visitation_heatmap: Optional[Any],
        display: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        _ = visitation_heatmap
        if metrics is None:
            metrics = {}

        current_step = self._resolve_dashboard_step(metrics=metrics)
        self._record_dashboard_history(
            current_step=current_step,
            metrics=metrics,
            display=display,
        )
        self.dashboard_error = None


    def _resolve_dashboard_step(self, *, metrics: Dict[str, Any]) -> int:
        global_step = self._optional_int(metrics.get("global_step"))
        if global_step is not None:
            return int(global_step)
        local_step = self._optional_int(metrics.get("step"))
        if local_step is not None:
            return int(local_step)
        return 0

    def update_dashboard_only(
        self,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        visitation_heatmap: Optional[Any] = None,
        display: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        if self._last_payload is None:
            self._last_payload = {
                "caption": "",
                "metrics": dict(metrics or {}),
                "visitation_heatmap": visitation_heatmap,
                "class_rows": [],
                "display": dict(display) if isinstance(display, dict) else {},
            }
        else:
            self._last_payload["metrics"] = dict(metrics or {})
            self._last_payload["visitation_heatmap"] = visitation_heatmap
            if isinstance(display, dict):
                self._last_payload["display"] = dict(display)
        self._update_dashboard(
            metrics=dict(self._last_payload.get("metrics") or {}),
            visitation_heatmap=visitation_heatmap,
            display=self._last_payload.get("display") if isinstance(self._last_payload, dict) else None,
        )
        self._emit_progress()
        self._emit_snapshot()

    def update_payload_only(
        self,
        *,
        caption: Any = _PAYLOAD_UNSET,
        metrics: Any = _PAYLOAD_UNSET,
        visitation_heatmap: Any = _PAYLOAD_UNSET,
        class_rows: Any = _PAYLOAD_UNSET,
        display: Any = _PAYLOAD_UNSET,
        program_context: Any = _PAYLOAD_UNSET,
        board_state: Any = _PAYLOAD_UNSET,
        merge_metrics: bool = False,
        update_dashboard: bool = True,
        force_snapshot: bool = False,
        bypass_snapshot_delivery_gate: bool = False,
    ) -> None:
        if not self.enabled:
            return
        base_payload = self._displayed_payload or self._last_payload or {}
        payload: Dict[str, Any] = {
            "caption": str(base_payload.get("caption") or ""),
            "metrics": dict(base_payload.get("metrics") or {}),
            "visitation_heatmap": base_payload.get("visitation_heatmap"),
            "class_rows": [
                dict(row)
                for row in (base_payload.get("class_rows") or ())
                if isinstance(row, dict)
            ],
            "display": dict(base_payload.get("display") or {}),
            "program_context": (
                dict(base_payload.get("program_context"))
                if isinstance(base_payload.get("program_context"), dict)
                else None
            ),
            "board_state": (
                dict(base_payload.get("board_state"))
                if isinstance(base_payload.get("board_state"), dict)
                else None
            ),
        }
        if caption is not _PAYLOAD_UNSET:
            payload["caption"] = str(caption or "")
        if metrics is not _PAYLOAD_UNSET:
            next_metrics = dict(metrics or {})
            if merge_metrics:
                merged_metrics = dict(payload.get("metrics") or {})
                merged_metrics.update(next_metrics)
                payload["metrics"] = merged_metrics
            else:
                payload["metrics"] = next_metrics
        if visitation_heatmap is not _PAYLOAD_UNSET:
            payload["visitation_heatmap"] = visitation_heatmap
        if class_rows is not _PAYLOAD_UNSET:
            payload["class_rows"] = [
                dict(row)
                for row in (class_rows or ())
                if isinstance(row, dict)
            ]
        if display is not _PAYLOAD_UNSET:
            payload["display"] = dict(display) if isinstance(display, dict) else {}
        if program_context is not _PAYLOAD_UNSET:
            payload["program_context"] = (
                dict(program_context)
                if isinstance(program_context, dict)
                else None
            )
        if board_state is not _PAYLOAD_UNSET:
            payload["board_state"] = (
                dict(board_state)
                if isinstance(board_state, dict)
                else None
            )

        self._last_payload = payload
        self._displayed_payload = payload
        if update_dashboard:
            self._update_dashboard(
                metrics=payload.get("metrics"),
                visitation_heatmap=payload.get("visitation_heatmap"),
                display=payload.get("display"),
            )
        self._emit_progress()
        self._emit_snapshot(
            force=force_snapshot,
            bypass_delivery_gate=bypass_snapshot_delivery_gate,
        )

    def _optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _first_optional_float(
        self,
        *,
        metrics: Dict[str, Any],
        keys: Sequence[str],
    ) -> Optional[float]:
        for key in keys:
            parsed = self._optional_float(metrics.get(key))
            if parsed is not None:
                return float(parsed)
        return None

    def _optional_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not float(parsed).is_integer():
            return None
        return int(parsed)

    def _resolve_current_class_index(
        self,
        *,
        metrics: Optional[Dict[str, Any]],
        payload: Any,
    ) -> Optional[int]:
        if isinstance(payload, dict):
            current_class_index = self._optional_int(payload.get("current_class_index"))
            if current_class_index is not None:
                return int(current_class_index)
            raw_current = payload.get("current_transition")
            if isinstance(raw_current, dict):
                current_class_index = self._optional_int(raw_current.get("class_index"))
                if current_class_index is not None:
                    return int(current_class_index)
        if isinstance(metrics, dict):
            current_class_index = self._optional_int(metrics.get("current_dynamics_class"))
            if current_class_index is not None:
                return int(current_class_index)
        return None

    def _resolve_current_class_id(
        self,
        *,
        metrics: Optional[Dict[str, Any]],
        payload: Any,
    ) -> Optional[int]:
        return self._resolve_current_class_index(metrics=metrics, payload=payload)

    def _format_current_class_text(
        self,
        current_class_index: Optional[int],
        *,
        payload: Any = None,
    ) -> str:
        group_label = None
        group_class_index = None
        if current_class_index is None or int(current_class_index) <= 0:
            if isinstance(payload, dict):
                raw_explained = payload.get("current_transition_explained_by_current_program")
                if raw_explained is False or bool(payload.get("current_is_new_dynamics_class")):
                    return "Current: ?"
            return "Current: none (0)"
        class_count = None
        if isinstance(payload, dict):
            raw_label = payload.get("current_group_label")
            if isinstance(raw_label, str) and raw_label.strip():
                group_label = raw_label.strip()
            elif isinstance(payload.get("current_group_id"), str) and str(payload.get("current_group_id")).strip():
                group_label = str(payload.get("current_group_id")).strip()
            elif isinstance(payload.get("current_class_label"), str) and str(payload.get("current_class_label")).strip():
                group_label = str(payload.get("current_class_label")).strip()
            raw_group_class_index = self._optional_int(
                payload.get("current_group_class_index")
            )
            if isinstance(raw_group_class_index, int) and int(raw_group_class_index) > 0:
                group_class_index = int(raw_group_class_index)
            raw_count = payload.get("current_class_count")
            if isinstance(raw_count, int) and raw_count >= 0:
                class_count = int(raw_count)
        resolved_class_index = (
            int(group_class_index)
            if isinstance(group_class_index, int) and int(group_class_index) > 0
            else int(current_class_index)
        )
        suffix_parts = []
        if isinstance(group_label, str) and group_label:
            suffix_parts.append(group_label)
        if isinstance(class_count, int):
            suffix_parts.append(f"{class_count} canon")
        suffix = f" [{' | '.join(suffix_parts)}]" if suffix_parts else ""
        return f"Current: C{int(resolved_class_index)}{suffix}"

    def _format_class_text(
        self,
        current_class_index: Optional[int],
        *,
        payload: Any = None,
    ) -> str:
        return self._format_current_class_text(current_class_index, payload=payload)

    def _is_dashboard_panel_enabled(self, panel: Any) -> bool:
        if not isinstance(panel, dict):
            return False
        enabled = self._optional_bool(panel.get("enabled"))
        return enabled is not False

    def _resolve_display_panel_spec(
        self,
        *,
        display: Optional[Dict[str, Any]],
        key: str,
    ) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"enabled": True}
        if not isinstance(display, dict):
            return spec
        raw_panel = display.get(str(key))
        if raw_panel is False:
            spec["enabled"] = False
            return spec
        if not isinstance(raw_panel, dict):
            return spec
        spec.update(raw_panel)
        return spec

    def _optional_bool(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if not text:
            return None
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        return None

    def _resolve_dashboard_spec(self, *, display: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        top_left_default = {
            "enabled": True,
            "title": "Contrastive Learning",
            "left_keys": ("contrastive_loss",),
            "left_label": "contrastive loss",
            "left_color": "#0f766e",
            "left_ylabel": "loss",
            "right_keys": ("prototype_top1_accuracy",),
            "right_label": "top-1 accuracy",
            "right_color": "#1D4ED8",
            "right_ylabel": "accuracy",
            "right_ylim": (-0.02, 1.02),
        }
        bottom_left_default = {
            "enabled": True,
            "title": "Intrinsic Reward",
            "left_keys": ("rtotal_mean", "rh_mean", "rz_mean"),
            "left_label": "train-batch mean intrinsic reward",
            "left_color": "#b45309",
            "left_ylabel": "train batch mean reward",
            "right_keys": ("rtotal", "rh", "rz"),
            "right_label": "cur transition intrinsic reward",
            "right_color": "#1D4ED8",
            "right_ylabel": "current transition reward",
            "shade_keys": ("rtotal_std",),
            "shade_color": "#f59e0b",
        }
        if not isinstance(display, dict):
            return {
                "top_left": top_left_default,
                "bottom_left": bottom_left_default,
            }

        raw_dashboard = display.get("dashboard")
        if not isinstance(raw_dashboard, dict):
            return {
                "top_left": top_left_default,
                "bottom_left": bottom_left_default,
            }

        def _merge_panel_spec(default_panel: Dict[str, Any], raw_panel: Any) -> Dict[str, Any]:
            merged = dict(default_panel)
            if raw_panel is False:
                merged["enabled"] = False
                return merged
            if not isinstance(raw_panel, dict):
                return merged
            for key, value in raw_panel.items():
                if key in {"left_keys", "right_keys", "shade_keys"}:
                    if isinstance(value, str):
                        merged[key] = (str(value).strip(),)
                    elif isinstance(value, (list, tuple)):
                        merged[key] = tuple(
                            str(item).strip()
                            for item in value
                            if str(item).strip()
                        )
                else:
                    merged[key] = value
            return merged

        return {
            "top_left": _merge_panel_spec(top_left_default, raw_dashboard.get("top_left")),
            "bottom_left": _merge_panel_spec(bottom_left_default, raw_dashboard.get("bottom_left")),
        }

    def _should_smooth_dashboard_series(
        self,
        *,
        panel: Any,
        key: str,
        default: bool = True,
    ) -> bool:
        if not isinstance(panel, dict):
            return bool(default)
        resolved = self._optional_bool(panel.get(str(key)))
        if resolved is None:
            return bool(default)
        return bool(resolved)

    def _format_metric(self, value: Any) -> str:
        parsed = self._optional_float(value)
        if parsed is None:
            return "-"
        if float(parsed).is_integer():
            return str(int(parsed))
        magnitude = abs(parsed)
        if magnitude < 1e-3 or magnitude >= 1e3:
            return f"{parsed:.2e}"
        return f"{parsed:.4f}"

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "dashboard_enabled": bool(self.enabled),
            "dashboard_error": self.dashboard_error,
            "dashboard_context_lines": list(self.dashboard_context_lines),
        }

    def has_payload(self) -> bool:
        return self._displayed_payload is not None or self._last_payload is not None

    def set_snapshot_callback(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        self._snapshot_callback = callback
        self._emit_snapshot()

    def set_snapshot_enabled_provider(
        self,
        provider: Optional[Callable[[], bool]],
    ) -> None:
        self._snapshot_enabled_provider = provider if callable(provider) else None
        self._emit_snapshot()

    def emit_snapshot(
        self,
        *,
        force: bool = False,
        bypass_delivery_gate: bool = False,
    ) -> None:
        self._emit_snapshot(
            force=force,
            bypass_delivery_gate=bypass_delivery_gate,
        )

    def is_snapshot_delivery_enabled(self) -> bool:
        if not self.enabled:
            return False
        enabled_provider = self._snapshot_enabled_provider
        if callable(enabled_provider):
            try:
                return bool(enabled_provider())
            except (TypeError, ValueError, RuntimeError, OSError):
                return False
        return True

    def should_prepare_snapshot_payload(
        self,
        *,
        force: bool = False,
        bypass_delivery_gate: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        if bool(force):
            if bool(bypass_delivery_gate):
                return True
            return self.is_snapshot_delivery_enabled()
        if (
            perf_counter() - float(self._last_snapshot_emit_at)
            < float(self._snapshot_min_interval_sec)
        ):
            return False
        if bool(bypass_delivery_gate):
            return True
        return self.is_snapshot_delivery_enabled()

    def set_visitation_heatmap_enabled_provider(
        self,
        provider: Optional[Callable[[], bool]],
    ) -> None:
        self._visitation_heatmap_enabled_provider = (
            provider if callable(provider) else None
        )

    def is_visitation_heatmap_enabled(self) -> bool:
        if not self.enabled:
            return False
        provider = self._visitation_heatmap_enabled_provider
        if not callable(provider):
            return True
        try:
            return bool(provider())
        except (TypeError, ValueError, RuntimeError, OSError):
            return False

    def set_visitation_heatmap_sparse_class_filter_provider(
        self,
        provider: Optional[Callable[[], bool]],
    ) -> None:
        self._visitation_heatmap_sparse_class_filter_provider = (
            provider if callable(provider) else None
        )

    def should_filter_sparse_visitation_heatmap_classes(self) -> bool:
        if not self.enabled:
            return False
        provider = self._visitation_heatmap_sparse_class_filter_provider
        if not callable(provider):
            return False
        return bool(provider())

    def set_progress_callback(
        self,
        callback: Optional[Callable[[], None]],
    ) -> None:
        self._progress_callback = callback if callable(callback) else None

    def set_agent_diagnostics_provider(
        self,
        provider: Optional[Callable[[], Dict[str, Any]]],
    ) -> None:
        self._agent_diagnostics_provider = provider if callable(provider) else None
        self._emit_snapshot()

    def _safe_snapshot_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {
                str(key): self._safe_snapshot_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._safe_snapshot_value(item) for item in value]
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return self._safe_snapshot_value(value.item())
            except (TypeError, ValueError, RuntimeError):
                return str(value)
        return str(value)

    def _resolve_visualization_config(self) -> Dict[str, Any]:
        resolved = self.env.get_visualization_config()
        return dict(resolved) if isinstance(resolved, dict) else {}

    def _build_agent_snapshot(self) -> Dict[str, Any]:
        provider = self._agent_diagnostics_provider
        if not callable(provider):
            return {}
        try:
            diagnostics = provider()
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            return {"diagnosticsError": str(exc)}
        if not isinstance(diagnostics, dict):
            return {}
        safe_diagnostics = self._safe_snapshot_value(dict(diagnostics))
        return safe_diagnostics if isinstance(safe_diagnostics, dict) else {}

    def build_web_snapshot(self) -> Dict[str, Any]:
        payload = self._displayed_payload or self._last_payload or {}
        display = payload.get("display") if isinstance(payload.get("display"), dict) else {}
        board_state = payload.get("board_state")
        hud_title, hud_text = self._build_hud_lines(payload)
        hud_status = self._build_status_text()
        dashboard_spec = self._resolve_dashboard_spec(display=display)
        dashboard_layout = self._compute_dashboard_layout(
            display=display
        )
        metric_histories = {
            str(key): self._safe_snapshot_value(self._dashboard_metric_history(str(key)))
            for key in sorted(self._dashboard_metric_histories.keys())
        }
        return {
            "caption": str(payload.get("caption") or ""),
            "metrics": self._safe_snapshot_value(dict(payload.get("metrics") or {})),
            "display": self._safe_snapshot_value(dict(display or {})),
            "classRows": self._safe_snapshot_value(
                [dict(row) for row in (payload.get("class_rows") or ()) if isinstance(row, dict)]
            ),
            "programContext": self._safe_snapshot_value(
                dict(payload.get("program_context") or {})
                if isinstance(payload.get("program_context"), dict)
                else {}
            ),
            "visitationHeatmap": self._safe_snapshot_value(payload.get("visitation_heatmap")),
            "dashboardContextLines": list(self.dashboard_context_lines),
            "dashboardSpec": self._safe_snapshot_value(dashboard_spec),
            "dashboardLayout": self._safe_snapshot_value(dashboard_layout),
            "history": {
                "steps": self._safe_snapshot_value(list(self._dashboard_step_history)),
                "metrics": metric_histories,
            },
            "agent": self._build_agent_snapshot(),
            "hud": {
                "title": self._safe_snapshot_value(hud_title),
                "text": self._safe_snapshot_value(hud_text),
                "statusText": self._safe_snapshot_value(hud_status),
                "contextText": self._safe_snapshot_value(self._build_dashboard_context_text()),
            },
            "boardState": dict(board_state) if isinstance(board_state, dict) else None,
            "visualConfig": self._resolve_visualization_config(),
            "diagnostics": self.get_diagnostics(),
        }

    def _emit_snapshot(
        self,
        *,
        force: bool = False,
        bypass_delivery_gate: bool = False,
    ) -> None:
        if not self.should_prepare_snapshot_payload(
            force=force,
            bypass_delivery_gate=bypass_delivery_gate,
        ):
            return
        callback = self._snapshot_callback
        if not callable(callback):
            return
        try:
            callback(self.build_web_snapshot())
            self._last_snapshot_emit_at = perf_counter()
        except (TypeError, ValueError, RuntimeError, OSError):
            return

    def _emit_progress(self) -> None:
        if not self.enabled:
            return
        callback = self._progress_callback
        if not callable(callback):
            return
        try:
            callback()
        except (TypeError, ValueError, RuntimeError, OSError):
            return
