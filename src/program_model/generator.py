import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional

from .patcher import ProgramPatcher


@dataclass
class ProgramPatchCandidate:
    patch_payload: Any
    raw_output: str
    reasoning: Optional[str]
    prompt: str

    def prompt_texts(self) -> Dict[str, str]:
        if isinstance(self.prompt, str) and self.prompt.strip():
            return {"patch_generation": self.prompt}
        return {}


class LLMCallBudgetExceeded(RuntimeError):
    """Raised when patch generation would exceed the configured LLM budget."""


class ProgramPatchGenerator:
    """Generate a single patch payload for world-model program updates."""

    def __init__(
        self,
        llm_predictor,
        patcher: ProgramPatcher,
        additional_instructions: Optional[str] = None,
        usage_tracker=None,
    ):
        self.llm_predictor = llm_predictor
        self.patcher = patcher
        self.usage_tracker = usage_tracker
        if isinstance(additional_instructions, str):
            normalized_instructions = additional_instructions.strip()
        else:
            normalized_instructions = ""
        self.additional_instructions = normalized_instructions
        self.prompt_templates_dir = Path(__file__).resolve().parent / "prompt_templates"

    @property
    def patch_format(self) -> str:
        return self.patcher.patch_format

    def generate(
        self,
        current_source: str,
        failed_cases: List[Dict],
        recent_errors: Optional[List[str]] = None,
        previous_program_source: Optional[str] = None,
        previous_failed_case: Optional[Dict] = None,
        previous_failure_reason: Optional[str] = None,
        previous_smoke_errors: Optional[List[Dict[str, Optional[str]]]] = None,
        regression_witness_cases: Optional[List[Dict]] = None,
        previous_selected_regression_group_ids: Optional[List[str]] = None,
        usage_context: Optional[Dict[str, Any]] = None,
        attempt_callback: Optional[Callable[[int, int], None]] = None,
        generation_record_callback: Optional[
            Callable[[int, int, str, Optional[str], Optional[str], Optional[str]], None]
        ] = None,
        should_stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Optional[ProgramPatchCandidate]:
        del previous_selected_regression_group_ids
        last_text = ""
        last_reasoning: Optional[str] = None
        last_prompt = ""

        recent_errors = list(recent_errors or [])
        regression_witness_cases = list(regression_witness_cases or [])
        previous_smoke_errors = list(previous_smoke_errors or [])
        total_attempts = 1

        if callable(should_stop_callback) and bool(should_stop_callback()):
            raise LLMCallBudgetExceeded(
                "LLM call budget exhausted before patch generation."
            )
        if callable(attempt_callback):
            attempt_callback(1, total_attempts)
        prompt = self._build_patch_generation_prompt(
            current_source=current_source,
            failed_cases=failed_cases,
            recent_errors=recent_errors,
            previous_program_source=previous_program_source,
            previous_failed_case=previous_failed_case,
            previous_failure_reason=previous_failure_reason,
            previous_smoke_errors=previous_smoke_errors,
            regression_witness_cases=regression_witness_cases,
        )
        last_prompt = prompt
        try:
            generated_text, reasoning = self.llm_predictor.generate_text(prompt)
        except Exception as exc:
            if self.usage_tracker is not None:
                self.usage_tracker.record(
                    component="patch_generation",
                    predictor=self.llm_predictor,
                    prompt_text=prompt,
                    response_text=None,
                    outcome="error",
                    error=str(exc),
                    context=usage_context,
                )
            self._safe_record_generation_attempt(
                generation_record_callback=generation_record_callback,
                attempt_number=1,
                total_attempts=total_attempts,
                prompt=prompt,
                response_text=None,
                reasoning=None,
                error=str(exc),
            )
            raise
        parsed = self._parse_patch_payload(generated_text)
        artifact_text = self._format_response_artifact_text(
            parsed_payload=parsed,
            response_text=generated_text,
        )
        if self.usage_tracker is not None:
            self.usage_tracker.record(
                component="patch_generation",
                predictor=self.llm_predictor,
                prompt_text=prompt,
                response_text=generated_text,
                outcome="success",
                context=usage_context,
            )
        self._safe_record_generation_attempt(
            generation_record_callback=generation_record_callback,
            attempt_number=1,
            total_attempts=total_attempts,
            prompt=prompt,
            response_text=artifact_text,
            reasoning=reasoning,
            error=None,
        )
        last_text = artifact_text
        last_reasoning = reasoning

        if parsed is not None:
            return ProgramPatchCandidate(
                patch_payload=parsed,
                raw_output=artifact_text,
                reasoning=reasoning,
                prompt=prompt,
            )

        if not last_text:
            return None
        return ProgramPatchCandidate(
            patch_payload="" if self.patch_format == "line" else {},
            raw_output=last_text,
            reasoning=last_reasoning,
            prompt=last_prompt,
        )

    def _parse_patch_payload(
        self,
        text: str,
    ) -> Optional[Any]:
        try:
            return self.patcher.normalize_patch_payload(text)
        except ValueError:
            return None

    def _format_response_artifact_text(
        self,
        *,
        parsed_payload: Optional[Any],
        response_text: str,
    ) -> str:
        if parsed_payload is None:
            return response_text
        if self.patch_format == "function":
            return json.dumps(parsed_payload, ensure_ascii=False, sort_keys=True)
        if isinstance(parsed_payload, str):
            return parsed_payload
        return json.dumps(parsed_payload, ensure_ascii=False, sort_keys=True)

    def _safe_record_generation_attempt(
        self,
        *,
        generation_record_callback: Optional[
            Callable[[int, int, str, Optional[str], Optional[str], Optional[str]], None]
        ],
        attempt_number: int,
        total_attempts: int,
        prompt: str,
        response_text: Optional[str],
        reasoning: Optional[str],
        error: Optional[str],
    ) -> None:
        if not callable(generation_record_callback):
            return
        try:
            generation_record_callback(
                int(attempt_number),
                int(total_attempts),
                prompt,
                response_text,
                reasoning,
                error,
            )
        except Exception:
            return

    def _render_prompt_template(self, template_name: str, **kwargs: str) -> str:
        template_path = self.prompt_templates_dir / template_name
        template_text = template_path.read_text(encoding="utf-8")
        return Template(template_text).safe_substitute(**kwargs)

    def _build_additional_instructions_block(self) -> str:
        if not self.additional_instructions:
            return ""
        return (
            "Additional experiment instructions:\n"
            f"{self.additional_instructions}\n\n"
        )

    def _sandbox_python_version(self) -> str:
        version_info = sys.version_info
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    def _build_sandbox_runtime_block(self) -> str:
        return (
            "Sandbox runtime:\n"
            f"- Candidate patches are evaluated with Python {self._sandbox_python_version()}.\n"
            "- Use only syntax supported by this Python version.\n\n"
        )

    def _build_patch_generation_prompt(
        self,
        *,
        current_source: str,
        failed_cases: List[Dict[str, Any]],
        recent_errors: List[str],
        previous_program_source: Optional[str],
        previous_failed_case: Optional[Dict[str, Any]],
        previous_failure_reason: Optional[str],
        previous_smoke_errors: List[Dict[str, Optional[str]]],
        regression_witness_cases: List[Dict[str, Any]],
    ) -> str:
        failed_text = self._format_case_list(
            cases=failed_cases,
            predicted_label="accepted_baseline_predicted_next_state",
            mismatch_label="accepted_baseline_predicted_next_state",
            include_group_metadata=False,
            limit=8,
        )
        previous_patch_context_block = self._build_previous_patch_context_block(
            previous_program_source=previous_program_source,
            previous_failed_case=previous_failed_case,
            previous_failure_reason=previous_failure_reason,
        )
        regression_witness_block = self._build_regression_witness_block(
            regression_witness_cases
        )
        previous_smoke_errors_block = self._build_previous_smoke_errors_block(
            previous_smoke_errors
        )
        current_errors_block = self._build_current_errors_block(recent_errors)
        return self._render_prompt_template(
            self.patcher.prompt_template_name,
            additional_instructions_block=self._build_additional_instructions_block(),
            sandbox_runtime_block=self._build_sandbox_runtime_block(),
            current_source=self._format_source_for_prompt(current_source),
            failed_text=failed_text,
            previous_patch_context_block=previous_patch_context_block,
            regression_witness_block=regression_witness_block,
            previous_smoke_errors_block=previous_smoke_errors_block,
            current_errors_block=current_errors_block,
        )

    def _build_previous_patch_context_block(
        self,
        *,
        previous_program_source: Optional[str],
        previous_failed_case: Optional[Dict[str, Any]],
        previous_failure_reason: Optional[str],
    ) -> str:
        has_previous_program = (
            isinstance(previous_program_source, str)
            and previous_program_source.strip()
        )
        has_previous_case = isinstance(previous_failed_case, dict) and previous_failed_case
        if not has_previous_program or not has_previous_case:
            return ""
        previous_case_text = self._format_case_block(
            case=previous_failed_case,
            index=1,
            predicted_label="previous_rejected_patch_predicted_next_state",
            mismatch_label="previous_rejected_patch_predicted_next_state",
        ).strip()
        reason_text = (
            previous_failure_reason.strip()
            if isinstance(previous_failure_reason, str)
            and previous_failure_reason.strip()
            else "none"
        )
        return (
            "Previous rejected patch attempt (reference only):\n"
            "- The accepted baseline remains the only program you may edit.\n"
            "- Use this only to understand what failed; do not edit this rejected patch directly.\n"
            f"- Previous rejection reason: {reason_text}\n"
            "Previous rejected patch program (reference only):\n"
            "```python\n"
            f"{self._format_source_for_prompt(previous_program_source)}"
            "\n```\n\n"
            "Previous rejected patch target result:\n"
            f"{previous_case_text}\n\n"
        )

    def _build_regression_witness_block(
        self,
        regression_witness_cases: List[Dict[str, Any]],
    ) -> str:
        if not regression_witness_cases:
            return ""
        witness_text = self._format_case_list(
            cases=regression_witness_cases,
            predicted_label="previous_rejected_patch_predicted_next_state",
            mismatch_label="previous_rejected_patch_predicted_next_state",
            include_group_metadata=True,
            limit=8,
        )
        return (
            "Protected witness transitions broken by the previous rejected patch:\n"
            "- These are not accepted-baseline failures.\n"
            "- They are guardrails: preserve these behaviors while fixing the target failures.\n"
            "- Any field named previous_rejected_patch_predicted_next_state is the rejected patch output.\n"
            f"{witness_text}\n\n"
        )

    def _build_previous_smoke_errors_block(
        self,
        previous_smoke_errors: List[Dict[str, Optional[str]]],
    ) -> str:
        smoke_text = self._format_previous_smoke_errors(previous_smoke_errors)
        if smoke_text == "- none":
            return ""
        return (
            "Compile/runtime errors from the previous rejected patch attempt:\n"
            "- Avoid repeating these errors in the next patch.\n"
            f"{smoke_text}\n\n"
        )

    def _build_current_errors_block(self, recent_errors: List[str]) -> str:
        errors_text = "\n".join(f"- {e}" for e in recent_errors[:10]) or "- none"
        if errors_text == "- none":
            return ""
        return (
            "Recent compile/runtime errors observed while evaluating the accepted baseline target failures:\n"
            "- Avoid repeating these errors in the next patch.\n"
            f"{errors_text}\n\n"
        )

    def _format_previous_smoke_errors(
        self,
        previous_smoke_errors: List[Dict[str, Optional[str]]],
    ) -> str:
        smoke_lines: List[str] = []
        for row in previous_smoke_errors[:5]:
            if not isinstance(row, dict):
                continue
            phase = row.get("phase") or "unknown_phase"
            message = row.get("message") or "unknown_error"
            exception_type = row.get("exception_type")
            if exception_type:
                smoke_lines.append(f"- [{phase}/{exception_type}] {message}")
            else:
                smoke_lines.append(f"- [{phase}] {message}")
        return "\n".join(smoke_lines) if smoke_lines else "- none"

    def _format_case_list(
        self,
        *,
        cases: List[Dict[str, Any]],
        predicted_label: str,
        mismatch_label: str,
        include_group_metadata: bool,
        limit: int,
    ) -> str:
        blocks = [
            self._format_case_block(
                case=case,
                index=i,
                predicted_label=predicted_label,
                mismatch_label=mismatch_label,
                include_group_metadata=include_group_metadata,
            )
            for i, case in enumerate(cases[:limit], 1)
        ]
        return "\n".join(blocks) or "No cases."

    def _format_case_block(
        self,
        case: Dict[str, Any],
        index: int,
        predicted_label: str,
        mismatch_label: str,
        include_group_metadata: bool = False,
    ) -> str:
        expected_difference_text = self._format_difference(
            case.get("expected_difference", case.get("difference"))
        )
        predicted_difference_text = (
            self._format_difference(case.get("predicted_difference"))
            if case.get("predicted_difference") is not None
            else "none"
        )
        mismatch_difference_text = (
            self._format_difference(case.get("mismatch_difference"))
            if case.get("mismatch_difference") is not None
            else "none"
        )
        lines = [f"Case {index}"]
        if include_group_metadata:
            commit_version = case.get("commit_version")
            subgroup_id = case.get("group_id")
            if commit_version is not None:
                lines.append(f"- commit_version: {commit_version}")
            if subgroup_id is not None:
                lines.append(f"- subgroup_id: {subgroup_id}")
        lines.extend(
            [
                f"- action: {case.get('action', '')}",
                f"- state: {case.get('state', '')}",
                (
                    "- expected_next_state (gt_next_state): "
                    f"{case.get('expected_next_state', '')}"
                ),
                (
                    f"- predicted_next_state ({predicted_label}): "
                    f"{case.get('predicted_next_state', '')}"
                ),
                (
                    "- expected_difference (state -> gt_next_state): "
                    f"{expected_difference_text}"
                ),
                (
                    f"- predicted_difference (state -> {predicted_label}): "
                    f"{predicted_difference_text}"
                ),
                (
                    f"- mismatch_difference ({mismatch_label} -> gt_next_state): "
                    f"{mismatch_difference_text}"
                ),
            ]
        )
        if case.get("error"):
            lines.append(f"- error: {case.get('error')}")
        return "\n".join(lines) + "\n"

    def _format_difference(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def _format_source_for_prompt(self, source: Optional[str]) -> str:
        if not isinstance(source, str):
            return ""
        return source.rstrip("\n")
