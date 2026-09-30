from dataclasses import dataclass
from typing import Any, Dict, Optional

from .sandbox import ProgramSandbox, SandboxError
from .state_codec import dump_state_json, parse_state_json


def _clone_plain_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_plain_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_plain_structure(item) for item in value]
    return value


@dataclass
class ProgramPredictionResult:
    success: bool
    predicted_state_obj: Optional[Dict[str, Any]] = None
    predicted_state_json: Optional[str] = None
    predicted_state_key: Optional[str] = None
    error: Optional[SandboxError] = None


class ProgramWorldModelPredictor:
    """
    Program-only predictor.
    """

    def __init__(self, sandbox: ProgramSandbox):
        self.sandbox = sandbox
        self._source: Optional[str] = None
        self._namespace = None
        self._compile_error: Optional[SandboxError] = None

    @property
    def source(self) -> Optional[str]:
        return self._source

    def set_program_source(self, source: str) -> None:
        compiled = self.sandbox.compile_source(source)
        self._source = source
        self._namespace = compiled.namespace if compiled.success else None
        self._compile_error = compiled.errors[0] if compiled.errors else None

    def set_compiled_program(
        self,
        source: str,
        namespace: Optional[Dict[str, Any]],
        compile_error: Optional[SandboxError] = None,
    ) -> None:
        self._source = source
        self._namespace = namespace
        self._compile_error = compile_error

    def predict_next_state(
        self,
        state: Any,
        action: str,
    ) -> str:
        result = self.predict(state=state, action=action)
        if not result.success or result.predicted_state_json is None:
            error_msg = result.error.message if result.error else "Unknown prediction error."
            raise RuntimeError(error_msg)
        return result.predicted_state_json

    def predict(
        self,
        state: Any,
        action: str,
    ) -> ProgramPredictionResult:
        if self._namespace is None:
            compile_error = self._compile_error or SandboxError(
                phase="compile",
                message="Program is not compiled.",
            )
            return ProgramPredictionResult(
                success=False,
                predicted_state_obj=None,
                predicted_state_json=None,
                predicted_state_key=None,
                error=compile_error,
            )

        try:
            if isinstance(state, dict):
                state_obj = _clone_plain_structure(state)
            else:
                state_obj = parse_state_json(str(state))
        except ValueError as e:
            return ProgramPredictionResult(
                success=False,
                predicted_state_obj=None,
                predicted_state_json=None,
                predicted_state_key=None,
                error=SandboxError(phase="decode", message=str(e)),
            )

        executed = self.sandbox.execute_predict(
            namespace=self._namespace,
            state=state_obj,
            action=action,
        )
        if executed.success and executed.output_state is not None:
            predicted_state = _clone_plain_structure(executed.output_state)
            predicted_state_json = executed.output_state_json
            if predicted_state_json is None:
                predicted_state_json = dump_state_json(predicted_state)
            return ProgramPredictionResult(
                success=True,
                predicted_state_obj=predicted_state,
                predicted_state_json=predicted_state_json,
                error=None,
            )
        return ProgramPredictionResult(
            success=False,
            predicted_state_obj=None,
            predicted_state_json=None,
            predicted_state_key=None,
            error=executed.errors[0] if executed.errors else None,
        )
