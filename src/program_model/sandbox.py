import ast
import builtins as py_builtins
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contract import PREDICT_FUNCTION_NAME, validate_predict_function_signature
from .state_codec import dump_state_json


@dataclass
class SandboxConfig:
    max_code_chars: Optional[int] = None
    max_ast_nodes: Optional[int] = None
    line_budget: Optional[int] = None
    timeout_ms: Optional[int] = None
    static_validation_enabled: bool = False
    allow_imports: bool = False


@dataclass
class SandboxError:
    phase: str
    message: str
    exception_type: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass
class CompilationResult:
    success: bool
    namespace: Optional[Dict[str, Any]] = None
    errors: List[SandboxError] = field(default_factory=list)


@dataclass
class ExecutionResult:
    success: bool
    output_state: Optional[Dict[str, Any]] = None
    output_state_json: Optional[str] = None
    errors: List[SandboxError] = field(default_factory=list)
    elapsed_ms: float = 0.0
    executed_lines: int = 0


@dataclass
class PreparedSourceResult:
    success: bool
    code_obj: Optional[Any] = None
    errors: List[SandboxError] = field(default_factory=list)


class ProgramSandbox:
    """Compile and execute model programs in a restricted environment."""

    _BANNED_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.Lambda,
        ast.With,
        ast.Global,
        ast.Nonlocal,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )

    _ALLOWED_BUILTINS = (
        "len",
        "range",
        "min",
        "max",
        "sum",
        "abs",
        "any",
        "all",
        "sorted",
        "reversed",
        "next",
        "enumerate",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "id",
        "isinstance",
        "zip",
        "ValueError",
        "TypeError",
        "RuntimeError",
    )

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._builtins = {
            name: getattr(py_builtins, name) for name in self._ALLOWED_BUILTINS
        }
        if self.config.allow_imports:
            self._builtins["__import__"] = py_builtins.__import__
        self._prepared_source_cache: Dict[str, PreparedSourceResult] = {}

    def compile_source(self, source: str) -> CompilationResult:
        prepared = self._prepare_source(source)
        if not prepared.success or prepared.code_obj is None:
            return CompilationResult(
                success=False,
                errors=self._clone_errors(prepared.errors),
            )

        globals_dict: Dict[str, Any] = {"__builtins__": self._builtins}
        try:
            exec(prepared.code_obj, globals_dict, globals_dict)
        except Exception as e:  # noqa: BLE001
            errors = [
                SandboxError(
                    phase="compile",
                    message=str(e),
                    exception_type=type(e).__name__,
                    traceback_text=traceback.format_exc(),
                )
            ]
            return CompilationResult(success=False, errors=errors)

        predict_fn = globals_dict.get(PREDICT_FUNCTION_NAME)
        if not callable(predict_fn):
            errors = [
                SandboxError(
                    phase="contract",
                    message=f"Missing callable `{PREDICT_FUNCTION_NAME}`.",
                )
            ]
            return CompilationResult(success=False, errors=errors)

        try:
            validate_predict_function_signature(predict_fn)
        except ValueError as e:
            errors = [SandboxError(phase="contract", message=str(e))]
            return CompilationResult(success=False, errors=errors)

        return CompilationResult(success=True, namespace=globals_dict, errors=[])

    def _prepare_source(self, source: str) -> PreparedSourceResult:
        cached = self._prepared_source_cache.get(source)
        if cached is not None:
            return cached

        errors: List[SandboxError] = []

        if self.config.static_validation_enabled:
            if (
                self.config.max_code_chars is not None
                and len(source) > self.config.max_code_chars
            ):
                errors.append(
                    SandboxError(
                        phase="ast_validate",
                        message=(
                            "Source length exceeds max_code_chars: "
                            f"{len(source)} > {self.config.max_code_chars}"
                        ),
                    )
                )
                prepared = PreparedSourceResult(success=False, errors=errors)
                self._prepared_source_cache[source] = prepared
                return prepared

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            errors.append(
                SandboxError(
                    phase="parse",
                    message=str(e),
                    exception_type=type(e).__name__,
                    traceback_text=traceback.format_exc(),
                )
            )
            prepared = PreparedSourceResult(success=False, errors=errors)
            self._prepared_source_cache[source] = prepared
            return prepared

        if self.config.static_validation_enabled:
            node_count = sum(1 for _ in ast.walk(tree))
            if (
                self.config.max_ast_nodes is not None
                and node_count > self.config.max_ast_nodes
            ):
                errors.append(
                    SandboxError(
                        phase="ast_validate",
                        message=(
                            "AST node count exceeds max_ast_nodes: "
                            f"{node_count} > {self.config.max_ast_nodes}"
                        ),
                    )
                )

            for node in ast.walk(tree):
                if isinstance(node, self._BANNED_NODES):
                    errors.append(
                        SandboxError(
                            phase="ast_validate",
                            message=f"Banned AST node used: {type(node).__name__}",
                        )
                    )

        if errors:
            prepared = PreparedSourceResult(success=False, errors=errors)
            self._prepared_source_cache[source] = prepared
            return prepared

        try:
            code_obj = compile(tree, "<program_model>", "exec")
        except Exception as e:  # noqa: BLE001
            errors.append(
                SandboxError(
                    phase="compile",
                    message=str(e),
                    exception_type=type(e).__name__,
                    traceback_text=traceback.format_exc(),
                )
            )
            prepared = PreparedSourceResult(success=False, errors=errors)
            self._prepared_source_cache[source] = prepared
            return prepared

        prepared = PreparedSourceResult(success=True, code_obj=code_obj, errors=[])
        self._prepared_source_cache[source] = prepared
        return prepared

    def _clone_errors(self, errors: List[SandboxError]) -> List[SandboxError]:
        return [
            SandboxError(
                phase=error.phase,
                message=error.message,
                exception_type=error.exception_type,
                traceback_text=error.traceback_text,
            )
            for error in errors
        ]

    def execute_predict(
        self,
        namespace: Dict[str, Any],
        state: Dict[str, Any],
        action: str,
    ) -> ExecutionResult:
        predict_fn = namespace.get(PREDICT_FUNCTION_NAME)
        if not callable(predict_fn):
            return ExecutionResult(
                success=False,
                errors=[
                    SandboxError(
                        phase="contract",
                        message=f"Missing callable `{PREDICT_FUNCTION_NAME}`.",
                    )
                ],
            )

        def runner() -> Any:
            return predict_fn(state, action)

        start = time.perf_counter()
        try:
            output, line_count = self._run_with_limits(runner)
        except Exception as e:  # noqa: BLE001
            return ExecutionResult(
                success=False,
                errors=[
                    SandboxError(
                        phase="execute",
                        message=str(e),
                        exception_type=type(e).__name__,
                        traceback_text=traceback.format_exc(),
                    )
                ],
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(output, dict):
            return ExecutionResult(
                success=False,
                errors=[
                    SandboxError(
                        phase="contract",
                        message="predict_next_state must return a dictionary.",
                    )
                ],
                elapsed_ms=elapsed_ms,
                executed_lines=line_count,
            )

        try:
            output_state_json = dump_state_json(output)
        except ValueError as e:
            detail = str(e)
            prefix = "State is not JSON-serializable: "
            if detail.startswith(prefix):
                detail = detail[len(prefix) :]
            root_error = e.__cause__ if e.__cause__ is not None else e
            return ExecutionResult(
                success=False,
                errors=[
                    SandboxError(
                        phase="contract",
                        message=f"Returned state is not JSON-serializable: {detail}",
                        exception_type=type(root_error).__name__,
                    )
                ],
                elapsed_ms=elapsed_ms,
                executed_lines=line_count,
            )

        return ExecutionResult(
            success=True,
            output_state=output,
            output_state_json=output_state_json,
            elapsed_ms=elapsed_ms,
            executed_lines=line_count,
        )

    def smoke_test(
        self,
        source: str,
        state: Optional[Dict[str, Any]] = None,
        action: str = "noop",
    ) -> Tuple[bool, List[SandboxError]]:
        sample_state = state or {
            "grid_size": [0, 0],
            "objects": [],
            "step": {"terminated": False},
        }
        compiled = self.compile_source(source)
        if not compiled.success or compiled.namespace is None:
            return False, compiled.errors
        executed = self.execute_predict(compiled.namespace, sample_state, action)
        return executed.success, executed.errors

    def _run_with_limits(self, fn) -> Tuple[Any, int]:
        timeout_sec = (
            None
            if self.config.timeout_ms is None
            else self.config.timeout_ms / 1000.0
        )
        line_budget = self.config.line_budget
        if timeout_sec is None and line_budget is None:
            return fn(), 0
        line_count = 0
        start = time.perf_counter()
        previous_trace = sys.gettrace()

        def tracer(frame, event, arg):  # noqa: ANN001, ANN202
            _ = arg
            nonlocal line_count
            if event == "line":
                line_count += 1
                if line_budget is not None and line_count > line_budget:
                    raise RuntimeError(
                        f"Line budget exceeded: {line_count} > {line_budget}"
                    )
                elapsed = time.perf_counter() - start
                if timeout_sec is not None and elapsed > timeout_sec:
                    raise TimeoutError(
                        f"Execution timeout exceeded: {elapsed:.6f}s > {timeout_sec:.6f}s"
                    )
            return tracer

        sys.settrace(tracer)
        try:
            result = fn()
        finally:
            sys.settrace(previous_trace)

        elapsed = time.perf_counter() - start
        if timeout_sec is not None and elapsed > timeout_sec:
            raise TimeoutError(
                f"Execution timeout exceeded: {elapsed:.6f}s > {timeout_sec:.6f}s"
            )
        return result, line_count
