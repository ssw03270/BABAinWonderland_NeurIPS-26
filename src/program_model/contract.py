from inspect import Signature, signature
from typing import Callable


PREDICT_FUNCTION_NAME = "predict_next_state"


BASELINE_PROGRAM_SOURCE = """def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy(v) for v in value)
    if isinstance(value, set):
        return {_deep_copy(v) for v in value}
    return value


def predict_next_state(state, action):
    if not isinstance(state, dict):
        raise ValueError("state must be a dict.")
    if not isinstance(action, str):
        raise ValueError("action must be a string.")
    return None
"""


def validate_predict_function_signature(func: Callable) -> None:
    """Contract: predict_next_state(state, action) -> dict."""
    sig: Signature = signature(func)
    params = list(sig.parameters.values())
    if len(params) != 2:
        raise ValueError("predict_next_state must accept exactly 2 parameters.")
    if params[0].name != "state" or params[1].name != "action":
        raise ValueError("predict_next_state signature must be (state, action).")
