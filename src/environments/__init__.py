from .state_serializer import StateSerializer

__all__ = ["BabaWrapper", "StateSerializer"]


def __getattr__(name: str):
    if name == "BabaWrapper":
        from .baba_wrapper import BabaWrapper

        return BabaWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
