from .base_agent import BaseExplorer
from .manual_transition_explorer import ManualTransitionExplorer
from .factory import (
    DEFAULT_AGENT_CONFIG_DIR,
    DEFAULT_AGENT_CONFIG_PATH,
    build_explorer,
    load_agent_config,
    parse_agent_spec,
)


def __getattr__(name: str):
    if name == "GraphContrastiveAgent":
        from .graph_contrastive_agent import GraphContrastiveAgent

        return GraphContrastiveAgent
    if name == "BFSExplorer":
        from .graph_search_agent import BFSExplorer

        return BFSExplorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphContrastiveAgent",
    "BFSExplorer",
    "ManualTransitionExplorer",
    "BaseExplorer",
    "DEFAULT_AGENT_CONFIG_DIR",
    "DEFAULT_AGENT_CONFIG_PATH",
    "build_explorer",
    "load_agent_config",
    "parse_agent_spec",
]
