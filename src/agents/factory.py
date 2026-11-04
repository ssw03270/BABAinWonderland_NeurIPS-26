"""Explorer factory and YAML config helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

from .base_agent import BaseExplorer
from src.program_model import ProgramEvaluator, SandboxConfig

if TYPE_CHECKING:
    from src.environments import BabaWrapper


DEFAULT_AGENT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_AGENT_CONFIG_PATH = DEFAULT_AGENT_CONFIG_DIR / "graph_contrastive_agent.yaml"


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"base_config", "extends"}:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_agent_config(
    config_path: Path | str,
    *,
    _seen: Optional[set[Path]] = None,
) -> Dict[str, Any]:
    import yaml

    path = Path(config_path).resolve()
    seen = set(_seen or set())
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Cyclic agent config inheritance detected: {chain}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid agent config (expected mapping): {path}")

    base_value = loaded.get("base_config", loaded.get("extends"))
    if not base_value:
        return loaded
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"`base_config` must be a non-empty string in {path}")

    base_path = Path(base_value.strip())
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    base_payload = load_agent_config(base_path, _seen=seen)
    return _deep_merge_dicts(base_payload, loaded)


def parse_agent_spec(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    block = config.get("agent", config)
    if not isinstance(block, dict):
        raise ValueError("Agent config must be a mapping.")

    raw_name = block.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("`agent.name` must be a non-empty string.")
    name = raw_name.strip().lower()

    params = block.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("`agent.params` must be a mapping.")

    return name, dict(params)


def build_explorer(
    env: "BabaWrapper",
    seed: int,
    agent_name: str,
    agent_params: Optional[Dict[str, Any]] = None,
    sandbox_config: Optional[SandboxConfig] = None,
    evaluator: Optional[ProgramEvaluator] = None,
) -> BaseExplorer:
    name = agent_name.strip().lower()
    explorer_cls: Type[BaseExplorer]

    if name == "bfs":
        from .graph_search_agent import BFSExplorer

        explorer_cls = BFSExplorer
    elif name == "graph_contrastive":
        from .graph_contrastive_agent import GraphContrastiveAgent

        explorer_cls = GraphContrastiveAgent
    else:
        supported = [
            "bfs",
            "graph_contrastive",
        ]
        raise ValueError(f"Unsupported agent `{agent_name}`. Supported: {supported}")

    params = explorer_cls.normalize_init_params(agent_params)
    params["seed"] = int(seed)
    if evaluator is not None:
        params.setdefault("evaluator", evaluator)
    if sandbox_config is not None:
        params.setdefault("sandbox_config", sandbox_config)
    return explorer_cls(env=env, **params)
