"""
Program World Model Discovery Pipeline Execution Script

Usage:
    python scripts/run_discovery.py --config configs/experiment_config_online.yaml
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import math
import re
from pathlib import Path
import shutil
import sys
from typing import Optional

import yaml

# Add the project root to the Python path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm import (
    build_predictor_from_config,
    load_llm_config,
)
from src.agents import (
    DEFAULT_AGENT_CONFIG_PATH,
    ManualTransitionExplorer,
    build_explorer,
    load_agent_config,
    parse_agent_spec,
)
from src.discovery import build_discovery_pipeline
from src.program_model import ProgramEvaluator, ProgramPatcher, SandboxConfig
from src.web.discovery_registry import DiscoveryRunPublisher


REMOVED_DATA_COLLECTION_KEYS = frozenset(
    {
        "max_transitions_per_episode",
        "world_transition_batch_limit",
    }
)


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in {"base_config", "extends"}:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, *, _seen: Optional[set[Path]] = None) -> dict:
    """Load a config file. Optional `base_config`/`extends` overlays are supported."""
    path = Path(config_path).resolve()
    seen = set(_seen or set())
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"Cyclic config inheritance detected: {chain}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config (expected mapping): {path}")

    base_value = payload.get("base_config", payload.get("extends"))
    if not base_value:
        return payload
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"`base_config` must be a non-empty string in {path}")

    base_path = Path(base_value.strip())
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    base_payload = load_config(base_path, _seen=seen)
    return _deep_merge_dicts(base_payload, payload)


def resolve_path(path_or_str: str | Path) -> Path:
    path = Path(path_or_str)
    if path.is_absolute():
        return path
    return project_root / path


def build_run_output_dir(exp_config: dict) -> Path:
    """Create a unique output directory for this run."""
    base_output_dir = Path(exp_config["experiment"]["output_dir"])
    if not base_output_dir.is_absolute():
        base_output_dir = project_root / base_output_dir

    exp_name = exp_config["experiment"].get("name", "run")
    safe_exp_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(exp_name)).strip("_") or "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_output_dir = base_output_dir / f"{safe_exp_name}_{timestamp}"
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = base_output_dir / f"{safe_exp_name}_{timestamp}_{suffix:02d}"
        suffix += 1

    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def save_runtime_config_snapshots(
    *,
    run_output_dir: Path,
    experiment_config_path: Path,
    llm_config_path: Path,
    env_config_path: Path,
    agent_config_path: Path | None,
    experiment_config_payload: dict | None = None,
    llm_config_payload: dict | None = None,
    env_config_payload: dict | None = None,
    agent_config_payload: dict | None = None,
) -> None:
    snapshot_dir = run_output_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(experiment_config_path, snapshot_dir / "experiment_config.yaml")
    shutil.copy2(llm_config_path, snapshot_dir / "llm_config.yaml")
    shutil.copy2(env_config_path, snapshot_dir / "env_config.yaml")
    if agent_config_path is not None and agent_config_path.exists():
        shutil.copy2(agent_config_path, snapshot_dir / "agent_config.yaml")
    resolved_payloads = {
        "experiment_config.resolved.yaml": experiment_config_payload,
        "llm_config.resolved.yaml": llm_config_payload,
        "env_config.resolved.yaml": env_config_payload,
        "agent_config.resolved.yaml": agent_config_payload,
    }
    for filename, payload in resolved_payloads.items():
        if isinstance(payload, dict):
            (snapshot_dir / filename).write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )


def resolve_agent_config_path(exp_config: dict, cli_agent_config: str | None) -> Path:
    if cli_agent_config:
        return resolve_path(cli_agent_config)

    agent_cfg = exp_config.get("agent", {})
    if isinstance(agent_cfg, dict):
        config_path = agent_cfg.get("config_path")
        if isinstance(config_path, str) and config_path.strip():
            return resolve_path(config_path.strip())

    return DEFAULT_AGENT_CONFIG_PATH


def resolve_referenced_config_path(
    exp_config: dict,
    *,
    cli_config_path: str | None,
    section_name: str,
    default_path: str,
) -> Path:
    if cli_config_path:
        return resolve_path(cli_config_path)

    section = exp_config.get(section_name, {})
    if isinstance(section, dict):
        config_path = section.get("config_path")
        if isinstance(config_path, str) and config_path.strip():
            return resolve_path(config_path.strip())

    return resolve_path(default_path)


def resolve_agent_param_paths(agent_params: dict) -> dict:
    resolved = dict(agent_params)
    raw_checkpoint_path = resolved.get("contrastive_checkpoint_path")
    if isinstance(raw_checkpoint_path, str) and raw_checkpoint_path.strip():
        resolved["contrastive_checkpoint_path"] = str(
            resolve_path(raw_checkpoint_path.strip())
        )
    return resolved


def sanitize_env_id(env_id: str) -> str:
    return env_id.replace("/", "__").replace("#", "_").replace("-", "_")


def resolve_manual_transition_dir(exp_config: dict, env_config: dict) -> Path:
    data_collection_cfg = exp_config.get("data_collection", {})
    if isinstance(data_collection_cfg, dict):
        configured_dir = data_collection_cfg.get("manual_transition_dir")
        if isinstance(configured_dir, str) and configured_dir.strip():
            return resolve_path(configured_dir.strip())

    env_name = str(env_config["environment"]["name"])
    return project_root / "manual_env_play" / sanitize_env_id(env_name)


def resolve_prompt_additional_instructions(exp_config: dict) -> str | None:
    prompting_cfg = exp_config.get("prompting", {})
    if not isinstance(prompting_cfg, dict):
        return None
    raw_text = prompting_cfg.get("additional_instructions")
    if not isinstance(raw_text, str):
        return None
    normalized = raw_text.strip()
    return normalized or None


def resolve_program_patch_format(exp_config: dict) -> str:
    program_cfg = exp_config.get("python_program", {})
    configured = "line"
    if isinstance(program_cfg, dict):
        configured = program_cfg.get("patch_format", "line")
    return ProgramPatcher.normalize_patch_format(configured)


def resolve_progress_bar_refresh_interval_sec(exp_config: dict) -> float:
    program_cfg = exp_config.get("python_program", {})
    raw_value = 5.0
    if isinstance(program_cfg, dict):
        raw_value = program_cfg.get("progress_bar_refresh_interval_sec", raw_value)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "python_program.progress_bar_refresh_interval_sec must be a number >= 0."
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "python_program.progress_bar_refresh_interval_sec must be a number >= 0."
        )
    return value


def build_program_sandbox_config(exp_config: dict) -> SandboxConfig:
    program_cfg = exp_config.get("python_program", {})
    sandbox_cfg = program_cfg.get("sandbox", {}) if isinstance(program_cfg, dict) else {}
    if not isinstance(sandbox_cfg, dict):
        sandbox_cfg = {}
    return SandboxConfig(
        max_code_chars=sandbox_cfg.get("max_code_chars"),
        max_ast_nodes=sandbox_cfg.get("max_ast_nodes"),
        line_budget=sandbox_cfg.get("line_budget"),
        timeout_ms=sandbox_cfg.get("timeout_ms"),
    )


def resolve_shuffle_collected_transitions(exp_config: dict) -> bool:
    agent_cfg = exp_config.get("agent", {})
    if isinstance(agent_cfg, dict) and "shuffle_collected_transitions" in agent_cfg:
        return bool(agent_cfg.get("shuffle_collected_transitions"))

    data_collection_cfg = exp_config.get("data_collection", {})
    if isinstance(data_collection_cfg, dict) and "shuffle_collected_transitions" in data_collection_cfg:
        return bool(data_collection_cfg.get("shuffle_collected_transitions"))

    return True


def resolve_dashboard_history_limit(exp_config: dict) -> int:
    data_collection_cfg = exp_config.get("data_collection", {})
    visualization_cfg = (
        data_collection_cfg.get("exploration_visualization", {})
        if isinstance(data_collection_cfg, dict)
        else {}
    )
    if not isinstance(visualization_cfg, dict):
        visualization_cfg = {}
    return max(
        16,
        int(visualization_cfg.get("dashboard_history_limit", 240)),
    )


def resolve_exploration_visualization_params(
    exp_config: dict,
    *,
    agent_name: str,
) -> dict:
    data_collection_cfg = exp_config.get("data_collection", {})
    visualization_cfg = (
        data_collection_cfg.get("exploration_visualization", {})
        if isinstance(data_collection_cfg, dict)
        else {}
    )
    if not isinstance(visualization_cfg, dict):
        visualization_cfg = {}

    params = {
        "dashboard_history_limit": resolve_dashboard_history_limit(exp_config),
    }
    if str(agent_name) == "graph_contrastive":
        params["transition_projection_tsne_max_points"] = max(
            64,
            int(visualization_cfg.get("transition_projection_tsne_max_points", 1200)),
        )
        params["transition_projection_tsne_min_points_per_class"] = max(
            1,
            int(
                visualization_cfg.get(
                    "transition_projection_tsne_min_points_per_class",
                    10,
                )
            ),
        )
        params["transition_projection_tsne_iters"] = max(
            1,
            int(visualization_cfg.get("transition_projection_tsne_iters", 450)),
        )
    return params


def resolve_analysis_archive_params(exp_config: dict) -> dict:
    archive_cfg = exp_config.get("analysis_archive", {})
    if archive_cfg is None:
        archive_cfg = {}
    if not isinstance(archive_cfg, dict):
        raise ValueError("analysis_archive must be a mapping if provided.")

    params = {
        "analysis_archive_enabled": bool(archive_cfg.get("enabled", False)),
        "analysis_archive_copy_interval_sec": max(
            0.1,
            float(archive_cfg.get("copy_interval_sec", 5.0)),
        ),
    }
    raw_local_dir = archive_cfg.get("local_tmp_dir")
    if isinstance(raw_local_dir, str) and raw_local_dir.strip():
        params["analysis_archive_local_dir"] = str(resolve_path(raw_local_dir.strip()))
    return params


def resolve_dashboard_headless(
    exp_config: dict,
    *,
    cli_headless: bool | None,
) -> bool:
    if cli_headless is not None:
        return bool(cli_headless)

    dashboard_cfg = exp_config.get("dashboard", {})
    if dashboard_cfg is None:
        return False
    if not isinstance(dashboard_cfg, dict):
        raise ValueError("dashboard must be a mapping if provided.")
    return bool(dashboard_cfg.get("headless", False))


def validate_ablation_config(exp_config: dict) -> None:
    ablation_cfg = exp_config.get("ablation", {})
    if not isinstance(ablation_cfg, dict):
        return
    if not bool(ablation_cfg.get("requires_implementation", False)):
        return
    ablation_group = str(ablation_cfg.get("group", "") or "").strip()
    ablation_name = str(ablation_cfg.get("name", "") or "").strip()
    label = "/".join(part for part in (ablation_group, ablation_name) if part)
    detail = str(ablation_cfg.get("implementation_note", "") or "").strip()
    message = (
        f"Ablation `{label or 'unknown'}` is marked as requiring code implementation."
    )
    if detail:
        message = f"{message} {detail}"
    raise ValueError(message)

_MISSING_LEARNING_STARTS = object()


def apply_auto_learning_starts(
    agent_params: dict,
    *,
    collect_transitions_per_verify: int,
) -> tuple[dict, str | None]:
    params = dict(agent_params)
    configured_learning_starts = params.get("learning_starts", _MISSING_LEARNING_STARTS)
    if isinstance(configured_learning_starts, str):
        normalized = configured_learning_starts.strip().lower()
        if normalized in {"none", "null"}:
            configured_learning_starts = None

    if configured_learning_starts is _MISSING_LEARNING_STARTS:
        return params, None
    if configured_learning_starts is not None:
        return params, None
    resolved_learning_starts = max(1, int(collect_transitions_per_verify))
    params["learning_starts"] = int(resolved_learning_starts)
    return (
        params,
        "Explorer learning_starts: auto "
        f"(max_collect_transitions_per_verify = {int(resolved_learning_starts)})",
    )


def _resolve_optional_positive_int(
    exp_config: dict,
    *,
    key: str,
) -> int | None:
    data_collection_cfg = exp_config.get("data_collection", {})
    configured_budget = None
    if isinstance(data_collection_cfg, dict):
        configured_budget = data_collection_cfg.get(key)

    if configured_budget is None:
        return None
    try:
        budget = int(configured_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"data_collection.{key} must be an integer > 0."
        ) from exc
    if budget <= 0:
        raise ValueError(
            f"data_collection.{key} must be an integer > 0."
        )
    return budget


def resolve_max_collect_transitions_per_verify(exp_config: dict) -> int:
    budget = _resolve_optional_positive_int(
        exp_config,
        key="max_collect_transitions_per_verify",
    )
    if budget is None:
        raise ValueError(
            "data_collection.max_collect_transitions_per_verify must be set to an integer > 0."
        )
    return int(budget)


def reject_removed_data_collection_keys(exp_config: dict) -> None:
    data_collection_cfg = exp_config.get("data_collection", {})
    if not isinstance(data_collection_cfg, dict):
        return
    removed = sorted(
        key for key in REMOVED_DATA_COLLECTION_KEYS if key in data_collection_cfg
    )
    if removed:
        raise ValueError(
            "Removed data_collection keys are still configured: "
            + ", ".join(removed)
            + ". Remove these legacy fields and use the current discovery loop "
            "contract."
        )


def resolve_total_llm_call_limit(exp_config: dict, llm_config: dict) -> int | None:
    configured_limit = None
    experiment_cfg = exp_config.get("experiment", {})
    if isinstance(experiment_cfg, dict):
        configured_limit = experiment_cfg.get("max_total_llm_calls")

    if configured_limit is None:
        inference_cfg = llm_config.get("inference", {})
        if isinstance(inference_cfg, dict):
            configured_limit = inference_cfg.get("max_total_llm_calls")

    if configured_limit is None:
        return None
    try:
        limit = int(configured_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "experiment.max_total_llm_calls must be an integer or null."
        ) from exc
    return limit if limit > 0 else None


def _format_dashboard_thinking_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_dashboard_model_name(value: object) -> str | None:
    text = _format_dashboard_thinking_value(value)
    if text is None:
        return None
    if text.startswith("models/"):
        return text[len("models/") :]
    return text


def resolve_dashboard_model_label(llm_config: dict) -> str:
    inference_cfg = llm_config.get("inference", {})
    if not isinstance(inference_cfg, dict):
        return "n/a"

    provider = str(inference_cfg.get("provider", "") or "").strip().lower()
    if provider == "gemini":
        return _normalize_dashboard_model_name(inference_cfg.get("gemini_model")) or "n/a"
    if provider == "openai":
        return _normalize_dashboard_model_name(inference_cfg.get("openai_model")) or "n/a"
    return "n/a"


def resolve_dashboard_thinking_label(llm_config: dict) -> str:
    inference_cfg = llm_config.get("inference", {})
    if not isinstance(inference_cfg, dict):
        return "n/a"

    provider = str(inference_cfg.get("provider", "") or "").strip().lower()
    if provider == "gemini":
        level = _format_dashboard_thinking_value(inference_cfg.get("gemini_thinking_level"))
        if level is not None:
            return level
        budget = _format_dashboard_thinking_value(inference_cfg.get("gemini_thinking_budget"))
        return f"budget={budget}" if budget is not None else "n/a"
    if provider == "openai":
        return _format_dashboard_thinking_value(inference_cfg.get("openai_reasoning_effort")) or "n/a"
    return _format_dashboard_thinking_value(inference_cfg.get("reasoning_effort")) or "n/a"


def build_exploration_dashboard_context(run_output_dir: Path, llm_config: dict) -> list[str]:
    inference_cfg = llm_config.get("inference", {})
    provider = "unknown"
    if isinstance(inference_cfg, dict):
        raw_provider = str(inference_cfg.get("provider", "") or "").strip()
        if raw_provider:
            provider = raw_provider
    model_label = resolve_dashboard_model_label(llm_config)
    thinking_label = resolve_dashboard_thinking_label(llm_config)
    exp_name = run_output_dir.name.strip() or str(run_output_dir)
    return [
        f"exp={exp_name}",
        f"llm={provider} | model={model_label}",
        f"thinking={thinking_label}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Program Discovery Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to experiment config. Defaults to configs/experiment_config_online.yaml, "
            "or the source run config snapshot when --resume-discovery-run is used."
        )
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        help=(
            "Path to LLM config. If omitted, uses `llm.config_path` from the "
            "experiment config, then configs/llm_config.yaml."
        )
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help=(
            "Path to environment config. If omitted, uses `env.config_path` from "
            "the experiment config, then configs/env_config.yaml."
        )
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional hard cap on total iterations (default: no cap, epoch-budget mode)"
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        default=None,
        help="Path to agent YAML config (default: from experiment config or graph contrastive agent config)",
    )
    parser.add_argument(
        "--resume-program-run",
        type=str,
        default=None,
        help=(
            "Existing run output directory to bootstrap program-only resume from. "
            "Transition collection and contrastive state still start from scratch."
        ),
    )
    parser.add_argument(
        "--resume-program-version",
        type=str,
        default=None,
        help=(
            "Accepted program version id to resume from (for example: v012). "
            "Copies program artifacts only up to this version into the new run."
        ),
    )
    parser.add_argument(
        "--resume-discovery-run",
        type=str,
        default=None,
        help=(
            "Existing run output directory to resume discovery from an iteration "
            "boundary. Restores state_store, canonical/protected transitions, "
            "contrastive replay/frontier graph state, and the selected checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-discovery-rollback-iterations",
        type=int,
        default=1,
        help=(
            "How many latest complete archive iterations to roll back before "
            "resuming discovery (default: 1)."
        ),
    )
    parser.add_argument(
        "--resume-discovery-through-iteration",
        type=int,
        default=None,
        help=(
            "Explicit completed archive iteration to resume through. Overrides "
            "--resume-discovery-rollback-iterations."
        ),
    )
    parser.add_argument(
        "--resume-contrastive-checkpoint",
        type=str,
        default="latest",
        help=(
            "Contrastive checkpoint policy for --resume-discovery-run: latest, "
            "none, or a checkpoint path relative to the source run."
        ),
    )
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help=(
            "Disable the discovery dashboard/web GUI entirely and skip dashboard-related "
            "bookkeeping inside the run loop."
        ),
    )
    headless_group.add_argument(
        "--dashboard",
        dest="headless",
        action="store_false",
        help=(
            "Force-enable the discovery dashboard/web GUI even if the experiment config "
            "requests headless mode."
        ),
    )
    parser.set_defaults(headless=None)

    args = parser.parse_args()
    if bool(args.resume_program_run) != bool(args.resume_program_version):
        raise ValueError(
            "--resume-program-run and --resume-program-version must be provided together."
        )
    if args.resume_discovery_run and args.resume_program_run:
        raise ValueError(
            "--resume-discovery-run cannot be combined with --resume-program-run. "
            "Discovery resume chooses the program version from the archive boundary."
        )

    # Load configs.
    print("Loading configurations...")
    resume_discovery_source_dir = (
        resolve_path(args.resume_discovery_run)
        if args.resume_discovery_run
        else None
    )

    def _resolve_resume_snapshot_file(
        filenames: tuple[str, ...],
        *,
        option_name: str,
    ) -> Path:
        if resume_discovery_source_dir is None:
            raise RuntimeError("resume_discovery_source_dir is not initialized.")
        snapshot_dir = resume_discovery_source_dir / "config_snapshot"
        for filename in filenames:
            candidate = snapshot_dir / filename
            if candidate.exists():
                return candidate
        names = ", ".join(filenames)
        raise FileNotFoundError(
            f"--resume-discovery-run needs {option_name} or one of [{names}] "
            f"under {snapshot_dir}."
        )

    config_path = (
        _resolve_resume_snapshot_file(
            ("experiment_config.resolved.yaml", "experiment_config.yaml"),
            option_name="--config",
        )
        if resume_discovery_source_dir is not None and args.config is None
        else resolve_path(args.config or "configs/experiment_config_online.yaml")
    )
    exp_config = load_config(config_path)
    llm_config_path = (
        _resolve_resume_snapshot_file(
            ("llm_config.resolved.yaml", "llm_config.yaml"),
            option_name="--llm-config",
        )
        if resume_discovery_source_dir is not None and args.llm_config is None
        else resolve_referenced_config_path(
            exp_config,
            cli_config_path=args.llm_config,
            section_name="llm",
            default_path="configs/llm_config.yaml",
        )
    )
    env_config_path = (
        _resolve_resume_snapshot_file(
            ("env_config.resolved.yaml", "env_config.yaml"),
            option_name="--env-config",
        )
        if resume_discovery_source_dir is not None and args.env_config is None
        else resolve_referenced_config_path(
            exp_config,
            cli_config_path=args.env_config,
            section_name="env",
            default_path="configs/env_config.yaml",
        )
    )
    llm_config = load_llm_config(llm_config_path)
    env_config = load_config(env_config_path)
    headless = resolve_dashboard_headless(
        exp_config,
        cli_headless=args.headless,
    )
    validate_ablation_config(exp_config)

    run_output_dir = build_run_output_dir(exp_config)
    resume_bootstrap = None
    discovery_resume_state = None
    data_collection_cfg = exp_config.get("data_collection", {})
    use_agent_collector = bool(
        data_collection_cfg.get("use_agent_collector", True)
    )
    agent_config_path = (
        resolve_path(args.agent_config)
        if use_agent_collector and args.agent_config
        else _resolve_resume_snapshot_file(
            ("agent_config.resolved.yaml", "agent_config.yaml"),
            option_name="--agent-config",
        )
        if use_agent_collector and resume_discovery_source_dir is not None
        else resolve_agent_config_path(exp_config, None)
        if use_agent_collector
        else None
    )
    dashboard_publisher: DiscoveryRunPublisher | None = None
    run_context = nullcontext()
    if not headless:
        dashboard_publisher = DiscoveryRunPublisher(
            run_output_dir=run_output_dir,
            config_path=str(config_path),
            llm_config_path=str(llm_config_path),
            llm_config_payload=llm_config,
            env_config_path=str(env_config_path),
            agent_config_path=str(agent_config_path) if agent_config_path is not None else None,
            max_iterations=args.max_iterations,
            host="127.0.0.1",
            port=8000,
            open_browser=True,
        )
        dashboard_url = dashboard_publisher.ensure_dashboard_server()
        run_context = dashboard_publisher.tee_streams()

    with run_context:
        if headless:
            print("Discovery dashboard: disabled (headless)")
        else:
            print(f"Discovery dashboard: {dashboard_url}")
            print(f"Attached run id: {dashboard_publisher.run_id}")
        print(f"Run output directory: {run_output_dir}")
        print(f"Experiment config: {config_path}")
        print(f"LLM config: {llm_config_path}")
        print(f"Environment config: {env_config_path}")
        ablation_cfg = exp_config.get("ablation", {})
        if isinstance(ablation_cfg, dict) and ablation_cfg:
            ablation_group = str(ablation_cfg.get("group", "") or "").strip()
            ablation_name = str(ablation_cfg.get("name", "") or "").strip()
            ablation_label = "/".join(
                part for part in (ablation_group, ablation_name) if part
            )
            if ablation_label:
                print(f"Ablation: {ablation_label}")
        if args.resume_discovery_run:
            from src.discovery.iteration_boundary_resume import (
                bootstrap_iteration_boundary_resume_artifacts,
            )

            print("Discovery resume bootstrap: starting")
            discovery_resume_state = bootstrap_iteration_boundary_resume_artifacts(
                source_run_dir=resume_discovery_source_dir,
                target_run_dir=run_output_dir,
                rollback_iterations=args.resume_discovery_rollback_iterations,
                through_iteration=args.resume_discovery_through_iteration,
                checkpoint_policy=args.resume_contrastive_checkpoint,
                dashboard_history_limit=resolve_dashboard_history_limit(exp_config),
            )
            resume_bootstrap = discovery_resume_state.program_bootstrap
            print("Discovery resume bootstrap: complete")
        elif args.resume_program_run and args.resume_program_version:
            from src.discovery.program_resume import bootstrap_program_resume_artifacts

            print("Program-only resume bootstrap: starting")
            resume_bootstrap = bootstrap_program_resume_artifacts(
                source_run_dir=resolve_path(args.resume_program_run),
                target_run_dir=run_output_dir,
                through_version_id=args.resume_program_version,
            )
            print("Program-only resume bootstrap: complete")
        if resume_bootstrap is not None:
            print(
                "Program-only resume source: "
                f"{resume_bootstrap.source_run_dir}"
            )
            print(
                "Program-only resume version: "
                f"{resume_bootstrap.through_version_id}"
            )
            print(
                "Program-only resume copied versions: "
                f"{', '.join(resume_bootstrap.copied_version_ids)}"
            )
            print(
                "Program-only resume copied patch dirs: "
                + (
                    ", ".join(resume_bootstrap.copied_patch_attempt_dirs)
                    if resume_bootstrap.copied_patch_attempt_dirs
                    else "(none)"
                )
            )
            print(
                "Program-only resume copied transition dirs: "
                + (
                    ", ".join(resume_bootstrap.copied_transition_image_dirs)
                    if resume_bootstrap.copied_transition_image_dirs
                    else "(none)"
                )
            )
        if discovery_resume_state is not None:
            print(
                "Discovery resume source: "
                f"{discovery_resume_state.source_run_dir}"
            )
            print(
                "Discovery resume boundary: "
                f"latest_iter={discovery_resume_state.latest_complete_iteration} "
                f"cutoff_iter={discovery_resume_state.cutoff_iteration} "
                f"global_step={discovery_resume_state.cutoff_global_step} "
                f"program={discovery_resume_state.cutoff_program_version_id}"
            )
            dashboard_history_points = (
                len(discovery_resume_state.dashboard_history.get("steps", ()))
                if isinstance(discovery_resume_state.dashboard_history, dict)
                else 0
            )
            print(
                "Discovery resume restored sizes: "
                f"states={discovery_resume_state.cutoff_state_count} "
                f"edges={discovery_resume_state.edge_count} "
                f"canonical={discovery_resume_state.canonical_count} "
                f"sample_store={discovery_resume_state.sample_store_count} "
                f"frontier={discovery_resume_state.frontier_size} "
                f"dashboard_history={dashboard_history_points}"
            )
            if discovery_resume_state.checkpoint_path is None:
                print("Discovery resume checkpoint: disabled")
            else:
                print(
                    "Discovery resume checkpoint: "
                    f"{discovery_resume_state.checkpoint_path} "
                    f"steps={discovery_resume_state.checkpoint_total_steps}"
                )
        if dashboard_publisher is not None:
            dashboard_publisher.mark_running()
        env = None
        explorer = None
        pipeline = None
        try:
            shuffle_collected_transitions = resolve_shuffle_collected_transitions(exp_config)
            verify_transition_budget = resolve_max_collect_transitions_per_verify(exp_config)
            reject_removed_data_collection_keys(exp_config)
            inference_cfg = llm_config["inference"]
            experiment_seed = int(exp_config.get("experiment", {}).get("seed", 42))
            serialization_cfg = env_config.get("serialization", {})
            experiment_serialization_cfg = exp_config.get("serialization", {})
            agent_config = None

            predictor = build_predictor_from_config(
                llm_config_path=llm_config_path,
                project_root=project_root,
                llm_config_payload=llm_config,
            )

            # Initialize the exploration data source.
            shared_sandbox_config = build_program_sandbox_config(exp_config)
            program_cfg = exp_config.get("python_program", {})
            shared_program_evaluator = ProgramEvaluator(
                sandbox_config=shared_sandbox_config,
                program_eval_workers=program_cfg.get("program_eval_workers", "auto"),
            )
            if use_agent_collector:
                agent_config = load_agent_config(agent_config_path)
                agent_name, agent_params = parse_agent_spec(agent_config)
                agent_params = resolve_agent_param_paths(agent_params)
                agent_params["dashboard_enabled"] = not headless
                if discovery_resume_state is not None and agent_name != "graph_contrastive":
                    raise ValueError(
                        "--resume-discovery-run currently requires the graph_contrastive explorer."
                    )
                if agent_name == "graph_contrastive":
                    agent_params.update(resolve_analysis_archive_params(exp_config))
                    if (
                        discovery_resume_state is not None
                        and discovery_resume_state.checkpoint_path is not None
                    ):
                        agent_params["contrastive_checkpoint_path"] = str(
                            discovery_resume_state.checkpoint_path
                        )
                if not headless:
                    agent_params.setdefault(
                        "dashboard_context_lines",
                        build_exploration_dashboard_context(run_output_dir, llm_config),
                    )
                    agent_params.update(
                        resolve_exploration_visualization_params(
                            exp_config,
                            agent_name=agent_name,
                        )
                    )

                print("Initializing environment...")
                default_state_format = "json"
                env_kwargs = env_config["environment"].get("kwargs", {})
                if not isinstance(env_kwargs, dict):
                    raise ValueError("environment.kwargs must be a mapping if provided.")

                from src.environments import BabaWrapper

                env = BabaWrapper(
                    env_name=env_config["environment"]["name"],
                    max_steps=env_config["environment"]["max_steps"],
                    render_mode=env_config["environment"].get("render_mode"),
                    state_format=serialization_cfg.get("format", default_state_format),
                    seed=experiment_seed,
                    env_kwargs=env_kwargs,
                    word_aliases=experiment_serialization_cfg.get("word_aliases"),
                )
                agent_params, learning_starts_message = apply_auto_learning_starts(
                    agent_params,
                    collect_transitions_per_verify=verify_transition_budget,
                )
                if agent_name in {"bfs", "graph_contrastive"}:
                    agent_params.setdefault(
                        "collect_env_factory",
                        partial(
                            BabaWrapper,
                            env_name=env_config["environment"]["name"],
                            max_steps=env_config["environment"]["max_steps"],
                            render_mode=None,
                            state_format=serialization_cfg.get("format", default_state_format),
                            seed=experiment_seed,
                            env_kwargs=env_kwargs,
                            word_aliases=experiment_serialization_cfg.get("word_aliases"),
                        ),
                    )

                explorer = build_explorer(
                    env=env,
                    seed=experiment_seed,
                    agent_name=agent_name,
                    agent_params=agent_params,
                    evaluator=shared_program_evaluator,
                )
                explorer_transition_batch_scope = str(
                    getattr(explorer, "transition_batch_scope", "none")
                ).strip().lower()
                explorer_topology = str(
                    getattr(explorer, "collection_topology", "generic")
                )
                if dashboard_publisher is not None:
                    dashboard_publisher.bind_visualizer(explorer)
                print(f"Explorer: {agent_name} ({agent_config_path})")
                if headless:
                    print("Explorer dashboard: disabled (headless)")
                else:
                    print("Explorer dashboard: web-only attach enabled")
                print(f"Explorer topology: {explorer_topology}")
                if learning_starts_message:
                    print(learning_starts_message)
            else:
                manual_transition_dir = resolve_manual_transition_dir(exp_config, env_config)
                explorer = ManualTransitionExplorer(
                    source_dir=manual_transition_dir,
                    seed=experiment_seed,
                )
                explorer_transition_batch_scope = "none"
                explorer_topology = "manual"
                print("Data source: manual transitions")
                print(f"Manual transition dir: {manual_transition_dir}")
                print(f"Manual transitions loaded: {explorer.total_transitions}")

            if explorer_transition_batch_scope == "world":
                explorer_world_transition_batch_limit = getattr(
                    explorer,
                    "world_transition_batch_limit",
                    None,
                )
                if explorer_world_transition_batch_limit is None:
                    print("World collect scheduling: fair active-map slices")
                else:
                    print(
                        "World transition batch limit: "
                        f"{int(explorer_world_transition_batch_limit)}"
                    )
            else:
                print("Transition batch limit: n/a for non-batched explorer")
            print(
                "Collect-to-verify transition budget: "
                f"{int(verify_transition_budget)}"
            )
            print(
                "Collected transition shuffle: "
                + ("enabled" if shuffle_collected_transitions else "disabled")
            )
            progress_bar_refresh_interval_sec = resolve_progress_bar_refresh_interval_sec(
                exp_config
            )
            print(
                "Progress bar refresh interval: "
                f"{progress_bar_refresh_interval_sec:.2f}s"
            )
            save_runtime_config_snapshots(
                run_output_dir=run_output_dir,
                experiment_config_path=config_path,
                llm_config_path=llm_config_path,
                env_config_path=env_config_path,
                agent_config_path=agent_config_path if use_agent_collector else None,
                experiment_config_payload=exp_config,
                llm_config_payload=llm_config,
                env_config_payload=env_config,
                agent_config_payload=agent_config if use_agent_collector else None,
            )

            # Initialize the pipeline.
            print("Initializing pipeline...")
            common_kwargs = {
                "explorer": explorer,
                "max_collect_transitions_per_verify": verify_transition_budget,
                "canonical_dataset_max_size": exp_config["data_collection"].get("canonical_dataset_max_size"),
                "shuffle_collected_transitions": shuffle_collected_transitions,
                "save_new_transition_images": exp_config.get("python_program", {}).get(
                    "save_new_transition_images", False
                ),
                "output_dir": str(run_output_dir),
            }
            max_total_llm_calls = resolve_total_llm_call_limit(exp_config, llm_config)

            ablation_cfg = exp_config.get("ablation", {})
            if not isinstance(ablation_cfg, dict):
                ablation_cfg = {}
            prompt_additional_instructions = resolve_prompt_additional_instructions(exp_config)
            pipeline = build_discovery_pipeline(
                llm_predictor=predictor,
                max_fail_count=program_cfg.get("max_fail_count", 3),
                saturation_global_step_window=program_cfg.get(
                    "saturation_global_step_window"
                ),
                patch_unexpected_error_max_attempts=inference_cfg.get(
                    "program_patch_unexpected_error_max_attempts",
                    None,
                ),
                max_total_llm_calls=max_total_llm_calls,
                regression_group_sample_k=program_cfg.get("regression_group_sample_k", 0),
                regression_witnesses_per_group=program_cfg.get(
                    "regression_witnesses_per_group",
                    1,
                ),
                dynamics_class_mode=ablation_cfg.get("dynamics_class_mode", "full"),
                regression_witness_mode=ablation_cfg.get(
                    "regression_witness_mode",
                    "class_aware",
                ),
                random_seed=experiment_seed,
                prompt_additional_instructions=prompt_additional_instructions,
                patch_format=resolve_program_patch_format(exp_config),
                progress_bar_refresh_interval_sec=progress_bar_refresh_interval_sec,
                evaluator=shared_program_evaluator,
                resume_program_only=resume_bootstrap is not None,
                **common_kwargs,
            )
            if discovery_resume_state is not None:
                restore_summary = pipeline.restore_iteration_boundary_state(
                    discovery_resume_state
                )
                print(
                    "Discovery resume attached: "
                    f"iteration={restore_summary['iteration']} "
                    f"global_step={restore_summary['global_step']} "
                    f"program={restore_summary['program_version']} "
                    f"canonical={restore_summary['canonical_count']} "
                    f"protected={restore_summary['protected_count']}"
                )
                explorer_restore_summary = restore_summary.get("explorer")
                if isinstance(explorer_restore_summary, dict):
                    print(
                        "Discovery resume explorer: "
                        f"steps={explorer_restore_summary.get('total_steps')} "
                        f"edges={explorer_restore_summary.get('edge_count')} "
                        f"frontier={explorer_restore_summary.get('frontier_size')} "
                        f"active_worlds={explorer_restore_summary.get('active_world_count')}"
                    )
            if dashboard_publisher is not None:
                dashboard_publisher.bind_final_dashboard_preparer(
                    pipeline.prepare_final_dashboard_snapshot
                )

            # Run.
            max_iter = args.max_iterations
            if max_iter is None:
                print("\nStarting discovery with epoch-budget iterations (no fixed max)...\n")
            else:
                print(f"\nStarting discovery with max {max_iter} iterations...\n")

            final_result = pipeline.run(max_iterations=max_iter)

            print(f"\nDiscovery complete!")
            lines = len(final_result.splitlines()) if isinstance(final_result, str) else 0
            print(f"Final program generated ({lines} lines)")
            print(f"Saved run output directory: {run_output_dir}")
            print(f"Experiment seed: {experiment_seed}")
            if dashboard_publisher is not None:
                print(f"Saved dashboard run id: {dashboard_publisher.run_id}")
            print(f"Used experiment config: {config_path}")
            print(f"Used LLM config: {llm_config_path}")
            print(f"Used environment config: {env_config_path}")
            if use_agent_collector and agent_config_path is not None:
                print(f"Used agent config: {agent_config_path}")
            if dashboard_publisher is not None:
                dashboard_publisher.mark_completed({"run_output_dir": str(run_output_dir)})
        except Exception as exc:
            if dashboard_publisher is not None:
                dashboard_publisher.mark_failed(exc)
            raise
        finally:
            try:
                explorer_close = getattr(explorer, "close", None)
                if callable(explorer_close):
                    explorer_close()
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    if pipeline is not None:
                        pipeline.close()


if __name__ == "__main__":
    main()
