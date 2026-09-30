"""
Base Explorer Interface

Base interface for exploration agents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.data.state_store import StateStore
from src.data.transition_buffer import Transition


class BaseExplorer(ABC):
    """Base class for exploration agents."""

    collection_topology = "generic"
    transition_batch_scope = "none"

    @classmethod
    def normalize_init_params(
        cls,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Common hook for normalizing agent init params.

        Explorer-specific init parameter names are preserved as-is.
        """
        return dict(params or {})

    @abstractmethod
    def collect(
        self,
        max_transitions: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_on_unknown_transition: bool = False,
    ) -> List[Transition]:
        """
        Explore the environment and collect transition data.

        Args:
            max_transitions: Maximum number of transitions to collect for the full collect call.
                If None, collect until the explorer-defined natural boundary.
                Also used as the collect-to-verify transition budget.
            progress_callback: Optional collection progress callback.
            stop_on_unknown_transition: Stop collection immediately when the current accepted
                program cannot explain a transition, so that transition can be patched first.

        Returns:
            Collected Transition list.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state."""
        pass

    def set_program_context(self, context: Optional[Dict[str, Any]]) -> None:
        """
        Hook for passing structured program context, including the latest accepted program
        versions and the current version. The default implementation is a no-op.
        """
        _ = context

    def set_state_store(self, state_store: Any) -> None:
        """
        Hook for passing the state store shared in discovery scope.
        The default implementation is a no-op.
        """
        _ = state_store

    def _require_state_store(self, state_store: Any, *, owner: str) -> StateStore:
        if not isinstance(state_store, StateStore):
            raise TypeError(f"{owner}.set_state_store expects a StateStore instance.")
        return state_store

    def _bind_env_state_store(self, state_store: StateStore) -> None:
        bind_env_store = getattr(getattr(self, "env", None), "set_state_store", None)
        if callable(bind_env_store):
            bind_env_store(state_store)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return a log dictionary for the agent's current internal state and learning statistics.
        The default implementation returns an empty dict.
        """
        return {}

    def set_collection_feedback(self, summary: Optional[Dict[str, Any]]) -> None:
        """
        Pass canonical-merge added/dataset/pending summaries to the explorer.
        The default implementation is a no-op.
        """
        _ = summary

    def record_iteration_metric_snapshot(
        self,
        *,
        output_dir: str | Path,
        iteration_summary: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Save a discovery-iteration metric snapshot.
        The default implementation is a no-op.
        """
        _ = output_dir
        _ = iteration_summary
        return None

    def commit_explained_transition(self, transition: Transition) -> Optional[Dict[str, Any]]:
        """
        Reflect a transition explained by the current accepted program in the explorer's
        internal learning state. The default implementation is a no-op.
        """
        _ = transition
        return None

    def commit_explained_transitions(
        self,
        transitions: List[Transition],
        *,
        assessments: Mapping[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Reflect a batch of transitions explained by the current accepted program in the
        explorer's internal learning state. The default implementation is a no-op.
        """
        _ = transitions
        _ = assessments
        return None

    def finalize_explained_transition_batch(
        self,
        transitions: List[Transition],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Called after the full transition batch explained during verify has been reflected
        in the explorer. The default implementation is a no-op.
        """
        _ = transitions
        _ = progress_callback
        return None

    def consume_current_source_transition_assessments(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the transition assessment snapshot computed during collect for the current source.
        The default implementation returns an empty dict.
        """
        return {}

    def refresh_current_source_transition_assessments(
        self,
        transitions: Optional[List[Transition]] = None,
    ) -> None:
        """
        Recompute transition assessment for the current accepted source and update internal
        caches. The default implementation is a no-op.
        """
        _ = transitions

    def save_training_artifacts(
        self,
        *,
        output_dir: str | Path,
        current_version_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Save explorer internal learning state as artifacts under output_dir.
        The default implementation is a no-op.
        """
        _ = output_dir
        _ = current_version_id
        return None

    def snapshot_training_artifacts_for_version(
        self,
        *,
        output_dir: str | Path,
        current_version_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Preserve the current latest learning artifacts as a version-pinned snapshot.
        The default implementation is a no-op.
        """
        _ = output_dir
        _ = current_version_id
        return None
