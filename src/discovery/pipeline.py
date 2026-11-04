"""Discovery pipeline factory (program-only)."""

from .program_pipeline import ProgramDiscoveryPipeline


def build_discovery_pipeline(**kwargs) -> ProgramDiscoveryPipeline:
    """Build program-based discovery pipeline."""
    return ProgramDiscoveryPipeline(**kwargs)


__all__ = ["build_discovery_pipeline", "ProgramDiscoveryPipeline"]
