"""
Saturation Checker

Track discovery saturation as the global-step gap since the last successful
program patch incorporation.
"""

from typing import Optional


class SaturationChecker:
    """Global-step-based saturation tracker."""

    def __init__(self, max_fail_count: int = 3):
        if int(max_fail_count) <= 0:
            raise ValueError("max_fail_count must be > 0.")
        self.max_fail_count = int(max_fail_count)
        self.fail_count = 0
        self._is_saturated = False
        self.last_seen_global_step: Optional[int] = None
        self.last_patch_global_step: Optional[int] = None

    def observe(self, *, global_step: Optional[int]) -> bool:
        """Update saturation state from the latest collection progress."""
        if global_step is None:
            return self._is_saturated

        resolved_global_step = max(0, int(global_step))
        if (
            self.last_seen_global_step is not None
            and resolved_global_step < self.last_seen_global_step
        ):
            self.last_patch_global_step = None
            self.fail_count = 0
            self._is_saturated = False

        self.last_seen_global_step = resolved_global_step

        reference_step = (
            int(self.last_patch_global_step)
            if isinstance(self.last_patch_global_step, int)
            else 0
        )
        self.fail_count = resolved_global_step - reference_step
        self._is_saturated = self.fail_count >= self.max_fail_count
        return self._is_saturated

    def mark_successful_patch(self, *, global_step: Optional[int]) -> None:
        if isinstance(global_step, int):
            resolved_step = max(0, int(global_step))
        elif isinstance(self.last_seen_global_step, int):
            resolved_step = int(self.last_seen_global_step)
        else:
            resolved_step = 0
        self.last_patch_global_step = resolved_step
        self._is_saturated = False
        if isinstance(self.last_seen_global_step, int):
            self.fail_count = int(self.last_seen_global_step) - resolved_step
        else:
            self.fail_count = 0

    @property
    def is_saturated(self) -> bool:
        return self._is_saturated

    def reset(self) -> None:
        self.fail_count = 0
        self._is_saturated = False
        self.last_seen_global_step = None
        self.last_patch_global_step = None

    def get_status(self) -> dict:
        return {
            "fail_count": self.fail_count,
            "max_fail_count": self.max_fail_count,
            "is_saturated": self._is_saturated,
            "last_seen_global_step": self.last_seen_global_step,
            "last_patch_global_step": self.last_patch_global_step,
        }
