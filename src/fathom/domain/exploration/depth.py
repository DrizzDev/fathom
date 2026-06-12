"""
Depth-floor policy for the DFS exploration strategy.
"""

from __future__ import annotations

from fathom.domain.exploration.config import DepthFloorConfig


class DepthFloorPolicy:
    """
    Forces longer flows by vetoing premature exhaustion on shallow DFS paths.
    """

    def __init__(self, *, config: DepthFloorConfig) -> None:
        self.__minimum = config.minimum

    @property
    def minimum(self) -> int:
        """
        Shortest DFS path length permitted to honour content exhaustion.
        """

        return self.__minimum

    def should_veto(self, *, depth: int, retries: int) -> bool:
        """
        Whether a first-time exhaustion on a shallow screen should be vetoed.
        """

        return depth < self.__minimum and retries == 0

    def is_active(self, *, depth: int, retries: int) -> bool:
        """
        Whether the depth-floor directive should be injected into context.
        """

        return depth < self.__minimum and retries > 0
