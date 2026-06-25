"""
Scroll-probe policy for the DFS exploration strategy.
"""

from __future__ import annotations

from fathom.core.exploration.config import ScrollProbeConfig


class ScrollProbePolicy:
    """
    Forces a deterministic scroll-probe before backtracking off an exhausted
    screen, stopping once a probe reveals nothing new or the probe cap is hit.
    """

    def __init__(self, *, config: ScrollProbeConfig) -> None:
        self.__maximum = config.maximum

    @property
    def maximum(self) -> int:
        """
        Maximum forced scroll-probes permitted on a single screen.
        """

        return self.__maximum

    def should_probe(self, *, probes: int, advanced: bool) -> bool:
        """
        Whether to force another scroll-probe before honouring exhaustion.

        Always allows a first probe; allows further probes only while the
        previous probe revealed new content and the cap is unspent.
        """

        if probes >= self.__maximum:
            return False
        if probes == 0:
            return True
        return advanced
