from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.healing import HealingDecision, HealingRequest


class HealingAgentPort(ABC):
    """
    Produces a bounded recovery decision for blocked runtime execution.
    """

    @abstractmethod
    async def decide(self, *, request: HealingRequest) -> HealingDecision:
        """
        Decide how to recover from one blocked execution state.
        """

        raise NotImplementedError
