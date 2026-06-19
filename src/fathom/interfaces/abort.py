from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.abort import AbortDecision


class AbortDetectorPort(ABC):
    """
    Domain port for classifying free-form HITL responses as workflow-abort intents.
    """

    @abstractmethod
    async def aborted(self, *, response: str) -> AbortDecision:
        """
        Classify whether the response commands the workflow to stop.
        """

        raise NotImplementedError

    @abstractmethod
    async def warmup(self) -> None:
        """
        Pre-load the underlying model to minimize first-call latency.
        """

        raise NotImplementedError
