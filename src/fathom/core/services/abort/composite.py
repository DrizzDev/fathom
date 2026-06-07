from __future__ import annotations

from logging import getLogger

from fathom.interfaces.abort import AbortDetectorPort
from fathom.schemas.abort import AbortDecision

logger = getLogger(__name__)


class CompositeAbortDetector(AbortDetectorPort):
    """
    Composes a primary abort detector with a fallback used when the primary abstains.
    """

    def __init__(self, *, primary: AbortDetectorPort, fallback: AbortDetectorPort) -> None:
        """
        Bind primary and fallback detectors; composite owns only the routing policy.
        """

        self.__primary = primary
        self.__fallback = fallback

    async def aborted(self, *, response: str) -> AbortDecision:
        """
        Try the primary first and route to the fallback when the primary signals abstention.
        """

        decision = await self.__primary.aborted(response=response)

        if not decision.fallback:
            return decision

        logger.info(
            "Primary abort detector abstained; consulting fallback",
            extra={
                "event": "abort.composite.fallback_invoked",
                "component": "core.services.abort.composite",
                "response.preview": response[:120],
            },
        )
        return await self.__fallback.aborted(response=response)

    async def warmup(self) -> None:
        """
        Warm both detectors so the first ASK_USER turn pays no cold-start tax.
        """

        await self.__primary.warmup()
        await self.__fallback.warmup()
