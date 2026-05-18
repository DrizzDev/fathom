from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Dict, Tuple

from fathom.constants.artifact import ArtifactComponent
from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import ArtifactMetadata, ArtifactReceipt

logger = getLogger(__name__)


class CompositeSink(ArtifactSinkPort):
    """
    Decorator that fans one persist call out to many sinks.

    Each downstream sink runs concurrently. The composite reports
    ``local_cleanup=True`` only when **every** downstream sink agrees;
    a single dissenting receipt keeps the local copy in place so no
    sink ever loses access to the artifact prematurely.
    """

    def __init__(self, *, sinks: Tuple[ArtifactSinkPort, ...]) -> None:
        """
        Bind this composite to one ordered tuple of downstream sinks.
        """

        self.__sinks = sinks

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
    ) -> ArtifactReceipt:
        """
        Dispatch to every downstream sink concurrently and aggregate receipts.
        """

        receipts = await asyncio.gather(
            *(sink.persist(metadata=metadata, content=content) for sink in self.__sinks),
            return_exceptions=True,
        )

        cleanup_safe = True
        identifiers = []
        for receipt in receipts:
            if isinstance(receipt, BaseException):
                logger.warning(
                    "Composite sink branch raised; leaving local copy in place",
                    extra={
                        "component": ArtifactComponent.SINK_CLOUD,
                        "event": "artifact.composite.branch_failed",
                        "artifact.kind": metadata.kind.value,
                        "error.message": str(receipt),
                    },
                )
                cleanup_safe = False
                continue
            if not receipt.local_cleanup:
                cleanup_safe = False
            if receipt.identifier:
                identifiers.append(receipt.identifier)

        return ArtifactReceipt(
            identifier=",".join(identifiers),
            local_cleanup=cleanup_safe,
        )

    @property
    def downstream_count(self) -> int:
        """
        Number of downstream sinks this composite dispatches to.
        """

        return len(self.__sinks)

    @property
    def context(self) -> Dict[str, Any]:
        """
        Structured-logging component identifier for this composite.
        """

        return {"component": ArtifactComponent.SINK_CLOUD}
