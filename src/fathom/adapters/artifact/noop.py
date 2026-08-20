from __future__ import annotations

from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import ArtifactMetadata, ArtifactReceipt


class NoopSink(ArtifactSinkPort):
    """
    Sink that persists nothing and keeps the locally staged copy.

    The receipt carries an empty identifier, so the pipeline records no durable artifact.
    Use it in development, fixture-replay loops, and tests where cloud upload is not wanted.
    """

    async def persist(
        self,
        *,
        content: bytes,
        metadata: ArtifactMetadata,
    ) -> ArtifactReceipt:
        """
        Acknowledge the call and leave the local copy in place.
        """

        _ = (metadata, content)
        return ArtifactReceipt(identifier="", local_cleanup=False)
