from __future__ import annotations

from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import ArtifactMetadata, ArtifactReceipt


class NoopSink(ArtifactSinkPort):
    """
    Sink that performs no remote persistence.

    Returns a receipt that asks the pipeline to keep the staged EFS
    copy — useful for development, fixture-replay loops, and tests
    where cloud upload is not desired.
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
