from __future__ import annotations

from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import ArtifactMetadata, ArtifactReceipt


class EfsSink(ArtifactSinkPort):
    """
    Sink that keeps the locally staged copy as the durable artifact.

    The pipeline stages bytes to local disk before any sink runs; this sink does no remote work
    and asks the pipeline to keep that file, marking the local-only-by-design path.
    It differs from :class:`NoopSink`, whose empty receipt means no durable artifact is kept.
    """

    async def persist(
        self,
        *,
        content: bytes,
        metadata: ArtifactMetadata,
    ) -> ArtifactReceipt:
        """
        Acknowledge the call and request that local files stay in place.
        """

        _ = (metadata, content)
        return ArtifactReceipt(identifier="efs.local", local_cleanup=False)
