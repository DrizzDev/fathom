from __future__ import annotations

from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import ArtifactMetadata, ArtifactReceipt


class EfsSink(ArtifactSinkPort):
    """
    Sink that explicitly retains the EFS-staged copy.

    The pipeline stages bytes to EFS before invoking any sink. This
    sink reports success without doing any remote work and asks the
    pipeline to keep the local file, distinguishing the "local-only by
    design" path from :class:`NoopSink` (which represents "drop
    everything; tests don't care about artifacts").
    """

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
    ) -> ArtifactReceipt:
        """
        Acknowledge the call and request that local files stay in place.
        """

        _ = (metadata, content)
        return ArtifactReceipt(identifier="efs.local", local_cleanup=False)
