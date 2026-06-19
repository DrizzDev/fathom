from __future__ import annotations

import unittest

from fathom.adapters.artifact.efs import EfsSink
from fathom.schemas.artifact import ArtifactKind, ArtifactMetadata


class EfsSinkTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that :class:`EfsSink` opts out of remote persistence while distinguishing itself from :class:`NoopSink`.

    The pipeline stages every payload to EFS before invoking the sink, so this sink need only acknowledge and ask the pipeline to keep the
    file in place. The non-empty identifier marks "EFS is the durable home" so downstream consumers can tell the dev / local-only path apart from the noop drop-everything path.
    """

    @staticmethod
    def __metadata() -> ArtifactMetadata:
        """
        Minimal :class:`ArtifactMetadata` fixture identifying the artifact.
        """

        return ArtifactMetadata(
            created=1,
            step_number=0,
            package_name="app",
            session_id="run-test",
            kind=ArtifactKind.SCREENSHOT,
            filename="step-000__screenshot__2026-01-01T00-00-00Z-000.png",
        )

    async def test_persist_returns_local_only_receipt(self) -> None:
        """
        The EFS sink reports ``local_cleanup=False`` with a sentinel identifier.
        """

        receipt = await EfsSink().persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertFalse(receipt.local_cleanup)
        self.assertEqual(receipt.identifier, "efs.local")
