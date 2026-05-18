from __future__ import annotations

import unittest

from fathom.adapters.artifact.efs import EfsSink
from fathom.schemas.artifact import ArtifactKind, ArtifactMetadata


class EfsSinkTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that :class:`EfsSink` opts out of remote persistence while
    distinguishing itself from :class:`NoopSink`.

    The pipeline stages every payload to EFS before invoking the sink,
    so this sink need only acknowledge and ask the pipeline to keep the
    file in place. The non-empty identifier marks "EFS is the durable
    home" so downstream consumers can tell the dev / local-only path
    apart from the noop drop-everything path.
    """

    @staticmethod
    def __metadata() -> ArtifactMetadata:
        """
        Minimal :class:`ArtifactMetadata` fixture identifying the artifact.
        """

        return ArtifactMetadata(
            kind=ArtifactKind.SCREENSHOT,
            session_id="run-test",
            package_name="app",
            step_number=0,
            created=1,
        )

    async def test_persist_returns_local_only_receipt(self) -> None:
        """
        The EFS sink reports ``local_cleanup=False`` with a sentinel identifier.
        """

        receipt = await EfsSink().persist(metadata=self.__metadata(), content=b"PNG")

        self.assertFalse(receipt.local_cleanup)
        self.assertEqual(receipt.identifier, "efs.local")
