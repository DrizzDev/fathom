from __future__ import annotations

import unittest

from fathom.adapters.artifact.noop import NoopSink
from fathom.schemas.artifact import ArtifactKind, ArtifactMetadata


class NoopSinkTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that :class:`NoopSink` performs no work and never claims cleanup.

    The pipeline stages bytes to EFS before calling any sink; with the
    noop sink the only correct receipt asks the pipeline to leave the
    local copy in place. This is the test-fixture path: production
    artifacts simply aren't surfaced anywhere when ``NoopSink`` is wired.
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

    async def test_persist_returns_no_cleanup_receipt(self) -> None:
        """
        The noop sink reports ``local_cleanup=False`` regardless of input.
        """

        receipt = await NoopSink().persist(metadata=self.__metadata(), content=b"PNG")

        self.assertFalse(receipt.local_cleanup)
        self.assertEqual(receipt.identifier, "")
