from __future__ import annotations

import unittest

from fathom.adapters.artifact.noop import NoopSink
from fathom.schemas.artifact import ArtifactKind, ArtifactMetadata


class NoopSinkTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that :class:`NoopSink` performs no work and never claims cleanup.

    The pipeline stages bytes to EFS before calling any sink; with the noop sink the only correct receipt asks the pipeline to leave the
    local copy in place. This is the test-fixture path: production artifacts simply aren't surfaced anywhere when ``NoopSink`` is wired.
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

    async def test_persist_returns_no_cleanup_receipt(self) -> None:
        """
        The noop sink reports ``local_cleanup=False`` regardless of input.
        """

        receipt = await NoopSink().persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertFalse(receipt.local_cleanup)
        self.assertEqual(receipt.identifier, "")
