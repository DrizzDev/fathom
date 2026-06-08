from __future__ import annotations

import unittest

from fathom.adapters.artifact.composite import CompositeSink
from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import ArtifactKind, ArtifactMetadata, ArtifactReceipt


class _FixedSink(ArtifactSinkPort):
    """
    Test double returning a frozen receipt for every persist call.
    """

    def __init__(self, *, receipt: ArtifactReceipt) -> None:
        """
        Bind this sink to one frozen receipt.
        """

        self.calls: int = 0
        self.__receipt = receipt

    async def persist(
        self,
        *,
        content: bytes,
        metadata: ArtifactMetadata,
    ) -> ArtifactReceipt:
        """
        Increment the call counter and return the bound receipt.
        """

        _ = (metadata, content)
        self.calls += 1
        return self.__receipt


class _RaisingSink(ArtifactSinkPort):
    """
    Test double that raises on every persist call.
    """

    async def persist(
        self,
        *,
        content: bytes,
        metadata: ArtifactMetadata,
    ) -> ArtifactReceipt:
        """
        Raise to drive the composite's branch-failure path.
        """

        _ = (metadata, content)
        raise RuntimeError("downstream failure")


class CompositeSinkTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :class:`CompositeSink` fan-out and conservative-cleanup policy.
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

    async def test_all_cleanup_safe_branches_clear_local(self) -> None:
        """
        When every branch returns ``local_cleanup=True`` the composite
        agrees with the unanimous vote.
        """

        receipts = (
            ArtifactReceipt(identifier="cloud://a", local_cleanup=True),
            ArtifactReceipt(identifier="cloud://b", local_cleanup=True),
        )
        composite = CompositeSink(sinks=tuple(_FixedSink(receipt=r) for r in receipts))

        outcome = await composite.persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertTrue(outcome.local_cleanup)
        self.assertIn("cloud://a", outcome.identifier)
        self.assertIn("cloud://b", outcome.identifier)

    async def test_single_dissenter_blocks_local_cleanup(self) -> None:
        """
        One ``local_cleanup=False`` branch is enough to keep the local copy.
        """

        receipts = (
            ArtifactReceipt(identifier="cloud://a", local_cleanup=True),
            ArtifactReceipt(identifier="", local_cleanup=False),
        )
        composite = CompositeSink(sinks=tuple(_FixedSink(receipt=r) for r in receipts))

        outcome = await composite.persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertFalse(outcome.local_cleanup)

    async def test_branch_exception_does_not_propagate(self) -> None:
        """
        A raising branch is logged but never crashes the composite;
        cleanup is suppressed conservatively.
        """

        composite = CompositeSink(
            sinks=(
                _FixedSink(receipt=ArtifactReceipt(identifier="cloud://a", local_cleanup=True)),
                _RaisingSink(),
            ),
        )

        outcome = await composite.persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertFalse(outcome.local_cleanup)
