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

        self.__receipt = receipt
        self.calls: int = 0

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
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
        metadata: ArtifactMetadata,
        content: bytes,
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
            kind=ArtifactKind.SCREENSHOT,
            session_id="run-test",
            package_name="app",
            step_number=0,
            created=1,
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

        outcome = await composite.persist(metadata=self.__metadata(), content=b"PNG")

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

        outcome = await composite.persist(metadata=self.__metadata(), content=b"PNG")

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

        outcome = await composite.persist(metadata=self.__metadata(), content=b"PNG")

        self.assertFalse(outcome.local_cleanup)
