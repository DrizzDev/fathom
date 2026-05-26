from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider


class IntentNodeProviderCancellationTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers cancellation propagation at the node-provider boundary.
    """

    async def test_cancelled_signal_sets_context_cancel_event(self) -> None:
        """
        A provider-level CANCELLED signal must become graph-context state
        so later routers that only read ``context.is_cancelled`` stop too.
        """

        context = SimpleNamespace(
            is_cancelled=False,
            hitl=SimpleNamespace(check_signal=AsyncMock(return_value="CANCELLED")),
            cancel=Mock(),
            llm=Mock(),
        )
        provider = IntentNodeProvider(
            context=context,  # type: ignore[arg-type]
            screen_comparator=Mock(),  # type: ignore[arg-type]
        )

        self.assertTrue(await provider.is_cancelled())
        context.cancel.assert_called_once_with()

    async def test_existing_context_cancellation_does_not_poll_signal(self) -> None:
        """
        Existing graph cancellation should short-circuit signal polling.
        """

        context = SimpleNamespace(
            is_cancelled=True,
            hitl=SimpleNamespace(check_signal=AsyncMock(return_value=None)),
            cancel=Mock(),
            llm=Mock(),
        )
        provider = IntentNodeProvider(
            context=context,  # type: ignore[arg-type]
            screen_comparator=Mock(),  # type: ignore[arg-type]
        )

        self.assertTrue(await provider.is_cancelled())
        context.hitl.check_signal.assert_not_awaited()
        context.cancel.assert_not_called()
