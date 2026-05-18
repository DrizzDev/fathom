from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode


class AnalyzeNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the ANALYZE node's cancellation and missing-capture branches.

    ANALYZE owns the planner call and is the single most expensive node
    in the graph. The pins verify that both early-exit branches short-
    circuit *before* invoking the LLM: cancellation propagates the
    cancelled completion reason; a missing/invalid capture flips
    ``SHOULD_RETRY`` so GROUND re-enters the loop.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        check, workflow id, and persistence hooks used on the early-exit
        branches. The planner / telemetry surfaces stay unmocked because
        they must never be reached on these branches.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        provider.persistence.restore = MagicMock()
        return provider

    async def test_cancellation_marks_complete_and_persists(self) -> None:
        """
        A cancelled run must terminate with :attr:`CompletionReason.CANCELLED`
        and persist the terminal state via the persistence helper so the
        checkpoint reflects the cancellation.
        """

        provider = self.__provider(cancelled=True)
        node = AnalyzeNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )
        provider.persistence.persist.assert_called()

    async def test_missing_capture_flips_should_retry(self) -> None:
        """
        Missing or invalid screen capture must not crash the planner.
        Instead the node flips ``SHOULD_RETRY`` so GROUND re-enters the
        loop and re-captures the screen on the next turn.
        """

        provider = self.__provider(cancelled=False)
        node = AnalyzeNode(provider=provider)

        result: Any = await node(state={CommonStateKey.CAPTURE: None})  # type: ignore[arg-type]

        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
