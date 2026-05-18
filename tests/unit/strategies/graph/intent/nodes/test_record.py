from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason
from fathom.strategies.graph.intent.nodes.record import RecordNode


class RecordNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the RECORD node's cancellation and missing-step-result branches.

    RECORD owns history persistence, audit logging, and sub-goal
    advancement. The pins verify both early exits leave the graph in a
    consistent state: cancellation marks the run complete with the
    cancelled reason; a missing :class:`StepResult` returns an empty
    patch (history accumulator stays untouched) instead of raising.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        check, workflow id, and persistence helper used on the early
        branches. The history / memory / context-manager surfaces stay
        unmocked because they must not be reached on these branches.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        return provider

    async def test_cancellation_marks_complete(self) -> None:
        """
        A cancelled run must terminate with :attr:`CompletionReason.CANCELLED`.
        """

        provider = self.__provider(cancelled=True)
        node = RecordNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )

    async def test_missing_step_result_returns_empty_patch(self) -> None:
        """
        A graph state without a :class:`StepResult` is upstream-broken
        (OBSERVE failed). RECORD must return an empty patch so the
        accumulator is not polluted and the run can resume on the next
        tick instead of force-closing here.
        """

        node = RecordNode(provider=self.__provider(cancelled=False))

        result: Any = await node(state={CommonStateKey.STEP_RESULT: None})  # type: ignore[arg-type]

        self.assertEqual(result, {})
