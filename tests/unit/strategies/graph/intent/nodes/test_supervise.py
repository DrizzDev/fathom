from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.strategies.graph.intent.nodes.supervise import SuperviseNode


class SuperviseNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the SUPERVISE node's cancellation and missing-state branches.

    SUPERVISE runs localization, supervision, and bounded healing against
    the planned step. Both pins verify that the node degrades gracefully
    when the prerequisite state is absent: cancellation propagates the
    cancellation reason; missing planned step or capture returns an empty
    patch (no execution context produced) instead of raising.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        check, workflow id, and persistence helper used by the early
        branches. The gate / observer / localizer surfaces stay unmocked
        because they must not be reached on these branches.
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
        node = SuperviseNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )

    async def test_missing_planned_step_or_capture_returns_empty_state(self) -> None:
        """
        Missing planned step or capture must return an empty patch rather
        than crashing. The graph then re-enters GROUND on the next tick
        instead of producing a partial execution context downstream
        nodes would have to defend against.
        """

        provider = self.__provider(cancelled=False)
        node = SuperviseNode(provider=provider)

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: None,
                IntentStateKey.PLANNED_STEP: None,
            },
        )

        self.assertEqual(result, {})
