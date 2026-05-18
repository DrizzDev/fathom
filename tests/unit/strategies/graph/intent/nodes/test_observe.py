from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import IntentStateKey
from fathom.strategies.graph.intent.nodes.observe import ObserveNode


class ObserveNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the OBSERVE node's two early-exit branches.

    OBSERVE captures the post-action evidence and classifies the outcome.
    When the supervisor blocked the action OBSERVE must skip the
    post-capture entirely — there is no action to observe. When the
    :class:`ExecutionContext` is missing or has no ``execution_result``
    the node returns an empty patch rather than crashing, so the graph
    can resume on the next tick.
    """

    @staticmethod
    def __provider() -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        flag, workflow id, and persistence helper used on the early
        branches. The post-action effects pipeline stays unmocked because
        it must not be reached on these branches.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        return provider

    async def test_execution_blocked_skips_post_capture(self) -> None:
        """
        ``EXECUTION_BLOCKED=True`` was set by SUPERVISE; the action never
        ran so the post-action capture is meaningless. OBSERVE must
        return an empty patch and leave the graph state untouched.
        """

        node = ObserveNode(provider=self.__provider())

        result: Any = await node(
            state={IntentStateKey.EXECUTION_BLOCKED: True},  # type: ignore[arg-type]
        )

        self.assertEqual(result, {})

    async def test_missing_execution_context_returns_empty_state(self) -> None:
        """
        Missing :class:`ExecutionContext` or absent ``execution_result``
        means EXECUTE did not commit. Returning an empty patch instead of
        raising lets the graph resume on the next tick rather than tearing
        down the whole run on a transient upstream bug.
        """

        node = ObserveNode(provider=self.__provider())

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: None},  # type: ignore[arg-type]
        )

        self.assertEqual(result, {})
