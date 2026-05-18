from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.strategies.graph.intent.nodes.execute import ExecuteNode


class ExecuteNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the EXECUTE node's three early-exit branches.

    EXECUTE drives the device executor (or the HITL bridge for
    ASK_USER actions). The pins verify three conditions where the node
    must skip execution: cancellation, supervisor-set ``EXECUTION_BLOCKED``
    flag, and missing :class:`ExecutionContext`. None of these branches
    may call the action executor — otherwise the runtime would either
    waste an action or crash on a partial execution context.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        check, workflow id, and persistence helper used on the early
        branches. The action executor and HITL bridge stay unmocked
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
        node = ExecuteNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )

    async def test_execution_blocked_skips_executor(self) -> None:
        """
        ``EXECUTION_BLOCKED=True`` is set by SUPERVISE when the gate or
        healing path could not produce an allowed action. EXECUTE must
        return an empty patch so the graph routes to OBSERVE/RECORD with
        the pre-built blocked-step result already in state.
        """

        provider = self.__provider(cancelled=False)
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_BLOCKED: True},  # type: ignore[arg-type]
        )

        self.assertEqual(result, {})

    async def test_missing_execution_context_returns_empty_state(self) -> None:
        """
        An absent :class:`ExecutionContext` indicates SUPERVISE did not
        run or did not commit. EXECUTE must return an empty patch rather
        than synthesising a context — that would mask the upstream bug.
        """

        provider = self.__provider(cancelled=False)
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: None},  # type: ignore[arg-type]
        )

        self.assertEqual(result, {})
