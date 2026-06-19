from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.exceptions import HITLNotAvailableError
from fathom.schemas.actions import Action
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.execute import ExecuteNode


class ExecuteNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the EXECUTE node's early-exit branches.

    EXECUTE drives the device executor (or the HITL bridge for
    ASK_USER actions). The pins verify two conditions where the node
    must skip execution: cancellation and missing :class:`ExecutionContext`.
    None of these branches
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

    @staticmethod
    def __capture() -> ScreenCapture:
        """Build a minimal screen capture for ExecutionContext."""

        return ScreenCapture(width=100, height=200, activity="app", image=b"", timestamp=1)

    @staticmethod
    def __ask_user_execution_context() -> ExecutionContext:
        """Build an ExecutionContext carrying an ASK_USER step."""

        action = Action(
            action_type=ActionType.ASK_USER,
            target="User",
            confidence=1.0,
            rationale="missing credentials",
            text="What is your OTP?",
        )
        step = Step(step_number=3, screen_hash="v", action=action)
        return ExecutionContext(
            step=step,
            capture=ExecuteNodeEarlyExitTest.__capture(),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                bounds=None,
                source=None,
                confidence=0.0,
                reason="ask_user_bypass",
            ),
            package="app",
        )

    async def test_ask_user_with_hitl_unavailable_routes_back_to_ground(self) -> None:
        """HITLNotAvailableError must clear the planned step and set SHOULD_RETRY."""

        provider = self.__provider(cancelled=False)
        provider.hitl.ask = AsyncMock(side_effect=HITLNotAvailableError())
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: self.__ask_user_execution_context()},
        )

        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertIsNone(result.get(IntentStateKey.PLAN))
        self.assertIsNone(result.get(IntentStateKey.PLANNED_STEP))
        self.assertIsNone(result.get(IntentStateKey.EXECUTION_CONTEXT))
        self.assertIn(
            CommonStateKey.FAILURE_DIAGNOSTIC,
            result,
        )
        provider.persistence.persist.assert_called()

    async def test_missing_execution_context_fails_terminally(self) -> None:
        """
        An absent :class:`ExecutionContext` indicates SUPERVISE did not
        run or did not commit. EXECUTE must fail terminally rather than
        returning an empty patch that lets downstream fixed edges loop.
        """

        provider = self.__provider(cancelled=False)
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: None},  # type: ignore[arg-type]
        )

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertFalse(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.FAILED.value,
        )
        self.assertIn("missing ExecutionContext", result.get(CommonStateKey.FAILURE_DIAGNOSTIC))
        provider.context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.FAILED.value
        )
