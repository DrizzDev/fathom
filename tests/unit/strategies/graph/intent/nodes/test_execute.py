from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from fathom.constants import ActionType
from fathom.constants.collaboration import TaskCode, TaskState
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.exceptions import HITLNotAvailableError, HITLTimeoutError
from fathom.schemas.actions import Action
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.results import ExecutionResult
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
        provider.context.catalog = CommandCatalogProvider().build()
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
        """
        Build a minimal screen capture for ExecutionContext.
        """

        return ScreenCapture(width=100, height=200, activity="app", image=b"", timestamp=1)

    @staticmethod
    def __ask_user_execution_context() -> ExecutionContext:
        """
        Build an ExecutionContext carrying an ASK_USER step.
        """

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

    @staticmethod
    def __tap_execution_context() -> ExecutionContext:
        """
        Build an ExecutionContext carrying a device action step.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Continue",
            confidence=1.0,
            rationale="tap continue",
        )
        step = Step(step_number=1, screen_hash="v", action=action)
        return ExecutionContext(
            step=step,
            capture=ExecuteNodeEarlyExitTest.__capture(),
            localization=LocalizationResult(
                status=LocalizationStatus.RESOLVED,
                bounds=None,
                source=None,
                confidence=1.0,
                reason="unit_test",
            ),
            package="app",
        )

    async def test_ask_user_unavailable_after_start_closes_task_failed_then_replans(self) -> None:
        """
        A post-start HITLNotAvailableError closes the opened step as FAILED/UNKNOWN_ERROR (no dangling RUNNING task), then replans rather than terminating.
        """

        provider = self.__provider(cancelled=False)
        provider.hitl.available = Mock(return_value=True)
        provider.context.tenant = "tenant"
        provider.context.thread = "thread"
        provider.context.responder = "responder"
        provider.context.workspace = None
        provider.context.workflow_id = "run-test"
        provider.context.execution_id = "exec"
        provider.context.recorder.record_step_started = AsyncMock()
        provider.context.recorder.record_step_finished = AsyncMock()
        provider.context.telemetry.warning = AsyncMock()
        provider.hitl.ask = AsyncMock(side_effect=HITLNotAvailableError())
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: self.__ask_user_execution_context()},
        )

        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertIsNone(result.get(IntentStateKey.PLAN))
        self.assertIsNone(result.get(IntentStateKey.PLANNED_STEP))
        self.assertIsNone(result.get(IntentStateKey.EXECUTION_CONTEXT))
        self.assertIn(CommonStateKey.FAILURE_DIAGNOSTIC, result)
        # The task was opened once and closed once — nothing left RUNNING.
        provider.context.recorder.record_step_started.assert_awaited_once()
        provider.context.recorder.record_step_finished.assert_awaited_once()
        completion = provider.context.recorder.record_step_finished.await_args.kwargs["completion"]
        self.assertEqual(completion.state, TaskState.FAILED)
        self.assertEqual(completion.code, TaskCode.UNKNOWN_ERROR)
        # A failed dispatch replans; it does not terminate the run.
        provider.context.agent_state.mark_complete.assert_not_called()
        provider.persistence.persist.assert_called()

    async def test_ask_user_unavailable_replans_without_opening_step(self) -> None:
        """
        With HITL unavailable, EXECUTE must replan before opening a persisted step so no task dangles.
        """

        provider = self.__provider(cancelled=False)
        provider.hitl.available = Mock(return_value=False)
        provider.context.recorder.record_step_started = AsyncMock()
        provider.hitl.ask = AsyncMock()
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: self.__ask_user_execution_context()},
        )

        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        provider.context.recorder.record_step_started.assert_not_called()
        provider.hitl.ask.assert_not_called()

    async def test_ask_user_follows_bridge_authority_on_capability_mismatch(self) -> None:
        """
        Single authority: when the bridge (the same authority ask() uses) reports unavailable, EXECUTE replans without opening a task even though agent_state's capabilities disagree — so no RUNNING task can dangle.
        """

        provider = self.__provider(cancelled=False)
        provider.context.agent_state.capabilities.hitl.enabled = True  # disagrees with the bridge
        provider.hitl.available = Mock(return_value=False)
        provider.context.recorder.record_step_started = AsyncMock()
        provider.hitl.ask = AsyncMock(side_effect=HITLNotAvailableError())
        node = ExecuteNode(provider=provider)

        result: Any = await node(
            state={IntentStateKey.EXECUTION_CONTEXT: self.__ask_user_execution_context()},
        )

        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        provider.context.recorder.record_step_started.assert_not_called()
        provider.hitl.ask.assert_not_called()

    async def test_ask_user_timeout_closes_step_and_terminates(self) -> None:
        """
        A timed-out intervention closes the opened step as EXPIRED/TIMEOUT and terminates the run.
        """

        provider = self.__provider(cancelled=False)
        provider.hitl.available = Mock(return_value=True)
        provider.context.tenant = "tenant"
        provider.context.thread = "thread"
        provider.context.workflow_id = "run-test"
        provider.context.execution_id = "exec"
        provider.context.recorder.record_step_started = AsyncMock()
        provider.context.recorder.record_step_finished = AsyncMock()
        provider.context.telemetry.warning = AsyncMock()
        provider.hitl.ask = AsyncMock(side_effect=HITLTimeoutError())
        node = ExecuteNode(provider=provider)

        with patch(
            "fathom.strategies.graph.intent.nodes.execute.time.time",
            side_effect=[1000.0, 1002.5],
        ):
            result: Any = await node(
                state={IntentStateKey.EXECUTION_CONTEXT: self.__ask_user_execution_context()},
            )

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.INTERVENTION_REQUIRED.value,
        )
        provider.context.recorder.record_step_finished.assert_awaited_once()
        completion = provider.context.recorder.record_step_finished.await_args.kwargs["completion"]
        self.assertEqual(completion.state, TaskState.EXPIRED)
        self.assertEqual(completion.code, TaskCode.TIMEOUT)
        # Timing honesty: the elapsed reflects the real wait, not a hardcoded zero.
        self.assertEqual(completion.elapsed, 2500)

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

    async def test_execution_context_carries_step_started_at_metadata(self) -> None:
        """
        EXECUTE must preserve deterministic step start time for recording.
        """

        provider = self.__provider(cancelled=False)
        provider.context.recorder = None
        provider.context.action_executor.act = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                duration=10,
                screen_changed=True,
            )
        )
        node = ExecuteNode(provider=provider)

        await node(
            state={IntentStateKey.EXECUTION_CONTEXT: self.__tap_execution_context()},
        )

        result = provider.persistence.persist.call_args.kwargs["result"]
        context = result[IntentStateKey.EXECUTION_CONTEXT]
        self.assertIn("started_at", context.step.metadata)
