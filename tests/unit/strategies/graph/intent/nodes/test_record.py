from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
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
        provider.persistence.should_skip_launcher.return_value = False
        provider.persistence.enqueue_history = MagicMock()
        provider.context.memory.store_experience = AsyncMock()
        provider.context.context_manager.commit = AsyncMock()
        provider.context.context_manager.get_full_context.return_value = {"active_count": 0}
        provider.context.telemetry.info = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.telemetry.warning = AsyncMock()
        provider.context.agent_state.is_stuck = False
        provider.context.agent_state.step_count = 0
        provider.context.auditor.log_step = MagicMock()
        provider.completion.evaluate = AsyncMock(return_value=None)
        provider.completion.recover_if_stuck = AsyncMock(return_value=None)
        provider.context.recovery.try_recover = AsyncMock(return_value=None)
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

    async def test_recorded_step_is_committed_to_agent_state(self) -> None:
        """
        RECORD must commit the executed step into AgentState exactly once.
        """

        provider = self.__provider(cancelled=False)
        provider.context.agent_state.record_step = MagicMock()
        node = RecordNode(provider=provider)
        step_result = StepResult(
            step=Step(
                action=Action(
                    action_type=ActionType.TAP,
                    target="Continue",
                    rationale="tap continue",
                    confidence=1.0,
                ),
                event_type="action",
                condition=None,
                screen_hash="0" * 16,
                step_number=0,
            ),
            success=True,
            pre_hash="0" * 16,
            post_hash="1" * 16,
            screen_changed=True,
            duration=10,
            observation="screen changed",
        )

        result: Any = await node(
            state={
                CommonStateKey.STEP_RESULT: step_result,
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=1080,
                    height=1920,
                    activity="app",
                    image=b"PNG",
                    timestamp=1,
                ),
                IntentStateKey.STEP_RESULTS: [],
            }
        )  # type: ignore[arg-type]

        provider.context.agent_state.record_step.assert_called_once_with(result=step_result)
        self.assertEqual(result.get(IntentStateKey.STEP_RESULTS), [step_result])

    async def test_failed_step_is_still_recorded(self) -> None:
        """
        Failed recorded steps must still be committed into AgentState.
        """

        provider = self.__provider(cancelled=False)
        provider.context.agent_state.record_step = MagicMock()
        node = RecordNode(provider=provider)
        step_result = StepResult(
            step=Step(
                action=Action(
                    action_type=ActionType.TAP,
                    target="Continue",
                    rationale="tap continue",
                    confidence=1.0,
                ),
                event_type="action",
                condition=None,
                screen_hash="0" * 16,
                step_number=0,
            ),
            success=False,
            pre_hash="0" * 16,
            post_hash="0" * 16,
            screen_changed=False,
            duration=10,
            error="still blocked",
            observation="screen unchanged",
        )

        await node(
            state={
                CommonStateKey.STEP_RESULT: step_result,
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=1080,
                    height=1920,
                    activity="app",
                    image=b"PNG",
                    timestamp=1,
                ),
                IntentStateKey.STEP_RESULTS: [],
            }
        )  # type: ignore[arg-type]

        provider.context.agent_state.record_step.assert_called_once_with(result=step_result)
