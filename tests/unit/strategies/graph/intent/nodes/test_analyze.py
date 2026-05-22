from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.command import CommandExecutionMode
from fathom.constants.runtime import DEFAULT_COMPLETE_DEFERRAL_BUDGET
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenCapture
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
        provider.hitl.prompt = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.max_steps = 10
        provider.context.agent_state.step_count = 0
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

    async def test_max_steps_terminates_before_planning(self) -> None:
        """
        ANALYZE must not plan step 11 when the recorded step ceiling is already reached.
        """

        provider = self.__provider(cancelled=False)
        provider.context.max_steps = 5
        provider.context.agent_state.step_count = 5
        node = AnalyzeNode(provider=provider)

        result: Any = await node(state={CommonStateKey.CAPTURE: None})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.MAX_STEPS.value,
        )
        provider.context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.MAX_STEPS.value
        )
        provider.hitl.prompt.assert_not_awaited()

    async def test_complete_verdict_is_deferred_while_sub_goals_remain(self) -> None:
        """
        ANALYZE must defer an ``is_complete`` planner verdict when sub-goals
        are still open and the local deferral budget is not exhausted.
        """

        provider = self.__provider(cancelled=False)
        provider.context.agent_state.step_count = 4
        provider.context.agent_state.has_sub_goals.return_value = True
        provider.context.agent_state.all_sub_goals_complete.return_value = False
        provider.context.agent_state.record_complete_deferral.return_value = 1
        provider.context.agent_state.reset_completion = MagicMock()
        provider.context.agent_state.reset_complete_deferrals = MagicMock()
        provider.context.device.get_dimensions = AsyncMock(return_value=(1080, 1920))
        provider.context.signal.supports_interruption.return_value = False
        provider.context.configuration.intent.prompt_user_if_stuck = False
        provider.context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        provider.context.context_manager.get_user_guidance.return_value = []
        provider.context.metrics.record = MagicMock()
        provider.context.planner.plan_step = AsyncMock(
            return_value=PlanResult(
                reason="planner claims done",
                is_complete=True,
                step=None,
            )
        )

        node = AnalyzeNode(provider=provider)
        capture = ScreenCapture(
            width=1080,
            height=1920,
            activity="app",
            image=b"PNG",
            timestamp=1,
        )

        result: Any = await node(state={CommonStateKey.CAPTURE: capture})  # type: ignore[arg-type]

        self.assertFalse(result.get(CommonStateKey.IS_COMPLETE))
        self.assertIsNone(result.get(CommonStateKey.COMPLETION_REASON))
        provider.context.agent_state.reset_completion.assert_called_once()
        provider.context.agent_state.reset_complete_deferrals.assert_not_called()

    async def test_complete_verdict_is_honoured_after_deferral_budget_exhausts(self) -> None:
        """
        Once the complete-deferral streak exceeds its budget, ANALYZE must
        stop bouncing back to GROUND and let VERIFY adjudicate the run.
        """

        provider = self.__provider(cancelled=False)
        provider.context.agent_state.step_count = 4
        provider.context.agent_state.has_sub_goals.return_value = True
        provider.context.agent_state.all_sub_goals_complete.return_value = False
        provider.context.agent_state.record_complete_deferral.return_value = (
            DEFAULT_COMPLETE_DEFERRAL_BUDGET + 1
        )
        provider.context.agent_state.reset_completion = MagicMock()
        provider.context.agent_state.reset_complete_deferrals = MagicMock()
        provider.context.device.get_dimensions = AsyncMock(return_value=(1080, 1920))
        provider.context.signal.supports_interruption.return_value = False
        provider.context.configuration.intent.prompt_user_if_stuck = False
        provider.context.configuration.intent.command_mode = CommandExecutionMode.STRICT
        provider.context.context_manager.get_user_guidance.return_value = []
        provider.context.metrics.record = MagicMock()
        provider.context.planner.plan_step = AsyncMock(
            return_value=PlanResult(
                reason="planner claims done",
                is_complete=True,
                step=None,
            )
        )

        node = AnalyzeNode(provider=provider)
        capture = ScreenCapture(
            width=1080,
            height=1920,
            activity="app",
            image=b"PNG",
            timestamp=1,
        )

        result: Any = await node(state={CommonStateKey.CAPTURE: capture})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(result.get(CommonStateKey.COMPLETION_REASON), "planner claims done")
        provider.context.agent_state.reset_complete_deferrals.assert_called_once()
