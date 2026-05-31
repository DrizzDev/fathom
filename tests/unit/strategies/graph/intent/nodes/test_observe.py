from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionExecutionKind, ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.strategies.graph.intent.nodes.observe import ObserveNode


class ObserveNodeFailureTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins terminal failure behavior for invalid OBSERVE inputs.
    """

    @staticmethod
    def __provider() -> MagicMock:
        """
        Return the provider surface used by the invalid-state branch.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        provider.is_cancelled = AsyncMock(return_value=False)
        return provider

    async def test_missing_execution_result_fails_terminally(self) -> None:
        """
        OBSERVE must not return an empty patch when EXECUTE did not stage a result.
        """

        provider = self.__provider()
        node = ObserveNode(provider=provider)

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
        provider.persistence.persist.assert_called_once()


class ObserveNodeStepSuccessTest(unittest.TestCase):
    """
    Pins gesture step-success handling after post-action effect classification.
    """

    def test_gesture_no_progress_is_not_recorded_as_success(self) -> None:
        """
        A dispatched gesture with a no-progress effect must fail the recorded step.
        """

        effect = ActionEffect(
            status=ActionEffectStatus.NO_PROGRESS,
            visual_progress=0.0,
            phash_distance=0,
        )

        result = ObserveNode._ObserveNode__step_success(  # noqa: SLF001
            action_type=ActionType.SWIPE_DOWN,
            action_effect=effect,
            action_execution_kind=ActionExecutionKind.DEVICE,
            execution_success=True,
        )

        self.assertFalse(result)

    def test_non_gesture_success_keeps_execution_result(self) -> None:
        """
        Non-gesture device actions still use the adapter execution result.
        """

        effect = ActionEffect(
            status=ActionEffectStatus.NO_PROGRESS,
            visual_progress=0.0,
            phash_distance=0,
        )

        result = ObserveNode._ObserveNode__step_success(  # noqa: SLF001
            action_type=ActionType.TAP,
            action_effect=effect,
            action_execution_kind=ActionExecutionKind.DEVICE,
            execution_success=True,
        )

        self.assertTrue(result)
