from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionExecutionKind, ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.schemas.actions import Action
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
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

    @staticmethod
    def __node() -> ObserveNode:
        """
        Build an ObserveNode whose provider context exposes the real command catalog.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.context.catalog = CommandCatalogProvider().build()
        return ObserveNode(provider=provider)

    def test_gesture_no_progress_is_not_recorded_as_success(self) -> None:
        """
        A dispatched gesture with a no-progress effect must fail the recorded step.
        """

        effect = ActionEffect(
            status=ActionEffectStatus.NO_PROGRESS,
            visual_progress=0.0,
            phash_distance=0,
        )

        result = self.__node()._ObserveNode__step_success(  # noqa: SLF001
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

        result = self.__node()._ObserveNode__step_success(  # noqa: SLF001
            action_type=ActionType.TAP,
            action_effect=effect,
            action_execution_kind=ActionExecutionKind.DEVICE,
            execution_success=True,
        )

        self.assertTrue(result)


class ObserveNodeExecutedWiringTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that OBSERVE sources StepResult.executed from raw ExecutionResult.success, independent of the vetoed step success.
    """

    @staticmethod
    def __context() -> ExecutionContext:
        """
        Build an execution context for a gesture whose device primitive ran successfully.
        """

        action = Action(action_type=ActionType.SWIPE_DOWN, rationale="scroll the list")
        return ExecutionContext(
            package="app",
            step=Step(step_number=3, screen_hash="v", action=action),
            capture=ScreenCapture(width=1080, height=2340, activity="app", image=b"", timestamp=1),
            localization=LocalizationResult(status=LocalizationStatus.RESOLVED, confidence=1.0),
            execution_result=ExecutionResult(success=True, duration=10),
        )

    def __node(self) -> ObserveNode:
        """
        Build an ObserveNode whose effects report a no-progress gesture over a successful primitive.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.workflow_id = "run-test"
        provider.context.workflow_id = "run-test"
        provider.context.catalog = CommandCatalogProvider().build()
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.persistence.persist = MagicMock()
        provider.observer.fallback_observation = AsyncMock(return_value=None)
        provider.effects.observe = AsyncMock(return_value=(None, None, "post", "app", None))
        provider.effects.changed = MagicMock(return_value=False)
        provider.effects.log_diff = MagicMock()
        provider.effects.effect_from = MagicMock(
            return_value=ActionEffect(
                status=ActionEffectStatus.NO_PROGRESS,
                visual_progress=0.0,
                phash_distance=0,
            )
        )
        return ObserveNode(provider=provider)

    async def test_no_progress_gesture_is_executed_but_step_unsuccessful(self) -> None:
        """
        A no-progress gesture yields StepResult.executed=True (raw success) while StepResult.success=False (vetoed).
        """

        result: Any = await self.__node()(
            state={IntentStateKey.EXECUTION_CONTEXT: self.__context()},  # type: ignore[arg-type]
        )

        step_result = result[CommonStateKey.STEP_RESULT]
        self.assertTrue(step_result.executed)
        self.assertFalse(step_result.success)
