from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants import ActionType
from fathom.constants.completion import GateOutcome, RetainReason
from fathom.core.agent.capture import StoreCaptureCompletionPolicy
from fathom.core.capture.store import CaptureStore
from fathom.schemas.actions import Action
from fathom.schemas.capture import Capture, CaptureRequest
from fathom.schemas.steps import Step, StepResult


class StoreCaptureCompletionPolicyTest(unittest.TestCase):
    """
    Pins STORE-only completion: advance solely on a successful, non-empty capture for the request name.
    """

    @staticmethod
    def __step_result(*, capture: Optional[CaptureRequest], success: bool = True) -> StepResult:
        """
        Build a STORE step result carrying the given capture request and execution success.
        """

        action = Action(action_type=ActionType.STORE, rationale="capture", capture=capture)
        return StepResult(
            step=Step(step_number=1, screen_hash="v", action=action),
            success=success,
            executed=success,
            duration=1,
            pre_hash="a",
            post_hash="b",
            screen_changed=False,
        )

    @staticmethod
    def __request() -> CaptureRequest:
        """
        Build a capture request stored under 'abc'.
        """

        return CaptureRequest(name="abc", subject="xyz", value="xyz")

    def test_advances_when_successful_capture_exists(self) -> None:
        """
        A successful, non-empty capture for the request name advances the sub-goal.
        """

        store = CaptureStore()
        store.write(capture=Capture.succeeded(name="abc", value="xyz", step=1))

        decision = StoreCaptureCompletionPolicy().evaluate(
            step_result=self.__step_result(capture=self.__request()),
            capture_store=store,
        )

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)
        self.assertIsNone(decision.retain_reason)

    def test_retains_when_capture_request_missing(self) -> None:
        """
        A STORE step with no capture request retains with MISSING_CAPTURE_REQUEST.
        """

        decision = StoreCaptureCompletionPolicy().evaluate(
            step_result=self.__step_result(capture=None),
            capture_store=CaptureStore(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_CAPTURE_REQUEST)

    def test_retains_when_step_execution_failed(self) -> None:
        """
        A STORE step whose execution failed retains with STEP_EXECUTION_FAILED.
        """

        decision = StoreCaptureCompletionPolicy().evaluate(
            step_result=self.__step_result(capture=self.__request(), success=False),
            capture_store=CaptureStore(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.STEP_EXECUTION_FAILED)

    def test_retains_when_store_missing_capture(self) -> None:
        """
        No capture recorded under the request name retains with MISSING_CAPTURE.
        """

        decision = StoreCaptureCompletionPolicy().evaluate(
            step_result=self.__step_result(capture=self.__request()),
            capture_store=CaptureStore(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_CAPTURE)

    def test_retains_when_capture_failed(self) -> None:
        """
        A recorded failed capture retains with CAPTURE_FAILED.
        """

        store = CaptureStore()
        store.write(capture=Capture.failed(name="abc", reason="no element", step=1))

        decision = StoreCaptureCompletionPolicy().evaluate(
            step_result=self.__step_result(capture=self.__request()),
            capture_store=store,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.CAPTURE_FAILED)

    def test_retains_when_capture_value_empty(self) -> None:
        """
        A capture that reached the store with an empty value retains with EMPTY_CAPTURE_VALUE (defensive guard).
        """

        store = CaptureStore()
        store.write(
            capture=Capture.model_construct(name="abc", step=1, success=True, value="", reason=None)
        )

        decision = StoreCaptureCompletionPolicy().evaluate(
            step_result=self.__step_result(capture=self.__request()),
            capture_store=store,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.EMPTY_CAPTURE_VALUE)
