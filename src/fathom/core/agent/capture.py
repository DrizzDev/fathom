from __future__ import annotations

from fathom.constants.completion import GateOutcome, RetainReason
from fathom.core.capture.store import CaptureStore
from fathom.schemas.completion import GateDecision
from fathom.schemas.steps import StepResult


class StoreCaptureCompletionPolicy:
    """
    Decides STORE sub-goal completion from the run-owned capture store; advances only on a real captured value.
    """

    def evaluate(self, *, step_result: StepResult, capture_store: CaptureStore) -> GateDecision:
        """
        Advance only when the step ran and a successful, non-empty capture exists for the request name.
        """

        request = step_result.step.action.capture
        if request is None:
            return self.__retain(reason=RetainReason.MISSING_CAPTURE_REQUEST)

        if not step_result.success:
            return self.__retain(reason=RetainReason.STEP_EXECUTION_FAILED)

        if not capture_store.exists(name=request.name):
            return self.__retain(reason=RetainReason.MISSING_CAPTURE)

        capture = capture_store.read(name=request.name)
        if not capture.success:
            return self.__retain(reason=RetainReason.CAPTURE_FAILED)

        if capture.value is None or not capture.value.strip():
            return self.__retain(reason=RetainReason.EMPTY_CAPTURE_VALUE)

        return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

    @staticmethod
    def __retain(*, reason: RetainReason) -> GateDecision:
        """
        Build a retain decision carrying the diagnostic reason.
        """

        return GateDecision(outcome=GateOutcome.RETAIN, retain_reason=reason)
