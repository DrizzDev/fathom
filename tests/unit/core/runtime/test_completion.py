from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.completion import CompletionEvidence
from fathom.core.runtime.completion import CompletionService
from fathom.schemas.actions import Action
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.tasks import (
    ExecutionTask,
    ExecutionTaskState,
    TaskAttemptState,
    TaskStatus,
)


def _hashes() -> ScreenHashBundle:
    """
    Return a minimal screen hash bundle for fixture observations.
    """

    return ScreenHashBundle(
        visual_hash="0" * 16,
        xml_hash="a" * 16,
        interaction_hash="b" * 16,
    )


def _observation() -> ScreenObservation:
    """
    Return a minimal screen observation for fixture outcomes.
    """

    return ScreenObservation(
        activity="bundl.swiggy.production",
        hashes=_hashes(),
        elements=(),
        keyboard=KeyboardObservation(visible=False),
    )


def _action() -> Action:
    """
    Return a minimal tap action for fixture outcomes.
    """

    return Action(
        confidence=0.9,
        action_type=ActionType.TAP,
        target="Continue button",
        rationale="fixture action",
    )


def _outcome(*, status: OutcomeStatus) -> ActionOutcome:
    """
    Build an ActionOutcome with the supplied status.
    """

    return ActionOutcome(
        status=status,
        action=_action(),
        before=_observation(),
        after=_observation(),
        diff=None,
        reason=f"fixture outcome status={status.value}",
    )


def _task(
    *,
    state: ExecutionTaskState = ExecutionTaskState.ACTIVE,
    count: int = 0,
    limit: int = 5,
) -> ExecutionTask:
    """
    Build an ExecutionTask with the supplied attempt accounting.
    """

    return ExecutionTask(
        identifier="task:0",
        objective="Tap Continue",
        criterion="Cart screen is visible.",
        state=state,
        attempts=TaskAttemptState(count=count, limit=limit),
    )


class CompletionServiceTest(unittest.TestCase):
    """
    Pins for the runtime CompletionService verdict fusion.
    """

    def test_succeeds_when_status_met_and_outcome_effective(self) -> None:
        """
        TaskStatus.MET combined with EFFECTIVE outcome must advance the task.
        """

        verdict = CompletionService().evaluate(
            task=_task(),
            status=TaskStatus.MET,
            outcome=_outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertIsInstance(verdict, CompletionVerdict)
        self.assertTrue(verdict.complete)
        self.assertEqual(verdict.next_state, ExecutionTaskState.SUCCEEDED)
        self.assertEqual(verdict.missing, [])

    def test_active_when_status_met_but_outcome_not_effective(self) -> None:
        """
        TaskStatus.MET without an EFFECTIVE outcome must keep the task ACTIVE.
        """

        verdict = CompletionService().evaluate(
            task=_task(),
            status=TaskStatus.MET,
            outcome=_outcome(status=OutcomeStatus.NO_EFFECT),
        )

        self.assertFalse(verdict.complete)
        self.assertEqual(verdict.next_state, ExecutionTaskState.ACTIVE)
        self.assertIn(CompletionEvidence.OUTCOME_EFFECTIVE, verdict.missing)

    def test_blocked_when_status_blocked(self) -> None:
        """
        TaskStatus.BLOCKED must mark the task BLOCKED regardless of outcome.
        """

        verdict = CompletionService().evaluate(
            task=_task(),
            status=TaskStatus.BLOCKED,
            outcome=_outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertFalse(verdict.complete)
        self.assertEqual(verdict.next_state, ExecutionTaskState.BLOCKED)
        self.assertIn(CompletionEvidence.OUTCOME_BLOCKED, verdict.missing)

    def test_failed_when_attempt_budget_exhausted(self) -> None:
        """
        An over-budget task must transition to FAILED regardless of status.
        """

        verdict = CompletionService().evaluate(
            task=_task(count=5, limit=5),
            status=TaskStatus.MET,
            outcome=_outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertFalse(verdict.complete)
        self.assertEqual(verdict.next_state, ExecutionTaskState.FAILED)
        self.assertIn(CompletionEvidence.BUDGET_EXHAUSTED, verdict.missing)

    def test_active_when_no_status_reported(self) -> None:
        """
        Missing task_status must keep the task ACTIVE with both evidence dimensions missing.
        """

        verdict = CompletionService().evaluate(
            task=_task(),
            status=None,
            outcome=None,
        )

        self.assertFalse(verdict.complete)
        self.assertEqual(verdict.next_state, ExecutionTaskState.ACTIVE)
        self.assertIn(CompletionEvidence.TASK_STATUS_MET, verdict.missing)
        self.assertIn(CompletionEvidence.OUTCOME_EFFECTIVE, verdict.missing)
