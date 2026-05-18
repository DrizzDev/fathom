from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoalKind
from fathom.schemas.tasks import TaskStatus
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator


class SubGoalEvaluatorClassifyTest(unittest.TestCase):
    """
    Pins :meth:`SubGoalEvaluator.classify`.

    The classifier decides whether a sub-goal is a VALIDATION or ACTION
    step. Two signals can flip a sub-goal to VALIDATION: an explicit
    ``event_type='validation'`` on the step, or a validation keyword
    (verify / confirm / etc.) in the sub-goal description. Anything
    else classifies as an action step.
    """

    @staticmethod
    def __step_result(*, event_type: str = "action") -> StepResult:
        """
        :class:`StepResult` fixture parameterised on step ``event_type``
        so the explicit-validation path can be driven by passing
        ``event_type='validation'``.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="x",
            rationale="t",
            confidence=1.0,
        )
        step = Step(
            action=action,
            event_type=event_type,
            condition="x",
            screen_hash="0" * 16,
            step_number=1,
        )
        return StepResult(
            step=step,
            success=True,
            pre_hash="0" * 16,
            post_hash="0" * 16,
            screen_changed=True,
            duration=10,
            generalized_target="x",
            is_positional=False,
        )

    def test_validation_event_type_routes_to_validation(self) -> None:
        """
        A step whose event_type is 'validation' classifies as a validation sub-goal.
        """

        self.assertEqual(
            SubGoalEvaluator.classify(
                description="Tap on Submit",
                step_result=self.__step_result(event_type="validation"),
            ),
            SubGoalKind.VALIDATION,
        )

    def test_validation_keyword_in_description_routes_to_validation(self) -> None:
        """
        A description carrying a validation keyword classifies as a validation sub-goal.
        """

        self.assertEqual(
            SubGoalEvaluator.classify(
                description="Verify the order summary is displayed",
                step_result=self.__step_result(),
            ),
            SubGoalKind.VALIDATION,
        )

    def test_plain_action_description_classifies_as_action(self) -> None:
        """
        An action description without validation keywords classifies as an action sub-goal.
        """

        self.assertEqual(
            SubGoalEvaluator.classify(
                description="Tap on Submit",
                step_result=self.__step_result(),
            ),
            SubGoalKind.ACTION,
        )


class SubGoalEvaluatorFloorTest(unittest.TestCase):
    """
    Pins :meth:`SubGoalEvaluator.floor` completion-floor decisions.

    The floor is a "refuse to advance" guard sitting between the model's
    self-reported task status and the supervisor's observed outcome. It
    must not advance a sub-goal when the two signals disagree, when the
    model reports BLOCKED/NOT_MET, or when no effect was observed.
    Validation-kind sub-goals bypass the floor entirely so a verifier
    step can advance without producing a screen effect.
    """

    @staticmethod
    def __action() -> Action:
        """
        :class:`Action` fixture. Its content is irrelevant to the floor
        logic; it exists only so :class:`AnalysisResult` validates.
        """

        return Action(
            action_type=ActionType.TAP,
            target="x",
            rationale="t",
            confidence=1.0,
        )

    def __analysis(self, *, status: TaskStatus) -> AnalysisResult:
        """
        :class:`AnalysisResult` fixture parameterised on ``task_status``
        so each test can drive the four model-reported states (MET,
        PARTIAL, NOT_MET, BLOCKED).
        """

        return AnalysisResult(
            action=self.__action(),
            reasoning="r",
            screen_description="s",
            task_status=status,
        )

    @classmethod
    def __outcome(cls, *, status: OutcomeStatus) -> ActionOutcome:
        """
        :class:`ActionOutcome` fixture parameterised on observed
        ``status`` so the floor's EFFECTIVE / NO_EFFECT branches can be
        driven independently. ``action`` and ``before`` are required by
        the schema; the floor does not read them.
        """

        return ActionOutcome(
            status=status,
            reason="x",
            action=cls.__action(),
            before=cls.__observation(),
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        :class:`ScreenObservation` placeholder satisfying the
        :class:`ActionOutcome` ``before`` requirement. The floor never
        inspects the observation.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="a" * 16,
                interaction_hash="b" * 16,
            ),
            elements=(),
            keyboard=KeyboardObservation(visible=False),
        )

    def test_validation_kind_never_invokes_floor(self) -> None:
        """
        Validation sub-goals bypass the completion floor entirely.
        """

        self.assertIsNone(
            SubGoalEvaluator.floor(
                kind=SubGoalKind.VALIDATION,
                analysis=self.__analysis(status=TaskStatus.NOT_MET),
                outcome=self.__outcome(status=OutcomeStatus.NO_EFFECT),
            ),
        )

    def test_task_blocked_returns_reason(self) -> None:
        """
        A BLOCKED task_status surfaces the supervision-handoff reason.
        """

        reason = SubGoalEvaluator.floor(
            kind=SubGoalKind.ACTION,
            analysis=self.__analysis(status=TaskStatus.BLOCKED),
            outcome=self.__outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("BLOCKED", reason)

    def test_task_not_met_returns_reason(self) -> None:
        """
        A NOT_MET task_status refuses to advance.
        """

        reason = SubGoalEvaluator.floor(
            kind=SubGoalKind.ACTION,
            analysis=self.__analysis(status=TaskStatus.NOT_MET),
            outcome=self.__outcome(status=OutcomeStatus.EFFECTIVE),
        )

        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("NOT_MET", reason)

    def test_task_met_with_effective_outcome_returns_none(self) -> None:
        """
        MET status combined with an EFFECTIVE outcome clears the floor.
        """

        self.assertIsNone(
            SubGoalEvaluator.floor(
                kind=SubGoalKind.ACTION,
                analysis=self.__analysis(status=TaskStatus.MET),
                outcome=self.__outcome(status=OutcomeStatus.EFFECTIVE),
            ),
        )

    def test_task_met_with_no_effect_returns_reason(self) -> None:
        """
        MET status with NO_EFFECT outcome is treated as a self-report mismatch.
        """

        reason = SubGoalEvaluator.floor(
            kind=SubGoalKind.ACTION,
            analysis=self.__analysis(status=TaskStatus.MET),
            outcome=self.__outcome(status=OutcomeStatus.NO_EFFECT),
        )

        self.assertIsNotNone(reason)

    def test_no_task_status_with_no_effect_returns_reason(self) -> None:
        """
        Missing task_status plus NO_EFFECT outcome refuses to advance.
        """

        analysis = AnalysisResult(
            action=self.__action(),
            reasoning="r",
            screen_description="s",
        )

        reason = SubGoalEvaluator.floor(
            kind=SubGoalKind.ACTION,
            analysis=analysis,
            outcome=self.__outcome(status=OutcomeStatus.NO_EFFECT),
        )

        self.assertIsNotNone(reason)

    def test_no_task_status_with_effective_outcome_returns_none(self) -> None:
        """
        Missing task_status plus EFFECTIVE outcome allows advancement.
        """

        analysis = AnalysisResult(
            action=self.__action(),
            reasoning="r",
            screen_description="s",
        )

        self.assertIsNone(
            SubGoalEvaluator.floor(
                kind=SubGoalKind.ACTION,
                analysis=analysis,
                outcome=self.__outcome(status=OutcomeStatus.EFFECTIVE),
            ),
        )
