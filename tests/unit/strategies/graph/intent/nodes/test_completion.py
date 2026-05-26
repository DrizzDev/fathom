"""
Unit pins for the criterion-first :class:`SubGoalEvaluator`.

Covers the three criterion-verdict paths (SATISFIED → advance,
UNSATISFIED → pending, UNCLEAR → implicit-completion streak fallback),
the missing-observation fallback, and the routing patches emitted on
retry vs final-verify.
"""

from __future__ import annotations

import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.reasoning import IMPLICIT_COMPLETION_THRESHOLD
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.criterion import (
    CriterionDecision,
    CriterionSource,
    CriterionVerdict,
)
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator


class _StubCriterionChecker:
    """
    Deterministic criterion checker returning a pre-staged decision per call.
    """

    def __init__(self, *, decisions: Tuple[CriterionDecision, ...]) -> None:
        self.__decisions = list(decisions)
        self.calls: int = 0

    async def check(
        self,
        *,
        workflow_id: str,
        sub_goal: SubGoal,
        observation: ScreenObservation,
    ) -> CriterionDecision:
        self.calls += 1
        index = min(self.calls - 1, len(self.__decisions) - 1)
        return self.__decisions[index]


class SubGoalEvaluatorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`SubGoalEvaluator.evaluate` criterion-first decision matrix.
    """

    @staticmethod
    def __sub_goal(
        *,
        directive: Optional[ActionType] = ActionType.TAP,
        description: str = "Tap on Show results",
        criterion: Optional[str] = "Show results visible on the screen.",
        index: int = 0,
        streak: int = 0,
    ) -> SubGoal:
        """
        Build a :class:`SubGoal` with the requested directive and criterion.
        """

        sub_goal = SubGoal(
            index=index,
            description=description,
            directive=directive,
            criterion=criterion,
        )
        sub_goal.completion_claim_streak = streak
        return sub_goal

    @staticmethod
    def __signal(
        *,
        flagged_complete: bool,
        action_executed: bool = True,
        screen_verified: bool = True,
        rationale_verified: bool = True,
    ) -> SubGoalCompletionSignal:
        """
        Build a :class:`SubGoalCompletionSignal` with the requested fields.
        """

        return SubGoalCompletionSignal(
            evidence="test",
            llm_confidence=1.0,
            keyword_match=False,
            action_executed=action_executed,
            flagged_complete=flagged_complete,
            rationale_verified=rationale_verified,
            trace_verified=False,
            screen_verified=screen_verified,
        )

    @staticmethod
    def __step_result(
        *,
        action_type: ActionType,
        success: bool = True,
        screen_changed: bool = True,
    ) -> StepResult:
        """
        Build a :class:`StepResult` with the planner-emitted action_type.
        """

        action = Action(
            action_type=action_type,
            target="t",
            rationale="r",
            confidence=1.0,
        )
        step = Step(action=action, step_number=0, screen_hash="pre")
        return StepResult(
            step=step,
            success=success,
            duration=10,
            screen_changed=screen_changed,
            pre_hash="pre",
            post_hash="post" if screen_changed else "pre",
        )

    @staticmethod
    def __plan_with_analysis() -> PlanResult:
        """
        Build a minimal :class:`PlanResult` carrying a synthetic analysis.
        """

        analysis = AnalysisResult(
            action=Action(
                action_type=ActionType.TAP,
                target="t",
                rationale="r",
                confidence=1.0,
            ),
            reasoning="r",
            screen_description="s",
            metadata={"tool_args": {}},
        )
        return PlanResult(
            step=None,
            is_complete=False,
            reason="t",
            metadata={PlanMetadataKey.ANALYSIS.value: analysis},
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Build a minimal :class:`ScreenObservation` for evaluator input.
        """

        element = PerceivedElement(
            identifier="e0",
            bounds=Bounds(x=0, y=0, width=10, height=10),
            source=ElementSource.XML,
            role=ElementRole.TEXT,
            confidence=1.0,
            text="dummy",
            tappable=False,
        )
        return ScreenObservation(
            activity="com.test.app",
            elements=(element,),
            hashes=ScreenHashBundle(
                visual_hash="vh0",
                xml_hash="0000000000000000",
                interaction_hash="0000000000000000",
            ),
            overlays=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )

    @staticmethod
    def __decision(
        *,
        verdict: CriterionVerdict,
        source: CriterionSource = CriterionSource.SYMBOLIC,
        confidence: float = 0.9,
    ) -> CriterionDecision:
        """
        Build a :class:`CriterionDecision` for stub responses.
        """

        return CriterionDecision(
            verdict=verdict,
            source=source,
            confidence=confidence,
            evidence=(),
            notes=None,
        )

    def __context(
        self,
        *,
        sub_goal: SubGoal,
        signal: SubGoalCompletionSignal,
        has_more: bool = True,
    ) -> MagicMock:
        """
        Build a :class:`GraphContext` mock surface.
        """

        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.agent_state.get_current_sub_goal.return_value = sub_goal
        context.agent_state.has_sub_goals.return_value = True
        context.agent_state.last_delta_score = None
        context.agent_state.mark_current_sub_goal_complete.return_value = has_more
        context.reasoner.analyze_subgoal_completion.return_value = signal
        return context

    async def test_satisfied_verdict_advances_sub_goal(self) -> None:
        """
        Criterion SATISFIED → advance with SHOULD_RETRY for next sub-goal.
        """

        sub_goal = self.__sub_goal(directive=ActionType.TAP)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(checker.calls, 1)

    async def test_satisfied_verdict_advances_even_on_directive_divergence(self) -> None:
        """
        Criterion SATISFIED with emit ≠ directive: still advance (directive is a hint, not a gate).
        """

        sub_goal = self.__sub_goal(directive=ActionType.SWIPE_DOWN)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.SWIPE_UP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))

    async def test_satisfied_verdict_on_final_sub_goal_routes_to_verify(self) -> None:
        """
        Final sub-goal SATISFIED → IS_COMPLETE + COMPLETION_REASON set.
        """

        sub_goal = self.__sub_goal(directive=ActionType.COMPLETE, description="Done")
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal, has_more=False),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertIn("All sub-goals", str(result.get(CommonStateKey.COMPLETION_REASON)))

    async def test_unsatisfied_verdict_keeps_sub_goal_pending_and_resets_streak(self) -> None:
        """
        Criterion UNSATISFIED → return None and reset completion_claim_streak.
        """

        sub_goal = self.__sub_goal(directive=ActionType.TAP, streak=1)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNSATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(sub_goal.completion_claim_streak, 0)

    async def test_unclear_verdict_with_completion_claim_increments_streak(self) -> None:
        """
        Criterion UNCLEAR + completion-shaped emit + flagged_complete → streak grows
        but stops short of advancing on the first claim.
        """

        sub_goal = self.__sub_goal(directive=ActionType.TAP, streak=0)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(sub_goal.completion_claim_streak, 1)

    async def test_unclear_verdict_advances_after_threshold(self) -> None:
        """
        Criterion UNCLEAR + sustained completion claim crosses IMPLICIT_COMPLETION_THRESHOLD → advance.
        """

        sub_goal = self.__sub_goal(
            directive=ActionType.TAP,
            streak=IMPLICIT_COMPLETION_THRESHOLD - 1,
        )
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(sub_goal.completion_claim_streak, 0)

    async def test_unclear_verdict_with_non_completion_emit_does_not_grow_streak(
        self,
    ) -> None:
        """
        UNCLEAR + non-completion emit (no streak preconditions) → no advance, streak stays at 0.
        """

        sub_goal = self.__sub_goal(directive=ActionType.TAP, streak=0)
        signal = self.__signal(flagged_complete=False)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(sub_goal.completion_claim_streak, 0)

    async def test_fraud_pattern_unsatisfied_does_not_advance_via_streak(self) -> None:
        """
        Original Swiggy fraud pattern: planner emits validate without doing the
        tap. The criterion check observes that the post-state is not present,
        returns UNSATISFIED, and the gate does NOT advance regardless of
        ``flagged_complete``. Streak cannot promote an UNSATISFIED verdict.
        """

        sub_goal = self.__sub_goal(
            directive=ActionType.TAP,
            streak=IMPLICIT_COMPLETION_THRESHOLD,
        )
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(
                self.__decision(verdict=CriterionVerdict.UNSATISFIED),
                self.__decision(verdict=CriterionVerdict.UNSATISFIED),
                self.__decision(verdict=CriterionVerdict.UNSATISFIED),
            ),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE, screen_changed=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(sub_goal.completion_claim_streak, 0)

    async def test_missing_observation_falls_back_to_streak(self) -> None:
        """
        When OBSERVE did not produce a typed ScreenObservation (capture failure
        etc.) the evaluator cannot run the criterion check and must use the
        legacy streak guard. With a completion claim, streak grows; without, no
        advance.
        """

        sub_goal = self.__sub_goal(directive=ActionType.TAP, streak=0)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE),
            accumulated=[],
            observation=None,
        )

        self.assertIsNone(result)
        self.assertEqual(sub_goal.completion_claim_streak, 1)
        self.assertEqual(checker.calls, 0)

    async def test_skips_when_step_failed(self) -> None:
        """
        Failed step → evaluator returns None without invoking the checker.
        """

        sub_goal = self.__sub_goal(directive=ActionType.TAP)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, signal=signal),
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP, success=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(checker.calls, 0)

    async def test_skips_when_no_active_sub_goal(self) -> None:
        """
        No active sub-goal → evaluator returns None without invoking the checker.
        """

        signal = self.__signal(flagged_complete=True)
        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.agent_state.get_current_sub_goal.return_value = None
        context.agent_state.has_sub_goals.return_value = False
        context.reasoner.analyze_subgoal_completion.return_value = signal
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=context,
            criterion_checker=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(checker.calls, 0)


if __name__ == "__main__":
    unittest.main()
