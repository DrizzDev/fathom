from __future__ import annotations

import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.state import (
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    PlanMetadataKey,
    VerifyMode,
)
from fathom.constants.turn.advancement import AdvanceThreshold
from fathom.constants.turn.validation import ValidationSource
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.capture.store import CaptureStore
from fathom.core.exceptions import InvariantViolation
from fathom.core.services.criterion import CriterionObserver
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.capture import Capture, CaptureRequest
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    CriterionEvidence,
    ScreenEvidence,
    ValidationEvidence,
)
from fathom.schemas.criterion import (
    CriterionDecision,
    CriterionSource,
    CriterionVerdict,
)
from fathom.schemas.effect import ActionEffectStatus, EffectReading
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
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.validation import Validation
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator


class _StubCriterionChecker(CriterionObserver):
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
        """
        Return the next pre-staged decision, repeating the last one if exhausted.
        """

        self.calls += 1
        index = min(self.calls - 1, len(self.__decisions) - 1)
        return self.__decisions[index]


class _StubReasoner:
    """
    Deterministic reasoner that yields a configured CompletionEvidence and signal.
    """

    def __init__(
        self,
        *,
        evidence: CompletionEvidence,
        signal: SubGoalCompletionSignal,
    ) -> None:
        self.__evidence = evidence
        self.__signal = signal

    def assess_completion(self, **_: object) -> CompletionEvidence:
        """
        Return the configured evidence regardless of inputs.
        """

        return self.__evidence

    def analyze_subgoal_completion(self, **_: object) -> SubGoalCompletionSignal:
        """
        Return the configured legacy signal used by mark_current_sub_goal_complete.
        """

        return self.__signal


class SubGoalEvaluatorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`SubGoalEvaluator.evaluate` multi-signal decision matrix.
    """

    @staticmethod
    def __sub_goal(
        *,
        kind: SubGoalKind = SubGoalKind.ACTION,
        directive: Optional[ActionType] = ActionType.TAP,
        description: str = "Tap on Show results",
        criterion: Optional[str] = "Show results visible on the screen.",
        index: int = 0,
    ) -> SubGoal:
        """
        Build a SubGoal with the requested kind and directive.
        """

        return SubGoal(
            index=index,
            description=description,
            directive=directive,
            criterion=criterion,
            kind=kind,
        )

    @staticmethod
    def __evidence(
        *,
        asserted: bool,
        dispatched: bool = True,
        evolved: bool = True,
        validation: bool = False,
        criterion_observed: Optional[bool] = None,
    ) -> CompletionEvidence:
        """
        Build a CompletionEvidence with the requested signal truth table.
        """

        criterion = (
            CriterionEvidence(observed=criterion_observed)
            if criterion_observed is not None
            else None
        )
        return CompletionEvidence(
            claim=ClaimEvidence(asserted=asserted),
            action=ActionEvidence(
                dispatched=dispatched,
                executed=dispatched,
            ),
            validation=ValidationEvidence(executed=validation),
            screen=ScreenEvidence(evolved=evolved),
            criterion=criterion,
        )

    @staticmethod
    def __progress() -> EffectReading:
        """
        A scoped-progress effect reading — the typed signal the advancement policy needs to advance.
        """

        return EffectReading(live=ActionEffectStatus.PROGRESS, trial=ActionEffectStatus.PROGRESS)

    @staticmethod
    def __signal(*, flagged_complete: bool = True) -> SubGoalCompletionSignal:
        """
        Build a minimal SubGoalCompletionSignal for storage post-advance.
        """

        return SubGoalCompletionSignal(
            evidence="test",
            llm_confidence=1.0,
            keyword_match=False,
            action_executed=True,
            flagged_complete=flagged_complete,
            rationale_verified=True,
            trace_verified=False,
            screen_verified=True,
        )

    @staticmethod
    def __step_result(
        *,
        action_type: ActionType,
        success: bool = True,
        screen_changed: bool = True,
    ) -> StepResult:
        """
        Build a StepResult with the planner-emitted action_type.
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
    def __store_step_result(*, success: bool = True) -> StepResult:
        """
        Build a successful STORE step result carrying a literal capture request.
        """

        action = Action(
            action_type=ActionType.STORE,
            rationale="capture",
            capture=CaptureRequest(name="abc", subject="xyz", value="xyz"),
        )
        step = Step(action=action, step_number=0, screen_hash="pre")
        return StepResult(
            step=step,
            success=success,
            executed=success,
            duration=1,
            pre_hash="pre",
            post_hash="pre",
            screen_changed=False,
        )

    async def test_store_subgoal_advances_on_successful_capture(self) -> None:
        """
        A STORE sub-goal routes to the capture policy and advances when a successful capture exists.
        """

        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        context = self.__context(
            sub_goal=self.__sub_goal(directive=ActionType.STORE),
            evidence=self.__evidence(asserted=False),
            signal=self.__signal(flagged_complete=True),
        )
        context.catalog = CommandCatalogProvider().build()
        store = CaptureStore()
        store.write(capture=Capture.succeeded(name="abc", value="xyz", step=0))
        context.capture_store = store
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__store_step_result(),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(checker.calls, 0)

    async def test_store_subgoal_retains_when_capture_missing(self) -> None:
        """
        A STORE sub-goal retains (no capture in the store) without invoking the legacy criterion/gate path.
        """

        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        context = self.__context(
            sub_goal=self.__sub_goal(directive=ActionType.STORE),
            evidence=self.__evidence(asserted=False),
            signal=self.__signal(flagged_complete=False),
        )
        context.catalog = CommandCatalogProvider().build()
        context.capture_store = CaptureStore()
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__store_step_result(),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(checker.calls, 0)

    async def test_store_action_during_non_store_subgoal_does_not_advance_via_capture(self) -> None:
        """
        A STORE action under a non-STORE sub-goal must fall to the legacy gate, never the capture policy.
        """

        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        context = self.__context(
            sub_goal=self.__sub_goal(directive=ActionType.TAP),
            evidence=self.__evidence(asserted=False),
            signal=self.__signal(flagged_complete=False),
        )
        context.catalog = CommandCatalogProvider().build()
        store = CaptureStore()
        store.write(capture=Capture.succeeded(name="abc", value="xyz", step=0))
        context.capture_store = store
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__store_step_result(),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(checker.calls, 1)

    async def test_store_subgoal_with_non_store_action_does_not_use_capture(self) -> None:
        """
        A STORE sub-goal evaluated against a non-STORE action still uses the capture contract.
        """

        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        context = self.__context(
            sub_goal=self.__sub_goal(directive=ActionType.STORE),
            evidence=self.__evidence(asserted=False),
            signal=self.__signal(flagged_complete=False),
        )
        context.catalog = CommandCatalogProvider().build()
        context.capture_store = CaptureStore()
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(checker.calls, 0)

    @staticmethod
    def __plan_with_analysis(*, validation: Optional[Validation] = None) -> PlanResult:
        """
        Build a minimal PlanResult carrying a synthetic AnalysisResult.
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
            validation=validation,
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
        Build a minimal ScreenObservation for evaluator input.
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
        Build a CriterionDecision for stub responses.
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
        evidence: CompletionEvidence,
        signal: SubGoalCompletionSignal,
        has_more: bool = True,
    ) -> MagicMock:
        """
        Build a GraphContext mock surface with a deterministic reasoner stub.
        """

        context = MagicMock(name="GraphContext")
        context.workflow_id = "run-test"
        context.catalog = CommandCatalogProvider().build()
        context.agent_state.get_current_sub_goal.return_value = sub_goal
        context.agent_state.get_recent_effects.return_value = []
        context.agent_state.has_sub_goals.return_value = True
        context.agent_state.has_active_final_sub_goal.return_value = not has_more
        context.agent_state.mark_current_sub_goal_complete.return_value = has_more
        context.agent_state.subgoal_retain_streak = 0
        context.reasoner = _StubReasoner(evidence=evidence, signal=signal)
        return context

    async def test_action_subgoal_advances_on_claim_with_progress_effect(
        self,
    ) -> None:
        """
        ACTION sub-goal with an asserted claim, a dispatched action and a scoped-progress
        effect advances regardless of criterion observer verdict.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=True)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNSATISFIED),),
        )
        context = self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal)
        evaluator = SubGoalEvaluator(
            context=context,
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            reading=self.__progress(),
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        context.agent_state.reset_complete_deferrals.assert_called_once()

    async def test_action_subgoal_advances_when_criterion_observer_unsatisfied(
        self,
    ) -> None:
        """
        Criterion observer reporting UNSATISFIED must NOT veto a conclusive
        ACTION sub-goal advancement.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=True, criterion_observed=False)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNSATISFIED),),
        )
        context = self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal)
        evaluator = SubGoalEvaluator(
            context=context,
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            reading=self.__progress(),
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        context.agent_state.mark_current_sub_goal_complete.assert_called_once()

    async def test_validation_subgoal_retains_on_claim_asserted_without_validate(
        self,
    ) -> None:
        """
        VALIDATION sub-goal cannot advance on ``claim.asserted`` alone.
        """

        sub_goal = self.__sub_goal(
            kind=SubGoalKind.VALIDATION,
            description="Validate Jars & Containers is visible",
        )
        evidence = self.__evidence(
            asserted=True,
            dispatched=False,
            evolved=False,
            validation=False,
        )
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        context = self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal)
        evaluator = SubGoalEvaluator(
            context=context,
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.VALIDATE, screen_changed=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        context.agent_state.mark_current_sub_goal_complete.assert_not_called()

    async def test_validation_subgoal_advances_on_validate_evidence(self) -> None:
        """
        VALIDATION sub-goal advances on concrete validate action evidence.
        """

        sub_goal = self.__sub_goal(
            kind=SubGoalKind.VALIDATION,
            description="Validate Jars & Containers is visible",
        )
        evidence = self.__evidence(
            asserted=True,
            dispatched=True,
            evolved=False,
            validation=True,
        )
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal),
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(
                validation=Validation(subject="Jars & Containers", source=ValidationSource.GOAL),
            ),
            step_result=self.__step_result(action_type=ActionType.VALIDATE, screen_changed=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))

    async def test_action_sub_goal_missing_screen_evolution_retains(self) -> None:
        """
        Action sub-goal with claim+explained+dispatched but no screen change → RETAIN.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=True, dispatched=True, evolved=False)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal),
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP, screen_changed=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_action_sub_goal_missing_claim_retains(self) -> None:
        """
        Action sub-goal without LLM completion claim → RETAIN.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=False)
        signal = self.__signal(flagged_complete=False)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal),
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_final_sub_goal_advance_routes_to_verify_with_is_complete(self) -> None:
        """
        Final sub-goal ADVANCE → IS_COMPLETE + COMPLETION_REASON set.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION, description="Done")
        evidence = self.__evidence(asserted=True)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        context = self.__context(
            sub_goal=sub_goal, evidence=evidence, signal=signal, has_more=False
        )
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            reading=self.__progress(),
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(IntentStateKey.VERIFY_MODE),
            VerifyMode.PENDING_FINAL_COMMIT.value,
        )
        self.assertFalse(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertIn("All sub-goals", str(result.get(CommonStateKey.COMPLETION_REASON)))
        context.agent_state.mark_current_sub_goal_complete.assert_not_called()
        context.agent_state.clear_verification_loop.assert_called_once()
        context.agent_state.reset_complete_deferrals.assert_called_once()

    async def test_non_final_cursor_reporting_no_remaining_subgoals_fails_fast(self) -> None:
        """
        Cursor accounting drift must not be masked as final verification.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION, description="Done")
        evidence = self.__evidence(asserted=True)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        context = self.__context(
            sub_goal=sub_goal,
            evidence=evidence,
            signal=signal,
            has_more=True,
        )
        context.agent_state.has_active_final_sub_goal.return_value = False
        context.agent_state.mark_current_sub_goal_complete.return_value = False
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        with self.assertRaises(InvariantViolation):
            await evaluator.evaluate(
                plan=self.__plan_with_analysis(),
                step_result=self.__step_result(action_type=ActionType.TAP),
                accumulated=[],
                reading=self.__progress(),
                observation=self.__observation(),
            )

    async def test_step_failed_skips_evaluation(self) -> None:
        """
        Failed step → return None without invoking the gate.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=True)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal),
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP, success=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(checker.calls, 0)

    async def test_missing_observation_skips_criterion_observer_still_runs_gate(self) -> None:
        """
        No ScreenObservation → criterion observer is skipped; gate still adjudicates.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=True)
        signal = self.__signal(flagged_complete=True)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal),
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            reading=self.__progress(),
            observation=None,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(checker.calls, 0)

    async def test_criterion_satisfied_alone_cannot_advance_action_sub_goal(self) -> None:
        """
        Criterion observer SATISFIED but no claim / dispatched / progress → RETAIN.
        The criterion observer is additive; it cannot rescue a missing main signal.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(
            asserted=False,
            dispatched=False,
            evolved=False,
            criterion_observed=True,
        )
        signal = self.__signal(flagged_complete=False)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.SATISFIED),),
        )
        evaluator = SubGoalEvaluator(
            context=self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal),
            criterion_observer=checker,
        )

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_retain_backstop_exhausted_escalates_and_terminates(self) -> None:
        """
        Once the retain streak reaches the backstop limit, a would-be RETAIN escalates:
        the run terminates as STUCK instead of looping forever.
        """

        sub_goal = self.__sub_goal(kind=SubGoalKind.ACTION)
        evidence = self.__evidence(asserted=False, dispatched=True, evolved=False)
        signal = self.__signal(flagged_complete=False)
        checker = _StubCriterionChecker(
            decisions=(self.__decision(verdict=CriterionVerdict.UNCLEAR),),
        )
        context = self.__context(sub_goal=sub_goal, evidence=evidence, signal=signal)
        context.agent_state.subgoal_retain_streak = int(AdvanceThreshold.RETAIN_ESCALATION)
        evaluator = SubGoalEvaluator(context=context, criterion_observer=checker)

        result = await evaluator.evaluate(
            plan=self.__plan_with_analysis(),
            step_result=self.__step_result(action_type=ActionType.TAP, screen_changed=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertFalse(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.STUCK.value,
        )
        context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.STUCK.value
        )
