from __future__ import annotations

import logging
import unittest
from typing import Optional

from fathom.constants import ActionType
from fathom.core.agent.opener import OpenerSignalPolicy
from fathom.core.agent.reasoner import Reasoner
from fathom.schemas.actions import Action
from fathom.schemas.criterion import (
    CriterionDecision,
    CriterionSource,
    CriterionVerdict,
)
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.results import AnalysisResult
from fathom.schemas.subgoal import SubGoal, SubGoalKind


class ReasonerAssessCompletionTest(unittest.TestCase):
    """
    Pins assess_completion against the documented signal computation.
    """

    @staticmethod
    def __reasoner(intent: str = "open meesho and find Jars & containers") -> Reasoner:
        """
        Build a Reasoner with a representative intent string.
        """

        return Reasoner(intent=intent, opener_policy=OpenerSignalPolicy())

    @staticmethod
    def __sub_goal(
        *,
        description: str = "Tap on Submit button",
        kind: SubGoalKind = SubGoalKind.ACTION,
    ) -> SubGoal:
        """
        Build a sub-goal fixture.
        """

        return SubGoal(index=0, description=description, kind=kind, directive=ActionType.TAP)

    @staticmethod
    def __analysis(
        *,
        is_sub_goal_complete: bool = False,
        is_goal_complete: bool = False,
        action_type: ActionType = ActionType.TAP,
        reasoning: str = "Submit tapped; new screen visible.",
        subgoal_completion_reason: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Build an AnalysisResult fixture with the requested flags.
        """

        return AnalysisResult(
            action=Action(
                action_type=action_type,
                target="t",
                rationale="r",
                confidence=1.0,
            ),
            reasoning=reasoning,
            screen_description="post-action screen",
            is_sub_goal_complete=is_sub_goal_complete,
            is_goal_complete=is_goal_complete,
            subgoal_completion_reason=subgoal_completion_reason,
            metadata={"tool_args": {}},
        )

    @staticmethod
    def __decision(verdict: CriterionVerdict) -> CriterionDecision:
        """
        Build a CriterionDecision fixture.
        """

        return CriterionDecision(
            verdict=verdict,
            source=CriterionSource.SYMBOLIC,
            confidence=0.9,
            evidence=(),
            notes=None,
        )

    def test_claim_asserted_from_sub_goal_complete_flag(self) -> None:
        """
        is_sub_goal_complete=True → claim.asserted=True.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(is_sub_goal_complete=True),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertTrue(evidence.claim.asserted)

    def test_claim_asserted_from_intent_complete_flag(self) -> None:
        """
        is_goal_complete=True → claim.asserted=True.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(is_goal_complete=True),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertTrue(evidence.claim.asserted)

    def test_claim_asserted_from_complete_action_type(self) -> None:
        """
        action_type=COMPLETE → claim.asserted=True regardless of flags.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(action_type=ActionType.COMPLETE),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertTrue(evidence.claim.asserted)

    def test_claim_not_asserted_when_all_flags_false(self) -> None:
        """
        No completion flag set → claim.asserted=False.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertFalse(evidence.claim.asserted)

    def test_claim_justified_when_explicit_reason_present(self) -> None:
        """
        Explicit subgoal_completion_reason with asserted claim → claim.justified=True.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(
                is_sub_goal_complete=True,
                subgoal_completion_reason="New screen shows confirmation toast.",
            ),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertTrue(evidence.claim.justified)

    def test_action_dispatched_for_tap_action(self) -> None:
        """
        Dispatch-able action type → action.dispatched=True.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(action_type=ActionType.TAP),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertTrue(evidence.action.dispatched)

    def test_action_dispatched_for_directional_swipe_actions(self) -> None:
        """
        Directional swipe variants are executable gestures → action.dispatched=True.
        """

        for action_type in (
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
        ):
            with self.subTest(action_type=action_type):
                evidence = self.__reasoner().assess_completion(
                    execution_success=True,
                    analysis=self.__analysis(action_type=action_type),
                    sub_goal=self.__sub_goal(),
                    screen_changed=True,
                )

                self.assertTrue(evidence.action.dispatched)

    def test_failed_command_is_dispatched_but_not_executed(self) -> None:
        """
        A TAP the device reported as failed stays dispatched (a real type) but executed=False.
        """

        evidence = self.__reasoner().assess_completion(
            analysis=self.__analysis(action_type=ActionType.TAP),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            execution_success=False,
        )

        self.assertTrue(evidence.action.dispatched)
        self.assertFalse(evidence.action.executed)

    def test_successful_back_is_executed_but_not_dispatched(self) -> None:
        """
        A BACK that ran successfully is executed=True even though BACK is not a dispatched type today.
        """

        evidence = self.__reasoner().assess_completion(
            analysis=self.__analysis(action_type=ActionType.BACK),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            execution_success=True,
        )

        self.assertFalse(evidence.action.dispatched)
        self.assertTrue(evidence.action.executed)

    def test_opening_sub_goal_completes_for_next_phase_actions(self) -> None:
        """
        Next-phase action types complete opener sub-goals when reasoning confirms follow-up work.
        """

        for action_type in (
            ActionType.SWIPE,
            ActionType.SCROLL,
            ActionType.VALIDATE,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
        ):
            with self.subTest(action_type=action_type):
                signal = self.__reasoner().analyze_completion(
                    analysis=self.__analysis(
                        action_type=action_type,
                        reasoning="Swipe and check the main content after app launch.",
                    ),
                    current_sub_goal="Open Tata 1mg app",
                )

                self.assertTrue(signal.success_indicator)
                self.assertIn(action_type.value, signal.evidence)

    def test_screen_evolved_via_delta_score_above_floor(self) -> None:
        """
        delta_score above the meaningful-delta floor → screen.evolved=True even
        when screen_changed=False. Critical for counter-style mutations where
        the screen layout is mostly identical but a text value changed.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=False,
            delta_score=1.0,
        )

        self.assertTrue(evidence.screen.evolved)

    def test_screen_evolved_false_when_both_signals_negative(self) -> None:
        """
        No screen_changed, delta_score below floor → screen.evolved=False.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=False,
            delta_score=0.0,
        )

        self.assertFalse(evidence.screen.evolved)

    def test_criterion_evidence_satisfied_maps_to_observed_true(self) -> None:
        """
        CriterionDecision verdict=SATISFIED → criterion.observed=True.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            criterion_decision=self.__decision(verdict=CriterionVerdict.SATISFIED),
        )

        self.assertIsNotNone(evidence.criterion)
        assert evidence.criterion is not None
        self.assertTrue(evidence.criterion.observed)

    def test_criterion_evidence_unsatisfied_maps_to_observed_false(self) -> None:
        """
        CriterionDecision verdict=UNSATISFIED → criterion.observed=False.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            criterion_decision=self.__decision(verdict=CriterionVerdict.UNSATISFIED),
        )

        self.assertIsNotNone(evidence.criterion)
        assert evidence.criterion is not None
        self.assertFalse(evidence.criterion.observed)

    def test_criterion_evidence_absent_when_decision_not_supplied(self) -> None:
        """
        Optional criterion_decision omitted → criterion field is None.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertIsNone(evidence.criterion)

    @staticmethod
    def __effect(status: ActionEffectStatus) -> ActionEffect:
        """
        Build a minimal ActionEffect fixture; only the status drives the veto path under test.
        """

        return ActionEffect(
            status=status,
            visual_progress=0.0,
            phash_distance=0,
        )

    def test_no_progress_effect_vetoes_screen_evolved_when_screen_changed_true(self) -> None:
        """
        ActionEffectStatus.NO_PROGRESS must veto screen.evolved even when the high-sensitivity flag reports True.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            effect=self.__effect(status=ActionEffectStatus.NO_PROGRESS),
        )

        self.assertFalse(evidence.screen.evolved)

    def test_no_progress_effect_vetoes_screen_evolved_when_delta_above_floor(self) -> None:
        """
        The veto must also block the magnitude path so a delta above the meaningful floor cannot revive evolved.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            delta_score=0.95,
            effect=self.__effect(status=ActionEffectStatus.NO_PROGRESS),
        )

        self.assertFalse(evidence.screen.evolved)

    def test_progress_effect_does_not_block_screen_evolved(self) -> None:
        """
        ActionEffectStatus.PROGRESS must never tighten the gate; the reasoner still trusts screen_changed.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            effect=self.__effect(status=ActionEffectStatus.PROGRESS),
        )

        self.assertTrue(evidence.screen.evolved)

    def test_uncertain_effect_does_not_block_screen_evolved(self) -> None:
        """
        ActionEffectStatus.UNCERTAIN means signals disagreed; vetoing it would produce false negatives.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
            effect=self.__effect(status=ActionEffectStatus.UNCERTAIN),
        )

        self.assertTrue(evidence.screen.evolved)

    def test_dispatched_tap_with_justified_claim_and_screen_change_produces_all_action_signals(
        self,
    ) -> None:
        """
        A dispatched TAP with a justified completion claim and an evolved
        screen yields all four ACTION-level evidence signals regardless of
        criterion verdict.
        """

        evidence = self.__reasoner().assess_completion(
            execution_success=True,
            analysis=self.__analysis(
                is_sub_goal_complete=True,
                subgoal_completion_reason="Home screen reached with Dwarka location.",
                action_type=ActionType.TAP,
            ),
            sub_goal=self.__sub_goal(
                description="Tap on Confirm location and continue button",
            ),
            screen_changed=True,
            criterion_decision=self.__decision(verdict=CriterionVerdict.UNSATISFIED),
        )

        self.assertTrue(evidence.claim.asserted)
        self.assertTrue(evidence.claim.justified)
        self.assertTrue(evidence.action.dispatched)
        self.assertTrue(evidence.screen.evolved)
        assert evidence.criterion is not None
        self.assertFalse(evidence.criterion.observed)


class ReasonerLateralCreditObservedTest(unittest.TestCase):
    """
    Pins completion.lateral_credit.observed: fires only when the model
    asserts completion against an active sub-goal whose description does
    not actually share enough text with the rationale to justify the claim.
    """

    @staticmethod
    def __reasoner() -> Reasoner:
        """
        Build a reasoner with a representative intent string.
        """

        return Reasoner(
            intent="open meesho and find Jars & containers",
            opener_policy=OpenerSignalPolicy(),
        )

    @staticmethod
    def __sub_goal(*, description: str) -> SubGoal:
        """
        Build a sub-goal carrying the supplied description as the active target.
        """

        return SubGoal(
            index=2,
            description=description,
            kind=SubGoalKind.ACTION,
            directive=ActionType.TAP,
        )

    @staticmethod
    def __analysis(
        *,
        is_sub_goal_complete: bool,
        reasoning: str,
        subgoal_completion_reason: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Build an analysis result with the supplied rationale and completion claim flag.
        """

        return AnalysisResult(
            action=Action(
                action_type=ActionType.TAP,
                target="t",
                rationale="r",
                confidence=1.0,
            ),
            reasoning=reasoning,
            screen_description="",
            is_sub_goal_complete=is_sub_goal_complete,
            is_goal_complete=False,
            subgoal_completion_reason=subgoal_completion_reason,
            metadata={"tool_args": {}},
        )

    def test_observed_when_asserted_with_unrelated_rationale(self) -> None:
        """
        Asserted claim against a sub-goal whose description shares almost nothing with the rationale fires the event.
        """

        sub_goal = self.__sub_goal(description="Apply the 10% off coupon at checkout")
        analysis = self.__analysis(
            is_sub_goal_complete=True,
            reasoning="Pizza delivered.",
            subgoal_completion_reason="Pizza delivered.",
        )

        with self.assertLogs("fathom.core.agent.reasoner", level=logging.INFO) as captured:
            self.__reasoner().assess_completion(
                execution_success=True,
                analysis=analysis,
                sub_goal=sub_goal,
                screen_changed=True,
            )

        records = [
            record
            for record in captured.records
            if record.__dict__.get("event") == "completion.lateral_credit.observed"
        ]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.__dict__["sub_goal.index"], sub_goal.index)
        self.assertEqual(record.__dict__["sub_goal.description"], sub_goal.description[:120])

    def test_not_observed_when_claim_not_asserted(self) -> None:
        """
        Without an asserted claim the lateral-credit event is suppressed regardless of similarity.
        """

        sub_goal = self.__sub_goal(description="Apply the 10% off coupon at checkout")
        analysis = self.__analysis(
            is_sub_goal_complete=False,
            reasoning="Some unrelated narration.",
        )

        with self.assertLogs("fathom.core.agent.reasoner", level=logging.INFO) as captured:
            self.__reasoner().assess_completion(
                execution_success=True,
                analysis=analysis,
                sub_goal=sub_goal,
                screen_changed=True,
            )
            logging.getLogger("fathom.core.agent.reasoner").info("sentinel")

        records = [
            record
            for record in captured.records
            if record.__dict__.get("event") == "completion.lateral_credit.observed"
        ]
        self.assertEqual(records, [])

    def test_not_observed_when_rationale_aligned_with_sub_goal(self) -> None:
        """
        Strong textual alignment between sub-goal description and rationale keeps the event quiet.
        """

        description = "Tap on Submit button"
        sub_goal = self.__sub_goal(description=description)
        analysis = self.__analysis(
            is_sub_goal_complete=True,
            reasoning=description,
            subgoal_completion_reason=description,
        )

        with self.assertLogs("fathom.core.agent.reasoner", level=logging.INFO) as captured:
            self.__reasoner().assess_completion(
                execution_success=True,
                analysis=analysis,
                sub_goal=sub_goal,
                screen_changed=True,
            )
            logging.getLogger("fathom.core.agent.reasoner").info("sentinel")

        records = [
            record
            for record in captured.records
            if record.__dict__.get("event") == "completion.lateral_credit.observed"
        ]
        self.assertEqual(records, [])
