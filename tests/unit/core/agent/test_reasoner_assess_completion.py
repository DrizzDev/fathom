"""
Unit pins for :meth:`Reasoner.assess_completion`.

The reasoner converts a planner turn into a typed CompletionEvidence bundle.
These tests verify the 4 mechanical signals (claim/action/screen) and the
optional criterion fold-in match main-branch behavior and the four bug-replay
scenarios.
"""

from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants import ActionType
from fathom.core.agent.reasoner import Reasoner
from fathom.schemas.actions import Action
from fathom.schemas.criterion import (
    CriterionDecision,
    CriterionSource,
    CriterionVerdict,
)
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

        return Reasoner(intent=intent)

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
            analysis=self.__analysis(action_type=ActionType.TAP),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertTrue(evidence.action.dispatched)

    def test_screen_evolved_via_delta_score_above_floor(self) -> None:
        """
        delta_score above the meaningful-delta floor → screen.evolved=True even
        when screen_changed=False. Critical for counter-style mutations where
        the screen layout is mostly identical but a text value changed.
        """

        evidence = self.__reasoner().assess_completion(
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
            analysis=self.__analysis(),
            sub_goal=self.__sub_goal(),
            screen_changed=True,
        )

        self.assertIsNone(evidence.criterion)

    def test_iahtk_replay_all_four_signals_present(self) -> None:
        """
        IahTk replay at the Reasoner level: successful tap with explicit
        completion claim + justification + screen evolution produces all 4
        signals true. The criterion can be observed False (post-tap screen
        no longer contains the criterion tokens) and the evidence remains
        valid for ACTION sub-goal advancement.
        """

        evidence = self.__reasoner().assess_completion(
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
