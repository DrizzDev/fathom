"""
Domain-level unit pins for :class:`CompletionGate`.

These tests exercise the gate in isolation against the per-kind threshold
policy (ACTION 3-of-3, VALIDATION asserted-short-circuit + 2-of-3 fallback)
and the diagnostic :class:`RetainReason` mapping.
"""

from __future__ import annotations

import unittest

from fathom.constants.completion import GateOutcome, RetainReason
from fathom.core.agent.completion import CompletionGate
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    CriterionEvidence,
    ScreenEvidence,
)
from fathom.schemas.subgoal import SubGoal, SubGoalKind


class CompletionGateActionTest(unittest.TestCase):
    """
    Pins :meth:`CompletionGate.adjudicate` for ACTION sub-goals (3-of-3).
    """

    @staticmethod
    def __sub_goal() -> SubGoal:
        """
        Build an ACTION sub-goal fixture.
        """

        return SubGoal(index=0, description="Tap on Submit", kind=SubGoalKind.ACTION)

    @staticmethod
    def __evidence(
        *,
        asserted: bool,
        justified: bool,
        dispatched: bool,
        evolved: bool,
    ) -> CompletionEvidence:
        """
        Build a CompletionEvidence with the requested signal truth table.
        """

        return CompletionEvidence(
            claim=ClaimEvidence(asserted=asserted, justified=justified),
            action=ActionEvidence(dispatched=dispatched),
            screen=ScreenEvidence(evolved=evolved),
        )

    def test_all_signals_advance(self) -> None:
        """
        Claim asserted + justified + action dispatched + screen evolved → ADVANCE.
        """

        decision = CompletionGate().adjudicate(
            evidence=self.__evidence(
                asserted=True, justified=True, dispatched=True, evolved=True
            ),
            sub_goal=self.__sub_goal(),
        )

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)
        self.assertIsNone(decision.retain_reason)

    def test_missing_claim_retains_with_diagnostic(self) -> None:
        """
        Claim not asserted → RETAIN with MISSING_CLAIM.
        """

        decision = CompletionGate().adjudicate(
            evidence=self.__evidence(
                asserted=False, justified=True, dispatched=True, evolved=True
            ),
            sub_goal=self.__sub_goal(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_CLAIM)

    def test_missing_justification_retains_with_diagnostic(self) -> None:
        """
        Claim asserted but not justified → RETAIN with MISSING_JUSTIFICATION.
        """

        decision = CompletionGate().adjudicate(
            evidence=self.__evidence(
                asserted=True, justified=False, dispatched=True, evolved=True
            ),
            sub_goal=self.__sub_goal(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_JUSTIFICATION)

    def test_missing_dispatch_retains_with_diagnostic(self) -> None:
        """
        No action dispatched → RETAIN with MISSING_DISPATCH.
        """

        decision = CompletionGate().adjudicate(
            evidence=self.__evidence(
                asserted=True, justified=True, dispatched=False, evolved=True
            ),
            sub_goal=self.__sub_goal(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_DISPATCH)

    def test_missing_screen_evolution_retains_with_diagnostic(self) -> None:
        """
        Action dispatched but screen unchanged → RETAIN with MISSING_SCREEN_EVOLUTION.
        """

        decision = CompletionGate().adjudicate(
            evidence=self.__evidence(
                asserted=True, justified=True, dispatched=True, evolved=False
            ),
            sub_goal=self.__sub_goal(),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_SCREEN_EVOLUTION)


class CompletionGateValidationTest(unittest.TestCase):
    """
    Pins :meth:`CompletionGate.adjudicate` for VALIDATION sub-goals.
    """

    @staticmethod
    def __sub_goal() -> SubGoal:
        """
        Build a VALIDATION sub-goal fixture.
        """

        return SubGoal(
            index=0,
            description="Validate Jars & Containers visible",
            kind=SubGoalKind.VALIDATION,
        )

    def test_asserted_claim_alone_advances(self) -> None:
        """
        Validation sub-goal: asserted claim short-circuits to ADVANCE.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=True, justified=False),
            action=ActionEvidence(dispatched=False),
            screen=ScreenEvidence(evolved=False),
        )

        decision = CompletionGate().adjudicate(evidence=evidence, sub_goal=self.__sub_goal())

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)

    def test_two_of_three_threshold_without_claim_advances(self) -> None:
        """
        Validation sub-goal: justified rationale + screen-verified dispatch → ADVANCE.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False, justified=True),
            action=ActionEvidence(dispatched=True),
            screen=ScreenEvidence(evolved=True),
        )

        decision = CompletionGate().adjudicate(evidence=evidence, sub_goal=self.__sub_goal())

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)

    def test_insufficient_signals_retains(self) -> None:
        """
        Validation sub-goal: no claim, single signal → RETAIN.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False, justified=True),
            action=ActionEvidence(dispatched=False),
            screen=ScreenEvidence(evolved=False),
        )

        decision = CompletionGate().adjudicate(evidence=evidence, sub_goal=self.__sub_goal())

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)


class CompletionGateCriterionAdditiveTest(unittest.TestCase):
    """
    Pins that the criterion evidence is additive only — never vetoes nor rescues.
    """

    def test_criterion_observed_true_does_not_rescue_missing_main_signals(self) -> None:
        """
        Criterion observed True but all 4 main signals missing → RETAIN, not ADVANCE.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False, justified=False),
            action=ActionEvidence(dispatched=False),
            screen=ScreenEvidence(evolved=False),
            criterion=CriterionEvidence(observed=True),
        )
        sub_goal = SubGoal(index=0, description="Tap on X", kind=SubGoalKind.ACTION)

        decision = CompletionGate().adjudicate(evidence=evidence, sub_goal=sub_goal)

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)

    def test_criterion_observed_false_does_not_veto_iahtk_replay(self) -> None:
        """
        IahTk replay at the Domain level: criterion observed False (the post-tap
        screen no longer contains the criterion tokens) does NOT veto an
        otherwise-conclusive ACTION sub-goal decision.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=True, justified=True),
            action=ActionEvidence(dispatched=True),
            screen=ScreenEvidence(evolved=True),
            criterion=CriterionEvidence(observed=False),
        )
        sub_goal = SubGoal(
            index=0,
            description="Tap on Confirm location and continue button",
            kind=SubGoalKind.ACTION,
        )

        decision = CompletionGate().adjudicate(evidence=evidence, sub_goal=sub_goal)

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)
        self.assertIsNone(decision.retain_reason)
