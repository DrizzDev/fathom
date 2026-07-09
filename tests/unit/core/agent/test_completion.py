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
    ValidationEvidence,
)
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind


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
        evolved: bool,
        asserted: bool,
        dispatched: bool,
        explained: bool,
    ) -> CompletionEvidence:
        """
        Build a CompletionEvidence with the requested signal truth table.
        """

        return CompletionEvidence(
            screen=ScreenEvidence(evolved=evolved),
            action=ActionEvidence(dispatched=dispatched, executed=dispatched),
            claim=ClaimEvidence(asserted=asserted, explained=explained),
        )

    def test_all_signals_advance(self) -> None:
        """
        Claim asserted + explained + action dispatched + screen evolved → ADVANCE.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(asserted=True, explained=True, dispatched=True, evolved=True),
        )

        self.assertIsNone(decision.retain_reason)
        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)

    def test_missing_claim_retains_with_diagnostic(self) -> None:
        """
        Claim not asserted → RETAIN with MISSING_CLAIM.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(asserted=False, explained=True, dispatched=True, evolved=True),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_CLAIM)

    def test_missing_justification_retains_with_diagnostic(self) -> None:
        """
        Claim asserted but not explained → RETAIN with MISSING_JUSTIFICATION.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(asserted=True, explained=False, dispatched=True, evolved=True),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_JUSTIFICATION)

    def test_missing_dispatch_retains_with_diagnostic(self) -> None:
        """
        No action dispatched → RETAIN with MISSING_DISPATCH.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(asserted=True, explained=True, dispatched=False, evolved=True),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_DISPATCH)

    def test_missing_screen_evolution_retains_with_diagnostic(self) -> None:
        """
        Action dispatched but screen unchanged → RETAIN with MISSING_SCREEN_EVOLUTION.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(asserted=True, explained=True, dispatched=True, evolved=False),
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
            kind=SubGoalKind.VALIDATION,
            description="Validate Jars & Containers visible",
        )

    def test_asserted_claim_without_validate_action_retains(self) -> None:
        """
        Validation sub-goal: a planner claim cannot complete validation without a validate action.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=False),
            action=ActionEvidence(dispatched=False, executed=False),
            claim=ClaimEvidence(asserted=True, explained=False),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_VALIDATION)

    def test_executed_validate_action_advances(self) -> None:
        """
        Validation sub-goal: recorded validate evidence advances without screen evolution.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=False),
            action=ActionEvidence(dispatched=True, executed=True),
            validation=ValidationEvidence(executed=True),
            claim=ClaimEvidence(asserted=True, explained=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)

    def test_non_validate_action_with_screen_evolution_retains(self) -> None:
        """
        Validation sub-goal: navigation progress cannot complete validation.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=True),
            action=ActionEvidence(dispatched=True, executed=True),
            validation=ValidationEvidence(executed=False),
            claim=ClaimEvidence(asserted=True, explained=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_VALIDATION)

    def test_insufficient_signals_retains(self) -> None:
        """
        Validation sub-goal: no validate action → RETAIN.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=False),
            action=ActionEvidence(dispatched=False, executed=False),
            claim=ClaimEvidence(asserted=False, explained=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_VALIDATION)


class CompletionGateCriterionAdditiveTest(unittest.TestCase):
    """
    Pins that the criterion evidence is additive only — never vetoes nor rescues.
    """

    def test_criterion_observed_true_does_not_rescue_missing_main_signals(self) -> None:
        """
        Criterion observed True but all 4 main signals missing → RETAIN, not ADVANCE.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=False),
            action=ActionEvidence(dispatched=False, executed=False),
            criterion=CriterionEvidence(observed=True),
            claim=ClaimEvidence(asserted=False, explained=False),
        )
        sub_goal = SubGoal(index=0, description="Tap on X", kind=SubGoalKind.ACTION)

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=sub_goal,
            action_kind=ActionKind.NAVIGATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)

    def test_action_subgoal_advances_when_criterion_unobserved_but_claim_conclusive(
        self,
    ) -> None:
        """
        Criterion observed False on an ACTION sub-goal must NOT veto an
        otherwise-conclusive claim + dispatched-action + evolved-screen verdict.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=True),
            action=ActionEvidence(dispatched=True, executed=True),
            criterion=CriterionEvidence(observed=False),
            claim=ClaimEvidence(asserted=True, explained=True),
        )
        sub_goal = SubGoal(
            index=0,
            kind=SubGoalKind.ACTION,
            description="Tap on Confirm location and continue button",
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=sub_goal,
            action_kind=ActionKind.NAVIGATION,
        )

        self.assertIsNone(decision.retain_reason)
        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)


class CompletionGateValidateEscapeTest(unittest.TestCase):
    """
    Pin recorded validate evidence on ACTION sub-goals.
    """

    @staticmethod
    def __sub_goal() -> SubGoal:
        """
        Build an ACTION sub-goal fixture for the escape branch.
        """

        return SubGoal(
            index=0,
            kind=SubGoalKind.ACTION,
            description="Tap on confirm and proceed",
        )

    @staticmethod
    def __evidence(
        *,
        asserted: bool,
        explained: bool,
        dispatched: bool,
        evolved: bool,
        validated: bool = False,
    ) -> CompletionEvidence:
        """
        Build a CompletionEvidence fixture with the requested truth table.
        """

        return CompletionEvidence(
            screen=ScreenEvidence(evolved=evolved),
            action=ActionEvidence(dispatched=dispatched, executed=dispatched),
            claim=ClaimEvidence(asserted=asserted, explained=explained),
            validation=ValidationEvidence(executed=validated),
        )

    def test_validate_evidence_advances_when_asserted_and_dispatched(self) -> None:
        """
        ACTION + recorded validate evidence + asserted + dispatched advances even when screen unchanged.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
            evidence=self.__evidence(
                asserted=True,
                explained=False,
                dispatched=True,
                evolved=False,
                validated=True,
            ),
        )

        self.assertIsNone(decision.retain_reason)
        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)

    def test_validate_kind_retains_when_claim_not_asserted(self) -> None:
        """
        Escape branch requires the planner's asserted completion claim; missing claim must retain.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
            evidence=self.__evidence(
                asserted=False, explained=True, dispatched=True, evolved=False
            ),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)

    def test_validate_kind_retains_when_action_not_dispatched(self) -> None:
        """
        Escape branch requires the validate action to have actually run on the device this turn.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
            evidence=self.__evidence(
                asserted=True, explained=True, dispatched=False, evolved=False
            ),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)

    def test_navigation_kind_keeps_strict_path_when_screen_not_evolved(self) -> None:
        """
        ACTION + NAVIGATION-kind (TAP) without screen evolution must not use the escape branch; strict path retains.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(asserted=True, explained=True, dispatched=True, evolved=False),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)

    def test_input_kind_keeps_strict_path_when_screen_not_evolved(self) -> None:
        """
        ACTION + INPUT-kind (TYPE) without screen evolution must not use the escape branch; strict path retains.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.INPUT,
            evidence=self.__evidence(asserted=True, explained=True, dispatched=True, evolved=False),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)

    def test_observation_kind_is_not_a_backdoor_through_escape_branch(self) -> None:
        """
        ACTION + OBSERVATION-kind (WAIT) cannot bypass the strict path; escape branch is VALIDATION-only.
        """

        decision = CompletionGate().adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.OBSERVATION,
            evidence=self.__evidence(asserted=True, explained=True, dispatched=True, evolved=False),
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)


class CompletionGateValidationMirrorRegressionTest(unittest.TestCase):
    """
    Locks validation sub-goals against ACTION-style shortcut drift.
    """

    @staticmethod
    def __sub_goal() -> SubGoal:
        """
        Build a VALIDATION sub-goal fixture for the mirror regression.
        """

        return SubGoal(
            index=0,
            kind=SubGoalKind.VALIDATION,
            description="Validate offerwall is displayed",
        )

    def test_validation_does_not_advance_on_explained_dispatched_screen_evolution(
        self,
    ) -> None:
        """
        Reason + dispatch + screen evolution still needs a validate action.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False, explained=True),
            action=ActionEvidence(dispatched=True, executed=True),
            screen=ScreenEvidence(evolved=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_VALIDATION)

    def test_validation_does_not_advance_without_explained_dispatched_evolution(
        self,
    ) -> None:
        """
        Dispatched + screen evolved alone does NOT advance a VALIDATION sub-goal.
        """

        evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False, explained=False),
            action=ActionEvidence(dispatched=True, executed=True),
            screen=ScreenEvidence(evolved=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)


class CompletionGateExecutedIndependenceTest(unittest.TestCase):
    """
    The gate ignores action.executed; flipping it never changes the decision (executed is telemetry only).
    """

    @staticmethod
    def __sub_goal() -> SubGoal:
        """
        Build an ACTION sub-goal fixture.
        """

        return SubGoal(index=0, description="Tap on Submit", kind=SubGoalKind.ACTION)

    @staticmethod
    def __evidence(*, executed: bool) -> CompletionEvidence:
        """
        Build an otherwise-ADVANCE evidence bundle with the requested executed bit.
        """

        return CompletionEvidence(
            screen=ScreenEvidence(evolved=True),
            action=ActionEvidence(dispatched=True, executed=executed),
            claim=ClaimEvidence(asserted=True, explained=True),
        )

    def test_gate_outcome_is_identical_regardless_of_executed(self) -> None:
        """
        Holding dispatched/claim/screen fixed, both executed values yield the same gate decision.
        """

        gate = CompletionGate()

        executed_true = gate.adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(executed=True),
        )
        executed_false = gate.adjudicate(
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
            evidence=self.__evidence(executed=False),
        )

        self.assertEqual(executed_true.outcome, executed_false.outcome)
        self.assertEqual(executed_true.retain_reason, executed_false.retain_reason)
        self.assertEqual(executed_true.outcome, GateOutcome.ADVANCE)


class CompletionGateDurableOutcomeTest(unittest.TestCase):
    """
    Pins durable outcome sub-goals against intermediate surface transitions.
    """

    @staticmethod
    def __sub_goal() -> SubGoal:
        """
        Build an ACTION sub-goal that needs durable outcome proof.
        """

        return SubGoal(
            index=0,
            kind=SubGoalKind.ACTION,
            description="Search and add 1 diet coke to cart",
            criterion="One diet coke is added to the cart.",
        )

    def test_intermediate_screen_transition_does_not_complete_durable_outcome(self) -> None:
        """
        Durable outcome: claim + screen evolution still needs recorded validation evidence.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=True),
            action=ActionEvidence(dispatched=True, executed=True),
            claim=ClaimEvidence(asserted=True, explained=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.NAVIGATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.RETAIN)
        self.assertEqual(decision.retain_reason, RetainReason.MISSING_OUTCOME_EVIDENCE)

    def test_recorded_validation_completes_durable_outcome(self) -> None:
        """
        Durable outcome: recorded validate evidence can complete without screen evolution.
        """

        evidence = CompletionEvidence(
            screen=ScreenEvidence(evolved=False),
            action=ActionEvidence(dispatched=True, executed=True),
            validation=ValidationEvidence(executed=True),
            claim=ClaimEvidence(asserted=True, explained=True),
        )

        decision = CompletionGate().adjudicate(
            evidence=evidence,
            sub_goal=self.__sub_goal(),
            action_kind=ActionKind.VALIDATION,
        )

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)
