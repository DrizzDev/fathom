from __future__ import annotations

import unittest
from typing import Optional

from fathom.constants import ActionType
from fathom.constants.capability import CompletionMode
from fathom.constants.completion import GateOutcome, RetainReason
from fathom.constants.turn.advancement import AdvanceKind, AdvanceThreshold
from fathom.constants.turn.stall import StallState
from fathom.constants.turn.validation import ValidationSource
from fathom.core.agent.advancement import AdvancementPolicy, AdvancementTrial
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.schemas.advancement import RetainHistory
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    ScreenEvidence,
)
from fathom.schemas.criterion import CriterionVerdict, Verdict
from fathom.schemas.effect import ActionEffectStatus, EffectReading
from fathom.schemas.stall import StallSignal
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.tasks import Task
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.validation import Validation
from fathom.schemas.vision import ActionKind


class AdvancementPolicyTest(unittest.TestCase):
    """
    Cover the decision table over the corpus's real failure classes.
    """

    def setUp(self) -> None:
        """
        Build the policy under test.
        """

        self.policy = AdvancementPolicy()

    def test_transient_task_advances_on_progress_corroborated_claim(self) -> None:
        """
        The strict-path advance: claim asserted, action dispatched, effect progressed.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.SCREEN_VERIFIED),
            evidence=self.__evidence(
                claim=True, dispatched=True, trial=ActionEffectStatus.PROGRESS
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.ADVANCE)

    def test_transient_task_retains_forgotten_claim(self) -> None:
        """
        The ISSUE-004 turn: work done, boolean forgotten — retained, never phantom-advanced.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.SCREEN_VERIFIED),
            evidence=self.__evidence(
                claim=False, dispatched=True, trial=ActionEffectStatus.PROGRESS
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.RETAIN)
        self.assertEqual(decision.reason, RetainReason.MISSING_CLAIM)

    def test_backstop_escalates_the_loop_at_the_limit(self) -> None:
        """
        The same forgotten-claim turn at the retain limit escalates instead of looping forever.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.SCREEN_VERIFIED),
            evidence=self.__evidence(
                claim=False, dispatched=True, trial=ActionEffectStatus.PROGRESS
            ),
            history=RetainHistory(consecutive=AdvanceThreshold.RETAIN_ESCALATION),
        )

        self.assertEqual(decision.kind, AdvanceKind.ESCALATE)
        self.assertFalse(decision.redispatch)

    def test_departed_foreground_blocks_advancement(self) -> None:
        """
        The OS-eaten swipe: an asserted claim cannot advance a turn that left the application.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.SCREEN_VERIFIED),
            evidence=self.__evidence(
                claim=True, dispatched=True, trial=ActionEffectStatus.REGRESSION
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.RETAIN)
        self.assertEqual(decision.reason, RetainReason.LEFT_APPLICATION)

    def test_durable_task_awaits_proof_without_redispatch(self) -> None:
        """
        The ISSUE-005 guard: an effective add-to-cart retains awaiting proof and bars re-dispatch.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.OUTCOME_VERIFIED),
            evidence=self.__evidence(
                claim=True, dispatched=True, trial=ActionEffectStatus.PROGRESS
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.RETAIN)
        self.assertEqual(decision.reason, RetainReason.AWAITING_PROOF)
        self.assertFalse(decision.redispatch)

    def test_durable_task_advances_on_observed_outcome(self) -> None:
        """
        The oracle reads the cart on the settled screen; the durable task advances.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.OUTCOME_VERIFIED),
            evidence=self.__evidence(
                claim=False,
                dispatched=True,
                trial=ActionEffectStatus.PROGRESS,
                verdict=Verdict(outcome=CriterionVerdict.SATISFIED, confidence=0.93),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.ADVANCE)

    def test_low_confidence_verdict_does_not_advance_durable_task(self) -> None:
        """
        An under-floor satisfied reading proposes nothing; the task keeps awaiting proof.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.OUTCOME_VERIFIED),
            evidence=self.__evidence(
                claim=True,
                dispatched=True,
                trial=ActionEffectStatus.PROGRESS,
                verdict=Verdict(outcome=CriterionVerdict.SATISFIED, confidence=0.4),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.RETAIN)
        self.assertEqual(decision.reason, RetainReason.AWAITING_PROOF)

    def test_satisfied_prior_when_criterion_already_true_without_dispatch(self) -> None:
        """
        The subsumed task: criterion observed satisfied before any action dispatched this turn.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.OUTCOME_VERIFIED),
            evidence=self.__evidence(
                claim=False,
                dispatched=False,
                trial=ActionEffectStatus.NO_PROGRESS,
                verdict=Verdict(outcome=CriterionVerdict.SATISFIED, confidence=0.9),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.SATISFIED_PRIOR)

    def test_refuted_outcome_at_exhaustion_is_unsatisfiable(self) -> None:
        """
        Observed refutation at the retain limit fails honestly instead of escalating blindly.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.OUTCOME_VERIFIED),
            evidence=self.__evidence(
                claim=True,
                dispatched=True,
                trial=ActionEffectStatus.NO_PROGRESS,
                verdict=Verdict(outcome=CriterionVerdict.UNSATISFIED, confidence=0.9),
            ),
            history=RetainHistory(consecutive=AdvanceThreshold.RETAIN_ESCALATION),
        )

        self.assertEqual(decision.kind, AdvanceKind.UNSATISFIABLE)

    def test_validation_task_requires_canonical_validation(self) -> None:
        """
        A validation task retains on a bare claim and advances on the typed assertion.
        """

        task = self.__task(completion=CompletionMode.CLAIM_VERIFIED)

        retained = self.policy.decide(
            task=task,
            evidence=self.__evidence(claim=True, dispatched=True, trial=None),
            history=RetainHistory(),
        )
        advanced = self.policy.decide(
            task=task,
            evidence=self.__evidence(
                claim=True,
                dispatched=True,
                trial=None,
                validation=Validation(
                    subject="Cart shows Diet Coke", source=ValidationSource.STATE
                ),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(retained.kind, AdvanceKind.RETAIN)
        self.assertEqual(retained.reason, RetainReason.MISSING_VALIDATION)
        self.assertEqual(advanced.kind, AdvanceKind.ADVANCE)

    def test_stalled_momentum_escalates_a_retaining_turn(self) -> None:
        """
        The ISSUE-007 link: a stalled effect stream escalates a would-retain turn immediately.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.SCREEN_VERIFIED),
            evidence=self.__evidence(
                claim=False,
                dispatched=True,
                trial=ActionEffectStatus.NO_PROGRESS,
                stall=StallSignal(state=StallState.STALLED, streak=3),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.ESCALATE)

    def test_stalled_momentum_never_blocks_an_advancing_turn(self) -> None:
        """
        A validation advance survives a stalled stream; validate actions never move the screen.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.CLAIM_VERIFIED),
            evidence=self.__evidence(
                claim=True,
                dispatched=True,
                trial=ActionEffectStatus.NO_PROGRESS,
                validation=Validation(subject="Note is listed", source=ValidationSource.COMMAND),
                stall=StallSignal(state=StallState.STALLED, streak=4),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.ADVANCE)

    def test_refuting_verdict_vetoes_claim_advance_on_transient_task(self) -> None:
        """
        A mislabeled durable task with a refuting oracle reading cannot phantom-advance on its claim.
        """

        decision = self.policy.decide(
            task=self.__task(completion=CompletionMode.SCREEN_VERIFIED),
            evidence=self.__evidence(
                claim=True,
                dispatched=True,
                trial=ActionEffectStatus.PROGRESS,
                verdict=Verdict(outcome=CriterionVerdict.UNSATISFIED, confidence=0.9),
            ),
            history=RetainHistory(),
        )

        self.assertEqual(decision.kind, AdvanceKind.RETAIN)
        self.assertEqual(decision.reason, RetainReason.AWAITING_PROOF)

    @staticmethod
    def __task(*, completion: CompletionMode) -> Task:
        """
        Build a task with the given proof requirement.
        """

        return Task(
            index=1,
            description="Add Diet Coke to the cart",
            kind=SubGoalKind.ACTION,
            completion=completion,
        )

    @staticmethod
    def __evidence(
        *,
        claim: bool,
        dispatched: bool,
        trial: Optional[ActionEffectStatus],
        verdict: Optional[Verdict] = None,
        validation: Optional[Validation] = None,
        stall: Optional[StallSignal] = None,
    ) -> TurnEvidence:
        """
        Build turn evidence from the decision-bearing signals.
        """

        return TurnEvidence(
            claim=ClaimEvidence(asserted=claim),
            action=ActionEvidence(dispatched=dispatched, executed=dispatched),
            effect=EffectReading(live=trial, trial=trial) if trial is not None else None,
            verdict=verdict,
            validation=validation,
            stall=stall,
        )


class AdvancementTrialTest(unittest.TestCase):
    """
    Cover the seam adapter: streak accounting, legacy-evidence synthesis, and gate projection.
    """

    def setUp(self) -> None:
        """
        Build the trial adapter over the full catalog.
        """

        self.trial = AdvancementTrial(catalog=CommandCatalogProvider().build())
        self.sub_goal = SubGoal(description="Open the notes list", index=1)
        self.loop_evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False),
            action=ActionEvidence(dispatched=True, executed=True),
            screen=ScreenEvidence(evolved=True),
        )

    def test_bounds_the_recorded_loop_class(self) -> None:
        """
        Replaying the forgotten-claim turn retains k times, then projects a FAIL escalation.
        """

        outcomes = [
            self.trial.adjudicate(
                sub_goal=self.sub_goal,
                action_kind=ActionKind.NAVIGATION,
                evidence=self.loop_evidence,
            ).outcome
            for _ in range(AdvanceThreshold.RETAIN_ESCALATION + 1)
        ]

        self.assertEqual(
            outcomes,
            [GateOutcome.RETAIN] * AdvanceThreshold.RETAIN_ESCALATION + [GateOutcome.FAIL],
        )

    def test_advance_resets_the_streak(self) -> None:
        """
        An advancing turn clears the retain streak for the task.
        """

        advancing = CompletionEvidence(
            claim=ClaimEvidence(asserted=True),
            action=ActionEvidence(dispatched=True, executed=True),
            screen=ScreenEvidence(evolved=True),
        )

        first = self.trial.adjudicate(
            sub_goal=self.sub_goal,
            action_kind=ActionKind.NAVIGATION,
            evidence=self.loop_evidence,
        )
        second = self.trial.adjudicate(
            sub_goal=self.sub_goal,
            action_kind=ActionKind.NAVIGATION,
            evidence=advancing,
        )
        third = self.trial.adjudicate(
            sub_goal=self.sub_goal,
            action_kind=ActionKind.NAVIGATION,
            evidence=self.loop_evidence,
        )

        self.assertEqual(first.outcome, GateOutcome.RETAIN)
        self.assertEqual(second.outcome, GateOutcome.ADVANCE)
        self.assertEqual(third.outcome, GateOutcome.RETAIN)

    def test_validation_sub_goal_projects_to_claim_verified(self) -> None:
        """
        A validation-kind sub-goal requires the typed validation through the projection.
        """

        decision = self.trial.adjudicate(
            sub_goal=SubGoal(
                description="Verify the note exists",
                index=3,
                kind=SubGoalKind.VALIDATION,
                directive=ActionType.VALIDATE,
            ),
            action_kind=ActionKind.VALIDATION,
            evidence=self.loop_evidence,
            measured=TurnEvidence(
                claim=ClaimEvidence(asserted=True),
                action=ActionEvidence(dispatched=True, executed=True),
                validation=Validation(subject="Note is listed", source=ValidationSource.COMMAND),
            ),
        )

        self.assertEqual(decision.outcome, GateOutcome.ADVANCE)
