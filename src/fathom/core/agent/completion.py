from __future__ import annotations

from fathom.constants.completion import (
    CompletionThreshold,
    GateOutcome,
    RetainReason,
)
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind


class CompletionGate:
    """
    Domain gate that adjudicates one turn's CompletionEvidence per sub-goal kind and emitted action kind.
    VALIDATION sub-goals short-circuit on asserted claim; ACTION sub-goals require screen-verified dispatch,
    with a VALIDATION-kind escape branch for implicit-completion when the world is already past the failed step.
    """

    def adjudicate(
        self,
        *,
        sub_goal: SubGoal,
        action_kind: ActionKind,
        evidence: CompletionEvidence,
    ) -> GateDecision:
        """
        Return the gate decision (outcome plus diagnostic retain reason) for this turn.
        """

        screen_verified_dispatch = evidence.action.dispatched and evidence.screen.evolved

        if sub_goal.kind is SubGoalKind.VALIDATION:
            return self.__adjudicate_validation(
                evidence=evidence,
                screen_verified_dispatch=screen_verified_dispatch,
            )

        return self.__adjudicate_action(
            evidence=evidence,
            action_kind=action_kind,
            screen_verified_dispatch=screen_verified_dispatch,
        )

    def __adjudicate_validation(
        self,
        *,
        evidence: CompletionEvidence,
        screen_verified_dispatch: bool,
    ) -> GateDecision:
        """
        Validation sub-goal: short-circuit on asserted claim, else 2-of-3 threshold.
        """

        if evidence.claim.asserted:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        met = sum((evidence.claim.justified, screen_verified_dispatch))
        if met >= CompletionThreshold.VALIDATION_WITHOUT_CLAIM:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        return GateDecision(
            outcome=GateOutcome.RETAIN,
            retain_reason=self.__diagnose(evidence=evidence),
        )

    def __adjudicate_action(
        self,
        *,
        action_kind: ActionKind,
        evidence: CompletionEvidence,
        screen_verified_dispatch: bool,
    ) -> GateDecision:
        """
        Action sub-goal: strict path needs screen-verified dispatch; VALIDATION-kind action advances on claim alone.
        """

        if evidence.claim.asserted and evidence.claim.justified and screen_verified_dispatch:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        if (
            evidence.claim.asserted
            and evidence.action.dispatched
            and action_kind is ActionKind.VALIDATION
        ):
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        return GateDecision(
            outcome=GateOutcome.RETAIN,
            retain_reason=self.__diagnose(evidence=evidence),
        )

    @staticmethod
    def __diagnose(*, evidence: CompletionEvidence) -> RetainReason:
        """
        Map the missing signals to a single diagnostic retain reason for RCA.
        """

        if not evidence.claim.asserted:
            return RetainReason.MISSING_CLAIM

        if not evidence.claim.justified:
            return RetainReason.MISSING_JUSTIFICATION

        if not evidence.action.dispatched:
            return RetainReason.MISSING_DISPATCH

        return RetainReason.MISSING_SCREEN_EVOLUTION
