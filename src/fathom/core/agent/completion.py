from __future__ import annotations

from fathom.constants.completion import (
    CompletionThreshold,
    GateOutcome,
    RetainReason,
)
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.subgoal import SubGoal, SubGoalKind


class CompletionGate:
    """
    Domain gate that adjudicates one turn's CompletionEvidence per sub-goal kind.

    Threshold policy mirrors main exactly:
      - VALIDATION: short-circuit on claim.asserted; else require any two of
        {claim.justified, action.dispatched-and-screen.evolved}.
      - ACTION: require claim.asserted AND claim.justified AND
        (action.dispatched AND screen.evolved).

    The criterion field on CompletionEvidence is intentionally never consulted by the gate;
    it is logged for RCA but never vetoes an otherwise-conclusive decision.
    """

    def adjudicate(self, *, evidence: CompletionEvidence, sub_goal: SubGoal) -> GateDecision:
        """
        Return a typed gate decision (outcome plus diagnostic retain reason).
        """

        screen_verified_dispatch = evidence.action.dispatched and evidence.screen.evolved

        if sub_goal.kind is SubGoalKind.VALIDATION:
            return self.__adjudicate_validation(
                evidence=evidence,
                screen_verified_dispatch=screen_verified_dispatch,
            )

        return self.__adjudicate_action(
            evidence=evidence,
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
        evidence: CompletionEvidence,
        screen_verified_dispatch: bool,
    ) -> GateDecision:
        """
        Action sub-goal: require asserted claim, justified rationale, and screen-verified dispatch.
        """

        if (
            evidence.claim.asserted
            and evidence.claim.justified
            and screen_verified_dispatch
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
