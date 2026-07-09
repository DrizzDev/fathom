from __future__ import annotations

import re
from typing import Optional

from fathom.constants.completion import (
    DURABLE_OUTCOME_TERMS,
    GateOutcome,
    RetainReason,
)
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind


class CompletionGate:
    """
    Domain gate that adjudicates one turn's CompletionEvidence per sub-goal kind and emitted action kind.
    VALIDATION sub-goals require a recorded validate action; ACTION sub-goals require screen-verified dispatch,
    with durable outcome sub-goals requiring recorded validation evidence before advancement.
    """

    def __init__(self, *, outcome_policy: Optional["OutcomeEvidencePolicy"] = None) -> None:
        """
        Bind the durable-outcome policy used for action sub-goals.
        """

        self.__outcome_policy = outcome_policy or OutcomeEvidencePolicy()

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

        effective = evidence.action.dispatched and evidence.screen.evolved

        if sub_goal.kind is SubGoalKind.VALIDATION:
            return self.__adjudicate_validation(evidence=evidence)

        return self.__adjudicate_action(
            sub_goal=sub_goal,
            evidence=evidence,
            effective=effective,
        )

    def __adjudicate_validation(self, *, evidence: CompletionEvidence) -> GateDecision:
        """
        Validation sub-goal: completion requires a concrete validate action.
        """

        if evidence.validation.executed:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        return GateDecision(
            outcome=GateOutcome.RETAIN,
            retain_reason=RetainReason.MISSING_VALIDATION,
        )

    def __adjudicate_action(
        self,
        *,
        sub_goal: SubGoal,
        evidence: CompletionEvidence,
        effective: bool,
    ) -> GateDecision:
        """
        Action sub-goal: strict path needs screen-verified dispatch unless durable outcome proof is required.
        """

        if self.__outcome_policy.needs_proof(sub_goal=sub_goal):
            return self.__adjudicate_durable_outcome(
                evidence=evidence,
                effective=effective,
            )

        if evidence.claim.asserted and evidence.claim.explained and effective:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        if evidence.claim.asserted and evidence.validation.executed:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        return GateDecision(
            outcome=GateOutcome.RETAIN,
            retain_reason=self.__diagnose(evidence=evidence),
        )

    def __adjudicate_durable_outcome(
        self,
        *,
        evidence: CompletionEvidence,
        effective: bool,
    ) -> GateDecision:
        """
        Durable action sub-goal: require validate evidence before accepting a model claim.
        """

        if evidence.claim.asserted and evidence.validation.executed:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        if evidence.claim.asserted and evidence.claim.explained and effective:
            return GateDecision(
                outcome=GateOutcome.RETAIN,
                retain_reason=RetainReason.MISSING_OUTCOME_EVIDENCE,
            )

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

        if not evidence.claim.explained:
            return RetainReason.MISSING_JUSTIFICATION

        if not evidence.action.dispatched:
            return RetainReason.MISSING_DISPATCH

        return RetainReason.MISSING_SCREEN_EVOLUTION


class OutcomeEvidencePolicy:
    """
    Classifies action sub-goals whose durable outcome needs validation evidence before advancement.
    """

    __WORD_PATTERN = re.compile(r"[a-z]+")

    def needs_proof(self, *, sub_goal: SubGoal) -> bool:
        """
        Return whether the sub-goal describes a durable outcome rather than a transient navigation step.
        """

        words = set(
            self.__WORD_PATTERN.findall(
                f"{sub_goal.description} {sub_goal.criterion or ''}".lower()
            )
        )
        return bool(words.intersection(DURABLE_OUTCOME_TERMS))
