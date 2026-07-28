from __future__ import annotations

from logging import getLogger
from typing import Dict, Optional

from fathom.constants.capability import CompletionMode
from fathom.constants.completion import GateOutcome, RetainReason
from fathom.constants.turn.advancement import AdvanceKind, AdvanceThreshold
from fathom.constants.turn.oracle import OracleThreshold
from fathom.constants.turn.stall import StallState
from fathom.core.agent.completion import OutcomeEvidencePolicy
from fathom.core.capability.catalog import CommandCatalog
from fathom.core.services.directive import DirectivePolicy
from fathom.schemas.advancement import Advancement, RetainHistory
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.criterion import CriterionVerdict
from fathom.schemas.effect import ActionEffectStatus, EffectReading
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.tasks import Task
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.vision import ActionKind

logger = getLogger(__name__)


class Backstop:
    """
    Bounds consecutive same-task retention; escalation at the limit is deterministic.
    """

    def __init__(self, *, limit: int = AdvanceThreshold.RETAIN_ESCALATION) -> None:
        """
        Bind the retention limit.
        """

        self.__limit = limit

    def exhausted(self, *, history: RetainHistory) -> bool:
        """
        Return whether the task's retain streak has reached the escalation limit.
        """

        return history.consecutive >= self.__limit


class AdvancementPolicy:
    """
    Decides task advancement from typed evidence; observed completion advances, claims corroborate.
    """

    def __init__(
        self,
        *,
        backstop: Optional[Backstop] = None,
        floor: float = OracleThreshold.CONFIDENCE_FLOOR,
    ) -> None:
        """
        Bind the retention backstop and the verdict confidence floor.
        """

        self.__floor = floor
        self.__backstop = backstop if backstop is not None else Backstop()

    def decide(self, *, task: Task, evidence: TurnEvidence, history: RetainHistory) -> Advancement:
        """
        Return the advancement decision for one turn of one task.
        A turn that would retain escalates instead once the streak is exhausted or momentum stalls.
        """

        decision = self.__adjudicate(task=task, evidence=evidence)

        if decision.kind is AdvanceKind.RETAIN and (
            self.__backstop.exhausted(history=history) or self.__stalled(evidence=evidence)
        ):
            return self.__escalation(evidence=evidence)

        return decision

    def __adjudicate(self, *, task: Task, evidence: TurnEvidence) -> Advancement:
        """
        Return the proof-requirement decision for the turn, before loop bounding.
        """

        if task.completion is CompletionMode.TERMINAL:
            return self.__terminal(evidence=evidence)

        if task.completion is CompletionMode.CLAIM_VERIFIED:
            return self.__validated(evidence=evidence)

        if task.completion is CompletionMode.CLAIM_OR_TIMEOUT:
            return self.__claimed(evidence=evidence)

        if task.completion is CompletionMode.CAPTURE_VERIFIED:
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.MISSING_CAPTURE)

        if task.completion is CompletionMode.OUTCOME_VERIFIED:
            return self.__durable(evidence=evidence)

        return self.__transient(evidence=evidence)

    def __terminal(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Terminal tasks advance on the dispatched completion command.
        """

        if evidence.action.dispatched:
            return Advancement(kind=AdvanceKind.ADVANCE)

        return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.MISSING_DISPATCH)

    def __validated(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Validation tasks advance only on an executed canonical validation.
        """

        if evidence.validation is not None:
            return Advancement(kind=AdvanceKind.ADVANCE)

        return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.MISSING_VALIDATION)

    def __claimed(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Claim-or-timeout tasks advance on the model's claim; timeouts resolve at the executor.
        """

        if evidence.claim.asserted:
            return Advancement(kind=AdvanceKind.ADVANCE)

        return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.MISSING_CLAIM)

    def __transient(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Screen-verified tasks advance on observed satisfaction or a progress-corroborated claim.
        An observed refutation of the criterion vetoes claim-based advancement.
        """

        if self.__regressed(evidence=evidence):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.LEFT_APPLICATION)

        if self.__satisfied(evidence=evidence):
            return self.__observed(evidence=evidence)

        if self.__refuted(evidence=evidence):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.AWAITING_PROOF)

        if evidence.validation is not None and evidence.claim.asserted:
            return Advancement(kind=AdvanceKind.ADVANCE)

        if (
            evidence.claim.asserted
            and evidence.action.dispatched
            and self.__progressed(evidence=evidence)
        ):
            return Advancement(kind=AdvanceKind.ADVANCE)

        return Advancement(kind=AdvanceKind.RETAIN, reason=self.__diagnose(evidence=evidence))

    def __durable(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Durable tasks advance only on observed or validated outcomes; re-dispatch is barred meanwhile.
        """

        if self.__satisfied(evidence=evidence):
            return self.__observed(evidence=evidence)

        if evidence.validation is not None:
            return Advancement(kind=AdvanceKind.ADVANCE)

        if self.__regressed(evidence=evidence):
            return Advancement(
                kind=AdvanceKind.RETAIN,
                reason=RetainReason.LEFT_APPLICATION,
                redispatch=False,
            )

        return Advancement(
            kind=AdvanceKind.RETAIN,
            reason=RetainReason.AWAITING_PROOF,
            redispatch=False,
        )

    def __observed(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Distinguish satisfaction produced by this turn's action from satisfaction found already true.
        """

        if evidence.action.dispatched:
            return Advancement(kind=AdvanceKind.ADVANCE)

        return Advancement(kind=AdvanceKind.SATISFIED_PRIOR)

    def __escalation(self, *, evidence: TurnEvidence) -> Advancement:
        """
        Close an exhausted retain streak: observed refutation is unsatisfiable, anything else escalates.
        """

        if self.__refuted(evidence=evidence):
            return Advancement(kind=AdvanceKind.UNSATISFIABLE, redispatch=False)

        return Advancement(kind=AdvanceKind.ESCALATE, redispatch=False)

    def __satisfied(self, *, evidence: TurnEvidence) -> bool:
        """
        Return whether the criterion is observed satisfied at or above the confidence floor.
        """

        return (
            evidence.verdict is not None
            and evidence.verdict.outcome is CriterionVerdict.SATISFIED
            and evidence.verdict.confidence >= self.__floor
        )

    def __refuted(self, *, evidence: TurnEvidence) -> bool:
        """
        Return whether the criterion is observed unsatisfied at or above the confidence floor.
        """

        return (
            evidence.verdict is not None
            and evidence.verdict.outcome is CriterionVerdict.UNSATISFIED
            and evidence.verdict.confidence >= self.__floor
        )

    @staticmethod
    def __stalled(*, evidence: TurnEvidence) -> bool:
        """
        Return whether the momentum reading classifies the action stream as stalled.
        """

        return evidence.stall is not None and evidence.stall.state is StallState.STALLED

    @staticmethod
    def __progressed(*, evidence: TurnEvidence) -> bool:
        """
        Return whether the trial effect reports progress.
        """

        return evidence.effect is not None and evidence.effect.trial is ActionEffectStatus.PROGRESS

    @staticmethod
    def __regressed(*, evidence: TurnEvidence) -> bool:
        """
        Return whether the trial effect reports the foreground left the application.
        """

        return (
            evidence.effect is not None and evidence.effect.trial is ActionEffectStatus.REGRESSION
        )

    @staticmethod
    def __diagnose(*, evidence: TurnEvidence) -> RetainReason:
        """
        Map the first missing signal to a single diagnostic retain reason.
        """

        if not evidence.claim.asserted:
            return RetainReason.MISSING_CLAIM

        if not evidence.action.dispatched:
            return RetainReason.MISSING_DISPATCH

        return RetainReason.MISSING_SCREEN_EVOLUTION


class AdvancementTrial:
    """
    Adjudicator-seam adapter that runs the advancement policy as the trial decider.
    """

    def __init__(
        self,
        *,
        catalog: CommandCatalog,
        policy: Optional[AdvancementPolicy] = None,
    ) -> None:
        """
        Bind the task projection and the policy under trial.
        """

        self.__streaks: Dict[int, int] = {}
        self.__wordlist = OutcomeEvidencePolicy()
        self.__projection = DirectivePolicy(catalog=catalog)
        self.__policy = policy if policy is not None else AdvancementPolicy()

    def adjudicate(
        self,
        *,
        sub_goal: SubGoal,
        action_kind: ActionKind,
        evidence: CompletionEvidence,
        measured: Optional[TurnEvidence] = None,
    ) -> GateDecision:
        """
        Decide via the advancement policy and project onto the gate-decision seam.
        """

        task = self.__projection.project(sub_goal=sub_goal)
        self.__tripwire(sub_goal=sub_goal, task=task)

        decision = self.__policy.decide(
            task=task,
            evidence=self.__turn(evidence=evidence, measured=measured),
            history=RetainHistory(consecutive=self.__streaks.get(sub_goal.index, 0)),
        )
        self.__track(index=sub_goal.index, decision=decision)

        return self.__gate(decision=decision)

    def __tripwire(self, *, sub_goal: SubGoal, task: Task) -> None:
        """
        Log word-list vs typed-proof disagreement so the legacy heuristic retires on measured evidence.
        """

        listed = self.__wordlist.needs_proof(sub_goal=sub_goal)
        typed = task.completion is CompletionMode.OUTCOME_VERIFIED

        if listed == typed:
            return

        logger.info(
            "Durable-proof tripwire disagreement",
            extra={
                "event": "proof.tripwire",
                "task.index": sub_goal.index,
                "proof.wordlist": listed,
                "proof.typed": typed,
                "proof.declared": sub_goal.proof.value if sub_goal.proof is not None else None,
                "task.completion": task.completion.value,
            },
        )

    @classmethod
    def __turn(
        cls,
        *,
        evidence: CompletionEvidence,
        measured: Optional[TurnEvidence],
    ) -> TurnEvidence:
        """
        Prefer the measured channel; synthesize the effect from legacy evidence when absent.
        """

        if measured is not None and measured.effect is not None:
            return measured

        effect = cls.__synthesized(evidence=evidence)
        if measured is not None:
            return measured.model_copy(update={"effect": effect})

        return TurnEvidence(claim=evidence.claim, action=evidence.action, effect=effect)

    @staticmethod
    def __synthesized(*, evidence: CompletionEvidence) -> EffectReading:
        """
        Project the legacy screen-evolution boolean into the effect vocabulary.
        """

        status = (
            ActionEffectStatus.PROGRESS
            if evidence.screen.evolved
            else ActionEffectStatus.NO_PROGRESS
        )

        return EffectReading(live=status, trial=status)

    def __track(self, *, index: int, decision: Advancement) -> None:
        """
        Advance or reset the task's retain streak from the decision just produced.
        """

        if decision.kind is AdvanceKind.RETAIN:
            self.__streaks[index] = self.__streaks.get(index, 0) + 1
            return

        self.__streaks[index] = 0

    @staticmethod
    def __gate(*, decision: Advancement) -> GateDecision:
        """
        Project the advancement decision onto the legacy gate-decision seam for comparison.
        """

        if decision.kind in {AdvanceKind.ADVANCE, AdvanceKind.SATISFIED_PRIOR}:
            return GateDecision(outcome=GateOutcome.ADVANCE, retain_reason=None)

        if decision.kind is AdvanceKind.RETAIN:
            return GateDecision(outcome=GateOutcome.RETAIN, retain_reason=decision.reason)

        return GateDecision(outcome=GateOutcome.FAIL, retain_reason=None)
