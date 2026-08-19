from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.constants.assessment import VisualVerdict
from fathom.constants.completion import RetainReason
from fathom.constants.turn.advancement import AdvanceKind, ObservationPhase
from fathom.constants.turn.binding import BindingState
from fathom.constants.turn.oracle import OracleThreshold
from fathom.constants.turn.stall import StallState
from fathom.core.capability.matcher import CommandMatcher
from fathom.schemas.advancement import Advancement
from fathom.schemas.criterion import CriterionVerdict
from fathom.schemas.steps import StepResult
from fathom.schemas.success import (
    CaptureSuccess,
    CommandSuccess,
    ObservationRequirement,
    ObservedSuccess,
    Success,
)
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.visual import VisualEvidence


class AdvancementPolicy:
    """
    Decides sub-goal advancement from canonical success and the turn's correlated typed evidence.
    """

    def __init__(
        self,
        *,
        matcher: Optional[CommandMatcher] = None,
        floor: float = OracleThreshold.CONFIDENCE_FLOOR,
    ) -> None:
        """
        Bind the verdict confidence floor and the command-matching authority.
        """

        self.__floor = floor
        self.__matcher = matcher if matcher is not None else CommandMatcher()

    def decide(self, *, success: Success, evidence: TurnEvidence) -> Advancement:
        """
        Return the advancement decision for one turn; a stalled retain resolves on observed proof, else escalates.
        """

        decision = self.__adjudicate(success=success, evidence=evidence)

        if decision.kind is AdvanceKind.RETAIN and self.__stalled(evidence=evidence):
            return self.__resolve_stall(success=success, evidence=evidence)

        return decision

    def __resolve_stall(self, *, success: Success, evidence: TurnEvidence) -> Advancement:
        """
        Close a stalled retain by advancing when the outcome is observably satisfied, else escalating for help.
        """

        if self.__satisfied_outcome(success=success, evidence=evidence):
            return Advancement(kind=AdvanceKind.ADVANCE)

        return self.__escalation()

    def __satisfied_outcome(self, *, success: Success, evidence: TurnEvidence) -> bool:
        """
        Whether settled evidence confirms the outcome, independent of the dispatch a command named; capture never qualifies.
        """

        if not isinstance(success, CommandSuccess):
            return False

        if success.postcondition is not None:
            return self.__confirms(evidence=evidence, observation=success.postcondition)

        return self.__graded(evidence=evidence, outcome=CriterionVerdict.SATISFIED)

    def __adjudicate(self, *, success: Success, evidence: TurnEvidence) -> Advancement:
        """
        Return the per-kind proof decision for the turn, before loop bounding.
        """

        if isinstance(success, ObservedSuccess):
            return self.__observed(success=success, evidence=evidence)

        if isinstance(success, CommandSuccess):
            return self.__command(success=success, evidence=evidence)

        return self.__capture(success=success, evidence=evidence)

    def __observed(self, *, success: ObservedSuccess, evidence: TurnEvidence) -> Advancement:
        """
        Observed success advances on a fresh satisfied verdict for its own observation; a pre-dispatch hit is prior.
        """

        if not self.__confirms(evidence=evidence, observation=success.observation):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.AWAITING_PROOF)

        if evidence.phase is ObservationPhase.PRE_DISPATCH:
            return Advancement(kind=AdvanceKind.SATISFIED_PRIOR)

        return Advancement(kind=AdvanceKind.ADVANCE)

    def __command(self, *, success: CommandSuccess, evidence: TurnEvidence) -> Advancement:
        """
        Command success advances only on a matching executed action, plus its postcondition when present.
        """

        execution = evidence.execution
        if execution is None or not execution.executed:
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.MISSING_DISPATCH)

        if execution.step.requirement != success.requirement or not self.__matcher.matches(
            requirement=success.requirement, action=execution.step.action
        ):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.AWAITING_PROOF)

        if not self.__grounded(evidence=evidence):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.AWAITING_PROOF)

        if success.postcondition is not None and not self.__confirms(
            evidence=evidence, observation=success.postcondition
        ):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.AWAITING_PROOF)

        return Advancement(kind=AdvanceKind.ADVANCE)

    @staticmethod
    def __grounded(*, evidence: TurnEvidence) -> bool:
        """
        Require the executed target to exist on the current screen: a present binding must not be MISSING.
        """

        return evidence.binding is None or evidence.binding.state is not BindingState.MISSING

    def __capture(self, *, success: CaptureSuccess, evidence: TurnEvidence) -> Advancement:
        """
        Capture success advances only on an executed STORE that committed the exact capture identity.
        """

        execution = evidence.execution
        if execution is None or not self.__captured(success=success, execution=execution):
            return Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.MISSING_CAPTURE)

        return Advancement(kind=AdvanceKind.ADVANCE)

    @staticmethod
    def __captured(*, success: CaptureSuccess, execution: StepResult) -> bool:
        """
        Return whether an executed STORE committed exactly the requested capture identity with a value.
        """

        request = execution.step.action.capture
        capture = execution.capture

        # Identity is the capture NAME (the stable variable the user named); ``subject`` is free-text
        # the planner and decomposer each phrase independently, so it is descriptive, not an identity
        # key, and is never required to match verbatim.
        return (
            execution.executed
            and execution.step.action.action_type is ActionType.STORE
            and request is not None
            and request.name == success.target.name
            and capture is not None
            and capture.success
            and bool(capture.value)
            and capture.name == success.target.name
            and capture.step == execution.step.step_number
        )

    @staticmethod
    def __escalation() -> Advancement:
        """
        Close a stalled retain by escalating for help; a vision refute may be a false negative, never terminal.
        """

        return Advancement(kind=AdvanceKind.ESCALATE, redispatch=False)

    def __confirms(self, *, evidence: TurnEvidence, observation: ObservationRequirement) -> bool:
        """
        Return whether the turn adjudicated this exact observation and found it satisfied.
        """

        return self.__reads(
            evidence=evidence, observation=observation, outcome=CriterionVerdict.SATISFIED
        )

    def __reads(
        self,
        *,
        evidence: TurnEvidence,
        outcome: CriterionVerdict,
        observation: ObservationRequirement,
    ) -> bool:
        """
        Read the outcome for this observation from the settled-screen visual evidence, else the oracle verdict.
        """

        if evidence.visual is not None:
            return self.__visual_reads(
                visual=evidence.visual, observation=observation, outcome=outcome
            )
        return evidence.observation == observation and self.__graded(
            evidence=evidence, outcome=outcome
        )

    def __visual_reads(
        self,
        *,
        visual: VisualEvidence,
        observation: ObservationRequirement,
        outcome: CriterionVerdict,
    ) -> bool:
        """
        Confirm from a schema-valid, action-free, package-consistent assessment of the exact observation.
        """

        assessment = visual.assessment
        if visual.observation != observation or assessment is None or visual.malformed:
            return False
        if outcome is CriterionVerdict.SATISFIED:
            return (
                assessment.verdict is VisualVerdict.SATISFIED
                and assessment.confidence >= self.__floor
                and not visual.action_present
                and self.__foreground_holds(visual=visual)
            )
        if outcome is CriterionVerdict.UNSATISFIED:
            return assessment.verdict is VisualVerdict.NOT_SATISFIED
        return False

    @staticmethod
    def __foreground_holds(*, visual: VisualEvidence) -> bool:
        """
        Require the foreground to be present and exactly the bound target; an unbound target imposes no check.
        """

        if not visual.authority.bound:
            return True
        return visual.foreground is not None and visual.foreground == visual.authority.package

    def __graded(self, *, evidence: TurnEvidence, outcome: CriterionVerdict) -> bool:
        """
        Return whether the fresh verdict holds the given outcome at or above the confidence floor.
        """

        return (
            evidence.verdict is not None
            and evidence.verdict.confidence >= self.__floor
            and evidence.verdict.outcome is outcome
        )

    @staticmethod
    def __stalled(*, evidence: TurnEvidence) -> bool:
        """
        Return whether the momentum reading classifies the action stream as stalled.
        """

        return evidence.stall is not None and evidence.stall.state is StallState.STALLED
