from __future__ import annotations

from typing import Optional, Tuple

from fathom.constants.assessment import PhaseIncomparability
from fathom.constants.turn.advancement import AdvanceKind, ObservationPhase
from fathom.core.agent.candidate import ShadowCandidate
from fathom.core.agent.eligibility import Eligibility
from fathom.core.capability.matcher import CommandMatcher
from fathom.schemas.advancement import Advancement
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.planner import PlannerMetrics
from fathom.schemas.results import AnalysisResult
from fathom.schemas.shadow import (
    ComparablePhase,
    GoalCursor,
    IncomparablePhase,
    ShadowAction,
    ShadowApplication,
    ShadowCursor,
    ShadowExecution,
    ShadowGoal,
    ShadowObservation,
    ShadowPhase,
    ShadowPostDispatch,
    ShadowTurn,
    ShadowTurnDraft,
)
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import GoalState, PendingProof
from fathom.schemas.success import CommandSuccess, ObservedSuccess, Success
from fathom.schemas.target import TargetAuthority


class ShadowRunner:
    """
    Build the pre-dispatch shadow draft in Analyze and finalize it with real post-dispatch execution facts.
    """

    __ADVANCING: Tuple[AdvanceKind, ...] = (AdvanceKind.ADVANCE, AdvanceKind.SATISFIED_PRIOR)

    def __init__(
        self,
        *,
        candidate: Optional[ShadowCandidate] = None,
        matcher: Optional[CommandMatcher] = None,
    ) -> None:
        """
        Bind the candidate producer and the command matcher.
        """

        self.__candidate = candidate if candidate is not None else ShadowCandidate()
        self.__matcher = matcher if matcher is not None else CommandMatcher()

    def draft(
        self,
        *,
        workflow_id: str,
        active: GoalState,
        analysis: AnalysisResult,
        metrics: PlannerMetrics,
        screen: str,
        foreground: Optional[str],
        authority: TargetAuthority,
        live_pre: Advancement,
        cursor_before: GoalCursor,
    ) -> ShadowTurnDraft:
        """
        Build the pre-dispatch draft: candidate and live decision computed on the settled pre-dispatch screen.
        """

        proof = active.progress.proof
        candidate = self.__candidate.decide(
            success=active.success,
            phase=ObservationPhase.PRE_DISPATCH,
            assessment=analysis.visual_assessment,
            malformed=analysis.assessment_malformed,
            action_present=analysis.action is not None,
            screen=screen,
            authority=authority,
            foreground=foreground,
            execution=proof.receipt if proof is not None else None,
        )
        return ShadowTurnDraft(
            workflow=workflow_id,
            goal=ShadowGoal(index=active.index, success=active.success),
            observation=ShadowObservation(
                requirement=Eligibility.observation(success=active.success),
                screen=screen,
                assessment=analysis.visual_assessment,
                malformed=analysis.assessment_malformed,
            ),
            action=ShadowAction(proposed=analysis.action),
            application=ShadowApplication(authority=authority, foreground=foreground),
            pre_dispatch=self.__pre_phase(
                success=active.success, candidate=candidate, live=live_pre
            ),
            cursor_before=cursor_before,
            metrics=metrics,
        )

    def finalize_undispatched(
        self, *, draft: ShadowTurnDraft, cursor_after: GoalCursor
    ) -> ShadowTurn:
        """
        Finalize a turn that never dispatched: no execution receipt, no post screen, and no post-dispatch phase.
        """

        return self.__finalize(
            draft=draft, execution=ShadowExecution(), post=None, cursor_after=cursor_after
        )

    def finalize_executed(
        self,
        *,
        draft: ShadowTurnDraft,
        active: GoalState,
        receipt: StepResult,
        live: Advancement,
        screen: Optional[str],
        foreground: Optional[str],
        cursor_after: GoalCursor,
        assessment: Optional[VisualAssessment] = None,
    ) -> ShadowTurn:
        """
        Finalize a successful dispatch: an observed goal reuses the single live vision verdict; a receipt-proving
        goal stays receipt-derived. Both are compared against the live decision on the settled post-action screen.
        """

        candidate = self.__post_candidate(
            active=active,
            receipt=receipt,
            screen=screen,
            foreground=foreground,
            draft=draft,
            assessment=assessment,
        )
        self.__reconcile(active=active, receipt=receipt, live=live, candidate=candidate)
        return self.__finalize(
            draft=draft,
            execution=ShadowExecution(receipt=receipt),
            post=ShadowPostDispatch(
                screen=screen,
                foreground=foreground,
                phase=self.__post_phase(
                    success=active.success,
                    candidate=candidate,
                    live=live,
                    assessed=assessment is not None,
                ),
            ),
            cursor_after=cursor_after,
        )

    def finalize_failed(
        self,
        *,
        draft: ShadowTurnDraft,
        active: GoalState,
        receipt: StepResult,
        live: Advancement,
        screen: Optional[str],
        foreground: Optional[str],
        cursor_after: GoalCursor,
    ) -> ShadowTurn:
        """
        Finalize a failed dispatch: the live path short-circuited, so its decision is never comparable to the candidate.
        """

        candidate = self.__post_candidate(
            active=active, receipt=receipt, screen=screen, foreground=foreground, draft=draft
        )
        self.__reconcile(active=active, receipt=receipt, live=live, candidate=candidate)
        return self.__finalize(
            draft=draft,
            execution=ShadowExecution(receipt=receipt),
            post=ShadowPostDispatch(
                screen=screen,
                foreground=foreground,
                phase=IncomparablePhase(
                    candidate=candidate, live=live, reason=PhaseIncomparability.EXECUTION_FAILED
                ),
            ),
            cursor_after=cursor_after,
        )

    def __post_candidate(
        self,
        *,
        active: GoalState,
        receipt: StepResult,
        screen: Optional[str],
        foreground: Optional[str],
        draft: ShadowTurnDraft,
        assessment: Optional[VisualAssessment] = None,
    ) -> Advancement:
        """
        Compute the post-dispatch candidate from the receipt and, for an observed goal, the real vision verdict.
        """

        return self.__candidate.decide(
            success=active.success,
            phase=ObservationPhase.POST_DISPATCH,
            assessment=assessment,
            malformed=False,
            action_present=False,
            screen=screen,
            authority=draft.application.authority,
            foreground=foreground,
            execution=receipt,
        )

    def __pre_phase(
        self, *, success: Success, candidate: Advancement, live: Advancement
    ) -> ShadowPhase:
        """
        A pre-dispatch phase is comparable only for observed goals, where both decisions read the same screen.
        """

        if isinstance(success, ObservedSuccess):
            return ComparablePhase(candidate=candidate, live=live)
        return IncomparablePhase(
            candidate=candidate, live=live, reason=PhaseIncomparability.EVIDENCE_SOURCE_DIFFERENT
        )

    def __post_phase(
        self, *, success: Success, candidate: Advancement, live: Advancement, assessed: bool
    ) -> ShadowPhase:
        """
        A post-dispatch phase is comparable when the goal proves from a receipt or a real post-action verdict was
        produced; it stays deferred only when the goal needs vision and none could be produced this turn.
        """

        if Eligibility.observation(success=success) is None or assessed:
            return ComparablePhase(candidate=candidate, live=live)
        return IncomparablePhase(
            candidate=candidate, live=live, reason=PhaseIncomparability.VISUAL_EVIDENCE_DEFERRED
        )

    @staticmethod
    def __finalize(
        *,
        draft: ShadowTurnDraft,
        execution: ShadowExecution,
        post: Optional[ShadowPostDispatch],
        cursor_after: GoalCursor,
    ) -> ShadowTurn:
        """
        Assemble the finalized turn from the draft plus post-dispatch facts.
        """

        return ShadowTurn(
            workflow=draft.workflow,
            goal=draft.goal,
            observation=draft.observation,
            action=draft.action,
            application=draft.application,
            metrics=draft.metrics,
            pre_dispatch=draft.pre_dispatch,
            execution=execution,
            post_dispatch=post,
            cursor=ShadowCursor(before=draft.cursor_before, after=cursor_after),
        )

    def __reconcile(
        self,
        *,
        active: GoalState,
        receipt: StepResult,
        live: Advancement,
        candidate: Advancement,
    ) -> None:
        """
        Clear pending proof once the goal advances; otherwise stash a fresh matching command postcondition receipt.
        """

        if live.kind in self.__ADVANCING or candidate.kind in self.__ADVANCING:
            active.progress.proof = None
            return
        if self.__matches_command(active=active, receipt=receipt):
            active.progress.proof = PendingProof(receipt=receipt)

    def __matches_command(self, *, active: GoalState, receipt: StepResult) -> bool:
        """
        Return whether the executed step is the active command's matching successful receipt awaiting its postcondition.
        """

        success = active.success
        if not isinstance(success, CommandSuccess) or success.postcondition is None:
            return False
        return (
            receipt.executed
            and receipt.success
            and receipt.step.requirement == success.requirement
            and self.__matcher.matches(requirement=success.requirement, action=receipt.step.action)
        )
