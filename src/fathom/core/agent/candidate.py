from __future__ import annotations

from typing import Optional

from fathom.constants.turn.advancement import ObservationPhase
from fathom.core.agent.advancement import AdvancementPolicy
from fathom.core.agent.eligibility import Eligibility
from fathom.schemas.advancement import Advancement
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.completion import ActionEvidence, ClaimEvidence
from fathom.schemas.steps import StepResult
from fathom.schemas.success import Success
from fathom.schemas.target import TargetAuthority
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.visual import VisualEvidence


class ShadowCandidate:
    """
    Compute the shadow advancement candidate for a turn by feeding host-owned visual evidence to AdvancementPolicy.
    """

    def __init__(self, *, policy: Optional[AdvancementPolicy] = None) -> None:
        """
        Bind the candidate to the single advancement authority; it decides, this only assembles evidence.
        """

        self.__policy = policy if policy is not None else AdvancementPolicy()

    def decide(
        self,
        *,
        success: Success,
        phase: ObservationPhase,
        assessment: Optional[VisualAssessment],
        malformed: bool,
        action_present: bool,
        screen: Optional[str],
        authority: TargetAuthority,
        foreground: Optional[str],
        execution: Optional[StepResult] = None,
    ) -> Advancement:
        """
        Assemble the settled-screen evidence for this goal and return the advancement policy's candidate decision.
        """

        return self.__policy.decide(
            success=success,
            evidence=self.__evidence(
                success=success,
                phase=phase,
                assessment=assessment,
                malformed=malformed,
                action_present=action_present,
                screen=screen,
                authority=authority,
                foreground=foreground,
                execution=execution,
            ),
        )

    @staticmethod
    def __evidence(
        *,
        success: Success,
        phase: ObservationPhase,
        assessment: Optional[VisualAssessment],
        malformed: bool,
        action_present: bool,
        screen: Optional[str],
        authority: TargetAuthority,
        foreground: Optional[str],
        execution: Optional[StepResult],
    ) -> TurnEvidence:
        """
        Build a turn's evidence: visual for a visually-proven goal, plus the correlated receipt when present.
        """

        requirement = Eligibility.observation(success=success)
        visual = (
            VisualEvidence(
                observation=requirement,
                assessment=assessment,
                malformed=malformed,
                phase=phase,
                action_present=action_present,
                screen=screen,
                authority=authority,
                foreground=foreground,
            )
            if requirement is not None and assessment is not None and screen is not None
            else None
        )
        return TurnEvidence(
            claim=ClaimEvidence(asserted=False),
            action=ActionEvidence(
                dispatched=execution is not None,
                executed=execution is not None and execution.executed,
            ),
            phase=phase,
            execution=execution,
            observation=requirement,
            visual=visual,
        )
