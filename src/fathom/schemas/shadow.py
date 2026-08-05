from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from fathom.constants.assessment import PhaseComparison, PhaseIncomparability
from fathom.schemas.actions import Action
from fathom.schemas.advancement import Advancement
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.planner import PlannerMetrics
from fathom.schemas.steps import StepResult
from fathom.schemas.success import ObservationRequirement, Success
from fathom.schemas.target import TargetAuthority


class GoalCursor(SealedModel):
    """
    The active sub-goal cursor position at a point in the turn.
    """

    index: int = Field(ge=0, description="Zero-based active sub-goal index.")
    total: int = Field(ge=0, description="Total sub-goals in the decomposition.")


class ShadowGoal(SealedModel):
    """
    The active goal a turn adjudicated, carried as its typed success definition.
    """

    index: int = Field(ge=0, description="Active sub-goal index this turn adjudicated.")
    success: Success = Field(description="The typed success definition of the active goal.")


class ShadowObservation(SealedModel):
    """
    The exact settled screen the assessment observed and the requirement it had to prove.
    """

    requirement: Optional[ObservationRequirement] = Field(
        default=None, description="Observation the assessment had to prove, when one applied."
    )
    screen: NonBlank = Field(description="Identity of the settled screen the assessment observed.")
    assessment: Optional[VisualAssessment] = Field(
        default=None, description="The model's assessment of that screen, or None."
    )
    malformed: bool = Field(description="Whether the assessment payload failed its schema.")


class ShadowAction(SealedModel):
    """
    The action the planner proposed on the same response, if any.
    """

    proposed: Optional[Action] = Field(
        default=None, description="The typed proposed action, or None when none was proposed."
    )

    @property
    def present(self) -> bool:
        """
        Whether an action was proposed on the same response.
        """

        return self.proposed is not None


class ShadowApplication(SealedModel):
    """
    The host-owned target authority and the foreground package on the observed screen.
    """

    authority: TargetAuthority = Field(description="Host-owned authoritative target for the run.")
    foreground: Optional[NonBlank] = Field(
        default=None, description="Foreground package on the observed screen, or None when unknown."
    )


class ComparablePhase(SealedModel):
    """
    A candidate and live decision computed from equivalent evidence on the same screen; divergence is meaningful.
    """

    kind: Literal[PhaseComparison.COMPARABLE] = Field(
        default=PhaseComparison.COMPARABLE, description="Discriminator for the comparable variant."
    )
    candidate: Advancement = Field(description="The shadow advancement candidate for this phase.")
    live: Advancement = Field(description="The actual live advancement decision for this phase.")

    @property
    def diverges(self) -> bool:
        """
        Whether the candidate and the live decision differ across the whole typed advancement.
        """

        return self.candidate != self.live


class IncomparablePhase(SealedModel):
    """
    A candidate and live decision resting on unequal evidence; the pairing carries a typed reason, never divergence.
    """

    kind: Literal[PhaseComparison.INCOMPARABLE] = Field(
        default=PhaseComparison.INCOMPARABLE, description="Discriminator for the incomparable variant."
    )
    candidate: Advancement = Field(description="The shadow advancement candidate for this phase.")
    live: Advancement = Field(description="The actual live advancement decision for this phase.")
    reason: PhaseIncomparability = Field(description="Why the two decisions cannot be compared.")


ShadowPhase = Annotated[
    Union[ComparablePhase, IncomparablePhase], Field(discriminator="kind")
]


class ShadowExecution(SealedModel):
    """
    The executed step correlated to the turn, when an action ran; execution owns no observation provenance.
    """

    receipt: Optional[StepResult] = Field(
        default=None, description="The actual executed step, or None when nothing dispatched."
    )


class ShadowPostDispatch(SealedModel):
    """
    The post-dispatch resulting screen, its foreground, and the post-dispatch comparison phase.
    """

    screen: Optional[NonBlank] = Field(
        default=None, description="Post-dispatch settled-screen identity; None when no post observation."
    )
    foreground: Optional[NonBlank] = Field(
        default=None, description="Post-dispatch foreground package; None when no post observation."
    )
    phase: ShadowPhase = Field(description="Candidate and live decision after execution.")


class ShadowCursor(SealedModel):
    """
    The active cursor before and after the live decision applied.
    """

    before: GoalCursor = Field(description="Cursor before the live decision applied.")
    after: GoalCursor = Field(description="Cursor read from actual state after the live decision.")


class ShadowTurnDraft(SealedModel):
    """
    The pre-dispatch half of a shadow turn built in Analyze and carried to Completion for finalization.
    """

    workflow: NonBlank = Field(description="Workflow identity for the run.")
    goal: ShadowGoal = Field(description="Active goal this turn adjudicated.")
    observation: ShadowObservation = Field(description="Settled pre-dispatch screen and requirement.")
    action: ShadowAction = Field(description="Proposed action for the turn.")
    application: ShadowApplication = Field(description="Target authority and pre-dispatch foreground.")
    pre_dispatch: ShadowPhase = Field(description="Candidate and live decision on the pre-dispatch screen.")
    cursor_before: GoalCursor = Field(description="Cursor before any decision applied.")
    metrics: PlannerMetrics = Field(description="Producer-owned planner metrics.")


class ShadowTurn(SealedModel):
    """
    One finalized shadow turn: pre-dispatch and post-dispatch decisions kept separate by screen and phase.
    """

    workflow: NonBlank = Field(description="Workflow identity for the run.")
    goal: ShadowGoal = Field(description="Active goal this turn adjudicated.")
    observation: ShadowObservation = Field(description="Settled pre-dispatch screen and requirement.")
    action: ShadowAction = Field(description="Proposed action for the turn.")
    application: ShadowApplication = Field(description="Target authority and pre-dispatch foreground.")
    metrics: PlannerMetrics = Field(description="Producer-owned planner metrics.")
    pre_dispatch: ShadowPhase = Field(description="Candidate and live decision on the pre-dispatch screen.")
    execution: ShadowExecution = Field(description="The executed step correlated to the turn.")
    post_dispatch: Optional[ShadowPostDispatch] = Field(
        default=None, description="Post-dispatch screen, foreground, and phase, when an action ran."
    )
    cursor: ShadowCursor = Field(description="Cursor before and after the live decision.")
