from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field, model_validator

from fathom.constants.state import RunOutcome
from fathom.schemas.authoring.artifact import AuthoringArtifactReference
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.base import SealedModel
from fathom.schemas.flow import (
    CompletionAssertion,
    Evidence,
    Flow,
    Report,
    TargetAnchors,
    TargetClaim,
    TargetStructure,
)
from fathom.schemas.steps import StepGoal


class AuthoringRun(SealedModel):
    """
    Run-level execution truth exposed to the authoring agent.
    """

    intent: str = Field(description="User intent recorded for the run.")
    goal: str = Field(description="Goal-state description recorded for the run.")

    package: str = Field(description="Target application package.")
    outcome: RunOutcome = Field(description="Terminal execution outcome for the run.")
    partial: bool = Field(description="Whether execution evidence is incomplete.")

    reason: Optional[str] = Field(default=None, description="Why the run is partial.")
    discarded: Tuple[int, ...] = Field(
        default_factory=tuple, description="Evidence step numbers discarded before authoring."
    )


class AuthoringEpisode(SealedModel):
    """
    Consecutive evidence steps that belong to one recorded sub-goal.
    """

    goal: StepGoal = Field(description="Recorded sub-goal shared by the episode.")
    steps: Tuple[int, ...] = Field(
        min_length=1, description="Evidence step numbers grouped under the goal."
    )


class AuthoringBaseline(SealedModel):
    """
    Deterministic script scaffold available to whole-run authoring.
    """

    content: str = Field(min_length=1, description="Rendered baseline script text.")
    partial: bool = Field(description="Whether the baseline is intentionally partial.")
    reason: Optional[str] = Field(default=None, description="Why the baseline is partial.")


class AuthoringCommand(SealedModel):
    """
    Executed command facts for one evidence step.
    """

    action: str = Field(description="Recorded action type.")
    event: str = Field(description="Recorded event category.")
    success: bool = Field(description="Whether execution reported success.")


class AuthoringTarget(SealedModel):
    """
    Target signals available for authoring one command.
    """

    name: Optional[str] = Field(default=None, description="Raw visible target name.")
    generalized: Optional[str] = Field(default=None, description="Dynamic target description.")
    export: Optional[str] = Field(default=None, description="Canonical exported target phrase.")

    positional: bool = Field(default=False, description="Whether the target was ordinal.")
    scroll: Optional[str] = Field(default=None, description="Recorded scroll destination.")
    element: Optional[str] = Field(default=None, description="Recorded target element role.")

    anchors: TargetAnchors = Field(
        default_factory=TargetAnchors, description="Evidence anchors grouped by available channel."
    )
    structure: TargetStructure = Field(
        default_factory=TargetStructure, description="Structured UI facts for replay targeting."
    )
    claim: TargetClaim = Field(
        default_factory=TargetClaim, description="Planner target claim and verification status."
    )


class AuthoringNarrative(SealedModel):
    """
    Planner-authored explanation and screen observation for one step.
    """

    reasoning: Optional[str] = Field(default=None, description="Planner reasoning for the step.")
    observation: Optional[str] = Field(default=None, description="Post-step screen observation.")


class AuthoringScreen(SealedModel):
    """
    Screen effect facts recorded for one step.
    """

    changed: bool = Field(description="Whether the screen changed after the command.")
    duration: Optional[int] = Field(
        default=None, ge=0, description="Recorded command duration in milliseconds."
    )


class AuthoringCapture(SealedModel):
    """
    STORE capture request and outcome exposed to authoring.
    """

    name: str = Field(description="Variable name requested for storage.")
    subject: str = Field(description="Subject the planner requested to capture.")

    success: bool = Field(description="Whether the capture succeeded.")
    value: Optional[str] = Field(default=None, description="Captured runtime value.")
    reason: Optional[str] = Field(default=None, description="Capture failure reason.")


class AuthoringValidation(SealedModel):
    """
    Validation claim facts recorded for one step.
    """

    subject: Optional[str] = Field(default=None, description="Recorded validation subject.")
    pattern: Optional[str] = Field(default=None, description="Recorded validation pattern.")


class AuthoringStep(SealedModel):
    """
    Authoring-owned view of one execution step.
    """

    index: int = Field(ge=0, description="Evidence step sequence number.")
    command: AuthoringCommand = Field(description="Recorded command facts.")

    target: AuthoringTarget = Field(description="Target signals.")
    screen: AuthoringScreen = Field(description="Screen effect facts.")
    narrative: AuthoringNarrative = Field(description="Reasoning and observation.")

    text: Optional[str] = Field(default=None, description="Typed text content.")
    goal: Optional[StepGoal] = Field(default=None, description="Active sub-goal for the step.")

    capture: Optional[AuthoringCapture] = Field(default=None, description="STORE capture facts.")
    validation: Optional[AuthoringValidation] = Field(
        default=None, description="Validation facts for the step."
    )
    artifacts: Tuple[AuthoringArtifactReference, ...] = Field(
        default_factory=tuple, description="Artifacts attached to this step."
    )


class RunAuthoringEvidence(SealedModel):
    """
    Whole-run evidence view for authoring.
    """

    run: AuthoringRun = Field(description="Run-level authoring facts.")
    source: Evidence = Field(
        exclude=True,
        description="Normalized execution evidence used for deterministic policy review.",
    )

    steps: Tuple[AuthoringStep, ...] = Field(
        default_factory=tuple, description="Authoring-owned ordered step views."
    )
    episodes: Tuple[AuthoringEpisode, ...] = Field(
        default_factory=tuple, description="Goal-grouped evidence step episodes."
    )

    artifacts: Tuple[AuthoringArtifactReference, ...] = Field(
        default_factory=tuple, description="Run and step artifacts available to authoring."
    )
    drafts: Tuple[AuthoringDraft, ...] = Field(
        default_factory=tuple, description="Step drafts available to whole-run authoring."
    )
    baseline: Optional[AuthoringBaseline] = Field(
        default=None, description="Deterministic baseline scaffold available to improve."
    )
    assertions: Tuple[CompletionAssertion, ...] = Field(
        default_factory=tuple, description="Terminal assertions available to completed scripts."
    )


class StepAuthoringEvidence(SealedModel):
    """
    Single-step evidence view for authoring.
    """

    run: AuthoringRun = Field(description="Run-level context for the selected step.")
    source: Evidence = Field(
        exclude=True,
        description="Normalized execution evidence used for deterministic policy review.",
    )

    step: AuthoringStep = Field(description="Authoring-owned selected step view.")
    step_index: int = Field(ge=0, description="Evidence step selected for authoring.")

    artifacts: Tuple[AuthoringArtifactReference, ...] = Field(
        default_factory=tuple, description="Artifacts available for the selected step."
    )

    @model_validator(mode="after")
    def __step_exists(self) -> "StepAuthoringEvidence":
        """
        Ensure the selected step exists in the supplied evidence.
        """

        if all(step.index != self.step_index for step in self.source.steps):
            raise ValueError(f"Evidence does not contain step {self.step_index}.")

        return self


class RepairAuthoringEvidence(SealedModel):
    """
    Evidence view for repairing an existing script or flow.
    """

    source: Optional[Evidence] = Field(
        default=None,
        exclude=True,
        description="Optional execution evidence used for deterministic policy review.",
    )
    flow: Optional[Flow] = Field(default=None, description="Existing flow to repair.")
    script: Optional[str] = Field(default=None, description="Existing script text to repair.")
    review: Optional[Report] = Field(default=None, description="Review issues guiding repair.")

    artifacts: Tuple[AuthoringArtifactReference, ...] = Field(
        default_factory=tuple, description="Artifacts available to repair authoring."
    )

    @model_validator(mode="after")
    def __has_repair_input(self) -> "RepairAuthoringEvidence":
        """
        Require at least one repair input.
        """

        if (
            self.source is None
            and self.flow is None
            and self.script is None
            and self.review is None
            and not self.artifacts
        ):
            raise ValueError("Repair authoring evidence requires at least one input.")

        return self


class AuthoringEvidence(SealedModel):
    """
    Task-specific evidence view consumed by the authoring agent.
    """

    run: Optional[RunAuthoringEvidence] = Field(default=None, description="Run authoring evidence.")

    step: Optional[StepAuthoringEvidence] = Field(
        default=None, description="Step authoring evidence."
    )
    repair: Optional[RepairAuthoringEvidence] = Field(
        default=None, description="Repair authoring evidence."
    )

    @model_validator(mode="after")
    def __exactly_one_view(self) -> "AuthoringEvidence":
        """
        Require exactly one task evidence view.
        """

        views = [self.run, self.step, self.repair]
        if sum(view is not None for view in views) != 1:
            raise ValueError("AuthoringEvidence requires exactly one evidence view.")

        return self
