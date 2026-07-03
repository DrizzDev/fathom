from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field, model_validator

from fathom.schemas.authoring.artifact import AuthoringArtifactReference
from fathom.schemas.base import SealedModel
from fathom.schemas.flow import Evidence, Flow, Report
from fathom.schemas.steps import StepGoal


class AuthoringEpisode(SealedModel):
    """
    Consecutive evidence steps that belong to one recorded sub-goal.
    """

    goal: StepGoal = Field(description="Recorded sub-goal shared by the episode.")
    steps: Tuple[int, ...] = Field(
        min_length=1, description="Evidence step numbers grouped under the goal."
    )


class RunAuthoringEvidence(SealedModel):
    """
    Whole-run evidence view for authoring.
    """

    evidence: Evidence = Field(description="Existing normalized execution evidence.")
    episodes: Tuple[AuthoringEpisode, ...] = Field(
        default_factory=tuple, description="Goal-grouped evidence step episodes."
    )
    artifacts: Tuple[AuthoringArtifactReference, ...] = Field(
        default_factory=tuple, description="Run and step artifacts available to authoring."
    )


class StepAuthoringEvidence(SealedModel):
    """
    Single-step evidence view for authoring.
    """

    evidence: Evidence = Field(description="Existing normalized execution evidence.")
    step_index: int = Field(ge=0, description="Evidence step selected for authoring.")
    artifacts: Tuple[AuthoringArtifactReference, ...] = Field(
        default_factory=tuple, description="Artifacts available for the selected step."
    )

    @model_validator(mode="after")
    def __step_exists(self) -> "StepAuthoringEvidence":
        """
        Ensure the selected step exists in the supplied evidence.
        """

        if all(step.index != self.step_index for step in self.evidence.steps):
            raise ValueError(f"Evidence does not contain step {self.step_index}.")

        return self


class RepairAuthoringEvidence(SealedModel):
    """
    Evidence view for repairing an existing script or flow.
    """

    evidence: Optional[Evidence] = Field(
        default=None, description="Optional execution evidence when repair is run inside Fathom."
    )
    script: Optional[str] = Field(default=None, description="Existing script text to repair.")
    flow: Optional[Flow] = Field(default=None, description="Existing flow to repair.")
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
            self.evidence is None
            and self.script is None
            and self.flow is None
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
