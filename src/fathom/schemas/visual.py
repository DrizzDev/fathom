from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.turn.advancement import ObservationPhase
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.success import ObservationRequirement
from fathom.schemas.target import TargetAuthority


class VisualEvidence(SealedModel):
    """
    Host-owned settled-screen visual evidence for the advancement decider; the model authors only the assessment.
    """

    observation: ObservationRequirement = Field(
        description="Host-selected observation the assessment must prove this turn."
    )
    assessment: Optional[VisualAssessment] = Field(
        default=None, description="The model's visual assessment, or None when it produced none."
    )
    malformed: bool = Field(
        default=False, description="Whether the assessment payload failed its schema."
    )
    phase: ObservationPhase = Field(
        description="Whether the settled screen was read before or after dispatch."
    )
    action_present: bool = Field(
        description="Whether the same planner response proposed an action."
    )
    screen: NonBlank = Field(description="Identity of the settled screen the assessment read.")
    authority: TargetAuthority = Field(
        description="Host-owned authoritative target for the run, bound or unbound."
    )
    foreground: Optional[str] = Field(
        default=None, description="Foreground package on the settled screen, or None when unknown."
    )
