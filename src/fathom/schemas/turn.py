from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.turn.advancement import ObservationPhase
from fathom.schemas.base.common import SealedModel
from fathom.schemas.binding import Binding
from fathom.schemas.completion import ActionEvidence, ClaimEvidence
from fathom.schemas.criterion import Verdict
from fathom.schemas.effect import EffectReading
from fathom.schemas.stall import StallSignal
from fathom.schemas.steps import StepResult
from fathom.schemas.success import ObservationRequirement
from fathom.schemas.validation import Validation
from fathom.schemas.visual import VisualEvidence


class TurnEvidence(SealedModel):
    """
    Measured facts from one turn, carried for the advancement decider.
    """

    claim: ClaimEvidence = Field(description="The model's completion claim this turn.")
    action: ActionEvidence = Field(description="Dispatch and execution facts for the action.")

    phase: ObservationPhase = Field(
        description="Whether this turn's observation was read before or after dispatch.",
    )
    execution: Optional[StepResult] = Field(
        default=None,
        description="Correlated executed step; absent during a pre-dispatch probe, present after execution.",
    )
    observation: Optional[ObservationRequirement] = Field(
        default=None,
        description="The exact observation identity the oracle adjudicated this turn, when one was posed.",
    )

    binding: Optional[Binding] = Field(
        default=None,
        description="Typed grounding result for the action's spatial target.",
    )
    effect: Optional[EffectReading] = Field(
        default=None,
        description="Direction-aware effect facets observed after the action.",
    )
    verdict: Optional[Verdict] = Field(
        default=None,
        description="Oracle reading of the task criterion on the settled screen.",
    )
    validation: Optional[Validation] = Field(
        default=None,
        description="Canonical validation assertion executed this turn.",
    )
    stall: Optional[StallSignal] = Field(
        default=None,
        description="Momentum reading over the recent effect stream.",
    )
    visual: Optional[VisualEvidence] = Field(
        default=None,
        description="Host-owned settled-screen visual evidence; drives the shadow candidate, absent on the live path.",
    )
