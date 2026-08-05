from __future__ import annotations

from pydantic import Field

from fathom.constants.assessment import ShadowDivergenceKind, VisualVerdict
from fathom.schemas.base.common import NonBlank, SealedModel


class ShadowDivergence(SealedModel):
    """
    One recorded way the shadow visual assessment disagreed with live behavior or the goal's evidence source.
    """

    kind: ShadowDivergenceKind = Field(description="Which kind of divergence was observed.")
    detail: NonBlank = Field(description="Concise explanation of the specific disagreement.")


class VisualAssessment(SealedModel):
    """
    The model's typed judgement of the active visual requirement on the current screenshot.

    Rides the single per-turn VLM response beside the proposed action. The host correlates it to the
    active goal and turn; the model never authors goal, run, or turn identity.
    """

    verdict: VisualVerdict = Field(
        description="Whether the visual requirement holds on this screen."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in the verdict, on a bounded 0..1 scale."
    )
    evidence: NonBlank = Field(
        description="Concise visible evidence the model cites for the verdict."
    )
