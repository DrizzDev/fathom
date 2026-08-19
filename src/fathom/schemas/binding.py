from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field

from fathom.constants.turn.binding import BindingOrigin, BindingState
from fathom.schemas.actions import Bounds
from fathom.schemas.base.common import SealedModel


class Binding(SealedModel):
    """
    Typed grounding result for one spatial action target.
    """

    state: BindingState = Field(description="How firmly the target is grounded.")
    origin: Optional[BindingOrigin] = Field(
        default=None,
        description="Perception channel that produced the geometry; absent when MISSING.",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Grounding confidence for the bound geometry.",
    )
    bounds: Optional[Bounds] = Field(
        default=None,
        description="Geometry of the element that should receive the action; absent when MISSING.",
    )
    anchor: Optional[str] = Field(
        default=None,
        description="Identifier of the interactive element the target anchors to.",
    )
    evidence: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Provenance strings explaining the grounding decision.",
    )
