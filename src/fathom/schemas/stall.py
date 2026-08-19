from __future__ import annotations

from pydantic import Field

from fathom.constants.turn.stall import StallState
from fathom.schemas.base.common import SealedModel


class StallSignal(SealedModel):
    """
    Typed momentum reading over the recent action stream.
    """

    state: StallState = Field(description="Momentum classification of the recent effects.")
    streak: int = Field(ge=0, description="Trailing count of effects that made no progress.")
