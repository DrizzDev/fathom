from __future__ import annotations

from typing import Tuple

from pydantic import Field

from fathom.schemas.base import SealedModel
from fathom.schemas.base.common import NonBlank
from fathom.schemas.subgoal import SubGoal


class Plan(SealedModel):
    """
    An accepted decomposition: its intent, ordered sub-goals, and the active cursor.
    """

    intent: NonBlank = Field(description="Intent this plan decomposes.")
    goals: Tuple[SubGoal, ...] = Field(min_length=1, description="Ordered accepted sub-goals.")
    cursor: int = Field(ge=0, description="Index of the active sub-goal.")
