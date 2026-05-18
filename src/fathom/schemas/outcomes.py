from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenDiff


class OutcomeStatus(StrEnum):
    """
    Action outcome after execution and post-screen observation.
    """

    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"
    EFFECTIVE = "EFFECTIVE"
    NO_EFFECT = "NO_EFFECT"


class ActionOutcome(BaseModel):
    """
    Observed effect of an attempted action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action = Field(description="Action that was attempted.")
    status: OutcomeStatus = Field(description="Observed effect status.")

    before: ScreenObservation = Field(description="Screen observation before execution.")
    after: Optional[ScreenObservation] = Field(
        default=None,
        description="Screen observation after execution when available.",
    )

    reason: str = Field(description="Actionable outcome reason.")
    diff: Optional[ScreenDiff] = Field(default=None, description="Screen comparison evidence.")
