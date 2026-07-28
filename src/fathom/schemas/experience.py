from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants import ActionType
from fathom.constants.turn.binding import BindingState
from fathom.schemas.base.common import SealedModel
from fathom.schemas.effect import ActionEffectStatus


class Experience(SealedModel):
    """
    Typed outcome of one executed action, recorded for reinforcement decisions.
    """

    workflow: str = Field(description="Workflow the action ran in.")
    session: str = Field(description="Execution session the action ran in.")
    screen: str = Field(description="Visual hash of the screen the action ran against.")

    action: ActionType = Field(description="Action type that was dispatched.")
    target: str = Field(default="", description="Target the action addressed.")

    executed: bool = Field(description="Whether the command ran without a device error.")
    transitioned: ActionEffectStatus = Field(
        description="Typed effect the action produced on the screen.",
    )
    advanced: bool = Field(
        description="Whether the completion gate advanced the task on this turn.",
    )
    binding: Optional[BindingState] = Field(
        default=None,
        description="Grounding state of the action's target when known.",
    )
