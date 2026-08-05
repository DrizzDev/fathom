from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.completion import RetainReason
from fathom.constants.turn.advancement import AdvanceKind
from fathom.schemas.base.common import SealedModel


class Advancement(SealedModel):
    """
    The advancement policy's structured decision for one turn.
    """

    kind: AdvanceKind = Field(description="Decision family for the turn.")
    reason: Optional[RetainReason] = Field(
        default=None,
        description="Diagnostic present only when the task is retained.",
    )
    redispatch: bool = Field(
        default=True,
        description=(
            "Whether re-dispatching the task's side-effecting action is safe; False while a "
            "durable outcome awaits proof."
        ),
    )
