from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action
from fathom.schemas.localization import LocalizationResult


class VerdictKind(StrEnum):
    """
    Runtime supervision verdict states.
    """

    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


class BlockReason(StrEnum):
    """
    Machine-readable reason an action cannot execute normally.
    """

    TARGET_UNRESOLVED = "target_unresolved"
    TARGET_AMBIGUOUS = "target_ambiguous"
    REPEATED_NO_EFFECT = "repeated_no_effect"
    KEYBOARD_OCCLUDING = "keyboard_occluding"
    NON_SCROLLABLE_SURFACE = "non_scrollable_surface"
    OVERLAY_STILL_PRESENT = "overlay_still_present"
    TASK_BUDGET_EXCEEDED = "task_budget_exceeded"
    UNSAFE_ACTION = "unsafe_action"


class SupervisionVerdict(BaseModel):
    """
    Decision from the runtime supervisor before action execution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: VerdictKind = Field(description="Verdict discriminator.")
    action: Optional[Action] = Field(default=None, description="Action approved for execution.")
    reason: Optional[BlockReason] = Field(default=None, description="Block or escalation reason.")

    localization: Optional[LocalizationResult] = Field(
        default=None,
        description="Localization evidence used for the verdict.",
    )
    message: str = Field(description="Actionable explanation for the next runtime step.")
