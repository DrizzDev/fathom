from __future__ import annotations

from enum import StrEnum
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.outcomes import ActionOutcome
from fathom.schemas.supervision import BlockReason
from fathom.schemas.tasks import ExecutionTask


class HealingDecisionKind(StrEnum):
    """
    Supported bounded healing decisions.
    """

    ASK_USER = "ask_user"
    TRY_ACTION = "try_action"
    FAIL_BOUNDED = "fail_bounded"
    REQUEST_REPLAN = "request_replan"


class ActionCapability(BaseModel):
    """
    Action capability available to the healing layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Capability name.")
    description: str = Field(description="Capability contract.")


class HealingRequest(BaseModel):
    """
    Bounded request sent to the healing layer after supervision blocks execution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: ExecutionTask = Field(description="Active execution task.")
    screen: ScreenObservation = Field(description="Current screen observation.")
    reason: BlockReason = Field(description="Reason normal execution was blocked.")

    failed: Tuple[ActionOutcome, ...] = Field(
        default_factory=tuple,
        description="Recent failed outcomes relevant to the block.",
    )
    capabilities: Tuple[ActionCapability, ...] = Field(
        default_factory=tuple,
        description="Capabilities available to healing.",
    )


class HealingDecision(BaseModel):
    """
    Bounded healing decision returned by a deterministic or agentic healer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(description="Reason for the healing decision.")
    kind: HealingDecisionKind = Field(description="Healing decision kind.")
    action: Optional[Action] = Field(default=None, description="Alternative action to try.")
