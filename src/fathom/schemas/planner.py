from __future__ import annotations

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import Field

from fathom.constants.planner import (
    EscalationPath,
    GuardOutcome,
    PlannerEventCategory,
    PlannerEventKind,
)
from fathom.constants.tools import ToolName, TurnMode
from fathom.schemas.base.common import NonBlank, SealedModel
from fathom.schemas.escalation import EscalationReason, StuckSource
from fathom.schemas.supervision import BlockReason


class PlannerMetrics(SealedModel):
    """
    Producer-owned planner metrics for one turn, measured at the vision boundary.
    """

    latency: float = Field(ge=0.0, description="Planner call wall-clock latency in seconds.")
    calls: int = Field(ge=1, description="Planner generate-call count; a produced analysis always ran at least once.")


class GoalRef(SealedModel):
    """
    A minimal reference to the active sub-goal an event concerns.
    """

    index: int = Field(ge=0, description="Active sub-goal index.")


class EscalationEvent(SealedModel):
    """
    A HITL escalation observation: detection, allowance, deferral, or a materialized ASK_USER.
    """

    category: Literal[PlannerEventCategory.ESCALATION] = Field(
        default=PlannerEventCategory.ESCALATION, description="Event family discriminator."
    )
    kind: PlannerEventKind = Field(description="Which escalation event occurred.")
    path: EscalationPath = Field(description="Planner path that produced the event.")
    stuck_source: Optional[StuckSource] = Field(
        default=None, description="Resolved stuck source the gate evaluated, when one was active."
    )
    reason: Optional[EscalationReason] = Field(
        default=None, description="Gate decision reason, when the event carries a decision."
    )
    deferrals: Optional[int] = Field(
        default=None, ge=0, description="Deferral count observed at the event."
    )
    goal: Optional[GoalRef] = Field(default=None, description="Active sub-goal at the event.")


class GuardEvent(SealedModel):
    """
    An action-guard observation: a blocked action or an operator-directive bypass.
    """

    category: Literal[PlannerEventCategory.GUARD] = Field(
        default=PlannerEventCategory.GUARD, description="Event family discriminator."
    )
    kind: PlannerEventKind = Field(description="Whether the action was blocked or the guard bypassed.")
    action: str = Field(description="Human-facing descriptor of the guarded action.")
    block_reason: Optional[BlockReason] = Field(
        default=None, description="Structured block reason, when the action was blocked."
    )
    goal: Optional[GoalRef] = Field(default=None, description="Active sub-goal at the event.")


class ToolScopeEvent(SealedModel):
    """
    The resolved tool scope for the turn.
    """

    category: Literal[PlannerEventCategory.TOOL_SCOPE] = Field(
        default=PlannerEventCategory.TOOL_SCOPE, description="Event family discriminator."
    )
    modes: Tuple[TurnMode, ...] = Field(default=(), description="Resolved turn modes.")
    tools: Tuple[ToolName, ...] = Field(default=(), description="Resolved allowed tool names.")
    goal: Optional[GoalRef] = Field(default=None, description="Active sub-goal at the event.")


class CommandRejectedEvent(SealedModel):
    """
    A parsed command rejected before execution by structural validation.
    """

    category: Literal[PlannerEventCategory.COMMAND_REJECTED] = Field(
        default=PlannerEventCategory.COMMAND_REJECTED, description="Event family discriminator."
    )
    reason: str = Field(description="Validation feedback explaining the rejection.")


PlannerEvent = Annotated[
    Union[EscalationEvent, GuardEvent, ToolScopeEvent, CommandRejectedEvent],
    Field(discriminator="category", description="One typed planner observability event."),
]


class GuardDecision(SealedModel):
    """
    The pure verdict of the action guard: whether to block, the blocked action, why, and the events observed.
    """

    outcome: GuardOutcome = Field(description="Whether the action is allowed or which rule blocked it.")
    action: Optional[NonBlank] = Field(
        default=None, description="Descriptor of the blocked action, when the guard blocked one."
    )
    reason: Optional[NonBlank] = Field(
        default=None, description="Semantic reason for a current-screen-repeat block."
    )
    block: Optional[BlockReason] = Field(
        default=None, description="Structured block reason, when the guard blocked the action."
    )
    events: Tuple[PlannerEvent, ...] = Field(
        default=(), description="Observability events produced while evaluating the guard rules."
    )
