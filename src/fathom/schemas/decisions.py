from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action


class DecisionKind(StrEnum):
    """
    Supported decisions returned by the reasoning layer.
    """

    ACT = "act"
    DONE = "done"
    ASK_USER = "ask_user"
    REQUEST_REPLAN = "request_replan"
    REPORT_UNACTIONABLE = "report_unactionable"


class ActDecision(BaseModel):
    """
    Decision to execute one UI action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[DecisionKind.ACT] = Field(
        default=DecisionKind.ACT,
        description="Decision discriminator.",
    )
    rationale: str = Field(description="Brief reason for the action.")
    action: Action = Field(description="Semantic action requested by the model.")


class AskUserDecision(BaseModel):
    """
    Decision to ask the user for guidance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[DecisionKind.ASK_USER] = Field(
        default=DecisionKind.ASK_USER,
        description="Decision discriminator.",
    )
    reason: str = Field(description="Reason user input is required.")
    question: str = Field(description="Question to show to the user.")


class ReplanDecision(BaseModel):
    """
    Decision to request runtime replanning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[DecisionKind.REQUEST_REPLAN] = Field(
        default=DecisionKind.REQUEST_REPLAN,
        description="Decision discriminator.",
    )
    reason: str = Field(description="Reason the current execution plan no longer applies.")


class UnactionableDecision(BaseModel):
    """
    Decision that the visible screen cannot satisfy the active task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[DecisionKind.REPORT_UNACTIONABLE] = Field(
        default=DecisionKind.REPORT_UNACTIONABLE, description="Decision discriminator."
    )
    reason: str = Field(description="Reason the current screen is not actionable.")


class DoneDecision(BaseModel):
    """
    Decision that the task appears complete.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[DecisionKind.DONE] = Field(
        default=DecisionKind.DONE,
        description="Decision discriminator.",
    )
    reason: str = Field(description="Reason the task appears complete.")


Decision = Annotated[
    Union[
        ActDecision,
        DoneDecision,
        ReplanDecision,
        AskUserDecision,
        UnactionableDecision,
    ],
    Field(discriminator="kind"),
]


class DecisionEnvelope(BaseModel):
    """
    Parsed reasoning-layer decision with screen and task context.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    screen: str = Field(description="Model description of the current screen.")
    reasoning: str = Field(description="Concise reasoning supplied by the model.")
    decision: Decision = Field(description="Structured decision returned by the model.")
    task: Optional[str] = Field(default=None, description="Active execution task identifier.")
