from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Dict, Optional, Tuple

from pydantic import Field, JsonValue

from fathom.constants.conversation import RunState
from fathom.schemas.conversation.base import (
    ConversationAliasSchema,
    ConversationSchema,
    SummaryBodySchema,
)
from fathom.schemas.conversation.views import (
    ExecutionReference,
    RuntimeReference,
    ThreadView,
    WorkflowReference,
)


class SummaryBody(SummaryBodySchema):
    """
    Base model for permissive summary source message bodies.
    """


class IntentBody(SummaryBody):
    """
    Typed view of a request message body that carries a crawler intent.
    """

    intent: Optional[str] = Field(default=None, description="Intent text from the caller.")
    package: Optional[str] = Field(default=None, description="Target package identifier.")

    starting_package: Optional[str] = Field(default=None, description="Optional launching package.")


class ResultBody(SummaryBody):
    """
    Typed view of a result message body that carries the terminal outcome.
    """

    status: Optional[str] = Field(default=None, description="Terminal run status.")
    summary: Optional[str] = Field(default=None, description="Short headline summary.")

    reason: Optional[str] = Field(default=None, description="Legacy reason text.")
    detail: Optional[str] = Field(default=None, description="Optional long-form detail.")


class ActionOverview(ConversationAliasSchema):
    """
    Human-facing action planned for one progress milestone.
    """

    type: Optional[str] = Field(default=None, description="Planned action type.")
    target: Optional[str] = Field(default=None, description="Planned action target.")
    rationale: Optional[str] = Field(default=None, description="Reason the action was selected.")
    confidence: Optional[float] = Field(
        ge=0.0,
        le=1.0,
        default=None,
        description="Planner confidence for the selected action; audit-only.",
    )


class ObservationOverview(ConversationAliasSchema):
    """
    Human-facing screen observation for one progress milestone.
    """

    summary: Optional[str] = Field(default=None, description="Visible outcome after the action.")
    evidence: Optional[str] = Field(
        default=None, description="Planner evidence for the observation."
    )
    screen: Optional[str] = Field(
        default=None, description="Screen hash or identifier when known; audit-only."
    )
    changed: Optional[bool] = Field(
        default=None, description="Whether the screen changed after action; audit-only."
    )


class ProgressBody(SummaryBody):
    """
    Typed view of a progress message body that carries one planning beat.
    """

    step: Optional[int] = Field(default=None, description="One-based step ordinal.")
    status: Optional[str] = Field(default=None, description="Progress status for this step.")
    action: Optional[Dict[str, JsonValue]] = Field(
        default=None,
        description="Raw planned-action dict from the stored body.",
    )
    analysis: Optional[str] = Field(default=None, description="Planner analysis for this step.")
    rationale: Optional[str] = Field(default=None, description="Reason the action was selected.")
    observation: Optional[Dict[str, JsonValue]] = Field(
        default=None,
        description="Raw observation dict from the stored body.",
    )
    summary: Optional[str] = Field(default=None, description="Step summary.")


class IntentPackages(ConversationSchema):
    """
    Package identifiers surfaced with a crawler intent.
    """

    target: Optional[str] = Field(
        default=None, description="Target package the intent runs against."
    )
    initial: Optional[str] = Field(
        default=None, description="Optional package the run launched from."
    )


class IntentOverview(ConversationAliasSchema):
    """
    Headline projection of one crawler-intent message.
    """

    text: Optional[str] = Field(default=None, description="Intent text as authored by the caller.")
    packages: Optional[IntentPackages] = Field(
        default=None,
        description="Package identifiers associated with the intent.",
    )
    recorded: Optional[datetime] = Field(
        default=None,
        serialization_alias="recorded_at",
        description="Timestamp when the intent message was recorded.",
    )


class OutcomeOverview(ConversationAliasSchema):
    """
    Headline projection of one result message.
    """

    detail: Optional[str] = Field(default=None, description="Optional long-form detail.")
    summary: Optional[str] = Field(default=None, description="Short headline for the bubble.")
    status: Optional[str] = Field(default=None, description="Terminal run status, when present.")

    recorded: Optional[datetime] = Field(
        default=None,
        serialization_alias="recorded_at",
        description="Timestamp when the result message was recorded.",
    )


class MilestoneOverview(ConversationAliasSchema):
    """
    One per-step planning beat projected from a progress message.
    """

    step: int = Field(ge=1, description="One-based step ordinal inside the run.")
    status: Optional[str] = Field(default=None, description="Progress status for this step.")
    action: Optional[ActionOverview] = Field(default=None, description="Planned action details.")
    analysis: Optional[str] = Field(
        default=None,
        description="Planner analysis for this step; audit-only.",
    )
    observation: Optional[ObservationOverview] = Field(
        default=None,
        description="Visible result observed for this step.",
    )
    summary: Optional[str] = Field(default=None, description="Planner-supplied step summary.")

    recorded: datetime = Field(
        serialization_alias="recorded_at",
        description="Timestamp when the progress message was recorded.",
    )


class ScriptOverview(ConversationAliasSchema):
    """
    Pointer to the latest saved script for one run.
    """

    id: str = Field(description="Stable script identifier.")
    title: Optional[str] = Field(default=None, description="User-facing script title.")

    size: int = Field(ge=0, description="Latest content size in bytes.")
    revision: int = Field(ge=1, description="Latest script revision number.")

    updated: datetime = Field(
        serialization_alias="updated_at",
        description="Timestamp when the script was last updated.",
    )


class RunOverview(ConversationAliasSchema):
    """
    One per-intent run summary inside a conversation.
    """

    task: str = Field(description="Run-root task identifier.")
    execution: ExecutionReference = Field(description="Execution that owns the run.")
    workflow: Optional[WorkflowReference] = Field(
        default=None,
        description="Runtime workflow that produced the run, when known.",
    )
    intent: IntentOverview = Field(description="Intent block for this run.")
    state: RunState = Field(description="Run lifecycle state surfaced to the client.")

    outcome: OutcomeOverview = Field(description="Outcome block for this run.")
    milestones: Tuple[MilestoneOverview, ...] = Field(
        default=(),
        description="Per-step planning beats for this run, newest first.",
    )

    script: Optional[ScriptOverview] = Field(
        default=None,
        description="Saved script for this run, when one exists.",
    )

    started: datetime = Field(
        serialization_alias="started_at",
        description="Timestamp when the run's request was recorded.",
    )
    completed: Optional[datetime] = Field(
        default=None,
        serialization_alias="completed_at",
        description="Timestamp when the run's result was recorded; null when still running.",
    )
    updated: datetime = Field(
        serialization_alias="updated_at",
        description="Last activity timestamp for this run.",
    )


class OverviewBlock(ConversationAliasSchema):
    """
    Header-level aggregates for the conversation overview tab.
    """

    status: Optional[RunState] = Field(
        default=None,
        description="State of the most recent run, when at least one run exists.",
    )
    activity: Optional[datetime] = Field(
        default=None,
        serialization_alias="activity_at",
        description="Most recent run activity across the conversation.",
    )
    digest: Optional[str] = Field(
        default=None,
        description="Optional auto-generated narrative digest.",
    )


class OverviewCounts(ConversationSchema):
    """
    Coarse counts used by the overview tab.
    """

    runs: int = Field(ge=0, description="Total runs in the conversation.")
    scripts: int = Field(ge=0, description="Total scripts in the conversation.")
    messages: int = Field(ge=0, description="Total messages in the conversation.")
    artifacts: int = Field(ge=0, description="Total artifacts in the conversation.")


class SummaryView(ConversationAliasSchema):
    """
    Composite overview projection for the conversation overview tab.
    """

    thread: ThreadView = Field(
        serialization_alias="conversation",
        description="Underlying thread metadata.",
    )
    runtime: Optional[RuntimeReference] = Field(
        default=None,
        description="Latest execution and workflow for the conversation.",
    )
    overview: OverviewBlock = Field(description="Header-level aggregates derived from the runs.")
    counts: OverviewCounts = Field(description="Coarse counts of conversation child collections.")
    runs: Tuple[RunOverview, ...] = Field(
        default=(), description="Per-intent run summaries, newest first."
    )
