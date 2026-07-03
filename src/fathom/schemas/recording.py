from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Dict, Optional, Tuple

from pydantic import Field, JsonValue

from fathom.constants.collaboration import (
    ArtifactBackend,
    ArtifactKind,
    ContextPurpose,
    Label,
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.constants.conversation import EntryKind
from fathom.schemas.conversation import ActorInput
from fathom.schemas.conversation.base import ConversationSchema, ThreadScope


class Members(ConversationSchema):
    """
    Stable membership identifiers for the primary run actors.
    """

    requester: str = Field(description="Membership identifier for the requesting actor.")
    responder: str = Field(description="Membership identifier for the responding actor.")


class RecordingEntity(ConversationSchema):
    """
    Base entity for immutable recorder boundary schemas.
    """


class ConversationScoped(ThreadScope):
    """
    Shared tenant and conversation scope for recorder payloads.
    """


class WorkflowScoped(ConversationScoped):
    """
    Shared scope for recorder payloads that may route through a workflow.
    """

    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )


class Recorded(ConversationScoped):
    """
    Shared creation metadata for conversation-scoped recorder payloads.
    """

    created: datetime = Field(description="Timestamp when the payload is recorded.")

    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical recorder metadata.",
    )


class WorkflowRecord(WorkflowScoped):
    """
    Shared creation metadata for workflow-scoped recorder payloads.
    """

    created: datetime = Field(description="Timestamp when the payload is recorded.")

    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical recorder metadata.",
    )


class Finished(RecordingEntity):
    """
    Shared terminal timing fields for recorder completion payloads.
    """

    finished: datetime = Field(description="Timestamp when the work finishes.")
    elapsed: int = Field(ge=0, description="Elapsed duration in milliseconds.")


class Run(Recorded):
    """
    Runtime-neutral description of a conversation-backed execution run.
    """

    workflow: str = Field(description="Runtime workflow identifier.")

    intent: str = Field(description="User goal for the run.")
    package: Optional[str] = Field(default=None, description="Optional target package reference.")

    requester: ActorInput = Field(description="Actor requesting the run.")
    responder: ActorInput = Field(description="Actor executing or coordinating the run.")

    task: Optional[str] = Field(
        default=None,
        description="Optional pre-reserved root task identifier for the run.",
    )
    execution: Optional[str] = Field(
        default=None,
        description="Optional pre-reserved execution identifier for the run.",
    )
    request: Optional[str] = Field(
        default=None,
        description="Optional pre-reserved message identifier for the original request.",
    )
    members: Optional[Members] = Field(
        default=None,
        description="Optional pre-reserved membership identifiers for primary actors.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional pre-reserved context identifier for the initial request recipe.",
    )


class Handle(ConversationScoped):
    """
    Stable identifiers returned after a run is recorded.
    """

    execution: str = Field(description="Execution identifier for the run.")

    task: str = Field(description="Root task identifier for the run.")
    workflow: str = Field(description="Runtime workflow identifier.")

    requester: str = Field(description="Actor that requested the run.")
    responder: str = Field(description="Actor executing or coordinating the run.")

    request: str = Field(description="Message identifier for the original request.")
    context: str = Field(description="Context identifier for the initial request recipe.")


class Completion(Finished):
    """
    Terminal outcome for a recorded run.
    """

    handle: Handle = Field(description="Stable identifiers for the recorded run.")

    steps: int = Field(ge=0, description="Number of steps executed.")
    result: str = Field(description="Message identifier for the final result.")

    status: str = Field(description="Runtime terminal status.")
    success: bool = Field(description="Whether the run achieved its goal.")

    reason: str = Field(description="Human-readable terminal headline.")
    summary: Optional[str] = Field(
        default=None,
        description="Short human-readable terminal headline shown in the result bubble.",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Optional long-form terminal explanation shown when the bubble is expanded.",
    )
    code: TaskCode = Field(description="Machine-readable terminal task code.")
    error: Optional[str] = Field(default=None, description="Optional execution error.")

    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical completion metadata.",
    )


class Step(WorkflowRecord):
    """
    Runtime-neutral task record for a graph step, agent, tool, or sub-agent.
    """

    id: str = Field(description="Stable task identifier.")

    execution: str = Field(description="Execution identifier that owns the task.")

    root: Optional[str] = Field(default=None, description="Optional root task identifier.")
    origin: Optional[str] = Field(default=None, description="Message that caused the task.")
    parent: Optional[str] = Field(default=None, description="Optional parent task identifier.")

    kind: TaskKind = Field(description="Task category.")
    actor: Optional[str] = Field(default=None, description="Actor assigned to the task.")

    objective: str = Field(description="Human-readable task objective.")
    reference: Optional[str] = Field(default=None, description="Optional target reference.")


class StepCompletion(WorkflowScoped, Finished):
    """
    Terminal outcome for a graph step, agent, tool, or sub-agent task.
    """

    task: str = Field(description="Task identifier to finish.")
    state: TaskState = Field(description="Terminal task state.")
    code: TaskCode = Field(description="Machine-readable terminal task code.")
    summary: Optional[str] = Field(default=None, description="Task result summary.")
    reason: Optional[str] = Field(default=None, description="Human-readable terminal reason.")


class Usage(RecordingEntity):
    """
    Token usage captured for one model call.
    """

    prompt: Optional[int] = Field(default=None, ge=0, description="Prompt token count.")

    total: Optional[int] = Field(default=None, ge=0, description="Total token count.")
    reasoning: Optional[int] = Field(default=None, ge=0, description="Reasoning token count.")
    cached: Optional[int] = Field(default=None, ge=0, description="Cached prompt token count.")
    completion: Optional[int] = Field(default=None, ge=0, description="Completion token count.")


class Metrics(RecordingEntity):
    """
    Runtime measurements captured for one planning step.
    """

    analysis: Optional[int] = Field(
        ge=0,
        default=None,
        description="Analysis duration in milliseconds.",
    )
    grounding: Optional[int] = Field(
        ge=0,
        default=None,
        description="Grounding duration in milliseconds.",
    )
    execution: Optional[int] = Field(
        ge=0,
        default=None,
        description="Device execution duration in milliseconds.",
    )
    total: Optional[int] = Field(
        ge=0,
        default=None,
        description="Total step duration in milliseconds.",
    )
    usage: Optional[Usage] = Field(default=None, description="Optional model token usage.")


class ActionSummary(RecordingEntity):
    """
    User-safe action projection for one planning step.
    """

    type: Optional[str] = Field(default=None, description="Action category.")
    text: Optional[str] = Field(default=None, description="Text submitted by the action.")
    target: Optional[str] = Field(default=None, description="Human-readable action target.")
    rationale: Optional[str] = Field(default=None, description="Reason for choosing the action.")

    confidence: Optional[float] = Field(
        ge=0.0,
        le=1.0,
        default=None,
        description="Planner confidence between 0 and 1.",
    )


class Observation(RecordingEntity):
    """
    User-safe observation projection for one planning step.
    """

    summary: Optional[str] = Field(default=None, description="Post-action observation.")
    evidence: Optional[str] = Field(default=None, description="Screen evidence seen by the model.")

    screen: Optional[str] = Field(default=None, description="Pre-action screen hash.")
    changed: Optional[bool] = Field(default=None, description="Whether the screen changed.")


class Analysis(WorkflowRecord):
    """
    Auditable model analysis summary for a thread or task.
    """

    id: str = Field(description="Stable message identifier for the planning record.")

    actor: str = Field(description="Actor that produced the planning record.")
    execution: str = Field(description="Execution identifier that owns the analysis.")
    task: Optional[str] = Field(default=None, description="Optional task identifier.")

    status: str = Field(default="completed", description="Step outcome status.")
    summary: str = Field(description="Short planning summary surfaced as the agent bubble title.")
    rationale: Optional[str] = Field(default=None, description="Reasoning for the chosen action.")
    observation: Optional[Observation] = Field(
        default=None,
        description="Observation surfaced in the timeline.",
    )

    evidence: Optional[str] = Field(
        default=None,
        description="Optional narrated observation surfaced as the agent bubble sub-line.",
    )
    step: int = Field(
        ge=1,
        description="One-based ordinal of the step the planning record describes.",
    )
    action: Optional[ActionSummary] = Field(
        default=None,
        description="Optional action projection shown in the timeline.",
    )
    metrics: Optional[Metrics] = Field(
        default=None,
        description="Optional runtime and token measurements for audit metadata.",
    )
    labels: Tuple[Label, ...] = Field(
        default=(),
        description="Policy labels attached to the planning record.",
    )


class Question(WorkflowRecord):
    """
    Human-in-the-loop question recorded as a conversation message.
    """

    id: str = Field(description="Stable message identifier for the question.")

    actor: str = Field(description="Actor asking the question.")
    execution: str = Field(description="Execution identifier that owns the question.")
    body: JsonValue = Field(description="JSON-safe question body.")
    task: Optional[str] = Field(default=None, description="Optional task identifier.")


class Answer(WorkflowRecord):
    """
    Human-in-the-loop answer recorded as a conversation message.
    """

    id: str = Field(description="Stable message identifier for the answer.")

    actor: str = Field(description="Actor answering the question.")
    execution: str = Field(description="Execution identifier that owns the answer.")
    task: Optional[str] = Field(default=None, description="Optional task identifier.")

    body: JsonValue = Field(description="JSON-safe answer body.")
    question: str = Field(description="Question message identifier being answered.")


class ContextSnapshot(WorkflowRecord):
    """
    Reference recipe captured for one runtime decision point.
    """

    id: str = Field(description="Stable context snapshot identifier.")

    actor: str = Field(description="Actor that consumed the context.")
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    execution: Optional[str] = Field(default=None, description="Optional execution identifier.")
    hash: Optional[str] = Field(default=None, description="Optional deterministic context hash.")

    model: Optional[str] = Field(default=None, description="Optional model identifier.")
    provider: Optional[str] = Field(default=None, description="Optional model provider identifier.")

    purpose: ContextPurpose = Field(
        default=ContextPurpose.EXECUTION,
        description="Purpose that explains why the context was built.",
    )

    events: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Event identifiers referenced by the context snapshot.",
    )
    messages: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Message identifiers referenced by the context snapshot.",
    )
    artifacts: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Artifact identifiers referenced by the context snapshot.",
    )


class Output(WorkflowRecord):
    """
    Artifact reference produced by a runtime, agent, tool, or graph step.
    """

    id: str = Field(description="Stable artifact identifier.")

    execution: Optional[str] = Field(
        default=None,
        description="Optional execution identifier that owns the artifact.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    actor: Optional[str] = Field(default=None, description="Actor that produced the artifact.")

    kind: ArtifactKind = Field(description="Artifact category.")
    uri: str = Field(description="Stable artifact location.")
    backend: ArtifactBackend = Field(description="Artifact storage backend.")
    mime: Optional[str] = Field(default=None, description="Optional media type.")
    retention: Optional[str] = Field(default=None, description="Retention class.")
    size: Optional[int] = Field(default=None, ge=0, description="Artifact size in bytes.")

    labels: Tuple[Label, ...] = Field(
        default_factory=tuple,
        description="Policy labels attached to the artifact.",
    )


class ScriptOutput(WorkflowRecord):
    """
    Reusable script content exported by runtime execution.
    """

    id: str = Field(description="Stable script identifier.")

    execution: Optional[str] = Field(
        default=None,
        description="Optional execution identifier that owns the script.",
    )
    task: Optional[str] = Field(default=None, description="Task that produced the script.")
    artifact: Optional[str] = Field(default=None, description="Export artifact identifier.")
    actor: Optional[str] = Field(default=None, description="Actor that produced the script.")

    title: Optional[str] = Field(default=None, description="User-facing script title.")

    content: str = Field(description="Editable script content.")
    format: ScriptFormat = Field(
        default=ScriptFormat.TEXT_PLAIN, description="Script content format."
    )

    status: ScriptStatus = Field(default=ScriptStatus.ACTIVE, description="Script state.")

    source: ScriptVersionSource = Field(
        default=ScriptVersionSource.GENERATED,
        description="Source of this script version.",
    )
    summary: Optional[str] = Field(default=None, description="Change summary for audit.")


class TelemetryEnvelope(ConversationSchema):
    """
    Structured telemetry payload emitted by the recorder alongside each durable write.
    Hosts route these to subscribed clients (the existing WebSocket gateway)
    so the chat UI can render live updates without polling the conversation HTTP API.
    """

    tenant: str = Field(min_length=1, description="Tenant the event belongs to")
    type: str = Field(min_length=1, description="Event type for telemetry routing")

    conversation_id: str = Field(
        min_length=1,
        description="Conversation thread the event belongs to",
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Workflow id for routing; absent on graph-internal events",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task id for the event when applicable",
    )

    kind: EntryKind = Field(
        description="Renderable entry kind the event maps to in the timeline",
    )
    payload: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Event-specific fields mirroring the timeline entry payload",
    )

    def as_kwargs(self) -> Dict[str, JsonValue]:
        """
        Flatten the envelope into kwargs suitable for TelemetryPort.info(...).
        """

        return {
            "type": self.type,
            "tenant": self.tenant,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "payload": self.payload,
            "workflow_id": self.workflow_id,
            "thread_id": self.conversation_id,
            "conversation_id": self.conversation_id,
        }
