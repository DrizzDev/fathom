from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from fathom.constants.collaboration import (
    ArtifactBackend,
    ArtifactKind,
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


class Members(BaseModel):
    """
    Stable membership identifiers for the primary run actors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    requester: str = Field(description="Membership identifier for the requesting actor.")
    responder: str = Field(description="Membership identifier for the responding actor.")


class Run(BaseModel):
    """
    Runtime-neutral description of a conversation-backed execution run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the run.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    task: str = Field(description="Root task identifier for the run.")
    workflow: str = Field(description="Runtime workflow identifier.")
    intent: str = Field(description="User goal for the run.")
    package: Optional[str] = Field(default=None, description="Optional target package reference.")
    requester: ActorInput = Field(description="Actor requesting the run.")
    responder: ActorInput = Field(description="Actor executing or coordinating the run.")
    members: Members = Field(description="Membership identifiers for primary actors.")
    request: str = Field(description="Message identifier for the original request.")
    context: str = Field(description="Context identifier for the initial request recipe.")
    created: datetime = Field(description="Timestamp when the run starts.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical run metadata.",
    )


class Handle(BaseModel):
    """
    Stable identifiers returned after a run is recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the run.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    task: str = Field(description="Root task identifier for the run.")
    workflow: str = Field(description="Runtime workflow identifier.")
    requester: str = Field(description="Actor that requested the run.")
    responder: str = Field(description="Actor executing or coordinating the run.")
    request: str = Field(description="Message identifier for the original request.")
    context: str = Field(description="Context identifier for the initial request recipe.")


class Completion(BaseModel):
    """
    Terminal outcome for a recorded run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    finished: datetime = Field(description="Timestamp when the run finishes.")
    elapsed: int = Field(ge=0, description="Elapsed run duration in milliseconds.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical completion metadata.",
    )


class Step(BaseModel):
    """
    Runtime-neutral task record for a graph step, agent, tool, or sub-agent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable task identifier.")
    tenant: str = Field(description="Tenant that owns the task.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )
    parent: Optional[str] = Field(default=None, description="Optional parent task identifier.")
    root: Optional[str] = Field(default=None, description="Optional root task identifier.")
    origin: Optional[str] = Field(default=None, description="Message that caused the task.")
    actor: Optional[str] = Field(default=None, description="Actor assigned to the task.")
    kind: TaskKind = Field(description="Task category.")
    objective: str = Field(description="Human-readable task objective.")
    reference: Optional[str] = Field(default=None, description="Optional target reference.")
    created: datetime = Field(description="Timestamp when the task starts.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical task metadata.",
    )


class StepCompletion(BaseModel):
    """
    Terminal outcome for a graph step, agent, tool, or sub-agent task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the task.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )
    task: str = Field(description="Task identifier to finish.")
    state: TaskState = Field(description="Terminal task state.")
    code: TaskCode = Field(description="Machine-readable terminal task code.")
    reason: Optional[str] = Field(default=None, description="Human-readable terminal reason.")
    summary: Optional[str] = Field(default=None, description="Task result summary.")
    finished: datetime = Field(description="Timestamp when the task finishes.")
    elapsed: int = Field(ge=0, description="Elapsed task duration in milliseconds.")


class Analysis(BaseModel):
    """
    Auditable model analysis summary for a thread or task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable message identifier for the planning record.")
    tenant: str = Field(description="Tenant that owns the planning record.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    actor: str = Field(description="Actor that produced the planning record.")
    summary: str = Field(description="Short planning summary surfaced as the agent bubble title.")
    evidence: Optional[str] = Field(
        default=None,
        description="Optional narrated observation surfaced as the agent bubble sub-line.",
    )
    step: int = Field(
        ge=1,
        description="One-based ordinal of the step the planning record describes.",
    )
    action: Optional[str] = Field(
        default=None,
        description="Optional action kind the agent planned for this step.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Optional natural-language target the planned action operates on.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional planner confidence between 0 and 1.",
    )
    labels: Tuple[Label, ...] = Field(
        default=(),
        description="Policy labels attached to the planning record.",
    )
    created: datetime = Field(description="Timestamp when the planning record is captured.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical planning metadata.",
    )


class Question(BaseModel):
    """
    Human-in-the-loop question recorded as a conversation message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable message identifier for the question.")
    tenant: str = Field(description="Tenant that owns the question.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    actor: str = Field(description="Actor asking the question.")
    body: JsonValue = Field(description="JSON-safe question body.")
    created: datetime = Field(description="Timestamp when the question is recorded.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical question metadata.",
    )


class Answer(BaseModel):
    """
    Human-in-the-loop answer recorded as a conversation message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable message identifier for the answer.")
    tenant: str = Field(description="Tenant that owns the answer.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    actor: str = Field(description="Actor answering the question.")
    question: str = Field(description="Question message identifier being answered.")
    body: JsonValue = Field(description="JSON-safe answer body.")
    created: datetime = Field(description="Timestamp when the answer is recorded.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical answer metadata.",
    )


class Output(BaseModel):
    """
    Artifact reference produced by a runtime, agent, tool, or graph step.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable artifact identifier.")
    tenant: str = Field(description="Tenant that owns the artifact.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow id for telemetry routing.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    actor: Optional[str] = Field(default=None, description="Actor that produced the artifact.")
    kind: ArtifactKind = Field(description="Artifact category.")
    uri: str = Field(description="Stable artifact location.")
    backend: ArtifactBackend = Field(description="Artifact storage backend.")
    mime: Optional[str] = Field(default=None, description="Optional media type.")
    size: Optional[int] = Field(default=None, ge=0, description="Artifact size in bytes.")
    retention: Optional[str] = Field(default=None, description="Retention class.")
    labels: Tuple[Label, ...] = Field(
        default_factory=tuple,
        description="Policy labels attached to the artifact.",
    )
    created: datetime = Field(description="Timestamp when the artifact is recorded.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical artifact metadata.",
    )


class ScriptOutput(BaseModel):
    """
    Reusable script content exported by runtime execution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable script identifier.")
    tenant: str = Field(description="Tenant that owns the script.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Conversation thread identifier.")
    workflow: Optional[str] = Field(default=None, description="Workflow id for telemetry.")
    task: Optional[str] = Field(default=None, description="Task that produced the script.")
    artifact: Optional[str] = Field(default=None, description="Export artifact identifier.")
    actor: Optional[str] = Field(default=None, description="Actor that produced the script.")
    title: Optional[str] = Field(default=None, description="User-facing script title.")
    format: ScriptFormat = Field(
        default=ScriptFormat.TEXT_PLAIN, description="Script content format."
    )
    status: ScriptStatus = Field(default=ScriptStatus.ACTIVE, description="Script state.")
    content: str = Field(description="Editable script content.")
    source: ScriptVersionSource = Field(
        default=ScriptVersionSource.GENERATED,
        description="Source of this script version.",
    )
    summary: Optional[str] = Field(default=None, description="Change summary for audit.")
    created: datetime = Field(description="Timestamp when the script is recorded.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical script metadata.",
    )


class TelemetryEnvelope(BaseModel):
    """
    Structured telemetry payload emitted by the recorder alongside each
    durable write. Hosts route these to subscribed clients (the existing
    WebSocket gateway) so the chat UI can render live updates without
    polling the conversation HTTP API.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(min_length=1, description="Event type for telemetry routing")
    tenant: str = Field(min_length=1, description="Tenant the event belongs to")
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
            "conversation_id": self.conversation_id,
            "thread_id": self.conversation_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "payload": self.payload,
        }
