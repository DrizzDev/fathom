from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fathom.constants.collaboration import (
    IdempotencyState,
    JobState,
)
from fathom.core.exceptions import InteractionError
from fathom.interfaces.interaction import InteractionPort
from fathom.schemas.interaction import (
    Actor,
    Artifact,
    ArtifactCursorQuery,
    ArtifactPage,
    ArtifactQuery,
    BeginRequest,
    BuildContext,
    ClaimJob,
    CleanupRequest,
    CleanupResult,
    Context,
    ContextCursorQuery,
    ContextPage,
    ContextQuery,
    CreateActor,
    CreateThread,
    Event,
    EventCursorQuery,
    EventPage,
    EventQuery,
    Execution,
    ExecutionQuery,
    FinishExecution,
    FinishJob,
    FinishRequest,
    FinishTask,
    Idempotency,
    IdempotencyQuery,
    Job,
    JobQuery,
    JoinThread,
    LinkArtifact,
    Membership,
    MembershipQuery,
    Message,
    MessageCursorQuery,
    MessagePage,
    MessageQuery,
    Metadata,
    OpenTask,
    Policy,
    PolicyQuery,
    RecordMessage,
    RecoverJob,
    RescheduleJob,
    Sanitize,
    SavePolicy,
    SaveScript,
    ScheduleJob,
    Script,
    ScriptListQuery,
    ScriptPage,
    ScriptQuery,
    ScriptVersion,
    ScriptVersionQuery,
    SetThreadTitle,
    StartExecution,
    Task,
    TaskOneQuery,
    TaskQuery,
    Thread,
    ThreadListQuery,
    ThreadPage,
    ThreadQuery,
    ThreadTransition,
    Timing,
)


class NoopInteraction(InteractionPort):
    """
    Interaction adapter that swallows writes and returns empty reads.

    Selected when a host explicitly disables conversation recording. Writes
    that carry all fields in their request return synthesized entity echoes
    so callers and the recorder treat them as successful. Writes that depend
    on prior persisted state (sanitize, finish-task, finish-job) raise
    InteractionError because there is no state to read; the recorder's
    failure-suppression layer handles this without crashing the run.

    Reads return None or empty collections.
    """

    async def initialize(self) -> None:
        """
        Realize adapter state. Noop adapter holds no schema or pool.
        """

        return None

    async def aclose(self) -> None:
        """
        Release adapter resources. Noop adapter holds no resources.
        """

        return None

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open a no-op grouped write boundary.
        """

        yield

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Echo a Thread shaped from the request without persisting.
        """

        return Thread(
            title=request.title,
            state=request.state,
            creator=request.creator,
            identity=request.identity,
            metadata=request.metadata,
            timing=Timing(created_at=request.created, updated_at=request.created),
        )

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Echo an Actor shaped from the request without persisting.
        """

        return Actor(
            kind=request.kind,
            name=request.name,
            skills=request.skills,
            runtime=request.runtime,
            external=request.external,
            metadata=request.metadata,
            identity=request.identity,
            timing=Timing(created_at=request.created, updated_at=request.created),
        )

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Echo a Membership shaped from the request without persisting.
        """

        return Membership(
            role=request.role,
            actor=request.actor,
            scope=request.scope,
            thread=request.thread,
            joined_at=request.joined,
            identity=request.identity,
            metadata=request.metadata,
        )

    async def find_membership(self, *, query: MembershipQuery) -> Optional[Membership]:
        """
        Return no membership because the noop store has no persisted state.
        """

        return None

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Echo a Task shaped from the request without persisting.
        """

        return Task(
            kind=request.kind,
            plan=request.plan,
            state=request.state,
            thread=request.thread,
            lineage=request.lineage,
            metadata=request.metadata,
            identity=request.identity,
            execution=request.execution,
            assignment=request.assignment,
            timing=Timing(created_at=request.created, updated_at=request.created),
        )

    async def start_execution(self, *, request: StartExecution) -> Execution:
        """
        Echo an Execution shaped from the request without persisting.
        """

        return Execution(
            outcome=Metadata(),
            state=request.state,
            thread=request.thread,
            intent=request.intent,
            metadata=request.metadata,
            identity=request.identity,
            timing=Timing(
                created_at=request.started,
                updated_at=request.started,
                started_at=request.started,
            ),
        )

    async def finish_execution(self, *, request: FinishExecution) -> Execution:
        """
        Finishing an execution requires prior state; not supported without storage.
        """

        _ = request

        raise InteractionError("Finish execution requires durable storage; noop is not supported")

    async def get_execution(self, *, query: ExecutionQuery) -> Optional[Execution]:
        """
        Return no execution because the noop store has no persisted state.
        """

        _ = query

        return None

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Echo a Message shaped from the request without persisting.
        """

        return Message(
            task=request.task,
            kind=request.kind,
            reply=request.reply,
            thread=request.thread,
            author=request.author,
            content=request.content,
            created_at=request.created,
            audience=request.audience,
            metadata=request.metadata,
            identity=request.identity,
            sequence=request.sequence or 0,
        )

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Sanitization requires the prior message; not supported without state.
        """

        _ = request

        raise InteractionError("Sanitization requires durable storage; noop is not supported")

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Finishing requires the prior task; not supported without state.
        """

        _ = request

        raise InteractionError("Finish task requires durable storage; noop is not supported")

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Always return None; nothing was persisted.
        """

        _ = query

        return None

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Title mutation requires durable storage; not supported by noop.
        """

        _ = request

        raise InteractionError("Set thread title requires durable storage; noop is not supported")

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Lifecycle mutation requires durable storage; not supported by noop.
        """

        _ = request

        raise InteractionError(
            "Thread lifecycle update requires durable storage; noop is not supported"
        )

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Cleanup is a durable-storage operation; the noop adapter has
        nothing to delete and returns empty counts.
        """

        _ = request

        return CleanupResult()

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Always return an empty thread page.
        """

        _ = query

        return ThreadPage(items=(), next=None, total=0)

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Always return None; nothing was persisted.
        """

        _ = query

        return None

    async def recent_task(self, *, query: TaskQuery) -> Optional[Task]:
        """
        Always return None; nothing was persisted.
        """

        _ = query

        return None

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Always return an empty message page.
        """

        _ = query

        return MessagePage(items=(), next=None, total=0)

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Always return an empty event page.
        """

        _ = query

        return EventPage(items=(), next=None, total=0)

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Echo an Artifact shaped from the request without persisting.
        """

        return Artifact(
            uri=request.uri,
            task=request.task,
            kind=request.kind,
            mime=request.mime,
            size=request.size,
            thread=request.thread,
            labels=request.labels,
            backend=request.backend,
            created_at=request.created,
            identity=request.identity,
            producer=request.producer,
            metadata=request.metadata,
            retention=request.retention,
        )

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Always return an empty artifact page.
        """

        _ = query

        return ArtifactPage(items=(), next=None, total=0)

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Echo a Script shaped from the request without persisting.
        """

        return Script(
            revision=1,
            task=request.task,
            title=request.title,
            format=request.format,
            thread=request.thread,
            status=request.status,
            content=request.content,
            identity=request.identity,
            created_by=request.actor,
            updated_by=request.actor,
            artifact=request.artifact,
            metadata=request.metadata,
            timing=Timing(created_at=request.created, updated_at=request.created),
        )

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Always return an empty page; nothing was persisted.
        """

        _ = query

        return ScriptPage(items=(), next=None, total=0)

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Echo a Policy shaped from the request without persisting.
        """

        return Policy(
            name=request.name,
            scope=request.scope,
            region=request.region,
            identity=request.identity,
            metadata=request.metadata,
            governance=request.governance,
            timing=Timing(created_at=request.created, updated_at=request.created),
        )

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Always return None; nothing was persisted.
        """

        _ = query

        return None

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Echo a pending Job shaped from the request without persisting.
        """

        return Job(
            attempts=0,
            task=request.task,
            kind=request.kind,
            thread=request.thread,
            state=JobState.PENDING,
            payload=request.payload,
            identity=request.identity,
            metadata=request.metadata,
            available_at=request.available,
            timing=Timing(created_at=request.created, updated_at=request.created),
        )

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Always return None; no jobs are tracked.
        """

        _ = request

        return None

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Finishing requires the prior job; not supported without state.
        """

        _ = request

        raise InteractionError("Finish job requires durable storage; noop is not supported")

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Always return an empty list; no jobs are tracked.
        """

        _ = request

        return []

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Rescheduling requires the prior job; not supported without state.
        """

        _ = request

        raise InteractionError("Reschedule job requires durable storage; noop is not supported")

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Always return an empty list; no jobs are tracked.
        """

        _ = query

        return []

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Echo an active Idempotency record without persisting.
        """

        return Idempotency(
            key=request.key,
            hash=request.hash,
            tenant=request.tenant,
            created_at=request.created,
            expires_at=request.expires,
            metadata=request.metadata,
            state=IdempotencyState.STARTED,
        )

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Finishing requires the prior idempotency record; not supported.
        """

        _ = request

        raise InteractionError("Finish request requires durable storage; noop is not supported")

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Always return None; nothing was persisted.
        """

        _ = query

        return None

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Echo a Context recipe shaped from the request without persisting.
        """

        return Context(
            task=request.task,
            hash=request.hash,
            model=request.model,
            thread=request.thread,
            budget=request.budget,
            purpose=request.purpose,
            builder=request.builder,
            filters=request.filters,
            created_at=request.created,
            expires_at=request.expires,
            identity=request.identity,
            consumer=request.consumer,
            provider=request.provider,
            metadata=request.metadata,
            references=request.references,
        )

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Always return an empty list; nothing was persisted.
        """

        _ = query

        return []

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Always return an empty context page.
        """

        _ = query

        return ContextPage(items=(), next=None, total=0)
