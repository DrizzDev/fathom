from __future__ import annotations

from typing import List, Optional, Protocol

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
    Message,
    MessageCursorQuery,
    MessagePage,
    MessageQuery,
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
    ScriptQuery,
    ScriptVersion,
    ScriptVersionQuery,
    SetThreadTitle,
    Task,
    TaskOneQuery,
    TaskQuery,
    Thread,
    ThreadListQuery,
    ThreadPage,
    ThreadQuery,
)


class ActorRepository(Protocol):
    """
    Protocol for actor persistence backends.
    """

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Persist one actor identity.
        """

        ...


class ThreadRepository(Protocol):
    """
    Protocol for thread persistence backends.
    """

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Persist one interaction thread.
        """

        ...

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one tenant-scoped thread.
        """

        ...

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set or replace the thread title.
        """

        ...

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load tenant-scoped threads with SQL-side cursor pagination.
        """

        ...


class MembershipRepository(Protocol):
    """
    Protocol for thread-membership persistence backends.
    """

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one active actor membership in a thread.
        """

        ...


class TaskRepository(Protocol):
    """
    Protocol for task persistence backends.
    """

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Persist one task in a thread.
        """

        ...

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Move one task to a terminal state.
        """

        ...

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        ...

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one tenant-scoped task by identifier.
        """

        ...


class MessageRepository(Protocol):
    """
    Protocol for message persistence backends.
    """

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message in a thread and optional task.
        """

        ...

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace stored message content with a sanitized version and event.
        """

        ...

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load tenant-scoped messages for one thread and optional task.
        """

        ...

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load messages with SQL-side cursor pagination.
        """

        ...


class EventRepository(Protocol):
    """
    Protocol for lifecycle-event persistence backends.
    """

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped events for one thread and optional task.
        """

        ...

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load lifecycle events with SQL-side cursor pagination.
        """

        ...


class ArtifactRepository(Protocol):
    """
    Protocol for artifact-reference persistence backends.
    """

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference and its lifecycle event.
        """

        ...

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts for one thread and optional task.
        """

        ...

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load artifacts with SQL-side cursor pagination.
        """

        ...


class ContextRepository(Protocol):
    """
    Protocol for reference-context persistence backends.
    """

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one reference-based context recipe and its lifecycle event.
        """

        ...

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load tenant-scoped contexts with optional task and purpose filters.
        """

        ...

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load contexts with SQL-side cursor pagination.
        """

        ...


class JobRepository(Protocol):
    """
    Protocol for background-job persistence backends.
    """

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one pending background job and its lifecycle event.
        """

        ...

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim one available job for a worker.
        """

        ...

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Move one claimed job to a terminal state.
        """

        ...

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs for retry.
        """

        ...

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one claimed job for retry after backoff.
        """

        ...

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load tenant-scoped jobs with any combination of optional filters.
        """

        ...


class IdempotencyRepository(Protocol):
    """
    Protocol for idempotent-request persistence backends.
    """

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start an idempotent request and return the active record.
        """

        ...

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Record the terminal state of an idempotent request.
        """

        ...

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one tenant-scoped requests record.
        """

        ...


class PolicyRepository(Protocol):
    """
    Protocol for tenant-policy persistence backends.
    """

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Persist one tenant or workspace policy.
        """

        ...

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one tenant-scoped policy.
        """

        ...


class ScriptRepository(Protocol):
    """
    Protocol for reusable-script persistence backends.
    """

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Persist one reusable script and its version history.
        """

        ...

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load tenant-scoped scripts.
        """

        ...

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one script.
        """

        ...


class CleanupRepository(Protocol):
    """
    Protocol for retention-sweep operations over the interaction store.
    """

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run a retention sweep over the store.
        """

        ...
