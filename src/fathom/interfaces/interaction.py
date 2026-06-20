from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncContextManager, List, Optional

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
    ScriptListQuery,
    ScriptPage,
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
    ThreadTransition,
)


class InteractionPort(ABC):
    """
    Application-facing persistence contract for interaction state.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Realize persistent state required before the adapter accepts traffic.

        Implementations create schemas, apply pending migrations, and open connection pools.
        Idempotent: safe to call more than once. Hosts invoke this from a startup lifecycle hook so readiness probes can distinguish "store ready" from "store will migrate on first write".
        """

        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """
        Release adapter resources.

        Implementations close any pools, connections, or files held open by the adapter.
        Idempotent: safe to call when no resources were ever realized. Hosts wire this into their shutdown lifecycle so adapter resources are released before the event loop terminates.
        """

        raise NotImplementedError

    @abstractmethod
    def atomic(self) -> AsyncContextManager[None]:
        """
        Open one adapter-managed transaction boundary for grouped writes.
        """

        raise NotImplementedError

    @abstractmethod
    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Create a durable interaction thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Create an actor identity.
        """

        raise NotImplementedError

    @abstractmethod
    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Add an actor membership to a thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Open a durable unit of work.
        """

        raise NotImplementedError

    @abstractmethod
    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Record a message in a thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace message content with a sanitized version.
        """

        raise NotImplementedError

    @abstractmethod
    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Finish a task with a terminal outcome.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one tenant-scoped thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set a thread's title only when the existing title is null. The method is idempotent:
        a non-null stored title is left unchanged and the existing thread is returned. Raises if the thread does not exist.
        """

        raise NotImplementedError

    @abstractmethod
    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run a retention sweep: delete idempotency, terminal jobs, old events, and physically remove soft-deleted entities
        older than the request's per-scope `before` thresholds. Per-scope thresholds of None skip that scope. Returns per-scope deletion counts.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load a cursor-paginated page of tenant-scoped threads.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one tenant-scoped task by identifier.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load tenant-scoped messages for one thread and optional task.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load a cursor-paginated page of messages.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped lifecycle events for one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load a cursor-paginated page of lifecycle events.
        """

        raise NotImplementedError

    @abstractmethod
    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Link an artifact to a thread and optional task.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts for one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load a cursor-paginated page of artifacts.
        """

        raise NotImplementedError

    @abstractmethod
    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Create or update a reusable script.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load tenant-scoped scripts.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one script.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Load a cursor-paginated page of scripts ordered by updated timestamp.
        """

        raise NotImplementedError

    @abstractmethod
    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Save a tenant or workspace policy.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one tenant-scoped policy.
        """

        raise NotImplementedError

    @abstractmethod
    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Schedule background work.
        """

        raise NotImplementedError

    @abstractmethod
    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim one available job for a worker.
        """

        raise NotImplementedError

    @abstractmethod
    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Finish a claimed job.
        """

        raise NotImplementedError

    @abstractmethod
    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs.
        """

        raise NotImplementedError

    @abstractmethod
    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one claimed job for retry after backoff.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load tenant-scoped jobs.
        """

        raise NotImplementedError

    @abstractmethod
    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start an idempotent request and return the active record.
        """

        raise NotImplementedError

    @abstractmethod
    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Record the terminal state of an idempotent request.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one tenant-scoped idempotency record.
        """

        raise NotImplementedError

    @abstractmethod
    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one reference-based context record.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load tenant-scoped contexts for one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load a cursor-paginated page of contexts.
        """

        raise NotImplementedError
