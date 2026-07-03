from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncContextManager, Optional

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
)


class LifecyclePort(ABC):
    """
    Resource lifecycle and transaction contract for interaction adapters.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Realize persistent state required before the adapter accepts traffic.
        """

        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """
        Release adapter resources.
        """

        raise NotImplementedError

    @abstractmethod
    def atomic(self) -> AsyncContextManager[None]:
        """
        Open one adapter-managed transaction boundary for grouped writes.
        """

        raise NotImplementedError


class ThreadPort(ABC):
    """
    Persistence contract for conversation thread rows.
    """

    @abstractmethod
    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Create a durable interaction thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_thread(self, *, query: ThreadQuery) -> Thread | None:
        """
        Load one tenant-scoped thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set a thread title when the stored title is empty.
        """

        raise NotImplementedError

    @abstractmethod
    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load a cursor-paginated page of tenant-scoped threads.
        """

        raise NotImplementedError


class ActorPort(ABC):
    """
    Persistence contract for actor rows.
    """

    @abstractmethod
    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Create an actor identity.
        """

        raise NotImplementedError


class MemberPort(ABC):
    """
    Persistence contract for thread membership rows.
    """

    @abstractmethod
    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Add an actor membership to a thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def find_membership(self, *, query: MembershipQuery) -> Optional[Membership]:
        """
        Load one active actor membership in a thread.
        """

        raise NotImplementedError


class TaskPort(ABC):
    """
    Persistence contract for task rows.
    """

    @abstractmethod
    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Open a durable unit of work.
        """

        raise NotImplementedError

    @abstractmethod
    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Finish a task with a terminal outcome.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_tasks(self, *, query: TaskQuery) -> list[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_task(self, *, query: TaskOneQuery) -> Task | None:
        """
        Load one tenant-scoped task by identifier.
        """

        raise NotImplementedError

    @abstractmethod
    async def recent_task(self, *, query: TaskQuery) -> Task | None:
        """
        Load the most recent non-archived task for one thread, if any.
        """

        raise NotImplementedError

    @abstractmethod
    async def top_root_tasks(self, *, query: TaskQuery, limit: int) -> list[Task]:
        """
        Load the top-N root tasks in a thread, ordered by created_at DESC, using SQL LIMIT.
        """

        raise NotImplementedError

    @abstractmethod
    async def task_descendants(self, *, query: TaskQuery, roots: list[str]) -> list[Task]:
        """
        Load every task whose root points to one of the supplied root ids.
        """

        raise NotImplementedError

    @abstractmethod
    async def task_subtree(self, *, query: TaskQuery, root: str) -> list[Task]:
        """
        Load one subtree rooted at the supplied task.
        """

        raise NotImplementedError


class ExecutionPort(ABC):
    """
    Persistence contract for user intent execution rows.
    """

    @abstractmethod
    async def start_execution(self, *, request: StartExecution) -> Execution:
        """
        Start a durable execution.
        """

        raise NotImplementedError

    @abstractmethod
    async def finish_execution(self, *, request: FinishExecution) -> Execution:
        """
        Finish a durable execution.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_execution(self, *, query: ExecutionQuery) -> Optional[Execution]:
        """
        Load one tenant-scoped execution by identifier.
        """

        raise NotImplementedError


class MessagePort(ABC):
    """
    Persistence contract for conversation message rows.
    """

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
    async def get_messages(self, *, query: MessageQuery) -> list[Message]:
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


class EventPort(ABC):
    """
    Persistence contract for lifecycle event rows.
    """

    @abstractmethod
    async def get_events(self, *, query: EventQuery) -> list[Event]:
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


class ArtifactPort(ABC):
    """
    Persistence contract for artifact rows.
    """

    @abstractmethod
    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Link an artifact to a thread and optional task.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_artifacts(self, *, query: ArtifactQuery) -> list[Artifact]:
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


class ScriptPort(ABC):
    """
    Persistence contract for script rows.
    """

    @abstractmethod
    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Create or update a reusable script.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_scripts(self, *, query: ScriptQuery) -> list[Script]:
        """
        Load tenant-scoped scripts.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_script_versions(self, *, query: ScriptVersionQuery) -> list[ScriptVersion]:
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


class PolicyPort(ABC):
    """
    Persistence contract for governance policy rows.
    """

    @abstractmethod
    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Save a tenant or workspace policy.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_policy(self, *, query: PolicyQuery) -> Policy | None:
        """
        Load one tenant-scoped policy.
        """

        raise NotImplementedError


class JobPort(ABC):
    """
    Persistence contract for durable job rows.
    """

    @abstractmethod
    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Schedule background work.
        """

        raise NotImplementedError

    @abstractmethod
    async def claim_job(self, *, request: ClaimJob) -> Job | None:
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
    async def recover_jobs(self, *, request: RecoverJob) -> list[Job]:
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
    async def get_jobs(self, *, query: JobQuery) -> list[Job]:
        """
        Load tenant-scoped jobs.
        """

        raise NotImplementedError


class RequestPort(ABC):
    """
    Persistence contract for idempotent request rows.
    """

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
    async def get_idempotency(self, *, query: IdempotencyQuery) -> Idempotency | None:
        """
        Load one tenant-scoped idempotency record.
        """

        raise NotImplementedError


class ContextPort(ABC):
    """
    Persistence contract for context rows.
    """

    @abstractmethod
    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one reference-based context record.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_contexts(self, *, query: ContextQuery) -> list[Context]:
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


class CleanupPort(ABC):
    """
    Persistence contract for retention cleanup.
    """

    @abstractmethod
    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run a retention sweep and return per-scope deletion counts.
        """

        raise NotImplementedError


class InteractionPort(
    LifecyclePort,
    ThreadPort,
    ActorPort,
    MemberPort,
    TaskPort,
    ExecutionPort,
    MessagePort,
    EventPort,
    ArtifactPort,
    ScriptPort,
    PolicyPort,
    JobPort,
    RequestPort,
    ContextPort,
    CleanupPort,
    ABC,
):
    """
    Composite adapter compatibility contract for the full interaction store.
    """
