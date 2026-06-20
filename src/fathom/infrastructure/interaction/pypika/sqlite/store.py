from __future__ import annotations

from typing import List, Optional

from fathom.infrastructure.interaction.pypika.sqlite.repositories.actors import ActorRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.artifacts import (
    ArtifactRepository,
)
from fathom.infrastructure.interaction.pypika.sqlite.repositories.cleanup import CleanupService
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import (
    StoreContext,
    StoreUnit,
)
from fathom.infrastructure.interaction.pypika.sqlite.repositories.contexts import ContextRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.events import EventRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.idempotency import (
    IdempotencyRepository,
)
from fathom.infrastructure.interaction.pypika.sqlite.repositories.jobs import JobRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.memberships import (
    MembershipRepository,
)
from fathom.infrastructure.interaction.pypika.sqlite.repositories.messages import MessageRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.policies import PolicyRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.script import ScriptRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.tasks import TaskRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.threads import ThreadRepository
from fathom.infrastructure.interaction.pypika.sqlite.row import RowMapper
from fathom.interaction.digest import EventDigest
from fathom.interaction.lifecycle import Lifecycle
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

__all__ = ["Store", "StoreUnit"]


class Store:
    """
    Repository facade for durable interaction entities.

    Composes per-aggregate repositories over a shared StoreContext so each
    aggregate's persistence logic lives in its own module, while the public
    surface stays identical to what adapters and tests expect.
    """

    def __init__(self, *, unit: StoreUnit, lifecycle: Lifecycle) -> None:
        """
        Wire the shared store context and one repository per aggregate.
        """

        context = StoreContext(
            unit=unit,
            lifecycle=lifecycle,
            rows=RowMapper(),
            event_digest=EventDigest(),
        )
        self.__threads = ThreadRepository(context=context)
        self.__actors = ActorRepository(context=context)
        self.__memberships = MembershipRepository(context=context)
        self.__tasks = TaskRepository(context=context)
        self.__messages = MessageRepository(context=context)
        self.__events = EventRepository(context=context)
        self.__artifacts = ArtifactRepository(context=context)
        self.__scripts = ScriptRepository(context=context)
        self.__policies = PolicyRepository(context=context)
        self.__jobs = JobRepository(context=context)
        self.__contexts = ContextRepository(context=context)
        self.__requests = IdempotencyRepository(context=context)
        self.__cleanup = CleanupService(context=context)

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Persist one interaction thread.
        """

        return await self.__threads.create_thread(request=request)

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one tenant-scoped thread.
        """

        return await self.__threads.get_thread(query=query)

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set the thread title only when the stored title is currently null.
        """

        return await self.__threads.set_thread_title(request=request)

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        return await self.__threads.transition(request=request)

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load tenant-scoped threads with SQL-side cursor pagination.
        """

        return await self.__threads.list_threads(query=query)

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Persist one actor identity.
        """

        return await self.__actors.create_actor(request=request)

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one active actor membership in a thread.
        """

        return await self.__memberships.join_thread(request=request)

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Persist one task in a thread.
        """

        return await self.__tasks.open_task(request=request)

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Move one task to a terminal state.
        """

        return await self.__tasks.finish_task(request=request)

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        return await self.__tasks.get_tasks(query=query)

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one tenant-scoped task by identifier.
        """

        return await self.__tasks.get_task(query=query)

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message in a thread and optional task.
        """

        return await self.__messages.record_message(request=request)

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace stored message content with a sanitized version and event.
        """

        return await self.__messages.sanitize_message(request=request)

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load tenant-scoped messages for one thread and optional task.
        """

        return await self.__messages.get_messages(query=query)

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load messages with SQL-side cursor pagination.
        """

        return await self.__messages.list_messages(query=query)

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped events for one thread and optional task.
        """

        return await self.__events.get_events(query=query)

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load lifecycle events with SQL-side cursor pagination.
        """

        return await self.__events.list_events(query=query)

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference and its lifecycle event.
        """

        return await self.__artifacts.link_artifact(request=request)

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts for one thread and optional task.
        """

        return await self.__artifacts.get_artifacts(query=query)

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load artifacts with SQL-side cursor pagination.
        """

        return await self.__artifacts.list_artifacts(query=query)

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Persist one reusable script and its version history.
        """

        return await self.__scripts.save_script(request=request)

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load tenant-scoped scripts.
        """

        return await self.__scripts.get_scripts(query=query)

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one script.
        """

        return await self.__scripts.get_script_versions(query=query)

    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Load scripts with SQL-side cursor pagination ordered by updated timestamp.
        """

        return await self.__scripts.list_scripts(query=query)

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Persist one tenant or workspace policy.
        """

        return await self.__policies.save_policy(request=request)

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one tenant-scoped policy.
        """

        return await self.__policies.get_policy(query=query)

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one pending background job and its lifecycle event.
        """

        return await self.__jobs.schedule_job(request=request)

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim one available job for a worker.
        """

        return await self.__jobs.claim_job(request=request)

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Move one claimed job to a terminal state.
        """

        return await self.__jobs.finish_job(request=request)

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs for retry.
        """

        return await self.__jobs.recover_jobs(request=request)

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one claimed job for retry after backoff.
        """

        return await self.__jobs.reschedule_job(request=request)

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load tenant-scoped jobs with any combination of optional filters.
        """

        return await self.__jobs.get_jobs(query=query)

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one reference-based context recipe and its lifecycle event.
        """

        return await self.__contexts.build_context(request=request)

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load tenant-scoped contexts with optional task and purpose filters.
        """

        return await self.__contexts.get_contexts(query=query)

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load contexts with SQL-side cursor pagination.
        """

        return await self.__contexts.list_contexts(query=query)

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start an idempotent request and return the active record.
        """

        return await self.__requests.begin_request(request=request)

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Record the terminal state of an idempotent request.
        """

        return await self.__requests.finish_request(request=request)

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one tenant-scoped requests record.
        """

        return await self.__requests.get_idempotency(query=query)

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run a retention sweep over the store.
        """

        return await self.__cleanup.cleanup(request=request)
