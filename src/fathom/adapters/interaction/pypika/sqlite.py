from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path  # noqa: TC003 — runtime type for legacy path constructor
from typing import AsyncGenerator, List, Optional

from fathom.infrastructure.interaction.pypika.sqlite.store import Store
from fathom.infrastructure.interaction.pypika.sqlite.unit import Unit
from fathom.interaction.lifecycle import Lifecycle
from fathom.interfaces.interaction import InteractionPort
from fathom.schemas.configuration import SQLiteInteractionConfiguration
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


class SQLiteInteraction(InteractionPort):
    """
    SQLite adapter implementing the interaction persistence port.
    """

    def __init__(
        self,
        *,
        path: Optional[Path] = None,
        configuration: Optional[SQLiteInteractionConfiguration] = None,
    ) -> None:
        """
        Initialize the adapter from either a typed configuration or a path.

        Either `configuration` (preferred, factory-supplied) or `path` (legacy convenience for tests and CLI fallback) must be provided.
        Passing neither or both raises ValueError so the call site is unambiguous.
        """

        if configuration is not None and path is not None:
            raise ValueError("SQLiteInteraction takes either configuration or path, not both")

        if configuration is None:
            if path is None:
                raise ValueError("SQLiteInteraction requires configuration or path")

            configuration = SQLiteInteractionConfiguration(path=path)

        self.__unit = Unit(configuration=configuration)
        self.__store = Store(unit=self.__unit, lifecycle=Lifecycle())

    async def initialize(self) -> None:
        """
        Realize the SQLite database file and apply pending migrations.

        Delegates to the underlying Unit, which guards re-entry with an
        asyncio lock so concurrent startup calls collapse to one migration.
        """

        await self.__unit.initialize()

    async def aclose(self) -> None:
        """
        Release adapter resources.

        SQLite owns no long-lived process resources here — connections are opened per-session — so the close is a no-op.
        The method exists so the host shutdown path can call it uniformly across backends.
        """

        return None

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open one SQLite transaction boundary for grouped interaction writes.
        """

        async with self.__unit.atomic():
            yield

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Create a durable interaction thread.
        """

        return await self.__store.create_thread(request=request)

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Create an actor identity.
        """

        return await self.__store.create_actor(request=request)

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Add an actor membership to a thread.
        """

        return await self.__store.join_thread(request=request)

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Open a durable unit of work.
        """

        return await self.__store.open_task(request=request)

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Record a message in a thread.
        """

        return await self.__store.record_message(request=request)

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace message content with a sanitized version.
        """

        return await self.__store.sanitize_message(request=request)

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Finish a task with a terminal outcome.
        """

        return await self.__store.finish_task(request=request)

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one tenant-scoped thread.
        """

        return await self.__store.get_thread(query=query)

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set the thread title only when the stored title is null. Idempotent.
        """

        return await self.__store.set_thread_title(request=request)

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        return await self.__store.transition(request=request)

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run a retention sweep across the SQLite interaction store.
        """

        return await self.__store.cleanup(request=request)

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load a cursor-paginated page of tenant-scoped threads.
        """

        return await self.__store.list_threads(query=query)

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        return await self.__store.get_tasks(query=query)

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one tenant-scoped task by identifier.
        """

        return await self.__store.get_task(query=query)

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load tenant-scoped messages for one thread and optional task.
        """

        return await self.__store.get_messages(query=query)

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load a cursor-paginated page of messages.
        """

        return await self.__store.list_messages(query=query)

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped lifecycle events for one thread.
        """

        return await self.__store.get_events(query=query)

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load a cursor-paginated page of lifecycle events.
        """

        return await self.__store.list_events(query=query)

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Link an artifact to a thread and optional task.
        """

        return await self.__store.link_artifact(request=request)

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts for one thread.
        """

        return await self.__store.get_artifacts(query=query)

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load a cursor-paginated page of artifacts.
        """

        return await self.__store.list_artifacts(query=query)

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Create or update a reusable script.
        """

        return await self.__store.save_script(request=request)

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load tenant-scoped scripts.
        """

        return await self.__store.get_scripts(query=query)

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one script.
        """

        return await self.__store.get_script_versions(query=query)

    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Load a cursor-paginated page of scripts ordered by updated timestamp.
        """

        return await self.__store.list_scripts(query=query)

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Save a tenant or workspace policy.
        """

        return await self.__store.save_policy(request=request)

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one tenant-scoped policy.
        """

        return await self.__store.get_policy(query=query)

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Schedule background work.
        """

        return await self.__store.schedule_job(request=request)

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim one available job for a worker.
        """

        return await self.__store.claim_job(request=request)

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Finish a claimed job.
        """

        return await self.__store.finish_job(request=request)

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs.
        """

        return await self.__store.recover_jobs(request=request)

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one claimed job for retry after backoff.
        """

        return await self.__store.reschedule_job(request=request)

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load tenant-scoped jobs.
        """

        return await self.__store.get_jobs(query=query)

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start an idempotent request.
        """

        return await self.__store.begin_request(request=request)

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Record the terminal state of an idempotent request.
        """

        return await self.__store.finish_request(request=request)

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one tenant-scoped idempotency record.
        """

        return await self.__store.get_idempotency(query=query)

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one reference-based context recipe.
        """

        return await self.__store.build_context(request=request)

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load tenant-scoped contexts for one thread.
        """

        return await self.__store.get_contexts(query=query)

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load a cursor-paginated page of contexts.
        """

        return await self.__store.list_contexts(query=query)
