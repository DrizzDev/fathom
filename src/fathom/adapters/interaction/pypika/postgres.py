from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fathom.core.exceptions import InteractionError, ThreadConflictError
from fathom.infrastructure.interaction.pypika.postgres.row import PostgresRowMapper
from fathom.infrastructure.interaction.pypika.postgres.store import PostgresStore
from fathom.infrastructure.interaction.pypika.postgres.unit import Unit
from fathom.interaction.lifecycle import Lifecycle
from fathom.interfaces.interaction import InteractionPort
from fathom.schemas.configuration import PostgresInteractionConfiguration
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


class PostgresInteraction(InteractionPort):
    """
    Postgres adapter implementing the interaction persistence port.

    Wraps the native Postgres repository facade. Lifecycle, idempotent
    replay, lease checks, cursor pagination, and cleanup semantics live inside the repositories;
    this adapter only handles port-shape translation and Postgres-specific integrity-error handling for the create/join paths.
    """

    def __init__(self, *, configuration: PostgresInteractionConfiguration) -> None:
        """
        Wire the native Postgres unit, row mapper, and store facade.
        """

        self.__rows = PostgresRowMapper()
        self.__unit = Unit(configuration=configuration)
        self.__store = PostgresStore(unit=self.__unit, lifecycle=Lifecycle())

    async def initialize(self) -> None:
        """
        Open the asyncpg pool and run pending migrations under advisory lock.

        Delegates to the underlying Unit, which guards re-entry so concurrent startup calls collapse to one schema bootstrap.
        """

        await self.__unit.initialize()

    async def aclose(self) -> None:
        """
        Close the underlying asyncpg pool.
        """

        await self.__unit.close()

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open one Postgres transaction boundary for grouped interaction writes.
        """

        async with self.__unit.atomic():
            yield

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Create one thread, mapping duplicate identities to ThreadConflictError.
        """

        try:
            return await self.__store.create_thread(request=request)
        except InteractionError as exception:
            if not self.__is_duplicate_integrity_error(exception=exception):
                raise

            existing = await self.get_thread(
                query=ThreadQuery(
                    thread=request.identity.id,
                    tenant=request.identity.tenant,
                )
            )
            if existing is None:
                raise

            raise ThreadConflictError(
                thread=request.identity.id,
                message="Thread identity was concurrently created.",
            ) from exception

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Persist one actor, returning the existing row on idempotent replay.
        """

        try:
            return await self.__store.create_actor(request=request)
        except InteractionError as exception:
            if not self.__is_duplicate_integrity_error(exception=exception):
                raise

            existing = await self.__actor(
                actor=request.identity.id,
                tenant=request.identity.tenant,
            )

            if existing is None or not self.__same_actor(actor=existing, request=request):
                raise

            return existing

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one membership, returning the existing row on idempotent replay.
        """

        try:
            return await self.__store.join_thread(request=request)
        except InteractionError as exception:
            if not self.__is_duplicate_integrity_error(exception=exception):
                raise

            existing = await self.__membership(
                tenant=request.identity.tenant,
                membership=request.identity.id,
            )
            if existing is None or not self.__same_membership(
                request=request,
                membership=existing,
            ):
                raise

            return existing

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Persist one task opened inside a thread.
        """

        return await self.__store.open_task(request=request)

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message and its timeline event.
        """

        return await self.__store.record_message(request=request)

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace message content with a sanitized version.
        """

        return await self.__store.sanitize_message(request=request)

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Move one task into a terminal state.
        """

        return await self.__store.finish_task(request=request)

    async def get_thread(self, *, query: ThreadQuery) -> Thread | None:
        """
        Load one tenant-scoped thread.
        """

        return await self.__store.get_thread(query=query)

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set a thread title only when the stored title is null.
        """

        return await self.__store.set_thread_title(request=request)

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        return await self.__store.transition(request=request)

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run a retention cleanup sweep.
        """

        return await self.__store.cleanup(request=request)

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Return a cursor-paginated page of tenant-scoped threads.
        """

        return await self.__store.list_threads(query=query)

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tasks for one tenant and optional thread/task filter.
        """

        return await self.__store.get_tasks(query=query)

    async def get_task(self, *, query: TaskOneQuery) -> Task | None:
        """
        Load one tenant-scoped task.
        """

        return await self.__store.get_task(query=query)

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load messages for one tenant and optional thread/task filter.
        """

        return await self.__store.get_messages(query=query)

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Return a cursor-paginated page of tenant-scoped messages.
        """

        return await self.__store.list_messages(query=query)

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load events for one tenant and optional thread/task filter.
        """

        return await self.__store.get_events(query=query)

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Return a cursor-paginated page of tenant-scoped lifecycle events.
        """

        return await self.__store.list_events(query=query)

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference and its timeline event.
        """

        return await self.__store.link_artifact(request=request)

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load artifacts for one tenant and optional thread/task filter.
        """

        return await self.__store.get_artifacts(query=query)

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Return a cursor-paginated page of tenant-scoped artifacts.
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
        Persist one policy document.
        """

        return await self.__store.save_policy(request=request)

    async def get_policy(self, *, query: PolicyQuery) -> Policy | None:
        """
        Load one policy by tenant and scope.
        """

        return await self.__store.get_policy(query=query)

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one pending background job.
        """

        return await self.__store.schedule_job(request=request)

    async def claim_job(self, *, request: ClaimJob) -> Job | None:
        """
        Claim the next available job; SKIP LOCKED semantics live in the repository.
        """

        return await self.__store.claim_job(request=request)

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Move one claimed job into a terminal state.
        """

        return await self.__store.finish_job(request=request)

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs for retry.
        """

        return await self.__store.recover_jobs(request=request)

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one claimed job after a handler failure.
        """

        return await self.__store.reschedule_job(request=request)

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load jobs for one tenant and optional filters.
        """

        return await self.__store.get_jobs(query=query)

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start or replay one idempotent request record.
        """

        return await self.__store.begin_request(request=request)

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Store the terminal outcome for an idempotent request.
        """

        return await self.__store.finish_request(request=request)

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Idempotency | None:
        """
        Load one idempotency record.
        """

        return await self.__store.get_idempotency(query=query)

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one context recipe and its lifecycle event.
        """

        return await self.__store.build_context(request=request)

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load contexts for one tenant and optional filters.
        """

        return await self.__store.get_contexts(query=query)

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Return a cursor-paginated page of tenant-scoped contexts.
        """

        return await self.__store.list_contexts(query=query)

    async def close(self) -> None:
        """
        Close the underlying asyncpg pool.
        """

        await self.__unit.close()

    async def __actor(self, *, tenant: str, actor: str) -> Actor | None:
        """
        Load one actor directly for duplicate-replay validation.
        """

        async with (
            self.__unit.session() as connection,
            connection.execute(
                "SELECT * FROM actors WHERE tenant = $1 AND id = $2 LIMIT 1", (tenant, actor)
            ) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.actor(row=row)

    async def __membership(self, *, tenant: str, membership: str) -> Membership | None:
        """
        Load one membership directly for duplicate-replay validation.
        """

        async with (
            self.__unit.session() as connection,
            connection.execute(
                "SELECT * FROM memberships WHERE tenant = $1 AND id = $2 LIMIT 1",
                (tenant, membership),
            ) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.membership(row=row)

    @staticmethod
    def __is_duplicate_integrity_error(*, exception: InteractionError) -> bool:
        """
        Return True when an InteractionError wraps a duplicate-key failure.
        """

        cause: object = exception.__cause__
        sqlstate = getattr(cause, "sqlstate", None)

        if sqlstate == "23505":
            return True

        return "duplicate key" in str(exception).lower()

    @staticmethod
    def __same_actor(*, actor: Actor, request: CreateActor) -> bool:
        """
        Compare a stored actor with the requested idempotent actor payload.
        """

        return (
            actor.name == request.name
            and actor.kind == request.kind
            and actor.skills == request.skills
            and actor.runtime == request.runtime
            and actor.external == request.external
            and actor.metadata == request.metadata
            and actor.identity.tenant == request.identity.tenant
            and actor.identity.workspace == request.identity.workspace
        )

    @staticmethod
    def __same_membership(*, membership: Membership, request: JoinThread) -> bool:
        """
        Compare a stored membership with the requested idempotent join payload.
        """

        return (
            membership.role == request.role
            and membership.actor == request.actor
            and membership.scope == request.scope
            and membership.thread == request.thread
            and membership.metadata == request.metadata
            and membership.identity.tenant == request.identity.tenant
            and membership.identity.workspace == request.identity.workspace
        )
