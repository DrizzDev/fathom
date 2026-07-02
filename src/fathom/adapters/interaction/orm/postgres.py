from __future__ import annotations

from contextlib import asynccontextmanager
from logging import getLogger
from typing import TYPE_CHECKING, List, Optional

import asyncpg

from fathom.constants.storage import InteractionBackend, PostgresMigrationMode
from fathom.core.exceptions import StorageConfigurationError
from fathom.infrastructure.interaction.orm.migration import (
    PostgresMigrator,
    PostgresSchemaValidator,
)
from fathom.infrastructure.interaction.orm.raw import InteractionSqlFiles, RawSql
from fathom.infrastructure.interaction.orm.repositories import (
    ActorRepository,
    ArtifactRepository,
    CleanupRepository,
    ContextRepository,
    EventRepository,
    ExecutionRepository,
    JobRepository,
    MembershipRepository,
    MessageRepository,
    PolicyRepository,
    ReferenceGuard,
    RequestRepository,
    ScriptRepository,
    TaskRepository,
    ThreadRepository,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    LifecycleRecorder,
    SequenceAllocator,
    UuidIdentifierSource,
)
from fathom.infrastructure.interaction.orm.runtime import PostgresInteractionRuntime
from fathom.interaction.digest import EventDigest
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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class PostgresInteraction(InteractionPort):
    """
    Postgres interaction adapter backed by persistent repositories.
    """

    def __init__(self, *, configuration: PostgresInteractionConfiguration) -> None:
        """
        Initialize repository collaborators from typed Postgres configuration.
        """

        self.__configuration = configuration
        self.__logger = getLogger(".".join((__name__, self.__class__.__name__)))

        self.__sql_files = InteractionSqlFiles.bundled()
        self.__runtime = PostgresInteractionRuntime(
            configuration=configuration, logger=self.__logger
        )

        self.__raw = RawSql(
            logger=self.__logger,
            root=self.__sql_files.root,
            slow_query_limit=configuration.slow_query_threshold,
        )

        self.__state_machine = Lifecycle()
        self.__identifiers = UuidIdentifierSource()

        self.__sequences = SequenceAllocator(
            raw=self.__raw,
            identifier_source=self.__identifiers,
        )
        self.__lifecycle = LifecycleRecorder(
            raw=self.__raw,
            event_digest=EventDigest(),
            sequence_allocator=self.__sequences,
            identifier_source=self.__identifiers,
        )
        self.__references = ReferenceGuard()

        self.__actors = ActorRepository()
        self.__threads = ThreadRepository(
            lifecycle=self.__lifecycle,
            transaction=self.__runtime,
        )

        self.__memberships = MembershipRepository(
            lifecycle=self.__lifecycle,
            transaction=self.__runtime,
        )
        self.__tasks = TaskRepository(
            recorder=self.__lifecycle,
            transaction=self.__runtime,
            lifecycle=self.__state_machine,
        )

        self.__executions = ExecutionRepository(transaction=self.__runtime)

        self.__messages = MessageRepository(
            recorder=self.__lifecycle,
            sequences=self.__sequences,
            transaction=self.__runtime,
            references=self.__references,
            lifecycle=self.__state_machine,
        )

        self.__events = EventRepository(references=self.__references)
        self.__artifacts = ArtifactRepository(
            lifecycle=self.__lifecycle,
            transaction=self.__runtime,
            references=self.__references,
        )
        self.__scripts = ScriptRepository(
            references=self.__references,
            transaction=self.__runtime,
            identifier_source=self.__identifiers,
        )

        self.__policies = PolicyRepository(
            transaction=self.__runtime,
            lifecycle=self.__state_machine,
        )
        self.__jobs = JobRepository(
            raw=self.__raw,
            transaction=self.__runtime,
            lifecycle=self.__lifecycle,
            references=self.__references,
            validator=self.__state_machine,
        )

        self.__contexts = ContextRepository(
            lifecycle=self.__lifecycle,
            transaction=self.__runtime,
            references=self.__references,
        )
        self.__requests = RequestRepository(
            transaction=self.__runtime,
            lifecycle=self.__state_machine,
            identifier_source=self.__identifiers,
        )
        self.__cleanup = CleanupRepository(transaction=self.__runtime)

        self.__initialized = False

    async def initialize(self) -> None:
        """
        Prepare schema according to migration mode and initialize the store connection pool.
        """

        if self.__initialized:
            return

        await self.__prepare_schema()
        await self.__runtime.initialize()

        self.__initialized = True

    async def aclose(self) -> None:
        """
        Close store-owned database connections.
        """

        if not self.__initialized:
            return

        await self.__runtime.close()
        self.__initialized = False

    async def close(self) -> None:
        """
        Close store-owned database connections.
        """

        await self.aclose()

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open one store transaction boundary for grouped interaction writes.
        """

        async with self.__runtime.transaction():
            yield

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Persist one conversation thread.
        """

        async with self.__runtime.session():
            return await self.__threads.create_thread(request=request)

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one active tenant-scoped thread.
        """

        async with self.__runtime.session():
            return await self.__threads.get_thread(query=query)

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set a thread title only when the stored title is null.
        """

        async with self.__runtime.session():
            return await self.__threads.set_thread_title(request=request)

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        async with self.__runtime.session():
            return await self.__threads.transition(request=request)

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load tenant-scoped threads with keyset pagination.
        """

        async with self.__runtime.session():
            return await self.__threads.list_threads(query=query)

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Persist one actor.
        """

        async with self.__runtime.session():
            return await self.__actors.create_actor(request=request)

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one actor membership in a thread.
        """

        async with self.__runtime.session():
            return await self.__memberships.join_thread(request=request)

    async def find_membership(self, *, query: MembershipQuery) -> Optional[Membership]:
        """
        Load one active actor membership in a thread.
        """

        async with self.__runtime.session():
            return await self.__memberships.find_membership(query=query)

    async def open_task(self, *, request: OpenTask) -> Task:
        """
        Persist one task in a thread.
        """

        async with self.__runtime.session():
            return await self.__tasks.open_task(request=request)

    async def finish_task(self, *, request: FinishTask) -> Task:
        """
        Move one task to a terminal state.
        """

        async with self.__runtime.session():
            return await self.__tasks.finish_task(request=request)

    async def get_tasks(self, *, query: TaskQuery) -> List[Task]:
        """
        Load tenant-scoped tasks for one thread.
        """

        async with self.__runtime.session():
            return await self.__tasks.get_tasks(query=query)

    async def get_task(self, *, query: TaskOneQuery) -> Optional[Task]:
        """
        Load one tenant-scoped task.
        """

        async with self.__runtime.session():
            return await self.__tasks.get_task(query=query)

    async def recent_task(self, *, query: TaskQuery) -> Optional[Task]:
        """
        Load the most recent non-archived task in the thread.
        """

        async with self.__runtime.session():
            return await self.__tasks.recent_task(query=query)

    async def start_execution(self, *, request: StartExecution) -> Execution:
        """
        Persist one execution in a thread.
        """

        async with self.__runtime.session():
            return await self.__executions.start_execution(request=request)

    async def finish_execution(self, *, request: FinishExecution) -> Execution:
        """
        Move one execution to a terminal state.
        """

        async with self.__runtime.session():
            return await self.__executions.finish_execution(request=request)

    async def get_execution(self, *, query: ExecutionQuery) -> Optional[Execution]:
        """
        Load one tenant-scoped execution.
        """

        async with self.__runtime.session():
            return await self.__executions.get_execution(query=query)

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message and timeline event.
        """

        async with self.__runtime.session():
            return await self.__messages.record_message(request=request)

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace message content with a sanitized version.
        """

        async with self.__runtime.session():
            return await self.__messages.sanitize_message(request=request)

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load tenant-scoped messages.
        """

        async with self.__runtime.session():
            return await self.__messages.get_messages(query=query)

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load cursor-paginated tenant-scoped messages.
        """

        async with self.__runtime.session():
            return await self.__messages.list_messages(query=query)

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped events.
        """

        async with self.__runtime.session():
            return await self.__events.get_events(query=query)

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load cursor-paginated tenant-scoped events.
        """

        async with self.__runtime.session():
            return await self.__events.list_events(query=query)

    async def link_artifact(self, *, request: LinkArtifact) -> Artifact:
        """
        Persist one artifact reference.
        """

        async with self.__runtime.session():
            return await self.__artifacts.link_artifact(request=request)

    async def get_artifacts(self, *, query: ArtifactQuery) -> List[Artifact]:
        """
        Load tenant-scoped artifacts.
        """

        async with self.__runtime.session():
            return await self.__artifacts.get_artifacts(query=query)

    async def list_artifacts(self, *, query: ArtifactCursorQuery) -> ArtifactPage:
        """
        Load cursor-paginated tenant-scoped artifacts.
        """

        async with self.__runtime.session():
            return await self.__artifacts.list_artifacts(query=query)

    async def save_script(self, *, request: SaveScript) -> Script:
        """
        Persist or update one script.
        """

        async with self.__runtime.session():
            return await self.__scripts.save_script(request=request)

    async def get_scripts(self, *, query: ScriptQuery) -> List[Script]:
        """
        Load tenant-scoped scripts.
        """

        async with self.__runtime.session():
            return await self.__scripts.get_scripts(query=query)

    async def get_script_versions(self, *, query: ScriptVersionQuery) -> List[ScriptVersion]:
        """
        Load immutable versions for one script.
        """

        async with self.__runtime.session():
            return await self.__scripts.get_script_versions(query=query)

    async def list_scripts(self, *, query: ScriptListQuery) -> ScriptPage:
        """
        Load cursor-paginated tenant-scoped scripts.
        """

        async with self.__runtime.session():
            return await self.__scripts.list_scripts(query=query)

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Persist one policy document.
        """

        async with self.__runtime.session():
            return await self.__policies.save_policy(request=request)

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one policy document.
        """

        async with self.__runtime.session():
            return await self.__policies.get_policy(query=query)

    async def schedule_job(self, *, request: ScheduleJob) -> Job:
        """
        Persist one pending job.
        """

        async with self.__runtime.session():
            return await self.__jobs.schedule_job(request=request)

    async def claim_job(self, *, request: ClaimJob) -> Optional[Job]:
        """
        Claim the next available job.
        """

        async with self.__runtime.session():
            return await self.__jobs.claim_job(request=request)

    async def finish_job(self, *, request: FinishJob) -> Job:
        """
        Move one claimed job into a terminal state.
        """

        async with self.__runtime.session():
            return await self.__jobs.finish_job(request=request)

    async def recover_jobs(self, *, request: RecoverJob) -> List[Job]:
        """
        Release stale claimed jobs for retry.
        """

        async with self.__runtime.session():
            return await self.__jobs.recover_jobs(request=request)

    async def reschedule_job(self, *, request: RescheduleJob) -> Job:
        """
        Release one owned claim for later retry.
        """

        async with self.__runtime.session():
            return await self.__jobs.reschedule_job(request=request)

    async def get_jobs(self, *, query: JobQuery) -> List[Job]:
        """
        Load tenant-scoped jobs.
        """

        async with self.__runtime.session():
            return await self.__jobs.get_jobs(query=query)

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start or replay one idempotent request.
        """

        async with self.__runtime.session():
            return await self.__requests.begin_request(request=request)

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Store the terminal outcome for one idempotent request.
        """

        async with self.__runtime.session():
            return await self.__requests.finish_request(request=request)

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one idempotency record.
        """

        async with self.__runtime.session():
            return await self.__requests.get_idempotency(query=query)

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one context recipe.
        """

        async with self.__runtime.session():
            return await self.__contexts.build_context(request=request)

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load tenant-scoped contexts.
        """

        async with self.__runtime.session():
            return await self.__contexts.get_contexts(query=query)

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load cursor-paginated tenant-scoped contexts.
        """

        async with self.__runtime.session():
            return await self.__contexts.list_contexts(query=query)

    async def cleanup(self, *, request: CleanupRequest) -> CleanupResult:
        """
        Run one bounded cleanup sweep.
        """

        async with self.__runtime.session():
            return await self.__cleanup.cleanup(request=request)

    async def __prepare_schema(self) -> None:
        """
        Apply, validate, or skip schema preparation according to configuration.
        """

        mode = self.__configuration.migration_mode
        if mode == PostgresMigrationMode.DISABLED:
            return

        if mode == PostgresMigrationMode.APPLY:
            await self.__apply_migrations()
            return

        if mode == PostgresMigrationMode.VALIDATE:
            await self.__validate_schema()
            return

        raise StorageConfigurationError(
            backend=InteractionBackend.POSTGRES.value,
            message=f"Unsupported Postgres migration mode: {mode.value}",
        )

    async def __apply_migrations(self) -> None:
        """
        Create the configured schema and apply the store baseline migration.
        """

        connection = await self.__connect_for_migration()

        try:
            schema = self.__quote_identifier(value=self.__configuration.schema_name)

            await PostgresMigrator().apply(connection=connection, schema=schema)
        finally:
            await connection.close()

    async def __validate_schema(self) -> None:
        """
        Validate the configured schema without applying migrations.
        """

        connection = await self.__connect_for_migration()
        try:
            schema = self.__quote_identifier(value=self.__configuration.schema_name)
            await connection.execute(f"SET search_path TO {schema}")
            await PostgresSchemaValidator().validate(connection=connection)
        finally:
            await connection.close()

    async def __connect_for_migration(self) -> asyncpg.Connection:
        """
        Open one temporary asyncpg connection for schema migration.
        """

        target = self.__runtime.connection_target()
        return await asyncpg.connect(
            host=target.host,
            port=target.port,
            user=target.user,
            password=target.password,
            database=target.database,
            ssl=self.__configuration.ssl.value,
            server_settings=self.__runtime.server_settings(),
        )

    @staticmethod
    def __quote_identifier(*, value: str) -> str:
        """
        Quote one Postgres identifier.
        """

        return '"' + value.replace('"', '""') + '"'
