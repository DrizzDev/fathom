from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

from pydantic import JsonValue
from pypika import PostgreSQLQuery
from pypika.terms import CustomFunction

from fathom.constants.collaboration import (
    EventKind,
    EventSource,
    JobState,
    TaskState,
)
from fathom.constants.storage import SqlParameterStyle
from fathom.conversation.cursor import OpaqueCursor
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.row import PostgresRowMapper
from fathom.infrastructure.interaction.pypika.query import CursorCoordinate, ParameterizedQuery
from fathom.interaction.digest import EventDigest
from fathom.interaction.lifecycle import Lifecycle
from fathom.schemas.interaction import (
    Actor,
    Artifact,
    Event,
    Job,
    Membership,
    Message,
    Metadata,
    Policy,
    References,
    Task,
    Thread,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime


T = TypeVar("T")


class PostgresUnitProtocol(Protocol):
    """
    Transaction lifecycle protocol implemented by the Postgres unit-of-work.
    """

    def session(self) -> AbstractAsyncContextManager["PostgresConnectionProtocol"]:
        """
        Open one transactional session and yield a connection.
        """

        ...

    def atomic(self) -> AbstractAsyncContextManager[None]:
        """
        Open one grouped transaction boundary that reuses the active session.
        """

        ...


class PostgresRowProtocol(Protocol):
    """
    Minimal asyncpg row surface consumed by repositories.
    """

    def __getitem__(self, key: str | int) -> object:
        """
        Return one column value by column name.
        """

        ...


class PostgresCursorProtocol(Protocol):
    """
    Minimal cursor surface returned by Postgres execution wrappers.
    """

    async def fetchone(self) -> Optional[PostgresRowProtocol]:
        """
        Return one row or None.
        """

        ...

    async def fetchall(self) -> List[PostgresRowProtocol]:
        """
        Return all remaining rows.
        """

        ...


class PostgresExecutionProtocol(Protocol):
    """
    Awaitable/context-manager execution surface used by repositories.
    """

    rowcount: int

    def __await__(self) -> "Generator[object, None, PostgresExecutionProtocol]":
        """
        Await the execution and return the execution result.
        """

        ...

    async def __aenter__(self) -> PostgresCursorProtocol:
        """
        Enter an async cursor context.
        """

        ...

    async def __aexit__(self, *args: object) -> None:
        """
        Exit an async cursor context.
        """

        ...


class PostgresConnectionProtocol(Protocol):
    """
    Minimal transactional connection surface consumed by repositories.
    """

    def execute(
        self,
        sql: str,
        parameters: Tuple[object, ...] | List[object] = (),
    ) -> PostgresExecutionProtocol:
        """
        Execute one statement with bound parameters.
        """

        ...


TASK_EVENT_KINDS: Dict[TaskState, EventKind] = {
    TaskState.SUCCEEDED: EventKind.TASK_SUCCEEDED,
    TaskState.FAILED: EventKind.TASK_FAILED,
    TaskState.CANCELLED: EventKind.TASK_CANCELLED,
    TaskState.EXPIRED: EventKind.TASK_EXPIRED,
    TaskState.DELETED: EventKind.TASK_DELETED,
}

JOB_EVENT_KINDS: Dict[JobState, EventKind] = {
    JobState.COMPLETED: EventKind.JOB_COMPLETED,
    JobState.FAILED: EventKind.JOB_FAILED,
    JobState.ABANDONED: EventKind.JOB_ABANDONED,
}


class PostgresContext:
    """
    Shared dependency container for Postgres native repositories.

    Mirrors the shared StoreContext but emits Postgres `$n` placeholders
    directly, treats JSONB values as native Python dicts/lists, and treats
    TIMESTAMPTZ values as timezone-aware datetimes (the asyncpg pool
    registers a JSON codec at acquire time so dict/list pass through).
    """

    def __init__(
        self,
        *,
        unit: PostgresUnitProtocol,
        lifecycle: Lifecycle,
        rows: PostgresRowMapper,
        event_digest: EventDigest,
    ) -> None:
        """
        Wire shared dependencies that all Postgres repositories use.
        """

        self.__unit = unit
        self.__lifecycle = lifecycle
        self.__rows = rows
        self.__event_digest = event_digest

    @property
    def unit(self) -> PostgresUnitProtocol:
        """
        Expose the transactional unit-of-work.
        """

        return self.__unit

    @property
    def lifecycle(self) -> Lifecycle:
        """
        Expose the lifecycle policy validator.
        """

        return self.__lifecycle

    @property
    def rows(self) -> PostgresRowMapper:
        """
        Expose the Postgres row mapper.
        """

        return self.__rows

    async def _record_event(
        self,
        *,
        connection: PostgresConnectionProtocol,
        subject: str,
        tenant: str,
        workspace: Optional[str],
        thread: str,
        task: Optional[str],
        actor: Optional[str],
        kind: EventKind,
        source: EventSource,
        payload: Metadata,
        created: "datetime",
    ) -> None:
        """
        Persist one lifecycle event within the active transaction.
        """

        sequence = await self._next_event_sequence(
            connection=connection,
            tenant=tenant,
            thread=thread,
        )
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        events = tables.EVENTS
        statement = (
            PostgreSQLQuery.into(events)
            .columns(
                events.id,
                events.tenant,
                events.workspace,
                events.thread,
                events.task,
                events.actor,
                events.sequence,
                events.kind,
                events.source,
                events.payload,
                events.created_at,
                events.metadata,
            )
            .insert(
                binder.bind(
                    value=self._build_event_id(subject=subject, kind=kind, sequence=sequence)
                ),
                binder.bind(value=tenant),
                binder.bind(value=workspace),
                binder.bind(value=thread),
                binder.bind(value=task),
                binder.bind(value=actor),
                binder.bind(value=sequence),
                binder.bind(value=kind.value),
                binder.bind(value=source.value),
                binder.bind(value=self._json(value=payload.entries)),
                binder.bind(value=self._time(value=created)),
                binder.bind(value=self._json(value={})),
            )
        )
        sql, parameters = binder.render(query=statement)
        await connection.execute(sql, parameters)
        digest = self.__event_digest.compute(
            kind=kind,
            source=source,
            payload=payload,
            created=created,
            sequence=sequence,
        )
        await self._touch_thread(
            connection=connection,
            tenant=tenant,
            thread=thread,
            updated=created,
            cursor=sequence,
            digest=digest,
        )

    async def _touch_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        updated: "datetime",
        cursor: int,
        digest: str,
    ) -> None:
        """
        Advance thread activity, digest, and cursor after a lifecycle event.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        threads = tables.THREADS
        greatest = CustomFunction("GREATEST", ["left", "right"])
        coalesce = CustomFunction("COALESCE", ["value", "fallback"])
        statement = (
            PostgreSQLQuery.update(threads)
            .set(
                threads.updated_at,
                greatest(threads.updated_at, binder.bind(value=self._time(value=updated))),
            )
            .set(threads.cursor, greatest(coalesce(threads.cursor, 0), binder.bind(value=cursor)))
            .set(threads.digest, binder.bind(value=digest))
            .where(threads.tenant == binder.bind(value=tenant))
            .where(threads.id == binder.bind(value=thread))
            .where(threads.deleted_at.isnull())
        )
        sql, parameters = binder.render(query=statement)
        await connection.execute(sql, parameters)

    async def _next_event_sequence(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
    ) -> int:
        """
        Return the next sequence value for a thread event timeline.
        """

        return await self._allocate_sequence(
            connection=connection, tenant=tenant, thread=thread, scope="event"
        )

    async def _next_message_sequence(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
    ) -> int:
        """
        Return the next sequence value for a thread message timeline.
        """

        return await self._allocate_sequence(
            connection=connection, tenant=tenant, thread=thread, scope="message"
        )

    async def _allocate_sequence(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        scope: str,
    ) -> int:
        """
        Atomically allocate and return the next sequence in a thread/scope.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        sequences = tables.SEQUENCES
        statement = (
            PostgreSQLQuery.into(sequences)
            .columns(sequences.tenant, sequences.thread, sequences.scope, sequences.value)
            .insert(
                binder.bind(value=tenant),
                binder.bind(value=thread),
                binder.bind(value=scope),
                1,
            )
            .on_conflict(sequences.tenant, sequences.thread, sequences.scope)
            .do_update(sequences.value, sequences.value + 1)
            .returning(sequences.value)
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise InteractionError(f"Sequence could not be allocated for scope '{scope}'.")

        return self.__integer(value=row["value"])

    def _build_event_id(self, *, subject: str, kind: EventKind, sequence: int) -> str:
        """
        Build an opaque unique event identifier for one entity transition.
        """

        return InteractionIdentity.stable(scope="event", parts=(kind.value, subject, sequence))

    def _task_event_kind(self, *, state: TaskState) -> EventKind:
        """
        Map terminal task state to lifecycle event kind.
        """

        kind = TASK_EVENT_KINDS.get(state)
        if kind is None:
            raise InteractionError(f"Task state '{state.value}' has no terminal event kind.")

        return kind

    def _job_event_kind(self, *, state: JobState) -> EventKind:
        """
        Map terminal job state to lifecycle event kind.
        """

        kind = JOB_EVENT_KINDS.get(state)
        if kind is None:
            raise InteractionError(f"Job state '{state.value}' has no terminal event kind.")

        return kind

    def _json(self, *, value: JsonValue) -> JsonValue:
        """
        Pass a JSON-compatible value through to asyncpg unchanged.
        """

        return value

    def _time(self, *, value: "datetime") -> "datetime":
        """
        Pass a datetime value through to asyncpg unchanged.
        """

        return value

    def _optional_time(self, *, value: Optional["datetime"]) -> Optional["datetime"]:
        """
        Pass an optional datetime value through to asyncpg unchanged.
        """

        return value

    def _optional_json(self, *, value: Optional[JsonValue]) -> Optional[JsonValue]:
        """
        Pass an optional JSON-compatible value through to asyncpg unchanged.
        """

        return value

    def _decode_cursor(self, *, value: Optional[str]) -> Optional[OpaqueCursor]:
        """
        Decode an opaque cursor value when the client supplied one.
        """

        if value is None:
            return None

        return OpaqueCursor.decode(value=value)

    def _decode_keyset_cursor(self, *, value: Optional[str]) -> Optional[CursorCoordinate]:
        """
        Decode an opaque cursor and project it onto a backend-serialized keyset boundary.
        """

        decoded = self._decode_cursor(value=value)
        if decoded is None:
            return None
        return CursorCoordinate(
            created=self._time(value=decoded.created),
            identifier=decoded.identifier,
        )

    def _paginate(
        self,
        *,
        rows: List[T],
        limit: int,
        timestamp: Callable[[T], "datetime"],
        identifier: Callable[[T], str],
    ) -> Tuple[List[T], Optional[str]]:
        """
        Trim an over-fetched result set and return the next opaque cursor.
        """

        items = rows[:limit]
        if len(rows) <= limit or not items:
            return items, None

        last = items[-1]
        return (
            items,
            OpaqueCursor(created=timestamp(last), identifier=identifier(last)).encode(),
        )

    async def _optional_count(
        self,
        *,
        connection: PostgresConnectionProtocol,
        sql: str,
        parameters: Tuple[object, ...],
        requested: bool,
    ) -> int:
        """
        Run a scalar COUNT only when the caller asked for an exact total.
        """

        if not requested:
            return 0
        return await self._count(connection=connection, sql=sql, parameters=parameters)

    async def _count(
        self,
        *,
        connection: PostgresConnectionProtocol,
        sql: str,
        parameters: Tuple[object, ...],
    ) -> int:
        """
        Execute a scalar COUNT query.
        """

        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise InteractionError("Count query returned no row.")

        return self.__integer(value=row[0])

    def __integer(self, *, value: object) -> int:
        """
        Coerce a database scalar to int with explicit diagnostics.
        """

        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        raise InteractionError(f"Expected integer database scalar; got {value!r}.")

    async def _load_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        include_archived: bool = False,
    ) -> Optional[Thread]:
        """
        Load one thread row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        threads = tables.THREADS
        statement = (
            PostgreSQLQuery.from_(threads)
            .select(threads.star)
            .where(threads.tenant == binder.bind(value=tenant))
            .where(threads.id == binder.bind(value=thread))
            .where(threads.deleted_at.isnull())
        )
        if not include_archived:
            statement = statement.where(threads.archived_at.isnull())

        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.thread(row=row)

    async def _load_actor(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        actor: str,
    ) -> Optional[Actor]:
        """
        Load one actor row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        actors = tables.ACTORS
        statement = (
            PostgreSQLQuery.from_(actors)
            .select(actors.star)
            .where(actors.tenant == binder.bind(value=tenant))
            .where(actors.id == binder.bind(value=actor))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.actor(row=row)

    async def _load_membership(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        membership: str,
    ) -> Optional[Membership]:
        """
        Load one membership row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        memberships = tables.MEMBERSHIPS
        statement = (
            PostgreSQLQuery.from_(memberships)
            .select(memberships.star)
            .where(memberships.tenant == binder.bind(value=tenant))
            .where(memberships.id == binder.bind(value=membership))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.membership(row=row)

    async def find_active_membership(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        actor: str,
    ) -> Optional[Membership]:
        """
        Load the active thread membership for an actor.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        memberships = tables.MEMBERSHIPS
        statement = (
            PostgreSQLQuery.from_(memberships)
            .select(memberships.star)
            .where(memberships.tenant == binder.bind(value=tenant))
            .where(memberships.thread == binder.bind(value=thread))
            .where(memberships.actor == binder.bind(value=actor))
            .where(memberships.departed_at.isnull())
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.membership(row=row)

    async def _load_task(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        task: str,
    ) -> Optional[Task]:
        """
        Load one task row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        tasks_table = tables.TASKS
        statement = (
            PostgreSQLQuery.from_(tasks_table)
            .select(tasks_table.star)
            .where(tasks_table.tenant == binder.bind(value=tenant))
            .where(tasks_table.id == binder.bind(value=task))
            .where(tasks_table.deleted_at.isnull())
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.task(row=row)

    async def _load_message(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        message: str,
    ) -> Optional[Message]:
        """
        Load one message row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        messages = tables.MESSAGES
        statement = (
            PostgreSQLQuery.from_(messages)
            .select(messages.star)
            .where(messages.tenant == binder.bind(value=tenant))
            .where(messages.id == binder.bind(value=message))
            .where(messages.deleted_at.isnull())
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.message(row=row)

    async def _load_event(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        event: str,
    ) -> Optional[Event]:
        """
        Load one event row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        events = tables.EVENTS
        statement = (
            PostgreSQLQuery.from_(events)
            .select(events.star)
            .where(events.tenant == binder.bind(value=tenant))
            .where(events.id == binder.bind(value=event))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.event(row=row)

    async def _load_artifact(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        artifact: str,
    ) -> Optional[Artifact]:
        """
        Load one artifact row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        artifacts = tables.ARTIFACTS
        statement = (
            PostgreSQLQuery.from_(artifacts)
            .select(artifacts.star)
            .where(artifacts.tenant == binder.bind(value=tenant))
            .where(artifacts.id == binder.bind(value=artifact))
            .where(artifacts.deleted_at.isnull())
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.artifact(row=row)

    async def _load_policy(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        policy: str,
    ) -> Optional[Policy]:
        """
        Load one policy row by identity.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        policies = tables.POLICIES
        statement = (
            PostgreSQLQuery.from_(policies)
            .select(policies.star)
            .where(policies.tenant == binder.bind(value=tenant))
            .where(policies.id == binder.bind(value=policy))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.policy(row=row)

    async def _load_job(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        job: str,
    ) -> Optional[Job]:
        """
        Load one job row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        jobs = tables.JOBS
        statement = (
            PostgreSQLQuery.from_(jobs)
            .select(jobs.star)
            .where(jobs.tenant == binder.bind(value=tenant))
            .where(jobs.id == binder.bind(value=job))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__rows.job(row=row)

    async def _require_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
    ) -> None:
        """
        Require that a thread exists for a tenant.
        """

        if await self._load_thread(connection=connection, tenant=tenant, thread=thread) is None:
            raise InteractionError("Thread does not exist.")

    async def _require_actor(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        actor: str,
    ) -> None:
        """
        Require that an actor exists for a tenant.
        """

        if await self._load_actor(connection=connection, tenant=tenant, actor=actor) is None:
            raise InteractionError("Actor does not exist.")

    async def _require_active_membership(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        actor: str,
    ) -> None:
        """
        Require that an actor has active membership in a thread.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        memberships = tables.MEMBERSHIPS
        statement = (
            PostgreSQLQuery.from_(memberships)
            .select(memberships.id)
            .where(memberships.tenant == binder.bind(value=tenant))
            .where(memberships.thread == binder.bind(value=thread))
            .where(memberships.actor == binder.bind(value=actor))
            .where(memberships.departed_at.isnull())
            .limit(1)
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise InteractionError("Actor is not an active member of the thread.")

    async def _require_task(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        task: str,
    ) -> None:
        """
        Require that a task exists for a tenant.
        """

        if await self._load_task(connection=connection, tenant=tenant, task=task) is None:
            raise InteractionError("Task does not exist.")

    async def _require_task_in_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        task: str,
    ) -> None:
        """
        Require that a task exists in the expected thread.
        """

        existing = await self._load_task(connection=connection, tenant=tenant, task=task)
        if existing is None:
            raise InteractionError("Task does not exist.")
        if existing.thread != thread:
            raise InteractionError("Task belongs to a different thread.")

    async def _require_message_in_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        message: str,
    ) -> None:
        """
        Require that a message exists in the expected thread.
        """

        existing = await self._load_message(connection=connection, tenant=tenant, message=message)
        if existing is None:
            raise InteractionError("Message does not exist.")
        if existing.thread != thread:
            raise InteractionError("Message belongs to a different thread.")

    async def _require_event_in_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        event: str,
    ) -> None:
        """
        Require that an event exists in the expected thread.
        """

        existing = await self._load_event(connection=connection, tenant=tenant, event=event)
        if existing is None:
            raise InteractionError("Event does not exist.")
        if existing.thread != thread:
            raise InteractionError("Event belongs to a different thread.")

    async def _require_artifact_in_thread(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        artifact: str,
    ) -> None:
        """
        Require that an artifact exists in the expected thread.
        """

        existing = await self._load_artifact(
            connection=connection, tenant=tenant, artifact=artifact
        )
        if existing is None:
            raise InteractionError("Artifact does not exist.")
        if existing.thread != thread:
            raise InteractionError("Artifact belongs to a different thread.")

    async def _require_references(
        self,
        *,
        connection: PostgresConnectionProtocol,
        tenant: str,
        thread: str,
        references: References,
    ) -> None:
        """
        Require local context references to exist in the same thread.
        """

        for message in references.messages:
            await self._require_message_in_thread(
                connection=connection,
                tenant=tenant,
                thread=thread,
                message=message,
            )
        for event in references.events:
            await self._require_event_in_thread(
                connection=connection,
                tenant=tenant,
                thread=thread,
                event=event,
            )
        for artifact in references.artifacts:
            await self._require_artifact_in_thread(
                connection=connection,
                tenant=tenant,
                thread=thread,
                artifact=artifact,
            )
