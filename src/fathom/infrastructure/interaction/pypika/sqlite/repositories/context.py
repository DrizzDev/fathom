from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

import aiosqlite
from pydantic import JsonValue
from pypika import SQLLiteQuery
from pypika.functions import Coalesce
from pypika.terms import Case

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
from fathom.infrastructure.interaction.pypika.query import CursorCoordinate, ParameterizedQuery
from fathom.infrastructure.interaction.pypika.sqlite import tables
from fathom.infrastructure.interaction.pypika.sqlite.row import RowMapper
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
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime


T = TypeVar("T")


class StoreUnit(Protocol):
    """
    Transaction lifecycle protocol shared by sqlite.Unit and postgres.Unit.

    The Store talks only to this protocol so the same repository code can
    drive both backends without naming a concrete class. Implementations
    must yield a connection that exposes `execute(sql, params)` returning
    the aiosqlite-compatible cursor/awaitable shape.
    """

    def session(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        """
        Open one transactional session and yield a connection.
        """

        ...

    def atomic(self) -> AbstractAsyncContextManager[None]:
        """
        Open one grouped transaction boundary that reuses the active session.
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


class StoreContext:
    """
    Shared dependency container and protected helper surface for repositories.

    Every per-aggregate repository receives a single StoreContext and uses
    it for transaction control, row mapping, lifecycle policy, privacy
    classification, sequence allocation, pagination, single-row loads, and
    foreign-key existence checks.
    """

    def __init__(
        self,
        *,
        unit: StoreUnit,
        lifecycle: Lifecycle,
        rows: RowMapper,
        event_digest: EventDigest,
    ) -> None:
        """
        Wire shared dependencies that all repositories use.
        """

        self.__unit = unit
        self.__lifecycle = lifecycle
        self.__rows = rows
        self.__event_digest = event_digest

    @property
    def unit(self) -> StoreUnit:
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
    def rows(self) -> RowMapper:
        """
        Expose the row mapper.
        """

        return self.__rows

    async def _record_event(
        self,
        *,
        connection: aiosqlite.Connection,
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

        The event identifier embeds the per-thread sequence allocated for the
        event, so two emits with the same (kind, subject) (e.g. RUNNING ->
        BLOCKED -> RUNNING -> BLOCKED) produce distinct rows. Idempotent
        replay is guarded upstream by the calling write path; if a caller
        ever reaches __record_event twice for the same logical transition
        the rows will be distinct timeline entries (acceptable: each was a
        real lifecycle event).
        """

        sequence = await self._next_event_sequence(
            connection=connection,
            tenant=tenant,
            thread=thread,
        )
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        events = tables.EVENTS
        statement = (
            SQLLiteQuery.into(events)
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
        connection: aiosqlite.Connection,
        tenant: str,
        thread: str,
        updated: "datetime",
        cursor: int,
        digest: str,
    ) -> None:
        """
        Advance thread activity, digest, and cursor after a lifecycle event.
        """

        serialized = self._time(value=updated)
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        threads = tables.THREADS
        updated_case = (
            Case()
            .when(threads.updated_at < binder.bind(value=serialized), binder.bind(value=serialized))
            .else_(threads.updated_at)
        )
        cursor_case = (
            Case()
            .when(
                Coalesce(threads.cursor, 0) < binder.bind(value=cursor), binder.bind(value=cursor)
            )
            .else_(threads.cursor)
        )
        statement = (
            SQLLiteQuery.update(threads)
            .set(threads.updated_at, updated_case)
            .set(threads.cursor, cursor_case)
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
        tenant: str,
        thread: str,
        scope: str,
    ) -> int:
        """
        Atomically allocate and return the next sequence in a thread/scope.

        Stored value is the *last allocated* sequence. INSERT path stores 1
        and returns 1; CONFLICT path increments and returns the post-update
        value. Either way the returned integer is the value just claimed by
        this caller, and concurrent callers see strictly monotonic results
        without any read-then-write race.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        sequences = tables.SEQUENCES
        insert_statement = (
            SQLLiteQuery.into(sequences)
            .columns(sequences.tenant, sequences.thread, sequences.scope, sequences.value)
            .insert(
                binder.bind(value=tenant),
                binder.bind(value=thread),
                binder.bind(value=scope),
                1,
            )
        )
        body, parameters = binder.render(query=insert_statement)
        # Pypika SQLLiteQuery lacks ON CONFLICT/RETURNING; append literal clauses.
        sql = (
            f"{body} ON CONFLICT(tenant, thread, scope) "  # nosec B608
            "DO UPDATE SET value = value + 1 RETURNING value"
        )
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise InteractionError(f"Sequence could not be allocated for scope '{scope}'.")

        return int(row[0])

    def _build_event_id(self, *, subject: str, kind: EventKind, sequence: int) -> str:
        """
        Build an opaque unique event identifier for one entity transition.

        The per-thread sequence is the uniqueness component so the same
        (kind, subject) pair can re-occur (e.g. blocked then unblocked then
        blocked again) without colliding on the events PRIMARY KEY.
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

    def _json(self, *, value: JsonValue) -> str:
        """
        Serialize one JSON-compatible value.
        """

        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    def _time(self, *, value: "datetime") -> str:
        """
        Serialize a required datetime value.
        """

        return value.isoformat()

    def _optional_time(self, *, value: Optional["datetime"]) -> Optional[str]:
        """
        Serialize an optional datetime value.
        """

        if value is None:
            return None

        return self._time(value=value)

    def _optional_json(self, *, value: Optional[JsonValue]) -> Optional[str]:
        """
        Serialize an optional JSON-compatible value.
        """

        if value is None:
            return None

        return self._json(value=value)

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
        connection: aiosqlite.Connection,
        sql: str,
        parameters: Tuple[object, ...],
        requested: bool,
    ) -> int:
        """
        Run a scalar COUNT only when the caller asked for an exact total.

        When the caller sets count_total=False on the cursor query, we skip
        the full table scan COUNT generates and return 0; the caller knows
        the field is meaningless in that mode (documented on each Page DTO).
        """

        if not requested:
            return 0
        return await self._count(connection=connection, sql=sql, parameters=parameters)

    async def _count(
        self,
        *,
        connection: aiosqlite.Connection,
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

        return int(row[0])

    async def _load_thread(
        self,
        *,
        connection: aiosqlite.Connection,
        tenant: str,
        thread: str,
        include_archived: bool = False,
    ) -> Optional[Thread]:
        """
        Load one thread row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        threads = tables.THREADS
        statement = (
            SQLLiteQuery.from_(threads)
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
        connection: aiosqlite.Connection,
        tenant: str,
        actor: str,
    ) -> Optional[Actor]:
        """
        Load one actor row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        actors = tables.ACTORS
        statement = (
            SQLLiteQuery.from_(actors)
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
        connection: aiosqlite.Connection,
        tenant: str,
        membership: str,
    ) -> Optional[Membership]:
        """
        Load one membership row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        memberships = tables.MEMBERSHIPS
        statement = (
            SQLLiteQuery.from_(memberships)
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
        connection: aiosqlite.Connection,
        tenant: str,
        thread: str,
        actor: str,
    ) -> Optional[Membership]:
        """
        Load the active thread membership for an actor.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        memberships = tables.MEMBERSHIPS
        statement = (
            SQLLiteQuery.from_(memberships)
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
        connection: aiosqlite.Connection,
        tenant: str,
        task: str,
    ) -> Optional[Task]:
        """
        Load one task row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        tasks_table = tables.TASKS
        statement = (
            SQLLiteQuery.from_(tasks_table)
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
        connection: aiosqlite.Connection,
        tenant: str,
        message: str,
    ) -> Optional[Message]:
        """
        Load one message row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        messages = tables.MESSAGES
        statement = (
            SQLLiteQuery.from_(messages)
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
        connection: aiosqlite.Connection,
        tenant: str,
        event: str,
    ) -> Optional[Event]:
        """
        Load one event row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        events = tables.EVENTS
        statement = (
            SQLLiteQuery.from_(events)
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
        connection: aiosqlite.Connection,
        tenant: str,
        artifact: str,
    ) -> Optional[Artifact]:
        """
        Load one artifact row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        artifacts = tables.ARTIFACTS
        statement = (
            SQLLiteQuery.from_(artifacts)
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
        connection: aiosqlite.Connection,
        tenant: str,
        policy: str,
    ) -> Optional[Policy]:
        """
        Load one policy row by identity.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        policies = tables.POLICIES
        statement = (
            SQLLiteQuery.from_(policies)
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
        connection: aiosqlite.Connection,
        tenant: str,
        job: str,
    ) -> Optional[Job]:
        """
        Load one job row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        jobs = tables.JOBS
        statement = (
            SQLLiteQuery.from_(jobs)
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
        tenant: str,
        thread: str,
        actor: str,
    ) -> None:
        """
        Require that an actor has active membership in a thread.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        memberships = tables.MEMBERSHIPS
        statement = (
            SQLLiteQuery.from_(memberships)
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
        connection: aiosqlite.Connection,
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
