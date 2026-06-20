from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pypika import PostgreSQLQuery
from pypika.functions import Coalesce, Lower

from fathom.constants.collaboration import EventKind, EventSource, ThreadState
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError, ThreadConflictError
from fathom.infrastructure.interaction.pypika.escaping import SqlLikeEscape
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import (
    PostgresConnectionProtocol,
    PostgresContext,
)
from fathom.infrastructure.interaction.pypika.query import (
    CursorPaginatedQuery,
    ParameterizedQuery,
    SortDirection,
    SortOrder,
)
from fathom.schemas.interaction import (
    CreateThread,
    Metadata,
    SetThreadTitle,
    Thread,
    ThreadListQuery,
    ThreadPage,
    ThreadQuery,
    ThreadTransition,
    Timing,
)

if TYPE_CHECKING:
    from datetime import datetime


class PostgresThreadRepository:
    """
    Postgres thread repository: persists and queries durable interaction threads.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for thread persistence.
        """

        self.__context = context

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Persist one interaction thread.
        """

        timing = Timing(created_at=request.created, updated_at=request.created)

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_thread(
                connection=connection,
                thread=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None:
                if not self.__same_thread(thread=existing, request=request):
                    raise ThreadConflictError(
                        thread=request.identity.id,
                        message="Thread identity already exists with different content.",
                    )

                return existing

            if request.creator is not None:
                await self.__context._require_actor(
                    connection=connection,
                    actor=request.creator,
                    tenant=request.identity.tenant,
                )

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            threads = tables.THREADS
            statement = (
                PostgreSQLQuery.into(threads)
                .columns(
                    threads.id,
                    threads.tenant,
                    threads.workspace,
                    threads.title,
                    threads.state,
                    threads.digest,
                    threads.cursor,
                    threads.creator,
                    threads.created_at,
                    threads.updated_at,
                    threads.archived_at,
                    threads.deleted_at,
                    threads.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.title),
                    binder.bind(value=request.state.value),
                    binder.bind(value=None),
                    binder.bind(value=None),
                    binder.bind(value=request.creator),
                    binder.bind(value=self.__context._time(value=timing.created)),
                    binder.bind(value=self.__context._time(value=timing.updated)),
                    binder.bind(value=None),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            thread = await self.__context._load_thread(
                connection=connection,
                thread=request.identity.id,
                tenant=request.identity.tenant,
            )

            await self.__context._record_event(
                task=None,
                connection=connection,
                actor=request.creator,
                created=request.created,
                thread=request.identity.id,
                subject=request.identity.id,
                tenant=request.identity.tenant,
                kind=EventKind.THREAD_CREATED,
                source=EventSource.INTERACTION,
                workspace=request.identity.workspace,
                payload=Metadata(entries={"state": request.state.value}),
            )

        if thread is None:
            raise InteractionError("Thread was not persisted.")

        return thread

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one tenant-scoped thread.
        """

        async with self.__context.unit.session() as connection:
            return await self.__context._load_thread(
                tenant=query.tenant,
                thread=query.thread,
                connection=connection,
            )

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set the thread title only when the stored title is currently null.
        """

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_thread(
                connection=connection,
                tenant=request.tenant,
                thread=request.thread,
            )
            if existing is None:
                raise InteractionError("Thread does not exist.")

            if existing.title is not None:
                return existing

            threads = tables.THREADS
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)

            statement = (
                PostgreSQLQuery.update(threads)
                .set(threads.title, binder.bind(value=request.title))
                .set(
                    threads.updated_at,
                    binder.bind(value=self.__context._time(value=request.updated)),
                )
                .where(threads.title.isnull())
                .where(threads.id == binder.bind(value=request.thread))
                .where(threads.tenant == binder.bind(value=request.tenant))
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            updated = await self.__context._load_thread(
                connection=connection,
                tenant=request.tenant,
                thread=request.thread,
            )

        if updated is None:
            raise InteractionError("Thread was not updated.")

        return updated

    async def transition(self, *, request: ThreadTransition) -> Thread:
        """
        Archive, unarchive, or soft-delete one thread.
        """

        if request.state not in (ThreadState.ACTIVE, ThreadState.ARCHIVED, ThreadState.DELETED):
            raise InteractionError("Unsupported thread lifecycle target state.")

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_thread(
                connection=connection,
                tenant=request.tenant,
                thread=request.thread,
                include_archived=True,
            )
            if existing is None:
                raise InteractionError("Thread does not exist.")

            threads = tables.THREADS
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)

            deleted = request.updated if request.state == ThreadState.DELETED else None
            archived = request.updated if request.state == ThreadState.ARCHIVED else None

            statement = (
                PostgreSQLQuery.update(threads)
                .set(threads.state, binder.bind(value=request.state.value))
                .set(
                    threads.updated_at,
                    binder.bind(value=self.__context._time(value=request.updated)),
                )
                .set(
                    threads.archived_at,
                    binder.bind(value=self.__context._optional_time(value=archived)),
                )
                .set(
                    threads.deleted_at,
                    binder.bind(value=self.__context._optional_time(value=deleted)),
                )
                .where(threads.deleted_at.isnull())
                .where(threads.id == binder.bind(value=request.thread))
                .where(threads.tenant == binder.bind(value=request.tenant))
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            if request.state == ThreadState.DELETED:
                await self.__soft_delete_thread_children(
                    connection=connection,
                    tenant=request.tenant,
                    thread=request.thread,
                    deleted=request.updated,
                )
                await self.__record_thread_lifecycle_event(
                    request=request,
                    thread=existing,
                    connection=connection,
                    kind=EventKind.THREAD_DELETED,
                )
                return existing.model_copy(
                    update={
                        "archived": None,
                        "deleted": request.updated,
                        "state": ThreadState.DELETED,
                        "timing": existing.timing.model_copy(update={"updated": request.updated}),
                    }
                )

            updated = await self.__context._load_thread(
                connection=connection,
                tenant=request.tenant,
                thread=request.thread,
                include_archived=True,
            )
            if updated is None:
                raise InteractionError("Thread was not updated.")

            await self.__record_thread_lifecycle_event(
                thread=updated,
                request=request,
                connection=connection,
                kind=(
                    EventKind.THREAD_ARCHIVED
                    if request.state == ThreadState.ARCHIVED
                    else EventKind.THREAD_UNARCHIVED
                ),
            )

        return updated

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load tenant-scoped threads with SQL-side cursor pagination.
        """

        threads = tables.THREADS

        helper = CursorPaginatedQuery(
            table=threads,
            ordering=SortOrder(
                tiebreaker="id",
                column="updated_at",
                direction=SortDirection.DESCENDING,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )

        helper.where(threads.deleted_at.isnull())
        helper.where(threads.tenant == helper.bind(value=query.tenant))

        if not query.include_archived:
            helper.where(threads.archived_at.isnull())

        if query.workspace is not None:
            helper.where(threads.workspace == helper.bind(value=query.workspace))

        if query.state is not None:
            helper.where(threads.state == helper.bind(value=query.state.value))

        if query.updated_since is not None:
            helper.where(
                threads.updated_at
                >= helper.bind(value=self.__context._time(value=query.updated_since))
            )

        if query.updated_until is not None:
            helper.where(
                threads.updated_at
                < helper.bind(value=self.__context._time(value=query.updated_until))
            )

        if query.title is not None:
            helper.where(
                SqlLikeEscape.prefix_clause(
                    binder=helper.binder,
                    prefix=query.title.lower(),
                    column=Lower(Coalesce(threads.title, "")),
                )
            )

        count_sql, count_parameters = helper.count_sql_and_parameters()
        page_sql, page_parameters = helper.page_sql_and_parameters(
            limit=query.limit + 1,
            cursor=self.__context._decode_keyset_cursor(value=query.cursor),
        )

        async with self.__context.unit.session() as connection:
            total = await self.__context._optional_count(
                sql=count_sql,
                connection=connection,
                parameters=count_parameters,
                requested=query.count_total,
            )

            async with connection.execute(page_sql, page_parameters) as sql_cursor:
                rows = await sql_cursor.fetchall()

        items, next_cursor = self.__context._paginate(
            limit=query.limit,
            identifier=lambda thread: thread.identity.id,
            timestamp=lambda thread: thread.timing.updated,
            rows=[self.__context.rows.thread(row=row) for row in rows],
        )
        return ThreadPage(items=tuple(items), next=next_cursor, total=total)

    async def __soft_delete_thread_children(
        self,
        *,
        tenant: str,
        thread: str,
        deleted: datetime,
        connection: PostgresConnectionProtocol,
    ) -> None:
        """
        Soft-delete thread-owned rows that expose a deleted_at column.
        """

        deleted_at = self.__context._time(value=deleted)

        for table in (tables.TASKS, tables.MESSAGES, tables.ARTIFACTS, tables.SCRIPTS):
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            statement = (
                PostgreSQLQuery.update(table)
                .set(table.deleted_at, binder.bind(value=deleted_at))
                .where(table.tenant == binder.bind(value=tenant))
                .where(table.thread == binder.bind(value=thread))
                .where(table.deleted_at.isnull())
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

    async def __record_thread_lifecycle_event(
        self,
        *,
        thread: Thread,
        kind: EventKind,
        request: ThreadTransition,
        connection: PostgresConnectionProtocol,
    ) -> None:
        """
        Record the lifecycle event for a thread transition.
        """

        await self.__context._record_event(
            task=None,
            kind=kind,
            actor=request.actor,
            connection=connection,
            created=request.updated,
            thread=thread.identity.id,
            subject=thread.identity.id,
            tenant=thread.identity.tenant,
            source=EventSource.INTERACTION,
            workspace=thread.identity.workspace,
            payload=Metadata(entries={"state": request.state.value}),
        )

    def __same_thread(self, *, thread: Thread, request: CreateThread) -> bool:
        """
        Check whether a thread request replays an already stored thread.
        """

        return (
            thread.title == request.title
            and thread.state == request.state
            and thread.creator == request.creator
            and thread.metadata == request.metadata
            and thread.timing.created == request.created
            and thread.identity.tenant == request.identity.tenant
            and thread.identity.workspace == request.identity.workspace
        )
