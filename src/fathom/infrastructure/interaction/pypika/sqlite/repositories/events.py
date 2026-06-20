from __future__ import annotations

from typing import List

from pypika import SQLLiteQuery

from fathom.constants.storage import SqlParameterStyle
from fathom.infrastructure.interaction.pypika.query import (
    CursorPaginatedQuery,
    ParameterizedQuery,
    SortDirection,
)
from fathom.infrastructure.interaction.pypika.query import (
    SortOrder as KeysetSortOrder,
)
from fathom.infrastructure.interaction.pypika.sqlite import tables
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.schemas.interaction import (
    Event,
    EventCursorQuery,
    EventPage,
    EventQuery,
    SortOrder,
)


class EventRepository:
    """
    Event repository: read-only access to lifecycle events for a thread/task.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for event reads.
        """

        self.__context = context

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped events for one thread and optional task.
        """

        if query.task is not None:
            return await self.__task_events(query=query)

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        events = tables.EVENTS
        statement = (
            SQLLiteQuery.from_(events)
            .select(events.star)
            .where(events.tenant == binder.bind(value=query.tenant))
            .where(events.thread == binder.bind(value=query.thread))
            .orderby(events.sequence)
        )
        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.event(row=row) for row in rows]

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load lifecycle events with SQL-side cursor pagination.
        """

        events = tables.EVENTS
        direction = (
            SortDirection.DESCENDING if query.order is SortOrder.DESC else SortDirection.ASCENDING
        )
        helper = CursorPaginatedQuery(
            table=events,
            ordering=KeysetSortOrder(
                tiebreaker="id",
                column="created_at",
                direction=direction,
            ),
            parameter_style=SqlParameterStyle.QUESTION_MARK,
        )
        helper.where(events.tenant == helper.bind(value=query.tenant))
        helper.where(events.thread == helper.bind(value=query.thread))

        if query.task is not None:
            helper.where(events.task == helper.bind(value=query.task))

        if query.actor is not None:
            helper.where(events.actor == helper.bind(value=query.actor))

        if query.kinds:
            helper.where(events.kind.isin([helper.bind(value=kind.value) for kind in query.kinds]))

        if query.since is not None:
            helper.where(
                events.created_at >= helper.bind(value=self.__context._time(value=query.since))
            )

        if query.until is not None:
            helper.where(
                events.created_at < helper.bind(value=self.__context._time(value=query.until))
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
            async with connection.execute(page_sql, page_parameters) as cursor_rows:
                rows = await cursor_rows.fetchall()

        items, next_cursor = self.__context._paginate(
            limit=query.limit,
            timestamp=lambda event: event.created,
            identifier=lambda event: event.identity.id,
            rows=[self.__context.rows.event(row=row) for row in rows],
        )

        return EventPage(items=tuple(items), next=next_cursor, total=total)

    async def __task_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load tenant-scoped events for one task.
        """

        events = tables.EVENTS
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)

        statement = (
            SQLLiteQuery.from_(events)
            .select(events.star)
            .where(events.tenant == binder.bind(value=query.tenant))
            .where(events.thread == binder.bind(value=query.thread))
            .where(events.task == binder.bind(value=query.task))
            .orderby(events.sequence)
        )
        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.event(row=row) for row in rows]
