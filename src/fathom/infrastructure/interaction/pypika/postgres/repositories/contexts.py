from __future__ import annotations

from typing import List, Optional

from pydantic import JsonValue
from pypika import PostgreSQLQuery

from fathom.constants.collaboration import EventKind, EventSource
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import (
    PostgresConnectionProtocol,
    PostgresContext,
)
from fathom.infrastructure.interaction.pypika.query import (
    CursorPaginatedQuery,
    ParameterizedQuery,
    SortDirection,
)
from fathom.infrastructure.interaction.pypika.query import (
    SortOrder as KeysetSortOrder,
)
from fathom.schemas.interaction import (
    BuildContext,
    Context,
    ContextCursorQuery,
    ContextPage,
    ContextQuery,
    Metadata,
    References,
    SortOrder,
)


class PostgresContextRepository:
    """
    Postgres context repository: persists reference-based context recipes.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for context-recipe persistence.
        """

        self.__context = context

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one reference-based context recipe and its lifecycle event.
        """

        async with self.__context.unit.session() as connection:
            await self.__context._require_thread(
                connection=connection,
                tenant=request.identity.tenant,
                thread=request.thread,
            )
            if request.task is not None:
                await self.__context._require_task_in_thread(
                    connection=connection,
                    tenant=request.identity.tenant,
                    thread=request.thread,
                    task=request.task,
                )
            if request.consumer is not None:
                await self.__context._require_actor(
                    connection=connection,
                    tenant=request.identity.tenant,
                    actor=request.consumer,
                )
                await self.__context._require_active_membership(
                    connection=connection,
                    tenant=request.identity.tenant,
                    thread=request.thread,
                    actor=request.consumer,
                )
            await self.__context._require_references(
                connection=connection,
                tenant=request.identity.tenant,
                thread=request.thread,
                references=request.references,
            )
            existing = await self.__load_context_row(
                connection=connection,
                tenant=request.identity.tenant,
                context=request.identity.id,
            )
            if existing is not None:
                if not self.__same_context(context=existing, request=request):
                    raise InteractionError(
                        "Context identity already exists with different content."
                    )

                return existing

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            contexts_table = tables.CONTEXTS
            statement = (
                PostgreSQLQuery.into(contexts_table)
                .columns(
                    contexts_table.id,
                    contexts_table.tenant,
                    contexts_table.workspace,
                    contexts_table.thread,
                    contexts_table.task,
                    contexts_table.consumer,
                    contexts_table.purpose,
                    contexts_table.builder,
                    contexts_table.references,
                    contexts_table.budget,
                    contexts_table.filters,
                    contexts_table.hash,
                    contexts_table.provider,
                    contexts_table.model,
                    contexts_table.created_at,
                    contexts_table.expires_at,
                    contexts_table.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.thread),
                    binder.bind(value=request.task),
                    binder.bind(value=request.consumer),
                    binder.bind(value=request.purpose.value),
                    binder.bind(value=request.builder),
                    binder.bind(
                        value=self.__context._json(
                            value=self.__references(value=request.references)
                        )
                    ),
                    binder.bind(value=self.__context._json(value=request.budget.entries)),
                    binder.bind(value=self.__context._json(value=request.filters.entries)),
                    binder.bind(value=request.hash),
                    binder.bind(value=request.provider),
                    binder.bind(value=request.model),
                    binder.bind(value=self.__context._time(value=request.created)),
                    binder.bind(value=self.__context._optional_time(value=request.expires)),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            # Postgres reserves `references`; the column list emits a bareword
            # under `quote_char=None`, so explicitly quote it to match the schema.
            sql = sql.replace(",references,", ',"references",')
            await connection.execute(sql, parameters)
            context = await self.__load_context_row(
                connection=connection,
                tenant=request.identity.tenant,
                context=request.identity.id,
            )
            await self.__context._record_event(
                connection=connection,
                subject=request.identity.id,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                thread=request.thread,
                task=request.task,
                actor=request.consumer,
                kind=EventKind.CONTEXT_BUILT,
                source=EventSource.INTERACTION,
                payload=Metadata(
                    entries={"purpose": request.purpose.value, "builder": request.builder}
                ),
                created=request.created,
            )

        if context is None:
            raise InteractionError("Context was not persisted.")

        return context

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load tenant-scoped contexts with optional task and purpose filters.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        contexts_table = tables.CONTEXTS
        statement = (
            PostgreSQLQuery.from_(contexts_table)
            .select(contexts_table.star)
            .where(contexts_table.tenant == binder.bind(value=query.tenant))
            .where(contexts_table.thread == binder.bind(value=query.thread))
        )
        if query.task is not None:
            statement = statement.where(contexts_table.task == binder.bind(value=query.task))
        if query.purpose is not None:
            statement = statement.where(
                contexts_table.purpose == binder.bind(value=query.purpose.value)
            )
        statement = statement.orderby(contexts_table.created_at).orderby(contexts_table.id)
        sql, parameters = binder.render(query=statement)
        async with (
            self.__context.unit.session() as connection,
            connection.execute(sql, parameters) as cursor,
        ):
            rows = await cursor.fetchall()

        return [self.__context.rows.context(row=row) for row in rows]

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load contexts with SQL-side cursor pagination.
        """

        contexts_table = tables.CONTEXTS
        direction = (
            SortDirection.DESCENDING if query.order is SortOrder.DESC else SortDirection.ASCENDING
        )
        helper = CursorPaginatedQuery(
            table=contexts_table,
            ordering=KeysetSortOrder(
                tiebreaker="id",
                column="created_at",
                direction=direction,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )
        helper.where(contexts_table.tenant == helper.bind(value=query.tenant))
        helper.where(contexts_table.thread == helper.bind(value=query.thread))

        if query.task is not None:
            helper.where(contexts_table.task == helper.bind(value=query.task))

        if query.consumer is not None:
            helper.where(contexts_table.consumer == helper.bind(value=query.consumer))

        if query.purpose is not None:
            helper.where(contexts_table.purpose == helper.bind(value=query.purpose.value))

        if query.since is not None:
            helper.where(
                contexts_table.created_at
                >= helper.bind(value=self.__context._time(value=query.since))
            )

        if query.until is not None:
            helper.where(
                contexts_table.created_at
                < helper.bind(value=self.__context._time(value=query.until))
            )

        count_sql, count_parameters = helper.count_sql_and_parameters()
        page_sql, page_parameters = helper.page_sql_and_parameters(
            limit=query.limit + 1,
            cursor=self.__context._decode_keyset_cursor(value=query.cursor),
        )

        async with self.__context.unit.session() as connection:
            total = await self.__context._optional_count(
                connection=connection,
                sql=count_sql,
                parameters=count_parameters,
                requested=query.count_total,
            )
            async with connection.execute(page_sql, page_parameters) as cursor_rows:
                rows = await cursor_rows.fetchall()

        items, next_cursor = self.__context._paginate(
            limit=query.limit,
            timestamp=lambda context: context.created,
            identifier=lambda context: context.identity.id,
            rows=[self.__context.rows.context(row=row) for row in rows],
        )
        return ContextPage(items=tuple(items), next=next_cursor, total=total)

    async def __load_context_row(
        self,
        *,
        tenant: str,
        context: str,
        connection: PostgresConnectionProtocol,
    ) -> Optional[Context]:
        """
        Load one context row.
        """

        contexts_table = tables.CONTEXTS
        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)

        statement = (
            PostgreSQLQuery.from_(contexts_table)
            .select(contexts_table.star)
            .where(contexts_table.tenant == binder.bind(value=tenant))
            .where(contexts_table.id == binder.bind(value=context))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.context(row=row)

    def __references(self, *, value: References) -> JsonValue:
        """
        Convert a References model into a stable JSON-compatible payload.
        """

        return {
            "events": list(value.events),
            "messages": list(value.messages),
            "artifacts": list(value.artifacts),
            "memories": [
                {"system": memory.system, "reference": memory.reference}
                for memory in value.memories
            ],
        }

    def __same_context(self, *, context: Context, request: BuildContext) -> bool:
        """
        Check whether a build context request replays an already stored recipe.
        """

        return (
            context.identity.tenant == request.identity.tenant
            and context.identity.workspace == request.identity.workspace
            and context.thread == request.thread
            and context.task == request.task
            and context.consumer == request.consumer
            and context.purpose == request.purpose
            and context.builder == request.builder
            and context.references == request.references
            and context.budget == request.budget
            and context.filters == request.filters
            and context.hash == request.hash
            and context.provider == request.provider
            and context.model == request.model
            and context.created == request.created
            and context.expires == request.expires
            and context.metadata == request.metadata
        )
