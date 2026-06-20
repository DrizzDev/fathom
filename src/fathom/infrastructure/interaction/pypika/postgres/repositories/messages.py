from __future__ import annotations

from typing import List

from pypika import PostgreSQLQuery

from fathom.constants.collaboration import EventKind, EventSource
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import PostgresContext
from fathom.infrastructure.interaction.pypika.query import (
    CursorPaginatedQuery,
    ParameterizedQuery,
    SortDirection,
)
from fathom.infrastructure.interaction.pypika.query import (
    SortOrder as KeysetSortOrder,
)
from fathom.schemas.interaction import (
    Content,
    Message,
    MessageCursorQuery,
    MessagePage,
    MessageQuery,
    Metadata,
    RecordMessage,
    Sanitize,
    SortOrder,
)


class PostgresMessageRepository:
    """
    Postgres message repository: persists, sanitizes, and lists thread messages.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for message persistence.
        """

        self.__context = context

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message in a thread and optional task.

        Idempotent-replay equality compares content treating labels as a set.
        """

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_message(
                connection=connection,
                message=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None:
                if not self.__same_record_message_input(message=existing, request=request):
                    raise InteractionError(
                        "Message identity already exists with different content."
                    )

                return existing

            await self.__context._require_thread(
                connection=connection,
                thread=request.thread,
                tenant=request.identity.tenant,
            )
            await self.__context._require_active_membership(
                actor=request.author,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

            task_state = None
            if request.task is not None:
                task = await self.__context._load_task(
                    task=request.task,
                    connection=connection,
                    tenant=request.identity.tenant,
                )
                if task is None:
                    raise InteractionError("Message task does not exist.")

                if task.thread != request.thread:
                    raise InteractionError("Message task belongs to a different thread.")

                task_state = task.state

            if request.reply is not None:
                await self.__context._require_message_in_thread(
                    connection=connection,
                    tenant=request.identity.tenant,
                    thread=request.thread,
                    message=request.reply,
                )

            self.__context.lifecycle.validate_message_recording(task_state=task_state)

            if request.sequence is None:
                sequence = await self.__context._next_message_sequence(
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                )
            else:
                sequence = request.sequence

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            messages = tables.MESSAGES
            statement = (
                PostgreSQLQuery.into(messages)
                .columns(
                    messages.id,
                    messages.tenant,
                    messages.workspace,
                    messages.thread,
                    messages.task,
                    messages.author,
                    messages.reply,
                    messages.sequence,
                    messages.kind,
                    messages.audience,
                    messages.body,
                    messages.labels,
                    messages.sanitized_at,
                    messages.sanitizer,
                    messages.created_at,
                    messages.deleted_at,
                    messages.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.thread),
                    binder.bind(value=request.task),
                    binder.bind(value=request.author),
                    binder.bind(value=request.reply),
                    binder.bind(value=sequence),
                    binder.bind(value=request.kind.value),
                    binder.bind(value=request.audience.value),
                    binder.bind(value=self.__context._json(value=request.content.body)),
                    binder.bind(
                        value=self.__context._json(
                            value=[label.value for label in request.content.labels]
                        )
                    ),
                    binder.bind(
                        value=self.__context._optional_time(value=request.content.sanitized)
                    ),
                    binder.bind(value=request.content.sanitizer),
                    binder.bind(value=self.__context._time(value=request.created)),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            message = await self.__context._load_message(
                connection=connection,
                message=request.identity.id,
                tenant=request.identity.tenant,
            )
            await self.__context._record_event(
                task=request.task,
                actor=request.author,
                thread=request.thread,
                connection=connection,
                created=request.created,
                subject=request.identity.id,
                tenant=request.identity.tenant,
                source=EventSource.INTERACTION,
                kind=EventKind.MESSAGE_RECORDED,
                workspace=request.identity.workspace,
                payload=Metadata(
                    entries={"kind": request.kind.value, "audience": request.audience.value}
                ),
            )

        if message is None:
            raise InteractionError("Message was not persisted.")

        return message

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace stored message content with a sanitized version and event.
        """

        target = request.content.model_copy(update={"sanitized": request.sanitized})

        if target.sanitizer is None:
            raise InteractionError("Sanitized content must include a sanitizer.")

        async with self.__context.unit.session() as connection:
            message = await self.__context._load_message(
                connection=connection,
                tenant=request.tenant,
                message=request.message,
            )
            if message is None:
                raise InteractionError("Message does not exist.")

            if message.content.sanitized is not None:
                if self.__same_content(stored=message.content, target=target):
                    return message

                raise InteractionError("Message already sanitized with different content.")

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            messages = tables.MESSAGES
            statement = (
                PostgreSQLQuery.update(messages)
                .set(messages.body, binder.bind(value=self.__context._json(value=target.body)))
                .set(
                    messages.labels,
                    binder.bind(
                        value=self.__context._json(value=[label.value for label in target.labels])
                    ),
                )
                .set(
                    messages.sanitized_at,
                    binder.bind(value=self.__context._time(value=request.sanitized)),
                )
                .set(messages.sanitizer, binder.bind(value=target.sanitizer))
                .where(messages.tenant == binder.bind(value=request.tenant))
                .where(messages.id == binder.bind(value=request.message))
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            sanitized = await self.__context._load_message(
                connection=connection,
                tenant=request.tenant,
                message=request.message,
            )
            await self.__context._record_event(
                task=message.task,
                actor=message.author,
                connection=connection,
                tenant=request.tenant,
                thread=message.thread,
                subject=request.message,
                source=EventSource.POLICY,
                kind=EventKind.CONTENT_SANITIZED,
                workspace=message.identity.workspace,
                payload=Metadata(
                    entries={
                        "sanitizer": target.sanitizer,
                        "labels": [label.value for label in target.labels],
                    }
                ),
                created=request.sanitized,
            )

        if sanitized is None:
            raise InteractionError("Message was not sanitized.")

        return sanitized

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load tenant-scoped messages for one thread and optional task.
        """

        if query.task is None:
            async with (
                self.__context.unit.session() as connection,
                connection.execute(
                    """
                    SELECT *
                    FROM messages
                    WHERE tenant = $1 AND thread = $2 AND deleted_at IS NULL
                    ORDER BY sequence ASC
                    """,
                    (query.tenant, query.thread),
                ) as cursor,
            ):
                rows = await cursor.fetchall()
        else:
            async with (
                self.__context.unit.session() as connection,
                connection.execute(
                    """
                    SELECT *
                    FROM messages
                    WHERE tenant = $1 AND thread = $2 AND task = $3 AND deleted_at IS NULL
                    ORDER BY sequence ASC
                    """,
                    (query.tenant, query.thread, query.task),
                ) as cursor,
            ):
                rows = await cursor.fetchall()

        return [self.__context.rows.message(row=row) for row in rows]

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load messages with SQL-side cursor pagination.
        """

        messages = tables.MESSAGES
        direction = (
            SortDirection.DESCENDING if query.order is SortOrder.DESC else SortDirection.ASCENDING
        )
        helper = CursorPaginatedQuery(
            table=messages,
            ordering=KeysetSortOrder(
                tiebreaker="id",
                column="created_at",
                direction=direction,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )
        helper.where(messages.tenant == helper.bind(value=query.tenant))
        helper.where(messages.thread == helper.bind(value=query.thread))
        helper.where(messages.deleted_at.isnull())
        if query.task is not None:
            helper.where(messages.task == helper.bind(value=query.task))
        if query.author is not None:
            helper.where(messages.author == helper.bind(value=query.author))
        if query.kinds:
            helper.where(
                messages.kind.isin([helper.bind(value=kind.value) for kind in query.kinds])
            )
        if query.since is not None:
            helper.where(
                messages.created_at >= helper.bind(value=self.__context._time(value=query.since))
            )
        if query.until is not None:
            helper.where(
                messages.created_at < helper.bind(value=self.__context._time(value=query.until))
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
            timestamp=lambda message: message.created,
            identifier=lambda message: message.identity.id,
            rows=[self.__context.rows.message(row=row) for row in rows],
        )
        return MessagePage(items=tuple(items), next=next_cursor, total=total)

    def __same_record_message_input(self, *, message: Message, request: RecordMessage) -> bool:
        """
        Replay-equality variant for persisted messages.
        """

        if not (
            message.task == request.task
            and message.kind == request.kind
            and message.reply == request.reply
            and message.thread == request.thread
            and message.author == request.author
            and message.created == request.created
            and message.audience == request.audience
            and message.metadata == request.metadata
            and message.identity.tenant == request.identity.tenant
            and message.identity.workspace == request.identity.workspace
        ):
            return False

        if message.content.body != request.content.body:
            return False

        if message.content.sanitized != request.content.sanitized:
            return False

        if message.content.sanitizer != request.content.sanitizer:
            return False

        return frozenset(request.content.labels) == frozenset(message.content.labels)

    def __same_content(self, *, stored: Content, target: Content) -> bool:
        """
        Compare two Content values treating labels as an order-independent set.
        """

        return (
            stored.body == target.body
            and stored.sanitized == target.sanitized
            and stored.sanitizer == target.sanitizer
            and frozenset(stored.labels) == frozenset(target.labels)
        )
