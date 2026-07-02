from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    Audience,
    EventKind,
    EventSource,
    Label,
    MessageKind,
    TaskState,
)
from fathom.constants.conversation import SequenceScope
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ExecutionRecord,
    MessageRecord,
    TaskRecord,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    SequenceAllocator,
    TransactionScope,
)
from fathom.infrastructure.interaction.orm.repositories.paginator import (
    KeysetPaginator,
    TimestampColumn,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.interaction.lifecycle import Lifecycle
from fathom.schemas.interaction import (
    Content,
    Identity,
    Message,
    MessageCursorQuery,
    MessagePage,
    MessageQuery,
    Metadata,
    RecordMessage,
    Sanitize,
    ThreadReference,
    ThreadScope,
    Visibility,
)

if TYPE_CHECKING:
    from datetime import datetime


class MessageRepository:
    """
    persistent-store backed repository for durable conversation messages.
    """

    def __init__(
        self,
        *,
        lifecycle: Lifecycle,
        recorder: LifecycleRecorder,
        references: ReferenceGuard,
        sequences: SequenceAllocator,
        transaction: TransactionScope,
    ) -> None:
        """
        Initialize message persistence collaborators.
        """

        self.__recorder = recorder
        self.__lifecycle = lifecycle

        self.__guard = references
        self.__sequences = sequences
        self.__transaction = transaction

    async def record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message or replay an identical existing message.
        """

        try:
            return await self.__record_message(request=request)
        except IntegrityError as exception:
            existing = await self.__load_message(
                connection=None,
                message=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None and self.__same_record_message_input(
                request=request, message=existing
            ):
                return existing

            raise InteractionError("Message insert conflicted with a different row.") from exception

    async def __record_message(self, *, request: RecordMessage) -> Message:
        """
        Persist one message inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            if existing := await self.__load_message(
                connection=connection,
                message=request.identity.id,
                tenant=request.identity.tenant,
            ):
                if not self.__same_record_message_input(message=existing, request=request):
                    raise InteractionError(
                        "Message identity already exists with different content."
                    )

                return existing

            await self.__guard.active_thread(
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            await self.__guard.active_membership(
                actor=request.author,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            task_state = None

            if request.task is not None:
                task_state = await self.__require_task_in_thread(
                    task=request.task,
                    thread=request.thread,
                    connection=connection,
                    tenant=request.identity.tenant,
                )
            if request.reply is not None:
                await self.__guard.present_message(
                    thread=request.thread,
                    message=request.reply,
                    connection=connection,
                    tenant=request.identity.tenant,
                )

            self.__lifecycle.validate_message_recording(task_state=task_state)
            sequence = request.sequence

            if sequence is None:
                sequence = await self.__sequences.next(
                    scope=SequenceScope.MESSAGE.value,
                    connection=connection,
                    thread=request.thread,
                    tenant=request.identity.tenant,
                )

            execution_id = await self.__execution(
                task=request.task,
                thread=request.thread,
                execution=request.execution,
                connection=connection,
                tenant=request.identity.tenant,
            )

            await MessageRecord.create(
                sequence=sequence,
                using_db=connection,
                task_id=request.task,
                author=request.author,
                reply_id=request.reply,
                id=request.identity.id,
                kind=request.kind.value,
                updated_by=request.author,
                created_by=request.author,
                execution_id=execution_id,
                body=request.content.body,
                created_at=request.created,
                conversation_id=request.thread,
                audience=[request.audience.value],
                tenant_id=request.identity.tenant,
                metadata=request.metadata.entries,
                sanitizer=request.content.sanitizer,
                sanitized_at=request.content.sanitized,
                workspace_id=request.identity.workspace,
                labels=[label.value for label in request.content.labels],
            )
            message = await self.__load_message(
                connection=connection,
                message=request.identity.id,
                tenant=request.identity.tenant,
            )
            if message is None:
                raise InteractionError("Message was not persisted.")

            await self.__recorder.record(
                task=request.task,
                actor=request.author,
                thread=request.thread,
                connection=connection,
                execution=execution_id,
                created=request.created,
                tenant=request.identity.tenant,
                kind=EventKind.MESSAGE_RECORDED,
                workspace=request.identity.workspace,
                payload=Metadata(
                    entries={"kind": request.kind.value, "audience": request.audience.value}
                ),
            )

            return message

    async def sanitize_message(self, *, request: Sanitize) -> Message:
        """
        Replace stored message content with sanitized content.
        """

        target = request.content.model_copy(update={"sanitized": request.sanitized})
        if target.sanitizer is None:
            raise InteractionError("Sanitized content must include a sanitizer.")

        async with self.__transaction.transaction() as connection:
            message = await self.__load_message(
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

            await (
                MessageRecord.filter(
                    tenant_id=request.tenant,
                    id=request.message,
                    **Visibility(archived=True).as_filters(),
                )
                .using_db(connection)
                .update(
                    body=target.body,
                    updated_by=message.author,
                    sanitizer=target.sanitizer,
                    sanitized_at=request.sanitized,
                    labels=[label.value for label in target.labels],
                )
            )
            sanitized = await self.__load_message(
                tenant=request.tenant,
                connection=connection,
                message=request.message,
            )
            if sanitized is None:
                raise InteractionError("Message was not sanitized.")

            await self.__recorder.record(
                task=message.task,
                actor=message.author,
                connection=connection,
                tenant=request.tenant,
                thread=message.thread,
                created=request.sanitized,
                source=EventSource.POLICY,
                execution=message.execution,
                kind=EventKind.CONTENT_SANITIZED,
                workspace=message.identity.workspace,
                payload=Metadata(
                    entries={
                        "sanitizer": target.sanitizer,
                        "labels": [label.value for label in target.labels],
                    }
                ),
            )

        return sanitized

    async def get_messages(self, *, query: MessageQuery) -> List[Message]:
        """
        Load active tenant-scoped messages for one thread and optional task.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return []

        queryset = MessageRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)

        rows = await queryset.order_by("sequence")
        return [self.__message(row=row) for row in rows]

    async def list_messages(self, *, query: MessageCursorQuery) -> MessagePage:
        """
        Load active messages with keyset pagination.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return MessagePage(items=(), next=None, total=0)

        queryset = MessageRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )

        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)

        if query.author is not None:
            queryset = queryset.filter(author=query.author)

        if query.since is not None:
            queryset = queryset.filter(created_at__gte=query.since)

        if query.until is not None:
            queryset = queryset.filter(created_at__lt=query.until)

        if query.kinds:
            queryset = queryset.filter(kind__in=tuple(kind.value for kind in query.kinds))

        total = await queryset.count() if query.count_total else 0

        page = await KeysetPaginator[MessageRecord, Message](
            column=TimestampColumn.CREATED,
        ).paginate(
            queryset=queryset,
            limit=query.limit,
            order=query.order,
            cursor=query.cursor,
            project=self.__page_message,
            stamp=self.__message_created,
            identity=self.__message_identity,
        )

        return MessagePage(items=page.items, next=page.next, total=total)

    def __scope(self, *, query: MessageQuery | MessageCursorQuery) -> ThreadScope:
        """
        Build a thread scope from a message read query.
        """

        return ThreadScope(
            reference=ThreadReference(tenant=query.tenant, thread=query.thread),
            visibility=Visibility(
                deleted=query.include_deleted,
                archived=query.include_archived,
            ),
        )

    async def __load_message(
        self,
        *,
        tenant: str,
        message: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Message]:
        """
        Load one active message row.
        """

        queryset = MessageRecord.filter(
            tenant_id=tenant,
            id=message,
            **Visibility(archived=True).as_filters(),
        )
        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()

        if row is None:
            return None

        return self.__message(row=row)

    async def __execution(
        self,
        *,
        tenant: str,
        thread: str,
        task: Optional[str],
        execution: Optional[str],
        connection: DatabaseConnection,
    ) -> Optional[str]:
        """
        Resolve and validate the execution id for the message.
        """

        if task is None:
            if execution is None:
                raise InteractionError("Message execution is required.")

            row = (
                await ExecutionRecord.filter(
                    tenant_id=tenant,
                    id=execution,
                    conversation_id=thread,
                    **Visibility(archived=True).as_filters(),
                )
                .using_db(connection)
                .get_or_none()
            )
            if row is None:
                raise InteractionError("Message execution does not exist.")

            return execution

        row = (
            await TaskRecord.filter(
                tenant_id=tenant,
                id=task,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )

        if row is None:
            raise InteractionError("Message task does not exist.")

        task_execution = row.execution_id

        if task_execution is not None and not isinstance(task_execution, str):
            raise InteractionError("Message task execution id is invalid.")

        if execution is not None and execution != task_execution:
            raise InteractionError("Message execution does not match task execution.")

        return task_execution

    async def __require_task_in_thread(
        self,
        *,
        task: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> Optional[TaskState]:
        """
        Require that a task exists in the expected thread and return its state.
        """

        row = (
            await TaskRecord.filter(
                tenant_id=tenant,
                id=task,
                conversation_id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )

        if row is None:
            raise InteractionError("Message task does not exist.")

        return self.__task_state(value=row.state)

    def __same_record_message_input(self, *, message: Message, request: RecordMessage) -> bool:
        """
        Check whether a message request matches an already stored message.
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

        if request.execution is not None and message.execution != request.execution:
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
        Compare two content values treating labels as an order-independent set.
        """

        return (
            stored.body == target.body
            and stored.sanitized == target.sanitized
            and stored.sanitizer == target.sanitizer
            and frozenset(stored.labels) == frozenset(target.labels)
        )

    def __message(self, *, row: MessageRecord) -> Message:
        """
        Convert one persistent message model into the interaction schema.
        """

        return Message(
            task=row.task_id,
            author=row.author,
            reply=row.reply_id,
            sequence=row.sequence,
            created_at=row.created_at,
            deleted_at=row.deleted_at,
            execution=row.execution_id,
            thread=row.conversation_id,
            kind=self.__kind(value=row.kind),
            audience=self.__audience(value=row.audience),
            content=Content(
                body=row.body,
                sanitizer=row.sanitizer,
                sanitized_at=row.sanitized_at,
                labels=self.__labels(value=row.labels),
            ),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __page_message(self, row: MessageRecord) -> Message:
        """
        Convert one message row for pagination.
        """

        return self.__message(row=row)

    def __message_created(self, message: Message) -> datetime:
        """
        Return the message creation timestamp used by pagination.
        """

        return message.created

    def __message_identity(self, message: Message) -> str:
        """
        Return the message identifier used by pagination.
        """

        return message.identity.id

    def __kind(self, *, value: str) -> MessageKind:
        """
        Convert stored message kind text into the public enum.
        """

        try:
            return MessageKind(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown message kind in row: {value}.") from exception

    def __audience(self, *, value: JsonValue) -> Audience:
        """
        Convert stored message audience JSON into the public enum.
        """

        if not isinstance(value, list) or len(value) != 1:
            raise InteractionError("Invalid message audience in row.")

        audience = value[0]

        if not isinstance(audience, str):
            raise InteractionError("Invalid message audience in row.")

        try:
            return Audience(audience)
        except ValueError as exception:
            raise InteractionError(f"Unknown message audience in row: {audience}.") from exception

    def __task_state(self, *, value: str) -> TaskState:
        """
        Convert stored task state text into the public enum.
        """

        try:
            return TaskState(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown task state in row: {value}.") from exception

    def __labels(self, *, value: JsonValue) -> Tuple[Label, ...]:
        """
        Convert stored label strings into label enums.
        """

        if not isinstance(value, list):
            raise InteractionError("Invalid message labels in row.")

        labels: List[Label] = []

        for label in value:
            if not isinstance(label, str):
                raise InteractionError("Invalid message label in row.")

            try:
                labels.append(Label(label))
            except ValueError as exception:
                raise InteractionError(f"Unknown message label in row: {label}.") from exception

        return tuple(labels)

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError(f"Invalid message {field} in row.")
