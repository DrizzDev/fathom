from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import ContextPurpose, EventKind
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ContextRecord,
    TaskRecord,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    TransactionScope,
)
from fathom.infrastructure.interaction.orm.repositories.paginator import (
    KeysetPaginator,
    TimestampColumn,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.schemas.interaction import (
    BuildContext,
    Context,
    ContextCursorQuery,
    ContextPage,
    ContextQuery,
    Identity,
    MemoryReference,
    Metadata,
    References,
    ThreadReference,
    ThreadScope,
    Visibility,
)

if TYPE_CHECKING:
    from datetime import datetime


class ContextRepository:
    """
    Repository for reusable context assembly records.
    """

    def __init__(
        self,
        *,
        references: ReferenceGuard,
        lifecycle: LifecycleRecorder,
        transaction: TransactionScope,
    ) -> None:
        """
        Initialize context persistence collaborators.
        """

        self.__guard = references
        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one context recipe or replay an identical recipe.
        """

        try:
            return await self.__build_context(request=request)
        except IntegrityError as exception:
            existing = await self.__load_context(
                connection=None,
                context=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None and self.__same_context(request=request, context=existing):
                return existing

            raise InteractionError("Context insert conflicted with a different row.") from exception

    async def __build_context(self, *, request: BuildContext) -> Context:
        """
        Persist one context recipe inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_context(
                connection=connection, context=request.identity.id, tenant=request.identity.tenant
            )
            if existing is not None:
                if not self.__same_context(context=existing, request=request):
                    raise InteractionError(
                        "Context identity already exists with different content."
                    )

                return existing

            execution = await self.__require_references(request=request, connection=connection)

            await ContextRecord.create(
                hash=request.hash,
                model=request.model,
                using_db=connection,
                task_id=request.task,
                execution_id=execution,
                id=request.identity.id,
                builder=request.builder,
                consumer=request.consumer,
                provider=request.provider,
                created_at=request.created,
                updated_by=request.consumer,
                created_by=request.consumer,
                expires_at=request.expires,
                purpose=request.purpose.value,
                budget=request.budget.entries,
                conversation_id=request.thread,
                filters=request.filters.entries,
                tenant_id=request.identity.tenant,
                metadata=request.metadata.entries,
                workspace_id=request.identity.workspace,
                references=self.__references_json(value=request.references),
            )

            await self.__lifecycle.record(
                task=request.task,
                execution=execution,
                connection=connection,
                thread=request.thread,
                actor=request.consumer,
                created=request.created,
                kind=EventKind.CONTEXT_BUILT,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                payload=Metadata(
                    entries={"purpose": request.purpose.value, "builder": request.builder}
                ),
            )
            context = await self.__load_context(
                connection=connection, context=request.identity.id, tenant=request.identity.tenant
            )
            if context is None:
                raise InteractionError("Context was not persisted.")

            return context

    async def get_contexts(self, *, query: ContextQuery) -> List[Context]:
        """
        Load visible contexts scoped to one thread.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return []

        queryset = ContextRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)
        if query.execution is not None:
            queryset = queryset.filter(execution_id=query.execution)
        if query.purpose is not None:
            queryset = queryset.filter(purpose=query.purpose.value)

        rows = await queryset.order_by("created_at", "id")
        return [self.__context(row=row) for row in rows]

    async def list_contexts(self, *, query: ContextCursorQuery) -> ContextPage:
        """
        Load visible contexts with keyset pagination ordered by creation timestamp.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return ContextPage(items=(), next=None, total=0)

        queryset = ContextRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)
        if query.execution is not None:
            queryset = queryset.filter(execution_id=query.execution)
        if query.consumer is not None:
            queryset = queryset.filter(consumer=query.consumer)
        if query.purpose is not None:
            queryset = queryset.filter(purpose=query.purpose.value)
        if query.since is not None:
            queryset = queryset.filter(created_at__gte=query.since)
        if query.until is not None:
            queryset = queryset.filter(created_at__lt=query.until)

        total = await queryset.count() if query.count_total else 0

        page = await KeysetPaginator[ContextRecord, Context](
            column=TimestampColumn.CREATED,
        ).paginate(
            queryset=queryset,
            limit=query.limit,
            order=query.order,
            cursor=query.cursor,
            project=self.__page_context,
            stamp=self.__context_created,
            identity=self.__context_identity,
        )

        return ContextPage(items=page.items, next=page.next, total=total)

    async def __load_context(
        self,
        *,
        tenant: str,
        context: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Context]:
        """
        Load one context row by identity.
        """

        queryset = ContextRecord.filter(
            tenant_id=tenant,
            id=context,
            **Visibility(archived=True).as_filters(),
        )

        if connection is not None:
            queryset = queryset.using_db(connection)

        if row := await queryset.get_or_none():
            return self.__context(row=row)

        return None

    async def __require_references(
        self,
        *,
        request: BuildContext,
        connection: DatabaseConnection,
    ) -> Optional[str]:
        """
        Validate local references used by a context recipe.
        """

        await self.__guard.active_thread(
            thread=request.thread, connection=connection, tenant=request.identity.tenant
        )

        execution = request.execution

        if request.task is not None:
            task_execution = await self.__require_task_in_thread(
                task=request.task,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            if execution is not None and task_execution != execution:
                raise InteractionError("Context execution does not match task execution.")

            execution = task_execution

        elif execution is not None:
            await self.__guard.present_execution(
                execution=execution,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

        else:
            raise InteractionError("Context execution is required.")

        if request.consumer is not None:
            await self.__guard.active_membership(
                connection=connection,
                thread=request.thread,
                actor=request.consumer,
                tenant=request.identity.tenant,
            )

        for message in request.references.messages:
            await self.__guard.present_message(
                message=message,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

        for event in request.references.events:
            await self.__guard.present_event(
                event=event,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

        for artifact in request.references.artifacts:
            await self.__guard.present_artifact(
                artifact=artifact,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )

        return execution

    async def __require_task_in_thread(
        self,
        *,
        task: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> str:
        """
        Require a live task in the target thread.
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
            raise InteractionError("Task does not exist.")

        return str(row.execution_id)

    def __scope(self, *, query: ContextQuery | ContextCursorQuery) -> ThreadScope:
        """
        Build a thread scope from a context read query.
        """

        return ThreadScope(
            reference=ThreadReference(tenant=query.tenant, thread=query.thread),
            visibility=Visibility(
                deleted=query.include_deleted,
                archived=query.include_archived,
            ),
        )

    def __context(self, *, row: ContextRecord) -> Context:
        """
        Convert one context row into the interaction schema.
        """

        return Context(
            hash=row.hash,
            model=row.model,
            task=row.task_id,
            builder=row.builder,
            provider=row.provider,
            consumer=row.consumer,
            created_at=row.created_at,
            expires_at=row.expires_at,
            thread=row.conversation_id,
            execution=row.execution_id,
            purpose=self.__purpose(value=row.purpose),
            references=self.__references(value=row.references),
            budget=self.__metadata(value=row.budget, field="budget"),
            filters=self.__metadata(value=row.filters, field="filters"),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __page_context(self, row: ContextRecord) -> Context:
        """
        Convert one context row for pagination.
        """

        return self.__context(row=row)

    def __context_created(self, context: Context) -> datetime:
        """
        Return the context creation timestamp used by pagination.
        """

        return context.created

    def __context_identity(self, context: Context) -> str:
        """
        Return the context identifier used by pagination.
        """

        return context.identity.id

    def __references(self, *, value: JsonValue) -> References:
        """
        Validate stored context references.
        """

        if not isinstance(value, dict):
            raise InteractionError("Stored context references are not an object.")

        events = self.__string_tuple(value=value.get("events"), field="events")
        messages = self.__string_tuple(value=value.get("messages"), field="messages")
        artifacts = self.__string_tuple(value=value.get("artifacts"), field="artifacts")

        memories = value.get("memories", ())
        if not isinstance(memories, list):
            raise InteractionError("Stored context memories are not a list.")

        memory_references: List[MemoryReference] = []

        for memory in memories:
            if not isinstance(memory, dict):
                raise InteractionError("Stored context memory reference is not an object.")

            system = memory.get("system")
            reference = memory.get("reference")

            if not isinstance(system, str) or not isinstance(reference, str):
                raise InteractionError("Stored context memory reference is invalid.")

            memory_references.append(MemoryReference(system=system, reference=reference))

        return References(
            events=events,
            messages=messages,
            artifacts=artifacts,
            memories=tuple(memory_references),
        )

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Validate one stored JSON object as metadata.
        """

        if not isinstance(value, dict):
            raise InteractionError(f"Stored context {field} is not an object.")

        if not all(isinstance(key, str) for key in value):
            raise InteractionError(f"Stored context {field} contains a non-string key.")

        return Metadata(entries=value)

    def __string_tuple(self, *, value: Optional[JsonValue], field: str) -> Tuple[str, ...]:
        """
        Validate one stored reference id list.
        """

        if value is None:
            return ()

        if not isinstance(value, list):
            raise InteractionError(f"Stored context {field} references are not a list.")

        items: List[str] = []

        for item in value:
            if not isinstance(item, str):
                raise InteractionError(f"Stored context {field} references contain non-strings.")

            items.append(item)

        return tuple(items)

    def __purpose(self, *, value: str) -> ContextPurpose:
        """
        Convert a stored context purpose into an enum.
        """

        try:
            return ContextPurpose(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown context purpose in row: {value}.") from exception

    def __references_json(self, *, value: References) -> Dict[str, JsonValue]:
        """
        Convert references into a stable JSON-compatible object.
        """

        events: List[JsonValue] = list(value.events)
        messages: List[JsonValue] = list(value.messages)
        artifacts: List[JsonValue] = list(value.artifacts)
        memories: List[JsonValue] = [
            {"system": memory.system, "reference": memory.reference} for memory in value.memories
        ]

        references: Dict[str, JsonValue] = {
            "events": events,
            "messages": messages,
            "memories": memories,
            "artifacts": artifacts,
        }

        return references

    def __same_context(self, *, context: Context, request: BuildContext) -> bool:
        """
        Check whether a build request replays an existing context.
        """

        return (
            context.task == request.task
            and context.hash == request.hash
            and context.model == request.model
            and context.thread == request.thread
            and context.budget == request.budget
            and context.purpose == request.purpose
            and context.builder == request.builder
            and context.filters == request.filters
            and context.created == request.created
            and context.expires == request.expires
            and context.identity == request.identity
            and context.consumer == request.consumer
            and context.provider == request.provider
            and context.metadata == request.metadata
            and context.references == request.references
            and (request.execution is None or context.execution == request.execution)
        )
