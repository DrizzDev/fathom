from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Type

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError
from tortoise.models import Model

from fathom.constants.collaboration import EventKind, ThreadState
from fathom.core.exceptions import InteractionError, ThreadConflictError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ArtifactRecord,
    ContextRecord,
    ConversationRecord,
    ExecutionRecord,
    JobRecord,
    MembershipRecord,
    MessageRecord,
    ScriptRecord,
    SequenceRecord,
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
from fathom.schemas.interaction import (
    CreateThread,
    Identity,
    MembershipVisibility,
    Metadata,
    SetThreadTitle,
    SortOrder,
    Thread,
    ThreadListQuery,
    ThreadPage,
    ThreadQuery,
    ThreadTransition,
    Timing,
    Visibility,
)

if TYPE_CHECKING:
    from datetime import datetime


class ThreadRepository:
    """
    Persistent-store backed repository for durable conversation threads.
    """

    def __init__(self, *, lifecycle: LifecycleRecorder, transaction: TransactionScope) -> None:
        """
        Initialize thread persistence collaborators.
        """

        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def create_thread(self, *, request: CreateThread) -> Thread:
        """
        Persist one thread or replay an identical existing thread.
        """

        try:
            return await self.__create_thread(request=request)
        except IntegrityError as exception:
            existing = await self.__load_live_thread(
                connection=None,
                thread=request.identity.id,
                tenant=request.identity.tenant,
            )

            if existing is not None:
                return self.__replay(thread=existing, request=request)

            raise InteractionError("Thread insert failed before reload.") from exception

    async def __create_thread(self, *, request: CreateThread) -> Thread:
        """
        Persist one thread inside a transaction.
        """

        async with self.__transaction.transaction() as connection:
            if existing := await self.__load_live_thread(
                connection=connection,
                thread=request.identity.id,
                tenant=request.identity.tenant,
            ):
                return self.__replay(thread=existing, request=request)

            if request.creator is not None:
                await self.__require_actor(
                    actor=request.creator,
                    connection=connection,
                    tenant=request.identity.tenant,
                )

            await ConversationRecord.create(
                title=request.title,
                using_db=connection,
                id=request.identity.id,
                created_at=request.created,
                created_by=request.creator,
                updated_by=request.creator,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
            )

            thread = await self.__load_live_thread(
                connection=connection,
                thread=request.identity.id,
                tenant=request.identity.tenant,
            )
            if thread is None:
                raise InteractionError("Thread was not persisted.")

            await self.__lifecycle.record(
                connection=connection,
                actor=request.creator,
                created=request.created,
                thread=thread.identity.id,
                kind=EventKind.THREAD_CREATED,
                tenant=thread.identity.tenant,
                workspace=thread.identity.workspace,
                payload=Metadata(entries={"state": request.state.value}),
            )
            created = await self.__load_live_thread(
                connection=connection,
                thread=request.identity.id,
                tenant=request.identity.tenant,
            )
            if created is None:
                raise InteractionError("Thread was not persisted.")

        return created

    async def get_thread(self, *, query: ThreadQuery) -> Optional[Thread]:
        """
        Load one active tenant-scoped thread.
        """

        return await self.__load_thread(
            connection=None,
            tenant=query.tenant,
            thread=query.thread,
            include_archived=query.include_archived,
            include_deleted=query.include_deleted,
        )

    async def set_thread_title(self, *, request: SetThreadTitle) -> Thread:
        """
        Set or replace a thread title.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_active_thread(
                tenant=request.tenant,
                thread=request.thread,
                connection=connection,
            )
            if existing is None:
                raise InteractionError("Thread does not exist.")

            metadata = dict(existing.metadata.entries)
            if request.metadata.entries:
                metadata["title"] = request.metadata.entries

            await (
                ConversationRecord.filter(
                    id=request.thread,
                    tenant_id=request.tenant,
                    **Visibility().as_filters(),
                )
                .using_db(connection)
                .update(
                    metadata=metadata,
                    title=request.title,
                    updated_at=request.updated,
                )
            )
            updated = await self.__load_active_thread(
                tenant=request.tenant,
                thread=request.thread,
                connection=connection,
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

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_live_thread(
                tenant=request.tenant,
                thread=request.thread,
                connection=connection,
            )
            if existing is None:
                raise InteractionError("Thread does not exist.")

            deleted = request.updated if request.state == ThreadState.DELETED else None
            archived = request.updated if request.state == ThreadState.ARCHIVED else None

            await (
                ConversationRecord.filter(
                    id=request.thread,
                    tenant_id=request.tenant,
                    **Visibility(archived=True).as_filters(),
                )
                .using_db(connection)
                .update(
                    deleted_at=deleted,
                    archived_at=archived,
                    updated_by=request.actor,
                    updated_at=request.updated,
                    deleted_by=request.actor if request.state == ThreadState.DELETED else None,
                )
            )

            if request.state == ThreadState.DELETED:
                await self.__soft_delete_children(
                    actor=request.actor,
                    tenant=request.tenant,
                    thread=request.thread,
                    connection=connection,
                    deleted=request.updated,
                )

                await self.__lifecycle.record(
                    actor=request.actor,
                    connection=connection,
                    created=request.updated,
                    thread=existing.identity.id,
                    kind=EventKind.THREAD_DELETED,
                    tenant=existing.identity.tenant,
                    workspace=existing.identity.workspace,
                    payload=Metadata(entries={"state": request.state.value}),
                    touch_thread=self.__touch_transition(state=request.state),
                )
                return existing.model_copy(
                    update={
                        "archived": None,
                        "deleted": request.updated,
                        "state": ThreadState.DELETED,
                        "timing": existing.timing.model_copy(update={"updated": request.updated}),
                    }
                )

            updated = await self.__load_live_thread(
                tenant=request.tenant,
                thread=request.thread,
                connection=connection,
            )
            if updated is None:
                raise InteractionError("Thread was not updated.")

            await self.__lifecycle.record(
                connection=connection,
                thread=updated.identity.id,
                tenant=updated.identity.tenant,
                workspace=updated.identity.workspace,
                kind=(
                    EventKind.THREAD_ARCHIVED
                    if request.state == ThreadState.ARCHIVED
                    else EventKind.THREAD_UNARCHIVED
                ),
                actor=request.actor,
                created=request.updated,
                payload=Metadata(entries={"state": request.state.value}),
            )

        return updated

    async def list_threads(self, *, query: ThreadListQuery) -> ThreadPage:
        """
        Load tenant-scoped threads with keyset pagination.
        """

        if query.actor is None:
            return ThreadPage(items=(), next=None, total=0)

        memberships = await MembershipRecord.filter(
            actor=query.actor,
            tenant_id=query.tenant,
            **MembershipVisibility(
                deleted=query.state is ThreadState.DELETED,
            ).as_filters(),
        ).values_list("conversation_id", flat=True)

        conversation_ids: List[str] = []
        for membership in memberships:
            if not isinstance(membership, str):
                raise InteractionError("Membership conversation id is invalid.")
            conversation_ids.append(membership)

        if not conversation_ids:
            return ThreadPage(items=(), next=None, total=0)

        if query.state is not None and query.state not in (
            ThreadState.ACTIVE,
            ThreadState.DELETED,
            ThreadState.ARCHIVED,
        ):
            return ThreadPage(items=(), next=None, total=0)

        queryset = ConversationRecord.filter(
            tenant_id=query.tenant,
            id__in=tuple(conversation_ids),
            **self.__lifecycle_filters(state=query.state, include_archived=query.include_archived),
        )
        if query.workspace is not None:
            queryset = queryset.filter(workspace_id=query.workspace)

        if query.title is not None:
            queryset = queryset.filter(title__istartswith=query.title)

        if query.updated_since is not None:
            queryset = queryset.filter(updated_at__gte=query.updated_since)

        if query.updated_until is not None:
            queryset = queryset.filter(updated_at__lt=query.updated_until)

        total = await queryset.count() if query.count_total else 0

        page = await KeysetPaginator[ConversationRecord, Thread](
            column=TimestampColumn.UPDATED,
        ).paginate(
            queryset=queryset,
            limit=query.limit,
            cursor=query.cursor,
            order=SortOrder.DESC,
            project=self.__page_thread,
            stamp=self.__thread_updated,
            identity=self.__thread_identity,
        )

        return ThreadPage(items=page.items, next=page.next, total=total)

    @staticmethod
    def __lifecycle_filters(
        *, state: Optional[ThreadState], include_archived: bool
    ) -> Dict[str, bool]:
        """
        Compose lifecycle visibility filters for the thread list query.
        """

        if state is ThreadState.DELETED:
            return {"deleted_at__isnull": False}

        if state is ThreadState.ARCHIVED:
            return {"deleted_at__isnull": True, "archived_at__isnull": False}

        if state is ThreadState.ACTIVE:
            return Visibility().as_filters()

        if include_archived:
            return Visibility(archived=True).as_filters()

        return Visibility().as_filters()

    async def __load_thread(
        self,
        *,
        tenant: str,
        thread: str,
        include_deleted: bool,
        include_archived: bool,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Thread]:
        """
        Load one thread row.
        """

        filters: Dict[str, object] = {"tenant_id": tenant, "id": thread}

        if not include_deleted:
            filters["deleted_at__isnull"] = True

        if not include_archived:
            filters["archived_at__isnull"] = True

        queryset = ConversationRecord.filter(**filters)

        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()
        if row is None:
            return None

        return self.__thread(row=row)

    async def __load_active_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Thread]:
        """
        Load one non-archived, non-deleted thread row.
        """

        return await self.__load_thread(
            tenant=tenant,
            thread=thread,
            connection=connection,
            include_deleted=False,
            include_archived=False,
        )

    async def __load_live_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Thread]:
        """
        Load one non-deleted thread row, including archived rows.
        """

        return await self.__load_thread(
            tenant=tenant,
            thread=thread,
            connection=connection,
            include_archived=True,
            include_deleted=False,
        )

    def __touch_transition(self, *, state: ThreadState) -> bool:
        """
        Return whether lifecycle recording should update the transitioned thread row.
        """

        return state is not ThreadState.DELETED

    async def __require_actor(
        self, *, tenant: str, actor: str, connection: DatabaseConnection
    ) -> None:
        """
        Require an actor to exist before a thread references it.
        """

        row = await ActorRecord.get_or_none(id=actor, tenant_id=tenant, using_db=connection)
        if row is None:
            raise InteractionError("Actor does not exist.")

    async def __soft_delete_children(
        self,
        *,
        tenant: str,
        thread: str,
        deleted: datetime,
        actor: Optional[str],
        connection: DatabaseConnection,
    ) -> None:
        """
        Soft-delete thread-owned rows that expose a deleted timestamp.
        """

        for model in (
            JobRecord,
            TaskRecord,
            ScriptRecord,
            MessageRecord,
            ContextRecord,
            SequenceRecord,
            ArtifactRecord,
            ExecutionRecord,
            MembershipRecord,
        ):
            await self.__soft_delete_model(
                model=model,
                actor=actor,
                tenant=tenant,
                thread=thread,
                deleted=deleted,
                connection=connection,
            )

    async def __soft_delete_model(
        self,
        *,
        tenant: str,
        thread: str,
        model: Type[Model],
        deleted: datetime,
        actor: Optional[str],
        connection: DatabaseConnection,
    ) -> None:
        """
        Soft-delete rows for one conversation-owned model.
        """

        queryset = model.filter(
            tenant_id=tenant,
            conversation_id=thread,
            **Visibility(archived=True).as_filters(),
        )

        await queryset.using_db(connection).update(
            updated_by=actor,
            deleted_by=actor,
            deleted_at=deleted,
            updated_at=deleted,
        )

    def __replay(self, *, thread: Thread, request: CreateThread) -> Thread:
        """
        Return identical replay rows and reject conflicting identity reuse.
        """

        if self.__same_thread(thread=thread, request=request):
            return thread

        raise ThreadConflictError(
            thread=request.identity.id,
            message="Thread identity already exists with different content.",
        )

    def __same_thread(self, *, thread: Thread, request: CreateThread) -> bool:
        """
        Check whether a thread request matches an already stored thread.
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

    def __thread(self, *, row: ConversationRecord) -> Thread:
        """
        Convert one persistent thread model into the interaction schema.
        """

        return Thread(
            cursor=None,
            title=row.title,
            digest=row.digest,
            creator=row.created_by,
            deleted_at=row.deleted_at,
            archived_at=row.archived_at,
            state=self.__state(row=row),
            metadata=self.__metadata(value=row.metadata),
            timing=Timing(created_at=row.created_at, updated_at=row.updated_at),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __page_thread(self, row: ConversationRecord) -> Thread:
        """
        Convert one thread row for pagination.
        """

        return self.__thread(row=row)

    def __thread_updated(self, thread: Thread) -> datetime:
        """
        Return the thread update timestamp used by pagination.
        """

        return thread.timing.updated

    def __thread_identity(self, thread: Thread) -> str:
        """
        Return the thread identifier used by pagination.
        """

        return thread.identity.id

    def __state(self, *, row: ConversationRecord) -> ThreadState:
        """
        Derive public conversation state from lifecycle timestamps.
        """

        if row.deleted_at is not None:
            return ThreadState.DELETED

        if row.archived_at is not None:
            return ThreadState.ARCHIVED

        return ThreadState.ACTIVE

    def __metadata(self, *, value: JsonValue) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError("Invalid thread metadata in row.")
