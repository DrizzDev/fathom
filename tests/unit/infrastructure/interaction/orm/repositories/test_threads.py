from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    EventKind,
    JobKind,
    JobState,
    MembershipRole,
    MembershipScope,
    MessageKind,
    ScriptFormat,
    ScriptStatus,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.core.exceptions import InteractionError, ThreadConflictError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ArtifactRecord,
    ContextRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    JobRecord,
    MembershipRecord,
    MessageRecord,
    ScriptRecord,
    SequenceRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    CreateThread,
    Identity,
    Metadata,
    SetThreadTitle,
    ThreadListQuery,
    ThreadQuery,
    ThreadTransition,
)


class TestThreadRepository:
    """
    Verify thread persistence through the persistent-store backed repository.
    """

    async def test_create_thread_persists_thread_and_records_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            actor = await self.__actor()
            request = self.__request(creator=actor, title="Plan")

            result = await InteractionRepositoryFactory().threads().create_thread(request=request)

            assert result.identity == request.identity
            assert result.title == "Plan"
            assert result.creator == actor
            assert result.state == ThreadState.ACTIVE
            stored = await ConversationRecord.get(id=request.identity.id)
            assert stored.digest is not None
            event = await EventRecord.get(conversation_id=request.identity.id, sequence=1)
            assert event.kind == EventKind.THREAD_CREATED.value

    async def test_identical_replay_returns_existing_without_new_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            actor = await self.__actor()
            request = self.__request(creator=actor)
            repository = InteractionRepositoryFactory().threads()

            created = await repository.create_thread(request=request)
            replayed = await repository.create_thread(request=request)

            assert replayed == created
            assert await EventRecord.filter(conversation_id=request.identity.id).count() == 1

    async def test_concurrent_identical_replay_creates_one_thread(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            request = self.__request()

            async def create() -> str:
                """
                Create the same thread from one concurrent worker.
                """

                result = (
                    await InteractionRepositoryFactory().threads().create_thread(request=request)
                )
                return result.identity.id

            results = await asyncio.gather(*(create() for _ in range(5)))

            assert set(results) == {request.identity.id}
            assert await ConversationRecord.filter(id=request.identity.id).count() == 1
            assert await EventRecord.filter(conversation_id=request.identity.id).count() == 1

    async def test_conflicting_replay_raises_thread_conflict(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            request = self.__request(title="first")
            repository = InteractionRepositoryFactory().threads()
            await repository.create_thread(request=request)
            conflict = request.model_copy(update={"title": "second"})

            with pytest.raises(ThreadConflictError):
                await repository.create_thread(request=conflict)

    async def test_create_thread_requires_existing_creator(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            request = self.__request(creator=str(uuid4()))

            with pytest.raises(InteractionError, match="Actor does not exist"):
                await InteractionRepositoryFactory().threads().create_thread(request=request)

    async def test_set_thread_title_replaces_existing_title(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            request = self.__request(title=None)
            repository = InteractionRepositoryFactory().threads()
            await repository.create_thread(request=request)

            first = await repository.set_thread_title(
                request=SetThreadTitle(
                    tenant=request.identity.tenant,
                    thread=request.identity.id,
                    title="First title",
                    metadata=Metadata(
                        entries={
                            "source": "intent",
                            "refreshed_at": (request.created + timedelta(seconds=1)).isoformat(),
                        }
                    ),
                    updated_at=request.created + timedelta(seconds=1),
                )
            )
            second = await repository.set_thread_title(
                request=SetThreadTitle(
                    tenant=request.identity.tenant,
                    thread=request.identity.id,
                    title="Second title",
                    updated_at=request.created + timedelta(seconds=2),
                )
            )

            assert first.title == "First title"
            assert first.metadata.entries["title"]["source"] == "intent"
            assert second.title == "Second title"

    async def test_archive_unarchive_and_delete_filter_public_reads(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            actor = await self.__actor()
            request = self.__request(creator=actor)
            repository = InteractionRepositoryFactory().threads()
            await repository.create_thread(request=request)
            await self.__membership(
                actor=actor,
                tenant=request.identity.tenant,
                thread=request.identity.id,
            )
            archived = await repository.transition(
                request=ThreadTransition(
                    tenant=request.identity.tenant,
                    thread=request.identity.id,
                    state=ThreadState.ARCHIVED,
                    updated_at=request.created + timedelta(seconds=1),
                    actor=actor,
                )
            )

            assert archived.state == ThreadState.ARCHIVED
            assert archived.archived == request.created + timedelta(seconds=1)
            assert (
                await repository.get_thread(
                    query=ThreadQuery(tenant=request.identity.tenant, thread=request.identity.id)
                )
                is None
            )
            archived_page = await repository.list_threads(
                query=ThreadListQuery(
                    tenant=request.identity.tenant,
                    actor=actor,
                    state=ThreadState.ARCHIVED,
                    include_archived=True,
                )
            )
            assert tuple(thread.identity.id for thread in archived_page.items) == (
                request.identity.id,
            )

            active = await repository.transition(
                request=ThreadTransition(
                    tenant=request.identity.tenant,
                    thread=request.identity.id,
                    state=ThreadState.ACTIVE,
                    updated_at=request.created + timedelta(seconds=2),
                    actor=actor,
                )
            )
            assert active.archived is None
            assert (
                await repository.get_thread(
                    query=ThreadQuery(tenant=request.identity.tenant, thread=request.identity.id)
                )
                is not None
            )

            deleted = await repository.transition(
                request=ThreadTransition(
                    tenant=request.identity.tenant,
                    thread=request.identity.id,
                    state=ThreadState.DELETED,
                    updated_at=request.created + timedelta(seconds=3),
                    actor=actor,
                )
            )
            assert deleted.deleted == request.created + timedelta(seconds=3)
            assert (
                await repository.get_thread(
                    query=ThreadQuery(tenant=request.identity.tenant, thread=request.identity.id)
                )
                is None
            )
            assert (
                await repository.list_threads(
                    query=ThreadListQuery(
                        tenant=request.identity.tenant,
                        actor=actor,
                        include_archived=True,
                    )
                )
            ).items == ()

    async def test_delete_soft_deletes_thread_children(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            request = self.__request()
            repository = InteractionRepositoryFactory().threads()
            await repository.create_thread(request=request)
            await self.__children(tenant=request.identity.tenant, thread=request.identity.id)
            deleted_at = request.created + timedelta(seconds=1)
            deleted_by = await self.__actor()

            await repository.transition(
                request=ThreadTransition(
                    tenant=request.identity.tenant,
                    thread=request.identity.id,
                    state=ThreadState.DELETED,
                    updated_at=deleted_at,
                    actor=deleted_by,
                )
            )

            execution = await ExecutionRecord.get(conversation_id=request.identity.id)
            membership = await MembershipRecord.get(conversation_id=request.identity.id)
            task = await TaskRecord.get(conversation_id=request.identity.id)
            message = await MessageRecord.get(conversation_id=request.identity.id)
            artifact = await ArtifactRecord.get(conversation_id=request.identity.id)
            script = await ScriptRecord.get(conversation_id=request.identity.id)
            context = await ContextRecord.get(conversation_id=request.identity.id)
            job = await JobRecord.get(conversation_id=request.identity.id)
            sequences = await SequenceRecord.filter(conversation_id=request.identity.id)

            for row in (
                execution,
                membership,
                task,
                message,
                artifact,
                script,
                context,
                job,
                *sequences,
            ):
                assert row.deleted_at == deleted_at
                assert row.deleted_by == deleted_by
                assert row.updated_at == deleted_at
                assert row.updated_by == deleted_by

    async def test_list_threads_filters_title_and_paginates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            repository = InteractionRepositoryFactory().threads()
            created = datetime(2026, 1, 1, tzinfo=timezone.utc)
            actor = await self.__actor()
            first = self.__request(title="Alpha one", creator=actor, created=created)
            second = self.__request(
                title="Alpha two",
                creator=actor,
                created=created + timedelta(seconds=1),
            )
            third = self.__request(
                title="Beta", creator=actor, created=created + timedelta(seconds=2)
            )
            await repository.create_thread(request=first)
            await repository.create_thread(request=second)
            await repository.create_thread(request=third)
            await self.__membership(actor=actor, tenant="tenant-a", thread=first.identity.id)
            await self.__membership(actor=actor, tenant="tenant-a", thread=second.identity.id)
            await self.__membership(actor=actor, tenant="tenant-a", thread=third.identity.id)

            page = await repository.list_threads(
                query=ThreadListQuery(tenant="tenant-a", actor=actor, title="alpha", limit=1)
            )
            next_page = await repository.list_threads(
                query=ThreadListQuery(
                    tenant="tenant-a",
                    actor=actor,
                    title="alpha",
                    limit=1,
                    cursor=page.next,
                )
            )

            assert page.total == 2
            assert tuple(thread.title for thread in page.items) == ("Alpha two",)
            assert tuple(thread.title for thread in next_page.items) == ("Alpha one",)
            assert next_page.next is None

    async def test_thread_state_is_derived_from_lifecycle_timestamps(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            repository = InteractionRepositoryFactory().threads()
            now = datetime.now(tz=timezone.utc)
            active = await ConversationRecord.create(
                id=str(uuid4()),
                tenant_id="tenant-a",
                workspace_id=None,
                title="Active",
                metadata={},
            )
            archived = await ConversationRecord.create(
                id=str(uuid4()),
                tenant_id="tenant-a",
                workspace_id=None,
                title="Archived",
                archived_at=now,
                metadata={},
            )
            deleted = await ConversationRecord.create(
                id=str(uuid4()),
                tenant_id="tenant-a",
                workspace_id=None,
                title="Deleted",
                archived_at=now,
                deleted_at=now,
                metadata={},
            )

            active_result = await repository.get_thread(
                query=ThreadQuery(tenant="tenant-a", thread=active.id)
            )
            archived_result = await repository.get_thread(
                query=ThreadQuery(
                    tenant="tenant-a",
                    thread=archived.id,
                    include_archived=True,
                )
            )

            assert active_result is not None
            assert archived_result is not None
            assert active_result.state == ThreadState.ACTIVE
            assert archived_result.state == ThreadState.ARCHIVED
            assert (
                await repository.get_thread(query=ThreadQuery(tenant="tenant-a", thread=deleted.id))
            ) is None
            visible_deleted = await repository.get_thread(
                query=ThreadQuery(
                    tenant="tenant-a",
                    thread=deleted.id,
                    include_archived=True,
                    include_deleted=True,
                )
            )
            assert visible_deleted is not None
            assert visible_deleted.state == ThreadState.DELETED

    async def test_actor_with_no_memberships_sees_no_threads(self) -> None:
        """
        A tenant actor with no memberships must not see tenant conversations.
        """

        async with InteractionPostgresSchema(prefix="conversation_thread_repository"):
            owner = await self.__actor()
            await (
                InteractionRepositoryFactory()
                .threads()
                .create_thread(request=self.__request(creator=owner))
            )

            page = (
                await InteractionRepositoryFactory()
                .threads()
                .list_threads(
                    query=ThreadListQuery(
                        tenant="tenant-a",
                        actor="fresh-user@example.com",
                        limit=100,
                    )
                )
            )

            assert page.items == ()
            assert page.total == 0
            assert page.next is None

    async def __actor(self) -> str:
        """
        Insert one actor row and return its identifier.
        """

        actor = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            kind=ActorKind.HUMAN.value,
            name="Operator",
            skills={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return actor

    async def __membership(self, *, actor: str, tenant: str, thread: str) -> None:
        """
        Insert one active membership row for list-access tests.
        """

        now = datetime.now(tz=timezone.utc)
        await MembershipRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id="workspace-a",
            conversation_id=thread,
            actor=actor,
            role=MembershipRole.OWNER.value,
            scope=MembershipScope.THREAD.value,
            joined_at=now,
            metadata={},
            created_at=now,
            updated_at=now,
        )

    async def __children(self, *, tenant: str, thread: str) -> None:
        """
        Insert one child row for each soft-deleted child table.
        """

        now = datetime.now(tz=timezone.utc)
        actor = await self.__actor()
        membership = await MembershipRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            actor=actor,
            role=MembershipRole.OWNER.value,
            scope=MembershipScope.THREAD.value,
            joined_at=now,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        execution = await ExecutionRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            intent="Do it",
            state="running",
            outcome={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await ContextRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution.id,
            task=None,
            consumer=actor,
            purpose=ContextPurpose.EXECUTION.value,
            builder="builder",
            references={"messages": [], "events": [], "artifacts": [], "memories": []},
            budget={},
            filters={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await JobRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution.id,
            task=None,
            kind=JobKind.EXECUTION.value,
            state=JobState.PENDING.value,
            attempts=0,
            available_at=now,
            payload={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await SequenceRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            scope="message",
            value=1,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await TaskRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution.id,
            assignee=actor,
            kind=TaskKind.FATHOM.value,
            objective="Do it",
            state=TaskState.QUEUED.value,
            progress=[],
            plan={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        assert membership.actor == actor
        await MessageRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution.id,
            author=actor,
            sequence=10,
            kind=MessageKind.REQUEST.value,
            audience=[Audience.THREAD.value],
            body={"text": "hello"},
            labels=[],
            metadata={},
            created_at=now,
        )
        await ArtifactRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution.id,
            kind=ArtifactKind.SCREENSHOT.value,
            uri="s3://bucket/key",
            backend=ArtifactBackend.OBJECT.value,
            retention={},
            labels=[],
            metadata={},
            created_at=now,
        )
        await ScriptRecord.create(
            id=str(uuid4()),
            tenant_id=tenant,
            workspace_id=None,
            conversation_id=thread,
            execution_id=execution.id,
            format=ScriptFormat.TEXT_PLAIN.value,
            status=ScriptStatus.ACTIVE.value,
            content="tap",
            revision=1,
            checksum="checksum",
            metadata={},
            created_at=now,
            updated_at=now,
        )

    def __request(
        self,
        *,
        title: Optional[str] = "Thread",
        creator: Optional[str] = None,
        created: Optional[datetime] = None,
    ) -> CreateThread:
        """
        Build one thread creation request.
        """

        return CreateThread(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace="workspace-a"),
            title=title,
            state=ThreadState.ACTIVE,
            creator=creator,
            created_at=created or datetime.now(tz=timezone.utc),
            metadata=Metadata(entries={"source": "test"}),
        )
