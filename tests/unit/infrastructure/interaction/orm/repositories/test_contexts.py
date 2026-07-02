from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    EventKind,
    EventSource,
    MembershipRole,
    MembershipScope,
    MessageKind,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ArtifactRecord,
    ContextRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    MembershipRecord,
    MessageRecord,
    SequenceRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    BuildContext,
    ContextCursorQuery,
    ContextQuery,
    Identity,
    MemoryReference,
    Metadata,
    References,
    SortOrder,
)


class TestContextRepository:
    """
    Verify context recipe persistence through the persistent-store backed repository.
    """

    async def test_build_context_persists_recipe_and_records_event(self) -> None:
        """
        Build one context recipe with local and external references.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture)

            result = await InteractionRepositoryFactory().contexts().build_context(request=request)

            assert result.identity == request.identity
            assert result.thread == fixture.thread
            assert result.task == fixture.task
            assert result.execution == fixture.task
            assert result.consumer == fixture.actor
            assert result.references == request.references
            assert result.budget == Metadata(entries={"tokens": 1200})
            assert result.filters == Metadata(entries={"labels": ["public"]})
            assert result.hash == "hash-a"
            assert result.provider == "provider-a"
            assert result.model == "model-a"
            assert result.expires == request.expires
            assert result.metadata == Metadata(entries={"source": "test"})
            row = await ContextRecord.get(id=result.identity.id)
            assert row.created_by == fixture.actor
            assert row.updated_by == fixture.actor
            event = await EventRecord.get(conversation_id=fixture.thread, sequence=2)
            assert event.kind == EventKind.CONTEXT_BUILT.value
            assert event.actor == fixture.actor
            assert event.created_by == fixture.actor
            assert event.execution_id == fixture.task

    async def test_identical_replay_returns_existing_without_new_event(self) -> None:
        """
        Replay an identical build request without duplicating rows.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture)
            repository = InteractionRepositoryFactory().contexts()

            created = await repository.build_context(request=request)
            replayed = await repository.build_context(request=request)

            assert replayed == created
            assert await ContextRecord.filter(conversation_id=fixture.thread).count() == 1
            assert await EventRecord.filter(conversation_id=fixture.thread).count() == 2

    async def test_list_contexts_filters_by_execution(self) -> None:
        """
        Return only contexts attached to the requested execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            first = await self.__conversation()
            second = await self.__conversation()
            repository = InteractionRepositoryFactory().contexts()
            await repository.build_context(request=self.__request(fixture=first))
            await repository.build_context(request=self.__request(fixture=second))

            page = await repository.list_contexts(
                query=ContextCursorQuery(
                    tenant="tenant-a",
                    thread=first.thread,
                    execution=first.task,
                )
            )

            assert len(page.items) == 1
            assert page.items[0].execution == first.task

    async def test_context_execution_must_match_task_execution(self) -> None:
        """
        Reject a context whose explicit execution disagrees with its task.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture).model_copy(update={"execution": str(uuid4())})

            with pytest.raises(InteractionError, match="execution does not match"):
                await InteractionRepositoryFactory().contexts().build_context(request=request)

    async def test_build_context_requires_execution_without_task(self) -> None:
        """
        Reject run-owned contexts that have neither task nor execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture).model_copy(
                update={"task": None, "execution": None}
            )

            with pytest.raises(InteractionError, match="Context execution is required"):
                await InteractionRepositoryFactory().contexts().build_context(request=request)

    async def test_conflicting_replay_raises_interaction_error(self) -> None:
        """
        Reject reused context ids with different context content.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture)
            repository = InteractionRepositoryFactory().contexts()
            await repository.build_context(request=request)
            conflict = request.model_copy(update={"builder": "different@1"})

            with pytest.raises(InteractionError, match="different content"):
                await repository.build_context(request=conflict)

    async def test_deleted_context_is_not_replayed_as_active(self) -> None:
        """
        Reject replay against a soft-deleted context row.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture)
            repository = InteractionRepositoryFactory().contexts()
            created = await repository.build_context(request=request)
            await ContextRecord.filter(id=created.identity.id).update(deleted_at=self.__now())

            with pytest.raises(InteractionError, match="insert conflicted"):
                await repository.build_context(request=request)

    async def test_identical_replay_returns_existing_after_parent_archived(self) -> None:
        """
        Replay an existing context before validating the archived parent thread.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            request = self.__request(fixture=fixture)
            repository = InteractionRepositoryFactory().contexts()
            created = await repository.build_context(request=request)
            await ConversationRecord.filter(id=fixture.thread).update(
                archived_at=self.__now(),
            )

            replayed = await repository.build_context(request=request)

            assert replayed == created
            assert await ContextRecord.filter(conversation_id=fixture.thread).count() == 1
            assert await EventRecord.filter(conversation_id=fixture.thread).count() == 2

    async def test_build_context_validates_local_references(self) -> None:
        """
        Reject references that are missing or belong to another thread.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            other = await self.__conversation()
            missing_message = self.__request(
                fixture=fixture,
                references=References(messages=(str(uuid4()),)),
            )
            wrong_thread_event = self.__request(
                fixture=fixture,
                references=References(events=(other.event,)),
            )

            with pytest.raises(InteractionError, match="Message does not exist"):
                await (
                    InteractionRepositoryFactory().contexts().build_context(request=missing_message)
                )
            with pytest.raises(InteractionError, match="Event belongs to a different thread"):
                await (
                    InteractionRepositoryFactory()
                    .contexts()
                    .build_context(request=wrong_thread_event)
                )

    async def test_build_context_requires_active_consumer_membership(self) -> None:
        """
        Require a context consumer to be an active thread member.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation(active_membership=False)

            with pytest.raises(InteractionError, match="active member"):
                await (
                    InteractionRepositoryFactory()
                    .contexts()
                    .build_context(request=self.__request(fixture=fixture))
                )

    async def test_get_contexts_filters_and_orders(self) -> None:
        """
        Load contexts by task and purpose in creation order.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            first = (
                await InteractionRepositoryFactory()
                .contexts()
                .build_context(
                    request=self.__request(
                        fixture=fixture,
                        created=self.__now(),
                        purpose=ContextPurpose.EXECUTION,
                    )
                )
            )
            second = (
                await InteractionRepositoryFactory()
                .contexts()
                .build_context(
                    request=self.__request(
                        fixture=fixture,
                        created=self.__now() + timedelta(seconds=1),
                        purpose=ContextPurpose.DIGEST,
                    )
                )
            )

            contexts = (
                await InteractionRepositoryFactory()
                .contexts()
                .get_contexts(
                    query=ContextQuery(
                        tenant="tenant-a",
                        thread=fixture.thread,
                        task=fixture.task,
                        purpose=ContextPurpose.EXECUTION,
                    )
                )
            )
            all_contexts = (
                await InteractionRepositoryFactory()
                .contexts()
                .get_contexts(query=ContextQuery(tenant="tenant-a", thread=fixture.thread))
            )

            assert tuple(context.identity.id for context in contexts) == (first.identity.id,)
            assert tuple(context.identity.id for context in all_contexts) == (
                first.identity.id,
                second.identity.id,
            )

    async def test_list_contexts_filters_paginates_and_skips_total(self) -> None:
        """
        Page contexts with cursor, consumer, time, and purpose filters.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            first = (
                await InteractionRepositoryFactory()
                .contexts()
                .build_context(request=self.__request(fixture=fixture, created=self.__now()))
            )
            second = (
                await InteractionRepositoryFactory()
                .contexts()
                .build_context(
                    request=self.__request(
                        fixture=fixture,
                        created=self.__now() + timedelta(seconds=1),
                    )
                )
            )

            page = (
                await InteractionRepositoryFactory()
                .contexts()
                .list_contexts(
                    query=ContextCursorQuery(
                        tenant="tenant-a",
                        thread=fixture.thread,
                        consumer=fixture.actor,
                        purpose=ContextPurpose.EXECUTION,
                        limit=1,
                        order=SortOrder.ASC,
                        count_total=False,
                    )
                )
            )
            next_page = (
                await InteractionRepositoryFactory()
                .contexts()
                .list_contexts(
                    query=ContextCursorQuery(
                        tenant="tenant-a",
                        thread=fixture.thread,
                        cursor=page.next,
                        limit=1,
                        order=SortOrder.ASC,
                    )
                )
            )

            assert tuple(context.identity.id for context in page.items) == (first.identity.id,)
            assert page.total == 0
            assert page.next is not None
            assert tuple(context.identity.id for context in next_page.items) == (
                second.identity.id,
            )

    async def test_archived_and_deleted_threads_hide_contexts(self) -> None:
        """
        Hide context rows after their parent thread is archived or deleted.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            await (
                InteractionRepositoryFactory()
                .contexts()
                .build_context(request=self.__request(fixture=fixture))
            )
            await ConversationRecord.filter(id=fixture.thread).update(
                archived_at=self.__now(),
            )

            contexts = (
                await InteractionRepositoryFactory()
                .contexts()
                .get_contexts(query=ContextQuery(tenant="tenant-a", thread=fixture.thread))
            )
            page = (
                await InteractionRepositoryFactory()
                .contexts()
                .list_contexts(query=ContextCursorQuery(tenant="tenant-a", thread=fixture.thread))
            )

            assert contexts == []
            assert page.items == ()
            assert page.total == 0

    async def test_corrupt_context_row_raises_interaction_error(self) -> None:
        """
        Reject stored rows with unknown purposes or invalid references.
        """

        async with InteractionPostgresSchema(prefix="conversation_context_repository"):
            fixture = await self.__conversation()
            with pytest.raises(IntegrityError):
                await ContextRecord.create(
                    id=str(uuid4()),
                    tenant_id="tenant-a",
                    workspace_id=None,
                    conversation_id=fixture.thread,
                    execution_id=fixture.task,
                    task=None,
                    consumer=None,
                    purpose="alien",
                    builder="builder@1",
                    references={"messages": [], "events": [], "artifacts": [], "memories": []},
                    budget={},
                    filters={},
                    hash=None,
                    provider=None,
                    model=None,
                    created_at=self.__now(),
                    expires_at=None,
                    metadata={},
                )

    async def __conversation(
        self,
        *,
        active_membership: bool = True,
    ) -> "_ContextFixture":
        """
        Insert the rows needed by context repository tests.
        """

        actor = str(uuid4())
        thread = str(uuid4())
        task = str(uuid4())
        message = str(uuid4())
        event = str(uuid4())
        artifact = str(uuid4())
        now = self.__now()

        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id=None,
            kind=ActorKind.HUMAN.value,
            name="Operator",
            external=None,
            runtime=None,
            provider=None,
            model=None,
            skills={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await ConversationRecord.create(
            id=thread,
            tenant_id="tenant-a",
            workspace_id=None,
            title="Thread",
            digest=None,
            created_by=actor,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        await MembershipRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            actor=actor,
            role=MembershipRole.OWNER.value,
            scope=MembershipScope.THREAD.value,
            joined_at=now,
            departed_at=None if active_membership else now,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await ExecutionRecord.create(
            id=task,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do work",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            created_at=now,
            created_by=actor,
            updated_at=now,
            updated_by=actor,
            metadata={},
        )
        await TaskRecord.create(
            id=task,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=task,
            created_by=actor,
            assignee=actor,
            origin=None,
            kind=TaskKind.AGENT.value,
            objective="Do work",
            reference=None,
            state=TaskState.RUNNING.value,
            code=None,
            detail=None,
            progress={},
            plan={},
            outcome={},
            summary=None,
            started_at=now,
            completed_at=None,
            elapsed=None,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        await MessageRecord.create(
            id=message,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=task,
            task=task,
            author=actor,
            reply=None,
            sequence=1,
            kind=MessageKind.REQUEST.value,
            audience=[Audience.THREAD.value],
            body={"text": "request"},
            labels=[],
            metadata={},
            created_at=now,
        )
        await EventRecord.create(
            id=event,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=task,
            task=task,
            actor=actor,
            sequence=1,
            kind=EventKind.TASK_STARTED.value,
            source=EventSource.INTERACTION.value,
            payload={},
            metadata={},
            created_at=now,
        )
        await SequenceRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            conversation_id=thread,
            scope="event",
            value=1,
        )
        await ArtifactRecord.create(
            id=artifact,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=task,
            task=task,
            producer=actor,
            kind=ArtifactKind.SCREENSHOT.value,
            uri="memory://artifact",
            backend=ArtifactBackend.LOCAL.value,
            mime="image/png",
            size=10,
            retention=None,
            labels=[],
            metadata={},
            created_at=now,
        )
        return _ContextFixture(
            actor=actor,
            thread=thread,
            task=task,
            message=message,
            event=event,
            artifact=artifact,
        )

    def __request(
        self,
        *,
        fixture: "_ContextFixture",
        references: Optional[References] = None,
        created: Optional[datetime] = None,
        purpose: ContextPurpose = ContextPurpose.EXECUTION,
    ) -> BuildContext:
        """
        Build one context request with a plain UUID identity.
        """

        return BuildContext(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace=None),
            thread=fixture.thread,
            task=fixture.task,
            consumer=fixture.actor,
            purpose=purpose,
            builder="builder@1",
            references=references
            or References(
                messages=(fixture.message,),
                events=(fixture.event,),
                artifacts=(fixture.artifact,),
                memories=(MemoryReference(system="semantic", reference="memory-1"),),
            ),
            budget=Metadata(entries={"tokens": 1200}),
            filters=Metadata(entries={"labels": ["public"]}),
            hash="hash-a",
            provider="provider-a",
            model="model-a",
            created_at=created or self.__now(),
            expires_at=(created or self.__now()) + timedelta(hours=1),
            metadata=Metadata(entries={"source": "test"}),
        )

    def __now(self) -> datetime:
        """
        Return a stable timezone-aware timestamp for tests.
        """

        return datetime(2026, 1, 1, tzinfo=timezone.utc)


class _ContextFixture:
    """
    Identifiers for rows used by context repository tests.
    """

    def __init__(
        self,
        *,
        actor: str,
        thread: str,
        task: str,
        message: str,
        event: str,
        artifact: str,
    ) -> None:
        """
        Capture fixture row identifiers.
        """

        self.actor = actor
        self.thread = thread
        self.task = task
        self.message = message
        self.event = event
        self.artifact = artifact
