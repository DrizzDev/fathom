from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import (
    ActorKind,
    Audience,
    EventKind,
    Label,
    MembershipRole,
    MembershipScope,
    MessageKind,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    MembershipRecord,
    MessageRecord,
    TaskRecord,
)
from fathom.schemas.interaction import (
    Content,
    Identity,
    MessageCursorQuery,
    MessageQuery,
    Metadata,
    RecordMessage,
    Sanitize,
    SortOrder,
)


class TestMessageRepository:
    """
    Verify message persistence through the persistent-store backed repository.
    """

    async def test_record_message_persists_message_and_records_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            request = self.__record_request(actor=actor, thread=thread)

            result = await InteractionRepositoryFactory().messages().record_message(request=request)

            assert result.identity == request.identity
            assert result.thread == thread
            assert result.author == actor
            assert result.sequence == 1
            assert result.kind == MessageKind.REQUEST
            assert result.audience == Audience.THREAD
            assert result.content.labels == (Label.DISPLAY_AUDIT, Label.MEMORY_SKIP)
            row = await MessageRecord.get(id=result.identity.id)
            assert row.created_by == actor
            assert row.updated_by == actor
            event = await EventRecord.get(conversation_id=thread, sequence=1)
            assert event.kind == EventKind.MESSAGE_RECORDED.value
            assert event.source == "interaction"
            assert event.created_by == actor
            assert event.execution_id == request.execution

    async def test_identical_record_replay_treats_labels_as_set(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            request = self.__record_request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().messages()

            created = await repository.record_message(request=request)
            replay = request.model_copy(
                update={
                    "content": request.content.model_copy(
                        update={"labels": tuple(reversed(request.content.labels))}
                    )
                }
            )
            replayed = await repository.record_message(request=replay)

            assert replayed == created
            assert await MessageRecord.filter(conversation_id=thread).count() == 1
            assert await EventRecord.filter(conversation_id=thread).count() == 1

    async def test_conflicting_record_replay_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            request = self.__record_request(actor=actor, thread=thread)
            repository = InteractionRepositoryFactory().messages()
            await repository.record_message(request=request)
            conflict = request.model_copy(update={"content": Content(body={"text": "changed"})})

            with pytest.raises(InteractionError, match="different content"):
                await repository.record_message(request=conflict)

    async def test_record_message_rejects_terminal_task(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(actor=actor, thread=thread, state=TaskState.SUCCEEDED)

            with pytest.raises(InteractionError, match="terminal task state"):
                await (
                    InteractionRepositoryFactory()
                    .messages()
                    .record_message(
                        request=self.__record_request(actor=actor, thread=thread, task=task)
                    )
                )

    async def test_record_message_stores_explicit_task_execution(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(actor=actor, thread=thread, state=TaskState.RUNNING)
            request = self.__record_request(
                actor=actor,
                thread=thread,
                task=task,
                execution=task,
            )

            result = await InteractionRepositoryFactory().messages().record_message(request=request)

            row = await MessageRecord.get(id=result.identity.id, tenant_id="tenant-a")
            assert row.execution_id == task
            event = await EventRecord.get(conversation_id=thread, sequence=1)
            assert event.execution_id == task

    async def test_record_message_rejects_task_execution_mismatch(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(actor=actor, thread=thread, state=TaskState.RUNNING)
            request = self.__record_request(
                actor=actor,
                thread=thread,
                task=task,
                execution=str(uuid4()),
            )

            with pytest.raises(InteractionError, match="execution does not match"):
                await InteractionRepositoryFactory().messages().record_message(request=request)

    async def test_record_message_requires_execution_without_task(self) -> None:
        """
        Reject run-owned messages that have neither task nor execution.
        """

        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            request = self.__record_request(actor=actor, thread=thread).model_copy(
                update={"execution": None}
            )

            with pytest.raises(InteractionError, match="Message execution is required"):
                await InteractionRepositoryFactory().messages().record_message(request=request)

    async def test_sanitize_message_updates_content_and_records_policy_event(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().messages()
            recorded = await repository.record_message(
                request=self.__record_request(actor=actor, thread=thread)
            )
            sanitized_at = datetime.now(tz=timezone.utc) + timedelta(seconds=1)
            request = Sanitize(
                tenant="tenant-a",
                message=recorded.identity.id,
                content=Content(
                    body={"text": "[redacted]"},
                    labels=(Label.PRIVACY_EMAIL,),
                    sanitizer="redactor@1",
                ),
                sanitized_at=sanitized_at,
            )

            sanitized = await repository.sanitize_message(request=request)
            replayed = await repository.sanitize_message(request=request)

            assert replayed == sanitized
            assert sanitized.content.body == {"text": "[redacted]"}
            assert sanitized.content.sanitized == sanitized_at
            assert sanitized.content.sanitizer == "redactor@1"
            row = await MessageRecord.get(id=recorded.identity.id)
            assert row.updated_by == actor
            event = await EventRecord.get(conversation_id=thread, sequence=2)
            assert event.kind == EventKind.CONTENT_SANITIZED.value
            assert event.source == "policy"
            assert event.created_by == actor
            assert await EventRecord.filter(conversation_id=thread).count() == 2

    async def test_get_messages_filters_task_orders_and_hides_deleted_rows(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            task = await self.__task(actor=actor, thread=thread, state=TaskState.RUNNING)
            repository = InteractionRepositoryFactory().messages()
            first = await repository.record_message(
                request=self.__record_request(actor=actor, thread=thread, task=task)
            )
            second = await repository.record_message(
                request=self.__record_request(actor=actor, thread=thread, task=task)
            )
            await MessageRecord.filter(id=second.identity.id).update(
                deleted_at=datetime.now(tz=timezone.utc)
            )

            messages = await repository.get_messages(
                query=MessageQuery(tenant="tenant-a", thread=thread, task=task)
            )

            assert tuple(message.identity.id for message in messages) == (first.identity.id,)

    async def test_list_messages_filters_and_paginates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            repository = InteractionRepositoryFactory().messages()
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            first = await repository.record_message(
                request=self.__record_request(
                    actor=actor,
                    thread=thread,
                    kind=MessageKind.NOTE,
                    created=base,
                )
            )
            second = await repository.record_message(
                request=self.__record_request(
                    actor=actor,
                    thread=thread,
                    kind=MessageKind.NOTE,
                    created=base + timedelta(seconds=1),
                )
            )
            await repository.record_message(
                request=self.__record_request(
                    actor=actor,
                    thread=thread,
                    kind=MessageKind.RESULT,
                    created=base + timedelta(seconds=2),
                )
            )

            page = await repository.list_messages(
                query=MessageCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    kinds=(MessageKind.NOTE,),
                    order=SortOrder.ASC,
                    limit=1,
                )
            )
            next_page = await repository.list_messages(
                query=MessageCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    kinds=(MessageKind.NOTE,),
                    order=SortOrder.ASC,
                    limit=1,
                    cursor=page.next,
                )
            )

            assert tuple(message.identity.id for message in page.items) == (first.identity.id,)
            assert tuple(message.identity.id for message in next_page.items) == (
                second.identity.id,
            )
            assert page.total == 2

    async def test_corrupt_message_row_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_message_repository"):
            actor, thread = await self.__conversation()
            request = self.__record_request(actor=actor, thread=thread, sequence=44)
            with pytest.raises(IntegrityError):
                await MessageRecord.create(
                    id=request.identity.id,
                    tenant_id=request.identity.tenant,
                    workspace_id=request.identity.workspace,
                    conversation_id=thread,
                    execution_id=request.execution,
                    task=None,
                    author=actor,
                    reply=None,
                    sequence=44,
                    kind="alien",
                    audience=[Audience.THREAD.value],
                    body=request.content.body,
                    labels=["unknown.label"],
                    sanitized_at=None,
                    sanitizer=None,
                    created_at=request.created,
                    deleted_at=None,
                    metadata=request.metadata.entries,
                )

    async def __conversation(self) -> Tuple[str, str]:
        """
        Persist one actor, active thread, and membership.
        """

        actor = await self.__actor()
        thread = await self.__thread(actor=actor)
        await self.__membership(actor=actor, thread=thread)
        self.__executions()[thread] = await self.__execution(actor=actor, thread=thread)
        return actor, thread

    def __executions(self) -> Dict[str, str]:
        """
        Return execution identifiers created for fixture conversations.
        """

        store = getattr(self, "__execution_by_thread", None)
        if store is None:
            store = {}
            setattr(self, "__execution_by_thread", store)

        return store

    async def __actor(self) -> str:
        """
        Persist one actor row and return its id.
        """

        actor = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            kind=ActorKind.HUMAN.value,
            name="operator",
            external=None,
            runtime=None,
            provider=None,
            model=None,
            skills={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return actor

    async def __thread(self, *, actor: str) -> str:
        """
        Persist one active thread row and return its id.
        """

        thread = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ConversationRecord.create(
            id=thread,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            title="Plan",
            digest=None,
            created_by=actor,
            archived_at=None,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return thread

    async def __membership(self, *, actor: str, thread: str) -> None:
        """
        Persist one active membership row.
        """

        await MembershipRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            actor=actor,
            role=MembershipRole.OWNER.value,
            scope=MembershipScope.THREAD.value,
            joined_at=datetime.now(tz=timezone.utc),
            departed_at=None,
            metadata={},
        )

    async def __execution(self, *, actor: str, thread: str) -> str:
        """
        Persist one execution row and return its id.
        """

        execution = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=execution,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            intent="Do work",
            state=TaskState.RUNNING.value,
            outcome={},
            started_at=now,
            completed_at=None,
            created_at=now,
            created_by=actor,
            updated_at=now,
            updated_by=actor,
            metadata={},
        )
        return execution

    async def __task(self, *, actor: str, thread: str, state: TaskState) -> str:
        """
        Persist one task row and return its id.
        """

        task = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=task,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            intent="Do work",
            state="running",
            outcome={},
            started_at=now if state == TaskState.RUNNING else None,
            completed_at=now if state == TaskState.SUCCEEDED else None,
            created_at=now,
            created_by=actor,
            updated_at=now,
            updated_by=actor,
            metadata={},
        )
        await TaskRecord.create(
            id=task,
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            conversation_id=thread,
            execution_id=task,
            created_by=actor,
            assignee=actor,
            parent=None,
            origin=None,
            kind=TaskKind.AGENT.value,
            objective="Do work",
            reference=None,
            state=state.value,
            code=None,
            detail=None,
            progress={},
            plan={},
            outcome={},
            summary=None,
            started_at=now if state == TaskState.RUNNING else None,
            completed_at=now if state == TaskState.SUCCEEDED else None,
            elapsed=None,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        return task

    def __record_request(
        self,
        *,
        actor: str,
        thread: str,
        task: Optional[str] = None,
        execution: Optional[str] = None,
        kind: MessageKind = MessageKind.REQUEST,
        sequence: Optional[int] = None,
        created: Optional[datetime] = None,
    ) -> RecordMessage:
        """
        Build one message record request.
        """

        resolved_execution = execution
        if task is None and resolved_execution is None:
            resolved_execution = self.__executions()[thread]

        return RecordMessage(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace="workspace-a"),
            thread=thread,
            task=task,
            execution=resolved_execution,
            author=actor,
            reply=None,
            sequence=sequence,
            kind=kind,
            audience=Audience.THREAD,
            content=Content(
                body={"text": "hello"},
                labels=(Label.DISPLAY_AUDIT, Label.MEMORY_SKIP),
            ),
            created_at=created or datetime.now(tz=timezone.utc),
            metadata=Metadata(entries={"source": "test"}),
        )
