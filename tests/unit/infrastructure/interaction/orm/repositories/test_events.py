from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple
from uuid import uuid4

import pytest
from asyncpg.exceptions import RaiseError
from pydantic import JsonValue
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise import connections

from fathom.constants.collaboration import (
    EVENT_SOURCE_ACTORS,
    ActorKind,
    EventKind,
    EventSource,
    TaskKind,
    TaskState,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    EventRecord,
    ExecutionRecord,
    TaskRecord,
)
from fathom.infrastructure.interaction.orm.raw import InteractionSqlFiles, RawSql
from fathom.infrastructure.interaction.orm.repositories import EventRepository
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    SequenceAllocator,
    UuidIdentifierSource,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.interaction.digest import EventDigest
from fathom.schemas.interaction import EventCursorQuery, EventQuery, Metadata, SortOrder


class _FixedSequenceAllocator(SequenceAllocator):
    """
    Supplies explicit event sequence values for race regression tests.
    """

    def __init__(self, *, values: Sequence[int]) -> None:
        """
        Store deterministic sequence values.
        """

        self.__values = tuple(values)
        self.__index = 0

    async def next(
        self, *, connection: DatabaseConnection, tenant: str, thread: str, scope: str
    ) -> int:
        """
        Return the next configured sequence.
        """

        value = self.__values[self.__index]
        self.__index += 1
        return value


class TestEventRepository:
    """
    Verify lifecycle event reads through the persistent-store backed repository.
    """

    async def test_get_events_orders_by_sequence_and_filters_task(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, thread = await self.__conversation()
            first_task = await self.__task(thread=thread, actor=actor)
            second_task = await self.__task(thread=thread, actor=actor)
            await self.__event(
                thread=thread,
                actor=actor,
                task=second_task,
                sequence=2,
                kind=EventKind.TASK_OPENED,
            )
            first = await self.__event(
                thread=thread,
                actor=actor,
                task=first_task,
                sequence=1,
                kind=EventKind.MESSAGE_RECORDED,
            )

            events = await EventRepository(references=ReferenceGuard()).get_events(
                query=EventQuery(tenant="tenant-a", thread=thread, task=first_task)
            )

            assert tuple(event.identity.id for event in events) == (first,)
            assert events[0].kind == EventKind.MESSAGE_RECORDED
            assert events[0].source == EventSource.INTERACTION

    async def test_list_events_filters_and_paginates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, thread = await self.__conversation()
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            first = await self.__event(
                thread=thread,
                actor=actor,
                sequence=1,
                kind=EventKind.TASK_OPENED,
                created=base,
            )
            second = await self.__event(
                thread=thread,
                actor=actor,
                sequence=2,
                kind=EventKind.TASK_OPENED,
                created=base + timedelta(seconds=1),
            )
            await self.__event(
                thread=thread,
                actor=actor,
                sequence=3,
                kind=EventKind.MESSAGE_RECORDED,
                created=base + timedelta(seconds=2),
            )

            page = await EventRepository(references=ReferenceGuard()).list_events(
                query=EventCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    actor=actor,
                    kinds=(EventKind.TASK_OPENED,),
                    since=base,
                    until=base + timedelta(seconds=2),
                    order=SortOrder.ASC,
                    limit=1,
                )
            )
            next_page = await EventRepository(references=ReferenceGuard()).list_events(
                query=EventCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    actor=actor,
                    kinds=(EventKind.TASK_OPENED,),
                    since=base,
                    until=base + timedelta(seconds=2),
                    order=SortOrder.ASC,
                    limit=1,
                    cursor=page.next,
                )
            )

            assert tuple(event.identity.id for event in page.items) == (first,)
            assert tuple(event.identity.id for event in next_page.items) == (second,)
            assert page.total == 2
            assert next_page.next is None

    async def test_list_events_hides_archived_and_deleted_threads(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, archived_thread = await self.__conversation()
            _, deleted_thread = await self.__conversation()
            await self.__event(thread=archived_thread, actor=actor, sequence=1)
            await self.__event(thread=deleted_thread, actor=actor, sequence=1)
            now = datetime.now(tz=timezone.utc)
            await ConversationRecord.filter(id=archived_thread).update(archived_at=now)
            await ConversationRecord.filter(id=deleted_thread).update(deleted_at=now)
            repository = EventRepository(references=ReferenceGuard())

            archived_page = await repository.list_events(
                query=EventCursorQuery(tenant="tenant-a", thread=archived_thread)
            )
            deleted_events = await repository.get_events(
                query=EventQuery(tenant="tenant-a", thread=deleted_thread)
            )

            assert archived_page.items == ()
            assert archived_page.total == 0
            assert deleted_events == []

    async def test_late_event_sequence_does_not_overwrite_newer_digest(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, thread = await self.__conversation()
            connection = connections.get("default")
            recorder = LifecycleRecorder(
                raw=RawSql(root=InteractionSqlFiles.bundled().root),
                sequence_allocator=_FixedSequenceAllocator(values=(10, 9)),
                event_digest=EventDigest(),
                identifier_source=UuidIdentifierSource(),
            )

            await recorder.record(
                connection=connection,
                tenant="tenant-a",
                workspace=None,
                thread=thread,
                actor=actor,
                kind=EventKind.THREAD_CREATED,
                payload=Metadata(entries={"value": "newer"}),
                created=datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
            )
            newer = await ConversationRecord.get(id=thread)

            await recorder.record(
                connection=connection,
                tenant="tenant-a",
                workspace=None,
                thread=thread,
                actor=actor,
                kind=EventKind.THREAD_ARCHIVED,
                payload=Metadata(entries={"value": "older"}),
                created=datetime(2026, 1, 1, 0, 0, 9, tzinfo=timezone.utc),
            )
            stored = await ConversationRecord.get(id=thread)

            assert stored.digest == newer.digest

    async def test_count_total_can_be_skipped(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, thread = await self.__conversation()
            await self.__event(thread=thread, actor=actor, sequence=1)

            page = await EventRepository(references=ReferenceGuard()).list_events(
                query=EventCursorQuery(
                    tenant="tenant-a",
                    thread=thread,
                    count_total=False,
                )
            )

            assert len(page.items) == 1
            assert page.total == 0

    async def test_event_rows_reject_updates(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, thread = await self.__conversation()
            await self.__event(
                thread=thread,
                actor=actor,
                sequence=1,
                kind=EventKind.MESSAGE_RECORDED,
            )
            with pytest.raises(RaiseError, match="append-only table events"):
                await EventRecord.filter(conversation_id=thread).update(kind="unknown.event")

    async def test_invalid_payload_shape_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            actor, thread = await self.__conversation()
            await self.__event(thread=thread, actor=actor, sequence=1, payload=[])

            with pytest.raises(InteractionError, match="Invalid event payload"):
                await EventRepository(references=ReferenceGuard()).get_events(
                    query=EventQuery(tenant="tenant-a", thread=thread)
                )

    async def test_null_actor_reads_as_source_actor(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            _, thread = await self.__conversation()
            await self.__event(
                thread=thread,
                actor=None,
                source=EventSource.INTERACTION,
                sequence=1,
            )

            events = await EventRepository(references=ReferenceGuard()).get_events(
                query=EventQuery(tenant="tenant-a", thread=thread)
            )

            assert events[0].actor == EVENT_SOURCE_ACTORS[EventSource.INTERACTION]

    async def test_lifecycle_recorder_projects_source_actor_when_actor_missing(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_event_repository"):
            _, thread = await self.__conversation()
            connection = connections.get("default")
            recorder = LifecycleRecorder(
                raw=RawSql(root=InteractionSqlFiles.bundled().root),
                sequence_allocator=_FixedSequenceAllocator(values=(1,)),
                event_digest=EventDigest(),
                identifier_source=UuidIdentifierSource(),
            )

            await recorder.record(
                connection=connection,
                tenant="tenant-a",
                workspace=None,
                thread=thread,
                kind=EventKind.THREAD_CREATED,
                payload=Metadata(entries={}),
                created=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            events = await EventRepository(references=ReferenceGuard()).get_events(
                query=EventQuery(tenant="tenant-a", thread=thread)
            )

            assert events[0].actor == EVENT_SOURCE_ACTORS[EventSource.INTERACTION]

    async def __conversation(self) -> Tuple[str, str]:
        """
        Insert one active actor and thread pair.
        """

        now = datetime.now(tz=timezone.utc)
        actor = str(uuid4())
        thread = str(uuid4())
        await ActorRecord.create(
            id=actor,
            tenant_id="tenant-a",
            workspace_id=None,
            kind=ActorKind.HUMAN.value,
            name="Operator",
            skills={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        await ConversationRecord.create(
            id=thread,
            tenant_id="tenant-a",
            workspace_id=None,
            created_by=actor,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return actor, thread

    async def __event(
        self,
        *,
        thread: str,
        actor: Optional[str],
        sequence: int,
        kind: EventKind = EventKind.THREAD_CREATED,
        task: Optional[str] = None,
        created: Optional[datetime] = None,
        payload: Optional[JsonValue] = None,
        source: EventSource = EventSource.INTERACTION,
    ) -> str:
        """
        Insert one event row.
        """

        identifier = str(uuid4())
        await EventRecord.create(
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            task_id=task,
            actor=actor,
            sequence=sequence,
            kind=kind.value,
            source=source.value,
            payload={} if payload is None else payload,
            metadata={},
            created_at=created or datetime.now(tz=timezone.utc),
        )
        return identifier

    async def __task(self, *, thread: str, actor: str) -> str:
        """
        Insert one task row for task-scoped events.
        """

        identifier = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ExecutionRecord.create(
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            intent="Do it",
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
            id=identifier,
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=thread,
            execution_id=identifier,
            created_by=actor,
            assignee=actor,
            kind=TaskKind.FATHOM.value,
            objective="Do it",
            state=TaskState.RUNNING.value,
            progress={},
            plan={},
            outcome={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return identifier
