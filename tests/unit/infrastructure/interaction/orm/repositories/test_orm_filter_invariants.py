from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema

from fathom.constants.collaboration import EventKind, EventSource
from fathom.infrastructure.interaction.orm.models import ConversationRecord, EventRecord


class TestOrmFilterInvariants:
    """
    Verify direct ORM filters never widen empty explicit filter sets.
    """

    async def test_empty_identifier_filter_returns_no_rows(self) -> None:
        """
        Empty id__in filter must not return every tenant row.
        """

        async with InteractionPostgresSchema(prefix="orm_filter_invariants"):
            await self.__conversation()

            total = await ConversationRecord.filter(
                tenant_id="tenant-a",
                id__in=(),
            ).count()

            assert total == 0

    async def test_empty_enum_filter_returns_no_rows(self) -> None:
        """
        Empty kind__in filter must not return every tenant row.
        """

        async with InteractionPostgresSchema(prefix="orm_filter_invariants"):
            conversation = await self.__conversation()
            await self.__event(conversation=conversation)

            total = await EventRecord.filter(
                tenant_id="tenant-a",
                kind__in=(),
            ).count()

            assert total == 0

    async def __conversation(self) -> str:
        """
        Persist one conversation row and return its id.
        """

        conversation = str(uuid4())
        now = datetime.now(tz=timezone.utc)
        await ConversationRecord.create(
            id=conversation,
            tenant_id="tenant-a",
            workspace_id=None,
            title="Visible",
            metadata={},
            created_at=now,
            updated_at=now,
        )
        return conversation

    async def __event(self, *, conversation: str) -> None:
        """
        Persist one event row for enum filter tests.
        """

        now = datetime.now(tz=timezone.utc)
        await EventRecord.create(
            id=str(uuid4()),
            tenant_id="tenant-a",
            workspace_id=None,
            conversation_id=conversation,
            actor=None,
            sequence=1,
            kind=EventKind.THREAD_CREATED.value,
            source=EventSource.INTERACTION.value,
            payload={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
