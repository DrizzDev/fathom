from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    AsyncContextManager,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
)
from uuid import uuid4

from tortoise.backends.base.client import BaseDBAsyncClient
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import EVENT_SOURCE_ACTORS, ActorKind, EventKind, EventSource
from fathom.constants.conversation import SequenceScope
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import ActorRecord, EventRecord
from fathom.infrastructure.interaction.orm.raw import RawSql
from fathom.interaction.digest import EventDigest
from fathom.schemas.interaction import Metadata

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.schemas.sql import SqlParameterValue

DatabaseConnection: TypeAlias = BaseDBAsyncClient


class IdentifierSource(Protocol):
    """
    Supplies persistence identifiers at explicit composition boundaries.
    """

    def next(self) -> str:
        """
        Return one new opaque identifier.
        """
        ...


class TransactionScope(Protocol):
    """
    Opens transaction boundaries for repository write operations.
    """

    def transaction(self) -> AsyncContextManager[DatabaseConnection]:
        """
        Open one active store transaction.
        """
        ...


class UuidIdentifierSource:
    """
    Supplies random UUID identifiers for persistence-only rows.
    """

    def next(self) -> str:
        """
        Return one UUID string.
        """

        return str(uuid4())


class RawConnectionAdapter:
    """
    Adapts an store connection client to the raw SQL executor protocol.
    """

    def __init__(self, *, connection: DatabaseConnection) -> None:
        """
        Capture the active store connection.
        """

        self.__connection = connection

    async def execute(self, query: str, *args: SqlParameterValue) -> str:
        """
        Execute one SQL statement.
        """

        row_count, _ = await self.__connection.execute_query(query, list(args))
        return str(row_count)

    async def fetch(self, query: str, *args: SqlParameterValue) -> Sequence[Mapping[str, object]]:
        """
        Fetch all rows for one SQL statement.
        """

        rows = await self.__connection.execute_query_dict(query, list(args))
        return tuple(dict(row) for row in rows)

    async def fetchrow(
        self, query: str, *args: SqlParameterValue
    ) -> Optional[Mapping[str, object]]:
        """
        Fetch one row for one SQL statement.
        """

        rows = await self.fetch(query, *args)
        if not rows:
            return None

        return rows[0]


class LifecycleRecorder:
    """
    Records lifecycle events and advances thread activity.
    """

    def __init__(
        self,
        *,
        raw: RawSql,
        event_digest: EventDigest,
        identifier_source: IdentifierSource,
        sequence_allocator: SequenceAllocator,
    ) -> None:
        """
        Initialize lifecycle persistence collaborators.
        """

        self.__raw = raw
        self.__event_digest = event_digest
        self.__identifier_source = identifier_source
        self.__sequence_allocator = sequence_allocator

    async def record(
        self,
        *,
        tenant: str,
        thread: str,
        kind: EventKind,
        payload: Metadata,
        created: datetime,
        workspace: Optional[str],
        touch_thread: bool = True,
        task: Optional[str] = None,
        actor: Optional[str] = None,
        execution: Optional[str] = None,
        connection: DatabaseConnection,
        source: EventSource = EventSource.INTERACTION,
    ) -> None:
        """
        Persist one lifecycle event and optionally touch the parent thread.
        """

        event_actor = actor or EVENT_SOURCE_ACTORS[source]

        if actor is None:
            await self.__ensure_system_actor(
                tenant=tenant,
                created=created,
                actor=event_actor,
                workspace=workspace,
                connection=connection,
            )

        sequence = await self.__next_event_sequence(
            tenant=tenant,
            thread=thread,
            connection=connection,
        )
        await EventRecord.create(
            task_id=task,
            kind=kind.value,
            tenant_id=tenant,
            actor=event_actor,
            sequence=sequence,
            created_at=created,
            using_db=connection,
            source=source.value,
            workspace_id=workspace,
            execution_id=execution,
            conversation_id=thread,
            updated_by=event_actor,
            created_by=event_actor,
            payload=payload.entries,
            id=self.__identifier_source.next(),
        )
        if touch_thread:
            await self.__touch_thread(
                tenant=tenant,
                thread=thread,
                updated=created,
                sequence=sequence,
                digest=self.__event_digest.compute(
                    kind=kind,
                    source=source,
                    payload=payload,
                    created=created,
                    sequence=sequence,
                ),
                connection=connection,
            )

    async def __ensure_system_actor(
        self,
        *,
        actor: str,
        tenant: str,
        created: datetime,
        workspace: Optional[str],
        connection: DatabaseConnection,
    ) -> None:
        """
        Ensure the generated lifecycle actor satisfies event audit constraints.
        """

        exists = await ActorRecord.filter(tenant_id=tenant, id=actor).using_db(connection).exists()
        if exists:
            return

        try:
            await ActorRecord.create(
                id=actor,
                name=actor,
                created_by=actor,
                updated_by=actor,
                tenant_id=tenant,
                created_at=created,
                updated_at=created,
                using_db=connection,
                workspace_id=workspace,
                kind=ActorKind.SYSTEM.value,
            )
        except IntegrityError:
            return

    async def __next_event_sequence(
        self, *, tenant: str, thread: str, connection: DatabaseConnection
    ) -> int:
        """
        Allocate the next event sequence for a thread.
        """

        return await self.__sequence_allocator.next(
            tenant=tenant,
            thread=thread,
            connection=connection,
            scope=SequenceScope.EVENT.value,
        )

    async def __touch_thread(
        self,
        *,
        tenant: str,
        thread: str,
        digest: str,
        sequence: int,
        updated: datetime,
        connection: DatabaseConnection,
    ) -> None:
        """
        Advance conversation activity and digest after an event.
        """

        affected = await self.__raw.execute(
            tenant=tenant,
            digest=digest,
            thread=thread,
            updated=updated,
            sequence=sequence,
            name="conversations/touch.sql",
            connection=RawConnectionAdapter(connection=connection),
        )

        if self.__no_rows_affected(status=affected):
            row = await self.__raw.fetchrow(
                tenant=tenant,
                thread=thread,
                name="conversations/exists.sql",
                connection=RawConnectionAdapter(connection=connection),
            )

            if row is None:
                raise InteractionError("Thread does not exist.")

    @staticmethod
    def __no_rows_affected(*, status: str) -> bool:
        """
        Return true when the execute status reports zero affected rows.
        """

        head, _, tail = status.rpartition(" ")
        candidate = tail if head else status
        try:
            return int(candidate) == 0
        except ValueError:
            return False


class SequenceAllocator:
    """
    Allocates tenant/thread-scoped monotonically increasing sequence numbers.
    """

    def __init__(self, *, raw: RawSql, identifier_source: IdentifierSource) -> None:
        """
        Initialize raw SQL sequence allocation.
        """

        self.__raw = raw
        self.__identifier_source = identifier_source

    async def next(
        self,
        *,
        scope: str,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> int:
        """
        Allocate the next sequence value for one thread scope.
        """

        row = await self.__raw.fetchrow(
            scope=scope,
            tenant=tenant,
            thread=thread,
            name="sequences/allocate.sql",
            id=self.__identifier_source.next(),
            connection=RawConnectionAdapter(connection=connection),
        )
        if row is None:
            raise InteractionError(f"Sequence could not be allocated for scope '{scope}'.")

        value = row["value"]

        if not isinstance(value, int):
            raise InteractionError(f"Sequence for scope '{scope}' returned a non-integer value.")

        return value
