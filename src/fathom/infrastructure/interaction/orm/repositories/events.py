from __future__ import annotations

from typing import TYPE_CHECKING, List

from pydantic import JsonValue

from fathom.constants.collaboration import EVENT_SOURCE_ACTORS, EventKind, EventSource
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import EventRecord
from fathom.infrastructure.interaction.orm.repositories.paginator import (
    KeysetPaginator,
    TimestampColumn,
)
from fathom.infrastructure.interaction.orm.repositories.reference import ReferenceGuard
from fathom.schemas.interaction import (
    Event,
    EventCursorQuery,
    EventPage,
    EventQuery,
    Identity,
    Metadata,
    ThreadReference,
    ThreadScope,
    Visibility,
)

if TYPE_CHECKING:
    from datetime import datetime


class EventRepository:
    """
    Read-only repository for lifecycle events.
    """

    def __init__(self, *, references: ReferenceGuard) -> None:
        """
        Initialize the event repository with a shared reference guard.
        """

        self.__guard = references

    async def get_events(self, *, query: EventQuery) -> List[Event]:
        """
        Load visible tenant-scoped events for one thread and optional task.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return []

        queryset = EventRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)

        rows = await queryset.order_by("sequence")
        return [self.__event(row=row) for row in rows]

    async def list_events(self, *, query: EventCursorQuery) -> EventPage:
        """
        Load visible lifecycle events with keyset pagination.
        """

        if not await self.__guard.thread_visible(scope=self.__scope(query=query)):
            return EventPage(items=(), next=None, total=0)

        queryset = EventRecord.filter(
            tenant_id=query.tenant,
            conversation_id=query.thread,
            **Visibility(deleted=query.include_deleted, archived=True).as_filters(),
        )
        if query.task is not None:
            queryset = queryset.filter(task_id=query.task)
        if query.actor is not None:
            queryset = queryset.filter(actor=query.actor)
        if query.since is not None:
            queryset = queryset.filter(created_at__gte=query.since)
        if query.until is not None:
            queryset = queryset.filter(created_at__lt=query.until)
        if query.kinds:
            queryset = queryset.filter(kind__in=tuple(kind.value for kind in query.kinds))

        total = await queryset.count() if query.count_total else 0

        page = await KeysetPaginator[EventRecord, Event](
            column=TimestampColumn.CREATED,
        ).paginate(
            queryset=queryset,
            limit=query.limit,
            order=query.order,
            cursor=query.cursor,
            project=self.__page_event,
            stamp=self.__event_created,
            identity=self.__event_identity,
        )

        return EventPage(items=page.items, next=page.next, total=total)

    def __scope(self, *, query: EventQuery | EventCursorQuery) -> ThreadScope:
        """
        Build a thread scope from an event read query.
        """

        return ThreadScope(
            reference=ThreadReference(tenant=query.tenant, thread=query.thread),
            visibility=Visibility(
                deleted=query.include_deleted,
                archived=query.include_archived,
            ),
        )

    def __event(self, *, row: EventRecord) -> Event:
        """
        Convert one persistent event model into the interaction schema.
        """

        return Event(
            task=row.task_id,
            sequence=row.sequence,
            created_at=row.created_at,
            thread=row.conversation_id,
            actor=self.__actor(row=row),
            kind=self.__kind(value=row.kind),
            source=self.__source(value=row.source),
            payload=self.__metadata(value=row.payload, field="payload"),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __actor(self, *, row: EventRecord) -> str:
        """
        Return the stored actor or the canonical source actor for system events.
        """

        if isinstance(row.actor, str):
            return row.actor

        return EVENT_SOURCE_ACTORS[self.__source(value=row.source)]

    def __page_event(self, row: EventRecord) -> Event:
        """
        Convert one event row for pagination.
        """

        return self.__event(row=row)

    def __event_created(self, event: Event) -> datetime:
        """
        Return the event creation timestamp used by pagination.
        """

        return event.created

    def __event_identity(self, event: Event) -> str:
        """
        Return the event identifier used by pagination.
        """

        return event.identity.id

    def __kind(self, *, value: str) -> EventKind:
        """
        Convert stored event kind text into the public enum.
        """

        try:
            return EventKind(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown event kind in row: {value}.") from exception

    def __source(self, *, value: str) -> EventSource:
        """
        Convert stored event source text into the public enum.
        """

        try:
            return EventSource(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown event source in row: {value}.") from exception

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError(f"Invalid event {field} metadata in row.")
