from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Generic, Optional, Tuple, TypeVar

from tortoise.expressions import Q
from tortoise.models import Model
from tortoise.queryset import QuerySet

from fathom.conversation.cursor import OpaqueCursor
from fathom.schemas.interaction import SortOrder

if TYPE_CHECKING:
    from datetime import datetime

Row = TypeVar("Row", bound=Model)
Item = TypeVar("Item")


class TimestampColumn(StrEnum):
    """
    Timestamp columns supported by time windows and keyset pagination.
    """

    CREATED = "created_at"
    UPDATED = "updated_at"


class Page(Generic[Item]):
    """
    Holds one repository page and its optional next cursor.
    """

    def __init__(self, *, items: Tuple[Item, ...], next: Optional[str]) -> None:
        """
        Store one page of projected repository items.
        """

        self.items = items
        self.next = next


class KeysetPaginator(Generic[Row, Item]):
    """
    Emits one keyset page for a queryset ordered by (timestamp, id).
    """

    def __init__(self, *, column: TimestampColumn) -> None:
        """
        Bind the timestamp column used for ordering and boundary predicates.
        """

        self.__column = column

    async def paginate(
        self,
        *,
        queryset: QuerySet[Row],
        limit: int,
        order: SortOrder,
        cursor: Optional[str],
        project: Callable[[Row], Item],
        identity: Callable[[Item], str],
        stamp: Callable[[Item], datetime],
    ) -> Page[Item]:
        """
        Load one keyset page using this paginator's timestamp column.
        """

        descending = order is SortOrder.DESC

        if cursor is not None:
            queryset = queryset.filter(self.__boundary(cursor=cursor, descending=descending))

        id_order = "-id" if descending else "id"
        order_field = f"-{self.__column.value}" if descending else self.__column.value

        rows = await queryset.order_by(order_field, id_order).limit(limit + 1)

        projected = tuple(project(row) for row in rows)
        items = projected[:limit]

        if len(projected) <= limit or not items:
            return Page(items=items, next=None)

        last = items[-1]
        marker = stamp(last)

        return Page(
            items=items,
            next=OpaqueCursor(created=marker, identifier=identity(last)).encode(),
        )

    def __boundary(self, *, cursor: str, descending: bool) -> Q:
        """
        Build a keyset boundary predicate from an opaque cursor.
        """

        boundary = OpaqueCursor.decode(value=cursor)
        if self.__column is TimestampColumn.CREATED:
            return self.__boundary_created(boundary=boundary, descending=descending)
        return self.__boundary_updated(boundary=boundary, descending=descending)

    @staticmethod
    def __boundary_created(*, boundary: OpaqueCursor, descending: bool) -> Q:
        """
        Build a keyset boundary predicate on the created_at column.
        """

        if descending:
            return Q(created_at__lt=boundary.created) | Q(
                id__lt=boundary.identifier, created_at=boundary.created
            )

        return Q(created_at__gt=boundary.created) | Q(
            id__gt=boundary.identifier, created_at=boundary.created
        )

    @staticmethod
    def __boundary_updated(*, boundary: OpaqueCursor, descending: bool) -> Q:
        """
        Build a keyset boundary predicate on the updated_at column.
        """

        if descending:
            return Q(updated_at__lt=boundary.created) | Q(
                id__lt=boundary.identifier, updated_at=boundary.created
            )

        return Q(updated_at__gt=boundary.created) | Q(
            id__gt=boundary.identifier, updated_at=boundary.created
        )
