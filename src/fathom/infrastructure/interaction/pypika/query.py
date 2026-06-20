from __future__ import annotations

from enum import StrEnum
from typing import List, Optional, Tuple, cast

from pydantic import BaseModel, ConfigDict, Field
from pypika import Order, Table
from pypika.dialects import PostgreSQLQuery, SQLLiteQuery
from pypika.queries import Query, QueryBuilder
from pypika.terms import Criterion, Parameter

from fathom.constants.storage import SqlParameterStyle
from fathom.schemas.sql import SqlParameterValue


class SortDirection(StrEnum):
    """
    Keyset-pagination scan direction.
    """

    ASCENDING = "ASC"
    DESCENDING = "DESC"


class SortOrder(BaseModel):
    """
    Keyset ordering: a primary timestamp column plus a stable tiebreaker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str = Field(min_length=1, description="Primary ordering column.")
    tiebreaker: str = Field(min_length=1, description="Stable tiebreaker column.")
    direction: SortDirection = Field(description="Direction applied to both columns.")


class CursorCoordinate(BaseModel):
    """
    Already-serialized keyset boundary (driver-shaped timestamp + identifier).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    created: SqlParameterValue = Field(description="Backend-serialized boundary timestamp.")
    identifier: str = Field(min_length=1, description="Tiebreaker identifier boundary.")


class ParameterizedQuery:
    """
    Allocates positional placeholders for Pypika query terms in bind order.
    """

    def __init__(self, *, parameter_style: SqlParameterStyle) -> None:
        """
        Capture the placeholder style for this builder.
        """

        self.__parameter_style: SqlParameterStyle = parameter_style
        self.__parameters: List[SqlParameterValue] = []

    @property
    def parameter_style(self) -> SqlParameterStyle:
        """
        Return the placeholder style for this builder.
        """

        return self.__parameter_style

    @property
    def parameters(self) -> Tuple[SqlParameterValue, ...]:
        """
        Return bound parameter values in placeholder order.
        """

        return tuple(self.__parameters)

    def bind(self, *, value: SqlParameterValue) -> Parameter:
        """
        Append a value and return its Pypika placeholder term.
        """

        return Parameter(self.bind_placeholder(value=value))

    def bind_placeholder(self, *, value: SqlParameterValue) -> str:
        """
        Append a value and return its dialect placeholder string.
        """

        self.__parameters.append(value)

        if self.__parameter_style is SqlParameterStyle.QUESTION_MARK:
            return "?"

        return f"${len(self.__parameters)}"

    def render(self, *, query: QueryBuilder) -> Tuple[str, Tuple[SqlParameterValue, ...]]:
        """
        Render the Pypika query to SQL plus the bound parameter tuple.
        """

        sql = query.get_sql(quote_char=None)
        return sql, tuple(self.__parameters)

    def snapshot(self) -> Tuple[SqlParameterValue, ...]:
        """
        Return an immutable snapshot of the parameters bound so far.
        """

        return tuple(self.__parameters)


class CursorPaginatedQuery:
    """
    Pypika-native COUNT and page builder for keyset cursor pagination.
    """

    def __init__(
        self,
        *,
        table: Table,
        ordering: SortOrder,
        parameter_style: SqlParameterStyle,
    ) -> None:
        """
        Capture the target table, ordering policy, and placeholder style.
        """

        self.__table: Table = table
        self.__ordering: SortOrder = ordering
        self.__parameter_style: SqlParameterStyle = parameter_style
        self.__binder: ParameterizedQuery = ParameterizedQuery(parameter_style=parameter_style)
        self.__filters: List[Criterion] = []
        self.__filter_parameter_count: int = 0
        self.__filters_frozen: bool = False

    @property
    def table(self) -> Table:
        """
        Return the target table this builder reads from.
        """

        return self.__table

    @property
    def binder(self) -> ParameterizedQuery:
        """
        Return the underlying parameter binder.
        """

        return self.__binder

    def bind(self, *, value: SqlParameterValue) -> Parameter:
        """
        Bind one value via the underlying binder and return its placeholder term.
        """

        if self.__filters_frozen:
            raise RuntimeError(
                "Cannot bind additional filter values after count_sql_and_parameters()."
            )
        return self.__binder.bind(value=value)

    def where(self, criterion: Criterion) -> None:
        """
        Add one filter criterion built from Pypika terms over the bound table.
        """

        if self.__filters_frozen:
            raise RuntimeError(
                "Cannot add filters after count_sql_and_parameters() has been called."
            )
        self.__filters.append(criterion)

    def count_sql_and_parameters(self) -> Tuple[str, Tuple[SqlParameterValue, ...]]:
        """
        Return COUNT(*) SQL plus snapshot of filter parameters (no cursor, no limit).
        """

        self.__freeze_filters()
        query = self.__dialect_query().from_(self.__table).select("COUNT(*)")
        for criterion in self.__filters:
            query = query.where(criterion)
        sql = query.get_sql(quote_char=None)
        return sql, self.__binder.parameters[: self.__filter_parameter_count]

    def page_sql_and_parameters(
        self,
        *,
        limit: int,
        cursor: Optional[CursorCoordinate],
    ) -> Tuple[str, Tuple[SqlParameterValue, ...]]:
        """
        Return page SQL plus parameters, appending the keyset cursor predicate if present.
        """

        self.__freeze_filters()

        query = self.__dialect_query().from_(self.__table).select(self.__table.star)
        for criterion in self.__filters:
            query = query.where(criterion)

        if cursor is not None:
            query = query.where(self.__keyset_predicate(cursor=cursor))

        order = self.__pypika_order()
        column_field = getattr(self.__table, self.__ordering.column)
        tiebreaker_field = getattr(self.__table, self.__ordering.tiebreaker)
        query = query.orderby(column_field, order=order).orderby(tiebreaker_field, order=order)
        query = query.limit(self.__binder.bind(value=limit))

        sql = query.get_sql(quote_char=None)
        return sql, self.__binder.parameters

    def __dialect_query(self) -> type[Query]:
        """
        Return the Pypika dialect class matching this builder's placeholder style.
        """

        if self.__parameter_style is SqlParameterStyle.QUESTION_MARK:
            return cast("type[Query]", SQLLiteQuery)

        return cast("type[Query]", PostgreSQLQuery)

    def __pypika_order(self) -> Order:
        """
        Return the Pypika Order constant for this builder's direction.
        """

        if self.__ordering.direction is SortDirection.ASCENDING:
            return Order.asc

        return Order.desc

    def __keyset_predicate(self, *, cursor: CursorCoordinate) -> Criterion:
        """
        Build the keyset boundary predicate `(c > t) OR (c = t AND id > i)`.
        """

        column = getattr(self.__table, self.__ordering.column)
        tiebreaker = getattr(self.__table, self.__ordering.tiebreaker)

        boundary_created_primary = self.__binder.bind(value=cursor.created)
        boundary_created_tie = self.__binder.bind(value=cursor.created)
        boundary_identifier = self.__binder.bind(value=cursor.identifier)

        if self.__ordering.direction is SortDirection.ASCENDING:
            primary = column > boundary_created_primary
            tie = (column == boundary_created_tie) & (tiebreaker > boundary_identifier)
        else:
            primary = column < boundary_created_primary
            tie = (column == boundary_created_tie) & (tiebreaker < boundary_identifier)

        return primary | tie

    def __freeze_filters(self) -> None:
        """
        Snapshot the filter-parameter cutoff before any cursor or limit binding.
        """

        if self.__filters_frozen:
            return
        self.__filter_parameter_count = len(self.__binder.parameters)
        self.__filters_frozen = True
