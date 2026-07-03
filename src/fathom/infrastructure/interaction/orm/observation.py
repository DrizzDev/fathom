from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Callable, List, Mapping, Optional, Protocol, Sequence, Tuple

from fathom.schemas.sql import SqlParameterValue

if TYPE_CHECKING:
    from logging import Logger


class QueryResult(Protocol):
    """
    Callable surface for Tortoise row-returning SQL execution.
    """

    async def __call__(
        self,
        query: str,
        values: Optional[List[SqlParameterValue]] = None,
    ) -> Tuple[int, Sequence[Mapping[str, object]]]:
        """
        Execute one SQL query and return affected count plus rows.
        """

        ...


class QueryDictResult(Protocol):
    """
    Callable surface for Tortoise dictionary-row SQL execution.
    """

    async def __call__(
        self,
        query: str,
        values: Optional[List[SqlParameterValue]] = None,
    ) -> List[Mapping[str, object]]:
        """
        Execute one SQL query and return dictionary rows.
        """

        ...


class InsertResult(Protocol):
    """
    Callable surface for Tortoise insert execution.
    """

    async def __call__(
        self,
        query: str,
        values: List[SqlParameterValue],
    ) -> object:
        """
        Execute one insert statement and return the driver result.
        """

        ...


class BatchResult(Protocol):
    """
    Callable surface for Tortoise batch SQL execution.
    """

    async def __call__(
        self,
        query: str,
        values: List[List[SqlParameterValue]],
    ) -> None:
        """
        Execute one batch statement.
        """

        ...


class ScriptResult(Protocol):
    """
    Callable surface for Tortoise SQL script execution.
    """

    async def __call__(self, query: str) -> None:
        """
        Execute one SQL script.
        """

        ...


class ObservableClient(Protocol):
    """
    Mutable database client methods observed for slow-query logging.
    """

    execute_query: QueryResult
    execute_query_dict: QueryDictResult

    execute_many: BatchResult
    execute_insert: InsertResult
    execute_script: ScriptResult


class QueryObserver:
    """
    Adds slow-query logging around the Tortoise database client.
    """

    __OBSERVED_ATTRIBUTE = "_fathom_query_observed"

    def __init__(
        self,
        *,
        logger: Logger,
        threshold: int,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        """
        Capture logging dependencies for query observation.
        """

        self.__clock = clock
        self.__logger = logger
        self.__threshold = threshold

    def observe(self, *, client: ObservableClient) -> None:
        """
        Attach slow-query observation to one Tortoise database client.
        """

        if getattr(client, self.__OBSERVED_ATTRIBUTE, False):
            return

        execute_query = client.execute_query
        execute_query_dict = client.execute_query_dict

        execute_many = client.execute_many
        execute_insert = client.execute_insert
        execute_script = client.execute_script

        client.execute_query = self.__query(call=execute_query)
        client.execute_query_dict = self.__dict_query(call=execute_query_dict)

        client.execute_many = self.__batch(call=execute_many)
        client.execute_insert = self.__insert(call=execute_insert)
        client.execute_script = self.__script(call=execute_script)

        setattr(client, self.__OBSERVED_ATTRIBUTE, True)

    def __query(self, *, call: QueryResult) -> QueryResult:
        """
        Return an observed row-returning query callable.
        """

        async def observed(
            query: str,
            values: Optional[List[SqlParameterValue]] = None,
        ) -> Tuple[int, Sequence[Mapping[str, object]]]:
            """
            Execute one observed row-returning query.
            """

            started = self.__clock()

            try:
                return await call(query, values)
            finally:
                self.__log(operation="interaction.orm.query", query=query, started=started)

        return observed

    def __dict_query(self, *, call: QueryDictResult) -> QueryDictResult:
        """
        Return an observed dictionary-row query callable.
        """

        async def observed(
            query: str,
            values: Optional[List[SqlParameterValue]] = None,
        ) -> List[Mapping[str, object]]:
            """
            Execute one observed dictionary-row query.
            """

            started = self.__clock()

            try:
                return await call(query, values)
            finally:
                self.__log(operation="interaction.orm.query_dict", query=query, started=started)

        return observed

    def __insert(self, *, call: InsertResult) -> InsertResult:
        """
        Return an observed insert callable.
        """

        async def observed(query: str, values: List[SqlParameterValue]) -> object:
            """
            Execute one observed insert statement.
            """

            started = self.__clock()

            try:
                return await call(query, values)
            finally:
                self.__log(operation="interaction.orm.insert", query=query, started=started)

        return observed

    def __batch(self, *, call: BatchResult) -> BatchResult:
        """
        Return an observed batch callable.
        """

        async def observed(query: str, values: List[List[SqlParameterValue]]) -> None:
            """
            Execute one observed batch statement.
            """

            started = self.__clock()

            try:
                await call(query, values)
            finally:
                self.__log(operation="interaction.orm.batch", query=query, started=started)

        return observed

    def __script(self, *, call: ScriptResult) -> ScriptResult:
        """
        Return an observed script callable.
        """

        async def observed(query: str) -> None:
            """
            Execute one observed SQL script.
            """

            started = self.__clock()

            try:
                await call(query)
            finally:
                self.__log(operation="interaction.orm.script", query=query, started=started)

        return observed

    def __log(self, *, operation: str, query: str, started: float) -> None:
        """
        Emit a structured warning when query duration crosses the threshold.
        """

        duration = int((self.__clock() - started) * 1000)

        if duration < self.__threshold:
            return

        self.__logger.warning(
            "Slow ORM query",
            extra={
                "operation": operation,
                "duration_ms": duration,
                "sql_head": self.__head(query=query),
            },
        )

    def __head(self, *, query: str) -> str:
        """
        Return a compact SQL prefix safe for structured logs.
        """

        return " ".join(query.split())[:160]
