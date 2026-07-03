from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Protocol, Self, Sequence, Tuple

from fathom.schemas.sql import SqlParameterValue

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger


class InteractionSqlFiles:
    """
    Locates bundled SQL files for the interaction store.
    """

    def __init__(self, *, root: Path) -> None:
        """
        Capture the SQL file root.
        """

        self.__root = root

    @classmethod
    def bundled(cls) -> Self:
        """
        Return the packaged interaction SQL directory.
        """

        return cls(root=Path(__file__).parent / "sql")

    @property
    def root(self) -> Path:
        """
        Return the SQL file root.
        """

        return self.__root


class RawSqlConnection(Protocol):
    """
    Async Postgres connection surface required by raw SQL execution.
    """

    async def execute(self, query: str, *args: SqlParameterValue) -> str:
        """
        Execute one SQL command.
        """

        ...

    async def fetch(self, query: str, *args: SqlParameterValue) -> Sequence[Mapping[str, object]]:
        """
        Fetch all rows for one SQL query.
        """

        ...

    async def fetchrow(
        self, query: str, *args: SqlParameterValue
    ) -> Optional[Mapping[str, object]]:
        """
        Fetch one row for one SQL query.
        """

        ...


class CompiledSql:
    """
    SQL text compiled from named placeholders into asyncpg positional placeholders.
    """

    def __init__(self, *, statement: str, parameters: Tuple[str, ...]) -> None:
        """
        Capture compiled SQL and parameter names in bind order.
        """

        self.__statement = statement
        self.__parameters = parameters

    @property
    def statement(self) -> str:
        """
        Return the compiled SQL statement.
        """

        return self.__statement

    @property
    def parameters(self) -> Tuple[str, ...]:
        """
        Return parameter names in asyncpg bind order.
        """

        return self.__parameters

    def bind(self, *, values: Mapping[str, SqlParameterValue]) -> Tuple[SqlParameterValue, ...]:
        """
        Return positional values for the compiled SQL statement.
        """

        missing = tuple(name for name in self.__parameters if name not in values)
        if missing:
            raise ValueError(f"Missing SQL parameter(s): {', '.join(missing)}.")

        unused = tuple(name for name in values if name not in self.__parameters)
        if unused:
            raise ValueError(f"Unused SQL parameter(s): {', '.join(unused)}.")

        return tuple(values[name] for name in self.__parameters)


class RawSql:
    """
    Executes named-parameter SQL files through an async Postgres connection.
    """

    __NAMED_PARAMETER_PATTERN = re.compile(r"(?<!:):(?P<name>[A-Za-z_][A-Za-z0-9_]*)")

    def __init__(
        self,
        *,
        root: Path,
        logger: Optional[Logger] = None,
        slow_query_limit: Optional[int] = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        """
        Capture the root directory containing SQL files.
        """

        self.__root = root
        self.__logger = logger
        self.__slow_query_limit = slow_query_limit

        self.__clock = clock
        self.__compiled: Dict[str, CompiledSql] = {}

    async def execute(
        self,
        *,
        name: str,
        connection: RawSqlConnection,
        **parameters: SqlParameterValue,
    ) -> str:
        """
        Execute one named SQL file.
        """

        compiled = self.compile(name=name)
        started = self.__clock()

        try:
            return await connection.execute(
                compiled.statement,
                *compiled.bind(values=parameters),
            )
        finally:
            self.__log_slow_query(name=name, started=started)

    async def fetch(
        self,
        *,
        name: str,
        connection: RawSqlConnection,
        **parameters: SqlParameterValue,
    ) -> Sequence[Mapping[str, object]]:
        """
        Fetch all rows from one named SQL file.
        """

        compiled = self.compile(name=name)
        started = self.__clock()

        try:
            return await connection.fetch(
                compiled.statement,
                *compiled.bind(values=parameters),
            )
        finally:
            self.__log_slow_query(name=name, started=started)

    async def fetchrow(
        self,
        *,
        name: str,
        connection: RawSqlConnection,
        **parameters: SqlParameterValue,
    ) -> Optional[Mapping[str, object]]:
        """
        Fetch one row from one named SQL file.
        """

        compiled = self.compile(name=name)
        started = self.__clock()

        try:
            return await connection.fetchrow(
                compiled.statement,
                *compiled.bind(values=parameters),
            )
        finally:
            self.__log_slow_query(name=name, started=started)

    def compile(self, *, name: str) -> CompiledSql:
        """
        Compile one SQL file into asyncpg SQL and bind-order metadata.
        """

        if compiled := self.__compiled.get(name):
            return compiled

        path = self.__resolve(name=name)
        statement = path.read_text(encoding="utf-8")

        parameters: List[str] = []
        parameter_index: Dict[str, int] = {}

        def replace(match: re.Match[str]) -> str:
            parameter = match.group("name")
            index = parameter_index.get(parameter)

            if index is None:
                parameters.append(parameter)

                index = len(parameters)
                parameter_index[parameter] = index

            return f"${index}"

        compiled_statement = self.__NAMED_PARAMETER_PATTERN.sub(replace, statement)
        compiled = CompiledSql(
            statement=compiled_statement,
            parameters=tuple(parameters),
        )
        self.__compiled[name] = compiled

        return compiled

    def __resolve(self, *, name: str) -> Path:
        """
        Resolve a SQL file name under the configured root.
        """

        candidate = (self.__root / name).resolve()
        root = self.__root.resolve()

        if not candidate.is_file():
            raise FileNotFoundError(f"SQL file not found: {name}.")

        if root not in candidate.parents:
            raise ValueError(f"SQL file escapes configured root: {name}.")

        return candidate

    def __log_slow_query(self, *, name: str, started: float) -> None:
        """
        Emit a structured warning when a raw SQL file exceeds the configured threshold.
        """

        if self.__logger is None or self.__slow_query_limit is None:
            return

        duration = int((self.__clock() - started) * 1000)
        if duration < self.__slow_query_limit:
            return

        self.__logger.warning(
            "Slow raw SQL query",
            extra={
                "sql_name": name,
                "duration_ms": duration,
                "operation": "interaction.raw_sql.slow",
            },
        )
