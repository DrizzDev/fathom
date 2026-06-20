from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple

from pydantic import JsonValue

if TYPE_CHECKING:
    from datetime import datetime


class FakeCursor:
    """
    Minimal cursor stand-in returning a configured list of result rows.
    """

    def __init__(self, *, rows: Tuple[Dict[str, Any], ...]) -> None:
        """
        Bind the rows that fetchall / fetchone will surface.
        """

        self.__rows = list(rows)

    async def fetchall(self) -> List[Dict[str, Any]]:
        """
        Return all configured rows in order.
        """

        return list(self.__rows)

    async def fetchone(self) -> Optional[Dict[str, Any]]:
        """
        Return the first row or None when empty.
        """

        return self.__rows[0] if self.__rows else None


class FakeExecution:
    """
    Awaitable + async-context wrapper mirroring PostgresExecution.
    """

    def __init__(
        self,
        *,
        rows: Tuple[Dict[str, Any], ...] = (),
        rowcount: int = 0,
    ) -> None:
        """
        Bind the rows surfaced as a cursor and the rowcount surfaced when awaited.
        """

        self.__rows = rows
        self.rowcount = rowcount

    def __await__(self):
        """
        Allow `result = await connection.execute(...)` and expose rowcount.
        """

        async def _resolve() -> "FakeExecution":
            return self

        return _resolve().__await__()

    async def __aenter__(self) -> FakeCursor:
        """
        Allow `async with connection.execute(...) as cursor:` for SELECTs.
        """

        return FakeCursor(rows=self.__rows)

    async def __aexit__(self, *args: object) -> None:
        """
        Leave the async context without suppressing exceptions.
        """

        return None


class FakeConnection:
    """
    Recording connection that hands out scripted executions in call order.
    """

    def __init__(self, *, responses: Tuple[FakeExecution, ...] = ()) -> None:
        """
        Bind the queue of scripted executions and reset the call log.
        """

        self.calls: List[Tuple[str, Tuple[object, ...]]] = []
        self.__responses = list(responses)

    def execute(
        self,
        sql: str,
        parameters: Tuple[object, ...] | List[object] = (),
    ) -> FakeExecution:
        """
        Capture the SQL/parameter pair and pop the next scripted execution.
        """

        self.calls.append((sql, tuple(parameters)))
        if not self.__responses:
            return FakeExecution()
        return self.__responses.pop(0)


class FakeUnit:
    """
    Unit-of-work facade that hands out the configured fake connection.
    """

    def __init__(self, *, connection: FakeConnection) -> None:
        """
        Bind the fake connection that session() will yield.
        """

        self.__connection = connection

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[FakeConnection, None]:
        """
        Yield the fake connection without opening a real transaction.
        """

        yield self.__connection

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Enter the transactional boundary that reuses the active session.
        """

        yield None


class FakePostgresContext:
    """
    Minimal Postgres context used by repository unit tests.

    Exposes the unit-of-work, datetime/json passthrough helpers, and
    hooks for `_load_*` / `_require_*` lookups. Tests configure the
    behaviour they care about by setting attributes on the instance.
    """

    def __init__(self, *, connection: FakeConnection) -> None:
        """
        Bind the recording connection that the unit will yield.
        """

        self.__unit = FakeUnit(connection=connection)
        self.load_actor_return: Optional[Any] = None
        self.load_thread_return: Optional[Any] = None
        self.load_task_return: Optional[Any] = None
        self.load_message_return: Optional[Any] = None
        self.load_artifact_return: Optional[Any] = None
        self.load_context_return: Optional[Any] = None
        self.load_membership_return: Optional[Any] = None
        self.load_job_return: Optional[Any] = None
        self.load_idempotency_return: Optional[Any] = None
        self.load_policy_return: Optional[Any] = None
        self.load_script_return: Optional[Any] = None
        self.next_event_sequence_return: int = 1
        self.next_message_sequence_return: int = 1
        self.record_event_calls: List[Dict[str, Any]] = []
        self.touch_thread_calls: List[Dict[str, Any]] = []
        self.require_actor_calls: List[Dict[str, Any]] = []
        self.require_thread_calls: List[Dict[str, Any]] = []
        self.require_task_calls: List[Dict[str, Any]] = []
        self.require_task_in_thread_calls: List[Dict[str, Any]] = []
        self.require_message_in_thread_calls: List[Dict[str, Any]] = []
        self.require_active_membership_calls: List[Dict[str, Any]] = []

    @property
    def unit(self) -> FakeUnit:
        """
        Expose the fake unit-of-work to the repository under test.
        """

        return self.__unit

    @property
    def rows(self) -> Any:
        """
        Return a placeholder row mapper that tests configure if needed.
        """

        return self

    def _time(self, *, value: datetime) -> datetime:
        """
        Pass datetimes through unchanged (asyncpg-native timestamptz path).
        """

        return value

    def _optional_time(self, *, value: Optional[datetime]) -> Optional[datetime]:
        """
        Pass optional datetimes through unchanged.
        """

        return value

    def _json(self, *, value: JsonValue) -> JsonValue:
        """
        Pass JSON values through unchanged (asyncpg-native JSONB path).
        """

        return value

    async def _load_actor(self, *, connection: Any, tenant: str, actor: str) -> Optional[Any]:
        """
        Return the test-configured actor lookup.
        """

        return self.load_actor_return

    async def _load_thread(self, *, connection: Any, tenant: str, thread: str) -> Optional[Any]:
        """
        Return the test-configured thread lookup.
        """

        return self.load_thread_return

    async def _load_task(self, *, connection: Any, tenant: str, task: str) -> Optional[Any]:
        """
        Return the test-configured task lookup.
        """

        return self.load_task_return

    async def _load_message(self, *, connection: Any, tenant: str, message: str) -> Optional[Any]:
        """
        Return the test-configured message lookup.
        """

        return self.load_message_return

    async def _load_artifact(self, *, connection: Any, tenant: str, artifact: str) -> Optional[Any]:
        """
        Return the test-configured artifact lookup.
        """

        return self.load_artifact_return

    async def _load_context(self, *, connection: Any, tenant: str, context: str) -> Optional[Any]:
        """
        Return the test-configured context lookup.
        """

        return self.load_context_return

    async def _load_membership(
        self, *, connection: Any, tenant: str, membership: str
    ) -> Optional[Any]:
        """
        Return the test-configured membership lookup.
        """

        return self.load_membership_return

    async def _load_job(self, *, connection: Any, tenant: str, job: str) -> Optional[Any]:
        """
        Return the test-configured job lookup.
        """

        return self.load_job_return

    async def _load_idempotency(self, *, connection: Any, tenant: str, key: str) -> Optional[Any]:
        """
        Return the test-configured idempotency lookup.
        """

        return self.load_idempotency_return

    async def _load_policy(self, *, connection: Any, tenant: str, policy: str) -> Optional[Any]:
        """
        Return the test-configured policy lookup.
        """

        return self.load_policy_return

    async def _load_script(self, *, connection: Any, tenant: str, script: str) -> Optional[Any]:
        """
        Return the test-configured script lookup.
        """

        return self.load_script_return

    async def _require_actor(self, *, connection: Any, tenant: str, actor: str) -> None:
        """
        Record the foreign-key actor existence check.
        """

        self.require_actor_calls.append({"tenant": tenant, "actor": actor})

    async def _require_thread(self, *, connection: Any, tenant: str, thread: str) -> None:
        """
        Record the foreign-key thread existence check.
        """

        self.require_thread_calls.append({"tenant": tenant, "thread": thread})

    async def _require_task(self, *, connection: Any, tenant: str, task: str) -> None:
        """
        Record the foreign-key task existence check.
        """

        self.require_task_calls.append({"tenant": tenant, "task": task})

    async def _require_task_in_thread(
        self, *, connection: Any, tenant: str, thread: str, task: str
    ) -> None:
        """
        Record the foreign-key task-in-thread check.
        """

        self.require_task_in_thread_calls.append({"tenant": tenant, "thread": thread, "task": task})

    async def _require_message_in_thread(
        self, *, connection: Any, tenant: str, thread: str, message: str
    ) -> None:
        """
        Record the foreign-key message-in-thread check.
        """

        self.require_message_in_thread_calls.append(
            {"tenant": tenant, "thread": thread, "message": message}
        )

    async def _require_active_membership(
        self, *, connection: Any, tenant: str, thread: str, actor: str
    ) -> None:
        """
        Record the active-membership precondition check.
        """

        self.require_active_membership_calls.append(
            {"tenant": tenant, "thread": thread, "actor": actor}
        )

    async def _record_event(self, **kwargs: Any) -> None:
        """
        Capture the structured event emission for later assertion.
        """

        self.record_event_calls.append(kwargs)

    async def _touch_thread(self, **kwargs: Any) -> None:
        """
        Capture the thread cursor/timestamp advance.
        """

        self.touch_thread_calls.append(kwargs)

    async def _next_event_sequence(self, *, connection: Any, tenant: str, thread: str) -> int:
        """
        Return the test-configured event sequence value.
        """

        return self.next_event_sequence_return

    async def _next_message_sequence(self, *, connection: Any, tenant: str, thread: str) -> int:
        """
        Return the test-configured message sequence value.
        """

        return self.next_message_sequence_return
