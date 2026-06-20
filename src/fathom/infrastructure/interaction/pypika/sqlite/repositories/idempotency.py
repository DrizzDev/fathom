from __future__ import annotations

from typing import Optional

import aiosqlite
from pypika import SQLLiteQuery

from fathom.constants.collaboration import IdempotencyState
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
from fathom.infrastructure.interaction.pypika.sqlite import tables
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.schemas.interaction import (
    BeginRequest,
    FinishRequest,
    Idempotency,
    IdempotencyQuery,
)


class IdempotencyRepository:
    """
    Idempotency repository: tracks request hashes and terminal outcomes.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for requests persistence.
        """

        self.__context = context

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start an idempotent request and return the active record.
        """

        async with self.__context.unit.session() as connection:
            existing = await self.__load_requests_row(
                connection=connection,
                tenant=request.tenant,
                key=request.key,
            )
            if existing is not None:
                if request.created >= existing.expires:
                    delete_binder = ParameterizedQuery(
                        parameter_style=SqlParameterStyle.QUESTION_MARK
                    )
                    requests = tables.REQUESTS
                    delete_statement = (
                        SQLLiteQuery.from_(requests)
                        .delete()
                        .where(requests.tenant == delete_binder.bind(value=request.tenant))
                        .where(requests.key == delete_binder.bind(value=request.key))
                    )
                    delete_sql, delete_parameters = delete_binder.render(query=delete_statement)
                    await connection.execute(delete_sql, delete_parameters)
                elif existing.hash != request.hash:
                    raise InteractionError(
                        "Idempotency key already exists with a different request hash."
                    )
                else:
                    return existing

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
            requests = tables.REQUESTS
            statement = (
                SQLLiteQuery.into(requests)
                .columns(
                    requests.tenant,
                    requests.key,
                    requests.hash,
                    requests.state,
                    requests.response,
                    requests.created_at,
                    requests.expires_at,
                    requests.metadata,
                )
                .insert(
                    binder.bind(value=request.tenant),
                    binder.bind(value=request.key),
                    binder.bind(value=request.hash),
                    binder.bind(value=IdempotencyState.STARTED.value),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._time(value=request.created)),
                    binder.bind(value=self.__context._time(value=request.expires)),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)
            record = await self.__load_requests_row(
                connection=connection,
                tenant=request.tenant,
                key=request.key,
            )

        if record is None:
            raise InteractionError("Idempotency record was not persisted.")

        return record

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Record the terminal state of an idempotent request.
        """

        async with self.__context.unit.session() as connection:
            existing = await self.__load_requests_row(
                connection=connection,
                tenant=request.tenant,
                key=request.key,
            )
            if existing is None:
                raise InteractionError("Idempotency record does not exist.")

            if existing.state == request.state and existing.state != IdempotencyState.STARTED:
                if not self.__same_finish_request(record=existing, request=request):
                    raise InteractionError(
                        "Idempotency record already finished with a different response."
                    )

                return existing

            self.__context.lifecycle.validate_request_finish(
                state=existing.state,
                target=request.state,
            )
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
            requests = tables.REQUESTS
            statement = (
                SQLLiteQuery.update(requests)
                .set(requests.state, binder.bind(value=request.state.value))
                .set(
                    requests.response,
                    binder.bind(value=self.__context._optional_json(value=request.response)),
                )
                .where(requests.tenant == binder.bind(value=request.tenant))
                .where(requests.key == binder.bind(value=request.key))
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)
            finished = await self.__load_requests_row(
                connection=connection,
                tenant=request.tenant,
                key=request.key,
            )

        if finished is None:
            raise InteractionError("Idempotency record was not updated.")

        return finished

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one tenant-scoped requests record.
        """

        async with self.__context.unit.session() as connection:
            return await self.__load_requests_row(
                connection=connection,
                tenant=query.tenant,
                key=query.key,
            )

    async def __load_requests_row(
        self,
        *,
        connection: aiosqlite.Connection,
        tenant: str,
        key: str,
    ) -> Optional[Idempotency]:
        """
        Load one requests row.
        """

        binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
        requests = tables.REQUESTS
        statement = (
            SQLLiteQuery.from_(requests)
            .select(requests.star)
            .where(requests.tenant == binder.bind(value=tenant))
            .where(requests.key == binder.bind(value=key))
        )
        sql, parameters = binder.render(query=statement)
        async with connection.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.idempotency(row=row)

    def __same_finish_request(self, *, record: Idempotency, request: FinishRequest) -> bool:
        """
        Check whether a finish request replays an already stored requests outcome.
        """

        return record.state == request.state and record.response == request.response
