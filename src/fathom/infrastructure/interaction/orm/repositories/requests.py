from __future__ import annotations

from typing import Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import IdempotencyState
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import RequestRecord
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    IdentifierSource,
    TransactionScope,
)
from fathom.interaction.lifecycle import Lifecycle
from fathom.schemas.interaction import (
    BeginRequest,
    FinishRequest,
    Idempotency,
    IdempotencyQuery,
    Metadata,
    Visibility,
)


class RequestRepository:
    """
    Repository for idempotent request records.
    """

    def __init__(
        self,
        *,
        lifecycle: Lifecycle,
        transaction: TransactionScope,
        identifier_source: IdentifierSource,
    ) -> None:
        """
        Initialize request persistence collaborators.
        """

        self.__lifecycle = lifecycle
        self.__transaction = transaction
        self.__identifier_source = identifier_source

    async def begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start an idempotent request or replay the active matching record.
        """

        try:
            return await self.__begin_request(request=request)
        except IntegrityError as exception:
            existing = await self.__load_request(
                key=request.key, connection=None, tenant=request.tenant
            )
            if existing is not None and existing.hash == request.hash:
                return existing

            raise InteractionError(
                "Idempotency request conflicted with another row."
            ) from exception

    async def __begin_request(self, *, request: BeginRequest) -> Idempotency:
        """
        Start one idempotency record inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_request(
                key=request.key, connection=connection, tenant=request.tenant
            )
            if existing is not None:
                if request.created >= existing.expires:
                    await (
                        RequestRecord.filter(tenant_id=request.tenant, key=request.key)
                        .using_db(connection)
                        .delete()
                    )
                elif existing.hash != request.hash:
                    raise InteractionError(
                        "Idempotency key already exists with a different request hash."
                    )
                else:
                    return existing

            await RequestRecord.create(
                key=request.key,
                hash=request.hash,
                using_db=connection,
                tenant_id=request.tenant,
                expires_at=request.expires,
                created_at=request.created,
                workspace_id=request.workspace,
                metadata=request.metadata.entries,
                id=self.__identifier_source.next(),
                state=IdempotencyState.STARTED.value,
            )
            record = await self.__load_request(
                key=request.key, connection=connection, tenant=request.tenant
            )
            if record is None:
                raise InteractionError("Idempotency record was not persisted.")

            return record

    async def finish_request(self, *, request: FinishRequest) -> Idempotency:
        """
        Record the terminal state of an idempotent request.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_request(
                key=request.key, connection=connection, tenant=request.tenant
            )
            if existing is None:
                raise InteractionError("Idempotency record does not exist.")

            if existing.state == request.state and existing.state != IdempotencyState.STARTED:
                if self.__same_finish(record=existing, request=request):
                    return existing

                raise InteractionError(
                    "Idempotency record already finished with a different response."
                )

            self.__lifecycle.validate_request_finish(state=existing.state, target=request.state)
            await (
                RequestRecord.filter(tenant_id=request.tenant, key=request.key)
                .using_db(connection)
                .update(
                    state=request.state.value,
                    response=request.response,
                    updated_at=request.finished,
                )
            )
            finished = await self.__load_request(
                key=request.key, connection=connection, tenant=request.tenant
            )
            if finished is None:
                raise InteractionError("Idempotency record was not updated.")

        return finished

    async def get_idempotency(self, *, query: IdempotencyQuery) -> Optional[Idempotency]:
        """
        Load one tenant-scoped idempotency record.
        """

        return await self.__load_request(key=query.key, connection=None, tenant=query.tenant)

    async def __load_request(
        self,
        *,
        key: str,
        tenant: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Idempotency]:
        """
        Load one idempotency record.
        """

        queryset = RequestRecord.filter(
            tenant_id=tenant,
            key=key,
            **Visibility(archived=True).as_filters(),
        )

        if connection is not None:
            queryset = queryset.using_db(connection)

        if row := await queryset.get_or_none():
            return self.__idempotency(row=row)

        return None

    def __same_finish(self, *, record: Idempotency, request: FinishRequest) -> bool:
        """
        Check whether a finish request replays an already stored outcome.
        """

        return record.state == request.state and record.response == request.response

    def __idempotency(self, *, row: RequestRecord) -> Idempotency:
        """
        Convert one persistent idempotency model into the interaction schema.
        """

        return Idempotency(
            key=row.key,
            hash=row.hash,
            tenant=row.tenant_id,
            response=row.response,
            created_at=row.created_at,
            expires_at=row.expires_at,
            workspace=row.workspace_id,
            state=self.__state(value=row.state),
            metadata=self.__metadata(value=row.metadata),
        )

    def __state(self, *, value: str) -> IdempotencyState:
        """
        Convert stored idempotency state text into the public enum.
        """

        try:
            return IdempotencyState(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown idempotency state in row: {value}.") from exception

    def __metadata(self, *, value: JsonValue) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError("Invalid idempotency metadata in row.")
