from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import IdempotencyState
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import RequestRecord
from fathom.schemas.interaction import BeginRequest, FinishRequest, IdempotencyQuery, Metadata


class TestRequestRepository:
    """
    Verify idempotency request persistence through the persistent-store backed repository.
    """

    async def test_begin_request_creates_started_record_with_private_uuid_id(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            request = self.__begin_request()

            result = await InteractionRepositoryFactory().requests().begin_request(request=request)
            row = await RequestRecord.get(tenant_id=request.tenant, key=request.key)

            assert UUID(row.id)
            assert result.tenant == request.tenant
            assert result.key == request.key
            assert result.hash == request.hash
            assert result.state == IdempotencyState.STARTED
            assert result.metadata == request.metadata

    async def test_begin_request_replays_matching_active_record(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            request = self.__begin_request()
            repository = InteractionRepositoryFactory().requests()

            created = await repository.begin_request(request=request)
            replayed = await repository.begin_request(request=request)

            assert replayed == created
            assert (
                await RequestRecord.filter(tenant_id=request.tenant, key=request.key).count() == 1
            )

    async def test_begin_request_rejects_active_hash_conflict(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            request = self.__begin_request()
            repository = InteractionRepositoryFactory().requests()
            await repository.begin_request(request=request)
            conflict = request.model_copy(update={"hash": "different"})

            with pytest.raises(InteractionError, match="different request hash"):
                await repository.begin_request(request=conflict)

    async def test_begin_request_replaces_expired_record(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            created = datetime(2026, 1, 1, tzinfo=timezone.utc)
            expired = self.__begin_request(
                created=created,
                expires=created + timedelta(seconds=1),
                hash_value="old",
            )
            repository = InteractionRepositoryFactory().requests()
            await repository.begin_request(request=expired)
            replacement = self.__begin_request(
                created=created + timedelta(seconds=2),
                expires=created + timedelta(seconds=10),
                hash_value="new",
            )

            result = await repository.begin_request(request=replacement)

            assert result.hash == "new"
            assert (
                await RequestRecord.filter(tenant_id=expired.tenant, key=expired.key).count() == 1
            )

    async def test_finish_request_records_terminal_response(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            repository = InteractionRepositoryFactory().requests()
            await repository.begin_request(request=begin)
            finish = self.__finish_request()

            result = await repository.finish_request(request=finish)

            assert result.state == IdempotencyState.COMPLETED
            assert result.response == {"id": "result"}

    async def test_finish_request_replays_matching_terminal_response(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            finish = self.__finish_request()
            repository = InteractionRepositoryFactory().requests()
            await repository.begin_request(request=begin)
            completed = await repository.finish_request(request=finish)
            replayed = await repository.finish_request(request=finish)

            assert replayed == completed

    async def test_finish_request_rejects_terminal_response_conflict(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            finish = self.__finish_request()
            repository = InteractionRepositoryFactory().requests()
            await repository.begin_request(request=begin)
            await repository.finish_request(request=finish)
            conflict = finish.model_copy(update={"response": {"id": "other"}})

            with pytest.raises(InteractionError, match="different response"):
                await repository.finish_request(request=conflict)

    async def test_finish_request_requires_existing_started_record(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            with pytest.raises(InteractionError, match="does not exist"):
                await (
                    InteractionRepositoryFactory()
                    .requests()
                    .finish_request(request=self.__finish_request())
                )

    async def test_finish_request_rejects_non_terminal_target(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            repository = InteractionRepositoryFactory().requests()
            await repository.begin_request(request=begin)

            with pytest.raises(InteractionError, match="not terminal"):
                await repository.finish_request(
                    request=self.__finish_request(state=IdempotencyState.STARTED)
                )

    async def test_get_idempotency_loads_existing_record(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            await InteractionRepositoryFactory().requests().begin_request(request=begin)

            result = (
                await InteractionRepositoryFactory()
                .requests()
                .get_idempotency(query=IdempotencyQuery(tenant=begin.tenant, key=begin.key))
            )

            assert result is not None
            assert result.key == begin.key

    async def test_get_idempotency_ignores_soft_deleted_record(self) -> None:
        """
        Hide soft-deleted idempotency records from replay lookup.
        """

        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            await InteractionRepositoryFactory().requests().begin_request(request=begin)
            await RequestRecord.filter(tenant_id=begin.tenant, key=begin.key).update(
                deleted_at=datetime.now(tz=timezone.utc)
            )

            result = (
                await InteractionRepositoryFactory()
                .requests()
                .get_idempotency(query=IdempotencyQuery(tenant=begin.tenant, key=begin.key))
            )

            assert result is None

    async def test_corrupt_state_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_request_repository"):
            begin = self.__begin_request()
            await InteractionRepositoryFactory().requests().begin_request(request=begin)
            with pytest.raises(IntegrityError):
                await RequestRecord.filter(tenant_id=begin.tenant, key=begin.key).update(
                    state="unknown"
                )

    def __begin_request(
        self,
        *,
        created: Optional[datetime] = None,
        expires: Optional[datetime] = None,
        hash_value: str = "hash-a",
    ) -> BeginRequest:
        """
        Build one begin-idempotency request.
        """

        started = created or datetime.now(tz=timezone.utc)
        return BeginRequest(
            tenant="tenant-a",
            key="request-key",
            hash=hash_value,
            created_at=started,
            expires_at=expires or started + timedelta(minutes=5),
            metadata=Metadata(entries={"source": "test"}),
        )

    def __finish_request(
        self,
        *,
        state: IdempotencyState = IdempotencyState.COMPLETED,
    ) -> FinishRequest:
        """
        Build one finish-idempotency request.
        """

        return FinishRequest(
            tenant="tenant-a",
            key="request-key",
            state=state,
            response={"id": "result"},
            finished=datetime.now(tz=timezone.utc),
        )
