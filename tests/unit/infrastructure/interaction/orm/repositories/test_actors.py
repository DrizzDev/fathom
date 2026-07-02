from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import ActorKind
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import ActorRecord
from fathom.infrastructure.interaction.orm.repositories import ActorRepository
from fathom.schemas.interaction import CreateActor, Identity, Metadata, Runtime


class TestActorRepository:
    """
    Verify actor persistence through the persistent-store backed repository.
    """

    async def test_create_actor_persists_and_returns_actor(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_actor_repository"):
            request = self.__request(
                kind=ActorKind.AGENT,
                name="rosie",
                external="ext-123",
                runtime=Runtime(kind="adb", provider="genymotion", model="pixel-7"),
                skills=Metadata(entries={"discovery": "ui"}),
                metadata=Metadata(entries={"team": "growth"}),
            )

            result = await ActorRepository().create_actor(request=request)

            assert result.identity == request.identity
            assert result.kind == ActorKind.AGENT
            assert result.name == "rosie"
            assert result.external == "ext-123"
            assert result.runtime == request.runtime
            assert result.skills == Metadata(entries={"discovery": "ui"})
            assert result.metadata == Metadata(entries={"team": "growth"})
            assert result.timing.created == request.created
            assert result.timing.updated >= result.timing.created
            row = await ActorRecord.get(id=request.identity.id)
            assert row.created_by == request.identity.id
            assert row.updated_by == request.identity.id

    async def test_identical_replay_returns_existing_actor(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_actor_repository"):
            request = self.__request()
            repository = ActorRepository()

            created = await repository.create_actor(request=request)
            replayed = await repository.create_actor(request=request)

            assert replayed == created

    async def test_conflicting_replay_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_actor_repository"):
            request = self.__request(name="alice")
            repository = ActorRepository()
            await repository.create_actor(request=request)
            conflict = request.model_copy(update={"name": "bob"})

            with pytest.raises(InteractionError, match="different content"):
                await repository.create_actor(request=conflict)

    async def test_unknown_actor_kind_raises_interaction_error(self) -> None:
        async with InteractionPostgresSchema(prefix="conversation_actor_repository"):
            request = self.__request()
            with pytest.raises(IntegrityError):
                await ActorRecord.create(
                    id=request.identity.id,
                    tenant_id=request.identity.tenant,
                    workspace_id=request.identity.workspace,
                    kind="alien",
                    name=request.name,
                    external=request.external,
                    runtime=request.runtime.kind,
                    provider=request.runtime.provider,
                    model=request.runtime.model,
                    skills=request.skills.entries,
                    metadata=request.metadata.entries,
                    created_at=request.created,
                    updated_at=request.created,
                )

    def __request(
        self,
        *,
        kind: ActorKind = ActorKind.HUMAN,
        name: str = "operator",
        external: Optional[str] = None,
        runtime: Optional[Runtime] = None,
        skills: Optional[Metadata] = None,
        metadata: Optional[Metadata] = None,
    ) -> CreateActor:
        """
        Build one actor creation request.
        """

        return CreateActor(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace="workspace-a"),
            kind=kind,
            name=name,
            external=external,
            runtime=runtime or Runtime(),
            skills=skills or Metadata(),
            created_at=datetime.now(tz=timezone.utc),
            metadata=metadata or Metadata(),
        )
