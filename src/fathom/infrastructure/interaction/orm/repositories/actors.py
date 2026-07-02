from __future__ import annotations

from typing import Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import ActorKind
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import ActorRecord
from fathom.schemas.interaction import Actor, CreateActor, Identity, Metadata, Runtime, Timing


class ActorRepository:
    """
    persistent-store backed repository for actor identities.
    """

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Persist one actor identity or replay an identical existing actor.
        """

        if existing := await self.__load_actor(request=request):
            return self.__replay(actor=existing, request=request)

        try:
            await ActorRecord.create(
                name=request.name,
                id=request.identity.id,
                kind=request.kind.value,
                external=request.external,
                created_at=request.created,
                model=request.runtime.model,
                runtime=request.runtime.kind,
                skills=request.skills.entries,
                created_by=request.identity.id,
                updated_by=request.identity.id,
                provider=request.runtime.provider,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
            )
        except IntegrityError:
            if existing := await self.__load_actor(request=request):
                return self.__replay(actor=existing, request=request)

            raise

        if created := await self.__load_actor(request=request):
            return created

        raise InteractionError("Actor was not persisted.")

    async def __load_actor(self, *, request: CreateActor) -> Optional[Actor]:
        """
        Load one actor by tenant-scoped identity.
        """

        row = await ActorRecord.get_or_none(
            id=request.identity.id,
            tenant_id=request.identity.tenant,
        )
        if row is None:
            return None

        return self.__actor(row=row)

    def __replay(self, *, actor: Actor, request: CreateActor) -> Actor:
        """
        Return identical replay rows and reject conflicting identity reuse.
        """

        if self.__same_actor(actor=actor, request=request):
            return actor

        raise InteractionError("Actor identity already exists with different content.")

    def __same_actor(self, *, actor: Actor, request: CreateActor) -> bool:
        """
        Check whether an actor request matches an already stored actor.
        """

        return (
            actor.kind == request.kind
            and actor.name == request.name
            and actor.skills == request.skills
            and actor.runtime == request.runtime
            and actor.external == request.external
            and actor.metadata == request.metadata
            and actor.identity.tenant == request.identity.tenant
            and actor.identity.workspace == request.identity.workspace
        )

    def __actor(self, *, row: ActorRecord) -> Actor:
        """
        Convert one persistent actor model into the interaction schema.
        """

        return Actor(
            name=row.name,
            external=row.external,
            runtime=Runtime(
                model=row.model,
                kind=row.runtime,
                provider=row.provider,
            ),
            kind=self.__kind(value=row.kind),
            skills=self.__metadata(value=row.skills, field="skills"),
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            timing=Timing(created_at=row.created_at, updated_at=row.updated_at),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __kind(self, *, value: str) -> ActorKind:
        """
        Convert stored actor kind text into the public enum.
        """

        try:
            return ActorKind(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown actor kind in row: {value}.") from exception

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError(f"Invalid actor {field} metadata in row.")
