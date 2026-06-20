from __future__ import annotations

from pypika import SQLLiteQuery

from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
from fathom.infrastructure.interaction.pypika.sqlite import tables
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.schemas.interaction import (
    Actor,
    CreateActor,
    Timing,
)


class ActorRepository:
    """
    Actor repository: persists and replays actor identities.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for actor persistence.
        """

        self.__context = context

    async def create_actor(self, *, request: CreateActor) -> Actor:
        """
        Persist one actor identity.
        """

        timing = Timing(created_at=request.created, updated_at=request.created)

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_actor(
                connection=connection,
                actor=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None:
                if not self.__same_actor(actor=existing, request=request):
                    raise InteractionError("Actor identity already exists with different content.")

                return existing

            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.QUESTION_MARK)
            actors = tables.ACTORS
            statement = (
                SQLLiteQuery.into(actors)
                .columns(
                    actors.id,
                    actors.tenant,
                    actors.workspace,
                    actors.kind,
                    actors.name,
                    actors.external,
                    actors.runtime,
                    actors.provider,
                    actors.model,
                    actors.skills,
                    actors.created_at,
                    actors.updated_at,
                    actors.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.kind.value),
                    binder.bind(value=request.name),
                    binder.bind(value=request.external),
                    binder.bind(value=request.runtime.kind),
                    binder.bind(value=request.runtime.provider),
                    binder.bind(value=request.runtime.model),
                    binder.bind(value=self.__context._json(value=request.skills.entries)),
                    binder.bind(value=self.__context._time(value=timing.created)),
                    binder.bind(value=self.__context._time(value=timing.updated)),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            actor = await self.__context._load_actor(
                connection=connection,
                actor=request.identity.id,
                tenant=request.identity.tenant,
            )

        if actor is None:
            raise InteractionError("Actor was not persisted.")

        return actor

    def __same_actor(self, *, actor: Actor, request: CreateActor) -> bool:
        """
        Check whether an actor request replays an already stored actor.
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
