from __future__ import annotations

from pypika import PostgreSQLQuery

from fathom.constants.collaboration import EventKind, EventSource
from fathom.constants.storage import SqlParameterStyle
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres import tables
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import PostgresContext
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery
from fathom.schemas.interaction import (
    JoinThread,
    Membership,
    Metadata,
)


class PostgresMembershipRepository:
    """
    Postgres membership repository: tracks active actor memberships in threads.
    """

    def __init__(self, *, context: PostgresContext) -> None:
        """
        Bind shared Postgres context for membership persistence.
        """

        self.__context = context

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one active actor membership in a thread.
        """

        async with self.__context.unit.session() as connection:
            existing = await self.__context._load_membership(
                connection=connection,
                tenant=request.identity.tenant,
                membership=request.identity.id,
            )
            if existing is not None:
                if not self.__same_membership(membership=existing, request=request):
                    raise InteractionError(
                        "Membership identity already exists with different content."
                    )

                return existing

            active = await self.__context.find_active_membership(
                actor=request.actor,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            if active is not None:
                return active

            await self.__context._require_thread(
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            await self.__context._require_actor(
                actor=request.actor,
                connection=connection,
                tenant=request.identity.tenant,
            )
            binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
            memberships = tables.MEMBERSHIPS
            statement = (
                PostgreSQLQuery.into(memberships)
                .columns(
                    memberships.id,
                    memberships.tenant,
                    memberships.workspace,
                    memberships.thread,
                    memberships.actor,
                    memberships.role,
                    memberships.scope,
                    memberships.joined_at,
                    memberships.departed_at,
                    memberships.metadata,
                )
                .insert(
                    binder.bind(value=request.identity.id),
                    binder.bind(value=request.identity.tenant),
                    binder.bind(value=request.identity.workspace),
                    binder.bind(value=request.thread),
                    binder.bind(value=request.actor),
                    binder.bind(value=request.role.value),
                    binder.bind(value=request.scope.value),
                    binder.bind(value=self.__context._time(value=request.joined)),
                    binder.bind(value=None),
                    binder.bind(value=self.__context._json(value=request.metadata.entries)),
                )
            )
            sql, parameters = binder.render(query=statement)
            await connection.execute(sql, parameters)

            membership = await self.__context._load_membership(
                connection=connection,
                tenant=request.identity.tenant,
                membership=request.identity.id,
            )
            await self.__context._record_event(
                task=None,
                actor=request.actor,
                connection=connection,
                thread=request.thread,
                created=request.joined,
                subject=request.identity.id,
                kind=EventKind.ACTOR_JOINED,
                source=EventSource.INTERACTION,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                payload=Metadata(
                    entries={"role": request.role.value, "scope": request.scope.value}
                ),
            )

        if membership is None:
            raise InteractionError("Membership was not persisted.")

        return membership

    def __same_membership(self, *, membership: Membership, request: JoinThread) -> bool:
        """
        Check whether a membership request replays an already stored membership.
        """

        return (
            membership.role == request.role
            and membership.actor == request.actor
            and membership.scope == request.scope
            and membership.thread == request.thread
            and membership.metadata == request.metadata
            and membership.identity.tenant == request.identity.tenant
            and membership.identity.workspace == request.identity.workspace
        )
