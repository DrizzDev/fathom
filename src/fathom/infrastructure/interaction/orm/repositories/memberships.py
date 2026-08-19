from __future__ import annotations

from typing import Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import EventKind, MembershipRole, MembershipScope
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import (
    ActorRecord,
    ConversationRecord,
    MembershipRecord,
)
from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
    DatabaseConnection,
    LifecycleRecorder,
    TransactionScope,
)
from fathom.schemas.interaction import (
    Identity,
    JoinThread,
    Membership,
    MembershipQuery,
    MembershipVisibility,
    Metadata,
    Visibility,
)


class MembershipRepository:
    """
    Persistent-store backed repository for actor membership in conversation threads.
    """

    def __init__(self, *, lifecycle: LifecycleRecorder, transaction: TransactionScope) -> None:
        """
        Initialize membership persistence collaborators.
        """

        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one active actor membership or replay an existing one.
        """

        try:
            return await self.__join_thread(request=request)
        except IntegrityError as exception:
            if (
                existing := await self.__load_membership(
                    connection=None,
                    tenant=request.identity.tenant,
                    membership=request.identity.id,
                )
            ) and self.__same_membership(membership=existing, request=request):
                return existing

            if active := await self.__find_active_membership(
                connection=None,
                actor=request.actor,
                thread=request.thread,
                tenant=request.identity.tenant,
            ):
                return active

            raise InteractionError(
                "Membership insert conflicted with a different row."
            ) from exception

    async def find_membership(self, *, query: MembershipQuery) -> Optional[Membership]:
        """
        Load one active actor membership.
        """

        return await self.__find_active_membership(
            connection=None,
            actor=query.actor,
            tenant=query.tenant,
            thread=query.thread,
        )

    async def __join_thread(self, *, request: JoinThread) -> Membership:
        """
        Persist one membership inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            if existing := await self.__load_membership(
                connection=connection,
                tenant=request.identity.tenant,
                membership=request.identity.id,
            ):
                if not self.__same_membership(membership=existing, request=request):
                    raise InteractionError(
                        "Membership identity already exists with different content."
                    )

                return existing

            if active := await self.__find_active_membership(
                actor=request.actor,
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            ):
                return active

            await self.__require_thread(
                thread=request.thread,
                connection=connection,
                tenant=request.identity.tenant,
            )
            await self.__require_actor(
                actor=request.actor,
                connection=connection,
                tenant=request.identity.tenant,
            )

            await MembershipRecord.create(
                using_db=connection,
                actor=request.actor,
                id=request.identity.id,
                role=request.role.value,
                joined_at=request.joined,
                updated_by=request.actor,
                scope=request.scope.value,
                created_at=request.joined,
                created_by=request.actor,
                conversation_id=request.thread,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
            )

            membership = await self.__load_membership(
                connection=connection,
                tenant=request.identity.tenant,
                membership=request.identity.id,
            )
            if membership is None:
                raise InteractionError("Membership was not persisted.")

            await self.__lifecycle.record(
                actor=request.actor,
                connection=connection,
                thread=request.thread,
                created=request.joined,
                kind=EventKind.ACTOR_JOINED,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                payload=Metadata(
                    entries={"role": request.role.value, "scope": request.scope.value}
                ),
            )

            return membership

    async def __load_membership(
        self,
        *,
        tenant: str,
        membership: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Membership]:
        """
        Load one membership by identity.
        """

        queryset = MembershipRecord.filter(
            tenant_id=tenant,
            id=membership,
            **MembershipVisibility().as_filters(),
        )

        if connection is not None:
            queryset = queryset.using_db(connection)

        if row := await queryset.get_or_none():
            return self.__membership(row=row)

        return None

    async def __find_active_membership(
        self,
        *,
        actor: str,
        tenant: str,
        thread: str,
        connection: Optional[DatabaseConnection],
    ) -> Optional[Membership]:
        """
        Load the active membership for one thread and actor.
        """

        queryset = MembershipRecord.filter(
            tenant_id=tenant,
            conversation_id=thread,
            actor=actor,
            **MembershipVisibility().as_filters(),
        )
        if connection is not None:
            queryset = queryset.using_db(connection)

        row = await queryset.get_or_none()

        if row is None:
            return None

        return self.__membership(row=row)

    async def __require_thread(
        self,
        *,
        tenant: str,
        thread: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an undeleted thread before a membership references it.
        """

        row = (
            await ConversationRecord.filter(
                tenant_id=tenant,
                id=thread,
                **Visibility(archived=True).as_filters(),
            )
            .using_db(connection)
            .get_or_none()
        )
        if row is None:
            raise InteractionError("Thread does not exist.")

    async def __require_actor(
        self,
        *,
        actor: str,
        tenant: str,
        connection: DatabaseConnection,
    ) -> None:
        """
        Require an actor before a membership references it.
        """

        row = await ActorRecord.get_or_none(id=actor, tenant_id=tenant, using_db=connection)
        if row is None:
            raise InteractionError("Actor does not exist.")

    def __same_membership(self, *, membership: Membership, request: JoinThread) -> bool:
        """
        Check whether a membership request matches an already stored membership.
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

    def __membership(self, *, row: MembershipRecord) -> Membership:
        """
        Convert one persistent membership model into the interaction schema.
        """

        return Membership(
            actor=row.actor,
            joined_at=row.joined_at,
            thread=row.conversation_id,
            departed_at=row.departed_at,
            role=self.__role(value=row.role),
            scope=self.__scope(value=row.scope),
            metadata=self.__metadata(value=row.metadata),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
        )

    def __role(self, *, value: str) -> MembershipRole:
        """
        Convert stored membership role text into the public enum.
        """

        try:
            return MembershipRole(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown membership role in row: {value}.") from exception

    def __scope(self, *, value: str) -> MembershipScope:
        """
        Convert stored membership scope text into the public enum.
        """

        try:
            return MembershipScope(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown membership scope in row: {value}.") from exception

    def __metadata(self, *, value: JsonValue) -> Metadata:
        """
        Convert stored JSON object into metadata.
        """

        if isinstance(value, dict):
            return Metadata(entries=value)

        raise InteractionError("Invalid membership metadata in row.")
