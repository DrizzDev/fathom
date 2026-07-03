from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import PolicyScope
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import PolicyRecord
from fathom.interaction.lifecycle import Lifecycle
from fathom.schemas.interaction import (
    Governance,
    Identity,
    Metadata,
    Policy,
    PolicyQuery,
    SavePolicy,
    Timing,
    Visibility,
)

if TYPE_CHECKING:
    from fathom.infrastructure.interaction.orm.repositories.lifecycle import (
        DatabaseConnection,
        TransactionScope,
    )


class PolicyRepository:
    """
    Repository for tenant and workspace governance policies.
    """

    def __init__(self, *, lifecycle: Lifecycle, transaction: "TransactionScope") -> None:
        """
        Initialize policy validation collaborators.
        """

        self.__lifecycle = lifecycle
        self.__transaction = transaction

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Persist one policy or replay an identical existing policy.
        """

        self.__lifecycle.validate_policy_scope(
            scope=request.scope,
            workspace=request.identity.workspace,
        )
        try:
            return await self.__save_policy(request=request)
        except IntegrityError as exception:
            existing = await self.__load_policy(
                connection=None,
                policy=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None and self.__same_policy(policy=existing, request=request):
                return existing

            named = await self.__policy_by_name(
                connection=None,
                name=request.name,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
            )
            if named is not None:
                raise InteractionError("Policy name already exists for this scope.") from exception

            raise InteractionError("Policy insert conflicted with a different row.") from exception

    async def __save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Persist one policy inside one transaction.
        """

        async with self.__transaction.transaction() as connection:
            existing = await self.__load_policy(
                connection=connection,
                policy=request.identity.id,
                tenant=request.identity.tenant,
            )
            if existing is not None:
                if not self.__same_policy(policy=existing, request=request):
                    raise InteractionError("Policy identity already exists with different content.")

                return existing

            named = await self.__policy_by_name(
                name=request.name,
                connection=connection,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
            )
            if named is not None:
                raise InteractionError("Policy name already exists for this scope.")

            await PolicyRecord.create(
                using_db=connection,
                name=request.name,
                region=request.region,
                id=request.identity.id,
                scope=request.scope.value,
                created_at=request.created,
                metadata=request.metadata.entries,
                tenant_id=request.identity.tenant,
                workspace_id=request.identity.workspace,
                labels=request.governance.labels.entries,
                memories=request.governance.memories.entries,
                artifacts=request.governance.artifacts.entries,
                retention=request.governance.retention.entries,
                sanitizers=request.governance.sanitizers.entries,
            )
            policy = await self.__load_policy(
                connection=connection,
                policy=request.identity.id,
                tenant=request.identity.tenant,
            )
            if policy is None:
                raise InteractionError("Policy was not persisted.")

            return policy

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one policy by tenant, workspace, and name.
        """

        return await self.__policy_by_name(
            name=query.name,
            connection=None,
            tenant=query.tenant,
            workspace=query.workspace,
        )

    async def __load_policy(
        self,
        *,
        tenant: str,
        policy: str,
        connection: "Optional[DatabaseConnection]",
    ) -> Optional[Policy]:
        """
        Load one policy row by identity.
        """

        queryset = PolicyRecord.filter(
            tenant_id=tenant,
            id=policy,
            **Visibility(archived=True).as_filters(),
        )

        if connection is not None:
            queryset = queryset.using_db(connection)

        if row := await queryset.get_or_none():
            return self.__policy(row=row)

        return None

    async def __policy_by_name(
        self,
        *,
        name: str,
        tenant: str,
        workspace: Optional[str],
        connection: "Optional[DatabaseConnection]",
    ) -> Optional[Policy]:
        """
        Load one policy row by scoped name.
        """

        queryset = PolicyRecord.filter(
            tenant_id=tenant,
            name=name,
            **Visibility(archived=True).as_filters(),
        )

        if workspace is None:
            queryset = queryset.filter(workspace_id__isnull=True)
        else:
            queryset = queryset.filter(workspace_id=workspace)

        if connection is not None:
            queryset = queryset.using_db(connection)

        if row := await queryset.get_or_none():
            return self.__policy(row=row)

        return None

    def __policy(self, *, row: PolicyRecord) -> Policy:
        """
        Convert one policy row into the interaction schema.
        """

        return Policy(
            scope=self.__scope(value=row.scope),
            name=row.name,
            region=row.region,
            metadata=self.__metadata(value=row.metadata, field="metadata"),
            timing=Timing(created_at=row.created_at, updated_at=row.updated_at),
            identity=Identity(id=row.id, tenant=row.tenant_id, workspace=row.workspace_id),
            governance=Governance(
                labels=self.__metadata(value=row.labels, field="labels"),
                memories=self.__metadata(value=row.memories, field="memories"),
                artifacts=self.__metadata(value=row.artifacts, field="artifacts"),
                retention=self.__metadata(value=row.retention, field="retention"),
                sanitizers=self.__metadata(value=row.sanitizers, field="sanitizers"),
            ),
        )

    def __metadata(self, *, value: JsonValue, field: str) -> Metadata:
        """
        Validate one stored JSON object as metadata.
        """

        if not isinstance(value, dict):
            raise InteractionError(f"Stored policy {field} is not an object.")

        if not all(isinstance(key, str) for key in value):
            raise InteractionError(f"Stored policy {field} contains a non-string key.")

        return Metadata(entries=value)

    def __scope(self, *, value: str) -> PolicyScope:
        """
        Convert a stored policy scope into an enum.
        """

        try:
            return PolicyScope(value)
        except ValueError as exception:
            raise InteractionError(f"Unknown policy scope in row: {value}.") from exception

    def __same_policy(self, *, policy: Policy, request: SavePolicy) -> bool:
        """
        Check whether a save request replays an existing policy.
        """

        return (
            policy.name == request.name
            and policy.scope == request.scope
            and policy.region == request.region
            and policy.identity == request.identity
            and policy.metadata == request.metadata
            and policy.governance == request.governance
            and policy.timing.created == request.created
        )
