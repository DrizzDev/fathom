from __future__ import annotations

from typing import Optional

import aiosqlite

from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.schemas.interaction import (
    Policy,
    PolicyQuery,
    SavePolicy,
    Timing,
)


class PolicyRepository:
    """
    Policy repository: persists tenant- and workspace-scoped policies.
    """

    def __init__(self, *, context: StoreContext) -> None:
        """
        Bind shared store context for policy persistence.
        """

        self.__context = context

    async def save_policy(self, *, request: SavePolicy) -> Policy:
        """
        Persist one tenant or workspace policy.
        """

        timing = Timing(created_at=request.created, updated_at=request.created)
        async with self.__context.unit.session() as connection:
            self.__context.lifecycle.validate_policy_scope(
                scope=request.scope,
                workspace=request.identity.workspace,
            )
            existing = await self.__context._load_policy(
                connection=connection,
                tenant=request.identity.tenant,
                policy=request.identity.id,
            )
            if existing is not None:
                if not self.__same_policy(policy=existing, request=request):
                    raise InteractionError("Policy identity already exists with different content.")

                return existing

            named = await self.__policy_by_name(
                connection=connection,
                tenant=request.identity.tenant,
                workspace=request.identity.workspace,
                name=request.name,
            )
            if named is not None:
                raise InteractionError("Policy name already exists for this scope.")

            await connection.execute(
                """
                INSERT INTO policies (
                    id, tenant, workspace, scope, name, region, retention, labels,
                    sanitizers, memories, artifacts, created_at, updated_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.identity.id,
                    request.identity.tenant,
                    request.identity.workspace,
                    request.scope.value,
                    request.name,
                    request.region,
                    self.__context._json(value=request.governance.retention.entries),
                    self.__context._json(value=request.governance.labels.entries),
                    self.__context._json(value=request.governance.sanitizers.entries),
                    self.__context._json(value=request.governance.memories.entries),
                    self.__context._json(value=request.governance.artifacts.entries),
                    self.__context._time(value=timing.created),
                    self.__context._time(value=timing.updated),
                    self.__context._json(value=request.metadata.entries),
                ),
            )
            policy = await self.__context._load_policy(
                connection=connection,
                tenant=request.identity.tenant,
                policy=request.identity.id,
            )

        if policy is None:
            raise InteractionError("Policy was not persisted.")

        return policy

    async def get_policy(self, *, query: PolicyQuery) -> Optional[Policy]:
        """
        Load one tenant-scoped policy.
        """

        async with self.__context.unit.session() as connection:
            if query.workspace is None:
                return await self.__tenant_policy(connection=connection, query=query)

            return await self.__workspace_policy(connection=connection, query=query)

    async def __tenant_policy(
        self,
        *,
        connection: aiosqlite.Connection,
        query: PolicyQuery,
    ) -> Optional[Policy]:
        """
        Load one tenant-level policy.
        """

        async with connection.execute(
            """
            SELECT *
            FROM policies
            WHERE tenant = ? AND workspace IS NULL AND name = ?
            LIMIT 1
            """,
            (query.tenant, query.name),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.policy(row=row)

    async def __workspace_policy(
        self,
        *,
        connection: aiosqlite.Connection,
        query: PolicyQuery,
    ) -> Optional[Policy]:
        """
        Load one workspace-level policy.
        """

        async with connection.execute(
            """
            SELECT *
            FROM policies
            WHERE tenant = ? AND workspace = ? AND name = ?
            LIMIT 1
            """,
            (query.tenant, query.workspace, query.name),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self.__context.rows.policy(row=row)

    async def __policy_by_name(
        self,
        *,
        connection: aiosqlite.Connection,
        tenant: str,
        workspace: Optional[str],
        name: str,
    ) -> Optional[Policy]:
        """
        Load one policy row by scoped name.
        """

        query = PolicyQuery(tenant=tenant, workspace=workspace, name=name)
        if workspace is None:
            return await self.__tenant_policy(connection=connection, query=query)

        return await self.__workspace_policy(connection=connection, query=query)

    def __same_policy(self, *, policy: Policy, request: SavePolicy) -> bool:
        """
        Check whether a policy request replays an already stored policy.
        """

        return (
            policy.identity.tenant == request.identity.tenant
            and policy.identity.workspace == request.identity.workspace
            and policy.scope == request.scope
            and policy.name == request.name
            and policy.region == request.region
            and policy.governance == request.governance
            and policy.timing.created == request.created
            and policy.metadata == request.metadata
        )
