from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import pytest
from tests.unit.infrastructure.interaction.orm.repositories.factories import (
    InteractionRepositoryFactory,
)
from tests.unit.infrastructure.interaction.orm.support import InteractionPostgresSchema
from tortoise.exceptions import IntegrityError

from fathom.constants.collaboration import PolicyScope
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.models import PolicyRecord
from fathom.schemas.interaction import (
    Governance,
    Identity,
    Metadata,
    PolicyQuery,
    SavePolicy,
)


class TestPolicyRepository:
    """
    Verify governance policy persistence through the persistent-store backed repository.
    """

    async def test_save_policy_persists_tenant_policy(self) -> None:
        """
        Save one tenant-scoped policy.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            request = self.__request(scope=PolicyScope.TENANT, workspace=None)

            result = await InteractionRepositoryFactory().policies().save_policy(request=request)

            assert result.identity == request.identity
            assert result.scope == PolicyScope.TENANT
            assert result.name == "default"
            assert result.governance == request.governance
            assert result.timing.created == request.created

    async def test_identical_replay_returns_existing_policy(self) -> None:
        """
        Replay an identical policy save without duplicating rows.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            request = self.__request(scope=PolicyScope.TENANT, workspace=None)
            repository = InteractionRepositoryFactory().policies()

            created = await repository.save_policy(request=request)
            replayed = await repository.save_policy(request=request)

            assert replayed == created
            assert await PolicyRecord.filter(tenant_id="tenant-a").count() == 1

    async def test_conflicting_replay_raises_interaction_error(self) -> None:
        """
        Reject a reused policy id with different content.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            request = self.__request(scope=PolicyScope.TENANT, workspace=None)
            repository = InteractionRepositoryFactory().policies()
            await repository.save_policy(request=request)
            conflict = request.model_copy(update={"region": "eu"})

            with pytest.raises(InteractionError, match="different content"):
                await repository.save_policy(request=conflict)

    async def test_duplicate_scoped_name_raises_interaction_error(self) -> None:
        """
        Reject duplicate names inside the same tenant/workspace scope.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            repository = InteractionRepositoryFactory().policies()
            first = self.__request(scope=PolicyScope.WORKSPACE, workspace="workspace-a")
            second = self.__request(scope=PolicyScope.WORKSPACE, workspace="workspace-a")
            await repository.save_policy(request=first)

            with pytest.raises(InteractionError, match="Policy name already exists"):
                await repository.save_policy(request=second)

    async def test_same_name_allowed_in_different_workspace_scope(self) -> None:
        """
        Allow equal policy names in different workspace scopes.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            repository = InteractionRepositoryFactory().policies()
            first = await repository.save_policy(
                request=self.__request(scope=PolicyScope.WORKSPACE, workspace="workspace-a")
            )
            second = await repository.save_policy(
                request=self.__request(scope=PolicyScope.WORKSPACE, workspace="workspace-b")
            )

            assert first.name == second.name
            assert first.identity.workspace == "workspace-a"
            assert second.identity.workspace == "workspace-b"

    async def test_policy_scope_validation(self) -> None:
        """
        Enforce tenant/workspace scope invariants before persistence.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            with pytest.raises(InteractionError, match="Workspace policy must include"):
                await (
                    InteractionRepositoryFactory()
                    .policies()
                    .save_policy(
                        request=self.__request(scope=PolicyScope.WORKSPACE, workspace=None)
                    )
                )
            with pytest.raises(InteractionError, match="Tenant policy must not include"):
                await (
                    InteractionRepositoryFactory()
                    .policies()
                    .save_policy(
                        request=self.__request(scope=PolicyScope.TENANT, workspace="workspace-a")
                    )
                )

    async def test_get_policy_loads_by_scope_name(self) -> None:
        """
        Load tenant and workspace policies by scoped name.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            repository = InteractionRepositoryFactory().policies()
            tenant_policy = await repository.save_policy(
                request=self.__request(scope=PolicyScope.TENANT, workspace=None)
            )
            workspace_policy = await repository.save_policy(
                request=self.__request(scope=PolicyScope.WORKSPACE, workspace="workspace-a")
            )

            tenant_result = await repository.get_policy(
                query=PolicyQuery(tenant="tenant-a", workspace=None, name="default")
            )
            workspace_result = await repository.get_policy(
                query=PolicyQuery(tenant="tenant-a", workspace="workspace-a", name="default")
            )
            missing = await repository.get_policy(
                query=PolicyQuery(tenant="tenant-a", workspace="missing", name="default")
            )

            assert tenant_result == tenant_policy
            assert workspace_result == workspace_policy
            assert missing is None

    async def test_get_policy_ignores_soft_deleted_policy(self) -> None:
        """
        Hide soft-deleted policies from scoped name lookup.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            repository = InteractionRepositoryFactory().policies()
            policy = await repository.save_policy(
                request=self.__request(scope=PolicyScope.TENANT, workspace=None)
            )
            await PolicyRecord.filter(id=policy.identity.id).update(deleted_at=self.__now())

            result = await repository.get_policy(
                query=PolicyQuery(tenant="tenant-a", workspace=None, name="default")
            )

            assert result is None

    async def test_corrupt_policy_row_raises_interaction_error(self) -> None:
        """
        Reject stored rows with unknown scope or invalid governance JSON.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            with pytest.raises(IntegrityError):
                await PolicyRecord.create(
                    id=str(uuid4()),
                    tenant_id="tenant-a",
                    workspace_id=None,
                    scope="alien",
                    name="default",
                    region=None,
                    retention={},
                    labels={},
                    sanitizers={},
                    memories={},
                    artifacts={},
                    created_at=self.__now(),
                    updated_at=self.__now(),
                    metadata={},
                )

    async def test_private_policy_ids_are_plain_uuid_strings(self) -> None:
        """
        Preserve plain UUID policy identifiers in storage.
        """

        async with InteractionPostgresSchema(prefix="conversation_policy_repository"):
            request = self.__request(scope=PolicyScope.TENANT, workspace=None)
            await InteractionRepositoryFactory().policies().save_policy(request=request)
            stored = await PolicyRecord.get(id=request.identity.id)

            assert str(UUID(stored.id)) == request.identity.id

    def __request(self, *, scope: PolicyScope, workspace: Optional[str]) -> SavePolicy:
        """
        Build one policy save request with a plain UUID identity.
        """

        return SavePolicy(
            identity=Identity(id=str(uuid4()), tenant="tenant-a", workspace=workspace),
            scope=scope,
            name="default",
            region="ap-south",
            governance=Governance(
                retention=Metadata(entries={"messages": "30d"}),
                labels=Metadata(entries={"privacy": ["mask"]}),
                sanitizers=Metadata(entries={"default": "redact"}),
                memories=Metadata(entries={"semantic": True}),
                artifacts=Metadata(entries={"screenshots": "short"}),
            ),
            created_at=self.__now(),
            metadata=Metadata(entries={"source": "test"}),
        )

    def __now(self) -> datetime:
        """
        Return a stable timezone-aware timestamp for tests.
        """

        return datetime(2026, 1, 1, tzinfo=timezone.utc)
