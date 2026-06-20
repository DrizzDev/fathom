from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fathom.constants.collaboration import ActorKind
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres.repositories.actors import (
    PostgresActorRepository,
)
from fathom.schemas.interaction import (
    Actor,
    CreateActor,
    Identity,
    Metadata,
    Runtime,
    Timing,
)
from tests.unit.infrastructure.interaction.pypika.postgres.repositories._fakes import (
    FakeConnection,
    FakePostgresContext,
)


class _ActorRepositoryTestBase(unittest.IsolatedAsyncioTestCase):
    """
    Shared scaffolding for Postgres actor repository tests.
    """

    def _build(
        self,
        *,
        load_returns: tuple,
    ) -> tuple:
        """
        Construct the repo wired to a recording connection.
        """

        connection = FakeConnection()
        context = FakePostgresContext(connection=connection)
        self.__load_queue = list(load_returns)

        async def scripted_load_actor(*, connection, tenant, actor):  # noqa: ARG001
            """
            Return the next configured actor lookup from the queue.

            connection / tenant / actor are accepted to satisfy the bound
            method signature on FakePostgresContext but are not inspected;
            the queue alone drives the returned value.
            """

            return self.__load_queue.pop(0) if self.__load_queue else None

        context._load_actor = scripted_load_actor  # type: ignore[assignment]
        return PostgresActorRepository(context=context), connection, context

    def _request(
        self,
        *,
        actor_id: str = "actor-1",
        tenant: str = "tenant-1",
        workspace: str | None = None,
        kind: ActorKind = ActorKind.HUMAN,
        name: str = "alice",
        external: str | None = None,
        runtime: Runtime | None = None,
        skills: Metadata | None = None,
        metadata: Metadata | None = None,
        created: datetime | None = None,
    ) -> CreateActor:
        """
        Build a CreateActor request with deterministic defaults.
        """

        return CreateActor(
            identity=Identity(id=actor_id, tenant=tenant, workspace=workspace),
            kind=kind,
            name=name,
            external=external,
            runtime=runtime if runtime is not None else Runtime(),
            skills=skills if skills is not None else Metadata(),
            created_at=created
            if created is not None
            else datetime(2026, 1, 1, tzinfo=timezone.utc),
            metadata=metadata if metadata is not None else Metadata(),
        )

    def _actor(
        self,
        *,
        actor_id: str = "actor-1",
        tenant: str = "tenant-1",
        workspace: str | None = None,
        kind: ActorKind = ActorKind.HUMAN,
        name: str = "alice",
        external: str | None = None,
        runtime: Runtime | None = None,
        skills: Metadata | None = None,
        metadata: Metadata | None = None,
        created: datetime | None = None,
    ) -> Actor:
        """
        Build a deterministic Actor model for stubbed lookups.
        """

        moment = created if created is not None else datetime(2026, 1, 1, tzinfo=timezone.utc)
        return Actor(
            identity=Identity(id=actor_id, tenant=tenant, workspace=workspace),
            kind=kind,
            name=name,
            external=external,
            runtime=runtime if runtime is not None else Runtime(),
            skills=skills if skills is not None else Metadata(),
            timing=Timing(created_at=moment, updated_at=moment),
            metadata=metadata if metadata is not None else Metadata(),
        )


class TestCreateActorInsertsNewIdentity(_ActorRepositoryTestBase):
    """
    Verifies the INSERT path when the actor identity does not exist yet.
    """

    async def test_insert_runs_with_all_columns_bound_in_schema_order(self) -> None:
        """
        Single INSERT emitted; placeholders dense from $1..$13; values bound
        in column order matching the schema.
        """

        request = self._request(
            actor_id="actor-new",
            tenant="tenant-9",
            workspace="ws-1",
            kind=ActorKind.AGENT,
            name="rosie",
            external="ext-123",
            runtime=Runtime(kind="adb", provider="genymotion", model="pixel-7"),
            skills=Metadata(entries={"discovery": "ui"}),
            metadata=Metadata(entries={"team": "growth"}),
        )
        repo, connection, context = self._build(
            load_returns=(
                None,
                self._actor(
                    actor_id="actor-new",
                    tenant="tenant-9",
                    workspace="ws-1",
                    kind=ActorKind.AGENT,
                    name="rosie",
                    external="ext-123",
                    runtime=Runtime(kind="adb", provider="genymotion", model="pixel-7"),
                    skills=Metadata(entries={"discovery": "ui"}),
                    metadata=Metadata(entries={"team": "growth"}),
                ),
            )
        )

        result = await repo.create_actor(request=request)

        self.assertEqual(len(connection.calls), 1)
        sql, parameters = connection.calls[0]
        self.assertIn("INSERT INTO actors", sql)
        for column in (
            "id",
            "tenant",
            "workspace",
            "kind",
            "name",
            "external",
            "runtime",
            "provider",
            "model",
            "skills",
            "created_at",
            "updated_at",
            "metadata",
        ):
            self.assertIn(column, sql)
        for index in range(1, 14):
            self.assertIn(f"${index}", sql)
        self.assertEqual(len(parameters), 13)
        self.assertEqual(parameters[0], "actor-new")
        self.assertEqual(parameters[1], "tenant-9")
        self.assertEqual(parameters[2], "ws-1")
        self.assertEqual(parameters[3], ActorKind.AGENT.value)
        self.assertEqual(parameters[4], "rosie")
        self.assertEqual(parameters[5], "ext-123")
        self.assertEqual(parameters[6], "adb")
        self.assertEqual(parameters[7], "genymotion")
        self.assertEqual(parameters[8], "pixel-7")
        self.assertEqual(parameters[9], {"discovery": "ui"})
        self.assertEqual(parameters[10], request.created)
        self.assertEqual(parameters[11], request.created)
        self.assertEqual(parameters[12], {"team": "growth"})
        self.assertEqual(result.identity.id, "actor-new")

    async def test_kind_bound_as_enum_value_string(self) -> None:
        """
        ActorKind enum is bound via `.value` so asyncpg sees plain text.
        """

        request = self._request(kind=ActorKind.COORDINATOR)
        repo, connection, _ = self._build(
            load_returns=(
                None,
                self._actor(kind=ActorKind.COORDINATOR),
            )
        )

        await repo.create_actor(request=request)
        _sql, parameters = connection.calls[0]

        self.assertEqual(parameters[3], "coordinator")
        self.assertNotIsInstance(parameters[3], ActorKind)


class TestCreateActorIdempotentReplay(_ActorRepositoryTestBase):
    """
    Verifies replay semantics when the same identity is created twice.
    """

    async def test_identical_replay_returns_existing_without_insert(self) -> None:
        """
        Second create with identical content short-circuits to the loaded row.
        """

        request = self._request()
        existing = self._actor()
        repo, connection, _ = self._build(load_returns=(existing,))

        result = await repo.create_actor(request=request)

        self.assertEqual(connection.calls, [])
        self.assertIs(result, existing)

    async def test_differing_kind_raises_interaction_error(self) -> None:
        """
        Replay with a different actor kind is rejected as a content conflict.
        """

        request = self._request(kind=ActorKind.HUMAN)
        existing = self._actor(kind=ActorKind.AGENT)
        repo, connection, _ = self._build(load_returns=(existing,))

        with self.assertRaises(InteractionError):
            await repo.create_actor(request=request)
        self.assertEqual(connection.calls, [])

    async def test_differing_name_raises_interaction_error(self) -> None:
        """
        Replay with a different display name is rejected as a content conflict.
        """

        request = self._request(name="alice")
        existing = self._actor(name="bob")
        repo, _connection, _ = self._build(load_returns=(existing,))

        with self.assertRaises(InteractionError):
            await repo.create_actor(request=request)

    async def test_differing_runtime_raises_interaction_error(self) -> None:
        """
        Replay with a different runtime (provider/model/kind) is rejected.
        """

        request = self._request(runtime=Runtime(kind="adb", provider="genymotion", model="pixel-7"))
        existing = self._actor(runtime=Runtime(kind="adb", provider="genymotion", model="pixel-8"))
        repo, _connection, _ = self._build(load_returns=(existing,))

        with self.assertRaises(InteractionError):
            await repo.create_actor(request=request)

    async def test_differing_external_raises_interaction_error(self) -> None:
        """
        Replay with a different external system reference is rejected.
        """

        request = self._request(external="ext-1")
        existing = self._actor(external="ext-2")
        repo, _connection, _ = self._build(load_returns=(existing,))

        with self.assertRaises(InteractionError):
            await repo.create_actor(request=request)

    async def test_differing_metadata_raises_interaction_error(self) -> None:
        """
        Replay with different metadata payload is rejected.
        """

        request = self._request(metadata=Metadata(entries={"a": "1"}))
        existing = self._actor(metadata=Metadata(entries={"a": "2"}))
        repo, _connection, _ = self._build(load_returns=(existing,))

        with self.assertRaises(InteractionError):
            await repo.create_actor(request=request)


class TestCreateActorPersistenceFailure(_ActorRepositoryTestBase):
    """
    Covers the error path where the post-insert reload returns None.
    """

    async def test_missing_post_insert_load_raises_interaction_error(self) -> None:
        """
        When the row vanishes between insert and reload the repo raises.
        """

        request = self._request()
        repo, _connection, _ = self._build(load_returns=(None, None))

        with self.assertRaises(InteractionError):
            await repo.create_actor(request=request)


if __name__ == "__main__":
    unittest.main()
