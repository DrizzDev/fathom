from __future__ import annotations

import re
from hashlib import sha256
from typing import Dict, List, Mapping, Optional, Tuple, Type

from fathom.infrastructure.interaction.orm.migration import (
    SCHEMA_VERSION,
    BaselineMigration,
    CompositeKeyMigration,
    ConversationStoreMigrations,
    PostgresMigrator,
    PostgresSchemaValidationError,
    PostgresSchemaValidator,
)
from fathom.infrastructure.interaction.orm.models import Catalog


class _FakeTransaction:
    """
    Fake async transaction context for migration tests.
    """

    async def __aenter__(self) -> object:
        """
        Enter the fake transaction.
        """

        return self

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        """
        Exit the fake transaction.
        """


class _FakeConnection:
    """
    Fake Postgres connection for migration-runner tests.
    """

    def __init__(
        self,
        *,
        valid_schema: bool = False,
        applied_checksum: Optional[str] = None,
        applied_checksums: Optional[Mapping[int, str]] = None,
    ) -> None:
        """
        Initialize the fake connection with an optional ledger row.
        """

        self.executed: List[str] = []
        self.inserted: List[Tuple[object, ...]] = []
        self.__valid_schema = valid_schema
        self.__applied_checksums = dict(applied_checksums or {})
        if applied_checksum is not None:
            self.__applied_checksums[BaselineMigration.VERSION] = applied_checksum

    async def execute(self, query: str, *args: object) -> str:
        """
        Record one executed statement.
        """

        self.executed.append(query)
        if query.strip().startswith("INSERT INTO migrations"):
            self.inserted.append(args)
        return "OK"

    async def fetchrow(self, query: str, *args: object) -> Optional[Mapping[str, object]]:
        """
        Return the configured migration ledger row.
        """

        if args and isinstance(args[0], int):
            checksum = self.__applied_checksums.get(args[0])
            if checksum is None:
                return None

            return {"checksum": checksum}

        if not self.__valid_schema:
            return None

        if "current_schema()" in query:
            return {"schema_name": "fathom"}

        if "to_regclass" in query:
            return {"relation": "relation"}

        if "information_schema.columns" in query:
            return {"column_name": args[1]}

        if "pg_constraint" in query:
            return {"conname": args[0]}

        if "pg_proc" in query:
            return {"proname": args[0]}

        if "pg_trigger" in query:
            return {"tgname": args[1]}

        return None

    def transaction(self) -> _FakeTransaction:
        """
        Return a fake transaction context.
        """

        return _FakeTransaction()


class _FakeValidationConnection:
    """
    Fake Postgres connection for schema-validator tests.
    """

    def __init__(
        self,
        *,
        missing_constraint: Optional[str] = None,
        missing_function: Optional[str] = None,
        missing_index: Optional[str] = None,
        missing_trigger: Optional[str] = None,
    ) -> None:
        """
        Store the optional schema object that should appear missing.
        """

        self.__missing_index = missing_index
        self.__missing_trigger = missing_trigger
        self.__missing_function = missing_function
        self.__missing_constraint = missing_constraint

    async def execute(self, query: str, *args: object) -> str:
        """
        Accept unused execute calls for protocol completeness.
        """

        return "OK"

    async def fetchrow(self, query: str, *args: object) -> Optional[Mapping[str, object]]:
        """
        Return rows for required schema objects unless configured missing.
        """

        if "current_schema()" in query and not args:
            return {"schema_name": "fathom"}

        if "to_regclass" in query:
            relation = str(args[0])
            if relation == self.__missing_index:
                return {"relation": None}
            return {"relation": relation}

        if "information_schema.columns" in query:
            return {"column_name": args[1]}

        if "pg_constraint" in query:
            constraint = str(args[0])
            if constraint == self.__missing_constraint:
                return None
            return {"conname": constraint}

        if "information_schema.routines" in query:
            function = str(args[0])
            if function == self.__missing_function:
                return None
            return {"routine_name": function}

        if "information_schema.triggers" in query:
            trigger = str(args[1])
            if trigger == self.__missing_trigger:
                return None
            return {"trigger_name": trigger}

        return None

    def transaction(self) -> _FakeTransaction:
        """
        Return a fake transaction context.
        """

        return _FakeTransaction()


class TestBaselineMigration:
    """
    Verify the final clean baseline before repository cutover.
    """

    def test_baseline_exports_one_versioned_step(self) -> None:
        step = BaselineMigration.step()

        assert step.version == 1
        assert step.name == "baseline"
        assert step.statements == BaselineMigration.statements()

    def test_baseline_creates_every_conversation_table(self) -> None:
        expected_tables = {
            "actors",
            "conversations",
            "executions",
            "memberships",
            "tasks",
            "messages",
            "events",
            "artifacts",
            "scripts",
            "script_versions",
            "policies",
            "contexts",
            "requests",
            "jobs",
            "sequences",
            "search",
        }
        statements = "\n".join(BaselineMigration.statements())
        actual_tables = set(re.findall(r"CREATE TABLE(?: IF NOT EXISTS)? ([a-z_]+) \(", statements))

        assert expected_tables <= actual_tables

    def test_baseline_uses_plain_id_primary_keys_for_entity_tables(self) -> None:
        statements = "\n".join(BaselineMigration.TABLES)

        assert "PRIMARY KEY (tenant_id, id)" not in statements
        assert "PRIMARY KEY (tenant_id, key)" not in statements
        assert "PRIMARY KEY (tenant_id, conversation_id, scope)" not in statements
        assert statements.count("id TEXT PRIMARY KEY") == 16

    def test_baseline_preserves_tenant_scoped_uniqueness(self) -> None:
        statements = "\n".join((*BaselineMigration.TABLES, *BaselineMigration.INDEXES))

        for constraint in (
            "index_memberships_active_actor",
            "UNIQUE (tenant_id, conversation_id, sequence)",
            "UNIQUE (tenant_id, script_id, version)",
            "UNIQUE (tenant_id, key)",
            "UNIQUE (tenant_id, conversation_id, scope)",
            "UNIQUE (tenant_id, conversation_id, source, source_id)",
        ):
            assert constraint in statements

    def test_baseline_uses_partial_timeline_indexes(self) -> None:
        """
        Timeline indexes must ship in the baseline and stay bounded to active rows.
        """

        statements = "\n".join(BaselineMigration.INDEXES)

        for index in (
            "index_messages_timeline_active ON messages(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
            "index_messages_kind_timeline_active ON messages(tenant_id, conversation_id, kind, created_at DESC, id DESC) WHERE deleted_at IS NULL",
            "index_events_timeline_active ON events(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
            "index_artifacts_timeline_active ON artifacts(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
            "index_contexts_timeline_active ON contexts(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
        ):
            assert index in statements

    def test_baseline_preserves_tenant_scoped_foreign_keys(self) -> None:
        statements = "\n".join(
            (
                *BaselineMigration.TABLES,
                *BaselineMigration.STRUCTURAL_CONSTRAINTS,
            )
        )

        for foreign_key in (
            "FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id)",
            "FOREIGN KEY (tenant_id, actor) REFERENCES actors(tenant_id, id)",
            "FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id)",
            "FOREIGN KEY (tenant_id, script_id) REFERENCES scripts(tenant_id, id)",
            "FOREIGN KEY (tenant_id, origin_id)",
        ):
            assert foreign_key in statements

    def test_baseline_keeps_search_support(self) -> None:
        statements = "\n".join(BaselineMigration.statements())

        assert "vector TSVECTOR" in statements
        assert "id TEXT PRIMARY KEY" in self.__table_sql(table="search")
        assert "UNIQUE (tenant_id, conversation_id, source, source_id)" in statements
        assert "INSERT INTO search (" in statements
        assert "conversation_id" in statements
        assert "index_search_vector" in statements
        assert "index_contexts_references_messages" in statements
        assert "index_contexts_references_events" in statements
        assert "index_contexts_references_artifacts" in statements
        assert "index_memberships_active_actor" in statements
        assert "fathom_message_search_document" in statements
        assert "messages_search_insert" in statements

    def test_baseline_uses_idempotent_ddl(self) -> None:
        """
        Baseline DDL must be safe to re-run when schema objects already exist.
        """

        statements = "\n".join(BaselineMigration.statements())

        assert "CREATE TABLE actors (" not in statements
        assert "CREATE INDEX idx_" not in statements
        assert "DROP TRIGGER IF EXISTS messages_search_insert" in statements
        assert "foreign_key_tasks_origin_messages" in statements

    def test_schema_object_names_fit_postgres_identifier_limit(self) -> None:
        """
        Required schema object names must fit PostgreSQL's identifier limit.
        """

        postgres_identifier_limit = 63
        names = {
            *BaselineMigration.REQUIRED_INDEXES,
            *BaselineMigration.REQUIRED_CONSTRAINTS,
            *BaselineMigration.REQUIRED_FUNCTIONS,
        }
        for triggers in BaselineMigration.REQUIRED_TRIGGERS.values():
            names.update(triggers)

        oversized = sorted(
            name for name in names if len(name.encode("utf-8")) > postgres_identifier_limit
        )

        assert oversized == []

    def test_schema_object_names_avoid_abbreviated_prefixes(self) -> None:
        """
        Required schema object names must avoid unclear abbreviated prefixes.
        """

        names = {
            *BaselineMigration.REQUIRED_INDEXES,
            *BaselineMigration.REQUIRED_CONSTRAINTS,
        }

        assert all(not name.startswith(("idx_", "fk_", "chk_")) for name in names)

    def __table_sql(self, *, table: str) -> str:
        """
        Return the baseline CREATE TABLE statement for one table.
        """

        marker = f"CREATE TABLE IF NOT EXISTS {table} ("
        for statement in BaselineMigration.statements():
            if marker in statement:
                return statement

        raise AssertionError(f"Missing table statement for {table}.")

    def test_baseline_preserves_enum_checks(self) -> None:
        """
        Enum-like text columns must retain database CHECK constraints.
        """

        statements = "\n".join(BaselineMigration.statements())

        for expected in (
            "ARRAY['executions','state'",
            "ARRAY['tasks','state'",
            "ARRAY['messages','kind'",
            "ARRAY['events','kind'",
            "ARRAY['jobs','state'",
        ):
            assert expected in statements

    def test_executions_store_workflow_only_as_runtime_correlation(self) -> None:
        """
        Execution rows store the Fathom id separately from runtime workflow correlation.
        """

        table = self.__table_sql(table="executions")

        assert "workflow_id TEXT" in table
        assert "run_id" not in table
        assert "package" not in table
        assert "state TEXT NOT NULL" in table
        assert "outcome JSONB NOT NULL" in table

    def test_run_owned_tables_require_execution_identifier(self) -> None:
        """
        Run-owned tables require execution_id while events may remain conversation-scoped.
        """

        required_tables = (
            "tasks",
            "messages",
            "artifacts",
            "scripts",
            "contexts",
            "jobs",
            "search",
        )

        for table in required_tables:
            assert "execution_id TEXT NOT NULL" in self.__table_sql(table=table)

        assert "execution_id TEXT NOT NULL" not in self.__table_sql(table="events")

    def test_append_only_tables_expose_standard_audit_columns(self) -> None:
        """
        Append-only tables still expose the standard audit column contract.
        """

        for table in ("events", "script_versions"):
            statement = self.__table_sql(table=table)

            assert "created_at TIMESTAMPTZ NOT NULL DEFAULT now()" in statement
            assert "created_by TEXT" in statement
            assert "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()" in statement
            assert "updated_by TEXT" in statement
            assert "deleted_at TIMESTAMPTZ" in statement
            assert "deleted_by TEXT" in statement

    def test_sequence_table_exposes_standard_audit_columns(self) -> None:
        """
        Sequence rows expose the same audit contract as other persisted rows.
        """

        statement = self.__table_sql(table="sequences")

        assert "workspace_id TEXT" in statement
        assert "metadata JSONB NOT NULL DEFAULT '{}'::jsonb" in statement
        assert "created_at TIMESTAMPTZ NOT NULL DEFAULT now()" in statement
        assert "created_by TEXT" in statement
        assert "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()" in statement
        assert "updated_by TEXT" in statement
        assert "deleted_at TIMESTAMPTZ" in statement
        assert "deleted_by TEXT" in statement

    def test_baseline_comments_every_described_model_table_and_column(self) -> None:
        """
        Emit database comments from model table and column descriptions.
        """

        comments = "\n".join(BaselineMigration.comments())

        for model in Catalog().all():
            table = model._meta.db_table
            assert f'COMMENT ON TABLE "{table}" IS ' in comments

            for field_name, field in model._meta.fields_map.items():
                if field.description is None:
                    continue

                column = field.source_field or field_name
                assert f'COMMENT ON COLUMN "{table}"."{column}" IS ' in comments


class TestPostgresMigrator:
    """
    Verify migration ledger behavior.
    """

    async def test_apply_records_pending_versions(self) -> None:
        connection = _FakeConnection()
        migrator = PostgresMigrator()

        await migrator.apply(connection=connection)

        assert connection.executed[0].startswith("SELECT pg_advisory_xact_lock")
        assert connection.executed[1] == BaselineMigration.MIGRATION_TABLE
        assert [inserted[0] for inserted in connection.inserted] == [1, 2]
        assert [inserted[1] for inserted in connection.inserted] == [
            "baseline",
            "composite_actor_policy_keys",
        ]
        assert all(isinstance(inserted[2], str) for inserted in connection.inserted)

    async def test_apply_checksum_excludes_generated_comments(self) -> None:
        connection = _FakeConnection()
        migrator = PostgresMigrator()

        await migrator.apply(connection=connection)

        expected = sha256(
            "\n".join(
                statement
                for statement in BaselineMigration.step().statements
                if not statement.lstrip().startswith("COMMENT ON ")
            ).encode("utf-8")
        ).hexdigest()

        assert connection.inserted[0][2] == expected

    async def test_apply_noops_when_checksum_matches(self) -> None:
        connection = _FakeConnection()
        migrator = PostgresMigrator()

        await migrator.apply(connection=connection)
        assert connection.inserted

        applied_checksums: Dict[int, str] = {}
        for inserted in connection.inserted:
            version = inserted[0]
            checksum = inserted[2]

            if not isinstance(version, int):
                raise AssertionError("Inserted migration version is not an integer.")

            if not isinstance(checksum, str):
                raise AssertionError("Inserted migration checksum is not text.")

            applied_checksums[version] = checksum

        replay = _FakeConnection(applied_checksums=applied_checksums)
        await migrator.apply(connection=replay)

        assert replay.inserted == []
        assert replay.executed == [
            "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)",
            BaselineMigration.MIGRATION_TABLE,
        ]

    async def test_apply_uses_transaction_scoped_advisory_lock(self) -> None:
        connection = _FakeConnection()
        migrator = PostgresMigrator()

        await migrator.apply(connection=connection)

        assert connection.executed[0] == "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)"
        assert connection.executed[1] == BaselineMigration.MIGRATION_TABLE

    async def test_apply_rejects_checksum_mismatch(self) -> None:
        connection = _FakeConnection(applied_checksum="wrong")
        migrator = PostgresMigrator()

        try:
            await migrator.apply(connection=connection)
        except RuntimeError as exception:
            assert "checksum mismatch" in str(exception)
        else:
            raise AssertionError("Expected checksum mismatch to fail.")

    async def test_apply_accepts_checksum_mismatch_when_schema_is_valid(self) -> None:
        legacy = {step.version: "legacy" for step in ConversationStoreMigrations.steps()}
        connection = _FakeConnection(applied_checksums=legacy, valid_schema=True)
        migrator = PostgresMigrator()

        await migrator.apply(connection=connection)

        assert connection.inserted == []

    def test_registry_exposes_current_schema_version(self) -> None:
        steps = ConversationStoreMigrations.steps()

        assert [step.version for step in steps] == [
            BaselineMigration.VERSION,
            CompositeKeyMigration.VERSION,
        ]
        assert SCHEMA_VERSION == CompositeKeyMigration.VERSION

    def test_baseline_checksum_is_deterministic(self) -> None:
        checksums = {
            sha256("\n".join(BaselineMigration.step().statements).encode("utf-8")).hexdigest()
            for _ in range(5)
        }

        assert len(checksums) == 1


class TestCompositeKeyMigration:
    """
    Verify the tenant-scoped primary-key promotion migration.
    """

    def test_step_is_versioned_after_baseline(self) -> None:
        step = CompositeKeyMigration.step()

        assert step.version == 2
        assert step.version > BaselineMigration.VERSION
        assert step.name == "composite_actor_policy_keys"

    def test_step_promotes_actor_and_policy_keys_only(self) -> None:
        statements = CompositeKeyMigration.step().statements

        assert len(statements) == 2
        assert any("'actors'::regclass" in statement for statement in statements)
        assert any("'policies'::regclass" in statement for statement in statements)

    def test_step_swaps_id_only_key_for_tenant_scoped_key(self) -> None:
        for statement in CompositeKeyMigration.step().statements:
            assert "= 'PRIMARY KEY (id)'" in statement
            assert "ADD PRIMARY KEY (tenant_id, id)" in statement

    def test_step_guards_promotion_for_idempotent_replay(self) -> None:
        for statement in CompositeKeyMigration.step().statements:
            assert statement.lstrip().startswith("DO $$")
            assert "= 'PRIMARY KEY (id)' THEN" in statement


class TestPostgresSchemaValidator:
    """
    Verify validate mode rejects structurally incomplete schemas.
    """

    async def test_validate_accepts_complete_schema(self) -> None:
        """
        Complete schema objects must satisfy validate mode.
        """

        await PostgresSchemaValidator().validate(connection=_FakeValidationConnection())

    async def test_validate_rejects_missing_index(self) -> None:
        """
        Missing required indexes must fail validate mode.
        """

        await self.__assert_invalid_schema(
            connection=_FakeValidationConnection(missing_index="index_search_vector"),
            expected="missing index",
        )

    async def test_validate_rejects_missing_constraint(self) -> None:
        """
        Missing enum and structural constraints must fail validate mode.
        """

        await self.__assert_invalid_schema(
            connection=_FakeValidationConnection(missing_constraint="check_tasks_state_values"),
            expected="missing constraint",
        )

    async def test_validate_rejects_missing_function(self) -> None:
        """
        Missing search functions must fail validate mode.
        """

        await self.__assert_invalid_schema(
            connection=_FakeValidationConnection(missing_function="fathom_messages_search_upsert"),
            expected="missing function",
        )

    async def test_validate_rejects_missing_trigger(self) -> None:
        """
        Missing search triggers must fail validate mode.
        """

        await self.__assert_invalid_schema(
            connection=_FakeValidationConnection(missing_trigger="messages_search_insert"),
            expected="missing trigger",
        )

    async def __assert_invalid_schema(
        self,
        *,
        connection: _FakeValidationConnection,
        expected: str,
    ) -> None:
        """
        Assert validate mode fails with the expected diagnostic fragment.
        """

        try:
            await PostgresSchemaValidator().validate(connection=connection)
        except PostgresSchemaValidationError as exception:
            assert expected in str(exception)
        else:
            raise AssertionError("Expected schema validation to fail.")
