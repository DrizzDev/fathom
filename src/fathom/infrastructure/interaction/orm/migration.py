from __future__ import annotations

from hashlib import sha256
from typing import AsyncContextManager, Final, List, Mapping, Optional, Protocol, Tuple

from fathom.infrastructure.interaction.orm.models import Catalog
from fathom.schemas.postgres import PostgresMigrationStep


class PostgresMigrationConnection(Protocol):
    """
    Async Postgres connection surface required by the migration runner.
    """

    async def execute(self, query: str, *args: object) -> str:
        """
        Execute one SQL command.
        """

        ...

    async def fetchrow(self, query: str, *args: object) -> Optional[Mapping[str, object]]:
        """
        Fetch one row from a SQL query.
        """

        ...

    def transaction(self) -> AsyncContextManager[object]:
        """
        Open a transaction context.
        """

        ...


class PostgresMigrator:
    """
    Applies conversation-store migrations with checksum-backed idempotency.
    """

    __LOCK_NAME: Final[str] = "fathom.conversation.migration"
    __LOCK_SQL: Final[str] = "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)"

    async def apply(
        self,
        *,
        schema: Optional[str] = None,
        connection: PostgresMigrationConnection,
    ) -> None:
        """
        Apply every pending migration in version order.
        """

        async with connection.transaction():
            await connection.execute(self.__LOCK_SQL, self.__LOCK_NAME)

            if schema is not None:
                await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                await connection.execute(f"SET search_path TO {schema}")

            await connection.execute(BaselineMigration.MIGRATION_TABLE)

            for step in ConversationStoreMigrations.steps():
                checksum = self.__checksum(step=step)
                applied = await connection.fetchrow(
                    "SELECT checksum FROM migrations WHERE version = $1", step.version
                )

                if applied is not None:
                    recorded = applied["checksum"]

                    if recorded != checksum:
                        await self.__validate_compatible_schema(
                            step=step,
                            recorded=recorded,
                            checksum=checksum,
                            connection=connection,
                        )
                    continue

                for statement in step.statements:
                    await connection.execute(statement)

                await connection.execute(
                    """
                    INSERT INTO migrations (version, name, checksum)
                    VALUES ($1, $2, $3)
                    """,
                    step.version,
                    step.name,
                    checksum,
                )

    @staticmethod
    def __checksum(*, step: PostgresMigrationStep) -> str:
        """
        Compute a deterministic checksum for a migration step.
        """

        payload = "\n".join(
            statement
            for statement in step.statements
            if not statement.lstrip().startswith("COMMENT ON ")
        ).encode("utf-8")

        return sha256(payload).hexdigest()

    @staticmethod
    async def __validate_compatible_schema(
        *,
        checksum: str,
        recorded: object,
        step: PostgresMigrationStep,
        connection: PostgresMigrationConnection,
    ) -> None:
        """
        Accept checksum drift only when the live schema satisfies the current contract.
        """

        try:
            await PostgresSchemaValidator().validate(connection=connection)
        except PostgresSchemaValidationError as exception:
            raise RuntimeError(
                "Recorded migration checksum mismatch for version "
                f"{step.version}; recorded={recorded!s}, expected={checksum}."
            ) from exception


class ConversationStoreMigrations:
    """
    Ordered migration registry for the persistent conversation store.
    """

    @staticmethod
    def steps() -> Tuple[PostgresMigrationStep, ...]:
        """
        Return all schema migrations in application order.
        """

        return (BaselineMigration.step(), CompositeKeyMigration.step())


class PostgresSchemaValidationError(RuntimeError):
    """
    Raised when an existing Postgres schema cannot satisfy the persistent-store contract.
    """


class PostgresSchemaValidator:
    """
    Validates an existing conversation schema without mutating migration history.
    """

    async def validate(self, *, connection: PostgresMigrationConnection) -> None:
        """
        Verify the existing schema satisfies the conversation-store contract.
        """

        await self.__validate_schema_selected(connection=connection)
        for table, columns in BaselineMigration.REQUIRED_COLUMNS.items():
            await self.__validate_table(connection=connection, table=table)
            for column in columns:
                await self.__validate_column(
                    connection=connection,
                    table=table,
                    column=column,
                )

        for index in BaselineMigration.REQUIRED_INDEXES:
            await self.__validate_index(connection=connection, index=index)

        for constraint in BaselineMigration.REQUIRED_CONSTRAINTS:
            await self.__validate_constraint(connection=connection, constraint=constraint)

        for function in BaselineMigration.REQUIRED_FUNCTIONS:
            await self.__validate_function(connection=connection, function=function)

        for table, triggers in BaselineMigration.REQUIRED_TRIGGERS.items():
            for trigger in triggers:
                await self.__validate_trigger(
                    table=table,
                    trigger=trigger,
                    connection=connection,
                )

    async def __validate_schema_selected(self, *, connection: PostgresMigrationConnection) -> None:
        """
        Ensure the session search path resolves to a concrete schema.
        """

        row = await connection.fetchrow("SELECT current_schema() AS schema_name")
        if row is None or row["schema_name"] is None:
            raise PostgresSchemaValidationError(
                "Postgres schema validation failed because the configured schema does not exist."
            )

    async def __validate_table(
        self, *, connection: PostgresMigrationConnection, table: str
    ) -> None:
        """
        Ensure one required table exists in the current schema.
        """

        row = await connection.fetchrow("SELECT to_regclass($1) AS relation", table)

        if row is None or row["relation"] is None:
            raise PostgresSchemaValidationError(
                f"Postgres schema validation failed: missing table `{table}`."
            )

    async def __validate_column(
        self,
        *,
        table: str,
        column: str,
        connection: PostgresMigrationConnection,
    ) -> None:
        """
        Ensure one required table column exists in the current schema.
        """

        row = await connection.fetchrow(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = $1
              AND column_name = $2
            """,
            table,
            column,
        )

        if row is None:
            raise PostgresSchemaValidationError(
                f"Postgres schema validation failed: missing column `{table}.{column}`."
            )

    async def __validate_index(
        self, *, connection: PostgresMigrationConnection, index: str
    ) -> None:
        """
        Ensure one required index exists in the current schema.
        """

        row = await connection.fetchrow("SELECT to_regclass($1) AS relation", index)

        if row is None or row["relation"] is None:
            raise PostgresSchemaValidationError(
                f"Postgres schema validation failed: missing index `{index}`."
            )

    async def __validate_constraint(
        self, *, connection: PostgresMigrationConnection, constraint: str
    ) -> None:
        """
        Ensure one required constraint exists in the current schema.
        """

        row = await connection.fetchrow(
            """
            SELECT constraint_record.conname
            FROM pg_constraint constraint_record
            JOIN pg_class table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_namespace namespace_record
              ON namespace_record.oid = table_record.relnamespace
            WHERE namespace_record.nspname = current_schema()
              AND constraint_record.conname = $1
            """,
            constraint,
        )
        if row is None:
            raise PostgresSchemaValidationError(
                f"Postgres schema validation failed: missing constraint `{constraint}`."
            )

    async def __validate_function(
        self, *, connection: PostgresMigrationConnection, function: str
    ) -> None:
        """
        Ensure one required function exists in the current schema.
        """

        row = await connection.fetchrow(
            """
            SELECT routine_name
            FROM information_schema.routines
            WHERE routine_schema = current_schema()
              AND routine_name = $1
            """,
            function,
        )
        if row is None:
            raise PostgresSchemaValidationError(
                f"Postgres schema validation failed: missing function `{function}`."
            )

    async def __validate_trigger(
        self,
        *,
        table: str,
        trigger: str,
        connection: PostgresMigrationConnection,
    ) -> None:
        """
        Ensure one required table trigger exists in the current schema.
        """

        row = await connection.fetchrow(
            """
            SELECT trigger_name
            FROM information_schema.triggers
            WHERE trigger_schema = current_schema()
              AND event_object_table = $1
              AND trigger_name = $2
            """,
            table,
            trigger,
        )
        if row is None:
            raise PostgresSchemaValidationError(
                f"Postgres schema validation failed: missing trigger `{table}.{trigger}`."
            )


class BaselineMigration:
    """
    Final Postgres baseline for the persistent-store backed conversation store.
    """

    VERSION: Final[int] = 1
    NAME: Final[str] = "baseline"

    AUDIT_COLUMNS: Final[Tuple[str, ...]] = (
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "deleted_at",
        "deleted_by",
    )
    MUTABLE_TABLES: Final[Tuple[str, ...]] = (
        "jobs",
        "tasks",
        "actors",
        "search",
        "scripts",
        "policies",
        "contexts",
        "requests",
        "messages",
        "artifacts",
        "executions",
        "memberships",
        "conversations",
    )
    APPEND_ONLY_TABLES: Final[Tuple[str, ...]] = ("events", "script_versions")

    MIGRATION_TABLE: Final[str] = """
    CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        checksum TEXT NOT NULL UNIQUE,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """

    TABLES: Final[Tuple[str, ...]] = (
        """
        CREATE TABLE IF NOT EXISTS actors (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            external TEXT,
            runtime TEXT,
            provider TEXT,
            model TEXT,
            skills JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            title TEXT,
            digest TEXT,
            archived_at TIMESTAMPTZ,
            archived_by TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS executions (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            workflow_id TEXT,
            intent TEXT NOT NULL,
            state TEXT NOT NULL,
            code TEXT,
            detail TEXT,
            summary TEXT,
            outcome JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memberships (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            scope TEXT NOT NULL,
            joined_at TIMESTAMPTZ NOT NULL,
            departed_at TIMESTAMPTZ,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, actor) REFERENCES actors(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            assignee TEXT,
            root_id TEXT,
            parent_id TEXT,
            origin_id TEXT,
            kind TEXT NOT NULL,
            objective TEXT NOT NULL,
            reference TEXT,
            state TEXT NOT NULL,
            code TEXT,
            detail TEXT,
            progress JSONB NOT NULL DEFAULT '{}'::jsonb,
            plan JSONB NOT NULL DEFAULT '{}'::jsonb,
            outcome JSONB NOT NULL DEFAULT '{}'::jsonb,
            summary TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            elapsed INTEGER CHECK (elapsed IS NULL OR elapsed >= 0),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, assignee) REFERENCES actors(tenant_id, id),
            FOREIGN KEY (tenant_id, parent_id) REFERENCES tasks(tenant_id, id),
            FOREIGN KEY (tenant_id, root_id) REFERENCES tasks(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            task_id TEXT,
            author TEXT NOT NULL,
            reply_id TEXT,
            sequence BIGINT NOT NULL CHECK (sequence > 0),
            kind TEXT NOT NULL,
            audience JSONB NOT NULL DEFAULT '[]'::jsonb,
            body JSONB NOT NULL,
            labels JSONB NOT NULL DEFAULT '[]'::jsonb,
            sanitized_at TIMESTAMPTZ,
            sanitizer TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, conversation_id, sequence),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id),
            FOREIGN KEY (tenant_id, author) REFERENCES actors(tenant_id, id),
            FOREIGN KEY (tenant_id, reply_id) REFERENCES messages(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT,
            task_id TEXT,
            actor TEXT,
            sequence BIGINT NOT NULL CHECK (sequence > 0),
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            payload JSONB NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, conversation_id, sequence),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id),
            FOREIGN KEY (tenant_id, actor) REFERENCES actors(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            task_id TEXT,
            producer TEXT,
            kind TEXT NOT NULL,
            uri TEXT NOT NULL,
            backend TEXT NOT NULL,
            mime TEXT,
            size BIGINT CHECK (size IS NULL OR size >= 0),
            retention TEXT,
            labels JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id),
            FOREIGN KEY (tenant_id, producer) REFERENCES actors(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scripts (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            task_id TEXT,
            title TEXT,
            format TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision > 0),
            checksum TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id),
            FOREIGN KEY (tenant_id, created_by) REFERENCES actors(tenant_id, id),
            FOREIGN KEY (tenant_id, updated_by) REFERENCES actors(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS script_versions (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            script_id TEXT NOT NULL,
            version INTEGER NOT NULL CHECK (version > 0),
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            checksum TEXT NOT NULL,
            summary TEXT,
            actor TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, script_id, version),
            FOREIGN KEY (tenant_id, script_id) REFERENCES scripts(tenant_id, id),
            FOREIGN KEY (tenant_id, actor) REFERENCES actors(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS policies (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            scope TEXT NOT NULL,
            name TEXT NOT NULL,
            region TEXT,
            retention JSONB NOT NULL,
            labels JSONB NOT NULL DEFAULT '[]'::jsonb,
            sanitizers JSONB NOT NULL DEFAULT '[]'::jsonb,
            memories JSONB NOT NULL DEFAULT '[]'::jsonb,
            artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS contexts (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            task_id TEXT,
            consumer TEXT,
            purpose TEXT NOT NULL,
            builder TEXT NOT NULL,
            "references" JSONB NOT NULL,
            budget JSONB NOT NULL DEFAULT '{}'::jsonb,
            filters JSONB NOT NULL DEFAULT '{}'::jsonb,
            hash TEXT,
            provider TEXT,
            model TEXT,
            expires_at TIMESTAMPTZ,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id),
            FOREIGN KEY (tenant_id, consumer) REFERENCES actors(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            key TEXT NOT NULL,
            hash TEXT NOT NULL,
            state TEXT NOT NULL,
            response JSONB,
            expires_at TIMESTAMPTZ NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            task_id TEXT,
            kind TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            owner TEXT,
            locked_at TIMESTAMPTZ,
            available_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            code TEXT,
            detail TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id),
            FOREIGN KEY (tenant_id, task_id) REFERENCES tasks(tenant_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sequences (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            scope TEXT NOT NULL CHECK (scope IN ('message', 'event')),
            value BIGINT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, conversation_id, scope),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id)
        )
        """,
        # TODO(conversation-ledger): search is a derived index; remove its actor audit columns during the next clean schema reset.
        """
        CREATE TABLE IF NOT EXISTS search (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workspace_id TEXT,
            conversation_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            document TEXT NOT NULL,
            vector TSVECTOR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by TEXT,
            deleted_at TIMESTAMPTZ,
            deleted_by TEXT,
            UNIQUE (tenant_id, conversation_id, source, source_id),
            FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations(tenant_id, id),
            FOREIGN KEY (tenant_id, execution_id) REFERENCES executions(tenant_id, id)
        )
        """,
    )

    COMMON_COLUMNS: Final[Tuple[str, ...]] = (
        "id",
        "metadata",
        "tenant_id",
        "workspace_id",
        *AUDIT_COLUMNS,
    )
    REQUIRED_COLUMNS: Final[Mapping[str, Tuple[str, ...]]] = {
        "actors": COMMON_COLUMNS
        + (
            "kind",
            "name",
            "external",
            "runtime",
            "provider",
            "model",
            "skills",
        ),
        "conversations": COMMON_COLUMNS
        + (
            "title",
            "digest",
            "archived_at",
            "archived_by",
        ),
        "executions": COMMON_COLUMNS
        + (
            "conversation_id",
            "workflow_id",
            "intent",
            "state",
            "code",
            "detail",
            "summary",
            "outcome",
            "started_at",
            "completed_at",
        ),
        "memberships": COMMON_COLUMNS
        + (
            "conversation_id",
            "actor",
            "role",
            "scope",
            "joined_at",
            "departed_at",
        ),
        "tasks": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "assignee",
            "parent_id",
            "root_id",
            "origin_id",
            "kind",
            "objective",
            "reference",
            "state",
            "code",
            "detail",
            "progress",
            "plan",
            "outcome",
            "summary",
            "started_at",
            "completed_at",
            "elapsed",
        ),
        "messages": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "task_id",
            "author",
            "reply_id",
            "sequence",
            "kind",
            "audience",
            "body",
            "labels",
            "sanitized_at",
            "sanitizer",
        ),
        "events": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "task_id",
            "actor",
            "sequence",
            "kind",
            "source",
            "payload",
        ),
        "artifacts": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "task_id",
            "producer",
            "kind",
            "uri",
            "backend",
            "mime",
            "size",
            "retention",
            "labels",
        ),
        "scripts": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "task_id",
            "title",
            "format",
            "status",
            "content",
            "revision",
            "checksum",
        ),
        "script_versions": COMMON_COLUMNS
        + (
            "script_id",
            "version",
            "source",
            "content",
            "checksum",
            "summary",
            "actor",
        ),
        "policies": COMMON_COLUMNS
        + (
            "scope",
            "name",
            "region",
            "retention",
            "labels",
            "sanitizers",
            "memories",
            "artifacts",
        ),
        "contexts": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "task_id",
            "consumer",
            "purpose",
            "builder",
            "references",
            "budget",
            "filters",
            "hash",
            "provider",
            "model",
            "expires_at",
        ),
        "requests": COMMON_COLUMNS
        + (
            "key",
            "hash",
            "state",
            "response",
            "expires_at",
        ),
        "jobs": COMMON_COLUMNS
        + (
            "conversation_id",
            "execution_id",
            "task_id",
            "kind",
            "state",
            "attempts",
            "owner",
            "locked_at",
            "available_at",
            "payload",
            "code",
            "detail",
        ),
        "sequences": COMMON_COLUMNS
        + (
            "conversation_id",
            "scope",
            "value",
        ),
        "search": (
            "id",
            "tenant_id",
            "workspace_id",
            "conversation_id",
            "execution_id",
            "source",
            "source_id",
            "document",
            "vector",
            *AUDIT_COLUMNS,
        ),
    }

    STRUCTURAL_CONSTRAINTS: Final[Tuple[str, ...]] = (
        """
        DO $body$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint constraint_record
                JOIN pg_class table_record
                  ON table_record.oid = constraint_record.conrelid
                JOIN pg_namespace namespace_record
                  ON namespace_record.oid = table_record.relnamespace
                WHERE namespace_record.nspname = current_schema()
                  AND table_record.relname = 'tasks'
                  AND constraint_record.conname = 'foreign_key_tasks_origin_messages'
            ) THEN
                ALTER TABLE tasks
                ADD CONSTRAINT foreign_key_tasks_origin_messages
                FOREIGN KEY (tenant_id, origin_id)
                REFERENCES messages(tenant_id, id)
                DEFERRABLE INITIALLY DEFERRED;
            END IF;
        END
        $body$;
        """,
    )

    INDEXES: Final[Tuple[str, ...]] = (
        "CREATE INDEX IF NOT EXISTS index_actors_tenant_kind ON actors(tenant_id, kind)",
        "CREATE INDEX IF NOT EXISTS index_conversations_active_updated ON conversations(tenant_id, workspace_id, updated_at DESC, id DESC) WHERE deleted_at IS NULL AND archived_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_conversations_archived_updated ON conversations(tenant_id, workspace_id, archived_at DESC, id DESC) WHERE deleted_at IS NULL AND archived_at IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS index_executions_conversation ON executions(tenant_id, conversation_id, created_at DESC)",
        "CREATE UNIQUE INDEX IF NOT EXISTS index_executions_workflow_unique ON executions(tenant_id, workflow_id) WHERE workflow_id IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS index_executions_workflow ON executions(workflow_id) WHERE workflow_id IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS index_memberships_conversation ON memberships(tenant_id, conversation_id)",
        "CREATE INDEX IF NOT EXISTS index_memberships_actor ON memberships(tenant_id, actor)",
        "CREATE UNIQUE INDEX IF NOT EXISTS index_memberships_active_actor ON memberships(tenant_id, conversation_id, actor) WHERE deleted_at IS NULL AND departed_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_tasks_conversation ON tasks(tenant_id, conversation_id)",
        "CREATE INDEX IF NOT EXISTS index_tasks_execution ON tasks(tenant_id, execution_id)",
        "CREATE INDEX IF NOT EXISTS index_tasks_parent ON tasks(tenant_id, parent_id)",
        "CREATE INDEX IF NOT EXISTS index_tasks_conversation_active ON tasks(tenant_id, conversation_id, created_at, id) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_messages_task ON messages(tenant_id, task_id, sequence)",
        "CREATE INDEX IF NOT EXISTS index_messages_execution ON messages(tenant_id, execution_id, sequence)",
        "CREATE INDEX IF NOT EXISTS index_messages_timeline_active ON messages(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_messages_kind_timeline_active ON messages(tenant_id, conversation_id, kind, created_at DESC, id DESC) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_events_task ON events(tenant_id, task_id, sequence)",
        "CREATE INDEX IF NOT EXISTS index_events_execution ON events(tenant_id, execution_id, sequence)",
        "CREATE INDEX IF NOT EXISTS index_events_conversation ON events(tenant_id, conversation_id, sequence)",
        "CREATE INDEX IF NOT EXISTS index_events_timeline_active ON events(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_artifacts_conversation ON artifacts(tenant_id, conversation_id)",
        "CREATE INDEX IF NOT EXISTS index_artifacts_execution ON artifacts(tenant_id, execution_id)",
        "CREATE INDEX IF NOT EXISTS index_artifacts_task ON artifacts(tenant_id, task_id)",
        "CREATE INDEX IF NOT EXISTS index_artifacts_timeline_active ON artifacts(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_scripts_conversation ON scripts(tenant_id, conversation_id, updated_at)",
        "CREATE INDEX IF NOT EXISTS index_scripts_execution ON scripts(tenant_id, execution_id)",
        "CREATE INDEX IF NOT EXISTS index_scripts_task ON scripts(tenant_id, task_id)",
        "CREATE INDEX IF NOT EXISTS index_scripts_conversation_active ON scripts(tenant_id, conversation_id, updated_at) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_script_versions_script ON script_versions(tenant_id, script_id, version)",
        "CREATE UNIQUE INDEX IF NOT EXISTS index_policies_lookup ON policies(tenant_id, COALESCE(workspace_id, ''), name)",
        "CREATE INDEX IF NOT EXISTS index_contexts_conversation ON contexts(tenant_id, conversation_id, created_at)",
        "CREATE INDEX IF NOT EXISTS index_contexts_execution ON contexts(tenant_id, execution_id, created_at)",
        "CREATE INDEX IF NOT EXISTS index_contexts_task ON contexts(tenant_id, task_id, created_at)",
        "CREATE INDEX IF NOT EXISTS index_contexts_timeline_active ON contexts(tenant_id, conversation_id, created_at DESC, id DESC) WHERE deleted_at IS NULL",
        "CREATE INDEX IF NOT EXISTS index_contexts_references_messages ON contexts USING GIN ((\"references\" -> 'messages'))",
        "CREATE INDEX IF NOT EXISTS index_contexts_references_events ON contexts USING GIN ((\"references\" -> 'events'))",
        "CREATE INDEX IF NOT EXISTS index_contexts_references_artifacts ON contexts USING GIN ((\"references\" -> 'artifacts'))",
        "CREATE INDEX IF NOT EXISTS index_requests_expires ON requests(tenant_id, expires_at)",
        "CREATE INDEX IF NOT EXISTS index_jobs_claim ON jobs(tenant_id, state, available_at, kind)",
        "CREATE INDEX IF NOT EXISTS index_jobs_conversation ON jobs(tenant_id, conversation_id)",
        "CREATE INDEX IF NOT EXISTS index_jobs_execution ON jobs(tenant_id, execution_id)",
        "CREATE INDEX IF NOT EXISTS index_sequences_conversation ON sequences(tenant_id, conversation_id, scope)",
        "CREATE INDEX IF NOT EXISTS index_search_vector ON search USING GIN(vector)",
        "CREATE INDEX IF NOT EXISTS index_search_conversation ON search(tenant_id, conversation_id, source)",
        "CREATE INDEX IF NOT EXISTS index_search_execution ON search(tenant_id, execution_id, source)",
    )
    REQUIRED_INDEXES: Final[Tuple[str, ...]] = tuple(
        statement.split(" IF NOT EXISTS ", 1)[1].split(" ON ", 1)[0] for statement in INDEXES
    )

    REQUIRED_CONSTRAINTS: Final[Tuple[str, ...]] = (
        "check_jobs_kind_values",
        "check_jobs_code_values",
        "check_tasks_kind_values",
        "check_tasks_code_values",
        "check_jobs_state_values",
        "check_actors_kind_values",
        "check_tasks_state_values",
        "check_events_kind_values",
        "check_messages_kind_values",
        "check_events_source_values",
        "check_artifacts_kind_values",
        "check_scripts_format_values",
        "check_scripts_status_values",
        "check_policies_scope_values",
        "check_requests_state_values",
        "check_executions_state_values",
        "check_memberships_role_values",
        "check_contexts_purpose_values",
        "check_memberships_scope_values",
        "check_artifacts_backend_values",
        "foreign_key_tasks_origin_messages",
        "check_script_versions_source_values",
    )
    REQUIRED_FUNCTIONS: Final[Tuple[str, ...]] = (
        "fathom_touch_updated_at",
        "fathom_messages_search_upsert",
        "fathom_messages_search_delete",
        "fathom_message_search_document",
        "fathom_reject_append_only_update",
    )
    REQUIRED_TRIGGERS: Final[Mapping[str, Tuple[str, ...]]] = {
        **{table: (f"{table}_touch_updated_at",) for table in MUTABLE_TABLES},
        "events": ("events_reject_update",),
        "script_versions": ("script_versions_reject_update",),
        "messages": (
            "messages_search_insert",
            "messages_search_update",
            "messages_search_delete",
            "messages_touch_updated_at",
        ),
    }

    ENUM_CHECKS: Final[Tuple[str, ...]] = (
        """
        DO $body$
        DECLARE
            spec TEXT[];
            constraint_name TEXT;
        BEGIN
            FOREACH spec SLICE 1 IN ARRAY ARRAY[
                ARRAY['actors','kind','human,agent,coordinator,team,tool,system'],
                ARRAY['executions','state','running,succeeded,failed,cancelled'],
                ARRAY['memberships','role','owner,requester,responder,coordinator,delegate,observer,system'],
                ARRAY['memberships','scope','thread,task,actor,team,system'],
                ARRAY['tasks','kind','agent,tool,coordination,delegation,fathom,script,clarification,analysis'],
                ARRAY['tasks','state','queued,running,blocked,waiting,succeeded,failed,cancelled,expired,deleted'],
                ARRAY['tasks','code','completed,worker_lost,user_cancelled,timeout,policy_blocked,validation_failed,unknown_error'],
                ARRAY['messages','kind','request,instruction,question,answer,progress,result,note,notice'],
                ARRAY['events','kind','thread.created,thread.archived,thread.unarchived,thread.deleted,actor.joined,task.opened,task.started,task.blocked,task.waiting,task.delegated,task.succeeded,task.failed,task.cancelled,task.expired,task.deleted,message.recorded,content.classified,content.sanitized,artifact.linked,context.built,job.scheduled,job.rescheduled,job.completed,job.failed,job.abandoned,client.disconnected,recovery.lost'],
                ARRAY['events','source','interaction,fathom,policy,worker,artifact,client,recovery'],
                ARRAY['artifacts','kind','screenshot,trace,structured_log,script,report,context_debug,tool_output,model_output'],
                ARRAY['artifacts','backend','local,object'],
                ARRAY['scripts','format','text/plain'],
                ARRAY['scripts','status','draft,active,archived,deleted'],
                ARRAY['script_versions','source','generated,edited,imported'],
                ARRAY['contexts','purpose','execution,conversation,digest,delegation,audit'],
                ARRAY['policies','scope','tenant,workspace'],
                ARRAY['requests','state','started,completed,failed'],
                ARRAY['jobs','state','pending,claimed,completed,failed,abandoned'],
                ARRAY['jobs','kind','execution,digest,sanitize,context,memory,artifact,recovery'],
                ARRAY['jobs','code','completed,retryable_error,permanent_error,worker_lost,cancelled,unknown_error']
            ]
            LOOP
                constraint_name := format('check_%s_%s_values', spec[1], spec[2]);
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint constraint_record
                    JOIN pg_class table_record
                      ON table_record.oid = constraint_record.conrelid
                    JOIN pg_namespace namespace_record
                      ON namespace_record.oid = table_record.relnamespace
                    WHERE namespace_record.nspname = current_schema()
                      AND table_record.relname = spec[1]
                      AND constraint_record.conname = constraint_name
                ) THEN
                    EXECUTE format(
                        'ALTER TABLE %I.%I ADD CONSTRAINT %I CHECK (%I = ANY (string_to_array(%L, '','')))',
                        current_schema(),
                        spec[1],
                        constraint_name,
                        spec[2],
                        spec[3]
                    );
                END IF;
            END LOOP;
        END
        $body$;
        """,
    )

    AUDIT_FUNCTIONS: Final[Tuple[str, ...]] = (
        """
        CREATE OR REPLACE FUNCTION fathom_touch_updated_at()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.updated_at IS NOT DISTINCT FROM OLD.updated_at THEN
                NEW.updated_at := now();
            END IF;
            RETURN NEW;
        END
        $$;
        """,
        """
        CREATE OR REPLACE FUNCTION fathom_reject_append_only_update()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'append-only table % cannot be updated', TG_TABLE_NAME;
        END
        $$;
        """,
    )

    SEARCH_FUNCTIONS: Final[Tuple[str, ...]] = (
        """
        CREATE OR REPLACE FUNCTION fathom_message_search_document(body JSONB)
        RETURNS TEXT
        LANGUAGE SQL
        IMMUTABLE
        AS $$
            SELECT trim(BOTH ' ' FROM concat_ws(
                ' ',
                body->>'text',
                body->>'message',
                body->>'summary',
                body->>'detail',
                body->>'error',
                body->>'progress',
                body->>'note',
                body->>'intent',
                body->>'package',
                body->>'status',
                body->>'reason',
                body->>'success',
                body->>'steps',
                body->>'workflow',
                body->>'evidence'
            ))
        $$;
        """,
        """
        CREATE OR REPLACE FUNCTION fathom_messages_search_upsert()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        DECLARE
            document TEXT;
        BEGIN
            document := fathom_message_search_document(NEW.body);
            INSERT INTO search (
                id,
                tenant_id,
                workspace_id,
                conversation_id,
                execution_id,
                source,
                source_id,
                document,
                vector,
                created_at,
                updated_at
            )
            VALUES (
                NEW.id,
                NEW.tenant_id,
                NEW.workspace_id,
                NEW.conversation_id,
                NEW.execution_id,
                'message',
                NEW.id,
                document,
                to_tsvector('simple', document),
                NEW.created_at,
                NEW.updated_at
            )
            ON CONFLICT (tenant_id, conversation_id, source, source_id)
            DO UPDATE SET
                document = EXCLUDED.document,
                vector = EXCLUDED.vector,
                updated_at = EXCLUDED.updated_at;
            RETURN NEW;
        END
        $$;
        """,
        """
        CREATE OR REPLACE FUNCTION fathom_messages_search_delete()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            DELETE FROM search
            WHERE tenant_id = OLD.tenant_id
              AND conversation_id = OLD.conversation_id
              AND source = 'message'
              AND source_id = OLD.id;
            RETURN OLD;
        END
        $$;
        """,
    )

    TRIGGERS: Final[Tuple[str, ...]] = (
        *tuple(
            f"""
            DROP TRIGGER IF EXISTS {table}_touch_updated_at ON {table};
            CREATE TRIGGER {table}_touch_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION fathom_touch_updated_at()
            """
            for table in MUTABLE_TABLES
        ),
        *tuple(
            f"""
            DROP TRIGGER IF EXISTS {table}_reject_update ON {table};
            CREATE TRIGGER {table}_reject_update
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION fathom_reject_append_only_update()
            """
            for table in APPEND_ONLY_TABLES
        ),
        """
        DROP TRIGGER IF EXISTS messages_search_insert ON messages;
        CREATE TRIGGER messages_search_insert
        AFTER INSERT ON messages
        FOR EACH ROW
        EXECUTE FUNCTION fathom_messages_search_upsert()
        """,
        """
        DROP TRIGGER IF EXISTS messages_search_update ON messages;
        CREATE TRIGGER messages_search_update
        AFTER UPDATE OF body ON messages
        FOR EACH ROW
        EXECUTE FUNCTION fathom_messages_search_upsert()
        """,
        """
        DROP TRIGGER IF EXISTS messages_search_delete ON messages;
        CREATE TRIGGER messages_search_delete
        AFTER DELETE ON messages
        FOR EACH ROW
        EXECUTE FUNCTION fathom_messages_search_delete()
        """,
    )

    @classmethod
    def comments(cls) -> Tuple[str, ...]:
        """
        Return table and column comments from ORM model metadata.
        """

        statements: List[str] = []

        for model in sorted(Catalog().all(), key=lambda item: item._meta.db_table):
            table = model._meta.db_table
            table_description = model._meta.table_description

            if table_description:
                statements.append(
                    "COMMENT ON TABLE "
                    f"{cls.__identifier(value=table)} IS {cls.__literal(value=table_description)}"
                )

            fields = sorted(
                model._meta.fields_map.items(),
                key=lambda item: item[1].source_field or item[0],
            )

            for field_name, field in fields:
                description = field.description

                if description is None:
                    continue

                column = field.source_field or field_name

                if column not in cls.REQUIRED_COLUMNS.get(table, ()):
                    continue

                statements.append(
                    "COMMENT ON COLUMN "
                    f"{cls.__identifier(value=table)}.{cls.__identifier(value=column)} "
                    f"IS {cls.__literal(value=description)}"
                )

        return tuple(statements)

    @classmethod
    def step(cls) -> PostgresMigrationStep:
        """
        Return the one clean baseline migration step.
        """

        return PostgresMigrationStep(
            version=cls.VERSION,
            name=cls.NAME,
            statements=(
                cls.MIGRATION_TABLE,
                *cls.TABLES,
                *cls.STRUCTURAL_CONSTRAINTS,
                *cls.INDEXES,
                *cls.ENUM_CHECKS,
                *cls.comments(),
                *cls.AUDIT_FUNCTIONS,
                *cls.SEARCH_FUNCTIONS,
                *cls.TRIGGERS,
            ),
        )

    @classmethod
    def statements(cls) -> Tuple[str, ...]:
        """
        Return baseline SQL statements in execution order.
        """

        return cls.step().statements

    @staticmethod
    def __identifier(*, value: str) -> str:
        """
        Quote one SQL identifier.
        """

        return '"' + value.replace('"', '""') + '"'

    @staticmethod
    def __literal(*, value: str) -> str:
        """
        Quote one SQL string literal.
        """

        return "'" + value.replace("'", "''") + "'"


class CompositeKeyMigration:
    """
    Promotes the actor and policy primary keys to tenant-scoped composite keys.
    """

    VERSION: Final[int] = 2
    NAME: Final[str] = "composite_actor_policy_keys"

    STATEMENTS: Final[Tuple[str, ...]] = (
        """
        DO $$ BEGIN
            IF (
                SELECT pg_get_constraintdef(constraint_record.oid)
                FROM pg_constraint constraint_record
                WHERE constraint_record.conrelid = 'actors'::regclass
                  AND constraint_record.contype = 'p'
            ) = 'PRIMARY KEY (id)' THEN
                ALTER TABLE actors DROP CONSTRAINT actors_pkey;
                ALTER TABLE actors ADD PRIMARY KEY (tenant_id, id);
            END IF;
        END $$;
        """,
        """
        DO $$ BEGIN
            IF (
                SELECT pg_get_constraintdef(constraint_record.oid)
                FROM pg_constraint constraint_record
                WHERE constraint_record.conrelid = 'policies'::regclass
                  AND constraint_record.contype = 'p'
            ) = 'PRIMARY KEY (id)' THEN
                ALTER TABLE policies DROP CONSTRAINT policies_pkey;
                ALTER TABLE policies ADD PRIMARY KEY (tenant_id, id);
            END IF;
        END $$;
        """,
    )

    @classmethod
    def step(cls) -> PostgresMigrationStep:
        """
        Return the tenant-scoped primary-key promotion migration step.
        """

        return PostgresMigrationStep(
            name=cls.NAME,
            version=cls.VERSION,
            statements=cls.STATEMENTS,
        )


SCHEMA_VERSION: Final[int] = CompositeKeyMigration.VERSION
MIGRATION_STEPS: Final[Tuple[PostgresMigrationStep, ...]] = ConversationStoreMigrations.steps()
