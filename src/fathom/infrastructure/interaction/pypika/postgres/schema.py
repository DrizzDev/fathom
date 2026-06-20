from __future__ import annotations

from typing import Final, Tuple

from fathom.infrastructure.interaction.schema import AuditColumns
from fathom.schemas.postgres import PostgresMigrationStep

MIGRATION_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS migrations (
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (version),
    UNIQUE (checksum)
)
"""

AUDIT_COLUMNS: Final[AuditColumns] = AuditColumns(
    timestamp_type="TIMESTAMPTZ",
    metadata_type="JSONB",
    indent="        ",
)


TABLES: Final[Tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS actors (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        external TEXT,
        runtime TEXT,
        provider TEXT,
        model TEXT,
        skills JSONB NOT NULL,
"""
    + AUDIT_COLUMNS.created_updated()
    + """,
        PRIMARY KEY (tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        title TEXT,
        state TEXT NOT NULL,
        digest TEXT,
        cursor INTEGER,
        creator TEXT,
        archived_at TIMESTAMPTZ,
"""
    + AUDIT_COLUMNS.created_updated_deleted()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, creator) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memberships (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        actor TEXT NOT NULL,
        role TEXT NOT NULL,
        scope TEXT NOT NULL,
        joined_at TIMESTAMPTZ NOT NULL,
        departed_at TIMESTAMPTZ,
        metadata JSONB NOT NULL,
        PRIMARY KEY (tenant, id),
        UNIQUE (tenant, thread, actor),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, actor) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        creator TEXT,
        assignee TEXT,
        parent TEXT,
        root TEXT,
        origin TEXT,
        kind TEXT NOT NULL,
        objective TEXT NOT NULL,
        reference TEXT,
        state TEXT NOT NULL,
        code TEXT,
        detail TEXT,
        progress JSONB NOT NULL,
        plan JSONB NOT NULL,
        summary TEXT,
        started_at TIMESTAMPTZ,
        ended_at TIMESTAMPTZ,
        elapsed INTEGER,
"""
    + AUDIT_COLUMNS.created_updated_deleted()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, creator) REFERENCES actors(tenant, id),
        FOREIGN KEY (tenant, assignee) REFERENCES actors(tenant, id),
        FOREIGN KEY (tenant, parent) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, root) REFERENCES tasks(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        task TEXT,
        author TEXT NOT NULL,
        reply TEXT,
        sequence INTEGER NOT NULL,
        kind TEXT NOT NULL,
        audience TEXT NOT NULL,
        body JSONB NOT NULL,
        labels JSONB NOT NULL,
        sanitized_at TIMESTAMPTZ,
        sanitizer TEXT,
"""
    + AUDIT_COLUMNS.created_deleted()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, author) REFERENCES actors(tenant, id),
        FOREIGN KEY (tenant, reply) REFERENCES messages(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        task TEXT,
        actor TEXT,
        sequence INTEGER NOT NULL,
        kind TEXT NOT NULL,
        source TEXT NOT NULL,
        payload JSONB NOT NULL,
"""
    + AUDIT_COLUMNS.created()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, actor) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        task TEXT,
        producer TEXT,
        kind TEXT NOT NULL,
        uri TEXT NOT NULL,
        backend TEXT NOT NULL,
        mime TEXT,
        size INTEGER,
        retention TEXT,
        labels JSONB NOT NULL,
"""
    + AUDIT_COLUMNS.created_deleted()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, producer) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scripts (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        task TEXT,
        artifact TEXT,
        title TEXT,
        format TEXT NOT NULL,
        status TEXT NOT NULL,
        content TEXT NOT NULL,
        revision INTEGER NOT NULL,
        created_by TEXT,
        updated_by TEXT,
"""
    + AUDIT_COLUMNS.created_updated_deleted()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, artifact) REFERENCES artifacts(tenant, id),
        FOREIGN KEY (tenant, created_by) REFERENCES actors(tenant, id),
        FOREIGN KEY (tenant, updated_by) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS script_versions (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        script TEXT NOT NULL,
        thread TEXT NOT NULL,
        task TEXT,
        artifact TEXT,
        version INTEGER NOT NULL,
        source TEXT NOT NULL,
        content TEXT NOT NULL,
        checksum TEXT NOT NULL,
        summary TEXT,
        actor TEXT,
"""
    + AUDIT_COLUMNS.created()
    + """,
        PRIMARY KEY (tenant, id),
        UNIQUE (tenant, script, version),
        FOREIGN KEY (tenant, script) REFERENCES scripts(tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, artifact) REFERENCES artifacts(tenant, id),
        FOREIGN KEY (tenant, actor) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS policies (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        scope TEXT NOT NULL,
        name TEXT NOT NULL,
        region TEXT,
        retention JSONB NOT NULL,
        labels JSONB NOT NULL,
        sanitizers JSONB NOT NULL,
        memories JSONB NOT NULL,
        artifacts JSONB NOT NULL,
"""
    + AUDIT_COLUMNS.created_updated()
    + """,
        PRIMARY KEY (tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS contexts (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        task TEXT,
        consumer TEXT,
        purpose TEXT NOT NULL,
        builder TEXT NOT NULL,
        "references" JSONB NOT NULL,
        budget JSONB NOT NULL,
        filters JSONB NOT NULL,
        hash TEXT,
        provider TEXT,
        model TEXT,
        expires_at TIMESTAMPTZ,
"""
    + AUDIT_COLUMNS.created()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
        FOREIGN KEY (tenant, consumer) REFERENCES actors(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS requests (
        tenant TEXT NOT NULL,
        key TEXT NOT NULL,
        hash TEXT NOT NULL,
        state TEXT NOT NULL,
        response JSONB,
        expires_at TIMESTAMPTZ NOT NULL,
"""
    + AUDIT_COLUMNS.created()
    + """,
        PRIMARY KEY (tenant, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT NOT NULL,
        tenant TEXT NOT NULL,
        workspace TEXT,
        thread TEXT NOT NULL,
        task TEXT,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        owner TEXT,
        locked_at TIMESTAMPTZ,
        available_at TIMESTAMPTZ NOT NULL,
        payload JSONB NOT NULL,
        code TEXT,
        detail TEXT,
"""
    + AUDIT_COLUMNS.created_updated()
    + """,
        PRIMARY KEY (tenant, id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
        FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sequences (
        tenant TEXT NOT NULL,
        thread TEXT NOT NULL,
        scope TEXT NOT NULL CHECK (scope IN ('message', 'event')),
        value INTEGER NOT NULL,
        PRIMARY KEY (tenant, thread, scope),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search (
        tenant TEXT NOT NULL,
        thread TEXT NOT NULL,
        source TEXT NOT NULL,
        source_id TEXT NOT NULL,
        document TEXT NOT NULL,
        vector TSVECTOR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (tenant, thread, source, source_id),
        FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id)
    )
    """,
)


INDEXES: Final[Tuple[str, ...]] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_thread_sequence "
    "ON messages(tenant, thread, sequence)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_thread_sequence "
    "ON events(tenant, thread, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_threads_tenant_state ON threads(tenant, state)",
    "CREATE INDEX IF NOT EXISTS idx_actors_tenant_kind ON actors(tenant, kind)",
    "CREATE INDEX IF NOT EXISTS idx_memberships_thread ON memberships(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_memberships_actor ON memberships(tenant, actor)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_thread ON tasks(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(tenant, parent)",
    "CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(tenant, task, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_events_task ON events(tenant, task, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_thread ON artifacts(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(tenant, task)",
    "CREATE INDEX IF NOT EXISTS idx_scripts_thread ON scripts(tenant, thread, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_scripts_task ON scripts(tenant, task)",
    "CREATE INDEX IF NOT EXISTS idx_scripts_artifact ON scripts(tenant, artifact)",
    "CREATE INDEX IF NOT EXISTS idx_script_versions_script "
    "ON script_versions(tenant, script, version)",
    "CREATE INDEX IF NOT EXISTS idx_script_versions_artifact ON script_versions(tenant, artifact)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_policies_lookup "
    "ON policies(tenant, COALESCE(workspace, ''), name)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(tenant, state, available_at, kind)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_thread ON jobs(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_requests_expires ON requests(tenant, expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_contexts_thread ON contexts(tenant, thread, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_contexts_task ON contexts(tenant, task, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_threads_active_updated "
    "ON threads(tenant, updated_at DESC) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_tasks_thread_active "
    "ON tasks(tenant, thread, created_at) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_thread_active "
    "ON artifacts(tenant, thread, created_at, id) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_messages_active "
    "ON messages(tenant, thread, sequence) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_search_vector ON search USING GIN(vector)",
)


BACKFILLS: Final[Tuple[str, ...]] = (
    """
    INSERT INTO sequences (tenant, thread, scope, value)
    SELECT tenant, thread, 'message', COALESCE(MAX(sequence), 0)
    FROM messages
    GROUP BY tenant, thread
    ON CONFLICT (tenant, thread, scope) DO NOTHING
    """,
    """
    INSERT INTO sequences (tenant, thread, scope, value)
    SELECT tenant, thread, 'event', COALESCE(MAX(sequence), 0)
    FROM events
    GROUP BY tenant, thread
    ON CONFLICT (tenant, thread, scope) DO NOTHING
    """,
)


STRUCTURAL_CONSTRAINTS: Final[Tuple[str, ...]] = (
    """
    DO $body$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = current_schema()
              AND t.relname = 'tasks'
              AND c.conname = 'fk_tasks_origin_messages'
        ) THEN
            EXECUTE format(
                'ALTER TABLE %I.tasks '
                'ADD CONSTRAINT fk_tasks_origin_messages '
                'FOREIGN KEY (tenant, origin) REFERENCES %I.messages(tenant, id) '
                'DEFERRABLE INITIALLY DEFERRED',
                current_schema(), current_schema()
            );
        END IF;
    END
    $body$;
    """,
)


MIGRATIONS: Final[Tuple[str, ...]] = (
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
            body->>'workflow'
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
        INSERT INTO search (tenant, thread, source, source_id, document, vector, created_at)
        VALUES (
            NEW.tenant,
            NEW.thread,
            'message',
            NEW.id,
            document,
            to_tsvector('simple', document),
            NEW.created_at
        )
        ON CONFLICT (tenant, thread, source, source_id)
        DO UPDATE SET
            document = EXCLUDED.document,
            vector = EXCLUDED.vector,
            created_at = EXCLUDED.created_at;
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
        WHERE tenant = OLD.tenant
          AND thread = OLD.thread
          AND source = 'message'
          AND source_id = OLD.id;
        RETURN OLD;
    END
    $$;
    """,
    """
    DROP TRIGGER IF EXISTS messages_search_insert ON messages;
    CREATE TRIGGER messages_search_insert
    AFTER INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION fathom_messages_search_upsert();
    """,
    """
    DROP TRIGGER IF EXISTS messages_search_update ON messages;
    CREATE TRIGGER messages_search_update
    AFTER UPDATE OF body ON messages
    FOR EACH ROW
    EXECUTE FUNCTION fathom_messages_search_upsert();
    """,
    """
    DROP TRIGGER IF EXISTS messages_search_delete ON messages;
    CREATE TRIGGER messages_search_delete
    AFTER DELETE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION fathom_messages_search_delete();
    """,
    """
    INSERT INTO search (tenant, thread, source, source_id, document, vector, created_at)
    SELECT tenant, thread, 'message', id,
           fathom_message_search_document(body),
           to_tsvector('simple', fathom_message_search_document(body)),
           created_at
    FROM messages
    ON CONFLICT (tenant, thread, source, source_id) DO NOTHING
    """,
    """
    DO $$
    DECLARE
        item TEXT[];
    BEGIN
        FOREACH item SLICE 1 IN ARRAY ARRAY[
            ARRAY['threads','created_at'], ARRAY['threads','updated_at'], ARRAY['threads','archived_at'], ARRAY['threads','deleted_at'],
            ARRAY['actors','created_at'], ARRAY['actors','updated_at'],
            ARRAY['memberships','joined_at'], ARRAY['memberships','departed_at'],
            ARRAY['tasks','started_at'], ARRAY['tasks','ended_at'], ARRAY['tasks','created_at'], ARRAY['tasks','updated_at'], ARRAY['tasks','deleted_at'],
            ARRAY['messages','sanitized_at'], ARRAY['messages','created_at'], ARRAY['messages','deleted_at'],
            ARRAY['events','created_at'],
            ARRAY['artifacts','created_at'], ARRAY['artifacts','deleted_at'],
            ARRAY['scripts','created_at'], ARRAY['scripts','updated_at'], ARRAY['scripts','deleted_at'],
            ARRAY['script_versions','created_at'],
            ARRAY['policies','created_at'], ARRAY['policies','updated_at'],
            ARRAY['contexts','created_at'], ARRAY['contexts','expires_at'],
            ARRAY['requests','created_at'], ARRAY['requests','expires_at'],
            ARRAY['jobs','locked_at'], ARRAY['jobs','available_at'], ARRAY['jobs','created_at'], ARRAY['jobs','updated_at']
        ]
        LOOP
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = item[1]
                  AND column_name = item[2]
                  AND data_type <> 'timestamp with time zone'
            ) THEN
                EXECUTE format(
                    'ALTER TABLE %I.%I ALTER COLUMN %I TYPE TIMESTAMPTZ USING NULLIF(%I::text, '''')::timestamptz',
                    current_schema(), item[1], item[2], item[2]
                );
            END IF;
        END LOOP;

        FOREACH item SLICE 1 IN ARRAY ARRAY[
            ARRAY['threads','metadata'],
            ARRAY['actors','skills'], ARRAY['actors','metadata'],
            ARRAY['memberships','metadata'],
            ARRAY['tasks','progress'], ARRAY['tasks','plan'], ARRAY['tasks','metadata'],
            ARRAY['messages','body'], ARRAY['messages','labels'], ARRAY['messages','metadata'],
            ARRAY['events','payload'], ARRAY['events','metadata'],
            ARRAY['artifacts','labels'], ARRAY['artifacts','metadata'],
            ARRAY['scripts','metadata'],
            ARRAY['script_versions','metadata'],
            ARRAY['policies','retention'], ARRAY['policies','labels'], ARRAY['policies','sanitizers'], ARRAY['policies','memories'], ARRAY['policies','artifacts'], ARRAY['policies','metadata'],
            ARRAY['contexts','references'], ARRAY['contexts','budget'], ARRAY['contexts','filters'], ARRAY['contexts','metadata'],
            ARRAY['requests','response'], ARRAY['requests','metadata'],
            ARRAY['jobs','payload'], ARRAY['jobs','metadata']
        ]
        LOOP
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = item[1]
                  AND column_name = item[2]
                  AND data_type <> 'jsonb'
            ) THEN
                EXECUTE format(
                    'ALTER TABLE %I.%I ALTER COLUMN %I TYPE JSONB USING %I::jsonb',
                    current_schema(), item[1], item[2], item[2]
                );
            END IF;
        END LOOP;
    END
    $$;
    """,
)


VIEWS: Final[Tuple[str, ...]] = ()
BOOTSTRAP_STEPS: Final[Tuple[str, ...]] = ()


# Add deterministic CHECK constraints for the enum-like text columns. Names
# are stable so the lookup against pg_constraint keeps the migration idempotent across re-runs and across schemas.
ENUM_CHECKS: Final[Tuple[str, ...]] = (
    """
    DO $body$
    DECLARE
        spec TEXT[];
        constraint_name TEXT;
    BEGIN
        FOREACH spec SLICE 1 IN ARRAY ARRAY[
            ARRAY['threads','state','active,paused,completed,archived,deleted'],
            ARRAY['actors','kind','human,agent,coordinator,team,tool,system'],
            ARRAY['memberships','role','owner,requester,responder,coordinator,delegate,observer,system'],
            ARRAY['memberships','scope','thread,task,actor,team,system'],
            ARRAY['tasks','kind','agent,tool,coordination,delegation,fathom,script,clarification,analysis'],
            ARRAY['tasks','state','queued,running,blocked,waiting,succeeded,failed,cancelled,expired,deleted'],
            ARRAY['tasks','code','completed,worker_lost,user_cancelled,timeout,policy_blocked,validation_failed,unknown_error'],
            ARRAY['messages','kind','request,instruction,question,answer,progress,result,note,notice'],
            ARRAY['messages','audience','thread,task,actor,team,system'],
            ARRAY['events','kind','thread.created,actor.joined,task.opened,task.started,task.blocked,task.waiting,task.delegated,task.succeeded,task.failed,task.cancelled,task.expired,task.deleted,message.recorded,content.classified,content.sanitized,artifact.linked,context.built,job.scheduled,job.rescheduled,job.completed,job.failed,job.abandoned,client.disconnected,recovery.lost'],
            ARRAY['events','source','interaction,fathom,policy,worker,artifact,client,recovery'],
            ARRAY['artifacts','kind','screenshot,trace,structured_log,script,report,context_debug,tool_output,model_output'],
            ARRAY['artifacts','backend','local,object'],
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
            constraint_name := format('chk_%s_%s_enum', spec[1], spec[2]);
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = current_schema()
                  AND t.relname = spec[1]
                  AND c.conname = constraint_name
            ) THEN
                EXECUTE format(
                    'ALTER TABLE %I.%I ADD CONSTRAINT %I CHECK (%I = ANY (string_to_array(%L, '','')))',
                    current_schema(), spec[1], constraint_name, spec[2], spec[3]
                );
            END IF;
        END LOOP;
    END
    $body$;
    """,
)

SEARCH_EVIDENCE: Final[Tuple[str, ...]] = (
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
    UPDATE search AS target
    SET document = source_document.document,
        vector = to_tsvector('simple', source_document.document)
    FROM (
        SELECT tenant, id, fathom_message_search_document(body) AS document
        FROM messages
        WHERE deleted_at IS NULL
    ) AS source_document
    WHERE target.tenant = source_document.tenant
      AND target.source = 'message'
      AND target.source_id = source_document.id
    """,
)


EVENT_KIND_ENUM_V2: Final[str] = """
    DO $body$
    BEGIN
        ALTER TABLE events DROP CONSTRAINT IF EXISTS chk_events_kind_enum;
        ALTER TABLE events ADD CONSTRAINT chk_events_kind_enum CHECK (
            kind = ANY (
                string_to_array(
                    'thread.created,thread.archived,thread.unarchived,thread.deleted,actor.joined,task.opened,task.started,task.blocked,task.waiting,task.delegated,task.succeeded,task.failed,task.cancelled,task.expired,task.deleted,message.recorded,content.classified,content.sanitized,artifact.linked,context.built,job.scheduled,job.rescheduled,job.completed,job.failed,job.abandoned,client.disconnected,recovery.lost',
                    ','
                )
            )
        );
    END
    $body$;
    """


# First public Postgres migration baseline. After this reaches any shared
# database, keep this step immutable and add future changes as v2+ migrations.
MIGRATION_STEPS: Final[Tuple[PostgresMigrationStep, ...]] = (
    PostgresMigrationStep(
        version=1,
        name="baseline",
        statements=(
            *TABLES,
            *STRUCTURAL_CONSTRAINTS,
            *INDEXES,
            *BACKFILLS,
            *MIGRATIONS,
            *ENUM_CHECKS,
            *SEARCH_EVIDENCE,
        ),
    ),
    PostgresMigrationStep(
        version=2,
        statements=(EVENT_KIND_ENUM_V2,),
        name="thread_lifecycle_event_kinds",
    ),
)


SCHEMA_VERSION: Final[int] = MIGRATION_STEPS[-1].version
