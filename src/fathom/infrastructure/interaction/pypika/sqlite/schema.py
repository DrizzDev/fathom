from __future__ import annotations

from typing import Final, Tuple

from fathom.infrastructure.interaction.schema import AuditColumns

AUDIT_COLUMNS: Final[AuditColumns] = AuditColumns(
    timestamp_type="TEXT",
    metadata_type="TEXT",
    indent="    ",
)

THREAD = (
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
    archived_at TEXT,
"""
    + AUDIT_COLUMNS.created_updated_deleted()
    + """,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, creator) REFERENCES actors(tenant, id)
)
"""
)

ACTOR = (
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
    skills TEXT NOT NULL,
"""
    + AUDIT_COLUMNS.created_updated()
    + """,
    PRIMARY KEY (tenant, id)
)
"""
)

MEMBERSHIP = """
CREATE TABLE IF NOT EXISTS memberships (
    id TEXT NOT NULL,
    tenant TEXT NOT NULL,
    workspace TEXT,
    thread TEXT NOT NULL,
    actor TEXT NOT NULL,
    role TEXT NOT NULL,
    scope TEXT NOT NULL,
    joined_at TEXT NOT NULL,
    departed_at TEXT,
    metadata TEXT NOT NULL,
    PRIMARY KEY (tenant, id),
    UNIQUE (tenant, thread, actor),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
    FOREIGN KEY (tenant, actor) REFERENCES actors(tenant, id)
)
"""

TASK = (
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
    progress TEXT NOT NULL,
    plan TEXT NOT NULL,
    summary TEXT,
    started_at TEXT,
    ended_at TEXT,
    elapsed INTEGER,
"""
    + AUDIT_COLUMNS.created_updated_deleted()
    + """,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
    FOREIGN KEY (tenant, creator) REFERENCES actors(tenant, id),
    FOREIGN KEY (tenant, assignee) REFERENCES actors(tenant, id),
    FOREIGN KEY (tenant, parent) REFERENCES tasks(tenant, id),
    FOREIGN KEY (tenant, root) REFERENCES tasks(tenant, id),
    FOREIGN KEY (tenant, origin) REFERENCES messages(tenant, id) DEFERRABLE INITIALLY DEFERRED
)
"""
)

MESSAGE = (
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
    body TEXT NOT NULL,
    labels TEXT NOT NULL,
    sanitized_at TEXT,
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
"""
)

EVENT = (
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
    payload TEXT NOT NULL,
"""
    + AUDIT_COLUMNS.created()
    + """,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
    FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
    FOREIGN KEY (tenant, actor) REFERENCES actors(tenant, id)
)
"""
)

ARTIFACT = (
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
    labels TEXT NOT NULL,
"""
    + AUDIT_COLUMNS.created_deleted()
    + """,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
    FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
    FOREIGN KEY (tenant, producer) REFERENCES actors(tenant, id)
)
"""
)

SCRIPT = (
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
"""
)

SCRIPT_VERSION = (
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
"""
)

POLICY = (
    """
CREATE TABLE IF NOT EXISTS policies (
    id TEXT NOT NULL,
    tenant TEXT NOT NULL,
    workspace TEXT,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    region TEXT,
    retention TEXT NOT NULL,
    labels TEXT NOT NULL,
    sanitizers TEXT NOT NULL,
    memories TEXT NOT NULL,
    artifacts TEXT NOT NULL,
"""
    + AUDIT_COLUMNS.created_updated()
    + """,
    PRIMARY KEY (tenant, id)
)
"""
)

CONTEXT = (
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
    "references" TEXT NOT NULL,
    budget TEXT NOT NULL,
    filters TEXT NOT NULL,
    hash TEXT,
    provider TEXT,
    model TEXT,
    expires_at TEXT,
"""
    + AUDIT_COLUMNS.created()
    + """,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
    FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id),
    FOREIGN KEY (tenant, consumer) REFERENCES actors(tenant, id)
)
"""
)

REQUESTS = (
    """
CREATE TABLE IF NOT EXISTS requests (
    tenant TEXT NOT NULL,
    key TEXT NOT NULL,
    hash TEXT NOT NULL,
    state TEXT NOT NULL,
    response TEXT,
    expires_at TEXT NOT NULL,
"""
    + AUDIT_COLUMNS.created()
    + """,
    PRIMARY KEY (tenant, key)
)
"""
)

JOB = (
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
    locked_at TEXT,
    available_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    code TEXT,
    detail TEXT,
"""
    + AUDIT_COLUMNS.created_updated()
    + """,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id),
    FOREIGN KEY (tenant, task) REFERENCES tasks(tenant, id)
)
"""
)

EVENT_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_thread_sequence "
    "ON events(tenant, thread, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_events_task ON events(tenant, task, sequence)",
)

ARTIFACT_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_artifacts_thread ON artifacts(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(tenant, task)",
)

SCRIPT_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_scripts_thread ON scripts(tenant, thread, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_scripts_task ON scripts(tenant, task)",
    "CREATE INDEX IF NOT EXISTS idx_scripts_artifact ON scripts(tenant, artifact)",
    "CREATE INDEX IF NOT EXISTS idx_script_versions_script "
    "ON script_versions(tenant, script, version)",
    "CREATE INDEX IF NOT EXISTS idx_script_versions_artifact ON script_versions(tenant, artifact)",
)

POLICY_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_policies_lookup "
    "ON policies(tenant, COALESCE(workspace, ''), name)",
)

JOB_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(tenant, state, available_at, kind)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_thread ON jobs(tenant, thread)",
)

REQUEST_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_requests_expires ON requests(tenant, expires_at)",
)

CONTEXT_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_contexts_thread ON contexts(tenant, thread, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_contexts_task ON contexts(tenant, task, created_at)",
)

BASE_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_thread_sequence "
    "ON messages(tenant, thread, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_threads_tenant_state ON threads(tenant, state)",
    "CREATE INDEX IF NOT EXISTS idx_actors_tenant_kind ON actors(tenant, kind)",
    "CREATE INDEX IF NOT EXISTS idx_memberships_thread ON memberships(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_memberships_actor ON memberships(tenant, actor)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_thread ON tasks(tenant, thread)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(tenant, parent)",
    "CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(tenant, task, sequence)",
)

INDEXES = (
    *BASE_INDEXES,
    *EVENT_INDEXES,
    *ARTIFACT_INDEXES,
    *SCRIPT_INDEXES,
    *POLICY_INDEXES,
    *JOB_INDEXES,
    *REQUEST_INDEXES,
    *CONTEXT_INDEXES,
)


MESSAGE_SEARCH_FIELDS: Final[Tuple[str, ...]] = (
    "text",
    "message",
    "summary",
    "detail",
    "error",
    "progress",
    "note",
    "intent",
    "package",
    "status",
    "reason",
    "success",
    "steps",
    "workflow",
    "evidence",
)

MESSAGE_SEARCH_NEW_BODY: Final[str] = " || ' ' || ".join(
    f"COALESCE(json_extract(new.body, '$.{field}'), '')" for field in MESSAGE_SEARCH_FIELDS
)
MESSAGE_SEARCH_OLD_BODY: Final[str] = " || ' ' || ".join(
    f"COALESCE(json_extract(old.body, '$.{field}'), '')" for field in MESSAGE_SEARCH_FIELDS
)
MESSAGE_SEARCH_ROW_BODY: Final[str] = " || ' ' || ".join(
    f"COALESCE(json_extract(m.body, '$.{field}'), '')" for field in MESSAGE_SEARCH_FIELDS
)
MESSAGE_SEARCH_BODY: Final[str] = " || ' ' || ".join(
    f"COALESCE(json_extract(body, '$.{field}'), '')" for field in MESSAGE_SEARCH_FIELDS
)

# Generated, indexable text projection of the message body for FTS5. SQLite
# computes this column lazily on read (VIRTUAL); it occupies no extra storage.
MESSAGES_BODY_TEXT_COLUMN = (
    "ALTER TABLE messages "
    "ADD COLUMN body_text TEXT "
    f"GENERATED ALWAYS AS ({MESSAGE_SEARCH_BODY}) VIRTUAL"
)

# FTS5 virtual table indexed over the extracted message document. tenant and
# thread are stored unindexed so search results can be tenant-/thread-scoped.
SEARCH_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
    body_text,
    tenant UNINDEXED,
    thread UNINDEXED,
    tokenize='porter unicode61'
)
"""

# Triggers that keep the FTS5 index in sync with the source messages table.
SEARCH_TRIGGER_INSERT_TEMPLATE = """
CREATE TRIGGER IF NOT EXISTS search_insert AFTER INSERT ON messages BEGIN
    INSERT INTO search(rowid, body_text, tenant, thread)
    VALUES (new.rowid, __MESSAGE_SEARCH_NEW_BODY__, new.tenant, new.thread);
END
"""
SEARCH_TRIGGER_INSERT = SEARCH_TRIGGER_INSERT_TEMPLATE.replace(
    "__MESSAGE_SEARCH_NEW_BODY__", MESSAGE_SEARCH_NEW_BODY
)

SEARCH_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS search_delete AFTER DELETE ON messages BEGIN
    DELETE FROM search WHERE rowid = old.rowid;
END
"""

SEARCH_TRIGGER_UPDATE_TEMPLATE = """
CREATE TRIGGER IF NOT EXISTS search_update
AFTER UPDATE OF body ON messages BEGIN
    DELETE FROM search WHERE rowid = old.rowid;
    INSERT INTO search(rowid, body_text, tenant, thread)
    VALUES (new.rowid, __MESSAGE_SEARCH_NEW_BODY__, new.tenant, new.thread);
END
"""
SEARCH_TRIGGER_UPDATE = SEARCH_TRIGGER_UPDATE_TEMPLATE.replace(
    "__MESSAGE_SEARCH_NEW_BODY__", MESSAGE_SEARCH_NEW_BODY
)

# Backfill the FTS table for any messages persisted before the upgrade.
# Anti-join skips rows already mirrored, so a partial backfill that crashed
# mid-flight can be safely re-run without duplicating existing FTS rows or
# leaving the index permanently incomplete.
SEARCH_BACKFILL = (
    "INSERT INTO search(rowid, body_text, tenant, thread) "  # nosec B608
    f"SELECT m.rowid, {MESSAGE_SEARCH_ROW_BODY}, m.tenant, m.thread "
    "FROM messages m "
    "WHERE NOT EXISTS ("
    "    SELECT 1 FROM search f WHERE f.rowid = m.rowid"
    ")"
)

# Per-thread monotonic sequence allocator. Replaces the SELECT MAX(sequence)+1
# pattern in messages/events with an atomic single-statement allocation that
# scales without per-thread contention and ports cleanly to Postgres MVCC.
# `scope` is constrained to the two domain values the store ever writes,
# so a typo cannot silently introduce a third allocator stream.
SEQUENCES = """
CREATE TABLE IF NOT EXISTS sequences (
    tenant TEXT NOT NULL,
    thread TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('message', 'event')),
    value INTEGER NOT NULL,
    PRIMARY KEY (tenant, thread, scope),
    FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id)
)
"""

# Backfill seeds for the v10 migration so existing threads see the next
# allocation as MAX(existing) + 1 without any double-allocation hazard.
SEQUENCES_BACKFILL_MESSAGES = (
    "INSERT INTO sequences (tenant, thread, scope, value) "
    "SELECT tenant, thread, 'message', COALESCE(MAX(sequence), 0) "
    "FROM messages "
    "GROUP BY tenant, thread"
)
SEQUENCES_BACKFILL_EVENTS = (
    "INSERT INTO sequences (tenant, thread, scope, value) "
    "SELECT tenant, thread, 'event', COALESCE(MAX(sequence), 0) "
    "FROM events "
    "GROUP BY tenant, thread"
)


# Partial indexes that cover the hot WHERE deleted_at IS NULL paths used by the
# read APIs. Active-only filtering avoids scanning soft-deleted rows.
ACTIVE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_threads_active_updated "
    "ON threads(tenant, updated_at DESC) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_tasks_thread_active "
    "ON tasks(tenant, thread, created_at) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_thread_active "
    "ON artifacts(tenant, thread, created_at, id) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_messages_active "
    "ON messages(tenant, thread, sequence) WHERE deleted_at IS NULL",
)
