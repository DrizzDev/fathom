from __future__ import annotations

from enum import StrEnum
from typing import Final


class StorageBackend(StrEnum):
    """
    Canonical identifiers for artifact storage backends.
    """

    LOCAL = "LOCAL"
    CLOUD = "CLOUD"



class InteractionBackend(StrEnum):
    """
    Supported backend kinds for the durable interaction storage port.
    """

    NOOP = "noop"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class SQLiteJournalMode(StrEnum):
    """
    Supported SQLite journal modes.

    DELETE / TRUNCATE / PERSIST are rollback-journal modes that work safely on
    network-mounted filesystems (NFS, EFS, FUSE). MEMORY/OFF disable durable
    rollback. WAL is the high-throughput shared-memory journal mode and is
    only safe on local-disk deployments — see SQLiteInteractionConfiguration.
    """

    WAL = "WAL"
    OFF = "OFF"

    MEMORY = "MEMORY"
    PERSIST = "PERSIST"

    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"


class SQLiteSynchronous(StrEnum):
    """
    Supported SQLite synchronous PRAGMA values.
    """

    OFF = "OFF"
    FULL = "FULL"
    EXTRA = "EXTRA"
    NORMAL = "NORMAL"


class PostgresSslMode(StrEnum):
    """
    Supported SSL negotiation modes for the asyncpg Postgres connection.

    Mirrors libpq's `sslmode` semantics so an operator can demand TLS on a managed Postgres (RDS / Cloud SQL) without code changes.
    `prefer` tries SSL and falls back to plaintext; `require` mandates TLS; `verify-ca` / `verify-full` additionally validate the server certificate chain.
    """

    ALLOW = "allow"
    PREFER = "prefer"
    DISABLE = "disable"
    REQUIRE = "require"
    VERIFY_CA = "verify-ca"
    VERIFY_FULL = "verify-full"


class SqlParameterStyle(StrEnum):
    """
    Supported SQL placeholder syntaxes used by interaction storage adapters.
    """

    NUMBERED = "numbered"
    QUESTION_MARK = "question_mark"


class RetentionClass(StrEnum):
    """
    Supported retention windows applied to artifact and message rows.

    Maps to tenant-policy retention configuration; the value is the canonical string persisted on disk.
    """

    LONG = "long"
    SHORT = "short"
    STANDARD = "standard"


# Default backend when neither host nor settings selects one.
INTERACTION_DEFAULT_BACKEND: Final[InteractionBackend] = InteractionBackend.SQLITE

# SQLite tunable's. DELETE journal mode (rollback journal) is safe across local disks AND network-mounted filesystems (EFS, NFS, FUSE).
# WAL needs shared memory coordination and is unsafe on shared filesystems; hosts that run on local disks can opt into WAL explicitly via SQLiteInteractionConfiguration.
INTERACTION_SQLITE_SYNCHRONOUS: Final[SQLiteSynchronous] = SQLiteSynchronous.NORMAL
INTERACTION_SQLITE_JOURNAL_MODE: Final[SQLiteJournalMode] = SQLiteJournalMode.DELETE

# Time the writer waits for a contended lock before SQLITE_BUSY. Value is in
# milliseconds because that is the unit the SQLite PRAGMA accepts directly.
INTERACTION_SQLITE_BUSY_TIMEOUT: Final[int] = 5000
INTERACTION_SQLITE_FOREIGN_KEYS: Final[bool] = True
INTERACTION_SQLITE_TEMP_STORE: Final[str] = "MEMORY"

# Memory-mapped read window for the database file. Value is in bytes; default
# is 256 MiB, large enough that hot-page reads land in the OS page cache.
INTERACTION_SQLITE_MMAP_SIZE: Final[int] = 268_435_456  # 256 MiB

# Postgres tunable.
INTERACTION_POSTGRES_DEFAULT_PORT: Final[int] = 5432
INTERACTION_POSTGRES_DEFAULT_SCHEMA: Final[str] = "fathom"
INTERACTION_POSTGRES_DEFAULT_POOL_MIN_SIZE: Final[int] = 10
INTERACTION_POSTGRES_DEFAULT_POOL_MAX_SIZE: Final[int] = 50

# Postgres `statement_timeout` setting applied per session. Value is in
# milliseconds because that is the unit the server-side parameter expects.

INTERACTION_POSTGRES_APPLICATION_NAME: Final[str] = "fathom"
INTERACTION_POSTGRES_DEFAULT_DATABASE: Final[str] = "fathom"
INTERACTION_POSTGRES_DEFAULT_STATEMENT_TIMEOUT: Final[int] = 10_000

# Slow-query observability threshold in milliseconds; queries exceeding this
# emit a structured log record. Default 500 ms keeps the log channel quiet
# for hot-path reads while surfacing any single query that drifts into the
# user-visible-latency band.
INTERACTION_SLOW_QUERY_THRESHOLD: Final[int] = 500

# SSL negotiation mode applied to every Postgres connection in the pool. The default `prefer` matches libpq behavior:
# Try TLS, fall back to plaintext. Managed Postgres (RDS, Cloud SQL) typically demands `require` or stricter.
INTERACTION_POSTGRES_DEFAULT_SSL: Final[PostgresSslMode] = PostgresSslMode.PREFER

# Single-character escape used in `LIKE ... ESCAPE '\'` predicates so that
# wildcards in user-supplied search prefixes (e.g. `%`, `_`) remain literal.
INTERACTION_SQL_LIKE_ESCAPE_CHARACTER: Final[str] = "\\"

# SQL LIKE wildcard characters that must be neutralized when user-supplied substrings reach the predicate as literal matchers.
SQL_LIKE_WILDCARD_PERCENT: Final[str] = "%"
SQL_LIKE_WILDCARD_UNDERSCORE: Final[str] = "_"
