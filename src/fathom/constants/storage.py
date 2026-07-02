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
    POSTGRES = "postgres"


class PostgresMigrationMode(StrEnum):
    """
    Supported startup behaviors for Postgres interaction schema handling.
    """

    APPLY = "apply"
    VALIDATE = "validate"
    DISABLED = "disabled"


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


class RetentionClass(StrEnum):
    """
    Supported retention windows applied to artifact and message rows.

    Maps to tenant-policy retention configuration; the value is the canonical string persisted on disk.
    """

    LONG = "long"
    SHORT = "short"
    STANDARD = "standard"


# Default backend when neither host nor settings selects one.
INTERACTION_DEFAULT_BACKEND: Final[InteractionBackend] = InteractionBackend.POSTGRES

# Postgres tunable.
INTERACTION_POSTGRES_DEFAULT_PORT: Final[int] = 5432
INTERACTION_POSTGRES_DEFAULT_SCHEMA: Final[str] = "fathom"
INTERACTION_POSTGRES_DEFAULT_POOL_MIN_SIZE: Final[int] = 1
INTERACTION_POSTGRES_DEFAULT_POOL_MAX_SIZE: Final[int] = 10

# Postgres `statement_timeout` setting applied per session. Value is in
# milliseconds because that is the unit the server-side parameter expects.

INTERACTION_POSTGRES_APPLICATION_NAME: Final[str] = "fathom"
INTERACTION_POSTGRES_DEFAULT_DATABASE: Final[str] = "fathom"
INTERACTION_POSTGRES_DEFAULT_STATEMENT_TIMEOUT: Final[int] = 10_000

INTERACTION_POSTGRES_DEFAULT_MIGRATION_MODE: Final[PostgresMigrationMode] = (
    PostgresMigrationMode.APPLY
)
INTERACTION_POSTGRES_ORM_APP: Final[str] = "models"
INTERACTION_POSTGRES_ORM_CONNECTION: Final[str] = "default"
INTERACTION_POSTGRES_ORM_ENGINE: Final[str] = "tortoise.backends.asyncpg"

# Slow-query observability threshold, in milliseconds. Queries exceeding this
# emit a structured log record. Default keeps the log channel quiet for
# hot-path reads while surfacing any single query that drifts into the
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
