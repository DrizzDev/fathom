from __future__ import annotations

from typing import Final

from pypika import Table

ACTORS: Final[Table] = Table("actors")
THREADS: Final[Table] = Table("threads")
MEMBERSHIPS: Final[Table] = Table("memberships")

TASKS: Final[Table] = Table("tasks")
EVENTS: Final[Table] = Table("events")
MESSAGES: Final[Table] = Table("messages")
CONTEXTS: Final[Table] = Table("contexts")
ARTIFACTS: Final[Table] = Table("artifacts")

JOBS: Final[Table] = Table("jobs")
SEARCH: Final[Table] = Table("search")
POLICIES: Final[Table] = Table("policies")
REQUESTS: Final[Table] = Table("requests")
SEQUENCES: Final[Table] = Table("sequences")

SCRIPTS: Final[Table] = Table("scripts")
SCRIPT_VERSIONS: Final[Table] = Table("script_versions")
