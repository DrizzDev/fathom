from __future__ import annotations

from typing import Final

from pypika import Table

ACTORS: Final[Table] = Table("actors")
THREADS: Final[Table] = Table("threads")
MEMBERSHIPS: Final[Table] = Table("memberships")
TASKS: Final[Table] = Table("tasks")
MESSAGES: Final[Table] = Table("messages")
EVENTS: Final[Table] = Table("events")
ARTIFACTS: Final[Table] = Table("artifacts")
CONTEXTS: Final[Table] = Table("contexts")
JOBS: Final[Table] = Table("jobs")
POLICIES: Final[Table] = Table("policies")
REQUESTS: Final[Table] = Table("requests")
SCRIPTS: Final[Table] = Table("scripts")
SCRIPT_VERSIONS: Final[Table] = Table("script_versions")
SEQUENCES: Final[Table] = Table("sequences")
SEARCH: Final[Table] = Table("search")
