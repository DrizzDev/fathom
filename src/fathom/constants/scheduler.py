from __future__ import annotations

from enum import StrEnum
from typing import Final


class JobSchedulerKind(StrEnum):
    """
    Supported kinds for the background-job dispatcher port.
    """

    NOOP = "NOOP"
    IN_PROCESS = "IN_PROCESS"


# Default scheduler when no host-supplied configuration is provided.
JOB_SCHEDULER_DEFAULT_KIND: Final[JobSchedulerKind] = JobSchedulerKind.IN_PROCESS

# Polling cadence for the in-process scheduler. The loop sleeps this long when no job is claimable.
JOB_SCHEDULER_DEFAULT_POLL_INTERVAL: Final[int] = 1_000

# Maximum jobs claimed per scheduler tick before yielding back to the loop.
JOB_SCHEDULER_DEFAULT_BATCH_SIZE: Final[int] = 10

# Time a worker holds a claimed job before recovery considers it stale.
JOB_SCHEDULER_DEFAULT_LEASE: Final[int] = 60_000

# Backoff between retries after a transient handler failure.
JOB_SCHEDULER_DEFAULT_RETRY_BACKOFF: Final[int] = 5_000

# Number of attempts a job is retried before being marked permanently failed.
JOB_SCHEDULER_DEFAULT_MAX_ATTEMPTS: Final[int] = 5

# Cadence at which the scheduler runs stale-claim recovery, separate from the claim/dispatch poll interval.
JOB_SCHEDULER_DEFAULT_RECOVERY_INTERVAL: Final[int] = 30_000

# Backoff applied after the loop catches an unexpected exception before the next iteration runs again.
JOB_SCHEDULER_DEFAULT_FAILURE_BACKOFF: Final[int] = 5_000
