"""
The benchmark trajectory stream.

A benchmark host runs Fathom as a subprocess and needs the run's shape as data, not
as prose in a log: how many steps were planned and executed, whether a human was ever
asked for anything, and how the run ended. Structured log lines carry some of that,
but their event names are Fathom's internal vocabulary and a host would be parsing an
interface nobody promised to keep.

This module emits a small, explicit contract on stdout instead, and only when a host
asks for it by setting ``PLUMB_RUN_ID``. Ordinary interactive runs print nothing extra.

Two line kinds:

* newline-delimited trajectory events, whose ``sequence`` is contiguous from zero so a
  reader can detect truncation rather than assume completeness;
* one final ``PLUMB_OUTCOME`` line carrying the machine-readable result.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from logging import getLogger
from typing import Any, Final, Mapping, Optional

from fathom.core.exceptions import (
    ProviderExhaustedError,
    WorkflowCancelledError,
    WorkflowTimeoutError,
)

OUTCOME_PREFIX = "PLUMB_OUTCOME "
RUN_ID_VARIABLE = "PLUMB_RUN_ID"
SCHEMA_VERSION = 1

logger = getLogger(__name__)


class OutcomeStatus(StrEnum):
    """
    How an execution attempt ended, in the host's vocabulary rather than Fathom's.

    Deliberately separate from ``StrategyStatus`` and ``CompletionReason``: those
    describe what the agent did and are free to change, while these five are a promise
    made to a process that cannot import Fathom to discover what changed.

    None of them says whether the task succeeded. A run that finished and failed its
    goal still COMPLETED - the host verifies the goal from the device, and an attempt
    that ended on its own terms is exactly the attempt it wants to score. The other
    four are the cases where scoring the model would be scoring the weather: the
    provider refused to serve the run, the clock ran out, an operator stopped it, or
    the harness broke.
    """

    COMPLETED = "COMPLETED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    PROVIDER_EXHAUSTED = "PROVIDER_EXHAUSTED"
    HARNESS_ERROR = "HARNESS_ERROR"


#: Fathom's workflow status strings translated into the contract. ``"failed"`` maps to
#: COMPLETED on purpose: the workflow ran to its own conclusion and reported that the
#: goal was not met, which is a result, not an incident.
#:
#: ``WorkflowResult`` defaults ``status`` to ``"unknown"`` and it is absent here on
#: purpose too. A status nobody set is a harness defect, and translating it to
#: COMPLETED would restore precisely the hardcoded answer this table exists to remove.
_RESULT_STATUSES: Final[Mapping[str, OutcomeStatus]] = {
    "completed": OutcomeStatus.COMPLETED,
    "failed": OutcomeStatus.COMPLETED,
    "cancelled": OutcomeStatus.CANCELLED,
}


def status_for_result(*, status: str) -> OutcomeStatus:
    """
    Translate a finished workflow's status into the contract.

    An unrecognised status reports HARNESS_ERROR rather than guessing. The host treats
    that as an invalid cell and excludes it, which is the right answer for a run whose
    ending Fathom could not name: excluding a good run costs one cell, while scoring a
    run of unknown provenance corrupts the comparison it lands in.
    """

    resolved = _RESULT_STATUSES.get(status.strip().lower())
    if resolved is not None:
        return resolved

    logger.warning(
        "Workflow reported a status the benchmark contract does not name",
        extra={
            "component": "runtime.plumb_stream",
            "event": "plumb.outcome.status.unmapped",
            "workflow.status": status,
        },
    )
    return OutcomeStatus.HARNESS_ERROR


def status_for_exception(*, exception: BaseException) -> OutcomeStatus:
    """
    Translate a run that raised instead of returning a result.

    Ordered most specific first. Provider exhaustion is checked before the timeouts
    because a provider that stops answering is often observed as a call that never
    returned, and attributing that to the clock would hide the one failure the host
    must not score against the model.
    """

    if isinstance(exception, (asyncio.CancelledError, KeyboardInterrupt, WorkflowCancelledError)):
        return OutcomeStatus.CANCELLED

    if isinstance(exception, ProviderExhaustedError):
        return OutcomeStatus.PROVIDER_EXHAUSTED

    if isinstance(exception, (WorkflowTimeoutError, TimeoutError)):
        return OutcomeStatus.TIMED_OUT

    return OutcomeStatus.HARNESS_ERROR


class TrajectoryStream:
    """
    Emits the benchmark trajectory for one run, or nothing at all when unrequested.
    """

    def __init__(self, *, run_id: Optional[str] = None) -> None:
        """
        Bind the run this stream describes; absent a run id the stream is inert.
        """

        self.__run_id = run_id or os.environ.get(RUN_ID_VARIABLE) or ""
        self.__sequence = 0

    @property
    def enabled(self) -> bool:
        """
        Whether a host asked for the stream.
        """

        return bool(self.__run_id)

    def event(self, *, kind: str, payload: Optional[dict[str, Any]] = None) -> None:
        """
        Emit one trajectory event, numbering it into the contiguous sequence.
        """

        if not self.enabled:
            return

        record = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "sequence": self.__sequence,
            "run_id": self.__run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "type": kind,
            "payload": payload or {},
        }
        self.__sequence += 1
        self.__write(record)

    def outcome(self, *, payload: dict[str, Any]) -> None:
        """
        Emit the terminal outcome line the host parses for the run's verdict.
        """

        if not self.enabled:
            return

        sys.stdout.write(OUTCOME_PREFIX + json.dumps(payload, default=str) + "\n")
        sys.stdout.flush()

    @staticmethod
    def __write(record: dict[str, Any]) -> None:
        """
        Write one NDJSON line, flushed, so a reader streaming stdout sees it promptly.
        """

        sys.stdout.write(json.dumps(record, default=str) + "\n")
        sys.stdout.flush()
