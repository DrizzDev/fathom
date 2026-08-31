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

import json
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

OUTCOME_PREFIX = "PLUMB_OUTCOME "
RUN_ID_VARIABLE = "PLUMB_RUN_ID"
SCHEMA_VERSION = 1


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
