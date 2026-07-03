from __future__ import annotations

from enum import StrEnum
from typing import Final, FrozenSet


class TelemetryEnvelopeKey(StrEnum):
    """
    Reserved keys the publish path injects into the telemetry envelope.
    """

    EVENT = "event"
    LEVEL = "level"
    COLOR = "color"
    SOURCE = "source"
    MESSAGE = "message"
    TIMESTAMP = "timestamp"
    REQUEST_ID = "requestId"
    SESSION_ID = "session_id"


GUARDED_ENVELOPE_KEYS: Final[FrozenSet[TelemetryEnvelopeKey]] = frozenset(
    {
        TelemetryEnvelopeKey.EVENT,
        TelemetryEnvelopeKey.LEVEL,
        TelemetryEnvelopeKey.COLOR,
        TelemetryEnvelopeKey.SOURCE,
        TelemetryEnvelopeKey.MESSAGE,
        TelemetryEnvelopeKey.TIMESTAMP,
        TelemetryEnvelopeKey.REQUEST_ID,
        TelemetryEnvelopeKey.SESSION_ID,
    }
)

# Prefix applied to caller-supplied context keys that collide with reserved
# envelope keys; the suffix counter is appended only on secondary collisions.
TELEMETRY_COLLISION_PREFIX: Final[str] = "context_"

# Counter starting value used to disambiguate secondary collisions when both
# `key` and `context_key` are already present in the context payload.
TELEMETRY_COLLISION_COUNTER_START: Final[int] = 2
