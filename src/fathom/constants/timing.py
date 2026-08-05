from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Tuple


class TimingPhase(StrEnum):
    """
    The instrumented compute phases of one agent step, plus the wait carved out separately.
    """

    WAIT = "wait"
    GROUND = "ground"
    VISION = "vision"
    RECORD = "record"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    OBSERVE = "observe"
    PLANNER = "planner"
    SUPERVISE = "supervise"


class TimingEvent(StrEnum):
    """
    Structured-log event names for per-step and per-run timing telemetry.
    """

    STEP = "step.timing"
    SUMMARY = "run.timing.summary"


class Scale(IntEnum):
    """
    Fixed numeric scales for timing conversions.
    """

    PERCENT = 100
    MILLIS_PER_SECOND = 1000


# Phases summed into agent compute; planner and vision are sub-durations, the wait is carved out.
COMPUTE_PHASES: Tuple[TimingPhase, ...] = (
    TimingPhase.GROUND,
    TimingPhase.ANALYZE,
    TimingPhase.SUPERVISE,
    TimingPhase.EXECUTE,
    TimingPhase.OBSERVE,
    TimingPhase.RECORD,
)
