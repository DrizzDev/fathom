"""Execution engine constants."""

from __future__ import annotations

from enum import StrEnum

# Visual hashing
VISUAL_HASH_LENGTH = 16

# Swipe and scroll distances (pixels)
DEFAULT_SWIPE_DISTANCE = 300
DEFAULT_SCROLL_DISTANCE = 200
BOUNDS_SWIPE_DISTANCE = 100

# Timing (milliseconds)
DEFAULT_SWIPE_DURATION = 500
DEFAULT_STABILITY_WAIT = 500  # Wait after action for screen to stabilize

# Retry configuration
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 500  # Base delay for exponential backoff


class SignalType(StrEnum):
    """HITL control signals."""
    
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    INJECT = "INJECT"
    ASK = "ASK"
    STOP = "STOP"
    CONTINUE = "CONTINUE"


class ExecutionPhase(StrEnum):
    """Execution DAG phases."""
    
    SIGNAL_CHECK = "signal_check"
    PERCEIVE = "perceive"
    REASON = "reason"
    ACT = "act"
    LEARN = "learn"
    CHECKPOINT = "checkpoint"
    EVALUATE = "evaluate"
