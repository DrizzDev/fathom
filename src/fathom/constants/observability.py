from __future__ import annotations

from enum import StrEnum


class CompletionEvent(StrEnum):
    """
    Structured log event names emitted across the completion-gate pipeline.
    """

    SUBGOAL_ADVANCED = "completion.subgoal.advanced"
    SUBGOAL_RETAINED = "completion.subgoal.retained"
    INTENT_COMPLETED = "completion.intent.completed"
    INTENT_PENDING = "completion.intent.pending"
    GATE_ADJUDICATED = "completion.gate.adjudicated"
    EVIDENCE_ASSESSED = "completion.evidence.assessed"
    CRITERION_OBSERVED = "completion.criterion.observed"


class SwipeEvent(StrEnum):
    """
    Structured log event names emitted across the swipe-gesture pipeline.
    """

    EFFECT_OBSERVED = "swipe.effect.observed"
    GESTURE_ADAPTED = "swipe.gesture.adapted"
    GESTURE_REJECTED = "swipe.gesture.rejected"
    GESTURE_DISPATCHED = "swipe.gesture.dispatched"


class ExecutorEvent(StrEnum):
    """
    Structured log event names emitted by the action executor at staging boundaries.
    """

    TRACE_STAGED = "executor.trace.staged"
    TRACE_SKIPPED = "executor.trace.skipped"
