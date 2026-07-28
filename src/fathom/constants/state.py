from enum import StrEnum
from typing import Final, FrozenSet, Tuple

LOOP_BACK_CONFIDENCE: Final[float] = 0.9
LOOP_SCROLL_CONFIDENCE: Final[float] = 0.8
LOOP_HOME_CONFIDENCE: Final[float] = 0.7

LOOP_BACK_RATIONALE: Final[str] = "Loop detected (screen repeating). Forcing BACK to break context."
LOOP_SCROLL_RATIONALE: Final[str] = (
    "Loop detected (screen repeating). Forcing SCROLL to reveal new state."
)
LOOP_HOME_RATIONALE: Final[str] = "Loop detected (screen repeating). Forcing HOME to reset agent."
LOOP_SCROLL_ACTION_TYPES: Final[FrozenSet[str]] = frozenset(
    {
        "scroll",
        "swipe",
        "swipe_up",
        "swipe_down",
        "swipe_left",
        "swipe_right",
    }
)


class CompletionReason(StrEnum):
    """
    Standard reasons for workflow completion or failure.
    """

    FAILED = "Failed"
    CANCELLED = "Cancelled"
    STUCK = "Stuck: No progress"
    MAX_STEPS = "Max steps reached"
    SUCCESS = "Completed successfully"
    OPERATOR_ABORTED = "Aborted by operator"
    INTERVENTION_REQUIRED = "Human intervention required"
    USER_DIRECTIVE = "Marked complete via user directive"
    NOT_EXECUTABLE = "Request is not an executable UI task"
    RETRY_BUDGET_EXHAUSTED = "Planner retry budget exhausted"
    ACTION_BLOCKED = "Action blocked: repeated without progress"
    UNSATISFIABLE = "Unsatisfiable: criterion observed refuted"


class VerifyMode(StrEnum):
    """
    Verification prompt and acceptance contract for a VERIFY turn.
    """

    SUB_GOAL = "SUB_GOAL"
    FULL_INTENT = "FULL_INTENT"
    PENDING_FINAL_COMMIT = "PENDING_FINAL_COMMIT"


# Completion reasons that must terminate the graph immediately (route to END, never to VERIFY) and surface as ``RunOutcome.FAILED`` to callers.
# The router in ``builder.py`` and the outcome assembler in ``intent.py`` MUST both consult this single source so new fatal reasons cannot be missed in one place but not the other.
TERMINAL_COMPLETION_REASONS: Final[Tuple[str, ...]] = (
    CompletionReason.STUCK.value,
    CompletionReason.FAILED.value,
    CompletionReason.MAX_STEPS.value,
    CompletionReason.CANCELLED.value,
    CompletionReason.UNSATISFIABLE.value,
    CompletionReason.ACTION_BLOCKED.value,
    CompletionReason.OPERATOR_ABORTED.value,
    CompletionReason.INTERVENTION_REQUIRED.value,
    CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
)


class RunOutcome(StrEnum):
    """
    Terminal state of the graph executor for a single run.
    """

    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class CommonStateKey(StrEnum):
    """
    Common state keys shared across all graph states.
    """

    # Configuration
    MAX_STEPS = "MAX_STEPS"

    # Execution
    STEP_NUMBER = "STEP_NUMBER"
    IS_COMPLETE = "IS_COMPLETE"
    COMPLETION_REASON = "COMPLETION_REASON"
    FAILURE_DIAGNOSTIC = "FAILURE_DIAGNOSTIC"

    # Artifacts
    CAPTURE = "CAPTURE"
    SCREEN_STATE = "SCREEN_STATE"
    IS_NEW_SCREEN = "IS_NEW_SCREEN"
    SCREEN_OBSERVATION = "SCREEN_OBSERVATION"

    # Analysis
    ANALYSIS = "ANALYSIS"

    # Execution Result
    STEP_RESULT = "STEP_RESULT"

    # Metrics
    ANALYSIS_DURATION = "ANALYSIS_DURATION"
    GROUNDING_DURATION = "GROUNDING_DURATION"
    EXECUTION_DURATION = "EXECUTION_DURATION"


class IntentStateKey(StrEnum):
    """
    Intent-specific state keys for IntentGraphState.
    """

    # Configuration
    INTENT = "INTENT"
    USE_XML = "USE_XML"

    # Execution
    SHOULD_RETRY = "SHOULD_RETRY"
    INJECTED_CONTEXT = "INJECTED_CONTEXT"
    VERIFY_MODE = "VERIFY_MODE"

    # Artifacts
    ELEMENTS = "ELEMENTS"
    XML_CONTENT = "XML_CONTENT"

    # Analysis
    PLAN = "PLAN"
    PLANNED_STEP = "PLANNED_STEP"

    # Execution coordination across SUPERVISE / EXECUTE / OBSERVE
    EXECUTION_CONTEXT = "EXECUTION_CONTEXT"

    # Measured evidence carried to the completion turn
    BINDING = "BINDING"
    EFFECT_READING = "EFFECT_READING"

    # Sub-goal state propagation (for global checkpoint persistence)
    CURRENT_SUB_GOAL_INDEX = "CURRENT_SUB_GOAL_INDEX"
    AGENT_STATE_CHECKPOINT = "AGENT_STATE_CHECKPOINT"

    # History
    STEP_RESULTS = "STEP_RESULTS"

    # Post-action activity captured in EXECUTE, consumed in RECORD
    POST_ACTIVITY = "POST_ACTIVITY"


class PlanMetadataKey(StrEnum):
    """
    Stable keys for fields the planner writes into PlanResult.metadata.
    """

    ANALYSIS = "analysis_result"
    OBSERVATION = "observation"


class ExplorationStateKey(StrEnum):
    """
    Exploration-specific state keys for ExplorationGraphState.
    """

    # BFS State
    BFS_PHASE = "BFS_PHASE"
    BFS_QUEUE = "BFS_QUEUE"

    PENDING_NAV = "PENDING_NAV"
    CURRENT_PATH = "CURRENT_PATH"

    ROOT_HASH = "ROOT_HASH"
    SCANNING_HASH = "SCANNING_HASH"
    VISITED_HASHES = "VISITED_HASHES"

    # Scan artifacts
    ACTION = "ACTION"
    CONTENT_EXHAUSTED = "CONTENT_EXHAUSTED"

    # History
    STEP_RESULTS = "STEP_RESULTS"
