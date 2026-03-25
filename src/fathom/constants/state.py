from enum import StrEnum


class CompletionReason(StrEnum):
    """
    Standard reasons for workflow completion or failure.
    """

    FAILED = "Failed"
    CANCELLED = "Cancelled"
    MAX_STEPS = "Max steps reached"
    SUCCESS = "Completed successfully"
    STUCK = "Stuck: Recovery exhausted"
    INTERVENTION_REQUIRED = "Human intervention required"
    USER_DIRECTIVE = "Marked complete via user directive"


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

    # Artifacts
    CAPTURE = "CAPTURE"
    SCREEN_STATE = "SCREEN_STATE"
    IS_NEW_SCREEN = "IS_NEW_SCREEN"

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

    # Artifacts
    ELEMENTS = "ELEMENTS"
    XML_CONTENT = "XML_CONTENT"

    # Analysis
    PLAN = "PLAN"
    PLANNED_STEP = "PLANNED_STEP"

    # Sub-goal state propagation (for global checkpoint persistence)
    CURRENT_SUB_GOAL_INDEX = "CURRENT_SUB_GOAL_INDEX"
    AGENT_STATE_CHECKPOINT = "AGENT_STATE_CHECKPOINT"

    # History
    STEP_RESULTS = "STEP_RESULTS"

    # Post-action activity captured in EXECUTE, consumed in RECORD
    POST_ACTIVITY = "POST_ACTIVITY"


class ExplorationStateKey(StrEnum):
    """
    Exploration-specific state keys for ExplorationGraphState.
    """

    # BFS State
    BFS_PHASE = "BFS_PHASE"
    BFS_QUEUE = "BFS_QUEUE"
    VISITED_HASHES = "VISITED_HASHES"
    CURRENT_PATH = "CURRENT_PATH"
    PENDING_NAV = "PENDING_NAV"
    SCANNING_HASH = "SCANNING_HASH"
    ROOT_HASH = "ROOT_HASH"

    # Scan artifacts
    ACTION = "ACTION"
    CONTENT_EXHAUSTED = "CONTENT_EXHAUSTED"

    # History
    STEP_RESULTS = "STEP_RESULTS"
