from enum import StrEnum


class NodeName(StrEnum):
    """
    Standardized names for graph nodes.
    """

    END = "__end__"
    RECORD = "record"
    GROUND = "ground"
    ANALYZE = "analyze"
    EXECUTE = "execute"


class GraphKey(StrEnum):
    """
    Standardized keys for graph state.
    """

    INPUT = "input"
    INJECTED_CONTEXT = "injected_context"

    IS_COMPLETE = "is_complete"
    SHOULD_RETRY = "should_retry"
    PLANNED_STEP = "planned_step"
