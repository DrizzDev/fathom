from enum import StrEnum


class NodeName(StrEnum):
    """
    Standardized names for graph nodes.
    """

    GROUND = "ground"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    RECORD = "record"
    END = "__end__"


class GraphKey(StrEnum):
    """
    Standardized keys for graph state.
    """

    INPUT = "input"
    INJECTED_CONTEXT = "injected_context"
    IS_COMPLETE = "is_complete"
    SHOULD_RETRY = "should_retry"
    PLANNED_STEP = "planned_step"
