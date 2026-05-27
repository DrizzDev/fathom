from enum import StrEnum


class NodeName(StrEnum):
    """Standardized names for graph nodes."""

    END = "__end__"
    SCAN = "scan"
    RECORD = "record"
    GROUND = "ground"
    VERIFY = "verify"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    OBSERVE = "observe"
    NAVIGATE = "navigate"
    SUPERVISE = "supervise"
    BFS_ROUTE = "bfs_route"


class RouteCause(StrEnum):
    """Reasons emitted by graph routing decisions for structured observability."""

    DEFAULT = "default"
    CANCELLED = "cancelled"
    SHOULD_RETRY = "should_retry"
    TERMINAL_COMPLETION = "terminal_completion"
    NON_TERMINAL_COMPLETION = "non_terminal_completion"
