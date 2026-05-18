from enum import StrEnum


class NodeName(StrEnum):
    """
    Standardized names for graph nodes.
    """

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
