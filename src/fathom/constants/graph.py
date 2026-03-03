from enum import StrEnum


class NodeName(StrEnum):
    """
    Standardized names for graph nodes.
    """

    END = "__end__"
    SCAN = "scan"
    RECORD = "record"
    GROUND = "ground"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    VERIFY = "verify"
    NAVIGATE = "navigate"
    BFS_ROUTE = "bfs_route"
