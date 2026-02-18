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
    NAVIGATE = "navigate"
    BFS_ROUTE = "bfs_route"
