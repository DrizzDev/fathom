"""
Pure pathfinding, reachability, and cycle algorithms over a screen graph.
"""

from __future__ import annotations

from collections import deque
from logging import getLogger
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode

logger = getLogger(__name__)


class GraphAlgorithms:
    """
    Stateless graph algorithms over a screen-node adjacency map.

    Each method receives the current ``nodes`` and ``edges`` so it stays
    correct across knowledge-graph reloads or in-place cache mutation.
    Hashes are expected to be already resolved to their canonical form.
    """

    @staticmethod
    def find_path(
        *,
        nodes: Dict[str, "GraphNode"],
        edges: Dict[str, List["GraphEdge"]],
        start_hash: str,
        end_hash: str,
        max_depth: int = 50,
    ) -> Optional[List[Tuple[str, Optional["GraphEdge"]]]]:
        """
        Shortest path between two screens via BFS, as [(node, edge_taken), ...]
        with edge ``None`` for the start node, or ``None`` when unreachable.
        """

        if start_hash not in nodes:
            logger.warning(f"Start hash {start_hash} not in graph")
            return None
        if end_hash not in nodes:
            logger.warning(f"End hash {end_hash} not in graph")
            return None
        if start_hash == end_hash:
            return [(start_hash, None)]

        queue: deque[Tuple[str, List[Tuple[str, Optional["GraphEdge"]]]]] = deque()
        visited: Set[str] = set()

        initial_path: List[Tuple[str, Optional["GraphEdge"]]] = [(start_hash, None)]
        queue.append((start_hash, initial_path))
        visited.add(start_hash)

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for edge in edges.get(current, []):
                next_hash = edge.destination_hash

                if next_hash == end_hash:
                    return path + [(next_hash, edge)]

                if next_hash not in visited:
                    visited.add(next_hash)
                    new_path = path + [(next_hash, edge)]
                    queue.append((next_hash, new_path))

        return None

    @staticmethod
    def find_all_paths(
        *,
        nodes: Dict[str, "GraphNode"],
        edges: Dict[str, List["GraphEdge"]],
        start_hash: str,
        end_hash: str,
        max_depth: int = 10,
    ) -> List[List[Tuple[str, Optional["GraphEdge"]]]]:
        """
        Every path between two screens up to ``max_depth`` via DFS.
        """

        if start_hash not in nodes or end_hash not in nodes:
            return []

        all_paths: List[List[Tuple[str, Optional["GraphEdge"]]]] = []

        def dfs(
            current: str,
            target: str,
            path: List[Tuple[str, Optional["GraphEdge"]]],
            visited: Set[str],
            depth: int,
        ) -> None:
            if depth > max_depth:
                return

            if current == target:
                all_paths.append(path[:])
                return

            for edge in edges.get(current, []):
                next_hash = edge.destination_hash
                if next_hash not in visited:
                    visited.add(next_hash)
                    path.append((next_hash, edge))
                    dfs(next_hash, target, path, visited, depth + 1)
                    path.pop()
                    visited.remove(next_hash)

        initial_path: List[Tuple[str, Optional["GraphEdge"]]] = [(start_hash, None)]
        visited_set: Set[str] = {start_hash}
        dfs(start_hash, end_hash, initial_path, visited_set, 0)

        return all_paths

    @staticmethod
    def detect_cycles(
        *,
        edges: Dict[str, List["GraphEdge"]],
        start_nodes: List[str],
    ) -> List[List[str]]:
        """
        Cycles reachable from ``start_nodes`` via DFS; each cycle is a list of
        node hashes whose first and last entries match.
        """

        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        parent_map: Dict[str, str] = {}

        def dfs_visit(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for edge in edges.get(node, []):
                next_node = edge.destination_hash

                if next_node not in visited:
                    parent_map[next_node] = node
                    dfs_visit(next_node, path)
                elif next_node in rec_stack:
                    cycle_start_idx = path.index(next_node)
                    cycle = path[cycle_start_idx:] + [next_node]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in start_nodes:
            if node not in visited:
                dfs_visit(node, [])

        return cycles

    @staticmethod
    def connected_component(
        *,
        edges: Dict[str, List["GraphEdge"]],
        start_hash: str,
    ) -> Set[str]:
        """All nodes forward-reachable from ``start_hash`` (inclusive)."""

        reachable: Set[str] = set()
        queue: deque[str] = deque([start_hash])
        reachable.add(start_hash)

        while queue:
            current = queue.popleft()
            for edge in edges.get(current, []):
                next_node = edge.destination_hash
                if next_node not in reachable:
                    reachable.add(next_node)
                    queue.append(next_node)

        return reachable

    @staticmethod
    def reverse_connected_component(
        *,
        edges: Dict[str, List["GraphEdge"]],
        end_hash: str,
    ) -> Set[str]:
        """All nodes that can reach ``end_hash`` (inclusive)."""

        reverse_edges: Dict[str, List[str]] = {}
        for source, edge_list in edges.items():
            for edge in edge_list:
                dest = edge.destination_hash
                reverse_edges.setdefault(dest, []).append(source)

        can_reach: Set[str] = set()
        queue: deque[str] = deque([end_hash])
        can_reach.add(end_hash)

        while queue:
            current = queue.popleft()
            for source in reverse_edges.get(current, []):
                if source not in can_reach:
                    can_reach.add(source)
                    queue.append(source)

        return can_reach

    @staticmethod
    def diameter(
        *,
        nodes: Dict[str, "GraphNode"],
        edges: Dict[str, List["GraphEdge"]],
    ) -> Optional[int]:
        """Longest shortest-path between any two nodes, or ``None`` if none."""

        if not nodes:
            return None

        node_hashes = list(nodes.keys())
        max_distance = 0

        for start in node_hashes:
            for end in node_hashes:
                if start != end:
                    path = GraphAlgorithms.find_path(
                        nodes=nodes, edges=edges, start_hash=start, end_hash=end
                    )
                    if path:
                        distance = len(path) - 1
                        max_distance = max(max_distance, distance)

        return max_distance if max_distance > 0 else None
