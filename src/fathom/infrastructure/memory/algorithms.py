"""
Pure pathfinding, reachability, and cycle algorithms over a screen graph.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, TypeVar

from fathom.constants.exploration import ALL_PATHS_SEARCH_MAX_DEPTH, PATH_SEARCH_MAX_DEPTH


class AdjacencyEdge(Protocol):
    """
    Read-only view of a transition edge's destination.
    """

    @property
    def destination_hash(self) -> str:
        """Canonical hash of the screen this edge leads to."""


EdgeT = TypeVar("EdgeT", bound=AdjacencyEdge)


class GraphAlgorithms:
    """
    Stateless graph algorithms over a screen-node adjacency map.

    Each method receives the current nodes and edges so it stays correct
    across knowledge-graph reloads or in-place cache mutation. Hashes are
    expected to be already resolved to their canonical form.
    """

    @staticmethod
    def find_path(
        *,
        nodes: Mapping[str, object],
        edges: Mapping[str, Sequence[EdgeT]],
        start_hash: str,
        end_hash: str,
        max_depth: int = PATH_SEARCH_MAX_DEPTH,
    ) -> Optional[List[Tuple[str, Optional[EdgeT]]]]:
        """
        Shortest path between two screens via BFS as [(node, edge_taken), ...],
        with edge None for the start node, or None when unreachable.
        """

        if start_hash not in nodes or end_hash not in nodes:
            return None
        if start_hash == end_hash:
            return [(start_hash, None)]

        queue: deque[Tuple[str, List[Tuple[str, Optional[EdgeT]]]]] = deque()
        visited: Set[str] = set()

        initial_path: List[Tuple[str, Optional[EdgeT]]] = [(start_hash, None)]
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
                    queue.append((next_hash, path + [(next_hash, edge)]))

        return None

    @staticmethod
    def find_all_paths(
        *,
        nodes: Mapping[str, object],
        edges: Mapping[str, Sequence[EdgeT]],
        start_hash: str,
        end_hash: str,
        max_depth: int = ALL_PATHS_SEARCH_MAX_DEPTH,
    ) -> List[List[Tuple[str, Optional[EdgeT]]]]:
        """
        Every path between two screens up to max_depth via DFS.
        """

        if start_hash not in nodes or end_hash not in nodes:
            return []

        all_paths: List[List[Tuple[str, Optional[EdgeT]]]] = []

        def walk(
            current: str,
            path: List[Tuple[str, Optional[EdgeT]]],
            visited: Set[str],
            depth: int,
        ) -> None:
            if depth > max_depth:
                return
            if current == end_hash:
                all_paths.append(path[:])
                return
            for edge in edges.get(current, []):
                next_hash = edge.destination_hash
                if next_hash not in visited:
                    visited.add(next_hash)
                    path.append((next_hash, edge))
                    walk(next_hash, path, visited, depth + 1)
                    path.pop()
                    visited.remove(next_hash)

        walk(start_hash, [(start_hash, None)], {start_hash}, 0)
        return all_paths

    @staticmethod
    def detect_cycles(
        *,
        edges: Mapping[str, Sequence[AdjacencyEdge]],
        start_nodes: Sequence[str],
    ) -> List[List[str]]:
        """
        Cycles reachable from start_nodes via DFS; each cycle is a list of node
        hashes whose first and last entries match.
        """

        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def walk(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for edge in edges.get(node, []):
                next_node = edge.destination_hash
                if next_node not in visited:
                    walk(next_node, path)
                elif next_node in rec_stack:
                    cycle_start = path.index(next_node)
                    cycles.append(path[cycle_start:] + [next_node])

            path.pop()
            rec_stack.remove(node)

        for node in start_nodes:
            if node not in visited:
                walk(node, [])

        return cycles

    @staticmethod
    def connected_component(
        *,
        edges: Mapping[str, Sequence[AdjacencyEdge]],
        start_hash: str,
    ) -> Set[str]:
        """
        All nodes forward-reachable from start_hash, inclusive.
        """

        reachable: Set[str] = {start_hash}
        queue: deque[str] = deque([start_hash])

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
        edges: Mapping[str, Sequence[AdjacencyEdge]],
        end_hash: str,
    ) -> Set[str]:
        """
        All nodes that can reach end_hash, inclusive.
        """

        reverse_edges: Dict[str, List[str]] = {}
        for source, edge_list in edges.items():
            for edge in edge_list:
                reverse_edges.setdefault(edge.destination_hash, []).append(source)

        can_reach: Set[str] = {end_hash}
        queue: deque[str] = deque([end_hash])

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
        nodes: Mapping[str, object],
        edges: Mapping[str, Sequence[AdjacencyEdge]],
    ) -> Optional[int]:
        """
        Longest shortest-path between any two nodes, or None when there is none.
        """

        if not nodes:
            return None

        max_distance = 0
        node_hashes = list(nodes.keys())

        for start in node_hashes:
            for end in node_hashes:
                if start == end:
                    continue
                path = GraphAlgorithms.find_path(
                    nodes=nodes, edges=edges, start_hash=start, end_hash=end
                )
                if path:
                    max_distance = max(max_distance, len(path) - 1)

        return max_distance if max_distance > 0 else None
