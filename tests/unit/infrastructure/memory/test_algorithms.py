"""
Unit tests for the pure graph algorithms.
"""

from __future__ import annotations

from fathom.infrastructure.memory.algorithms import GraphAlgorithms
from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode


class TestGraphAlgorithms:
    """
    Pathfinding, cycles, components, and diameter on small in-memory graphs.
    """

    @staticmethod
    def __edge(source: str, dest: str) -> GraphEdge:
        return GraphEdge(
            source_hash=source, destination_hash=dest, action_type="tap", action_target="t"
        )

    @classmethod
    def __diamond(cls) -> tuple[dict[str, GraphNode], dict[str, list[GraphEdge]]]:
        # a -> b -> c, plus a -> c shortcut
        nodes = {h: GraphNode(visual_hash=h, activity="X") for h in ("a", "b", "c")}
        edges = {
            "a": [cls.__edge("a", "b"), cls.__edge("a", "c")],
            "b": [cls.__edge("b", "c")],
        }
        return nodes, edges

    def test_find_path_returns_shortest(self) -> None:
        nodes, edges = self.__diamond()
        path = GraphAlgorithms.find_path(nodes=nodes, edges=edges, start_hash="a", end_hash="c")
        assert path is not None
        assert [h for h, _ in path] == ["a", "c"]

    def test_find_path_none_when_unreachable(self) -> None:
        nodes, edges = self.__diamond()
        nodes["d"] = GraphNode(visual_hash="d", activity="X")
        assert (
            GraphAlgorithms.find_path(nodes=nodes, edges=edges, start_hash="c", end_hash="d")
            is None
        )

    def test_find_all_paths_enumerates_routes(self) -> None:
        nodes, edges = self.__diamond()
        paths = GraphAlgorithms.find_all_paths(
            nodes=nodes, edges=edges, start_hash="a", end_hash="c"
        )
        assert len(paths) == 2

    def test_detect_cycles_finds_loop(self) -> None:
        nodes, edges = self.__diamond()
        edges.setdefault("c", []).append(self.__edge("c", "a"))
        assert GraphAlgorithms.detect_cycles(edges=edges, start_nodes=list(nodes))

    def test_connected_components_are_symmetric_on_a_chain(self) -> None:
        _, edges = self.__diamond()
        assert GraphAlgorithms.connected_component(edges=edges, start_hash="a") == {"a", "b", "c"}
        assert GraphAlgorithms.reverse_connected_component(edges=edges, end_hash="c") == {
            "a",
            "b",
            "c",
        }

    def test_diameter_is_longest_shortest_path(self) -> None:
        nodes = {h: GraphNode(visual_hash=h, activity="X") for h in ("a", "b", "c")}
        edges = {"a": [self.__edge("a", "b")], "b": [self.__edge("b", "c")]}
        assert GraphAlgorithms.diameter(nodes=nodes, edges=edges) == 2
