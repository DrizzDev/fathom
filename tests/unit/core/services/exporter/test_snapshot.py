from __future__ import annotations

import unittest
from unittest.mock import Mock

from fathom.core.services.exporter.snapshot import ExplorationSnapshotBuilder
from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode


class TestExplorationSnapshotBuilder(unittest.TestCase):
    """The builder maps a knowledge graph into a typed snapshot."""

    @staticmethod
    def __graph() -> Mock:
        home = GraphNode(
            visual_hash="home", activity="com.app/.Home", description="Home", visit_count=2
        )
        cart = GraphNode(
            visual_hash="cart", activity="com.app/.Cart", description=None, visit_count=1
        )
        edge = GraphEdge(
            source_hash="home",
            destination_hash="cart",
            action_type="tap",
            action_target="Cart",
            count=3,
        )
        return Mock(
            nodes={"home": home, "cart": cart},
            edges={"home": [edge]},
            get_stats=Mock(
                return_value={
                    "unique_screens": 2,
                    "total_transitions": 1,
                    "total_visits": 3,
                    "activities": ["com.app/.Cart", "com.app/.Home"],
                    "unexplored": 1,
                }
            ),
        )

    def test_build_maps_screens_transitions_and_stats(self) -> None:
        snapshot = ExplorationSnapshotBuilder().build(graph=self.__graph())

        self.assertEqual({screen.hash for screen in snapshot.screens}, {"home", "cart"})
        self.assertEqual(len(snapshot.transitions), 1)

        transition = snapshot.transitions[0]
        self.assertEqual(transition.source, "home")
        self.assertEqual(transition.destination, "cart")
        self.assertEqual(transition.action, "tap")
        self.assertEqual(transition.count, 3)

        self.assertEqual(snapshot.stats.screens, 2)
        self.assertEqual(snapshot.stats.transitions, 1)
        self.assertEqual(snapshot.stats.unexplored, 1)

    def test_blank_action_target_becomes_none(self) -> None:
        graph = self.__graph()
        graph.edges = {
            "home": [
                GraphEdge(
                    source_hash="home",
                    destination_hash="cart",
                    action_type="back",
                    action_target="",
                )
            ]
        }

        snapshot = ExplorationSnapshotBuilder().build(graph=graph)

        self.assertIsNone(snapshot.transitions[0].target)


if __name__ == "__main__":
    unittest.main()
