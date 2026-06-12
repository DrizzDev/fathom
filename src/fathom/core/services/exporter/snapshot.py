"""
Builds a serialisable exploration snapshot from a live knowledge graph.
"""

from __future__ import annotations

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.exploration import (
    ExplorationSnapshot,
    ExplorationStats,
    ExploredScreen,
    ScreenTransition,
)


class ExplorationSnapshotBuilder:
    """
    Translates a knowledge graph into a typed, serialisable snapshot.
    """

    def build(self, *, graph: KnowledgeGraph) -> ExplorationSnapshot:
        """
        Produces a typed snapshot of the graph's screens, transitions, and stats.
        """

        screens = [
            ExploredScreen(
                hash=node.visual_hash,
                activity=node.activity,
                description=node.description,
                visits=node.visit_count,
            )
            for node in graph.nodes.values()
        ]
        transitions = [
            ScreenTransition(
                source=edge.source_hash,
                destination=edge.destination_hash,
                action=edge.action_type,
                target=edge.action_target or None,
                count=edge.count,
            )
            for edges in graph.edges.values()
            for edge in edges
        ]
        return ExplorationSnapshot(
            screens=screens, transitions=transitions, stats=self.__stats(graph=graph)
        )

    @staticmethod
    def __stats(*, graph: KnowledgeGraph) -> ExplorationStats:
        """
        Maps the graph's summary statistics into a typed model.
        """

        raw = graph.get_stats()
        return ExplorationStats(
            screens=int(raw.get("unique_screens", 0)),
            transitions=int(raw.get("total_transitions", 0)),
            visits=int(raw.get("total_visits", 0)),
            activities=list(raw.get("activities", [])),
            unexplored=int(raw.get("unexplored", 0)),
        )
