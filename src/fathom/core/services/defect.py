"""
Post-run defect analysis over the explored screen graph.
"""

from __future__ import annotations

from typing import List, Sequence

from fathom.infrastructure.memory.knowledge_graph import GraphNode, KnowledgeGraph
from fathom.interfaces.defect import DefectRepositoryPort, ScreenDefectDetectorPort
from fathom.schemas.defect import ScreenSnapshot


class DefectAnalysisService:
    """
    Runs screen-level defect detectors over every unique screen after a crawl.
    """

    def __init__(
        self,
        *,
        detectors: Sequence[ScreenDefectDetectorPort],
        repository: DefectRepositoryPort,
    ) -> None:
        self.__detectors = detectors
        self.__repository = repository

    async def analyze(self, *, graph: KnowledgeGraph, session: str) -> int:
        """
        Detects and persists defects for each unique screen; returns the count persisted.
        """

        persisted = 0
        for node in graph.nodes.values():
            snapshot = self.__snapshot(node=node)
            for detector in self.__detectors:
                for defect in await detector.inspect_screen(snapshot=snapshot):
                    await self.__repository.record(session=session, defect=defect)
                    persisted += 1
        return persisted

    @staticmethod
    def __snapshot(*, node: GraphNode) -> ScreenSnapshot:
        """
        Projects a graph node into the read-only view detectors consume.
        """

        texts: List[str] = [text for text in (node.description, node.rich_description) if text]
        return ScreenSnapshot(
            screen=node.visual_hash,
            activity=node.activity,
            texts=texts,
        )
