from __future__ import annotations

import unittest
from typing import List
from unittest.mock import Mock

from fathom.constants.defect import DefectSignal
from fathom.core.defect.content import ContentDefectDetector
from fathom.core.services.defect import DefectAnalysisService
from fathom.infrastructure.memory.knowledge_graph import GraphNode
from fathom.interfaces.defect import DefectRepositoryPort
from fathom.schemas.defect import Defect


class _RecordingRepository(DefectRepositoryPort):
    """
    In-memory defect repository that records what the service persists.
    """

    def __init__(self) -> None:
        self.recorded: List[Defect] = []

    async def record(self, *, session: str, defect: Defect) -> None:
        self.recorded.append(defect)

    async def for_screen(self, *, session: str, screen: str) -> List[Defect]:
        return [defect for defect in self.recorded if defect.evidence.screen == screen]

    async def for_run(self, *, session: str) -> List[Defect]:
        return list(self.recorded)


class DefectAnalysisServiceTest(unittest.IsolatedAsyncioTestCase):
    """
    Verifies the post-run pass runs detectors over each screen and persists findings.
    """

    async def test_persists_content_defects_per_screen(self) -> None:
        """
        Each unique screen is inspected and its defects are persisted.
        """

        graph = Mock(
            nodes={
                "clean": GraphNode(
                    visual_hash="clean", activity="com.app/.Home", description="A tidy home screen"
                ),
                "dirty": GraphNode(
                    visual_hash="dirty",
                    activity="com.app/.Detail",
                    rich_description="Body shows Lorem ipsum dolor",
                ),
            }
        )
        repository = _RecordingRepository()
        service = DefectAnalysisService(detectors=[ContentDefectDetector()], repository=repository)

        persisted = await service.analyze(graph=graph, session="wf-1")

        self.assertEqual(persisted, 1)
        self.assertEqual(len(repository.recorded), 1)
        self.assertEqual(repository.recorded[0].signal, DefectSignal.LOREM_IPSUM)
        self.assertEqual(repository.recorded[0].evidence.screen, "dirty")

    async def test_no_detectors_persists_nothing(self) -> None:
        """
        With no detectors the pass is a no-op.
        """

        graph = Mock(nodes={"a": GraphNode(visual_hash="a", activity="com.app/.Home")})
        repository = _RecordingRepository()

        persisted = await DefectAnalysisService(detectors=[], repository=repository).analyze(
            graph=graph, session="wf-1"
        )

        self.assertEqual(persisted, 0)
        self.assertEqual(repository.recorded, [])


if __name__ == "__main__":
    unittest.main()
