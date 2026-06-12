from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.constants.exploration import CriticalScreenKind, RecommendationLevel
from fathom.core.services.exporter.report import (
    ExplorationReportGenerator,
    MarkdownReportRenderer,
)
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.actions import Action
from fathom.schemas.report import (
    ActivityCoverage,
    ComponentSummary,
    CoverageSummary,
    CriticalScreen,
    ExplorationReport,
    NavigationCycle,
    Recommendation,
    ReportMetadata,
    ScreenInsight,
)
from fathom.schemas.screens import ScreenState

_GENERATED_AT = "2026-06-12T00:00:00"


class _FakeProvider:
    """In-memory IMemoryProvider stand-in whose writes are no-ops."""

    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None:
        return None

    async def update_rich_description(self, visual_hash: str, rich_description: str) -> None:
        return None

    async def store_transition(
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None:
        return None

    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None:
        return None

    async def retrieve_knowledge(self, visual_hash: str) -> Dict[str, Any]:
        return {}

    async def retrieve_transitions(self, visual_hash: str) -> List[Dict[str, Any]]:
        return []

    async def get_all_knowledge(self) -> Dict[str, Any]:
        return {}

    async def get_all_screens(self) -> List[Dict[str, Any]]:
        return []

    async def get_all_transitions(self) -> List[Dict[str, Any]]:
        return []


class TestExplorationReportGenerator(unittest.IsolatedAsyncioTestCase):
    """The generator analyses a knowledge graph into a typed report."""

    __HASHES = {
        "home": "0000000000000000",
        "s1": "ffffffffffffffff",
        "s2": "0f0f0f0f0f0f0f0f",
        "s3": "f0f0f0f0f0f0f0f0",
        "s4": "00ff00ff00ff00ff",
        "s5": "ff00ff00ff00ff00",
    }

    @staticmethod
    def __screen(*, key: str, visual_hash: str) -> ScreenState:
        return ScreenState(
            activity=f"com.app/.{key.title()}",
            timestamp=0,
            activity_hash=f"act-{key}",
            visual_hash=visual_hash,
        )

    async def __graph(self) -> KnowledgeGraph:
        graph = KnowledgeGraph(provider=_FakeProvider())
        for key, visual_hash in self.__HASHES.items():
            await graph.add_screen(
                state=self.__screen(key=key, visual_hash=visual_hash), description=key
            )
        # Home twice more so it ranks as the most visited screen.
        await graph.add_screen(state=self.__screen(key="home", visual_hash=self.__HASHES["home"]))

        # Home fans out to five screens (a hub) and s1 loops back (a cycle).
        for key in ("s1", "s2", "s3", "s4", "s5"):
            await graph.record_transition(
                source_hash=self.__HASHES["home"],
                action=Action(
                    action_type=ActionType.TAP, rationale="r", natural_language_target=key
                ),
                destination_hash=self.__HASHES[key],
            )
        await graph.record_transition(
            source_hash=self.__HASHES["s1"],
            action=Action(action_type=ActionType.BACK, rationale="back"),
            destination_hash=self.__HASHES["home"],
        )
        return graph

    async def test_report_reflects_graph_metrics(self) -> None:
        graph = await self.__graph()
        report = ExplorationReportGenerator(graph=graph).generate(
            workflow="wf", package="com.app", generated_at=_GENERATED_AT, duration=12.5
        )

        self.assertEqual(report.metadata.workflow, "wf")
        self.assertEqual(report.metadata.generated_at, _GENERATED_AT)
        self.assertEqual(report.coverage.screens, graph.node_count)
        self.assertEqual(report.coverage.cycles, len(graph.detect_cycles()))
        # Home has the most visits, so it heads the ranking.
        self.assertEqual(report.most_visited[0].description, "home")
        self.assertGreaterEqual(report.most_visited[0].outgoing, 5)

    async def test_home_is_reported_as_a_hub(self) -> None:
        graph = await self.__graph()
        report = ExplorationReportGenerator(graph=graph).generate(
            workflow="wf", package="com.app", generated_at=_GENERATED_AT
        )

        hubs = [
            screen for screen in report.critical_screens if screen.kind == CriticalScreenKind.HUB
        ]
        self.assertTrue(hubs)
        self.assertGreaterEqual(hubs[0].connectivity, 5)

    async def test_recommendations_are_never_empty(self) -> None:
        graph = await self.__graph()
        report = ExplorationReportGenerator(graph=graph).generate(
            workflow="wf", package="com.app", generated_at=_GENERATED_AT
        )
        self.assertTrue(report.recommendations)


class TestMarkdownReportRenderer(unittest.TestCase):
    """The renderer turns a typed report into structured Markdown."""

    @staticmethod
    def __report() -> ExplorationReport:
        return ExplorationReport(
            metadata=ReportMetadata(
                workflow="wf", package="com.app", generated_at=_GENERATED_AT, duration=12.5
            ),
            coverage=CoverageSummary(
                screens=4,
                transitions=5,
                visits=9,
                activities=3,
                unexplored=1,
                coverage=75.0,
                diameter=3,
                cycles=1,
            ),
            most_visited=[
                ScreenInsight(
                    hash="home",
                    activity="com.app/.Home",
                    description="Home",
                    visits=4,
                    outgoing=5,
                    inbound=1,
                    in_cycle=True,
                )
            ],
            critical_screens=[
                CriticalScreen(
                    name="Home",
                    activity="com.app/.Home",
                    kind=CriticalScreenKind.HUB,
                    connectivity=6,
                    forward_reach=4,
                    backward_reach=2,
                )
            ],
            activities=[ActivityCoverage(activity="Home", screens=1)],
            cycles=[NavigationCycle(length=2, screens=["Home", "Search"])],
            components=ComponentSummary(count=1, largest=4),
            recommendations=[
                Recommendation(level=RecommendationLevel.WARNING, message="Low coverage detected.")
            ],
        )

    def test_render_includes_all_sections(self) -> None:
        markdown = MarkdownReportRenderer().render(report=self.__report())

        for heading in (
            "# Exploration Report",
            "## Coverage",
            "## Activities",
            "## Most Visited Screens",
            "## Critical Screens",
            "## Navigation Cycles",
            "## Recommendations",
        ):
            self.assertIn(heading, markdown)

        self.assertIn("com.app", markdown)
        self.assertIn("75.0%", markdown)
        self.assertIn("[WARNING] Low coverage detected.", markdown)
        self.assertIn("Home -> Search", markdown)

    def test_empty_sections_are_omitted(self) -> None:
        report = self.__report()
        report.activities = []
        report.cycles = []

        markdown = MarkdownReportRenderer().render(report=report)

        self.assertNotIn("## Activities", markdown)
        self.assertNotIn("## Navigation Cycles", markdown)
        self.assertIn("## Coverage", markdown)


if __name__ == "__main__":
    unittest.main()
