from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from fathom.constants import ActionType
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.screens import ScreenState


class _FakeProvider:
    """In-memory IMemoryProvider stand-in: writes are no-ops, reads serve canned rows."""

    def __init__(
        self,
        *,
        screens: Optional[List[Dict[str, Any]]] = None,
        transitions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.__screens = screens or []
        self.__transitions = transitions or []

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
        return self.__screens

    async def get_all_transitions(self) -> List[Dict[str, Any]]:
        return self.__transitions


class TestKnowledgeGraph(unittest.IsolatedAsyncioTestCase):
    """The knowledge graph dedups screens, merges transitions, and builds scan context."""

    def setUp(self) -> None:
        self.__graph = KnowledgeGraph(provider=_FakeProvider())

    @staticmethod
    def __screen(
        *,
        visual_hash: str,
        activity: str = "com.app/.Feed",
        activity_hash: str = "act",
        xml_hash: Optional[str] = None,
        interaction_hash: Optional[str] = None,
    ) -> ScreenState:
        return ScreenState(
            activity=activity,
            timestamp=0,
            activity_hash=activity_hash,
            visual_hash=visual_hash,
            xml_hash=xml_hash,
            interaction_hash=interaction_hash,
        )

    @staticmethod
    def __tap(*, target: str, x: int = 100, y: int = 200) -> Action:
        return Action(
            action_type=ActionType.TAP,
            rationale="explore",
            natural_language_target=target,
            bounds=Bounds(x=x, y=y, width=10, height=10),
            region="content",
            element_category="content_item",
        )

    async def test_add_screen_counts_visits(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))
        node = await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))

        self.assertEqual(self.__graph.node_count, 1)
        self.assertEqual(node.visit_count, 2)

    async def test_add_screen_merges_fuzzy_duplicate(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="ffffffffffffffff"))
        await self.__graph.add_screen(state=self.__screen(visual_hash="fffffffffffffffe"))

        self.assertEqual(self.__graph.node_count, 1)

    async def test_record_transition_dedups_by_target(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))
        await self.__graph.add_screen(
            state=self.__screen(visual_hash="bbbbbbbbbbbbbbbb", activity_hash="actB")
        )

        for _ in range(2):
            await self.__graph.record_transition(
                source_hash="aaaaaaaaaaaaaaaa",
                action=self.__tap(target="Card"),
                destination_hash="bbbbbbbbbbbbbbbb",
            )

        neighbors = self.__graph.get_neighbors(visual_hash="aaaaaaaaaaaaaaaa")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].count, 2)

    async def test_get_tried_actions_excludes_back(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=self.__tap(target="Card"),
            destination_hash="bbbbbbbbbbbbbbbb",
        )
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=Action(action_type=ActionType.BACK, rationale="back"),
            destination_hash="cccccccccccccccc",
        )

        tried = self.__graph.get_tried_actions(visual_hash="aaaaaaaaaaaaaaaa")

        self.assertEqual(len(tried), 1)
        self.assertEqual(tried[0][1], "Card")

    async def test_count_category_taps_aggregates_activity(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=self.__tap(target="First card", x=100, y=200),
            destination_hash="bbbbbbbbbbbbbbbb",
        )
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=self.__tap(target="Second card", x=100, y=500),
            destination_hash="cccccccccccccccc",
        )

        count = self.__graph.count_category_taps(
            visual_hash="aaaaaaaaaaaaaaaa", category="content_item"
        )
        self.assertEqual(count, 2)

    async def test_get_stats(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))
        await self.__graph.add_screen(
            state=self.__screen(visual_hash="bbbbbbbbbbbbbbbb", activity_hash="actB")
        )
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=self.__tap(target="Card"),
            destination_hash="bbbbbbbbbbbbbbbb",
        )

        stats = self.__graph.get_stats()
        self.assertEqual(stats["unique_screens"], 2)
        self.assertEqual(stats["total_transitions"], 1)

    async def test_build_exploration_context_lists_tried_and_forbidden(self) -> None:
        await self.__graph.add_screen(
            state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"), description="Feed screen"
        )
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=self.__tap(target="Home tab"),
            destination_hash="bbbbbbbbbbbbbbbb",
        )

        context = self.__graph.build_exploration_context(current_hash="aaaaaaaaaaaaaaaa")

        self.assertIn("ALREADY TRIED ON THIS SCREEN:", context)
        self.assertIn("Home tab", context)
        self.assertIn("FORBIDDEN TARGETS", context)

    async def test_build_exploration_context_fresh_screen(self) -> None:
        await self.__graph.add_screen(state=self.__screen(visual_hash="aaaaaaaaaaaaaaaa"))

        context = self.__graph.build_exploration_context(current_hash="aaaaaaaaaaaaaaaa")

        self.assertIn("this screen is fresh", context)

    async def test_find_path_across_transitions(self) -> None:
        for visual_hash, activity_hash in (
            ("aaaaaaaaaaaaaaaa", "actA"),
            ("bbbbbbbbbbbbbbbb", "actB"),
            ("cccccccccccccccc", "actC"),
        ):
            await self.__graph.add_screen(
                state=self.__screen(visual_hash=visual_hash, activity_hash=activity_hash)
            )
        await self.__graph.record_transition(
            source_hash="aaaaaaaaaaaaaaaa",
            action=self.__tap(target="to B"),
            destination_hash="bbbbbbbbbbbbbbbb",
        )
        await self.__graph.record_transition(
            source_hash="bbbbbbbbbbbbbbbb",
            action=self.__tap(target="to C"),
            destination_hash="cccccccccccccccc",
        )

        path = self.__graph.find_path(start_hash="aaaaaaaaaaaaaaaa", end_hash="cccccccccccccccc")

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(
            [node for node, _ in path],
            ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"],
        )

    async def test_load_hydrates_from_provider(self) -> None:
        provider = _FakeProvider(
            screens=[
                {
                    "visual_hash": "aaaaaaaaaaaaaaaa",
                    "activity": "com.app/.A",
                    "description": "Screen A",
                    "first_seen": 1,
                    "last_seen": 2,
                    "visit_count": 3,
                    "rich_description": None,
                    "activity_hash": "actA",
                    "xml_hash": None,
                    "interaction_hash": None,
                }
            ],
            transitions=[
                {
                    "source_hash": "aaaaaaaaaaaaaaaa",
                    "destination_hash": "bbbbbbbbbbbbbbbb",
                    "action_type": "tap",
                    "action_target": "Card",
                    "coord_bucket": "2_4",
                    "coord_region": "content",
                    "element_category": "content_item",
                    "count": 2,
                    "first_seen": 1,
                    "last_seen": 2,
                }
            ],
        )
        graph = KnowledgeGraph(provider=provider)

        await graph.load()

        self.assertEqual(graph.node_count, 1)
        self.assertEqual(graph.edge_count, 1)
        self.assertEqual(graph.get_screen(visual_hash="aaaaaaaaaaaaaaaa").visit_count, 3)  # type: ignore[union-attr]
