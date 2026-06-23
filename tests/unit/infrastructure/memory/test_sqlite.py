from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Optional

from fathom.constants import ActionType
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.screens import ScreenState


class TestSQLiteMemoryProvider(unittest.IsolatedAsyncioTestCase):
    """The provider persists screens, transitions, and experiences with upsert semantics."""

    def setUp(self) -> None:
        self.__tmp = tempfile.TemporaryDirectory()
        self.__provider = SQLiteMemoryProvider(database_path=Path(self.__tmp.name) / "knowledge.db")

    def tearDown(self) -> None:
        self.__tmp.cleanup()

    @staticmethod
    def __screen(
        *,
        visual_hash: str = "a1b2c3d4e5f60718",
        activity: str = "com.app/.Home",
        activity_hash: str = "acthash",
        xml_hash: Optional[str] = "xmlhash",
        interaction_hash: Optional[str] = "inthash",
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
    def __action(*, target_name: str = "Restaurant card") -> Action:
        return Action(
            action_type=ActionType.TAP,
            rationale="open the card",
            natural_language_target=target_name,
            bounds=Bounds(x=100, y=200, width=10, height=10),
            region="content",
            element_category="content_item",
        )

    async def test_store_observation_increments_visit_count(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(len(screens), 1)
        self.assertEqual(screens[0]["visit_count"], 2)

    async def test_store_observation_preserves_first_meaningful_description(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.store_observation(screen=self.__screen(), description="unknown")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["description"], "Home feed")

    async def test_get_all_screens_returns_mlsia_hashes(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description=None)

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["activity_hash"], "acthash")
        self.assertEqual(screens[0]["xml_hash"], "xmlhash")
        self.assertEqual(screens[0]["interaction_hash"], "inthash")

    async def test_store_transition_upserts_and_counts(self) -> None:
        await self.__provider.store_transition(
            source_hash="src", action=self.__action(), destination_hash="dst"
        )
        await self.__provider.store_transition(
            source_hash="src", action=self.__action(), destination_hash="dst"
        )

        transitions = await self.__provider.get_all_transitions()

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["count"], 2)
        self.assertEqual(transitions[0]["coord_region"], "content")
        self.assertEqual(transitions[0]["element_category"], "content_item")
        self.assertEqual(transitions[0]["coord_bucket"], "2_4")

    async def test_retrieve_knowledge_includes_transitions(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.store_transition(
            source_hash="a1b2c3d4e5f60718", action=self.__action(), destination_hash="dst"
        )

        knowledge = await self.__provider.retrieve_knowledge(visual_hash="a1b2c3d4e5f60718")

        self.assertEqual(knowledge["description"], "Home feed")
        self.assertEqual(len(knowledge["transitions"]), 1)
        self.assertEqual(knowledge["transitions"][0]["destination"], "dst")

    async def test_update_rich_description(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description=None)
        await self.__provider.update_rich_description(
            visual_hash="a1b2c3d4e5f60718", rich_description="## Elements\nCard"
        )

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["rich_description"], "## Elements\nCard")

    async def test_mark_exhausted_persists(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.mark_exhausted(visual_hash="a1b2c3d4e5f60718")

        screens = await self.__provider.get_all_screens()

        self.assertTrue(screens[0]["exhausted"])

    async def test_screens_default_to_not_exhausted(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertFalse(screens[0]["exhausted"])

    async def test_revisit_preserves_exhausted_flag(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.mark_exhausted(visual_hash="a1b2c3d4e5f60718")
        # A later visit upserts the row but must not clear the exhaustion flag.
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertTrue(screens[0]["exhausted"])

    async def test_set_relevance_persists(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.set_relevance(visual_hash="a1b2c3d4e5f60718", relevance="on_focus")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["relevance"], "on_focus")

    async def test_screens_default_to_unscoped_relevance(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["relevance"], "unscoped")

    async def test_revisit_preserves_relevance(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.set_relevance(visual_hash="a1b2c3d4e5f60718", relevance="on_focus")
        # A later visit upserts the row but must not reset the recorded relevance.
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["relevance"], "on_focus")

    async def test_set_category_persists(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.set_category(visual_hash="a1b2c3d4e5f60718", category="payment")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["category"], "payment")

    async def test_screens_default_to_other_category(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["category"], "other")

    async def test_revisit_preserves_category(self) -> None:
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")
        await self.__provider.set_category(visual_hash="a1b2c3d4e5f60718", category="detail")
        # A later visit upserts the row but must not reset the recorded category.
        await self.__provider.store_observation(screen=self.__screen(), description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["category"], "detail")

    async def test_structure_hash_persists_and_hydrates(self) -> None:
        state = ScreenState(
            activity="com.app/.Home",
            timestamp=0,
            activity_hash="acthash",
            visual_hash="a1b2c3d4e5f60718",
            structure_hash="structurehash01",
        )
        await self.__provider.store_observation(screen=state, description="Home feed")

        screens = await self.__provider.get_all_screens()

        self.assertEqual(screens[0]["structure_hash"], "structurehash01")

    async def test_readonly_provider_drops_writes(self) -> None:
        readonly = SQLiteMemoryProvider(
            database_path=Path(self.__tmp.name) / "readonly.db", readonly=True
        )

        await readonly.store_observation(screen=self.__screen(), description="Home feed")

        self.assertEqual(await readonly.get_all_screens(), [])

    async def test_launcher_screen_knowledge_is_not_retrieved(self) -> None:
        screen = self.__screen(activity="com.google.android.apps.nexuslauncher")
        await self.__provider.store_observation(screen=screen, description="Opening the app.")
        await self.__provider.store_experience(
            visual_hash=screen.visual_hash, action=self.__action(), success=True
        )

        knowledge = await self.__provider.retrieve_knowledge(visual_hash=screen.visual_hash)

        self.assertIsNone(knowledge["description"])
        self.assertEqual(knowledge["previous_actions"], [])

    async def test_app_screen_knowledge_is_retrieved(self) -> None:
        screen = self.__screen(activity="in.swiggy.android")
        await self.__provider.store_observation(screen=screen, description="App home.")

        knowledge = await self.__provider.retrieve_knowledge(visual_hash=screen.visual_hash)

        self.assertEqual(knowledge["description"], "App home.")
