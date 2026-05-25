from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fathom.constants import ActionType
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState


class SQLiteMemoryProviderTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers persistent screen-memory filtering.
    """

    @staticmethod
    def __action() -> Action:
        """
        Build a representative tap action.
        """

        return Action(
            action_type=ActionType.TAP,
            target="Swiggy app icon",
            rationale="Open app.",
            confidence=1.0,
        )

    @staticmethod
    def __screen(*, activity: str) -> ScreenState:
        """
        Build a screen state with the supplied activity.
        """

        return ScreenState(
            activity=activity,
            timestamp=1,
            activity_hash="a" * 16,
            visual_hash="b" * 16,
        )

    async def test_launcher_screen_experiences_are_not_retrieved(self) -> None:
        """
        Launcher memories must not be exposed as current app guidance.
        """

        with tempfile.TemporaryDirectory() as directory:
            provider = SQLiteMemoryProvider(database_path=Path(directory) / "memory.db")
            screen = self.__screen(activity="com.google.android.apps.nexuslauncher")

            await provider.store_observation(screen=screen, description="Opening Swiggy.")
            await provider.store_experience(
                visual_hash=screen.visual_hash,
                action=self.__action(),
                success=True,
            )

            knowledge = await provider.retrieve_knowledge(visual_hash=screen.visual_hash)

        self.assertIsNone(knowledge["description"])
        self.assertEqual(knowledge["previous_actions"], [])

    async def test_app_screen_experiences_are_retrieved(self) -> None:
        """
        Non-launcher screen memories remain available.
        """

        with tempfile.TemporaryDirectory() as directory:
            provider = SQLiteMemoryProvider(database_path=Path(directory) / "memory.db")
            screen = self.__screen(activity="in.swiggy.android")

            await provider.store_observation(screen=screen, description="Swiggy home.")
            await provider.store_experience(
                visual_hash=screen.visual_hash,
                action=self.__action(),
                success=True,
            )

            knowledge = await provider.retrieve_knowledge(visual_hash=screen.visual_hash)

        self.assertEqual(knowledge["description"], "Swiggy home.")
        self.assertEqual(len(knowledge["previous_actions"]), 1)
