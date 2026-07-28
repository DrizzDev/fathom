from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import aiosqlite

from fathom.constants import ActionType
from fathom.constants.turn.binding import BindingState
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.schemas.actions import Action
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.experience import Experience
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

    async def test_typed_outcome_round_trips_through_the_store(self) -> None:
        """
        A stored typed outcome persists every reinforcement-bearing field.
        """

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            provider = SQLiteMemoryProvider(database_path=path)

            await provider.store_outcome(
                experience=Experience(
                    workflow="6cfc5fd2",
                    session="exec-1",
                    screen="b" * 16,
                    action=ActionType.TAP,
                    target="ADD button",
                    executed=True,
                    transitioned=ActionEffectStatus.PROGRESS,
                    advanced=False,
                    binding=BindingState.BOUND,
                )
            )

            async with aiosqlite.connect(path) as db:
                cursor = await db.execute(
                    "SELECT workflow, action, executed, transitioned, advanced, binding FROM outcome"
                )
                rows = await cursor.fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "6cfc5fd2")
        self.assertEqual(rows[0][1], "tap")
        self.assertEqual(rows[0][2], 1)
        self.assertEqual(rows[0][3], "progress")
        self.assertEqual(rows[0][4], 0)
        self.assertEqual(rows[0][5], "BOUND")
