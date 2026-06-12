"""
Unit tests for KnowledgeGraph description preservation across revisits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fathom.infrastructure.memory.knowledge_graph import (
    KnowledgeGraph,
    _has_meaningful_description,
)
from fathom.schemas.screens import ScreenState


class TestHasMeaningfulDescription:
    """
    Placeholder and empty descriptions are treated as not meaningful.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, False),
            ("", False),
            ("   ", False),
            ("unknown", False),
            ("Unknown", False),
            ("  Tool-based analysis  ", False),
            ("Home screen with a search bar", True),
            ("x", True),
        ],
    )
    def test_meaningfulness(self, value: object, expected: bool) -> None:
        assert _has_meaningful_description(value) is expected


class TestDescriptionPreservation:
    """
    add_screen keeps the first meaningful description and upgrades placeholders.
    """

    @staticmethod
    def __kg() -> KnowledgeGraph:
        return KnowledgeGraph(provider=AsyncMock())

    @staticmethod
    def __state(hash_: str, activity: str = "com.example.app/.X") -> ScreenState:
        return ScreenState(
            visual_hash=hash_,
            activity=activity,
            timestamp=0,
            activity_hash=hash_[:16],
            structural_hash=hash_[:16],
        )

    @pytest.mark.asyncio
    async def test_revisit_does_not_overwrite_meaningful(self) -> None:
        kg = self.__kg()
        h = "a" * 64
        first = "Home screen with a bottom navigation bar and a list of restaurants"
        second = "Totally different description of a different-looking sibling screen"

        await kg.add_screen(self.__state(h), description=first)
        await kg.add_screen(self.__state(h), description=second)

        node = kg.get_screen(h)
        assert node is not None
        assert node.description == first
        assert node.visit_count == 2

    @pytest.mark.asyncio
    async def test_revisit_upgrades_from_empty(self) -> None:
        kg = self.__kg()
        h = "b" * 64
        await kg.add_screen(self.__state(h), description=None)
        await kg.add_screen(self.__state(h), description="A real description arrived later")

        node = kg.get_screen(h)
        assert node is not None
        assert node.description == "A real description arrived later"

    @pytest.mark.asyncio
    async def test_revisit_upgrades_from_placeholder(self) -> None:
        kg = self.__kg()
        h = "c" * 64
        await kg.add_screen(self.__state(h), description="unknown")
        await kg.add_screen(self.__state(h), description="Login screen with email and password")

        node = kg.get_screen(h)
        assert node is not None
        assert node.description == "Login screen with email and password"
