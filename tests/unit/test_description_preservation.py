"""Tests for KnowledgeGraph preserving the first meaningful description on revisit."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fathom.infrastructure.memory.knowledge_graph import (
    KnowledgeGraph,
    _has_meaningful_description,
)
from fathom.schemas.screens import ScreenState


@pytest.fixture
def kg():
    graph = KnowledgeGraph()
    graph._KnowledgeGraph__provider = AsyncMock()
    graph._KnowledgeGraph__nodes = {}
    graph._KnowledgeGraph__edges = {}
    graph._KnowledgeGraph__hash_aliases = {}
    graph._KnowledgeGraph__loaded = True
    return graph


def _state(hash_: str, activity: str = "com.example.app/.X") -> ScreenState:
    return ScreenState(
        visual_hash=hash_,
        activity=activity,
        timestamp=0,
        activity_hash=hash_[:16],
        structural_hash=hash_[:16],
    )


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
def test_has_meaningful_description(value, expected):
    assert _has_meaningful_description(value) is expected


@pytest.mark.asyncio
async def test_revisit_does_not_overwrite_meaningful_description(kg):
    h = "a" * 64
    first = "Home screen with a bottom navigation bar and a list of restaurants"
    second = "Totally different description of a different-looking sibling screen"

    await kg.add_screen(_state(h), description=first)
    await kg.add_screen(_state(h), description=second)

    node = kg.get_screen(h)
    assert node is not None
    assert node.description == first
    assert node.visit_count == 2


@pytest.mark.asyncio
async def test_revisit_upgrades_from_empty_description(kg):
    h = "b" * 64

    await kg.add_screen(_state(h), description=None)
    await kg.add_screen(_state(h), description="A real description arrived later")

    assert kg.get_screen(h).description == "A real description arrived later"


@pytest.mark.asyncio
async def test_revisit_upgrades_from_placeholder_description(kg):
    h = "c" * 64

    await kg.add_screen(_state(h), description="unknown")
    await kg.add_screen(_state(h), description="Login screen with email and password")

    assert kg.get_screen(h).description == "Login screen with email and password"
