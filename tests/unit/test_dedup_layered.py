"""Tests for the layered MLSIA dedup in KnowledgeGraph._resolve_canonical_for_state."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.screens import ScreenState


def _kg() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g._KnowledgeGraph__provider = AsyncMock()
    g._KnowledgeGraph__nodes = {}
    g._KnowledgeGraph__edges = {}
    g._KnowledgeGraph__hash_aliases = {}
    g._KnowledgeGraph__loaded = True
    return g


def _state(
    visual: str,
    *,
    activity: str = "com.example.app/.X",
    activity_hash: str = "aaaaaaaaaaaaaaaa",
    structural: str = "bbbbbbbbbbbbbbbb",
    xml: str | None = None,
    interaction: str | None = None,
) -> ScreenState:
    return ScreenState(
        visual_hash=visual,
        activity=activity,
        timestamp=0,
        activity_hash=activity_hash,
        structural_hash=structural,
        xml_hash=xml,
        interaction_hash=interaction,
    )


@pytest.mark.asyncio
async def test_structural_hash_match_merges_visually_drifted_screens():
    kg = _kg()
    h1 = "1" * 64
    h2 = "f" * 64  # 256-bit Hamming distance from h1 — well past threshold
    await kg.add_screen(_state(h1))
    # Same activity + structural_hash, different visual_hash → should merge.
    await kg.add_screen(_state(h2))

    assert len(kg.nodes) == 1
    canonical = next(iter(kg.nodes))
    assert canonical == h1
    assert kg.nodes[canonical].visit_count == 2


@pytest.mark.asyncio
async def test_xml_hash_match_when_structural_differs():
    kg = _kg()
    h1 = "1" * 64
    h2 = "f" * 64
    await kg.add_screen(_state(h1, structural="aaaaaaaaaaaaaaaa", xml="cafecafecafe0001"))
    # Different structural_hash but matching xml_hash → still merge.
    await kg.add_screen(_state(h2, structural="0123456789abcdef", xml="cafecafecafe0001"))

    assert len(kg.nodes) == 1


@pytest.mark.asyncio
async def test_zero_hash_does_not_falsely_merge():
    kg = _kg()
    h1 = "1" * 64
    h2 = "f" * 64
    placeholder = "0000000000000000"
    await kg.add_screen(_state(h1, structural=placeholder, xml=placeholder))
    # All structural identifiers are placeholder zeros — should NOT merge,
    # and the visual-hash Hamming fallback also can't merge (h2 is far).
    await kg.add_screen(_state(h2, structural=placeholder, xml=placeholder))

    assert len(kg.nodes) == 2


@pytest.mark.asyncio
async def test_different_activity_blocks_structural_merge():
    kg = _kg()
    h1 = "1" * 64
    h2 = "f" * 64
    await kg.add_screen(_state(h1, activity_hash="0000000000000001"))
    # Same structural_hash but a different activity → must not merge.
    await kg.add_screen(_state(h2, activity_hash="0000000000000002"))

    assert len(kg.nodes) == 2


@pytest.mark.asyncio
async def test_visual_hash_hamming_fallback_still_works():
    kg = _kg()
    h1 = "0" * 64
    h2 = "0" * 63 + "1"  # 1-bit difference, well within HAMMING_THRESHOLD
    # Both lack structural identifiers — should fall back to visual Hamming
    # matching and still merge.
    await kg.add_screen(
        _state(
            h1,
            structural="0000000000000000",
            xml=None,
            interaction=None,
            activity_hash="0000000000000000",
        )
    )
    await kg.add_screen(
        _state(
            h2,
            structural="0000000000000000",
            xml=None,
            interaction=None,
            activity_hash="0000000000000000",
        )
    )

    assert len(kg.nodes) == 1


@pytest.mark.asyncio
async def test_hashes_are_recorded_on_first_observation():
    kg = _kg()
    h = "abcd" * 16
    await kg.add_screen(
        _state(
            h, structural="deadbeefdeadbeef", xml="cafecafecafecafe", interaction="feedfeedfeedfeed"
        )
    )

    node = kg.get_screen(h)
    assert node is not None
    assert node.activity_hash == "aaaaaaaaaaaaaaaa"
    assert node.structural_hash == "deadbeefdeadbeef"
    assert node.xml_hash == "cafecafecafecafe"
    assert node.interaction_hash == "feedfeedfeedfeed"
