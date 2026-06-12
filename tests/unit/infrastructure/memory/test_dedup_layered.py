"""
Unit tests for the layered MLSIA screen merge in KnowledgeGraph.add_screen.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.schemas.screens import ScreenState


class TestLayeredScreenMerge:
    """
    add_screen merges visually-drifted screens via structural / xml / interaction
    hashes, and falls back to visual-hash Hamming distance.
    """

    @staticmethod
    def __kg() -> KnowledgeGraph:
        return KnowledgeGraph(provider=AsyncMock())

    @staticmethod
    def __state(
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
    async def test_structural_hash_match_merges_visually_drifted_screens(self) -> None:
        kg = self.__kg()
        await kg.add_screen(self.__state("1" * 64))
        # Same activity + structural_hash, different visual_hash → should merge.
        await kg.add_screen(self.__state("f" * 64))

        assert len(kg.nodes) == 1
        canonical = next(iter(kg.nodes))
        assert canonical == "1" * 64
        assert kg.nodes[canonical].visit_count == 2

    @pytest.mark.asyncio
    async def test_xml_hash_match_when_structural_differs(self) -> None:
        kg = self.__kg()
        await kg.add_screen(
            self.__state("1" * 64, structural="aaaaaaaaaaaaaaaa", xml="cafecafecafe0001")
        )
        await kg.add_screen(
            self.__state("f" * 64, structural="0123456789abcdef", xml="cafecafecafe0001")
        )
        assert len(kg.nodes) == 1

    @pytest.mark.asyncio
    async def test_zero_hash_does_not_falsely_merge(self) -> None:
        kg = self.__kg()
        placeholder = "0000000000000000"
        await kg.add_screen(self.__state("1" * 64, structural=placeholder, xml=placeholder))
        await kg.add_screen(self.__state("f" * 64, structural=placeholder, xml=placeholder))
        assert len(kg.nodes) == 2

    @pytest.mark.asyncio
    async def test_different_activity_blocks_structural_merge(self) -> None:
        kg = self.__kg()
        await kg.add_screen(self.__state("1" * 64, activity_hash="0000000000000001"))
        await kg.add_screen(self.__state("f" * 64, activity_hash="0000000000000002"))
        assert len(kg.nodes) == 2

    @pytest.mark.asyncio
    async def test_visual_hash_hamming_fallback_still_works(self) -> None:
        kg = self.__kg()
        zeros = "0000000000000000"
        await kg.add_screen(self.__state("0" * 64, structural=zeros, activity_hash=zeros))
        await kg.add_screen(self.__state("0" * 63 + "1", structural=zeros, activity_hash=zeros))
        assert len(kg.nodes) == 1

    @pytest.mark.asyncio
    async def test_hashes_are_recorded_on_first_observation(self) -> None:
        kg = self.__kg()
        h = "abcd" * 16
        await kg.add_screen(
            self.__state(
                h,
                structural="deadbeefdeadbeef",
                xml="cafecafecafecafe",
                interaction="feedfeedfeedfeed",
            )
        )
        node = kg.get_screen(h)
        assert node is not None
        assert node.activity_hash == "aaaaaaaaaaaaaaaa"
        assert node.structural_hash == "deadbeefdeadbeef"
        assert node.xml_hash == "cafecafecafecafe"
        assert node.interaction_hash == "feedfeedfeedfeed"
