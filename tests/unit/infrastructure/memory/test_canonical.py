from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional

from fathom.constants.screen import ZERO_HASH
from fathom.infrastructure.memory.canonical import ScreenCanonicalizer
from fathom.schemas.screens import ScreenState

# 16-hex (64-bit) hashes with controlled Hamming distances from VISUAL_BASE.
VISUAL_BASE = "ffffffff00000000"
VISUAL_NEAR = "ffffffff00000003"  # 2 bits from base (<= threshold)
VISUAL_DRIFT = "ffffffff0000003f"  # 6 bits from base (<= threshold)
VISUAL_FAR = "00000000ffffffff"  # 64 bits from base (> threshold)

ACTIVITY_A = "aaaaaaaaaaaaaaaa"
ACTIVITY_B = "bbbbbbbbbbbbbbbb"

STRUCTURE_X = "1111111111111111"
STRUCTURE_Y = "2222222222222222"

XML_SHARED = "cccccccccccccccc"


@dataclass
class _FakeNode:
    """Minimal CanonicalNode: just the identity hashes the canonicalizer reads."""

    activity_hash: Optional[str] = None
    xml_hash: Optional[str] = None
    interaction_hash: Optional[str] = None
    structure_hash: Optional[str] = None


class TestScreenCanonicalizerIdentityGate(unittest.TestCase):
    """A visual near-duplicate merges only when no stronger identity signal contradicts it."""

    def setUp(self) -> None:
        self.__canonicalizer = ScreenCanonicalizer()
        self.__nodes = {
            VISUAL_BASE: _FakeNode(activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X)
        }

    def __resolve(
        self,
        *,
        visual_hash: str,
        activity_hash: Optional[str],
        structure_hash: Optional[str],
    ) -> str:
        return self.__canonicalizer.resolve_identity(
            visual_hash=visual_hash,
            activity_hash=activity_hash,
            structure_hash=structure_hash,
            nodes=self.__nodes,
            aliases={},
        )

    def test_same_activity_and_structure_merges(self) -> None:
        resolved = self.__resolve(
            visual_hash=VISUAL_NEAR, activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X
        )
        self.assertEqual(resolved, VISUAL_BASE)

    def test_content_variant_same_layout_still_merges(self) -> None:
        resolved = self.__resolve(
            visual_hash=VISUAL_DRIFT, activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X
        )
        self.assertEqual(resolved, VISUAL_BASE)

    def test_cross_activity_near_duplicate_stays_separate(self) -> None:
        resolved = self.__resolve(
            visual_hash=VISUAL_NEAR, activity_hash=ACTIVITY_B, structure_hash=STRUCTURE_X
        )
        self.assertEqual(resolved, VISUAL_NEAR)

    def test_different_structure_same_activity_stays_separate(self) -> None:
        resolved = self.__resolve(
            visual_hash=VISUAL_NEAR, activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_Y
        )
        self.assertEqual(resolved, VISUAL_NEAR)

    def test_far_visual_never_merges(self) -> None:
        resolved = self.__resolve(
            visual_hash=VISUAL_FAR, activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X
        )
        self.assertEqual(resolved, VISUAL_FAR)

    def test_missing_structure_does_not_veto(self) -> None:
        self.__nodes = {VISUAL_BASE: _FakeNode(activity_hash=ACTIVITY_A, structure_hash=None)}
        resolved = self.__resolve(
            visual_hash=VISUAL_NEAR, activity_hash=ACTIVITY_A, structure_hash=None
        )
        self.assertEqual(resolved, VISUAL_BASE)

    def test_zero_structure_is_not_a_veto(self) -> None:
        self.__nodes = {VISUAL_BASE: _FakeNode(activity_hash=ACTIVITY_A, structure_hash=ZERO_HASH)}
        resolved = self.__resolve(
            visual_hash=VISUAL_NEAR, activity_hash=ACTIVITY_A, structure_hash=ZERO_HASH
        )
        self.assertEqual(resolved, VISUAL_BASE)

    def test_missing_activity_does_not_gate(self) -> None:
        self.__nodes = {VISUAL_BASE: _FakeNode(activity_hash=None, structure_hash=STRUCTURE_X)}
        resolved = self.__resolve(
            visual_hash=VISUAL_NEAR, activity_hash=None, structure_hash=STRUCTURE_X
        )
        self.assertEqual(resolved, VISUAL_BASE)

    def test_exact_hash_resolves_to_self(self) -> None:
        resolved = self.__resolve(
            visual_hash=VISUAL_BASE, activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X
        )
        self.assertEqual(resolved, VISUAL_BASE)

    def test_alias_short_circuits_without_scanning(self) -> None:
        aliases = {VISUAL_NEAR: VISUAL_BASE}
        resolved = self.__canonicalizer.resolve_identity(
            visual_hash=VISUAL_NEAR,
            activity_hash=ACTIVITY_B,
            structure_hash=STRUCTURE_Y,
            nodes=self.__nodes,
            aliases=aliases,
        )
        self.assertEqual(resolved, VISUAL_BASE)


class TestScreenCanonicalizerStateFallback(unittest.TestCase):
    """resolve_for_state keeps structural merging but routes its visual fallback through the gate."""

    def setUp(self) -> None:
        self.__canonicalizer = ScreenCanonicalizer()

    @staticmethod
    def __state(
        *,
        activity_hash: str,
        visual_hash: str,
        structure_hash: Optional[str] = None,
        xml_hash: Optional[str] = None,
    ) -> ScreenState:
        return ScreenState(
            activity=f"pkg/{activity_hash}",
            timestamp=0,
            activity_hash=activity_hash,
            visual_hash=visual_hash,
            xml_hash=xml_hash,
            structure_hash=structure_hash,
        )

    def test_visual_fallback_honours_activity_gate(self) -> None:
        nodes = {VISUAL_BASE: _FakeNode(activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X)}
        state = self.__state(
            activity_hash=ACTIVITY_B, visual_hash=VISUAL_NEAR, structure_hash=STRUCTURE_X
        )
        resolved = self.__canonicalizer.resolve_for_state(state=state, nodes=nodes, aliases={})
        self.assertEqual(resolved, VISUAL_NEAR)

    def test_visual_fallback_merges_when_identity_agrees(self) -> None:
        nodes = {VISUAL_BASE: _FakeNode(activity_hash=ACTIVITY_A, structure_hash=STRUCTURE_X)}
        state = self.__state(
            activity_hash=ACTIVITY_A, visual_hash=VISUAL_NEAR, structure_hash=STRUCTURE_X
        )
        resolved = self.__canonicalizer.resolve_for_state(state=state, nodes=nodes, aliases={})
        self.assertEqual(resolved, VISUAL_BASE)

    def test_structural_match_still_merges_despite_far_visual(self) -> None:
        nodes = {
            VISUAL_BASE: _FakeNode(
                activity_hash=ACTIVITY_A, xml_hash=XML_SHARED, structure_hash=STRUCTURE_X
            )
        }
        state = self.__state(activity_hash=ACTIVITY_A, visual_hash=VISUAL_FAR, xml_hash=XML_SHARED)
        resolved = self.__canonicalizer.resolve_for_state(state=state, nodes=nodes, aliases={})
        self.assertEqual(resolved, VISUAL_BASE)


if __name__ == "__main__":
    unittest.main()
