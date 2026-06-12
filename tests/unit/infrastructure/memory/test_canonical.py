"""
Unit tests for screen-identity canonicalization.
"""

from __future__ import annotations

from fathom.infrastructure.memory.canonical import ScreenCanonicalizer
from fathom.infrastructure.memory.knowledge_graph import GraphNode
from fathom.schemas.screens import ScreenState


class TestScreenCanonicalizer:
    """
    Hamming distance, hash meaningfulness, and layered MLSIA resolution.
    """

    def test_hamming_distance(self) -> None:
        assert ScreenCanonicalizer.hamming_distance("ff", "ff") == 0
        assert ScreenCanonicalizer.hamming_distance("00", "01") == 1
        assert ScreenCanonicalizer.hamming_distance("abc", "ab") == 256  # length mismatch
        assert ScreenCanonicalizer.hamming_distance("zz", "zz") == 256  # non-hex

    def test_is_meaningful_hash(self) -> None:
        assert ScreenCanonicalizer.is_meaningful_hash("abc") is True
        assert ScreenCanonicalizer.is_meaningful_hash(None) is False
        assert ScreenCanonicalizer.is_meaningful_hash("") is False
        assert ScreenCanonicalizer.is_meaningful_hash("0000") is False

    def test_resolve_returns_self_for_known_hash(self) -> None:
        canonical = "a" * 64
        nodes = {canonical: GraphNode(visual_hash=canonical, activity="X")}
        assert (
            ScreenCanonicalizer.resolve(visual_hash=canonical, nodes=nodes, aliases={}) == canonical
        )

    def test_resolve_merges_within_threshold_and_records_alias(self) -> None:
        canonical = "0" * 64
        drifted = "0" * 63 + "1"  # 1-bit difference, within HAMMING_THRESHOLD
        nodes = {canonical: GraphNode(visual_hash=canonical, activity="X")}
        aliases: dict[str, str] = {}
        assert (
            ScreenCanonicalizer.resolve(visual_hash=drifted, nodes=nodes, aliases=aliases)
            == canonical
        )
        assert aliases[drifted] == canonical

    def test_resolve_keeps_distinct_beyond_threshold(self) -> None:
        nodes = {"0" * 64: GraphNode(visual_hash="0" * 64, activity="X")}
        far = "f" * 64  # 256-bit distance, far beyond threshold
        assert ScreenCanonicalizer.resolve(visual_hash=far, nodes=nodes, aliases={}) == far

    def test_resolve_for_state_merges_on_structural_identity(self) -> None:
        canonical = "1" * 64
        nodes = {
            canonical: GraphNode(
                visual_hash=canonical, activity="X", activity_hash="aaaa", structural_hash="bbbb"
            )
        }
        # Different (far) visual hash but same activity + structural hash → merge.
        state = ScreenState(
            visual_hash="f" * 64,
            activity="X",
            timestamp=0,
            activity_hash="aaaa",
            structural_hash="bbbb",
            xml_hash=None,
            interaction_hash=None,
        )
        assert (
            ScreenCanonicalizer.resolve_for_state(state=state, nodes=nodes, aliases={}) == canonical
        )
