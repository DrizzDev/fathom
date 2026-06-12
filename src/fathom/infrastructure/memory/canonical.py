"""
Screen-identity resolution: maps a screen's hashes to the canonical node hash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from fathom.schemas.screens import ScreenState

if TYPE_CHECKING:
    from fathom.infrastructure.memory.knowledge_graph import GraphNode

# Maximum Hamming distance (in bits, out of 256 for a 16×16 dHash) for two
# visual hashes to be considered the same logical screen.  12 bits ≈ 95%
# similarity — safely merges minor pixel variations (status-bar clock, cursor
# blink, animation frames) while keeping genuinely different screens apart.
HAMMING_THRESHOLD = 12


class ScreenCanonicalizer:
    """
    Resolves a screen's hashes to an existing node's canonical hash.

    Uses layered MLSIA structural identity first, then a Hamming-distance
    fallback on the visual hash. Matches are memoised into the supplied
    ``aliases`` map so later visual-hash-only lookups short-circuit.
    """

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Bit difference between two equal-length hex hashes (256 if mismatched)."""

        if len(hash1) != len(hash2):
            return 256
        try:
            return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")
        except ValueError:
            return 256

    @staticmethod
    def is_meaningful_hash(value: Optional[str]) -> bool:
        """True when *value* is a usable hash (not None, empty, or all-zero)."""

        if not value:
            return False
        return any(c not in ("0", " ") for c in value)

    @classmethod
    def resolve(
        cls,
        *,
        visual_hash: str,
        nodes: Dict[str, "GraphNode"],
        aliases: Dict[str, str],
    ) -> str:
        """Map *visual_hash* to an existing node within HAMMING_THRESHOLD."""

        if visual_hash in nodes:
            return visual_hash
        if visual_hash in aliases:
            return aliases[visual_hash]

        best_hash: Optional[str] = None
        best_distance = HAMMING_THRESHOLD + 1

        for existing_hash in nodes:
            d = cls.hamming_distance(visual_hash, existing_hash)
            if d < best_distance:
                best_distance = d
                best_hash = existing_hash

        if best_hash is not None and best_distance <= HAMMING_THRESHOLD:
            aliases[visual_hash] = best_hash
            return best_hash

        return visual_hash

    @classmethod
    def resolve_for_state(
        cls,
        *,
        state: ScreenState,
        nodes: Dict[str, "GraphNode"],
        aliases: Dict[str, str],
    ) -> str:
        """
        Layered MLSIA dedup: prefer structural identity, fall back to Hamming.

        Order (mirrors :meth:`ScreenState.is_same_screen`):
          1. ``state.visual_hash`` already a canonical or alias.
          2. Same ``activity_hash`` AND same ``structural_hash`` (both meaningful).
          3. Same ``activity_hash`` AND same ``xml_hash`` (both meaningful).
          4. Same ``activity_hash`` AND same ``interaction_hash`` (both meaningful).
          5. Hamming-on-``visual_hash`` ≤ ``HAMMING_THRESHOLD`` fallback.
        """

        vh = state.visual_hash
        if vh in nodes:
            return vh
        if vh in aliases:
            return aliases[vh]

        if cls.is_meaningful_hash(state.activity_hash):
            for field, candidate in (
                ("structural_hash", state.structural_hash),
                ("xml_hash", state.xml_hash),
                ("interaction_hash", state.interaction_hash),
            ):
                if not cls.is_meaningful_hash(candidate):
                    continue
                for existing_hash, node in nodes.items():
                    if not cls.is_meaningful_hash(getattr(node, field, None)):
                        continue
                    if node.activity_hash != state.activity_hash:
                        continue
                    if getattr(node, field) != candidate:
                        continue
                    aliases[vh] = existing_hash
                    return existing_hash

        return cls.resolve(visual_hash=vh, nodes=nodes, aliases=aliases)
