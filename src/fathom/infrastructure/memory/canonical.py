"""
Screen-identity resolution: maps a screen's hashes to the canonical node hash.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, Protocol

from fathom.constants.screen import DEFAULT_SAME_SCREEN_THRESHOLD, MAX_VISUAL_HASH_DISTANCE
from fathom.schemas.screens import ScreenState


class CanonicalNode(Protocol):
    """
    Read-only view of a stored screen node's identity hashes.
    """

    @property
    def activity_hash(self) -> Optional[str]:
        """Hash of the node's activity name, when available."""

    @property
    def xml_hash(self) -> Optional[str]:
        """Structural hash of the node's hierarchy, when available."""

    @property
    def interaction_hash(self) -> Optional[str]:
        """Hash of the node's interactive elements, when available."""


class ScreenCanonicalizer:
    """
    Resolves a screen's hashes to an existing node's canonical hash.

    Uses layered MLSIA structural identity first, then a Hamming-distance
    fallback on the visual hash. Matches are memoised into the supplied
    aliases map so later visual-hash-only lookups short-circuit.
    """

    def __init__(self, *, threshold: int = DEFAULT_SAME_SCREEN_THRESHOLD) -> None:
        self.__threshold = threshold

    @property
    def threshold(self) -> int:
        """
        Maximum visual Hamming distance for two hashes to be one screen.
        """

        return self.__threshold

    @staticmethod
    def hamming_distance(*, left_hash: str, right_hash: str) -> int:
        """
        Bit difference between two equal-length hex hashes.
        """

        if not left_hash or not right_hash or len(left_hash) != len(right_hash):
            return MAX_VISUAL_HASH_DISTANCE

        try:
            return bin(int(left_hash, 16) ^ int(right_hash, 16)).count("1")
        except ValueError:
            return MAX_VISUAL_HASH_DISTANCE

    @staticmethod
    def is_meaningful_hash(value: Optional[str]) -> bool:
        """
        Whether the value is a usable hash, not None, empty, or all-zero.
        """

        if not value:
            return False
        return any(character not in ("0", " ") for character in value)

    def resolve(
        self,
        *,
        visual_hash: str,
        nodes: Mapping[str, CanonicalNode],
        aliases: MutableMapping[str, str],
    ) -> str:
        """
        Map a visual hash to an existing node within the configured threshold.
        """

        if visual_hash in nodes:
            return visual_hash
        if visual_hash in aliases:
            return aliases[visual_hash]

        best_hash: Optional[str] = None
        best_distance = self.__threshold + 1

        for existing_hash in nodes:
            distance = self.hamming_distance(left_hash=visual_hash, right_hash=existing_hash)
            if distance < best_distance:
                best_distance = distance
                best_hash = existing_hash

        if best_hash is not None and best_distance <= self.__threshold:
            aliases[visual_hash] = best_hash
            return best_hash

        return visual_hash

    def resolve_for_state(
        self,
        *,
        state: ScreenState,
        nodes: Mapping[str, CanonicalNode],
        aliases: MutableMapping[str, str],
    ) -> str:
        """
        Layered MLSIA dedup: prefer structural identity, fall back to Hamming.
        """

        visual_hash = state.visual_hash
        if visual_hash in nodes:
            return visual_hash
        if visual_hash in aliases:
            return aliases[visual_hash]

        if self.is_meaningful_hash(state.activity_hash):
            structural_match = self.__match_structural(state=state, nodes=nodes)
            if structural_match is not None:
                aliases[visual_hash] = structural_match
                return structural_match

        return self.resolve(visual_hash=visual_hash, nodes=nodes, aliases=aliases)

    def __match_structural(
        self,
        *,
        state: ScreenState,
        nodes: Mapping[str, CanonicalNode],
    ) -> Optional[str]:
        """
        Find a node sharing the activity and a meaningful structural hash.
        """

        xml_layer = (state.xml_hash, True)
        interaction_layer = (state.interaction_hash, False)

        for candidate, is_xml in (xml_layer, interaction_layer):
            if not self.is_meaningful_hash(candidate):
                continue
            for existing_hash, node in nodes.items():
                node_value = node.xml_hash if is_xml else node.interaction_hash
                if not self.is_meaningful_hash(node_value):
                    continue
                if node.activity_hash != state.activity_hash:
                    continue
                if node_value == candidate:
                    return existing_hash

        return None
