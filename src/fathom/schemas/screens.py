from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScreenState(BaseModel):
    """Immutable screen state representation.

    Uses a hybrid 3-layer hashing approach for efficient screen comparison.
    """

    model_config = ConfigDict(frozen=True)

    activity: str = Field(description="Current activity/screen identifier")
    timestamp: int = Field(description="Capture timestamp in milliseconds")

    activity_hash: str = Field(description="Hash of activity name")
    structural_hash: str = Field(description="Hash of screen structure")
    visual_hash: str = Field(description="Perceptual hash (pHash) of screen")
    xml_hash: Optional[str] = Field(default=None, description="Semantic structural hash of XML")
    interaction_hash: Optional[str] = Field(
        default=None, description="Hash of interactive elements"
    )

    def is_same_screen(self, other: "ScreenState", threshold: int = 10) -> bool:
        """
        Check if two screen states represent the same screen.

        Uses Multi-Layer State Identity Algorithm (MLSIA):
        1. Activity check (must match)
        2. Structural check (XML tree structure)
        3. Interaction check (Clickable elements)
        4. Visual check (dHash distance)

        Returns True ONLY if ALL available signals match.
        """

        if self.activity_hash != other.activity_hash:
            return False

        # 1. Structural Identity (Primary)
        # If XML structure changes (nodes added/removed), it's a new screen.
        if (
            self.xml_hash
            and other.xml_hash
            and self.xml_hash != "0000000000000000"
            and self.xml_hash != other.xml_hash
        ):
            return False

        # 2. Interaction Identity (Secondary)
        # If the set of clickable actions changes (even if structure is same), it's a new screen.
        if (
            self.interaction_hash
            and other.interaction_hash
            and self.interaction_hash != "0000000000000000"
            and self.interaction_hash != other.interaction_hash
        ):
            return False

        # 3. Visual Identity (Fallback/Confirmation)
        # If both structural and interaction hashes match (or are missing),
        # we verify visual similarity to catch "paint-only" changes.
        distance = self.__hamming_distance(self.visual_hash, other.visual_hash)
        return distance <= threshold

    @staticmethod
    def __hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculate hamming distance between two hex hash strings.
        """

        if len(hash1) != len(hash2):
            return 64

        return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")


class ScreenCapture(BaseModel):
    """
    Screen capture with image data and metadata.
    """

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0, description="Screen width in pixels")
    height: int = Field(gt=0, description="Screen height in pixels")

    activity: str = Field(description="Current activity name")
    image: bytes = Field(description="Raw PNG image bytes", repr=False)
    xml_content: Optional[str] = Field(default=None, description="Raw XML hierarchy", repr=False)
    timestamp: int = Field(description="Capture timestamp in milliseconds")

    state: Optional[ScreenState] = Field(
        default=None,
        description="Computed screen state (may be populated lazily)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional capture metadata"
    )
