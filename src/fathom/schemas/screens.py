from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.screen import (
    ACTION_EFFECT_CONTENT_DIFF_RATIO_THRESHOLD,
    ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD,
    ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX,
    ACTION_EFFECT_SSIM_THRESHOLD,
    ACTIVITY_CHANGED_SIGNAL_WEIGHT,
    CONTENT_DIFF_SIGNAL_WEIGHT,
    DEFAULT_SAME_SCREEN_THRESHOLD,
    INTERACTION_HASH_CHANGED_SIGNAL_WEIGHT,
    MAX_VISUAL_HASH_DISTANCE,
    MEANINGFUL_STATE_CONTENT_DIFF_RATIO_THRESHOLD,
    MEANINGFUL_STATE_PHASH_DISTANCE_THRESHOLD,
    MEANINGFUL_STATE_SCROLL_DISTANCE_THRESHOLD_PX,
    MEANINGFUL_STATE_SIGNAL_WEIGHT_THRESHOLD,
    MEANINGFUL_STATE_SSIM_THRESHOLD,
    PHASH_CHANGED_SIGNAL_WEIGHT,
    SCROLL_CHANGED_SIGNAL_WEIGHT,
    SSIM_CHANGED_SIGNAL_WEIGHT,
    XML_HASH_CHANGED_SIGNAL_WEIGHT,
    ZERO_HASH,
    HierarchyProvenance,
)
from fathom.schemas.artifacts import StepArtifacts


class ScreenState(BaseModel):
    """
    Immutable screen state representation.
    Uses a hybrid 3-layer hashing approach for efficient screen comparison.
    """

    model_config = ConfigDict(frozen=True)

    activity: str = Field(description="Current activity/screen identifier")
    timestamp: int = Field(description="Capture timestamp in milliseconds")

    activity_hash: str = Field(description="Hash of activity name")
    visual_hash: str = Field(description="Perceptual hash (pHash) of screen")

    xml_hash: Optional[str] = Field(default=None, description="Semantic structural hash of XML")
    interaction_hash: Optional[str] = Field(
        default=None, description="Hash of interactive elements"
    )

    def is_same_screen(
        self,
        other: "ScreenState",
        threshold: int = DEFAULT_SAME_SCREEN_THRESHOLD,
    ) -> bool:
        """
        Check if two screen states represent the same screen.

        Compares layered signals in order:
        1. Activity check (must match)
        2. Structural check (XML tree structure + content)
        3. Interaction check (clickable elements)
        4. Visual check (pHash distance)

        Returns True only if all available signals match.
        """

        if self.activity_hash != other.activity_hash:
            return False

        # 1. Structural Identity (Primary)
        # If XML structure changes (nodes added/removed), it's a new screen.
        if (
            self.xml_hash
            and other.xml_hash
            and self.xml_hash != other.xml_hash
            and self.xml_hash != ZERO_HASH
            and other.xml_hash != ZERO_HASH
        ):
            return False

        # 2. Interaction Identity (Secondary)
        # If the set of clickable actions changes (even if structure is same), it's a new screen.
        if (
            self.interaction_hash
            and other.interaction_hash
            and self.interaction_hash != ZERO_HASH
            and other.interaction_hash != ZERO_HASH
            and self.interaction_hash != other.interaction_hash
        ):
            return False

        # 3. Visual Identity (Fallback/Confirmation)
        # If both structural and interaction hashes match (or are missing),
        # we verify visual similarity to catch "paint-only" changes.
        distance = self.hamming_distance(
            left_hash=self.visual_hash,
            right_hash=other.visual_hash,
        )
        return distance <= threshold

    @staticmethod
    def hamming_distance(*, left_hash: str, right_hash: str) -> int:
        """
        Calculate hamming distance between two hex hash strings.
        """

        if not left_hash or not right_hash or len(left_hash) != len(right_hash):
            return MAX_VISUAL_HASH_DISTANCE

        try:
            return bin(int(left_hash, 16) ^ int(right_hash, 16)).count("1")
        except (ValueError, TypeError):
            return MAX_VISUAL_HASH_DISTANCE

    def has_visual_progress_from(
        self, *, previous: Optional["ScreenState"], threshold: int
    ) -> bool:
        """
        Whether the visual pHash hamming distance from ``previous`` exceeds the near-duplicate threshold
        (i.e. the screen visually moved forward, not just had animation noise). First screen always counts as progress.
        """

        if previous is None:
            return True

        distance = self.hamming_distance(
            left_hash=previous.visual_hash, right_hash=self.visual_hash
        )

        return distance > threshold


class ScreenDiff(BaseModel):
    """
    Rich comparison result between two consecutive screen captures.

    Two properties intentionally use different sensitivities:
    - `action_had_effect`: high sensitivity for EXECUTE validation
    - `is_genuinely_different_state`: moderate sensitivity for loop detection
    """

    model_config = ConfigDict(frozen=True)

    phash_distance: int = Field(
        description="Hamming distance between visual pHash values (0=identical, 64=max)"
    )

    xml_hash_changed: bool = Field(description="Whether the structural XML tree hash changed")
    interaction_hash_changed: bool = Field(
        description="Whether the interactive element identity set changed"
    )
    activity_changed: bool = Field(
        description="Whether the foreground activity or screen name changed"
    )

    ssim_score: Optional[float] = Field(
        default=None,
        description="SSIM similarity in content region (0.0=different, 1.0=identical)",
    )
    content_pixel_diff_ratio: Optional[float] = Field(
        default=None,
        description="Fraction of changed pixels in content region (status bar excluded)",
    )
    changed_regions: List["ScreenChangeRegion"] = Field(
        default_factory=list,
        description="Changed content regions detected between two captures",
    )
    scroll_translation: Optional["ScreenScrollTranslation"] = Field(
        default=None,
        description="Estimated screen translation between two captures",
    )

    @property
    def action_had_effect(self) -> bool:
        """
        True when the action produced an observable screen effect.
        Structural-only signals (xml_hash / interaction_hash / changed_regions)
        require a visual or scroll co-signal to fire, gating render-loop
        noise and frequently-updating ignorable elements.
        """

        if self.activity_changed:
            return True

        return self.__has_visual_signal() or self.__has_action_scroll_signal()

    def __has_visual_signal(self) -> bool:
        """
        True when phash / ssim / content_diff crosses the action-effect threshold.
        """

        if self.phash_distance > ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD:
            return True

        if self.ssim_score is not None and self.ssim_score < ACTION_EFFECT_SSIM_THRESHOLD:
            return True

        return (
            self.content_pixel_diff_ratio is not None
            and self.content_pixel_diff_ratio > ACTION_EFFECT_CONTENT_DIFF_RATIO_THRESHOLD
        )

    @property
    def is_genuinely_different_state(self) -> bool:
        """
        Moderate-sensitivity check for loop detection and state transitions.
        """

        signal_weight = 0

        if self.activity_changed:
            signal_weight += ACTIVITY_CHANGED_SIGNAL_WEIGHT

        if self.xml_hash_changed:
            signal_weight += XML_HASH_CHANGED_SIGNAL_WEIGHT

        if self.interaction_hash_changed:
            signal_weight += INTERACTION_HASH_CHANGED_SIGNAL_WEIGHT

        if self.phash_distance > MEANINGFUL_STATE_PHASH_DISTANCE_THRESHOLD:
            signal_weight += PHASH_CHANGED_SIGNAL_WEIGHT

        if self.ssim_score is not None and self.ssim_score < MEANINGFUL_STATE_SSIM_THRESHOLD:
            signal_weight += SSIM_CHANGED_SIGNAL_WEIGHT

        if (
            self.content_pixel_diff_ratio is not None
            and self.content_pixel_diff_ratio > MEANINGFUL_STATE_CONTENT_DIFF_RATIO_THRESHOLD
        ):
            signal_weight += CONTENT_DIFF_SIGNAL_WEIGHT

        if self.__has_meaningful_scroll_signal():
            signal_weight += SCROLL_CHANGED_SIGNAL_WEIGHT

        return signal_weight >= MEANINGFUL_STATE_SIGNAL_WEIGHT_THRESHOLD

    def __has_action_scroll_signal(self) -> bool:
        """
        Return whether scroll displacement indicates any action effect.
        """

        if self.scroll_translation is None:
            return False

        dx = self.scroll_translation.dx
        dy = self.scroll_translation.dy

        return (
            abs(dx) > ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX
            or abs(dy) > ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX
        )

    def __has_meaningful_scroll_signal(self) -> bool:
        """
        Return whether scroll displacement indicates a meaningful state transition.
        """

        if self.scroll_translation is None:
            return False

        dx = self.scroll_translation.dx
        dy = self.scroll_translation.dy

        return (
            abs(dx) > MEANINGFUL_STATE_SCROLL_DISTANCE_THRESHOLD_PX
            or abs(dy) > MEANINGFUL_STATE_SCROLL_DISTANCE_THRESHOLD_PX
        )


class ScreenChangeRegion(BaseModel):
    """
    Rectangular content region that changed between two captures.
    """

    model_config = ConfigDict(frozen=True)

    x: int = Field(description="Left coordinate in pixels")
    y: int = Field(description="Top coordinate in pixels")
    width: int = Field(description="Region width in pixels")
    height: int = Field(description="Region height in pixels")


class ScreenScrollTranslation(BaseModel):
    """
    Estimated translation between two captures in pixels.
    """

    model_config = ConfigDict(frozen=True)

    dx: float = Field(description="Horizontal translation in pixels")
    dy: float = Field(description="Vertical translation in pixels")


class ScreenHashBundle(BaseModel):
    """
    Precomputed screen hashes derived from one capture.
    """

    model_config = ConfigDict(frozen=True)

    visual_hash: str = Field(description="Perceptual visual hash for the capture")
    xml_hash: str = Field(description="Structural hierarchy hash for the capture")
    interaction_hash: str = Field(description="Interactive element identity hash for the capture")


class StructuralComparisonSignals(BaseModel):
    """
    Structural comparison signals derived from two screen states.
    """

    model_config = ConfigDict(frozen=True)

    phash_distance: int = Field(description="Hamming distance between two visual hashes")
    xml_hash_changed: bool = Field(description="Whether XML hashes differ")
    interaction_hash_changed: bool = Field(description="Whether interaction hashes differ")


class PostActionScreenComparison(BaseModel):
    """
    Result of post-action capture comparison for one executed step.
    """

    model_config = ConfigDict(frozen=True)

    post_visual_hash: Optional[str] = Field(
        default=None,
        description="Visual hash of the post-action capture when available",
    )
    screen_diff: Optional[ScreenDiff] = Field(
        default=None,
        description="Rich screen comparison between pre-action and post-action captures",
    )
    artifacts: Optional[StepArtifacts] = Field(
        default=None,
        description=(
            "Namespaced artifact envelope produced for this step "
            "(screen.before, screen.after, future namespaces such as hierarchy/trace)."
        ),
    )


class ScreenCapture(BaseModel):
    """
    Screen capture with image data and metadata.
    """

    model_config = ConfigDict(frozen=True)

    width: int = Field(
        gt=0,
        description=(
            "Capture width in the dispatch coordinate space. Whether that space is "
            "LOGICAL points or DEVICE PIXELS depends on the adapter: some populate it "
            "from the PNG header (pixels), others from the platform's logical screen "
            "size, and on a retina device those differ by the scale factor (e.g. 1180 "
            "logical vs 2360 pixels at 2x). `ScreenObservation.__capture_dimension_system` "
            "resolves which it is at runtime by comparing against the decoded image. "
            "Do NOT assume this equals the pixel size of `image` — code mapping "
            "coordinates out of `image` must decode the PNG for its true dimensions."
        ),
    )
    height: int = Field(
        gt=0,
        description=(
            "Capture height in the dispatch coordinate space. See `width` — may be "
            "logical points or device pixels depending on the adapter."
        ),
    )

    activity: str = Field(description="Current activity name")
    image: bytes = Field(
        repr=False,
        description=(
            "Canonical raw PNG image bytes, always at DEVICE-PIXEL resolution. This is "
            "the space OCR/vision results come back in; decode it when the true pixel "
            "size is needed rather than relying on `width`/`height`."
        ),
    )
    annotated_image: Optional[bytes] = Field(
        repr=False,
        default=None,
        description="Optional annotated PNG bytes for prompt grounding or debugging",
    )
    xml_content: Optional[str] = Field(default=None, description="Raw XML hierarchy", repr=False)
    timestamp: int = Field(description="Capture timestamp in milliseconds")

    state: Optional[ScreenState] = Field(
        default=None,
        description="Computed screen state (may be populated lazily)",
    )
    screenshot_uri: Optional[str] = Field(
        default=None,
        description="External handle returned when the raw screenshot was published for the step",
    )
    annotated_uri: Optional[str] = Field(
        default=None,
        description="External handle returned when the annotated screenshot was published",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional capture metadata"
    )

    @property
    def identity(self) -> str:
        """
        Stable screen identity from the perceived visual hash, else the activity name.
        """

        if self.state is not None and self.state.visual_hash:
            return self.state.visual_hash[:16]

        return hashlib.md5(self.activity.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class HierarchySnapshot(BaseModel):
    """
    One perception snapshot with the provenance explaining any missing hierarchy.
    """

    model_config = ConfigDict(frozen=True)

    image: bytes = Field(description="Captured screenshot bytes", repr=False)
    hierarchy: Optional[str] = Field(
        default=None, description="View hierarchy when a dump was attempted and returned content"
    )
    provenance: Optional[HierarchyProvenance] = Field(
        default=None, description="Why the hierarchy is absent; None when a dump succeeded"
    )
