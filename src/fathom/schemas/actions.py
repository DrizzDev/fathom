from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.constants import CONTROL_ACTION_TYPES, ActionExecutionKind, ActionType
from fathom.constants.exploration import COORD_BUCKET_GRID_SIZE, ExpectedOutcome


class CoordinateSource(StrEnum):
    """
    Evidence source that produced a :class:`Bounds`.

    Stages of the localization cascade map to distinct values so production
    log filters can attribute a tap-failure to a specific failure mode:

    - ``OCR``: phrase-matched OCR tokens (cascade stage 3).
    - ``XML``: snapped against the parsed XML manifest (cascade stages 1-2).
    - ``VISION``: vision-LLM ensemble member or DocumentAI layout resolved the bounds (cascade stage 5).
    - ``VIEWPORT``: derived from rendered viewport pixels (pixel overlay, icon template, region clipping).
    - ``MODEL``: planner LLM bounds dispatched without independent corroboration (cascade stage 6, last-resort blind trust).
    - ``MODEL_GROUNDED``: planner LLM bounds corroborated by OCR phrase evidence inside the proposed region (cascade stage 4).

    """

    XML = "xml"
    OCR = "ocr"
    MODEL = "model"
    VISION = "vision"
    VIEWPORT = "viewport"
    MODEL_GROUNDED = "model_grounded"

    @property
    def is_corroborated(self) -> bool:
        """
        Whether the bounds carry evidence beyond a blind planner guess.
        """

        return self not in _BLIND_COORDINATE_SOURCES


# Coordinate sources dispatched without independent corroboration: a planner LLM
# guess with no OCR/XML/vision/pixel evidence behind the proposed region.
_BLIND_COORDINATE_SOURCES: frozenset[CoordinateSource] = frozenset({CoordinateSource.MODEL})


class InputContextSource(StrEnum):
    """
    Evidence source that populated an :class:`InputContext`.
    """

    XML = "xml"

    MODEL = "model"
    VISION = "vision"
    ACCESSIBILITY = "accessibility"


class CoordinateSystem(StrEnum):
    """
    Explicit coordinate space in which a :class:`Bounds` carries its values.

    Three values cover every producer in the pipeline:

    - ``LOGICAL``: device-independent points (appium / WDA dispatch space).
      Matches what ``/window/rect`` reports (e.g., 430x932 on iPhone 15 Pro Max).
    - ``DEVICE_PIXEL``: physical screenshot pixels. iOS retina multiplies
      logical by the device scale factor (2x or 3x). OCR / icon / pixel-overlay
      adapters and the XML drawer's scaled label-map all live here.
    - ``NORMALIZED``: dimensionless 0-1000 grid. Gemini vision-localizer
      and direct LLM ``bbox`` tool calls return values in this space.
    """

    LOGICAL = "logical"
    NORMALIZED = "normalized"
    DEVICE_PIXEL = "device_pixel"

    @classmethod
    def from_legacy(cls, raw: Any) -> "CoordinateSystem":
        """
        Map any legacy/external coordinate-system value onto the enum.

        Accepts a :class:`CoordinateSystem`, a known modern string
        (``"logical"``, ``"device_pixel"``, ``"normalized"``), or the
        legacy ``"pixel"`` alias that pre-dated :class:`CoordinateSystem`.
        Any other input raises ``ValueError`` so silent drift surfaces.
        """

        if isinstance(raw, cls):
            return raw

        if isinstance(raw, str):
            if (alias := _LEGACY_COORDINATE_SYSTEM_ALIASES.get(raw.lower())) is not None:
                return alias
            return cls(raw)

        raise ValueError(f"unsupported coordinate-system value: {raw!r}")


_LEGACY_COORDINATE_SYSTEM_ALIASES: Dict[str, CoordinateSystem] = {
    "logical": CoordinateSystem.LOGICAL,
    "pixel": CoordinateSystem.DEVICE_PIXEL,
    "normalized": CoordinateSystem.NORMALIZED,
    "device_pixel": CoordinateSystem.DEVICE_PIXEL,
}


class InputContext(BaseModel):
    """
    Optional execution context for text input actions.

    Populated during resolution when element metadata is available (XML, accessibility, etc.).
    When absent, typing falls back to visual focus with no locator or clear behavior.
    """

    locator: Optional[str] = Field(
        default=None,
        description="Provider-neutral locator for the input element.",
    )
    prefilled: str = Field(
        default="",
        description="Observed text already present in the input field.",
    )
    source: Optional[InputContextSource] = Field(
        default=None,
        description="Evidence source used to derive the input context.",
    )


class Bounds(BaseModel):
    """
    Bounds for UI elements.
    """

    model_config = ConfigDict(populate_by_name=True)

    x: int = Field(ge=0, le=5000, description="Top-left X coordinate")
    y: int = Field(ge=0, le=5000, description="Top-left Y coordinate")
    width: int = Field(ge=0, le=5000, description="Width of the element")
    height: int = Field(ge=0, le=5000, description="Height of the element")

    system: CoordinateSystem = Field(
        alias="coordinate_system",
        default=CoordinateSystem.NORMALIZED,
        description="Coordinate space of x/y/width/height. See CoordinateSystem.",
    )
    source: Optional[CoordinateSource] = Field(
        default=None,
        description="Evidence source used to derive these coordinates.",
    )

    @field_validator("system", mode="before")
    @classmethod
    def __coerce_legacy_system(cls, raw: Any) -> Any:
        """
        Migrate legacy ``system`` string values to :class:`CoordinateSystem`.
        """

        if isinstance(raw, CoordinateSystem):
            return raw

        if isinstance(raw, str):
            return _LEGACY_COORDINATE_SYSTEM_ALIASES.get(raw.lower(), raw)

        return raw

    @property
    def is_normalized(self) -> bool:
        """
        Deprecated heuristic — kept for callers awaiting migration.
        """

        return self.system is CoordinateSystem.NORMALIZED

    def has_normalized_extent_violation(self) -> bool:
        """
        Return whether normalized coordinates exceed the 0-1000 extent contract.
        """

        if self.system is not CoordinateSystem.NORMALIZED:
            return False

        return any(
            (
                self.x > 1000,
                self.y > 1000,
                self.width > 1000,
                self.height > 1000,
                self.x + self.width > 1000,
                self.y + self.height > 1000,
            )
        )

    @property
    def center_x(self) -> int:
        """
        Calculates the horizontal center in the bounds' native coordinate space.
        """

        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """
        Calculates the vertical center in the bounds' native coordinate space.
        """

        return self.y + self.height // 2

    def coord_bucket(self, *, grid: int = COORD_BUCKET_GRID_SIZE) -> str:
        """
        Quantized centre on the coordinate grid for stable element dedup.
        """

        center_x = self.center_x if self.width > 0 else self.x
        center_y = self.center_y if self.height > 0 else self.y
        return f"{center_x // grid}_{center_y // grid}"

    def to_logical_dispatch(
        self,
        *,
        logical_width: int,
        logical_height: int,
        pixel_width: Optional[int] = None,
        pixel_height: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Translate the bounds to the logical-point space used for tap dispatch.

        Each :class:`CoordinateSystem` branch first converts to logical
        points, then clamps to the on-screen logical rectangle so the
        return value is always a valid dispatch coordinate.
        """

        if self.system is CoordinateSystem.LOGICAL:
            x, y, width, height = (
                float(self.x),
                float(self.y),
                float(self.width),
                float(self.height),
            )
        elif self.system is CoordinateSystem.DEVICE_PIXEL:
            effective_pixel_width = (
                pixel_width if pixel_width and pixel_width > 0 else logical_width
            )
            effective_pixel_height = (
                pixel_height if pixel_height and pixel_height > 0 else logical_height
            )
            scale_x = effective_pixel_width / max(1, logical_width)
            scale_y = effective_pixel_height / max(1, logical_height)

            x = self.x / scale_x
            y = self.y / scale_y
            width = self.width / scale_x
            height = self.height / scale_y

        elif self.system is CoordinateSystem.NORMALIZED:
            x = self.x * logical_width / 1000.0
            y = self.y * logical_height / 1000.0
            width = self.width * logical_width / 1000.0
            height = self.height * logical_height / 1000.0
        else:
            raise ValueError(f"unknown coordinate system: {self.system!r}")

        max_x = max(0, logical_width - 1)
        max_y = max(0, logical_height - 1)

        x = max(0, min(int(x), max_x))
        y = max(0, min(int(y), max_y))

        width = max(1, min(int(width), max(1, logical_width - x)))
        height = max(1, min(int(height), max(1, logical_height - y)))

        return x, y, width, height

    def to_pixels(self, screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
        """
        Legacy alias of :meth:`to_logical_dispatch` with a single screen pair.

        Retained so call sites that haven't been migrated to pass pixel
        dimensions still produce a valid dispatch coordinate when the
        device's pixel and logical spaces coincide (Android, 1x devices).
        New code must call :meth:`to_logical_dispatch` directly with both
        the logical and the pixel dimensions.
        """

        return self.to_logical_dispatch(
            logical_width=screen_width,
            logical_height=screen_height,
        )


class ExecutionRegion(BaseModel):
    """
    Region used to derive executable coordinates for an action.
    """

    x: int = Field(ge=0, le=5000, description="Left edge of the region in screen pixels.")
    y: int = Field(ge=0, le=5000, description="Top edge of the region in screen pixels.")

    width: int = Field(ge=1, le=5000, description="Region width in screen pixels.")
    height: int = Field(ge=1, le=5000, description="Region height in screen pixels.")
    source: CoordinateSource = Field(description="Coordinate evidence used to derive this region.")

    model_config = ConfigDict(frozen=True)


class GesturePath(BaseModel):
    """
    Concrete pointer path used to execute a gesture.
    """

    start_x: int = Field(ge=0, le=5000, description="Gesture start x-coordinate in screen pixels.")
    start_y: int = Field(ge=0, le=5000, description="Gesture start y-coordinate in screen pixels.")

    end_x: int = Field(ge=0, le=5000, description="Gesture end x-coordinate in screen pixels.")
    end_y: int = Field(ge=0, le=5000, description="Gesture end y-coordinate in screen pixels.")

    duration: int = Field(ge=0, description="Gesture duration in milliseconds.")

    model_config = ConfigDict(frozen=True)

    @property
    def distance(self) -> int:
        """
        Return the gesture travel distance in screen pixels.
        """

        horizontal = abs(self.end_x - self.start_x)
        vertical = abs(self.end_y - self.start_y)

        return max(horizontal, vertical)

    def to_coordinates(self) -> Tuple[int, int, int, int]:
        """
        Return the path as trace and device coordinates.
        """

        return self.start_x, self.start_y, self.end_x, self.end_y


class Action(BaseModel):
    """
    Represents an atomic action to be performed on the mobile device.
    """

    action_type: ActionType = Field(description="The type of interaction to perform")

    rationale: str = Field(description="The reasoning behind choosing this action")
    target: str = Field(default="element", description="Grounding label ID or technical target")
    natural_language_target: Optional[str] = Field(
        default=None, description="Human-friendly name of the target element."
    )

    text: Optional[str] = Field(default=None, description="Text content for typing actions")
    bounds: Optional[Bounds] = Field(default=None, description="Bounding box for the interaction")
    label_id: Optional[str] = Field(default=None, description="Numeric label ID from XML grounding")

    input_context: Optional[InputContext] = Field(
        default=None,
        description="Optional execution context for text input, populated during resolution when element metadata is available.",
    )

    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    wait_duration: Optional[float] = Field(default=None, description="Duration to wait in seconds")
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None, description="Key-value pairs to store in persistent memory"
    )

    # Inline Validation
    is_valid: bool = Field(default=True, description="Self-validation of the action")
    validation_reason: Optional[str] = Field(
        default=None, description="Reason if action is invalid"
    )

    # Conditional Execution
    condition: Optional[str] = Field(
        default=None,
        description="Condition required (e.g. 'Popup is visible', 'Section is collapsed', 'Error displayed')",
    )
    is_conditional: bool = Field(
        default=False,
        description="True when the action should execute only under a visible guard condition.",
    )
    conditional_type: Optional[Literal["blocker", "transient", "error", "optional"]] = Field(
        default=None,
        description="Optional conditional category: blocker, transient, error, or optional.",
    )
    overlay_detected: bool = Field(
        default=False,
        description="True when this action is specifically handling an overlay/popup blocker.",
    )

    # Exploration grounding (VLM-provided; optional; consumed by the exploration strategy)
    region: Optional[
        Literal["top_bar", "bottom_nav", "content", "modal", "overlay", "fab", "footer"]
    ] = Field(
        default=None,
        description="Screen region the element lives in, for stable grouping and dedup.",
    )
    element_category: Optional[
        Literal[
            "global_navigation",
            "primary_action",
            "content_item",
            "filter_or_category",
            "secondary_control",
            "overlay_dismiss",
        ]
    ] = Field(
        default=None,
        description="Priority-bucketed element category (P1-P5 / overlay_dismiss) for sampling.",
    )
    expected_outcome: Optional[ExpectedOutcome] = Field(
        default=None,
        description="Predicted screen effect of the action, used to verify it against the result.",
    )

    # Script export classification (VLM-provided; optional; fallback is TargetClassifier)
    target_type: Optional[Literal["stable", "positional", "dynamic"]] = Field(
        default=None,
        description="How the target should be referenced in exported scripts: stable (fixed label), positional (ordinal in list), or dynamic (content that may change). Leave unset if unsure.",
    )
    script_target: Optional[str] = Field(
        default=None,
        description="When target_type is positional or dynamic, the exact phrase for script export (e.g. 'the first search result', 'the promotional banner'). Omit for stable.",
    )
    surface: Optional[str] = Field(
        default=None,
        description="Specific section, container, or on-screen area the action belongs to.",
    )

    # Launch semantics (optional; used to disambiguate launcher icon taps from regular taps)
    is_app_launcher: bool = Field(
        default=False,
        description="Set to true when this tap action is specifically intended to launch or focus the target app. Helps the exporter replace launcher taps with OPEN_APP semantics.",
    )

    # Structured signal details for export (VLM-provided; authoritative)
    export_target: Optional[str] = Field(
        default=None,
        description=(
            "Canonical phrase for this action in exported test scripts. Must be specific "
            "and human-readable (e.g., 'Search box', 'the first search result', 'Add to cart button'). "
            "NEVER use generic placeholders like 'element', 'button', 'label'."
        ),
    )
    scroll_target: Optional[str] = Field(
        default=None,
        description="For scroll/swipe actions: the element or section being scrolled to find (e.g., 'Vitamins and supplements', 'Lab tests and packages'). Use the exact phrase from the UI when possible.",
    )
    wait_subject: Optional[str] = Field(
        default=None,
        description="For wait actions: what we're waiting for (e.g., 'app to load', 'search results to appear', 'Home page content'). Describe the expected state or element.",
    )
    validation_subject: Optional[str] = Field(
        default=None,
        description="For validate actions: what specifically is being validated (e.g., 'login status', 'banner visibility', 'item alignment'). Be specific about the validation target.",
    )
    target_is_generic: Optional[bool] = Field(
        default=None,
        description="Set to true when this action taps/selects a non-specific target (e.g., 'any item', 'random category', 'first result'). Signals that target should be generalized in export.",
    )
    target_element_type: Optional[
        Literal["button", "icon", "option", "link", "field", "text", "checkbox"]
    ] = Field(
        default=None,
        description="For tap/interact actions: the element type/role (button, icon, option, etc.). Helps refine target descriptions when product-specific elements are tapped.",
    )
    validation_pattern: Optional[Literal["blocker", "transient", "error", "generic"]] = Field(
        default=None,
        description="For validate actions: the pattern category - blocker (permission/popup/consent), transient (loading/spinner), error (network/validation error), or generic check.",
    )
    wait_pattern: Optional[Literal["ad", "splash", "load", "search", "generic"]] = Field(
        default=None,
        description="For wait actions: the wait category - ad (ad to finish), splash (app splash screen), load (content loading), search (search results), or generic.",
    )

    @property
    def execution_kind(self) -> ActionExecutionKind:
        """
        Return whether this action executes on the device or through control flow.
        """

        if self.action_type in CONTROL_ACTION_TYPES:
            return ActionExecutionKind.CONTROL
        return ActionExecutionKind.DEVICE

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def to_description(self) -> str:
        """
        Generates a human-readable description of the action.
        """

        # Resolve best target description.
        # For validate actions, prefer validation_subject over generic targets.
        if self.action_type == ActionType.VALIDATE and self.validation_subject:
            return f"Validate {self.validation_subject}"

        name = self.natural_language_target

        lowered = (name or "").strip().lower()
        if not name or lowered in ("element", "ui element", "none", "label", "unknown"):
            # Fallback to label ID or bounds if natural language target is generic/missing
            if self.label_id:
                name = f"Element (Label {self.label_id})"

            elif self.bounds:
                name = f"Element at [{self.bounds.x}, {self.bounds.y}]"

            else:
                name = self.target or "element"

        if self.action_type == ActionType.VALIDATE:
            return f"Validate {name}"

        if self.action_type == ActionType.TAP:
            return f"Tap on {name}"

        if self.action_type == ActionType.TYPE:
            text_val = self.text if self.text is not None else ""
            return f"Type '{text_val}' in {name}"

        if "swipe" in self.action_type.value:
            direction = (
                self.action_type.value.split("_")[-1]
                if "_" in self.action_type.value
                else "content"
            )
            return f"Swipe {direction} on {self.surface or name}"

        if self.action_type == ActionType.SCROLL:
            return f"Scroll {self.surface or name}"

        if self.action_type == ActionType.LONG_PRESS:
            return f"Long press on {name}"

        if self.action_type == ActionType.BACK:
            return "Press back button"

        if self.action_type == ActionType.HOME:
            return "Press home button"

        if self.action_type == ActionType.WAIT:
            if self.wait_duration:
                return f"Wait for {self.wait_duration} seconds"
            return f"Wait for {name}"

        if self.action_type == ActionType.COMPLETE:
            return f"Validate {name} (Goal complete)"

        if self.action_type == ActionType.ASK_USER:
            msg = self.text or self.rationale or name
            return f"Ask user: {msg}"

        return f"{self.action_type.value.capitalize()} on {name}"


BoundingBox = Bounds
