from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import ActionType


class Bounds(BaseModel):
    """
    Bounds for UI elements.
    Expected to be normalized (0-1000 scale) but supports raw pixels for robustness.
    """

    x: int = Field(ge=0, le=5000, description="Top-left X coordinate")
    y: int = Field(ge=0, le=5000, description="Top-left Y coordinate")
    width: int = Field(ge=0, le=5000, description="Width of the element")
    height: int = Field(ge=0, le=5000, description="Height of the element")
    system: str = Field(
        default="normalized", description="Coordinate system used", alias="coord_system"
    )

    @property
    def is_normalized(self) -> bool:
        """
        Heuristic to check if coordinates are likely normalized (0-1000).
        """

        return self.x <= 1000 and self.y <= 1000 and self.width <= 1000 and self.height <= 1000

    @property
    def center_x(self) -> int:
        """
        Calculates the horizontal center.
        """

        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """
        Calculates the vertical center.
        """

        return self.y + self.height // 2

    def coord_bucket(self, grid: int = 50) -> str:
        """
        Quantized center on the normalized 0-1000 grid for stable dedup.

        Two taps aimed at the same visual element from the same screen
        land in the same bucket even when the LLM's freeform ``target_name``
        label drifts between calls.
        """

        cx = self.center_x if self.width > 0 else self.x
        cy = self.center_y if self.height > 0 else self.y
        return f"{cx // grid}_{cy // grid}"

    def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        """
        Converts coordinates to absolute device pixels.
        Handles both normalized and already-pixel coordinates.
        Clamps results to valid device bounds.
        """

        # If explicitly told it's pixels, don't normalize
        if self.system == "pixel":
            x, y, w, h = self.x, self.y, self.width, self.height
        # Use heuristic if system is normalized (default)
        elif self.is_normalized:
            x = int(self.x * screen_width / 1000)
            y = int(self.y * screen_height / 1000)
            w = int(self.width * screen_width / 1000)
            h = int(self.height * screen_height / 1000)
        else:
            # Fallback for large values that must be pixels
            x, y, w, h = self.x, self.y, self.width, self.height

        # Clamp to valid device bounds to prevent out-of-screen taps/swipes
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        w = max(1, min(w, screen_width - x))
        h = max(1, min(h, screen_height - y))

        return x, y, w, h


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

    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    wait_duration: Optional[int] = Field(
        default=None, description="Duration to wait in milliseconds"
    )
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None, description="Key-value pairs to store in persistent memory"
    )

    # Inline Validation
    is_valid: bool = Field(default=True, description="Self-validation of the action")
    validation_reason: Optional[str] = Field(
        default=None, description="Reason if action is invalid"
    )

    overlay_detected: bool = Field(
        default=False,
        description="True when this action is specifically handling an overlay/popup blocker.",
    )

    region: Optional[
        Literal["top_bar", "bottom_nav", "content", "modal", "overlay", "fab", "footer"]
    ] = Field(
        default=None,
        description=(
            "Which region of the screen the element lives in. Enum-constrained so "
            "reports and prompts can group and dedup on a stable structural axis."
        ),
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
        description=(
            "Priority-bucketed element category (P1-P5 / overlay_dismiss). "
            "Enum-constrained so the sampling guard can count 'how many content_items "
            "have been tapped on this screen' without depending on freeform target_name."
        ),
    )

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def to_description(self) -> str:
        """
        Generates a human-readable description of the action.
        """

        # Resolve best target description.
        name = self.natural_language_target

        if not name or name.lower() in ("element", "ui element", "none", "label", "unknown"):
            if self.bounds:
                name = f"Element at [{self.bounds.x}, {self.bounds.y}]"
            else:
                name = self.target or "UI Element"

        if self.action_type == ActionType.TAP:
            return f"Tap on {name}"

        if self.action_type == ActionType.TYPE:
            return f"Type '{self.text}' in {name}"

        if "swipe" in self.action_type.value:
            direction = (
                self.action_type.value.split("_")[-1]
                if "_" in self.action_type.value
                else "content"
            )
            return f"Swipe {direction} on {name}"

        if self.action_type == ActionType.SCROLL:
            return f"Scroll until you see {name}"

        if self.action_type == ActionType.LONG_PRESS:
            return f"Long press on {name}"

        if self.action_type == ActionType.BACK:
            return "Press back button"

        if self.action_type == ActionType.HOME:
            return "Press home button"

        if self.action_type == ActionType.WAIT:
            return f"Wait for {name}"

        if self.action_type == ActionType.VALIDATE:
            return f"Validate {name}"

        if self.action_type == ActionType.COMPLETE:
            return f"Validate {name} (Goal complete)"

        return f"{self.action_type.value.capitalize()} on {name}"


BoundingBox = Bounds
