from __future__ import annotations

from typing import Dict, Optional

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

    def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        """
        Converts coordinates to absolute device pixels.
        Handles both normalized and already-pixel coordinates.
        """

        # If explicitly told it's pixels, don't normalize
        if self.system == "pixel":
            return self.x, self.y, self.width, self.height

        # Use heuristic if system is normalized (default)
        if self.is_normalized:
            x_pixel = int(self.x * screen_width / 1000)
            y_pixel = int(self.y * screen_height / 1000)
            width_pixel = int(self.width * screen_width / 1000)
            height_pixel = int(self.height * screen_height / 1000)

            return x_pixel, y_pixel, width_pixel, height_pixel

        # Fallback for large values that must be pixels
        return self.x, self.y, self.width, self.height


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

    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    wait_duration: Optional[int] = Field(
        default=None, description="Duration to wait in milliseconds"
    )
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None, description="Key-value pairs to store in persistent memory"
    )

    # Inline Validation
    is_valid: bool = Field(default=True, description="Self-validation of the action")
    validation_reason: Optional[str] = Field(default=None, description="Reason if action is invalid")

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def to_description(self) -> str:
        """
        Generates a human-readable description of the action.
        """

        # Resolve best target description.
        name = self.natural_language_target

        if not name or name.lower() in ("element", "ui element", "none", "label", "unknown"):
            # Fallback to label ID or bounds if natural language target is generic/missing
            if self.label_id:
                name = f"Element (Label {self.label_id})"

            elif self.bounds:
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

        if self.action_type == ActionType.COMPLETE:
            return f"Validate {name} (Goal complete)"

        return f"{self.action_type.value.capitalize()} on {name}"


BoundingBox = Bounds
