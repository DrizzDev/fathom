from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from fathom.constants import ActionType


class BoundingBox(BaseModel):
    """
    Normalized bounding box (0-1000 scale) for UI elements.
    """

    x: int = Field(ge=0, le=1000, description="Top-left X coordinate (0-1000)")
    y: int = Field(ge=0, le=1000, description="Top-left Y coordinate (0-1000)")
    width: int = Field(ge=0, le=1000, description="Width of the element (0-1000)")
    height: int = Field(ge=0, le=1000, description="Height of the element (0-1000)")
    coord_system: str = Field(default="normalized", description="Coordinate system used")

    @property
    def center_x(self) -> int:
        """
        Calculates the horizontal center of the bounding box.
        """
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """
        Calculates the vertical center of the bounding box.
        """
        return self.y + self.height // 2

    def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        """
        Converts normalized coordinates to absolute device pixels.
        """
        x_pixel = int(self.x * screen_width / 1000)
        y_pixel = int(self.y * screen_height / 1000)
        width_pixel = int(self.width * screen_width / 1000)
        height_pixel = int(self.height * screen_height / 1000)
        return x_pixel, y_pixel, width_pixel, height_pixel


class Action(BaseModel):
    """


    Represents an atomic action to be performed on the mobile device.


    """

    action_type: ActionType = Field(description="The type of interaction to perform")

    rationale: str = Field(description="The reasoning behind choosing this action")

    target: str = Field(default="element", description="Grounding label ID or technical target")

    natural_language_target: Optional[str] = Field(
        default=None, description="Human-friendly name of the target element (e.g., 'Search Bar')"
    )

    label_id: Optional[str] = Field(default=None, description="Numeric label ID from XML grounding")

    bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box for the interaction"
    )

    text: Optional[str] = Field(default=None, description="Text content for typing actions")

    wait_duration: Optional[int] = Field(
        default=None, description="Duration to wait in milliseconds"
    )

    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")

    model_config = {"frozen": True, "populate_by_name": True}

    def to_description(self) -> str:
        """


        Generates a human-readable description of the action.


        """

        # Resolve best target description.

        # Priority: natural_language_target -> target -> label_id -> bbox

        name = self.natural_language_target or self.target

        is_generic = not name or name.lower() in ("element", "ui element", "none", "label")

        if is_generic:
            if self.label_id:
                name = f"label {self.label_id}"

            elif self.bbox:
                name = f"bounds [{self.bbox.x}, {self.bbox.y}]"

            else:
                name = "element"

        if self.action_type == ActionType.TAP:
            return f"Tap on {name}"

        elif self.action_type == ActionType.TYPE:
            return f"Type '{self.text}' in {name}"

        elif self.action_type == ActionType.SWIPE:
            return f"Swipe on {name}"

        elif self.action_type == ActionType.SCROLL:
            return f"Scroll {name}"

        elif self.action_type == ActionType.LONG_PRESS:
            return f"Long press on {name}"

        elif self.action_type == ActionType.BACK:
            return "Press back button"

        elif self.action_type == ActionType.HOME:
            return "Press home button"

        elif self.action_type == ActionType.WAIT:
            return f"Wait for {name}"

        elif self.action_type == ActionType.COMPLETE:
            return "Goal completed"

        return f"{self.action_type.value} on {name}"
