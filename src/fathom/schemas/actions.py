from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from fathom.constants import ActionType


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates (0-1000 scale).

    Using normalized coordinates allows device-independent targeting.
    Actual pixel coordinates are computed at execution time.
    """

    model_config = {"frozen": True}

    x: int = Field(ge=0, le=1000, description="Left edge (normalized)")
    y: int = Field(ge=0, le=1000, description="Top edge (normalized)")
    width: int = Field(ge=1, le=1000, description="Width (normalized)")
    height: int = Field(ge=1, le=1000, description="Height (normalized)")

    @property
    def center_x(self) -> int:
        """
        X coordinate of bounding box center.
        """
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """
        Y coordinate of bounding box center.
        """
        return self.y + self.height // 2

    def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates.

        Args:
            screen_width: Actual screen width in pixels.
            screen_height: Actual screen height in pixels.

        Returns:
            Tuple of (x, y, width, height) in pixels.
        """
        return (
            self.x * screen_width // 1000,
            self.y * screen_height // 1000,
            self.width * screen_width // 1000,
            self.height * screen_height // 1000,
        )


class Action(BaseModel):
    """
    An action to execute on a device.
    Actions are produced by the agent's planning phase and consumed by the tool layer for execution.
    """

    model_config = {"frozen": True}

    action_type: ActionType = Field(description="Type of action to execute")
    target: str = Field(
        min_length=1,
        max_length=500,
        description="Human-readable description of the target element",
    )
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Target bounding box (required for tap, swipe)",
    )
    label_id: Optional[str] = Field(
        default=None,
        description="Label ID from annotated screenshot (XML mode)",
    )
    text: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Text to type (required for type action)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Confidence score from planning",
    )
    reasoning: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Explanation of why this action was chosen",
    )

    def to_description(self) -> str:
        """
        Generate human-readable description of this action.
        """

        if self.action_type == ActionType.TAP:
            return f"Tap on {self.target}"

        elif self.action_type == ActionType.TYPE:
            return f"Type '{self.text}' in {self.target}"

        elif self.action_type == ActionType.SWIPE:
            return f"Swipe on {self.target}"

        elif self.action_type == ActionType.SCROLL:
            return f"Scroll {self.target}"

        elif self.action_type == ActionType.LONG_PRESS:
            return f"Long press on {self.target}"

        elif self.action_type == ActionType.BACK:
            return "Press back button"

        elif self.action_type == ActionType.HOME:
            return "Press home button"

        elif self.action_type == ActionType.WAIT:
            return f"Wait for {self.target}"

        elif self.action_type == ActionType.COMPLETE:
            return "Goal completed"

        else:
            return f"{self.action_type.value} on {self.target}"
